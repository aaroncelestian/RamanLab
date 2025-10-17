"""Cosmic ray detection and removal for Raman spectra."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from scipy.signal import find_peaks, peak_widths
import multiprocessing
from joblib import Parallel, delayed


@dataclass
class CosmicRayConfig:
    """Sophisticated configuration for intelligent cosmic ray detection."""
    enabled: bool = True
    # Optimized thresholds for both isolated and consecutive cosmic rays
    absolute_threshold: float = 1000  # Lower threshold to catch more cosmic rays
    neighbor_ratio: float = 5.0       # Lower ratio to handle consecutive cosmic rays better
    apply_during_load: bool = False
    
    # Parallel processing configuration
    use_parallel_processing: bool = True  # Enable multi-core processing for batch operations
    n_jobs: int = -1  # Number of CPU cores to use (-1 = all available)
    
    # Shape analysis parameters for distinguishing cosmic rays from Raman peaks
    # More lenient settings to better handle consecutive cosmic rays
    max_cosmic_fwhm: float = 20.0     # Maximum FWHM for cosmic rays (in data points) - flexible for wide events
    min_sharpness_ratio: float = 3.0  # Minimum ratio of peak height to width for cosmic rays - lowered for consecutive CRs
    max_asymmetry_factor: float = 0.5 # Maximum asymmetry for cosmic rays - increased for consecutive CRs
    gradient_threshold: float = 100.0 # Minimum gradient for cosmic ray edges (counts/point) - lowered for consecutive CRs
    enable_shape_analysis: bool = True # Enable advanced shape-based discrimination
    
    # NEW: Range-based removal parameters for better CR elimination
    removal_range: int = 3            # Number of data points to remove on each side of detected CR
    adaptive_range: bool = True       # Adapt removal range based on CR intensity
    max_removal_range: int = 8        # Maximum removal range for very intense CRs
    intensity_range_factor: float = 2.0  # Multiplier for adaptive range calculation
    
    # NEW: Intelligent noise addition for natural-looking cosmic ray repair
    enable_intelligent_noise: bool = True  # Add realistic variation to repaired regions
    noise_intensity: float = 0.8      # Intensity of added noise (0.0-1.0) - increased for better matching
    peak_restoration_chance: float = 0.15  # Chance of adding subtle peak features (0.0-1.0)
    noise_modeling_aggressiveness: float = 1.0  # How aggressively to match local noise (0.5-2.0)
    curvature_base_enhancement: bool = False  # Use curvature analysis for more realistic base interpolation
    simple_local_noise_matching: bool = True  # Use simple but effective local noise matching


class SimpleCosmicRayManager:
    """Ultra-fast, conservative cosmic ray removal - only removes obvious anomalies."""
    
    def __init__(self, config: CosmicRayConfig = None):
        self.config = config or CosmicRayConfig()
        self.statistics = {
            'total_spectra': 0,
            'spectra_with_cosmic_rays': 0,
            'total_cosmic_rays_removed': 0,
            'shape_analysis_rejections': 0,  # Peaks rejected due to shape analysis
            'false_positive_prevention': 0   # Strong Raman peaks saved from misclassification
        }
    
    def _modified_z_score(self, spectrum: np.ndarray) -> np.ndarray:
        """Calculates the modified z-scores of a given spectrum."""
        median_val = np.median(spectrum)
        mad_term = np.median(np.abs(spectrum - median_val))
        if mad_term == 0:
            return np.zeros_like(spectrum)
        modified_z_scores = 0.6745 * (spectrum - median_val) / mad_term
        return modified_z_scores

    def _whitaker_hayes_modified_z_score(self, spectrum: np.ndarray) -> np.ndarray:
        """Calculates the Whitaker-Hayes modified z-scores of a given spectrum."""
        if len(spectrum) < 2:
            return np.array([])
        return np.abs(self._modified_z_score(np.diff(spectrum)))

    def _whitaker_hayes_spectrum(self, intensity_values_array: np.ndarray, 
                                kernel_size: int, threshold: float) -> tuple[np.ndarray, list]:
        """
        Apply WhitakerHayes cosmic ray removal to a single spectrum.
        Returns (cleaned_spectrum, detected_indices)
        """
        spectrum_array = intensity_values_array.copy()
        detected_indices = []
        
        if len(spectrum_array) < 3:
            return spectrum_array, detected_indices
        
        spikes = self._whitaker_hayes_modified_z_score(spectrum_array) > threshold
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while np.any(spikes) and iteration < max_iterations:
            changes = False
            
            for i in range(len(spikes)):
                if spikes[i]:
                    # Note: spikes array is shorter by 1 due to np.diff
                    # Map back to original spectrum index
                    spike_idx = i + 1  # +1 because diff reduces array size
                    if spike_idx >= len(spectrum_array):
                        continue
                    
                    # Find neighbors for replacement
                    start_idx = max(0, spike_idx - kernel_size)
                    end_idx = min(len(spectrum_array), spike_idx + kernel_size + 1)
                    
                    # Get neighbor indices, excluding the spike itself
                    neighbor_indices = []
                    for j in range(start_idx, end_idx):
                        if j != spike_idx:
                            neighbor_indices.append(j)
                    
                    if neighbor_indices:
                        # Check which neighbors are not spikes
                        valid_neighbors = []
                        for j in neighbor_indices:
                            # Check if this neighbor was also detected as a spike
                            neighbor_is_spike = False
                            if j > 0 and j - 1 < len(spikes):
                                neighbor_is_spike = spikes[j - 1]
                            
                            if not neighbor_is_spike:
                                valid_neighbors.append(spectrum_array[j])
                        
                        if valid_neighbors:
                            fixed_value = np.mean(valid_neighbors) 
                            if not np.isnan(fixed_value):
                                spectrum_array[spike_idx] = fixed_value
                                detected_indices.append(spike_idx)
                                spikes[i] = False
                                changes = True
            
            if not changes:
                break
                
            # Recalculate spikes for next iteration
            spikes = self._whitaker_hayes_modified_z_score(spectrum_array) > threshold
            iteration += 1
        
        return spectrum_array, detected_indices
    
    def _analyze_peak_shape(self, intensities: np.ndarray, peak_idx: int, 
                           background_level: float) -> dict:
        """
        Analyze the shape characteristics of a peak to distinguish cosmic rays from Raman peaks.
        
        Cosmic rays typically have:
        - Very sharp, narrow peaks (high sharpness ratio)
        - Symmetric shape (low asymmetry)
        - Steep intensity gradients on both sides
        - Small FWHM
        
        Raman peaks typically have:
        - Broader, more gradual peaks
        - May be asymmetric
        - Gentler intensity gradients
        - Larger FWHM
        """
        # Get local region around the peak
        window_size = 10
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(intensities), peak_idx + window_size + 1)
        local_intensities = intensities[start_idx:end_idx]
        local_peak_idx = peak_idx - start_idx
        
        peak_height = intensities[peak_idx]
        
        # 1. Calculate FWHM (Full Width at Half Maximum)
        half_max = background_level + (peak_height - background_level) / 2
        
        # Find left and right half-maximum points
        left_half_idx = local_peak_idx
        right_half_idx = local_peak_idx
        
        # Search left
        for i in range(local_peak_idx - 1, -1, -1):
            if local_intensities[i] <= half_max:
                left_half_idx = i
                break
        
        # Search right
        for i in range(local_peak_idx + 1, len(local_intensities)):
            if local_intensities[i] <= half_max:
                right_half_idx = i
                break
        
        fwhm = right_half_idx - left_half_idx
        
        # 2. Calculate sharpness ratio (peak height / width)
        peak_width = max(fwhm, 1)  # Avoid division by zero
        sharpness_ratio = (peak_height - background_level) / peak_width
        
        # 3. Calculate asymmetry factor
        left_width = local_peak_idx - left_half_idx
        right_width = right_half_idx - local_peak_idx
        total_width = left_width + right_width
        if total_width > 0:
            asymmetry_factor = abs(left_width - right_width) / total_width
        else:
            asymmetry_factor = 0
        
        # 4. Calculate intensity gradients on both sides
        left_gradient = 0
        right_gradient = 0
        
        if local_peak_idx > 0:
            left_gradient = abs(local_intensities[local_peak_idx] - 
                              local_intensities[local_peak_idx - 1])
        
        if local_peak_idx < len(local_intensities) - 1:
            right_gradient = abs(local_intensities[local_peak_idx] - 
                               local_intensities[local_peak_idx + 1])
        
        avg_gradient = (left_gradient + right_gradient) / 2
        
        # 5. Calculate second derivative (curvature) at peak
        second_derivative = 0
        if 1 <= local_peak_idx <= len(local_intensities) - 2:
            second_derivative = (local_intensities[local_peak_idx - 1] - 
                               2 * local_intensities[local_peak_idx] + 
                               local_intensities[local_peak_idx + 1])
        
        return {
            'fwhm': fwhm,
            'sharpness_ratio': sharpness_ratio,
            'asymmetry_factor': asymmetry_factor,
            'avg_gradient': avg_gradient,
            'second_derivative': abs(second_derivative),
            'peak_height': peak_height,
            'background_level': background_level
        }
    
    def _is_cosmic_ray_by_shape(self, shape_analysis: dict) -> tuple[bool, str]:
        """
        Determine if a peak is a cosmic ray based on shape analysis.
        Returns (is_cosmic_ray, reason)
        """
        reasons = []
        cosmic_ray_indicators = 0
        raman_peak_indicators = 0
        
        # Check FWHM - cosmic rays are typically very narrow
        if shape_analysis['fwhm'] <= self.config.max_cosmic_fwhm:
            cosmic_ray_indicators += 1
            reasons.append(f"Very narrow FWHM ({shape_analysis['fwhm']:.1f} points)")
        else:
            raman_peak_indicators += 1
            reasons.append(f"Broad peak FWHM ({shape_analysis['fwhm']:.1f} points)")
        
        # Check sharpness ratio - cosmic rays have extreme sharpness
        if shape_analysis['sharpness_ratio'] >= self.config.min_sharpness_ratio:
            cosmic_ray_indicators += 1
            reasons.append(f"Very sharp peak (ratio: {shape_analysis['sharpness_ratio']:.1f})")
        else:
            raman_peak_indicators += 1
            reasons.append(f"Gradual peak (ratio: {shape_analysis['sharpness_ratio']:.1f})")
        
        # Check asymmetry - cosmic rays should be symmetric
        if shape_analysis['asymmetry_factor'] <= self.config.max_asymmetry_factor:
            cosmic_ray_indicators += 1
            reasons.append(f"Symmetric peak (asymmetry: {shape_analysis['asymmetry_factor']:.2f})")
        else:
            raman_peak_indicators += 1
            reasons.append(f"Asymmetric peak (asymmetry: {shape_analysis['asymmetry_factor']:.2f})")
        
        # Check gradient - cosmic rays have very steep edges
        if shape_analysis['avg_gradient'] >= self.config.gradient_threshold:
            cosmic_ray_indicators += 1
            reasons.append(f"Steep gradients ({shape_analysis['avg_gradient']:.0f} counts/point)")
        else:
            raman_peak_indicators += 1
            reasons.append(f"Gentle gradients ({shape_analysis['avg_gradient']:.0f} counts/point)")
        
        # Decision logic: require multiple cosmic ray indicators
        is_cosmic_ray = cosmic_ray_indicators >= 3  # At least 3 out of 4 criteria
        
        reason_str = "; ".join(reasons)
        if is_cosmic_ray:
            return True, f"COSMIC RAY: {reason_str}"
        else:
            return False, f"RAMAN PEAK: {reason_str}"
    
    def detect_and_remove_cosmic_rays(self, wavenumbers: np.ndarray, intensities: np.ndarray, 
                                    spectrum_id: str = None) -> Tuple[bool, np.ndarray, dict]:
        """
        Hybrid cosmic ray detection combining WhitakerHayes and advanced shape analysis.
        
        Two-stage approach:
        1. WhitakerHayes method for fast, robust detection of obvious cosmic rays
        2. Our advanced method for edge cases and validation
        """
        if not self.config.enabled or len(intensities) < 5:
            return False, intensities.copy(), {}
        
        cleaned_intensities = intensities.copy()
        cosmic_ray_indices = []
        
        # Get spectrum statistics for adaptive thresholding
        spectrum_max = np.max(intensities)
        spectrum_median = np.median(intensities)
        spectrum_99th = np.percentile(intensities, 99)
        spectrum_std = np.std(intensities)
        spectrum_75th = np.percentile(intensities, 75)
        
        # Dynamic minimum intensity threshold - cosmic rays should be significant above baseline
        # Don't detect cosmic rays in very low intensity regions (likely just noise)
        min_cosmic_intensity = max(
            spectrum_median + 3 * spectrum_std,  # 3-sigma above median
            spectrum_75th * 0.5,                # Half of 75th percentile  
            100  # Absolute minimum threshold
        )
        
        # ADVANCED COSMIC RAY DETECTION: Using our superior method only
        potential_cosmic_rays = []
        
        # Pass 1: Traditional isolated cosmic ray detection
        current_working_intensities = cleaned_intensities
        
        for i in range(1, len(current_working_intensities) - 1):
            current = current_working_intensities[i]
            
            # Skip if below minimum cosmic ray intensity
            if current < min_cosmic_intensity:
                continue
            
            # Get immediate neighbors
            left1 = current_working_intensities[i - 1] 
            right1 = current_working_intensities[i + 1]
            
            # Criterion 1: Must be above absolute threshold
            meets_absolute = current > self.config.absolute_threshold
            if not meets_absolute:
                continue
            
            # Criterion 2: Traditional isolated spike test
            max_immediate_neighbor = max(left1, right1)
            effective_neighbor_ratio = min(self.config.neighbor_ratio, 20.0)
            
            is_isolated_spike = (max_immediate_neighbor > 0 and 
                               current > max_immediate_neighbor * effective_neighbor_ratio)
            
            if is_isolated_spike and i not in potential_cosmic_rays:
                potential_cosmic_rays.append(i)
        
        # Pass 2: Detect cosmic ray clusters (consecutive high-intensity regions)
        # This handles detector overflow and closely spaced cosmic rays
        for i in range(1, len(current_working_intensities) - 1):
            current = current_working_intensities[i]
            
            # Skip if already detected in pass 1
            if i in potential_cosmic_rays:
                continue
            
            # Must be above absolute threshold
            if current < self.config.absolute_threshold:
                continue
            
            # Look for cosmic ray cluster pattern
            # Check if this point is part of a high-intensity region
            cluster_start = i
            cluster_end = i
            
            # Expand cluster to include adjacent high points
            while (cluster_start > 0 and 
                   current_working_intensities[cluster_start - 1] >= self.config.absolute_threshold):
                cluster_start -= 1
            
            while (cluster_end < len(current_working_intensities) - 1 and 
                   current_working_intensities[cluster_end + 1] >= self.config.absolute_threshold):
                cluster_end += 1
            
            cluster_size = cluster_end - cluster_start + 1
            
            # If we have a cluster of high points, check if it's cosmic ray overflow
            if cluster_size >= 2:  # At least 2 consecutive high points
                # Check background levels around the cluster
                background_left = max(0, cluster_start - 3)
                background_right = min(len(current_working_intensities), cluster_end + 4)
                
                background_points = []
                for j in range(background_left, background_right):
                    if j < cluster_start or j > cluster_end:  # Outside the cluster
                        background_points.append(current_working_intensities[j])
                
                if background_points:
                    background_level = np.median(background_points)
                    background_std = np.std(background_points)
                    
                    # Cluster should be significantly above background
                    cluster_min = np.min(current_working_intensities[cluster_start:cluster_end + 1])
                    is_above_background = cluster_min > background_level + 5 * background_std
                    
                    # Additional check: cluster should be much higher than background
                    intensity_ratio = cluster_min / (background_level + 1e-10)
                    is_significant_cluster = intensity_ratio > 3.0  # At least 3× background
                    
                    if is_above_background and is_significant_cluster:
                        # Add all points in the cluster as cosmic rays
                        for j in range(cluster_start, cluster_end + 1):
                            if j not in potential_cosmic_rays:
                                potential_cosmic_rays.append(j)
        
        # Apply shape analysis to all potential cosmic rays if enabled
        confirmed_cosmic_rays = []
        
        for i in potential_cosmic_rays:
            current = current_working_intensities[i]
            
            # Apply shape analysis if enabled
            if self.config.enable_shape_analysis:
                # Get local background for shape analysis (use original intensities for shape analysis)
                local_start = max(0, i - 5)
                local_end = min(len(intensities), i + 6)
                local_background_points = []
                
                for j in range(local_start, local_end):
                    if abs(j - i) > 2:  # Skip the peak and immediate neighbors
                        local_background_points.append(intensities[j])  # Use original for shape analysis
                
                local_background = np.median(local_background_points) if local_background_points else spectrum_median
                
                # Perform detailed shape analysis on original data
                shape_analysis = self._analyze_peak_shape(intensities, i, local_background)
                is_cosmic_ray, shape_reason = self._is_cosmic_ray_by_shape(shape_analysis)
                
                if not is_cosmic_ray:
                    # Shape analysis suggests this is a Raman peak, not a cosmic ray
                    self.statistics['false_positive_prevention'] += 1
                    continue
            
            # If we reach here, this point is confirmed as a cosmic ray
            confirmed_cosmic_rays.append(i)
        
        cosmic_ray_indices = confirmed_cosmic_rays
        
        # Use improved range-based cosmic ray removal
        cleaned_intensities = self._remove_cosmic_rays_with_range(
            intensities, cosmic_ray_indices, spectrum_median, min_cosmic_intensity
        )
        
        # Update statistics
        has_cosmic_ray = len(cosmic_ray_indices) > 0
        if spectrum_id:
            self.statistics['total_spectra'] += 1
            if has_cosmic_ray:
                self.statistics['spectra_with_cosmic_rays'] += 1
                self.statistics['total_cosmic_rays_removed'] += len(cosmic_ray_indices)
        
        detection_info = {
            'cosmic_ray_indices': cosmic_ray_indices,
            'cosmic_ray_count': len(cosmic_ray_indices),
            'shape_analysis_enabled': self.config.enable_shape_analysis,
            'false_positives_prevented': self.statistics.get('false_positive_prevention', 0),
            'spectrum_stats': {
                'max': spectrum_max,
                'median': spectrum_median,
                '99th_percentile': spectrum_99th,
                'std': spectrum_std,
                'min_cosmic_intensity': min_cosmic_intensity
            }
        }
        
        return has_cosmic_ray, cleaned_intensities, detection_info
    
    def get_statistics(self) -> dict:
        """Get cosmic ray detection statistics."""
        stats = self.statistics.copy()
        # Ensure all required keys exist
        stats.setdefault('false_positive_prevention', 0)
        
        if stats['total_spectra'] > 0:
            stats['percentage_with_cosmic_rays'] = (stats['spectra_with_cosmic_rays'] / 
                                                   stats['total_spectra'] * 100)
            stats['average_cosmic_rays_per_spectrum'] = (stats['total_cosmic_rays_removed'] / 
                                                       stats['total_spectra'])
            stats['false_positive_prevention_rate'] = (stats.get('false_positive_prevention', 0) / 
                                                      max(stats['total_spectra'], 1) * 100)
        else:
            stats['percentage_with_cosmic_rays'] = 0
            stats['average_cosmic_rays_per_spectrum'] = 0
            stats['false_positive_prevention_rate'] = 0
        
        return stats
    
    def diagnose_peak_shape(self, wavenumbers: np.ndarray, intensities: np.ndarray, 
                           peak_indices: list = None, display_details: bool = True) -> dict:
        """
        Diagnose peak shapes in a spectrum to understand cosmic ray vs Raman peak classification.
        Useful for fine-tuning shape analysis parameters.
        """
        if not self.config.enable_shape_analysis:
            return {"error": "Shape analysis is disabled"}
        
        # If no peak indices provided, find high-intensity peaks
        if peak_indices is None:
            spectrum_median = np.median(intensities)
            spectrum_std = np.std(intensities)
            threshold = spectrum_median + 5 * spectrum_std
            peak_indices = [i for i, val in enumerate(intensities) if val > threshold]
        
        if not peak_indices:
            return {"message": "No significant peaks found for analysis"}
        
        diagnosis = {
            'total_peaks_analyzed': len(peak_indices),
            'cosmic_rays_identified': 0,
            'raman_peaks_identified': 0,
            'peak_details': []
        }
        
        spectrum_median = np.median(intensities)
        
        for idx in peak_indices:
            if idx < 2 or idx >= len(intensities) - 2:
                continue  # Skip edge peaks
                
            # Perform shape analysis
            shape_analysis = self._analyze_peak_shape(intensities, idx, spectrum_median)
            is_cosmic_ray, reason = self._is_cosmic_ray_by_shape(shape_analysis)
            
            peak_detail = {
                'index': idx,
                'wavenumber': wavenumbers[idx] if idx < len(wavenumbers) else None,
                'intensity': intensities[idx],
                'classification': 'cosmic_ray' if is_cosmic_ray else 'raman_peak',
                'reason': reason,
                'shape_metrics': shape_analysis
            }
            
            diagnosis['peak_details'].append(peak_detail)
            
            if is_cosmic_ray:
                diagnosis['cosmic_rays_identified'] += 1
            else:
                diagnosis['raman_peaks_identified'] += 1
        
        return diagnosis
    
    def _remove_cosmic_rays_with_range(self, intensities: np.ndarray, cosmic_ray_indices: list, 
                                      spectrum_median: float, min_cosmic_intensity: float) -> np.ndarray:
        """
        IMPROVED: Remove cosmic rays using range-based elimination with intelligent noise addition
        to create more natural-looking repairs that can restore overlapped peaks.
        """
        cleaned_intensities = intensities.copy()
        
        if not cosmic_ray_indices:
            return cleaned_intensities
        
        # First, expand each cosmic ray detection to include surrounding points
        # This eliminates remnant shoulder peaks
        expanded_cosmic_rays = set()
        
        for cr_idx in cosmic_ray_indices:
            cr_intensity = intensities[cr_idx]
            
            # Calculate adaptive removal range based on CR intensity
            if hasattr(self.config, 'adaptive_range') and getattr(self.config, 'adaptive_range', True):
                # Adaptive range: more intense CRs get wider removal
                intensity_factor = min(cr_intensity / spectrum_median, 10.0)  # Cap at 10x
                adaptive_range = int(getattr(self.config, 'removal_range', 3) * 
                                   getattr(self.config, 'intensity_range_factor', 2.0) * 
                                   np.log(1 + intensity_factor))
                removal_range = min(adaptive_range, getattr(self.config, 'max_removal_range', 8))
            else:
                removal_range = getattr(self.config, 'removal_range', 3)
            
            # Add the CR and surrounding points to removal list
            for offset in range(-removal_range, removal_range + 1):
                target_idx = cr_idx + offset
                if 0 <= target_idx < len(intensities):
                    expanded_cosmic_rays.add(target_idx)
        
        # Group expanded cosmic rays into contiguous regions
        cosmic_ray_regions = []
        current_region = []
        
        for idx in sorted(expanded_cosmic_rays):
            if not current_region or idx == current_region[-1] + 1:
                current_region.append(idx)
            else:
                cosmic_ray_regions.append(current_region)
                current_region = [idx]
        if current_region:
            cosmic_ray_regions.append(current_region)
        
        # Remove each cosmic ray region with intelligent interpolation and noise addition
        for region in cosmic_ray_regions:
            start_idx = region[0]
            end_idx = region[-1]
            
            # Find clean neighbors with wider search for better interpolation
            left_search_range = 8  # Extended search for better context
            right_search_range = 8
            
            left_idx = start_idx - 1
            right_idx = end_idx + 1
            
            # Search for clean baseline points using a more robust approach
            # Use local baseline estimation instead of global median multiplier
            local_median = np.median(intensities[max(0, start_idx-15):min(len(intensities), end_idx+15)])
            local_std = np.std(intensities[max(0, start_idx-15):min(len(intensities), end_idx+15)])
            baseline_threshold = local_median + 2 * local_std
            
            # Find the best clean reference points
            for offset in range(1, left_search_range + 1):
                test_idx = start_idx - offset
                if test_idx >= 0:
                    if intensities[test_idx] < baseline_threshold:
                        left_idx = test_idx
                        break
            
            for offset in range(1, right_search_range + 1):
                test_idx = end_idx + offset
                if test_idx < len(intensities):
                    if intensities[test_idx] < baseline_threshold:
                        right_idx = test_idx
                        break
            
            # Perform intelligent interpolation with realistic noise addition
            if left_idx >= 0 and right_idx < len(intensities):
                # Get extended context for better trend analysis
                context_left = max(0, left_idx - 10)
                context_right = min(len(intensities), right_idx + 10)
                
                # Analyze local trends and noise characteristics
                left_context = intensities[context_left:left_idx+1]
                right_context = intensities[right_idx:context_right+1]
                
                # Calculate local noise characteristics from clean regions
                local_noise_std = np.std(np.concatenate([left_context, right_context]))
                local_mean_intensity = np.mean(np.concatenate([left_context, right_context]))
                
                # Detect if there might be an underlying peak structure
                # Look for gradual intensity changes that suggest a peak was overlapped
                left_trend = self._calculate_local_trend(left_context)
                right_trend = self._calculate_local_trend(right_context[::-1])  # Reverse for consistent direction
                
                # Base interpolation using cubic spline
                try:
                    from scipy.interpolate import CubicSpline
                    # Create interpolation points with extended context
                    x_points = [left_idx, right_idx]
                    y_points = [intensities[left_idx], intensities[right_idx]]
                    
                    # Add additional context points for better spline fitting
                    if left_idx > 0:
                        x_points.insert(0, left_idx - 1)
                        y_points.insert(0, intensities[left_idx - 1])
                    if right_idx < len(intensities) - 1:
                        x_points.append(right_idx + 1)
                        y_points.append(intensities[right_idx + 1])
                    
                    if len(x_points) >= 2:
                        cs = CubicSpline(x_points, y_points)
                        
                        # Apply enhanced interpolation with intelligent noise
                        for i, idx in enumerate(region):
                            # Base interpolated value
                            base_val = cs(idx)
                            
                            # Add intelligent variation based on local characteristics
                            enhanced_val = self._add_intelligent_variation(
                                base_val, idx, region, local_noise_std, 
                                left_trend, right_trend, local_mean_intensity,
                                intensities, left_context, right_context
                            )
                            
                            # Ensure value stays within reasonable bounds
                            final_val = max(enhanced_val, local_median * 0.3)
                            final_val = min(final_val, local_mean_intensity * 3)  # Prevent unrealistic values
                            
                            cleaned_intensities[idx] = final_val
                    else:
                        # Fallback to enhanced linear interpolation
                        self._apply_enhanced_linear_interpolation(
                            cleaned_intensities, region, left_idx, right_idx, 
                            intensities, local_noise_std, left_trend, right_trend, 
                            local_mean_intensity, local_median, left_context, right_context
                        )
                        
                except (ImportError, ValueError):
                    # Fallback to enhanced linear interpolation
                    self._apply_enhanced_linear_interpolation(
                        cleaned_intensities, region, left_idx, right_idx, 
                        intensities, local_noise_std, left_trend, right_trend, 
                        local_mean_intensity, local_median, left_context, right_context
                    )
                    
            elif left_idx >= 0:
                # Only left neighbor available - use trend extrapolation with noise
                left_context = intensities[max(0, left_idx-5):left_idx+1]
                trend = self._calculate_local_trend(left_context)
                noise_std = np.std(left_context)
                
                for i, idx in enumerate(region):
                    distance = idx - left_idx
                    base_val = intensities[left_idx] + trend * distance
                    
                    # Add noise if enabled
                    if getattr(self.config, 'enable_intelligent_noise', True):
                        noise_intensity = getattr(self.config, 'noise_intensity', 0.4)
                        noise = np.random.normal(0, noise_std * noise_intensity * 0.5)  # Gentle noise
                        final_val = max(base_val + noise, local_median * 0.5)
                    else:
                        final_val = max(base_val, local_median * 0.5)
                    
                    cleaned_intensities[idx] = final_val
                    
            elif right_idx < len(intensities):
                # Only right neighbor available - use trend extrapolation with noise
                right_context = intensities[right_idx:min(len(intensities), right_idx+6)]
                trend = self._calculate_local_trend(right_context[::-1])  # Reverse for consistent direction
                noise_std = np.std(right_context)
                
                for i, idx in enumerate(region):
                    distance = right_idx - idx
                    base_val = intensities[right_idx] + trend * distance
                    
                    # Add noise if enabled
                    if getattr(self.config, 'enable_intelligent_noise', True):
                        noise_intensity = getattr(self.config, 'noise_intensity', 0.4)
                        noise = np.random.normal(0, noise_std * noise_intensity * 0.5)  # Gentle noise
                        final_val = max(base_val + noise, local_median * 0.5)
                    else:
                        final_val = max(base_val, local_median * 0.5)
                    
                    cleaned_intensities[idx] = final_val
            else:
                # Fallback: use local median with gentle noise
                for idx in region:
                    if getattr(self.config, 'enable_intelligent_noise', True):
                        noise_intensity = getattr(self.config, 'noise_intensity', 0.4)
                        noise_std = local_std * noise_intensity * 0.3  # Very gentle noise
                        noise = np.random.normal(0, noise_std)
                        final_val = max(local_median + noise, 0)
                    else:
                        final_val = max(local_median, 0)
                    
                    cleaned_intensities[idx] = final_val
        
        return cleaned_intensities
    
    def _calculate_local_trend(self, data_segment: np.ndarray) -> float:
        """Calculate the local trend (slope) in a data segment."""
        if len(data_segment) < 2:
            return 0.0
        
        # Use linear regression to find trend
        x = np.arange(len(data_segment))
        try:
            # Simple linear regression
            n = len(data_segment)
            sum_x = np.sum(x)
            sum_y = np.sum(data_segment)
            sum_xy = np.sum(x * data_segment)
            sum_x2 = np.sum(x * x)
            
            # Calculate slope (trend)
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) > 1e-10:
                slope = (n * sum_xy - sum_x * sum_y) / denominator
                return slope
            else:
                return 0.0
        except:
            return 0.0
    
    def _analyze_local_noise_characteristics(self, intensities: np.ndarray, left_context: np.ndarray, 
                                           right_context: np.ndarray) -> dict:
        """Analyze the noise characteristics of the clean regions surrounding a cosmic ray."""
        # Combine left and right contexts for analysis
        clean_data = np.concatenate([left_context, right_context])
        
        if len(clean_data) < 5:
            # Fallback for very short contexts
            return {
                'noise_std': np.std(clean_data) if len(clean_data) > 0 else 1.0,
                'local_variations': clean_data if len(clean_data) > 0 else np.array([0]),
                'autocorr_lag1': 0.0,
                'smoothness_factor': 1.0
            }
        
        # Calculate detrended variations to understand pure noise characteristics
        # Remove slow trend to isolate noise
        from scipy.signal import detrend
        try:
            detrended = detrend(clean_data, type='linear')
        except:
            detrended = clean_data - np.mean(clean_data)
        
        # Calculate autocorrelation at lag 1 to understand noise correlation
        autocorr_lag1 = 0.0
        if len(detrended) > 1:
            try:
                autocorr_lag1 = np.corrcoef(detrended[:-1], detrended[1:])[0, 1]
                if np.isnan(autocorr_lag1):
                    autocorr_lag1 = 0.0
            except:
                autocorr_lag1 = 0.0
        
        # Calculate local smoothness (how much adjacent points differ)
        if len(clean_data) > 1:
            point_to_point_diff = np.abs(np.diff(clean_data))
            smoothness_factor = np.mean(point_to_point_diff)
        else:
            smoothness_factor = 1.0
        
        # Get actual variations from the clean data for realistic modeling
        local_variations = detrended
        
        return {
            'noise_std': np.std(detrended),
            'local_variations': local_variations,
            'autocorr_lag1': autocorr_lag1,
            'smoothness_factor': smoothness_factor,
            'clean_data': clean_data
        }
    
    def _generate_realistic_noise_sequence(self, length: int, noise_characteristics: dict, 
                                         base_trend: np.ndarray) -> np.ndarray:
        """Generate a realistic noise sequence that matches local characteristics."""
        
        if length <= 0:
            return np.array([])
        
        noise_std = noise_characteristics['noise_std']
        local_variations = noise_characteristics['local_variations']
        autocorr_lag1 = noise_characteristics['autocorr_lag1']
        smoothness_factor = noise_characteristics['smoothness_factor']
        
        # Method 1: If we have enough local variation data, sample from it with interpolation
        if len(local_variations) >= length:
            # Randomly sample segments from the local variations
            start_idx = np.random.randint(0, len(local_variations) - length + 1)
            noise_sequence = local_variations[start_idx:start_idx + length].copy()
        
        # Method 2: Generate correlated noise based on autocorrelation characteristics
        elif autocorr_lag1 > 0.1:  # Significant correlation
            # Generate AR(1) process: x[n] = a*x[n-1] + noise
            noise_sequence = np.zeros(length)
            noise_sequence[0] = np.random.normal(0, noise_std)
            
            for i in range(1, length):
                noise_sequence[i] = (autocorr_lag1 * noise_sequence[i-1] + 
                                   np.random.normal(0, noise_std * np.sqrt(1 - autocorr_lag1**2)))
        
                 # Method 3: Use resampling from local variations with noise augmentation
        else:
            if len(local_variations) >= length * 2:
                # We have enough data - use sliding window sampling for better correlation
                noise_sequence = np.zeros(length)
                # Randomly choose a starting point that gives us enough data
                max_start = len(local_variations) - length
                start_idx = np.random.randint(0, max_start + 1)
                
                # Use a sliding window from the local variations
                base_segment = local_variations[start_idx:start_idx + length].copy()
                
                # Add small variation to avoid exact copying
                variation_scale = noise_std * 0.2
                random_variation = np.random.normal(0, variation_scale, length)
                noise_sequence = base_segment + random_variation
                
            elif len(local_variations) > 0:
                # Bootstrap resampling with smooth transitions
                noise_sequence = np.zeros(length)
                for i in range(length):
                    # Sample from local variations with some randomization
                    sample_idx = np.random.randint(0, len(local_variations))
                    base_noise = local_variations[sample_idx]
                    # Add small additional variation to avoid exact repetition
                    additional_noise = np.random.normal(0, noise_std * 0.3)
                    noise_sequence[i] = base_noise + additional_noise
            else:
                # Fallback to simple Gaussian noise
                noise_sequence = np.random.normal(0, noise_std, length)
        
        # Apply smoothing based on local smoothness characteristics
        # If the local data is very smooth, apply smoothing to our generated noise
        if smoothness_factor < noise_std * 0.5:  # Local data is smoother than expected
            # Apply light smoothing
            from scipy.ndimage import gaussian_filter1d
            try:
                noise_sequence = gaussian_filter1d(noise_sequence, sigma=0.8)
            except:
                pass  # If scipy not available, skip smoothing
        
        return noise_sequence
    
    def _add_intelligent_variation(self, base_val: float, idx: int, region: list, 
                                  noise_std: float, left_trend: float, right_trend: float, 
                                  local_mean: float, intensities: np.ndarray = None,
                                  left_context: np.ndarray = None, right_context: np.ndarray = None) -> float:
        """Add intelligent variation to interpolated values to create more natural-looking data."""
        
        # Check if intelligent noise is enabled
        if not getattr(self.config, 'enable_intelligent_noise', True):
            return base_val
        
        # If we have context data, use sophisticated noise modeling
        if left_context is not None and right_context is not None and len(left_context) > 0 and len(right_context) > 0:
            return self._add_sophisticated_variation(
                base_val, idx, region, left_context, right_context, intensities
            )
        
        # Fallback to simpler method if no context available
        # Position within the region (0.0 to 1.0)
        region_start = region[0]
        region_end = region[-1]
        region_length = region_end - region_start
        if region_length > 0:
            position = (idx - region_start) / region_length
        else:
            position = 0.5
        
        # Blend trends from left and right sides
        blended_trend = left_trend * (1 - position) + right_trend * position
        trend_influence = blended_trend * (position - 0.5) * region_length * 0.3
        
        # Add realistic noise
        noise_intensity = getattr(self.config, 'noise_intensity', 0.4)
        intensity_factor = min(base_val / (local_mean + 1e-10), 3.0)
        scaled_noise_std = noise_std * noise_intensity * (0.5 + 0.5 * intensity_factor)
        noise_base = np.random.normal(0, scaled_noise_std)
        
        # Combine components
        enhanced_val = base_val + trend_influence + noise_base
        return enhanced_val
     
    def _create_realistic_base_signal(self, region: list, left_context: np.ndarray, 
                                    right_context: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Create a realistic base signal that incorporates local structural patterns."""
        
        region_length = len(region)
        if region_length == 0:
            return np.array([])
        
        # Get boundary values
        left_val = left_context[-1] if len(left_context) > 0 else intensities[region[0]-1] if region[0] > 0 else 0
        right_val = right_context[0] if len(right_context) > 0 else intensities[region[-1]+1] if region[-1] < len(intensities)-1 else 0
        
        # Analyze local patterns to detect potential underlying structure
        combined_context = np.concatenate([left_context, right_context])
        
        if len(combined_context) < 4:
            # Fallback to linear interpolation for short contexts
            return np.linspace(left_val, right_val, region_length)
        
        # Detect local curvature patterns if enhancement is enabled
        if (getattr(self.config, 'curvature_base_enhancement', True) and 
            len(left_context) >= 3 and len(right_context) >= 3):
            # Calculate second derivatives to understand curvature
            left_curvature = np.mean(np.diff(left_context, n=2))
            right_curvature = np.mean(np.diff(right_context, n=2))
            
            # Create a base signal that incorporates local curvature
            positions = np.linspace(0, 1, region_length)
            
            # Start with linear interpolation
            base_signal = left_val + positions * (right_val - left_val)
            
            # Add curvature influence - stronger in center, weaker at edges
            for i, pos in enumerate(positions):
                # Blend curvatures based on position
                blended_curvature = left_curvature * (1 - pos) + right_curvature * pos
                
                # Apply curvature with maximum effect in the center
                curvature_weight = 4 * pos * (1 - pos)  # Parabolic weight (max at center)
                curvature_adjustment = blended_curvature * curvature_weight * region_length
                
                base_signal[i] += curvature_adjustment
        else:
            # Simple linear interpolation
            base_signal = np.linspace(left_val, right_val, region_length)
        
        return base_signal
    
    def _add_sophisticated_variation(self, base_val: float, idx: int, region: list,
                                   left_context: np.ndarray, right_context: np.ndarray,
                                   intensities: np.ndarray) -> float:
        """Add sophisticated variation based on detailed analysis of surrounding clean regions."""
        
        # Check if simple local noise matching is enabled (simpler but often more effective)
        if getattr(self.config, 'simple_local_noise_matching', True):
            return self._add_simple_local_noise_variation(
                base_val, idx, region, left_context, right_context, intensities
            )
        
        # Complex approach (keep as fallback)
        # Analyze noise characteristics of clean regions
        noise_characteristics = self._analyze_local_noise_characteristics(
            intensities, left_context, right_context
        )
        
        # Position within the region
        region_start = region[0]
        region_end = region[-1]
        region_length = region_end - region_start
        if region_length > 0:
            position = (idx - region_start) / region_length
        else:
            position = 0.5
        
        # Create a more realistic base signal for the entire region first
        if not hasattr(self, '_current_region_base_signal') or self._current_region != region:
            self._current_region = region
            self._current_region_base_signal = self._create_realistic_base_signal(
                region, left_context, right_context, intensities
            )
        
        # Get the base value from our realistic signal
        current_idx_in_region = idx - region_start
        if 0 <= current_idx_in_region < len(self._current_region_base_signal):
            realistic_base_val = self._current_region_base_signal[current_idx_in_region]
        else:
            realistic_base_val = base_val  # Fallback
        
        # Generate noise that matches local characteristics
        mini_sequence_length = min(7, region_length)  # Slightly longer for better correlation
        if mini_sequence_length > 0:
            # Generate realistic noise sequence for better correlation
            noise_sequence = self._generate_realistic_noise_sequence(
                mini_sequence_length, noise_characteristics, np.zeros(mini_sequence_length)
            )
            
            # Extract noise value for our position with some smoothing
            sequence_idx = int(position * (mini_sequence_length - 1)) if mini_sequence_length > 1 else 0
            
            # Use weighted average of nearby noise values for smoother transitions
            if mini_sequence_length > 2 and 0 < sequence_idx < mini_sequence_length - 1:
                weights = np.array([0.25, 0.5, 0.25])
                indices = np.array([sequence_idx - 1, sequence_idx, sequence_idx + 1])
                noise_value = np.average(noise_sequence[indices], weights=weights)
            else:
                noise_value = noise_sequence[sequence_idx]
        else:
            # Fallback to sampling from local variations
            if len(noise_characteristics['local_variations']) > 0:
                sample_idx = np.random.randint(0, len(noise_characteristics['local_variations']))
                noise_value = noise_characteristics['local_variations'][sample_idx]
            else:
                noise_value = np.random.normal(0, noise_characteristics['noise_std'])
        
        # Scale noise based on configuration and aggressiveness
        noise_intensity = getattr(self.config, 'noise_intensity', 0.8)
        aggressiveness = getattr(self.config, 'noise_modeling_aggressiveness', 1.0)
        
        # Reduce noise intensity in the center slightly to prevent over-correction
        center_damping = 1.0 - 0.2 * (4 * position * (1 - position))  # Slight reduction at center
        scaled_noise = noise_value * noise_intensity * aggressiveness * center_damping
        
        # Combine realistic base with scaled noise
        enhanced_val = realistic_base_val + scaled_noise
        
        return enhanced_val
    
    def _add_simple_local_noise_variation(self, base_val: float, idx: int, region: list,
                                        left_context: np.ndarray, right_context: np.ndarray,
                                        intensities: np.ndarray) -> float:
        """Simple but effective approach: directly sample from local clean variations."""
        
        # Combine contexts and analyze only the actual point-to-point variations
        combined_clean = np.concatenate([left_context, right_context])
        
        if len(combined_clean) < 3:
            return base_val  # No meaningful local data
        
        # Get point-to-point differences (the actual "noise" we want to mimic)
        local_diffs = np.diff(combined_clean)
        
        if len(local_diffs) == 0:
            return base_val
        
        # Simple approach: just pick a random local difference and apply it
        random_diff = np.random.choice(local_diffs)
        
        # Scale it based on configuration
        noise_intensity = getattr(self.config, 'noise_intensity', 0.8)
        scaled_diff = random_diff * noise_intensity
        
        # Apply to base value
        enhanced_val = base_val + scaled_diff
        
        return enhanced_val
    
    def _apply_enhanced_linear_interpolation(self, cleaned_intensities: np.ndarray, region: list,
                                           left_idx: int, right_idx: int, intensities: np.ndarray,
                                           noise_std: float, left_trend: float, right_trend: float,
                                           local_mean: float, local_median: float, 
                                           left_context: np.ndarray = None, right_context: np.ndarray = None):
        """Apply enhanced linear interpolation with intelligent variation."""
        left_val = intensities[left_idx]
        right_val = intensities[right_idx]
        
        # Get context if not provided
        if left_context is None:
            context_start = max(0, left_idx - 10)
            left_context = intensities[context_start:left_idx+1]
        if right_context is None:
            context_end = min(len(intensities), right_idx + 10)
            right_context = intensities[right_idx:context_end+1]
        
        for i, idx in enumerate(region):
            # Basic linear interpolation
            total_span = right_idx - left_idx
            current_span = idx - left_idx
            weight = current_span / total_span if total_span > 0 else 0.5
            base_val = left_val + weight * (right_val - left_val)
            
            # Add intelligent variation with context
            enhanced_val = self._add_intelligent_variation(
                base_val, idx, region, noise_std, left_trend, right_trend, local_mean,
                intensities, left_context, right_context
            )
            
            # Apply bounds
            final_val = max(enhanced_val, local_median * 0.3)
            final_val = min(final_val, local_mean * 3)
            
            cleaned_intensities[idx] = final_val

    def process_batch_parallel(self, spectra_data: List[Tuple[np.ndarray, np.ndarray, str]]) -> List[Tuple[bool, np.ndarray, dict]]:
        """
        Process multiple spectra in parallel using all available CPU cores.
        
        Args:
            spectra_data: List of (wavenumbers, intensities, spectrum_id) tuples
            
        Returns:
            List of (cosmic_detected, cleaned_intensities, detection_info) tuples
        """
        if not self.config.use_parallel_processing or len(spectra_data) < 10:
            # For small batches, sequential processing is faster due to overhead
            return [self.detect_and_remove_cosmic_rays(wn, intens, sid) 
                    for wn, intens, sid in spectra_data]
        
        # Determine number of jobs
        n_jobs = self.config.n_jobs
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        # Process in parallel using joblib
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(self.detect_and_remove_cosmic_rays)(wavenumbers, intensities, spectrum_id)
            for wavenumbers, intensities, spectrum_id in spectra_data
        )
        
        return results
    
    def reset_statistics(self):
        """Reset cosmic ray statistics."""
        self.statistics = {
            'total_spectra': 0,
            'spectra_with_cosmic_rays': 0,
            'total_cosmic_rays_removed': 0,
            'shape_analysis_rejections': 0,
            'false_positive_prevention': 0
        } 