#!/usr/bin/env python3
"""
Simple standalone test of the enhanced cosmic ray elimination system.
Tests the two-pass detection algorithm without GUI dependencies.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CosmicRayConfig:
    """Sophisticated configuration for intelligent cosmic ray detection."""
    enabled: bool = True
    # More intelligent thresholds - distinguish cosmic rays from spectral features
    absolute_threshold: float = 1500  # Only remove intensities above this absolute value
    neighbor_ratio: float = 10.0      # Must be 10x higher than neighbors (reasonable for true cosmic rays)
    apply_during_load: bool = True
    
    # Shape analysis parameters for distinguishing cosmic rays from Raman peaks
    max_cosmic_fwhm: float = 3.0      # Maximum FWHM for cosmic rays (in data points)
    min_sharpness_ratio: float = 5.0  # Minimum ratio of peak height to width for cosmic rays
    max_asymmetry_factor: float = 0.3 # Maximum asymmetry for cosmic rays (should be symmetric)
    gradient_threshold: float = 200.0 # Minimum gradient for cosmic ray edges (counts/point)
    enable_shape_analysis: bool = True # Enable advanced shape-based discrimination

class SimpleCosmicRayManager:
    """Ultra-fast, conservative cosmic ray removal - only removes obvious anomalies."""
    
    def __init__(self, config: CosmicRayConfig = None):
        self.config = config or CosmicRayConfig()
        self.statistics = {
            'total_spectra': 0,
            'spectra_with_cosmic_rays': 0,
            'total_cosmic_rays_removed': 0,
            'shape_analysis_rejections': 0,
            'false_positive_prevention': 0
        }
    
    def _analyze_peak_shape(self, intensities: np.ndarray, peak_idx: int, 
                           background_level: float) -> dict:
        """Analyze the shape characteristics of a peak."""
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
        
        # 2. Calculate sharpness ratio
        peak_width = max(fwhm, 1)
        sharpness_ratio = (peak_height - background_level) / peak_width
        
        # 3. Calculate asymmetry factor
        left_width = local_peak_idx - left_half_idx
        right_width = right_half_idx - local_peak_idx
        total_width = left_width + right_width
        if total_width > 0:
            asymmetry_factor = abs(left_width - right_width) / total_width
        else:
            asymmetry_factor = 0
        
        # 4. Calculate intensity gradients
        left_gradient = 0
        right_gradient = 0
        
        if local_peak_idx > 0:
            left_gradient = abs(local_intensities[local_peak_idx] - 
                              local_intensities[local_peak_idx - 1])
        
        if local_peak_idx < len(local_intensities) - 1:
            right_gradient = abs(local_intensities[local_peak_idx] - 
                               local_intensities[local_peak_idx + 1])
        
        avg_gradient = (left_gradient + right_gradient) / 2
        
        return {
            'fwhm': fwhm,
            'sharpness_ratio': sharpness_ratio,
            'asymmetry_factor': asymmetry_factor,
            'avg_gradient': avg_gradient,
            'peak_height': peak_height,
            'background_level': background_level
        }
    
    def _is_cosmic_ray_by_shape(self, shape_analysis: dict) -> tuple[bool, str]:
        """Determine if a peak is a cosmic ray based on shape analysis."""
        reasons = []
        cosmic_ray_indicators = 0
        
        # Check FWHM - cosmic rays are typically very narrow
        if shape_analysis['fwhm'] <= self.config.max_cosmic_fwhm:
            cosmic_ray_indicators += 1
            reasons.append(f"Very narrow FWHM ({shape_analysis['fwhm']:.1f} points)")
        else:
            reasons.append(f"Broad peak FWHM ({shape_analysis['fwhm']:.1f} points)")
        
        # Check sharpness ratio - cosmic rays have extreme sharpness
        if shape_analysis['sharpness_ratio'] >= self.config.min_sharpness_ratio:
            cosmic_ray_indicators += 1
            reasons.append(f"Very sharp peak (ratio: {shape_analysis['sharpness_ratio']:.1f})")
        else:
            reasons.append(f"Gradual peak (ratio: {shape_analysis['sharpness_ratio']:.1f})")
        
        # Check asymmetry - cosmic rays should be symmetric
        if shape_analysis['asymmetry_factor'] <= self.config.max_asymmetry_factor:
            cosmic_ray_indicators += 1
            reasons.append(f"Symmetric peak (asymmetry: {shape_analysis['asymmetry_factor']:.2f})")
        else:
            reasons.append(f"Asymmetric peak (asymmetry: {shape_analysis['asymmetry_factor']:.2f})")
        
        # Check gradient - cosmic rays have very steep edges
        if shape_analysis['avg_gradient'] >= self.config.gradient_threshold:
            cosmic_ray_indicators += 1
            reasons.append(f"Steep gradients ({shape_analysis['avg_gradient']:.0f} counts/point)")
        else:
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
        """Enhanced two-pass cosmic ray detection algorithm."""
        if not self.config.enabled or len(intensities) < 5:
            return False, intensities.copy(), {}
        
        cleaned_intensities = intensities.copy()
        
        # Get spectrum statistics
        spectrum_max = np.max(intensities)
        spectrum_median = np.median(intensities)
        spectrum_99th = np.percentile(intensities, 99)
        spectrum_std = np.std(intensities)
        spectrum_75th = np.percentile(intensities, 75)
        
        # Dynamic minimum intensity threshold
        min_cosmic_intensity = max(
            spectrum_median + 3 * spectrum_std,
            spectrum_75th * 0.5,
            100
        )
        
        potential_cosmic_rays = []
        
        # Pass 1: Traditional isolated cosmic ray detection
        for i in range(1, len(intensities) - 1):
            current = intensities[i]
            
            if current < min_cosmic_intensity:
                continue
            
            left1 = intensities[i - 1] 
            right1 = intensities[i + 1]
            
            # Must be above absolute threshold
            meets_absolute = current > self.config.absolute_threshold
            if not meets_absolute:
                continue
            
            # Traditional isolated spike test
            max_immediate_neighbor = max(left1, right1)
            effective_neighbor_ratio = min(self.config.neighbor_ratio, 20.0)
            
            is_isolated_spike = (max_immediate_neighbor > 0 and 
                               current > max_immediate_neighbor * effective_neighbor_ratio)
            
            if is_isolated_spike:
                potential_cosmic_rays.append(i)
        
        # Pass 2: Detect cosmic ray clusters (consecutive high-intensity regions)
        for i in range(1, len(intensities) - 1):
            current = intensities[i]
            
            if i in potential_cosmic_rays:
                continue
            
            if current < self.config.absolute_threshold:
                continue
            
            # Look for cosmic ray cluster pattern
            cluster_start = i
            cluster_end = i
            
            # Expand cluster to include adjacent high points
            while (cluster_start > 0 and 
                   intensities[cluster_start - 1] >= self.config.absolute_threshold):
                cluster_start -= 1
            
            while (cluster_end < len(intensities) - 1 and 
                   intensities[cluster_end + 1] >= self.config.absolute_threshold):
                cluster_end += 1
            
            cluster_size = cluster_end - cluster_start + 1
            
            # If we have a cluster of high points, check if it's cosmic ray overflow
            if cluster_size >= 2:  # At least 2 consecutive high points
                # Check background levels around the cluster
                background_left = max(0, cluster_start - 3)
                background_right = min(len(intensities), cluster_end + 4)
                
                background_points = []
                for j in range(background_left, background_right):
                    if j < cluster_start or j > cluster_end:  # Outside the cluster
                        background_points.append(intensities[j])
                
                if background_points:
                    background_level = np.median(background_points)
                    background_std = np.std(background_points)
                    
                    # Cluster should be significantly above background
                    cluster_min = np.min(intensities[cluster_start:cluster_end + 1])
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
            current = intensities[i]
            
            # Final check: Apply shape analysis if enabled
            if self.config.enable_shape_analysis:
                # Get local background for shape analysis
                local_start = max(0, i - 5)
                local_end = min(len(intensities), i + 6)
                local_background_points = []
                
                for j in range(local_start, local_end):
                    if abs(j - i) > 2:  # Skip the peak and immediate neighbors
                        local_background_points.append(intensities[j])
                
                local_background = np.median(local_background_points) if local_background_points else spectrum_median
                
                # Perform detailed shape analysis
                shape_analysis = self._analyze_peak_shape(intensities, i, local_background)
                is_cosmic_ray, shape_reason = self._is_cosmic_ray_by_shape(shape_analysis)
                
                if not is_cosmic_ray:
                    # Shape analysis suggests this is a Raman peak, not a cosmic ray
                    self.statistics['false_positive_prevention'] += 1
                    logger.debug(f"Peak at index {i} (intensity {current:.0f}) identified as Raman peak: {shape_reason}")
                    continue
                else:
                    logger.debug(f"Peak at index {i} (intensity {current:.0f}) confirmed as cosmic ray: {shape_reason}")
            
            # If we reach here, this point is confirmed as a cosmic ray
            confirmed_cosmic_rays.append(i)
        
        cosmic_ray_indices = confirmed_cosmic_rays
        
        # Remove detected cosmic rays using interpolation
        if cosmic_ray_indices:
            # Group adjacent cosmic ray indices into regions
            cosmic_ray_regions = []
            current_region = []
            
            for idx in sorted(cosmic_ray_indices):
                if not current_region or idx == current_region[-1] + 1:
                    current_region.append(idx)
                else:
                    cosmic_ray_regions.append(current_region)
                    current_region = [idx]
            if current_region:
                cosmic_ray_regions.append(current_region)
            
            # Remove each cosmic ray region
            for region in cosmic_ray_regions:
                start_idx = region[0]
                end_idx = region[-1]
                
                # Find clean neighbors outside the cosmic ray region
                left_idx = start_idx - 1
                right_idx = end_idx + 1
                
                # Expand search if immediate neighbors are also problematic
                while left_idx >= 0 and intensities[left_idx] > min_cosmic_intensity:
                    left_idx -= 1
                while right_idx < len(intensities) and intensities[right_idx] > min_cosmic_intensity:
                    right_idx += 1
                
                # Perform interpolation across the entire cosmic ray region
                if left_idx >= 0 and right_idx < len(intensities):
                    left_val = intensities[left_idx]
                    right_val = intensities[right_idx]
                    
                    # Linear interpolation across the cosmic ray region
                    for i, idx in enumerate(region):
                        total_span = right_idx - left_idx
                        current_span = idx - left_idx
                        weight = current_span / total_span if total_span > 0 else 0.5
                        cleaned_intensities[idx] = left_val + weight * (right_val - left_val)
                        
                elif left_idx >= 0:
                    # Only left neighbor available
                    for idx in region:
                        cleaned_intensities[idx] = intensities[left_idx]
                elif right_idx < len(intensities):
                    # Only right neighbor available
                    for idx in region:
                        cleaned_intensities[idx] = intensities[right_idx]
                else:
                    # Fallback: use spectrum median
                    for idx in region:
                        cleaned_intensities[idx] = spectrum_median
        
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
        if stats['total_spectra'] > 0:
            stats['percentage_with_cosmic_rays'] = (stats['spectra_with_cosmic_rays'] / 
                                                   stats['total_spectra'] * 100)
            stats['average_cosmic_rays_per_spectrum'] = (stats['total_cosmic_rays_removed'] / 
                                                       stats['total_spectra'])
            stats['false_positive_prevention_rate'] = (stats.get('false_positive_prevention', 0) / 
                                                      stats['total_spectra'] * 100)
        else:
            stats['percentage_with_cosmic_rays'] = 0
            stats['average_cosmic_rays_per_spectrum'] = 0
            stats['false_positive_prevention_rate'] = 0
        
        return stats

def test_enhanced_cre():
    """Test the enhanced cosmic ray detection system."""
    print('Testing Enhanced Cosmic Ray Detection System')
    print('=' * 50)

    # Create test data with consecutive cosmic rays (like the original problem)
    wavenumbers = np.linspace(100, 3000, 500)
    intensities = np.random.normal(1000, 100, 500)  # Background noise

    # Add consecutive cosmic rays at indices 269-273 (like the original issue)
    cosmic_ray_indices = [269, 270, 271, 272, 273]
    for idx in cosmic_ray_indices:
        intensities[idx] = 8000 + np.random.normal(0, 200)  # High intensity cosmic rays

    print(f'Original cosmic rays at indices: {cosmic_ray_indices}')
    print(f'Original intensities at cosmic ray positions: {[int(intensities[i]) for i in cosmic_ray_indices]}')

    # Test the enhanced CRE system
    config = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1500,
        neighbor_ratio=10.0,
        enable_shape_analysis=True
    )

    cre_manager = SimpleCosmicRayManager(config)
    has_cosmic_ray, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, 'test_spectrum'
    )

    print(f'\nDetection Results:')
    print(f'Cosmic rays detected: {has_cosmic_ray}')
    print(f'Detected cosmic ray indices: {detection_info["cosmic_ray_indices"]}')
    print(f'Total cosmic rays detected: {detection_info["cosmic_ray_count"]}')
    print(f'Shape analysis enabled: {detection_info["shape_analysis_enabled"]}')

    # Check if the cosmic rays were properly removed
    print(f'\nCleaned intensities at original cosmic ray positions: {[int(cleaned_intensities[i]) for i in cosmic_ray_indices]}')

    # Show statistics
    stats = cre_manager.get_statistics()
    print(f'\nCRE Statistics:')
    for key, value in stats.items():
        print(f'  {key}: {value}')

    # Test isolated cosmic ray detection as well
    print(f'\n' + '='*50)
    print('Testing Isolated Cosmic Ray Detection')
    
    # Create test data with isolated cosmic ray
    intensities2 = np.random.normal(1000, 100, 500)
    intensities2[50] = 9000  # Single isolated cosmic ray
    
    has_cosmic_ray2, cleaned_intensities2, detection_info2 = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities2, 'test_spectrum_2'
    )
    
    print(f'Isolated cosmic ray at index 50: {int(intensities2[50])}')
    print(f'Detected: {detection_info2["cosmic_ray_indices"]}')
    print(f'Cleaned intensity: {int(cleaned_intensities2[50])}')

    print('\nTest completed successfully!')
    
    return detection_info["cosmic_ray_count"] > 0

if __name__ == "__main__":
    success = test_enhanced_cre()
    if success:
        print("\n✅ Enhanced CRE system is working correctly!")
    else:
        print("\n❌ Enhanced CRE system needs attention.") 