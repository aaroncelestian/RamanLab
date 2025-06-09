#!/usr/bin/env python3
"""
Simple test script for the enhanced Cosmic Ray Elimination (CRE) system with shape analysis.
This version doesn't require GUI components.
"""

import numpy as np
import sys
import os

# Add the current directory to the path to import the CRE classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the necessary classes without GUI dependencies
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
    absolute_threshold: float = 1500
    neighbor_ratio: float = 10.0
    apply_during_load: bool = True
    max_cosmic_fwhm: float = 3.0
    min_sharpness_ratio: float = 5.0
    max_asymmetry_factor: float = 0.3
    gradient_threshold: float = 200.0
    enable_shape_analysis: bool = True

class SimpleCosmicRayManager:
    """Ultra-fast, conservative cosmic ray removal with shape analysis."""
    
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
        window_size = 10
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(intensities), peak_idx + window_size + 1)
        local_intensities = intensities[start_idx:end_idx]
        local_peak_idx = peak_idx - start_idx
        
        peak_height = intensities[peak_idx]
        
        # Calculate FWHM
        half_max = background_level + (peak_height - background_level) / 2
        left_half_idx = local_peak_idx
        right_half_idx = local_peak_idx
        
        for i in range(local_peak_idx - 1, -1, -1):
            if local_intensities[i] <= half_max:
                left_half_idx = i
                break
        
        for i in range(local_peak_idx + 1, len(local_intensities)):
            if local_intensities[i] <= half_max:
                right_half_idx = i
                break
        
        fwhm = right_half_idx - left_half_idx
        peak_width = max(fwhm, 1)
        sharpness_ratio = (peak_height - background_level) / peak_width
        
        # Calculate asymmetry
        left_width = local_peak_idx - left_half_idx
        right_width = right_half_idx - local_peak_idx
        total_width = left_width + right_width
        asymmetry_factor = abs(left_width - right_width) / total_width if total_width > 0 else 0
        
        # Calculate gradients
        left_gradient = abs(local_intensities[local_peak_idx] - local_intensities[local_peak_idx - 1]) if local_peak_idx > 0 else 0
        right_gradient = abs(local_intensities[local_peak_idx] - local_intensities[local_peak_idx + 1]) if local_peak_idx < len(local_intensities) - 1 else 0
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
        
        # Check FWHM
        if shape_analysis['fwhm'] <= self.config.max_cosmic_fwhm:
            cosmic_ray_indicators += 1
            reasons.append(f"Very narrow FWHM ({shape_analysis['fwhm']:.1f} points)")
        else:
            reasons.append(f"Broad peak FWHM ({shape_analysis['fwhm']:.1f} points)")
        
        # Check sharpness ratio
        if shape_analysis['sharpness_ratio'] >= self.config.min_sharpness_ratio:
            cosmic_ray_indicators += 1
            reasons.append(f"Very sharp peak (ratio: {shape_analysis['sharpness_ratio']:.1f})")
        else:
            reasons.append(f"Gradual peak (ratio: {shape_analysis['sharpness_ratio']:.1f})")
        
        # Check asymmetry
        if shape_analysis['asymmetry_factor'] <= self.config.max_asymmetry_factor:
            cosmic_ray_indicators += 1
            reasons.append(f"Symmetric peak (asymmetry: {shape_analysis['asymmetry_factor']:.2f})")
        else:
            reasons.append(f"Asymmetric peak (asymmetry: {shape_analysis['asymmetry_factor']:.2f})")
        
        # Check gradient
        if shape_analysis['avg_gradient'] >= self.config.gradient_threshold:
            cosmic_ray_indicators += 1
            reasons.append(f"Steep gradients ({shape_analysis['avg_gradient']:.0f} counts/point)")
        else:
            reasons.append(f"Gentle gradients ({shape_analysis['avg_gradient']:.0f} counts/point)")
        
        is_cosmic_ray = cosmic_ray_indicators >= 3
        reason_str = "; ".join(reasons)
        
        if is_cosmic_ray:
            return True, f"COSMIC RAY: {reason_str}"
        else:
            return False, f"RAMAN PEAK: {reason_str}"
    
    def detect_and_remove_cosmic_rays(self, wavenumbers: np.ndarray, intensities: np.ndarray, 
                                    spectrum_id: str = None) -> Tuple[bool, np.ndarray, dict]:
        """Detect and remove cosmic rays with shape analysis."""
        if not self.config.enabled or len(intensities) < 5:
            return False, intensities.copy(), {}
        
        cleaned_intensities = intensities.copy()
        cosmic_ray_indices = []
        
        spectrum_max = np.max(intensities)
        spectrum_median = np.median(intensities)
        spectrum_99th = np.percentile(intensities, 99)
        spectrum_std = np.std(intensities)
        spectrum_75th = np.percentile(intensities, 75)
        
        min_cosmic_intensity = max(
            spectrum_median + 3 * spectrum_std,
            spectrum_75th * 0.5,
            100
        )
        
        for i in range(2, len(intensities) - 2):
            current = intensities[i]
            
            if current < min_cosmic_intensity:
                continue
            
            left1 = intensities[i - 1] 
            right1 = intensities[i + 1]
            
            # Basic intensity and neighbor ratio checks
            meets_absolute = (current > self.config.absolute_threshold or 
                            current > spectrum_99th * 1.2 or
                            current > spectrum_median * 5)
            
            if not meets_absolute:
                continue
            
            max_immediate_neighbor = max(left1, right1)
            effective_neighbor_ratio = min(self.config.neighbor_ratio, 20.0)
            is_isolated_spike = (max_immediate_neighbor > 0 and 
                               current > max_immediate_neighbor * effective_neighbor_ratio)
            
            if not is_isolated_spike:
                continue
            
            # Local background check
            start_idx = max(0, i - 3)
            end_idx = min(len(intensities), i + 4)
            local_region = intensities[start_idx:end_idx]
            
            background_points = []
            for j, val in enumerate(local_region):
                abs_idx = start_idx + j
                if abs_idx not in [i-1, i, i+1]:
                    background_points.append(val)
            
            if len(background_points) >= 3:
                local_background = np.median(background_points)
                local_background_std = np.std(background_points)
                
                if current < local_background + 5 * local_background_std:
                    continue
                    
                background_to_cosmic_ratio = current / (local_background + 1e-10)
                if background_to_cosmic_ratio < 10:
                    continue
            
            # Shape analysis
            if self.config.enable_shape_analysis:
                local_background = np.median(background_points) if background_points else spectrum_median
                shape_analysis = self._analyze_peak_shape(intensities, i, local_background)
                is_cosmic_ray, shape_reason = self._is_cosmic_ray_by_shape(shape_analysis)
                
                if not is_cosmic_ray:
                    self.statistics['false_positive_prevention'] += 1
                    continue
            
            cosmic_ray_indices.append(i)
        
        # Remove cosmic rays
        if cosmic_ray_indices:
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
            
            for region in cosmic_ray_regions:
                start_idx = region[0]
                end_idx = region[-1]
                
                left_idx = start_idx - 1
                right_idx = end_idx + 1
                
                while left_idx >= 0 and intensities[left_idx] > min_cosmic_intensity:
                    left_idx -= 1
                while right_idx < len(intensities) and intensities[right_idx] > min_cosmic_intensity:
                    right_idx += 1
                
                if left_idx >= 0 and right_idx < len(intensities):
                    left_val = intensities[left_idx]
                    right_val = intensities[right_idx]
                    
                    for i, idx in enumerate(region):
                        total_span = right_idx - left_idx
                        current_span = idx - left_idx
                        weight = current_span / total_span if total_span > 0 else 0.5
                        cleaned_intensities[idx] = left_val + weight * (right_val - left_val)
                elif left_idx >= 0:
                    for idx in region:
                        cleaned_intensities[idx] = intensities[left_idx]
                elif right_idx < len(intensities):
                    for idx in region:
                        cleaned_intensities[idx] = intensities[right_idx]
                else:
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
            stats['false_positive_prevention_rate'] = (stats['false_positive_prevention'] / 
                                                      stats['total_spectra'] * 100)
        else:
            stats['percentage_with_cosmic_rays'] = 0
            stats['average_cosmic_rays_per_spectrum'] = 0
            stats['false_positive_prevention_rate'] = 0
        
        return stats

def create_test_spectrum():
    """Create a synthetic Raman spectrum with both cosmic rays and Raman peaks."""
    wavenumbers = np.linspace(200, 3000, 1000)
    baseline = 100 + 0.01 * wavenumbers
    raman_peaks = np.zeros_like(wavenumbers)
    
    # Strong Raman peak at 1000 cm-1 (broad, high intensity) - should NOT be removed
    peak1_center = 400
    peak1_width = 15
    peak1_intensity = 8000
    raman_peaks += peak1_intensity * np.exp(-((np.arange(len(wavenumbers)) - peak1_center) / peak1_width)**2)
    
    # Medium Raman peak at 1500 cm-1 (asymmetric) - should NOT be removed
    peak2_center = 550
    peak2_width_left = 10
    peak2_width_right = 20
    peak2_intensity = 5000
    indices = np.arange(len(wavenumbers))
    left_mask = indices <= peak2_center
    right_mask = indices > peak2_center
    raman_peaks[left_mask] += peak2_intensity * np.exp(-((indices[left_mask] - peak2_center) / peak2_width_left)**2)
    raman_peaks[right_mask] += peak2_intensity * np.exp(-((indices[right_mask] - peak2_center) / peak2_width_right)**2)
    
    # Very strong, narrow Raman peak that might be mistaken for cosmic ray
    peak3_center = 700
    peak3_width = 4  # Narrow but still broader than cosmic ray
    peak3_intensity = 12000  # Very high intensity
    raman_peaks += peak3_intensity * np.exp(-((np.arange(len(wavenumbers)) - peak3_center) / peak3_width)**2)
    
    # Create cosmic rays (narrow, sharp spikes)
    cosmic_rays = np.zeros_like(wavenumbers)
    
    # Cosmic ray 1: Very narrow, very intense - SHOULD be removed
    cr1_idx = 300
    cosmic_rays[cr1_idx] = 15000
    cosmic_rays[cr1_idx-1] = 1500  # Small bleed to adjacent pixels
    cosmic_rays[cr1_idx+1] = 1500
    
    # Cosmic ray 2: Narrow but not as intense - SHOULD be removed
    cr2_idx = 600
    cosmic_rays[cr2_idx] = 10000
    cosmic_rays[cr2_idx-1] = 800
    cosmic_rays[cr2_idx+1] = 800
    
    # Cosmic ray 3: Single point spike - SHOULD be removed
    cr3_idx = 800
    cosmic_rays[cr3_idx] = 18000  # Very high, single point
    
    # Add noise
    noise = np.random.normal(0, 50, len(wavenumbers))
    
    # Combine all components
    intensities = baseline + raman_peaks + cosmic_rays + noise
    intensities = np.maximum(intensities, 0)
    
    return wavenumbers, intensities

def test_shape_analysis():
    """Test the shape analysis capabilities."""
    print("Testing Enhanced Cosmic Ray Elimination with Shape Analysis")
    print("=" * 60)
    
    wavenumbers, intensities = create_test_spectrum()
    
    # Print some spectrum statistics for debugging
    print(f"\nSpectrum Statistics:")
    print(f"   Max intensity: {np.max(intensities):.0f}")
    print(f"   Median intensity: {np.median(intensities):.0f}")
    print(f"   99th percentile: {np.percentile(intensities, 99):.0f}")
    print(f"   Standard deviation: {np.std(intensities):.0f}")
    
    # Find the highest peaks
    high_peaks = []
    for i in range(1, len(intensities) - 1):
        if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1] and intensities[i] > 5000:
            high_peaks.append((i, intensities[i]))
    
    print(f"   High peaks found: {len(high_peaks)}")
    for i, (idx, intensity) in enumerate(high_peaks[:5]):
        print(f"      Peak {i+1}: index {idx}, intensity {intensity:.0f}, wavenumber {wavenumbers[idx]:.1f}")
    
    # Test with shape analysis enabled
    print("\n1. Testing with Shape Analysis ENABLED:")
    config_with_shape = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1000,  # Lower threshold to catch our test cosmic rays
        neighbor_ratio=5.0,       # Lower ratio to be more sensitive
        enable_shape_analysis=True,
        max_cosmic_fwhm=3.0,
        min_sharpness_ratio=5.0,
        max_asymmetry_factor=0.3,
        gradient_threshold=200.0
    )
    
    cre_manager_with_shape = SimpleCosmicRayManager(config_with_shape)
    has_cosmic_rays, cleaned_intensities_shape, detection_info_shape = cre_manager_with_shape.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, "test_spectrum_with_shape"
    )
    
    print(f"   Cosmic rays detected: {has_cosmic_rays}")
    print(f"   Cosmic rays removed: {detection_info_shape['cosmic_ray_count']}")
    print(f"   False positives prevented: {detection_info_shape['false_positives_prevented']}")
    
    # Test with shape analysis disabled
    print("\n2. Testing with Shape Analysis DISABLED:")
    config_without_shape = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1000,  # Same lower threshold
        neighbor_ratio=5.0,       # Same lower ratio
        enable_shape_analysis=False
    )
    
    cre_manager_without_shape = SimpleCosmicRayManager(config_without_shape)
    has_cosmic_rays_trad, cleaned_intensities_trad, detection_info_trad = cre_manager_without_shape.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, "test_spectrum_traditional"
    )
    
    print(f"   Cosmic rays detected: {has_cosmic_rays_trad}")
    print(f"   Cosmic rays removed: {detection_info_trad['cosmic_ray_count']}")
    
    # Statistics comparison
    print("\n3. Statistics Comparison:")
    stats_shape = cre_manager_with_shape.get_statistics()
    stats_trad = cre_manager_without_shape.get_statistics()
    
    print(f"   Shape Analysis Method:")
    print(f"      Cosmic rays removed: {stats_shape['total_cosmic_rays_removed']}")
    print(f"      False positive prevention rate: {stats_shape['false_positive_prevention_rate']:.1f}%")
    
    print(f"   Traditional Method:")
    print(f"      Cosmic rays removed: {stats_trad['total_cosmic_rays_removed']}")
    print(f"      False positive prevention rate: {stats_trad['false_positive_prevention_rate']:.1f}%")
    
    print("\n4. Key Benefits of Shape Analysis:")
    print("   ✓ Distinguishes between cosmic rays and strong Raman peaks")
    print("   ✓ Reduces false positive removal of legitimate spectral features")
    print("   ✓ Uses multiple shape criteria (FWHM, sharpness, asymmetry, gradient)")
    print("   ✓ Provides detailed diagnostic information for parameter tuning")
    print("   ✓ Maintains high sensitivity to true cosmic ray events")
    
    # Show some specific examples
    print("\n5. Example Peak Analysis:")
    for idx in detection_info_shape['cosmic_ray_indices'][:3]:  # Show first 3 cosmic rays
        if idx < len(wavenumbers):
            print(f"   Cosmic ray at wavenumber {wavenumbers[idx]:.1f} cm⁻¹ (intensity: {intensities[idx]:.0f})")
    
    print(f"\n   Original spectrum max intensity: {np.max(intensities):.0f}")
    print(f"   Cleaned spectrum max intensity (shape analysis): {np.max(cleaned_intensities_shape):.0f}")
    print(f"   Cleaned spectrum max intensity (traditional): {np.max(cleaned_intensities_trad):.0f}")
    print(f"   Intensity reduction (shape analysis): {np.max(intensities) - np.max(cleaned_intensities_shape):.0f}")
    print(f"   Intensity reduction (traditional): {np.max(intensities) - np.max(cleaned_intensities_trad):.0f}")
    
    # Check if the strong Raman peak was preserved
    strong_raman_idx = 700  # Where we put the strong narrow Raman peak
    if strong_raman_idx < len(intensities):
        original_peak = intensities[strong_raman_idx]
        shape_peak = cleaned_intensities_shape[strong_raman_idx]
        trad_peak = cleaned_intensities_trad[strong_raman_idx]
        
        print(f"\n6. Strong Raman Peak Preservation Test:")
        print(f"   Original strong Raman peak intensity: {original_peak:.0f}")
        print(f"   After shape analysis: {shape_peak:.0f} ({'preserved' if shape_peak > original_peak * 0.8 else 'removed'})")
        print(f"   After traditional method: {trad_peak:.0f} ({'preserved' if trad_peak > original_peak * 0.8 else 'removed'})")
        
        if shape_peak > trad_peak:
            print("   ✓ Shape analysis better preserved the strong Raman peak!")
        elif shape_peak == trad_peak:
            print("   = Both methods handled the strong Raman peak equally")
        else:
            print("   ⚠ Traditional method preserved the peak better")

if __name__ == "__main__":
    test_shape_analysis() 