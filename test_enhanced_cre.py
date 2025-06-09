#!/usr/bin/env python3
"""
Test script for the enhanced Cosmic Ray Elimination (CRE) system with shape analysis.

This script demonstrates the new shape-based discrimination between cosmic rays and Raman peaks.
"""

import numpy as np
import matplotlib.pyplot as plt
from map_analysis_2d.core import CosmicRayConfig, SimpleCosmicRayManager

def create_test_spectrum():
    """Create a synthetic Raman spectrum with both cosmic rays and Raman peaks."""
    # Create wavenumber range
    wavenumbers = np.linspace(200, 3000, 1000)
    
    # Create baseline
    baseline = 100 + 0.01 * wavenumbers
    
    # Add some Raman peaks (broad, gradual)
    raman_peaks = np.zeros_like(wavenumbers)
    
    # Strong Raman peak at 1000 cm-1 (broad, high intensity)
    peak1_center = 400  # index for ~1000 cm-1
    peak1_width = 15    # broad peak
    peak1_intensity = 8000  # very strong
    raman_peaks += peak1_intensity * np.exp(-((np.arange(len(wavenumbers)) - peak1_center) / peak1_width)**2)
    
    # Medium Raman peak at 1500 cm-1 (asymmetric)
    peak2_center = 550  # index for ~1500 cm-1
    peak2_width_left = 10
    peak2_width_right = 20  # asymmetric
    peak2_intensity = 5000
    indices = np.arange(len(wavenumbers))
    left_mask = indices <= peak2_center
    right_mask = indices > peak2_center
    raman_peaks[left_mask] += peak2_intensity * np.exp(-((indices[left_mask] - peak2_center) / peak2_width_left)**2)
    raman_peaks[right_mask] += peak2_intensity * np.exp(-((indices[right_mask] - peak2_center) / peak2_width_right)**2)
    
    # Smaller Raman peak at 2000 cm-1
    peak3_center = 700  # index for ~2000 cm-1
    peak3_width = 8
    peak3_intensity = 3000
    raman_peaks += peak3_intensity * np.exp(-((np.arange(len(wavenumbers)) - peak3_center) / peak3_width)**2)
    
    # Create cosmic rays (narrow, sharp spikes)
    cosmic_rays = np.zeros_like(wavenumbers)
    
    # Cosmic ray 1: Very narrow, very intense
    cr1_idx = 300
    cosmic_rays[cr1_idx] = 15000  # Very high intensity
    cosmic_rays[cr1_idx-1] = 2000  # Small bleed to adjacent pixels
    cosmic_rays[cr1_idx+1] = 2000
    
    # Cosmic ray 2: Narrow but not as intense (should still be detected)
    cr2_idx = 600
    cosmic_rays[cr2_idx] = 8000
    cosmic_rays[cr2_idx-1] = 1000
    cosmic_rays[cr2_idx+1] = 1000
    
    # Cosmic ray 3: Wide cosmic ray (detector overflow)
    cr3_idx = 800
    for i in range(-2, 3):  # 5-point wide cosmic ray
        if 0 <= cr3_idx + i < len(cosmic_rays):
            cosmic_rays[cr3_idx + i] = 12000 + np.random.normal(0, 500)  # Plateau with noise
    
    # Add noise
    noise = np.random.normal(0, 50, len(wavenumbers))
    
    # Combine all components
    intensities = baseline + raman_peaks + cosmic_rays + noise
    
    # Ensure no negative values
    intensities = np.maximum(intensities, 0)
    
    return wavenumbers, intensities

def test_shape_analysis():
    """Test the shape analysis capabilities of the enhanced CRE system."""
    print("Testing Enhanced Cosmic Ray Elimination with Shape Analysis")
    print("=" * 60)
    
    # Create test spectrum
    wavenumbers, intensities = create_test_spectrum()
    
    # Test with shape analysis enabled
    print("\n1. Testing with Shape Analysis ENABLED:")
    config_with_shape = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1500,
        neighbor_ratio=10.0,
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
    
    # Test with shape analysis disabled (traditional method)
    print("\n2. Testing with Shape Analysis DISABLED:")
    config_without_shape = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1500,
        neighbor_ratio=10.0,
        enable_shape_analysis=False
    )
    
    cre_manager_without_shape = SimpleCosmicRayManager(config_without_shape)
    has_cosmic_rays_trad, cleaned_intensities_trad, detection_info_trad = cre_manager_without_shape.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, "test_spectrum_traditional"
    )
    
    print(f"   Cosmic rays detected: {has_cosmic_rays_trad}")
    print(f"   Cosmic rays removed: {detection_info_trad['cosmic_ray_count']}")
    
    # Run shape analysis diagnosis
    print("\n3. Shape Analysis Diagnosis:")
    diagnosis = cre_manager_with_shape.diagnose_peak_shape(wavenumbers, intensities)
    
    if 'peak_details' in diagnosis:
        print(f"   Total peaks analyzed: {diagnosis['total_peaks_analyzed']}")
        print(f"   Cosmic rays identified: {diagnosis['cosmic_rays_identified']}")
        print(f"   Raman peaks identified: {diagnosis['raman_peaks_identified']}")
        
        print("\n   Peak Details:")
        for i, peak in enumerate(diagnosis['peak_details'][:5]):  # Show first 5 peaks
            print(f"   Peak {i+1}: {peak['classification']} at index {peak['index']}")
            print(f"      Intensity: {peak['intensity']:.0f}")
            print(f"      FWHM: {peak['shape_metrics']['fwhm']:.1f}")
            print(f"      Sharpness: {peak['shape_metrics']['sharpness_ratio']:.1f}")
            print(f"      Asymmetry: {peak['shape_metrics']['asymmetry_factor']:.2f}")
            print(f"      Gradient: {peak['shape_metrics']['avg_gradient']:.0f}")
            print(f"      Reason: {peak['reason']}")
            print()
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Original spectrum
    plt.subplot(2, 2, 1)
    plt.plot(wavenumbers, intensities, 'b-', linewidth=1, label='Original')
    plt.title('Original Spectrum with Cosmic Rays')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Shape analysis enabled
    plt.subplot(2, 2, 2)
    plt.plot(wavenumbers, intensities, 'b-', linewidth=1, alpha=0.5, label='Original')
    plt.plot(wavenumbers, cleaned_intensities_shape, 'r-', linewidth=1, label='Shape Analysis Enabled')
    plt.title(f'Shape Analysis Enabled\n({detection_info_shape["cosmic_ray_count"]} cosmic rays removed)')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Shape analysis disabled
    plt.subplot(2, 2, 3)
    plt.plot(wavenumbers, intensities, 'b-', linewidth=1, alpha=0.5, label='Original')
    plt.plot(wavenumbers, cleaned_intensities_trad, 'g-', linewidth=1, label='Traditional Method')
    plt.title(f'Traditional Method\n({detection_info_trad["cosmic_ray_count"]} cosmic rays removed)')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparison
    plt.subplot(2, 2, 4)
    plt.plot(wavenumbers, cleaned_intensities_shape, 'r-', linewidth=1, label='Shape Analysis')
    plt.plot(wavenumbers, cleaned_intensities_trad, 'g-', linewidth=1, label='Traditional')
    plt.title('Comparison of Methods')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cre_shape_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics comparison
    print("\n4. Statistics Comparison:")
    stats_shape = cre_manager_with_shape.get_statistics()
    stats_trad = cre_manager_without_shape.get_statistics()
    
    print(f"   Shape Analysis Method:")
    print(f"      Cosmic rays removed: {stats_shape['total_cosmic_rays_removed']}")
    print(f"      False positive prevention rate: {stats_shape['false_positive_prevention_rate']:.1f}%")
    
    print(f"   Traditional Method:")
    print(f"      Cosmic rays removed: {stats_trad['total_cosmic_rays_removed']}")
    print(f"      False positive prevention rate: {stats_trad['false_positive_prevention_rate']:.1f}%")
    
    print("\n5. Key Benefits of Shape Analysis:")
    print("   ✓ Distinguishes between cosmic rays and strong Raman peaks")
    print("   ✓ Reduces false positive removal of legitimate spectral features")
    print("   ✓ Uses multiple shape criteria (FWHM, sharpness, asymmetry, gradient)")
    print("   ✓ Provides detailed diagnostic information for parameter tuning")
    print("   ✓ Maintains high sensitivity to true cosmic ray events")

if __name__ == "__main__":
    test_shape_analysis() 