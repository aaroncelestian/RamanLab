#!/usr/bin/env python3
"""
Test script to verify that cosmic ray elimination is working properly
and being reflected in the map visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the current directory to the path so we can import the map analysis module
sys.path.append('.')

from map_analysis_2d.core import CosmicRayConfig, SimpleCosmicRayManager

def create_test_spectrum_with_cosmic_rays():
    """Create a test spectrum with known cosmic rays."""
    # Create wavenumber array
    wavenumbers = np.linspace(800, 1000, 200)
    
    # Create base Raman spectrum with some peaks
    base_intensity = 100 + 50 * np.exp(-((wavenumbers - 850) / 20)**2)  # Peak at 850
    base_intensity += 80 * np.exp(-((wavenumbers - 950) / 15)**2)       # Peak at 950
    base_intensity += np.random.normal(0, 5, len(wavenumbers))          # Noise
    
    # Add cosmic rays at specific locations
    cosmic_ray_indices = [50, 120, 150]  # Indices where cosmic rays occur
    cosmic_ray_intensities = [2500, 3000, 2800]  # Very high intensities
    
    intensities = base_intensity.copy()
    for idx, cr_intensity in zip(cosmic_ray_indices, cosmic_ray_intensities):
        intensities[idx] = cr_intensity  # Sharp cosmic ray spike
    
    return wavenumbers, intensities, cosmic_ray_indices

def test_cosmic_ray_detection_and_removal():
    """Test that cosmic rays are properly detected and removed."""
    print("Testing Cosmic Ray Detection and Removal...")
    print("=" * 50)
    
    # Create test spectrum
    wavenumbers, original_intensities, known_cosmic_rays = create_test_spectrum_with_cosmic_rays()
    
    # Create CRE manager with shape analysis enabled
    config = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1500,
        neighbor_ratio=10.0,
        apply_during_load=True,
        enable_shape_analysis=True,
        max_cosmic_fwhm=3.0,
        min_sharpness_ratio=5.0,
        max_asymmetry_factor=0.3,
        gradient_threshold=200.0
    )
    
    cre_manager = SimpleCosmicRayManager(config)
    
    # Apply cosmic ray detection and removal
    cosmic_detected, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, original_intensities, "test_spectrum"
    )
    
    print(f"Cosmic rays detected: {cosmic_detected}")
    print(f"Known cosmic ray indices: {known_cosmic_rays}")
    print(f"Detected cosmic ray indices: {detection_info.get('cosmic_ray_indices', [])}")
    
    # Check detection accuracy
    detected_indices = detection_info.get('cosmic_ray_indices', [])
    correctly_detected = len(set(known_cosmic_rays) & set(detected_indices))
    false_positives = len(set(detected_indices) - set(known_cosmic_rays))
    missed = len(set(known_cosmic_rays) - set(detected_indices))
    
    print(f"\nDetection Results:")
    print(f"  Correctly detected: {correctly_detected}/{len(known_cosmic_rays)}")
    print(f"  False positives: {false_positives}")
    print(f"  Missed detections: {missed}")
    
    # Check intensity reduction at cosmic ray locations
    print(f"\nIntensity Reduction at Cosmic Ray Locations:")
    for idx in known_cosmic_rays:
        original = original_intensities[idx]
        cleaned = cleaned_intensities[idx]
        reduction = original - cleaned
        print(f"  Index {idx}: {original:.0f} → {cleaned:.0f} (reduced by {reduction:.0f})")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(wavenumbers, original_intensities, 'b-', label='Original spectrum', alpha=0.7)
    plt.plot(wavenumbers, cleaned_intensities, 'r-', label='Cosmic ray cleaned', linewidth=2)
    plt.scatter(wavenumbers[known_cosmic_rays], original_intensities[known_cosmic_rays], 
                c='red', s=100, marker='x', label='Known cosmic rays', linewidth=3)
    if detected_indices:
        plt.scatter(wavenumbers[detected_indices], original_intensities[detected_indices], 
                    c='orange', s=50, marker='o', label='Detected cosmic rays', alpha=0.7)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title('Cosmic Ray Detection and Removal Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    difference = original_intensities - cleaned_intensities
    plt.plot(wavenumbers, difference, 'g-', label='Removed intensity')
    plt.scatter(wavenumbers[known_cosmic_rays], difference[known_cosmic_rays], 
                c='red', s=100, marker='x', label='Known cosmic ray locations')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Removed Intensity')
    plt.title('Intensity Removed by CRE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cosmic_ray_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test shape analysis diagnosis
    print(f"\nShape Analysis Diagnosis:")
    diagnosis = cre_manager.diagnose_peak_shape(wavenumbers, original_intensities)
    
    if 'peaks_found' in diagnosis:
        for i, peak_data in enumerate(diagnosis['peaks_found']):
            print(f"  Peak {i+1}: {peak_data['classification']} - {peak_data['reasoning']}")
    
    # Get statistics
    stats = cre_manager.get_statistics()
    print(f"\nCRE Statistics:")
    print(f"  Total spectra processed: {stats['total_spectra']}")
    print(f"  Spectra with cosmic rays: {stats['spectra_with_cosmic_rays']}")
    print(f"  Total cosmic rays removed: {stats['total_cosmic_rays_removed']}")
    print(f"  False positive prevention rate: {stats['false_positive_prevention_rate']:.1f}%")
    
    return correctly_detected == len(known_cosmic_rays) and false_positives == 0

def test_map_integration_scenario():
    """Test the specific scenario described by the user."""
    print("\n" + "=" * 50)
    print("Testing Map Integration Scenario")
    print("=" * 50)
    
    # Simulate the user's scenario with multiple cosmic rays detected
    wavenumbers = np.linspace(800, 1000, 274)  # Similar to user's data
    intensities = np.random.normal(200, 50, len(wavenumbers))
    
    # Add cosmic rays at indices 269-273 (similar to user's report)
    cosmic_indices = [269, 270, 271, 272, 273]
    for idx in cosmic_indices:
        if idx < len(intensities):
            intensities[idx] = np.random.uniform(2500, 3500)  # High intensity cosmic rays
    
    # Test with shape analysis enabled
    config = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1500,
        neighbor_ratio=5.0,
        enable_shape_analysis=True,
        max_cosmic_fwhm=3.0,
        min_sharpness_ratio=5.0,
        max_asymmetry_factor=0.3,
        gradient_threshold=200.0
    )
    
    cre_manager = SimpleCosmicRayManager(config)
    
    # Simulate what happens during data loading
    print("Simulating data loading with CRE enabled...")
    cosmic_detected, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, "user_spectrum_276_5"
    )
    
    print(f"Cosmic rays detected: {cosmic_detected}")
    print(f"Detection info: {detection_info}")
    
    # Simulate what should happen in the map visualization
    print("\nSimulating map integration:")
    
    # Original data (what would be in .intensities)
    original_integrated = np.trapz(intensities, wavenumbers)
    
    # Cleaned data (what should be in processed_intensities after CRE)
    cleaned_integrated = np.trapz(cleaned_intensities, wavenumbers)
    
    print(f"Original integrated intensity: {original_integrated:.0f}")
    print(f"Cleaned integrated intensity: {cleaned_integrated:.0f}")
    print(f"Difference due to cosmic ray removal: {original_integrated - cleaned_integrated:.0f}")
    
    # Check that cosmic rays are actually removed
    cosmic_removed = any(intensities[i] != cleaned_intensities[i] for i in cosmic_indices if i < len(intensities))
    print(f"Cosmic rays actually removed from data: {cosmic_removed}")
    
    # Show the impact on map visualization
    print(f"\nMap visualization impact:")
    print(f"- If 'Use Processed Data' is checked: Map shows {cleaned_integrated:.0f}")
    print(f"- If 'Use Processed Data' is unchecked: Map shows {original_integrated:.0f}")
    print(f"- User should see a difference of {abs(original_integrated - cleaned_integrated):.0f}")
    
    return cosmic_removed

if __name__ == "__main__":
    print("Cosmic Ray Elimination Integration Test")
    print("=" * 60)
    
    # Run tests
    success1 = test_cosmic_ray_detection_and_removal()
    success2 = test_map_integration_scenario()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"Basic CRE functionality: {'PASS' if success1 else 'FAIL'}")
    print(f"Map integration scenario: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n✅ All tests passed! CRE should be working correctly.")
        print("\nIf cosmic rays are still visible in the map:")
        print("1. Ensure 'Use Processed Data' is checked")
        print("2. Click 'Reprocess All' after changing CRE parameters")
        print("3. Click 'Update Map' button to refresh the visualization")
    else:
        print("\n❌ Some tests failed. Check the CRE implementation.") 