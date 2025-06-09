#!/usr/bin/env python3
"""
Simplified test script to verify cosmic ray elimination logic without GUI components.
"""

import numpy as np
import sys

# Add the current directory to the path so we can import the map analysis module
sys.path.append('.')

# Import only the cosmic ray components, not the GUI
from map_analysis_2d.core import CosmicRayConfig, SimpleCosmicRayManager

def test_user_scenario():
    """Test the specific scenario described by the user with 5 cosmic rays."""
    print("Testing User's Specific Scenario")
    print("=" * 40)
    
    # Create a spectrum similar to the user's case
    wavenumbers = np.linspace(800, 1000, 274)  # Similar length to user's data
    intensities = np.random.normal(500, 100, len(wavenumbers))  # Base spectrum
    
    # Add 5 cosmic rays at indices 269-273 with very high intensities (similar to user's report)
    cosmic_indices = [269, 270, 271, 272, 273]
    cosmic_intensities = [2593, 2535, 2596, 3503, 2235]  # High intensity values
    
    for idx, intensity in zip(cosmic_indices, cosmic_intensities):
        if idx < len(intensities):
            intensities[idx] = intensity
    
    print(f"Added cosmic rays at indices: {cosmic_indices}")
    print(f"With intensities: {cosmic_intensities}")
    
    # Test with the enhanced CRE system
    config = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1500,  # Same as UI default
        neighbor_ratio=5.0,       # Same as UI default (changed from 10.0)
        enable_shape_analysis=True,
        max_cosmic_fwhm=3.0,
        min_sharpness_ratio=5.0,
        max_asymmetry_factor=0.3,
        gradient_threshold=200.0
    )
    
    cre_manager = SimpleCosmicRayManager(config)
    
    # Apply cosmic ray detection and removal
    cosmic_detected, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, "spectrum_276_5"
    )
    
    print(f"\nCosmic Ray Detection Results:")
    print(f"Cosmic rays detected: {cosmic_detected}")
    print(f"Number of cosmic rays found: {len(detection_info.get('cosmic_ray_indices', []))}")
    print(f"Detected indices: {detection_info.get('cosmic_ray_indices', [])}")
    
    # Check what happened to each cosmic ray
    print(f"\nDetailed Results for Each Cosmic Ray:")
    detected_indices = set(detection_info.get('cosmic_ray_indices', []))
    for i, (idx, original_intensity) in enumerate(zip(cosmic_indices, cosmic_intensities)):
        if idx < len(cleaned_intensities):
            cleaned_intensity = cleaned_intensities[idx]
            was_detected = idx in detected_indices
            reduction = original_intensity - cleaned_intensity
            
            print(f"  Cosmic Ray {i+1} (Index {idx}):")
            print(f"    Original intensity: {original_intensity}")
            print(f"    Cleaned intensity: {cleaned_intensity:.1f}")
            print(f"    Reduction: {reduction:.1f}")
            print(f"    Detected: {was_detected}")
    
    # Calculate map impact
    original_integrated = np.trapz(intensities, wavenumbers)
    cleaned_integrated = np.trapz(cleaned_intensities, wavenumbers)
    map_difference = original_integrated - cleaned_integrated
    
    print(f"\nMap Integration Impact:")
    print(f"Original integrated intensity: {original_integrated:.0f}")
    print(f"Cleaned integrated intensity: {cleaned_integrated:.0f}")
    print(f"Difference (cosmic ray removal): {map_difference:.0f}")
    print(f"Percentage reduction: {(map_difference/original_integrated)*100:.2f}%")
    
    # Check if cosmic rays were actually removed
    cosmic_rays_removed = map_difference > 100  # Should see significant reduction
    
    print(f"\nVerification:")
    print(f"Cosmic rays successfully removed: {cosmic_rays_removed}")
    
    if cosmic_rays_removed:
        print("\n✅ SUCCESS: Cosmic rays detected and removed!")
        print("   The map should show different values when:")
        print("   - 'Use Processed Data' is checked (cosmic rays removed)")
        print("   - 'Use Processed Data' is unchecked (cosmic rays present)")
    else:
        print("\n❌ ISSUE: Cosmic rays were not effectively removed")
        print("   This could explain why the map doesn't update")
    
    # Test shape analysis diagnosis
    print(f"\nShape Analysis Diagnosis:")
    diagnosis = cre_manager.diagnose_peak_shape(wavenumbers, intensities, display_details=False)
    
    if 'peaks_found' in diagnosis:
        print(f"Found {len(diagnosis['peaks_found'])} peaks for analysis:")
        for i, peak_data in enumerate(diagnosis['peaks_found']):
            print(f"  Peak {i+1} (Index {peak_data['index']}): {peak_data['classification']}")
            print(f"    Reason: {peak_data['reasoning']}")
    
    return cosmic_rays_removed

def test_parameter_synchronization():
    """Test that parameter changes affect detection properly."""
    print("\n" + "=" * 40)
    print("Testing Parameter Synchronization")
    print("=" * 40)
    
    # Create a spectrum with a borderline cosmic ray
    wavenumbers = np.linspace(800, 1000, 200)
    intensities = np.random.normal(300, 50, len(wavenumbers))
    intensities[100] = 2000  # Borderline cosmic ray
    
    # Test with different thresholds
    thresholds = [1500, 2500]  # Below and above the cosmic ray intensity
    
    for threshold in thresholds:
        config = CosmicRayConfig(
            enabled=True,
            absolute_threshold=threshold,
            neighbor_ratio=5.0,
            enable_shape_analysis=True
        )
        
        cre_manager = SimpleCosmicRayManager(config)
        cosmic_detected, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
            wavenumbers, intensities, f"test_threshold_{threshold}"
        )
        
        detected_count = len(detection_info.get('cosmic_ray_indices', []))
        print(f"Threshold {threshold}: Detected {detected_count} cosmic rays")
    
    print("✅ Parameter synchronization test complete")

if __name__ == "__main__":
    print("Cosmic Ray Elimination - User Scenario Test")
    print("=" * 50)
    
    # Test the user's specific scenario
    success = test_user_scenario()
    
    # Test parameter changes
    test_parameter_synchronization()
    
    print("\n" + "=" * 50)
    print("TROUBLESHOOTING GUIDE:")
    print("=" * 50)
    
    if success:
        print("✅ CRE is working correctly. If cosmic rays still appear in the map:")
        print("")
        print("1. Check 'Use Processed Data' checkbox is CHECKED")
        print("2. Click '⚡ Reprocess All' after changing any CRE parameters")
        print("3. Verify CRE is enabled with the main checkbox")
        print("4. Check that shape analysis is enabled (checkable group box)")
        print("5. Look at the spectrum plot - it should show both raw and cleaned lines")
    else:
        print("❌ CRE may not be working as expected")
        print("   Check the CRE parameters and thresholds")
    
    print("\nKey Points:")
    print("- Cosmic rays are removed during data loading if 'Apply During Load' is true")
    print("- Maps use processed_intensities when 'Use Processed Data' is checked")
    print("- Parameter changes require clicking 'Reprocess All' to apply to existing data")
    print("- The enhanced shape analysis should distinguish cosmic rays from Raman peaks") 