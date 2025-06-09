#!/usr/bin/env python3
"""
Simple test to verify cosmic ray detection is working.
"""

import numpy as np
import sys

sys.path.append('.')
from map_analysis_2d.core import CosmicRayConfig, SimpleCosmicRayManager

def test_simple_cosmic_ray():
    """Test with very simple, obvious cosmic rays."""
    print("Testing Simple Cosmic Ray Detection")
    print("=" * 40)
    
    # Create a simple spectrum
    wavenumbers = np.linspace(800, 1000, 100)
    intensities = np.full(100, 100.0)  # Flat baseline at 100
    
    # Add obvious cosmic rays
    intensities[50] = 5000  # Single isolated cosmic ray
    intensities[70:73] = [3000, 3500, 2800]  # Cluster of cosmic rays
    
    print(f"Baseline intensity: 100")
    print(f"Cosmic ray at index 50: {intensities[50]}")
    print(f"Cosmic ray cluster at 70-72: {intensities[70:73]}")
    
    # Test with very permissive settings
    config = CosmicRayConfig(
        enabled=True,
        absolute_threshold=500,  # Very low threshold
        neighbor_ratio=2.0,      # Very low neighbor ratio
        enable_shape_analysis=False  # Disable shape analysis for simplicity
    )
    
    cre_manager = SimpleCosmicRayManager(config)
    cosmic_detected, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, "simple_test"
    )
    
    print(f"\nResults:")
    print(f"Cosmic rays detected: {cosmic_detected}")
    print(f"Detected indices: {detection_info.get('cosmic_ray_indices', [])}")
    print(f"Expected indices: [50, 70, 71, 72]")
    
    # Check specific points
    print(f"\nIntensity changes:")
    for i in [50, 70, 71, 72]:
        original = intensities[i]
        cleaned = cleaned_intensities[i]
        print(f"  Index {i}: {original} → {cleaned:.1f}")
    
    return len(detection_info.get('cosmic_ray_indices', [])) > 0

def test_user_scenario_aggressive():
    """Test user scenario with very aggressive settings."""
    print("\n" + "=" * 40)
    print("Testing User Scenario with Aggressive Settings")
    print("=" * 40)
    
    # User's scenario
    wavenumbers = np.linspace(800, 1000, 274)
    intensities = np.random.normal(500, 100, len(wavenumbers))
    
    # Add cosmic rays
    cosmic_indices = [269, 270, 271, 272, 273]
    cosmic_intensities = [2593, 2535, 2596, 3503, 2235]
    
    for idx, intensity in zip(cosmic_indices, cosmic_intensities):
        if idx < len(intensities):
            intensities[idx] = intensity
    
    # Very aggressive settings
    config = CosmicRayConfig(
        enabled=True,
        absolute_threshold=1000,  # Lower threshold
        neighbor_ratio=1.5,       # Very low neighbor ratio
        enable_shape_analysis=False  # Disable shape analysis
    )
    
    cre_manager = SimpleCosmicRayManager(config)
    cosmic_detected, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, "aggressive_test"
    )
    
    print(f"Results with aggressive settings:")
    print(f"Cosmic rays detected: {cosmic_detected}")
    print(f"Detected indices: {detection_info.get('cosmic_ray_indices', [])}")
    print(f"Expected indices: {cosmic_indices}")
    
    detected_count = len(detection_info.get('cosmic_ray_indices', []))
    print(f"Detection rate: {detected_count}/{len(cosmic_indices)}")
    
    return detected_count > 0

if __name__ == "__main__":
    print("Simple Cosmic Ray Detection Test")
    print("=" * 50)
    
    success1 = test_simple_cosmic_ray()
    success2 = test_user_scenario_aggressive()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Simple test: {'PASS' if success1 else 'FAIL'}")
    print(f"Aggressive test: {'PASS' if success2 else 'FAIL'}")
    
    if not success1:
        print("\n❌ Basic cosmic ray detection is not working!")
        print("   There may be a fundamental issue with the detection algorithm.")
    elif not success2:
        print("\n⚠️  Basic detection works, but consecutive cosmic rays are problematic.")
        print("   The cluster detection logic needs improvement.")
    else:
        print("\n✅ Cosmic ray detection is working!")
        print("   The issue may be with the default parameter settings.") 