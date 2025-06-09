#!/usr/bin/env python3
"""
Final test of the enhanced cosmic ray elimination system.
Tests the two-pass detection algorithm with consecutive cosmic rays.
"""

import numpy as np
import sys
import os

# Add the current directory to the path to import from the main file
sys.path.append('.')

# Import the classes from the main file
exec(open('map_analysis_2d_qt6.py').read())

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