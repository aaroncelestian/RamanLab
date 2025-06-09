#!/usr/bin/env python3
"""
Test cosmic ray detection with tuned parameters for consecutive cosmic rays.
"""

import numpy as np
from test_cre_simple_standalone import CosmicRayConfig, SimpleCosmicRayManager

def test_consecutive_cosmic_rays():
    """Test detection with parameters optimized for consecutive cosmic rays."""
    print('Testing Consecutive Cosmic Ray Detection with Tuned Parameters')
    print('=' * 60)

    # Create test data
    wavenumbers = np.linspace(100, 3000, 500)
    intensities = np.random.normal(1000, 100, 500)  # Background noise

    # Add consecutive cosmic rays at indices 269-273
    cosmic_ray_indices = [269, 270, 271, 272, 273]
    for idx in cosmic_ray_indices:
        intensities[idx] = 8000 + np.random.normal(0, 200)

    print(f'Original cosmic rays at indices: {cosmic_ray_indices}')
    print(f'Original intensities: {[int(intensities[i]) for i in cosmic_ray_indices]}')

    # Test different parameter combinations
    test_configs = [
        {
            'name': 'Standard Settings',
            'config': CosmicRayConfig(
                absolute_threshold=1500,
                neighbor_ratio=10.0,
                enable_shape_analysis=True,
                max_cosmic_fwhm=3.0,
                min_sharpness_ratio=5.0,
                max_asymmetry_factor=0.3,
                gradient_threshold=200.0
            )
        },
        {
            'name': 'More Lenient Shape Analysis',
            'config': CosmicRayConfig(
                absolute_threshold=1500,
                neighbor_ratio=10.0,
                enable_shape_analysis=True,
                max_cosmic_fwhm=5.0,      # Allow wider peaks
                min_sharpness_ratio=3.0,  # Lower sharpness requirement
                max_asymmetry_factor=0.5, # Allow more asymmetry
                gradient_threshold=100.0  # Lower gradient threshold
            )
        },
        {
            'name': 'Lower Thresholds',
            'config': CosmicRayConfig(
                absolute_threshold=1000,
                neighbor_ratio=5.0,
                enable_shape_analysis=True,
                max_cosmic_fwhm=5.0,
                min_sharpness_ratio=3.0,
                max_asymmetry_factor=0.5,
                gradient_threshold=100.0
            )
        },
        {
            'name': 'No Shape Analysis',
            'config': CosmicRayConfig(
                absolute_threshold=1500,
                neighbor_ratio=10.0,
                enable_shape_analysis=False  # Disable shape analysis
            )
        },
        {
            'name': 'Very Lenient',
            'config': CosmicRayConfig(
                absolute_threshold=1000,
                neighbor_ratio=3.0,
                enable_shape_analysis=False
            )
        }
    ]

    best_detection = 0
    best_config_name = ""

    for test in test_configs:
        print(f'\n--- Testing: {test["name"]} ---')
        
        cre_manager = SimpleCosmicRayManager(test['config'])
        has_cosmic_ray, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
            wavenumbers, intensities.copy(), 'test_spectrum'
        )
        
        detected_count = detection_info["cosmic_ray_count"]
        detected_indices = detection_info["cosmic_ray_indices"]
        
        print(f'Detected {detected_count}/5 cosmic rays')
        print(f'Detected indices: {detected_indices}')
        print(f'Shape analysis enabled: {detection_info["shape_analysis_enabled"]}')
        
        if detected_count > best_detection:
            best_detection = detected_count
            best_config_name = test["name"]
        
        # Show false positive prevention
        stats = cre_manager.get_statistics()
        false_positives_prevented = stats.get('false_positive_prevention', 0)
        print(f'False positives prevented: {false_positives_prevented}')
        
        if detected_count > 0:
            print(f'Cleaned intensities at detected positions: {[int(cleaned_intensities[i]) for i in detected_indices]}')

    print(f'\n' + '='*60)
    print(f'Best configuration: {best_config_name} (detected {best_detection}/5 cosmic rays)')
    
    return best_detection

def test_isolated_cosmic_ray():
    """Test detection of isolated cosmic ray for comparison."""
    print('\n' + '='*60)
    print('Testing Isolated Cosmic Ray for Comparison')
    print('='*60)
    
    wavenumbers = np.linspace(100, 3000, 500)
    intensities = np.random.normal(1000, 100, 500)
    intensities[50] = 9000  # Single isolated cosmic ray
    
    config = CosmicRayConfig(
        absolute_threshold=1500,
        neighbor_ratio=10.0,
        enable_shape_analysis=True
    )
    
    cre_manager = SimpleCosmicRayManager(config)
    has_cosmic_ray, cleaned_intensities, detection_info = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities, 'isolated_test'
    )
    
    print(f'Isolated cosmic ray at index 50: {int(intensities[50])}')
    print(f'Detected: {detection_info["cosmic_ray_indices"]}')
    print(f'Detection count: {detection_info["cosmic_ray_count"]}')
    
    if detection_info["cosmic_ray_count"] > 0:
        print(f'Cleaned intensity: {int(cleaned_intensities[50])}')

if __name__ == "__main__":
    best_consecutive = test_consecutive_cosmic_rays()
    test_isolated_cosmic_ray()
    
    print(f'\n' + '='*60)
    print('Summary:')
    print(f'Best consecutive cosmic ray detection: {best_consecutive}/5')
    if best_consecutive > 0:
        print("✅ Enhanced CRE system can detect consecutive cosmic rays with proper tuning!")
    else:
        print("❌ Enhanced CRE system needs further adjustment for consecutive cosmic rays.") 