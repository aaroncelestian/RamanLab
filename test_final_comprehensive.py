#!/usr/bin/env python3
"""
Final comprehensive test of the enhanced cosmic ray elimination system
with updated default parameters.
"""

import numpy as np
from test_cre_simple_standalone import CosmicRayConfig, SimpleCosmicRayManager

def test_comprehensive_cre():
    """Comprehensive test of the enhanced CRE system."""
    print('Comprehensive Enhanced CRE System Test')
    print('=' * 50)
    
    # Test 1: Consecutive cosmic rays (original problem)
    print('\n1. Testing Consecutive Cosmic Rays (Original Problem)')
    print('-' * 50)
    
    wavenumbers = np.linspace(100, 3000, 500)
    intensities1 = np.random.normal(1000, 100, 500)
    
    # Add consecutive cosmic rays at indices 269-273
    consecutive_indices = [269, 270, 271, 272, 273]
    for idx in consecutive_indices:
        intensities1[idx] = 8000 + np.random.normal(0, 200)
    
    print(f'Original consecutive cosmic rays at: {consecutive_indices}')
    print(f'Original intensities: {[int(intensities1[i]) for i in consecutive_indices]}')
    
    # Use updated default parameters
    config = CosmicRayConfig()  # Uses new default parameters
    cre_manager = SimpleCosmicRayManager(config)
    
    has_cr1, cleaned1, info1 = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities1, 'consecutive_test'
    )
    
    print(f'Detected: {info1["cosmic_ray_indices"]} ({info1["cosmic_ray_count"]}/5)')
    print(f'Shape analysis enabled: {info1["shape_analysis_enabled"]}')
    if info1["cosmic_ray_count"] > 0:
        print(f'Cleaned intensities: {[int(cleaned1[i]) for i in info1["cosmic_ray_indices"]]}')
    
    # Test 2: Isolated cosmic ray
    print('\n2. Testing Isolated Cosmic Ray')
    print('-' * 50)
    
    intensities2 = np.random.normal(1000, 100, 500)
    intensities2[50] = 9000  # Single isolated cosmic ray
    
    print(f'Isolated cosmic ray at index 50: {int(intensities2[50])}')
    
    has_cr2, cleaned2, info2 = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities2, 'isolated_test'
    )
    
    print(f'Detected: {info2["cosmic_ray_indices"]} ({info2["cosmic_ray_count"]}/1)')
    if info2["cosmic_ray_count"] > 0:
        print(f'Cleaned intensity: {int(cleaned2[50])}')
    
    # Test 3: Mixed scenario
    print('\n3. Testing Mixed Scenario (Isolated + Consecutive)')
    print('-' * 50)
    
    intensities3 = np.random.normal(1000, 100, 500)
    
    # Add isolated cosmic ray
    intensities3[100] = 8500
    
    # Add consecutive cosmic rays
    consecutive_indices2 = [200, 201, 202]
    for idx in consecutive_indices2:
        intensities3[idx] = 7500 + np.random.normal(0, 100)
    
    # Add another isolated cosmic ray
    intensities3[350] = 9200
    
    total_expected = 1 + len(consecutive_indices2) + 1  # 5 total cosmic rays
    
    print(f'Isolated cosmic rays at: [100, 350]')
    print(f'Consecutive cosmic rays at: {consecutive_indices2}')
    print(f'Total expected: {total_expected}')
    
    has_cr3, cleaned3, info3 = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities3, 'mixed_test'
    )
    
    print(f'Detected: {info3["cosmic_ray_indices"]} ({info3["cosmic_ray_count"]}/{total_expected})')
    
    # Test 4: No cosmic rays (should not false positive)
    print('\n4. Testing Normal Spectrum (No Cosmic Rays)')
    print('-' * 50)
    
    intensities4 = np.random.normal(1000, 100, 500)
    # Add some normal Raman peaks
    intensities4[150:155] = np.random.normal(3000, 200, 5)  # Broad Raman peak
    intensities4[300:305] = np.random.normal(2500, 150, 5)  # Another broad peak
    
    has_cr4, cleaned4, info4 = cre_manager.detect_and_remove_cosmic_rays(
        wavenumbers, intensities4, 'normal_test'
    )
    
    print(f'Detected cosmic rays: {info4["cosmic_ray_count"]} (should be 0)')
    print(f'False positives prevented: {cre_manager.get_statistics().get("false_positive_prevention", 0)}')
    
    # Summary
    print('\n' + '=' * 50)
    print('SUMMARY')
    print('=' * 50)
    
    stats = cre_manager.get_statistics()
    print(f'Total spectra processed: {stats["total_spectra"]}')
    print(f'Spectra with cosmic rays: {stats["spectra_with_cosmic_rays"]}')
    print(f'Total cosmic rays removed: {stats["total_cosmic_rays_removed"]}')
    print(f'False positives prevented: {stats.get("false_positive_prevention", 0)}')
    
    test_results = {
        'consecutive_detection': info1["cosmic_ray_count"],
        'isolated_detection': info2["cosmic_ray_count"],
        'mixed_detection': info3["cosmic_ray_count"],
        'false_positives': info4["cosmic_ray_count"]
    }
    
    return test_results

def evaluate_results(results):
    """Evaluate the test results."""
    print('\nEVALUATION')
    print('=' * 50)
    
    score = 0
    max_score = 4
    
    # Test 1: Consecutive cosmic rays (most important)
    if results['consecutive_detection'] >= 3:  # At least 3/5 is good
        print('✅ Consecutive cosmic ray detection: GOOD')
        score += 1
    else:
        print('❌ Consecutive cosmic ray detection: NEEDS IMPROVEMENT')
    
    # Test 2: Isolated cosmic rays
    if results['isolated_detection'] >= 1:
        print('✅ Isolated cosmic ray detection: GOOD')
        score += 1
    else:
        print('❌ Isolated cosmic ray detection: NEEDS IMPROVEMENT')
    
    # Test 3: Mixed scenario
    if results['mixed_detection'] >= 3:  # At least 3/5 is acceptable
        print('✅ Mixed scenario detection: GOOD')
        score += 1
    else:
        print('❌ Mixed scenario detection: NEEDS IMPROVEMENT')
    
    # Test 4: False positives
    if results['false_positives'] == 0:
        print('✅ False positive prevention: EXCELLENT')
        score += 1
    else:
        print('❌ False positive prevention: NEEDS IMPROVEMENT')
    
    print(f'\nOverall Score: {score}/{max_score}')
    
    if score >= 3:
        print('🎉 Enhanced CRE system is working well!')
        return True
    else:
        print('⚠️  Enhanced CRE system needs further tuning.')
        return False

if __name__ == "__main__":
    results = test_comprehensive_cre()
    success = evaluate_results(results)
    
    print(f'\nDetailed Results:')
    print(f'  Consecutive cosmic rays detected: {results["consecutive_detection"]}/5')
    print(f'  Isolated cosmic ray detected: {results["isolated_detection"]}/1')
    print(f'  Mixed scenario detected: {results["mixed_detection"]}/5')
    print(f'  False positives: {results["false_positives"]}/0') 