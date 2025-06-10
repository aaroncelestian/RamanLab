#!/usr/bin/env python3
"""
Debug script for hybrid analysis functionality.
Run this to test hybrid analysis without the full GUI.
"""

import logging
import sys
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_array_operations():
    """Test the array operations that were causing issues."""
    logger.info("Testing array operations...")
    
    # Test 1: Array comparison
    try:
        test_array = np.array([1.5, 2.5, 3.5, 1.0, 4.0])
        threshold = 2.0
        
        # This should work fine
        mask = test_array > threshold
        logger.info(f"Array comparison test: {test_array} > {threshold} = {mask}")
        
        count = np.sum(mask)
        logger.info(f"Count above threshold: {count}")
        
        # Test boolean indexing
        for i, value in enumerate(test_array):
            is_above = mask[i]
            logger.info(f"Position {i}: value={value}, above_threshold={bool(is_above)}")
            
    except Exception as e:
        logger.error(f"Array comparison test failed: {e}")
    
    # Test 2: Multi-dimensional array handling
    try:
        test_2d = np.random.rand(5, 5) * 10
        threshold = 5.0
        
        candidates = test_2d > threshold
        logger.info(f"2D array shape: {test_2d.shape}")
        logger.info(f"Candidates shape: {candidates.shape}")
        
        # Test accessing individual elements
        for i in range(test_2d.shape[0]):
            for j in range(test_2d.shape[1]):
                value = test_2d[i, j]
                is_candidate = candidates[i, j]
                logger.info(f"Position ({i},{j}): value={value:.3f}, candidate={bool(is_candidate)}")
                
    except Exception as e:
        logger.error(f"2D array test failed: {e}")

def simulate_nmf_data():
    """Create simulated NMF data to test with."""
    logger.info("Creating simulated NMF data...")
    
    # Simulate some NMF components
    n_spectra = 100
    n_components = 5
    
    # Create random component data
    components = []
    for comp in range(n_components):
        # Each component is a list of intensities for each spectrum
        component_data = np.random.exponential(scale=2.0, size=n_spectra)
        components.append(component_data.tolist())
    
    nmf_results = {
        'components': components,
        'n_components': n_components,
        'explained_variance_ratio': np.random.rand(n_components).tolist()
    }
    
    logger.info(f"Created {n_components} components with {n_spectra} spectra each")
    logger.info(f"Component 0 type: {type(components[0])}")
    logger.info(f"Component 0 length: {len(components[0])}")
    logger.info(f"First few values of component 0: {components[0][:5]}")
    
    return nmf_results

def test_hybrid_analysis_logic():
    """Test the core logic of hybrid analysis."""
    logger.info("Testing hybrid analysis logic...")
    
    # Create simulated data
    nmf_results = simulate_nmf_data()
    components = nmf_results['components']
    
    # Test component selection
    component_index = 2  # Use component 3 (0-based index)
    threshold = 2.0
    
    if component_index < len(components):
        component_data = components[component_index]
        logger.info(f"Selected component {component_index + 1}")
        logger.info(f"Component data type: {type(component_data)}")
        logger.info(f"Component data length: {len(component_data)}")
        
        # Test the array conversion and comparison
        try:
            component_array = np.asarray(component_data).flatten()
            logger.info(f"Converted to array shape: {component_array.shape}")
            
            # Test threshold comparison
            mask = component_array > threshold
            above_threshold = np.sum(mask)
            total = len(component_array)
            percentage = (above_threshold / total) * 100
            
            logger.info(f"Above threshold ({threshold}): {above_threshold}/{total} ({percentage:.3f}%)")
            
        except Exception as e:
            logger.error(f"Array conversion test failed: {e}")
            return False
    else:
        logger.error(f"Component index {component_index} not available")
        return False
    
    return True

def main():
    """Main test function."""
    logger.info("Starting hybrid analysis debug tests...")
    
    try:
        # Run tests
        test_array_operations()
        test_hybrid_analysis_logic()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 