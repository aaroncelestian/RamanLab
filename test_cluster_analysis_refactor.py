#!/usr/bin/env python3
"""
Test script for the refactored cluster analysis module.

This script tests that all components can be imported and instantiated correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all refactored components can be imported."""
    print("Testing imports...")
    
    try:
        # Test main module import
        from cluster_analysis import RamanClusterAnalysisQt6
        print("✓ Main cluster analysis module imported successfully")
        
        # Test core components
        from cluster_analysis.core import ClusteringEngine, DataProcessor
        print("✓ Core components imported successfully")
        
        # Test visualization
        from cluster_analysis.visualization import ClusterPlotter
        print("✓ Visualization components imported successfully")
        
        # Test dialogs
        from cluster_analysis.dialogs import DatabaseImportDialog
        print("✓ Dialog components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_component_instantiation():
    """Test that components can be instantiated."""
    print("\nTesting component instantiation...")
    
    try:
        # Test clustering engine
        from cluster_analysis.core import ClusteringEngine
        engine = ClusteringEngine()
        print("✓ ClusteringEngine instantiated successfully")
        
        # Test data processor
        from cluster_analysis.core import DataProcessor
        processor = DataProcessor()
        print("✓ DataProcessor instantiated successfully")
        
        # Test plotter
        from cluster_analysis.visualization import ClusterPlotter
        plotter = ClusterPlotter()
        print("✓ ClusterPlotter instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Component instantiation failed: {e}")
        return False

def test_clustering_functionality():
    """Test basic clustering functionality."""
    print("\nTesting clustering functionality...")
    
    try:
        from cluster_analysis.core import ClusteringEngine
        
        # Create test data
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        
        # Generate 3 clusters of data
        cluster1 = np.random.normal(0, 1, (n_samples//3, n_features))
        cluster2 = np.random.normal(3, 1, (n_samples//3, n_features))
        cluster3 = np.random.normal(-3, 1, (n_samples//3, n_features))
        
        test_data = np.vstack([cluster1, cluster2, cluster3])
        
        # Test clustering engine
        engine = ClusteringEngine()
        
        # Scale features
        scaled_data = engine.scale_features(test_data)
        print("✓ Feature scaling works")
        
        # Perform clustering
        labels, linkage_matrix, distance_matrix, algorithm = engine.perform_hierarchical_clustering(
            scaled_data, n_clusters=3, linkage_method='ward', distance_metric='euclidean'
        )
        
        print(f"✓ Clustering completed using {algorithm}")
        print(f"✓ Found {len(np.unique(labels))} clusters")
        print(f"✓ Labels shape: {labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Clustering functionality test failed: {e}")
        return False

def test_data_processing():
    """Test data processing functionality."""
    print("\nTesting data processing functionality...")
    
    try:
        from cluster_analysis.core import DataProcessor
        
        # Create test spectral data
        np.random.seed(42)
        n_spectra = 20
        n_points = 100
        wavenumbers = np.linspace(100, 1000, n_points)
        
        # Generate synthetic spectra with peaks
        intensities = np.zeros((n_spectra, n_points))
        for i in range(n_spectra):
            # Add some random peaks
            for peak_pos in np.random.choice(range(20, 80), size=3):
                peak = np.exp(-(np.arange(n_points) - peak_pos)**2 / 50)
                intensities[i] += peak * np.random.uniform(0.5, 2.0)
            
            # Add noise
            intensities[i] += np.random.normal(0, 0.1, n_points)
        
        processor = DataProcessor()
        
        # Test normalization
        normalized = processor.normalize_spectra(intensities, method='max')
        print("✓ Spectrum normalization works")
        
        # Test feature extraction
        features = processor.extract_vibrational_features(normalized, wavenumbers)
        print(f"✓ Feature extraction works: {features.shape}")
        
        # Test peak detection
        peaks = processor.detect_peaks(intensities[0], wavenumbers)
        print(f"✓ Peak detection works: found {len(peaks['peak_indices'])} peaks")
        
        return True
        
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Refactored Cluster Analysis Module")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_component_instantiation,
        test_clustering_functionality,
        test_data_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
