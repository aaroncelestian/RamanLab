#!/usr/bin/env python3
"""
Test script for NMF integration in the map analysis UI.

This script creates a simple synthetic dataset and tests the NMF functionality
without requiring real Raman map data.
"""

import sys
import numpy as np
import logging
from PySide6.QtWidgets import QApplication

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_synthetic_data():
    """Create synthetic Raman-like spectroscopy data for testing."""
    # Create wavenumbers (typical Raman range)
    wavenumbers = np.linspace(200, 3500, 500)
    
    # Create synthetic spectral components
    # Component 1: Strong peak around 1000 cm-1 (calcite-like)
    component1 = np.exp(-((wavenumbers - 1086) / 50) ** 2) * 100
    
    # Component 2: Broader feature around 1500 cm-1 (organic-like)
    component2 = np.exp(-((wavenumbers - 1500) / 100) ** 2) * 80
    
    # Component 3: Multiple peaks (mixed mineral-like)
    component3 = (np.exp(-((wavenumbers - 700) / 40) ** 2) * 60 +
                  np.exp(-((wavenumbers - 2900) / 80) ** 2) * 40)
    
    # Create spatial distributions (10x10 map)
    map_size = 10
    spectra = {}
    
    for x in range(map_size):
        for y in range(map_size):
            # Create spatial mixing coefficients
            coeff1 = np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 8)  # Top-left corner
            coeff2 = np.exp(-((x - 7) ** 2 + (y - 7) ** 2) / 8)  # Bottom-right corner
            coeff3 = np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 12)  # Center
            
            # Normalize coefficients
            total = coeff1 + coeff2 + coeff3 + 0.1
            coeff1 /= total
            coeff2 /= total
            coeff3 /= total
            
            # Create mixed spectrum with noise
            spectrum = (coeff1 * component1 + 
                       coeff2 * component2 + 
                       coeff3 * component3)
            
            # Add noise
            noise = np.random.normal(0, 5, len(wavenumbers))
            spectrum += noise
            
            # Ensure non-negative (required for NMF)
            spectrum = np.maximum(spectrum, 0)
            
            # Create spectrum object-like structure
            class MockSpectrum:
                def __init__(self, x, y, wn, intensities):
                    self.x_pos = x
                    self.y_pos = y
                    self.wavenumbers = wn
                    self.intensities = intensities
                    self.processed_intensities = intensities.copy()  # Assume no processing needed
            
            spectra[(x, y)] = MockSpectrum(x, y, wavenumbers, spectrum)
    
    return spectra

def test_nmf_analyzer():
    """Test the NMF analyzer directly."""
    print("Testing NMF Analyzer...")
    
    from map_analysis_2d.analysis.nmf_analysis import NMFAnalyzer
    
    # Create synthetic data
    spectra = create_synthetic_data()
    
    # Prepare data matrix
    data_matrix = []
    for spectrum in spectra.values():
        data_matrix.append(spectrum.intensities)
    
    X = np.array(data_matrix)
    print(f"Data matrix shape: {X.shape}")
    
    # Run NMF
    analyzer = NMFAnalyzer()
    results = analyzer.run_analysis(X, n_components=3, max_iter=100)
    
    if results['success']:
        print("✓ NMF analysis successful!")
        print(f"  Components: {results['n_components']}")
        print(f"  Reconstruction error: {results['reconstruction_error']:.4f}")
        print(f"  Iterations: {results.get('n_iterations', 'N/A')}")
        return True
    else:
        print(f"✗ NMF analysis failed: {results.get('error', 'Unknown error')}")
        return False

def test_ui_integration():
    """Test the UI integration with synthetic data."""
    print("\nTesting UI Integration...")
    
    try:
        from map_analysis_2d.ui.main_window import MapAnalysisMainWindow
        from map_analysis_2d.core import RamanMapData
        
        app = QApplication(sys.argv)
        
        # Create main window
        window = MapAnalysisMainWindow()
        
        # Create mock map data
        class MockMapData:
            def __init__(self, spectra):
                self.spectra = spectra
        
        spectra = create_synthetic_data()
        window.map_data = MockMapData(spectra)
        
        print("✓ UI window created successfully!")
        print(f"✓ Mock data loaded: {len(spectra)} spectra")
        
        # Test NMF analysis through UI
        try:
            # Switch to NMF tab
            window.tab_widget.setCurrentIndex(3)
            
            # Get control panel
            control_panel = window.get_current_nmf_control_panel()
            if control_panel:
                print("✓ NMF control panel found!")
                
                # Set parameters
                control_panel.n_components_spin.setValue(3)
                control_panel.max_iter_spin.setValue(100)
                
                print("✓ UI integration test successful!")
                return True
            else:
                print("✗ NMF control panel not found")
                return False
                
        except Exception as e:
            print(f"✗ UI integration test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Run all tests."""
    print("NMF Integration Test")
    print("===================")
    
    # Test 1: Direct NMF analyzer
    test1_passed = test_nmf_analyzer()
    
    # Test 2: UI integration (without actually running the GUI)
    test2_passed = test_ui_integration()
    
    # Summary
    print("\nTest Summary:")
    print(f"  NMF Analyzer: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  UI Integration: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! NMF integration is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 