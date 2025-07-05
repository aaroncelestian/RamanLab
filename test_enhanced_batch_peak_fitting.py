#!/usr/bin/env python3
"""
Test Enhanced Batch Peak Fitting
Verifies the new stacked tab UI design and improved ALS background functionality
"""

import numpy as np
import sys
from PySide6.QtWidgets import QApplication, QMessageBox

# Import the enhanced batch peak fitting
try:
    from batch_peak_fitting.main import launch_batch_peak_fitting
    print("✓ Successfully imported enhanced batch peak fitting module")
except ImportError as e:
    print(f"✗ Failed to import batch peak fitting: {e}")
    sys.exit(1)

def generate_test_spectrum():
    """Generate a test spectrum with multiple peaks and baseline"""
    print("Generating test spectrum...")
    
    # Create wavenumber array
    wavenumbers = np.linspace(100, 2000, 1000)
    
    # Generate synthetic baseline (exponential decay + linear)
    baseline = 50 * np.exp(-wavenumbers / 800) + 0.01 * wavenumbers + 20
    
    # Add multiple peaks
    intensities = baseline.copy()
    
    # Peak 1: Strong peak at 400 cm-1
    intensities += 200 * np.exp(-((wavenumbers - 400) / 30) ** 2)
    
    # Peak 2: Medium peak at 800 cm-1  
    intensities += 120 * np.exp(-((wavenumbers - 800) / 25) ** 2)
    
    # Peak 3: Weak peak at 1200 cm-1
    intensities += 80 * np.exp(-((wavenumbers - 1200) / 35) ** 2)
    
    # Peak 4: Sharp peak at 1600 cm-1
    intensities += 150 * np.exp(-((wavenumbers - 1600) / 20) ** 2)
    
    # Add some noise
    noise = np.random.normal(0, 5, len(wavenumbers))
    intensities += noise
    
    print(f"✓ Generated test spectrum: {len(wavenumbers)} points")
    print(f"  Wavenumber range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
    print(f"  Intensity range: {np.min(intensities):.1f} - {np.max(intensities):.1f}")
    print(f"  Expected peaks at: 400, 800, 1200, 1600 cm⁻¹")
    
    return wavenumbers, intensities

def test_ui_features():
    """Test the new UI features"""
    print("\n=== Testing Enhanced UI Features ===")
    
    print("✓ New features to test:")
    print("  • Stacked tab design (Background, Peak Detection, Peak Fitting, Analysis, Results)")
    print("  • Live ALS background preview with sliders")
    print("  • Interactive peak selection mode")
    print("  • Enhanced parameter controls")
    print("  • Real-time parameter updates")

def test_als_parameters():
    """Test different ALS parameter combinations"""
    print("\n=== ALS Parameter Test Combinations ===")
    
    test_params = [
        {"lambda": 1e4, "p": 0.05, "niter": 15, "description": "Aggressive (follows peaks closely)"},
        {"lambda": 1e5, "p": 0.01, "niter": 10, "description": "Moderate (balanced)"},
        {"lambda": 1e6, "p": 0.001, "niter": 10, "description": "Conservative (smooth baseline)"},
        {"lambda": 1e7, "p": 0.002, "niter": 20, "description": "Ultra-smooth (very conservative)"},
    ]
    
    print("Recommended ALS parameters to test:")
    for i, params in enumerate(test_params, 1):
        print(f"  {i}. {params['description']}")
        print(f"     Lambda: {params['lambda']:.0e}, P: {params['p']:.3f}, Iterations: {params['niter']}")
    
    return test_params

def main():
    """Main test function"""
    print("Enhanced Batch Peak Fitting Test")
    print("=" * 40)
    
    # Generate test data
    wavenumbers, intensities = generate_test_spectrum()
    
    # Test UI features info
    test_ui_features()
    
    # Test ALS parameters info
    test_als_parameters()
    
    print("\n=== Launching Enhanced Batch Peak Fitting ===")
    print("Instructions for testing:")
    print("1. Check that the new stacked tab interface is visible")
    print("2. Go to 'Background' tab and test ALS live preview:")
    print("   - Move Lambda slider (smoothness): try values 1e4 to 1e7")
    print("   - Move P slider (asymmetry): try values 0.001 to 0.050")
    print("   - Move Iterations slider: try values 5 to 30")
    print("   - Background should update in real-time!")
    print("3. Go to 'Peak Detection' tab and test live peak detection:")
    print("   - Adjust Height, Distance, and Prominence sliders")
    print("   - Peaks should update in real-time!")
    print("4. Test interactive peak selection mode")
    print("5. Go to 'Peak Fitting' tab and fit the detected peaks")
    print("6. Check 'Analysis' and 'Results' tabs for output")
    
    # Check for existing QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app_created = True
    else:
        app_created = False
    
    try:
        # Launch the enhanced application
        controller = launch_batch_peak_fitting(
            parent=None,
            wavenumbers=wavenumbers, 
            intensities=intensities
        )
        
        # Show success message
        if controller:
            print("✓ Enhanced batch peak fitting launched successfully!")
            print("✓ Test spectrum loaded with 4 synthetic peaks")
            print("✓ Ready for UI and ALS testing")
            
            # Show instructions dialog
            QMessageBox.information(
                None,
                "Enhanced Batch Peak Fitting Test",
                "Test spectrum loaded successfully!\n\n"
                "Features to test:\n"
                "• New stacked tab interface\n" 
                "• Live ALS background preview\n"
                "• Real-time peak detection\n"
                "• Interactive peak selection\n\n"
                "Check the console for detailed testing instructions."
            )
        
    except Exception as e:
        print(f"✗ Error launching application: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n=== Test completed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 