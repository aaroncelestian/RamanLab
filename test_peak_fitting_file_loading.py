#!/usr/bin/env python3
"""
Test script for the enhanced Peak Fitting Qt6 app with file loading capability.

This script demonstrates how to:
1. Launch the peak fitting app as a standalone application
2. Load spectrum files directly from the app
3. Use the enhanced file operations

Usage:
    python test_peak_fitting_file_loading.py
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_standalone_launch():
    """Test launching the peak fitting app as standalone with file loading."""
    print("Testing standalone Peak Fitting Qt6 app with file loading...")
    print("Features to test:")
    print("1. File → Open Spectrum File (Ctrl+O)")
    print("2. Load various file formats (.txt, .csv, .dat, .asc)")
    print("3. Status bar showing file information")
    print("4. Window title updates with file name")
    print("5. Analysis menu with quick actions")
    print("6. Export results functionality")
    print("\nLaunching app...")
    
    try:
        from peak_fitting_qt6 import launch_standalone_spectral_deconvolution
        launch_standalone_spectral_deconvolution()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the RamanLab directory")
    except Exception as e:
        print(f"Error launching app: {e}")

def test_with_existing_data():
    """Test launching with existing spectrum data."""
    print("Testing with pre-loaded spectrum data...")
    
    try:
        import numpy as np
        from peak_fitting_qt6 import launch_spectral_deconvolution
        from PySide6.QtWidgets import QApplication, QWidget
        
        # Create sample data
        wavenumbers = np.linspace(200, 4000, 1000)
        intensities = (
            100 * np.exp(-((wavenumbers - 1000) / 50) ** 2) +  # Peak at 1000
            80 * np.exp(-((wavenumbers - 1500) / 40) ** 2) +   # Peak at 1500
            60 * np.exp(-((wavenumbers - 2000) / 60) ** 2) +   # Peak at 2000
            20 + 0.1 * np.random.normal(0, 1, len(wavenumbers))  # Baseline + noise
        )
        
        app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
        parent = QWidget()
        parent.hide()
        
        print("Launching with sample spectrum data...")
        print("You can still use File → Open to load a different file")
        
        launch_spectral_deconvolution(parent, wavenumbers, intensities)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PySide6 and numpy are installed")
    except Exception as e:
        print(f"Error: {e}")

def test_file_formats():
    """Show information about supported file formats."""
    print("\nSupported file formats:")
    print("=" * 40)
    
    try:
        from utils.file_loaders import SpectrumLoader
        loader = SpectrumLoader()
        extensions = loader.get_supported_extensions()
        
        print("File extensions:")
        for ext in extensions:
            print(f"  • {ext}")
        
        print("\nFile format requirements:")
        print("• Two-column format: wavenumber, intensity")
        print("• Delimiters: tab, comma, space, or semicolon")
        print("• Headers are automatically detected and skipped")
        print("• Comments starting with #, %, or ; are ignored")
        print("• Data points should be numeric")
        
    except ImportError:
        print("File loading utilities not available")
        print("Install required dependencies for file loading")

def main():
    """Main test function."""
    print("Peak Fitting Qt6 File Loading Test")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nChoose test mode:")
        print("1. Standalone app (no data)")
        print("2. App with sample data")
        print("3. Show file format info")
        print("4. All tests")
        
        choice = input("\nEnter choice (1-4): ").strip()
        modes = {'1': 'standalone', '2': 'sample', '3': 'formats', '4': 'all'}
        mode = modes.get(choice, 'standalone')
    
    if mode == 'standalone':
        test_standalone_launch()
    elif mode == 'sample':
        test_with_existing_data()
    elif mode == 'formats':
        test_file_formats()
    elif mode == 'all':
        test_file_formats()
        print("\n" + "=" * 50)
        print("Choose which app to launch:")
        print("1. Standalone (no data)")
        print("2. With sample data")
        choice = input("Enter choice (1-2): ").strip()
        if choice == '2':
            test_with_existing_data()
        else:
            test_standalone_launch()

if __name__ == "__main__":
    main() 