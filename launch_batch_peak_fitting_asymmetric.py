#!/usr/bin/env python3
"""
Launch Batch Peak Fitting with Asymmetric Voigt Support
========================================================

This launcher demonstrates how to use the batch peak fitting module
with the new asymmetric Voigt peak model for advanced peak analysis.

Features now available:
- Asymmetric Voigt model for peaks with tailing
- All existing models: Gaussian, Lorentzian, Pseudo-Voigt, Voigt
- Proper parameter handling for variable model complexity
- Compatible with trends analysis and geothermometry
"""

import sys
import os
from PySide6.QtWidgets import QApplication

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the batch peak fitting application"""
    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Import the batch peak fitting launcher
        from batch_peak_fitting.main import launch_batch_peak_fitting
        
        print("🔬 Launching Batch Peak Fitting with Asymmetric Voigt Support")
        print("=" * 60)
        print("Available Peak Models:")
        print("  • Gaussian - symmetric peaks")
        print("  • Lorentzian - symmetric peaks with broader tails")
        print("  • Pseudo-Voigt - mix of Gaussian and Lorentzian")
        print("  • Voigt - convolution of Gaussian and Lorentzian")
        print("  • Asymmetric Voigt - asymmetric peaks with tailing 🆕")
        print()
        print("Key Features:")
        print("  ✓ Interactive peak detection and fitting")
        print("  ✓ ALS baseline correction")
        print("  ✓ Batch processing across multiple files")
        print("  ✓ Trends analysis with parameter tracking")
        print("  ✓ Geothermometry analysis support")
        print("  ✓ Export results to CSV")
        print("=" * 60)
        
        # Launch the application
        dialog = launch_batch_peak_fitting()
        
        if dialog:
            dialog.show()
            app.exec()
        else:
            print("Failed to launch batch peak fitting")
            
    except ImportError as e:
        print(f"Error importing batch peak fitting: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 