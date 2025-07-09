#!/usr/bin/env python3
"""
Launch script for RamanLab Mixture Analysis GUI
Enhanced spectral mixture analysis with iterative decomposition
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the mixture analysis GUI."""
    print("🔬 RamanLab - Mixture Analysis")
    print("=" * 40)
    
    try:
        from raman_mixture_analysis_qt6 import main as run_gui
        return run_gui()
    except ImportError as e:
        print(f"Error importing mixture analysis GUI: {e}")
        print("Make sure you have PySide6 installed: pip install PySide6")
        print("Also ensure all dependencies are installed: pip install -r requirements_qt6.txt")
        return 1
    except Exception as e:
        print(f"Error running mixture analysis GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 