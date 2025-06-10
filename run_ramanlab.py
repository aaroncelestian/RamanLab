#!/usr/bin/env python3
"""
RamanLab Launcher Script (Python version)
This script sets the necessary environment variables to prevent segmentation faults
on macOS when using Qt6 with scientific computing libraries.
"""

import os
import sys
import subprocess

def setup_environment():
    """Set up environment variables for stable operation."""
    # Set threading environment variables to prevent conflicts
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Optional: Set Qt6 specific environment variables for better macOS compatibility
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

def main():
    """Main launcher function."""
    print("Starting RamanLab with optimized settings...")
    
    # Setup environment
    setup_environment()
    
    try:
        # Import and run the application directly
        from main_qt6 import main as run_main_app
        return run_main_app()
    except Exception as e:
        print(f"Error starting RamanLab: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 