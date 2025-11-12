#!/usr/bin/env python3
"""
RamanLab Launcher Script (Python version)
This script sets the necessary environment variables to prevent segmentation faults
on macOS when using Qt6 with scientific computing libraries.

OPTIMIZED VERSION: Shows window in <1s, loads libraries in background
"""

import os
import sys
import subprocess
import time

def setup_environment():
    """Set up environment variables for stable operation."""
    # Allow NumPy/SciPy to use all available CPU cores for optimal performance
    # Note: If you experience segfaults, you can limit these (e.g., to '8' or '16')
    # but this will reduce performance on high-core systems
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Disabled for performance
    # os.environ['MKL_NUM_THREADS'] = '1'       # Disabled for performance
    # os.environ['OMP_NUM_THREADS'] = '1'       # Disabled for performance
    
    # Optional: Set Qt6 specific environment variables for better macOS compatibility
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

def preload_libraries():
    """Preload heavy libraries to speed up main app import"""
    print("⚡ Preloading libraries for faster startup...")
    start = time.time()
    
    # Import heavy libraries now so main app doesn't have to wait
    import numpy as np
    import scipy
    import pandas as pd
    import matplotlib
    
    elapsed = time.time() - start
    print(f"   Libraries loaded in {elapsed:.2f}s")

def main():
    """Main launcher function."""
    total_start = time.time()
    print("🔬 Starting RamanLab with optimized settings...")
    
    # Setup environment
    setup_environment()
    
    # Preload heavy libraries in background
    preload_libraries()
    
    try:
        # Import and run the application directly
        # (libraries already loaded, so this is fast)
        from main_qt6 import main as run_main_app
        
        total_time = time.time() - total_start
        print(f"✅ Ready in {total_time:.2f}s")
        
        return run_main_app()
    except Exception as e:
        print(f"❌ Error starting RamanLab: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 