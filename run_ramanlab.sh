#!/bin/bash

# RamanLab Launcher Script
# This script sets the necessary environment variables to prevent segmentation faults
# on macOS when using Qt6 with scientific computing libraries

echo "Starting RamanLab with optimized settings..."

# Set threading environment variables to prevent conflicts
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1  
export OMP_NUM_THREADS=1

# Optional: Set Qt6 specific environment variables for better macOS compatibility
export QT_MAC_WANTS_LAYER=1

# Run the application
python main_qt6.py

echo "RamanLab has closed." 