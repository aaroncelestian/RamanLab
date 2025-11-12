#!/bin/bash

# RamanLab Launcher Script
# This script sets the necessary environment variables to prevent segmentation faults
# on macOS when using Qt6 with scientific computing libraries

echo "Starting RamanLab with optimized settings..."

# Allow NumPy/SciPy to use all available CPU cores for optimal performance
# Note: If you experience segfaults, you can limit these (e.g., to 8 or 16)
# but this will reduce performance on high-core systems
# export OPENBLAS_NUM_THREADS=1  # Disabled for performance
# export MKL_NUM_THREADS=1       # Disabled for performance
# export OMP_NUM_THREADS=1       # Disabled for performance

# Optional: Set Qt6 specific environment variables for better macOS compatibility
export QT_MAC_WANTS_LAYER=1

# Run the application
python main_qt6.py

echo "RamanLab has closed." 