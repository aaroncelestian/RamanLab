#!/usr/bin/env python3
"""
Launcher for Standalone Crystal Orientation Optimizer
=====================================================

This script launches the standalone Crystal Orientation Optimization widget
that can be used independently of the main RamanLab application.

Features:
- Complete trilogy optimization implementation
- Modern Qt6/Qt5 compatible interface
- Real-time visualization
- Comprehensive results analysis
- Export functionality

Usage:
    python launch_orientation_optimizer.py
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from polarization_ui.orientation_optimizer_widget import OrientationOptimizerWidget, main
    
    if __name__ == "__main__":
        print("🎯 Launching Crystal Orientation Optimizer...")
        print("=" * 50)
        print("Features:")
        print("• Stage 1: Enhanced Individual Peak Optimization")
        print("• Stage 2: Probabilistic Bayesian Framework") 
        print("• Stage 3: Advanced Multi-Objective Optimization")
        print("• Real-time visualization and progress tracking")
        print("• Comprehensive uncertainty quantification")
        print("• Export results for 3D visualization")
        print("=" * 50)
        
        # Run the main application
        main()
        
except ImportError as e:
    print(f"❌ Error importing orientation optimizer: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're in the RamanLab directory")
    print("2. Check that polarization_ui/orientation_optimizer_widget.py exists")
    print("3. Install required dependencies:")
    print("   pip install PySide6 matplotlib numpy scipy")
    print("   pip install scikit-learn emcee  # optional for advanced features")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1) 