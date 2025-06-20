#!/usr/bin/env python3
"""
Demo script for 3D Visualization Tab

This script demonstrates the new 3D visualization features:
- User selectable 3D tensor shapes 
- Optical axis vectors based on crystal symmetry
- Crystal shapes with optimized orientation
- Crystal structure visualization
"""

import sys
import os
import numpy as np

# Add the current directory to Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_3d_visualization():
    """Demonstrate the 3D visualization capabilities."""
    print("=== 3D Visualization Demo ===")
    print()
    
    try:
        # Use PySide6 as the official Qt for Python binding
        from PySide6.QtWidgets import QApplication
        
        from raman_polarization_analyzer_qt6 import RamanPolarizationAnalyzerQt6
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create main analyzer window
        analyzer = RamanPolarizationAnalyzerQt6()
        
        # Create some test tensor data
        print("Creating test tensor data...")
        test_tensors = create_test_tensor_data()
        analyzer.calculated_raman_tensors = test_tensors
        analyzer.current_crystal_system = "Tetragonal"
        
        # Show the window and navigate to 3D visualization tab
        analyzer.show()
        
        # Find and activate the 3D visualization tab
        for i in range(analyzer.main_tabs.count()):
            if "3D Visualization" in analyzer.main_tabs.tabText(i):
                analyzer.main_tabs.setCurrentIndex(i)
                print(f"✓ Activated 3D Visualization tab (index {i})")
                break
        
        # Update the 3D widget if it exists
        if hasattr(analyzer, 'visualization_3d_widget'):
            analyzer.visualization_3d_widget.refresh_data()
            
            # Force tensor checkbox to be checked
            analyzer.visualization_3d_widget.show_tensor_cb.setChecked(True)
            
            # Force selection of first tensor if available
            if analyzer.visualization_3d_widget.tensor_combo.count() > 1:
                analyzer.visualization_3d_widget.tensor_combo.setCurrentIndex(1)
                analyzer.visualization_3d_widget.on_tensor_selection_changed(
                    analyzer.visualization_3d_widget.tensor_combo.currentText()
                )
            
            print("✓ Refreshed 3D visualization data")
        elif hasattr(analyzer, 'fallback_tensor_combo'):
            analyzer.update_fallback_tensor_combo()
            analyzer.render_basic_3d_visualization()
            print("✓ Using fallback 3D visualization")
        
        print()
        print("3D Visualization Features:")
        print("- ✓ User selectable 3D tensor shapes")
        print("- ✓ Laser beam direction (Z-axis, gold arrow)")
        print("- ✓ Optical axis vectors based on crystal symmetry")
        print("- ✓ Crystal shape visualization")
        print("- ✓ Interactive orientation controls")
        print("- ✓ Export capabilities")
        print()
        print("Demo window is now open. Close it to exit.")
        
        # Run the application
        return app.exec()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install PySide6 matplotlib numpy")
        return 1
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return 1

def create_test_tensor_data():
    """Create test tensor data for demonstration."""
    tensors = {}
    
    # A1g mode (isotropic-like)
    tensor_a1g = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.2]
    ])
    
    tensors[1085.0] = {
        'tensor': tensor_a1g,
        'character': 'A1g',
        'properties': {
            'eigenvalues': np.linalg.eigvals(tensor_a1g),
            'trace': np.trace(tensor_a1g),
            'determinant': np.linalg.det(tensor_a1g),
            'anisotropy': 0.2,
            'tensor_norm': np.linalg.norm(tensor_a1g),
            'spherical_part': np.trace(tensor_a1g) / 3,
            'deviatoric_norm': np.linalg.norm(tensor_a1g - np.eye(3) * np.trace(tensor_a1g) / 3)
        }
    }
    
    # B1g mode (4-lobed pattern)
    tensor_b1g = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    tensors[515.0] = {
        'tensor': tensor_b1g,
        'character': 'B1g',
        'properties': {
            'eigenvalues': np.linalg.eigvals(tensor_b1g),
            'trace': np.trace(tensor_b1g),
            'determinant': np.linalg.det(tensor_b1g),
            'anisotropy': 1.0,
            'tensor_norm': np.linalg.norm(tensor_b1g),
            'spherical_part': np.trace(tensor_b1g) / 3,
            'deviatoric_norm': np.linalg.norm(tensor_b1g - np.eye(3) * np.trace(tensor_b1g) / 3)
        }
    }
    
    # Eg mode (degenerate)
    tensor_eg = np.array([
        [0.8, 0.3, 0.0],
        [0.3, 0.8, 0.0],
        [0.0, 0.0, 0.4]
    ])
    
    tensors[325.0] = {
        'tensor': tensor_eg,
        'character': 'Eg',
        'properties': {
            'eigenvalues': np.linalg.eigvals(tensor_eg),
            'trace': np.trace(tensor_eg),
            'determinant': np.linalg.det(tensor_eg),
            'anisotropy': 0.4,
            'tensor_norm': np.linalg.norm(tensor_eg),
            'spherical_part': np.trace(tensor_eg) / 3,
            'deviatoric_norm': np.linalg.norm(tensor_eg - np.eye(3) * np.trace(tensor_eg) / 3)
        }
    }
    
    return tensors

def print_features():
    """Print the key features of the 3D visualization system."""
    print("🔬 3D Raman Polarization Visualization Features:")
    print()
    print("📊 Tensor Visualization:")
    print("   • User-selectable 3D tensor shapes")
    print("   • Angular dependence surfaces")
    print("   • B1g/B2g mode node structure (red/blue lobes)")
    print("   • Real-time orientation adjustment")
    print()
    print("🔬 Laser Configuration:")
    print("   • Laser beam direction indicator (Z-axis)")
    print("   • Adjustable laser direction sliders")
    print("   • Relative orientation to crystal")
    print()
    print("🔍 Crystal Properties:")
    print("   • Optical axis vectors (uniaxial/biaxial)")
    print("   • Crystal shape based on symmetry")
    print("   • Cubic, tetragonal, hexagonal shapes")
    print("   • Optimized crystal structure display")
    print()
    print("⚙️ Interactive Controls:")
    print("   • φ, θ, ψ orientation sliders")
    print("   • Use optimized orientation button")
    print("   • Real-time visualization updates")
    print("   • Export capabilities (PNG, PDF, SVG)")
    print()
    print("🎯 Crystal Systems Supported:")
    print("   • Cubic (isotropic)")
    print("   • Tetragonal (uniaxial)")
    print("   • Hexagonal (uniaxial)")
    print("   • Orthorhombic (biaxial)")
    print("   • Monoclinic (biaxial)")
    print("   • Triclinic (biaxial)")

if __name__ == "__main__":
    print_features()
    print()
    print("Starting 3D Visualization Demo...")
    print("=" * 50)
    sys.exit(demo_3d_visualization()) 