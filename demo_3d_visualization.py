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

# Import safe file handling
from pkl_utils import get_workspace_root, get_example_data_paths, print_available_example_files
from pathlib import Path

# Add the current directory to Python path so we can import modules
workspace_root = get_workspace_root()
sys.path.insert(0, str(workspace_root))

def setup_safe_environment():
    """Set up a safe environment for the demo."""
    print("🔧 Setting up Safe Environment")
    print("=" * 50)
    
    # Get workspace paths
    paths = get_example_data_paths()
    
    print(f"📁 Workspace Root: {paths['workspace_root']}")
    print(f"📁 Example Data: {paths['example_data']}")
    print(f"📁 Test Data: {paths['test_data']}")
    
    # Check for required modules
    required_modules = [
        'raman_polarization_analyzer_qt6',
        'polarization_ui'
    ]
    
    for module in required_modules:
        module_path = paths['workspace_root'] / module
        if module_path.exists():
            print(f"✅ Found module: {module}")
        else:
            print(f"❌ Missing module: {module}")
    
    return paths

def demo_3d_visualization():
    """Demonstrate the 3D visualization capabilities."""
    print("🔬 3D Visualization Demo")
    print("=" * 50)
    
    # Set up safe environment
    paths = setup_safe_environment()
    
    # Show available example files
    print("\n📄 Available Example Files:")
    print_available_example_files()
    
    print("\n📊 Starting 3D Visualization...")
    
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
        
        print("\n🎯 3D Visualization Features:")
        print("   • User selectable 3D tensor shapes")
        print("   • Laser beam direction (Z-axis, gold arrow)")
        print("   • Optical axis vectors based on crystal symmetry")
        print("   • Crystal shape visualization")
        print("   • Interactive orientation controls")
        print("   • Export capabilities")
        
        # Show workspace information
        print(f"\n📁 Workspace Information:")
        print(f"   • Root: {paths['workspace_root']}")
        print(f"   • Available data directories: {len([d for d in paths.values() if isinstance(d, Path) and d.is_dir()])}")
        
        print("\n🚀 Demo window is now open. Close it to exit.")
        
        # Run the application
        return app.exec()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install PySide6 matplotlib numpy")
        print(f"  Make sure you're running from: {paths['workspace_root']}")
        return 1
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
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

def create_safe_demo_data():
    """Create demonstration data and save to workspace."""
    print("\n💾 Creating Demo Data")
    print("=" * 30)
    
    # Get workspace paths
    paths = get_example_data_paths()
    
    # Create demo data directory
    demo_dir = paths['workspace_root'] / "demo_data"
    demo_dir.mkdir(exist_ok=True)
    
    # Create tensor data
    tensors = create_test_tensor_data()
    
    # Save tensor data
    import pickle
    tensor_file = demo_dir / "demo_tensors.pkl"
    with open(tensor_file, 'wb') as f:
        pickle.dump(tensors, f)
    
    print(f"✅ Demo tensors saved to: {tensor_file}")
    
    # Create demo spectrum data
    wavenumbers = np.linspace(100, 1600, 800)
    intensities = np.random.exponential(0.1, len(wavenumbers))
    
    # Add some peaks
    peaks = [464, 515, 1085, 325]
    for peak in peaks:
        if peak >= wavenumbers.min() and peak <= wavenumbers.max():
            peak_idx = np.argmin(np.abs(wavenumbers - peak))
            intensities[peak_idx-5:peak_idx+5] += np.random.exponential(1.0, 10)
    
    # Save spectrum data
    spectrum_file = demo_dir / "demo_spectrum.txt"
    with open(spectrum_file, 'w') as f:
        f.write("# Demo spectrum for 3D visualization\n")
        f.write("# Wavenumber (cm-1)\tIntensity\n")
        for wn, intensity in zip(wavenumbers, intensities):
            f.write(f"{wn:.2f}\t{intensity:.6f}\n")
    
    print(f"✅ Demo spectrum saved to: {spectrum_file}")
    
    return {
        'tensor_file': tensor_file,
        'spectrum_file': spectrum_file,
        'demo_dir': demo_dir
    }

def print_features():
    """Print the key features of the 3D visualization system."""
    print("🔬 3D Raman Polarization Visualization Features:")
    print("=" * 50)
    
    print("\n📊 Tensor Visualization:")
    print("   • User-selectable 3D tensor shapes")
    print("   • Angular dependence surfaces")
    print("   • B1g/B2g mode node structure (red/blue lobes)")
    print("   • Real-time orientation adjustment")
    
    print("\n🔬 Laser Configuration:")
    print("   • Laser beam direction indicator (Z-axis)")
    print("   • Adjustable laser direction sliders")
    print("   • Relative orientation to crystal")
    
    print("\n🔍 Crystal Properties:")
    print("   • Optical axis vectors (uniaxial/biaxial)")
    print("   • Crystal shape based on symmetry")
    print("   • Cubic, tetragonal, hexagonal shapes")
    print("   • Optimized crystal structure display")
    
    print("\n⚙️ Interactive Controls:")
    print("   • φ, θ, ψ orientation sliders")
    print("   • Use optimized orientation button")
    print("   • Real-time visualization updates")
    print("   • Export capabilities (PNG, PDF, SVG)")
    
    print("\n🎯 Crystal Systems Supported:")
    print("   • Cubic (isotropic)")
    print("   • Tetragonal (uniaxial)")
    print("   • Hexagonal (uniaxial)")
    print("   • Orthorhombic (biaxial)")
    print("   • Monoclinic (biaxial)")
    print("   • Triclinic (biaxial)")
    
    print("\n💾 Safe File Handling:")
    print("   • Automatic workspace detection")
    print("   • Cross-platform compatibility")
    print("   • Safe path resolution")
    print("   • Example data integration")

def demonstrate_safe_workflow():
    """Demonstrate the complete safe workflow."""
    print("\n🔄 Safe Workflow Demonstration")
    print("=" * 50)
    
    # Create demo data
    demo_data = create_safe_demo_data()
    
    # Show available resources
    paths = get_example_data_paths()
    
    print("\n📁 Available Resources:")
    print(f"   • Workspace root: {paths['workspace_root']}")
    print(f"   • Demo data directory: {demo_data['demo_dir']}")
    print(f"   • Tensor data file: {demo_data['tensor_file']}")
    print(f"   • Spectrum data file: {demo_data['spectrum_file']}")
    
    # Check if we can load the data back
    print("\n🔄 Data Integrity Check:")
    try:
        import pickle
        with open(demo_data['tensor_file'], 'rb') as f:
            loaded_tensors = pickle.load(f)
        print(f"✅ Loaded {len(loaded_tensors)} tensor entries")
        
        with open(demo_data['spectrum_file'], 'r') as f:
            lines = f.readlines()
        data_lines = [line for line in lines if not line.startswith('#')]
        print(f"✅ Loaded {len(data_lines)} spectrum data points")
        
    except Exception as e:
        print(f"❌ Data integrity check failed: {e}")
    
    print("\n🎯 Workflow Benefits:")
    print("   • Automatic workspace detection")
    print("   • Safe file path handling")
    print("   • Cross-platform compatibility")
    print("   • Persistent demo data")
    print("   • Error handling and recovery")

if __name__ == "__main__":
    print("🚀 Starting 3D Visualization Demo")
    print("=" * 50)
    
    # Print features first
    print_features()
    
    # Demonstrate safe workflow
    demonstrate_safe_workflow()
    
    # Ask user if they want to run the GUI demo
    print("\n🎮 GUI Demo Options:")
    print("   1. Run interactive 3D visualization (GUI)")
    print("   2. Exit and review safe workflow")
    
    try:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == '1':
            print("\n🎯 Starting GUI Demo...")
            exit_code = demo_3d_visualization()
            print(f"\n✅ GUI Demo completed with exit code: {exit_code}")
        else:
            print("\n📋 Safe workflow demonstration completed.")
            print("   All demo data has been created in the workspace.")
            print("   Run this script again and choose option 1 to see the GUI.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n�� Demo finished.") 