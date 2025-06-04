#!/usr/bin/env python3

import tkinter as tk
import raman_polarization_analyzer as rpa

def test_data_availability():
    """Test what data is available in the application."""
    print("=== Testing Data Availability ===")
    
    # Create application instance
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    try:
        app = rpa.RamanPolarizationAnalyzer(root)
        
        # Check what data is available
        print(f"Has raman_tensors: {hasattr(app, 'raman_tensors')}")
        if hasattr(app, 'raman_tensors'):
            print(f"  raman_tensors is None: {app.raman_tensors is None}")
            print(f"  raman_tensors bool value: {bool(app.raman_tensors)}")
            print(f"  Type: {type(app.raman_tensors)}")
            if app.raman_tensors is not None:
                if isinstance(app.raman_tensors, dict):
                    print(f"  Keys: {list(app.raman_tensors.keys())}")
                    for key, value in app.raman_tensors.items():
                        if hasattr(value, '__len__'):
                            print(f"    {key}: {type(value)} with length {len(value)}")
                        else:
                            print(f"    {key}: {type(value)}")
                elif hasattr(app.raman_tensors, '__len__'):
                    print(f"  Length: {len(app.raman_tensors)}")
                else:
                    print(f"  Value: {app.raman_tensors}")
        else:
            print("  raman_tensors attribute does not exist")
        
        print(f"Has current_spectrum: {hasattr(app, 'current_spectrum') and app.current_spectrum is not None}")
        print(f"Has original_spectrum: {hasattr(app, 'original_spectrum') and app.original_spectrum is not None}")
        print(f"Has wavenumbers: {hasattr(app, 'wavenumbers') and app.wavenumbers is not None}")
        print(f"Has intensities: {hasattr(app, 'intensities') and app.intensities is not None}")
        print(f"Has tensor_data_3d: {hasattr(app, 'tensor_data_3d') and app.tensor_data_3d is not None}")
        
        # Test loading demo data
        print("\n=== Testing Demo Data Loading ===")
        try:
            app.load_demo_data_3d()
            print("✓ Demo data loading completed")
            
            # Check what was loaded
            print(f"  raman_tensors after demo: {bool(app.raman_tensors)}")
            print(f"  tensor_data_3d after demo: {app.tensor_data_3d is not None}")
            print(f"  current_spectrum after demo: {app.current_spectrum is not None}")
            
            if app.tensor_data_3d:
                print(f"  tensor_data_3d keys: {list(app.tensor_data_3d.keys())}")
                if 'wavenumbers' in app.tensor_data_3d:
                    print(f"  Number of peaks: {len(app.tensor_data_3d['wavenumbers'])}")
                    
        except Exception as e:
            print(f"✗ Demo data loading failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test tensor import
        print("\n=== Testing Tensor Import ===")
        if hasattr(app, 'raman_tensors'):
            print(f"raman_tensors exists, checking if it's usable...")
            print(f"  Is None: {app.raman_tensors is None}")
            print(f"  Bool value: {bool(app.raman_tensors)}")
            
            if app.raman_tensors is not None and bool(app.raman_tensors):
                try:
                    app.tensor_data_3d = app.raman_tensors.copy()
                    print("✓ Tensor import successful")
                    print(f"  tensor_data_3d type: {type(app.tensor_data_3d)}")
                    if isinstance(app.tensor_data_3d, dict):
                        print(f"  tensor_data_3d keys: {list(app.tensor_data_3d.keys())}")
                except Exception as e:
                    print(f"✗ Tensor import failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("✗ raman_tensors is None or empty")
        else:
            print("✗ No raman_tensors attribute found")
        
        # Test spectrum calculation
        print("\n=== Testing Spectrum Calculation ===")
        if hasattr(app, 'tensor_data_3d') and app.tensor_data_3d:
            try:
                # Check if tensor data has required keys
                if 'tensors' in app.tensor_data_3d and 'wavenumbers' in app.tensor_data_3d:
                    tensors = app.tensor_data_3d['tensors']
                    wavenumbers = app.tensor_data_3d['wavenumbers']
                    print(f"✓ Found {len(tensors)} tensors and {len(wavenumbers)} wavenumbers")
                    
                    # Test a simple calculation
                    import numpy as np
                    if len(tensors) > 0:
                        tensor = tensors[0]
                        print(f"  First tensor shape: {tensor.shape if hasattr(tensor, 'shape') else 'No shape'}")
                        print(f"  First tensor type: {type(tensor)}")
                        
                        # Test orientation matrix
                        orientation_matrix = np.eye(3)
                        rotated_tensor = np.dot(orientation_matrix, np.dot(tensor, orientation_matrix.T))
                        print(f"✓ Tensor rotation test successful")
                        
                        # Test intensity calculation
                        pol_in = np.array([1, 0, 0])
                        pol_out = np.array([1, 0, 0])
                        intensity = np.abs(np.dot(pol_out, np.dot(rotated_tensor, pol_in)))**2
                        print(f"✓ Intensity calculation successful: {intensity}")
                        
                else:
                    print(f"✗ Missing required keys in tensor_data_3d")
                    if isinstance(app.tensor_data_3d, dict):
                        print(f"  Available keys: {list(app.tensor_data_3d.keys())}")
            except Exception as e:
                print(f"✗ Spectrum calculation test failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ No tensor_data_3d available for spectrum calculation")
        
    finally:
        root.destroy()

if __name__ == "__main__":
    test_data_availability() 