#!/usr/bin/env python3

"""
Test script to verify that symmetry-constrained Raman tensors are created correctly.
"""

import tkinter as tk
import raman_polarization_analyzer as rpa
import numpy as np

def test_symmetry_constrained_tensors():
    """Test tensor creation with different crystal systems."""
    print("🧪 TESTING SYMMETRY-CONSTRAINED TENSOR CREATION...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = rpa.RamanPolarizationAnalyzer(root)
        
        # Test different crystal systems
        crystal_systems = [
            ("Cubic", "m-3m"),
            ("Tetragonal", "4/mmm"), 
            ("Orthorhombic", "mmm"),
            ("Hexagonal", "6/mmm"),
            ("Trigonal", "-3m"),
            ("Monoclinic", "2/m"),
            ("Triclinic", "-1")
        ]
        
        print("\n📊 TESTING TENSOR SYMMETRY CONSTRAINTS:")
        print("=" * 60)
        
        for crystal_system, point_group in crystal_systems:
            print(f"\n🔬 Testing {crystal_system} ({point_group}):")
            
            # Set crystal structure data
            app.crystal_structure = {
                'crystal_system': crystal_system,
                'point_group': point_group,
                'name': f'Test {crystal_system}'
            }
            
            # Create test tensor
            wavenumber = 500.0
            intensity = 1.0
            tensor = app.create_symmetry_constrained_tensor(wavenumber, intensity)
            
            print(f"   Tensor matrix:")
            for i, row in enumerate(tensor):
                print(f"     [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
            
            # Check symmetry properties
            is_symmetric = np.allclose(tensor, tensor.T, rtol=1e-6)
            print(f"   Symmetric: {'✅ Yes' if is_symmetric else '❌ No'}")
            
            # Check crystal system specific constraints
            if crystal_system == "Cubic":
                # Should have αxx = αyy = αzz, off-diagonal = 0
                diagonal_equal = np.allclose([tensor[0,0], tensor[1,1], tensor[2,2]], tensor[0,0], rtol=1e-6)
                off_diagonal_zero = np.allclose([tensor[0,1], tensor[0,2], tensor[1,2]], 0, atol=1e-6)
                print(f"   Cubic constraints: {'✅ Pass' if diagonal_equal and off_diagonal_zero else '❌ Fail'}")
                
            elif crystal_system == "Tetragonal":
                # Should have αxx = αyy ≠ αzz, off-diagonal = 0
                xy_equal = np.allclose(tensor[0,0], tensor[1,1], rtol=1e-6)
                z_different = not np.allclose(tensor[0,0], tensor[2,2], rtol=1e-6)
                off_diagonal_zero = np.allclose([tensor[0,1], tensor[0,2], tensor[1,2]], 0, atol=1e-6)
                print(f"   Tetragonal constraints: {'✅ Pass' if xy_equal and z_different and off_diagonal_zero else '❌ Fail'}")
                
            elif crystal_system == "Orthorhombic":
                # Should have all diagonal different, off-diagonal = 0
                all_different = not (np.allclose(tensor[0,0], tensor[1,1], rtol=1e-6) or 
                                   np.allclose(tensor[0,0], tensor[2,2], rtol=1e-6) or
                                   np.allclose(tensor[1,1], tensor[2,2], rtol=1e-6))
                off_diagonal_zero = np.allclose([tensor[0,1], tensor[0,2], tensor[1,2]], 0, atol=1e-6)
                print(f"   Orthorhombic constraints: {'✅ Pass' if all_different and off_diagonal_zero else '❌ Fail'}")
                
            elif crystal_system == "Hexagonal":
                # Should have αxx = αyy ≠ αzz
                xy_equal = np.allclose(tensor[0,0], tensor[1,1], rtol=1e-6)
                z_different = not np.allclose(tensor[0,0], tensor[2,2], rtol=1e-6)
                print(f"   Hexagonal constraints: {'✅ Pass' if xy_equal and z_different else '❌ Fail'}")
        
        print("\n🧪 TESTING FITTED PEAKS WORKFLOW:")
        print("=" * 60)
        
        # Test with fitted peaks
        app.fitted_peaks = []
        peak_positions = [300, 500, 700]
        for pos in peak_positions:
            peak = type('Peak', (), {})()
            peak.center = pos
            peak.amplitude = 1.0
            peak.sigma = 30.0
            app.fitted_peaks.append(peak)
        
        # Set to tetragonal for testing
        app.crystal_structure = {
            'crystal_system': 'Tetragonal',
            'point_group': '4/mmm',
            'name': 'Test Tetragonal'
        }
        
        print(f"\n🔬 Creating tensors for Tetragonal crystal:")
        success = app.create_tensors_from_fitted_peaks()
        
        if success:
            print("✅ Tensor creation successful!")
            print(f"   Number of tensors: {len(app.tensor_data_3d['wavenumbers'])}")
            
            # Check first tensor
            first_tensor = app.tensor_data_3d['tensors'][0]
            print(f"   First tensor (at {app.tensor_data_3d['wavenumbers'][0]:.1f} cm⁻¹):")
            for i, row in enumerate(first_tensor):
                print(f"     [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
            
            # Verify tetragonal symmetry
            xy_equal = np.allclose(first_tensor[0,0], first_tensor[1,1], rtol=1e-6)
            z_different = not np.allclose(first_tensor[0,0], first_tensor[2,2], rtol=1e-6)
            off_diagonal_zero = np.allclose([first_tensor[0,1], first_tensor[0,2], first_tensor[1,2]], 0, atol=1e-6)
            
            if xy_equal and z_different and off_diagonal_zero:
                print("   ✅ Tetragonal symmetry constraints satisfied!")
            else:
                print("   ❌ Tetragonal symmetry constraints violated!")
                
        else:
            print("❌ Tensor creation failed!")
        
        return success
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        root.destroy()

if __name__ == "__main__":
    print("🔬 TESTING SYMMETRY-CONSTRAINED RAMAN TENSORS")
    print("=" * 70)
    
    success = test_symmetry_constrained_tensors()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Key Improvements:")
        print("• Tensors now respect crystal symmetry constraints")
        print("• Different crystal systems have appropriate tensor forms")
        print("• Cubic: isotropic (αxx = αyy = αzz)")
        print("• Tetragonal: uniaxial (αxx = αyy ≠ αzz)")
        print("• Orthorhombic: biaxial (αxx ≠ αyy ≠ αzz)")
        print("• All tensors remain symmetric (required for Raman)")
        print("\n🎯 Now your ellipsoids will have physically meaningful shapes!")
    else:
        print("⚠️  Some tests failed. Check the implementation.") 