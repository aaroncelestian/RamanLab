#!/usr/bin/env python3

"""
Test script to verify that the tensor import fix works correctly.
This simulates the user's workflow: Import from Optimization -> Import Structure -> Import Tensors
"""

import tkinter as tk
import raman_polarization_analyzer as rpa
import numpy as np

def test_tensor_import_workflow():
    """Test the complete workflow that was causing issues."""
    print("🧪 TESTING TENSOR IMPORT WORKFLOW...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = rpa.RamanPolarizationAnalyzer(root)
        
        # Step 1: Create test data (simulating fitted peaks)
        print("\n1. Creating test fitted peaks...")
        peak_positions = [300, 500, 700, 900, 1100]
        app.fitted_peaks = []
        for pos in peak_positions:
            peak = type('Peak', (), {})()
            peak.center = pos
            peak.amplitude = np.random.uniform(0.5, 1.5)
            peak.sigma = 30.0
            app.fitted_peaks.append(peak)
        print(f"   ✅ Created {len(app.fitted_peaks)} fitted peaks")
        
        # Step 2: Create tensors from fitted peaks (simulating "Import from Optimization")
        print("\n2. Creating tensors from fitted peaks...")
        success = app.create_tensors_from_fitted_peaks()
        if success:
            print("   ✅ Tensor data created successfully")
            print(f"   📊 Tensor data keys: {list(app.tensor_data_3d.keys())}")
            print(f"   🔢 Number of tensors: {len(app.tensor_data_3d.get('wavenumbers', []))}")
        else:
            print("   ❌ Failed to create tensor data")
            return False
        
        # Step 3: Import structure data
        print("\n3. Creating test structure data...")
        app.structure_data = {
            'lattice_vectors': np.array([
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0], 
                [0.0, 0.0, 3.0]
            ]),
            'atoms': [
                {'element': 'Si', 'frac_coords': [0.0, 0.0, 0.0]},
                {'element': 'O', 'frac_coords': [0.5, 0.0, 0.0]},
            ],
            'crystal_system': 'Cubic'
        }
        app.crystal_structure = app.structure_data.copy()
        print("   ✅ Structure data created")
        
        # Step 4: Test the problematic "Import Tensors" button
        print("\n4. Testing 'Import Tensors' button (this was causing the issue)...")
        
        # Store current tensor data for comparison
        tensor_data_before = app.tensor_data_3d.copy() if app.tensor_data_3d else None
        raman_tensors_before = app.raman_tensors.copy() if hasattr(app, 'raman_tensors') else {}
        
        print(f"   📊 Tensor data before import: {tensor_data_before is not None}")
        print(f"   🔢 Raman tensors before import: {len(raman_tensors_before)} entries")
        
        # This should now ask the user if they want to overwrite existing data
        # For testing, we'll simulate clicking "No" to keep existing data
        print("   🔄 Calling import_tensor_data_3d()...")
        
        # Temporarily override messagebox to simulate user clicking "No"
        original_askyesno = tk.messagebox.askyesno
        def mock_askyesno(title, message):
            print(f"   💬 Dialog: {title}")
            print(f"      Message: {message}")
            print("   👤 User choice: No (keep existing data)")
            return False  # Simulate clicking "No"
        
        tk.messagebox.askyesno = mock_askyesno
        
        try:
            app.import_tensor_data_3d()
        finally:
            tk.messagebox.askyesno = original_askyesno
        
        # Check if tensor data is preserved
        tensor_data_after = app.tensor_data_3d
        raman_tensors_after = getattr(app, 'raman_tensors', {})
        
        print(f"   📊 Tensor data after import: {tensor_data_after is not None}")
        print(f"   🔢 Raman tensors after import: {len(raman_tensors_after)} entries")
        
        # Verify data is preserved
        if tensor_data_after is not None and len(raman_tensors_after) > 0:
            print("   ✅ SUCCESS: Tensor data preserved!")
            return True
        else:
            print("   ❌ FAILURE: Tensor data was lost!")
            return False
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        root.destroy()

def test_import_with_overwrite():
    """Test the import with overwrite option."""
    print("\n\n🧪 TESTING TENSOR IMPORT WITH OVERWRITE...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = rpa.RamanPolarizationAnalyzer(root)
        
        # Create initial tensor data
        app.fitted_peaks = []
        peak = type('Peak', (), {})()
        peak.center = 500
        peak.amplitude = 1.0
        peak.sigma = 30.0
        app.fitted_peaks.append(peak)
        
        app.create_tensors_from_fitted_peaks()
        print("   ✅ Initial tensor data created")
        
        # Create different tensor data in raman_tensors (simulating Tensor Analysis tab)
        app.raman_tensors = {
            '800.0': {
                'tensor': np.eye(3) * 2.0,
                'wavenumber': 800.0,
                'intensity': 2.0
            }
        }
        print("   ✅ Different tensor data in Tensor Analysis tab")
        
        # Test import with overwrite
        original_askyesno = tk.messagebox.askyesno
        def mock_askyesno_yes(title, message):
            print(f"   💬 Dialog: {title}")
            print("   👤 User choice: Yes (overwrite)")
            return True  # Simulate clicking "Yes"
        
        tk.messagebox.askyesno = mock_askyesno_yes
        
        try:
            app.import_tensor_data_3d()
        finally:
            tk.messagebox.askyesno = original_askyesno
        
        # Check if data was overwritten
        if hasattr(app, 'tensor_data_3d') and '800.0' in str(app.tensor_data_3d):
            print("   ✅ SUCCESS: Data overwritten correctly!")
            return True
        else:
            print("   ❌ FAILURE: Data not overwritten!")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    finally:
        root.destroy()

if __name__ == "__main__":
    print("🔬 TESTING TENSOR IMPORT FIX")
    print("=" * 50)
    
    # Test 1: Preserve existing data
    test1_passed = test_tensor_import_workflow()
    
    # Test 2: Overwrite when requested
    test2_passed = test_import_with_overwrite()
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS:")
    print(f"   Test 1 (Preserve data): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Test 2 (Overwrite data): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The fix should work correctly.")
        print("\nNow when you click 'Import Tensors':")
        print("• If you have existing tensor data, it will ask before overwriting")
        print("• If you click 'No', your existing data is preserved")
        print("• If you click 'Yes', it will import new data from Tensor Analysis tab")
        print("• If import fails, your original data is restored")
    else:
        print("\n⚠️  Some tests failed. The fix may need more work.") 