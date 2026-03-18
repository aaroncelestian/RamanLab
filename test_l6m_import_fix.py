"""
Test the fixed .l6m import with correct SpectrumData signature
"""

import sys
import numpy as np
from pathlib import Path

def test_fixed_import():
    """Test that the import creates SpectrumData objects correctly."""
    
    print("=" * 70)
    print("Testing Fixed .l6m Import")
    print("=" * 70)
    
    # Load the map
    from utils.labspec6_map_parser import load_labspec6_map
    
    file_path = '/Users/aaroncelestian/Python/RamanLab/demo_data/crtl+HHL_testmap_02.l6m'
    wavenumbers, x_coords, y_coords, cube, metadata = load_labspec6_map(file_path)
    
    if wavenumbers is None:
        print(f"❌ Failed to load: {metadata}")
        return False
    
    print(f"✅ Map loaded: {cube.shape}")
    
    # Test SpectrumData creation with correct signature
    print("\nTesting SpectrumData creation...")
    
    try:
        from map_analysis_2d.core.spectrum_data import SpectrumData
        
        # Create test spectrum with correct parameters
        n_y, n_x, n_pts = cube.shape
        sample_name = metadata.get('sample_name', 'test')
        
        test_spectrum = SpectrumData(
            x_pos=0,  # int, not float
            y_pos=0,  # int, not float
            wavenumbers=wavenumbers.copy(),
            intensities=cube[0, 0, :].copy(),
            filename=f"{sample_name}_y0_x0"  # filename, not metadata
        )
        
        print(f"✅ SpectrumData created successfully")
        print(f"   - x_pos: {test_spectrum.x_pos} (type: {type(test_spectrum.x_pos).__name__})")
        print(f"   - y_pos: {test_spectrum.y_pos} (type: {type(test_spectrum.y_pos).__name__})")
        print(f"   - filename: {test_spectrum.filename}")
        print(f"   - wavenumbers: {len(test_spectrum.wavenumbers)} points")
        print(f"   - intensities: {len(test_spectrum.intensities)} points")
        
    except Exception as e:
        print(f"❌ Error creating SpectrumData: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ All tests PASSED!")
    print("=" * 70)
    print("\nThe .l6m import should now work correctly in the Map Analysis tool.")
    print("\nTo use:")
    print("1. Launch Map Analysis")
    print("2. File → Import LabSpec6 Map (.l6m) to PKL...")
    print("3. Select your .l6m file")
    print("4. Confirm conversion")
    print("5. Save as PKL")
    
    return True

if __name__ == '__main__':
    success = test_fixed_import()
    sys.exit(0 if success else 1)
