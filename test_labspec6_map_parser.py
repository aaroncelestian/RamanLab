"""
Test script for LabSpec6 .l6m map parser
"""

import sys
import numpy as np
from utils.labspec6_map_parser import load_labspec6_map

def test_l6m_parser():
    """Test the LabSpec6 map parser."""
    
    # Find a test .l6m file
    import glob
    l6m_files = glob.glob('/Users/aaroncelestian/Python/RamanLab/demo_data/*.l6m')
    
    if not l6m_files:
        print("❌ No .l6m test files found in demo_data/")
        return False
    
    file_path = l6m_files[0]
    
    print("=" * 70)
    print("Testing LabSpec6 Map Parser (.l6m)")
    print("=" * 70)
    print(f"\nFile: {file_path}")
    
    # Load the file
    wavenumbers, x_coords, y_coords, cube, metadata = load_labspec6_map(file_path)
    
    if wavenumbers is None:
        print(f"\n❌ Error: {metadata.get('error', 'Unknown error')}")
        return False
    
    print("\n✅ File loaded successfully!")
    
    # Display metadata
    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # Display data summary
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Cube shape: {cube.shape} (n_y, n_x, n_pts)")
    print(f"Number of spectra: {metadata['n_spectra']}")
    print(f"Wavenumber range: {wavenumbers[0]:.2f} - {wavenumbers[-1]:.2f} cm⁻¹")
    print(f"X range: {x_coords[0]:.3f} - {x_coords[-1]:.3f} µm")
    print(f"Y range: {y_coords[0]:.3f} - {y_coords[-1]:.3f} µm")
    print(f"Intensity range: {cube.min():.2f} - {cube.max():.2f}")
    
    # Check data quality
    print("\n" + "=" * 70)
    print("DATA QUALITY")
    print("=" * 70)
    n_nonfinite = np.sum(~np.isfinite(cube))
    n_negative = np.sum(cube < 0)
    print(f"Non-finite values: {n_nonfinite}")
    print(f"Negative values: {n_negative}")
    
    if n_nonfinite == 0 and n_negative == 0:
        print("✅ Data is clean (no NaNs, no negatives)")
    
    print("\n" + "=" * 70)
    print("✅ Test completed successfully!")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    success = test_l6m_parser()
    sys.exit(0 if success else 1)
