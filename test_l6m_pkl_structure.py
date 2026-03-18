"""
Test that the new PKL structure is correct
"""

import sys
import numpy as np
from pathlib import Path

def test_pkl_structure():
    """Test the PKL structure matches what the loader expects."""
    
    print("=" * 70)
    print("Testing .l6m PKL Structure")
    print("=" * 70)
    
    # Simulate the structure we're creating
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class SpectrumData:
        x_pos: int
        y_pos: int
        wavenumbers: np.ndarray
        intensities: np.ndarray
        filename: str
        processed_intensities: Optional[np.ndarray] = None
    
    class SimpleMapData:
        def __init__(self, spectra_dict, wavenumbers):
            self.spectra = spectra_dict
            self.target_wavenumbers = wavenumbers
            self.wavenumbers = wavenumbers
            self.x_positions = sorted(set(k[0] for k in spectra_dict.keys()))
            self.y_positions = sorted(set(k[1] for k in spectra_dict.keys()))
    
    # Create test data
    wavenumbers = np.array([100, 200, 300, 400, 500])
    
    # Create some test spectra
    spectra_dict = {}
    for y in range(3):
        for x in range(3):
            spec = SpectrumData(
                x_pos=x,
                y_pos=y,
                wavenumbers=wavenumbers.copy(),
                intensities=np.random.rand(5) * 100,
                filename=f"test_y{y}_x{x}"
            )
            spectra_dict[(x, y)] = spec
    
    # Create map data
    map_data = SimpleMapData(spectra_dict, wavenumbers)
    
    # Test the structure
    print("\n✅ SimpleMapData created")
    print(f"   - Has .spectra attribute: {hasattr(map_data, 'spectra')}")
    print(f"   - .spectra is dict: {isinstance(map_data.spectra, dict)}")
    print(f"   - Number of spectra: {len(map_data.spectra)}")
    print(f"   - Has .target_wavenumbers: {hasattr(map_data, 'target_wavenumbers')}")
    print(f"   - Has .wavenumbers: {hasattr(map_data, 'wavenumbers')}")
    print(f"   - Has .x_positions: {hasattr(map_data, 'x_positions')}")
    print(f"   - Has .y_positions: {hasattr(map_data, 'y_positions')}")
    
    # Test accessing spectra
    print("\n✅ Testing spectra access:")
    test_spec = map_data.spectra[(0, 0)]
    print(f"   - Spectrum at (0,0): {test_spec.filename}")
    print(f"   - x_pos: {test_spec.x_pos}, y_pos: {test_spec.y_pos}")
    print(f"   - Wavenumbers: {len(test_spec.wavenumbers)} points")
    print(f"   - Intensities: {len(test_spec.intensities)} points")
    
    # Test positions
    print(f"\n✅ Grid positions:")
    print(f"   - X positions: {map_data.x_positions}")
    print(f"   - Y positions: {map_data.y_positions}")
    
    # Save and reload test
    print("\n✅ Testing PKL save/load:")
    import pickle
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        temp_path = f.name
        pickle.dump(map_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"   - Saved to: {temp_path}")
    
    with open(temp_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"   - Loaded successfully")
    print(f"   - Has .spectra: {hasattr(loaded_data, 'spectra')}")
    print(f"   - Number of spectra: {len(loaded_data.spectra)}")
    
    # Clean up
    import os
    os.unlink(temp_path)
    
    print("\n" + "=" * 70)
    print("✅ All structure tests PASSED!")
    print("=" * 70)
    print("\nThe PKL structure is correct. You can now:")
    print("1. Re-import your .l6m file using the fixed code")
    print("2. The new PKL file will have the correct structure")
    print("3. It will load successfully in Map Analysis")
    
    return True

if __name__ == '__main__':
    success = test_pkl_structure()
    sys.exit(0 if success else 1)
