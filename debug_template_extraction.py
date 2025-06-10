#!/usr/bin/env python3
"""
Debug script for template extraction and preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt

def debug_template_preprocessing():
    """Debug what happens to templates during preprocessing."""
    print("=== Template Extraction Debug ===")
    print()
    
    # This would be run in your application context
    print("To debug your template extraction, add this code to your template extraction:")
    print()
    
    debug_code = '''
    # Add this BEFORE preprocessing in _confirm_template_extraction:
    print(f"Original template - Min: {np.min(int_array)}, Max: {np.max(int_array)}, Mean: {np.mean(int_array)}")
    print(f"Original wavenumbers - Min: {np.min(wn_array)}, Max: {np.max(wn_array)}, Length: {len(wn_array)}")
    
    # Add this AFTER preprocessing:
    print(f"Processed template - Min: {np.min(processed_intensities)}, Max: {np.max(processed_intensities)}, Mean: {np.mean(processed_intensities)}")
    print(f"Target wavenumbers - Min: {np.min(self.template_manager.target_wavenumbers)}, Max: {np.max(self.template_manager.target_wavenumbers)}, Length: {len(self.template_manager.target_wavenumbers)}")
    
    # Check if wavenumbers match map data
    if hasattr(self, 'map_data') and self.map_data:
        first_spectrum = next(iter(self.map_data.spectra.values()))
        map_wn = first_spectrum.wavenumbers
        print(f"Map wavenumbers - Min: {np.min(map_wn)}, Max: {np.max(map_wn)}, Length: {len(map_wn)}")
        print(f"Wavenumber match: {np.array_equal(self.template_manager.target_wavenumbers, map_wn)}")
    '''
    
    print(debug_code)
    print()
    print("=== Potential Issues ===")
    print("1. Template wavenumbers don't match map wavenumbers")
    print("2. Over-aggressive smoothing destroying spectral features")  
    print("3. Normalization making templates indistinguishable")
    print("4. Interpolation artifacts")
    print()
    print("=== Quick Fix Suggestions ===")
    print("1. Skip preprocessing for map-extracted templates")
    print("2. Use map wavenumbers directly as target wavenumbers")
    print("3. Compare original vs processed templates visually")

if __name__ == "__main__":
    debug_template_preprocessing() 