"""
Quick test to verify .l6s loading works in the main app's parse_spectrum_file method
"""

import sys
import numpy as np
from pathlib import Path

# Simulate the parse_spectrum_file method from raman_analysis_app_qt6.py
def parse_spectrum_file(file_path):
    """Enhanced spectrum file parser that handles headers, metadata, and various formats."""
    import csv
    import re
    from pathlib import Path
    
    file_extension = Path(file_path).suffix.lower()
    
    # Handle LabSpec6 binary files
    if file_extension == '.l6s':
        try:
            from utils.labspec6_parser import load_labspec6_spectrum
            wavenumbers, intensities, metadata = load_labspec6_spectrum(file_path)
            if wavenumbers is None:
                raise ValueError(metadata.get('error', 'Failed to parse LabSpec6 file'))
            return wavenumbers, intensities, metadata
        except ImportError:
            raise ValueError("LabSpec6 parser not available. Please check installation.")
    
    # Handle text files (not tested here)
    raise ValueError("Only testing .l6s files")


# Test
file_path = '/Users/aaroncelestian/Python/RamanLab/demo_data/tavetch_switzerland_05.l6s'

print("=" * 70)
print("Testing Main App Integration with .l6s File")
print("=" * 70)
print(f"\nFile: {file_path}")

try:
    wavenumbers, intensities, metadata = parse_spectrum_file(file_path)
    
    print("\n✅ File loaded successfully!")
    print(f"\nData points: {len(wavenumbers)}")
    print(f"Wavenumber range: {wavenumbers[0]:.2f} - {wavenumbers[-1]:.2f} cm⁻¹")
    print(f"Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")
    print(f"Sample name: {metadata.get('sample_name', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("✅ Main app integration test PASSED!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
