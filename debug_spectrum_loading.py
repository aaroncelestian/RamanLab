#!/usr/bin/env python3
"""
Debug Script: Spectrum Loading Issues

This script helps diagnose and fix spectrum loading problems by showing
the actual data structure and providing working solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pkl_utils import get_example_data_paths, get_example_spectrum_file
from utils.file_loaders import load_spectrum_file

def debug_data_structure(spectra_dict):
    """Debug the structure of a spectra dictionary."""
    print("🔍 Debugging Data Structure")
    print("=" * 50)
    
    print(f"Type of spectra_dict: {type(spectra_dict)}")
    print(f"Number of items: {len(spectra_dict)}")
    print()
    
    # Show first few keys and their types
    print("First 3 items structure:")
    for i, (key, value) in enumerate(list(spectra_dict.items())[:3]):
        print(f"  Key {i+1}: {repr(key)} (type: {type(key)})")
        print(f"  Value {i+1}: {type(value)}")
        
        if isinstance(value, str):
            print(f"    String value: {repr(value[:100])}...")  # First 100 chars
        elif isinstance(value, dict):
            print(f"    Dict keys: {list(value.keys())}")
        elif hasattr(value, 'shape'):
            print(f"    Array shape: {value.shape}")
        else:
            print(f"    Value: {repr(value)}")
        print()

def load_spectra_correctly():
    """Demonstrate correct spectrum loading using safe approach."""
    print("✅ Correct Spectrum Loading")
    print("=" * 50)
    
    # Get available example files
    paths = get_example_data_paths()
    
    # Load multiple spectra safely
    spectra_dict = {}
    mineral_samples = ['quartz', 'calcite', 'muscovite', 'feldspar']
    
    for mineral in mineral_samples:
        spectrum_file = get_example_spectrum_file(mineral)
        
        if spectrum_file and spectrum_file.exists():
            print(f"📄 Loading {mineral}: {spectrum_file.name}")
            
            wavenumbers, intensities, metadata = load_spectrum_file(str(spectrum_file))
            
            if wavenumbers is not None and intensities is not None:
                spectra_dict[mineral] = {
                    'wavenumbers': wavenumbers,
                    'intensities': intensities,
                    'metadata': metadata,
                    'filename': spectrum_file.name
                }
                print(f"   ✅ Loaded: {len(wavenumbers)} data points")
            else:
                print(f"   ❌ Failed to load {mineral}")
        else:
            print(f"   ⚠️  {mineral}: File not found")
    
    return spectra_dict

def plot_spectrum_safely(spectra_dict):
    """Plot spectrum with proper error checking."""
    print("\n📊 Plotting Spectrum Safely")
    print("=" * 30)
    
    if not spectra_dict:
        print("❌ No spectra loaded!")
        return
    
    # Get first spectrum
    first_key = list(spectra_dict.keys())[0]
    spectrum_data = spectra_dict[first_key]
    
    print(f"Selected spectrum: {first_key}")
    print(f"Data type: {type(spectrum_data)}")
    
    # Check if we have the expected structure
    if isinstance(spectrum_data, dict) and 'wavenumbers' in spectrum_data and 'intensities' in spectrum_data:
        wavenumbers = spectrum_data['wavenumbers']
        intensities = spectrum_data['intensities']
        filename = spectrum_data.get('filename', first_key)
        
        print(f"✅ Valid spectrum data structure")
        print(f"   • Wavenumbers: {len(wavenumbers)} points")
        print(f"   • Range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
        print(f"   • Intensities: {len(intensities)} points")
        print(f"   • Range: {intensities.min():.1f} - {intensities.max():.1f}")
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(wavenumbers, intensities, 'b-', linewidth=1)
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title(f'Raman Spectrum: {filename}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("✅ Plot created successfully!")
        
    else:
        print("❌ Unexpected data structure!")
        print(f"Expected: dict with 'wavenumbers' and 'intensities' keys")
        print(f"Got: {type(spectrum_data)}")
        if isinstance(spectrum_data, dict):
            print(f"Available keys: {list(spectrum_data.keys())}")

def fix_common_issues():
    """Show solutions for common spectrum loading issues."""
    print("\n🔧 Common Issues and Solutions")
    print("=" * 50)
    
    print("1. TypeError: string indices must be integers")
    print("   Problem: spectrum_data is a string, not a dict")
    print("   Solution: Check data loading - values should be dicts, not strings")
    print()
    
    print("2. KeyError: 'wavenumbers' or 'intensities'")
    print("   Problem: Dictionary doesn't have expected keys")
    print("   Solution: Check actual keys with list(spectrum_data.keys())")
    print()
    
    print("3. FileNotFoundError")
    print("   Problem: Spectrum files not found")
    print("   Solution: Use get_example_spectrum_file() for safe loading")
    print()
    
    print("4. Empty or None data")
    print("   Problem: File loading failed")
    print("   Solution: Check file format and use load_spectrum_file()")

def working_example():
    """Complete working example."""
    print("\n🚀 Complete Working Example")
    print("=" * 50)
    
    try:
        # Load spectra correctly
        spectra_dict = load_spectra_correctly()
        
        if spectra_dict:
            # Plot first spectrum
            plot_spectrum_safely(spectra_dict)
            
            # Show available spectra
            print(f"\n📋 Available Spectra ({len(spectra_dict)}):")
            for name, data in spectra_dict.items():
                wn_count = len(data['wavenumbers'])
                wn_range = f"{data['wavenumbers'][0]:.1f}-{data['wavenumbers'][-1]:.1f}"
                print(f"   • {name}: {wn_count} points ({wn_range} cm⁻¹)")
        else:
            print("❌ No spectra could be loaded!")
            
    except Exception as e:
        print(f"❌ Error in working example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Spectrum Loading Debugger")
    print("=" * 60)
    
    # If user provides their spectra_dict, we can debug it
    # For now, we'll create a working example
    
    # Show common issues and solutions
    fix_common_issues()
    
    # Demonstrate correct loading
    working_example()
    
    print("\n💡 To debug your specific issue:")
    print("   1. Add: debug_data_structure(your_spectra_dict)")
    print("   2. Check the actual structure before plotting")
    print("   3. Use the safe loading functions from pkl_utils")
    print("   4. Always validate data structure before accessing keys") 