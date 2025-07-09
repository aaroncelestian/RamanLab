#!/usr/bin/env python3
"""
Working Spectrum Plot Example

This demonstrates the correct way to load and plot spectra using the safe approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from pkl_utils import get_example_spectrum_file
from utils.file_loaders import load_spectrum_file

def load_and_plot_spectrum(mineral_name='quartz'):
    """Load and plot a spectrum safely."""
    print(f"🔍 Loading {mineral_name} spectrum...")
    
    # Step 1: Get the spectrum file safely
    spectrum_file = get_example_spectrum_file(mineral_name)
    
    if not spectrum_file or not spectrum_file.exists():
        print(f"❌ {mineral_name} spectrum not found!")
        return None
    
    print(f"📄 Found: {spectrum_file}")
    
    # Step 2: Load the spectrum data properly
    wavenumbers, intensities, metadata = load_spectrum_file(str(spectrum_file))
    
    if wavenumbers is None or intensities is None:
        print(f"❌ Failed to load {mineral_name} data!")
        return None
    
    print(f"✅ Loaded: {len(wavenumbers)} data points")
    print(f"   • Range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
    print(f"   • Intensity: {intensities.min():.1f} - {intensities.max():.1f}")
    
    # Step 3: Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(wavenumbers, intensities, 'b-', linewidth=1)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Raman Spectrum: {mineral_name.title()}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✅ Plot created successfully!")
    
    return {
        'wavenumbers': wavenumbers,
        'intensities': intensities,
        'metadata': metadata,
        'filename': spectrum_file.name
    }

def load_multiple_spectra():
    """Load multiple spectra into a properly structured dictionary."""
    print("📊 Loading Multiple Spectra")
    print("=" * 40)
    
    spectra_dict = {}
    mineral_samples = ['quartz', 'calcite', 'muscovite', 'feldspar']
    
    for mineral in mineral_samples:
        spectrum_file = get_example_spectrum_file(mineral)
        
        if spectrum_file and spectrum_file.exists():
            print(f"📄 Loading {mineral}...")
            
            wavenumbers, intensities, metadata = load_spectrum_file(str(spectrum_file))
            
            if wavenumbers is not None and intensities is not None:
                # THIS IS THE CORRECT STRUCTURE
                spectra_dict[mineral] = {
                    'wavenumbers': wavenumbers,
                    'intensities': intensities,
                    'metadata': metadata,
                    'filename': spectrum_file.name
                }
                print(f"   ✅ {mineral}: {len(wavenumbers)} points")
            else:
                print(f"   ❌ {mineral}: Loading failed")
        else:
            print(f"   ⚠️  {mineral}: File not found")
    
    return spectra_dict

def plot_first_spectrum_correctly(spectra_dict):
    """Plot the first spectrum with proper error checking."""
    print("\n🎯 Plotting First Spectrum (Correct Method)")
    print("=" * 50)
    
    if not spectra_dict:
        print("❌ No spectra available!")
        return
    
    # Get first spectrum
    first_key = list(spectra_dict.keys())[0]
    spectrum_data = spectra_dict[first_key]
    
    # DEBUG: Check the data structure
    print(f"Selected: {first_key}")
    print(f"Data type: {type(spectrum_data)}")
    
    if isinstance(spectrum_data, dict):
        print(f"Available keys: {list(spectrum_data.keys())}")
    else:
        print(f"❌ ERROR: Expected dict, got {type(spectrum_data)}")
        return
    
    # Check for required keys
    if 'wavenumbers' not in spectrum_data or 'intensities' not in spectrum_data:
        print(f"❌ Missing required keys!")
        print(f"   Expected: ['wavenumbers', 'intensities']")
        print(f"   Found: {list(spectrum_data.keys())}")
        return
    
    # Extract data safely
    wavenumbers = spectrum_data['wavenumbers']
    intensities = spectrum_data['intensities']
    filename = spectrum_data.get('filename', first_key)
    
    print(f"✅ Data structure is correct!")
    print(f"   • Wavenumbers: {len(wavenumbers)} points")
    print(f"   • Intensities: {len(intensities)} points")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(wavenumbers, intensities, 'b-', linewidth=1)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Raman Spectrum: {filename}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✅ Plot created successfully!")

if __name__ == "__main__":
    print("🚀 Working Spectrum Plot Example")
    print("=" * 60)
    
    # Method 1: Load and plot single spectrum
    print("\n1️⃣ Single Spectrum Loading:")
    spectrum_data = load_and_plot_spectrum('quartz')
    
    # Method 2: Load multiple spectra correctly
    print("\n2️⃣ Multiple Spectra Loading:")
    spectra_dict = load_multiple_spectra()
    
    # Method 3: Plot first spectrum with error checking
    if spectra_dict:
        plot_first_spectrum_correctly(spectra_dict)
    
    print("\n✅ All methods completed successfully!")
    print("\n💡 Key Points:")
    print("   • Always use load_spectrum_file() for loading")
    print("   • Structure: spectra_dict[name] = {'wavenumbers': array, 'intensities': array}")
    print("   • Check data types before accessing keys")
    print("   • Use get_example_spectrum_file() for safe file paths") 