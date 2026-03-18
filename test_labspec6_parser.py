"""
Test script for LabSpec6 .l6s parser
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.labspec6_parser import load_labspec6_spectrum

def test_labspec6_parser():
    """Test the LabSpec6 parser with the demo file."""
    
    file_path = '/Users/aaroncelestian/Python/RamanLab/demo_data/tavetch_switzerland_05.l6s'
    
    print("="*70)
    print("Testing LabSpec6 (.l6s) Parser")
    print("="*70)
    print(f"\nFile: {file_path}")
    
    # Load the file
    wavenumbers, intensities, metadata = load_labspec6_spectrum(file_path)
    
    if wavenumbers is None:
        print(f"\n❌ Error: {metadata.get('error', 'Unknown error')}")
        return False
    
    print("\n✅ File loaded successfully!")
    
    # Display metadata
    print("\n" + "="*70)
    print("METADATA")
    print("="*70)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # Display data summary
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"Number of data points: {len(wavenumbers)}")
    print(f"Wavenumber range: {wavenumbers[0]:.2f} - {wavenumbers[-1]:.2f} cm⁻¹")
    print(f"Intensity range: {np.min(intensities):.2f} - {np.max(intensities):.2f}")
    print(f"Mean intensity: {np.mean(intensities):.2f}")
    print(f"Std intensity: {np.std(intensities):.2f}")
    
    # Show sample data points
    print("\n" + "="*70)
    print("SAMPLE DATA POINTS")
    print("="*70)
    print("First 10 points:")
    for i in range(min(10, len(wavenumbers))):
        print(f"  {wavenumbers[i]:8.2f} cm⁻¹ : {intensities[i]:10.2f}")
    
    print("\nLast 10 points:")
    for i in range(max(0, len(wavenumbers)-10), len(wavenumbers)):
        print(f"  {wavenumbers[i]:8.2f} cm⁻¹ : {intensities[i]:10.2f}")
    
    # Create a simple plot
    print("\n" + "="*70)
    print("Creating plot...")
    print("="*70)
    
    plt.figure(figsize=(12, 6))
    plt.plot(wavenumbers, intensities, 'b-', linewidth=0.8)
    plt.xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    plt.ylabel('Intensity (counts)', fontsize=12)
    plt.title(f'Raman Spectrum: {metadata["filename"]}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = '/Users/aaroncelestian/Python/RamanLab/test_labspec6_spectrum.png'
    plt.savefig(output_path, dpi=150)
    print(f"✅ Plot saved to: {output_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)
    
    return True

if __name__ == '__main__':
    success = test_labspec6_parser()
    sys.exit(0 if success else 1)
