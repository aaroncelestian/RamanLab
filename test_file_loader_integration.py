"""
Test script to verify LabSpec6 integration with file loaders
"""

from utils.file_loaders import SpectrumLoader

def test_l6s_loading():
    """Test loading a .l6s file through the SpectrumLoader."""
    
    loader = SpectrumLoader()
    
    print("=" * 70)
    print("Testing LabSpec6 Integration with SpectrumLoader")
    print("=" * 70)
    
    # Check if .l6s is supported
    print(f"\nSupported extensions: {loader.supported_extensions}")
    
    if '.l6s' not in loader.supported_extensions:
        print("\n❌ ERROR: .l6s extension not in supported list")
        return False
    
    # Try to load the test file
    file_path = '/Users/aaroncelestian/Python/RamanLab/demo_data/tavetch_switzerland_05.l6s'
    print(f"\nLoading: {file_path}")
    
    wavenumbers, intensities, metadata = loader.load_spectrum(file_path)
    
    if wavenumbers is None:
        print(f"\n❌ ERROR: {metadata.get('error', 'Unknown error')}")
        return False
    
    print("\n✅ File loaded successfully!")
    
    # Display results
    print("\n" + "=" * 70)
    print("METADATA")
    print("=" * 70)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"Number of data points: {len(wavenumbers)}")
    print(f"Wavenumber range: {wavenumbers[0]:.2f} - {wavenumbers[-1]:.2f} cm⁻¹")
    print(f"Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Integration test PASSED!")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    import sys
    success = test_l6s_loading()
    sys.exit(0 if success else 1)
