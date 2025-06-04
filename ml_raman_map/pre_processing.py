import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.interpolate import interp1d
import sys
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

# Import our enhanced file reading utilities
from ml_raman_map.file_reader_utils import read_raman_spectrum, batch_read_spectra

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares Smoothing for baseline correction.
    
    Parameters:
    -----------
    y : array-like
        Input spectrum.
    lam : float
        Smoothness parameter (default: 1e5).
    p : float
        Asymmetry parameter (default: 0.01).
    niter : int
        Number of iterations (default: 10).
        
    Returns:
    --------
    array-like
        Estimated baseline.
    """
    L = len(y)
    D = csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    
    for i in range(niter):
        W = csc_matrix((w, (np.arange(L), np.arange(L))))
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    
    return z

def resample_spectrum(wavenumbers, intensities, target_wavenumbers):
    """
    Resample a spectrum to match target wavenumbers using linear interpolation.
    
    Parameters:
    -----------
    wavenumbers : ndarray
        Original wavenumber values
    intensities : ndarray
        Original intensity values
    target_wavenumbers : ndarray
        Target wavenumber values for resampling
        
    Returns:
    --------
    ndarray
        Resampled intensity values
    """
    # Create interpolation function
    f = interp1d(wavenumbers, intensities, kind='linear', bounds_error=False, fill_value=0)
    
    # Resample to target wavenumbers
    resampled = f(target_wavenumbers)
    
    return resampled

def preprocess_spectrum(wavenumbers, intensities, target_wavenumbers=None):
    """
    Preprocess Raman spectrum with baseline correction and smoothing.
    
    Parameters:
    -----------
    wavenumbers : ndarray
        Raman shift values (cm^-1)
    intensities : ndarray
        Intensity values
    target_wavenumbers : ndarray, optional
        Target wavenumber values for resampling. If None, original wavenumbers are used.
        
    Returns:
    --------
    ndarray
        Preprocessed spectrum
    """
    # Add error checking
    if wavenumbers is None or intensities is None:
        raise ValueError("Input wavenumbers or intensities are None")
    
    if len(wavenumbers) != len(intensities):
        raise ValueError(f"Wavenumbers and intensities have different lengths: {len(wavenumbers)} vs {len(intensities)}")
    
    # Apply Savitzky-Golay filter for smoothing
    smoothed = savgol_filter(intensities, window_length=11, polyorder=3)
    
    # Simple baseline correction (can be replaced with more sophisticated methods)
    baseline = baseline_als(smoothed)
    corrected = smoothed - baseline
    
    # Normalize to maximum intensity
    if np.max(corrected) > 0:
        normalized = corrected / np.max(corrected)
    else:
        normalized = corrected
    
    # Resample if target wavenumbers are provided
    if target_wavenumbers is not None:
        normalized = resample_spectrum(wavenumbers, normalized, target_wavenumbers)
        wavenumbers = target_wavenumbers
        
    # Return both wavenumbers and normalized intensities
    return np.column_stack((wavenumbers, normalized))

def has_plastic_peaks(wavenumbers, intensities, threshold=0.15):
    """
    Detect if spectrum contains characteristic plastic peaks.
    
    Parameters:
    -----------
    wavenumbers : ndarray
        Raman shift values (cm^-1)
    intensities : ndarray
        Intensity values
    threshold : float
        Peak height threshold (relative to max intensity)
        
    Returns:
    --------
    bool
        True if plastic peaks are detected, False otherwise
    """
    # Preprocess spectrum
    processed = preprocess_spectrum(wavenumbers, intensities)
    
    # Plastic characteristic peak regions (cm^-1)
    # Common peaks for various plastics (PE, PP, PET, PS, etc.)
    plastic_regions = [
        (2800, 3000),  # C-H stretching (most plastics)
        (1430, 1470),  # CH2 bending (PE, PP)
        (1050, 1150),  # C-O stretching (PET)
        (990, 1020),   # Ring breathing (PS)
        (1725, 1775)   # C=O stretching (PET, PMMA)
    ]
    
    # Find peaks in the spectrum
    peaks, _ = find_peaks(processed[:, 1], height=threshold)
    peak_wavenumbers = processed[peaks, 0]
    
    # Check if any peaks fall within plastic regions
    for region_start, region_end in plastic_regions:
        region_peaks = [(peak >= region_start) & (peak <= region_end) for peak in peak_wavenumbers]
        if any(region_peaks):
            return True
            
    return False

def process_file(file_path):
    """
    Process a single Raman spectrum file.
    
    Parameters:
    -----------
    file_path : str
        Path to the spectrum file
        
    Returns:
    --------
    tuple
        (file_path, boolean result)
    """
    try:
        # Load spectrum using enhanced file reading
        wavenumbers, intensities = read_raman_spectrum(file_path)
        
        # Check for plastic peaks
        result = has_plastic_peaks(wavenumbers, intensities)
        return (file_path, result)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (file_path, None)

def main(data_directory, output_file='plastic_detection_results.csv', file_ext=None):
    """
    Process all Raman spectra in a directory.
    
    Parameters:
    -----------
    data_directory : str
        Path to directory containing Raman spectra files
    output_file : str
        Path to output CSV file
    file_ext : str or list
        File extension(s) to include. If None, uses ['.txt', '.csv', '.dat']
    """
    # Set default file extensions if none provided
    if file_ext is None:
        file_ext = ['.txt', '.csv', '.dat']
    
    # Read all spectra from the directory
    print(f"Reading spectra from {data_directory}...")
    spectra = batch_read_spectra(data_directory, file_ext=file_ext, verbose=False)  # Set verbose to False
    
    results = []
    
    # Process each spectrum
    print(f"Processing {len(spectra)} spectra...")
    with ProcessPoolExecutor() as executor:
        file_list = [os.path.join(data_directory, filename) for filename in spectra.keys()]
        # Use tqdm with a custom format to show cleaner progress
        for result in tqdm(executor.map(process_file, file_list), 
                         total=len(file_list),
                         desc="Processing spectra",
                         unit="spectra",
                         ncols=80,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                         file=sys.stdout):  # Force output to stdout
            results.append(result)
    
    # Create DataFrame with results
    df = pd.DataFrame(results, columns=['file_path', 'contains_plastic'])
    
    # Save results
    df.to_csv(output_file, index=False)
    
    # Print summary
    positive_count = df['contains_plastic'].sum()
    print(f"\nAnalysis complete. Found {positive_count} of {len(df)} spectra containing plastic signatures.")
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        results_df = main(data_dir)
    else:
        # Example usage
        results_df = main("./example_dir")
    
    # Visualize a few example results
    plastic_files = results_df[results_df['contains_plastic'] == True]['file_path'].tolist()
    non_plastic_files = results_df[results_df['contains_plastic'] == False]['file_path'].tolist()
    
    # Plot some examples
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    if plastic_files:
        for i, file in enumerate(plastic_files[:2]):
            wavenumbers, intensities = read_raman_spectrum(file)
            processed = preprocess_spectrum(wavenumbers, intensities)
            
            ax = axes[0, i]
            ax.plot(processed[:, 0], processed[:, 1])
            ax.set_title(f"Plastic Spectrum Example {i+1}")
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Normalized Intensity")
    
    if non_plastic_files:
        for i, file in enumerate(non_plastic_files[:2]):
            wavenumbers, intensities = read_raman_spectrum(file)
            processed = preprocess_spectrum(wavenumbers, intensities)
            
            ax = axes[1, i]
            ax.plot(processed[:, 0], processed[:, 1])
            ax.set_title(f"Non-Plastic Spectrum Example {i+1}")
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Normalized Intensity")
    
    plt.tight_layout()
    plt.savefig("example_spectra.png")
    plt.show()