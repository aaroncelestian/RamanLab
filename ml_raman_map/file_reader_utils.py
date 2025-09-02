import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import chardet

def detect_encoding(file_path):
    """
    Detect the encoding of a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
        
    Returns:
    --------
    str
        Detected encoding
    """
    try:
        # Read a sample of the file to detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            if result['encoding'] is None:
                return 'utf-8'  # Default to utf-8 if detection fails
            return result['encoding']
    except Exception:
        return 'utf-8'  # Default to utf-8 if any error occurs

def detect_delimiter(file_path, encoding='utf-8'):
    """
    Detect the most likely delimiter in a file.
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    encoding : str
        File encoding
        
    Returns:
    --------
    str
        Detected delimiter
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            sample = f.read(2000)  # Read a sample of the file
        
        # Count occurrences of common delimiters
        delimiters = {'\t': 0, ',': 0, ';': 0, ' ': 0}
        for line in sample.split('\n'):
            for delim in delimiters:
                delimiters[delim] += line.count(delim)
        
        # Return the most common delimiter
        max_delim = max(delimiters, key=delimiters.get)
        return max_delim if delimiters[max_delim] > 0 else '\t'  # Default to tab if none found
    except Exception:
        return '\t'  # Default to tab if any error occurs

def read_raman_spectrum(file_path, verbose=False, timeout=10):
    """
    Reads a Raman spectrum file, optimized for various delimited numerical data.
    
    Parameters:
    -----------
    file_path : str
        Path to the spectrum file
    verbose : bool
        Whether to print debug information during reading
    timeout : int
        Maximum time (in seconds) to spend reading a file
        
    Returns:
    --------
    tuple
        (wavenumbers, intensities) as numpy arrays
    
    Raises:
    -------
    ValueError
        If the file cannot be read or processed
    """
    if verbose:
        print(f"\nReading file: {file_path}")
    
    def read_with_timeout():
        # Store all exceptions to provide better error messages
        all_exceptions = []
        
        # First attempt: detect encoding and delimiter automatically
        try:
            encoding = detect_encoding(file_path)
            delimiter = detect_delimiter(file_path, encoding)
            
            if verbose:
                print(f"Detected encoding: {encoding}, delimiter: {delimiter}")
            
            # Try pandas first
            try:
                df = pd.read_csv(file_path, 
                              sep=delimiter,
                              header=None,
                              encoding=encoding,
                              on_bad_lines='skip',
                              engine='python')  # Use python engine for better flexibility
                
                if df.shape[1] >= 2:
                    data = df.iloc[:, 0:2].values
                    if verbose:
                        print(f"Successfully read with pandas, shape: {data.shape}")
                    return data
                else:
                    err = f"Data has insufficient columns: {df.shape[1]}"
                    all_exceptions.append(f"Pandas (auto): {err}")
                    if verbose:
                        print(err)
                    raise ValueError(err)
            except Exception as e:
                err = f"Pandas reading failed with auto-detected parameters: {str(e)}"
                all_exceptions.append(err)
                if verbose:
                    print(err)
                # Continue to next method
        except Exception as e:
            err = f"Auto-detection failed: {str(e)}"
            all_exceptions.append(err)
            if verbose:
                print(err)
            # Continue to next method
        
        # Second attempt: try multiple explicit delimiter and encoding combinations with pandas
        delimiters = ['\t', ',', ';', ' ']
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16', 'utf-16le', 'utf-16be']
        
        for enc in encodings:
            for delim in delimiters:
                try:
                    df = pd.read_csv(file_path, 
                                  sep=delim, 
                                  header=None, 
                                  encoding=enc,
                                  on_bad_lines='skip',
                                  engine='python')
                    
                    if df.shape[1] >= 2:
                        data = df.iloc[:, 0:2].values
                        if verbose:
                            print(f"Successfully read with pandas using {enc} encoding and {delim} delimiter, shape: {data.shape}")
                        return data
                except Exception as e:
                    if verbose:
                        print(f"Pandas reading failed with {enc}/{delim}: {str(e)}")
                    all_exceptions.append(f"Pandas ({enc}/{delim}): {str(e)}")
                    continue
        
        # Third attempt: try numpy with different delimiters and encodings
        for enc in encodings:
            for delim in delimiters:
                try:
                    data = np.loadtxt(file_path, delimiter=delim, encoding=enc)
                    if data.shape[1] >= 2:
                        if verbose:
                            print(f"Successfully read with numpy using {enc} encoding and {delim} delimiter, shape: {data.shape}")
                        return data
                except Exception as e:
                    if verbose:
                        print(f"Numpy reading failed with {enc}/{delim}: {str(e)}")
                    all_exceptions.append(f"Numpy ({enc}/{delim}): {str(e)}")
                    continue
        
        # Final fallback: try with manual processing
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    lines = f.readlines()
                
                # Extract numeric values from lines
                data_points = []
                for line in lines:
                    # Skip empty lines and comments
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try to extract numeric values
                    numbers = []
                    for part in line.replace(',', ' ').replace(';', ' ').replace('\t', ' ').split():
                        try:
                            numbers.append(float(part))
                        except ValueError:
                            continue
                    
                    if len(numbers) >= 2:
                        data_points.append(numbers[:2])  # Take first two numbers as X, Y
                
                if data_points:
                    data = np.array(data_points)
                    if verbose:
                        print(f"Successfully read with manual processing using {enc} encoding, shape: {data.shape}")
                    return data
            except Exception as e:
                if verbose:
                    print(f"Manual processing failed with {enc} encoding: {str(e)}")
                all_exceptions.append(f"Manual ({enc}): {str(e)}")
                continue
        
        # If we get here, all methods have failed
        raise ValueError(f"Failed to read file with all methods. Errors: {'; '.join(all_exceptions)}")
    
    # Use ThreadPoolExecutor to implement timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read_with_timeout)
        try:
            data = future.result(timeout=timeout)
            
            # Verify that we got a numpy array with the right shape
            if not isinstance(data, np.ndarray) or len(data.shape) != 2 or data.shape[1] < 2:
                raise ValueError(f"Invalid data format: expected 2D array with at least 2 columns, got {type(data)} with shape {getattr(data, 'shape', 'unknown')}")
            
            # Extract wavenumbers and intensities
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
            
            # Ensure wavenumbers are in ascending order
            if not np.all(np.diff(wavenumbers) >= 0):
                sort_idx = np.argsort(wavenumbers)
                wavenumbers = wavenumbers[sort_idx]
                intensities = intensities[sort_idx]
                if verbose:
                    print("Sorted wavenumbers in ascending order")
            
            # Verify no NaN or inf values
            if np.any(np.isnan(wavenumbers)) or np.any(np.isnan(intensities)) or \
               np.any(np.isinf(wavenumbers)) or np.any(np.isinf(intensities)):
                raise ValueError("Data contains NaN or infinite values")
            
            return wavenumbers, intensities
            
        except TimeoutError:
            raise ValueError(f"Reading file {file_path} timed out after {timeout} seconds")
        except Exception as e:
            # Re-raise with more informative message
            raise ValueError(f"Failed to read spectrum data from {file_path}: {str(e)}")

def batch_read_spectra(directory, file_ext=None, verbose=False, max_workers=4):
    """
    Read all Raman spectra files in a directory.
    
    Parameters:
    -----------
    directory : str
        Path to directory containing Raman spectra files
    file_ext : str or list
        File extension(s) to include. If None, uses ['.txt', '.csv', '.dat']
    verbose : bool
        Whether to print detailed progress information
    max_workers : int
        Maximum number of parallel workers for reading files
        
    Returns:
    --------
    dict
        Dictionary mapping filenames to (wavenumbers, intensities) tuples
    """
    if file_ext is None:
        file_ext = ['.txt', '.csv', '.dat']
    elif isinstance(file_ext, str):
        file_ext = [file_ext]
    
    # Get all files with specified extensions, filtering out macOS metadata files
    all_files = []
    for ext in file_ext:
        all_files.extend([os.path.join(directory, f) for f in os.listdir(directory) 
                         if f.lower().endswith(ext.lower()) and not f.startswith('._')])
    
    if verbose:
        print(f"\nFound {len(all_files)} files to process in {directory}")
    
    # Read all spectra
    spectra = {}
    error_files = []
    
    def process_file(file_path):
        try:
            wavenumbers, intensities = read_raman_spectrum(file_path, verbose=verbose)
            return os.path.basename(file_path), (wavenumbers, intensities), None
        except Exception as e:
            return os.path.basename(file_path), None, str(e)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in all_files]
        
        # Create progress bar with custom format
        pbar = tqdm(total=len(all_files), 
                   desc="Progress",
                   unit="files",
                   ncols=80,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for future in futures:
            try:
                file_name, result, error = future.result()
                if error is None:
                    spectra[file_name] = result
                    if verbose:
                        print(f"\n✓ Successfully read {file_name}")
                else:
                    error_files.append((file_name, error))
                    if verbose:
                        print(f"\n✗ Error reading {file_name}: {error}")
            except Exception as e:
                if verbose:
                    print(f"\n✗ Unexpected error: {str(e)}")
            finally:
                pbar.update(1)
        
        pbar.close()
    
    if verbose:
        print(f"\nSummary:")
        print(f"✓ Successfully read: {len(spectra)} files")
        if error_files:
            print(f"✗ Failed to read: {len(error_files)} files")
            print("\nFailed files:")
            for file_name, error in error_files:
                print(f"✗ {file_name}: {error}")
    
    return spectra