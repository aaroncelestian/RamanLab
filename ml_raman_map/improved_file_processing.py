def process_directory(self, data_directory, output_file='plastic_detection_results.csv', file_ext=None):
    """
    Process all spectrum files in a directory with improved error handling.
    
    Parameters:
    -----------
    data_directory : str
        Path to directory containing spectrum files
    output_file : str
        Path to output CSV file
    file_ext : str or list
        File extension(s) of spectrum files. If None, uses ['.txt', '.csv', '.dat']
            
    Returns:
    --------
    DataFrame
        Results with file paths, predictions, and confidence scores
    """
    import os
    import glob
    import pandas as pd
    import time
    
    print(f"Processing directory: {data_directory}")
    print(f"Will save results to: {output_file}")
    print(f"Current working directory: {os.getcwd()}")
    
    if self.model is None:
        print("ERROR: Model not trained. Please train or load a model first.")
        raise ValueError("Model not trained. Please train or load a model first.")
    
    # Set default file extensions if none provided
    if file_ext is None:
        file_ext = ['.txt', '.csv', '.dat']
    
    if isinstance(file_ext, str):
        file_ext = [file_ext]
    
    # IMPROVED FILE FINDING STRATEGY
    print("Scanning for files...")
    all_files = []
    for ext in file_ext:
        # First try with glob pattern, filtering out macOS metadata files
        pattern = os.path.join(data_directory, f"*{ext}")
        found_files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith('._')]
        print(f"Found {len(found_files)} files with pattern {pattern} (excluding ._* files)")
        
        # If that didn't work, try direct listing
        if not found_files:
            print(f"Trying direct directory listing for {ext}...")
            try:
                dir_files = os.listdir(data_directory)
                found_files = [os.path.join(data_directory, f) for f in dir_files 
                             if f.lower().endswith(ext.lower()) and not f.startswith('._')]
                print(f"Found {len(found_files)} files with extension {ext} through direct listing (excluding ._* files)")
            except Exception as e:
                print(f"Error listing directory: {str(e)}")
        
        all_files.extend(found_files)
    
    # Validate the file list - limit to reasonable number for testing
    if len(all_files) > 1000:
        print(f"WARNING: Found {len(all_files)} files - this seems excessive. Limiting to first 1000 for safety.")
        all_files = all_files[:1000]
    
    if not all_files:
        print(f"WARNING: No files found in {data_directory} with extensions {file_ext}")
        print(f"Files in directory: {os.listdir(data_directory) if os.path.exists(data_directory) else 'DIRECTORY NOT FOUND'}")
        # Return empty DataFrame
        return pd.DataFrame(columns=['file_path', 'is_plastic', 'confidence'])
    
    # Now process files ONE BY ONE instead of using batch_read_spectra
    # This gives us better control and error reporting
    print(f"Processing {len(all_files)} files one by one...")
    
    results = []
    processed_count = 0
    error_count = 0
    
    for file_path in all_files:
        try:
            # Only print progress occasionally to avoid flooding the console
            if processed_count % 10 == 0:
                print(f"Processing file {processed_count+1}/{len(all_files)}: {os.path.basename(file_path)}")
            
            # Read the file
            try:
                # Try numpy's loadtxt with various delimiters
                for delimiter in ['\t', ',', ' ']:
                    try:
                        import numpy as np
                        data = np.loadtxt(file_path, delimiter=delimiter)
                        if data.shape[1] >= 2:  # At least two columns
                            wavenumbers = data[:, 0]
                            intensities = data[:, 1]
                            break
                    except Exception:
                        if delimiter == ' ':  # Last attempt failed
                            raise ValueError(f"Could not parse data with any delimiter from {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                error_count += 1
                continue
            
            # Preprocess
            processed = self.preprocess_spectrum(wavenumbers, intensities)
            X = processed.reshape(1, -1)
            
            # Predict
            is_plastic = bool(self.model.predict(X)[0])
            
            # Get probability score (confidence)
            proba = self.model.predict_proba(X)[0]
            confidence = proba[1] if is_plastic else proba[0]
            
            results.append((file_path, is_plastic, confidence))
            processed_count += 1
            
            # Periodically save results to avoid losing progress
            if processed_count % 100 == 0:
                temp_df = pd.DataFrame(results, columns=['file_path', 'is_plastic', 'confidence'])
                temp_df.to_csv(f"{output_file}.temp", index=False)
                print(f"Saved intermediate results ({processed_count} files processed)")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            error_count += 1
    
    print(f"Processing complete. Successfully processed {processed_count} files with {error_count} errors.")
    
    # Create DataFrame with results
    df = pd.DataFrame(results, columns=['file_path', 'is_plastic', 'confidence'])
    
    # Save results
    print(f"Saving results to {output_file}...")
    try:
        df.to_csv(output_file, index=False)
        print(f"Results saved successfully to {output_file}")
        
        # Double-check file was created
        if os.path.exists(output_file):
            print(f"Verified that {output_file} exists")
            file_size = os.path.getsize(output_file)
            print(f"File size: {file_size} bytes")
        else:
            print(f"WARNING: {output_file} was not created!")
    except Exception as e:
        print(f"Error saving results to CSV: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    valid_results = df.dropna()
    if not valid_results.empty:
        positive_count = valid_results['is_plastic'].sum()
        total_count = len(valid_results)
        print(f"Analysis complete. Found {positive_count} of {total_count} spectra classified as plastic.")
    
    return df