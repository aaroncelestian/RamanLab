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
        # First try with glob pattern
        pattern = os.path.join(data_directory, f"*{ext}")
        found_files = glob.glob(pattern)
        print(f"Found {len(found_files)} files with pattern {pattern}")
        
        # If that didn't work, try direct listing
        if not found_files:
            print(f"Trying direct directory listing for {ext}...")
            try:
                dir_files = os.listdir(data_directory)
                found_files = [os.path.join(data_directory, f) for f in dir_files if f.lower().endswith(ext.lower())]
                print(f"Found {len(found_files)} files with extension {ext} through direct listing")
            except Exception as e:
                print(f"Error listing directory: {str(e)}")
        
        all_files.extend(found_files)
    
    # Report the total number of files found, but don't limit
    print(f"Found a total of {len(all_files)} files to process")
    
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
    
    start_time = time.time()
    last_update_time = start_time
    
    for file_path in all_files:
        try:
            current_time = time.time()
            # Only print progress occasionally to avoid flooding the console
            # Print every 10 files OR every 5 seconds, whichever comes first
            if processed_count % 10 == 0 or (current_time - last_update_time) >= 5:
                elapsed = current_time - start_time
                if processed_count > 0:
                    avg_time_per_file = elapsed / processed_count
                    remaining_files = len(all_files) - processed_count
                    est_remaining_time = remaining_files * avg_time_per_file
                    print(f"Processing file {processed_count+1}/{len(all_files)}: {os.path.basename(file_path)}")
                    print(f"  Progress: {processed_count/len(all_files)*100:.1f}% complete")
                    print(f"  Elapsed time: {elapsed:.1f} seconds")
                    print(f"  Estimated remaining time: {est_remaining_time:.1f} seconds ({est_remaining_time/60:.1f} minutes)")
                else:
                    print(f"Processing file {processed_count+1}/{len(all_files)}: {os.path.basename(file_path)}")
                
                last_update_time = current_time
            
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
            # Save more frequently for large datasets
            save_interval = min(100, max(10, len(all_files) // 20))  # Save at least every 10, at most every 100
            if processed_count % save_interval == 0:
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