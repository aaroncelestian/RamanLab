import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from ml_classifier import RamanSpectraClassifier
import glob
import sys

# Try to import file_reader_utils, but provide fallback if not available
try:
    from file_reader_utils import read_raman_spectrum, batch_read_spectra
    print("Successfully imported file_reader_utils")
except ImportError:
    print("WARNING: file_reader_utils not found, using fallback methods")
    # Define simplified fallback methods (same as in ml_classifier.py)
    def read_raman_spectrum(file_path, verbose=False):
        """Simplified fallback method to read spectra files."""
        if verbose:
            print(f"Reading file: {file_path}")
            
        try:
            # Try numpy's loadtxt with various delimiters
            for delimiter in ['\t', ',', ' ']:
                try:
                    data = np.loadtxt(file_path, delimiter=delimiter)
                    if data.shape[1] >= 2:  # At least two columns
                        return data[:, 0], data[:, 1]  # wavenumbers, intensities
                except Exception:
                    continue
                
            # If all attempts failed, try manual reading
            data_list = []
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty/comment lines
                        continue
                    
                    # Try different delimiters
                    for delimiter in ['\t', ',', ' ']:
                        values = line.split(delimiter)
                        if len(values) >= 2:
                            try:
                                wavenumber = float(values[0].strip())
                                intensity = float(values[1].strip())
                                data_list.append([wavenumber, intensity])
                                break
                            except ValueError:
                                continue
            
            if data_list:
                data = np.array(data_list)
                return data[:, 0], data[:, 1]  # wavenumbers, intensities
            
            raise ValueError(f"Could not parse data from {file_path}")
            
        except Exception as e:
            if verbose:
                print(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def batch_read_spectra(directory, file_ext=None, verbose=False):
        """Simplified fallback method to read multiple spectra files."""
        if file_ext is None:
            file_ext = ['.txt', '.csv', '.dat']
            
        if isinstance(file_ext, str):
            file_ext = [file_ext]
            
        results = {}
        for ext in file_ext:
            for filename in os.listdir(directory):
                if filename.lower().endswith(ext.lower()):
                    file_path = os.path.join(directory, filename)
                    try:
                        wavenumbers, intensities = read_raman_spectrum(file_path, verbose)
                        results[filename] = (wavenumbers, intensities)
                        if verbose:
                            print(f"Successfully read {filename}")
                    except Exception as e:
                        if verbose:
                            print(f"Failed to read {filename}: {str(e)}")
        
        return results

def process_unknown_samples(unknown_dir, output_csv='unknown_spectra_results.csv'):
    """
    Process all unknown samples in the specified directory and save results to CSV.
    
    Parameters:
    -----------
    unknown_dir : str
        Path to directory containing unknown spectra
    output_csv : str
        Path to output CSV file for results
    """
    print(f"Current working directory: {os.getcwd()}")
    print(f"Unknown directory path: {unknown_dir}")
    print(f"Output CSV path: {output_csv}")
    
    # Check if unknown_dir exists
    if not os.path.exists(unknown_dir):
        print(f"ERROR: Directory '{unknown_dir}' does not exist!")
        return False
    
    # List all files with extensions we support
    file_extensions = ['.txt', '.csv', '.dat']
    all_files = []
    for ext in file_extensions:
        files = glob.glob(os.path.join(unknown_dir, f"*{ext}"))
        all_files.extend(files)
    
    print(f"Found {len(all_files)} files with extensions {file_extensions} in {unknown_dir}")
    if len(all_files) == 0:
        print("WARNING: No files found to process!")
        # Create empty directory to place sample files
        print("Creating sample files for testing...")
        try:
            # Create a sample file for testing
            sample_path = os.path.join(unknown_dir, "sample_test.txt")
            with open(sample_path, "w") as f:
                for i in range(100):
                    f.write(f"{i*10}\t{i*0.01}\n")
            print(f"Created sample file at {sample_path}")
        except Exception as e:
            print(f"Error creating sample file: {str(e)}")
    
    # Define path to trained model
    model_path = 'raman_plastic_classifier.joblib'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found. Please train the model first.")
        return False
    
    # Initialize classifier
    print(f"Initializing classifier...")
    classifier = RamanSpectraClassifier(wavenumber_range=(200, 3500), n_points=1000)
    
    # Load trained model
    print(f"Loading trained model from {model_path}...")
    try:
        classifier.load_model(model_path)
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        return False
    
    # Process unknown spectra
    print(f"Processing unknown spectra in {unknown_dir}...")
    try:
        # Process all files in the directory
        results_df = classifier.process_directory(unknown_dir, output_file=output_csv)
        
        # Check if results dataframe is valid
        if results_df is None or len(results_df) == 0:
            print("WARNING: No valid results generated!")
            return False
            
        # Verify that CSV was created
        if not os.path.exists(output_csv):
            print(f"ERROR: Output CSV file not created at {output_csv}")
            
            # Try to save the dataframe directly
            print("Attempting to save results directly...")
            try:
                results_df.to_csv(output_csv, index=False)
                print(f"Successfully saved results to {output_csv}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
                return False
        else:
            print(f"Results saved to {output_csv}")
            
        # Create a visual summary
        print("Creating summary visualization...")
        create_summary_plot(results_df)
        
        return True
        
    except Exception as e:
        print(f"ERROR during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_summary_plot(results_df):
    """
    Create a summary plot of classification results.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results DataFrame with file paths, predictions, and confidence scores
    """
    # Drop rows with missing values
    valid_results = results_df.dropna()
    
    if valid_results.empty:
        print("No valid results to plot.")
        return
    
    # Get class names from the results
    class_names = valid_results['Prediction'].unique()
    if len(class_names) != 2:
        print(f"Warning: Expected 2 classes, found {len(class_names)}")
        return
    
    # Count predictions for each class
    class_counts = valid_results['Prediction'].value_counts()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar chart of class distribution
    ax1.bar(class_counts.index, class_counts.values)
    ax1.set_title('Distribution of Predictions')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Histogram of confidence scores
    ax2.hist(valid_results['Confidence'], bins=20, alpha=0.7)
    ax2.set_title('Distribution of Confidence Scores')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('classification_summary.png')
    print("Summary plot saved to 'classification_summary.png'")
    
    # Print summary statistics
    print("\nClassification Summary:")
    print("-" * 50)
    for class_name in class_names:
        count = class_counts[class_name]
        percentage = (count / len(valid_results)) * 100
        avg_confidence = valid_results[valid_results['Prediction'] == class_name]['Confidence'].mean()
        print(f"{class_name}:")
        print(f"  Count: {count} ({percentage:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.2f}")
    print("-" * 50)

def main():
    """
    Main function to classify unknown Raman spectra.
    """
    print("===== TEST_UNKNOWN.PY STARTING =====")
    print(f"Python version: {sys.version}")
    
    # Define path to unknown spectra
    unknown_dir = "./unknown_dir"
    
    # Process unknown samples
    success = process_unknown_samples(unknown_dir, 'unknown_spectra_results.csv')
    
    if success:
        print("Classification complete. Results saved to 'unknown_spectra_results.csv'")
    else:
        print("Classification failed. Check the error messages above.")
    
    print("===== TEST_UNKNOWN.PY FINISHED =====")

if __name__ == "__main__":
    main()