import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from joblib import dump, load
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import traceback

# Import our enhanced file reading utilities
try:
    from file_reader_utils import read_raman_spectrum, batch_read_spectra
    print("Successfully imported file_reader_utils")
except ImportError:
    print("WARNING: file_reader_utils module not found. Using built-in methods.")
    
    # Define simplified fallback methods
    def read_raman_spectrum(file_path, verbose=False):
        """
        Reads a Raman spectrum file, optimized for tab-delimited numerical data.
        
        Parameters:
        -----------
        file_path : str
            Path to the spectrum file
        verbose : bool
            Whether to print debug information during reading
            
        Returns:
        --------
        tuple
            (wavenumbers, intensities) as numpy arrays
        """
        if verbose:
            print(f"\nReading file: {file_path}")
        
        try:
            # First try with numpy's loadtxt since we know these are tab-delimited
            data = np.loadtxt(file_path, delimiter='\t', encoding='utf-8')
            if verbose:
                print(f"Successfully read with numpy using tab delimiter, shape: {data.shape}")
        except Exception as e:
            if verbose:
                print(f"Numpy reading with tab delimiter failed: {str(e)}")
            
            # If that fails, try pandas with explicit encoding
            try:
                df = pd.read_csv(file_path, 
                               sep='\t',  # Explicitly use tab delimiter
                               header=None,
                               encoding='utf-8',
                               on_bad_lines='warn' if verbose else 'skip')
                if df.shape[1] >= 2:
                    data = df.iloc[:, 0:2].values
                    if verbose:
                        print(f"Successfully read with pandas, shape: {data.shape}")
                else:
                    raise ValueError(f"Data has insufficient columns: {df.shape[1]}")
            except Exception as e:
                if verbose:
                    print(f"Pandas reading failed: {str(e)}")
                raise ValueError(f"Could not read spectrum data from {file_path}")
        
        # Extract wavenumbers and intensities
        if data.shape[1] >= 2:
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
            
            # Ensure they are in ascending order
            if not np.all(np.diff(wavenumbers) >= 0):
                sort_idx = np.argsort(wavenumbers)
                wavenumbers = wavenumbers[sort_idx]
                intensities = intensities[sort_idx]
                if verbose:
                    print("Sorted wavenumbers in ascending order")
            
            return wavenumbers, intensities
        else:
            raise ValueError(f"Not enough columns in data from {file_path}")
    
    def batch_read_spectra(directory, file_ext=None, verbose=True):
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
            
        Returns:
        --------
        dict
            Dictionary mapping filenames to (wavenumbers, intensities) tuples
        """
        if file_ext is None:
            file_ext = ['.txt', '.csv', '.dat']
        elif isinstance(file_ext, str):
            file_ext = [file_ext]
        
        # Get all files with specified extensions
        all_files = []
        for ext in file_ext:
            all_files.extend([os.path.join(directory, f) for f in os.listdir(directory) 
                             if f.endswith(ext)])
        
        if verbose:
            print(f"\nFound {len(all_files)} files to process in {directory}")
        
        # Read all spectra
        spectra = {}
        error_files = []
        for file_path in tqdm(all_files, desc="Reading files", unit="files", 
                             disable=not verbose, ncols=80,
                             bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            try:
                wavenumbers, intensities = read_raman_spectrum(file_path, verbose=verbose)
                spectra[os.path.basename(file_path)] = (wavenumbers, intensities)
                if verbose:
                    print(f"Successfully read {os.path.basename(file_path)}")
            except Exception as e:
                error_files.append((os.path.basename(file_path), str(e)))
                if verbose:
                    print(f"Error reading {os.path.basename(file_path)}: {e}")
        
        if verbose:
            print(f"\nSuccessfully read {len(spectra)} files")
            if error_files:
                print("\nFiles with errors:")
                for file_name, error in error_files:
                    print(f"- {file_name}: {error}")
        
        return spectra

class RamanSpectraClassifier:
    def __init__(self, wavenumber_range=(200, 3500), n_points=1000, 
                 positive_class_name="Class A", negative_class_name="Class B",
                 model_name="raman_classifier"):
        """
        Initialize the Raman spectra classifier.
        
        Parameters:
        -----------
        wavenumber_range : tuple
            Range of wavenumbers to consider (cm^-1)
        n_points : int
            Number of points to interpolate spectra to
        positive_class_name : str
            Name of the positive class (e.g., "Plastic", "Material A")
        negative_class_name : str
            Name of the negative class (e.g., "Non-plastic", "Material B")
        model_name : str
            Base name for saved model and output files
        """
        self.wavenumber_range = wavenumber_range
        self.n_points = n_points
        self.wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)
        self.model = None
        self.positive_class_name = positive_class_name
        self.negative_class_name = negative_class_name
        self.model_name = model_name
        print(f"Initialized classifier with {n_points} points from {wavenumber_range[0]} to {wavenumber_range[1]} cm^-1")
        print(f"Classifying between {positive_class_name} and {negative_class_name}")
        
    def preprocess_spectrum(self, original_wavenumbers, intensities):
        """
        Preprocess a single Raman spectrum.
        
        Parameters:
        -----------
        original_wavenumbers : ndarray
            Original Raman shift values (cm^-1)
        intensities : ndarray
            Original intensity values
            
        Returns:
        --------
        ndarray
            Preprocessed spectrum interpolated to standard wavenumber range
        """
        # Apply Savitzky-Golay filter for smoothing
        if len(intensities) > 11:  # Ensure spectrum has enough points for smoothing
            smoothed = savgol_filter(intensities, window_length=11, polyorder=3)
        else:
            smoothed = intensities
            
        # Baseline correction (can be enhanced with more advanced methods)
        # Simple method: subtract minimum and normalize
        baseline_corrected = smoothed - np.percentile(smoothed, 5)
        
        # Handle cases where spectrum might be all zeros after correction
        if np.max(baseline_corrected) > 0:
            normalized = baseline_corrected / np.max(baseline_corrected)
        else:
            normalized = baseline_corrected
            
        # Interpolate to standard wavenumber range
        # First, ensure wavenumbers are in ascending order
        sort_idx = np.argsort(original_wavenumbers)
        sorted_wavenumbers = original_wavenumbers[sort_idx]
        sorted_intensities = normalized[sort_idx]
        
        # Create interpolation function
        # Use bounds_error=False to handle wavenumbers outside our target range
        f = interp1d(sorted_wavenumbers, sorted_intensities, bounds_error=False, fill_value=0)
        
        # Apply interpolation to standard wavenumber grid
        standardized = f(self.wavenumbers)
        
        return standardized
    
    def load_and_preprocess_data(self, positive_dir, negative_dir, file_ext=None, verbose=True):
        """
        Load and preprocess spectra from positive and negative class directories.
        
        Parameters:
        -----------
        positive_dir : str
            Directory containing positive class Raman spectra
        negative_dir : str
            Directory containing negative class Raman spectra
        file_ext : str or list
            File extension(s) of spectrum files. If None, uses ['.txt', '.csv', '.dat']
        verbose : bool
            Whether to print detailed progress information
            
        Returns:
        --------
        X : ndarray
            Features (preprocessed spectra)
        y : ndarray
            Labels (1 for positive class, 0 for negative class)
        """
        print(f"Current working directory: {os.getcwd()}")
        print(f"Loading data from {self.positive_class_name} directory: {positive_dir}")
        print(f"Loading data from {self.negative_class_name} directory: {negative_dir}")
        
        # Validate directories exist
        if not os.path.exists(positive_dir):
            print(f"{self.positive_class_name} directory does not exist: {positive_dir}")
            raise ValueError(f"{self.positive_class_name} directory does not exist: {positive_dir}")
        if not os.path.exists(negative_dir):
            print(f"{self.negative_class_name} directory does not exist: {negative_dir}")
            raise ValueError(f"{self.negative_class_name} directory does not exist: {negative_dir}")

        # Set default file extensions if none provided
        if file_ext is None:
            file_ext = ['.txt', '.csv', '.dat']
            
        # List files in directories to verify
        positive_files = []
        for ext in file_ext:
            positive_files.extend([os.path.join(positive_dir, f) for f in os.listdir(positive_dir) 
                               if f.lower().endswith(ext.lower())])
        print(f"Found {len(positive_files)} {self.positive_class_name} files")
        
        negative_files = []
        for ext in file_ext:
            negative_files.extend([os.path.join(negative_dir, f) for f in os.listdir(negative_dir) 
                                if f.lower().endswith(ext.lower())])
        print(f"Found {len(negative_files)} {self.negative_class_name} files")
        
        if not positive_files:
            print(f"No valid files found in {self.positive_class_name} directory: {positive_dir}")
            raise ValueError(f"No valid files found in {self.positive_class_name} directory: {positive_dir}")
        if not negative_files:
            print(f"No valid files found in {self.negative_class_name} directory: {negative_dir}")
            raise ValueError(f"No valid files found in {self.negative_class_name} directory: {negative_dir}")
        
        # Read all spectra from both directories
        print(f"Reading {self.positive_class_name} spectra from {positive_dir}...")
        try:
            positive_spectra = batch_read_spectra(positive_dir, file_ext=file_ext, verbose=verbose)
        except Exception as e:
            print(f"ERROR: Failed to read {self.positive_class_name} spectra: {str(e)}")
            raise
        
        print(f"Reading {self.negative_class_name} spectra from {negative_dir}...")
        try:
            negative_spectra = batch_read_spectra(negative_dir, file_ext=file_ext, verbose=verbose)
        except Exception as e:
            print(f"ERROR: Failed to read {self.negative_class_name} spectra: {str(e)}")
            raise
        
        # Handle case where no spectra were successfully read
        if not positive_spectra:
            raise ValueError(f"No {self.positive_class_name} spectra could be read from {positive_dir}")
        if not negative_spectra:
            raise ValueError(f"No {self.negative_class_name} spectra could be read from {negative_dir}")
        
        # Prepare data arrays
        X = []
        y = []
        
        # Process positive class spectra
        print(f"Preprocessing {self.positive_class_name} spectra...")
        for filename, (wavenumbers, intensities) in positive_spectra.items():
            try:
                # Preprocess spectrum
                processed = self.preprocess_spectrum(wavenumbers, intensities)
                
                # Validate processed spectrum
                if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                    print(f"Warning: Invalid processed spectrum for {filename}, skipping")
                    continue
                
                X.append(processed)
                y.append(1)  # 1 for positive class
            except Exception as e:
                print(f"Error preprocessing {self.positive_class_name} file {filename}: {str(e)}")
        
        # Process negative class spectra
        print(f"Preprocessing {self.negative_class_name} spectra...")
        for filename, (wavenumbers, intensities) in negative_spectra.items():
            try:
                # Preprocess spectrum
                processed = self.preprocess_spectrum(wavenumbers, intensities)
                
                # Validate processed spectrum
                if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                    print(f"Warning: Invalid processed spectrum for {filename}, skipping")
                    continue
                
                X.append(processed)
                y.append(0)  # 0 for negative class
            except Exception as e:
                print(f"Error preprocessing {self.negative_class_name} file {filename}: {str(e)}")
        
        if not X:
            raise ValueError("No data was successfully loaded from the files")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nSuccessfully processed {len(X)} spectra:")
        print(f"- {self.positive_class_name} spectra: {np.sum(y == 1)}")
        print(f"- {self.negative_class_name} spectra: {np.sum(y == 0)}")
        
        return X, y
    
    def train(self, X, y):
        """
        Train the classifier model.
        
        Parameters:
        -----------
        X : ndarray
            Features (preprocessed spectra)
        y : ndarray
            Labels (1 for positive class, 0 for negative class)
        """
        print(f"Training model with {len(X)} samples")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipeline with feature scaling and classifier
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_jobs=-1, random_state=42))
        ])
        
        # Define hyperparameter grid
        param_grid = {
            'classifier__n_estimators': [100, 200],  # Reduced for faster execution
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2, 5]
        }
        
        # Perform grid search with cross-validation
        print("Performing grid search...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Set the best model
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        
        print("Validation Accuracy:", accuracy_score(y_val, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        
        # Feature importance analysis (useful for understanding which wavenumbers are most discriminative)
        if hasattr(self.model['classifier'], 'feature_importances_'):
            importances = self.model['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(10), importances[indices[:10]])
            plt.xticks(range(10), [f"{self.wavenumbers[i]:.0f} cm⁻¹" for i in indices[:10]])
            plt.xlabel('Wavenumber')
            plt.ylabel('Feature Importance')
            plt.title('Top 10 Most Important Wavenumber Regions')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("Feature importance plot saved to 'feature_importance.png'")
            
            # Print most important wavenumber regions
            print("\nTop 10 Important Wavenumber Regions:")
            for i in range(10):
                print(f"{i+1}. {self.wavenumbers[indices[i]]:.0f} cm⁻¹ (Importance: {importances[indices[i]]:.4f})")
    
    def save_model(self, file_path=None):
        """Save the trained model to a file."""
        if file_path is None:
            file_path = f"{self.model_name}.joblib"
        print(f"Attempting to save model to {file_path}")
        if self.model is not None:
            try:
                # Try to save the model
                dump(self.model, file_path)
                print(f"Model saved to {file_path}")
            except Exception as e:
                print(f"Error saving model: {str(e)}")
        else:
            print("No trained model to save.")
    
    def load_model(self, file_path=None):
        """Load a trained model from a file."""
        if file_path is None:
            file_path = f"{self.model_name}.joblib"
        self.model = load(file_path)
        print(f"Model loaded from {file_path}")
    
    def predict(self, wavenumbers, intensities):
        """
        Predict whether a spectrum corresponds to plastic.
        
        Parameters:
        -----------
        wavenumbers : ndarray
            Raman shift values (cm^-1)
        intensities : ndarray
            Intensity values
            
        Returns:
        --------
        bool
            True if predicted as plastic, False otherwise
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train or load a model first.")
        
        # Preprocess the spectrum
        processed = self.preprocess_spectrum(wavenumbers, intensities)
        
        # Reshape for prediction (single sample)
        X = processed.reshape(1, -1)
        
        # Predict and return boolean result
        return bool(self.model.predict(X)[0])
    
    def process_file(self, file_path):
        """
        Process a single spectrum file.
        
        Parameters:
        -----------
        file_path : str
            Path to the spectrum file
            
        Returns:
        --------
        tuple
            (file_path, prediction result, confidence score)
        """
        try:
            # Load spectrum using our enhanced reader
            wavenumbers, intensities = read_raman_spectrum(file_path)
            
            # Preprocess
            processed = self.preprocess_spectrum(wavenumbers, intensities)
            X = processed.reshape(1, -1)
            
            # Predict
            is_positive = bool(self.model.predict(X)[0])
            
            # Get probability score (confidence)
            proba = self.model.predict_proba(X)[0]
            confidence = proba[1] if is_positive else proba[0]
            
            return (file_path, self.positive_class_name if is_positive else self.negative_class_name, confidence)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return (file_path, None, None)
    
    def process_directory(self, data_directory, output_file=None, file_ext=None):
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
        if output_file is None:
            output_file = f"{self.model_name}_results.csv"
            
        print(f"\nProcessing directory: {data_directory}")
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
        
        # Get all files with specified extensions
        all_files = []
        for ext in file_ext:
            all_files.extend([os.path.join(data_directory, f) for f in os.listdir(data_directory) 
                           if f.lower().endswith(ext.lower())])
        
        if not all_files:
            print(f"No valid files found in directory: {data_directory}")
            return pd.DataFrame()
        
        print(f"Found {len(all_files)} files to process")
        
        # Process all files with improved progress bar
        results = []
        with tqdm(total=len(all_files), desc="Processing files", unit="files", 
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for file_path in all_files:
                try:
                    # Load spectrum using our enhanced reader
                    wavenumbers, intensities = read_raman_spectrum(file_path)
                    
                    # Preprocess
                    processed = self.preprocess_spectrum(wavenumbers, intensities)
                    X = processed.reshape(1, -1)
                    
                    # Predict
                    is_positive = bool(self.model.predict(X)[0])
                    
                    # Get probability score (confidence)
                    proba = self.model.predict_proba(X)[0]
                    confidence = proba[1] if is_positive else proba[0]
                    
                    # Add to results
                    results.append({
                        'File': os.path.basename(file_path),
                        'Prediction': self.positive_class_name if is_positive else self.negative_class_name,
                        'Confidence': confidence
                    })
                    
                except Exception as e:
                    print(f"\nError processing {file_path}: {str(e)}")
                    results.append({
                        'File': os.path.basename(file_path),
                        'Prediction': 'Error',
                        'Confidence': None
                    })
                
                pbar.update(1)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        success_count = len([r for r in results if r['Prediction'] != 'Error'])
        error_count = len([r for r in results if r['Prediction'] == 'Error'])
        print(f"\nProcessing complete:")
        print(f"- Successfully processed: {success_count} files")
        print(f"- Errors encountered: {error_count} files")
        
        return df


def visualize_model_performance(classifier, X_val, y_val, n_samples=5):
    """
    Visualize model performance on validation samples.
    
    Parameters:
    -----------
    classifier : RamanSpectraClassifier
        Trained classifier
    X_val : ndarray
        Validation features (preprocessed spectra)
    y_val : ndarray
        Validation labels
    n_samples : int
        Number of samples to visualize
    """
    # Get predictions
    y_pred = classifier.model.predict(X_val)
    
    # Calculate prediction probabilities
    y_proba = classifier.model.predict_proba(X_val)
    
    # Find correct and incorrect predictions
    correct = np.where(y_pred == y_val)[0]
    incorrect = np.where(y_pred != y_val)[0]
    
    # Prepare figure
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
    
    # Plot some correct predictions
    correct_samples = np.random.choice(correct, min(n_samples, len(correct)), replace=False)
    for i, idx in enumerate(correct_samples):
        conf = y_proba[idx, y_pred[idx]]
        true_label = "Plastic" if y_val[idx] == 1 else "Non-plastic"
        axes[0, i].plot(classifier.wavenumbers, X_val[idx])
        axes[0, i].set_title(f"Correct: {true_label}\nConf: {conf:.2f}")
        axes[0, i].set_xlabel("Wavenumber (cm⁻¹)")
        if i == 0:
            axes[0, i].set_ylabel("Intensity (a.u.)")
    
    # Plot some incorrect predictions (if any)
    if len(incorrect) > 0:
        incorrect_samples = np.random.choice(incorrect, min(n_samples, len(incorrect)), replace=False)
        for i, idx in enumerate(incorrect_samples):
            if i >= n_samples:
                break
            conf = y_proba[idx, y_pred[idx]]
            true_label = "Plastic" if y_val[idx] == 1 else "Non-plastic"
            pred_label = "Plastic" if y_pred[idx] == 1 else "Non-plastic"
            axes[1, i].plot(classifier.wavenumbers, X_val[idx])
            axes[1, i].set_title(f"Incorrect: {true_label} as {pred_label}\nConf: {conf:.2f}")
            axes[1, i].set_xlabel("Wavenumber (cm⁻¹)")
            if i == 0:
                axes[1, i].set_ylabel("Intensity (a.u.)")
    
    # If no incorrect predictions, show more correct ones
    if len(incorrect) == 0:
        more_correct = np.random.choice(correct, min(n_samples, len(correct)), replace=False)
        for i, idx in enumerate(more_correct):
            conf = y_proba[idx, y_pred[idx]]
            true_label = "Plastic" if y_val[idx] == 1 else "Non-plastic"
            axes[1, i].plot(classifier.wavenumbers, X_val[idx])
            axes[1, i].set_title(f"Correct: {true_label}\nConf: {conf:.2f}")
            axes[1, i].set_xlabel("Wavenumber (cm⁻¹)")
            if i == 0:
                axes[1, i].set_ylabel("Intensity (a.u.)")
    
    plt.tight_layout()
    plt.savefig('model_performance_visualization.png')
    print("Performance visualization saved to 'model_performance_visualization.png'")


def main():
    """Main function to demonstrate the workflow."""
    print(f"Starting main() function in {os.getcwd()}")
    
    # Initialize classifier
    classifier = RamanSpectraClassifier(wavenumber_range=(200, 3500), n_points=1000)
    
    # Define paths to training data
    positive_dir = "./positive_dir"
    negative_dir = "./negative_dir"
    
    # Check if directories exist
    print("Checking required directories...")
    dirs_exist = True
    for dirname in [positive_dir, negative_dir]:
        if not os.path.exists(dirname):
            print(f"Directory not found: {dirname}")
            dirs_exist = False
    
    if not dirs_exist:
        print("Creating required directories...")
        for dirname in [positive_dir, negative_dir, "./unknown_dir"]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                print(f"Created directory: {dirname}")
        
        print("\nPlease place your spectra files in the following directories:")
        print(f"- {positive_dir}: Known {classifier.positive_class_name} spectra")
        print(f"- {negative_dir}: Known {classifier.negative_class_name} spectra")
        print("Then run this script again.")
        return
    
    # Load and preprocess training data
    print("Loading and preprocessing training data...")
    try:
        X, y = classifier.load_and_preprocess_data(positive_dir, negative_dir, verbose=True)
    except Exception as e:
        print(f"Error loading and preprocessing data: {str(e)}")
        return
    
    # Train the model
    print("Training the model...")
    try:
        classifier.train(X, y)
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return
    
    # Save the trained model
    classifier.save_model()
    
    # Split data to get validation set for visualization
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Visualize model performance
    print("Creating model performance visualization...")
    try:
        visualize_model_performance(classifier, X_val, y_val)
    except Exception as e:
        print(f"Error visualizing model performance: {str(e)}")
    
    print("Processing complete. You can now use the trained model to classify unknown spectra.")


if __name__ == "__main__":
    print("Script started")
    main()
    print("Script finished")