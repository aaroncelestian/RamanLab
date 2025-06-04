import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from pre_processing import preprocess_spectrum
from file_reader_utils import read_raman_spectrum, batch_read_spectra

class RamanPreprocessor:
    def __init__(self, target_wavenumbers=None):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        target_wavenumbers : ndarray, optional
            Target wavenumber values for resampling. If None, a default range is used.
        """
        if target_wavenumbers is None:
            # Default wavenumber range from 100 to 3500 cm^-1 with 400 points
            self.target_wavenumbers = np.linspace(100, 3500, 400)
        else:
            self.target_wavenumbers = target_wavenumbers
        
    def process_file(self, file_path):
        """
        Process a single Raman spectrum file with enhanced error handling.
        """
        try:
            # Load spectrum using enhanced file reading
            wavenumbers, intensities = read_raman_spectrum(file_path)
            
            # Verify data integrity before preprocessing
            if wavenumbers is None or intensities is None:
                raise ValueError("read_raman_spectrum returned None values")
                
            if len(wavenumbers) < 5:  # Not enough data points for reliable analysis
                raise ValueError(f"Insufficient data points: {len(wavenumbers)}")
                
            # Preprocess the spectrum with resampling
            processed = preprocess_spectrum(wavenumbers, intensities, self.target_wavenumbers)
            return processed[:, 1]  # Return only the intensities for ML
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return None

class RamanAnalyzer:
    def __init__(self, class_a_dir='class_a_dir', class_b_dir='class_b_dir', unknown_dir='unknown_dir'):
        """
        Initialize the Raman Analyzer.
        
        Parameters:
        -----------
        class_a_dir (str): Directory containing Class A spectra files
        class_b_dir (str): Directory containing Class B spectra files
        unknown_dir (str): Directory containing unknown spectra files
        """
        self.class_a_dir = class_a_dir
        self.class_b_dir = class_b_dir
        self.unknown_dir = unknown_dir
        self.preprocessor = RamanPreprocessor()
        self.model = None
        self.scaler = StandardScaler()
        
    def load_training_data(self, file_ext=None):
        """Load and preprocess training data."""
        if file_ext is None:
            file_ext = ['.txt', '.csv', '.dat']
            
        # Load Class A spectra
        print(f"Loading Class A spectra from {self.class_a_dir}...")
        class_a_spectra = batch_read_spectra(self.class_a_dir, file_ext=file_ext, verbose=True)
        
        if not class_a_spectra:
            raise ValueError("No Class A spectra found!")
            
        # Load Class B spectra
        print(f"Loading Class B spectra from {self.class_b_dir}...")
        class_b_spectra = batch_read_spectra(self.class_b_dir, file_ext=file_ext, verbose=True)
        
        if not class_b_spectra:
            raise ValueError("No Class B spectra found!")
        
        X = []
        y = []
        
        # Process Class A spectra
        for filename, (wavenumbers, intensities) in class_a_spectra.items():
            try:
                # Preprocess the spectrum with resampling
                processed = preprocess_spectrum(wavenumbers, intensities, self.preprocessor.target_wavenumbers)
                X.append(processed[:, 1])  # Only intensities
                y.append(1)  # 1 for Class A
            except Exception as e:
                print(f"Error processing Class A file {filename}: {str(e)}")
        
        # Process Class B spectra
        for filename, (wavenumbers, intensities) in class_b_spectra.items():
            try:
                # Preprocess the spectrum with resampling
                processed = preprocess_spectrum(wavenumbers, intensities, self.preprocessor.target_wavenumbers)
                X.append(processed[:, 1])  # Only intensities
                y.append(0)  # 0 for Class B
            except Exception as e:
                print(f"Error processing Class B file {filename}: {str(e)}")
        
        if not X:
            raise ValueError("No valid spectra found to process!")
            
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"Successfully processed {len(X)} spectra:")
        print(f"- Class A spectra: {np.sum(y == 1)}")
        print(f"- Class B spectra: {np.sum(y == 0)}")
        
        return X, y