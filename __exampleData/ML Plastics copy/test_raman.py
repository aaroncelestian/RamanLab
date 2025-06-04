from raman_analyzer import RamanAnalyzer
import numpy as np

def main():
    # Initialize the analyzer
    analyzer = RamanAnalyzer()
    
    try:
        # Test processing a single file
        preprocessor = analyzer.preprocessor
        test_file = "plastic_dir/Acrylic 1. Green Yarn.txt"
        spectrum = preprocessor.process_file(test_file)
        if spectrum is not None:
            print(f"Single file spectrum shape: {spectrum.shape}")
            print(f"First few values: {spectrum[:5]}")
        
        # Load and preprocess all data
        X, y = analyzer.load_and_preprocess_data()
        
        # Print some basic statistics
        print(f"Successfully loaded {len(X)} spectra")
        print(f"Number of plastic spectra: {np.sum(y == 1)}")
        print(f"Number of non-plastic spectra: {np.sum(y == 0)}")
        print(f"Shape of X: {X.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 