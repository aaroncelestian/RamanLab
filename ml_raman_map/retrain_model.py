from ml_classifier import RamanSpectraClassifier
import os
import sys
import traceback

def validate_directory(directory_path, class_name):
    """Validate that a directory exists and contains valid files."""
    if not os.path.exists(directory_path):
        print(f"Error: {class_name} directory does not exist: {directory_path}")
        return False
    
    files = [f for f in os.listdir(directory_path) if f.endswith(('.txt', '.csv', '.dat'))]
    if not files:
        print(f"Error: No valid spectrum files found in {class_name} directory: {directory_path}")
        return False
    
    print(f"Found {len(files)} files in {class_name} directory")
    return True

def main():
    try:
        # Initialize classifier
        classifier = RamanSpectraClassifier(
            wavenumber_range=(200, 3500),
            n_points=1000,
            positive_class_name="Class A",
            negative_class_name="Class B",
            model_name="raman_classifier"
        )
        
        # Validate input directories
        if not validate_directory("class_a_dir", "Class A"):
            sys.exit(1)
        if not validate_directory("class_b_dir", "Class B"):
            sys.exit(1)
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        try:
            X, y = classifier.load_and_preprocess_data(
                positive_dir="class_a_dir",
                negative_dir="class_b_dir",
                verbose=True
            )
            
            if X is None or y is None or len(X) == 0 or len(y) == 0:
                print("Error: No valid data was loaded from the directories")
                sys.exit(1)
                
            print(f"Successfully loaded {len(X)} samples")
            
        except Exception as e:
            print(f"Error during data loading and preprocessing: {str(e)}")
            print("Detailed error:")
            traceback.print_exc()
            sys.exit(1)
        
        # Train model
        print("\nTraining model...")
        try:
            classifier.train(X, y)
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            print("Detailed error:")
            traceback.print_exc()
            sys.exit(1)
        
        # Save model
        print("\nSaving model...")
        try:
            classifier.save_model()
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print("Detailed error:")
            traceback.print_exc()
            sys.exit(1)
        
        print("\nModel retraining complete!")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print("Detailed error:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 