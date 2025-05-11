#!/usr/bin/env python3
"""
Script to create a test model for the Raman classification feature.
This generates a simple RandomForestClassifier and saves it to disk.
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import argparse

def create_test_model(output_file='raman_plastic_classifier.joblib'):
    """Create a simple test model for classification."""
    print(f"Creating test model at {output_file}...")
    
    # Create a simple random forest classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create a simple dataset (this would normally be your Raman spectra)
    # 2 classes, 1000 points per spectrum
    X_train = np.random.rand(100, 1000)  # 100 spectra with 1000 data points each
    y_train = np.array([0] * 50 + [1] * 50)  # 50 samples of each class
    
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    # Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Save the model
    print(f"Saving model to {output_file}")
    joblib.dump(pipeline, output_file)
    
    print(f"Test model created successfully at {output_file}")
    
    # Print some basic info about the model
    print("\nModel Details:")
    print(f"- Features: {X_train.shape[1]}")
    print(f"- Classes: 2 (Class A and Class B)")
    print(f"- Training samples: {X_train.shape[0]}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create a test model for Raman classification")
    parser.add_argument("--output", default="raman_plastic_classifier.joblib", 
                       help="Output model file path (default: raman_plastic_classifier.joblib)")
    
    args = parser.parse_args()
    create_test_model(args.output)

if __name__ == "__main__":
    main() 