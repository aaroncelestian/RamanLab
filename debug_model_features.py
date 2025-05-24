#!/usr/bin/env python3
"""
Debug script to analyze model feature mismatch issue
"""

import joblib
import numpy as np
from pathlib import Path
import sys

def analyze_model(model_path):
    """Analyze a saved model file to understand the feature mismatch."""
    
    print(f"Analyzing model: {model_path}")
    print("=" * 50)
    
    try:
        # Load the model
        loaded_data = joblib.load(model_path)
        
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            # New format with metadata
            model = loaded_data['model']
            metadata = loaded_data.get('metadata', {})
            
            print("Model format: New format with metadata")
            print(f"Model type: {type(model)}")
            
            # Check actual model features
            if hasattr(model, 'n_features_in_'):
                print(f"Model n_features_in_: {model.n_features_in_}")
            else:
                print("Model has no n_features_in_ attribute")
            
            # Check metadata
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            # Check for discrepancies
            model_features = getattr(model, 'n_features_in_', None)
            metadata_expected = metadata.get('expected_features', None)
            metadata_n_features = metadata.get('n_features', None)
            
            print("\nFeature Analysis:")
            print(f"  Model n_features_in_: {model_features}")
            print(f"  Metadata expected_features: {metadata_expected}")
            print(f"  Metadata n_features: {metadata_n_features}")
            
            if model_features and metadata_expected and model_features != metadata_expected:
                print("\n🚨 MISMATCH DETECTED!")
                print(f"  Model expects {model_features} but metadata says {metadata_expected}")
                print("  This explains the discrepancy you're seeing.")
            elif model_features == metadata_expected:
                print("\n✅ Features match between model and metadata")
            
        else:
            # Old format (direct model)
            model = loaded_data
            print("Model format: Old format (direct model)")
            print(f"Model type: {type(model)}")
            
            if hasattr(model, 'n_features_in_'):
                print(f"Model n_features_in_: {model.n_features_in_}")
            else:
                print("Model has no n_features_in_ attribute")
                
    except Exception as e:
        print(f"Error analyzing model: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_model_features.py <model_file.joblib>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    analyze_model(model_path)

if __name__ == "__main__":
    main() 