#!/usr/bin/env python3
"""
Test script to demonstrate the new cluster export functionality
"""

import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raman_cluster_analysis_qt6 import launch_cluster_analysis

def create_test_data():
    """Create some test spectral data for demonstration."""
    # Create synthetic wavenumbers (400-4000 cm^-1)
    wavenumbers = np.linspace(400, 4000, 1000)
    
    # Create synthetic spectra with different patterns
    n_spectra = 50
    intensities = []
    metadata = []
    
    # Cluster 0: High intensity around 1000 cm^-1
    for i in range(20):
        spectrum = np.random.normal(0, 0.1, len(wavenumbers))
        # Add peak around 1000 cm^-1
        peak_idx = np.argmin(np.abs(wavenumbers - 1000))
        spectrum[peak_idx-50:peak_idx+50] += np.random.normal(2, 0.3, 100)
        intensities.append(spectrum)
        metadata.append({
            'filename': f'cluster0_spectrum_{i:03d}.txt',
            'sample_id': f'Sample_{i:03d}',
            'cluster': 0
        })
    
    # Cluster 1: High intensity around 1500 cm^-1
    for i in range(15):
        spectrum = np.random.normal(0, 0.1, len(wavenumbers))
        # Add peak around 1500 cm^-1
        peak_idx = np.argmin(np.abs(wavenumbers - 1500))
        spectrum[peak_idx-50:peak_idx+50] += np.random.normal(1.8, 0.3, 100)
        intensities.append(spectrum)
        metadata.append({
            'filename': f'cluster1_spectrum_{i:03d}.txt',
            'sample_id': f'Sample_{i+20:03d}',
            'cluster': 1
        })
    
    # Cluster 2: High intensity around 2000 cm^-1
    for i in range(15):
        spectrum = np.random.normal(0, 0.1, len(wavenumbers))
        # Add peak around 2000 cm^-1
        peak_idx = np.argmin(np.abs(wavenumbers - 2000))
        spectrum[peak_idx-50:peak_idx+50] += np.random.normal(1.5, 0.3, 100)
        intensities.append(spectrum)
        metadata.append({
            'filename': f'cluster2_spectrum_{i:03d}.txt',
            'sample_id': f'Sample_{i+35:03d}',
            'cluster': 2
        })
    
    return wavenumbers, np.array(intensities), metadata

def main():
    """Main function to test the cluster export functionality."""
    print("Testing RamanLab Cluster Analysis Export Functionality")
    print("=" * 60)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Launch cluster analysis
    cluster_window = launch_cluster_analysis(None, None)
    
    # Create test data
    print("Creating test spectral data...")
    wavenumbers, intensities, metadata = create_test_data()
    
    # Simulate clustering by assigning labels
    labels = np.array([metadata[i]['cluster'] for i in range(len(metadata))])
    
    # Set up the cluster data
    cluster_window.cluster_data['wavenumbers'] = wavenumbers
    cluster_window.cluster_data['intensities'] = intensities
    cluster_window.cluster_data['spectrum_metadata'] = metadata
    cluster_window.cluster_data['labels'] = labels
    
    # Create some dummy features for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    features = pca.fit_transform(intensities)
    cluster_window.cluster_data['features'] = features
    cluster_window.cluster_data['features_scaled'] = features
    
    print(f"Created {len(intensities)} spectra with {len(np.unique(labels))} clusters")
    print("Cluster distribution:")
    for cluster_id in np.unique(labels):
        count = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} spectra")
    
    print("\nNew Export Features Available:")
    print("1. Export to Folders - Each cluster's spectra saved to separate folders")
    print("2. Export Summed Spectra - Averaged spectra for each cluster as a plot")
    print("3. Export Cluster Overview - Comprehensive overview with all clusters")
    print("\nTo test:")
    print("1. Go to Visualization tab -> Scatter Plot")
    print("2. Use the export buttons at the bottom of the scatter plot tab")
    print("3. Each export option provides different ways to analyze your clusters")
    
    # Show the window
    cluster_window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 