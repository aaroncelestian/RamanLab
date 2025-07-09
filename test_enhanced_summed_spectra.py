#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced summed spectra functionality
"""

import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raman_cluster_analysis_qt6 import launch_cluster_analysis

def create_realistic_spectral_data():
    """Create realistic spectral data with known peaks and noise."""
    # Create realistic wavenumbers (400-4000 cm^-1)
    wavenumbers = np.linspace(400, 4000, 1000)
    
    # Define realistic Raman peaks for different materials
    peak_definitions = {
        0: [  # Cluster 0: Quartz-like (SiO2)
            {'center': 465, 'width': 15, 'height': 2.0, 'name': 'Si-O-Si bend'},
            {'center': 800, 'width': 20, 'height': 1.5, 'name': 'Si-O-Si sym stretch'},
            {'center': 1080, 'width': 25, 'height': 2.5, 'name': 'Si-O asym stretch'},
            {'center': 1160, 'width': 18, 'height': 1.8, 'name': 'Si-O sym stretch'},
        ],
        1: [  # Cluster 1: Calcite-like (CaCO3)
            {'center': 280, 'width': 12, 'height': 1.2, 'name': 'Ca-O lattice'},
            {'center': 712, 'width': 16, 'height': 2.8, 'name': 'CO3 in-plane bend'},
            {'center': 1086, 'width': 22, 'height': 3.2, 'name': 'CO3 sym stretch'},
            {'center': 1435, 'width': 30, 'height': 1.5, 'name': 'CO3 asym stretch'},
        ],
        2: [  # Cluster 2: Hematite-like (Fe2O3)
            {'center': 225, 'width': 10, 'height': 1.0, 'name': 'Fe-O lattice'},
            {'center': 245, 'width': 12, 'height': 1.3, 'name': 'Fe-O lattice'},
            {'center': 290, 'width': 15, 'height': 1.8, 'name': 'Fe-O lattice'},
            {'center': 410, 'width': 18, 'height': 2.2, 'name': 'Fe-O stretch'},
            {'center': 500, 'width': 20, 'height': 2.5, 'name': 'Fe-O stretch'},
            {'center': 610, 'width': 22, 'height': 2.8, 'name': 'Fe-O stretch'},
        ]
    }
    
    def gaussian_peak(x, center, width, height):
        """Create a Gaussian peak."""
        return height * np.exp(-((x - center) / width) ** 2)
    
    def create_spectrum_with_peaks(peaks, wavenumbers, noise_level=0.1, baseline_drift=0.05):
        """Create a spectrum with specified peaks, noise, and baseline drift."""
        spectrum = np.zeros_like(wavenumbers)
        
        # Add peaks
        for peak in peaks:
            spectrum += gaussian_peak(wavenumbers, peak['center'], peak['width'], peak['height'])
        
        # Add baseline drift (polynomial)
        baseline = baseline_drift * (wavenumbers - wavenumbers[0]) / (wavenumbers[-1] - wavenumbers[0]) ** 2
        spectrum += baseline
        
        # Add random noise
        noise = np.random.normal(0, noise_level, len(wavenumbers))
        spectrum += noise
        
        # Ensure non-negative values
        spectrum = np.maximum(spectrum, 0)
        
        return spectrum
    
    # Create spectra for each cluster
    intensities = []
    metadata = []
    
    for cluster_id, peaks in peak_definitions.items():
        n_spectra = 15 if cluster_id == 0 else (12 if cluster_id == 1 else 18)
        
        for i in range(n_spectra):
            # Add some variation to peak parameters
            varied_peaks = []
            for peak in peaks:
                varied_peak = peak.copy()
                # Vary peak height by ±20%
                varied_peak['height'] *= np.random.uniform(0.8, 1.2)
                # Vary peak width by ±10%
                varied_peak['width'] *= np.random.uniform(0.9, 1.1)
                # Vary peak center by ±5 cm^-1
                varied_peak['center'] += np.random.uniform(-5, 5)
                varied_peaks.append(varied_peak)
            
            # Create spectrum with realistic noise and drift
            noise_level = np.random.uniform(0.05, 0.15)  # Variable noise
            baseline_drift = np.random.uniform(0.02, 0.08)  # Variable drift
            
            spectrum = create_spectrum_with_peaks(varied_peaks, wavenumbers, noise_level, baseline_drift)
            intensities.append(spectrum)
            
            metadata.append({
                'filename': f'cluster{cluster_id}_spectrum_{i:03d}.txt',
                'sample_id': f'Sample_{cluster_id}_{i:03d}',
                'cluster': cluster_id,
                'material_type': ['Quartz-like', 'Calcite-like', 'Hematite-like'][cluster_id],
                'acquisition_date': f'2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}',
                'integration_time': np.random.randint(10, 60),
                'laser_power': np.random.uniform(0.5, 2.0)
            })
    
    return wavenumbers, np.array(intensities), metadata

def main():
    """Main function to test the enhanced summed spectra functionality."""
    print("Testing Enhanced Summed Spectra Functionality")
    print("=" * 60)
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Launch cluster analysis
    cluster_window = launch_cluster_analysis(None, None)
    
    # Create realistic test data
    print("Creating realistic spectral data...")
    wavenumbers, intensities, metadata = create_realistic_spectral_data()
    
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
        material = metadata[np.where(labels == cluster_id)[0][0]]['material_type']
        print(f"  Cluster {cluster_id}: {count} spectra ({material})")
    
    print("\n🎯 Enhanced Summed Spectra Features:")
    print("1. 📁 Individual .txt files for each cluster's summed spectrum")
    print("2. 🔧 Advanced signal processing pipeline:")
    print("   • Baseline correction (polynomial fitting)")
    print("   • Area normalization for consistent scaling")
    print("   • Savitzky-Golay smoothing for noise reduction")
    print("   • Outlier rejection for quality control")
    print("3. 📈 Signal-to-noise improvement of ~√N")
    print("4. 📊 Professional visualization with processing details")
    print("5. 📋 Complete metadata preservation")
    
    print("\n🚀 To test the enhanced functionality:")
    print("1. Go to Visualization tab → Scatter Plot")
    print("2. Click 'Export Summed Spectra' button")
    print("3. Choose export directory for .txt files")
    print("4. Choose filename for the visualization plot")
    print("5. Check the generated files:")
    print("   • cluster_0_summed_spectrum.txt (Quartz-like)")
    print("   • cluster_1_summed_spectrum.txt (Calcite-like)")
    print("   • cluster_2_summed_spectrum.txt (Hematite-like)")
    print("   • cluster_summed_spectra.png (visualization)")
    
    print("\n💡 Expected improvements:")
    print("• Much cleaner spectra with reduced noise")
    print("• Better defined peaks for peak fitting")
    print("• Consistent baseline across all spectra")
    print("• Professional quality for publications")
    
    # Show the window
    cluster_window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 