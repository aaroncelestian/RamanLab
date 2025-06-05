#!/usr/bin/env python3
"""
Example: Advanced Cluster Analysis with RamanLab
Demonstrates comprehensive analysis capabilities for any mineral system with ion exchange, 
cation substitution, or structural changes. Includes examples for multiple mineral systems.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication

# Import the enhanced cluster analysis module
from raman_cluster_analysis_qt6 import RamanClusterAnalysisQt6


def create_synthetic_mineral_data(system_type="hilairite"):
    """
    Create synthetic Raman data for different mineral systems.
    
    Available systems:
    - hilairite: Y-for-Na ion exchange
    - zeolite: Multi-cation exchange
    - feldspar: Al-Si ordering
    - pyroxene: Fe-Mg substitution
    - clay: Interlayer cation exchange
    """
    
    if system_type == "hilairite":
        return create_hilairite_exchange_data()
    elif system_type == "zeolite":
        return create_zeolite_exchange_data()
    elif system_type == "feldspar":
        return create_feldspar_ordering_data()
    elif system_type == "pyroxene":
        return create_pyroxene_substitution_data()
    elif system_type == "clay":
        return create_clay_exchange_data()
    else:
        return create_hilairite_exchange_data()  # Default


def create_hilairite_exchange_data():
    """Create hilairite Y-for-Na exchange data (original example)."""
    wavenumbers = np.linspace(100, 1300, 1000)
    n_spectra_per_stage = 20
    all_spectra = []
    all_labels = []
    time_data = []
    
    for stage in range(5):
        exchange_fraction = stage / 4.0
        
        for spectrum_idx in range(n_spectra_per_stage):
            local_exchange = exchange_fraction + np.random.normal(0, 0.05)
            local_exchange = np.clip(local_exchange, 0, 1)
            
            spectrum = generate_hilairite_spectrum(wavenumbers, local_exchange)
            
            all_spectra.append(spectrum)
            all_labels.append(stage + 1)
            time_data.append(stage * 24 + np.random.normal(0, 2))
    
    return wavenumbers, np.array(all_spectra), np.array(all_labels), np.array(time_data)


def create_zeolite_exchange_data():
    """Create zeolite cation exchange data (Na/K/Ca/Mg exchange)."""
    wavenumbers = np.linspace(200, 4000, 1200)
    n_spectra_per_stage = 15
    all_spectra = []
    all_labels = []
    time_data = []
    
    # 4 exchange stages: Na-zeolite → K-zeolite → Ca-zeolite → Mg-zeolite
    for stage in range(4):
        exchange_fraction = stage / 3.0
        
        for spectrum_idx in range(n_spectra_per_stage):
            local_exchange = exchange_fraction + np.random.normal(0, 0.04)
            local_exchange = np.clip(local_exchange, 0, 1)
            
            spectrum = generate_zeolite_spectrum(wavenumbers, local_exchange)
            
            all_spectra.append(spectrum)
            all_labels.append(stage + 1)
            time_data.append(stage * 12 + np.random.normal(0, 1.5))  # 12-hour intervals
    
    return wavenumbers, np.array(all_spectra), np.array(all_labels), np.array(time_data)


def create_feldspar_ordering_data():
    """Create feldspar Al-Si ordering data."""
    wavenumbers = np.linspace(150, 1300, 900)
    n_spectra_per_stage = 18
    all_spectra = []
    all_labels = []
    time_data = []
    
    # 6 ordering stages from disordered to ordered
    for stage in range(6):
        ordering_degree = stage / 5.0
        
        for spectrum_idx in range(n_spectra_per_stage):
            local_ordering = ordering_degree + np.random.normal(0, 0.03)
            local_ordering = np.clip(local_ordering, 0, 1)
            
            spectrum = generate_feldspar_spectrum(wavenumbers, local_ordering)
            
            all_spectra.append(spectrum)
            all_labels.append(stage + 1)
            time_data.append(stage * 100 + np.random.normal(0, 5))  # Temperature steps (°C)
    
    return wavenumbers, np.array(all_spectra), np.array(all_labels), np.array(time_data)


def create_pyroxene_substitution_data():
    """Create pyroxene Fe-Mg substitution data."""
    wavenumbers = np.linspace(200, 1200, 800)
    n_spectra_per_stage = 16
    all_spectra = []
    all_labels = []
    time_data = []
    
    # 5 substitution stages from Mg-rich to Fe-rich
    for stage in range(5):
        fe_content = stage / 4.0  # Fe/(Fe+Mg) ratio
        
        for spectrum_idx in range(n_spectra_per_stage):
            local_fe = fe_content + np.random.normal(0, 0.04)
            local_fe = np.clip(local_fe, 0, 1)
            
            spectrum = generate_pyroxene_spectrum(wavenumbers, local_fe)
            
            all_spectra.append(spectrum)
            all_labels.append(stage + 1)
            time_data.append(stage * 0.1 + np.random.normal(0, 0.01))  # Composition (Fe content)
    
    return wavenumbers, np.array(all_spectra), np.array(all_labels), np.array(time_data)


def create_clay_exchange_data():
    """Create clay interlayer cation exchange data."""
    wavenumbers = np.linspace(100, 4000, 1300)
    n_spectra_per_stage = 20
    all_spectra = []
    all_labels = []
    time_data = []
    
    # 4 hydration/exchange stages
    for stage in range(4):
        hydration_level = stage / 3.0
        
        for spectrum_idx in range(n_spectra_per_stage):
            local_hydration = hydration_level + np.random.normal(0, 0.05)
            local_hydration = np.clip(local_hydration, 0, 1)
            
            spectrum = generate_clay_spectrum(wavenumbers, local_hydration)
            
            all_spectra.append(spectrum)
            all_labels.append(stage + 1)
            time_data.append(stage * 6 + np.random.normal(0, 0.5))  # 6-hour intervals
    
    return wavenumbers, np.array(all_spectra), np.array(all_labels), np.array(time_data)


def generate_hilairite_spectrum(wavenumbers, exchange_fraction):
    """Generate hilairite spectrum (original function)."""
    spectrum = np.zeros_like(wavenumbers)
    
    # Framework breathing modes (200-600 cm⁻¹)
    framework_peaks = [250, 350, 450, 550]
    for peak_pos in framework_peaks:
        shifted_pos = peak_pos - exchange_fraction * 10
        width = 20 + exchange_fraction * 5
        intensity = 0.8 + exchange_fraction * 0.3
        spectrum += intensity * np.exp(-((wavenumbers - shifted_pos) / width)**2)
    
    # Si-O stretching modes (800-1200 cm⁻¹)
    sio_peaks = [900, 1000, 1100]
    for peak_pos in sio_peaks:
        shifted_pos = peak_pos + exchange_fraction * 5
        intensity = 1.0 - exchange_fraction * 0.2
        spectrum += intensity * np.exp(-((wavenumbers - shifted_pos) / 30)**2)
    
    # Na-O coordination (decreasing)
    na_peaks = [150, 200, 280]
    for peak_pos in na_peaks:
        intensity = (1 - exchange_fraction) * 0.6
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 25)**2)
    
    # Y-O coordination (increasing)
    y_peaks = [380, 420, 480]
    for peak_pos in y_peaks:
        intensity = exchange_fraction * 0.8
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 20)**2)
    
    spectrum += 0.1 + np.random.normal(0, 0.02, len(wavenumbers))
    return spectrum


def generate_zeolite_spectrum(wavenumbers, exchange_fraction):
    """Generate zeolite cation exchange spectrum."""
    spectrum = np.zeros_like(wavenumbers)
    
    # Framework T-O-T modes (400-600 cm⁻¹)
    framework_peaks = [450, 500, 580]
    for peak_pos in framework_peaks:
        intensity = 1.0 + exchange_fraction * 0.2
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 25)**2)
    
    # T-O stretching (900-1200 cm⁻¹)
    to_peaks = [950, 1050, 1150]
    for peak_pos in to_peaks:
        shifted_pos = peak_pos + exchange_fraction * 8
        intensity = 0.9 - exchange_fraction * 0.1
        spectrum += intensity * np.exp(-((wavenumbers - shifted_pos) / 35)**2)
    
    # Water stretching (3200-3600 cm⁻¹)
    water_peaks = [3300, 3450]
    for peak_pos in water_peaks:
        intensity = 0.5 + exchange_fraction * 0.4  # More hydrated with larger cations
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 60)**2)
    
    # Cation-O modes (changes with exchange)
    cation_peaks = [250, 320]
    for i, peak_pos in enumerate(cation_peaks):
        intensity = 0.4 + (i * exchange_fraction) * 0.6
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 20)**2)
    
    spectrum += 0.08 + np.random.normal(0, 0.015, len(wavenumbers))
    return spectrum


def generate_feldspar_spectrum(wavenumbers, ordering_degree):
    """Generate feldspar Al-Si ordering spectrum."""
    spectrum = np.zeros_like(wavenumbers)
    
    # T-O-T bending (400-550 cm⁻¹)
    bending_peaks = [420, 480, 520]
    for peak_pos in bending_peaks:
        # Peak splitting increases with ordering
        split = ordering_degree * 15
        intensity = 0.7 + ordering_degree * 0.3
        spectrum += intensity * np.exp(-((wavenumbers - (peak_pos - split/2)) / 20)**2)
        spectrum += intensity * np.exp(-((wavenumbers - (peak_pos + split/2)) / 20)**2)
    
    # Al-O stretching (700-800 cm⁻¹)
    al_peaks = [720, 780]
    for peak_pos in al_peaks:
        intensity = ordering_degree * 0.6  # Increases with Al-Si ordering
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 25)**2)
    
    # Si-O stretching (950-1200 cm⁻¹)
    si_peaks = [980, 1080, 1150]
    for peak_pos in si_peaks:
        intensity = 1.0 - ordering_degree * 0.2
        width = 30 - ordering_degree * 5  # Sharper peaks with ordering
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / width)**2)
    
    spectrum += 0.06 + np.random.normal(0, 0.012, len(wavenumbers))
    return spectrum


def generate_pyroxene_spectrum(wavenumbers, fe_content):
    """Generate pyroxene Fe-Mg substitution spectrum."""
    spectrum = np.zeros_like(wavenumbers)
    
    # Si-O stretching (900-1100 cm⁻¹)
    sio_peaks = [950, 1020]
    for peak_pos in sio_peaks:
        shifted_pos = peak_pos - fe_content * 12  # Shift to lower frequency with Fe
        intensity = 0.9
        spectrum += intensity * np.exp(-((wavenumbers - shifted_pos) / 28)**2)
    
    # M-O stretching (600-800 cm⁻¹)
    mo_peaks = [650, 720]
    for peak_pos in mo_peaks:
        shifted_pos = peak_pos - fe_content * 8
        intensity = 0.8
        spectrum += intensity * np.exp(-((wavenumbers - shifted_pos) / 22)**2)
    
    # Fe-O modes (increasing)
    fe_peaks = [280, 320]
    for peak_pos in fe_peaks:
        intensity = fe_content * 0.7
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 18)**2)
    
    # Mg-O modes (decreasing)
    mg_peaks = [380, 420]
    for peak_pos in mg_peaks:
        intensity = (1 - fe_content) * 0.6
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 16)**2)
    
    spectrum += 0.05 + np.random.normal(0, 0.01, len(wavenumbers))
    return spectrum


def generate_clay_spectrum(wavenumbers, hydration_level):
    """Generate clay interlayer exchange spectrum."""
    spectrum = np.zeros_like(wavenumbers)
    
    # OH stretching (3400-3700 cm⁻¹)
    oh_peaks = [3450, 3620]
    for peak_pos in oh_peaks:
        intensity = 0.6 + hydration_level * 0.4
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 40)**2)
    
    # H2O stretching (3000-3400 cm⁻¹)
    water_peak = 3200
    intensity = hydration_level * 0.8  # Increases with hydration
    spectrum += intensity * np.exp(-((wavenumbers - water_peak) / 80)**2)
    
    # H2O bending (1600-1700 cm⁻¹)
    bending_peak = 1640
    intensity = hydration_level * 0.5
    spectrum += intensity * np.exp(-((wavenumbers - bending_peak) / 30)**2)
    
    # Si-O stretching (950-1100 cm⁻¹)
    sio_peaks = [980, 1030]
    for peak_pos in sio_peaks:
        intensity = 0.8
        spectrum += intensity * np.exp(-((wavenumbers - peak_pos) / 35)**2)
    
    # Layer lattice modes (100-400 cm⁻¹)
    lattice_peaks = [200, 300]
    for peak_pos in lattice_peaks:
        shifted_pos = peak_pos + hydration_level * 10  # Expansion with hydration
        intensity = 0.4
        spectrum += intensity * np.exp(-((wavenumbers - shifted_pos) / 25)**2)
    
    spectrum += 0.04 + np.random.normal(0, 0.008, len(wavenumbers))
    return spectrum


def demonstrate_advanced_analysis():
    """Demonstrate the advanced cluster analysis workflow."""
    
    print("=== RamanLab Advanced Cluster Analysis Demo ===\n")
    print("🔬 General-Purpose Mineral Analysis Tool")
    print("   Supports any ion exchange, cation substitution, or structural transition\n")
    
    # Let user choose system or use default
    print("Available Mineral Systems:")
    systems = {
        '1': ('hilairite', 'Hilairite - Y-for-Na ion exchange'),
        '2': ('zeolite', 'Zeolites - Multi-cation exchange (Na/K/Ca/Mg)'),
        '3': ('feldspar', 'Feldspars - Al-Si ordering transitions'),
        '4': ('pyroxene', 'Pyroxenes - Fe-Mg substitution'),
        '5': ('clay', 'Clay minerals - Interlayer cation exchange')
    }
    
    for key, (_, description) in systems.items():
        print(f"   {key}. {description}")
    
    print("\nUsing Hilairite example (default)...")
    system_type = "hilairite"  # Can be changed to any system
    
    # Create synthetic data
    print(f"\n1. Generating synthetic {system_type} data...")
    wavenumbers, spectra, labels, time_data = create_synthetic_mineral_data(system_type)
    
    print(f"   - Generated {len(spectra)} spectra across {len(np.unique(labels))} stages")
    print(f"   - Wavenumber range: {wavenumbers[0]:.0f}-{wavenumbers[-1]:.0f} cm⁻¹")
    print(f"   - Parameter range: {time_data.min():.1f}-{time_data.max():.1f}")
    
    # Initialize Qt Application
    app = QApplication(sys.argv)
    
    # Create the advanced cluster analysis window
    print("\n2. Launching Advanced Cluster Analysis Interface...")
    cluster_window = RamanClusterAnalysisQt6()
    
    # Load the synthetic data directly
    cluster_window.cluster_data['wavenumbers'] = wavenumbers
    cluster_window.cluster_data['intensities'] = spectra
    cluster_window.cluster_data['labels'] = labels
    cluster_window.cluster_data['temporal_data'] = time_data
    
    # Extract and scale features
    from sklearn.preprocessing import StandardScaler
    features = cluster_window.extract_vibrational_features(spectra, wavenumbers)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    cluster_window.cluster_data['features'] = features
    cluster_window.cluster_data['features_scaled'] = features_scaled
    
    print("   ✓ Data loaded successfully")
    print("   ✓ Features extracted and scaled")
    
    # Update UI to reflect loaded data
    cluster_window.update_ui_after_import(len(spectra))
    
    # Show the window
    cluster_window.show()
    
    print("\n3. 🎯 NEW: Universal Mineral System Support")
    print("   📋 Preset Systems Available:")
    print("      - Hilairite (Y-for-Na Exchange)")
    print("      - Zeolites (Cation Exchange)")
    print("      - Feldspars (Al-Si Ordering)")
    print("      - Pyroxenes (Fe-Mg Substitution)")
    print("      - Clay Minerals (Interlayer Exchange)")
    print("      - Olivine (Fe-Mg Exchange)")
    print("      - Garnet (Cation Substitution)")
    print("      - Carbonates (Mg-Ca Exchange)")
    print("      - Spinels (Cation Ordering)")
    print("      - Custom Configuration")
    
    print("\n   🔧 Dynamic Region Management:")
    print("      - Add/remove spectral regions of interest")
    print("      - Customize region names and wavenumber ranges")
    print("      - Chemical/structural descriptions for each region")
    print("      - Save/load custom configurations")
    
    print("\n4. Enhanced Structural Analysis Tab Features:")
    print("   🎛️  Preset Selection: Choose from 9+ predefined mineral systems")
    print("   ➕ Add Regions: Dynamic addition of custom spectral regions")
    print("   ❌ Remove Regions: Select and delete unwanted regions")
    print("   💾 Save/Load: Export and import custom configurations")
    print("   📝 Custom Descriptions: Add chemical context to each region")
    
    print("\n5. 📊 Complete Analysis Workflow:")
    print("   1. Select system preset or create custom configuration")
    print("   2. Run clustering analysis")
    print("   3. Analyze ion exchange progression")
    print("   4. Fit kinetic models")
    print("   5. Perform structural characterization")
    print("   6. Validate with statistical tests")
    print("   7. Generate publication-ready results")
    
    print("\n6. 🎯 Applications Beyond Hilairite:")
    print("   - Any ion exchange process")
    print("   - Cation ordering/disordering")
    print("   - Phase transitions")
    print("   - Solid-state reactions")
    print("   - Compositional gradients")
    print("   - Hydration/dehydration processes")
    
    print("\n🚀 Universal Analysis Tool Ready!")
    print("   Go to 'Structural Analysis' tab to explore system presets")
    print("   Customize regions for your specific mineral system")
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    demonstrate_advanced_analysis() 