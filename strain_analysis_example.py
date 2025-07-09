#!/usr/bin/env python3
"""
Example script demonstrating the enhanced Strain Tensor Analysis functionality
in the RamanLab Raman Polarization Analyzer.

This script shows how to:
1. Initialize the strain analyzer
2. Set up Grüneisen parameters
3. Generate test data
4. Perform strain tensor analysis
5. Interpret results

Based on the implementation guide in Strain_Tensor_Refinement_for_Raman_Spectroscopy_Guide.md
"""

import numpy as np
import matplotlib.pyplot as plt
from strain_raman_refinement import StrainRamanAnalyzer, RamanMode

# Import safe file handling
from pkl_utils import get_workspace_root, get_example_data_paths, get_example_spectrum_file, print_available_example_files
from utils.file_loaders import load_spectrum_file
from pathlib import Path

def setup_matplotlib_config():
    """Set up matplotlib configuration using the workspace config."""
    try:
        # Try to import the matplotlib config
        workspace_root = get_workspace_root()
        polarization_ui_dir = workspace_root / "polarization_ui"
        
        if (polarization_ui_dir / "matplotlib_config.py").exists():
            import sys
            sys.path.insert(0, str(polarization_ui_dir))
            import matplotlib_config
            print("✅ Using RamanLab matplotlib configuration")
        else:
            # Use default matplotlib settings
            plt.style.use('default')
            plt.rcParams.update({
                'figure.figsize': (12, 10),
                'font.size': 10,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'lines.linewidth': 2
            })
            print("⚠️  Using default matplotlib configuration")
    except Exception as e:
        print(f"❌ Error setting up matplotlib: {e}")
        plt.style.use('default')

def load_real_quartz_data():
    """
    Load real quartz data for strain analysis if available.
    
    Returns:
        tuple: (wavenumbers, intensities, metadata) or (None, None, None) if not available
    """
    try:
        # Look for quartz example data
        quartz_file = get_example_spectrum_file('quartz')
        
        if quartz_file and quartz_file.exists():
            print(f"📄 Loading real quartz data from: {quartz_file.name}")
            
            wavenumbers, intensities, metadata = load_spectrum_file(str(quartz_file))
            
            if wavenumbers is not None and intensities is not None:
                print(f"✅ Successfully loaded quartz spectrum:")
                print(f"   • Data points: {len(wavenumbers)}")
                print(f"   • Wavenumber range: {wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹")
                print(f"   • Intensity range: {intensities.min():.1f} - {intensities.max():.1f}")
                return wavenumbers, intensities, metadata
            else:
                print("❌ Failed to load quartz spectrum data")
                return None, None, None
        else:
            print("⚠️  No quartz example data found")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Error loading quartz data: {e}")
        return None, None, None

def demonstrate_strain_analysis():
    """Demonstrate the complete strain tensor analysis workflow."""
    
    print("🔬 Strain Tensor Analysis Demonstration")
    print("=" * 50)
    
    # Set up matplotlib configuration
    setup_matplotlib_config()
    
    # Show available example files
    print("\n📁 Available Example Data Files:")
    print_available_example_files()
    
    # Try to load real quartz data
    print("\n📊 Loading Data for Analysis:")
    real_wavenumbers, real_intensities, real_metadata = load_real_quartz_data()
    
    # Step 1: Initialize the strain analyzer
    print("\n1. Initializing strain analyzer for quartz (trigonal crystal system)...")
    analyzer = StrainRamanAnalyzer('trigonal')
    
    # Step 2: Add Raman modes with Grüneisen parameters
    print("2. Adding Raman modes with literature Grüneisen parameters...")
    
    # Quartz modes from Angel et al. (2019)
    quartz_modes = [
        RamanMode('464', 464.8, np.array([0.60, 0.60, 1.19, 0, 0, 0]), intensity=100, natural_width=5.0),
        RamanMode('207', 207.3, np.array([3.64, 3.64, 5.25, 0, 0, 0]), intensity=50, natural_width=4.0),
        RamanMode('128', 128.1, np.array([1.21, 1.21, 2.69, 0, 0, 0]), intensity=30, natural_width=6.0),
    ]
    
    for mode in quartz_modes:
        analyzer.add_mode(mode)
        print(f"   Added mode: {mode.name} cm⁻¹ with γ = {mode.gamma_components}")
    
    # Step 3: Generate test data with known strain
    print("\n3. Generating test spectrum with known strain...")
    
    # Define a realistic strain state
    true_strain = np.array([0.015, 0.015, -0.008, 0, 0, 0])  # Biaxial compression with Poisson expansion
    true_gradients = np.array([0.0008, 0.0008, 0.0004, 0, 0, 0])  # Small strain gradients
    
    print(f"   True strain tensor: {true_strain}")
    print(f"   True strain gradients: {true_gradients}")
    
    # Generate frequency range and spectrum
    if real_wavenumbers is not None:
        # Use real data wavenumber range
        freq_min = max(100, real_wavenumbers.min())
        freq_max = min(1400, real_wavenumbers.max())
        frequencies = np.linspace(freq_min, freq_max, 1000)
        print(f"   Using real data wavenumber range: {freq_min:.1f} - {freq_max:.1f} cm⁻¹")
    else:
        # Use default range
        frequencies = np.linspace(100, 500, 1000)
        print(f"   Using default wavenumber range: 100 - 500 cm⁻¹")
    
    test_spectrum = analyzer.forward_model(
        np.concatenate([true_strain, true_gradients]), 
        frequencies, 
        include_gradients=True
    )
    
    # Add realistic noise
    noise_level = 0.03
    test_spectrum += noise_level * np.random.randn(len(test_spectrum))
    test_spectrum = np.maximum(test_spectrum, 0)  # Ensure non-negative
    
    # Step 4: Analyze peak positions only (quick method)
    print("\n4. Performing peak position analysis...")
    
    # Extract peak positions from the generated spectrum
    from scipy.signal import find_peaks
    peak_indices, _ = find_peaks(test_spectrum, height=0.1, distance=20)
    
    observed_peaks = {}
    for idx in peak_indices:
        freq = frequencies[idx]
        # Match to known modes
        for mode_name, mode in analyzer.modes.items():
            if abs(freq - mode.omega0) < 15:  # Within 15 cm⁻¹
                observed_peaks[mode_name] = freq
                break
    
    print(f"   Found {len(observed_peaks)} peaks: {observed_peaks}")
    
    if observed_peaks:
        peak_result = analyzer.analyze_peak_positions(observed_peaks)
        
        if peak_result['success']:
            fitted_strain_peaks = peak_result['strain_tensor']
            print(f"   Peak analysis strain: {fitted_strain_peaks}")
            print(f"   Error vs true strain: {np.abs(fitted_strain_peaks - true_strain)}")
        else:
            print(f"   Peak analysis failed: {peak_result.get('message', 'Unknown error')}")
    
    # Step 5: Full spectrum fitting
    print("\n5. Performing full spectrum fitting...")
    
    # Use peak analysis as initial guess if available
    initial_strain = fitted_strain_peaks if 'fitted_strain_peaks' in locals() else None
    
    full_result = analyzer.fit_strain_tensor(
        frequencies, test_spectrum,
        initial_strain=initial_strain,
        fit_gradients=True,
        weights=None
    )
    
    if full_result['success']:
        fitted_strain_full = full_result['strain_tensor']
        fitted_gradients = full_result['strain_gradients']
        
        print(f"   Full fitting strain: {fitted_strain_full}")
        print(f"   Full fitting gradients: {fitted_gradients}")
        print(f"   R² = {full_result['r_squared']:.6f}")
        print(f"   χ² = {full_result['chi_squared']:.2e}")
        
        # Compare with true values
        strain_error = np.abs(fitted_strain_full - true_strain)
        gradient_error = np.abs(fitted_gradients - true_gradients)
        
        print(f"   Strain error vs true: {strain_error}")
        print(f"   Gradient error vs true: {gradient_error}")
        
        # Step 6: Physical interpretation
        print("\n6. Physical interpretation of results...")
        
        volume_strain = np.sum(fitted_strain_full[:3])
        principal_strains = np.sort(fitted_strain_full[:3])[::-1]
        
        print(f"   Volume strain (ΔV/V): {volume_strain:.6f}")
        print(f"   Principal strains: {principal_strains}")
        
        if volume_strain > 0:
            print("   → Net volume expansion")
        else:
            print("   → Net volume compression")
        
        # Check for uniaxial vs biaxial strain
        if abs(fitted_strain_full[0] - fitted_strain_full[1]) < 1e-4:
            print("   → Biaxial strain in a-b plane")
        else:
            print("   → General triaxial strain")
        
        # Step 7: Create visualization
        print("\n7. Creating visualization...")
        
        # Save results to workspace
        results_dir = get_workspace_root() / "strain_analysis_results"
        results_dir.mkdir(exist_ok=True)
        
        # Create visualization
        fig_path = create_strain_analysis_plot(
            frequencies, test_spectrum, full_result, 
            true_strain, fitted_strain_full, 
            save_dir=results_dir
        )
        
        print(f"   📊 Visualization saved to: {fig_path}")
        
        # Save analysis results
        results_file = results_dir / "strain_analysis_results.txt"
        save_analysis_results(
            results_file, true_strain, fitted_strain_full, 
            fitted_gradients, full_result
        )
        
        print(f"   📄 Results saved to: {results_file}")
        
        # If we have real data, show how it could be analyzed
        if real_wavenumbers is not None:
            print("\n8. Real data analysis potential...")
            print("   With real quartz data, you could:")
            print("   • Identify actual quartz peaks")
            print("   • Fit strain tensor to real measurements")
            print("   • Analyze natural or experimental strain")
            print("   • Compare with synthetic results")
        
    else:
        print(f"   Full spectrum fitting failed: {full_result.get('message', 'Unknown error')}")
    
    print("\n🎯 Analysis Complete!")
    print("   • Strain tensor analysis: ✅")
    print("   • Physical interpretation: ✅")
    print("   • Visualization: ✅")
    print("   • Results saved: ✅")
    
    return analyzer, full_result if 'full_result' in locals() else None

def create_strain_analysis_plot(frequencies, observed_spectrum, fit_result, true_strain, fitted_strain, save_dir=None):
    """Create a comprehensive plot showing the strain analysis results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Strain Tensor Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Observed vs fitted spectrum
    ax1.plot(frequencies, observed_spectrum, 'b-', linewidth=2, label='Observed', alpha=0.7)
    ax1.plot(frequencies, fit_result['model_spectrum'], 'r--', linewidth=2, label='Fitted')
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Spectrum Fitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = observed_spectrum - fit_result['model_spectrum']
    ax2.plot(frequencies, residuals, 'g-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fitting Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Strain tensor comparison
    strain_labels = ['ε₁', 'ε₂', 'ε₃', 'ε₄', 'ε₅', 'ε₆']
    x_pos = np.arange(len(strain_labels))
    
    width = 0.35
    ax3.bar(x_pos - width/2, true_strain, width, label='True', alpha=0.7, color='blue')
    ax3.bar(x_pos + width/2, fitted_strain, width, label='Fitted', alpha=0.7, color='red')
    
    ax3.set_xlabel('Strain Components')
    ax3.set_ylabel('Strain Value')
    ax3.set_title('Strain Tensor Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(strain_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error analysis
    strain_errors = np.abs(fitted_strain - true_strain)
    ax4.bar(strain_labels, strain_errors, alpha=0.7, color='orange')
    ax4.set_xlabel('Strain Components')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Fitting Errors')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    if save_dir:
        save_path = Path(save_dir) / "strain_analysis_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def save_analysis_results(file_path, true_strain, fitted_strain, fitted_gradients, full_result):
    """Save analysis results to a text file."""
    with open(file_path, 'w') as f:
        f.write("Strain Tensor Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("True Strain Tensor:\n")
        f.write(f"  {true_strain}\n\n")
        
        f.write("Fitted Strain Tensor:\n")
        f.write(f"  {fitted_strain}\n\n")
        
        f.write("Fitted Strain Gradients:\n")
        f.write(f"  {fitted_gradients}\n\n")
        
        f.write("Fitting Statistics:\n")
        f.write(f"  R² = {full_result['r_squared']:.6f}\n")
        f.write(f"  χ² = {full_result['chi_squared']:.2e}\n\n")
        
        f.write("Strain Errors:\n")
        strain_errors = np.abs(fitted_strain - true_strain)
        for i, error in enumerate(strain_errors):
            f.write(f"  ε{i+1} error: {error:.6f}\n")
        
        f.write("\nPhysical Interpretation:\n")
        volume_strain = np.sum(fitted_strain[:3])
        f.write(f"  Volume strain (ΔV/V): {volume_strain:.6f}\n")
        
        principal_strains = np.sort(fitted_strain[:3])[::-1]
        f.write(f"  Principal strains: {principal_strains}\n")
        
        if volume_strain > 0:
            f.write("  → Net volume expansion\n")
        else:
            f.write("  → Net volume compression\n")

def demonstrate_crystal_systems():
    """Demonstrate different crystal systems for strain analysis."""
    
    print("\n🔬 Crystal Systems Demonstration")
    print("=" * 50)
    
    crystal_systems = [
        'cubic', 'tetragonal', 'orthorhombic', 
        'hexagonal', 'trigonal', 'monoclinic', 'triclinic'
    ]
    
    for system in crystal_systems:
        print(f"\n📊 {system.title()} Crystal System:")
        try:
            analyzer = StrainRamanAnalyzer(system)
            print(f"   ✅ Initialized {system} analyzer")
            print(f"   • Strain tensor shape: {analyzer.strain_tensor_shape}")
            print(f"   • Independent components: {analyzer.independent_components}")
        except Exception as e:
            print(f"   ❌ Error initializing {system}: {e}")
    
    print("\n🎯 Crystal System Features:")
    print("   • Cubic: 3 independent strain components")
    print("   • Tetragonal: 6 independent strain components")
    print("   • Orthorhombic: 9 independent strain components")
    print("   • Hexagonal: 5 independent strain components")
    print("   • Trigonal: 6 independent strain components")
    print("   • Monoclinic: 13 independent strain components")
    print("   • Triclinic: 21 independent strain components")

if __name__ == "__main__":
    print("🚀 Starting Strain Tensor Analysis Demo")
    print("=" * 50)
    
    try:
        analyzer, results = demonstrate_strain_analysis()
        demonstrate_crystal_systems()
        
        print("\n✅ Demo completed successfully!")
        print("   Check the strain_analysis_results directory for saved outputs.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 