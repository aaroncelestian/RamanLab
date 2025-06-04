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

def demonstrate_strain_analysis():
    """Demonstrate the complete strain tensor analysis workflow."""
    
    print("=== Strain Tensor Analysis Demonstration ===\n")
    
    # Step 1: Initialize the strain analyzer
    print("1. Initializing strain analyzer for quartz (trigonal crystal system)...")
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
    frequencies = np.linspace(100, 500, 1000)
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
        create_strain_analysis_plot(frequencies, test_spectrum, full_result, true_strain, fitted_strain_full)
        
    else:
        print(f"   Full spectrum fitting failed: {full_result.get('message', 'Unknown error')}")
    
    print("\n=== Analysis Complete ===")
    
    return analyzer, full_result if 'full_result' in locals() else None

def create_strain_analysis_plot(frequencies, observed_spectrum, fit_result, true_strain, fitted_strain):
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
    
    # Add text box with statistics
    stats_text = f"R² = {fit_result['r_squared']:.6f}\n"
    stats_text += f"χ² = {fit_result['chi_squared']:.2e}\n"
    stats_text += f"Max error = {np.max(strain_errors):.6f}"
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def demonstrate_crystal_systems():
    """Demonstrate strain analysis for different crystal systems."""
    
    print("\n=== Crystal System Demonstration ===\n")
    
    crystal_systems = ['cubic', 'tetragonal', 'trigonal', 'orthorhombic']
    
    for crystal_system in crystal_systems:
        print(f"Crystal system: {crystal_system}")
        
        analyzer = StrainRamanAnalyzer(crystal_system)
        
        # Add a generic mode
        generic_mode = RamanMode(
            'test_mode', 400.0, 
            np.array([1.0, 1.0, 1.2, 0.1, 0.1, 0.1]),  # Will be constrained by symmetry
            intensity=100, natural_width=5.0
        )
        analyzer.add_mode(generic_mode)
        
        # Check applied constraints
        constrained_gamma = analyzer.modes['test_mode'].gamma_components
        print(f"   Constrained Grüneisen tensor: {constrained_gamma}")
        
        # Show which components are independent
        constraints = analyzer.symmetry_constraints
        print(f"   Independent components: {constraints['independent']}")
        print(f"   Zero components: {constraints['zeros']}")
        print(f"   Equal pairs: {constraints['equal_pairs']}")
        print()

if __name__ == "__main__":
    # Run the main demonstration
    analyzer, results = demonstrate_strain_analysis()
    
    # Demonstrate different crystal systems
    demonstrate_crystal_systems()
    
    print("\nTo use this functionality in the GUI:")
    print("1. Open the Stress/Strain Analysis tab")
    print("2. Select your crystal system and initialize the analyzer")
    print("3. Set up Grüneisen parameters (literature values or DFT data)")
    print("4. Load or generate test spectrum data")
    print("5. Run peak position analysis or full spectrum fitting")
    print("6. View comprehensive results and export if needed") 