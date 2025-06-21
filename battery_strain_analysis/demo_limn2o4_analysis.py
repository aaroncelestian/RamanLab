"""
LiMn2O4 Battery Strain Analysis Demo
===================================

Complete demonstration of the battery strain analysis system for LiMn2O4
showing H/Li exchange effects during time series measurements.

This demo shows:
1. Setting up the analysis system
2. Loading and processing time series data
3. Running strain analysis across time points
4. Visualizing results and generating reports
5. Interpreting physical effects (Jahn-Teller, phase transitions, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our battery strain analysis modules
from limn2o4_analyzer import LiMn2O4StrainAnalyzer
from time_series_processor import TimeSeriesProcessor
from spinel_modes import SpinelRamanModes
from strain_visualization import StrainVisualizer

def generate_synthetic_time_series():
    """
    Generate synthetic time series data to demonstrate the analysis
    
    This simulates H/Li exchange in LiMn2O4 showing:
    - Peak frequency shifts due to composition changes
    - Broadening due to disorder
    - Intensity changes due to structural transitions
    - Jahn-Teller distortion effects
    """
    
    print("Generating synthetic time series data...")
    
    # Time points (seconds) - simulating a 1000 second experiment
    time_points = np.linspace(0, 1000, 20)
    
    # Simulate H/Li exchange: starts Li-rich, becomes H-rich
    li_content = 1.0 - 0.8 * (1 - np.exp(-time_points / 300))  # Exponential decay
    h_content = 1.0 - li_content
    
    # Base frequency range
    frequencies = np.linspace(200, 700, 1000)
    
    # Get mode information
    spinel_modes = SpinelRamanModes()
    
    time_series_data = []
    
    for i, (t, li, h) in enumerate(zip(time_points, li_content, h_content)):
        print(f"  Generating spectrum {i+1}/20 at t={t:.0f}s (Li={li:.2f}, H={h:.2f})")
        
        # Initialize spectrum
        spectrum = np.zeros_like(frequencies)
        
        # Calculate composition-dependent effects
        mn3_fraction = 1.0 - li  # Mn3+ increases as Li decreases
        jt_parameter = 0.1 * mn3_fraction  # Jahn-Teller strength
        disorder_level = li * h * 4  # Maximum disorder at 50/50
        
        # Generate peaks for each mode
        for mode_name, mode in spinel_modes.modes.items():
            base_freq = mode.omega0_pure
            base_intensity = mode.intensity_pure
            natural_width = mode.natural_width
            
            # Composition-dependent frequency shift
            composition_shift = np.sum(mode.composition_sensitivity * h) if mode.composition_sensitivity is not None else 0
            freq_shift = composition_shift * base_freq * 0.01  # 1% shift per composition unit
            
            # Jahn-Teller splitting/shifting
            jt_shift = mode.jahn_teller_coupling * jt_parameter * base_freq * 0.01
            
            # Disorder broadening
            disorder_broadening = disorder_level * 2.0  # Additional width from disorder
            
            # Intensity changes (some modes get weaker with composition change)
            intensity_factor = 1.0
            if mode_name == 'Li_O':
                intensity_factor = li  # Li-O mode weakens as Li is removed
            elif mode_name == 'Disorder':
                intensity_factor = disorder_level  # Disorder mode appears with mixing
            
            # Calculate peak parameters
            peak_freq = base_freq + freq_shift + jt_shift
            peak_intensity = base_intensity * intensity_factor
            peak_width = natural_width + disorder_broadening
            
            # Add some random noise
            peak_freq += np.random.normal(0, 1.0)  # ±1 cm⁻¹ noise
            peak_intensity += np.random.normal(0, peak_intensity * 0.1)  # 10% intensity noise
            
            # Generate Gaussian peak
            if peak_intensity > 1.0:  # Only add significant peaks
                peak = peak_intensity * np.exp(-((frequencies - peak_freq) / peak_width)**2)
                spectrum += peak
        
        # Add baseline and noise
        baseline = 50 + 20 * np.random.random()
        noise = np.random.normal(0, 5, len(frequencies))
        spectrum = spectrum + baseline + noise
        
        # Ensure no negative values
        spectrum = np.maximum(spectrum, 0)
        
        # Store data point
        data_point = {
            'time': t,
            'frequencies': frequencies,
            'intensities': spectrum,
            'composition': {'Li': li, 'H': h},
            'mn3_fraction': mn3_fraction,
            'jt_parameter': jt_parameter,
            'disorder_level': disorder_level
        }
        
        time_series_data.append(data_point)
    
    print(f"Generated {len(time_series_data)} synthetic spectra")
    return time_series_data

def run_complete_analysis():
    """Run the complete LiMn2O4 strain analysis workflow"""
    
    print("LiMn2O4 Battery Strain Analysis Demo")
    print("=" * 50)
    
    # 1. Generate synthetic data
    time_series_data = generate_synthetic_time_series()
    
    # 2. Initialize analyzer
    print("\nInitializing LiMn2O4 strain analyzer...")
    analyzer = LiMn2O4StrainAnalyzer(temperature=298.0)
    
    # Show available modes
    print("\nAvailable Raman modes:")
    for mode_name in analyzer.spinel_modes.get_mode_names():
        mode = analyzer.spinel_modes.modes[mode_name]
        print(f"  {mode_name}: {mode.omega0_pure:.1f} cm⁻¹ (JT coupling: {mode.jahn_teller_coupling:.1f})")
    
    # 3. Run time series analysis
    print("\nRunning time series strain analysis...")
    results = analyzer.analyze_time_series(time_series_data)
    
    # 4. Print analysis summary
    print("\nAnalysis Results Summary:")
    print(f"  Time points analyzed: {len(results['time_points'])}")
    print(f"  Time range: {min(results['time_points']):.0f} - {max(results['time_points']):.0f} seconds")
    
    if 'analysis_summary' in results:
        summary = results['analysis_summary']
        print(f"  Max strain rate: {summary.get('max_strain_rate', 'N/A')}")
        print(f"  Composition rate: {summary.get('composition_rate', 'N/A')}")
        
        # Look for Jahn-Teller transitions
        if 'jt_transition_time' in summary:
            print(f"  Jahn-Teller transition detected at: {summary['jt_transition_time']:.0f}s")
    
    # 5. Show peak tracking results
    print("\nPeak Evolution Summary:")
    for mode_name, tracking_data in results.get('peak_tracking', {}).items():
        if tracking_data['frequencies']:
            freqs = np.array(tracking_data['frequencies'])
            freq_change = freqs[-1] - freqs[0]
            print(f"  {mode_name}: {freq_change:+.1f} cm⁻¹ total shift")
    
    # 6. Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = StrainVisualizer()
    
    # Create output directory
    output_dir = Path("limn2o4_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Overview plot
        fig1 = visualizer.plot_time_series_overview(results)
        fig1.savefig(output_dir / "time_series_overview.png", dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/time_series_overview.png")
        plt.close(fig1)
        
        # Final strain tensor 3D
        if 'strain_evolution' in results and len(results['strain_evolution']) > 0:
            final_strain = results['strain_evolution'][-1]
            fig2 = visualizer.plot_strain_tensor_3d(final_strain, "Final Strain State")
            fig2.savefig(output_dir / "final_strain_3d.png", dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_dir}/final_strain_3d.png")
            plt.close(fig2)
            
    except Exception as e:
        print(f"  Warning: Some plots failed to generate: {e}")
    
    # 7. Save detailed results
    print("\nSaving detailed results...")
    
    # Save strain evolution data
    if 'strain_evolution' in results:
        strain_data = results['strain_evolution']
        np.savetxt(output_dir / "strain_evolution.csv", strain_data, 
                  delimiter=',', header='e11,e22,e33,e12,e13,e23')
        print(f"  Saved: {output_dir}/strain_evolution.csv")
    
    # Save composition evolution
    if 'composition_evolution' in results:
        comp_data = results['composition_evolution']
        times = results['time_points']
        combined = np.column_stack([times, comp_data, 1-comp_data])
        np.savetxt(output_dir / "composition_evolution.csv", combined,
                  delimiter=',', header='time,H_fraction,Li_fraction')
        print(f"  Saved: {output_dir}/composition_evolution.csv")
    
    # Create summary report
    summary_file = output_dir / "analysis_report.txt"
    with open(summary_file, 'w') as f:
        f.write("LiMn2O4 Strain Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Experiment Overview:\n")
        f.write(f"  Material: LiMn2O4 spinel\n")
        f.write(f"  Process: H/Li exchange\n")
        f.write(f"  Temperature: {analyzer.temperature}K\n")
        f.write(f"  Duration: {max(results['time_points']):.0f} seconds\n")
        f.write(f"  Data points: {len(results['time_points'])}\n\n")
        
        f.write("Key Findings:\n")
        
        # Strain analysis
        if 'strain_evolution' in results:
            strain_data = results['strain_evolution']
            final_strain = strain_data[-1]
            f.write(f"  Final strain tensor: [{', '.join(f'{x:.4f}' for x in final_strain)}]\n")
            
            # Hydrostatic and deviatoric components
            hydrostatic = np.mean(final_strain[:3])
            deviatoric = np.sqrt(np.sum((final_strain[:3] - hydrostatic)**2))
            f.write(f"  Hydrostatic strain: {hydrostatic:.4f}\n")
            f.write(f"  Deviatoric strain: {deviatoric:.4f}\n")
        
        # Composition changes
        if 'composition_evolution' in results:
            comp_data = results['composition_evolution']
            initial_h = comp_data[0]
            final_h = comp_data[-1]
            f.write(f"  Initial H content: {initial_h:.2f}\n")
            f.write(f"  Final H content: {final_h:.2f}\n")
            f.write(f"  H content change: {final_h - initial_h:+.2f}\n")
        
        # Jahn-Teller effects
        if 'jt_evolution' in results:
            jt_data = results['jt_evolution']
            max_jt = np.max(jt_data)
            f.write(f"  Maximum Jahn-Teller parameter: {max_jt:.4f}\n")
        
        f.write("\nPeak Assignments and Changes:\n")
        for mode_name, tracking_data in results.get('peak_tracking', {}).items():
            if tracking_data['frequencies']:
                freqs = np.array(tracking_data['frequencies'])
                mode = analyzer.spinel_modes.modes[mode_name]
                f.write(f"  {mode_name} ({mode.omega0_pure:.0f} cm⁻¹): {freqs[-1] - freqs[0]:+.1f} cm⁻¹ shift\n")
        
        f.write(f"\nAnalysis completed successfully.\n")
        f.write(f"Results saved in: {output_dir.absolute()}\n")
    
    print(f"  Saved: {summary_file}")
    
    print(f"\nAnalysis complete! Results saved in: {output_dir.absolute()}")
    
    return results, analyzer

def demonstrate_individual_components():
    """Demonstrate individual components of the analysis system"""
    
    print("\n" + "=" * 60)
    print("Individual Component Demonstrations")
    print("=" * 60)
    
    # 1. Spinel modes
    print("\n1. Spinel Raman Modes:")
    spinel = SpinelRamanModes()
    
    print("   Primary modes:")
    for mode_name in spinel.get_primary_modes():
        mode = spinel.modes[mode_name]
        print(f"     {mode_name}: {mode.omega0_pure:.1f} cm⁻¹")
    
    print("   Composition-sensitive modes:")
    for mode_name in spinel.get_composition_sensitive_modes():
        print(f"     {mode_name}")
    
    # Test mode splitting
    jt_param = 0.05
    splitting = spinel.calculate_mode_splitting(jt_param)
    print(f"\n   Mode splitting with JT parameter {jt_param}:")
    for mode, (low, high) in splitting.items():
        if abs(low - high) > 1.0:
            print(f"     {mode}: {low:.1f} - {high:.1f} cm⁻¹")
    
    # 2. Time series processor
    print("\n2. Time Series Processor:")
    processor = TimeSeriesProcessor()
    print("   Processor initialized")
    print("   Supported file formats: .txt, .csv, .dat")
    print("   Features: background subtraction, smoothing, normalization")
    
    # 3. Strain analyzer
    print("\n3. LiMn2O4 Strain Analyzer:")
    analyzer = LiMn2O4StrainAnalyzer()
    print(f"   Crystal system: {analyzer.crystal_system}")
    print(f"   Composition model: {analyzer.composition_model}")
    print(f"   Number of modes: {len(analyzer.modes)}")
    
    # Test battery state
    analyzer.set_battery_state(li_content=0.5, h_content=0.5)
    print(f"   Example state - Li: {analyzer.lithium_content}, H: {analyzer.hydrogen_content}")
    print(f"   Mn3+ fraction: {analyzer.mn3_fraction:.2f}")
    print(f"   JT parameter: {analyzer.jahn_teller_parameter:.4f}")
    
    # 4. Visualizer
    print("\n4. Strain Visualizer:")
    visualizer = StrainVisualizer()
    print("   Available plot types:")
    print("     - Time series overview")
    print("     - 3D strain tensor")
    print("     - Peak evolution")
    print("     - Phase transition maps")
    print("     - Correlation matrices")

def main():
    """Main function for running the complete LiMn2O4 strain analysis demo"""
    try:
        # Run the complete analysis demonstration
        results, analyzer = run_complete_analysis()
        
        # Show individual components
        demonstrate_individual_components()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Replace synthetic data with your experimental time series")
        print("2. Adjust mode parameters based on your specific LiMn2O4 sample")
        print("3. Calibrate composition-voltage relationships for your system")
        print("4. Use the analysis results to understand your battery's behavior")
        
        return True
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis is likely due to missing dependencies or import issues.")
        print("Make sure all required packages are installed (numpy, scipy, matplotlib, pandas)")
        return False

if __name__ == "__main__":
    main() 