#!/usr/bin/env python3
"""
Demo script for the new Trend Analysis functionality in Raman Density Analysis
This demonstrates how to use the trend analysis features with simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to access the density analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_simulated_sequence_data(n_spectra=20, material_type='Quartz'):
    """Create simulated batch density results showing realistic trends."""
    
    # Simulate a scan sequence with varying crystallinity
    sequence_indices = np.arange(n_spectra)
    
    # Create realistic trends for different scenarios
    scenarios = {
        'crystallization_process': {
            'description': 'Crystallization process (increasing trend)',
            'cdi_trend': lambda x: 0.3 + 0.4 * x / (n_spectra - 1) + np.random.normal(0, 0.05, len(x)),
            'density_factor': 1.2
        },
        'devitrification': {
            'description': 'Devitrification (decreasing trend)',
            'cdi_trend': lambda x: 0.8 - 0.3 * x / (n_spectra - 1) + np.random.normal(0, 0.04, len(x)),
            'density_factor': 0.9
        },
        'stable_material': {
            'description': 'Stable crystalline material (no trend)',
            'cdi_trend': lambda x: 0.65 + np.random.normal(0, 0.03, len(x)),
            'density_factor': 1.0
        },
        'oscillating_process': {
            'description': 'Oscillating process (complex trend)',
            'cdi_trend': lambda x: 0.5 + 0.2 * np.sin(2 * np.pi * x / 8) + np.random.normal(0, 0.04, len(x)),
            'density_factor': 1.1
        }
    }
    
    # Select scenario (change this to test different trends)
    scenario_name = 'crystallization_process'  # Change to test different trends
    scenario = scenarios[scenario_name]
    
    print(f"Creating simulated data for: {scenario['description']}")
    
    # Generate CDI values with trend
    cdi_values = scenario['cdi_trend'](sequence_indices)
    cdi_values = np.clip(cdi_values, 0.0, 1.0)  # Keep in valid range
    
    # Generate corresponding density and spectral parameters
    base_density = 2.5  # Base density for quartz-like material
    density_values = base_density + (cdi_values - 0.5) * 0.8 * scenario['density_factor']
    
    # Generate correlated spectral parameters
    peak_heights = 500 + cdi_values * 800 + np.random.normal(0, 50, n_spectra)
    peak_widths = 15 - cdi_values * 5 + np.random.normal(0, 1, n_spectra)
    peak_positions = 464 + np.random.normal(0, 2, n_spectra)  # Quartz main peak
    
    # Create simulated batch results structure
    batch_results = []
    for i in range(n_spectra):
        # Generate realistic spectrum data
        wavenumbers = np.linspace(200, 1200, 500)
        
        # Create a realistic Raman spectrum with noise
        intensities = (
            100 +  # Baseline
            peak_heights[i] * np.exp(-((wavenumbers - peak_positions[i]) / peak_widths[i])**2) +  # Main peak
            50 * np.exp(-((wavenumbers - 300) / 25)**2) +  # Secondary peak
            np.random.normal(0, 20, len(wavenumbers))  # Noise
        )
        
        # Ensure positive intensities
        intensities = np.maximum(intensities, 10)
        
        result = {
            'filename': f'spectrum_{i+1:03d}.txt',
            'file_path': f'/simulated/data/spectrum_{i+1:03d}.txt',
            'cdi': cdi_values[i],
            'apparent_density': density_values[i],
            'specialized_density': density_values[i],
            'metrics': {
                'main_peak_height': peak_heights[i],
                'main_peak_position': peak_positions[i],
                'baseline_intensity': 100 + np.random.normal(0, 10),
                'peak_width': peak_widths[i],
                'spectral_contrast': cdi_values[i] * 0.8 + np.random.normal(0, 0.05)
            },
            'original_r_squared': 0.85 + cdi_values[i] * 0.1 + np.random.normal(0, 0.05),
            'wavenumbers': wavenumbers,
            'intensities': intensities
        }
        batch_results.append(result)
    
    return batch_results, scenario_name, scenario['description']

def demo_trend_analysis():
    """Demonstrate the trend analysis functionality."""
    print("="*60)
    print("🔬 RAMAN DENSITY ANALYSIS - TREND ANALYSIS DEMO")
    print("="*60)
    
    try:
        # Import the density GUI
        from density_gui_launcher import DensityAnalysisGUI
        from PySide6.QtWidgets import QApplication
        
        # Create Qt application if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Create the GUI
        print("📊 Creating Density Analysis GUI...")
        gui = DensityAnalysisGUI()
        
        # Generate simulated data
        print("🎲 Generating simulated batch data...")
        batch_results, scenario_name, scenario_desc = create_simulated_sequence_data(n_spectra=25)
        
        print(f"✓ Created {len(batch_results)} simulated spectra")
        print(f"📈 Scenario: {scenario_desc}")
        
        # Load the batch data into the GUI
        print("📥 Loading batch data into GUI...")
        gui.set_batch_fitting_results(batch_results)
        
        # Store batch density results directly (simulating completion of batch analysis)
        gui.batch_density_results = batch_results
        
        # Switch to trend analysis tab
        print("🔄 Switching to Trend Analysis tab...")
        gui.analysis_tabs.setCurrentIndex(1)  # Switch to trend analysis tab
        
        # Configure trend analysis parameters
        print("⚙️ Configuring trend analysis parameters...")
        gui.confidence_level.setValue(0.95)
        gui.moving_avg_window.setValue(3)
        gui.trend_sensitivity.setValue(0.02)
        gui.sequence_order_combo.setCurrentText("Filename (numerical)")
        
        # Enable all parameters for analysis
        gui.analyze_cdi_checkbox.setChecked(True)
        gui.analyze_density_checkbox.setChecked(True)
        gui.analyze_peak_height_checkbox.setChecked(True)
        gui.analyze_peak_width_checkbox.setChecked(True)
        
        # Enable the trend analysis button
        gui.analyze_trends_btn.setEnabled(True)
        
        print("🎯 Ready for trend analysis!")
        print("\nDemo Instructions:")
        print("1. The GUI is now loaded with simulated batch data")
        print("2. You're on the 'Trend Analysis' tab")
        print("3. Click 'Analyze Trends' to see the analysis")
        print("4. Try different sequence ordering options")
        print("5. Adjust confidence levels and sensitivity")
        print("6. Export results using the export buttons")
        print(f"\nScenario: {scenario_desc}")
        print("Expected trend: Check the CDI and density parameters")
        
        # Show the GUI
        gui.show()
        
        # Create a simple summary plot to show what we expect
        create_summary_plot(batch_results, scenario_name)
        
        return gui, app
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have PySide6 installed and the density analyzer module available")
        return None, None
    except Exception as e:
        print(f"❌ Error during demo setup: {e}")
        return None, None

def create_summary_plot(batch_results, scenario_name):
    """Create a summary plot showing the expected trends."""
    try:
        # Extract data for plotting
        sequence_indices = np.arange(len(batch_results))
        cdi_values = [r['cdi'] for r in batch_results]
        density_values = [r['specialized_density'] for r in batch_results]
        peak_heights = [r['metrics']['main_peak_height'] for r in batch_results]
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Expected Trends - {scenario_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        
        # CDI trend
        axes[0, 0].plot(sequence_indices, cdi_values, 'bo-', alpha=0.7)
        axes[0, 0].set_title('CDI (Crystalline Density Index)')
        axes[0, 0].set_xlabel('Sequence Index')
        axes[0, 0].set_ylabel('CDI')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Density trend
        axes[0, 1].plot(sequence_indices, density_values, 'ro-', alpha=0.7)
        axes[0, 1].set_title('Specialized Density')
        axes[0, 1].set_xlabel('Sequence Index')
        axes[0, 1].set_ylabel('Density (g/cm³)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Peak height trend
        axes[1, 0].plot(sequence_indices, peak_heights, 'go-', alpha=0.7)
        axes[1, 0].set_title('Peak Height')
        axes[1, 0].set_xlabel('Sequence Index')
        axes[1, 0].set_ylabel('Intensity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation plot
        axes[1, 1].scatter(cdi_values, density_values, c=sequence_indices, cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('CDI vs Density')
        axes[1, 1].set_xlabel('CDI')
        axes[1, 1].set_ylabel('Density (g/cm³)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Summary plot created showing expected trends")
        
    except Exception as e:
        print(f"⚠️ Could not create summary plot: {e}")

def main():
    """Main function to run the demo."""
    print("Starting Trend Analysis Demo...")
    
    gui, app = demo_trend_analysis()
    
    if gui and app:
        print("\n" + "="*60)
        print("🚀 Demo is ready! The GUI should be visible.")
        print("Click 'Analyze Trends' in the Trend Analysis tab to see the results.")
        print("="*60)
        
        # Start the Qt event loop if this script is run directly
        if __name__ == "__main__":
            sys.exit(app.exec())
    else:
        print("❌ Demo failed to start. Check error messages above.")

if __name__ == "__main__":
    main() 