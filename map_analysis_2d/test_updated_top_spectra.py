#!/usr/bin/env python3
"""
Test: Updated Top Spectra Display

This script demonstrates the updated Top 5 spectra functionality that:
1. Uses quantitative analysis results instead of old "interesting spectra"
2. Places the legend outside the plot area
3. Shows detailed quality metrics for each spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the analysis directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ui'))

from analysis.top_spectra_selector import TopSpectraSelector, SpectrumRanking
from analysis.quantitative_analysis import RobustQuantitativeAnalyzer, ComponentResult

# Load matplotlib configuration
try:
    import matplotlib_config
    print("✓ Loaded matplotlib configuration")
except ImportError:
    print("⚠ Matplotlib config not found - using defaults")

def create_test_data():
    """Create test data matching your analysis workflow."""
    
    print("Creating test data for updated top spectra display...")
    
    n_pixels = 16383
    np.random.seed(42)
    
    # Create quantitative analysis result
    detection_prob = 0.003  # Sparse detection like your real data
    detection_map = np.random.random(n_pixels) < detection_prob
    n_detected = np.sum(detection_map)
    
    # Create confidence and percentage maps
    confidence_map = np.zeros(n_pixels)
    percentage_map = np.zeros(n_pixels)
    
    detected_indices = np.where(detection_map)[0]
    confidence_map[detected_indices] = np.random.beta(2, 1, n_detected) * 0.4 + 0.6
    percentage_map[detected_indices] = np.random.gamma(2, 15) + 20
    
    # Create component result
    component_result = ComponentResult(
        component_name="Target Material",
        detection_map=detection_map,
        confidence_map=confidence_map,
        percentage_map=percentage_map,
        intensity_map=confidence_map * percentage_map / 100,
        statistics={'detection_rate': n_detected / n_pixels},
        method_contributions={'template': confidence_map * 0.4,
                            'nmf': confidence_map * 0.3,
                            'ml': confidence_map * 0.3}
    )
    
    # Create supporting analysis data
    template_r_squared = np.random.random((n_pixels, 1))
    template_r_squared[detected_indices] += 0.3
    template_r_squared = np.clip(template_r_squared, 0, 1)
    
    nmf_components = np.random.gamma(1, 2, (n_pixels, 5))
    nmf_components[detected_indices, 2] *= 3
    
    ml_probabilities = np.random.random((n_pixels, 2))
    ml_probabilities = ml_probabilities / ml_probabilities.sum(axis=1, keepdims=True)
    ml_probabilities[detected_indices, 1] = np.random.beta(3, 1, n_detected) * 0.4 + 0.6
    ml_probabilities[detected_indices, 0] = 1 - ml_probabilities[detected_indices, 1]
    
    # Create spectral data
    wavenumbers = np.linspace(400, 1800, 500)
    intensities = np.zeros((n_pixels, len(wavenumbers)))
    
    # Background spectra
    base_spectrum = np.exp(-(wavenumbers - 1000)**2 / (200**2)) * 0.3
    for i in range(n_pixels):
        noise = np.random.normal(0, 0.1, len(wavenumbers))
        intensities[i] = base_spectrum + noise
    
    # Add target component to detected pixels
    target_peaks = [500, 750, 1200, 1450]
    target_spectrum = np.zeros_like(wavenumbers)
    for peak in target_peaks:
        target_spectrum += np.exp(-(wavenumbers - peak)**2 / (30**2))
    
    for idx in detected_indices:
        component_strength = np.random.uniform(0.5, 2.0)
        intensities[idx] += target_spectrum * component_strength
    
    print(f"Created test data: {n_detected} detected pixels out of {n_pixels}")
    
    return {
        'component_result': component_result,
        'template_r_squared': template_r_squared,
        'nmf_components': nmf_components,
        'ml_probabilities': ml_probabilities,
        'wavenumbers': wavenumbers,
        'intensities': intensities,
        'detected_indices': detected_indices
    }

def demonstrate_updated_top_spectra_plot():
    """Demonstrate the updated top spectra plotting with external legend."""
    
    print("\nDEMONSTRATING UPDATED TOP SPECTRA DISPLAY")
    print("=" * 60)
    
    # Create test data
    data = create_test_data()
    
    # Initialize top spectra selector
    selector = TopSpectraSelector(n_top_spectra=5)
    
    # Select top spectra using quantitative analysis
    top_spectra = selector.select_top_spectra(
        quantitative_result=data['component_result'],
        template_r_squared=data['template_r_squared'],
        nmf_components=data['nmf_components'],
        ml_probabilities=data['ml_probabilities'],
        map_shape=(128, 128),
        template_index=0,
        nmf_component_index=2,
        ml_class_index=1
    )
    
    print(f"\nSelected {len(top_spectra)} top spectra")
    
    # Create the improved plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot the spectra with external legend (like the new implementation)
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_spectra)))
    spectrum_indices = [spectrum.pixel_index for spectrum in top_spectra]
    
    plotted_count = 0
    legend_labels = []
    
    for i, (ranking, pixel_idx, color) in enumerate(zip(top_spectra, spectrum_indices, colors)):
        wavenumbers = data['wavenumbers']
        intensities = data['intensities'][pixel_idx]
        
        # Normalize for display with offset
        intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min()) + plotted_count * 0.35
        
        # Plot spectrum
        ax.plot(wavenumbers, intensities_norm, color=color, 
               linewidth=1.8, alpha=0.85)
        
        # Create detailed label with quality metrics
        label = f"Rank {i+1}: Pixel {pixel_idx}\n"
        label += f"Conf: {ranking.confidence_score:.2f}, "
        label += f"Pct: {ranking.component_percentage:.1f}%\n"
        label += f"Quality: {ranking.combined_quality_score:.3f}"
        
        if ranking.template_fit_quality is not None:
            label += f", R²: {ranking.template_fit_quality:.2f}"
        
        legend_labels.append(label)
        plotted_count += 1
    
    # Set labels and title
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_ylabel('Normalized Intensity (offset)', fontsize=11)
    ax.set_title(f'Top {plotted_count} Best-Fitting Spectra from Quantitative Analysis', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # NEW: Create legend OUTSIDE the plot area
    legend_elements = []
    for i, (color, label) in enumerate(zip(colors[:plotted_count], legend_labels)):
        from matplotlib.lines import Line2D
        line = Line2D([0], [0], color=color, linewidth=2, alpha=0.85)
        legend_elements.append(line)
    
    # Place legend outside the plot on the right side
    legend = ax.legend(legend_elements, legend_labels, 
                     bbox_to_anchor=(1.05, 1), loc='upper left',
                     fontsize=9, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Highlight target material peaks
    target_peaks = [500, 750, 1200, 1450]
    for peak in target_peaks:
        ax.axvline(peak, color='red', linestyle='--', alpha=0.4, linewidth=1)
    
    plt.show()
    
    # Print detailed report
    report = selector.generate_top_spectra_report(top_spectra, "Target Material")
    print(f"\n{report}")

def demonstrate_workflow_integration():
    """Show how this integrates with the main workflow."""
    
    print("\nWORKFLOW INTEGRATION SUMMARY")
    print("=" * 40)
    
    workflow_info = """
UPDATED RESULTS TAB WORKFLOW:

1. RUN ANALYSES:
   • Template fitting
   • NMF decomposition  
   • ML classification

2. CLICK "🎯 Run Quantitative Analysis":
   • Combines all methods intelligently
   • Detects and corrects ML class flipping
   • Ranks spectra by combined quality metrics

3. RESULTS TAB AUTOMATICALLY SHOWS:
   ✓ Top 5 best-fitting spectra (not random "interesting" ones)
   ✓ Detailed quality information in legend (OUTSIDE plot)
   ✓ Confidence scores, percentages, template fit quality
   ✓ Combined quality rankings from all methods

4. BENEFITS:
   • Validation: Visually inspect the BEST examples
   • Quality control: See why each spectrum was selected
   • Reliability: Based on quantitative analysis, not guesswork
   • Clean display: Legend outside plot area
   
KEY IMPROVEMENTS:
- Legend moved outside plot (no overlap)
- Uses quantitative analysis results (not fallback methods)
- Shows quality metrics for each spectrum
- Ranks by combined quality from all analysis methods
- Provides validation of analysis results
"""
    
    print(workflow_info)

def main():
    """Run the complete updated top spectra demonstration."""
    
    print("UPDATED TOP SPECTRA DISPLAY TEST")
    print("=" * 50)
    print("Testing the new functionality that shows the top 5 best-fitting")
    print("spectra from quantitative analysis with legend outside the plot.")
    
    # Demonstrate the updated plotting
    demonstrate_updated_top_spectra_plot()
    
    # Show workflow integration
    demonstrate_workflow_integration()
    
    print("\n✅ UPDATED FUNCTIONALITY COMPLETE!")
    print("The Results tab now shows the top 5 best-fitting spectra")
    print("from quantitative analysis with the legend outside the plot.")

if __name__ == "__main__":
    main() 