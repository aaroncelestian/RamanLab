#!/usr/bin/env python3
"""
Demonstration: Top Spectra Selection for Results Tab

This script shows how to integrate the TopSpectraSelector with your Results tab
to display the top 5 best-fitting spectra from the positive class.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the analysis directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'analysis'))

from top_spectra_selector import TopSpectraSelector, SpectrumRanking
from quantitative_analysis import RobustQuantitativeAnalyzer, ComponentResult

# Load matplotlib configuration from ui directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ui'))
try:
    import matplotlib_config
except ImportError:
    pass  # Matplotlib config is optional for this demo

def create_simulated_results_data():
    """Create simulated data matching your analysis results."""
    
    print("Creating simulated quantitative analysis results...")
    
    n_pixels = 16383  # Your actual map size
    
    # Simulate quantitative analysis results
    np.random.seed(42)
    
    # Create simulated detection map (sparse, as in your real data)
    detection_prob = 0.003  # ~0.3% detection rate (like your 38-55 out of 16383)
    detection_map = np.random.random(n_pixels) < detection_prob
    n_detected = np.sum(detection_map)
    
    # Create confidence and percentage maps
    confidence_map = np.zeros(n_pixels)
    percentage_map = np.zeros(n_pixels)
    
    # For detected pixels, assign realistic confidence and percentage values
    detected_indices = np.where(detection_map)[0]
    confidence_map[detected_indices] = np.random.beta(2, 1, n_detected) * 0.4 + 0.6  # 0.6-1.0 range
    percentage_map[detected_indices] = np.random.gamma(2, 15) + 20  # 20-80% range
    
    # Create component result
    component_result = ComponentResult(
        component_name="Target Material",
        detection_map=detection_map,
        confidence_map=confidence_map,
        percentage_map=percentage_map,
        intensity_map=confidence_map * percentage_map / 100,
        statistics={'detection_rate': n_detected / n_pixels,
                   'mean_confidence': np.mean(confidence_map[detected_indices]),
                   'mean_percentage': np.mean(percentage_map[detected_indices])},
        method_contributions={'template': confidence_map * 0.4,
                            'nmf': confidence_map * 0.3,
                            'ml': confidence_map * 0.3}
    )
    
    # Simulate template fitting results
    template_r_squared = np.random.random((n_pixels, 1))  # Single template
    template_r_squared[detected_indices] += 0.3  # Boost R² for detected pixels
    template_r_squared = np.clip(template_r_squared, 0, 1)
    
    # Simulate NMF components
    nmf_components = np.random.gamma(1, 2, (n_pixels, 5))  # 5 components
    nmf_components[detected_indices, 2] *= 3  # Boost component 2 for detected pixels
    
    # Simulate ML probabilities (with class flip correction applied)
    ml_probabilities = np.random.random((n_pixels, 2))
    # Normalize to sum to 1
    ml_probabilities = ml_probabilities / ml_probabilities.sum(axis=1, keepdims=True)
    # Boost positive class probability for detected pixels
    ml_probabilities[detected_indices, 1] = np.random.beta(3, 1, n_detected) * 0.4 + 0.6
    ml_probabilities[detected_indices, 0] = 1 - ml_probabilities[detected_indices, 1]
    
    print(f"Created simulated data with {n_detected} detected pixels out of {n_pixels}")
    
    return {
        'component_result': component_result,
        'template_r_squared': template_r_squared,
        'nmf_components': nmf_components,
        'ml_probabilities': ml_probabilities,
        'map_shape': (128, 128),  # Approximate map shape
        'detected_indices': detected_indices
    }

def simulate_spectral_data(n_pixels, detected_indices):
    """Create simulated spectral data for the detected pixels."""
    
    print("Creating simulated spectral data...")
    
    # Create wavenumber axis
    wavenumbers = np.linspace(400, 1800, 500)
    
    # Create base spectrum (background)
    base_spectrum = np.exp(-(wavenumbers - 1000)**2 / (200**2)) * 0.3
    
    # Create target component spectrum
    target_peaks = [500, 750, 1200, 1450]  # Target material peaks
    target_spectrum = np.zeros_like(wavenumbers)
    for peak in target_peaks:
        target_spectrum += np.exp(-(wavenumbers - peak)**2 / (30**2))
    
    # Create full spectral dataset
    intensities = np.zeros((n_pixels, len(wavenumbers)))
    
    # Fill with background spectra + noise
    for i in range(n_pixels):
        noise = np.random.normal(0, 0.1, len(wavenumbers))
        intensities[i] = base_spectrum + noise
        
    # Add target component to detected pixels
    for idx in detected_indices:
        # Varying target component strength
        component_strength = np.random.uniform(0.5, 2.0)
        intensities[idx] += target_spectrum * component_strength
        
    return {
        'wavenumbers': wavenumbers,
        'intensities': intensities
    }

def demonstrate_top_spectra_selection():
    """Demonstrate the top spectra selection process."""
    
    print("DEMONSTRATION: Top Spectra Selection for Results Tab")
    print("=" * 60)
    
    # Create simulated data
    sim_data = create_simulated_results_data()
    spectral_data = simulate_spectral_data(16383, sim_data['detected_indices'])
    
    # Initialize the top spectra selector
    selector = TopSpectraSelector(n_top_spectra=5)
    
    print(f"\nRanking weights: {selector.ranking_weights}")
    
    # Select top spectra
    top_spectra = selector.select_top_spectra(
        quantitative_result=sim_data['component_result'],
        template_r_squared=sim_data['template_r_squared'],
        nmf_components=sim_data['nmf_components'],
        ml_probabilities=sim_data['ml_probabilities'],
        map_shape=sim_data['map_shape'],
        template_index=0,      # First template
        nmf_component_index=2, # Third NMF component
        ml_class_index=1       # Positive class (after flip correction)
    )
    
    # Generate report
    report = selector.generate_top_spectra_report(top_spectra, "Target Material")
    print(f"\n{report}")
    
    # Get spectrum indices for extraction
    spectrum_indices = selector.get_spectrum_indices_for_extraction(top_spectra)
    print(f"\nPixel indices for Results tab: {spectrum_indices}")
    
    return top_spectra, spectral_data, spectrum_indices

def visualize_top_spectra(top_spectra, spectral_data, spectrum_indices):
    """Create visualization showing the top spectra (like Results tab would)."""
    
    print("\nCreating Results tab visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Top 5 Interesting Spectra - Results Tab Display', fontsize=16, fontweight='bold')
    
    # Plot the top 5 spectra
    for i, (ranking, pixel_idx) in enumerate(zip(top_spectra, spectrum_indices)):
        
        row = i // 3
        col = i % 3
        
        if i < 5:  # Only plot first 5
            ax = axes[row, col]
            
            # Extract spectrum
            wavenumbers = spectral_data['wavenumbers']
            intensities = spectral_data['intensities'][pixel_idx]
            
            # Plot spectrum
            ax.plot(wavenumbers, intensities, 'b-', linewidth=1.5)
            
            # Create title with ranking information
            title = f"Rank {i+1}: Pixel {pixel_idx}\n"
            title += f"Conf: {ranking.confidence_score:.2f}, "
            title += f"Pct: {ranking.component_percentage:.1f}%\n"
            title += f"Quality: {ranking.combined_quality_score:.3f}"
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
            
            # Highlight target material peaks
            target_peaks = [500, 750, 1200, 1450]
            for peak in target_peaks:
                ax.axvline(peak, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
    # Remove empty subplot
    if len(top_spectra) < 6:
        axes[1, 2].axis('off')
        
    plt.tight_layout()
    plt.show()
    
    # Create ranking quality summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Quality metrics comparison
    ranks = [f"Rank {i+1}" for i in range(len(top_spectra))]
    confidence_scores = [s.confidence_score for s in top_spectra]
    quality_scores = [s.combined_quality_score for s in top_spectra]
    percentages = [s.component_percentage for s in top_spectra]
    
    x = np.arange(len(ranks))
    width = 0.25
    
    ax1.bar(x - width, confidence_scores, width, label='Confidence', alpha=0.8)
    ax1.bar(x, quality_scores, width, label='Combined Quality', alpha=0.8)
    ax1.bar(x + width, np.array(percentages)/100, width, label='Percentage (scaled)', alpha=0.8)
    
    ax1.set_xlabel('Spectrum Rank')
    ax1.set_ylabel('Score')
    ax1.set_title('Quality Metrics for Top Spectra')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Selection reasons
    reason_counts = {}
    for spectrum in top_spectra:
        for reason in spectrum.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
    reasons = list(reason_counts.keys())[:6]  # Top 6 reasons
    counts = [reason_counts[r] for r in reasons]
    
    ax2.barh(reasons, counts, alpha=0.8)
    ax2.set_xlabel('Number of Spectra')
    ax2.set_title('Most Common Selection Reasons')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_results_tab_integration_example():
    """Show how to integrate this with your actual Results tab UI."""
    
    integration_code = '''
# RESULTS TAB INTEGRATION EXAMPLE
# Add this to your main analysis class:

def update_results_tab_with_top_spectra(self):
    """Update Results tab to show top 5 best-fitting spectra from positive class."""
    
    from analysis.top_spectra_selector import TopSpectraSelector
    
    # Get quantitative analysis results
    if not hasattr(self, 'quantitative_result') or self.quantitative_result is None:
        print("No quantitative analysis results available")
        return
        
    # Initialize selector
    selector = TopSpectraSelector(n_top_spectra=5)
    
    # Gather analysis data
    template_data = getattr(self, 'template_r_squared', None)
    nmf_data = None
    if hasattr(self, 'nmf_analyzer') and self.nmf_analyzer:
        nmf_data = self.nmf_analyzer.get_components()
        
    ml_data = None
    if hasattr(self, 'classification_results'):
        ml_data = self.classification_results.get('probabilities', None)
        
    # Select top spectra
    top_spectra = selector.select_top_spectra(
        quantitative_result=self.quantitative_result,
        template_r_squared=template_data,
        nmf_components=nmf_data,
        ml_probabilities=ml_data,
        map_shape=self.map_shape,
        template_index=0,  # Adjust based on your template index
        nmf_component_index=2,  # Adjust based on your NMF component
        ml_class_index=1  # Use positive class (after flip correction)
    )
    
    # Clear existing Results tab spectra
    self.clear_results_tab()
    
    # Add top spectra to Results tab
    spectrum_indices = selector.get_spectrum_indices_for_extraction(top_spectra)
    
    for i, (ranking, pixel_idx) in enumerate(zip(top_spectra, spectrum_indices)):
        # Extract spectrum data
        wavenumbers = self.original_data['wavenumbers']
        intensities = self.original_data['intensities'][pixel_idx]
        
        # Create informative label
        label = f"Top {i+1}: Pixel {pixel_idx} " \\
               f"(Conf: {ranking.confidence_score:.2f}, " \\
               f"Pct: {ranking.component_percentage:.1f}%)"
        
        # Add to Results tab
        self.add_spectrum_to_results_tab(
            wavenumbers=wavenumbers,
            intensities=intensities,
            label=label,
            pixel_index=pixel_idx
        )
        
    # Add summary information
    report = selector.generate_top_spectra_report(top_spectra, 
                                                 self.quantitative_result.component_name)
    self.update_results_tab_summary(report)
    
    print(f"Added {len(top_spectra)} top spectra to Results tab")

# Add this button to your UI:
def add_top_spectra_button(self):
    """Add button to Results tab for showing top spectra."""
    
    top_spectra_button = tk.Button(
        self.results_frame,
        text="Show Top 5 Best Fits",
        command=self.update_results_tab_with_top_spectra,
        bg='#4CAF50',
        fg='white',
        font=('Arial', 10, 'bold')
    )
    top_spectra_button.pack(pady=5)
    '''
    
    print("RESULTS TAB INTEGRATION CODE:")
    print("=" * 50)
    print(integration_code)

def main():
    """Run the complete demonstration."""
    
    # Demonstrate top spectra selection
    top_spectra, spectral_data, spectrum_indices = demonstrate_top_spectra_selection()
    
    # Visualize results
    visualize_top_spectra(top_spectra, spectral_data, spectrum_indices)
    
    # Show integration example
    create_results_tab_integration_example()
    
    print("\nDEMONSTRATION COMPLETE!")
    print("The TopSpectraSelector successfully identified the 5 best-fitting spectra")
    print("from your quantitative analysis results, ready for Results tab display.")

if __name__ == "__main__":
    main() 