#!/usr/bin/env python3
"""
Demo: What happens after clicking "🎯 Run Quantitative Analysis" button

This shows the expected behavior after the user clicks the quantitative analysis button:
1. Results tab updates automatically
2. Shows top 5 best-fitting spectra (not fallback)
3. Legend appears outside plot
4. Detailed quality metrics displayed
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import sys
import os

# Add analysis path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis'))

@dataclass
class SpectrumRanking:
    """Mock spectrum ranking result for demo."""
    pixel_index: int
    confidence_score: float
    component_percentage: float
    combined_quality_score: float
    template_fit_quality: Optional[float] = None
    selection_reason: str = "high_confidence"

def demo_quantitative_analysis_button_click():
    """Demonstrate what happens after clicking the quantitative analysis button."""
    
    print("🎯 QUANTITATIVE ANALYSIS BUTTON CLICKED!")
    print("=" * 60)
    
    # Simulate what the system does automatically
    print("1. Extracting template analysis results...")
    print("2. Extracting NMF analysis results...")  
    print("3. Extracting ML classification results...")
    print("4. Running class flip detection...")
    print("5. Combining methods with quality scoring...")
    print("6. Selecting top 5 best-fitting spectra...")
    print("7. Updating Results tab automatically...")
    
    # Mock quantitative analysis results (what the button produces)
    top_spectra = [
        SpectrumRanking(
            pixel_index=1247,
            confidence_score=0.89,
            component_percentage=76.3,
            combined_quality_score=0.856,
            template_fit_quality=0.91,
            selection_reason="highest_template_fit"
        ),
        SpectrumRanking(
            pixel_index=3891,
            confidence_score=0.82,
            component_percentage=71.2,
            combined_quality_score=0.823,
            template_fit_quality=0.87,
            selection_reason="high_confidence_combined"
        ),
        SpectrumRanking(
            pixel_index=7256,
            confidence_score=0.78,
            component_percentage=68.9,
            combined_quality_score=0.798,
            template_fit_quality=0.84,
            selection_reason="strong_nmf_agreement"
        ),
        SpectrumRanking(
            pixel_index=2103,
            confidence_score=0.75,
            component_percentage=65.1,
            combined_quality_score=0.771,
            template_fit_quality=0.81,
            selection_reason="consistent_across_methods"
        ),
        SpectrumRanking(
            pixel_index=9437,
            confidence_score=0.71,
            component_percentage=62.8,
            combined_quality_score=0.748,
            template_fit_quality=0.78,
            selection_reason="good_ml_probability"
        )
    ]
    
    # Generate mock spectral data
    wavenumbers = np.linspace(400, 1800, 500)
    
    # Create the plot that would appear in Results tab
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Adjust layout for external legend
    fig.subplots_adjust(right=0.75)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_spectra)))
    
    # Plot each spectrum with realistic data
    for i, (ranking, color) in enumerate(zip(top_spectra, colors)):
        # Generate realistic Raman spectrum
        # Base spectrum with major peaks
        intensities = (
            0.3 * np.exp(-((wavenumbers - 1000) / 50)**2) +  # Major peak at 1000
            0.5 * np.exp(-((wavenumbers - 1300) / 80)**2) +  # Strong peak at 1300
            0.2 * np.exp(-((wavenumbers - 800) / 40)**2) +   # Medium peak at 800
            0.15 * np.exp(-((wavenumbers - 1600) / 60)**2) + # Smaller peak at 1600
            0.1 * np.random.random(len(wavenumbers))          # Noise
        )
        
        # Add some variation per spectrum
        intensities *= (0.8 + 0.4 * np.random.random())
        
        # Normalize and offset for display
        intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        intensities_norm += i * 0.35
        
        # Plot spectrum
        ax.plot(wavenumbers, intensities_norm, color=color, linewidth=1.8, alpha=0.85)
    
    # Create detailed legend labels (what you'd see after clicking button)
    legend_labels = []
    for i, ranking in enumerate(top_spectra):
        label = f"Rank {i+1}: Pixel {ranking.pixel_index}\n"
        label += f"Conf: {ranking.confidence_score:.2f}, "
        label += f"Pct: {ranking.component_percentage:.1f}%\n"
        label += f"Quality: {ranking.combined_quality_score:.3f}"
        if ranking.template_fit_quality:
            label += f", R²: {ranking.template_fit_quality:.2f}"
        legend_labels.append(label)
    
    # External legend (key improvement!)
    legend = ax.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left',
                       fontsize=8, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Normalized Intensity (offset)')
    ax.set_title('Top 5 Best-Fitting Spectra (After Quantitative Analysis)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ RESULTS TAB UPDATED!")
    print("=" * 60)
    print("BEFORE clicking button: 'Top 5 Spectra (Fallback Method)'")
    print("AFTER clicking button:  'Top 5 Best-Fitting Spectra'")
    print("\nKEY IMPROVEMENTS:")
    print("• Legend moved outside plot area (no overlap)")
    print("• Shows actual best-fitting spectra (not random)")
    print("• Detailed quality metrics in legend")
    print("• Ranking based on combined analysis methods")
    print("• Automatic class flip detection applied")
    
    print(f"\nDETECTION SUMMARY:")
    print(f"• Found {len(top_spectra)} high-quality spectra")
    print(f"• Best confidence: {max(s.confidence_score for s in top_spectra):.2f}")
    print(f"• Best template fit: {max(s.template_fit_quality for s in top_spectra):.2f}")
    print(f"• Average quality: {np.mean([s.combined_quality_score for s in top_spectra]):.3f}")

if __name__ == "__main__":
    demo_quantitative_analysis_button_click() 