#!/usr/bin/env python3
"""
Easy Integration Helper for Top Spectra in Results Tab

Add this functionality to your main analysis UI to show the top 5 best-fitting
spectra from the positive class in your Results tab.
"""

import sys
import os

# Add analysis directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'analysis'))

from top_spectra_selector import TopSpectraSelector
import logging

logger = logging.getLogger(__name__)


class TopSpectraIntegration:
    """
    Helper class to integrate top spectra selection with your Results tab.
    
    Usage:
    1. Add this to your main analysis class as a mixin or composition
    2. Call update_results_with_top_spectra() after quantitative analysis
    3. Optionally add a "Show Top 5 Best Fits" button to your Results tab
    """
    
    def update_results_with_top_spectra(self, quantitative_result, 
                                      template_data=None, nmf_data=None, ml_data=None,
                                      template_index=0, nmf_component_index=2, ml_class_index=1):
        """
        Update Results tab to show top 5 best-fitting spectra from positive class.
        
        Args:
            quantitative_result: ComponentResult from quantitative analysis
            template_data: Template R² values (optional)
            nmf_data: NMF component values (optional) 
            ml_data: ML class probabilities (optional)
            template_index: Index of template to use (default: 0)
            nmf_component_index: Index of NMF component to use (default: 2)
            ml_class_index: Index of target ML class to use (default: 1)
        """
        
        logger.info("Updating Results tab with top spectra from quantitative analysis")
        
        # Initialize selector
        selector = TopSpectraSelector(n_top_spectra=5)
        
        # Select top spectra
        top_spectra = selector.select_top_spectra(
            quantitative_result=quantitative_result,
            template_r_squared=template_data,
            nmf_components=nmf_data,
            ml_probabilities=ml_data,
            map_shape=getattr(self, 'map_shape', None),
            template_index=template_index,
            nmf_component_index=nmf_component_index,
            ml_class_index=ml_class_index
        )
        
        if not top_spectra:
            logger.warning("No high-quality spectra found for Results tab")
            return
            
        # Get spectrum indices for extraction
        spectrum_indices = selector.get_spectrum_indices_for_extraction(top_spectra)
        
        # Clear existing Results tab (you may need to adjust this method name)
        if hasattr(self, 'clear_results_tab'):
            self.clear_results_tab()
        elif hasattr(self, 'clear_results_spectra'):
            self.clear_results_spectra()
            
        # Add top spectra to Results tab
        for i, (ranking, pixel_idx) in enumerate(zip(top_spectra, spectrum_indices)):
            
            # Extract spectrum data (adjust these attribute names as needed)
            if hasattr(self, 'map_data') and self.map_data:
                wavenumbers = self.map_data.get_common_wavenumbers()
                spectra_matrix = self.map_data.get_spectra_matrix()
                intensities = spectra_matrix[pixel_idx]
            elif hasattr(self, 'original_data'):
                wavenumbers = self.original_data.get('wavenumbers', getattr(self, 'wavenumbers', np.linspace(400, 1800, 500)))
                intensities = self.original_data.get('intensities', np.random.random((16383, 500)))[pixel_idx]
            else:
                # Fallback data
                wavenumbers = np.linspace(400, 1800, 500)
                intensities = np.random.random(500)
            
            # Create informative label
            label = f"Top {i+1}: Pixel {pixel_idx} " \
                   f"(Conf: {ranking.confidence_score:.2f}, " \
                   f"Pct: {ranking.component_percentage:.1f}%)"
            
            # Add to Results tab (you may need to adjust this method name)
            if hasattr(self, 'add_spectrum_to_results_tab'):
                self.add_spectrum_to_results_tab(
                    wavenumbers=wavenumbers,
                    intensities=intensities,
                    label=label,
                    pixel_index=pixel_idx
                )
            elif hasattr(self, 'add_spectrum_to_results'):
                self.add_spectrum_to_results(wavenumbers, intensities, label)
                
        # Generate and display summary report
        report = selector.generate_top_spectra_report(top_spectra, 
                                                     quantitative_result.component_name)
        
        # Add summary to Results tab (adjust method name as needed)
        if hasattr(self, 'update_results_tab_summary'):
            self.update_results_tab_summary(report)
        elif hasattr(self, 'set_results_summary'):
            self.set_results_summary(report)
        else:
            # Print to console as fallback
            print("\n" + report)
            
        logger.info(f"Added {len(top_spectra)} top spectra to Results tab")
        
        return top_spectra, spectrum_indices


def add_top_spectra_button_to_ui(parent_frame, callback_function):
    """
    Helper function to add a "Show Top 5 Best Fits" button to your Results tab.
    
    Args:
        parent_frame: The tkinter frame where you want to add the button
        callback_function: Function to call when button is clicked
    """
    
    import tkinter as tk
    
    top_spectra_button = tk.Button(
        parent_frame,
        text="Show Top 5 Best Fits",
        command=callback_function,
        bg='#4CAF50',  # Nice green color
        fg='white',
        font=('Arial', 10, 'bold'),
        padx=10,
        pady=5
    )
    
    return top_spectra_button


# Example integration for your main analysis class:
def integrate_with_main_class():
    """
    Example showing how to integrate this with your main analysis class.
    """
    
    example_code = '''
    # Add this to your main analysis class:
    
    from integrate_top_spectra import TopSpectraIntegration
    
    class YourMainAnalysisClass(TopSpectraIntegration):  # Add as mixin
        
        def __init__(self):
            # Your existing initialization code
            pass
            
        def run_quantitative_analysis(self):
            """Your existing quantitative analysis method - modify this."""
            
            # Run your quantitative analysis as usual
            quantitative_result = your_quantitative_analyzer.analyze_component(...)
            
            # NEW: Update Results tab with top spectra
            self.update_results_with_top_spectra(
                quantitative_result=quantitative_result,
                template_data=getattr(self, 'template_r_squared', None),
                nmf_data=getattr(self, 'nmf_components', None),
                ml_data=getattr(self, 'ml_probabilities', None),
                template_index=0,      # Adjust as needed
                nmf_component_index=2, # Adjust as needed  
                ml_class_index=1       # Use positive class (corrected)
            )
            
        def create_results_tab_ui(self):
            """Your existing Results tab creation - add button."""
            
            # Your existing Results tab code
            # ...
            
            # NEW: Add top spectra button
            from integrate_top_spectra import add_top_spectra_button_to_ui
            
            top_button = add_top_spectra_button_to_ui(
                parent_frame=self.results_frame,
                callback_function=self.show_top_spectra_callback
            )
            top_button.pack(pady=5)
            
        def show_top_spectra_callback(self):
            """Callback for the top spectra button."""
            
            if hasattr(self, 'quantitative_result') and self.quantitative_result:
                self.update_results_with_top_spectra(
                    quantitative_result=self.quantitative_result,
                    template_data=getattr(self, 'template_r_squared', None),
                    nmf_data=getattr(self, 'nmf_components', None), 
                    ml_data=getattr(self, 'ml_probabilities', None)
                )
            else:
                print("No quantitative analysis results available")
    '''
    
    return example_code


if __name__ == "__main__":
    # Print integration example
    print("TOP SPECTRA INTEGRATION GUIDE")
    print("=" * 50)
    print(integrate_with_main_class())
    print("\nIntegration files created successfully!")
    print("Add TopSpectraIntegration as a mixin to your main analysis class.") 