#!/usr/bin/env python3
"""
Test script for applying quantitative analysis to your real data.

This script helps you test the quantitative analysis approach with your 
actual template, NMF, and ML results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import logging
from integrate_quantitative_analysis import QuantitativeAnalysisIntegrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_quantitative_analysis_with_real_data():
    """
    Test the quantitative analysis with your real analysis results.
    
    This function shows you how to apply the quantitative analysis
    to your actual data step by step.
    """
    
    print("TESTING QUANTITATIVE ANALYSIS WITH REAL DATA")
    print("=" * 60)
    
    # Step 1: Check if you have a main window instance
    # If you're running this from your existing UI, you'd pass your main window here
    # For now, we'll work without it and show you how to load results manually
    
    integrator = QuantitativeAnalysisIntegrator(main_window=None)
    
    print("Step 1: Loading your analysis results...")
    print("Note: This script assumes you have results from your existing analyses.")
    print("If you don't have results yet, run template fitting, NMF, or ML classification first.")
    print()
    
    # Step 2: Load results manually (if not using main window)
    # This is where you'd load your actual results
    
    print("Step 2: Manual result loading (modify this section for your data)...")
    
    # Example: Load template results
    # Replace these with your actual file paths and results
    template_results_available = False
    nmf_results_available = False
    ml_results_available = False
    
    # Check for template results
    try:
        # Example: Load from your saved results
        # template_coefficients = np.load('your_template_coefficients.npy')
        # template_r_squared = np.load('your_template_r_squared.npy')
        
        print("  Template results: Not loaded (modify script to load your results)")
        template_results_available = False
        
    except Exception as e:
        print(f"  Template results: Not available ({str(e)})")
    
    # Check for NMF results
    try:
        # Example: Load from your saved results
        # nmf_components = np.load('your_nmf_components.npy')
        # nmf_feature_components = np.load('your_nmf_feature_components.npy')
        
        print("  NMF results: Not loaded (modify script to load your results)")
        nmf_results_available = False
        
    except Exception as e:
        print(f"  NMF results: Not available ({str(e)})")
    
    # Check for ML results
    try:
        # Example: Load from your saved results
        # ml_probabilities = np.load('your_ml_probabilities.npy')
        # ml_predictions = np.load('your_ml_predictions.npy')
        
        print("  ML results: Not loaded (modify script to load your results)")
        ml_results_available = False
        
    except Exception as e:
        print(f"  ML results: Not available ({str(e)})")
    
    print()
    
    # Step 3: Show how to set up the analysis once you have results
    print("Step 3: Setting up quantitative analysis...")
    
    if not any([template_results_available, nmf_results_available, ml_results_available]):
        print("No analysis results available. Here's how to set them up:")
        print()
        print("TEMPLATE RESULTS:")
        print("  If you have template fitting results, load them like this:")
        print("  ```python")
        print("  integrator.analyzer.set_template_results(")
        print("      template_manager=your_template_manager,")
        print("      template_coefficients=your_coefficients,  # Shape: (n_pixels, n_templates)")
        print("      template_r_squared=your_r_squared         # Shape: (n_pixels, n_templates)")
        print("  )")
        print("  ```")
        print()
        
        print("NMF RESULTS:")
        print("  If you have NMF results, load them like this:")
        print("  ```python")
        print("  integrator.analyzer.set_nmf_results(")
        print("      nmf_components=your_nmf_components,              # Shape: (n_pixels, n_components)")
        print("      nmf_feature_components=your_nmf_feature_matrix   # Shape: (n_components, n_features)")
        print("  )")
        print("  ```")
        print()
        
        print("ML RESULTS:")
        print("  If you have ML classification results, load them like this:")
        print("  ```python")
        print("  integrator.analyzer.set_ml_results(")
        print("      ml_probabilities=your_probabilities,  # Shape: (n_pixels, n_classes)")
        print("      ml_predictions=your_predictions       # Shape: (n_pixels,)")
        print("  )")
        print("  ```")
        print()
        
        print("EXAMPLE ANALYSIS:")
        print("  Once you have results loaded, analyze like this:")
        print("  ```python")
        print("  result = integrator.analyze_component_by_name(")
        print("      component_name='Polypropylene',")
        print("      template_name='your_template_name',     # Optional")
        print("      nmf_component_index=2,                  # Optional")
        print("      ml_class_name='plastic1',               # Optional")
        print("      confidence_threshold=0.3                # Adjust as needed")
        print("  )")
        print("  ")
        print("  if result:")
        print("      print(f'Component detected in {result.statistics[\"detection_percentage\"]:.1f}% of pixels')")
        print("      print(f'Average percentage: {result.statistics[\"mean_percentage_detected\"]:.1f}%')")
        print("  ```")
        print()
        
        return False
    
    # If we have results, proceed with analysis
    print("Analysis results available - proceeding with quantitative analysis...")
    
    # Step 4: Run the analysis
    print("Step 4: Running quantitative analysis...")
    
    # This is where you'd run the actual analysis with your data
    # result = integrator.analyze_component_by_name(
    #     component_name="YourComponent",
    #     template_name="your_template",
    #     nmf_component_index=2,
    #     ml_class_name="your_class",
    #     confidence_threshold=0.3
    # )
    
    return True

def create_example_integration_code():
    """Create example code for integration with your existing workflow."""
    
    example_code = '''
# Example: Integration with your existing analysis workflow

def run_quantitative_analysis_on_real_data():
    """
    Run quantitative analysis on your real data.
    Modify this function to match your data and workflow.
    """
    
    from integrate_quantitative_analysis import QuantitativeAnalysisIntegrator
    
    # Initialize integrator
    integrator = QuantitativeAnalysisIntegrator()
    
    # Load your template results (if available)
    if you_have_template_results:
        integrator.analyzer.set_template_results(
            template_manager=your_template_manager,
            template_coefficients=your_template_coefficients,
            template_r_squared=your_template_r_squared
        )
    
    # Load your NMF results (if available)
    if you_have_nmf_results:
        integrator.analyzer.set_nmf_results(
            nmf_components=your_nmf_components,
            nmf_feature_components=your_nmf_feature_components
        )
    
    # Load your ML results (if available)
    if you_have_ml_results:
        integrator.analyzer.set_ml_results(
            ml_probabilities=your_ml_probabilities,
            ml_predictions=your_ml_predictions
        )
    
    # Analyze your component of interest
    result = integrator.analyze_component_by_name(
        component_name="Polypropylene",           # Your component name
        template_name="PP_template",              # Your template name
        nmf_component_index=2,                    # NMF component index
        ml_class_name="plastic1",                 # ML class name
        confidence_threshold=0.3                  # Adjust for sensitivity
    )
    
    if result:
        # Print quantitative results
        stats = result.statistics
        print(f"Quantitative Analysis Results for {result.component_name}:")
        print(f"  Total pixels analyzed: {stats['total_pixels']:,}")
        print(f"  Pixels with component detected: {stats['detected_pixels']:,}")
        print(f"  Detection percentage: {stats['detection_percentage']:.2f}%")
        print(f"  Average component percentage: {stats['mean_percentage_detected']:.1f}% ± {stats['std_percentage_detected']:.1f}%")
        print(f"  Confidence threshold used: {stats['confidence_threshold_used']:.2f}")
        
        # Create maps for visualization
        maps = integrator.create_analysis_maps(result, your_map_shape)
        
        # Use the maps in your visualization
        # maps['intensity']   - Component intensity map
        # maps['confidence']  - Confidence map
        # maps['percentage']  - Percentage map
        # maps['detection']   - Binary detection map
        
        return result, maps
    
    return None, None

# Example usage in your main analysis pipeline:
# result, maps = run_quantitative_analysis_on_real_data()
'''
    
    return example_code

def main():
    """Main function."""
    
    # Test with real data
    success = test_quantitative_analysis_with_real_data()
    
    if not success:
        print("TO GET STARTED WITH YOUR REAL DATA:")
        print("-" * 40)
        print("1. Run your existing template fitting, NMF, or ML classification")
        print("2. Save the results to numpy arrays or make them accessible")
        print("3. Modify this script to load your actual results")
        print("4. Run the quantitative analysis")
        print()
        print("EXAMPLE INTEGRATION CODE:")
        print("-" * 40)
        print(create_example_integration_code())
        
    print("\nREADY TO HELP!")
    print("=" * 40)
    print("If you have questions about:")
    print("• Loading your specific results")
    print("• Adjusting parameters for your data")
    print("• Interpreting the quantitative results")
    print("• Integrating with your existing UI")
    print()
    print("Just let me know what specific help you need!")

if __name__ == "__main__":
    main() 