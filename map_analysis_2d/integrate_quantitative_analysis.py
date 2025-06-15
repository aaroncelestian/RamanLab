#!/usr/bin/env python3
"""  
Integration script for Robust Quantitative Analysis

This script shows how to integrate the new quantitative analysis approach
with your existing map analysis workflow and UI.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import logging
from typing import Dict, Optional, List, Tuple

# Handle imports for both running from map_analysis_2d directory and from parent directory
try:
    from analysis.quantitative_analysis import RobustQuantitativeAnalyzer, ComponentResult
except ImportError:
    try:
        from map_analysis_2d.analysis.quantitative_analysis import RobustQuantitativeAnalyzer, ComponentResult
    except ImportError:
        # As a last resort, try absolute import with current directory in path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from analysis.quantitative_analysis import RobustQuantitativeAnalyzer, ComponentResult

logger = logging.getLogger(__name__)


class QuantitativeAnalysisIntegrator:
    """
    Integrates quantitative analysis with existing map analysis workflow.
    
    This class handles the extraction of results from your existing analyses
    (template fitting, NMF, ML classification) and provides a unified interface
    for quantitative component analysis.
    """
    
    def __init__(self, main_window=None):
        """
        Initialize the integrator.
        
        Args:
            main_window: Reference to main UI window (optional)
        """
        self.main_window = main_window
        self.analyzer = RobustQuantitativeAnalyzer()
        self.available_methods = {}
        self.last_results = {}
        
    def extract_template_results(self) -> bool:
        """
        Extract template fitting results from the main window.
        
        Returns:
            True if template results were successfully extracted
        """
        try:
            if not self.main_window:
                logger.warning("No main window reference - cannot extract template results")
                return False
                
            # Check if template analysis has been run
            if not hasattr(self.main_window, 'template_coefficients'):
                logger.info("No template analysis results found")
                return False
                
            # Extract template data
            template_coeffs = getattr(self.main_window, 'template_coefficients', None)
            template_r2 = getattr(self.main_window, 'template_r_squared', None)
            template_manager = getattr(self.main_window, 'template_manager', None)
            
            if template_coeffs is not None and template_r2 is not None:
                self.analyzer.set_template_results(
                    template_manager=template_manager,
                    template_coefficients=template_coeffs,
                    template_r_squared=template_r2
                )
                
                self.available_methods['template'] = {
                    'n_templates': template_coeffs.shape[1] if len(template_coeffs.shape) > 1 else 1,
                    'template_names': template_manager.get_template_names() if template_manager else []
                }
                
                logger.info(f"Extracted template results: {self.available_methods['template']['n_templates']} templates")
                return True
            
        except Exception as e:
            logger.error(f"Failed to extract template results: {str(e)}")
            
        return False
        
    def extract_nmf_results(self) -> bool:
        """
        Extract NMF results from the main window.
        
        Returns:
            True if NMF results were successfully extracted
        """
        try:
            if not self.main_window:
                logger.warning("No main window reference - cannot extract NMF results")
                return False
                
            # Look for NMF analyzer in main window
            nmf_analyzer = getattr(self.main_window, 'nmf_analyzer', None)
            
            if nmf_analyzer and hasattr(nmf_analyzer, 'components'):
                components = nmf_analyzer.get_components()
                feature_components = nmf_analyzer.get_feature_components()
                
                if components is not None and feature_components is not None:
                    self.analyzer.set_nmf_results(
                        nmf_components=components,
                        nmf_feature_components=feature_components
                    )
                    
                    self.available_methods['nmf'] = {
                        'n_components': components.shape[1],
                        'component_names': [f"NMF Component {i+1}" for i in range(components.shape[1])]
                    }
                    
                    logger.info(f"Extracted NMF results: {self.available_methods['nmf']['n_components']} components")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to extract NMF results: {str(e)}")
            
        return False
        
    def extract_ml_results(self) -> bool:
        """
        Extract ML classification results from the main window.
        
        Returns:
            True if ML results were successfully extracted
        """
        try:
            if not self.main_window:
                logger.warning("No main window reference - cannot extract ML results")
                return False
                
            # Look for ML results in main window
            ml_results = getattr(self.main_window, 'classification_results', None)
            ml_analyzer = getattr(self.main_window, 'supervised_analyzer', None)
            
            if ml_results and 'probabilities' in ml_results:
                probabilities = ml_results['probabilities']
                predictions = ml_results.get('predictions', np.argmax(probabilities, axis=1))
                
                self.analyzer.set_ml_results(
                    ml_probabilities=probabilities,
                    ml_predictions=predictions
                )
                
                # Get class names if available
                class_names = []
                if ml_analyzer and hasattr(ml_analyzer, 'class_names'):
                    class_names = ml_analyzer.class_names
                else:
                    class_names = [f"Class {i}" for i in range(probabilities.shape[1])]
                
                self.available_methods['ml'] = {
                    'n_classes': probabilities.shape[1],
                    'class_names': class_names
                }
                
                logger.info(f"Extracted ML results: {self.available_methods['ml']['n_classes']} classes")
                return True
                
        except Exception as e:
            logger.error(f"Failed to extract ML results: {str(e)}")
            
        return False
        
    def auto_extract_all_results(self) -> Dict[str, bool]:
        """
        Automatically extract all available results from the main window.
        
        Returns:
            Dictionary indicating which methods were successfully extracted
        """
        extraction_results = {
            'template': self.extract_template_results(),
            'nmf': self.extract_nmf_results(),
            'ml': self.extract_ml_results()
        }
        
        # Set map info if available
        if self.main_window:
            try:
                map_shape = getattr(self.main_window, 'map_shape', None)
                wavenumbers = getattr(self.main_window, 'wavenumbers', None)
                
                if map_shape and wavenumbers is not None:
                    self.analyzer.set_map_info(map_shape, wavenumbers)
                    
            except Exception as e:
                logger.warning(f"Could not extract map info: {str(e)}")
        
        n_methods = sum(extraction_results.values())
        logger.info(f"Successfully extracted {n_methods} analysis methods")
        
        return extraction_results
        
    def get_available_analysis_options(self) -> Dict[str, List[Dict]]:
        """
        Get available analysis options for the user interface.
        
        Returns:
            Dictionary of available options for each method
        """
        options = {}
        
        if 'template' in self.available_methods:
            template_info = self.available_methods['template']
            options['templates'] = []
            for i in range(template_info['n_templates']):
                name = (template_info['template_names'][i] 
                       if i < len(template_info['template_names']) 
                       else f"Template {i+1}")
                options['templates'].append({
                    'index': i,
                    'name': name,
                    'display_name': f"{name} (Template {i+1})"
                })
                
        if 'nmf' in self.available_methods:
            nmf_info = self.available_methods['nmf']
            options['nmf_components'] = []
            for i in range(nmf_info['n_components']):
                options['nmf_components'].append({
                    'index': i,
                    'name': f"Component {i+1}",
                    'display_name': f"NMF Component {i+1}"
                })
                
        if 'ml' in self.available_methods:
            ml_info = self.available_methods['ml']
            options['ml_classes'] = []
            for i, class_name in enumerate(ml_info['class_names']):
                options['ml_classes'].append({
                    'index': i,
                    'name': class_name,
                    'display_name': f"{class_name} (Class {i+1})"
                })
                
        return options
        
    def analyze_component_by_name(self, component_name: str,
                                 template_name: Optional[str] = None,
                                 nmf_component_index: Optional[int] = None,
                                 ml_class_name: Optional[str] = None,
                                 confidence_threshold: float = 0.5) -> Optional[ComponentResult]:
        """
        Analyze a component using named parameters.
        
        Args:
            component_name: Name for the component being analyzed
            template_name: Name of template to use (if any)
            nmf_component_index: Index of NMF component to use (if any)
            ml_class_name: Name of ML class to use (if any)
            confidence_threshold: Confidence threshold for detection
            
        Returns:
            ComponentResult or None if analysis failed
        """
        try:
            # Set confidence threshold
            self.analyzer.confidence_threshold = confidence_threshold
            
            # Convert names to indices
            template_index = None
            if template_name and 'template' in self.available_methods:
                template_names = self.available_methods['template']['template_names']
                try:
                    template_index = template_names.index(template_name)
                except (ValueError, AttributeError):
                    logger.warning(f"Template '{template_name}' not found")
                    
            ml_class_index = None
            if ml_class_name and 'ml' in self.available_methods:
                class_names = self.available_methods['ml']['class_names']
                try:
                    ml_class_index = class_names.index(ml_class_name)
                except (ValueError, AttributeError):
                    logger.warning(f"ML class '{ml_class_name}' not found")
                    
            # Perform analysis
            result = self.analyzer.analyze_component(
                component_name=component_name,
                template_index=template_index,
                nmf_component=nmf_component_index,
                target_class_index=ml_class_index
            )
            
            # Store result
            self.last_results[component_name] = result
            
            logger.info(f"Successfully analyzed component '{component_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze component '{component_name}': {str(e)}")
            return None
            
    def create_analysis_maps(self, result: ComponentResult, 
                           map_shape: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
        """
        Create 2D maps from analysis results.
        
        Args:
            result: ComponentResult from analysis
            map_shape: Shape for 2D maps (if not set in analyzer)
            
        Returns:
            Dictionary of 2D maps
        """
        if map_shape is None:
            map_shape = self.analyzer.map_shape
            
        if map_shape is None:
            # Try to infer square-ish shape
            n_pixels = len(result.intensity_map)
            map_size = int(np.sqrt(n_pixels))
            if map_size * map_size < n_pixels:
                map_size += 1
            map_shape = (map_size, map_size)
            
        n_pixels = len(result.intensity_map)
        target_pixels = map_shape[0] * map_shape[1]
        
        maps = {}
        
        for map_name, data in [
            ('intensity', result.intensity_map),
            ('confidence', result.confidence_map),
            ('percentage', result.percentage_map),
            ('detection', result.detection_map.astype(float))
        ]:
            if len(data) == target_pixels:
                # Data fits exactly
                maps[map_name] = data.reshape(map_shape)
            elif len(data) < target_pixels:
                # Pad data
                padded = np.pad(data, (0, target_pixels - len(data)), 
                              mode='constant', constant_values=0)
                maps[map_name] = padded.reshape(map_shape)
            else:
                # Truncate data
                maps[map_name] = data[:target_pixels].reshape(map_shape)
                
        return maps
        
    def generate_analysis_summary(self, results: List[ComponentResult]) -> str:
        """
        Generate a comprehensive analysis summary.
        
        Args:
            results: List of ComponentResults
            
        Returns:
            Formatted summary string
        """
        if not results:
            return "No analysis results available."
            
        summary = self.analyzer.generate_summary_report(results)
        
        # Add method availability info
        method_info = ["\nAVAILABLE ANALYSIS METHODS:"]
        method_info.append("-" * 40)
        
        for method, info in self.available_methods.items():
            if method == 'template':
                method_info.append(f"Template Fitting: {info['n_templates']} templates loaded")
            elif method == 'nmf':
                method_info.append(f"NMF Analysis: {info['n_components']} components available")
            elif method == 'ml':
                method_info.append(f"ML Classification: {info['n_classes']} classes trained")
                
        if not self.available_methods:
            method_info.append("No analysis methods currently available")
            
        return summary + "\n" + "\n".join(method_info)


def create_ui_integration_example():
    """
    Example of how to integrate with your existing UI.
    This would be called from your main UI code.
    """
    # Example usage in your main UI class:
    
    code_example = '''
    # In your main UI class (e.g., in main_window.py)
    
    def setup_quantitative_analysis(self):
        """Set up quantitative analysis integration."""
        self.quantitative_integrator = QuantitativeAnalysisIntegrator(main_window=self)
        
    def run_quantitative_analysis(self):
        """Run quantitative analysis using available methods."""
        
        # Extract all available results
        extraction_results = self.quantitative_integrator.auto_extract_all_results()
        
        if not any(extraction_results.values()):
            self.show_error_dialog("No analysis results available. Please run template fitting, NMF, or ML classification first.")
            return
            
        # Get available options for user selection
        options = self.quantitative_integrator.get_available_analysis_options()
        
        # Show dialog for user to select component and methods
        dialog = QuantitativeAnalysisDialog(options, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            
            # Get user selections
            component_name = dialog.get_component_name()
            template_name = dialog.get_selected_template()
            nmf_component = dialog.get_selected_nmf_component()
            ml_class = dialog.get_selected_ml_class()
            confidence_threshold = dialog.get_confidence_threshold()
            
            # Run analysis
            result = self.quantitative_integrator.analyze_component_by_name(
                component_name=component_name,
                template_name=template_name,
                nmf_component_index=nmf_component,
                ml_class_name=ml_class,
                confidence_threshold=confidence_threshold
            )
            
            if result:
                # Create maps for visualization
                maps = self.quantitative_integrator.create_analysis_maps(result, self.map_shape)
                
                # Add to map dropdown
                self.add_quantitative_maps_to_dropdown(component_name, maps)
                
                # Show summary
                summary = self.quantitative_integrator.generate_analysis_summary([result])
                self.show_analysis_summary_dialog(summary)
                
    def add_quantitative_maps_to_dropdown(self, component_name, maps):
        """Add quantitative analysis maps to the visualization dropdown."""
        
        for map_type, map_data in maps.items():
            feature_name = f"Quantitative {component_name}: {map_type.title()}"
            
            # Store map data (you'd adapt this to your existing map storage system)
            self.quantitative_maps[feature_name] = map_data
            
            # Add to dropdown
            if hasattr(self, 'map_feature_dropdown'):
                self.map_feature_dropdown.addItem(feature_name)
    '''
    
    return code_example


if __name__ == "__main__":
    print("QUANTITATIVE ANALYSIS INTEGRATION")
    print("=" * 50)
    print("This module provides integration between the robust quantitative")
    print("analysis and your existing map analysis workflow.")
    print()
    print("Key features:")
    print("• Automatic extraction of results from template, NMF, and ML analyses")
    print("• Unified interface for component analysis")
    print("• Integration with existing UI workflow")
    print("• Automatic map creation for visualization")
    print()
    print("Example integration code:")
    print("-" * 30)
    print(create_ui_integration_example()) 