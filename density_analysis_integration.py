"""
Density Analysis Integration Module for RamanLab Batch Peak Fitting

This module provides density analysis functionality that can be integrated
with batch peak fitting results.
"""

import os
import sys
import numpy as np
from PySide6.QtWidgets import QMessageBox

# Check if Density module is available
DENSITY_AVAILABLE = False
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Density'))
    from advanced_analysis.density_gui_launcher import DensityAnalysisGUI, CustomMaterialDialog
    from advanced_analysis.raman_density_analysis import MaterialConfigs
    DENSITY_AVAILABLE = True
    print("✓ Density analysis module loaded successfully")
except ImportError as e:
    print(f"⚠️ Density analysis not available: {e}")
    DENSITY_AVAILABLE = False


class DensityAnalysisIntegrator:
    """Handles integration between batch peak fitting and density analysis."""
    
    def __init__(self, parent_window):
        self.parent = parent_window
    
    def is_available(self):
        """Check if density analysis is available."""
        return DENSITY_AVAILABLE
    
    def launch_density_analysis(self):
        """Launch the standalone density analysis GUI."""
        if not DENSITY_AVAILABLE:
            QMessageBox.warning(self.parent, "Module Not Available", 
                              "Density analysis module is not available.\n"
                              "Please check if the Density folder is present in your RamanLab directory.")
            return
        
        try:
            density_gui = DensityAnalysisGUI()
            density_gui.show()
            QMessageBox.information(self.parent, "Launched", 
                                  "Density Analysis tool has been launched in a new window.")
        except Exception as e:
            QMessageBox.critical(self.parent, "Launch Error", 
                               f"Failed to launch density analysis:\n{str(e)}")
    
    def quick_density_analysis(self):
        """Launch density analysis with batch results if available."""
        if not DENSITY_AVAILABLE:
            QMessageBox.warning(self.parent, "Module Not Available", 
                              "Density analysis module is not available.")
            return
        
        try:
            density_gui = DensityAnalysisGUI()
            
            # If batch results are available, pass them
            if hasattr(self.parent, 'batch_results') and self.parent.batch_results:
                density_gui.set_batch_fitting_results(self.parent.batch_results)
                QMessageBox.information(self.parent, "Integration", 
                                      "Density analysis launched with your batch fitting results!")
            else:
                QMessageBox.information(self.parent, "Launched", 
                                      "Density analysis launched. Run batch fitting first to integrate results.")
            
            density_gui.show()
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Launch Error", 
                               f"Failed to launch density analysis:\n{str(e)}")
    
    def configure_custom_material_from_results(self):
        """Configure custom material for density analysis from batch fitting results."""
        if not DENSITY_AVAILABLE:
            QMessageBox.warning(self.parent, "Module Not Available", 
                              "Density analysis module is not available.")
            return
        
        if not hasattr(self.parent, 'batch_results') or not self.parent.batch_results:
            QMessageBox.warning(self.parent, "No Results", 
                              "No batch fitting results available.\nRun batch processing first.")
            return
        
        try:
            # Analyze batch results to extract material parameters
            analysis_results = self._analyze_batch_for_material_config()
            
            if not analysis_results:
                QMessageBox.warning(self.parent, "Insufficient Data", 
                    "Could not extract sufficient peak data from batch results.\n"
                    "Ensure your batch fitting has successfully fitted peaks.")
                return
            
            # Create material configuration dialog pre-populated with batch data
            dialog = CustomMaterialDialog(self.parent)
            dialog.setWindowTitle("Custom Material from Batch Results")
            
            # Pre-populate with analyzed data
            self._populate_dialog_from_batch(dialog, analysis_results)
            
            # Show preview information
            preview_text = self._create_batch_analysis_preview(analysis_results)
            QMessageBox.information(self.parent, "Batch Analysis Results", preview_text)
            
            if dialog.exec() == dialog.Accepted:
                custom_config = dialog.get_config()
                
                # Launch density analysis with the custom configuration
                try:
                    # Create density analysis window
                    density_gui = DensityAnalysisGUI()
                    
                    # Set the custom configuration
                    if custom_config:
                        material_name = custom_config['name']
                        MaterialConfigs.add_custom_material(material_name, custom_config)
                        
                        # Update the material selection
                        density_gui.material_combo.clear()
                        density_gui.material_combo.addItems(MaterialConfigs.get_available_materials())
                        density_gui.material_combo.setCurrentText(material_name)
                        density_gui.on_material_changed()
                        
                        # Pass batch results for reference intensity calculation
                        density_gui.set_batch_fitting_results(self.parent.batch_results)
                    
                    # Show the density analysis GUI
                    density_gui.show()
                    
                    QMessageBox.information(self.parent, "Success", 
                        f"Custom material '{custom_config['name']}' created and density analysis launched!\n\n"
                        f"Your batch fitting results are now available for reference intensity calculation.")
                    
                except Exception as e:
                    QMessageBox.critical(self.parent, "Launch Error", 
                        f"Failed to launch density analysis:\n{str(e)}")
                
        except Exception as e:
            QMessageBox.critical(self.parent, "Configuration Error", 
                f"Failed to configure custom material from batch results:\n{str(e)}")
    
    def _analyze_batch_for_material_config(self):
        """Analyze batch results to extract material configuration parameters."""
        successful_results = [r for r in self.parent.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            return None
        
        # Extract peak information
        all_peaks = []
        peak_intensities = []
        
        for result in successful_results:
            if 'peaks' in result and result['peaks']:
                for peak in result['peaks']:
                    all_peaks.append(peak['center'])
                    peak_intensities.append(peak['amplitude'])
        
        if not all_peaks:
            return None
        
        # Find the most common peaks (characteristic peaks)
        peak_array = np.array(all_peaks)
        intensity_array = np.array(peak_intensities)
        
        # Group peaks by position (within 10 cm-1 tolerance)
        peak_groups = []
        tolerance = 10
        
        for peak_pos in peak_array:
            # Find if this peak belongs to an existing group
            group_found = False
            for group in peak_groups:
                if abs(peak_pos - group['center']) <= tolerance:
                    group['positions'].append(peak_pos)
                    group['intensities'].append(intensity_array[len(group['positions'])-1])
                    group_found = True
                    break
            
            if not group_found:
                # Create new group
                idx = np.where(peak_array == peak_pos)[0][0]
                peak_groups.append({
                    'center': peak_pos,
                    'positions': [peak_pos],
                    'intensities': [intensity_array[idx]]
                })
        
        # Sort groups by average intensity (strongest peaks first)
        for group in peak_groups:
            group['avg_intensity'] = np.mean(group['intensities'])
            group['avg_position'] = np.mean(group['positions'])
        
        peak_groups.sort(key=lambda x: x['avg_intensity'], reverse=True)
        
        # Extract top 3 peaks as characteristic peaks
        main_peak = peak_groups[0]['avg_position'] if len(peak_groups) > 0 else 1000
        secondary_peak = peak_groups[1]['avg_position'] if len(peak_groups) > 1 else 500
        tertiary_peak = peak_groups[2]['avg_position'] if len(peak_groups) > 2 else 1500
        
        # Calculate reference intensity (median of main peak intensities)
        main_peak_intensities = peak_groups[0]['intensities'] if len(peak_groups) > 0 else [800]
        reference_intensity = int(np.median(main_peak_intensities))
        
        # Determine spectral range for reference regions
        min_wavenumber = min(all_peaks) if all_peaks else 200
        max_wavenumber = max(all_peaks) if all_peaks else 1800
        
        return {
            'main_peak': main_peak,
            'secondary_peak': secondary_peak,
            'tertiary_peak': tertiary_peak,
            'reference_intensity': reference_intensity,
            'spectral_range': (min_wavenumber, max_wavenumber),
            'peak_groups': peak_groups,
            'total_peaks_analyzed': len(all_peaks),
            'total_spectra': len(successful_results)
        }
    
    def _populate_dialog_from_batch(self, dialog, analysis_results):
        """Populate the custom material dialog with batch analysis results."""
        # Set peak positions
        dialog.main_peak_spin.setValue(analysis_results['main_peak'])
        dialog.secondary_peak_spin.setValue(analysis_results['secondary_peak'])
        dialog.tertiary_peak_spin.setValue(analysis_results['tertiary_peak'])
        
        # Set reference intensity
        dialog.reference_intensity.setValue(analysis_results['reference_intensity'])
        
        # Set reasonable reference regions based on spectral range
        min_wn, max_wn = analysis_results['spectral_range']
        
        # Baseline region (below main peaks)
        baseline_start = max(100, min_wn - 50)
        baseline_end = min(analysis_results['main_peak'] - 50, baseline_start + 200)
        dialog.baseline_start.setValue(baseline_start)
        dialog.baseline_end.setValue(baseline_end)
        
        # Fingerprint region (around main peaks)
        fingerprint_start = max(analysis_results['main_peak'] - 200, baseline_end + 50)
        fingerprint_end = min(analysis_results['main_peak'] + 200, max_wn)
        dialog.fingerprint_start.setValue(fingerprint_start)
        dialog.fingerprint_end.setValue(fingerprint_end)
        
        # Set default material name based on analysis
        suggested_name = f"Custom_Material_{len(analysis_results['peak_groups'])}peaks"
        dialog.material_name.setText(suggested_name)
    
    def _create_batch_analysis_preview(self, analysis_results):
        """Create preview text showing batch analysis results."""
        preview = f"""Batch Analysis Results:
        
📊 PEAK ANALYSIS:
• Total Spectra Analyzed: {analysis_results['total_spectra']}
• Total Peaks Found: {analysis_results['total_peaks_analyzed']}
• Peak Groups Identified: {len(analysis_results['peak_groups'])}

🎯 CHARACTERISTIC PEAKS:
• Main Peak: {analysis_results['main_peak']:.1f} cm⁻¹
• Secondary Peak: {analysis_results['secondary_peak']:.1f} cm⁻¹
• Tertiary Peak: {analysis_results['tertiary_peak']:.1f} cm⁻¹

📈 INTENSITY ANALYSIS:
• Reference Intensity: {analysis_results['reference_intensity']}
• Spectral Range: {analysis_results['spectral_range'][0]:.0f} - {analysis_results['spectral_range'][1]:.0f} cm⁻¹

These parameters will be used to create a custom material configuration for density analysis.
"""
        return preview 