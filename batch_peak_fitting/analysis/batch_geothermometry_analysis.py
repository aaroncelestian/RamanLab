#!/usr/bin/env python3
"""
Comprehensive Geothermometry Analysis Module for RamanLab Batch Peak Fitting

This module provides complete geothermometry analysis functionality for batch processing
of Raman spectra, with advanced parameter extraction, multiple methods, and comprehensive
results reporting.

Features:
- Multiple geothermometry methods (Beyssac 2002, Aoya 2010, etc.)
- Automatic D and G band identification
- Batch processing with progress tracking
- Detailed results display and export
- Error handling and validation
- Integration with RamanLab's batch peak fitting system
"""

import numpy as np
import os
import sys
import csv
from pathlib import Path
from enum import Enum

# Qt6 imports
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QMessageBox, QProgressDialog, QApplication, QInputDialog, QGroupBox,
    QFileDialog, QTextEdit, QComboBox, QFormLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Check if geothermometry module is available
GEOTHERMOMETRY_AVAILABLE = False
GeothermometerMethod = None
RamanGeothermometry = None

try:
    from raman_geothermometry import RamanGeothermometry, GeothermometerMethod
    GEOTHERMOMETRY_AVAILABLE = True
    print("✓ Geothermometry analysis module loaded successfully")
except ImportError as e:
    print(f"⚠️ Geothermometry analysis not available: {e}")
    # Create dummy enum for when module is not available
    class GeothermometerMethod(Enum):
        BEYSSAC_2002 = "Beyssac et al. (2002)"
        AOYA_2010_514 = "Aoya et al. (2010) 514nm"
        AOYA_2010_532 = "Aoya et al. (2010) 532nm"
        RAHL_2005 = "Rahl et al. (2005)"
        KOUKETSU_2014_D1 = "Kouketsu et al. (2014) D1"
        KOUKETSU_2014_D2 = "Kouketsu et al. (2014) D2"
        RANTITSCH_2004 = "Rantitsch et al. (2004)"


class BatchGeothermometryAnalyzer:
    """Comprehensive geothermometry analysis handler for batch peak fitting operations."""
    
    def __init__(self, main_window):
        """Initialize the geothermometry analyzer with reference to main window."""
        self.main_window = main_window
        self.geothermometry_results = []
        
        # Parameter extraction configuration
        self.required_params = {
            GeothermometerMethod.BEYSSAC_2002: ['R2'],
            GeothermometerMethod.AOYA_2010_514: ['R2'], 
            GeothermometerMethod.AOYA_2010_532: ['R2'],
            GeothermometerMethod.RAHL_2005: ['R1', 'R2'],
            GeothermometerMethod.KOUKETSU_2014_D1: ['D1_FWHM'],
            GeothermometerMethod.KOUKETSU_2014_D2: ['D2_FWHM'],
            GeothermometerMethod.RANTITSCH_2004: ['R2']
        }
    
    def is_available(self):
        """Check if geothermometry analysis is available."""
        return GEOTHERMOMETRY_AVAILABLE
    
    def launch_geothermometry_analysis(self):
        """Launch a detailed geothermometry analysis configuration dialog."""
        if not GEOTHERMOMETRY_AVAILABLE:
            QMessageBox.warning(self.main_window, "Module Not Available", 
                              "Geothermometry analysis module is not available.\n"
                              "Please check if raman_geothermometry.py is present.")
            return
        
        if not hasattr(self.main_window, 'batch_results') or not self.main_window.batch_results:
            QMessageBox.warning(self.main_window, "No Results", 
                              "No batch fitting results available.\nRun batch processing first.")
            return
        
        try:
            dialog = GeothermometryConfigurationDialog(self.main_window)
            
            if dialog.exec() == QDialog.Accepted:
                method = dialog.get_selected_method()
                output_option = dialog.get_output_option()
                
                # Run analysis with selected configuration
                self._run_comprehensive_analysis(method, output_option)
                
        except Exception as e:
            QMessageBox.critical(self.main_window, "Launch Error", 
                               f"Failed to launch geothermometry analysis:\n{str(e)}")
    
    def quick_geothermometry_analysis(self):
        """Perform quick geothermometry analysis on the current spectrum."""
        if not GEOTHERMOMETRY_AVAILABLE:
            QMessageBox.warning(self.main_window, "Not Available", 
                              "Geothermometry analysis module is not available.")
            return
            
        if len(self.main_window.wavenumbers) == 0 or len(self.main_window.intensities) == 0:
            QMessageBox.warning(self.main_window, "No Data", 
                              "No spectrum data loaded for analysis.")
            return
        
        if not self.main_window.fit_params or len(self.main_window.fit_params) == 0:
            QMessageBox.warning(self.main_window, "No Peak Fitting", 
                              "Please perform peak fitting first to enable geothermometry analysis.")
            return
        
        try:
            # Ask user for method
            geothermo_calc = RamanGeothermometry()
            methods = geothermo_calc.get_all_methods()
            method_name, ok = QInputDialog.getItem(
                self.main_window, "Select Geothermometry Method", 
                "Choose the geothermometry method:",
                methods, 0, False
            )
            
            if not ok:
                return
            
            # Find the corresponding GeothermometerMethod enum
            method_enum = None
            for method in GeothermometerMethod:
                if method.value == method_name:
                    method_enum = method
                    break
            
            if method_enum is None:
                QMessageBox.warning(self.main_window, "Invalid Method", f"Could not find method: {method_name}")
                return
            
            # Create a mock fitting result from current data
            fitting_result = {
                'fit_params': self.main_window.fit_params,
                'wavenumbers': self.main_window.wavenumbers
            }
            
            # Extract parameters
            params = self._extract_geothermometry_parameters(fitting_result, method_enum)
            
            # Calculate temperature
            temperature, status = geothermo_calc.calculate_temperature(method_enum, **params)
            
            if temperature is None:
                QMessageBox.warning(self.main_window, "Calculation Failed", 
                                  f"Temperature calculation failed: {status}")
                return
            
            # Display results
            result_text = f"""
Geothermometry Analysis Results:

Method: {method_name}
Temperature: {temperature:.1f}°C
Status: {status}

Parameters Used:
"""
            for param, value in params.items():
                result_text += f"• {param}: {value:.4f}\n"
            
            # Add method information
            method_info = geothermo_calc.get_method_info(method_enum)
            result_text += f"""
Method Details:
• Temperature Range: {method_info.temp_range}
• Typical Error: {method_info.error}
• Description: {method_info.description}
"""
            
            QMessageBox.information(self.main_window, "Geothermometry Results", result_text)
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Analysis Error", 
                               f"Failed to perform geothermometry analysis:\n{str(e)}")
    
    def run_batch_geothermometry_analysis(self, auto_mode=False):
        """Run geothermometry analysis on all loaded spectra.
        
        Args:
            auto_mode (bool): If True, runs automatically after batch processing without showing dialogs
        """
        if not GEOTHERMOMETRY_AVAILABLE:
            if not auto_mode:
                QMessageBox.warning(self.main_window, "Not Available", "Geothermometry analysis module is not available.")
            return
        
        if len(self.main_window.spectra_files) == 0:
            if not auto_mode:
                QMessageBox.warning(self.main_window, "No Data", "No spectra loaded for analysis.")
            return
        
        try:
            # Use sensible defaults for batch processing (detailed config in specialized tab)
            method_enum = GeothermometerMethod.BEYSSAC_2002  # Most commonly used method
            output_option = "Temperature + Parameters"  # Show results with parameters
            
            self._run_comprehensive_analysis(method_enum, output_option, auto_mode)
            
        except Exception as e:
            error_msg = f"Failed to run geothermometry analysis:\n{str(e)}"
            if not auto_mode:
                QMessageBox.critical(self.main_window, "Analysis Error", error_msg)
            else:
                self.main_window.batch_status_text.append(f"❌ Geothermometry analysis error: {str(e)}")
    
    def _run_comprehensive_analysis(self, method_enum, output_option, auto_mode=False):
        """Perform comprehensive geothermometry analysis with full error handling."""
        successful_results = [r for r in self.main_window.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            QMessageBox.warning(self.main_window, "No Data", "No successful fitting results available for analysis.")
            return
        
        geothermo_calc = RamanGeothermometry()
        method_name = method_enum.value
        
        # Progress dialog
        progress = QProgressDialog("Running geothermometry analysis...", "Cancel", 0, len(successful_results), self.main_window)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # Clear previous results
        self.geothermometry_results.clear()
        
        if hasattr(self.main_window, 'batch_status_text'):
            self.main_window.batch_status_text.append(f"\nStarting batch geothermometry analysis ({method_name})...")
        
        success_count = 0
        
        for i, result in enumerate(successful_results):
            if progress.wasCanceled():
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Analyzing: {os.path.basename(result['file'])}")
            QApplication.processEvents()
            
            try:
                # Extract required parameters from peak fitting results
                params = self._extract_geothermometry_parameters(result, method_enum)
                
                # Calculate temperature
                temperature, status = geothermo_calc.calculate_temperature(method_enum, **params)
                
                if temperature is None:
                    raise ValueError(f"Temperature calculation failed: {status}")
                
                # Store results
                result_data = {
                    'file': result['file'],
                    'method': method_name,
                    'temperature': temperature,
                    'status': status,
                    'parameters': params,
                    'output_option': output_option,
                    'success': True
                }
                
                self.geothermometry_results.append(result_data)
                
                if hasattr(self.main_window, 'batch_status_text'):
                    self.main_window.batch_status_text.append(f"SUCCESS: {os.path.basename(result['file'])} - T = {temperature:.1f}°C ({status})")
                success_count += 1
                
            except Exception as e:
                error_msg = str(e)
                self.geothermometry_results.append({
                    'file': result['file'],
                    'method': method_name,
                    'success': False,
                    'failed': True,
                    'error': error_msg
                })
                if hasattr(self.main_window, 'batch_status_text'):
                    self.main_window.batch_status_text.append(f"FAILED: {os.path.basename(result['file'])} - {error_msg}")
        
        progress.setValue(len(successful_results))
        progress.close()
        
        if hasattr(self.main_window, 'batch_status_text'):
            self.main_window.batch_status_text.append(f"\nGeothermometry analysis complete: {success_count}/{len(successful_results)} successful")
        
        # Store results in main window for compatibility
        self.main_window.geothermometry_results = self.geothermometry_results
        
        # Update trends plot with latest data
        if hasattr(self.main_window, 'update_trends_plot'):
            self.main_window.update_trends_plot()
            
        if not auto_mode:
            # Show results based on output option
            if output_option == "Export to CSV":
                self._export_geothermometry_results()
            else:
                self._display_geothermometry_results(output_option)
            
            QMessageBox.information(self.main_window, "Geothermometry Complete", 
                                  f"Processed {len(successful_results)} spectra.\n"
                                  f"Successful: {success_count}\n"
                                  f"Failed: {len(successful_results) - success_count}")
        else:
            # Auto mode: just report success to status text and update summary display
            if hasattr(self.main_window, 'batch_status_text'):
                self.main_window.batch_status_text.append(f"✅ Geothermometry analysis completed automatically!")
                if success_count > 0:
                    successful_temps = [r['temperature'] for r in self.geothermometry_results if r.get('success', False)]
                    if successful_temps:
                        avg_temp = sum(successful_temps) / len(successful_temps)
                        min_temp = min(successful_temps)
                        max_temp = max(successful_temps)
                        self.main_window.batch_status_text.append(f"   📊 Temperature range: {min_temp:.1f}°C - {max_temp:.1f}°C (avg: {avg_temp:.1f}°C)")
    
    def _extract_geothermometry_parameters(self, fitting_result, method_enum):
        """Extract required parameters for geothermometry from peak fitting results."""
        needed_params = self.required_params.get(method_enum, [])
        
        # Extract peak information from fitting results
        fit_params = fitting_result.get('fit_params', [])
        wavenumbers = fitting_result.get('wavenumbers', [])
        
        if len(fit_params) == 0:
            raise ValueError("No peak fitting parameters available")
        
        # Determine parameters per peak based on model
        model = fitting_result.get('model', 'Gaussian')
        if model == 'Gaussian' or model == 'Lorentzian':
            params_per_peak = 3  # amplitude, center, width
        elif model == 'Pseudo-Voigt' or model == 'Voigt':
            params_per_peak = 4  # amplitude, center, width, eta/gamma  
        elif model == 'Asymmetric Voigt':
            params_per_peak = 5  # amplitude, center, width, gamma, alpha
        else:
            params_per_peak = 3  # default fallback
        
        # Find D and G bands (approximately 1350 and 1580 cm-1)
        n_peaks = len(fit_params) // params_per_peak
        peaks_info = []
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            if start_idx + 2 < len(fit_params):
                amp, cen, wid = fit_params[start_idx:start_idx+3]
                peaks_info.append({'position': cen, 'amplitude': amp, 'width': wid})
        
        if len(peaks_info) < 2:
            raise ValueError("At least 2 peaks required for geothermometry analysis")
        
        # Sort peaks by position
        peaks_info.sort(key=lambda x: x['position'])
        
        # Find D band (around 1350 cm-1) and G band (around 1580 cm-1)
        d_band = None
        g_band = None
        
        for peak in peaks_info:
            pos = peak['position']
            if 1300 <= pos <= 1400 and d_band is None:
                d_band = peak
            elif 1500 <= pos <= 1650 and g_band is None:
                g_band = peak
        
        if d_band is None or g_band is None:
            raise ValueError("Could not identify D and G bands in spectrum")
        
        # Calculate parameters based on method requirements
        params = {}
        
        if 'R2' in needed_params:
            # R2 = Area_D / Area_G (approximated as amplitude ratio for Gaussian peaks)
            params['R2'] = d_band['amplitude'] / g_band['amplitude']
        
        if 'R1' in needed_params:
            # R1 = Height_D / Height_G  
            params['R1'] = d_band['amplitude'] / g_band['amplitude']
        
        if 'D1_FWHM' in needed_params:
            # D1 FWHM (assuming D1 is the D band)
            params['D1_FWHM'] = d_band['width'] * 2 * np.sqrt(2 * np.log(2))  # Convert from Gaussian sigma to FWHM
        
        if 'D2_FWHM' in needed_params:
            # D2 FWHM - this would require more sophisticated peak deconvolution
            # For now, use an estimate
            params['D2_FWHM'] = d_band['width'] * 2 * np.sqrt(2 * np.log(2)) * 0.8
        
        return params
    
    def _display_geothermometry_results(self, output_option):
        """Display comprehensive geothermometry results in a dialog."""
        if not self.geothermometry_results:
            return
        
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Geothermometry Analysis Results")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Results text
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setFont(QFont("Consolas", 10))
        
        results_text = "GEOTHERMOMETRY ANALYSIS RESULTS\n"
        results_text += "=" * 50 + "\n\n"
        
        successful_results = [r for r in self.geothermometry_results if r.get('success', False)]
        failed_results = [r for r in self.geothermometry_results if not r.get('success', False)]
        
        # Summary statistics
        if successful_results:
            temperatures = [r['temperature'] for r in successful_results]
            results_text += f"📊 SUMMARY STATISTICS:\n"
            results_text += f"• Total Analyzed: {len(self.geothermometry_results)}\n"
            results_text += f"• Successful: {len(successful_results)}\n"
            results_text += f"• Failed: {len(failed_results)}\n"
            results_text += f"• Temperature Range: {min(temperatures):.1f} - {max(temperatures):.1f}°C\n"
            results_text += f"• Average Temperature: {np.mean(temperatures):.1f}°C\n"
            results_text += f"• Standard Deviation: {np.std(temperatures):.1f}°C\n\n"
        
        # Individual results
        results_text += "🌡️ DETAILED RESULTS:\n"
        results_text += "-" * 30 + "\n\n"
        
        for result in successful_results:
            filename = os.path.basename(result['file'])
            temperature = result['temperature']
            status = result['status']
            method = result['method']
            
            results_text += f"File: {filename}\n"
            results_text += f"Method: {method}\n"
            results_text += f"Temperature: {temperature:.1f}°C\n"
            results_text += f"Status: {status}\n"
            
            if output_option in ["Temperature + Parameters", "Full Analysis Report"]:
                params = result.get('parameters', {})
                results_text += "Parameters:\n"
                for param, value in params.items():
                    results_text += f"  {param}: {value:.4f}\n"
            
            if output_option == "Full Analysis Report":
                # Add method information
                if GEOTHERMOMETRY_AVAILABLE:
                    geothermo_calc = RamanGeothermometry()
                    for method_enum in GeothermometerMethod:
                        if method_enum.value == method:
                            method_info = geothermo_calc.get_method_info(method_enum)
                            results_text += f"Method Details:\n"
                            results_text += f"  Range: {method_info.temp_range}\n"
                            results_text += f"  Error: {method_info.error}\n"
                            results_text += f"  Description: {method_info.description}\n"
                            break
            
            results_text += "\n" + "-" * 40 + "\n\n"
        
        if failed_results:
            results_text += "\n❌ FAILED ANALYSES:\n"
            results_text += "=" * 20 + "\n"
            for result in failed_results:
                filename = os.path.basename(result['file'])
                error = result.get('error', 'Unknown error')
                results_text += f"File: {filename}\n"
                results_text += f"Error: {error}\n\n"
        
        text_widget.setText(results_text)
        layout.addWidget(text_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(lambda: self._export_geothermometry_results())
        button_layout.addWidget(export_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _export_geothermometry_results(self):
        """Export comprehensive geothermometry results to CSV."""
        if not self.geothermometry_results:
            QMessageBox.warning(self.main_window, "No Data", "No geothermometry results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Geothermometry Results", 
            "geothermometry_results.csv",
            "CSV files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Determine all possible columns
                fieldnames = ['Filename', 'Method', 'Temperature_C', 'Status', 'Success', 'Failed', 'Error']
                
                # Add parameter columns
                param_columns = set()
                for result in self.geothermometry_results:
                    if 'parameters' in result:
                        param_columns.update(result['parameters'].keys())
                
                fieldnames.extend(sorted(param_columns))
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.geothermometry_results:
                    row = {
                        'Filename': os.path.basename(result['file']),
                        'Method': result['method'],
                        'Success': result.get('success', False),
                        'Failed': result.get('failed', False),
                        'Error': result.get('error', '')
                    }
                    
                    if result.get('success', False):
                        row['Temperature_C'] = result['temperature']
                        row['Status'] = result['status']
                        
                        # Add parameters
                        params = result.get('parameters', {})
                        for param in param_columns:
                            row[param] = params.get(param, '')
                    
                    writer.writerow(row)
            
            QMessageBox.information(self.main_window, "Export Successful", 
                                  f"Geothermometry results exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Export Error", 
                               f"Failed to export geothermometry results:\n{str(e)}")


class GeothermometryConfigurationDialog(QDialog):
    """Advanced configuration dialog for geothermometry analysis."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Geothermometry Analysis Configuration")
        self.setMinimumSize(500, 400)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Configure Geothermometry Analysis")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header)
        
        # Configuration form
        form_layout = QFormLayout()
        
        # Method selection
        self.method_combo = QComboBox()
        if GEOTHERMOMETRY_AVAILABLE:
            for method in GeothermometerMethod:
                self.method_combo.addItem(method.value)
        else:
            self.method_combo.addItem("Geothermometry module not available")
            self.method_combo.setEnabled(False)
        form_layout.addRow("Analysis Method:", self.method_combo)
        
        # Output options
        self.output_combo = QComboBox()
        self.output_combo.addItems([
            "Temperature Only", 
            "Temperature + Parameters", 
            "Full Analysis Report",
            "Export to CSV"
        ])
        self.output_combo.setCurrentText("Temperature + Parameters")
        form_layout.addRow("Output Format:", self.output_combo)
        
        layout.addLayout(form_layout)
        
        # Information section
        info_group = QGroupBox("Method Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.update_method_info()
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # Connect method change to info update
        self.method_combo.currentTextChanged.connect(self.update_method_info)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self.accept)
        run_btn.setDefault(True)
        button_layout.addWidget(run_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def update_method_info(self):
        """Update the method information display."""
        if not GEOTHERMOMETRY_AVAILABLE:
            self.info_text.setText("Geothermometry module is not available.")
            return
        
        method_name = self.method_combo.currentText()
        
        # Find method enum
        method_enum = None
        for method in GeothermometerMethod:
            if method.value == method_name:
                method_enum = method
                break
        
        if method_enum:
            try:
                geothermo_calc = RamanGeothermometry()
                method_info = geothermo_calc.get_method_info(method_enum)
                
                info_text = f"""Method: {method_name}

Description: {method_info.description}

Temperature Range: {method_info.temp_range}
Typical Error: {method_info.error}
Best For: {method_info.best_for}

Limitations: {method_info.limitations}

Required Parameters: {', '.join(BatchGeothermometryAnalyzer(None).required_params.get(method_enum, []))}
"""
                self.info_text.setText(info_text)
            except Exception as e:
                self.info_text.setText(f"Error retrieving method information: {str(e)}")
    
    def get_selected_method(self):
        """Get the selected geothermometry method."""
        method_text = self.method_combo.currentText()
        for method in GeothermometerMethod:
            if method.value == method_text:
                return method
        return GeothermometerMethod.BEYSSAC_2002  # Default fallback
    
    def get_output_option(self):
        """Get the selected output option."""
        return self.output_combo.currentText()


print("✅ Comprehensive Geothermometry Analysis Module loaded successfully") 