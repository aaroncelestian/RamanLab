#!/usr/bin/env python3
"""
Density Analysis Module for Batch Peak Fitting
Provides density analysis functionality for RamanLab batch processing.
"""

import numpy as np
import os
import sys
from pathlib import Path

# Qt6 imports
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QMessageBox, QProgressDialog, QApplication, QInputDialog, QGroupBox,
    QFileDialog, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

# Matplotlib imports
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Import the density analysis module
try:
    from advanced_analysis.raman_density_analysis import RamanDensityAnalyzer
    DENSITY_AVAILABLE = True
except ImportError:
    DENSITY_AVAILABLE = False


class BatchDensityAnalyzer:
    """Handles density analysis for batch peak fitting operations."""
    
    def __init__(self, main_window):
        """Initialize the density analyzer with reference to main window."""
        self.main_window = main_window
        self.density_results = []
    
    def is_available(self):
        """Check if density analysis is available."""
        return DENSITY_AVAILABLE
    
    def launch_density_analysis(self):
        """Launch the density analysis tool as a separate process."""
        try:
            # Get the path to the density analysis GUI launcher
            script_path = Path(__file__).parent.parent.parent / "Density" / "density_gui_launcher.py"
            
            if not script_path.exists():
                # Fallback to the main analysis script
                script_path = Path(__file__).parent.parent.parent / "Density" / "raman_density_analysis.py"
                if not script_path.exists():
                    QMessageBox.warning(self.main_window, "File Not Found", 
                                      f"Density analysis scripts not found in Density/ folder")
                    return
            
            # Launch as separate Python process
            import subprocess
            subprocess.Popen([sys.executable, str(script_path)], 
                           cwd=Path(__file__).parent.parent.parent)
            
            QMessageBox.information(self.main_window, "Launched", 
                                  "Density Analysis tool has been launched in a separate window.")
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Launch Error", 
                               f"Failed to launch density analysis tool:\n{str(e)}")
    
    def quick_density_analysis(self):
        """Perform quick density analysis on the current spectrum."""
        if not DENSITY_AVAILABLE:
            QMessageBox.warning(self.main_window, "Not Available", 
                              "Density analysis module is not available.")
            return
            
        if len(self.main_window.wavenumbers) == 0 or len(self.main_window.intensities) == 0:
            QMessageBox.warning(self.main_window, "No Data", 
                              "No spectrum data loaded for analysis.")
            return
        
        try:
            from advanced_analysis.raman_density_analysis import MaterialConfigs
            
            # Ask user for material type
            material_types = MaterialConfigs.get_available_materials()
            material_type, ok = QInputDialog.getItem(
                self.main_window, "Select Material Type", 
                "Choose the material type for analysis:",
                material_types, 0, False
            )
            
            if not ok:
                return
            
            # Create analyzer instance with selected material
            analyzer = RamanDensityAnalyzer(material_type)
            
            # Use current spectrum data
            wavenumbers = self.main_window.wavenumbers
            intensities = self.main_window.intensities
            
            # Preprocess spectrum
            wn, corrected_int = analyzer.preprocess_spectrum(wavenumbers, intensities)
            
            # Calculate CDI and density
            cdi, metrics = analyzer.calculate_crystalline_density_index(wn, corrected_int)
            apparent_density = analyzer.calculate_apparent_density(cdi)
            
            # Use appropriate density calculation
            if material_type == 'Kidney Stones (COM)':
                specialized_density = analyzer.calculate_biofilm_density(cdi, 'mixed')
                density_label = "Biofilm-Calibrated Density"
            else:
                specialized_density = analyzer.calculate_density_by_type(cdi, 'mixed')
                density_label = "Specialized Density"
            
            # Get classification
            thresholds = analyzer.classification_thresholds
            if cdi < thresholds['low']:
                classification = 'Low crystallinity'
            elif cdi < thresholds['medium']:
                classification = 'Mixed regions'
            elif cdi < thresholds['high']:
                classification = 'Mixed crystalline'
            else:
                classification = 'Pure crystalline'
            
            # Display results
            result_text = f"""
Density Analysis Results ({material_type}):

Crystalline Density Index (CDI): {cdi:.4f}
Standard Apparent Density: {apparent_density:.3f} g/cm³
{density_label}: {specialized_density:.3f} g/cm³

Metrics:
• Main Peak Height: {metrics['main_peak_height']:.1f}
• Main Peak Position: {metrics['main_peak_position']} cm⁻¹
• Baseline Intensity: {metrics['baseline_intensity']:.1f}
• Peak Width (FWHM): {metrics['peak_width']:.1f} cm⁻¹
• Spectral Contrast: {metrics['spectral_contrast']:.4f}

Classification: {classification} region

Material-Specific Guidelines:
• CDI < {thresholds['low']:.2f}: Low crystallinity regions
• CDI {thresholds['low']:.2f}-{thresholds['medium']:.2f}: Mixed regions
• CDI {thresholds['medium']:.2f}-{thresholds['high']:.2f}: Mixed crystalline
• CDI > {thresholds['high']:.2f}: Pure crystalline
            """
            
            QMessageBox.information(self.main_window, "Density Analysis Results", result_text)
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Analysis Error", 
                               f"Failed to perform density analysis:\n{str(e)}")
    
    def run_batch_density_analysis(self):
        """Run density analysis on all loaded spectra."""
        if not DENSITY_AVAILABLE:
            QMessageBox.warning(self.main_window, "Not Available", "Density analysis module is not available.")
            return
        
        if len(self.main_window.spectra_files) == 0:
            QMessageBox.warning(self.main_window, "No Data", "No spectra loaded for analysis.")
            return
        
        try:
            from advanced_analysis.raman_density_analysis import RamanDensityAnalyzer
            
            # Use default analysis parameters for streamlined batch processing
            material_type = 'Kidney Stones (COM)'
            density_type = 'mixed'
            
            # Create analyzer
            analyzer = RamanDensityAnalyzer(material_type)
            
            # Progress dialog
            progress = QProgressDialog("Running density analysis...", "Cancel", 0, len(self.main_window.spectra_files), self.main_window)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Clear previous results
            self.density_results.clear()
            
            self.main_window.batch_status_text.append(f"\nStarting batch density analysis ({material_type})...")
            
            success_count = 0
            
            for i, file_path in enumerate(self.main_window.spectra_files):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(f"Analyzing: {os.path.basename(file_path)}")
                QApplication.processEvents()
                
                try:
                    # Load spectrum
                    wavenumbers, intensities = self.main_window.load_spectrum_robust(file_path)
                    
                    if wavenumbers is None or intensities is None:
                        raise ValueError("Failed to load spectrum data")
                    
                    # Preprocess spectrum with validation
                    try:
                        wn, corrected_int = analyzer.preprocess_spectrum(wavenumbers, intensities)
                        
                        # Validate preprocessed data
                        if len(wn) == 0 or len(corrected_int) == 0:
                            raise ValueError("Empty data after preprocessing")
                        if np.all(np.isnan(corrected_int)) or np.all(np.isinf(corrected_int)):
                            raise ValueError("Invalid intensity data after preprocessing")
                            
                    except Exception as e:
                        raise ValueError(f"Preprocessing failed: {str(e)}")
                    
                    # Calculate density metrics with validation
                    try:
                        cdi, metrics = analyzer.calculate_crystalline_density_index(wn, corrected_int)
                        
                        # Validate CDI calculation
                        if np.isnan(cdi) or np.isinf(cdi):
                            raise ValueError("CDI calculation returned invalid value")
                        if cdi < 0:
                            cdi = 0.0  # Ensure non-negative CDI
                            
                        apparent_density = analyzer.calculate_apparent_density(cdi)
                        
                        # Validate density calculation
                        if np.isnan(apparent_density) or np.isinf(apparent_density):
                            raise ValueError("Density calculation returned invalid value")
                            
                    except Exception as e:
                        raise ValueError(f"Density analysis failed: {str(e)}")
                    
                    # Use appropriate density calculation
                    if material_type == 'Kidney Stones (COM)':
                        specialized_density = analyzer.calculate_biofilm_density(cdi, density_type)
                    else:
                        specialized_density = analyzer.calculate_density_by_type(cdi, density_type)
                    
                    # Get classification
                    thresholds = analyzer.classification_thresholds
                    if cdi < thresholds['low']:
                        classification = 'Low crystallinity'
                    elif cdi < thresholds['medium']:
                        classification = 'Mixed regions'
                    elif cdi < thresholds['high']:
                        classification = 'Mixed crystalline'
                    else:
                        classification = 'Pure crystalline'
                    
                    # Store results
                    density_result = {
                        'file': file_path,
                        'material_type': material_type,
                        'density_type': density_type,
                        'cdi': cdi,
                        'apparent_density': apparent_density,
                        'specialized_density': specialized_density,
                        'classification': classification,
                        'metrics': metrics,
                        'wavenumbers': wn,
                        'corrected_intensity': corrected_int,
                        'success': True
                    }
                    
                    self.density_results.append(density_result)
                    success_count += 1
                    
                    # Add debug information for density calculation
                    debug_info = f"CDI={cdi:.4f}, ρ={specialized_density:.3f} g/cm³"
                    if 'metrics' in locals():
                        debug_info += f", peak_height={metrics.get('main_peak_height', 'N/A'):.1f}"
                        debug_info += f", baseline={metrics.get('baseline_intensity', 'N/A'):.1f}"
                    
                    self.main_window.batch_status_text.append(f"DENSITY SUCCESS: {os.path.basename(file_path)} - {debug_info}")
                    
                except Exception as e:
                    # Store failed result
                    density_result = {
                        'file': file_path,
                        'material_type': material_type,
                        'density_type': density_type,
                        'success': False,
                        'error': str(e)
                    }
                    
                    self.density_results.append(density_result)
                    self.main_window.batch_status_text.append(f"DENSITY FAILED: {os.path.basename(file_path)} - {str(e)}")
            
            progress.setValue(len(self.main_window.spectra_files))
            progress.close()
            
            # Update plots if we have density results
            if success_count > 0:
                # Store results in main window for compatibility
                self.main_window.density_results = self.density_results
                if hasattr(self.main_window, 'update_trends_plot'):
                    self.main_window.update_trends_plot()
                if hasattr(self.main_window, 'update_heatmap_plot'):
                    self.main_window.update_heatmap_plot()
                
                # Generate comprehensive density analysis visualization
                self.show_comprehensive_density_plots()
            
            self.main_window.batch_status_text.append(f"\nDensity analysis complete: {success_count}/{len(self.main_window.spectra_files)} successful")
            
            QMessageBox.information(self.main_window, "Density Analysis Complete", 
                                  f"Analyzed {len(self.main_window.spectra_files)} spectra for density.\n"
                                  f"Successful: {success_count}\n"
                                  f"Failed: {len(self.main_window.spectra_files) - success_count}\n\n"
                                  f"Material: {material_type}\n"
                                  f"Density Type: {density_type}")
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Analysis Error", f"Failed to run batch density analysis:\n{str(e)}")
    
    def show_comprehensive_density_plots(self):
        """Generate comprehensive density analysis plots."""
        if not self.density_results:
            return
            
        try:
            from advanced_analysis.raman_density_analysis import RamanDensityAnalyzer
            
            # Get successful density results with valid data
            successful_results = []
            for r in self.density_results:
                if (r.get('success', False) and 
                    not np.isnan(r.get('cdi', np.nan)) and 
                    not np.isnan(r.get('specialized_density', np.nan))):
                    successful_results.append(r)
            
            if not successful_results:
                QMessageBox.warning(self.main_window, "No Valid Data", 
                                  "No density analysis results with valid data to plot.")
                return
            
            # Use the first successful result to get the material type and analyzer
            first_result = successful_results[0]
            material_type = first_result['material_type']
            analyzer = RamanDensityAnalyzer(material_type)
            
            # Prepare data for comprehensive analysis
            positions = list(range(len(successful_results)))
            cdi_profile = [r['cdi'] for r in successful_results]
            density_profile = [r['specialized_density'] for r in successful_results]
            apparent_density_profile = [r['apparent_density'] for r in successful_results]
            classifications = [r['classification'] for r in successful_results]
            
            # Calculate statistics
            mean_density = np.mean(density_profile)
            std_density = np.std(density_profile)
            mean_cdi = np.mean(cdi_profile)
            std_cdi = np.std(cdi_profile)
            
            # Create density profile data structure for plotting
            density_analysis_data = {
                'positions': positions,
                'cdi_profile': cdi_profile,
                'density_profile': density_profile,
                'apparent_density_profile': apparent_density_profile,
                'layer_classification': classifications,
                'statistics': {
                    'mean_density': mean_density,
                    'std_density': std_density,
                    'mean_cdi': mean_cdi,
                    'std_cdi': std_cdi,
                    'n_spectra': len(successful_results)
                },
                'file_names': [os.path.basename(r['file']) for r in successful_results]
            }
            
            # Create a comprehensive density visualization
            self.create_density_visualization_window(density_analysis_data, analyzer, material_type)
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Visualization Error", 
                               f"Failed to create density visualization:\n{str(e)}")
    
    def create_density_visualization_window(self, density_data, analyzer, material_type):
        """Create a separate window for comprehensive density analysis visualization."""
        # Create a new window
        density_window = QDialog(self.main_window)
        density_window.setWindowTitle(f"Comprehensive Density Analysis - {material_type}")
        density_window.setMinimumSize(1200, 900)
        density_window.resize(1400, 1000)
        
        layout = QVBoxLayout(density_window)
        
        # Add summary information at the top
        info_group = QGroupBox("Analysis Summary")
        info_layout = QVBoxLayout(info_group)
        
        stats = density_data['statistics']
        summary_text = f"""
Material Type: {material_type}
Number of Spectra: {stats['n_spectra']}
Mean CDI: {stats['mean_cdi']:.4f} ± {stats['std_cdi']:.4f}
Mean Density: {stats['mean_density']:.3f} ± {stats['std_density']:.3f} g/cm³
"""
        info_label = QLabel(summary_text)
        info_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 10px;")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Create matplotlib figure for comprehensive plots
        figure = Figure(figsize=(14, 10))
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, density_window)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # Create the comprehensive density analysis plots
        self._create_density_analysis_plots(figure, density_data, analyzer, material_type)
        
        # Add export button
        export_btn = QPushButton("Export Density Plots")
        export_btn.clicked.connect(lambda: self.export_density_plots(figure, material_type))
        layout.addWidget(export_btn)
        
        # Show the window
        density_window.exec_()
    
    def _create_density_analysis_plots(self, figure, density_data, analyzer, material_type):
        """Create the 4-panel density analysis visualization."""
        figure.clear()
        
        # Create 2x2 subplot layout
        ax1 = figure.add_subplot(2, 2, 1)  # CDI Across All Spectra
        ax2 = figure.add_subplot(2, 2, 2)  # Density Across All Spectra  
        ax3 = figure.add_subplot(2, 2, 3)  # Density Distribution
        ax4 = figure.add_subplot(2, 2, 4)  # CDI vs Density Correlation
        
        positions = density_data['positions']
        cdi_profile = density_data['cdi_profile']
        density_profile = density_data['density_profile']
        classifications = density_data['layer_classification']
        stats = density_data['statistics']
        
        # 1. CDI Across All Spectra
        ax1.plot(positions, cdi_profile, 'g-', linewidth=2, marker='o', markersize=4)
        
        # Add classification threshold line
        threshold = analyzer.classification_thresholds.get('medium', 0.5)
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label='Classification threshold')
        
        ax1.set_xlabel('Spectrum Number')
        ax1.set_ylabel('Crystalline Density Index')
        ax1.set_title('CDI Across All Spectra')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Density Across All Spectra
        ax2.plot(positions, density_profile, 'b-', linewidth=2, marker='o', markersize=4)
        
        # Add mean line
        mean_density = stats['mean_density']
        ax2.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_density:.3f}')
        
        ax2.set_xlabel('Spectrum Number')
        ax2.set_ylabel('Density (g/cm³)')
        ax2.set_title('Density Across All Spectra')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Density Distribution
        ax3.hist(density_profile, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=mean_density, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_density:.3f}')
        ax3.set_xlabel('Density (g/cm³)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Density Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. CDI vs Density Correlation
        # Color-code by classification
        classification_colors = {
            'Low crystallinity': 'lightblue',
            'Mixed regions': 'orange', 
            'Mixed crystalline': 'purple',
            'Pure crystalline': 'darkblue',
            'bacterial': 'cyan',
            'organic': 'red',
            'crystalline': 'blue'
        }
        
        # Plot points colored by classification
        for classification in set(classifications):
            mask = [c == classification for c in classifications]
            cdi_subset = [cdi_profile[i] for i in range(len(cdi_profile)) if mask[i]]
            density_subset = [density_profile[i] for i in range(len(density_profile)) if mask[i]]
            
            color = classification_colors.get(classification, 'gray')
            ax4.scatter(cdi_subset, density_subset, 
                       c=color, label=classification, alpha=0.7, s=50)
        
        ax4.set_xlabel('Crystalline Density Index')
        ax4.set_ylabel('Density (g/cm³)')
        ax4.set_title('CDI vs Density Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add vertical line at classification threshold
        ax4.axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)
        
        figure.suptitle(f'{material_type} - Comprehensive Density Analysis\n'
                       f'N={stats["n_spectra"]} spectra', 
                       fontsize=14, fontweight='bold')
        
        figure.tight_layout()
        
        # Redraw the canvas
        try:
            figure.canvas.draw()
        except:
            pass
    
    def export_density_plots(self, figure, material_type):
        """Export the density analysis plots to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Export Density Plots", 
            f"density_analysis_{material_type.replace(' ', '_').replace('(', '').replace(')', '')}.png",
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        
        if file_path:
            try:
                figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self.main_window, "Export Successful", 
                                      f"Density analysis plots exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self.main_window, "Export Error", 
                                   f"Failed to export plots:\n{str(e)}") 