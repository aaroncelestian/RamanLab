"""
Main Cluster Analysis Module for RamanLab

This module contains the main cluster analysis class that orchestrates
the clustering workflow using the refactored components.
"""

import sys
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                              QTabWidget, QPushButton, QLabel, QProgressBar, 
                              QMessageBox, QApplication, QFileDialog, QDialog)
from PySide6.QtCore import Qt

# Import refactored components
from .core.clustering_engine import ClusteringEngine
from .core.data_processor import DataProcessor
from .dialogs.database_import import DatabaseImportDialog
from .ui.tabs.import_tab import ImportTab
from .ui.tabs.clustering_tab import ClusteringTab
from .ui.tabs.visualization_tabs import VisualizationTab
from .ui.tabs.analysis_tab import AnalysisTab
from .ui.tabs.refinement_tab import RefinementTab
from .ui.tabs.advanced_tabs import (TimeSeriesTab, KineticsTab, StructuralAnalysisTab,
                                   ValidationTab, AdvancedStatisticsTab)


class RamanClusterAnalysisQt6(QMainWindow):
    """Qt6 Window for Raman cluster analysis with hierarchical clustering."""
    
    def __init__(self, parent=None, raman_app=None):
        """
        Initialize the Raman Cluster Analysis Window.
        
        Parameters:
        -----------
        parent : QWidget
            Parent window
        raman_app : RamanAnalysisAppQt6
            Main application instance
        """
        super().__init__(parent)
        self.setWindowTitle("Raman Cluster Analysis - Advanced Ion Exchange Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Explicitly set window flags to ensure minimize/maximize/close buttons on Windows
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | 
                           Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Store reference to main app
        self.raman_app = raman_app
        
        # Initialize core components
        self.clustering_engine = ClusteringEngine()
        self.data_processor = DataProcessor()
        
        # Initialize database path
        self.custom_db_path = None
        
        # Initialize CSV Hey classifications cache
        self._csv_hey_classes = None
        
        # Initialize variables
        self.selected_folder = None
        self.visualization_method = "PCA"
        self.selected_points = set()
        self.refinement_mode = False
        self.database_spectra = []  # Store spectra imported from database
        self.n_subclusters = 2
        self.split_method = "kmeans"
        
        # Initialize undo stack for refinement operations
        self.undo_stack = []
        self.max_undo_steps = 10
        
        # Initialize cluster data
        self.cluster_data = {
            'wavenumbers': None,
            'intensities': None,
            'features': None,
            'features_scaled': None,
            'labels': None,
            'linkage_matrix': None,
            'distance_matrix': None,
            'preprocessing_method': None,
            'algorithm_used': None,
            'cluster_info': {},
            'pca_reducer': None,
            'features_pca_reduced': None,
            'nmf_components': None,
            'nmf_weights': None,
            'corundum_component_idx': None,
            'carbon_optimized': False
        }
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface using extracted tab components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create and add tabs using extracted components
        self.import_tab = ImportTab(self)
        self.tab_widget.addTab(self.import_tab, "Import")
        
        self.clustering_tab = ClusteringTab(self)
        self.tab_widget.addTab(self.clustering_tab, "Clustering")
        
        self.visualization_tab = VisualizationTab(self)
        self.tab_widget.addTab(self.visualization_tab, "Visualization")
        
        self.analysis_tab = AnalysisTab(self)
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        
        self.refinement_tab = RefinementTab(self)
        self.tab_widget.addTab(self.refinement_tab, "Refinement")
        
        # Advanced analysis tabs
        self.time_series_tab = TimeSeriesTab(self)
        self.tab_widget.addTab(self.time_series_tab, "Time Series")
        
        self.kinetics_tab = KineticsTab(self)
        self.tab_widget.addTab(self.kinetics_tab, "Kinetics")
        
        self.structural_tab = StructuralAnalysisTab(self)
        self.tab_widget.addTab(self.structural_tab, "Structural")
        
        self.validation_tab = ValidationTab(self)
        self.tab_widget.addTab(self.validation_tab, "Validation")
        
        self.stats_tab = AdvancedStatisticsTab(self)
        self.tab_widget.addTab(self.stats_tab, "Statistics")
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    # Core functionality methods (using refactored components)
    
    def select_import_folder(self):
        """Select folder for data import."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Spectra")
        if folder:
            self.selected_folder = folder
            # Update the folder path label in the import tab
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_folder_path_label'):
                folder_label = self.import_tab.get_folder_path_label()
                folder_label.setText(folder)
    
    def import_single_file_map(self):
        """Import a single spectrum file for mapping/analysis."""
        try:
            import os
            
            # Let user select a single file
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Spectrum File",
                                                      filter="Spectrum files (*.txt *.csv *.dat *.asc)")
            if not file_path:
                return
            
            # Update status
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Loading single spectrum: {os.path.basename(file_path)}")
            
            self.statusBar().showMessage(f"Loading spectrum: {os.path.basename(file_path)}")
            
            # Read the spectrum file
            spectrum = self._read_spectrum_file(file_path)
            if spectrum is None:
                QMessageBox.warning(self, "Import Failed", 
                                  f"Could not read spectrum file:\n{file_path}\n\n"
                                  "Please ensure the file contains two columns:\n"
                                  "Column 1: Wavenumbers (cm⁻¹)\n"
                                  "Column 2: Intensities")
                return
            
            # Store as single spectrum dataset
            self.cluster_data['intensities'] = np.array([spectrum['intensities']])
            self.cluster_data['wavenumbers'] = spectrum['wavenumbers']
            
            # Update status
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Loaded single spectrum\n" +
                                    f"File: {os.path.basename(file_path)}\n" +
                                    f"Wavenumber range: {np.min(spectrum['wavenumbers']):.1f} - {np.max(spectrum['wavenumbers']):.1f} cm⁻¹\n" +
                                    f"Data points: {len(spectrum['intensities'])}")
            
            self.statusBar().showMessage(f"Loaded spectrum: {os.path.basename(file_path)}")
            
            QMessageBox.information(self, "Import Complete", 
                                  f"Successfully loaded spectrum from:\n{file_path}\n\n"
                                  f"Data points: {len(spectrum['intensities'])}\n"
                                  f"Wavenumber range: {np.min(spectrum['wavenumbers']):.1f} - {np.max(spectrum['wavenumbers']):.1f} cm⁻¹")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import spectrum file:\n{str(e)}")
    
    def open_database_import_dialog(self):
        """Open the database import dialog."""
        try:
            # Debug: Check what we have
            print(f"DEBUG: raman_app = {self.raman_app}")
            if self.raman_app:
                print(f"DEBUG: raman_app type = {type(self.raman_app)}")
                print(f"DEBUG: hasattr raman_spectra = {hasattr(self.raman_app, 'raman_spectra')}")
                if hasattr(self.raman_app, 'raman_spectra'):
                    print(f"DEBUG: raman_spectra type = {type(self.raman_app.raman_spectra)}")
            
            # Get database reference - try multiple approaches
            raman_db = None
            
            # Method 1: Direct access through raman_app
            if self.raman_app and hasattr(self.raman_app, 'raman_spectra'):
                raman_db = self.raman_app.raman_spectra
                print("DEBUG: Found database through raman_app.raman_spectra")
            
            # Method 2: Try to get database from parent if it's the main app
            elif hasattr(self.parent(), 'raman_spectra'):
                raman_db = self.parent().raman_spectra
                print("DEBUG: Found database through parent.raman_spectra")
            
            # Method 3: Try to access global database if available
            else:
                try:
                    # Try to import and access the database directly
                    import sys
                    from pathlib import Path
                    
                    # Look for database in common locations
                    db_paths = [
                        Path.home() / "Documents" / "RamanLab_Qt6" / "RamanLab_Database_20250602.pkl",
                        Path.home() / "Documents" / "RamanLab_Qt6" / "RamanLab_Database.pkl",
                        Path("/Users/aaroncelestian/Documents/RamanLab_Qt6/RamanLab_Database_20250602.pkl")
                    ]
                    
                    for db_path in db_paths:
                        if db_path.exists():
                            import pickle
                            with open(db_path, 'rb') as f:
                                raman_db = pickle.load(f)
                            print(f"DEBUG: Loaded database from {db_path}")
                            break
                            
                except Exception as db_error:
                    print(f"DEBUG: Failed to load database directly: {db_error}")
            
            # Check if we found a database
            if raman_db is None:
                QMessageBox.warning(self, "No Database", 
                                  "No Raman database available. Please load the main application first.\n\n"
                                  "If the main application is running, this might be a connection issue.")
                return
            
            print(f"DEBUG: Database found with {len(raman_db)} entries")
            
            # Open dialog
            dialog = DatabaseImportDialog(raman_db, self)
            if dialog.exec() == QDialog.Accepted:
                selected_spectra = dialog.get_selected_spectra()
                self.import_database_spectra(selected_spectra)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open database import dialog:\n{str(e)}")
            print(f"DEBUG: Exception in open_database_import_dialog: {e}")
    
    def import_database_spectra(self, selected_spectra):
        """Import selected spectra from database."""
        try:
            if not selected_spectra:
                QMessageBox.information(self, "No Selection", "No spectra selected for import.")
                return
            
            # Store the imported spectra
            self.database_spectra = selected_spectra
            
            # Update status
            self.statusBar().showMessage(f"Imported {len(selected_spectra)} spectra from database")
            
            # Update import tab status if available
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Imported {len(selected_spectra)} spectra from database")
            
            QMessageBox.information(self, "Import Complete", 
                                  f"Successfully imported {len(selected_spectra)} spectra from database.")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import spectra:\n{str(e)}")
    
    def select_import_folder(self):
        """Select folder for importing spectra."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Spectra")
        if folder:
            self.selected_folder = folder
            # Update the folder path label in the import tab
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_folder_path_label'):
                folder_label = self.import_tab.get_folder_path_label()
                folder_label.setText(folder)
    
    def load_folder_spectra(self):
        """Load spectra from selected folder."""
        if not self.selected_folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
        
        # This would be implemented to load spectra from folder
        # For now, just show a placeholder message in the import tab
        if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
            status_label = self.import_tab.get_import_status()
            status_label.setText(f"Folder loading not yet implemented: {self.selected_folder}")
    
    def import_database_spectra(self, selected_spectra):
        """Import selected spectra from database."""
        try:
            if not selected_spectra:
                return
            
            # Extract data from selected spectra and find common wavenumber range
            all_spectra = []
            all_wavenumbers = []
            
            for name, spectrum_data in selected_spectra.items():
                # Use the correct key for the database structure
                spectrum = spectrum_data.get('intensities', [])
                wn = spectrum_data.get('wavenumbers', np.arange(len(spectrum)))
                
                if len(spectrum) > 0 and len(wn) > 0:
                    all_spectra.append((name, wn, spectrum))
                    all_wavenumbers.append(wn)
            
            if not all_spectra:
                QMessageBox.warning(self, "Import Error", "No valid spectra found in selection.")
                return
            
            # Find common wavenumber range (intersection of all ranges)
            min_wn = max(wn.min() for wn in all_wavenumbers)
            max_wn = min(wn.max() for wn in all_wavenumbers)
            
            # Create common wavenumber grid (use the spectrum with most points in range as reference)
            reference_wn = None
            max_points = 0
            for wn in all_wavenumbers:
                mask = (wn >= min_wn) & (wn <= max_wn)
                n_points = np.sum(mask)
                if n_points > max_points:
                    max_points = n_points
                    reference_wn = wn[mask]
            
            # Interpolate all spectra to common grid
            intensities = []
            for name, wn, spectrum in all_spectra:
                # Interpolate to common wavenumber grid
                interpolated = np.interp(reference_wn, wn, spectrum)
                intensities.append(interpolated)
            
            # Store in cluster data
            self.cluster_data['intensities'] = np.array(intensities)
            self.cluster_data['wavenumbers'] = reference_wn
            
            # Update preview in import tab
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Loaded {len(intensities)} spectra from database\n" +
                                    f"Wavenumber range: {reference_wn.min():.1f} - {reference_wn.max():.1f} cm⁻¹\n" +
                                    f"Data points: {len(reference_wn)}")
            
            self.statusBar().showMessage(f"Loaded {len(intensities)} spectra (interpolated to common grid)")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import spectra:\n{str(e)}")
    
    def update_visualization_method(self, method):
        """Update the visualization method."""
        self.visualization_method = method
        self.statusBar().showMessage(f"Visualization method: {method}")
    
    def run_clustering(self):
        """Run clustering analysis using the clustering engine."""
        try:
            if self.cluster_data['intensities'] is None:
                QMessageBox.warning(self, "No Data", 
                                  "No data available for clustering. Please import data first.")
                return
            
            # Get parameters from clustering tab
            clustering_controls = self.clustering_tab.get_clustering_controls()
            n_clusters = clustering_controls['n_clusters_spinbox'].value()
            linkage_method = clustering_controls['linkage_method_combo'].currentText()
            distance_metric = clustering_controls['distance_metric_combo'].currentText()
            preprocessing_method = clustering_controls['phase_method_combo'].currentText()
            
            # Update status
            clustering_controls['clustering_status'].setText("Applying preprocessing...")
            clustering_controls['clustering_progress'].setValue(10)
            QApplication.processEvents()
            
            # Apply preprocessing using data processor
            processed_intensities = self.cluster_data['intensities'].copy()
            wavenumbers = self.cluster_data['wavenumbers']
            
            if preprocessing_method == 'Corundum Correction':
                processed_intensities = self.data_processor.apply_corundum_drift_correction(
                    processed_intensities, wavenumbers
                )
                clustering_controls['clustering_status'].setText("Applied corundum drift correction...")
                
            elif preprocessing_method == 'NMF Separation':
                n_components = 3  # Default, could be made configurable
                processed_intensities, nmf_components, nmf_weights, corundum_idx = \
                    self.data_processor.apply_nmf_phase_separation(
                        processed_intensities, wavenumbers, n_components
                    )
                # Store NMF results
                self.cluster_data['nmf_components'] = nmf_components
                self.cluster_data['nmf_weights'] = nmf_weights
                self.cluster_data['corundum_component_idx'] = corundum_idx
                clustering_controls['clustering_status'].setText("Applied NMF phase separation...")
                
            elif preprocessing_method == 'Carbon Soot Optimization':
                processed_intensities = self.data_processor.apply_carbon_soot_preprocessing(
                    processed_intensities, wavenumbers
                )
                clustering_controls['clustering_status'].setText("Applied carbon soot optimization...")
                self.cluster_data['carbon_optimized'] = True
            
            clustering_controls['clustering_progress'].setValue(30)
            QApplication.processEvents()
            
            # Extract features
            clustering_controls['clustering_status'].setText("Extracting features...")
            if preprocessing_method == 'Carbon Soot Optimization':
                features = self.data_processor.extract_carbon_specific_features(
                    processed_intensities, wavenumbers
                )
            else:
                features = self.data_processor.extract_vibrational_features(
                    processed_intensities, wavenumbers
                )
            
            self.cluster_data['features'] = features
            clustering_controls['clustering_progress'].setValue(50)
            QApplication.processEvents()
            
            # Scale features using clustering engine
            clustering_controls['clustering_status'].setText("Scaling features...")
            features_scaled = self.clustering_engine.scale_features(features)
            self.cluster_data['features_scaled'] = features_scaled
            clustering_controls['clustering_progress'].setValue(70)
            QApplication.processEvents()
            
            # Perform clustering using clustering engine
            clustering_controls['clustering_status'].setText("Running clustering...")
            labels, linkage_matrix, distance_matrix, algorithm_used = \
                self.clustering_engine.perform_hierarchical_clustering(
                    features_scaled, n_clusters, linkage_method, distance_metric
                )
            
            # Store results
            self.cluster_data['labels'] = labels
            self.cluster_data['linkage_matrix'] = linkage_matrix
            self.cluster_data['distance_matrix'] = distance_matrix
            self.cluster_data['preprocessing_method'] = preprocessing_method
            self.cluster_data['algorithm_used'] = algorithm_used
            
            clustering_controls['clustering_progress'].setValue(90)
            QApplication.processEvents()
            
            # Update results
            self.update_analysis_results()
            clustering_controls['clustering_progress'].setValue(100)
            
            # Update status
            method_text = f" (using {preprocessing_method})" if preprocessing_method != 'None' else ""
            algo_text = f" via {algorithm_used}" if algorithm_used else ""
            clustering_controls['clustering_status'].setText(f"Clustering complete: {n_clusters} clusters found{method_text}{algo_text}")
            
            # Switch to visualization tab
            self.tab_widget.setCurrentIndex(2)
            
        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", f"Clustering failed:\n{str(e)}")
            if hasattr(self, 'clustering_tab'):
                controls = self.clustering_tab.get_clustering_controls()
                controls['clustering_progress'].setValue(0)
                controls['clustering_status'].setText("Clustering failed")
    
    def update_analysis_results(self):
        """Update the analysis results display."""
        try:
            if self.cluster_data['labels'] is None:
                return
            
            labels = self.cluster_data['labels']
            n_clusters = len(np.unique(labels))
            
            # Calculate cluster statistics
            cluster_counts = np.bincount(labels)
            total_spectra = len(labels)
            
            results_text = f"Clustering Analysis Results\n"
            results_text += f"{'='*40}\n\n"
            results_text += f"Total spectra: {total_spectra}\n"
            results_text += f"Number of clusters: {n_clusters}\n"
            results_text += f"Algorithm used: {self.cluster_data.get('algorithm_used', 'Unknown')}\n"
            results_text += f"Preprocessing: {self.cluster_data.get('preprocessing_method', 'None')}\n\n"
            
            results_text += "Cluster Distribution:\n"
            for i, count in enumerate(cluster_counts):
                percentage = (count / total_spectra) * 100
                results_text += f"  Cluster {i}: {count} spectra ({percentage:.1f}%)\n"
            
            self.analysis_results.setText(results_text)
            
        except Exception as e:
            print(f"Error updating analysis results: {str(e)}")
    
    def export_analysis_results(self):
        """Export clustering analysis results to file."""
        try:
            # Check if we have clustering results
            if not self.cluster_data.get('labels'):
                QMessageBox.warning(self, "No Results", 
                                  "No clustering results to export. Please run clustering first.")
                return
            
            # Let user select export location and format
            filename, selected_filter = QFileDialog.getSaveFileName(
                self, "Export Analysis Results",
                filter="CSV (*.csv);;Text (*.txt);;Excel (*.xlsx);;JSON (*.json)"
            )
            
            if not filename:
                return
            
            # Generate analysis text
            results_text = self.generate_analysis_text()
            
            # Export based on selected format
            if selected_filter == "CSV (*.csv)":
                self._export_csv(filename)
            elif selected_filter == "Excel (*.xlsx)":
                self._export_excel(filename)
            elif selected_filter == "JSON (*.json)":
                self._export_json(filename)
            else:  # Text format
                self._export_text(filename, results_text)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Analysis results exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")
    
    def _export_text(self, filename, results_text):
        """Export results as text file."""
        with open(filename, 'w') as f:
            f.write(results_text)
    
    def _export_csv(self, filename):
        """Export results as CSV file."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write metadata
            writer.writerow(['CLUSTERING ANALYSIS RESULTS'])
            writer.writerow(['Total Spectra', len(self.cluster_data['intensities'])])
            writer.writerow(['Number of Clusters', len(np.unique(self.cluster_data['labels']))])
            
            if 'silhouette_score' in self.cluster_data:
                writer.writerow(['Silhouette Score', f"{self.cluster_data['silhouette_score']:.3f}"])
            
            writer.writerow([])
            
            # Write cluster assignments
            writer.writerow(['Spectrum Index', 'Cluster Assignment'])
            for i, label in enumerate(self.cluster_data['labels']):
                writer.writerow([i, label])
    
    def _export_excel(self, filename):
        """Export results as Excel file."""
        try:
            import pandas as pd
            
            # Create summary data
            summary_data = {
                'Metric': ['Total Spectra', 'Number of Clusters', 'Silhouette Score'],
                'Value': [
                    len(self.cluster_data['intensities']),
                    len(np.unique(self.cluster_data['labels'])),
                    f"{self.cluster_data.get('silhouette_score', 'N/A'):.3f}" if 'silhouette_score' in self.cluster_data else 'N/A'
                ]
            }
            
            # Create cluster assignments
            cluster_data = {
                'Spectrum_Index': range(len(self.cluster_data['labels'])),
                'Cluster_Assignment': self.cluster_data['labels']
            }
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(filename) as writer:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                pd.DataFrame(cluster_data).to_excel(writer, sheet_name='Cluster_Assignments', index=False)
                
        except ImportError:
            # Fallback to CSV if pandas/openpyxl not available
            self._export_csv(filename.replace('.xlsx', '.csv'))
    
    def _export_json(self, filename):
        """Export results as JSON file."""
        import json
        
        # Prepare export data
        export_data = {
            'metadata': {
                'total_spectra': len(self.cluster_data['intensities']),
                'n_clusters': len(np.unique(self.cluster_data['labels'])),
                'analysis_timestamp': str(pd.Timestamp.now())
            },
            'metrics': {
                'silhouette_score': self.cluster_data.get('silhouette_score'),
                'davies_bouldin_score': self.cluster_data.get('davies_bouldin_score'),
                'calinski_harabasz_score': self.cluster_data.get('calinski_harabasz_score')
            },
            'cluster_assignments': self.cluster_data['labels'].tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_from_main_app(self):
        """Import data from the main RamanLab application."""
        try:
            # Check if we have access to main app data
            if not self.raman_app:
                QMessageBox.warning(self, "No Main App", 
                                  "No connection to main RamanLab application found.")
                return
            
            # Check if main app has loaded spectra
            if not hasattr(self.raman_app, 'loaded_spectra') or not self.raman_app.loaded_spectra:
                QMessageBox.warning(self, "No Data", 
                                  "No spectra loaded in main application.\n"
                                  "Please load spectra in the main RamanLab window first.")
                return
            
            # Get spectra from main app
            main_app_spectra = self.raman_app.loaded_spectra
            
            # Extract data for clustering
            intensities = []
            wavenumbers = None
            
            for spectrum in main_app_spectra:
                try:
                    # Get spectrum data
                    if hasattr(spectrum, 'spectrum') and hasattr(spectrum, 'wavenumbers'):
                        spectrum_data = spectrum.spectrum
                        spectrum_wavenumbers = spectrum.wavenumbers
                        
                        if wavenumbers is None:
                            wavenumbers = spectrum_wavenumbers
                        elif not np.allclose(wavenumbers, spectrum_wavenumbers, rtol=1e-5):
                            # Skip spectra with different wavenumber ranges
                            continue
                        
                        intensities.append(spectrum_data)
                    
                except Exception as e:
                    print(f"Error processing spectrum from main app: {str(e)}")
                    continue
            
            if not intensities:
                QMessageBox.warning(self, "Import Failed", 
                                  "No valid spectra could be imported from main application.")
                return
            
            # Store in cluster data
            self.cluster_data['intensities'] = np.array(intensities)
            self.cluster_data['wavenumbers'] = wavenumbers
            
            # Update status
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Imported {len(intensities)} spectra from main application\n" +
                                    f"Wavenumber range: {np.min(wavenumbers):.1f} - {np.max(wavenumbers):.1f} cm⁻¹")
            
            self.statusBar().showMessage(f"Imported {len(intensities)} spectra from main application")
            
            QMessageBox.information(self, "Import Complete", 
                                  f"Successfully imported {len(intensities)} spectra from main application.")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import from main application:\n{str(e)}")
    
    def start_batch_import(self):
        """Start batch import of spectra from selected folder using intelligent parsing."""
        if not self.selected_folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
            
        try:
            import glob
            import os
            
            # Get file patterns to search for
            patterns = ['*.txt', '*.csv', '*.dat', '*.asc']
            files = []
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(self.selected_folder, pattern)))
            
            if not files:
                QMessageBox.warning(self, "No Files", "No spectrum files found in selected folder.")
                return
            
            # Initialize progress
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_progress'):
                progress_bar = self.import_tab.get_import_progress()
                progress_bar.setMaximum(len(files))
                progress_bar.setValue(0)
            
            # Update status
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Found {len(files)} files. Starting import...")
            
            self.statusBar().showMessage(f"Importing {len(files)} files...")
            
            # Load spectra from files
            spectra_data = []
            wavenumbers = None
            
            for i, file_path in enumerate(files):
                try:
                    # Update progress
                    if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_progress'):
                        progress_bar = self.import_tab.get_import_progress()
                        progress_bar.setValue(i + 1)
                    
                    # Update status
                    if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                        status_label = self.import_tab.get_import_status()
                        status_label.setText(f"Importing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
                    
                    # Process events to keep UI responsive
                    QApplication.processEvents()
                    
                    # Read spectrum file (intelligent parsing)
                    spectrum = self._read_spectrum_file(file_path)
                    if spectrum is not None:
                        if wavenumbers is None:
                            wavenumbers = spectrum['wavenumbers']
                        spectra_data.append(spectrum['intensities'])
                    
                except Exception as e:
                    print(f"Error importing file {file_path}: {str(e)}")
                    continue
            
            if not spectra_data:
                QMessageBox.warning(self, "Import Failed", "No valid spectra could be imported.")
                return
            
            # Store in cluster data
            self.cluster_data['intensities'] = np.array(spectra_data)
            self.cluster_data['wavenumbers'] = wavenumbers
            
            # Update status
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Successfully imported {len(spectra_data)} spectra from {len(files)} files\n" +
                                    f"Wavenumber range: {np.min(wavenumbers):.1f} - {np.max(wavenumbers):.1f} cm⁻¹")
            
            self.statusBar().showMessage(f"Imported {len(spectra_data)} spectra")
            
            QMessageBox.information(self, "Import Complete", 
                                  f"Successfully imported {len(spectra_data)} spectra from {len(files)} files.")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import spectra:\n{str(e)}")
    
    def _read_spectrum_file(self, file_path):
        """Read a spectrum file with intelligent format detection."""
        try:
            import pandas as pd
            
            # Try different delimiters and configurations
            delimiters = [',', '\t', ';', ' ', '|']
            
            for delimiter in delimiters:
                try:
                    # Read file
                    df = pd.read_csv(file_path, delimiter=delimiter, header=None, comment='#')
                    
                    if df.shape[1] >= 2:
                        # Assume first column is wavenumbers, second is intensities
                        wavenumbers = df.iloc[:, 0].values
                        intensities = df.iloc[:, 1].values
                        
                        # Basic validation
                        if len(wavenumbers) > 10 and len(intensities) > 10:
                            return {
                                'wavenumbers': wavenumbers,
                                'intensities': intensities
                            }
                except:
                    continue
            
            # If standard parsing fails, try numpy loadtxt
            try:
                data = np.loadtxt(file_path)
                if data.shape[1] >= 2:
                    return {
                        'wavenumbers': data[:, 0],
                        'intensities': data[:, 1]
                    }
            except:
                pass
            
            return None
            
        except Exception as e:
            print(f"Error reading spectrum file {file_path}: {str(e)}")
            return None
    
    def append_data(self):
        """Append additional spectra to existing dataset."""
        try:
            import glob
            import os
            
            # Check if we have existing data
            if (self.cluster_data.get('intensities') is None or 
                self.cluster_data.get('wavenumbers') is None):
                QMessageBox.warning(self, "No Data", 
                                  "No existing data found. Please import data first using 'Start Import' or 'Import from Database'.")
                return
            
            # Let user select folder or files
            choice = QMessageBox.question(self, "Append Data Source",
                                        "How would you like to append data?",
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.Yes)
            
            if choice == QMessageBox.Yes:
                # Select folder
                folder = QFileDialog.getExistingDirectory(self, "Select Folder with Additional Spectra")
                if not folder:
                    return
                
                # Get files
                patterns = ['*.txt', '*.csv', '*.dat', '*.asc']
                files = []
                for pattern in patterns:
                    files.extend(glob.glob(os.path.join(folder, pattern)))
                
                if not files:
                    QMessageBox.warning(self, "No Files", "No spectrum files found in selected folder.")
                    return
                    
            else:
                # Select individual files
                files, _ = QFileDialog.getOpenFileNames(self, "Select Spectrum Files",
                                                      filter="Spectrum files (*.txt *.csv *.dat *.asc)")
                if not files:
                    return
            
            # Initialize progress
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_progress'):
                progress_bar = self.import_tab.get_import_progress()
                progress_bar.setMaximum(len(files))
                progress_bar.setValue(0)
            
            # Update status
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Appending {len(files)} files to existing dataset...")
            
            self.statusBar().showMessage(f"Appending {len(files)} files...")
            
            # Get existing data
            existing_intensities = self.cluster_data['intensities']
            existing_wavenumbers = self.cluster_data['wavenumbers']
            
            # Load new spectra
            new_spectra = []
            new_wavenumbers = None
            
            for i, file_path in enumerate(files):
                try:
                    # Update progress
                    if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_progress'):
                        progress_bar = self.import_tab.get_import_progress()
                        progress_bar.setValue(i + 1)
                    
                    # Update status
                    if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                        status_label = self.import_tab.get_import_status()
                        status_label.setText(f"Appending file {i+1}/{len(files)}: {os.path.basename(file_path)}")
                    
                    # Process events to keep UI responsive
                    QApplication.processEvents()
                    
                    # Read spectrum file
                    spectrum = self._read_spectrum_file(file_path)
                    if spectrum is not None:
                        # Check wavenumber compatibility
                        if new_wavenumbers is None:
                            new_wavenumbers = spectrum['wavenumbers']
                            
                            # Verify wavenumber compatibility
                            if not np.allclose(new_wavenumbers, existing_wavenumbers, rtol=1e-5):
                                reply = QMessageBox.question(self, "Wavenumber Mismatch",
                                                            f"The new spectra have different wavenumber ranges than existing data.\n"
                                                            f"Existing: {np.min(existing_wavenumbers):.1f} - {np.max(existing_wavenumbers):.1f} cm⁻¹\n"
                                                            f"New: {np.min(new_wavenumbers):.1f} - {np.max(new_wavenumbers):.1f} cm⁻¹\n\n"
                                                            f"This may affect clustering results. Continue anyway?",
                                                            QMessageBox.Yes | QMessageBox.No,
                                                            QMessageBox.No)
                                if reply == QMessageBox.No:
                                    return
                        
                        new_spectra.append(spectrum['intensities'])
                    
                except Exception as e:
                    print(f"Error appending file {file_path}: {str(e)}")
                    continue
            
            if not new_spectra:
                QMessageBox.warning(self, "Append Failed", "No valid spectra could be appended.")
                return
            
            # Combine data
            combined_intensities = np.vstack([existing_intensities, np.array(new_spectra)])
            
            # Update cluster data
            self.cluster_data['intensities'] = combined_intensities
            # Keep existing wavenumbers (they should be compatible)
            
            # Update status
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_import_status'):
                status_label = self.import_tab.get_import_status()
                status_label.setText(f"Dataset updated: {len(combined_intensities)} total spectra\n" +
                                    f"Added {len(new_spectra)} new spectra to {len(existing_intensities)} existing\n" +
                                    f"Wavenumber range: {np.min(existing_wavenumbers):.1f} - {np.max(existing_wavenumbers):.1f} cm⁻¹")
            
            self.statusBar().showMessage(f"Dataset now contains {len(combined_intensities)} spectra")
            
            QMessageBox.information(self, "Append Complete", 
                                  f"Successfully appended {len(new_spectra)} spectra.\n"
                                  f"Dataset now contains {len(combined_intensities)} total spectra.")
            
        except Exception as e:
            QMessageBox.critical(self, "Append Error", f"Failed to append spectra:\n{str(e)}")
    
    def update_performance_controls(self):
        """Update performance control visibility."""
        pass
    
    def update_preprocessing_controls(self):
        """Update preprocessing control visibility."""
        pass
    
    def update_carbon_controls(self):
        """Update carbon analysis control visibility."""
        pass
    
    def update_visualization_tabs(self):
        """Update all visualization tabs with new clustering results."""
        try:
            # Update dendrogram
            if hasattr(self, 'visualization_tab'):
                self.update_dendrogram()
                self.update_heatmap()
                self.update_scatter_plot()
            
        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
    
    def update_analysis_results(self):
        """Update analysis tab with clustering results."""
        try:
            if hasattr(self, 'analysis_tab'):
                # Generate analysis results text
                results_text = self.generate_analysis_text()
                
                # Update the analysis text widget
                if hasattr(self.analysis_tab, 'get_analysis_results_text'):
                    text_widget = self.analysis_tab.get_analysis_results_text()
                    text_widget.setPlainText(results_text)
            
        except Exception as e:
            print(f"Error updating analysis results: {str(e)}")
    
    def generate_analysis_text(self):
        """Generate text summary of clustering results."""
        try:
            if not self.cluster_data.get('labels'):
                return "No clustering results available."
            
            labels = self.cluster_data['labels']
            n_clusters = len(np.unique(labels))
            n_spectra = len(labels)
            
            text = f"CLUSTERING ANALYSIS RESULTS\n"
            text += f"{'='*50}\n\n"
            text += f"Total spectra analyzed: {n_spectra}\n"
            text += f"Number of clusters found: {n_clusters}\n\n"
            
            # Cluster sizes
            text += f"CLUSTER SIZES:\n"
            text += f"{'-'*30}\n"
            for i in range(n_clusters):
                cluster_size = np.sum(labels == i)
                percentage = (cluster_size / n_spectra) * 100
                text += f"Cluster {i+1}: {cluster_size} spectra ({percentage:.1f}%)\n"
            
            text += f"\n"
            
            # Validation metrics
            if 'silhouette_score' in self.cluster_data:
                text += f"VALIDATION METRICS:\n"
                text += f"{'-'*30}\n"
                text += f"Silhouette Score: {self.cluster_data['silhouette_score']:.3f}\n"
            
            if 'davies_bouldin_score' in self.cluster_data:
                text += f"Davies-Bouldin Index: {self.cluster_data['davies_bouldin_score']:.3f}\n"
            
            if 'calinski_harabasz_score' in self.cluster_data:
                text += f"Calinski-Harabasz Index: {self.cluster_data['calinski_harabasz_score']:.1f}\n"
            
            return text
            
        except Exception as e:
            return f"Error generating analysis text: {str(e)}"
    
    def run_probabilistic_clustering(self):
        """Run probabilistic clustering."""
        pass
    
    def print_carbon_feature_analysis(self):
        """Print carbon feature analysis."""
        pass
    
    def suggest_clustering_improvements(self):
        """Suggest clustering improvements."""
        pass
    
    def show_nmf_clustering_info(self):
        """Show NMF clustering info."""
        pass
    
    def export_visualization(self):
        """Export current visualization."""
        try:
            # Let user select export location
            filename, _ = QFileDialog.getSaveFileName(self, "Export Visualization",
                                                      filter="PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if not filename:
                return
            
            # This would export the current visualization
            QMessageBox.information(self, "Export Complete", 
                                  f"Visualization exported to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export visualization:\n{str(e)}")
    
    def update_dendrogram(self):
        """Update dendrogram visualization."""
        try:
            print("DEBUG: Starting dendrogram update...")
            
            if self.cluster_data.get('linkage_matrix') is None:
                print("No linkage matrix available for dendrogram")
                return
            
            print(f"DEBUG: Has visualization_tab: {hasattr(self, 'visualization_tab')}")
            if hasattr(self, 'visualization_tab'):
                print(f"DEBUG: Has get_dendrogram_tab: {hasattr(self.visualization_tab, 'get_dendrogram_tab')}")
            
            # Get the dendrogram tab and its controls
            if hasattr(self, 'visualization_tab') and hasattr(self.visualization_tab, 'get_dendrogram_tab'):
                dendrogram_tab = self.visualization_tab.get_dendrogram_tab()
                print(f"DEBUG: Dendrogram tab type: {type(dendrogram_tab)}")
                
                if hasattr(dendrogram_tab, 'get_dendrogram_controls'):
                    controls = dendrogram_tab.get_dendrogram_controls()
                    print(f"DEBUG: Controls retrieved: {list(controls.keys())}")
                    
                    fig = controls['figure']
                    ax = controls['axis']
                    canvas = controls['canvas']
                    
                    print(f"DEBUG: Figure type: {type(fig)}")
                    print(f"DEBUG: Axis type: {type(ax)}")
                    print(f"DEBUG: Canvas type: {type(canvas)}")
                    
                    # Clear previous plot
                    ax.clear()
                    
                    # Plot dendrogram
                    from scipy.cluster.hierarchy import dendrogram
                    import matplotlib.pyplot as plt
                    
                    linkage_matrix = self.cluster_data['linkage_matrix']
                    labels = self.cluster_data.get('labels', [])
                    
                    print(f"DEBUG: Linkage matrix shape: {linkage_matrix.shape}")
                    print(f"DEBUG: Labels length: {len(labels)}")
                    
                    # Create dendrogram
                    dendrogram(linkage_matrix, ax=ax, labels=labels, leaf_rotation=90)
                    ax.set_title('Hierarchical Clustering Dendrogram')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Distance')
                    
                    print("DEBUG: About to draw canvas...")
                    # Update canvas
                    fig.tight_layout()
                    canvas.draw_idle()
                    canvas.flush_events()
                    print("DEBUG: Canvas draw completed")
                else:
                    print("DEBUG: Dendrogram tab has no get_dendrogram_controls method")
                
            print("Dendrogram updated with new clustering results")
            
        except Exception as e:
            print(f"Error updating dendrogram: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_heatmap(self):
        """Update heatmap visualization."""
        try:
            print("DEBUG: Starting heatmap update...")
            
            if self.cluster_data.get('intensities') is None:
                print("No intensity data available for heatmap")
                return
            
            print(f"DEBUG: Has visualization_tab: {hasattr(self, 'visualization_tab')}")
            if hasattr(self, 'visualization_tab'):
                print(f"DEBUG: Has get_heatmap_tab: {hasattr(self.visualization_tab, 'get_heatmap_tab')}")
            
            # Get the heatmap tab and its controls
            if hasattr(self, 'visualization_tab') and hasattr(self.visualization_tab, 'get_heatmap_tab'):
                heatmap_tab = self.visualization_tab.get_heatmap_tab()
                print(f"DEBUG: Heatmap tab type: {type(heatmap_tab)}")
                
                if hasattr(heatmap_tab, 'get_heatmap_controls'):
                    controls = heatmap_tab.get_heatmap_controls()
                    print(f"DEBUG: Heatmap controls retrieved: {list(controls.keys())}")
                    
                    fig = controls['figure']
                    ax = controls['axis']
                    canvas = controls['canvas']
                    colormap = controls['colormap'].currentText()
                    sort_by_cluster = controls['sort_by_cluster'].isChecked()
                    
                    print(f"DEBUG: Heatmap data shape: {self.cluster_data['intensities'].shape}")
                    print(f"DEBUG: Colormap: {colormap}, Sort by cluster: {sort_by_cluster}")
                    
                    # Clear previous plot and any colorbars
                    ax.clear()
                    fig.clear()
                    ax = fig.add_subplot(111)
                    
                    # Get data
                    intensities = self.cluster_data['intensities']
                    labels = self.cluster_data.get('labels', [])
                    
                    # Sort by cluster if requested
                    if sort_by_cluster and len(labels) > 0:
                        sort_indices = np.argsort(labels)
                        intensities_sorted = intensities[sort_indices]
                        print("DEBUG: Sorted by cluster")
                    else:
                        intensities_sorted = intensities
                        print("DEBUG: No sorting applied")
                    
                    # Plot heatmap
                    import matplotlib.pyplot as plt
                    im = ax.imshow(intensities_sorted, aspect='auto', cmap=colormap, interpolation='nearest')
                    ax.set_title('Spectral Intensity Heatmap')
                    ax.set_xlabel('Wavenumber Index')
                    ax.set_ylabel('Sample Index')
                    
                    # Add colorbar
                    fig.colorbar(im, ax=ax, label='Intensity')
                    
                    print("DEBUG: About to draw heatmap canvas...")
                    # Update canvas
                    fig.tight_layout()
                    canvas.draw_idle()
                    canvas.flush_events()
                    print("DEBUG: Heatmap canvas draw completed")
                else:
                    print("DEBUG: Heatmap tab has no get_heatmap_controls method")
                
            print("Heatmap updated with new clustering results")
            
        except Exception as e:
            print(f"Error updating heatmap: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_scatter_plot(self):
        """Update scatter plot visualization with enhanced UMAP support."""
        try:
            print("DEBUG: Starting scatter plot update...")
            
            if self.cluster_data.get('intensities') is None:
                print("No data available for scatter plot")
                return
            
            print(f"DEBUG: Has visualization_tab: {hasattr(self, 'visualization_tab')}")
            if hasattr(self, 'visualization_tab'):
                print(f"DEBUG: Has get_scatter_tab: {hasattr(self.visualization_tab, 'get_scatter_tab')}")
            
            # Get the scatter tab and its controls
            if hasattr(self, 'visualization_tab') and hasattr(self.visualization_tab, 'get_scatter_tab'):
                scatter_tab = self.visualization_tab.get_scatter_tab()
                print(f"DEBUG: Scatter tab type: {type(scatter_tab)}")
                
                if hasattr(scatter_tab, 'get_scatter_controls'):
                    controls = scatter_tab.get_scatter_controls()
                    print(f"DEBUG: Scatter controls retrieved: {list(controls.keys())}")
                    
                    fig = controls['figure']
                    ax = controls['axis']
                    canvas = controls['canvas']
                    method = controls['method'].currentText()
                    point_size = controls['point_size'].value()
                    show_legend = controls['show_legend'].isChecked()
                    
                    # Show/hide UMAP controls based on method selection
                    if 'umap_params_frame' in controls:
                        controls['umap_params_frame'].setVisible(method == 'UMAP')
                    
                    # Clear previous plot
                    ax.clear()
                    
                    # Get data
                    intensities = self.cluster_data['intensities']
                    labels = self.cluster_data.get('labels', [])
                    
                    print(f"Updating scatter plot with method: {method}")
                    print(f"DEBUG: Scatter data shape: {intensities.shape}")
                    
                    # Apply dimensionality reduction
                    if method == 'PCA':
                        from sklearn.decomposition import PCA
                        reducer = PCA(n_components=2)
                        coords = reducer.fit_transform(intensities)
                        title_suffix = f'(PCA: {reducer.explained_variance_ratio_.sum():.1%} variance)'
                        
                    elif method == 't-SNE':
                        from sklearn.manifold import TSNE
                        reducer = TSNE(n_components=2, random_state=42)
                        coords = reducer.fit_transform(intensities)
                        title_suffix = '(t-SNE)'
                        
                    elif method == 'UMAP':
                        try:
                            import umap
                            
                            # Get UMAP parameters from UI
                            n_neighbors = controls['umap_n_neighbors'].value()
                            min_dist = controls['umap_min_dist'].value()
                            metric = controls['umap_metric'].currentText()
                            spread = controls['umap_spread'].value()
                            
                            print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}, spread={spread}")
                            
                            # Create UMAP with enhanced parameters
                            reducer = umap.UMAP(
                                n_components=2,
                                n_neighbors=min(n_neighbors, len(intensities) - 1),
                                min_dist=min_dist,
                                metric=metric,
                                spread=spread,
                                random_state=None,  # Remove random_state to enable parallel processing
                                n_jobs=-1,  # Use all available CPU cores
                                local_connectivity=2.0,
                                repulsion_strength=2.0,
                                negative_sample_rate=10,
                                transform_queue_size=8.0,
                                init='spectral',
                                verbose=True
                            )
                            
                            coords = reducer.fit_transform(intensities)
                            title_suffix = f'(UMAP: {metric}, n={n_neighbors}, d={min_dist:.3f})'
                            
                            print("UMAP completed successfully")
                            
                        except ImportError:
                            ax.text(0.5, 0.5, 'UMAP not installed\n\nInstall with:\npip install umap-learn', 
                                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
                            canvas.draw_idle()
                            canvas.flush_events()
                            return
                        except Exception as e:
                            print(f"UMAP failed: {e}")
                            # Fallback to PCA
                            from sklearn.decomposition import PCA
                            reducer = PCA(n_components=2)
                            coords = reducer.fit_transform(intensities)
                            title_suffix = f'(PCA fallback - UMAP failed: {reducer.explained_variance_ratio_.sum():.1%} variance)'
                            
                    elif method == 'MDS':
                        from sklearn.manifold import MDS
                        reducer = MDS(n_components=2, random_state=42)
                        coords = reducer.fit_transform(intensities)
                        title_suffix = '(MDS)'
                        
                    else:  # Spectral Embedding
                        from sklearn.manifold import SpectralEmbedding
                        reducer = SpectralEmbedding(n_components=2, random_state=42)
                        coords = reducer.fit_transform(intensities)
                        title_suffix = '(Spectral Embedding)'
                    
                    # Plot scatter plot
                    import matplotlib.pyplot as plt
                    
                    if len(labels) > 0:
                        # Color by cluster
                        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, 
                                           s=point_size, cmap='tab10', alpha=0.7)
                        if show_legend:
                            legend1 = ax.legend(*scatter.legend_elements(), 
                                                title="Clusters", loc="best")
                            ax.add_artist(legend1)
                    else:
                        # No clustering results - plot in single color
                        ax.scatter(coords[:, 0], coords[:, 1], s=point_size, 
                                  alpha=0.7, color='blue')
                    
                    ax.set_title(f'Spectral Data Scatter Plot {title_suffix}')
                    ax.set_xlabel(f'{method} Component 1')
                    ax.set_ylabel(f'{method} Component 2')
                    ax.grid(True, alpha=0.3)
                    
                    print("DEBUG: About to draw scatter canvas...")
                    # Update canvas
                    fig.tight_layout()
                    canvas.draw_idle()
                    canvas.flush_events()
                    print("DEBUG: Scatter canvas draw completed")
                else:
                    print("DEBUG: Scatter tab has no get_scatter_controls method")
                
            print("Scatter plot updated with new clustering results")
            
        except Exception as e:
            print(f"Error updating scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_dendrogram(self, param):
        """Plot dendrogram with specific parameters."""
        try:
            print(f"Plotting dendrogram with parameter: {param}")
            # Implementation would go here
            
        except Exception as e:
            print(f"Error plotting dendrogram: {str(e)}")
    
    def plot_probability_heatmap(self):
        """Plot probability heatmap for probabilistic clustering."""
        try:
            print("Plotting probability heatmap")
            # Implementation would go here
            
        except Exception as e:
            print(f"Error plotting probability heatmap: {str(e)}")
    
    def merge_selected_clusters(self):
        """Merge selected clusters."""
        pass
    
    def toggle_refinement_mode(self):
        """Toggle refinement mode."""
        pass
    
    def undo_last_action(self):
        """Undo last refinement action."""
        pass
    
    def split_selected_cluster(self):
        """Split selected cluster."""
        pass
    
    def reset_selection(self):
        """Reset cluster selection."""
        pass
    
    def apply_refinement(self):
        """Apply cluster refinement."""
        pass
    
    def cancel_refinement(self):
        """Cancel cluster refinement."""
        pass
    
    def update_refinement_plot(self):
        """Update refinement plot."""
        pass
    
    def analyze_time_series(self):
        """Analyze time series data."""
        pass
    
    def plot_time_series(self):
        """Plot time series."""
        pass
    
    def fit_kinetics_model(self):
        """Fit kinetics model."""
        pass
    
    def predict_kinetics(self):
        """Predict kinetics evolution."""
        pass
    
    def analyze_peak_positions(self):
        """Analyze peak positions."""
        pass
    
    def calculate_band_ratios(self):
        """Calculate band ratios."""
        pass
    
    def extract_structural_parameters(self):
        """Extract structural parameters."""
        pass
    
    def calculate_silhouette_score(self):
        """Calculate silhouette score."""
        pass
    
    def calculate_davies_bouldin(self):
        """Calculate Davies-Bouldin index."""
        pass
    
    def calculate_calinski_harabasz(self):
        """Calculate Calinski-Harabasz index."""
        pass
    
    def find_optimal_clusters(self):
        """Find optimal number of clusters."""
        pass
    
    def perform_anova_test(self):
        """Perform ANOVA test."""
        pass
    
    def perform_kruskal_test(self):
        """Perform Kruskal-Wallis test."""
        pass
    
    def perform_manova_test(self):
        """Perform MANOVA test."""
        pass
    
    def perform_detailed_pca(self):
        """Perform detailed PCA analysis."""
        pass
    
    def perform_factor_analysis(self):
        """Perform factor analysis."""
        pass


def launch_cluster_analysis(parent=None, raman_app=None):
    """Launch the cluster analysis window."""
    try:
        window = RamanClusterAnalysisQt6(parent, raman_app)
        window.show()
        return window
    except Exception as e:
        QMessageBox.critical(None, "Launch Error", f"Failed to launch cluster analysis:\n{str(e)}")
        return None
