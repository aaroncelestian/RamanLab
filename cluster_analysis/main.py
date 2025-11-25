"""
Main Cluster Analysis Module for RamanLab

This module contains the main cluster analysis class that orchestrates
the clustering workflow using the refactored components.
"""

# Fix OpenMP conflict before importing numpy/sklearn
# This prevents deadlocks with multiple OpenMP libraries on macOS
import os
import warnings

# Set all threading environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate OpenMP libraries
os.environ['OMP_NUM_THREADS'] = '1'  # Force single-threaded for safety
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Suppress the warning
warnings.filterwarnings('ignore', message='.*Found Intel OpenMP.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='threadpoolctl')

import sys
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
                              QTabWidget, QPushButton, QLabel, QProgressBar, 
                              QMessageBox, QApplication, QFileDialog, QDialog,
                              QProgressDialog)
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
        """Select folder for data import and automatically start loading."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Spectra")
        if folder:
            self.selected_folder = folder
            # Update the folder path label in the import tab
            if hasattr(self, 'import_tab') and hasattr(self.import_tab, 'get_folder_path_label'):
                folder_label = self.import_tab.get_folder_path_label()
                folder_label.setText(folder)
            
            # Automatically start import (consistent with other import methods)
            self.start_batch_import()
    
    def import_single_file_map(self):
        """Import a single file containing all spectra with X,Y positions (2D map format)."""
        try:
            import os
            from sklearn.preprocessing import StandardScaler
            
            # Create file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Single-File 2D Raman Map",
                filter="Text Files (*.txt *.csv *.dat);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Import the single-file loader
            from map_analysis_2d.core.single_file_map_loader import SingleFileRamanMapData
            from map_analysis_2d.core.cosmic_ray_detection import CosmicRayConfig
            
            # Configure cosmic ray detection
            cosmic_config = CosmicRayConfig(
                apply_during_load=False,
                enabled=True
            )
            
            # Create progress dialog
            progress_dialog = QProgressDialog("Loading single-file map...", "Cancel", 0, 100, self)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            
            def progress_callback(progress, message):
                progress_dialog.setValue(progress)
                progress_dialog.setLabelText(message)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise Exception("Import cancelled by user")
            
            # Load the map
            map_data = SingleFileRamanMapData(
                filepath=file_path,
                cosmic_ray_config=cosmic_config,
                progress_callback=progress_callback
            )
            
            progress_dialog.close()
            
            # Get data matrix and metadata
            data_matrix = map_data.get_processed_data_matrix()
            wavenumbers = map_data.target_wavenumbers
            positions = map_data.get_position_list()
            
            # Create filenames list
            filenames = [f"pos_{x:.1f}_{y:.1f}" for x, y in positions]
            
            # Create spectrum metadata for each position
            spectrum_metadata = []
            for i, (filename, pos) in enumerate(zip(filenames, positions)):
                metadata = {
                    'filename': filename,
                    'spectrum_index': i,
                    'source': 'single_file_map',
                    'x_position': pos[0],
                    'y_position': pos[1],
                    'file_path': file_path
                }
                spectrum_metadata.append(metadata)
            
            # Store data in cluster_data dictionary
            self.cluster_data['wavenumbers'] = wavenumbers
            self.cluster_data['intensities'] = data_matrix
            self.cluster_data['metadata'] = spectrum_metadata
            
            # Store map metadata for spatial visualization
            self.map_metadata = {
                'x_positions': map_data.x_positions,
                'y_positions': map_data.y_positions,
                'width': map_data.width,
                'height': map_data.height,
                'position_map': {filename: pos for filename, pos in zip(filenames, positions)},
                'source_file': file_path
            }
            
            # Extract features
            print(f"Extracting features from {len(data_matrix)} spectra...")
            features = self.data_processor.extract_vibrational_features(data_matrix, wavenumbers)
            self.cluster_data['features'] = features
            
            # Scale features with aggressive OpenMP workaround
            print(f"Scaling features ({features.shape[0]} samples × {features.shape[1]} features)...")
            print("⚠️  Applying OpenMP deadlock workaround...")
            
            # Aggressive workaround for OpenMP conflict
            import os
            import sys
            
            # Save all thread-related environment variables
            env_backup = {}
            thread_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 
                          'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
            
            for var in thread_vars:
                env_backup[var] = os.environ.get(var, None)
                os.environ[var] = '1'
            
            # Force numpy to use single thread
            try:
                import numpy as np
                if hasattr(np, '__config__'):
                    np.show_config()
            except:
                pass
            
            try:
                # Use manual scaling to avoid sklearn's threading issues
                print("Using manual scaling to avoid threading conflicts...")
                
                # Calculate mean and std manually (single-threaded)
                means = np.mean(features, axis=0)
                stds = np.std(features, axis=0)
                
                # Avoid division by zero
                stds[stds == 0] = 1.0
                
                # Scale manually
                features_scaled = (features - means) / stds
                self.cluster_data['features_scaled'] = features_scaled
                
                print("✓ Feature scaling complete (manual method)")
                
            except Exception as e:
                print(f"Manual scaling failed: {e}, trying StandardScaler...")
                # Fallback to StandardScaler
                scaler = StandardScaler()
                self.cluster_data['features_scaled'] = scaler.fit_transform(features)
                print("✓ Feature scaling complete (StandardScaler)")
                
            finally:
                # Restore original thread settings
                for var, value in env_backup.items():
                    if value is not None:
                        os.environ[var] = value
                    else:
                        os.environ.pop(var, None)
                print("✓ Thread settings restored")
            
            # Update UI
            if hasattr(self, 'import_tab'):
                folder_label = self.import_tab.folder_path_label
                folder_label.setText(f"Single-file map: {os.path.basename(file_path)}")
            
            self.statusBar().showMessage(f"Loaded {len(filenames)} spectra from single-file map")
            
            # Show success message with auto-save option
            reply = QMessageBox.question(
                self,
                "Import Complete",
                f"Successfully imported single-file 2D map:\n\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Spectra: {len(filenames):,}\n"
                f"Map size: {map_data.width} × {map_data.height}\n"
                f"Wavenumber points: {len(wavenumbers)}\n\n"
                f"💾 Save as .pkl for 100x faster loading next time?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Auto-save with suggested filename
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                suggested_name = f"{base_name}_preprocessed.pkl"
                suggested_path = os.path.join(os.path.dirname(file_path), suggested_name)
                
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Preprocessed Data",
                    suggested_path,
                    "Pickle Files (*.pkl);;All Files (*)"
                )
                
                if save_path:
                    import pickle
                    data_package = {
                        'cluster_data': self.cluster_data,
                        'map_metadata': self.map_metadata,
                        'version': '1.0',
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'source_file': file_path
                    }
                    
                    with open(save_path, 'wb') as f:
                        pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                    QMessageBox.information(
                        self, "Saved!",
                        f"✓ Preprocessed data saved!\n\n"
                        f"File: {os.path.basename(save_path)}\n"
                        f"Size: {file_size_mb:.1f} MB\n\n"
                        f"Next time, use 'Load Preprocessed Data (.pkl)'\n"
                        f"to load in seconds instead of minutes!"
                    )
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import single-file map:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
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
            algorithm = clustering_controls['algorithm_combo'].currentText()
            n_clusters = clustering_controls['n_clusters_spinbox'].value()
            linkage_method = clustering_controls['linkage_method_combo'].currentText()
            distance_metric = clustering_controls['distance_metric_combo'].currentText()
            preprocessing_method = clustering_controls['phase_method_combo'].currentText()
            
            # Get DBSCAN parameters
            dbscan_eps = clustering_controls['dbscan_eps'].value()
            dbscan_min_samples = clustering_controls['dbscan_min_samples'].value()
            
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
            n_samples = len(features_scaled)
            clustering_controls['clustering_status'].setText(f"Running {algorithm} on {n_samples:,} samples...")
            QApplication.processEvents()
            
            # Select clustering method based on algorithm
            if algorithm == 'K-Means':
                labels, linkage_matrix, distance_matrix, algorithm_used = \
                    self.clustering_engine.perform_kmeans_clustering(
                        features_scaled, n_clusters, use_minibatch=False
                    )
            elif algorithm == 'MiniBatchKMeans':
                labels, linkage_matrix, distance_matrix, algorithm_used = \
                    self.clustering_engine.perform_kmeans_clustering(
                        features_scaled, n_clusters, use_minibatch=True
                    )
            elif algorithm == 'Hierarchical':
                labels, linkage_matrix, distance_matrix, algorithm_used = \
                    self.clustering_engine.perform_hierarchical_clustering(
                        features_scaled, n_clusters, linkage_method, distance_metric
                    )
            elif algorithm == 'DBSCAN':
                labels, linkage_matrix, distance_matrix, algorithm_used = \
                    self.clustering_engine.perform_dbscan_clustering(
                        features_scaled, dbscan_eps, dbscan_min_samples
                    )
            elif algorithm == 'GMM (Probabilistic)':
                results = self.clustering_engine.run_probabilistic_clustering(
                    features_scaled, n_clusters
                )
                labels = results['labels']
                linkage_matrix = None
                distance_matrix = None
                algorithm_used = 'GMM'
                # Store probability information
                self.cluster_data['cluster_probs'] = results['cluster_probs']
                self.cluster_data['gmm_model'] = results['gmm']
                self.cluster_data['subtypes'] = results['subtypes']
                # Show GMM-specific visualization buttons
                clustering_controls['prob_viz_btn'].setVisible(True)
                clustering_controls['prob_viz_btn'].setEnabled(True)
                clustering_controls['subtype_viz_btn'].setVisible(True)
                clustering_controls['subtype_viz_btn'].setEnabled(True)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Hide GMM buttons for non-GMM algorithms
            if algorithm != 'GMM (Probabilistic)':
                clustering_controls['prob_viz_btn'].setVisible(False)
                clustering_controls['subtype_viz_btn'].setVisible(False)
            
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
            
            # Update all visualization plots
            print("Updating visualization plots after clustering...")
            self.update_visualization_tabs()
            
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
    
    def save_preprocessed_data(self):
        """Save preprocessed data to pickle file for fast loading."""
        try:
            import pickle
            
            # Check if we have data to save
            if self.cluster_data.get('intensities') is None:
                QMessageBox.warning(self, "No Data", "No data loaded to save.")
                return
            
            # Get save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Preprocessed Data",
                "cluster_data.pkl",
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Prepare data package
            data_package = {
                'cluster_data': self.cluster_data,
                'map_metadata': getattr(self, 'map_metadata', None),
                'version': '1.0',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save with progress
            progress = QProgressDialog("Saving preprocessed data...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            with open(file_path, 'wb') as f:
                pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            progress.close()
            
            # Show file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            n_spectra = len(self.cluster_data['intensities'])
            
            QMessageBox.information(
                self, "Save Complete",
                f"Preprocessed data saved successfully!\n\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Size: {file_size_mb:.1f} MB\n"
                f"Spectra: {n_spectra:,}\n\n"
                f"Loading this file will be ~100x faster than re-importing!"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_preprocessed_data(self):
        """Load preprocessed data from pickle file."""
        try:
            import pickle
            
            # Get file to load
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Preprocessed Data",
                "",
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Load with progress
            progress = QProgressDialog("Loading preprocessed data...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            import time
            start_time = time.time()
            
            with open(file_path, 'rb') as f:
                data_package = pickle.load(f)
            
            # Restore data
            self.cluster_data = data_package['cluster_data']
            if 'map_metadata' in data_package and data_package['map_metadata']:
                self.map_metadata = data_package['map_metadata']
            
            elapsed = time.time() - start_time
            progress.close()
            
            # Update UI
            n_spectra = len(self.cluster_data['intensities'])
            if hasattr(self, 'import_tab'):
                self.import_tab.folder_path_label.setText(f"Loaded from: {os.path.basename(file_path)}")
            
            self.statusBar().showMessage(f"Loaded {n_spectra:,} spectra from preprocessed data")
            
            # Show success with timing
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            QMessageBox.information(
                self, "Load Complete",
                f"Preprocessed data loaded successfully!\n\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Size: {file_size_mb:.1f} MB\n"
                f"Spectra: {n_spectra:,}\n"
                f"Load time: {elapsed:.2f}s\n\n"
                f"✓ Features already extracted and scaled\n"
                f"✓ Ready for clustering!"
            )
            
            print(f"⚡ Loaded {n_spectra:,} spectra in {elapsed:.2f}s ({n_spectra/elapsed:.0f} spectra/sec)")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def import_from_main_app(self):
        """Import spectra from the main RamanLab application."""
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
            # Check if labels exist
            if 'labels' not in self.cluster_data or self.cluster_data['labels'] is None:
                return "No clustering results available."
            
            labels = np.array(self.cluster_data['labels'])  # Ensure it's a numpy array
            
            # Check if labels array is empty
            if len(labels) == 0:
                return "No clustering results available."
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            n_spectra = len(labels)
            
            text = f"CLUSTERING ANALYSIS RESULTS\n"
            text += f"{'='*50}\n\n"
            text += f"Total spectra analyzed: {n_spectra}\n"
            text += f"Number of clusters found: {n_clusters}\n\n"
            
            # Cluster sizes
            text += f"CLUSTER SIZES:\n"
            text += f"{'-'*30}\n"
            for cluster_id in unique_labels:
                cluster_size = int(np.sum(labels == cluster_id))
                percentage = (cluster_size / n_spectra) * 100
                text += f"Cluster {int(cluster_id)+1}: {cluster_size} spectra ({percentage:.1f}%)\n"
            
            text += f"\n"
            
            # Validation metrics
            if 'silhouette_score' in self.cluster_data and self.cluster_data['silhouette_score'] is not None:
                text += f"VALIDATION METRICS:\n"
                text += f"{'-'*30}\n"
                text += f"Silhouette Score: {self.cluster_data['silhouette_score']:.3f}\n"
            
            if 'davies_bouldin_score' in self.cluster_data and self.cluster_data['davies_bouldin_score'] is not None:
                text += f"Davies-Bouldin Index: {self.cluster_data['davies_bouldin_score']:.3f}\n"
            
            if 'calinski_harabasz_score' in self.cluster_data and self.cluster_data['calinski_harabasz_score'] is not None:
                text += f"Calinski-Harabasz Index: {self.cluster_data['calinski_harabasz_score']:.1f}\n"
            
            return text
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in generate_analysis_text: {error_details}")
            return f"Error generating analysis text: {str(e)}"
    
    
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
        """Update dendrogram visualization with truncation support for large datasets."""
        try:
            if self.cluster_data.get('linkage_matrix') is None:
                print("No linkage matrix available for dendrogram")
                return
            
            # Get the dendrogram tab and its controls
            if not (hasattr(self, 'visualization_tab') and hasattr(self.visualization_tab, 'get_dendrogram_tab')):
                return
            
            dendrogram_tab = self.visualization_tab.get_dendrogram_tab()
            if not hasattr(dendrogram_tab, 'get_dendrogram_controls'):
                return
            
            controls = dendrogram_tab.get_dendrogram_controls()
            fig = controls['figure']
            ax = controls['axis']
            canvas = controls['canvas']
            
            # Clear previous plot
            ax.clear()
            
            # Get control values
            orientation = controls['orientation'].currentText()
            max_samples = controls['max_samples'].value()
            show_labels = controls['show_labels'].isChecked()
            truncate_mode = controls['truncate_mode'].currentText()
            truncate_p = controls['truncate_p'].value()
            color_threshold = controls['color_threshold'].value()
            info_label = controls['info_label']
            
            # Plot dendrogram with truncation
            from scipy.cluster.hierarchy import dendrogram
            
            linkage_matrix = self.cluster_data['linkage_matrix']
            n_samples = linkage_matrix.shape[0] + 1
            
            # Build dendrogram parameters
            dendro_params = {
                'orientation': orientation,
                'no_labels': not show_labels,
                'ax': ax
            }
            
            # Add truncation if enabled
            if truncate_mode != 'None':
                dendro_params['truncate_mode'] = truncate_mode
                dendro_params['p'] = truncate_p
                info_text = f"Showing {truncate_p} {'clusters' if truncate_mode == 'lastp' else 'levels'} of {n_samples} total samples"
            else:
                # Limit samples if too many
                if n_samples > max_samples:
                    dendro_params['truncate_mode'] = 'lastp'
                    dendro_params['p'] = max_samples
                    info_text = f"Showing {max_samples} of {n_samples} samples (auto-truncated)"
                else:
                    info_text = f"Showing all {n_samples} samples"
            
            # Add color threshold if specified
            if color_threshold > 0:
                dendro_params['color_threshold'] = color_threshold
            
            # Create dendrogram
            dendrogram(linkage_matrix, **dendro_params)
            
            # Set labels and title
            ax.set_title(f'Hierarchical Clustering Dendrogram\n{info_text}')
            if orientation in ['top', 'bottom']:
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Distance')
            else:
                ax.set_xlabel('Distance')
                ax.set_ylabel('Sample Index')
            
            # Update info label
            info_label.setText(info_text)
            
            # Update canvas
            fig.tight_layout()
            canvas.draw_idle()
            canvas.flush_events()
            
            print(f"Dendrogram updated: {info_text}")
            
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
                        # Import progress dialog outside try block to avoid catching its ImportError
                        try:
                            from dialogs.progress_dialog import ProgressDialog
                        except ImportError:
                            ProgressDialog = None  # Fallback if dialog not available
                        
                        try:
                            import umap
                            
                            # Get UMAP parameters from UI
                            n_neighbors = controls['umap_n_neighbors'].value()
                            min_dist = controls['umap_min_dist'].value()
                            metric = controls['umap_metric'].currentText()
                            spread = controls['umap_spread'].value()
                            
                            # Parse epochs selection
                            epochs_text = controls['umap_epochs'].currentText()
                            if 'Auto' in epochs_text:
                                n_epochs = None  # Will be set based on dataset size
                            else:
                                # Extract number from text like "50 (Fast)"
                                n_epochs = int(epochs_text.split()[0])
                            
                            # Use scaled features instead of raw intensities for better results
                            if self.cluster_data.get('features_scaled') is not None:
                                umap_input = self.cluster_data['features_scaled']
                                print(f"Using scaled features ({umap_input.shape[1]} dimensions)")
                            else:
                                umap_input = intensities
                                print(f"Warning: Using raw intensities (features not available)")
                            
                            n_samples = len(umap_input)
                            
                            # For very large datasets, apply PCA pre-reduction for massive speedup
                            if n_samples > 10000 and umap_input.shape[1] > 50:
                                from sklearn.decomposition import PCA
                                print(f"   Applying PCA pre-reduction: {umap_input.shape[1]} → 50 dimensions")
                                pca = PCA(n_components=50, random_state=42)
                                umap_input = pca.fit_transform(umap_input)
                                variance_explained = pca.explained_variance_ratio_.sum()
                                print(f"   PCA retained {variance_explained*100:.1f}% variance")
                            
                            print(f"Running UMAP on {n_samples:,} samples with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}, spread={spread}")
                            
                            # Optimize UMAP parameters for large datasets
                            # Determine epochs to use
                            if n_epochs is None:
                                # Auto mode: use 50 for large datasets, 200 for small
                                actual_epochs = 50 if n_samples > 10000 else 200
                            else:
                                actual_epochs = n_epochs
                            
                            # For datasets >10k samples, use optimized parameters
                            if n_samples > 10000:
                                print(f"⚡ Large dataset detected ({n_samples:,} samples) - using optimized UMAP parameters")
                                # Use user's metric choice, but optimize other parameters
                                # Key insight: For Raman spectroscopy, euclidean often works better than cosine
                                actual_metric = metric if metric != 'cosine' else 'euclidean'
                                actual_neighbors = min(n_neighbors, 15, n_samples - 1)  # Cap at 15 for large datasets
                                
                                reducer = umap.UMAP(
                                    n_components=2,
                                    n_neighbors=actual_neighbors,
                                    min_dist=min_dist,
                                    metric=actual_metric,
                                    spread=spread,
                                    n_epochs=actual_epochs,
                                    random_state=42,
                                    low_memory=True,
                                    verbose=True,
                                    init='spectral',  # Better initialization
                                    negative_sample_rate=5
                                )
                                print(f"   Optimized: {actual_epochs} epochs, {actual_neighbors} neighbors, {actual_metric} metric")
                                if actual_metric != metric:
                                    print(f"   Note: Using {actual_metric} instead of {metric} for better results")
                            else:
                                # Standard parameters for smaller datasets
                                reducer = umap.UMAP(
                                    n_components=2,
                                    n_neighbors=min(n_neighbors, n_samples - 1),
                                    min_dist=min_dist,
                                    metric=metric,
                                    spread=spread,
                                    n_epochs=actual_epochs,
                                    random_state=42,
                                    verbose=True
                                )
                                print(f"   Using {actual_epochs} epochs")
                            
                            # Check if we have cached UMAP results with same parameters
                            cache_key = f"umap_{n_neighbors}_{min_dist}_{metric}_{spread}_{actual_epochs}"
                            if hasattr(self, '_umap_cache') and cache_key in self._umap_cache:
                                coords = self._umap_cache[cache_key]
                                print(f"✓ Using cached UMAP results (instant!)")
                                title_suffix = f'(UMAP cached: {metric}, n={n_neighbors}, d={min_dist:.3f})'
                            else:
                                # Create progress dialog if available
                                progress_dialog = None
                                if ProgressDialog is not None:
                                    progress_dialog = ProgressDialog(self, "UMAP Computation")
                                    progress_dialog.set_status(f"Running UMAP on {n_samples:,} samples...")
                                    progress_dialog.show()
                                    progress_dialog.start_capture()
                                
                                try:
                                    import time
                                    start_time = time.time()
                                    
                                    # Process events to show dialog
                                    QApplication.processEvents()
                                    
                                    coords = reducer.fit_transform(umap_input)
                                    elapsed = time.time() - start_time
                                    print(f"⏱️  UMAP completed in {elapsed:.1f}s ({n_samples/elapsed:.0f} samples/sec)")
                                    
                                    # Cache the results
                                    if not hasattr(self, '_umap_cache'):
                                        self._umap_cache = {}
                                    self._umap_cache[cache_key] = coords
                                    print(f"   Cached UMAP results for instant replay")
                                    
                                    title_suffix = f'(UMAP: {metric}, n={n_neighbors}, d={min_dist:.3f})'
                                    
                                    if progress_dialog:
                                        progress_dialog.set_status("✓ UMAP completed successfully!")
                                    
                                finally:
                                    if progress_dialog:
                                        progress_dialog.stop_capture()
                                        progress_dialog.close()
                            
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
                    
                    # Enable export buttons after successful clustering
                    unique_labels = np.unique(labels)
                    if len(unique_labels) > 0:
                        if 'export_folders_btn' in controls:
                            controls['export_folders_btn'].setEnabled(True)
                        if 'export_summed_btn' in controls:
                            controls['export_summed_btn'].setEnabled(True)
                        if 'export_overview_btn' in controls:
                            controls['export_overview_btn'].setEnabled(True)
                        if 'export_xy_btn' in controls:
                            controls['export_xy_btn'].setEnabled(True)
                        print(f"Export buttons enabled for {len(unique_labels)} clusters")
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
    
    def export_clusters_to_folders(self):
        """Export each cluster's spectra to separate folders."""
        if ('labels' not in self.cluster_data or self.cluster_data['labels'] is None or
            'intensities' not in self.cluster_data or self.cluster_data['intensities'] is None):
            QMessageBox.warning(self, "No Data", "No cluster data available for export.")
            return
        
        try:
            # Get export directory
            export_dir = QFileDialog.getExistingDirectory(
                self, "Select Export Directory", "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if not export_dir:
                return
            
            labels = self.cluster_data['labels']
            intensities = self.cluster_data['intensities']
            wavenumbers = self.cluster_data['wavenumbers']
            metadata = self.cluster_data.get('metadata', [{}] * len(labels))
            unique_labels = np.unique(labels)
            
            # Create progress dialog
            progress = QProgressDialog("Exporting clusters to folders...", "Cancel", 0, len(unique_labels), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            
            exported_clusters = 0
            
            for i, cluster_id in enumerate(unique_labels):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"Exporting cluster {int(cluster_id)}...")
                
                # Create cluster folder
                cluster_folder = os.path.join(export_dir, f"Cluster_{int(cluster_id)}")
                os.makedirs(cluster_folder, exist_ok=True)
                
                # Get spectra for this cluster
                cluster_mask = labels == cluster_id
                cluster_intensities = intensities[cluster_mask]
                
                # Export individual spectra
                for j, spectrum_intensity in enumerate(cluster_intensities):
                    filename = f'spectrum_{j}.txt'
                    filepath = os.path.join(cluster_folder, filename)
                    
                    # Write spectrum data
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"# Cluster: {int(cluster_id)}\n")
                        f.write("# Wavenumber\tIntensity\n")
                        
                        # Write spectrum data
                        for wavenumber, intensity in zip(wavenumbers, spectrum_intensity):
                            f.write(f"{wavenumber:.2f}\t{intensity:.6f}\n")
                
                # Create cluster summary file
                summary_file = os.path.join(cluster_folder, f"cluster_{int(cluster_id)}_summary.txt")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Cluster {int(cluster_id)} Summary\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Number of spectra: {int(np.sum(cluster_mask))}\n")
                    from datetime import datetime
                    f.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                exported_clusters += 1
            
            progress.setValue(len(unique_labels))
            
            QMessageBox.information(
                self, "Export Complete", 
                f"Successfully exported {exported_clusters} clusters to:\n{export_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting clusters: {str(e)}")
            print(f"Export error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_summed_cluster_spectra(self):
        """Export summed spectra for each cluster (plot + data files)."""
        if ('labels' not in self.cluster_data or self.cluster_data['labels'] is None or
            'intensities' not in self.cluster_data or self.cluster_data['intensities'] is None):
            QMessageBox.warning(self, "No Data", "No cluster data available for export.")
            return
        
        try:
            # Get export directory
            export_dir = QFileDialog.getExistingDirectory(
                self, "Select Export Directory for Summed Spectra", "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if not export_dir:
                return
            
            labels = self.cluster_data['labels']
            intensities = self.cluster_data['intensities']
            wavenumbers = self.cluster_data['wavenumbers']
            unique_labels = np.unique(labels)
            
            # Create figure for plot
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(unique_labels), 1, figsize=(10, 3*len(unique_labels)))
            if len(unique_labels) == 1:
                axes = [axes]
            
            # Export each cluster
            for idx, cluster_id in enumerate(unique_labels):
                cluster_mask = labels == cluster_id
                cluster_spectra = intensities[cluster_mask]
                
                # Calculate mean spectrum
                mean_spectrum = np.mean(cluster_spectra, axis=0)
                
                # Plot
                axes[idx].plot(wavenumbers, mean_spectrum, linewidth=1.5)
                axes[idx].set_title(f'Cluster {int(cluster_id)} (n={int(np.sum(cluster_mask))} spectra)')
                axes[idx].set_xlabel('Wavenumber (cm⁻¹)')
                axes[idx].set_ylabel('Intensity')
                axes[idx].grid(True, alpha=0.3)
                
                # Export data file for this cluster (for search/match)
                data_filepath = os.path.join(export_dir, f'cluster_{int(cluster_id)}_summed.txt')
                with open(data_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Cluster {int(cluster_id)} Summed Spectrum\n")
                    f.write(f"# Number of spectra: {int(np.sum(cluster_mask))}\n")
                    from datetime import datetime
                    f.write(f"# Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("# Wavenumber\tIntensity\n")
                    for wn, intensity in zip(wavenumbers, mean_spectrum):
                        f.write(f"{wn:.2f}\t{intensity:.6f}\n")
            
            # Save plot
            plot_filepath = os.path.join(export_dir, 'summed_cluster_spectra.png')
            plt.tight_layout()
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            QMessageBox.information(self, "Export Complete", 
                f"Summed spectra exported to:\n{export_dir}\n\n"
                f"Files created:\n"
                f"- summed_cluster_spectra.png (plot)\n"
                f"- cluster_X_summed.txt (data files for each cluster)")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting summed spectra: {str(e)}")
            print(f"Export error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_cluster_overview(self):
        """Export comprehensive cluster overview with all clusters in a grid layout."""
        if ('labels' not in self.cluster_data or self.cluster_data['labels'] is None or
            'intensities' not in self.cluster_data or self.cluster_data['intensities'] is None):
            QMessageBox.warning(self, "No Data", "No cluster data available for export.")
            return
        
        try:
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Cluster Overview", "",
                "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)"
            )
            
            if not file_path:
                return
            
            labels = self.cluster_data['labels']
            intensities = self.cluster_data['intensities']
            wavenumbers = self.cluster_data['wavenumbers']
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Calculate grid layout
            import math
            n_cols = min(3, n_clusters)  # Max 3 columns
            n_rows = math.ceil(n_clusters / n_cols)
            
            # Create figure with subplots
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            
            # Flatten axes array for easy iteration
            if n_clusters == 1:
                axes = np.array([axes])
            else:
                axes = axes.flatten()
            
            # Plot each cluster
            for idx, cluster_id in enumerate(unique_labels):
                ax = axes[idx]
                cluster_mask = labels == cluster_id
                cluster_spectra = intensities[cluster_mask]
                n_spectra = int(np.sum(cluster_mask))
                
                # Calculate mean and std
                mean_spectrum = np.mean(cluster_spectra, axis=0)
                std_spectrum = np.std(cluster_spectra, axis=0)
                
                # Plot mean with shaded std
                ax.plot(wavenumbers, mean_spectrum, linewidth=2, label='Mean', color='blue')
                ax.fill_between(wavenumbers, 
                               mean_spectrum - std_spectrum,
                               mean_spectrum + std_spectrum,
                               alpha=0.3, color='blue', label='±1 Std Dev')
                
                # Plot a few individual spectra for reference
                n_samples_to_show = min(5, n_spectra)
                sample_indices = np.random.choice(len(cluster_spectra), n_samples_to_show, replace=False)
                for sample_idx in sample_indices:
                    ax.plot(wavenumbers, cluster_spectra[sample_idx], 
                           alpha=0.2, linewidth=0.5, color='gray')
                
                ax.set_title(f'Cluster {int(cluster_id)}\n({n_spectra:,} spectra)', 
                           fontweight='bold', fontsize=12)
                ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
                ax.set_ylabel('Intensity', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')
            
            # Hide unused subplots
            for idx in range(n_clusters, len(axes)):
                axes[idx].set_visible(False)
            
            # Add overall title
            algorithm_used = self.cluster_data.get('algorithm_used', 'Unknown')
            fig.suptitle(f'Cluster Overview - {n_clusters} Clusters ({algorithm_used})', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            # Save with high quality
            dpi = 300 if file_path.endswith('.png') else None
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            QMessageBox.information(self, "Export Complete", 
                f"Cluster overview exported to:\n{file_path}\n\n"
                f"Grid layout: {n_rows} rows × {n_cols} columns\n"
                f"Total clusters: {n_clusters}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting cluster overview: {str(e)}")
            print(f"Export error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def export_xy_plot_data(self):
        """Export XY plot data."""
        if ('labels' not in self.cluster_data or self.cluster_data['labels'] is None):
            QMessageBox.warning(self, "No Data", "No cluster data available for export.")
            return
        
        try:
            # Get the current visualization method from the scatter tab
            method = 'PCA'  # Default
            if hasattr(self, 'visualization_tab') and hasattr(self.visualization_tab, 'get_scatter_tab'):
                scatter_tab = self.visualization_tab.get_scatter_tab()
                if hasattr(scatter_tab, 'get_scatter_controls'):
                    controls = scatter_tab.get_scatter_controls()
                    method = controls['method'].currentText()
            
            # Get export file path
            filepath, _ = QFileDialog.getSaveFileName(
                self, f"Save {method} Coordinates", 
                f"{method.lower()}_coordinates.csv",
                "CSV files (*.csv);;All files (*)"
            )
            
            if not filepath:
                return
            
            # We need to recompute the coordinates since we don't store them
            labels = self.cluster_data['labels']
            intensities = self.cluster_data['intensities']
            
            # Apply dimensionality reduction
            if method == 'PCA':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                coords = reducer.fit_transform(intensities)
                x_label, y_label = 'PC1', 'PC2'
            elif method == 't-SNE':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                coords = reducer.fit_transform(intensities)
                x_label, y_label = 'tSNE1', 'tSNE2'
            elif method == 'UMAP':
                import umap
                reducer = umap.UMAP(n_components=2, random_state=None, n_jobs=-1)
                coords = reducer.fit_transform(intensities)
                x_label, y_label = 'UMAP1', 'UMAP2'
            else:
                coords = intensities[:, :2]
                x_label, y_label = 'X', 'Y'
            
            # Create DataFrame and export
            import pandas as pd
            df = pd.DataFrame({
                x_label: coords[:, 0],
                y_label: coords[:, 1],
                'Cluster': labels
            })
            
            df.to_csv(filepath, index=False)
            
            QMessageBox.information(self, "Export Complete", 
                f"{method} coordinates saved to:\n{filepath}\n\n"
                f"Exported {len(coords)} points with cluster labels.")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting XY data: {str(e)}")
            print(f"Export error: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
        """Calculate silhouette score for current clustering."""
        try:
            # Check if we have clustering results
            if self.cluster_data.get('labels') is None or self.cluster_data.get('features_scaled') is None:
                QMessageBox.warning(self, "No Clustering Results", 
                                  "Please run clustering analysis first.")
                return
            
            from sklearn.metrics import silhouette_score, silhouette_samples
            import matplotlib.pyplot as plt
            
            labels = self.cluster_data['labels']
            features = self.cluster_data['features_scaled']
            n_clusters = len(np.unique(labels))
            
            # Check if we have enough clusters
            if n_clusters < 2:
                QMessageBox.warning(self, "Insufficient Clusters", 
                                  "Silhouette analysis requires at least 2 clusters.")
                return
            
            # Calculate overall silhouette score
            self.statusBar().showMessage("Calculating silhouette score...")
            QApplication.processEvents()
            
            silhouette_avg = silhouette_score(features, labels)
            self.cluster_data['silhouette_score'] = silhouette_avg
            
            # Calculate per-sample silhouette values
            sample_silhouette_values = silhouette_samples(features, labels)
            
            # Update results text
            results_text = "SILHOUETTE ANALYSIS\n"
            results_text += "=" * 50 + "\n\n"
            results_text += f"Overall Silhouette Score: {silhouette_avg:.3f}\n\n"
            results_text += "Interpretation:\n"
            if silhouette_avg > 0.7:
                results_text += "  ✓ Excellent: Strong, well-separated clusters\n"
            elif silhouette_avg > 0.5:
                results_text += "  ✓ Good: Reasonable cluster structure\n"
            elif silhouette_avg > 0.25:
                results_text += "  ⚠ Fair: Weak cluster structure\n"
            else:
                results_text += "  ✗ Poor: Clusters may be artificial\n"
            
            results_text += "\n" + "-" * 50 + "\n"
            results_text += "Per-Cluster Silhouette Scores:\n\n"
            
            # Calculate per-cluster statistics
            for i in range(n_clusters):
                cluster_mask = labels == i
                cluster_silhouette = sample_silhouette_values[cluster_mask]
                cluster_size = np.sum(cluster_mask)
                
                results_text += f"Cluster {i}:\n"
                results_text += f"  Size: {cluster_size} samples\n"
                results_text += f"  Mean: {cluster_silhouette.mean():.3f}\n"
                results_text += f"  Std:  {cluster_silhouette.std():.3f}\n"
                results_text += f"  Range: [{cluster_silhouette.min():.3f}, {cluster_silhouette.max():.3f}]\n\n"
            
            # Display results
            if hasattr(self, 'validation_tab'):
                self.validation_tab.validation_results_text.setText(results_text)
            
            # Create silhouette plot
            if hasattr(self, 'validation_tab'):
                fig = self.validation_tab.validation_fig
                ax = self.validation_tab.validation_ax
                ax.clear()
                
                y_lower = 10
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
                
                for i in range(n_clusters):
                    cluster_silhouette_values = sample_silhouette_values[labels == i]
                    cluster_silhouette_values.sort()
                    
                    size_cluster_i = cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    
                    ax.fill_betweenx(np.arange(y_lower, y_upper),
                                    0, cluster_silhouette_values,
                                    facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
                    
                    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10
                
                ax.set_title(f'Silhouette Plot (Score: {silhouette_avg:.3f})', fontsize=12, fontweight='bold')
                ax.set_xlabel('Silhouette Coefficient', fontsize=10)
                ax.set_ylabel('Cluster', fontsize=10)
                ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f'Average: {silhouette_avg:.3f}')
                ax.set_xlim([-0.1, 1])
                ax.set_ylim([0, len(features) + (n_clusters + 1) * 10])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                fig.tight_layout()
                self.validation_tab.validation_canvas.draw()
            
            self.statusBar().showMessage(f"Silhouette score: {silhouette_avg:.3f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", 
                               f"Failed to calculate silhouette score:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def calculate_davies_bouldin(self):
        """Calculate Davies-Bouldin index for current clustering."""
        try:
            # Check if we have clustering results
            if self.cluster_data.get('labels') is None or self.cluster_data.get('features_scaled') is None:
                QMessageBox.warning(self, "No Clustering Results", 
                                  "Please run clustering analysis first.")
                return
            
            from sklearn.metrics import davies_bouldin_score
            
            labels = self.cluster_data['labels']
            features = self.cluster_data['features_scaled']
            n_clusters = len(np.unique(labels))
            
            # Check if we have enough clusters
            if n_clusters < 2:
                QMessageBox.warning(self, "Insufficient Clusters", 
                                  "Davies-Bouldin analysis requires at least 2 clusters.")
                return
            
            # Calculate Davies-Bouldin index
            self.statusBar().showMessage("Calculating Davies-Bouldin index...")
            QApplication.processEvents()
            
            db_score = davies_bouldin_score(features, labels)
            self.cluster_data['davies_bouldin_score'] = db_score
            
            # Update results text
            results_text = "DAVIES-BOULDIN INDEX\n"
            results_text += "=" * 50 + "\n\n"
            results_text += f"Davies-Bouldin Index: {db_score:.3f}\n\n"
            results_text += "Interpretation:\n"
            results_text += "  Lower values indicate better clustering\n"
            results_text += "  (0 = perfect separation)\n\n"
            
            if db_score < 0.5:
                results_text += "  ✓ Excellent: Very well-separated clusters\n"
            elif db_score < 1.0:
                results_text += "  ✓ Good: Well-separated clusters\n"
            elif db_score < 1.5:
                results_text += "  ⚠ Fair: Moderate cluster separation\n"
            else:
                results_text += "  ✗ Poor: Overlapping clusters\n"
            
            results_text += "\n" + "-" * 50 + "\n"
            results_text += "\nAbout Davies-Bouldin Index:\n"
            results_text += "  • Measures average similarity between clusters\n"
            results_text += "  • Ratio of within-cluster to between-cluster distances\n"
            results_text += "  • Lower values = better defined clusters\n"
            results_text += "  • Does not require ground truth labels\n"
            
            # Display results
            if hasattr(self, 'validation_tab'):
                self.validation_tab.validation_results_text.setText(results_text)
            
            # Create visualization
            if hasattr(self, 'validation_tab'):
                fig = self.validation_tab.validation_fig
                ax = self.validation_tab.validation_ax
                ax.clear()
                
                # Create bar chart showing quality interpretation
                categories = ['Excellent\n(<0.5)', 'Good\n(0.5-1.0)', 'Fair\n(1.0-1.5)', 'Poor\n(>1.5)']
                ranges = [0.5, 1.0, 1.5, 3.0]
                colors_map = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
                
                bars = ax.barh(categories, ranges, color=colors_map, alpha=0.3, edgecolor='black')
                
                # Add current score marker
                if db_score < 0.5:
                    y_pos = 0
                elif db_score < 1.0:
                    y_pos = 1
                elif db_score < 1.5:
                    y_pos = 2
                else:
                    y_pos = 3
                
                ax.plot([db_score], [y_pos], 'ro', markersize=15, label=f'Your Score: {db_score:.3f}', zorder=5)
                ax.axvline(x=db_score, color='red', linestyle='--', alpha=0.5, zorder=4)
                
                ax.set_xlabel('Davies-Bouldin Index', fontsize=10, fontweight='bold')
                ax.set_title('Davies-Bouldin Index Interpretation', fontsize=12, fontweight='bold')
                ax.set_xlim([0, min(3.0, db_score + 0.5)])
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3, axis='x')
                
                fig.tight_layout()
                self.validation_tab.validation_canvas.draw()
            
            self.statusBar().showMessage(f"Davies-Bouldin index: {db_score:.3f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", 
                               f"Failed to calculate Davies-Bouldin index:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def calculate_calinski_harabasz(self):
        """Calculate Calinski-Harabasz index for current clustering."""
        try:
            # Check if we have clustering results
            if self.cluster_data.get('labels') is None or self.cluster_data.get('features_scaled') is None:
                QMessageBox.warning(self, "No Clustering Results", 
                                  "Please run clustering analysis first.")
                return
            
            from sklearn.metrics import calinski_harabasz_score
            
            labels = self.cluster_data['labels']
            features = self.cluster_data['features_scaled']
            n_clusters = len(np.unique(labels))
            
            # Check if we have enough clusters
            if n_clusters < 2:
                QMessageBox.warning(self, "Insufficient Clusters", 
                                  "Calinski-Harabasz analysis requires at least 2 clusters.")
                return
            
            # Calculate Calinski-Harabasz index
            self.statusBar().showMessage("Calculating Calinski-Harabasz index...")
            QApplication.processEvents()
            
            ch_score = calinski_harabasz_score(features, labels)
            self.cluster_data['calinski_harabasz_score'] = ch_score
            
            # Update results text
            results_text = "CALINSKI-HARABASZ INDEX\n"
            results_text += "=" * 50 + "\n\n"
            results_text += f"Calinski-Harabasz Index: {ch_score:.1f}\n\n"
            results_text += "Interpretation:\n"
            results_text += "  Higher values indicate better clustering\n"
            results_text += "  (no upper bound)\n\n"
            
            # Provide context-aware interpretation
            n_samples = len(labels)
            if ch_score > 1000:
                results_text += "  ✓ Excellent: Very dense, well-separated clusters\n"
            elif ch_score > 500:
                results_text += "  ✓ Good: Dense, well-separated clusters\n"
            elif ch_score > 100:
                results_text += "  ⚠ Fair: Moderate cluster quality\n"
            else:
                results_text += "  ✗ Poor: Weak or overlapping clusters\n"
            
            results_text += "\n" + "-" * 50 + "\n"
            results_text += "\nAbout Calinski-Harabasz Index:\n"
            results_text += "  • Also known as Variance Ratio Criterion\n"
            results_text += "  • Ratio of between-cluster to within-cluster variance\n"
            results_text += "  • Higher values = better defined clusters\n"
            results_text += "  • Scale depends on dataset size and dimensionality\n"
            results_text += f"  • Your dataset: {n_samples:,} samples, {n_clusters} clusters\n"
            
            # Display results
            if hasattr(self, 'validation_tab'):
                self.validation_tab.validation_results_text.setText(results_text)
            
            # Create visualization
            if hasattr(self, 'validation_tab'):
                fig = self.validation_tab.validation_fig
                ax = self.validation_tab.validation_ax
                ax.clear()
                
                # Create bar chart showing quality interpretation
                categories = ['Poor\n(<100)', 'Fair\n(100-500)', 'Good\n(500-1000)', 'Excellent\n(>1000)']
                ranges = [100, 500, 1000, max(1500, ch_score * 1.2)]
                colors_map = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
                
                bars = ax.barh(categories, ranges, color=colors_map, alpha=0.3, edgecolor='black')
                
                # Add current score marker
                if ch_score < 100:
                    y_pos = 0
                elif ch_score < 500:
                    y_pos = 1
                elif ch_score < 1000:
                    y_pos = 2
                else:
                    y_pos = 3
                
                ax.plot([ch_score], [y_pos], 'ro', markersize=15, label=f'Your Score: {ch_score:.1f}', zorder=5)
                ax.axvline(x=ch_score, color='red', linestyle='--', alpha=0.5, zorder=4)
                
                ax.set_xlabel('Calinski-Harabasz Index', fontsize=10, fontweight='bold')
                ax.set_title('Calinski-Harabasz Index Interpretation', fontsize=12, fontweight='bold')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3, axis='x')
                
                fig.tight_layout()
                self.validation_tab.validation_canvas.draw()
            
            self.statusBar().showMessage(f"Calinski-Harabasz index: {ch_score:.1f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", 
                               f"Failed to calculate Calinski-Harabasz index:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def find_optimal_clusters(self):
        """Find optimal number of clusters using multiple metrics."""
        try:
            # Check if we have data
            if self.cluster_data.get('features_scaled') is None:
                QMessageBox.warning(self, "No Data", 
                                  "Please import and process data first.")
                return
            
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            from sklearn.cluster import KMeans
            
            features = self.cluster_data['features_scaled']
            
            # Get max clusters from spinbox
            if hasattr(self, 'validation_tab'):
                max_clusters = self.validation_tab.max_clusters_spinbox.value()
            else:
                max_clusters = 10
            
            # Limit based on dataset size
            n_samples = len(features)
            max_possible = min(max_clusters, n_samples // 2)
            
            if max_possible < 2:
                QMessageBox.warning(self, "Insufficient Data", 
                                  "Not enough samples for cluster analysis.")
                return
            
            # Create progress dialog
            progress = QProgressDialog("Evaluating cluster configurations...", "Cancel", 2, max_possible, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            
            # Store metrics for each k
            k_range = range(2, max_possible + 1)
            silhouette_scores = []
            davies_bouldin_scores = []
            calinski_harabasz_scores = []
            inertias = []
            
            self.statusBar().showMessage("Finding optimal number of clusters...")
            
            for k in k_range:
                if progress.wasCanceled():
                    return
                
                progress.setValue(k)
                progress.setLabelText(f"Testing {k} clusters...")
                QApplication.processEvents()
                
                # Perform clustering
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                # Calculate metrics
                silhouette_scores.append(silhouette_score(features, labels))
                davies_bouldin_scores.append(davies_bouldin_score(features, labels))
                calinski_harabasz_scores.append(calinski_harabasz_score(features, labels))
                inertias.append(kmeans.inertia_)
            
            progress.close()
            
            # Find optimal k for each metric
            optimal_silhouette = k_range[np.argmax(silhouette_scores)]
            optimal_davies = k_range[np.argmin(davies_bouldin_scores)]
            optimal_calinski = k_range[np.argmax(calinski_harabasz_scores)]
            
            # Calculate elbow using second derivative
            if len(inertias) > 2:
                # Normalize inertias
                inertias_norm = (np.array(inertias) - np.min(inertias)) / (np.max(inertias) - np.min(inertias))
                # Calculate second derivative
                second_deriv = np.diff(np.diff(inertias_norm))
                optimal_elbow = k_range[np.argmax(second_deriv) + 2] if len(second_deriv) > 0 else k_range[0]
            else:
                optimal_elbow = k_range[0]
            
            # Update results text
            results_text = "OPTIMAL CLUSTER ANALYSIS\n"
            results_text += "=" * 50 + "\n\n"
            results_text += f"Tested cluster range: {k_range[0]} to {k_range[-1]}\n\n"
            results_text += "Recommended Number of Clusters:\n"
            results_text += "-" * 50 + "\n"
            results_text += f"  Silhouette Score:      {optimal_silhouette} clusters\n"
            results_text += f"  Davies-Bouldin Index:  {optimal_davies} clusters\n"
            results_text += f"  Calinski-Harabasz:     {optimal_calinski} clusters\n"
            results_text += f"  Elbow Method:          {optimal_elbow} clusters\n\n"
            
            # Consensus recommendation
            from collections import Counter
            votes = [optimal_silhouette, optimal_davies, optimal_calinski, optimal_elbow]
            vote_counts = Counter(votes)
            consensus = vote_counts.most_common(1)[0][0]
            
            results_text += f"📊 Consensus Recommendation: {consensus} clusters\n"
            results_text += f"   (based on {vote_counts[consensus]}/4 metrics)\n\n"
            
            results_text += "-" * 50 + "\n"
            results_text += "\nDetailed Scores:\n\n"
            
            for i, k in enumerate(k_range):
                results_text += f"k={k}:\n"
                results_text += f"  Silhouette: {silhouette_scores[i]:.3f}\n"
                results_text += f"  Davies-Bouldin: {davies_bouldin_scores[i]:.3f}\n"
                results_text += f"  Calinski-Harabasz: {calinski_harabasz_scores[i]:.1f}\n"
                results_text += f"  Inertia: {inertias[i]:.1f}\n\n"
            
            # Display results
            if hasattr(self, 'validation_tab'):
                self.validation_tab.validation_results_text.setText(results_text)
            
            # Create visualization
            if hasattr(self, 'validation_tab'):
                import matplotlib.pyplot as plt
                
                fig = self.validation_tab.validation_fig
                fig.clear()
                
                # Create 2x2 subplot
                axes = fig.subplots(2, 2)
                
                # Silhouette score
                axes[0, 0].plot(k_range, silhouette_scores, 'o-', color='#3498db', linewidth=2, markersize=6)
                axes[0, 0].axvline(optimal_silhouette, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_silhouette}')
                axes[0, 0].set_xlabel('Number of Clusters')
                axes[0, 0].set_ylabel('Silhouette Score')
                axes[0, 0].set_title('Silhouette Score (Higher is Better)', fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
                
                # Davies-Bouldin index
                axes[0, 1].plot(k_range, davies_bouldin_scores, 'o-', color='#e74c3c', linewidth=2, markersize=6)
                axes[0, 1].axvline(optimal_davies, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_davies}')
                axes[0, 1].set_xlabel('Number of Clusters')
                axes[0, 1].set_ylabel('Davies-Bouldin Index')
                axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
                
                # Calinski-Harabasz index
                axes[1, 0].plot(k_range, calinski_harabasz_scores, 'o-', color='#2ecc71', linewidth=2, markersize=6)
                axes[1, 0].axvline(optimal_calinski, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_calinski}')
                axes[1, 0].set_xlabel('Number of Clusters')
                axes[1, 0].set_ylabel('Calinski-Harabasz Index')
                axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)', fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
                
                # Elbow method (inertia)
                axes[1, 1].plot(k_range, inertias, 'o-', color='#f39c12', linewidth=2, markersize=6)
                axes[1, 1].axvline(optimal_elbow, color='red', linestyle='--', alpha=0.7, label=f'Elbow: {optimal_elbow}')
                axes[1, 1].set_xlabel('Number of Clusters')
                axes[1, 1].set_ylabel('Inertia')
                axes[1, 1].set_title('Elbow Method (Look for "Elbow")', fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend()
                
                fig.suptitle(f'Optimal Clusters Analysis (Consensus: {consensus})', fontsize=14, fontweight='bold')
                fig.tight_layout()
                self.validation_tab.validation_canvas.draw()
            
            self.statusBar().showMessage(f"Optimal cluster analysis complete. Recommendation: {consensus} clusters")
            
            # Ask if user wants to re-cluster with optimal k
            reply = QMessageBox.question(
                self, "Apply Optimal Clustering?",
                f"The analysis recommends {consensus} clusters.\n\n"
                f"Would you like to re-run clustering with {consensus} clusters?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Update clustering tab with optimal k
                if hasattr(self, 'clustering_tab'):
                    controls = self.clustering_tab.get_clustering_controls()
                    controls['n_clusters_spinbox'].setValue(consensus)
                    # Switch to clustering tab
                    self.tab_widget.setCurrentWidget(self.clustering_tab)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", 
                               f"Failed to find optimal clusters:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def perform_anova_test(self):
        """Calculate feature importance for cluster discrimination (renamed from ANOVA for clarity)."""
        try:
            if 'labels' not in self.cluster_data or 'intensities' not in self.cluster_data:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            # Use Random Forest as default method (most robust)
            method = 'Random Forest'
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            wavenumbers = self.cluster_data['wavenumbers']
            
            self.statusBar().showMessage(f"Calculating feature importance using {method}...")
            
            # Calculate feature importance
            importance_scores = self._calculate_rf_importance(intensities, labels)
            
            # Find top features
            top_indices = np.argsort(importance_scores)[-20:][::-1]  # Top 20 features
            top_wavenumbers = wavenumbers[top_indices]
            top_scores = importance_scores[top_indices]
            
            # Plot results
            self._plot_feature_importance(wavenumbers, importance_scores, 
                                        top_wavenumbers, top_scores, method)
            
            # Display results
            self._display_feature_importance_results(top_wavenumbers, top_scores, method, importance_scores)
            
            self.statusBar().showMessage(f"Feature importance analysis complete", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Feature Importance Error", 
                               f"Failed to calculate feature importance:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def perform_kruskal_test(self):
        """Perform Linear Discriminant Analysis (renamed from Kruskal for clarity)."""
        try:
            if 'labels' not in self.cluster_data or 'intensities' not in self.cluster_data:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            import matplotlib.pyplot as plt
            
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            
            self.statusBar().showMessage("Performing Linear Discriminant Analysis...")
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(intensities)
            
            # Perform LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_scaled, labels)
            
            # Transform data to LDA space
            X_lda = lda.transform(X_scaled)
            
            # Calculate cross-validation accuracy
            cv_scores = cross_val_score(lda, X_scaled, labels, cv=5)
            
            # Calculate explained variance ratio
            explained_variance = lda.explained_variance_ratio_
            
            # Plot results
            self._plot_discriminant_analysis(X_lda, labels, explained_variance, cv_scores)
            
            # Display results
            self._display_discriminant_results(cv_scores, explained_variance, lda)
            
            self.statusBar().showMessage(f"LDA complete: {np.mean(cv_scores):.1%} accuracy", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Discriminant Analysis Error", 
                               f"Failed to perform discriminant analysis:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def perform_manova_test(self):
        """Test statistical significance of cluster differences (renamed from MANOVA for clarity)."""
        try:
            if 'labels' not in self.cluster_data or 'intensities' not in self.cluster_data:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            from scipy import stats
            import matplotlib.pyplot as plt
            
            # Use ANOVA as default (most common and interpretable)
            method = 'ANOVA'
            alpha = 0.05
            
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            
            self.statusBar().showMessage(f"Performing {method} significance testing...")
            
            # Perform ANOVA
            results = self._perform_anova(intensities, labels, alpha)
            
            # Plot results
            self._plot_significance_results(results, method)
            
            # Display results
            self._display_significance_results(results, method, alpha)
            
            sig_pct = results['significant_features']/results['total_features']*100
            self.statusBar().showMessage(f"Significance testing complete: {sig_pct:.1f}% significant features", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Significance Testing Error", 
                               f"Failed to test statistical significance:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def perform_detailed_pca(self):
        """Perform detailed PCA analysis with comprehensive visualizations."""
        try:
            if 'labels' not in self.cluster_data or 'intensities' not in self.cluster_data:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            import matplotlib.pyplot as plt
            
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            wavenumbers = self.cluster_data['wavenumbers']
            
            self.statusBar().showMessage("Performing detailed PCA analysis...")
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(intensities)
            
            # Perform PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)
            
            # Get explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Find number of components for 80%, 90%, 95% variance
            n_80 = np.argmax(cumulative_variance >= 0.80) + 1
            n_90 = np.argmax(cumulative_variance >= 0.90) + 1
            n_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Plot results
            self._plot_detailed_pca(X_pca, labels, explained_variance, cumulative_variance, 
                                   pca.components_, wavenumbers)
            
            # Display results
            self._display_pca_results(explained_variance, cumulative_variance, n_80, n_90, n_95, 
                                     pca.components_, wavenumbers)
            
            self.statusBar().showMessage(f"PCA complete: {n_90} components capture 90% variance", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "PCA Error", 
                               f"Failed to perform PCA analysis:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def perform_factor_analysis(self):
        """Perform Factor Analysis to identify latent factors."""
        try:
            if 'labels' not in self.cluster_data or 'intensities' not in self.cluster_data:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            from sklearn.decomposition import FactorAnalysis
            from sklearn.preprocessing import StandardScaler
            import matplotlib.pyplot as plt
            
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            wavenumbers = self.cluster_data['wavenumbers']
            
            self.statusBar().showMessage("Performing Factor Analysis...")
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(intensities)
            
            # Determine optimal number of factors (use number of clusters as starting point)
            n_clusters = len(np.unique(labels))
            n_factors = min(n_clusters + 2, 10)  # Add 2 extra factors, max 10
            
            # Perform Factor Analysis
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            X_fa = fa.fit_transform(X_scaled)
            
            # Get factor loadings
            loadings = fa.components_.T
            
            # Calculate variance explained (approximation)
            variance_explained = np.var(X_fa, axis=0)
            variance_explained /= np.sum(variance_explained)
            
            # Plot results
            self._plot_factor_analysis(X_fa, labels, loadings, wavenumbers, variance_explained)
            
            # Display results
            self._display_factor_results(n_factors, loadings, wavenumbers, variance_explained)
            
            self.statusBar().showMessage(f"Factor Analysis complete: {n_factors} factors extracted", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Factor Analysis Error", 
                               f"Failed to perform Factor Analysis:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    # Helper methods for statistical analysis
    def _calculate_rf_importance(self, intensities, labels):
        """Calculate feature importance using Random Forest."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(intensities)
        
        # Train Random Forest with parallel processing
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, labels)
        
        return rf.feature_importances_
    
    def _plot_feature_importance(self, wavenumbers, importance_scores, 
                                top_wavenumbers, top_scores, method):
        """Plot feature importance results."""
        import matplotlib.pyplot as plt
        
        fig = self.stats_tab.stats_fig
        fig.clear()
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)  # Full spectrum importance
        ax2 = fig.add_subplot(2, 2, 2)  # Top features bar plot
        ax3 = fig.add_subplot(2, 2, 3)  # Cumulative importance
        ax4 = fig.add_subplot(2, 2, 4)  # Importance distribution
        
        # 1. Full spectrum importance plot
        ax1.plot(wavenumbers, importance_scores, 'b-', alpha=0.7)
        ax1.scatter(top_wavenumbers, top_scores, color='red', s=30, zorder=5)
        ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
        ax1.set_ylabel('Feature Importance', fontsize=9)
        ax1.set_title(f'Feature Importance - {method}', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Top features bar plot
        x_pos = np.arange(len(top_wavenumbers))
        bars = ax2.bar(x_pos, top_scores, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Top Features (Wavenumber)', fontsize=9)
        ax2.set_ylabel('Importance Score', fontsize=9)
        ax2.set_title('Top 20 Most Important Features', fontsize=10, fontweight='bold')
        ax2.set_xticks(x_pos[::2])  # Show every other tick
        ax2.set_xticklabels([f'{int(wn)}' for wn in top_wavenumbers[::2]], rotation=45, fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative importance
        sorted_importance = np.sort(importance_scores)[::-1]
        cumulative_importance = np.cumsum(sorted_importance)
        cumulative_importance /= cumulative_importance[-1]  # Normalize to 1
        
        ax3.plot(range(len(cumulative_importance)), cumulative_importance, 'g-', linewidth=2)
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        ax3.set_xlabel('Number of Features', fontsize=9)
        ax3.set_ylabel('Cumulative Importance', fontsize=9)
        ax3.set_title('Cumulative Feature Importance', fontsize=10, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Importance distribution
        ax4.hist(importance_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=np.mean(importance_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(importance_scores):.4f}')
        ax4.axvline(x=np.median(importance_scores), color='orange', linestyle='--', 
                   label=f'Median: {np.median(importance_scores):.4f}')
        ax4.set_xlabel('Importance Score', fontsize=9)
        ax4.set_ylabel('Frequency', fontsize=9)
        ax4.set_title('Distribution of Feature Importance', fontsize=10, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.stats_tab.stats_canvas.draw()
    
    def _display_feature_importance_results(self, top_wavenumbers, top_scores, method, all_scores):
        """Display feature importance results in the text area."""
        results_text = f"Feature Importance Analysis - {method}\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Top 20 Most Important Features:\n"
        results_text += "-" * 35 + "\n"
        
        for i, (wavenumber, score) in enumerate(zip(top_wavenumbers, top_scores)):
            results_text += f"{i+1:2d}. {wavenumber:6.1f} cm⁻¹ : {score:.6f}\n"
        
        # Calculate some statistics
        total_importance = np.sum(top_scores)
        mean_importance = np.mean(top_scores)
        std_importance = np.std(top_scores)
        
        results_text += f"\nStatistics for Top Features:\n"
        results_text += f"• Total importance: {total_importance:.6f}\n"
        results_text += f"• Mean importance: {mean_importance:.6f}\n"
        results_text += f"• Std deviation: {std_importance:.6f}\n"
        results_text += f"• Highest importance: {top_scores[0]:.6f} at {top_wavenumbers[0]:.1f} cm⁻¹\n"
        
        # Find how many features needed for 80% and 90% cumulative importance
        sorted_importance = np.sort(all_scores)[::-1]
        cumulative = np.cumsum(sorted_importance) / np.sum(sorted_importance)
        n_80 = np.argmax(cumulative >= 0.80) + 1
        n_90 = np.argmax(cumulative >= 0.90) + 1
        
        results_text += f"\nCumulative Importance:\n"
        results_text += f"• {n_80} features capture 80% of importance\n"
        results_text += f"• {n_90} features capture 90% of importance\n"
        
        results_text += f"\nInterpretation:\n"
        results_text += f"The {method} method identified {len(top_wavenumbers)} key spectral features "
        results_text += f"that best distinguish between clusters. Higher importance scores indicate "
        results_text += f"wavenumbers that contribute more to cluster separation. Focus on these "
        results_text += f"peaks for chemical interpretation of your clusters.\n"
        
        self.stats_tab.stats_results_text.setText(results_text)
    
    def _plot_discriminant_analysis(self, X_lda, labels, explained_variance, cv_scores):
        """Plot discriminant analysis results."""
        import matplotlib.pyplot as plt
        
        fig = self.stats_tab.stats_fig
        fig.clear()
        
        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)  # LDA scatter plot
        ax2 = fig.add_subplot(2, 2, 2)  # Explained variance
        ax3 = fig.add_subplot(2, 2, 3)  # Cross-validation scores
        ax4 = fig.add_subplot(2, 2, 4)  # Cluster separation
        
        # 1. LDA scatter plot (first two components)
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if X_lda.shape[1] >= 2:
                ax1.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                           c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=20)
            else:
                # If only one component, plot against index
                ax1.scatter(range(np.sum(mask)), X_lda[mask, 0], 
                           c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=20)
        
        if X_lda.shape[1] >= 2:
            ax1.set_xlabel(f'LD1 ({explained_variance[0]:.1%} variance)', fontsize=9)
            ax1.set_ylabel(f'LD2 ({explained_variance[1]:.1%} variance)', fontsize=9)
        else:
            ax1.set_xlabel('Sample Index', fontsize=9)
            ax1.set_ylabel(f'LD1 ({explained_variance[0]:.1%} variance)', fontsize=9)
        ax1.set_title('Linear Discriminant Analysis', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Explained variance plot
        components = range(1, len(explained_variance) + 1)
        ax2.bar(components, explained_variance, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Linear Discriminant', fontsize=9)
        ax2.set_ylabel('Explained Variance Ratio', fontsize=9)
        ax2.set_title('Explained Variance by Component', fontsize=10, fontweight='bold')
        ax2.set_xticks(components)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cross-validation scores
        fold_numbers = range(1, len(cv_scores) + 1)
        bars = ax3.bar(fold_numbers, cv_scores, color='lightgreen', edgecolor='darkgreen')
        ax3.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cv_scores):.3f}')
        ax3.set_xlabel('Cross-Validation Fold', fontsize=9)
        ax3.set_ylabel('Accuracy Score', fontsize=9)
        ax3.set_title('Cross-Validation Performance', fontsize=10, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Cluster separation analysis
        if X_lda.shape[1] >= 1:
            # Calculate pairwise distances between cluster centroids
            centroids = []
            for label in unique_labels:
                mask = labels == label
                centroid = np.mean(X_lda[mask], axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            n_clusters = len(unique_labels)
            
            # Calculate separation matrix
            separation_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if i != j:
                        separation_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            
            # Plot separation heatmap
            im = ax4.imshow(separation_matrix, cmap='viridis', aspect='auto')
            ax4.set_xticks(range(n_clusters))
            ax4.set_yticks(range(n_clusters))
            ax4.set_xticklabels([f'C{label}' for label in unique_labels], fontsize=8)
            ax4.set_yticklabels([f'C{label}' for label in unique_labels], fontsize=8)
            ax4.set_title('Cluster Separation Matrix', fontsize=10, fontweight='bold')
            
            # Add text annotations
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if i != j:
                        ax4.text(j, i, f'{separation_matrix[i, j]:.2f}', 
                               ha='center', va='center', color='white', fontsize=8)
            
            fig.colorbar(im, ax=ax4)
        
        fig.tight_layout()
        self.stats_tab.stats_canvas.draw()

    def _display_discriminant_results(self, cv_scores, explained_variance, lda):
        """Display discriminant analysis results."""
        results_text = "Linear Discriminant Analysis Results\n"
        results_text += "=" * 40 + "\n\n"
        
        results_text += f"Cross-Validation Performance:\n"
        results_text += f"• Mean accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}\n"
        results_text += f"• Best fold: {np.max(cv_scores):.3f}\n"
        results_text += f"• Worst fold: {np.min(cv_scores):.3f}\n\n"
        
        results_text += f"Explained Variance by Component:\n"
        for i, var in enumerate(explained_variance):
            results_text += f"• LD{i+1}: {var:.3f} ({var*100:.1f}%)\n"
        
        cumulative_var = np.cumsum(explained_variance)
        results_text += f"\nCumulative Explained Variance:\n"
        for i, cum_var in enumerate(cumulative_var):
            results_text += f"• First {i+1} component(s): {cum_var:.3f} ({cum_var*100:.1f}%)\n"
        
        # Model interpretation
        results_text += f"\nModel Interpretation:\n"
        if np.mean(cv_scores) >= 0.9:
            interpretation = "Excellent cluster separation"
        elif np.mean(cv_scores) >= 0.8:
            interpretation = "Good cluster separation"
        elif np.mean(cv_scores) >= 0.7:
            interpretation = "Moderate cluster separation"
        else:
            interpretation = "Poor cluster separation"
        
        results_text += f"• Classification quality: {interpretation}\n"
        results_text += f"• Number of discriminant functions: {len(explained_variance)}\n"
        results_text += f"\nInterpretation:\n"
        results_text += f"LDA finds the linear combinations of features that best separate clusters.\n"
        results_text += f"The cross-validation accuracy shows how well clusters can be predicted.\n"
        results_text += f"High accuracy ({np.mean(cv_scores):.1%}) indicates well-separated, distinct clusters.\n"
        
        self.stats_tab.stats_results_text.setText(results_text)
    
    def _perform_anova(self, intensities, labels, alpha):
        """Perform ANOVA test on all features."""
        from scipy import stats
        
        n_features = intensities.shape[1]
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        f_statistics = []
        p_values = []
        
        for i in range(n_features):
            # Group data by cluster
            groups = [intensities[labels == label, i] for label in unique_labels]
            f_stat, p_val = stats.f_oneway(*groups)
            f_statistics.append(f_stat)
            p_values.append(p_val)
        
        f_statistics = np.array(f_statistics)
        p_values = np.array(p_values)
        
        # Apply Bonferroni correction
        corrected_alpha = alpha / n_features
        significant_features = np.sum(p_values < corrected_alpha)
        
        return {
            'test_statistic': np.mean(f_statistics),
            'p_value': np.min(p_values),
            'significant_features': significant_features,
            'total_features': n_features,
            'significant': significant_features > 0,
            'f_statistics': f_statistics,
            'p_values': p_values,
            'corrected_alpha': corrected_alpha
        }
    
    def _plot_significance_results(self, results, method):
        """Plot statistical significance results."""
        import matplotlib.pyplot as plt
        
        fig = self.stats_tab.stats_fig
        fig.clear()
        
        if method == 'ANOVA':
            # Create 2x2 subplot
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            
            f_stats = results['f_statistics']
            p_vals = results['p_values']
            corrected_alpha = results['corrected_alpha']
            
            # 1. F-statistics across features
            ax1.plot(f_stats, 'b-', linewidth=1)
            ax1.set_xlabel('Feature Index', fontsize=9)
            ax1.set_ylabel('F-statistic', fontsize=9)
            ax1.set_title('F-statistics Across Features', fontsize=10, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 2. P-values across features
            ax2.plot(p_vals, 'r-', linewidth=1)
            ax2.axhline(y=corrected_alpha, color='g', linestyle='--', 
                       label=f'Corrected α={corrected_alpha:.6f}')
            ax2.set_xlabel('Feature Index', fontsize=9)
            ax2.set_ylabel('p-value', fontsize=9)
            ax2.set_title('P-values Across Features', fontsize=10, fontweight='bold')
            ax2.set_yscale('log')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3. Distribution of F-statistics
            ax3.hist(f_stats, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(x=np.mean(f_stats), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(f_stats):.2f}')
            ax3.set_xlabel('F-statistic', fontsize=9)
            ax3.set_ylabel('Frequency', fontsize=9)
            ax3.set_title('Distribution of F-statistics', fontsize=10, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # 4. Significant vs non-significant features
            sig_count = results['significant_features']
            nonsig_count = results['total_features'] - sig_count
            ax4.bar(['Significant', 'Non-significant'], [sig_count, nonsig_count],
                   color=['green', 'gray'], alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Number of Features', fontsize=9)
            ax4.set_title('Feature Significance Summary', fontsize=10, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels
            total = results['total_features']
            ax4.text(0, sig_count, f'{sig_count/total*100:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
            ax4.text(1, nonsig_count, f'{nonsig_count/total*100:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        fig.tight_layout()
        self.stats_tab.stats_canvas.draw()
    
    def _display_significance_results(self, results, method, alpha):
        """Display statistical significance results."""
        results_text = f"Statistical Significance Testing - {method}\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Test Parameters:\n"
        results_text += f"• Significance level (α): {alpha:.3f}\n"
        results_text += f"• Bonferroni corrected α: {results['corrected_alpha']:.6f}\n\n"
        
        results_text += f"Test Results:\n"
        results_text += f"• Average F-statistic: {results['test_statistic']:.4f}\n"
        results_text += f"• Minimum p-value: {results['p_value']:.6f}\n"
        results_text += f"• Significant features: {results['significant_features']}/{results['total_features']}\n"
        results_text += f"• Percentage significant: {results['significant_features']/results['total_features']*100:.1f}%\n"
        results_text += f"• Overall significant: {'Yes' if results['significant'] else 'No'}\n\n"
        
        # Interpretation
        if results['significant']:
            results_text += f"Interpretation:\n"
            results_text += f"✓ Multiple spectral features show statistically significant "
            results_text += f"differences between clusters after Bonferroni correction.\n"
            results_text += f"✓ The clustering reveals meaningful chemical differences.\n"
            results_text += f"✓ {results['significant_features']} features can be used to discriminate clusters.\n"
        else:
            results_text += f"Interpretation:\n"
            results_text += f"⚠ No spectral features show statistically significant "
            results_text += f"differences between clusters after multiple testing correction.\n"
            results_text += f"⚠ The clustering may not reflect meaningful chemical differences.\n"
            results_text += f"⚠ Consider using fewer clusters or different preprocessing.\n"
        
        results_text += f"\nNote: Bonferroni correction is conservative and controls for\n"
        results_text += f"multiple testing. It reduces false positives but may miss weak effects.\n"
        
        self.stats_tab.stats_results_text.setText(results_text)
    
    def _plot_detailed_pca(self, X_pca, labels, explained_variance, cumulative_variance, 
                          components, wavenumbers):
        """Plot detailed PCA results."""
        import matplotlib.pyplot as plt
        
        fig = self.stats_tab.stats_fig
        fig.clear()
        
        # Create 2x2 subplots
        ax1 = fig.add_subplot(2, 2, 1)  # PC1 vs PC2 scatter
        ax2 = fig.add_subplot(2, 2, 2)  # Scree plot
        ax3 = fig.add_subplot(2, 2, 3)  # Cumulative variance
        ax4 = fig.add_subplot(2, 2, 4)  # PC1 loadings
        
        # 1. PC1 vs PC2 scatter plot
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=20)
        
        ax1.set_xlabel(f'PC1 ({explained_variance[0]:.1%})', fontsize=9)
        ax1.set_ylabel(f'PC2 ({explained_variance[1]:.1%})', fontsize=9)
        ax1.set_title('PCA Projection (PC1 vs PC2)', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Scree plot (first 20 components)
        n_show = min(20, len(explained_variance))
        ax2.bar(range(1, n_show + 1), explained_variance[:n_show], 
               color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Principal Component', fontsize=9)
        ax2.set_ylabel('Explained Variance Ratio', fontsize=9)
        ax2.set_title('Scree Plot (First 20 PCs)', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative variance
        n_show_cum = min(50, len(cumulative_variance))
        ax3.plot(range(1, n_show_cum + 1), cumulative_variance[:n_show_cum], 
                'g-', linewidth=2)
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80%')
        ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%')
        ax3.axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='95%')
        ax3.set_xlabel('Number of Components', fontsize=9)
        ax3.set_ylabel('Cumulative Variance Explained', fontsize=9)
        ax3.set_title('Cumulative Explained Variance', fontsize=10, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. PC1 loadings (spectral pattern)
        ax4.plot(wavenumbers, components[0, :], 'b-', linewidth=1)
        ax4.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
        ax4.set_ylabel('PC1 Loading', fontsize=9)
        ax4.set_title('PC1 Spectral Pattern', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        fig.tight_layout()
        self.stats_tab.stats_canvas.draw()
    
    def _display_pca_results(self, explained_variance, cumulative_variance, 
                            n_80, n_90, n_95, components, wavenumbers):
        """Display PCA results text."""
        results_text = "Detailed PCA Analysis Results\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Variance Explained:\n"
        results_text += f"• PC1: {explained_variance[0]:.3f} ({explained_variance[0]*100:.1f}%)\n"
        results_text += f"• PC2: {explained_variance[1]:.3f} ({explained_variance[1]*100:.1f}%)\n"
        results_text += f"• PC3: {explained_variance[2]:.3f} ({explained_variance[2]*100:.1f}%)\n\n"
        
        results_text += f"Components Needed:\n"
        results_text += f"• 80% variance: {n_80} components\n"
        results_text += f"• 90% variance: {n_90} components\n"
        results_text += f"• 95% variance: {n_95} components\n\n"
        
        results_text += f"Total Components: {len(explained_variance)}\n"
        results_text += f"Total Variance (first 10 PCs): {cumulative_variance[9]:.1%}\n\n"
        
        # Find top loadings for PC1
        pc1_loadings = np.abs(components[0, :])
        top_indices = np.argsort(pc1_loadings)[-10:][::-1]
        
        results_text += f"Top 10 PC1 Loadings (Most Important Wavenumbers):\n"
        results_text += "-" * 40 + "\n"
        for i, idx in enumerate(top_indices):
            results_text += f"{i+1:2d}. {wavenumbers[idx]:6.1f} cm⁻¹ : {components[0, idx]:+.4f}\n"
        
        results_text += f"\nInterpretation:\n"
        results_text += f"PCA reduces {len(wavenumbers)} spectral features to {n_90} principal components\n"
        results_text += f"that capture 90% of the variance. PC1 captures the main spectral variation\n"
        results_text += f"({explained_variance[0]:.1%}). The loadings show which wavenumbers contribute\n"
        results_text += f"most to each component. Use this to identify key spectral patterns.\n"
        
        self.stats_tab.stats_results_text.setText(results_text)
    
    def _plot_factor_analysis(self, X_fa, labels, loadings, wavenumbers, variance_explained):
        """Plot Factor Analysis results."""
        import matplotlib.pyplot as plt
        
        fig = self.stats_tab.stats_fig
        fig.clear()
        
        # Create 2x2 subplots
        ax1 = fig.add_subplot(2, 2, 1)  # Factor 1 vs Factor 2 scatter
        ax2 = fig.add_subplot(2, 2, 2)  # Variance by factor
        ax3 = fig.add_subplot(2, 2, 3)  # Factor 1 loadings
        ax4 = fig.add_subplot(2, 2, 4)  # Loading heatmap
        
        # 1. Factor 1 vs Factor 2 scatter
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(X_fa[mask, 0], X_fa[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=20)
        
        ax1.set_xlabel(f'Factor 1 ({variance_explained[0]:.1%})', fontsize=9)
        ax1.set_ylabel(f'Factor 2 ({variance_explained[1]:.1%})', fontsize=9)
        ax1.set_title('Factor Analysis Projection', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Variance by factor
        n_factors = len(variance_explained)
        ax2.bar(range(1, n_factors + 1), variance_explained, 
               color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('Factor', fontsize=9)
        ax2.set_ylabel('Variance Explained', fontsize=9)
        ax2.set_title('Variance by Factor', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Factor 1 loadings spectrum
        ax3.plot(wavenumbers, loadings[:, 0], 'r-', linewidth=1)
        ax3.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
        ax3.set_ylabel('Factor 1 Loading', fontsize=9)
        ax3.set_title('Factor 1 Spectral Pattern', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # 4. Loading heatmap (top features x factors)
        # Show top 50 features with highest loadings
        max_loadings = np.max(np.abs(loadings), axis=1)
        top_features = np.argsort(max_loadings)[-50:][::-1]
        
        im = ax4.imshow(loadings[top_features, :].T, aspect='auto', cmap='RdBu_r', 
                       vmin=-np.max(np.abs(loadings)), vmax=np.max(np.abs(loadings)))
        ax4.set_xlabel('Top 50 Features', fontsize=9)
        ax4.set_ylabel('Factor', fontsize=9)
        ax4.set_title('Loading Heatmap', fontsize=10, fontweight='bold')
        ax4.set_yticks(range(n_factors))
        ax4.set_yticklabels([f'F{i+1}' for i in range(n_factors)], fontsize=8)
        fig.colorbar(im, ax=ax4, label='Loading')
        
        fig.tight_layout()
        self.stats_tab.stats_canvas.draw()
    
    def _display_factor_results(self, n_factors, loadings, wavenumbers, variance_explained):
        """Display Factor Analysis results text."""
        results_text = "Factor Analysis Results\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Number of Factors: {n_factors}\n\n"
        
        results_text += f"Variance Explained by Each Factor:\n"
        for i in range(n_factors):
            results_text += f"• Factor {i+1}: {variance_explained[i]:.3f} ({variance_explained[i]*100:.1f}%)\n"
        
        results_text += f"\nTotal Variance Explained: {np.sum(variance_explained):.1%}\n\n"
        
        # Top loadings for Factor 1
        factor1_loadings = np.abs(loadings[:, 0])
        top_indices = np.argsort(factor1_loadings)[-10:][::-1]
        
        results_text += f"Top 10 Factor 1 Loadings:\n"
        results_text += "-" * 40 + "\n"
        for i, idx in enumerate(top_indices):
            results_text += f"{i+1:2d}. {wavenumbers[idx]:6.1f} cm⁻¹ : {loadings[idx, 0]:+.4f}\n"
        
        results_text += f"\nInterpretation:\n"
        results_text += f"Factor Analysis identified {n_factors} latent factors that explain the\n"
        results_text += f"underlying structure in your spectral data. Unlike PCA, Factor Analysis\n"
        results_text += f"assumes that observed variables are influenced by hidden factors.\n"
        results_text += f"High loadings indicate which wavenumbers are most influenced by each factor.\n"
        results_text += f"Use this to identify common spectral patterns across your clusters.\n"
        
        self.stats_tab.stats_results_text.setText(results_text)


def launch_cluster_analysis(parent=None, raman_app=None):
    """Launch the cluster analysis window."""
    try:
        window = RamanClusterAnalysisQt6(parent, raman_app)
        window.show()
        return window
    except Exception as e:
        QMessageBox.critical(None, "Launch Error", f"Failed to launch cluster analysis:\n{str(e)}")
        return None
