#!/usr/bin/env python3
"""
RamanLab Qt6 Version - Raman Cluster Analysis
Raman Cluster Analysis GUI with Advanced Ion Exchange Analysis
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist
import glob
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import csv

# Fix matplotlib backend for Qt6/PySide6
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure

# Import Qt6-compatible matplotlib backends
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    # Fallback for older matplotlib versions
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar

# Qt6 imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QSpinBox, QDoubleSpinBox, QFrame, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QFormLayout, QDialogButtonBox,
    QListWidget, QListWidgetItem, QInputDialog, QApplication, QGridLayout,
    QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QAction, QFont, QPixmap

# Try to import UMAP for advanced visualization
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with 'pip install umap-learn' for additional visualization options.")


class DatabaseImportDialog(QDialog):
    """Dialog for selecting spectra from the database for import into cluster analysis."""
    
    def __init__(self, raman_db, parent=None):
        """Initialize the database import dialog."""
        super().__init__(parent)
        self.raman_db = raman_db
        self.selected_spectra = {}
        
        self.setWindowTitle("Import Spectra from Database")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        self.setup_ui()
        self.load_database_spectra()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Search controls
        search_group = QGroupBox("Search & Filter")
        search_group_layout = QVBoxLayout(search_group)
        
        # Text search
        text_search_layout = QHBoxLayout()
        text_search_layout.addWidget(QLabel("Text Search:"))
        self.search_entry = QLineEdit()
        self.search_entry.setPlaceholderText("Search by name, mineral, formula, description...")
        self.search_entry.textChanged.connect(self.filter_spectra)
        text_search_layout.addWidget(self.search_entry)
        
        clear_search_btn = QPushButton("Clear")
        clear_search_btn.clicked.connect(self.clear_search)
        text_search_layout.addWidget(clear_search_btn)
        
        search_group_layout.addLayout(text_search_layout)
        
        # Filter dropdowns
        filter_layout = QHBoxLayout()
        
        # Hey Classification filter
        filter_layout.addWidget(QLabel("Hey Classification:"))
        self.hey_filter = QComboBox()
        self.hey_filter.setEditable(True)
        self.hey_filter.addItem("")  # Empty option for "All"
        self.hey_filter.currentTextChanged.connect(self.filter_spectra)
        filter_layout.addWidget(self.hey_filter)
        
        # Chemical Family filter (anion types)
        filter_layout.addWidget(QLabel("Chemical Family:"))
        self.family_filter = QComboBox()
        self.family_filter.setEditable(True)
        self.family_filter.addItem("")  # Empty option for "All"
        self.family_filter.currentTextChanged.connect(self.filter_spectra)
        filter_layout.addWidget(self.family_filter)
        
        search_group_layout.addLayout(filter_layout)
        layout.addWidget(search_group)
        
        # Spectra table
        self.spectra_table = QTableWidget()
        self.spectra_table.setColumnCount(8)
        self.spectra_table.setHorizontalHeaderLabels([
            "Select", "Name", "Mineral", "Formula", "Hey Classification", "Chemical Family", "Data Points", "Description"
        ])
        
        # Set column widths
        header = self.spectra_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.resizeSection(0, 60)   # Select column
        header.resizeSection(1, 150)  # Name column
        header.resizeSection(2, 120)  # Mineral column
        header.resizeSection(3, 100)  # Formula column
        header.resizeSection(4, 150)  # Hey Classification column
        header.resizeSection(5, 120)  # Chemical Family column
        header.resizeSection(6, 80)   # Data Points column
        
        layout.addWidget(self.spectra_table)
        
        # Selection controls
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        selection_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self.select_none)
        selection_layout.addWidget(select_none_btn)
        
        selection_layout.addStretch()
        
        self.selection_label = QLabel("0 spectra selected")
        selection_layout.addWidget(self.selection_label)
        
        layout.addLayout(selection_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        import_btn = QPushButton("Import Selected")
        import_btn.clicked.connect(self.accept_selection)
        import_btn.setDefault(True)
        button_layout.addWidget(import_btn)
        
        layout.addLayout(button_layout)
    
    def load_database_spectra(self):
        """Load spectra from the database into the table."""
        try:
            if not hasattr(self.raman_db, 'database') or not self.raman_db.database:
                self.spectra_table.setRowCount(1)
                self.spectra_table.setItem(0, 1, QTableWidgetItem("No spectra found in database"))
                return
            
            spectra_list = []
            for name, spectrum_data in self.raman_db.database.items():
                try:
                    # Extract metadata - handle both nested metadata and direct structure
                    metadata = spectrum_data.get('metadata', spectrum_data)
                    
                    # Extract mineral name with multiple fallbacks
                    mineral_name = (metadata.get('NAME') or 
                                   metadata.get('name') or 
                                   metadata.get('mineral_name') or 
                                   name or 'Unknown')
                    
                    # Extract formula with comprehensive fallback options, including Ideal Chemistry
                    formula = (metadata.get('FORMULA') or 
                              metadata.get('Formula') or 
                              metadata.get('formula') or 
                              metadata.get('CHEMICAL_FORMULA') or 
                              metadata.get('Chemical_Formula') or 
                              metadata.get('chemical_formula') or 
                              metadata.get('IDEAL CHEMISTRY') or
                              metadata.get('Ideal Chemistry') or
                              metadata.get('ideal_chemistry') or
                              metadata.get('IDEAL_CHEMISTRY') or
                              metadata.get('composition') or 
                              spectrum_data.get('formula', ''))
                    
                    # Extract description with multiple fallbacks
                    description = (metadata.get('DESCRIPTION') or 
                                  metadata.get('Description') or 
                                  metadata.get('description') or 
                                  metadata.get('desc') or '')
                    
                    # Hey classification with comprehensive fallbacks
                    hey_class = (metadata.get('HEY CLASSIFICATION') or 
                               metadata.get('Hey Classification') or 
                               metadata.get('hey_classification') or 
                               metadata.get('HEY_CLASSIFICATION') or
                               metadata.get('classification') or
                               metadata.get('mineral_class') or
                               metadata.get('MINERAL_CLASS') or '')
                    
                    # Chemical family (anion type) with comprehensive fallbacks
                    chemical_family = (metadata.get('CHEMICAL FAMILY') or 
                                     metadata.get('Chemical Family') or 
                                     metadata.get('chemical_family') or 
                                     metadata.get('CHEMICAL_FAMILY') or
                                     metadata.get('family') or
                                     metadata.get('anion_group') or
                                     metadata.get('ANION_GROUP') or '')
                    
                    # Get data points count
                    wavenumbers = spectrum_data.get('wavenumbers', [])
                    data_points = len(wavenumbers) if wavenumbers is not None else 0
                    
                    spectra_list.append({
                        'name': name,
                        'mineral': mineral_name,
                        'formula': formula,
                        'hey_classification': hey_class,
                        'chemical_family': chemical_family,
                        'description': description,
                        'data_points': data_points,
                        'spectrum_data': spectrum_data
                    })
                    
                except Exception as e:
                    print(f"Error processing spectrum {name}: {str(e)}")
                    continue
            
            # Sort by name
            spectra_list.sort(key=lambda x: x['name'])
            
            # Populate table
            self.spectra_table.setRowCount(len(spectra_list))
            self.all_spectra = spectra_list  # Store for filtering
            
            # Collect unique values for filters
            hey_classifications = set()
            chemical_families = set()
            
            for row, spectrum_info in enumerate(spectra_list):
                # Collect filter values
                if spectrum_info['hey_classification']:
                    hey_classifications.add(spectrum_info['hey_classification'])
                if spectrum_info['chemical_family']:
                    chemical_families.add(spectrum_info['chemical_family'])
                
                # Checkbox for selection
                checkbox = QCheckBox()
                checkbox.stateChanged.connect(self.update_selection_count)
                self.spectra_table.setCellWidget(row, 0, checkbox)
                
                # Spectrum information
                self.spectra_table.setItem(row, 1, QTableWidgetItem(spectrum_info['name']))
                self.spectra_table.setItem(row, 2, QTableWidgetItem(spectrum_info['mineral']))
                self.spectra_table.setItem(row, 3, QTableWidgetItem(spectrum_info['formula']))
                self.spectra_table.setItem(row, 4, QTableWidgetItem(spectrum_info['hey_classification']))
                self.spectra_table.setItem(row, 5, QTableWidgetItem(spectrum_info['chemical_family']))
                self.spectra_table.setItem(row, 6, QTableWidgetItem(str(spectrum_info['data_points'])))
                self.spectra_table.setItem(row, 7, QTableWidgetItem(spectrum_info['description']))
                
                # Store spectrum data in the table item
                self.spectra_table.item(row, 1).setData(Qt.UserRole, spectrum_info['spectrum_data'])
            
            # Populate filter dropdowns with improved organization
            self._populate_hey_filter(hey_classifications)
            self._populate_family_filter(chemical_families)
            
            self.update_selection_count()
            
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to load database spectra:\n{str(e)}")
    
    def filter_spectra(self):
        """Filter the spectra table based on search text and dropdown filters."""
        search_text = self.search_entry.text().lower()
        hey_filter = self.hey_filter.currentText().strip()
        family_filter = self.family_filter.currentText().strip()
        
        for row in range(self.spectra_table.rowCount()):
            should_show = True
            
            # Text search filter
            if search_text:
                text_match = False
                # Check all text columns
                for col in range(1, self.spectra_table.columnCount()):
                    item = self.spectra_table.item(row, col)
                    if item and search_text in item.text().lower():
                        text_match = True
                        break
                if not text_match:
                    should_show = False
            
            # Hey classification filter - improved selective matching
            if should_show and hey_filter:
                hey_item = self.spectra_table.item(row, 4)
                if not hey_item:
                    should_show = False
                else:
                    should_show = self._matches_hey_classification(hey_item.text(), hey_filter)
            
            # Chemical family filter - improved selective matching
            if should_show and family_filter:
                family_item = self.spectra_table.item(row, 5)
                if not family_item:
                    should_show = False
                else:
                    should_show = self._matches_chemical_family(family_item.text(), family_filter)
            
            self.spectra_table.setRowHidden(row, not should_show)
    
    def _matches_hey_classification(self, classification_text, filter_text):
        """
        Strict Hey classification matching logic.
        
        Returns True if the classification matches the filter criteria.
        For pure classifications (e.g., "Borates"), only matches exact "Borates", 
        not "Borates with other anions".
        """
        if not classification_text or not filter_text:
            return False
        
        classification_lower = classification_text.lower().strip()
        filter_lower = filter_text.lower().strip()
        
        # Handle compound filter option (remove [Compound] prefix)
        if filter_lower.startswith('[compound] '):
            compound_filter = filter_lower[11:]  # Remove '[compound] ' prefix
            # For compound filters, allow exact match of the full compound string
            return classification_lower == compound_filter
        
        # Exact match first (highest priority)
        if classification_lower == filter_lower:
            return True
        
        # For strict filtering, only allow exact matches of individual components
        # Split classification on common delimiters
        class_parts = []
        for delimiter in [',', ' or ', ' and ', ';', '/', '|']:
            if delimiter in classification_text:
                class_parts = [part.strip() for part in classification_text.split(delimiter)]
                break
        
        if not class_parts:
            class_parts = [classification_text.strip()]
        
        # Check if any part is an EXACT match to the filter
        for part in class_parts:
            part_lower = part.lower().strip()
            if part_lower == filter_lower:
                return True
        
        return False
    
    def _matches_chemical_family(self, family_text, filter_text):
        """
        Strict chemical family matching logic.
        
        Returns True if the family matches the filter criteria.
        For pure families (e.g., "Borates"), only matches exact "Borates", 
        not "Borates with other anions".
        """
        if not family_text or not filter_text:
            return False
        
        family_lower = family_text.lower().strip()
        filter_lower = filter_text.lower().strip()
        
        # Handle compound filter option (remove [Compound] prefix)
        if filter_lower.startswith('[compound] '):
            compound_filter = filter_lower[11:]  # Remove '[compound] ' prefix
            # For compound filters, allow exact match of the full compound string
            return family_lower == compound_filter
        
        # Exact match first
        if family_lower == filter_lower:
            return True
        
        # For strict filtering, only allow exact matches of individual components
        # Split family on common delimiters
        family_parts = []
        for delimiter in [',', ' or ', ' and ', ';', '/', '|', ' - ']:
            if delimiter in family_text:
                family_parts = [part.strip() for part in family_text.split(delimiter)]
                break
        
        if not family_parts:
            family_parts = [family_text.strip()]
        
        # Check if any part is an EXACT match to the filter
        for part in family_parts:
            part_lower = part.lower().strip()
            if part_lower == filter_lower:
                return True
        
        return False
    
    def _populate_hey_filter(self, hey_classifications):
        """
        Populate Hey classification filter with organized options.
        
        Separates simple classifications from compound ones and provides
        individual classification options for better filtering.
        """
        simple_classes = set()
        compound_classes = set()
        all_individual_classes = set()
        
        for classification in hey_classifications:
            if not classification:
                continue
                
            # Check if it's a compound classification
            is_compound = any(delimiter in classification for delimiter in [',', ' or ', ' and ', ';', '/', '|'])
            
            if is_compound:
                compound_classes.add(classification)
                # Extract individual components
                for delimiter in [',', ' or ', ' and ', ';', '/', '|']:
                    if delimiter in classification:
                        parts = [part.strip() for part in classification.split(delimiter)]
                        for part in parts:
                            # Clean up the part (remove trailing descriptors)
                            clean_part = part.split(' with ')[0].split(' - ')[0].strip()
                            if clean_part and len(clean_part) > 2:  # Avoid very short fragments
                                all_individual_classes.add(clean_part)
                        break
            else:
                simple_classes.add(classification)
                all_individual_classes.add(classification)
        
        # Combine and sort options
        filter_options = sorted(all_individual_classes)
        
        # Add compound classifications at the end if they provide unique information
        for compound in sorted(compound_classes):
            if compound not in filter_options:
                filter_options.append(f"[Compound] {compound}")
        
        self.hey_filter.addItems(filter_options)
    
    def _populate_family_filter(self, chemical_families):
        """
        Populate chemical family filter with organized options.
        
        Similar to Hey classification but for chemical families.
        """
        simple_families = set()
        compound_families = set()
        all_individual_families = set()
        
        for family in chemical_families:
            if not family:
                continue
                
            # Check if it's a compound family
            is_compound = any(delimiter in family for delimiter in [',', ' or ', ' and ', ';', '/', '|', ' - '])
            
            if is_compound:
                compound_families.add(family)
                # Extract individual components
                for delimiter in [',', ' or ', ' and ', ';', '/', '|', ' - ']:
                    if delimiter in family:
                        parts = [part.strip() for part in family.split(delimiter)]
                        for part in parts:
                            # Clean up the part
                            clean_part = part.split(' with ')[0].split(' - ')[0].strip()
                            if clean_part and len(clean_part) > 2:
                                all_individual_families.add(clean_part)
                        break
            else:
                simple_families.add(family)
                all_individual_families.add(family)
        
        # Combine and sort options
        filter_options = sorted(all_individual_families)
        
        # Add compound families at the end
        for compound in sorted(compound_families):
            if compound not in filter_options:
                filter_options.append(f"[Compound] {compound}")
        
        self.family_filter.addItems(filter_options)
    
    def clear_search(self):
        """Clear all search filters and show all spectra."""
        self.search_entry.clear()
        self.hey_filter.setCurrentText("")
        self.family_filter.setCurrentText("")
    
    def select_all(self):
        """Select all visible spectra."""
        for row in range(self.spectra_table.rowCount()):
            if not self.spectra_table.isRowHidden(row):
                checkbox = self.spectra_table.cellWidget(row, 0)
                if checkbox:
                    checkbox.setChecked(True)
    
    def select_none(self):
        """Deselect all spectra."""
        for row in range(self.spectra_table.rowCount()):
            checkbox = self.spectra_table.cellWidget(row, 0)
            if checkbox:
                checkbox.setChecked(False)
    
    def update_selection_count(self):
        """Update the selection count label."""
        count = 0
        for row in range(self.spectra_table.rowCount()):
            checkbox = self.spectra_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                count += 1
        
        self.selection_label.setText(f"{count} spectra selected")
    
    def accept_selection(self):
        """Accept the current selection and close the dialog."""
        self.selected_spectra = {}
        
        for row in range(self.spectra_table.rowCount()):
            checkbox = self.spectra_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                name_item = self.spectra_table.item(row, 1)
                if name_item:
                    spectrum_name = name_item.text()
                    spectrum_data = name_item.data(Qt.UserRole)
                    if spectrum_data:
                        self.selected_spectra[spectrum_name] = spectrum_data
        
        if not self.selected_spectra:
            QMessageBox.warning(self, "No Selection", "Please select at least one spectrum to import.")
            return
        
        self.accept()
    
    def get_selected_spectra(self):
        """Return the selected spectra data."""
        return self.selected_spectra


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
        
        # Store reference to main app
        self.raman_app = raman_app
        
        # Initialize database path
        self.custom_db_path = None
        
        # Initialize CSV Hey classifications cache
        self._csv_hey_classes = None
        
        # Initialize variables
        self.selected_folder = None
        self.visualization_method = "PCA"
        self.selected_points = set()
        self.refinement_mode = False
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
            'temporal_data': None,  # For time-series analysis
            'composition_data': None,  # For composition tracking
            'umap_embedding': None,  # Store UMAP results
            'silhouette_scores': None,  # Store silhouette analysis
            # Probabilistic clustering
            'cluster_probs': None,  # Cluster probabilities from GMM
            'gmm': None,  # Fitted GMM model
            'subtypes': None  # Sub-type information
        }
        
        # Initialize advanced analysis data
        self.analysis_results = {
            'cluster_centroids': None,
            'differential_spectra': None,
            'kinetic_models': None,
            'statistical_tests': None,
            'feature_importance': None
        }
        
        # Setup UI
        self.setup_ui()
        
        # Initialize UI state
        self.initialize_ui_state()
        
        # Update database status
        self.update_database_status()

    def setup_ui(self):
        """Set up the main user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_import_tab()
        self.create_clustering_tab()
        self.create_visualization_tab()
        self.create_pca_components_tab()
        self.create_analysis_tab()
        self.create_refinement_tab()
        
        # New advanced analysis tabs
        self.create_time_series_tab()
        self.create_kinetics_tab()
        self.create_structural_analysis_tab()
        self.create_validation_tab()
        self.create_advanced_statistics_tab()
        
        # Create status bar
        self.status_bar = QLabel("Initializing...")
        self.status_bar.setStyleSheet("QLabel { border: 1px solid gray; padding: 2px; }")
        main_layout.addWidget(self.status_bar)
        
        # Initially disable clustering tab
        self.tab_widget.setTabEnabled(1, False)  # Disable clustering tab

    def initialize_ui_state(self):
        """Initialize the UI state."""
        # Set initial status messages
        self.import_status.setText("No data imported")
        self.clustering_status.setText("No data available for clustering")

    def update_clustering_controls(self):
        """Update the state of clustering controls based on data availability."""
        try:
            if self.cluster_data is None or self.cluster_data['features'] is None:
                self.n_clusters_spinbox.setEnabled(False)
                self.linkage_method_combo.setEnabled(False)
                self.distance_metric_combo.setEnabled(False)
                self.clustering_status.setText("No data available for clustering")
                return
            
            # Enable controls
            self.n_clusters_spinbox.setEnabled(True)
            self.linkage_method_combo.setEnabled(True)
            self.distance_metric_combo.setEnabled(True)
            self.clustering_status.setText(f"Ready to cluster {len(self.cluster_data['features'])} spectra")
            
        except Exception as e:
            print(f"Error updating controls: {str(e)}")

    def update_ui_after_import(self, num_spectra):
        """Update UI after successful data import."""
        try:
            # Update import status
            self.import_status.setText(f"Successfully imported {num_spectra} spectra")
            self.import_progress.setValue(100)
            
            # Enable clustering tab
            self.tab_widget.setTabEnabled(1, True)
            
            # Update clustering controls
            self.update_clustering_controls()
            
            # Update kinetics UI with new data
            if hasattr(self, 'update_time_input_controls'):
                self.update_time_input_controls()
            
            # Switch to clustering tab
            self.tab_widget.setCurrentIndex(1)
            
        except Exception as e:
            print(f"Error updating UI: {str(e)}")

    def update_database_status(self):
        """Update database connection status."""
        # This is a placeholder - actual implementation would check database connectivity
        self.status_bar.setText("Ready")

    def create_import_tab(self):
        """Create the import tab."""
        import_widget = QWidget()
        layout = QVBoxLayout(import_widget)
        
        # Create folder selection frame
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout(folder_group)
        
        # Add folder path display
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_path_label)
        
        # Add select folder button
        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.select_import_folder)
        folder_layout.addWidget(select_folder_btn)
        
        # Add import from database button
        import_db_btn = QPushButton("Import from Database")
        import_db_btn.clicked.connect(self.open_database_import_dialog)
        folder_layout.addWidget(import_db_btn)
        
        # Add import from main app button
        import_main_btn = QPushButton("Import from Main App")
        import_main_btn.clicked.connect(self.import_from_main_app)
        folder_layout.addWidget(import_main_btn)
        
        layout.addWidget(folder_group)
        
        # Create data configuration frame - simplified since we're using smart parsing
        config_group = QGroupBox("Import Options")
        config_layout = QVBoxLayout(config_group)
        
        # Info label about automatic parsing
        info_label = QLabel("Files will be automatically parsed using intelligent format detection:\n"
                           "• Automatic delimiter detection (comma, tab, space)\n"
                           "• Header and metadata extraction\n"
                           "• Multiple file format support (.txt, .csv, .dat, .asc)")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 4px;")
        config_layout.addWidget(info_label)
        
        layout.addWidget(config_group)
        
        # Add start import button
        start_import_btn = QPushButton("Start Import")
        start_import_btn.clicked.connect(self.start_batch_import)
        layout.addWidget(start_import_btn)
        
        # Add progress bar
        self.import_progress = QProgressBar()
        layout.addWidget(self.import_progress)
        
        # Add status label
        self.import_status = QLabel("")
        layout.addWidget(self.import_status)
        
        # Add append data section
        append_group = QGroupBox("Append Additional Data")
        append_layout = QVBoxLayout(append_group)
        
        append_btn = QPushButton("Append Data")
        append_btn.clicked.connect(self.append_data)
        append_layout.addWidget(append_btn)
        
        layout.addWidget(append_group)
        
        layout.addStretch()
        self.tab_widget.addTab(import_widget, "Import")

    def create_clustering_tab(self):
        """Create the clustering tab."""
        clustering_widget = QWidget()
        layout = QVBoxLayout(clustering_widget)
        
        # Create controls frame
        controls_group = QGroupBox("Clustering Controls")
        controls_layout = QFormLayout(controls_group)
        
        # Add number of clusters selection
        self.n_clusters_spinbox = QSpinBox()
        self.n_clusters_spinbox.setRange(2, 20)
        self.n_clusters_spinbox.setValue(5)
        controls_layout.addRow("Number of Clusters:", self.n_clusters_spinbox)
        
        # Add linkage method selection
        self.linkage_method_combo = QComboBox()
        self.linkage_method_combo.addItems(['ward', 'complete', 'average', 'single'])
        self.linkage_method_combo.setCurrentText('ward')
        controls_layout.addRow("Linkage Method:", self.linkage_method_combo)
        
        # Add distance metric selection
        self.distance_metric_combo = QComboBox()
        self.distance_metric_combo.addItems(['euclidean', 'cosine', 'correlation'])
        self.distance_metric_combo.setCurrentText('euclidean')
        controls_layout.addRow("Distance Metric:", self.distance_metric_combo)
        
        layout.addWidget(controls_group)
        
        # Add preprocessing controls
        preprocessing_group = QGroupBox("Preprocessing")
        preprocessing_layout = QFormLayout(preprocessing_group)
        
        # Add phase separation method selection
        self.phase_method_combo = QComboBox()
        self.phase_method_combo.addItems(['None', 'Exclude Regions', 'Corundum Correction', 'NMF Separation', 'Carbon Soot Optimization'])
        self.phase_method_combo.setCurrentText('None')
        self.phase_method_combo.currentTextChanged.connect(self.update_preprocessing_controls)
        preprocessing_layout.addRow("Phase Separation Method:", self.phase_method_combo)
        
        # Add exclusion regions control
        self.exclusion_regions_edit = QLineEdit()
        self.exclusion_regions_edit.setPlaceholderText("Enter exclusion regions (e.g., 400-500,600-700)")
        self.exclusion_regions_edit.setEnabled(False)
        preprocessing_layout.addRow("Exclusion Regions:", self.exclusion_regions_edit)
        
        # Add NMF components control
        self.nmf_components_spinbox = QSpinBox()
        self.nmf_components_spinbox.setRange(2, 10)
        self.nmf_components_spinbox.setValue(3)
        self.nmf_components_spinbox.setEnabled(False)
        preprocessing_layout.addRow("NMF Components:", self.nmf_components_spinbox)
        
        layout.addWidget(preprocessing_group)
        
        # Carbon-specific analysis section
        carbon_group = QGroupBox("Carbon Soot Analysis")
        carbon_layout = QVBoxLayout(carbon_group)

        # Enable carbon optimization
        self.enable_carbon_analysis_cb = QCheckBox("Enable Carbon Soot Optimization")
        self.enable_carbon_analysis_cb.setChecked(False)
        self.enable_carbon_analysis_cb.stateChanged.connect(self.update_carbon_controls)
        carbon_layout.addWidget(self.enable_carbon_analysis_cb)

        # Carbon analysis parameters
        carbon_params_frame = QFrame()
        carbon_params_layout = QFormLayout(carbon_params_frame)

        # D-band range
        self.d_band_range_edit = QLineEdit("1300-1400")
        carbon_params_layout.addRow("D-band Range (cm⁻¹):", self.d_band_range_edit)

        # G-band range
        self.g_band_range_edit = QLineEdit("1550-1620")
        carbon_params_layout.addRow("G-band Range (cm⁻¹):", self.g_band_range_edit)

        # Analysis buttons
        carbon_buttons = QHBoxLayout()

        analyze_carbon_btn = QPushButton("Analyze Carbon Features")
        analyze_carbon_btn.clicked.connect(self.print_carbon_feature_analysis)
        carbon_buttons.addWidget(analyze_carbon_btn)

        suggest_improvements_btn = QPushButton("Suggest Improvements")
        suggest_improvements_btn.clicked.connect(self.suggest_clustering_improvements)
        carbon_buttons.addWidget(suggest_improvements_btn)
        
        # Add NMF info button
        nmf_info_btn = QPushButton("NMF Clustering Info")
        nmf_info_btn.clicked.connect(self.show_nmf_clustering_info)
        carbon_buttons.addWidget(nmf_info_btn)

        carbon_params_layout.addRow(carbon_buttons)

        carbon_layout.addWidget(carbon_params_frame)
        layout.addWidget(carbon_group)

        # Initially hide carbon controls
        carbon_params_frame.setVisible(False)
        self.carbon_controls = [carbon_params_frame]
        
        # Add run buttons
        button_layout = QHBoxLayout()
        
        # Standard clustering button
        run_clustering_btn = QPushButton("Run Standard Clustering")
        run_clustering_btn.clicked.connect(self.run_clustering)
        button_layout.addWidget(run_clustering_btn)
        
        # Probabilistic clustering button
        run_prob_btn = QPushButton("Run Probabilistic Clustering")
        run_prob_btn.clicked.connect(self.run_probabilistic_clustering)
        run_prob_btn.setToolTip("Perform clustering with probability estimates and sub-type identification")
        button_layout.addWidget(run_prob_btn)
        
        layout.addLayout(button_layout)
        
        # Add visualization buttons for probabilistic results
        prob_vis_layout = QHBoxLayout()
        
        self.prob_viz_btn = QPushButton("Show Probability Heatmap")
        self.prob_viz_btn.setToolTip("Visualize cluster assignment probabilities")
        self.prob_viz_btn.clicked.connect(self.plot_probability_heatmap)
        self.prob_viz_btn.setEnabled(False)  # Disabled until clustering is run
        prob_vis_layout.addWidget(self.prob_viz_btn)
        
        self.subtype_viz_btn = QPushButton("View Sub-type Hierarchies")
        self.subtype_viz_btn.setToolTip("View hierarchical sub-type structures")
        self.subtype_viz_btn.clicked.connect(lambda: self.plot_dendrogram(None))
        self.subtype_viz_btn.setEnabled(False)  # Disabled until clustering is run
        prob_vis_layout.addWidget(self.subtype_viz_btn)
        
        layout.addLayout(prob_vis_layout)
        
        # Add progress bar
        self.clustering_progress = QProgressBar()
        layout.addWidget(self.clustering_progress)
        
        # Add status label
        self.clustering_status = QLabel("")
        layout.addWidget(self.clustering_status)
        
        layout.addStretch()
        self.tab_widget.addTab(clustering_widget, "Clustering")

    def create_visualization_tab(self):
        """Create the visualization tab."""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)
        
        # Create tab widget for different visualizations
        self.viz_tab_widget = QTabWidget()
        layout.addWidget(self.viz_tab_widget)
        
        # Create dendrogram tab
        self.create_dendrogram_tab()
        
        # Create heatmap tab
        self.create_heatmap_tab()
        
        # Create scatter tab
        self.create_scatter_tab()
        
        # Add export button
        export_btn = QPushButton("Export Visualization")
        export_btn.clicked.connect(self.export_visualization)
        layout.addWidget(export_btn)
        
        self.tab_widget.addTab(viz_widget, "Visualization")

    def create_dendrogram_tab(self):
        """Create the dendrogram visualization tab."""
        dendro_widget = QWidget()
        layout = QVBoxLayout(dendro_widget)
        
        # Add controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Orientation control
        controls_layout.addWidget(QLabel("Orientation:"))
        self.dendro_orientation = QComboBox()
        self.dendro_orientation.addItems(['top', 'bottom', 'left', 'right'])
        self.dendro_orientation.setCurrentText('top')
        self.dendro_orientation.currentTextChanged.connect(self.update_dendrogram)
        controls_layout.addWidget(self.dendro_orientation)
        
        # Max samples to show
        controls_layout.addWidget(QLabel("Max Samples:"))
        self.dendro_max_samples = QSpinBox()
        self.dendro_max_samples.setRange(10, 200)
        self.dendro_max_samples.setValue(50)
        self.dendro_max_samples.valueChanged.connect(self.update_dendrogram)
        controls_layout.addWidget(self.dendro_max_samples)
        
        # Label display option
        self.dendro_show_labels = QCheckBox("Show Labels")
        self.dendro_show_labels.setChecked(True)
        self.dendro_show_labels.toggled.connect(self.update_dendrogram)
        controls_layout.addWidget(self.dendro_show_labels)
        
        # Update button
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.update_dendrogram)
        controls_layout.addWidget(update_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_frame)
        
        # Create dendrogram figure
        self.dendro_fig = Figure(figsize=(10, 6))
        self.dendrogram_ax = self.dendro_fig.add_subplot(111)
        self.dendro_canvas = FigureCanvas(self.dendro_fig)
        layout.addWidget(self.dendro_canvas)
        
        # Add toolbar
        self.dendro_toolbar = NavigationToolbar(self.dendro_canvas, dendro_widget)
        layout.addWidget(self.dendro_toolbar)
        
        self.viz_tab_widget.addTab(dendro_widget, "Dendrogram")

    def create_heatmap_tab(self):
        """Create the heatmap visualization tab."""
        heatmap_widget = QWidget()
        layout = QVBoxLayout(heatmap_widget)
        
        # Add controls for heatmap appearance
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Colormap selection
        controls_layout.addWidget(QLabel("Colormap:"))
        self.heatmap_colormap = QComboBox()
        self.heatmap_colormap.addItems(['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu_r', 'jet'])
        self.heatmap_colormap.setCurrentText('viridis')
        self.heatmap_colormap.currentTextChanged.connect(self.delayed_heatmap_update)
        controls_layout.addWidget(self.heatmap_colormap)
        
        # Normalization method
        controls_layout.addWidget(QLabel("Normalization:"))
        self.heatmap_norm = QComboBox()
        self.heatmap_norm.addItems(['linear', 'log', 'sqrt', 'row', 'column'])
        self.heatmap_norm.setCurrentText('linear')
        self.heatmap_norm.currentTextChanged.connect(self.delayed_heatmap_update)
        controls_layout.addWidget(self.heatmap_norm)
        
        # Contrast adjustment
        controls_layout.addWidget(QLabel("Contrast:"))
        self.heatmap_contrast = QSlider(Qt.Horizontal)
        self.heatmap_contrast.setRange(1, 50)  # 0.1 to 5.0 scaled by 10
        self.heatmap_contrast.setValue(10)  # 1.0
        self.heatmap_contrast.valueChanged.connect(self.delayed_heatmap_update)
        self.heatmap_contrast.setToolTip("Adjust contrast (0.1x to 5.0x)")
        controls_layout.addWidget(self.heatmap_contrast)
        
        # Add contrast value label
        self.heatmap_contrast_label = QLabel("1.0x")
        self.heatmap_contrast_label.setMinimumWidth(40)
        self.heatmap_contrast.valueChanged.connect(lambda v: self.heatmap_contrast_label.setText(f"{v/10.0:.1f}x"))
        controls_layout.addWidget(self.heatmap_contrast_label)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.force_heatmap_refresh)
        refresh_btn.setToolTip("Force heatmap refresh (use if display issues occur)")
        controls_layout.addWidget(refresh_btn)
        
        # Add nuclear reset button
        # nuclear_btn = QPushButton("Nuclear Reset")
        # nuclear_btn.clicked.connect(self.nuclear_heatmap_reset)
        # nuclear_btn.setToolTip("Complete heatmap rebuild (last resort for display issues)")
        # nuclear_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; font-weight: bold; }")
        # controls_layout.addWidget(nuclear_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_frame)
        
        # Create heatmap figure
        self.heatmap_fig = Figure(figsize=(10, 6))
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        layout.addWidget(self.heatmap_canvas)
        
        # Add toolbar
        self.heatmap_toolbar = NavigationToolbar(self.heatmap_canvas, heatmap_widget)
        layout.addWidget(self.heatmap_toolbar)
        
        self.viz_tab_widget.addTab(heatmap_widget, "Heatmap")

    def create_scatter_tab(self):
        """Create the scatter plot visualization tab."""
        scatter_widget = QWidget()
        layout = QVBoxLayout(scatter_widget)
        
        # Add visualization method selector and parameters
        method_frame = QFrame()
        method_layout = QVBoxLayout(method_frame)
        
        # Method selection row
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Visualization Method:"))
        self.visualization_method_combo = QComboBox()
        self.visualization_method_combo.addItems(['PCA'])
        if UMAP_AVAILABLE:
            self.visualization_method_combo.addItem('UMAP')
        self.visualization_method_combo.setCurrentText('PCA')
        self.visualization_method_combo.currentTextChanged.connect(self.update_visualization_params)
        method_row.addWidget(self.visualization_method_combo)
        
        # Colormap selection
        method_row.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        # Add common matplotlib colormaps
        colormaps = [
            'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples',
            'coolwarm', 'RdYlBu', 'RdBu', 'Spectral', 'rainbow',
            'hsv', 'spring', 'summer', 'autumn', 'winter',
            'Pastel1', 'Pastel2', 'Dark2', 'Accent', 'Paired'
        ]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText('Set1')
        self.colormap_combo.setToolTip("Select colormap for cluster visualization")
        self.colormap_combo.currentTextChanged.connect(self.update_scatter_plot)
        method_row.addWidget(self.colormap_combo)
        
        # Reverse colormap checkbox
        self.reverse_colormap_cb = QCheckBox("Reverse")
        self.reverse_colormap_cb.setToolTip("Reverse the colormap order")
        self.reverse_colormap_cb.stateChanged.connect(self.update_scatter_plot)
        method_row.addWidget(self.reverse_colormap_cb)
        
        # Colormap preview button
        self.colormap_preview_btn = QPushButton("Preview")
        self.colormap_preview_btn.setToolTip("Show a preview of the selected colormap")
        self.colormap_preview_btn.clicked.connect(self.show_colormap_preview)
        self.colormap_preview_btn.setMaximumWidth(60)
        method_row.addWidget(self.colormap_preview_btn)
        
        method_row.addStretch()
        method_layout.addLayout(method_row)
        
        # UMAP parameters (initially hidden)
        self.umap_params_frame = QFrame()
        umap_params_layout = QGridLayout(self.umap_params_frame)
        
        # n_neighbors parameter
        umap_params_layout.addWidget(QLabel("Neighbors:"), 0, 0)
        self.umap_n_neighbors = QSpinBox()
        self.umap_n_neighbors.setRange(2, 200)
        self.umap_n_neighbors.setValue(15)
        self.umap_n_neighbors.setToolTip("Number of neighbors for UMAP (2-200). Higher values preserve more global structure.")
        umap_params_layout.addWidget(self.umap_n_neighbors, 0, 1)
        
        # min_dist parameter  
        umap_params_layout.addWidget(QLabel("Min Distance:"), 0, 2)
        self.umap_min_dist = QDoubleSpinBox()
        self.umap_min_dist.setRange(0.001, 0.99)
        self.umap_min_dist.setSingleStep(0.05)
        self.umap_min_dist.setValue(0.1)
        self.umap_min_dist.setDecimals(3)
        self.umap_min_dist.setToolTip("Minimum distance between points (0.001-0.99). Lower values allow more clustering.")
        umap_params_layout.addWidget(self.umap_min_dist, 0, 3)
        
        # metric parameter
        umap_params_layout.addWidget(QLabel("Metric:"), 1, 0)
        self.umap_metric = QComboBox()
        self.umap_metric.addItems(['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine', 'correlation'])
        self.umap_metric.setCurrentText('euclidean')
        self.umap_metric.setToolTip("Distance metric for UMAP")
        umap_params_layout.addWidget(self.umap_metric, 1, 1)
        
        # spread parameter
        umap_params_layout.addWidget(QLabel("Spread:"), 1, 2) 
        self.umap_spread = QDoubleSpinBox()
        self.umap_spread.setRange(0.1, 3.0)
        self.umap_spread.setSingleStep(0.1)
        self.umap_spread.setValue(1.0)
        self.umap_spread.setDecimals(1)
        self.umap_spread.setToolTip("Effective scale of embedded points")
        umap_params_layout.addWidget(self.umap_spread, 1, 3)
        
        # UMAP presets for different material types
        umap_params_layout.addWidget(QLabel("UMAP Presets:"), 2, 0)
        self.umap_preset_combo = QComboBox()
        self.umap_preset_combo.addItems([
            'Custom',
            'Carbon Soot (Tight Clusters)',
            'Carbon Soot (Broad Clusters)',
            'Carbon Manifold (Continuous)', 
            'General Spectroscopy',
            'High Noise Data'
        ])
        self.umap_preset_combo.currentTextChanged.connect(self.apply_umap_preset)
        self.umap_preset_combo.setToolTip("Apply optimized UMAP parameters for different data types")
        umap_params_layout.addWidget(self.umap_preset_combo, 2, 1, 1, 3)
        
        # Update button
        self.update_umap_btn = QPushButton("Update UMAP")
        self.update_umap_btn.clicked.connect(self.update_scatter_plot)
        umap_params_layout.addWidget(self.update_umap_btn, 3, 0, 1, 4)
        
        # Initially hide UMAP parameters
        self.umap_params_frame.setVisible(False)
        method_layout.addWidget(self.umap_params_frame)
        
        layout.addWidget(method_frame)
        
        # Create scatter figure
        self.viz_fig = Figure(figsize=(10, 6))
        self.viz_ax = self.viz_fig.add_subplot(111)
        self.viz_canvas = FigureCanvas(self.viz_fig)
        
        # Initialize hover annotation (hidden initially)
        self.hover_annotation = None
        self.scatter_points = []  # Store scatter plot objects for hover detection
        
        # Connect mouse events for hover tooltips
        self.viz_canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        layout.addWidget(self.viz_canvas)
        
        # Add toolbar
        self.viz_toolbar = NavigationToolbar(self.viz_canvas, scatter_widget)
        layout.addWidget(self.viz_toolbar)
        
        # Add cluster export controls
        export_frame = QFrame()
        export_layout = QVBoxLayout(export_frame)
        
        # Export section title
        export_title = QLabel("Cluster Export Options")
        export_title.setStyleSheet("font-weight: bold; font-size: 11px; color: #333; padding: 4px;")
        export_layout.addWidget(export_title)
        
        # Export buttons row
        export_buttons_row = QHBoxLayout()
        
        # Export to folders button
        self.export_folders_btn = QPushButton("Export to Folders")
        self.export_folders_btn.setToolTip("Export each cluster's spectra to separate folders for individual analysis")
        self.export_folders_btn.clicked.connect(self.export_clusters_to_folders)
        self.export_folders_btn.setEnabled(False)
        export_buttons_row.addWidget(self.export_folders_btn)
        
        # Export summed spectra button
        self.export_summed_btn = QPushButton("Export Summed Spectra")
        self.export_summed_btn.setToolTip("Export summed/averaged spectra for each cluster as a single plot")
        self.export_summed_btn.clicked.connect(self.export_summed_cluster_spectra)
        self.export_summed_btn.setEnabled(False)
        export_buttons_row.addWidget(self.export_summed_btn)
        
        # Export cluster overview button
        self.export_overview_btn = QPushButton("Export Cluster Overview")
        self.export_overview_btn.setToolTip("Export a comprehensive overview with all clusters in a grid layout")
        self.export_overview_btn.clicked.connect(self.export_cluster_overview)
        self.export_overview_btn.setEnabled(False)
        export_buttons_row.addWidget(self.export_overview_btn)
        
        # Export XY plot data button
        self.export_xy_btn = QPushButton("Export XY Plot Data")
        self.export_xy_btn.setToolTip("Export the XY coordinates of the current scatter plot (PCA or UMAP)")
        self.export_xy_btn.clicked.connect(self.export_xy_plot_data)
        self.export_xy_btn.setEnabled(False)
        export_buttons_row.addWidget(self.export_xy_btn)
        
        export_layout.addLayout(export_buttons_row)
        
        # Export status
        self.export_status = QLabel("No clusters available for export")
        self.export_status.setStyleSheet("color: #666; font-size: 9px; padding: 2px;")
        export_layout.addWidget(self.export_status)
        
        layout.addWidget(export_frame)
        
        self.viz_tab_widget.addTab(scatter_widget, "Scatter Plot")

    def update_visualization_params(self):
        """Update parameter visibility based on selected visualization method."""
        if hasattr(self, 'umap_params_frame'):
            method = self.visualization_method_combo.currentText()
            self.umap_params_frame.setVisible(method == 'UMAP')
            # Auto-update when switching methods
            if hasattr(self, 'cluster_data') and self.cluster_data['features_scaled'] is not None:
                self.update_scatter_plot()

    def apply_umap_preset(self):
        """Apply UMAP parameter presets optimized for different data types."""
        preset = self.umap_preset_combo.currentText()
        
        if preset == 'Carbon Soot (Tight Clusters)':
            self.umap_n_neighbors.setValue(5)
            self.umap_min_dist.setValue(0.01)
            self.umap_metric.setCurrentText('cosine')
            self.umap_spread.setValue(0.5)
            
        elif preset == 'Carbon Soot (Broad Clusters)':
            self.umap_n_neighbors.setValue(15)
            self.umap_min_dist.setValue(0.1)
            self.umap_metric.setCurrentText('cosine')
            self.umap_spread.setValue(1.0)
            
        elif preset == 'Carbon Manifold (Continuous)':
            self.umap_n_neighbors.setValue(20)
            self.umap_min_dist.setValue(0.1)
            self.umap_metric.setCurrentText('cosine')
            self.umap_spread.setValue(1.0)
            
        elif preset == 'General Spectroscopy':
            self.umap_n_neighbors.setValue(15)
            self.umap_min_dist.setValue(0.1)
            self.umap_metric.setCurrentText('euclidean')
            self.umap_spread.setValue(1.0)
            
        elif preset == 'High Noise Data':
            self.umap_n_neighbors.setValue(30)
            self.umap_min_dist.setValue(0.5)
            self.umap_metric.setCurrentText('manhattan')
            self.umap_spread.setValue(2.0)

    def update_scatter_plot(self):
        """
        Enhanced UMAP implementation specifically optimized for carbon soot clustering.
        """
        if (self.cluster_data['features_scaled'] is None or 
            self.cluster_data['labels'] is None):
            return
        
        try:
            self.viz_ax.clear()
            
            # Clear previous scatter points data and hover annotation
            self.scatter_points = []
            self.hover_annotation = None
            
            # Get visualization method
            method = self.visualization_method_combo.currentText()
            
            # Use standard features (carbon optimization disabled for better results)
            features_scaled = self.cluster_data['features_scaled']
            
            labels = self.cluster_data['labels']
            
            # Get hover labels
            hover_labels = self.get_hover_labels()
            
            if method == 'PCA':
                # Standard PCA implementation
                pca = PCA(n_components=2)
                coords = pca.fit_transform(features_scaled)
                
                self.cluster_data['pca_coords'] = coords
                self.cluster_data['pca_model'] = pca
                
                xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'
                ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
                title = 'PCA Visualization of Clusters'
                
            elif method == 'UMAP' and UMAP_AVAILABLE:
                # Optimized UMAP parameters for carbon soot discrimination
                n_neighbors = max(5, min(50, len(features_scaled) // 3))  # Adaptive neighbors
                min_dist = 0.01  # Very small for tight clusters
                metric = 'cosine'  # Better for spectral data
                spread = 0.5  # Tighter spread
                
                # Override with UI values if available
                if hasattr(self, 'umap_n_neighbors'):
                    n_neighbors = min(self.umap_n_neighbors.value(), len(features_scaled) - 1)
                if hasattr(self, 'umap_min_dist'):
                    min_dist = self.umap_min_dist.value()
                if hasattr(self, 'umap_metric'):
                    metric = self.umap_metric.currentText()
                if hasattr(self, 'umap_spread'):
                    spread = self.umap_spread.value()
                
                try:
                    print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
                    
                    # Check if we want manifold structure or discrete clustering
                    use_manifold_params = (hasattr(self, 'umap_preset_combo') and 
                                         self.umap_preset_combo.currentText() == 'Carbon Manifold (Continuous)')
                    
                    if use_manifold_params:
                        # Use gentle parameters to preserve manifold structure
                        umap_model = umap.UMAP(
                            n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                            spread=spread,
                            random_state=42,
                            n_jobs=1,
                            # Gentle parameters for manifold preservation
                            local_connectivity=1.0,  # Standard connectivity
                            repulsion_strength=1.0,  # Standard repulsion
                            init='spectral',         # Good for spectral data
                            verbose=False
                        )
                    else:
                        # Create UMAP with optimized parameters for carbon discrimination
                        umap_model = umap.UMAP(
                            n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                            spread=spread,
                            random_state=42,
                            n_jobs=1,
                            # Additional parameters for better carbon clustering
                            local_connectivity=2.0,  # Increase local connectivity
                            repulsion_strength=2.0,   # Increase repulsion for better separation
                            negative_sample_rate=10,  # More negative samples for better structure
                            transform_queue_size=8.0, # Larger queue for stability
                            a=None, b=None,          # Let UMAP optimize these
                            init='spectral',         # Better initialization for spectral data
                            densmap=False,           # Focus on topology, not density
                            dens_lambda=2.0,
                            dens_frac=0.3,
                            dens_var_shift=0.1,
                            output_dens=False,
                            verbose=True             # Show progress
                        )
                    
                    coords = umap_model.fit_transform(features_scaled)
                    
                    # Store UMAP results
                    self.cluster_data['umap_coords'] = coords
                    self.cluster_data['umap_model'] = umap_model
                    if self.cluster_data.get('carbon_optimized', False):
                        self.cluster_data['carbon_features'] = carbon_features
                        self.cluster_data['carbon_features_scaled'] = features_scaled
                    
                    xlabel = f'UMAP 1 (carbon-optimized: {metric} metric)'
                    ylabel = f'UMAP 2 (neighbors={n_neighbors}, min_dist={min_dist:.3f})'
                    title = 'UMAP Visualization of Clusters'
                    
                    print("UMAP completed successfully")
                    
                except Exception as e:
                    print(f"UMAP failed with optimized parameters: {e}")
                    print("Falling back to more conservative UMAP settings...")
                    
                    try:
                        # Fallback UMAP with very conservative settings
                        fallback_umap = umap.UMAP(
                            n_components=2,
                            n_neighbors=min(15, len(features_scaled) - 1),
                            min_dist=0.1,
                            metric='euclidean',
                            random_state=42,
                            n_jobs=1
                        )
                        coords = fallback_umap.fit_transform(features_scaled)
                        
                        xlabel = 'UMAP 1 (fallback mode)'
                        ylabel = 'UMAP 2 (fallback mode)'
                        title = 'UMAP Visualization (Fallback)'
                        
                    except Exception as e2:
                        print(f"Fallback UMAP also failed: {e2}")
                        # Final fallback to PCA
                        pca = PCA(n_components=2)
                        coords = pca.fit_transform(features_scaled)
                        xlabel = f'PC1 (UMAP failed - {pca.explained_variance_ratio_[0]:.1%})'
                        ylabel = f'PC2 (UMAP failed - {pca.explained_variance_ratio_[1]:.1%})'
                        title = 'PCA Visualization (UMAP Failed)'
            
            else:
                # Fallback to PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(features_scaled)
                xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
                ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
                title = 'PCA Visualization'
            
            # Create enhanced scatter plot
            unique_labels = np.unique(labels)
            
            # Get colormap
            colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'Set1'
            reverse_colormap = self.reverse_colormap_cb.isChecked() if hasattr(self, 'reverse_colormap_cb') else False
            
            try:
                colormap = plt.cm.get_cmap(colormap_name)
                if reverse_colormap:
                    colormap = colormap.reversed()
                colors = colormap(np.linspace(0, 1, len(unique_labels)))
            except Exception as e:
                print(f"Error loading colormap '{colormap_name}': {e}")
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            # Plot clusters with enhanced visualization
            for i, label in enumerate(unique_labels):
                mask = labels == label
                cluster_coords = coords[mask]
                cluster_hover_labels = [hover_labels[j] for j in range(len(hover_labels)) if mask[j]]
                
                # Create scatter plot for this cluster with enhanced styling
                scatter = self.viz_ax.scatter(
                    cluster_coords[:, 0], 
                    cluster_coords[:, 1], 
                    c=[colors[i]], 
                    label=f'Cluster {label} (n={len(cluster_coords)})',
                    alpha=0.8,
                    s=60,  # Slightly larger points
                    edgecolors='white',
                    linewidths=1.0  # Thicker edge for better visibility
                )
                
                # Store scatter point information for hover detection
                self.scatter_points.append({
                    'scatter': scatter,
                    'coords': cluster_coords,
                    'labels': cluster_hover_labels,
                    'cluster': label
                })
                
                # Add cluster centroid with matching color
                centroid = np.mean(cluster_coords, axis=0)
                self.viz_ax.scatter(centroid[0], centroid[1], 
                                  c=[colors[i]], marker='x', s=200, 
                                  edgecolors='black', linewidths=2,  # Black outline for visibility
                                  zorder=10)  # Ensure centroids are on top
            
            self.viz_ax.set_xlabel(xlabel)
            self.viz_ax.set_ylabel(ylabel)
            self.viz_ax.set_title(title)
            self.viz_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.viz_ax.grid(True, alpha=0.3)
            
            # Add text box with optimization info (carbon optimization disabled)
            # if method == 'UMAP' and self.cluster_data.get('carbon_optimized', False):
            #     info_text = f"Carbon-optimized UMAP\nFeatures: D/G ratios, peak positions,\nwidths, and carbon-specific signatures"
            #     self.viz_ax.text(0.02, 0.98, info_text, transform=self.viz_ax.transAxes,
            #                    verticalalignment='top', bbox=dict(boxstyle='round',
            #                    facecolor='lightblue', alpha=0.8), fontsize=9)
            
            self.viz_fig.tight_layout()
            self.viz_canvas.draw()
            
            # Enable export buttons
            if hasattr(self, 'export_folders_btn') and unique_labels is not None:
                self.export_folders_btn.setEnabled(True)
                self.export_summed_btn.setEnabled(True)
                self.export_overview_btn.setEnabled(True)
                self.export_xy_btn.setEnabled(True)
                self.export_status.setText(f"Ready to export {len(unique_labels)} clusters")
            
            # Print feature importance for debugging (carbon optimization disabled)
            # if method == 'UMAP' and self.cluster_data.get('carbon_optimized', False):
            #     self.print_carbon_feature_analysis()
            
        except Exception as e:
            print(f"Error updating scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_hover_labels(self):
        """Get hover labels (last 8 characters of filenames) for each spectrum."""
        hover_labels = []
        
        if 'spectrum_metadata' in self.cluster_data and self.cluster_data['spectrum_metadata']:
            for metadata in self.cluster_data['spectrum_metadata']:
                filename = metadata.get('filename', 'Unknown')
                # Extract last 8 characters of filename (excluding extension)
                if '.' in filename:
                    name_part = filename.rsplit('.', 1)[0]  # Remove extension
                else:
                    name_part = filename
                
                # Get last 8 characters
                short_name = name_part[-8:] if len(name_part) > 8 else name_part
                hover_labels.append(short_name)
        else:
            # Fallback if no metadata available
            n_spectra = len(self.cluster_data['labels']) if self.cluster_data['labels'] is not None else 0
            hover_labels = [f"Spec_{i:03d}" for i in range(n_spectra)]
        
        return hover_labels

    def on_hover(self, event):
        """Handle mouse hover events for showing spectrum tooltips."""
        if event.inaxes != self.viz_ax or not hasattr(self, 'scatter_points') or not self.scatter_points:
            if hasattr(self, 'hover_annotation') and self.hover_annotation:
                self.hover_annotation.set_visible(False)
                self.viz_canvas.draw_idle()
            return
        
        # Check if mouse is over any scatter point
        found_point = False
        

        
        for cluster_info in self.scatter_points:
            scatter = cluster_info['scatter']
            coords = cluster_info['coords']
            labels = cluster_info['labels']
            cluster = cluster_info['cluster']
            
            # Check if mouse is over this cluster's points
            for i, (x, y) in enumerate(coords):
                # Calculate distance from mouse to point
                if event.xdata is not None and event.ydata is not None:
                    distance = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
                    
                    # Convert to display coordinates for more accurate detection
                    point_display = self.viz_ax.transData.transform([[x, y]])[0]
                    mouse_display = [event.x, event.y]
                    display_distance = np.sqrt((mouse_display[0] - point_display[0])**2 + 
                                             (mouse_display[1] - point_display[1])**2)
                    
                    # If close enough (within 10 pixels), show tooltip
                    if display_distance < 10:
                        self.show_hover_tooltip(event, labels[i], cluster, x, y)
                        found_point = True
                        break
            
            if found_point:
                break
        
        # Hide tooltip if not over any point
        if not found_point and hasattr(self, 'hover_annotation') and self.hover_annotation:
            self.hover_annotation.set_visible(False)
            self.viz_canvas.draw_idle()

    def show_hover_tooltip(self, event, label, cluster, x, y):
        """Show hover tooltip with spectrum information."""
        try:
            if not hasattr(self, 'hover_annotation') or self.hover_annotation is None:
                # Create annotation if it doesn't exist
                self.hover_annotation = self.viz_ax.annotate(
                    '', 
                    xy=(0, 0), 
                    xytext=(10, 10), 
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
                )
            
            # Update annotation content and position
            tooltip_text = f"{label}\nCluster {cluster}"
            self.hover_annotation.set_text(tooltip_text)
            self.hover_annotation.xy = (x, y)
            self.hover_annotation.set_visible(True)
            
            # Redraw canvas
            self.viz_canvas.draw_idle()
        except Exception as e:
            print(f"Error showing tooltip: {e}")
            # Reset hover annotation if there's an error
            self.hover_annotation = None

    def show_colormap_preview(self):
        """Show a preview of the selected colormap."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
        from matplotlib.backends.qt_compat import QtWidgets
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        try:
            # Create preview dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Colormap Preview")
            dialog.setFixedSize(500, 200)
            layout = QVBoxLayout(dialog)
            
            # Get colormap settings
            colormap_name = self.colormap_combo.currentText()
            reverse_colormap = self.reverse_colormap_cb.isChecked()
            
            # Create figure for preview
            fig = Figure(figsize=(8, 2))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Create colormap preview
            colormap = plt.cm.get_cmap(colormap_name)
            if reverse_colormap:
                colormap = colormap.reversed()
            
            # Create sample data to show colormap
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(gradient, aspect='auto', cmap=colormap)
            ax.set_xlim(0, 256)
            ax.set_yticks([])
            ax.set_xlabel('Color Values')
            ax.set_title(f'Colormap: {colormap_name}{"_r" if reverse_colormap else ""}')
            
            fig.tight_layout()
            layout.addWidget(canvas)
            
            # Add info label
            info_label = QLabel(f"Preview of '{colormap_name}' colormap")
            info_label.setStyleSheet("font-weight: bold; margin: 5px;")
            layout.addWidget(info_label)
            
            dialog.exec()
            
        except Exception as e:
            print(f"Error showing colormap preview: {e}")

    def create_pca_components_tab(self):
        """Create the PCA components analysis tab."""
        pca_widget = QWidget()
        layout = QVBoxLayout(pca_widget)
        
        # Add controls
        controls_group = QGroupBox("PCA Components Analysis")
        controls_layout = QFormLayout(controls_group)
        
        # Number of components to show
        self.pca_n_components = QSpinBox()
        self.pca_n_components.setRange(1, 10)
        self.pca_n_components.setValue(3)
        self.pca_n_components.valueChanged.connect(self.update_pca_components_plot)
        controls_layout.addRow("Number of Components:", self.pca_n_components)
        
        layout.addWidget(controls_group)
        
        # Create PCA components figure
        self.pca_fig = Figure(figsize=(10, 8))
        self.pca_canvas = FigureCanvas(self.pca_fig)
        layout.addWidget(self.pca_canvas)
        
        # Add toolbar
        self.pca_toolbar = NavigationToolbar(self.pca_canvas, pca_widget)
        layout.addWidget(self.pca_toolbar)
        
        self.tab_widget.addTab(pca_widget, "PCA Components")

    def create_analysis_tab(self):
        """Create the analysis results tab."""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        
        # Create results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        # Results text area
        self.analysis_results_text = QTextEdit()
        self.analysis_results_text.setReadOnly(True)
        results_layout.addWidget(self.analysis_results_text)
        
        layout.addWidget(results_group)
        
        # Export button
        export_results_btn = QPushButton("Export Analysis Results")
        export_results_btn.clicked.connect(self.export_analysis_results)
        layout.addWidget(export_results_btn)
        
        self.tab_widget.addTab(analysis_widget, "Analysis")

    def create_refinement_tab(self):
        """Create the cluster refinement tab."""
        refinement_widget = QWidget()
        layout = QVBoxLayout(refinement_widget)
        layout.setSpacing(8)  # Reduce overall spacing
        
        # Create controls frame with tighter layout
        controls_group = QGroupBox("Refinement Controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setSpacing(6)  # Tighter spacing within group
        
        # Visualization method control
        viz_frame = QFrame()
        viz_layout = QHBoxLayout(viz_frame)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(8)
        
        viz_layout.addWidget(QLabel("Visualization:"))
        self.refinement_viz_combo = QComboBox()
        self.refinement_viz_combo.addItems(['PCA'])
        # Add UMAP if available
        try:
            import umap
            self.refinement_viz_combo.addItem('UMAP')
        except ImportError:
            pass
        self.refinement_viz_combo.setCurrentText('PCA')
        self.refinement_viz_combo.currentTextChanged.connect(self.update_refinement_plot)
        viz_layout.addWidget(self.refinement_viz_combo)
        
        viz_layout.addStretch()
        
        controls_layout.addWidget(viz_frame)
        
        # Mode control row
        mode_frame = QFrame()
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        
        self.refinement_mode_btn = QPushButton("Start Refinement Mode")
        self.refinement_mode_btn.clicked.connect(self.toggle_refinement_mode)
        mode_layout.addWidget(self.refinement_mode_btn)
        
        self.undo_btn = QPushButton("Undo Last Action")
        self.undo_btn.clicked.connect(self.undo_last_action)
        self.undo_btn.setEnabled(False)
        mode_layout.addWidget(self.undo_btn)
        
        controls_layout.addWidget(mode_frame)
        
        # Split controls group
        split_group = QGroupBox("Split Cluster")
        split_group.setMaximumHeight(80)  # Constrain height
        split_layout = QHBoxLayout(split_group)
        split_layout.setSpacing(6)
        
        split_layout.addWidget(QLabel("Method:"))
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItems(['kmeans', 'hierarchical'])
        self.split_method_combo.setCurrentText('kmeans')
        self.split_method_combo.setMaximumWidth(100)
        split_layout.addWidget(self.split_method_combo)
        
        split_layout.addWidget(QLabel("Subclusters:"))
        self.n_subclusters_spinbox = QSpinBox()
        self.n_subclusters_spinbox.setRange(2, 10)
        self.n_subclusters_spinbox.setValue(2)
        self.n_subclusters_spinbox.setMaximumWidth(60)
        split_layout.addWidget(self.n_subclusters_spinbox)
        
        split_layout.addStretch()
        
        self.split_btn = QPushButton("Split Selected Cluster")
        self.split_btn.clicked.connect(self.split_selected_cluster)
        self.split_btn.setEnabled(False)
        split_layout.addWidget(self.split_btn)
        
        controls_layout.addWidget(split_group)
        
        # Cluster operations row
        operations_frame = QFrame()
        operations_layout = QHBoxLayout(operations_frame)
        operations_layout.setContentsMargins(0, 0, 0, 0)
        operations_layout.setSpacing(8)
        
        self.merge_btn = QPushButton("Merge Selected Clusters")
        self.merge_btn.clicked.connect(self.merge_selected_clusters)
        self.merge_btn.setEnabled(False)
        operations_layout.addWidget(self.merge_btn)
        
        self.reset_selection_btn = QPushButton("Reset Selection")
        self.reset_selection_btn.clicked.connect(self.reset_selection)
        self.reset_selection_btn.setEnabled(False)
        operations_layout.addWidget(self.reset_selection_btn)
        
        operations_layout.addStretch()
        
        controls_layout.addWidget(operations_frame)
        
        # Apply/Cancel row
        action_frame = QFrame()
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        
        self.apply_refinement_btn = QPushButton("Apply Refinement")
        self.apply_refinement_btn.clicked.connect(self.apply_refinement)
        self.apply_refinement_btn.setEnabled(False)
        self.apply_refinement_btn.setStyleSheet("""
            QPushButton {
                background-color: #5CB85C;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #449D44;
            }
            QPushButton:disabled {
                background-color: #CCC;
                color: #888;
            }
        """)
        action_layout.addWidget(self.apply_refinement_btn)
        
        self.cancel_refinement_btn = QPushButton("Cancel Refinement")
        self.cancel_refinement_btn.clicked.connect(self.cancel_refinement)
        self.cancel_refinement_btn.setEnabled(False)
        self.cancel_refinement_btn.setStyleSheet("""
            QPushButton {
                background-color: #D9534F;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C9302C;
            }
            QPushButton:disabled {
                background-color: #CCC;
                color: #888;
            }
        """)
        action_layout.addWidget(self.cancel_refinement_btn)
        
        action_layout.addStretch()
        
        controls_layout.addWidget(action_frame)
        
        layout.addWidget(controls_group)
        
        # Selection status with better styling
        status_frame = QFrame()
        status_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; }")
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(8, 4, 8, 4)
        
        status_icon = QLabel("ⓘ")
        status_icon.setStyleSheet("color: #007bff; font-weight: bold; font-size: 12px;")
        status_layout.addWidget(status_icon)
        
        self.selection_status = QLabel("No clusters selected")
        self.selection_status.setStyleSheet("color: #495057; font-size: 11px;")
        status_layout.addWidget(self.selection_status)
        
        status_layout.addStretch()
        
        layout.addWidget(status_frame)
        
        # Create refinement visualization with minimal spacing
        self.refinement_fig = Figure(figsize=(10, 6))
        self.refinement_ax = self.refinement_fig.add_subplot(111)
        self.refinement_canvas = FigureCanvas(self.refinement_fig)
        layout.addWidget(self.refinement_canvas)
        
        # Add toolbar
        self.refinement_toolbar = NavigationToolbar(self.refinement_canvas, refinement_widget)
        layout.addWidget(self.refinement_toolbar)
        
        # Initialize refinement state
        self.selected_points = set()
        self.update_refinement_controls()
        
        self.tab_widget.addTab(refinement_widget, "Refinement")

    def create_time_series_tab(self):
        """Create time-series progression analysis tab."""
        time_series_widget = QWidget()
        layout = QVBoxLayout(time_series_widget)
        
        # Title and description
        title_label = QLabel("Time-Series Progression Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        desc_label = QLabel("Analyze temporal cluster ordering and progression pathways")
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Controls section
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QFormLayout(controls_group)
        
        # Temporal ordering method
        self.temporal_ordering_combo = QComboBox()
        self.temporal_ordering_combo.addItems(['UMAP Distance', 'Cluster Centroid Distance', 'Manual Ordering'])
        controls_layout.addRow("Temporal Ordering Method:", self.temporal_ordering_combo)
        
        # Reference cluster for progression
        self.reference_cluster_spinbox = QSpinBox()
        self.reference_cluster_spinbox.setRange(1, 20)
        self.reference_cluster_spinbox.setValue(1)
        controls_layout.addRow("Reference Cluster (Start):", self.reference_cluster_spinbox)
        
        layout.addWidget(controls_group)
        
        # Analysis buttons
        buttons_layout = QHBoxLayout()
        
        analyze_progression_btn = QPushButton("Analyze Progression")
        analyze_progression_btn.clicked.connect(self.analyze_time_series_progression)
        buttons_layout.addWidget(analyze_progression_btn)
        
        calculate_distances_btn = QPushButton("Calculate Inter-cluster Distances")
        calculate_distances_btn.clicked.connect(self.calculate_intercluster_distances)
        buttons_layout.addWidget(calculate_distances_btn)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Results visualization
        self.time_series_fig = Figure(figsize=(12, 8))
        self.time_series_canvas = FigureCanvas(self.time_series_fig)
        layout.addWidget(self.time_series_canvas)
        
        # Add toolbar
        self.time_series_toolbar = NavigationToolbar(self.time_series_canvas, time_series_widget)
        layout.addWidget(self.time_series_toolbar)
        
        # Results text
        self.time_series_results = QTextEdit()
        self.time_series_results.setMaximumHeight(150)
        self.time_series_results.setReadOnly(True)
        layout.addWidget(self.time_series_results)
        
        self.tab_widget.addTab(time_series_widget, "Time-Series")

    def create_kinetics_tab(self):
        """Create kinetics modeling tab."""
        kinetics_widget = QWidget()
        layout = QVBoxLayout(kinetics_widget)
        
        # Compact title and description on same line
        title_layout = QHBoxLayout()
        title_label = QLabel("Exchange Kinetics Modeling")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_layout.addWidget(title_label)
        
        desc_label = QLabel("- Fit cluster populations to kinetic models for rate constants and mechanisms")
        desc_label.setStyleSheet("color: #666; font-style: italic;")
        title_layout.addWidget(desc_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Combined controls section
        controls_group = QGroupBox("Kinetic Model & Time Series Parameters")
        controls_layout = QVBoxLayout(controls_group)
        
        # Kinetic model parameters row
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.kinetic_model_combo = QComboBox()
        self.kinetic_model_combo.addItems([
            'Pseudo-first-order', 
            'Pseudo-second-order',
            'Avrami Equation',
            'Diffusion-controlled',
            'Multi-exponential'
        ])
        model_layout.addWidget(self.kinetic_model_combo)
        
        model_layout.addWidget(QLabel("Units:"))
        self.time_units_edit = QLineEdit("hours")
        self.time_units_edit.setMaximumWidth(80)
        model_layout.addWidget(self.time_units_edit)
        
        model_layout.addWidget(QLabel("Data Source:"))
        self.data_input_combo = QComboBox()
        self.data_input_combo.addItems(['Constant Interval', 'Load from File', 'Extract from Metadata'])
        model_layout.addWidget(self.data_input_combo)
        
        model_layout.addStretch()
        controls_layout.addLayout(model_layout)
        
        # Interval input controls (shown when Constant Interval is selected)
        self.interval_controls = QWidget()
        interval_layout = QHBoxLayout(self.interval_controls)
        
        interval_layout.addWidget(QLabel("Start Time:"))
        self.start_time_spinbox = QDoubleSpinBox()
        self.start_time_spinbox.setRange(-999999, 999999)
        self.start_time_spinbox.setValue(0.0)
        self.start_time_spinbox.setDecimals(3)
        self.start_time_spinbox.setMaximumWidth(100)
        interval_layout.addWidget(self.start_time_spinbox)
        
        interval_layout.addWidget(QLabel("Interval:"))
        self.time_interval_spinbox = QDoubleSpinBox()
        self.time_interval_spinbox.setRange(0.001, 999999)
        self.time_interval_spinbox.setValue(1.0)
        self.time_interval_spinbox.setDecimals(3)
        self.time_interval_spinbox.setMaximumWidth(100)
        interval_layout.addWidget(self.time_interval_spinbox)
        
        interval_layout.addWidget(QLabel("Spectra:"))
        self.spectra_count_label = QLabel("No data loaded")
        self.spectra_count_label.setStyleSheet("color: #666; font-style: italic;")
        interval_layout.addWidget(self.spectra_count_label)
        
        interval_layout.addStretch()
        controls_layout.addWidget(self.interval_controls)
        
        # Connect data input method change to show/hide controls
        self.data_input_combo.currentTextChanged.connect(self.update_time_input_controls)
        
        # Connect spinbox changes to update preview
        self.start_time_spinbox.valueChanged.connect(self.update_time_input_controls)
        self.time_interval_spinbox.valueChanged.connect(self.update_time_input_controls)
        
        # Action buttons row
        buttons_layout = QHBoxLayout()
        
        input_time_btn = QPushButton("Generate/Load Time Data")
        input_time_btn.clicked.connect(self.input_time_data)
        buttons_layout.addWidget(input_time_btn)
        
        fit_kinetics_btn = QPushButton("Fit Kinetic Models")
        fit_kinetics_btn.clicked.connect(self.fit_kinetic_models)
        buttons_layout.addWidget(fit_kinetics_btn)
        
        compare_models_btn = QPushButton("Compare Models")
        compare_models_btn.clicked.connect(self.compare_kinetic_models)
        buttons_layout.addWidget(compare_models_btn)
        
        buttons_layout.addStretch()
        controls_layout.addLayout(buttons_layout)
        
        # Time data display (compact)
        self.time_data_display = QTextEdit()
        self.time_data_display.setMaximumHeight(60)
        self.time_data_display.setPlaceholderText("Time data will appear here...")
        controls_layout.addWidget(self.time_data_display)
        
        layout.addWidget(controls_group)
        
        # Initialize UI controls visibility
        self.update_time_input_controls()
        
        # Results visualization - ensure proper size allocation
        self.kinetics_fig = Figure(figsize=(12, 6))
        self.kinetics_canvas = FigureCanvas(self.kinetics_fig)
        self.kinetics_canvas.setMinimumHeight(400)  # Ensure minimum height for visibility
        layout.addWidget(self.kinetics_canvas)
        
        # Add toolbar
        self.kinetics_toolbar = NavigationToolbar(self.kinetics_canvas, kinetics_widget)
        layout.addWidget(self.kinetics_toolbar)
        
        # Results text (compact)
        self.kinetics_results = QTextEdit()
        self.kinetics_results.setMaximumHeight(100)
        self.kinetics_results.setReadOnly(True)
        layout.addWidget(self.kinetics_results)
        
        self.tab_widget.addTab(kinetics_widget, "Kinetics")

    def create_structural_analysis_tab(self):
        """Create structural characterization tab."""
        structural_widget = QWidget()
        layout = QHBoxLayout(structural_widget)  # Changed to horizontal layout
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)  # Limit width to give more space to graphs
        left_layout = QVBoxLayout(left_panel)
        
        # Title and description
        title_label = QLabel("Structural Characterization Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(title_label)
        
        desc_label = QLabel("Analyze chemical environments and coordination changes - Customizable for any mineral system")
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        desc_label.setWordWrap(True)
        left_layout.addWidget(desc_label)
        
        # Preset systems and custom configuration
        preset_group = QGroupBox("System Presets & Configuration")
        preset_layout = QVBoxLayout(preset_group)
        
        # Preset selection row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset System:"))
        
        self.system_preset_combo = QComboBox()
        self.system_preset_combo.addItems([
            'Custom Configuration',
            'Zeolites (Cation Exchange)', 
            'Feldspars (Al-Si Ordering)',
            'Pyroxenes (Fe-Mg Substitution)',
            'Clay Minerals (Interlayer Exchange)',
            'Olivine (Fe-Mg Exchange)',
            'Garnet (Cation Substitution)',
            'Carbonates (Mg-Ca Exchange)',
            'Spinels (Cation Ordering)'
        ])
        self.system_preset_combo.currentTextChanged.connect(self.load_system_preset)
        preset_row.addWidget(self.system_preset_combo)
        preset_layout.addLayout(preset_row)
        
        # Preset action buttons
        preset_buttons = QHBoxLayout()
        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.clicked.connect(self.load_system_preset)
        preset_buttons.addWidget(load_preset_btn)
        
        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self.save_custom_configuration)
        preset_buttons.addWidget(save_config_btn)
        
        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self.load_custom_configuration)
        preset_buttons.addWidget(load_config_btn)
        
        preset_layout.addLayout(preset_buttons)
        
        # System description
        self.system_description = QLabel("Select a preset system or configure custom spectral regions")
        self.system_description.setStyleSheet("font-style: italic; color: #555; padding: 5px;")
        self.system_description.setWordWrap(True)
        preset_layout.addWidget(self.system_description)
        
        left_layout.addWidget(preset_group)
        
        # Dynamic spectral regions configuration
        regions_group = QGroupBox("Spectral Regions of Interest (cm⁻¹)")
        regions_layout = QVBoxLayout(regions_group)
        
        # Control buttons for regions
        region_controls = QHBoxLayout()
        
        add_region_btn = QPushButton("Add Region")
        add_region_btn.clicked.connect(self.add_spectral_region)
        region_controls.addWidget(add_region_btn)
        
        remove_region_btn = QPushButton("Remove Selected")
        remove_region_btn.clicked.connect(self.remove_spectral_region)
        region_controls.addWidget(remove_region_btn)
        
        clear_regions_btn = QPushButton("Clear All")
        clear_regions_btn.clicked.connect(self.clear_spectral_regions)
        region_controls.addWidget(clear_regions_btn)
        
        regions_layout.addLayout(region_controls)
        
        # Horizontally scrollable area for regions
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)  # Reduced height
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.regions_widget = QWidget()
        self.regions_widget.setMinimumWidth(720)  # Accommodate wider region widgets
        self.regions_layout = QVBoxLayout(self.regions_widget)
        self.regions_layout.setSpacing(5)
        
        scroll_area.setWidget(self.regions_widget)
        regions_layout.addWidget(scroll_area)
        
        # Store region widgets for dynamic management
        self.region_widgets = []
        
        left_layout.addWidget(regions_group)
        
        # Biofilm/COM Analysis Section
        biofilm_group = QGroupBox("Biofilm & COM Analysis")
        biofilm_layout = QVBoxLayout(biofilm_group)
        
        # Enable biofilm analysis
        self.enable_biofilm_analysis_cb = QCheckBox("Enable COM vs Bacterial Analysis")
        self.enable_biofilm_analysis_cb.setChecked(False)
        self.enable_biofilm_analysis_cb.stateChanged.connect(self.update_biofilm_controls)
        biofilm_layout.addWidget(self.enable_biofilm_analysis_cb)
        
        # COM peak settings
        com_frame = QFrame()
        com_layout = QFormLayout(com_frame)
        
        self.com_peak1_spinbox = QDoubleSpinBox()
        self.com_peak1_spinbox.setRange(1200, 1400)
        self.com_peak1_spinbox.setValue(1320.0)
        self.com_peak1_spinbox.setSuffix(" cm⁻¹")
        com_layout.addRow("COM Peak 1 (C-O stretch):", self.com_peak1_spinbox)
        
        self.com_peak2_spinbox = QDoubleSpinBox()
        self.com_peak2_spinbox.setRange(1500, 1700)
        self.com_peak2_spinbox.setValue(1620.0)
        self.com_peak2_spinbox.setSuffix(" cm⁻¹")
        com_layout.addRow("COM Peak 2 (C=O stretch):", self.com_peak2_spinbox)
        
        # Bacterial signature settings
        self.protein_peak_spinbox = QDoubleSpinBox()
        self.protein_peak_spinbox.setRange(1600, 1700)
        self.protein_peak_spinbox.setValue(1655.0)
        self.protein_peak_spinbox.setSuffix(" cm⁻¹")
        com_layout.addRow("Bacterial Protein (Amide I):", self.protein_peak_spinbox)
        
        self.lipid_peak_spinbox = QDoubleSpinBox()
        self.lipid_peak_spinbox.setRange(1400, 1500)
        self.lipid_peak_spinbox.setValue(1450.0)
        self.lipid_peak_spinbox.setSuffix(" cm⁻¹")
        com_layout.addRow("Bacterial Lipid (CH₂):", self.lipid_peak_spinbox)
        
        # Peak window size
        self.peak_window_spinbox = QDoubleSpinBox()
        self.peak_window_spinbox.setRange(5.0, 50.0)
        self.peak_window_spinbox.setValue(15.0)
        self.peak_window_spinbox.setSuffix(" cm⁻¹")
        com_layout.addRow("Peak Window (±):", self.peak_window_spinbox)
        
        biofilm_layout.addWidget(com_frame)
        
        # Biofilm analysis buttons
        biofilm_buttons = QHBoxLayout()
        
        analyze_biofilm_btn = QPushButton("Analyze COM vs Bacterial Ratios")
        analyze_biofilm_btn.clicked.connect(self.analyze_com_bacterial_ratios)
        biofilm_buttons.addWidget(analyze_biofilm_btn)
        
        correlate_clusters_btn = QPushButton("Correlate with Clusters")
        correlate_clusters_btn.clicked.connect(self.correlate_ratios_with_clusters)
        biofilm_buttons.addWidget(correlate_clusters_btn)
        
        biofilm_layout.addLayout(biofilm_buttons)
        
        # Initially hide biofilm controls
        com_frame.setVisible(False)
        analyze_biofilm_btn.setVisible(False)
        correlate_clusters_btn.setVisible(False)
        
        # Store references for visibility control
        self.biofilm_controls = [com_frame, analyze_biofilm_btn, correlate_clusters_btn]
        
        left_layout.addWidget(biofilm_group)

        # Analysis controls
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.calculate_mean_spectra_cb = QCheckBox("Calculate Cluster Mean Spectra")
        self.calculate_mean_spectra_cb.setChecked(True)
        analysis_layout.addWidget(self.calculate_mean_spectra_cb)
        
        self.track_peak_shifts_cb = QCheckBox("Track Peak Shifts")
        self.track_peak_shifts_cb.setChecked(True)
        analysis_layout.addWidget(self.track_peak_shifts_cb)
        
        self.analyze_peak_ratios_cb = QCheckBox("Analyze Peak Intensity Ratios")
        self.analyze_peak_ratios_cb.setChecked(True)
        analysis_layout.addWidget(self.analyze_peak_ratios_cb)
        
        left_layout.addWidget(analysis_group)
        
        # Analysis buttons
        buttons_layout = QVBoxLayout()
        
        analyze_structure_btn = QPushButton("Analyze Structural Changes")
        analyze_structure_btn.clicked.connect(self.analyze_structural_changes)
        buttons_layout.addWidget(analyze_structure_btn)
        
        calc_differential_btn = QPushButton("Calculate Differential Spectra")
        calc_differential_btn.clicked.connect(self.calculate_differential_spectra)
        buttons_layout.addWidget(calc_differential_btn)
        
        left_layout.addLayout(buttons_layout)
        
        # Results text (compact)
        self.structural_results = QTextEdit()
        self.structural_results.setMaximumHeight(120)
        self.structural_results.setReadOnly(True)
        left_layout.addWidget(self.structural_results)
        
        left_layout.addStretch()
        
        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results visualization - much larger now
        self.structural_fig = Figure(figsize=(14, 10))
        self.structural_canvas = FigureCanvas(self.structural_fig)
        self.structural_canvas.setMinimumHeight(500)  # Ensure good visibility
        right_layout.addWidget(self.structural_canvas)
        
        # Add toolbar
        self.structural_toolbar = NavigationToolbar(self.structural_canvas, structural_widget)
        right_layout.addWidget(self.structural_toolbar)
        
        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        
        # Initialize with default regions
        self.initialize_default_regions()
        
        self.tab_widget.addTab(structural_widget, "Structural Analysis")

    def create_validation_tab(self):
        """Create quantitative cluster validation tab."""
        validation_widget = QWidget()
        layout = QVBoxLayout(validation_widget)
        
        # Title and description
        title_label = QLabel("Quantitative Cluster Validation")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        desc_label = QLabel("Statistical validation and quality assessment of cluster assignments")
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Validation methods
        methods_group = QGroupBox("Validation Methods")
        methods_layout = QVBoxLayout(methods_group)
        
        # Silhouette analysis
        silhouette_frame = QFrame()
        silhouette_layout = QHBoxLayout(silhouette_frame)
        
        self.silhouette_analysis_cb = QCheckBox("Silhouette Analysis")
        self.silhouette_analysis_cb.setChecked(True)
        silhouette_layout.addWidget(self.silhouette_analysis_cb)
        
        silhouette_btn = QPushButton("Calculate Silhouette Scores")
        silhouette_btn.clicked.connect(self.calculate_silhouette_analysis)
        silhouette_layout.addWidget(silhouette_btn)
        
        silhouette_layout.addStretch()
        methods_layout.addWidget(silhouette_frame)
        
        # Cluster transition analysis
        transition_frame = QFrame()
        transition_layout = QHBoxLayout(transition_frame)
        
        self.transition_analysis_cb = QCheckBox("Cluster Transition Analysis")
        self.transition_analysis_cb.setChecked(True)
        transition_layout.addWidget(self.transition_analysis_cb)
        
        transition_btn = QPushButton("Analyze Cluster Boundaries")
        transition_btn.clicked.connect(self.analyze_cluster_transitions)
        transition_layout.addWidget(transition_btn)
        
        transition_layout.addStretch()
        methods_layout.addWidget(transition_frame)
        
        # Stability analysis
        stability_frame = QFrame()
        stability_layout = QHBoxLayout(stability_frame)
        
        self.stability_analysis_cb = QCheckBox("Cluster Stability Analysis")
        self.stability_analysis_cb.setChecked(True)
        stability_layout.addWidget(self.stability_analysis_cb)
        
        stability_btn = QPushButton("Test Cluster Stability")
        stability_btn.clicked.connect(self.test_cluster_stability)
        stability_layout.addWidget(stability_btn)
        
        stability_layout.addStretch()
        methods_layout.addWidget(stability_frame)
        
        layout.addWidget(methods_group)
        
        # Parameters
        params_group = QGroupBox("Validation Parameters")
        params_layout = QFormLayout(params_group)
        
        self.min_silhouette_threshold = QDoubleSpinBox()
        self.min_silhouette_threshold.setRange(0.0, 1.0)
        self.min_silhouette_threshold.setSingleStep(0.1)
        self.min_silhouette_threshold.setValue(0.7)
        params_layout.addRow("Min Silhouette Threshold:", self.min_silhouette_threshold)
        
        self.bootstrap_iterations = QSpinBox()
        self.bootstrap_iterations.setRange(10, 1000)
        self.bootstrap_iterations.setValue(100)
        params_layout.addRow("Bootstrap Iterations:", self.bootstrap_iterations)
        
        layout.addWidget(params_group)
        
        # Run all validation button
        run_all_btn = QPushButton("Run Complete Validation Suite")
        run_all_btn.clicked.connect(self.run_complete_validation)
        run_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #5CB85C;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #449D44;
            }
        """)
        layout.addWidget(run_all_btn)
        
        # Results visualization
        self.validation_fig = Figure(figsize=(12, 10))
        self.validation_canvas = FigureCanvas(self.validation_fig)
        layout.addWidget(self.validation_canvas)
        
        # Add toolbar
        self.validation_toolbar = NavigationToolbar(self.validation_canvas, validation_widget)
        layout.addWidget(self.validation_toolbar)
        
        # Results text
        self.validation_results = QTextEdit()
        self.validation_results.setMaximumHeight(150)
        self.validation_results.setReadOnly(True)
        layout.addWidget(self.validation_results)
        
        self.tab_widget.addTab(validation_widget, "Validation")

    def create_advanced_statistics_tab(self):
        """Create advanced statistical analysis tab."""
        stats_widget = QWidget()
        layout = QVBoxLayout(stats_widget)
        
        # Title and description
        title_label = QLabel("Advanced Statistical Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        desc_label = QLabel("Feature importance, discriminant analysis, and statistical significance testing")
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Analysis options
        options_group = QGroupBox("Statistical Analysis Options")
        options_layout = QVBoxLayout(options_group)
        
        # Feature importance
        feature_frame = QFrame()
        feature_layout = QHBoxLayout(feature_frame)
        
        self.feature_importance_cb = QCheckBox("Feature Importance Analysis")
        self.feature_importance_cb.setChecked(True)
        feature_layout.addWidget(self.feature_importance_cb)
        
        self.feature_method_combo = QComboBox()
        self.feature_method_combo.addItems(['Random Forest', 'Linear Discriminant Analysis', 'Mutual Information'])
        feature_layout.addWidget(self.feature_method_combo)
        
        feature_btn = QPushButton("Calculate Feature Importance")
        feature_btn.clicked.connect(self.calculate_feature_importance)
        feature_layout.addWidget(feature_btn)
        
        feature_layout.addStretch()
        options_layout.addWidget(feature_frame)
        
        # Discriminant analysis
        discriminant_frame = QFrame()
        discriminant_layout = QHBoxLayout(discriminant_frame)
        
        self.discriminant_analysis_cb = QCheckBox("Discriminant Analysis")
        self.discriminant_analysis_cb.setChecked(True)
        discriminant_layout.addWidget(self.discriminant_analysis_cb)
        
        discriminant_btn = QPushButton("Perform Discriminant Analysis")
        discriminant_btn.clicked.connect(self.perform_discriminant_analysis)
        discriminant_layout.addWidget(discriminant_btn)
        
        discriminant_layout.addStretch()
        options_layout.addWidget(discriminant_frame)
        
        # Statistical significance
        significance_frame = QFrame()
        significance_layout = QHBoxLayout(significance_frame)
        
        self.significance_testing_cb = QCheckBox("Statistical Significance Testing")
        self.significance_testing_cb.setChecked(True)
        significance_layout.addWidget(self.significance_testing_cb)
        
        self.significance_method_combo = QComboBox()
        self.significance_method_combo.addItems(['PERMANOVA', 'ANOSIM', 'Kruskal-Wallis', 'ANOVA'])
        significance_layout.addWidget(self.significance_method_combo)
        
        significance_btn = QPushButton("Test Significance")
        significance_btn.clicked.connect(self.test_statistical_significance)
        significance_layout.addWidget(significance_btn)
        
        significance_layout.addStretch()
        options_layout.addWidget(significance_frame)
        
        layout.addWidget(options_group)
        
        # Parameters
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout(params_group)
        
        self.significance_level = QDoubleSpinBox()
        self.significance_level.setRange(0.001, 0.1)
        self.significance_level.setSingleStep(0.001)
        self.significance_level.setValue(0.05)
        self.significance_level.setDecimals(3)
        params_layout.addRow("Significance Level (α):", self.significance_level)
        
        self.permutations = QSpinBox()
        self.permutations.setRange(100, 10000)
        self.permutations.setValue(999)
        params_layout.addRow("Permutations:", self.permutations)
        
        layout.addWidget(params_group)
        
        # Run comprehensive analysis
        comprehensive_btn = QPushButton("Run Comprehensive Statistical Analysis")
        comprehensive_btn.clicked.connect(self.run_comprehensive_statistics)
        comprehensive_btn.setStyleSheet("""
            QPushButton {
                background-color: #337AB7;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #286090;
            }
        """)
        layout.addWidget(comprehensive_btn)
        
        # Results visualization
        self.stats_fig = Figure(figsize=(12, 10))
        self.stats_canvas = FigureCanvas(self.stats_fig)
        layout.addWidget(self.stats_canvas)
        
        # Add toolbar
        self.stats_toolbar = NavigationToolbar(self.stats_canvas, stats_widget)
        layout.addWidget(self.stats_toolbar)
        
        # Results text
        self.stats_results = QTextEdit()
        self.stats_results.setMaximumHeight(150)
        self.stats_results.setReadOnly(True)
        layout.addWidget(self.stats_results)
        
        self.tab_widget.addTab(stats_widget, "Advanced Statistics")

    # Enhanced Structural Analysis Methods for General Mineral Systems
    def initialize_default_regions(self):
        """Initialize with default generic regions."""
        default_regions = [
            {"name": "Low Frequency", "min_wn": "100", "max_wn": "400", "description": "Lattice and framework vibrations"},
            {"name": "Medium Frequency", "min_wn": "400", "max_wn": "800", "description": "Bending and deformation modes"},
            {"name": "High Frequency", "min_wn": "800", "max_wn": "1200", "description": "Stretching vibrations"},
            {"name": "OH/NH Region", "min_wn": "3000", "max_wn": "3800", "description": "Hydroxyl and amine groups"}
        ]
        
        for region in default_regions:
            self.add_spectral_region_widget(region)

    def get_system_presets(self):
        """Get predefined system configurations."""
        presets = {
            'Zeolites (Cation Exchange)': {
                'description': 'Cation exchange in zeolites: K⁺, Na⁺, Ca²⁺, Mg²⁺ substitutions affecting framework and water coordination',
                'regions': [
                    {"name": "Framework T-O-T", "min_wn": "400", "max_wn": "600", "description": "Tetrahedral framework vibrations"},
                    {"name": "T-O Stretching", "min_wn": "900", "max_wn": "1200", "description": "Si-O and Al-O stretching modes"},
                    {"name": "Cation-O Modes", "min_wn": "200", "max_wn": "400", "description": "Cation-oxygen coordination"},
                    {"name": "Water Modes", "min_wn": "3200", "max_wn": "3600", "description": "H₂O stretching vibrations"},
                    {"name": "OH Modes", "min_wn": "3600", "max_wn": "3800", "description": "Hydroxyl group vibrations"}
                ]
            },
            'Feldspars (Al-Si Ordering)': {
                'description': 'Al-Si ordering in feldspars: Tetrahedral site occupancy changes, framework distortion',
                'regions': [
                    {"name": "T-O-T Bending", "min_wn": "400", "max_wn": "550", "description": "Tetrahedral framework bending"},
                    {"name": "Al-O Stretching", "min_wn": "700", "max_wn": "800", "description": "Al-O tetrahedral stretching"},
                    {"name": "Si-O Stretching", "min_wn": "950", "max_wn": "1200", "description": "Si-O tetrahedral stretching"},
                    {"name": "Framework Modes", "min_wn": "150", "max_wn": "350", "description": "Overall framework vibrations"}
                ]
            },
            'Pyroxenes (Fe-Mg Substitution)': {
                'description': 'Fe²⁺-Mg²⁺ substitution in pyroxenes: M1/M2 site occupancy, chain distortion effects',
                'regions': [
                    {"name": "Si-O Stretching", "min_wn": "900", "max_wn": "1100", "description": "Silicate chain stretching"},
                    {"name": "M-O Stretching", "min_wn": "600", "max_wn": "800", "description": "Metal-oxygen stretching"},
                    {"name": "Chain Bending", "min_wn": "400", "max_wn": "600", "description": "Silicate chain bending"},
                    {"name": "M-Site Modes", "min_wn": "200", "max_wn": "400", "description": "M1/M2 site vibrations"},
                    {"name": "Fe-O Modes", "min_wn": "250", "max_wn": "350", "description": "Iron-oxygen coordination"},
                    {"name": "Mg-O Modes", "min_wn": "350", "max_wn": "450", "description": "Magnesium-oxygen coordination"}
                ]
            },
            'Clay Minerals (Interlayer Exchange)': {
                'description': 'Interlayer cation exchange in clays: Hydration changes, layer charge effects, swelling behavior',
                'regions': [
                    {"name": "OH Stretching", "min_wn": "3400", "max_wn": "3700", "description": "Hydroxyl group vibrations"},
                    {"name": "H₂O Stretching", "min_wn": "3000", "max_wn": "3400", "description": "Water molecule stretching"},
                    {"name": "OH Bending", "min_wn": "1600", "max_wn": "1700", "description": "Water bending modes"},
                    {"name": "Si-O Stretching", "min_wn": "950", "max_wn": "1100", "description": "Tetrahedral sheet vibrations"},
                    {"name": "Al-OH Bending", "min_wn": "900", "max_wn": "950", "description": "Octahedral sheet modes"},
                    {"name": "M-O Modes", "min_wn": "400", "max_wn": "800", "description": "Metal-oxygen octahedral"},
                    {"name": "Lattice Modes", "min_wn": "100", "max_wn": "400", "description": "Layer lattice vibrations"}
                ]
            },
            'Olivine (Fe-Mg Exchange)': {
                'description': 'Fe²⁺-Mg²⁺ exchange in olivine: Forsterite-fayalite solid solution, M1/M2 site preferences',
                'regions': [
                    {"name": "Si-O Stretching", "min_wn": "800", "max_wn": "900", "description": "Silicate tetrahedra stretching"},
                    {"name": "M-O Stretching", "min_wn": "500", "max_wn": "650", "description": "Metal-oxygen octahedral"},
                    {"name": "Si-O Bending", "min_wn": "400", "max_wn": "500", "description": "Tetrahedral bending modes"},
                    {"name": "Fe-O Modes", "min_wn": "200", "max_wn": "350", "description": "Iron-oxygen coordination"},
                    {"name": "Mg-O Modes", "min_wn": "350", "max_wn": "450", "description": "Magnesium-oxygen coordination"}
                ]
            },
            'Garnet (Cation Substitution)': {
                'description': 'Complex cation substitutions in garnets: Dodecahedral (X), octahedral (Y), tetrahedral (Z) site changes',
                'regions': [
                    {"name": "Si-O Stretching", "min_wn": "800", "max_wn": "1000", "description": "Tetrahedral SiO₄ stretching"},
                    {"name": "Y-O Stretching", "min_wn": "500", "max_wn": "700", "description": "Octahedral site vibrations"},
                    {"name": "X-O Stretching", "min_wn": "300", "max_wn": "500", "description": "Dodecahedral site modes"},
                    {"name": "Framework Modes", "min_wn": "150", "max_wn": "350", "description": "Overall framework vibrations"},
                    {"name": "Fe³⁺-O Modes", "min_wn": "400", "max_wn": "550", "description": "Ferric iron coordination"},
                    {"name": "Al-O Modes", "min_wn": "550", "max_wn": "650", "description": "Aluminum-oxygen coordination"}
                ]
            },
            'Carbonates (Mg-Ca Exchange)': {
                'description': 'Mg²⁺-Ca²⁺ substitution in carbonates: Calcite-magnesite-dolomite series, structural distortion',
                'regions': [
                    {"name": "CO₃ Symmetric", "min_wn": "1080", "max_wn": "1120", "description": "Symmetric CO₃ stretching"},
                    {"name": "CO₃ Antisymmetric", "min_wn": "1400", "max_wn": "1500", "description": "Antisymmetric CO₃ stretching"},
                    {"name": "CO₃ Bending", "min_wn": "700", "max_wn": "750", "description": "In-plane CO₃ bending"},
                    {"name": "Lattice Modes", "min_wn": "150", "max_wn": "350", "description": "Lattice vibrations"},
                    {"name": "Ca-O Modes", "min_wn": "200", "max_wn": "300", "description": "Calcium-oxygen coordination"},
                    {"name": "Mg-O Modes", "min_wn": "300", "max_wn": "400", "description": "Magnesium-oxygen coordination"}
                ]
            },
            'Spinels (Cation Ordering)': {
                'description': 'Cation ordering in spinels: Normal vs inverse spinel structures, tetrahedral/octahedral site preferences',
                'regions': [
                    {"name": "T-O Stretching", "min_wn": "600", "max_wn": "800", "description": "Tetrahedral site metal-oxygen"},
                    {"name": "O-O Stretching", "min_wn": "400", "max_wn": "600", "description": "Octahedral site metal-oxygen"},
                    {"name": "Lattice Modes", "min_wn": "200", "max_wn": "400", "description": "Framework lattice vibrations"},
                    {"name": "Fe³⁺ Modes", "min_wn": "300", "max_wn": "450", "description": "Ferric iron coordination"},
                    {"name": "Fe²⁺ Modes", "min_wn": "250", "max_wn": "350", "description": "Ferrous iron coordination"}
                ]
            }
        }
        return presets

    def load_system_preset(self):
        """Load a predefined system configuration."""
        preset_name = self.system_preset_combo.currentText()
        
        if preset_name == 'Custom Configuration':
            self.system_description.setText("Custom configuration - Add, remove, and modify spectral regions as needed")
            return
        
        presets = self.get_system_presets()
        
        if preset_name in presets:
            preset = presets[preset_name]
            
            # Update description
            self.system_description.setText(preset['description'])
            
            # Clear existing regions
            self.clear_spectral_regions()
            
            # Add preset regions
            for region in preset['regions']:
                self.add_spectral_region_widget(region)

    def add_spectral_region(self):
        """Add a new spectral region widget."""
        region_data = {
            "name": "New Region",
            "min_wn": "100",
            "max_wn": "200", 
            "description": "Custom spectral region"
        }
        self.add_spectral_region_widget(region_data)

    def add_spectral_region_widget(self, region_data):
        """Add a spectral region widget with given data."""
        region_frame = QFrame()
        region_frame.setFrameStyle(QFrame.Box)
        region_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; margin: 2px; padding: 5px; }")
        region_frame.setMinimumWidth(700)  # Ensure minimum width for wider content
        
        region_layout = QHBoxLayout(region_frame)
        region_layout.setSpacing(10)
        
        # Selection checkbox
        checkbox = QCheckBox()
        checkbox.setFixedWidth(20)
        region_layout.addWidget(checkbox)
        
        # Region name
        name_edit = QLineEdit(region_data["name"])
        name_edit.setPlaceholderText("Region name")
        name_edit.setMaximumWidth(120)  # Slightly wider for better readability
        region_layout.addWidget(name_edit)
        
        # Wavenumber range
        region_layout.addWidget(QLabel("Range:"))
        min_edit = QLineEdit(region_data["min_wn"])
        min_edit.setPlaceholderText("Min")
        min_edit.setMaximumWidth(60)
        region_layout.addWidget(min_edit)
        
        region_layout.addWidget(QLabel("-"))
        
        max_edit = QLineEdit(region_data["max_wn"])
        max_edit.setPlaceholderText("Max")
        max_edit.setMaximumWidth(60)
        region_layout.addWidget(max_edit)
        
        # Remove redundant cm⁻¹ label since it's now in group title
        
        # Description - much wider now
        desc_edit = QLineEdit(region_data["description"])
        desc_edit.setPlaceholderText("Chemical/structural description (vibrational modes, assignments, etc.)")
        desc_edit.setMinimumWidth(300)  # Much wider for detailed descriptions
        region_layout.addWidget(desc_edit)
        
        # Store widget references
        region_widget = {
            'frame': region_frame,
            'checkbox': checkbox,
            'name': name_edit,
            'min_wn': min_edit,
            'max_wn': max_edit,
            'description': desc_edit
        }
        
        if not hasattr(self, 'region_widgets'):
            self.region_widgets = []
        
        self.region_widgets.append(region_widget)
        
        if hasattr(self, 'regions_layout'):
            self.regions_layout.addWidget(region_frame)

    def remove_spectral_region(self):
        """Remove selected spectral regions."""
        if not hasattr(self, 'region_widgets'):
            return
            
        regions_to_remove = []
        
        for i, region_widget in enumerate(self.region_widgets):
            if region_widget['checkbox'].isChecked():
                regions_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(regions_to_remove):
            region_widget = self.region_widgets[i]
            if hasattr(self, 'regions_layout'):
                self.regions_layout.removeWidget(region_widget['frame'])
            region_widget['frame'].deleteLater()
            del self.region_widgets[i]

    def clear_spectral_regions(self):
        """Clear all spectral regions."""
        if not hasattr(self, 'region_widgets'):
            return
            
        for region_widget in self.region_widgets:
            if hasattr(self, 'regions_layout'):
                self.regions_layout.removeWidget(region_widget['frame'])
            region_widget['frame'].deleteLater()
        
        self.region_widgets.clear()

    def get_spectral_regions(self):
        """Parse spectral region settings from widgets."""
        regions = {}
        
        try:
            # Check if we have the new dynamic regions
            if hasattr(self, 'region_widgets') and self.region_widgets:
                for region_widget in self.region_widgets:
                    name = region_widget['name'].text().strip()
                    if not name:
                        continue
                    
                    min_wn = float(region_widget['min_wn'].text())
                    max_wn = float(region_widget['max_wn'].text())
                    
                    # Use sanitized name as key
                    key = name.lower().replace(' ', '_').replace('-', '_')
                    regions[key] = (min_wn, max_wn)
                    
                    # Store additional metadata
                    if not hasattr(self, 'region_metadata'):
                        self.region_metadata = {}
                    self.region_metadata[key] = {
                        'display_name': name,
                        'description': region_widget['description'].text().strip()
                    }
            else:
                # Fallback to old hardcoded fields for backward compatibility
                try:
                    # Framework vibrations
                    if hasattr(self, 'framework_range_edit'):
                        framework_range = self.framework_range_edit.text().split('-')
                        regions['framework'] = (float(framework_range[0]), float(framework_range[1]))
                    
                    # Si-O stretching
                    if hasattr(self, 'sio_range_edit'):
                        sio_range = self.sio_range_edit.text().split('-')
                        regions['si_o'] = (float(sio_range[0]), float(sio_range[1]))
                    
                    # Y-O coordination
                    if hasattr(self, 'yo_range_edit'):
                        yo_range = self.yo_range_edit.text().split('-')
                        regions['y_o'] = (float(yo_range[0]), float(yo_range[1]))
                    
                    # Na-O coordination
                    if hasattr(self, 'nao_range_edit'):
                        nao_range = self.nao_range_edit.text().split('-')
                        regions['na_o'] = (float(nao_range[0]), float(nao_range[1]))
                except:
                    pass  # Ignore errors from old interface
                
        except ValueError as e:
            raise Exception(f"Invalid wavenumber format: {str(e)}")
        
        return regions

    def save_custom_configuration(self):
        """Save current region configuration to file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Region Configuration",
                "custom_regions.json",
                "JSON files (*.json);;All files (*.*)"
            )
            
            if file_path:
                import json
                
                config = {
                    'system_name': 'Custom Configuration',
                    'description': self.system_description.text() if hasattr(self, 'system_description') else '',
                    'regions': []
                }
                
                if hasattr(self, 'region_widgets'):
                    for region_widget in self.region_widgets:
                        region_data = {
                            'name': region_widget['name'].text(),
                            'min_wn': region_widget['min_wn'].text(),
                            'max_wn': region_widget['max_wn'].text(),
                            'description': region_widget['description'].text()
                        }
                        config['regions'].append(region_data)
                
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                QMessageBox.information(self, "Configuration Saved", f"Configuration saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration:\n{str(e)}")

    def load_custom_configuration(self):
        """Load region configuration from file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Region Configuration",
                "",
                "JSON files (*.json);;All files (*.*)"
            )
            
            if file_path:
                import json
                
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Update description
                if 'description' in config and hasattr(self, 'system_description'):
                    self.system_description.setText(config['description'])
                
                # Clear existing regions
                self.clear_spectral_regions()
                
                # Load regions
                if 'regions' in config:
                    for region_data in config['regions']:
                        self.add_spectral_region_widget(region_data)
                
                QMessageBox.information(self, "Configuration Loaded", f"Configuration loaded from:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load configuration:\n{str(e)}")

    # Data Import Methods
    def select_import_folder(self):
        """Select folder for importing spectra."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Spectra")
        if folder:
            self.selected_folder = folder
            self.folder_path_label.setText(folder)

    def import_from_main_app(self):
        """Import spectra from the main application."""
        if not self.raman_app:
            QMessageBox.warning(self, "No Main App", "No main application reference available.")
            return
            
        if (not hasattr(self.raman_app, 'current_wavenumbers') or 
            self.raman_app.current_wavenumbers is None):
            QMessageBox.warning(self, "No Data", "No spectrum loaded in main application.")
            return
            
        try:
            # Get current spectrum from main app
            wavenumbers = self.raman_app.current_wavenumbers.copy()
            intensities = self.raman_app.processed_intensities.copy() if self.raman_app.processed_intensities is not None else self.raman_app.current_intensities.copy()
            
            # Create metadata for the main app spectrum
            spectrum_metadata = {
                'filename': 'Main_App_Spectrum',
                'spectrum_index': 0,
                'source': 'main_application'
            }
            
            # Try to get additional metadata from main app if available
            if hasattr(self.raman_app, 'current_filename') and self.raman_app.current_filename:
                spectrum_metadata['filename'] = os.path.basename(self.raman_app.current_filename)
                spectrum_metadata['file_path'] = self.raman_app.current_filename
            
            # Store as single spectrum dataset
            self.cluster_data['wavenumbers'] = wavenumbers
            self.cluster_data['intensities'] = np.array([intensities])
            self.cluster_data['spectrum_metadata'] = [spectrum_metadata]
            
            # Extract features
            features = self.extract_vibrational_features(self.cluster_data['intensities'], wavenumbers)
            self.cluster_data['features'] = features
            
            # Scale features
            scaler = StandardScaler()
            self.cluster_data['features_scaled'] = scaler.fit_transform(features)
            
            # Update UI
            self.update_ui_after_import(1)
            QMessageBox.information(self, "Import Complete", "Successfully imported spectrum from main application.")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import from main app:\n{str(e)}")

    def start_batch_import(self):
        """Start batch import of spectra from selected folder using intelligent parsing."""
        if not self.selected_folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
            
        try:
            # Get file patterns to search for
            patterns = ['*.txt', '*.csv', '*.dat', '*.asc']
            files = []
            for pattern in patterns:
                files.extend(glob.glob(os.path.join(self.selected_folder, pattern)))
            
            if not files:
                QMessageBox.warning(self, "No Files", "No spectrum files found in selected folder.")
                return
            
            # Initialize progress
            self.import_progress.setMaximum(len(files))
            self.import_progress.setValue(0)
            
            # Initialize storage
            all_wavenumbers = []
            all_intensities = []
            all_metadata = []  # Store metadata for each spectrum
            
            # Process files using the robust parsing method
            successful_imports = 0
            failed_files = []
            
            for i, file_path in enumerate(files):
                try:
                    # Use the same robust parsing method as the main app
                    wavenumbers, intensities, metadata = self.parse_spectrum_file(file_path)
                    
                    # Basic validation
                    if len(wavenumbers) < 10 or len(intensities) < 10:
                        failed_files.append(f"{file_path}: Insufficient data points")
                        continue
                    
                    # Extract filename and add to metadata
                    filename = os.path.basename(file_path)
                    metadata['filename'] = filename
                    metadata['file_path'] = file_path
                    metadata['spectrum_index'] = successful_imports
                    
                    all_wavenumbers.append(wavenumbers)
                    all_intensities.append(intensities)
                    all_metadata.append(metadata)
                    successful_imports += 1
                    
                except Exception as e:
                    failed_files.append(f"{file_path}: {str(e)}")
                    continue
                
                # Update progress
                self.import_progress.setValue(i + 1)
                QApplication.processEvents()
            
            if successful_imports == 0:
                error_msg = "No files could be successfully imported.\n\n"
                if failed_files:
                    error_msg += "Errors encountered:\n" + "\n".join(failed_files[:5])
                    if len(failed_files) > 5:
                        error_msg += f"\n... and {len(failed_files) - 5} more errors."
                QMessageBox.warning(self, "Import Failed", error_msg)
                return
            
            # Find common wavenumber range
            min_len = min(len(wn) for wn in all_wavenumbers)
            common_wavenumbers = all_wavenumbers[0][:min_len]
            
            # Interpolate all spectra to common wavenumber grid
            processed_intensities = []
            for wavenumbers, intensities in zip(all_wavenumbers, all_intensities):
                # Simple interpolation - in practice you might want more sophisticated alignment
                if len(intensities) != len(common_wavenumbers):
                    processed_intensities.append(intensities[:min_len])
                else:
                    processed_intensities.append(intensities)
            
            # Store data
            self.cluster_data['wavenumbers'] = common_wavenumbers
            self.cluster_data['intensities'] = np.array(processed_intensities)
            self.cluster_data['spectrum_metadata'] = all_metadata  # Store metadata
            
            # Extract features
            features = self.extract_vibrational_features(self.cluster_data['intensities'], common_wavenumbers)
            self.cluster_data['features'] = features
            
            # Scale features
            scaler = StandardScaler()
            self.cluster_data['features_scaled'] = scaler.fit_transform(features)
            
            # Update UI
            self.update_ui_after_import(successful_imports)
            
            # Show summary message
            summary_msg = f"Successfully imported {successful_imports} out of {len(files)} files."
            if failed_files:
                summary_msg += f"\n\n{len(failed_files)} files failed to import."
                if len(failed_files) <= 3:
                    summary_msg += "\n\nFailed files:\n" + "\n".join(failed_files)
                else:
                    summary_msg += f"\n\nFirst 3 failed files:\n" + "\n".join(failed_files[:3])
                    summary_msg += f"\n... and {len(failed_files) - 3} more."
                    
            QMessageBox.information(self, "Import Complete", summary_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import spectra:\n{str(e)}")

    def parse_spectrum_file(self, file_path):
        """Enhanced spectrum file parser that handles headers, metadata, and various formats.
        
        This is the same robust parsing method used in the main application.
        """
        import csv
        import re
        from pathlib import Path
        
        wavenumbers = []
        intensities = []
        metadata = {}
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                lines = file.readlines()
        
        # First pass: extract metadata and find data start
        data_start_line = 0
        delimiter = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Handle metadata lines starting with #
            if line.startswith('#'):
                self.parse_metadata_line(line, metadata)
                data_start_line = i + 1
                continue
            
            # Check if this looks like a header line (non-numeric first column)
            if self.is_header_line(line):
                data_start_line = i + 1
                continue
            
            # This should be the first data line - detect delimiter
            if delimiter is None:
                delimiter = self.detect_delimiter(line, file_extension)
                break
        
        # Second pass: read the actual data
        for i in range(data_start_line, len(lines)):
            line = lines[i].strip()
            
            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            
            try:
                # Parse the data line
                values = self.parse_data_line(line, delimiter)
                
                if len(values) >= 2:
                    # Convert to float
                    wavenumber = float(values[0])
                    intensity = float(values[1])
                    
                    wavenumbers.append(wavenumber)
                    intensities.append(intensity)
                    
            except (ValueError, IndexError) as e:
                # Skip lines that can't be parsed as numeric data
                continue
        
        # Convert to numpy arrays
        wavenumbers = np.array(wavenumbers)
        intensities = np.array(intensities)
        
        # Add file information to metadata
        metadata['file_path'] = str(file_path)
        metadata['file_name'] = Path(file_path).name
        metadata['data_points'] = len(wavenumbers)
        if len(wavenumbers) > 0:
            metadata['wavenumber_range'] = f"{wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹"
        
        return wavenumbers, intensities, metadata

    def parse_metadata_line(self, line, metadata):
        """Parse a metadata line starting with #."""
        # Remove the # and strip whitespace
        content = line[1:].strip()
        
        if not content:
            return
        
        # Try to parse as key: value or key = value
        for separator in [':', '=']:
            if separator in content:
                parts = content.split(separator, 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    metadata[key] = value
                    return
        
        # If no separator found, store as a general comment
        if 'comments' not in metadata:
            metadata['comments'] = []
        metadata['comments'].append(content)

    def is_header_line(self, line):
        """Check if a line looks like a header (contains non-numeric data in first column)."""
        # Split the line using common delimiters
        for delimiter in [',', '\t', ' ']:
            parts = [part.strip() for part in line.split(delimiter) if part.strip()]
            if len(parts) >= 2:
                try:
                    # Try to convert first two parts to float
                    float(parts[0])
                    float(parts[1])
                    return False  # Successfully parsed as numbers, not a header
                except ValueError:
                    return True  # Can't parse as numbers, likely a header
        
        return False

    def detect_delimiter(self, line, file_extension):
        """Detect the delimiter used in the data file."""
        # For CSV files, prefer comma
        if file_extension == '.csv':
            if ',' in line:
                return ','
        
        # Count occurrences of different delimiters
        comma_count = line.count(',')
        tab_count = line.count('\t')
        space_count = len([x for x in line.split(' ') if x.strip()]) - 1
        
        # Choose delimiter with highest count
        if comma_count > 0 and comma_count >= tab_count and comma_count >= space_count:
            return ','
        elif tab_count > 0 and tab_count >= space_count:
            return '\t'
        else:
            return None  # Will use split() for whitespace

    def parse_data_line(self, line, delimiter):
        """Parse a data line using the detected delimiter."""
        if delimiter == ',':
            # Use CSV reader for proper comma handling
            import csv
            reader = csv.reader([line])
            values = next(reader)
        elif delimiter == '\t':
            values = line.split('\t')
        else:
            # Default to whitespace splitting
            values = line.split()
        
        # Strip whitespace from each value
        return [value.strip() for value in values if value.strip()]

    def open_database_import_dialog(self):
        """Open database import dialog with filtering options."""
        if not self.raman_app or not hasattr(self.raman_app, 'raman_db'):
            QMessageBox.warning(self, "No Database", "No database connection available.")
            return
        
        # Create database import dialog
        dialog = DatabaseImportDialog(self.raman_app.raman_db, parent=self)
        if dialog.exec() == QDialog.Accepted:
            # Get the selected spectra from the dialog
            selected_spectra = dialog.get_selected_spectra()
            if selected_spectra:
                self.import_database_spectra(selected_spectra)

    def import_database_spectra(self, selected_spectra):
        """Import selected spectra from database."""
        try:
            # Initialize progress
            self.import_progress.setMaximum(len(selected_spectra))
            self.import_progress.setValue(0)
            self.import_status.setText("Importing spectra from database...")
            
            # Initialize storage
            all_wavenumbers = []
            all_intensities = []
            all_metadata = []
            
            # Process selected spectra
            successful_imports = 0
            for i, (name, data) in enumerate(selected_spectra.items()):
                try:
                    wavenumbers = np.array(data['wavenumbers'])
                    intensities = np.array(data['intensities'])
                    
                    # Basic validation
                    if len(wavenumbers) < 10 or len(intensities) < 10:
                        continue
                    
                    # Extract metadata from database entry
                    db_metadata = data.get('metadata', {})
                    spectrum_metadata = {
                        'filename': name,
                        'spectrum_index': successful_imports,
                        'source': 'database'
                    }
                    
                    # Add database metadata
                    if 'NAME' in db_metadata:
                        spectrum_metadata['mineral_name'] = db_metadata['NAME']
                    elif 'mineral_name' in db_metadata:
                        spectrum_metadata['mineral_name'] = db_metadata['mineral_name']
                    
                    if 'HEY CLASSIFICATION' in db_metadata:
                        spectrum_metadata['hey_classification'] = db_metadata['HEY CLASSIFICATION']
                    elif 'Hey Classification' in db_metadata:
                        spectrum_metadata['hey_classification'] = db_metadata['Hey Classification']
                    
                    if 'FORMULA' in db_metadata:
                        spectrum_metadata['formula'] = db_metadata['FORMULA']
                    elif 'Formula' in db_metadata:
                        spectrum_metadata['formula'] = db_metadata['Formula']
                    
                    if 'CHEMICAL FAMILY' in db_metadata:
                        spectrum_metadata['chemical_family'] = db_metadata['CHEMICAL FAMILY']
                    elif 'Chemical Family' in db_metadata:
                        spectrum_metadata['chemical_family'] = db_metadata['Chemical Family']
                    
                    all_wavenumbers.append(wavenumbers)
                    all_intensities.append(intensities)
                    all_metadata.append(spectrum_metadata)
                    successful_imports += 1
                    
                except Exception as e:
                    print(f"Error processing spectrum {name}: {str(e)}")
                    continue
                
                # Update progress
                self.import_progress.setValue(i + 1)
                QApplication.processEvents()
            
            if successful_imports == 0:
                QMessageBox.warning(self, "Import Failed", "No spectra could be successfully imported from database.")
                return
            
            # Find the maximum common wavenumber range that covers all spectra
            # Get the overall min and max wavenumbers across all spectra
            all_min_wn = min(wn[0] for wn in all_wavenumbers if len(wn) > 0)
            all_max_wn = max(wn[-1] for wn in all_wavenumbers if len(wn) > 0)
            
            print(f"Full wavenumber range found: {all_min_wn:.1f} - {all_max_wn:.1f} cm⁻¹")
            
            # Create a common wavenumber grid that spans the full range
            # Use the densest sampling available
            max_points = max(len(wn) for wn in all_wavenumbers)
            common_wavenumbers = np.linspace(all_min_wn, all_max_wn, max_points)
            
            print(f"Created common grid: {len(common_wavenumbers)} points from {common_wavenumbers[0]:.1f} to {common_wavenumbers[-1]:.1f} cm⁻¹")
            
            # Interpolate all spectra to the common wavenumber grid
            processed_intensities = []
            for wavenumbers, intensities in zip(all_wavenumbers, all_intensities):
                # Interpolate to common grid
                interpolated_intensities = np.interp(common_wavenumbers, wavenumbers, intensities)
                processed_intensities.append(interpolated_intensities)
            
            # Store data
            self.cluster_data['wavenumbers'] = common_wavenumbers
            self.cluster_data['intensities'] = np.array(processed_intensities)
            self.cluster_data['spectrum_metadata'] = all_metadata
            
            # Extract features
            features = self.extract_vibrational_features(self.cluster_data['intensities'], common_wavenumbers)
            self.cluster_data['features'] = features
            
            # Scale features
            scaler = StandardScaler()
            self.cluster_data['features_scaled'] = scaler.fit_transform(features)
            
            # Update UI
            self.update_ui_after_import(successful_imports)
            
            QMessageBox.information(
                self, 
                "Database Import Complete", 
                f"Successfully imported {successful_imports} spectra from database."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Database Import Error", f"Failed to import from database:\n{str(e)}")

    def append_data(self):
        """Append additional data to existing dataset."""
        QMessageBox.information(self, "Append Data", 
                              "Append data functionality will be implemented.\n\n"
                              "This will allow adding more spectra to the existing dataset.")

    # Clustering Methods
    def run_probabilistic_clustering(self):
        """Run probabilistic clustering with GMM and hierarchical sub-typing."""
        if not hasattr(self, 'cluster_data') or self.cluster_data.get('features_scaled') is None:
            QMessageBox.warning(self, "No Data", "Please import and preprocess data first.")
            return
        
        try:
            features = self.cluster_data['features_scaled']
            
            # Get number of clusters from UI or use a default
            n_components = self.n_clusters_spinbox.value()
            
            # Show progress
            progress = QProgressDialog("Running Probabilistic Clustering...", 
                                     "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setValue(10)
            
            # 1. First-level clustering with GMM
            progress.setLabelText("Fitting Gaussian Mixture Model...")
            gmm = GaussianMixture(n_components=n_components, 
                                covariance_type='full', 
                                random_state=42)
            
            # Get cluster probabilities and hard assignments
            gmm.fit(features)  # First fit the model
            cluster_probs = gmm.predict_proba(features)  # Then get probabilities
            hard_labels = gmm.predict(features)  # Get hard assignments
            progress.setValue(50)
            
            # Store results
            self.cluster_data['cluster_probs'] = cluster_probs
            self.cluster_data['labels'] = hard_labels
            self.cluster_data['gmm'] = gmm
            
            # 2. Hierarchical sub-typing
            progress.setLabelText("Identifying sub-types...")
            self._identify_subtypes(features, hard_labels, n_components)
            progress.setValue(80)
            
            # Update visualizations
            self.update_visualizations()
            progress.setValue(100)
            
            # Enable visualization buttons
            self.prob_viz_btn.setEnabled(True)
            self.subtype_viz_btn.setEnabled(True)
            
            QMessageBox.information(self, "Success", 
                                  f"Probabilistic clustering completed with {n_components} clusters")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clustering failed: {str(e)}")
        finally:
            progress.close()
    
    def _identify_subtypes(self, features, labels, n_clusters):
        """Identify sub-types within each cluster using hierarchical clustering."""
        self.cluster_data['subtypes'] = {}
        
        for cluster_id in range(n_clusters):
            # Get samples in this cluster
            mask = (labels == cluster_id)
            cluster_features = features[mask]
            
            if len(cluster_features) < 5:  # Skip small clusters
                continue
                
            try:
                # Determine number of sub-clusters (you can make this configurable)
                n_subtypes = min(3, len(cluster_features) // 5)
                
                if n_subtypes > 1:
                    # Calculate distance matrix and linkage
                    dist_matrix = pdist(cluster_features, 'euclidean')
                    Z = linkage(dist_matrix, method='ward')
                    
                    # Get sub-cluster labels
                    sub_labels = fcluster(Z, t=n_subtypes, criterion='maxclust')
                    
                    # Store results
                    self.cluster_data['subtypes'][cluster_id] = {
                        'linkage': Z,
                        'n_subtypes': n_subtypes,
                        'sub_labels': sub_labels,
                        'sample_indices': np.where(mask)[0]  # Store original indices
                    }
                    
            except Exception as e:
                print(f"Error in sub-clustering cluster {cluster_id}: {str(e)}")
    
    def plot_probability_heatmap(self):
        """Plot a heatmap of cluster probabilities."""
        if 'cluster_probs' not in self.cluster_data:
            QMessageBox.warning(self, "No Data", "Run probabilistic clustering first.")
            return
            
        # Create a new window
        win = QDialog(self)
        win.setWindowTitle("Cluster Probability Heatmap")
        win.setMinimumSize(800, 600)
        layout = QVBoxLayout(win)
        
        # Create figure and canvas
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Get data
        probs = self.cluster_data['cluster_probs']
        labels = self.cluster_data['labels']
        
        # Sort by cluster for better visualization
        sort_idx = np.argsort(labels)
        sorted_probs = probs[sort_idx]
        
        # Create heatmap
        ax = fig.add_subplot(111)
        sns.heatmap(sorted_probs, ax=ax, cmap='viridis', 
                    yticklabels=50, cbar_kws={'label': 'Probability'})
        ax.set_title('Cluster Assignment Probabilities')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Samples (sorted by cluster)')
        
        # Add a button to save the figure
        btn_save = QPushButton("Save Figure")
        def save_fig():
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
        
        btn_save.clicked.connect(save_fig)
        layout.addWidget(btn_save)
        
        win.exec()
    
    def plot_dendrogram(self, cluster_id=None):
        """Plot dendrogram for a specific cluster's sub-types."""
        if 'subtypes' not in self.cluster_data or not self.cluster_data['subtypes']:
            QMessageBox.warning(self, "No Data", "No sub-type information available.")
            return
        
        # If no cluster_id provided, show a dialog to select one
        if cluster_id is None:
            cluster_id, ok = QInputDialog.getInt(
                self, "Select Cluster", 
                "Enter cluster ID:", 
                value=0,
                minValue=0, 
                maxValue=len(self.cluster_data.get('labels', [0]))-1,
                step=1
            )
            if not ok:
                return
        
        if cluster_id not in self.cluster_data['subtypes']:
            QMessageBox.warning(self, "No Data", f"No sub-types found for cluster {cluster_id}")
            return
            
        subtype_info = self.cluster_data['subtypes'][cluster_id]
        
        # Create a new window
        win = QDialog(self)
        win.setWindowTitle(f"Cluster {cluster_id} Sub-types")
        win.setMinimumSize(800, 600)
        layout = QVBoxLayout(win)
        
        # Create figure and canvas
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Plot dendrogram
        ax = fig.add_subplot(111)
        dendrogram(subtype_info['linkage'], 
                  labels=[f"Sample {i+1}" for i in range(len(subtype_info['sub_labels']))],
                  orientation='left')
        ax.set_title(f'Cluster {cluster_id} Sub-type Hierarchy')
        ax.set_xlabel('Distance')
        
        # Add a button to save the figure
        btn_save = QPushButton("Save Figure")
        def save_fig():
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
        
        btn_save.clicked.connect(save_fig)
        layout.addWidget(btn_save)
        
        win.exec()
        
    def run_clustering(self):
        """Run hierarchical clustering analysis."""
        if self.cluster_data['intensities'] is None:
            QMessageBox.warning(self, "No Data", "No data available for clustering. Please import data first.")
            return
        
        try:
            # Get parameters
            n_clusters = self.n_clusters_spinbox.value()
            linkage_method = self.linkage_method_combo.currentText()
            distance_metric = self.distance_metric_combo.currentText()
            preprocessing_method = self.phase_method_combo.currentText()
            
            # Update status
            self.clustering_status.setText("Applying preprocessing...")
            self.clustering_progress.setValue(10)
            QApplication.processEvents()
            
            # Apply preprocessing based on selected method
            processed_intensities = self.cluster_data['intensities'].copy()
            wavenumbers = self.cluster_data['wavenumbers']
            
            if preprocessing_method == 'Corundum Correction':
                # Apply corundum drift correction
                processed_intensities = self.apply_corundum_drift_correction(
                    processed_intensities, wavenumbers
                )
                self.clustering_status.setText("Applied corundum drift correction...")
                
            elif preprocessing_method == 'NMF Separation':
                # Apply NMF phase separation
                n_components = self.nmf_components_spinbox.value()
                processed_intensities, nmf_components, nmf_weights, corundum_idx = self.apply_nmf_phase_separation(
                    processed_intensities, wavenumbers, n_components
                )
                # Store NMF results for analysis
                self.cluster_data['nmf_components'] = nmf_components
                self.cluster_data['nmf_weights'] = nmf_weights  
                self.cluster_data['corundum_component_idx'] = corundum_idx
                self.clustering_status.setText("Applied NMF phase separation...")
            
            elif preprocessing_method == 'Carbon Soot Optimization':
                # Apply carbon-specific preprocessing
                processed_intensities = self.apply_carbon_soot_preprocessing(
                    processed_intensities, wavenumbers
                )
                self.clustering_status.setText("Applied carbon soot optimization...")
                # Mark as carbon optimized for later use
                self.cluster_data['carbon_optimized'] = True
            
            self.clustering_progress.setValue(30)
            QApplication.processEvents()
            
            # Extract features from processed intensities
            self.clustering_status.setText("Extracting features...")
            if preprocessing_method == 'Exclude Regions':
                # For exclusion method, extract_vibrational_features handles the exclusion
                features = self.extract_vibrational_features(processed_intensities, wavenumbers)
            elif preprocessing_method == 'Carbon Soot Optimization':
                # Use carbon-specific features
                features = self.extract_carbon_specific_features(processed_intensities, wavenumbers)
                self.clustering_status.setText("Extracted carbon-specific features...")
            else:
                # For other methods, use the processed intensities directly
                features = []
                for spectrum in processed_intensities:
                    # Basic normalization
                    spectrum_features = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-8)
                    features.append(spectrum_features)
                features = np.array(features)
            
            self.cluster_data['features'] = features
            self.clustering_progress.setValue(50)
            QApplication.processEvents()
            
            # Scale features
            self.clustering_status.setText("Scaling features...")
            scaler = StandardScaler()
            self.cluster_data['features_scaled'] = scaler.fit_transform(features)
            self.clustering_progress.setValue(70)
            QApplication.processEvents()
            
            # Perform clustering
            self.clustering_status.setText("Running clustering...")
            labels, linkage_matrix, distance_matrix = self.perform_hierarchical_clustering(
                self.cluster_data['features_scaled'], n_clusters, linkage_method, distance_metric
            )
            
            # Store results
            self.cluster_data['labels'] = labels
            self.cluster_data['linkage_matrix'] = linkage_matrix
            self.cluster_data['distance_matrix'] = distance_matrix
            self.cluster_data['preprocessing_method'] = preprocessing_method
            
            # Update progress
            self.clustering_progress.setValue(90)
            QApplication.processEvents()
            
            # Update visualizations
            self.clustering_status.setText("Updating visualizations...")
            self.update_visualizations()
            
            # Update analysis results
            self.update_analysis_results()
            
            # Update progress
            self.clustering_progress.setValue(100)
            
            # Update status
            method_text = f" (using {preprocessing_method})" if preprocessing_method != 'None' else ""
            self.clustering_status.setText(f"Clustering complete: {n_clusters} clusters found{method_text}")
            
            # Switch to visualization tab
            self.tab_widget.setCurrentIndex(2)
            
        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", f"Clustering failed:\n{str(e)}")
            self.clustering_progress.setValue(0)
            self.clustering_status.setText("Clustering failed")

    def perform_hierarchical_clustering(self, features, n_clusters, linkage_method, distance_metric):
        """Perform hierarchical clustering on the features."""
        try:
            # Compute distance matrix if needed
            if distance_metric == 'euclidean':
                distance_matrix = pdist(features, metric='euclidean')
            elif distance_metric == 'cosine':
                distance_matrix = pdist(features, metric='cosine')
            elif distance_metric == 'correlation':
                distance_matrix = pdist(features, metric='correlation')
            else:
                distance_matrix = pdist(features, metric='euclidean')
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method=linkage_method)
            
            # Get cluster labels
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            return labels, linkage_matrix, distance_matrix
            
        except Exception as e:
            raise Exception(f"Clustering failed: {str(e)}")

    def extract_vibrational_features(self, intensities, wavenumbers):
        """Extract vibrational features from spectra."""
        try:
            features = []
            
            # Get exclusion regions from UI settings
            exclusion_regions = self.get_exclusion_regions()
            
            for spectrum in intensities:
                # Create a copy of the spectrum
                spectrum_features = spectrum.copy()
                
                # Apply exclusion regions if specified
                if exclusion_regions:
                    exclusion_mask = np.ones(len(wavenumbers), dtype=bool)
                    for min_wn, max_wn in exclusion_regions:
                        # Find indices within exclusion range
                        exclude_indices = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
                        exclusion_mask &= ~exclude_indices
                    
                    # Use only non-excluded regions for features
                    spectrum_features = spectrum_features[exclusion_mask]
                
                # Normalize the filtered spectrum
                if len(spectrum_features) > 0:
                    spectrum_features = (spectrum_features - np.min(spectrum_features)) / (np.max(spectrum_features) - np.min(spectrum_features) + 1e-8)
                else:
                    # Fallback if all regions are excluded
                    spectrum_features = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-8)
                
                features.append(spectrum_features)
            
            return np.array(features)
            
        except Exception as e:
            raise Exception(f"Feature extraction failed: {str(e)}")

    def extract_carbon_specific_features(self, intensities, wavenumbers):
        """
        Extract carbon-specific features optimized for soot clustering.
        Focuses on D/G band characteristics, disorder levels, and structural parameters.
        """
        try:
            features = []
            
            for spectrum in intensities:
                spectrum_features = []
                
                # 1. D-band characteristics (1300-1400 cm⁻¹)
                d_band_mask = (wavenumbers >= 1300) & (wavenumbers <= 1400)
                if np.any(d_band_mask):
                    d_region = spectrum[d_band_mask]
                    d_wavenumbers = wavenumbers[d_band_mask]
                    
                    # Peak position (disorder indicator)
                    d_peak_idx = np.argmax(d_region)
                    d_peak_position = d_wavenumbers[d_peak_idx] if d_peak_idx < len(d_wavenumbers) else 1350
                    spectrum_features.append(d_peak_position)
                    
                    # Peak intensity
                    d_intensity = np.max(d_region)
                    spectrum_features.append(d_intensity)
                    
                    # Peak width (FWHM estimate)
                    half_max = d_intensity / 2
                    indices = np.where(d_region >= half_max)[0]
                    if len(indices) > 1:
                        d_width = d_wavenumbers[indices[-1]] - d_wavenumbers[indices[0]]
                    else:
                        d_width = 50  # Default width
                    spectrum_features.append(d_width)
                    
                    # Integrated intensity
                    d_integrated = np.trapz(d_region, d_wavenumbers)
                    spectrum_features.append(d_integrated)
                else:
                    spectrum_features.extend([1350, 0, 50, 0])  # Default D-band values
                
                # 2. G-band characteristics (1550-1620 cm⁻¹)
                g_band_mask = (wavenumbers >= 1550) & (wavenumbers <= 1620)
                if np.any(g_band_mask):
                    g_region = spectrum[g_band_mask]
                    g_wavenumbers = wavenumbers[g_band_mask]
                    
                    # Peak position (graphitic quality)
                    g_peak_idx = np.argmax(g_region)
                    g_peak_position = g_wavenumbers[g_peak_idx] if g_peak_idx < len(g_wavenumbers) else 1580
                    spectrum_features.append(g_peak_position)
                    
                    # Peak intensity
                    g_intensity = np.max(g_region)
                    spectrum_features.append(g_intensity)
                    
                    # Peak width
                    half_max = g_intensity / 2
                    indices = np.where(g_region >= half_max)[0]
                    if len(indices) > 1:
                        g_width = g_wavenumbers[indices[-1]] - g_wavenumbers[indices[0]]
                    else:
                        g_width = 30  # Default width
                    spectrum_features.append(g_width)
                    
                    # Integrated intensity
                    g_integrated = np.trapz(g_region, g_wavenumbers)
                    spectrum_features.append(g_integrated)
                else:
                    spectrum_features.extend([1580, 0, 30, 0])  # Default G-band values
                
                # 3. ID/IG ratio (key discriminator for carbon materials)
                d_intensity = spectrum_features[1]  # D-band intensity
                g_intensity = spectrum_features[5]  # G-band intensity
                
                if g_intensity > 0:
                    id_ig_ratio = d_intensity / g_intensity
                else:
                    id_ig_ratio = 0
                spectrum_features.append(id_ig_ratio)
                
                # 4. D' band characteristics (1610-1650 cm⁻¹)
                d_prime_mask = (wavenumbers >= 1610) & (wavenumbers <= 1650)
                if np.any(d_prime_mask):
                    d_prime_region = spectrum[d_prime_mask]
                    d_prime_intensity = np.max(d_prime_region)
                    spectrum_features.append(d_prime_intensity)
                    
                    # D'/G ratio
                    if g_intensity > 0:
                        d_prime_g_ratio = d_prime_intensity / g_intensity
                    else:
                        d_prime_g_ratio = 0
                    spectrum_features.append(d_prime_g_ratio)
                else:
                    spectrum_features.extend([0, 0])  # Default D' values
                
                # 5. Low-frequency structural modes (200-800 cm⁻¹)
                low_freq_mask = (wavenumbers >= 200) & (wavenumbers <= 800)
                if np.any(low_freq_mask):
                    low_freq_region = spectrum[low_freq_mask]
                    low_freq_wavenumbers = wavenumbers[low_freq_mask]
                    low_freq_integrated = np.trapz(low_freq_region, low_freq_wavenumbers)
                    spectrum_features.append(low_freq_integrated)
                else:
                    spectrum_features.append(0)
                
                # 6. RBM modes (100-300 cm⁻¹) for nanotube detection
                rbm_mask = (wavenumbers >= 100) & (wavenumbers <= 300)
                if np.any(rbm_mask):
                    rbm_region = spectrum[rbm_mask]
                    rbm_intensity = np.max(rbm_region)
                    spectrum_features.append(rbm_intensity)
                else:
                    spectrum_features.append(0)
                
                # 7. 2D band characteristics (2600-2800 cm⁻¹)
                band_2d_mask = (wavenumbers >= 2600) & (wavenumbers <= 2800)
                if np.any(band_2d_mask):
                    band_2d_region = spectrum[band_2d_mask]
                    band_2d_intensity = np.max(band_2d_region)
                    spectrum_features.append(band_2d_intensity)
                    
                    # 2D/G ratio
                    if g_intensity > 0:
                        band_2d_g_ratio = band_2d_intensity / g_intensity
                    else:
                        band_2d_g_ratio = 0
                    spectrum_features.append(band_2d_g_ratio)
                else:
                    spectrum_features.extend([0, 0])
                
                # 8. G-band asymmetry (crystallinity indicator)
                if np.any(g_band_mask):
                    g_region = spectrum[g_band_mask]
                    g_wavenumbers = wavenumbers[g_band_mask]
                    g_peak_idx = np.argmax(g_region)
                    
                    if 0 < g_peak_idx < len(g_region) - 1:
                        # Calculate asymmetry as ratio of left/right areas
                        left_area = np.trapz(g_region[:g_peak_idx], g_wavenumbers[:g_peak_idx])
                        right_area = np.trapz(g_region[g_peak_idx:], g_wavenumbers[g_peak_idx:])
                        
                        if right_area > 0:
                            g_asymmetry = left_area / right_area
                        else:
                            g_asymmetry = 1.0
                    else:
                        g_asymmetry = 1.0
                    spectrum_features.append(g_asymmetry)
                else:
                    spectrum_features.append(1.0)
                
                # 9. Background slope (amorphous content indicator)
                if len(wavenumbers) > 10:
                    # Calculate background slope from 1000-1200 cm⁻¹ region
                    bg_mask = (wavenumbers >= 1000) & (wavenumbers <= 1200)
                    if np.any(bg_mask):
                        bg_region = spectrum[bg_mask]
                        bg_wavenumbers = wavenumbers[bg_mask]
                        
                        if len(bg_region) > 5:
                            # Linear fit to estimate background slope
                            coeffs = np.polyfit(bg_wavenumbers, bg_region, 1)
                            background_slope = coeffs[0]
                        else:
                            background_slope = 0
                    else:
                        background_slope = 0
                    spectrum_features.append(background_slope)
                else:
                    spectrum_features.append(0)
                
                # Ensure consistent feature vector length
                while len(spectrum_features) < 17:
                    spectrum_features.append(0)
                
                features.append(spectrum_features[:17])  # Trim to exact length
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error in carbon feature extraction: {e}")
            # Fallback to basic features
            return self.extract_vibrational_features(intensities, wavenumbers)

    def apply_carbon_soot_preprocessing(self, intensities, wavenumbers):
        """
        Enhanced carbon soot-specific preprocessing pipeline.
        Optimized for diesel/car exhaust/charcoal discrimination with improved methods.
        """
        try:
            processed_spectra = []
            
            print("Applying enhanced carbon soot preprocessing...")
            
            for i, spectrum in enumerate(intensities):
                if i % 10 == 0 and i > 0:
                    print(f"Processed {i}/{len(intensities)} spectra...")
                
                # 1. Remove cosmic rays (important for carbon analysis)
                processed_spectrum = self._remove_cosmic_rays(spectrum, wavenumbers)
                
                # 2. Enhanced carbon-specific baseline removal
                processed_spectrum = self._remove_enhanced_carbon_baseline(processed_spectrum, wavenumbers)
                
                # 3. Remove fluorescence background with carbon-specific approach
                processed_spectrum = self._remove_carbon_fluorescence_background(processed_spectrum, wavenumbers)
                
                # 4. Apply carbon-specific smoothing
                processed_spectrum = self._apply_carbon_smoothing(processed_spectrum, wavenumbers)
                
                # 5. Enhanced normalization for carbon materials
                processed_spectrum = self._normalize_carbon_spectrum_enhanced(processed_spectrum, wavenumbers)
                
                processed_spectra.append(processed_spectrum)
            
            print(f"Enhanced carbon preprocessing completed for {len(processed_spectra)} spectra")
            return np.array(processed_spectra)
            
        except Exception as e:
            print(f"Error in enhanced carbon preprocessing: {e}")
            return intensities  # Return original if preprocessing fails

    def _remove_carbon_baseline(self, spectrum, wavenumbers):
        """Remove baseline using carbon-specific approach."""
        try:
            from scipy.sparse import csc_matrix, eye, diags
            from scipy.sparse.linalg import spsolve
            
            # Asymmetric Least Squares (ALS) baseline correction
            # Parameters optimized for carbon materials
            lam = 1e6  # Smoothing parameter (higher for carbon)
            p = 0.001  # Asymmetry parameter (lower for carbon)
            niter = 10
            
            L = len(spectrum)
            if L < 10:
                return spectrum
            
            D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            D = lam * D.dot(D.transpose())
            
            w = np.ones(L)
            W = diags(w, 0, shape=(L, L))
            
            for i in range(niter):
                W.setdiag(w)
                Z = W + D
                baseline = spsolve(csc_matrix(Z), w * spectrum)
                w = p * (spectrum > baseline) + (1-p) * (spectrum < baseline)
            
            return spectrum - baseline
            
        except Exception:
            # Fallback: simple linear baseline
            return spectrum - np.linspace(spectrum[0], spectrum[-1], len(spectrum))

    def _remove_fluorescence_background(self, spectrum, wavenumbers):
        """Remove fluorescence background common in carbon materials."""
        try:
            from scipy import signal
            
            # Use a polynomial baseline fitting approach
            # Optimized for carbon soot fluorescence patterns
            
            # Find local minima as baseline points
            min_indices = signal.argrelextrema(spectrum, np.less, order=20)[0]
            
            if len(min_indices) < 3:
                # Not enough minima, use endpoints and middle
                min_indices = [0, len(spectrum)//2, len(spectrum)-1]
            
            # Add endpoints
            min_indices = np.concatenate([[0], min_indices, [len(spectrum)-1]])
            min_indices = np.unique(min_indices)
            
            # Fit polynomial to baseline points
            baseline_wn = wavenumbers[min_indices]
            baseline_intensities = spectrum[min_indices]
            
            # Use degree 3 polynomial for smooth baseline
            coeffs = np.polyfit(baseline_wn, baseline_intensities, min(3, len(min_indices)-1))
            baseline = np.polyval(coeffs, wavenumbers)
            
            # Subtract baseline
            corrected = spectrum - baseline
            
            # Ensure no negative values
            corrected = np.maximum(corrected, 0)
            
            return corrected
            
        except Exception:
            # Fallback: return original spectrum
            return spectrum

    def _normalize_carbon_spectrum(self, spectrum, wavenumbers):
        """Normalize spectrum using carbon-specific approach."""
        try:
            # Method 1: Normalize to G-band if present
            g_band_mask = (wavenumbers >= 1550) & (wavenumbers <= 1620)
            
            if np.any(g_band_mask) and np.max(spectrum[g_band_mask]) > 0:
                # Normalize to G-band maximum
                g_max = np.max(spectrum[g_band_mask])
                return spectrum / g_max
            
            # Method 2: Normalize to total area in carbon region
            carbon_mask = (wavenumbers >= 1200) & (wavenumbers <= 1700)
            if np.any(carbon_mask):
                carbon_area = np.trapz(spectrum[carbon_mask], wavenumbers[carbon_mask])
                if carbon_area > 0:
                    return spectrum / carbon_area * 1000  # Scale for better numerical stability
            
            # Method 3: Standard min-max normalization
            spectrum_min = np.min(spectrum)
            spectrum_max = np.max(spectrum)
            
            if spectrum_max > spectrum_min:
                return (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
            else:
                return spectrum
                
        except Exception:
            # Fallback: standard normalization
            return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-8)

    def _remove_cosmic_rays(self, spectrum, wavenumbers):
        """Remove cosmic ray spikes using median filtering and outlier detection."""
        try:
            from scipy.signal import medfilt
            from scipy import stats
            
            # Apply median filter to detect outliers
            filtered = medfilt(spectrum, kernel_size=5)
            residual = spectrum - filtered
            
            # Identify cosmic rays using z-score threshold
            z_scores = np.abs(stats.zscore(residual))
            cosmic_ray_mask = z_scores > 3.0  # Threshold for cosmic ray detection
            
            # Replace cosmic rays with median-filtered values
            cleaned_spectrum = spectrum.copy()
            cleaned_spectrum[cosmic_ray_mask] = filtered[cosmic_ray_mask]
            
            return cleaned_spectrum
            
        except Exception:
            return spectrum  # Return original if cosmic ray removal fails

    def _remove_enhanced_carbon_baseline(self, spectrum, wavenumbers):
        """Enhanced baseline removal optimized for carbon materials."""
        try:
            from scipy.sparse import csc_matrix, eye, diags
            from scipy.sparse.linalg import spsolve
            
            # Enhanced Asymmetric Least Squares (ALS) baseline correction
            # Parameters optimized specifically for carbon Raman spectra
            L = len(spectrum)
            if L < 10:
                return spectrum
            
            # Adaptive parameters based on spectral region
            carbon_mask = (wavenumbers >= 1200) & (wavenumbers <= 1700)
            if np.any(carbon_mask):
                # Higher smoothing in carbon region to preserve D/G bands
                lam_values = np.full(L, 1e5)  # Base smoothing
                lam_values[carbon_mask] = 1e7  # Higher smoothing in carbon region
                lam = np.mean(lam_values)
            else:
                lam = 1e6  # Default smoothing
            
            p = 0.0005  # Lower asymmetry for carbon (more symmetric baseline)
            niter = 15   # More iterations for better convergence
            
            # Create difference matrix
            D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            D = lam * D.dot(D.transpose())
            
            w = np.ones(L)
            W = diags(w, 0, shape=(L, L))
            
            for i in range(niter):
                W.setdiag(w)
                Z = W + D
                baseline = spsolve(csc_matrix(Z), w * spectrum)
                w = p * (spectrum > baseline) + (1-p) * (spectrum < baseline)
                
                # Adaptive weight adjustment for carbon regions
                if np.any(carbon_mask):
                    # Preserve carbon band regions from over-correction
                    w[carbon_mask] = np.maximum(w[carbon_mask], 0.1)
            
            # Ensure baseline doesn't exceed spectrum in carbon regions
            if np.any(carbon_mask):
                carbon_spectrum = spectrum[carbon_mask]
                carbon_baseline = baseline[carbon_mask]
                
                # Adjust baseline if it's too high in carbon regions
                if np.any(carbon_baseline > carbon_spectrum):
                    excess_mask = carbon_baseline > carbon_spectrum
                    carbon_indices = np.where(carbon_mask)[0]
                    for idx in carbon_indices[excess_mask]:
                        baseline[idx] = spectrum[idx] * 0.9  # Keep baseline slightly below
            
            return spectrum - baseline
            
        except Exception as e:
            print(f"Enhanced baseline removal failed: {e}")
            # Fallback to simple linear baseline
            return spectrum - np.linspace(spectrum[0], spectrum[-1], len(spectrum))

    def _remove_carbon_fluorescence_background(self, spectrum, wavenumbers):
        """Enhanced fluorescence background removal for carbon materials."""
        try:
            # Carbon materials often have complex fluorescence backgrounds
            # Use polynomial fitting with carbon-region protection
            
            # Identify regions likely to contain carbon bands
            d_region_mask = (wavenumbers >= 1300) & (wavenumbers <= 1400)
            g_region_mask = (wavenumbers >= 1550) & (wavenumbers <= 1620)
            carbon_mask = d_region_mask | g_region_mask
            
            # Create background estimation points (avoiding carbon regions)
            background_mask = ~carbon_mask
            
            # Add boundary points for better fitting
            if len(wavenumbers) > 50:
                boundary_indices = [0, len(wavenumbers)//4, 3*len(wavenumbers)//4, len(wavenumbers)-1]
                for idx in boundary_indices:
                    background_mask[idx] = True
            
            if np.sum(background_mask) > 10:  # Need enough points for fitting
                # Fit polynomial to background regions
                bg_wavenumbers = wavenumbers[background_mask]
                bg_intensities = spectrum[background_mask]
                
                # Use adaptive polynomial degree
                n_bg_points = len(bg_wavenumbers)
                if n_bg_points > 50:
                    poly_degree = 3
                elif n_bg_points > 20:
                    poly_degree = 2
                else:
                    poly_degree = 1
                
                # Robust polynomial fitting (reduce influence of outliers)
                weights = np.ones_like(bg_intensities)
                for iteration in range(3):  # Iterative reweighting
                    coeffs = np.polyfit(bg_wavenumbers, bg_intensities, poly_degree, w=weights)
                    bg_fit = np.polyval(coeffs, bg_wavenumbers)
                    residuals = np.abs(bg_intensities - bg_fit)
                    
                    # Downweight outliers
                    median_residual = np.median(residuals)
                    mad = np.median(np.abs(residuals - median_residual))
                    if mad > 0:
                        outlier_threshold = median_residual + 3 * mad
                        weights = np.where(residuals > outlier_threshold, 0.1, 1.0)
                
                # Apply background correction to entire spectrum
                background_full = np.polyval(coeffs, wavenumbers)
                
                # Ensure background doesn't exceed spectrum in carbon regions
                corrected_spectrum = spectrum - background_full
                
                # Prevent over-correction in carbon regions
                if np.any(carbon_mask):
                    carbon_corrected = corrected_spectrum[carbon_mask]
                    if np.any(carbon_corrected < 0):
                        # Adjust background to prevent negative values in carbon regions
                        min_carbon = np.min(carbon_corrected)
                        corrected_spectrum = corrected_spectrum - min_carbon
                
                return corrected_spectrum
            else:
                # Not enough background points, return original
                return spectrum
                
        except Exception as e:
            print(f"Fluorescence removal failed: {e}")
            return spectrum

    def _apply_carbon_smoothing(self, spectrum, wavenumbers):
        """Apply intelligent smoothing preserving carbon band features."""
        try:
            from scipy.signal import savgol_filter
            
            # Adaptive smoothing: less in carbon regions, more elsewhere
            smoothed_spectrum = spectrum.copy()
            
            # Define carbon regions where we want minimal smoothing
            d_region_mask = (wavenumbers >= 1300) & (wavenumbers <= 1400)
            g_region_mask = (wavenumbers >= 1550) & (wavenumbers <= 1620)
            carbon_mask = d_region_mask | g_region_mask
            
            # Smooth non-carbon regions more aggressively
            non_carbon_mask = ~carbon_mask
            if np.any(non_carbon_mask):
                # Apply moderate smoothing to non-carbon regions
                try:
                    smoothed_spectrum[non_carbon_mask] = savgol_filter(
                        spectrum[non_carbon_mask], 
                        window_length=min(7, len(spectrum[non_carbon_mask])//2 * 2 + 1), 
                        polyorder=2
                    )
                except:
                    pass  # Skip if smoothing fails
            
            # Light smoothing in carbon regions to preserve band shapes
            if np.any(carbon_mask):
                try:
                    carbon_spectrum = spectrum[carbon_mask]
                    if len(carbon_spectrum) > 5:
                        window_length = min(5, len(carbon_spectrum)//2 * 2 + 1)
                        if window_length >= 3:
                            smoothed_spectrum[carbon_mask] = savgol_filter(
                                carbon_spectrum, 
                                window_length=window_length, 
                                polyorder=1
                            )
                except:
                    pass  # Keep original if smoothing fails
            
            return smoothed_spectrum
            
        except Exception:
            return spectrum  # Return original if smoothing fails

    def _normalize_carbon_spectrum_enhanced(self, spectrum, wavenumbers):
        """Enhanced normalization optimized for carbon type discrimination."""
        try:
            # Method 1: Try normalization to G-band maximum (most stable reference)
            g_band_mask = (wavenumbers >= 1550) & (wavenumbers <= 1620)
            
            if np.any(g_band_mask):
                g_region = spectrum[g_band_mask]
                g_max = np.max(g_region)
                
                if g_max > 0 and not np.isnan(g_max) and not np.isinf(g_max):
                    # Normalize to G-band maximum
                    normalized = spectrum / g_max
                    
                    # Verify normalization quality
                    if np.all(np.isfinite(normalized)) and np.max(normalized) > 0:
                        return normalized
            
            # Method 2: Normalize to total area in carbon-active region (1200-1700 cm⁻¹)
            carbon_mask = (wavenumbers >= 1200) & (wavenumbers <= 1700)
            if np.any(carbon_mask):
                carbon_region = spectrum[carbon_mask]
                carbon_wavenumbers = wavenumbers[carbon_mask]
                
                # Calculate area under curve
                if len(carbon_region) > 1:
                    carbon_area = np.trapz(np.maximum(carbon_region, 0), carbon_wavenumbers)
                    
                    if carbon_area > 0 and not np.isnan(carbon_area) and not np.isinf(carbon_area):
                        normalized = spectrum / carbon_area * 1000  # Scale for numerical stability
                        
                        if np.all(np.isfinite(normalized)):
                            return normalized
            
            # Method 3: Vector normalization (L2 norm)
            spectrum_positive = spectrum - np.min(spectrum)  # Ensure all positive
            l2_norm = np.linalg.norm(spectrum_positive)
            
            if l2_norm > 0 and not np.isnan(l2_norm) and not np.isinf(l2_norm):
                normalized = spectrum_positive / l2_norm
                
                if np.all(np.isfinite(normalized)):
                    return normalized
            
            # Method 4: Standard min-max normalization (fallback)
            spectrum_min = np.min(spectrum)
            spectrum_max = np.max(spectrum)
            
            if spectrum_max > spectrum_min and np.isfinite(spectrum_max) and np.isfinite(spectrum_min):
                normalized = (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
                
                if np.all(np.isfinite(normalized)):
                    return normalized
            
            # Method 5: Return original if all normalizations fail
            print("Warning: All normalization methods failed, returning original spectrum")
            return spectrum
                
        except Exception as e:
            print(f"Enhanced normalization failed: {e}")
            # Final fallback
            try:
                return (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum) + 1e-8)
            except:
                return spectrum
            
    def get_exclusion_regions(self):
        """Get regions to exclude from clustering analysis (e.g., corundum peaks)."""
        exclusion_regions = []
        
        # Check if exclusion is enabled
        if hasattr(self, 'enable_exclusion_cb') and self.enable_exclusion_cb.isChecked():
            if hasattr(self, 'exclusion_widgets') and self.exclusion_widgets:
                for widget in self.exclusion_widgets:
                    try:
                        min_wn = float(widget['min_wn'].text())
                        max_wn = float(widget['max_wn'].text())
                        exclusion_regions.append((min_wn, max_wn))
                    except ValueError:
                        continue  # Skip invalid entries
        
        return exclusion_regions

    def apply_corundum_drift_correction(self, intensities, wavenumbers):
        """Apply drift correction using the corundum peak at ~520 cm-1."""
        try:
            corrected_intensities = []
            
            # Get corundum settings from UI
            corundum_center = self.corundum_center_spinbox.value() if hasattr(self, 'corundum_center_spinbox') else 520.0
            corundum_window = self.corundum_window_spinbox.value() if hasattr(self, 'corundum_window_spinbox') else 10.0
            
            # Find corundum peak region indices
            corundum_mask = (wavenumbers >= corundum_center - corundum_window) & \
                           (wavenumbers <= corundum_center + corundum_window)
            
            if not np.any(corundum_mask):
                print("Warning: Corundum peak region not found in spectrum")
                return intensities
            
            # Calculate reference corundum intensity (from first spectrum)
            if len(intensities) > 0:
                reference_corundum = np.max(intensities[0][corundum_mask])
            else:
                return intensities
            
            for spectrum in intensities:
                # Find corundum peak intensity in current spectrum
                current_corundum = np.max(spectrum[corundum_mask])
                
                if current_corundum > 0:
                    # Apply drift correction factor
                    correction_factor = reference_corundum / current_corundum
                    corrected_spectrum = spectrum * correction_factor
                else:
                    corrected_spectrum = spectrum.copy()
                
                corrected_intensities.append(corrected_spectrum)
            
            return np.array(corrected_intensities)
            
        except Exception as e:
            print(f"Warning: Corundum drift correction failed: {e}")
            return intensities

    def apply_nmf_phase_separation(self, intensities, wavenumbers, n_components=2):
        """Use NMF to separate mixed phases (e.g., sample + corundum)."""
        try:
            from sklearn.decomposition import NMF
            
            # Ensure all intensities are positive for NMF
            shifted_intensities = intensities - np.min(intensities) + 1e-6
            
            # Apply NMF
            nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
            W = nmf.fit_transform(shifted_intensities)  # Mixing coefficients
            H = nmf.components_  # Pure component spectra
            
            # Get corundum settings from UI
            corundum_center = self.corundum_center_spinbox.value() if hasattr(self, 'corundum_center_spinbox') else 520.0
            corundum_window = self.corundum_window_spinbox.value() if hasattr(self, 'corundum_window_spinbox') else 20.0
            
            # Identify corundum component (should have peak at ~520 cm-1)
            corundum_mask = (wavenumbers >= corundum_center - corundum_window) & \
                           (wavenumbers <= corundum_center + corundum_window)
            
            # Find which component has strongest signal in corundum region
            corundum_scores = []
            for i in range(n_components):
                if np.any(corundum_mask):
                    score = np.max(H[i][corundum_mask])
                else:
                    score = 0
                corundum_scores.append(score)
            
            corundum_component_idx = np.argmax(corundum_scores)
            
            # Extract sample component(s) - all except corundum
            sample_components = []
            for i in range(n_components):
                if i != corundum_component_idx:
                    sample_components.append(i)
            
            # Reconstruct sample-only spectra
            sample_intensities = []
            for spectrum_idx in range(len(intensities)):
                sample_spectrum = np.zeros_like(wavenumbers)
                for comp_idx in sample_components:
                    sample_spectrum += W[spectrum_idx, comp_idx] * H[comp_idx]
                sample_intensities.append(sample_spectrum)
            
            return np.array(sample_intensities), H, W, corundum_component_idx
            
        except ImportError:
            print("Warning: scikit-learn not available for NMF. Using original spectra.")
            return intensities, None, None, None
        except Exception as e:
            print(f"Warning: NMF phase separation failed: {e}")
            return intensities, None, None, None

    def update_preprocessing_controls(self):
        """Update visibility of preprocessing controls based on selected method."""
        if not hasattr(self, 'phase_method_combo'):
            return
            
        method = self.phase_method_combo.currentText()
        
        # Show/hide corundum settings
        corundum_visible = method in ['Corundum Correction', 'NMF Separation']
        if hasattr(self, 'corundum_center_spinbox'):
            self.corundum_center_spinbox.parent().setVisible(corundum_visible)
        if hasattr(self, 'corundum_window_spinbox'):
            self.corundum_window_spinbox.parent().setVisible(corundum_visible)
        
        # Show/hide and enable/disable NMF settings
        nmf_visible = method == 'NMF Separation'
        if hasattr(self, 'nmf_components_spinbox'):
            self.nmf_components_spinbox.setEnabled(nmf_visible)
            self.nmf_components_spinbox.parent().setVisible(nmf_visible)
        
        # Show/hide and enable/disable exclusion settings
        exclusion_visible = method == 'Exclude Regions'
        if hasattr(self, 'exclusion_regions_edit'):
            self.exclusion_regions_edit.setEnabled(exclusion_visible)
        if hasattr(self, 'enable_exclusion_cb'):
            self.enable_exclusion_cb.parent().setVisible(exclusion_visible)
        if hasattr(self, 'exclusion_regions_frame'):
            self.exclusion_regions_frame.setVisible(exclusion_visible)
        
        # Auto-enable exclusion checkbox if method is selected
        if exclusion_visible and hasattr(self, 'enable_exclusion_cb'):
            self.enable_exclusion_cb.setChecked(True)

    def update_carbon_controls(self):
        """Update visibility of carbon analysis controls."""
        is_enabled = self.enable_carbon_analysis_cb.isChecked()
        for control in self.carbon_controls:
            control.setVisible(is_enabled)
        
        if is_enabled:
            # Store the current method before switching to Carbon Soot Optimization
            if not hasattr(self, 'previous_phase_method'):
                self.previous_phase_method = self.phase_method_combo.currentText()
            # Auto-select carbon optimization if enabled
            self.phase_method_combo.setCurrentText('Carbon Soot Optimization')
        else:
            # Restore the previous phase method when disabling carbon analysis
            if hasattr(self, 'previous_phase_method'):
                self.phase_method_combo.setCurrentText(self.previous_phase_method)
                delattr(self, 'previous_phase_method')
        
    def add_exclusion_region(self):
        """Add a new exclusion region widget."""
        region_data = {
            'name': 'New Region',
            'min_wn': '500',
            'max_wn': '600',
            'description': 'Custom exclusion region'
        }
        self.add_exclusion_region_widget(region_data)
        
    def add_exclusion_region_widget(self, region_data):
        """Add an exclusion region widget with the given data."""
        # Create widget container
        region_widget = QFrame()
        region_widget.setFrameStyle(QFrame.StyledPanel)
        region_layout = QGridLayout(region_widget)
        
        # Create widgets
        widgets = {}
        
        # Name field
        widgets['name'] = QLineEdit(region_data.get('name', ''))
        region_layout.addWidget(QLabel("Name:"), 0, 0)
        region_layout.addWidget(widgets['name'], 0, 1)
        
        # Min wavenumber
        widgets['min_wn'] = QLineEdit(region_data.get('min_wn', ''))
        widgets['min_wn'].setPlaceholderText("cm⁻¹")
        region_layout.addWidget(QLabel("Min:"), 0, 2)
        region_layout.addWidget(widgets['min_wn'], 0, 3)
        
        # Max wavenumber
        widgets['max_wn'] = QLineEdit(region_data.get('max_wn', ''))
        widgets['max_wn'].setPlaceholderText("cm⁻¹")
        region_layout.addWidget(QLabel("Max:"), 0, 4)
        region_layout.addWidget(widgets['max_wn'], 0, 5)
        
        # Description field
        widgets['description'] = QLineEdit(region_data.get('description', ''))
        widgets['description'].setPlaceholderText("Description")
        region_layout.addWidget(QLabel("Description:"), 1, 0)
        region_layout.addWidget(widgets['description'], 1, 1, 1, 5)
        
        # Add checkbox for selection
        widgets['selected'] = QCheckBox()
        region_layout.addWidget(widgets['selected'], 0, 6)
        
        # Store widget references
        region_widget.widgets = widgets
        if not hasattr(self, 'exclusion_widgets'):
            self.exclusion_widgets = []
        self.exclusion_widgets.append(widgets)
        
        # Add to layout
        if hasattr(self, 'exclusion_regions_layout'):
            self.exclusion_regions_layout.addWidget(region_widget)
        
    def remove_exclusion_region(self):
        """Remove selected exclusion regions."""
        if not hasattr(self, 'exclusion_widgets'):
            return
            
        # Find selected widgets
        to_remove = []
        for i, widget_dict in enumerate(self.exclusion_widgets):
            if widget_dict['selected'].isChecked():
                to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(to_remove):
            widget_dict = self.exclusion_widgets[i]
            # Find the parent widget and remove it
            if hasattr(self, 'exclusion_regions_layout'):
                for j in range(self.exclusion_regions_layout.count()):
                    item = self.exclusion_regions_layout.itemAt(j)
                    if item and item.widget() and hasattr(item.widget(), 'widgets'):
                        if item.widget().widgets == widget_dict:
                            item.widget().setParent(None)
                            break
            
            # Remove from list
            del self.exclusion_widgets[i]

    # Visualization Methods
    def update_visualizations(self):
        """Update all visualization plots."""
        try:
            self.update_dendrogram()
            self.update_heatmap()
            self.update_scatter_plot()
            self.update_pca_components_plot()
            self.update_refinement_plot()  # Add refinement plot update
        except Exception as e:
            print(f"Error updating visualizations: {str(e)}")
    
    def refresh_spectra_list(self):
        """Refresh the spectra list from the database."""
        self.clear_filters()
        self.load_filter_options()
        self.apply_filters()
    
    def clear_filters(self):
        """Clear all filter inputs."""
        self.hey_classification_combo.setCurrentText("")
        self.chemical_family_combo.setCurrentText("")
        self.mineral_name_edit.clear()
        self.formula_edit.clear()
        self.apply_filters()
    
    def update_spectra_table(self):
        """Update the spectra table with filtered results."""
        try:
            self.spectra_table.setRowCount(len(self.filtered_spectra))
            self.results_count_label.setText(f"{len(self.filtered_spectra)} spectra found")
            
            for i, (name, data) in enumerate(self.filtered_spectra.items()):
                metadata = data.get('metadata', {})
                
                # Checkbox for selection
                checkbox = QCheckBox()
                checkbox.stateChanged.connect(self.update_selection_count)
                self.spectra_table.setCellWidget(i, 0, checkbox)
                
                # Mineral name
                mineral_name = metadata.get('NAME') or metadata.get('mineral_name') or name
                self.spectra_table.setItem(i, 1, QTableWidgetItem(mineral_name))
                
                # Hey classification
                hey_class = metadata.get('HEY CLASSIFICATION') or metadata.get('Hey Classification', '')
                self.spectra_table.setItem(i, 2, QTableWidgetItem(hey_class))
                
                # Formula
                formula = metadata.get('FORMULA') or metadata.get('Formula', '')
                self.spectra_table.setItem(i, 3, QTableWidgetItem(formula))
            
            # Resize columns
            self.spectra_table.resizeColumnsToContents()
            self.update_selection_count()
            
        except Exception as e:
            print(f"Error updating spectra table: {str(e)}")
    
    def select_all_spectra(self):
        """Select all visible spectra."""
        for i in range(self.spectra_table.rowCount()):
            checkbox = self.spectra_table.cellWidget(i, 0)
            if checkbox:
                checkbox.setChecked(True)
    
    def select_no_spectra(self):
        """Deselect all spectra."""
        for i in range(self.spectra_table.rowCount()):
            checkbox = self.spectra_table.cellWidget(i, 0)
            if checkbox:
                checkbox.setChecked(False)
    
    def update_selection_count(self):
        """Update the selection count label."""
        selected_count = 0
        for i in range(self.spectra_table.rowCount()):
            checkbox = self.spectra_table.cellWidget(i, 0)
            if checkbox and checkbox.isChecked():
                selected_count += 1
        
        self.selected_count_label.setText(f"{selected_count} spectra selected")
    
    def get_selected_spectra(self):
        """Get the dictionary of selected spectra."""
        selected = {}
        
        for i, (name, data) in enumerate(self.filtered_spectra.items()):
            if i < self.spectra_table.rowCount():
                checkbox = self.spectra_table.cellWidget(i, 0)
                if checkbox and checkbox.isChecked():
                    selected[name] = data
        
        return selected

    def update_dendrogram(self):
        """Update the dendrogram visualization."""
        if (self.cluster_data['linkage_matrix'] is None or 
            self.cluster_data['labels'] is None):
            return
        
        try:
            self.dendrogram_ax.clear()
            
            # Get parameters
            orientation = self.dendro_orientation.currentText()
            max_samples = self.dendro_max_samples.value()
            show_labels = self.dendro_show_labels.isChecked()
            
            # Create dendrogram
            dendrogram(
                self.cluster_data['linkage_matrix'],
                ax=self.dendrogram_ax,
                orientation=orientation,
                truncate_mode='lastp',
                p=max_samples,
                show_leaf_counts=True,
                leaf_rotation=90 if orientation in ['top', 'bottom'] else 0,
                labels=None if not show_labels else [f"S{i}" for i in range(len(self.cluster_data['labels']))]
            )
            
            self.dendrogram_ax.set_title('Hierarchical Clustering Dendrogram')
            self.dendro_fig.tight_layout()
            self.dendro_canvas.draw()
            
        except Exception as e:
            print(f"Error updating dendrogram: {str(e)}")

    def update_heatmap(self):
        """Update the heatmap visualization of Raman spectra sorted by cluster."""
        if (self.cluster_data['intensities'] is None or 
            self.cluster_data['labels'] is None or
            self.cluster_data['wavenumbers'] is None):
            return
        
        try:
            print("DEBUG: === STARTING AGGRESSIVE HEATMAP REBUILD ===")
            
            # STEP 1: Complete figure cleanup
            self.heatmap_fig.clear()
            print("DEBUG: Cleared entire figure")
            
            # STEP 2: Recreate axis
            self.heatmap_ax = self.heatmap_fig.add_subplot(111)
            print("DEBUG: Recreated axis")
            
            # STEP 3: Reset colorbar reference
            self._heatmap_colorbar = None
            print("DEBUG: Reset colorbar reference")
            
            # Get parameters
            colormap = self.heatmap_colormap.currentText()
            norm_method = self.heatmap_norm.currentText()
            contrast = self.heatmap_contrast.value() / 10.0  # Scale from slider
            
            print(f"DEBUG: Using colormap={colormap}, norm={norm_method}, contrast={contrast}")
            
            # Use original Raman spectra intensities instead of processed features
            intensities = self.cluster_data['intensities'].copy()
            wavenumbers = self.cluster_data['wavenumbers']
            labels = self.cluster_data['labels']
            
            # Sort spectra by cluster labels
            sort_indices = np.argsort(labels)
            sorted_intensities = intensities[sort_indices]
            sorted_labels = labels[sort_indices]
            
            # Apply normalization to the spectra
            if norm_method == 'row':
                # Normalize each spectrum to its maximum
                max_vals = np.max(sorted_intensities, axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1  # Avoid division by zero
                sorted_intensities = sorted_intensities / max_vals
            elif norm_method == 'column':
                # Normalize each wavenumber across all spectra
                max_vals = np.max(sorted_intensities, axis=0, keepdims=True)
                max_vals[max_vals == 0] = 1  # Avoid division by zero
                sorted_intensities = sorted_intensities / max_vals
            elif norm_method == 'log':
                # Log normalization with better handling of zeros and negatives
                sorted_intensities = np.log1p(np.abs(sorted_intensities))
            elif norm_method == 'sqrt':
                # Square root normalization with better handling of negatives
                sorted_intensities = np.sqrt(np.abs(sorted_intensities))
            elif norm_method == 'linear':
                # Simple min-max normalization across all data
                min_val = np.min(sorted_intensities)
                max_val = np.max(sorted_intensities)
                if max_val > min_val:
                    sorted_intensities = (sorted_intensities - min_val) / (max_val - min_val)
            
            # Apply contrast adjustment with better scaling
            if contrast != 1.0:
                # Center the data around 0.5 for better contrast adjustment
                if norm_method == 'linear':
                    sorted_intensities = 0.5 + (sorted_intensities - 0.5) * contrast
                else:
                    # For other normalizations, apply contrast directly
                    mean_val = np.mean(sorted_intensities)
                    sorted_intensities = mean_val + (sorted_intensities - mean_val) * contrast
            
            # Ensure we're displaying the correct wavenumber range
            n_spectra, n_wavenumbers = sorted_intensities.shape
            wn_min, wn_max = wavenumbers[0], wavenumbers[-1]
            
            # Create heatmap with proper extent to show actual wavenumber range
            extent = [wn_min, wn_max, 0, n_spectra]
            
            print(f"DEBUG: Creating imshow with colormap={colormap}")
            print(f"DEBUG: Data range: {np.min(sorted_intensities):.3f} to {np.max(sorted_intensities):.3f}")
            print(f"DEBUG: Extent: {extent}")
            
            im = self.heatmap_ax.imshow(
                sorted_intensities,
                cmap=colormap, 
                aspect='auto',
                interpolation='nearest',
                origin='lower',
                extent=extent
            )
            
            print(f"DEBUG: imshow created, colormap applied: {im.get_cmap().name}")
            print(f"DEBUG: Image data shape: {im.get_array().shape}")
            
            # STEP 4: Add colorbar (no need to remove since we cleared the figure)
            self._heatmap_colorbar = self.heatmap_fig.colorbar(im, ax=self.heatmap_ax, 
                                                             label='Normalized Intensity')
            print(f"DEBUG: Added colorbar with colormap: {colormap}")
            
            # Add cluster boundaries as horizontal lines
            cluster_boundaries = []
            current_cluster = sorted_labels[0]
            for i, cluster in enumerate(sorted_labels):
                if cluster != current_cluster:
                    cluster_boundaries.append(i)
                    current_cluster = cluster
            
            for boundary in cluster_boundaries:
                self.heatmap_ax.axhline(y=boundary, color='white', linewidth=2, alpha=0.8)
            
            # Set up proper axis labels and ticks for wavenumber axis
            # X-axis: Show wavenumber values across actual range
            if n_wavenumbers > 15:
                # Create approximately 8-10 evenly spaced ticks across the wavenumber range
                n_ticks = min(10, n_wavenumbers)
                tick_values = np.linspace(wn_min, wn_max, n_ticks)
                self.heatmap_ax.set_xticks(tick_values)
                self.heatmap_ax.set_xticklabels([f"{wn:.0f}" for wn in tick_values], 
                                               rotation=45, ha='right')
            else:
                # For smaller datasets, show all wavenumbers
                self.heatmap_ax.set_xticks(wavenumbers[::max(1, len(wavenumbers)//10)])
                self.heatmap_ax.set_xticklabels([f"{wn:.0f}" for wn in wavenumbers[::max(1, len(wavenumbers)//10)]], 
                                               rotation=45, ha='right')
            
            # Y-axis: Add cluster labels at cluster centers
            unique_labels = np.unique(sorted_labels)
            cluster_positions = []
            cluster_names = []
            
            for cluster_id in unique_labels:
                cluster_mask = sorted_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_center = (cluster_indices[0] + cluster_indices[-1]) / 2
                cluster_positions.append(cluster_center)
                cluster_names.append(f"Cluster {cluster_id}\n({len(cluster_indices)} spectra)")
            
            # Set y-axis ticks at cluster centers
            self.heatmap_ax.set_yticks(cluster_positions)
            self.heatmap_ax.set_yticklabels(cluster_names, fontsize=9)
            
            # Set proper axis limits
            self.heatmap_ax.set_xlim(wn_min, wn_max)
            self.heatmap_ax.set_ylim(0, n_spectra)
            
            # Labels and title
            range_info = f" ({wn_min:.0f}-{wn_max:.0f} cm⁻¹)"
            self.heatmap_ax.set_title(f'Raman Spectra Heatmap (Sorted by Cluster){range_info}', fontsize=12, pad=20)
            self.heatmap_ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
            self.heatmap_ax.set_ylabel('Spectra by Cluster', fontsize=10)
            
            # Add text showing current settings and data info
            settings_text = (f"Colormap: {colormap} | Norm: {norm_method} | Contrast: {contrast:.1f}\n"
                           f"Range: {wn_min:.0f}-{wn_max:.0f} cm⁻¹ | {n_spectra} spectra, {len(unique_labels)} clusters")
            self.heatmap_ax.text(0.02, 0.98, settings_text, transform=self.heatmap_ax.transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round', 
                                facecolor='white', alpha=0.8), fontsize=8)
            
            # Debug info - print wavenumber range to console
            print(f"DEBUG: Heatmap Update:")
            print(f"  Wavenumber range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹")
            print(f"  Data shape: {sorted_intensities.shape}")
            print(f"  Colormap: {colormap}, Normalization: {norm_method}, Contrast: {contrast:.1f}")
            print(f"  Canvas widget: {type(self.heatmap_canvas)}")
            print(f"  Figure: {type(self.heatmap_fig)}")
            
            # STEP 5: Apply layout and force complete redraw
            print("DEBUG: Applying tight_layout")
            self.heatmap_fig.tight_layout()
            
            # STEP 6: Aggressive canvas refresh sequence
            print("DEBUG: Starting aggressive canvas refresh...")
            from PySide6.QtWidgets import QApplication
            import matplotlib.pyplot as plt
            
            # Force matplotlib to flush any cached rendering
            try:
                plt.draw()
                print("DEBUG: Called plt.draw() - global flush")
            except:
                print("DEBUG: plt.draw() not available")
            
            # Force immediate draw
            self.heatmap_canvas.draw()
            print("DEBUG: Called canvas.draw() - immediate")
            
            # Process all pending Qt events
            QApplication.processEvents()
            print("DEBUG: Processed Qt events")
            
            # Force widget repaint
            self.heatmap_canvas.repaint()
            print("DEBUG: Called canvas.repaint()")
            
            # Force figure rendering
            try:
                self.heatmap_fig.canvas.flush_events()
                print("DEBUG: Called figure.canvas.flush_events()")
            except:
                print("DEBUG: figure.canvas.flush_events() not available")
            
            # Another round of event processing
            QApplication.processEvents()
            print("DEBUG: Final Qt event processing")
            
            # Final check - verify colormap is applied
            if hasattr(self, 'heatmap_ax') and len(self.heatmap_ax.images) > 0:
                current_cmap = self.heatmap_ax.images[0].get_cmap().name
                print(f"DEBUG: Final verification - active colormap: {current_cmap}")
                if current_cmap != colormap:
                    print(f"DEBUG: WARNING - Colormap mismatch! Expected: {colormap}, Got: {current_cmap}")
            else:
                print("DEBUG: No images found on axis")
            
            print("DEBUG: === AGGRESSIVE HEATMAP REBUILD COMPLETE ===")
            
        except Exception as e:
            print(f"ERROR: Exception in update_heatmap: {str(e)}")
            import traceback
            traceback.print_exc()

    def force_heatmap_refresh(self):
        """Force a complete heatmap refresh with enhanced debugging."""
        print("DEBUG: === FORCE HEATMAP REFRESH INITIATED ===")
        
        # Step 1: Clear everything
        print("DEBUG: Step 1 - Clearing heatmap axis")
        self.heatmap_ax.clear()
        
        # Step 2: Force canvas draw to clear
        print("DEBUG: Step 2 - Drawing cleared canvas")
        self.heatmap_canvas.draw()
        
        # Step 3: Remove colorbar completely
        if hasattr(self, '_heatmap_colorbar') and self._heatmap_colorbar is not None:
            print("DEBUG: Step 3 - Removing colorbar")
            try:
                self._heatmap_colorbar.remove()
                self._heatmap_colorbar = None
                print("DEBUG: Colorbar removed successfully")
            except Exception as e:
                print(f"DEBUG: Error removing colorbar: {e}")
                self._heatmap_colorbar = None
        
        # Step 4: Process Qt events
        print("DEBUG: Step 4 - Processing Qt events")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Step 5: Call update_heatmap
        print("DEBUG: Step 5 - Calling update_heatmap")
        self.update_heatmap()
        
        # Step 6: Final forced refresh
        print("DEBUG: Step 6 - Final canvas refresh")
        self.heatmap_canvas.draw()
        self.heatmap_canvas.draw_idle()
        QApplication.processEvents()
        
        print("DEBUG: === FORCE HEATMAP REFRESH COMPLETED ===")

    def nuclear_heatmap_reset(self):
        """Nuclear option: completely recreate the heatmap canvas and figure."""
        print("DEBUG: === NUCLEAR HEATMAP RESET INITIATED ===")
        
        try:
            # Find the heatmap tab
            heatmap_tab_index = -1
            for i in range(self.viz_tab_widget.count()):
                if self.viz_tab_widget.tabText(i) == "Heatmap":
                    heatmap_tab_index = i
                    break
            
            if heatmap_tab_index == -1:
                print("DEBUG: Could not find heatmap tab")
                return
            
            # Get the current tab widget
            current_widget = self.viz_tab_widget.widget(heatmap_tab_index)
            layout = current_widget.layout()
            
            # Remove old canvas and toolbar
            if hasattr(self, 'heatmap_canvas'):
                layout.removeWidget(self.heatmap_canvas)
                self.heatmap_canvas.setParent(None)
                self.heatmap_canvas.deleteLater()
                print("DEBUG: Removed old canvas")
                
            if hasattr(self, 'heatmap_toolbar'):
                layout.removeWidget(self.heatmap_toolbar)
                self.heatmap_toolbar.setParent(None)
                self.heatmap_toolbar.deleteLater()
                print("DEBUG: Removed old toolbar")
            
            # Create new figure and canvas
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            
            self.heatmap_fig = Figure(figsize=(10, 6))
            self.heatmap_ax = self.heatmap_fig.add_subplot(111)
            self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
            self.heatmap_toolbar = NavigationToolbar(self.heatmap_canvas, current_widget)
            
            # Add new widgets to layout
            layout.addWidget(self.heatmap_canvas)
            layout.addWidget(self.heatmap_toolbar)
            
            print("DEBUG: Created new canvas and toolbar")
            
            # Reset colorbar reference
            self._heatmap_colorbar = None
            
            # Force widget update
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Now update the heatmap
            self.update_heatmap()
            
            print("DEBUG: === NUCLEAR HEATMAP RESET COMPLETED ===")
            
        except Exception as e:
            print(f"DEBUG: Nuclear reset failed: {e}")
            import traceback
            traceback.print_exc()

    def delayed_heatmap_update(self):
        """Handle delayed heatmap updates to prevent rapid successive calls."""
        print("DEBUG: Delayed heatmap update triggered")
        
        # Cancel any existing timer
        if hasattr(self, '_heatmap_update_timer'):
            try:
                self._heatmap_update_timer.stop()
                print("DEBUG: Stopped existing timer")
            except:
                pass
        
        # Create new timer for delayed update
        from PySide6.QtCore import QTimer
        self._heatmap_update_timer = QTimer()
        self._heatmap_update_timer.setSingleShot(True)
        self._heatmap_update_timer.timeout.connect(self.update_heatmap)
        self._heatmap_update_timer.start(100)  # 100ms delay
        print("DEBUG: Started delayed update timer (100ms)")

    def update_pca_components_plot(self):
        """Update the PCA components visualization."""
        if self.cluster_data['features_scaled'] is None:
            return
        
        try:
            self.pca_fig.clear()
            
            # Get number of components to show
            n_components = self.pca_n_components.value()
            
            features = self.cluster_data['features_scaled']
            
            # Perform PCA
            pca = PCA(n_components=min(n_components, features.shape[1]))
            pca_result = pca.fit_transform(features)
            
            # Store PCA model
            self.cluster_data['pca_full'] = pca
            
            # Create subplots
            n_cols = min(2, n_components)
            n_rows = (n_components + n_cols - 1) // n_cols
            
            for i in range(n_components):
                ax = self.pca_fig.add_subplot(n_rows, n_cols, i + 1)
                
                # Plot component loadings
                component = pca.components_[i]
                ax.plot(component, 'b-', linewidth=1)
                ax.set_title(f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%} var)')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Loading')
                ax.grid(True, alpha=0.3)
            
            # Add explained variance plot
            if n_components > 1:
                ax_var = self.pca_fig.add_subplot(n_rows, n_cols, n_components + 1)
                ax_var.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                          pca.explained_variance_ratio_)
                ax_var.set_title('Explained Variance by Component')
                ax_var.set_xlabel('Principal Component')
                ax_var.set_ylabel('Explained Variance Ratio')
                ax_var.grid(True, alpha=0.3)
            
            self.pca_fig.tight_layout()
            self.pca_canvas.draw()
            
        except Exception as e:
            print(f"Error updating PCA components plot: {str(e)}")

    def update_analysis_results(self):
        """Update the analysis results text."""
        try:
            if (self.cluster_data['labels'] is None or 
                self.cluster_data['features_scaled'] is None):
                return
        
            labels = self.cluster_data['labels']
            features = self.cluster_data['features_scaled']
            
            # Calculate basic statistics
            n_clusters = len(np.unique(labels))
            n_samples = len(labels)
            
            # Calculate silhouette score
            try:
                silhouette_avg = silhouette_score(features, labels)
            except:
                silhouette_avg = 0.0
            
            # Calculate cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(unique_labels, counts))
            
            # Format results
            results_text = f"""Clustering Analysis Results
=====================================

Dataset Summary:
• Total spectra: {n_samples}
• Number of clusters: {n_clusters}
• Average silhouette score: {silhouette_avg:.3f}

Cluster Sizes:
"""
            
            for cluster_id, size in cluster_sizes.items():
                percentage = (size / n_samples) * 100
                results_text += f"• Cluster {cluster_id}: {size} spectra ({percentage:.1f}%)\n"
            
            # Add preprocessing information
            if 'preprocessing_method' in self.cluster_data:
                preprocessing = self.cluster_data['preprocessing_method']
                results_text += f"\nPreprocessing Method: {preprocessing}\n"
            
            # Add PCA information if available
            if 'pca_model' in self.cluster_data:
                pca = self.cluster_data['pca_model']
                results_text += f"\nPCA Information:\n"
                results_text += f"• First 2 components explain {pca.explained_variance_ratio_[:2].sum():.1%} of variance\n"
            
            # Add detailed cluster composition
            results_text += self.generate_cluster_composition_details(labels, unique_labels)
            
            self.analysis_results_text.setText(results_text)
            
        except Exception as e:
            print(f"Error updating analysis results: {str(e)}")
    
    def generate_cluster_composition_details(self, labels, unique_labels):
        """Generate detailed cluster composition with spectrum listings."""
        composition_text = "\n\nDetailed Cluster Composition:\n"
        composition_text += "=" * 40 + "\n\n"
        
        # Get spectrum metadata if available
        spectrum_metadata = self.cluster_data.get('spectrum_metadata', [])
        has_metadata = len(spectrum_metadata) > 0
        
        for cluster_id in sorted(unique_labels):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            composition_text += f"Cluster {cluster_id} ({len(cluster_indices)} spectra):\n"
            composition_text += "-" * (15 + len(str(cluster_id))) + "\n"
            
            # Group spectra by type if metadata is available
            if has_metadata:
                cluster_spectra = self.organize_cluster_spectra(cluster_indices, spectrum_metadata)
                composition_text += self.format_cluster_spectra(cluster_spectra)
            else:
                # Simple listing if no metadata
                for i, idx in enumerate(cluster_indices, 1):
                    composition_text += f"  {i:2d}. Spectrum {idx:03d}\n"
            
            composition_text += "\n"
        
        return composition_text
    
    def organize_cluster_spectra(self, cluster_indices, spectrum_metadata):
        """Organize cluster spectra by available metadata categories."""
        organized = {
            'database_spectra': [],
            'file_spectra': [],
            'main_app_spectra': [],
            'unknown_spectra': []
        }
        
        for idx in cluster_indices:
            if idx < len(spectrum_metadata):
                metadata = spectrum_metadata[idx]
                filename = metadata.get('filename', f'spectrum_{idx}')
                source = metadata.get('source', 'unknown')
                
                spectrum_info = {
                    'index': idx,
                    'filename': filename,
                    'metadata': metadata
                }
                
                if source == 'database':
                    organized['database_spectra'].append(spectrum_info)
                elif source == 'main_application':
                    organized['main_app_spectra'].append(spectrum_info)
                elif 'file_path' in metadata:
                    organized['file_spectra'].append(spectrum_info)
                else:
                    organized['unknown_spectra'].append(spectrum_info)
            else:
                # Fallback for missing metadata
                organized['unknown_spectra'].append({
                    'index': idx,
                    'filename': f'spectrum_{idx}',
                    'metadata': {}
                })
        
        return organized
    
    def format_cluster_spectra(self, cluster_spectra):
        """Format the organized cluster spectra for display with sorting."""
        formatted_text = ""
        
        # Database spectra
        if cluster_spectra['database_spectra']:
            formatted_text += "  Database Spectra:\n"
            # Sort by filename
            sorted_db_spectra = sorted(cluster_spectra['database_spectra'], 
                                     key=lambda x: x['filename'].lower())
            for i, spec_info in enumerate(sorted_db_spectra, 1):
                metadata = spec_info['metadata']
                filename = spec_info['filename']
                
                # Get additional database info
                mineral_name = metadata.get('mineral_name', metadata.get('NAME', ''))
                hey_class = metadata.get('hey_classification', metadata.get('HEY CLASSIFICATION', ''))
                formula = metadata.get('formula', metadata.get('FORMULA', ''))
                
                formatted_text += f"    {i:2d}. {filename}"
                
                # Add mineral information if available
                if mineral_name:
                    formatted_text += f" ({mineral_name}"
                    if hey_class:
                        formatted_text += f", {hey_class}"
                    if formula:
                        formatted_text += f", {formula}"
                    formatted_text += ")"
                
                formatted_text += "\n"
            formatted_text += "\n"
        
        # File-based spectra
        if cluster_spectra['file_spectra']:
            formatted_text += "  File-based Spectra:\n"
            # Sort by filename
            sorted_file_spectra = sorted(cluster_spectra['file_spectra'], 
                                       key=lambda x: x['filename'].lower())
            for i, spec_info in enumerate(sorted_file_spectra, 1):
                filename = spec_info['filename']
                metadata = spec_info['metadata']
                
                formatted_text += f"    {i:2d}. {filename}"
                
                # Add file path info if available
                file_path = metadata.get('file_path', '')
                if file_path:
                    # Show just the parent directory
                    import os
                    parent_dir = os.path.basename(os.path.dirname(file_path))
                    if parent_dir:
                        formatted_text += f" (from {parent_dir}/)"
                
                formatted_text += "\n"
            formatted_text += "\n"
        
        # Main application spectra
        if cluster_spectra['main_app_spectra']:
            formatted_text += "  Main Application Spectra:\n"
            # Sort by filename
            sorted_main_spectra = sorted(cluster_spectra['main_app_spectra'], 
                                       key=lambda x: x['filename'].lower())
            for i, spec_info in enumerate(sorted_main_spectra, 1):
                filename = spec_info['filename']
                formatted_text += f"    {i:2d}. {filename}\n"
            formatted_text += "\n"
        
        # Unknown/other spectra
        if cluster_spectra['unknown_spectra']:
            formatted_text += "  Other Spectra:\n"
            # Sort by filename
            sorted_unknown_spectra = sorted(cluster_spectra['unknown_spectra'], 
                                          key=lambda x: x['filename'].lower())
            for i, spec_info in enumerate(sorted_unknown_spectra, 1):
                filename = spec_info['filename']
                idx = spec_info['index']
                formatted_text += f"    {i:2d}. {filename} (index {idx})\n"
            formatted_text += "\n"
        
        return formatted_text

    # ... existing code ...

    def export_visualization(self):
        """Export current visualization to file."""
        try:
            current_tab = self.viz_tab_widget.currentIndex()
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Visualization", 
                "cluster_visualization.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            
            if filename:
                if current_tab == 0:  # Dendrogram
                    self.dendro_fig.savefig(filename, dpi=300, bbox_inches='tight')
                elif current_tab == 1:  # Heatmap
                    self.heatmap_fig.savefig(filename, dpi=300, bbox_inches='tight')
                elif current_tab == 2:  # Scatter plot
                    self.viz_fig.savefig(filename, dpi=300, bbox_inches='tight')
                
                QMessageBox.information(self, "Export Complete", f"Visualization saved to:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export visualization:\n{str(e)}")

    def export_analysis_results(self):
        """Export analysis results to file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Analysis Results", 
                "cluster_analysis_results.txt",
                "Text files (*.txt);;CSV files (*.csv)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                        f.write(self.analysis_results_text.toPlainText())
                
                QMessageBox.information(self, "Export Complete", f"Results saved to:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")

    # Refinement Methods
    def toggle_refinement_mode(self):
        """Toggle refinement mode on/off."""
        self.refinement_mode = not self.refinement_mode
        
        if self.refinement_mode:
            self.refinement_mode_btn.setText("Exit Refinement Mode")
            self.apply_refinement_btn.setEnabled(True)
            self.cancel_refinement_btn.setEnabled(True)
            self.selection_status.setText("Refinement mode: Click points to select clusters")
            
            # Enable interactive selection on refinement plot
            self.update_refinement_plot()
        else:
            self.refinement_mode_btn.setText("Start Refinement Mode")
            self.apply_refinement_btn.setEnabled(False)
            self.cancel_refinement_btn.setEnabled(False)
            self.selection_status.setText("No clusters selected")
            self.selected_points.clear()

    def undo_last_action(self):
        """Undo the last refinement action."""
        if self.undo_stack:
            # Restore previous state
            previous_state = self.undo_stack.pop()
            self.cluster_data['labels'] = previous_state['labels'].copy()
            
            # Update visualizations
            self.update_visualizations()
            self.update_refinement_plot()
            
            # Update undo button state
            self.undo_btn.setEnabled(len(self.undo_stack) > 0)

    def split_selected_cluster(self):
        """Split selected cluster into subclusters."""
        if not self.selected_points:
            QMessageBox.warning(self, "No Selection", "Please select a cluster to split.")
            return
        
        try:
            # Get the cluster to split (assume single cluster selection for now)
            selected_cluster = list(self.selected_points)[0]
            
            # Save current state for undo
            self.save_state_for_undo()
            
            # Get points in selected cluster
            cluster_mask = self.cluster_data['labels'] == selected_cluster
            cluster_features = self.cluster_data['features_scaled'][cluster_mask]
            
            if len(cluster_features) < 2:
                QMessageBox.warning(self, "Insufficient Data", "Cluster has too few points to split.")
            return
        
            # Perform subclustering
            n_subclusters = self.n_subclusters_spinbox.value()
            method = self.split_method_combo.currentText()
            
            if method == 'kmeans':
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                sub_labels = kmeans.fit_predict(cluster_features)
            else:  # hierarchical
                linkage_matrix = linkage(cluster_features, method='ward')
                sub_labels = fcluster(linkage_matrix, n_subclusters, criterion='maxclust')
            
            # Update cluster labels
            max_label = np.max(self.cluster_data['labels'])
            cluster_indices = np.where(cluster_mask)[0]
            
            for i, sub_label in enumerate(sub_labels):
                if sub_label > 1:  # Keep first subcluster as original label
                    self.cluster_data['labels'][cluster_indices[i]] = max_label + sub_label - 1
            
            # Update visualizations
            self.update_visualizations()
            self.update_refinement_plot()
            
            # Update undo button
            self.undo_btn.setEnabled(True)
                
        except Exception as e:
            QMessageBox.critical(self, "Split Error", f"Failed to split cluster:\n{str(e)}")

    def merge_selected_clusters(self):
        """Merge selected clusters into one."""
        if len(self.selected_points) < 2:
            QMessageBox.warning(self, "Insufficient Selection", "Please select at least 2 clusters to merge.")
            return
        
        try:
            # Save current state for undo
            self.save_state_for_undo()
            
            # Get selected clusters
            selected_clusters = list(self.selected_points)
            target_cluster = min(selected_clusters)  # Use lowest ID as target
            
            # Merge clusters
            for cluster_id in selected_clusters:
                if cluster_id != target_cluster:
                    self.cluster_data['labels'][self.cluster_data['labels'] == cluster_id] = target_cluster
            
            # Update visualizations
            self.update_visualizations()
            self.update_refinement_plot()
            
            # Update undo button
            self.undo_btn.setEnabled(True)
            
            # Clear selection
            self.selected_points.clear()
            self.update_selection_status()
            
        except Exception as e:
            QMessageBox.critical(self, "Merge Error", f"Failed to merge clusters:\n{str(e)}")

    def reset_selection(self):
        """Reset current cluster selection."""
        self.selected_points.clear()
        self.update_selection_status()
        self.update_refinement_plot()

    def apply_refinement(self):
        """Apply refinement changes and exit refinement mode."""
        try:
            # Update analysis results with new clustering
            self.update_analysis_results()
            
            # Exit refinement mode
            self.refinement_mode = False
            self.toggle_refinement_mode()
            
            QMessageBox.information(self, "Refinement Applied", "Cluster refinement has been applied successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply refinement:\n{str(e)}")

    def cancel_refinement(self):
        """Cancel refinement and restore original clustering."""
        try:
            # Restore original clustering if available
            if self.undo_stack:
                original_state = self.undo_stack[0]  # Get original state
                self.cluster_data['labels'] = original_state['labels'].copy()
                
                # Clear undo stack
                self.undo_stack.clear()
                self.undo_btn.setEnabled(False)
            
            # Exit refinement mode
            self.refinement_mode = False
            self.toggle_refinement_mode()
            
            # Update visualizations
            self.update_visualizations()
            
        except Exception as e:
            QMessageBox.critical(self, "Cancel Error", f"Failed to cancel refinement:\n{str(e)}")

    def save_state_for_undo(self):
        """Save current state for undo functionality."""
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)  # Remove oldest state
        
        state = {
            'labels': self.cluster_data['labels'].copy()
        }
        self.undo_stack.append(state)

    def update_selection_status(self):
        """Update the selection status display."""
        if not self.selected_points:
            self.selection_status.setText("No clusters selected")
            self.split_btn.setEnabled(False)
            self.merge_btn.setEnabled(False)
            self.reset_selection_btn.setEnabled(False)
        else:
            count = len(self.selected_points)
            clusters_text = ", ".join(map(str, sorted(self.selected_points)))
            self.selection_status.setText(f"{count} cluster(s) selected: {clusters_text}")
            self.split_btn.setEnabled(count == 1)
            self.merge_btn.setEnabled(count >= 2)
            self.reset_selection_btn.setEnabled(True)

    def update_refinement_plot(self):
        """Update the refinement visualization plot with interactive cluster selection."""
        try:
            if (self.cluster_data['features_scaled'] is None or 
                self.cluster_data['labels'] is None):
                return
        
            self.refinement_ax.clear()
            
            features = self.cluster_data['features_scaled']
            labels = self.cluster_data['labels']
            
            # Get visualization method
            viz_method = self.refinement_viz_combo.currentText() if hasattr(self, 'refinement_viz_combo') else 'PCA'
            
            # Apply dimensionality reduction
            if viz_method == 'UMAP':
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    coords = reducer.fit_transform(features)
                    xlabel, ylabel = 'UMAP 1', 'UMAP 2'
                except ImportError:
                    # Fallback to PCA if UMAP not available
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(features)
                    xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'
                    ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
            else:  # PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(features)
                xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'
                ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
            
            # Store coordinates for interaction
            self.refinement_coords = coords
            self.refinement_labels = labels
            
            # Plot clusters with different colors
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            # Store scatter plot objects for interaction
            self.refinement_scatters = {}
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                color = colors[i]
                
                # Highlight selected clusters
                if hasattr(self, 'selected_points') and label in self.selected_points:
                    scatter = self.refinement_ax.scatter(
                        coords[mask, 0], coords[mask, 1], 
                        c=[color], label=f'Cluster {label} (selected)',
                        alpha=0.8, s=100, edgecolors='red', linewidths=2,
                        picker=True
                    )
                else:
                    scatter = self.refinement_ax.scatter(
                        coords[mask, 0], coords[mask, 1], 
                        c=[color], label=f'Cluster {label}',
                        alpha=0.7, s=50, picker=True
                    )
                
                self.refinement_scatters[label] = scatter
            
            self.refinement_ax.set_xlabel(xlabel)
            self.refinement_ax.set_ylabel(ylabel)
            self.refinement_ax.set_title(f'Cluster Refinement View ({viz_method})')
            self.refinement_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.refinement_ax.grid(True, alpha=0.3)
            
            # Connect click event for cluster selection
            self.refinement_canvas.mpl_connect('button_press_event', self.on_refinement_click)
            
            self.refinement_fig.tight_layout()
            self.refinement_canvas.draw()
                
        except Exception as e:
            print(f"Error updating refinement plot: {str(e)}")
    
    def on_refinement_click(self, event):
        """Handle click events in refinement plot for cluster selection."""
        if event.inaxes != self.refinement_ax or not hasattr(self, 'refinement_coords'):
            return
        
        # Find nearest cluster point
        click_point = np.array([event.xdata, event.ydata])
        
        # Calculate distances to all points
        distances = np.linalg.norm(self.refinement_coords - click_point, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Get the cluster label of the nearest point
        clicked_cluster = self.refinement_labels[nearest_idx]
        
        # Initialize selected_points if not exists
        if not hasattr(self, 'selected_points'):
            self.selected_points = set()
        
        # Toggle cluster selection
        if clicked_cluster in self.selected_points:
            self.selected_points.remove(clicked_cluster)
        else:
            self.selected_points.add(clicked_cluster)
        
        # Update selection status and refinement controls
        self.update_selection_status()
        self.update_refinement_controls()
        
        # Refresh the plot to show selection changes
        self.update_refinement_plot()
    
    def update_refinement_controls(self):
        """Update refinement control button states based on selection."""
        if not hasattr(self, 'selected_points'):
            self.selected_points = set()
        
        num_selected = len(self.selected_points)
        
        # Enable/disable buttons based on selection
        if hasattr(self, 'split_btn'):
            self.split_btn.setEnabled(num_selected == 1)
        
        if hasattr(self, 'merge_btn'):
            self.merge_btn.setEnabled(num_selected >= 2)
        
        if hasattr(self, 'reset_selection_btn'):
            self.reset_selection_btn.setEnabled(num_selected > 0)
        
        if hasattr(self, 'apply_refinement_btn'):
            self.apply_refinement_btn.setEnabled(num_selected > 0)
        
        if hasattr(self, 'cancel_refinement_btn'):
            self.cancel_refinement_btn.setEnabled(num_selected > 0)

    # ... existing code ...

    # Advanced Analysis Methods
    def analyze_time_series_progression(self):
        """Analyze temporal cluster ordering and progression pathways."""
        try:
            if (self.cluster_data['features_scaled'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            # Check if temporal data is available
            if 'temporal_data' not in self.cluster_data or self.cluster_data['temporal_data'] is None:
                QMessageBox.warning(self, "No Time Data", 
                                  "No temporal data available. Please generate or load time data first.")
                return
            
            features = self.cluster_data['features_scaled']
            labels = self.cluster_data['labels']
            time_data = self.cluster_data['temporal_data']
            
            # Get analysis parameters
            ordering_method = self.temporal_ordering_combo.currentText()
            reference_cluster = self.reference_cluster_spinbox.value()
            
            # Calculate cluster centroids and temporal evolution
            unique_labels = np.unique(labels)
            cluster_centroids = {}
            cluster_time_evolution = {}
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                cluster_times = time_data[cluster_mask]
                
                # Calculate centroid
                centroid = np.mean(cluster_features, axis=0)
                cluster_centroids[label] = centroid
                
                # Calculate temporal evolution (mean time point for cluster)
                mean_time = np.mean(cluster_times)
                cluster_time_evolution[label] = {
                    'mean_time': mean_time,
                    'time_range': (np.min(cluster_times), np.max(cluster_times)),
                    'sample_count': len(cluster_times)
                }
            
            # Calculate progression pathway based on selected method
            if ordering_method == 'UMAP Distance':
                progression_order = self.calculate_umap_progression(cluster_centroids, reference_cluster)
            elif ordering_method == 'Cluster Centroid Distance':
                progression_order = self.calculate_centroid_progression(cluster_centroids, reference_cluster)
            else:  # Manual Ordering
                progression_order = sorted(unique_labels)
            
            # Calculate transition probabilities (if temporal overlap exists)
            transition_matrix = self.calculate_transition_probabilities(labels, time_data, unique_labels)
            
            # Create comprehensive visualization
            self.plot_time_series_analysis(
                cluster_centroids, cluster_time_evolution, progression_order, 
                transition_matrix, time_data, labels
            )
            
            # Generate detailed results
            results_text = self.generate_time_series_results(
                cluster_time_evolution, progression_order, transition_matrix, ordering_method
            )
            
            self.time_series_results.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Time-series progression analysis failed:\n{str(e)}")
    
    def calculate_umap_progression(self, cluster_centroids, reference_cluster):
        """Calculate progression order using UMAP embedding distances."""
        try:
            if not UMAP_AVAILABLE:
                # Fallback to centroid distance method
                return self.calculate_centroid_progression(cluster_centroids, reference_cluster)
            
            # Get centroids as array
            labels = list(cluster_centroids.keys())
            centroids = np.array([cluster_centroids[label] for label in labels])
            
            # Apply UMAP to centroids
            umap_model = umap.UMAP(n_components=2, n_neighbors=min(3, len(labels)-1), 
                                 random_state=42)
            umap_coords = umap_model.fit_transform(centroids)
            
            # Find reference cluster index
            ref_idx = labels.index(reference_cluster) if reference_cluster in labels else 0
            ref_coord = umap_coords[ref_idx]
            
            # Calculate distances from reference in UMAP space
            distances = []
            for i, coord in enumerate(umap_coords):
                dist = np.linalg.norm(coord - ref_coord)
                distances.append((dist, labels[i]))
            
            # Sort by distance to create progression order
            distances.sort()
            progression_order = [label for _, label in distances]
            
            return progression_order
            
        except Exception as e:
            print(f"UMAP progression calculation failed: {e}")
            return self.calculate_centroid_progression(cluster_centroids, reference_cluster)
    
    def calculate_centroid_progression(self, cluster_centroids, reference_cluster):
        """Calculate progression order using centroid distances."""
        labels = list(cluster_centroids.keys())
        
        # Find reference cluster
        if reference_cluster not in labels:
            reference_cluster = labels[0]
        
        ref_centroid = cluster_centroids[reference_cluster]
        
        # Calculate distances from reference cluster
        distances = []
        for label in labels:
            if label != reference_cluster:
                dist = np.linalg.norm(cluster_centroids[label] - ref_centroid)
                distances.append((dist, label))
        
        # Sort by distance
        distances.sort()
        progression_order = [reference_cluster] + [label for _, label in distances]
        
        return progression_order
    
    def calculate_transition_probabilities(self, labels, time_data, unique_labels):
        """Calculate transition probabilities between clusters based on temporal overlap."""
        n_clusters = len(unique_labels)
        transition_matrix = np.zeros((n_clusters, n_clusters))
        
        # Create label to index mapping
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        # Sort data by time
        time_sorted_indices = np.argsort(time_data)
        sorted_labels = labels[time_sorted_indices]
        sorted_times = time_data[time_sorted_indices]
        
        # Count transitions in temporal sequence
        for i in range(len(sorted_labels) - 1):
            current_cluster = sorted_labels[i]
            next_cluster = sorted_labels[i + 1]
            
            # Only count transitions within a reasonable time window
            time_diff = sorted_times[i + 1] - sorted_times[i]
            max_time_diff = np.std(time_data) * 2  # Within 2 standard deviations
            
            if time_diff <= max_time_diff:
                current_idx = label_to_idx[current_cluster]
                next_idx = label_to_idx[next_cluster]
                transition_matrix[current_idx, next_idx] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_clusters):
            if row_sums[i] > 0:
                transition_matrix[i, :] /= row_sums[i]
        
        return transition_matrix
    
    def plot_time_series_analysis(self, cluster_centroids, cluster_time_evolution, 
                                progression_order, transition_matrix, time_data, labels):
        """Create comprehensive time-series analysis visualization."""
        try:
            self.time_series_fig.clear()
            
            # Create 2x2 subplot layout
            ax1 = self.time_series_fig.add_subplot(2, 2, 1)  # Temporal distribution
            ax2 = self.time_series_fig.add_subplot(2, 2, 2)  # Progression pathway
            ax3 = self.time_series_fig.add_subplot(2, 2, 3)  # Transition matrix
            ax4 = self.time_series_fig.add_subplot(2, 2, 4)  # Cluster timeline
            
            unique_labels = list(cluster_centroids.keys())
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # 1. Temporal Distribution Plot
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_times = time_data[cluster_mask]
                ax1.hist(cluster_times, alpha=0.6, label=f'Cluster {label}', 
                        color=color_map[label], bins=20)
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Temporal Distribution of Clusters')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Progression Pathway Plot
            # Create a simple 2D representation of progression
            if len(progression_order) > 1:
                x_coords = np.arange(len(progression_order))
                y_coords = np.zeros(len(progression_order))
                
                for i, label in enumerate(progression_order):
                    evolution = cluster_time_evolution[label]
                    y_coords[i] = evolution['mean_time']
                    
                    ax2.scatter(x_coords[i], y_coords[i], 
                              c=[color_map[label]], s=100, 
                              label=f'Cluster {label}')
                
                # Draw progression arrows
                for i in range(len(progression_order) - 1):
                    ax2.annotate('', xy=(x_coords[i+1], y_coords[i+1]), 
                               xytext=(x_coords[i], y_coords[i]),
                               arrowprops=dict(arrowstyle='->', lw=2, alpha=0.7))
                
                ax2.set_xlabel('Progression Step')
                ax2.set_ylabel('Mean Time')
                ax2.set_title('Cluster Progression Pathway')
                ax2.grid(True, alpha=0.3)
                
                # Set x-axis labels
                ax2.set_xticks(x_coords)
                ax2.set_xticklabels([f'C{label}' for label in progression_order])
            
            # 3. Transition Matrix Heatmap
            if transition_matrix.size > 0:
                im = ax3.imshow(transition_matrix, cmap='Blues', aspect='auto')
                ax3.set_xlabel('To Cluster')
                ax3.set_ylabel('From Cluster')
                ax3.set_title('Transition Probability Matrix')
                
                # Add colorbar
                self.time_series_fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                
                # Set tick labels
                ax3.set_xticks(range(len(unique_labels)))
                ax3.set_yticks(range(len(unique_labels)))
                ax3.set_xticklabels([f'C{label}' for label in unique_labels])
                ax3.set_yticklabels([f'C{label}' for label in unique_labels])
                
                # Add probability values as text
                for i in range(len(unique_labels)):
                    for j in range(len(unique_labels)):
                        if transition_matrix[i, j] > 0.01:  # Only show significant transitions
                            ax3.text(j, i, f'{transition_matrix[i, j]:.2f}',
                                   ha='center', va='center', fontsize=8)
            
            # 4. Cluster Timeline
            for i, label in enumerate(unique_labels):
                cluster_mask = labels == label
                cluster_times = time_data[cluster_mask]
                y_positions = np.full_like(cluster_times, i)
                
                ax4.scatter(cluster_times, y_positions, 
                          c=[color_map[label]], alpha=0.6, s=30)
                
                # Add mean time line
                mean_time = cluster_time_evolution[label]['mean_time']
                ax4.axvline(x=mean_time, color=color_map[label], 
                          linestyle='--', alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Cluster')
            ax4.set_title('Cluster Timeline Distribution')
            ax4.set_yticks(range(len(unique_labels)))
            ax4.set_yticklabels([f'Cluster {label}' for label in unique_labels])
            ax4.grid(True, alpha=0.3)
            
            self.time_series_fig.tight_layout()
            self.time_series_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting time-series analysis: {str(e)}")
    
    def generate_time_series_results(self, cluster_time_evolution, progression_order, 
                                   transition_matrix, ordering_method):
        """Generate detailed text results for time-series analysis."""
        results_text = f"Time-Series Progression Analysis Results\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Analysis Method: {ordering_method}\n"
        results_text += f"Number of Clusters: {len(cluster_time_evolution)}\n\n"
        
        # Cluster temporal characteristics
        results_text += "Cluster Temporal Characteristics:\n"
        results_text += "-" * 35 + "\n"
        
        for label, evolution in cluster_time_evolution.items():
            mean_time = evolution['mean_time']
            time_range = evolution['time_range']
            sample_count = evolution['sample_count']
            
            results_text += f"Cluster {label}:\n"
            results_text += f"  • Mean time: {mean_time:.2f}\n"
            results_text += f"  • Time range: {time_range[0]:.2f} - {time_range[1]:.2f}\n"
            results_text += f"  • Sample count: {sample_count}\n"
            results_text += f"  • Time span: {time_range[1] - time_range[0]:.2f}\n\n"
        
        # Progression pathway
        results_text += "Suggested Progression Pathway:\n"
        results_text += "-" * 30 + "\n"
        
        pathway_text = " → ".join([f"Cluster {label}" for label in progression_order])
        results_text += f"{pathway_text}\n\n"
        
        # Temporal ordering
        results_text += "Temporal Ordering (by mean time):\n"
        results_text += "-" * 35 + "\n"
        
        # Sort clusters by mean time
        temporal_order = sorted(cluster_time_evolution.items(), 
                              key=lambda x: x[1]['mean_time'])
        
        for i, (label, evolution) in enumerate(temporal_order):
            results_text += f"{i+1}. Cluster {label} (t = {evolution['mean_time']:.2f})\n"
        
        results_text += "\n"
        
        # Transition analysis
        if transition_matrix.size > 0:
            results_text += "Significant Transitions (>10% probability):\n"
            results_text += "-" * 45 + "\n"
            
            unique_labels = list(cluster_time_evolution.keys())
            significant_transitions = []
            
            for i, from_label in enumerate(unique_labels):
                for j, to_label in enumerate(unique_labels):
                    prob = transition_matrix[i, j]
                    if prob > 0.1 and i != j:  # Significant non-self transitions
                        significant_transitions.append((from_label, to_label, prob))
            
            if significant_transitions:
                # Sort by probability
                significant_transitions.sort(key=lambda x: x[2], reverse=True)
                
                for from_label, to_label, prob in significant_transitions:
                    results_text += f"  • Cluster {from_label} → Cluster {to_label}: {prob:.1%}\n"
            else:
                results_text += "  • No significant transitions detected\n"
        
        return results_text

    def calculate_intercluster_distances(self):
        """Calculate inter-cluster distances for progression analysis."""
        if (self.cluster_data['features_scaled'] is None or 
            self.cluster_data['labels'] is None):
            QMessageBox.warning(self, "No Data", "No clustering data available.")
            return
        
        try:
            features = self.cluster_data['features_scaled']
            labels = self.cluster_data['labels']
        
            # Calculate cluster centroids
            unique_labels = np.unique(labels)
            centroids = []
            cluster_info = {}
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                centroid = np.mean(cluster_features, axis=0)
                centroids.append(centroid)
                
                # Calculate cluster statistics
                cluster_info[label] = {
                    'size': np.sum(cluster_mask),
                    'centroid': centroid,
                    'std': np.std(cluster_features, axis=0),
                    'mean_std': np.mean(np.std(cluster_features, axis=0))
                }
            
            centroids = np.array(centroids)
            
            # Calculate pairwise distances between centroids using multiple metrics
            euclidean_distances = pdist(centroids, metric='euclidean')
            cosine_distances = pdist(centroids, metric='cosine')
            correlation_distances = pdist(centroids, metric='correlation')
            
            euclidean_matrix = squareform(euclidean_distances)
            cosine_matrix = squareform(cosine_distances)
            correlation_matrix = squareform(correlation_distances)
            
            # Create comprehensive visualization
            self.plot_intercluster_distances(
                euclidean_matrix, cosine_matrix, correlation_matrix, 
                unique_labels, cluster_info
            )
            
            # Generate detailed text results
            results_text = self.generate_distance_results(
                euclidean_matrix, cosine_matrix, correlation_matrix, 
                unique_labels, cluster_info
            )
            
            self.time_series_results.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate distances:\n{str(e)}")
    
    def plot_intercluster_distances(self, euclidean_matrix, cosine_matrix, correlation_matrix, 
                                  unique_labels, cluster_info):
        """Create comprehensive visualization of inter-cluster distances."""
        try:
            self.time_series_fig.clear()
            
            # Create 2x2 subplot layout
            ax1 = self.time_series_fig.add_subplot(2, 2, 1)  # Euclidean distance heatmap
            ax2 = self.time_series_fig.add_subplot(2, 2, 2)  # Cosine distance heatmap
            ax3 = self.time_series_fig.add_subplot(2, 2, 3)  # Correlation distance heatmap
            ax4 = self.time_series_fig.add_subplot(2, 2, 4)  # Distance comparison plot
            
            # 1. Euclidean Distance Heatmap
            im1 = ax1.imshow(euclidean_matrix, cmap='viridis', aspect='auto')
            ax1.set_title('Euclidean Distances')
            ax1.set_xlabel('Cluster')
            ax1.set_ylabel('Cluster')
            
            # Add text annotations
            for i in range(len(unique_labels)):
                for j in range(len(unique_labels)):
                    if i != j:  # Don't show diagonal (distance to self = 0)
                        text_color = 'white' if euclidean_matrix[i, j] > np.max(euclidean_matrix) * 0.6 else 'black'
                        ax1.text(j, i, f'{euclidean_matrix[i, j]:.2f}',
                               ha='center', va='center', color=text_color, fontsize=9)
            
            ax1.set_xticks(range(len(unique_labels)))
            ax1.set_yticks(range(len(unique_labels)))
            ax1.set_xticklabels([f'C{label}' for label in unique_labels])
            ax1.set_yticklabels([f'C{label}' for label in unique_labels])
            
            # Add colorbar
            self.time_series_fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # 2. Cosine Distance Heatmap
            im2 = ax2.imshow(cosine_matrix, cmap='plasma', aspect='auto')
            ax2.set_title('Cosine Distances')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Cluster')
            
            # Add text annotations
            for i in range(len(unique_labels)):
                for j in range(len(unique_labels)):
                    if i != j:
                        text_color = 'white' if cosine_matrix[i, j] > np.max(cosine_matrix) * 0.6 else 'black'
                        ax2.text(j, i, f'{cosine_matrix[i, j]:.3f}',
                               ha='center', va='center', color=text_color, fontsize=9)
            
            ax2.set_xticks(range(len(unique_labels)))
            ax2.set_yticks(range(len(unique_labels)))
            ax2.set_xticklabels([f'C{label}' for label in unique_labels])
            ax2.set_yticklabels([f'C{label}' for label in unique_labels])
            
            # Add colorbar
            self.time_series_fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # 3. Correlation Distance Heatmap
            im3 = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            ax3.set_title('Correlation Distances')
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel('Cluster')
            
            # Add text annotations
            for i in range(len(unique_labels)):
                for j in range(len(unique_labels)):
                    if i != j:
                        text_color = 'white' if correlation_matrix[i, j] > np.max(correlation_matrix) * 0.6 else 'black'
                        ax3.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                               ha='center', va='center', color=text_color, fontsize=9)
            
            ax3.set_xticks(range(len(unique_labels)))
            ax3.set_yticks(range(len(unique_labels)))
            ax3.set_xticklabels([f'C{label}' for label in unique_labels])
            ax3.set_yticklabels([f'C{label}' for label in unique_labels])
            
            # Add colorbar
            self.time_series_fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            # 4. Distance Comparison Plot
            # Create a comparison of different distance metrics for cluster pairs
            cluster_pairs = []
            euclidean_vals = []
            cosine_vals = []
            correlation_vals = []
            
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    cluster_pairs.append(f'C{unique_labels[i]}-C{unique_labels[j]}')
                    euclidean_vals.append(euclidean_matrix[i, j])
                    cosine_vals.append(cosine_matrix[i, j])
                    correlation_vals.append(correlation_matrix[i, j])
            
            x_pos = np.arange(len(cluster_pairs))
            width = 0.25
            
            # Normalize values to [0, 1] for comparison
            euclidean_norm = np.array(euclidean_vals) / np.max(euclidean_vals) if np.max(euclidean_vals) > 0 else np.zeros_like(euclidean_vals)
            cosine_norm = np.array(cosine_vals) / np.max(cosine_vals) if np.max(cosine_vals) > 0 else np.zeros_like(cosine_vals)
            correlation_norm = np.array(correlation_vals) / np.max(correlation_vals) if np.max(correlation_vals) > 0 else np.zeros_like(correlation_vals)
            
            ax4.bar(x_pos - width, euclidean_norm, width, label='Euclidean', alpha=0.8)
            ax4.bar(x_pos, cosine_norm, width, label='Cosine', alpha=0.8)
            ax4.bar(x_pos + width, correlation_norm, width, label='Correlation', alpha=0.8)
            
            ax4.set_xlabel('Cluster Pairs')
            ax4.set_ylabel('Normalized Distance')
            ax4.set_title('Distance Metric Comparison')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(cluster_pairs, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            self.time_series_fig.tight_layout()
            self.time_series_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting inter-cluster distances: {str(e)}")
    
    def generate_distance_results(self, euclidean_matrix, cosine_matrix, correlation_matrix, 
                                unique_labels, cluster_info):
        """Generate detailed text results for inter-cluster distance analysis."""
        results_text = "Inter-cluster Distance Analysis Results\n"
        results_text += "=" * 45 + "\n\n"
        
        results_text += f"Number of Clusters: {len(unique_labels)}\n\n"
        
        # Cluster characteristics
        results_text += "Cluster Characteristics:\n"
        results_text += "-" * 25 + "\n"
        
        for label in unique_labels:
            info = cluster_info[label]
            results_text += f"Cluster {label}:\n"
            results_text += f"  • Size: {info['size']} spectra\n"
            results_text += f"  • Mean std deviation: {info['mean_std']:.3f}\n"
            results_text += f"  • Compactness: {'High' if info['mean_std'] < 0.5 else 'Medium' if info['mean_std'] < 1.0 else 'Low'}\n\n"
        
        # Distance analysis for each metric
        metrics = [
            ('Euclidean', euclidean_matrix),
            ('Cosine', cosine_matrix),
            ('Correlation', correlation_matrix)
        ]
        
        for metric_name, distance_matrix in metrics:
            results_text += f"{metric_name} Distance Analysis:\n"
            results_text += "-" * (len(metric_name) + 18) + "\n"
            
            # Find closest and farthest cluster pairs
            distances_list = []
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    dist = distance_matrix[i, j]
                    distances_list.append((dist, unique_labels[i], unique_labels[j]))
            
            distances_list.sort()
            
            # Closest pairs
            results_text += "Closest cluster pairs:\n"
            for i, (dist, label1, label2) in enumerate(distances_list[:3]):
                results_text += f"  {i+1}. Cluster {label1} ↔ Cluster {label2}: {dist:.3f}\n"
            
            # Farthest pairs
            results_text += "\nFarthest cluster pairs:\n"
            for i, (dist, label1, label2) in enumerate(distances_list[-3:]):
                results_text += f"  {i+1}. Cluster {label1} ↔ Cluster {label2}: {dist:.3f}\n"
            
            # Distance statistics
            all_distances = [dist for dist, _, _ in distances_list]
            results_text += f"\nDistance Statistics:\n"
            results_text += f"  • Mean distance: {np.mean(all_distances):.3f}\n"
            results_text += f"  • Std deviation: {np.std(all_distances):.3f}\n"
            results_text += f"  • Min distance: {np.min(all_distances):.3f}\n"
            results_text += f"  • Max distance: {np.max(all_distances):.3f}\n\n"
        
        # Similarity recommendations
        results_text += "Cluster Similarity Recommendations:\n"
        results_text += "-" * 35 + "\n"
        
        # Use euclidean distance for recommendations
        euclidean_list = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                dist = euclidean_matrix[i, j]
                euclidean_list.append((dist, unique_labels[i], unique_labels[j]))
        
        euclidean_list.sort()
        
        # Very similar clusters (might be over-segmented)
        threshold_similar = np.percentile([d[0] for d in euclidean_list], 25)
        similar_pairs = [pair for pair in euclidean_list if pair[0] <= threshold_similar]
        
        if similar_pairs:
            results_text += "Very similar clusters (consider merging):\n"
            for dist, label1, label2 in similar_pairs:
                results_text += f"  • Cluster {label1} ↔ Cluster {label2} (distance: {dist:.3f})\n"
        else:
            results_text += "• No overly similar clusters detected\n"
        
        results_text += "\n"
        
        # Well-separated clusters
        threshold_separated = np.percentile([d[0] for d in euclidean_list], 75)
        separated_pairs = [pair for pair in euclidean_list if pair[0] >= threshold_separated]
        
        if separated_pairs:
            results_text += "Well-separated clusters:\n"
            for dist, label1, label2 in separated_pairs[:5]:  # Show top 5
                results_text += f"  • Cluster {label1} ↔ Cluster {label2} (distance: {dist:.3f})\n"
        
        return results_text

    def update_time_input_controls(self):
        """Update visibility and content of time input controls based on method selection."""
        if not hasattr(self, 'data_input_combo'):
            return
            
        method = self.data_input_combo.currentText()
        
        # Show/hide interval controls
        if hasattr(self, 'interval_controls'):
            self.interval_controls.setVisible(method == 'Constant Interval')
        
        # Update spectra count
        if hasattr(self, 'spectra_count_label') and hasattr(self, 'cluster_data'):
            if self.cluster_data['intensities'] is not None:
                count = len(self.cluster_data['intensities'])
                self.spectra_count_label.setText(f"{count} spectra loaded")
                
                # Update time data preview for constant interval
                if method == 'Constant Interval' and hasattr(self, 'time_data_display'):
                    start_time = self.start_time_spinbox.value()
                    interval = self.time_interval_spinbox.value()
                    units = self.time_units_edit.text()
                    
                    time_points = [start_time + i * interval for i in range(count)]
                    preview_text = f"Time points ({units}): {time_points[:5]}"
                    if count > 5:
                        preview_text += f" ... (showing first 5 of {count})"
                    
                    self.time_data_display.setText(preview_text)

    def input_time_data(self):
        """Generate or load time data based on selected method."""
        try:
            if self.cluster_data['intensities'] is None:
                QMessageBox.warning(self, "No Data", "No spectral data loaded.")
                return
            
            method = self.data_input_combo.currentText()
            count = len(self.cluster_data['intensities'])
            
            if method == 'Constant Interval':
                start_time = self.start_time_spinbox.value()
                interval = self.time_interval_spinbox.value()
                time_data = np.array([start_time + i * interval for i in range(count)])
                
            elif method == 'Load from File':
                filename, _ = QFileDialog.getOpenFileName(
                    self, "Load Time Data", "", 
                    "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
                )
                
                if not filename:
                    return
                
                time_data = np.loadtxt(filename)
                if len(time_data) != count:
                    QMessageBox.warning(self, "Data Mismatch", 
                                      f"Time data has {len(time_data)} points but "
                                      f"spectral data has {count} spectra.")
                    return
        
            elif method == 'Extract from Metadata':
                # Try to extract time data from spectrum metadata
                if 'spectrum_metadata' in self.cluster_data:
                    time_data = []
                    for metadata in self.cluster_data['spectrum_metadata']:
                        # Look for time-related fields in metadata
                        time_value = None
                        for key in ['time', 'timestamp', 'Time', 'Timestamp']:
                            if key in metadata:
                                try:
                                    time_value = float(metadata[key])
                                    break
                                except:
                                    continue
                        
                        if time_value is None:
                            # Extract from filename if possible
                            filename = metadata.get('filename', '')
                            import re
                            time_match = re.search(r'(\d+\.?\d*)', filename)
                            if time_match:
                                time_value = float(time_match.group(1))
                            else:
                                time_value = len(time_data)  # Default sequential
                        
                        time_data.append(time_value)
                    
                    time_data = np.array(time_data)
                else:
                    QMessageBox.warning(self, "No Metadata", "No metadata available for time extraction.")
                    return
            
            # Store time data
            self.cluster_data['temporal_data'] = time_data
            
            # Update display
            units = self.time_units_edit.text()
            self.time_data_display.setText(
                f"Time data loaded: {len(time_data)} points, "
                f"range: {np.min(time_data):.2f} - {np.max(time_data):.2f} {units}"
            )
            
            QMessageBox.information(self, "Time Data Loaded", 
                                  f"Successfully loaded time data for {len(time_data)} spectra.")
            
        except Exception as e:
            QMessageBox.critical(self, "Time Data Error", f"Failed to load time data:\n{str(e)}")

    def fit_kinetic_models(self):
        """Fit kinetic models to cluster populations over time."""
        try:
            # Check data availability
            if (self.cluster_data['intensities'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            if 'temporal_data' not in self.cluster_data or self.cluster_data['temporal_data'] is None:
                QMessageBox.warning(self, "No Time Data", 
                                  "Please generate or load time data first.")
                return
            
            # Get data
            labels = self.cluster_data['labels']
            time_data = self.cluster_data['temporal_data']
            unique_labels = np.unique(labels)
            
            # Calculate cluster populations over time
            cluster_populations = {}
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_populations[label] = np.sum(cluster_mask)
            
            # If only one time point per cluster, create time series based on ordering
            if len(time_data) == len(unique_labels):
                # Assume data is ordered by time
                time_points = time_data
                population_data = [cluster_populations[label] for label in unique_labels]
            else:
                # Multiple spectra per time point - calculate average populations
                time_points = np.unique(time_data)
                population_data = {}
                
                for label in unique_labels:
                    populations = []
                    for t in time_points:
                        time_mask = time_data == t
                        cluster_at_time = labels[time_mask]
                        pop_at_time = np.sum(cluster_at_time == label) / len(cluster_at_time)
                        populations.append(pop_at_time)
                    population_data[label] = np.array(populations)
            
            # Get selected kinetic model
            model_name = self.kinetic_model_combo.currentText()
            
            # Define kinetic model functions
            def pseudo_first_order(t, k, A0, C):
                """Pseudo-first-order: A(t) = A0 * exp(-k*t) + C"""
                return A0 * np.exp(-k * t) + C
            
            def pseudo_second_order(t, k, A0, C):
                """Pseudo-second-order: A(t) = A0 / (1 + k*A0*t) + C"""
                return A0 / (1 + k * A0 * t) + C
            
            def avrami_equation(t, k, n, A0, C):
                """Avrami equation: A(t) = A0 * exp(-(k*t)^n) + C"""
                return A0 * np.exp(-np.power(k * t, n)) + C
            
            def diffusion_controlled(t, k, A0, C):
                """Diffusion-controlled: A(t) = A0 * (1 - (k*t)^0.5) + C"""
                return A0 * (1 - np.sqrt(k * t)) + C
            
            def multi_exponential(t, k1, k2, A1, A2, C):
                """Multi-exponential: A(t) = A1*exp(-k1*t) + A2*exp(-k2*t) + C"""
                return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C
            
            # Select model function
            if model_name == 'Pseudo-first-order':
                model_func = pseudo_first_order
                param_names = ['Rate constant k', 'Initial amount A0', 'Constant C']
                initial_guess = [0.1, 1.0, 0.0]
            elif model_name == 'Pseudo-second-order':
                model_func = pseudo_second_order
                param_names = ['Rate constant k', 'Initial amount A0', 'Constant C']
                initial_guess = [0.1, 1.0, 0.0]
            elif model_name == 'Avrami Equation':
                model_func = avrami_equation
                param_names = ['Rate constant k', 'Avrami exponent n', 'Initial amount A0', 'Constant C']
                initial_guess = [0.1, 1.5, 1.0, 0.0]
            elif model_name == 'Diffusion-controlled':
                model_func = diffusion_controlled
                param_names = ['Rate constant k', 'Initial amount A0', 'Constant C']
                initial_guess = [0.1, 1.0, 0.0]
            elif model_name == 'Multi-exponential':
                model_func = multi_exponential
                param_names = ['Rate constant k1', 'Rate constant k2', 'Amplitude A1', 'Amplitude A2', 'Constant C']
                initial_guess = [0.1, 0.01, 0.5, 0.5, 0.0]
            
            # Store fitted models
            fitted_models = {}
            fit_statistics = {}
            
            # Fit models for each cluster or overall population
            if isinstance(population_data, dict):
                # Multiple clusters
                for label in unique_labels:
                    pop_values = population_data[label]
                    
                    try:
                        # Fit the model
                        popt, pcov = curve_fit(model_func, time_points, pop_values, 
                                             p0=initial_guess, maxfev=5000)
                        
                        # Calculate fit statistics
                        y_pred = model_func(time_points, *popt)
                        r_squared = 1 - np.sum((pop_values - y_pred)**2) / np.sum((pop_values - np.mean(pop_values))**2)
                        rmse = np.sqrt(np.mean((pop_values - y_pred)**2))
                        
                        fitted_models[label] = {
                            'parameters': popt,
                            'covariance': pcov,
                            'r_squared': r_squared,
                            'rmse': rmse,
                            'time_points': time_points,
                            'experimental_data': pop_values,
                            'fitted_data': y_pred
                        }
                        
                        # Calculate parameter uncertainties
                        param_errors = np.sqrt(np.diag(pcov))
                        fit_statistics[label] = {
                            'parameters': dict(zip(param_names, popt)),
                            'errors': dict(zip(param_names, param_errors)),
                            'r_squared': r_squared,
                            'rmse': rmse
                        }
                        
                    except Exception as e:
                        print(f"Failed to fit model for cluster {label}: {str(e)}")
                        fitted_models[label] = None
                        fit_statistics[label] = {'error': str(e)}
            else:
                # Single population data
                try:
                    popt, pcov = curve_fit(model_func, time_points, population_data, 
                                         p0=initial_guess, maxfev=5000)
                    
                    y_pred = model_func(time_points, *popt)
                    r_squared = 1 - np.sum((population_data - y_pred)**2) / np.sum((population_data - np.mean(population_data))**2)
                    rmse = np.sqrt(np.mean((population_data - y_pred)**2))
                    
                    fitted_models['overall'] = {
                        'parameters': popt,
                        'covariance': pcov,
                        'r_squared': r_squared,
                        'rmse': rmse,
                        'time_points': time_points,
                        'experimental_data': population_data,
                        'fitted_data': y_pred
                    }
                    
                    param_errors = np.sqrt(np.diag(pcov))
                    fit_statistics['overall'] = {
                        'parameters': dict(zip(param_names, popt)),
                        'errors': dict(zip(param_names, param_errors)),
                        'r_squared': r_squared,
                        'rmse': rmse
                    }
                    
                except Exception as e:
                    QMessageBox.critical(self, "Fitting Error", f"Failed to fit kinetic model:\n{str(e)}")
                    return
            
            # Store results
            self.analysis_results['kinetic_models'] = {
                'model_name': model_name,
                'fitted_models': fitted_models,
                'fit_statistics': fit_statistics,
                'model_function': model_func,
                'parameter_names': param_names
            }
            
            # Plot results
            self.plot_kinetic_models(fitted_models, model_name)
            
            # Display results
            self.display_kinetic_results(fit_statistics, model_name)
            
            QMessageBox.information(self, "Kinetic Modeling Complete", 
                                  f"Successfully fitted {model_name} model to cluster populations.")
            
        except Exception as e:
            QMessageBox.critical(self, "Kinetic Modeling Error", 
                               f"Failed to fit kinetic models:\n{str(e)}")

    def compare_kinetic_models(self):
        """Compare different kinetic models for best fit."""
        try:
            # Check data availability
            if (self.cluster_data['intensities'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            if 'temporal_data' not in self.cluster_data or self.cluster_data['temporal_data'] is None:
                QMessageBox.warning(self, "No Time Data", 
                                  "Please generate or load time data first.")
                return
            
            # Get data
            labels = self.cluster_data['labels']
            time_data = self.cluster_data['temporal_data']
            unique_labels = np.unique(labels)
            
            # Calculate cluster populations over time
            if len(time_data) == len(unique_labels):
                time_points = time_data
                # Use relative populations
                total_spectra = len(labels)
                population_data = {}
                for label in unique_labels:
                    cluster_mask = labels == label
                    population_data[label] = np.sum(cluster_mask) / total_spectra
            else:
                time_points = np.unique(time_data)
                population_data = {}
                
                for label in unique_labels:
                    populations = []
                    for t in time_points:
                        time_mask = time_data == t
                        cluster_at_time = labels[time_mask]
                        if len(cluster_at_time) > 0:
                            pop_at_time = np.sum(cluster_at_time == label) / len(cluster_at_time)
                        else:
                            pop_at_time = 0
                        populations.append(pop_at_time)
                    population_data[label] = np.array(populations)
            
            # Define all kinetic models
            models = {
                'Pseudo-first-order': {
                    'func': lambda t, k, A0, C: A0 * np.exp(-k * t) + C,
                    'initial_guess': [0.1, 1.0, 0.0],
                    'param_names': ['Rate constant k', 'Initial amount A0', 'Constant C']
                },
                'Pseudo-second-order': {
                    'func': lambda t, k, A0, C: A0 / (1 + k * A0 * t) + C,
                    'initial_guess': [0.1, 1.0, 0.0],
                    'param_names': ['Rate constant k', 'Initial amount A0', 'Constant C']
                },
                'Avrami Equation': {
                    'func': lambda t, k, n, A0, C: A0 * np.exp(-np.power(k * t, n)) + C,
                    'initial_guess': [0.1, 1.5, 1.0, 0.0],
                    'param_names': ['Rate constant k', 'Avrami exponent n', 'Initial amount A0', 'Constant C']
                },
                'Diffusion-controlled': {
                    'func': lambda t, k, A0, C: A0 * (1 - np.sqrt(np.maximum(k * t, 0))) + C,
                    'initial_guess': [0.1, 1.0, 0.0],
                    'param_names': ['Rate constant k', 'Initial amount A0', 'Constant C']
                }
            }
            
            # Fit all models and compare
            model_comparison = {}
            
            for model_name, model_info in models.items():
                model_func = model_info['func']
                initial_guess = model_info['initial_guess']
                param_names = model_info['param_names']
                
                cluster_fits = {}
                total_aic = 0
                total_bic = 0
                total_r_squared = 0
                successful_fits = 0
                
                for label in unique_labels:
                    if isinstance(population_data, dict):
                        pop_values = population_data[label]
                    else:
                        pop_values = population_data
                    
                    try:
                        # Fit the model
                        popt, pcov = curve_fit(model_func, time_points, pop_values, 
                                             p0=initial_guess, maxfev=5000)
                        
                        # Calculate statistics
                        y_pred = model_func(time_points, *popt)
                        n = len(time_points)
                        k = len(popt)  # number of parameters
                        
                        # R-squared
                        ss_res = np.sum((pop_values - y_pred)**2)
                        ss_tot = np.sum((pop_values - np.mean(pop_values))**2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        # AIC and BIC
                        mse = ss_res / n
                        if mse > 0:
                            aic = n * np.log(mse) + 2 * k
                            bic = n * np.log(mse) + k * np.log(n)
                        else:
                            aic = float('inf')
                            bic = float('inf')
                        
                        cluster_fits[label] = {
                            'parameters': popt,
                            'r_squared': r_squared,
                            'aic': aic,
                            'bic': bic,
                            'rmse': np.sqrt(mse)
                        }
                        
                        total_aic += aic
                        total_bic += bic
                        total_r_squared += r_squared
                        successful_fits += 1
                        
                    except Exception as e:
                        cluster_fits[label] = {'error': str(e)}
                
                if successful_fits > 0:
                    avg_r_squared = total_r_squared / successful_fits
                    avg_aic = total_aic / successful_fits
                    avg_bic = total_bic / successful_fits
                else:
                    avg_r_squared = 0
                    avg_aic = float('inf')
                    avg_bic = float('inf')
                
                model_comparison[model_name] = {
                    'cluster_fits': cluster_fits,
                    'avg_r_squared': avg_r_squared,
                    'avg_aic': avg_aic,
                    'avg_bic': avg_bic,
                    'successful_fits': successful_fits,
                    'total_clusters': len(unique_labels)
                }
            
            # Find best model based on average R-squared
            best_model = max(model_comparison.keys(), 
                           key=lambda x: model_comparison[x]['avg_r_squared'])
            
            # Store comparison results
            self.analysis_results['model_comparison'] = {
                'comparison': model_comparison,
                'best_model': best_model,
                'ranking_metric': 'avg_r_squared'
            }
            
            # Plot comparison
            self.plot_model_comparison(model_comparison, population_data, time_points)
            
            # Display comparison results
            self.display_model_comparison(model_comparison, best_model)
            
            QMessageBox.information(self, "Model Comparison Complete", 
                                  f"Best fitting model: {best_model}\n"
                                  f"Based on average R-squared: {model_comparison[best_model]['avg_r_squared']:.4f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Model Comparison Error", 
                               f"Failed to compare kinetic models:\n{str(e)}")

    def plot_kinetic_models(self, fitted_models, model_name):
        """Plot kinetic model fitting results."""
        try:
            self.kinetics_fig.clear()
            
            # Create subplots
            if len(fitted_models) == 1:
                # Single plot
                ax = self.kinetics_fig.add_subplot(111)
                
                for label, fit_data in fitted_models.items():
                    if fit_data is not None:
                        time_points = fit_data['time_points']
                        experimental = fit_data['experimental_data']
                        fitted = fit_data['fitted_data']
                        r_squared = fit_data['r_squared']
                        
                        ax.scatter(time_points, experimental, label=f'Experimental (Cluster {label})', 
                                 alpha=0.7, s=60)
                        ax.plot(time_points, fitted, '--', linewidth=2, 
                               label=f'Fitted {model_name} (R² = {r_squared:.3f})')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Population')
                ax.set_title(f'Kinetic Model Fitting: {model_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            else:
                # Multiple subplots for multiple clusters
                n_clusters = len([f for f in fitted_models.values() if f is not None])
                if n_clusters == 0:
                    return
                
                n_cols = min(3, n_clusters)
                n_rows = (n_clusters + n_cols - 1) // n_cols
                
                plot_idx = 1
                for label, fit_data in fitted_models.items():
                    if fit_data is not None:
                        ax = self.kinetics_fig.add_subplot(n_rows, n_cols, plot_idx)
                        
                        time_points = fit_data['time_points']
                        experimental = fit_data['experimental_data']
                        fitted = fit_data['fitted_data']
                        r_squared = fit_data['r_squared']
                        
                        ax.scatter(time_points, experimental, alpha=0.7, s=50, 
                                 color=f'C{plot_idx-1}', label='Experimental')
                        ax.plot(time_points, fitted, '--', linewidth=2, 
                               color=f'C{plot_idx-1}', label=f'Fitted (R² = {r_squared:.3f})')
                        
                        ax.set_title(f'Cluster {label}')
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Population')
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
                        
                        plot_idx += 1
                
                self.kinetics_fig.suptitle(f'Kinetic Model Fitting: {model_name}', fontsize=14)
            
            self.kinetics_fig.tight_layout()
            self.kinetics_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting kinetic models: {str(e)}")
    
    def display_kinetic_results(self, fit_statistics, model_name):
        """Display kinetic model fitting results in text form."""
        try:
            results_text = f"Kinetic Model Fitting Results: {model_name}\n"
            results_text += "=" * 60 + "\n\n"
            
            for label, stats in fit_statistics.items():
                if 'error' in stats:
                    results_text += f"Cluster {label}: FITTING FAILED\n"
                    results_text += f"  Error: {stats['error']}\n\n"
                    continue
                
                results_text += f"Cluster {label}:\n"
                results_text += f"  R-squared: {stats['r_squared']:.4f}\n"
                results_text += f"  RMSE: {stats['rmse']:.4f}\n"
                results_text += "  Parameters:\n"
                
                for param_name, value in stats['parameters'].items():
                    error = stats['errors'][param_name]
                    results_text += f"    {param_name}: {value:.4f} ± {error:.4f}\n"
                
                results_text += "\n"
            
            self.kinetics_results.setText(results_text)
            
        except Exception as e:
            print(f"Error displaying kinetic results: {str(e)}")
    
    def plot_model_comparison(self, model_comparison, population_data, time_points):
        """Plot comparison of different kinetic models."""
        try:
            self.kinetics_fig.clear()
            
            # Create subplots for model comparison
            n_models = len(model_comparison)
            n_cols = min(2, n_models)
            n_rows = (n_models + n_cols - 1) // n_cols
            
            # First plot: R-squared comparison
            ax1 = self.kinetics_fig.add_subplot(2, 2, 1)
            model_names = list(model_comparison.keys())
            r_squared_values = [model_comparison[name]['avg_r_squared'] for name in model_names]
            
            bars = ax1.bar(model_names, r_squared_values, alpha=0.7)
            ax1.set_ylabel('Average R-squared')
            ax1.set_title('Model Comparison: R-squared')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Annotate bars with values
            for bar, value in zip(bars, r_squared_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            # Second plot: AIC comparison
            ax2 = self.kinetics_fig.add_subplot(2, 2, 2)
            aic_values = [model_comparison[name]['avg_aic'] for name in model_names]
            finite_aic = [v for v in aic_values if np.isfinite(v)]
            
            if finite_aic:
                bars2 = ax2.bar(model_names, aic_values, alpha=0.7, color='orange')
                ax2.set_ylabel('Average AIC')
                ax2.set_title('Model Comparison: AIC (lower is better)')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars2, aic_values):
                    if np.isfinite(value):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.1f}', ha='center', va='bottom')
            
            # Third plot: Successful fits comparison
            ax3 = self.kinetics_fig.add_subplot(2, 2, 3)
            success_rates = [model_comparison[name]['successful_fits'] / model_comparison[name]['total_clusters'] 
                           for name in model_names]
            
            bars3 = ax3.bar(model_names, success_rates, alpha=0.7, color='green')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Model Comparison: Fitting Success Rate')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars3, success_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # Fourth plot: Example fits for best model
            ax4 = self.kinetics_fig.add_subplot(2, 2, 4)
            best_model = max(model_comparison.keys(), 
                           key=lambda x: model_comparison[x]['avg_r_squared'])
            
            # Plot experimental data points and example fits
            if isinstance(population_data, dict):
                for i, (label, pop_values) in enumerate(population_data.items()):
                    color = f'C{i}'
                    ax4.scatter(time_points, pop_values, alpha=0.7, color=color, 
                              label=f'Cluster {label}', s=50)
                    
                    # Try to plot fitted line for best model
                    if label in model_comparison[best_model]['cluster_fits']:
                        fit_info = model_comparison[best_model]['cluster_fits'][label]
                        if 'error' not in fit_info:
                            # Recreate the fitted line
                            models = {
                                'Pseudo-first-order': lambda t, k, A0, C: A0 * np.exp(-k * t) + C,
                                'Pseudo-second-order': lambda t, k, A0, C: A0 / (1 + k * A0 * t) + C,
                                'Avrami Equation': lambda t, k, n, A0, C: A0 * np.exp(-np.power(k * t, n)) + C,
                                'Diffusion-controlled': lambda t, k, A0, C: A0 * (1 - np.sqrt(np.maximum(k * t, 0))) + C
                            }
                            
                            if best_model in models:
                                model_func = models[best_model]
                                params = fit_info['parameters']
                                fitted_y = model_func(time_points, *params)
                                ax4.plot(time_points, fitted_y, '--', color=color, linewidth=2)
            
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Population')
            ax4.set_title(f'Best Model Fits: {best_model}')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            self.kinetics_fig.tight_layout()
            self.kinetics_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting model comparison: {str(e)}")
    
    def display_model_comparison(self, model_comparison, best_model):
        """Display model comparison results in text form."""
        try:
            results_text = "Kinetic Model Comparison Results\n"
            results_text += "=" * 50 + "\n\n"
            
            # Sort models by R-squared (descending)
            sorted_models = sorted(model_comparison.items(), 
                                 key=lambda x: x[1]['avg_r_squared'], reverse=True)
            
            results_text += "Model Rankings (by Average R-squared):\n"
            results_text += "-" * 40 + "\n"
            
            for i, (model_name, stats) in enumerate(sorted_models, 1):
                results_text += f"{i}. {model_name}\n"
                results_text += f"   R-squared: {stats['avg_r_squared']:.4f}\n"
                results_text += f"   AIC: {stats['avg_aic']:.2f}\n"
                results_text += f"   BIC: {stats['avg_bic']:.2f}\n"
                results_text += f"   Success rate: {stats['successful_fits']}/{stats['total_clusters']}\n"
                
                if model_name == best_model:
                    results_text += "   *** BEST MODEL ***\n"
                
                results_text += "\n"
            
            results_text += f"\nRecommended Model: {best_model}\n"
            results_text += f"Average R-squared: {model_comparison[best_model]['avg_r_squared']:.4f}\n"
            
            self.kinetics_results.setText(results_text)
            
        except Exception as e:
            print(f"Error displaying model comparison: {str(e)}")

    def analyze_structural_changes(self):
        """Analyze structural changes in defined spectral regions."""
        try:
            if (self.cluster_data['intensities'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
        
            # Get spectral regions
            regions = self.get_spectral_regions()
            if not regions:
                QMessageBox.warning(self, "No Regions", "Please define spectral regions first.")
                return
            
            # Calculate mean spectra for each cluster
            wavenumbers = self.cluster_data['wavenumbers']
            intensities = self.cluster_data['intensities']
            labels = self.cluster_data['labels']
            
            unique_labels = np.unique(labels)
            cluster_means = {}
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_spectra = intensities[cluster_mask]
                mean_spectrum = np.mean(cluster_spectra, axis=0)
                cluster_means[label] = mean_spectrum
            
            # Analyze each region
            results_text = "Structural Analysis Results:\n" + "="*40 + "\n\n"
        
            for region_name, (min_wn, max_wn) in regions.items():
                # Find wavenumber indices for this region
                region_mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
                
                if not np.any(region_mask):
                    continue
                
                region_wn = wavenumbers[region_mask]
                
                results_text += f"Region: {region_name} ({min_wn}-{max_wn} cm⁻¹)\n"
                results_text += "-" * 50 + "\n"
                
                # Calculate peak positions and intensities for each cluster
                for label in unique_labels:
                    region_spectrum = cluster_means[label][region_mask]
                    
                    # Find peak position
                    peak_idx = np.argmax(region_spectrum)
                    peak_position = region_wn[peak_idx]
                    peak_intensity = region_spectrum[peak_idx]
                    
                    # Calculate integrated intensity
                    integrated_intensity = np.trapz(region_spectrum, region_wn)
                    
                    results_text += f"  Cluster {label}:\n"
                    results_text += f"    Peak position: {peak_position:.1f} cm⁻¹\n"
                    results_text += f"    Peak intensity: {peak_intensity:.3f}\n"
                    results_text += f"    Integrated intensity: {integrated_intensity:.3f}\n"
                
                results_text += "\n"
            
            self.structural_results.setText(results_text)
            
            # Create visualization
            self.plot_structural_analysis(cluster_means, regions)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Structural analysis failed:\n{str(e)}")

    def plot_structural_analysis(self, cluster_means, regions):
        """Plot structural analysis results."""
        try:
            self.structural_fig.clear()
        
            wavenumbers = self.cluster_data['wavenumbers']
            unique_labels = list(cluster_means.keys())
            
            # Create subplots for each region
            n_regions = len(regions)
            if n_regions == 0:
                return
        
            n_cols = min(2, n_regions)
            n_rows = (n_regions + n_cols - 1) // n_cols
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, (region_name, (min_wn, max_wn)) in enumerate(regions.items()):
                ax = self.structural_fig.add_subplot(n_rows, n_cols, i + 1)
                
                # Find wavenumber indices for this region
                region_mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
                
                if not np.any(region_mask):
                    continue
                
                region_wn = wavenumbers[region_mask]
                
                # Plot cluster mean spectra in this region
                for j, label in enumerate(unique_labels):
                    region_spectrum = cluster_means[label][region_mask]
                    ax.plot(region_wn, region_spectrum, 
                           color=colors[j], label=f'Cluster {label}', linewidth=2)
                
                ax.set_xlabel('Wavenumber (cm⁻¹)')
                ax.set_ylabel('Intensity')
                ax.set_title(f'{region_name}\n({min_wn}-{max_wn} cm⁻¹)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            self.structural_fig.tight_layout()
            self.structural_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting structural analysis: {str(e)}")

    def calculate_differential_spectra(self):
        """Calculate differential spectra between clusters."""
        try:
            if (self.cluster_data['intensities'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            wavenumbers = self.cluster_data['wavenumbers']
            intensities = self.cluster_data['intensities']
            labels = self.cluster_data['labels']
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                QMessageBox.warning(self, "Insufficient Clusters", "Need at least 2 clusters for differential analysis.")
                return
            
            # Calculate mean spectra for each cluster
            cluster_means = {}
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_spectra = intensities[cluster_mask]
                cluster_means[label] = np.mean(cluster_spectra, axis=0)
            
            # Create differential spectra visualization
            self.structural_fig.clear()
        
            n_comparisons = len(unique_labels) * (len(unique_labels) - 1) // 2
            n_cols = 2
            n_rows = (n_comparisons + n_cols - 1) // n_cols
            
            plot_idx = 1
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    if i < j:
                        ax = self.structural_fig.add_subplot(n_rows, n_cols, plot_idx)
                        
                        # Calculate differential spectrum
                        diff_spectrum = cluster_means[label1] - cluster_means[label2]
                        
                        ax.plot(wavenumbers, diff_spectrum, 'b-', linewidth=1)
                        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                        ax.set_xlabel('Wavenumber (cm⁻¹)')
                        ax.set_ylabel('Intensity Difference')
                        ax.set_title(f'Cluster {label1} - Cluster {label2}')
                        ax.grid(True, alpha=0.3)
                        
                        plot_idx += 1
            
            self.structural_fig.tight_layout()
            self.structural_canvas.draw()

            QMessageBox.information(self, "Differential Analysis", "Differential spectra calculated and plotted.")
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate differential spectra:\n{str(e)}")

    def update_biofilm_controls(self):
        """Update visibility of biofilm analysis controls."""
        is_enabled = self.enable_biofilm_analysis_cb.isChecked()
        for control in self.biofilm_controls:
            control.setVisible(is_enabled)

    def analyze_com_bacterial_ratios(self):
        """Analyze COM vs bacterial peak intensity ratios."""
        try:
            if (self.cluster_data['intensities'] is None or 
                self.cluster_data['wavenumbers'] is None):
                QMessageBox.warning(self, "No Data", "No spectral data available.")
                return

            wavenumbers = self.cluster_data['wavenumbers']
            intensities = self.cluster_data['intensities']
            
            # Get peak positions from UI
            com_peak1 = self.com_peak1_spinbox.value()
            com_peak2 = self.com_peak2_spinbox.value()
            protein_peak = self.protein_peak_spinbox.value()
            lipid_peak = self.lipid_peak_spinbox.value()
            window = self.peak_window_spinbox.value()
            
            # Calculate ratios for each spectrum
            com_ratios = []
            bacterial_ratios = []
            com_scores = []
            bacterial_scores = []
            
            for spectrum in intensities:
                # Calculate COM ratio (peak1/peak2)
                com1_intensity = self._get_peak_intensity(spectrum, wavenumbers, com_peak1, window)
                com2_intensity = self._get_peak_intensity(spectrum, wavenumbers, com_peak2, window)
                com_ratio = com1_intensity / (com2_intensity + 1e-6)
                com_ratios.append(com_ratio)
                
                # Calculate bacterial ratio (protein/lipid)
                protein_intensity = self._get_peak_intensity(spectrum, wavenumbers, protein_peak, window)
                lipid_intensity = self._get_peak_intensity(spectrum, wavenumbers, lipid_peak, window)
                bacterial_ratio = protein_intensity / (lipid_intensity + 1e-6)
                bacterial_ratios.append(bacterial_ratio)
                
                # Calculate COM score (combined COM peak intensity)
                com_score = (com1_intensity + com2_intensity) / 2.0
                com_scores.append(com_score)
                
                # Calculate bacterial score (combined bacterial peak intensity)
                bacterial_score = (protein_intensity + lipid_intensity) / 2.0
                bacterial_scores.append(bacterial_score)
            
            # Store results for correlation analysis
            self.cluster_data['com_ratios'] = np.array(com_ratios)
            self.cluster_data['bacterial_ratios'] = np.array(bacterial_ratios)
            self.cluster_data['com_scores'] = np.array(com_scores)
            self.cluster_data['bacterial_scores'] = np.array(bacterial_scores)
            
            # Create visualization
            self._plot_biofilm_analysis(com_ratios, bacterial_ratios, com_scores, bacterial_scores)
            
            # Display results
            results_text = "COM vs Bacterial Analysis Results:\n" + "="*50 + "\n\n"
            results_text += f"Analyzed {len(intensities)} spectra\n\n"
            results_text += f"COM Ratio (I₁₃₂₀/I₁₆₂₀):\n"
            results_text += f"  Mean: {np.mean(com_ratios):.3f} ± {np.std(com_ratios):.3f}\n"
            results_text += f"  Range: {np.min(com_ratios):.3f} - {np.max(com_ratios):.3f}\n\n"
            results_text += f"Bacterial Ratio (Protein/Lipid):\n"
            results_text += f"  Mean: {np.mean(bacterial_ratios):.3f} ± {np.std(bacterial_ratios):.3f}\n"
            results_text += f"  Range: {np.min(bacterial_ratios):.3f} - {np.max(bacterial_ratios):.3f}\n\n"
            
            # Classification thresholds
            high_com_threshold = np.percentile(com_scores, 75)
            high_bacterial_threshold = np.percentile(bacterial_scores, 75)
            
            results_text += f"Classification Thresholds:\n"
            results_text += f"  High COM Score: >{high_com_threshold:.3f}\n"
            results_text += f"  High Bacterial Score: >{high_bacterial_threshold:.3f}\n"
            
            self.structural_results.setText(results_text)
            
            QMessageBox.information(self, "Analysis Complete", 
                                  "COM vs bacterial analysis completed. Use 'Correlate with Clusters' to see cluster relationships.")
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"COM vs bacterial analysis failed:\n{str(e)}")

    def correlate_ratios_with_clusters(self):
        """Correlate peak ratios with cluster assignments."""
        try:
            if (self.cluster_data['labels'] is None or 
                'com_ratios' not in self.cluster_data):
                QMessageBox.warning(self, "No Data", "Run COM vs bacterial analysis first.")
                return

            labels = self.cluster_data['labels']
            com_ratios = self.cluster_data['com_ratios']
            bacterial_ratios = self.cluster_data['bacterial_ratios']
            com_scores = self.cluster_data['com_scores']
            bacterial_scores = self.cluster_data['bacterial_scores']
            
            unique_labels = np.unique(labels)
            
            # Calculate cluster statistics
            cluster_stats = {}
            for label in unique_labels:
                mask = labels == label
                cluster_stats[label] = {
                    'count': np.sum(mask),
                    'com_ratio_mean': np.mean(com_ratios[mask]),
                    'com_ratio_std': np.std(com_ratios[mask]),
                    'bacterial_ratio_mean': np.mean(bacterial_ratios[mask]),
                    'bacterial_ratio_std': np.std(bacterial_ratios[mask]),
                    'com_score_mean': np.mean(com_scores[mask]),
                    'bacterial_score_mean': np.mean(bacterial_scores[mask]),
                }
            
            # Classify clusters
            self._classify_clusters_biofilm(cluster_stats)
            
            # Create correlation visualization
            self._plot_cluster_correlations(labels, com_ratios, bacterial_ratios, com_scores, bacterial_scores, cluster_stats)
            
            # Display correlation results
            results_text = "Cluster-Ratio Correlation Results:\n" + "="*60 + "\n\n"
            
            for label in unique_labels:
                stats = cluster_stats[label]
                results_text += f"Cluster {label} (n={stats['count']}):\n"
                results_text += f"  COM Ratio: {stats['com_ratio_mean']:.3f} ± {stats['com_ratio_std']:.3f}\n"
                results_text += f"  Bacterial Ratio: {stats['bacterial_ratio_mean']:.3f} ± {stats['bacterial_ratio_std']:.3f}\n"
                results_text += f"  COM Score: {stats['com_score_mean']:.3f}\n"
                results_text += f"  Bacterial Score: {stats['bacterial_score_mean']:.3f}\n"
                
                # Add classification
                if 'classification' in stats:
                    results_text += f"  Classification: {stats['classification']}\n"
                
                results_text += "\n"
            
            # Statistical tests
            from scipy.stats import f_oneway, kruskal
            try:
                # ANOVA for COM ratios
                com_groups = [com_ratios[labels == label] for label in unique_labels]
                f_stat_com, p_val_com = f_oneway(*com_groups)
                
                # ANOVA for bacterial ratios
                bacterial_groups = [bacterial_ratios[labels == label] for label in unique_labels]
                f_stat_bacterial, p_val_bacterial = f_oneway(*bacterial_groups)
                
                results_text += "Statistical Significance:\n"
                results_text += f"  COM Ratio ANOVA: F={f_stat_com:.3f}, p={p_val_com:.4f}\n"
                results_text += f"  Bacterial Ratio ANOVA: F={f_stat_bacterial:.3f}, p={p_val_bacterial:.4f}\n"
                
                if p_val_com < 0.05:
                    results_text += "  ✓ COM ratios show significant cluster differences\n"
                if p_val_bacterial < 0.05:
                    results_text += "  ✓ Bacterial ratios show significant cluster differences\n"
                    
            except Exception as e:
                results_text += f"Statistical analysis error: {str(e)}\n"
            
            self.structural_results.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Correlation Error", f"Cluster correlation analysis failed:\n{str(e)}")

    def _get_peak_intensity(self, spectrum, wavenumbers, peak_center, window):
        """Get peak intensity within specified window."""
        mask = (wavenumbers >= peak_center - window) & (wavenumbers <= peak_center + window)
        if not np.any(mask):
            return 0.0
        return np.max(spectrum[mask])

    def _plot_biofilm_analysis(self, com_ratios, bacterial_ratios, com_scores, bacterial_scores):
        """Plot biofilm analysis results."""
        self.structural_fig.clear()
        
        # Create 2x2 subplot layout
        gs = self.structural_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. COM vs Bacterial Ratios scatter plot
        ax1 = self.structural_fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(com_ratios, bacterial_ratios, alpha=0.6, c='blue', s=50)
        ax1.set_xlabel('COM Ratio (I₁₃₂₀/I₁₆₂₀)')
        ax1.set_ylabel('Bacterial Ratio (Protein/Lipid)')
        ax1.set_title('COM vs Bacterial Signatures')
        ax1.grid(True, alpha=0.3)
        
        # 2. COM vs Bacterial Scores
        ax2 = self.structural_fig.add_subplot(gs[0, 1])
        ax2.scatter(com_scores, bacterial_scores, alpha=0.6, c='green', s=50)
        ax2.set_xlabel('COM Score (Average Intensity)')
        ax2.set_ylabel('Bacterial Score (Average Intensity)')
        ax2.set_title('COM vs Bacterial Intensity Scores')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram of COM ratios
        ax3 = self.structural_fig.add_subplot(gs[1, 0])
        ax3.hist(com_ratios, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax3.set_xlabel('COM Ratio (I₁₃₂₀/I₁₆₂₀)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of COM Ratios')
        ax3.grid(True, alpha=0.3)
        
        # 4. Histogram of bacterial ratios
        ax4 = self.structural_fig.add_subplot(gs[1, 1])
        ax4.hist(bacterial_ratios, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Bacterial Ratio (Protein/Lipid)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Bacterial Ratios')
        ax4.grid(True, alpha=0.3)
        
        self.structural_canvas.draw()

    def _plot_cluster_correlations(self, labels, com_ratios, bacterial_ratios, com_scores, bacterial_scores, cluster_stats):
        """Plot cluster correlations with biofilm metrics."""
        self.structural_fig.clear()
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        # Create 2x2 subplot layout
        gs = self.structural_fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Colored scatter plot by cluster
        ax1 = self.structural_fig.add_subplot(gs[0, 0])
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(com_ratios[mask], bacterial_ratios[mask], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7, s=50)
        ax1.set_xlabel('COM Ratio (I₁₃₂₀/I₁₆₂₀)')
        ax1.set_ylabel('Bacterial Ratio (Protein/Lipid)')
        ax1.set_title('Clusters in COM vs Bacterial Space')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plots of COM ratios by cluster
        ax2 = self.structural_fig.add_subplot(gs[0, 1])
        com_data = [com_ratios[labels == label] for label in unique_labels]
        bp1 = ax2.boxplot(com_data, labels=[f'C{label}' for label in unique_labels], patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('COM Ratio')
        ax2.set_title('COM Ratios by Cluster')
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plots of bacterial ratios by cluster
        ax3 = self.structural_fig.add_subplot(gs[1, 0])
        bacterial_data = [bacterial_ratios[labels == label] for label in unique_labels]
        bp2 = ax3.boxplot(bacterial_data, labels=[f'C{label}' for label in unique_labels], patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Bacterial Ratio')
        ax3.set_title('Bacterial Ratios by Cluster')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cluster means comparison
        ax4 = self.structural_fig.add_subplot(gs[1, 1])
        cluster_labels = [f'Cluster {label}' for label in unique_labels]
        com_means = [cluster_stats[label]['com_ratio_mean'] for label in unique_labels]
        bacterial_means = [cluster_stats[label]['bacterial_ratio_mean'] for label in unique_labels]
        
        x = np.arange(len(cluster_labels))
        width = 0.35
        
        ax4.bar(x - width/2, com_means, width, label='COM Ratio', alpha=0.7)
        ax4.bar(x + width/2, bacterial_means, width, label='Bacterial Ratio', alpha=0.7)
        
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Mean Ratio')
        ax4.set_title('Mean Ratios by Cluster')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cluster_labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.structural_canvas.draw()

    def _classify_clusters_biofilm(self, cluster_stats):
        """Classify clusters based on COM vs bacterial signatures."""
        # Calculate global thresholds
        all_com_scores = [stats['com_score_mean'] for stats in cluster_stats.values()]
        all_bacterial_scores = [stats['bacterial_score_mean'] for stats in cluster_stats.values()]
        
        com_threshold = np.median(all_com_scores)
        bacterial_threshold = np.median(all_bacterial_scores)
        
        for label, stats in cluster_stats.items():
            com_high = stats['com_score_mean'] > com_threshold
            bacterial_high = stats['bacterial_score_mean'] > bacterial_threshold
            
            if com_high and not bacterial_high:
                classification = "COM-dominant"
            elif bacterial_high and not com_high:
                classification = "Bacterial-dominant"
            elif com_high and bacterial_high:
                classification = "Mixed (COM + Bacterial)"
            else:
                classification = "Low signature/Background"
            
            stats['classification'] = classification

    # Validation Methods
    def calculate_silhouette_analysis(self):
        """Calculate silhouette analysis for cluster validation."""
        try:
            if (self.cluster_data['features_scaled'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            features = self.cluster_data['features_scaled']
            labels = self.cluster_data['labels']
            
            # Calculate silhouette scores
            silhouette_avg = silhouette_score(features, labels)
            sample_silhouette_values = silhouette_samples(features, labels)
            
            # Store results
            self.cluster_data['silhouette_scores'] = {
                'average': silhouette_avg,
                'samples': sample_silhouette_values,
                'threshold': self.min_silhouette_threshold.value()
            }
            
            # Analyze results
            unique_labels = np.unique(labels)
            cluster_silhouettes = {}
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_silhouettes[label] = {
                    'mean': np.mean(sample_silhouette_values[cluster_mask]),
                    'std': np.std(sample_silhouette_values[cluster_mask]),
                    'min': np.min(sample_silhouette_values[cluster_mask]),
                    'max': np.max(sample_silhouette_values[cluster_mask]),
                    'samples': sample_silhouette_values[cluster_mask]
                }
            
            # Create silhouette plot
            self.plot_silhouette_analysis(unique_labels, cluster_silhouettes, silhouette_avg)
            
            # Generate results text
            results_text = self.generate_silhouette_results(
                silhouette_avg, cluster_silhouettes, sample_silhouette_values, labels
            )
            
            self.validation_results.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Silhouette Analysis Error", 
                               f"Failed to calculate silhouette analysis:\n{str(e)}")

    def analyze_cluster_transitions(self):
        """Analyze cluster transition boundaries."""
        try:
            if (self.cluster_data['features_scaled'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            features = self.cluster_data['features_scaled']
            labels = self.cluster_data['labels']
            
            # Calculate inter-cluster boundaries using PCA for visualization
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(features)
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Calculate cluster centroids in 2D space
            centroids_2d = []
            for label in unique_labels:
                cluster_mask = labels == label
                centroid = np.mean(coords_2d[cluster_mask], axis=0)
                centroids_2d.append(centroid)
            centroids_2d = np.array(centroids_2d)
            
            # Find boundary/transition points
            transition_analysis = {}
            
            for i, label_i in enumerate(unique_labels):
                for j, label_j in enumerate(unique_labels):
                    if i < j:
                        # Find points near the boundary between clusters
                        mask_i = labels == label_i
                        mask_j = labels == label_j
                        
                        coords_i = coords_2d[mask_i]
                        coords_j = coords_2d[mask_j]
                        
                        # Calculate distances from each cluster's points to the other cluster's centroid
                        centroid_i = centroids_2d[i]
                        centroid_j = centroids_2d[j]
                        
                        # Find boundary points (points that are close to the other cluster)
                        distances_i_to_j = np.linalg.norm(coords_i - centroid_j, axis=1)
                        distances_j_to_i = np.linalg.norm(coords_j - centroid_i, axis=1)
                        
                        # Define boundary threshold as percentile of distances
                        boundary_threshold = np.percentile(
                            np.concatenate([distances_i_to_j, distances_j_to_i]), 25
                        )
                        
                        boundary_points_i = coords_i[distances_i_to_j <= boundary_threshold]
                        boundary_points_j = coords_j[distances_j_to_i <= boundary_threshold]
                        
                        transition_analysis[f"{label_i}-{label_j}"] = {
                            'centroid_distance': np.linalg.norm(centroid_i - centroid_j),
                            'boundary_thickness': boundary_threshold,
                            'boundary_points_i': boundary_points_i,
                            'boundary_points_j': boundary_points_j,
                            'n_boundary_i': len(boundary_points_i),
                            'n_boundary_j': len(boundary_points_j)
                        }
            
            # Store results
            self.cluster_data['transition_analysis'] = transition_analysis
            
            # Create transition visualization
            self.plot_transition_analysis(coords_2d, labels, centroids_2d, transition_analysis, unique_labels)
            
            # Generate results text
            results_text = self.generate_transition_results(transition_analysis, unique_labels)
            
            self.validation_results.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Transition Analysis Error", 
                               f"Failed to analyze cluster transitions:\n{str(e)}")

    def test_cluster_stability(self):
        """Test cluster stability through bootstrap analysis."""
        try:
            if (self.cluster_data['features_scaled'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            features = self.cluster_data['features_scaled']
            original_labels = self.cluster_data['labels']
            n_iterations = self.bootstrap_iterations.value()
            
            # Get original clustering parameters
            n_clusters = len(np.unique(original_labels))
            linkage_method = self.linkage_method_combo.currentText() if hasattr(self, 'linkage_method_combo') else 'ward'
            distance_metric = self.distance_metric_combo.currentText() if hasattr(self, 'distance_metric_combo') else 'euclidean'
            
            # Bootstrap stability analysis
            stability_scores = []
            cluster_consistency = {label: [] for label in np.unique(original_labels)}
            
            for iteration in range(n_iterations):
                # Bootstrap sample
                n_samples = len(features)
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_features = features[bootstrap_indices]
                
                # Perform clustering on bootstrap sample
                try:
                    bootstrap_labels, _, _ = self.perform_hierarchical_clustering(
                        bootstrap_features, n_clusters, linkage_method, distance_metric
                    )
                    
                    # Map bootstrap labels back to original indices
                    original_bootstrap_labels = np.full(n_samples, -1)
                    for i, bootstrap_idx in enumerate(bootstrap_indices):
                        original_bootstrap_labels[bootstrap_idx] = bootstrap_labels[i]
                    
                    # Calculate stability using Adjusted Rand Index
                    # Only consider samples that were selected in bootstrap
                    valid_mask = original_bootstrap_labels != -1
                    if np.sum(valid_mask) > 1:
                        from sklearn.metrics import adjusted_rand_score
                        ari = adjusted_rand_score(
                            original_labels[valid_mask], 
                            original_bootstrap_labels[valid_mask]
                        )
                        stability_scores.append(ari)
                        
                        # Track cluster consistency
                        for original_label in np.unique(original_labels):
                            original_cluster_mask = (original_labels == original_label) & valid_mask
                            if np.sum(original_cluster_mask) > 0:
                                bootstrap_cluster_labels = original_bootstrap_labels[original_cluster_mask]
                                # Calculate most common label in bootstrap
                                unique_bootstrap_labels, counts = np.unique(bootstrap_cluster_labels, return_counts=True)
                                if len(unique_bootstrap_labels) > 0:
                                    consistency = np.max(counts) / len(bootstrap_cluster_labels)
                                    cluster_consistency[original_label].append(consistency)
                
                except Exception as e:
                    print(f"Bootstrap iteration {iteration} failed: {e}")
                    continue
            
            # Calculate stability statistics
            stability_stats = {
                'mean_ari': np.mean(stability_scores) if stability_scores else 0.0,
                'std_ari': np.std(stability_scores) if stability_scores else 0.0,
                'min_ari': np.min(stability_scores) if stability_scores else 0.0,
                'max_ari': np.max(stability_scores) if stability_scores else 0.0,
                'successful_iterations': len(stability_scores),
                'total_iterations': n_iterations
            }
            
            # Calculate cluster-specific stability
            cluster_stability = {}
            for label, consistencies in cluster_consistency.items():
                if consistencies:
                    cluster_stability[label] = {
                        'mean_consistency': np.mean(consistencies),
                        'std_consistency': np.std(consistencies),
                        'min_consistency': np.min(consistencies),
                        'max_consistency': np.max(consistencies)
                    }
                else:
                    cluster_stability[label] = {
                        'mean_consistency': 0.0,
                        'std_consistency': 0.0,
                        'min_consistency': 0.0,
                        'max_consistency': 0.0
                    }
            
            # Store results
            self.cluster_data['stability_analysis'] = {
                'stability_stats': stability_stats,
                'cluster_stability': cluster_stability,
                'stability_scores': stability_scores
            }
            
            # Create stability visualization
            self.plot_stability_analysis(stability_stats, cluster_stability, stability_scores)
            
            # Generate results text
            results_text = self.generate_stability_results(stability_stats, cluster_stability)
            
            self.validation_results.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Stability Analysis Error", 
                               f"Failed to test cluster stability:\n{str(e)}")

    def run_complete_validation(self):
        """Run complete validation suite."""
        try:
            if (self.cluster_data['features_scaled'] is None or 
                self.cluster_data['labels'] is None):
                QMessageBox.warning(self, "No Data", "No clustering data available.")
                return
            
            # Run all validation analyses
            validation_results = {}
            
            # 1. Silhouette Analysis
            if self.silhouette_analysis_cb.isChecked():
                try:
                    self.calculate_silhouette_analysis()
                    validation_results['silhouette'] = self.cluster_data.get('silhouette_scores', {})
                except Exception as e:
                    validation_results['silhouette'] = {'error': str(e)}
            
            # 2. Transition Analysis  
            if self.transition_analysis_cb.isChecked():
                try:
                    self.analyze_cluster_transitions()
                    validation_results['transitions'] = self.cluster_data.get('transition_analysis', {})
                except Exception as e:
                    validation_results['transitions'] = {'error': str(e)}
            
            # 3. Stability Analysis
            if self.stability_analysis_cb.isChecked():
                try:
                    self.test_cluster_stability()
                    validation_results['stability'] = self.cluster_data.get('stability_analysis', {})
                except Exception as e:
                    validation_results['stability'] = {'error': str(e)}
            
            # Create comprehensive validation visualization
            self.plot_comprehensive_validation(validation_results)
            
            # Generate comprehensive results text
            results_text = self.generate_comprehensive_validation_results(validation_results)
            
            self.validation_results.setText(results_text)
            
            # Store complete validation results
            self.cluster_data['complete_validation'] = validation_results
            
        except Exception as e:
            QMessageBox.critical(self, "Complete Validation Error", 
                               f"Failed to run complete validation:\n{str(e)}")

    # Advanced Statistics Methods
    def calculate_feature_importance(self):
        """Calculate feature importance for cluster discrimination."""
        try:
            if self.cluster_data['labels'] is None:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            # Get the method
            method = self.feature_method_combo.currentText()
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            wavenumbers = self.cluster_data['wavenumbers']
            
            # Calculate feature importance
            if method == 'Random Forest':
                importance_scores = self._calculate_rf_importance(intensities, labels)
            elif method == 'Linear Discriminant Analysis':
                importance_scores = self._calculate_lda_importance(intensities, labels)
            elif method == 'Mutual Information':
                importance_scores = self._calculate_mutual_info_importance(intensities, labels)
            else:
                importance_scores = self._calculate_rf_importance(intensities, labels)
            
            # Find top features
            top_indices = np.argsort(importance_scores)[-20:][::-1]  # Top 20 features
            top_wavenumbers = wavenumbers[top_indices]
            top_scores = importance_scores[top_indices]
            
            # Plot results
            self._plot_feature_importance(wavenumbers, importance_scores, 
                                        top_wavenumbers, top_scores, method)
            
            # Display results
            self._display_feature_importance_results(top_wavenumbers, top_scores, method)
            
        except Exception as e:
            QMessageBox.critical(self, "Feature Importance Error", 
                               f"Failed to calculate feature importance:\n{str(e)}")

    def _calculate_rf_importance(self, intensities, labels):
        """Calculate feature importance using Random Forest."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(intensities)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, labels)
        
        return rf.feature_importances_
    
    def _calculate_lda_importance(self, intensities, labels):
        """Calculate feature importance using Linear Discriminant Analysis."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(intensities)
        
        # Train LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_scaled, labels)
        
        # Calculate importance as absolute value of LDA coefficients
        if hasattr(lda, 'coef_') and lda.coef_ is not None:
            if lda.coef_.ndim > 1:
                importance = np.abs(lda.coef_[0])  # First component
            else:
                importance = np.abs(lda.coef_)
        else:
            # Fallback to between-class to within-class scatter ratio
            n_features = intensities.shape[1]
            importance = np.zeros(n_features)
            
            for i in range(n_features):
                feature_data = intensities[:, i]
                between_variance = np.var([np.mean(feature_data[labels == label]) 
                                         for label in np.unique(labels)])
                within_variance = np.mean([np.var(feature_data[labels == label]) 
                                         for label in np.unique(labels)])
                importance[i] = between_variance / (within_variance + 1e-8)
        
        return importance
    
    def _calculate_mutual_info_importance(self, intensities, labels):
        """Calculate feature importance using Mutual Information."""
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(intensities)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_scaled, labels, random_state=42)
        
        return mi_scores
    
    def _plot_feature_importance(self, wavenumbers, importance_scores, 
                                top_wavenumbers, top_scores, method):
        """Plot feature importance results."""
        self.stats_fig.clear()
        
        # Create subplots
        ax1 = self.stats_fig.add_subplot(2, 2, 1)  # Full spectrum importance
        ax2 = self.stats_fig.add_subplot(2, 2, 2)  # Top features bar plot
        ax3 = self.stats_fig.add_subplot(2, 2, 3)  # Cumulative importance
        ax4 = self.stats_fig.add_subplot(2, 2, 4)  # Importance distribution
        
        # 1. Full spectrum importance plot
        ax1.plot(wavenumbers, importance_scores, 'b-', alpha=0.7)
        ax1.scatter(top_wavenumbers, top_scores, color='red', s=30, zorder=5)
        ax1.set_xlabel('Wavenumber (cm⁻¹)')
        ax1.set_ylabel('Feature Importance')
        ax1.set_title(f'Feature Importance - {method}')
        ax1.grid(True, alpha=0.3)
        
        # 2. Top features bar plot
        x_pos = np.arange(len(top_wavenumbers))
        bars = ax2.bar(x_pos, top_scores, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Top Features (Wavenumber)')
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Top 20 Most Important Features')
        ax2.set_xticks(x_pos[::2])  # Show every other tick
        ax2.set_xticklabels([f'{int(wn)}' for wn in top_wavenumbers[::2]], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative importance
        sorted_importance = np.sort(importance_scores)[::-1]
        cumulative_importance = np.cumsum(sorted_importance)
        cumulative_importance /= cumulative_importance[-1]  # Normalize to 1
        
        ax3.plot(range(len(cumulative_importance)), cumulative_importance, 'g-', linewidth=2)
        ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Cumulative Importance')
        ax3.set_title('Cumulative Feature Importance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Importance distribution
        ax4.hist(importance_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=np.mean(importance_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(importance_scores):.4f}')
        ax4.axvline(x=np.median(importance_scores), color='orange', linestyle='--', 
                   label=f'Median: {np.median(importance_scores):.4f}')
        ax4.set_xlabel('Importance Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Feature Importance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.stats_fig.tight_layout()
        self.stats_canvas.draw()
    
    def _display_feature_importance_results(self, top_wavenumbers, top_scores, method):
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
        
        # Check for known spectral regions
        known_regions = {
            'Protein (Amide I)': (1600, 1700),
            'Lipids (CH₂)': (1400, 1500),
            'COM (C-O stretch)': (1300, 1350),
            'COM (C=O stretch)': (1580, 1650),
            'Nucleic acids': (750, 850),
            'Carbohydrates': (900, 1200),
        }
        
        results_text += f"\nSpectral Region Analysis:\n"
        for region_name, (min_wn, max_wn) in known_regions.items():
            region_mask = (top_wavenumbers >= min_wn) & (top_wavenumbers <= max_wn)
            if np.any(region_mask):
                region_importance = np.sum(top_scores[region_mask])
                region_count = np.sum(region_mask)
                results_text += f"• {region_name}: {region_count} features, total importance: {region_importance:.6f}\n"
        
        results_text += f"\nInterpretation:\n"
        results_text += f"The {method} method identified {len(top_wavenumbers)} key spectral features "
        results_text += f"that best distinguish between clusters. Higher importance scores indicate "
        results_text += f"wavenumbers that contribute more to cluster separation.\n"
        
        self.stats_results.setText(results_text)
    
    def _display_significance_results(self, results, method, alpha):
        """Display statistical significance results."""
        results_text = f"Statistical Significance Testing - {method}\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Test Parameters:\n"
        results_text += f"• Significance level (α): {alpha:.3f}\n"
        
        if method in ['PERMANOVA', 'ANOSIM']:
            results_text += f"• Permutations: {len(results['permuted_stats'])}\n\n"
            
            results_text += f"Test Results:\n"
            results_text += f"• {method} statistic: {results['test_statistic']:.4f}\n"
            results_text += f"• p-value: {results['p_value']:.4f}\n"
            results_text += f"• Significant: {'Yes' if results['significant'] else 'No'}\n\n"
            
            if method == 'PERMANOVA':
                results_text += f"• Degrees of freedom (between): {results['df_between']}\n"
                results_text += f"• Degrees of freedom (within): {results['df_within']}\n"
            
            # Interpretation
            if results['significant']:
                results_text += f"\nInterpretation:\n"
                results_text += f"The clusters show statistically significant differences "
                results_text += f"(p < {alpha:.3f}). The observed clustering is unlikely "
                results_text += f"to have occurred by chance.\n"
            else:
                results_text += f"\nInterpretation:\n"
                results_text += f"The clusters do not show statistically significant differences "
                results_text += f"(p ≥ {alpha:.3f}). The observed clustering may have "
                results_text += f"occurred by chance.\n"
        
        else:
            # Feature-wise methods
            results_text += f"\nTest Results:\n"
            results_text += f"• Average test statistic: {results['test_statistic']:.4f}\n"
            results_text += f"• Minimum p-value: {results['p_value']:.4f}\n"
            results_text += f"• Significant features: {results['significant_features']}/{results['total_features']}\n"
            results_text += f"• Percentage significant: {results['significant_features']/results['total_features']*100:.1f}%\n"
            results_text += f"• Overall significant: {'Yes' if results['significant'] else 'No'}\n\n"
            
            # Interpretation
            if results['significant']:
                results_text += f"Interpretation:\n"
                results_text += f"Multiple spectral features show statistically significant "
                results_text += f"differences between clusters after Bonferroni correction. "
                results_text += f"The clustering reveals meaningful chemical differences.\n"
            else:
                results_text += f"Interpretation:\n"
                results_text += f"No spectral features show statistically significant "
                results_text += f"differences between clusters after multiple testing correction. "
                results_text += f"The clustering may not reflect meaningful chemical differences.\n"
        
        self.stats_results.setText(results_text)

    def perform_discriminant_analysis(self):
        """Perform discriminant analysis on clusters."""
        try:
            if self.cluster_data['labels'] is None:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            wavenumbers = self.cluster_data['wavenumbers']
            
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
            
        except Exception as e:
            QMessageBox.critical(self, "Discriminant Analysis Error", 
                               f"Failed to perform discriminant analysis:\n{str(e)}")

    def _plot_discriminant_analysis(self, X_lda, labels, explained_variance, cv_scores):
        """Plot discriminant analysis results."""
        self.stats_fig.clear()
        
        # Create subplots
        ax1 = self.stats_fig.add_subplot(2, 2, 1)  # LDA scatter plot
        ax2 = self.stats_fig.add_subplot(2, 2, 2)  # Explained variance
        ax3 = self.stats_fig.add_subplot(2, 2, 3)  # Cross-validation scores
        ax4 = self.stats_fig.add_subplot(2, 2, 4)  # Cluster separation
        
        # 1. LDA scatter plot (first two components)
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if X_lda.shape[1] >= 2:
                ax1.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                           c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
            else:
                # If only one component, plot against index
                ax1.scatter(range(np.sum(mask)), X_lda[mask, 0], 
                           c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
        
        if X_lda.shape[1] >= 2:
            ax1.set_xlabel(f'LD1 ({explained_variance[0]:.1%} variance)')
            ax1.set_ylabel(f'LD2 ({explained_variance[1]:.1%} variance)')
        else:
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel(f'LD1 ({explained_variance[0]:.1%} variance)')
        ax1.set_title('Linear Discriminant Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Explained variance plot
        components = range(1, len(explained_variance) + 1)
        ax2.bar(components, explained_variance, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Linear Discriminant')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Explained Variance by Component')
        ax2.set_xticks(components)
        ax2.grid(True, alpha=0.3)
        
        # 3. Cross-validation scores
        fold_numbers = range(1, len(cv_scores) + 1)
        bars = ax3.bar(fold_numbers, cv_scores, color='lightgreen', edgecolor='darkgreen')
        ax3.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cv_scores):.3f}')
        ax3.set_xlabel('Cross-Validation Fold')
        ax3.set_ylabel('Accuracy Score')
        ax3.set_title('Cross-Validation Performance')
        ax3.set_ylim(0, 1)
        ax3.legend()
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
            ax4.set_xticklabels([f'C{label}' for label in unique_labels])
            ax4.set_yticklabels([f'C{label}' for label in unique_labels])
            ax4.set_title('Cluster Separation Matrix')
            
            # Add text annotations
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if i != j:
                        ax4.text(j, i, f'{separation_matrix[i, j]:.2f}', 
                               ha='center', va='center', color='white')
            
            self.stats_fig.colorbar(im, ax=ax4)
        
        self.stats_fig.tight_layout()
        self.stats_canvas.draw()

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
        
        self.stats_results.setText(results_text)

    def test_statistical_significance(self):
        """Test statistical significance of cluster differences."""
        try:
            if self.cluster_data['labels'] is None:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            method = self.significance_method_combo.currentText()
            alpha = self.significance_level.value()
            n_permutations = self.permutations.value()
            
            intensities = np.array(self.cluster_data['intensities'])
            labels = self.cluster_data['labels']
            
            if method == 'PERMANOVA':
                results = self._perform_permanova(intensities, labels, alpha, n_permutations)
            elif method == 'ANOSIM':
                results = self._perform_anosim(intensities, labels, alpha, n_permutations)
            elif method == 'Kruskal-Wallis':
                results = self._perform_kruskal_wallis(intensities, labels, alpha)
            elif method == 'ANOVA':
                results = self._perform_anova(intensities, labels, alpha)
            else:
                results = self._perform_permanova(intensities, labels, alpha, n_permutations)
            
            # Plot results
            self._plot_significance_results(results, method)
            
            # Display results
            self._display_significance_results(results, method, alpha)
            
        except Exception as e:
            QMessageBox.critical(self, "Statistical Significance Error", 
                               f"Failed to test statistical significance:\n{str(e)}")

    def _perform_permanova(self, intensities, labels, alpha, n_permutations):
        """Perform PERMANOVA test."""
        from scipy.spatial.distance import pdist, squareform
        
        # Calculate distance matrix
        distances = pdist(intensities, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Calculate observed F-statistic
        def calculate_f_stat(dist_matrix, group_labels):
            unique_groups = np.unique(group_labels)
            n_total = len(group_labels)
            n_groups = len(unique_groups)
            
            # Total sum of squares
            grand_mean_distances = np.mean(dist_matrix)
            total_ss = np.sum((dist_matrix - grand_mean_distances) ** 2)
            
            # Within-group sum of squares
            within_ss = 0
            for group in unique_groups:
                group_mask = group_labels == group
                group_distances = dist_matrix[np.ix_(group_mask, group_mask)]
                group_mean = np.mean(group_distances)
                within_ss += np.sum((group_distances - group_mean) ** 2)
            
            # Between-group sum of squares
            between_ss = total_ss - within_ss
            
            # Degrees of freedom
            df_between = n_groups - 1
            df_within = n_total - n_groups
            
            # F-statistic
            if df_within > 0 and within_ss > 0:
                f_stat = (between_ss / df_between) / (within_ss / df_within)
            else:
                f_stat = 0
            
            return f_stat, df_between, df_within
        
        observed_f, df_between, df_within = calculate_f_stat(distance_matrix, labels)
        
        # Permutation test
        permuted_f_stats = []
        for _ in range(n_permutations):
            permuted_labels = np.random.permutation(labels)
            perm_f, _, _ = calculate_f_stat(distance_matrix, permuted_labels)
            permuted_f_stats.append(perm_f)
        
        permuted_f_stats = np.array(permuted_f_stats)
        p_value = np.sum(permuted_f_stats >= observed_f) / n_permutations
        
        return {
            'test_statistic': observed_f,
            'p_value': p_value,
            'df_between': df_between,
            'df_within': df_within,
            'significant': p_value < alpha,
            'permuted_stats': permuted_f_stats,
            'method': 'PERMANOVA'
        }

    def _perform_anosim(self, intensities, labels, alpha, n_permutations):
        """Perform ANOSIM test."""
        from scipy.spatial.distance import pdist, squareform
        
        # Calculate distance matrix
        distances = pdist(intensities, metric='euclidean')
        distance_matrix = squareform(distances)
        
        def calculate_anosim_r(dist_matrix, group_labels):
            unique_groups = np.unique(group_labels)
            n_samples = len(group_labels)
            
            # Calculate within-group and between-group distances
            within_distances = []
            between_distances = []
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if group_labels[i] == group_labels[j]:
                        within_distances.append(dist_matrix[i, j])
                    else:
                        between_distances.append(dist_matrix[i, j])
            
            within_distances = np.array(within_distances)
            between_distances = np.array(between_distances)
            
            # Calculate ANOSIM R statistic
            if len(within_distances) > 0 and len(between_distances) > 0:
                mean_within = np.mean(within_distances)
                mean_between = np.mean(between_distances)
                r_stat = (mean_between - mean_within) / (mean_between + mean_within) * 2
            else:
                r_stat = 0
            
            return r_stat
        
        observed_r = calculate_anosim_r(distance_matrix, labels)
        
        # Permutation test
        permuted_r_stats = []
        for _ in range(n_permutations):
            permuted_labels = np.random.permutation(labels)
            perm_r = calculate_anosim_r(distance_matrix, permuted_labels)
            permuted_r_stats.append(perm_r)
        
        permuted_r_stats = np.array(permuted_r_stats)
        p_value = np.sum(permuted_r_stats >= observed_r) / n_permutations
        
        return {
            'test_statistic': observed_r,
            'p_value': p_value,
            'significant': p_value < alpha,
            'permuted_stats': permuted_r_stats,
            'method': 'ANOSIM'
        }

    def _perform_kruskal_wallis(self, intensities, labels, alpha):
        """Perform Kruskal-Wallis test."""
        from scipy.stats import kruskal
        
        # Test each wavenumber separately
        n_features = intensities.shape[1]
        h_statistics = []
        p_values = []
        
        unique_labels = np.unique(labels)
        
        for feature_idx in range(n_features):
            feature_data = intensities[:, feature_idx]
            group_data = [feature_data[labels == label] for label in unique_labels]
            
            try:
                h_stat, p_val = kruskal(*group_data)
                h_statistics.append(h_stat)
                p_values.append(p_val)
            except:
                h_statistics.append(0)
                p_values.append(1)
        
        h_statistics = np.array(h_statistics)
        p_values = np.array(p_values)
        
        # Multiple testing correction (Bonferroni)
        corrected_p_values = p_values * n_features
        corrected_p_values = np.minimum(corrected_p_values, 1.0)
        
        # Overall test result
        significant_features = np.sum(corrected_p_values < alpha)
        overall_significant = significant_features > 0
        
        return {
            'test_statistic': np.mean(h_statistics),
            'p_value': np.min(p_values),
            'corrected_p_values': corrected_p_values,
            'significant': overall_significant,
            'significant_features': significant_features,
            'total_features': n_features,
            'method': 'Kruskal-Wallis'
        }

    def _perform_anova(self, intensities, labels, alpha):
        """Perform one-way ANOVA test."""
        from scipy.stats import f_oneway
        
        # Test each wavenumber separately
        n_features = intensities.shape[1]
        f_statistics = []
        p_values = []
        
        unique_labels = np.unique(labels)
        
        for feature_idx in range(n_features):
            feature_data = intensities[:, feature_idx]
            group_data = [feature_data[labels == label] for label in unique_labels]
            
            try:
                f_stat, p_val = f_oneway(*group_data)
                f_statistics.append(f_stat)
                p_values.append(p_val)
            except:
                f_statistics.append(0)
                p_values.append(1)
        
        f_statistics = np.array(f_statistics)
        p_values = np.array(p_values)
        
        # Multiple testing correction (Bonferroni)
        corrected_p_values = p_values * n_features
        corrected_p_values = np.minimum(corrected_p_values, 1.0)
        
        # Overall test result
        significant_features = np.sum(corrected_p_values < alpha)
        overall_significant = significant_features > 0
        
        return {
            'test_statistic': np.mean(f_statistics),
            'p_value': np.min(p_values),
            'corrected_p_values': corrected_p_values,
            'significant': overall_significant,
            'significant_features': significant_features,
            'total_features': n_features,
            'method': 'ANOVA'
        }

    def _plot_significance_results(self, results, method):
        """Plot statistical significance results."""
        self.stats_fig.clear()
        
        if method in ['PERMANOVA', 'ANOSIM']:
            # Permutation-based methods
            ax1 = self.stats_fig.add_subplot(2, 2, 1)  # Null distribution
            ax2 = self.stats_fig.add_subplot(2, 2, 2)  # Test result
            
            # Plot null distribution
            permuted_stats = results['permuted_stats']
            observed_stat = results['test_statistic']
            
            ax1.hist(permuted_stats, bins=50, alpha=0.7, color='lightblue', 
                    edgecolor='black', label='Null distribution')
            ax1.axvline(x=observed_stat, color='red', linestyle='--', linewidth=2,
                       label=f'Observed: {observed_stat:.3f}')
            ax1.set_xlabel(f'{method} Statistic')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{method} Null Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot test result
            p_value = results['p_value']
            significance = "Significant" if results['significant'] else "Not Significant"
            
            colors = ['red' if results['significant'] else 'green']
            bars = ax2.bar(['Test Result'], [p_value], color=colors, alpha=0.7)
            ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α = 0.05')
            ax2.set_ylabel('p-value')
            ax2.set_title(f'{method} Test Result\n{significance}')
            ax2.set_ylim(0, max(0.1, p_value * 1.2))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add text annotation
            ax2.text(0, p_value + 0.01, f'p = {p_value:.4f}', 
                    ha='center', va='bottom', fontweight='bold')
            
        else:
            # Feature-wise methods (ANOVA, Kruskal-Wallis)
            ax1 = self.stats_fig.add_subplot(2, 2, 1)  # p-value distribution
            ax2 = self.stats_fig.add_subplot(2, 2, 2)  # Significant features
            ax3 = self.stats_fig.add_subplot(2, 2, 3)  # Corrected p-values
            
            corrected_p_values = results['corrected_p_values']
            wavenumbers = self.cluster_data['wavenumbers']
            
            # p-value distribution
            ax1.hist(corrected_p_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            ax1.set_xlabel('Corrected p-value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Corrected p-values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Significant features summary
            significant_features = results['significant_features']
            total_features = results['total_features']
            
            labels = ['Significant', 'Not Significant']
            sizes = [significant_features, total_features - significant_features]
            colors = ['red', 'lightgray']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Feature Significance\n({significant_features}/{total_features} significant)')
            
            # Plot corrected p-values vs wavenumber
            if len(wavenumbers) == len(corrected_p_values):
                significant_mask = corrected_p_values < 0.05
                ax3.scatter(wavenumbers[~significant_mask], corrected_p_values[~significant_mask], 
                           c='gray', alpha=0.5, s=20, label='Not significant')
                if np.any(significant_mask):
                    ax3.scatter(wavenumbers[significant_mask], corrected_p_values[significant_mask], 
                               c='red', s=30, label='Significant')
                ax3.axhline(y=0.05, color='black', linestyle='--', alpha=0.7)
                ax3.set_xlabel('Wavenumber (cm⁻¹)')
                ax3.set_ylabel('Corrected p-value')
                ax3.set_title('Significance by Wavenumber')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        self.stats_fig.tight_layout()
        self.stats_canvas.draw()

    def run_comprehensive_statistics(self):
        """Run comprehensive statistical analysis."""
        try:
            if self.cluster_data['labels'] is None:
                QMessageBox.warning(self, "No Clustering Data", 
                                  "Please run clustering analysis first.")
                return
            
            # Run all analyses if their checkboxes are checked
            results = {}
            
            if self.feature_importance_cb.isChecked():
                self.calculate_feature_importance()
                results['feature_importance'] = "Completed"
            
            if self.discriminant_analysis_cb.isChecked():
                self.perform_discriminant_analysis()
                results['discriminant_analysis'] = "Completed"
            
            if self.significance_testing_cb.isChecked():
                self.test_statistical_significance()
                results['significance_testing'] = "Completed"
            
            # Display comprehensive summary
            self._display_comprehensive_summary(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Comprehensive Statistics Error", 
                               f"Failed to run comprehensive statistics:\n{str(e)}")
    
    def _display_comprehensive_summary(self, results):
        """Display comprehensive analysis summary."""
        from PySide6.QtCore import QDateTime
        
        results_text = "Comprehensive Statistical Analysis Summary\n"
        results_text += "=" * 50 + "\n\n"
        
        results_text += f"Analysis Date: {QDateTime.currentDateTime().toString()}\n"
        results_text += f"Dataset: {len(self.cluster_data['labels'])} spectra, "
        results_text += f"{len(np.unique(self.cluster_data['labels']))} clusters\n\n"
        
        results_text += "Completed Analyses:\n"
        results_text += "-" * 20 + "\n"
        
        for analysis, status in results.items():
            analysis_name = analysis.replace('_', ' ').title()
            results_text += f"• {analysis_name}: {status}\n"
        
        if not results:
            results_text += "No analyses were selected or completed.\n"
        
        results_text += f"\nRecommendations:\n"
        results_text += "-" * 15 + "\n"
        
        n_clusters = len(np.unique(self.cluster_data['labels']))
        n_samples = len(self.cluster_data['labels'])
        
        if n_samples < 30:
            results_text += "• Consider collecting more data for robust statistical analysis\n"
        
        if n_clusters > n_samples / 10:
            results_text += "• Large number of clusters relative to sample size may indicate overfitting\n"
        
        results_text += "• Examine feature importance results to identify key spectral regions\n"
        results_text += "• Use discriminant analysis results to assess cluster separability\n"
        results_text += "• Consider significance testing results when interpreting clustering validity\n"
        
        results_text += f"\nNext Steps:\n"
        results_text += "• Export results for further analysis\n"
        results_text += "• Consider validation with independent datasets\n"
        results_text += "• Investigate chemical interpretation of important features\n"
        
        # Append to existing results (don't overwrite)
        current_text = self.stats_results.toPlainText()
        if current_text:
            full_text = current_text + "\n\n" + results_text
        else:
            full_text = results_text
        
        self.stats_results.setText(full_text)

    # Database Import Dialog placeholder class
    # Method removed - using the functional version at line 2526

    def show_nmf_clustering_info(self):
        """Display information about NMF clustering options and how to use them."""
        print("\n" + "="*60)
        print("NMF (Non-negative Matrix Factorization) CLUSTERING GUIDE")
        print("="*60)
        
        print("\n🔍 WHAT IS NMF CLUSTERING?")
        print("NMF separates your spectra into non-negative components:")
        print("• Each component represents a 'pure' spectral signature")
        print("• Mixing weights show how much each component contributes")
        print("• Useful for separating mixed phases (e.g., sample + corundum)")
        print("• Can remove contaminants or substrates automatically")
        
        print("\n📍 HOW TO ACCESS NMF CLUSTERING:")
        print("1. In the 'Clustering' tab")
        print("2. Go to 'Preprocessing' section")
        print("3. Change 'Phase Separation Method' to 'NMF Separation'")
        print("4. Adjust 'NMF Components' (2-10, default=3)")
        print("5. Run clustering normally")
        
        print("\n⚙️  NMF PARAMETERS:")
        print("• Components: Number of pure spectral components to find")
        print("  - 2-3: Simple mixtures (sample + substrate)")
        print("  - 3-5: Complex mixtures (multiple phases)")
        print("  - 5+: Very complex samples (many components)")
        
        print("\n💡 WHEN TO USE NMF:")
        print("✅ Your samples have substrate peaks (corundum, diamond, etc.)")
        print("✅ You want to separate sample from contaminants")
        print("✅ Your spectra are mixtures of known components")
        print("✅ You need to remove background signatures")
        
        print("\n❌ WHEN NOT TO USE NMF:")
        print("❌ Pure single-phase samples")
        print("❌ When you want full spectral information")
        print("❌ Samples with negative spectral features")
        
        print("\n🎯 CURRENT PREPROCESSING METHOD:")
        current_method = self.phase_method_combo.currentText() if hasattr(self, 'phase_method_combo') else 'Unknown'
        print(f"Currently selected: {current_method}")
        
        if current_method != 'NMF Separation':
            print("⚠️  NMF is NOT currently selected!")
            print("To enable NMF clustering:")
            print("1. Change 'Phase Separation Method' to 'NMF Separation'")
            print("2. The 'NMF Components' control will become available")
            print("3. Set the number of components you expect")
            print("4. Run clustering as normal")
        else:
            print("✅ NMF is currently enabled!")
            n_components = self.nmf_components_spinbox.value() if hasattr(self, 'nmf_components_spinbox') else 3
            print(f"Number of components: {n_components}")
            
        print("\n📊 NMF vs OTHER METHODS:")
        print("• NMF Separation: Removes contaminants, finds pure components")
        print("• Carbon Soot Optimization: Carbon-specific features (D/G bands)")
        print("• Exclude Regions: Manually removes wavenumber ranges")
        print("• Corundum Correction: Specifically removes corundum peaks")
        
        print("\n🔬 EXAMPLE APPLICATIONS:")
        print("1. Carbon on silicon substrate → 2-3 components")
        print("2. Mineral mixture analysis → 3-5 components")
        print("3. Pharmaceutical formulations → 2-4 components")
        print("4. Coating + substrate → 2 components")
        
        print("\n📈 INTERPRETING NMF RESULTS:")
        print("• Components matrix (H): Pure spectral signatures")
        print("• Weights matrix (W): How much each component contributes")
        print("• Clustering uses the weights to group similar spectra")
        print("• Samples with similar mixing ratios cluster together")
        
        # Check if NMF results are available
        if hasattr(self, 'cluster_data') and 'nmf_components' in self.cluster_data:
            print(f"\n✅ NMF RESULTS AVAILABLE:")
            print(f"• Components found: {self.cluster_data['nmf_components'].shape[0] if self.cluster_data['nmf_components'] is not None else 'None'}")
            print(f"• Samples processed: {len(self.cluster_data.get('nmf_weights', []))}")
            if 'corundum_component_idx' in self.cluster_data:
                print(f"• Corundum component index: {self.cluster_data['corundum_component_idx']}")
        else:
            print(f"\n❌ NO NMF RESULTS AVAILABLE")
            print("Run clustering with 'NMF Separation' method to generate results")
        
        print("\n" + "="*60)

    def export_results(self):
        """Export all analysis results to files."""
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            
            if folder:
                # Export clustering results
                if hasattr(self, 'analysis_results_text') and self.analysis_results_text.toPlainText():
                    with open(os.path.join(folder, 'cluster_analysis.txt'), 'w') as f:
                        f.write(self.analysis_results_text.toPlainText())
                
                # Export cluster assignments if available
                if self.cluster_data['labels'] is not None:
                    import pandas as pd
                    
                    # Create results dataframe
                    results_data = {
                        'spectrum_index': range(len(self.cluster_data['labels'])),
                        'cluster_label': self.cluster_data['labels']
                    }
                    
                    # Add metadata if available
                    if 'spectrum_metadata' in self.cluster_data:
                        filenames = [meta.get('filename', f'spectrum_{i}') 
                                   for i, meta in enumerate(self.cluster_data['spectrum_metadata'])]
                        results_data['filename'] = filenames
                    
                    df = pd.DataFrame(results_data)
                    df.to_csv(os.path.join(folder, 'cluster_assignments.csv'), index=False)
                
                QMessageBox.information(self, "Export Complete", f"Results exported to:\n{folder}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")

    def plot_silhouette_analysis(self, unique_labels, cluster_silhouettes, silhouette_avg):
        """Plot silhouette analysis results."""
        try:
            self.validation_fig.clear()
            
            # Create 2x2 subplot layout
            ax1 = self.validation_fig.add_subplot(2, 2, 1)  # Silhouette plot
            ax2 = self.validation_fig.add_subplot(2, 2, 2)  # Cluster silhouette scores
            ax3 = self.validation_fig.add_subplot(2, 2, 3)  # Sample distribution
            ax4 = self.validation_fig.add_subplot(2, 2, 4)  # Threshold analysis
            
            # 1. Classic silhouette plot
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            y_lower = 10
            
            for i, label in enumerate(unique_labels):
                cluster_silhouette_values = cluster_silhouettes[label]['samples']
                cluster_silhouette_values.sort()
                
                size_cluster = len(cluster_silhouette_values)
                y_upper = y_lower + size_cluster
                
                color = colors[i]
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
                y_lower = y_upper + 10
            
            ax1.set_xlabel('Silhouette coefficient values')
            ax1.set_ylabel('Cluster label')
            ax1.set_title('Silhouette Plot for Individual Samples')
            
            # Add average silhouette line
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--", 
                       label=f'Average Score: {silhouette_avg:.3f}')
            ax1.legend()
            
            # 2. Cluster silhouette scores
            cluster_means = [cluster_silhouettes[label]['mean'] for label in unique_labels]
            cluster_stds = [cluster_silhouettes[label]['std'] for label in unique_labels]
            
            x_pos = np.arange(len(unique_labels))
            bars = ax2.bar(x_pos, cluster_means, yerr=cluster_stds, 
                          capsize=5, color=colors, alpha=0.7)
            
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Mean Silhouette Score')
            ax2.set_title('Mean Silhouette Score by Cluster')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'C{label}' for label in unique_labels])
            ax2.axhline(y=silhouette_avg, color="red", linestyle="--", alpha=0.7)
            ax2.grid(True, alpha=0.3)
            
            # 3. Sample distribution histogram
            all_samples = np.concatenate([cluster_silhouettes[label]['samples'] 
                                        for label in unique_labels])
            ax3.hist(all_samples, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(x=silhouette_avg, color="red", linestyle="--", 
                       label=f'Average: {silhouette_avg:.3f}')
            ax3.axvline(x=self.min_silhouette_threshold.value(), color="orange", 
                       linestyle="--", label=f'Threshold: {self.min_silhouette_threshold.value():.3f}')
            ax3.set_xlabel('Silhouette Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Silhouette Scores')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Threshold analysis
            threshold = self.min_silhouette_threshold.value()
            good_samples = np.sum(all_samples >= threshold)
            poor_samples = np.sum(all_samples < threshold)
            
            wedges, texts, autotexts = ax4.pie([good_samples, poor_samples], 
                                              labels=[f'Good (≥{threshold:.1f})', f'Poor (<{threshold:.1f})'],
                                              autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax4.set_title('Sample Quality Distribution')
            
            self.validation_fig.tight_layout()
            self.validation_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting silhouette analysis: {str(e)}")
    
    def generate_silhouette_results(self, silhouette_avg, cluster_silhouettes, sample_silhouette_values, labels):
        """Generate detailed silhouette analysis results text."""
        results_text = "Silhouette Analysis Results\n"
        results_text += "=" * 30 + "\n\n"
        
        results_text += f"Overall Silhouette Score: {silhouette_avg:.3f}\n"
        threshold = self.min_silhouette_threshold.value()
        results_text += f"Quality Threshold: {threshold:.3f}\n\n"
        
        # Overall assessment
        if silhouette_avg >= 0.7:
            assessment = "Excellent clustering quality"
        elif silhouette_avg >= 0.5:
            assessment = "Good clustering quality"
        elif silhouette_avg >= 0.3:
            assessment = "Moderate clustering quality"
        else:
            assessment = "Poor clustering quality"
        
        results_text += f"Assessment: {assessment}\n\n"
        
        # Cluster-specific results
        results_text += "Cluster-Specific Silhouette Scores:\n"
        results_text += "-" * 35 + "\n"
        
        for label in sorted(cluster_silhouettes.keys()):
            stats = cluster_silhouettes[label]
            results_text += f"Cluster {label}:\n"
            results_text += f"  • Mean score: {stats['mean']:.3f}\n"
            results_text += f"  • Std deviation: {stats['std']:.3f}\n"
            results_text += f"  • Range: {stats['min']:.3f} to {stats['max']:.3f}\n"
            
            # Quality assessment for this cluster
            if stats['mean'] >= threshold:
                quality = "Good"
            else:
                quality = "Needs improvement"
            results_text += f"  • Quality: {quality}\n\n"
        
        # Sample quality statistics
        good_samples = np.sum(sample_silhouette_values >= threshold)
        total_samples = len(sample_silhouette_values)
        poor_samples = total_samples - good_samples
        
        results_text += "Sample Quality Distribution:\n"
        results_text += "-" * 28 + "\n"
        results_text += f"• Good samples (≥{threshold:.1f}): {good_samples} ({good_samples/total_samples*100:.1f}%)\n"
        results_text += f"• Poor samples (<{threshold:.1f}): {poor_samples} ({poor_samples/total_samples*100:.1f}%)\n\n"
        
        # Recommendations
        results_text += "Recommendations:\n"
        results_text += "-" * 15 + "\n"
        
        if silhouette_avg < 0.3:
            results_text += "• Consider different clustering parameters\n"
            results_text += "• Try different number of clusters\n"
            results_text += "• Check data preprocessing\n"
        elif poor_samples > total_samples * 0.3:
            results_text += "• Some samples may be outliers\n"
            results_text += "• Consider outlier detection\n"
        else:
            results_text += "• Clustering quality is acceptable\n"
            results_text += "• Consider fine-tuning if needed\n"
        
        return results_text
    
    def plot_transition_analysis(self, coords_2d, labels, centroids_2d, transition_analysis, unique_labels):
        """Plot cluster transition analysis results."""
        try:
            self.validation_fig.clear()
            
            # Create 2x2 subplot layout
            ax1 = self.validation_fig.add_subplot(2, 2, 1)  # 2D cluster plot with boundaries
            ax2 = self.validation_fig.add_subplot(2, 2, 2)  # Boundary thickness analysis
            ax3 = self.validation_fig.add_subplot(2, 2, 3)  # Centroid distances
            ax4 = self.validation_fig.add_subplot(2, 2, 4)  # Boundary points count
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # 1. 2D cluster plot with boundary regions
            for i, label in enumerate(unique_labels):
                cluster_mask = labels == label
                cluster_coords = coords_2d[cluster_mask]
                
                ax1.scatter(cluster_coords[:, 0], cluster_coords[:, 1], 
                          c=[color_map[label]], label=f'Cluster {label}', 
                          alpha=0.7, s=30)
                
                # Plot centroid
                ax1.scatter(centroids_2d[i, 0], centroids_2d[i, 1], 
                          c=[color_map[label]], marker='x', s=100, linewidths=3)
            
            # Highlight boundary points
            for pair_key, analysis in transition_analysis.items():
                boundary_points_i = analysis['boundary_points_i']
                boundary_points_j = analysis['boundary_points_j']
                
                if len(boundary_points_i) > 0:
                    ax1.scatter(boundary_points_i[:, 0], boundary_points_i[:, 1], 
                              c='red', marker='o', s=20, alpha=0.5)
                if len(boundary_points_j) > 0:
                    ax1.scatter(boundary_points_j[:, 0], boundary_points_j[:, 1], 
                              c='red', marker='o', s=20, alpha=0.5)
            
            ax1.set_xlabel('PC1')
            ax1.set_ylabel('PC2')
            ax1.set_title('Clusters with Boundary Points (red)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Boundary thickness analysis
            pair_names = list(transition_analysis.keys())
            boundary_thicknesses = [transition_analysis[pair]['boundary_thickness'] 
                                  for pair in pair_names]
            
            x_pos = np.arange(len(pair_names))
            bars = ax2.bar(x_pos, boundary_thicknesses, alpha=0.7, color='orange')
            
            ax2.set_xlabel('Cluster Pairs')
            ax2.set_ylabel('Boundary Thickness')
            ax2.set_title('Boundary Thickness Between Clusters')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'C{pair}' for pair in pair_names], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 3. Centroid distances
            centroid_distances = [transition_analysis[pair]['centroid_distance'] 
                                for pair in pair_names]
            
            bars = ax3.bar(x_pos, centroid_distances, alpha=0.7, color='skyblue')
            
            ax3.set_xlabel('Cluster Pairs')
            ax3.set_ylabel('Centroid Distance')
            ax3.set_title('Distance Between Cluster Centroids')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f'C{pair}' for pair in pair_names], rotation=45)
            ax3.grid(True, alpha=0.3)
            
                        # 4. Boundary points count
            boundary_counts = [(transition_analysis[pair]['n_boundary_i'] + 
                              transition_analysis[pair]['n_boundary_j']) 
                             for pair in pair_names]
            
            bars = ax4.bar(x_pos, boundary_counts, alpha=0.7, color='lightgreen')
            
            ax4.set_xlabel('Cluster Pairs')
            ax4.set_ylabel('Number of Boundary Points')
            ax4.set_title('Boundary Points Count')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'C{pair}' for pair in pair_names], rotation=45)
            ax4.grid(True, alpha=0.3)
            
            self.validation_fig.tight_layout()
            self.validation_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting transition analysis: {str(e)}")
    
    def generate_transition_results(self, transition_analysis, unique_labels):
        """Generate detailed transition analysis results text."""
        results_text = "Cluster Transition Analysis Results\n"
        results_text += "=" * 35 + "\n\n"
        
        results_text += f"Number of Cluster Pairs Analyzed: {len(transition_analysis)}\n\n"
        
        # Analyze each cluster pair
        results_text += "Cluster Pair Analysis:\n"
        results_text += "-" * 22 + "\n"
        
        # Sort by centroid distance
        sorted_pairs = sorted(transition_analysis.items(), 
                            key=lambda x: x[1]['centroid_distance'])
        
        for pair_key, analysis in sorted_pairs:
            results_text += f"Clusters {pair_key}:\n"
            results_text += f"  • Centroid distance: {analysis['centroid_distance']:.3f}\n"
            results_text += f"  • Boundary thickness: {analysis['boundary_thickness']:.3f}\n"
            results_text += f"  • Boundary points: {analysis['n_boundary_i'] + analysis['n_boundary_j']}\n"
            
            # Separation assessment
            separation_ratio = analysis['centroid_distance'] / analysis['boundary_thickness']
            if separation_ratio > 3:
                separation = "Well separated"
            elif separation_ratio > 2:
                separation = "Moderately separated"
            else:
                separation = "Poorly separated"
            
            results_text += f"  • Separation: {separation} (ratio: {separation_ratio:.2f})\n\n"
        
        # Overall statistics
        all_distances = [analysis['centroid_distance'] for analysis in transition_analysis.values()]
        all_thicknesses = [analysis['boundary_thickness'] for analysis in transition_analysis.values()]
        all_boundary_counts = [(analysis['n_boundary_i'] + analysis['n_boundary_j']) 
                             for analysis in transition_analysis.values()]
        
        results_text += "Overall Statistics:\n"
        results_text += "-" * 18 + "\n"
        results_text += f"• Mean centroid distance: {np.mean(all_distances):.3f}\n"
        results_text += f"• Mean boundary thickness: {np.mean(all_thicknesses):.3f}\n"
        results_text += f"• Mean boundary points: {np.mean(all_boundary_counts):.1f}\n\n"
        
        # Recommendations
        results_text += "Separation Quality Assessment:\n"
        results_text += "-" * 28 + "\n"
        
        well_separated = sum(1 for analysis in transition_analysis.values() 
                           if analysis['centroid_distance'] / analysis['boundary_thickness'] > 3)
        total_pairs = len(transition_analysis)
        
        results_text += f"• Well separated pairs: {well_separated}/{total_pairs} ({well_separated/total_pairs*100:.1f}%)\n"
        
        if well_separated < total_pairs * 0.5:
            results_text += "\nRecommendations:\n"
            results_text += "• Consider increasing number of clusters\n"
            results_text += "• Some clusters may need to be merged\n"
            results_text += "• Review clustering parameters\n"
        else:
            results_text += "\nGood cluster separation achieved!\n"
        
        return results_text
    
    def plot_stability_analysis(self, stability_stats, cluster_stability, stability_scores):
        """Plot cluster stability analysis results."""
        try:
            self.validation_fig.clear()
            
            # Create 2x2 subplot layout
            ax1 = self.validation_fig.add_subplot(2, 2, 1)  # ARI distribution
            ax2 = self.validation_fig.add_subplot(2, 2, 2)  # Cluster consistency
            ax3 = self.validation_fig.add_subplot(2, 2, 3)  # Stability over iterations
            ax4 = self.validation_fig.add_subplot(2, 2, 4)  # Summary statistics
            
            # 1. ARI distribution histogram
            if stability_scores:
                ax1.hist(stability_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(x=stability_stats['mean_ari'], color='red', linestyle='--', 
                           label=f"Mean: {stability_stats['mean_ari']:.3f}")
                ax1.set_xlabel('Adjusted Rand Index')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Distribution of Stability Scores')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No stability data available', 
                         transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title('Stability Score Distribution')
            
            # 2. Cluster consistency
            cluster_labels = list(cluster_stability.keys())
            if cluster_labels:
                consistencies = [cluster_stability[label]['mean_consistency'] for label in cluster_labels]
                consistency_stds = [cluster_stability[label]['std_consistency'] for label in cluster_labels]
                
                x_pos = np.arange(len(cluster_labels))
                bars = ax2.bar(x_pos, consistencies, yerr=consistency_stds, 
                              capsize=5, alpha=0.7, color='lightgreen')
                
                ax2.set_xlabel('Cluster')
                ax2.set_ylabel('Consistency Score')
                ax2.set_title('Cluster Consistency Across Bootstrap Samples')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([f'C{label}' for label in cluster_labels])
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No cluster data available', 
                         transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Cluster Consistency')
            
            # 3. Stability over iterations
            if stability_scores:
                iterations = range(1, len(stability_scores) + 1)
                ax3.plot(iterations, stability_scores, 'b-', alpha=0.7, linewidth=1)
                ax3.axhline(y=stability_stats['mean_ari'], color='red', linestyle='--', 
                           label=f"Mean: {stability_stats['mean_ari']:.3f}")
                ax3.set_xlabel('Bootstrap Iteration')
                ax3.set_ylabel('Adjusted Rand Index')
                ax3.set_title('Stability Across Bootstrap Iterations')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No iteration data available', 
                         transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Stability Over Iterations')
            
            # 4. Summary statistics
            ax4.axis('off')
            
            summary_text = f"""Stability Analysis Summary
            
Mean ARI: {stability_stats['mean_ari']:.3f}
Std ARI: {stability_stats['std_ari']:.3f}
Min ARI: {stability_stats['min_ari']:.3f}
Max ARI: {stability_stats['max_ari']:.3f}

Successful Iterations: {stability_stats['successful_iterations']}
Total Iterations: {stability_stats['total_iterations']}
Success Rate: {stability_stats['successful_iterations']/stability_stats['total_iterations']*100:.1f}%

Stability Assessment:"""
            
            # Add stability assessment
            mean_ari = stability_stats['mean_ari']
            if mean_ari >= 0.8:
                assessment = "Excellent"
            elif mean_ari >= 0.6:
                assessment = "Good"
            elif mean_ari >= 0.4:
                assessment = "Moderate"
            else:
                assessment = "Poor"
            
            summary_text += f" {assessment}"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
            
            self.validation_fig.tight_layout()
            self.validation_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting stability analysis: {str(e)}")

    def export_clusters_to_folders(self):
        """Export each cluster's spectra to separate folders."""
        if (self.cluster_data['labels'] is None or 
            self.cluster_data['intensities'] is None or
            self.cluster_data['spectrum_metadata'] is None):
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
            metadata = self.cluster_data['spectrum_metadata']
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
                progress.setLabelText(f"Exporting cluster {cluster_id}...")
                
                # Create cluster folder
                cluster_folder = os.path.join(export_dir, f"Cluster_{cluster_id}")
                os.makedirs(cluster_folder, exist_ok=True)
                
                # Get spectra for this cluster
                cluster_mask = labels == cluster_id
                cluster_intensities = intensities[cluster_mask]
                cluster_metadata = [metadata[j] for j in range(len(metadata)) if cluster_mask[j]]
                
                # Export individual spectra
                for j, (spectrum_intensity, spectrum_metadata) in enumerate(zip(cluster_intensities, cluster_metadata)):
                    filename = spectrum_metadata.get('filename', f'spectrum_{j}.txt')
                    # Clean filename
                    clean_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                    if not clean_filename.endswith('.txt'):
                        clean_filename += '.txt'
                    
                    filepath = os.path.join(cluster_folder, clean_filename)
                    
                    # Write spectrum data
                    with open(filepath, 'w') as f:
                        # Write metadata header
                        f.write(f"# Cluster: {cluster_id}\n")
                        f.write(f"# Original file: {filename}\n")
                        for key, value in spectrum_metadata.items():
                            if key != 'filename':
                                f.write(f"# {key}: {value}\n")
                        f.write("# Wavenumber\tIntensity\n")
                        
                        # Write spectrum data
                        for wavenumber, intensity in zip(wavenumbers, spectrum_intensity):
                            f.write(f"{wavenumber:.2f}\t{intensity:.6f}\n")
                
                # Create cluster summary file
                summary_file = os.path.join(cluster_folder, f"cluster_{cluster_id}_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"Cluster {cluster_id} Summary\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Number of spectra: {len(cluster_intensities)}\n")
                    f.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("Spectra files:\n")
                    for j, spectrum_metadata in enumerate(cluster_metadata):
                        filename = spectrum_metadata.get('filename', f'spectrum_{j}.txt')
                        f.write(f"  {j+1}. {filename}\n")
                
                exported_clusters += 1
            
            progress.setValue(len(unique_labels))
            
            QMessageBox.information(
                self, "Export Complete", 
                f"Successfully exported {exported_clusters} clusters to:\n{export_dir}"
            )
            
            # Restore window focus after file dialog
            try:
                from core.window_focus_manager import restore_window_focus_after_dialog
                restore_window_focus_after_dialog(self)
            except ImportError:
                # Fallback if focus manager not available
                self.raise_()
                self.activateWindow()
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting clusters: {str(e)}")
            print(f"Export error: {str(e)}")

    def export_summed_cluster_spectra(self):
        """Export high-quality summed spectra for each cluster as individual files and plot."""
        if (self.cluster_data['labels'] is None or 
            self.cluster_data['intensities'] is None or
            self.cluster_data['wavenumbers'] is None):
            QMessageBox.warning(self, "No Data", "No cluster data available for export.")
            return
        
        try:
            # Get export directory for summed spectra files
            export_dir = QFileDialog.getExistingDirectory(
                self, "Select Export Directory for Summed Spectra", "",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if not export_dir:
                return
            
            # Get plot file path
            plot_filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Summed Spectra Plot", 
                "cluster_summed_spectra.png",
                "PNG files (*.png);;PDF files (*.pdf);;All files (*)"
            )
            
            if not plot_filepath:
                return
            
            # Import matplotlib config
            try:
                from core.matplotlib_config import apply_theme
                apply_theme('publication')
            except ImportError:
                pass  # Use default matplotlib settings
            
            labels = self.cluster_data['labels']
            intensities = self.cluster_data['intensities']
            wavenumbers = self.cluster_data['wavenumbers']
            unique_labels = np.unique(labels)
            
            # Create progress dialog
            progress = QProgressDialog("Processing summed spectra...", "Cancel", 0, len(unique_labels), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            
            # Store processed spectra for plotting
            processed_spectra = {}
            
            for i, cluster_id in enumerate(unique_labels):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"Processing cluster {cluster_id}...")
                
                # Get spectra for this cluster
                cluster_mask = labels == cluster_id
                cluster_intensities = intensities[cluster_mask]
                
                # Step 1: Preprocessing and normalization
                processed_cluster_spectra = self._preprocess_cluster_spectra(cluster_intensities, wavenumbers)
                
                # Step 2: Create high-quality summed spectrum
                summed_spectrum = self._create_summed_spectrum(processed_cluster_spectra, wavenumbers)
                
                # Step 3: Save individual summed spectrum file
                spectrum_filename = f"cluster_{cluster_id}_summed_spectrum.txt"
                spectrum_filepath = os.path.join(export_dir, spectrum_filename)
                
                with open(spectrum_filepath, 'w') as f:
                    f.write(f"# Cluster {cluster_id} Summed Spectrum\n")
                    f.write(f"# Number of spectra combined: {len(cluster_intensities)}\n")
                    f.write(f"# Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Processing method: Advanced signal enhancement with normalization\n")
                    f.write(f"# Signal-to-noise improvement: ~{np.sqrt(len(cluster_intensities)):.1f}x\n")
                    f.write("# Wavenumber (cm⁻¹)\tIntensity\tStd_Deviation\n")
                    
                    # Calculate standard deviation for each point
                    std_values = np.std(processed_cluster_spectra, axis=0)
                    
                    for j, (wavenumber, intensity, std_val) in enumerate(zip(wavenumbers, summed_spectrum, std_values)):
                        f.write(f"{wavenumber:.2f}\t{intensity:.6f}\t{std_val:.6f}\n")
                
                # Store for plotting
                processed_spectra[cluster_id] = {
                    'summed': summed_spectrum,
                    'individual': processed_cluster_spectra,
                    'std': std_values,
                    'count': len(cluster_intensities)
                }
            
            progress.setValue(len(unique_labels))
            
            # Create visualization
            self._create_summed_spectra_plot(processed_spectra, wavenumbers, plot_filepath)
            
            QMessageBox.information(
                self, "Export Complete", 
                f"Summed spectra exported to:\n{export_dir}\n\nPlot saved to:\n{plot_filepath}"
            )
            
            # Restore window focus after file dialog
            try:
                from core.window_focus_manager import restore_window_focus_after_dialog
                restore_window_focus_after_dialog(self)
            except ImportError:
                # Fallback if focus manager not available
                self.raise_()
                self.activateWindow()
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting summed spectra: {str(e)}")
            print(f"Export error: {str(e)}")

    def _preprocess_cluster_spectra(self, cluster_intensities, wavenumbers):
        """Preprocess cluster spectra for optimal summation."""
        processed_spectra = []
        
        for spectrum in cluster_intensities:
            # Step 1: Remove baseline using polynomial fitting
            baseline_corrected = self._remove_baseline(spectrum, wavenumbers)
            
            # Step 2: Normalize by area under curve
            area_normalized = self._normalize_by_area(baseline_corrected)
            
            # Step 3: Apply smoothing to reduce noise
            smoothed = self._smooth_spectrum(area_normalized)
            
            processed_spectra.append(smoothed)
        
        return np.array(processed_spectra)

    def _remove_baseline(self, spectrum, wavenumbers, degree=3):
        """Remove baseline using polynomial fitting."""
        try:
            # Fit polynomial to spectrum
            coeffs = np.polyfit(wavenumbers, spectrum, degree)
            baseline = np.polyval(coeffs, wavenumbers)
            
            # Subtract baseline
            corrected = spectrum - baseline
            
            # Ensure non-negative values
            corrected = np.maximum(corrected, 0)
            
            return corrected
        except:
            # Fallback: simple minimum subtraction
            return spectrum - np.min(spectrum)

    def _normalize_by_area(self, spectrum):
        """Normalize spectrum by area under the curve."""
        area = np.trapz(spectrum)
        if area > 0:
            return spectrum / area
        else:
            return spectrum

    def _smooth_spectrum(self, spectrum, window_size=5):
        """Apply Savitzky-Golay smoothing to reduce noise."""
        try:
            from scipy.signal import savgol_filter
            # Use odd window size
            if window_size % 2 == 0:
                window_size += 1
            return savgol_filter(spectrum, window_size, 2)
        except ImportError:
            # Fallback: simple moving average
            kernel = np.ones(window_size) / window_size
            return np.convolve(spectrum, kernel, mode='same')

    def _create_summed_spectrum(self, processed_spectra, wavenumbers):
        """Create high-quality summed spectrum with signal enhancement."""
        # Step 1: Calculate mean spectrum
        mean_spectrum = np.mean(processed_spectra, axis=0)
        
        # Step 2: Apply outlier rejection (remove spectra that deviate too much)
        std_spectrum = np.std(processed_spectra, axis=0)
        threshold = 2.0  # Standard deviations
        
        # Calculate deviation for each spectrum
        deviations = []
        for spectrum in processed_spectra:
            deviation = np.mean(np.abs(spectrum - mean_spectrum) / (std_spectrum + 1e-10))
            deviations.append(deviation)
        
        # Keep spectra within threshold
        good_indices = [i for i, dev in enumerate(deviations) if dev < threshold]
        
        if len(good_indices) > 0:
            # Recalculate mean with good spectra only
            filtered_spectra = processed_spectra[good_indices]
            mean_spectrum = np.mean(filtered_spectra, axis=0)
        
        # Step 3: Apply final smoothing for optimal signal
        final_spectrum = self._smooth_spectrum(mean_spectrum, window_size=7)
        
        return final_spectrum

    def _create_summed_spectra_plot(self, processed_spectra, wavenumbers, filepath):
        """Create visualization of summed spectra."""
        n_clusters = len(processed_spectra)
        n_cols = min(3, n_clusters)  # Max 3 columns
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        # Handle different axes layouts properly
        if n_clusters == 1:
            # Single cluster: axes is a single Axes object
            axes_array = [[axes]]
        elif n_rows == 1:
            # Single row: axes is a 1D array
            axes_array = [axes] if n_cols == 1 else axes.reshape(1, -1)
        else:
            # Multiple rows: axes is a 2D array
            axes_array = axes
        
        # Get colormap
        colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'Set1'
        try:
            colormap = plt.cm.get_cmap(colormap_name)
            colors = colormap(np.linspace(0, 1, len(processed_spectra)))
        except:
            colors = plt.cm.Set1(np.linspace(0, 1, len(processed_spectra)))
        
        for i, (cluster_id, data) in enumerate(processed_spectra.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes_array[row][col]
            
            # Plot individual spectra (very transparent)
            for spectrum in data['individual']:
                ax.plot(wavenumbers, spectrum, alpha=0.1, color=colors[i], linewidth=0.3)
            
            # Plot summed spectrum (bold)
            ax.plot(wavenumbers, data['summed'], color=colors[i], linewidth=3, 
                   label=f'Summed (n={data["count"]})')
            
            # Add confidence interval
            ax.fill_between(wavenumbers, 
                          data['summed'] - data['std'], 
                          data['summed'] + data['std'], 
                          alpha=0.2, color=colors[i])
            
            # Add SNR improvement info
            snr_improvement = np.sqrt(data['count'])
            ax.text(0.02, 0.98, f'SNR improvement: ~{snr_improvement:.1f}x', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Normalized Intensity')
            ax.set_title(f'Cluster {cluster_id} - High-Quality Summed Spectrum')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_clusters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < len(axes_array) and col < len(axes_array[row]):
                ax = axes_array[row][col]
                ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def export_xy_plot_data(self):
        """Export the XY coordinates of the current scatter plot (PCA or UMAP) to a CSV file."""
        if not hasattr(self, 'cluster_data') or self.cluster_data.get('labels') is None:
            QMessageBox.warning(self, "No Data", "No cluster data available to export.")
            return
            
        # Determine which coordinates to export based on current visualization method
        method = self.visualization_method_combo.currentText()
        if method == 'PCA' and 'pca_coords' in self.cluster_data:
            coords = self.cluster_data['pca_coords']
            x_label = 'PC1'
            y_label = 'PC2'
        elif method == 'UMAP' and 'umap_coords' in self.cluster_data:
            coords = self.cluster_data['umap_coords']
            x_label = 'UMAP1'
            y_label = 'UMAP2'
        else:
            QMessageBox.warning(self, "No Coordinates", 
                              f"No {method} coordinates available for export.")
            return
        
        # Get the labels and filenames
        labels = self.cluster_data['labels']
        filenames = [f"spectrum_{i}" for i in range(len(labels))]  # Default filenames
        
        # Try to get actual filenames from metadata if available
        if 'spectrum_metadata' in self.cluster_data and self.cluster_data['spectrum_metadata']:
            filenames = [meta.get('filename', f'spectrum_{i}') 
                        for i, meta in enumerate(self.cluster_data['spectrum_metadata'])]
        
        # Get save file path
        default_name = f"{method.lower()}_coordinates.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            f"Export {method} Coordinates",
            default_name,
            "CSV files (*.csv);;All files (*)"
        )
        
        if not filename:
            return  # User cancelled
            
        try:
            # Create a DataFrame with the data
            import pandas as pd
            
            data = {
                'filename': filenames,
                'cluster': labels,
                x_label: coords[:, 0],
                y_label: coords[:, 1]
            }
            
            # Add additional metadata if available
            if 'spectrum_metadata' in self.cluster_data and self.cluster_data['spectrum_metadata']:
                # Add any additional metadata fields that might be useful
                for field in ['sample_id', 'description', 'mineral', 'formula']:
                    if field in self.cluster_data['spectrum_metadata'][0]:
                        data[field] = [meta.get(field, '') for meta in self.cluster_data['spectrum_metadata']]
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Successfully exported {len(df)} data points to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export {method} coordinates:\n{str(e)}")

    def export_cluster_overview(self):
        """Export a comprehensive overview with all clusters in a grid layout."""
        if (self.cluster_data['labels'] is None or 
            self.cluster_data['intensities'] is None or
            self.cluster_data['wavenumbers'] is None):
            QMessageBox.warning(self, "No Data", "No cluster data available for export.")
            return
        
        try:
            # Get export file path
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Cluster Overview", 
                "cluster_overview.png",
                "PNG files (*.png);;PDF files (*.pdf);;All files (*)"
            )
            
            if not filepath:
                return
            
            # Import matplotlib config
            try:
                from core.matplotlib_config import apply_theme
                apply_theme('publication')
            except ImportError:
                pass  # Use default matplotlib settings
            
            labels = self.cluster_data['labels']
            intensities = self.cluster_data['intensities']
            wavenumbers = self.cluster_data['wavenumbers']
            unique_labels = np.unique(labels)
            
            # Create comprehensive overview
            fig = plt.figure(figsize=(16, 12))
            
            # Main plot: All clusters overlaid
            ax1 = plt.subplot(2, 2, 1)
            colormap_name = self.colormap_combo.currentText() if hasattr(self, 'colormap_combo') else 'Set1'
            try:
                colormap = plt.cm.get_cmap(colormap_name)
                colors = colormap(np.linspace(0, 1, len(unique_labels)))
            except:
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, cluster_id in enumerate(unique_labels):
                cluster_mask = labels == cluster_id
                cluster_intensities = intensities[cluster_mask]
                mean_spectrum = np.mean(cluster_intensities, axis=0)
                std_spectrum = np.std(cluster_intensities, axis=0)
                
                ax1.plot(wavenumbers, mean_spectrum, color=colors[i], linewidth=2, 
                        label=f'Cluster {cluster_id} (n={len(cluster_intensities)})')
                ax1.fill_between(wavenumbers, 
                               mean_spectrum - std_spectrum, 
                               mean_spectrum + std_spectrum, 
                               alpha=0.2, color=colors[i])
            
            ax1.set_xlabel('Wavenumber (cm⁻¹)')
            ax1.set_ylabel('Intensity')
            ax1.set_title('All Clusters Overlaid')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Cluster size distribution
            ax2 = plt.subplot(2, 2, 2)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            bars = ax2.bar(range(len(unique_labels)), cluster_sizes, color=colors)
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Number of Spectra')
            ax2.set_title('Cluster Size Distribution')
            ax2.set_xticks(range(len(unique_labels)))
            ax2.set_xticklabels([f'Cluster {label}' for label in unique_labels])
            
            # Add value labels on bars
            for bar, size in zip(bars, cluster_sizes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{size}', ha='center', va='bottom')
            
            # Individual cluster plots (2x2 grid for remaining clusters)
            n_remaining = len(unique_labels)
            if n_remaining > 0:
                # Calculate subplot layout
                n_cols = min(2, n_remaining)
                n_rows = (n_remaining + n_cols - 1) // n_cols
                
                for i, cluster_id in enumerate(unique_labels):
                    if i < 4:  # Only show first 4 clusters in overview
                        row = 2 + (i // 2)
                        col = 1 + (i % 2)
                        ax = plt.subplot(2, 2, col + 2)
                        
                        cluster_mask = labels == cluster_id
                        cluster_intensities = intensities[cluster_mask]
                        
                        # Plot individual spectra
                        for spectrum in cluster_intensities:
                            ax.plot(wavenumbers, spectrum, alpha=0.3, color=colors[i], linewidth=0.5)
                        
                        # Plot mean
                        mean_spectrum = np.mean(cluster_intensities, axis=0)
                        ax.plot(wavenumbers, mean_spectrum, color=colors[i], linewidth=2, 
                               label=f'Mean (n={len(cluster_intensities)})')
                        
                        ax.set_xlabel('Wavenumber (cm⁻¹)')
                        ax.set_ylabel('Intensity')
                        ax.set_title(f'Cluster {cluster_id} Details')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        break  # Only show first cluster in detail view
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            QMessageBox.information(
                self, "Export Complete", 
                f"Cluster overview saved to:\n{filepath}"
            )
            
            # Restore window focus after file dialog
            try:
                from core.window_focus_manager import restore_window_focus_after_dialog
                restore_window_focus_after_dialog(self)
            except ImportError:
                # Fallback if focus manager not available
                self.raise_()
                self.activateWindow()
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting cluster overview: {str(e)}")
            print(f"Export error: {str(e)}")

    def print_carbon_feature_analysis(self):
        """Print analysis of carbon-specific features for debugging."""
        try:
            if 'carbon_features' not in self.cluster_data:
                print("No carbon features available. Run clustering with Carbon Soot Optimization first.")
                return
            
            features = self.cluster_data['carbon_features']
            labels = self.cluster_data['labels']
            
            feature_names = [
                'D_peak_position', 'D_intensity', 'D_width', 'D_integrated',
                'G_peak_position', 'G_intensity', 'G_width', 'G_integrated',
                'ID_IG_ratio', 'D_prime_intensity', 'D_prime_G_ratio',
                'Low_freq_integrated', 'RBM_intensity', '2D_intensity', '2D_G_ratio',
                'G_asymmetry', 'Background_slope'
            ]
            
            print("\n=== Carbon Feature Analysis ===")
            unique_labels = np.unique(labels)
            
            for feature_idx, feature_name in enumerate(feature_names[:features.shape[1]]):
                print(f"\n{feature_name}:")
                for label in unique_labels:
                    mask = labels == label
                    feature_values = features[mask, feature_idx]
                    mean_val = np.mean(feature_values)
                    std_val = np.std(feature_values)
                    print(f"  Cluster {label}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Calculate ID/IG ratios specifically
            if features.shape[1] > 8:
                print(f"\n=== ID/IG Ratio Analysis (Key Discriminator) ===")
                for label in unique_labels:
                    mask = labels == label
                    id_ig_values = features[mask, 8]  # ID/IG ratio is feature index 8
                    mean_ratio = np.mean(id_ig_values)
                    std_ratio = np.std(id_ig_values)
                    print(f"Cluster {label} - ID/IG: {mean_ratio:.3f} ± {std_ratio:.3f}")
                    
                    # Interpretation
                    if mean_ratio > 1.5:
                        interpretation = "High disorder (amorphous/nanocrystalline)"
                    elif mean_ratio > 1.0:
                        interpretation = "Moderate disorder"
                    elif mean_ratio > 0.5:
                        interpretation = "Low disorder (more graphitic)"
                    else:
                        interpretation = "Highly graphitic"
                    print(f"  Interpretation: {interpretation}")
            
        except Exception as e:
            print(f"Error in carbon feature analysis: {e}")

    def suggest_clustering_improvements(self):
        """Suggest improvements based on current clustering results."""
        if (self.cluster_data['labels'] is None or 
            'carbon_features' not in self.cluster_data):
            print("No clustering results available. Run clustering with Carbon Soot Optimization first.")
            return
        
        try:
            features = self.cluster_data['carbon_features']
            labels = self.cluster_data['labels']
            unique_labels = np.unique(labels)
            
            print("\n=== Clustering Improvement Suggestions ===")
            
            # 1. Check if clusters are well-separated in feature space
            from sklearn.metrics import silhouette_score
            if features.shape[0] > 1 and len(unique_labels) > 1:
                silhouette_avg = silhouette_score(features, labels)
                print(f"Current silhouette score: {silhouette_avg:.3f}")
                
                if silhouette_avg < 0.3:
                    print("⚠️  Poor cluster separation detected!")
                    print("Suggestions:")
                    print("- Try different UMAP parameters (lower min_dist, higher n_neighbors)")
                    print("- Check data quality and preprocessing")
                    print("- Consider fewer clusters")
                elif silhouette_avg < 0.5:
                    print("⚠️  Moderate cluster separation")
                    print("Suggestions:")
                    print("- Fine-tune UMAP parameters")
                    print("- Verify sample labeling if known groups exist")
            
            # 2. Check feature variance
            feature_vars = np.var(features, axis=0)
            low_variance_features = np.where(feature_vars < 1e-6)[0]
            
            if len(low_variance_features) > 0:
                print(f"⚠️  {len(low_variance_features)} features have very low variance")
                print("Consider removing these features or checking data preprocessing")
            
            # 3. Check for potential outliers
            from scipy.stats import zscore
            z_scores = np.abs(zscore(features, axis=0))
            outlier_threshold = 3.0
            potential_outliers = np.where(np.any(z_scores > outlier_threshold, axis=1))[0]
            
            if len(potential_outliers) > 0:
                print(f"⚠️  {len(potential_outliers)} potential outlier samples detected")
                print("Consider reviewing these samples for data quality")
            
            print("\n=== Carbon-Specific Recommendations ===")
            print("For diesel/unleaded/charcoal discrimination, focus on:")
            print("1. ID/IG ratio (disorder level)")
            print("2. G-band position (graphitic quality)")
            print("3. D-band width (crystallite size)")
            print("4. Background slope (amorphous content)")
            print("5. Low-frequency modes (structural differences)")
            
        except Exception as e:
            print(f"Error in clustering improvement analysis: {e}")


def launch_cluster_analysis(parent, raman_app):
    """Launch the cluster analysis window."""
    cluster_window = RamanClusterAnalysisQt6(parent, raman_app)
    cluster_window.show()
    return cluster_window