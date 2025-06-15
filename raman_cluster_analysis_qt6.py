#!/usr/bin/env python3
"""
RamanLab Qt6 Version - Raman Cluster Analysis
Qt6 conversion of the Raman Cluster Analysis GUI with Advanced Ion Exchange Analysis
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages

import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist
import glob
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
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
    from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    # Fallback for older matplotlib versions
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar

# Qt6 imports
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QSpinBox, QDoubleSpinBox, QFrame, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QFormLayout, QDialogButtonBox,
    QListWidget, QListWidgetItem, QInputDialog, QApplication, QGridLayout
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
            'silhouette_scores': None  # Store silhouette analysis
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
        self.phase_method_combo.addItems(['None', 'Exclude Regions', 'Corundum Correction', 'NMF Separation'])
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
        
        # Add run clustering button
        run_clustering_btn = QPushButton("Run Clustering")
        run_clustering_btn.clicked.connect(self.run_clustering)
        layout.addWidget(run_clustering_btn)
        
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
        self.heatmap_colormap.currentTextChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.heatmap_colormap)
        
        # Normalization method
        controls_layout.addWidget(QLabel("Normalization:"))
        self.heatmap_norm = QComboBox()
        self.heatmap_norm.addItems(['linear', 'log', 'sqrt', 'row', 'column'])
        self.heatmap_norm.setCurrentText('linear')
        self.heatmap_norm.currentTextChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.heatmap_norm)
        
        # Contrast adjustment
        controls_layout.addWidget(QLabel("Contrast:"))
        self.heatmap_contrast = QSlider(Qt.Horizontal)
        self.heatmap_contrast.setRange(1, 20)  # 0.1 to 2.0 scaled by 10
        self.heatmap_contrast.setValue(10)  # 1.0
        self.heatmap_contrast.valueChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.heatmap_contrast)
        
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
        
        # Add visualization method selector
        method_frame = QFrame()
        method_layout = QHBoxLayout(method_frame)
        
        method_layout.addWidget(QLabel("Visualization Method:"))
        self.visualization_method_combo = QComboBox()
        self.visualization_method_combo.addItems(['PCA'])
        if UMAP_AVAILABLE:
            self.visualization_method_combo.addItem('UMAP')
        self.visualization_method_combo.setCurrentText('PCA')
        self.visualization_method_combo.currentTextChanged.connect(self.update_scatter_plot)
        method_layout.addWidget(self.visualization_method_combo)
        
        method_layout.addStretch()
        layout.addWidget(method_frame)
        
        # Create scatter figure
        self.viz_fig = Figure(figsize=(10, 6))
        self.viz_ax = self.viz_fig.add_subplot(111)
        self.viz_canvas = FigureCanvas(self.viz_fig)
        layout.addWidget(self.viz_canvas)
        
        # Add toolbar
        self.viz_toolbar = NavigationToolbar(self.viz_canvas, scatter_widget)
        layout.addWidget(self.viz_toolbar)
        
        self.viz_tab_widget.addTab(scatter_widget, "Scatter Plot")

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
            
            # Find common wavenumber range
            min_len = min(len(wn) for wn in all_wavenumbers)
            common_wavenumbers = all_wavenumbers[0][:min_len]
            
            # Interpolate all spectra to common wavenumber grid
            processed_intensities = []
            for wavenumbers, intensities in zip(all_wavenumbers, all_intensities):
                if len(intensities) != len(common_wavenumbers):
                    processed_intensities.append(intensities[:min_len])
                else:
                    processed_intensities.append(intensities)
            
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
            
            self.clustering_progress.setValue(30)
            QApplication.processEvents()
            
            # Extract features from processed intensities
            self.clustering_status.setText("Extracting features...")
            if preprocessing_method == 'Exclude Regions':
                # For exclusion method, extract_vibrational_features handles the exclusion
                features = self.extract_vibrational_features(processed_intensities, wavenumbers)
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
        
        # Show/hide NMF settings
        nmf_visible = method == 'NMF Separation'
        if hasattr(self, 'nmf_components_spinbox'):
            self.nmf_components_spinbox.parent().setVisible(nmf_visible)
        
        # Show/hide exclusion settings
        exclusion_visible = method == 'Exclude Regions'
        if hasattr(self, 'enable_exclusion_cb'):
            self.enable_exclusion_cb.parent().setVisible(exclusion_visible)
        if hasattr(self, 'exclusion_regions_frame'):
            self.exclusion_regions_frame.setVisible(exclusion_visible)
        
        # Auto-enable exclusion checkbox if method is selected
        if exclusion_visible and hasattr(self, 'enable_exclusion_cb'):
            self.enable_exclusion_cb.setChecked(True)
        
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
        """Update the heatmap visualization."""
        if (self.cluster_data['features_scaled'] is None or 
            self.cluster_data['labels'] is None):
            return
        
        try:
            self.heatmap_ax.clear()
            
            # Get parameters
            colormap = self.heatmap_colormap.currentText()
            norm_method = self.heatmap_norm.currentText()
            contrast = self.heatmap_contrast.value() / 10.0  # Scale from slider
            
            # Prepare data
            features = self.cluster_data['features_scaled'].copy()
            labels = self.cluster_data['labels']
            
            # Sort by cluster labels
            sort_indices = np.argsort(labels)
            sorted_features = features[sort_indices]
            sorted_labels = labels[sort_indices]
            
            # Apply normalization
            if norm_method == 'row':
                sorted_features = (sorted_features.T / np.max(np.abs(sorted_features), axis=1)).T
            elif norm_method == 'column':
                sorted_features = sorted_features / np.max(np.abs(sorted_features), axis=0)
            elif norm_method == 'log':
                sorted_features = np.log1p(np.abs(sorted_features)) * np.sign(sorted_features)
            elif norm_method == 'sqrt':
                sorted_features = np.sqrt(np.abs(sorted_features)) * np.sign(sorted_features)
            
            # Apply contrast
            sorted_features = sorted_features * contrast
            
            # Create heatmap
            im = self.heatmap_ax.imshow(
                sorted_features,
                                      cmap=colormap, 
                aspect='auto',
                interpolation='nearest'
            )
            
            # Add colorbar
            if hasattr(self, '_heatmap_colorbar'):
                self._heatmap_colorbar.remove()
            self._heatmap_colorbar = self.heatmap_fig.colorbar(im, ax=self.heatmap_ax)
            
            # Add cluster boundaries
            cluster_boundaries = []
            current_cluster = sorted_labels[0]
            for i, cluster in enumerate(sorted_labels):
                if cluster != current_cluster:
                    cluster_boundaries.append(i - 0.5)
                    current_cluster = cluster
            
            for boundary in cluster_boundaries:
                self.heatmap_ax.axhline(y=boundary, color='white', linewidth=2)
            
            self.heatmap_ax.set_title('Cluster Feature Heatmap')
            self.heatmap_ax.set_xlabel('Features')
            self.heatmap_ax.set_ylabel('Spectra (sorted by cluster)')
            
            self.heatmap_fig.tight_layout()
            self.heatmap_canvas.draw()
            
        except Exception as e:
            print(f"Error updating heatmap: {str(e)}")

    def update_scatter_plot(self):
        """Update the scatter plot visualization."""
        if (self.cluster_data['features_scaled'] is None or 
            self.cluster_data['labels'] is None):
            return
        
        try:
            self.viz_ax.clear()
            
            # Get visualization method
            method = self.visualization_method_combo.currentText()
            
            features = self.cluster_data['features_scaled']
            labels = self.cluster_data['labels']
            
            if method == 'PCA':
                # Perform PCA
                pca = PCA(n_components=2)
                coords = pca.fit_transform(features)
                
                # Store PCA results
                self.cluster_data['pca_coords'] = coords
                self.cluster_data['pca_model'] = pca
                
                xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'
                ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
                title = 'PCA Visualization of Clusters'
                
            elif method == 'UMAP' and UMAP_AVAILABLE:
                # Perform UMAP
                umap_model = umap.UMAP(n_components=2, random_state=42)
                coords = umap_model.fit_transform(features)
                
                # Store UMAP results
                self.cluster_data['umap_coords'] = coords
                self.cluster_data['umap_model'] = umap_model
                
                xlabel = 'UMAP 1'
                ylabel = 'UMAP 2'
                title = 'UMAP Visualization of Clusters'
                
            else:
                # Fallback to first two features
                coords = features[:, :2]
                xlabel = 'Feature 1'
                ylabel = 'Feature 2'
                title = 'Feature Space Visualization'
            
            # Create scatter plot
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                self.viz_ax.scatter(
                    coords[mask, 0], 
                    coords[mask, 1], 
                    c=[colors[i]], 
                    label=f'Cluster {label}',
                                        alpha=0.7,
                    s=50
                )
            
            self.viz_ax.set_xlabel(xlabel)
            self.viz_ax.set_ylabel(ylabel)
            self.viz_ax.set_title(title)
            self.viz_ax.legend()
            self.viz_ax.grid(True, alpha=0.3)
            
            self.viz_fig.tight_layout()
            self.viz_canvas.draw()
            
        except Exception as e:
            print(f"Error updating scatter plot: {str(e)}")

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
            
            self.analysis_results_text.setText(results_text)
            
        except Exception as e:
            print(f"Error updating analysis results: {str(e)}")

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
        """Update the refinement visualization plot."""
        try:
            if (self.cluster_data['features_scaled'] is None or 
                self.cluster_data['labels'] is None):
                return
        
            self.refinement_ax.clear()
            
            # Use PCA for 2D visualization
            features = self.cluster_data['features_scaled']
            labels = self.cluster_data['labels']
            
            pca = PCA(n_components=2)
            coords = pca.fit_transform(features)
            
            # Plot clusters with different colors
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                color = colors[i]
                
                # Highlight selected clusters
                if label in self.selected_points:
                    self.refinement_ax.scatter(
                        coords[mask, 0], coords[mask, 1], 
                        c=[color], label=f'Cluster {label} (selected)',
                        alpha=0.8, s=100, edgecolors='red', linewidths=2
                    )
                else:
                    self.refinement_ax.scatter(
                        coords[mask, 0], coords[mask, 1], 
                        c=[color], label=f'Cluster {label}',
                        alpha=0.7, s=50
                    )
            
            self.refinement_ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            self.refinement_ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            self.refinement_ax.set_title('Cluster Refinement View')
            self.refinement_ax.legend()
            self.refinement_ax.grid(True, alpha=0.3)
            
            self.refinement_fig.tight_layout()
            self.refinement_canvas.draw()
                
        except Exception as e:
            print(f"Error updating refinement plot: {str(e)}")

    # ... existing code ...

    # Advanced Analysis Methods
    def analyze_time_series_progression(self):
        """Analyze temporal cluster ordering and progression pathways."""
        QMessageBox.information(self, "Time-Series Analysis", 
                              "Time-series progression analysis functionality will be implemented.\n\n"
                              "This will analyze temporal cluster ordering and identify progression pathways.")

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
            
            for label in unique_labels:
                cluster_mask = labels == label
                centroid = np.mean(features[cluster_mask], axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            # Calculate pairwise distances between centroids
            distances = pdist(centroids, metric='euclidean')
            distance_matrix = squareform(distances)
            
            # Display results
            results_text = "Inter-cluster Distances:\n" + "="*30 + "\n\n"
            
            for i, label_i in enumerate(unique_labels):
                for j, label_j in enumerate(unique_labels):
                    if i < j:  # Only show upper triangle
                        dist = distance_matrix[i, j]
                        results_text += f"Cluster {label_i} ↔ Cluster {label_j}: {dist:.3f}\n"
            
            self.time_series_results.setText(results_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Failed to calculate distances:\n{str(e)}")

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
        QMessageBox.information(self, "Kinetic Modeling", 
                              "Kinetic modeling functionality will be implemented.\n\n"
                              "This will fit cluster populations to kinetic models for rate constants.")

    def compare_kinetic_models(self):
        """Compare different kinetic models for best fit."""
        QMessageBox.information(self, "Model Comparison", 
                              "Kinetic model comparison functionality will be implemented.\n\n"
                              "This will compare different kinetic models and provide statistics.")

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

    # Validation Methods
    def calculate_silhouette_analysis(self):
        """Calculate silhouette analysis for cluster validation."""
        QMessageBox.information(self, "Silhouette Analysis", 
                              "Silhouette analysis functionality will be implemented.\n\n"
                              "This will calculate silhouette scores for cluster validation.")

    def analyze_cluster_transitions(self):
        """Analyze cluster transition boundaries."""
        QMessageBox.information(self, "Cluster Transitions", 
                              "Cluster transition analysis functionality will be implemented.\n\n"
                              "This will analyze cluster boundaries and transition regions.")

    def test_cluster_stability(self):
        """Test cluster stability through bootstrap analysis."""
        QMessageBox.information(self, "Stability Analysis", 
                              "Cluster stability analysis functionality will be implemented.\n\n"
                              "This will test cluster stability using bootstrap methods.")

    def run_complete_validation(self):
        """Run complete validation suite."""
        QMessageBox.information(self, "Complete Validation", 
                              "Complete validation suite functionality will be implemented.\n\n"
                              "This will run all validation methods and provide comprehensive results.")

    # Advanced Statistics Methods
    def calculate_feature_importance(self):
        """Calculate feature importance for cluster discrimination."""
        QMessageBox.information(self, "Feature Importance", 
                              "Feature importance analysis functionality will be implemented.\n\n"
                              "This will calculate feature importance using the selected method.")

    def perform_discriminant_analysis(self):
        """Perform discriminant analysis on clusters."""
        QMessageBox.information(self, "Discriminant Analysis", 
                              "Discriminant analysis functionality will be implemented.\n\n"
                              "This will perform Linear Discriminant Analysis on the clusters.")

    def test_statistical_significance(self):
        """Test statistical significance of cluster differences."""
        QMessageBox.information(self, "Statistical Significance", 
                              "Statistical significance testing functionality will be implemented.\n\n"
                              "This will test significance using the selected method.")

    def run_comprehensive_statistics(self):
        """Run comprehensive statistical analysis."""
        QMessageBox.information(self, "Comprehensive Statistics", 
                              "Comprehensive statistical analysis functionality will be implemented.\n\n"
                              "This will run all statistical analyses and provide detailed results.")

    # Database Import Dialog placeholder class
    def open_database_import_dialog(self):
        """Open database import dialog - simplified version for now."""
        QMessageBox.information(self, "Database Import", 
                              "Database import dialog functionality will be enhanced.\n\n"
                              "For now, please use the folder import or main app import options.")

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

    # ... existing code ...


def launch_cluster_analysis(parent, raman_app):
    """Launch the cluster analysis window."""
    cluster_window = RamanClusterAnalysisQt6(parent, raman_app)
    cluster_window.show()
    return cluster_window 