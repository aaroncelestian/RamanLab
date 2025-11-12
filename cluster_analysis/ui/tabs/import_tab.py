"""
Import Tab for RamanLab Cluster Analysis

This module contains the import tab UI and functionality.
"""

import multiprocessing
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                              QLabel, QPushButton, QProgressBar, QCheckBox, 
                              QSpinBox, QFormLayout)


class ImportTab(QWidget):
    """Import tab for cluster analysis data import."""
    
    def __init__(self, parent_window):
        """Initialize the import tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the import tab UI."""
        layout = QVBoxLayout(self)
        
        # Create folder selection frame
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout(folder_group)
        
        # Add folder path display
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_path_label)
        
        # Add select folder button
        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.parent_window.select_import_folder)
        folder_layout.addWidget(select_folder_btn)
        
        # Add import single file button (for single-file 2D maps)
        import_single_file_btn = QPushButton("Import Single File (2D Map)")
        import_single_file_btn.clicked.connect(self.parent_window.import_single_file_map)
        import_single_file_btn.setToolTip("Import a single file containing all spectra with X,Y positions")
        folder_layout.addWidget(import_single_file_btn)
        
        # Add import from database button
        import_db_btn = QPushButton("Import from Database")
        import_db_btn.clicked.connect(self.parent_window.open_database_import_dialog)
        folder_layout.addWidget(import_db_btn)
        
        # Add import from main app button
        import_main_btn = QPushButton("Import from Main App")
        import_main_btn.clicked.connect(self.parent_window.import_from_main_app)
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
        
        # Performance optimization settings
        perf_group = QGroupBox("Performance Optimization (for large datasets)")
        perf_layout = QVBoxLayout(perf_group)
        
        perf_info = QLabel("⚡ Recommended for datasets with >10,000 spectra\n"
                          "💡 Especially beneficial for UMAP analysis (10-100x faster)")
        perf_info.setStyleSheet("color: #d97706; font-weight: bold; padding: 4px;")
        perf_info.setWordWrap(True)
        perf_layout.addWidget(perf_info)
        
        perf_form = QFormLayout()
        
        # Spectral downsampling
        self.enable_spectral_downsample = QCheckBox("Enable spectral downsampling")
        self.enable_spectral_downsample.setToolTip("Reduce number of wavenumber points while preserving spectral features")
        self.enable_spectral_downsample.stateChanged.connect(self.parent_window.update_performance_controls)
        perf_form.addRow(self.enable_spectral_downsample)
        
        self.downsample_factor = QSpinBox()
        self.downsample_factor.setRange(2, 10)
        self.downsample_factor.setValue(3)
        self.downsample_factor.setSuffix("x reduction")
        self.downsample_factor.setToolTip("Factor by which to reduce spectral resolution (3x = keep every 3rd point)")
        self.downsample_factor.setEnabled(False)
        perf_form.addRow("  Downsample Factor:", self.downsample_factor)
        
        # PCA pre-reduction
        self.enable_pca_prereduction = QCheckBox("Enable PCA pre-reduction")
        self.enable_pca_prereduction.setToolTip("Reduce dimensionality before clustering (faster, preserves variance)")
        self.enable_pca_prereduction.stateChanged.connect(self.parent_window.update_performance_controls)
        perf_form.addRow(self.enable_pca_prereduction)
        
        self.pca_prereduction_components = QSpinBox()
        self.pca_prereduction_components.setRange(10, 500)
        self.pca_prereduction_components.setValue(100)
        self.pca_prereduction_components.setSuffix(" components")
        self.pca_prereduction_components.setToolTip("Number of PCA components to retain (captures most variance)")
        self.pca_prereduction_components.setEnabled(False)
        perf_form.addRow("  PCA Components:", self.pca_prereduction_components)
        
        # Random subset sampling
        self.enable_subset_sampling = QCheckBox("Enable subset sampling (exploration mode)")
        self.enable_subset_sampling.setToolTip("Randomly sample subset of spectra for initial exploration")
        self.enable_subset_sampling.stateChanged.connect(self.parent_window.update_performance_controls)
        perf_form.addRow(self.enable_subset_sampling)
        
        self.subset_sample_size = QSpinBox()
        self.subset_sample_size.setRange(1000, 50000)
        self.subset_sample_size.setValue(10000)
        self.subset_sample_size.setSingleStep(1000)
        self.subset_sample_size.setSuffix(" spectra")
        self.subset_sample_size.setToolTip("Number of spectra to randomly sample")
        self.subset_sample_size.setEnabled(False)
        perf_form.addRow("  Sample Size:", self.subset_sample_size)
        
        perf_layout.addLayout(perf_form)
        
        # Algorithm auto-selection info
        n_cores = multiprocessing.cpu_count()
        algo_info = QLabel(f"🤖 Clustering Algorithm Auto-Selection ({n_cores} CPU cores):\n"
                          "• <1,000 spectra: Hierarchical clustering (dendrogram available)\n"
                          "• 1,000-5,000 spectra: KMeans (fast, accurate, multi-core)\n"
                          "• >5,000 spectra: MiniBatchKMeans (optimized for large datasets, multi-core)")
        algo_info.setStyleSheet("color: #0891b2; font-size: 10px; padding: 4px; background-color: #ecfeff; border-radius: 3px;")
        algo_info.setWordWrap(True)
        perf_layout.addWidget(algo_info)
        
        # Performance estimate
        self.perf_estimate_label = QLabel("")
        self.perf_estimate_label.setStyleSheet("color: #059669; font-size: 10px; padding: 4px; font-style: italic;")
        self.perf_estimate_label.setWordWrap(True)
        perf_layout.addWidget(self.perf_estimate_label)
        
        layout.addWidget(perf_group)
        
        # Add start import button
        start_import_btn = QPushButton("Start Import")
        start_import_btn.clicked.connect(self.parent_window.start_batch_import)
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
        append_btn.clicked.connect(self.parent_window.append_data)
        append_layout.addWidget(append_btn)
        
        layout.addWidget(append_group)
        
        layout.addStretch()
    
    def get_folder_path_label(self):
        """Get the folder path label widget."""
        return self.folder_path_label
    
    def get_import_progress(self):
        """Get the import progress bar widget."""
        return self.import_progress
    
    def get_import_status(self):
        """Get the import status label widget."""
        return self.import_status
    
    def get_performance_controls(self):
        """Get all performance optimization controls."""
        return {
            'enable_spectral_downsample': self.enable_spectral_downsample,
            'downsample_factor': self.downsample_factor,
            'enable_pca_prereduction': self.enable_pca_prereduction,
            'pca_prereduction_components': self.pca_prereduction_components,
            'enable_subset_sampling': self.enable_subset_sampling,
            'subset_sample_size': self.subset_sample_size,
            'perf_estimate_label': self.perf_estimate_label
        }
