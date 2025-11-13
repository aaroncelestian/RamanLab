"""
Clustering Tab for RamanLab Cluster Analysis

This module contains the clustering tab UI and functionality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                              QFormLayout, QSpinBox, QComboBox, QLineEdit, 
                              QCheckBox, QFrame, QPushButton, QProgressBar, 
                              QLabel)


class ClusteringTab(QWidget):
    """Clustering tab for cluster analysis configuration and execution."""
    
    def __init__(self, parent_window):
        """Initialize the clustering tab."""
        super().__init__()
        self.parent_window = parent_window
        self.carbon_controls = []
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the clustering tab UI."""
        layout = QVBoxLayout(self)
        
        # Create controls frame
        controls_group = QGroupBox("Clustering Controls")
        controls_layout = QFormLayout(controls_group)
        
        # Add algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['K-Means', 'MiniBatchKMeans', 'Hierarchical', 'DBSCAN', 'GMM (Probabilistic)'])
        self.algorithm_combo.setCurrentText('MiniBatchKMeans')
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        self.algorithm_combo.setToolTip(
            "K-Means: Standard, good quality\n"
            "MiniBatchKMeans: Fast, best for large datasets (>10k)\n"
            "Hierarchical: Best quality, slow for large datasets\n"
            "DBSCAN: Density-based, finds arbitrary shapes\n"
            "GMM: Probabilistic, gives cluster probabilities"
        )
        controls_layout.addRow("Algorithm:", self.algorithm_combo)
        
        # Add number of clusters selection
        self.n_clusters_spinbox = QSpinBox()
        self.n_clusters_spinbox.setRange(2, 50)
        self.n_clusters_spinbox.setValue(5)
        controls_layout.addRow("Number of Clusters:", self.n_clusters_spinbox)
        
        # Add linkage method selection (for Hierarchical)
        self.linkage_method_combo = QComboBox()
        self.linkage_method_combo.addItems(['ward', 'complete', 'average', 'single'])
        self.linkage_method_combo.setCurrentText('ward')
        self.linkage_method_row = controls_layout.addRow("Linkage Method:", self.linkage_method_combo)
        
        # Add distance metric selection (for Hierarchical)
        self.distance_metric_combo = QComboBox()
        self.distance_metric_combo.addItems(['euclidean', 'cosine', 'correlation'])
        self.distance_metric_combo.setCurrentText('euclidean')
        self.distance_metric_row = controls_layout.addRow("Distance Metric:", self.distance_metric_combo)
        
        # Add DBSCAN parameters
        self.dbscan_eps = QSpinBox()
        self.dbscan_eps.setRange(1, 100)
        self.dbscan_eps.setValue(5)
        self.dbscan_eps.setPrefix("eps: ")
        self.dbscan_eps_row = controls_layout.addRow("DBSCAN Epsilon:", self.dbscan_eps)
        
        self.dbscan_min_samples = QSpinBox()
        self.dbscan_min_samples.setRange(2, 100)
        self.dbscan_min_samples.setValue(5)
        self.dbscan_min_samples_row = controls_layout.addRow("DBSCAN Min Samples:", self.dbscan_min_samples)
        
        # Initially hide algorithm-specific controls
        self.on_algorithm_changed('MiniBatchKMeans')
        
        layout.addWidget(controls_group)
        
        # Add preprocessing controls
        preprocessing_group = QGroupBox("Preprocessing")
        preprocessing_layout = QFormLayout(preprocessing_group)
        
        # Add phase separation method selection
        self.phase_method_combo = QComboBox()
        self.phase_method_combo.addItems(['None', 'Exclude Regions', 'Corundum Correction', 'NMF Separation', 'Carbon Soot Optimization'])
        self.phase_method_combo.setCurrentText('None')
        self.phase_method_combo.currentTextChanged.connect(self.parent_window.update_preprocessing_controls)
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
        self.enable_carbon_analysis_cb.stateChanged.connect(self.parent_window.update_carbon_controls)
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
        analyze_carbon_btn.clicked.connect(self.parent_window.print_carbon_feature_analysis)
        carbon_buttons.addWidget(analyze_carbon_btn)

        suggest_improvements_btn = QPushButton("Suggest Improvements")
        suggest_improvements_btn.clicked.connect(self.parent_window.suggest_clustering_improvements)
        carbon_buttons.addWidget(suggest_improvements_btn)
        
        # Add NMF info button
        nmf_info_btn = QPushButton("NMF Clustering Info")
        nmf_info_btn.clicked.connect(self.parent_window.show_nmf_clustering_info)
        carbon_buttons.addWidget(nmf_info_btn)

        carbon_params_layout.addRow(carbon_buttons)

        carbon_layout.addWidget(carbon_params_frame)
        layout.addWidget(carbon_group)

        # Initially hide carbon controls
        carbon_params_frame.setVisible(False)
        self.carbon_controls = [carbon_params_frame]
        
        # Add run button
        button_layout = QHBoxLayout()
        
        # Main clustering button
        run_clustering_btn = QPushButton("Run Clustering")
        run_clustering_btn.clicked.connect(self.parent_window.run_clustering)
        run_clustering_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        run_clustering_btn.setToolTip("Run clustering with selected algorithm")
        button_layout.addWidget(run_clustering_btn)
        
        layout.addLayout(button_layout)
        
        # Add GMM-specific visualization buttons (only shown when GMM is used)
        prob_vis_layout = QHBoxLayout()
        
        self.prob_viz_btn = QPushButton("Show Probability Heatmap")
        self.prob_viz_btn.setToolTip("Visualize cluster assignment probabilities (GMM only)")
        self.prob_viz_btn.clicked.connect(self.parent_window.plot_probability_heatmap)
        self.prob_viz_btn.setEnabled(False)
        self.prob_viz_btn.setVisible(False)  # Hidden until GMM is run
        prob_vis_layout.addWidget(self.prob_viz_btn)
        
        self.subtype_viz_btn = QPushButton("View Sub-type Hierarchies")
        self.subtype_viz_btn.setToolTip("View hierarchical sub-type structures (GMM only)")
        self.subtype_viz_btn.clicked.connect(lambda: self.parent_window.plot_dendrogram(None))
        self.subtype_viz_btn.setEnabled(False)
        self.subtype_viz_btn.setVisible(False)  # Hidden until GMM is run
        prob_vis_layout.addWidget(self.subtype_viz_btn)
        
        layout.addLayout(prob_vis_layout)
        
        # Add progress bar
        self.clustering_progress = QProgressBar()
        layout.addWidget(self.clustering_progress)
        
        # Add status label
        self.clustering_status = QLabel("")
        layout.addWidget(self.clustering_status)
        
        layout.addStretch()
    
    def on_algorithm_changed(self, algorithm):
        """Handle algorithm selection changes."""
        # Hide all algorithm-specific controls first
        self.linkage_method_combo.setVisible(False)
        self.distance_metric_combo.setVisible(False)
        self.dbscan_eps.setVisible(False)
        self.dbscan_min_samples.setVisible(False)
        
        # Show relevant controls based on algorithm
        if algorithm == 'Hierarchical':
            self.linkage_method_combo.setVisible(True)
            self.distance_metric_combo.setVisible(True)
            self.n_clusters_spinbox.setEnabled(True)
        elif algorithm == 'DBSCAN':
            self.dbscan_eps.setVisible(True)
            self.dbscan_min_samples.setVisible(True)
            self.n_clusters_spinbox.setEnabled(False)  # DBSCAN finds clusters automatically
        else:  # K-Means, MiniBatchKMeans, GMM
            self.n_clusters_spinbox.setEnabled(True)
    
    def get_clustering_controls(self):
        """Get all clustering control widgets."""
        return {
            'algorithm_combo': self.algorithm_combo,
            'n_clusters_spinbox': self.n_clusters_spinbox,
            'linkage_method_combo': self.linkage_method_combo,
            'distance_metric_combo': self.distance_metric_combo,
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples,
            'phase_method_combo': self.phase_method_combo,
            'exclusion_regions_edit': self.exclusion_regions_edit,
            'nmf_components_spinbox': self.nmf_components_spinbox,
            'enable_carbon_analysis_cb': self.enable_carbon_analysis_cb,
            'd_band_range_edit': self.d_band_range_edit,
            'g_band_range_edit': self.g_band_range_edit,
            'prob_viz_btn': self.prob_viz_btn,
            'subtype_viz_btn': self.subtype_viz_btn,
            'clustering_progress': self.clustering_progress,
            'clustering_status': self.clustering_status,
            'carbon_controls': self.carbon_controls
        }
    
    def update_carbon_controls_visibility(self, visible):
        """Update visibility of carbon-specific controls."""
        for control in self.carbon_controls:
            control.setVisible(visible)
