"""
Visualization Tabs for RamanLab Cluster Analysis

This module contains all visualization-related tabs and functionality.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, 
                              QPushButton, QLabel, QComboBox, QCheckBox, QSpinBox,
                              QTabWidget, QGridLayout, QDoubleSpinBox, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar


class VisualizationTab(QWidget):
    """Main visualization tab containing sub-tabs for different plot types."""
    
    def __init__(self, parent_window):
        """Initialize the visualization tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the visualization tab UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different visualizations
        self.viz_tab_widget = QTabWidget()
        layout.addWidget(self.viz_tab_widget)
        
        # Create individual visualization tabs
        self.dendrogram_tab = DendrogramTab(self.parent_window)
        self.heatmap_tab = HeatmapTab(self.parent_window)
        self.scatter_tab = ScatterTab(self.parent_window)
        
        # Add tabs to widget
        self.viz_tab_widget.addTab(self.dendrogram_tab, "Dendrogram")
        self.viz_tab_widget.addTab(self.heatmap_tab, "Heatmap")
        self.viz_tab_widget.addTab(self.scatter_tab, "Scatter Plot")
        
        # Add export button
        export_btn = QPushButton("Export Visualization")
        export_btn.clicked.connect(self.parent_window.export_visualization)
        layout.addWidget(export_btn)
    
    def get_viz_tab_widget(self):
        """Get the visualization tab widget."""
        return self.viz_tab_widget
    
    def get_dendrogram_tab(self):
        """Get the dendrogram tab."""
        return self.dendrogram_tab
    
    def get_heatmap_tab(self):
        """Get the heatmap tab."""
        return self.heatmap_tab
    
    def get_scatter_tab(self):
        """Get the scatter tab."""
        return self.scatter_tab


class DendrogramTab(QWidget):
    """Dendrogram visualization tab."""
    
    def __init__(self, parent_window):
        """Initialize the dendrogram tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the dendrogram tab UI."""
        layout = QVBoxLayout(self)
        
        # Add controls
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        
        # First row of controls
        row1 = QHBoxLayout()
        
        # Orientation control
        row1.addWidget(QLabel("Orientation:"))
        self.dendro_orientation = QComboBox()
        self.dendro_orientation.addItems(['top', 'bottom', 'left', 'right'])
        self.dendro_orientation.setCurrentText('top')
        self.dendro_orientation.currentTextChanged.connect(self.parent_window.update_dendrogram)
        row1.addWidget(self.dendro_orientation)
        
        # Max samples to show
        row1.addWidget(QLabel("Max Samples:"))
        self.dendro_max_samples = QSpinBox()
        self.dendro_max_samples.setRange(10, 1000)
        self.dendro_max_samples.setValue(100)
        self.dendro_max_samples.setSingleStep(50)
        self.dendro_max_samples.valueChanged.connect(self.parent_window.update_dendrogram)
        row1.addWidget(self.dendro_max_samples)
        
        # Truncation mode
        row1.addWidget(QLabel("Truncate:"))
        self.dendro_truncate = QComboBox()
        self.dendro_truncate.addItems(['None', 'lastp', 'level'])
        self.dendro_truncate.setCurrentText('lastp')
        self.dendro_truncate.setToolTip("lastp: Show only last p merged clusters\nlevel: Show only last p levels")
        self.dendro_truncate.currentTextChanged.connect(self.parent_window.update_dendrogram)
        row1.addWidget(self.dendro_truncate)
        
        # Truncation parameter
        row1.addWidget(QLabel("p:"))
        self.dendro_p = QSpinBox()
        self.dendro_p.setRange(5, 100)
        self.dendro_p.setValue(30)
        self.dendro_p.setToolTip("Number of clusters/levels to show when truncating")
        self.dendro_p.valueChanged.connect(self.parent_window.update_dendrogram)
        row1.addWidget(self.dendro_p)
        
        row1.addStretch()
        controls_layout.addLayout(row1)
        
        # Second row of controls
        row2 = QHBoxLayout()
        
        # Label display option
        self.dendro_show_labels = QCheckBox("Show Labels")
        self.dendro_show_labels.setChecked(False)  # Default off for large datasets
        self.dendro_show_labels.toggled.connect(self.parent_window.update_dendrogram)
        row2.addWidget(self.dendro_show_labels)
        
        # Color threshold
        row2.addWidget(QLabel("Color Threshold:"))
        self.dendro_color_threshold = QDoubleSpinBox()
        self.dendro_color_threshold.setRange(0, 1000)
        self.dendro_color_threshold.setValue(0)
        self.dendro_color_threshold.setSingleStep(10)
        self.dendro_color_threshold.setDecimals(1)
        self.dendro_color_threshold.setSpecialValueText("Auto")
        self.dendro_color_threshold.setToolTip("Distance threshold for coloring clusters (0 = auto)")
        self.dendro_color_threshold.valueChanged.connect(self.parent_window.update_dendrogram)
        row2.addWidget(self.dendro_color_threshold)
        
        # Update button
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.parent_window.update_dendrogram)
        row2.addWidget(update_btn)
        
        # Info label
        self.dendro_info = QLabel("")
        self.dendro_info.setStyleSheet("color: #666; font-style: italic;")
        row2.addWidget(self.dendro_info)
        
        row2.addStretch()
        controls_layout.addLayout(row2)
        
        layout.addWidget(controls_frame)
        
        # Create dendrogram figure
        self.dendro_fig = Figure(figsize=(10, 6))
        self.dendrogram_ax = self.dendro_fig.add_subplot(111)
        self.dendro_canvas = FigureCanvas(self.dendro_fig)
        self.dendro_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.dendro_canvas)
        
        # Add toolbar
        self.dendro_toolbar = NavigationToolbar(self.dendro_canvas, self)
        layout.addWidget(self.dendro_toolbar)
    
    def get_dendrogram_controls(self):
        """Get dendrogram control widgets."""
        return {
            'orientation': self.dendro_orientation,
            'max_samples': self.dendro_max_samples,
            'show_labels': self.dendro_show_labels,
            'truncate_mode': self.dendro_truncate,
            'truncate_p': self.dendro_p,
            'color_threshold': self.dendro_color_threshold,
            'info_label': self.dendro_info,
            'figure': self.dendro_fig,
            'axis': self.dendrogram_ax,
            'canvas': self.dendro_canvas,
            'toolbar': self.dendro_toolbar
        }


class HeatmapTab(QWidget):
    """Heatmap visualization tab."""
    
    def __init__(self, parent_window):
        """Initialize the heatmap tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the heatmap tab UI."""
        layout = QVBoxLayout(self)
        
        # Add controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Color map control
        controls_layout.addWidget(QLabel("Colormap:"))
        self.heatmap_colormap = QComboBox()
        self.heatmap_colormap.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool'])
        self.heatmap_colormap.setCurrentText('viridis')
        self.heatmap_colormap.currentTextChanged.connect(self.parent_window.update_heatmap)
        controls_layout.addWidget(self.heatmap_colormap)
        
        # Sort by cluster option
        self.heatmap_sort_by_cluster = QCheckBox("Sort by Cluster")
        self.heatmap_sort_by_cluster.setChecked(True)
        self.heatmap_sort_by_cluster.toggled.connect(self.parent_window.update_heatmap)
        controls_layout.addWidget(self.heatmap_sort_by_cluster)
        
        # Update button
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.parent_window.update_heatmap)
        controls_layout.addWidget(update_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_frame)
        
        # Create heatmap figure
        self.heatmap_fig = Figure(figsize=(10, 8))
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmap_canvas = FigureCanvas(self.heatmap_fig)
        layout.addWidget(self.heatmap_canvas)
        
        # Add toolbar
        self.heatmap_toolbar = NavigationToolbar(self.heatmap_canvas, self)
        layout.addWidget(self.heatmap_toolbar)
    
    def get_heatmap_controls(self):
        """Get heatmap control widgets."""
        return {
            'colormap': self.heatmap_colormap,
            'sort_by_cluster': self.heatmap_sort_by_cluster,
            'figure': self.heatmap_fig,
            'axis': self.heatmap_ax,
            'canvas': self.heatmap_canvas,
            'toolbar': self.heatmap_toolbar
        }


class ScatterTab(QWidget):
    """Scatter plot visualization tab."""
    
    def __init__(self, parent_window):
        """Initialize the scatter tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the scatter tab UI."""
        layout = QVBoxLayout(self)
        
        # Add controls
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # Visualization method
        controls_layout.addWidget(QLabel("Method:"))
        self.scatter_method = QComboBox()
        self.scatter_method.addItems(['PCA', 't-SNE', 'UMAP', 'MDS', 'Spectral Embedding'])
        self.scatter_method.setCurrentText('PCA')
        self.scatter_method.currentTextChanged.connect(self.parent_window.update_scatter_plot)
        controls_layout.addWidget(self.scatter_method)
        
        # Point size control
        controls_layout.addWidget(QLabel("Point Size:"))
        self.scatter_point_size = QSpinBox()
        self.scatter_point_size.setRange(10, 200)
        self.scatter_point_size.setValue(50)
        self.scatter_point_size.valueChanged.connect(self.parent_window.update_scatter_plot)
        controls_layout.addWidget(self.scatter_point_size)
        
        # Show legend option
        self.scatter_show_legend = QCheckBox("Show Legend")
        self.scatter_show_legend.setChecked(True)
        self.scatter_show_legend.toggled.connect(self.parent_window.update_scatter_plot)
        controls_layout.addWidget(self.scatter_show_legend)
        
        controls_layout.addStretch()
        layout.addWidget(controls_frame)
        
        # UMAP parameters frame (initially hidden)
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
        
        # epochs parameter
        umap_params_layout.addWidget(QLabel("Epochs:"), 1, 4)
        self.umap_epochs = QComboBox()
        self.umap_epochs.addItems(['Auto (50 for large datasets)', '20 (Very Fast)', '50 (Fast)', '100 (Balanced)', '200 (High Quality)', '500 (Publication)'])
        self.umap_epochs.setCurrentText('Auto (50 for large datasets)')
        self.umap_epochs.setToolTip("Number of optimization iterations\n20-50: Fast exploration\n100-200: Better quality\n500+: Publication-ready")
        umap_params_layout.addWidget(self.umap_epochs, 1, 5)
        
        # UMAP presets
        umap_params_layout.addWidget(QLabel("UMAP Presets:"), 2, 0)
        self.umap_preset_combo = QComboBox()
        self.umap_preset_combo.addItems([
            'Custom',
            '⚡ Fast Mode (Large Datasets)',
            'General Spectroscopy',
            'Tight Clustering',
            'Broad Clusters', 
            'Manifold Structure',
            'High Noise Data'
        ])
        self.umap_preset_combo.currentTextChanged.connect(self.apply_umap_preset)
        self.umap_preset_combo.setToolTip("Apply optimized UMAP parameters for different data types\n⚡ Fast Mode: Optimized for 10k+ spectra (10-20x faster)")
        umap_params_layout.addWidget(self.umap_preset_combo, 2, 1, 1, 3)
        
        # Update button
        self.update_umap_btn = QPushButton("Update UMAP")
        self.update_umap_btn.clicked.connect(self.parent_window.update_scatter_plot)
        umap_params_layout.addWidget(self.update_umap_btn, 3, 0, 1, 4)
        
        # Initially hide UMAP parameters
        self.umap_params_frame.setVisible(False)
        layout.addWidget(self.umap_params_frame)
        
        # Create scatter plot figure
        self.scatter_fig = Figure(figsize=(10, 8))
        self.scatter_ax = self.scatter_fig.add_subplot(111)
        self.scatter_canvas = FigureCanvas(self.scatter_fig)
        layout.addWidget(self.scatter_canvas)
        
        # Add toolbar
        self.scatter_toolbar = NavigationToolbar(self.scatter_canvas, self)
        layout.addWidget(self.scatter_toolbar)
        
        # Add export buttons below the plot
        export_buttons_frame = QFrame()
        export_buttons_layout = QHBoxLayout(export_buttons_frame)
        
        # Export to folders button
        self.export_folders_btn = QPushButton("Export to Folders")
        self.export_folders_btn.setToolTip("Export each cluster's spectra to separate folders")
        self.export_folders_btn.setEnabled(False)
        self.export_folders_btn.clicked.connect(self.parent_window.export_clusters_to_folders)
        export_buttons_layout.addWidget(self.export_folders_btn)
        
        # Export summed spectra button
        self.export_summed_btn = QPushButton("Export Summed Spectra")
        self.export_summed_btn.setToolTip("Export summed/averaged spectra for each cluster as a single plot")
        self.export_summed_btn.setEnabled(False)
        self.export_summed_btn.clicked.connect(self.parent_window.export_summed_cluster_spectra)
        export_buttons_layout.addWidget(self.export_summed_btn)
        
        # Export cluster overview button
        self.export_overview_btn = QPushButton("Export Cluster Overview")
        self.export_overview_btn.setToolTip("Export a comprehensive overview with all clusters in a grid layout")
        self.export_overview_btn.setEnabled(False)
        self.export_overview_btn.clicked.connect(self.parent_window.export_cluster_overview)
        export_buttons_layout.addWidget(self.export_overview_btn)
        
        # Export XY plot data button
        self.export_xy_btn = QPushButton("Export XY Plot Data")
        self.export_xy_btn.setToolTip("Export the XY coordinates of the current scatter plot (PCA or UMAP)")
        self.export_xy_btn.setEnabled(False)
        self.export_xy_btn.clicked.connect(self.parent_window.export_xy_plot_data)
        export_buttons_layout.addWidget(self.export_xy_btn)
        
        export_buttons_layout.addStretch()
        layout.addWidget(export_buttons_frame)
    
    def apply_umap_preset(self):
        """Apply UMAP parameter presets optimized for different data types."""
        preset = self.umap_preset_combo.currentText()
        
        if preset == '⚡ Fast Mode (Large Datasets)':
            # Optimized for speed AND quality on large datasets (10k+ spectra)
            self.umap_n_neighbors.setValue(15)  # Good balance
            self.umap_min_dist.setValue(0.1)
            self.umap_metric.setCurrentText('euclidean')  # Better for Raman features
            self.umap_spread.setValue(1.0)
            # Note: n_epochs=50 and other optimizations set automatically in main.py
            
        elif preset == 'Tight Clustering':
            self.umap_n_neighbors.setValue(5)
            self.umap_min_dist.setValue(0.01)
            self.umap_metric.setCurrentText('cosine')
            self.umap_spread.setValue(0.5)
            
        elif preset == 'Broad Clusters':
            self.umap_n_neighbors.setValue(15)
            self.umap_min_dist.setValue(0.1)
            self.umap_metric.setCurrentText('euclidean')
            self.umap_spread.setValue(1.0)
            
        elif preset == 'Manifold Structure':
            self.umap_n_neighbors.setValue(30)
            self.umap_min_dist.setValue(0.3)
            self.umap_metric.setCurrentText('correlation')
            self.umap_spread.setValue(2.0)
            
        elif preset == 'General Spectroscopy':
            self.umap_n_neighbors.setValue(15)
            self.umap_min_dist.setValue(0.1)
            self.umap_metric.setCurrentText('euclidean')
            self.umap_spread.setValue(1.0)
            
        elif preset == 'High Noise Data':
            self.umap_n_neighbors.setValue(50)
            self.umap_min_dist.setValue(0.5)
            self.umap_metric.setCurrentText('manhattan')
            self.umap_spread.setValue(2.0)

    def get_scatter_controls(self):
        """Get scatter plot control widgets."""
        return {
            'method': self.scatter_method,
            'point_size': self.scatter_point_size,
            'show_legend': self.scatter_show_legend,
            'figure': self.scatter_fig,
            'axis': self.scatter_ax,
            'canvas': self.scatter_canvas,
            'toolbar': self.scatter_toolbar,
            'umap_params_frame': self.umap_params_frame,
            'umap_n_neighbors': self.umap_n_neighbors,
            'umap_min_dist': self.umap_min_dist,
            'umap_metric': self.umap_metric,
            'umap_spread': self.umap_spread,
            'umap_epochs': self.umap_epochs,
            'umap_preset_combo': self.umap_preset_combo,
            'update_umap_btn': self.update_umap_btn,
            'export_folders_btn': self.export_folders_btn,
            'export_summed_btn': self.export_summed_btn,
            'export_overview_btn': self.export_overview_btn,
            'export_xy_btn': self.export_xy_btn
        }
