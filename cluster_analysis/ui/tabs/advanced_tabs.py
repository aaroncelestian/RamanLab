"""
Advanced Analysis Tabs for RamanLab Cluster Analysis

This module contains advanced analysis tabs including time-series, kinetics,
structural analysis, validation, and statistical analysis.
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                              QLabel, QPushButton, QSpinBox, QComboBox, QTextEdit,
                              QFrame)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar


class TimeSeriesTab(QWidget):
    """Time-series progression analysis tab."""
    
    def __init__(self, parent_window):
        """Initialize the time-series tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the time-series tab UI."""
        layout = QVBoxLayout(self)
        
        # Title and description
        title_label = QLabel("Time-Series Progression Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Controls
        controls_group = QGroupBox("Time-Series Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Time column selection
        time_frame = QFrame()
        time_layout = QHBoxLayout(time_frame)
        time_layout.addWidget(QLabel("Time Column:"))
        self.time_column_combo = QComboBox()
        self.time_column_combo.addItems(['Auto-detect', 'Column 1', 'Column 2', 'Column 3'])
        time_layout.addWidget(self.time_column_combo)
        time_layout.addStretch()
        controls_layout.addWidget(time_frame)
        
        # Analysis buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        
        analyze_btn = QPushButton("Analyze Progression")
        analyze_btn.clicked.connect(self.parent_window.analyze_time_series)
        button_layout.addWidget(analyze_btn)
        
        plot_btn = QPushButton("Plot Time Series")
        plot_btn.clicked.connect(self.parent_window.plot_time_series)
        button_layout.addWidget(plot_btn)
        
        button_layout.addStretch()
        controls_layout.addWidget(button_frame)
        
        layout.addWidget(controls_group)
        
        # Results display
        self.time_series_fig = Figure(figsize=(10, 6))
        self.time_series_ax = self.time_series_fig.add_subplot(111)
        self.time_series_canvas = FigureCanvas(self.time_series_fig)
        layout.addWidget(self.time_series_canvas)
        
        # Toolbar
        self.time_series_toolbar = NavigationToolbar(self.time_series_canvas, self)
        layout.addWidget(self.time_series_toolbar)


class KineticsTab(QWidget):
    """Kinetics modeling tab."""
    
    def __init__(self, parent_window):
        """Initialize the kinetics tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the kinetics tab UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Kinetics Modeling")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Controls
        controls_group = QGroupBox("Kinetics Parameters")
        controls_layout = QVBoxLayout(controls_group)
        
        # Model selection
        model_frame = QFrame()
        model_layout = QHBoxLayout(model_frame)
        model_layout.addWidget(QLabel("Model:"))
        self.kinetics_model_combo = QComboBox()
        self.kinetics_model_combo.addItems(['First Order', 'Second Order', 'Diffusion', 'Avrami'])
        model_layout.addWidget(self.kinetics_model_combo)
        model_layout.addStretch()
        controls_layout.addWidget(model_frame)
        
        # Analysis buttons
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        
        fit_btn = QPushButton("Fit Kinetics")
        fit_btn.clicked.connect(self.parent_window.fit_kinetics_model)
        button_layout.addWidget(fit_btn)
        
        predict_btn = QPushButton("Predict Evolution")
        predict_btn.clicked.connect(self.parent_window.predict_kinetics)
        button_layout.addWidget(predict_btn)
        
        button_layout.addStretch()
        controls_layout.addWidget(button_frame)
        
        layout.addWidget(controls_group)
        
        # Results display
        self.kinetics_fig = Figure(figsize=(10, 6))
        self.kinetics_ax = self.kinetics_fig.add_subplot(111)
        self.kinetics_canvas = FigureCanvas(self.kinetics_fig)
        layout.addWidget(self.kinetics_canvas)
        
        # Toolbar
        self.kinetics_toolbar = NavigationToolbar(self.kinetics_canvas, self)
        layout.addWidget(self.kinetics_toolbar)


class StructuralAnalysisTab(QWidget):
    """Structural characterization tab."""
    
    def __init__(self, parent_window):
        """Initialize the structural analysis tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the structural analysis tab UI."""
        layout = QHBoxLayout(self)  # Changed to horizontal layout
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title_label = QLabel("Structural Characterization")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(title_label)
        
        # Analysis controls
        controls_group = QGroupBox("Analysis Options")
        controls_layout = QVBoxLayout(controls_group)
        
        # Peak analysis
        peak_btn = QPushButton("Analyze Peak Positions")
        peak_btn.clicked.connect(self.parent_window.analyze_peak_positions)
        controls_layout.addWidget(peak_btn)
        
        # Band ratios
        ratio_btn = QPushButton("Calculate Band Ratios")
        ratio_btn.clicked.connect(self.parent_window.calculate_band_ratios)
        controls_layout.addWidget(ratio_btn)
        
        # Structural parameters
        struct_btn = QPushButton("Extract Structural Parameters")
        struct_btn.clicked.connect(self.parent_window.extract_structural_parameters)
        controls_layout.addWidget(struct_btn)
        
        left_layout.addWidget(controls_group)
        
        # Results text
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.structural_results_text = QTextEdit()
        self.structural_results_text.setReadOnly(True)
        self.structural_results_text.setMaximumHeight(200)
        results_layout.addWidget(self.structural_results_text)
        
        left_layout.addWidget(results_group)
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.structural_fig = Figure(figsize=(8, 6))
        self.structural_ax = self.structural_fig.add_subplot(111)
        self.structural_canvas = FigureCanvas(self.structural_fig)
        right_layout.addWidget(self.structural_canvas)
        
        # Toolbar
        self.structural_toolbar = NavigationToolbar(self.structural_canvas, self)
        right_layout.addWidget(self.structural_toolbar)
        
        layout.addWidget(right_panel)


class ValidationTab(QWidget):
    """Quantitative cluster validation tab."""
    
    def __init__(self, parent_window):
        """Initialize the validation tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the validation tab UI."""
        layout = QHBoxLayout(self)  # Changed to horizontal layout
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title_label = QLabel("Cluster Validation")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(title_label)
        
        # Validation metrics
        metrics_group = QGroupBox("Validation Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # Silhouette analysis
        silhouette_btn = QPushButton("Calculate Silhouette Score")
        silhouette_btn.clicked.connect(self.parent_window.calculate_silhouette_score)
        metrics_layout.addWidget(silhouette_btn)
        
        # Davies-Bouldin index
        davies_btn = QPushButton("Davies-Bouldin Index")
        davies_btn.clicked.connect(self.parent_window.calculate_davies_bouldin)
        metrics_layout.addWidget(davies_btn)
        
        # Calinski-Harabasz index
        calinski_btn = QPushButton("Calinski-Harabasz Index")
        calinski_btn.clicked.connect(self.parent_window.calculate_calinski_harabasz)
        metrics_layout.addWidget(calinski_btn)
        
        left_layout.addWidget(metrics_group)
        
        # Optimal cluster analysis
        optimal_group = QGroupBox("Optimal Cluster Analysis")
        optimal_layout = QVBoxLayout(optimal_group)
        
        self.max_clusters_spinbox = QSpinBox()
        self.max_clusters_spinbox.setRange(2, 20)
        self.max_clusters_spinbox.setValue(10)
        optimal_layout.addWidget(QLabel("Max Clusters:"))
        optimal_layout.addWidget(self.max_clusters_spinbox)
        
        optimal_btn = QPushButton("Find Optimal Clusters")
        optimal_btn.clicked.connect(self.parent_window.find_optimal_clusters)
        optimal_layout.addWidget(optimal_btn)
        
        left_layout.addWidget(optimal_group)
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results text
        self.validation_results_text = QTextEdit()
        self.validation_results_text.setReadOnly(True)
        right_layout.addWidget(self.validation_results_text)
        
        # Visualization
        self.validation_fig = Figure(figsize=(8, 4))
        self.validation_ax = self.validation_fig.add_subplot(111)
        self.validation_canvas = FigureCanvas(self.validation_fig)
        right_layout.addWidget(self.validation_canvas)
        
        layout.addWidget(right_panel)


class AdvancedStatisticsTab(QWidget):
    """Advanced statistical analysis tab."""
    
    def __init__(self, parent_window):
        """Initialize the advanced statistics tab."""
        super().__init__()
        self.parent_window = parent_window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the advanced statistics tab UI."""
        layout = QHBoxLayout(self)  # Changed to horizontal layout
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title_label = QLabel("Advanced Statistical Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        left_layout.addWidget(title_label)
        
        # Statistical tests
        tests_group = QGroupBox("Statistical Tests")
        tests_layout = QVBoxLayout(tests_group)
        
        # ANOVA
        anova_btn = QPushButton("ANOVA Test")
        anova_btn.clicked.connect(self.parent_window.perform_anova_test)
        tests_layout.addWidget(anova_btn)
        
        # Kruskal-Wallis
        kruskal_btn = QPushButton("Kruskal-Wallis Test")
        kruskal_btn.clicked.connect(self.parent_window.perform_kruskal_test)
        tests_layout.addWidget(kruskal_btn)
        
        # MANOVA
        manova_btn = QPushButton("MANOVA Test")
        manova_btn.clicked.connect(self.parent_window.perform_manova_test)
        tests_layout.addWidget(manova_btn)
        
        left_layout.addWidget(tests_group)
        
        # Multivariate analysis
        multivariate_group = QGroupBox("Multivariate Analysis")
        multivariate_layout = QVBoxLayout(multivariate_group)
        
        # PCA analysis
        pca_btn = QPushButton("Detailed PCA Analysis")
        pca_btn.clicked.connect(self.parent_window.perform_detailed_pca)
        multivariate_layout.addWidget(pca_btn)
        
        # Factor analysis
        factor_btn = QPushButton("Factor Analysis")
        factor_btn.clicked.connect(self.parent_window.perform_factor_analysis)
        multivariate_layout.addWidget(factor_btn)
        
        left_layout.addWidget(multivariate_group)
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results display
        self.stats_results_text = QTextEdit()
        self.stats_results_text.setReadOnly(True)
        right_layout.addWidget(self.stats_results_text)
        
        # Visualization
        self.stats_fig = Figure(figsize=(8, 4))
        self.stats_ax = self.stats_fig.add_subplot(111)
        self.stats_canvas = FigureCanvas(self.stats_fig)
        right_layout.addWidget(self.stats_canvas)
        
        layout.addWidget(right_panel)
