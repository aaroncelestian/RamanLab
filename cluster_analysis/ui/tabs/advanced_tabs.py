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
        silhouette_row = QHBoxLayout()
        silhouette_btn = QPushButton("Calculate Silhouette Score")
        silhouette_btn.clicked.connect(self.parent_window.calculate_silhouette_score)
        silhouette_row.addWidget(silhouette_btn)
        
        silhouette_help_btn = QPushButton("?")
        silhouette_help_btn.setMaximumWidth(30)
        silhouette_help_btn.setStyleSheet("font-weight: bold;")
        silhouette_help_btn.clicked.connect(self.show_silhouette_help)
        silhouette_row.addWidget(silhouette_help_btn)
        metrics_layout.addLayout(silhouette_row)
        
        # Davies-Bouldin index
        davies_row = QHBoxLayout()
        davies_btn = QPushButton("Davies-Bouldin Index")
        davies_btn.clicked.connect(self.parent_window.calculate_davies_bouldin)
        davies_row.addWidget(davies_btn)
        
        davies_help_btn = QPushButton("?")
        davies_help_btn.setMaximumWidth(30)
        davies_help_btn.setStyleSheet("font-weight: bold;")
        davies_help_btn.clicked.connect(self.show_davies_bouldin_help)
        davies_row.addWidget(davies_help_btn)
        metrics_layout.addLayout(davies_row)
        
        # Calinski-Harabasz index
        calinski_row = QHBoxLayout()
        calinski_btn = QPushButton("Calinski-Harabasz Index")
        calinski_btn.clicked.connect(self.parent_window.calculate_calinski_harabasz)
        calinski_row.addWidget(calinski_btn)
        
        calinski_help_btn = QPushButton("?")
        calinski_help_btn.setMaximumWidth(30)
        calinski_help_btn.setStyleSheet("font-weight: bold;")
        calinski_help_btn.clicked.connect(self.show_calinski_harabasz_help)
        calinski_row.addWidget(calinski_help_btn)
        metrics_layout.addLayout(calinski_row)
        
        left_layout.addWidget(metrics_group)
        
        # Optimal cluster analysis
        optimal_group = QGroupBox("Optimal Cluster Analysis")
        optimal_layout = QVBoxLayout(optimal_group)
        
        self.max_clusters_spinbox = QSpinBox()
        self.max_clusters_spinbox.setRange(2, 20)
        self.max_clusters_spinbox.setValue(10)
        optimal_layout.addWidget(QLabel("Max Clusters:"))
        optimal_layout.addWidget(self.max_clusters_spinbox)
        
        optimal_row = QHBoxLayout()
        optimal_btn = QPushButton("Find Optimal Clusters")
        optimal_btn.clicked.connect(self.parent_window.find_optimal_clusters)
        optimal_row.addWidget(optimal_btn)
        
        optimal_help_btn = QPushButton("?")
        optimal_help_btn.setMaximumWidth(30)
        optimal_help_btn.setStyleSheet("font-weight: bold;")
        optimal_help_btn.clicked.connect(self.show_optimal_clusters_help)
        optimal_row.addWidget(optimal_help_btn)
        optimal_layout.addLayout(optimal_row)
        
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
        
        # Add toolbar
        self.validation_toolbar = NavigationToolbar(self.validation_canvas, self)
        right_layout.addWidget(self.validation_toolbar)
        
        layout.addWidget(right_panel)
    
    def show_silhouette_help(self):
        """Show help dialog for Silhouette Score."""
        from PySide6.QtWidgets import QMessageBox
        
        help_text = """<h3>Silhouette Score - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
The Silhouette Score tells you how well your data points fit into their assigned clusters. 
Think of it like measuring whether people at a party are standing with the right groups.</p>

<p><b>What It Measures:</b></p>
<ul>
<li>How close each data point is to others in its own cluster (cohesion)</li>
<li>How far each data point is from points in other clusters (separation)</li>
</ul>

<p><b>How to Read the Score:</b></p>
<ul>
<li><b>0.7 to 1.0 (Excellent):</b> Your clusters are very distinct and well-separated. 
Each spectrum clearly belongs to its cluster.</li>
<li><b>0.5 to 0.7 (Good):</b> Clusters are reasonably well-defined. Most spectra are 
in the right groups.</li>
<li><b>0.25 to 0.5 (Fair):</b> Clusters overlap somewhat. Some spectra might be on 
the boundary between groups.</li>
<li><b>Below 0.25 (Poor):</b> Clusters are not well-separated. You might have too 
many or too few clusters.</li>
</ul>

<p><b>What to Do With This:</b><br>
If your score is low, try changing the number of clusters or using different 
preprocessing methods. A higher score means your clustering is more reliable.</p>

<p><b>Visual Interpretation:</b><br>
The plot shows each cluster as a colored band. Wider bands that extend far to the 
right indicate better-defined clusters.</p>"""
        
        QMessageBox.information(self, "Silhouette Score Help", help_text)
    
    def show_davies_bouldin_help(self):
        """Show help dialog for Davies-Bouldin Index."""
        from PySide6.QtWidgets import QMessageBox
        
        help_text = """<h3>Davies-Bouldin Index - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
The Davies-Bouldin Index measures how much your clusters overlap with each other. 
Think of it like checking if different groups at a party are standing too close together 
or are nicely spread out.</p>

<p><b>What It Measures:</b></p>
<ul>
<li>How spread out the points are within each cluster</li>
<li>How far apart different clusters are from each other</li>
<li>The ratio of these two measurements</li>
</ul>

<p><b>How to Read the Score:</b></p>
<ul>
<li><b>Below 0.5 (Excellent):</b> Clusters are very well-separated with minimal overlap. 
Your groupings are highly distinct.</li>
<li><b>0.5 to 1.0 (Good):</b> Clusters are reasonably separated. Most groups are 
clearly different from each other.</li>
<li><b>1.0 to 1.5 (Fair):</b> Some overlap between clusters. Groups may share 
similar characteristics.</li>
<li><b>Above 1.5 (Poor):</b> Significant overlap. Your clusters may not represent 
truly different groups.</li>
</ul>

<p><b>Important Note:</b><br>
<b>LOWER is BETTER!</b> Unlike most scores, you want this number to be as small as possible. 
A score of 0 would mean perfect separation (though this is rare in real data).</p>

<p><b>What to Do With This:</b><br>
If your score is high (above 1.5), consider using fewer clusters or different 
clustering parameters. The goal is to find natural groupings in your data.</p>"""
        
        QMessageBox.information(self, "Davies-Bouldin Index Help", help_text)
    
    def show_calinski_harabasz_help(self):
        """Show help dialog for Calinski-Harabasz Index."""
        from PySide6.QtWidgets import QMessageBox
        
        help_text = """<h3>Calinski-Harabasz Index - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
The Calinski-Harabasz Index (also called the Variance Ratio Criterion) measures how 
tight and well-separated your clusters are. Think of it like measuring whether groups 
at a party are tightly huddled together AND far apart from other groups.</p>

<p><b>What It Measures:</b></p>
<ul>
<li>How tightly packed points are within each cluster (compactness)</li>
<li>How far apart the cluster centers are from each other (separation)</li>
<li>The ratio between these measurements</li>
</ul>

<p><b>How to Read the Score:</b></p>
<ul>
<li><b>Above 1000 (Excellent):</b> Clusters are very dense and well-separated. 
Your groupings are highly reliable.</li>
<li><b>500 to 1000 (Good):</b> Clusters are reasonably compact and distinct. 
Good quality clustering.</li>
<li><b>100 to 500 (Fair):</b> Moderate cluster quality. Groups exist but may 
not be very distinct.</li>
<li><b>Below 100 (Poor):</b> Weak clustering. Groups may be artificial or poorly defined.</li>
</ul>

<p><b>Important Note:</b><br>
<b>HIGHER is BETTER!</b> Larger numbers indicate better clustering. However, the actual 
values depend on your dataset size and complexity, so compare relative scores rather 
than absolute numbers.</p>

<p><b>What to Do With This:</b><br>
Use this metric alongside others (like Silhouette Score) to validate your clustering. 
If this score is low while others are high, it might indicate that your clusters are 
well-separated but not very compact.</p>

<p><b>Context Matters:</b><br>
The score scales with your dataset size. A score of 200 might be good for a small 
dataset but poor for a large one. Look at the relative changes when comparing 
different numbers of clusters.</p>"""
        
        QMessageBox.information(self, "Calinski-Harabasz Index Help", help_text)
    
    def show_optimal_clusters_help(self):
        """Show help dialog for Optimal Cluster Analysis."""
        from PySide6.QtWidgets import QMessageBox
        
        help_text = """<h3>Find Optimal Clusters - What Does It Do?</h3>
        
<p><b>In Simple Terms:</b><br>
This tool automatically tests different numbers of clusters (from 2 up to your maximum) 
and tells you which number works best for your data. It's like trying on different 
sized shoes to find the perfect fit!</p>

<p><b>How It Works:</b></p>
<ol>
<li>Tests each possible number of clusters (e.g., 2, 3, 4, 5... up to your max)</li>
<li>Calculates <b>four different quality metrics</b> for each configuration</li>
<li>Identifies which number of clusters scores best on each metric</li>
<li>Provides a <b>consensus recommendation</b> based on all metrics</li>
</ol>

<p><b>The Four Metrics Used:</b></p>
<ul>
<li><b>Silhouette Score:</b> Measures cluster cohesion and separation</li>
<li><b>Davies-Bouldin Index:</b> Measures cluster overlap (lower is better)</li>
<li><b>Calinski-Harabasz Index:</b> Measures cluster density and separation</li>
<li><b>Elbow Method:</b> Looks for the "elbow" where adding more clusters 
doesn't help much</li>
</ul>

<p><b>Understanding the Results:</b></p>
<ul>
<li>You'll see <b>four plots</b>, one for each metric</li>
<li>Red vertical lines show the optimal number for each metric</li>
<li>The <b>consensus recommendation</b> is the number that most metrics agree on</li>
<li>If all metrics agree, that's a strong signal!</li>
<li>If metrics disagree, your data might have multiple valid groupings</li>
</ul>

<p><b>What to Do With This:</b></p>
<ol>
<li>Look at the consensus recommendation first</li>
<li>Check if the plots show clear peaks or elbows</li>
<li>If offered, you can automatically re-run clustering with the optimal number</li>
<li>Remember: the "optimal" number is a suggestion - domain knowledge matters too!</li>
</ol>

<p><b>Pro Tip:</b><br>
If you know your data should have a specific number of groups (e.g., you're analyzing 
3 different minerals), trust your domain knowledge over the metrics. These tools help 
when you're exploring unknown data.</p>"""
        
        QMessageBox.information(self, "Optimal Cluster Analysis Help", help_text)


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
        
        # Feature Importance
        feature_row = QHBoxLayout()
        feature_btn = QPushButton("Feature Importance Analysis")
        feature_btn.clicked.connect(self.parent_window.perform_anova_test)
        feature_row.addWidget(feature_btn)
        
        feature_help_btn = QPushButton("?")
        feature_help_btn.setMaximumWidth(30)
        feature_help_btn.setStyleSheet("font-weight: bold;")
        feature_help_btn.clicked.connect(self.show_feature_importance_help)
        feature_row.addWidget(feature_help_btn)
        tests_layout.addLayout(feature_row)
        
        # Discriminant Analysis
        lda_row = QHBoxLayout()
        lda_btn = QPushButton("Discriminant Analysis (LDA)")
        lda_btn.clicked.connect(self.parent_window.perform_kruskal_test)
        lda_row.addWidget(lda_btn)
        
        lda_help_btn = QPushButton("?")
        lda_help_btn.setMaximumWidth(30)
        lda_help_btn.setStyleSheet("font-weight: bold;")
        lda_help_btn.clicked.connect(self.show_lda_help)
        lda_row.addWidget(lda_help_btn)
        tests_layout.addLayout(lda_row)
        
        # Statistical Significance
        significance_row = QHBoxLayout()
        significance_btn = QPushButton("Statistical Significance (ANOVA)")
        significance_btn.clicked.connect(self.parent_window.perform_manova_test)
        significance_row.addWidget(significance_btn)
        
        significance_help_btn = QPushButton("?")
        significance_help_btn.setMaximumWidth(30)
        significance_help_btn.setStyleSheet("font-weight: bold;")
        significance_help_btn.clicked.connect(self.show_significance_help)
        significance_row.addWidget(significance_help_btn)
        tests_layout.addLayout(significance_row)
        
        left_layout.addWidget(tests_group)
        
        # Multivariate analysis
        multivariate_group = QGroupBox("Multivariate Analysis")
        multivariate_layout = QVBoxLayout(multivariate_group)
        
        # PCA analysis
        pca_row = QHBoxLayout()
        pca_btn = QPushButton("Detailed PCA Analysis")
        pca_btn.clicked.connect(self.parent_window.perform_detailed_pca)
        pca_row.addWidget(pca_btn)
        
        pca_help_btn = QPushButton("?")
        pca_help_btn.setMaximumWidth(30)
        pca_help_btn.setStyleSheet("font-weight: bold;")
        pca_help_btn.clicked.connect(self.show_pca_help)
        pca_row.addWidget(pca_help_btn)
        multivariate_layout.addLayout(pca_row)
        
        # Factor analysis
        factor_row = QHBoxLayout()
        factor_btn = QPushButton("Factor Analysis")
        factor_btn.clicked.connect(self.parent_window.perform_factor_analysis)
        factor_row.addWidget(factor_btn)
        
        factor_help_btn = QPushButton("?")
        factor_help_btn.setMaximumWidth(30)
        factor_help_btn.setStyleSheet("font-weight: bold;")
        factor_help_btn.clicked.connect(self.show_factor_help)
        factor_row.addWidget(factor_help_btn)
        multivariate_layout.addLayout(factor_row)
        
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
        
        # Add toolbar
        self.stats_toolbar = NavigationToolbar(self.stats_canvas, self)
        right_layout.addWidget(self.stats_toolbar)
        
        layout.addWidget(right_panel)
    
    def _show_compact_help(self, title, html_content):
        """Show a compact help dialog with scroll bar."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        from PySide6.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(300)  # Half the original height
        dialog.setMaximumHeight(400)
        
        layout = QVBoxLayout(dialog)
        
        # Scrollable text area
        text_edit = QTextEdit()
        text_edit.setHtml(html_content)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        # Close button
        close_btn = QPushButton("Got it!")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 24px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_feature_importance_help(self):
        """Show help dialog for Feature Importance Analysis."""
        help_text = """<h3>Feature Importance Analysis - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
Feature Importance tells you which Raman peaks (wavenumbers) are most important 
for distinguishing between your clusters. Think of it like finding which test 
questions best separate A students from B students from C students.</p>

<p><b>What It Measures:</b></p>
<ul>
<li>Which spectral features contribute most to cluster separation</li>
<li>The relative importance of each wavenumber for classification</li>
<li>Uses Random Forest machine learning to rank features</li>
<li>Identifies the "signature peaks" for each cluster</li>
</ul>

<p><b>How to Read the Results:</b></p>
<ul>
<li><b>Top 20 Features:</b> The most important wavenumbers listed with scores</li>
<li><b>Higher scores = more important</b> for distinguishing clusters</li>
<li><b>Full spectrum plot:</b> Shows importance across all wavenumbers</li>
<li><b>Cumulative plot:</b> Shows how many features capture 80-90% of information</li>
</ul>

<p><b>What to Do With This:</b></p>
<ul>
<li>Focus on the <b>top-ranked wavenumbers</b> for chemical interpretation</li>
<li>These peaks are the "fingerprints" that define each cluster</li>
<li>Look up these wavenumbers in Raman databases to identify compounds</li>
<li>Use these features for targeted analysis or validation</li>
</ul>

<p><b>Visual Outputs (4 plots):</b></p>
<ul>
<li><b>Full Spectrum:</b> Importance across all wavenumbers (red dots = top features)</li>
<li><b>Top 20 Bar Chart:</b> Highest importance features ranked</li>
<li><b>Cumulative Importance:</b> How many features needed to capture most info</li>
<li><b>Distribution:</b> Histogram showing spread of importance scores</li>
</ul>

<p><b>Pro Tip:</b><br>
If only a few features have high importance, your clusters are defined by specific 
peaks (good for chemical interpretation). If many features are important, clusters 
differ in overall spectral shape.</p>"""
        
        self._show_compact_help("Feature Importance Help", help_text)
    
    def show_lda_help(self):
        """Show help dialog for Linear Discriminant Analysis."""
        help_text = """<h3>Linear Discriminant Analysis (LDA) - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
LDA finds the best way to draw lines (or planes) that separate your clusters. 
Think of it like finding the perfect angle to photograph a group so everyone 
is clearly separated and identifiable.</p>

<p><b>What It Measures:</b></p>
<ul>
<li>How well clusters can be separated using linear combinations of features</li>
<li>The "discriminant functions" that best distinguish clusters</li>
<li>How accurately clusters can be predicted (classification accuracy)</li>
<li>Which directions in spectral space maximize cluster separation</li>
</ul>

<p><b>How to Read the Results:</b></p>
<ul>
<li><b>Cross-Validation Accuracy:</b> How well clusters can be predicted
    <ul>
    <li><b>> 90%:</b> Excellent separation - clusters are very distinct</li>
    <li><b>80-90%:</b> Good separation - clusters are well-defined</li>
    <li><b>70-80%:</b> Moderate separation - some overlap exists</li>
    <li><b>< 70%:</b> Poor separation - clusters may not be meaningful</li>
    </ul>
</li>
<li><b>Explained Variance:</b> How much information each discriminant captures</li>
</ul>

<p><b>Visual Outputs (4 plots):</b></p>
<ul>
<li><b>LDA Scatter Plot:</b> Clusters projected onto discriminant axes (LD1 vs LD2)</li>
<li><b>Explained Variance:</b> Bar chart showing importance of each discriminant</li>
<li><b>Cross-Validation:</b> Accuracy across different data subsets</li>
<li><b>Separation Matrix:</b> Heatmap showing distances between cluster centers</li>
</ul>

<p><b>What to Do With This:</b></p>
<ul>
<li>If accuracy is high (>80%), your clusters are well-separated and reliable</li>
<li>The scatter plot shows if clusters overlap or are distinct</li>
<li>Separation matrix shows which cluster pairs are most/least similar</li>
<li>Use this to validate that your clustering makes sense</li>
</ul>

<p><b>When to Use LDA:</b></p>
<ul>
<li>✅ To validate cluster quality and separation</li>
<li>✅ To visualize high-dimensional clusters in 2D/3D</li>
<li>✅ To assess if clusters can be reliably predicted</li>
<li>✅ To identify which clusters are most similar/different</li>
</ul>

<p><b>Pro Tip:</b><br>
LDA is supervised (uses cluster labels), so it shows the <i>best possible</i> 
separation. High LDA accuracy confirms your clusters are real and distinct, 
not just random groupings.</p>"""
        
        self._show_compact_help("Linear Discriminant Analysis Help", help_text)
    
    def show_significance_help(self):
        """Show help dialog for Statistical Significance Testing."""
        help_text = """<h3>Statistical Significance Testing (ANOVA) - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
Statistical Significance Testing tells you whether the differences between your 
clusters are real or just random chance. It's like asking: "Are these groups 
genuinely different, or could I get the same result by randomly shuffling the data?"</p>

<p><b>What It Measures:</b></p>
<ul>
<li>Tests each spectral feature (wavenumber) for significant differences</li>
<li>Uses ANOVA (Analysis of Variance) with Bonferroni correction</li>
<li>Identifies which features show statistically meaningful differences</li>
<li>Controls for multiple testing to avoid false positives</li>
</ul>

<p><b>How to Read the Results:</b></p>
<ul>
<li><b>F-statistics:</b> Larger values = bigger differences between clusters</li>
<li><b>p-values:</b> Probability that differences are due to chance
    <ul>
    <li><b>p < corrected α:</b> Feature is significantly different (after correction)</li>
    <li><b>p > corrected α:</b> No significant difference detected</li>
    </ul>
</li>
<li><b>Significant Features:</b> Number and percentage of features that differ</li>
</ul>

<p><b>Visual Outputs (4 plots):</b></p>
<ul>
<li><b>F-statistics Plot:</b> Shows which wavenumbers have largest differences</li>
<li><b>P-values Plot:</b> Shows statistical significance across spectrum (log scale)</li>
<li><b>F-statistic Distribution:</b> Histogram of test statistics</li>
<li><b>Summary Bar Chart:</b> Significant vs non-significant features</li>
</ul>

<p><b>What to Do With This:</b></p>
<ul>
<li>If many features are significant (>20%), clusters are statistically distinct</li>
<li>If few/no features are significant, clusters may not be meaningful</li>
<li>Significant features are the ones that truly differ between clusters</li>
<li>Use this to validate that clustering captured real chemical differences</li>
</ul>

<p><b>About Bonferroni Correction:</b></p>
<ul>
<li>Tests hundreds/thousands of features simultaneously</li>
<li>Correction prevents false positives from multiple testing</li>
<li>Conservative approach - reduces false discoveries</li>
<li>Corrected α = 0.05 / number of features (very strict)</li>
</ul>

<p><b>Pro Tip:</b><br>
This test is very conservative. Even if only a few features pass, those are 
<i>highly reliable</i> differences. If many features pass, your clusters are 
extremely well-differentiated!</p>"""
        
        self._show_compact_help("Statistical Significance Help", help_text)
    
    def show_pca_help(self):
        """Show help dialog for Detailed PCA Analysis."""
        help_text = """<h3>Detailed PCA Analysis - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
PCA (Principal Component Analysis) finds the "main themes" in your spectral data. 
Imagine you have a 1000-page book - PCA creates a summary that captures the most 
important information in just a few pages. It identifies the main patterns of 
variation in your Raman spectra.</p>

<p><b>What It Measures:</b></p>
<ul>
<li>The main directions of variation in your spectral data</li>
<li>Which combinations of peaks vary together</li>
<li>How much information each "principal component" captures</li>
<li>Whether your clusters separate along these main patterns</li>
</ul>

<p><b>Key Concepts:</b></p>
<ul>
<li><b>Principal Components (PCs):</b> New variables that capture patterns in your data
    <ul>
    <li>PC1 captures the most variation</li>
    <li>PC2 captures the second most (independent of PC1)</li>
    <li>PC3, PC4, etc. capture progressively less</li>
    </ul>
</li>
<li><b>Explained Variance:</b> Percentage of total variation captured by each PC
    <ul>
    <li>PC1 might explain 60% of variation</li>
    <li>PC1+PC2 together might explain 85%</li>
    <li>First few PCs usually capture most information</li>
    </ul>
</li>
<li><b>Loadings:</b> Show which spectral features contribute to each PC
    <ul>
    <li>High loadings = important peaks for that pattern</li>
    <li>Positive/negative loadings show opposing trends</li>
    </ul>
</li>
</ul>

<p><b>What to Do With This:</b></p>
<ul>
<li><b>Scree Plot:</b> Shows how many PCs are meaningful (look for "elbow")</li>
<li><b>Score Plot:</b> Shows how clusters separate in PC space</li>
<li><b>Loading Plot:</b> Identifies which Raman peaks drive cluster differences</li>
<li><b>Biplot:</b> Combines scores and loadings to show relationships</li>
</ul>

<p><b>Practical Applications:</b></p>
<ul>
<li>✅ Visualize high-dimensional data in 2D or 3D</li>
<li>✅ Identify which spectral features are most important</li>
<li>✅ Detect outliers and unusual spectra</li>
<li>✅ Reduce data complexity while retaining information</li>
<li>✅ Validate that clusters separate along main variation patterns</li>
</ul>

<p><b>Interpreting Results:</b></p>
<ul>
<li>If clusters separate well in PC1-PC2 plot → strong, natural groupings</li>
<li>If you need many PCs to separate clusters → subtle differences</li>
<li>Loading peaks show which Raman bands characterize each cluster</li>
</ul>

<p><b>Pro Tip:</b><br>
PCA is exploratory - it shows you patterns in your data without assuming clusters 
exist. If your clusters separate clearly in PCA space, that's strong evidence they 
represent real chemical differences!</p>"""
        
        self._show_compact_help("Detailed PCA Analysis Help", help_text)
    
    def show_factor_help(self):
        """Show help dialog for Factor Analysis."""
        help_text = """<h3>Factor Analysis - What Does It Tell You?</h3>
        
<p><b>In Simple Terms:</b><br>
Factor Analysis is like PCA's cousin that looks for <i>hidden causes</i> behind your 
data patterns. While PCA finds patterns, Factor Analysis tries to identify the 
underlying "factors" (like different chemical components) that create those patterns. 
Think of it like a detective finding the root causes rather than just describing symptoms.</p>

<p><b>What It Measures:</b></p>
<ul>
<li>Latent (hidden) factors that explain correlations in your spectra</li>
<li>Which spectral features are caused by the same underlying factor</li>
<li>How much each factor contributes to each spectrum</li>
<li>The unique variance in each feature not explained by common factors</li>
</ul>

<p><b>PCA vs Factor Analysis:</b></p>
<ul>
<li><b>PCA:</b> "What patterns exist in my data?" (descriptive)</li>
<li><b>Factor Analysis:</b> "What hidden causes create these patterns?" (explanatory)</li>
<li>PCA uses all variance; Factor Analysis focuses on <i>shared</i> variance</li>
<li>Factor Analysis assumes measurement error; PCA doesn't</li>
</ul>

<p><b>Key Concepts:</b></p>
<ul>
<li><b>Factors:</b> Hypothetical underlying variables (e.g., chemical components)
    <ul>
    <li>Each factor represents a source of variation</li>
    <li>Factors are often rotated for easier interpretation</li>
    </ul>
</li>
<li><b>Factor Loadings:</b> Correlation between factors and spectral features
    <ul>
    <li>High loadings = feature strongly related to that factor</li>
    <li>Shows which peaks belong together</li>
    </ul>
</li>
<li><b>Communality:</b> Proportion of variance explained by all factors
    <ul>
    <li>High communality = feature well-explained by factors</li>
    <li>Low communality = feature is mostly noise or unique</li>
    </ul>
</li>
</ul>

<p><b>What to Do With This:</b></p>
<ul>
<li>Identify groups of peaks that vary together (likely from same component)</li>
<li>Estimate how many distinct chemical components are in your samples</li>
<li>Assign spectral features to underlying chemical factors</li>
<li>Understand the chemical basis of your clusters</li>
</ul>

<p><b>Practical Applications for Raman:</b></p>
<ul>
<li>✅ Identify distinct chemical components in mixtures</li>
<li>✅ Group peaks belonging to the same molecule</li>
<li>✅ Separate signal from noise (low communality = noise)</li>
<li>✅ Understand chemical basis of cluster differences</li>
<li>✅ Validate that clusters represent different compositions</li>
</ul>

<p><b>When to Use Factor Analysis:</b></p>
<ul>
<li>✅ When you believe clusters represent different chemical compositions</li>
<li>✅ When you want to identify underlying chemical components</li>
<li>✅ When spectral features are highly correlated</li>
<li>⚠️ Requires good quality data with clear patterns</li>
<li>⚠️ Interpretation requires chemical knowledge</li>
</ul>

<p><b>Pro Tip:</b><br>
Use Factor Analysis after clustering to understand <i>why</i> clusters are different. 
Each factor might represent a different mineral phase, chemical component, or 
structural variation in your samples.</p>"""
        
        self._show_compact_help("Factor Analysis Help", help_text)
