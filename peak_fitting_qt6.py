#!/usr/bin/env python3
"""
Enhanced Peak Fitting and Spectral Deconvolution Module for RamanLab Qt6
Includes:
• Component separation
• Overlapping peak resolution  
• Principal component analysis
• Non-negative matrix factorization
"""

import numpy as np
import sys
from pathlib import Path

# Import Qt6
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QPushButton, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QGroupBox, QSplitter, QMessageBox, QProgressBar, QSpinBox, 
    QDoubleSpinBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QFrame, QGridLayout, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

# Import matplotlib for Qt6
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme

# Scientific computing
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd

# Machine learning imports for advanced features
try:
    from sklearn.decomposition import PCA, NMF
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class StackedTabWidget(QWidget):
    """Custom tab widget with stacked tab buttons."""
    
    def __init__(self):
        super().__init__()
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.tabs = []
        self.current_index = 0
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create button grid
        button_frame = QFrame()
        button_frame.setFrameStyle(QFrame.Box)
        button_frame.setMaximumHeight(80)
        self.button_layout = QGridLayout(button_frame)
        self.button_layout.setContentsMargins(2, 2, 2, 2)
        self.button_layout.setSpacing(2)
        
        layout.addWidget(button_frame)
        
        # Create stacked widget for tab contents
        self.stacked_widget = QWidget()
        self.stacked_layout = QVBoxLayout(self.stacked_widget)
        self.stacked_layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(self.stacked_widget)
        
        # Button group for exclusive selection
        self.buttons = []
    
    def add_tab(self, widget, text, row=0, col=0):
        """Add a tab to the stacked widget."""
        # Create button
        button = QPushButton(text)
        button.setCheckable(True)
        button.setMaximumHeight(35)
        button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 4px 8px;
                text-align: center;
                font-size: 10px;
            }
            QPushButton:checked {
                background-color: #4A90E2;
                color: white;
                border: 1px solid #357ABD;
            }
            QPushButton:hover:!checked {
                background-color: #e0e0e0;
            }
        """)
        
        # Connect button to show tab
        tab_index = len(self.tabs)
        button.clicked.connect(lambda: self.show_tab(tab_index))
        
        # Add to button layout
        self.button_layout.addWidget(button, row, col)
        self.buttons.append(button)
        
        # Add widget to tabs list
        self.tabs.append(widget)
        widget.hide()
        self.stacked_layout.addWidget(widget)
        
        # Show first tab by default
        if len(self.tabs) == 1:
            self.show_tab(0)
    
    def show_tab(self, index):
        """Show the specified tab."""
        if 0 <= index < len(self.tabs):
            # Hide current tab
            if self.current_index < len(self.tabs):
                self.tabs[self.current_index].hide()
                if self.current_index < len(self.buttons):
                    self.buttons[self.current_index].setChecked(False)
            
            # Show new tab
            self.current_index = index
            self.tabs[index].show()
            self.buttons[index].setChecked(True)

class SpectralDeconvolutionQt6(QDialog):
    """Enhanced spectral deconvolution window with advanced analysis capabilities."""
    
    def __init__(self, parent, wavenumbers, intensities):
        super().__init__(parent)
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.parent = parent
        self.wavenumbers = np.array(wavenumbers)
        self.original_intensities = np.array(intensities)
        self.processed_intensities = self.original_intensities.copy()
        
        # Analysis data
        self.peaks = np.array([])  # Initialize as empty numpy array
        self.manual_peaks = np.array([])  # Manually selected peaks
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.background_preview_active = False  # Track background preview state
        self.residuals = None
        self.components = []
        self.pca_result = None
        self.nmf_result = None
        
        # UI state
        self.show_individual_peaks = True
        self.show_components = True
        self.current_model = "Gaussian"
        
        # Interactive peak selection state
        self.interactive_mode = False
        self.click_connection = None
        
        self.setup_ui()
        self.initial_plot()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Spectral Deconvolution & Advanced Analysis")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Left panel - controls (splitter for resizing)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Control panel
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Visualization panel
        viz_panel = self.create_visualization_panel()
        splitter.addWidget(viz_panel)
        
        # Set splitter proportions (30% controls, 70% visualization)
        splitter.setSizes([400, 1200])
        
    def create_control_panel(self):
        """Create the control panel with tabs."""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Create tab widget with stacked layout
        self.tab_widget = StackedTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs with 2-row layout:
        # Row 1: Background, Peak Detection, Peak Fitting
        # Row 2: Deconvolution, Analysis Results
        self.tab_widget.add_tab(self.create_background_tab(), "Background", row=0, col=0)
        self.tab_widget.add_tab(self.create_peak_detection_tab(), "Peak Detection", row=0, col=1)
        self.tab_widget.add_tab(self.create_fitting_tab(), "Peak Fitting", row=0, col=2)
        self.tab_widget.add_tab(self.create_deconvolution_tab(), "Deconvolution", row=1, col=0)
        self.tab_widget.add_tab(self.create_analysis_tab(), "Analysis Results", row=1, col=1)
        
        return panel
        
    def create_visualization_panel(self):
        """Create the visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        
        # Create subplots with custom height ratios - main plot gets 75%, residual gets 25%
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        self.ax_main = self.figure.add_subplot(gs[0])      # Main spectrum (top, larger)
        self.ax_residual = self.figure.add_subplot(gs[1])  # Residuals (bottom, smaller)
        
        self.figure.tight_layout(pad=3.0)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        return panel
        
    def create_background_tab(self):
        """Create background subtraction tab with Process tab styling."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Background subtraction group
        bg_group = QGroupBox("Background Subtraction")
        bg_layout = QVBoxLayout(bg_group)
        
        # Background method selection
        bg_method_layout = QHBoxLayout()
        bg_method_layout.addWidget(QLabel("Method:"))
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["ALS (Asymmetric Least Squares)", "Linear", "Polynomial", "Moving Average"])
        self.bg_method_combo.currentTextChanged.connect(self.on_bg_method_changed)
        self.bg_method_combo.currentTextChanged.connect(self.preview_background_live)
        bg_method_layout.addWidget(self.bg_method_combo)
        bg_layout.addLayout(bg_method_layout)
        
        # ALS parameters (visible by default)
        self.als_params_widget = QWidget()
        als_params_layout = QVBoxLayout(self.als_params_widget)
        als_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Lambda parameter
        lambda_layout = QHBoxLayout()
        lambda_layout.addWidget(QLabel("λ (Smoothness):"))
        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setRange(3, 7)  # 10^3 to 10^7
        self.lambda_slider.setValue(5)  # 10^5 (default)
        lambda_layout.addWidget(self.lambda_slider)
        self.lambda_label = QLabel("1e5")
        lambda_layout.addWidget(self.lambda_label)
        als_params_layout.addLayout(lambda_layout)
        
        # p parameter
        p_layout = QHBoxLayout()
        p_layout.addWidget(QLabel("p (Asymmetry):"))
        self.p_slider = QSlider(Qt.Horizontal)
        self.p_slider.setRange(1, 50)  # 0.001 to 0.05
        self.p_slider.setValue(10)  # 0.01 (default)
        p_layout.addWidget(self.p_slider)
        self.p_label = QLabel("0.01")
        p_layout.addWidget(self.p_label)
        als_params_layout.addLayout(p_layout)
        
        # Iterations parameter
        niter_layout = QHBoxLayout()
        niter_layout.addWidget(QLabel("Iterations:"))
        self.niter_slider = QSlider(Qt.Horizontal)
        self.niter_slider.setRange(5, 30)
        self.niter_slider.setValue(10)
        niter_layout.addWidget(self.niter_slider)
        self.niter_label = QLabel("10")
        niter_layout.addWidget(self.niter_label)
        als_params_layout.addLayout(niter_layout)
        
        # Connect sliders to update labels and live preview
        self.lambda_slider.valueChanged.connect(self.update_lambda_label)
        self.p_slider.valueChanged.connect(self.update_p_label)
        self.niter_slider.valueChanged.connect(self.update_niter_label)
        
        # Connect sliders to live preview
        self.lambda_slider.valueChanged.connect(self.preview_background_live)
        self.p_slider.valueChanged.connect(self.preview_background_live)
        self.niter_slider.valueChanged.connect(self.preview_background_live)
        
        bg_layout.addWidget(self.als_params_widget)
        
        # Background subtraction buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply Background Subtraction")
        apply_btn.clicked.connect(self.apply_background)
        button_layout.addWidget(apply_btn)
        
        clear_btn = QPushButton("Clear Preview")
        clear_btn.clicked.connect(self.clear_background_preview)
        button_layout.addWidget(clear_btn)
        
        bg_layout.addLayout(button_layout)
        
        reset_btn = QPushButton("Reset Spectrum")
        reset_btn.clicked.connect(self.reset_spectrum)
        bg_layout.addWidget(reset_btn)
        
        layout.addWidget(bg_group)
        layout.addStretch()
        
        return tab
        
    def create_peak_detection_tab(self):
        """Create peak detection tab with interactive selection."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Peak detection group
        peak_group = QGroupBox("Automatic Peak Detection")
        peak_layout = QVBoxLayout(peak_group)
        
        # Parameters
        params_layout = QFormLayout()
        
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(1, 100)
        self.height_slider.setValue(10)
        self.height_label = QLabel("10%")
        height_layout = QHBoxLayout()
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_label)
        params_layout.addRow("Min Height:", height_layout)
        
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(10)
        self.distance_label = QLabel("10")
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(self.distance_slider)
        distance_layout.addWidget(self.distance_label)
        params_layout.addRow("Min Distance:", distance_layout)
        
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 50)
        self.prominence_slider.setValue(5)
        self.prominence_label = QLabel("5%")
        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(self.prominence_slider)
        prominence_layout.addWidget(self.prominence_label)
        params_layout.addRow("Prominence:", prominence_layout)
        
        peak_layout.addLayout(params_layout)
        
        # Connect sliders
        self.height_slider.valueChanged.connect(self.update_height_label)
        self.distance_slider.valueChanged.connect(self.update_distance_label)
        self.prominence_slider.valueChanged.connect(self.update_prominence_label)
        
        # Connect sliders to live peak detection
        self.height_slider.valueChanged.connect(self.detect_peaks_live)
        self.distance_slider.valueChanged.connect(self.detect_peaks_live)
        self.prominence_slider.valueChanged.connect(self.detect_peaks_live)
        
        # Automatic detection buttons
        auto_button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear All Peaks")
        clear_btn.clicked.connect(self.clear_peaks)
        auto_button_layout.addWidget(clear_btn)
        
        peak_layout.addLayout(auto_button_layout)
        
        layout.addWidget(peak_group)
        
        # Individual Peak Management group
        peak_management_group = QGroupBox("Individual Peak Management")
        peak_management_layout = QVBoxLayout(peak_management_group)
        
        # Peak list widget
        list_label = QLabel("Current Peaks (click to select, then remove):")
        list_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        peak_management_layout.addWidget(list_label)
        
        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setMaximumHeight(120)
        self.peak_list_widget.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #fafafa;
            }
            QListWidget::item {
                padding: 2px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e6f3ff;
            }
        """)
        peak_management_layout.addWidget(self.peak_list_widget)
        
        # Individual peak removal buttons
        individual_buttons_layout = QHBoxLayout()
        
        remove_selected_btn = QPushButton("Remove Selected Peak")
        remove_selected_btn.clicked.connect(self.remove_selected_peak)
        remove_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #800020;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #660018;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        refresh_list_btn = QPushButton("Refresh List")
        refresh_list_btn.clicked.connect(self.update_peak_list)
        refresh_list_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        
        individual_buttons_layout.addWidget(remove_selected_btn)
        individual_buttons_layout.addWidget(refresh_list_btn)
        peak_management_layout.addLayout(individual_buttons_layout)
        
        layout.addWidget(peak_management_group)
        
        # Interactive peak selection group
        interactive_group = QGroupBox("Interactive Peak Selection")
        interactive_layout = QVBoxLayout(interactive_group)
        
        # Instructions
        instructions = QLabel(
            "Enable interactive mode and click on the spectrum plot to manually select peaks.\n"
            "Click near existing peaks to remove them."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #555; font-size: 10px; margin: 5px;")
        interactive_layout.addWidget(instructions)
        
        # Interactive mode toggle button
        self.interactive_btn = QPushButton("🖱️ Enable Interactive Selection")
        self.interactive_btn.setCheckable(True)
        self.interactive_btn.clicked.connect(self.toggle_interactive_mode)
        self.interactive_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #FF5722;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:checked:hover {
                background-color: #E64A19;
            }
        """)
        interactive_layout.addWidget(self.interactive_btn)
        
        # Manual peak controls
        manual_controls_layout = QHBoxLayout()
        
        clear_manual_btn = QPushButton("Clear Manual Peaks")
        clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        clear_manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #800020;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #660018;
            }
        """)
        
        combine_btn = QPushButton("Combine Auto + Manual")
        combine_btn.clicked.connect(self.combine_peaks)
        combine_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        
        manual_controls_layout.addWidget(clear_manual_btn)
        manual_controls_layout.addWidget(combine_btn)
        interactive_layout.addLayout(manual_controls_layout)
        
        layout.addWidget(interactive_group)
        
        # Peak count and status
        status_group = QGroupBox("Peak Status")
        status_layout = QVBoxLayout(status_group)
        
        self.peak_count_label = QLabel("Auto peaks: 0 | Manual peaks: 0 | Total: 0")
        self.peak_count_label.setStyleSheet("font-weight: bold; color: #333;")
        status_layout.addWidget(self.peak_count_label)
        
        self.interactive_status_label = QLabel("Interactive mode: OFF")
        self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
        status_layout.addWidget(self.interactive_status_label)
        
        layout.addWidget(status_group)
        layout.addStretch()
        
        return tab
        
    def create_fitting_tab(self):
        """Create peak fitting tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_group = QGroupBox("Peak Model")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_peaks_check = QCheckBox("Show Individual Peaks")
        self.show_peaks_check.setChecked(True)
        self.show_peaks_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_peaks_check)
        
        self.show_components_check = QCheckBox("Show Components")
        self.show_components_check.setChecked(True)
        self.show_components_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_components_check)
        
        layout.addWidget(display_group)
        
        # Fitting
        fit_group = QGroupBox("Peak Fitting")
        fit_layout = QVBoxLayout(fit_group)
        
        fit_btn = QPushButton("Fit Peaks")
        fit_btn.clicked.connect(self.fit_peaks)
        fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        fit_layout.addWidget(fit_btn)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        fit_layout.addWidget(self.results_text)
        
        layout.addWidget(fit_group)
        layout.addStretch()
        
        return tab
        
    def create_deconvolution_tab(self):
        """Create spectral deconvolution tab with advanced features."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Component separation
        comp_group = QGroupBox("Component Separation")
        comp_layout = QVBoxLayout(comp_group)
        
        # Number of components
        comp_params = QFormLayout()
        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(1, 10)
        self.n_components_spin.setValue(3)
        comp_params.addRow("Components:", self.n_components_spin)
        comp_layout.addLayout(comp_params)
        
        # Separation methods
        method_layout = QHBoxLayout()
        overlap_btn = QPushButton("Resolve Overlaps")
        separate_btn = QPushButton("Separate Components")
        
        overlap_btn.clicked.connect(self.resolve_overlapping_peaks)
        separate_btn.clicked.connect(self.separate_components)
        
        method_layout.addWidget(overlap_btn)
        method_layout.addWidget(separate_btn)
        comp_layout.addLayout(method_layout)
        
        layout.addWidget(comp_group)
        
        # Advanced analysis (only if ML available)
        if ML_AVAILABLE:
            analysis_group = QGroupBox("Advanced Analysis")
            analysis_layout = QVBoxLayout(analysis_group)
            
            # PCA
            pca_btn = QPushButton("Principal Component Analysis")
            pca_btn.clicked.connect(self.perform_pca)
            analysis_layout.addWidget(pca_btn)
            
            # NMF
            nmf_btn = QPushButton("Non-negative Matrix Factorization")
            nmf_btn.clicked.connect(self.perform_nmf)
            analysis_layout.addWidget(nmf_btn)
            
            layout.addWidget(analysis_group)
        else:
            # Show info about missing ML capabilities
            info_label = QLabel("Install scikit-learn for advanced ML features")
            info_label.setStyleSheet("color: orange; font-style: italic;")
            layout.addWidget(info_label)
        
        layout.addStretch()
        return tab
        
    def create_analysis_tab(self):
        """Create analysis results tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results table
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Peak", "Position", "Amplitude", "Width"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        # Export options
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(export_btn)
        
        layout.addWidget(export_group)
        layout.addStretch()
        
        return tab 

    # Implementation methods
    def initial_plot(self):
        """Create initial plot."""
        self.update_plot()
        
    def update_plot(self):
        """Update all plots."""
        # Clear all axes
        self.ax_main.clear()
        self.ax_residual.clear()
        
        # Main spectrum plot
        self.ax_main.plot(self.wavenumbers, self.processed_intensities, 'b-', 
                         linewidth=1.5, label='Spectrum')
        
        # Plot background if available
        if self.background is not None:
            self.ax_main.plot(self.wavenumbers, self.background, 'r--', 
                             linewidth=1, alpha=0.7, label='Background')
        
        # Plot fitted peaks if available
        if self.fit_result is not None and self.show_individual_peaks:
            # Plot total fitted curve
            fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
            self.ax_main.plot(self.wavenumbers, fitted_curve, 'g-', 
                             linewidth=2, label='Fitted')
            
            # Plot individual peaks
            self.plot_individual_peaks()
        
        # Plot automatically detected peaks - FIX: Handle numpy array boolean check properly
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            # Validate peak indices before plotting
            valid_auto_peaks = self.validate_peak_indices(self.peaks)
            if len(valid_auto_peaks) > 0:
                auto_peak_positions = [self.wavenumbers[p] for p in valid_auto_peaks]
                auto_peak_intensities = [self.processed_intensities[p] for p in valid_auto_peaks]
                self.ax_main.plot(auto_peak_positions, auto_peak_intensities, 'ro', 
                                 markersize=8, label='Auto Peaks', alpha=0.8)
        
        # Plot manually selected peaks - FIX: Validate indices to prevent IndexError
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            # Validate peak indices before plotting
            valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
            if len(valid_manual_peaks) > 0:
                manual_peak_positions = [self.wavenumbers[p] for p in valid_manual_peaks]
                manual_peak_intensities = [self.processed_intensities[p] for p in valid_manual_peaks]
                self.ax_main.plot(manual_peak_positions, manual_peak_intensities, 'gs', 
                                 markersize=10, label='Manual Peaks', alpha=0.8, markeredgecolor='darkgreen')
        
        # Add interactive mode indicator
        if self.interactive_mode:
            self.ax_main.text(0.02, 0.98, '🖱️ Interactive Mode ON\nClick to select peaks', 
                             transform=self.ax_main.transAxes, 
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Spectrum and Peak Analysis')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
        # Residuals plot
        if self.residuals is not None:
            self.ax_residual.plot(self.wavenumbers, self.residuals, 'k-', linewidth=1)
            self.ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_residual.set_ylabel('Residuals')
            self.ax_residual.set_title('Fit Residuals')
            self.ax_residual.grid(True, alpha=0.3)
        
        # Update peak list if it exists
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        self.canvas.draw()

    def validate_peak_indices(self, peak_indices):
        """Validate peak indices to ensure they're within bounds and are integers."""
        if peak_indices is None or len(peak_indices) == 0:
            return np.array([])
        
        valid_peaks = []
        max_index = len(self.wavenumbers) - 1
        
        for peak_idx in peak_indices:
            try:
                # Convert to integer if possible
                peak_idx = int(peak_idx)
                
                # Check if within bounds
                if 0 <= peak_idx <= max_index:
                    valid_peaks.append(peak_idx)
                    
            except (ValueError, TypeError):
                # Skip invalid indices
                continue
        
        return np.array(valid_peaks, dtype=int)

    def plot_individual_peaks(self):
        """Plot individual fitted peaks."""
        # Check if we have fit parameters and any peaks to plot
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return
        
        # Get all peaks that were used for fitting (auto + manual)
        all_fitted_peaks = self.get_all_peaks_for_fitting()
        if len(all_fitted_peaks) == 0:
            return
        
        # Validate peaks before using them
        validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
        if len(validated_peaks) == 0:
            return
            
        n_peaks = len(validated_peaks)
        model = self.current_model
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                
                # Generate individual peak curve
                if model == "Gaussian":
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                elif model == "Lorentzian":
                    peak_curve = self.lorentzian(self.wavenumbers, amp, cen, wid)
                elif model == "Pseudo-Voigt":
                    peak_curve = self.pseudo_voigt(self.wavenumbers, amp, cen, wid)
                else:
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)  # Default
                
                # Plot individual peak - only label the first one to avoid legend clutter
                label = 'Individual Peaks' if i == 0 else None
                self.ax_main.plot(self.wavenumbers, peak_curve, '--', 
                                 linewidth=1.2, alpha=0.6, color='orange',
                                 label=label)

    def plot_analysis_results(self):
        """Plot PCA/NMF analysis results on the main plot."""
        # Note: This method is kept for compatibility but analysis results
        # will now be displayed on the main plot when PCA/NMF methods are called
        pass

    # Background subtraction methods
    def on_bg_method_changed(self):
        """Handle change in background method."""
        method = self.bg_method_combo.currentText()
        # Show ALS parameters only when ALS is selected
        self.als_params_widget.setVisible(method.startswith("ALS"))
        
        # Clear any active background preview when method changes
        if hasattr(self, 'background_preview_active') and self.background_preview_active:
            self.clear_background_preview()

    def update_lambda_label(self):
        """Update the lambda label based on the slider value."""
        value = self.lambda_slider.value()
        self.lambda_label.setText(f"1e{value}")

    def update_p_label(self):
        """Update the p label based on the slider value."""
        value = self.p_slider.value()
        p_value = value / 1000.0  # Convert to 0.001-0.05 range
        self.p_label.setText(f"{p_value:.3f}")
    
    def update_niter_label(self):
        """Update the iterations label based on the slider value."""
        value = self.niter_slider.value()
        self.niter_label.setText(str(value))

    def preview_background_live(self):
        """Live preview background subtraction with current slider values."""
        try:
            method = self.bg_method_combo.currentText()
            
            if method.startswith("ALS"):
                # Get ALS parameters from sliders
                lambda_value = 10 ** self.lambda_slider.value()
                p_value = self.p_slider.value() / 1000.0
                niter_value = self.niter_slider.value()
                
                # Calculate background using ALS
                self.background = self.baseline_als(self.original_intensities, lambda_value, p_value, niter_value)
                
            elif method == "Linear":
                # Linear background subtraction
                background = np.linspace(
                    self.original_intensities[0],
                    self.original_intensities[-1],
                    len(self.original_intensities)
                )
                self.background = background
                
            elif method == "Polynomial":
                # Polynomial background fitting (order 2)
                x = np.arange(len(self.original_intensities))
                coeffs = np.polyfit(x, self.original_intensities, 2)
                background = np.polyval(coeffs, x)
                self.background = background
                
            elif method == "Moving Average":
                # Moving average background
                from scipy import ndimage
                window_size = max(int(len(self.original_intensities) * 0.1), 50)
                background = ndimage.uniform_filter1d(self.original_intensities, size=window_size)
                self.background = background
            
            # Set preview flag
            self.background_preview_active = True
            self.update_plot()
            
        except Exception as e:
            print(f"Background preview error: {str(e)}")

    def clear_background_preview(self):
        """Clear the background preview."""
        self.background = None
        self.background_preview_active = False
        self.update_plot()

    def detect_peaks_live(self):
        """Live peak detection based on current slider values."""
        try:
            # Get current slider values
            height_percent = self.height_slider.value()
            distance = self.distance_slider.value()
            prominence_percent = self.prominence_slider.value()
            
            # Calculate thresholds - ensure they are scalars
            max_intensity = float(np.max(self.processed_intensities))
            height_threshold = (height_percent / 100.0) * max_intensity if height_percent > 0 else None
            prominence_threshold = (prominence_percent / 100.0) * max_intensity if prominence_percent > 0 else None
            
            # Ensure distance is an integer
            distance = int(distance) if distance > 0 else None
            
            # Find peaks with proper parameter handling
            peak_kwargs = {}
            if height_threshold is not None:
                peak_kwargs['height'] = height_threshold
            if distance is not None:
                peak_kwargs['distance'] = distance
            if prominence_threshold is not None:
                peak_kwargs['prominence'] = prominence_threshold
            
            self.peaks, properties = find_peaks(self.processed_intensities, **peak_kwargs)
            
            self.update_peak_count_display()
            self.update_plot()
            
        except Exception as e:
            print(f"Live peak detection error: {str(e)}")

    def preview_background(self):
        """Preview background subtraction (legacy method)."""
        self.preview_background_live()
    
    def apply_background(self):
        """Apply background subtraction."""
        if self.background is not None:
            self.processed_intensities = self.original_intensities - self.background
            # Clear preview flag
            self.background_preview_active = False
            self.update_plot()
        else:
            # If no preview, generate background first
            self.preview_background_live()
            if self.background is not None:
                self.apply_background()
    
    def reset_spectrum(self):
        """Reset to original spectrum."""
        self.processed_intensities = self.original_intensities.copy()
        self.background = None
        self.background_preview_active = False  # Clear background preview state
        self.peaks = np.array([])  # Reset as empty numpy array
        self.manual_peaks = np.array([])  # Reset manual peaks
        self.fit_params = []
        self.fit_result = None
        self.residuals = None  # Clear residuals
        self.components = []
        
        # Disable interactive mode
        if self.interactive_mode:
            self.interactive_mode = False
            self.interactive_btn.setChecked(False)
            self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: OFF")
            self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Disconnect mouse click event
            if self.click_connection is not None:
                self.canvas.mpl_disconnect(self.click_connection)
                self.click_connection = None
        
        self.update_peak_count_display()
        self.update_peak_list()  # Update the peak list widget
        self.update_plot()
    
    def baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """Asymmetric Least Squares baseline correction."""
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        
        return z
    
    # Peak detection methods
    def update_height_label(self):
        """Update height slider label."""
        value = self.height_slider.value()
        self.height_label.setText(f"{value}%")
    
    def update_distance_label(self):
        """Update distance slider label."""
        value = self.distance_slider.value()
        self.distance_label.setText(str(value))
    
    def update_prominence_label(self):
        """Update prominence slider label."""
        value = self.prominence_slider.value()
        self.prominence_label.setText(f"{value}%")
    
    def detect_peaks(self):
        """Detect peaks in the spectrum (legacy method - now calls live detection)."""
        self.detect_peaks_live()
    
    def clear_peaks(self):
        """Clear all detected peaks (both automatic and manual)."""
        self.peaks = np.array([])  # Initialize as empty numpy array
        self.manual_peaks = np.array([])  # Clear manual peaks too
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        self.update_peak_count_display()
        self.update_peak_list()  # Update the peak list widget
        self.update_plot()
    
    def update_peak_count_display(self):
        """Update the peak count display with auto and manual counts."""
        auto_count = len(self.peaks) if hasattr(self, 'peaks') and self.peaks is not None else 0
        manual_count = len(self.manual_peaks) if hasattr(self, 'manual_peaks') and self.manual_peaks is not None else 0
        total_count = auto_count + manual_count
        
        self.peak_count_label.setText(f"Auto peaks: {auto_count} | Manual peaks: {manual_count} | Total: {total_count}")

    # Interactive peak selection methods
    def toggle_interactive_mode(self):
        """Toggle interactive peak selection mode."""
        self.interactive_mode = self.interactive_btn.isChecked()
        
        if self.interactive_mode:
            # Enable interactive mode
            self.interactive_btn.setText("🖱️ Disable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: ON - Click on spectrum to select peaks")
            self.interactive_status_label.setStyleSheet("color: #4CAF50; font-size: 10px; font-weight: bold;")
            
            # Connect mouse click event
            self.click_connection = self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
            
        else:
            # Disable interactive mode
            self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: OFF")
            self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Disconnect mouse click event
            if self.click_connection is not None:
                self.canvas.mpl_disconnect(self.click_connection)
                self.click_connection = None
        
        self.update_plot()

    def on_canvas_click(self, event):
        """Handle mouse clicks on the canvas for peak selection."""
        # Only respond to clicks on the main spectrum plot
        if event.inaxes != self.ax_main:
            return
        
        # Only respond to left mouse button
        if event.button != 1:
            return
        
        if not self.interactive_mode:
            return
        
        # Get click coordinates
        click_x = event.xdata
        click_y = event.ydata
        
        if click_x is None or click_y is None:
            return
        
        try:
            # Find the closest data point to the click
            click_wavenumber = click_x
            
            # Find the closest wavenumber index
            closest_idx = np.argmin(np.abs(self.wavenumbers - click_wavenumber))
            
            # Validate the index
            if closest_idx < 0 or closest_idx >= len(self.wavenumbers):
                return
            
            # Check if we're clicking near an existing peak to remove it
            removal_threshold = 20  # wavenumber units
            
            # Check automatic peaks
            removed_auto = False
            if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                valid_auto_peaks = self.validate_peak_indices(self.peaks)
                for i, peak_idx in enumerate(valid_auto_peaks):
                    peak_wavenumber = self.wavenumbers[peak_idx]
                    if abs(peak_wavenumber - click_wavenumber) < removal_threshold:
                        # Remove this automatic peak
                        peak_list = list(self.peaks)
                        if peak_idx in peak_list:
                            peak_list.remove(peak_idx)
                            self.peaks = np.array(peak_list)
                            removed_auto = True
                            break
            
            # Check manual peaks
            removed_manual = False
            if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
                valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
                for i, peak_idx in enumerate(valid_manual_peaks):
                    peak_wavenumber = self.wavenumbers[peak_idx]
                    if abs(peak_wavenumber - click_wavenumber) < removal_threshold:
                        # Remove this manual peak
                        peak_list = list(self.manual_peaks)
                        if peak_idx in peak_list:
                            peak_list.remove(peak_idx)
                            self.manual_peaks = np.array(peak_list)
                            removed_manual = True
                            break
            
            # If we didn't remove any peak, add a new manual peak
            if not removed_auto and not removed_manual:
                # Add new manual peak
                if not hasattr(self, 'manual_peaks') or self.manual_peaks is None:
                    self.manual_peaks = np.array([closest_idx])
                else:
                    # Check if this peak is already in manual peaks
                    should_add_peak = True
                    if len(self.manual_peaks) > 0:
                        should_add_peak = closest_idx not in self.manual_peaks.tolist()
                    
                    if should_add_peak:
                        self.manual_peaks = np.append(self.manual_peaks, closest_idx)
            
            # Update display
            self.update_peak_count_display()
            self.update_peak_list()  # Update the peak list widget
            self.update_plot()
            
        except Exception as e:
            print(f"Error in interactive peak selection: {e}")

    def clear_manual_peaks(self):
        """Clear only manually selected peaks."""
        self.manual_peaks = np.array([])
        self.update_peak_count_display()
        self.update_peak_list()  # Update the peak list widget
        self.update_plot()

    def combine_peaks(self):
        """Combine automatic and manual peaks into the main peaks list."""
        # Combine automatic and manual peaks
        all_peaks = []
        
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            all_peaks.extend(self.peaks.tolist())
        
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            all_peaks.extend(self.manual_peaks.tolist())
        
        # Remove duplicates and sort
        if len(all_peaks) > 0:
            unique_peaks = sorted(list(set(all_peaks)))
            self.peaks = np.array(unique_peaks)
        else:
            self.peaks = np.array([])
        
        # Clear manual peaks since they're now in the main list
        self.manual_peaks = np.array([])
        
        # Update display
        self.update_peak_count_display()
        self.update_peak_list()  # Update the peak list widget
        self.update_plot()
        
        # Show confirmation
        QMessageBox.information(self, "Peaks Combined", 
                              f"Combined peaks into main list.\nTotal peaks: {len(self.peaks)}")

    # Peak fitting methods
    def on_model_changed(self):
        """Handle model selection change."""
        self.current_model = self.model_combo.currentText()
        if self.fit_result is not None:
            self.update_plot()
    
    def on_display_changed(self):
        """Handle display option changes."""
        self.show_individual_peaks = self.show_peaks_check.isChecked()
        self.show_components = self.show_components_check.isChecked()
        self.update_plot()
    
    def gaussian(self, x, amp, cen, wid):
        """Gaussian peak function."""
        return amp * np.exp(-((x - cen) / wid)**2)
    
    def lorentzian(self, x, amp, cen, wid):
        """Lorentzian peak function."""
        return amp / (1 + ((x - cen) / wid)**2)
    
    def pseudo_voigt(self, x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function."""
        gaussian = self.gaussian(x, amp, cen, wid)
        lorentzian = self.lorentzian(x, amp, cen, wid)
        return eta * lorentzian + (1 - eta) * gaussian
    
    def multi_peak_model(self, x, *params):
        """Multi-peak model function."""
        
        if not hasattr(self, 'peaks') or self.peaks is None or len(self.peaks) == 0:
            return np.zeros_like(x)
        
        # Validate peaks before using them
        validated_peaks = self.validate_peak_indices(self.peaks)
        
        if len(validated_peaks) == 0:
            return np.zeros_like(x)
        
        # Ensure we have the right number of parameters for the peaks
        n_peaks = len(validated_peaks)
        expected_params = n_peaks * 3
        if len(params) < expected_params:
            # Pad with default values if not enough parameters
            params = list(params) + [100, 1500, 50] * (expected_params - len(params) // 3)
        
        model = np.zeros_like(x)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(params):
                amp, cen, wid = params[start_idx:start_idx+3]
                
                # Ensure positive width
                wid = max(abs(wid), 1.0)
                
                if self.current_model == "Gaussian":
                    component = self.gaussian(x, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    component = self.lorentzian(x, amp, cen, wid)
                elif self.current_model == "Pseudo-Voigt":
                    component = self.pseudo_voigt(x, amp, cen, wid)
                else:
                    component = self.gaussian(x, amp, cen, wid)  # Default
                
                model += component
        
        return model
    
    def fit_peaks(self):
        """Fit peaks to the spectrum (uses combined automatic and manual peaks)."""
        try:
            # Get all peaks (automatic + manual) for fitting
            all_peaks = self.get_all_peaks_for_fitting()
            
            if len(all_peaks) == 0:
                QMessageBox.warning(self, "No Peaks", 
                                  "Detect peaks or select peaks manually first before fitting.\\n\\n"
                                  "Use 'Combine Auto + Manual' button to merge peak lists if needed.")
                return
            
            # Store peaks for the model function
            self.peaks = np.array(all_peaks)
            
            # Create initial parameter guesses
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            for i, peak_idx in enumerate(all_peaks):
                if 0 <= peak_idx < len(self.wavenumbers):
                    # Amplitude: Use actual intensity at peak
                    amp = self.processed_intensities[peak_idx]
                    
                    # Center: Use wavenumber at peak
                    cen = self.wavenumbers[peak_idx]
                    
                    # Width: Estimate from local curvature
                    wid = self.estimate_peak_width(peak_idx)
                    
                    initial_params.extend([amp, cen, wid])
                    
                    # Set reasonable bounds
                    bounds_lower.extend([amp * 0.1, cen - wid * 2, wid * 0.3])
                    bounds_upper.extend([amp * 10, cen + wid * 2, wid * 3])
            
            if not initial_params:
                QMessageBox.warning(self, "Invalid Peaks", "No valid peak parameters could be estimated.")
                return
            
            # Prepare bounds
            bounds = (bounds_lower, bounds_upper)
            
            # Define fitting strategies with different approaches
            strategies = [
                # Strategy 1: Standard fitting
                {
                    'p0': initial_params,
                    'bounds': bounds,
                    'method': 'trf',
                    'max_nfev': 2000
                },
                # Strategy 2: Relaxed bounds
                {
                    'p0': initial_params,
                    'bounds': ([b * 0.5 for b in bounds_lower], [b * 1.5 for b in bounds_upper]),
                    'method': 'lm',
                    'max_nfev': 1000
                },
                # Strategy 3: No bounds (if others fail)
                {
                    'p0': initial_params,
                    'method': 'lm',
                    'max_nfev': 3000
                }
            ]
            
            fit_success = False
            best_params = None
            best_r_squared = -1
            
            for i, strategy in enumerate(strategies):
                try:
                    # Apply strategy
                    if 'bounds' in strategy:
                        popt, pcov = curve_fit(
                            self.multi_peak_model, 
                            self.wavenumbers, 
                            self.processed_intensities,
                            p0=strategy['p0'],
                            bounds=strategy['bounds'],
                            method=strategy['method'],
                            max_nfev=strategy['max_nfev']
                        )
                    else:
                        popt, pcov = curve_fit(
                            self.multi_peak_model, 
                            self.wavenumbers, 
                            self.processed_intensities,
                            p0=strategy['p0'],
                            method=strategy['method'],
                            max_nfev=strategy['max_nfev']
                        )
                    
                    # Calculate R-squared
                    fitted_y = self.multi_peak_model(self.wavenumbers, *popt)
                    ss_res = np.sum((self.processed_intensities - fitted_y) ** 2)
                    ss_tot = np.sum((self.processed_intensities - np.mean(self.processed_intensities)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Check if this is the best fit so far
                    if r_squared > best_r_squared:
                        best_params = popt
                        best_r_squared = r_squared
                        fit_success = True
                    
                    break  # If we reach here, fitting was successful
                    
                except Exception as e:
                    continue  # Try next strategy
            
            if fit_success:
                # Store the best results
                self.fit_params = best_params
                self.fit_result = True
                
                # Calculate residuals properly
                fitted_curve = self.multi_peak_model(self.wavenumbers, *best_params)
                self.residuals = self.processed_intensities - fitted_curve
                
                # Update displays
                self.update_plot()
                self.display_fit_results(best_r_squared, all_peaks)
                self.update_results_table()
                
                QMessageBox.information(self, "Success", 
                                      f"Peak fitting completed successfully!\\n"
                                      f"R² = {best_r_squared:.4f}\\n"
                                      f"Fitted {len(all_peaks)} peaks")
            else:
                QMessageBox.warning(self, "Fitting Failed", 
                                  "Peak fitting failed with all strategies.\\n\\n"
                                  "Try:\\n"
                                  "• Adjusting peak detection parameters\\n"
                                  "• Reducing the number of peaks\\n"
                                  "• Improving background subtraction\\n"
                                  "• Using a different fitting model")
                
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Peak fitting failed: {str(e)}")
    
    def estimate_peak_width(self, peak_idx):
        """Estimate peak width based on local data around peak."""
        try:
            # Look at points around the peak
            window = 20  # points on each side
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(self.processed_intensities), peak_idx + window + 1)
            
            local_intensities = self.processed_intensities[start_idx:end_idx]
            local_wavenumbers = self.wavenumbers[start_idx:end_idx]
            
            peak_intensity = self.processed_intensities[peak_idx]
            half_max = peak_intensity / 2
            
            # Find FWHM (Full Width at Half Maximum)
            above_half = local_intensities > half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                if len(indices) > 1:
                    fwhm_indices = [indices[0], indices[-1]]
                    fwhm = abs(local_wavenumbers[fwhm_indices[1]] - local_wavenumbers[fwhm_indices[0]])
                    # Convert FWHM to Gaussian sigma (width parameter)
                    width = fwhm / (2 * np.sqrt(2 * np.log(2)))
                    return max(width, 5.0)  # minimum width
            
            # Fallback: estimate based on wavenumber spacing
            wavenumber_spacing = np.mean(np.diff(self.wavenumbers))
            return max(10 * wavenumber_spacing, 5.0)
            
        except Exception:
            # Default fallback
            return 10.0

    def get_all_peaks_for_fitting(self):
        """Get all peaks (automatic + manual) for fitting operations."""
        all_peaks = []
        
        # Get automatic peaks with validation
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            validated_auto = self.validate_peak_indices(self.peaks)
            all_peaks.extend(validated_auto.tolist())
        
        # Get manual peaks with validation
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            validated_manual = self.validate_peak_indices(self.manual_peaks)
            all_peaks.extend(validated_manual.tolist())
        
        # Remove duplicates and sort
        if len(all_peaks) > 0:
            unique_peaks = sorted(list(set(all_peaks)))
            return unique_peaks
        else:
            return []

    def display_fit_results(self, r_squared, fitted_peaks=None):
        """Display fitting results."""
        if fitted_peaks is None:
            fitted_peaks = self.get_all_peaks_for_fitting()
        
        # Ensure fitted_peaks are validated
        validated_fitted_peaks = []
        if isinstance(fitted_peaks, (list, np.ndarray)) and len(fitted_peaks) > 0:
            validated_fitted_peaks = self.validate_peak_indices(np.array(fitted_peaks))
        
        results = f"Peak Fitting Results\n{'='*30}\n\n"
        results += f"Model: {self.current_model}\n"
        results += f"Number of peaks fitted: {len(validated_fitted_peaks)}\n"
        results += f"R² = {r_squared:.4f}\n\n"
        
        # FIX: Properly handle numpy array boolean check
        if (self.fit_params is not None and 
            hasattr(self.fit_params, '__len__') and 
            len(self.fit_params) > 0 and 
            len(validated_fitted_peaks) > 0):
            
            results += "Peak Parameters:\n"
            n_peaks = len(validated_fitted_peaks)
            for i in range(n_peaks):
                start_idx = i * 3
                if start_idx + 2 < len(self.fit_params):
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    # Determine peak type safely
                    peak_type = "Auto"
                    if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                        validated_auto_peaks = self.validate_peak_indices(self.peaks)
                        if len(validated_auto_peaks) > 0 and validated_fitted_peaks[i] not in validated_auto_peaks.tolist():
                            peak_type = "Manual"
                    results += f"Peak {i+1} ({peak_type}): Center={cen:.1f}, Amplitude={amp:.1f}, Width={wid:.1f}\n"
        
        self.results_text.setPlainText(results)

    def update_results_table(self):
        """Update the results table."""
        # FIX: Properly handle numpy array boolean check
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return
        
        fitted_peaks = self.get_all_peaks_for_fitting()
        if len(fitted_peaks) == 0:
            return
        
        # Validate the fitted peaks
        validated_fitted_peaks = self.validate_peak_indices(np.array(fitted_peaks))
        n_peaks = len(validated_fitted_peaks)
        self.results_table.setRowCount(n_peaks)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                
                # Determine peak type safely
                peak_type = "Auto"
                if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                    validated_auto_peaks = self.validate_peak_indices(self.peaks)
                    if len(validated_auto_peaks) > 0 and validated_fitted_peaks[i] not in validated_auto_peaks.tolist():
                        peak_type = "Manual"
                
                self.results_table.setItem(i, 0, QTableWidgetItem(f"Peak {i+1} ({peak_type})"))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{cen:.1f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{amp:.1f}"))
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{wid:.1f}"))

    # Advanced deconvolution methods
    def resolve_overlapping_peaks(self):
        """Resolve overlapping peaks using advanced algorithms."""
        if not hasattr(self, 'peaks') or self.peaks is None or len(self.peaks) < 2:
            QMessageBox.warning(self, "Insufficient Peaks", 
                              "Need at least 2 peaks to resolve overlaps.")
            return
        
        try:
            # Use iterative peak fitting with constraints
            self.fit_peaks()  # First, do basic fitting
            
            if self.fit_result:
                # Identify overlapping peaks (peaks within 3*width of each other)
                overlapping_groups = self.identify_overlapping_groups()
                
                if overlapping_groups:
                    # Apply deconvolution to overlapping groups
                    self.deconvolve_overlapping_groups(overlapping_groups)
                    
                    QMessageBox.information(self, "Success", 
                                          f"Resolved {len(overlapping_groups)} overlapping peak groups.")
                else:
                    QMessageBox.information(self, "No Overlaps", 
                                          "No significantly overlapping peaks detected.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Overlap resolution failed: {str(e)}")
    
    def identify_overlapping_groups(self):
        """Identify groups of overlapping peaks."""
        # FIX: Properly handle numpy array boolean check
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0 or 
            not hasattr(self, 'peaks') or 
            self.peaks is None or 
            len(self.peaks) == 0):
            return []
        
        overlapping_groups = []
        n_peaks = len(self.peaks)
        processed_peaks = set()
        
        for i in range(n_peaks):
            if i in processed_peaks:
                continue
                
            start_idx = i * 3
            if start_idx + 2 >= len(self.fit_params):
                continue
                
            amp_i, cen_i, wid_i = self.fit_params[start_idx:start_idx+3]
            group = [i]
            
            # Find overlapping peaks
            for j in range(i + 1, n_peaks):
                if j in processed_peaks:
                    continue
                    
                start_idx_j = j * 3
                if start_idx_j + 2 >= len(self.fit_params):
                    continue
                    
                amp_j, cen_j, wid_j = self.fit_params[start_idx_j:start_idx_j+3]
                
                # Check if peaks overlap (within 3*width)
                if abs(cen_i - cen_j) < 3 * max(wid_i, wid_j):
                    group.append(j)
                    processed_peaks.add(j)
            
            if len(group) > 1:
                overlapping_groups.append(group)
                for peak_idx in group:
                    processed_peaks.add(peak_idx)
        
        return overlapping_groups
    
    def deconvolve_overlapping_groups(self, overlapping_groups):
        """Apply advanced deconvolution to overlapping peak groups."""
        for group in overlapping_groups:
            # Extract region around overlapping peaks
            group_centers = []
            group_widths = []
            
            for peak_idx in group:
                start_idx = peak_idx * 3
                if start_idx + 2 < len(self.fit_params):
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    group_centers.append(cen)
                    group_widths.append(wid)
            
            if not group_centers:
                continue
            
            # Define region of interest
            min_center = min(group_centers)
            max_center = max(group_centers)
            max_width = max(group_widths)
            
            # Extend region by 5*max_width on each side
            region_start = min_center - 5 * max_width
            region_end = max_center + 5 * max_width
            
            # Extract region indices
            region_mask = (self.wavenumbers >= region_start) & (self.wavenumbers <= region_end)
            region_wavenumbers = self.wavenumbers[region_mask]
            region_intensities = self.processed_intensities[region_mask]
            
            if len(region_wavenumbers) < 10:
                continue
            
            # Apply Richardson-Lucy deconvolution or similar
            deconvolved = self.richardson_lucy_deconvolution(region_intensities)
            
            # Update the spectrum in this region
            self.processed_intensities[region_mask] = deconvolved
        
        self.update_plot()
    
    def richardson_lucy_deconvolution(self, data, iterations=10):
        """Apply Richardson-Lucy deconvolution algorithm."""
        # Simple implementation - could be enhanced with proper PSF estimation
        deconvolved = data.copy()
        
        for _ in range(iterations):
            # Estimate point spread function
            psf = self.estimate_psf(len(data))
            
            # Convolve current estimate
            convolved = np.convolve(deconvolved, psf, mode='same')
            
            # Avoid division by zero
            ratio = np.divide(data, convolved + 1e-10)
            
            # Update estimate
            deconvolved *= np.convolve(ratio, psf[::-1], mode='same')
            
            # Ensure non-negativity
            deconvolved = np.maximum(deconvolved, 0)
        
        return deconvolved
    
    def estimate_psf(self, size, sigma=2.0):
        """Estimate point spread function for deconvolution."""
        x = np.arange(size) - size // 2
        psf = np.exp(-0.5 * (x / sigma)**2)
        return psf / np.sum(psf)
    
    def separate_components(self):
        """Separate spectral components using signal processing."""
        try:
            n_components = self.n_components_spin.value()
            
            # Method 1: Peak-based separation
            if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0 and self.fit_result:
                self.separate_by_peaks(n_components)
            else:
                # Method 2: Wavelet-based separation
                self.separate_by_wavelets(n_components)
            
            QMessageBox.information(self, "Success", 
                                  f"Separated spectrum into {len(self.components)} components.")
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Component separation failed: {str(e)}")
    
    def separate_by_peaks(self, n_components):
        """Separate components based on fitted peaks."""
        # FIX: Properly handle numpy array boolean check
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return
        
        self.components = []
        n_peaks = len(self.peaks)
        
        # Group peaks into components
        peaks_per_component = max(1, n_peaks // n_components)
        
        for i in range(n_components):
            component = np.zeros_like(self.wavenumbers)
            
            start_peak = i * peaks_per_component
            end_peak = min((i + 1) * peaks_per_component, n_peaks)
            
            for peak_idx in range(start_peak, end_peak):
                param_idx = peak_idx * 3
                if param_idx + 2 < len(self.fit_params):
                    amp, cen, wid = self.fit_params[param_idx:param_idx+3]
                    component += self.gaussian(self.wavenumbers, amp, cen, wid)
            
            self.components.append(component)
    
    def separate_by_wavelets(self, n_components):
        """Separate components using wavelet decomposition."""
        try:
            import pywt
            
            # Decompose signal using wavelets
            coeffs = pywt.wavedec(self.processed_intensities, 'db4', level=4)
            
            # Reconstruct components from different detail levels
            self.components = []
            
            # Low-frequency component (approximation)
            low_freq = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], 'db4')
            self.components.append(low_freq[:len(self.wavenumbers)])
            
            # High-frequency components (details)
            for i in range(1, min(len(coeffs), n_components)):
                detail_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
                detail_coeffs[i] = coeffs[i]
                detail = pywt.waverec(detail_coeffs, 'db4')
                self.components.append(detail[:len(self.wavenumbers)])
            
        except ImportError:
            # Fallback: Simple frequency filtering
            self.separate_by_frequency_filtering(n_components)
    
    def separate_by_frequency_filtering(self, n_components):
        """Separate components using frequency domain filtering."""
        # Simple frequency-based separation
        fft = np.fft.fft(self.processed_intensities)
        freqs = np.fft.fftfreq(len(self.processed_intensities))
        
        self.components = []
        
        # Divide frequency spectrum into bands
        freq_bands = np.linspace(0, 0.5, n_components + 1)
        
        for i in range(n_components):
            # Create filter for this frequency band
            mask = (np.abs(freqs) >= freq_bands[i]) & (np.abs(freqs) < freq_bands[i + 1])
            filtered_fft = fft.copy()
            filtered_fft[~mask] = 0
            
            # Convert back to time domain
            component = np.real(np.fft.ifft(filtered_fft))
            self.components.append(component)
    
    def perform_pca(self):
        """Perform Principal Component Analysis."""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "ML Not Available", 
                              "Install scikit-learn for PCA functionality.")
            return
        
        try:
            # Prepare data matrix (could use multiple spectra or sliding windows)
            # For single spectrum, use sliding window approach
            window_size = 50
            data_matrix = []
            
            for i in range(len(self.processed_intensities) - window_size + 1):
                window = self.processed_intensities[i:i + window_size]
                data_matrix.append(window)
            
            data_matrix = np.array(data_matrix)
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_matrix)
            
            # Perform PCA
            n_components = min(5, data_scaled.shape[1])
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(data_scaled)
            
            self.pca_result = pca
            
            # Display results
            results = "PCA Results:\n"
            results += f"Components: {n_components}\n"
            results += f"Explained variance: {pca.explained_variance_ratio_[:3]}\n"
            results += f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.2%}\n"
            
            # Add component details
            results += "\nComponent Details:\n"
            for i, (ratio, component) in enumerate(zip(pca.explained_variance_ratio_[:3], pca.components_[:3])):
                results += f"PC{i+1}: {ratio:.1%} variance, peak at index {np.argmax(np.abs(component))}\n"
            
            self.results_text.setPlainText(results)
            self.update_plot()
            
            QMessageBox.information(self, "PCA Complete", 
                                  f"PCA analysis complete. {n_components} components extracted.")
            
        except Exception as e:
            QMessageBox.critical(self, "PCA Error", f"PCA analysis failed: {str(e)}")
    
    def perform_nmf(self):
        """Perform Non-negative Matrix Factorization."""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "ML Not Available", 
                              "Install scikit-learn for NMF functionality.")
            return
        
        try:
            # Prepare data matrix
            window_size = 50
            data_matrix = []
            
            for i in range(len(self.processed_intensities) - window_size + 1):
                window = self.processed_intensities[i:i + window_size]
                # Ensure non-negative values for NMF
                window = np.maximum(window, 0)
                data_matrix.append(window)
            
            data_matrix = np.array(data_matrix)
            
            # Perform NMF
            n_components = self.n_components_spin.value()
            nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
            transformed = nmf.fit_transform(data_matrix)
            
            self.nmf_result = nmf
            
            # Display results
            results = "NMF Results:\n"
            results += f"Components: {n_components}\n"
            results += f"Reconstruction error: {nmf.reconstruction_err_:.6f}\n"
            results += f"Iterations: {nmf.n_iter_}\n"
            
            # Add component details
            results += "\nComponent Details:\n"
            for i, component in enumerate(nmf.components_[:3]):
                max_idx = np.argmax(component)
                results += f"NMF{i+1}: peak contribution at index {max_idx}, max value {component[max_idx]:.3f}\n"
            
            self.results_text.setPlainText(results)
            self.update_plot()
            
            QMessageBox.information(self, "NMF Complete", 
                                  f"NMF analysis complete. {n_components} components extracted.")
            
        except Exception as e:
            QMessageBox.critical(self, "NMF Error", f"NMF analysis failed: {str(e)}")
    
    def export_results(self):
        """Export analysis results."""
        try:
            from PySide6.QtWidgets import QFileDialog
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", 
                "CSV files (*.csv);;Text files (*.txt);;All files (*.*)"
            )
            
            if file_path:
                self.export_to_file(file_path)
                QMessageBox.information(self, "Export Complete", 
                                      f"Results exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")
    
    def export_to_file(self, file_path):
        """Export results to specified file."""
        data = {
            'Wavenumber': self.wavenumbers,
            'Original_Intensity': self.original_intensities,
            'Processed_Intensity': self.processed_intensities
        }
        
        # Add background if available
        if self.background is not None:
            data['Background'] = self.background
        
        # Add residuals if available
        if self.residuals is not None:
            data['Residuals'] = self.residuals
        
        # Add components if available
        for i, component in enumerate(self.components):
            data[f'Component_{i+1}'] = component
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, sep='\t', index=False)

    def update_peak_list(self):
        """Update the peak list widget with current peaks."""
        if not hasattr(self, 'peak_list_widget'):
            return
        
        self.peak_list_widget.clear()
        
        # Add automatic peaks
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            valid_auto_peaks = self.validate_peak_indices(self.peaks)
            for i, peak_idx in enumerate(valid_auto_peaks):
                wavenumber = self.wavenumbers[peak_idx]
                intensity = self.processed_intensities[peak_idx]
                item_text = f"🔴 Auto Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I: {intensity:.1f})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, ('auto', peak_idx))
                self.peak_list_widget.addItem(item)
        
        # Add manual peaks
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
            for i, peak_idx in enumerate(valid_manual_peaks):
                wavenumber = self.wavenumbers[peak_idx]
                intensity = self.processed_intensities[peak_idx]
                item_text = f"🟢 Manual Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I: {intensity:.1f})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, ('manual', peak_idx))
                self.peak_list_widget.addItem(item)

    def remove_selected_peak(self):
        """Remove the selected peak from the list."""
        if not hasattr(self, 'peak_list_widget'):
            return
        
        current_item = self.peak_list_widget.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "No Selection", "Please select a peak to remove from the list.")
            return
        
        # Get peak data
        peak_data = current_item.data(Qt.UserRole)
        if peak_data is None:
            return
        
        peak_type, peak_idx = peak_data
        
        try:
            if peak_type == 'auto':
                # Remove from automatic peaks
                if hasattr(self, 'peaks') and self.peaks is not None:
                    peak_indices = list(self.peaks)
                    if peak_idx in peak_indices:
                        peak_indices.remove(peak_idx)
                        self.peaks = np.array(peak_indices)
                        
            elif peak_type == 'manual':
                # Remove from manual peaks
                if hasattr(self, 'manual_peaks') and self.manual_peaks is not None:
                    peak_indices = list(self.manual_peaks)
                    if peak_idx in peak_indices:
                        peak_indices.remove(peak_idx)
                        self.manual_peaks = np.array(peak_indices)
            
            # Update displays
            self.update_peak_count_display()
            self.update_peak_list()
            self.update_plot()
            
            # Show confirmation
            wavenumber = self.wavenumbers[peak_idx]
            QMessageBox.information(self, "Peak Removed", 
                                  f"Removed {peak_type} peak at {wavenumber:.1f} cm⁻¹")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove peak: {str(e)}")


# Launch function for integration with main app
def launch_spectral_deconvolution(parent, wavenumbers, intensities):
    """Launch the spectral deconvolution window."""
    dialog = SpectralDeconvolutionQt6(parent, wavenumbers, intensities)
    dialog.exec() 