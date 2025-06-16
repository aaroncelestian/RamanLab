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
from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme

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
        
        # Background update timer for responsive live preview
        self.bg_update_timer = QTimer()
        self.bg_update_timer.setSingleShot(True)
        self.bg_update_timer.timeout.connect(self._update_background_calculation)
        self.bg_update_delay = 150  # milliseconds
        
        # Peak detection timer for responsive live preview
        self.peak_update_timer = QTimer()
        self.peak_update_timer.setSingleShot(True)
        self.peak_update_timer.timeout.connect(self._update_peak_detection)
        self.peak_update_delay = 100  # milliseconds
        
        # Plot line references for efficient updates
        self.spectrum_line = None
        self.background_line = None
        self.fitted_line = None
        self.auto_peaks_scatter = None
        self.manual_peaks_scatter = None
        self.individual_peak_lines = []
        self.filter_preview_line = None
        
        # Fourier analysis data storage
        self.fft_data = None
        self.fft_frequencies = None
        self.fft_magnitude = None
        self.fft_phase = None
        
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
        self.bg_method_combo.currentTextChanged.connect(self._trigger_background_update)
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
        
        # Connect sliders to update labels and live preview with debouncing
        self.lambda_slider.valueChanged.connect(self.update_lambda_label)
        self.p_slider.valueChanged.connect(self.update_p_label)
        self.niter_slider.valueChanged.connect(self.update_niter_label)
        
        # Connect sliders to debounced live preview
        self.lambda_slider.valueChanged.connect(self._trigger_background_update)
        self.p_slider.valueChanged.connect(self._trigger_background_update)
        self.niter_slider.valueChanged.connect(self._trigger_background_update)
        
        bg_layout.addWidget(self.als_params_widget)
        
        # Linear parameters
        self.linear_params_widget = QWidget()
        linear_params_layout = QVBoxLayout(self.linear_params_widget)
        linear_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Start and end point weighting
        start_weight_layout = QHBoxLayout()
        start_weight_layout.addWidget(QLabel("Start Point Weight:"))
        self.start_weight_slider = QSlider(Qt.Horizontal)
        self.start_weight_slider.setRange(1, 20)  # 0.1 to 2.0
        self.start_weight_slider.setValue(10)  # 1.0 (default)
        start_weight_layout.addWidget(self.start_weight_slider)
        self.start_weight_label = QLabel("1.0")
        start_weight_layout.addWidget(self.start_weight_label)
        linear_params_layout.addLayout(start_weight_layout)
        
        end_weight_layout = QHBoxLayout()
        end_weight_layout.addWidget(QLabel("End Point Weight:"))
        self.end_weight_slider = QSlider(Qt.Horizontal)
        self.end_weight_slider.setRange(1, 20)  # 0.1 to 2.0
        self.end_weight_slider.setValue(10)  # 1.0 (default)
        end_weight_layout.addWidget(self.end_weight_slider)
        self.end_weight_label = QLabel("1.0")
        end_weight_layout.addWidget(self.end_weight_label)
        linear_params_layout.addLayout(end_weight_layout)
        
        # Connect sliders to update labels and live preview
        self.start_weight_slider.valueChanged.connect(self.update_start_weight_label)
        self.end_weight_slider.valueChanged.connect(self.update_end_weight_label)
        self.start_weight_slider.valueChanged.connect(self._trigger_background_update)
        self.end_weight_slider.valueChanged.connect(self._trigger_background_update)
        
        bg_layout.addWidget(self.linear_params_widget)
        
        # Polynomial parameters
        self.poly_params_widget = QWidget()
        poly_params_layout = QVBoxLayout(self.poly_params_widget)
        poly_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Polynomial order
        poly_order_layout = QHBoxLayout()
        poly_order_layout.addWidget(QLabel("Polynomial Order:"))
        self.poly_order_slider = QSlider(Qt.Horizontal)
        self.poly_order_slider.setRange(1, 6)
        self.poly_order_slider.setValue(2)  # Default order 2
        poly_order_layout.addWidget(self.poly_order_slider)
        self.poly_order_label = QLabel("2")
        poly_order_layout.addWidget(self.poly_order_label)
        poly_params_layout.addLayout(poly_order_layout)
        
        # Fitting method
        poly_method_layout = QHBoxLayout()
        poly_method_layout.addWidget(QLabel("Fitting Method:"))
        self.poly_method_combo = QComboBox()
        self.poly_method_combo.addItems(["Least Squares", "Robust"])
        poly_method_layout.addWidget(self.poly_method_combo)
        poly_params_layout.addLayout(poly_method_layout)
        
        # Connect controls to live preview
        self.poly_order_slider.valueChanged.connect(self.update_poly_order_label)
        self.poly_order_slider.valueChanged.connect(self._trigger_background_update)
        self.poly_method_combo.currentTextChanged.connect(self._trigger_background_update)
        
        bg_layout.addWidget(self.poly_params_widget)
        
        # Moving Average parameters
        self.moving_avg_params_widget = QWidget()
        moving_avg_params_layout = QVBoxLayout(self.moving_avg_params_widget)
        moving_avg_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Window size
        window_size_layout = QHBoxLayout()
        window_size_layout.addWidget(QLabel("Window Size (%):"))
        self.window_size_slider = QSlider(Qt.Horizontal)
        self.window_size_slider.setRange(1, 50)  # 1% to 50% of spectrum length
        self.window_size_slider.setValue(10)  # 10% default
        window_size_layout.addWidget(self.window_size_slider)
        self.window_size_label = QLabel("10%")
        window_size_layout.addWidget(self.window_size_label)
        moving_avg_params_layout.addLayout(window_size_layout)
        
        # Window type
        window_type_layout = QHBoxLayout()
        window_type_layout.addWidget(QLabel("Window Type:"))
        self.window_type_combo = QComboBox()
        self.window_type_combo.addItems(["Uniform", "Gaussian", "Hann", "Hamming"])
        window_type_layout.addWidget(self.window_type_combo)
        moving_avg_params_layout.addLayout(window_type_layout)
        
        # Connect controls to live preview
        self.window_size_slider.valueChanged.connect(self.update_window_size_label)
        self.window_size_slider.valueChanged.connect(self._trigger_background_update)
        self.window_type_combo.currentTextChanged.connect(self._trigger_background_update)
        
        bg_layout.addWidget(self.moving_avg_params_widget)
        
        # Initially hide all parameter widgets except ALS (default)
        self.linear_params_widget.setVisible(False)
        self.poly_params_widget.setVisible(False)
        self.moving_avg_params_widget.setVisible(False)
        
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
        
        # Connect sliders to update labels
        self.height_slider.valueChanged.connect(self.update_height_label)
        self.distance_slider.valueChanged.connect(self.update_distance_label)
        self.prominence_slider.valueChanged.connect(self.update_prominence_label)
        
        # Connect sliders to live peak detection with debouncing
        self.height_slider.valueChanged.connect(self._trigger_peak_update)
        self.distance_slider.valueChanged.connect(self._trigger_peak_update)
        self.prominence_slider.valueChanged.connect(self._trigger_peak_update)
        
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
        
        self.show_legend_check = QCheckBox("Show Legend")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_legend_check)
        
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_grid_check)
        
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
        """Create spectral deconvolution tab with Fourier-based analysis."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Fourier Analysis
        fourier_group = QGroupBox("Fourier Transform Analysis")
        fourier_layout = QVBoxLayout(fourier_group)
        
        # Fourier transform display
        fft_btn = QPushButton("Show Frequency Spectrum")
        fft_btn.clicked.connect(self.show_frequency_spectrum)
        fft_btn.setStyleSheet("""
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
        fourier_layout.addWidget(fft_btn)
        
        # Power spectral density
        psd_btn = QPushButton("Power Spectral Density")
        psd_btn.clicked.connect(self.show_power_spectral_density)
        fourier_layout.addWidget(psd_btn)
        
        layout.addWidget(fourier_group)
        
        # Fourier Filtering
        filter_group = QGroupBox("Fourier Filtering")
        filter_layout = QVBoxLayout(filter_group)
        
        # Filter type selection
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Filter Type:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Low-pass", "High-pass", "Band-pass", "Band-stop", "Butterworth Low-pass", "Butterworth High-pass", "Butterworth Band-pass", "Butterworth Band-stop"])
        self.filter_type_combo.currentTextChanged.connect(self.on_filter_type_changed)
        filter_type_layout.addWidget(self.filter_type_combo)
        filter_layout.addLayout(filter_type_layout)
        
        # Cutoff frequency controls
        cutoff_layout = QFormLayout()
        
        # Low cutoff (for band filters)
        self.low_cutoff_slider = QSlider(Qt.Horizontal)
        self.low_cutoff_slider.setRange(1, 50)
        self.low_cutoff_slider.setValue(10)
        self.low_cutoff_label = QLabel("10%")
        low_cutoff_layout = QHBoxLayout()
        low_cutoff_layout.addWidget(self.low_cutoff_slider)
        low_cutoff_layout.addWidget(self.low_cutoff_label)
        cutoff_layout.addRow("Low Cutoff:", low_cutoff_layout)
        
        # High cutoff
        self.high_cutoff_slider = QSlider(Qt.Horizontal)
        self.high_cutoff_slider.setRange(10, 90)
        self.high_cutoff_slider.setValue(50)
        self.high_cutoff_label = QLabel("50%")
        high_cutoff_layout = QHBoxLayout()
        high_cutoff_layout.addWidget(self.high_cutoff_slider)
        high_cutoff_layout.addWidget(self.high_cutoff_label)
        cutoff_layout.addRow("High Cutoff:", high_cutoff_layout)
        
        # Butterworth filter order (initially hidden)
        self.butterworth_order_slider = QSlider(Qt.Horizontal)
        self.butterworth_order_slider.setRange(1, 10)
        self.butterworth_order_slider.setValue(4)
        self.butterworth_order_label = QLabel("4")
        butterworth_order_layout = QHBoxLayout()
        butterworth_order_layout.addWidget(self.butterworth_order_slider)
        butterworth_order_layout.addWidget(self.butterworth_order_label)
        self.butterworth_order_row = cutoff_layout.addRow("Butterworth Order:", butterworth_order_layout)
        
        # Initially hide Butterworth order control
        self.butterworth_order_slider.setVisible(False)
        self.butterworth_order_label.setVisible(False)
        
        # Connect sliders to update labels
        self.low_cutoff_slider.valueChanged.connect(self.update_low_cutoff_label)
        self.high_cutoff_slider.valueChanged.connect(self.update_high_cutoff_label)
        self.butterworth_order_slider.valueChanged.connect(self.update_butterworth_order_label)
        
        filter_layout.addLayout(cutoff_layout)
        
        # Filter buttons
        filter_buttons_layout = QHBoxLayout()
        
        preview_filter_btn = QPushButton("Preview Filter")
        preview_filter_btn.clicked.connect(self.preview_fourier_filter)
        filter_buttons_layout.addWidget(preview_filter_btn)
        
        apply_filter_btn = QPushButton("Apply Filter")
        apply_filter_btn.clicked.connect(self.apply_fourier_filter)
        filter_buttons_layout.addWidget(apply_filter_btn)
        
        filter_layout.addLayout(filter_buttons_layout)
        
        layout.addWidget(filter_group)
        
        # Fourier Enhancement
        enhance_group = QGroupBox("Fourier Enhancement")
        enhance_layout = QVBoxLayout(enhance_group)
        
        # Smoothing
        smooth_btn = QPushButton("Fourier Smoothing")
        smooth_btn.clicked.connect(self.apply_fourier_smoothing)
        enhance_layout.addWidget(smooth_btn)
        
        # Deconvolution
        deconv_btn = QPushButton("Richardson-Lucy Deconvolution")
        deconv_btn.clicked.connect(self.apply_richardson_lucy)
        enhance_layout.addWidget(deconv_btn)
        
        # Apodization
        apod_btn = QPushButton("Apply Apodization")
        apod_btn.clicked.connect(self.apply_apodization)
        enhance_layout.addWidget(apod_btn)
        
        layout.addWidget(enhance_group)
        
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
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Peak", "Position", "Amplitude", "Width", "R²"
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
    
    # Debounced update methods for responsive live preview
    def _trigger_background_update(self):
        """Trigger debounced background update."""
        self.bg_update_timer.stop()
        self.bg_update_timer.start(self.bg_update_delay)
    
    def _trigger_peak_update(self):
        """Trigger debounced peak detection update."""
        self.peak_update_timer.stop()
        self.peak_update_timer.start(self.peak_update_delay)
    
    def _update_background_calculation(self):
        """Perform background calculation and update plot efficiently."""
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
                # Linear background subtraction with weighted endpoints
                start_weight = self.start_weight_slider.value() / 10.0
                end_weight = self.end_weight_slider.value() / 10.0
                
                # Create weighted linear background
                start_val = self.original_intensities[0] * start_weight
                end_val = self.original_intensities[-1] * end_weight
                background = np.linspace(start_val, end_val, len(self.original_intensities))
                self.background = background
                
            elif method == "Polynomial":
                # Polynomial background fitting with adjustable order and method
                x = np.arange(len(self.original_intensities))
                poly_order = self.poly_order_slider.value()
                poly_method = self.poly_method_combo.currentText()
                
                if poly_method == "Robust":
                    # Use robust polynomial fitting (less sensitive to outliers)
                    try:
                        from numpy.polynomial import polynomial as P
                        # Robust fitting using iterative reweighting
                        coeffs = np.polyfit(x, self.original_intensities, poly_order)
                        background = np.polyval(coeffs, x)
                        
                        # Apply robust reweighting
                        for _ in range(3):  # 3 iterations of robust fitting
                            residuals = np.abs(self.original_intensities - background)
                            weights = 1.0 / (1.0 + residuals / np.median(residuals))
                            coeffs = np.polyfit(x, self.original_intensities, poly_order, w=weights)
                            background = np.polyval(coeffs, x)
                    except:
                        # Fallback to regular fitting
                        coeffs = np.polyfit(x, self.original_intensities, poly_order)
                        background = np.polyval(coeffs, x)
                else:
                    # Regular least squares fitting
                    coeffs = np.polyfit(x, self.original_intensities, poly_order)
                    background = np.polyval(coeffs, x)
                
                self.background = background
                
            elif method == "Moving Average":
                # Moving average background with adjustable window and type
                window_percent = self.window_size_slider.value()
                window_type = self.window_type_combo.currentText()
                
                # Calculate window size as percentage of spectrum length
                window_size = max(int(len(self.original_intensities) * window_percent / 100.0), 3)
                
                if window_type == "Uniform":
                    # Simple uniform moving average
                    from scipy import ndimage
                    background = ndimage.uniform_filter1d(self.original_intensities, size=window_size)
                elif window_type == "Gaussian":
                    # Gaussian-weighted moving average
                    from scipy import ndimage
                    sigma = window_size / 4.0  # Standard deviation
                    background = ndimage.gaussian_filter1d(self.original_intensities, sigma=sigma)
                elif window_type in ["Hann", "Hamming"]:
                    # Windowed moving average
                    if window_type == "Hann":
                        window = np.hanning(window_size)
                    else:  # Hamming
                        window = np.hamming(window_size)
                    
                    window = window / np.sum(window)  # Normalize
                    background = np.convolve(self.original_intensities, window, mode='same')
                else:
                    # Fallback to uniform
                    from scipy import ndimage
                    background = ndimage.uniform_filter1d(self.original_intensities, size=window_size)
                
                self.background = background
            
            # Set preview flag
            self.background_preview_active = True
            
            # Efficient plot update - only update background line
            self._update_background_line()
            
        except Exception as e:
            print(f"Background preview error: {str(e)}")
    
    def _update_peak_detection(self):
        """Perform live peak detection with current slider values."""
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
            
            # Efficient plot update - only update peak markers
            self._update_peak_markers()
            
        except Exception as e:
            print(f"Live peak detection error: {str(e)}")
    
    def _update_background_line(self):
        """Efficiently update only the background line in the plot."""
        if self.background is not None and self.ax_main is not None:
            # Remove existing background line if it exists
            if self.background_line is not None:
                try:
                    self.background_line.remove()
                except:
                    pass
            
            # Add new background line
            self.background_line, = self.ax_main.plot(
                self.wavenumbers, self.background, 'r--', 
                linewidth=1, alpha=0.7, label='Background'
            )
            
            # Update legend
            self.ax_main.legend()
            
            # Redraw canvas
            self.canvas.draw_idle()
    
    def _update_peak_markers(self):
        """Efficiently update only the peak markers in the plot."""
        # Remove existing peak markers
        if self.auto_peaks_scatter is not None:
            try:
                self.auto_peaks_scatter.remove()
            except:
                pass
            self.auto_peaks_scatter = None
        
        # Add new auto peak markers
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            valid_auto_peaks = self.validate_peak_indices(self.peaks)
            if len(valid_auto_peaks) > 0:
                auto_peak_positions = [self.wavenumbers[p] for p in valid_auto_peaks]
                auto_peak_intensities = [self.processed_intensities[p] for p in valid_auto_peaks]
                self.auto_peaks_scatter = self.ax_main.scatter(
                    auto_peak_positions, auto_peak_intensities, 
                    c='red', s=64, marker='o', label='Auto Peaks', alpha=0.8, zorder=5
                )
        
        # Update legend
        self.ax_main.legend()
        
        # Update peak list
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        # Redraw canvas
        self.canvas.draw_idle()
    
    def _update_manual_peak_markers(self):
        """Efficiently update only the manual peak markers in the plot."""
        # Remove existing manual peak markers
        if self.manual_peaks_scatter is not None:
            try:
                self.manual_peaks_scatter.remove()
            except:
                pass
            self.manual_peaks_scatter = None
        
        # Add new manual peak markers
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
            if len(valid_manual_peaks) > 0:
                manual_peak_positions = [self.wavenumbers[p] for p in valid_manual_peaks]
                manual_peak_intensities = [self.processed_intensities[p] for p in valid_manual_peaks]
                self.manual_peaks_scatter = self.ax_main.scatter(
                    manual_peak_positions, manual_peak_intensities, 
                    c='green', s=100, marker='s', label='Manual Peaks', 
                    alpha=0.8, edgecolor='darkgreen', zorder=5
                )
        
        # Update legend
        self.ax_main.legend()
        
        # Update peak list
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        # Redraw canvas
        self.canvas.draw_idle()
        
    def update_plot(self):
        """Update all plots with line reference storage for efficient updates."""
        # Clear all axes and reset line references
        self.ax_main.clear()
        self.ax_residual.clear()
        self._reset_line_references()
        
        # Main spectrum plot
        self.spectrum_line, = self.ax_main.plot(self.wavenumbers, self.processed_intensities, 'b-', 
                                              linewidth=1.5, label='Spectrum')
        
        # Plot background if available
        if self.background is not None:
            self.background_line, = self.ax_main.plot(self.wavenumbers, self.background, 'r--', 
                                                    linewidth=1, alpha=0.7, label='Background')
        
        # Plot fitted peaks if available
        if self.fit_result is not None and self.show_individual_peaks:
            # Plot total fitted curve
            fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
            
            # Calculate total R² for the global fit
            total_r2 = self.calculate_total_r2()
            
            self.fitted_line, = self.ax_main.plot(self.wavenumbers, fitted_curve, 'g-', 
                                                linewidth=2, label=f'Total Fit (R²={total_r2:.4f})')
            
            # Plot individual peaks
            self.plot_individual_peaks()
        
        # Plot automatically detected peaks - Handle numpy array boolean check properly
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            # Validate peak indices before plotting
            valid_auto_peaks = self.validate_peak_indices(self.peaks)
            if len(valid_auto_peaks) > 0:
                auto_peak_positions = [self.wavenumbers[p] for p in valid_auto_peaks]
                auto_peak_intensities = [self.processed_intensities[p] for p in valid_auto_peaks]
                self.auto_peaks_scatter = self.ax_main.scatter(auto_peak_positions, auto_peak_intensities, 
                                                             c='red', s=64, marker='o', label='Auto Peaks', 
                                                             alpha=0.8, zorder=5)
        
        # Plot manually selected peaks - Validate indices to prevent IndexError
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            # Validate peak indices before plotting
            valid_manual_peaks = self.validate_peak_indices(self.manual_peaks)
            if len(valid_manual_peaks) > 0:
                manual_peak_positions = [self.wavenumbers[p] for p in valid_manual_peaks]
                manual_peak_intensities = [self.processed_intensities[p] for p in valid_manual_peaks]
                self.manual_peaks_scatter = self.ax_main.scatter(manual_peak_positions, manual_peak_intensities, 
                                                               c='green', s=100, marker='s', label='Manual Peaks', 
                                                               alpha=0.8, edgecolor='darkgreen', zorder=5)
        
        # Add interactive mode indicator
        if self.interactive_mode:
            self.ax_main.text(0.02, 0.98, '🖱️ Interactive Mode ON\nClick to select peaks', 
                             transform=self.ax_main.transAxes, 
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Spectrum and Peak Analysis')
        
        # Show/hide legend based on checkbox
        if hasattr(self, 'show_legend_check') and self.show_legend_check.isChecked():
            self.ax_main.legend()
        
        # Show/hide grid based on checkbox
        if hasattr(self, 'show_grid_check'):
            self.ax_main.grid(self.show_grid_check.isChecked(), alpha=0.3)
        else:
            self.ax_main.grid(True, alpha=0.3)  # Default behavior
        
        # Residuals plot
        if self.residuals is not None:
            self.ax_residual.plot(self.wavenumbers, self.residuals, 'k-', linewidth=1)
            self.ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_residual.set_ylabel('Residuals')
            self.ax_residual.set_title('Fit Residuals')
            
            # Show/hide grid based on checkbox for residuals plot too
            if hasattr(self, 'show_grid_check'):
                self.ax_residual.grid(self.show_grid_check.isChecked(), alpha=0.3)
            else:
                self.ax_residual.grid(True, alpha=0.3)  # Default behavior
        
        # Update peak list if it exists
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        self.canvas.draw()
    
    def _reset_line_references(self):
        """Reset all line references for clean plot updates."""
        self.spectrum_line = None
        self.background_line = None
        self.fitted_line = None
        self.auto_peaks_scatter = None
        self.manual_peaks_scatter = None
        self.individual_peak_lines = []
        self.filter_preview_line = None

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
        """Plot individual fitted peaks with different colors and R² values."""
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
        
        # Define a color palette for individual peaks
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        # Calculate individual R² values for each peak
        individual_r2_values = self.calculate_individual_r2_values()
        
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
                
                # Select color from palette (cycle if more peaks than colors)
                color = colors[i % len(colors)]
                
                # Get individual R² value
                r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
                # Create label with peak info and R²
                label = f'Peak {i+1} ({cen:.1f} cm⁻¹, R²={r2_value:.3f})'
                
                # Plot individual peak
                line, = self.ax_main.plot(self.wavenumbers, peak_curve, '--', 
                                        linewidth=1.5, alpha=0.8, color=color,
                                        label=label)
                self.individual_peak_lines.append(line)
                
                # Add peak position label on the plot
                peak_max_idx = np.argmax(peak_curve)
                peak_max_intensity = peak_curve[peak_max_idx]
                
                # Offset label slightly above the peak
                label_y = peak_max_intensity + np.max(self.processed_intensities) * 0.05
                
                self.ax_main.annotate(f'{cen:.1f}', 
                                    xy=(cen, peak_max_intensity),
                                    xytext=(cen, label_y),
                                    ha='center', va='bottom',
                                    fontsize=9, fontweight='bold',
                                    color=color,
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='white', 
                                            edgecolor=color, 
                                            alpha=0.8),
                                    arrowprops=dict(arrowstyle='->', 
                                                  color=color, 
                                                  alpha=0.6, 
                                                  lw=1))
    
    def calculate_individual_r2_values(self):
        """Calculate R² values for individual peaks using a straightforward regional method."""
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return []
        
        all_fitted_peaks = self.get_all_peaks_for_fitting()
        if len(all_fitted_peaks) == 0:
            return []
        
        validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
        n_peaks = len(validated_peaks)
        model = self.current_model
        
        individual_r2_values = []
        
        # Generate total fit for reference
        total_fit = self.multi_peak_model(self.wavenumbers, *self.fit_params)
        
        # Calculate R² for each peak in its local region
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
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                
                # Calculate R² for this peak in a focused region
                individual_r2 = self._calculate_simple_regional_r2(cen, wid, peak_curve, total_fit)
                individual_r2_values.append(individual_r2)
            else:
                individual_r2_values.append(0.0)
        
        return individual_r2_values
    
    def _calculate_simple_regional_r2(self, peak_center, peak_width, peak_curve, total_fit):
        """Calculate R² for a peak using a simple, focused regional approach."""
        try:
            # Define region around peak center (3 times the width, minimum 20 cm⁻¹)
            region_width = max(abs(peak_width) * 3, 20)
            region_start = peak_center - region_width
            region_end = peak_center + region_width
            
            # Find indices within this region
            region_mask = (self.wavenumbers >= region_start) & (self.wavenumbers <= region_end)
            
            if not np.any(region_mask) or np.sum(region_mask) < 5:
                return 0.0
            
            # Extract regional data
            region_data = self.processed_intensities[region_mask]
            region_total_fit = total_fit[region_mask]
            region_individual_peak = peak_curve[region_mask]
            
            # Method: Compare how much the individual peak contributes to the total fit quality in this region
            # Calculate what the fit would be WITHOUT this peak
            fit_without_this_peak = region_total_fit - region_individual_peak
            
            # Calculate residuals with and without this peak
            residuals_with_peak = region_data - region_total_fit
            residuals_without_peak = region_data - fit_without_this_peak
            
            # R² represents how much this peak improves the fit
            ss_res_with = np.sum(residuals_with_peak ** 2)
            ss_res_without = np.sum(residuals_without_peak ** 2)
            
            # The improvement ratio gives us the individual peak R²
            if ss_res_without > 0:
                improvement_ratio = (ss_res_without - ss_res_with) / ss_res_without
                r2 = max(0.0, min(1.0, improvement_ratio))
            else:
                # If no improvement possible, calculate direct correlation
                r2 = self._calculate_correlation_r2(region_data, region_individual_peak, region_total_fit)
            
            return r2
            
        except Exception as e:
            return 0.0
    
    def _calculate_correlation_r2(self, region_data, peak_curve, total_fit):
        """Calculate R² based on how well the peak correlates with the data in its region."""
        try:
            # Remove baseline trend
            baseline = np.linspace(region_data[0], region_data[-1], len(region_data))
            data_corrected = region_data - baseline
            
            # Scale peak to match the data magnitude in this region
            if np.max(peak_curve) > 0:
                peak_scaled = peak_curve * (np.max(data_corrected) / np.max(peak_curve))
            else:
                return 0.0
            
            # Calculate correlation-based R²
            mean_data = np.mean(data_corrected)
            ss_tot = np.sum((data_corrected - mean_data) ** 2)
            
            if ss_tot > 0:
                # Use the scaled peak as the model
                ss_res = np.sum((data_corrected - peak_scaled) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                return max(0.0, min(1.0, r2))
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_regional_r2(self, peak_index, peak_curve, all_individual_peaks):
        """Calculate R² for a peak in its local region, accounting for overlaps."""
        try:
            start_idx = peak_index * 3
            if start_idx + 2 >= len(self.fit_params):
                return 0.0
                
            amp, cen, wid = self.fit_params[start_idx:start_idx+3]
            
            # Define a region around this peak (2.5 sigma to reduce overlap issues)
            peak_width = abs(wid) * 2.5
            region_mask = (self.wavenumbers >= cen - peak_width) & \
                         (self.wavenumbers <= cen + peak_width)
            
            if not np.any(region_mask):
                return 0.0
            
            # Extract regional data
            region_data = self.processed_intensities[region_mask]
            region_peak = peak_curve[region_mask]
            
            # Calculate baseline for this region (linear interpolation)
            region_wavenumbers = self.wavenumbers[region_mask]
            if len(region_wavenumbers) < 3:
                return 0.0
                
            baseline = np.linspace(region_data[0], region_data[-1], len(region_data))
            region_data_corrected = region_data - baseline
            
            # Calculate R² comparing peak to baseline-corrected data
            mean_data = np.mean(region_data_corrected)
            ss_tot = np.sum((region_data_corrected - mean_data) ** 2)
            
            if ss_tot > 0:
                # For overlapping regions, subtract contributions from other significant peaks
                other_peaks_contribution = np.zeros_like(region_peak)
                for j, other_peak in enumerate(all_individual_peaks):
                    if j != peak_index:
                        other_region_peak = other_peak[region_mask]
                        # Only subtract if the other peak contributes significantly in this region
                        if np.max(other_region_peak) > np.max(region_peak) * 0.1:
                            other_peaks_contribution += other_region_peak
                
                # Adjusted data = original - other peaks
                adjusted_data = region_data_corrected - other_peaks_contribution
                
                ss_res = np.sum((adjusted_data - region_peak) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                return max(0.0, min(1.0, r2))
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def calculate_total_r2(self):
        """Calculate total R² for the global fit."""
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0):
            return 0.0
        
        # Generate total fitted curve
        fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
        
        # Calculate R² for global fit
        ss_res = np.sum((self.processed_intensities - fitted_curve) ** 2)
        ss_tot = np.sum((self.processed_intensities - np.mean(self.processed_intensities)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Ensure R² is reasonable (between 0 and 1)
        return max(0.0, min(1.0, r2))

    def plot_analysis_results(self):
        """Plot PCA/NMF analysis results on the main plot."""
        # Note: This method is kept for compatibility but analysis results
        # will now be displayed on the main plot when PCA/NMF methods are called
        pass

    # Background subtraction methods
    def on_bg_method_changed(self):
        """Handle change in background method."""
        method = self.bg_method_combo.currentText()
        
        # Show/hide parameter widgets based on selected method
        self.als_params_widget.setVisible(method.startswith("ALS"))
        self.linear_params_widget.setVisible(method == "Linear")
        self.poly_params_widget.setVisible(method == "Polynomial")
        self.moving_avg_params_widget.setVisible(method == "Moving Average")
        
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
    
    def update_start_weight_label(self):
        """Update the start weight label based on the slider value."""
        value = self.start_weight_slider.value()
        weight_value = value / 10.0  # Convert to 0.1-2.0 range
        self.start_weight_label.setText(f"{weight_value:.1f}")
    
    def update_end_weight_label(self):
        """Update the end weight label based on the slider value."""
        value = self.end_weight_slider.value()
        weight_value = value / 10.0  # Convert to 0.1-2.0 range
        self.end_weight_label.setText(f"{weight_value:.1f}")
    
    def update_poly_order_label(self):
        """Update the polynomial order label based on the slider value."""
        value = self.poly_order_slider.value()
        self.poly_order_label.setText(str(value))
    
    def update_window_size_label(self):
        """Update the window size label based on the slider value."""
        value = self.window_size_slider.value()
        self.window_size_label.setText(f"{value}%")

    def clear_background_preview(self):
        """Clear the background preview."""
        self.background = None
        self.background_preview_active = False
        self.update_plot()

    def preview_background(self):
        """Preview background subtraction (legacy method)."""
        self._trigger_background_update()
    
    def apply_background(self):
        """Apply background subtraction."""
        if self.background is not None:
            self.processed_intensities = self.original_intensities - self.background
            # Clear preview flag
            self.background_preview_active = False
            self.update_plot()
        else:
            # If no preview, generate background first
            self._update_background_calculation()
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
        """Detect peaks in the spectrum (legacy method - now calls debounced detection)."""
        self._trigger_peak_update()
    
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
            # Use efficient updates for interactive changes
            self._update_peak_markers()
            self._update_manual_peak_markers()
            
        except Exception as e:
            print(f"Error in interactive peak selection: {e}")

    def clear_manual_peaks(self):
        """Clear only manually selected peaks."""
        self.manual_peaks = np.array([])
        self.update_peak_count_display()
        self._update_manual_peak_markers()

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
        self._update_peak_markers()
        self._update_manual_peak_markers()
        
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
                
                # Use the more accurate total R² calculation for display
                total_r2 = self.calculate_total_r2()
                self.display_fit_results(total_r2, all_peaks)
                self.update_results_table()
                
                QMessageBox.information(self, "Success", 
                                      f"Peak fitting completed successfully!\\n"
                                      f"Total R² = {total_r2:.4f}\\n"
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
        """Display fitting results with individual and total R² values."""
        if fitted_peaks is None:
            fitted_peaks = self.get_all_peaks_for_fitting()
        
        # Ensure fitted_peaks are validated
        validated_fitted_peaks = []
        if isinstance(fitted_peaks, (list, np.ndarray)) and len(fitted_peaks) > 0:
            validated_fitted_peaks = self.validate_peak_indices(np.array(fitted_peaks))
        
        # Calculate individual R² values
        individual_r2_values = self.calculate_individual_r2_values()
        total_r2 = self.calculate_total_r2()
        
        results = f"Peak Fitting Results\n{'='*30}\n\n"
        results += f"Model: {self.current_model}\n"
        results += f"Number of peaks fitted: {len(validated_fitted_peaks)}\n"
        results += f"Total R² = {total_r2:.4f}\n\n"
        
        # Handle numpy array boolean check properly
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
                    
                    # Get individual R² value
                    individual_r2 = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                    
                    # Determine peak type safely
                    peak_type = "Auto"
                    if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                        validated_auto_peaks = self.validate_peak_indices(self.peaks)
                        if len(validated_auto_peaks) > 0 and validated_fitted_peaks[i] not in validated_auto_peaks.tolist():
                            peak_type = "Manual"
                    
                    results += (f"Peak {i+1} ({peak_type}): Center={cen:.1f} cm⁻¹, "
                              f"Amplitude={amp:.1f}, Width={wid:.1f}, R²={individual_r2:.3f}\n")
            
            # Add summary statistics
            if len(individual_r2_values) > 0:
                avg_individual_r2 = np.mean(individual_r2_values)
                results += f"\nAverage Individual R²: {avg_individual_r2:.3f}\n"
        
        self.results_text.setPlainText(results)

    def update_results_table(self):
        """Update the results table with R² values."""
        # Properly handle numpy array boolean check
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
        
        # Calculate individual R² values
        individual_r2_values = self.calculate_individual_r2_values()
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                
                # Get individual R² value
                individual_r2 = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
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
                self.results_table.setItem(i, 4, QTableWidgetItem(f"{individual_r2:.3f}"))

         # Fourier Transform Analysis Methods
    def show_frequency_spectrum(self):
        """Show the frequency spectrum of the processed intensities."""
        try:
            # Calculate FFT
            fft_data = np.fft.fft(self.processed_intensities)
            fft_magnitude = np.abs(fft_data)
            fft_phase = np.angle(fft_data)
            
            # Create frequency array (normalized to Nyquist frequency)
            n_points = len(self.processed_intensities)
            frequencies = np.fft.fftfreq(n_points, d=1.0)[:n_points//2]  # Only positive frequencies
            magnitude_spectrum = fft_magnitude[:n_points//2]
            phase_spectrum = fft_phase[:n_points//2]
            
            # Store for other operations
            self.fft_data = fft_data
            self.fft_frequencies = frequencies
            self.fft_magnitude = magnitude_spectrum
            self.fft_phase = phase_spectrum
            
            # Create a new plot window or update residual plot
            self.ax_residual.clear()
            
            # Plot magnitude spectrum
            self.ax_residual.semilogy(frequencies, magnitude_spectrum, 'b-', linewidth=1.5, label='Magnitude')
            self.ax_residual.set_xlabel('Normalized Frequency')
            self.ax_residual.set_ylabel('Magnitude (log scale)')
            self.ax_residual.set_title('Frequency Spectrum (FFT Magnitude)')
            self.ax_residual.grid(True, alpha=0.3)
            self.ax_residual.legend()
            
            # Display results
            dominant_freq_idx = np.argmax(magnitude_spectrum[1:]) + 1  # Skip DC component
            dominant_freq = frequencies[dominant_freq_idx]
            max_magnitude = magnitude_spectrum[dominant_freq_idx]
            
            results = "Frequency Spectrum Analysis:\n"
            results += f"Total data points: {n_points}\n"
            results += f"Frequency resolution: {frequencies[1]:.6f}\n"
            results += f"DC component magnitude: {magnitude_spectrum[0]:.2f}\n"
            results += f"Dominant frequency: {dominant_freq:.4f}\n"
            results += f"Dominant magnitude: {max_magnitude:.2f}\n"
            results += f"Total spectral energy: {np.sum(magnitude_spectrum**2):.2e}\n"
            
            self.results_text.setPlainText(results)
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "FFT Error", f"Frequency spectrum calculation failed: {str(e)}")
    
    def show_power_spectral_density(self):
        """Show the power spectral density."""
        try:
            # Calculate power spectral density
            if not hasattr(self, 'fft_data'):
                self.show_frequency_spectrum()  # Calculate FFT first
            
            # PSD is the square of the magnitude spectrum
            psd = self.fft_magnitude ** 2
            
            # Normalize by frequency resolution and total power
            psd_normalized = psd / (len(self.processed_intensities) * np.sum(psd))
            
            # Plot PSD
            self.ax_residual.clear()
            self.ax_residual.semilogy(self.fft_frequencies, psd_normalized, 'r-', linewidth=1.5, label='Power Spectral Density')
            self.ax_residual.set_xlabel('Normalized Frequency')
            self.ax_residual.set_ylabel('Power Density (log scale)')
            self.ax_residual.set_title('Power Spectral Density')
            self.ax_residual.grid(True, alpha=0.3)
            self.ax_residual.legend()
            
            # Calculate spectral statistics
            total_power = np.sum(psd)
            mean_freq = np.sum(self.fft_frequencies * psd) / total_power
            spectral_centroid = np.sum(self.fft_frequencies * psd) / np.sum(psd)
            
            # Spectral bandwidth (standard deviation)
            spectral_bandwidth = np.sqrt(np.sum(((self.fft_frequencies - spectral_centroid) ** 2) * psd) / np.sum(psd))
            
            results = "Power Spectral Density Analysis:\n"
            results += f"Total power: {total_power:.2e}\n"
            results += f"Spectral centroid: {spectral_centroid:.4f}\n"
            results += f"Spectral bandwidth: {spectral_bandwidth:.4f}\n"
            results += f"Peak PSD frequency: {self.fft_frequencies[np.argmax(psd)]:.4f}\n"
            results += f"Peak PSD value: {np.max(psd):.2e}\n"
            
            self.results_text.setPlainText(results)
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "PSD Error", f"Power spectral density calculation failed: {str(e)}")
    
    # Fourier Filter Label Update Methods
    def update_low_cutoff_label(self):
        """Update low cutoff frequency label."""
        value = self.low_cutoff_slider.value()
        self.low_cutoff_label.setText(f"{value}%")
    
    def update_high_cutoff_label(self):
        """Update high cutoff frequency label."""
        value = self.high_cutoff_slider.value()
        self.high_cutoff_label.setText(f"{value}%")
    
    def update_butterworth_order_label(self):
        """Update Butterworth filter order label."""
        value = self.butterworth_order_slider.value()
        self.butterworth_order_label.setText(str(value))
    
    def on_filter_type_changed(self):
        """Handle filter type change to show/hide Butterworth order control."""
        filter_type = self.filter_type_combo.currentText()
        is_butterworth = "Butterworth" in filter_type
        
        # Show/hide Butterworth order control
        self.butterworth_order_slider.setVisible(is_butterworth)
        self.butterworth_order_label.setVisible(is_butterworth)
    
    def preview_fourier_filter(self):
        """Preview the effect of Fourier filtering."""
        try:
            filtered_spectrum = self._apply_fourier_filter_internal(preview_only=True)
            
            # Update main plot with preview
            if hasattr(self, 'filter_preview_line') and self.filter_preview_line is not None:
                try:
                    self.filter_preview_line.remove()
                except:
                    pass
            
            self.filter_preview_line, = self.ax_main.plot(
                self.wavenumbers, filtered_spectrum, 'orange', 
                linewidth=2, alpha=0.7, linestyle='--', label='Filter Preview'
            )
            self.ax_main.legend()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Preview Error", f"Filter preview failed: {str(e)}")
    
    def apply_fourier_filter(self):
        """Apply the Fourier filter to the spectrum."""
        try:
            filtered_spectrum = self._apply_fourier_filter_internal(preview_only=False)
            
            # Apply to processed intensities
            self.processed_intensities = filtered_spectrum.copy()
            
            # Clear any existing filter preview
            if hasattr(self, 'filter_preview_line') and self.filter_preview_line is not None:
                try:
                    self.filter_preview_line.remove()
                    self.filter_preview_line = None
                except:
                    pass
            
            # Update plot
            self.update_plot()
            
            filter_type = self.filter_type_combo.currentText()
            QMessageBox.information(self, "Filter Applied", f"{filter_type} filter applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Filter application failed: {str(e)}")
    
    def _apply_fourier_filter_internal(self, preview_only=False):
        """Internal method to apply Fourier filtering."""
        # Calculate FFT
        fft_data = np.fft.fft(self.processed_intensities)
        n_points = len(fft_data)
        frequencies = np.fft.fftfreq(n_points)
        
        # Get filter parameters
        filter_type = self.filter_type_combo.currentText()
        low_cutoff = self.low_cutoff_slider.value() / 100.0  # Convert to fraction
        high_cutoff = self.high_cutoff_slider.value() / 100.0
        
        # Check if it's a Butterworth filter
        is_butterworth = "Butterworth" in filter_type
        
        if is_butterworth:
            # Get Butterworth filter order
            order = self.butterworth_order_slider.value()
            
            # Create Butterworth filter response
            filter_response = self._create_butterworth_response(frequencies, filter_type, low_cutoff, high_cutoff, order)
            
            # Apply filter to FFT data (multiply by response, not mask)
            filtered_fft = fft_data * filter_response
        else:
            # Original binary mask filters
            filter_mask = np.ones_like(frequencies, dtype=bool)
            
            # Apply filter based on type
            freq_magnitude = np.abs(frequencies)
            
            if filter_type == "Low-pass":
                filter_mask = freq_magnitude <= high_cutoff
            elif filter_type == "High-pass":
                filter_mask = freq_magnitude >= low_cutoff
            elif filter_type == "Band-pass":
                filter_mask = (freq_magnitude >= low_cutoff) & (freq_magnitude <= high_cutoff)
            elif filter_type == "Band-stop":
                filter_mask = (freq_magnitude < low_cutoff) | (freq_magnitude > high_cutoff)
            
            # Apply filter to FFT data
            filtered_fft = fft_data.copy()
            filtered_fft[~filter_mask] = 0
        
        # Convert back to time domain
        filtered_spectrum = np.real(np.fft.ifft(filtered_fft))
        
        return filtered_spectrum
    
    def _create_butterworth_response(self, frequencies, filter_type, low_cutoff, high_cutoff, order):
        """Create Butterworth filter frequency response."""
        freq_magnitude = np.abs(frequencies)
        
        # Initialize response array
        response = np.ones_like(frequencies, dtype=float)
        
        # Avoid division by zero
        epsilon = 1e-10
        freq_magnitude = np.maximum(freq_magnitude, epsilon)
        
        if filter_type == "Butterworth Low-pass":
            # |H(ω)|² = 1 / (1 + (ω/ωc)^(2n))
            response = 1.0 / (1.0 + (freq_magnitude / max(high_cutoff, epsilon)) ** (2 * order))
            
        elif filter_type == "Butterworth High-pass":
            # |H(ω)|² = (ω/ωc)^(2n) / (1 + (ω/ωc)^(2n))
            ratio = freq_magnitude / max(low_cutoff, epsilon)
            response = (ratio ** (2 * order)) / (1.0 + ratio ** (2 * order))
            
        elif filter_type == "Butterworth Band-pass":
            # Combination of high-pass and low-pass
            # High-pass component
            ratio_low = freq_magnitude / max(low_cutoff, epsilon)
            high_pass_response = (ratio_low ** (2 * order)) / (1.0 + ratio_low ** (2 * order))
            
            # Low-pass component
            ratio_high = freq_magnitude / max(high_cutoff, epsilon)
            low_pass_response = 1.0 / (1.0 + ratio_high ** (2 * order))
            
            # Combine
            response = high_pass_response * low_pass_response
            
        elif filter_type == "Butterworth Band-stop":
            # Inverse of band-pass: 1 - band_pass_response
            # High-pass component
            ratio_low = freq_magnitude / max(low_cutoff, epsilon)
            high_pass_response = (ratio_low ** (2 * order)) / (1.0 + ratio_low ** (2 * order))
            
            # Low-pass component  
            ratio_high = freq_magnitude / max(high_cutoff, epsilon)
            low_pass_response = 1.0 / (1.0 + ratio_high ** (2 * order))
            
            # Band-pass response
            band_pass_response = high_pass_response * low_pass_response
            
            # Band-stop is inverse
            response = 1.0 - band_pass_response
        
        # Take square root to get magnitude response (since we calculated |H(ω)|²)
        response = np.sqrt(np.maximum(response, 0))
        
        return response
    
    def apply_fourier_smoothing(self):
        """Apply Fourier-based smoothing to reduce noise."""
        try:
            # Calculate FFT
            fft_data = np.fft.fft(self.processed_intensities)
            n_points = len(fft_data)
            frequencies = np.fft.fftfreq(n_points)
            
            # Create Gaussian smoothing filter
            sigma = 0.1  # Smoothing parameter
            smoothing_filter = np.exp(-0.5 * (frequencies / sigma) ** 2)
            
            # Apply smoothing in frequency domain
            smoothed_fft = fft_data * smoothing_filter
            
            # Convert back to time domain
            smoothed_spectrum = np.real(np.fft.ifft(smoothed_fft))
            
            # Apply to processed intensities
            self.processed_intensities = smoothed_spectrum.copy()
            
            # Update plot
            self.update_plot()
            
            QMessageBox.information(self, "Smoothing Applied", "Fourier smoothing applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"Fourier smoothing failed: {str(e)}")
    
    def apply_richardson_lucy(self):
        """Apply Richardson-Lucy deconvolution for resolution enhancement."""
        try:
            # Enhanced Richardson-Lucy implementation
            iterations = 20
            deconvolved = self.richardson_lucy_deconvolution(self.processed_intensities, iterations)
            
            # Apply to processed intensities
            self.processed_intensities = deconvolved.copy()
            
            # Update plot
            self.update_plot()
            
            QMessageBox.information(self, "Deconvolution Applied", 
                                  f"Richardson-Lucy deconvolution applied with {iterations} iterations!")
            
        except Exception as e:
            QMessageBox.critical(self, "Deconvolution Error", f"Richardson-Lucy deconvolution failed: {str(e)}")
    
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
    
    def apply_apodization(self):
        """Apply apodization (windowing) to the spectrum."""
        try:
            # Create apodization options dialog
            from PySide6.QtWidgets import QInputDialog
            
            window_types = ["Hann", "Hamming", "Blackman", "Gaussian", "Tukey"]
            window_type, ok = QInputDialog.getItem(
                self, "Select Window Type", "Choose apodization window:", 
                window_types, 0, False
            )
            
            if not ok:
                return
            
            n_points = len(self.processed_intensities)
            
            # Create window function
            if window_type == "Hann":
                window = np.hanning(n_points)
            elif window_type == "Hamming":
                window = np.hamming(n_points)
            elif window_type == "Blackman":
                window = np.blackman(n_points)
            elif window_type == "Gaussian":
                sigma = n_points / 8
                x = np.arange(n_points) - n_points // 2
                window = np.exp(-0.5 * (x / sigma) ** 2)
            elif window_type == "Tukey":
                # Simple Tukey window implementation
                alpha = 0.5
                window = np.ones(n_points)
                n_taper = int(alpha * n_points / 2)
                
                # Left taper
                for i in range(n_taper):
                    window[i] = 0.5 * (1 + np.cos(np.pi * (2 * i / (alpha * n_points) - 1)))
                
                # Right taper
                for i in range(n_points - n_taper, n_points):
                    window[i] = 0.5 * (1 + np.cos(np.pi * (2 * (i - n_points + n_taper) / (alpha * n_points) - 1)))
            
            # Apply window to spectrum
            windowed_spectrum = self.processed_intensities * window
            
            # Normalize to preserve total intensity
            windowed_spectrum *= np.sum(self.processed_intensities) / np.sum(windowed_spectrum)
            
            # Apply to processed intensities
            self.processed_intensities = windowed_spectrum.copy()
            
            # Update plot
            self.update_plot()
            
            QMessageBox.information(self, "Apodization Applied", f"{window_type} window applied successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Apodization Error", f"Apodization failed: {str(e)}")
    
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
                
                # Count additional files created
                base_name = file_path.replace('.csv', '').replace('.txt', '')
                additional_files = []
                
                if (self.fit_result is not None and self.fit_params is not None and 
                    hasattr(self, 'peaks') and len(self.get_all_peaks_for_fitting()) > 0):
                    
                    n_peaks = len(self.get_all_peaks_for_fitting())
                    additional_files.append(f"• Peak parameters: {base_name}_peak_parameters.csv")
                    additional_files.append(f"• {n_peaks} individual peak region files")
                
                message = f"Export completed successfully!\n\nMain file: {file_path}\n"
                if additional_files:
                    message += "\nAdditional files created:\n" + "\n".join(additional_files)
                    message += f"\n\nTotal files exported: {1 + len(additional_files)}"
                
                QMessageBox.information(self, "Export Complete", message)
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")
    
    def export_to_file(self, file_path):
        """Export results to specified file with enhanced peak data."""
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
        
        # Add total fitted curve if available
        if self.fit_result is not None and self.fit_params is not None:
            fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
            data['Total_Fitted_Curve'] = fitted_curve
        
        # Add individual peak curves (full spectrum)
        if (self.fit_result is not None and self.fit_params is not None and 
            hasattr(self, 'peaks') and len(self.get_all_peaks_for_fitting()) > 0):
            
            individual_r2_values = self.calculate_individual_r2_values()
            all_fitted_peaks = self.get_all_peaks_for_fitting()
            validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
            n_peaks = len(validated_peaks)
            
            for i in range(n_peaks):
                start_idx = i * 3
                if start_idx + 2 < len(self.fit_params):
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                    
                    # Generate individual peak curve
                    if self.current_model == "Gaussian":
                        peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                    elif self.current_model == "Lorentzian":
                        peak_curve = self.lorentzian(self.wavenumbers, amp, cen, wid)
                    elif self.current_model == "Pseudo-Voigt":
                        peak_curve = self.pseudo_voigt(self.wavenumbers, amp, cen, wid)
                    else:
                        peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                    
                    data[f'Peak_{i+1}_Full_Curve'] = peak_curve
        
        # Add components if available
        for i, component in enumerate(self.components):
            data[f'Component_{i+1}'] = component
        
        # Create main DataFrame
        df = pd.DataFrame(data)
        
        # Save main spectral data
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
            base_path = file_path.replace('.csv', '')
        else:
            df.to_csv(file_path, sep='\t', index=False)
            base_path = file_path.replace('.txt', '') if file_path.endswith('.txt') else file_path
        
        # Export peak parameters and regional curves
        self._export_peak_details(base_path)
    
    def _export_peak_details(self, base_path):
        """Export detailed peak parameters and regional curves."""
        if (self.fit_result is None or self.fit_params is None or 
            not hasattr(self, 'peaks') or len(self.get_all_peaks_for_fitting()) == 0):
            return
        
        # Get peak data
        individual_r2_values = self.calculate_individual_r2_values()
        total_r2 = self.calculate_total_r2()
        all_fitted_peaks = self.get_all_peaks_for_fitting()
        validated_peaks = self.validate_peak_indices(np.array(all_fitted_peaks))
        n_peaks = len(validated_peaks)
        
        # Export 1: Peak Parameters Summary
        self._export_peak_parameters(base_path, n_peaks, individual_r2_values, total_r2)
        
        # Export 2: Regional Peak Curves (+/- 75 cm^-1 around each peak)
        self._export_regional_peak_curves(base_path, n_peaks, individual_r2_values)
    
    def _export_peak_parameters(self, base_path, n_peaks, individual_r2_values, total_r2):
        """Export peak parameters to a separate file."""
        peak_params = []
        
        # Add header information
        peak_params.append({
            'Parameter': 'Analysis Summary',
            'Value': '',
            'Units': '',
            'Notes': f'Model: {self.current_model}, Total R²: {total_r2:.4f}'
        })
        peak_params.append({
            'Parameter': 'Number of Peaks',
            'Value': n_peaks,
            'Units': '',
            'Notes': f'Average Individual R²: {np.mean(individual_r2_values):.3f}' if individual_r2_values else ''
        })
        peak_params.append({
            'Parameter': '',
            'Value': '',
            'Units': '',
            'Notes': ''
        })  # Empty row
        
        # Add individual peak parameters
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
                # Determine peak type
                peak_type = "Auto"
                if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                    validated_auto_peaks = self.validate_peak_indices(self.peaks)
                    all_fitted_peaks = self.get_all_peaks_for_fitting()
                    if (len(validated_auto_peaks) > 0 and i < len(all_fitted_peaks) and 
                        all_fitted_peaks[i] not in validated_auto_peaks.tolist()):
                        peak_type = "Manual"
                
                # Calculate additional peak properties
                fwhm = self._calculate_fwhm(wid)
                area = self._calculate_peak_area(amp, wid)
                
                peak_params.extend([
                    {
                        'Parameter': f'Peak {i+1} Type',
                        'Value': peak_type,
                        'Units': '',
                        'Notes': f'R² = {r2_value:.3f}'
                    },
                    {
                        'Parameter': f'Peak {i+1} Center',
                        'Value': f'{cen:.2f}',
                        'Units': 'cm⁻¹',
                        'Notes': 'Peak centroid position'
                    },
                    {
                        'Parameter': f'Peak {i+1} Amplitude',
                        'Value': f'{amp:.2f}',
                        'Units': 'intensity',
                        'Notes': 'Peak height'
                    },
                    {
                        'Parameter': f'Peak {i+1} Width',
                        'Value': f'{wid:.2f}',
                        'Units': 'cm⁻¹',
                        'Notes': f'Model parameter (FWHM ≈ {fwhm:.2f})'
                    },
                    {
                        'Parameter': f'Peak {i+1} Area',
                        'Value': f'{area:.2f}',
                        'Units': 'intensity·cm⁻¹',
                        'Notes': 'Integrated peak area'
                    },
                    {
                        'Parameter': '',
                        'Value': '',
                        'Units': '',
                        'Notes': ''
                    }  # Empty row
                ])
        
        # Save peak parameters
        peak_df = pd.DataFrame(peak_params)
        param_file = f"{base_path}_peak_parameters.csv"
        peak_df.to_csv(param_file, index=False)
    
    def _export_regional_peak_curves(self, base_path, n_peaks, individual_r2_values):
        """Export individual peak curves in +/- 75 cm^-1 regions around centroids."""
        regional_data = {}
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                r2_value = individual_r2_values[i] if i < len(individual_r2_values) else 0.0
                
                # Define region: +/- 75 cm^-1 around centroid
                region_start = cen - 75
                region_end = cen + 75
                
                # Find indices within this region
                region_mask = (self.wavenumbers >= region_start) & (self.wavenumbers <= region_end)
                
                if np.any(region_mask):
                    # Extract regional data
                    region_wavenumbers = self.wavenumbers[region_mask]
                    region_original = self.original_intensities[region_mask]
                    region_processed = self.processed_intensities[region_mask]
                    
                    # Generate individual peak curve for this region
                    if self.current_model == "Gaussian":
                        region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
                    elif self.current_model == "Lorentzian":
                        region_peak = self.lorentzian(region_wavenumbers, amp, cen, wid)
                    elif self.current_model == "Pseudo-Voigt":
                        region_peak = self.pseudo_voigt(region_wavenumbers, amp, cen, wid)
                    else:
                        region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
                    
                    # Generate total fit for this region
                    region_total_fit = self.multi_peak_model(region_wavenumbers, *self.fit_params)
                    
                    # Add background if available
                    region_background = None
                    if self.background is not None:
                        region_background = self.background[region_mask]
                    
                    # Store data with consistent length
                    max_length = max(len(regional_data.get('Wavenumber', [])), len(region_wavenumbers))
                    
                    # Extend existing columns if needed
                    for key in regional_data:
                        while len(regional_data[key]) < max_length:
                            regional_data[key].append(np.nan)
                    
                    # Add new data (pad if shorter than existing data)
                    if 'Wavenumber' not in regional_data:
                        regional_data['Wavenumber'] = []
                    
                    regional_data['Wavenumber'].extend(region_wavenumbers.tolist())
                    
                    # Pad wavenumber if needed
                    while len(regional_data['Wavenumber']) < max_length:
                        regional_data['Wavenumber'].append(np.nan)
                    
                    # Add peak-specific columns
                    col_prefix = f'Peak_{i+1}_{cen:.1f}cm'
                    
                    regional_data[f'{col_prefix}_Original'] = [np.nan] * max_length
                    regional_data[f'{col_prefix}_Processed'] = [np.nan] * max_length
                    regional_data[f'{col_prefix}_Individual'] = [np.nan] * max_length
                    regional_data[f'{col_prefix}_Total_Fit'] = [np.nan] * max_length
                    
                    if region_background is not None:
                        regional_data[f'{col_prefix}_Background'] = [np.nan] * max_length
                    
                    # Fill in the actual data for this peak's region
                    start_fill = max_length - len(region_wavenumbers)
                    for j, val in enumerate(region_original):
                        regional_data[f'{col_prefix}_Original'][start_fill + j] = val
                    for j, val in enumerate(region_processed):
                        regional_data[f'{col_prefix}_Processed'][start_fill + j] = val
                    for j, val in enumerate(region_peak):
                        regional_data[f'{col_prefix}_Individual'][start_fill + j] = val
                    for j, val in enumerate(region_total_fit):
                        regional_data[f'{col_prefix}_Total_Fit'][start_fill + j] = val
                    
                    if region_background is not None:
                        for j, val in enumerate(region_background):
                            regional_data[f'{col_prefix}_Background'][start_fill + j] = val
        
        # Create a simpler approach: separate file for each peak region
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                self._export_single_peak_region(base_path, i+1, amp, cen, wid, individual_r2_values)
    
    def _export_single_peak_region(self, base_path, peak_num, amp, cen, wid, individual_r2_values):
        """Export a single peak's regional data to its own file."""
        # Define region: +/- 75 cm^-1 around centroid
        region_start = cen - 75
        region_end = cen + 75
        
        # Find indices within this region
        region_mask = (self.wavenumbers >= region_start) & (self.wavenumbers <= region_end)
        
        if not np.any(region_mask):
            return
        
        # Extract regional data
        region_wavenumbers = self.wavenumbers[region_mask]
        region_original = self.original_intensities[region_mask]
        region_processed = self.processed_intensities[region_mask]
        
        # Generate curves for this region
        if self.current_model == "Gaussian":
            region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
        elif self.current_model == "Lorentzian":
            region_peak = self.lorentzian(region_wavenumbers, amp, cen, wid)
        elif self.current_model == "Pseudo-Voigt":
            region_peak = self.pseudo_voigt(region_wavenumbers, amp, cen, wid)
        else:
            region_peak = self.gaussian(region_wavenumbers, amp, cen, wid)
        
        # Generate total fit for this region
        region_total_fit = self.multi_peak_model(region_wavenumbers, *self.fit_params)
        
        # Create data dictionary
        peak_data = {
            'Wavenumber': region_wavenumbers,
            'Original_Intensity': region_original,
            'Processed_Intensity': region_processed,
            'Individual_Peak_Fit': region_peak,
            'Total_Fit': region_total_fit,
            'Residual': region_processed - region_total_fit
        }
        
        # Add background if available
        if self.background is not None:
            region_background = self.background[region_mask]
            peak_data['Background'] = region_background
        
        # Create DataFrame and save
        peak_df = pd.DataFrame(peak_data)
        r2_value = individual_r2_values[peak_num-1] if (peak_num-1) < len(individual_r2_values) else 0.0
        region_file = f"{base_path}_peak_{peak_num:02d}_{cen:.1f}cm_R2_{r2_value:.3f}.csv"
        peak_df.to_csv(region_file, index=False)
    
    def _calculate_fwhm(self, width_param):
        """Calculate FWHM from model width parameter."""
        if self.current_model == "Gaussian":
            # For Gaussian: FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
            return 2.355 * abs(width_param)
        elif self.current_model == "Lorentzian":
            # For Lorentzian: FWHM = 2 * gamma
            return 2 * abs(width_param)
        else:
            # Default to Gaussian approximation
            return 2.355 * abs(width_param)
    
    def _calculate_peak_area(self, amplitude, width_param):
        """Calculate integrated peak area."""
        if self.current_model == "Gaussian":
            # For Gaussian: Area = amplitude * width * sqrt(2*pi)
            return abs(amplitude * width_param * np.sqrt(2 * np.pi))
        elif self.current_model == "Lorentzian":
            # For Lorentzian: Area = amplitude * width * pi
            return abs(amplitude * width_param * np.pi)
        else:
            # Default to Gaussian approximation
            return abs(amplitude * width_param * np.sqrt(2 * np.pi))

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