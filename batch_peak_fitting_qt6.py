#!/usr/bin/env python3
"""
Enhanced Batch Peak Fitting Module for RamanLab Qt6
Allows sequential refinement of peak positions, shapes, and backgrounds
across multiple spectra, with visualization of trends.
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
    QHeaderView, QScrollArea, QFrame, QGridLayout, QListWidget, QListWidgetItem,
    QFileDialog, QApplication, QMenuBar, QMenu, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QAction, QKeySequence

# Import matplotlib for Qt6
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme

# Scientific computing
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import os
import chardet

class BatchPeakFittingQt6(QDialog):
    """Enhanced batch peak fitting dialog for processing multiple Raman spectra."""
    
    def __init__(self, parent, wavenumbers=None, intensities=None):
        super().__init__(parent)
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.parent = parent
        
        # Initialize data storage
        self.spectra_files = []  # List of file paths
        self.current_spectrum_index = 0
        self.batch_results = []
        self.reference_peaks = None
        self.reference_background = None
        
        # Current spectrum data
        self.wavenumbers = np.array(wavenumbers) if wavenumbers is not None else np.array([])
        self.intensities = np.array(intensities) if intensities is not None else np.array([])
        self.original_intensities = self.intensities.copy() if len(self.intensities) > 0 else np.array([])
        
        # Peak fitting data
        self.peaks = np.array([], dtype=int)
        self.manual_peaks = np.array([], dtype=int)  # Manually selected peaks
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.residuals = None
        self.current_model = "Gaussian"
        
        # UI state
        self.show_individual_peaks = True
        self.show_fitted_peaks = True
        self._stop_batch = False
        self.interactive_mode = False  # Interactive peak selection mode
        self.click_connection = None  # Mouse click connection
        
        # Preview states for UI consistency
        self.background_preview_active = False
        self.smoothing_preview_active = False
        
        self.setup_ui()
        self.initial_plot()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Batch Peak Fitting")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create and add menu bar
        menu_bar = self.create_menu_bar()
        main_layout.addWidget(menu_bar)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - visualization
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([450, 1150])
        
        # Setup click handlers for plot interaction after UI is created
        QTimer.singleShot(100, self.setup_plot_click_handlers)  # Delay to ensure all widgets are created
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menu_bar = QMenuBar(self)
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        add_files_action = QAction("Add Files", self)
        add_files_action.triggered.connect(self.add_files)
        file_menu.addAction(add_files_action)
        
        export_action = QAction("Export Results", self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        help_action = QAction("Help", self)
        help_action.setShortcut(QKeySequence.HelpContents)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        return menu_bar
        
    def create_control_panel(self):
        """Create the control panel with tabs."""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.tab_widget.addTab(self.create_file_selection_tab(), "File Selection")
        self.tab_widget.addTab(self.create_peak_controls_tab(), "Peak Controls")
        self.tab_widget.addTab(self.create_batch_tab(), "Batch Processing")
        self.tab_widget.addTab(self.create_results_tab(), "Results")
        
        return panel
        
    def create_file_selection_tab(self):
        """Create file selection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # File list
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemDoubleClicked.connect(self.on_file_double_click)
        file_layout.addWidget(self.file_list_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Files")
        add_btn.clicked.connect(self.add_files)
        button_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_files)
        button_layout.addWidget(remove_btn)
        
        file_layout.addLayout(button_layout)
        
        # Navigation controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        first_btn = QPushButton("First")
        first_btn.clicked.connect(lambda: self.navigate_spectrum(0))
        nav_layout.addWidget(first_btn)
        
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(lambda: self.navigate_spectrum(-1))
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(lambda: self.navigate_spectrum(1))
        nav_layout.addWidget(next_btn)
        
        last_btn = QPushButton("Last")
        last_btn.clicked.connect(lambda: self.navigate_spectrum(-2))
        nav_layout.addWidget(last_btn)
        
        layout.addWidget(file_group)
        layout.addWidget(nav_group)
        
        # Status label
        self.current_file_label = QLabel("No files loaded")
        layout.addWidget(self.current_file_label)
        
        layout.addStretch()
        return tab
        
    def create_peak_controls_tab(self):
        """Create peak controls tab with sub-tabs for better organization."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create sub-tab widget
        sub_tab_widget = QTabWidget()
        layout.addWidget(sub_tab_widget)
        
        # Create sub-tabs
        bg_smooth_tab = self.create_background_smoothing_tab()
        sub_tab_widget.addTab(bg_smooth_tab, "Background & Smoothing")
        
        peak_detection_tab = self.create_peak_detection_tab()
        sub_tab_widget.addTab(peak_detection_tab, "Peak Detection & Management")
        
        layout.addStretch()
        return tab
        
    def create_background_smoothing_tab(self):
        """Create background subtraction and smoothing sub-tab."""
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
        
        # Connect sliders to update labels
        self.lambda_slider.valueChanged.connect(self.update_lambda_label)
        self.p_slider.valueChanged.connect(self.update_p_label)
        
        bg_layout.addWidget(self.als_params_widget)
        
        # Background subtraction buttons
        button_layout = QVBoxLayout()  # Changed from QHBoxLayout to QVBoxLayout
        
        subtract_btn = QPushButton("Apply Background Subtraction")
        subtract_btn.clicked.connect(self.apply_background_subtraction)
        button_layout.addWidget(subtract_btn)
        
        preview_btn = QPushButton("Clear Preview")
        preview_btn.clicked.connect(self.clear_background_preview)
        button_layout.addWidget(preview_btn)
        
        bg_layout.addLayout(button_layout)
        
        reset_btn = QPushButton("Reset Spectrum")
        reset_btn.clicked.connect(self.reset_spectrum)
        bg_layout.addWidget(reset_btn)
        
        layout.addWidget(bg_group)
        
        # Smoothing group
        smooth_group = QGroupBox("Spectral Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)
        
        # Savitzky-Golay parameters
        sg_layout = QHBoxLayout()
        sg_layout.addWidget(QLabel("Window Length:"))
        self.sg_window_spin = QSpinBox()
        self.sg_window_spin.setRange(3, 51)
        self.sg_window_spin.setValue(5)
        self.sg_window_spin.setSingleStep(2)  # Only odd numbers
        sg_layout.addWidget(self.sg_window_spin)
        
        sg_layout.addWidget(QLabel("Poly Order:"))
        self.sg_order_spin = QSpinBox()
        self.sg_order_spin.setRange(1, 5)
        self.sg_order_spin.setValue(2)
        sg_layout.addWidget(self.sg_order_spin)
        smooth_layout.addLayout(sg_layout)
        
        # Smoothing buttons
        smooth_button_layout = QVBoxLayout()  # Changed from QHBoxLayout to QVBoxLayout
        
        smooth_btn = QPushButton("Apply Savitzky-Golay Smoothing")
        smooth_btn.clicked.connect(self.apply_smoothing)
        smooth_button_layout.addWidget(smooth_btn)
        
        clear_smooth_btn = QPushButton("Clear Preview")
        clear_smooth_btn.clicked.connect(self.clear_smoothing_preview)
        smooth_button_layout.addWidget(clear_smooth_btn)
        
        smooth_layout.addLayout(smooth_button_layout)
        
        layout.addWidget(smooth_group)
        layout.addStretch()
        return tab
        
    def create_peak_detection_tab(self):
        """Create peak detection and management sub-tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Peak detection group with real-time sliders
        peak_group = QGroupBox("Peak Detection")
        peak_layout = QVBoxLayout(peak_group)
        
        # Height parameter
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Min Height:"))
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(0, 100)
        self.height_slider.setValue(10)
        self.height_slider.valueChanged.connect(self.update_peak_detection)
        height_layout.addWidget(self.height_slider)
        self.height_label = QLabel("10%")
        height_layout.addWidget(self.height_label)
        peak_layout.addLayout(height_layout)
        
        # Distance parameter
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Min Distance:"))
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(10)
        self.distance_slider.valueChanged.connect(self.update_peak_detection)
        distance_layout.addWidget(self.distance_slider)
        self.distance_label = QLabel("10")
        distance_layout.addWidget(self.distance_label)
        peak_layout.addLayout(distance_layout)
        
        # Prominence parameter
        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(QLabel("Prominence:"))
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(0, 50)
        self.prominence_slider.setValue(5)
        self.prominence_slider.valueChanged.connect(self.update_peak_detection)
        prominence_layout.addWidget(self.prominence_slider)
        self.prominence_label = QLabel("5%")
        prominence_layout.addWidget(self.prominence_label)
        peak_layout.addLayout(prominence_layout)
        
        # Manual peak detection button
        detect_btn = QPushButton("Detect Peaks")
        detect_btn.clicked.connect(self.find_peaks)
        peak_layout.addWidget(detect_btn)
        
        # Peak count display
        self.peak_count_label = QLabel("Peaks found: 0")
        peak_layout.addWidget(self.peak_count_label)
        
        layout.addWidget(peak_group)
        
        # Peak management group
        peak_mgmt_group = QGroupBox("Peak Management")
        peak_mgmt_layout = QVBoxLayout(peak_mgmt_group)
        
        # Peak list display and deletion
        peak_list_layout = QHBoxLayout()
        
        # Peak list
        peak_list_left = QVBoxLayout()
        peak_list_left.addWidget(QLabel("Detected Peaks:"))
        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setMaximumHeight(120)
        peak_list_left.addWidget(self.peak_list_widget)
        peak_list_layout.addLayout(peak_list_left)
        
        # Delete buttons
        delete_buttons_layout = QVBoxLayout()
        
        delete_selected_btn = QPushButton("Delete Selected")
        delete_selected_btn.clicked.connect(self.delete_selected_peaks)
        delete_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        delete_buttons_layout.addWidget(delete_selected_btn)
        
        delete_all_btn = QPushButton("Delete All Peaks")
        delete_all_btn.clicked.connect(self.clear_peaks)
        delete_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        delete_buttons_layout.addWidget(delete_all_btn)
        
        delete_buttons_layout.addStretch()
        peak_list_layout.addLayout(delete_buttons_layout)
        
        peak_mgmt_layout.addLayout(peak_list_layout)
        
        layout.addWidget(peak_mgmt_group)
        
        # Manual peak controls
        manual_group = QGroupBox("Manual Peak Control")
        manual_layout = QVBoxLayout(manual_group)
        
        # Interactive mode toggle
        self.interactive_btn = QPushButton("🖱️ Enable Interactive Selection")
        self.interactive_btn.setCheckable(True)
        self.interactive_btn.clicked.connect(self.toggle_interactive_mode)
        self.interactive_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:checked {
                background-color: #E64A19;
            }
            QPushButton:checked:hover {
                background-color: #D84315;
            }
        """)
        manual_layout.addWidget(self.interactive_btn)
        
        # Manual peak controls
        manual_controls_layout = QHBoxLayout()
        
        clear_manual_btn = QPushButton("Clear Manual")
        clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        clear_manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        
        combine_btn = QPushButton("Combine Auto + Manual")
        combine_btn.clicked.connect(self.combine_peaks)
        combine_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        manual_controls_layout.addWidget(clear_manual_btn)
        manual_controls_layout.addWidget(combine_btn)
        manual_layout.addLayout(manual_controls_layout)
        
        # Peak status display
        self.peak_status_label = QLabel("Auto: 0 | Manual: 0 | Total: 0")
        self.peak_status_label.setStyleSheet("font-weight: bold; color: #333;")
        manual_layout.addWidget(self.peak_status_label)
        
        self.interactive_status_label = QLabel("Interactive mode: OFF")
        self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
        manual_layout.addWidget(self.interactive_status_label)
        
        layout.addWidget(manual_group)
        
        # Model selection and fitting
        model_group = QGroupBox("Peak Model & Fitting")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        fit_btn = QPushButton("Fit Peaks")
        fit_btn.clicked.connect(self.fit_peaks)
        model_layout.addWidget(fit_btn)
        
        layout.addWidget(model_group)
        layout.addStretch()
        return tab
        
    def create_batch_tab(self):
        """Create batch processing tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Batch processing controls
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout(batch_group)
        
        set_ref_btn = QPushButton("Set as Reference")
        set_ref_btn.clicked.connect(self.set_reference)
        batch_layout.addWidget(set_ref_btn)
        
        # Processing buttons
        process_layout = QHBoxLayout()
        apply_all_btn = QPushButton("Apply to All")
        apply_all_btn.clicked.connect(self.apply_to_all)
        process_layout.addWidget(apply_all_btn)
        
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_batch)
        process_layout.addWidget(stop_btn)
        
        batch_layout.addLayout(process_layout)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        batch_layout.addWidget(export_btn)
        
        layout.addWidget(batch_group)
        
        # Progress log
        log_group = QGroupBox("Progress Log")
        log_layout = QVBoxLayout(log_group)
        
        self.batch_status_text = QTextEdit()
        self.batch_status_text.setMaximumHeight(200)
        self.batch_status_text.setReadOnly(True)
        self.batch_status_text.append("Ready. Set a reference spectrum and click 'Apply to All' to begin batch processing.")
        log_layout.addWidget(self.batch_status_text)
        
        layout.addWidget(log_group)
        layout.addStretch()
        return tab
        
    def create_live_view_tab(self):
        """Create live view tab for real-time batch processing visualization."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Current file info
        info_group = QGroupBox("Current Processing")
        info_layout = QVBoxLayout(info_group)
        
        self.current_file_info = QLabel("No file being processed")
        self.current_file_info.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        info_layout.addWidget(self.current_file_info)
        
        # Fit statistics
        stats_layout = QHBoxLayout()
        
        self.overall_r2_label = QLabel("Overall R²: --")
        self.overall_r2_label.setStyleSheet("font-weight: bold; color: #2196F3; font-size: 14px;")
        stats_layout.addWidget(self.overall_r2_label)
        
        self.peak_count_live = QLabel("Peaks: --")
        self.peak_count_live.setStyleSheet("font-weight: bold; color: #4CAF50; font-size: 14px;")
        stats_layout.addWidget(self.peak_count_live)
        
        info_layout.addLayout(stats_layout)
        layout.addWidget(info_group)
        
        # Live plot
        plot_group = QGroupBox("Live Fit Visualization")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create matplotlib figure for live view
        self.figure_live = Figure(figsize=(10, 6))
        self.canvas_live = FigureCanvas(self.figure_live)
        self.toolbar_live = NavigationToolbar(self.canvas_live, plot_group)
        
        # Create subplot with height ratios
        gs = self.figure_live.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)
        self.ax_live_main = self.figure_live.add_subplot(gs[0, 0])
        self.ax_live_residual = self.figure_live.add_subplot(gs[1, 0])
        
        self.figure_live.tight_layout(pad=3.0)
        
        plot_layout.addWidget(self.toolbar_live)
        plot_layout.addWidget(self.canvas_live)
        
        layout.addWidget(plot_group)
        
        # Smart Residual Analyzer
        analyzer_group = QGroupBox("Smart Residual Analyzer")
        analyzer_layout = QVBoxLayout(analyzer_group)
        
        # Analysis display
        self.analysis_text = QTextEdit()
        self.analysis_text.setMaximumHeight(120)
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        analyzer_layout.addWidget(self.analysis_text)
        
        # Analysis control buttons
        analysis_controls = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Analyze Current Fit")
        self.analyze_btn.clicked.connect(self.analyze_current_fit)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        analysis_controls.addWidget(self.analyze_btn)
        
        self.export_batch_csv_btn = QPushButton("Export All Batch Data (CSV)")
        self.export_batch_csv_btn.clicked.connect(self.export_comprehensive_batch_data)
        self.export_batch_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e7e34;
            }
        """)
        analysis_controls.addWidget(self.export_batch_csv_btn)
        
        analyzer_layout.addLayout(analysis_controls)
        layout.addWidget(analyzer_group)
        
        # Initialize analysis text
        self.analysis_text.append("🔍 Smart Residual Analyzer Ready")
        self.analysis_text.append("This tool will analyze your fits and suggest improvements.")
        self.analysis_text.append("Process some spectra and click 'Analyze Current Fit' to begin.")
        
        # Initialize saved fits storage
        self.saved_fits = {}  # Dictionary to store saved fits by filename
        
        return tab
        
    def create_results_tab(self):
        """Create results tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Peak visibility controls
        visibility_group = QGroupBox("Peak Visibility")
        visibility_layout = QVBoxLayout(visibility_group)
        
        # Show/hide all buttons
        button_layout = QHBoxLayout()
        show_all_btn = QPushButton("Show All")
        show_all_btn.clicked.connect(self.show_all_peaks)
        button_layout.addWidget(show_all_btn)
        
        hide_all_btn = QPushButton("Hide All")
        hide_all_btn.clicked.connect(self.hide_all_peaks)
        button_layout.addWidget(hide_all_btn)
        
        visibility_layout.addLayout(button_layout)
        
        # Peak checkboxes (will be populated dynamically)
        self.peak_visibility_scroll = QScrollArea()
        self.peak_visibility_widget = QWidget()
        self.peak_visibility_layout = QVBoxLayout(self.peak_visibility_widget)
        self.peak_visibility_scroll.setWidget(self.peak_visibility_widget)
        self.peak_visibility_scroll.setWidgetResizable(True)
        self.peak_visibility_scroll.setMaximumHeight(150)
        visibility_layout.addWidget(self.peak_visibility_scroll)
        
        self.peak_visibility_vars = []
        
        layout.addWidget(visibility_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_boundary_check = QCheckBox("Show 95% Boundary")
        self.show_boundary_check.setChecked(True)
        self.show_boundary_check.stateChanged.connect(self.update_all_plots)
        display_layout.addWidget(self.show_boundary_check)
        
        self.show_grid_check = QCheckBox("Show Grid Lines")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.stateChanged.connect(self.update_all_plots)
        display_layout.addWidget(self.show_grid_check)
        
        self.show_peak_labels_check = QCheckBox("Show Peak Labels")
        self.show_peak_labels_check.setChecked(True)
        self.show_peak_labels_check.stateChanged.connect(self.update_all_plots)
        display_layout.addWidget(self.show_peak_labels_check)
        
        layout.addWidget(display_group)
        
        # Plot interaction options
        interaction_group = QGroupBox("Plot Interaction")
        interaction_layout = QVBoxLayout(interaction_group)
        
        popup_info = QLabel("💡 Click any plot to open it in a new window")
        popup_info.setStyleSheet("color: #666; font-style: italic;")
        interaction_layout.addWidget(popup_info)
        
        layout.addWidget(interaction_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        # Export all data button
        export_all_btn = QPushButton("Export All Plot Data as CSV")
        export_all_btn.clicked.connect(self.export_all_plot_data)
        export_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        export_layout.addWidget(export_all_btn)
        
        # Export individual results button (existing functionality)
        export_results_btn = QPushButton("Export Peak Fitting Results")
        export_results_btn.clicked.connect(self.export_results)
        export_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        export_layout.addWidget(export_results_btn)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        return tab
        
    def create_visualization_panel(self):
        """Create the visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        self.viz_tab_widget = QTabWidget()
        layout.addWidget(self.viz_tab_widget)
        
        # Current spectrum tab
        current_tab = self.create_current_spectrum_tab()
        self.viz_tab_widget.addTab(current_tab, "Current Spectrum")
        
        # Live view tab
        live_view_tab = self.create_live_view_tab()
        self.viz_tab_widget.addTab(live_view_tab, "Live View")
        
        # Trends tab
        trends_tab = self.create_trends_tab()
        self.viz_tab_widget.addTab(trends_tab, "Trends")
        
        # Waterfall tab
        waterfall_tab = self.create_waterfall_tab()
        self.viz_tab_widget.addTab(waterfall_tab, "Waterfall")
        
        # Heatmap tab
        heatmap_tab = self.create_heatmap_tab()
        self.viz_tab_widget.addTab(heatmap_tab, "Heatmap")
        
        return panel
        
    def create_current_spectrum_tab(self):
        """Create current spectrum visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create matplotlib figure
        self.figure_current = Figure(figsize=(10, 8))
        self.canvas_current = FigureCanvas(self.figure_current)
        self.toolbar_current = NavigationToolbar(self.canvas_current, tab)
        
        # Create subplots with height ratios (main plot 3x larger than residuals)
        gs = self.figure_current.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        self.ax_main = self.figure_current.add_subplot(gs[0, 0])
        self.ax_residual = self.figure_current.add_subplot(gs[1, 0])
        
        self.figure_current.tight_layout(pad=3.0)
        
        layout.addWidget(self.toolbar_current)
        layout.addWidget(self.canvas_current)
        
        return tab
        
    def create_trends_tab(self):
        """Create trends visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create matplotlib figure for trends
        self.figure_trends = Figure(figsize=(12, 10))
        self.canvas_trends = FigureCanvas(self.figure_trends)
        self.toolbar_trends = NavigationToolbar(self.canvas_trends, tab)
        
        layout.addWidget(self.toolbar_trends)
        layout.addWidget(self.canvas_trends)
        
        return tab
        
    def create_waterfall_tab(self):
        """Create waterfall plot visualization tab with comprehensive controls."""
        tab = QWidget()
        layout = QHBoxLayout(tab)  # Use horizontal layout for controls and plot
        
        # Left panel - controls
        controls_panel = QWidget()
        controls_panel.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_panel)
        
        # Data selection group
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)
        
        # Data type selection
        data_type_layout = QHBoxLayout()
        data_type_layout.addWidget(QLabel("Data Type:"))
        self.waterfall_data_combo = QComboBox()
        self.waterfall_data_combo.addItems([
            "Raw Intensity", "Background Corrected", "Fitted Peaks", "Residuals"
        ])
        self.waterfall_data_combo.currentTextChanged.connect(self.update_waterfall_plot)
        data_type_layout.addWidget(self.waterfall_data_combo)
        data_layout.addLayout(data_type_layout)
        
        # X-axis range selection
        range_group = QGroupBox("X-axis Range")
        range_layout = QVBoxLayout(range_group)
        
        # Auto range checkbox
        self.waterfall_auto_range = QCheckBox("Auto Range")
        self.waterfall_auto_range.setChecked(True)
        self.waterfall_auto_range.stateChanged.connect(self.on_waterfall_auto_range_changed)
        range_layout.addWidget(self.waterfall_auto_range)
        
        # Manual range controls
        self.waterfall_range_controls_widget = QWidget()
        waterfall_range_controls_layout = QVBoxLayout(self.waterfall_range_controls_widget)
        waterfall_range_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        range_min_layout = QHBoxLayout()
        range_min_layout.addWidget(QLabel("Min:"))
        self.waterfall_range_min = QDoubleSpinBox()
        self.waterfall_range_min.setRange(0, 4000)
        self.waterfall_range_min.setValue(200)
        self.waterfall_range_min.valueChanged.connect(self.update_waterfall_plot)
        range_min_layout.addWidget(self.waterfall_range_min)
        waterfall_range_controls_layout.addLayout(range_min_layout)
        
        range_max_layout = QHBoxLayout()
        range_max_layout.addWidget(QLabel("Max:"))
        self.waterfall_range_max = QDoubleSpinBox()
        self.waterfall_range_max.setRange(0, 4000)
        self.waterfall_range_max.setValue(1800)
        self.waterfall_range_max.valueChanged.connect(self.update_waterfall_plot)
        range_max_layout.addWidget(self.waterfall_range_max)
        waterfall_range_controls_layout.addLayout(range_max_layout)
        
        range_layout.addWidget(self.waterfall_range_controls_widget)
        self.waterfall_range_controls_widget.setEnabled(False)  # Disabled by default (auto range)
        
        data_layout.addWidget(range_group)
        controls_layout.addWidget(data_group)
        
        # Waterfall-specific controls
        waterfall_group = QGroupBox("Waterfall Layout")
        waterfall_layout = QVBoxLayout(waterfall_group)
        
        # Skip spectra
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Skip Spectra:"))
        self.waterfall_skip_spin = QSpinBox()
        self.waterfall_skip_spin.setRange(1, 50)
        self.waterfall_skip_spin.setValue(1)
        self.waterfall_skip_spin.valueChanged.connect(self.update_waterfall_plot)
        skip_layout.addWidget(self.waterfall_skip_spin)
        waterfall_layout.addLayout(skip_layout)
        
        # Y offset
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Y Offset:"))
        self.waterfall_offset_spin = QDoubleSpinBox()
        self.waterfall_offset_spin.setRange(0, 10000)
        self.waterfall_offset_spin.setValue(100)
        self.waterfall_offset_spin.valueChanged.connect(self.update_waterfall_plot)
        offset_layout.addWidget(self.waterfall_offset_spin)
        waterfall_layout.addLayout(offset_layout)
        
        # Auto offset
        self.waterfall_auto_offset = QCheckBox("Auto Calculate Offset")
        self.waterfall_auto_offset.setChecked(False)
        self.waterfall_auto_offset.stateChanged.connect(self.update_waterfall_plot)
        waterfall_layout.addWidget(self.waterfall_auto_offset)
        
        controls_layout.addWidget(waterfall_group)
        
        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance_group)
        
        # Color scheme selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color Scheme:"))
        self.waterfall_color_scheme = QComboBox()
        self.waterfall_color_scheme.addItems([
            "Individual Colors", "Gradient", "Single Color", "By Intensity"
        ])
        self.waterfall_color_scheme.setCurrentText("Individual Colors")
        self.waterfall_color_scheme.currentTextChanged.connect(self.update_waterfall_plot)
        color_layout.addWidget(self.waterfall_color_scheme)
        appearance_layout.addLayout(color_layout)
        
        # Colormap for gradient scheme
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.waterfall_colormap = QComboBox()
        self.waterfall_colormap.addItems([
            "viridis", "plasma", "inferno", "magma", "coolwarm", 
            "rainbow", "jet", "hsv", "spring", "summer", "autumn", "winter"
        ])
        self.waterfall_colormap.setCurrentText("viridis")
        self.waterfall_colormap.currentTextChanged.connect(self.update_waterfall_plot)
        colormap_layout.addWidget(self.waterfall_colormap)
        appearance_layout.addLayout(colormap_layout)
        
        # Line properties
        line_group = QGroupBox("Line Properties")
        line_layout = QVBoxLayout(line_group)
        
        # Line width
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Line Width:"))
        self.waterfall_line_width = QSlider(Qt.Horizontal)
        self.waterfall_line_width.setRange(1, 50)  # 0.1 to 5.0
        self.waterfall_line_width.setValue(10)  # 1.0
        self.waterfall_line_width.valueChanged.connect(self.update_waterfall_line_width_label)
        self.waterfall_line_width.valueChanged.connect(self.update_waterfall_plot)
        width_layout.addWidget(self.waterfall_line_width)
        self.waterfall_line_width_label = QLabel("1.0")
        width_layout.addWidget(self.waterfall_line_width_label)
        line_layout.addLayout(width_layout)
        
        # Transparency
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Transparency:"))
        self.waterfall_alpha = QSlider(Qt.Horizontal)
        self.waterfall_alpha.setRange(10, 100)  # 0.1 to 1.0
        self.waterfall_alpha.setValue(80)  # 0.8
        self.waterfall_alpha.valueChanged.connect(self.update_waterfall_alpha_label)
        self.waterfall_alpha.valueChanged.connect(self.update_waterfall_plot)
        alpha_layout.addWidget(self.waterfall_alpha)
        self.waterfall_alpha_label = QLabel("0.8")
        alpha_layout.addWidget(self.waterfall_alpha_label)
        line_layout.addLayout(alpha_layout)
        
        appearance_layout.addWidget(line_group)
        
        # Enhancement controls
        enhancement_group = QGroupBox("Enhancement")
        enhancement_layout = QVBoxLayout(enhancement_group)
        
        # Contrast control
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.waterfall_contrast = QSlider(Qt.Horizontal)
        self.waterfall_contrast.setRange(-100, 100)
        self.waterfall_contrast.setValue(0)
        self.waterfall_contrast.valueChanged.connect(self.update_waterfall_contrast_label)
        self.waterfall_contrast.valueChanged.connect(self.update_waterfall_plot)
        contrast_layout.addWidget(self.waterfall_contrast)
        self.waterfall_contrast_label = QLabel("0")
        contrast_layout.addWidget(self.waterfall_contrast_label)
        enhancement_layout.addLayout(contrast_layout)
        
        # Brightness control
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.waterfall_brightness = QSlider(Qt.Horizontal)
        self.waterfall_brightness.setRange(-100, 100)
        self.waterfall_brightness.setValue(0)
        self.waterfall_brightness.valueChanged.connect(self.update_waterfall_brightness_label)
        self.waterfall_brightness.valueChanged.connect(self.update_waterfall_plot)
        brightness_layout.addWidget(self.waterfall_brightness)
        self.waterfall_brightness_label = QLabel("0")
        brightness_layout.addWidget(self.waterfall_brightness_label)
        enhancement_layout.addLayout(brightness_layout)
        
        # Gamma control
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.waterfall_gamma = QSlider(Qt.Horizontal)
        self.waterfall_gamma.setRange(10, 300)  # 0.1 to 3.0
        self.waterfall_gamma.setValue(100)  # 1.0
        self.waterfall_gamma.valueChanged.connect(self.update_waterfall_gamma_label)
        self.waterfall_gamma.valueChanged.connect(self.update_waterfall_plot)
        gamma_layout.addWidget(self.waterfall_gamma)
        self.waterfall_gamma_label = QLabel("1.0")
        gamma_layout.addWidget(self.waterfall_gamma_label)
        enhancement_layout.addLayout(gamma_layout)
        
        appearance_layout.addWidget(enhancement_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        # Anti-aliasing
        self.waterfall_antialias = QCheckBox("Anti-aliasing")
        self.waterfall_antialias.setChecked(True)
        self.waterfall_antialias.stateChanged.connect(self.update_waterfall_plot)
        display_layout.addWidget(self.waterfall_antialias)
        
        # Grid
        self.waterfall_grid = QCheckBox("Show Grid")
        self.waterfall_grid.setChecked(True)
        self.waterfall_grid.stateChanged.connect(self.update_waterfall_plot)
        display_layout.addWidget(self.waterfall_grid)
        
        # Auto normalize
        self.waterfall_auto_normalize = QCheckBox("Auto Normalize")
        self.waterfall_auto_normalize.setChecked(False)
        self.waterfall_auto_normalize.stateChanged.connect(self.update_waterfall_plot)
        display_layout.addWidget(self.waterfall_auto_normalize)
        
        # Show labels
        self.waterfall_show_labels = QCheckBox("Show File Labels")
        self.waterfall_show_labels.setChecked(True)
        self.waterfall_show_labels.stateChanged.connect(self.update_waterfall_plot)
        display_layout.addWidget(self.waterfall_show_labels)
        
        appearance_layout.addWidget(display_group)
        controls_layout.addWidget(appearance_group)
        
        # Update button
        update_btn = QPushButton("Update Waterfall")
        update_btn.clicked.connect(self.update_waterfall_plot)
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        controls_layout.addWidget(update_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_panel)
        
        # Right panel - visualization
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        
        # Create matplotlib figure for waterfall
        self.figure_waterfall = Figure(figsize=(12, 10))
        self.canvas_waterfall = FigureCanvas(self.figure_waterfall)
        self.toolbar_waterfall = NavigationToolbar(self.canvas_waterfall, viz_panel)
        
        viz_layout.addWidget(self.toolbar_waterfall)
        viz_layout.addWidget(self.canvas_waterfall)
        
        layout.addWidget(viz_panel)
        
        return tab
        
    def create_heatmap_tab(self):
        """Create heatmap visualization tab with comprehensive controls."""
        tab = QWidget()
        layout = QHBoxLayout(tab)  # Use horizontal layout for controls and plot
        
        # Left panel - controls
        controls_panel = QWidget()
        controls_panel.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_panel)
        
        # Data selection group
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)
        
        # Data type selection
        data_type_layout = QHBoxLayout()
        data_type_layout.addWidget(QLabel("Data Type:"))
        self.heatmap_data_combo = QComboBox()
        self.heatmap_data_combo.addItems([
            "Raw Intensity", "Background Corrected", "Fitted Peaks", "Residuals"
        ])
        self.heatmap_data_combo.currentTextChanged.connect(self.update_heatmap_plot)
        data_type_layout.addWidget(self.heatmap_data_combo)
        data_layout.addLayout(data_type_layout)
        
        # X-axis range selection
        range_group = QGroupBox("X-axis Range")
        range_layout = QVBoxLayout(range_group)
        
        # Auto range checkbox
        self.heatmap_auto_range = QCheckBox("Auto Range")
        self.heatmap_auto_range.setChecked(True)
        self.heatmap_auto_range.stateChanged.connect(self.on_auto_range_changed)
        range_layout.addWidget(self.heatmap_auto_range)
        
        # Manual range controls
        self.range_controls_widget = QWidget()
        range_controls_layout = QVBoxLayout(self.range_controls_widget)
        range_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        range_min_layout = QHBoxLayout()
        range_min_layout.addWidget(QLabel("Min:"))
        self.heatmap_range_min = QDoubleSpinBox()
        self.heatmap_range_min.setRange(0, 4000)
        self.heatmap_range_min.setValue(200)
        self.heatmap_range_min.valueChanged.connect(self.update_heatmap_plot)
        range_min_layout.addWidget(self.heatmap_range_min)
        range_controls_layout.addLayout(range_min_layout)
        
        range_max_layout = QHBoxLayout()
        range_max_layout.addWidget(QLabel("Max:"))
        self.heatmap_range_max = QDoubleSpinBox()
        self.heatmap_range_max.setRange(0, 4000)
        self.heatmap_range_max.setValue(1800)
        self.heatmap_range_max.valueChanged.connect(self.update_heatmap_plot)
        range_max_layout.addWidget(self.heatmap_range_max)
        range_controls_layout.addLayout(range_max_layout)
        
        range_layout.addWidget(self.range_controls_widget)
        self.range_controls_widget.setEnabled(False)  # Disabled by default (auto range)
        
        data_layout.addWidget(range_group)
        controls_layout.addWidget(data_group)
        
        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance_group)
        
        # Colormap selection
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.heatmap_colormap = QComboBox()
        self.heatmap_colormap.addItems([
            "viridis", "plasma", "inferno", "magma", "cividis", 
            "hot", "cool", "spring", "summer", "autumn", "winter",
            "gray", "bone", "copper", "pink", "RdYlBu", "RdBu", "seismic"
        ])
        self.heatmap_colormap.setCurrentText("viridis")
        self.heatmap_colormap.currentTextChanged.connect(self.update_heatmap_plot)
        colormap_layout.addWidget(self.heatmap_colormap)
        appearance_layout.addLayout(colormap_layout)
        
        # Contrast control
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.heatmap_contrast = QSlider(Qt.Horizontal)
        self.heatmap_contrast.setRange(-100, 100)
        self.heatmap_contrast.setValue(0)
        self.heatmap_contrast.valueChanged.connect(self.update_contrast_label)
        self.heatmap_contrast.valueChanged.connect(self.update_heatmap_plot)
        contrast_layout.addWidget(self.heatmap_contrast)
        self.contrast_label = QLabel("0")
        contrast_layout.addWidget(self.contrast_label)
        appearance_layout.addLayout(contrast_layout)
        
        # Brightness control
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.heatmap_brightness = QSlider(Qt.Horizontal)
        self.heatmap_brightness.setRange(-100, 100)
        self.heatmap_brightness.setValue(0)
        self.heatmap_brightness.valueChanged.connect(self.update_brightness_label)
        self.heatmap_brightness.valueChanged.connect(self.update_heatmap_plot)
        brightness_layout.addWidget(self.heatmap_brightness)
        self.brightness_label = QLabel("0")
        brightness_layout.addWidget(self.brightness_label)
        appearance_layout.addLayout(brightness_layout)
        
        # Gamma control
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.heatmap_gamma = QSlider(Qt.Horizontal)
        self.heatmap_gamma.setRange(10, 300)  # 0.1 to 3.0
        self.heatmap_gamma.setValue(100)  # 1.0
        self.heatmap_gamma.valueChanged.connect(self.update_gamma_label)
        self.heatmap_gamma.valueChanged.connect(self.update_heatmap_plot)
        gamma_layout.addWidget(self.heatmap_gamma)
        self.gamma_label = QLabel("1.0")
        gamma_layout.addWidget(self.gamma_label)
        appearance_layout.addLayout(gamma_layout)
        
        # Anti-aliasing options
        self.heatmap_antialias = QCheckBox("Anti-aliasing")
        self.heatmap_antialias.setChecked(True)
        self.heatmap_antialias.stateChanged.connect(self.update_heatmap_plot)
        appearance_layout.addWidget(self.heatmap_antialias)
        
        # Grid options
        self.heatmap_grid = QCheckBox("Show Grid")
        self.heatmap_grid.setChecked(False)
        self.heatmap_grid.stateChanged.connect(self.update_heatmap_plot)
        appearance_layout.addWidget(self.heatmap_grid)
        
        # Auto renormalization
        self.heatmap_auto_normalize = QCheckBox("Auto Normalize")
        self.heatmap_auto_normalize.setChecked(True)
        self.heatmap_auto_normalize.stateChanged.connect(self.update_heatmap_plot)
        appearance_layout.addWidget(self.heatmap_auto_normalize)
        
        controls_layout.addWidget(appearance_group)
        
        # Interpolation group
        interp_group = QGroupBox("Interpolation")
        interp_layout = QVBoxLayout(interp_group)
        
        # Interpolation method
        interp_method_layout = QHBoxLayout()
        interp_method_layout.addWidget(QLabel("Method:"))
        self.heatmap_interpolation = QComboBox()
        self.heatmap_interpolation.addItems([
            "none", "nearest", "bilinear", "bicubic", "spline16", "spline36"
        ])
        self.heatmap_interpolation.setCurrentText("nearest")
        self.heatmap_interpolation.currentTextChanged.connect(self.update_heatmap_plot)
        interp_method_layout.addWidget(self.heatmap_interpolation)
        interp_layout.addLayout(interp_method_layout)
        
        # Resolution
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        self.heatmap_resolution = QSpinBox()
        self.heatmap_resolution.setRange(50, 1000)
        self.heatmap_resolution.setValue(200)
        self.heatmap_resolution.valueChanged.connect(self.update_heatmap_plot)
        resolution_layout.addWidget(self.heatmap_resolution)
        interp_layout.addLayout(resolution_layout)
        
        controls_layout.addWidget(interp_group)
        
        # Update button
        update_btn = QPushButton("Update Heatmap")
        update_btn.clicked.connect(self.update_heatmap_plot)
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        controls_layout.addWidget(update_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_panel)
        
        # Right panel - visualization
        viz_panel = QWidget()
        viz_layout = QVBoxLayout(viz_panel)
        
        # Create matplotlib figure for heatmap
        self.figure_heatmap = Figure(figsize=(12, 10))
        self.canvas_heatmap = FigureCanvas(self.figure_heatmap)
        self.toolbar_heatmap = NavigationToolbar(self.canvas_heatmap, viz_panel)
        
        viz_layout.addWidget(self.toolbar_heatmap)
        viz_layout.addWidget(self.canvas_heatmap)
        
        layout.addWidget(viz_panel)
        
        return tab

    # Data management methods
    def add_files(self):
        """Add spectrum files for batch processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Spectrum Files", "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_paths:
            for file_path in file_paths:
                if file_path not in self.spectra_files:
                    self.spectra_files.append(file_path)
                    filename = os.path.basename(file_path)
                    self.file_list_widget.addItem(filename)
            
            self.update_file_status()
            
            # Load first spectrum if none loaded
            if len(self.spectra_files) == len(file_paths):
                self.load_spectrum(0)
            
            # Update waterfall and heatmap plots since new files are available
            self.update_waterfall_plot()
            self.update_heatmap_plot()
            
    def remove_selected_files(self):
        """Remove selected files from the list."""
        selected_items = self.file_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select files to remove.")
            return
            
        for item in selected_items:
            row = self.file_list_widget.row(item)
            if 0 <= row < len(self.spectra_files):
                self.spectra_files.pop(row)
                self.file_list_widget.takeItem(row)
        
        self.update_file_status()
        
        # Load first remaining spectrum if current was removed
        if self.spectra_files and self.current_spectrum_index >= len(self.spectra_files):
            self.current_spectrum_index = 0
            self.load_spectrum(0)
        elif not self.spectra_files:
            self.clear_current_spectrum()
        
        # Update waterfall and heatmap plots since files changed
        self.update_waterfall_plot()
        self.update_heatmap_plot()
        
    def on_file_double_click(self, item):
        """Handle double-click on file list item."""
        row = self.file_list_widget.row(item)
        if 0 <= row < len(self.spectra_files):
            self.load_spectrum(row)
            
    def navigate_spectrum(self, direction):
        """Navigate through spectra."""
        if not self.spectra_files:
            return
            
        if direction == 0:  # First
            new_index = 0
        elif direction == -2:  # Last
            new_index = len(self.spectra_files) - 1
        elif direction == -1:  # Previous
            new_index = max(0, self.current_spectrum_index - 1)
        elif direction == 1:  # Next
            new_index = min(len(self.spectra_files) - 1, self.current_spectrum_index + 1)
        else:
            return
            
        if new_index != self.current_spectrum_index:
            self.load_spectrum(new_index)
            
    def load_spectrum(self, index):
        """Load a spectrum by index."""
        if not (0 <= index < len(self.spectra_files)):
            return
            
        file_path = self.spectra_files[index]
        try:
            # Check if we have batch results for this file
            batch_result = None
            if self.batch_results:
                for result in self.batch_results:
                    if result['file'] == file_path and not result.get('fit_failed', True):
                        batch_result = result
                        break
            
            if batch_result:
                # Load fitted data from batch results
                self.wavenumbers = np.array(batch_result['wavenumbers'])
                self.intensities = np.array(batch_result['intensities'])
                self.original_intensities = np.array(batch_result['original_intensities'])
                self.background = np.array(batch_result['background'])
                self.peaks = np.array(batch_result['peaks'])
                self.fit_params = batch_result['fit_params']
                self.fit_result = True
                self.residuals = np.array(batch_result['residuals'])
                
                # Clear manual peaks when loading fitted data
                self.manual_peaks = np.array([], dtype=int)
                
                self.current_spectrum_index = index
                
                # Update UI
                self.update_file_status()
                self.update_peak_count_display()
                self.update_current_plot()
                
                # Update Live View with current spectrum for analysis
                self.update_live_view_with_current_spectrum()
                
                # Highlight current file in list
                self.file_list_widget.setCurrentRow(index)
                
                return
            
            # Load the spectrum data normally if no batch result available
            data = self.load_spectrum_robust(file_path)
            if data is not None:
                self.wavenumbers, self.intensities = data
                self.original_intensities = self.intensities.copy()
                self.current_spectrum_index = index
                
                # Reset peak data for new spectrum
                self.peaks = np.array([], dtype=int)
                self.manual_peaks = np.array([], dtype=int)
                self.fit_params = []
                self.fit_result = None
                self.background = None
                self.residuals = None
                
                # Update UI
                self.update_file_status()
                self.update_peak_count_display()
                self.update_current_plot()
                
                # Highlight current file in list
                self.file_list_widget.setCurrentRow(index)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load spectrum: {str(e)}")
            
    def load_spectrum_robust(self, file_path):
        """Robust spectrum loading with multiple strategies."""
        try:
            # Strategy 1: Try numpy loadtxt
            try:
                data = np.loadtxt(file_path)
                if data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
            except:
                pass
            
            # Strategy 2: Try pandas
            try:
                df = pd.read_csv(file_path, sep=None, engine='python')
                if len(df.columns) >= 2:
                    return df.iloc[:, 0].values, df.iloc[:, 1].values
            except:
                pass
            
            # Strategy 3: Manual parsing
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            wavenumbers = []
            intensities = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            wavenumbers.append(float(parts[0]))
                            intensities.append(float(parts[1]))
                    except ValueError:
                        continue
                        
            if len(wavenumbers) > 0:
                return np.array(wavenumbers), np.array(intensities)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return None
        
    def update_file_status(self):
        """Update file status display."""
        if self.spectra_files:
            filename = os.path.basename(self.spectra_files[self.current_spectrum_index]) if self.current_spectrum_index < len(self.spectra_files) else "None"
            status = f"File {self.current_spectrum_index + 1} of {len(self.spectra_files)}: {filename}"
        else:
            status = "No files loaded"
        self.current_file_label.setText(status)
        
    def clear_current_spectrum(self):
        """Clear current spectrum data."""
        self.wavenumbers = np.array([])
        self.intensities = np.array([])
        self.original_intensities = np.array([])
        self.peaks = np.array([], dtype=int)
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.residuals = None
        self.update_current_plot()
        
    # Peak detection and fitting methods
    def update_lambda_label(self):
        """Update the lambda label based on the slider value."""
        value = self.lambda_slider.value()
        self.lambda_label.setText(f"1e{value}")

    def update_p_label(self):
        """Update the p label based on the slider value."""
        value = self.p_slider.value()
        p_value = value / 1000.0  # Convert to 0.001-0.05 range
        self.p_label.setText(f"{p_value:.3f}")

    def on_bg_method_changed(self):
        """Handle change in background method."""
        method = self.bg_method_combo.currentText()
        # Show ALS parameters only when ALS is selected
        self.als_params_widget.setVisible(method.startswith("ALS"))

    def update_peak_detection(self):
        """Update peak detection in real-time based on slider values."""
        if len(self.intensities) == 0:
            return
            
        # Get current slider values
        height_percent = self.height_slider.value()
        distance = self.distance_slider.value()
        prominence_percent = self.prominence_slider.value()
        
        # Update labels
        self.height_label.setText(f"{height_percent}%")
        self.distance_label.setText(str(distance))
        self.prominence_label.setText(f"{prominence_percent}%")
        
        # Calculate actual values
        max_intensity = np.max(self.intensities)
        height_threshold = (height_percent / 100.0) * max_intensity if height_percent > 0 else None
        prominence_threshold = (prominence_percent / 100.0) * max_intensity if prominence_percent > 0 else None
        
        # Find peaks with current parameters
        try:
            peak_kwargs = {}
            if height_threshold is not None:
                peak_kwargs['height'] = height_threshold
            if distance > 1:
                peak_kwargs['distance'] = distance
            if prominence_threshold is not None:
                peak_kwargs['prominence'] = prominence_threshold
                
            self.peaks, properties = find_peaks(self.intensities, **peak_kwargs)
            
            # Update peak count
            self.peak_count_label.setText(f"Peaks found: {len(self.peaks)}")
            
            # Update peak count display (which should update the list)
            self.update_peak_count_display()
            
            # Update plot
            self.update_current_plot()
            
        except Exception as e:
            self.peak_count_label.setText(f"Peak detection error: {str(e)}")

    def apply_background_subtraction(self):
        """Apply background subtraction to the current spectrum."""
        if len(self.intensities) == 0:
            QMessageBox.warning(self, "No Data", "Load a spectrum first.")
            return
            
        try:
            method = self.bg_method_combo.currentText()
            
            if method.startswith("ALS"):
                # Get ALS parameters from sliders
                lambda_value = 10 ** self.lambda_slider.value()
                p_value = self.p_slider.value() / 1000.0
                
                self.background = self.baseline_als(self.original_intensities, lambda_value, p_value)
                self.intensities = self.original_intensities - self.background
            else:
                QMessageBox.information(self, "Not Implemented", 
                                      f"{method} background subtraction not yet implemented in batch mode.")
                return
                
            self.update_current_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Background subtraction failed: {str(e)}")

    def clear_background_preview(self):
        """Clear any background preview (placeholder for consistency)."""
        # This is for UI consistency with main app - batch mode doesn't use preview
        pass

    def reset_spectrum(self):
        """Reset spectrum to original state."""
        if len(self.original_intensities) > 0:
            self.intensities = self.original_intensities.copy()
            self.background = None
            self.peaks = np.array([], dtype=int)
            self.manual_peaks = np.array([], dtype=int)  # Also clear manual peaks
            self.fit_params = []
            self.fit_result = None
            self.residuals = None
            self.update_peak_count_display()
            self.update_current_plot()

    def apply_smoothing(self):
        """Apply Savitzky-Golay smoothing to the spectrum."""
        if len(self.intensities) == 0:
            QMessageBox.warning(self, "No Data", "Load a spectrum first.")
            return
            
        try:
            window_length = self.sg_window_spin.value()
            poly_order = self.sg_order_spin.value()
            
            # Ensure window length is odd and greater than poly_order
            if window_length % 2 == 0:
                window_length += 1
                self.sg_window_spin.setValue(window_length)
                
            if window_length <= poly_order:
                QMessageBox.warning(
                    self, 
                    "Invalid Parameters", 
                    f"Window length ({window_length}) must be greater than polynomial order ({poly_order})."
                )
                return
                
            # Apply Savitzky-Golay filter
            smoothed = savgol_filter(self.intensities, window_length, poly_order)
            self.intensities = smoothed
            
            # Update plot
            self.update_current_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"Failed to apply smoothing:\n{str(e)}")

    def clear_smoothing_preview(self):
        """Clear any smoothing preview (placeholder for consistency)."""
        # This is for UI consistency with main app - batch mode doesn't use preview
        pass

    def on_model_changed(self):
        """Handle model selection change."""
        self.current_model = self.model_combo.currentText()
        
    def find_peaks(self):
        """Find peaks in the current spectrum - manual trigger."""
        self.update_peak_detection()
        
    def clear_peaks(self):
        """Clear all peaks."""
        self.peaks = np.array([], dtype=int)
        self.manual_peaks = np.array([], dtype=int)  # Also clear manual peaks
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        self.update_peak_count_display()
        self.update_current_plot()

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

    def fit_peaks(self):
        """Fit peaks to the current spectrum."""
        if len(self.peaks) == 0:
            QMessageBox.warning(self, "No Peaks", "Detect peaks first.")
            return
            
        try:
            # Create initial parameter guesses
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            for peak_idx in self.peaks:
                if 0 <= peak_idx < len(self.wavenumbers):
                    # Amplitude
                    amp = self.intensities[peak_idx]
                    # Center
                    cen = self.wavenumbers[peak_idx]
                    # Width (estimate)
                    wid = 10.0
                    
                    initial_params.extend([amp, cen, wid])
                    bounds_lower.extend([amp * 0.1, cen - wid * 2, wid * 0.3])
                    bounds_upper.extend([amp * 10, cen + wid * 2, wid * 3])
                    
            if not initial_params:
                QMessageBox.warning(self, "Error", "No valid peaks for fitting.")
                return
                
            # Fit the peaks
            bounds = (bounds_lower, bounds_upper)
            popt, pcov = curve_fit(
                self.multi_peak_model, 
                self.wavenumbers, 
                self.intensities,
                p0=initial_params,
                bounds=bounds,
                max_nfev=2000
            )
            
            self.fit_params = popt
            self.fit_result = True
            
            # Calculate residuals - ensure we have valid fit_params before proceeding
            if (self.fit_params is not None and 
                hasattr(self.fit_params, '__len__') and 
                len(self.fit_params) > 0):
                fitted_curve = self.multi_peak_model(self.wavenumbers, *popt)
                self.residuals = self.intensities - fitted_curve
                
                # Update the current plot
                self.update_current_plot()
            else:
                QMessageBox.warning(self, "Error", "Invalid fit parameters generated.")
                return
            
            # Calculate R-squared
            ss_res = np.sum(self.residuals ** 2)
            ss_tot = np.sum((self.intensities - np.mean(self.intensities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            QMessageBox.information(self, "Success", 
                                  f"Peak fitting completed!\nR² = {r_squared:.4f}\nFitted {len(self.peaks)} peaks")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Peak fitting failed: {str(e)}")

    def gaussian(self, x, amp, cen, wid):
        """Gaussian peak function."""
        return amp * np.exp(-((x - cen) / wid)**2)
        
    def lorentzian(self, x, amp, cen, wid):
        """Lorentzian peak function."""
        return amp / (1 + ((x - cen) / wid)**2)
        
    def multi_peak_model(self, x, *params):
        """Multi-peak model function."""
        # Calculate number of peaks from parameters (3 params per peak)
        n_peaks = len(params) // 3
        
        if n_peaks == 0:
            return np.zeros_like(x)
            
        model = np.zeros_like(x)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(params):
                amp, cen, wid = params[start_idx:start_idx+3]
                wid = max(abs(wid), 1.0)  # Ensure positive width
                
                if self.current_model == "Gaussian":
                    component = self.gaussian(x, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    component = self.lorentzian(x, amp, cen, wid)
                else:
                    component = self.gaussian(x, amp, cen, wid)  # Default
                    
                model += component
                
        return model

    # Batch processing methods
    def set_reference(self):
        """Set current spectrum as reference for batch processing."""
        if len(self.peaks) == 0:
            QMessageBox.warning(self, "No Peaks", "Detect and fit peaks first.")
            return
            
        if not self.fit_result:
            QMessageBox.warning(self, "No Fit", "Fit peaks first.")
            return
            
        self.reference_peaks = self.peaks.copy()
        self.reference_background = self.background.copy() if (hasattr(self, 'background') and self.background is not None) else None
        
        QMessageBox.information(self, "Reference Set", 
                              f"Set reference with {len(self.peaks)} peaks.")
        
        self.batch_status_text.append(f"Reference set: {len(self.peaks)} peaks from {os.path.basename(self.spectra_files[self.current_spectrum_index])}")
        
    def apply_to_all(self):
        """Apply reference parameters to all loaded spectra."""
        if self.reference_peaks is None:
            QMessageBox.warning(self, "No Reference", "Set a reference spectrum first.")
            return
            
        if not self.spectra_files:
            QMessageBox.warning(self, "No Files", "Load spectrum files first.")
            return
            
        self._stop_batch = False
        self.batch_results = []
        
        # Switch to live view tab during processing
        self.viz_tab_widget.setCurrentIndex(1)  # Live View tab is now index 1 in visualization panel
        
        # Create progress dialog
        progress = QProgressDialog("Processing spectra...", "Stop", 0, len(self.spectra_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        for i, file_path in enumerate(self.spectra_files):
            if self._stop_batch:
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Processing {os.path.basename(file_path)}...")
            
            # Update live view with current file
            self.current_file_info.setText(f"Processing: {os.path.basename(file_path)} ({i+1}/{len(self.spectra_files)})")
            QApplication.processEvents()
            
            try:
                # Load spectrum
                data = self.load_spectrum_robust(file_path)
                if data is None:
                    self.batch_results.append({
                        'file': file_path,
                        'fit_failed': True,
                        'error': 'Failed to load spectrum'
                    })
                    self.batch_status_text.append(f"FAILED: {os.path.basename(file_path)} - Failed to load")
                    continue
                    
                wavenumbers, intensities = data
                original_intensities = intensities.copy()
                
                # Always apply background subtraction with current parameters
                bg_method = self.bg_method_combo.currentText()
                if bg_method.startswith("ALS"):
                    # Get current ALS parameters from UI
                    lambda_value = 10 ** self.lambda_slider.value()
                    p_value = self.p_slider.value() / 1000.0
                    
                    # Apply background subtraction to this spectrum
                    background = self.baseline_als(original_intensities, lambda_value, p_value)
                    intensities = original_intensities - background
                else:
                    # For now, fall back to ALS with default parameters if other methods selected
                    # TODO: Implement other background methods
                    background = self.baseline_als(original_intensities, 1e5, 0.01)
                    intensities = original_intensities - background
                
                # Fit peaks using reference positions as initial guess
                initial_params = []
                bounds_lower = []
                bounds_upper = []
                
                for peak_idx in self.reference_peaks:
                    # Find closest wavenumber in current spectrum
                    closest_idx = np.argmin(np.abs(wavenumbers - self.wavenumbers[peak_idx]))
                    
                    if 0 <= closest_idx < len(wavenumbers):
                        amp = intensities[closest_idx]
                        cen = wavenumbers[closest_idx]
                        wid = 10.0
                        
                        # Ensure amplitude is positive for bounds calculation
                        min_amp = max(0.1, abs(amp) * 0.1)  # Minimum positive amplitude
                        max_amp = abs(amp) * 10 if abs(amp) > 0 else 1000  # Maximum amplitude
                        
                        # Position bounds - allow reasonable variation around center
                        min_cen = cen - wid * 2
                        max_cen = cen + wid * 2
                        
                        # Width bounds - ensure positive width
                        min_wid = wid * 0.1
                        max_wid = wid * 5
                        
                        # Validate bounds to ensure lower < upper
                        if min_amp >= max_amp:
                            max_amp = min_amp * 10
                        if min_cen >= max_cen:
                            max_cen = min_cen + 20  # At least 20 cm⁻¹ range
                        if min_wid >= max_wid:
                            max_wid = min_wid * 10
                        
                        initial_params.extend([abs(amp), cen, wid])  # Use absolute amplitude
                        bounds_lower.extend([min_amp, min_cen, min_wid])
                        bounds_upper.extend([max_amp, max_cen, max_wid])
                
                if initial_params:
                    try:
                        # Validate bounds arrays before fitting
                        bounds_lower = np.array(bounds_lower)
                        bounds_upper = np.array(bounds_upper)
                        
                        # Ensure all lower bounds are strictly less than upper bounds
                        invalid_bounds = bounds_lower >= bounds_upper
                        if np.any(invalid_bounds):
                            # Fix invalid bounds
                            for i in np.where(invalid_bounds)[0]:
                                if bounds_lower[i] >= bounds_upper[i]:
                                    bounds_upper[i] = bounds_lower[i] * 2 + 1
                        
                        # Double-check all bounds are valid
                        if np.any(bounds_lower >= bounds_upper):
                            raise ValueError("Unable to create valid parameter bounds")
                        
                        # Temporarily store data for model function
                        temp_wavenumbers = self.wavenumbers
                        temp_peaks = self.peaks
                        self.wavenumbers = wavenumbers
                        self.peaks = self.reference_peaks
                        
                        bounds = (bounds_lower.tolist(), bounds_upper.tolist())
                        popt, pcov = curve_fit(
                            self.multi_peak_model, 
                            wavenumbers, 
                            intensities,
                            p0=initial_params,
                            bounds=bounds,
                            max_nfev=2000,
                            method='trf'  # Trust Region Reflective algorithm is more robust
                        )
                        
                        # Restore original data
                        self.wavenumbers = temp_wavenumbers
                        self.peaks = temp_peaks
                        
                        # Calculate R-squared
                        fitted_curve = self.multi_peak_model(wavenumbers, *popt)
                        residuals = intensities - fitted_curve
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        # Store results
                        result_data = {
                            'file': file_path,
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'original_intensities': original_intensities,
                            'background': background,
                            'peaks': self.reference_peaks.copy(),
                            'fit_params': popt,
                            'fit_cov': pcov,
                            'residuals': residuals,
                            'r_squared': r_squared,
                            'fitted_curve': fitted_curve,  # Add fitted curve for live view
                            'fit_failed': False
                        }
                        
                        self.batch_results.append(result_data)
                        
                        # Update live view
                        self.update_live_view(result_data, wavenumbers, popt)
                        
                        self.batch_status_text.append(f"SUCCESS: {os.path.basename(file_path)} - R² = {r_squared:.4f}")
                        
                    except Exception as e:
                        # Try fallback fitting without bounds if bounds caused the issue
                        fallback_success = False
                        if "bound" in str(e).lower():
                            try:
                                # Temporarily store data for model function
                                temp_wavenumbers = self.wavenumbers
                                temp_peaks = self.peaks
                                self.wavenumbers = wavenumbers
                                self.peaks = self.reference_peaks
                                
                                # Try fitting without bounds
                                popt, pcov = curve_fit(
                                    self.multi_peak_model, 
                                    wavenumbers, 
                                    intensities,
                                    p0=initial_params,
                                    max_nfev=2000,
                                    method='lm'  # Levenberg-Marquardt for unbounded problems
                                )
                                
                                # Restore original data
                                self.wavenumbers = temp_wavenumbers
                                self.peaks = temp_peaks
                                
                                # Calculate R-squared
                                fitted_curve = self.multi_peak_model(wavenumbers, *popt)
                                residuals = intensities - fitted_curve
                                ss_res = np.sum(residuals ** 2)
                                ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                                
                                # Store results
                                result_data = {
                                    'file': file_path,
                                    'wavenumbers': wavenumbers,
                                    'intensities': intensities,
                                    'original_intensities': original_intensities,
                                    'background': background,
                                    'peaks': self.reference_peaks.copy(),
                                    'fit_params': popt,
                                    'fit_cov': pcov,
                                    'residuals': residuals,
                                    'r_squared': r_squared,
                                    'fitted_curve': fitted_curve,  # Add fitted curve for live view
                                    'fit_failed': False
                                }
                                
                                self.batch_results.append(result_data)
                                
                                # Update live view
                                self.update_live_view(result_data, wavenumbers, popt)
                                
                                self.batch_status_text.append(f"SUCCESS (fallback): {os.path.basename(file_path)} - R² = {r_squared:.4f}")
                                fallback_success = True
                                
                            except Exception as fallback_e:
                                # Fallback also failed
                                pass
                        
                        # If fallback didn't work or wasn't attempted, record the failure
                        if not fallback_success:
                            self.batch_results.append({
                                'file': file_path,
                                'fit_failed': True,
                                'error': str(e)
                            })
                            
                            # Provide more detailed error information
                            error_msg = str(e)
                            if "bound" in error_msg.lower():
                                error_msg += f" (Peaks: {len(self.reference_peaks)}, Params: {len(initial_params) if initial_params else 0})"
                            
                            self.batch_status_text.append(f"FAILED: {os.path.basename(file_path)} - {error_msg}")
                            
                            # Log additional debug info for bounds errors
                            if "bound" in str(e).lower() and len(initial_params) > 0:
                                self.batch_status_text.append(f"  Debug: Initial params range: {min(initial_params):.3f} to {max(initial_params):.3f}")
                                if len(bounds_lower) > 0 and len(bounds_upper) > 0:
                                    self.batch_status_text.append(f"  Debug: Bounds range: [{min(bounds_lower):.3f}, {max(bounds_upper):.3f}]")
                        
                else:
                    self.batch_results.append({
                        'file': file_path,
                        'fit_failed': True,
                        'error': 'No valid peak parameters'
                    })
                    self.batch_status_text.append(f"FAILED: {os.path.basename(file_path)} - No valid peaks")
                    
            except Exception as e:
                self.batch_results.append({
                    'file': file_path,
                    'fit_failed': True,
                    'error': str(e)
                })
                self.batch_status_text.append(f"FAILED: {os.path.basename(file_path)} - {str(e)}")
        
        progress.setValue(len(self.spectra_files))
        progress.close()
        
        # Update visibility controls and trends plot
        self.update_peak_visibility_controls()
        self.update_trends_plot()
        
        successful = sum(1 for result in self.batch_results if not result.get('fit_failed', True))
        self.batch_status_text.append(f"\nBatch processing complete: {successful}/{len(self.spectra_files)} successful")
        
        QMessageBox.information(self, "Batch Complete", 
                              f"Processed {len(self.spectra_files)} spectra.\n"
                              f"Successful: {successful}\n"
                              f"Failed: {len(self.spectra_files) - successful}")
        
    def stop_batch(self):
        """Stop batch processing."""
        self._stop_batch = True
        self.batch_status_text.append("Batch processing stopped by user.")
        
    # Plotting methods
    def initial_plot(self):
        """Create initial plot."""
        self.update_current_plot()
        
    def update_current_plot(self):
        """Update current spectrum plot."""
        # Clear axes
        self.ax_main.clear()
        self.ax_residual.clear()
        
        if len(self.wavenumbers) == 0:
            self.ax_main.text(0.5, 0.5, 'No spectrum loaded', 
                             ha='center', va='center', transform=self.ax_main.transAxes)
            self.canvas_current.draw()
            return
        
        # Main spectrum plot
        self.ax_main.plot(self.wavenumbers, self.intensities, 'b-', 
                         linewidth=1.5, label='Spectrum')
        
        # Plot background if available
        if self.background is not None:
            self.ax_main.plot(self.wavenumbers, self.background, 'r--', 
                             linewidth=1, alpha=0.7, label='Background')
        
        # Plot peaks
        if len(self.peaks) > 0:
            peak_positions = [self.wavenumbers[int(p)] for p in self.peaks if 0 <= int(p) < len(self.wavenumbers)]
            peak_intensities = [self.intensities[int(p)] for p in self.peaks if 0 <= int(p) < len(self.intensities)]
            self.ax_main.plot(peak_positions, peak_intensities, 'ro', 
                             markersize=8, label='Auto Peaks')
        
        # Plot manual peaks
        if hasattr(self, 'manual_peaks') and len(self.manual_peaks) > 0:
            manual_peak_positions = [self.wavenumbers[int(p)] for p in self.manual_peaks if 0 <= int(p) < len(self.wavenumbers)]
            manual_peak_intensities = [self.intensities[int(p)] for p in self.manual_peaks if 0 <= int(p) < len(self.intensities)]
            self.ax_main.plot(manual_peak_positions, manual_peak_intensities, 'gs', 
                             markersize=10, label='Manual Peaks', alpha=0.8, markeredgecolor='darkgreen')
        
        # Show interactive mode status
        if hasattr(self, 'interactive_mode') and self.interactive_mode:
            self.ax_main.text(0.02, 0.98, '🖱️ Interactive Mode ON\nClick to add/remove peaks',
                             transform=self.ax_main.transAxes, fontsize=10, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Plot fitted curve if available
        if (self.fit_result and 
            self.fit_params is not None and 
            hasattr(self.fit_params, '__len__') and 
            len(self.fit_params) > 0):
            fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
            self.ax_main.plot(self.wavenumbers, fitted_curve, 'g-', 
                             linewidth=2, label='Fitted')
            
            # Plot individual peaks if requested
            if self.show_individual_peaks:
                self.plot_individual_peaks()
        
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Current Spectrum')
        self.ax_main.legend()
        if self.show_grid_check.isChecked():
            self.ax_main.grid(True, alpha=0.3)
        
        # Residuals plot
        if self.residuals is not None:
            self.ax_residual.plot(self.wavenumbers, self.residuals, 'k-', linewidth=1)
            self.ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_residual.set_ylabel('Residuals')
            self.ax_residual.set_title('Fit Residuals')
            if self.show_grid_check.isChecked():
                self.ax_residual.grid(True, alpha=0.3)
        
        self.canvas_current.draw()
        
        # Update live view when current plot updates
        if hasattr(self, 'wavenumbers') and len(self.wavenumbers) > 0:
            try:
                self.update_live_view_with_current_spectrum()
            except:
                pass  # Ignore errors during initialization
    
    def plot_individual_peaks(self):
        """Plot individual fitted peaks."""
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0 or 
            len(self.peaks) == 0):
            return
            
        n_peaks = len(self.peaks)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(self.fit_params):
                amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                
                if self.current_model == "Gaussian":
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    peak_curve = self.lorentzian(self.wavenumbers, amp, cen, wid)
                else:
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                
                # Calculate individual R² for this peak
                individual_r2 = self.calculate_individual_peak_r2(i, self.fit_params)
                    
                # Plot individual peak curve with individual R² in legend
                label = f'Peak {i+1}: {cen:.0f}cm⁻¹ (R²={individual_r2:.3f})'
                self.ax_main.plot(self.wavenumbers, peak_curve, '--', 
                                linewidth=1, alpha=0.7, label=label)
    
    def update_peak_visibility_controls(self):
        """Update peak visibility controls based on batch results."""
        # Clear existing controls
        for i in reversed(range(self.peak_visibility_layout.count())):
            self.peak_visibility_layout.itemAt(i).widget().setParent(None)
        
        self.peak_visibility_vars = []
        
        if not self.batch_results:
            return
            
        # Get number of peaks from reference
        if self.reference_peaks is not None:
            n_peaks = len(self.reference_peaks)
            
            for i in range(n_peaks):
                checkbox = QCheckBox(f"Peak {i+1}")
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(self.update_trends_plot)
                self.peak_visibility_layout.addWidget(checkbox)
                self.peak_visibility_vars.append(checkbox)
    
    def show_all_peaks(self):
        """Show all peaks in trends plot."""
        for checkbox in self.peak_visibility_vars:
            checkbox.setChecked(True)
        self.update_trends_plot()
        
    def hide_all_peaks(self):
        """Hide all peaks in trends plot."""
        for checkbox in self.peak_visibility_vars:
            checkbox.setChecked(False)
        self.update_trends_plot()
    
    def update_trends_plot(self):
        """Update trends plot showing peak parameters across spectra."""
        if not hasattr(self, 'figure_trends'):
            return
            
        self.figure_trends.clear()
        
        if not self.batch_results:
            ax = self.figure_trends.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No batch results available', 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas_trends.draw()
            return
        
        # Extract successful results
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            ax = self.figure_trends.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No successful fits available', 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas_trends.draw()
            return
        
        # Create subplots
        fig = self.figure_trends
        ax1 = fig.add_subplot(2, 2, 1)  # Position
        ax2 = fig.add_subplot(2, 2, 2)  # Amplitude
        ax3 = fig.add_subplot(2, 2, 3)  # Width
        ax4 = fig.add_subplot(2, 2, 4)  # R-squared
        
        # Extract data
        file_indices = list(range(len(successful_results)))
        
        n_peaks = len(self.reference_peaks) if self.reference_peaks is not None else 0
        
        for peak_idx in range(n_peaks):
            if peak_idx >= len(self.peak_visibility_vars) or not self.peak_visibility_vars[peak_idx].isChecked():
                continue
                
            positions = []
            amplitudes = []
            widths = []
            
            for result in successful_results:
                if 'fit_params' in result and result['fit_params'] is not None:
                    params = result['fit_params']
                    start_idx = peak_idx * 3
                    if start_idx + 2 < len(params):
                        amp, cen, wid = params[start_idx:start_idx+3]
                        positions.append(cen)
                        amplitudes.append(amp)
                        widths.append(wid)
                    else:
                        positions.append(np.nan)
                        amplitudes.append(np.nan)
                        widths.append(np.nan)
                else:
                    positions.append(np.nan)
                    amplitudes.append(np.nan)
                    widths.append(np.nan)
            
            # Plot trends
            label = f'Peak {peak_idx + 1}'
            ax1.plot(file_indices, positions, 'o-', label=label, alpha=0.7)
            ax2.plot(file_indices, amplitudes, 'o-', label=label, alpha=0.7)
            ax3.plot(file_indices, widths, 'o-', label=label, alpha=0.7)
        
        # Plot R-squared
        r_squared_values = [result.get('r_squared', 0) for result in successful_results]
        ax4.plot(file_indices, r_squared_values, 'ko-', label='R²')
        
        # Format plots
        ax1.set_title('Peak Positions')
        ax1.set_ylabel('Wavenumber (cm⁻¹)')
        ax1.legend()
        if self.show_grid_check.isChecked():
            ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Peak Amplitudes')
        ax2.set_ylabel('Intensity')
        ax2.legend()
        if self.show_grid_check.isChecked():
            ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Peak Widths')
        ax3.set_xlabel('Spectrum Index')
        ax3.set_ylabel('Width')
        ax3.legend()
        if self.show_grid_check.isChecked():
            ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Fit Quality (R²)')
        ax4.set_xlabel('Spectrum Index')
        ax4.set_ylabel('R²')
        # Auto-range R² plot to better show variation
        if r_squared_values and len(r_squared_values) > 0:
            r2_min = min(r_squared_values)
            r2_max = max(r_squared_values)
            if r2_max > r2_min:
                # Add 5% padding to top and bottom
                r2_range = r2_max - r2_min
                ax4.set_ylim(max(0, r2_min - 0.05 * r2_range), min(1, r2_max + 0.05 * r2_range))
            else:
                # If all values are the same, show a small range around that value
                ax4.set_ylim(max(0, r2_min - 0.1), min(1, r2_max + 0.1))
        else:
            ax4.set_ylim(0, 1)  # Fallback to full range if no data
        if self.show_grid_check.isChecked():
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.canvas_trends.draw()
    
    def update_waterfall_plot(self):
        """Update waterfall plot with comprehensive controls."""
        if not hasattr(self, 'figure_waterfall'):
            return
            
        self.figure_waterfall.clear()
        ax = self.figure_waterfall.add_subplot(1, 1, 1)
        
        # First try to use loaded spectrum files directly
        if self.spectra_files:
            try:
                # Get control settings
                data_type = self.waterfall_data_combo.currentText()
                skip = self.waterfall_skip_spin.value()
                y_offset = self.waterfall_offset_spin.value()
                auto_offset = self.waterfall_auto_offset.isChecked()
                color_scheme = self.waterfall_color_scheme.currentText()
                colormap_name = self.waterfall_colormap.currentText()
                line_width = self.waterfall_line_width.value() / 10.0
                alpha = self.waterfall_alpha.value() / 100.0
                contrast = self.waterfall_contrast.value() / 100.0
                brightness = self.waterfall_brightness.value() / 100.0
                gamma = self.waterfall_gamma.value() / 100.0
                show_labels = self.waterfall_show_labels.isChecked()
                auto_normalize = self.waterfall_auto_normalize.isChecked()
                
                # Load spectra data directly from files
                loaded_spectra = []
                for file_path in self.spectra_files:
                    spectrum_data = self.load_spectrum_robust(file_path)
                    if spectrum_data is not None:
                        wavenumbers, intensities = spectrum_data
                        original_intensities = intensities.copy()
                        
                        # Apply background subtraction if needed for corrected data
                        if data_type == "Background Corrected":
                            bg_method = self.bg_method_combo.currentText()
                            if bg_method.startswith("ALS"):
                                lambda_value = 10 ** self.lambda_slider.value()
                                p_value = self.p_slider.value() / 1000.0
                                background = self.baseline_als(original_intensities, lambda_value, p_value)
                                intensities = original_intensities - background
                            else:
                                background = self.baseline_als(original_intensities, 1e5, 0.01)
                                intensities = original_intensities - background
                        
                        loaded_spectra.append({
                            'file': file_path,
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'original_intensities': original_intensities
                        })
                
                if not loaded_spectra:
                    ax.text(0.5, 0.5, 'No valid spectrum files could be loaded', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    self.canvas_waterfall.draw()
                    return
                
                # Use loaded spectra
                data_source = loaded_spectra
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading spectrum files:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
                self.canvas_waterfall.draw()
                return
                
        # Fallback to batch results if no files loaded
        elif self.batch_results:
            # Get successful results
            successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
            
            if not successful_results:
                ax.text(0.5, 0.5, 'No successful fits available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas_waterfall.draw()
                return
            
            data_source = successful_results
            
            # Get control settings
            data_type = self.waterfall_data_combo.currentText()
            skip = self.waterfall_skip_spin.value()
            y_offset = self.waterfall_offset_spin.value()
            auto_offset = self.waterfall_auto_offset.isChecked()
            color_scheme = self.waterfall_color_scheme.currentText()
            colormap_name = self.waterfall_colormap.currentText()
            line_width = self.waterfall_line_width.value() / 10.0
            alpha = self.waterfall_alpha.value() / 100.0
            contrast = self.waterfall_contrast.value() / 100.0
            brightness = self.waterfall_brightness.value() / 100.0
            gamma = self.waterfall_gamma.value() / 100.0
            show_labels = self.waterfall_show_labels.isChecked()
            auto_normalize = self.waterfall_auto_normalize.isChecked()
            
        else:
            ax.text(0.5, 0.5, 'No spectrum files loaded\nAdd files in File Selection tab', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas_waterfall.draw()
            return
        
        try:
            # X-axis range
            if self.waterfall_auto_range.isChecked():
                # Find common range across all spectra
                all_wavenumbers = []
                for result in data_source:
                    if 'wavenumbers' in result:
                        all_wavenumbers.extend(result['wavenumbers'])
                if all_wavenumbers:
                    x_min, x_max = min(all_wavenumbers), max(all_wavenumbers)
                else:
                    x_min, x_max = 200, 1800
            else:
                x_min = self.waterfall_range_min.value()
                x_max = self.waterfall_range_max.value()
            
            # Select spectra to plot
            plotted_results = data_source[::skip]
            
            # Calculate auto offset if needed
            if auto_offset and plotted_results:
                max_intensities = []
                for result in plotted_results:
                    if 'wavenumbers' in result:
                        wavenumbers = np.array(result['wavenumbers'])
                        # Get data based on type and source
                        if data_type == "Raw Intensity":
                            intensities = np.array(result.get('original_intensities', []))
                        elif data_type == "Background Corrected":
                            intensities = np.array(result.get('intensities', []))
                        elif data_type == "Fitted Peaks" and 'fit_params' in result:
                            if result['fit_params'] is not None:
                                temp_wavenumbers = self.wavenumbers
                                temp_peaks = self.peaks
                                self.wavenumbers = wavenumbers
                                self.peaks = self.reference_peaks if self.reference_peaks is not None else []
                                intensities = self.multi_peak_model(wavenumbers, *result['fit_params'])
                                self.wavenumbers = temp_wavenumbers
                                self.peaks = temp_peaks
                            else:
                                intensities = np.array(result.get('intensities', []))
                        elif data_type == "Residuals":
                            intensities = np.array(result.get('residuals', []))
                        else:
                            intensities = np.array(result.get('intensities', []))
                        
                        if len(intensities) > 0:
                            # Filter to X range
                            mask = (wavenumbers >= x_min) & (wavenumbers <= x_max)
                            if np.any(mask):
                                filtered_intensities = intensities[mask]
                                max_intensities.append(np.max(filtered_intensities) - np.min(filtered_intensities))
                
                if max_intensities:
                    y_offset = np.mean(max_intensities) * 1.2
                else:
                    y_offset = 100
            
            # Set up colors
            if color_scheme == "Individual Colors":
                colors = plt.cm.tab10(np.linspace(0, 1, len(plotted_results)))
            elif color_scheme == "Gradient":
                colormap = plt.cm.get_cmap(colormap_name)
                colors = colormap(np.linspace(0, 1, len(plotted_results)))
            elif color_scheme == "Single Color":
                colors = ['blue'] * len(plotted_results)
            else:  # By Intensity
                # Calculate intensities for coloring
                intensity_values = []
                for result in plotted_results:
                    if 'intensities' in result:
                        intensity_values.append(np.mean(result['intensities']))
                    else:
                        intensity_values.append(0)
                if intensity_values:
                    norm_intensities = np.array(intensity_values)
                    norm_intensities = (norm_intensities - np.min(norm_intensities)) / (np.max(norm_intensities) - np.min(norm_intensities))
                    colormap = plt.cm.get_cmap(colormap_name)
                    colors = colormap(norm_intensities)
                else:
                    colors = ['blue'] * len(plotted_results)
            
            # Plot spectra
            legend_entries = []
            for i, result in enumerate(plotted_results):
                if 'wavenumbers' in result:
                    wavenumbers = np.array(result['wavenumbers'])
                    
                    # Get data based on type
                    if data_type == "Raw Intensity":
                        intensities = np.array(result.get('original_intensities', []))
                    elif data_type == "Background Corrected":
                        intensities = np.array(result.get('intensities', []))
                    elif data_type == "Fitted Peaks" and 'fit_params' in result:
                        if result['fit_params'] is not None:
                            temp_wavenumbers = self.wavenumbers
                            temp_peaks = self.peaks
                            self.wavenumbers = wavenumbers
                            self.peaks = self.reference_peaks if self.reference_peaks is not None else []
                            intensities = self.multi_peak_model(wavenumbers, *result['fit_params'])
                            self.wavenumbers = temp_wavenumbers
                            self.peaks = temp_peaks
                        else:
                            intensities = np.array(result.get('intensities', []))
                    elif data_type == "Residuals":
                        intensities = np.array(result.get('residuals', []))
                    else:
                        intensities = np.array(result.get('intensities', []))
                    
                    if len(intensities) > 0:
                        # Apply normalization if requested
                        if auto_normalize:
                            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
                        
                        # Apply enhancement
                        intensities = intensities + brightness
                        intensities = intensities * (1 + contrast)
                        intensities = np.sign(intensities) * np.power(np.abs(intensities), gamma)
                        
                        # Apply Y offset
                        intensities = intensities + i * y_offset
                        
                        # Filter to X range
                        mask = (wavenumbers >= x_min) & (wavenumbers <= x_max)
                        if np.any(mask):
                            plot_wavenumbers = wavenumbers[mask]
                            plot_intensities = intensities[mask]
                            
                            color = colors[i] if i < len(colors) else 'blue'
                            filename = os.path.basename(result['file'])
                            
                            line = ax.plot(plot_wavenumbers, plot_intensities, 
                                         color=color, linewidth=line_width, alpha=alpha)[0]
                            
                            if show_labels and i < 15:  # Limit legend entries
                                legend_entries.append((line, filename))
            
            # Configure plot
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity (offset)')
            ax.set_title(f'Waterfall Plot: {data_type}')
            ax.set_xlim(x_min, x_max)
            
            # Grid
            if self.waterfall_grid.isChecked():
                ax.grid(True, alpha=0.3)
            
            # Anti-aliasing
            if not self.waterfall_antialias.isChecked():
                for line in ax.get_lines():
                    line.set_rasterized(True)
            
            # Legend
            if legend_entries and len(legend_entries) <= 15:
                lines, labels = zip(*legend_entries)
                ax.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            self.figure_waterfall.tight_layout()
            self.canvas_waterfall.draw()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating waterfall plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            self.canvas_waterfall.draw()
    
    # Heatmap methods
    def update_contrast_label(self):
        """Update contrast label."""
        value = self.heatmap_contrast.value()
        self.contrast_label.setText(str(value))
        
    def update_brightness_label(self):
        """Update brightness label."""
        value = self.heatmap_brightness.value()
        self.brightness_label.setText(str(value))
        
    def update_gamma_label(self):
        """Update gamma label."""
        value = self.heatmap_gamma.value() / 100.0
        self.gamma_label.setText(f"{value:.1f}")
        
    def on_auto_range_changed(self):
        """Handle auto range checkbox change."""
        auto_enabled = self.heatmap_auto_range.isChecked()
        self.range_controls_widget.setEnabled(not auto_enabled)
        if auto_enabled:
            self.update_heatmap_plot()
    
    def update_heatmap_plot(self):
        """Update heatmap plot with current settings."""
        if not hasattr(self, 'figure_heatmap'):
            return
            
        self.figure_heatmap.clear()
        
        # First try to use loaded spectrum files directly
        if self.spectra_files:
            try:
                # Load spectra data directly from files
                loaded_spectra = []
                data_type = self.heatmap_data_combo.currentText()
                
                for file_path in self.spectra_files:
                    spectrum_data = self.load_spectrum_robust(file_path)
                    if spectrum_data is not None:
                        wavenumbers, intensities = spectrum_data
                        original_intensities = intensities.copy()
                        
                        # Apply background subtraction if needed for corrected data
                        if data_type == "Background Corrected":
                            bg_method = self.bg_method_combo.currentText()
                            if bg_method.startswith("ALS"):
                                lambda_value = 10 ** self.lambda_slider.value()
                                p_value = self.p_slider.value() / 1000.0
                                background = self.baseline_als(original_intensities, lambda_value, p_value)
                                intensities = original_intensities - background
                            else:
                                background = self.baseline_als(original_intensities, 1e5, 0.01)
                                intensities = original_intensities - background
                        
                        loaded_spectra.append({
                            'file': file_path,
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'original_intensities': original_intensities
                        })
                
                if not loaded_spectra:
                    ax = self.figure_heatmap.add_subplot(1, 1, 1)
                    ax.text(0.5, 0.5, 'No valid spectrum files could be loaded', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    self.canvas_heatmap.draw()
                    return
                
                # Use loaded spectra
                data_source = loaded_spectra
                
            except Exception as e:
                ax = self.figure_heatmap.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, f'Error loading spectrum files:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
                self.canvas_heatmap.draw()
                return
                
        # Fallback to batch results if no files loaded
        elif self.batch_results:
            # Get successful results
            successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
            
            if not successful_results:
                ax = self.figure_heatmap.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No successful fits available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas_heatmap.draw()
                return
            
            data_source = successful_results
            data_type = self.heatmap_data_combo.currentText()
            
        else:
            ax = self.figure_heatmap.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No spectrum files loaded\nAdd files in File Selection tab', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas_heatmap.draw()
            return
        
        try:
            # Collect data for heatmap
            heatmap_data = []
            wavenumber_ranges = []
            
            for result in data_source:
                wavenumbers = np.array(result.get('wavenumbers', []))
                
                if data_type == "Raw Intensity":
                    intensities = np.array(result.get('original_intensities', []))
                elif data_type == "Background Corrected":
                    intensities = np.array(result.get('intensities', []))
                elif data_type == "Fitted Peaks" and 'fit_params' in result:
                    # Use fitted curve if available
                    if result['fit_params'] is not None:
                        # Temporarily store current data to use model function
                        temp_wavenumbers = self.wavenumbers
                        temp_peaks = self.peaks
                        self.wavenumbers = wavenumbers
                        self.peaks = self.reference_peaks if self.reference_peaks is not None else []
                        
                        intensities = self.multi_peak_model(wavenumbers, *result['fit_params'])
                        
                        # Restore original data
                        self.wavenumbers = temp_wavenumbers
                        self.peaks = temp_peaks
                    else:
                        intensities = np.array(result.get('intensities', []))
                elif data_type == "Residuals":
                    intensities = np.array(result.get('residuals', []))
                else:
                    intensities = np.array(result.get('intensities', []))
                
                if len(wavenumbers) > 0 and len(intensities) > 0:
                    heatmap_data.append(intensities)
                    wavenumber_ranges.append(wavenumbers)
            
            if not heatmap_data:
                ax = self.figure_heatmap.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No valid data for heatmap', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas_heatmap.draw()
                return
            
            # Determine wavenumber range
            if self.heatmap_auto_range.isChecked():
                # Find common range across all spectra
                all_wavenumbers = np.concatenate(wavenumber_ranges)
                wn_min, wn_max = np.min(all_wavenumbers), np.max(all_wavenumbers)
            else:
                wn_min = self.heatmap_range_min.value()
                wn_max = self.heatmap_range_max.value()
            
            # Create common wavenumber grid
            resolution = self.heatmap_resolution.value()
            common_wavenumbers = np.linspace(wn_min, wn_max, resolution)
            
            # Interpolate all spectra to common grid
            interpolated_data = []
            for i, (wavenumbers, intensities) in enumerate(zip(wavenumber_ranges, heatmap_data)):
                # Only interpolate if we have data in the range
                if wavenumbers.max() >= wn_min and wavenumbers.min() <= wn_max:
                    interp_intensities = np.interp(common_wavenumbers, wavenumbers, intensities)
                    interpolated_data.append(interp_intensities)
                else:
                    # Fill with zeros if no overlap
                    interpolated_data.append(np.zeros(len(common_wavenumbers)))
            
            if not interpolated_data:
                ax = self.figure_heatmap.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No data in selected range', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas_heatmap.draw()
                return
            
            # Convert to 2D array
            heatmap_array = np.array(interpolated_data)
            
            # Apply normalization if requested
            if self.heatmap_auto_normalize.isChecked():
                # Normalize each spectrum individually
                for i in range(heatmap_array.shape[0]):
                    spectrum = heatmap_array[i, :]
                    if np.max(spectrum) > np.min(spectrum):
                        heatmap_array[i, :] = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
            
            # Apply contrast, brightness, and gamma adjustments
            contrast = self.heatmap_contrast.value() / 100.0
            brightness = self.heatmap_brightness.value() / 100.0
            gamma = self.heatmap_gamma.value() / 100.0
            
            # Apply brightness (additive)
            heatmap_array = heatmap_array + brightness
            
            # Apply contrast (multiplicative around midpoint)
            midpoint = 0.5
            heatmap_array = (heatmap_array - midpoint) * (1 + contrast) + midpoint
            
            # Apply gamma correction
            heatmap_array = np.sign(heatmap_array) * np.power(np.abs(heatmap_array), gamma)
            
            # Clip to valid range
            heatmap_array = np.clip(heatmap_array, 0, None)
            
            # Create plot
            ax = self.figure_heatmap.add_subplot(1, 1, 1)
            
            # Plot heatmap
            colormap = self.heatmap_colormap.currentText()
            interpolation = self.heatmap_interpolation.currentText()
            
            if not self.heatmap_antialias.isChecked():
                interpolation = 'none'
            
            extent = [common_wavenumbers.min(), common_wavenumbers.max(), 0, len(heatmap_array)]
            
            im = ax.imshow(
                heatmap_array, 
                aspect='auto',
                cmap=colormap,
                interpolation=interpolation,
                extent=extent,
                origin='lower'
            )
            
            # Add colorbar
            cbar = self.figure_heatmap.colorbar(im, ax=ax)
            cbar.set_label(f'{data_type} Intensity', rotation=270, labelpad=20)
            
            # Configure axes
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Spectrum Index')
            ax.set_title(f'Heatmap: {data_type}')
            
            # Grid
            if self.heatmap_grid.isChecked():
                ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            # Format y-axis to show file names if not too many spectra
            if len(data_source) <= 20:
                y_ticks = range(len(data_source))
                y_labels = [os.path.basename(result['file'])[:15] + '...' if len(os.path.basename(result['file'])) > 15 
                           else os.path.basename(result['file']) for result in data_source]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontsize=8)
            
            self.figure_heatmap.tight_layout()
            self.canvas_heatmap.draw()
            
        except Exception as e:
            ax = self.figure_heatmap.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
            self.canvas_heatmap.draw()

    def clear_manual_peaks(self):
        """Clear only manually selected peaks."""
        self.manual_peaks = np.array([], dtype=int)
        self.update_peak_count_display()
        self.update_current_plot()

    def combine_peaks(self):
        """Combine automatic and manual peaks into the main peaks list."""
        # Combine automatic and manual peaks
        all_peaks = []
        
        if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
            all_peaks.extend([int(p) for p in self.peaks.tolist()])
        
        if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
            all_peaks.extend([int(p) for p in self.manual_peaks.tolist()])
        
        # Remove duplicates and sort
        if len(all_peaks) > 0:
            unique_peaks = sorted(list(set(all_peaks)))
            self.peaks = np.array(unique_peaks, dtype=int)
        else:
            self.peaks = np.array([], dtype=int)
        
        # Clear manual peaks since they're now in the main list
        self.manual_peaks = np.array([], dtype=int)
        
        # Update display
        self.update_peak_count_display()
        self.update_current_plot()
        
        # Show confirmation
        QMessageBox.information(self, "Peaks Combined", 
                              f"Combined peaks into main list.\nTotal peaks: {len(self.peaks)}")

    def update_peak_count_display(self):
        """Update peak count display and peak list widget."""
        auto_count = len(self.peaks) if hasattr(self, 'peaks') and self.peaks is not None else 0
        manual_count = len(self.manual_peaks) if hasattr(self, 'manual_peaks') and self.manual_peaks is not None else 0
        total_count = auto_count + manual_count
        
        self.peak_count_label.setText(f"Peaks found: {auto_count}")
        if hasattr(self, 'peak_status_label'):
            self.peak_status_label.setText(f"Auto: {auto_count} | Manual: {manual_count} | Total: {total_count}")
            
        # Update peak list widget
        if hasattr(self, 'peak_list_widget'):
            self.peak_list_widget.clear()
            
            # Add automatic peaks
            if hasattr(self, 'peaks') and self.peaks is not None and len(self.wavenumbers) > 0:
                for i, peak_idx in enumerate(self.peaks):
                    if 0 <= peak_idx < len(self.wavenumbers):
                        wavenumber = self.wavenumbers[peak_idx]
                        intensity = self.intensities[peak_idx] if peak_idx < len(self.intensities) else 0
                        item_text = f"Auto Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I={intensity:.0f})"
                        self.peak_list_widget.addItem(item_text)
            
            # Add manual peaks
            if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.wavenumbers) > 0:
                for i, peak_idx in enumerate(self.manual_peaks):
                    if 0 <= peak_idx < len(self.wavenumbers):
                        wavenumber = self.wavenumbers[peak_idx]
                        intensity = self.intensities[peak_idx] if peak_idx < len(self.intensities) else 0
                        item_text = f"Manual Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I={intensity:.0f})"
                        self.peak_list_widget.addItem(item_text)
                
            print(f"DEBUG: Peak list widget now has {self.peak_list_widget.count()} items")
        else:
            print("DEBUG: No peak_list_widget found!")

    # Live view methods
    def update_live_view(self, result_data, wavenumbers, fit_params):
        """Update the live view with current fitting results."""
        try:
            # Clear previous plots
            self.ax_live_main.clear()
            self.ax_live_residual.clear()
            
            # Extract data with safe defaults for preview operations
            original_intensities = result_data.get('original_intensities', [])
            intensities = result_data.get('intensities', [])
            background = result_data.get('background', [])
            fitted_curve = result_data.get('fitted_curve', [])
            residuals = result_data.get('residuals', [])
            r_squared = result_data.get('r_squared', 0)
            
            # Main plot - always plot original and corrected data
            if len(original_intensities) > 0:
                self.ax_live_main.plot(wavenumbers, original_intensities, 'b-', 
                                      linewidth=1, alpha=0.7, label='Original')
            
            if len(background) > 0:
                self.ax_live_main.plot(wavenumbers, background, 'r--', 
                                      linewidth=1.5, alpha=0.8, label='Background')
            
            if len(intensities) > 0:
                self.ax_live_main.plot(wavenumbers, intensities, 'k-', 
                                      linewidth=1.5, label='Corrected')
            
            # Only plot fitted curve if it exists
            if len(fitted_curve) > 0:
                self.ax_live_main.plot(wavenumbers, fitted_curve, 'g-', 
                                      linewidth=2, label='Fitted')
                
                # Plot individual peaks only if we have fitted data
                if (fit_params is not None and 
                    hasattr(fit_params, '__len__') and 
                    len(fit_params) > 0):
                    self.plot_live_individual_peaks(wavenumbers, fit_params)
                
                # Mark peak positions only if we have fitted curve and peaks
                if hasattr(self, 'reference_peaks') and self.reference_peaks is not None:
                    peak_positions = []
                    peak_intensities = []
                    for peak_idx in self.reference_peaks:
                        if hasattr(self, 'wavenumbers') and len(self.wavenumbers) > 0:
                            closest_idx = np.argmin(np.abs(wavenumbers - self.wavenumbers[peak_idx]))
                            if 0 <= closest_idx < len(wavenumbers):
                                peak_positions.append(wavenumbers[closest_idx])
                                peak_intensities.append(fitted_curve[closest_idx])
                    
                    if peak_positions:
                        self.ax_live_main.plot(peak_positions, peak_intensities, 'ro', 
                                              markersize=8, label='Peak Positions')
            
            # Set title based on available data
            filename = result_data.get('file', 'Current Spectrum')
            if isinstance(filename, str):
                filename = os.path.basename(filename)
            
            if r_squared > 0:
                title = f'Live Fit: {filename} (R² = {r_squared:.4f})'
            else:
                title = f'Live View: {filename}'
            
            self.ax_live_main.set_ylabel('Intensity')
            self.ax_live_main.set_title(title)
            self.ax_live_main.legend()
            self.ax_live_main.grid(True, alpha=0.3)
            
            # Residuals plot - only if residuals exist
            if len(residuals) > 0:
                self.ax_live_residual.plot(wavenumbers, residuals, 'k-', linewidth=1)
                self.ax_live_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                self.ax_live_residual.set_xlabel('Wavenumber (cm⁻¹)')
                self.ax_live_residual.set_ylabel('Residuals')
                self.ax_live_residual.set_title('Fit Residuals')
                self.ax_live_residual.grid(True, alpha=0.3)
            else:
                # Show a placeholder if no residuals
                self.ax_live_residual.text(0.5, 0.5, 'No residuals available\n(Background/Smoothing preview)', 
                                         ha='center', va='center', transform=self.ax_live_residual.transAxes,
                                         fontsize=12, style='italic', color='gray')
                self.ax_live_residual.set_xlabel('Wavenumber (cm⁻¹)')
                self.ax_live_residual.set_ylabel('Residuals')
                self.ax_live_residual.set_title('Fit Residuals')
            
            # Update statistics
            if hasattr(self, 'overall_r2_label'):
                if r_squared > 0:
                    self.overall_r2_label.setText(f"Overall R²: {r_squared:.4f}")
                else:
                    self.overall_r2_label.setText("Overall R²: --")
            
            if hasattr(self, 'peak_count_live'):
                if hasattr(self, 'reference_peaks') and self.reference_peaks is not None:
                    self.peak_count_live.setText(f"Peaks: {len(self.reference_peaks)}")
                else:
                    self.peak_count_live.setText("Peaks: --")
            
            # Store current data for analysis
            self.current_live_data = result_data.copy()
            
            # Enable analyze button for Smart Residual Analyzer only if we have fitted data
            if hasattr(self, 'analyze_btn'):
                self.analyze_btn.setEnabled(len(fitted_curve) > 0 and r_squared > 0)
            
            self.canvas_live.draw()
            QApplication.processEvents()
            
        except Exception as e:
            print(f"Error updating live view: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_live_individual_peaks(self, wavenumbers, fit_params):
        """Plot individual fitted peaks in live view."""
        if len(self.reference_peaks) == 0 or len(fit_params) == 0:
            return
            
        n_peaks = len(self.reference_peaks)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(fit_params):
                amp, cen, wid = fit_params[start_idx:start_idx+3]
                
                if self.current_model == "Gaussian":
                    peak_curve = self.gaussian(wavenumbers, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    peak_curve = self.lorentzian(wavenumbers, amp, cen, wid)
                else:
                    peak_curve = self.gaussian(wavenumbers, amp, cen, wid)
                
                # Calculate individual R² for this peak using the live data
                individual_r2 = 0
                if hasattr(self, 'current_live_data') and 'residuals' in self.current_live_data:
                    try:
                        # Temporarily use live data for calculation
                        temp_wavenumbers = self.wavenumbers
                        temp_intensities = self.intensities
                        temp_residuals = self.residuals
                        
                        self.wavenumbers = wavenumbers
                        self.intensities = self.current_live_data['intensities']
                        self.residuals = self.current_live_data['residuals']
                        
                        individual_r2 = self.calculate_individual_peak_r2(i, fit_params)
                        
                        # Restore original data
                        self.wavenumbers = temp_wavenumbers
                        self.intensities = temp_intensities
                        self.residuals = temp_residuals
                        
                    except Exception as e:
                        print(f"Error calculating live peak R²: {e}")
                        individual_r2 = 0
                    
                # Plot individual peak curve with individual R² in legend
                label = f'Peak {i+1}: {cen:.0f}cm⁻¹ (R²={individual_r2:.3f})'
                self.ax_live_main.plot(wavenumbers, peak_curve, '--', 
                                     linewidth=1, alpha=0.6, label=label)
    
    def update_peak_statistics(self, fit_params, overall_r2):
        """Update individual peak statistics display."""
        # Clear existing statistics
        for i in reversed(range(self.peak_stats_layout.count())):
            child = self.peak_stats_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        if len(self.reference_peaks) == 0 or len(fit_params) == 0:
            return
            
        n_peaks = len(self.reference_peaks)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(fit_params):
                amp, cen, wid = fit_params[start_idx:start_idx+3]
                
                # Calculate individual peak R² (simplified estimation)
                # This is a rough estimate based on peak amplitude relative to overall signal
                peak_contribution = abs(amp) / (np.sum([abs(fit_params[j*3]) for j in range(n_peaks)]) + 1e-10)
                estimated_r2 = overall_r2 * peak_contribution
                
                peak_info = QLabel(f"Peak {i+1}: Pos={cen:.1f} cm⁻¹, Amp={amp:.1f}, Width={wid:.1f}, R²≈{estimated_r2:.3f}")
                peak_info.setStyleSheet("font-family: monospace; font-size: 10px; color: #333;")
                self.peak_stats_layout.addWidget(peak_info)
    
    def save_current_fit(self):
        """Save the current fit for later review."""
        if not hasattr(self, 'current_live_data'):
            QMessageBox.warning(self, "No Data", "No current fit data to save.")
            return
            
        filename = os.path.basename(self.current_live_data['file'])
        self.saved_fits[filename] = self.current_live_data.copy()
        
        QMessageBox.information(self, "Saved", f"Fit for {filename} has been saved for review.")
    
    def view_saved_fits(self):
        """Open a dialog to view saved fits."""
        if not self.saved_fits:
            QMessageBox.information(self, "No Saved Fits", "No fits have been saved yet.")
            return
            
        # Create a dialog for viewing saved fits
        dialog = QDialog(self)
        dialog.setWindowTitle("Saved Fits Viewer")
        dialog.setMinimumSize(1000, 700)
        dialog.resize(1200, 800)
        
        layout = QVBoxLayout(dialog)
        
        # File selection
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Select fit to view:"))
        
        file_combo = QComboBox()
        file_combo.addItems(list(self.saved_fits.keys()))
        file_layout.addWidget(file_combo)
        
        layout.addLayout(file_layout)
        
        # Plot area
        figure = Figure(figsize=(12, 8))
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, dialog)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # Function to update plot when selection changes
        def update_saved_fit_plot():
            selected_file = file_combo.currentText()
            if selected_file in self.saved_fits:
                data = self.saved_fits[selected_file]
                
                figure.clear()
                gs = figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)
                ax_main = figure.add_subplot(gs[0, 0])
                ax_residual = figure.add_subplot(gs[1, 0])
                
                # Plot the saved fit
                wavenumbers = data['wavenumbers']
                original_intensities = data['original_intensities']
                intensities = data['intensities']
                background = data['background']
                fitted_curve = data['fitted_curve']
                residuals = data['residuals']
                r_squared = data['r_squared']
                
                ax_main.plot(wavenumbers, original_intensities, 'b-', 
                           linewidth=1, alpha=0.7, label='Original')
                ax_main.plot(wavenumbers, background, 'r--', 
                           linewidth=1.5, alpha=0.8, label='Background')
                ax_main.plot(wavenumbers, intensities, 'k-', 
                           linewidth=1.5, label='Corrected')
                ax_main.plot(wavenumbers, fitted_curve, 'g-', 
                           linewidth=2, label='Fitted')
                
                ax_main.set_ylabel('Intensity')
                ax_main.set_title(f'Saved Fit: {selected_file} (R² = {r_squared:.4f})')
                ax_main.legend()
                ax_main.grid(True, alpha=0.3)
                
                ax_residual.plot(wavenumbers, residuals, 'k-', linewidth=1)
                ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
                ax_residual.set_ylabel('Residuals')
                ax_residual.set_title('Fit Residuals')
                ax_residual.grid(True, alpha=0.3)
                
                figure.tight_layout()
                canvas.draw()
        
        file_combo.currentTextChanged.connect(update_saved_fit_plot)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        # Initial plot
        if file_combo.count() > 0:
            update_saved_fit_plot()
        
        dialog.exec()

    def _create_waterfall_plot_in_figure(self, figure):
        """Helper method to create waterfall plot in a specific figure."""
        figure.clear()
        ax = figure.add_subplot(1, 1, 1)
        
        # First try to use loaded spectrum files directly
        if self.spectra_files:
            try:
                # Load spectra data directly from files
                loaded_spectra = []
                for file_path in self.spectra_files:
                    spectrum_data = self.load_spectrum_robust(file_path)
                    if spectrum_data is not None:
                        wavenumbers, intensities = spectrum_data
                        loaded_spectra.append({
                            'file': file_path,
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'original_intensities': intensities.copy()
                        })
                
                if loaded_spectra:
                    skip = self.waterfall_skip_spin.value()
                    y_offset = self.waterfall_offset_spin.value()
                    plotted_results = loaded_spectra[::skip]
                    
                    for i, result in enumerate(plotted_results):
                        if 'wavenumbers' in result and 'intensities' in result:
                            wavenumbers = result['wavenumbers']
                            intensities = result['intensities'] + i * y_offset
                            filename = os.path.basename(result['file'])
                            
                            ax.plot(wavenumbers, intensities, linewidth=1, 
                                   label=filename if i < 10 else "")
                    
                    ax.set_xlabel('Wavenumber (cm⁻¹)')
                    ax.set_ylabel('Intensity (offset)')
                    ax.set_title('Waterfall Plot of Spectra')
                    if len(plotted_results) <= 10:
                        ax.legend()
                    if self.show_grid_check.isChecked():
                        ax.grid(True, alpha=0.3)
                    
                    figure.tight_layout()
                    return
                    
            except Exception as e:
                pass  # Fall through to batch results
        
        # Fallback to batch results
        if not self.batch_results:
            ax.text(0.5, 0.5, 'No spectrum files or batch results available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            ax.text(0.5, 0.5, 'No successful fits available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        skip = self.waterfall_skip_spin.value()
        y_offset = self.waterfall_offset_spin.value()
        plotted_results = successful_results[::skip]
        
        for i, result in enumerate(plotted_results):
            if 'wavenumbers' in result and 'intensities' in result:
                wavenumbers = result['wavenumbers']
                intensities = result['intensities'] + i * y_offset
                filename = os.path.basename(result['file'])
                
                ax.plot(wavenumbers, intensities, linewidth=1, 
                       label=filename if i < 10 else "")
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Intensity (offset)')
        ax.set_title('Waterfall Plot of Spectra')
        if len(plotted_results) <= 10:
            ax.legend()
        if self.show_grid_check.isChecked():
            ax.grid(True, alpha=0.3)
        
        figure.tight_layout()

    def _create_heatmap_plot_in_figure(self, figure):
        """Helper method to create heatmap plot in a specific figure."""
        figure.clear()
        
        # First try to use loaded spectrum files directly
        if self.spectra_files:
            try:
                # Load spectra data directly from files
                loaded_spectra = []
                for file_path in self.spectra_files:
                    spectrum_data = self.load_spectrum_robust(file_path)
                    if spectrum_data is not None:
                        wavenumbers, intensities = spectrum_data
                        loaded_spectra.append({
                            'file': file_path,
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'original_intensities': intensities.copy()
                        })
                
                if loaded_spectra:
                    # Create simplified heatmap for popup
                    heatmap_data = []
                    for result in loaded_spectra:
                        if 'intensities' in result:
                            heatmap_data.append(result['intensities'])
                    
                    if heatmap_data:
                        ax = figure.add_subplot(1, 1, 1)
                        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
                        figure.colorbar(im, ax=ax)
                        ax.set_xlabel('Wavenumber Index')
                        ax.set_ylabel('Spectrum Index')
                        ax.set_title('Intensity Heatmap')
                        if self.show_grid_check.isChecked():
                            ax.grid(True, alpha=0.3)
                        
                        figure.tight_layout()
                        return
                        
            except Exception as e:
                pass  # Fall through to batch results
        
        # Fallback to batch results
        if not self.batch_results:
            ax = figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No spectrum files or batch results available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            ax = figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No successful fits available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Simplified heatmap creation for popup
        heatmap_data = []
        for result in successful_results:
            if 'intensities' in result:
                heatmap_data.append(result['intensities'])
        
        if heatmap_data:
            ax = figure.add_subplot(1, 1, 1)
            im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
            figure.colorbar(im, ax=ax)
            ax.set_xlabel('Wavenumber Index')
            ax.set_ylabel('Spectrum Index')
            ax.set_title('Intensity Heatmap')
            if self.show_grid_check.isChecked():
                ax.grid(True, alpha=0.3)
        
        figure.tight_layout()

    # Export and utility methods
    def export_results(self):
        """Export batch processing results to CSV."""
        if not self.batch_results:
            QMessageBox.warning(self, "No Results", "No batch processing results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", 
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.export_to_csv(file_path)
                QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def export_to_csv(self, file_path):
        """Export results to CSV file."""
        data = []
        
        for i, result in enumerate(self.batch_results):
            row = {
                'Spectrum': i,
                'File': os.path.basename(result['file']),
                'Fit_Success': not result.get('fit_failed', True)
            }
            
            if not result.get('fit_failed', True) and 'fit_params' in result:
                row['R_Squared'] = result.get('r_squared', 0)
                
                # Add peak parameters
                params = result['fit_params']
                n_peaks = len(params) // 3
                
                for peak_idx in range(n_peaks):
                    start_idx = peak_idx * 3
                    if start_idx + 2 < len(params):
                        amp, cen, wid = params[start_idx:start_idx+3]
                        row[f'Peak_{peak_idx+1}_Position'] = cen
                        row[f'Peak_{peak_idx+1}_Amplitude'] = amp
                        row[f'Peak_{peak_idx+1}_Width'] = wid
            else:
                row['Error'] = result.get('error', 'Unknown error')
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    def show_help(self):
        """Show help dialog."""
        help_text = """
        Batch Peak Fitting Help
        
        This tool allows you to process multiple Raman spectra with consistent peak fitting parameters.
        
        Workflow:
        1. File Selection: Add spectrum files for batch processing
        2. Peak Controls: Adjust background subtraction and peak detection parameters
        3. Manual Peak Selection: Use interactive mode to add/remove peaks by clicking
        4. Batch Processing: Set current spectrum as reference and apply to all files
        5. Live View: Monitor real-time fitting with live visualization and statistics (right panel)
        6. Results: View trends and export data
        
        Navigation:
        - Double-click files to view them
        - Use navigation buttons to move between spectra
        
        Peak Detection & Manual Selection:
        - Adjust height, distance, and prominence sliders for automatic detection
        - Enable Interactive Selection mode to manually add/remove peaks
        - Click near peaks to remove them, click elsewhere to add new peaks
        - Use "Clear Manual" to remove only manually added peaks
        - Use "Combine Auto + Manual" to merge both types into main peak list
        - Green squares show manual peaks, red circles show automatic peaks
        
        Background Subtraction:
        - Background is automatically calculated for each spectrum individually
        - ALS (Asymmetric Least Squares) parameters are applied consistently
        - Adjust λ (smoothness) and p (asymmetry) for optimal background removal
        - Background correction significantly improves R² values
        
        Peak Fitting:
        - Choose appropriate peak model (Gaussian/Lorentzian)
        - Fit peaks before setting as reference
        - Improved bounds handling with automatic fallback strategies
        
        Batch Processing:
        - Set a well-fitted spectrum as reference
        - 'Apply to All' processes all loaded files with individual background correction
        - View progress in the log with detailed R² values
        - Automatic fallback to unbounded fitting if bounds fail
        - Automatically switches to Live View tab (right panel) during processing
        
        Live View Features (Right Panel):
        - Real-time visualization during batch processing
        - Shows original spectrum, background, corrected data, and fitted curves
        - Individual peak components displayed as dashed lines
        - Live statistics: Overall R² and individual peak parameters
        - Save interesting fits for later review
        - View saved fits with full interactive plots
        
        Visualization Tabs (Right Panel):
        - Current Spectrum: View individual spectra with fits (3:1 ratio main:residuals)
        - Live View: Real-time fitting visualization with statistics
        - Trends: See how peak parameters change across spectra
        - Waterfall: Stacked view of multiple spectra (works with imported files)
        - Heatmap: 2D intensity visualization (works with imported files)
        
        Plot Interaction:
        - Double-click any plot to open it in a popup window
        - Use Results tab checkboxes to control grid and boundary display
        - Export all plot data as comprehensive CSV files
        
        Peak Status Display:
        - Shows count of automatic peaks, manual peaks, and total
        - Interactive mode indicator shows current selection state
        - Live R² values for overall fit and individual peak contributions
        
        Tips for Better Results:
        - Ensure proper background subtraction parameters for your data
        - Use manual peak selection to refine automatic detection
        - Monitor live view during batch processing to catch issues early
        - Save fits with good R² values for comparison and validation
        - Waterfall and heatmap plots work immediately after loading files
        - Live View tab (right panel) provides real-time feedback during processing
        """
        
        QMessageBox.information(self, "Help", help_text)

    # Manual peak handling methods
    def toggle_interactive_mode(self):
        """Toggle interactive peak selection mode."""
        self.interactive_mode = self.interactive_btn.isChecked()
        
        if self.interactive_mode:
            # Enable interactive mode
            self.interactive_btn.setText("🖱️ Disable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: ON - Click on spectrum to select peaks")
            self.interactive_status_label.setStyleSheet("color: #4CAF50; font-size: 10px; font-weight: bold;")
            
            # Connect mouse click event to current spectrum canvas
            if hasattr(self, 'canvas_current'):
                self.click_connection = self.canvas_current.mpl_connect('button_press_event', self.on_canvas_click)
            
        else:
            # Disable interactive mode
            self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: OFF")
            self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Disconnect mouse click event
            if self.click_connection is not None and hasattr(self, 'canvas_current'):
                self.canvas_current.mpl_disconnect(self.click_connection)
                self.click_connection = None
        
        self.update_current_plot()
        self.update_peak_count_display()

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
            click_wavenumber = float(click_x)
            
            # Find the closest wavenumber index
            closest_idx = int(np.argmin(np.abs(self.wavenumbers - click_wavenumber)))
            
            # Validate the index
            if closest_idx < 0 or closest_idx >= len(self.wavenumbers):
                return
            
            # Check if we're clicking near an existing peak to remove it
            removal_threshold = 20  # wavenumber units
            
            # Check automatic peaks
            removed_auto = False
            if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks) > 0:
                for i, peak_idx in enumerate(self.peaks):
                    peak_idx = int(peak_idx)  # Ensure it's a Python int
                    if 0 <= peak_idx < len(self.wavenumbers):
                        peak_wavenumber = float(self.wavenumbers[peak_idx])
                        if abs(peak_wavenumber - click_wavenumber) < removal_threshold:
                            # Remove this automatic peak
                            peak_list = list(self.peaks)
                            if peak_idx in peak_list:
                                peak_list.remove(peak_idx)
                                self.peaks = np.array(peak_list, dtype=int)
                                removed_auto = True
                                break
            
            # Check manual peaks
            removed_manual = False
            if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.manual_peaks) > 0:
                for i, peak_idx in enumerate(self.manual_peaks):
                    peak_idx = int(peak_idx)  # Ensure it's a Python int
                    if 0 <= peak_idx < len(self.wavenumbers):
                        peak_wavenumber = float(self.wavenumbers[peak_idx])
                        if abs(peak_wavenumber - click_wavenumber) < removal_threshold:
                            # Remove this manual peak
                            peak_list = list(self.manual_peaks)
                            if peak_idx in peak_list:
                                peak_list.remove(peak_idx)
                                self.manual_peaks = np.array(peak_list, dtype=int)
                                removed_manual = True
                                break
            
            # If we didn't remove any peak, add a new manual peak
            if not removed_auto and not removed_manual:
                # Add new manual peak
                if not hasattr(self, 'manual_peaks') or self.manual_peaks is None:
                    self.manual_peaks = np.array([closest_idx], dtype=int)
                else:
                    # Check if this peak is already in manual peaks
                    should_add_peak = True
                    if len(self.manual_peaks) > 0:
                        manual_peaks_list = [int(p) for p in self.manual_peaks]
                        should_add_peak = closest_idx not in manual_peaks_list
                    
                    if should_add_peak:
                        self.manual_peaks = np.append(self.manual_peaks, closest_idx).astype(int)
            
            # Update display
            self.update_peak_count_display()
            self.update_current_plot()
            
        except Exception as e:
            print(f"Error in interactive peak selection: {e}")
            import traceback
            traceback.print_exc()

    def update_all_plots(self):
        """Update all plots when display options change."""
        # Update current spectrum plot
        self.update_current_plot()
        
        # Update trends plot
        self.update_trends_plot()
        
        # Update waterfall plot
        self.update_waterfall_plot()
        
        # Update heatmap plot
        self.update_heatmap_plot()
    
    def setup_plot_click_handlers(self):
        """Setup click event handlers for all plots to enable popup functionality."""
        # Setup click handlers after creating the figures
        if hasattr(self, 'canvas_current'):
            self.canvas_current.mpl_connect('button_press_event', 
                                          lambda event: self.on_plot_click(event, 'current'))
        
        if hasattr(self, 'canvas_trends'):
            self.canvas_trends.mpl_connect('button_press_event', 
                                         lambda event: self.on_plot_click(event, 'trends'))
        
        if hasattr(self, 'canvas_waterfall'):
            self.canvas_waterfall.mpl_connect('button_press_event', 
                                            lambda event: self.on_plot_click(event, 'waterfall'))
        
        if hasattr(self, 'canvas_heatmap'):
            self.canvas_heatmap.mpl_connect('button_press_event', 
                                          lambda event: self.on_plot_click(event, 'heatmap'))
    
    def on_plot_click(self, event, plot_type):
        """Handle click events on plots to open popup windows."""
        if event.dblclick:  # Only respond to double-clicks
            self.open_plot_popup(plot_type)
    
    def open_plot_popup(self, plot_type):
        """Open a popup window with the specified plot."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout
        
        popup = QDialog(self)
        popup.setWindowTitle(f"{plot_type.title()} Plot - Popup View")
        popup.setMinimumSize(800, 600)
        popup.resize(1000, 750)
        
        layout = QVBoxLayout(popup)
        
        # Create new figure and canvas for the popup
        if plot_type == 'current':
            figure = Figure(figsize=(12, 10))
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, popup)
            
            # Recreate current spectrum plot
            ax_main = figure.add_subplot(2, 1, 1)
            ax_residual = figure.add_subplot(2, 1, 2)
            
            if len(self.wavenumbers) > 0:
                # Main spectrum plot
                ax_main.plot(self.wavenumbers, self.intensities, 'b-', 
                           linewidth=1.5, label='Spectrum')
                
                # Plot background if available
                if self.background is not None:
                    ax_main.plot(self.wavenumbers, self.background, 'r--', 
                               linewidth=1, alpha=0.7, label='Background')
                
                # Plot peaks
                if len(self.peaks) > 0:
                    peak_positions = [self.wavenumbers[int(p)] for p in self.peaks if 0 <= int(p) < len(self.wavenumbers)]
                    peak_intensities = [self.intensities[int(p)] for p in self.peaks if 0 <= int(p) < len(self.intensities)]
                    ax_main.plot(peak_positions, peak_intensities, 'ro', 
                               markersize=8, label='Auto Peaks')
                
                # Plot fitted curve if available
                if (self.fit_result and 
                    self.fit_params is not None and 
                    hasattr(self.fit_params, '__len__') and 
                    len(self.fit_params) > 0):
                    fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
                    ax_main.plot(self.wavenumbers, fitted_curve, 'g-', 
                               linewidth=2, label='Fitted')
                
                ax_main.set_xlabel('Wavenumber (cm⁻¹)')
                ax_main.set_ylabel('Intensity')
                ax_main.set_title('Current Spectrum')
                ax_main.legend()
                if self.show_grid_check.isChecked():
                    ax_main.grid(True, alpha=0.3)
                
                # Residuals plot
                if self.residuals is not None:
                    ax_residual.plot(self.wavenumbers, self.residuals, 'k-', linewidth=1)
                    ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
                    ax_residual.set_ylabel('Residuals')
                    ax_residual.set_title('Fit Residuals')
                    if self.show_grid_check.isChecked():
                        ax_residual.grid(True, alpha=0.3)
            
            figure.tight_layout(pad=3.0)
            
        elif plot_type == 'trends':
            figure = Figure(figsize=(14, 12))
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, popup)
            
            # Recreate trends plot
            self._create_trends_plot_in_figure(figure)
            
        elif plot_type == 'waterfall':
            figure = Figure(figsize=(14, 10))
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, popup)
            
            # Recreate waterfall plot
            self._create_waterfall_plot_in_figure(figure)
            
        elif plot_type == 'heatmap':
            figure = Figure(figsize=(12, 10))
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, popup)
            
            # Recreate heatmap plot
            self._create_heatmap_plot_in_figure(figure)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(popup.close)
        layout.addWidget(close_btn)
        
        popup.exec()
    
    def _create_trends_plot_in_figure(self, figure):
        """Helper method to create trends plot in a specific figure."""
        figure.clear()
        
        if not self.batch_results:
            ax = figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No batch results available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            ax = figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No successful fits available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create subplots
        ax1 = figure.add_subplot(2, 2, 1)  # Position
        ax2 = figure.add_subplot(2, 2, 2)  # Amplitude
        ax3 = figure.add_subplot(2, 2, 3)  # Width
        ax4 = figure.add_subplot(2, 2, 4)  # R-squared
        
        # Extract data
        file_indices = list(range(len(successful_results)))
        n_peaks = len(self.reference_peaks) if self.reference_peaks is not None else 0
        
        for peak_idx in range(n_peaks):
            if peak_idx >= len(self.peak_visibility_vars) or not self.peak_visibility_vars[peak_idx].isChecked():
                continue
                
            positions = []
            amplitudes = []
            widths = []
            
            for result in successful_results:
                if 'fit_params' in result and result['fit_params'] is not None:
                    params = result['fit_params']
                    start_idx = peak_idx * 3
                    if start_idx + 2 < len(params):
                        amp, cen, wid = params[start_idx:start_idx+3]
                        positions.append(cen)
                        amplitudes.append(amp)
                        widths.append(wid)
                    else:
                        positions.append(np.nan)
                        amplitudes.append(np.nan)
                        widths.append(np.nan)
                else:
                    positions.append(np.nan)
                    amplitudes.append(np.nan)
                    widths.append(np.nan)
            
            # Plot trends
            label = f'Peak {peak_idx + 1}'
            ax1.plot(file_indices, positions, 'o-', label=label, alpha=0.7)
            ax2.plot(file_indices, amplitudes, 'o-', label=label, alpha=0.7)
            ax3.plot(file_indices, widths, 'o-', label=label, alpha=0.7)
        
        # Plot R-squared
        r_squared_values = [result.get('r_squared', 0) for result in successful_results]
        ax4.plot(file_indices, r_squared_values, 'ko-', label='R²')
        
        # Format plots
        ax1.set_title('Peak Positions')
        ax1.set_ylabel('Wavenumber (cm⁻¹)')
        ax1.legend()
        if self.show_grid_check.isChecked():
            ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Peak Amplitudes')
        ax2.set_ylabel('Intensity')
        ax2.legend()
        if self.show_grid_check.isChecked():
            ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Peak Widths')
        ax3.set_xlabel('Spectrum Index')
        ax3.set_ylabel('Width')
        ax3.legend()
        if self.show_grid_check.isChecked():
            ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Fit Quality (R²)')
        ax4.set_xlabel('Spectrum Index')
        ax4.set_ylabel('R²')
        # Auto-range R² plot to better show variation
        if r_squared_values and len(r_squared_values) > 0:
            r2_min = min(r_squared_values)
            r2_max = max(r_squared_values)
            if r2_max > r2_min:
                # Add 5% padding to top and bottom
                r2_range = r2_max - r2_min
                ax4.set_ylim(max(0, r2_min - 0.05 * r2_range), min(1, r2_max + 0.05 * r2_range))
            else:
                # If all values are the same, show a small range around that value
                ax4.set_ylim(max(0, r2_min - 0.1), min(1, r2_max + 0.1))
        else:
            ax4.set_ylim(0, 1)  # Fallback to full range if no data
        if self.show_grid_check.isChecked():
            ax4.grid(True, alpha=0.3)
        
        figure.tight_layout()

    def update_waterfall_line_width_label(self):
        """Update waterfall line width label."""
        value = self.waterfall_line_width.value() / 10.0
        self.waterfall_line_width_label.setText(f"{value:.1f}")
    
    def update_waterfall_alpha_label(self):
        """Update waterfall alpha label."""
        value = self.waterfall_alpha.value() / 100.0
        self.waterfall_alpha_label.setText(f"{value:.1f}")
    
    def update_waterfall_contrast_label(self):
        """Update waterfall contrast label."""
        value = self.waterfall_contrast.value()
        self.waterfall_contrast_label.setText(str(value))
    
    def update_waterfall_gamma_label(self):
        """Update waterfall gamma label."""
        value = self.waterfall_gamma.value() / 100.0
        self.waterfall_gamma_label.setText(f"{value:.1f}")
    
    def update_waterfall_brightness_label(self):
        """Update waterfall brightness label."""
        value = self.waterfall_brightness.value()
        self.waterfall_brightness_label.setText(str(value))
    
    def on_waterfall_auto_range_changed(self):
        """Handle waterfall auto range checkbox change."""
        auto_enabled = self.waterfall_auto_range.isChecked()
        self.waterfall_range_controls_widget.setEnabled(not auto_enabled)
        if auto_enabled:
            self.update_waterfall_plot()

    def export_all_plot_data(self):
        """Export all plot data points to a comprehensive CSV file."""
        if not self.batch_results:
            QMessageBox.warning(self, "No Data", "No batch processing results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export All Plot Data", "", 
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                self._export_comprehensive_csv(file_path)
                QMessageBox.information(self, "Export Complete", 
                                      f"All plot data exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", 
                                   f"Failed to export plot data: {str(e)}")
    
    def _export_comprehensive_csv(self, file_path):
        """Export comprehensive CSV with all plot data."""
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            # Create empty file with headers
            with open(file_path, 'w') as f:
                f.write("No successful results to export\n")
            return
        
        all_data = []
        
        # Export data based on available results
        for spec_idx, result in enumerate(successful_results):
            filename = os.path.basename(result['file'])
            
            if 'wavenumbers' in result:
                wavenumbers = result['wavenumbers']
                original_intensities = result.get('original_intensities', [])
                bg_corrected_intensities = result.get('intensities', [])
                residuals = result.get('residuals', [])
                
                for pt_idx, wn in enumerate(wavenumbers):
                    row = {
                        'DataType': 'Raw_Spectral_Data',
                        'SpectrumIndex': spec_idx,
                        'SpectrumFile': filename,
                        'PointIndex': pt_idx,
                        'Wavenumber': wn,
                        'OriginalIntensity': original_intensities[pt_idx] if pt_idx < len(original_intensities) else '',
                        'BackgroundCorrectedIntensity': bg_corrected_intensities[pt_idx] if pt_idx < len(bg_corrected_intensities) else '',
                        'ResidualValue': residuals[pt_idx] if pt_idx < len(residuals) else '',
                        'R_Squared': result.get('r_squared', '')
                    }
                    all_data.append(row)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        df.to_csv(file_path, index=False)

    # Smart Residual Analyzer
    def analyze_current_fit(self):
        """Analyze the current fit and suggest improvements."""
        if not hasattr(self, 'current_live_data') or not self.current_live_data:
            self.analysis_text.clear()
            self.analysis_text.append("❌ No current fit data to analyze.")
            self.analysis_text.append("Process some spectra first, then return to analyze.")
            return
        
        self.analysis_text.clear()
        self.analysis_text.append("🔍 SMART RESIDUAL ANALYSIS")
        self.analysis_text.append("=" * 40)
        
        try:
            # Get current fit data
            residuals = self.current_live_data['residuals']
            intensities = self.current_live_data['intensities']
            r_squared = self.current_live_data['r_squared']
            fit_params = self.current_live_data['fit_params']
            wavenumbers = self.current_live_data['wavenumbers']
            
            # Calculate analysis metrics
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            max_residual = np.max(np.abs(residuals))
            signal_range = np.max(intensities) - np.min(intensities)
            noise_level = residual_std / signal_range * 100 if signal_range > 0 else 0
            
            # Quality assessment
            self.analysis_text.append(f"📊 FIT QUALITY METRICS:")
            self.analysis_text.append(f"   R² Value: {r_squared:.4f}")
            self.analysis_text.append(f"   Residual Std: {residual_std:.2f}")
            self.analysis_text.append(f"   Max Residual: {max_residual:.2f}")
            self.analysis_text.append(f"   Noise Level: {noise_level:.1f}%")
            self.analysis_text.append("")
            
            # Smart suggestions
            suggestions = []
            
            # R² analysis
            if r_squared < 0.85:
                suggestions.append("⚠️  Low R² (<0.85): Consider more peaks or different model")
            elif r_squared < 0.95:
                suggestions.append("⚡ Moderate R² (0.85-0.95): Good fit, minor improvements possible")
            else:
                suggestions.append("✅ Excellent R² (>0.95): High quality fit!")
            
            # Residual pattern analysis
            residual_trend = np.polyfit(wavenumbers, residuals, 1)[0]
            if abs(residual_trend) > residual_std * 0.1:
                suggestions.append("📈 Systematic residual trend detected: Check background subtraction")
            
            # Check for outlier residuals
            outlier_threshold = 3 * residual_std
            outliers = np.sum(np.abs(residuals) > outlier_threshold)
            if outliers > len(residuals) * 0.05:  # More than 5% outliers
                suggestions.append(f"🎯 {outliers} outlier points detected: Consider additional peaks")
            
            # Noise analysis
            if noise_level > 10:
                suggestions.append("🔊 High noise level (>10%): Consider smoothing or longer integration")
            elif noise_level < 1:
                suggestions.append("🔇 Very low noise (<1%): Excellent data quality!")
            
            # Peak width analysis
            if len(fit_params) >= 3:
                widths = [fit_params[i*3 + 2] for i in range(len(fit_params)//3)]
                avg_width = np.mean(widths)
                if avg_width > 50:
                    suggestions.append("📏 Wide peaks detected: Check spectral resolution or peak model")
                elif avg_width < 5:
                    suggestions.append("📐 Very narrow peaks: Excellent resolution or check for artifacts")
            
            # Background assessment
            if hasattr(self, 'lambda_slider') and hasattr(self, 'p_slider'):
                lambda_val = 10 ** self.lambda_slider.value()
                p_val = self.p_slider.value() / 1000.0
                if lambda_val < 1e4:
                    suggestions.append("🔧 Low λ value: Background may be under-smoothed")
                elif lambda_val > 1e6:
                    suggestions.append("🔧 High λ value: Background may be over-smoothed")
                
                if p_val < 0.005:
                    suggestions.append("⚖️  Low p value: Background may fit peaks too closely")
                elif p_val > 0.02:
                    suggestions.append("⚖️  High p value: Background may ignore baseline features")
            
            # Display suggestions
            if suggestions:
                self.analysis_text.append("💡 IMPROVEMENT SUGGESTIONS:")
                for i, suggestion in enumerate(suggestions, 1):
                    self.analysis_text.append(f"   {i}. {suggestion}")
            else:
                self.analysis_text.append("🎉 No specific improvements suggested - excellent fit!")
            
            self.analysis_text.append("")
            self.analysis_text.append("📝 TIP: Use 'Export All Batch Data' for detailed analysis")
            
        except Exception as e:
            self.analysis_text.append(f"❌ Analysis error: {str(e)}")
            self.analysis_text.append("Check that fit data is complete and valid.")
    
    def export_comprehensive_batch_data(self):
        """Export comprehensive batch data including all metrics."""
        if not self.batch_results:
            QMessageBox.warning(self, "No Data", "No batch processing results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comprehensive Batch Data", "", 
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                self._export_detailed_batch_csv(file_path)
                QMessageBox.information(self, "Export Complete", 
                                      f"Comprehensive batch data exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", 
                                   f"Failed to export batch data: {str(e)}")
    
    def _export_detailed_batch_csv(self, file_path):
        """Export detailed batch CSV with analysis metrics."""
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            with open(file_path, 'w') as f:
                f.write("No successful batch results to export\n")
            return
        
        all_data = []
        
        for spec_idx, result in enumerate(successful_results):
            filename = os.path.basename(result['file'])
            r_squared = result.get('r_squared', 0)
            
            # Calculate analysis metrics
            if 'residuals' in result and 'intensities' in result:
                residuals = result['residuals']
                intensities = result['intensities']
                residual_std = np.std(residuals)
                max_residual = np.max(np.abs(residuals))
                signal_range = np.max(intensities) - np.min(intensities)
                noise_level = residual_std / signal_range * 100 if signal_range > 0 else 0
            else:
                residual_std = max_residual = noise_level = 0
            
            # Peak parameters
            if 'fit_params' in result and result['fit_params'] is not None:
                params = result['fit_params']
                n_peaks = len(params) // 3
                
                for peak_idx in range(n_peaks):
                    start_idx = peak_idx * 3
                    if start_idx + 2 < len(params):
                        amp, cen, wid = params[start_idx:start_idx+3]
                        
                        row = {
                            'SpectrumIndex': spec_idx,
                            'Filename': filename,
                            'PeakNumber': peak_idx + 1,
                            'PeakPosition_cm': cen,
                            'PeakAmplitude': amp,
                            'PeakWidth': wid,
                            'OverallR2': r_squared,
                            'ResidualStd': residual_std,
                            'MaxResidual': max_residual,
                            'NoiseLevel_percent': noise_level,
                            'FitQuality': 'Excellent' if r_squared > 0.95 else 'Good' if r_squared > 0.85 else 'Poor'
                        }
                        all_data.append(row)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_data)
        df.to_csv(file_path, index=False)

    def update_live_view_with_current_spectrum(self):
        """Update the live view with the current spectrum data."""
        if not hasattr(self, 'wavenumbers') or len(self.wavenumbers) == 0:
            return
            
        # Calculate R² if we have fit results
        r_squared = 0
        fitted_curve = []
        
        if (self.fit_result and 
            self.fit_params is not None and 
            hasattr(self.fit_params, '__len__') and 
            len(self.fit_params) > 0 and 
            self.residuals is not None):
            ss_res = np.sum(self.residuals ** 2)
            ss_tot = np.sum((self.intensities - np.mean(self.intensities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate fitted curve
            try:
                fitted_curve = self.calculate_fitted_curve(self.wavenumbers, self.fit_params, self.current_model)
            except Exception:
                fitted_curve = []
        
        # Get current file name
        if hasattr(self, 'spectra_files') and self.spectra_files and self.current_spectrum_index < len(self.spectra_files):
            filename = os.path.basename(self.spectra_files[self.current_spectrum_index])
        else:
            filename = "Current Spectrum"
        
        # Create data structure for live view - fix array boolean evaluation
        result_data = {
            'file': filename,
            'wavenumbers': self.wavenumbers,
            'intensities': self.intensities,
            'original_intensities': self.original_intensities,
            'background': self.background.copy() if (hasattr(self, 'background') and self.background is not None) else [],
            'fitted_curve': fitted_curve,  # Include fitted curve when available
            'peaks': self.peaks if len(self.peaks) > 0 else [],
            'fit_params': self.fit_params if (self.fit_params is not None and hasattr(self.fit_params, '__len__') and len(self.fit_params) > 0) else [],
            'residuals': self.residuals.copy() if (hasattr(self, 'residuals') and self.residuals is not None) else [],
            'r_squared': r_squared,
            'fit_failed': False
        }
        
        # Store as current live data for analyzer
        self.current_live_data = result_data.copy()
        
        # Update the live view display
        try:
            self.update_live_view(result_data, self.wavenumbers, self.fit_params if (self.fit_params is not None and hasattr(self.fit_params, '__len__') and len(self.fit_params) > 0) else [])
        except Exception as e:
            print(f"Error updating live view: {e}")
            # Enable analyze button even if live view update fails
            if hasattr(self, 'analyze_btn'):
                self.analyze_btn.setEnabled(True)

    def delete_selected_peaks(self):
        """Delete selected peaks from the list."""
        selected_items = self.peak_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select peaks to delete.")
            return
        
        # Process selections and determine which peaks to delete
        auto_peaks_to_delete = []
        manual_peaks_to_delete = []
        
        for item in selected_items:
            item_text = item.text()
            
            if item_text.startswith("Auto Peak"):
                # Extract peak number from "Auto Peak X: ..."
                try:
                    peak_num = int(item_text.split()[2].rstrip(':')) - 1  # Convert to 0-based index
                    if 0 <= peak_num < len(self.peaks):
                        auto_peaks_to_delete.append(peak_num)
                except (ValueError, IndexError):
                    continue
                    
            elif item_text.startswith("Manual Peak"):
                # Extract peak number from "Manual Peak X: ..."
                try:
                    peak_num = int(item_text.split()[2].rstrip(':')) - 1  # Convert to 0-based index
                    if 0 <= peak_num < len(self.manual_peaks):
                        manual_peaks_to_delete.append(peak_num)
                except (ValueError, IndexError):
                    continue
        
        # Delete peaks (in reverse order to preserve indices)
        if auto_peaks_to_delete:
            auto_peaks_to_delete.sort(reverse=True)
            for peak_idx in auto_peaks_to_delete:
                self.peaks = np.delete(self.peaks, peak_idx)
        
        if manual_peaks_to_delete:
            manual_peaks_to_delete.sort(reverse=True)
            for peak_idx in manual_peaks_to_delete:
                self.manual_peaks = np.delete(self.manual_peaks, peak_idx)
        
        # Update displays
        self.update_peak_count_display()
        self.update_current_plot()
        
        # Show confirmation
        deleted_count = len(auto_peaks_to_delete) + len(manual_peaks_to_delete)
        if deleted_count > 0:
            QMessageBox.information(self, "Peaks Deleted", 
                                  f"Deleted {deleted_count} peak(s):\n"
                                  f"Auto: {len(auto_peaks_to_delete)}, Manual: {len(manual_peaks_to_delete)}")
        else:
            QMessageBox.warning(self, "No Deletion", "No valid peaks were selected for deletion.")

    def calculate_individual_peak_r2(self, peak_index, fit_params):
        """Calculate R² for an individual peak."""
        if (self.residuals is None or 
            fit_params is None or 
            len(fit_params) < (peak_index + 1) * 3):
            return 0
            
        try:
            # Get parameters for this specific peak
            start_idx = peak_index * 3
            amp, cen, wid = fit_params[start_idx:start_idx+3]
            
            # Create individual peak curve
            if self.current_model == "Gaussian":
                individual_peak = self.gaussian(self.wavenumbers, amp, cen, wid)
            elif self.current_model == "Lorentzian":
                individual_peak = self.lorentzian(self.wavenumbers, amp, cen, wid)
            else:
                individual_peak = self.gaussian(self.wavenumbers, amp, cen, wid)
            
            # Find data points near this peak (within 3*width)
            peak_center_idx = np.argmin(np.abs(self.wavenumbers - cen))
            width_in_points = int(wid * 3 / (self.wavenumbers[1] - self.wavenumbers[0])) if len(self.wavenumbers) > 1 else 20
            
            start_idx = max(0, peak_center_idx - width_in_points)
            end_idx = min(len(self.wavenumbers), peak_center_idx + width_in_points)
            
            # Calculate R² for this peak region
            y_true = self.intensities[start_idx:end_idx]
            y_pred = individual_peak[start_idx:end_idx]
            
            if len(y_true) > 1:
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                return max(0, min(1, r_squared))  # Clamp between 0 and 1
            else:
                return 0
                
        except Exception as e:
            print(f"Error calculating individual peak R²: {e}")
            return 0

    def calculate_fitted_curve(self, wavenumbers, params, model_type="Gaussian"):
        """Calculate fitted curve without relying on internal state."""
        n_peaks = len(params) // 3
        
        if n_peaks == 0:
            return np.zeros_like(wavenumbers)
            
        model = np.zeros_like(wavenumbers)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(params):
                amp, cen, wid = params[start_idx:start_idx+3]
                wid = max(abs(wid), 1.0)  # Ensure positive width
                
                if model_type == "Gaussian":
                    component = self.gaussian(wavenumbers, amp, cen, wid)
                elif model_type == "Lorentzian":
                    component = self.lorentzian(wavenumbers, amp, cen, wid)
                else:
                    component = self.gaussian(wavenumbers, amp, cen, wid)  # Default
                    
                model += component
                
        return model

    def multi_peak_model(self, x, *params):
        """Multi-peak model function."""
        # Calculate number of peaks from parameters (3 params per peak)
        n_peaks = len(params) // 3
        
        if n_peaks == 0:
            return np.zeros_like(x)
            
        model = np.zeros_like(x)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(params):
                amp, cen, wid = params[start_idx:start_idx+3]
                wid = max(abs(wid), 1.0)  # Ensure positive width
                
                if self.current_model == "Gaussian":
                    component = self.gaussian(x, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    component = self.lorentzian(x, amp, cen, wid)
                else:
                    component = self.gaussian(x, amp, cen, wid)  # Default
                    
                model += component
                
        return model

# Launch function for integration with main app
def launch_batch_peak_fitting(parent, wavenumbers=None, intensities=None):
    """Launch the batch peak fitting window."""
    dialog = BatchPeakFittingQt6(parent, wavenumbers, intensities)
    dialog.exec() 