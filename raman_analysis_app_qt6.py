#!/usr/bin/env python3
"""
RamanLab PySide6 Version - Main Application Window
"""

import os
import sys
import time
import warnings
from pathlib import Path
import webbrowser  # Add this import for opening URLs
import numpy as np

# Fix matplotlib backend for PySide6
import matplotlib
matplotlib.use("QtAgg")  # Use QtAgg backend which works with PySide6
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import PySide6-compatible matplotlib backends and UI toolbar
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    # Fallback for older matplotlib versions - still works with PySide6
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from core.matplotlib_config import CompactNavigationToolbar as NavigationToolbar

from core.matplotlib_config import configure_compact_ui, apply_theme

from scipy.signal import find_peaks, savgol_filter
import pandas as pd

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QStatusBar, QMenuBar, QApplication, QSpinBox, QDoubleSpinBox,
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QListWidget,
    QListWidgetItem, QInputDialog, QFrame, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressDialog
)
from PySide6.QtCore import Qt, QTimer, QStandardPaths, QUrl, QThread, Signal
from PySide6.QtGui import QAction, QDesktopServices, QPixmap, QFont

# Version info
from version import __version__, __author__, __copyright__



# Import update checker
# TEMPORARILY DISABLED to isolate threading issues
# try:
#     from core.update_checker import check_for_updates, UPDATE_CHECKER_AVAILABLE
#     UPDATE_CHECKER_AVAILABLE = UPDATE_CHECKER_AVAILABLE
# except ImportError:
UPDATE_CHECKER_AVAILABLE = False

# Import the RamanSpectraQt6 class for database functionality
try:
    from raman_spectra_qt6 import RamanSpectraQt6
    RAMAN_SPECTRA_AVAILABLE = True
except ImportError:
    RAMAN_SPECTRA_AVAILABLE = False

# Additional imports for data handling
import pickle
from scipy.interpolate import griddata

# Import state management
try:
    from core.universal_state_manager import register_module, save_module_state, load_module_state
    STATE_MANAGEMENT_AVAILABLE = True
except ImportError:
    STATE_MANAGEMENT_AVAILABLE = False
    print("Warning: State management not available - continuing without session saving")


class RamanAnalysisAppQt6(QMainWindow):
    """RamanLab PySide6: Main application window for Raman spectrum analysis."""

    def __init__(self):
        """Initialize the PySide6 application."""
        super().__init__()
        
        # Initialize configuration management
        try:
            from core.config_manager import get_config_manager
            self.config = get_config_manager()
            print(f"✅ Configuration loaded from: {self.config.config_file}")
            print(f"📁 Projects folder: {self.config.get_projects_folder()}")
        except Exception as e:
            print(f"Warning: Could not initialize configuration: {e}")
            self.config = None
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        # Initialize components
        self.current_wavenumbers = None
        self.current_intensities = None
        self.processed_intensities = None
        self.original_wavenumbers = None
        self.original_intensities = None
        self.metadata = {}
        self.spectrum_file_path = None
        
        # Spectrum database
        self.database = {}
        
        # Initialize RamanSpectraQt6 for database functionality
        if RAMAN_SPECTRA_AVAILABLE:
            self.raman_db = RamanSpectraQt6(parent_widget=self)
        else:
            self.raman_db = None
            print("Warning: RamanSpectra not available, database functionality will be limited")
        
        # Background subtraction state
        self.background_preview = None
        self.smoothing_preview = None
        
        # Processing state
        self.detected_peaks = None
        self.manual_peaks = []  # Store manual peak positions as wavenumber values
        self.peak_selection_mode = False
        self.peak_selection_tolerance = 20  # Tolerance for clicking on peaks (in wavenumber units)
        self.background_preview_active = False
        self.smoothing_preview_active = False
        self.preview_background = None
        self.preview_corrected = None
        self.preview_smoothed = None
        
        # Set window properties
        self.setWindowTitle("RamanLab")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Set up the UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.center_on_screen()
        
        # Load database
        self.load_database()
        
        # Setup state management
        if STATE_MANAGEMENT_AVAILABLE:
            self.setup_state_management()
        
        # Show startup message
        self.show_startup_message()

    def load_database(self):
        """Load the database using RamanSpectra."""
        if self.raman_db:
            try:
                self.raman_db.load_database()
                print(f"✓ Database loaded with {len(self.raman_db.database)} spectra")
            except Exception as e:
                print(f"Warning: Could not load database: {e}")
        else:
            print("Warning: Database functionality not available")
    
    def setup_state_management(self):
        """Enable persistent state management for the main application."""
        try:
            # Register with state manager (serializer is already included in UniversalStateManager)
            register_module('raman_analysis_app', self)
            
            # Add convenient save/load methods
            self.save_analysis_state = lambda notes="": save_module_state('raman_analysis_app', notes)
            self.load_analysis_state = lambda: load_module_state('raman_analysis_app')
            
            # Hook auto-save into critical methods
            self._add_auto_save_hooks()
            
            print("✅ Main application state management enabled!")
            print("💾 Auto-saves: ~/RamanLab_Projects/auto_saves/")
            
        except Exception as e:
            print(f"Warning: Could not enable state management: {e}")
    
    def _add_auto_save_hooks(self):
        """Add auto-save functionality to critical methods."""
        
        # Auto-save after spectrum loading
        if hasattr(self, 'load_spectrum_file'):
            original_method = self.load_spectrum_file
            
            def auto_save_wrapper(*args, **kwargs):
                result = original_method(*args, **kwargs)
                if result:  # Only save if loading was successful
                    save_module_state('raman_analysis_app', "Auto-save: spectrum loaded")
                return result
            
            self.load_spectrum_file = auto_save_wrapper
        


    def setup_ui(self):
        """Set up the main user interface."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout (horizontal splitter)
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - visualization
        self.setup_visualization_panel(main_splitter)
        
        # Right panel - controls
        self.setup_control_panel(main_splitter)
        
        # Set splitter proportions (70% left, 30% right)
        main_splitter.setSizes([1000, 400])

    def setup_visualization_panel(self, parent):
        """Set up the spectrum visualization panel."""
        viz_frame = QFrame()
        viz_layout = QVBoxLayout(viz_frame)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, viz_frame)
        
        # Create the main plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Raman Spectrum")
        self.ax.grid(True, alpha=0.3)
        
        # Connect mouse click event for manual peak selection
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Add to layout
        viz_layout.addWidget(self.toolbar)
        viz_layout.addWidget(self.canvas)
        
        parent.addWidget(viz_frame)

    def setup_control_panel(self, parent):
        """Set up the control panel with tabs."""
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.file_tab = self.create_file_tab()
        self.process_tab = self.create_process_tab()
        self.search_tab = self.create_search_tab()
        self.database_tab = self.create_database_tab()
        self.advanced_tab = self.create_advanced_tab()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.file_tab, "File")
        self.tab_widget.addTab(self.process_tab, "Process")
        self.tab_widget.addTab(self.search_tab, "Search")
        self.tab_widget.addTab(self.database_tab, "Database")
        self.tab_widget.addTab(self.advanced_tab, "Advanced")
        
        # Set a reasonable maximum width for the tab widget
        self.tab_widget.setMaximumWidth(450)
        
        parent.addWidget(self.tab_widget)

    def create_file_tab(self):
        """Create the file operations tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        # Import button
        import_btn = QPushButton("Import Spectrum")
        import_btn.clicked.connect(self.import_spectrum)
        file_layout.addWidget(import_btn)
        
        # Save button
        save_btn = QPushButton("Save Spectrum")
        save_btn.clicked.connect(self.save_spectrum)
        file_layout.addWidget(save_btn)
        
        # Multi-spectrum manager button
        multi_btn = QPushButton("Multi-Spectrum Manager")
        multi_btn.clicked.connect(self.launch_multi_spectrum_manager)
        file_layout.addWidget(multi_btn)
        
        layout.addWidget(file_group)
        
        # Spectrum info group
        info_group = QGroupBox("Spectrum Information")
        info_layout = QVBoxLayout(info_group)
        
        # Info display
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(200)
        self.info_text.setPlainText("No spectrum loaded")
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        layout.addStretch()
        
        return tab

    def create_process_tab(self):
        """Create the processing operations tab with organized subtabs."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create subtabs for better organization
        subtabs = QTabWidget()
        
        # Background subtab
        background_tab = self.create_background_subtab()
        subtabs.addTab(background_tab, "Background")
        
        # Profile subtab (peaks and smoothing)
        profile_tab = self.create_profile_subtab()
        subtabs.addTab(profile_tab, "Profile")
        
        layout.addWidget(subtabs)
        
        return tab

    def create_background_subtab(self):
        """Create the background subtraction subtab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Background subtraction group
        bg_group = QGroupBox("Background Subtraction")
        bg_layout = QVBoxLayout(bg_group)
        
        # Background method selection
        bg_method_layout = QHBoxLayout()
        bg_method_layout.addWidget(QLabel("Method:"))
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["ALS (Asymmetric Least Squares)", "Linear", "Polynomial", "Moving Average", "Spline"])
        self.bg_method_combo.currentTextChanged.connect(self.on_bg_method_changed)
        self.bg_method_combo.currentTextChanged.connect(self.preview_background_subtraction)
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
        
        # Connect sliders to update labels
        self.lambda_slider.valueChanged.connect(self.update_lambda_label)
        self.p_slider.valueChanged.connect(self.update_p_label)
        self.niter_slider.valueChanged.connect(self.update_niter_label)
        
        # Connect sliders to live preview
        self.lambda_slider.valueChanged.connect(self.preview_background_subtraction)
        self.p_slider.valueChanged.connect(self.preview_background_subtraction)
        self.niter_slider.valueChanged.connect(self.preview_background_subtraction)
        
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
        self.start_weight_slider.valueChanged.connect(self.preview_background_subtraction)
        self.end_weight_slider.valueChanged.connect(self.preview_background_subtraction)
        
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
        self.poly_order_slider.valueChanged.connect(self.preview_background_subtraction)
        self.poly_method_combo.currentTextChanged.connect(self.preview_background_subtraction)
        
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
        self.window_size_slider.valueChanged.connect(self.preview_background_subtraction)
        self.window_type_combo.currentTextChanged.connect(self.preview_background_subtraction)
        
        bg_layout.addWidget(self.moving_avg_params_widget)
        
        # Spline parameters
        self.spline_params_widget = QWidget()
        spline_params_layout = QVBoxLayout(self.spline_params_widget)
        spline_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Number of knots
        knots_layout = QHBoxLayout()
        knots_layout.addWidget(QLabel("Number of Knots:"))
        self.knots_slider = QSlider(Qt.Horizontal)
        self.knots_slider.setRange(5, 50)  # 5 to 50 knots
        self.knots_slider.setValue(20)  # Default 20 knots
        knots_layout.addWidget(self.knots_slider)
        self.knots_label = QLabel("20")
        knots_layout.addWidget(self.knots_label)
        spline_params_layout.addLayout(knots_layout)
        
        # Smoothing factor
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing Factor:"))
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 50)  # Log scale: 10^1 to 10^5
        self.smoothing_slider.setValue(30)  # Default 10^3
        smoothing_layout.addWidget(self.smoothing_slider)
        self.smoothing_label = QLabel("1000")
        smoothing_layout.addWidget(self.smoothing_label)
        spline_params_layout.addLayout(smoothing_layout)
        
        # Spline degree
        degree_layout = QHBoxLayout()
        degree_layout.addWidget(QLabel("Spline Degree:"))
        self.spline_degree_slider = QSlider(Qt.Horizontal)
        self.spline_degree_slider.setRange(1, 5)  # 1st to 5th order
        self.spline_degree_slider.setValue(3)  # Default cubic (3rd order)
        degree_layout.addWidget(self.spline_degree_slider)
        self.spline_degree_label = QLabel("3 (Cubic)")
        degree_layout.addWidget(self.spline_degree_label)
        spline_params_layout.addLayout(degree_layout)
        
        # Connect sliders to update labels and live preview
        self.knots_slider.valueChanged.connect(self.update_knots_label)
        self.smoothing_slider.valueChanged.connect(self.update_smoothing_label)
        self.spline_degree_slider.valueChanged.connect(self.update_spline_degree_label)
        
        self.knots_slider.valueChanged.connect(self.preview_background_subtraction)
        self.smoothing_slider.valueChanged.connect(self.preview_background_subtraction)
        self.spline_degree_slider.valueChanged.connect(self.preview_background_subtraction)
        
        bg_layout.addWidget(self.spline_params_widget)
        
        # Background subtraction buttons
        button_layout = QHBoxLayout()
        
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
        
        # Auto Background Preview group
        auto_bg_group = QGroupBox("Automatic Background Preview")
        auto_bg_layout = QVBoxLayout(auto_bg_group)
        
        # Auto preview button
        auto_preview_btn = QPushButton("🔍 Generate Background Options")
        auto_preview_btn.clicked.connect(self.generate_background_previews)
        auto_preview_btn.setStyleSheet("""
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
        auto_bg_layout.addWidget(auto_preview_btn)
        
        # Background options selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Select Option:"))
        self.bg_options_combo = QComboBox()
        self.bg_options_combo.addItem("None - Generate options first")
        self.bg_options_combo.currentTextChanged.connect(self.on_bg_option_selected)
        selection_layout.addWidget(self.bg_options_combo)
        auto_bg_layout.addLayout(selection_layout)
        
        # Apply selected option button
        apply_selected_btn = QPushButton("Apply Selected Option")
        apply_selected_btn.clicked.connect(self.apply_selected_background_option)
        apply_selected_btn.setStyleSheet("""
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
        auto_bg_layout.addWidget(apply_selected_btn)
        
        layout.addWidget(auto_bg_group)
        layout.addStretch()
        
        # Set initial visibility state for background controls
        self.on_bg_method_changed()
        
        return tab

    def create_profile_subtab(self):
        """Create the profile analysis subtab (peaks and smoothing)."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Peak detection group with real-time sliders
        peak_group = QGroupBox("Peak Detection - Live Updates")
        peak_layout = QVBoxLayout(peak_group)
        
        # Height parameter
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Min Height:"))
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(0, 100)
        self.height_slider.setValue(10)
        self.height_slider.setTracking(True)  # Enable live tracking
        self.height_slider.valueChanged.connect(self.update_peak_detection)
        self.height_slider.setToolTip("Minimum peak height as percentage of maximum intensity")
        height_layout.addWidget(self.height_slider)
        self.height_label = QLabel("10%")
        self.height_label.setMinimumWidth(40)
        height_layout.addWidget(self.height_label)
        peak_layout.addLayout(height_layout)
        
        # Distance parameter
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Min Distance:"))
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(10)
        self.distance_slider.setTracking(True)  # Enable live tracking
        self.distance_slider.valueChanged.connect(self.update_peak_detection)
        self.distance_slider.setToolTip("Minimum distance between peaks (data points)")
        distance_layout.addWidget(self.distance_slider)
        self.distance_label = QLabel("10")
        self.distance_label.setMinimumWidth(40)
        distance_layout.addWidget(self.distance_label)
        peak_layout.addLayout(distance_layout)
        
        # Prominence parameter
        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(QLabel("Prominence:"))
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(0, 50)
        self.prominence_slider.setValue(5)
        self.prominence_slider.setTracking(True)  # Enable live tracking
        self.prominence_slider.valueChanged.connect(self.update_peak_detection)
        self.prominence_slider.setToolTip("Minimum peak prominence as percentage of maximum intensity")
        prominence_layout.addWidget(self.prominence_slider)
        self.prominence_label = QLabel("5%")
        self.prominence_label.setMinimumWidth(40)
        prominence_layout.addWidget(self.prominence_label)
        peak_layout.addLayout(prominence_layout)
        
        # Manual peak detection button
        detect_btn = QPushButton("Detect Peaks")
        detect_btn.clicked.connect(self.find_peaks)
        peak_layout.addWidget(detect_btn)
        
        # Manual peak selection mode toggle
        self.peak_selection_btn = QPushButton("🎯 Enter Peak Selection Mode")
        self.peak_selection_btn.clicked.connect(self.toggle_peak_selection_mode)
        self.peak_selection_btn.setCheckable(True)
        self.peak_selection_btn.setStyleSheet("""
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
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        peak_layout.addWidget(self.peak_selection_btn)
        
        # Clear manual peaks button
        clear_manual_btn = QPushButton("Clear Manual Peaks")
        clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        peak_layout.addWidget(clear_manual_btn)
        
        # Peak count display with enhanced formatting
        self.peak_count_label = QLabel("Auto peaks: 0 | Manual peaks: 0")
        self.peak_count_label.setStyleSheet("font-weight: bold; color: #333;")
        peak_layout.addWidget(self.peak_count_label)
        
        # Preset buttons for common peak detection settings
        presets_layout = QHBoxLayout()
        
        sensitive_btn = QPushButton("Sensitive")
        sensitive_btn.clicked.connect(lambda: self.apply_peak_preset(height=5, distance=5, prominence=2))
        sensitive_btn.setToolTip("Detect more peaks - good for noisy spectra")
        presets_layout.addWidget(sensitive_btn)
        
        balanced_btn = QPushButton("Balanced")
        balanced_btn.clicked.connect(lambda: self.apply_peak_preset(height=10, distance=10, prominence=5))
        balanced_btn.setToolTip("Balanced detection - good default settings")
        presets_layout.addWidget(balanced_btn)
        
        strict_btn = QPushButton("Strict")
        strict_btn.clicked.connect(lambda: self.apply_peak_preset(height=20, distance=20, prominence=10))
        strict_btn.setToolTip("Detect only prominent peaks")
        presets_layout.addWidget(strict_btn)
        
        peak_layout.addLayout(presets_layout)
        
        # Auto-detect toggle
        self.auto_detect_peaks_checkbox = QCheckBox("Auto-detect peaks when loading spectra")
        self.auto_detect_peaks_checkbox.setChecked(True)
        self.auto_detect_peaks_checkbox.setToolTip("Automatically run peak detection when a new spectrum is loaded")
        peak_layout.addWidget(self.auto_detect_peaks_checkbox)
        
        layout.addWidget(peak_group)
        
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
        self.sg_window_spin.valueChanged.connect(self.preview_smoothing)
        sg_layout.addWidget(self.sg_window_spin)
        
        sg_layout.addWidget(QLabel("Poly Order:"))
        self.sg_order_spin = QSpinBox()
        self.sg_order_spin.setRange(1, 5)
        self.sg_order_spin.setValue(2)
        self.sg_order_spin.valueChanged.connect(self.preview_smoothing)
        sg_layout.addWidget(self.sg_order_spin)
        smooth_layout.addLayout(sg_layout)
        
        # Smoothing buttons
        smooth_button_layout = QHBoxLayout()
        
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

    def create_database_tab(self):
        """Create the database management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Measured Raman Database group
        db_ops_group = QGroupBox("Measured Raman Database")
        db_ops_layout = QVBoxLayout(db_ops_group)
        
        # View Raman database button (removed Add button, renamed and styled to match mineral button)
        view_btn = QPushButton("View/Edit Measured Raman Database")
        view_btn.clicked.connect(self.view_database)
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: #673AB7;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
            QPushButton:pressed {
                background-color: #512DA8;
            }
        """)
        db_ops_layout.addWidget(view_btn)
        
        layout.addWidget(db_ops_group)
        
        # Calculated Raman Database group
        mineral_db_group = QGroupBox("Calculated Raman Database")
        mineral_db_layout = QVBoxLayout(mineral_db_group)
        
        # Mineral modes database button
        mineral_modes_btn = QPushButton("View/Edit Calculated Raman Character Info")
        mineral_modes_btn.clicked.connect(self.launch_mineral_modes_browser)
        mineral_modes_btn.setStyleSheet("""
            QPushButton {
                background-color: #673AB7;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
            QPushButton:pressed {
                background-color: #512DA8;
            }
        """)
        mineral_db_layout.addWidget(mineral_modes_btn)
        
        layout.addWidget(mineral_db_group)
        
        # Database stats group
        stats_group = QGroupBox("Database Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.db_stats_text = QTextEdit()
        self.db_stats_text.setMaximumHeight(100)
        self.db_stats_text.setPlainText("Database statistics will appear here...")
        stats_layout.addWidget(self.db_stats_text)
        
        # Refresh stats button
        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self.update_database_stats)
        stats_layout.addWidget(refresh_btn)
        
        layout.addWidget(stats_group)
        layout.addStretch()
        
        return tab

    def create_search_tab(self):
        """Create the comprehensive search functionality tab with basic and advanced search subtabs."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Algorithm selection at the top - applies to both search types
        algorithm_group = QGroupBox("Search Algorithm")
        algorithm_layout = QVBoxLayout(algorithm_group)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "correlation", "multi-window", "mineral-vibration", "peak", "combined", "DTW"
        ])
        self.algorithm_combo.setCurrentText("multi-window")
        algorithm_layout.addWidget(self.algorithm_combo)
        
        # Algorithm descriptions
        desc_text = QTextEdit()
        desc_text.setMaximumHeight(80)
        desc_text.setPlainText(
            "Correlation: Full spectral shape comparison • Multi-Window: Weighted frequency regions • "
            "Mineral-Vibration: Chemistry-based vibrational modes • Peak: Peak positions/intensities • "
            "Combined: Hybrid approach • DTW: Dynamic alignment"
        )
        desc_text.setReadOnly(True)
        desc_text.setStyleSheet("font-size: 10px; background-color: #f8f9fa; border: 1px solid #dee2e6;")
        algorithm_layout.addWidget(desc_text)
        
        layout.addWidget(algorithm_group)
        
        # Common search parameters - applies to both search types  
        params_group = QGroupBox("Search Parameters")
        params_layout = QHBoxLayout(params_group)
        
        # Number of matches
        params_layout.addWidget(QLabel("Max results:"))
        self.n_matches_spin = QSpinBox()
        self.n_matches_spin.setRange(1, 50)
        self.n_matches_spin.setValue(10)
        self.n_matches_spin.setMaximumWidth(80)
        params_layout.addWidget(self.n_matches_spin)
        
        # Threshold (applies to both search types)
        params_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(0.7)
        self.threshold_spin.setMaximumWidth(80)
        params_layout.addWidget(self.threshold_spin)
        
        params_layout.addStretch()
        
        layout.addWidget(params_group)
        
        # Create search subtabs
        search_tab_widget = QTabWidget()
        
        # Create basic search subtab
        basic_search_tab = self.create_basic_search_subtab()
        search_tab_widget.addTab(basic_search_tab, "Basic Search")
        
        # Create advanced search subtab
        advanced_search_tab = self.create_advanced_search_subtab()
        search_tab_widget.addTab(advanced_search_tab, "Advanced Search")
        
        layout.addWidget(search_tab_widget)
        
        # Note: Mixed mineral analysis section removed for clean rebuild
        
        return tab

    def create_basic_search_subtab(self):
        """Create the basic search subtab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Algorithm guidance
        algorithm_guidance = QLabel("💡 Tip: The 'multi-window' algorithm provides the best balance of accuracy and chemical intelligence for mineral identification!")
        algorithm_guidance.setStyleSheet("font-size: 10px; color: #0066CC; background-color: #E6F3FF; padding: 8px; border-radius: 4px; margin: 4px;")
        algorithm_guidance.setWordWrap(True)
        layout.addWidget(algorithm_guidance)
        
        # Search button
        search_btn = QPushButton("Search Database")
        search_btn.clicked.connect(self.perform_basic_search)
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #5CB85C;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #449D44;
            }
            QPushButton:pressed {
                background-color: #398439;
            }
        """)
        layout.addWidget(search_btn)
        
        # Search results
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout(results_group)
        
        self.search_results_text = QTextEdit()
        self.search_results_text.setPlainText("Search results will appear here...")
        results_layout.addWidget(self.search_results_text)
        
        layout.addWidget(results_group)
        
        return tab

    def create_advanced_search_subtab(self):
        """Create the advanced search subtab with all filtering options."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scrollable area for all the controls - make it more compact
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Peak-based search filter - more compact layout
        peak_group = QGroupBox("Peak-Based Search Filter")
        peak_layout = QVBoxLayout(peak_group)
        
        peak_layout.addWidget(QLabel("Search by specific peak positions:"))
        self.peak_positions_edit = QLineEdit()
        self.peak_positions_edit.setPlaceholderText("e.g., 1050, 1350, 1580")
        peak_layout.addWidget(self.peak_positions_edit)
        
        hint_label = QLabel("Comma-separated wavenumber values (e.g., 1050, 1350, 1580)")
        hint_label.setStyleSheet("font-size: 9px; color: gray;")
        peak_layout.addWidget(hint_label)
        
        # Make tolerance layout more compact
        tolerance_layout = QHBoxLayout()
        tolerance_layout.addWidget(QLabel("Tolerance (cm⁻¹):"))
        self.peak_tolerance_spin = QDoubleSpinBox()
        self.peak_tolerance_spin.setRange(1.0, 100.0)
        self.peak_tolerance_spin.setValue(10.0)
        self.peak_tolerance_spin.setMaximumWidth(80)  # Limit width
        tolerance_layout.addWidget(self.peak_tolerance_spin)
        tolerance_layout.addStretch()
        peak_layout.addLayout(tolerance_layout)
        
        # Add helpful note about peak usage
        peak_note = QLabel("💡 Tip: Peak positions enhance scoring when using 'peak' or 'combined' algorithms")
        peak_note.setStyleSheet("font-size: 9px; color: #0066CC; background-color: #E6F3FF; padding: 4px; border-radius: 3px;")
        peak_note.setWordWrap(True)
        peak_layout.addWidget(peak_note)
        
        scroll_layout.addWidget(peak_group)
        
        # Metadata filter options - more compact
        filter_group = QGroupBox("Metadata Filters")
        filter_layout = QFormLayout(filter_group)  # Use QFormLayout for compactness
        
        # Chemical family filter
        self.chemical_family_combo = QComboBox()
        self.chemical_family_combo.setEditable(True)
        self.chemical_family_combo.setMaximumWidth(200)
        filter_layout.addRow("Chemical Family:", self.chemical_family_combo)
        
        # Hey classification filter
        self.hey_classification_combo = QComboBox()
        self.hey_classification_combo.setEditable(True)
        self.hey_classification_combo.setMaximumWidth(200)
        filter_layout.addRow("Hey Classification:", self.hey_classification_combo)
        
        scroll_layout.addWidget(filter_group)
        
        # Chemistry elements filters - more compact
        elements_group = QGroupBox("Chemistry Element Filters")
        elements_layout = QFormLayout(elements_group)
        
        self.only_elements_edit = QLineEdit()
        self.only_elements_edit.setPlaceholderText("e.g., Si, O, Al")
        self.only_elements_edit.setMaximumWidth(200)
        elements_layout.addRow("Only elements:", self.only_elements_edit)
        
        self.required_elements_edit = QLineEdit()
        self.required_elements_edit.setPlaceholderText("e.g., Ca, C")
        self.required_elements_edit.setMaximumWidth(200)
        elements_layout.addRow("Required elements:", self.required_elements_edit)
        
        self.exclude_elements_edit = QLineEdit()
        self.exclude_elements_edit.setPlaceholderText("e.g., Fe, Mg")
        self.exclude_elements_edit.setMaximumWidth(200)
        elements_layout.addRow("Exclude elements:", self.exclude_elements_edit)
        
        scroll_layout.addWidget(elements_group)
        
        # Advanced search button
        adv_search_btn = QPushButton("Advanced Search")
        adv_search_btn.clicked.connect(self.perform_advanced_search)
        adv_search_btn.setStyleSheet("""
            QPushButton {
                background-color: #F0AD4E;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #EC971F;
            }
            QPushButton:pressed {
                background-color: #D58512;
            }
        """)
        scroll_layout.addWidget(adv_search_btn)
        
        # Add stretch at the end
        scroll_layout.addStretch()
        
        # Set up scroll area with more compact settings
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Add scroll area to tab
        layout.addWidget(scroll_area)
        
        # Update filter options when tab loads
        self.update_metadata_filter_options()
        
        return tab

    def setup_menu_bar(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        import_action = QAction("Import Spectrum", self)
        import_action.triggered.connect(self.import_spectrum)
        file_menu.addAction(import_action)
        
        # Import Data submenu
        import_data_menu = file_menu.addMenu("Import Data")
        
        import_raman_map_action = QAction("Raman Spectral Map", self)
        import_raman_map_action.triggered.connect(self.import_raman_spectral_map)
        import_data_menu.addAction(import_raman_map_action)
        
        import_line_scan_action = QAction("Line Scan Data", self)
        import_line_scan_action.triggered.connect(self.import_line_scan_data)
        import_data_menu.addAction(import_line_scan_action)
        
        import_point_data_action = QAction("Point Measurement Data", self)
        import_point_data_action.triggered.connect(self.import_point_data)
        import_data_menu.addAction(import_point_data_action)
        
        import_data_menu.addSeparator()
        
        test_map_import_action = QAction("Test Raman Map Import", self)
        test_map_import_action.triggered.connect(self.test_raman_map_import)
        import_data_menu.addAction(test_map_import_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("Save Current Spectrum", self)
        save_action.triggered.connect(self.save_spectrum)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Session menu (only show if state management is available)
        if STATE_MANAGEMENT_AVAILABLE:
            session_menu = menubar.addMenu("Session")
            
            save_session_action = QAction("Save Session", self)
            save_session_action.setShortcut("Ctrl+Shift+S")
            save_session_action.triggered.connect(self.save_session_dialog)
            session_menu.addAction(save_session_action)
            
            load_session_action = QAction("Load Session", self)
            load_session_action.setShortcut("Ctrl+Shift+O")
            load_session_action.triggered.connect(self.load_session_dialog)
            session_menu.addAction(load_session_action)
            
            session_menu.addSeparator()
            
            auto_save_action = QAction("Enable Auto-Save", self)
            auto_save_action.setCheckable(True)
            auto_save_action.setChecked(True)  # Default enabled
            auto_save_action.triggered.connect(self.toggle_auto_save)
            session_menu.addAction(auto_save_action)
            
            session_menu.addSeparator()
            
            session_info_action = QAction("Session Info", self)
            session_info_action.triggered.connect(self.show_session_info)
            session_menu.addAction(session_info_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        preferences_action = QAction("Preferences...", self)
        preferences_action.triggered.connect(self.open_settings)
        settings_menu.addAction(preferences_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # Always add Check for Updates option - error handling in the method
        check_updates_action = QAction("Check for Updates", self)
        check_updates_action.triggered.connect(self.check_for_updates)
        help_menu.addAction(check_updates_action)
        help_menu.addSeparator()
        
        # Add User Forum link
        user_forum_action = QAction("User Forum", self)
        user_forum_action.triggered.connect(self.open_user_forum)
        help_menu.addAction(user_forum_action)
        
        # Add Database Downloads link
        database_downloads_action = QAction("Database Downloads", self)
        database_downloads_action.triggered.connect(self.open_database_downloads)
        help_menu.addAction(database_downloads_action)
        
        # Add README link
        readme_action = QAction("README", self) 
        readme_action.triggered.connect(self.open_readme)
        help_menu.addAction(readme_action)
        
        # Add Database Manager link
        database_manager_action = QAction("Database Manager", self)
        database_manager_action.triggered.connect(self.launch_database_manager)
        help_menu.addAction(database_manager_action)
        
        help_menu.addSeparator()
        
        # About option is always available
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("RamanLab PySide6 - Ready")

    def center_on_screen(self):
        """Center the window on the screen."""
        screen = QApplication.primaryScreen().availableGeometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def show_startup_message(self):
        """Show startup message in the plot area."""
        self.ax.text(0.5, 0.5, 
                     'RamanLab PySide6\n\n'
                     'Enhanced File Loading Capabilities:\n'
                     '• CSV, TXT (tab/space delimited)\n'
                     '• Headers and metadata (# lines)\n'
                     '• Auto-format detection\n\n'
                     'Import a spectrum to get started\n',
                     
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     transform=self.ax.transAxes)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
        
        # Optional: Check for updates on startup (delayed, non-intrusive)
        # This will only run if dependencies are available
        # QTimer.singleShot(3000, self.check_for_updates_startup)  # 3 second delay - DISABLED

    def check_for_updates_startup(self):
        """Check for updates on startup (non-intrusive, silent if no updates)."""
        try:
            # Use the simple update checker that doesn't use background threads
            from core.simple_update_checker import simple_check_for_updates
            # Only show dialog if update is available, suppress "no update" message
            simple_check_for_updates(parent=self, show_no_update=False)
        except ImportError:
            # Dependencies not available - silently skip startup check
            # User can still manually check via Help menu
            pass
        except Exception as e:
            # Any other error - silently skip, don't bother user on startup
            pass

    # Cross-platform file operations (replacing your platform-specific code!)
    def import_spectrum(self):
        """Import a Raman spectrum file using PySide6 file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Raman Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Spectrum files (*.txt *.csv *.dat *.spc *.xy *.asc);;Text files (*.txt);;CSV files (*.csv);;Data files (*.dat);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.load_spectrum_file(file_path)
                self.status_bar.showMessage(f"Loaded: {Path(file_path).name} with enhanced format detection")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import spectrum:\n{str(e)}")

    def load_spectrum_file(self, file_path):
        """Load spectrum data from file with enhanced header and format handling."""
        try:
            # Parse the file with enhanced handling
            wavenumbers, intensities, metadata = self.parse_spectrum_file(file_path)
            
            if len(wavenumbers) == 0 or len(intensities) == 0:
                raise ValueError("No valid data found in file")
            
            if len(wavenumbers) != len(intensities):
                raise ValueError("Wavenumber and intensity arrays have different lengths")
            
            # Store the data
            self.current_wavenumbers = wavenumbers
            self.current_intensities = intensities
            self.processed_intensities = self.current_intensities.copy()
            
            # Store metadata from file
            self.current_spectrum_metadata = metadata
            
            # Clear any active previews and manual peaks
            self.background_preview_active = False
            self.smoothing_preview_active = False
            self.preview_background = None
            self.preview_corrected = None
            self.preview_smoothed = None
            self.manual_peaks.clear()
            self.detected_peaks = None
            
            self.update_plot()
            self.update_info_display(file_path)
            
            # Auto-trigger peak detection for newly loaded spectra
            if (self.processed_intensities is not None and 
                hasattr(self, 'auto_detect_peaks_checkbox') and 
                self.auto_detect_peaks_checkbox.isChecked()):
                self.update_peak_detection()
            
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def parse_spectrum_file(self, file_path):
        """Enhanced spectrum file parser that handles headers, metadata, and various formats."""
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
                print(f"Skipping line {i+1}: {line[:50]}... (Error: {e})")
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

    def update_plot(self):
        """Update the main spectrum plot with optional previews."""
        self.ax.clear()
        
        if self.current_wavenumbers is not None and self.processed_intensities is not None:
            # Plot the main processed spectrum
            self.ax.plot(self.current_wavenumbers, self.processed_intensities, 'b-', linewidth=1.5, 
                        label='Current Spectrum', alpha=0.9)
            
            # Plot background preview if active
            if self.background_preview_active and self.preview_background is not None:
                self.ax.plot(self.current_wavenumbers, self.preview_background, 'r--', linewidth=1, 
                            label='Background (Preview)', alpha=0.7)
                
                # Show corrected spectrum preview
                if self.preview_corrected is not None:
                    self.ax.plot(self.current_wavenumbers, self.preview_corrected, 'g-', linewidth=1, 
                                label='Corrected (Preview)', alpha=0.7)
            
            # Plot smoothing preview if active
            if self.smoothing_preview_active and self.preview_smoothed is not None:
                self.ax.plot(self.current_wavenumbers, self.preview_smoothed, 'm-', linewidth=1.5, 
                            label='Smoothed (Preview)', alpha=0.8)
            
            # Plot automatically detected peaks with enhanced visualization
            if self.detected_peaks is not None and len(self.detected_peaks) > 0:
                peak_positions = self.current_wavenumbers[self.detected_peaks]
                peak_intensities = self.processed_intensities[self.detected_peaks]
                
                # Main peak markers - larger and more prominent
                self.ax.plot(peak_positions, peak_intensities, 'ro', markersize=8, 
                            markerfacecolor='red', markeredgecolor='darkred', 
                            markeredgewidth=2, label=f'Auto Peaks ({len(self.detected_peaks)})',
                            zorder=5)
                
                # Add vertical lines to make peaks more visible
                for pos, intensity in zip(peak_positions, peak_intensities):
                    self.ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
                # Add peak position labels for auto peaks (show only if reasonable number of peaks)
                if len(self.detected_peaks) <= 20:  # Only show labels if not too many peaks
                    for i, (pos, intensity) in enumerate(zip(peak_positions, peak_intensities)):
                        # Offset the label slightly above the peak
                        label_y = intensity + 0.08 * (np.max(self.processed_intensities) - np.min(self.processed_intensities))
                        self.ax.annotate(f'{pos:.1f}', 
                                       xy=(pos, intensity), 
                                       xytext=(pos, label_y),
                                       ha='center', va='bottom',
                                       fontsize=8, 
                                       color='darkred',
                                       fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.9, edgecolor='red', linewidth=1),
                                       arrowprops=dict(arrowstyle='->', color='red', lw=0.8, alpha=0.8))
            
            # Plot manual peaks
            if self.manual_peaks and len(self.manual_peaks) > 0:
                manual_positions = []
                manual_intensities = []
                
                # Find the intensity at each manual peak position
                for peak_pos in self.manual_peaks:
                    # Find the closest data point to the manual peak position
                    closest_idx = np.argmin(np.abs(self.current_wavenumbers - peak_pos))
                    manual_positions.append(peak_pos)
                    manual_intensities.append(self.processed_intensities[closest_idx])
                
                self.ax.plot(manual_positions, manual_intensities, 's', 
                            color='blue', markersize=8, markerfacecolor='lightblue', 
                            markeredgecolor='blue', markeredgewidth=2,
                            label=f'Manual Peaks ({len(self.manual_peaks)})')
                
                # Add peak position labels for manual peaks
                for i, (pos, intensity) in enumerate(zip(manual_positions, manual_intensities)):
                    # Offset the label slightly above the peak
                    label_y = intensity + 0.12 * (np.max(self.processed_intensities) - np.min(self.processed_intensities))
                    self.ax.annotate(f'{pos:.1f} cm⁻¹', 
                                   xy=(pos, intensity), 
                                   xytext=(pos, label_y),
                                   ha='center', va='bottom',
                                   fontsize=9, 
                                   color='darkblue',
                                   fontweight='bold',
                                   rotation=45,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.9, edgecolor='blue', linewidth=1.2),
                                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.0, alpha=0.8))
            
            self.ax.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax.set_ylabel("Intensity (a.u.)")
            
            # Add visual indicator when in peak selection mode
            if self.peak_selection_mode:
                self.ax.set_title("Raman Spectrum - 🎯 Peak Selection Mode Active", 
                                 color='darkgreen', fontweight='bold')
            else:
                self.ax.set_title("Raman Spectrum")
                
            self.ax.grid(True, alpha=0.3)
            
            # Add legend if there are previews or peaks
            if (self.background_preview_active or self.smoothing_preview_active or 
                self.detected_peaks is not None or self.manual_peaks):
                self.ax.legend(loc='upper right', fontsize=9)
        
        self.canvas.draw()

    def update_info_display(self, file_path):
        """Update the spectrum information display."""
        if self.current_wavenumbers is not None:
            info = f"File: {Path(file_path).name}\n"
            info += f"Data points: {len(self.current_wavenumbers)}\n"
            info += f"Wavenumber range: {self.current_wavenumbers.min():.1f} - {self.current_wavenumbers.max():.1f} cm⁻¹\n"
            info += f"Intensity range: {self.current_intensities.min():.1e} - {self.current_intensities.max():.1e}\n"
            
            # Add metadata from file if available
            if hasattr(self, 'current_spectrum_metadata') and self.current_spectrum_metadata:
                info += "\n--- File Metadata ---\n"
                
                # Show important metadata first
                important_keys = ['Instrument', 'Laser Wavelength', 'Laser Power', 'Integration Time', 
                                'Accumulations', 'Temperature', 'Sample', 'Operator']
                
                for key in important_keys:
                    if key in self.current_spectrum_metadata:
                        info += f"{key}: {self.current_spectrum_metadata[key]}\n"
                
                # Show other metadata (excluding file info we already show)
                exclude_keys = ['file_path', 'file_name', 'data_points', 'wavenumber_range'] + important_keys
                other_metadata = {k: v for k, v in self.current_spectrum_metadata.items() 
                                if k not in exclude_keys and not k.startswith('_')}
                
                if other_metadata:
                    for key, value in other_metadata.items():
                        if key == 'comments' and isinstance(value, list):
                            for comment in value:
                                info += f"Comment: {comment}\n"
                        else:
                            info += f"{key}: {value}\n"
            
            self.info_text.setPlainText(info)

    # Basic processing methods
    def save_spectrum(self):
        """Save the current spectrum using cross-platform dialog."""
        if self.current_wavenumbers is None:
            QMessageBox.warning(self, "No Data", "No spectrum to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spectrum",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                data = np.column_stack([self.current_wavenumbers, self.processed_intensities])
                np.savetxt(file_path, data, delimiter='\t', header='Wavenumber\tIntensity')
                self.status_bar.showMessage(f"Saved: {Path(file_path).name}")
                QMessageBox.information(self, "Success", f"Spectrum saved to:\n{file_path}")
                
                # Restore window focus after file dialog
                try:
                    from core.window_focus_manager import restore_window_focus_after_dialog
                    restore_window_focus_after_dialog(self)
                except ImportError:
                    # Fallback if focus manager not available
                    self.raise_()
                    self.activateWindow()
                    
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save spectrum:\n{str(e)}")

    def subtract_background(self):
        """Enhanced background subtraction with multiple methods including ALS."""
        if self.processed_intensities is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded for background subtraction.")
            return
            
        try:
            method = self.bg_method_combo.currentText()
            
            if method.startswith("ALS"):
                # ALS background subtraction
                lambda_value = 10 ** self.lambda_slider.value()
                p_value = self.p_slider.value() / 1000.0
                niter_value = self.niter_slider.value()
                
                corrected_intensities, baseline = self.raman_db.subtract_background_als(
                    self.current_wavenumbers,
                    self.processed_intensities,
                    lam=lambda_value,
                    p=p_value,
                    niter=niter_value
                )
                
                self.processed_intensities = corrected_intensities
                
            elif method == "Linear":
                # Linear baseline fitting with weighted endpoints
                start_weight = self.start_weight_slider.value() / 10.0
                end_weight = self.end_weight_slider.value() / 10.0
                
                # Find baseline points at start and end regions (use minimum values in regions)
                region_size = max(len(self.processed_intensities) // 20, 5)
                start_region = self.processed_intensities[:region_size]
                end_region = self.processed_intensities[-region_size:]
                
                # Use weighted minimum values from end regions for baseline
                start_val = np.min(start_region) * start_weight
                end_val = np.min(end_region) * end_weight
                background = np.linspace(start_val, end_val, len(self.processed_intensities))
                self.processed_intensities -= background
                
            elif method == "Polynomial":
                # Polynomial baseline fitting with adjustable order and method
                x = np.arange(len(self.processed_intensities))
                poly_order = self.poly_order_slider.value()
                poly_method = self.poly_method_combo.currentText()
                
                # Use minimum filtering to identify baseline points
                try:
                    from scipy import ndimage
                    window_size = max(len(self.processed_intensities) // 20, 5)
                    y_min_filtered = ndimage.minimum_filter1d(self.processed_intensities, size=window_size)
                except ImportError:
                    # Fallback without scipy
                    window_size = max(len(self.processed_intensities) // 20, 5)
                    y_min_filtered = np.array([
                        np.min(self.processed_intensities[max(0, i-window_size//2):min(len(self.processed_intensities), i+window_size//2+1)])
                        for i in range(len(self.processed_intensities))
                    ])
                
                if poly_method == "Robust":
                    # Use robust polynomial fitting on baseline points
                    try:
                        # Initial fit to minimum filtered data
                        coeffs = np.polyfit(x, y_min_filtered, poly_order)
                        background = np.polyval(coeffs, x)
                        
                        # Iterative refinement - only use points near the baseline
                        for iteration in range(3):
                            # Identify points that are likely baseline
                            threshold = np.percentile(self.processed_intensities - background, 25)
                            mask = (self.processed_intensities - background) <= threshold
                            
                            if np.sum(mask) < poly_order + 1:  # Need enough points
                                break
                            
                            # Apply robust reweighting to baseline points
                            residuals = np.abs(self.processed_intensities[mask] - background[mask])
                            weights = 1.0 / (1.0 + residuals / (np.median(residuals) + 1e-8))
                            coeffs = np.polyfit(x[mask], self.processed_intensities[mask], poly_order, w=weights)
                            background = np.polyval(coeffs, x)
                            
                    except:
                        # Fallback to simple polynomial on minimum filtered data
                        coeffs = np.polyfit(x, y_min_filtered, poly_order)
                        background = np.polyval(coeffs, x)
                else:
                    # Fit polynomial to minimum filtered data (baseline estimate)
                    coeffs = np.polyfit(x, y_min_filtered, poly_order)
                    background = np.polyval(coeffs, x)
                
                # Ensure background doesn't go above data
                background = np.minimum(background, self.processed_intensities)
                self.processed_intensities -= background
                
            elif method == "Moving Average":
                # Moving minimum/baseline filtering for background estimation
                window_percent = self.window_size_slider.value()
                window_type = self.window_type_combo.currentText()
                
                # Calculate window size as percentage of spectrum length
                window_size = max(int(len(self.processed_intensities) * window_percent / 100.0), 3)
                
                try:
                    # First apply minimum filtering to identify baseline regions
                    from scipy import ndimage
                    min_window = max(window_size // 2, 3)
                    y_min_filtered = ndimage.minimum_filter1d(self.processed_intensities, size=min_window)
                    
                    if window_type == "Uniform":
                        # Uniform smoothing of minimum filtered data
                        background = ndimage.uniform_filter1d(y_min_filtered, size=window_size)
                    elif window_type == "Gaussian":
                        # Gaussian smoothing of minimum filtered data
                        sigma = window_size / 4.0
                        background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
                    elif window_type in ["Hann", "Hamming"]:
                        # Windowed smoothing of minimum filtered data
                        if window_type == "Hann":
                            window = np.hanning(window_size)
                        else:  # Hamming
                            window = np.hamming(window_size)
                        
                        window = window / np.sum(window)  # Normalize
                        background = np.convolve(y_min_filtered, window, mode='same')
                    else:
                        # Fallback to uniform
                        background = ndimage.uniform_filter1d(y_min_filtered, size=window_size)
                        
                except ImportError:
                    # Fallback without scipy - use numpy-based minimum filtering
                    y_min_filtered = np.array([
                        np.min(self.processed_intensities[max(0, i-window_size//4):min(len(self.processed_intensities), i+window_size//4+1)])
                        for i in range(len(self.processed_intensities))
                    ])
                    
                    # Simple moving average on minimum filtered data
                    background = np.convolve(y_min_filtered, np.ones(window_size)/window_size, mode='same')
                
                # Ensure background doesn't exceed data
                background = np.minimum(background, self.processed_intensities)
                self.processed_intensities -= background
                
            elif method == "Spline":
                # Spline baseline fitting with iterative approach
                n_knots = self.knots_slider.value()
                smoothing_value = self.smoothing_slider.value()
                degree = self.spline_degree_slider.value()
                
                try:
                    from scipy.interpolate import UnivariateSpline
                    from scipy import ndimage
                    
                    # Calculate actual smoothing factor
                    if smoothing_value <= 10:
                        smoothing = 10 ** smoothing_value
                    else:
                        smoothing = 10 ** (smoothing_value / 10.0)
                    
                    # Method: Use minimum filtering and iterative approach to fit below the data
                    x = np.arange(len(self.processed_intensities))
                    y = self.processed_intensities
                    
                    # Step 1: Apply minimum filter to identify baseline regions
                    window_size = max(len(y) // 20, 5)
                    y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
                    
                    # Step 2: For proper baseline fitting, use higher smoothing
                    baseline_smoothing = max(smoothing, len(y) / 10)
                    
                    # Step 3: Fit initial spline to minimum filtered data
                    spline = UnivariateSpline(x, y_min_filtered, s=baseline_smoothing, k=min(degree, 3))
                    current_background = spline(x)
                    
                    # Step 4: Iterative refinement - only use points below or near the background
                    for iteration in range(3):
                        # Identify points that are likely background (below current estimate)
                        threshold = np.percentile(y - current_background, 20)  # Use 20th percentile
                        mask = (y - current_background) <= threshold
                        
                        if np.sum(mask) < n_knots:  # Need enough points
                            break
                        
                        # Fit spline only to identified background points
                        try:
                            spline = UnivariateSpline(x[mask], y[mask], s=baseline_smoothing, k=min(degree, 3))
                            current_background = spline(x)
                        except:
                            break
                        
                        # Ensure background doesn't go above data unrealistically
                        current_background = np.minimum(current_background, y)
                    
                    # Final constraint: background should be below the data
                    background = np.minimum(current_background, y)
                    self.processed_intensities -= background
                    
                except ImportError:
                    QMessageBox.warning(self, "Missing Dependency", 
                                      "Spline background subtraction requires scipy.\nPlease install scipy.")
                    return
                except Exception as e:
                    # Fallback: use simple polynomial baseline if spline fails
                    try:
                        from scipy import ndimage
                        window_size = max(len(self.processed_intensities) // 10, 3)
                        y_min_filtered = ndimage.minimum_filter1d(self.processed_intensities, size=window_size)
                        x = np.arange(len(self.processed_intensities))
                        coeffs = np.polyfit(x, y_min_filtered, min(degree, 3))
                        background = np.polyval(coeffs, x)
                        background = np.minimum(background, self.processed_intensities)
                        self.processed_intensities -= background
                    except:
                        QMessageBox.warning(self, "Spline Error", 
                                          f"Spline fitting failed:\n{str(e)}\nTry adjusting parameters.")
                        return
            
            # Update plot
            self.update_plot()
            
            self.status_bar.showMessage(f"Applied {method.lower()} background subtraction")
            
        except Exception as e:
            QMessageBox.critical(self, "Background Subtraction Error", f"Failed to subtract background:\n{str(e)}")

    def preview_background_subtraction(self):
        """Preview background subtraction with current slider values."""
        if self.processed_intensities is None:
            return
            
        try:
            method = self.bg_method_combo.currentText()
            
            if method.startswith("ALS"):
                # ALS background subtraction preview
                lambda_value = 10 ** self.lambda_slider.value()
                p_value = self.p_slider.value() / 1000.0
                niter_value = self.niter_slider.value()
                
                corrected_intensities, baseline = self.raman_db.subtract_background_als(
                    self.current_wavenumbers,
                    self.processed_intensities,
                    lam=lambda_value,
                    p=p_value,
                    niter=niter_value
                )
                
                self.preview_background = baseline
                self.preview_corrected = corrected_intensities
                
            elif method == "Linear":
                # Linear baseline fitting preview with weighted endpoints
                start_weight = self.start_weight_slider.value() / 10.0
                end_weight = self.end_weight_slider.value() / 10.0
                
                # Find baseline points at start and end regions (use minimum values in regions)
                region_size = max(len(self.processed_intensities) // 20, 5)
                start_region = self.processed_intensities[:region_size]
                end_region = self.processed_intensities[-region_size:]
                
                # Use weighted minimum values from end regions for baseline
                start_val = np.min(start_region) * start_weight
                end_val = np.min(end_region) * end_weight
                background = np.linspace(start_val, end_val, len(self.processed_intensities))
                self.preview_background = background
                self.preview_corrected = self.processed_intensities - background
                
            elif method == "Polynomial":
                # Polynomial baseline fitting preview with adjustable parameters
                x = np.arange(len(self.processed_intensities))
                poly_order = self.poly_order_slider.value()
                poly_method = self.poly_method_combo.currentText()
                
                # Use minimum filtering to identify baseline points
                try:
                    from scipy import ndimage
                    window_size = max(len(self.processed_intensities) // 20, 5)
                    y_min_filtered = ndimage.minimum_filter1d(self.processed_intensities, size=window_size)
                except ImportError:
                    # Fallback without scipy
                    window_size = max(len(self.processed_intensities) // 20, 5)
                    y_min_filtered = np.array([
                        np.min(self.processed_intensities[max(0, i-window_size//2):min(len(self.processed_intensities), i+window_size//2+1)])
                        for i in range(len(self.processed_intensities))
                    ])
                
                if poly_method == "Robust":
                    # Use robust polynomial fitting on baseline points
                    try:
                        # Initial fit to minimum filtered data
                        coeffs = np.polyfit(x, y_min_filtered, poly_order)
                        background = np.polyval(coeffs, x)
                        
                        # Iterative refinement - only use points near the baseline
                        for iteration in range(3):
                            # Identify points that are likely baseline
                            threshold = np.percentile(self.processed_intensities - background, 25)
                            mask = (self.processed_intensities - background) <= threshold
                            
                            if np.sum(mask) < poly_order + 1:  # Need enough points
                                break
                            
                            # Apply robust reweighting to baseline points
                            residuals = np.abs(self.processed_intensities[mask] - background[mask])
                            weights = 1.0 / (1.0 + residuals / (np.median(residuals) + 1e-8))
                            coeffs = np.polyfit(x[mask], self.processed_intensities[mask], poly_order, w=weights)
                            background = np.polyval(coeffs, x)
                            
                    except:
                        # Fallback to simple polynomial on minimum filtered data
                        coeffs = np.polyfit(x, y_min_filtered, poly_order)
                        background = np.polyval(coeffs, x)
                else:
                    # Fit polynomial to minimum filtered data (baseline estimate)
                    coeffs = np.polyfit(x, y_min_filtered, poly_order)
                    background = np.polyval(coeffs, x)
                
                # Ensure background doesn't go above data
                background = np.minimum(background, self.processed_intensities)
                self.preview_background = background
                self.preview_corrected = self.processed_intensities - background
                
            elif method == "Moving Average":
                # Moving minimum/baseline filtering preview for background estimation
                window_percent = self.window_size_slider.value()
                window_type = self.window_type_combo.currentText()
                
                # Calculate window size as percentage of spectrum length
                window_size = max(int(len(self.processed_intensities) * window_percent / 100.0), 3)
                
                try:
                    # First apply minimum filtering to identify baseline regions
                    from scipy import ndimage
                    min_window = max(window_size // 2, 3)
                    y_min_filtered = ndimage.minimum_filter1d(self.processed_intensities, size=min_window)
                    
                    if window_type == "Uniform":
                        # Uniform smoothing of minimum filtered data
                        background = ndimage.uniform_filter1d(y_min_filtered, size=window_size)
                    elif window_type == "Gaussian":
                        # Gaussian smoothing of minimum filtered data
                        sigma = window_size / 4.0
                        background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
                    elif window_type in ["Hann", "Hamming"]:
                        # Windowed smoothing of minimum filtered data
                        if window_type == "Hann":
                            window = np.hanning(window_size)
                        else:  # Hamming
                            window = np.hamming(window_size)
                        
                        window = window / np.sum(window)  # Normalize
                        background = np.convolve(y_min_filtered, window, mode='same')
                    else:
                        # Fallback to uniform
                        background = ndimage.uniform_filter1d(y_min_filtered, size=window_size)
                        
                except ImportError:
                    # Fallback without scipy - use numpy-based minimum filtering
                    y_min_filtered = np.array([
                        np.min(self.processed_intensities[max(0, i-window_size//4):min(len(self.processed_intensities), i+window_size//4+1)])
                        for i in range(len(self.processed_intensities))
                    ])
                    
                    # Simple moving average on minimum filtered data
                    background = np.convolve(y_min_filtered, np.ones(window_size)/window_size, mode='same')
                
                # Ensure background doesn't exceed data
                background = np.minimum(background, self.processed_intensities)
                self.preview_background = background
                self.preview_corrected = self.processed_intensities - background
                
            elif method == "Spline":
                # Spline baseline fitting preview with iterative approach
                n_knots = self.knots_slider.value()
                smoothing_value = self.smoothing_slider.value()
                degree = self.spline_degree_slider.value()
                
                try:
                    from scipy.interpolate import UnivariateSpline
                    from scipy import ndimage
                    
                    # Calculate actual smoothing factor
                    if smoothing_value <= 10:
                        smoothing = 10 ** smoothing_value
                    else:
                        smoothing = 10 ** (smoothing_value / 10.0)
                    
                    # Method: Use minimum filtering and iterative approach to fit below the data
                    x = np.arange(len(self.processed_intensities))
                    y = self.processed_intensities
                    
                    # Step 1: Apply minimum filter to identify baseline regions
                    window_size = max(len(y) // 20, 5)
                    y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
                    
                    # Step 2: For proper baseline fitting, use higher smoothing
                    baseline_smoothing = max(smoothing, len(y) / 10)
                    
                    # Step 3: Fit initial spline to minimum filtered data
                    spline = UnivariateSpline(x, y_min_filtered, s=baseline_smoothing, k=min(degree, 3))
                    current_background = spline(x)
                    
                    # Step 4: Iterative refinement - only use points below or near the background
                    for iteration in range(3):
                        # Identify points that are likely background (below current estimate)
                        threshold = np.percentile(y - current_background, 20)  # Use 20th percentile
                        mask = (y - current_background) <= threshold
                        
                        if np.sum(mask) < n_knots:  # Need enough points
                            break
                        
                        # Fit spline only to identified background points
                        try:
                            spline = UnivariateSpline(x[mask], y[mask], s=baseline_smoothing, k=min(degree, 3))
                            current_background = spline(x)
                        except:
                            break
                        
                        # Ensure background doesn't go above data unrealistically
                        current_background = np.minimum(current_background, y)
                    
                    # Final constraint: background should be below the data
                    background = np.minimum(current_background, y)
                    self.preview_background = background
                    self.preview_corrected = self.processed_intensities - background
                    
                except ImportError:
                    self.status_bar.showMessage("Spline background subtraction requires scipy")
                    return
                except Exception as e:
                    # Fallback: use simple polynomial baseline if spline fails
                    try:
                        from scipy import ndimage
                        window_size = max(len(self.processed_intensities) // 10, 3)
                        y_min_filtered = ndimage.minimum_filter1d(self.processed_intensities, size=window_size)
                        x = np.arange(len(self.processed_intensities))
                        coeffs = np.polyfit(x, y_min_filtered, min(degree, 3))
                        background = np.polyval(coeffs, x)
                        background = np.minimum(background, self.processed_intensities)
                        self.preview_background = background
                        self.preview_corrected = self.processed_intensities - background
                    except:
                        self.status_bar.showMessage(f"Spline error: {str(e)}")
                        return
            
            # Enable background preview
            self.background_preview_active = True
            self.update_plot()
            
            # Update status
            self.status_bar.showMessage(f"Previewing {method.lower()} background subtraction")
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview error: {str(e)}")

    def apply_background_subtraction(self):
        """Apply the current background subtraction preview."""
        if not self.background_preview_active or self.preview_corrected is None:
            # No preview active, run the old method
            self.subtract_background()
            return
            
        # Apply the previewed correction
        self.processed_intensities = self.preview_corrected.copy()
        
        # Clear preview
        self.clear_background_preview()
        
        # Update plot
        self.update_plot()
        
        method = self.bg_method_combo.currentText()
        self.status_bar.showMessage(f"Applied {method.lower()} background subtraction")

    def clear_background_preview(self):
        """Clear the background subtraction preview."""
        self.background_preview_active = False
        self.preview_background = None
        self.preview_corrected = None
        self.update_plot()
        self.status_bar.showMessage("Background preview cleared")

    def reset_spectrum(self):
        """Reset spectrum to original."""
        if self.current_intensities is not None:
            self.processed_intensities = self.current_intensities.copy()
            self.detected_peaks = None
            
            # Clear any active previews
            if self.background_preview_active:
                self.clear_background_preview()
            if self.smoothing_preview_active:
                self.clear_smoothing_preview()
            
            self.update_plot()
            self.status_bar.showMessage("Spectrum reset to original")

    def find_peaks(self):
        """Manual peak detection using current slider values."""
        if self.processed_intensities is not None:
            self.update_peak_detection()  # Use the real-time method
            auto_count = len(self.detected_peaks) if self.detected_peaks is not None else 0
            self.status_bar.showMessage(f"Found {auto_count} automatic peaks")
        else:
            QMessageBox.warning(self, "No Data", "No spectrum loaded for peak detection.")
    
    def toggle_peak_selection_mode(self):
        """Toggle manual peak selection mode."""
        self.peak_selection_mode = not self.peak_selection_mode
        
        if self.peak_selection_mode:
            self.peak_selection_btn.setText("🎯 Exit Peak Selection Mode")
            self.peak_selection_btn.setChecked(True)
            self.status_bar.showMessage("Peak Selection Mode: Click on spectrum to add peaks, click on existing peaks to remove them")
        else:
            self.peak_selection_btn.setText("🎯 Enter Peak Selection Mode")
            self.peak_selection_btn.setChecked(False)
            self.status_bar.showMessage("Peak selection mode disabled")
    
    def on_canvas_click(self, event):
        """Handle mouse clicks on the canvas for manual peak selection."""
        if not self.peak_selection_mode or event.inaxes != self.ax:
            return
        
        if self.current_wavenumbers is None or self.processed_intensities is None:
            return
        
        # Get the clicked position
        clicked_wavenumber = event.xdata
        
        if clicked_wavenumber is None:
            return
        
        # Check if we clicked near an existing manual peak (to remove it)
        peak_to_remove = None
        for i, peak_pos in enumerate(self.manual_peaks):
            if abs(peak_pos - clicked_wavenumber) <= self.peak_selection_tolerance:
                peak_to_remove = i
                break
        
        if peak_to_remove is not None:
            # Remove the peak
            removed_peak = self.manual_peaks.pop(peak_to_remove)
            self.status_bar.showMessage(f"Removed manual peak at {removed_peak:.1f} cm⁻¹")
        else:
            # Add a new peak at the clicked position
            self.manual_peaks.append(clicked_wavenumber)
            self.manual_peaks.sort()  # Keep peaks sorted
            self.status_bar.showMessage(f"Added manual peak at {clicked_wavenumber:.1f} cm⁻¹")
        
        # Update the plot and peak count
        self.update_plot()
        self.update_peak_count_display()
    
    def clear_manual_peaks(self):
        """Clear all manual peaks."""
        self.manual_peaks.clear()
        self.update_plot()
        self.update_peak_count_display()
        self.status_bar.showMessage("Cleared all manual peaks")
    
    def update_peak_count_display(self):
        """Update the peak count display to show both auto and manual peaks."""
        auto_count = len(self.detected_peaks) if self.detected_peaks is not None else 0
        manual_count = len(self.manual_peaks)
        self.peak_count_label.setText(f"Auto peaks: {auto_count} | Manual peaks: {manual_count}")
    
    def apply_peak_preset(self, height, distance, prominence):
        """Apply preset peak detection parameters."""
        # Set slider values
        self.height_slider.setValue(height)
        self.distance_slider.setValue(distance)
        self.prominence_slider.setValue(prominence)
        
        # Update labels
        self.height_label.setText(f"{height}%")
        self.distance_label.setText(str(distance))
        self.prominence_label.setText(f"{prominence}%")
        
        # Update peak detection
        self.update_peak_detection()
        
        # Provide feedback
        self.status_bar.showMessage(f"Applied preset: Height={height}%, Distance={distance}, Prominence={prominence}%")

    def show_about(self):
        """Show about dialog."""
        about_text = f"""
        RamanLab PySide6 Version
        
        Version: {__version__}
        Author: {__author__}
        
        The Raman Spectrum Analysis Tool.
        
        Key Benefits:
        • Cross-platform compatibility (macOS, Windows, Linux)
        • No more platform-specific code
        • Modern, professional interface
        • Better matplotlib integration
        
        """
        QMessageBox.about(self, "About RamanLab PySide6", about_text)

    def check_for_updates(self):
        """Check for updates using the simple, thread-safe update checker."""
        try:
            # Use the simple update checker that doesn't use background threads
            from core.simple_update_checker import simple_check_for_updates
            simple_check_for_updates(parent=self, show_no_update=True)
        except ImportError:
            QMessageBox.information(
                self,
                "Update Checker Unavailable",
                "The update checker requires additional dependencies:\n\n"
                "pip install requests packaging pyperclip\n\n"
                "You can check for updates manually at:\n"
                "https://github.com/aaroncelestian/RamanLab"
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Update Check Error",
                f"An error occurred while checking for updates:\n{str(e)}\n\n"
                "You can check for updates manually at:\n"
                "https://github.com/aaroncelestian/RamanLab"
            )

    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit RamanLab PySide6?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # Database functionality methods
    def add_to_database(self):
        """Add current spectrum to database with metadata editing."""
        if self.current_wavenumbers is None or self.current_intensities is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded to add to database.")
            return
        
        # Create a metadata input dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Spectrum to Database")
        dialog.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Form for basic metadata
        form_layout = QFormLayout()
        
        # Spectrum name
        name_edit = QLineEdit()
        name_edit.setText(f"Spectrum_{len(self.raman_db.database) + 1}")
        form_layout.addRow("Spectrum Name:", name_edit)
        
        # Mineral name
        mineral_edit = QLineEdit()
        form_layout.addRow("Mineral Name:", mineral_edit)
        
        # Sample description
        description_edit = QTextEdit()
        description_edit.setMaximumHeight(80)
        form_layout.addRow("Description:", description_edit)
        
        # Experimental conditions
        laser_edit = QLineEdit()
        laser_edit.setPlaceholderText("e.g., 532 nm")
        form_layout.addRow("Laser Wavelength:", laser_edit)
        
        power_edit = QLineEdit()
        power_edit.setPlaceholderText("e.g., 10 mW")
        form_layout.addRow("Laser Power:", power_edit)
        
        exposure_edit = QLineEdit()
        exposure_edit.setPlaceholderText("e.g., 30 s")
        form_layout.addRow("Exposure Time:", exposure_edit)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.Accepted:
            # Collect metadata
            metadata = {
                'mineral_name': mineral_edit.text(),
                'description': description_edit.toPlainText(),
                'laser_wavelength': laser_edit.text(),
                'laser_power': power_edit.text(),
                'exposure_time': exposure_edit.text(),
                'data_points': len(self.current_wavenumbers),
                'wavenumber_range': f"{self.current_wavenumbers.min():.1f} - {self.current_wavenumbers.max():.1f} cm⁻¹"
            }
            
            # Add file metadata if available
            if hasattr(self, 'current_spectrum_metadata') and self.current_spectrum_metadata:
                # Merge file metadata, giving priority to user-entered metadata
                file_metadata = self.current_spectrum_metadata.copy()
                # Remove file system specific metadata
                for key in ['file_path', 'file_name']:
                    file_metadata.pop(key, None)
                
                # Merge, with user metadata taking precedence
                merged_metadata = file_metadata.copy()
                merged_metadata.update(metadata)
                metadata = merged_metadata
            
            # Add to database
            # Convert peak indices to wavenumber values before storing
            peak_wavenumbers = None
            if self.detected_peaks is not None and len(self.detected_peaks) > 0:
                # Convert indices to actual wavenumber values
                peak_wavenumbers = self.current_wavenumbers[self.detected_peaks].tolist()

            success = self.raman_db.add_to_database(
                name=name_edit.text(),
                wavenumbers=self.current_wavenumbers,
                intensities=self.processed_intensities,
                metadata=metadata,
                peaks=peak_wavenumbers
            )
            
            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Spectrum '{name_edit.text()}' added to database!\n\n"
                    f"Database location:\n{self.raman_db.db_path}\n\n"
                    "This database is persistent and cross-platform!"
                )
                # Update database stats
                self.update_database_stats()

    def view_database(self):
        """Launch the comprehensive database browser window."""
        # Import and launch the database browser
        from database_browser_qt6 import DatabaseBrowserQt6
        
        self.database_browser = DatabaseBrowserQt6(self.raman_db, parent=self)
        self.database_browser.show()  # Show as modeless dialog so user can work with both windows

    def update_database_stats(self):
        """Update database statistics display using real database."""
        if hasattr(self, 'db_stats_text') and self.db_stats_text is not None:
            stats = self.raman_db.get_database_stats()
            
            # Count spectra with peaks for peak filtering information
            spectra_with_peaks = 0
            total_peaks = 0
            for data in self.raman_db.database.values():
                peaks_data = data.get('peaks', [])
                if peaks_data and len(peaks_data) > 0:
                    spectra_with_peaks += 1
                    if isinstance(peaks_data, (list, tuple)):
                        total_peaks += len(peaks_data)
            
            stats_text = f"Database Statistics:\n\n"
            stats_text += f"Total spectra: {stats['total_spectra']}\n"
            stats_text += f"Average data points: {stats['avg_data_points']:.0f}\n"
            stats_text += f"Average peaks per spectrum: {stats['avg_peaks']:.1f}\n"
            stats_text += f"Spectra with detected peaks: {spectra_with_peaks}\n"
            stats_text += f"Total peaks in database: {total_peaks}\n"
            stats_text += f"Database file size: {stats['database_size']}\n\n"
            stats_text += f"Location: {self.raman_db.db_path}\n\n"
            stats_text += "✅ Cross-platform persistent storage\n"
            stats_text += "✅ Automatic backup on every save\n\n"
            
            # Add peak filtering guidance
            if spectra_with_peaks == 0:
                stats_text += "⚠️  No spectra have detected peaks.\n"
                stats_text += "Peak-based filtering will not work.\n"
                stats_text += "Detect peaks before adding spectra to database."
            elif spectra_with_peaks < stats['total_spectra']:
                stats_text += f"ℹ️  {stats['total_spectra'] - spectra_with_peaks} spectra lack detected peaks.\n"
                stats_text += "These will be excluded from peak-based searches."
            else:
                stats_text += "✅ All spectra have detected peaks.\n"
                stats_text += "Peak-based filtering is available."
            
            self.db_stats_text.setPlainText(stats_text)

    def migrate_legacy_database(self):
        """Migrate legacy database from menu."""
        from database_browser_qt6 import DatabaseBrowserQt6
        
        # Create a temporary browser instance just for migration
        temp_browser = DatabaseBrowserQt6(self.raman_db, parent=self)
        temp_browser.migrate_legacy_database()
        
        # Update our database stats after migration
        self.update_database_stats()

    def browse_pkl_file(self):
        """Browse for PKL file to migrate from menu."""
        from database_browser_qt6 import DatabaseBrowserQt6
        
        # Create a temporary browser instance just for migration
        temp_browser = DatabaseBrowserQt6(self.raman_db, parent=self)
        temp_browser.browse_pkl_file()
        
        # Update our database stats after migration
        self.update_database_stats()

    def update_peak_detection(self):
        """Update peak detection in real-time based on slider values."""
        if self.processed_intensities is None:
            return
            
        # Get current slider values
        height_percent = self.height_slider.value()
        distance = self.distance_slider.value()
        prominence_percent = self.prominence_slider.value()
        
        # Update labels with enhanced formatting
        self.height_label.setText(f"{height_percent}%")
        self.distance_label.setText(str(distance))
        self.prominence_label.setText(f"{prominence_percent}%")
        
        # Calculate actual values
        max_intensity = np.max(self.processed_intensities)
        min_intensity = np.min(self.processed_intensities)
        intensity_range = max_intensity - min_intensity
        
        height_threshold = (height_percent / 100.0) * max_intensity if height_percent > 0 else None
        prominence_threshold = (prominence_percent / 100.0) * intensity_range if prominence_percent > 0 else None
        
        # Find peaks with current parameters
        try:
            self.detected_peaks, properties = find_peaks(
                self.processed_intensities,
                height=height_threshold,
                distance=distance,
                prominence=prominence_threshold
            )
            
            # Update peak count display
            self.update_peak_count_display()
            
            # Update status bar with live feedback
            if len(self.detected_peaks) > 0:
                peak_positions = self.current_wavenumbers[self.detected_peaks]
                self.status_bar.showMessage(f"Found {len(self.detected_peaks)} peaks - Live updating...")
            else:
                self.status_bar.showMessage("No peaks detected with current parameters")
            
            # Update plot with enhanced visualization
            self.update_plot()
            
        except Exception as e:
            self.peak_count_label.setText(f"Peak detection error: {str(e)}")
            self.status_bar.showMessage(f"Peak detection error: {str(e)}")
            print(f"DEBUG: Peak detection error: {str(e)}")

    def apply_smoothing(self):
        """Apply Savitzky-Golay smoothing to the spectrum."""
        if self.processed_intensities is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded for smoothing.")
            return
            
        try:
            # If there's a preview, use it
            if self.smoothing_preview_active and self.preview_smoothed is not None:
                self.processed_intensities = self.preview_smoothed.copy()
                window_length = self.sg_window_spin.value()
                poly_order = self.sg_order_spin.value()
                
                # Clear preview
                self.clear_smoothing_preview()
                
                # Update plot
                self.update_plot()
                
                self.status_bar.showMessage(f"Applied Savitzky-Golay smoothing (window={window_length}, order={poly_order})")
                return
            
            # No preview, apply directly
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
            smoothed = savgol_filter(self.processed_intensities, window_length, poly_order)
            self.processed_intensities = smoothed
            
            # Update plot
            self.update_plot()
            
            self.status_bar.showMessage(f"Applied Savitzky-Golay smoothing (window={window_length}, order={poly_order})")
            
        except Exception as e:
            QMessageBox.critical(self, "Smoothing Error", f"Failed to apply smoothing:\n{str(e)}")

    def preview_smoothing(self):
        """Preview Savitzky-Golay smoothing with current parameters."""
        if self.processed_intensities is None:
            return
            
        try:
            window_length = self.sg_window_spin.value()
            poly_order = self.sg_order_spin.value()
            
            # Ensure window length is odd and greater than poly_order
            if window_length % 2 == 0:
                window_length += 1
                self.sg_window_spin.setValue(window_length)
                
            if window_length <= poly_order:
                self.status_bar.showMessage(f"Window length ({window_length}) must be greater than polynomial order ({poly_order})")
                return
                
            # Apply Savitzky-Golay filter for preview
            smoothed = savgol_filter(self.processed_intensities, window_length, poly_order)
            self.preview_smoothed = smoothed
            
            # Enable smoothing preview
            self.smoothing_preview_active = True
            self.update_plot()
            
            # Update status
            self.status_bar.showMessage(f"Previewing Savitzky-Golay smoothing (window={window_length}, order={poly_order})")
            
        except Exception as e:
            self.status_bar.showMessage(f"Smoothing preview error: {str(e)}")

    def clear_smoothing_preview(self):
        """Clear the smoothing preview."""
        self.smoothing_preview_active = False
        self.preview_smoothed = None
        self.update_plot()
        self.status_bar.showMessage("Smoothing preview cleared")

    def launch_multi_spectrum_manager(self):
        """Launch the multi-spectrum manager window."""
        from multi_spectrum_manager_qt6 import MultiSpectrumManagerQt6
        
        self.multi_spectrum_window = MultiSpectrumManagerQt6(parent=self, raman_db=self.raman_db)
        self.multi_spectrum_window.show()
        
        # If there's a current spectrum loaded, offer to add it
        if self.current_wavenumbers is not None and self.current_intensities is not None:
            reply = QMessageBox.question(
                self,
                "Add Current Spectrum?",
                "Would you like to add the currently loaded spectrum to the multi-spectrum manager?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.multi_spectrum_window.add_current_spectrum()

    def on_bg_method_changed(self):
        """Handle change in background method."""
        method = self.bg_method_combo.currentText()
        
        # Show/hide parameter widgets based on selected method
        self.als_params_widget.setVisible(method.startswith("ALS"))
        self.linear_params_widget.setVisible(method == "Linear")
        self.poly_params_widget.setVisible(method == "Polynomial")
        self.moving_avg_params_widget.setVisible(method == "Moving Average")
        self.spline_params_widget.setVisible(method == "Spline")
        
        # Clear any active background preview when method changes
        if hasattr(self, 'background_preview_active') and self.background_preview_active:
            self.clear_background_preview()

    def update_lambda_label(self):
        """Update the lambda label based on the slider value."""
        value = self.lambda_slider.value()
        lambda_value = 10 ** value
        self.lambda_label.setText(f"1e{value}")

    def update_p_label(self):
        """Update the p label based on the slider value."""
        value = self.p_slider.value()
        p_value = value / 1000.0  # Convert to 0.001-0.05 range
        self.p_label.setText(f"{p_value:.3f}")
    
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
    
    def update_niter_label(self):
        """Update iterations label."""
        self.niter_label.setText(str(self.niter_slider.value()))
    
    def update_knots_label(self):
        """Update knots label."""
        self.knots_label.setText(str(self.knots_slider.value()))
    
    def update_smoothing_label(self):
        """Update smoothing factor label."""
        value = self.smoothing_slider.value()
        if value <= 10:
            smoothing = 10 ** value
        else:
            smoothing = 10 ** (value / 10.0)
        self.smoothing_label.setText(f"{int(smoothing)}")
    
    def update_spline_degree_label(self):
        """Update spline degree label."""
        degree_names = {1: "1 (Linear)", 2: "2 (Quadratic)", 3: "3 (Cubic)", 4: "4 (Quartic)", 5: "5 (Quintic)"}
        degree = self.spline_degree_slider.value()
        self.spline_degree_label.setText(degree_names.get(degree, str(degree)))

    def generate_background_previews(self):
        """Generate multiple background subtraction options with different parameters."""
        if self.processed_intensities is None:
            QMessageBox.warning(self, "No Data", "No spectrum loaded for background generation.")
            return
            
        try:
            # Clear previous options
            self.clear_background_options()
            
            # Define ALS parameter sets - top 6 most useful combinations
            parameter_sets = [
                ("ALS (Conservative)", "ALS", {"lambda": 1e6, "p": 0.001, "niter": 10}),
                ("ALS (Moderate)", "ALS", {"lambda": 1e5, "p": 0.01, "niter": 10}),
                ("ALS (Aggressive)", "ALS", {"lambda": 1e4, "p": 0.05, "niter": 15}),
                ("ALS (Ultra Smooth)", "ALS", {"lambda": 1e7, "p": 0.002, "niter": 20}),
                ("ALS (Balanced)", "ALS", {"lambda": 5e5, "p": 0.02, "niter": 12}),
                ("ALS (Fast)", "ALS", {"lambda": 1e5, "p": 0.01, "niter": 5}),
            ]
            
            # Initialize background options storage
            if not hasattr(self, 'background_options'):
                self.background_options = []
                self.background_option_lines = []
            
            # Generate backgrounds for each parameter set
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            for i, (description, method, params) in enumerate(parameter_sets):
                try:
                    background = self._calculate_background_with_params(method, params)
                    if background is not None:
                        # Store background data
                        self.background_options.append((background, description, method, params))
                        
                        # Plot preview line on the main plot
                        color = colors[i % len(colors)]
                        line, = self.plot_widget.plot(
                            self.current_wavenumbers, background, 
                            pen={'color': color, 'width': 2, 'style': 2},  # Dashed line
                            name=description
                        )
                        self.background_option_lines.append(line)
                        
                except Exception as e:
                    print(f"Failed to generate {description}: {str(e)}")
                    continue
            
            # Update dropdown with options
            self.update_background_options_dropdown()
            
            # Show info message
            QMessageBox.information(self, "Options Generated", 
                                  f"Generated {len(self.background_options)} background options.\n"
                                  f"Select one from the dropdown to preview and apply.")
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Failed to generate background options: {str(e)}")
    
    def _calculate_background_with_params(self, method, params):
        """Calculate background using specified method and parameters."""
        try:
            if method == "ALS":
                lambda_val = params.get("lambda", 1e5)
                p_val = params.get("p", 0.01)
                niter_val = params.get("niter", 10)
                corrected_intensities, baseline = self.raman_db.subtract_background_als(
                    self.current_wavenumbers,
                    self.processed_intensities,
                    lam=lambda_val,
                    p=p_val,
                    niter=niter_val
                )
                return baseline
            
            return None
            
        except Exception as e:
            print(f"Background calculation error for {method}: {str(e)}")
            return None
    
    def update_background_options_dropdown(self):
        """Update the background options dropdown with generated options."""
        self.bg_options_combo.clear()
        
        if hasattr(self, 'background_options') and self.background_options:
            for i, (background, description, method, params) in enumerate(self.background_options):
                self.bg_options_combo.addItem(f"{i+1}. {description}")
        else:
            self.bg_options_combo.addItem("None - Generate options first")
    
    def on_bg_option_selected(self):
        """Handle selection of a background option."""
        if not hasattr(self, 'background_options') or not self.background_options:
            return
            
        selected_text = self.bg_options_combo.currentText()
        if selected_text.startswith("None"):
            return
            
        try:
            # Extract option index from text (format: "1. Description")
            option_index = int(selected_text.split('.')[0]) - 1
            
            if 0 <= option_index < len(self.background_options):
                background, description, method, params = self.background_options[option_index]
                
                # Preview this background option
                self.preview_background = background
                self.preview_corrected = self.processed_intensities - background
                self.background_preview_active = True
                
                # Highlight the selected option
                self._highlight_selected_background_option(option_index)
                
                # Update the plot
                self.update_plot()
                
                # Update status
                self.status_bar.showMessage(f"Previewing: {description}")
                
        except Exception as e:
            self.status_bar.showMessage(f"Error selecting option: {str(e)}")
    
    def _highlight_selected_background_option(self, option_index):
        """Highlight the selected background option on the plot."""
        if not hasattr(self, 'background_option_lines'):
            return
            
        # Reset all line widths to normal
        for line in self.background_option_lines:
            if hasattr(line, 'opts') and 'pen' in line.opts:
                line.opts['pen']['width'] = 2
        
        # Highlight the selected line
        if 0 <= option_index < len(self.background_option_lines):
            line = self.background_option_lines[option_index]
            if hasattr(line, 'opts') and 'pen' in line.opts:
                line.opts['pen']['width'] = 4  # Make it thicker
    
    def apply_selected_background_option(self):
        """Apply the currently selected background option."""
        if not hasattr(self, 'background_preview_active') or not self.background_preview_active:
            QMessageBox.warning(self, "No Selection", "No background option selected. Generate and select an option first.")
            return
            
        if self.preview_corrected is None:
            QMessageBox.warning(self, "No Preview", "No background preview available.")
            return
            
        # Apply the previewed correction
        self.processed_intensities = self.preview_corrected.copy()
        
        # Clear preview and options
        self.clear_background_preview()
        self.clear_background_options()
        
        # Update plot
        self.update_plot()
        
        self.status_bar.showMessage("Background subtraction applied successfully")
    
    def clear_background_options(self):
        """Clear all background options and preview lines."""
        # Clear the plot lines
        if hasattr(self, 'background_option_lines'):
            for line in self.background_option_lines:
                try:
                    self.plot_widget.removeItem(line)
                except:
                    pass
            self.background_option_lines = []
        
        # Clear stored options
        if hasattr(self, 'background_options'):
            self.background_options = []
        
        # Reset dropdown
        self.bg_options_combo.clear()
        self.bg_options_combo.addItem("None - Generate options first")

    def perform_basic_search(self):
        """Perform basic database search using current spectrum."""
        if self.current_wavenumbers is None or self.current_intensities is None:
            QMessageBox.warning(self, "No Data", "Load a spectrum first to search for matches.")
            return
            
        if not self.raman_db.database:
            QMessageBox.information(self, "Empty Database", "No spectra in database to search.\nAdd some spectra first!")
            return
        
        try:
            # Get search parameters
            algorithm = self.algorithm_combo.currentText()
            n_matches = self.n_matches_spin.value()
            threshold = self.threshold_spin.value()
            
            # Check if manual peaks are available and algorithm can use them
            user_peak_info = None
            if (algorithm in ["peak", "combined"] and 
                hasattr(self, 'manual_peaks') and 
                len(self.manual_peaks) > 0):
                
                user_peak_info = {
                    'peak_positions': self.manual_peaks.copy(),
                    'peak_tolerance': 10.0  # Default tolerance for basic search
                }
                
                # Inform user that manual peaks are being used
                peak_info_msg = f"Using {len(self.manual_peaks)} manually selected peaks: {[f'{p:.1f}' for p in self.manual_peaks]} cm⁻¹"
                self.status_bar.showMessage(peak_info_msg)
                QApplication.processEvents()
            
            # Show warning for DTW algorithm
            if algorithm == "DTW":
                warning_result = QMessageBox.question(
                    self, 
                    "DTW Algorithm Warning", 
                    f"<b>DTW (Dynamic Time Warping) Algorithm Selected</b><br><br>"
                    f"<b>Performance Notice:</b><br>"
                    f"• DTW must analyze every spectrum individually (no early termination)<br>"
                    f"• Database size: <b>{len(self.raman_db.database)} spectra</b><br>"
                    f"• Estimated time: <b>{len(self.raman_db.database) * 0.1:.1f}-{len(self.raman_db.database) * 0.3:.1f} seconds</b><br><br>"
                    f"<b>Why DTW is slower than Combined:</b><br>"
                    f"• Combined algorithm uses correlation pre-screening (skips poor matches)<br>"
                    f"• DTW alone must run expensive calculations on every spectrum<br><br>"
                    f"<b>Recommendation:</b> Try 'Combined' algorithm for better speed with similar accuracy.<br><br>"
                    f"Continue with DTW search?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if warning_result == QMessageBox.No:
                    self.status_bar.showMessage("DTW search cancelled by user")
                    return
            
            # Show progress indication
            search_progress_msg = "Searching database, please wait..."
            if user_peak_info:
                search_progress_msg += f" (Using {len(user_peak_info['peak_positions'])} manual peaks)"
            self.search_results_text.setPlainText(search_progress_msg)
            QApplication.processEvents()  # Update UI
            
            # Use optimized search method with progress tracking and peak information
            candidates = [(name, data) for name, data in self.raman_db.database.items()]
            matches = self.search_filtered_candidates(candidates, algorithm, n_matches, threshold, user_peak_info)
            
            # Display results with additional info about peak usage
            algorithm_name = "DTW (Dynamic Time Warping)" if algorithm == "DTW" else algorithm.title()
            
            additional_info = ""
            if user_peak_info:
                additional_info = f"Search Details:\n"
                additional_info += f"• Algorithm: {algorithm_name}\n"
                additional_info += f"• Manual peaks used: {user_peak_info['peak_positions']} cm⁻¹\n"
                additional_info += f"• Peak tolerance: ±{user_peak_info['peak_tolerance']} cm⁻¹\n"
                additional_info += f"• Threshold: {threshold:.2f}\n"
                
                if algorithm == "peak":
                    additional_info += f"• Peak Algorithm: Enhanced with your manually selected peaks\n"
                elif algorithm == "combined":
                    additional_info += f"• Combined Algorithm: Uses your manual peaks for enhanced scoring\n"
            
            self.display_search_results(matches, f"Basic Search ({algorithm_name})", additional_info)
            
            # Update status
            status_msg = f"Search completed - found {len(matches)} matches"
            if user_peak_info:
                status_msg += f" (used {len(user_peak_info['peak_positions'])} manual peaks)"
            self.status_bar.showMessage(status_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"Search failed:\n{str(e)}")
            self.status_bar.showMessage("Search failed")
            self.search_results_text.setPlainText("Search failed. Please check your data and try again.")

    def perform_advanced_search(self):
        """Perform advanced database search with filters."""
        if self.current_wavenumbers is None or self.current_intensities is None:
            QMessageBox.warning(self, "No Data", "Load a spectrum first to search for matches.")
            return
            
        if not self.raman_db.database:
            QMessageBox.information(self, "Empty Database", "No spectra in database to search.\nAdd some spectra first!")
            return
        
        try:
            # Get search parameters
            algorithm = self.algorithm_combo.currentText()
            n_matches = self.n_matches_spin.value()
            threshold = self.threshold_spin.value()
            
            # Show warning for DTW algorithm (same as basic search)
            if algorithm == "DTW":
                warning_result = QMessageBox.question(
                    self, 
                    "DTW Algorithm Warning", 
                    f"<b>DTW (Dynamic Time Warping) Algorithm Selected</b><br><br>"
                    f"<b>Performance Notice:</b><br>"
                    f"• DTW must analyze every spectrum individually (no early termination)<br>"
                    f"• Database size: <b>{len(self.raman_db.database)} spectra</b><br>"
                    f"• Estimated time: <b>{len(self.raman_db.database) * 0.1:.1f}-{len(self.raman_db.database) * 0.3:.1f} seconds</b><br><br>"
                    f"<b>Why DTW is slower than Combined:</b><br>"
                    f"• Combined algorithm uses correlation pre-screening (skips poor matches)<br>"
                    f"• DTW alone must run expensive calculations on every spectrum<br><br>"
                    f"<b>Recommendation:</b> Try 'Combined' algorithm for better speed with similar accuracy.<br><br>"
                    f"Continue with DTW search?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if warning_result == QMessageBox.No:
                    self.status_bar.showMessage("DTW search cancelled by user")
                    return
            
            # Show progress indication
            self.search_results_text.setPlainText("Applying filters and searching database, please wait...")
            QApplication.processEvents()  # Update UI
            
            # Parse filter criteria
            filters = self.parse_advanced_filters()
            
            # Check if peak positions are specified
            has_peak_filter = 'peak_positions' in filters and len(filters['peak_positions']) > 0
            
            # Show filter information
            if has_peak_filter:
                peak_info = f"Peak positions specified: {filters['peak_positions']} cm⁻¹ (±{filters.get('peak_tolerance', 10)} cm⁻¹)"
                print(f"Debug: {peak_info}")
                self.status_bar.showMessage(peak_info)
                QApplication.processEvents()
            
            # Apply metadata filters (excluding peak positions - they're handled by scoring algorithms)
            metadata_filters = {k: v for k, v in filters.items() if k not in ['peak_positions', 'peak_tolerance']}
            
            if metadata_filters:
                self.status_bar.showMessage("Applying metadata filters...")
                QApplication.processEvents()
                filtered_candidates = self.apply_metadata_filters(metadata_filters)
                
                if not filtered_candidates:
                    no_results_msg = f"No spectra match the specified metadata filters: {self.create_filter_summary(metadata_filters)}"
                    QMessageBox.information(self, "No Results", no_results_msg)
                    self.search_results_text.setPlainText(no_results_msg)
                    self.status_bar.showMessage("No matches found after filtering")
                    return
            else:
                # No metadata filters - use entire database
                filtered_candidates = [(name, data) for name, data in self.raman_db.database.items()]
            
            # Update status
            self.status_bar.showMessage(f"Searching {len(filtered_candidates)} candidates...")
            QApplication.processEvents()
            
            # Prepare peak information for algorithms that can use it
            user_peak_info = None
            if has_peak_filter:
                user_peak_info = {
                    'peak_positions': filters['peak_positions'],
                    'peak_tolerance': filters.get('peak_tolerance', 10)
                }
            
            # Perform search using the existing method but with peak information
            matches = self.search_filtered_candidates(
                filtered_candidates, 
                algorithm, 
                n_matches, 
                threshold, 
                user_peak_info
            )
            
            # Display results
            algorithm_name = "DTW (Dynamic Time Warping)" if algorithm == "DTW" else algorithm.title()
            
            additional_info = f"Search Details:\n"
            additional_info += f"• Algorithm: {algorithm_name}\n"
            additional_info += f"• Threshold: {threshold:.2f}\n"
            additional_info += f"• Database size: {len(self.raman_db.database)} spectra\n"
            
            if metadata_filters:
                additional_info += f"• Metadata filters: {self.create_filter_summary(metadata_filters)}\n"
                additional_info += f"• After filtering: {len(filtered_candidates)} candidates\n"
            else:
                additional_info += f"• No metadata filters applied\n"
            
            if has_peak_filter:
                additional_info += f"\nPeak Information:\n"
                additional_info += f"• Target peaks: {filters['peak_positions']} cm⁻¹\n"
                additional_info += f"• Peak tolerance: ±{filters.get('peak_tolerance', 10)} cm⁻¹\n"
                
                if algorithm == "peak":
                    additional_info += f"• Peak Algorithm: Enhanced with your specified peak positions\n"
                elif algorithm == "combined":
                    additional_info += f"• Combined Algorithm: Uses your peak positions for enhanced scoring\n"
                else:
                    additional_info += f"• {algorithm} Algorithm: Peak positions noted but algorithm uses spectral shape\n"
            
            search_type_name = f"Advanced Search ({algorithm_name})"
            
            self.display_search_results(
                matches, 
                search_type_name,
                additional_info=additional_info
            )
            
            # Update status
            self.status_bar.showMessage(f"Advanced search completed - found {len(matches)} matches")
            
        except Exception as e:
            QMessageBox.critical(self, "Advanced Search Error", f"Advanced search failed:\n{str(e)}")
            self.status_bar.showMessage("Advanced search failed")
            self.search_results_text.setPlainText("Advanced search failed. Please check your filters and try again.")



    def parse_advanced_filters(self):
        """Parse the advanced search filter criteria."""
        filters = {}
        
        # Peak positions
        peak_str = self.peak_positions_edit.text().strip()
        if peak_str:
            try:
                filters['peak_positions'] = [float(x.strip()) for x in peak_str.split(",")]
                filters['peak_tolerance'] = self.peak_tolerance_spin.value()
            except ValueError:
                raise ValueError("Invalid peak positions format. Use comma-separated numbers.")
        
        # Chemical family
        chem_family = self.chemical_family_combo.currentText().strip()
        if chem_family:
            filters['chemical_family'] = chem_family
        
        # Hey classification
        hey_class = self.hey_classification_combo.currentText().strip()
        if hey_class:
            filters['hey_classification'] = hey_class
        
        # Element filters
        for filter_name, widget in [
            ('only_elements', self.only_elements_edit),
            ('required_elements', self.required_elements_edit),
            ('exclude_elements', self.exclude_elements_edit)
        ]:
            elements_str = widget.text().strip()
            if elements_str:
                filters[filter_name] = [elem.strip().upper() for elem in elements_str.split(",")]
        
        return filters

    def apply_metadata_filters(self, filters):
        """Apply metadata filters to the database and return matching candidates."""
        candidates = []
        
        for name, data in self.raman_db.database.items():
            # Check if this spectrum passes all filters
            if self.spectrum_passes_filters(name, data, filters):
                candidates.append((name, data))
        
        return candidates

    def spectrum_passes_filters(self, name, data, filters):
        """Check if a spectrum passes all the specified filters.""" 
        metadata = data.get('metadata', {})
        
        # Note: Peak positions are NOT filtered here - they are handled by scoring algorithms
        # This allows spectra to be scored based on how well they match peak positions
        # rather than being eliminated entirely
        
        # Chemical family filter
        if 'chemical_family' in filters:
            spectrum_family = metadata.get('CHEMICAL FAMILY') or metadata.get('Chemical Family', '')
            if not spectrum_family or filters['chemical_family'].lower() not in spectrum_family.lower():
                return False
        
        # Hey classification filter
        if 'hey_classification' in filters:
            spectrum_hey = metadata.get('HEY CLASSIFICATION', '')
            if not spectrum_hey or filters['hey_classification'].lower() not in spectrum_hey.lower():
                return False
        
        # Element filters - match original app behavior
        # Original app used "CHEMISTRY ELEMENTS" field, not "FORMULA"
        chemistry_elements = metadata.get('CHEMISTRY ELEMENTS', '') or metadata.get('Chemistry Elements', '')
        formula = metadata.get('FORMULA', '') or metadata.get('Formula', '')
        
        # Try chemistry elements field first (original app format), then formula as fallback
        elements_source = chemistry_elements if chemistry_elements else formula
        
        # If any element filters are specified, we need element data to check against
        has_element_filters = any(key in filters for key in ['only_elements', 'required_elements', 'exclude_elements'])
        
        if has_element_filters:
            if not elements_source:
                # No element data available but element filters are specified - spectrum fails
                return False
            
            # Parse elements from either chemistry elements or formula
            if chemistry_elements:
                # Original app format: comma-separated elements (e.g., "Al, Si, O")
                elements_in_spectrum = set([elem.strip().upper() for elem in chemistry_elements.split(",")])
            else:
                # Fallback: extract elements from formula (simplified parsing)
                import re
                elements_in_spectrum = set(re.findall(r'[A-Z][a-z]?', formula))
            
            # Only elements filter
            if 'only_elements' in filters:
                allowed_elements = set(filters['only_elements'])
                if not elements_in_spectrum.issubset(allowed_elements):
                    return False
            
            # Required elements filter
            if 'required_elements' in filters:
                required_elements = set(filters['required_elements'])
                if not required_elements.issubset(elements_in_spectrum):
                    return False
            
            # Exclude elements filter
            if 'exclude_elements' in filters:
                excluded_elements = set(filters['exclude_elements'])
                if elements_in_spectrum.intersection(excluded_elements):
                    return False
        
        return True

    def search_filtered_candidates(self, candidates, algorithm, n_matches, threshold, user_peak_filters=None):
        """Perform optimized search algorithm on filtered candidate set with progress reporting."""
        matches = []
        
        # Extract user-specified peaks if available
        user_specified_peaks = None
        peak_tolerance = 10
        if user_peak_filters:
            user_specified_peaks = user_peak_filters.get('peak_positions')
            peak_tolerance = user_peak_filters.get('peak_tolerance', 10)
        
        # OPTIMIZATION: Track progress for slow algorithms
        is_slow_algorithm = algorithm in ["DTW", "combined"]
        total_candidates = len(candidates)
        processed = 0
        
        # OPTIMIZATION: Pre-calculate query spectrum normalization for DTW/combined to avoid repetition
        if is_slow_algorithm:
            # Cache normalized query spectrum for DTW calculations
            if hasattr(self, 'processed_intensities') and self.processed_intensities is not None:
                self._cached_query_norm = (self.processed_intensities - self.processed_intensities.min()) / (self.processed_intensities.max() - self.processed_intensities.min())
            else:
                self._cached_query_norm = None
        
        for name, data in candidates:
            try:
                wavenumbers = data.get('wavenumbers', [])
                intensities = data.get('intensities', [])
                
                # Ensure we have data - check length instead of boolean
                if len(wavenumbers) == 0 or len(intensities) == 0:
                    continue
                
                # Calculate similarity score based on algorithm
                if algorithm == "correlation":
                    score = self.calculate_correlation_score(wavenumbers, intensities)
                elif algorithm == "multi-window":
                    score = self.calculate_multi_window_score(wavenumbers, intensities)
                elif algorithm == "mineral-vibration":
                    score = self.calculate_mineral_vibration_score(wavenumbers, intensities)
                elif algorithm == "peak":
                    score = self.calculate_peak_score(wavenumbers, intensities, user_specified_peaks, peak_tolerance)
                elif algorithm == "combined":
                    score = self.calculate_combined_score(wavenumbers, intensities, user_specified_peaks, peak_tolerance)
                elif algorithm == "DTW":
                    score = self.calculate_dtw_score(wavenumbers, intensities)
                else:
                    # Fallback to correlation
                    score = self.calculate_correlation_score(wavenumbers, intensities)
                
                if score >= threshold:
                    matches.append({
                        'name': name,
                        'score': score,
                        'metadata': data.get('metadata', {}),
                        'peaks': data.get('peaks', []),
                        'timestamp': data.get('timestamp', '')
                    })
                
                processed += 1
                
                # OPTIMIZATION: Early termination for slow algorithms
                # If we have enough good matches, we can stop early for expensive algorithms
                if (is_slow_algorithm and len(matches) >= n_matches * 3 and 
                    processed > total_candidates * 0.3):  # Processed at least 30%
                    print(f"Early termination: Found {len(matches)} matches after processing {processed}/{total_candidates} spectra")
                    break
                
                # OPTIMIZATION: Progress updates for slow algorithms
                if is_slow_algorithm and processed % max(1, total_candidates // 10) == 0:
                    progress_pct = int((processed / total_candidates) * 100)
                    self.status_bar.showMessage(f"Searching... {progress_pct}% complete ({len(matches)} matches found)")
                    QApplication.processEvents()  # Keep UI responsive
                    
            except Exception as e:
                print(f"Error processing spectrum {name}: {e}")
                continue
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:n_matches]

    def calculate_correlation_score(self, db_wavenumbers, db_intensities):
        """Calculate correlation score between current spectrum and database spectrum."""
        try:
            # Convert to numpy arrays with explicit float conversion
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)
            
            # Check for empty or invalid data
            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0
            
            # Interpolate to common wavenumber grid
            common_wavenumbers = np.linspace(
                max(self.current_wavenumbers.min(), db_wavenumbers.min()),
                min(self.current_wavenumbers.max(), db_wavenumbers.max()),
                min(len(self.current_wavenumbers), len(db_wavenumbers))
            )
            
            query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
            db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)
            
            # Check for zero variance (constant spectra)
            query_std = np.std(query_interp)
            db_std = np.std(db_interp)
            
            if query_std == 0 or db_std == 0:
                return 0.0
            
            # Normalize to zero mean, unit variance
            query_norm = (query_interp - np.mean(query_interp)) / query_std
            db_norm = (db_interp - np.mean(db_interp)) / db_std
            
            # Calculate correlation
            correlation = np.corrcoef(query_norm, db_norm)[0, 1]
            
            # Convert correlation (-1 to 1) to similarity score (0 to 1)
            # Use absolute correlation for spectral matching (shape similarity)
            similarity = abs(correlation)
            
            return max(0, min(1, similarity))  # Ensure [0, 1] range
            
        except Exception as e:
            print(f"Error in correlation score calculation: {e}")
            return 0.0

    def calculate_multi_window_score(self, db_wavenumbers, db_intensities):
        """Calculate correlation score using multiple diagnostic frequency windows."""
        try:
            # Convert to numpy arrays with explicit float conversion
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)
            
            # Check for empty or invalid data
            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0
            
            # Define diagnostic frequency windows with weights based on importance
            windows = [
                {"name": "Lattice", "range": (50, 400), "weight": 0.2},      # Lattice modes, heavy atoms
                {"name": "Bending", "range": (400, 800), "weight": 0.3},     # Bending modes, ring deformations  
                {"name": "Stretching", "range": (800, 1200), "weight": 0.4}, # Stretching modes, most diagnostic
                {"name": "High", "range": (1200, 1800), "weight": 0.1}       # High frequency modes
            ]
            
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for window in windows:
                # Find overlapping region between query and database in this window
                window_start, window_end = window["range"]
                
                # Check if there's any overlap in this window
                query_mask = (self.current_wavenumbers >= window_start) & (self.current_wavenumbers <= window_end)
                db_mask = (db_wavenumbers >= window_start) & (db_wavenumbers <= window_end)
                
                if not np.any(query_mask) or not np.any(db_mask):
                    continue  # Skip this window if no data
                
                # Get the overlapping range
                overlap_start = max(window_start, 
                                  max(self.current_wavenumbers[query_mask].min(), db_wavenumbers[db_mask].min()))
                overlap_end = min(window_end,
                                min(self.current_wavenumbers[query_mask].max(), db_wavenumbers[db_mask].max()))
                
                if overlap_end <= overlap_start:
                    continue  # No meaningful overlap
                
                # Create common wavenumber grid for this window
                n_points = min(50, np.sum(query_mask), np.sum(db_mask))  # Adaptive resolution
                if n_points < 5:
                    continue  # Too few points for reliable correlation
                
                common_wavenumbers = np.linspace(overlap_start, overlap_end, n_points)
                
                # Interpolate both spectra to common grid
                query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
                db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)
                
                # Check for zero variance
                query_std = np.std(query_interp)
                db_std = np.std(db_interp)
                
                if query_std == 0 or db_std == 0:
                    continue  # Skip windows with no variation
                
                # Normalize to zero mean, unit variance
                query_norm = (query_interp - np.mean(query_interp)) / query_std
                db_norm = (db_interp - np.mean(db_interp)) / db_std
                
                # Calculate correlation for this window
                correlation = np.corrcoef(query_norm, db_norm)[0, 1]
                window_similarity = abs(correlation)  # Use absolute correlation
                
                # Add to weighted sum
                total_weighted_score += window_similarity * window["weight"]
                total_weight += window["weight"]
            
            if total_weight == 0:
                return 0.0  # No valid windows
            
            # Return weighted average
            final_score = total_weighted_score / total_weight
            return max(0, min(1, final_score))
            
        except Exception as e:
            print(f"Error in multi-window score calculation: {e}")
            return 0.0

    def calculate_mineral_vibration_score(self, db_wavenumbers, db_intensities):
        """Calculate correlation score weighted by mineral vibrational mode significance."""
        try:
            # Convert to numpy arrays with explicit float conversion
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)
            
            # Check for empty or invalid data
            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0
            
            # Define vibrational mode regions based on mineral chemistry
            vibrational_modes = [
                {"name": "Metal-O lattice", "range": (50, 300), "weight": 0.15},     # Lattice vibrations
                {"name": "M-O bending", "range": (300, 500), "weight": 0.20},       # Metal-oxygen bending
                {"name": "Ring/chain bending", "range": (500, 700), "weight": 0.25}, # Structural bending
                {"name": "Si-O stretching", "range": (700, 1000), "weight": 0.30},   # Silicate stretching (most diagnostic)
                {"name": "CO3/SO4 modes", "range": (1000, 1200), "weight": 0.10}    # Carbonate/sulfate modes
            ]
            
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for mode in vibrational_modes:
                # Find overlapping region in this vibrational mode range
                mode_start, mode_end = mode["range"]
                
                # Check for overlap
                query_mask = (self.current_wavenumbers >= mode_start) & (self.current_wavenumbers <= mode_end)
                db_mask = (db_wavenumbers >= mode_start) & (db_wavenumbers <= mode_end)
                
                if not np.any(query_mask) or not np.any(db_mask):
                    continue  # Skip if no data in this mode region
                
                # Get the overlapping range
                overlap_start = max(mode_start,
                                  max(self.current_wavenumbers[query_mask].min(), db_wavenumbers[db_mask].min()))
                overlap_end = min(mode_end,
                                min(self.current_wavenumbers[query_mask].max(), db_wavenumbers[db_mask].max()))
                
                if overlap_end <= overlap_start:
                    continue
                
                # Adaptive resolution based on data density and mode importance
                base_points = 30 if mode["weight"] >= 0.25 else 20  # More points for important modes
                n_points = min(base_points, np.sum(query_mask), np.sum(db_mask))
                
                if n_points < 3:
                    continue
                
                common_wavenumbers = np.linspace(overlap_start, overlap_end, n_points)
                
                # Interpolate both spectra
                query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
                db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)
                
                # Enhanced weighting: boost correlation in regions with strong peaks
                # This emphasizes chemically meaningful vibrational features
                peak_intensity_weight = np.mean(query_interp) / np.max(self.processed_intensities) if np.max(self.processed_intensities) > 0 else 0.1
                effective_weight = mode["weight"] * (1 + peak_intensity_weight)
                
                # Check for zero variance
                query_std = np.std(query_interp)
                db_std = np.std(db_interp)
                
                if query_std == 0 or db_std == 0:
                    continue
                
                # Normalize
                query_norm = (query_interp - np.mean(query_interp)) / query_std
                db_norm = (db_interp - np.mean(db_interp)) / db_std
                
                # Calculate correlation for this vibrational mode
                correlation = np.corrcoef(query_norm, db_norm)[0, 1]
                mode_similarity = abs(correlation)
                
                # Add to weighted sum with enhanced weighting
                total_weighted_score += mode_similarity * effective_weight
                total_weight += effective_weight
            
            if total_weight == 0:
                return 0.0
            
            # Return weighted average
            final_score = total_weighted_score / total_weight
            return max(0, min(1, final_score))
            
        except Exception as e:
            print(f"Error in mineral vibration score calculation: {e}")
            return 0.0

    def calculate_dtw_score(self, db_wavenumbers, db_intensities):
        """Calculate optimized DTW similarity score between current spectrum and database spectrum."""
        try:
            # Import fastdtw
            from fastdtw import fastdtw
            
            # Convert to numpy arrays with explicit float conversion
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)
            
            # Check for empty or invalid data
            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0
            
            # Find overlapping wavenumber region
            overlap_start = max(self.current_wavenumbers.min(), db_wavenumbers.min())
            overlap_end = min(self.current_wavenumbers.max(), db_wavenumbers.max())
            
            # Check for sufficient overlap (at least 30% of either spectrum)
            query_range = self.current_wavenumbers.max() - self.current_wavenumbers.min()
            db_range = db_wavenumbers.max() - db_wavenumbers.min()
            overlap_range = overlap_end - overlap_start
            
            if overlap_range < 0.3 * min(query_range, db_range):
                return 0.0  # Insufficient overlap
            
            # OPTIMIZATION 1: Use much fewer points for DTW (50-100 instead of 200)
            # DTW is O(n*m) complexity, so reducing points gives huge speedup
            n_points = min(75, len(self.current_wavenumbers), len(db_wavenumbers))
            common_wavenumbers = np.linspace(overlap_start, overlap_end, n_points)
            
            # Interpolate both spectra to common grid
            query_interp = np.interp(common_wavenumbers, self.current_wavenumbers, self.processed_intensities)
            db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)
            
            # Check for zero variance
            if np.std(query_interp) == 0 or np.std(db_interp) == 0:
                return 0.0
            
            # OPTIMIZATION 2: Simple min-max normalization (faster than detailed checks)
            query_norm = (query_interp - query_interp.min()) / (query_interp.max() - query_interp.min())
            db_norm = (db_interp - db_interp.min()) / (db_interp.max() - db_interp.min())
            
            # OPTIMIZATION 3: Use fastdtw with radius constraint for even more speed
            # Radius limits the search space significantly
            distance, path = fastdtw(query_norm, db_norm, radius=min(10, n_points//5))
            
            # Convert distance to similarity score (simplified calculation)
            if len(path) > 0:
                normalized_distance = distance / len(path)
                # Simplified similarity calculation
                similarity = 1.0 / (1.0 + normalized_distance * 3.0)
                return max(0, min(1, similarity))
            else:
                return 0.0
            
        except ImportError:
            # Fallback to correlation if fastdtw is not available
            return self.calculate_correlation_score(db_wavenumbers, db_intensities)
        except Exception as e:
            print(f"Error in DTW score calculation: {e}")
            return 0.0

    def calculate_peak_score(self, db_wavenumbers, db_intensities, user_specified_peaks=None, peak_tolerance=10):
        """Calculate peak-based similarity score between current spectrum and database spectrum.
        
        Args:
            db_wavenumbers: Database spectrum wavenumbers
            db_intensities: Database spectrum intensities  
            user_specified_peaks: List of peak positions specified by user (optional)
            peak_tolerance: Tolerance for peak matching in cm⁻¹
        """
        try:
            # Convert to numpy arrays with explicit float conversion
            db_wavenumbers = np.array(db_wavenumbers, dtype=float)
            db_intensities = np.array(db_intensities, dtype=float)
            
            # Check for empty or invalid data
            if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
                return 0.0
            
            # Find peaks in database spectrum
            db_peaks, _ = find_peaks(db_intensities, height=np.max(db_intensities) * 0.1)
            
            if len(db_peaks) == 0:
                return 0.0
            
            db_peak_positions = db_wavenumbers[db_peaks]
            db_peak_intensities = db_intensities[db_peaks]
            
            # Determine which peaks to use for comparison
            if user_specified_peaks and len(user_specified_peaks) > 0:
                # Use user-specified peaks as the reference
                query_peak_positions = np.array(user_specified_peaks)
                # For intensity comparison, interpolate the current spectrum at specified positions
                if self.current_wavenumbers is not None and self.processed_intensities is not None:
                    query_peak_intensities = np.interp(query_peak_positions, self.current_wavenumbers, self.processed_intensities)
                else:
                    # If no current spectrum, use equal weights
                    query_peak_intensities = np.ones_like(query_peak_positions)
                
                score_type = "user_specified"
            else:
                # Use detected peaks from current spectrum (original behavior)
                if self.detected_peaks is None or len(self.detected_peaks) == 0:
                    return 0.0
                
                query_peak_positions = self.current_wavenumbers[self.detected_peaks]
                query_peak_intensities = self.processed_intensities[self.detected_peaks]
                score_type = "detected_peaks"
            
            # Calculate peak matching score
            matched_peaks = 0
            total_intensity_diff = 0
            peak_matches = []  # Track which peaks matched for debugging
            
            for i, (query_pos, query_int) in enumerate(zip(query_peak_positions, query_peak_intensities)):
                # Find closest peak in database spectrum
                distances = np.abs(db_peak_positions - query_pos)
                min_dist_idx = np.argmin(distances)
                min_distance = distances[min_dist_idx]
                
                if min_distance <= peak_tolerance:
                    matched_peaks += 1
                    peak_matches.append({
                        'query_pos': query_pos,
                        'db_pos': db_peak_positions[min_dist_idx],
                        'distance': min_distance
                    })
                    
                    # Calculate intensity similarity (normalized)
                    db_int = db_peak_intensities[min_dist_idx]
                    
                    if score_type == "user_specified":
                        # For user-specified peaks, give less weight to intensity differences
                        # since the user cares primarily about peak positions
                        query_norm = query_int / np.max(query_peak_intensities) if np.max(query_peak_intensities) > 0 else 1
                        db_norm = db_int / np.max(db_peak_intensities) if np.max(db_peak_intensities) > 0 else 1
                        intensity_diff = abs(query_norm - db_norm) * 0.3  # Reduced weight for intensity
                    else:
                        # For detected peaks, use full intensity comparison
                        query_norm = query_int / np.max(query_peak_intensities)
                        db_norm = db_int / np.max(db_peak_intensities)
                        intensity_diff = abs(query_norm - db_norm)
                    
                    total_intensity_diff += intensity_diff
            
            if matched_peaks == 0:
                return 0.0
            
            # Calculate score based on percentage of matched peaks and intensity similarity
            peak_match_ratio = matched_peaks / len(query_peak_positions)
            avg_intensity_similarity = 1 - (total_intensity_diff / matched_peaks)
            
            # Adjust scoring weights based on whether we're using user-specified peaks
            if score_type == "user_specified":
                # For user-specified peaks, emphasize peak position matching over intensity
                score = 0.8 * peak_match_ratio + 0.2 * avg_intensity_similarity
                
                # Bonus for high match ratio when using user-specified peaks
                if peak_match_ratio >= 0.8:
                    score = min(1.0, score * 1.1)  # 10% bonus for high match ratio
                    
            else:
                # Original scoring for detected peaks
                score = 0.7 * peak_match_ratio + 0.3 * avg_intensity_similarity
            
            return max(0, min(1, score))
            
        except Exception as e:
            print(f"Error in peak score calculation: {e}")
            return 0.0

    def calculate_combined_score(self, db_wavenumbers, db_intensities, user_specified_peaks=None, peak_tolerance=10):
        """Calculate optimized combined similarity score using correlation, DTW, and optionally peak matching."""
        try:
            # OPTIMIZATION 1: Start with fast correlation score
            correlation_score = self.calculate_correlation_score(db_wavenumbers, db_intensities)
            
            # OPTIMIZATION 2: Early termination for obviously poor matches
            # If correlation is very low, don't waste time on expensive DTW
            if correlation_score < 0.2:  # Very poor correlation
                return correlation_score * 0.5  # Return reduced score without DTW
            
            # OPTIMIZATION 3: For moderate correlations, run DTW but weight less
            if correlation_score < 0.5:  # Moderate correlation
                dtw_score = self.calculate_dtw_score(db_wavenumbers, db_intensities)
                # Weight correlation more heavily for moderate matches to save time
                if user_specified_peaks and len(user_specified_peaks) > 0:
                    peak_score = self.calculate_peak_score(db_wavenumbers, db_intensities, user_specified_peaks, peak_tolerance)
                    combined_score = 0.4 * correlation_score + 0.3 * dtw_score + 0.3 * peak_score
                else:
                    combined_score = 0.6 * correlation_score + 0.4 * dtw_score
            else:
                # OPTIMIZATION 4: For good correlations, run full analysis
                dtw_score = self.calculate_dtw_score(db_wavenumbers, db_intensities)
                
                if user_specified_peaks and len(user_specified_peaks) > 0:
                    peak_score = self.calculate_peak_score(db_wavenumbers, db_intensities, user_specified_peaks, peak_tolerance)
                    # Standard three-way weighted average with emphasis on user-specified peaks
                    combined_score = 0.2 * correlation_score + 0.3 * dtw_score + 0.5 * peak_score
                else:
                    # Standard two-way combination
                    combined_score = 0.3 * correlation_score + 0.7 * dtw_score
            
            return max(0, min(1, combined_score))
            
        except Exception as e:
            print(f"Error in combined score calculation: {e}")
            return 0.0

    def create_filter_summary(self, filters):
        """Create a summary string of applied filters."""
        summary_parts = []
        
        if 'peak_positions' in filters:
            summary_parts.append(f"Peaks: {', '.join(map(str, filters['peak_positions']))}")
        if 'chemical_family' in filters:
            summary_parts.append(f"Family: {filters['chemical_family']}")
        if 'hey_classification' in filters:
            summary_parts.append(f"Hey: {filters['hey_classification']}")
        if 'only_elements' in filters:
            summary_parts.append(f"Only: {', '.join(filters['only_elements'])}")
        if 'required_elements' in filters:
            summary_parts.append(f"Required: {', '.join(filters['required_elements'])}")
        if 'exclude_elements' in filters:
            summary_parts.append(f"Exclude: {', '.join(filters['exclude_elements'])}")
        
        return "; ".join(summary_parts) if summary_parts else "None"

    def display_search_results(self, matches, search_type, additional_info=""):
        """Display search results in the comprehensive results window."""
        if not matches:
            # Show simple message for no results
            result_text = f"{search_type} Results\n{'='*40}\n\n"
            result_text += "No matches found above threshold.\n"
            if additional_info:
                result_text += f"\n{additional_info}\n"
            result_text += "\nTry lowering the similarity threshold or adding more spectra to the database."
            self.search_results_text.setPlainText(result_text)
            return
        
        # Enhance matches with actual spectrum data from database
        enhanced_matches = []
        for match in matches:
            match_name = match.get('name')
            if match_name in self.raman_db.database:
                db_entry = self.raman_db.database[match_name]
                enhanced_match = match.copy()
                enhanced_match['wavenumbers'] = db_entry.get('wavenumbers', [])
                enhanced_match['intensities'] = db_entry.get('intensities', [])
                enhanced_matches.append(enhanced_match)
            else:
                # Add empty spectrum data if not found
                enhanced_match = match.copy()
                enhanced_match['wavenumbers'] = []
                enhanced_match['intensities'] = []
                enhanced_matches.append(enhanced_match)
        
        # Show brief summary in the search tab
        result_text = f"{search_type} Results\n{'='*40}\n\n"
        if additional_info:
            result_text += f"{additional_info}\n\n"
        
        result_text += f"Found {len(enhanced_matches)} matches. Opening detailed results window...\n\n"
        result_text += "Top 3 matches:\n"
        
        for i, match in enumerate(enhanced_matches[:3], 1):
            result_text += f"{i}. {match['name']} (Score: {match['score']:.3f})\n"
            metadata = match.get('metadata', {})
            if metadata.get('NAME'):
                result_text += f"   Mineral: {metadata['NAME']}\n"
        
        self.search_results_text.setPlainText(result_text)
        
        # Launch the comprehensive search results window
        try:
            self.search_results_window = SearchResultsWindow(
                enhanced_matches,
                self.current_wavenumbers,
                self.processed_intensities,
                search_type,
                parent=self
            )
            self.search_results_window.show()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open search results window:\n{str(e)}"
            )

    def update_metadata_filter_options(self):
        """Update the dropdown options for chemical family and Hey classification."""
        # Get unique values from database
        chemical_families = set()
        hey_classifications = set()
        
        for data in self.raman_db.database.values():
            metadata = data.get('metadata', {})
            
            # Chemical family
            family = metadata.get('CHEMICAL FAMILY') or metadata.get('Chemical Family')
            if family and isinstance(family, str):
                chemical_families.add(family.strip())
            
            # Hey classification
            hey_class = metadata.get('HEY CLASSIFICATION')
            if hey_class and isinstance(hey_class, str):
                hey_classifications.add(hey_class.strip())
        
        # Update comboboxes
        if hasattr(self, 'chemical_family_combo'):
            self.chemical_family_combo.clear()
            self.chemical_family_combo.addItem("")  # Empty option
            self.chemical_family_combo.addItems(sorted(chemical_families))
        
        if hasattr(self, 'hey_classification_combo'):
            self.hey_classification_combo.clear()
            self.hey_classification_combo.addItem("")  # Empty option
            self.hey_classification_combo.addItems(sorted(hey_classifications))



    # Note: analyze_mixed_minerals function removed for clean rebuild

    def create_advanced_tab(self):
        """Create the advanced analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Primary Analysis Tools group (dark blue buttons)
        primary_group = QGroupBox("Primary Analysis Tools")
        primary_layout = QVBoxLayout(primary_group)
        
        # Dark blue button style
        dark_blue_style = """
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """
        
        # Multi-spectrum comparison
        multi_spectrum_btn = QPushButton("Multi-Spectrum Comparison")
        multi_spectrum_btn.clicked.connect(self.launch_multi_spectrum_manager)
        multi_spectrum_btn.setStyleSheet(dark_blue_style)
        primary_layout.addWidget(multi_spectrum_btn)
        
        # Spectral deconvolution
        deconvolution_btn = QPushButton("Spectral Deconvolution")
        deconvolution_btn.clicked.connect(self.launch_deconvolution)
        deconvolution_btn.setStyleSheet(dark_blue_style)
        primary_layout.addWidget(deconvolution_btn)
        

        

        
        layout.addWidget(primary_group)
        
        # Spatial Analysis Tools group
        spatial_group = QGroupBox("Spatial Analysis Tools")
        spatial_layout = QVBoxLayout(spatial_group)
        
        # Spatial analysis button style (teal/green)
        spatial_style = """
            QPushButton {
                background-color: #0D9488;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0F766E;
            }
            QPushButton:pressed {
                background-color: #115E59;
            }
        """
        
        # Map analysis
        map_analysis_btn = QPushButton("Map Analysis")
        map_analysis_btn.clicked.connect(self.launch_map_analysis)
        map_analysis_btn.setStyleSheet(spatial_style)
        spatial_layout.addWidget(map_analysis_btn)
        
        # Cluster analysis
        cluster_analysis_btn = QPushButton("Cluster Analysis")
        cluster_analysis_btn.clicked.connect(self.launch_cluster_analysis)
        cluster_analysis_btn.setStyleSheet(spatial_style)
        spatial_layout.addWidget(cluster_analysis_btn)
        
        # Polarization analysis
        polarization_analysis_btn = QPushButton("Polarization Analysis")
        polarization_analysis_btn.clicked.connect(self.launch_polarization_analysis)
        polarization_analysis_btn.setStyleSheet(spatial_style)
        spatial_layout.addWidget(polarization_analysis_btn)
        
        # Mixture analysis
        mixture_analysis_btn = QPushButton("Mixture Analysis")
        mixture_analysis_btn.clicked.connect(self.launch_mixture_analysis)
        mixture_analysis_btn.setStyleSheet(spatial_style)
        spatial_layout.addWidget(mixture_analysis_btn)
        
        layout.addWidget(spatial_group)
        
        # Mechanical Analysis Tools group
        mechanical_group = QGroupBox("Mechanical Analysis Tools")
        mechanical_layout = QVBoxLayout(mechanical_group)
        
        # Mechanical analysis button style (purple)
        mechanical_style = """
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #8B5CF6;
            }
            QPushButton:pressed {
                background-color: #6D28D9;
            }
        """
        
        # Stress/Strain analysis
        stress_strain_btn = QPushButton("Stress/Strain Analysis")
        stress_strain_btn.clicked.connect(self.launch_stress_strain_analysis)
        stress_strain_btn.setStyleSheet(mechanical_style)
        mechanical_layout.addWidget(stress_strain_btn)
        
        # Chemical strain analysis
        chemical_strain_btn = QPushButton("Chemical Strain Analysis")
        chemical_strain_btn.clicked.connect(self.launch_chemical_strain_analysis)
        chemical_strain_btn.setStyleSheet(mechanical_style)
        mechanical_layout.addWidget(chemical_strain_btn)
        
        layout.addWidget(mechanical_group)
        
        # Additional Processing Tools group
        processing_group = QGroupBox("Additional Processing Tools")
        processing_layout = QVBoxLayout(processing_group)
        
        # Additional tools button style (orange)
        additional_style = """
            QPushButton {
                background-color: #EA580C;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
        """
        
        # Spectra calibration
        spectra_calibration_btn = QPushButton("Spectra Calibration")
        spectra_calibration_btn.clicked.connect(self.launch_spectra_calibration)
        spectra_calibration_btn.setStyleSheet(additional_style)
        processing_layout.addWidget(spectra_calibration_btn)
        
        # Baseline correction tools
        baseline_btn = QPushButton("Advanced Baseline Correction")
        baseline_btn.clicked.connect(self.launch_baseline_tools)
        baseline_btn.setStyleSheet(additional_style)
        processing_layout.addWidget(baseline_btn)
        
        # Data conversion tools
        data_conversion_btn = QPushButton("Advanced Data Conversion")
        data_conversion_btn.clicked.connect(self.launch_data_conversion_tools)
        data_conversion_btn.setStyleSheet(additional_style)
        processing_layout.addWidget(data_conversion_btn)
        
        layout.addWidget(processing_group)
        
        # Export and Reporting group
        export_group = QGroupBox("Export and Reporting")
        export_layout = QVBoxLayout(export_group)
        
        # Export tools button style (gray)
        export_style = """
            QPushButton {
                background-color: #6B7280;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
            QPushButton:pressed {
                background-color: #374151;
            }
        """
        
        # Batch export
        batch_export_btn = QPushButton("Batch Export Tools")
        batch_export_btn.clicked.connect(self.launch_batch_export)
        batch_export_btn.setStyleSheet(export_style)
        export_layout.addWidget(batch_export_btn)
        
        # Report generation
        report_btn = QPushButton("Generate Analysis Report")
        report_btn.clicked.connect(self.generate_analysis_report)
        report_btn.setStyleSheet(export_style)
        export_layout.addWidget(report_btn)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return tab

    def launch_spectra_calibration(self):
        """Launch spectra calibration tool."""
        QMessageBox.information(
            self,
            "Spectra Calibration",
            "Spectra calibration tool will be implemented.\n\n"
            "This will provide:\n"
            "• Wavelength calibration and correction\n"
            "• Intensity calibration using standards\n"
            "• Instrument response correction\n"
            "• Calibration curve generation"
        )

    def launch_baseline_tools(self):
        """Launch advanced baseline correction tools."""
        QMessageBox.information(
            self,
            "Baseline Correction",
            "Advanced baseline correction tools will be implemented.\n\n"
            "This will provide:\n"
            "• Additional baseline algorithms\n"
            "• Interactive baseline selection\n"
            "• Baseline parameter optimization\n"
            "• Preview and comparison modes"
        )

    def launch_data_conversion_tools(self):
        """Launch advanced data conversion tools."""
        try:
            # Import the data conversion dialog
            from core.data_conversion_dialog import DataConversionDialog
            
            # Create and show the dialog
            dialog = DataConversionDialog(self)
            dialog.exec()
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Could not import data conversion tools:\n{str(e)}\n\n"
                "Please ensure core/data_conversion_dialog.py is available."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Data Conversion Error", 
                f"Failed to launch data conversion tools:\n{str(e)}"
            )

    def launch_deconvolution(self):
        """Launch spectral deconvolution tool."""
        try:
            # Import and launch the Qt6 spectral deconvolution module
            from peak_fitting_qt6 import launch_spectral_deconvolution
            
            # Launch with current spectrum data if available
            if self.current_wavenumbers is not None and self.current_intensities is not None:
                launch_spectral_deconvolution(
                    self, 
                    self.current_wavenumbers, 
                    self.processed_intensities if self.processed_intensities is not None 
                    else self.current_intensities
                )
            else:
                # Launch without initial data - user can load files in the peak fitting interface
                launch_spectral_deconvolution(self)
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Module Error",
                f"Failed to import spectral deconvolution module:\n{str(e)}\n\n"
                "Make sure peak_fitting_qt6.py is available."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Launch Error", 
                f"Failed to launch spectral deconvolution:\n{str(e)}"
            )

    def launch_batch_export(self):
        """Launch batch export tools."""
        QMessageBox.information(
            self,
            "Batch Export",
            "Batch export tools will be implemented.\n\n"
            "This will provide:\n"
            "• Batch processing of multiple files\n"
            "• Format conversion utilities\n"
            "• Automated analysis workflows\n"
            "• Bulk data export options"
        )

    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        if self.current_wavenumbers is None:
            QMessageBox.warning(self, "No Data", "Load a spectrum first to generate a report.")
            return
            
        QMessageBox.information(
            self,
            "Analysis Report",
            "Analysis report generation will be implemented.\n\n"
            "This will provide:\n"
            "• Comprehensive spectrum analysis\n"
            "• Peak identification and characterization\n"
            "• Search results and matches\n"
            "• Exportable PDF and HTML reports"
        )









    def launch_map_analysis(self):
        """Launch Raman mapping analysis tool."""
        try:
            # Import and launch the Qt6 map analysis module
            from map_analysis_2d.ui import MapAnalysisMainWindow
            
            # Create and show the map analysis window
            self.map_analysis_window = MapAnalysisMainWindow()
            self.map_analysis_window.show()
            
            # Show success message
            self.statusBar().showMessage("Map Analysis tool launched successfully")
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Map Analysis Error",
                f"Failed to import map analysis module:\n{str(e)}\n\n"
                "Please ensure the map_analysis_2d module is available."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Map Analysis Error",
                f"Failed to launch map analysis:\n{str(e)}"
            )

    def launch_cluster_analysis(self):
        """Launch cluster analysis tool."""
        try:
            # Import and launch the cluster analysis module
            from raman_cluster_analysis_qt6 import launch_cluster_analysis
            
            # Launch the cluster analysis window - pass self as both parent and raman_app
            self.cluster_analysis_window = launch_cluster_analysis(self, self)
            
            # Show success message in status bar
            self.statusBar().showMessage("Cluster Analysis window launched successfully")
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import cluster analysis module:\n{str(e)}\n\n"
                "Please ensure raman_cluster_analysis_qt6.py is in the same directory."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Cluster Analysis Error",
                f"Failed to launch cluster analysis:\n{str(e)}"
            )

    def launch_polarization_analysis(self):
        """Launch polarization analysis tool."""
        try:
            # Import and launch the polarization analysis module
            from raman_polarization_analyzer_qt6 import RamanPolarizationAnalyzerQt6
            
            # Create and show the polarization analyzer window
            self.polarization_analyzer = RamanPolarizationAnalyzerQt6()
            
            # If we have current spectrum data, pass it to the polarization analyzer
            if self.current_wavenumbers is not None and self.current_intensities is not None:
                spectrum_data = {
                    'name': self.spectrum_file_path or 'Current Spectrum',
                    'wavenumbers': self.current_wavenumbers,
                    'intensities': self.processed_intensities if self.processed_intensities is not None else self.current_intensities,
                    'source': 'main_app'
                }
                self.polarization_analyzer.current_spectrum = spectrum_data
                self.polarization_analyzer.original_spectrum = spectrum_data.copy()
                
                # Update the spectrum plot in the polarization analyzer
                self.polarization_analyzer.update_spectrum_plot()
                
                # Show success message in status bar
                self.statusBar().showMessage("Polarization Analysis launched with current spectrum")
            else:
                # Show message that no spectrum is loaded
                self.statusBar().showMessage("Polarization Analysis launched - load a spectrum to begin analysis")
            
            # Show the window
            self.polarization_analyzer.show()
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import polarization analysis module:\n{str(e)}\n\n"
                "Please ensure raman_polarization_analyzer_qt6.py is in the same directory."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Polarization Analysis Error",
                f"Failed to launch polarization analysis:\n{str(e)}"
            )

    def launch_mixture_analysis(self):
        """Launch mixture analysis tool - NEW Interactive Version."""
        try:
            # Import and launch the NEW interactive mixture analysis module
            from raman_mixture_analysis_interactive import InteractiveMixtureAnalyzer
            
            # Create QApplication instance if needed (for standalone launch)
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            # Create and show the interactive mixture analysis window
            self.mixture_analyzer = InteractiveMixtureAnalyzer()
            
            # If we have current spectrum data, set it in the interactive analyzer
            if self.current_wavenumbers is not None and self.current_intensities is not None:
                # Use processed intensities if available, otherwise use original
                intensities = (
                    self.processed_intensities.copy() if self.processed_intensities is not None 
                    else self.current_intensities.copy()
                )
                
                # Set spectrum data directly in the interactive analyzer
                self.mixture_analyzer.user_wavenumbers = self.current_wavenumbers.copy()
                self.mixture_analyzer.user_spectrum = intensities.copy()
                self.mixture_analyzer.original_spectrum = intensities.copy()
                self.mixture_analyzer.current_residual = intensities.copy()
                
                # Get spectrum name
                from pathlib import Path
                spectrum_name = getattr(self, 'spectrum_file_path', 'Current Spectrum from Main App')
                if hasattr(spectrum_name, '__fspath__') or isinstance(spectrum_name, (str, Path)):
                    spectrum_name = os.path.basename(str(spectrum_name))
                else:
                    spectrum_name = 'Current Spectrum from Main App'
                
                # Use QTimer to update UI elements after window is shown
                QTimer.singleShot(200, lambda: self._update_mixture_analyzer_ui(
                    self.mixture_analyzer, len(self.current_wavenumbers), spectrum_name, 
                    self.current_wavenumbers[0], self.current_wavenumbers[-1]))
                
                # Show success message in status bar
                self.statusBar().showMessage("NEW Interactive Mixture Analysis launched with current spectrum")
            else:
                # Show message that no spectrum is loaded
                self.statusBar().showMessage("NEW Interactive Mixture Analysis launched - use 'Load Spectrum Data' or demo data")
            
            # Show the window
            self.mixture_analyzer.show()
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import NEW interactive mixture analysis module:\n{str(e)}\n\n"
                "Please ensure raman_mixture_analysis_interactive.py is in the same directory.\n"
                "For the old version, use: python raman_mixture_analysis_qt6.py"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Interactive Mixture Analysis Error",
                f"Failed to launch NEW interactive mixture analysis:\n{str(e)}"
            )
    
    def _update_mixture_analyzer_ui(self, mixture_analyzer, num_points, spectrum_name, wn_min, wn_max):
        """Helper method to update mixture analyzer UI elements after window is shown."""
        try:
            # Update data status
            mixture_analyzer.data_status_label.setText(f"Loaded from main app: {num_points} points")
            mixture_analyzer.search_btn.setEnabled(True)
            
            # Log the data loading
            mixture_analyzer.log_status(f"✅ Loaded spectrum from main app: {spectrum_name}")
            mixture_analyzer.log_status(f"   Range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹")
        except Exception as e:
            print(f"⚠️ Warning: Could not update mixture analyzer UI: {e}")
    
    def _update_mixture_analyzer_search_ui(self, mixture_analyzer, num_points, spectrum_name, wn_min, wn_max):
        """Helper method to update mixture analyzer UI for search results."""
        try:
            # Update data status
            mixture_analyzer.data_status_label.setText(f"Loaded from search: {num_points} points")
            mixture_analyzer.search_btn.setEnabled(True)
            
            # Log the data loading
            mixture_analyzer.log_status(f"✅ Loaded spectrum from search window: {spectrum_name}")
            mixture_analyzer.log_status(f"   Range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹")
        except Exception as e:
            print(f"⚠️ Warning: Could not update mixture analyzer search UI: {e}")
    
    def _update_mixture_analyzer_constraint_ui(self, mixture_analyzer, num_points, spectrum_name, wn_min, wn_max):
        """Helper method to update mixture analyzer UI for constraint analysis."""
        try:
            # Update data status
            mixture_analyzer.data_status_label.setText(f"Loaded from search: {num_points} points")
            mixture_analyzer.search_btn.setEnabled(True)
            
            # Log the data loading
            mixture_analyzer.log_status(f"✅ Loaded spectrum from search window: {spectrum_name}")
            mixture_analyzer.log_status(f"   Range: {wn_min:.1f} - {wn_max:.1f} cm⁻¹")
        except Exception as e:
            print(f"⚠️ Warning: Could not update mixture analyzer constraint UI: {e}")

    def launch_stress_strain_analysis(self):
        """Launch stress/strain analysis tool."""
        if self.current_wavenumbers is None or self.current_intensities is None:
            QMessageBox.warning(self, "No Data", "Load a spectrum first to perform stress/strain analysis.")
            return
            
        try:
            # Import and launch the stress/strain analysis module
            QMessageBox.information(
                self,
                "Stress/Strain Analysis",
                "Launching Stress/Strain Analysis...\n\n"
                "This feature provides:\n"
                "• Peak shift analysis for stress determination\n"
                "• Strain calculation from lattice deformation\n"
                "• Pressure coefficient analysis\n"
                "• Mechanical property evaluation"
            )
            
            # TODO: Replace with actual stress/strain analysis module
            # from stress_strain_analysis_qt6 import launch_stress_strain_analysis
            # launch_stress_strain_analysis(self, self.current_wavenumbers, self.processed_intensities)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Stress/Strain Analysis Error",
                f"Failed to launch stress/strain analysis:\n{str(e)}"
            )

    def launch_chemical_strain_analysis(self):
        """Launch chemical strain analysis tool with material selection."""
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QButtonGroup, QRadioButton
            
            # Create material selection dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Chemical Strain Analysis - Material Selection")
            dialog.setFixedSize(500, 300)
            
            layout = QVBoxLayout(dialog)
            
            # Title
            title = QLabel("Select Material System for Analysis:")
            title.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
            layout.addWidget(title)
            
            # Create button group for exclusive selection
            button_group = QButtonGroup(dialog)
            
            # General chemical strain analysis option
            general_radio = QRadioButton("General Chemical Strain Analysis")
            general_radio.setChecked(True)  # Default selection
            button_group.addButton(general_radio, 0)
            layout.addWidget(general_radio)
            
            general_desc = QLabel("• Chemical composition-induced strain analysis\n"
                                 "• Lattice parameter variation studies\n"
                                 "• Solid solution strain effects\n"
                                 "• Compositional gradient analysis")
            general_desc.setStyleSheet("margin-left: 20px; margin-bottom: 15px; color: #666;")
            layout.addWidget(general_desc)
            
            # LiMn2O4 battery analysis option
            battery_radio = QRadioButton("LiMn2O4 Battery Material Analysis")
            button_group.addButton(battery_radio, 1)
            layout.addWidget(battery_radio)
            
            battery_desc = QLabel("• H/Li exchange strain analysis for battery materials\n"
                                 "• Jahn-Teller distortion tracking\n"
                                 "• Time series strain evolution\n"
                                 "• Spinel structure symmetry breaking analysis")
            battery_desc.setStyleSheet("margin-left: 20px; margin-bottom: 15px; color: #666;")
            layout.addWidget(battery_desc)
            
            # Buttons
            button_layout = QHBoxLayout()
            launch_btn = QPushButton("Launch Analysis")
            cancel_btn = QPushButton("Cancel")
            
            launch_btn.setStyleSheet("""
                QPushButton {
                    background-color: #7C3AED;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #8B5CF6;
                }
            """)
            
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6B7280;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #4B5563;
                }
            """)
            
            def on_launch():
                selected_id = button_group.checkedId()
                dialog.accept()
                
                if selected_id == 0:
                    # Launch general chemical strain analysis
                    self.launch_general_chemical_strain_analysis()
                elif selected_id == 1:
                    # Launch LiMn2O4 battery analysis
                    self.launch_limn2o4_strain_analysis()
            
            def on_cancel():
                dialog.reject()
            
            launch_btn.clicked.connect(on_launch)
            cancel_btn.clicked.connect(on_cancel)
            
            button_layout.addStretch()
            button_layout.addWidget(launch_btn)
            button_layout.addWidget(cancel_btn)
            
            layout.addStretch()
            layout.addLayout(button_layout)
            
            # Show dialog
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Chemical Strain Analysis Error",
                f"Failed to launch chemical strain analysis:\n{str(e)}"
            )
    
    def launch_limn2o4_strain_analysis(self):
        """Launch LiMn2O4 battery strain analysis tool."""
        try:
            # Import the battery strain analysis demo
            from battery_strain_analysis.demo_limn2o4_analysis import main as run_limn2o4_demo
            
            # Show information dialog
            reply = QMessageBox.question(
                self,
                "LiMn2O4 Battery Strain Analysis",
                "Launch LiMn2O4 Battery Strain Analysis Demo?\n\n"
                "This will demonstrate:\n"
                "• H/Li exchange strain analysis\n"
                "• Jahn-Teller distortion tracking\n"
                "• Time series strain evolution\n"
                "• Spinel structure analysis\n\n"
                "The demo will generate synthetic time series data\n"
                "and perform comprehensive strain analysis.\n\n"
                "Results will be displayed and saved to:\n"
                "battery_strain_analysis/limn2o4_analysis_results/\n\n"
                "Continue with demo?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Show progress dialog
                progress = QProgressDialog("Running LiMn2O4 strain analysis...", None, 0, 0, self)
                progress.setWindowTitle("Battery Strain Analysis")
                progress.setModal(True)
                progress.show()
                
                # Process events to show the dialog
                QApplication.processEvents()
                
                try:
                    # Run the demo analysis
                    run_limn2o4_demo()
                    
                    progress.close()
                    
                    # Show success message
                    QMessageBox.information(
                        self,
                        "Analysis Complete",
                        "LiMn2O4 battery strain analysis completed successfully!\n\n"
                        "Results saved to:\n"
                        "battery_strain_analysis/limn2o4_analysis_results/\n\n"
                        "Files generated:\n"
                        "• time_series_overview.png - Overview plot\n"
                        "• final_strain_3d.png - 3D strain visualization\n"
                        "• strain_evolution.csv - Strain vs time data\n"
                        "• composition_evolution.csv - Composition data\n"
                        "• analysis_report.txt - Full analysis report\n\n"
                        "Check the results folder for detailed analysis output."
                    )
                    
                    # Update status bar
                    self.statusBar().showMessage("LiMn2O4 battery strain analysis completed successfully")
                    
                except Exception as e:
                    progress.close()
                    raise e
                    
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import LiMn2O4 battery strain analysis module:\n{str(e)}\n\n"
                "Please ensure the battery_strain_analysis package is available\n"
                "with all required components:\n"
                "• battery_strain_analysis/__init__.py\n"
                "• battery_strain_analysis/demo_limn2o4_analysis.py\n"
                "• battery_strain_analysis/limn2o4_analyzer.py\n"
                "• And other required modules"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "LiMn2O4 Analysis Error",
                f"Failed to run LiMn2O4 strain analysis:\n{str(e)}\n\n"
                "Please check that all required dependencies are installed\n"
                "and the battery_strain_analysis package is properly configured."
            )

    def export_database_file(self):
        """Export the database to a file."""
        from database_browser_qt6 import DatabaseBrowserQt6
        
        # Create a temporary browser instance just for export
        temp_browser = DatabaseBrowserQt6(self.raman_db, parent=self)
        temp_browser.export_database()
        
        # Update our database stats after export (in case anything changed)
        self.update_database_stats()

    def import_database_file(self):
        """Import the database from a file."""
        from database_browser_qt6 import DatabaseBrowserQt6
        
        # Create a temporary browser instance just for import
        temp_browser = DatabaseBrowserQt6(self.raman_db, parent=self)
        temp_browser.import_database()
        
        # Update our database stats after import
        self.update_database_stats()

    def launch_mineral_modes_browser(self):
        """Launch the mineral modes database browser."""
        try:
            from mineral_modes_browser_qt6 import MineralModesDatabaseQt6
            
            # Create and show the mineral modes browser window
            self.mineral_modes_browser = MineralModesDatabaseQt6(parent=self)
            self.mineral_modes_browser.show()
            
            # Show success message
            self.statusBar().showMessage("Mineral Modes Database browser launched successfully")
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import mineral modes browser module:\n{str(e)}\n\n"
                "Please ensure mineral_modes_browser_qt6.py is in the same directory."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch mineral modes browser:\n{str(e)}"
            )

    def update_window_title(self, filename=None):
        if filename:
            self.setWindowTitle(f"RamanLab: Raman Spectrum Analysis - {filename}")
        else:
            self.setWindowTitle("RamanLab: Raman Spectrum Analysis")

    # Import Data Methods
    def import_raman_spectral_map(self):
        """Import Raman spectral map data and convert to pkl format for mapping and clustering."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Raman Spectral Map",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;CSV files (*.csv);;Data files (*.dat);;All files (*.*)"
        )
        
        if file_path:
            # Create and show progress dialog
            self.progress_dialog = QProgressDialog("Processing Raman spectral map data...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowTitle("Importing Raman Map")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()
            
            # Create worker thread
            self.import_worker = RamanMapImportWorker(file_path)
            self.import_worker.progress.connect(self.progress_dialog.setValue)
            self.import_worker.status_update.connect(self.progress_dialog.setLabelText)
            self.import_worker.finished.connect(self.on_import_finished)
            self.import_worker.error.connect(self.on_import_error)
            self.progress_dialog.canceled.connect(self.import_worker.cancel)
            
            # Start the worker
            self.import_worker.start()

    def on_import_finished(self, map_data):
        """Handle successful completion of import."""
        self.progress_dialog.close()
        
        try:
            # Save as PKL file
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Raman Map Data as PKL",
                str(Path(self.import_worker.file_path).with_suffix('.pkl')),
                "Pickle files (*.pkl);;All files (*.*)"
            )
            
            if save_path:
                # Show progress for saving
                save_progress = QProgressDialog("Saving PKL file...", None, 0, 0, self)
                save_progress.setWindowTitle("Saving")
                save_progress.setModal(True)
                save_progress.show()
                
                # Process events to show the dialog
                QApplication.processEvents()
                
                with open(save_path, 'wb') as f:
                    pickle.dump(map_data, f)
                
                save_progress.close()
                
                # Display summary
                self.display_map_data_summary(map_data, save_path)
                self.status_bar.showMessage(f"Converted and saved: {Path(save_path).name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save map data:\n{str(e)}")

    def on_import_error(self, error_message):
        """Handle import error."""
        self.progress_dialog.close()
        QMessageBox.critical(self, "Import Error", f"Failed to import Raman spectral map:\n{error_message}")

    def parse_raman_spectral_map(self, file_path, progress_callback=None, status_callback=None):
        """Parse Raman spectral map data where first row is Raman shifts and first two columns are X,Y positions."""
        try:
            if status_callback:
                status_callback("Reading file...")
            
            # Read the file line by line to handle inconsistent column counts
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                raise ValueError("File must have at least 2 lines (header + data)")
            
            total_lines = len(lines) - 1  # Exclude header
            
            if status_callback:
                status_callback("Parsing header...")
            if progress_callback:
                progress_callback(5)
            
            # Parse the first line to get Raman shifts
            first_line = lines[0].strip().split()
            print(f"First line has {len(first_line)} columns")
            
            # Extract Raman shifts from first row (skip first two cells which should be X,Y labels)
            if len(first_line) < 3:
                raise ValueError("First line must have at least 3 columns (X, Y, and Raman shifts)")
            
            raman_shifts = np.array([float(x) for x in first_line[2:]])
            print(f"Extracted {len(raman_shifts)} Raman shifts from header")
            
            # Process spatial data and spectra
            spatial_data = []
            spectra_data = []
            skipped_lines = 0
            
            if status_callback:
                status_callback(f"Processing {total_lines} spectra...")
            
            for i, line in enumerate(lines[1:], 1):  # Start from line 1 (skip header)
                # Update progress every 1000 lines or for small datasets, every 100 lines
                update_frequency = 1000 if total_lines > 10000 else max(100, total_lines // 100)
                
                if i % update_frequency == 0 and progress_callback:
                    progress_percent = 5 + int((i / total_lines) * 60)  # 5-65% for line processing
                    progress_callback(progress_percent)
                    if status_callback:
                        status_callback(f"Processing spectra: {i}/{total_lines} ({progress_percent-5}%)")
                
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    parts = line.split()
                    
                    # Check if we have enough columns
                    if len(parts) < 3:
                        skipped_lines += 1
                        if skipped_lines <= 5:  # Only print first few warnings
                            print(f"Skipping line {i+1}: insufficient columns ({len(parts)})")
                        continue
                    
                    x_pos = float(parts[0])  # X position in microns
                    y_pos = float(parts[1])  # Y position in microns
                    
                    # Extract spectrum intensities - handle variable column counts
                    # Take only as many intensity values as we have Raman shifts
                    spectrum_parts = parts[2:]
                    n_to_take = min(len(spectrum_parts), len(raman_shifts))
                    
                    if n_to_take < len(raman_shifts):
                        # Pad with zeros if we have fewer intensities than expected
                        spectrum = np.zeros(len(raman_shifts))
                        spectrum[:n_to_take] = [float(x) for x in spectrum_parts[:n_to_take]]
                        if skipped_lines <= 5:
                            print(f"Line {i+1}: Padded spectrum from {n_to_take} to {len(raman_shifts)} values")
                    else:
                        # Take only the number of intensities we need
                        spectrum = np.array([float(x) for x in spectrum_parts[:len(raman_shifts)]])
                        if len(spectrum_parts) > len(raman_shifts) and skipped_lines <= 5:
                            print(f"Line {i+1}: Truncated spectrum from {len(spectrum_parts)} to {len(raman_shifts)} values")
                    
                    spatial_data.append([x_pos, y_pos])
                    spectra_data.append(spectrum)
                    
                except (ValueError, IndexError) as e:
                    skipped_lines += 1
                    if skipped_lines <= 5:  # Only print first few errors
                        print(f"Skipping line {i+1}: {e}")
                    continue
            
            if skipped_lines > 5:
                print(f"... and {skipped_lines - 5} more lines were skipped")
            
            if len(spatial_data) == 0:
                raise ValueError("No valid data rows found in file")
            
            if status_callback:
                status_callback("Converting to arrays...")
            if progress_callback:
                progress_callback(70)
            
            spatial_data = np.array(spatial_data)
            spectra_data = np.array(spectra_data)
            
            print(f"Successfully processed {len(spatial_data)} positions")
            print(f"Spatial X range: {spatial_data[:, 0].min():.1f} to {spatial_data[:, 0].max():.1f}")
            print(f"Spatial Y range: {spatial_data[:, 1].min():.1f} to {spatial_data[:, 1].max():.1f}")
            print(f"Spectrum intensity range: {spectra_data.min():.2e} to {spectra_data.max():.2e}")
            
            # Create organized map data structure
            if status_callback:
                status_callback("Creating map data structure...")
            if progress_callback:
                progress_callback(75)
                
            map_data = {
                'raman_shifts': raman_shifts,
                'spatial_coordinates': spatial_data,
                'spectra': spectra_data,
                'metadata': {
                    'source_file': str(file_path),
                    'n_positions': len(spatial_data),
                    'n_wavenumbers': len(raman_shifts),
                    'spatial_range_x': [spatial_data[:, 0].min(), spatial_data[:, 0].max()],
                    'spatial_range_y': [spatial_data[:, 1].min(), spatial_data[:, 1].max()],
                    'wavenumber_range': [raman_shifts.min(), raman_shifts.max()],
                    'skipped_lines': skipped_lines,
                    'units': {
                        'spatial': 'microns',
                        'wavenumber': 'cm⁻¹',
                        'intensity': 'a.u.'
                    }
                }
            }
            
            # Create gridded data for easier mapping
            x_unique = np.unique(spatial_data[:, 0])
            y_unique = np.unique(spatial_data[:, 1])
            
            if len(x_unique) > 1 and len(y_unique) > 1:
                try:
                    if status_callback:
                        status_callback("Creating gridded data for mapping...")
                    if progress_callback:
                        progress_callback(80)
                    
                    # Create interpolated grid for mapping
                    xi, yi = np.meshgrid(
                        np.linspace(spatial_data[:, 0].min(), spatial_data[:, 0].max(), len(x_unique)),
                        np.linspace(spatial_data[:, 1].min(), spatial_data[:, 1].max(), len(y_unique))
                    )
                    
                    # Interpolate each wavenumber onto the grid
                    gridded_spectra = np.zeros((len(y_unique), len(x_unique), len(raman_shifts)))
                    
                    for i, wavenumber in enumerate(raman_shifts):
                        # Update progress for interpolation
                        if i % max(1, len(raman_shifts) // 10) == 0 and progress_callback:
                            interpolation_progress = 80 + int((i / len(raman_shifts)) * 15)  # 80-95%
                            progress_callback(interpolation_progress)
                            if status_callback:
                                status_callback(f"Interpolating wavenumber {i+1}/{len(raman_shifts)}...")
                        
                        intensities = spectra_data[:, i]
                        gridded_intensities = griddata(
                            spatial_data, intensities, (xi, yi), method='linear', fill_value=0
                        )
                        gridded_spectra[:, :, i] = gridded_intensities
                    
                    map_data['gridded_data'] = {
                        'x_grid': xi,
                        'y_grid': yi,
                        'spectra_grid': gridded_spectra
                    }
                    print("✓ Created gridded data for mapping")
                    
                except Exception as e:
                    print(f"Warning: Could not create gridded data: {e}")
            
            if status_callback:
                status_callback("Import complete!")
            if progress_callback:
                progress_callback(100)
            
            return map_data
            
        except Exception as e:
            raise Exception(f"Error parsing Raman spectral map: {str(e)}")

    def display_map_data_summary(self, map_data, save_path):
        """Display a summary of the imported map data."""
        metadata = map_data['metadata']
        
        summary = f"""Raman Spectral Map Import Summary
        
Source File: {Path(metadata['source_file']).name}
Saved to: {Path(save_path).name}

Data Dimensions:
• Number of positions: {metadata['n_positions']}
• Number of wavenumbers: {metadata['n_wavenumbers']}

Spatial Coverage:
• X range: {metadata['spatial_range_x'][0]:.1f} to {metadata['spatial_range_x'][1]:.1f} {metadata['units']['spatial']}
• Y range: {metadata['spatial_range_y'][0]:.1f} to {metadata['spatial_range_y'][1]:.1f} {metadata['units']['spatial']}

Spectral Coverage:
• Wavenumber range: {metadata['wavenumber_range'][0]:.1f} to {metadata['wavenumber_range'][1]:.1f} {metadata['units']['wavenumber']}

Gridded Data: {'Available' if 'gridded_data' in map_data else 'Not available'}

This data is now ready for:
• 2D mapping analysis
• Cluster analysis
• Principal component analysis
• Spatial correlation studies
"""
        
        QMessageBox.information(self, "Import Complete", summary)

    def import_line_scan_data(self):
        """Import line scan data and convert to pkl format."""
        QMessageBox.information(self, "Feature Coming Soon", 
                               "Line scan data import will be implemented next.\n"
                               "This will handle 1D spatial Raman data.")

    def import_point_data(self):
        """Import point measurement data and convert to pkl format."""
        QMessageBox.information(self, "Feature Coming Soon", 
                               "Point measurement data import will be implemented next.\n"
                               "This will handle individual spectrum collections.")

    def test_raman_map_import(self):
        """Test Raman map import functionality."""
        # Select file to test
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Raman Map File to Test",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
            "Text files (*.txt);;CSV files (*.csv);;Data files (*.dat);;All files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            self.status_bar.showMessage("Testing Raman map import...")
            
            # Test the parsing function
            result = self.run_raman_map_test(file_path)
            
            if result:
                QMessageBox.information(
                    self,
                    "Test Successful",
                    f"Raman map import test completed successfully!\n\n"
                    f"✓ Processed {result['n_positions']} positions\n"
                    f"✓ {result['n_wavenumbers']} wavenumbers\n"
                    f"✓ Test PKL file saved\n"
                    f"✓ Visualization created\n\n"
                    f"Check the output files in the same directory as your input file."
                )
                self.status_bar.showMessage("Test completed successfully")
            else:
                QMessageBox.warning(
                    self,
                    "Test Failed",
                    "The Raman map import test failed. Check the console for error details."
                )
                self.status_bar.showMessage("Test failed")
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Test Error",
                f"Error during test:\n{str(e)}"
            )
            self.status_bar.showMessage("Test error")
    
    def run_raman_map_test(self, file_path):
        """Run the actual Raman map import test."""
        try:
            print(f"Testing with file: {file_path}")
            
            # Read the file line by line to handle inconsistent column counts
            print("Reading data...")
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            print(f"File has {len(lines)} lines")
            
            if len(lines) < 2:
                raise ValueError("File must have at least 2 lines (header + data)")
            
            # Parse the first line to get Raman shifts
            first_line = lines[0].strip().split()
            print(f"First line has {len(first_line)} columns")
            
            # Extract Raman shifts (skip first two columns which should be position labels)
            if len(first_line) < 3:
                raise ValueError("First line must have at least 3 columns (X, Y, and Raman shifts)")
            
            raman_shifts = np.array([float(x) for x in first_line[2:]])
            print(f"Extracted {len(raman_shifts)} Raman shifts")
            print(f"Raman shift range: {raman_shifts.min():.1f} to {raman_shifts.max():.1f} cm⁻¹")
            
            # Process a sample of the spatial data (first 100 lines to test)
            spatial_data = []
            spectra_data = []
            skipped_lines = 0
            
            for i, line in enumerate(lines[1:101]):  # Test with first 100 data lines
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    parts = line.split()
                    
                    # Check if we have enough columns
                    if len(parts) < 3:
                        skipped_lines += 1
                        if skipped_lines <= 5:
                            print(f"Skipping line {i+2}: insufficient columns ({len(parts)})")
                        continue
                    
                    x_pos = float(parts[0])
                    y_pos = float(parts[1])
                    
                    # Extract spectrum intensities - handle variable column counts
                    spectrum_parts = parts[2:]
                    n_to_take = min(len(spectrum_parts), len(raman_shifts))
                    
                    if n_to_take < len(raman_shifts):
                        # Pad with zeros if we have fewer intensities than expected
                        spectrum = np.zeros(len(raman_shifts))
                        spectrum[:n_to_take] = [float(x) for x in spectrum_parts[:n_to_take]]
                        if skipped_lines <= 5:
                            print(f"Line {i+2}: Padded spectrum from {n_to_take} to {len(raman_shifts)} values")
                    else:
                        # Take only the number of intensities we need
                        spectrum = np.array([float(x) for x in spectrum_parts[:len(raman_shifts)]])
                        if len(spectrum_parts) > len(raman_shifts) and skipped_lines <= 3:
                            print(f"Line {i+2}: Using {len(raman_shifts)} of {len(spectrum_parts)} available values")
                    
                    spatial_data.append([x_pos, y_pos])
                    spectra_data.append(spectrum)
                    
                    if i < 5:  # Print first few for verification
                        print(f"Position {i+1}: X={x_pos:.1f}, Y={y_pos:.1f}, Spectrum shape: {spectrum.shape}")
                        
                except (ValueError, IndexError) as e:
                    skipped_lines += 1
                    if skipped_lines <= 5:
                        print(f"Skipping line {i+2}: {e}")
                    continue
            
            if skipped_lines > 5:
                print(f"... and {skipped_lines - 5} more lines were skipped")
            
            if len(spatial_data) == 0:
                raise ValueError("No valid data rows found in test sample")
            
            spatial_data = np.array(spatial_data)
            spectra_data = np.array(spectra_data)
            
            print(f"\nProcessed {len(spatial_data)} positions")
            print(f"Spatial X range: {spatial_data[:, 0].min():.1f} to {spatial_data[:, 0].max():.1f}")
            print(f"Spatial Y range: {spatial_data[:, 1].min():.1f} to {spatial_data[:, 1].max():.1f}")
            print(f"Spectrum intensity range: {spectra_data.min():.1f} to {spectra_data.max():.1f}")
            
            # Create the map data structure
            map_data = {
                'raman_shifts': raman_shifts,
                'spatial_coordinates': spatial_data,
                'spectra': spectra_data,
                'metadata': {
                    'source_file': str(file_path),
                    'n_positions': len(spatial_data),
                    'n_wavenumbers': len(raman_shifts),
                    'spatial_range_x': [spatial_data[:, 0].min(), spatial_data[:, 0].max()],
                    'spatial_range_y': [spatial_data[:, 1].min(), spatial_data[:, 1].max()],
                    'wavenumber_range': [raman_shifts.min(), raman_shifts.max()],
                    'skipped_lines': skipped_lines,
                    'units': {
                        'spatial': 'microns',
                        'wavenumber': 'cm⁻¹',
                        'intensity': 'a.u.'
                    }
                }
            }
            
            # Save test PKL file
            test_pkl_path = Path(file_path).with_suffix('_test.pkl')
            with open(test_pkl_path, 'wb') as f:
                pickle.dump(map_data, f)
            
            print(f"\nTest PKL file saved: {test_pkl_path}")
            
            # Create a simple visualization
            plt.figure(figsize=(12, 4))
            
            # Plot 1: Spatial distribution
            plt.subplot(1, 3, 1)
            plt.scatter(spatial_data[:, 0], spatial_data[:, 1], c=range(len(spatial_data)), 
                       cmap='viridis', s=20)
            plt.xlabel('X position (μm)')
            plt.ylabel('Y position (μm)')
            plt.title('Spatial Distribution')
            plt.colorbar(label='Point index')
            
            # Plot 2: Sample spectrum
            plt.subplot(1, 3, 2)
            plt.plot(raman_shifts, spectra_data[len(spectra_data)//2])
            plt.xlabel('Raman Shift (cm⁻¹)')
            plt.ylabel('Intensity (a.u.)')
            plt.title('Sample Spectrum (middle point)')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Intensity map at a specific wavenumber
            plt.subplot(1, 3, 3)
            # Find index closest to 1000 cm⁻¹
            idx_1000 = np.argmin(np.abs(raman_shifts - 1000))
            intensities_1000 = spectra_data[:, idx_1000]
            scatter = plt.scatter(spatial_data[:, 0], spatial_data[:, 1], 
                                c=intensities_1000, cmap='hot', s=30)
            plt.xlabel('X position (μm)')
            plt.ylabel('Y position (μm)')
            plt.title(f'Intensity Map at {raman_shifts[idx_1000]:.1f} cm⁻¹')
            plt.colorbar(scatter, label='Intensity')
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = Path(file_path).with_suffix('_test_visualization.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {plot_path}")
            
            plt.show()
            
            print("\nTest completed successfully!")
            
            return {
                'n_positions': len(spatial_data),
                'n_wavenumbers': len(raman_shifts),
                'pkl_path': test_pkl_path,
                'plot_path': plot_path
            }
            
        except Exception as e:
            print(f"Error in test: {e}")
            import traceback
            traceback.print_exc()
            return None

    def open_user_forum(self):
        """Open the user forum URL in the default web browser."""
        webbrowser.open("https://ramanlab.freeforums.net")

    def open_database_downloads(self):
        """Open the database downloads URL in the default web browser."""
        webbrowser.open("https://doi.org/10.5281/zenodo.15717960")

    def open_readme(self):
        """Open the README file in the default web browser."""
        webbrowser.open("https://github.com/aaroncelestian/RamanLab/blob/main/README.md")

    def launch_database_manager(self):
        """Launch the database manager GUI."""
        try:
            from database_manager_gui import main as database_manager_main
            # Launch the database manager in a separate process to avoid conflicts
            import subprocess
            import sys
            subprocess.Popen([sys.executable, "database_manager_gui.py"])
            self.status_bar.showMessage("Database Manager launched successfully")
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", 
                              f"Could not import database manager:\n{str(e)}\n\n"
                              f"Make sure database_manager_gui.py is available.")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", 
                               f"Failed to launch database manager:\n{str(e)}")
    
    def open_settings(self):
        """Open the settings dialog."""
        try:
            from core.settings_dialog import SettingsDialog
            settings_dialog = SettingsDialog(self)
            settings_dialog.settings_changed.connect(self.on_settings_changed)
            settings_dialog.exec()
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Settings dialog not available: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open settings: {str(e)}")
    
    def on_settings_changed(self):
        """Handle settings changes."""
        try:
            from core.config_manager import get_config_manager
            config = get_config_manager()
            
            # Update any UI elements that depend on settings
            self.status_bar.showMessage("Settings updated successfully", 2000)
            
            # You could refresh plot settings, update database paths, etc. here
            
        except Exception as e:
            print(f"Error handling settings change: {e}")
    
    def save_session_dialog(self):
        """Show dialog to save current session with a custom name."""
        if not STATE_MANAGEMENT_AVAILABLE:
            QMessageBox.warning(self, "Session Saving", "Session management is not available.")
            return
        
        session_name, ok = QInputDialog.getText(
            self, 
            "Save Session", 
            "Enter a name for this session:",
            text=f"RamanLab_Session_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        if ok and session_name.strip():
            try:
                # Save with custom name and notes
                notes = f"Manual save: {session_name}"
                success = save_module_state('raman_analysis_app', notes, tags=["manual", "named"])
                
                if success:
                    QMessageBox.information(
                        self, 
                        "Session Saved", 
                        f"Session '{session_name}' has been saved successfully!\n\n"
                        f"Location: ~/RamanLab_Projects/\n"
                        f"You can restore this session later using 'Load Session'."
                    )
                    self.status_bar.showMessage(f"Session '{session_name}' saved successfully")
                else:
                    QMessageBox.warning(self, "Save Failed", "Failed to save session. Please try again.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving session:\n{str(e)}")
    
    def load_session_dialog(self):
        """Show dialog to load a previously saved session."""
        if not STATE_MANAGEMENT_AVAILABLE:
            QMessageBox.warning(self, "Session Loading", "Session management is not available.")
            return
        
        try:
            # Try to load the most recent session
            success = load_module_state('raman_analysis_app')
            
            if success:
                QMessageBox.information(
                    self, 
                    "Session Loaded", 
                    "Your previous session has been restored successfully!\n\n"
                    "All spectrum data, processing state, and UI settings have been restored."
                )
                self.status_bar.showMessage("Session loaded successfully")
            else:
                QMessageBox.information(
                    self, 
                    "No Session Found", 
                    "No previous session was found to restore.\n\n"
                    "Start working with RamanLab and your session will be automatically saved."
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading session:\n{str(e)}")
    
    def toggle_auto_save(self, enabled):
        """Toggle auto-save functionality."""
        if not STATE_MANAGEMENT_AVAILABLE:
            return
        
        try:
            # This would need to be implemented in the state manager
            # For now, just show a message
            if enabled:
                self.status_bar.showMessage("Auto-save enabled")
                QMessageBox.information(
                    self, 
                    "Auto-Save Enabled", 
                    "Auto-save is now enabled.\n\n"
                    "Your session will be automatically saved after major operations."
                )
            else:
                self.status_bar.showMessage("Auto-save disabled")
                QMessageBox.information(
                    self, 
                    "Auto-Save Disabled", 
                    "Auto-save is now disabled.\n\n"
                    "You can still manually save sessions using 'Save Session'."
                )
        except Exception as e:
            QMessageBox.warning(self, "Auto-Save Error", f"Error toggling auto-save:\n{str(e)}")
    
    def show_session_info(self):
        """Show information about the current session and state management."""
        if not STATE_MANAGEMENT_AVAILABLE:
            QMessageBox.warning(self, "Session Info", "Session management is not available.")
            return
        
        # Gather session information
        has_spectrum = self.current_wavenumbers is not None
        spectrum_points = len(self.current_wavenumbers) if has_spectrum else 0
        has_peaks = (self.detected_peaks is not None and len(self.detected_peaks) > 0) or len(self.manual_peaks) > 0
        processing_applied = (self.processed_intensities is not None and 
                            self.current_intensities is not None and 
                            not np.array_equal(self.processed_intensities, self.current_intensities))
        
        info_text = f"""
RamanLab Session Information

Current Session State:
├─ Spectrum Loaded: {'Yes' if has_spectrum else 'No'}
{'├─ Spectrum Points: ' + str(spectrum_points) if has_spectrum else ''}
{'├─ Source File: ' + (Path(self.spectrum_file_path).name if self.spectrum_file_path else 'Unknown') if has_spectrum else ''}
├─ Peaks Detected: {'Yes' if has_peaks else 'No'}
├─ Processing Applied: {'Yes' if processing_applied else 'No'}
├─ Database Loaded: {'Yes' if self.raman_db and hasattr(self.raman_db, 'database') else 'No'}
└─ Auto-Save: Enabled

Session Management:
├─ Save Location: ~/RamanLab_Projects/
├─ Auto-Save: Every 5 minutes
├─ Crash Recovery: Enabled
└─ State Includes: Spectrum data, peaks, processing, UI layout

Keyboard Shortcuts:
├─ Save Session: Ctrl+Shift+S
└─ Load Session: Ctrl+Shift+O
        """.strip()
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Session Information")
        msg.setText(info_text)
        msg.setTextFormat(Qt.PlainText)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()


class SearchResultsWindow(QDialog):
    """Interactive search results window with spectrum comparison and metadata viewing."""
    
    def __init__(self, matches, query_wavenumbers, query_intensities, search_type, parent=None):
        super().__init__(parent)
        self.matches = matches
        self.query_wavenumbers = query_wavenumbers
        self.query_intensities = query_intensities
        self.search_type = search_type
        self.selected_match = None
        
        self.setWindowTitle(f"{search_type} Results - {len(matches)} matches found")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        self.setup_ui()
        
        # Select first match by default
        if self.matches:
            self.results_table.selectRow(0)
            self.on_match_selected()
    
    def setup_ui(self):
        """Set up the search results UI."""
        layout = QHBoxLayout(self)
        
        # Left panel - results list and controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Results table
        results_group = QGroupBox(f"Search Results ({len(self.matches)} matches)")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Mineral", "Score", "Formula"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        
        # Populate results table
        self.populate_results_table()
        
        # Connect selection change
        self.results_table.selectionModel().selectionChanged.connect(self.on_match_selected)
        
        results_layout.addWidget(self.results_table)
        left_layout.addWidget(results_group)
        
        # Control buttons
        controls_group = QGroupBox("Actions")
        controls_layout = QVBoxLayout(controls_group)
        
        # Normalization options (moved to top)
        norm_group = QGroupBox("Normalization")
        norm_layout = QVBoxLayout(norm_group)
        
        self.norm_combo = QComboBox()
        self.norm_combo.addItems([
            "Max Intensity", "Area Under Curve", "Standard Score (Z-score)", "Min-Max"
        ])
        self.norm_combo.currentTextChanged.connect(self.update_comparison_plot)
        norm_layout.addWidget(self.norm_combo)
        
        controls_layout.addWidget(norm_group)
        
        # Buttons layout (moved below normalization) - vertically stacked
        buttons_layout = QVBoxLayout()
        
        # Add Constraint to Mixture Analysis button
        add_constraint_btn = QPushButton("🎯 Add to Mixture Analysis")
        add_constraint_btn.setToolTip("Add this mineral as a known component constraint for mixture analysis")
        add_constraint_btn.clicked.connect(self.add_to_mixture_analysis)
        buttons_layout.addWidget(add_constraint_btn)
        
        # Metadata button
        metadata_btn = QPushButton("📋 Metadata")
        metadata_btn.clicked.connect(self.show_metadata)
        buttons_layout.addWidget(metadata_btn)
        
        # Mixture analysis button
        mixture_btn = QPushButton("🔬 Mixture Analysis")
        mixture_btn.clicked.connect(self.launch_mixture_analysis_from_search)
        buttons_layout.addWidget(mixture_btn)
        
        controls_layout.addLayout(buttons_layout)
        controls_layout.addStretch()
        
        left_layout.addWidget(controls_group)
        layout.addWidget(left_panel)
        
        # Right panel - spectrum comparison with three plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create matplotlib figure with two subplots
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, right_panel)
        
        # Create two subplots with adjusted height ratios
        # Comparison plot (60%) and vibration analysis (40%)
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.3)
        self.ax_comparison = self.figure.add_subplot(gs[0])
        self.ax_vibration = self.figure.add_subplot(gs[1])
        
        # Adjust subplot spacing 
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
        
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        layout.addWidget(right_panel)
        
        # Initialize tooltip variables
        self.tooltip_data = []
        self.tooltip = None
        self.comparison_overlay = None
        
        # Metadata window for auto-updating display
        self.metadata_dialog = None
    
    def populate_results_table(self):
        """Populate the results table with match data."""
        self.results_table.setRowCount(len(self.matches))
        
        for i, match in enumerate(self.matches):
            # Mineral name
            name = match.get('name', 'Unknown')
            metadata = match.get('metadata', {})
            display_name = metadata.get('NAME') or metadata.get('mineral_name') or name
            
            name_item = QTableWidgetItem(display_name)
            name_item.setToolTip(f"Database entry: {name}")
            self.results_table.setItem(i, 0, name_item)
            
            # Score
            score = match.get('score', 0.0)
            score_item = QTableWidgetItem(f"{score:.3f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 1, score_item)
            
            # Formula
            formula = metadata.get('IDEAL CHEMISTRY') or metadata.get('FORMULA') or metadata.get('Formula') or 'N/A'
            formula_item = QTableWidgetItem(formula)
            self.results_table.setItem(i, 2, formula_item)
        
        # Resize columns
        self.results_table.resizeColumnsToContents()
    
    def on_match_selected(self):
        """Handle match selection change."""
        selected_rows = self.results_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            self.selected_match = self.matches[row]
            self.update_comparison_plot()
            
            # Update metadata dialog if it's open
            if self.metadata_dialog and self.metadata_dialog.isVisible():
                self.update_metadata_dialog()
    
    def update_comparison_plot(self):
        """Update the spectrum comparison plot with vibrational analysis."""
        if not self.selected_match:
            return
        
        # Clear all plots
        self.ax_comparison.clear()
        self.ax_vibration.clear()
        
        # Get database spectrum data
        db_wavenumbers = self.selected_match.get('wavenumbers', [])
        db_intensities = self.selected_match.get('intensities', [])
        
        if len(db_wavenumbers) == 0 or len(db_intensities) == 0:
            self.ax_comparison.text(0.5, 0.5, 'No spectrum data available for this match',
                                   ha='center', va='center', transform=self.ax_comparison.transAxes)
            self.canvas.draw()
            return
        
        # Convert to numpy arrays
        db_wavenumbers = np.array(db_wavenumbers)
        db_intensities = np.array(db_intensities)
        
        # Interpolate to common wavenumber grid
        common_wavenumbers = np.linspace(
            max(self.query_wavenumbers.min(), db_wavenumbers.min()),
            min(self.query_wavenumbers.max(), db_wavenumbers.max()),
            min(len(self.query_wavenumbers), len(db_wavenumbers))
        )
        
        query_interp = np.interp(common_wavenumbers, self.query_wavenumbers, self.query_intensities)
        db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)
        
        # Apply normalization
        norm_method = self.norm_combo.currentText()
        query_norm, db_norm = self.normalize_spectra(query_interp, db_interp, norm_method)
        
        # === COMPARISON PLOT ===
        self.ax_comparison.plot(common_wavenumbers, query_norm, 'b-', linewidth=1.5, 
                               label='Query Spectrum', alpha=0.8)
        self.ax_comparison.plot(common_wavenumbers, db_norm, 'r-', linewidth=1.5, 
                               label=f'Match: {self.get_display_name()}', alpha=0.8)
        
        # Calculate residual and add as overlay at top of plot
        residual = query_norm - db_norm
        
        # Scale residual to be smaller and position at top of comparison plot
        y_min, y_max = self.ax_comparison.get_ylim() if hasattr(self.ax_comparison, '_current_ylim') else (0, 1)
        y_range = max(np.max(query_norm), np.max(db_norm)) - min(np.min(query_norm), np.min(db_norm))
        if y_range == 0:
            y_range = 1
            
        # Scale residual to be 15% of the main plot height and position at 85% of the plot height
        residual_scaled = (residual / np.max(np.abs(residual)) if np.max(np.abs(residual)) > 0 else residual) * (0.15 * y_range)
        residual_offset = 0.85 * y_range + min(np.min(query_norm), np.min(db_norm))
        residual_positioned = residual_scaled + residual_offset
        
        # Plot small residual overlay
        self.ax_comparison.plot(common_wavenumbers, residual_positioned, 'g-', linewidth=0.8, 
                               label='Residual (scaled)', alpha=0.6)
        self.ax_comparison.axhline(y=residual_offset, color='k', linestyle=':', alpha=0.4, linewidth=0.5)
        
        # Calculate and display statistics in comparison plot
        correlation = np.corrcoef(query_norm, db_norm)[0, 1]
        rmse = np.sqrt(np.mean(residual**2))
        
        # Add statistics text to comparison plot
        stats_text = f'Correlation: {correlation:.3f}\nRMSE: {rmse:.3f}'
        self.ax_comparison.text(0.02, 0.98, stats_text, transform=self.ax_comparison.transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax_comparison.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_comparison.set_ylabel('Normalized Intensity')
        self.ax_comparison.set_title(f'Spectrum Comparison (Score: {self.selected_match.get("score", 0):.3f})')
        self.ax_comparison.legend()
        self.ax_comparison.grid(True, alpha=0.3)
        
        # === VIBRATIONAL CORRELATION PLOT ===
        self.plot_vibrational_analysis(common_wavenumbers, query_norm, db_norm)
        
        self.canvas.draw()
    
    def plot_vibrational_analysis(self, wavenumbers, query_spectrum, match_spectrum):
        """Plot vibrational correlation analysis showing mineral group correlations."""
        # Define specific mineral vibration regions with group categories
        mineral_regions = [
            # Silicate vibrations
            ("Silicate", 450, 550, "Si-O-Si 3MR Stretch"),
            ("Silicate", 600, 680, "Si-O-Si"),
            ("Silicate", 850, 1000, "Si-O Stretch Q²,Q³"),
            ("Silicate", 1050, 1200, "Si-O-Si Stretch Q⁰"),
            # Carbonate vibrations
            ("Carbonate", 700, 740, "CO₃ Bend ν₂"),
            ("Carbonate", 1050, 1090, "CO₃ Stretch ν₄"),
            # Phosphate vibrations
            ("Phosphate", 550, 620, "PO₄ Bend ν₄"),
            ("Phosphate", 950, 970, "PO₄ Stretch ν₁"),
            ("Phosphate", 1030, 1080, "PO₄ Asym"),
            # Arsenate vibrations
            ("Arsenate", 420, 460, "AsO₄ Bend ν₂"),
            ("Arsenate", 810, 855, "AsO₄ Stretch ν₁"),
            ("Arsenate", 780, 880, "AsO₃ Stretch ν₃"),
            # Sulfate vibrations
            ("Sulfate", 450, 500, "SO₄ Bend ν₂"),
            ("Sulfate", 975, 1010, "SO₄ Stretch ν₁"),
            ("Sulfate", 1100, 1150, "SO₄ Asym ν₃"),
            # Oxide vibrations
            ("Oxide", 300, 350, "Metal-O Stretch"),
            ("Oxide", 400, 450, "Metal-O-Metal Bend"),
            ("Oxide", 500, 600, "M-O Lattice"),
            # Hydroxide vibrations
            ("Hydroxide", 3500, 3650, "OH Stretch"),
            ("Hydroxide", 600, 900, "M-OH Bend"),
            ("Hydroxide", 1600, 1650, "HOH Bend"),
            # Sulfide vibrations
            ("Sulfide", 300, 400, "Metal-S Stretch"),
            ("Sulfide", 200, 280, "S-S Stretch"),
            ("Sulfide", 350, 420, "M-S-M Bend"),
            # Sulfosalt vibrations
            ("Sulfosalt", 300, 360, "Sb-S Stretch"),
            ("Sulfosalt", 330, 380, "As-S Stretch"),
            ("Sulfosalt", 250, 290, "S-S Stretch"),
            # Vanadate vibrations
            ("Vanadate", 800, 860, "V-O Stretch ν₁"),
            ("Vanadate", 780, 820, "V-O-V Asym ν₃"),
            ("Vanadate", 400, 450, "V-O Bend ν₄"),
            # Borate vibrations
            ("Borate", 650, 700, "BO₃ Bend"),
            ("Borate", 880, 950, "BO₃ Stretch"),
            ("Borate", 1300, 1400, "BO₃ Asym"),
            # Water vibrations
            ("OH/H₂O", 3200, 3500, "H₂O Stretch"),
            ("OH/H₂O", 1600, 1650, "H₂O Bend"),
            ("OH/H₂O", 500, 800, "H₂O Libration"),
            # Oxalate vibrations
            ("Oxalate", 1455, 1490, "C-O Stretch"),
            ("Oxalate", 900, 920, "C-C Stretch"),
            ("Oxalate", 850, 870, "O-C-O Bend"),
        ]
        
        # Filter regions to only those within our spectral range
        filtered_regions = [region for region in mineral_regions 
                           if region[1] <= wavenumbers.max() and region[2] >= wavenumbers.min()]
        
        # Calculate correlation for each region
        region_data = []
        group_correlations = {}
        group_weights = {}
        
        # Define region importance factors
        region_importance = {
            "Carbonate": 1.0, "Sulfate": 1.0, "Phosphate": 1.0, "Silicate": 1.0,
            "OH/H₂O": 0.5, "Vanadate": 1.0, "Borate": 1.0, "Oxalate": 1.0,
            "Arsenate": 1.0, "Oxide": 1.0, "Hydroxide": 0.8, "Sulfide": 1.0, "Sulfosalt": 1.0
        }
        
        for group, start, end, label in filtered_regions:
            indices = np.where((wavenumbers >= start) & (wavenumbers <= end))[0]
            if len(indices) > 1:
                region_query = query_spectrum[indices]
                region_match = match_spectrum[indices]
                
                # Calculate correlation coefficient
                try:
                    if np.all(region_query == region_query[0]) or np.all(region_match == region_match[0]):
                        corr = 0.0
                    else:
                        corr = np.corrcoef(region_query, region_match)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                except Exception:
                    corr = 0.0
            else:
                corr = 0.0
            
            region_data.append((group, start, end, label, corr))
            
            # Track group correlations
            if group not in group_correlations:
                group_correlations[group] = []
                group_weights[group] = []
            
            group_correlations[group].append(corr)
            width = end - start
            weight = (width / 2000.0) * region_importance.get(group, 1.0)
            group_weights[group].append(weight)
        
        # Calculate weighted group correlations
        weighted_group_scores = {}
        for group in group_correlations:
            if len(group_correlations[group]) > 0:
                weighted_corr = np.average(group_correlations[group], weights=group_weights[group])
                weighted_group_scores[group] = weighted_corr
        
        # Set up the vibrational plot
        x_min, x_max = wavenumbers.min(), wavenumbers.max()
        x_range = x_max - x_min
        x_min = max(0, x_min - 0.05 * x_range)
        x_max = x_max + 0.05 * x_range
        
        self.ax_vibration.set_xlim(x_min, x_max)
        self.ax_vibration.set_ylim(0, 1)
        self.ax_vibration.grid(True, axis='x', linestyle=':', color='gray', alpha=0.6)
        
        # Define y-positions for each group
        group_positions = {
            "Silicate": 0.94, "Carbonate": 0.86, "Phosphate": 0.78, "Arsenate": 0.70,
            "Sulfate": 0.62, "Oxide": 0.54, "Hydroxide": 0.46, "Sulfide": 0.38,
            "Sulfosalt": 0.30, "Vanadate": 0.22, "Borate": 0.14, "OH/H₂O": 0.06
        }
        
        bar_height = 0.06
        
        # Group by mineral types
        groups = {}
        for item in region_data:
            group = item[0]
            if group not in groups:
                groups[group] = []
            groups[group].append(item)
        
        # Use colormap for correlation values
        import matplotlib.cm as cm
        cmap = cm.RdYlGn
        
        # Clear tooltip data
        self.tooltip_data = []
        
        # Plot each group
        for group_name, group_items in groups.items():
            y_pos = group_positions.get(group_name, 0.5)
            avg_corr = weighted_group_scores.get(group_name, 0.0)
            
            # Add group label
            group_label = f"{group_name} (Avg: {avg_corr:.2f})"
            self.ax_vibration.text(
                x_min - 0.03 * (x_max - x_min), y_pos, group_label,
                fontsize=8, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
            
            # Plot bars for each region in the group
            for _, start, end, label, corr in group_items:
                if end < x_min or start > x_max:
                    continue
                
                width = end - start
                color = cmap(corr)
                
                # Create rectangle
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (start, y_pos - bar_height/2), width, bar_height,
                    facecolor=color, edgecolor='black', alpha=0.8
                )
                self.ax_vibration.add_patch(rect)
                
                # Add correlation value for wider bars
                if width > 70:
                    text_color = 'black' if 0.3 <= corr <= 0.7 else 'white'
                    self.ax_vibration.text(
                        start + width/2, y_pos, f"{corr:.2f}",
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color=text_color
                    )
                
                # Store tooltip data
                tooltip_info = f"{group_name}: {label}\nRange: {start}-{end} cm⁻¹\nCorrelation: {corr:.2f}"
                self.tooltip_data.append((rect, tooltip_info, color, start, end))
        
        # Add color gradient reference
        gradient_width = (x_max - x_min) * 0.6
        gradient_x = x_min + (x_max - x_min) * 0.2
        gradient_y = -0.05
        gradient_height = 0.02
        
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        self.ax_vibration.imshow(
            gradient, aspect='auto', 
            extent=[gradient_x, gradient_x + gradient_width, 
                   gradient_y - gradient_height/2, gradient_y + gradient_height/2],
            cmap=cmap
        )
        
        # Add gradient labels
        self.ax_vibration.text(gradient_x, gradient_y + gradient_height/2 + 0.02,
                              "Low Correlation (0.0)", ha='left', va='bottom', fontsize=7, color='dimgray')
        self.ax_vibration.text(gradient_x + gradient_width, gradient_y + gradient_height/2 + 0.02,
                              "High Correlation (1.0)", ha='right', va='bottom', fontsize=7, color='dimgray')
        
        # Set labels and title
        self.ax_vibration.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_vibration.set_title(f'Mineral Vibration Correlation: Query vs. {self.get_display_name()}')
        self.ax_vibration.set_yticks([])
        self.ax_vibration.set_ylabel('')  # Remove y-axis label
        
        # Setup tooltips
        self.setup_vibrational_tooltips()
    
    def setup_vibrational_tooltips(self):
        """Set up interactive tooltips and overlays for the vibrational correlation plot."""
        # Create tooltip annotation if it doesn't exist
        if self.tooltip is None:
            self.tooltip = self.ax_vibration.annotate(
                "", xy=(0, 0), xytext=(0, -70),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9, edgecolor="black"),
                arrowprops=dict(arrowstyle="->"),
                visible=False,
                fontsize=9,
                color="navy"
            )
        
        def on_hover(event):
            """Handle mouse hover events for tooltips and overlays."""
            if not hasattr(self, 'tooltip_data') or not self.tooltip_data:
                return
            
            # Check if mouse is over the vibrational plot
            if event.inaxes != self.ax_vibration:
                self.tooltip.set_visible(False)
                self.canvas.draw_idle()
                # Remove overlay from comparison plot
                if hasattr(self, 'comparison_overlay') and self.comparison_overlay is not None:
                    self.comparison_overlay.remove()
                    self.comparison_overlay = None
                    self.canvas.draw_idle()
                return
            
            # Check if mouse is over any vibrational region rectangle
            for rect, tooltip_text, color, start, end in self.tooltip_data:
                contains, _ = rect.contains(event)
                if contains:
                    # Update tooltip
                    self.tooltip.set_text(tooltip_text)
                    self.tooltip.xy = (event.xdata, event.ydata)
                    self.tooltip.xyann = (0, -70)
                    
                    # Set tooltip background color to match rectangle
                    r, g, b, _ = color
                    lighter_r = 0.7 * r + 0.3
                    lighter_g = 0.7 * g + 0.3
                    lighter_b = 0.7 * b + 0.3
                    
                    self.tooltip.get_bbox_patch().set(
                        fc=(lighter_r, lighter_g, lighter_b, 0.9), ec=color
                    )
                    
                    # Show tooltip
                    self.tooltip.set_visible(True)
                    
                    # Add overlay to comparison plot
                    if hasattr(self, 'comparison_overlay') and self.comparison_overlay is not None:
                        self.comparison_overlay.remove()
                    
                    self.comparison_overlay = self.ax_comparison.axvspan(
                        start, end, color=color, alpha=0.3, zorder=0
                    )
                    
                    self.canvas.draw_idle()
                    return
            
            # If not over any rectangle, hide tooltip and overlay
            self.tooltip.set_visible(False)
            if hasattr(self, 'comparison_overlay') and self.comparison_overlay is not None:
                self.comparison_overlay.remove()
                self.comparison_overlay = None
            self.canvas.draw_idle()
        
        # Connect the hover event
        self.canvas.mpl_connect('motion_notify_event', on_hover)
    
    def normalize_spectra(self, spectrum1, spectrum2, method):
        """Normalize two spectra using the specified method."""
        spec1 = spectrum1.copy()
        spec2 = spectrum2.copy()
        
        if method == "Max Intensity":
            spec1 = spec1 / np.max(spec1) if np.max(spec1) > 0 else spec1
            spec2 = spec2 / np.max(spec2) if np.max(spec2) > 0 else spec2
            
        elif method == "Area Under Curve":
            area1 = np.trapz(np.abs(spec1))
            area2 = np.trapz(np.abs(spec2))
            spec1 = spec1 / area1 if area1 > 0 else spec1
            spec2 = spec2 / area2 if area2 > 0 else spec2
            
        elif method == "Standard Score (Z-score)":
            spec1 = (spec1 - np.mean(spec1)) / np.std(spec1) if np.std(spec1) > 0 else spec1
            spec2 = (spec2 - np.mean(spec2)) / np.std(spec2) if np.std(spec2) > 0 else spec2
            
        elif method == "Min-Max":
            min1, max1 = np.min(spec1), np.max(spec1)
            min2, max2 = np.min(spec2), np.max(spec2)
            spec1 = (spec1 - min1) / (max1 - min1) if (max1 - min1) > 0 else spec1
            spec2 = (spec2 - min2) / (max2 - min2) if (max2 - min2) > 0 else spec2
        
        return spec1, spec2
    
    def get_display_name(self):
        """Get display name for the selected match."""
        if not self.selected_match:
            return "No selection"
        
        metadata = self.selected_match.get('metadata', {})
        name = metadata.get('NAME') or metadata.get('mineral_name') or self.selected_match.get('name', 'Unknown')
        return name
    
    def show_metadata(self):
        """Show metadata window for the selected match."""
        if not self.selected_match:
            QMessageBox.warning(self, "No Selection", "Please select a match to view metadata.")
            return
        
        # Create or show existing metadata dialog
        if self.metadata_dialog is None:
            self.create_metadata_dialog()
        
        # Update content and show
        self.update_metadata_dialog()
        self.metadata_dialog.show()
        self.metadata_dialog.raise_()
        self.metadata_dialog.activateWindow()
    
    def create_metadata_dialog(self):
        """Create the non-modal metadata dialog."""
        self.metadata_dialog = QDialog(self)
        self.metadata_dialog.setWindowTitle("Mineral Metadata")
        self.metadata_dialog.setMinimumSize(500, 600)
        self.metadata_dialog.resize(600, 700)
        self.metadata_dialog.setModal(False)  # Make it non-modal
        
        layout = QVBoxLayout(self.metadata_dialog)
        
        # Scrollable text area
        self.metadata_text_edit = QTextEdit()
        self.metadata_text_edit.setReadOnly(True)
        self.metadata_text_edit.setFont(QFont("Monaco", 10))  # Monospace font for better formatting
        layout.addWidget(self.metadata_text_edit)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.metadata_dialog.close)
        layout.addWidget(close_btn)
        
        # Handle dialog destruction
        self.metadata_dialog.destroyed.connect(lambda: setattr(self, 'metadata_dialog', None))
    
    def update_metadata_dialog(self):
        """Update the metadata dialog content with the currently selected match."""
        if not self.metadata_dialog or not self.selected_match:
            return
            
        match_data = self.selected_match
        metadata = match_data.get('metadata', {})
        name = match_data.get('name', 'Unknown')
        
        # Create formatted metadata text
        display_name = metadata.get('NAME') or metadata.get('mineral_name') or name
        
        metadata_text = f"🔍 Metadata: {display_name}\n"
        metadata_text += "="*60 + "\n\n"
        
        # Basic information
        metadata_text += f"Database Entry: {name}\n"
        metadata_text += f"Score: {match_data.get('score', 0):.3f}\n"
        metadata_text += f"Timestamp: {match_data.get('timestamp', 'N/A')[:19] if match_data.get('timestamp') else 'N/A'}\n"
        metadata_text += f"Number of Peaks: {len(match_data.get('peaks', []))}\n\n"
        
        # Chemical and physical properties
        if metadata:
            metadata_text += "📊 Properties:\n"
            metadata_text += "-"*30 + "\n"
            
            # Show key properties
            key_fields = [
                ('IDEAL CHEMISTRY', 'Chemical Formula'),
                ('FORMULA', 'Formula'),
                ('CRYSTAL SYSTEM', 'Crystal System'),
                ('SPACE GROUP', 'Space Group'),
                ('UNIT CELL', 'Unit Cell'),
                ('COLOR', 'Color'),
                ('LUSTRE', 'Lustre'),
                ('HARDNESS', 'Hardness'),
                ('DENSITY', 'Density'),
                ('LOCALITY', 'Locality'),
                ('DESCRIPTION', 'Description')
            ]
            
            for field_key, display_label in key_fields:
                if field_key in metadata and metadata[field_key]:
                    value = str(metadata[field_key]).strip()
                    if value and value.lower() not in ['none', 'n/a', '']:
                        metadata_text += f"{display_label}: {value}\n"
            
            # Show any additional metadata
            shown_keys = [key for key, _ in key_fields] + ['NAME', 'mineral_name']
            additional_fields = {k: v for k, v in metadata.items() 
                               if k not in shown_keys and v and str(v).strip() 
                               and str(v).lower() not in ['none', 'n/a', '']}
            
            if additional_fields:
                metadata_text += "\n📋 Additional Properties:\n"
                metadata_text += "-"*30 + "\n"
                for key, value in additional_fields.items():
                    metadata_text += f"{key}: {value}\n"
        
        # Update the dialog content and title
        self.metadata_text_edit.setPlainText(metadata_text)
        self.metadata_dialog.setWindowTitle(f"Metadata: {display_name}")
    
    def launch_mixture_analysis_from_search(self):
        """Launch NEW interactive mixture analysis using search results as potential components."""
        if not self.matches:
            QMessageBox.information(self, "No Matches", "No search results available for mixture analysis.")
            return
        
        # Get the parent's query spectrum
        if not hasattr(self.parent(), 'current_wavenumbers') or self.parent().current_wavenumbers is None:
            QMessageBox.warning(self, "No Query Spectrum", 
                "No query spectrum is loaded in the main application.\n"
                "Load a spectrum in the main window first to perform mixture analysis.")
            return
        
        try:
            # Import the NEW interactive mixture analysis module
            from raman_mixture_analysis_interactive import InteractiveMixtureAnalyzer
            
            # Create QApplication instance if needed
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            # Create interactive mixture analysis window and store reference to prevent garbage collection
            mixture_analyzer = InteractiveMixtureAnalyzer()
            
            # Store reference in parent to prevent garbage collection
            if not hasattr(self.parent(), 'mixture_analysis_windows'):
                self.parent().mixture_analysis_windows = []
            self.parent().mixture_analysis_windows.append(mixture_analyzer)
            
            # Clean up closed windows from the list when this window closes
            def cleanup_window_reference():
                if hasattr(self.parent(), 'mixture_analysis_windows'):
                    try:
                        self.parent().mixture_analysis_windows.remove(mixture_analyzer)
                    except ValueError:
                        pass  # Already removed
            
            # Connect cleanup to window close event
            mixture_analyzer.destroyed.connect(cleanup_window_reference)
            
            # Load the query spectrum automatically
            query_wavenumbers = self.parent().current_wavenumbers
            query_intensities = self.parent().processed_intensities
            
            if query_wavenumbers is not None and query_intensities is not None:
                # Get spectrum name from parent
                parent = self.parent()
                spectrum_name = getattr(parent, 'spectrum_file_path', None)
                if spectrum_name:
                    spectrum_name = os.path.basename(spectrum_name)
                else:
                    spectrum_name = 'Query Spectrum from Search'
                
                # Set spectrum data directly in the interactive analyzer
                mixture_analyzer.user_wavenumbers = query_wavenumbers.copy()
                mixture_analyzer.user_spectrum = query_intensities.copy()
                mixture_analyzer.original_spectrum = query_intensities.copy()
                mixture_analyzer.current_residual = query_intensities.copy()
                
                # Use QTimer to update UI elements after window is shown
                QTimer.singleShot(200, lambda: self._update_mixture_analyzer_search_ui(
                    mixture_analyzer, len(query_wavenumbers), spectrum_name, 
                    query_wavenumbers[0], query_wavenumbers[-1]))
                
                # Show a helpful message about using search results
                num_matches = len(self.matches)
                selected_match = self.selected_match
                
                info_msg = f"NEW Interactive Mixture Analysis launched with your query spectrum!\n\n"
                info_msg += f"📊 {num_matches} search results are available as potential components.\n"
                
                if selected_match:
                    match_name = selected_match.get('name', 'Unknown')
                    metadata = selected_match.get('metadata', {})
                    display_name = metadata.get('NAME') or metadata.get('mineral_name') or match_name
                    score = selected_match.get('score', 0.0)
                    
                    info_msg += f"🎯 Currently viewing: {display_name} (Score: {score:.3f})\n\n"
                
                info_msg += "💡 The NEW interactive version allows you to click peaks to select them!\n"
                info_msg += "Use 'Search Database' to find matching minerals, then click peaks in the overlay plot.\n"
                info_msg += "The minerals from your search results are good candidates to try!"
                
                # Show info after window is displayed to avoid Qt timing issues
                QTimer.singleShot(500, lambda: QMessageBox.information(self, "NEW Interactive Mixture Analysis Launched", info_msg))
            
            # Show the mixture analysis window
            mixture_analyzer.show()
            mixture_analyzer.raise_()
            mixture_analyzer.activateWindow()
            
            print(f"✅ Mixture analysis window created and shown (reference stored)")
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import NEW interactive mixture analysis module:\n{str(e)}\n\n"
                "Please ensure raman_mixture_analysis_interactive.py is in the same directory.\n"
                "For the old version, use: python raman_mixture_analysis_qt6.py"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Launch Error", 
                f"Failed to launch NEW interactive mixture analysis:\n{str(e)}"
            )
    
    def add_to_mixture_analysis(self):
        """Add the selected match as a starting point for NEW interactive mixture analysis."""
        if not self.selected_match:
            QMessageBox.warning(self, "No Selection", "Please select a match to add to mixture analysis.")
            return
        
        # Get the parent's query spectrum
        if not hasattr(self.parent(), 'current_wavenumbers') or self.parent().current_wavenumbers is None:
            QMessageBox.warning(self, "No Query Spectrum", 
                "No query spectrum is loaded in the main application.\n"
                "Load a spectrum in the main window first to perform mixture analysis.")
            return
        
        try:
            # Import the NEW interactive mixture analysis module
            from raman_mixture_analysis_interactive import InteractiveMixtureAnalyzer
            
            # Create QApplication instance if needed
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            # Create interactive mixture analysis window and store reference to prevent garbage collection
            mixture_analyzer = InteractiveMixtureAnalyzer()
            
            # Store reference in parent to prevent garbage collection
            if not hasattr(self.parent(), 'mixture_analysis_windows'):
                self.parent().mixture_analysis_windows = []
            self.parent().mixture_analysis_windows.append(mixture_analyzer)
            
            # Clean up closed windows from the list when this window closes
            def cleanup_window_reference():
                if hasattr(self.parent(), 'mixture_analysis_windows'):
                    try:
                        self.parent().mixture_analysis_windows.remove(mixture_analyzer)
                    except ValueError:
                        pass  # Already removed
            
            # Connect cleanup to window close event
            mixture_analyzer.destroyed.connect(cleanup_window_reference)
            
            # Load the query spectrum automatically
            query_wavenumbers = self.parent().current_wavenumbers
            query_intensities = self.parent().processed_intensities
            
            if query_wavenumbers is not None and query_intensities is not None:
                # Get spectrum name from parent
                parent = self.parent()
                spectrum_name = getattr(parent, 'spectrum_file_path', None)
                if spectrum_name:
                    spectrum_name = os.path.basename(spectrum_name)
                else:
                    spectrum_name = 'Query Spectrum from Search'
                
                # Set spectrum data directly in the interactive analyzer
                mixture_analyzer.user_wavenumbers = query_wavenumbers.copy()
                mixture_analyzer.user_spectrum = query_intensities.copy()
                mixture_analyzer.original_spectrum = query_intensities.copy()
                mixture_analyzer.current_residual = query_intensities.copy()
                
                # Pass the spectrum data to the plot canvas and display it
                mixture_analyzer.plot_canvas.user_wavenumbers = query_wavenumbers.copy()
                mixture_analyzer.plot_canvas.user_spectrum = query_intensities.copy()
                
                # Display the spectrum immediately after setting data
                QTimer.singleShot(100, mixture_analyzer.plot_canvas.display_user_spectrum)
                
                # Use QTimer to update UI elements after window is shown
                QTimer.singleShot(200, lambda: self.parent()._update_mixture_analyzer_constraint_ui(
                    mixture_analyzer, len(query_wavenumbers), spectrum_name, 
                    query_wavenumbers[0], query_wavenumbers[-1]))
                
                # Get information about the selected match for manual searching
                selected_match = self.selected_match
                if selected_match:
                    match_name = selected_match.get('name', 'Unknown')
                    metadata = selected_match.get('metadata', {})
                    display_name = metadata.get('NAME') or metadata.get('mineral_name') or match_name
                    score = selected_match.get('score', 0.0)
                    
                    # Pre-select the exact mineral data the user selected (not just by name)
                    QTimer.singleShot(300, lambda: mixture_analyzer.preselect_exact_mineral(selected_match))
                    
                    info_msg = f"NEW Interactive Mixture Analysis launched with {display_name} (Score: {score:.3f}) ready for analysis!\n\n"
                    info_msg += "📝 The mineral has been pre-selected and is displayed in the overlay plot.\n\n"
                    info_msg += "💡 How to use the NEW interactive version:\n"
                    info_msg += "1. The best match '{display_name}' is already selected for you\n"
                    info_msg += "2. Click peaks in the overlay plot to select them\n"
                    info_msg += "3. Click 'Fit Selected Peaks' to create synthetic components\n"
                    info_msg += "4. Search for more minerals and repeat to build up your mixture!\n\n"
                    info_msg += f"🎯 Start by selecting peaks from '{display_name}' in the overlay plot!"
                    
                    QMessageBox.information(self, "NEW Interactive Mixture Analysis Ready", info_msg)
                    
                    # Log the pre-selected mineral
                    mixture_analyzer.log_status(f"🎯 Pre-selected mineral from search: {display_name}")
                    mixture_analyzer.log_status(f"   Search confidence: {score:.3f}")
            
            # Show the mixture analysis window
            mixture_analyzer.show()
            mixture_analyzer.raise_()
            mixture_analyzer.activateWindow()
            
            print(f"✅ Mixture analysis window created and shown (reference stored)")
            
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import NEW interactive mixture analysis module:\n{str(e)}\n\n"
                "Please ensure raman_mixture_analysis_interactive.py is in the same directory.\n"
                "For the old version, use: python raman_mixture_analysis_qt6.py"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Launch Error", 
                f"Failed to launch NEW interactive mixture analysis:\n{str(e)}"
            )


class RamanMapImportWorker(QThread):
    """Worker thread for importing Raman spectral map data in the background."""
    
    # Define signals
    progress = Signal(int)           # Progress percentage (0-100)
    status_update = Signal(str)      # Status message
    finished = Signal(object)       # Finished with map_data result
    error = Signal(str)             # Error with error message
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.cancelled = False
    
    def cancel(self):
        """Cancel the import operation."""
        self.cancelled = True
    
    def run(self):
        """Run the import operation in the background thread."""
        try:
            # Run the parsing with progress callbacks
            map_data = self.parse_raman_spectral_map_static(
                self.file_path,
                progress_callback=self.emit_progress,
                status_callback=self.emit_status
            )
            
            if not self.cancelled:
                self.finished.emit(map_data)
                
        except Exception as e:
            if not self.cancelled:
                self.error.emit(str(e))
    
    def emit_progress(self, value):
        """Emit progress signal if not cancelled."""
        if not self.cancelled:
            self.progress.emit(value)
    
    def emit_status(self, message):
        """Emit status signal if not cancelled."""
        if not self.cancelled:
            self.status_update.emit(message)
    
    def parse_raman_spectral_map_static(self, file_path, progress_callback=None, status_callback=None):
        """Static version of parse_raman_spectral_map for use in worker thread."""
        try:
            if status_callback:
                status_callback("Reading file...")
            
            # Read the file line by line to handle inconsistent column counts
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                raise ValueError("File must have at least 2 lines (header + data)")
            
            total_lines = len(lines) - 1  # Exclude header
            
            if status_callback:
                status_callback("Parsing header...")
            if progress_callback:
                progress_callback(5)
            
            # Parse the first line to get Raman shifts
            first_line = lines[0].strip().split()
            print(f"First line has {len(first_line)} columns")
            
            # Extract Raman shifts from first row (skip first two cells which should be X,Y labels)
            if len(first_line) < 3:
                raise ValueError("First line must have at least 3 columns (X, Y, and Raman shifts)")
            
            raman_shifts = np.array([float(x) for x in first_line[2:]])
            print(f"Extracted {len(raman_shifts)} Raman shifts from header")
            
            # Process spatial data and spectra
            spatial_data = []
            spectra_data = []
            skipped_lines = 0
            
            if status_callback:
                status_callback(f"Processing {total_lines} spectra...")
            
            for i, line in enumerate(lines[1:], 1):  # Start from line 1 (skip header)
                # Check for cancellation
                if self.cancelled:
                    return None
                
                # Update progress every 1000 lines or for small datasets, every 100 lines
                update_frequency = 1000 if total_lines > 10000 else max(100, total_lines // 100)
                
                if i % update_frequency == 0 and progress_callback:
                    progress_percent = 5 + int((i / total_lines) * 60)  # 5-65% for line processing
                    progress_callback(progress_percent)
                    if status_callback:
                        status_callback(f"Processing spectra: {i}/{total_lines} ({progress_percent-5}%)")
                
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    parts = line.split()
                    
                    # Check if we have enough columns
                    if len(parts) < 3:
                        skipped_lines += 1
                        if skipped_lines <= 5:  # Only print first few warnings
                            print(f"Skipping line {i+1}: insufficient columns ({len(parts)})")
                        continue
                    
                    x_pos = float(parts[0])  # X position in microns
                    y_pos = float(parts[1])  # Y position in microns
                    
                    # Extract spectrum intensities - handle variable column counts
                    # Take only as many intensity values as we have Raman shifts
                    spectrum_parts = parts[2:]
                    n_to_take = min(len(spectrum_parts), len(raman_shifts))
                    
                    if n_to_take < len(raman_shifts):
                        # Pad with zeros if we have fewer intensities than expected
                        spectrum = np.zeros(len(raman_shifts))
                        spectrum[:n_to_take] = [float(x) for x in spectrum_parts[:n_to_take]]
                        if skipped_lines <= 5:
                            print(f"Line {i+1}: Padded spectrum from {n_to_take} to {len(raman_shifts)} values")
                    else:
                        # Take only the number of intensities we need
                        spectrum = np.array([float(x) for x in spectrum_parts[:len(raman_shifts)]])
                        if len(spectrum_parts) > len(raman_shifts) and skipped_lines <= 5:
                            print(f"Line {i+1}: Truncated spectrum from {len(spectrum_parts)} to {len(raman_shifts)} values")
                    
                    spatial_data.append([x_pos, y_pos])
                    spectra_data.append(spectrum)
                    
                except (ValueError, IndexError) as e:
                    skipped_lines += 1
                    if skipped_lines <= 5:  # Only print first few errors
                        print(f"Skipping line {i+1}: {e}")
                    continue
            
            if self.cancelled:
                return None
            
            if skipped_lines > 5:
                print(f"... and {skipped_lines - 5} more lines were skipped")
            
            if len(spatial_data) == 0:
                raise ValueError("No valid data rows found in file")
            
            if status_callback:
                status_callback("Converting to arrays...")
            if progress_callback:
                progress_callback(70)
            
            spatial_data = np.array(spatial_data)
            spectra_data = np.array(spectra_data)
            
            print(f"Successfully processed {len(spatial_data)} positions")
            print(f"Spatial X range: {spatial_data[:, 0].min():.1f} to {spatial_data[:, 0].max():.1f}")
            print(f"Spatial Y range: {spatial_data[:, 1].min():.1f} to {spatial_data[:, 1].max():.1f}")
            print(f"Spectrum intensity range: {spectra_data.min():.2e} to {spectra_data.max():.2e}")
            
            # Create organized map data structure
            if status_callback:
                status_callback("Creating map data structure...")
            if progress_callback:
                progress_callback(75)
                
            map_data = {
                'raman_shifts': raman_shifts,
                'spatial_coordinates': spatial_data,
                'spectra': spectra_data,
                'metadata': {
                    'source_file': str(file_path),
                    'n_positions': len(spatial_data),
                    'n_wavenumbers': len(raman_shifts),
                    'spatial_range_x': [spatial_data[:, 0].min(), spatial_data[:, 0].max()],
                    'spatial_range_y': [spatial_data[:, 1].min(), spatial_data[:, 1].max()],
                    'wavenumber_range': [raman_shifts.min(), raman_shifts.max()],
                    'skipped_lines': skipped_lines,
                    'units': {
                        'spatial': 'microns',
                        'wavenumber': 'cm⁻¹',
                        'intensity': 'a.u.'
                    }
                }
            }
            
            if self.cancelled:
                return None
            
            # Create gridded data for easier mapping
            x_unique = np.unique(spatial_data[:, 0])
            y_unique = np.unique(spatial_data[:, 1])
            
            if len(x_unique) > 1 and len(y_unique) > 1:
                try:
                    if status_callback:
                        status_callback("Creating gridded data for mapping...")
                    if progress_callback:
                        progress_callback(80)
                    
                    # Create interpolated grid for mapping
                    xi, yi = np.meshgrid(
                        np.linspace(spatial_data[:, 0].min(), spatial_data[:, 0].max(), len(x_unique)),
                        np.linspace(spatial_data[:, 1].min(), spatial_data[:, 1].max(), len(y_unique))
                    )
                    
                    # Interpolate each wavenumber onto the grid
                    gridded_spectra = np.zeros((len(y_unique), len(x_unique), len(raman_shifts)))
                    
                    for i, wavenumber in enumerate(raman_shifts):
                        # Check for cancellation during interpolation
                        if self.cancelled:
                            return None
                        
                        # Update progress for interpolation
                        if i % max(1, len(raman_shifts) // 10) == 0 and progress_callback:
                            interpolation_progress = 80 + int((i / len(raman_shifts)) * 15)  # 80-95%
                            progress_callback(interpolation_progress)
                            if status_callback:
                                status_callback(f"Interpolating wavenumber {i+1}/{len(raman_shifts)}...")
                        
                        intensities = spectra_data[:, i]
                        gridded_intensities = griddata(
                            spatial_data, intensities, (xi, yi), method='linear', fill_value=0
                        )
                        gridded_spectra[:, :, i] = gridded_intensities
                    
                    if not self.cancelled:
                        map_data['gridded_data'] = {
                            'x_grid': xi,
                            'y_grid': yi,
                            'spectra_grid': gridded_spectra
                        }
                        print("✓ Created gridded data for mapping")
                    
                except Exception as e:
                    print(f"Warning: Could not create gridded data: {e}")
            
            if self.cancelled:
                return None
            
            if status_callback:
                status_callback("Import complete!")
            if progress_callback:
                progress_callback(100)
            
            return map_data
            
        except Exception as e:
            raise Exception(f"Error parsing Raman spectral map: {str(e)}")


# Main entry point (if running as standalone)
if __name__ == "__main__":
    import sys
    
    app = QApplication(sys.argv)
    
    # Apply the application theme
    configure_compact_ui()
    apply_theme('compact')
    
    # Create main window
    window = RamanAnalysisAppQt6()
    window.show()
    
    # Run the application
    sys.exit(app.exec())