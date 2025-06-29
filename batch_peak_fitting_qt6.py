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
    QFileDialog, QApplication, QMenuBar, QMenu, QProgressDialog, QInputDialog,
    QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QAction, QKeySequence

# Import matplotlib for Qt6
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme

# Scientific computing
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import os
import chardet
import subprocess
import sys
import csv
from pathlib import Path

# Import the density analysis module
try:
    from Density.raman_density_analysis import RamanDensityAnalyzer
    DENSITY_AVAILABLE = True
except ImportError:
    DENSITY_AVAILABLE = False

# Import the geothermometry module
try:
    from raman_geothermometry import RamanGeothermometry, GeothermometerMethod
    GEOTHERMOMETRY_AVAILABLE = True
except ImportError:
    GEOTHERMOMETRY_AVAILABLE = False

# Import the density analysis module
try:
    from Density.raman_density_analysis import RamanDensityAnalyzer
    DENSITY_AVAILABLE = True
except ImportError:
    DENSITY_AVAILABLE = False

# Import the geothermometry module
try:
    from raman_geothermometry import RamanGeothermometry, GeothermometerMethod
    GEOTHERMOMETRY_AVAILABLE = True
except ImportError:
    GEOTHERMOMETRY_AVAILABLE = False

# Add state management import
try:
    from core.universal_state_manager import get_state_manager, register_module, save_module_state, load_module_state
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False
    print("Universal State Manager not available - continuing without persistent state")


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
        self.density_results = []  # Initialize density results storage
        self.geothermometry_results = []  # Initialize geothermometry results storage
        self.reference_peaks = []
        self.reference_fit_params = []
        self.reference_model = "Gaussian"
        self.reference_wavenumbers = np.array([])
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
        self.background_preview = None  # Store preview background
        
        # Live view storage
        self.saved_fits = {}  # Store saved fits for review
        
        self.setup_ui()
        self.initial_plot()
        
        # Add state management at the end of __init__
        if STATE_MANAGER_AVAILABLE:
            self.setup_state_management()
            self.add_state_management_ui()
    
    def setup_state_management(self):
        """Enable persistent state management for this analysis session"""
        try:
            # Register with state manager
            register_module('batch_peak_fitting', self)
            
            # Add save/load methods to this instance
            self.save_analysis_state = lambda notes="": save_module_state('batch_peak_fitting', notes)
            self.load_analysis_state = lambda: load_module_state('batch_peak_fitting')
            
            # Hook auto-save into critical methods
            self._add_auto_save_hooks()
            
            print("✅ State management enabled - your work will be auto-saved!")
            print("💾 Save location: ~/RamanLab_Projects/auto_saves/")
            
        except Exception as e:
            print(f"Warning: Could not enable state management: {e}")
    
    def _add_auto_save_hooks(self):
        """Add auto-save functionality to critical methods"""
        
        # Hook into manual fit saving
        if hasattr(self, 'update_batch_results_with_manual_fit'):
            original_method = self.update_batch_results_with_manual_fit
            
            def auto_save_wrapper(*args, **kwargs):
                result = original_method(*args, **kwargs)
                # Auto-save after manual adjustments
                save_module_state('batch_peak_fitting', "Auto-save: manual fit updated")
                return result
            
            self.update_batch_results_with_manual_fit = auto_save_wrapper
        
        # Hook into reference setting
        if hasattr(self, 'set_reference'):
            original_method = self.set_reference
            
            def auto_save_wrapper(*args, **kwargs):
                result = original_method(*args, **kwargs)
                # Auto-save after setting reference
                save_module_state('batch_peak_fitting', "Auto-save: reference set")
                return result
            
            self.set_reference = auto_save_wrapper
        
        # Hook into batch completion
        if hasattr(self, 'apply_to_all'):
            original_method = self.apply_to_all
            
            def auto_save_wrapper(*args, **kwargs):
                result = original_method(*args, **kwargs)
                # Auto-save after batch completion
                save_module_state('batch_peak_fitting', "Batch analysis completed")
                return result
            
            self.apply_to_all = auto_save_wrapper

    def add_state_management_ui(self):
        """Add visible UI elements for state management"""
        try:
            # Create state management tab in the tab widget
            if hasattr(self, 'tab_widget') and self.tab_widget is not None:
                # Create session management tab
                session_tab = QWidget()
                session_layout = QVBoxLayout(session_tab)
                
                # Status label
                self.state_status_label = QLabel("✅ Auto-save enabled")
                self.state_status_label.setStyleSheet("color: green; font-weight: bold; font-size: 12px;")
                session_layout.addWidget(self.state_status_label)
                
                # Save location info
                save_location = QLabel("📁 Files saved to: ~/RamanLab_Projects/auto_saves/")
                save_location.setWordWrap(True)
                save_location.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
                session_layout.addWidget(save_location)
                
                # Auto-save info
                auto_save_info = QLabel("🔄 Auto-saves after:\n• Manual peak adjustments\n• Setting reference spectra\n• Batch processing completion")
                auto_save_info.setWordWrap(True)
                auto_save_info.setStyleSheet("color: #444; font-size: 10px; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
                session_layout.addWidget(auto_save_info)
                
                # Manual controls group
                controls_group = QGroupBox("Manual Session Control")
                controls_layout = QVBoxLayout(controls_group)
                
                # Manual save button
                save_btn = QPushButton("💾 Save Session Now")
                save_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
                save_btn.clicked.connect(self.manual_save_session)
                controls_layout.addWidget(save_btn)
                
                # Load session button
                load_btn = QPushButton("📂 Load Session")
                load_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
                load_btn.clicked.connect(self.manual_load_session)
                controls_layout.addWidget(load_btn)
                
                # Show session folder button
                folder_btn = QPushButton("📁 Open Session Folder")
                folder_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
                folder_btn.clicked.connect(self.open_session_folder)
                controls_layout.addWidget(folder_btn)
                
                session_layout.addWidget(controls_group)
                session_layout.addStretch()
                
                # Add as new tab
                self.tab_widget.addTab(session_tab, "📋 Session")
                
                print("✅ State management UI added as 'Session' tab")
            else:
                print("⚠️ Could not find tab widget to add state management UI")
                
        except Exception as e:
            print(f"Warning: Could not add state management UI: {e}")
    
    def manual_save_session(self):
        """Manually save the current session"""
        try:
            if hasattr(self, 'save_analysis_state'):
                result = self.save_analysis_state("Manual save by user")
                if result:
                    self.state_status_label.setText("✅ Session saved successfully!")
                    self.state_status_label.setStyleSheet("color: green; font-weight: bold;")
                    QTimer.singleShot(3000, lambda: self.state_status_label.setText("✅ Auto-save enabled"))
                else:
                    self.state_status_label.setText("❌ Save failed")
                    self.state_status_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.state_status_label.setText("❌ State management not available")
        except Exception as e:
            print(f"Error saving session: {e}")
            self.state_status_label.setText("❌ Save error")
    
    def manual_load_session(self):
        """Manually load a saved session"""
        try:
            if hasattr(self, 'load_analysis_state'):
                result = self.load_analysis_state()
                if result:
                    self.state_status_label.setText("✅ Session loaded successfully!")
                    self.state_status_label.setStyleSheet("color: green; font-weight: bold;")
                    QTimer.singleShot(3000, lambda: self.state_status_label.setText("✅ Auto-save enabled"))
                    # Refresh UI after loading
                    self.update_plots()
                else:
                    self.state_status_label.setText("❌ No saved session found")
                    self.state_status_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.state_status_label.setText("❌ State management not available")
        except Exception as e:
            print(f"Error loading session: {e}")
            self.state_status_label.setText("❌ Load error")
    
    def open_session_folder(self):
        """Open the session folder in file explorer"""
        try:
            from pathlib import Path
            import subprocess
            import platform
            
            session_folder = Path.home() / "RamanLab_Projects" / "auto_saves"
            session_folder.mkdir(parents=True, exist_ok=True)
            
            # Open folder based on operating system
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(session_folder)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(session_folder)])
            else:  # Linux
                subprocess.run(["xdg-open", str(session_folder)])
                
            self.state_status_label.setText("📁 Session folder opened")
        except Exception as e:
            print(f"Error opening session folder: {e}")
            self.state_status_label.setText("❌ Could not open folder")

    def get_params_per_peak(self):
        """Get the number of parameters per peak for the current model."""
        if self.current_model in ["Pseudo-Voigt", "Asymmetric Voigt"]:
            if self.current_model == "Pseudo-Voigt":
                return 4  # amp, cen, wid, eta
            else:  # Asymmetric Voigt
                return 5  # amp, cen, wid, eta, asym
        else:
            return 3  # amp, cen, wid for Gaussian and Lorentzian
        
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
        self.bg_method_combo.addItems(["ALS (Asymmetric Least Squares)", "Linear", "Polynomial", "Moving Average", "Spline"])
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
        
        # Connect sliders to update labels and preview
        self.lambda_slider.valueChanged.connect(self.update_lambda_label)
        self.lambda_slider.valueChanged.connect(self.preview_background_subtraction)
        self.p_slider.valueChanged.connect(self.update_p_label)
        self.p_slider.valueChanged.connect(self.preview_background_subtraction)
        self.niter_slider.valueChanged.connect(self.update_niter_label)
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
        
        # Connect sliders to update labels and preview
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
        self.poly_order_slider.setValue(2)
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
        
        # Connect controls to preview
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
        self.window_size_slider.setRange(1, 50)
        self.window_size_slider.setValue(10)
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
        
        # Connect controls to preview
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
        self.knots_slider.setRange(5, 50)
        self.knots_slider.setValue(20)
        knots_layout.addWidget(self.knots_slider)
        self.knots_label = QLabel("20")
        knots_layout.addWidget(self.knots_label)
        spline_params_layout.addLayout(knots_layout)
        
        # Smoothing factor
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing Factor:"))
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 50)
        self.smoothing_slider.setValue(30)
        smoothing_layout.addWidget(self.smoothing_slider)
        self.smoothing_label = QLabel("1000")
        smoothing_layout.addWidget(self.smoothing_label)
        spline_params_layout.addLayout(smoothing_layout)
        
        # Spline degree
        degree_layout = QHBoxLayout()
        degree_layout.addWidget(QLabel("Spline Degree:"))
        self.spline_degree_slider = QSlider(Qt.Horizontal)
        self.spline_degree_slider.setRange(1, 5)
        self.spline_degree_slider.setValue(3)
        degree_layout.addWidget(self.spline_degree_slider)
        self.spline_degree_label = QLabel("3 (Cubic)")
        degree_layout.addWidget(self.spline_degree_label)
        spline_params_layout.addLayout(degree_layout)
        
        # Connect controls to preview
        self.knots_slider.valueChanged.connect(self.update_knots_label)
        self.smoothing_slider.valueChanged.connect(self.update_smoothing_label)
        self.spline_degree_slider.valueChanged.connect(self.update_spline_degree_label)
        self.knots_slider.valueChanged.connect(self.preview_background_subtraction)
        self.smoothing_slider.valueChanged.connect(self.preview_background_subtraction)
        self.spline_degree_slider.valueChanged.connect(self.preview_background_subtraction)
        
        bg_layout.addWidget(self.spline_params_widget)
        
        # Background subtraction buttons
        button_layout = QVBoxLayout()  # Changed from QHBoxLayout to QVBoxLayout
        
        subtract_btn = QPushButton("Apply Background Subtraction")
        subtract_btn.clicked.connect(self.apply_background_subtraction)
        button_layout.addWidget(subtract_btn)
        
        preview_btn = QPushButton("Clear Background Preview")
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
        
        # Set initial visibility state for background controls
        self.on_bg_method_changed()
        
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
        detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
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
        peak_list_left.addWidget(QLabel("Peak Positions:"))
        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setMaximumHeight(120)
        self.peak_list_widget.setToolTip(
            "Peak Positions:\n"
            "• 🔍 Detected = Initial rough positions from peak detection\n"
            "• ✓ Fitted = Refined positions from curve fitting (more accurate)\n"
            "• 👆 Manual = User-selected positions\n\n"
            "After fitting, the legend shows fitted positions (most accurate)"
        )
        peak_list_left.addWidget(self.peak_list_widget)
        peak_list_layout.addLayout(peak_list_left)
        
        # Delete buttons
        delete_buttons_layout = QVBoxLayout()
        
        delete_selected_btn = QPushButton("Delete Selected")
        delete_selected_btn.clicked.connect(self.delete_selected_peaks)
        delete_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #7F1D1D;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #991B1B;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
        """)
        delete_buttons_layout.addWidget(delete_selected_btn)
        
        delete_all_btn = QPushButton("Delete All Peaks")
        delete_all_btn.clicked.connect(self.clear_peaks)
        delete_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #B91C1C;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #EF4444;
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
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
            QPushButton:checked {
                background-color: #3B82F6;
            }
            QPushButton:checked:hover {
                background-color: #60A5FA;
            }
        """)
        manual_layout.addWidget(self.interactive_btn)
        
        # Manual peak controls
        manual_controls_layout = QHBoxLayout()
        
        clear_manual_btn = QPushButton("Clear Manual")
        clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        clear_manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #B91C1C;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #EF4444;
            }
        """)
        
        combine_btn = QPushButton("Combine Auto + Manual")
        combine_btn.clicked.connect(self.combine_peaks)
        combine_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
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
        fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
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
        set_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        batch_layout.addWidget(set_ref_btn)
        
        # Processing buttons
        process_layout = QHBoxLayout()
        apply_all_btn = QPushButton("Apply to All")
        apply_all_btn.clicked.connect(self.apply_to_all)
        apply_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        process_layout.addWidget(apply_all_btn)
        
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_batch)
        stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #B91C1C;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #EF4444;
            }
        """)
        process_layout.addWidget(stop_btn)
        
        batch_layout.addLayout(process_layout)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
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
        
        # Peak information display
        peak_info_group = QGroupBox("Current Peak Information")
        peak_info_layout = QVBoxLayout(peak_info_group)
        
        self.peak_info_text = QTextEdit()
        self.peak_info_text.setMaximumHeight(200)
        self.peak_info_text.setReadOnly(True)
        self.peak_info_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        self.peak_info_text.append("No peaks fitted yet")
        peak_info_layout.addWidget(self.peak_info_text)
        
        layout.addWidget(peak_info_group)
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
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
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
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        analysis_controls.addWidget(self.export_batch_csv_btn)
        
        # Manual fit update button
        self.update_manual_fit_btn = QPushButton("Save Manual Fit to Results")
        self.update_manual_fit_btn.clicked.connect(self.update_batch_results_with_manual_fit)
        self.update_manual_fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC2626;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #EF4444;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
        """)
        self.update_manual_fit_btn.setToolTip(
            "Save your current manual background and peak adjustments to batch results.\n"
            "This will preserve your manual fit and prevent it from being overwritten\n"
            "during future batch operations."
        )
        analysis_controls.addWidget(self.update_manual_fit_btn)
        
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
        show_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        button_layout.addWidget(show_all_btn)
        
        hide_all_btn = QPushButton("Hide All")
        hide_all_btn.clicked.connect(self.hide_all_peaks)
        hide_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
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
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        export_layout.addWidget(export_all_btn)
        
        # Export individual results button (existing functionality)
        export_results_btn = QPushButton("Export Peak Fitting Results")
        export_results_btn.clicked.connect(self.export_results)
        export_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        export_layout.addWidget(export_results_btn)
        
        layout.addWidget(export_group)
        
        layout.addStretch()
        return tab
        

        
    def create_specialized_tab(self):
        """Create specialized tools tab for launching external analysis programs."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header_label = QLabel("Specialized Analysis Tools")
        header_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(header_label)
        
        # Density Analysis Group
        density_group = QGroupBox("Density Analysis")
        density_layout = QVBoxLayout(density_group)
        
        # Description and button layout
        density_content_layout = QHBoxLayout()
        
        # Description
        density_desc = QLabel(
            "Launch the Raman Density Analysis tool for quantitative density\n"
            "analysis of kidney stone Raman spectroscopy data with micro-CT correlation.\n"
            "Supports bacterial biofilm analysis and crystalline density indexing."
        )
        density_desc.setWordWrap(True)
        density_content_layout.addWidget(density_desc)
        
        # Button container
        if DENSITY_AVAILABLE:
            button_container = QWidget()
            button_layout = QVBoxLayout(button_container)
            button_layout.setContentsMargins(20, 0, 0, 0)
            
            # Add stretch to center buttons vertically
            button_layout.addStretch()
            
            density_btn = QPushButton("Launch Density Analysis")
            density_btn.clicked.connect(self.launch_density_analysis)
            density_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            density_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1E3A8A;
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #1E40AF;
                }
                QPushButton:pressed {
                    background-color: #1D4ED8;
                }
            """)
            button_layout.addWidget(density_btn)
            
            # Quick analysis button for current spectrum
            if len(self.wavenumbers) > 0 and len(self.intensities) > 0:
                quick_analysis_btn = QPushButton("Analyze Current Spectrum")
                quick_analysis_btn.clicked.connect(self.quick_density_analysis)
                quick_analysis_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                quick_analysis_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1E3A8A;
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #1E40AF;
                    }
                    QPushButton:pressed {
                        background-color: #1D4ED8;
                    }
                """)
                button_layout.addWidget(quick_analysis_btn)
            
            # Batch analysis button - enabled when batch results are available
            batch_density_btn = QPushButton("Analyze Batch Results")
            batch_density_btn.clicked.connect(self.run_batch_density_analysis)
            batch_density_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            batch_density_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1E3A8A;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1E40AF;
                }
                QPushButton:pressed {
                    background-color: #1D4ED8;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)
            batch_density_btn.setEnabled(False)  # Enable when batch results are available
            batch_density_btn.setToolTip("Run density analysis on all batch processing results")
            button_layout.addWidget(batch_density_btn)
            
            # Store reference for enabling/disabling
            self.batch_density_btn = batch_density_btn
            
            # Add stretch to center buttons vertically
            button_layout.addStretch()
            density_content_layout.addWidget(button_container)
        else:
            error_label = QLabel("Density analysis module not available.\nCheck Density/ folder for raman_density_analysis.py")
            error_label.setStyleSheet("color: red; font-style: italic;")
            density_content_layout.addWidget(error_label)
        
        density_layout.addLayout(density_content_layout)
        
        layout.addWidget(density_group)
        
        # Geothermometry Analysis Group
        geothermo_group = QGroupBox("Geothermometry Analysis")
        geothermo_layout = QVBoxLayout(geothermo_group)
        
        # Description and button layout
        geothermo_content_layout = QHBoxLayout()
        
        # Description
        geothermo_desc = QLabel(
            "Launch the Raman Geothermometry tool for metamorphic temperature\n"
            "determination using carbonaceous material in metamorphic rocks.\n"
            "Supports multiple calibrations (Beyssac, Aoya, Rahl, Kouketsu, Rantitsch)."
        )
        geothermo_desc.setWordWrap(True)
        geothermo_content_layout.addWidget(geothermo_desc)
        
        # Button container
        if GEOTHERMOMETRY_AVAILABLE:
            button_container = QWidget()
            button_layout = QVBoxLayout(button_container)
            button_layout.setContentsMargins(20, 0, 0, 0)
            
            # Add stretch to center buttons vertically
            button_layout.addStretch()
            
            geothermo_btn = QPushButton("Launch Geothermometry Analysis")
            geothermo_btn.clicked.connect(self.launch_geothermometry_analysis)
            geothermo_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            geothermo_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1E3A8A;
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #1E40AF;
                }
                QPushButton:pressed {
                    background-color: #1D4ED8;
                }
            """)
            button_layout.addWidget(geothermo_btn)
            
            # Quick analysis button for current spectrum
            if len(self.wavenumbers) > 0 and len(self.intensities) > 0:
                quick_geothermo_btn = QPushButton("Analyze Current Spectrum")
                quick_geothermo_btn.clicked.connect(self.quick_geothermometry_analysis)
                quick_geothermo_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                quick_geothermo_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1E3A8A;
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #1E40AF;
                    }
                    QPushButton:pressed {
                        background-color: #1D4ED8;
                    }
                """)
                button_layout.addWidget(quick_geothermo_btn)
            
            # Batch analysis button - enabled when batch results are available
            batch_geothermo_btn = QPushButton("Analyze Batch Results")
            batch_geothermo_btn.clicked.connect(self.run_batch_geothermometry_analysis)
            batch_geothermo_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            batch_geothermo_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1E3A8A;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1E40AF;
                }
                QPushButton:pressed {
                    background-color: #1D4ED8;
                }
                QPushButton:disabled {
                    background-color: #BDBDBD;
                }
            """)
            batch_geothermo_btn.setEnabled(False)  # Enable when batch results are available
            batch_geothermo_btn.setToolTip("Run geothermometry analysis on all batch processing results")
            button_layout.addWidget(batch_geothermo_btn)
            
            # Store reference for enabling/disabling
            self.batch_geothermo_btn = batch_geothermo_btn
            
            # Add stretch to center buttons vertically
            button_layout.addStretch()
            geothermo_content_layout.addWidget(button_container)
        else:
            error_label = QLabel("Geothermometry analysis module not available.\nCheck for raman_geothermometry.py")
            error_label.setStyleSheet("color: red; font-style: italic;")
            geothermo_content_layout.addWidget(error_label)
        
        geothermo_layout.addLayout(geothermo_content_layout)
        
        layout.addWidget(geothermo_group)
        
        # Future tools placeholder
        future_group = QGroupBox("Future Analysis Tools")
        future_layout = QVBoxLayout(future_group)
        
        future_desc = QLabel(
            "Additional specialized analysis tools will be added here:\n"
            "• Feldspar Ca/Na/K ratio chemical analysis\n"
            "• Zircon zoning analysis\n"
            "• Garnet zoning analysis"
        )
        future_desc.setWordWrap(True)
        future_desc.setStyleSheet("color: gray; font-style: italic;")
        future_layout.addWidget(future_desc)
        
        layout.addWidget(future_group)
        
        layout.addStretch()
        return tab
        
    def launch_density_analysis(self):
        """Launch the density analysis tool as a separate process."""
        try:
            # Get the path to the density analysis GUI launcher
            script_path = Path(__file__).parent / "Density" / "density_gui_launcher.py"
            
            if not script_path.exists():
                # Fallback to the main analysis script
                script_path = Path(__file__).parent / "Density" / "raman_density_analysis.py"
                if not script_path.exists():
                    QMessageBox.warning(self, "File Not Found", 
                                      f"Density analysis scripts not found in Density/ folder")
                    return
            
            # Launch as separate Python process
            subprocess.Popen([sys.executable, str(script_path)], 
                           cwd=Path(__file__).parent)
            
            QMessageBox.information(self, "Launched", 
                                  "Density Analysis tool has been launched in a separate window.")
            
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", 
                               f"Failed to launch density analysis tool:\n{str(e)}")
    
    def quick_density_analysis(self):
        """Perform quick density analysis on the current spectrum."""
        if not DENSITY_AVAILABLE:
            QMessageBox.warning(self, "Not Available", 
                              "Density analysis module is not available.")
            return
            
        if len(self.wavenumbers) == 0 or len(self.intensities) == 0:
            QMessageBox.warning(self, "No Data", 
                              "No spectrum data loaded for analysis.")
            return
        
        try:
            from Density.raman_density_analysis import MaterialConfigs
            
            # Ask user for material type
            material_types = MaterialConfigs.get_available_materials()
            material_type, ok = QInputDialog.getItem(
                self, "Select Material Type", 
                "Choose the material type for analysis:",
                material_types, 0, False
            )
            
            if not ok:
                return
            
            # Create analyzer instance with selected material
            analyzer = RamanDensityAnalyzer(material_type)
            
            # Use current spectrum data
            wavenumbers = self.wavenumbers
            intensities = self.intensities
            
            # Preprocess spectrum
            wn, corrected_int = analyzer.preprocess_spectrum(wavenumbers, intensities)
            
            # Calculate CDI and density
            cdi, metrics = analyzer.calculate_crystalline_density_index(wn, corrected_int)
            apparent_density = analyzer.calculate_apparent_density(cdi)
            
            # Use appropriate density calculation
            if material_type == 'Kidney Stones (COM)':
                specialized_density = analyzer.calculate_biofilm_density(cdi, 'mixed')
                density_label = "Biofilm-Calibrated Density"
            else:
                specialized_density = analyzer.calculate_density_by_type(cdi, 'mixed')
                density_label = "Specialized Density"
            
            # Get classification
            thresholds = analyzer.classification_thresholds
            if cdi < thresholds['low']:
                classification = 'Low crystallinity'
            elif cdi < thresholds['medium']:
                classification = 'Mixed regions'
            elif cdi < thresholds['high']:
                classification = 'Mixed crystalline'
            else:
                classification = 'Pure crystalline'
            
            # Display results
            result_text = f"""
Density Analysis Results ({material_type}):

Crystalline Density Index (CDI): {cdi:.4f}
Standard Apparent Density: {apparent_density:.3f} g/cm³
{density_label}: {specialized_density:.3f} g/cm³

Metrics:
• Main Peak Height: {metrics['main_peak_height']:.1f}
• Main Peak Position: {metrics['main_peak_position']} cm⁻¹
• Baseline Intensity: {metrics['baseline_intensity']:.1f}
• Peak Width (FWHM): {metrics['peak_width']:.1f} cm⁻¹
• Spectral Contrast: {metrics['spectral_contrast']:.4f}

Classification: {classification} region

Material-Specific Guidelines:
• CDI < {thresholds['low']:.2f}: Low crystallinity regions
• CDI {thresholds['low']:.2f}-{thresholds['medium']:.2f}: Mixed regions
• CDI {thresholds['medium']:.2f}-{thresholds['high']:.2f}: Mixed crystalline
• CDI > {thresholds['high']:.2f}: Pure crystalline
            """
            
            QMessageBox.information(self, "Density Analysis Results", result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", 
                               f"Failed to perform density analysis:\n{str(e)}")
        
    def create_visualization_panel(self):
        """Create the visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        self.viz_tab_widget = QTabWidget()
        layout.addWidget(self.viz_tab_widget)
        
        # Connect tab change signal to update plots when needed
        self.viz_tab_widget.currentChanged.connect(self.on_viz_tab_changed)
        
        # Current spectrum tab
        current_tab = self.create_current_spectrum_tab()
        self.viz_tab_widget.addTab(current_tab, "Current Spectrum")
        
        # Live view tab
        live_view_tab = self.create_live_view_tab()
        self.viz_tab_widget.addTab(live_view_tab, "Live View")
        
        # Trends tab
        trends_tab = self.create_trends_tab()
        self.viz_tab_widget.addTab(trends_tab, "Trends")
        
        # Fitting Quality tab
        fitting_quality_tab = self.create_fitting_quality_tab()
        self.viz_tab_widget.addTab(fitting_quality_tab, "Fitting Quality")
        
        # Waterfall tab
        waterfall_tab = self.create_waterfall_tab()
        self.viz_tab_widget.addTab(waterfall_tab, "Waterfall")
        
        # Heatmap tab
        heatmap_tab = self.create_heatmap_tab()
        self.viz_tab_widget.addTab(heatmap_tab, "Heatmap")
        
        # Specialized tab
        specialized_tab = self.create_specialized_tab()
        self.viz_tab_widget.addTab(specialized_tab, "Specialized")
        
        return panel
        
    def on_viz_tab_changed(self, index):
        """Handle visualization tab changes to update plots as needed."""
        tab_names = ["Current Spectrum", "Live View", "Trends", "Fitting Quality", "Waterfall", "Heatmap", "Specialized"]
        if 0 <= index < len(tab_names):
            tab_name = tab_names[index]
            print(f"DEBUG: Switched to {tab_name} tab (index {index})")
            
            # Update the fitting quality plot when its tab is selected
            if tab_name == "Fitting Quality":
                self.update_fitting_quality_plot()
                
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
        """Create trends visualization tab with controls."""
        tab = QWidget()
        layout = QHBoxLayout(tab)  # Use horizontal layout for controls and plot
        
        # Left panel - controls
        controls_panel = QWidget()
        controls_panel.setMaximumWidth(300)
        controls_layout = QVBoxLayout(controls_panel)
        
        # Display options for trends
        trends_display_group = QGroupBox("Display Options")
        trends_display_layout = QVBoxLayout(trends_display_group)
        
        # Create trends-specific checkboxes
        self.trends_show_grid = QCheckBox("Show Grid Lines")
        self.trends_show_grid.setChecked(True)
        self.trends_show_grid.stateChanged.connect(self.update_trends_plot)
        trends_display_layout.addWidget(self.trends_show_grid)
        
        self.trends_show_peak_labels = QCheckBox("Show Peak Labels")
        self.trends_show_peak_labels.setChecked(True)
        self.trends_show_peak_labels.stateChanged.connect(self.update_trends_plot)
        trends_display_layout.addWidget(self.trends_show_peak_labels)
        
        self.trends_show_trend_lines = QCheckBox("Show Trend Lines")
        self.trends_show_trend_lines.setChecked(False)
        self.trends_show_trend_lines.stateChanged.connect(self.update_trends_plot)
        trends_display_layout.addWidget(self.trends_show_trend_lines)
        
        controls_layout.addWidget(trends_display_group)
        
        # Info panel
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel(
            "💡 Tips:\n"
            "• Click plots to open in new window\n"
            "• Use checkboxes to customize display\n"
            "• Trend lines show data progression\n"
            "• All parameter plots are displayed"
        )
        info_text.setStyleSheet("color: #666; font-size: 10px;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        controls_layout.addWidget(info_group)
        controls_layout.addStretch()
        
        layout.addWidget(controls_panel)
        
        # Right panel - plot
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        
        # Create matplotlib figure for trends
        self.figure_trends = Figure(figsize=(12, 10))
        self.canvas_trends = FigureCanvas(self.figure_trends)
        self.toolbar_trends = NavigationToolbar(self.canvas_trends, plot_panel)
        
        plot_layout.addWidget(self.toolbar_trends)
        plot_layout.addWidget(self.canvas_trends)
        
        layout.addWidget(plot_panel)
        
        return tab
        
    def create_fitting_quality_tab(self):
        """Create fitting quality visualization tab with 3x3 grid."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create matplotlib figure for fitting quality grid
        self.figure_fitting_quality = Figure(figsize=(15, 12))
        self.canvas_fitting_quality = FigureCanvas(self.figure_fitting_quality)
        self.toolbar_fitting_quality = NavigationToolbar(self.canvas_fitting_quality, tab)
        
        # Create 3x3 grid of subplots
        self.figure_fitting_quality.clear()
        gs = self.figure_fitting_quality.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Store axes for easy access
        self.axes_fitting_quality = []
        for i in range(3):
            row_axes = []
            for j in range(3):
                ax = self.figure_fitting_quality.add_subplot(gs[i, j])
                row_axes.append(ax)
            self.axes_fitting_quality.append(row_axes)
        
        # Set row titles
        self.axes_fitting_quality[0][1].set_title("Best Fitting Results (Top R²)", fontsize=14, fontweight='bold', pad=20)
        self.axes_fitting_quality[1][1].set_title("Median Fitting Results", fontsize=14, fontweight='bold', pad=20)
        self.axes_fitting_quality[2][1].set_title("Poorest Fitting Results (Lowest R²)", fontsize=14, fontweight='bold', pad=20)
        
        layout.addWidget(self.toolbar_fitting_quality)
        layout.addWidget(self.canvas_fitting_quality)
        
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
                    # Safely handle fit_failed flag - ensure it's a boolean
                    fit_failed = result.get('fit_failed', True)
                    if isinstance(fit_failed, np.ndarray):
                        fit_failed = bool(fit_failed.any()) if fit_failed.size > 0 else True
                    elif not isinstance(fit_failed, bool):
                        fit_failed = bool(fit_failed) if fit_failed is not None else True
                    
                    if result['file'] == file_path and not fit_failed:
                        batch_result = result
                        break
            
            if batch_result:
                # Load fitted data from batch results
                self.wavenumbers = np.array(batch_result['wavenumbers'])
                self.intensities = np.array(batch_result['intensities'])
                self.original_intensities = np.array(batch_result['original_intensities'])
                
                # Handle background safely
                if batch_result.get('background') is not None:
                    self.background = np.array(batch_result['background'])
                else:
                    self.background = None
                
                # Handle peaks safely - check both 'peaks' and 'reference_peaks' keys
                reference_peaks = batch_result.get('reference_peaks', None)
                peaks = batch_result.get('peaks', None)
                
                # Safely check if reference_peaks exist and are not empty
                if (reference_peaks is not None and 
                    (isinstance(reference_peaks, (list, np.ndarray)) and len(reference_peaks) > 0)):
                    self.reference_peaks = np.array(reference_peaks)
                    self.peaks = self.reference_peaks.copy()  # For compatibility
                elif (peaks is not None and 
                      (isinstance(peaks, (list, np.ndarray)) and len(peaks) > 0)):
                    self.peaks = np.array(peaks)
                    self.reference_peaks = self.peaks.copy()
                else:
                    self.peaks = np.array([], dtype=int)
                    self.reference_peaks = np.array([], dtype=int)
                
                # Handle fit parameters safely
                self.fit_params = batch_result.get('fit_params', [])
                if self.fit_params is None:
                    self.fit_params = []
                
                self.fit_result = True
                
                # Handle residuals safely
                if batch_result.get('residuals') is not None:
                    self.residuals = np.array(batch_result['residuals'])
                else:
                    self.residuals = None
                
                # Clear manual peaks when loading fitted data
                self.manual_peaks = np.array([], dtype=int)
                
                self.current_spectrum_index = index
                
                # Update UI
                self.update_file_status()
                self.update_peak_count_display()
                self.update_current_plot()
                
                # Update Live View with current spectrum for analysis
                self._update_live_view_with_current_spectrum()
                
                # Update peak info display for batch result
                self.update_peak_info_display_for_batch_result(batch_result)
                
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
                self.background_preview = None  # Clear background preview
                self.background_preview_active = False
                self.residuals = None
                
                # Update UI
                self.update_file_status()
                self.update_peak_count_display()
                self.update_current_plot()
                
                # Auto-enable background preview when spectrum is loaded
                if len(self.original_intensities) > 0:
                    self.preview_background_subtraction()
                
                # Update peak info display for regular spectrum
                self.update_peak_info_display()
                
                # Highlight current file in list
                self.file_list_widget.setCurrentRow(index)
                
        except Exception as e:
            # More graceful error handling - don't show critical dialog for minor issues
            print(f"Warning: Issue loading spectrum {index}: {str(e)}")
            
            # Try to load the file anyway without batch data
            try:
                data = self.load_spectrum_robust(file_path)
                if data is not None:
                    self.wavenumbers, self.intensities = data
                    self.original_intensities = self.intensities.copy()
                    self.current_spectrum_index = index
                    
                    # Reset peak data for new spectrum
                    self.peaks = np.array([], dtype=int)
                    self.reference_peaks = np.array([], dtype=int)
                    self.manual_peaks = np.array([], dtype=int)
                    self.fit_params = []
                    self.fit_result = None
                    self.background = None
                    self.background_preview = None
                    self.background_preview_active = False
                    self.residuals = None
                    
                    # Update UI
                    self.update_file_status()
                    self.update_peak_count_display()
                    self.update_current_plot()
                    self.update_peak_info_display()
                    self.file_list_widget.setCurrentRow(index)
                    
                    print(f"Successfully loaded spectrum {index} as new file (no batch data)")
                    return
            except Exception as load_error:
                print(f"Error: Could not load spectrum file {file_path}: {load_error}")
                QMessageBox.warning(self, "Load Warning", 
                                   f"Could not load spectrum {os.path.basename(file_path)}.\n"
                                   f"File may be corrupted or in unsupported format.")
                return
            
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
        self.background_preview = None  # Clear background preview
        self.background_preview_active = False
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
    
    def update_niter_label(self):
        """Update the niter label based on the slider value."""
        value = self.niter_slider.value()
        self.niter_label.setText(str(value))
    
    def update_start_weight_label(self):
        """Update the start weight label based on the slider value."""
        value = self.start_weight_slider.value() / 10.0  # Convert to 0.1-2.0 range
        self.start_weight_label.setText(f"{value:.1f}")
    
    def update_end_weight_label(self):
        """Update the end weight label based on the slider value."""
        value = self.end_weight_slider.value() / 10.0  # Convert to 0.1-2.0 range
        self.end_weight_label.setText(f"{value:.1f}")
    
    def update_poly_order_label(self):
        """Update the polynomial order label based on the slider value."""
        value = self.poly_order_slider.value()
        self.poly_order_label.setText(str(value))
    
    def update_window_size_label(self):
        """Update the window size label based on the slider value."""
        value = self.window_size_slider.value()
        self.window_size_label.setText(f"{value}%")
    
    def update_knots_label(self):
        """Update the knots label based on the slider value."""
        value = self.knots_slider.value()
        self.knots_label.setText(str(value))
    
    def update_smoothing_label(self):
        """Update the smoothing label based on the slider value."""
        value = self.smoothing_slider.value()
        # Convert slider value to smoothing factor (exponential scale)
        if value <= 10:
            smoothing_factor = 10 ** value
        else:
            smoothing_factor = 10 ** (1 + (value - 1) / 9 * 4)  # 10^1 to 10^5
        self.smoothing_label.setText(f"{int(smoothing_factor)}")
    
    def update_spline_degree_label(self):
        """Update the spline degree label based on the slider value."""
        value = self.spline_degree_slider.value()
        degree_names = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic", 5: "Quintic"}
        degree_name = degree_names.get(value, f"Order {value}")
        self.spline_degree_label.setText(f"{value} ({degree_name})")

    def on_bg_method_changed(self):
        """Handle change in background method."""
        method = self.bg_method_combo.currentText()
        # Show/hide parameter widgets based on selected method
        self.als_params_widget.setVisible(method.startswith("ALS"))
        self.linear_params_widget.setVisible(method == "Linear")
        self.poly_params_widget.setVisible(method == "Polynomial")
        self.moving_avg_params_widget.setVisible(method == "Moving Average")
        self.spline_params_widget.setVisible(method == "Spline")

    def update_peak_detection(self):
        """Improved peak detection with noise filtering."""
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
        
        # Calculate actual values with improved baseline estimation
        intensities = self.intensities
        
        # Use rolling minimum for better baseline estimation
        baseline_window = max(len(intensities) // 20, 5)
        baseline = np.array([np.min(intensities[max(0, i-baseline_window//2):min(len(intensities), i+baseline_window//2+1)]) 
                           for i in range(len(intensities))])
        
        # Calculate signal above baseline for better peak detection
        signal_above_baseline = intensities - baseline
        max_signal = np.max(signal_above_baseline)
        
        height_threshold = (height_percent / 100.0) * max_signal if height_percent > 0 else None
        prominence_threshold = (prominence_percent / 100.0) * max_signal if prominence_percent > 0 else None
        
        # Find peaks with current parameters
        try:
            peak_kwargs = {}
            if height_threshold is not None and height_threshold > 0:
                peak_kwargs['height'] = height_threshold
            if distance > 1:
                peak_kwargs['distance'] = distance
            if prominence_threshold is not None and prominence_threshold > 0:
                peak_kwargs['prominence'] = prominence_threshold
            
            # Apply peak detection to signal above baseline
            self.peaks, properties = find_peaks(signal_above_baseline, **peak_kwargs)
            
            # Filter out peaks that are too close to edges
            edge_buffer = max(len(intensities) // 50, 2)
            valid_peaks = self.peaks[(self.peaks >= edge_buffer) & (self.peaks < len(intensities) - edge_buffer)]
            self.peaks = valid_peaks
            
            # Update peak count
            self.peak_count_label.setText(f"Peaks found: {len(self.peaks)}")
            
            # Update reference_peaks to match current peak state after detection
            if hasattr(self, 'manual_peaks'):
                all_peaks = np.concatenate([self.peaks, self.manual_peaks]) if len(self.manual_peaks) > 0 else self.peaks
                self.reference_peaks = all_peaks.copy() if len(all_peaks) > 0 else np.array([], dtype=int)
                print(f"DEBUG: Updated reference_peaks to {len(self.reference_peaks)} peaks after detection")
            
            # Update peak count display (which should update the list)
            self.update_peak_count_display()
            
            # Update plot and peak info display
            self.update_current_plot()
            self.update_peak_info_display()
            
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
                niter = self.niter_slider.value()
                
                self.background = self.baseline_als(self.original_intensities, lambda_value, p_value, niter)
                self.intensities = self.original_intensities - self.background
                
            elif method == "Linear":
                # Get linear parameters
                start_weight = self.start_weight_slider.value() / 10.0
                end_weight = self.end_weight_slider.value() / 10.0
                
                self.background = self._calculate_linear_background(start_weight, end_weight)
                self.intensities = self.original_intensities - self.background
                
            elif method == "Polynomial":
                # Get polynomial parameters
                poly_order = self.poly_order_slider.value()
                poly_method = self.poly_method_combo.currentText()
                
                self.background = self._calculate_polynomial_background(poly_order, poly_method)
                self.intensities = self.original_intensities - self.background
                
            elif method == "Moving Average":
                # Get moving average parameters
                window_percent = self.window_size_slider.value()
                window_type = self.window_type_combo.currentText()
                
                self.background = self._calculate_moving_average_background(window_percent, window_type)
                self.intensities = self.original_intensities - self.background
                
            elif method == "Spline":
                # Get spline parameters
                n_knots = self.knots_slider.value()
                smoothing_value = self.smoothing_slider.value()
                degree = self.spline_degree_slider.value()
                
                # Convert smoothing slider value to smoothing factor
                if smoothing_value <= 10:
                    smoothing = 10 ** smoothing_value
                else:
                    smoothing = 10 ** (1 + (smoothing_value - 1) / 9 * 4)
                
                self.background = self._calculate_spline_background(n_knots, smoothing, degree)
                self.intensities = self.original_intensities - self.background
                
            else:
                QMessageBox.information(self, "Not Implemented", 
                                      f"{method} background subtraction not yet implemented.")
                return
            
            # Clear any background preview state to show clean result
            self.background_preview = None
            self.background_preview_active = False
                
            self.update_current_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Background subtraction failed: {str(e)}")

    def preview_background_subtraction(self):
        """Preview background subtraction in real-time as sliders move."""
        if len(self.original_intensities) == 0:
            return
            
        try:
            method = self.bg_method_combo.currentText()
            
            if method.startswith("ALS"):
                # Get ALS parameters from sliders
                lambda_value = 10 ** self.lambda_slider.value()
                p_value = self.p_slider.value() / 1000.0
                niter = self.niter_slider.value()
                
                self.background_preview = self.baseline_als(self.original_intensities, lambda_value, p_value, niter)
                
            elif method == "Linear":
                # Get linear parameters
                start_weight = self.start_weight_slider.value() / 10.0
                end_weight = self.end_weight_slider.value() / 10.0
                
                self.background_preview = self._calculate_linear_background(start_weight, end_weight)
                
            elif method == "Polynomial":
                # Get polynomial parameters
                poly_order = self.poly_order_slider.value()
                poly_method = self.poly_method_combo.currentText()
                
                self.background_preview = self._calculate_polynomial_background(poly_order, poly_method)
                
            elif method == "Moving Average":
                # Get moving average parameters
                window_percent = self.window_size_slider.value()
                window_type = self.window_type_combo.currentText()
                
                self.background_preview = self._calculate_moving_average_background(window_percent, window_type)
                
            elif method == "Spline":
                # Get spline parameters
                n_knots = self.knots_slider.value()
                smoothing_value = self.smoothing_slider.value()
                degree = self.spline_degree_slider.value()
                
                # Convert smoothing slider value to smoothing factor
                if smoothing_value <= 10:
                    smoothing = 10 ** smoothing_value
                else:
                    smoothing = 10 ** (1 + (smoothing_value - 1) / 9 * 4)
                
                self.background_preview = self._calculate_spline_background(n_knots, smoothing, degree)
            
            self.background_preview_active = True
            
            # Update plot to show preview
            self.update_current_plot()
            
        except Exception as e:
            print(f"Background preview error: {e}")
            self.background_preview = None
            self.background_preview_active = False

    def clear_background_preview(self):
        """Clear background preview and return to original view."""
        self.background_preview = None
        self.background_preview_active = False
        
        # If we have batch results, make sure we show the batch result view
        if (hasattr(self, 'batch_results') and 
            len(self.batch_results) > 0 and 
            hasattr(self, 'current_spectrum_index') and 
            0 <= self.current_spectrum_index < len(self.batch_results)):
            # This will trigger batch result mode in update_current_plot
            pass
        
        self.update_current_plot()

    def reset_spectrum(self):
        """Reset spectrum to original state."""
        if len(self.original_intensities) > 0:
            self.intensities = self.original_intensities.copy()
            self.background = None
            self.background_preview = None  # Clear background preview
            self.background_preview_active = False
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
        self.reference_peaks = np.array([], dtype=int)  # Also clear reference peaks
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        print(f"DEBUG: Cleared all peaks including reference_peaks")
        self.update_peak_count_display()
        self.update_current_plot()
        self.update_peak_info_display()

    def delete_selected_peaks(self):
        """Delete selected peaks from the peak list."""
        if not hasattr(self, 'peak_list_widget'):
            return
            
        # Get selected items
        selected_items = self.peak_list_widget.selectedItems()
        if not selected_items:
            return
            
        # Extract peak indices to delete
        indices_to_delete = []
        manual_indices_to_delete = []
        fitted_peaks_to_delete = []
        
        for item in selected_items:
            text = item.text()
            
            # Parse the peak text to extract index
            if "Fitted Peak" in text:
                # Extract fitted peak number (1-based)
                import re
                match = re.search(r'Fitted Peak (\d+)', text)
                if match:
                    peak_num = int(match.group(1)) - 1  # Convert to 0-based
                    fitted_peaks_to_delete.append(peak_num)
                    
            elif "Detected Peak" in text:
                # Extract detected peak number (1-based)
                import re
                match = re.search(r'Detected Peak (\d+)', text)
                if match:
                    peak_num = int(match.group(1)) - 1  # Convert to 0-based
                    indices_to_delete.append(peak_num)
                    
            elif "Manual Peak" in text:
                # Extract manual peak number (1-based)
                import re
                match = re.search(r'Manual Peak (\d+)', text)
                if match:
                    peak_num = int(match.group(1)) - 1  # Convert to 0-based
                    manual_indices_to_delete.append(peak_num)
        
        # Delete fitted peaks (remove from fit_params)
        if fitted_peaks_to_delete and hasattr(self, 'fit_params') and self.fit_params is not None and len(self.fit_params) > 0:
            params_per_peak = self.get_params_per_peak()
            new_fit_params = []
            
            for peak_idx in range(len(self.fit_params) // params_per_peak):
                if peak_idx not in fitted_peaks_to_delete:
                    start_idx = peak_idx * params_per_peak
                    end_idx = start_idx + params_per_peak
                    new_fit_params.extend(self.fit_params[start_idx:end_idx])
            
            self.fit_params = new_fit_params
            
        # Delete detected peaks (remove from self.peaks)
        if indices_to_delete and hasattr(self, 'peaks') and len(self.peaks) > 0:
            # Convert to numpy array if needed
            if not isinstance(self.peaks, np.ndarray):
                self.peaks = np.array(self.peaks, dtype=int)
            
            # Remove peaks at specified indices
            mask = np.ones(len(self.peaks), dtype=bool)
            for idx in sorted(indices_to_delete, reverse=True):
                if 0 <= idx < len(self.peaks):
                    mask[idx] = False
            self.peaks = self.peaks[mask]
        
        # Delete manual peaks (remove from self.manual_peaks)
        if manual_indices_to_delete and hasattr(self, 'manual_peaks') and len(self.manual_peaks) > 0:
            # Convert to numpy array if needed
            if not isinstance(self.manual_peaks, np.ndarray):
                self.manual_peaks = np.array(self.manual_peaks, dtype=int)
            
            # Remove peaks at specified indices
            mask = np.ones(len(self.manual_peaks), dtype=bool)
            for idx in sorted(manual_indices_to_delete, reverse=True):
                if 0 <= idx < len(self.manual_peaks):
                    mask[idx] = False
            self.manual_peaks = self.manual_peaks[mask]
        
        # Update reference_peaks to match current peak state
        # This is crucial for the peak list display and fitting to work correctly
        if hasattr(self, 'peaks') and hasattr(self, 'manual_peaks'):
            all_peaks = np.concatenate([self.peaks, self.manual_peaks]) if len(self.manual_peaks) > 0 else self.peaks
            self.reference_peaks = all_peaks.copy() if len(all_peaks) > 0 else np.array([], dtype=int)
            print(f"DEBUG: Updated reference_peaks to {len(self.reference_peaks)} peaks after deletion")
        
        # Update displays
        self.update_peak_count_display()
        self.update_current_plot()
        self.update_peak_info_display()
        
        print(f"Deleted {len(indices_to_delete)} detected peaks, {len(manual_indices_to_delete)} manual peaks, and {len(fitted_peaks_to_delete)} fitted peaks")

    def baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """Improved Asymmetric Least Squares baseline correction."""
        try:
            L = len(y)
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w*y)
                w = p * (y > z) + (1-p) * (y < z)
            
            return z
        except Exception as e:
            print(f"ALS baseline error: {e}")
            # Fallback to simple linear baseline
            return np.linspace(y[0], y[-1], len(y))

    def _calculate_linear_background(self, start_weight, end_weight):
        """Simplified linear background calculation."""
        try:
            y = self.original_intensities
            
            # Use percentile-based endpoint estimation for robustness
            n_points = min(len(y) // 10, 50)  # Use 10% of points or max 50 points
            
            # Get start region baseline estimate
            start_region = y[:n_points] if len(y) > n_points else y[:len(y)//3]
            start_val = np.percentile(start_region, 10) * start_weight
            
            # Get end region baseline estimate  
            end_region = y[-n_points:] if len(y) > n_points else y[-len(y)//3:]
            end_val = np.percentile(end_region, 10) * end_weight
            
            # Create linear background
            background = np.linspace(start_val, end_val, len(y))
            
            return background
            
        except Exception as e:
            print(f"Linear background calculation error: {str(e)}")
            # Final fallback
            y = self.original_intensities
            return np.linspace(y[0], y[-1], len(y))

    def _calculate_polynomial_background(self, poly_order, poly_method):
        """Simplified polynomial background calculation."""
        try:
            y = self.original_intensities
            x = np.arange(len(y))
            
            # Use lower envelope approach
            # Rolling minimum to identify baseline candidates
            window_size = max(len(y) // 20, 5)
            
            # Create rolling minimum
            baseline_candidates = []
            for i in range(len(y)):
                start_idx = max(0, i - window_size//2)
                end_idx = min(len(y), i + window_size//2 + 1)
                baseline_candidates.append(np.min(y[start_idx:end_idx]))
            
            baseline_y = np.array(baseline_candidates)
            
            # Additional smoothing of baseline candidates
            if len(baseline_y) > 5:
                from scipy.signal import savgol_filter
                try:
                    smooth_window = min(len(baseline_y)//3, 51)
                    if smooth_window % 2 == 0:
                        smooth_window += 1
                    if smooth_window >= 3:
                        baseline_y = savgol_filter(baseline_y, smooth_window, 2)
                except:
                    pass
            
            # Fit polynomial to smoothed baseline
            poly_order = min(poly_order, len(x) - 1)  # Ensure valid order
            
            if poly_method == "Robust":
                # Simple robust fitting - iterative reweighting
                coeffs = np.polyfit(x, baseline_y, poly_order)
                for iteration in range(2):
                    poly_fit = np.polyval(coeffs, x)
                    residuals = np.abs(baseline_y - poly_fit)
                    mad = np.median(residuals)
                    weights = 1.0 / (1.0 + (residuals / (mad + 1e-10)))
                    coeffs = np.polyfit(x, baseline_y, poly_order, w=weights)
            else:
                coeffs = np.polyfit(x, baseline_y, poly_order)
            
            background = np.polyval(coeffs, x)
            
            return background
            
        except Exception as e:
            print(f"Polynomial background calculation error: {str(e)}")
            # Fallback to linear
            return self._calculate_linear_background(1.0, 1.0)

    def _calculate_moving_average_background(self, window_percent, window_type):
        """Simplified moving average background calculation."""
        try:
            y = self.original_intensities
            
            # Calculate window size
            window_size = max(int(len(y) * window_percent / 100.0), 3)
            
            # Use rolling minimum as baseline estimate
            baseline = np.array([np.min(y[max(0, i-window_size//2):min(len(y), i+window_size//2+1)]) 
                               for i in range(len(y))])
            
            # Apply smoothing filter
            if window_type == "Gaussian" and len(baseline) > 5:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    sigma = window_size / 6.0
                    baseline = gaussian_filter1d(baseline, sigma=sigma)
                except:
                    # Fallback to simple moving average
                    baseline = np.convolve(baseline, np.ones(min(window_size, len(baseline)))/(min(window_size, len(baseline))), mode='same')
            else:
                # Simple moving average
                if window_size < len(baseline):
                    baseline = np.convolve(baseline, np.ones(window_size)/window_size, mode='same')
            
            return baseline
            
        except Exception as e:
            print(f"Moving average background calculation error: {str(e)}")
            # Fallback to linear
            return self._calculate_linear_background(1.0, 1.0)

    def _optimize_background_for_spectrum(self, wavenumbers, intensities, bg_method):
        """
        Optimize background parameters for a specific spectrum to maximize peak fitting quality.
        
        Returns:
            tuple: (background_corrected_intensities, background, params_used_description)
        """
        best_r2 = -1
        best_intensities = None
        best_background = None
        best_params = None
        
        # Use reference peaks as targets for optimization
        if (not hasattr(self, 'reference_peaks') or 
            self.reference_peaks is None or 
            (hasattr(self.reference_peaks, '__len__') and len(self.reference_peaks) == 0)):
            raise ValueError("No reference peaks available for background optimization")
        
        print(f"  Trying background method: {bg_method}")
        
        if bg_method.startswith("ALS"):
            # Try multiple ALS parameter combinations
            ui_lambda = 10 ** self.lambda_slider.value()
            ui_p = self.p_slider.value() / 1000.0
            ui_niter = self.niter_slider.value()
            
            # Test multiple lambda values around UI setting
            lambda_candidates = [
                ui_lambda * 0.1,   # More flexible
                ui_lambda * 0.5,   # Moderately flexible  
                ui_lambda,         # UI setting
                ui_lambda * 2.0,   # Less flexible
                ui_lambda * 5.0    # Much less flexible
            ]
            
            # Test multiple p values  
            p_candidates = [
                max(0.001, ui_p * 0.5),  # More asymmetric weighting
                ui_p,                    # UI setting
                min(0.1, ui_p * 2.0)     # Less asymmetric weighting
            ]
            
            for lam in lambda_candidates:
                for p in p_candidates:
                    try:
                        bg = self.baseline_als(intensities, lam, p, ui_niter)
                        corrected = intensities - bg
                        r2 = self._evaluate_background_quality(wavenumbers, corrected)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_intensities = corrected
                            best_background = bg
                            best_params = f"ALS(λ={lam:.0e}, p={p:.3f}, iter={ui_niter})"
                    except:
                        continue
                        
        elif bg_method == "Linear":
            # Try multiple linear weight combinations
            ui_start = self.start_weight_slider.value() / 10.0
            ui_end = self.end_weight_slider.value() / 10.0
            
            weight_candidates = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different endpoint weightings
            
            for start_w in weight_candidates:
                for end_w in weight_candidates:
                    try:
                        bg = self._calculate_linear_background_for_data(intensities, start_w, end_w)
                        corrected = intensities - bg
                        r2 = self._evaluate_background_quality(wavenumbers, corrected)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_intensities = corrected  
                            best_background = bg
                            best_params = f"Linear(start={start_w:.1f}, end={end_w:.1f})"
                    except:
                        continue
                        
        elif bg_method == "Polynomial":
            # Try multiple polynomial orders and methods
            poly_methods = ["percentile", "robust", "endpoints"]
            orders = [1, 2, 3, 4]
            
            for order in orders:
                for method in poly_methods:
                    try:
                        bg = self._calculate_polynomial_background_for_data(intensities, order, method)
                        corrected = intensities - bg
                        r2 = self._evaluate_background_quality(wavenumbers, corrected)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_intensities = corrected
                            best_background = bg
                            best_params = f"Poly(order={order}, method={method})"
                    except:
                        continue
                        
        elif bg_method == "Moving Average":
            # Try multiple window sizes and types
            window_types = ["minimum", "percentile"]
            window_sizes = [5, 10, 15, 20, 25]  # Different window percentages
            
            for window_size in window_sizes:
                for window_type in window_types:
                    try:
                        bg = self._calculate_moving_average_background_for_data(intensities, window_size, window_type)
                        corrected = intensities - bg
                        r2 = self._evaluate_background_quality(wavenumbers, corrected)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_intensities = corrected
                            best_background = bg
                            best_params = f"MovingAvg(window={window_size}%, type={window_type})"
                    except:
                        continue
        
        # If no optimization worked, use UI parameters as fallback
        if best_intensities is None:
            print(f"  Background optimization failed, using UI parameters")
            try:
                if bg_method.startswith("ALS"):
                    lambda_value = 10 ** self.lambda_slider.value()
                    p_value = self.p_slider.value() / 1000.0
                    niter = self.niter_slider.value()
                    best_background = self.baseline_als(intensities, lambda_value, p_value, niter)
                    best_params = f"ALS(UI: λ={lambda_value:.0e}, p={p_value:.3f})"
                elif bg_method == "Linear":
                    start_weight = self.start_weight_slider.value() / 10.0
                    end_weight = self.end_weight_slider.value() / 10.0
                    best_background = self._calculate_linear_background_for_data(intensities, start_weight, end_weight)
                    best_params = f"Linear(UI: start={start_weight:.1f}, end={end_weight:.1f})"
                elif bg_method == "Polynomial":
                    poly_order = self.poly_order_slider.value()
                    poly_method = self.poly_method_combo.currentText()
                    best_background = self._calculate_polynomial_background_for_data(intensities, poly_order, poly_method)
                    best_params = f"Poly(UI: order={poly_order}, method={poly_method})"
                else:
                    # Ultimate fallback
                    best_background = self.baseline_als(intensities, 1e5, 0.01, 10)
                    best_params = "ALS(default fallback)"
                    
                best_intensities = intensities - best_background
                
            except Exception as e:
                # Last resort: linear background
                best_background = np.linspace(intensities[0], intensities[-1], len(intensities))
                best_intensities = intensities - best_background
                best_params = "Linear(last resort)"
        
        print(f"  Best background R²: {best_r2:.3f}, params: {best_params}")
        return best_intensities, best_background, best_params

    def debug_background_optimization(self, wavenumbers, intensities, bg_method):
        """
        Debug version of background optimization that shows all attempts and scores.
        This helps diagnose why ALS backgrounds might appear linear.
        """
        print(f"\n=== DEBUGGING BACKGROUND OPTIMIZATION ===")
        print(f"Method: {bg_method}")
        print(f"Spectrum length: {len(intensities)}")
        print(f"Intensity range: {np.min(intensities):.1f} to {np.max(intensities):.1f}")
        
        all_attempts = []
        
        # Use reference peaks as targets for optimization
        if (not hasattr(self, 'reference_peaks') or 
            self.reference_peaks is None or 
            (hasattr(self.reference_peaks, '__len__') and len(self.reference_peaks) == 0)):
            print("ERROR: No reference peaks available!")
            return None, None, None
        
        if bg_method.startswith("ALS"):
            # Try multiple ALS parameter combinations
            ui_lambda = 10 ** self.lambda_slider.value()
            ui_p = self.p_slider.value() / 1000.0
            ui_niter = self.niter_slider.value()
            
            print(f"UI ALS Parameters: λ={ui_lambda:.0e}, p={ui_p:.3f}, iter={ui_niter}")
            
            # Test multiple lambda values around UI setting
            lambda_candidates = [
                ui_lambda * 0.1,   # More flexible
                ui_lambda * 0.5,   # Moderately flexible  
                ui_lambda,         # UI setting
                ui_lambda * 2.0,   # Less flexible
                ui_lambda * 5.0    # Much less flexible
            ]
            
            # Test multiple p values  
            p_candidates = [
                max(0.001, ui_p * 0.5),  # More asymmetric weighting
                ui_p,                    # UI setting
                min(0.1, ui_p * 2.0)     # Less asymmetric weighting
            ]
            
            print(f"Testing {len(lambda_candidates)} λ values × {len(p_candidates)} p values = {len(lambda_candidates) * len(p_candidates)} combinations")
            
            for lam in lambda_candidates:
                for p in p_candidates:
                    try:
                        # Apply ALS background
                        bg = self.baseline_als(intensities, lam, p, ui_niter)
                        corrected = intensities - bg
                        
                        # Check if background looks linear
                        bg_linearity = self._check_background_linearity(bg)
                        
                        # Evaluate quality
                        r2 = self._evaluate_background_quality(wavenumbers, corrected)
                        
                        attempt_info = {
                            'method': f"ALS(λ={lam:.0e}, p={p:.3f})",
                            'r2': r2,
                            'background': bg,
                            'corrected': corrected,
                            'linearity': bg_linearity,
                            'params': {'lambda': lam, 'p': p, 'niter': ui_niter}
                        }
                        all_attempts.append(attempt_info)
                        
                        print(f"  ALS λ={lam:.0e}, p={p:.3f} → R²={r2:.3f}, linearity={bg_linearity:.3f}")
                        
                    except Exception as e:
                        print(f"  ALS λ={lam:.0e}, p={p:.3f} → FAILED: {e}")
                        continue
        
        # Also try linear for comparison
        print(f"\nTesting Linear backgrounds for comparison:")
        weight_candidates = [0.1, 0.5, 0.9]
        
        for start_w in weight_candidates:
            for end_w in weight_candidates:
                try:
                    bg = self._calculate_linear_background_for_data(intensities, start_w, end_w)
                    corrected = intensities - bg
                    bg_linearity = self._check_background_linearity(bg)
                    r2 = self._evaluate_background_quality(wavenumbers, corrected)
                    
                    attempt_info = {
                        'method': f"Linear(start={start_w:.1f}, end={end_w:.1f})",
                        'r2': r2,
                        'background': bg,
                        'corrected': corrected,
                        'linearity': bg_linearity,
                        'params': {'start_weight': start_w, 'end_weight': end_w}
                    }
                    all_attempts.append(attempt_info)
                    
                    print(f"  Linear start={start_w:.1f}, end={end_w:.1f} → R²={r2:.3f}, linearity={bg_linearity:.3f}")
                    
                except Exception as e:
                    print(f"  Linear start={start_w:.1f}, end={end_w:.1f} → FAILED: {e}")
                    continue
        
        # Find best attempt
        if all_attempts:
            best_attempt = max(all_attempts, key=lambda x: x['r2'])
            print(f"\nBEST RESULT:")
            print(f"  Method: {best_attempt['method']}")
            print(f"  R²: {best_attempt['r2']:.3f}")
            print(f"  Linearity: {best_attempt['linearity']:.3f}")
            
            # Show top 3 for comparison
            sorted_attempts = sorted(all_attempts, key=lambda x: x['r2'], reverse=True)
            print(f"\nTop 3 methods:")
            for i, attempt in enumerate(sorted_attempts[:3]):
                print(f"  {i+1}. {attempt['method']} (R²={attempt['r2']:.3f}, linearity={attempt['linearity']:.3f})")
            
            return best_attempt['corrected'], best_attempt['background'], best_attempt['method']
        else:
            print("ERROR: No valid background methods worked!")
            return None, None, None

    def _check_background_linearity(self, background):
        """
        Check how linear a background is (0.0 = perfectly linear, 1.0 = highly non-linear)
        """
        try:
            # Fit a linear line to the background
            x = np.arange(len(background))
            linear_fit = np.polyfit(x, background, 1)
            linear_bg = np.polyval(linear_fit, x)
            
            # Calculate how much the background deviates from linear
            deviations = np.abs(background - linear_bg)
            max_deviation = np.max(deviations)
            bg_range = np.max(background) - np.min(background)
            
            # Normalize deviation by background range
            if bg_range > 1e-10:
                linearity_score = 1.0 - (max_deviation / bg_range)
            else:
                linearity_score = 1.0  # Perfectly flat (linear)
            
            return max(0.0, min(1.0, linearity_score))
        except:
            return 0.0

    def test_current_spectrum_background_debug(self):
        """
        Test background optimization on the current spectrum with full debugging.
        This helps diagnose why ALS backgrounds might look linear.
        """
        if len(self.original_intensities) == 0:
            QMessageBox.warning(self, "No Data", "Load a spectrum first.")
            return
        
        # Safely check reference_peaks
        if (not hasattr(self, 'reference_peaks') or 
            self.reference_peaks is None or 
            (hasattr(self.reference_peaks, '__len__') and len(self.reference_peaks) == 0)):
            QMessageBox.warning(self, "No Reference", "Set reference peaks first.")
            return
        
        # Run debug optimization
        bg_method = self.bg_method_combo.currentText()
        corrected, background, method_used = self.debug_background_optimization(
            self.wavenumbers, self.original_intensities, bg_method
        )
        
        if corrected is not None:
            # Update the current view with debug results
            self.intensities = corrected
            self.background = background
            self.update_current_plot()
            
            QMessageBox.information(self, "Debug Complete", 
                                  f"Debug optimization complete!\n"
                                  f"Best method: {method_used}\n\n"
                                  f"Check console output for detailed analysis.")
        else:
            QMessageBox.critical(self, "Debug Failed", 
                               "Background optimization debugging failed.\n"
                               "Check console output for details.")
     
    def _evaluate_background_quality(self, wavenumbers, background_corrected_intensities):
        """
        Evaluate how well background correction preserves ALL peak structure, not just reference peaks.
        Returns a score based on peak preservation and spectral integrity.
        """
        try:
            from scipy.signal import find_peaks
            
            # Method 1: Detect ALL peaks in the background-corrected spectrum
            # This ensures we preserve peaks that might not be in the reference
            
            # Calculate noise level for peak detection
            noise_level = np.std(background_corrected_intensities) * 0.5
            min_peak_height = max(np.percentile(background_corrected_intensities, 75), noise_level * 3)
            
            # Find all significant peaks in the corrected spectrum
            peak_indices, peak_properties = find_peaks(
                background_corrected_intensities,
                height=min_peak_height,
                prominence=noise_level * 2,
                width=2,
                distance=5  # Minimum separation between peaks
            )
            
            if len(peak_indices) == 0:
                # No peaks detected - background might be too aggressive
                return 0.0
            
            # Method 2: Evaluate signal-to-noise ratio in peak regions
            total_peak_signal = 0.0
            baseline_level = np.percentile(background_corrected_intensities, 10)
            
            for peak_idx in peak_indices:
                peak_height = background_corrected_intensities[peak_idx] - baseline_level
                total_peak_signal += max(0, peak_height)
            
            # Method 3: Check for negative intensities (sign of over-correction)
            negative_penalty = 0.0
            if np.any(background_corrected_intensities < 0):
                negative_count = np.sum(background_corrected_intensities < 0)
                negative_penalty = negative_count / len(background_corrected_intensities)
            
            # Method 4: Check for over-aggressive background subtraction
            # Look for artifacts like steep drops near edges or unnatural curvature
            background_quality_score = 1.0
            
            # Check for edge artifacts (steep drops near spectrum edges)
            edge_threshold = len(background_corrected_intensities) // 10  # First/last 10% of spectrum
            left_edge_slope = (background_corrected_intensities[edge_threshold] - background_corrected_intensities[0]) / edge_threshold
            right_edge_slope = (background_corrected_intensities[-1] - background_corrected_intensities[-edge_threshold]) / edge_threshold
            
            # Penalize steep negative slopes at edges (sign of poor linear background)
            if left_edge_slope < -total_peak_signal * 0.05:  # Steep negative slope on left
                background_quality_score *= 0.7
            if right_edge_slope > total_peak_signal * 0.05:  # Steep positive slope on right  
                background_quality_score *= 0.7
            
            # Method 5: Evaluate spectral smoothness in non-peak regions
            # Create a mask for non-peak regions
            non_peak_mask = np.ones(len(background_corrected_intensities), dtype=bool)
            for peak_idx in peak_indices:
                # Exclude region around each peak (±20 points or ±peak_width*2)
                peak_width = 20  # Default width
                if 'widths' in peak_properties:
                    peak_width = max(20, int(peak_properties['widths'][np.where(peak_indices == peak_idx)[0][0]] * 2))
                start_exclude = max(0, peak_idx - peak_width)
                end_exclude = min(len(background_corrected_intensities), peak_idx + peak_width)
                non_peak_mask[start_exclude:end_exclude] = False
            
            # Calculate smoothness in non-peak regions
            if np.sum(non_peak_mask) > 10:  # Need enough non-peak points
                non_peak_data = background_corrected_intensities[non_peak_mask]
                # Calculate second derivative (curvature) to detect unnatural bending
                if len(non_peak_data) > 4:
                    smoothness = np.std(np.gradient(np.gradient(non_peak_data)))
                    # Normalize by signal level
                    normalized_smoothness = smoothness / (np.std(non_peak_data) + 1e-6)
                    # Penalize high curvature (unnatural bending)
                    if normalized_smoothness > 0.5:
                        background_quality_score *= 0.8
            
            # Method 6: Check for preservation of reference peaks specifically
            reference_peak_score = 1.0
            if hasattr(self, 'reference_peaks') and len(self.reference_peaks) > 0:
                preserved_ref_peaks = 0
                for ref_peak_idx in self.reference_peaks:
                    if 0 <= ref_peak_idx < len(background_corrected_intensities):
                        ref_peak_height = background_corrected_intensities[ref_peak_idx] - baseline_level
                        if ref_peak_height > min_peak_height * 0.5:  # At least 50% of detection threshold
                            preserved_ref_peaks += 1
                
                reference_peak_score = preserved_ref_peaks / len(self.reference_peaks)
            
            # Combine all factors into final score
            signal_score = min(1.0, total_peak_signal / (np.std(background_corrected_intensities) * 10 + 1e-6))
            negative_score = 1.0 - negative_penalty
            
            final_score = (signal_score * 0.4 +           # 40% - peak signal preservation
                          background_quality_score * 0.3 +  # 30% - background quality  
                          negative_score * 0.2 +            # 20% - no negative values
                          reference_peak_score * 0.1)       # 10% - reference peak preservation
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            # If evaluation fails, return low score
            return 0.0

    def _detect_additional_peaks(self, wavenumbers, intensities, fitted_curve, residuals):
        """
        Detect additional peaks in the residuals that weren't included in the reference.
        Returns list of peak positions in wavenumber units.
        """
        try:
            from scipy.signal import find_peaks
            
            # Look for peaks in the residuals that are significantly above noise
            noise_level = np.std(residuals) 
            
            # Also look for peaks in the original data that might have been missed
            # due to poor background correction
            
            # Method 1: Find peaks in residuals (missed peaks should show up as positive residuals)
            residual_peaks, residual_props = find_peaks(
                residuals,
                height=noise_level * 3,  # At least 3x noise level
                prominence=noise_level * 2,
                width=2,
                distance=8  # Minimum separation
            )
            
            # Method 2: Find peaks in background-corrected data that are far from reference peaks
            signal_peaks, signal_props = find_peaks(
                intensities,
                height=np.percentile(intensities, 75),  # Above 75th percentile
                prominence=noise_level * 3,
                width=2,
                distance=8
            )
            
            additional_positions = []
            
            # Convert residual peak indices to wavenumber positions
            for peak_idx in residual_peaks:
                if 0 <= peak_idx < len(wavenumbers):
                    peak_pos = wavenumbers[peak_idx]
                    
                    # Check if this peak is far enough from reference peaks
                    min_distance_to_ref = float('inf')
                    if hasattr(self, 'reference_peaks') and len(self.reference_peaks) > 0:
                        distances = np.abs(np.array(self.reference_peaks) - peak_pos)
                        min_distance_to_ref = np.min(distances)
                    
                    # Only add if it's far from reference peaks (>30 cm⁻¹ separation)
                    if min_distance_to_ref > 30:
                        additional_positions.append(peak_pos)
            
            # Also check signal peaks that are far from reference
            for peak_idx in signal_peaks:
                if 0 <= peak_idx < len(wavenumbers):
                    peak_pos = wavenumbers[peak_idx]
                    
                    # Check distance from reference peaks
                    min_distance_to_ref = float('inf')
                    if hasattr(self, 'reference_peaks') and len(self.reference_peaks) > 0:
                        distances = np.abs(np.array(self.reference_peaks) - peak_pos)
                        min_distance_to_ref = np.min(distances)
                    
                    # Check distance from already found additional peaks
                    min_distance_to_additional = float('inf')
                    if len(additional_positions) > 0:
                        distances = np.abs(np.array(additional_positions) - peak_pos)
                        min_distance_to_additional = np.min(distances)
                    
                    # Only add if it's far from both reference and other additional peaks
                    if min_distance_to_ref > 30 and min_distance_to_additional > 20:
                        # Double-check that this peak is actually significant
                        peak_height = intensities[peak_idx]
                        local_baseline = np.percentile(
                            intensities[max(0, peak_idx-10):min(len(intensities), peak_idx+11)], 
                            10
                        )
                        if peak_height - local_baseline > noise_level * 4:
                            additional_positions.append(peak_pos)
            
            # Remove duplicates and sort
            additional_positions = sorted(list(set(additional_positions)))
            
            # Limit to reasonable number of additional peaks
            if len(additional_positions) > 5:
                # Keep only the strongest peaks
                peak_strengths = []
                for pos in additional_positions:
                    closest_idx = np.argmin(np.abs(wavenumbers - pos))
                    strength = intensities[closest_idx] - np.percentile(intensities, 10)
                    peak_strengths.append((strength, pos))
                
                # Sort by strength and keep top 5
                peak_strengths.sort(reverse=True)
                additional_positions = [pos for strength, pos in peak_strengths[:5]]
                additional_positions.sort()
            
            return additional_positions
            
        except Exception as e:
            print(f"Additional peak detection failed: {e}")
            return []

    def _try_peak_fitting_with_background(self, wavenumbers, intensities, background, bounds, params_per_peak, n_peaks):
        """
        Helper method to try peak fitting with a specific background and return R².
        Used for testing different background parameter optimizations.
        """
        try:
            # Set up for fitting
            self.wavenumbers = wavenumbers
            self.peaks = self.reference_peaks
            self.current_model = self.reference_model
            
            # Create initial parameters from reference
            initial_params = []
            for peak_i in range(n_peaks):
                start_idx = peak_i * params_per_peak
                if start_idx + 2 < len(self.reference_fit_params):
                    ref_amp = self.reference_fit_params[start_idx]
                    ref_cen = self.reference_fit_params[start_idx + 1] 
                    ref_wid = self.reference_fit_params[start_idx + 2]
                    
                    # Scale amplitude based on current spectrum
                    closest_idx = np.argmin(np.abs(wavenumbers - ref_cen))
                    current_amp = intensities[closest_idx] if 0 <= closest_idx < len(intensities) else ref_amp
                    
                    if abs(ref_amp) > 1e-6:
                        amp_scale = abs(current_amp) / abs(ref_amp)
                        amp_scale = np.clip(amp_scale, 0.05, 20.0)
                        initial_amp = ref_amp * amp_scale
                    else:
                        initial_amp = abs(current_amp)
                    
                    initial_params.extend([abs(initial_amp), ref_cen, abs(ref_wid)])
                    
                    # Add Voigt parameters if needed
                    if self.reference_model == "Pseudo-Voigt" and start_idx + 3 < len(self.reference_fit_params):
                        initial_params.append(self.reference_fit_params[start_idx + 3])
                    elif self.reference_model == "Asymmetric Voigt" and start_idx + 4 < len(self.reference_fit_params):
                        initial_params.extend([self.reference_fit_params[start_idx + 3], self.reference_fit_params[start_idx + 4]])
            
            # Try fitting
            popt, pcov = curve_fit(
                self.multi_peak_model,
                wavenumbers,
                intensities,
                p0=initial_params,
                bounds=bounds,
                max_nfev=5000,
                method='trf'
            )
            
            # Calculate R² with emphasis on individual peak quality
            fitted_curve = self.multi_peak_model(wavenumbers, *popt)
            residuals = intensities - fitted_curve
            
            # Calculate individual peak R² values
            try:
                individual_r2_values, peak_avg_r2 = self.calculate_individual_peak_r_squared(
                    wavenumbers, intensities, popt, self.reference_model
                )
                
                # Weight individual peak R² more heavily for weak data assessment
                if len(individual_r2_values) > 0 and peak_avg_r2 > 0.1:
                    # For weak data, use peak-based R² as primary metric
                    return peak_avg_r2
                else:
                    # Fallback to traditional R² if peak-based fails
                    ss_res = np.sum(residuals ** 2)
                    baseline_intensity = np.percentile(intensities, 5)
                    signal_above_baseline = intensities - baseline_intensity
                    ss_tot = np.sum(signal_above_baseline ** 2)
                    
                    if ss_tot > 1e-10:
                        traditional_r2 = 1 - (ss_res / ss_tot)
                        return max(0.0, min(1.0, traditional_r2))
                    else:
                        return 0.0
            except:
                # Last resort traditional calculation
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
                if ss_tot > 1e-10:
                    return max(0.0, min(1.0, 1 - (ss_res / ss_tot)))
                else:
                    return 0.0
                    
        except Exception as e:
            return None

    def _fit_hybrid_peaks(self, wavenumbers, intensities, reference_params, additional_peak_positions):
        """
        Fit spectrum with both reference peaks and additional detected peaks.
        Returns improved fitting result if successful.
        """
        try:
            if len(additional_peak_positions) == 0:
                return None
            
            # Determine model parameters per peak
            params_per_peak = self.get_params_per_peak()
            n_ref_peaks = len(reference_params) // params_per_peak
            n_additional_peaks = len(additional_peak_positions)
            total_peaks = n_ref_peaks + n_additional_peaks
            
            # Create initial parameters: reference params + estimated additional params
            hybrid_initial_params = list(reference_params)  # Start with reference parameters
            
            # Add parameters for additional peaks
            for add_pos in additional_peak_positions:
                # Find intensity at additional peak position
                closest_idx = np.argmin(np.abs(wavenumbers - add_pos))
                peak_intensity = intensities[closest_idx]
                
                # Estimate amplitude (above local baseline)
                local_region = slice(max(0, closest_idx-10), min(len(intensities), closest_idx+11))
                local_baseline = np.percentile(intensities[local_region], 10)
                estimated_amp = peak_intensity - local_baseline
                
                # Estimate width (start with reasonable default)
                estimated_width = 15.0  # Default width for Raman peaks
                
                # Add basic parameters [amp, center, width]
                hybrid_initial_params.extend([estimated_amp, add_pos, estimated_width])
                
                # Add model-specific parameters
                if self.reference_model == "Pseudo-Voigt":
                    hybrid_initial_params.append(0.5)  # eta parameter
                elif self.reference_model == "Asymmetric Voigt":
                    hybrid_initial_params.extend([0.5, 0.0])  # eta, asymmetry
            
            # Create bounds for hybrid fitting
            bounds_lower = []
            bounds_upper = []
            
            # Bounds for reference peaks (more relaxed)
            for i in range(n_ref_peaks):
                start_idx = i * params_per_peak
                
                # Amplitude bounds
                ref_amp = abs(reference_params[start_idx])
                bounds_lower.append(ref_amp * 0.01)
                bounds_upper.append(ref_amp * 50)
                
                # Center bounds (allow more movement)
                ref_center = reference_params[start_idx + 1]
                ref_width = abs(reference_params[start_idx + 2])
                bounds_lower.append(ref_center - ref_width * 5)
                bounds_upper.append(ref_center + ref_width * 5)
                
                # Width bounds
                bounds_lower.append(ref_width * 0.2)
                bounds_upper.append(ref_width * 5)
                
                # Model-specific bounds
                if self.reference_model == "Pseudo-Voigt":
                    bounds_lower.append(0.0)
                    bounds_upper.append(1.0)
                elif self.reference_model == "Asymmetric Voigt":
                    bounds_lower.extend([0.0, -0.5])
                    bounds_upper.extend([1.0, 0.5])
            
            # Bounds for additional peaks
            for add_pos in additional_peak_positions:
                closest_idx = np.argmin(np.abs(wavenumbers - add_pos))
                peak_intensity = intensities[closest_idx]
                
                # Amplitude bounds
                bounds_lower.append(peak_intensity * 0.01)
                bounds_upper.append(peak_intensity * 10)
                
                # Center bounds (allow some movement around detected position)
                bounds_lower.append(add_pos - 25)
                bounds_upper.append(add_pos + 25)
                
                # Width bounds
                bounds_lower.append(3.0)
                bounds_upper.append(50.0)
                
                # Model-specific bounds
                if self.reference_model == "Pseudo-Voigt":
                    bounds_lower.append(0.0)
                    bounds_upper.append(1.0)
                elif self.reference_model == "Asymmetric Voigt":
                    bounds_lower.extend([0.0, -0.5])
                    bounds_upper.extend([1.0, 0.5])
            
            # Validate bounds
            bounds_lower = np.array(bounds_lower)
            bounds_upper = np.array(bounds_upper)
            
            invalid_bounds = bounds_lower >= bounds_upper
            if np.any(invalid_bounds):
                for i in np.where(invalid_bounds)[0]:
                    bounds_upper[i] = bounds_lower[i] * 2 + 1
            
            # Set up for hybrid fitting
            temp_wavenumbers = self.wavenumbers
            temp_peaks = self.peaks
            temp_model = self.current_model
            
            # Create combined peak list for model function
            all_peak_positions = list(self.reference_peaks) + additional_peak_positions
            
            self.wavenumbers = wavenumbers
            self.peaks = np.array(all_peak_positions)
            self.current_model = self.reference_model
            
            # Perform hybrid fitting
            hybrid_popt, hybrid_pcov = curve_fit(
                self.multi_peak_model,
                wavenumbers,
                intensities,
                p0=hybrid_initial_params,
                bounds=(bounds_lower, bounds_upper),
                max_nfev=8000,  # More iterations for complex fit
                method='trf'
            )
            
            # Calculate hybrid fit quality
            hybrid_fitted_curve = self.multi_peak_model(wavenumbers, *hybrid_popt)
            hybrid_residuals = intensities - hybrid_fitted_curve
            hybrid_ss_res = np.sum(hybrid_residuals ** 2)
            
            baseline_intensity = np.percentile(intensities, 5)
            signal_above_baseline = intensities - baseline_intensity
            ss_tot = np.sum(signal_above_baseline ** 2)
            
            if ss_tot > 1e-10:
                hybrid_r_squared = 1 - (hybrid_ss_res / ss_tot)
                hybrid_r_squared = max(0.0, min(1.0, hybrid_r_squared))
            else:
                hybrid_r_squared = 0.0
            
            # Extract fitted centers
            hybrid_fitted_centers = []
            for i in range(total_peaks):
                param_start = i * params_per_peak
                if param_start + 1 < len(hybrid_popt):
                    hybrid_fitted_centers.append(hybrid_popt[param_start + 1])
            
            # Restore original state
            self.wavenumbers = temp_wavenumbers
            self.peaks = temp_peaks
            self.current_model = temp_model
            
            return {
                'fit_params': hybrid_popt,
                'fitted_curve': hybrid_fitted_curve,
                'residuals': hybrid_residuals,
                'r_squared': hybrid_r_squared,
                'n_peaks': total_peaks,
                'fitted_centers': hybrid_fitted_centers
            }
            
        except Exception as e:
            print(f"Hybrid fitting failed: {e}")
            return None

    def _calculate_linear_background_for_data(self, data, start_weight, end_weight):
        """Calculate linear background for external data array."""
        try:
            # Use percentile-based endpoint estimation for robustness
            n_points = min(len(data) // 10, 50)
            
            start_region = data[:n_points] if len(data) > n_points else data[:len(data)//3]
            start_val = np.percentile(start_region, 10) * start_weight
            
            end_region = data[-n_points:] if len(data) > n_points else data[-len(data)//3:]
            end_val = np.percentile(end_region, 10) * end_weight
            
            return np.linspace(start_val, end_val, len(data))
        except:
            return np.linspace(data[0], data[-1], len(data))

    def _calculate_polynomial_background_for_data(self, data, poly_order, poly_method):
        """Calculate polynomial background for external data array."""
        try:
            x = np.arange(len(data))
            window_size = max(len(data) // 20, 5)
            
            baseline_candidates = []
            for i in range(len(data)):
                start_idx = max(0, i - window_size//2)
                end_idx = min(len(data), i + window_size//2 + 1)
                baseline_candidates.append(np.min(data[start_idx:end_idx]))
            
            baseline_y = np.array(baseline_candidates)
            poly_order = min(poly_order, len(x) - 1)
            
            if poly_method == "Robust":
                coeffs = np.polyfit(x, baseline_y, poly_order)
                for iteration in range(2):
                    poly_fit = np.polyval(coeffs, x)
                    residuals = np.abs(baseline_y - poly_fit)
                    mad = np.median(residuals)
                    weights = 1.0 / (1.0 + (residuals / (mad + 1e-10)))
                    coeffs = np.polyfit(x, baseline_y, poly_order, w=weights)
            else:
                coeffs = np.polyfit(x, baseline_y, poly_order)
            
            return np.polyval(coeffs, x)
        except:
            return np.linspace(data[0], data[-1], len(data))

    def _calculate_moving_average_background_for_data(self, data, window_percent, window_type):
        """Calculate moving average background for external data array."""
        try:
            window_size = max(int(len(data) * window_percent / 100.0), 3)
            
            baseline = np.array([np.min(data[max(0, i-window_size//2):min(len(data), i+window_size//2+1)]) 
                               for i in range(len(data))])
            
            if window_type == "Gaussian" and len(baseline) > 5:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    sigma = window_size / 6.0
                    baseline = gaussian_filter1d(baseline, sigma=sigma)
                except:
                    baseline = np.convolve(baseline, np.ones(min(window_size, len(baseline)))/(min(window_size, len(baseline))), mode='same')
            else:
                if window_size < len(baseline):
                    baseline = np.convolve(baseline, np.ones(window_size)/window_size, mode='same')
            
            return baseline
        except:
            return np.linspace(data[0], data[-1], len(data))

    def _calculate_spline_background(self, n_knots, smoothing, degree):
        """Calculate spline-based background that fits below the peaks for background subtraction."""
        try:
            from scipy.interpolate import UnivariateSpline
            
            # Create x values (indices or wavenumbers)
            x = np.arange(len(self.original_intensities))
            y = self.original_intensities
            
            # For proper background subtraction, we need to fit the baseline, not the peaks
            # Method: Use minimum filtering and iterative approach to fit below the data
            
            # Step 1: Apply a minimum filter to identify baseline regions
            from scipy import ndimage
            window_size = max(len(y) // 20, 5)  # Adaptive window size
            y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
            
            # Step 2: Create initial baseline estimate using the minimum filtered data
            if n_knots <= 2:
                n_knots = 3  # Minimum for spline
            
            # For background subtraction, use higher smoothing to avoid fitting peaks
            background_smoothing = max(smoothing, len(y) / 10)  # Ensure minimum smoothing
            
            try:
                # Fit spline to minimum filtered data for initial background estimate
                spline = UnivariateSpline(x, y_min_filtered, s=background_smoothing, k=min(degree, 3))
                initial_background = spline(x)
                
                # Step 3: Iterative refinement - only use points below or near the background
                current_background = initial_background.copy()
                
                for iteration in range(3):  # Limited iterations
                    # Identify points that are likely background (below or close to current estimate)
                    threshold = np.percentile(y - current_background, 20)  # Use 20th percentile
                    mask = (y - current_background) <= threshold
                    
                    if np.sum(mask) < n_knots:  # Need enough points
                        break
                    
                    # Fit spline only to identified background points
                    spline = UnivariateSpline(x[mask], y[mask], s=background_smoothing, k=min(degree, 3))
                    current_background = spline(x)
                    
                    # Ensure background doesn't go above data unrealistically
                    current_background = np.minimum(current_background, y)
                
                # Final constraint: background should be below the data
                background = np.minimum(current_background, y)
                
                return background
                
            except Exception:
                # Fallback: use simple percentile-based baseline
                from scipy.signal import savgol_filter
                
                # Use Savitzky-Golay filter on minimum filtered data
                window_length = min(len(y) // 5, 51)
                if window_length % 2 == 0:
                    window_length += 1  # Must be odd
                
                background = savgol_filter(y_min_filtered, window_length, polyorder=min(degree, 3))
                return np.minimum(background, y)
                
        except ImportError:
            # If scipy is not available, fallback to simple polynomial on minimum values
            print("scipy not available, using polynomial fallback for spline background")
            
            # Simple approach: fit polynomial to lower envelope
            window_size = max(len(y) // 10, 3)
            y_smooth = np.array([np.min(y[max(0, i-window_size):min(len(y), i+window_size+1)]) 
                               for i in range(len(y))])
            
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y_smooth, min(degree, 3))
            background = np.polyval(coeffs, x)
            
            return np.minimum(background, y)
            
        except Exception as e:
            print(f"Spline background calculation error: {str(e)}")
            # Final fallback: simple linear baseline
            background = np.linspace(y[0], y[-1], len(y))
            return np.minimum(background, y)

    def fit_peaks(self):
        """Improved peak fitting with better parameter estimation."""
        if len(self.peaks) == 0:
            QMessageBox.warning(self, "No Peaks", "Detect peaks first.")
            return
            
        try:
            # Create initial parameter guesses with improved estimates
            initial_params = []
            bounds_lower = []
            bounds_upper = []
            
            # Calculate average peak spacing for width estimation
            if len(self.peaks) > 1:
                peak_wavenumbers = [self.wavenumbers[idx] for idx in self.peaks if 0 <= idx < len(self.wavenumbers)]
                if len(peak_wavenumbers) > 1:
                    avg_spacing = np.mean(np.diff(sorted(peak_wavenumbers)))
                    estimated_width = avg_spacing / 4.0  # Start with narrower peaks
                else:
                    estimated_width = 15.0
            else:
                estimated_width = 15.0
            
            for peak_idx in self.peaks:
                if 0 <= peak_idx < len(self.wavenumbers):
                    # Amplitude - use the peak height
                    amp = max(self.intensities[peak_idx], 1.0)  # Ensure positive
                    
                    # Center
                    cen = self.wavenumbers[peak_idx]
                    
                    # Width - improved estimation
                    wid = estimated_width
                    
                    # Base parameters for all models
                    initial_params.extend([amp, cen, wid])
                    
                    # More generous bounds for better convergence
                    bounds_lower.extend([amp * 0.01, cen - wid * 5, wid * 0.1])
                    bounds_upper.extend([amp * 50, cen + wid * 5, wid * 10])
                    
                    # Additional parameters for Voigt models
                    if self.current_model == "Pseudo-Voigt":
                        # Add eta parameter (mixing parameter)
                        initial_params.append(0.5)  # Start with 50% Gaussian, 50% Lorentzian
                        bounds_lower.append(0.0)    # Pure Gaussian
                        bounds_upper.append(1.0)    # Pure Lorentzian
                    elif self.current_model == "Asymmetric Voigt":
                        # Add eta and asym parameters
                        initial_params.extend([0.5, 0.0])  # eta=0.5, no asymmetry
                        bounds_lower.extend([0.0, -0.5])   # eta bounds and asymmetry bounds
                        bounds_upper.extend([1.0, 0.5])
                    
            if not initial_params:
                QMessageBox.warning(self, "Error", "No valid peaks for fitting.")
                return
                
            # Fit the peaks with improved settings
            bounds = (bounds_lower, bounds_upper)
            popt, pcov = curve_fit(
                self.multi_peak_model, 
                self.wavenumbers, 
                self.intensities,
                p0=initial_params,
                bounds=bounds,
                max_nfev=5000,  # Increased iterations
                method='trf'    # Trust Region Reflective algorithm
            )
            
            self.fit_params = popt
            self.fit_result = True
            
            # Calculate residuals - ensure we have valid fit_params before proceeding
            if (self.fit_params is not None and 
                hasattr(self.fit_params, '__len__') and 
                len(self.fit_params) > 0):
                fitted_curve = self.multi_peak_model(self.wavenumbers, *popt)
                self.residuals = self.intensities - fitted_curve
                
                # Update the current plot and peak info display
                self.update_current_plot()
                self.update_peak_info_display()
            else:
                QMessageBox.warning(self, "Error", "Invalid fit parameters generated.")
                return
            
            # Calculate R-squared using improved peak-based method
            individual_peak_r2, peak_avg_r2 = self.calculate_individual_peak_r_squared(
                self.wavenumbers, self.intensities, self.fit_params, self.current_model
            )
            
            # Use peak-based R² as primary metric
            r_squared = peak_avg_r2
            
            # Fallback to traditional R² if peak-based fails
            if r_squared <= 0.01 or len(individual_peak_r2) == 0:
                print("Peak-based R² calculation failed, using traditional method...")
                ss_res = np.sum(self.residuals ** 2)
                baseline_intensity = np.percentile(self.intensities, 5)
                signal_above_baseline = self.intensities - baseline_intensity
                ss_tot = np.sum(signal_above_baseline ** 2)
                
                if ss_tot > 1e-10:
                    r_squared = 1 - (ss_res / ss_tot)
                    r_squared = max(0.0, min(1.0, r_squared))
                else:
                    r_squared = 0.0
            
            # Create detailed results message
            if len(individual_peak_r2) > 0:
                peak_r2_str = ', '.join([f'{x:.3f}' for x in individual_peak_r2])
                message = (f"Peak fitting completed!\n"
                          f"Average Peak R² = {r_squared:.4f}\n"
                          f"Individual Peak R²: [{peak_r2_str}]\n"
                          f"Fitted {len(self.peaks)} peaks")
            else:
                message = (f"Peak fitting completed!\n"
                          f"R² = {r_squared:.4f}\n"
                          f"Fitted {len(self.peaks)} peaks")
            
            QMessageBox.information(self, "Success", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Peak fitting failed: {str(e)}")

    def gaussian(self, x, amp, cen, wid):
        """Gaussian peak function."""
        return amp * np.exp(-((x - cen) / wid)**2)
        
    def lorentzian(self, x, amp, cen, wid):
        """Lorentzian peak function."""
        return amp / (1 + ((x - cen) / wid)**2)
    
    def pseudo_voigt(self, x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function (linear combination of Gaussian and Lorentzian).
        
        Args:
            x: input array
            amp: amplitude
            cen: center position
            wid: width parameter
            eta: mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
        """
        eta = np.clip(eta, 0, 1)  # Ensure eta is between 0 and 1
        gaussian_part = np.exp(-((x - cen) / wid)**2)
        lorentzian_part = 1 / (1 + ((x - cen) / wid)**2)
        return amp * (eta * lorentzian_part + (1 - eta) * gaussian_part)
    
    def asymmetric_voigt(self, x, amp, cen, wid, eta=0.5, asym=0.0):
        """Asymmetric Voigt peak function.
        
        Args:
            x: input array
            amp: amplitude
            cen: center position
            wid: width parameter
            eta: mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
            asym: asymmetry parameter (controls peak shape asymmetry)
        """
        eta = np.clip(eta, 0, 1)  # Ensure eta is between 0 and 1
        
        # Apply asymmetry by modifying the width on different sides of the peak
        dx = x - cen
        wid_eff = wid * (1 + asym * np.sign(dx))
        wid_eff = np.maximum(wid_eff, 0.1)  # Prevent negative or zero widths
        
        # Calculate pseudo-Voigt with asymmetric widths
        gaussian_part = np.exp(-((dx / wid_eff)**2))
        lorentzian_part = 1 / (1 + (dx / wid_eff)**2)
        return amp * (eta * lorentzian_part + (1 - eta) * gaussian_part)
        
    def multi_peak_model(self, x, *params):
        """Multi-peak model function."""
        # Determine parameters per peak based on model type
        if self.current_model in ["Pseudo-Voigt", "Asymmetric Voigt"]:
            if self.current_model == "Pseudo-Voigt":
                params_per_peak = 4  # amp, cen, wid, eta
            else:  # Asymmetric Voigt
                params_per_peak = 5  # amp, cen, wid, eta, asym
        else:
            params_per_peak = 3  # amp, cen, wid for Gaussian and Lorentzian
        
        # Calculate number of peaks from parameters
        n_peaks = len(params) // params_per_peak
        
        if n_peaks == 0:
            return np.zeros_like(x)
            
        model = np.zeros_like(x)
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            
            if start_idx + (params_per_peak - 1) < len(params):
                if self.current_model == "Gaussian":
                    amp, cen, wid = params[start_idx:start_idx+3]
                    wid = max(abs(wid), 1.0)  # Ensure positive width
                    component = self.gaussian(x, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    amp, cen, wid = params[start_idx:start_idx+3]
                    wid = max(abs(wid), 1.0)  # Ensure positive width
                    component = self.lorentzian(x, amp, cen, wid)
                elif self.current_model == "Pseudo-Voigt":
                    amp, cen, wid, eta = params[start_idx:start_idx+4]
                    wid = max(abs(wid), 1.0)  # Ensure positive width
                    eta = np.clip(eta, 0, 1)  # Ensure eta is between 0 and 1
                    component = self.pseudo_voigt(x, amp, cen, wid, eta)
                elif self.current_model == "Asymmetric Voigt":
                    amp, cen, wid, eta, asym = params[start_idx:start_idx+5]
                    wid = max(abs(wid), 1.0)  # Ensure positive width
                    eta = np.clip(eta, 0, 1)  # Ensure eta is between 0 and 1
                    asym = np.clip(asym, -0.5, 0.5)  # Limit asymmetry parameter
                    component = self.asymmetric_voigt(x, amp, cen, wid, eta, asym)
                else:
                    # Default to Gaussian for unknown models
                    amp, cen, wid = params[start_idx:start_idx+3]
                    wid = max(abs(wid), 1.0)  # Ensure positive width
                    component = self.gaussian(x, amp, cen, wid)
                    
                model += component
                
        return model

    def calculate_individual_peak_r_squared(self, wavenumbers, intensities, fit_params, model='Gaussian'):
        """
        Calculate R² for each individual peak and overall peak-based R².
        
        Args:
            wavenumbers: array of wavenumber values
            intensities: array of measured intensity values
            fit_params: fitted parameters for all peaks
            model: peak model type ('Gaussian', 'Lorentzian', etc.)
            
        Returns:
            tuple: (individual_peak_r2_list, average_peak_r2)
        """
        try:
            if len(fit_params) == 0 or len(wavenumbers) != len(intensities):
                return [], 0.0
            
            # Determine number of parameters per peak based on model
            if model == "Pseudo-Voigt":
                params_per_peak = 4
            elif model == "Asymmetric Voigt":
                params_per_peak = 5
            else:
                params_per_peak = 3  # Gaussian, Lorentzian
            
            n_peaks = len(fit_params) // params_per_peak
            if n_peaks == 0:
                return [], 0.0
            
            individual_r2_values = []
            
            # Calculate overall mean for SS_tot calculation
            mean_intensity = np.mean(intensities)
            ss_tot = np.sum((intensities - mean_intensity) ** 2)
            
            # Calculate R² for each individual peak
            for i in range(n_peaks):
                start_idx = i * params_per_peak
                
                try:
                    # Extract peak parameters
                    if model == "Gaussian":
                        if start_idx + 2 < len(fit_params):
                            amp, cen, wid = fit_params[start_idx:start_idx+3]
                            peak_curve = self.gaussian(wavenumbers, amp, cen, wid)
                        else:
                            individual_r2_values.append(0.0)
                            continue
                    elif model == "Lorentzian":
                        if start_idx + 2 < len(fit_params):
                            amp, cen, wid = fit_params[start_idx:start_idx+3]
                            peak_curve = self.lorentzian(wavenumbers, amp, cen, wid)
                        else:
                            individual_r2_values.append(0.0)
                            continue
                    elif model == "Pseudo-Voigt":
                        if start_idx + 3 < len(fit_params):
                            amp, cen, wid, eta = fit_params[start_idx:start_idx+4]
                            peak_curve = self.pseudo_voigt(wavenumbers, amp, cen, wid, eta)
                        else:
                            individual_r2_values.append(0.0)
                            continue
                    elif model == "Asymmetric Voigt":
                        if start_idx + 4 < len(fit_params):
                            amp, cen, wid, eta, asym = fit_params[start_idx:start_idx+5]
                            peak_curve = self.asymmetric_voigt(wavenumbers, amp, cen, wid, eta, asym)
                        else:
                            individual_r2_values.append(0.0)
                            continue
                    else:
                        # Default to Gaussian
                        if start_idx + 2 < len(fit_params):
                            amp, cen, wid = fit_params[start_idx:start_idx+3]
                            peak_curve = self.gaussian(wavenumbers, amp, cen, wid)
                        else:
                            individual_r2_values.append(0.0)
                            continue
                    
                    # Define peak region for focused R² calculation
                    # Use 3*width around peak center for evaluation
                    peak_center = cen
                    peak_width = abs(wid)
                    
                    # Find indices within peak region
                    peak_region_mask = np.abs(wavenumbers - peak_center) <= (3 * peak_width)
                    
                    if not np.any(peak_region_mask):
                        # If no points in region, use broader approach
                        peak_region_mask = np.abs(wavenumbers - peak_center) <= (5 * peak_width)
                    
                    if not np.any(peak_region_mask):
                        # If still no points, use whole spectrum
                        peak_region_mask = np.ones_like(wavenumbers, dtype=bool)
                    
                    # Extract data in peak region
                    region_wavenumbers = wavenumbers[peak_region_mask]
                    region_intensities = intensities[peak_region_mask]
                    region_peak_curve = peak_curve[peak_region_mask]
                    
                    if len(region_intensities) == 0:
                        individual_r2_values.append(0.0)
                        continue
                    
                    # Calculate R² for this peak in its region
                    region_mean = np.mean(region_intensities)
                    ss_res = np.sum((region_intensities - region_peak_curve) ** 2)
                    ss_tot_region = np.sum((region_intensities - region_mean) ** 2)
                    
                    if ss_tot_region > 1e-10:
                        peak_r2 = 1 - (ss_res / ss_tot_region)
                        # Clamp R² to reasonable range
                        peak_r2 = max(0.0, min(1.0, peak_r2))
                    else:
                        peak_r2 = 0.0
                    
                    individual_r2_values.append(peak_r2)
                    
                except Exception as e:
                    print(f"Error calculating R² for peak {i+1}: {e}")
                    individual_r2_values.append(0.0)
            
            # Calculate average R² across all peaks
            if individual_r2_values:
                # Weight by peak amplitudes for more meaningful average
                amplitudes = []
                for i in range(n_peaks):
                    start_idx = i * params_per_peak
                    if start_idx < len(fit_params):
                        amplitudes.append(abs(fit_params[start_idx]))  # amplitude is first parameter
                    else:
                        amplitudes.append(0.0)
                
                if sum(amplitudes) > 1e-10:
                    # Weighted average by amplitude
                    weights = np.array(amplitudes) / sum(amplitudes)
                    average_r2 = np.sum(np.array(individual_r2_values) * weights)
                else:
                    # Simple average if no amplitude weighting possible
                    average_r2 = np.mean(individual_r2_values)
            else:
                average_r2 = 0.0
            
            return individual_r2_values, float(average_r2)
            
        except Exception as e:
            print(f"Error in calculate_individual_peak_r_squared: {e}")
            return [], 0.0

    # Batch processing methods
    def set_reference(self):
        """Set current spectrum as reference for batch processing with full fitted parameters."""
        # Combine auto-detected and manual peaks
        all_peaks = np.concatenate([self.peaks, self.manual_peaks]) if len(self.manual_peaks) > 0 else self.peaks
        
        if len(all_peaks) == 0:
            QMessageBox.warning(self, "No Peaks", "Detect and fit peaks first.")
            return
            
        if not self.fit_result or len(self.fit_params) == 0:
            QMessageBox.warning(self, "No Fit", "Fit peaks first.")
            return
            
        # Store both peak indices and fitted parameters
        self.reference_peaks = all_peaks.copy()
        self.reference_fit_params = self.fit_params.copy()
        self.reference_model = self.current_model
        self.reference_wavenumbers = self.wavenumbers.copy()
        self.reference_background = self.background.copy() if (hasattr(self, 'background') and self.background is not None) else None
        
        # Extract peak positions from fitted parameters for validation
        params_per_peak = self.get_params_per_peak()
        n_peaks = len(self.fit_params) // params_per_peak
        peak_positions = []
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            if start_idx + 1 < len(self.fit_params):  # Need at least amp, center
                center = self.fit_params[start_idx + 1]  # Center is second parameter
                peak_positions.append(center)
        
        QMessageBox.information(self, "Reference Set", 
                              f"Set reference with {n_peaks} fitted peaks at positions: {[f'{pos:.1f}' for pos in peak_positions]} cm⁻¹")
        
        self.batch_status_text.append(f"Reference set: {n_peaks} fitted peaks from {os.path.basename(self.spectra_files[self.current_spectrum_index])}")
        self.batch_status_text.append(f"Peak positions: {', '.join([f'{pos:.1f}' for pos in peak_positions])} cm⁻¹")
        
    def apply_to_all(self):
        """
        Apply reference parameters to all loaded spectra using optimized workflow.
        
        NEW IMPROVED ORDER OF OPERATIONS:
        1. STEP 1: Optimize background parameters for each individual spectrum
                   - Uses UI method as starting point
                   - Tries multiple parameter combinations 
                   - Evaluates background quality by peak fitting performance
                   - Selects optimal background parameters for this spectrum
                   
        2. STEP 2: Fit peaks on the optimized background-corrected spectrum
                   - Uses reference peak parameters as initial guess
                   - Performs peak fitting on optimally background-corrected data
                   
        This ensures proper background subtraction BEFORE peak fitting,
        rather than using fixed UI parameters for all spectra.
        """
        # Safely check reference_peaks and reference_fit_params
        if (not hasattr(self, 'reference_peaks') or 
            self.reference_peaks is None or 
            (hasattr(self.reference_peaks, '__len__') and len(self.reference_peaks) == 0) or
            not hasattr(self, 'reference_fit_params') or 
            len(self.reference_fit_params) == 0):
            QMessageBox.warning(self, "No Reference", "Set a reference spectrum first using 'Set Reference'.")
            return
            
        if not self.spectra_files:
            QMessageBox.warning(self, "No Files", "Load spectrum files first.")
            return
            
        self._stop_batch = False
        self.batch_results = []
        
        # Clear any background preview state to avoid confusion
        self.background_preview_active = False
        self.background_preview = None
        
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
                # Check if this spectrum already has a manual fit
                existing_manual_fit = None
                for existing_result in self.batch_results:
                    if (existing_result.get('file') == file_path and 
                        existing_result.get('manual_fit', False)):
                        existing_manual_fit = existing_result
                        break
                
                if existing_manual_fit:
                    # Skip this spectrum - it has been manually fitted
                    self.batch_status_text.append(f"SKIPPED: {os.path.basename(file_path)} - Manual fit preserved")
                    
                    # Update live view with the existing manual fit
                    self.update_live_view(existing_manual_fit, 
                                        existing_manual_fit.get('wavenumbers', []), 
                                        existing_manual_fit.get('fit_params', []))
                    continue
                
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
                
                # STEP 1: Optimize background for this specific spectrum
                print(f"Optimizing background for {os.path.basename(file_path)}...")
                bg_method = self.bg_method_combo.currentText()
                
                # Get optimized background parameters and corrected intensities
                try:
                    intensities, background, bg_params_used = self._optimize_background_for_spectrum(
                        wavenumbers, original_intensities, bg_method
                    )
                    print(f"  Background optimized using {bg_method} with params: {bg_params_used}")
                    
                except Exception as bg_error:
                    print(f"Background optimization failed: {bg_error}")
                    # Fallback to simple linear background
                    background = np.linspace(original_intensities[0], original_intensities[-1], len(original_intensities))
                    intensities = original_intensities - background
                    bg_params_used = "Linear fallback"
                
                # STEP 2: Fit peaks on the optimized background-corrected spectrum
                print(f"  Fitting peaks on optimized background...")
                
                # Use fitted reference parameters as starting guess
                if not hasattr(self, 'reference_fit_params') or len(self.reference_fit_params) == 0:
                    self.batch_results.append({
                        'file': file_path,
                        'fit_failed': True,
                        'error': 'No reference fit parameters available'
                    })
                    continue
                
                initial_params = []
                bounds_lower = []
                bounds_upper = []
                
                # Extract parameters from reference fit
                params_per_peak = 4 if self.reference_model == "Pseudo-Voigt" else 5 if self.reference_model == "Asymmetric Voigt" else 3
                n_peaks = len(self.reference_fit_params) // params_per_peak
                
                for peak_i in range(n_peaks):
                    start_idx = peak_i * params_per_peak
                    
                    if start_idx + 2 < len(self.reference_fit_params):  # Need at least amp, center, width
                        # Extract reference parameters
                        ref_amp = self.reference_fit_params[start_idx]
                        ref_cen = self.reference_fit_params[start_idx + 1] 
                        ref_wid = self.reference_fit_params[start_idx + 2]
                        
                        # Find closest point in new spectrum to reference center
                        closest_idx = np.argmin(np.abs(wavenumbers - ref_cen))
                        
                        # Use current spectrum intensity at closest position, but scale based on reference
                        current_amp = intensities[closest_idx] if 0 <= closest_idx < len(intensities) else ref_amp
                        
                        # Adjust amplitude based on intensity ratio but keep reference as guide
                        if abs(ref_amp) > 1e-6:
                            amp_scale = abs(current_amp) / abs(ref_amp)
                            amp_scale = np.clip(amp_scale, 0.1, 10.0)  # Reasonable scaling limits
                            initial_amp = ref_amp * amp_scale
                        else:
                            initial_amp = abs(current_amp)
                        
                        # Use reference center as starting point (should be close)
                        initial_cen = ref_cen
                        
                        # Use reference width as starting point
                        initial_wid = abs(ref_wid)
                        
                        # Create generous bounds around reference values
                        min_amp = abs(initial_amp) * 0.01
                        max_amp = abs(initial_amp) * 100
                        
                        min_cen = ref_cen - abs(ref_wid) * 10  # Allow significant position variation
                        max_cen = ref_cen + abs(ref_wid) * 10
                        
                        min_wid = abs(ref_wid) * 0.1
                        max_wid = abs(ref_wid) * 10
                        
                        # Add base parameters
                        initial_params.extend([abs(initial_amp), initial_cen, initial_wid])
                        bounds_lower.extend([min_amp, min_cen, min_wid])
                        bounds_upper.extend([max_amp, max_cen, max_wid])
                        
                        # Add additional parameters for Voigt models
                        if self.reference_model == "Pseudo-Voigt" and start_idx + 3 < len(self.reference_fit_params):
                            ref_eta = self.reference_fit_params[start_idx + 3]
                            initial_params.append(ref_eta)
                            bounds_lower.append(0.0)
                            bounds_upper.append(1.0)
                        elif self.reference_model == "Asymmetric Voigt" and start_idx + 4 < len(self.reference_fit_params):
                            ref_eta = self.reference_fit_params[start_idx + 3]
                            ref_asym = self.reference_fit_params[start_idx + 4]
                            initial_params.extend([ref_eta, ref_asym])
                            bounds_lower.extend([0.0, -0.5])
                            bounds_upper.extend([1.0, 0.5])
                
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
                        temp_model = self.current_model
                        
                        self.wavenumbers = wavenumbers
                        self.peaks = self.reference_peaks
                        self.current_model = self.reference_model
                        
                        bounds = (bounds_lower, bounds_upper)
                        popt, pcov = curve_fit(
                            self.multi_peak_model, 
                            wavenumbers, 
                            intensities,
                            p0=initial_params,
                            bounds=bounds,
                            max_nfev=5000,  # Increased iterations
                            method='trf'
                        )
                        
                        # Restore original data
                        self.wavenumbers = temp_wavenumbers
                        self.peaks = temp_peaks
                        self.current_model = temp_model
                        
                        # Calculate R-squared using improved peak-based method
                        fitted_curve = self.multi_peak_model(wavenumbers, *popt)
                        residuals = intensities - fitted_curve
                        
                        # NEW: Calculate peak-based R² for better quality assessment
                        individual_peak_r2, peak_avg_r2 = self.calculate_individual_peak_r_squared(
                            wavenumbers, intensities, popt, self.reference_model
                        )
                        
                        # For weak data, weight individual peak R² more heavily
                        # Use composite scoring that prioritizes peak fitting over background fitting
                        if len(individual_peak_r2) > 0 and peak_avg_r2 > 0.05:  # Lower threshold for weak data
                            # Calculate traditional R² for reference
                            ss_res = np.sum(residuals ** 2)
                            baseline_intensity = np.percentile(intensities, 5)
                            signal_above_baseline = intensities - baseline_intensity
                            ss_tot = np.sum(signal_above_baseline ** 2)
                            
                            traditional_r2 = 0.0
                            if ss_tot > 1e-10:
                                traditional_r2 = 1 - (ss_res / ss_tot)
                                traditional_r2 = max(0.0, min(1.0, traditional_r2))
                            
                            # For weak data, use weighted combination favoring peak-based R²
                            # Peak-based R² gets 70% weight, traditional gets 30%
                            r_squared = 0.7 * peak_avg_r2 + 0.3 * traditional_r2
                            r_squared = max(0.0, min(1.0, r_squared))
                            
                            print(f"  Peak-based R²: {peak_avg_r2:.3f} (individual: {[f'{x:.2f}' for x in individual_peak_r2]})")
                            print(f"  Traditional R²: {traditional_r2:.3f}")
                            print(f"  Composite R² (70% peak + 30% traditional): {r_squared:.3f}")
                        else:
                            print(f"  Peak-based R² failed ({peak_avg_r2:.3f}), using traditional method...")
                            # Traditional R² calculation as fallback
                            ss_res = np.sum(residuals ** 2)
                            baseline_intensity = np.percentile(intensities, 5)
                            signal_above_baseline = intensities - baseline_intensity
                            ss_tot = np.sum(signal_above_baseline ** 2)
                            
                            if ss_tot > 1e-10:
                                r_squared = 1 - (ss_res / ss_tot)
                                r_squared = max(0.0, min(1.0, r_squared))
                            else:
                                r_squared = 0.0
                        
                        # Extract peak centers for validation
                        fitted_centers = []
                        for peak_i in range(n_peaks):
                            param_start = peak_i * params_per_peak
                            if param_start + 1 < len(popt):
                                fitted_centers.append(popt[param_start + 1])
                        
                        # If R² is poor, try optimizing parameters within the SAME background method only
                        # Use lower threshold for weak data to avoid over-optimization
                        if r_squared < 0.75:  # More reasonable threshold for weak data
                            print(f"Poor fit (R²={r_squared:.3f}) for {os.path.basename(file_path)}, optimizing within chosen method...")
                            print(f"  RESPECTING USER'S CHOICE: Optimizing parameters within {bg_method} method only")
                            
                            # RESPECT USER'S BACKGROUND CHOICE - only try different parameters of the same method
                            fallback_tried = False
                            
                            # Try more aggressive parameter optimization within the chosen method
                            try:
                                if bg_method.startswith("ALS"):
                                    # Try more aggressive ALS parameters for weak data
                                    aggressive_als_params = [
                                        (1e7, 0.0001, 20),   # Very stiff, high asymmetry
                                        (1e4, 0.01, 25),     # More flexible, moderate asymmetry  
                                        (5e6, 0.0005, 30),   # Balanced approach
                                        (1e6, 0.005, 15),    # Moderate stiffness
                                    ]
                                    
                                    for lam, p, niter in aggressive_als_params:
                                        try:
                                            fb_background = self.baseline_als(original_intensities, lam, p, niter)
                                            fb_intensities = original_intensities - fb_background
                                            
                                            # Try fitting with this background
                                            fb_r_squared = self._try_peak_fitting_with_background(
                                                wavenumbers, fb_intensities, fb_background, bounds, params_per_peak, n_peaks
                                            )
                                            
                                            if fb_r_squared and fb_r_squared > r_squared + 0.03:  # Modest improvement threshold
                                                print(f"Optimized ALS (λ={lam:.0e}, p={p:.4f}, iter={niter}) improved R² to {fb_r_squared:.3f}")
                                                # Update with optimized results
                                                intensities = fb_intensities
                                                background = fb_background  
                                                r_squared = fb_r_squared
                                                fallback_tried = True
                                                break
                                        except:
                                            continue
                                
                                elif bg_method == "Linear":
                                    # Try different linear endpoint weightings for weak data
                                    linear_params = [
                                        (0.1, 0.9),   # Strong left weighting
                                        (0.9, 0.1),   # Strong right weighting
                                        (0.2, 0.8),   # Moderate left weighting
                                        (0.8, 0.2),   # Moderate right weighting
                                        (0.5, 0.5),   # Equal weighting
                                    ]
                                    
                                    for start_w, end_w in linear_params:
                                        try:
                                            fb_background = self._calculate_linear_background_for_data(original_intensities, start_w, end_w)
                                            fb_intensities = original_intensities - fb_background
                                            
                                            fb_r_squared = self._try_peak_fitting_with_background(
                                                wavenumbers, fb_intensities, fb_background, bounds, params_per_peak, n_peaks
                                            )
                                            
                                            if fb_r_squared and fb_r_squared > r_squared + 0.03:
                                                print(f"Optimized Linear (start={start_w:.1f}, end={end_w:.1f}) improved R² to {fb_r_squared:.3f}")
                                                intensities = fb_intensities
                                                background = fb_background
                                                r_squared = fb_r_squared
                                                fallback_tried = True
                                                break
                                        except:
                                            continue
                                
                                elif bg_method == "Polynomial":
                                    # Try different polynomial orders and methods for weak data
                                    poly_params = [
                                        (1, "percentile"),
                                        (2, "robust"),
                                        (3, "percentile"),
                                        (2, "endpoints"),
                                        (4, "robust"),
                                    ]
                                    
                                    for order, method in poly_params:
                                        try:
                                            fb_background = self._calculate_polynomial_background_for_data(original_intensities, order, method)
                                            fb_intensities = original_intensities - fb_background
                                            
                                            fb_r_squared = self._try_peak_fitting_with_background(
                                                wavenumbers, fb_intensities, fb_background, bounds, params_per_peak, n_peaks
                                            )
                                            
                                            if fb_r_squared and fb_r_squared > r_squared + 0.03:
                                                print(f"Optimized Polynomial (order={order}, method={method}) improved R² to {fb_r_squared:.3f}")
                                                intensities = fb_intensities
                                                background = fb_background
                                                r_squared = fb_r_squared
                                                fallback_tried = True
                                                break
                                        except:
                                            continue
                            
                            except Exception as opt_error:
                                print(f"Parameter optimization failed: {opt_error}")
                            
                            if not fallback_tried:
                                print(f"Parameter optimization within {bg_method} did not improve fit for {os.path.basename(file_path)}")
                        
                        # STEP 3: Try hybrid approach - detect additional peaks not in reference
                        final_popt = popt
                        final_r_squared = r_squared
                        final_fitted_curve = fitted_curve
                        final_residuals = residuals
                        final_n_peaks = n_peaks
                        final_fitted_centers = fitted_centers
                        
                        # If the fit isn't perfect, look for additional peaks in residuals
                        if r_squared < 0.95:
                            try:
                                additional_peaks = self._detect_additional_peaks(
                                    wavenumbers, intensities, fitted_curve, residuals
                                )
                                
                                if len(additional_peaks) > 0:
                                    print(f"  Found {len(additional_peaks)} additional peaks at {[f'{p:.1f}' for p in additional_peaks]} cm⁻¹")
                                    
                                    # Refit with both reference and additional peaks
                                    hybrid_result = self._fit_hybrid_peaks(
                                        wavenumbers, intensities, popt, additional_peaks
                                    )
                                    
                                    if hybrid_result and hybrid_result['r_squared'] > r_squared + 0.02:  # Significant improvement
                                        print(f"  Hybrid fit improved R² from {r_squared:.3f} to {hybrid_result['r_squared']:.3f}")
                                        final_popt = hybrid_result['fit_params']
                                        final_r_squared = hybrid_result['r_squared']
                                        final_fitted_curve = hybrid_result['fitted_curve']
                                        final_residuals = hybrid_result['residuals']
                                        final_n_peaks = hybrid_result['n_peaks']
                                        final_fitted_centers = hybrid_result['fitted_centers']
                                    else:
                                        print(f"  Hybrid fit did not improve results")
                                        
                            except Exception as hybrid_error:
                                print(f"  Hybrid peak detection failed: {hybrid_error}")
                        
                        # Calculate individual peak R² for storage
                        try:
                            individual_r2_storage, _ = self.calculate_individual_peak_r_squared(
                                wavenumbers, intensities, final_popt, self.reference_model
                            )
                        except:
                            individual_r2_storage = []
                        
                        # Store results
                        result_data = {
                            'file': file_path,
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'original_intensities': original_intensities,
                            'background': background,
                            'peaks': self.reference_peaks.copy(),
                            'fit_params': final_popt,
                            'fit_cov': pcov,
                            'residuals': final_residuals,
                            'r_squared': final_r_squared,
                            'peak_based_r_squared': True,  # Flag to indicate we used peak-based R²
                            'individual_peak_r2': individual_r2_storage,  # Store individual peak R² values
                            'fitted_curve': final_fitted_curve,
                            'n_peaks_fitted': final_n_peaks,
                            'fitted_centers': final_fitted_centers,
                            'fit_failed': False
                        }
                        
                        self.batch_results.append(result_data)
                        
                        # Update live view and peak info panel
                        self.update_live_view(result_data, wavenumbers, popt)
                        self.update_peak_info_display_for_batch_result(result_data)
                        
                        # Validate peak count
                        expected_peaks = len(self.reference_fit_params) // params_per_peak
                        centers_str = ', '.join([f'{c:.1f}' for c in fitted_centers])
                        
                        if len(fitted_centers) != expected_peaks:
                            self.batch_status_text.append(f"WARNING: {os.path.basename(file_path)} - Expected {expected_peaks} peaks, fitted {len(fitted_centers)}")
                        
                        status_color = "SUCCESS" if final_r_squared >= 0.85 else "POOR"
                        
                        # Enhanced status message with peak-based R² information
                        if len(individual_r2_storage) > 0:
                            min_peak_r2 = min(individual_r2_storage)
                            max_peak_r2 = max(individual_r2_storage)
                            individual_r2_str = ', '.join([f'{x:.2f}' for x in individual_r2_storage])
                            self.batch_status_text.append(f"{status_color}: {os.path.basename(file_path)} - Peak-Avg R² = {final_r_squared:.4f}")
                            self.batch_status_text.append(f"  Individual Peak R²: [{individual_r2_str}] (range: {min_peak_r2:.2f}-{max_peak_r2:.2f})")
                            self.batch_status_text.append(f"  {len(fitted_centers)} peaks at {centers_str} cm⁻¹")
                        else:
                            self.batch_status_text.append(f"{status_color}: {os.path.basename(file_path)} - R² = {final_r_squared:.4f}, {len(fitted_centers)} peaks at {centers_str} cm⁻¹")
                        self.batch_status_text.append(f"  Background: {bg_params_used}")
                        
                        # Restore original data after successful fit
                        self.wavenumbers = temp_wavenumbers
                        self.peaks = temp_peaks
                        self.current_model = temp_model
                        
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
                                
                                # Calculate R-squared using improved peak-based method
                                fitted_curve = self.multi_peak_model(wavenumbers, *popt)
                                residuals = intensities - fitted_curve
                                
                                # Use peak-based R² for better quality assessment
                                individual_peak_r2, peak_avg_r2 = self.calculate_individual_peak_r_squared(
                                    wavenumbers, intensities, popt, self.reference_model
                                )
                                
                                r_squared = peak_avg_r2
                                
                                # Fallback to traditional R² if needed
                                if r_squared <= 0.01 or len(individual_peak_r2) == 0:
                                    ss_res = np.sum(residuals ** 2)
                                    baseline_intensity = np.percentile(intensities, 5)
                                    signal_above_baseline = intensities - baseline_intensity
                                    ss_tot = np.sum(signal_above_baseline ** 2)
                                    
                                    if ss_tot > 1e-10:
                                        r_squared = 1 - (ss_res / ss_tot)
                                        r_squared = max(0.0, min(1.0, r_squared))
                                    else:
                                        r_squared = 0.0
                                
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
                                
                                # Update live view and peak info panel
                                self.update_live_view(result_data, wavenumbers, popt)
                                self.update_peak_info_display_for_batch_result(result_data)
                                
                                self.batch_status_text.append(f"SUCCESS (fallback): {os.path.basename(file_path)} - R² = {r_squared:.4f}")
                                self.batch_status_text.append(f"  Background: {bg_params_used}")
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
                            self.batch_status_text.append(f"  Background was: {bg_params_used}")
                            
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
                    self.batch_status_text.append(f"  Background was: {bg_params_used}")
                    
            except Exception as e:
                self.batch_results.append({
                    'file': file_path,
                    'fit_failed': True,
                    'error': str(e)
                })
                self.batch_status_text.append(f"FAILED: {os.path.basename(file_path)} - {str(e)}")
                # Include background info if available
                if 'bg_params_used' in locals():
                    self.batch_status_text.append(f"  Background was: {bg_params_used}")
        
        progress.setValue(len(self.spectra_files))
        progress.close()
        
        # Update visibility controls and plots
        self.update_peak_visibility_controls()
        self.update_trends_plot()
        self.update_fitting_quality_plot()
        
        successful = sum(1 for result in self.batch_results if not result.get('fit_failed', True))
        self.batch_status_text.append(f"\nBatch processing complete: {successful}/{len(self.spectra_files)} successful")
        
        # Enable custom material configuration if we have successful results
        if successful > 0 and hasattr(self, 'config_custom_material_btn'):
            self.config_custom_material_btn.setEnabled(True)
            self.batch_status_text.append("✓ Custom material configuration is now available!")
        
        # Enable specialized analysis buttons now that batch results are available
        if successful > 0:
            # Enable density analysis button in specialized tab if available
            if hasattr(self, 'batch_density_btn'):
                self.batch_density_btn.setEnabled(True)
            
            # Enable geothermometry analysis button in specialized tab if available
            if hasattr(self, 'batch_geothermo_btn'):
                self.batch_geothermo_btn.setEnabled(True)
        
        QMessageBox.information(self, "Batch Complete", 
                              f"Processed {len(self.spectra_files)} spectra.\n"
                              f"Successful: {successful}\n"
                              f"Failed: {len(self.spectra_files) - successful}\n\n"
                              f"You can now run specialized analyses from the Specialized tab.")
        
    def stop_batch(self):
        """Stop batch processing."""
        self._stop_batch = True
        self.batch_status_text.append("Batch processing stopped by user.")
    
    def configure_custom_material_from_results(self):
        """Configure custom material for density analysis from batch fitting results."""
        if not hasattr(self, 'batch_results') or not self.batch_results:
            QMessageBox.warning(self, "No Results", "No batch fitting results available.\nRun batch processing first.")
            return
        
        try:
            # Analyze batch results to extract material parameters
            analysis_results = self._analyze_batch_for_material_config()
            
            if not analysis_results:
                QMessageBox.warning(self, "Insufficient Data", 
                    "Could not extract sufficient peak data from batch results.\n"
                    "Ensure your batch fitting has successfully fitted peaks.")
                return
            
            # Create material configuration dialog pre-populated with batch data
            from Density.density_gui_launcher import CustomMaterialDialog
            
            dialog = CustomMaterialDialog(self)
            dialog.setWindowTitle("Custom Material from Batch Results")
            
            # Pre-populate with analyzed data
            self._populate_dialog_from_batch(dialog, analysis_results)
            
            # Show preview information
            preview_text = self._create_batch_analysis_preview(analysis_results)
            QMessageBox.information(self, "Batch Analysis Results", preview_text)
            
            if dialog.exec() == dialog.Accepted:
                custom_config = dialog.get_config()
                
                # Launch density analysis with the custom configuration
                try:
                    from Density.density_gui_launcher import DensityAnalysisGUI
                    
                    # Create density analysis window
                    density_gui = DensityAnalysisGUI()
                    
                    # Set the custom configuration
                    if custom_config:
                        from Density.raman_density_analysis import MaterialConfigs
                        material_name = custom_config['name']
                        MaterialConfigs.add_custom_material(material_name, custom_config)
                        
                        # Update the material selection
                        density_gui.material_combo.clear()
                        density_gui.material_combo.addItems(MaterialConfigs.get_available_materials())
                        density_gui.material_combo.setCurrentText(material_name)
                        density_gui.on_material_changed()
                        
                        # Pass batch results for reference intensity calculation
                        density_gui.set_batch_fitting_results(self.batch_results)
                    
                    # Show the density analysis GUI
                    density_gui.show()
                    
                    QMessageBox.information(self, "Success", 
                        f"Custom material '{custom_config['name']}' created and density analysis launched!\n\n"
                        f"Your batch fitting results are now available for reference intensity calculation.")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Launch Error", 
                        f"Failed to launch density analysis:\n{str(e)}")
                
        except Exception as e:
            QMessageBox.critical(self, "Configuration Error", 
                f"Failed to configure custom material from batch results:\n{str(e)}")
    
    def _analyze_batch_for_material_config(self):
        """Analyze batch results to extract material configuration parameters."""
        import numpy as np
        
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if not successful_results:
            return None
        
        # Extract peak information
        all_peaks = []
        peak_intensities = []
        
        for result in successful_results:
            if 'peaks' in result and result['peaks']:
                for peak in result['peaks']:
                    all_peaks.append(peak['center'])
                    peak_intensities.append(peak['amplitude'])
        
        if not all_peaks:
            return None
        
        # Find the most common peaks (characteristic peaks)
        peak_array = np.array(all_peaks)
        intensity_array = np.array(peak_intensities)
        
        # Group peaks by position (within 10 cm-1 tolerance)
        peak_groups = []
        tolerance = 10
        
        for peak_pos in peak_array:
            # Find if this peak belongs to an existing group
            group_found = False
            for group in peak_groups:
                if abs(peak_pos - group['center']) <= tolerance:
                    group['positions'].append(peak_pos)
                    group['intensities'].append(intensity_array[len(group['positions'])-1])
                    group_found = True
                    break
            
            if not group_found:
                # Create new group
                idx = np.where(peak_array == peak_pos)[0][0]
                peak_groups.append({
                    'center': peak_pos,
                    'positions': [peak_pos],
                    'intensities': [intensity_array[idx]]
                })
        
        # Sort groups by average intensity (strongest peaks first)
        for group in peak_groups:
            group['avg_intensity'] = np.mean(group['intensities'])
            group['avg_position'] = np.mean(group['positions'])
        
        peak_groups.sort(key=lambda x: x['avg_intensity'], reverse=True)
        
        # Extract top 3 peaks as characteristic peaks
        main_peak = peak_groups[0]['avg_position'] if len(peak_groups) > 0 else 1000
        secondary_peak = peak_groups[1]['avg_position'] if len(peak_groups) > 1 else 500
        tertiary_peak = peak_groups[2]['avg_position'] if len(peak_groups) > 2 else 1500
        
        # Calculate reference intensity (median of main peak intensities)
        main_peak_intensities = peak_groups[0]['intensities'] if len(peak_groups) > 0 else [800]
        reference_intensity = int(np.median(main_peak_intensities))
        
        # Determine spectral range for reference regions
        min_wavenumber = min(all_peaks) if all_peaks else 200
        max_wavenumber = max(all_peaks) if all_peaks else 1800
        
        return {
            'main_peak': main_peak,
            'secondary_peak': secondary_peak,
            'tertiary_peak': tertiary_peak,
            'reference_intensity': reference_intensity,
            'spectral_range': (min_wavenumber, max_wavenumber),
            'peak_groups': peak_groups,
            'total_peaks_analyzed': len(all_peaks),
            'total_spectra': len(successful_results)
        }
    
    def _populate_dialog_from_batch(self, dialog, analysis_results):
        """Populate the custom material dialog with batch analysis results."""
        # Set peak positions
        dialog.main_peak_spin.setValue(analysis_results['main_peak'])
        dialog.secondary_peak_spin.setValue(analysis_results['secondary_peak'])
        dialog.tertiary_peak_spin.setValue(analysis_results['tertiary_peak'])
        
        # Set reference intensity
        dialog.reference_intensity.setValue(analysis_results['reference_intensity'])
        
        # Set reasonable reference regions based on spectral range
        min_wn, max_wn = analysis_results['spectral_range']
        
        # Baseline region (below main peaks)
        baseline_start = max(100, min_wn - 50)
        baseline_end = min(analysis_results['main_peak'] - 50, baseline_start + 200)
        dialog.baseline_start.setValue(baseline_start)
        dialog.baseline_end.setValue(baseline_end)
        
        # Fingerprint region (around main peaks)
        fingerprint_start = max(analysis_results['main_peak'] - 100, baseline_end + 10)
        fingerprint_end = min(analysis_results['main_peak'] + 200, max_wn)
        dialog.fingerprint_start.setValue(fingerprint_start)
        dialog.fingerprint_end.setValue(fingerprint_end)
        
        # High frequency region (above main peaks)
        high_freq_start = max(fingerprint_end + 10, analysis_results['main_peak'] + 100)
        high_freq_end = min(max_wn + 100, 4000)
        dialog.high_freq_start.setValue(high_freq_start)
        dialog.high_freq_end.setValue(high_freq_end)
        
        # Set material name based on main peak
        main_peak_int = int(analysis_results['main_peak'])
        dialog.material_name_edit.setPlainText(f"Custom Material ({main_peak_int} cm⁻¹)")
    
    def _create_batch_analysis_preview(self, analysis_results):
        """Create preview text showing the batch analysis results."""
        preview = f"""📊 Batch Analysis Results

📈 Peak Analysis:
• Total spectra analyzed: {analysis_results['total_spectra']}
• Total peaks found: {analysis_results['total_peaks_analyzed']}
• Peak groups identified: {len(analysis_results['peak_groups'])}

🎯 Characteristic Peaks:
• Main Peak: {analysis_results['main_peak']:.1f} cm⁻¹
• Secondary Peak: {analysis_results['secondary_peak']:.1f} cm⁻¹  
• Tertiary Peak: {analysis_results['tertiary_peak']:.1f} cm⁻¹

📊 Reference Intensity:
• Calculated: {analysis_results['reference_intensity']} (median from main peak)

📏 Spectral Range:
• {analysis_results['spectral_range'][0]:.0f} - {analysis_results['spectral_range'][1]:.0f} cm⁻¹

✅ These parameters will be used to pre-populate your custom material configuration.
You can adjust any values in the next dialog before saving."""
        
        return preview
    

    
    def run_batch_density_analysis(self):
        """Run density analysis on all loaded spectra."""
        if not DENSITY_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "Density analysis module is not available.")
            return
        
        if len(self.spectra_files) == 0:
            QMessageBox.warning(self, "No Data", "No spectra loaded for analysis.")
            return
        
        try:
            from Density.raman_density_analysis import RamanDensityAnalyzer
            
            # Use default analysis parameters for streamlined batch processing
            material_type = 'Kidney Stones (COM)'
            density_type = 'mixed'
            
            # Create analyzer
            analyzer = RamanDensityAnalyzer(material_type)
            
            # Progress dialog
            progress = QProgressDialog("Running density analysis...", "Cancel", 0, len(self.spectra_files), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Initialize density results storage
            if not hasattr(self, 'density_results'):
                self.density_results = []
            self.density_results.clear()
            
            self.batch_status_text.append(f"\nStarting batch density analysis ({material_type})...")
            
            success_count = 0
            
            for i, file_path in enumerate(self.spectra_files):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(f"Analyzing: {os.path.basename(file_path)}")
                QApplication.processEvents()
                
                try:
                    # Load spectrum
                    wavenumbers, intensities = self.load_spectrum_robust(file_path)
                    
                    if wavenumbers is None or intensities is None:
                        raise ValueError("Failed to load spectrum data")
                    
                    # Preprocess spectrum with validation
                    try:
                        wn, corrected_int = analyzer.preprocess_spectrum(wavenumbers, intensities)
                        
                        # Validate preprocessed data
                        if len(wn) == 0 or len(corrected_int) == 0:
                            raise ValueError("Empty data after preprocessing")
                        if np.all(np.isnan(corrected_int)) or np.all(np.isinf(corrected_int)):
                            raise ValueError("Invalid intensity data after preprocessing")
                            
                    except Exception as e:
                        raise ValueError(f"Preprocessing failed: {str(e)}")
                    
                    # Calculate density metrics with validation
                    try:
                        cdi, metrics = analyzer.calculate_crystalline_density_index(wn, corrected_int)
                        
                        # Validate CDI calculation
                        if np.isnan(cdi) or np.isinf(cdi):
                            raise ValueError("CDI calculation returned invalid value")
                        if cdi < 0:
                            cdi = 0.0  # Ensure non-negative CDI
                            
                        apparent_density = analyzer.calculate_apparent_density(cdi)
                        
                        # Validate density calculation
                        if np.isnan(apparent_density) or np.isinf(apparent_density):
                            raise ValueError("Density calculation returned invalid value")
                            
                    except Exception as e:
                        raise ValueError(f"Density analysis failed: {str(e)}")
                    
                    # Use appropriate density calculation
                    if material_type == 'Kidney Stones (COM)':
                        specialized_density = analyzer.calculate_biofilm_density(cdi, density_type)
                    else:
                        specialized_density = analyzer.calculate_density_by_type(cdi, density_type)
                    
                    # Get classification
                    thresholds = analyzer.classification_thresholds
                    if cdi < thresholds['low']:
                        classification = 'Low crystallinity'
                    elif cdi < thresholds['medium']:
                        classification = 'Mixed regions'
                    elif cdi < thresholds['high']:
                        classification = 'Mixed crystalline'
                    else:
                        classification = 'Pure crystalline'
                    
                    # Store results
                    density_result = {
                        'file': file_path,
                        'material_type': material_type,
                        'density_type': density_type,
                        'cdi': cdi,
                        'apparent_density': apparent_density,
                        'specialized_density': specialized_density,
                        'classification': classification,
                        'metrics': metrics,
                        'wavenumbers': wn,
                        'corrected_intensity': corrected_int,
                        'success': True
                    }
                    
                    self.density_results.append(density_result)
                    success_count += 1
                    
                    # Add debug information for density calculation
                    debug_info = f"CDI={cdi:.4f}, ρ={specialized_density:.3f} g/cm³"
                    if 'metrics' in locals():
                        debug_info += f", peak_height={metrics.get('main_peak_height', 'N/A'):.1f}"
                        debug_info += f", baseline={metrics.get('baseline_intensity', 'N/A'):.1f}"
                    
                    self.batch_status_text.append(f"DENSITY SUCCESS: {os.path.basename(file_path)} - {debug_info}")
                    
                except Exception as e:
                    # Store failed result
                    density_result = {
                        'file': file_path,
                        'material_type': material_type,
                        'density_type': density_type,
                        'success': False,
                        'error': str(e)
                    }
                    
                    self.density_results.append(density_result)
                    self.batch_status_text.append(f"DENSITY FAILED: {os.path.basename(file_path)} - {str(e)}")
            
            progress.setValue(len(self.spectra_files))
            progress.close()
            
            # Update plots if we have density results
            if success_count > 0:
                self.update_trends_plot()  # This will now include density data
                self.update_heatmap_plot()  # This will now include density data
                
                # Generate comprehensive density analysis visualization
                self.show_comprehensive_density_plots()
            
            self.batch_status_text.append(f"\nDensity analysis complete: {success_count}/{len(self.spectra_files)} successful")
            
            QMessageBox.information(self, "Density Analysis Complete", 
                                  f"Analyzed {len(self.spectra_files)} spectra for density.\n"
                                  f"Successful: {success_count}\n"
                                  f"Failed: {len(self.spectra_files) - success_count}\n\n"
                                  f"Material: {material_type}\n"
                                  f"Density Type: {density_type}")
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to run batch density analysis:\n{str(e)}")
    
    def show_comprehensive_density_plots(self):
        """Generate comprehensive density analysis plots using the RamanDensityAnalyzer visualization."""
        if not hasattr(self, 'density_results') or not self.density_results:
            return
            
        try:
            from Density.raman_density_analysis import RamanDensityAnalyzer
            import sys
            
            # Get successful density results with valid data
            successful_results = []
            for r in self.density_results:
                if (r.get('success', False) and 
                    not np.isnan(r.get('cdi', np.nan)) and 
                    not np.isnan(r.get('specialized_density', np.nan))):
                    successful_results.append(r)
            
            if not successful_results:
                QMessageBox.warning(self, "No Valid Data", 
                                  "No density analysis results with valid data to plot.\n"
                                  "This may indicate issues with the spectra or analysis parameters.")
                return
            
            # Use the first successful result to get the material type and analyzer
            first_result = successful_results[0]
            material_type = first_result['material_type']
            analyzer = RamanDensityAnalyzer(material_type)
            
            # Prepare data for comprehensive analysis
            positions = list(range(len(successful_results)))  # Use spectrum indices as positions
            
            # Extract density profile data
            cdi_profile = [r['cdi'] for r in successful_results]
            density_profile = [r['specialized_density'] for r in successful_results]
            apparent_density_profile = [r['apparent_density'] for r in successful_results]
            
            # Get classifications
            classifications = [r['classification'] for r in successful_results]
            
            # Calculate statistics
            mean_density = np.mean(density_profile)
            std_density = np.std(density_profile)
            mean_cdi = np.mean(cdi_profile)
            std_cdi = np.std(cdi_profile)
            
            # Create density profile data structure for plotting
            density_analysis_data = {
                'positions': positions,
                'cdi_profile': cdi_profile,
                'density_profile': density_profile,
                'apparent_density_profile': apparent_density_profile,
                'layer_classification': classifications,
                'statistics': {
                    'mean_density': mean_density,
                    'std_density': std_density,
                    'mean_cdi': mean_cdi,
                    'std_cdi': std_cdi,
                    'n_spectra': len(successful_results)
                },
                'file_names': [os.path.basename(r['file']) for r in successful_results]
            }
            
            # Create a comprehensive density visualization
            self.create_density_visualization_window(density_analysis_data, analyzer, material_type)
            
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", 
                               f"Failed to create density visualization:\n{str(e)}")
    
    def create_density_visualization_window(self, density_data, analyzer, material_type):
        """Create a separate window for comprehensive density analysis visualization."""
        from matplotlib.backends.qt_compat import QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure
        
        # Create a new window
        density_window = QDialog(self)
        density_window.setWindowTitle(f"Comprehensive Density Analysis - {material_type}")
        density_window.setMinimumSize(1200, 900)
        density_window.resize(1400, 1000)
        
        layout = QVBoxLayout(density_window)
        
        # Add summary information at the top
        info_group = QGroupBox("Analysis Summary")
        info_layout = QVBoxLayout(info_group)
        
        stats = density_data['statistics']
        summary_text = f"""
Material Type: {material_type}
Number of Spectra: {stats['n_spectra']}
Mean CDI: {stats['mean_cdi']:.4f} ± {stats['std_cdi']:.4f}
Mean Density: {stats['mean_density']:.3f} ± {stats['std_density']:.3f} g/cm³
"""
        info_label = QLabel(summary_text)
        info_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 10px;")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Create matplotlib figure for comprehensive plots
        figure = Figure(figsize=(14, 10))
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, density_window)
        
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        
        # Create the comprehensive density analysis plots
        self._create_density_analysis_plots(figure, density_data, analyzer, material_type)
        
        # Add export button
        export_btn = QPushButton("Export Density Plots")
        export_btn.clicked.connect(lambda: self.export_density_plots(figure, material_type))
        layout.addWidget(export_btn)
        
        # Show the window
        density_window.exec_()
    
    def _create_density_analysis_plots(self, figure, density_data, analyzer, material_type):
        """Create the 4-panel density analysis visualization similar to the reference paper."""
        try:
            # Import matplotlib configuration if available
            try:
                import sys
                sys.path.append(os.path.join(os.path.dirname(__file__), 'polarization_ui'))
                from matplotlib_config import setup_matplotlib_style
                setup_matplotlib_style()
            except ImportError:
                pass  # Continue without custom styling
        except:
            pass
        
        figure.clear()
        
        # Create 2x2 subplot layout
        ax1 = figure.add_subplot(2, 2, 1)  # CDI Across All Spectra
        ax2 = figure.add_subplot(2, 2, 2)  # Density Across All Spectra  
        ax3 = figure.add_subplot(2, 2, 3)  # Density Distribution
        ax4 = figure.add_subplot(2, 2, 4)  # CDI vs Density Correlation
        
        positions = density_data['positions']
        cdi_profile = density_data['cdi_profile']
        density_profile = density_data['density_profile']
        classifications = density_data['layer_classification']
        stats = density_data['statistics']
        
        # 1. CDI Across All Spectra (similar to top-right of reference figure)
        ax1.plot(positions, cdi_profile, 'g-', linewidth=2, marker='o', markersize=4)
        
        # Add classification threshold line
        threshold = analyzer.classification_thresholds.get('medium', 0.5)
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label='Classification threshold')
        
        ax1.set_xlabel('Spectrum Number')
        ax1.set_ylabel('Crystalline Density Index')
        ax1.set_title('CDI Across All Spectra')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Density Across All Spectra (similar to top-left of reference figure)  
        ax2.plot(positions, density_profile, 'b-', linewidth=2, marker='o', markersize=4)
        
        # Add mean line
        mean_density = stats['mean_density']
        ax2.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7,
                   label=f'Mean: {mean_density:.3f}')
        
        ax2.set_xlabel('Spectrum Number')
        ax2.set_ylabel('Density (g/cm³)')
        ax2.set_title('Density Across All Spectra')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Density Distribution (similar to bottom-left of reference figure)
        ax3.hist(density_profile, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=mean_density, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_density:.3f}')
        ax3.set_xlabel('Density (g/cm³)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Density Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. CDI vs Density Correlation (similar to bottom-right of reference figure)
        # Color-code by classification
        classification_colors = {
            'Low crystallinity': 'lightblue',
            'Mixed regions': 'orange', 
            'Mixed crystalline': 'purple',
            'Pure crystalline': 'darkblue',
            'bacterial': 'cyan',
            'organic': 'red',
            'crystalline': 'blue'
        }
        
        # Plot points colored by classification
        for classification in set(classifications):
            mask = [c == classification for c in classifications]
            cdi_subset = [cdi_profile[i] for i in range(len(cdi_profile)) if mask[i]]
            density_subset = [density_profile[i] for i in range(len(density_profile)) if mask[i]]
            
            color = classification_colors.get(classification, 'gray')
            ax4.scatter(cdi_subset, density_subset, 
                       c=color, label=classification, alpha=0.7, s=50)
        
        ax4.set_xlabel('Crystalline Density Index')
        ax4.set_ylabel('Density (g/cm³)')
        ax4.set_title('CDI vs Density Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add vertical line at classification threshold
        ax4.axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)
        
        figure.suptitle(f'{material_type} - Comprehensive Density Analysis\n'
                       f'N={stats["n_spectra"]} spectra', 
                       fontsize=14, fontweight='bold')
        
        figure.tight_layout()
        
        # Redraw the canvas
        try:
            figure.canvas.draw()
        except:
            pass  # In case canvas is not available yet
    
    def export_density_plots(self, figure, material_type):
        """Export the density analysis plots to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Density Plots", 
            f"density_analysis_{material_type.replace(' ', '_').replace('(', '').replace(')', '')}.png",
            "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
        )
        
        if file_path:
            try:
                figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export Successful", 
                                      f"Density analysis plots saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", 
                                   f"Failed to export plots:\n{str(e)}")
    
    # Geothermometry analysis methods
    
    def run_batch_geothermometry_analysis(self, auto_mode=False):
        """Run geothermometry analysis on all loaded spectra.
        
        Args:
            auto_mode (bool): If True, runs automatically after batch processing without showing dialogs
        """
        if not GEOTHERMOMETRY_AVAILABLE:
            if not auto_mode:
                QMessageBox.warning(self, "Not Available", "Geothermometry analysis module is not available.")
            return
        
        if len(self.spectra_files) == 0:
            if not auto_mode:
                QMessageBox.warning(self, "No Data", "No spectra loaded for analysis.")
            return
        
        try:
            # Use sensible defaults for batch processing (detailed config in specialized tab)
            method_enum = GeothermometerMethod.BEYSSAC_2002  # Most commonly used method
            method_name = method_enum.value
            output_option = "Temperature + Parameters"  # Show results with parameters
            
            geothermo_calc = RamanGeothermometry()
            
            # Progress dialog
            progress = QProgressDialog("Running geothermometry analysis...", "Cancel", 0, len(self.spectra_files), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Initialize geothermometry results storage
            if not hasattr(self, 'geothermometry_results'):
                self.geothermometry_results = []
            self.geothermometry_results.clear()
            
            self.batch_status_text.append(f"\nStarting batch geothermometry analysis ({method_name})...")
            
            success_count = 0
            
            for i, file_path in enumerate(self.spectra_files):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(f"Analyzing: {os.path.basename(file_path)}")
                QApplication.processEvents()
                
                try:
                    # Get peak fitting results for this file
                    fitting_result = None
                    for result in self.batch_results:
                        if result.get('file') == file_path and not result.get('fit_failed', True):
                            fitting_result = result
                            break
                    
                    if fitting_result is None:
                        raise ValueError("No valid peak fitting results found for this spectrum")
                    
                    # Extract required parameters from peak fitting results
                    params = self._extract_geothermometry_parameters(fitting_result, method_enum)
                    
                    # Calculate temperature
                    temperature, status = geothermo_calc.calculate_temperature(method_enum, **params)
                    
                    if temperature is None:
                        raise ValueError(f"Temperature calculation failed: {status}")
                    
                    # Store results
                    result_data = {
                        'file': file_path,
                        'method': method_name,
                        'temperature': temperature,
                        'status': status,
                        'parameters': params,
                        'output_option': output_option
                    }
                    
                    self.geothermometry_results.append(result_data)
                    
                    self.batch_status_text.append(f"SUCCESS: {os.path.basename(file_path)} - T = {temperature:.1f}°C ({status})")
                    success_count += 1
                    
                except Exception as e:
                    error_msg = str(e)
                    self.geothermometry_results.append({
                        'file': file_path,
                        'method': method_name,
                        'failed': True,
                        'error': error_msg
                    })
                    self.batch_status_text.append(f"FAILED: {os.path.basename(file_path)} - {error_msg}")
            
            progress.setValue(len(self.spectra_files))
            progress.close()
            
            self.batch_status_text.append(f"\nGeothermometry analysis complete: {success_count}/{len(self.spectra_files)} successful")
            
            # Update trends plot with latest data
            if hasattr(self, 'update_trends_plot'):
                self.update_trends_plot()
                
            if not auto_mode:
                # Show results based on output option
                if output_option == "Export to CSV":
                    self._export_geothermometry_results()
                else:
                    self._display_geothermometry_results(output_option)
                
                QMessageBox.information(self, "Geothermometry Complete", 
                                      f"Processed {len(self.spectra_files)} spectra.\n"
                                      f"Successful: {success_count}\n"
                                      f"Failed: {len(self.spectra_files) - success_count}")
            else:
                # Auto mode: just report success to status text and update summary display
                self.batch_status_text.append(f"✅ Geothermometry analysis completed automatically!")
                if success_count > 0:
                    successful_temps = [r['temperature'] for r in self.geothermometry_results if not r.get('failed', False)]
                    if successful_temps:
                        avg_temp = sum(successful_temps) / len(successful_temps)
                        min_temp = min(successful_temps)
                        max_temp = max(successful_temps)
                        self.batch_status_text.append(f"   📊 Temperature range: {min_temp:.1f}°C - {max_temp:.1f}°C (avg: {avg_temp:.1f}°C)")
            
        except Exception as e:
            error_msg = f"Failed to run geothermometry analysis:\n{str(e)}"
            if not auto_mode:
                QMessageBox.critical(self, "Analysis Error", error_msg)
            else:
                self.batch_status_text.append(f"❌ Geothermometry analysis error: {str(e)}")
    
    def _extract_geothermometry_parameters(self, fitting_result, method_enum):
        """Extract required parameters for geothermometry from peak fitting results."""
        required_params = {
            GeothermometerMethod.BEYSSAC_2002: ['R2'],
            GeothermometerMethod.AOYA_2010_514: ['R2'], 
            GeothermometerMethod.AOYA_2010_532: ['R2'],
            GeothermometerMethod.RAHL_2005: ['R1', 'R2'],
            GeothermometerMethod.KOUKETSU_2014_D1: ['D1_FWHM'],
            GeothermometerMethod.KOUKETSU_2014_D2: ['D2_FWHM'],
            GeothermometerMethod.RANTITSCH_2004: ['R2']
        }
        
        needed_params = required_params.get(method_enum, [])
        
        # Extract peak information from fitting results
        fit_params = fitting_result.get('fit_params', [])
        wavenumbers = fitting_result.get('wavenumbers', [])
        
        if len(fit_params) == 0:
            raise ValueError("No peak fitting parameters available")
        
        # Find D and G bands (approximately 1350 and 1580 cm-1)
        # This is a simplified approach - in practice, you might need more sophisticated peak identification
        n_peaks = len(fit_params) // 3
        peaks_info = []
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(fit_params):
                amp, cen, wid = fit_params[start_idx:start_idx+3]
                peaks_info.append({'position': cen, 'amplitude': amp, 'width': wid})
        
        if len(peaks_info) < 2:
            raise ValueError("At least 2 peaks required for geothermometry analysis")
        
        # Sort peaks by position
        peaks_info.sort(key=lambda x: x['position'])
        
        # Find D band (around 1350 cm-1) and G band (around 1580 cm-1)
        d_band = None
        g_band = None
        
        for peak in peaks_info:
            pos = peak['position']
            if 1300 <= pos <= 1400 and d_band is None:
                d_band = peak
            elif 1500 <= pos <= 1650 and g_band is None:
                g_band = peak
        
        if d_band is None or g_band is None:
            raise ValueError("Could not identify D and G bands in spectrum")
        
        # Calculate parameters
        params = {}
        
        if 'R2' in needed_params:
            # R2 = Area_D / Area_G (approximated as amplitude ratio for Gaussian peaks)
            params['R2'] = d_band['amplitude'] / g_band['amplitude']
        
        if 'R1' in needed_params:
            # R1 = Height_D / Height_G  
            params['R1'] = d_band['amplitude'] / g_band['amplitude']
        
        if 'D1_FWHM' in needed_params:
            # D1 FWHM (assuming D1 is the D band)
            params['D1_FWHM'] = d_band['width'] * 2 * np.sqrt(2 * np.log(2))  # Convert from Gaussian sigma to FWHM
        
        if 'D2_FWHM' in needed_params:
            # D2 FWHM - this would require more sophisticated peak deconvolution
            # For now, use an estimate
            params['D2_FWHM'] = d_band['width'] * 2 * np.sqrt(2 * np.log(2)) * 0.8
        
        return params
    
    def _display_geothermometry_results(self, output_option):
        """Display geothermometry results in a dialog."""
        if not self.geothermometry_results:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Geothermometry Analysis Results")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Results text
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setFont(QFont("Consolas", 10))
        
        results_text = "GEOTHERMOMETRY ANALYSIS RESULTS\n"
        results_text += "=" * 50 + "\n\n"
        
        successful_results = [r for r in self.geothermometry_results if not r.get('failed', False)]
        failed_results = [r for r in self.geothermometry_results if r.get('failed', False)]
        
        for result in successful_results:
            filename = os.path.basename(result['file'])
            temperature = result['temperature']
            status = result['status']
            method = result['method']
            
            results_text += f"File: {filename}\n"
            results_text += f"Method: {method}\n"
            results_text += f"Temperature: {temperature:.1f}°C\n"
            results_text += f"Status: {status}\n"
            
            if output_option in ["Temperature + Parameters", "Full Analysis Report"]:
                params = result.get('parameters', {})
                results_text += "Parameters:\n"
                for param, value in params.items():
                    results_text += f"  {param}: {value:.4f}\n"
            
            if output_option == "Full Analysis Report":
                # Add method information
                geothermo_calc = RamanGeothermometry()
                for method_enum in GeothermometerMethod:
                    if method_enum.value == method:
                        method_info = geothermo_calc.get_method_info(method_enum)
                        results_text += f"Method Details:\n"
                        results_text += f"  Range: {method_info.temp_range}\n"
                        results_text += f"  Error: {method_info.error}\n"
                        results_text += f"  Description: {method_info.description}\n"
                        break
            
            results_text += "\n" + "-" * 40 + "\n\n"
        
        if failed_results:
            results_text += "\nFAILED ANALYSES:\n"
            results_text += "=" * 20 + "\n"
            for result in failed_results:
                filename = os.path.basename(result['file'])
                error = result.get('error', 'Unknown error')
                results_text += f"File: {filename}\n"
                results_text += f"Error: {error}\n\n"
        
        text_widget.setText(results_text)
        layout.addWidget(text_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(lambda: self._export_geothermometry_results())
        button_layout.addWidget(export_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _export_geothermometry_results(self):
        """Export geothermometry results to CSV."""
        if not self.geothermometry_results:
            QMessageBox.warning(self, "No Data", "No geothermometry results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Geothermometry Results", 
            "geothermometry_results.csv",
            "CSV files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Determine all possible columns
                fieldnames = ['Filename', 'Method', 'Temperature_C', 'Status', 'Failed', 'Error']
                
                # Add parameter columns
                param_columns = set()
                for result in self.geothermometry_results:
                    if 'parameters' in result:
                        param_columns.update(result['parameters'].keys())
                
                fieldnames.extend(sorted(param_columns))
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.geothermometry_results:
                    row = {
                        'Filename': os.path.basename(result['file']),
                        'Method': result['method'],
                        'Failed': result.get('failed', False),
                        'Error': result.get('error', '')
                    }
                    
                    if not result.get('failed', False):
                        row['Temperature_C'] = result['temperature']
                        row['Status'] = result['status']
                        
                        # Add parameters
                        params = result.get('parameters', {})
                        for param in param_columns:
                            row[param] = params.get(param, '')
                    
                    writer.writerow(row)
            
            QMessageBox.information(self, "Export Successful", 
                                  f"Geothermometry results exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Failed to export geothermometry results:\n{str(e)}")
    
    def launch_geothermometry_analysis(self):
        """Launch a detailed geothermometry analysis window."""
        try:
            dialog = GeothermometryAnalysisDialog(self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", 
                               f"Failed to launch geothermometry analysis:\n{str(e)}")
    
    def quick_geothermometry_analysis(self):
        """Perform quick geothermometry analysis on the current spectrum."""
        if not GEOTHERMOMETRY_AVAILABLE:
            QMessageBox.warning(self, "Not Available", 
                              "Geothermometry analysis module is not available.")
            return
            
        if len(self.wavenumbers) == 0 or len(self.intensities) == 0:
            QMessageBox.warning(self, "No Data", 
                              "No spectrum data loaded for analysis.")
            return
        
        if not self.fit_params or len(self.fit_params) == 0:
            QMessageBox.warning(self, "No Peak Fitting", 
                              "Please perform peak fitting first to enable geothermometry analysis.")
            return
        
        try:
            # Ask user for method
            geothermo_calc = RamanGeothermometry()
            methods = geothermo_calc.get_all_methods()
            method_name, ok = QInputDialog.getItem(
                self, "Select Geothermometry Method", 
                "Choose the geothermometry method:",
                methods, 0, False
            )
            
            if not ok:
                return
            
            # Find the corresponding GeothermometerMethod enum
            method_enum = None
            for method in GeothermometerMethod:
                if method.value == method_name:
                    method_enum = method
                    break
            
            if method_enum is None:
                QMessageBox.warning(self, "Invalid Method", f"Could not find method: {method_name}")
                return
            
            # Create a mock fitting result from current data
            fitting_result = {
                'fit_params': self.fit_params,
                'wavenumbers': self.wavenumbers
            }
            
            # Extract parameters
            params = self._extract_geothermometry_parameters(fitting_result, method_enum)
            
            # Calculate temperature
            temperature, status = geothermo_calc.calculate_temperature(method_enum, **params)
            
            if temperature is None:
                QMessageBox.warning(self, "Calculation Failed", f"Temperature calculation failed: {status}")
                return
            
            # Get method information
            method_info = geothermo_calc.get_method_info(method_enum)
            
            # Display results
            result_text = f"""
Geothermometry Analysis Results:

Method: {method_name}
Temperature: {temperature:.1f}°C
Status: {status}

Method Details:
• Description: {method_info.description}
• Temperature Range: {method_info.temp_range}
• Error: {method_info.error}
• Best for: {method_info.best_for}

Parameters Used:
"""
            for param, value in params.items():
                result_text += f"• {param}: {value:.4f}\n"
            
            result_text += f"""

Limitations:
{method_info.limitations}

Note: This analysis is based on current peak fitting results.
For more accurate results, ensure proper peak identification
of D and G bands in carbonaceous material.
            """
            
            QMessageBox.information(self, "Geothermometry Results", result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", 
                               f"Failed to perform geothermometry analysis:\n{str(e)}")
    
    def update_peak_info_display(self):
        """Update the peak information display in the left panel."""
        if not hasattr(self, 'peak_info_text'):
            return
            
        self.peak_info_text.clear()
        
        if (self.fit_result and 
            self.fit_params is not None and 
            hasattr(self.fit_params, '__len__') and 
            len(self.fit_params) > 0 and
            len(self.peaks) > 0):
            
            # Calculate overall R²
            if self.residuals is not None:
                ss_res = np.sum(self.residuals ** 2)
                ss_tot = np.sum((self.intensities - np.mean(self.intensities)) ** 2)
                overall_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                overall_r2 = 0
            
            self.peak_info_text.append(f"📊 FITTED PEAKS SUMMARY")
            self.peak_info_text.append(f"Overall R²: {overall_r2:.4f}")
            self.peak_info_text.append(f"Model: {self.current_model}")
            self.peak_info_text.append(f"Total Peaks: {len(self.peaks)}")
            self.peak_info_text.append("")
            
            # Background information (if available)
            if self.background is not None and len(self.background) > 0:
                bg_min = np.min(self.background)
                bg_max = np.max(self.background)
                bg_mean = np.mean(self.background)
                self.peak_info_text.append(f"📊 BACKGROUND ANALYSIS")
                self.peak_info_text.append(f"Applied Range: {bg_min:.0f} - {bg_max:.0f}")
                self.peak_info_text.append(f"Average Background: {bg_mean:.0f}")
                self.peak_info_text.append("")
            
            # Determine parameters per peak based on model type
            if self.current_model in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                if self.current_model == "Pseudo-Voigt":
                    params_per_peak = 4  # amp, cen, wid, eta
                else:  # Asymmetric Voigt
                    params_per_peak = 5  # amp, cen, wid, eta, asym
            else:
                params_per_peak = 3  # amp, cen, wid for Gaussian and Lorentzian
            
            # Display individual peak parameters
            self.peak_info_text.append(f"🎯 FITTED PEAKS ({len(self.peaks)} total)")
            self.peak_info_text.append("")
            
            n_peaks = len(self.peaks)
            for i in range(n_peaks):
                start_idx = i * params_per_peak
                if start_idx + (params_per_peak - 1) < len(self.fit_params):
                    if self.current_model in ["Gaussian", "Lorentzian"]:
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        self.peak_info_text.append(f"Peak {i+1}: {cen:.1f}±{wid:.1f} cm⁻¹ (A:{amp:.0f})")
                        self.peak_info_text.append(f"  Position: {cen:.1f} cm⁻¹")
                        self.peak_info_text.append(f"  Amplitude: {amp:.0f}")
                        self.peak_info_text.append(f"  Width: {wid:.2f}")
                    elif self.current_model == "Pseudo-Voigt":
                        amp, cen, wid, eta = self.fit_params[start_idx:start_idx+4]
                        self.peak_info_text.append(f"Peak {i+1}: {cen:.1f}±{wid:.1f} cm⁻¹ (A:{amp:.0f})")
                        self.peak_info_text.append(f"  Position: {cen:.1f} cm⁻¹")
                        self.peak_info_text.append(f"  Amplitude: {amp:.0f}")
                        self.peak_info_text.append(f"  Width: {wid:.2f}")
                        self.peak_info_text.append(f"  Mixing (η): {eta:.3f}")
                    elif self.current_model == "Asymmetric Voigt":
                        amp, cen, wid, eta, asym = self.fit_params[start_idx:start_idx+5]
                        self.peak_info_text.append(f"Peak {i+1}: {cen:.1f}±{wid:.1f} cm⁻¹ (A:{amp:.0f})")
                        self.peak_info_text.append(f"  Position: {cen:.1f} cm⁻¹")
                        self.peak_info_text.append(f"  Amplitude: {amp:.0f}")
                        self.peak_info_text.append(f"  Width: {wid:.2f}")
                        self.peak_info_text.append(f"  Mixing (η): {eta:.3f}")
                        self.peak_info_text.append(f"  Asymmetry: {asym:.3f}")
                    
                    self.peak_info_text.append("")
        
        elif len(self.peaks) > 0:
            self.peak_info_text.append(f"📍 DETECTED PEAKS")
            self.peak_info_text.append(f"Total Peaks: {len(self.peaks)}")
            self.peak_info_text.append("Status: Detected but not fitted")
            self.peak_info_text.append("")
            
            for i, peak_idx in enumerate(self.peaks):
                if 0 <= peak_idx < len(self.wavenumbers):
                    wavenumber = self.wavenumbers[peak_idx]
                    intensity = self.intensities[peak_idx] if peak_idx < len(self.intensities) else 0
                    self.peak_info_text.append(f"Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I={intensity:.0f})")
        
        else:
            self.peak_info_text.append("No peaks detected")
            self.peak_info_text.append("")
            self.peak_info_text.append("Use Peak Detection controls to find peaks,")
            self.peak_info_text.append("then click 'Fit Peaks' to get detailed parameters.")
    
    def update_peak_info_display_for_batch_result(self, result_data):
        """Update the peak information display for a batch result with comprehensive details."""
        if not hasattr(self, 'peak_info_text'):
            return
            
        self.peak_info_text.clear()
        
        if result_data.get('fit_failed', True):
            self.peak_info_text.append("❌ BATCH PROCESSING FAILED")
            self.peak_info_text.append(f"Error: {result_data.get('error', 'Unknown error')}")
            return
        
        # Display batch processing results with comprehensive details
        filename = os.path.basename(result_data['file'])
        r_squared = result_data.get('r_squared', 0)
        fit_params = result_data.get('fit_params', [])
        
        self.peak_info_text.append(f"🔄 BATCH PROCESSING RESULT")
        self.peak_info_text.append(f"File: {filename}")
        self.peak_info_text.append(f"Overall R²: {r_squared:.4f}")
        self.peak_info_text.append(f"Model: {self.current_model}")
        self.peak_info_text.append(f"Status: SUCCESS")
        self.peak_info_text.append("")
        
        # Background information
        background_data = result_data.get('background', None)
        if background_data is not None and len(background_data) > 0:
            bg_min = np.min(background_data)
            bg_max = np.max(background_data)
            bg_mean = np.mean(background_data)
            self.peak_info_text.append(f"📊 BACKGROUND ANALYSIS")
            self.peak_info_text.append(f"Optimized Range: {bg_min:.0f} - {bg_max:.0f}")
            self.peak_info_text.append(f"Average Background: {bg_mean:.0f}")
            self.peak_info_text.append("")
        
        if len(fit_params) > 0:
            # Determine parameters per peak based on model type
            if self.current_model in ["Pseudo-Voigt", "Asymmetric Voigt"]:
                if self.current_model == "Pseudo-Voigt":
                    params_per_peak = 4  # amp, cen, wid, eta
                else:  # Asymmetric Voigt
                    params_per_peak = 5  # amp, cen, wid, eta, asym
            else:
                params_per_peak = 3  # amp, cen, wid for Gaussian and Lorentzian
            
            n_peaks = len(fit_params) // params_per_peak
            self.peak_info_text.append(f"🎯 FITTED PEAKS ({n_peaks} total)")
            self.peak_info_text.append("")
            
            # Display individual peak parameters
            for i in range(n_peaks):
                start_idx = i * params_per_peak
                if start_idx + (params_per_peak - 1) < len(fit_params):
                    if self.current_model in ["Gaussian", "Lorentzian"]:
                        amp, cen, wid = fit_params[start_idx:start_idx+3]
                        self.peak_info_text.append(f"Peak {i+1}: {cen:.1f}±{wid:.1f} cm⁻¹ (A:{amp:.0f})")
                        self.peak_info_text.append(f"  Position: {cen:.1f} cm⁻¹")
                        self.peak_info_text.append(f"  Amplitude: {amp:.0f}")
                        self.peak_info_text.append(f"  Width: {wid:.2f}")
                    elif self.current_model == "Pseudo-Voigt":
                        amp, cen, wid, eta = fit_params[start_idx:start_idx+4]
                        self.peak_info_text.append(f"Peak {i+1}: {cen:.1f}±{wid:.1f} cm⁻¹ (A:{amp:.0f})")
                        self.peak_info_text.append(f"  Position: {cen:.1f} cm⁻¹")
                        self.peak_info_text.append(f"  Amplitude: {amp:.0f}")
                        self.peak_info_text.append(f"  Width: {wid:.2f}")
                        self.peak_info_text.append(f"  Mixing (η): {eta:.3f}")
                    elif self.current_model == "Asymmetric Voigt":
                        amp, cen, wid, eta, asym = fit_params[start_idx:start_idx+5]
                        self.peak_info_text.append(f"Peak {i+1}: {cen:.1f}±{wid:.1f} cm⁻¹ (A:{amp:.0f})")
                        self.peak_info_text.append(f"  Position: {cen:.1f} cm⁻¹")
                        self.peak_info_text.append(f"  Amplitude: {amp:.0f}")
                        self.peak_info_text.append(f"  Width: {wid:.2f}")
                        self.peak_info_text.append(f"  Mixing (η): {eta:.3f}")
                        self.peak_info_text.append(f"  Asymmetry: {asym:.3f}")
                    
                    self.peak_info_text.append("")
        else:
            self.peak_info_text.append("No peak parameters available")
    
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
        
        # Determine what to show based on current state
        # Priority: batch results > preview mode > normal mode
        showing_batch_result = (hasattr(self, 'batch_results') and 
                               len(self.batch_results) > 0 and 
                               hasattr(self, 'current_spectrum_index') and 
                               0 <= self.current_spectrum_index < len(self.batch_results))
        
        if showing_batch_result:
            # BATCH RESULT MODE: Show the optimized background from batch processing
            # Clear preview mode when showing batch results
            self.background_preview_active = False
            self.background_preview = None
            
            # Use batch result data
            result = self.batch_results[self.current_spectrum_index]
            original_intensities = result.get('original_intensities', self.original_intensities)
            corrected_intensities = result.get('intensities', self.intensities)
            batch_background = result.get('background', None)
            fitted_curve = result.get('fitted_curve', None)
            
            # Plot original spectrum
            if len(original_intensities) > 0:
                self.ax_main.plot(self.wavenumbers, original_intensities, 'b-', 
                                 linewidth=1.5, alpha=0.7, label='Original Spectrum')
            
            # Plot optimized background from batch processing
            if batch_background is not None and len(batch_background) > 0:
                self.ax_main.plot(self.wavenumbers, batch_background, 'r--', 
                                 linewidth=2, alpha=0.8, label='Optimized Background')
            
            # Plot background-corrected spectrum
            if len(corrected_intensities) > 0:
                self.ax_main.plot(self.wavenumbers, corrected_intensities, 'k-', 
                                 linewidth=1.5, label='Background Corrected')
            
            # Plot fitted curve if available
            if fitted_curve is not None and len(fitted_curve) > 0:
                self.ax_main.plot(self.wavenumbers, fitted_curve, 'g-', 
                                 linewidth=2, label='Fitted Peaks')
        
        elif self.background_preview_active and self.background_preview is not None:
            # PREVIEW MODE: Show real-time background parameter preview
            self.ax_main.plot(self.wavenumbers, self.original_intensities, 'b-', 
                             linewidth=1.5, alpha=0.7, label='Original Spectrum')
            self.ax_main.plot(self.wavenumbers, self.background_preview, 'orange', 
                             linewidth=2, alpha=0.8, linestyle='--', label='Background Preview')
            
            # Show the preview corrected spectrum
            preview_corrected = self.original_intensities - self.background_preview
            self.ax_main.plot(self.wavenumbers, preview_corrected, 'g-', 
                             linewidth=1.5, alpha=0.7, label='Preview Corrected')
        
        elif self.background is not None:
            # APPLIED BACKGROUND MODE: Show manually applied background
            self.ax_main.plot(self.wavenumbers, self.background, 'r--', 
                             linewidth=1.5, alpha=0.8, label='Applied Background')
            self.ax_main.plot(self.wavenumbers, self.intensities, 'b-', 
                             linewidth=1.5, label='Background Corrected')
        
        else:
            # RAW SPECTRUM MODE: Show original spectrum only
            self.ax_main.plot(self.wavenumbers, self.intensities, 'b-', 
                             linewidth=1.5, label='Raw Spectrum')
        
        # Plot peaks
        if len(self.peaks) > 0:
            peak_positions = [self.wavenumbers[int(p)] for p in self.peaks if 0 <= int(p) < len(self.wavenumbers)]
            peak_intensities = [self.intensities[int(p)] for p in self.peaks if 0 <= int(p) < len(self.intensities)]
            self.ax_main.plot(peak_positions, peak_intensities, 'ro', 
                             markersize=8, label='Peaks')
        
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
        
        # Add mode indicator and appropriate title
        if showing_batch_result:
            mode_text = "Batch Result View"
            result_file = os.path.basename(self.batch_results[self.current_spectrum_index].get('file', ''))
            r2_value = self.batch_results[self.current_spectrum_index].get('r_squared', 0)
            title_text = f'{mode_text}: {result_file} (R² = {r2_value:.4f})' if r2_value > 0 else f'{mode_text}: {result_file}'
        elif self.background_preview_active:
            mode_text = "Background Preview Mode"
            bg_method = self.bg_method_combo.currentText()
            title_text = f'{mode_text} - {bg_method}'
        else:
            mode_text = "Interactive Mode"
            title_text = f'{mode_text}'
        
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title(title_text)
        
        # Create transparent legend (90% transparency)
        legend = self.ax_main.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, 
                                    facecolor='white', edgecolor='gray', framealpha=0.1)
        legend.set_zorder(1000)  # Ensure legend is on top but transparent
        if self.show_grid_check.isChecked():
            self.ax_main.grid(True, alpha=0.3)
        
        # Residuals plot - show appropriate residuals based on mode
        if showing_batch_result:
            result = self.batch_results[self.current_spectrum_index]
            batch_residuals = result.get('residuals', None)
            if batch_residuals is not None and len(batch_residuals) > 0:
                self.ax_residual.plot(self.wavenumbers, batch_residuals, 'k-', linewidth=1)
                self.ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
                self.ax_residual.set_ylabel('Residuals')
                self.ax_residual.set_title('Batch Fit Residuals')
                if self.show_grid_check.isChecked():
                    self.ax_residual.grid(True, alpha=0.3)
        elif self.residuals is not None:
            self.ax_residual.plot(self.wavenumbers, self.residuals, 'k-', linewidth=1)
            self.ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_residual.set_ylabel('Residuals')
            self.ax_residual.set_title('Interactive Fit Residuals')
            if self.show_grid_check.isChecked():
                self.ax_residual.grid(True, alpha=0.3)
        
        self.canvas_current.draw()
        
        # Update live view when current plot updates
        if hasattr(self, 'wavenumbers') and len(self.wavenumbers) > 0:
            try:
                self._update_live_view_with_current_spectrum()
            except:
                pass  # Ignore errors during initialization
    
    def plot_individual_peaks(self):
        """Plot individual fitted peaks."""
        if (self.fit_params is None or 
            not hasattr(self.fit_params, '__len__') or 
            len(self.fit_params) == 0 or 
            len(self.peaks) == 0):
            return
            
        # Determine parameters per peak based on model type
        if self.current_model in ["Pseudo-Voigt", "Asymmetric Voigt"]:
            if self.current_model == "Pseudo-Voigt":
                params_per_peak = 4  # amp, cen, wid, eta
            else:  # Asymmetric Voigt
                params_per_peak = 5  # amp, cen, wid, eta, asym
        else:
            params_per_peak = 3  # amp, cen, wid for Gaussian and Lorentzian
            
        n_peaks = len(self.peaks)
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            if start_idx + (params_per_peak - 1) < len(self.fit_params):
                if self.current_model == "Gaussian":
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                elif self.current_model == "Lorentzian":
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    peak_curve = self.lorentzian(self.wavenumbers, amp, cen, wid)
                elif self.current_model == "Pseudo-Voigt":
                    amp, cen, wid, eta = self.fit_params[start_idx:start_idx+4]
                    peak_curve = self.pseudo_voigt(self.wavenumbers, amp, cen, wid, eta)
                elif self.current_model == "Asymmetric Voigt":
                    amp, cen, wid, eta, asym = self.fit_params[start_idx:start_idx+5]
                    peak_curve = self.asymmetric_voigt(self.wavenumbers, amp, cen, wid, eta, asym)
                else:
                    # Default to Gaussian for unknown models
                    amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                    peak_curve = self.gaussian(self.wavenumbers, amp, cen, wid)
                
                # Plot individual peak curve
                label = f'Peak {i+1}: {cen:.1f}cm⁻¹'
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
        
        # Create clean 2x2 layout for peak data only
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
        
        # Get display options from checkboxes
        show_grid = hasattr(self, 'trends_show_grid') and self.trends_show_grid.isChecked()
        show_peak_labels = hasattr(self, 'trends_show_peak_labels') and self.trends_show_peak_labels.isChecked()
        show_trend_lines = hasattr(self, 'trends_show_trend_lines') and self.trends_show_trend_lines.isChecked()
        
        # Add trend lines if requested
        if show_trend_lines and len(file_indices) > 1:
            # Add linear trend lines to each plot
            x_trend = np.array(file_indices)
            
            # Position trends
            for peak_idx in range(n_peaks):
                if peak_idx >= len(self.peak_visibility_vars) or not self.peak_visibility_vars[peak_idx].isChecked():
                    continue
                positions = []
                for result in successful_results:
                    if 'fit_params' in result and result['fit_params'] is not None:
                        params = result['fit_params']
                        start_idx = peak_idx * 3
                        if start_idx + 2 < len(params):
                            positions.append(params[start_idx + 1])  # center position
                        else:
                            positions.append(np.nan)
                    else:
                        positions.append(np.nan)
                
                # Fit linear trend if we have valid data
                valid_mask = ~np.isnan(positions)
                if np.sum(valid_mask) > 1:
                    z = np.polyfit(x_trend[valid_mask], np.array(positions)[valid_mask], 1)
                    p = np.poly1d(z)
                    ax1.plot(x_trend, p(x_trend), '--', alpha=0.5, linewidth=1)
            
            # R² trend
            if len(r_squared_values) > 1:
                z = np.polyfit(x_trend, r_squared_values, 1)
                p = np.poly1d(z)
                ax4.plot(x_trend, p(x_trend), 'k--', alpha=0.5, linewidth=1)
        
        # Format plots
        ax1.set_title('Peak Positions')
        ax1.set_ylabel('Wavenumber (cm⁻¹)')
        if show_peak_labels:
            ax1.legend()
        if show_grid:
            ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Peak Amplitudes')
        ax2.set_ylabel('Intensity')
        if show_peak_labels:
            ax2.legend()
        if show_grid:
            ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Peak Widths')
        ax3.set_xlabel('Spectrum Index')
        ax3.set_ylabel('Width')
        if show_peak_labels:
            ax3.legend()
        if show_grid:
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
        if show_grid:
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
                        elif data_type == "Residuals":
                            # Calculate residuals on-the-fly
                            bg_method = self.bg_method_combo.currentText()
                            if bg_method.startswith("ALS"):
                                lambda_value = 10 ** self.lambda_slider.value()
                                p_value = self.p_slider.value() / 1000.0
                                background = self.baseline_als(original_intensities, lambda_value, p_value)
                                bg_corrected_intensities = original_intensities - background
                            else:
                                background = self.baseline_als(original_intensities, 1e5, 0.01)
                                bg_corrected_intensities = original_intensities - background
                            
                            # Calculate fitted curve if we have reference peaks and fit parameters
                            if (self.reference_peaks is not None and len(self.reference_peaks) > 0 and 
                                self.fit_params is not None and len(self.fit_params) > 0):
                                
                                # Temporarily store data for model function
                                temp_wavenumbers = self.wavenumbers
                                temp_peaks = self.peaks
                                self.wavenumbers = wavenumbers
                                self.peaks = self.reference_peaks
                                
                                try:
                                    fitted_curve = self.multi_peak_model(wavenumbers, *self.fit_params)
                                    intensities = bg_corrected_intensities - fitted_curve  # Calculate residuals
                                except Exception as e:
                                    # If fitting fails, show zeros or background corrected data
                                    intensities = np.zeros_like(bg_corrected_intensities)
                                finally:
                                    # Restore original data
                                    self.wavenumbers = temp_wavenumbers
                                    self.peaks = temp_peaks
                            else:
                                # No fit available, show zeros or inform user
                                intensities = np.zeros_like(bg_corrected_intensities)
                        
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
                
                # Add warning message for residuals without proper fitting
                ref_peaks_invalid = (self.reference_peaks is None or 
                                   (hasattr(self.reference_peaks, '__len__') and len(self.reference_peaks) == 0))
                fit_params_invalid = (self.fit_params is None or len(self.fit_params) == 0)
                
                if data_type == "Residuals" and (ref_peaks_invalid or fit_params_invalid):
                    ax.text(0.5, 0.95, 'Warning: No reference peaks or fit parameters available.\nRun peak fitting and "Set Reference" first, then "Apply to All" for proper residuals.', 
                           ha='center', va='top', transform=ax.transAxes, fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                
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
    
    # Waterfall UI methods
    def update_waterfall_line_width_label(self):
        """Update waterfall line width label."""
        value = self.waterfall_line_width.value() / 10.0
        self.waterfall_line_width_label.setText(f"{value:.1f}")
    
    def update_waterfall_alpha_label(self):
        """Update waterfall alpha label.""" 
        value = self.waterfall_alpha.value() / 100.0
        self.waterfall_alpha_label.setText(f"{value:.2f}")
    
    def update_waterfall_contrast_label(self):
        """Update waterfall contrast label."""
        value = self.waterfall_contrast.value()
        self.waterfall_contrast_label.setText(str(value))
    
    def update_waterfall_brightness_label(self):
        """Update waterfall brightness label."""
        value = self.waterfall_brightness.value()
        self.waterfall_brightness_label.setText(str(value))
    
    def update_waterfall_gamma_label(self):
        """Update waterfall gamma label."""
        value = self.waterfall_gamma.value() / 100.0
        self.waterfall_gamma_label.setText(f"{value:.1f}")
    
    def on_waterfall_auto_range_changed(self):
        """Handle waterfall auto range checkbox change."""
        auto_enabled = self.waterfall_auto_range.isChecked()
        self.waterfall_range_min.setEnabled(not auto_enabled)
        self.waterfall_range_max.setEnabled(not auto_enabled)
        if auto_enabled:
            self.update_waterfall_plot()

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
        
        data_type = self.heatmap_data_combo.currentText()
        
        # For residuals and fitted peaks, we need batch results
        if data_type in ["Residuals", "Fitted Peaks"] and self.batch_results:
            # Get successful results
            successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
            
            if not successful_results:
                ax = self.figure_heatmap.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, 'No successful fits available\nfor residuals or fitted peaks', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas_heatmap.draw()
                return
            
            data_source = successful_results
        
        # For other data types, try to use loaded spectrum files directly
        elif self.spectra_files:
            try:
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
            if data_type in ["Residuals", "Fitted Peaks"]:
                ax.text(0.5, 0.5, f'No batch results available for {data_type}\n\n'
                       'To view residuals or fitted peaks:\n'
                       '1. Load spectrum files\n'
                       '2. Fit peaks and set reference\n'
                       '3. Run "Apply to All" batch processing', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            else:
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
                    # If no residuals available, skip this spectrum
                    if len(intensities) == 0:
                        continue
                else:
                    intensities = np.array(result.get('intensities', []))
                
                if len(wavenumbers) > 0 and len(intensities) > 0:
                    heatmap_data.append(intensities)
                    wavenumber_ranges.append(wavenumbers)
            
            if not heatmap_data:
                ax = self.figure_heatmap.add_subplot(1, 1, 1)
                if data_type == "Residuals":
                    ax.text(0.5, 0.5, 'No residuals data available\n\n'
                           'Residuals are calculated during batch processing.\n'
                           'Make sure to run "Apply to All" after setting\n'
                           'reference peaks to generate residuals data.', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                else:
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
            
            # Grid - explicitly control grid display
            if self.heatmap_grid.isChecked():
                ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            else:
                ax.grid(False)  # Explicitly turn off grid
            
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

    def update_fitting_quality_plot(self):
        """Update fitting quality plot with 3x3 grid showing best, median, and worst fits."""
        if not hasattr(self, 'figure_fitting_quality') or not hasattr(self, 'canvas_fitting_quality'):
            print("DEBUG: No fitting quality figure or canvas found")
            return
        
        # Clear all axes
        for i in range(3):
            for j in range(3):
                self.axes_fitting_quality[i][j].clear()
        
        # Check if we have batch results
        print(f"DEBUG: batch_results length: {len(self.batch_results) if hasattr(self, 'batch_results') else 'No batch_results attribute'}")
        if not self.batch_results:
            # Show message in center plot
            ax = self.axes_fitting_quality[1][1]
            ax.text(0.5, 0.5, 'No batch results available.\nRun batch processing first.',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas_fitting_quality.draw()
            return
        
        # Get successful results and sort by R²
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        print(f"DEBUG: successful_results length: {len(successful_results)}")
        if len(successful_results) > 0:
            print(f"DEBUG: First result keys: {list(successful_results[0].keys())}")
        
        if len(successful_results) == 0:
            # Show message in center plot
            ax = self.axes_fitting_quality[1][1]
            ax.text(0.5, 0.5, 'No successful fits available.',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas_fitting_quality.draw()
            return
        
        # Sort by R² value (descending - best first)
        successful_results.sort(key=lambda x: x.get('r_squared', 0), reverse=True)
        
        n_results = len(successful_results)
        
        # Determine which results to show in each row
        if n_results >= 9:
            # Best 3 (top R²)
            best_results = successful_results[:3]
            # Median 3 (middle R²)
            median_start = max(0, n_results // 2 - 1)
            median_results = successful_results[median_start:median_start + 3]
            # Worst 3 (bottom R²)
            worst_results = successful_results[-3:]
        elif n_results >= 6:
            # Best 2, median 2, worst 2 (pad with empty)
            best_results = successful_results[:2] + [None]
            median_start = max(0, n_results // 2 - 1)
            median_results = successful_results[median_start:median_start + 2] + [None]
            worst_results = successful_results[-2:] + [None]
        elif n_results >= 3:
            # Show available results across the three rows
            best_results = [successful_results[0]] + [None, None]
            median_idx = n_results // 2
            median_results = [successful_results[median_idx]] + [None, None]
            worst_results = [successful_results[-1]] + [None, None]
        else:
            # Very few results - distribute what we have
            best_results = [successful_results[0] if n_results > 0 else None] + [None, None]
            median_results = [successful_results[1] if n_results > 1 else None] + [None, None]
            worst_results = [successful_results[-1] if n_results > 2 else None] + [None, None]
        
        # Plot each row
        rows = [best_results, median_results, worst_results]
        row_labels = ["Best", "Median", "Worst"]
        
        for row_idx, (results, label) in enumerate(zip(rows, row_labels)):
            for col_idx, result in enumerate(results):
                ax = self.axes_fitting_quality[row_idx][col_idx]
                
                if result is None:
                    # Empty subplot
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    continue
                
                # Extract data safely
                wavenumbers = np.array(result.get('wavenumbers', []))
                intensities = np.array(result.get('intensities', []))  # Background corrected
                original_intensities = np.array(result.get('original_intensities', []))  # Raw data
                background = np.array(result.get('background', []))
                fitted_curve = result.get('fitted_curve', None)
                r_squared = result.get('r_squared', 0)
                filename = os.path.basename(result['file'])
                
                # Check if we have valid data
                if len(wavenumbers) == 0 or len(intensities) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Plot original spectrum (raw data)
                if len(original_intensities) > 0:
                    ax.plot(wavenumbers, original_intensities, 'lightblue', linewidth=1, alpha=0.6, label='Raw')
                
                # Plot background if available
                if len(background) > 0:
                    ax.plot(wavenumbers, background, 'orange', linewidth=1, alpha=0.7, label='Background')
                
                # Plot background-corrected spectrum
                ax.plot(wavenumbers, intensities, 'b-', linewidth=1.2, alpha=0.8, label='Corrected')
                
                # Plot fitted curve if available
                if fitted_curve is not None and len(fitted_curve) > 0:
                    ax.plot(wavenumbers, fitted_curve, 'r-', linewidth=1.5, label='Fitted')
                
                    # Plot individual peaks if available
                    if 'fit_params' in result and result['fit_params'] is not None:
                        fit_params = result['fit_params']
                        if len(fit_params) >= 3:  # At least one peak (amp, cen, wid)
                            # Plot individual peak contributions
                            colors = ['green', 'purple', 'brown', 'pink', 'gray', 'olive']
                            peak_count = 0
                            for i in range(0, len(fit_params), 3):
                                if i + 2 < len(fit_params):
                                    amp, cen, wid = fit_params[i:i+3]
                                    # Create individual peak curve
                                    peak_curve = self.gaussian(wavenumbers, amp, cen, wid)
                                    color = colors[peak_count % len(colors)]
                                    ax.plot(wavenumbers, peak_curve, '--', linewidth=1, alpha=0.6, color=color,
                                           label=f'Peak {peak_count+1}' if peak_count < 3 else '')
                                    peak_count += 1
                
                # Set title with filename and R² (indicate if peak-based)
                short_filename = filename[:12] + '...' if len(filename) > 15 else filename
                if result.get('peak_based_r_squared', False):
                    ax.set_title(f"{short_filename}\nPeak-Avg R² = {r_squared:.4f}", fontsize=8, pad=3)
                else:
                    ax.set_title(f"{short_filename}\nR² = {r_squared:.4f}", fontsize=8, pad=3)
                
                # Format axes
                ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=8)
                ax.set_ylabel('Intensity', fontsize=8)
                ax.tick_params(labelsize=7)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Add legend only to first plot of each row
                if col_idx == 0:
                    ax.legend(fontsize=6, loc='upper right', frameon=True, fancybox=True, 
                             framealpha=0.8, edgecolor='gray')
        
        # Set overall row titles
        self.axes_fitting_quality[0][1].set_title("Best Fitting Results (Top Peak-Based R²)", fontsize=12, fontweight='bold', pad=15)
        self.axes_fitting_quality[1][1].set_title("Median Fitting Results", fontsize=12, fontweight='bold', pad=15)
        self.axes_fitting_quality[2][1].set_title("Poorest Fitting Results (Lowest Peak-Based R²)", fontsize=12, fontweight='bold', pad=15)
        
        # Tight layout and draw
        self.figure_fitting_quality.tight_layout(pad=2.0)
        self.canvas_fitting_quality.draw()

    def clear_manual_peaks(self):
        """Clear only manually selected peaks."""
        self.manual_peaks = np.array([], dtype=int)
        self.update_peak_count_display()
        self.update_peak_info_display()
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
        self.update_peak_info_display()
        
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
            
            # Check if we have fitted parameters - if so, show fitted positions (more accurate)
            # Fixed: Check for ANY peaks (auto OR manual), not just auto peaks
            total_peaks = auto_count + manual_count
            if (hasattr(self, 'fit_params') and self.fit_params is not None and 
                len(self.fit_params) > 0 and total_peaks > 0):
                
                # Show fitted peak centers (from curve fitting - more accurate)
                params_per_peak = self.get_params_per_peak()
                n_peaks = len(self.fit_params) // params_per_peak
                
                print(f"DEBUG: Found {n_peaks} fitted peaks to display (from {len(self.fit_params)} parameters)")
                
                for i in range(n_peaks):
                    start_idx = i * params_per_peak
                    if start_idx + 2 < len(self.fit_params):
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        item_text = f"✓ Fitted Peak {i+1}: {cen:.1f} cm⁻¹ (Amp={amp:.0f})"
                        self.peak_list_widget.addItem(item_text)
                        print(f"DEBUG: Added fitted peak {i+1}: {cen:.1f} cm⁻¹")
            else:
                # Fallback to detected peak positions (initial rough estimates)
                # Add automatic peaks
                if hasattr(self, 'peaks') and self.peaks is not None and len(self.wavenumbers) > 0:
                    for i, peak_idx in enumerate(self.peaks):
                        if 0 <= peak_idx < len(self.wavenumbers):
                            wavenumber = self.wavenumbers[peak_idx]
                            intensity = self.intensities[peak_idx] if peak_idx < len(self.intensities) else 0
                            item_text = f"🔍 Detected Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I={intensity:.0f})"
                            self.peak_list_widget.addItem(item_text)
                
                # Add manual peaks
                if hasattr(self, 'manual_peaks') and self.manual_peaks is not None and len(self.wavenumbers) > 0:
                    for i, peak_idx in enumerate(self.manual_peaks):
                        if 0 <= peak_idx < len(self.wavenumbers):
                            wavenumber = self.wavenumbers[peak_idx]
                            intensity = self.intensities[peak_idx] if peak_idx < len(self.intensities) else 0
                            item_text = f"👆 Manual Peak {i+1}: {wavenumber:.1f} cm⁻¹ (I={intensity:.0f})"
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
            
            # Ensure None values are converted to empty lists for safe length checking
            if original_intensities is None:
                original_intensities = []
            if intensities is None:
                intensities = []
            if background is None:
                background = []
            if fitted_curve is None:
                fitted_curve = []
            if residuals is None:
                residuals = []
            
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
            
            # Initialize peak_based_r2 before using it
            peak_based_r2 = 0.0
            individual_peak_r2 = []
            
            # Calculate peak-based R² early if we have fitted data
            if len(fitted_curve) > 0 and fit_params is not None and len(fit_params) > 0:
                try:
                    individual_peak_r2, peak_based_r2 = self.calculate_individual_peak_r_squared(
                        wavenumbers, intensities, fit_params, getattr(self, 'reference_model', 'Gaussian')
                    )
                except Exception as e:
                    print(f"Error calculating peak-based R² for live view: {e}")
                    peak_based_r2 = r_squared  # Fallback to original R²
            
            # Use peak-based R² for title if available
            if peak_based_r2 > 0:
                title = f'Live Fit: {filename} (R² = {peak_based_r2:.4f})'
            elif r_squared > 0:
                title = f'Live Fit: {filename} (R² = {r_squared:.4f})'
            else:
                title = f'Live View: {filename}'
            
            self.ax_live_main.set_ylabel('Intensity')
            self.ax_live_main.set_title(title)
            
            # Create transparent legend (90% transparency)
            legend = self.ax_live_main.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, 
                                             facecolor='white', edgecolor='gray', framealpha=0.1)
            legend.set_zorder(1000)  # Ensure legend is on top but transparent
            
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
            
            # Display debug information if we calculated peak-based R²
            if peak_based_r2 > 0:
                print(f"LIVE VIEW DEBUG:")
                print(f"  File: {os.path.basename(result_data.get('file', 'Unknown'))}")
                print(f"  Traditional R²: {r_squared:.4f}")
                print(f"  Peak-based R²: {peak_based_r2:.4f}")
                print(f"  Individual peak R²: {[f'{x:.3f}' for x in individual_peak_r2]}")
                print(f"  Peak parameters: {fit_params}")
                print(f"  Background method appears to be: {'Linear' if len(set(np.diff(result_data.get('background', [])))) < 3 else 'Non-linear'}")
            
            # Update statistics with peak-based R²
            if hasattr(self, 'overall_r2_label'):
                if peak_based_r2 > 0:
                    self.overall_r2_label.setText(f"Peak-Avg R²: {peak_based_r2:.4f}")
                elif r_squared > 0:
                    self.overall_r2_label.setText(f"Overall R²: {r_squared:.4f}")
                else:
                    self.overall_r2_label.setText("R²: --")
            
            if hasattr(self, 'peak_count_live'):
                if hasattr(self, 'reference_peaks') and self.reference_peaks is not None:
                    self.peak_count_live.setText(f"Peaks: {len(self.reference_peaks)}")
                else:
                    self.peak_count_live.setText("Peaks: --")
            
                        # Update peak statistics with individual R² values
            if fit_params is not None and len(fit_params) > 0 and hasattr(self, 'peak_stats_layout'):
                try:
                    effective_r2 = peak_based_r2 if peak_based_r2 > 0 else r_squared
                    self.update_peak_statistics(fit_params, effective_r2, individual_peak_r2)
                except Exception as e:
                    print(f"Error updating peak statistics: {e}")
            
            # Store current data for analysis
            self.current_live_data = result_data.copy()
            
            # Enable analyze button for Smart Residual Analyzer only if we have fitted data
            if hasattr(self, 'analyze_btn'):
                self.analyze_btn.setEnabled(len(fitted_curve) > 0 and (peak_based_r2 > 0 or r_squared > 0))
            
            self.canvas_live.draw()
            QApplication.processEvents()
            
        except Exception as e:
            print(f"Error updating live view: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_live_view_with_current_spectrum(self):
        """Update live view using current spectrum data - helper method."""
        try:
            if not hasattr(self, 'wavenumbers') or len(self.wavenumbers) == 0:
                return
                
            # Prepare result data structure for live view
            result_data = {
                'file': self.spectra_files[self.current_spectrum_index] if hasattr(self, 'current_spectrum_index') and self.current_spectrum_index < len(self.spectra_files) else 'Current Spectrum',
                'original_intensities': getattr(self, 'original_intensities', self.intensities),
                'intensities': self.intensities,
                'background': getattr(self, 'background', None),
                'fitted_curve': None,
                'residuals': getattr(self, 'residuals', None),
                'r_squared': 0
            }
            
            # Add fitted curve if available
            if (hasattr(self, 'fit_params') and self.fit_params is not None and 
                len(self.fit_params) > 0 and hasattr(self, 'peaks') and len(self.peaks) > 0):
                try:
                    fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
                    result_data['fitted_curve'] = fitted_curve
                    
                    # Calculate R² if we have fitted curve
                    if self.residuals is not None:
                        ss_res = np.sum(self.residuals ** 2)
                        ss_tot = np.sum((self.intensities - np.mean(self.intensities)) ** 2)
                        result_data['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                except Exception as e:
                    print(f"Error calculating fitted curve for live view: {e}")
            
            # Update live view with prepared data
            self.update_live_view(result_data, self.wavenumbers, getattr(self, 'fit_params', []))
            
        except Exception as e:
            print(f"Error updating live view with current spectrum: {e}")
    
    def plot_live_individual_peaks(self, wavenumbers, fit_params):
        """Plot individual fitted peaks in live view."""
        # Safely check reference_peaks
        if (self.reference_peaks is None or 
            (hasattr(self.reference_peaks, '__len__') and len(self.reference_peaks) == 0) or 
            fit_params is None or len(fit_params) == 0):
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
                
                # Plot individual peak curve
                label = f'Peak {i+1}: {cen:.1f}cm⁻¹'
                self.ax_live_main.plot(wavenumbers, peak_curve, '--', 
                                     linewidth=1, alpha=0.6, label=label)
    
    def update_peak_statistics(self, fit_params, overall_r2, individual_peak_r2=None):
        """Update individual peak statistics display with actual peak-based R² if available."""
        # Clear existing statistics
        for i in reversed(range(self.peak_stats_layout.count())):
            child = self.peak_stats_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Safely check reference_peaks
        if (self.reference_peaks is None or 
            (hasattr(self.reference_peaks, '__len__') and len(self.reference_peaks) == 0) or 
            fit_params is None or len(fit_params) == 0):
            return
            
        n_peaks = len(self.reference_peaks)
        
        for i in range(n_peaks):
            start_idx = i * 3
            if start_idx + 2 < len(fit_params):
                amp, cen, wid = fit_params[start_idx:start_idx+3]
                
                # Use actual individual peak R² if available, otherwise estimate
                if individual_peak_r2 and i < len(individual_peak_r2):
                    peak_r2 = individual_peak_r2[i]
                    r2_label = f"R²={peak_r2:.3f}"
                else:
                    # Fallback to estimation for backward compatibility
                    peak_contribution = abs(amp) / (np.sum([abs(fit_params[j*3]) for j in range(n_peaks)]) + 1e-10)
                    estimated_r2 = overall_r2 * peak_contribution
                    r2_label = f"R²≈{estimated_r2:.3f}"
                
                peak_info = QLabel(f"Peak {i+1}: Pos={cen:.1f} cm⁻¹, Amp={amp:.1f}, Width={wid:.1f}, {r2_label}")
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
    
    def update_batch_results_with_manual_fit(self):
        """Update batch results with the current manual fit adjustments."""
        if not hasattr(self, 'current_spectrum_index') or self.current_spectrum_index < 0:
            QMessageBox.warning(self, "No Spectrum Selected", "Please select a spectrum to update.")
            return
            
        if not hasattr(self, 'wavenumbers') or len(self.wavenumbers) == 0:
            QMessageBox.warning(self, "No Data", "No spectrum data loaded.")
            return
        
        try:
            # Get current file path
            current_file = self.spectra_files[self.current_spectrum_index]
            
            # Create a manual fit result data structure
            ref_peaks = self.reference_peaks.copy() if (self.reference_peaks is not None and len(self.reference_peaks) > 0) else []
            manual_result = {
                'file': current_file,
                'wavenumbers': self.wavenumbers.copy(),
                'original_intensities': getattr(self, 'original_intensities', self.intensities).copy(),
                'intensities': self.intensities.copy(),
                'background': getattr(self, 'background', None),
                'fitted_curve': None,
                'residuals': getattr(self, 'residuals', None),
                'r_squared': 0,
                'fit_params': getattr(self, 'fit_params', []),
                'fit_failed': False,
                'manual_fit': True,  # Flag to indicate this was manually adjusted
                'reference_peaks': ref_peaks,
                'peaks': ref_peaks,  # Add for compatibility with loading logic
                'background_method': self.bg_method_combo.currentText() if hasattr(self, 'bg_method_combo') else 'Unknown',
                'peak_model': getattr(self, 'current_model', 'Gaussian'),
                'n_peaks_fitted': len(ref_peaks),
                'fitted_centers': []  # Will be populated below if fit_params exist
            }
            
            # Calculate fitted curve if we have fit parameters
            if (hasattr(self, 'fit_params') and self.fit_params is not None and 
                len(self.fit_params) > 0 and hasattr(self, 'reference_peaks') and 
                self.reference_peaks is not None and len(self.reference_peaks) > 0):
                
                try:
                    fitted_curve = self.multi_peak_model(self.wavenumbers, *self.fit_params)
                    manual_result['fitted_curve'] = fitted_curve
                    
                    # Extract peak centers from fit parameters
                    params_per_peak = 3  # Assuming Gaussian/Lorentzian model
                    if hasattr(self, 'current_model'):
                        if self.current_model == "Pseudo-Voigt":
                            params_per_peak = 4
                        elif self.current_model == "Asymmetric Voigt":
                            params_per_peak = 5
                    
                    fitted_centers = []
                    n_peaks = len(self.fit_params) // params_per_peak
                    for i in range(n_peaks):
                        center_idx = i * params_per_peak + 1  # Center is second parameter
                        if center_idx < len(self.fit_params):
                            fitted_centers.append(self.fit_params[center_idx])
                    manual_result['fitted_centers'] = fitted_centers
                    
                    # Calculate R²
                    if self.residuals is not None and len(self.residuals) > 0:
                        ss_res = np.sum(self.residuals ** 2)
                        ss_tot = np.sum((self.intensities - np.mean(self.intensities)) ** 2)
                        manual_result['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    else:
                        # Calculate residuals if not available
                        residuals = self.intensities - fitted_curve
                        manual_result['residuals'] = residuals
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((self.intensities - np.mean(self.intensities)) ** 2)
                        manual_result['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                except Exception as e:
                    print(f"Error calculating fitted curve for manual fit: {e}")
                    manual_result['fit_failed'] = True
            
            # Update or add to batch results
            updated = False
            for i, result in enumerate(self.batch_results):
                if result.get('file') == current_file:
                    # Update existing result
                    self.batch_results[i] = manual_result
                    updated = True
                    break
            
            if not updated:
                # Add new result
                self.batch_results.append(manual_result)
            
            # Update all plots to reflect the changes
            self.update_all_plots()
            
            # Explicitly update trends plot and peak visibility controls
            if hasattr(self, 'update_trends_plot'):
                self.update_trends_plot()
            if hasattr(self, 'update_peak_visibility_controls'):
                self.update_peak_visibility_controls()
            
            # Show confirmation
            r2_text = f" (R² = {manual_result['r_squared']:.4f})" if manual_result['r_squared'] > 0 else ""
            peaks_text = f" with {len(manual_result['reference_peaks'])} peaks" if (manual_result['reference_peaks'] is not None and len(manual_result['reference_peaks']) > 0) else ""
            
            QMessageBox.information(
                self, 
                "Manual Fit Saved", 
                f"Manual fit for {os.path.basename(current_file)} has been saved to batch results{r2_text}{peaks_text}.\n\n"
                f"This fit is now marked as manually adjusted and will be preserved during future batch operations."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update batch results with manual fit:\n{str(e)}")
            print(f"Error updating batch results with manual fit: {e}")
            import traceback
            traceback.print_exc()
    
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
                        if self.heatmap_grid.isChecked():
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
            if self.heatmap_grid.isChecked():
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

    def analyze_current_fit(self):
        """Analyze the current fit using Smart Residual Analyzer."""
        try:
            if not hasattr(self, 'current_live_data') or not self.current_live_data:
                QMessageBox.warning(self, "No Data", "No fit data available for analysis.\nPlease perform peak fitting first.")
                return
            
            # Implement advanced residual analysis here
            # This is a placeholder for the Smart Residual Analyzer functionality
            QMessageBox.information(self, "Analysis", "Smart Residual Analyzer would analyze the current fit here.\nThis feature is under development.")
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze current fit: {str(e)}")

    def export_comprehensive_batch_data(self):
        """Export comprehensive batch processing data to CSV."""
        if not self.batch_results:
            QMessageBox.warning(self, "No Results", "No batch processing results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Comprehensive Batch Data", "", 
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Create comprehensive data export including all parameters
                data = []
                
                for i, result in enumerate(self.batch_results):
                    base_row = {
                        'Spectrum_Index': i,
                        'Filename': os.path.basename(result['file']),
                        'Fit_Success': not result.get('fit_failed', True),
                        'R_Squared': result.get('r_squared', 0),
                        'Background_Method': result.get('background_method', 'Unknown')
                    }
                    
                    if not result.get('fit_failed', True) and 'fit_params' in result:
                        params = result['fit_params']
                        n_peaks = len(params) // 3
                        
                        # Add individual peak data
                        for peak_idx in range(n_peaks):
                            start_idx = peak_idx * 3
                            if start_idx + 2 < len(params):
                                amp, cen, wid = params[start_idx:start_idx+3]
                                peak_row = base_row.copy()
                                peak_row.update({
                                    'Peak_Number': peak_idx + 1,
                                    'Peak_Position': cen,
                                    'Peak_Amplitude': amp,
                                    'Peak_Width': wid,
                                    'Peak_Area': amp * wid * np.sqrt(2 * np.pi)  # Approximate Gaussian area
                                })
                                data.append(peak_row)
                    else:
                        base_row['Error'] = result.get('error', 'Unknown error')
                        data.append(base_row)
                
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Complete", f"Comprehensive batch data exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export comprehensive data: {str(e)}")

    def export_all_plot_data(self):
        """Export all plot data (spectra, fits, residuals) to files."""
        if not self.batch_results:
            QMessageBox.warning(self, "No Results", "No batch processing results to export.")
            return
        
        # Ask user to select directory for export
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not directory:
            return
        
        try:
            successful_exports = 0
            
            for i, result in enumerate(self.batch_results):
                if result.get('fit_failed', True):
                    continue
                
                filename = os.path.splitext(os.path.basename(result['file']))[0]
                
                # Prepare data for export
                wavenumbers = np.array(result.get('wavenumbers', []))
                original_intensities = np.array(result.get('original_intensities', []))
                intensities = np.array(result.get('intensities', []))
                fitted_curve = np.array(result.get('fitted_curve', []))
                residuals = np.array(result.get('residuals', []))
                background = np.array(result.get('background', []))
                
                if len(wavenumbers) > 0:
                    # Create export data
                    export_data = {
                        'Wavenumber': wavenumbers,
                        'Original_Intensity': original_intensities if len(original_intensities) > 0 else np.zeros_like(wavenumbers),
                        'Background_Corrected': intensities if len(intensities) > 0 else np.zeros_like(wavenumbers),
                        'Background': background if len(background) > 0 else np.zeros_like(wavenumbers),
                        'Fitted_Curve': fitted_curve if len(fitted_curve) > 0 else np.zeros_like(wavenumbers),
                        'Residuals': residuals if len(residuals) > 0 else np.zeros_like(wavenumbers)
                    }
                    
                    # Export to CSV
                    export_file = os.path.join(directory, f"{filename}_all_data.csv")
                    df = pd.DataFrame(export_data)
                    df.to_csv(export_file, index=False)
                    successful_exports += 1
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Exported data for {successful_exports} spectra to {directory}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export plot data: {str(e)}")

    def show_help(self):
        """Show help dialog."""
        help_text = """
        <html>
        <head>
            <title>Batch Peak Fitting Help</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2E86AB; }
                h2 { color: #A23B72; }
                h3 { color: #F18F01; }
                .step { background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .warning { background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }
                .tip { background-color: #d1ecf1; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }
                code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>📊 Batch Peak Fitting Tool</h1>
            <p>This tool allows you to fit peaks across multiple Raman spectra using a reference spectrum approach.</p>
            
            <h2>🚀 Quick Start Guide</h2>
            
            <div class="step">
                <h3>Step 1: Load Spectrum Files</h3>
                <p>Go to the <strong>File Selection</strong> tab and click "Add Files" to load your Raman spectra (.txt files).</p>
            </div>
            
            <div class="step">
                <h3>Step 2: Set Up Peak Detection</h3>
                <p>Navigate through spectra using the arrow buttons and use the <strong>Peak Detection</strong> tab to find peaks in a representative spectrum.</p>
            </div>
            
            <div class="step">
                <h3>Step 3: Fit Peaks</h3>
                <p>Click "Fit Peaks" to perform the fitting on the current spectrum. Adjust parameters as needed.</p>
            </div>
            
            <div class="step">
                <h3>Step 4: Set Reference</h3>
                <p>Once you have a good fit, go to the <strong>Batch</strong> tab and click "Set Reference" to use the current spectrum as your template.</p>
            </div>
            
            <div class="step">
                <h3>Step 5: Apply to All</h3>
                <p>Click "Apply to All" to fit all loaded spectra using the reference peak positions.</p>
            </div>
            
            <h2>📈 Visualization Options</h2>
            
            <h3>Waterfall Plot</h3>
            <p>The waterfall plot supports multiple data types:</p>
            <ul>
                <li><strong>Raw Intensity:</strong> Original spectrum data</li>
                <li><strong>Background Corrected:</strong> Data after background subtraction</li>
                <li><strong>Fitted Peaks:</strong> Reconstructed spectra from fitted peak parameters</li>
                <li><strong>Residuals:</strong> Difference between measured and fitted data</li>
            </ul>
            
            <div class="tip">
                <h4>💡 Residuals Analysis</h4>
                <p>Residuals show the difference between your measured data and the fitted model. This is useful for:</p>
                <ul>
                    <li>Evaluating fit quality - smaller residuals indicate better fits</li>
                    <li>Identifying systematic errors or missed peaks</li>
                    <li>Comparing fit performance across different spectra</li>
                </ul>
                <p><strong>To view residuals:</strong></p>
                <ol>
                    <li>Complete peak fitting and set a reference spectrum</li>
                    <li>Run "Apply to All" to generate batch results</li>
                    <li>In the Waterfall tab, select "Residuals" from the Data Type dropdown</li>
                </ol>
            </div>
            
            <div class="warning">
                <h4>⚠️ Important Note for Residuals</h4>
                <p>Residuals are only meaningful after peak fitting has been performed. If you select "Residuals" without first fitting peaks and running batch processing, you'll see a warning message and either zero values or incomplete data.</p>
            </div>
            
            <h3>Trends Plot</h3>
            <p>Shows how peak parameters (position, amplitude, width) change across spectra.</p>
            
            <h3>Heatmap</h3>
            <p>2D visualization of intensity variations across wavenumber and spectrum index.</p>
            
            <h2>🎛️ Advanced Features</h2>
            
            <h3>Background Subtraction</h3>
            <p>Use the <strong>Background/Smoothing</strong> tab to apply ALS (Asymmetric Least Squares) background correction.</p>
            
            <h3>Interactive Mode</h3>
            <p>Enable interactive mode to manually add or remove peaks by clicking on the spectrum.</p>
            
            <h3>Smart Residual Analyzer</h3>
            <p>The Live View tab includes a Smart Residual Analyzer that provides detailed analysis of fitting quality.</p>
            
            <h2>📁 File Export</h2>
            <p>Export results as CSV files containing peak parameters, R² values, and fitted data for further analysis.</p>
            
            <h2>🔧 Tips for Best Results</h2>
            <ul>
                <li>Choose a high-quality spectrum with clear peaks as your reference</li>
                <li>Adjust background subtraction parameters for your specific data</li>
                <li>Use the Live View to monitor fitting progress during batch processing</li>
                <li>Check residuals to validate fit quality</li>
                <li>Export comprehensive data for external analysis</li>
            </ul>
            
            <div class="tip">
                <h4>💡 Troubleshooting Common Issues</h4>
                <ul>
                    <li><strong>Residuals showing zeros:</strong> Make sure you've completed peak fitting and set a reference before viewing residuals</li>
                    <li><strong>Poor fits:</strong> Check background subtraction parameters and peak detection settings</li>
                    <li><strong>Batch processing fails:</strong> Ensure all spectra have similar wavenumber ranges and data quality</li>
                </ul>
            </div>
        </body>
        </html>
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
            
            # Update reference_peaks to match current peak state
            # This is crucial for the peak list display and fitting to work correctly
            if hasattr(self, 'peaks') and hasattr(self, 'manual_peaks'):
                all_peaks = np.concatenate([self.peaks, self.manual_peaks]) if len(self.manual_peaks) > 0 else self.peaks
                self.reference_peaks = all_peaks.copy() if len(all_peaks) > 0 else np.array([], dtype=int)
                print(f"DEBUG: Updated reference_peaks to {len(self.reference_peaks)} peaks after interactive click")
            
            # Update displays - both list widget and detailed info panel
            self.update_peak_count_display()
            self.update_peak_info_display()
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
        
        # Update fitting quality plot
        self.update_fitting_quality_plot()
        
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
        
        if hasattr(self, 'canvas_fitting_quality'):
            self.canvas_fitting_quality.mpl_connect('button_press_event', 
                                                   lambda event: self.on_plot_click(event, 'fitting_quality'))
        
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
                ax_main.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, 
                              facecolor='white', edgecolor='black', framealpha=1.0)
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
            
        elif plot_type == 'fitting_quality':
            figure = Figure(figsize=(16, 12))
            canvas = FigureCanvas(figure)
            toolbar = NavigationToolbar(canvas, popup)
            
            # Recreate fitting quality plot
            self._create_fitting_quality_plot_in_figure(figure)
            
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
        
        # Get display options (use defaults if trends tab checkboxes don't exist)
        show_grid = hasattr(self, 'trends_show_grid') and self.trends_show_grid.isChecked()
        show_peak_labels = hasattr(self, 'trends_show_peak_labels') and self.trends_show_peak_labels.isChecked()
        
        # Format plots
        ax1.set_title('Peak Positions')
        ax1.set_ylabel('Wavenumber (cm⁻¹)')
        if show_peak_labels:
            ax1.legend()
        if show_grid:
            ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Peak Amplitudes')
        ax2.set_ylabel('Intensity')
        if show_peak_labels:
            ax2.legend()
        if show_grid:
            ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Peak Widths')
        ax3.set_xlabel('Spectrum Index')
        ax3.set_ylabel('Width')
        if show_peak_labels:
            ax3.legend()
        if show_grid:
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
        if show_grid:
            ax4.grid(True, alpha=0.3)
        
        figure.tight_layout()

    def _create_fitting_quality_plot_in_figure(self, figure):
        """Helper method to create fitting quality plot in a specific figure for popup."""
        figure.clear()
        
        # Create 3x3 grid of subplots
        gs = figure.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Store axes
        axes = []
        for i in range(3):
            row_axes = []
            for j in range(3):
                ax = figure.add_subplot(gs[i, j])
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Check if we have batch results
        if not self.batch_results:
            # Show message in center plot
            ax = axes[1][1]
            ax.text(0.5, 0.5, 'No batch results available.\nRun batch processing first.',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Get successful results and sort by R²
        successful_results = [r for r in self.batch_results if not r.get('fit_failed', True)]
        
        if len(successful_results) == 0:
            # Show message in center plot
            ax = axes[1][1]
            ax.text(0.5, 0.5, 'No successful fits available.',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Sort by R² value (descending - best first)
        successful_results.sort(key=lambda x: x.get('r_squared', 0), reverse=True)
        
        n_results = len(successful_results)
        
        # Determine which results to show in each row
        if n_results >= 9:
            # Best 3 (top R²)
            best_results = successful_results[:3]
            # Median 3 (middle R²)
            median_start = max(0, n_results // 2 - 1)
            median_results = successful_results[median_start:median_start + 3]
            # Worst 3 (bottom R²)
            worst_results = successful_results[-3:]
        elif n_results >= 6:
            # Best 2, median 2, worst 2 (pad with empty)
            best_results = successful_results[:2] + [None]
            median_start = max(0, n_results // 2 - 1)
            median_results = successful_results[median_start:median_start + 2] + [None]
            worst_results = successful_results[-2:] + [None]
        elif n_results >= 3:
            # Show available results across the three rows
            best_results = [successful_results[0]] + [None, None]
            median_idx = n_results // 2
            median_results = [successful_results[median_idx]] + [None, None]
            worst_results = [successful_results[-1]] + [None, None]


# Launch function for integration with main app
def launch_batch_peak_fitting(parent, wavenumbers=None, intensities=None):
    """Launch the batch peak fitting window."""
    dialog = BatchPeakFittingQt6(parent, wavenumbers, intensities)
    dialog.exec()


# Standalone application launcher
if __name__ == "__main__":
    """Launch batch peak fitting as standalone application."""
    print("🚀 Launching RamanLab Batch Peak Fitting (Standalone)")
    print("=" * 50)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("RamanLab - Batch Peak Fitting")
    app.setApplicationVersion("2.0")
    
    # Create main window (use None as parent for standalone)
    window = BatchPeakFittingQt6(parent=None)
    window.setWindowTitle("RamanLab - Batch Peak Fitting")
    
    # Show window (use show() instead of exec() for main window)
    window.show()
    
    # Run application
    sys.exit(app.exec())
