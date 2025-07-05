#!/usr/bin/env python3
"""
Enhanced Peak Fitting and Spectral Deconvolution Module for RamanLab Qt6
Includes:
• Component separation
• Overlapping peak resolution  
• Principal component analysis
• Non-negative matrix factorization
• Direct file loading capability
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
    QMenuBar, QMenu, QFileDialog, QStatusBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QAction

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
from scipy.sparse import csc_matrix
import pandas as pd

# File loading utilities
try:
    from utils.file_loaders import SpectrumLoader, load_spectrum_file
    FILE_LOADING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: File loading utilities not available: {e}")
    FILE_LOADING_AVAILABLE = False

# Centralized peak fitting imports
try:
    from core.peak_fitting import PeakFitter, PeakData, auto_find_peaks, baseline_correct_spectrum
    from core.peak_fitting_ui import BackgroundControlsWidget, PeakFittingControlsWidget
    CENTRALIZED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Centralized peak fitting not available: {e}")
    CENTRALIZED_AVAILABLE = False

# Machine learning imports for advanced features
try:
    from sklearn.decomposition import PCA, NMF
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# Enhanced PeakFitter with baseline_als support
class EnhancedPeakFitter(PeakFitter if CENTRALIZED_AVAILABLE else object):
    """Enhanced peak fitter with baseline correction capabilities."""
    
    def __init__(self):
        if CENTRALIZED_AVAILABLE:
            super().__init__()
    
    @staticmethod
    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares Smoothing for baseline correction.
        
        Parameters:
        -----------
        y : array-like
            Input spectrum.
        lam : float
            Smoothness parameter (default: 1e5).
        p : float
            Asymmetry parameter (default: 0.01).
        niter : int
            Number of iterations (default: 10).
            
        Returns:
        --------
        array-like
            Estimated baseline.
        """
        L = len(y)
        D = csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        
        for i in range(niter):
            W = csc_matrix((w, (np.arange(L), np.arange(L))))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        
        return z
    
    # Ensure backward compatibility for standalone operation
    @staticmethod
    def gaussian(x, amp, cen, wid):
        """Gaussian peak function."""
        if CENTRALIZED_AVAILABLE:
            return PeakFitter.gaussian(x, amp, cen, wid)
        else:
            # Fallback implementation
            width = abs(wid) + 1e-10
            return amp * np.exp(-((x - cen) / width) ** 2)
    
    @staticmethod
    def lorentzian(x, amp, cen, wid):
        """Lorentzian peak function."""
        if CENTRALIZED_AVAILABLE:
            return PeakFitter.lorentzian(x, amp, cen, wid)
        else:
            # Fallback implementation
            width = abs(wid) + 1e-10
            return amp * (width**2) / ((x - cen)**2 + width**2)
    
    @staticmethod
    def pseudo_voigt(x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function."""
        if CENTRALIZED_AVAILABLE:
            return PeakFitter.pseudo_voigt(x, amp, cen, wid, eta)
        else:
            # Fallback implementation
            eta = np.clip(eta, 0, 1)
            gaussian_part = EnhancedPeakFitter.gaussian(x, 1.0, cen, wid)
            lorentzian_part = EnhancedPeakFitter.lorentzian(x, 1.0, cen, wid)
            return amp * ((1 - eta) * gaussian_part + eta * lorentzian_part)


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
    
    def __init__(self, parent, wavenumbers=None, intensities=None):
        super().__init__(parent)
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.parent = parent
        
        # Initialize spectrum data - can be empty if loading from file
        if wavenumbers is not None and intensities is not None:
            self.wavenumbers = np.array(wavenumbers)
            self.original_intensities = np.array(intensities)
            self.current_file = None
        else:
            # Empty spectrum - will be loaded from file
            self.wavenumbers = np.array([])
            self.original_intensities = np.array([])
            self.current_file = None
        
        self.processed_intensities = self.original_intensities.copy()
        
        # File loading utilities
        if FILE_LOADING_AVAILABLE:
            self.spectrum_loader = SpectrumLoader()
        else:
            self.spectrum_loader = None
        
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
        
        # Background auto-preview data
        self.background_options = []  # List of (background_data, description, params) tuples
        self.background_option_lines = []  # List of plot line references
        
        # Fourier analysis data storage
        self.fft_data = None
        self.fft_frequencies = None
        self.fft_magnitude = None
        self.fft_phase = None
        
        self.setup_ui()
        self.initial_plot()
        
    def setup_ui(self):
        """Set up the user interface."""
        self.update_window_title()
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create menu bar
        self.setup_menu_bar()
        main_layout.addWidget(self.menu_bar)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.update_status_bar()
        
        # Content layout (horizontal splitter)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # Left panel - controls (splitter for resizing)
        splitter = QSplitter(Qt.Horizontal)
        content_layout.addWidget(splitter)
        
        # Control panel
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Visualization panel
        viz_panel = self.create_visualization_panel()
        splitter.addWidget(viz_panel)
        
        # Set splitter proportions (30% controls, 70% visualization)
        splitter.setSizes([400, 1200])
        
        # Add status bar at the bottom
        main_layout.addWidget(self.status_bar)
    
    def setup_menu_bar(self):
        """Create the menu bar with file operations."""
        self.menu_bar = QMenuBar()
        
        # File menu
        file_menu = self.menu_bar.addMenu("File")
        
        # Open file action
        open_action = QAction("Open Spectrum File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        if not FILE_LOADING_AVAILABLE:
            open_action.setEnabled(False)
            open_action.setToolTip("File loading not available - install required dependencies")
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Recent files submenu (placeholder for future enhancement)
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self.recent_files_menu.addAction("No recent files").setEnabled(False)
        
        file_menu.addSeparator()
        
        # Save/Export actions
        save_action = QAction("Export Results...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.export_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Close action
        close_action = QAction("Close", self)
        close_action.setShortcut("Ctrl+W")
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        
        # Analysis menu
        analysis_menu = self.menu_bar.addMenu("Analysis")
        
        # Quick actions
        reset_action = QAction("Reset Analysis", self)
        reset_action.triggered.connect(self.reset_spectrum)
        analysis_menu.addAction(reset_action)
        
        clear_peaks_action = QAction("Clear All Peaks", self)
        clear_peaks_action.triggered.connect(self.clear_peaks)
        analysis_menu.addAction(clear_peaks_action)
        
        analysis_menu.addSeparator()
        
        # Background subtraction
        apply_bg_action = QAction("Apply Background Subtraction", self)
        apply_bg_action.triggered.connect(self.apply_background)
        analysis_menu.addAction(apply_bg_action)
        
        # Peak fitting
        fit_peaks_action = QAction("Fit Peaks", self)
        fit_peaks_action.setShortcut("Ctrl+F")
        fit_peaks_action.triggered.connect(self.fit_peaks)
        analysis_menu.addAction(fit_peaks_action)
    
    def update_window_title(self):
        """Update the window title with current file information."""
        base_title = "Spectral Deconvolution & Advanced Analysis"
        if self.current_file:
            file_name = Path(self.current_file).name
            self.setWindowTitle(f"{base_title} - {file_name}")
        else:
            if len(self.wavenumbers) > 0:
                self.setWindowTitle(f"{base_title} - {len(self.wavenumbers)} data points")
            else:
                self.setWindowTitle(f"{base_title} - No data loaded")
    
    def update_status_bar(self):
        """Update the status bar with current spectrum information."""
        if len(self.wavenumbers) > 0:
            wn_range = f"{self.wavenumbers[0]:.1f} - {self.wavenumbers[-1]:.1f} cm⁻¹"
            intensity_range = f"{np.min(self.processed_intensities):.1f} - {np.max(self.processed_intensities):.1f}"
            n_peaks = len(self.get_all_peaks_for_fitting())
            
            status_text = f"Data points: {len(self.wavenumbers)} | Range: {wn_range} | Intensity: {intensity_range} | Peaks: {n_peaks}"
            
            if self.current_file:
                file_size = Path(self.current_file).stat().st_size / 1024  # KB
                status_text += f" | File: {file_size:.1f} KB"
            
            self.status_bar.showMessage(status_text)
        else:
            self.status_bar.showMessage("No spectrum data loaded - use File → Open to load a spectrum file")
    
    def open_file(self):
        """Open a spectrum file and load the data."""
        if not FILE_LOADING_AVAILABLE:
            QMessageBox.warning(self, "Feature Unavailable", 
                              "File loading is not available. Please install required dependencies.")
            return
        
        # Get supported file types for the dialog
        extensions = self.spectrum_loader.get_supported_extensions()
        file_filter = "Spectrum files ({});;All files (*.*)".format(
            " ".join([f"*{ext}" for ext in extensions])
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectrum File",
            "",
            file_filter
        )
        
        if file_path:
            self.load_spectrum_file(file_path)
    
    def load_spectrum_file(self, file_path):
        """Load spectrum data from the specified file."""
        if not FILE_LOADING_AVAILABLE:
            QMessageBox.critical(self, "Error", "File loading utilities not available.")
            return
        
        try:
            # Show loading status
            self.status_bar.showMessage("Loading spectrum file...")
            
            # Load the spectrum data
            wavenumbers, intensities, metadata = self.spectrum_loader.load_spectrum(file_path)
            
            if wavenumbers is None or intensities is None:
                error_msg = metadata.get("error", "Unknown error occurred")
                QMessageBox.critical(self, "Loading Error", 
                                   f"Failed to load spectrum file:\n{error_msg}")
                self.status_bar.showMessage("Ready")
                return
            
            # Update spectrum data
            self.wavenumbers = wavenumbers
            self.original_intensities = intensities
            self.processed_intensities = self.original_intensities.copy()
            self.current_file = file_path
            
            # Reset analysis state
            self.reset_analysis_state()
            
            # Update UI
            self.update_window_title()
            self.update_status_bar()
            self.update_plot()
            
            # Show success message
            n_points = len(wavenumbers)
            wn_range = f"{wavenumbers[0]:.1f} - {wavenumbers[-1]:.1f} cm⁻¹"
            
            QMessageBox.information(self, "File Loaded Successfully", 
                                  f"Loaded spectrum from:\n{Path(file_path).name}\n\n"
                                  f"Data points: {n_points}\n"
                                  f"Wavenumber range: {wn_range}\n"
                                  f"File size: {Path(file_path).stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            QMessageBox.critical(self, "Loading Error", 
                               f"An error occurred while loading the file:\n{str(e)}")
            self.status_bar.showMessage("Ready")
    
    def reset_analysis_state(self):
        """Reset all analysis state when loading new data."""
        # Clear peaks
        self.peaks = np.array([])
        self.manual_peaks = np.array([])
        
        # Clear fitting results
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        
        # Clear background
        self.background = None
        self.background_preview_active = False
        
        # Clear components and analysis results
        self.components = []
        self.pca_result = None
        self.nmf_result = None
        
        # Clear background options
        self.clear_background_options()
        
        # Disable interactive mode
        if self.interactive_mode:
            self.interactive_mode = False
            if hasattr(self, 'interactive_btn'):
                self.interactive_btn.setChecked(False)
                self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            if hasattr(self, 'interactive_status_label'):
                self.interactive_status_label.setText("Interactive mode: OFF")
                self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
            
            # Disconnect mouse click event
            if self.click_connection is not None:
                self.canvas.mpl_disconnect(self.click_connection)
                self.click_connection = None
        
        # Update peak displays
        if hasattr(self, 'peak_count_label'):
            self.update_peak_count_display()
        if hasattr(self, 'peak_list_widget'):
            self.update_peak_list()
        
        # Clear results
        if hasattr(self, 'results_text'):
            self.results_text.clear()
        if hasattr(self, 'results_table'):
            self.results_table.setRowCount(0)
        
        # Clear Fourier analysis data
        self.fft_data = None
        self.fft_frequencies = None
        self.fft_magnitude = None
        self.fft_phase = None
        
        # Reset plot line references
        self._reset_line_references()
        
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
        """Create background subtraction tab using unified components."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        if CENTRALIZED_AVAILABLE:
            # Use unified background controls
            try:
                self.background_controls = BackgroundControlsWidget()
                self.background_controls.parameters_changed.connect(self._trigger_background_update)
                self.background_controls.background_method_changed.connect(self.on_bg_method_changed)
                layout.addWidget(self.background_controls)
            except Exception as e:
                print(f"Warning: Could not create BackgroundControlsWidget: {e}")
                self._create_fallback_background_controls(layout)
        else:
            # Fallback to basic controls
            self._create_fallback_background_controls(layout)
        
        # Action buttons (tool-specific functionality)
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_background)
        button_layout.addWidget(apply_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_background_preview)
        button_layout.addWidget(clear_btn)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_spectrum)
        button_layout.addWidget(reset_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return tab
    
    def _create_fallback_background_controls(self, layout):
        """Create fallback background controls when centralized UI is not available."""
        method_group = QGroupBox("Background Method")
        method_layout = QVBoxLayout(method_group)
        
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["ALS", "Linear", "Polynomial"])
        method_layout.addWidget(self.bg_method_combo)
        
        layout.addWidget(method_group)
        
        # Basic ALS parameters
        als_group = QGroupBox("ALS Parameters")
        als_layout = QFormLayout(als_group)
        
        self.lambda_input = QLineEdit("100000")
        als_layout.addRow("Lambda:", self.lambda_input)
        
        self.p_input = QLineEdit("0.01")
        als_layout.addRow("P:", self.p_input)
        
        self.niter_input = QLineEdit("10")
        als_layout.addRow("Iterations:", self.niter_input)
        
        layout.addWidget(als_group)
    
    def get_fallback_background_parameters(self):
        """Get background parameters from fallback controls."""
        try:
            method = self.bg_method_combo.currentText() if hasattr(self, 'bg_method_combo') else "ALS"
            
            if method == "ALS":
                return {
                    'method': 'ALS',
                    'lambda': float(self.lambda_input.text()) if hasattr(self, 'lambda_input') else 1e5,
                    'p': float(self.p_input.text()) if hasattr(self, 'p_input') else 0.01,
                    'niter': int(self.niter_input.text()) if hasattr(self, 'niter_input') else 10
                }
            else:
                return {
                    'method': method,
                    'order': 2,
                    'start_weight': 1.0,
                    'end_weight': 1.0
                }
        except Exception:
            # Ultimate fallback
            return {'method': 'ALS', 'lambda': 1e5, 'p': 0.01, 'niter': 10}
    
    def get_fallback_peak_parameters(self):
        """Get peak parameters from fallback controls."""
        try:
            if hasattr(self, 'height_slider') and hasattr(self, 'distance_slider') and hasattr(self, 'prominence_slider'):
                return {
                    'model': self.current_model,
                    'height': self.height_slider.value() / 100.0,
                    'distance': self.distance_slider.value(),
                    'prominence': self.prominence_slider.value() / 100.0
                }
            else:
                # Ultimate fallback
                return {
                    'model': 'Gaussian',
                    'height': 0.1,
                    'distance': 10,
                    'prominence': 0.05
                }
        except Exception:
            # Ultimate fallback
            return {
                'model': 'Gaussian',
                'height': 0.1,
                'distance': 10,
                'prominence': 0.05
                         }
    
    def get_peak_parameters(self):
        """Get peak parameters from centralized or fallback controls."""
        if CENTRALIZED_AVAILABLE and hasattr(self, 'peak_controls'):
            return self.peak_controls.get_peak_parameters()
        else:
            return self.get_fallback_peak_parameters()
         
    def create_peak_detection_tab(self):
        """Create peak detection tab with interactive selection using centralized controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Use centralized peak fitting controls if available
        if CENTRALIZED_AVAILABLE:
            try:
                # Use centralized peak fitting controls
                self.peak_controls = PeakFittingControlsWidget()
                self.peak_controls.parameters_changed.connect(self._trigger_peak_update)
                self.peak_controls.model_changed.connect(self.on_centralized_model_changed)
                layout.addWidget(self.peak_controls)
                
                # Action buttons for centralized controls
                centralized_buttons_layout = QHBoxLayout()
                
                clear_btn = QPushButton("Clear All Peaks")
                clear_btn.clicked.connect(self.clear_peaks)
                centralized_buttons_layout.addWidget(clear_btn)
                
                layout.addLayout(centralized_buttons_layout)
                
            except Exception as e:
                print(f"Warning: Could not create PeakFittingControlsWidget: {e}")
                self._create_fallback_peak_controls(layout)
        else:
            # Fallback to manual controls
            self._create_fallback_peak_controls(layout)
        
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
        """Create peak fitting tab with centralized model selection."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection - use centralized controls if available for consistency
        if CENTRALIZED_AVAILABLE and hasattr(self, 'peak_controls'):
            # Model selection is already handled by centralized controls
            model_info = QLabel("Peak model selection is managed in the Peak Detection tab.")
            model_info.setStyleSheet("color: #666; font-style: italic; margin: 10px;")
            layout.addWidget(model_info)
        else:
            # Fallback model selection for when centralized controls aren't available
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
    
    def _create_fallback_peak_controls(self, layout):
        """Create fallback peak detection controls when centralized widget is not available."""
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
        if len(self.wavenumbers) > 0:
            self.update_plot()
        else:
            # Show empty plot with instruction message
            self.setup_empty_plot()
    
    def setup_empty_plot(self):
        """Setup an empty plot with instruction message."""
        # Clear axes
        self.ax_main.clear()
        self.ax_residual.clear()
        
        # Add instruction text
        self.ax_main.text(0.5, 0.5, 
                         'No spectrum data loaded\n\n'
                         'Use File → Open Spectrum File (Ctrl+O)\n'
                         'to load a spectrum file\n\n'
                         'Supported formats: .txt, .csv, .dat, .asc, .spc',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.ax_main.transAxes,
                         fontsize=12,
                         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Spectral Deconvolution & Advanced Analysis - No Data')
        self.ax_main.grid(True, alpha=0.3)
        
        # Empty residuals plot
        self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_residual.set_ylabel('Residuals')
        self.ax_residual.set_title('Residuals - No Data')
        self.ax_residual.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
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
        """Perform background calculation and update plot efficiently using unified widget."""
        try:
            if CENTRALIZED_AVAILABLE and hasattr(self, 'background_controls'):
                # Get parameters from unified background controls widget
                try:
                    params = self.background_controls.get_background_parameters()
                    method = params['method']
                except (AttributeError, Exception) as e:
                    print(f"Warning: Could not get background parameters: {e}")
                    # Fallback to default parameters
                    params = {'method': 'ALS', 'lambda': 1e5, 'p': 0.01, 'niter': 10}
                    method = 'ALS'
                
                if method == "ALS":
                    # Calculate background using ALS with unified parameters
                    self.background = self._get_baseline_fitter().baseline_als(
                        self.original_intensities, 
                        params['lambda'], 
                        params['p'], 
                        params['niter']
                    )
                else:
                    # Handle other methods
                    self.background = self._calculate_background_with_method(method, params)
            else:
                # Fallback method selection
                params = self.get_fallback_background_parameters()
                method = params['method']
                
                if method == "ALS":
                    self.background = self._get_baseline_fitter().baseline_als(
                        self.original_intensities, 
                        params['lambda'], 
                        params['p'], 
                        params['niter']
                    )
                else:
                    # Handle other methods
                    self.background = self._calculate_background_with_method(method, params)
            
            # Set preview flag
            self.background_preview_active = True
            
            # Efficient plot update - only update background line
            self._update_background_line()
            
        except Exception as e:
            print(f"Background preview error: {str(e)}")
    
    def _calculate_background_with_method(self, method, params):
        """Calculate background using specified method and parameters."""
        if method == "Linear":
            return self._calculate_linear_background(
                params.get('start_weight', 1.0), 
                params.get('end_weight', 1.0)
            )
        elif method == "Polynomial":
            return self._calculate_polynomial_background(
                params.get('order', 2), 
                params.get('method', "Least Squares")
            )
        else:
            # Default to ALS
            return self._get_baseline_fitter().baseline_als(self.original_intensities)
    
    def _calculate_linear_background(self, start_weight, end_weight):
        """Calculate linear background between weighted endpoints."""
        try:
            y = self.original_intensities
            start_val = y[0] * start_weight
            end_val = y[-1] * end_weight
            return np.linspace(start_val, end_val, len(y))
        except Exception as e:
            print(f"Linear background calculation error: {str(e)}")
            return np.linspace(y[0], y[-1], len(y))
    
    def _calculate_polynomial_background(self, order, method):
        """Calculate polynomial background fit."""
        try:
            y = self.original_intensities
            x = np.arange(len(y))
            
            # Fit polynomial to the data
            if method == "Robust":
                # Simple robust fitting approach
                coeffs = np.polyfit(x, y, min(order, len(y)-1))
                background = np.polyval(coeffs, x)
                
                # Apply one round of robust reweighting
                residuals = np.abs(y - background)
                weights = 1.0 / (1.0 + residuals / (np.median(residuals) + 1e-10))
                coeffs = np.polyfit(x, y, min(order, len(y)-1), w=weights)
                background = np.polyval(coeffs, x)
            else:
                # Standard least squares
                coeffs = np.polyfit(x, y, min(order, len(y)-1))
                background = np.polyval(coeffs, x)
            
            return background
            
        except Exception as e:
            print(f"Polynomial background calculation error: {str(e)}")
            # Fallback to linear
            return np.linspace(y[0], y[-1], len(y))
    
    def _update_peak_detection(self):
        """Perform live peak detection using centralized or fallback methods."""
        try:
            if CENTRALIZED_AVAILABLE and hasattr(self, 'peak_controls'):
                # Use centralized peak detection
                params = self.peak_controls.get_peak_parameters()
                
                # Use centralized auto_find_peaks function
                self.peaks = auto_find_peaks(
                    self.processed_intensities,
                    height=params['height'],
                    distance=params['distance'],
                    prominence=params['prominence']
                )
                
                # Update current model from centralized controls
                self.current_model = params['model']
                
            else:
                # Fallback to manual slider values
                height_percent = self.height_slider.value()
                distance = self.distance_slider.value()
                prominence_percent = self.prominence_slider.value()
                
                # Calculate thresholds - ensure they are scalars
                max_intensity = float(np.max(self.processed_intensities))
                height_threshold = (height_percent / 100.0) * max_intensity if height_percent > 0 else None
                prominence_threshold = (prominence_percent / 100.0) * max_intensity if prominence_percent > 0 else None
                
                # Ensure distance is an integer
                distance = int(distance) if distance > 0 else None
                
                # Find peaks with proper parameter handling using scipy directly
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
        
        # Update status bar
        if hasattr(self, 'status_bar'):
            self.update_status_bar()
        
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
    def on_bg_method_changed(self, method=None):
        """Handle change in background method."""
        # Clear any active background preview when method changes
        if hasattr(self, 'background_preview_active') and self.background_preview_active:
            self.clear_background_preview()

    # REFACTORED: Removed all background label update methods - handled by unified widget now!

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
        
        # Clear background options
        self.clear_background_options()
        
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
        """Asymmetric Least Squares baseline correction using centralized implementation."""
        return self._get_baseline_fitter().baseline_als(y, lam, p, niter)
    
    # Peak detection methods (fallback controls only)
    def update_height_label(self):
        """Update height slider label (fallback controls only)."""
        if hasattr(self, 'height_slider') and hasattr(self, 'height_label'):
            value = self.height_slider.value()
            self.height_label.setText(f"{value}%")
    
    def update_distance_label(self):
        """Update distance slider label (fallback controls only)."""
        if hasattr(self, 'distance_slider') and hasattr(self, 'distance_label'):
            value = self.distance_slider.value()
            self.distance_label.setText(str(value))
    
    def update_prominence_label(self):
        """Update prominence slider label (fallback controls only)."""
        if hasattr(self, 'prominence_slider') and hasattr(self, 'prominence_label'):
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
        """Handle model selection change (fallback controls)."""
        if hasattr(self, 'model_combo'):
            self.current_model = self.model_combo.currentText()
            if self.fit_result is not None:
                self.update_plot()
    
    def on_centralized_model_changed(self, model_name):
        """Handle model selection change from centralized controls."""
        self.current_model = model_name
        if self.fit_result is not None:
            self.update_plot()
    
    def on_display_changed(self):
        """Handle display option changes."""
        self.show_individual_peaks = self.show_peaks_check.isChecked()
        self.show_components = self.show_components_check.isChecked()
        self.update_plot()
    
    # REFACTORED: Use centralized peak fitting functions instead of duplicating them
    def gaussian(self, x, amp, cen, wid):
        """Gaussian peak function using centralized implementation."""
        return self._get_baseline_fitter().gaussian(x, amp, cen, wid)
    
    def lorentzian(self, x, amp, cen, wid):
        """Lorentzian peak function using centralized implementation."""
        return self._get_baseline_fitter().lorentzian(x, amp, cen, wid)
    
    def pseudo_voigt(self, x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function using centralized implementation."""
        return self._get_baseline_fitter().pseudo_voigt(x, amp, cen, wid, eta)
    
    def _get_baseline_fitter(self):
        """Get baseline correction fitter instance."""
        if not hasattr(self, '_baseline_fitter'):
            self._baseline_fitter = EnhancedPeakFitter()
        return self._baseline_fitter
    
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
            
            # Get current model from centralized or fallback controls
            current_params = self.get_peak_parameters()
            self.current_model = current_params.get('model', self.current_model)
            
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
    
    # Background Auto-Preview Methods
    def generate_background_previews(self):
        """Generate multiple background subtraction options with different parameters."""
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
            
            # Generate backgrounds for each parameter set
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                     '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                     '#F1948A', '#73C6B6', '#AED6F1', '#A9DFBF', '#F9E79F']
            
            for i, (description, method, params) in enumerate(parameter_sets):
                try:
                    background = self._calculate_background_with_params(method, params)
                    if background is not None:
                        # Store background data
                        self.background_options.append((background, description, method, params))
                        
                        # Plot preview line
                        color = colors[i % len(colors)]
                        line, = self.ax_main.plot(
                            self.wavenumbers, background, 
                            color=color, linewidth=1.5, alpha=0.7, 
                            linestyle='--', label=description
                        )
                        self.background_option_lines.append(line)
                        
                except Exception as e:
                    print(f"Failed to generate {description}: {str(e)}")
                    continue
            
            # Update dropdown with options
            self.update_background_options_dropdown()
            
            # Update legend and redraw
            self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.canvas.draw()
            
            # Show info message
            QMessageBox.information(self, "Options Generated", 
                                  f"Generated {len(self.background_options)} background options.\n"
                                  f"Select one from the dropdown and preview it by clicking on the option.")
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Failed to generate background options: {str(e)}")
    
    def _calculate_background_with_params(self, method, params):
        """Calculate background using specified method and parameters."""
        try:
            if method == "ALS":
                lambda_val = params.get("lambda", 1e5)
                p_val = params.get("p", 0.01)
                niter_val = params.get("niter", 10)
                return self.baseline_als(self.original_intensities, lambda_val, p_val, niter_val)
                
            elif method == "Linear":
                start_weight = params.get("start_weight", 1.0)
                end_weight = params.get("end_weight", 1.0)
                start_val = self.original_intensities[0] * start_weight
                end_val = self.original_intensities[-1] * end_weight
                return np.linspace(start_val, end_val, len(self.original_intensities))
                
            elif method == "Polynomial":
                order = params.get("order", 2)
                method_type = params.get("method", "Least Squares")
                x = np.arange(len(self.original_intensities))
                
                if method_type == "Robust":
                    # Robust fitting with iterative reweighting
                    coeffs = np.polyfit(x, self.original_intensities, order)
                    background = np.polyval(coeffs, x)
                    
                    # Apply robust reweighting
                    for _ in range(3):
                        residuals = np.abs(self.original_intensities - background)
                        weights = 1.0 / (1.0 + residuals / np.median(residuals))
                        coeffs = np.polyfit(x, self.original_intensities, order, w=weights)
                        background = np.polyval(coeffs, x)
                    
                    return background
                else:
                    coeffs = np.polyfit(x, self.original_intensities, order)
                    return np.polyval(coeffs, x)
                    
            elif method == "Moving Average":
                window_percent = params.get("window_percent", 10)
                window_type = params.get("window_type", "Gaussian")
                
                window_size = max(int(len(self.original_intensities) * window_percent / 100.0), 3)
                
                if window_type == "Gaussian":
                    from scipy import ndimage
                    sigma = window_size / 4.0
                    return ndimage.gaussian_filter1d(self.original_intensities, sigma=sigma)
                elif window_type == "Uniform":
                    from scipy import ndimage
                    return ndimage.uniform_filter1d(self.original_intensities, size=window_size)
                else:
                    # Default to Gaussian
                    from scipy import ndimage
                    sigma = window_size / 4.0
                    return ndimage.gaussian_filter1d(self.original_intensities, sigma=sigma)
                     
            elif method == "Spline":
                n_knots = params.get("n_knots", 20)
                smoothing = params.get("smoothing", 500)
                return self._calculate_spline_background(n_knots, smoothing)
            
            return None
            
        except Exception as e:
            print(f"Background calculation error for {method}: {str(e)}")
            return None
    
    def _calculate_spline_background_for_subtraction(self, n_knots, smoothing, degree):
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
    
    def _calculate_spline_background(self, n_knots, smoothing):
        """Legacy method - calculate spline-based background using UnivariateSpline."""
        # This method kept for compatibility with auto-preview (though spline removed from auto-preview)
        return self._calculate_spline_background_for_subtraction(n_knots, smoothing, 3)
    
    def _calculate_linear_background(self, start_weight, end_weight):
        """Calculate linear background that fits the baseline, not the data."""
        try:
            from scipy import ndimage
            
            y = self.original_intensities
            
            # Method 1: Use minimum filtering to identify baseline regions
            window_size = max(len(y) // 15, 5)  # Adaptive window size for minimum filtering
            y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
            
            # Method 2: Identify baseline points using percentile approach
            # Take points in the lower percentile as likely baseline
            percentile_threshold = 30  # Use bottom 30% as baseline candidates
            threshold = np.percentile(y, percentile_threshold)
            baseline_mask = y <= threshold
            
            # Combine minimum filtered data with baseline mask
            baseline_indices = np.where(baseline_mask)[0]
            
            if len(baseline_indices) < 2:  # Need at least 2 points for linear fit
                # Fallback: use endpoint-weighted linear fit to minimum filtered data
                start_val = y_min_filtered[0] * start_weight
                end_val = y_min_filtered[-1] * end_weight
            else:
                # Fit line to identified baseline points
                baseline_y = y[baseline_indices]
                
                # Apply weights to endpoints
                if len(baseline_indices) > 0:
                    # Find closest baseline points to start and end
                    start_idx = baseline_indices[0]
                    end_idx = baseline_indices[-1]
                    
                    start_val = y[start_idx] * start_weight
                    end_val = y[end_idx] * end_weight
                else:
                    start_val = y[0] * start_weight
                    end_val = y[-1] * end_weight
            
            # Create linear background
            background = np.linspace(start_val, end_val, len(y))
            
            # Ensure background doesn't exceed data unrealistically
            background = np.minimum(background, y)
            
            return background
            
        except ImportError:
            # Fallback without scipy
            y = self.original_intensities
            
            # Simple approach: weight the endpoints and create linear baseline
            start_val = y[0] * start_weight
            end_val = y[-1] * end_weight
            background = np.linspace(start_val, end_val, len(y))
            
            return np.minimum(background, y)
        
        except Exception as e:
            print(f"Linear background calculation error: {str(e)}")
            # Final fallback
            y = self.original_intensities
            background = np.linspace(y[0], y[-1], len(y))
            return np.minimum(background, y)
    
    def _calculate_polynomial_background(self, poly_order, poly_method):
        """Calculate polynomial background that fits the baseline, not the data."""
        try:
            from scipy import ndimage
            
            y = self.original_intensities
            x = np.arange(len(y))
            
            # Step 1: Use minimum filtering to identify baseline regions
            window_size = max(len(y) // 20, 5)
            y_min_filtered = ndimage.minimum_filter1d(y, size=window_size)
            
            # Step 2: Iterative baseline fitting approach
            current_background = y_min_filtered.copy()
            
            for iteration in range(3):  # Limited iterations
                # Identify points likely to be baseline (below current estimate)
                threshold = np.percentile(y - current_background, 25)  # Use 25th percentile
                mask = (y - current_background) <= threshold
                
                if np.sum(mask) < poly_order + 2:  # Need enough points for polynomial
                    break
                
                # Fit polynomial to identified baseline points
                if poly_method == "Robust":
                    # Robust polynomial fitting with iterative reweighting
                    try:
                        coeffs = np.polyfit(x[mask], y[mask], poly_order)
                        poly_fit = np.polyval(coeffs, x)
                        
                        # Apply robust reweighting for 2 iterations
                        for _ in range(2):
                            residuals = np.abs(y[mask] - np.polyval(coeffs, x[mask]))
                            weights = 1.0 / (1.0 + residuals / (np.median(residuals) + 1e-10))
                            coeffs = np.polyfit(x[mask], y[mask], poly_order, w=weights)
                        
                        current_background = np.polyval(coeffs, x)
                    except:
                        # Fallback to regular fitting
                        coeffs = np.polyfit(x[mask], y[mask], poly_order)
                        current_background = np.polyval(coeffs, x)
                else:
                    # Regular least squares fitting to baseline points
                    coeffs = np.polyfit(x[mask], y[mask], poly_order)
                    current_background = np.polyval(coeffs, x)
                
                # Ensure background doesn't go above data
                current_background = np.minimum(current_background, y)
            
            return current_background
            
        except ImportError:
            # Fallback without scipy
            y = self.original_intensities
            x = np.arange(len(y))
            
            # Simple approach: use lower envelope points
            window_size = max(len(y) // 10, 3)
            y_envelope = np.array([np.min(y[max(0, i-window_size):min(len(y), i+window_size+1)]) 
                                 for i in range(len(y))])
            
            # Fit polynomial to envelope
            coeffs = np.polyfit(x, y_envelope, min(poly_order, len(y)-1))
            background = np.polyval(coeffs, x)
            
            return np.minimum(background, y)
            
        except Exception as e:
            print(f"Polynomial background calculation error: {str(e)}")
            # Final fallback
            y = self.original_intensities
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, min(2, len(y)-1))  # Simple quadratic fallback
            background = np.polyval(coeffs, x)
            return np.minimum(background, y)
    
    def _calculate_moving_average_background(self, window_percent, window_type):
        """Calculate moving average background that fits the baseline, not the data."""
        try:
            from scipy import ndimage
            
            y = self.original_intensities
            
            # Calculate window size as percentage of spectrum length
            window_size = max(int(len(y) * window_percent / 100.0), 3)
            
            # Step 1: Apply minimum filtering to get baseline candidate
            min_window = max(window_size // 2, 3)
            y_min_filtered = ndimage.minimum_filter1d(y, size=min_window)
            
            # Step 2: Apply the specified moving average filter to the minimum filtered data
            if window_type == "Uniform":
                background = ndimage.uniform_filter1d(y_min_filtered, size=window_size)
            elif window_type == "Gaussian":
                sigma = window_size / 4.0  # Standard deviation
                background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
            elif window_type in ["Hann", "Hamming"]:
                # Apply windowed convolution to minimum filtered data
                if window_type == "Hann":
                    window = np.hanning(window_size)
                else:  # Hamming
                    window = np.hamming(window_size)
                
                window = window / np.sum(window)  # Normalize
                background = np.convolve(y_min_filtered, window, mode='same')
            else:
                # Default to Gaussian
                sigma = window_size / 4.0
                background = ndimage.gaussian_filter1d(y_min_filtered, sigma=sigma)
            
            # Step 3: Additional constraint - ensure it stays below original data
            background = np.minimum(background, y)
            
            # Step 4: Optional second pass for better baseline fitting
            # Create mask for points close to the current background estimate
            tolerance = np.std(y - background) * 0.5
            baseline_mask = (y - background) <= tolerance
            
            if np.sum(baseline_mask) > window_size:
                # Apply the filter again, but only to baseline regions
                baseline_points = y.copy()
                baseline_points[~baseline_mask] = background[~baseline_mask]  # Replace peaks with current estimate
                
                if window_type == "Uniform":
                    refined_background = ndimage.uniform_filter1d(baseline_points, size=window_size)
                elif window_type == "Gaussian":
                    sigma = window_size / 4.0
                    refined_background = ndimage.gaussian_filter1d(baseline_points, sigma=sigma)
                else:
                    refined_background = background  # Keep original if windowed
                
                background = np.minimum(refined_background, y)
            
            return background
            
        except ImportError:
            # Fallback without scipy
            y = self.original_intensities
            window_size = max(int(len(y) * window_percent / 100.0), 3)
            
            # Simple moving minimum approach
            background = np.array([np.min(y[max(0, i-window_size//2):min(len(y), i+window_size//2+1)]) 
                                 for i in range(len(y))])
            
            # Simple smoothing
            for _ in range(2):  # 2 passes of smoothing
                smoothed = background.copy()
                for i in range(1, len(background)-1):
                    smoothed[i] = (background[i-1] + background[i] + background[i+1]) / 3
                background = smoothed
            
            return np.minimum(background, y)
            
        except Exception as e:
            print(f"Moving average background calculation error: {str(e)}")
            # Final fallback
            y = self.original_intensities
            return np.full_like(y, np.min(y))

    def update_background_options_dropdown(self):
        """Update the background options dropdown."""
        self.bg_options_combo.clear()
        self.bg_options_combo.addItem("None - Select an option")
        
        for i, (_, description, _, _) in enumerate(self.background_options):
            self.bg_options_combo.addItem(f"{i+1}. {description}")
    
    def on_bg_option_selected(self):
        """Handle selection of a background option."""
        selected_text = self.bg_options_combo.currentText()
        
        if selected_text.startswith("None") or not self.background_options:
            return
        
        try:
            # Extract option index
            option_index = int(selected_text.split('.')[0]) - 1
            
            if 0 <= option_index < len(self.background_options):
                # Highlight the selected option
                self._highlight_selected_background_option(option_index)
                
        except (ValueError, IndexError):
            pass
    
    def _highlight_selected_background_option(self, option_index):
        """Highlight the selected background option on the plot."""
        # Reset all line styles to normal
        for line in self.background_option_lines:
            try:
                line.set_linewidth(1.5)
                line.set_alpha(0.7)
            except:
                pass
        
        # Highlight selected option
        if 0 <= option_index < len(self.background_option_lines):
            try:
                selected_line = self.background_option_lines[option_index]
                selected_line.set_linewidth(3.0)
                selected_line.set_alpha(1.0)
                
                # Update canvas
                self.canvas.draw_idle()
                
                # Show option details
                _, description, method, params = self.background_options[option_index]
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                
                self.results_text.setPlainText(
                    f"Selected Background Option:\n"
                    f"Description: {description}\n"
                    f"Method: {method}\n"
                    f"Parameters: {param_str}\n\n"
                    f"Click 'Apply Selected' to use this background subtraction."
                )
                
            except Exception as e:
                print(f"Error highlighting option: {str(e)}")
    
    def apply_selected_background_option(self):
        """Apply the selected background option."""
        selected_text = self.bg_options_combo.currentText()
        
        if selected_text.startswith("None") or not self.background_options:
            QMessageBox.warning(self, "No Selection", "Please select a background option first.")
            return
        
        try:
            # Extract option index
            option_index = int(selected_text.split('.')[0]) - 1
            
            if 0 <= option_index < len(self.background_options):
                background_data, description, method, params = self.background_options[option_index]
                
                # Apply the background subtraction
                self.background = background_data.copy()
                self.processed_intensities = self.original_intensities - self.background
                self.background_preview_active = False
                
                # Update the manual parameter controls to match this selection
                self._update_manual_controls_from_params(method, params)
                
                # Clear the options and update plot
                self.clear_background_options()
                self.update_plot()
                
                # Show confirmation
                QMessageBox.information(self, "Background Applied", 
                                      f"Applied background subtraction:\n{description}\n\n"
                                      f"Method: {method}\n"
                                      f"Manual controls have been updated to match.")
                
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Selection Error", f"Invalid selection: {str(e)}")
    
    def _update_manual_controls_from_params(self, method, params):
        """Update manual parameter controls to match the selected option."""
        try:
            # Update method combo box
            method_mapping = {
                "ALS": "ALS (Asymmetric Least Squares)",
                "Linear": "Linear",
                "Polynomial": "Polynomial",
                "Moving Average": "Moving Average",
                "Spline": "Spline"
            }
            
            if method in method_mapping:
                combo_text = method_mapping[method]
                index = self.bg_method_combo.findText(combo_text)
                if index >= 0:
                    self.bg_method_combo.setCurrentIndex(index)
                    self.on_bg_method_changed()  # Update visibility of parameter widgets
            
            # Update method-specific parameters
            if method == "ALS":
                if "lambda" in params:
                    lambda_val = params["lambda"]
                    log_val = int(np.log10(lambda_val))
                    self.lambda_slider.setValue(max(3, min(7, log_val)))
                    self.update_lambda_label()
                
                if "p" in params:
                    p_val = params["p"]
                    slider_val = int(p_val * 1000)
                    self.p_slider.setValue(max(1, min(50, slider_val)))
                    self.update_p_label()
                
                if "niter" in params:
                    niter_val = params["niter"]
                    self.niter_slider.setValue(max(5, min(30, niter_val)))
                    self.update_niter_label()
                    
            elif method == "Linear":
                if "start_weight" in params:
                    start_val = int(params["start_weight"] * 10)
                    self.start_weight_slider.setValue(max(1, min(20, start_val)))
                    self.update_start_weight_label()
                
                if "end_weight" in params:
                    end_val = int(params["end_weight"] * 10)
                    self.end_weight_slider.setValue(max(1, min(20, end_val)))
                    self.update_end_weight_label()
                    
            elif method == "Polynomial":
                if "order" in params:
                    order_val = params["order"]
                    self.poly_order_slider.setValue(max(1, min(6, order_val)))
                    self.update_poly_order_label()
                
                if "method" in params:
                    method_type = params["method"]
                    index = self.poly_method_combo.findText(method_type)
                    if index >= 0:
                        self.poly_method_combo.setCurrentIndex(index)
                        
            elif method == "Moving Average":
                if "window_percent" in params:
                    window_val = params["window_percent"]
                    self.window_size_slider.setValue(max(1, min(50, window_val)))
                    self.update_window_size_label()
                
                if "window_type" in params:
                    window_type = params["window_type"]
                    index = self.window_type_combo.findText(window_type)
                    if index >= 0:
                        self.window_type_combo.setCurrentIndex(index)
            
        except Exception as e:
            print(f"Error updating manual controls: {str(e)}")
    
    def clear_background_options(self):
        """Clear all background options and their preview lines."""
        try:
            # Clear plot lines
            for line in self.background_option_lines:
                try:
                    line.remove()
                except:
                    pass
            
            # Clear data
            self.background_options.clear()
            self.background_option_lines.clear()
            
            # Reset dropdown
            self.bg_options_combo.clear()
            self.bg_options_combo.addItem("None - Generate options first")
            
            # Clear results text if it was showing option info
            if hasattr(self, 'results_text') and "Selected Background Option:" in self.results_text.toPlainText():
                self.results_text.clear()
            
            # Update plot
            if hasattr(self, 'ax_main'):
                self.ax_main.legend()
                self.canvas.draw_idle()
                
        except Exception as e:
            print(f"Error clearing background options: {str(e)}")

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
def launch_spectral_deconvolution(parent, wavenumbers=None, intensities=None):
    """Launch the spectral deconvolution window."""
    dialog = SpectralDeconvolutionQt6(parent, wavenumbers, intensities)
    dialog.exec()

# Standalone launch function
def launch_standalone_spectral_deconvolution():
    """Launch the spectral deconvolution window as a standalone application."""
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
    
    # Create a minimal parent widget
    from PySide6.QtWidgets import QWidget
    parent = QWidget()
    parent.hide()
    
    dialog = SpectralDeconvolutionQt6(parent)
    dialog.show()
    
    if app:
        sys.exit(app.exec())

# Allow running as standalone script
if __name__ == "__main__":
    launch_standalone_spectral_deconvolution() 