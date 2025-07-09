#!/usr/bin/env python3
"""
RamanLab Qt6 - Mixture Analysis GUI
Enhanced spectral mixture analysis with iterative decomposition
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Fix matplotlib backend for PySide6
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Import matplotlib configuration as per user rules
try:
    from polarization_ui.matplotlib_config import (
        CompactNavigationToolbar as NavigationToolbar, 
        apply_theme, 
        configure_compact_ui
    )
except ImportError:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    def apply_theme(theme):
        pass
    def configure_compact_ui():
        pass

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSlider, QCheckBox, QComboBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QSpinBox, QDoubleSpinBox, QFormLayout, QListWidget, QListWidgetItem,
    QFrame, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QAction

# Import the mixture analysis engine
from raman_mixture_analysis import RamanMixtureAnalyzer

# File loading utilities
try:
    from utils.file_loaders import load_spectrum_file
    FILE_LOADING_AVAILABLE = True
except ImportError:
    FILE_LOADING_AVAILABLE = False

# State management
try:
    from core.universal_state_manager import register_module, save_module_state, load_module_state
    STATE_MANAGEMENT_AVAILABLE = True
except ImportError:
    STATE_MANAGEMENT_AVAILABLE = False


class MixtureAnalysisWorker(QThread):
    """Worker thread for running mixture analysis without blocking UI."""
    
    # Signals
    progress_updated = Signal(str, int)  # message, percentage
    analysis_completed = Signal(dict)  # results
    analysis_failed = Signal(str)  # error message
    
    def __init__(self, analyzer, wavenumbers, spectrum, max_components, analysis_name=None, constrained_components=None):
        super().__init__()
        self.analyzer = analyzer
        self.wavenumbers = wavenumbers
        self.spectrum = spectrum
        self.max_components = max_components
        self.analysis_name = analysis_name
        self.constrained_components = constrained_components or []
        
    def run(self):
        """Run the mixture analysis in background thread."""
        try:
            if self.constrained_components:
                self.progress_updated.emit(f"Starting constrained analysis with {len(self.constrained_components)} known components...", 10)
            else:
                self.progress_updated.emit("Starting mixture analysis...", 10)
            
            # Run the analysis with constraints
            results = self.analyzer.analyze(
                wavenumbers=self.wavenumbers,
                spectrum=self.spectrum,
                max_components=self.max_components,
                plot_results=False,  # Don't plot in worker thread
                save_to_database=True,
                analysis_name=self.analysis_name,
                constrained_components=self.constrained_components
            )
            
            self.progress_updated.emit("Analysis completed!", 100)
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.analysis_failed.emit(str(e))


class ParametersDialog(QDialog):
    """Dialog for adjusting analysis parameters."""
    
    def __init__(self, parent=None, current_params=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Parameters")
        self.setModal(True)
        self.resize(400, 600)
        
        # Default parameters
        self.params = current_params or {
            'noise_threshold': 0.02,
            'max_iterations': 10,
            'correlation_threshold': 0.7,
            'convergence_threshold': 0.95,
            'reduced_chi_squared_target': 1.2,
            'residual_std_threshold': 0.025,
            'bootstrap_samples': 100,
            'monte_carlo_samples': 50
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the parameters dialog UI."""
        layout = QVBoxLayout(self)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QFormLayout(scroll_widget)
        
        # Analysis parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QFormLayout(analysis_group)
        
        self.noise_threshold = QDoubleSpinBox()
        self.noise_threshold.setRange(0.001, 1.0)
        self.noise_threshold.setSingleStep(0.001)
        self.noise_threshold.setDecimals(3)
        self.noise_threshold.setValue(self.params['noise_threshold'])
        analysis_layout.addRow("Noise Threshold:", self.noise_threshold)
        
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 50)
        self.max_iterations.setValue(self.params['max_iterations'])
        analysis_layout.addRow("Max Iterations:", self.max_iterations)
        
        self.correlation_threshold = QDoubleSpinBox()
        self.correlation_threshold.setRange(0.1, 1.0)
        self.correlation_threshold.setSingleStep(0.01)
        self.correlation_threshold.setDecimals(2)
        self.correlation_threshold.setValue(self.params['correlation_threshold'])
        analysis_layout.addRow("Correlation Threshold:", self.correlation_threshold)
        
        self.convergence_threshold = QDoubleSpinBox()
        self.convergence_threshold.setRange(0.5, 1.0)
        self.convergence_threshold.setSingleStep(0.01)
        self.convergence_threshold.setDecimals(2)
        self.convergence_threshold.setValue(self.params['convergence_threshold'])
        analysis_layout.addRow("Convergence Threshold (R²):", self.convergence_threshold)
        
        scroll_layout.addWidget(analysis_group)
        
        # Advanced parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QFormLayout(advanced_group)
        
        self.chi_squared_target = QDoubleSpinBox()
        self.chi_squared_target.setRange(0.5, 5.0)
        self.chi_squared_target.setSingleStep(0.1)
        self.chi_squared_target.setDecimals(1)
        self.chi_squared_target.setValue(self.params['reduced_chi_squared_target'])
        advanced_layout.addRow("Reduced χ² Target:", self.chi_squared_target)
        
        self.residual_threshold = QDoubleSpinBox()
        self.residual_threshold.setRange(0.001, 0.1)
        self.residual_threshold.setSingleStep(0.001)
        self.residual_threshold.setDecimals(3)
        self.residual_threshold.setValue(self.params['residual_std_threshold'])
        advanced_layout.addRow("Residual Std Threshold:", self.residual_threshold)
        
        scroll_layout.addWidget(advanced_group)
        
        # Uncertainty analysis parameters
        uncertainty_group = QGroupBox("Uncertainty Analysis")
        uncertainty_layout = QFormLayout(uncertainty_group)
        
        self.bootstrap_samples = QSpinBox()
        self.bootstrap_samples.setRange(10, 1000)
        self.bootstrap_samples.setValue(self.params['bootstrap_samples'])
        uncertainty_layout.addRow("Bootstrap Samples:", self.bootstrap_samples)
        
        self.monte_carlo_samples = QSpinBox()
        self.monte_carlo_samples.setRange(10, 500)
        self.monte_carlo_samples.setValue(self.params['monte_carlo_samples'])
        uncertainty_layout.addRow("Monte Carlo Samples:", self.monte_carlo_samples)
        
        scroll_layout.addWidget(uncertainty_group)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_parameters(self):
        """Get the current parameter values."""
        return {
            'noise_threshold': self.noise_threshold.value(),
            'max_iterations': self.max_iterations.value(),
            'correlation_threshold': self.correlation_threshold.value(),
            'convergence_threshold': self.convergence_threshold.value(),
            'reduced_chi_squared_target': self.chi_squared_target.value(),
            'residual_std_threshold': self.residual_threshold.value(),
            'bootstrap_samples': self.bootstrap_samples.value(),
            'monte_carlo_samples': self.monte_carlo_samples.value()
        }


class RamanMixtureAnalysisQt6(QMainWindow):
    """Qt6 GUI for RamanLab Mixture Analysis."""
    
    def __init__(self, parent_app=None):
        super().__init__()
        
        # Apply matplotlib theme
        apply_theme('compact')
        configure_compact_ui()
        
        # Store parent reference
        self.parent_app = parent_app
        self.launched_from_main_app = parent_app is not None
        
        # Initialize analyzer in fast mode by default for interactive use
        self.analyzer = RamanMixtureAnalyzer(fast_mode=True)
        
        # Data storage
        self.current_wavenumbers = None
        self.current_spectrum = None
        self.current_results = None
        self.analysis_history = []
        
        # Constrained analysis support
        self.constrained_components = []  # List of known components from search results
        
        # Analysis worker
        self.analysis_worker = None
        
        # UI initialization flags
        self._ui_ready = False
        self._pending_constraints = []  # Store constraints added before UI is ready
        
        # UI setup
        self.setWindowTitle("RamanLab - Mixture Analysis")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        # Mark UI as ready after setup
        self._ui_ready = True
        
        # Process any pending constraints
        self._process_pending_constraints()
        
        # State management
        if STATE_MANAGEMENT_AVAILABLE:
            self.setup_state_management()
        
        # Show welcome message
        self.show_welcome_message()
        
        # Defer initial canvas drawing until window is shown
        self._initial_draw_pending = True
        
    def on_fast_mode_toggled(self, checked):
        """Handle fast mode toggle changes."""
        try:
            # Reinitialize analyzer with new mode
            self.analyzer = RamanMixtureAnalyzer(fast_mode=checked)
            
            # Update status
            mode_text = "⚡ Fast Mode" if checked else "🔬 Research Mode"
            time_text = "2-5 minutes" if checked else "20-50 minutes"
            
            if hasattr(self, 'status_label'):
                current_text = self.status_label.text()
                if "ready" in current_text.lower():
                    self.status_label.setText(f"✅ Spectrum ready for {mode_text} analysis ({time_text})")
                    
            # Update status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Analysis mode changed to: {mode_text} (Expected time: {time_text})")
                
        except Exception as e:
            print(f"Error switching analysis mode: {e}")
            # Revert checkbox if there was an error
            self.fast_mode_checkbox.setChecked(not checked)
        
    def set_spectrum_from_main_app(self, wavenumbers, intensities, spectrum_name="Current Spectrum from Main App"):
        """Set spectrum data when launched from main RamanLab app."""
        if wavenumbers is None or intensities is None:
            return
            
        # Store the data
        self.current_wavenumbers = wavenumbers.copy()
        self.current_spectrum = intensities.copy()
        
        # Update UI to reflect that spectrum is loaded from main app
        self.file_label.setText(f"✅ Loaded from main app: {spectrum_name}")
        self.status_label.setText(f"✅ Spectrum ready: {len(wavenumbers)} points")
        self.run_btn.setEnabled(True)
        
        # Try to plot the spectrum (will work after canvas is ready)
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.plot_original_spectrum()
        # If canvas not ready, plotting will happen in showEvent
        
    def setup_ui(self):
        """Setup the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - controls
        self.setup_control_panel(main_splitter)
        
        # Right panel - visualization
        self.setup_visualization_panel(main_splitter)
        
        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([480, 1120])
        
    def setup_control_panel(self, parent):
        """Setup the control panel."""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_frame.setMaximumWidth(500)
        control_layout = QVBoxLayout(control_frame)
        
        # File operations
        file_group = QGroupBox("Spectrum Status")
        file_layout = QVBoxLayout(file_group)
        
        # Only show load button if not launched from main app
        if not self.launched_from_main_app:
            load_btn = QPushButton("📁 Load Spectrum")
            load_btn.clicked.connect(self.load_spectrum_file)
            file_layout.addWidget(load_btn)
            self.file_label = QLabel("No file loaded")
        else:
            self.file_label = QLabel("⏳ Waiting for spectrum from main app...")
            
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        
        control_layout.addWidget(file_group)
        
        # Analysis parameters
        params_group = QGroupBox("Analysis Settings")
        params_layout = QFormLayout(params_group)
        
        self.max_components = QSpinBox()
        self.max_components.setRange(1, 10)
        self.max_components.setValue(5)
        params_layout.addRow("Max Components:", self.max_components)
        
        # Fast mode toggle
        self.fast_mode_checkbox = QCheckBox("Fast Mode (2-5 min)")
        self.fast_mode_checkbox.setChecked(True)  # Default to fast mode
        self.fast_mode_checkbox.setToolTip(
            "Fast Mode: 2-5 minute analysis with optimized parameters\n"
            "Research Mode: 20-50 minute analysis with full statistical rigor"
        )
        self.fast_mode_checkbox.toggled.connect(self.on_fast_mode_toggled)
        params_layout.addRow("", self.fast_mode_checkbox)
        
        params_btn = QPushButton("⚙️ Advanced Parameters")
        params_btn.clicked.connect(self.show_parameters_dialog)
        params_layout.addRow(params_btn)
        
        control_layout.addWidget(params_group)
        
        # Analysis controls
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.run_btn = QPushButton("🔬 Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        analysis_layout.addWidget(self.run_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        analysis_layout.addWidget(self.progress_bar)
        
        if self.launched_from_main_app:
            self.status_label = QLabel("⏳ Waiting for spectrum from main app...")
        else:
            self.status_label = QLabel("📁 Load a spectrum file to begin analysis")
        self.status_label.setWordWrap(True)
        analysis_layout.addWidget(self.status_label)
        
        # Constrained components management
        constraints_group = QGroupBox("Known Components (Constraints)")
        constraints_layout = QVBoxLayout(constraints_group)
        
        # Information label
        info_label = QLabel("Add known dominant minerals from search results to guide analysis:")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        constraints_layout.addWidget(info_label)
        
        # Constraints list
        self.constraints_list = QListWidget()
        self.constraints_list.setMaximumHeight(120)
        self.constraints_list.setToolTip("Known components that will be fitted first before searching for additional phases")
        constraints_layout.addWidget(self.constraints_list)
        
        # Constraints buttons
        constraints_btn_layout = QHBoxLayout()
        
        self.add_constraint_btn = QPushButton("➕ Add Known Mineral")
        self.add_constraint_btn.clicked.connect(self.add_constraint_dialog)
        self.add_constraint_btn.setToolTip("Manually add a known mineral component")
        constraints_btn_layout.addWidget(self.add_constraint_btn)
        
        self.remove_constraint_btn = QPushButton("➖ Remove")
        self.remove_constraint_btn.clicked.connect(self.remove_selected_constraint)
        self.remove_constraint_btn.setEnabled(False)
        constraints_btn_layout.addWidget(self.remove_constraint_btn)
        
        self.clear_constraints_btn = QPushButton("🗑️ Clear All")
        self.clear_constraints_btn.clicked.connect(self.clear_all_constraints)
        self.clear_constraints_btn.setEnabled(False)
        constraints_btn_layout.addWidget(self.clear_constraints_btn)
        
        constraints_layout.addLayout(constraints_btn_layout)
        
        # Enable constraint buttons based on list state
        self.constraints_list.itemSelectionChanged.connect(self.on_constraint_selection_changed)
        
        control_layout.addWidget(analysis_group)
        
        # Results summary
        self.results_group = QGroupBox("Results Summary")
        self.results_layout = QVBoxLayout(self.results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        self.results_layout.addWidget(self.results_text)
        
        export_btn = QPushButton("💾 Export Results")
        export_btn.clicked.connect(self.export_results)
        self.results_layout.addWidget(export_btn)
        
        control_layout.addWidget(self.results_group)
        self.results_group.setVisible(False)
        
        # Components list
        self.components_group = QGroupBox("Identified Components")
        self.components_layout = QVBoxLayout(self.components_group)
        
        self.components_list = QListWidget()
        self.components_list.itemClicked.connect(self.on_component_selected)
        self.components_layout.addWidget(self.components_list)
        
        control_layout.addWidget(self.components_group)
        self.components_group.setVisible(False)
        
        control_layout.addStretch()
        parent.addWidget(control_frame)
        
    def setup_visualization_panel(self, parent):
        """Setup the visualization panel."""
        viz_frame = QFrame()
        viz_frame.setFrameStyle(QFrame.StyledPanel)
        viz_layout = QVBoxLayout(viz_frame)
        
        # Store references for deferred initialization
        self.viz_frame = viz_frame
        self.viz_layout = viz_layout
        
        # Create placeholder widget
        placeholder_label = QLabel("Initializing plots...")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("font-size: 14px; color: #666; padding: 50px;")
        viz_layout.addWidget(placeholder_label)
        
        parent.addWidget(viz_frame)
        
    def setup_empty_plots(self):
        """Setup empty plots for initial display."""
        self.figure.clear()
        
        # Main spectrum plot
        self.ax_main = self.figure.add_subplot(2, 2, 1)
        self.ax_main.set_title("Original Spectrum")
        self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_main.set_ylabel("Intensity")
        self.ax_main.grid(True, alpha=0.3)
        
        # Components plot
        self.ax_components = self.figure.add_subplot(2, 2, 2)
        self.ax_components.set_title("Identified Components")
        self.ax_components.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_components.set_ylabel("Intensity")
        self.ax_components.grid(True, alpha=0.3)
        
        # Fit quality plot
        self.ax_fit = self.figure.add_subplot(2, 2, 3)
        self.ax_fit.set_title("Fit Quality")
        self.ax_fit.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_fit.set_ylabel("Intensity")
        self.ax_fit.grid(True, alpha=0.3)
        
        # Residuals plot
        self.ax_residuals = self.figure.add_subplot(2, 2, 4)
        self.ax_residuals.set_title("Residuals")
        self.ax_residuals.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_residuals.set_ylabel("Residual")
        self.ax_residuals.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        
        # Draw the canvas (only called after canvas is properly created)
        try:
            if hasattr(self, 'canvas') and self.canvas is not None and hasattr(self.canvas, 'draw'):
                self.canvas.draw()
            else:
                print("Warning: Canvas not available for empty plots setup")
        except RuntimeError as e:
            print(f"Warning: Could not draw empty plots - {e}")
        
    def setup_menu_bar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Spectrum...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_spectrum_file)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        
        run_action = QAction("Run Analysis", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self.run_analysis)
        analysis_menu.addAction(run_action)
        
        params_action = QAction("Parameters...", self)
        params_action.triggered.connect(self.show_parameters_dialog)
        analysis_menu.addAction(params_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_action = QAction("Refresh Plots", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_plots)
        view_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Load a spectrum to begin analysis")
        
    def setup_state_management(self):
        """Setup state management for session persistence."""
        try:
            register_module('mixture_analysis', self)
            print("✅ Mixture analysis state management enabled")
        except Exception as e:
            print(f"Warning: Could not enable state management: {e}")
            
    def show_welcome_message(self):
        """Show welcome message in results area."""
        welcome_text = """
        Welcome to RamanLab Mixture Analysis!
        
        This tool performs advanced spectral mixture analysis using:
        • Iterative spectral decomposition
        • Multiple peak profile fitting
        • Uncertainty analysis via bootstrap resampling
        • Integration with RamanLab database
        
        To get started:
        1. Load a Raman spectrum
        2. Adjust analysis parameters if needed
        3. Run the analysis
        """
        self.results_text.setText(welcome_text)
        self.results_group.setVisible(True)
        
    def load_spectrum_file(self):
        """Load a spectrum file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Spectrum File",
            "",
            "Text Files (*.txt *.dat *.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Try to load the file using different methods
            wavenumbers = None
            intensities = None
            
            # Method 1: Try RamanLab file loader if available
            if FILE_LOADING_AVAILABLE:
                try:
                    data = load_spectrum_file(file_path)
                    if isinstance(data, dict):
                        wavenumbers = np.array(data['wavenumbers'])
                        intensities = np.array(data['intensities'])
                    else:
                        raise ValueError("Unexpected data format from file loader")
                except Exception as e:
                    print(f"RamanLab file loader failed: {e}")
                    # Fall through to numpy loading
            
            # Method 2: Try numpy loadtxt
            if wavenumbers is None or intensities is None:
                try:
                    data = np.loadtxt(file_path, delimiter=None)
                    if len(data.shape) == 1:
                        raise ValueError("File contains only one column - need wavenumbers and intensities")
                    elif data.shape[1] >= 2:
                        wavenumbers = data[:, 0]
                        intensities = data[:, 1]
                    else:
                        raise ValueError("File must contain at least 2 columns (wavenumbers, intensities)")
                except Exception as e:
                    print(f"Numpy loadtxt failed: {e}")
                    # Try with different delimiters
                    for delimiter in ['\t', ',', ' ', ';']:
                        try:
                            data = np.loadtxt(file_path, delimiter=delimiter)
                            if len(data.shape) >= 2 and data.shape[1] >= 2:
                                wavenumbers = data[:, 0]
                                intensities = data[:, 1]
                                break
                        except:
                            continue
            
            # Check if we successfully loaded data
            if wavenumbers is None or intensities is None:
                raise ValueError("Could not parse file. Please ensure it contains two columns: wavenumbers and intensities")
            
            # Validate data
            if len(wavenumbers) == 0 or len(intensities) == 0:
                raise ValueError("No valid data found in file")
            
            if len(wavenumbers) != len(intensities):
                raise ValueError(f"Wavenumber and intensity arrays have different lengths: {len(wavenumbers)} vs {len(intensities)}")
            
            # Store data
            self.current_wavenumbers = np.array(wavenumbers)
            self.current_spectrum = np.array(intensities)
            
            # Update UI
            self.file_label.setText(f"✅ Loaded: {Path(file_path).name}")
            self.status_label.setText(f"✅ Spectrum loaded: {len(wavenumbers)} points")
            self.run_btn.setEnabled(True)
            
            # Plot the spectrum
            self.plot_original_spectrum()
            
            # Update status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Loaded spectrum: {Path(file_path).name}")
            
        except Exception as e:
            error_msg = f"Could not load file: {Path(file_path).name}\n\nError: {str(e)}\n\nPlease ensure your file:\n• Contains two columns (wavenumbers, intensities)\n• Uses tab, comma, or space delimiters\n• Has numeric data only"
            QMessageBox.critical(self, "Error Loading File", error_msg)
            
    def plot_original_spectrum(self):
        """Plot the original spectrum."""
        if self.current_wavenumbers is None or self.current_spectrum is None:
            return
            
        # Check if canvas and axes are properly initialized (deferred initialization)
        if not hasattr(self, 'canvas') or self.canvas is None:
            print("Warning: Canvas not yet initialized, deferring original spectrum plot")
            return
            
        if not hasattr(self, 'ax_main') or self.ax_main is None:
            print("Warning: Main axis not yet initialized, deferring original spectrum plot")
            return
            
        try:
            self.ax_main.clear()
            self.ax_main.plot(self.current_wavenumbers, self.current_spectrum, 'b-', linewidth=1.5)
            self.ax_main.set_title("Original Spectrum")
            self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax_main.set_ylabel("Intensity")
            self.ax_main.grid(True, alpha=0.3)
            
            # Safely draw the canvas
            if hasattr(self.canvas, 'draw') and callable(self.canvas.draw):
                self.canvas.draw()
            else:
                print("Warning: Canvas draw method not available")
                
        except RuntimeError as e:
            print(f"Warning: Could not plot original spectrum - {e}")
            # Don't raise the error, just continue
        
    def update_query_plot(self):
        """Update the plot with the loaded query spectrum."""
        # Check if query data is available
        if not hasattr(self, 'query_wavenumbers') or not hasattr(self, 'query_intensities'):
            return
            
        if self.query_wavenumbers is None or self.query_intensities is None:
            return
            
        # Check if canvas and axes are properly initialized (deferred initialization)
        if not hasattr(self, 'canvas') or self.canvas is None:
            print("Warning: Canvas not yet initialized, will plot when ready")
            # Store query data for later plotting
            self.current_wavenumbers = self.query_wavenumbers.copy()
            self.current_spectrum = self.query_intensities.copy()
            return
            
        if not hasattr(self, 'ax_main') or self.ax_main is None:
            print("Warning: Main axis not yet initialized, will plot when ready")
            return
            
        # Set the query data as current spectrum data
        self.current_wavenumbers = self.query_wavenumbers.copy()
        self.current_spectrum = self.query_intensities.copy()
        
        try:
            # Update the plot
            self.ax_main.clear()
            self.ax_main.plot(self.current_wavenumbers, self.current_spectrum, 'b-', linewidth=1.5)
            self.ax_main.set_title("Query Spectrum (from Search)")
            self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
            self.ax_main.set_ylabel("Intensity")
            self.ax_main.grid(True, alpha=0.3)
            
            # Safely draw the canvas
            if hasattr(self.canvas, 'draw') and callable(self.canvas.draw):
                self.canvas.draw()
            else:
                print("Warning: Canvas draw method not available")
                
        except RuntimeError as e:
            print(f"Warning: Could not update plot - {e}")
            # Don't raise the error, just continue
        
        # Update file label
        if hasattr(self, 'file_label') and self.file_label is not None:
            self.file_label.setText("Query spectrum loaded from search results")
        
        # Enable analysis controls
        if hasattr(self, 'run_btn') and self.run_btn is not None:
            self.run_btn.setEnabled(True)
        
        # Update status
        if hasattr(self, 'status_label') and self.status_label is not None:
            data_points = len(self.current_spectrum)
            wavenumber_range = f"{self.current_wavenumbers[0]:.1f}-{self.current_wavenumbers[-1]:.1f}"
            self.status_label.setText(f"✅ Query spectrum ready for analysis\n{data_points} data points, {wavenumber_range} cm⁻¹")
        
    def show_parameters_dialog(self):
        """Show the parameters dialog."""
        # Get current parameters from analyzer
        current_params = {
            'noise_threshold': self.analyzer.noise_threshold,
            'max_iterations': self.analyzer.max_iterations,
            'correlation_threshold': self.analyzer.correlation_threshold,
            'convergence_threshold': self.analyzer.convergence_threshold,
            'reduced_chi_squared_target': self.analyzer.reduced_chi_squared_target,
            'residual_std_threshold': self.analyzer.residual_std_threshold,
            'bootstrap_samples': self.analyzer.bootstrap_samples,
            'monte_carlo_samples': self.analyzer.monte_carlo_samples
        }
        
        dialog = ParametersDialog(self, current_params)
        if dialog.exec() == QDialog.Accepted:
            # Update analyzer parameters
            params = dialog.get_parameters()
            self.analyzer.noise_threshold = params['noise_threshold']
            self.analyzer.max_iterations = params['max_iterations']
            self.analyzer.correlation_threshold = params['correlation_threshold']
            self.analyzer.convergence_threshold = params['convergence_threshold']
            self.analyzer.reduced_chi_squared_target = params['reduced_chi_squared_target']
            self.analyzer.residual_std_threshold = params['residual_std_threshold']
            self.analyzer.bootstrap_samples = params['bootstrap_samples']
            self.analyzer.monte_carlo_samples = params['monte_carlo_samples']
            
            self.status_bar.showMessage("Analysis parameters updated")
            
    def run_analysis(self):
        """Run the mixture analysis."""
        if self.current_wavenumbers is None or self.current_spectrum is None:
            if self.launched_from_main_app:
                QMessageBox.warning(self, "No Spectrum Data", 
                    "No spectrum data was passed from the main RamanLab application.\n\n"
                    "Please:\n"
                    "1. Load a spectrum in the main RamanLab window\n"
                    "2. Then launch Mixture Analysis again")
            else:
                QMessageBox.warning(self, "No Data", "Please load a spectrum file first using the Load Spectrum button.")
            return
            
        # Disable controls during analysis
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create analysis name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_name = f"mixture_analysis_{timestamp}"
        
        # Start worker thread
        self.analysis_worker = MixtureAnalysisWorker(
            self.analyzer,
            self.current_wavenumbers,
            self.current_spectrum,
            self.max_components.value(),
            analysis_name,
            self.constrained_components
        )
        
        self.analysis_worker.progress_updated.connect(self.on_progress_updated)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.analysis_failed.connect(self.on_analysis_failed)
        
        self.analysis_worker.start()
        
    def on_progress_updated(self, message, percentage):
        """Handle progress updates from worker."""
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)
        
    def on_analysis_completed(self, results):
        """Handle completed analysis."""
        self.current_results = results
        self.analysis_history.append(results)
        
        # Re-enable controls
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update displays
        self.update_results_display()
        self.update_components_list()
        self.plot_results()
        
        # Show results groups
        self.results_group.setVisible(True)
        self.components_group.setVisible(True)
        
        self.status_bar.showMessage("Analysis completed successfully")
        
    def on_analysis_failed(self, error_message):
        """Handle failed analysis."""
        # Re-enable controls
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Analysis Failed", f"Analysis failed:\n{error_message}")
        self.status_bar.showMessage("Analysis failed")
        
    def update_results_display(self):
        """Update the results text display."""
        if not self.current_results:
            return
            
        results_text = []
        results_text.append("=== MIXTURE ANALYSIS RESULTS ===\n")
        
        # Basic info
        metadata = self.current_results.get('analysis_metadata', {})
        results_text.append(f"Components found: {metadata.get('components_found', 0)}")
        results_text.append(f"Analysis time: {metadata.get('analysis_time', 0):.2f} seconds")
        results_text.append(f"Database saved: {'✓' if metadata.get('database_saved', False) else '✗'}")
        results_text.append("")
        
        # Components
        components = self.current_results.get('identified_components', [])
        if components:
            results_text.append("Identified Components:")
            for i, comp in enumerate(components, 1):
                mineral = comp['mineral']
                correlation = comp['correlation']
                percentage = comp.get('percentage', 0)
                profile = comp['fit_parameters'].get('selected_profile', 'unknown')
                
                results_text.append(f"  {i}. {mineral}")
                results_text.append(f"     Correlation: {correlation:.3f}")
                results_text.append(f"     Percentage: {percentage:.1f}%")
                results_text.append(f"     Profile: {profile}")
                results_text.append("")
        
        # Final statistics
        final_stats = self.current_results.get('final_statistics', {})
        if final_stats:
            results_text.append("Final Statistics:")
            results_text.append(f"  R² value: {final_stats.get('r_squared', 0):.4f}")
            results_text.append(f"  RMS residual: {final_stats.get('rms_residual', 0):.6f}")
            results_text.append("")
        
        # Convergence info
        conv_details = self.current_results.get('convergence_details', {})
        if conv_details:
            results_text.append("Convergence Status:")
            for criterion, status in conv_details.get('criteria_met', {}).items():
                icon = "✓" if status else "✗"
                results_text.append(f"  {icon} {criterion}")
        
        self.results_text.setText("\n".join(results_text))
        
    def update_components_list(self):
        """Update the components list widget."""
        self.components_list.clear()
        
        if not self.current_results:
            return
            
        components = self.current_results.get('identified_components', [])
        for i, comp in enumerate(components):
            mineral = comp['mineral']
            correlation = comp['correlation']
            percentage = comp.get('percentage', 0)
            
            item_text = f"{mineral} ({percentage:.1f}%, r={correlation:.3f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # Store component index
            self.components_list.addItem(item)
            
    def on_component_selected(self, item):
        """Handle component selection."""
        comp_index = item.data(Qt.UserRole)
        if comp_index is not None and self.current_results:
            components = self.current_results.get('identified_components', [])
            if comp_index < len(components):
                comp = components[comp_index]
                # Highlight this component in the plot
                self.highlight_component(comp_index)
                
    def highlight_component(self, comp_index):
        """Highlight a specific component in the plot."""
        if not self.current_results:
            return
            
        components = self.current_results.get('identified_components', [])
        if comp_index >= len(components):
            return
            
        # Re-plot with highlighted component
        self.plot_results(highlight_index=comp_index)
        
    def plot_results(self, highlight_index=None):
        """Plot the analysis results."""
        if not self.current_results or self.current_wavenumbers is None:
            return
            
        # Check if canvas and axes are ready (deferred initialization)
        if not hasattr(self, 'canvas') or self.canvas is None:
            print("Warning: Canvas not yet initialized, deferring results plot")
            return
            
        if not hasattr(self, 'ax_main') or self.ax_main is None:
            print("Warning: Axes not yet initialized, deferring results plot")
            return
            
        # Clear all subplots
        for ax in [self.ax_main, self.ax_components, self.ax_fit, self.ax_residuals]:
            ax.clear()
            
        components = self.current_results.get('identified_components', [])
        final_stats = self.current_results.get('final_statistics', {})
        
        # Original spectrum
        self.ax_main.plot(self.current_wavenumbers, self.current_spectrum, 'b-', 
                         linewidth=1.5, label='Original')
        self.ax_main.set_title("Original Spectrum")
        self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_main.set_ylabel("Intensity")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend()
        
        # Components
        colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
        for i, (comp, color) in enumerate(zip(components, colors)):
            synthetic_spectrum = comp.get('synthetic_spectrum', [])
            if len(synthetic_spectrum) == len(self.current_wavenumbers):
                linewidth = 3.0 if i == highlight_index else 1.5
                alpha = 1.0 if i == highlight_index else 0.7
                self.ax_components.plot(self.current_wavenumbers, synthetic_spectrum, 
                                      color=color, linewidth=linewidth, alpha=alpha,
                                      label=f"{comp['mineral']} ({comp.get('percentage', 0):.1f}%)")
        
        self.ax_components.set_title("Identified Components")
        self.ax_components.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_components.set_ylabel("Intensity")
        self.ax_components.grid(True, alpha=0.3)
        if components:
            self.ax_components.legend(fontsize=8)
        
        # Fit quality
        self.ax_fit.plot(self.current_wavenumbers, self.current_spectrum, 'b-', 
                        linewidth=1.5, label='Original', alpha=0.7)
        
        total_synthetic = final_stats.get('total_synthetic', [])
        if len(total_synthetic) == len(self.current_wavenumbers):
            self.ax_fit.plot(self.current_wavenumbers, total_synthetic, 'r-', 
                           linewidth=1.5, label='Total Fit')
        
        r_squared = final_stats.get('r_squared', 0)
        self.ax_fit.set_title(f"Fit Quality (R² = {r_squared:.4f})")
        self.ax_fit.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_fit.set_ylabel("Intensity")
        self.ax_fit.grid(True, alpha=0.3)
        self.ax_fit.legend()
        
        # Residuals
        final_residual = final_stats.get('final_residual', [])
        if len(final_residual) == len(self.current_wavenumbers):
            self.ax_residuals.plot(self.current_wavenumbers, final_residual, 'g-', 
                                 linewidth=1.0)
            self.ax_residuals.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        rms_residual = final_stats.get('rms_residual', 0)
        self.ax_residuals.set_title(f"Residuals (RMS = {rms_residual:.6f})")
        self.ax_residuals.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_residuals.set_ylabel("Residual")
        self.ax_residuals.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        
        # Safely draw the canvas
        try:
            if hasattr(self, 'canvas') and self.canvas is not None and hasattr(self.canvas, 'draw'):
                self.canvas.draw()
            else:
                print("Warning: Canvas not available for plot results")
        except RuntimeError as e:
            print(f"Warning: Could not draw plot results - {e}")
        
    def export_results(self):
        """Export analysis results."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "mixture_analysis_results.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as f:
                f.write("RamanLab Mixture Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                
                # Analysis metadata
                metadata = self.current_results.get('analysis_metadata', {})
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Components Found: {metadata.get('components_found', 0)}\n")
                f.write(f"Analysis Time: {metadata.get('analysis_time', 0):.2f} seconds\n")
                f.write(f"Database Saved: {'Yes' if metadata.get('database_saved', False) else 'No'}\n\n")
                
                # Components
                components = self.current_results.get('identified_components', [])
                if components:
                    f.write("Identified Components:\n")
                    f.write("-" * 30 + "\n")
                    for i, comp in enumerate(components, 1):
                        f.write(f"{i}. {comp['mineral']}\n")
                        f.write(f"   Correlation: {comp['correlation']:.6f}\n")
                        f.write(f"   Percentage: {comp.get('percentage', 0):.2f}%\n")
                        f.write(f"   Profile: {comp['fit_parameters'].get('selected_profile', 'unknown')}\n")
                        f.write(f"   R²: {comp['fit_parameters'].get('r_squared', 0):.6f}\n\n")
                
                # Final statistics
                final_stats = self.current_results.get('final_statistics', {})
                if final_stats:
                    f.write("Final Statistics:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Overall R²: {final_stats.get('r_squared', 0):.6f}\n")
                    f.write(f"RMS Residual: {final_stats.get('rms_residual', 0):.8f}\n\n")
                
                # Convergence details
                conv_details = self.current_results.get('convergence_details', {})
                if conv_details:
                    f.write("Convergence Status:\n")
                    f.write("-" * 25 + "\n")
                    for criterion, status in conv_details.get('criteria_met', {}).items():
                        f.write(f"  {criterion}: {'Passed' if status else 'Failed'}\n")
                    f.write(f"\nReason: {conv_details.get('reason', 'Unknown')}\n")
            
            self.status_bar.showMessage(f"Results exported to {Path(file_path).name}")
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not export results:\n{str(e)}")
            
    def refresh_plots(self):
        """Refresh all plots."""
        # Check if canvas is ready (deferred initialization)
        if not hasattr(self, 'canvas') or self.canvas is None:
            print("Warning: Canvas not yet initialized, deferring plot refresh")
            return
            
        if self.current_results:
            self.plot_results()
        elif self.current_wavenumbers is not None:
            self.plot_original_spectrum()
        else:
            self.setup_empty_plots()
            
    def show_about(self):
        """Show about dialog."""
        about_text = """
        RamanLab Mixture Analysis
        
        Advanced spectral mixture analysis using iterative decomposition
        with uncertainty quantification and database integration.
        
        Features:
        • Multiple peak profile fitting (Gaussian, Lorentzian, Voigt, Asymmetric)
        • Bootstrap resampling for uncertainty analysis
        • Monte Carlo parameter estimation
        • Integration with RamanLab database
        • Comprehensive convergence monitoring
        
        Part of the RamanLab suite by Aaron Celestian
        """
        QMessageBox.about(self, "About Mixture Analysis", about_text)
        
    def _is_widget_valid(self, widget):
        """Check if a Qt widget is still valid and not deleted."""
        try:
            if widget is None:
                return False
            # Try to access a basic property to test if the C++ object exists
            widget.isVisible()
            return True
        except RuntimeError:
            # C++ object has been deleted
            return False
        except Exception:
            # Any other error means the widget is not usable
            return False
    
    def _process_pending_constraints(self):
        """Process any pending constraints added before the UI was ready."""
        while self._pending_constraints:
            constraint = self._pending_constraints.pop(0)
            self.add_constraint_from_search(constraint['mineral'], constraint['confidence'], constraint['metadata'])

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any running analysis
        if self.analysis_worker and self.analysis_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Analysis Running",
                "Analysis is currently running. Do you want to close anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            else:
                self.analysis_worker.terminate()
                self.analysis_worker.wait()
        
        # Mark UI as no longer ready
        self._ui_ready = False
        
        # Save state if available
        if STATE_MANAGEMENT_AVAILABLE:
            try:
                save_module_state('mixture_analysis', "Auto-save on close")
            except:
                pass
        
        event.accept()

    def showEvent(self, event):
        """Handle window show event to perform deferred initialization."""
        super().showEvent(event)
        
        # Perform deferred canvas drawing
        if getattr(self, '_initial_draw_pending', False):
            self._initial_draw_pending = False
            
            # Use QTimer to ensure canvas drawing happens after window is fully shown
            QTimer.singleShot(100, self._perform_initial_canvas_draw)
            
    def _perform_initial_canvas_draw(self):
        """Create the canvas and perform the initial drawing after window is shown."""
        try:
            print("🎨 Creating matplotlib canvas...")
            
            # Clear the placeholder
            for i in reversed(range(self.viz_layout.count())):
                child = self.viz_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            # Create matplotlib figure and canvas
            self.figure = Figure(figsize=(12, 10))
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self.viz_frame)
            
            # Add to layout
            self.viz_layout.addWidget(self.toolbar)
            self.viz_layout.addWidget(self.canvas)
            
            # Setup empty plots
            self.setup_empty_plots()
            
            # If spectrum data is available, plot it now
            if self.current_wavenumbers is not None and self.current_spectrum is not None:
                self.plot_original_spectrum()
                print("✅ Canvas created and spectrum plotted successfully")
            else:
                print("✅ Canvas creation and initial drawing completed successfully")
            
        except RuntimeError as e:
            print(f"Warning: Could not create canvas - {e}")
        except Exception as e:
            print(f"Warning: Unexpected error during canvas creation - {e}")

    def add_constraint_from_search(self, mineral_name: str, confidence: float, metadata: dict = None):
        """Add a constraint from search results."""
        # If UI is not ready, store the constraint for later processing
        if not self._ui_ready or not self._is_widget_valid(getattr(self, 'constraints_list', None)):
            self._pending_constraints.append({
                'mineral': mineral_name,
                'confidence': confidence,
                'metadata': metadata or {}
            })
            print(f"📝 Queued constraint for later: {mineral_name} (UI not ready)")
            return
        
        constraint = {
            'mineral': mineral_name,
            'confidence': confidence,
            'metadata': metadata or {},
            'source': 'search_results'
        }
        
        # Check if this mineral is already in constraints
        for existing in self.constrained_components:
            if existing['mineral'] == mineral_name:
                # Update existing constraint with better confidence if applicable
                if confidence > existing['confidence']:
                    existing['confidence'] = confidence
                    existing['metadata'] = metadata or {}
                    self.update_constraints_display()
                return
        
        # Add new constraint
        self.constrained_components.append(constraint)
        self.update_constraints_display()
        
        # Update status
        if self._is_widget_valid(getattr(self, 'status_bar', None)):
            self.status_bar.showMessage(f"Added constraint: {mineral_name} (confidence: {confidence:.3f})")
    
    def add_constraint_dialog(self):
        """Show dialog to manually add a known mineral constraint."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Known Mineral Component")
        dialog.setFixedSize(400, 200)
        
        layout = QVBoxLayout(dialog)
        
        # Mineral selection
        mineral_layout = QHBoxLayout()
        mineral_layout.addWidget(QLabel("Mineral:"))
        
        mineral_combo = QComboBox()
        # Add minerals from database
        if hasattr(self.analyzer, 'database'):
            mineral_names = sorted(list(self.analyzer.database.keys()))
            mineral_combo.addItems(mineral_names[:100])  # Show first 100 for performance
        mineral_layout.addWidget(mineral_combo)
        layout.addLayout(mineral_layout)
        
        # Confidence setting
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        
        confidence_spin = QDoubleSpinBox()
        confidence_spin.setRange(0.0, 1.0)
        confidence_spin.setSingleStep(0.1)
        confidence_spin.setValue(0.8)
        confidence_spin.setDecimals(3)
        confidence_layout.addWidget(confidence_spin)
        layout.addLayout(confidence_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add")
        cancel_btn = QPushButton("Cancel")
        
        def add_constraint():
            mineral = mineral_combo.currentText()
            confidence = confidence_spin.value()
            if mineral:
                self.add_constraint_from_search(mineral, confidence, {'source': 'manual'})
                dialog.accept()
        
        add_btn.clicked.connect(add_constraint)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(add_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def remove_selected_constraint(self):
        """Remove the selected constraint from the list."""
        # Check if widgets exist before accessing them
        if not self._is_widget_valid(getattr(self, 'constraints_list', None)):
            return
            
        current_row = self.constraints_list.currentRow()
        if current_row >= 0 and current_row < len(self.constrained_components):
            removed = self.constrained_components.pop(current_row)
            self.update_constraints_display()
            
            if self._is_widget_valid(getattr(self, 'status_bar', None)):
                self.status_bar.showMessage(f"Removed constraint: {removed['mineral']}")
    
    def clear_all_constraints(self):
        """Clear all constraints."""
        self.constrained_components.clear()
        self.update_constraints_display()
        
        if self._is_widget_valid(getattr(self, 'status_bar', None)):
            self.status_bar.showMessage("All constraints cleared")
    
    def on_constraint_selection_changed(self):
        """Handle constraint selection changes."""
        # Check if widgets exist before accessing them
        if not self._is_widget_valid(getattr(self, 'constraints_list', None)):
            return
        if not self._is_widget_valid(getattr(self, 'remove_constraint_btn', None)):
            return
        if not self._is_widget_valid(getattr(self, 'clear_constraints_btn', None)):
            return
            
        has_selection = self.constraints_list.currentRow() >= 0
        has_constraints = len(self.constrained_components) > 0
        
        self.remove_constraint_btn.setEnabled(has_selection)
        self.clear_constraints_btn.setEnabled(has_constraints)
    
    def update_constraints_display(self):
        """Update the constraints list display."""
        # Check if widgets exist before accessing them
        if not self._is_widget_valid(getattr(self, 'constraints_list', None)):
            print("Warning: constraints_list widget not available, skipping display update")
            return
            
        try:
            self.constraints_list.clear()
            
            for constraint in self.constrained_components:
                mineral = constraint['mineral']
                confidence = constraint['confidence']
                source = constraint.get('source', 'unknown')
                
                display_text = f"{mineral} (confidence: {confidence:.3f}, from: {source})"
                self.constraints_list.addItem(display_text)
            
            # Update button states
            self.on_constraint_selection_changed()
            
            # Update analysis button text if constraints exist
            if self._is_widget_valid(getattr(self, 'run_btn', None)):
                if self.constrained_components:
                    constraint_count = len(self.constrained_components)
                    self.run_btn.setText(f"🎯 Run Constrained Analysis ({constraint_count} known)")
                    self.run_btn.setToolTip(f"Run mixture analysis starting with {constraint_count} known components")
                else:
                    self.run_btn.setText("🔬 Run Analysis")
                    self.run_btn.setToolTip("Run standard mixture analysis")
                    
        except RuntimeError as e:
            print(f"Warning: Could not update constraints display - widget may have been deleted: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error updating constraints display: {e}")


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("RamanLab Mixture Analysis")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("RamanLab")
    
    # Create and show main window
    window = RamanMixtureAnalysisQt6()
    window.show()
    
    # Center window on screen
    screen = app.primaryScreen().geometry()
    window_size = window.geometry()
    x = (screen.width() - window_size.width()) // 2
    y = (screen.height() - window_size.height()) // 2
    window.move(x, y)
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main()) 