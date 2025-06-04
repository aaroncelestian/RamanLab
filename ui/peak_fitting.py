#!/usr/bin/env python3
"""
Peak Fitting UI Module for RamanLab

This module provides a professional PySide6 widget for interactive peak fitting
with support for multiple peak models, real-time parameter adjustment,
and comprehensive analysis capabilities.

Features:
- Interactive peak detection and manual selection
- Multiple fitting models (Gaussian, Lorentzian, Pseudo-Voigt, Asymmetric Voigt)
- Real-time parameter adjustment
- Deconvolution and component separation
- Professional multi-panel visualization
- Fit quality metrics and residual analysis
"""

import sys
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QTextEdit, QTableWidget, QTableWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QFormLayout, QFrame
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPalette

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.colors as mcolors

# Configure matplotlib for smaller toolbar
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'toolbar2'

try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Peak fitting functionality will be limited.")


class PeakFittingWidget(QWidget):
    """
    Professional peak fitting widget with advanced analysis capabilities.
    
    Features:
    - Interactive peak detection and selection
    - Multiple peak models and fitting algorithms
    - Real-time parameter adjustment
    - Multi-panel visualization with residuals
    - Component separation and deconvolution
    """
    
    # Signals for communication with other modules
    peak_fitted = Signal(dict)  # Emitted when peaks are fitted
    parameters_changed = Signal(dict)  # Emitted when parameters change
    analysis_completed = Signal(dict)  # Emitted when analysis is complete
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.wavenumbers = None
        self.intensities = None
        self.processed_intensities = None
        self.peaks = []
        self.manual_peaks = []
        self.fit_params = None
        self.fit_result = None
        self.residuals = None
        self.individual_peaks = []
        
        # Analysis parameters
        self.current_model = "Gaussian"
        self.interactive_mode = False
        self.show_individual_peaks = True
        self.show_residuals = True
        
        # Peak detection parameters
        self.min_peak_height = 0.1
        self.min_peak_distance = 10
        self.peak_prominence = 0.05
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the main user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        controls_widget = self.create_controls_panel()
        controls_widget.setMaximumWidth(350)
        controls_widget.setMinimumWidth(300)
        splitter.addWidget(controls_widget)
        
        # Right panel - Visualization
        plot_widget = self.create_plot_panel()
        splitter.addWidget(plot_widget)
        
        # Set splitter proportions
        splitter.setSizes([300, 800])
        
    def create_controls_panel(self):
        """Create the left control panel with tabs."""
        controls_widget = QWidget()
        layout = QVBoxLayout(controls_widget)
        
        # Title
        title_label = QLabel("🔍 Peak Fitting Analysis")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2c3e50; padding: 10px; background-color: #ecf0f1; border-radius: 5px; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Tab widget for organized controls
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.tab_widget.addTab(self.create_peak_detection_tab(), "🎯 Detection")
        self.tab_widget.addTab(self.create_fitting_tab(), "📊 Fitting")
        self.tab_widget.addTab(self.create_analysis_tab(), "🔬 Analysis")
        self.tab_widget.addTab(self.create_results_tab(), "📋 Results")
        
        return controls_widget
        
    def create_peak_detection_tab(self):
        """Create the peak detection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Auto Detection Parameters
        auto_group = QGroupBox("Automatic Detection")
        auto_layout = QFormLayout(auto_group)
        
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.01, 1.0)
        self.height_spin.setValue(self.min_peak_height)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setDecimals(3)
        auto_layout.addRow("Min Height:", self.height_spin)
        
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(self.min_peak_distance)
        auto_layout.addRow("Min Distance:", self.distance_spin)
        
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.001, 1.0)
        self.prominence_spin.setValue(self.peak_prominence)
        self.prominence_spin.setSingleStep(0.001)
        self.prominence_spin.setDecimals(4)
        auto_layout.addRow("Prominence:", self.prominence_spin)
        
        layout.addWidget(auto_group)
        
        # Detection Buttons
        detection_buttons = QVBoxLayout()
        
        self.detect_btn = QPushButton("🔍 Auto Detect Peaks")
        self.detect_btn.clicked.connect(self.detect_peaks)
        detection_buttons.addWidget(self.detect_btn)
        
        self.interactive_btn = QPushButton("✋ Manual Selection")
        self.interactive_btn.setCheckable(True)
        self.interactive_btn.clicked.connect(self.toggle_interactive_mode)
        detection_buttons.addWidget(self.interactive_btn)
        
        clear_layout = QHBoxLayout()
        self.clear_auto_btn = QPushButton("Clear Auto")
        self.clear_manual_btn = QPushButton("Clear Manual")
        self.clear_auto_btn.clicked.connect(self.clear_auto_peaks)
        self.clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        clear_layout.addWidget(self.clear_auto_btn)
        clear_layout.addWidget(self.clear_manual_btn)
        detection_buttons.addLayout(clear_layout)
        
        layout.addLayout(detection_buttons)
        
        # Peak Count Status
        status_group = QGroupBox("Peak Status")
        status_layout = QVBoxLayout(status_group)
        
        self.peak_count_label = QLabel("Auto: 0 | Manual: 0 | Total: 0")
        self.peak_count_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        status_layout.addWidget(self.peak_count_label)
        
        self.interactive_status_label = QLabel("Interactive mode: OFF")
        self.interactive_status_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        status_layout.addWidget(self.interactive_status_label)
        
        layout.addWidget(status_group)
        layout.addStretch()
        
        return tab
        
    def create_fitting_tab(self):
        """Create the fitting tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model Selection
        model_group = QGroupBox("Peak Model")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Gaussian", "Lorentzian", "Pseudo-Voigt", "Asymmetric Voigt"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_group)
        
        # Display Options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_peaks_check = QCheckBox("Show Individual Peaks")
        self.show_peaks_check.setChecked(True)
        self.show_peaks_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_peaks_check)
        
        self.show_residuals_check = QCheckBox("Show Residuals")
        self.show_residuals_check.setChecked(True)
        self.show_residuals_check.toggled.connect(self.on_display_changed)
        display_layout.addWidget(self.show_residuals_check)
        
        layout.addWidget(display_group)
        
        # Fitting Controls
        fit_group = QGroupBox("Peak Fitting")
        fit_layout = QVBoxLayout(fit_group)
        
        self.fit_btn = QPushButton("🎯 Fit Peaks")
        self.fit_btn.clicked.connect(self.fit_peaks)
        fit_layout.addWidget(self.fit_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        fit_layout.addWidget(self.progress_bar)
        
        layout.addWidget(fit_group)
        layout.addStretch()
        
        return tab
        
    def create_analysis_tab(self):
        """Create the analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Component Analysis
        comp_group = QGroupBox("Component Analysis")
        comp_layout = QFormLayout(comp_group)
        
        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(1, 10)
        self.n_components_spin.setValue(3)
        comp_layout.addRow("Components:", self.n_components_spin)
        
        deconv_buttons = QHBoxLayout()
        self.resolve_btn = QPushButton("Resolve Overlaps")
        self.separate_btn = QPushButton("Separate Components")
        self.resolve_btn.clicked.connect(self.resolve_overlapping_peaks)
        self.separate_btn.clicked.connect(self.separate_components)
        deconv_buttons.addWidget(self.resolve_btn)
        deconv_buttons.addWidget(self.separate_btn)
        comp_layout.addRow(deconv_buttons)
        
        layout.addWidget(comp_group)
        
        # Quality Metrics
        quality_group = QGroupBox("Fit Quality")
        quality_layout = QVBoxLayout(quality_group)
        
        self.r_squared_label = QLabel("R²: --")
        self.rmse_label = QLabel("RMSE: --")
        self.chi_squared_label = QLabel("χ²: --")
        
        quality_layout.addWidget(self.r_squared_label)
        quality_layout.addWidget(self.rmse_label)
        quality_layout.addWidget(self.chi_squared_label)
        
        layout.addWidget(quality_group)
        layout.addStretch()
        
        return tab
        
    def create_results_tab(self):
        """Create the results tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results Text Area
        results_group = QGroupBox("Fit Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Export Controls
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        self.export_params_btn = QPushButton("📄 Export Parameters")
        self.export_fit_btn = QPushButton("📊 Export Fit Data")
        self.export_report_btn = QPushButton("📋 Generate Report")
        
        export_layout.addWidget(self.export_params_btn)
        export_layout.addWidget(self.export_fit_btn)
        export_layout.addWidget(self.export_report_btn)
        
        layout.addWidget(export_group)
        layout.addStretch()
        
        return tab
        
    def create_plot_panel(self):
        """Create the right plot panel."""
        plot_widget = QWidget()
        layout = QVBoxLayout(plot_widget)
        
        # Create matplotlib figure with subplots
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Create subplots
        self.setup_subplots()
        
        return plot_widget
        
    def setup_subplots(self):
        """Setup the matplotlib subplots."""
        self.figure.clear()
        
        # Main spectrum plot (2/3 of height)
        self.ax_main = self.figure.add_subplot(3, 1, (1, 2))
        self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_main.set_ylabel("Intensity")
        self.ax_main.set_title("Peak Fitting Analysis", fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        
        # Residuals plot (1/3 of height)
        self.ax_residuals = self.figure.add_subplot(3, 1, 3)
        self.ax_residuals.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_residuals.set_ylabel("Residuals")
        self.ax_residuals.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        
    def setup_connections(self):
        """Setup signal connections."""
        # Connect parameter changes to real-time updates
        self.height_spin.valueChanged.connect(self.on_parameters_changed)
        self.distance_spin.valueChanged.connect(self.on_parameters_changed)
        self.prominence_spin.valueChanged.connect(self.on_parameters_changed)
        
        # Connect mouse events for interactive peak selection
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
    def load_spectrum(self, wavenumbers, intensities):
        """Load spectrum data for analysis."""
        self.wavenumbers = np.array(wavenumbers)
        self.intensities = np.array(intensities)
        self.processed_intensities = np.copy(self.intensities)
        
        # Clear previous results
        self.clear_all_peaks()
        self.fit_params = None
        self.fit_result = None
        self.residuals = None
        
        # Update plot
        self.update_plot()
        
    def set_spectrum(self, spectrum_data):
        """
        Set spectrum data from a dictionary (compatibility method).
        
        Args:
            spectrum_data: Dictionary containing 'wavenumbers' and 'intensities' keys
        """
        if isinstance(spectrum_data, dict):
            wavenumbers = spectrum_data.get('wavenumbers')
            intensities = spectrum_data.get('intensities')
            if wavenumbers is not None and intensities is not None:
                self.load_spectrum(wavenumbers, intensities)
        else:
            # Assume it's a tuple/list of (wavenumbers, intensities)
            self.load_spectrum(spectrum_data[0], spectrum_data[1])
        
    def detect_peaks(self):
        """Detect peaks automatically using scipy."""
        if not SCIPY_AVAILABLE or self.processed_intensities is None:
            QMessageBox.warning(self, "Error", "SciPy not available or no data loaded.")
            return
            
        try:
            # Get parameters
            height = self.height_spin.value() * np.max(self.processed_intensities)
            distance = self.distance_spin.value()
            prominence = self.prominence_spin.value() * np.max(self.processed_intensities)
            
            # Find peaks
            peak_indices, properties = find_peaks(
                self.processed_intensities,
                height=height,
                distance=distance,
                prominence=prominence
            )
            
            # Store peak positions
            self.peaks = peak_indices.tolist()
            self.update_peak_count()
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Peak detection failed: {str(e)}")
            
    def toggle_interactive_mode(self):
        """Toggle interactive peak selection mode."""
        self.interactive_mode = self.interactive_btn.isChecked()
        
        if self.interactive_mode:
            self.interactive_btn.setText("✋ Interactive: ON")
            self.interactive_status_label.setText("Interactive mode: ON - Click to add peaks")
            self.interactive_status_label.setStyleSheet("color: #27ae60; font-size: 10px; font-weight: bold;")
        else:
            self.interactive_btn.setText("✋ Manual Selection")
            self.interactive_status_label.setText("Interactive mode: OFF")
            self.interactive_status_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
            
    def on_canvas_click(self, event):
        """Handle canvas clicks for interactive peak selection."""
        if not self.interactive_mode or not event.inaxes or self.wavenumbers is None:
            return
            
        # Only respond to clicks on the main plot
        if event.inaxes != self.ax_main:
            return
            
        x_click = event.xdata
        if x_click is None:
            return
            
        # Find nearest wavenumber index
        nearest_idx = np.argmin(np.abs(self.wavenumbers - x_click))
        
        # Add to manual peaks if not too close to existing peaks
        min_distance = self.distance_spin.value()
        too_close = False
        
        for peak_idx in self.manual_peaks + self.peaks:
            if abs(peak_idx - nearest_idx) < min_distance:
                too_close = True
                break
                
        if not too_close:
            self.manual_peaks.append(nearest_idx)
            self.update_peak_count()
            self.update_plot()
            
    def clear_auto_peaks(self):
        """Clear automatically detected peaks."""
        self.peaks = []
        self.update_peak_count()
        self.update_plot()
        
    def clear_manual_peaks(self):
        """Clear manually selected peaks."""
        self.manual_peaks = []
        self.update_peak_count()
        self.update_plot()
        
    def clear_all_peaks(self):
        """Clear all peaks."""
        self.peaks = []
        self.manual_peaks = []
        self.update_peak_count()
        
    def update_peak_count(self):
        """Update the peak count display."""
        auto_count = len(self.peaks)
        manual_count = len(self.manual_peaks)
        total_count = auto_count + manual_count
        
        self.peak_count_label.setText(f"Auto: {auto_count} | Manual: {manual_count} | Total: {total_count}")
        
    def on_model_changed(self):
        """Handle model selection change."""
        self.current_model = self.model_combo.currentText()
        self.parameters_changed.emit({"model": self.current_model})
        
    def on_display_changed(self):
        """Handle display option changes."""
        self.show_individual_peaks = self.show_peaks_check.isChecked()
        self.show_residuals = self.show_residuals_check.isChecked()
        self.update_plot()
        
    def on_parameters_changed(self):
        """Handle parameter changes."""
        self.min_peak_height = self.height_spin.value()
        self.min_peak_distance = self.distance_spin.value()
        self.peak_prominence = self.prominence_spin.value()
        
        params = {
            "height": self.min_peak_height,
            "distance": self.min_peak_distance,
            "prominence": self.peak_prominence
        }
        self.parameters_changed.emit(params)
        
    def fit_peaks(self):
        """Fit peaks using the selected model."""
        if not SCIPY_AVAILABLE:
            QMessageBox.warning(self, "Error", "SciPy not available for fitting.")
            return
            
        all_peaks = self.peaks + self.manual_peaks
        if not all_peaks or self.wavenumbers is None:
            QMessageBox.warning(self, "Warning", "No peaks to fit or no data loaded.")
            return
            
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(20)
            
            # Prepare initial parameters and bounds
            initial_params, bounds_lower, bounds_upper = self.prepare_fit_parameters(all_peaks)
            
            self.progress_bar.setValue(40)
            
            # Perform the fit
            bounds = (bounds_lower, bounds_upper)
            popt, pcov = curve_fit(
                self.multi_peak_model,
                self.wavenumbers,
                self.processed_intensities,
                p0=initial_params,
                bounds=bounds,
                maxfev=10000
            )
            
            self.progress_bar.setValue(70)
            
            # Store results
            self.fit_params = popt
            self.fit_result = self.multi_peak_model(self.wavenumbers, *popt)
            self.residuals = self.processed_intensities - self.fit_result
            
            # Calculate individual peaks
            self.calculate_individual_peaks()
            
            self.progress_bar.setValue(90)
            
            # Display results
            self.display_fit_results()
            self.calculate_quality_metrics()
            self.update_plot()
            
            self.progress_bar.setValue(100)
            
            # Emit signals
            self.peak_fitted.emit({
                "model": self.current_model,
                "parameters": popt.tolist(),
                "r_squared": getattr(self, 'r_squared', 0),
                "peaks": all_peaks
            })
            
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Peak fitting failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def prepare_fit_parameters(self, peak_indices):
        """Prepare initial parameters and bounds for fitting."""
        initial_params = []
        bounds_lower = []
        bounds_upper = []
        
        x_min, x_max = np.min(self.wavenumbers), np.max(self.wavenumbers)
        y_min, y_max = np.min(self.processed_intensities), np.max(self.processed_intensities)
        
        for peak_idx in peak_indices:
            if 0 <= peak_idx < len(self.wavenumbers):
                # Amplitude
                amp = self.processed_intensities[peak_idx]
                # Center
                cen = self.wavenumbers[peak_idx]
                # Width estimate
                wid = 10.0
                
                if self.current_model in ["Gaussian", "Lorentzian"]:
                    initial_params.extend([amp, cen, wid])
                    bounds_lower.extend([0, max(x_min, cen - 50), 1])
                    bounds_upper.extend([y_max * 2, min(x_max, cen + 50), 100])
                elif self.current_model == "Pseudo-Voigt":
                    initial_params.extend([amp, cen, wid, 0.5])
                    bounds_lower.extend([0, max(x_min, cen - 50), 1, 0])
                    bounds_upper.extend([y_max * 2, min(x_max, cen + 50), 100, 1])
                elif self.current_model == "Asymmetric Voigt":
                    initial_params.extend([amp, cen, wid, wid, 0.5])
                    bounds_lower.extend([0, max(x_min, cen - 50), 1, 1, 0])
                    bounds_upper.extend([y_max * 2, min(x_max, cen + 50), 100, 100, 1])
                    
        return initial_params, bounds_lower, bounds_upper
        
    def multi_peak_model(self, x, *params):
        """Multi-peak model for fitting."""
        if self.current_model == "Gaussian":
            return self.multi_gaussian(x, *params)
        elif self.current_model == "Lorentzian":
            return self.multi_lorentzian(x, *params)
        elif self.current_model == "Pseudo-Voigt":
            return self.multi_pseudo_voigt(x, *params)
        elif self.current_model == "Asymmetric Voigt":
            return self.multi_asymmetric_voigt(x, *params)
        else:
            return self.multi_gaussian(x, *params)
            
    def multi_gaussian(self, x, *params):
        """Multiple Gaussian peaks model."""
        result = np.zeros_like(x)
        n_peaks = len(params) // 3
        for i in range(n_peaks):
            amp, cen, wid = params[i*3:(i+1)*3]
            result += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
        return result
        
    def multi_lorentzian(self, x, *params):
        """Multiple Lorentzian peaks model."""
        result = np.zeros_like(x)
        n_peaks = len(params) // 3
        for i in range(n_peaks):
            amp, cen, wid = params[i*3:(i+1)*3]
            result += amp * wid**2 / ((x - cen)**2 + wid**2)
        return result
        
    def multi_pseudo_voigt(self, x, *params):
        """Multiple Pseudo-Voigt peaks model."""
        result = np.zeros_like(x)
        n_peaks = len(params) // 4
        for i in range(n_peaks):
            amp, cen, wid, eta = params[i*4:(i+1)*4]
            gaussian = amp * np.exp(-(x - cen)**2 / (2 * wid**2))
            lorentzian = amp * wid**2 / ((x - cen)**2 + wid**2)
            result += eta * lorentzian + (1 - eta) * gaussian
        return result
        
    def multi_asymmetric_voigt(self, x, *params):
        """Multiple Asymmetric Voigt peaks model."""
        result = np.zeros_like(x)
        n_peaks = len(params) // 5
        for i in range(n_peaks):
            amp, cen, wid_l, wid_r, eta = params[i*5:(i+1)*5]
            
            # Left side
            left_mask = x <= cen
            if np.any(left_mask):
                gaussian_l = amp * np.exp(-(x[left_mask] - cen)**2 / (2 * wid_l**2))
                lorentzian_l = amp * wid_l**2 / ((x[left_mask] - cen)**2 + wid_l**2)
                result[left_mask] += eta * lorentzian_l + (1 - eta) * gaussian_l
                
            # Right side
            right_mask = x > cen
            if np.any(right_mask):
                gaussian_r = amp * np.exp(-(x[right_mask] - cen)**2 / (2 * wid_r**2))
                lorentzian_r = amp * wid_r**2 / ((x[right_mask] - cen)**2 + wid_r**2)
                result[right_mask] += eta * lorentzian_r + (1 - eta) * gaussian_r
                
        return result
        
    def calculate_individual_peaks(self):
        """Calculate individual peak contributions."""
        if self.fit_params is None:
            return
            
        self.individual_peaks = []
        all_peaks = self.peaks + self.manual_peaks
        
        if self.current_model in ["Gaussian", "Lorentzian"]:
            params_per_peak = 3
        elif self.current_model == "Pseudo-Voigt":
            params_per_peak = 4
        elif self.current_model == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3
            
        n_peaks = len(self.fit_params) // params_per_peak
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            end_idx = start_idx + params_per_peak
            peak_params = self.fit_params[start_idx:end_idx]
            
            if self.current_model == "Gaussian":
                peak_y = peak_params[0] * np.exp(-(self.wavenumbers - peak_params[1])**2 / (2 * peak_params[2]**2))
            elif self.current_model == "Lorentzian":
                peak_y = peak_params[0] * peak_params[2]**2 / ((self.wavenumbers - peak_params[1])**2 + peak_params[2]**2)
            elif self.current_model == "Pseudo-Voigt":
                gaussian = peak_params[0] * np.exp(-(self.wavenumbers - peak_params[1])**2 / (2 * peak_params[2]**2))
                lorentzian = peak_params[0] * peak_params[2]**2 / ((self.wavenumbers - peak_params[1])**2 + peak_params[2]**2)
                peak_y = peak_params[3] * lorentzian + (1 - peak_params[3]) * gaussian
            else:  # Asymmetric Voigt
                peak_y = np.zeros_like(self.wavenumbers)
                # Simplified version for display
                amp, cen, wid_l, wid_r, eta = peak_params
                left_mask = self.wavenumbers <= cen
                right_mask = self.wavenumbers > cen
                
                if np.any(left_mask):
                    gaussian_l = amp * np.exp(-(self.wavenumbers[left_mask] - cen)**2 / (2 * wid_l**2))
                    lorentzian_l = amp * wid_l**2 / ((self.wavenumbers[left_mask] - cen)**2 + wid_l**2)
                    peak_y[left_mask] = eta * lorentzian_l + (1 - eta) * gaussian_l
                    
                if np.any(right_mask):
                    gaussian_r = amp * np.exp(-(self.wavenumbers[right_mask] - cen)**2 / (2 * wid_r**2))
                    lorentzian_r = amp * wid_r**2 / ((self.wavenumbers[right_mask] - cen)**2 + wid_r**2)
                    peak_y[right_mask] = eta * lorentzian_r + (1 - eta) * gaussian_r
                    
            self.individual_peaks.append(peak_y)
            
    def calculate_quality_metrics(self):
        """Calculate fit quality metrics."""
        if self.fit_result is None or self.residuals is None:
            return
            
        # R-squared
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.processed_intensities - np.mean(self.processed_intensities)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # RMSE
        self.rmse = np.sqrt(np.mean(self.residuals ** 2))
        
        # Chi-squared
        self.chi_squared = ss_res / len(self.residuals)
        
        # Update labels
        self.r_squared_label.setText(f"R²: {self.r_squared:.4f}")
        self.rmse_label.setText(f"RMSE: {self.rmse:.2f}")
        self.chi_squared_label.setText(f"χ²: {self.chi_squared:.2e}")
        
    def display_fit_results(self):
        """Display fitting results in the text area."""
        if self.fit_params is None:
            return
            
        results_text = f"Peak Fitting Results - {self.current_model} Model\n"
        results_text += "=" * 50 + "\n\n"
        
        all_peaks = self.peaks + self.manual_peaks
        
        if self.current_model in ["Gaussian", "Lorentzian"]:
            params_per_peak = 3
            param_names = ["Amplitude", "Center (cm⁻¹)", "Width"]
        elif self.current_model == "Pseudo-Voigt":
            params_per_peak = 4
            param_names = ["Amplitude", "Center (cm⁻¹)", "Width", "Eta"]
        elif self.current_model == "Asymmetric Voigt":
            params_per_peak = 5
            param_names = ["Amplitude", "Center (cm⁻¹)", "Width L", "Width R", "Eta"]
        else:
            params_per_peak = 3
            param_names = ["Amplitude", "Center (cm⁻¹)", "Width"]
            
        n_peaks = len(self.fit_params) // params_per_peak
        
        for i in range(n_peaks):
            results_text += f"Peak {i+1}:\n"
            start_idx = i * params_per_peak
            for j, param_name in enumerate(param_names):
                param_value = self.fit_params[start_idx + j]
                results_text += f"  {param_name}: {param_value:.3f}\n"
            results_text += "\n"
            
        if hasattr(self, 'r_squared'):
            results_text += f"Fit Quality:\n"
            results_text += f"  R²: {self.r_squared:.4f}\n"
            results_text += f"  RMSE: {self.rmse:.2f}\n"
            results_text += f"  χ²: {self.chi_squared:.2e}\n"
            
        self.results_text.setText(results_text)
        
    def resolve_overlapping_peaks(self):
        """Resolve overlapping peaks using deconvolution."""
        QMessageBox.information(self, "Info", "Overlap resolution feature coming soon!")
        
    def separate_components(self):
        """Separate spectral components."""
        QMessageBox.information(self, "Info", "Component separation feature coming soon!")
        
    def update_plot(self):
        """Update the main plot with current data and fits."""
        if self.wavenumbers is None:
            return
            
        # Clear axes
        self.ax_main.clear()
        self.ax_residuals.clear()
        
        # Plot original spectrum
        self.ax_main.plot(self.wavenumbers, self.processed_intensities, 'b-', 
                         linewidth=1.5, label='Spectrum', alpha=0.8)
        
        # Plot detected peaks
        if self.peaks:
            peak_positions = [self.wavenumbers[i] for i in self.peaks if i < len(self.wavenumbers)]
            peak_intensities = [self.processed_intensities[i] for i in self.peaks if i < len(self.wavenumbers)]
            self.ax_main.scatter(peak_positions, peak_intensities, c='red', s=50, 
                               marker='o', label='Auto Peaks', zorder=5)
                               
        # Plot manual peaks
        if self.manual_peaks:
            manual_positions = [self.wavenumbers[i] for i in self.manual_peaks if i < len(self.wavenumbers)]
            manual_intensities = [self.processed_intensities[i] for i in self.manual_peaks if i < len(self.wavenumbers)]
            self.ax_main.scatter(manual_positions, manual_intensities, c='orange', s=60, 
                               marker='s', label='Manual Peaks', zorder=5)
        
        # Plot fit result
        if self.fit_result is not None:
            self.ax_main.plot(self.wavenumbers, self.fit_result, 'g-', 
                             linewidth=2, label='Fit', alpha=0.9)
                             
        # Plot individual peaks
        if self.show_individual_peaks and self.individual_peaks:
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.individual_peaks)))
            for i, (peak_y, color) in enumerate(zip(self.individual_peaks, colors)):
                self.ax_main.plot(self.wavenumbers, peak_y, '--', 
                                color=color, linewidth=1, alpha=0.7, 
                                label=f'Peak {i+1}')
        
        # Plot residuals
        if self.show_residuals and self.residuals is not None:
            self.ax_residuals.plot(self.wavenumbers, self.residuals, 'r-', 
                                 linewidth=1, alpha=0.8)
            self.ax_residuals.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_residuals.set_ylabel("Residuals")
            
        # Set labels and formatting
        self.ax_main.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_main.set_ylabel("Intensity")
        self.ax_main.set_title("Peak Fitting Analysis", fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend()
        
        self.ax_residuals.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax_residuals.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def get_analysis_results(self):
        """Get current analysis results."""
        results = {
            "model": self.current_model,
            "auto_peaks": self.peaks,
            "manual_peaks": self.manual_peaks,
            "fit_parameters": self.fit_params.tolist() if self.fit_params is not None else None,
            "fit_quality": {
                "r_squared": getattr(self, 'r_squared', None),
                "rmse": getattr(self, 'rmse', None),
                "chi_squared": getattr(self, 'chi_squared', None)
            }
        }
        return results


# Test function for standalone usage
def main():
    """Test the PeakFittingWidget."""
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Create test window
    widget = PeakFittingWidget()
    widget.setWindowTitle("Peak Fitting Module Test")
    widget.resize(1200, 800)
    widget.show()
    
    # Generate test data
    wavenumbers = np.linspace(200, 2000, 1000)
    # Multi-peak synthetic spectrum
    peaks_centers = [400, 700, 1100, 1400]
    peaks_amps = [100, 150, 80, 120]
    peaks_widths = [15, 20, 12, 18]
    
    spectrum = np.zeros_like(wavenumbers)
    for center, amp, width in zip(peaks_centers, peaks_amps, peaks_widths):
        spectrum += amp * np.exp(-(wavenumbers - center)**2 / (2 * width**2))
    
    # Add noise and baseline
    spectrum += np.random.normal(0, 5, len(wavenumbers))
    spectrum += 20 + 0.01 * wavenumbers
    
    # Load test data
    widget.load_spectrum(wavenumbers, spectrum)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 