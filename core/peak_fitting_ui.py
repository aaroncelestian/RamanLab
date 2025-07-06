"""
Unified Peak Fitting UI Components
Shared controls and widgets for peak fitting across all RamanLab tools
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                               QComboBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QPushButton, QGroupBox, QTabWidget, QListWidget,
                               QListWidgetItem, QFrame, QSplitter)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

# Import matplotlib for advanced visualization
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

try:
    from polarization_ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar, apply_theme
    MATPLOTLIB_CONFIG_AVAILABLE = True
except ImportError:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    MATPLOTLIB_CONFIG_AVAILABLE = False

from .peak_fitting import PeakFitter, auto_find_peaks
from scipy.signal import find_peaks


class BackgroundControlsWidget(QWidget):
    """Reusable background subtraction controls"""
    
    background_method_changed = Signal(str)
    parameters_changed = Signal()
    apply_background = Signal()
    reset_spectrum = Signal()
    
    def __init__(self):
        super().__init__()
        self.peak_fitter = PeakFitter()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Background method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems([
            "ALS (Asymmetric Least Squares)",
            "Linear Baseline",
            "Polynomial Fit",
            "Moving Average",
            "Spline Interpolation"
        ])
        self.bg_method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.bg_method_combo)
        layout.addLayout(method_layout)
        
        # Parameter controls (stacked based on method)
        self.create_als_controls(layout)
        self.create_linear_controls(layout)
        self.create_polynomial_controls(layout)
        
        # Note: Action buttons are handled by the parent application
        
    def create_als_controls(self, parent_layout):
        """Create ALS parameter controls with sliders"""
        self.als_group = QGroupBox("ALS Parameters")
        als_layout = QGridLayout(self.als_group)
        
        # Lambda parameter (log scale slider)
        als_layout.addWidget(QLabel("Lambda:"), 0, 0)
        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setRange(3, 6)  # log10 range: 10^3 to 10^6 (more reasonable for ALS)
        self.lambda_slider.setValue(5)  # Default 10^5
        self.lambda_slider.valueChanged.connect(self.on_lambda_changed)
        als_layout.addWidget(self.lambda_slider, 0, 1)
        self.lambda_label = QLabel("1e5")
        als_layout.addWidget(self.lambda_label, 0, 2)
        
        # P parameter
        als_layout.addWidget(QLabel("P:"), 1, 0)
        self.p_slider = QSlider(Qt.Horizontal)
        self.p_slider.setRange(1, 50)  # 0.001 to 0.05, scaled by 1000 (more reasonable range)
        self.p_slider.setValue(10)  # Default 0.01
        self.p_slider.valueChanged.connect(self.on_p_changed)
        als_layout.addWidget(self.p_slider, 1, 1)
        self.p_label = QLabel("0.010")
        als_layout.addWidget(self.p_label, 1, 2)
        
        # Iterations
        als_layout.addWidget(QLabel("Iterations:"), 2, 0)
        self.niter_slider = QSlider(Qt.Horizontal)
        self.niter_slider.setRange(1, 50)
        self.niter_slider.setValue(10)
        self.niter_slider.valueChanged.connect(self.on_niter_changed)
        als_layout.addWidget(self.niter_slider, 2, 1)
        self.niter_label = QLabel("10")
        als_layout.addWidget(self.niter_label, 2, 2)
        
        parent_layout.addWidget(self.als_group)
        
    def create_linear_controls(self, parent_layout):
        """Create linear baseline controls with sliders"""
        self.linear_group = QGroupBox("Linear Parameters")
        linear_layout = QGridLayout(self.linear_group)
        
        # Start weight
        linear_layout.addWidget(QLabel("Start Weight:"), 0, 0)
        self.start_weight_slider = QSlider(Qt.Horizontal)
        self.start_weight_slider.setRange(1, 100)  # 0.1 to 10.0, scaled by 10
        self.start_weight_slider.setValue(10)  # Default 1.0
        self.start_weight_slider.valueChanged.connect(self.on_start_weight_changed)
        linear_layout.addWidget(self.start_weight_slider, 0, 1)
        self.start_weight_label = QLabel("1.0")
        linear_layout.addWidget(self.start_weight_label, 0, 2)
        
        # End weight
        linear_layout.addWidget(QLabel("End Weight:"), 1, 0)
        self.end_weight_slider = QSlider(Qt.Horizontal)
        self.end_weight_slider.setRange(1, 100)  # 0.1 to 10.0, scaled by 10
        self.end_weight_slider.setValue(10)  # Default 1.0
        self.end_weight_slider.valueChanged.connect(self.on_end_weight_changed)
        linear_layout.addWidget(self.end_weight_slider, 1, 1)
        self.end_weight_label = QLabel("1.0")
        linear_layout.addWidget(self.end_weight_label, 1, 2)
        
        parent_layout.addWidget(self.linear_group)
        self.linear_group.hide()  # Hidden by default
        
    def create_polynomial_controls(self, parent_layout):
        """Create polynomial fit controls with sliders"""
        self.poly_group = QGroupBox("Polynomial Parameters")
        poly_layout = QGridLayout(self.poly_group)
        
        # Polynomial order
        poly_layout.addWidget(QLabel("Order:"), 0, 0)
        self.poly_order_slider = QSlider(Qt.Horizontal)
        self.poly_order_slider.setRange(1, 10)
        self.poly_order_slider.setValue(3)
        self.poly_order_slider.valueChanged.connect(self.on_poly_order_changed)
        poly_layout.addWidget(self.poly_order_slider, 0, 1)
        self.poly_order_label = QLabel("3")
        poly_layout.addWidget(self.poly_order_label, 0, 2)
        
        parent_layout.addWidget(self.poly_group)
        self.poly_group.hide()  # Hidden by default
        
    def on_method_changed(self, method_text):
        """Handle background method change"""
        # Hide all parameter groups
        self.als_group.hide()
        self.linear_group.hide()
        self.poly_group.hide()
        
        # Show relevant group
        if "ALS" in method_text:
            self.als_group.show()
        elif "Linear" in method_text:
            self.linear_group.show()
        elif "Polynomial" in method_text:
            self.poly_group.show()
            
        self.background_method_changed.emit(method_text)
        self.parameters_changed.emit()
    
    def on_lambda_changed(self, value):
        """Handle lambda slider change"""
        lambda_val = 10 ** value
        self.lambda_label.setText(f"1e{value}")
        self.parameters_changed.emit()
    
    def on_p_changed(self, value):
        """Handle P slider change"""
        p_val = value / 1000.0
        self.p_label.setText(f"{p_val:.3f}")
        self.parameters_changed.emit()
    
    def on_niter_changed(self, value):
        """Handle iterations slider change"""
        self.niter_label.setText(str(value))
        self.parameters_changed.emit()
    
    def on_start_weight_changed(self, value):
        """Handle start weight slider change"""
        weight_val = value / 10.0
        self.start_weight_label.setText(f"{weight_val:.1f}")
        self.parameters_changed.emit()
    
    def on_end_weight_changed(self, value):
        """Handle end weight slider change"""
        weight_val = value / 10.0
        self.end_weight_label.setText(f"{weight_val:.1f}")
        self.parameters_changed.emit()
    
    def on_poly_order_changed(self, value):
        """Handle polynomial order slider change"""
        self.poly_order_label.setText(str(value))
        self.parameters_changed.emit()
        
    def get_background_parameters(self):
        """Get current background parameters"""
        method = self.bg_method_combo.currentText()
        
        if "ALS" in method:
            return {
                'method': 'ALS',
                'lambda': 10 ** self.lambda_slider.value(),
                'p': self.p_slider.value() / 1000.0,
                'niter': self.niter_slider.value()
            }
        elif "Linear" in method:
            return {
                'method': 'Linear',
                'start_weight': self.start_weight_slider.value() / 10.0,
                'end_weight': self.end_weight_slider.value() / 10.0
            }
        elif "Polynomial" in method:
            return {
                'method': 'Polynomial',
                'order': self.poly_order_slider.value()
            }
        else:
            return {'method': 'ALS', 'lambda': 1e5, 'p': 0.01, 'niter': 10}


class PeakFittingControlsWidget(QWidget):
    """Reusable peak fitting controls"""
    
    model_changed = Signal(str)
    parameters_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Peak Model:"))
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Gaussian",
            "Lorentzian", 
            "Pseudo-Voigt",
            "Voigt",
            "Asymmetric Voigt"
        ])
        self.model_combo.currentTextChanged.connect(self.model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Peak detection parameters
        detection_group = QGroupBox("Peak Detection")
        detection_layout = QGridLayout(detection_group)
        
        # Height threshold
        detection_layout.addWidget(QLabel("Height (%):"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 100)
        self.height_spin.setValue(15)
        self.height_spin.setSuffix("%")
        self.height_spin.valueChanged.connect(self.parameters_changed)
        detection_layout.addWidget(self.height_spin, 0, 1)
        
        # Distance threshold
        detection_layout.addWidget(QLabel("Distance:"), 1, 0)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(1, 100)
        self.distance_spin.setValue(20)
        self.distance_spin.valueChanged.connect(self.parameters_changed)
        detection_layout.addWidget(self.distance_spin, 1, 1)
        
        # Prominence threshold
        detection_layout.addWidget(QLabel("Prominence (%):"), 2, 0)
        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(1, 100)
        self.prominence_spin.setValue(20)
        self.prominence_spin.setSuffix("%")
        self.prominence_spin.valueChanged.connect(self.parameters_changed)
        detection_layout.addWidget(self.prominence_spin, 2, 1)
        
        layout.addWidget(detection_group)
        
    def get_peak_parameters(self):
        """Get current peak fitting parameters"""
        return {
            'model': self.model_combo.currentText(),
            'height': self.height_spin.value() / 100.0,  # Convert to fraction
            'distance': self.distance_spin.value(),
            'prominence': self.prominence_spin.value() / 100.0  # Convert to fraction
        }


class AdvancedPeakFittingWidget(QWidget):
    """
    Advanced peak fitting widget with interactive visualization and manual peak selection.
    Incorporates all the advanced features from peak_fitting_qt6.py into a reusable component.
    """
    
    # Signals for communication with main applications
    background_calculated = Signal(object)  # np.ndarray
    peaks_detected = Signal(list)  # Peak positions
    peaks_fitted = Signal(dict)  # Fitting results
    spectrum_updated = Signal()  # When spectrum is modified
    
    def __init__(self):
        super().__init__()
        
        # Apply matplotlib theme if available
        if MATPLOTLIB_CONFIG_AVAILABLE:
            apply_theme('compact')
        
        # Initialize components
        self.peak_fitter = PeakFitter()
        self.setup_ui()
        
        # Data storage
        self.current_wavenumbers = None
        self.current_intensities = None
        self.original_intensities = None
        self.background = None
        self.detected_peaks = np.array([])
        self.manual_peaks = np.array([])
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        
        # Interactive state
        self.interactive_mode = False
        self.click_connection = None
        
        # Plot references for efficient updates
        self.spectrum_line = None
        self.background_line = None
        self.fitted_line = None
        self.auto_peaks_scatter = None
        self.manual_peaks_scatter = None
        self.individual_peak_lines = []
        
        # Update timers for responsive interaction
        self.peak_update_timer = QTimer()
        self.peak_update_timer.setSingleShot(True)
        self.peak_update_timer.timeout.connect(self._update_peak_detection)
        
    def setup_ui(self):
        """Set up the user interface with controls and visualization."""
        main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - controls
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        # Right panel - visualization
        viz_panel = self._create_visualization_panel()
        splitter.addWidget(viz_panel)
        
        # Set splitter proportions (30% controls, 70% visualization)
        splitter.setSizes([400, 1000])
        
    def _create_control_panel(self):
        """Create the control panel with tabs."""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Background controls tab
        self.background_controls = BackgroundControlsWidget()
        self.background_controls.parameters_changed.connect(self._on_background_changed)
        tabs.addTab(self.background_controls, "Background")
        
        # Peak detection tab
        peak_tab = self._create_peak_detection_tab()
        tabs.addTab(peak_tab, "Peak Detection")
        
        # Peak fitting tab
        fitting_tab = self._create_peak_fitting_tab()
        tabs.addTab(fitting_tab, "Peak Fitting")
        
        layout.addWidget(tabs)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.apply_background_btn = QPushButton("Apply Background")
        self.apply_background_btn.clicked.connect(self.apply_background)
        action_layout.addWidget(self.apply_background_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_spectrum)
        action_layout.addWidget(self.reset_btn)
        
        layout.addLayout(action_layout)
        
        return panel
        
    def _create_peak_detection_tab(self):
        """Create the peak detection tab with interactive features."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Peak detection parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QGridLayout(params_group)
        
        # Height threshold slider
        params_layout.addWidget(QLabel("Height (%):"), 0, 0)
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(1, 100)
        self.height_slider.setValue(15)
        self.height_slider.valueChanged.connect(self._trigger_peak_update)
        params_layout.addWidget(self.height_slider, 0, 1)
        
        self.height_label = QLabel("15%")
        self.height_slider.valueChanged.connect(lambda v: self.height_label.setText(f"{v}%"))
        params_layout.addWidget(self.height_label, 0, 2)
        
        # Distance threshold slider
        params_layout.addWidget(QLabel("Distance:"), 1, 0)
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(10)
        self.distance_slider.valueChanged.connect(self._trigger_peak_update)
        params_layout.addWidget(self.distance_slider, 1, 1)
        
        self.distance_label = QLabel("10")
        self.distance_slider.valueChanged.connect(lambda v: self.distance_label.setText(str(v)))
        params_layout.addWidget(self.distance_label, 1, 2)
        
        # Prominence threshold slider
        params_layout.addWidget(QLabel("Prominence (%):"), 2, 0)
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 50)
        self.prominence_slider.setValue(5)
        self.prominence_slider.valueChanged.connect(self._trigger_peak_update)
        params_layout.addWidget(self.prominence_slider, 2, 1)
        
        self.prominence_label = QLabel("5%")
        self.prominence_slider.valueChanged.connect(lambda v: self.prominence_label.setText(f"{v}%"))
        params_layout.addWidget(self.prominence_label, 2, 2)
        
        layout.addWidget(params_group)
        
        # Interactive peak selection
        interactive_group = QGroupBox("Interactive Peak Selection")
        interactive_layout = QVBoxLayout(interactive_group)
        
        # Interactive mode toggle
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
        """)
        interactive_layout.addWidget(self.interactive_btn)
        
        # Peak management buttons
        peak_buttons_layout = QHBoxLayout()
        
        clear_auto_btn = QPushButton("Clear Auto Peaks")
        clear_auto_btn.clicked.connect(self.clear_auto_peaks)
        peak_buttons_layout.addWidget(clear_auto_btn)
        
        clear_manual_btn = QPushButton("Clear Manual")
        clear_manual_btn.clicked.connect(self.clear_manual_peaks)
        peak_buttons_layout.addWidget(clear_manual_btn)
        
        interactive_layout.addLayout(peak_buttons_layout)
        
        layout.addWidget(interactive_group)
        
        # Peak list widget
        list_group = QGroupBox("Current Peaks")
        list_layout = QVBoxLayout(list_group)
        
        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setMaximumHeight(120)
        list_layout.addWidget(self.peak_list_widget)
        
        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.clicked.connect(self.remove_selected_peak)
        list_layout.addWidget(remove_selected_btn)
        
        layout.addWidget(list_group)
        
        # Peak count status
        self.peak_count_label = QLabel("Auto: 0 | Manual: 0 | Total: 0")
        self.peak_count_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(self.peak_count_label)
        
        layout.addStretch()
        return tab
        
    def _create_peak_fitting_tab(self):
        """Create the peak fitting tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_group = QGroupBox("Peak Model")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Gaussian", "Lorentzian", "Pseudo-Voigt", "Voigt", "Asymmetric Voigt"])
        model_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_individual_check = QCheckBox("Show Individual Peaks")
        self.show_individual_check.setChecked(True)
        self.show_individual_check.toggled.connect(self.update_plot)
        display_layout.addWidget(self.show_individual_check)
        
        self.show_legend_check = QCheckBox("Show Legend")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.toggled.connect(self.update_plot)
        display_layout.addWidget(self.show_legend_check)
        
        layout.addWidget(display_group)
        
        # Fitting actions
        fit_group = QGroupBox("Peak Fitting")
        fit_layout = QVBoxLayout(fit_group)
        
        self.fit_peaks_btn = QPushButton("Fit Peaks")
        self.fit_peaks_btn.clicked.connect(self.fit_peaks)
        self.fit_peaks_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        fit_layout.addWidget(self.fit_peaks_btn)
        
        layout.addWidget(fit_group)
        
        layout.addStretch()
        return tab
        
    def _create_visualization_panel(self):
        """Create the visualization panel with matplotlib."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create matplotlib figure with subplots
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Create toolbar
        if MATPLOTLIB_CONFIG_AVAILABLE:
            self.toolbar = NavigationToolbar(self.canvas, panel)
        else:
            self.toolbar = NavigationToolbar(self.canvas, panel)
        
        # Create subplots - main plot and residuals
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        self.ax_main = self.figure.add_subplot(gs[0])
        self.ax_residual = self.figure.add_subplot(gs[1])
        
        self.figure.tight_layout(pad=3.0)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        return panel
        
    def set_spectrum_data(self, wavenumbers, intensities):
        """Set the spectrum data for analysis."""
        self.current_wavenumbers = np.array(wavenumbers)
        self.original_intensities = np.array(intensities)
        self.current_intensities = self.original_intensities.copy()
        
        # Reset analysis state
        self.background = None
        self.detected_peaks = np.array([])
        self.manual_peaks = np.array([])
        self.fit_params = []
        self.fit_result = None
        self.residuals = None
        
        # Update plot
        self.update_plot()
        
        # Auto-detect peaks
        self._trigger_peak_update()
        
    def _trigger_peak_update(self):
        """Trigger debounced peak detection update."""
        self.peak_update_timer.stop()
        self.peak_update_timer.start(100)  # 100ms delay
        
    def _update_peak_detection(self):
        """Perform peak detection with current parameters."""
        if self.current_wavenumbers is None or self.current_intensities is None:
            return
            
        try:
            # Get parameters
            height_percent = self.height_slider.value()
            distance = self.distance_slider.value()
            prominence_percent = self.prominence_slider.value()
            
            # Calculate thresholds
            max_intensity = float(np.max(self.current_intensities))
            height_threshold = (height_percent / 100.0) * max_intensity if height_percent > 0 else None
            prominence_threshold = (prominence_percent / 100.0) * max_intensity if prominence_percent > 0 else None
            
            # Find peaks
            peak_kwargs = {}
            if height_threshold is not None:
                peak_kwargs['height'] = height_threshold
            if distance > 0:
                peak_kwargs['distance'] = int(distance)
            if prominence_threshold is not None:
                peak_kwargs['prominence'] = prominence_threshold
            
            peaks, _ = find_peaks(self.current_intensities, **peak_kwargs)
            self.detected_peaks = peaks
            
            # Update display
            self._update_peak_count_display()
            self._update_peak_list()
            self._update_peak_markers()
            
            # Emit signal
            peak_positions = [self.current_wavenumbers[p] for p in peaks if p < len(self.current_wavenumbers)]
            self.peaks_detected.emit(peak_positions)
            
        except Exception as e:
            print(f"Peak detection error: {str(e)}")
            
    def toggle_interactive_mode(self):
        """Toggle interactive peak selection mode."""
        self.interactive_mode = self.interactive_btn.isChecked()
        
        if self.interactive_mode:
            self.interactive_btn.setText("🖱️ Disable Interactive Selection")
            # Connect mouse click event
            self.click_connection = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        else:
            self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            # Disconnect mouse click event
            if self.click_connection is not None:
                self.canvas.mpl_disconnect(self.click_connection)
                self.click_connection = None
                
        self.update_plot()
        
    def _on_canvas_click(self, event):
        """Handle canvas clicks for interactive peak selection."""
        if not self.interactive_mode or event.inaxes != self.ax_main:
            return
            
        if event.button != 1:  # Only left clicks
            return
            
        click_x = event.xdata
        if click_x is None:
            return
            
        # Find closest data point
        closest_idx = np.argmin(np.abs(self.current_wavenumbers - click_x))
        
        # Check if clicking near existing peak to remove it
        removal_threshold = 20  # wavenumber units
        
        # Check manual peaks first
        removed = False
        if len(self.manual_peaks) > 0:
            for i, peak_idx in enumerate(self.manual_peaks):
                peak_wavenumber = self.current_wavenumbers[peak_idx]
                if abs(peak_wavenumber - click_x) < removal_threshold:
                    self.manual_peaks = np.delete(self.manual_peaks, i)
                    removed = True
                    break
        
        # Check auto peaks
        if not removed and len(self.detected_peaks) > 0:
            for i, peak_idx in enumerate(self.detected_peaks):
                peak_wavenumber = self.current_wavenumbers[peak_idx]
                if abs(peak_wavenumber - click_x) < removal_threshold:
                    self.detected_peaks = np.delete(self.detected_peaks, i)
                    removed = True
                    break
        
        # If not removing, add new manual peak
        if not removed:
            self.manual_peaks = np.append(self.manual_peaks, closest_idx)
            
        # Update display
        self._update_peak_count_display()
        self._update_peak_list()
        self._update_peak_markers()
        
    def get_all_peaks(self):
        """Get all peaks (auto + manual) for fitting."""
        all_peaks = []
        if len(self.detected_peaks) > 0:
            all_peaks.extend(self.detected_peaks.tolist())
        if len(self.manual_peaks) > 0:
            all_peaks.extend(self.manual_peaks.tolist())
        return sorted(list(set(all_peaks)))  # Remove duplicates and sort
        
    def fit_peaks(self):
        """Fit peaks to the spectrum."""
        all_peaks = self.get_all_peaks()
        if len(all_peaks) == 0:
            return
            
        try:
            # Convert peak indices to positions
            peak_positions = [self.current_wavenumbers[p] for p in all_peaks if p < len(self.current_wavenumbers)]
            
            # Use centralized peak fitting
            model = self.model_combo.currentText()
            fitted_peaks = self.peak_fitter.fit_multiple_peaks(
                self.current_wavenumbers,
                self.current_intensities,
                peak_positions,
                shape=model
            )
            
            # Store results for plotting
            self.fit_result = fitted_peaks
            self.fit_params = []
            
            # Extract parameters for plotting based on model type
            for peak in fitted_peaks:
                if model == "Gaussian" or model == "Lorentzian":
                    self.fit_params.extend([peak.amplitude, peak.position, peak.width])
                elif model == "Pseudo-Voigt":
                    # Use eta=0.5 as default for pseudo-Voigt if not available
                    eta = 0.5 if len(peak.parameters) < 4 else peak.parameters[3]
                    self.fit_params.extend([peak.amplitude, peak.position, peak.width, eta])
                elif model == "Voigt":
                    gamma = peak.gamma if peak.gamma is not None else peak.width / 2
                    self.fit_params.extend([peak.amplitude, peak.position, peak.width, gamma])
                elif model == "Asymmetric Voigt":
                    gamma = peak.gamma if peak.gamma is not None else peak.width / 2
                    alpha = peak.alpha if peak.alpha is not None else 0.0
                    self.fit_params.extend([peak.amplitude, peak.position, peak.width, gamma, alpha])
                else:
                    # Fallback to basic parameters
                    self.fit_params.extend([peak.amplitude, peak.position, peak.width])
            
            # Calculate residuals
            if len(self.fit_params) > 0:
                fitted_curve = self._multi_peak_model(self.current_wavenumbers, *self.fit_params)
                self.residuals = self.current_intensities - fitted_curve
                
            # Update plot
            self.update_plot()
            
            # Emit signal
            fit_result = {
                'fitted_peaks': fitted_peaks,
                'model': model,
                'n_peaks': len(fitted_peaks),
                'peak_positions': peak_positions,
                'fit_params': self.fit_params,
                'residuals': self.residuals
            }
            self.peaks_fitted.emit(fit_result)
            
        except Exception as e:
            print(f"Peak fitting error: {str(e)}")
            
    def _multi_peak_model(self, x, *params):
        """Multi-peak model for plotting fitted curves."""
        if len(params) == 0:
            return np.zeros_like(x)
            
        model = np.zeros_like(x)
        current_model = self.model_combo.currentText()
        
        # Determine parameters per peak based on model
        if current_model in ["Gaussian", "Lorentzian", "Pseudo-Voigt"]:
            params_per_peak = 3 if current_model != "Pseudo-Voigt" else 4
        elif current_model == "Voigt":
            params_per_peak = 4
        elif current_model == "Asymmetric Voigt":
            params_per_peak = 5
        else:
            params_per_peak = 3  # Default fallback
        
        n_peaks = len(params) // params_per_peak
        
        for i in range(n_peaks):
            start_idx = i * params_per_peak
            if start_idx + params_per_peak - 1 < len(params):
                
                if current_model == "Gaussian":
                    amp, cen, wid = params[start_idx:start_idx+3]
                    component = self.peak_fitter.gaussian(x, amp, cen, wid)
                elif current_model == "Lorentzian":
                    amp, cen, wid = params[start_idx:start_idx+3]
                    component = self.peak_fitter.lorentzian(x, amp, cen, wid)
                elif current_model == "Pseudo-Voigt":
                    amp, cen, wid, eta = params[start_idx:start_idx+4]
                    component = self.peak_fitter.pseudo_voigt(x, amp, cen, wid, eta)
                elif current_model == "Voigt":
                    amp, cen, sigma, gamma = params[start_idx:start_idx+4]
                    component = self.peak_fitter.voigt(x, amp, cen, sigma, gamma)
                elif current_model == "Asymmetric Voigt":
                    amp, cen, sigma, gamma, alpha = params[start_idx:start_idx+5]
                    component = self.peak_fitter.asymmetric_voigt(x, amp, cen, sigma, gamma, alpha)
                else:
                    # Fallback to Gaussian
                    amp, cen, wid = params[start_idx:start_idx+3]
                    component = self.peak_fitter.gaussian(x, amp, cen, wid)
                    
                model += component
                
        return model
        
    def update_plot(self):
        """Update the visualization plot."""
        if self.current_wavenumbers is None or self.current_intensities is None:
            return
            
        # Clear plots
        self.ax_main.clear()
        self.ax_residual.clear()
        
        # Plot main spectrum
        self.ax_main.plot(self.current_wavenumbers, self.current_intensities, 'b-', 
                         linewidth=1.5, label='Spectrum')
        
        # Plot background if available
        if self.background is not None:
            self.ax_main.plot(self.current_wavenumbers, self.background, 'r--', 
                             linewidth=1, alpha=0.7, label='Background')
        
        # Plot detected peaks (red dots)
        if len(self.detected_peaks) > 0:
            valid_peaks = [p for p in self.detected_peaks if p < len(self.current_wavenumbers)]
            if valid_peaks:
                peak_x = [self.current_wavenumbers[p] for p in valid_peaks]
                peak_y = [self.current_intensities[p] for p in valid_peaks]
                self.ax_main.scatter(peak_x, peak_y, c='red', s=64, marker='o', 
                                   label='Auto Peaks', alpha=0.8, zorder=5)
        
        # Plot manual peaks (green squares)
        if len(self.manual_peaks) > 0:
            valid_manual = [p for p in self.manual_peaks if p < len(self.current_wavenumbers)]
            if valid_manual:
                manual_x = [self.current_wavenumbers[p] for p in valid_manual]
                manual_y = [self.current_intensities[p] for p in valid_manual]
                self.ax_main.scatter(manual_x, manual_y, c='green', s=100, marker='s', 
                                   label='Manual Peaks', alpha=0.8, edgecolor='darkgreen', zorder=5)
        
        # Plot fitted peaks if available
        if self.fit_result is not None and len(self.fit_params) > 0 and self.show_individual_check.isChecked():
            # Plot total fit
            fitted_curve = self._multi_peak_model(self.current_wavenumbers, *self.fit_params)
            total_r2 = self._calculate_total_r2(fitted_curve)
            self.ax_main.plot(self.current_wavenumbers, fitted_curve, 'g-', 
                             linewidth=2, label=f'Total Fit (R²={total_r2:.4f})')
            
            # Plot individual peaks with colors and R² values
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                     '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
            
            current_model = self.model_combo.currentText()
            
            # Determine parameters per peak
            if current_model in ["Gaussian", "Lorentzian"]:
                params_per_peak = 3
            elif current_model == "Pseudo-Voigt":
                params_per_peak = 4
            elif current_model == "Voigt":
                params_per_peak = 4
            elif current_model == "Asymmetric Voigt":
                params_per_peak = 5
            else:
                params_per_peak = 3
            
            n_peaks = len(self.fit_params) // params_per_peak
            for i in range(n_peaks):
                start_idx = i * params_per_peak
                if start_idx + params_per_peak - 1 < len(self.fit_params):
                    
                    # Generate individual peak based on model
                    if current_model == "Gaussian":
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        peak_curve = self.peak_fitter.gaussian(self.current_wavenumbers, amp, cen, wid)
                    elif current_model == "Lorentzian":
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        peak_curve = self.peak_fitter.lorentzian(self.current_wavenumbers, amp, cen, wid)
                    elif current_model == "Pseudo-Voigt":
                        amp, cen, wid, eta = self.fit_params[start_idx:start_idx+4]
                        peak_curve = self.peak_fitter.pseudo_voigt(self.current_wavenumbers, amp, cen, wid, eta)
                    elif current_model == "Voigt":
                        amp, cen, sigma, gamma = self.fit_params[start_idx:start_idx+4]
                        peak_curve = self.peak_fitter.voigt(self.current_wavenumbers, amp, cen, sigma, gamma)
                    elif current_model == "Asymmetric Voigt":
                        amp, cen, sigma, gamma, alpha = self.fit_params[start_idx:start_idx+5]
                        peak_curve = self.peak_fitter.asymmetric_voigt(self.current_wavenumbers, amp, cen, sigma, gamma, alpha)
                    else:
                        # Fallback
                        amp, cen, wid = self.fit_params[start_idx:start_idx+3]
                        peak_curve = self.peak_fitter.gaussian(self.current_wavenumbers, amp, cen, wid)
                    
                    # Calculate individual R²
                    individual_r2 = self._calculate_individual_r2(i, peak_curve, fitted_curve)
                    
                    # Create label with asymmetry info for asymmetric Voigt
                    if current_model == "Asymmetric Voigt":
                        label = f'Peak {i+1} ({cen:.1f} cm⁻¹, α={alpha:.3f}, R²={individual_r2:.3f})'
                    else:
                        label = f'Peak {i+1} ({cen:.1f} cm⁻¹, R²={individual_r2:.3f})'
                    
                    color = colors[i % len(colors)]
                    self.ax_main.plot(self.current_wavenumbers, peak_curve, '--', 
                                     linewidth=1.5, alpha=0.8, color=color,
                                     label=label)
                    
                    # Add peak position label
                    peak_max_idx = np.argmax(peak_curve)
                    self.ax_main.annotate(f'{cen:.1f}', 
                                        xy=(cen, peak_curve[peak_max_idx]),
                                        xytext=(cen, peak_curve[peak_max_idx] + np.max(self.current_intensities) * 0.05),
                                        ha='center', va='bottom',
                                        fontsize=9, fontweight='bold',
                                        color=color,
                                        bbox=dict(boxstyle='round,pad=0.2', 
                                                facecolor='white', 
                                                edgecolor=color, 
                                                alpha=0.8))
        
        # Set labels and title
        self.ax_main.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.set_title('Spectrum and Peak Analysis')
        self.ax_main.grid(True, alpha=0.3)
        
        # Show legend if enabled
        if self.show_legend_check.isChecked():
            self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add interactive mode indicator
        if self.interactive_mode:
            self.ax_main.text(0.02, 0.98, '🖱️ Interactive Mode ON\nClick to select peaks', 
                             transform=self.ax_main.transAxes, 
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Plot residuals if available
        if self.residuals is not None:
            self.ax_residual.plot(self.current_wavenumbers, self.residuals, 'k-', linewidth=1)
            self.ax_residual.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            self.ax_residual.set_xlabel('Wavenumber (cm⁻¹)')
            self.ax_residual.set_ylabel('Residuals')
            self.ax_residual.set_title('Fit Residuals')
            self.ax_residual.grid(True, alpha=0.3)
        
        # Update canvas
        self.canvas.draw()
        
    def _calculate_total_r2(self, fitted_curve):
        """Calculate total R² for the fit."""
        ss_res = np.sum((self.current_intensities - fitted_curve) ** 2)
        ss_tot = np.sum((self.current_intensities - np.mean(self.current_intensities)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
    def _calculate_individual_r2(self, peak_index, peak_curve, total_fit):
        """Calculate R² for individual peak."""
        try:
            # Determine parameters per peak based on current model
            current_model = self.model_combo.currentText()
            if current_model in ["Gaussian", "Lorentzian"]:
                params_per_peak = 3
            elif current_model == "Pseudo-Voigt":
                params_per_peak = 4
            elif current_model == "Voigt":
                params_per_peak = 4
            elif current_model == "Asymmetric Voigt":
                params_per_peak = 5
            else:
                params_per_peak = 3
            
            # Simple regional R² calculation
            start_idx = peak_index * params_per_peak
            if start_idx + params_per_peak - 1 >= len(self.fit_params):
                return 0.0
                
            # Extract center position (always the second parameter)
            cen = self.fit_params[start_idx + 1]
            
            # For width, use the appropriate parameter based on model
            if current_model in ["Gaussian", "Lorentzian"]:
                wid = self.fit_params[start_idx + 2]
            elif current_model == "Pseudo-Voigt":
                wid = self.fit_params[start_idx + 2] 
            elif current_model in ["Voigt", "Asymmetric Voigt"]:
                # For Voigt profiles, use sigma (3rd parameter) for width estimation
                wid = self.fit_params[start_idx + 2]
            else:
                wid = self.fit_params[start_idx + 2]
            
            # Define region around peak
            region_mask = (self.current_wavenumbers >= cen - abs(wid) * 3) & \
                         (self.current_wavenumbers <= cen + abs(wid) * 3)
            
            if not np.any(region_mask):
                return 0.0
                
            region_data = self.current_intensities[region_mask]
            region_peak = peak_curve[region_mask]
            
            # Calculate R² for this peak in its region
            mean_data = np.mean(region_data)
            ss_tot = np.sum((region_data - mean_data) ** 2)
            
            if ss_tot > 0:
                # Compare peak to baseline in region
                baseline = np.linspace(region_data[0], region_data[-1], len(region_data))
                region_corrected = region_data - baseline
                
                # Scale peak to match data
                if np.max(region_peak) > 0:
                    peak_scaled = region_peak * (np.max(region_corrected) / np.max(region_peak))
                    ss_res = np.sum((region_corrected - peak_scaled) ** 2)
                    r2 = 1 - (ss_res / ss_tot)
                    return max(0.0, min(1.0, r2))
                    
            return 0.0
            
        except Exception:
            return 0.0
        
    def _update_peak_markers(self):
        """Update peak markers on the plot efficiently."""
        self.update_plot()  # For now, do full update - can optimize later
        
    def _update_peak_count_display(self):
        """Update the peak count display."""
        auto_count = len(self.detected_peaks)
        manual_count = len(self.manual_peaks)
        total_count = auto_count + manual_count
        self.peak_count_label.setText(f"Auto: {auto_count} | Manual: {manual_count} | Total: {total_count}")
        
    def _update_peak_list(self):
        """Update the peak list widget."""
        self.peak_list_widget.clear()
        
        # Add auto peaks
        for i, peak_idx in enumerate(self.detected_peaks):
            if peak_idx < len(self.current_wavenumbers):
                wavenumber = self.current_wavenumbers[peak_idx]
                intensity = self.current_intensities[peak_idx]
                item_text = f"🔴 Auto {i+1}: {wavenumber:.1f} cm⁻¹ (I: {intensity:.1f})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, ('auto', peak_idx))
                self.peak_list_widget.addItem(item)
        
        # Add manual peaks
        for i, peak_idx in enumerate(self.manual_peaks):
            if peak_idx < len(self.current_wavenumbers):
                wavenumber = self.current_wavenumbers[peak_idx]
                intensity = self.current_intensities[peak_idx]
                item_text = f"🟢 Manual {i+1}: {wavenumber:.1f} cm⁻¹ (I: {intensity:.1f})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, ('manual', peak_idx))
                self.peak_list_widget.addItem(item)
                
    def remove_selected_peak(self):
        """Remove the selected peak from the list."""
        current_item = self.peak_list_widget.currentItem()
        if current_item is None:
            return
            
        peak_data = current_item.data(Qt.UserRole)
        if peak_data is None:
            return
            
        peak_type, peak_idx = peak_data
        
        if peak_type == 'auto':
            self.detected_peaks = self.detected_peaks[self.detected_peaks != peak_idx]
        elif peak_type == 'manual':
            self.manual_peaks = self.manual_peaks[self.manual_peaks != peak_idx]
            
        self._update_peak_count_display()
        self._update_peak_list()
        self.update_plot()
        
    def clear_auto_peaks(self):
        """Clear automatically detected peaks."""
        self.detected_peaks = np.array([])
        self._update_peak_count_display()
        self._update_peak_list()
        self.update_plot()
        
    def clear_manual_peaks(self):
        """Clear manually selected peaks."""
        self.manual_peaks = np.array([])
        self._update_peak_count_display()
        self._update_peak_list()
        self.update_plot()
        
    def _on_background_changed(self):
        """Handle background parameter changes."""
        if self.current_wavenumbers is None or self.current_intensities is None:
            return
            
        # Calculate background
        params = self.background_controls.get_background_parameters()
        
        if params['method'] == 'ALS':
            background = PeakFitter.baseline_als(
                self.original_intensities,
                lam=params['lambda'],
                p=params['p'],
                niter=params['niter']
            )
        else:
            # Simple fallback for other methods
            background = np.linspace(
                self.original_intensities[0],
                self.original_intensities[-1],
                len(self.original_intensities)
            )
            
        self.background = background
        self.update_plot()
        self.background_calculated.emit(background)
        
    def apply_background(self):
        """Apply background subtraction to the spectrum."""
        if self.background is not None:
            self.current_intensities = self.original_intensities - self.background
            self.background = None  # Clear background preview
            self._trigger_peak_update()  # Re-detect peaks on corrected spectrum
            self.update_plot()
            self.spectrum_updated.emit()
            
    def reset_spectrum(self):
        """Reset spectrum to original state."""
        if self.original_intensities is not None:
            self.current_intensities = self.original_intensities.copy()
            self.background = None
            self.detected_peaks = np.array([])
            self.manual_peaks = np.array([])
            self.fit_params = []
            self.fit_result = None
            self.residuals = None
            
            # Disable interactive mode
            if self.interactive_mode:
                self.interactive_mode = False
                self.interactive_btn.setChecked(False)
                self.interactive_btn.setText("🖱️ Enable Interactive Selection")
                if self.click_connection is not None:
                    self.canvas.mpl_disconnect(self.click_connection)
                    self.click_connection = None
                    
            self._update_peak_count_display()
            self._update_peak_list()
            self.update_plot()
            self.spectrum_updated.emit() 