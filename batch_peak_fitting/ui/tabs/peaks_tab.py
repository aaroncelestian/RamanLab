"""
Peaks Tab Component - Enhanced with Stacked Tab Interface
Matches the elegant design from peak_fitting_qt6.py with live preview capabilities
Handles peak detection, fitting parameters, and background subtraction controls
"""

import numpy as np
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, QWidget, QFrame,
    QPushButton, QLabel, QSlider, QComboBox, QSpinBox, QGridLayout,
    QFormLayout, QCheckBox, QListWidget, QListWidgetItem, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from ..base_tab import BaseTab

# Import enhanced peak fitting components if available
try:
    from core.peak_fitting_ui import BackgroundControlsWidget, PeakFittingControlsWidget
    CENTRALIZED_UI_AVAILABLE = True
except ImportError:
    CENTRALIZED_UI_AVAILABLE = False

# Import matplotlib config for consistent theming
try:
    from polarization_ui.matplotlib_config import apply_theme
    apply_theme('compact')
    MATPLOTLIB_CONFIG_AVAILABLE = True
except ImportError:
    MATPLOTLIB_CONFIG_AVAILABLE = False


class StackedTabWidget(QWidget):
    """Custom tab widget with stacked tab buttons like peak_fitting_qt6.py"""
    
    def __init__(self):
        super().__init__()
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


class PeaksTab(BaseTab):
    """
    Enhanced Peak detection and fitting controls component.
    Uses the elegant stacked tab design from peak_fitting_qt6.py
    """
    
    def __init__(self, parent=None):
        # Initialize key widgets
        self.bg_method_combo = None
        self.height_slider = None
        self.distance_slider = None
        self.prominence_slider = None
        self.model_combo = None
        
        # ALS parameter widgets
        self.lambda_slider = None
        self.p_slider = None
        self.niter_slider = None
        
        # Label widgets for live updates
        self.height_label = None
        self.distance_label = None
        self.prominence_label = None
        self.lambda_label = None
        self.p_label = None
        self.niter_label = None
        self.peak_count_label = None
        
        # Interactive peak selection
        self.interactive_mode = False
        self.peak_list_widget = None
        self.interactive_btn = None
        self.interactive_status_label = None
        
        # Live preview timers for responsive updates
        self.bg_update_timer = QTimer()
        self.bg_update_timer.setSingleShot(True)
        self.bg_update_timer.timeout.connect(self._update_background_preview)
        self.bg_update_delay = 150  # milliseconds
        
        self.peak_update_timer = QTimer()
        self.peak_update_timer.setSingleShot(True)
        self.peak_update_timer.timeout.connect(self._update_peak_detection)
        self.peak_update_delay = 100  # milliseconds
        
        super().__init__(parent)
        self.tab_name = "Peaks"
    
    def setup_ui(self):
        """Create the elegant stacked tab UI matching peak_fitting_qt6.py"""
        # Create stacked tab widget
        self.stacked_tabs = StackedTabWidget()
        self.main_layout.addWidget(self.stacked_tabs)
        
        # Create tabs with 2-row layout like peak_fitting_qt6.py:
        # Row 1: Background, Peak Detection, Peak Fitting
        # Row 2: Analysis, Results
        self.stacked_tabs.add_tab(self._create_background_tab(), "Background", row=0, col=0)
        self.stacked_tabs.add_tab(self._create_peak_detection_tab(), "Peak Detection", row=0, col=1)
        self.stacked_tabs.add_tab(self._create_fitting_tab(), "Peak Fitting", row=0, col=2)
        self.stacked_tabs.add_tab(self._create_analysis_tab(), "Analysis", row=1, col=0)
        self.stacked_tabs.add_tab(self._create_results_tab(), "Results", row=1, col=1)
    
    def _create_background_tab(self):
        """Create background subtraction tab with ALS live preview"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Method selection
        method_group = QGroupBox("Background Method")
        method_layout = QVBoxLayout(method_group)
        
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems([
            "ALS (Asymmetric Least Squares)",
            "Linear", 
            "Polynomial",
            "Moving Average"
        ])
        self.bg_method_combo.currentTextChanged.connect(self._on_bg_method_changed)
        method_layout.addWidget(self.bg_method_combo)
        
        layout.addWidget(method_group)
        
        # ALS Parameters with live preview
        self.als_group = QGroupBox("ALS Parameters")
        als_layout = QFormLayout(self.als_group)
        
        # Lambda (smoothness) - log scale slider
        lambda_layout = QHBoxLayout()
        self.lambda_slider = QSlider(Qt.Horizontal)
        self.lambda_slider.setRange(3, 7)  # 10^3 to 10^7
        self.lambda_slider.setValue(5)  # 10^5 default
        self.lambda_slider.valueChanged.connect(self._update_lambda_label)
        self.lambda_slider.valueChanged.connect(self._trigger_background_update)
        
        self.lambda_label = QLabel("1e5")
        lambda_layout.addWidget(self.lambda_slider)
        lambda_layout.addWidget(self.lambda_label)
        als_layout.addRow("Lambda (smoothness):", lambda_layout)
        
        # P (asymmetry) - linear scale
        p_layout = QHBoxLayout()
        self.p_slider = QSlider(Qt.Horizontal)
        self.p_slider.setRange(1, 50)  # 0.001 to 0.050
        self.p_slider.setValue(10)  # 0.01 default
        self.p_slider.valueChanged.connect(self._update_p_label)
        self.p_slider.valueChanged.connect(self._trigger_background_update)
        
        self.p_label = QLabel("0.010")
        p_layout.addWidget(self.p_slider)
        p_layout.addWidget(self.p_label)
        als_layout.addRow("P (asymmetry):", p_layout)
        
        # Iterations
        niter_layout = QHBoxLayout()
        self.niter_slider = QSlider(Qt.Horizontal)
        self.niter_slider.setRange(5, 30)
        self.niter_slider.setValue(10)
        self.niter_slider.valueChanged.connect(self._update_niter_label)
        self.niter_slider.valueChanged.connect(self._trigger_background_update)
        
        self.niter_label = QLabel("10")
        niter_layout.addWidget(self.niter_slider)
        niter_layout.addWidget(self.niter_label)
        als_layout.addRow("Iterations:", niter_layout)
        
        layout.addWidget(self.als_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("""
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
        apply_btn.clicked.connect(self._apply_background)
        button_layout.addWidget(apply_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_background)
        button_layout.addWidget(clear_btn)
        
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_spectrum)
        button_layout.addWidget(reset_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        # Initialize labels
        self._update_lambda_label()
        self._update_p_label()
        self._update_niter_label()
        
        return tab
    
    def _create_peak_detection_tab(self):
        """Create peak detection tab with live preview"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Automatic Peak Detection
        auto_group = QGroupBox("Automatic Peak Detection")
        auto_layout = QFormLayout(auto_group)
        
        # Height threshold
        height_layout = QHBoxLayout()
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(1, 100)
        self.height_slider.setValue(15)  # 15% default
        self.height_slider.valueChanged.connect(self._update_height_label)
        self.height_slider.valueChanged.connect(self._trigger_peak_update)
        
        self.height_label = QLabel("15.0%")
        height_layout.addWidget(self.height_slider)
        height_layout.addWidget(self.height_label)
        auto_layout.addRow("Height (%):", height_layout)
        
        # Distance threshold
        distance_layout = QHBoxLayout()
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(20)
        self.distance_slider.valueChanged.connect(self._update_distance_label)
        self.distance_slider.valueChanged.connect(self._trigger_peak_update)
        
        self.distance_label = QLabel("20")
        distance_layout.addWidget(self.distance_slider)
        distance_layout.addWidget(self.distance_label)
        auto_layout.addRow("Distance:", distance_layout)
        
        # Prominence threshold
        prominence_layout = QHBoxLayout()
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 50)
        self.prominence_slider.setValue(20)  # 20% default
        self.prominence_slider.valueChanged.connect(self._update_prominence_label)
        self.prominence_slider.valueChanged.connect(self._trigger_peak_update)
        
        self.prominence_label = QLabel("20.0%")
        prominence_layout.addWidget(self.prominence_slider)
        prominence_layout.addWidget(self.prominence_label)
        auto_layout.addRow("Prominence (%):", prominence_layout)
        
        layout.addWidget(auto_group)
        
        # Peak Management
        management_group = QGroupBox("Peak Management")
        management_layout = QVBoxLayout(management_group)
        
        # Interactive selection
        self.interactive_btn = QPushButton("🖱️ Enable Interactive Selection")
        self.interactive_btn.setCheckable(True)
        self.interactive_btn.clicked.connect(self._toggle_interactive_mode)
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
        management_layout.addWidget(self.interactive_btn)
        
        # Peak list
        self.peak_list_widget = QListWidget()
        self.peak_list_widget.setMaximumHeight(120)
        management_layout.addWidget(self.peak_list_widget)
        
        # Management buttons
        mgmt_buttons_layout = QHBoxLayout()
        
        clear_peaks_btn = QPushButton("Clear All Peaks")
        clear_peaks_btn.clicked.connect(self._clear_peaks)
        mgmt_buttons_layout.addWidget(clear_peaks_btn)
        
        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.clicked.connect(self._remove_selected_peak)
        mgmt_buttons_layout.addWidget(remove_selected_btn)
        
        management_layout.addLayout(mgmt_buttons_layout)
        layout.addWidget(management_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.peak_count_label = QLabel("Auto peaks: 0 | Manual peaks: 0 | Total: 0")
        self.peak_count_label.setStyleSheet("font-weight: bold; color: #333;")
        status_layout.addWidget(self.peak_count_label)
        
        self.interactive_status_label = QLabel("Interactive mode: OFF")
        self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
        status_layout.addWidget(self.interactive_status_label)
        
        layout.addWidget(status_group)
        layout.addStretch()
        
        # Initialize labels
        self._update_height_label()
        self._update_distance_label()
        self._update_prominence_label()
        
        return tab
    
    def _create_fitting_tab(self):
        """Create peak fitting tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_group = QGroupBox("Peak Model")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Gaussian", "Lorentzian", "Pseudo-Voigt", "Voigt", "Asymmetric Voigt"
        ])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_individual_peaks = QCheckBox("Show Individual Peaks")
        self.show_individual_peaks.setChecked(True)
        display_layout.addWidget(self.show_individual_peaks)
        
        self.show_fit_statistics = QCheckBox("Show Fit Statistics")
        self.show_fit_statistics.setChecked(True)
        display_layout.addWidget(self.show_fit_statistics)
        
        layout.addWidget(display_group)
        
        # Fitting controls
        fitting_group = QGroupBox("Peak Fitting")
        fitting_layout = QVBoxLayout(fitting_group)
        
        fit_btn = QPushButton("Fit Peaks")
        fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
        """)
        fit_btn.clicked.connect(self._fit_peaks)
        fitting_layout.addWidget(fit_btn)
        
        layout.addWidget(fitting_group)
        layout.addStretch()
        
        return tab
    
    def _create_analysis_tab(self):
        """Create analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Peak statistics
        stats_group = QGroupBox("Peak Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No peaks fitted yet")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Analysis controls
        analysis_group = QGroupBox("Analysis Tools")
        analysis_layout = QVBoxLayout(analysis_group)
        
        analyze_btn = QPushButton("Generate Report")
        analyze_btn.clicked.connect(self._generate_analysis_report)
        analysis_layout.addWidget(analyze_btn)
        
        export_btn = QPushButton("Export Peak Data")
        export_btn.clicked.connect(self._export_peak_data)
        analysis_layout.addWidget(export_btn)
        
        layout.addWidget(analysis_group)
        layout.addStretch()
        
        return tab
    
    def _create_results_tab(self):
        """Create results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results display
        results_group = QGroupBox("Fitting Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_label = QLabel("No results available")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 9px;
            }
        """)
        results_layout.addWidget(self.results_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        return tab
    
    # Live update methods with debouncing
    def _trigger_background_update(self):
        """Trigger debounced background update"""
        self.bg_update_timer.stop()
        self.bg_update_timer.start(self.bg_update_delay)
    
    def _trigger_peak_update(self):
        """Trigger debounced peak detection update"""
        self.peak_update_timer.stop()
        self.peak_update_timer.start(self.peak_update_delay)
    
    def _update_background_preview(self):
        """Update background calculation with live preview"""
        try:
            if self.peak_fitter and hasattr(self.peak_fitter, 'set_als_parameters'):
                # Update ALS parameters
                lambda_val = 10 ** self.lambda_slider.value()
                p_val = self.p_slider.value() / 1000.0
                niter_val = self.niter_slider.value()
                
                self.peak_fitter.set_als_parameters(lambda_val, p_val, niter_val)
                
                # Set background method
                method = self.bg_method_combo.currentText()
                self.peak_fitter.set_background_method(method)
                
                # Trigger live background update
                if hasattr(self.main_controller, '_update_live_background'):
                    self.main_controller._update_live_background()
        except Exception as e:
            print(f"Background preview error: {e}")
    
    def _update_peak_detection(self):
        """Update peak detection with live preview"""
        try:
            if self.peak_fitter and hasattr(self.peak_fitter, 'set_peak_detection_parameters'):
                height = self.height_slider.value() / 100.0
                distance = self.distance_slider.value()
                prominence = self.prominence_slider.value() / 100.0
                
                self.peak_fitter.set_peak_detection_parameters(height, distance, prominence)
                
                # Trigger peak detection
                self.emit_action('detect_peaks', {})
        except Exception as e:
            print(f"Peak detection error: {e}")
    
    # Label update methods
    def _update_lambda_label(self):
        value = 10 ** self.lambda_slider.value()
        self.lambda_label.setText(f"{value:.0e}")
    
    def _update_p_label(self):
        value = self.p_slider.value() / 1000.0
        self.p_label.setText(f"{value:.3f}")
    
    def _update_niter_label(self):
        value = self.niter_slider.value()
        self.niter_label.setText(str(value))
    
    def _update_height_label(self):
        value = self.height_slider.value()
        self.height_label.setText(f"{value}.0%")
    
    def _update_distance_label(self):
        value = self.distance_slider.value()
        self.distance_label.setText(str(value))
    
    def _update_prominence_label(self):
        value = self.prominence_slider.value()
        self.prominence_label.setText(f"{value}.0%")
    
    # Action methods
    def _on_bg_method_changed(self):
        """Handle background method change"""
        method = self.bg_method_combo.currentText()
        
        # Show/hide ALS parameters based on method
        is_als = "ALS" in method
        self.als_group.setVisible(is_als)
        
        # Trigger background update
        self._trigger_background_update()
    
    def _on_model_changed(self):
        """Handle model change"""
        model = self.model_combo.currentText()
        if self.peak_fitter and hasattr(self.peak_fitter, 'set_model'):
            self.peak_fitter.set_model(model)
    
    def _apply_background(self):
        """Apply background subtraction"""
        self.emit_action('apply_background', {})
    
    def _clear_background(self):
        """Clear background preview"""
        self.emit_action('clear_background', {})
    
    def _reset_spectrum(self):
        """Reset spectrum to original"""
        self.emit_action('reset_spectrum', {})
    
    def _fit_peaks(self):
        """Fit peaks"""
        model = self.model_combo.currentText() if self.model_combo else "Gaussian"
        self.emit_action('fit_peaks', {'model': model})
    
    def _clear_peaks(self):
        """Clear all peaks"""
        self.emit_action('clear_peaks', {})
    
    def _remove_selected_peak(self):
        """Remove selected peak"""
        current_item = self.peak_list_widget.currentItem()
        if current_item:
            # Implementation would depend on peak data structure
            pass
    
    def _toggle_interactive_mode(self):
        """Toggle interactive peak selection"""
        self.interactive_mode = self.interactive_btn.isChecked()
        
        if self.interactive_mode:
            self.interactive_btn.setText("🖱️ Disable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: ON - Click on spectrum to select peaks")
            self.interactive_status_label.setStyleSheet("color: #4CAF50; font-size: 10px; font-weight: bold;")
        else:
            self.interactive_btn.setText("🖱️ Enable Interactive Selection")
            self.interactive_status_label.setText("Interactive mode: OFF")
            self.interactive_status_label.setStyleSheet("color: #666; font-size: 10px;")
        
        self.emit_action('toggle_interactive_mode', {'enabled': self.interactive_mode})
    
    def _generate_analysis_report(self):
        """Generate analysis report"""
        self.emit_action('generate_report', {})
    
    def _export_peak_data(self):
        """Export peak data"""
        self.emit_action('export_peak_data', {})
    
    # Interface methods for external updates
    def update_peak_count(self, auto_count, manual_count):
        """Update peak count display"""
        total = auto_count + manual_count
        self.peak_count_label.setText(f"Auto peaks: {auto_count} | Manual peaks: {manual_count} | Total: {total}")
    
    def update_peak_list(self, peaks_data):
        """Update the peak list widget"""
        self.peak_list_widget.clear()
        
        for i, peak in enumerate(peaks_data):
            item_text = f"Peak {i+1}: {peak.get('position', 0):.1f} cm⁻¹"
            item = QListWidgetItem(item_text)
            self.peak_list_widget.addItem(item)
    
    def update_results(self, results_text):
        """Update results display"""
        if hasattr(self, 'results_label'):
            self.results_label.setText(results_text)
    
    def update_stats(self, stats_text):
        """Update statistics display"""
        if hasattr(self, 'stats_label'):
            self.stats_label.setText(stats_text)
    
    # BaseTab interface methods
    def connect_signals(self):
        """Connect signals to UI manager"""
        # Signals are connected via emit_action calls
        pass
    
    def connect_core_signals(self):
        """Connect to core component signals"""
        if self.peak_fitter:
            # Connect to peak fitter signals if available
            if hasattr(self.peak_fitter, 'peaks_detected'):
                self.peak_fitter.peaks_detected.connect(self._on_peaks_detected)
            if hasattr(self.peak_fitter, 'background_calculated'):
                self.peak_fitter.background_calculated.connect(self._on_background_calculated)
            if hasattr(self.peak_fitter, 'fitting_completed'):
                self.peak_fitter.fitting_completed.connect(self._on_fitting_completed)
    
    def _on_peaks_detected(self, peaks):
        """Handle peaks detected signal"""
        self.update_peak_count(len(peaks), 0)  # Assuming no manual peaks for now
    
    def _on_background_calculated(self, background):
        """Handle background calculated signal"""
        pass  # Visual update handled by visualization manager
    
    def _on_fitting_completed(self, results):
        """Handle fitting completed signal"""
        if results and results.get('success', False):
            # Update results display
            self.update_results("Peak fitting completed successfully!")
    
    def get_tab_data(self):
        """Get current tab configuration"""
        return {
            'background_method': self.bg_method_combo.currentText() if self.bg_method_combo else "ALS",
            'lambda': 10 ** self.lambda_slider.value() if self.lambda_slider else 1e5,
            'p': self.p_slider.value() / 1000.0 if self.p_slider else 0.01,
            'niter': self.niter_slider.value() if self.niter_slider else 10,
            'height': self.height_slider.value() / 100.0 if self.height_slider else 0.15,
            'distance': self.distance_slider.value() if self.distance_slider else 20,
            'prominence': self.prominence_slider.value() / 100.0 if self.prominence_slider else 0.20,
            'model': self.model_combo.currentText() if self.model_combo else "Gaussian",
            'interactive_mode': self.interactive_mode
        }
    
    def update_from_peak_fitter(self, data=None):
        """Update UI from peak fitter state"""
        if data and self.peak_fitter:
            # Update UI elements based on peak fitter data
            pass
    
    def reset_to_defaults(self):
        """Reset all controls to default values"""
        if self.bg_method_combo:
            self.bg_method_combo.setCurrentIndex(0)  # ALS
        if self.lambda_slider:
            self.lambda_slider.setValue(5)  # 10^5
        if self.p_slider:
            self.p_slider.setValue(10)  # 0.01
        if self.niter_slider:
            self.niter_slider.setValue(10)
        if self.height_slider:
            self.height_slider.setValue(15)  # 15%
        if self.distance_slider:
            self.distance_slider.setValue(20)
        if self.prominence_slider:
            self.prominence_slider.setValue(20)  # 20%
        if self.model_combo:
            self.model_combo.setCurrentIndex(0)  # Gaussian
        
        # Update labels
        self._update_lambda_label()
        self._update_p_label()
        self._update_niter_label()
        self._update_height_label()
        self._update_distance_label()
        self._update_prominence_label()

    def _on_peaks_detected(self, peaks):
        """Handle peaks detected signal"""
        self.update_peak_count(len(peaks), 0)  # Assuming no manual peaks for now
    
    def _on_background_calculated(self, background):
        """Handle background calculated signal"""
        pass  # Visual update handled by visualization manager
    
    def _on_fitting_completed(self, results):
        """Handle fitting completed signal"""
        if results and results.get('success', False):
            # Update results display
            self.update_results("Peak fitting completed successfully!")
    
    def get_tab_data(self):
        """Get current tab configuration"""
        return {
            'background_method': self.bg_method_combo.currentText() if self.bg_method_combo else "ALS",
            'lambda': 10 ** self.lambda_slider.value() if self.lambda_slider else 1e5,
            'p': self.p_slider.value() / 1000.0 if self.p_slider else 0.01,
            'niter': self.niter_slider.value() if self.niter_slider else 10,
            'height': self.height_slider.value() / 100.0 if self.height_slider else 0.15,
            'distance': self.distance_slider.value() if self.distance_slider else 20,
            'prominence': self.prominence_slider.value() / 100.0 if self.prominence_slider else 0.20,
            'model': self.model_combo.currentText() if self.model_combo else "Gaussian",
            'interactive_mode': self.interactive_mode
        }
    
    def update_from_peak_fitter(self, data=None):
        """Update UI from peak fitter state"""
        if data and self.peak_fitter:
            # Update UI elements based on peak fitter data
            pass
    
    def reset_to_defaults(self):
        """Reset all controls to default values"""
        if self.bg_method_combo:
            self.bg_method_combo.setCurrentIndex(0)  # ALS
        if self.lambda_slider:
            self.lambda_slider.setValue(5)  # 10^5
        if self.p_slider:
            self.p_slider.setValue(10)  # 0.01
        if self.niter_slider:
            self.niter_slider.setValue(10)
        if self.height_slider:
            self.height_slider.setValue(15)  # 15%
        if self.distance_slider:
            self.distance_slider.setValue(20)
        if self.prominence_slider:
            self.prominence_slider.setValue(20)  # 20%
        if self.model_combo:
            self.model_combo.setCurrentIndex(0)  # Gaussian
        
        # Update labels
        self._update_lambda_label()
        self._update_p_label()
        self._update_niter_label()
        self._update_height_label()
        self._update_distance_label()
        self._update_prominence_label() 