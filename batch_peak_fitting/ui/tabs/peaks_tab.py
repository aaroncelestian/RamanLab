"""
Peaks Tab Component
Handles peak detection, fitting parameters, and background subtraction controls
Simplified version of the original peak controls functionality
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget,
    QPushButton, QLabel, QSlider, QComboBox, QSpinBox, QWidget
)
from PySide6.QtCore import Qt

from ..base_tab import BaseTab

# REFACTORED: Import centralized UI components
try:
    from core.peak_fitting_ui import BackgroundControlsWidget, PeakFittingControlsWidget
    CENTRALIZED_UI_AVAILABLE = True
except ImportError:
    CENTRALIZED_UI_AVAILABLE = False
    print("Warning: Centralized UI components not available, using fallback implementation")


class PeaksTab(BaseTab):
    """
    Peak detection and fitting controls component.
    Combines background subtraction, smoothing, and peak detection.
    """
    
    def __init__(self, parent=None):
        # Initialize widgets that will be created
        self.bg_method_combo = None
        self.height_slider = None
        self.distance_slider = None
        self.prominence_slider = None
        self.model_combo = None
        
        # Label widgets for parameter display
        self.height_label = None
        self.distance_label = None
        self.prominence_label = None
        self.peak_count_label = None
        
        super().__init__(parent)
        self.tab_name = "Peaks"
    
    def setup_ui(self):
        """Create the peak controls UI"""
        # Create sub-tab widget for organization
        sub_tab_widget = QTabWidget()
        self.main_layout.addWidget(sub_tab_widget)
        
        # Background Subtraction tab
        background_tab = self._create_background_tab()
        sub_tab_widget.addTab(background_tab, "Background")
        
        # Peak Detection & Fitting tab (combined for better workflow)
        peak_detection_tab = self._create_peak_detection_tab()
        sub_tab_widget.addTab(peak_detection_tab, "Peak Detection & Fitting")
        
        # Peak Management tab
        peak_management_tab = self._create_peak_management_tab()
        sub_tab_widget.addTab(peak_management_tab, "Peak Management")
    
    def _create_background_tab(self):
        """Create background subtraction controls - using centralized widget when available"""
        container = QVBoxLayout()
        
        # REFACTORED: Use centralized BackgroundControlsWidget when available
        if CENTRALIZED_UI_AVAILABLE:
            self.bg_controls_widget = BackgroundControlsWidget()
            
            # Connect signals from centralized widget
            self.bg_controls_widget.background_method_changed.connect(self._on_bg_method_changed)
            self.bg_controls_widget.parameters_changed.connect(self._on_bg_parameters_changed)
            self.bg_controls_widget.apply_background.connect(self._apply_background)
            self.bg_controls_widget.reset_spectrum.connect(self._clear_background)
            
            container.addWidget(self.bg_controls_widget)
            
            # Store references for compatibility with existing methods
            self.bg_method_combo = self.bg_controls_widget.bg_method_combo
            self.lambda_slider = self.bg_controls_widget.lambda_slider
            self.lambda_label = self.bg_controls_widget.lambda_label
            self.p_slider = self.bg_controls_widget.p_slider
            self.p_label = self.bg_controls_widget.p_label
            self.niter_slider = self.bg_controls_widget.niter_slider
            self.niter_label = self.bg_controls_widget.niter_label
            
            # Create dummy button references for compatibility (centralized widget handles buttons)
            self.apply_bg_btn = None
            self.preview_bg_btn = None  
            self.clear_bg_btn = None
            
        else:
            # Fallback: Create background controls manually
            bg_group = QGroupBox("Background Subtraction")
            bg_layout = QVBoxLayout(bg_group)
            
            # Method selection
            method_layout = QHBoxLayout()
            method_layout.addWidget(QLabel("Method:"))
            self.bg_method_combo = QComboBox()
            self.bg_method_combo.addItems([
                "ALS (Asymmetric Least Squares)",
                "Linear",
                "Polynomial", 
                "Moving Average",
                "Spline"
            ])
            method_layout.addWidget(self.bg_method_combo)
            bg_layout.addLayout(method_layout)
        
            # ALS parameters (live adjustment) - fallback implementation
            als_group = QGroupBox("ALS Parameters (Live)")
            als_layout = QVBoxLayout(als_group)
            
            # Lambda parameter
            lambda_layout = QHBoxLayout()
            lambda_layout.addWidget(QLabel("Lambda (Smoothness):"))
            self.lambda_slider = QSlider(Qt.Horizontal)
            self.lambda_slider.setRange(1, 8)  # 10^1 to 10^8
            self.lambda_slider.setValue(5)  # 10^5 = 100000
            lambda_layout.addWidget(self.lambda_slider)
            self.lambda_label = QLabel("10^5")
            lambda_layout.addWidget(self.lambda_label)
            als_layout.addLayout(lambda_layout)
            
            # P parameter
            p_layout = QHBoxLayout()
            p_layout.addWidget(QLabel("P (Asymmetry):"))
            self.p_slider = QSlider(Qt.Horizontal)
            self.p_slider.setRange(1, 50)  # 0.001 to 0.1
            self.p_slider.setValue(10)  # 0.01
            p_layout.addWidget(self.p_slider)
            self.p_label = QLabel("0.01")
            p_layout.addWidget(self.p_label)
            als_layout.addLayout(p_layout)
            
            # Iterations
            niter_layout = QHBoxLayout()
            niter_layout.addWidget(QLabel("Iterations:"))
            self.niter_slider = QSlider(Qt.Horizontal)
            self.niter_slider.setRange(5, 50)
            self.niter_slider.setValue(10)
            niter_layout.addWidget(self.niter_slider)
            self.niter_label = QLabel("10")
            niter_layout.addWidget(self.niter_label)
            als_layout.addLayout(niter_layout)
            
            bg_layout.addWidget(als_group)
            
            # Background controls
            controls_layout = QHBoxLayout()
            
            apply_bg_btn = QPushButton("Apply Background")
            apply_bg_btn.setStyleSheet("""
                QPushButton {
                    background-color: #388E3C;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2E7D32;
                }
            """)
            controls_layout.addWidget(apply_bg_btn)
            
            preview_bg_btn = QPushButton("Preview")
            preview_bg_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1976D2;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1565C0;
                }
            """)
            controls_layout.addWidget(preview_bg_btn)
            
            clear_bg_btn = QPushButton("Clear")
            clear_bg_btn.setStyleSheet("""
                QPushButton {
                    background-color: #F57C00;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #EF6C00;
                }
            """)
            controls_layout.addWidget(clear_bg_btn)
            
            bg_layout.addLayout(controls_layout)
            
            container.addWidget(bg_group)
            
            # Store button references
            self.apply_bg_btn = apply_bg_btn
            self.preview_bg_btn = preview_bg_btn
            self.clear_bg_btn = clear_bg_btn
        
        container.addStretch()
        
        # Create widget to contain the layout
        widget = QWidget()
        widget.setLayout(container)
        return widget
    
    def _create_peak_detection_tab(self):
        """Create peak detection and fitting controls"""
        container = QVBoxLayout()
        
        # Peak detection group
        detection_group = QGroupBox("Automatic Peak Detection")
        detection_layout = QVBoxLayout(detection_group)
        
        # Height threshold
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Min Height:"))
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(1, 100)
        self.height_slider.setValue(15)  # 15% default - reasonable for most spectra
        height_layout.addWidget(self.height_slider)
        self.height_label = QLabel("15%")
        height_layout.addWidget(self.height_label)
        detection_layout.addLayout(height_layout)
        
        # Distance threshold
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Min Distance:"))
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 100)
        self.distance_slider.setValue(20)  # Keep distance the same
        distance_layout.addWidget(self.distance_slider)
        self.distance_label = QLabel("20")
        distance_layout.addWidget(self.distance_label)
        detection_layout.addLayout(distance_layout)
        
        # Prominence threshold
        prominence_layout = QHBoxLayout()
        prominence_layout.addWidget(QLabel("Min Prominence:"))
        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 100)
        self.prominence_slider.setValue(20)  # 20% default - reasonable for most spectra
        prominence_layout.addWidget(self.prominence_slider)
        self.prominence_label = QLabel("20%")
        prominence_layout.addWidget(self.prominence_label)
        detection_layout.addLayout(prominence_layout)
        
        # Detection buttons
        detection_buttons_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("Detect Peaks")
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        detection_buttons_layout.addWidget(self.detect_btn)
        
        self.clear_peaks_btn = QPushButton("Clear Peaks")
        self.clear_peaks_btn.setStyleSheet("""
            QPushButton {
                background-color: #F57C00;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #EF6C00;
            }
        """)
        detection_buttons_layout.addWidget(self.clear_peaks_btn)
        
        detection_layout.addLayout(detection_buttons_layout)
        
        # Peak count display
        self.peak_count_label = QLabel("Peaks found: 0")
        self.peak_count_label.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border: 1px solid #E0E0E0;
                padding: 5px;
                border-radius: 3px;
                font-family: monospace;
            }
        """)
        detection_layout.addWidget(self.peak_count_label)
        
        container.addWidget(detection_group)
        
        # Peak fitting group
        fitting_group = QGroupBox("Peak Fitting with Mathematical Models")
        fitting_layout = QVBoxLayout(fitting_group)
        
        # Model selection
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Gaussian",
            "Lorentzian",
            "Voigt", 
            "Pseudo-Voigt",
            "Asymmetric Voigt"
        ])
        model_selection_layout.addWidget(self.model_combo)
        fitting_layout.addLayout(model_selection_layout)
        
        # Fitting controls
        fitting_controls_layout = QHBoxLayout()
        
        self.fit_btn = QPushButton("Fit Peaks")
        self.fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #C62828;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.fit_btn.setEnabled(False)  # Initially disabled until peaks are detected
        self.fit_btn.setToolTip("Detect peaks first, then click to fit mathematical functions to them")
        fitting_controls_layout.addWidget(self.fit_btn)
        
        fitting_layout.addLayout(fitting_controls_layout)
        
        # Add fitting status label
        self.fitting_status_label = QLabel("No peaks detected yet - use 'Detect Peaks' first")
        self.fitting_status_label.setStyleSheet("""
            QLabel {
                background-color: #FFF3CD;
                border: 1px solid #FFEAA7;
                padding: 5px;
                border-radius: 3px;
                color: #856404;
                font-style: italic;
            }
        """)
        fitting_layout.addWidget(self.fitting_status_label)
        
        container.addWidget(fitting_group)
        container.addStretch()
        
        # Create widget to contain the layout
        widget = QWidget()
        widget.setLayout(container)
        return widget
    
    def _create_peak_management_tab(self):
        """Create peak management controls"""
        container = QVBoxLayout()
        
        # Reference peaks group
        ref_group = QGroupBox("Reference Peaks")
        ref_layout = QVBoxLayout(ref_group)
        
        ref_info = QLabel("Set reference peaks for batch processing:")
        ref_layout.addWidget(ref_info)
        
        ref_buttons_layout = QHBoxLayout()
        
        set_ref_btn = QPushButton("Set Reference")
        set_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6D28D9;
            }
        """)
        ref_buttons_layout.addWidget(set_ref_btn)
        
        clear_ref_btn = QPushButton("Clear Reference")
        clear_ref_btn.setStyleSheet("""
            QPushButton {
                background-color: #6B7280;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
        """)
        ref_buttons_layout.addWidget(clear_ref_btn)
        
        ref_layout.addLayout(ref_buttons_layout)
        
        # Manual peak management
        manual_group = QGroupBox("Manual Peak Management")
        manual_layout = QVBoxLayout(manual_group)
        
        manual_info = QLabel("Click on spectrum to add manual peaks")
        manual_layout.addWidget(manual_info)
        
        # Manual peaks list
        from PySide6.QtWidgets import QListWidget, QAbstractItemView
        
        peaks_list_label = QLabel("Manual Peaks:")
        peaks_list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        manual_layout.addWidget(peaks_list_label)
        
        self.manual_peaks_list = QListWidget()
        self.manual_peaks_list.setMaximumHeight(120)
        self.manual_peaks_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.manual_peaks_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #EEEEEE;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #E8F4FD;
            }
        """)
        manual_layout.addWidget(self.manual_peaks_list)
        
        manual_buttons_layout = QHBoxLayout()
        
        delete_selected_btn = QPushButton("Delete Selected")
        delete_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF8C00;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF7F00;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        delete_selected_btn.setEnabled(False)  # Initially disabled
        manual_buttons_layout.addWidget(delete_selected_btn)
        
        clear_manual_btn = QPushButton("Clear All")
        clear_manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
        """)
        manual_buttons_layout.addWidget(clear_manual_btn)
        
        manual_layout.addLayout(manual_buttons_layout)
        
        container.addWidget(ref_group)
        container.addWidget(manual_group)
        container.addStretch()
        
        # Store button references
        self.set_ref_btn = set_ref_btn
        self.clear_ref_btn = clear_ref_btn
        self.delete_selected_btn = delete_selected_btn
        self.clear_manual_btn = clear_manual_btn
        
        # Create widget to contain the layout
        widget = QWidget()
        widget.setLayout(container)
        return widget
    
    def connect_signals(self):
        """Connect internal signals"""
        # Peak detection parameter sliders
        self.height_slider.valueChanged.connect(self._update_height_label)
        self.distance_slider.valueChanged.connect(self._update_distance_label)
        self.prominence_slider.valueChanged.connect(self._update_prominence_label)
        
        # REFACTORED: Only connect individual sliders if NOT using centralized widget
        # When using centralized widget, it handles all parameter connections internally
        if not (hasattr(self, 'bg_controls_widget') and CENTRALIZED_UI_AVAILABLE):
            # Fallback mode: connect individual sliders 
            if hasattr(self, 'lambda_slider') and self.lambda_slider is not None:
                self.lambda_slider.valueChanged.connect(self._update_lambda_label)
            if hasattr(self, 'p_slider') and self.p_slider is not None:
                self.p_slider.valueChanged.connect(self._update_p_label)
            if hasattr(self, 'niter_slider') and self.niter_slider is not None:
                self.niter_slider.valueChanged.connect(self._update_niter_label)
        
        # Background controls (only connect if they exist and are not None)
        if hasattr(self, 'bg_method_combo') and self.bg_method_combo is not None:
            self.bg_method_combo.currentTextChanged.connect(self._on_bg_method_changed)
        if hasattr(self, 'apply_bg_btn') and self.apply_bg_btn is not None:
            self.apply_bg_btn.clicked.connect(self._apply_background)
        if hasattr(self, 'preview_bg_btn') and self.preview_bg_btn is not None:
            self.preview_bg_btn.clicked.connect(self._preview_background)
        if hasattr(self, 'clear_bg_btn') and self.clear_bg_btn is not None:
            self.clear_bg_btn.clicked.connect(self._clear_background)
        
        # Peak detection
        self.detect_btn.clicked.connect(self._detect_peaks)
        self.clear_peaks_btn.clicked.connect(self._clear_peaks)
        
        # Peak fitting
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.fit_btn.clicked.connect(self._fit_peaks)
        
        # Peak management
        self.set_ref_btn.clicked.connect(self._set_reference)
        self.clear_ref_btn.clicked.connect(self._clear_reference)
        self.delete_selected_btn.clicked.connect(self._delete_selected_peaks)
        self.clear_manual_btn.clicked.connect(self._clear_manual_peaks)
        
        # Manual peaks list selection
        self.manual_peaks_list.itemSelectionChanged.connect(self._on_manual_peaks_selection_changed)
    
    def connect_core_signals(self):
        """Connect to core component signals"""
        if self.peak_fitter:
            self.peak_fitter.peaks_fitted.connect(self._on_peaks_fitted)
            self.peak_fitter.background_calculated.connect(self._on_background_calculated)
        
        if self.data_processor:
            self.data_processor.spectrum_loaded.connect(self._on_spectrum_loaded)
    
    def _update_height_label(self):
        """Update height threshold label"""
        value = self.height_slider.value()
        self.height_label.setText(f"{value}%")
        # Convert to 0-1 range for percentage
        height_fraction = value / 100.0
        self.emit_action("peak_detection_params_changed", {"height": height_fraction})
    
    def _update_distance_label(self):
        """Update distance label"""
        value = self.distance_slider.value()
        self.distance_label.setText(str(value))
        self.emit_action("peak_detection_params_changed", {"distance": value})
    
    def _update_prominence_label(self):
        """Update prominence label"""
        value = self.prominence_slider.value()
        self.prominence_label.setText(f"{value}%")
        # Convert to 0-1 range for percentage
        prominence_fraction = value / 100.0
        self.emit_action("peak_detection_params_changed", {"prominence": prominence_fraction})
    
    def _update_lambda_label(self):
        """Update lambda parameter label and trigger live background update"""
        value = self.lambda_slider.value()
        lambda_val = 10 ** value
        self.lambda_label.setText(f"10^{value}")
        self.emit_action("background_params_changed", {"lambda": lambda_val, "param_type": "lambda"})
    
    def _update_p_label(self):
        """Update p parameter label and trigger live background update"""
        value = self.p_slider.value()
        p_val = value / 1000.0  # Convert to 0.001 - 0.05 range
        self.p_label.setText(f"{p_val:.3f}")
        self.emit_action("background_params_changed", {"p": p_val, "param_type": "p"})
    
    def _update_niter_label(self):
        """Update iterations parameter label and trigger live background update"""
        value = self.niter_slider.value()
        self.niter_label.setText(str(value))
        self.emit_action("background_params_changed", {"niter": value, "param_type": "niter"})
    
    def _on_bg_method_changed(self):
        """Handle background method change"""
        method = self.bg_method_combo.currentText()
        if self.peak_fitter:
            # Pass the full method text to let peak_fitter parse it
            self.peak_fitter.set_background_method(method)
        self.emit_action("background_method_changed", {"method": method})
    
    def _on_model_changed(self):
        """Handle peak model change"""
        model = self.model_combo.currentText()
        if self.peak_fitter:
            self.peak_fitter.set_model(model)
        self.emit_action("peak_model_changed", {"model": model})
        
        # Update fitting status to reflect new model
        if hasattr(self, 'peak_count_label'):
            current_text = self.peak_count_label.text()
            # Extract number of peaks from current text
            import re
            match = re.search(r'(\d+)', current_text)
            n_peaks = int(match.group(1)) if match else 0
            self._update_fitting_status(n_peaks)
    
    def _apply_background(self):
        """Apply background subtraction"""
        self.emit_action("apply_background", {})
        self.emit_status("Background subtraction applied")
    
    def _preview_background(self):
        """Preview background subtraction"""
        self.emit_action("preview_background", {})
        self.emit_status("Background preview updated")
    
    def _clear_background(self):
        """Clear background subtraction"""
        self.emit_action("clear_background", {})
        self.emit_status("Background cleared")
    
    def _fit_peaks(self):
        """Fit peaks to current spectrum"""
        self.emit_action("fit_peaks", {})
        self.emit_status("Peak fitting initiated")
    
    def _detect_peaks(self):
        """Detect peaks automatically"""
        params = {
            "height": self.height_slider.value() / 100.0,  # Convert to 0-1 range for consistency
            "distance": self.distance_slider.value(),
            "prominence": self.prominence_slider.value() / 100.0  # Convert to 0-1 range for consistency
        }
        self.emit_action("detect_peaks", params)
        self.emit_status("Peak detection initiated")
    
    def _clear_peaks(self):
        """Clear detected peaks"""
        self.emit_action("clear_peaks", {})
        self.emit_status("Peaks cleared")
        self.peak_count_label.setText("Peaks found: 0")
        self._update_fitting_status(0)  # Update fitting status when peaks are cleared
    
    def _set_reference(self):
        """Set current peaks as reference"""
        self.emit_action("set_reference", {})
        self.emit_status("Reference peaks set")
    
    def _clear_reference(self):
        """Clear reference peaks"""
        self.emit_action("clear_reference", {})
        self.emit_status("Reference peaks cleared")
    
    def _clear_manual_peaks(self):
        """Clear manual peaks"""
        self.emit_action("clear_manual_peaks", {})
        self.emit_status("Manual peaks cleared")
        self._update_manual_peaks_list([])  # Clear the list display
    
    def _delete_selected_peaks(self):
        """Delete selected manual peaks"""
        selected_items = self.manual_peaks_list.selectedItems()
        if not selected_items:
            return
        
        # Extract peak indices from selected items
        peak_indices_to_remove = []
        for item in selected_items:
            # Extract peak index from item text (format: "Peak at 1234.5 cm⁻¹ [index: 123]")
            text = item.text()
            try:
                start_idx = text.rfind("[index: ") + 8
                end_idx = text.rfind("]")
                if start_idx > 7 and end_idx > start_idx:
                    peak_index = int(text[start_idx:end_idx])
                    peak_indices_to_remove.append(peak_index)
            except ValueError:
                continue
        
        # Remove peaks through action system
        for peak_index in peak_indices_to_remove:
            self.emit_action("remove_manual_peak", {"peak_index": peak_index})
        
        self.emit_status(f"Removed {len(peak_indices_to_remove)} manual peak{'s' if len(peak_indices_to_remove) != 1 else ''}")
    
    def _on_manual_peaks_selection_changed(self):
        """Handle manual peaks list selection changes"""
        selected_items = self.manual_peaks_list.selectedItems()
        self.delete_selected_btn.setEnabled(len(selected_items) > 0)
    
    def _update_manual_peaks_list(self, manual_peaks, wavenumbers=None):
        """Update the manual peaks list display"""
        self.manual_peaks_list.clear()
        
        if not manual_peaks or len(manual_peaks) == 0:
            self.delete_selected_btn.setEnabled(False)
            return
        
        # Sort peaks by wavenumber for better display
        if wavenumbers is not None and len(wavenumbers) > 0:
            # Create list of (peak_index, wavenumber) pairs
            peaks_with_wavenumbers = []
            for peak_idx in manual_peaks:
                if 0 <= peak_idx < len(wavenumbers):
                    peaks_with_wavenumbers.append((peak_idx, wavenumbers[peak_idx]))
            
            # Sort by wavenumber
            peaks_with_wavenumbers.sort(key=lambda x: x[1])
            
            # Add items to list
            for peak_idx, wavenumber in peaks_with_wavenumbers:
                item_text = f"Peak at {wavenumber:.1f} cm⁻¹ [index: {peak_idx}]"
                self.manual_peaks_list.addItem(item_text)
        else:
            # Fallback: just show indices
            for peak_idx in sorted(manual_peaks):
                item_text = f"Peak at index {peak_idx}"
                self.manual_peaks_list.addItem(item_text)
        
        # Update button states
        self.delete_selected_btn.setEnabled(False)  # Reset selection
    
    def _on_peaks_fitted(self, results):
        """Handle peak fitting results"""
        if results.get('success', False):
            n_peaks = len(results.get('peak_positions', []))
            r2 = results.get('r_squared', 0)
            model = results.get('model', 'Unknown')
            self.emit_status(f"Peak fitting successful: {n_peaks} peaks fitted with {model} model, R² = {r2:.4f}")
            
            # Update fitting status label with success message
            if hasattr(self, 'fitting_status_label'):
                self.fitting_status_label.setText(f"✓ Fitted {n_peaks} peak{'s' if n_peaks != 1 else ''} with {model} model (R² = {r2:.4f})")
                self.fitting_status_label.setStyleSheet("""
                    QLabel {
                        background-color: #D1ECF1;
                        border: 1px solid #BEE5EB;
                        padding: 5px;
                        border-radius: 3px;
                        color: #0C5460;
                        font-style: italic;
                        font-weight: bold;
                    }
                """)
        else:
            error_msg = results.get('error', 'Unknown error')
            self.emit_status(f"Peak fitting failed: {error_msg}")
            
            # Update fitting status label with error message
            if hasattr(self, 'fitting_status_label'):
                self.fitting_status_label.setText(f"✗ Fitting failed: {error_msg}")
                self.fitting_status_label.setStyleSheet("""
                    QLabel {
                        background-color: #F8D7DA;
                        border: 1px solid #F5C6CB;
                        padding: 5px;
                        border-radius: 3px;
                        color: #721C24;
                        font-style: italic;
                    }
                """)
    
    def _on_background_calculated(self, background):
        """Handle background calculation"""
        self.emit_status(f"Background calculated: {len(background)} points")
    
    def _on_spectrum_loaded(self, spectrum_data):
        """Handle spectrum loading"""
        n_peaks = len(spectrum_data.get('peaks', []))
        self.peak_count_label.setText(f"Peaks found: {n_peaks}")
        
        # Update manual peaks list
        manual_peaks = spectrum_data.get('manual_peaks', [])
        wavenumbers = spectrum_data.get('wavenumbers', [])
        self._update_manual_peaks_list(manual_peaks, wavenumbers)
        
        # Update fitting status and button state
        self._update_fitting_status(n_peaks)
    
    def _update_fitting_status(self, n_peaks):
        """Update the fitting button state and status message based on detected peaks"""
        if hasattr(self, 'fit_btn') and hasattr(self, 'fitting_status_label'):
            if n_peaks > 0:
                self.fit_btn.setEnabled(True)
                self.fitting_status_label.setText(f"Ready to fit {n_peaks} peak{'s' if n_peaks != 1 else ''} with {self.model_combo.currentText() if self.model_combo else 'Gaussian'} model")
                self.fitting_status_label.setStyleSheet("""
                    QLabel {
                        background-color: #D4EDDA;
                        border: 1px solid #C3E6CB;
                        padding: 5px;
                        border-radius: 3px;
                        color: #155724;
                        font-style: italic;
                    }
                """)
            else:
                self.fit_btn.setEnabled(False)
                self.fitting_status_label.setText("No peaks detected yet - use 'Detect Peaks' first")
                self.fitting_status_label.setStyleSheet("""
                    QLabel {
                        background-color: #FFF3CD;
                        border: 1px solid #FFEAA7;
                        padding: 5px;
                        border-radius: 3px;
                        color: #856404;
                        font-style: italic;
                    }
                """)
    
    def update_from_peak_fitter(self, data=None):
        """Update tab when peak fitter state changes"""
        if self.peak_fitter:
            params = self.peak_fitter.get_parameters()
            
            # Update model selection
            current_model = params.get('model', 'Gaussian')
            index = self.model_combo.findText(current_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            
            # Update background method
            current_bg = params.get('background_method', 'ALS')
            for i in range(self.bg_method_combo.count()):
                if self.bg_method_combo.itemText(i).startswith(current_bg):
                    self.bg_method_combo.setCurrentIndex(i)
                    break
    
    def get_tab_data(self):
        """Get current tab state"""
        base_data = super().get_tab_data()
        
        base_data.update({
            'background_method': self.bg_method_combo.currentText() if self.bg_method_combo else "ALS",
            'peak_model': self.model_combo.currentText() if self.model_combo else "Gaussian",
            'height_threshold': self.height_slider.value() if self.height_slider else 20,
            'distance_threshold': self.distance_slider.value() if self.distance_slider else 20,
            'prominence_threshold': self.prominence_slider.value() if self.prominence_slider else 30
        })
        
        return base_data
    
    def reset_to_defaults(self):
        """Reset tab to default state"""
        if self.height_slider:
            self.height_slider.setValue(15)  # 15% height threshold
        if self.distance_slider:
            self.distance_slider.setValue(20)  # 20 points distance
        if self.prominence_slider:
            self.prominence_slider.setValue(20)  # 20% prominence threshold
        if self.bg_method_combo:
            self.bg_method_combo.setCurrentIndex(0)
        if self.model_combo:
            self.model_combo.setCurrentIndex(0)
        if self.peak_count_label:
            self.peak_count_label.setText("Peaks found: 0")
        
        self.emit_status("Tab reset to defaults")
    
    def _on_bg_parameters_changed(self):
        """Handle background parameter changes from centralized widget"""
        if hasattr(self, 'bg_controls_widget') and CENTRALIZED_UI_AVAILABLE:
            # Get parameters from centralized widget and directly trigger background calculation
            params = self.bg_controls_widget.get_background_parameters()
            
            if not self.peak_fitter or not self.data_processor:
                return
                
            # Get current spectrum data
            spectrum_data = self.data_processor.get_current_spectrum()
            if not spectrum_data or len(spectrum_data.get('wavenumbers', [])) == 0:
                return
            
            # Handle ALL background methods directly to avoid signal re-emission
            method = params.get('method', 'ALS')
            
            if method == 'ALS':
                lambda_val = params.get('lambda', 1e5)
                p_val = params.get('p', 0.01)
                niter_val = params.get('niter', 10)
                
                # Debug output to check parameter values
                print(f"ALS Parameters: lambda={lambda_val:.0e}, p={p_val:.3f}, niter={niter_val}")
                
                self.peak_fitter.set_als_parameters(lambda_val, p_val, niter_val)
                
            elif method == 'Linear':
                # Set method for linear background
                self.peak_fitter.set_background_method('Linear')
                print(f"Linear Parameters: start_weight={params.get('start_weight', 1.0)}, end_weight={params.get('end_weight', 1.0)}")
                
            elif method == 'Polynomial':
                # Set method for polynomial background  
                self.peak_fitter.set_background_method('Polynomial')
                print(f"Polynomial Parameters: order={params.get('order', 3)}")
                
            else:
                # For Moving Average, Spline, etc.
                self.peak_fitter.set_background_method(method)
                print(f"Background method: {method}")
            
            # Calculate background with new parameters - NO signal re-emission
            background = self.peak_fitter.calculate_background(
                spectrum_data['wavenumbers'],
                spectrum_data['intensities']
            )
            
            if background is not None:
                # Preview the background (updates data but doesn't apply permanently)
                self.data_processor.preview_background_subtraction(background)
                
                # **CRITICAL FIX**: Update the plot to show the new background
                self.emit_action("plot_update", {"plot_type": "current_spectrum"})
                
        else:
            # Fallback: trigger parameter update for manual controls
            self.emit_action("background_parameters_changed", {}) 