"""
Control panel widgets for the map analysis application.
"""

import logging
from typing import Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
    QLabel, QCheckBox, QComboBox, QLineEdit, QListWidget, QTabWidget,
    QTextEdit, QSlider, QDoubleSpinBox, QSizePolicy
)
from PySide6.QtCore import Signal, Qt

from .base_widgets import (
    ParameterGroupBox, ButtonGroup, TitleLabel, StandardButton,
    PrimaryButton, SuccessButton, WarningButton, InfoButton, SafeWidgetMixin,
    apply_icon_button_style
)

logger = logging.getLogger(__name__)


class BaseControlPanel(QWidget, SafeWidgetMixin):
    """Base class for control panels."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)  # Add consistent margins
        self.layout.setSpacing(8)  # Slightly more spacing for better visual separation
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Ensure content aligns to top
        
        # Add title
        title_label = TitleLabel(title)
        self.layout.addWidget(title_label)
        
        # Setup specific controls
        self.setup_controls()
        
    def setup_controls(self):
        """Setup panel-specific controls. Override in subclasses."""
        pass


class MapViewControlPanel(BaseControlPanel):
    """Control panel for map view tab."""
    
    # Signals
    feature_changed = Signal(str)
    use_processed_changed = Signal(bool)
    show_spectrum_toggled = Signal(bool)
    intensity_scaling_changed = Signal(float, float)  # vmin, vmax
    wavenumber_range_changed = Signal(float, float)  # center_wavenumber, range_width
    cosmic_ray_enabled_changed = Signal(bool)
    cosmic_ray_params_changed = Signal()
    reprocess_cosmic_rays_requested = Signal()
    apply_cre_to_all_files_requested = Signal()
    show_cosmic_ray_stats = Signal()
    show_cosmic_ray_diagnostics = Signal()
    test_shape_analysis_requested = Signal()
    fit_templates_to_map_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__("Map View Controls", parent)
        
    def setup_controls(self):
        """Setup map view controls."""
        # Features section
        feature_group = QGroupBox("Map Features")
        feature_layout = QVBoxLayout(feature_group)
        
        self.feature_combo = QComboBox()
        # Initialize with base features - can be updated later via refresh methods
        self.feature_combo.addItems(self._get_initial_features())
        self.feature_combo.currentTextChanged.connect(self.feature_changed.emit)
        feature_layout.addWidget(self.feature_combo)
        
        self.layout.addWidget(feature_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.use_processed_cb = QCheckBox("Use Processed Data")
        self.use_processed_cb.setChecked(True)
        self.use_processed_cb.toggled.connect(self.use_processed_changed.emit)
        display_layout.addWidget(self.use_processed_cb)
        
        self.show_spectrum_plot_cb = QCheckBox("Show Spectrum Plot")
        self.show_spectrum_plot_cb.toggled.connect(self.show_spectrum_toggled.emit)
        display_layout.addWidget(self.show_spectrum_plot_cb)
        
        # Add helpful label
        info_label = QLabel("Click on map to view spectrum at that position")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        display_layout.addWidget(info_label)
        
        self.layout.addWidget(display_group)
        
        # Wavenumber Range Integration section
        wavenumber_group = QGroupBox("Wavenumber Range Integration")
        wavenumber_layout = QVBoxLayout(wavenumber_group)
        
        # Integration mode selection
        self.integration_mode = QComboBox()
        self.integration_mode.addItems(["Full Spectrum", "Custom Range"])
        self.integration_mode.currentTextChanged.connect(self._on_integration_mode_changed)
        wavenumber_layout.addWidget(QLabel("Integration Mode:"))
        wavenumber_layout.addWidget(self.integration_mode)
        
        # Range controls (initially hidden)
        self.range_controls = ParameterGroupBox("Integration Range")
        
        # Center wavenumber slider - compact layout (move to top)
        self.center_wavenumber_slider = QSlider(Qt.Horizontal)
        self.center_wavenumber_slider.setRange(200, 4000)  # 200-4000 cm⁻¹ range
        self.center_wavenumber_slider.setValue(1000)  # Default center
        self.center_wavenumber_slider.setSingleStep(50)  # 50 cm⁻¹ increments
        self.center_wavenumber_slider.setPageStep(100)  # Larger jumps
        self.center_wavenumber_slider.setTickPosition(QSlider.TicksBelow)
        self.center_wavenumber_slider.setTickInterval(500)  # Tick marks every 500 cm⁻¹
        self.center_wavenumber_slider.setToolTip("Drag to set center wavenumber for integration (50 cm⁻¹ increments)")

        # Add the slider first
        slider_widget = QWidget()
        slider_layout = QVBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(self.center_wavenumber_slider)
        self.range_controls.layout.addWidget(slider_widget)

        # Integration range width (fixed at 100 cm⁻¹) - below slider
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Range Width (cm⁻¹):"))
        self.range_width_spin = QDoubleSpinBox()
        self.range_width_spin.setRange(50.0, 500.0)
        self.range_width_spin.setValue(100.0)
        self.range_width_spin.setSingleStep(25.0)
        self.range_width_spin.setToolTip("Width of integration range")
        self.range_width_spin.setMaximumWidth(80)
        width_layout.addWidget(self.range_width_spin)
        width_layout.addStretch()  # Push everything to the left

        width_widget = QWidget()
        width_widget.setLayout(width_layout)
        self.range_controls.layout.addWidget(width_widget)

        # Show calculated integration range - below the slider
        self.range_display = QLabel("Range: 950 - 1050 cm⁻¹")
        self.range_display.setStyleSheet("QLabel { color: #666; font-size: 10px; text-align: center; }")
        self.range_display.setAlignment(Qt.AlignCenter)
        self.range_controls.layout.addWidget(self.range_display)
        
        # Initially hide range controls
        self.range_controls.setVisible(False)
        
        # Connect parameter changes
        self.center_wavenumber_slider.valueChanged.connect(self._update_wavenumber_range)
        self.range_width_spin.valueChanged.connect(self._update_wavenumber_range)
        
        wavenumber_layout.addWidget(self.range_controls)
        
        # Intensity Scaling section (separate from wavenumber integration)
        intensity_subgroup = QGroupBox("Display Scaling")
        intensity_sublayout = QVBoxLayout(intensity_subgroup)
        
        # Auto-scale checkbox
        self.auto_scale_cb = QCheckBox("Auto Scale")
        self.auto_scale_cb.setChecked(True)
        self.auto_scale_cb.toggled.connect(self._on_auto_scale_toggled)
        intensity_sublayout.addWidget(self.auto_scale_cb)
        
        # Manual scaling controls
        self.intensity_params = ParameterGroupBox("Manual Scaling")
        self.vmin_spin = self.intensity_params.add_double_spinbox(
            "Min Value", -1000000.0, 1000000.0, 0.0, 100.0)
        self.vmax_spin = self.intensity_params.add_double_spinbox(
            "Max Value", -1000000.0, 1000000.0, 1000.0, 100.0)
        
        # Initially disable manual controls
        self.intensity_params.setEnabled(False)
        
        # Connect parameter changes
        self.vmin_spin.valueChanged.connect(self._emit_intensity_scaling)
        self.vmax_spin.valueChanged.connect(self._emit_intensity_scaling)
        
        intensity_sublayout.addWidget(self.intensity_params)
        
        # Reset button
        reset_btn = StandardButton("Reset to Auto")
        reset_btn.clicked.connect(self._reset_intensity_scaling)
        intensity_sublayout.addWidget(reset_btn)
        
        wavenumber_layout.addWidget(intensity_subgroup)
        
        self.layout.addWidget(wavenumber_group)
        
        # Cosmic Ray Detection section
        cosmic_ray_group = QGroupBox("Cosmic Ray Detection")
        cosmic_ray_layout = QVBoxLayout(cosmic_ray_group)
        
        # Enable/disable cosmic ray detection
        self.cosmic_ray_enabled_cb = QCheckBox("Enable Cosmic Ray Detection")
        self.cosmic_ray_enabled_cb.setChecked(True)
        self.cosmic_ray_enabled_cb.toggled.connect(self.cosmic_ray_enabled_changed.emit)
        cosmic_ray_layout.addWidget(self.cosmic_ray_enabled_cb)
        
        # Basic cosmic ray parameters
        self.cosmic_ray_params = ParameterGroupBox("CR Detection")
        self.threshold_spin = self.cosmic_ray_params.add_double_spinbox(
            "Threshold", 100.0, 5000.0, 1000.0, 100.0)
        self.neighbor_ratio_spin = self.cosmic_ray_params.add_double_spinbox(
            "Neighbor Ratio", 1.0, 20.0, 5.0, 0.5)
        
        # Connect parameter changes
        self.threshold_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        self.neighbor_ratio_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        
        cosmic_ray_layout.addWidget(self.cosmic_ray_params)
        
        # Shape analysis parameters
        self.shape_params = ParameterGroupBox("Shape Analysis")
        self.enable_shape_cb = self.shape_params.add_checkbox("Enable Shape Analysis", True)
        self.max_fwhm_spin = self.shape_params.add_double_spinbox(
            "Max FWHM", 5.0, 50.0, 20.0, 1.0)
        self.min_sharpness_spin = self.shape_params.add_double_spinbox(
            "Min Sharpness", 1.0, 10.0, 3.0, 0.5)
        self.max_asymmetry_spin = self.shape_params.add_double_spinbox(
            "Max Asymmetry", 0.1, 1.0, 0.5, 0.1)
        self.gradient_thresh_spin = self.shape_params.add_double_spinbox(
            "Gradient Threshold", 50.0, 500.0, 100.0, 10.0)
        
        # Connect shape parameter changes
        self.enable_shape_cb.toggled.connect(lambda: self.cosmic_ray_params_changed.emit())
        self.max_fwhm_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        self.min_sharpness_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        self.max_asymmetry_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        self.gradient_thresh_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        
        cosmic_ray_layout.addWidget(self.shape_params)
        
        # Range-based removal parameters
        self.removal_params = ParameterGroupBox("Removal Range")
        self.removal_range_spin = self.removal_params.add_spinbox(
            "Range", 1, 15, 3)
        self.adaptive_range_cb = self.removal_params.add_checkbox("Adaptive Range", True)
        self.max_removal_range_spin = self.removal_params.add_spinbox(
            "Max Range", 3, 20, 8)
        
        # Connect removal parameter changes
        self.removal_range_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        self.adaptive_range_cb.toggled.connect(lambda: self.cosmic_ray_params_changed.emit())
        self.max_removal_range_spin.valueChanged.connect(lambda: self.cosmic_ray_params_changed.emit())
        
        cosmic_ray_layout.addWidget(self.removal_params)
        
        # Cosmic ray actions
        cr_buttons = ButtonGroup()
        cr_buttons.add_button_row([
            ("Reprocess Map", self.reprocess_cosmic_rays_requested.emit),
            ("Apply CRE to All", self.apply_cre_to_all_files_requested.emit)
        ])
        cr_buttons.add_button_row([
            ("Statistics", self.show_cosmic_ray_stats.emit),
            ("Diagnostics", self.show_cosmic_ray_diagnostics.emit)
        ])
        cr_buttons.add_button_row([
            ("Test Shape", self.test_shape_analysis_requested.emit),
            ("CR Map", lambda: self.feature_changed.emit("Cosmic Ray Map"))
        ])
        cosmic_ray_layout.addWidget(cr_buttons)
        
        self.layout.addWidget(cosmic_ray_group)
        
        # Template Fitting section (only show if templates are loaded)
        self.template_fitting_group = QGroupBox("Template Fitting")
        template_layout = QVBoxLayout(self.template_fitting_group)
        
        # Template fitting status
        self.template_status_label = QLabel("No templates loaded")
        self.template_status_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        template_layout.addWidget(self.template_status_label)
        
        # Template fitting button
        self.fit_templates_btn = StandardButton("Fit Templates to Map")
        self.fit_templates_btn.clicked.connect(self.fit_templates_to_map_requested.emit)
        self.fit_templates_btn.setEnabled(False)  # Disabled until templates are loaded
        template_layout.addWidget(self.fit_templates_btn)
        
        # Fitting results info
        self.fitting_results_label = QLabel("No fitting results available")
        self.fitting_results_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        template_layout.addWidget(self.fitting_results_label)
        
        self.layout.addWidget(self.template_fitting_group)
        
        # Initially hide template fitting section
        self.template_fitting_group.setVisible(False)
    
    def _get_initial_features(self):
        """Get the initial set of features for the dropdown."""
        return [
            "Integrated Intensity",
            "Peak Height", 
            "Cosmic Ray Map"
        ]
    
    def get_available_features(self) -> list:
        """Get list of available features in the combo box."""
        return [self.feature_combo.itemText(i) for i in range(self.feature_combo.count())]
    
    def update_feature_list(self, features: list):
        """Update the feature combo box with new features."""
        current_feature = self.feature_combo.currentText()
        self.feature_combo.clear()
        self.feature_combo.addItems(features)
        
        # Try to restore previous selection
        if current_feature in features:
            self.feature_combo.setCurrentText(current_feature)
    
    def _on_auto_scale_toggled(self, checked: bool):
        """Handle auto scale toggle."""
        self.intensity_params.setEnabled(not checked)
        if checked:
            # Emit signal to reset to auto scaling
            self.intensity_scaling_changed.emit(float('nan'), float('nan'))
    
    def _emit_intensity_scaling(self):
        """Emit intensity scaling change signal."""
        if not self.auto_scale_cb.isChecked():
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()
            self.intensity_scaling_changed.emit(vmin, vmax)
    
    def _reset_intensity_scaling(self):
        """Reset intensity scaling to auto."""
        self.auto_scale_cb.setChecked(True)
        self.intensity_params.setEnabled(False)
        self.intensity_scaling_changed.emit(float('nan'), float('nan'))
    
    def update_intensity_range(self, vmin: float, vmax: float):
        """Update the intensity range controls with current data range."""
        # Update spinbox ranges and values when auto-scaling
        if self.auto_scale_cb.isChecked():
            self.vmin_spin.setValue(vmin)
            self.vmax_spin.setValue(vmax)
    
    def _on_integration_mode_changed(self, mode: str):
        """Handle integration mode changes."""
        if mode == "Custom Range":
            self.range_controls.setVisible(True)
            # Emit initial range when switching to custom mode
            self._update_wavenumber_range()
        else:
            self.range_controls.setVisible(False)
            # Emit signal to use full spectrum integration
            self.wavenumber_range_changed.emit(0.0, 0.0)  # Special values for full spectrum
    
    def _update_wavenumber_range(self):
        """Update wavenumber range display and emit signal."""
        center = self.center_wavenumber_slider.value()
        width = self.range_width_spin.value()
        
        min_wavenumber = center - width / 2
        max_wavenumber = center + width / 2
        
        # Update range display label (center value is now shown in tooltip)
        self.range_display.setText(f"Range: {min_wavenumber:.0f} - {max_wavenumber:.0f} cm⁻¹")
        
        # Emit signal if in custom range mode
        if self.integration_mode.currentText() == "Custom Range":
            self.wavenumber_range_changed.emit(center, width)
    
    def get_integration_mode(self) -> str:
        """Get current integration mode."""
        return self.integration_mode.currentText()
    
    def get_wavenumber_range(self) -> tuple:
        """Get current wavenumber range (center, width)."""
        return self.center_wavenumber_slider.value(), self.range_width_spin.value()
    
    def set_spectrum_midpoint(self, wavenumbers):
        """Set the integration slider to the spectrum midpoint."""
        if wavenumbers is not None and len(wavenumbers) > 0:
            min_wn = wavenumbers.min()
            max_wn = wavenumbers.max()
            midpoint = (min_wn + max_wn) / 2
            
            # Round to nearest 50 cm⁻¹ increment
            midpoint_rounded = round(midpoint / 50) * 50
            
            # Ensure it's within slider range
            midpoint_rounded = max(200, min(4000, midpoint_rounded))
            
            # Update slider and range
            self.center_wavenumber_slider.setValue(int(midpoint_rounded))
            self._update_wavenumber_range()
            
        else:
            # Fallback to default center
            self.center_wavenumber_slider.setValue(1000)
            self._update_wavenumber_range()
    
    def update_slider_range(self, wavenumbers):
        """Update slider range based on actual spectrum data."""
        if wavenumbers is not None and len(wavenumbers) > 0:
            min_wn = int(wavenumbers.min())
            max_wn = int(wavenumbers.max())
            
            # Set slider range with some padding
            self.center_wavenumber_slider.setRange(min_wn, max_wn)
            
            # Update tick intervals based on range
            range_span = max_wn - min_wn
            if range_span > 2000:
                tick_interval = 500
            elif range_span > 1000:
                tick_interval = 200
            else:
                tick_interval = 100
            
            self.center_wavenumber_slider.setTickInterval(tick_interval)


class DimensionalityReductionControlPanel(BaseControlPanel):
    """Combined control panel for PCA and NMF analysis."""
    
    # Signals
    run_pca_requested = Signal()
    run_nmf_requested = Signal()
    rerun_pca_requested = Signal()
    rerun_nmf_requested = Signal()
    save_nmf_requested = Signal()
    load_nmf_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__("Dimensionality Reduction", parent)
        
    def setup_controls(self):
        """Setup combined PCA/NMF analysis controls."""
        
        # Create tab widget for PCA and NMF
        self.tab_widget = QTabWidget()
        
        # === PCA TAB ===
        self.pca_tab = QWidget()
        pca_layout = QVBoxLayout(self.pca_tab)
        pca_layout.setSpacing(6)
        pca_layout.setContentsMargins(4, 4, 4, 4)
        
        # PCA Parameters
        pca_params_group = ParameterGroupBox("PCA Parameters")
        self.pca_n_components_spin = pca_params_group.add_spinbox(
            "Components", 2, 50, 5, width=60)
        self.pca_n_components_spin.setToolTip("Number of principal components to extract")
        pca_layout.addWidget(pca_params_group)
        
        # PCA Info
        pca_info_group = QGroupBox("About PCA")
        pca_info_layout = QVBoxLayout(pca_info_group)
        pca_info_label = QLabel("PCA reduces dimensionality while preserving\n"
                               "the most important variance in spectral data.\n\n"
                               "• Use fewer components for faster analysis\n"
                               "• Use more components for better detail")
        pca_info_label.setStyleSheet("QLabel { color: #666; font-size: 10px; padding: 8px; }")
        pca_info_label.setWordWrap(True)
        pca_info_layout.addWidget(pca_info_label)
        pca_layout.addWidget(pca_info_group)
        
        # PCA Actions
        pca_run_btn = StandardButton("Run PCA Analysis")
        pca_run_btn.clicked.connect(self.run_pca_requested.emit)
        pca_layout.addWidget(pca_run_btn)
        

        pca_rerun_btn = SuccessButton("Re-run PCA (Update Clustering)")
        pca_rerun_btn.clicked.connect(self.rerun_pca_requested.emit)
        pca_layout.addWidget(pca_rerun_btn)
        
        pca_layout.addStretch()  # Push content to top
        
        # === NMF TAB ===
        self.nmf_tab = QWidget()
        nmf_layout = QVBoxLayout(self.nmf_tab)
        nmf_layout.setSpacing(6)
        nmf_layout.setContentsMargins(4, 4, 4, 4)
        
        # NMF Basic Parameters
        nmf_params_group = ParameterGroupBox("NMF Parameters")
        self.nmf_n_components_spin = nmf_params_group.add_spinbox(
            "Components", 2, 20, 5, width=60)
        self.nmf_n_components_spin.setToolTip("Number of NMF components to extract")
        
        self.nmf_max_iter_spin = nmf_params_group.add_spinbox(
            "Max Iterations", 100, 1000, 200, width=60)
        self.nmf_max_iter_spin.setToolTip("Maximum iterations for NMF convergence")
        
        self.nmf_random_state_spin = nmf_params_group.add_spinbox(
            "Random State", 0, 999, 42, width=60)
        self.nmf_random_state_spin.setToolTip("Random seed for reproducible results")
        nmf_layout.addWidget(nmf_params_group)
        
        # NMF Advanced Parameters
        nmf_advanced_group = ParameterGroupBox("Advanced Options")
        self.nmf_batch_size_spin = nmf_advanced_group.add_spinbox(
            "Batch Size", 500, 10000, 2000, width=80)
        self.nmf_batch_size_spin.setToolTip("Maximum samples for fitting (larger datasets)")
        
        # Solver selection
        self.nmf_solver_combo = nmf_advanced_group.add_combobox(
            "Solver", ["mu", "cd"], 0, width=100)
        self.nmf_solver_combo.setToolTip("mu: Multiplicative Update (stable), cd: Coordinate Descent (faster)")
        nmf_layout.addWidget(nmf_advanced_group)
        
        # NMF Info
        nmf_info_group = QGroupBox("About NMF")
        nmf_info_layout = QVBoxLayout(nmf_info_group)
        
        self.nmf_info_text = QTextEdit()
        self.nmf_info_text.setMaximumHeight(80)
        self.nmf_info_text.setReadOnly(True)
        self.nmf_info_text.setPlainText("NMF decomposes spectra into non-negative components.\n"
                                       "Each component represents a spectral signature and\n"
                                       "their spatial distribution in the map.")
        self.nmf_info_text.setStyleSheet("QTextEdit { font-size: 10px; }")
        nmf_info_layout.addWidget(self.nmf_info_text)
        nmf_layout.addWidget(nmf_info_group)
        
        # NMF Actions
        nmf_run_btn = StandardButton("Run NMF Analysis")
        nmf_run_btn.clicked.connect(self.run_nmf_requested.emit)
        nmf_layout.addWidget(nmf_run_btn)
        
        nmf_rerun_btn = SuccessButton("Re-run NMF (Update Clustering)")
        nmf_rerun_btn.clicked.connect(self.rerun_nmf_requested.emit)
        nmf_layout.addWidget(nmf_rerun_btn)
        
        # NMF Save/Load section
        nmf_save_load_group = QGroupBox("Save/Load Results")
        nmf_save_load_layout = QVBoxLayout(nmf_save_load_group)
        
        nmf_save_btn = StandardButton("Save NMF Results")
        nmf_save_btn.clicked.connect(self.save_nmf_requested.emit)
        nmf_save_load_layout.addWidget(nmf_save_btn)
        
        nmf_load_btn = StandardButton("Load NMF Results")
        nmf_load_btn.clicked.connect(self.load_nmf_requested.emit)
        nmf_save_load_layout.addWidget(nmf_load_btn)
        
        nmf_layout.addWidget(nmf_save_load_group)
        nmf_layout.addStretch()  # Push content to top
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.pca_tab, "PCA")
        self.tab_widget.addTab(self.nmf_tab, "NMF")
        
        # Add tab widget to main layout
        self.layout.addWidget(self.tab_widget)
    
    # PCA methods
    def get_pca_n_components(self) -> int:
        """Get the number of PCA components."""
        return self.pca_n_components_spin.value()
    
    # NMF methods
    def get_nmf_n_components(self) -> int:
        """Get the number of NMF components.""" 
        return self.nmf_n_components_spin.value()
        
    def get_nmf_max_iter(self) -> int:
        """Get the maximum iterations for NMF."""
        return self.nmf_max_iter_spin.value()
        
    def get_nmf_random_state(self) -> int:
        """Get the random state for NMF."""
        return self.nmf_random_state_spin.value()
        
    def get_nmf_batch_size(self) -> int:
        """Get the batch size parameter."""
        return self.nmf_batch_size_spin.value()
    
    def get_nmf_solver(self) -> str:
        """Get the selected solver."""
        return self.nmf_solver_combo.currentText()
    
    def update_nmf_info(self, message: str):
        """Update the NMF info text with analysis results."""
        self.nmf_info_text.setPlainText(message)


class PCAControlPanel(BaseControlPanel):
    """Control panel for PCA analysis."""
    
    # Signals
    run_pca_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__("PCA Analysis", parent)
        
    def setup_controls(self):
        """Setup PCA analysis controls."""
        # Parameters
        params_group = ParameterGroupBox("PCA Parameters")
        self.n_components_spin = params_group.add_spinbox(
            "Components", 2, 50, 5)
        self.n_components_spin.setToolTip("Number of principal components to extract")
        
        self.layout.addWidget(params_group)
        
        # Analysis Info
        info_group = QGroupBox("Analysis Info")
        info_layout = QVBoxLayout(info_group)
        
        info_label = QLabel("PCA reduces dimensionality while preserving\n"
                          "the most important variance in the spectral data.\n\n"
                          "Use fewer components for faster analysis,\n"
                          "more components for better detail.")
        info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 10px; }")
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        
        self.layout.addWidget(info_group)
        
        # Actions
        run_btn = StandardButton("Run PCA Analysis")
        run_btn.clicked.connect(self.run_pca_requested.emit)
        self.layout.addWidget(run_btn)


class TemplateControlPanel(BaseControlPanel):
    """Control panel for template analysis."""
    
    # Signals
    load_template_file_requested = Signal()
    load_template_folder_requested = Signal()
    extract_from_map_requested = Signal()  # New signal for map extraction
    remove_template_requested = Signal(int)  # template index
    clear_templates_requested = Signal()
    plot_templates_requested = Signal()
    fit_templates_requested = Signal()
    normalize_templates_requested = Signal()
    show_detailed_stats = Signal()
    export_statistics = Signal()
    show_chemical_analysis = Signal()
            # Removed: show_hybrid_analysis and show_quantitative_calibration signals
    show_pp_analysis = Signal()  # Template-only polypropylene analysis
    
    def __init__(self, parent=None):
        super().__init__("Template Analysis", parent)
        
    def setup_controls(self):
        """Setup template analysis controls."""
        # Template management
        template_group = QGroupBox("Template Management")
        template_layout = QVBoxLayout(template_group)
        
        # Template loading buttons
        load_buttons = ButtonGroup()
        load_buttons.add_button_row([
            ("Load Single File", self.load_template_file_requested.emit),
            ("Load Folder", self.load_template_folder_requested.emit)
        ])
        template_layout.addWidget(load_buttons)
        
        # Extract from Map button on separate line
        extract_buttons = ButtonGroup()
        extract_buttons.add_button_row([
            ("📍 Extract from Map", self.extract_from_map_requested.emit)
        ])
        template_layout.addWidget(extract_buttons)
        
        # Template list box
        list_label = QLabel("Loaded Templates:")
        template_layout.addWidget(list_label)
        
        self.template_list = QListWidget()
        self.template_list.setMaximumHeight(150)
        self.template_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        template_layout.addWidget(self.template_list)
        
        # Template management buttons
        manage_buttons = ButtonGroup()
        manage_buttons.add_button_row([
            ("Remove Selected", self._on_remove_template),
            ("Clear All", self.clear_templates_requested.emit)
        ])
        template_layout.addWidget(manage_buttons)
        
        self.layout.addWidget(template_group)
        
        # Template Statistics (only shown after fitting)
        self.stats_group = QGroupBox("Template Fitting Statistics")
        self.stats_group.setVisible(False)  # Hidden until fitting is complete
        stats_layout = QVBoxLayout(self.stats_group)
        
        # Statistics display area
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        stats_layout.addWidget(self.stats_text)
        
        # Statistics control buttons
        stats_buttons = ButtonGroup()
        stats_buttons.add_button_row([
            ("Show Detailed Stats", self.show_detailed_stats),
            ("Export Statistics", self.export_statistics)
        ])
        stats_layout.addWidget(stats_buttons)
        
        # Chemical analysis button
        chemical_analysis_btn = StandardButton("Chemical Validity Analysis")
        chemical_analysis_btn.clicked.connect(self.show_chemical_analysis)
        stats_layout.addWidget(chemical_analysis_btn)
        
        # Removed: Hybrid analysis and Quantitative calibration buttons
        
        # Template-only material analysis button
        material_analysis_btn = StandardButton("📊 Material Analysis (Template Only)")
        material_analysis_btn.clicked.connect(self.show_pp_analysis)
        material_analysis_btn.setToolTip("Calculate material statistics from template fitting only (bypasses NMF)")
        stats_layout.addWidget(material_analysis_btn)
        
        self.layout.addWidget(self.stats_group)
        
        # Normalization options
        norm_group = QGroupBox("Normalization")
        norm_layout = QVBoxLayout(norm_group)
        
        self.normalize_cb = QCheckBox("Normalize Templates")
        self.normalize_cb.setChecked(True)
        self.normalize_cb.setToolTip("Normalize template intensities to [0,1] range")
        norm_layout.addWidget(self.normalize_cb)
        
        self.norm_method_combo = QComboBox()
        self.norm_method_combo.addItems([
            "Min-Max (0-1)",
            "Max Normalization", 
            "Area Normalization",
            "Standard Normalization"
        ])
        self.norm_method_combo.setToolTip("Choose normalization method")
        norm_layout.addWidget(self.norm_method_combo)
        
        normalize_btn = StandardButton("Apply Normalization")
        normalize_btn.clicked.connect(self.normalize_templates_requested.emit)
        norm_layout.addWidget(normalize_btn)
        
        self.layout.addWidget(norm_group)
        
        # Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        plot_templates_btn = StandardButton("Plot All Templates")
        plot_templates_btn.clicked.connect(self.plot_templates_requested.emit)
        viz_layout.addWidget(plot_templates_btn)
        
        # Display options
        self.show_raw_cb = QCheckBox("Show Raw Data")
        self.show_raw_cb.setChecked(False)
        viz_layout.addWidget(self.show_raw_cb)
        
        self.show_processed_cb = QCheckBox("Show Processed Data")
        self.show_processed_cb.setChecked(True)
        viz_layout.addWidget(self.show_processed_cb)
        
        self.layout.addWidget(viz_group)
        
        # Fitting options
        fitting_group = ParameterGroupBox("Fitting Options")
        self.method_combo = fitting_group.add_combobox(
            "Method", ["NNLS", "LSQR"], 0)
        self.use_baseline_cb = fitting_group.add_checkbox("Use Baseline", True)
        
        self.layout.addWidget(fitting_group)
        
        # Actions
        fit_btn = StandardButton("Fit Templates to Map")
        fit_btn.clicked.connect(self.fit_templates_requested.emit)
        self.layout.addWidget(fit_btn)
        
    def _on_remove_template(self):
        """Handle remove template button click."""
        current_row = self.template_list.currentRow()
        if current_row >= 0:
            self.remove_template_requested.emit(current_row)
    
    def update_template_list(self, template_names: list):
        """Update the template list widget."""
        self.template_list.clear()
        for name in template_names:
            self.template_list.addItem(name)
    
    def get_normalization_method(self) -> str:
        """Get the selected normalization method."""
        return self.norm_method_combo.currentText()
    
    def is_normalization_enabled(self) -> bool:
        """Check if normalization is enabled."""
        return self.normalize_cb.isChecked()
    
    def get_fitting_method(self) -> str:
        """Get the selected fitting method."""
        return self.method_combo.currentText().lower()
    
    def use_baseline_fitting(self) -> bool:
        """Check if baseline fitting is enabled."""
        return self.use_baseline_cb.isChecked()
    
    def show_raw_data(self) -> bool:
        """Check if raw data should be shown."""
        return self.show_raw_cb.isChecked()
    
    def show_processed_data(self) -> bool:
        """Check if processed data should be shown."""
        return self.show_processed_cb.isChecked()
    
    def _show_detailed_stats(self):
        """Handle show detailed stats button click."""
        self.show_detailed_stats.emit()
    
    def _export_statistics(self):
        """Handle export statistics button click."""
        self.export_statistics.emit()
    
    def update_statistics_display(self, stats_text: str):
        """Update the statistics display area."""
        self.stats_text.setPlainText(stats_text)
        self.stats_group.setVisible(True)
    
    def hide_statistics(self):
        """Hide the statistics group."""
        self.stats_group.setVisible(False)


class NMFControlPanel(BaseControlPanel):
    """Control panel for NMF analysis."""
    
    # Signals
    run_nmf_requested = Signal()
    save_nmf_requested = Signal()
    load_nmf_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__("NMF Analysis", parent)
        
    def setup_controls(self):
        """Setup NMF analysis controls."""
        # Basic Parameters
        params_group = ParameterGroupBox("NMF Parameters")
        self.n_components_spin = params_group.add_spinbox(
            "Components", 2, 20, 5)
        self.n_components_spin.setToolTip("Number of NMF components to extract")
        
        self.max_iter_spin = params_group.add_spinbox(
            "Max Iterations", 100, 1000, 200)
        self.max_iter_spin.setToolTip("Maximum iterations for NMF convergence")
        
        self.random_state_spin = params_group.add_spinbox(
            "Random State", 0, 999, 42)
        self.random_state_spin.setToolTip("Random seed for reproducible results")
        
        self.layout.addWidget(params_group)
        
        # Advanced Parameters
        advanced_group = ParameterGroupBox("Advanced Options")
        self.batch_size_spin = advanced_group.add_spinbox(
            "Batch Size", 500, 10000, 2000)
        self.batch_size_spin.setToolTip("Maximum samples for fitting (larger datasets)")
        
        # Solver selection
        from PySide6.QtWidgets import QComboBox, QLabel, QHBoxLayout, QWidget
        solver_widget = QWidget()
        solver_layout = QHBoxLayout(solver_widget)
        solver_layout.setContentsMargins(0, 0, 0, 0)
        
        solver_label = QLabel("Solver:")
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["mu", "cd"])
        self.solver_combo.setCurrentText("mu")
        self.solver_combo.setToolTip("mu: Multiplicative Update (stable), cd: Coordinate Descent (faster)")
        
        solver_layout.addWidget(solver_label)
        solver_layout.addWidget(self.solver_combo)
        solver_layout.addStretch()
        
        advanced_group.layout.addWidget(solver_widget)
        
        self.layout.addWidget(advanced_group)
        
        # Analysis Info
        info_group = QGroupBox("Analysis Info")
        info_layout = QVBoxLayout(info_group)
        
        from PySide6.QtWidgets import QTextEdit
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("NMF decomposes spectra into non-negative components.\n"
                                   "Each component represents a spectral signature and\n"
                                   "their spatial distribution in the map.")
        info_layout.addWidget(self.info_text)
        
        self.layout.addWidget(info_group)
        
        # Actions
        run_btn = StandardButton("Run NMF Analysis")
        run_btn.clicked.connect(self.run_nmf_requested.emit)
        self.layout.addWidget(run_btn)
        
        # Save/Load section
        save_load_group = QGroupBox("Save/Load Results")
        save_load_layout = QVBoxLayout(save_load_group)
        
        save_btn = StandardButton("Save NMF Results")
        save_btn.clicked.connect(self.save_nmf_requested.emit)
        save_load_layout.addWidget(save_btn)
        
        load_btn = StandardButton("Load NMF Results")
        load_btn.clicked.connect(self.load_nmf_requested.emit)
        save_load_layout.addWidget(load_btn)
        
        self.layout.addWidget(save_load_group)
    
    def get_batch_size(self) -> int:
        """Get the batch size parameter."""
        return self.batch_size_spin.value()
    
    def get_solver(self) -> str:
        """Get the selected solver."""
        return self.solver_combo.currentText()
    
    def update_info(self, message: str):
        """Update the info text with analysis results."""
        self.info_text.setPlainText(message)


class MLControlPanel(BaseControlPanel):
    """Control panel for ML classification and clustering."""
    
    # Signals
    train_supervised_requested = Signal()
    train_unsupervised_requested = Signal()
    classify_map_requested = Signal()
    load_training_data_requested = Signal()
    use_clustering_labels_requested = Signal()
    save_named_model_requested = Signal()
    load_model_requested = Signal()
    remove_model_requested = Signal()
    apply_selected_model_requested = Signal()
    model_selection_changed = Signal(str)
    show_feature_info_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__("ML Analysis", parent)
        
    def setup_controls(self):
        """Setup ML analysis controls with tabbed interface."""
        # === WORKFLOW GUIDE ===
        guide_group = QGroupBox("📋 Quick Start")
        guide_layout = QVBoxLayout(guide_group)
        
        guide_text = QLabel(
            "1️⃣ Choose tab (Classification/Clustering)\n"
            "2️⃣ Set data & params → 3️⃣ Train → 4️⃣ Apply\n"
            "5️⃣ Save model for reuse"
        )
        guide_text.setStyleSheet("color: #666; font-size: 10px; padding: 4px;")
        guide_layout.addWidget(guide_text)
        
        self.layout.addWidget(guide_group)
        
        # === TABBED INTERFACE ===
        from PySide6.QtWidgets import QTabWidget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.supervised_tab = QWidget()
        self.unsupervised_tab = QWidget()
        
        self.tab_widget.addTab(self.supervised_tab, "🔍 Classification")
        self.tab_widget.addTab(self.unsupervised_tab, "🧩 Clustering")
        
        self.layout.addWidget(self.tab_widget)
        
        # Setup each tab
        self._setup_supervised_tab()
        self._setup_unsupervised_tab()
        
        # === MODEL MANAGEMENT ===
        self._setup_model_management()
        
        # Results Info
        info_group = QGroupBox("ℹ️ Status")
        info_layout = QVBoxLayout(info_group)
        
        from PySide6.QtWidgets import QTextEdit
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(80)  # Compact height
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Ready! Choose Classification or Clustering tab above.")
        info_layout.addWidget(self.info_text)
        
        self.layout.addWidget(info_group)
    
    def _setup_supervised_tab(self):
        """Setup the supervised learning tab with condensed layout."""
        supervised_layout = QVBoxLayout(self.supervised_tab)
        supervised_layout.setSpacing(10)  # Normal spacing
        supervised_layout.setContentsMargins(6, 6, 6, 6)  # Normal margins
        
        # Training data setup - condensed
        training_data_group = QGroupBox("📁 Data")
        training_data_layout = QVBoxLayout(training_data_group)
        
        load_data_btn = PrimaryButton("📂 Load Training Folders")
        load_data_btn.clicked.connect(self.load_training_data_requested.emit)
        load_data_btn.setMaximumWidth(200)  # Compact width
        training_data_layout.addWidget(load_data_btn)
        
        # Add button to use clustering as training labels
        use_clustering_btn = InfoButton("🧩 Use Clustering Labels")
        use_clustering_btn.clicked.connect(self.use_clustering_labels_requested.emit)
        use_clustering_btn.setMaximumWidth(200)  # Compact width
        use_clustering_btn.setToolTip("Use PCA/NMF/ML clustering results as training labels")
        training_data_layout.addWidget(use_clustering_btn)
        
        self.training_data_label = QLabel("No data loaded")
        self.training_data_label.setStyleSheet("QLabel { color: #666; font-size: 9px; }")
        training_data_layout.addWidget(self.training_data_label)
        
        # Feature options - condensed
        feature_group = QGroupBox("📊 Features")
        feature_layout = QVBoxLayout(feature_group)
        
        self.use_raw_features_cb = QCheckBox("✅ Full Spectrum")
        self.use_raw_features_cb.setChecked(True)
        self.use_raw_features_cb.setToolTip("Use all wavenumber points")
        feature_layout.addWidget(self.use_raw_features_cb)
        
        self.use_pca_features_cb = QCheckBox("📈 PCA Features")
        self.use_pca_features_cb.setToolTip("Use PCA components (run PCA first)")
        feature_layout.addWidget(self.use_pca_features_cb)
        
        self.use_nmf_features_cb = QCheckBox("🧩 NMF Features")
        self.use_nmf_features_cb.setToolTip("Use NMF components (run NMF first)")
        feature_layout.addWidget(self.use_nmf_features_cb)
        
        # Model parameters - better spaced
        params_group = QGroupBox("⚙️ Parameters")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(8)  # Reasonable spacing between elements
        
        params_label = QLabel("🤖 Random Forest")
        params_label.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        params_layout.addWidget(params_label)
        
        # Add some space after the label
        params_layout.addSpacing(8)
        
        # Parameters with natural sizing
        supervised_params = ParameterGroupBox("Settings")
        # Remove aggressive spacing and let the grid layout work naturally
        supervised_params.layout.setSpacing(8)  # Normal spacing between parameters
        supervised_params.layout.setContentsMargins(12, 12, 12, 12)  # Normal padding
        
        self.n_estimators_spin = supervised_params.add_spinbox("Trees", 10, 1000, 100)
        self.max_depth_spin = supervised_params.add_spinbox("Depth", 3, 100, 10)
        self.test_size_spin = supervised_params.add_double_spinbox("Test Size", 0.1, 0.5, 0.2, 0.1)
        
        params_layout.addWidget(supervised_params)
        
        # Training actions - normal sizing
        train_group = QGroupBox("🚀 Train & Apply")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(8)  # Normal spacing between buttons
        train_layout.setContentsMargins(8, 8, 8, 8)  # Normal padding
        
        self.train_supervised_btn = SuccessButton("🎯 Train Model")
        self.train_supervised_btn.clicked.connect(self.train_supervised_requested.emit)
        train_layout.addWidget(self.train_supervised_btn)
        
        self.classify_btn = WarningButton("🗺️ Apply to Map")
        self.classify_btn.clicked.connect(self.classify_map_requested.emit)
        self.classify_btn.setEnabled(False)
        train_layout.addWidget(self.classify_btn)
        
        self.feature_info_btn = InfoButton("📊 Feature Info")
        self.feature_info_btn.clicked.connect(self.show_feature_info_requested.emit)
        self.feature_info_btn.setToolTip("Show information about ML features being used")
        train_layout.addWidget(self.feature_info_btn)
        
        # Add all groups to layout with normal spacing
        supervised_layout.addWidget(training_data_group)
        supervised_layout.addWidget(feature_group)
        supervised_layout.addWidget(params_group)
        supervised_layout.addWidget(train_group)
        supervised_layout.addStretch()  # Push everything to top
    
    def _setup_unsupervised_tab(self):
        """Setup the unsupervised learning tab with condensed layout."""
        unsupervised_layout = QVBoxLayout(self.unsupervised_tab)
        
        # Algorithm selection - condensed
        algo_group = QGroupBox("🤖 Algorithm")
        algo_layout = QVBoxLayout(algo_group)
        
        self.clustering_method_combo = QComboBox()
        self.clustering_method_combo.addItems([
            "K-Means", "Gaussian Mixture", "DBSCAN", "Hierarchical"
        ])
        self.clustering_method_combo.setMaximumWidth(150)
        self.clustering_method_combo.setToolTip("Choose clustering algorithm")
        algo_layout.addWidget(self.clustering_method_combo)
        
        # Parameters - condensed
        cluster_params = ParameterGroupBox("Settings")
        self.n_clusters_spin = cluster_params.add_spinbox("Clusters", 2, 20, 3)
        self.n_clusters_spin.setMaximumWidth(70)
        
        self.eps_spin = cluster_params.add_double_spinbox("Eps", 0.1, 10.0, 0.5, 0.1)
        self.eps_spin.setMaximumWidth(70)
        
        self.min_samples_spin = cluster_params.add_spinbox("Min Samples", 2, 20, 5)
        self.min_samples_spin.setMaximumWidth(70)
        
        # Training action - condensed
        action_group = QGroupBox("🚀 Run")
        action_layout = QVBoxLayout(action_group)
        
        self.train_unsupervised_btn = PrimaryButton("🧩 Find Clusters")
        self.train_unsupervised_btn.clicked.connect(self.train_unsupervised_requested.emit)
        self.train_unsupervised_btn.setMaximumWidth(150)
        action_layout.addWidget(self.train_unsupervised_btn)
        
        # Add all groups to layout
        unsupervised_layout.addWidget(algo_group)
        unsupervised_layout.addWidget(cluster_params)
        unsupervised_layout.addWidget(action_group)
        unsupervised_layout.addStretch()  # Push everything to top
    
    def _setup_model_management(self):
        """Setup model management section with icon buttons."""
        model_mgmt_group = QGroupBox("💾 Models")
        model_mgmt_layout = QVBoxLayout(model_mgmt_group)
        
        # Model naming - condensed
        model_mgmt_layout.addWidget(QLabel("Name:"))
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Model_v1")
        self.model_name_edit.setMaximumHeight(25)
        model_mgmt_layout.addWidget(self.model_name_edit)
        
        # Icon-only buttons for save/load/remove
        from PySide6.QtWidgets import QPushButton

        
        # Save button - icon only
        save_model_btn = QPushButton("💾")
        apply_icon_button_style(save_model_btn)
        save_model_btn.clicked.connect(self.save_named_model_requested.emit)
        save_model_btn.setToolTip("Save model")
        save_model_btn.setMaximumWidth(40)
        save_model_btn.setMaximumHeight(30)
        
        # Model selection
        model_mgmt_layout.addWidget(QLabel("Select:"))
        self.trained_models_combo = QComboBox()
        self.trained_models_combo.setMaximumHeight(25)
        self.trained_models_combo.addItem("No models")
        self.trained_models_combo.currentTextChanged.connect(self.model_selection_changed.emit)
        model_mgmt_layout.addWidget(self.trained_models_combo)
        
        # Icon buttons layout
        button_layout = QHBoxLayout()
        
        # Load button - icon only
        load_model_btn = QPushButton("📂")
        apply_icon_button_style(load_model_btn)
        load_model_btn.clicked.connect(self.load_model_requested.emit)
        load_model_btn.setToolTip("Load model")
        load_model_btn.setMaximumWidth(40)
        load_model_btn.setMaximumHeight(30)
        
        # Remove button - icon only
        remove_model_btn = QPushButton("🗑️")
        apply_icon_button_style(remove_model_btn)
        remove_model_btn.clicked.connect(self.remove_model_requested.emit)
        remove_model_btn.setToolTip("Remove model")
        remove_model_btn.setMaximumWidth(40)
        remove_model_btn.setMaximumHeight(30)
        
        button_layout.addWidget(save_model_btn)
        button_layout.addWidget(load_model_btn)
        button_layout.addWidget(remove_model_btn)
        button_layout.addStretch()
        
        model_mgmt_layout.addLayout(button_layout)
        
        # Apply model button
        apply_model_btn = SuccessButton("📈 Apply Selected")
        apply_model_btn.clicked.connect(self.apply_selected_model_requested.emit)
        apply_model_btn.setMaximumWidth(150)
        model_mgmt_layout.addWidget(apply_model_btn)
        
        self.layout.addWidget(model_mgmt_group)
        
    def _on_analysis_type_changed(self):
        """Handle analysis type changes - no longer needed with tabs."""
        pass
    
    def get_analysis_type(self) -> str:
        """Get the selected analysis type based on current tab."""
        if hasattr(self, 'tab_widget'):
            current_index = self.tab_widget.currentIndex()
            if current_index == 0:
                return "Supervised Classification"
            else:
                return "Unsupervised Clustering"
        return "Supervised Classification"  # Default fallback
    
    def get_supervised_model(self) -> str:
        """Get the selected supervised model."""
        return "Random Forest"  # Fixed to Random Forest in the new design
    
    def get_clustering_method(self) -> str:
        """Get the selected clustering method."""
        return self.clustering_method_combo.currentText()
    
    def get_n_estimators(self) -> int:
        """Get number of estimators for supervised learning."""
        return self.n_estimators_spin.value()
    
    def get_max_depth(self) -> int:
        """Get max depth for tree-based models."""
        return self.max_depth_spin.value()
    
    def get_test_size(self) -> float:
        """Get test size for supervised learning."""
        return self.test_size_spin.value()
    
    def get_n_clusters(self) -> int:
        """Get number of clusters for clustering."""
        return self.n_clusters_spin.value()
    
    def get_eps(self) -> float:
        """Get eps parameter for DBSCAN."""
        return self.eps_spin.value()
    
    def get_min_samples(self) -> int:
        """Get min samples for DBSCAN."""
        return self.min_samples_spin.value()
    
    def get_feature_options(self) -> Dict[str, bool]:
        """Get selected feature options."""
        return {
            'use_raw': self.use_raw_features_cb.isChecked(),
            'use_pca': self.use_pca_features_cb.isChecked(),
            'use_nmf': self.use_nmf_features_cb.isChecked()
        }
    
    def update_training_data_info(self, info: str):
        """Update training data information display."""
        self.training_data_label.setText(info)
    
    def update_info(self, message: str):
        """Update the info text with analysis results."""
        self.info_text.setPlainText(message)
    
    def get_model_name(self) -> str:
        """Get the model name from the input field."""
        return self.model_name_edit.text().strip()
    
    def clear_model_name(self):
        """Clear the model name input field."""
        self.model_name_edit.clear()
    
    def get_selected_model(self) -> str:
        """Get the currently selected model name."""
        current_text = self.trained_models_combo.currentText()
        return current_text if current_text != "No models loaded" else ""
    
    def update_model_list(self, model_names: list):
        """Update the list of available models."""
        self.trained_models_combo.clear()
        if model_names:
            self.trained_models_combo.addItems(model_names)
        else:
            self.trained_models_combo.addItem("No models loaded")
    
    def add_model_to_list(self, model_name: str):
        """Add a new model to the list."""
        # Remove "No models loaded" if it exists
        if self.trained_models_combo.count() == 1 and self.trained_models_combo.itemText(0) == "No models loaded":
            self.trained_models_combo.clear()
        
        # Add the new model and select it
        self.trained_models_combo.addItem(model_name)
        self.trained_models_combo.setCurrentText(model_name)
    
    def remove_selected_model(self):
        """Remove the currently selected model from the list."""
        current_index = self.trained_models_combo.currentIndex()
        if current_index >= 0 and self.trained_models_combo.currentText() != "No models loaded":
            self.trained_models_combo.removeItem(current_index)
            
            # Add "No models loaded" if list is empty
            if self.trained_models_combo.count() == 0:
                self.trained_models_combo.addItem("No models loaded")
    
    def enable_classify_button(self):
        """Enable the classify button after successful training."""
        if hasattr(self, 'classify_btn'):
            self.classify_btn.setEnabled(True)
            self.classify_btn.setText("🗺️ Apply Model to Map")
    
    def disable_classify_button(self):
        """Disable the classify button when no model is available."""
        if hasattr(self, 'classify_btn'):
            self.classify_btn.setEnabled(False)
            self.classify_btn.setText("🗺️ Train Model First")


class ResultsControlPanel(BaseControlPanel):
    """Control panel for results summary."""
    
    # Signals
    generate_report_requested = Signal()
    export_results_requested = Signal()
    run_quantitative_analysis_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__("Results Summary", parent)
        
    def setup_controls(self):
        """Setup results controls."""
        # Results Summary Info
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        summary_label = QLabel("This section provides tools to analyze and export\n"
                             "results from all analysis methods:\n\n"
                             "🎯 Quantitative Analysis: Combines template,\n"
                             "   NMF, and ML results with automatic class\n"
                             "   flip detection. Shows top 5 best spectra.\n\n"
                             "📊 Export tools for comprehensive reporting:\n"
                             "• Map analysis data • Template fitting results\n" 
                             "• PCA decomposition • NMF components\n"
                             "• ML classification results")
        summary_label.setStyleSheet("QLabel { color: #666; font-size: 11px; padding: 10px; }")
        summary_label.setWordWrap(True)
        summary_layout.addWidget(summary_label)
        
        self.layout.addWidget(summary_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_statistics_cb = QCheckBox("Include Statistical Analysis")
        self.show_statistics_cb.setChecked(True)
        self.show_statistics_cb.setToolTip("Include numerical statistics in reports")
        display_layout.addWidget(self.show_statistics_cb)
        
        self.show_plots_cb = QCheckBox("Include Summary Plots")
        self.show_plots_cb.setChecked(True)
        self.show_plots_cb.setToolTip("Include visualization plots in reports")
        display_layout.addWidget(self.show_plots_cb)
        
        self.include_raw_data_cb = QCheckBox("Include Raw Data")
        self.include_raw_data_cb.setChecked(False)
        self.include_raw_data_cb.setToolTip("Include raw spectral data in exports")
        display_layout.addWidget(self.include_raw_data_cb)
        
        self.layout.addWidget(display_group)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)
        
        format_label = QLabel("Export Format:")
        export_layout.addWidget(format_label)
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["PDF Report", "PNG Images", "CSV Data", "Excel Workbook", "All Formats"])
        self.export_format_combo.setToolTip("Choose export format for results")
        export_layout.addWidget(self.export_format_combo)
        
        self.layout.addWidget(export_group)
        
        # Actions
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        # NEW: Quantitative analysis button
        quantitative_btn = StandardButton("🎯 Run Quantitative Analysis")
        quantitative_btn.clicked.connect(self.run_quantitative_analysis_requested.emit)
        quantitative_btn.setToolTip("Run robust quantitative analysis and show top 5 best-fitting spectra")
        quantitative_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        action_layout.addWidget(quantitative_btn)
        
        report_btn = StandardButton("Generate Comprehensive Report")
        report_btn.clicked.connect(self.generate_report_requested.emit)
        report_btn.setToolTip("Generate a complete analysis report")
        action_layout.addWidget(report_btn)
        
        export_btn = StandardButton("Export Selected Results")
        export_btn.clicked.connect(self.export_results_requested.emit)
        export_btn.setToolTip("Export results in the selected format")
        action_layout.addWidget(export_btn)
        
        self.layout.addWidget(action_group)
