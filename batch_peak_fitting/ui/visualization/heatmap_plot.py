"""
Heatmap Plot Component
Shows 2D heatmap visualization of spectral data across multiple spectra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, 
                              QComboBox, QLabel, QSpinBox, QDoubleSpinBox, 
                              QGroupBox, QSlider, QPushButton)
from PySide6.QtCore import Qt

from .base_plot import BasePlot


class HeatmapPlot(BasePlot):
    """
    Heatmap plot showing 2D visualization of spectral data
    """
    
    def __init__(self):
        super().__init__("heatmap", "Heatmap Plot - 2D Spectral Data")
        
        # Plot-specific settings
        self.settings.update({
            'data_type': 'Raw Intensity',  # Raw Intensity, Background Corrected, Fitted Peaks, Residuals
            'colormap': 'viridis',
            'auto_normalize': False,
            'auto_range': True,
            'x_min': 200,
            'x_max': 1800,
            'resolution': 200,  # Number of wavenumber points
            'contrast': 1.0,
            'brightness': 0.0,
            'gamma': 1.0,
            'show_colorbar': True,
            'interpolation': 'bilinear',  # nearest, bilinear, bicubic
            'auto_update_on_data_change': False  # OPTIMIZATION: Only update when batch processing completes
        })
        
        # Data cache
        self.batch_results = []
        self.loaded_spectra = []
        self.heatmap_data = None
        
    def create_controls(self):
        """Create heatmap-specific controls"""
        controls_widget = QWidget()
        main_layout = QHBoxLayout(controls_widget)
        
        # Data selection group
        data_group = QGroupBox("Data")
        data_layout = QVBoxLayout(data_group)
        
        # Data type selection
        data_layout.addWidget(QLabel("Data Type:"))
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(['Raw Intensity', 'Background Corrected', 'Fitted Peaks', 'Residuals'])
        self.data_type_combo.setCurrentText(self.settings['data_type'])
        self.data_type_combo.currentTextChanged.connect(self.on_data_type_changed)
        data_layout.addWidget(self.data_type_combo)
        
        # Resolution
        data_layout.addWidget(QLabel("Resolution:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setMinimum(50)
        self.resolution_spin.setMaximum(1000)
        self.resolution_spin.setValue(self.settings['resolution'])
        self.resolution_spin.valueChanged.connect(self.on_resolution_changed)
        data_layout.addWidget(self.resolution_spin)
        
        main_layout.addWidget(data_group)
        
        # Display options group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        
        # Colormap
        display_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'coolwarm', 'RdYlBu'])
        self.colormap_combo.setCurrentText(self.settings['colormap'])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        display_layout.addWidget(self.colormap_combo)
        
        # Interpolation
        display_layout.addWidget(QLabel("Interpolation:"))
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(['nearest', 'bilinear', 'bicubic'])
        self.interpolation_combo.setCurrentText(self.settings['interpolation'])
        self.interpolation_combo.currentTextChanged.connect(self.on_interpolation_changed)
        display_layout.addWidget(self.interpolation_combo)
        
        # Options
        self.auto_normalize_check = QCheckBox("Auto Normalize")
        self.auto_normalize_check.setChecked(self.settings['auto_normalize'])
        self.auto_normalize_check.stateChanged.connect(self.on_normalize_toggled)
        display_layout.addWidget(self.auto_normalize_check)
        
        self.show_colorbar_check = QCheckBox("Show Colorbar")
        self.show_colorbar_check.setChecked(self.settings['show_colorbar'])
        self.show_colorbar_check.stateChanged.connect(self.on_colorbar_toggled)
        display_layout.addWidget(self.show_colorbar_check)
        
        main_layout.addWidget(display_group)
        
        # Range group
        range_group = QGroupBox("Range")
        range_layout = QVBoxLayout(range_group)
        
        self.auto_range_check = QCheckBox("Auto Range")
        self.auto_range_check.setChecked(self.settings['auto_range'])
        self.auto_range_check.stateChanged.connect(self.on_auto_range_toggled)
        range_layout.addWidget(self.auto_range_check)
        
        # X range
        x_range_layout = QHBoxLayout()
        x_range_layout.addWidget(QLabel("X Range:"))
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setMinimum(0)
        self.x_min_spin.setMaximum(4000)
        self.x_min_spin.setValue(self.settings['x_min'])
        self.x_min_spin.valueChanged.connect(self.on_range_changed)
        x_range_layout.addWidget(self.x_min_spin)
        
        x_range_layout.addWidget(QLabel("to"))
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setMinimum(0)
        self.x_max_spin.setMaximum(4000)
        self.x_max_spin.setValue(self.settings['x_max'])
        self.x_max_spin.valueChanged.connect(self.on_range_changed)
        x_range_layout.addWidget(self.x_max_spin)
        
        range_layout.addLayout(x_range_layout)
        
        # Enhancement controls
        enhance_layout = QVBoxLayout()
        
        # Contrast
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setMinimum(0.1)
        self.contrast_spin.setMaximum(5.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(self.settings['contrast'])
        self.contrast_spin.valueChanged.connect(self.on_contrast_changed)
        contrast_layout.addWidget(self.contrast_spin)
        enhance_layout.addLayout(contrast_layout)
        
        range_layout.addLayout(enhance_layout)
        main_layout.addWidget(range_group)
        
        # Update button group
        update_group = QGroupBox("Update")
        update_layout = QVBoxLayout(update_group)
        
        update_btn = QPushButton("Update Heatmap")
        update_btn.clicked.connect(self.update_plot)
        update_btn.setToolTip("Update heatmap plot with current data")
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC2626;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #B91C1C;
            }
        """)
        update_layout.addWidget(update_btn)
        
        main_layout.addWidget(update_group)
        
        return controls_widget
    
    def initialize_plot(self):
        """Initialize the heatmap plot"""
        if not self.figure:
            return
            
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        
        # Setup axes
        self.axes.set_xlabel('Wavenumber (cm⁻¹)')
        self.axes.set_ylabel('Spectrum Index')
        self.axes.set_title(self.title)
        
        # Apply tight layout (with error handling for matplotlib warnings)
        try:
            self.figure.tight_layout()
        except (UserWarning, ValueError):
            # Some matplotlib configurations may not be compatible with tight_layout
            pass
        self.canvas.draw()
    
    def plot_data_on_axes(self):
        """Plot heatmap data on the axes"""
        if not self.axes:
            return
            
        # Update data cache
        self.update_data_cache()
        
        if not self.loaded_spectra and not self.batch_results:
            self.axes.text(0.5, 0.5, 'No spectra available\nLoad files or run batch processing', 
                          transform=self.axes.transAxes, 
                          ha='center', va='center', fontsize=12)
            return
        
        # Choose data source
        data_source = self.loaded_spectra if self.loaded_spectra else self.batch_results
        data_type = self.settings['data_type']
        
        if not data_source:
            self.axes.text(0.5, 0.5, 'No valid data for heatmap', 
                          transform=self.axes.transAxes, 
                          ha='center', va='center', fontsize=12)
            return
        
        # Collect data for heatmap
        heatmap_data = []
        wavenumber_ranges = []
        
        for result in data_source:
            wavenumbers = np.array(result.get('wavenumbers', []))
            intensities = self.extract_intensities(result, data_type)
            
            if len(wavenumbers) > 0 and len(intensities) > 0:
                heatmap_data.append(intensities)
                wavenumber_ranges.append(wavenumbers)
        
        if not heatmap_data:
            if data_type == "Residuals":
                self.axes.text(0.5, 0.5, 'No residuals data available\n\nRun batch processing with\nreference peaks to generate residuals', 
                              transform=self.axes.transAxes, 
                              ha='center', va='center', fontsize=12)
            else:
                self.axes.text(0.5, 0.5, 'No valid data for heatmap', 
                              transform=self.axes.transAxes, 
                              ha='center', va='center', fontsize=12)
            return
        
        # Determine wavenumber range
        if self.settings['auto_range']:
            all_wavenumbers = np.concatenate(wavenumber_ranges)
            wn_min, wn_max = np.min(all_wavenumbers), np.max(all_wavenumbers)
        else:
            wn_min, wn_max = self.settings['x_min'], self.settings['x_max']
        
        # Create common wavenumber grid
        resolution = self.settings['resolution']
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
            self.axes.text(0.5, 0.5, 'No data in selected range', 
                          transform=self.axes.transAxes, 
                          ha='center', va='center', fontsize=12)
            return
        
        # Convert to 2D array
        heatmap_array = np.array(interpolated_data)
        
        # Apply normalization if requested
        if self.settings['auto_normalize']:
            # Normalize each spectrum individually
            for i in range(heatmap_array.shape[0]):
                spectrum = heatmap_array[i, :]
                if np.max(spectrum) > np.min(spectrum):
                    heatmap_array[i, :] = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
        
        # Apply contrast enhancement
        if self.settings['contrast'] != 1.0:
            heatmap_array = np.power(heatmap_array, 1.0 / self.settings['contrast'])
        
        # Create the heatmap
        im = self.axes.imshow(heatmap_array, 
                             cmap=self.settings['colormap'],
                             aspect='auto',
                             interpolation=self.settings['interpolation'],
                             extent=[wn_min, wn_max, len(heatmap_array), 0],
                             origin='upper')
        
        # Add colorbar if requested
        if self.settings['show_colorbar']:
            if hasattr(self, 'colorbar'):
                self.colorbar.remove()
            self.colorbar = self.figure.colorbar(im, ax=self.axes)
            self.colorbar.set_label('Intensity')
        
        # Set axis limits
        self.axes.set_xlim(wn_min, wn_max)
        self.axes.set_ylim(len(heatmap_data), 0)  # Reverse Y-axis so first spectrum is at top
        
        # Store data for export
        self.heatmap_data = {
            'array': heatmap_array,
            'wavenumbers': common_wavenumbers,
            'extent': [wn_min, wn_max, len(heatmap_array), 0]
        }
    
    def extract_intensities(self, result, data_type):
        """Extract intensities based on data type"""
        if data_type == "Raw Intensity":
            return np.array(result.get('original_intensities', result.get('intensities', [])))
        elif data_type == "Background Corrected":
            return np.array(result.get('intensities', []))
        elif data_type == "Fitted Peaks":
            # Would need access to fitting model - simplified for now
            return np.array(result.get('fitted_curve', result.get('intensities', [])))
        elif data_type == "Residuals":
            return np.array(result.get('residuals', []))
        else:
            return np.array(result.get('intensities', []))
    
    def update_data_cache(self):
        """Update cached data from data processor"""
        if not self.data_processor:
            return
        
        # Get loaded spectra from data processor
        self.loaded_spectra = []
        try:
            # Try to get all loaded files
            all_files = self.data_processor.get_all_files()
            
            for file_path in all_files:
                # Get spectrum data for each file
                spectrum_data = self.data_processor.get_spectrum_data(file_path)
                if spectrum_data:
                    # Ensure we have the required structure
                    formatted_data = {
                        'file': file_path,
                        'wavenumbers': spectrum_data.get('wavenumbers', []),
                        'intensities': spectrum_data.get('intensities', []),
                        'original_intensities': spectrum_data.get('original_intensities', 
                                                               spectrum_data.get('intensities', [])),
                        'background': spectrum_data.get('background', []),
                        'fitted_curve': spectrum_data.get('fitted_curve', []),
                        'residuals': spectrum_data.get('residuals', [])
                    }
                    self.loaded_spectra.append(formatted_data)
        except (AttributeError, Exception):
            # Fallback: try to get current spectrum data
            try:
                current_spectrum = self.data_processor.get_current_spectrum()
                if current_spectrum:
                    self.loaded_spectra = [current_spectrum]
            except Exception:
                self.loaded_spectra = []
        
        # Get batch results if available
        try:
            if hasattr(self.data_processor, 'results') and self.data_processor.results:
                self.batch_results = self.data_processor.results
            elif hasattr(self.data_processor, 'batch_results'):
                self.batch_results = getattr(self.data_processor, 'batch_results', [])
            else:
                self.batch_results = []
        except Exception:
            self.batch_results = []
    
    # Event handlers
    def on_data_type_changed(self, data_type):
        self.settings['data_type'] = data_type
        self.update_plot()
    
    def on_resolution_changed(self, value):
        self.settings['resolution'] = value
        self.update_plot()
    
    def on_colormap_changed(self, colormap):
        self.settings['colormap'] = colormap
        self.update_plot()
    
    def on_interpolation_changed(self, interpolation):
        self.settings['interpolation'] = interpolation
        self.update_plot()
    
    def on_normalize_toggled(self, state):
        self.settings['auto_normalize'] = state == Qt.Checked
        self.update_plot()
    
    def on_colorbar_toggled(self, state):
        self.settings['show_colorbar'] = state == Qt.Checked
        self.update_plot()
    
    def on_auto_range_toggled(self, state):
        self.settings['auto_range'] = state == Qt.Checked
        self.update_plot()
    
    def on_range_changed(self):
        self.settings['x_min'] = self.x_min_spin.value()
        self.settings['x_max'] = self.x_max_spin.value()
        self.update_plot()
    
    def on_contrast_changed(self, value):
        self.settings['contrast'] = value
        self.update_plot()
    
    def get_plot_data(self):
        """Get current plot data"""
        return {
            'type': 'heatmap',
            'settings': self.settings.copy(),
            'heatmap_data': self.heatmap_data,
            'loaded_spectra': len(self.loaded_spectra),
            'batch_results': len(self.batch_results)
        } 