"""
Waterfall Plot Component
Shows multiple spectra stacked vertically for easy comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QCheckBox, 
                              QComboBox, QLabel, QSpinBox, QDoubleSpinBox, 
                              QGroupBox, QSlider, QPushButton)
from PySide6.QtCore import Qt

from .base_plot import BasePlot


class WaterfallPlot(BasePlot):
    """
    Waterfall plot showing multiple spectra stacked with vertical offset
    """
    
    def __init__(self):
        super().__init__("waterfall", "Waterfall Plot - Multiple Spectra")
        
        # Plot-specific settings
        self.settings.update({
            'data_type': 'Raw Intensity',  # Raw Intensity, Background Corrected, Fitted Peaks, Residuals
            'skip_spectra': 1,  # Show every nth spectrum
            'y_offset': 1000,  # Vertical offset between spectra
            'auto_offset': True,
            'color_scheme': 'Sequential',  # Sequential, Discrete, Rainbow
            'colormap': 'viridis',
            'line_width': 1.0,
            'alpha': 0.8,
            'show_labels': True,
            'auto_normalize': False,
            'auto_range': True,
            'x_min': 200,
            'x_max': 1800,
            'contrast': 1.0,
            'brightness': 0.0,
            'gamma': 1.0,
            'auto_update_on_data_change': False  # OPTIMIZATION: Only update when batch processing completes
        })
        
        # Data cache
        self.batch_results = []
        self.loaded_spectra = []
        
    def create_controls(self):
        """Create waterfall-specific controls"""
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
        
        # Skip spectra
        data_layout.addWidget(QLabel("Show every nth:"))
        self.skip_spin = QSpinBox()
        self.skip_spin.setMinimum(1)
        self.skip_spin.setMaximum(20)
        self.skip_spin.setValue(self.settings['skip_spectra'])
        self.skip_spin.valueChanged.connect(self.on_skip_changed)
        data_layout.addWidget(self.skip_spin)
        
        main_layout.addWidget(data_group)
        
        # Display options group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        
        # Y offset
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Y Offset:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setMinimum(0)
        self.offset_spin.setMaximum(10000)
        self.offset_spin.setValue(self.settings['y_offset'])
        self.offset_spin.valueChanged.connect(self.on_offset_changed)
        offset_layout.addWidget(self.offset_spin)
        
        self.auto_offset_check = QCheckBox("Auto")
        self.auto_offset_check.setChecked(self.settings['auto_offset'])
        self.auto_offset_check.stateChanged.connect(self.on_auto_offset_toggled)
        offset_layout.addWidget(self.auto_offset_check)
        display_layout.addLayout(offset_layout)
        
        # Colormap
        display_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow', 'coolwarm', 'tab10'])
        self.colormap_combo.setCurrentText(self.settings['colormap'])
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        display_layout.addWidget(self.colormap_combo)
        
        # Alpha
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setMinimum(0.1)
        self.alpha_spin.setMaximum(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(self.settings['alpha'])
        self.alpha_spin.valueChanged.connect(self.on_alpha_changed)
        alpha_layout.addWidget(self.alpha_spin)
        display_layout.addLayout(alpha_layout)
        
        # Options
        self.show_labels_check = QCheckBox("Show Labels")
        self.show_labels_check.setChecked(self.settings['show_labels'])
        self.show_labels_check.stateChanged.connect(self.on_labels_toggled)
        display_layout.addWidget(self.show_labels_check)
        
        self.auto_normalize_check = QCheckBox("Auto Normalize")
        self.auto_normalize_check.setChecked(self.settings['auto_normalize'])
        self.auto_normalize_check.stateChanged.connect(self.on_normalize_toggled)
        display_layout.addWidget(self.auto_normalize_check)
        
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
        
        main_layout.addWidget(range_group)
        
        # Update button group
        update_group = QGroupBox("Update")
        update_layout = QVBoxLayout(update_group)
        
        update_btn = QPushButton("Update Waterfall")
        update_btn.clicked.connect(self.update_plot)
        update_btn.setToolTip("Update waterfall plot with current data")
        update_btn.setStyleSheet("""
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
        update_layout.addWidget(update_btn)
        
        main_layout.addWidget(update_group)
        
        return controls_widget
    
    def initialize_plot(self):
        """Initialize the waterfall plot"""
        if not self.figure:
            return
            
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        
        # Setup axes
        self.axes.set_xlabel('Wavenumber (cm⁻¹)')
        self.axes.set_ylabel('Intensity + Offset')
        self.axes.set_title(self.title)
        
        if self.settings['show_grid']:
            self.axes.grid(True, alpha=0.3)
        
        # Apply tight layout (with error handling for matplotlib warnings)
        try:
            self.figure.tight_layout()
        except (UserWarning, ValueError):
            # Some matplotlib configurations may not be compatible with tight_layout
            pass
        self.canvas.draw()
    
    def plot_data_on_axes(self):
        """Plot waterfall data on the axes"""
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
        
        # Filter by skip parameter
        plotted_data = data_source[::self.settings['skip_spectra']]
        
        if not plotted_data:
            self.axes.text(0.5, 0.5, 'No data to display', 
                          transform=self.axes.transAxes, 
                          ha='center', va='center', fontsize=12)
            return
        
        # Determine range
        if self.settings['auto_range']:
            all_wavenumbers = []
            for result in plotted_data:
                wavenumbers = result.get('wavenumbers', [])
                if len(wavenumbers) > 0:
                    all_wavenumbers.extend(wavenumbers)
            
            if all_wavenumbers:
                x_min, x_max = min(all_wavenumbers), max(all_wavenumbers)
            else:
                x_min, x_max = self.settings['x_min'], self.settings['x_max']
        else:
            x_min, x_max = self.settings['x_min'], self.settings['x_max']
        
        # Calculate auto offset if needed
        y_offset = self.settings['y_offset']
        if self.settings['auto_offset']:
            max_intensities = []
            for result in plotted_data:
                wavenumbers = np.array(result.get('wavenumbers', []))
                intensities = self.extract_intensities(result, data_type)
                
                if len(wavenumbers) > 0 and len(intensities) > 0:
                    # Filter to range
                    mask = (wavenumbers >= x_min) & (wavenumbers <= x_max)
                    if np.any(mask):
                        filtered_intensities = intensities[mask]
                        max_intensities.append(np.max(filtered_intensities) - np.min(filtered_intensities))
            
            if max_intensities:
                y_offset = np.mean(max_intensities) * 1.2
        
        # Set up colormap
        colormap = plt.get_cmap(self.settings['colormap'])
        n_spectra = len(plotted_data)
        
        # Plot each spectrum
        for i, result in enumerate(plotted_data):
            wavenumbers = np.array(result.get('wavenumbers', []))
            intensities = self.extract_intensities(result, data_type)
            
            if len(wavenumbers) == 0 or len(intensities) == 0:
                continue
            
            # Apply normalization if requested
            if self.settings['auto_normalize']:
                if np.max(intensities) > np.min(intensities):
                    intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            
            # Apply vertical offset
            intensities_offset = intensities + i * y_offset
            
            # Get color
            color = colormap(i / max(1, n_spectra - 1))
            
            # Plot spectrum
            self.axes.plot(wavenumbers, intensities_offset, 
                          color=color, alpha=self.settings['alpha'], 
                          linewidth=self.settings['line_width'])
            
            # Add label if requested
            if self.settings['show_labels']:
                # Get filename or index
                label = result.get('file', f'Spectrum {i + 1}')
                if isinstance(label, str):
                    label = label.split('/')[-1]  # Just filename
                
                # Position label at the end of the spectrum
                if len(wavenumbers) > 0:
                    label_x = wavenumbers[-1] if wavenumbers[-1] <= x_max else x_max
                    label_y = intensities_offset[-1] if len(intensities_offset) > 0 else i * y_offset
                    
                    self.axes.annotate(label, xy=(label_x, label_y), 
                                     xytext=(5, 0), textcoords='offset points',
                                     fontsize=8, va='center', alpha=0.8)
        
        # Set axis limits
        self.axes.set_xlim(x_min, x_max)
        
        # Y limits based on data
        if plotted_data:
            y_max = (len(plotted_data) - 1) * y_offset
            if self.settings['auto_normalize']:
                y_max += 1
            else:
                # Add some padding for the highest spectrum
                last_result = plotted_data[-1]
                last_wavenumbers = np.array(last_result.get('wavenumbers', []))
                last_intensities = self.extract_intensities(last_result, data_type)
                if len(last_intensities) > 0:
                    y_max += np.max(last_intensities) * 0.1
            
            self.axes.set_ylim(-y_offset * 0.1, y_max + y_offset * 0.1)
    
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
    
    def on_skip_changed(self, value):
        self.settings['skip_spectra'] = value
        self.update_plot()
    
    def on_offset_changed(self, value):
        self.settings['y_offset'] = value
        self.update_plot()
    
    def on_auto_offset_toggled(self, state):
        self.settings['auto_offset'] = state == Qt.Checked
        self.update_plot()
    
    def on_colormap_changed(self, colormap):
        self.settings['colormap'] = colormap
        self.update_plot()
    
    def on_alpha_changed(self, value):
        self.settings['alpha'] = value
        self.update_plot()
    
    def on_normalize_toggled(self, state):
        self.settings['auto_normalize'] = state == Qt.Checked
        self.update_plot()
    
    def on_auto_range_toggled(self, state):
        self.settings['auto_range'] = state == Qt.Checked
        self.update_plot()
    
    def on_range_changed(self):
        self.settings['x_min'] = self.x_min_spin.value()
        self.settings['x_max'] = self.x_max_spin.value()
        self.update_plot()
    
    def on_labels_toggled(self, state):
        self.settings['show_labels'] = state == Qt.Checked
        self.update_plot()
    
    def get_plot_data(self):
        """Get current plot data"""
        return {
            'type': 'waterfall',
            'settings': self.settings.copy(),
            'loaded_spectra': len(self.loaded_spectra),
            'batch_results': len(self.batch_results)
        } 