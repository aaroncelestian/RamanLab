"""
Trends Plot Component
Shows how peak parameters change across multiple spectra in batch processing
"""

import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QComboBox, QLabel
from PySide6.QtCore import Qt

from .base_plot import BasePlot


class TrendsPlot(BasePlot):
    """
    Trends plot showing peak parameter evolution across spectra
    """
    
    def __init__(self):
        super().__init__("trends", "Peak Parameter Trends")
        
        # Plot-specific settings
        self.settings.update({
            'show_position_trends': True,
            'show_intensity_trends': True,
            'show_width_trends': True,
            'show_trend_lines': True,
            'show_error_bars': False,
            'parameter_type': 'position',  # position, intensity, width, area
            'peak_selection': 'all',  # all, selected, or specific peak numbers
            'auto_update_on_data_change': False  # OPTIMIZATION: Only update when batch processing completes
        })
        
        # Data cache
        self.batch_results = []
        self.trend_data = {}
        
    def create_controls(self):
        """Create trends-specific controls"""
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        # Parameter selection
        controls_layout.addWidget(QLabel("Parameter:"))
        self.parameter_combo = QComboBox()
        self.parameter_combo.addItems(['Position', 'Intensity', 'Width', 'Area'])
        self.parameter_combo.currentTextChanged.connect(self.on_parameter_changed)
        controls_layout.addWidget(self.parameter_combo)
        
        # Display options
        self.show_trend_lines_check = QCheckBox("Trend Lines")
        self.show_trend_lines_check.setChecked(self.settings['show_trend_lines'])
        self.show_trend_lines_check.stateChanged.connect(self.on_trend_lines_toggled)
        controls_layout.addWidget(self.show_trend_lines_check)
        
        self.show_error_bars_check = QCheckBox("Error Bars")
        self.show_error_bars_check.setChecked(self.settings['show_error_bars'])
        self.show_error_bars_check.stateChanged.connect(self.on_error_bars_toggled)
        controls_layout.addWidget(self.show_error_bars_check)
        
        # Grid and labels
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(self.settings['show_grid'])
        self.grid_checkbox.stateChanged.connect(self.on_grid_toggled)
        controls_layout.addWidget(self.grid_checkbox)
        
        self.labels_checkbox = QCheckBox("Peak Labels")
        self.labels_checkbox.setChecked(self.settings['show_labels'])
        self.labels_checkbox.stateChanged.connect(self.on_labels_toggled)
        controls_layout.addWidget(self.labels_checkbox)
        
        controls_layout.addStretch()
        
        return controls_widget
    
    def initialize_plot(self):
        """Initialize the trends plot"""
        if not self.figure:
            return
            
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        
        # Setup axes
        self.axes.set_xlabel('Spectrum Index')
        self.axes.set_title(self.title)
        
        # Y-axis label depends on parameter type
        parameter = self.settings['parameter_type']
        if parameter == 'position':
            self.axes.set_ylabel('Peak Position (cm⁻¹)')
        elif parameter == 'intensity':
            self.axes.set_ylabel('Peak Intensity')
        elif parameter == 'width':
            self.axes.set_ylabel('Peak Width (cm⁻¹)')
        elif parameter == 'area':
            self.axes.set_ylabel('Peak Area')
        
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
        """Plot trends data on the axes"""
        if not self.axes:
            return
            
        # Update data cache
        self.update_data_cache()
        
        if not self.trend_data:
            self.axes.text(0.5, 0.5, 'No batch results available', 
                          transform=self.axes.transAxes, 
                          ha='center', va='center', fontsize=12)
            return
        
        parameter = self.settings['parameter_type']
        
        # Plot trends for each peak
        for peak_idx, peak_data in self.trend_data.items():
            if parameter not in peak_data:
                continue
                
            spectrum_indices = peak_data['spectrum_indices']
            values = peak_data[parameter]
            
            if len(values) == 0:
                continue
            
            # Color for this peak
            color = plt.cm.tab10(peak_idx % 10)
            
            # Plot data points
            self.axes.scatter(spectrum_indices, values, 
                            color=color, alpha=0.7, s=30,
                            label=f'Peak {peak_idx + 1}')
            
            # Plot trend line if enabled
            if self.settings['show_trend_lines'] and len(values) > 1:
                # Fit linear trend
                z = np.polyfit(spectrum_indices, values, 1)
                p = np.poly1d(z)
                self.axes.plot(spectrum_indices, p(spectrum_indices), 
                              '--', color=color, alpha=0.8, linewidth=1.5)
            
            # Plot error bars if enabled and available
            if self.settings['show_error_bars'] and 'errors' in peak_data:
                errors = peak_data['errors'].get(parameter, None)
                if errors is not None:
                    self.axes.errorbar(spectrum_indices, values, yerr=errors,
                                     fmt='none', color=color, alpha=0.5)
            
            # Add peak labels if enabled
            if self.settings['show_labels']:
                # Label the last point
                if len(spectrum_indices) > 0:
                    last_idx = len(spectrum_indices) - 1
                    self.axes.annotate(f'P{peak_idx + 1}', 
                                     xy=(spectrum_indices[last_idx], values[last_idx]),
                                     xytext=(5, 5), 
                                     textcoords='offset points',
                                     fontsize=8, color=color,
                                     bbox=dict(boxstyle='round,pad=0.3', 
                                              facecolor='white', 
                                              alpha=0.7))
        
        # Set axis limits
        if self.trend_data:
            all_indices = []
            all_values = []
            
            for peak_data in self.trend_data.values():
                if parameter in peak_data and len(peak_data[parameter]) > 0:
                    all_indices.extend(peak_data['spectrum_indices'])
                    all_values.extend(peak_data[parameter])
            
            if all_indices and all_values:
                self.axes.set_xlim(min(all_indices) - 0.5, max(all_indices) + 0.5)
                
                # Add some padding to y-axis
                y_range = max(all_values) - min(all_values)
                y_padding = y_range * 0.1 if y_range > 0 else 1
                self.axes.set_ylim(min(all_values) - y_padding, 
                                  max(all_values) + y_padding)
        
        # Add legend if multiple peaks
        if len(self.trend_data) > 1:
            self.axes.legend(loc='best', fontsize=8)
    
    def update_data_cache(self):
        """Update cached data from batch results"""
        if not self.data_processor:
            return
            
        # Get batch results from data processor
        try:
            self.batch_results = self.data_processor.get_batch_results()
            if not self.batch_results:
                # Try alternative method if available
                if hasattr(self.data_processor, 'batch_results'):
                    self.batch_results = self.data_processor.batch_results
                else:
                    self.batch_results = []
        except Exception as e:
            print(f"Error getting batch results for trends: {e}")
            self.batch_results = []
        
        # Process batch results into trend data
        self.trend_data = {}
        
        for spectrum_idx, result in enumerate(self.batch_results):
            if not result.get('success', False):
                continue
                
            # Get fit parameters from the result
            fit_params = result.get('fit_params', [])
            peak_count = result.get('peak_count', 0)
            
            if not fit_params or peak_count == 0:
                continue
            
            # Determine parameters per peak based on model
            model = result.get('model', 'Gaussian')
            if model == 'Gaussian' or model == 'Lorentzian':
                params_per_peak = 3  # amplitude, center, width
            elif model == 'Pseudo-Voigt' or model == 'Voigt':
                params_per_peak = 4  # amplitude, center, width, eta/gamma  
            elif model == 'Asymmetric Voigt':
                params_per_peak = 5  # amplitude, center, width, gamma, alpha
            else:
                params_per_peak = 3  # default fallback
            
            for peak_idx in range(peak_count):
                param_start = peak_idx * params_per_peak
                
                if param_start + 2 >= len(fit_params):
                    continue
                
                # Initialize peak data if not exists
                if peak_idx not in self.trend_data:
                    self.trend_data[peak_idx] = {
                        'spectrum_indices': [],
                        'position': [],
                        'intensity': [],
                        'width': [],
                        'area': []
                    }
                
                # Extract peak parameters (Gaussian: amplitude, center, width)
                try:
                    amplitude = float(fit_params[param_start])
                    center = float(fit_params[param_start + 1])
                    width = abs(float(fit_params[param_start + 2]))  # Ensure positive width
                    
                    # Calculate area (for Gaussian: amplitude * width * sqrt(2*pi))
                    area = amplitude * width * np.sqrt(2 * np.pi)
                    
                    # Store data
                    peak_data = self.trend_data[peak_idx]
                    peak_data['spectrum_indices'].append(spectrum_idx)
                    peak_data['position'].append(center)
                    peak_data['intensity'].append(amplitude)
                    peak_data['width'].append(width)
                    peak_data['area'].append(area)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error processing peak {peak_idx} in spectrum {spectrum_idx}: {e}")
                    continue
        
        # Debug output reduced to prevent performance issues during background parameter changes
        if len(self.batch_results) > 0:  # Only print when there are actual results
            print(f"Trends plot: Processed {len(self.batch_results)} results, found {len(self.trend_data)} peaks")
    
    def on_parameter_changed(self, parameter_text):
        """Handle parameter selection change"""
        parameter_map = {
            'Position': 'position',
            'Intensity': 'intensity', 
            'Width': 'width',
            'Area': 'area'
        }
        
        self.settings['parameter_type'] = parameter_map.get(parameter_text, 'position')
        self.initialize_plot()
        self.plot_data_on_axes()
        self.canvas.draw()
    
    def on_trend_lines_toggled(self, state):
        """Handle trend lines toggle"""
        self.settings['show_trend_lines'] = state == Qt.Checked
        self.update_plot()
    
    def on_error_bars_toggled(self, state):
        """Handle error bars toggle"""
        self.settings['show_error_bars'] = state == Qt.Checked
        self.update_plot()
    
    def get_plot_data(self):
        """Get current plot data"""
        return {
            'trend_data': self.trend_data,
            'batch_results': self.batch_results,
            'parameter_type': self.settings['parameter_type']
        } 