"""
Current Spectrum Plot Component
Shows individual spectrum with peaks, fitting curves, background, and residuals
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PySide6.QtWidgets import QWidget, QHBoxLayout, QCheckBox, QComboBox, QLabel, QSpinBox
from PySide6.QtCore import Qt

from .base_plot import BasePlot


class CurrentSpectrumPlot(BasePlot):
    """
    Current spectrum plot showing raw data, fitted curves, peaks, and residuals
    """
    
    def __init__(self):
        super().__init__("current_spectrum", "Current Spectrum")
        
        # Plot-specific settings
        self.settings.update({
            'show_raw_spectrum': True,
            'show_fitted_curve': True,
            'show_individual_peaks': True,
            'show_background': True,
            'show_residuals': True,
            'show_peak_positions': True,
            'show_peak_boundaries': False,
            'residuals_scale': 1.0,
            'peak_alpha': 0.6,
            'alpha': 1.0  # Transparency for raw spectrum line
        })
        
        # Plot elements
        self.main_axes = None
        self.residuals_axes = None
        
        # Cached data
        self.current_wavenumbers = None
        self.current_intensities = None
        self.fitted_curve = None
        self.individual_peaks = []
        self.peak_positions = []
        self.background = None
        self.residuals = None
        
    def create_controls(self):
        """Create spectrum-specific controls"""
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        # Data display options
        self.show_raw_check = QCheckBox("Raw Spectrum")
        self.show_raw_check.setChecked(self.settings['show_raw_spectrum'])
        self.show_raw_check.stateChanged.connect(self.on_show_raw_toggled)
        controls_layout.addWidget(self.show_raw_check)
        
        self.show_fitted_check = QCheckBox("Fitted Curve")
        self.show_fitted_check.setChecked(self.settings['show_fitted_curve'])
        self.show_fitted_check.stateChanged.connect(self.on_show_fitted_toggled)
        controls_layout.addWidget(self.show_fitted_check)
        
        self.show_peaks_check = QCheckBox("Individual Peaks")
        self.show_peaks_check.setChecked(self.settings['show_individual_peaks'])
        self.show_peaks_check.stateChanged.connect(self.on_show_peaks_toggled)
        controls_layout.addWidget(self.show_peaks_check)
        
        self.show_background_check = QCheckBox("Background")
        self.show_background_check.setChecked(self.settings['show_background'])
        self.show_background_check.stateChanged.connect(self.on_show_background_toggled)
        controls_layout.addWidget(self.show_background_check)
        
        self.show_residuals_check = QCheckBox("Residuals")
        self.show_residuals_check.setChecked(self.settings['show_residuals'])
        self.show_residuals_check.stateChanged.connect(self.on_show_residuals_toggled)
        controls_layout.addWidget(self.show_residuals_check)
        
        # Grid and labels from parent
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
        """Initialize the spectrum plot with main and residuals axes"""
        if not self.figure:
            return
            
        self.figure.clear()
        
        # Get dynamic title with filename
        plot_title = self.get_dynamic_title()
        
        # Create subplot layout with residuals
        if self.settings['show_residuals']:
            # Create subplot with shared x-axis
            self.main_axes = self.figure.add_subplot(211)
            self.residuals_axes = self.figure.add_subplot(212, sharex=self.main_axes)
            
            # Setup main plot
            self.main_axes.set_ylabel('Intensity')
            self.main_axes.set_title(plot_title)
            
            # Setup residuals plot  
            self.residuals_axes.set_xlabel('Wavenumber (cm⁻¹)')
            self.residuals_axes.set_ylabel('Residuals')
            self.residuals_axes.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
        else:
            # Single plot
            self.main_axes = self.figure.add_subplot(111)
            self.main_axes.set_xlabel('Wavenumber (cm⁻¹)')
            self.main_axes.set_ylabel('Intensity')
            self.main_axes.set_title(plot_title)
            self.residuals_axes = None
        
        if self.settings['show_grid']:
            self.main_axes.grid(True, alpha=0.3)
            if self.residuals_axes:
                self.residuals_axes.grid(True, alpha=0.3)
        
        # Apply tight layout (with error handling for matplotlib warnings)
        try:
            self.figure.tight_layout()
        except (UserWarning, ValueError):
            # Some matplotlib configurations may not be compatible with tight_layout
            pass
        self.canvas.draw()
    
    def plot_data_on_axes(self):
        """Plot spectrum data on the axes"""
        if not self.main_axes:
            return
            
        # Get current spectrum data
        self.update_data_cache()
        
        if (self.current_wavenumbers is None or self.current_intensities is None or 
            len(self.current_wavenumbers) == 0 or len(self.current_intensities) == 0):
            self.main_axes.text(0.5, 0.5, 'No spectrum loaded', 
                               transform=self.main_axes.transAxes, 
                               ha='center', va='center', fontsize=12)
            return
        
        # Plot raw spectrum
        if self.settings['show_raw_spectrum']:
            self.main_axes.plot(self.current_wavenumbers, self.current_intensities, 
                               'b-', linewidth=self.settings['line_width'], 
                               alpha=self.settings['alpha'], label='Raw Spectrum')
        
        # Plot background
        if self.settings['show_background'] and self.background is not None:
            self.main_axes.plot(self.current_wavenumbers, self.background, 
                               'g--', linewidth=1.0, alpha=0.7, label='Background')
        
        # Plot fitted curve
        if self.settings['show_fitted_curve'] and self.fitted_curve is not None:
            self.main_axes.plot(self.current_wavenumbers, self.fitted_curve, 
                               'r-', linewidth=self.settings['line_width'] + 0.5, 
                               alpha=0.8, label='Fitted Curve')
        
        # Plot individual peaks
        if (self.settings['show_individual_peaks'] and self.individual_peaks is not None and 
            len(self.individual_peaks) > 0):
            for i, peak_curve in enumerate(self.individual_peaks):
                color = plt.cm.tab10(i % 10)
                self.main_axes.plot(self.current_wavenumbers, peak_curve, 
                                   '--', color=color, linewidth=1.0, 
                                   alpha=self.settings['peak_alpha'], 
                                   label=f'Peak {i+1}')
        
        # Plot peak positions (automatic + manual)
        spectrum_data = self.data_processor.get_current_spectrum() if self.data_processor else None
        if spectrum_data and self.settings['show_peak_positions']:
            # Get all peaks (automatic + manual)
            auto_peaks = spectrum_data.get('peaks', [])
            manual_peaks = spectrum_data.get('manual_peaks', [])
            
            # Plot automatic peaks - ensure proper array handling
            if auto_peaks is not None and len(auto_peaks) > 0:
                for i, peak_idx in enumerate(auto_peaks):
                    if peak_idx < len(self.current_wavenumbers):
                        x_pos = self.current_wavenumbers[peak_idx]
                        y_pos = self.current_intensities[peak_idx]
                        
                        # Vertical line
                        self.main_axes.axvline(x=x_pos, color='red', linestyle=':', alpha=0.7)
                        
                        # Peak label
                        if self.settings['show_labels']:
                            self.main_axes.annotate(f'A{i+1}', 
                                                   xy=(x_pos, y_pos), 
                                                   xytext=(5, 5), 
                                                   textcoords='offset points',
                                                   fontsize=8, color='red',
                                                   bbox=dict(boxstyle='round,pad=0.3', 
                                                            facecolor='white', 
                                                            alpha=0.7))
            
            # Plot manual peaks - ensure proper array handling
            if manual_peaks is not None and len(manual_peaks) > 0:
                for i, peak_idx in enumerate(manual_peaks):
                    if peak_idx < len(self.current_wavenumbers):
                        x_pos = self.current_wavenumbers[peak_idx]
                        y_pos = self.current_intensities[peak_idx]
                        
                        # Vertical line (different style for manual)
                        self.main_axes.axvline(x=x_pos, color='blue', linestyle='--', alpha=0.8, linewidth=2)
                        
                        # Peak label
                        if self.settings['show_labels']:
                            self.main_axes.annotate(f'M{i+1}', 
                                                   xy=(x_pos, y_pos), 
                                                   xytext=(5, -15), 
                                                   textcoords='offset points',
                                                   fontsize=8, color='blue',
                                                   bbox=dict(boxstyle='round,pad=0.3', 
                                                            facecolor='lightblue', 
                                                            alpha=0.7))
        
        # Set axis limits
        if len(self.current_wavenumbers) > 0:
            self.main_axes.set_xlim(self.current_wavenumbers[0], self.current_wavenumbers[-1])
            # Also set y limits based on data
            if len(self.current_intensities) > 0:
                y_min, y_max = np.min(self.current_intensities), np.max(self.current_intensities)
                y_margin = (y_max - y_min) * 0.1
                self.main_axes.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Add legend if multiple elements are shown
        plotted_elements = sum([
            self.settings['show_raw_spectrum'],
            self.settings['show_fitted_curve'] and self.fitted_curve is not None,
            self.settings['show_background'] and self.background is not None,
            self.settings['show_individual_peaks'] and self.individual_peaks is not None and len(self.individual_peaks) > 0
        ])
        
        if plotted_elements > 1:
            self.main_axes.legend(loc='upper right', fontsize=8)
        
        # Plot residuals
        if (self.residuals_axes and self.settings['show_residuals'] and 
            self.residuals is not None and len(self.residuals) > 0):
            self.residuals_axes.plot(self.current_wavenumbers, self.residuals, 
                                    'k-', linewidth=1.0, alpha=0.7)
            
            # Set residuals y-axis limits safely
            try:
                abs_residuals = np.abs(self.residuals)
                if len(abs_residuals) > 0:
                    max_abs_residual = np.max(abs_residuals)
                    if max_abs_residual > 0:
                        self.residuals_axes.set_ylim(-max_abs_residual * 1.1, max_abs_residual * 1.1)
            except Exception as e:
                print(f"Warning: Could not set residuals y-limits: {e}")
        
        # Force canvas to redraw
        self.canvas.draw_idle()
    
    def get_dynamic_title(self):
        """Get dynamic title including current filename"""
        base_title = "Current Spectrum"
        
        if not self.data_processor:
            return base_title
        
        try:
            # Get current filename
            current_filename = self.data_processor.get_current_file_name()
            if current_filename and current_filename != "None":
                # Remove file extension for cleaner display
                filename_without_ext = current_filename.rsplit('.', 1)[0] if '.' in current_filename else current_filename
                return f"{base_title} - {filename_without_ext}"
            else:
                return base_title
        except Exception as e:
            print(f"Warning: Could not get filename for title: {e}")
            return base_title
    
    def update_data_cache(self):
        """Update cached data from core components"""
        if not self.data_processor:
            return
            
        # Get current spectrum
        spectrum_data = None
        try:
            spectrum_data = self.data_processor.get_current_spectrum()
            if spectrum_data:
                # Safely extract wavenumbers and intensities
                wavenumbers = spectrum_data.get('wavenumbers')
                intensities = spectrum_data.get('intensities')
                
                # Ensure arrays are properly handled
                if wavenumbers is not None and hasattr(wavenumbers, '__len__'):
                    self.current_wavenumbers = np.asarray(wavenumbers) if len(wavenumbers) > 0 else None
                else:
                    self.current_wavenumbers = None
                    
                if intensities is not None and hasattr(intensities, '__len__'):
                    self.current_intensities = np.asarray(intensities) if len(intensities) > 0 else None
                else:
                    self.current_intensities = None
            else:
                self.current_wavenumbers = None
                self.current_intensities = None
        except Exception as e:
            print(f"Error getting spectrum data: {e}")
            self.current_wavenumbers = None
            self.current_intensities = None
        
        # Get fitting results from data processor (more reliable)
        if spectrum_data:
            fit_result = spectrum_data.get('fit_result')
            if fit_result and isinstance(fit_result, dict):
                # Safely extract fitting data
                fitted_curve = fit_result.get('fitted_curve')
                individual_peaks = fit_result.get('individual_peaks', [])
                residuals = fit_result.get('residuals')
                
                self.fitted_curve = np.asarray(fitted_curve) if fitted_curve is not None else None
                self.individual_peaks = individual_peaks if individual_peaks else []
                self.residuals = np.asarray(residuals) if residuals is not None else None
            else:
                # Clear fitting data if no valid fit result
                self.fitted_curve = None
                self.individual_peaks = []
                self.residuals = None
        else:
            # Clear all cached data if no spectrum data
            self.fitted_curve = None
            self.individual_peaks = []
            self.residuals = None
        
        # Get background from data processor
        if spectrum_data:
            background = spectrum_data.get('background')
            self.background = np.asarray(background) if background is not None else None
        else:
            self.background = None
    
    def on_show_raw_toggled(self, state):
        """Handle raw spectrum toggle"""
        self.settings['show_raw_spectrum'] = state == Qt.Checked
        self.update_plot()
    
    def on_show_fitted_toggled(self, state):
        """Handle fitted curve toggle"""
        self.settings['show_fitted_curve'] = state == Qt.Checked
        self.update_plot()
    
    def on_show_peaks_toggled(self, state):
        """Handle individual peaks toggle"""
        self.settings['show_individual_peaks'] = state == Qt.Checked
        self.update_plot()
    
    def on_show_background_toggled(self, state):
        """Handle background toggle"""
        self.settings['show_background'] = state == Qt.Checked
        self.update_plot()
    
    def on_show_residuals_toggled(self, state):
        """Handle residuals toggle"""
        self.settings['show_residuals'] = state == Qt.Checked
        # Need to reinitialize plot structure
        self.initialize_plot()
        self.plot_data_on_axes()
        self.canvas.draw()
    
    def update_plot(self):
        """Update the plot with current data (override from BasePlot)"""
        if not self.is_initialized:
            return
            
        # Clear and redraw using our specific axes structure
        if self.main_axes:
            self.main_axes.clear()
        if self.residuals_axes:
            self.residuals_axes.clear()
                
        self.initialize_plot()
        self.plot_data_on_axes()
        
        # Update title in case filename changed
        if self.main_axes:
            self.main_axes.set_title(self.get_dynamic_title())
        
        self.canvas.draw()
            
        self.plot_updated.emit(self.plot_type)
    
    def get_plot_data(self):
        """Get current plot data"""
        return {
            'wavenumbers': self.current_wavenumbers,
            'intensities': self.current_intensities,
            'fitted_curve': self.fitted_curve,
            'individual_peaks': self.individual_peaks,
            'peak_positions': self.peak_positions,
            'background': self.background,
            'residuals': self.residuals
        } 