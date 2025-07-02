"""
Base Plot Component
Provides common functionality for all plot types in the visualization system
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QComboBox, QLabel
from PySide6.QtCore import QObject, Signal, Qt


class BasePlot(QObject):
    """
    Base class for all plot components providing common functionality
    """
    
    # Signals for plot interactions
    plot_updated = Signal(str)  # Emits plot type when updated
    plot_clicked = Signal(str, float, float)  # plot_type, x, y coordinates
    plot_settings_changed = Signal(str, dict)  # plot_type, settings
    
    def __init__(self, plot_type, title="Plot"):
        super().__init__()
        
        self.plot_type = plot_type
        self.title = title
        
        # Core components (set via dependency injection)
        self.data_processor = None
        self.peak_fitter = None
        self.ui_manager = None
        
        # Plot state
        self.is_initialized = False
        self.widget = None
        self.figure = None
        self.axes = None
        self.canvas = None
        
        # Data storage
        self.plot_data = {}
        
        # Plot configuration
        self.settings = {
            'show_grid': True,
            'show_labels': False,
            'interactive': True,
            'auto_update_on_data_change': True,  # NEW: Allow plots to opt out of auto-updates
            'colormap': 'viridis',
            'line_width': 1.5,
            'marker_size': 30
        }
        
    def set_core_components(self, data_processor, peak_fitter, ui_manager):
        """Inject core components"""
        self.data_processor = data_processor
        self.peak_fitter = peak_fitter
        self.ui_manager = ui_manager
        
        # Connect to core signals if needed
        self.connect_core_signals()
        
    def connect_core_signals(self):
        """Connect to core component signals - override in subclasses"""
        if self.data_processor:
            self.data_processor.spectrum_loaded.connect(self.on_data_changed)
            self.data_processor.current_spectrum_changed.connect(self.on_spectrum_changed)
            
        if self.peak_fitter:
            self.peak_fitter.peaks_fitted.connect(self.on_peaks_fitted)
            self.peak_fitter.background_calculated.connect(self.on_background_changed)
    
    def create_widget(self):
        """Create the plot widget with matplotlib canvas"""
        if self.widget is not None:
            return self.widget
            
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Add canvas to layout
        layout.addWidget(self.canvas)
        
        # Create controls if needed
        controls = self.create_controls()
        if controls:
            layout.addWidget(controls)
        
        # Connect canvas events
        self.setup_canvas_events()
        
        # Initial plot
        self.initialize_plot()
        
        self.is_initialized = True
        return self.widget
    
    def create_controls(self):
        """Create plot-specific controls - override in subclasses"""
        # Basic controls that most plots can use
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        # Grid toggle
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.settings['show_grid'])
        self.grid_checkbox.stateChanged.connect(self.on_grid_toggled)
        controls_layout.addWidget(self.grid_checkbox)
        
        # Labels toggle
        self.labels_checkbox = QCheckBox("Show Labels")
        self.labels_checkbox.setChecked(self.settings['show_labels'])
        self.labels_checkbox.stateChanged.connect(self.on_labels_toggled)
        controls_layout.addWidget(self.labels_checkbox)
        
        # Update button
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.update_plot)
        update_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)
        controls_layout.addWidget(update_btn)
        
        controls_layout.addStretch()
        
        return controls_widget
    
    def setup_canvas_events(self):
        """Setup canvas event handlers"""
        if self.canvas and self.settings['interactive']:
            self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
    
    def initialize_plot(self):
        """Initialize the plot - override in subclasses"""
        if not self.figure:
            return
            
        self.axes = self.figure.add_subplot(111)
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
    
    def update_plot(self):
        """Update the plot with current data - override in subclasses"""
        if not self.is_initialized:
            return
            
        # Clear and redraw
        if self.axes:
            self.axes.clear()
            self.initialize_plot()
            self.plot_data_on_axes()
            self.canvas.draw()
            
        self.plot_updated.emit(self.plot_type)
    
    def plot_data_on_axes(self):
        """Plot data on axes - override in subclasses"""
        # Default implementation - just show a placeholder
        if self.axes:
            self.axes.text(0.5, 0.5, f'{self.title}\nNo data to display', 
                          transform=self.axes.transAxes, 
                          ha='center', va='center', fontsize=12)
    
    def clear_plot(self):
        """Clear the plot"""
        if self.axes:
            self.axes.clear()
            self.initialize_plot()
            self.canvas.draw()
    
    def export_plot(self, file_path, dpi=300):
        """Export plot to file"""
        if self.figure:
            self.figure.savefig(file_path, dpi=dpi, bbox_inches='tight')
    
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if event.inaxes and event.xdata is not None and event.ydata is not None:
            self.plot_clicked.emit(self.plot_type, float(event.xdata), float(event.ydata))
    
    def on_grid_toggled(self, state):
        """Handle grid toggle"""
        self.settings['show_grid'] = state == Qt.Checked
        if self.axes:
            self.axes.grid(self.settings['show_grid'], alpha=0.3)
            self.canvas.draw()
        self.plot_settings_changed.emit(self.plot_type, {'show_grid': self.settings['show_grid']})
    
    def on_labels_toggled(self, state):
        """Handle labels toggle"""
        self.settings['show_labels'] = state == Qt.Checked
        self.update_plot()
        self.plot_settings_changed.emit(self.plot_type, {'show_labels': self.settings['show_labels']})
    
    def update_settings(self, new_settings):
        """Update plot settings"""
        self.settings.update(new_settings)
        self.update_plot()
    
    def get_settings(self):
        """Get current plot settings"""
        return self.settings.copy()
    
    def get_plot_data(self):
        """Get current plot data - override in subclasses"""
        return self.plot_data.copy()
    
    # Event handlers for core component signals
    def on_data_changed(self, data):
        """Handle data processor changes"""
        if self.settings['auto_update_on_data_change']:
            self.update_plot()
    
    def on_spectrum_changed(self, index):
        """Handle spectrum change"""
        if self.settings['auto_update_on_data_change']:
            self.update_plot()
    
    def on_peaks_fitted(self, results):
        """Handle peak fitting results"""
        if self.settings['auto_update_on_data_change']:
            self.update_plot()
    
    def on_background_changed(self, background):
        """Handle background calculation"""
        if self.settings['auto_update_on_data_change']:
            self.update_plot()
    
    def get_status(self):
        """Get plot status"""
        return {
            'plot_type': self.plot_type,
            'initialized': self.is_initialized,
            'has_data': bool(self.plot_data),
            'settings': self.settings
        } 