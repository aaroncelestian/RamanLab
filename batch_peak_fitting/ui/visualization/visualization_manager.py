"""
Visualization Manager
Coordinates all plot components and provides unified interface for visualization system
"""

from PySide6.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSplitter
from PySide6.QtCore import QObject, Signal, Qt

from .current_spectrum_plot import CurrentSpectrumPlot
from .trends_plot import TrendsPlot
from .waterfall_plot import WaterfallPlot
from .heatmap_plot import HeatmapPlot

# Additional components can be added here as needed
STATS_AVAILABLE = True
SPECIALIZED_AVAILABLE = True


class VisualizationManager(QObject):
    """
    Central coordinator for all visualization components
    Manages plot creation, updates, and interactions
    """
    
    # Signals for visualization events
    plot_updated = Signal(str)  # plot_type
    plot_clicked = Signal(str, float, float)  # plot_type, x, y
    visualization_ready = Signal()
    
    def __init__(self):
        super().__init__()
        
        # Core components (injected)
        self.data_processor = None
        self.peak_fitter = None
        self.ui_manager = None
        
        # Visualization components
        self.plot_components = {}
        self.main_widget = None
        self.plot_tabs = None
        
        # State
        self.is_initialized = False
        self.current_plot_type = "current_spectrum"
        
    def initialize(self, data_processor, peak_fitter, ui_manager):
        """Initialize the visualization manager with core components"""
        self.data_processor = data_processor
        self.peak_fitter = peak_fitter
        self.ui_manager = ui_manager
        
        # Create main widget
        self.create_main_widget()
        
        # Create plot components
        self.create_plot_components()
        
        # Setup initial plots
        self.setup_initial_plots()
        
        self.is_initialized = True
        self.visualization_ready.emit()
        
        print("VisualizationManager: Initialized with plot components")
    
    def create_main_widget(self):
        """Create the main visualization widget"""
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)
        
        # Create tabbed interface for plots
        self.plot_tabs = QTabWidget()
        main_layout.addWidget(self.plot_tabs)
        
        # Add control buttons at bottom
        controls_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh All")
        refresh_btn.clicked.connect(self.refresh_all_plots)
        refresh_btn.setToolTip("Force refresh all plots")
        refresh_btn.setStyleSheet("""
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
        controls_layout.addWidget(refresh_btn)
        
        export_btn = QPushButton("Export Current")
        export_btn.clicked.connect(self.export_current_plot)
        export_btn.setStyleSheet("""
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
        controls_layout.addWidget(export_btn)
        
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout)
    
    def create_plot_components(self):
        """Create individual plot components"""
        
        # Current Spectrum Plot
        current_spectrum_plot = CurrentSpectrumPlot()
        current_spectrum_plot.set_core_components(
            self.data_processor, self.peak_fitter, self.ui_manager
        )
        current_spectrum_plot.plot_updated.connect(self.on_plot_updated)
        current_spectrum_plot.plot_clicked.connect(self.on_plot_clicked)
        
        self.plot_components["current_spectrum"] = current_spectrum_plot
        
        # Trends Plot
        trends_plot = TrendsPlot()
        trends_plot.set_core_components(
            self.data_processor, self.peak_fitter, self.ui_manager
        )
        trends_plot.plot_updated.connect(self.on_plot_updated)
        trends_plot.plot_clicked.connect(self.on_plot_clicked)
        
        self.plot_components["trends"] = trends_plot
        
        # Waterfall Plot
        waterfall_plot = WaterfallPlot()
        waterfall_plot.set_core_components(
            self.data_processor, self.peak_fitter, self.ui_manager
        )
        waterfall_plot.plot_updated.connect(self.on_plot_updated)
        waterfall_plot.plot_clicked.connect(self.on_plot_clicked)
        
        self.plot_components["waterfall"] = waterfall_plot
        
        # Heatmap Plot
        heatmap_plot = HeatmapPlot()
        heatmap_plot.set_core_components(
            self.data_processor, self.peak_fitter, self.ui_manager
        )
        heatmap_plot.plot_updated.connect(self.on_plot_updated)
        heatmap_plot.plot_clicked.connect(self.on_plot_clicked)
        
        self.plot_components["heatmap"] = heatmap_plot
        
        print(f"VisualizationManager: Created {len(self.plot_components)} plot components")
    
    def setup_initial_plots(self):
        """Setup initial plot displays in tabs"""
        if not self.plot_tabs:
            return
        
        # Define the desired tab order and names
        tab_order = [
            ("current_spectrum", "Spectrum"),
            ("trends", "Trends"),
            ("stats", "Stats"),
            ("waterfall", "Waterfall"),
            ("heatmap", "Heatmap"),
            ("specialized", "Specialized")
        ]
        
        # Add tabs in the specified order
        for plot_type, tab_title in tab_order:
            if plot_type in self.plot_components:
                # Use existing plot component
                plot_widget = self.plot_components[plot_type].create_widget()
            elif plot_type == "stats":
                # Create Stats placeholder (can be enhanced later)
                plot_widget = self._create_stats_placeholder()
            elif plot_type == "specialized":
                # Create Specialized placeholder (can be enhanced later)
                plot_widget = self._create_specialized_placeholder()
            else:
                continue
                
            self.plot_tabs.addTab(plot_widget, tab_title)
        
        # Connect tab change signal
        self.plot_tabs.currentChanged.connect(self.on_tab_changed)
        
        print(f"VisualizationManager: Setup {self.plot_tabs.count()} plot tabs")
    
    def _create_stats_placeholder(self):
        """Create a placeholder widget for Stats tab (fitting quality)"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header = QLabel("📊 Fitting Quality Statistics")
        header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #1976D2;
                padding: 10px;
                background-color: #F5F5F5;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)
        
        # Placeholder content
        content = QLabel(
            "📈 Fitting quality analysis will be displayed here\n\n"
            "This will include:\n"
            "• R² values for all fits\n"
            "• Peak fitting accuracy metrics\n"
            "• Residual analysis\n"
            "• Best/worst fitting results comparison\n\n"
            "💡 Load spectra and run batch processing to see results"
        )
        content.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 20px;
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
            }
        """)
        layout.addWidget(content)
        layout.addStretch()
        
        return widget
    
    def _create_specialized_placeholder(self):
        """Create a placeholder widget for Specialized tab"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header = QLabel("🔬 Specialized Analysis Tools")
        header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #1976D2;
                padding: 10px;
                background-color: #F5F5F5;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)
        
        # Tools group
        tools_group = QGroupBox("Available Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        # Density Analysis
        density_btn = QPushButton("🧬 Launch Density Analysis")
        density_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
        """)
        tools_layout.addWidget(density_btn)
        
        # Geothermometry Analysis  
        geothermo_btn = QPushButton("🌡️ Launch Geothermometry Analysis")
        geothermo_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
        """)
        tools_layout.addWidget(geothermo_btn)
        
        # Future tools placeholder
        future_label = QLabel("• Additional specialized tools will be added here")
        future_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        tools_layout.addWidget(future_label)
        
        layout.addWidget(tools_group)
        layout.addStretch()
        
        return widget
    
    def get_main_widget(self):
        """Get the main visualization widget"""
        return self.main_widget
    
    def get_plot_component(self, plot_type):
        """Get a specific plot component"""
        return self.plot_components.get(plot_type)
    
    def update_current_spectrum_only(self):
        """Update only the current spectrum plot (optimized for background parameter changes)"""
        if self.plot_components.get("current_spectrum"):
            self.plot_components["current_spectrum"].update_plot()
    
    def update_batch_dependent_plots(self):
        """Update only batch-dependent plots (trends, waterfall, heatmap) - use after batch processing"""
        batch_plots = ["trends", "waterfall", "heatmap"]
        for plot_type in batch_plots:
            plot_component = self.plot_components.get(plot_type)
            if plot_component and plot_component.is_initialized:
                plot_component.update_plot()
    
    def update_plot(self, plot_type):
        """Update a specific plot"""
        plot_component = self.plot_components.get(plot_type)
        if plot_component:
            plot_component.update_plot()
    
    def update_all_plots(self):
        """Update all plots"""
        for plot_component in self.plot_components.values():
            if plot_component.is_initialized:
                plot_component.update_plot()
    
    def refresh_all_plots(self):
        """Refresh all plots (public interface)"""
        self.update_all_plots()
        print("VisualizationManager: All plots refreshed")
    
    def force_update_all_plots(self):
        """Force update all plots"""
        for plot_type, plot_component in self.plot_components.items():
            try:
                plot_component.update_plot()
            except Exception as e:
                print(f"VisualizationManager: Error updating {plot_type}: {e}")
    
    def export_current_plot(self):
        """Export the currently visible plot"""
        current_index = self.plot_tabs.currentIndex()
        if current_index >= 0:
            plot_types = list(self.plot_components.keys())
            if current_index < len(plot_types):
                plot_type = plot_types[current_index]
                plot_component = self.plot_components[plot_type]
                
                # For now, just print - would open file dialog in real implementation
                print(f"VisualizationManager: Exporting {plot_type} plot")
                # plot_component.export_plot(file_path)
    
    def clear_all_plots(self):
        """Clear all plots"""
        for plot_component in self.plot_components.values():
            if plot_component.is_initialized:
                plot_component.clear_plot()
    
    def set_plot_settings(self, plot_type, settings):
        """Update settings for a specific plot"""
        plot_component = self.plot_components.get(plot_type)
        if plot_component:
            plot_component.update_settings(settings)
    
    def get_plot_settings(self, plot_type):
        """Get settings for a specific plot"""
        plot_component = self.plot_components.get(plot_type)
        if plot_component:
            return plot_component.get_settings()
        return {}
    
    def get_plot_data(self, plot_type):
        """Get data from a specific plot"""
        plot_component = self.plot_components.get(plot_type)
        if plot_component:
            return plot_component.get_plot_data()
        return {}
    
    def get_all_plot_data(self):
        """Get data from all plots"""
        all_data = {}
        for plot_type, plot_component in self.plot_components.items():
            all_data[plot_type] = plot_component.get_plot_data()
        return all_data
    
    # Event handlers
    def on_plot_updated(self, plot_type):
        """Handle plot update event"""
        self.plot_updated.emit(plot_type)
    
    def on_plot_clicked(self, plot_type, x, y):
        """Handle plot click event"""
        self.plot_clicked.emit(plot_type, x, y)
        
        # Handle specific plot interactions
        if plot_type == "current_spectrum":
            self.handle_spectrum_click(x, y)
    
    def on_tab_changed(self, index):
        """Handle tab change"""
        plot_types = list(self.plot_components.keys())
        if 0 <= index < len(plot_types):
            self.current_plot_type = plot_types[index]
            
            # Update the current plot when tab is selected
            self.update_plot(self.current_plot_type)
    
    def handle_spectrum_click(self, x, y):
        """Handle click on spectrum plot"""
        # Could be used to add manual peaks, zoom, etc.
        print(f"VisualizationManager: Spectrum clicked at ({x:.2f}, {y:.2f})")
    
    # Core component event handlers
    def on_spectrum_loaded(self, spectrum_data):
        """Handle spectrum loading"""
        self.update_plot("current_spectrum")
    
    def on_peaks_fitted(self, results):
        """Handle peak fitting completion"""
        self.update_plot("current_spectrum")
        self.update_plot("trends")
    
    def on_batch_completed(self, results):
        """Handle batch processing completion"""
        self.update_all_plots()
    
    def get_status(self):
        """Get visualization manager status"""
        plot_status = {}
        for plot_type, plot_component in self.plot_components.items():
            plot_status[plot_type] = plot_component.get_status()
        
        return {
            'initialized': self.is_initialized,
            'plot_count': len(self.plot_components),
            'current_plot': self.current_plot_type,
            'plots': plot_status
        }
    
    def connect_core_signals(self):
        """Connect to core component signals"""
        if self.data_processor:
            self.data_processor.spectrum_loaded.connect(self.on_spectrum_loaded)
        
        if self.peak_fitter:
            if hasattr(self.peak_fitter, 'peaks_fitted'):
                self.peak_fitter.peaks_fitted.connect(self.on_peaks_fitted)
            if hasattr(self.peak_fitter, 'fitting_completed'):
                self.peak_fitter.fitting_completed.connect(self.on_peaks_fitted) 