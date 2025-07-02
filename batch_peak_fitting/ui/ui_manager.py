"""
UI Manager for Batch Peak Fitting
Coordinates all UI components and manages the overall interface
Central coordinator between tabs and core components
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QSplitter, QFrame, QLabel, QStatusBar
)
from PySide6.QtCore import QObject, Signal, Qt

from .tabs.file_tab import FileTab
from .tabs.peaks_tab import PeaksTab
from .tabs.batch_tab import BatchTab
from .tabs.results_tab import ResultsTab
from .tabs.session_tab import SessionTab
from .visualization.visualization_manager import VisualizationManager


class UIManager(QObject):
    """
    Central UI coordinator for the batch peak fitting interface.
    Manages all tab components and coordinates communication between them
    and the core components (DataProcessor, PeakFitter, MainController).
    """
    
    # Signals for communicating with main controller
    action_requested = Signal(str, dict)  # (action_name, parameters)
    status_updated = Signal(str)  # Status message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Core component references (set by main controller)
        self.data_processor = None
        self.peak_fitter = None
        self.main_controller = None
        
        # UI components
        self.main_widget = None
        self.left_tabs = None  # Control tabs
        self.status_bar = None
        
        # Tab instances
        self.file_tab = None
        self.peaks_tab = None
        self.batch_tab = None
        self.results_tab = None
        self.session_tab = None
        
        # Visualization component
        self.visualization_manager = None
        
        # Track initialization state
        self.is_initialized = False
    
    def initialize(self, data_processor, peak_fitter, main_controller):
        """
        Initialize the UI manager with core components.
        Must be called before setup_ui().
        """
        self.data_processor = data_processor
        self.peak_fitter = peak_fitter
        self.main_controller = main_controller
        
        print("UIManager: Core components set")  # Debug
    
    def setup_ui(self, parent_widget):
        """
        Create and setup the complete UI structure.
        Returns the main widget that should be added to the parent.
        """
        if not self.data_processor or not self.peak_fitter or not self.main_controller:
            raise RuntimeError("UIManager must be initialized with core components before setup_ui()")
        
        # Create main layout structure
        self.main_widget = QWidget(parent_widget)
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Create left panel (control tabs)
        self.left_tabs = QTabWidget()
        self.left_tabs.setMinimumWidth(350)
        self.left_tabs.setMaximumWidth(450)
        
        # Create and add tab components
        self._create_tab_components()
        self._setup_tab_connections()
        
        # Add tabs to left panel
        self.left_tabs.addTab(self.file_tab, "File")
        self.left_tabs.addTab(self.peaks_tab, "Peaks") 
        self.left_tabs.addTab(self.batch_tab, "Batch")
        self.left_tabs.addTab(self.results_tab, "Results")
        self.left_tabs.addTab(self.session_tab, "Session")
        
        # Create right panel placeholder (for visualization components)
        right_panel = self._create_right_panel()
        
        # Add panels to splitter
        main_splitter.addWidget(self.left_tabs)
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([300, 700])
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("UI Manager initialized - Ready")
        main_layout.addWidget(self.status_bar)
        
        # Connect status updates
        self._connect_status_signals()
        
        self.is_initialized = True
        print("UIManager: UI setup complete")  # Debug
        
        return self.main_widget
    
    def _create_tab_components(self):
        """Create all tab component instances"""
        print("UIManager: Creating tab components...")  # Debug
        
        # Create tab instances
        self.file_tab = FileTab()
        self.peaks_tab = PeaksTab()
        self.batch_tab = BatchTab()
        self.results_tab = ResultsTab()
        self.session_tab = SessionTab()
        
        # Create visualization manager
        self.visualization_manager = VisualizationManager()
        
        # Set core component references for all tabs
        tabs = [self.file_tab, self.peaks_tab, self.batch_tab, 
                self.results_tab, self.session_tab]
        
        for tab in tabs:
            tab.set_core_components(
                self.data_processor, 
                self.peak_fitter, 
                self.main_controller
            )
        
        # Initialize visualization manager
        self.visualization_manager.initialize(
            self.data_processor,
            self.peak_fitter, 
            self
        )
        
        print(f"UIManager: Created {len(tabs)} tab components + visualization manager")  # Debug
    
    def _setup_tab_connections(self):
        """Setup signal connections between tabs and UI manager"""
        tabs = [self.file_tab, self.peaks_tab, self.batch_tab, 
                self.results_tab, self.session_tab]
        
        for tab in tabs:
            # Connect tab actions to UI manager
            tab.action_requested.connect(self._handle_tab_action)
            tab.status_updated.connect(self._handle_tab_status)
            tab.data_changed.connect(self._handle_tab_data_change)
        
        print("UIManager: Tab connections established")  # Debug
    
    def _create_right_panel(self):
        """Create visualization panel with plot components"""
        if self.visualization_manager and self.visualization_manager.is_initialized:
            # Return the actual visualization widget
            return self.visualization_manager.get_main_widget()
        else:
            # Fallback placeholder if visualization manager isn't ready
            right_panel = QFrame()
            right_panel.setFrameStyle(QFrame.StyledPanel)
            right_layout = QVBoxLayout(right_panel)
            
            placeholder_label = QLabel("Visualization Loading...")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_label.setStyleSheet("""
                QLabel {
                    background-color: #F5F5F5;
                    border: 2px dashed #CCCCCC;
                    color: #666666;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 20px;
                }
            """)
            
            right_layout.addWidget(placeholder_label)
            right_layout.addStretch()
            
            return right_panel
    
    def _connect_status_signals(self):
        """Connect status update signals"""
        # Connect UI manager status to main status bar
        self.status_updated.connect(self._update_status_bar)
    
    def _handle_tab_action(self, action_name, parameters):
        """Handle action requests from tabs"""
        sender_tab = self.sender()
        tab_name = sender_tab.tab_name if hasattr(sender_tab, 'tab_name') else 'Unknown'
        
        print(f"UIManager: Action from {tab_name}: {action_name}")  # Debug
        
        # Add source tab information
        parameters['source_tab'] = tab_name
        
        # Forward to main controller
        self.action_requested.emit(action_name, parameters)
    
    def _handle_tab_status(self, status_message):
        """Handle status updates from tabs"""
        self.status_updated.emit(status_message)
    
    def _handle_tab_data_change(self, data):
        """Handle data changes from tabs"""
        sender_tab = self.sender()
        tab_name = sender_tab.tab_name if hasattr(sender_tab, 'tab_name') else 'Unknown'
        
        print(f"UIManager: Data change from {tab_name}")  # Debug
        
        # Notify other tabs if needed
        self._notify_tabs_of_data_change(tab_name, data)
    
    def _notify_tabs_of_data_change(self, source_tab_name, data):
        """Notify other tabs of data changes"""
        # This could implement cross-tab communication if needed
        # For now, we rely on core component signals
        pass
    
    def _update_status_bar(self, message):
        """Update the status bar"""
        if self.status_bar:
            self.status_bar.showMessage(message, 5000)  # Show for 5 seconds
    
    # Public interface methods for main controller
    
    def get_current_tab(self):
        """Get the currently active tab"""
        if not self.left_tabs:
            return None
        
        current_index = self.left_tabs.currentIndex()
        return self.left_tabs.widget(current_index)
    
    def set_current_tab(self, tab_name):
        """Set the current tab by name"""
        if not self.left_tabs:
            return False
        
        tab_map = {
            'File': 0,
            'Peaks': 1, 
            'Batch': 2,
            'Results': 3,
            'Session': 4
        }
        
        index = tab_map.get(tab_name)
        if index is not None:
            self.left_tabs.setCurrentIndex(index)
            return True
        return False
    
    def get_tab_by_name(self, tab_name):
        """Get a tab instance by name"""
        tab_map = {
            'File': self.file_tab,
            'Peaks': self.peaks_tab,
            'Batch': self.batch_tab,
            'Results': self.results_tab,
            'Session': self.session_tab
        }
        
        return tab_map.get(tab_name)
    
    def get_visualization_manager(self):
        """Get the visualization manager instance"""
        return self.visualization_manager
    
    def update_all_visualizations(self):
        """Update all visualization components"""
        if self.visualization_manager and self.visualization_manager.is_initialized:
            self.visualization_manager.update_all_plots()
    
    def update_visualization(self, plot_type):
        """Update a specific visualization"""
        if self.visualization_manager and self.visualization_manager.is_initialized:
            self.visualization_manager.update_plot(plot_type)
    
    def update_batch_visualizations(self):
        """Update visualizations that depend on batch results (waterfall, heatmap, trends)"""
        if self.visualization_manager and self.visualization_manager.is_initialized:
            self.visualization_manager.update_plot("trends")
            self.visualization_manager.update_plot("waterfall")
            self.visualization_manager.update_plot("heatmap")
    
    def force_update_all_plots(self):
        """Force update of all plots (use sparingly)"""
        if self.visualization_manager and self.visualization_manager.is_initialized:
            self.visualization_manager.force_update_all_plots()
    
    def update_all_tabs_from_data_processor(self):
        """Update all tabs when data processor state changes"""
        if not self.is_initialized:
            return
        
        tabs = [self.file_tab, self.peaks_tab, self.batch_tab, 
                self.results_tab, self.session_tab]
        
        for tab in tabs:
            tab.update_from_data_processor()
        
        # OPTIMIZED: Only update current spectrum plot for data changes
        # Waterfall and heatmap updates are handled separately when needed
        if self.visualization_manager and self.visualization_manager.is_initialized:
            self.visualization_manager.update_current_spectrum_only()
    
    def update_all_tabs_from_peak_fitter(self):
        """Update all tabs when peak fitter state changes"""
        if not self.is_initialized:
            return
        
        tabs = [self.file_tab, self.peaks_tab, self.batch_tab, 
                self.results_tab, self.session_tab]
        
        for tab in tabs:
            tab.update_from_peak_fitter()
        
        # OPTIMIZED: Only update current spectrum plot for peak fitting changes
        # Batch-dependent plots (trends, waterfall, heatmap) updated separately
        if self.visualization_manager and self.visualization_manager.is_initialized:
            self.visualization_manager.update_current_spectrum_only()
    
    def enable_tabs(self, enabled=True):
        """Enable or disable all tabs"""
        if not self.left_tabs:
            return
        
        self.left_tabs.setEnabled(enabled)
        
        if enabled:
            self.status_updated.emit("Interface enabled")
        else:
            self.status_updated.emit("Interface disabled")
    
    def reset_all_tabs(self):
        """Reset all tabs to default state"""
        if not self.is_initialized:
            return
        
        tabs = [self.file_tab, self.peaks_tab, self.batch_tab, 
                self.results_tab, self.session_tab]
        
        for tab in tabs:
            tab.reset_to_defaults()
        
        self.status_updated.emit("All tabs reset to defaults")
    
    def get_all_tab_data(self):
        """Get state data from all tabs"""
        if not self.is_initialized:
            return {}
        
        tab_data = {}
        
        if self.file_tab:
            tab_data['file'] = self.file_tab.get_tab_data()
        if self.peaks_tab:
            tab_data['peaks'] = self.peaks_tab.get_tab_data()
        if self.batch_tab:
            tab_data['batch'] = self.batch_tab.get_tab_data()
        if self.results_tab:
            tab_data['results'] = self.results_tab.get_tab_data()
        if self.session_tab:
            tab_data['session'] = self.session_tab.get_tab_data()
        
        return tab_data
    
    def set_all_tab_data(self, tab_data):
        """Set state data for all tabs"""
        if not self.is_initialized:
            return
        
        tabs = {
            'file': self.file_tab,
            'peaks': self.peaks_tab,
            'batch': self.batch_tab,
            'results': self.results_tab,
            'session': self.session_tab
        }
        
        for tab_name, data in tab_data.items():
            tab = tabs.get(tab_name)
            if tab and hasattr(tab, 'set_tab_data'):
                tab.set_tab_data(data)
    
    def validate_all_tabs(self):
        """Validate all tabs and return validation results"""
        if not self.is_initialized:
            return {'valid': False, 'errors': ['UI not initialized']}
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        tabs = [self.file_tab, self.peaks_tab, self.batch_tab, 
                self.results_tab, self.session_tab]
        
        for tab in tabs:
            is_valid, error_message = tab.validate_input()
            if not is_valid:
                validation_results['valid'] = False
                validation_results['errors'].append(f"{tab.tab_name}: {error_message}")
        
        return validation_results
    
    def get_ui_state_summary(self):
        """Get a summary of the current UI state"""
        if not self.is_initialized:
            return "UI Manager not initialized"
        
        current_tab = self.get_current_tab()
        current_tab_name = current_tab.tab_name if current_tab and hasattr(current_tab, 'tab_name') else 'Unknown'
        
        validation = self.validate_all_tabs()
        
        return {
            'initialized': self.is_initialized,
            'current_tab': current_tab_name,
            'total_tabs': 5,
            'validation': validation,
            'enabled': self.left_tabs.isEnabled() if self.left_tabs else False
        }
    
    def __repr__(self):
        return f"UIManager(initialized={self.is_initialized}, tabs=5)" 