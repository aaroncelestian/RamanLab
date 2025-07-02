"""
Base Tab Component for Batch Peak Fitting UI
Provides common functionality for all tab components
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QObject, Signal


class BaseTab(QWidget):
    """
    Base class for all tab components in the batch peak fitting interface.
    Provides common functionality and standardized interface.
    """
    
    # Common signals that tabs can emit
    data_changed = Signal(dict)  # Generic data change signal
    action_requested = Signal(str, dict)  # Request action from parent (action_name, params)
    status_updated = Signal(str)  # Status message update
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # References to core components (set by UI manager)
        self.data_processor = None
        self.peak_fitter = None
        self.main_controller = None
        
        # Tab state
        self.is_initialized = False
        self.tab_name = self.__class__.__name__
        
        # Setup UI
        self.main_layout = QVBoxLayout(self)
        self.setup_ui()
        self.connect_signals()
        
        self.is_initialized = True
    
    def setup_ui(self):
        """
        Setup the UI for this tab.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement setup_ui()")
    
    def connect_signals(self):
        """
        Connect internal signals within the tab.
        Can be overridden by subclasses for additional connections.
        """
        pass
    
    def set_core_components(self, data_processor, peak_fitter, main_controller):
        """
        Set references to core components.
        Called by UI manager during initialization.
        """
        self.data_processor = data_processor
        self.peak_fitter = peak_fitter
        self.main_controller = main_controller
        
        # Connect to core component signals if needed
        self.connect_core_signals()
    
    def connect_core_signals(self):
        """
        Connect to core component signals.
        Can be overridden by subclasses.
        """
        pass
    
    def update_from_data_processor(self, data=None):
        """
        Update tab contents when data processor state changes.
        Can be overridden by subclasses.
        """
        pass
    
    def update_from_peak_fitter(self, data=None):
        """
        Update tab contents when peak fitter state changes.
        Can be overridden by subclasses.
        """
        pass
    
    def get_tab_data(self):
        """
        Get current tab state as dictionary.
        Can be overridden by subclasses.
        """
        return {
            'tab_name': self.tab_name,
            'is_initialized': self.is_initialized
        }
    
    def set_tab_data(self, data):
        """
        Set tab state from dictionary.
        Can be overridden by subclasses.
        """
        pass
    
    def validate_input(self):
        """
        Validate current input state.
        Returns (is_valid, error_message)
        """
        return True, ""
    
    def reset_to_defaults(self):
        """
        Reset tab to default state.
        Can be overridden by subclasses.
        """
        pass
    
    def enable_tab(self, enabled=True):
        """
        Enable or disable the entire tab.
        """
        self.setEnabled(enabled)
    
    def emit_action(self, action_name, params=None):
        """
        Convenience method to emit action requests.
        """
        if params is None:
            params = {}
        self.action_requested.emit(action_name, params)
    
    def emit_status(self, message):
        """
        Convenience method to emit status updates.
        """
        self.status_updated.emit(f"[{self.tab_name}] {message}")
    
    def get_component_reference(self, component_name):
        """
        Get reference to a core component by name.
        """
        if component_name == "data_processor":
            return self.data_processor
        elif component_name == "peak_fitter":
            return self.peak_fitter
        elif component_name == "main_controller":
            return self.main_controller
        else:
            return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(initialized={self.is_initialized})" 