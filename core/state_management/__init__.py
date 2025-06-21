"""
RamanLab Application State Management System

This module provides comprehensive state management capabilities for RamanLab,
allowing users to save and restore complete application sessions including
window layouts, data, analysis results, and preferences.
"""

from .state_manager import ApplicationStateManager
from .stateful_module import StatefulModule
from .session_manager import SessionManager
from .window_state import WindowStateManager
from .data_state import DataStateManager

__all__ = [
    'ApplicationStateManager',
    'StatefulModule', 
    'SessionManager',
    'WindowStateManager',
    'DataStateManager'
] 