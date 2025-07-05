"""
Database Manager Module

This module provides a simplified interface to the MineralDatabase class
from the core.database module.
"""

from .database import MineralDatabase

# Create an alias for backward compatibility
DatabaseManager = MineralDatabase

# Re-export the main class
__all__ = ['DatabaseManager', 'MineralDatabase'] 