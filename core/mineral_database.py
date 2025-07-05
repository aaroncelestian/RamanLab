"""
Mineral Database Module

This module provides a simplified interface to the MineralDatabase class
from the core.database module.
"""

from .database import MineralDatabase

# Re-export the main class
__all__ = ['MineralDatabase'] 