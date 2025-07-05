"""
Polarized Spectrum Generator Module

This module provides a simplified interface to the PolarizedSpectrumGenerator class
from the core.polarization module.
"""

from .polarization import PolarizedSpectrumGenerator, get_polarization_factor_simple

# Re-export the main class and utility functions
__all__ = ['PolarizedSpectrumGenerator', 'get_polarization_factor_simple'] 