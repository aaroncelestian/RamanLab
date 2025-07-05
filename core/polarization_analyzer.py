"""
Polarization Analyzer Module

This module provides a simplified interface to the PolarizationAnalyzer class
from the core.polarization module.
"""

from .polarization import PolarizationAnalyzer, PolarizationData, DepolarizationResult, TensorAnalysisResult

# Re-export the main class and related data structures
__all__ = ['PolarizationAnalyzer', 'PolarizationData', 'DepolarizationResult', 'TensorAnalysisResult'] 