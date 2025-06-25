"""
Battery Strain Analysis Package
==============================

Implementation of chemical strain enhancement for battery materials,
specifically focusing on Li/H exchange in materials like LiMn2O4.

Modules:
- limn2o4_analyzer: LiMn2O4 specific strain analysis
- time_series_processor: Handle time series Raman data
- spinel_modes: Raman modes for spinel crystal structures
- strain_visualization: Plotting and visualization tools
"""

__version__ = "1.0.2"
__author__ = "RamanLab"

from .limn2o4_analyzer import LiMn2O4StrainAnalyzer
from .time_series_processor import TimeSeriesProcessor
from .spinel_modes import SpinelRamanModes
from .strain_visualization import StrainVisualizer

__all__ = [
    'LiMn2O4StrainAnalyzer',
    'TimeSeriesProcessor', 
    'SpinelRamanModes',
    'StrainVisualizer'
] 