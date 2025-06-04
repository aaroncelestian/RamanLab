"""
Core package for Raman Polarization Analyzer

This package contains the core business logic modules for Raman spectroscopy analysis:
- database: Database operations and mineral data management
- peak_fitting: Peak detection and fitting algorithms
- polarization: Polarization analysis and Raman tensor calculations

Additional modules to be implemented:
- spectrum: Spectrum processing and manipulation utilities
- file_io: File I/O operations for various spectrum formats
"""

# Database operations
from .database import MineralDatabase

# Peak fitting functionality  
from .peak_fitting import PeakFitter, PeakData, auto_find_peaks

# Polarization analysis
from .polarization import (
    PolarizationAnalyzer,
    PolarizedSpectrumGenerator,
    PolarizationData,
    DepolarizationResult,
    TensorAnalysisResult
)

# CIF Parser functionality
try:
    from parsers.cif_parser import CIFParser, CrystalStructure
    _parsers_available = True
except ImportError:
    _parsers_available = False

__all__ = [
    # Database
    'MineralDatabase',
    
    # Peak fitting
    'PeakFitter',
    'PeakData',
    'auto_find_peaks',
    
    # Polarization
    'PolarizationAnalyzer',
    'PolarizedSpectrumGenerator', 
    'PolarizationData',
    'DepolarizationResult',
    'TensorAnalysisResult',
]

# Add parsers to __all__ if available
if _parsers_available:
    __all__.extend(['CIFParser', 'CrystalStructure']) 