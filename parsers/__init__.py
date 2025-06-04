"""
File Parsing Modules for the Raman Polarization Analyzer.

This package contains modules for parsing various file formats including
CIF (Crystallographic Information File), spectrum files, and database formats.
"""

from .cif_parser import CifStructureParser, StructureData

__all__ = [
    'CifStructureParser',
    'StructureData'
] 