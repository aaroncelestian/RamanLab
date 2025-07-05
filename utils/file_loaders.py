"""
File Loaders Module

This module provides file loading functionality for various spectrum formats.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import os
from pathlib import Path


class SpectrumLoader:
    """
    Utility class for loading spectrum files in various formats.
    
    Supports:
    - Text files with tab, comma, or space delimited data
    - Simple two-column format (wavenumber, intensity)
    - Files with headers and metadata
    """
    
    def __init__(self):
        """Initialize the spectrum loader."""
        self.supported_extensions = ['.txt', '.dat', '.csv', '.asc', '.spc']
    
    def load_spectrum(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Load a spectrum file and return wavenumbers, intensities, and metadata.
        
        Args:
            file_path: Path to the spectrum file
            
        Returns:
            Tuple of (wavenumbers, intensities, metadata)
            Returns (None, None, {}) if loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None, None, {"error": f"File not found: {file_path}"}
            
            # Try to parse the file
            wavenumbers, intensities, metadata = self._parse_text_file(str(file_path))
            
            if wavenumbers is not None and intensities is not None:
                metadata.update({
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "data_points": len(wavenumbers)
                })
            
            return wavenumbers, intensities, metadata
            
        except Exception as e:
            return None, None, {"error": f"Error loading file: {str(e)}"}
    
    def _parse_text_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Parse a text file containing spectrum data.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple of (wavenumbers, intensities, metadata)
        """
        metadata = {}
        wavenumbers = []
        intensities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Look for data starting point
            data_start = 0
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                    
                # Check if this looks like a header/comment line
                if line.startswith('#') or line.startswith('%') or line.startswith(';'):
                    metadata[f"header_{i}"] = line
                    data_start = i + 1
                    continue
                
                # Try to parse as data
                try:
                    parts = self._split_line(line)
                    if len(parts) >= 2:
                        # Try to convert to numbers
                        float(parts[0])
                        float(parts[1])
                        # If successful, this is the start of data
                        data_start = i
                        break
                except (ValueError, IndexError):
                    # Not numeric data, probably still header
                    metadata[f"header_{i}"] = line
                    data_start = i + 1
                    continue
            
            # Parse data lines
            for line in lines[data_start:]:
                line = line.strip()
                
                if not line or line.startswith('#') or line.startswith('%'):
                    continue
                
                try:
                    parts = self._split_line(line)
                    if len(parts) >= 2:
                        wavenumber = float(parts[0])
                        intensity = float(parts[1])
                        wavenumbers.append(wavenumber)
                        intensities.append(intensity)
                except (ValueError, IndexError):
                    # Skip invalid lines
                    continue
            
            if len(wavenumbers) > 0:
                return np.array(wavenumbers), np.array(intensities), metadata
            else:
                return None, None, {"error": "No valid data found in file"}
                
        except Exception as e:
            return None, None, {"error": f"Error parsing file: {str(e)}"}
    
    def _split_line(self, line: str) -> list:
        """
        Split a line using various possible delimiters.
        
        Args:
            line: Line to split
            
        Returns:
            List of parts
        """
        # Try different delimiters
        for delimiter in ['\t', ',', ' ', ';']:
            parts = [part.strip() for part in line.split(delimiter) if part.strip()]
            if len(parts) >= 2:
                return parts
        
        # If no delimiter works, try splitting on whitespace
        return line.split()
    
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported by this loader.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if supported, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions
    
    def get_supported_extensions(self) -> list:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return self.supported_extensions.copy()


# Convenience function for simple loading
def load_spectrum_file(file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Convenience function to load a spectrum file.
    
    Args:
        file_path: Path to the spectrum file
        
    Returns:
        Tuple of (wavenumbers, intensities, metadata)
    """
    loader = SpectrumLoader()
    return loader.load_spectrum(file_path)


# Export main classes and functions
__all__ = ['SpectrumLoader', 'load_spectrum_file'] 