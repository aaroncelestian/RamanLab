"""
Core File I/O Module for Raman Polarization Analyzer

This module provides comprehensive file I/O functionality including:
- Loading spectrum data from various file formats (txt, csv, dat)
- Saving spectrum data to files
- File format detection and validation
- Metadata extraction and preservation
- Error handling and data validation

Dependencies:
    - numpy: Numerical computations
    - pandas: Data frame operations
    - pathlib: Modern path handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import os
from datetime import datetime

from .spectrum import SpectrumData


@dataclass
class FileInfo:
    """Information about a spectrum file."""
    path: Path
    size: int
    modified: datetime
    format: str
    encoding: str = "utf-8"
    delimiter: str = None
    header_lines: int = 0
    columns: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.columns is None:
            self.columns = []
        if self.metadata is None:
            self.metadata = {}


class SpectrumFileIO:
    """
    Advanced file I/O class for Raman spectrum data.
    
    This class provides comprehensive functionality for loading and saving
    spectrum data in various formats with automatic format detection,
    validation, and error handling.
    """
    
    def __init__(self):
        """Initialize the file I/O handler."""
        self.supported_extensions = {'.txt', '.csv', '.dat', '.asc', '.xy', '.spc'}
        self.common_delimiters = ['\t', ',', ' ', ';', '|']
        self.default_encoding = 'utf-8'
        
    def detect_file_format(self, file_path: Union[str, Path]) -> FileInfo:
        """
        Detect file format and extract basic information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileInfo object with detected format information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get basic file information
        stat = file_path.stat()
        file_info = FileInfo(
            path=file_path,
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
            format=file_path.suffix.lower()
        )
        
        # Try to detect delimiter and structure
        try:
            with open(file_path, 'r', encoding=self.default_encoding) as f:
                lines = [f.readline().strip() for _ in range(10)]  # Read first 10 lines
            
            # Count header lines (lines that don't start with numbers)
            header_lines = 0
            for line in lines:
                if line and not self._line_starts_with_number(line):
                    header_lines += 1
                else:
                    break
            
            # Detect delimiter from first data line
            if header_lines < len(lines):
                data_line = lines[header_lines]
                delimiter = self._detect_delimiter(data_line)
                file_info.delimiter = delimiter
                file_info.header_lines = header_lines
                
                # Try to detect column names
                if header_lines > 0:
                    header_line = lines[header_lines - 1]
                    if delimiter:
                        columns = [col.strip() for col in header_line.split(delimiter)]
                        file_info.columns = columns
                        
        except Exception as e:
            warnings.warn(f"Could not analyze file structure: {e}")
        
        return file_info
    
    def load_spectrum(self, file_path: Union[str, Path],
                     delimiter: str = None,
                     skip_rows: int = None,
                     usecols: Tuple[int, int] = (0, 1),
                     encoding: str = None) -> SpectrumData:
        """
        Load spectrum data from file with automatic format detection.
        
        Args:
            file_path: Path to the spectrum file
            delimiter: Column delimiter (auto-detected if None)
            skip_rows: Number of header rows to skip (auto-detected if None)
            usecols: Tuple of (wavenumber_col, intensity_col) indices
            encoding: File encoding (utf-8 if None)
            
        Returns:
            SpectrumData object containing the loaded spectrum
        """
        file_path = Path(file_path)
        encoding = encoding or self.default_encoding
        
        # Detect file format if parameters not provided
        if delimiter is None or skip_rows is None:
            file_info = self.detect_file_format(file_path)
            delimiter = delimiter or file_info.delimiter
            skip_rows = skip_rows or file_info.header_lines
        
        try:
            # Load data using pandas for robust parsing
            if delimiter:
                df = pd.read_csv(file_path, delimiter=delimiter, skiprows=skip_rows,
                               encoding=encoding, header=None)
            else:
                # Try to auto-detect delimiter
                df = pd.read_csv(file_path, skiprows=skip_rows, encoding=encoding,
                               header=None, delim_whitespace=True)
            
            # Extract wavenumbers and intensities
            if df.shape[1] < 2:
                raise ValueError("File must contain at least 2 columns (wavenumber, intensity)")
            
            wavenumber_col, intensity_col = usecols
            if wavenumber_col >= df.shape[1] or intensity_col >= df.shape[1]:
                raise ValueError(f"Column indices {usecols} exceed available columns ({df.shape[1]})")
            
            wavenumbers = df.iloc[:, wavenumber_col].values
            intensities = df.iloc[:, intensity_col].values
            
            # Validate data
            if len(wavenumbers) == 0:
                raise ValueError("No data found in file")
            
            # Remove NaN values
            valid_mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
            wavenumbers = wavenumbers[valid_mask]
            intensities = intensities[valid_mask]
            
            if len(wavenumbers) == 0:
                raise ValueError("No valid data points found after removing NaN values")
            
            # Sort by wavenumber if not already sorted
            if not np.all(wavenumbers[:-1] <= wavenumbers[1:]):
                sort_indices = np.argsort(wavenumbers)
                wavenumbers = wavenumbers[sort_indices]
                intensities = intensities[sort_indices]
            
            # Create metadata
            metadata = {
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'load_timestamp': datetime.now().isoformat(),
                'delimiter': delimiter,
                'skip_rows': skip_rows,
                'num_points': len(wavenumbers),
                'wavenumber_range': (float(np.min(wavenumbers)), float(np.max(wavenumbers))),
                'intensity_range': (float(np.min(intensities)), float(np.max(intensities)))
            }
            
            return SpectrumData(
                name=file_path.stem,
                wavenumbers=wavenumbers,
                intensities=intensities,
                source='file',
                metadata=metadata
            )
            
        except Exception as e:
            raise RuntimeError(f"Error loading spectrum from {file_path}: {str(e)}")
    
    def save_spectrum(self, spectrum: SpectrumData,
                     file_path: Union[str, Path],
                     format: str = 'auto',
                     delimiter: str = '\t',
                     include_header: bool = True,
                     precision: int = 6) -> None:
        """
        Save spectrum data to file.
        
        Args:
            spectrum: SpectrumData object to save
            file_path: Output file path
            format: Output format ('auto', 'txt', 'csv')
            delimiter: Column delimiter
            include_header: Whether to include column headers
            precision: Number of decimal places for numeric data
        """
        file_path = Path(file_path)
        
        # Auto-detect format from extension
        if format == 'auto':
            ext = file_path.suffix.lower()
            if ext == '.csv':
                delimiter = ','
            elif ext in ['.txt', '.dat', '.asc']:
                delimiter = '\t'
        
        # Prepare data
        data = np.column_stack((spectrum.wavenumbers, spectrum.intensities))
        
        # Create header
        header = None
        if include_header:
            header = f"Wavenumber{delimiter}Intensity"
            if spectrum.metadata:
                # Add metadata as comments
                metadata_lines = []
                for key, value in spectrum.metadata.items():
                    if isinstance(value, (int, float, str)):
                        metadata_lines.append(f"# {key}: {value}")
                
                if metadata_lines:
                    header = '\n'.join(metadata_lines) + '\n' + header
        
        try:
            # Save using numpy
            np.savetxt(file_path, data, delimiter=delimiter, header=header,
                      fmt=f'%.{precision}f', comments='')
            
        except Exception as e:
            raise RuntimeError(f"Error saving spectrum to {file_path}: {str(e)}")
    
    def load_multiple_spectra(self, file_paths: List[Union[str, Path]],
                             **kwargs) -> List[SpectrumData]:
        """
        Load multiple spectrum files.
        
        Args:
            file_paths: List of file paths to load
            **kwargs: Additional arguments passed to load_spectrum
            
        Returns:
            List of SpectrumData objects
        """
        spectra = []
        errors = []
        
        for file_path in file_paths:
            try:
                spectrum = self.load_spectrum(file_path, **kwargs)
                spectra.append(spectrum)
            except Exception as e:
                errors.append(f"Error loading {file_path}: {str(e)}")
        
        if errors:
            warnings.warn(f"Failed to load {len(errors)} files:\n" + '\n'.join(errors))
        
        return spectra
    
    def validate_spectrum_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a spectrum file and return validation results.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        file_path = Path(file_path)
        
        try:
            # Check if file exists
            if not file_path.exists():
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Check file size
            size = file_path.stat().st_size
            validation_result['info']['file_size'] = size
            
            if size == 0:
                validation_result['errors'].append("File is empty")
                return validation_result
            
            if size > 100 * 1024 * 1024:  # 100 MB
                validation_result['warnings'].append("File is very large (>100MB)")
            
            # Try to detect format
            file_info = self.detect_file_format(file_path)
            validation_result['info']['format_info'] = file_info
            
            # Try to load a small sample
            try:
                spectrum = self.load_spectrum(file_path)
                validation_result['info']['num_points'] = len(spectrum.wavenumbers)
                validation_result['info']['wavenumber_range'] = (
                    float(np.min(spectrum.wavenumbers)),
                    float(np.max(spectrum.wavenumbers))
                )
                validation_result['info']['intensity_range'] = (
                    float(np.min(spectrum.intensities)),
                    float(np.max(spectrum.intensities))
                )
                
                # Check for common issues
                if len(spectrum.wavenumbers) < 10:
                    validation_result['warnings'].append("Very few data points (<10)")
                
                if np.any(np.diff(spectrum.wavenumbers) <= 0):
                    validation_result['warnings'].append("Non-monotonic wavenumber values")
                
                if np.any(spectrum.intensities < 0):
                    validation_result['warnings'].append("Negative intensity values found")
                
                validation_result['valid'] = True
                
            except Exception as e:
                validation_result['errors'].append(f"Could not parse file content: {str(e)}")
        
        except Exception as e:
            validation_result['errors'].append(f"Unexpected error: {str(e)}")
        
        return validation_result
    
    def _line_starts_with_number(self, line: str) -> bool:
        """Check if a line starts with a number (possibly negative)."""
        line = line.strip()
        if not line:
            return False
        
        try:
            # Try to parse the first token as a number
            first_token = line.split()[0]
            float(first_token)
            return True
        except (ValueError, IndexError):
            return False
    
    def _detect_delimiter(self, line: str) -> Optional[str]:
        """Detect the delimiter used in a data line."""
        # Count occurrences of common delimiters
        delimiter_counts = {}
        for delimiter in self.common_delimiters:
            count = line.count(delimiter)
            if count > 0:
                delimiter_counts[delimiter] = count
        
        if not delimiter_counts:
            return None
        
        # Return delimiter with highest count
        return max(delimiter_counts, key=delimiter_counts.get)
    
    def export_to_csv(self, spectrum: SpectrumData,
                     file_path: Union[str, Path],
                     include_metadata: bool = True) -> None:
        """
        Export spectrum to CSV format with optional metadata.
        
        Args:
            spectrum: SpectrumData object to export
            file_path: Output CSV file path
            include_metadata: Whether to include metadata as comments
        """
        self.save_spectrum(spectrum, file_path, format='csv',
                         delimiter=',', include_header=True)
    
    def export_to_excel(self, spectra: List[SpectrumData],
                       file_path: Union[str, Path],
                       sheet_names: List[str] = None) -> None:
        """
        Export multiple spectra to Excel file with separate sheets.
        
        Args:
            spectra: List of SpectrumData objects
            file_path: Output Excel file path
            sheet_names: Optional list of sheet names
        """
        if not spectra:
            raise ValueError("No spectra provided for export")
        
        file_path = Path(file_path)
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for i, spectrum in enumerate(spectra):
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Wavenumber': spectrum.wavenumbers,
                        'Intensity': spectrum.intensities
                    })
                    
                    # Determine sheet name
                    if sheet_names and i < len(sheet_names):
                        sheet_name = sheet_names[i]
                    else:
                        sheet_name = spectrum.name or f"Spectrum_{i+1}"
                    
                    # Ensure valid sheet name
                    sheet_name = sheet_name[:31]  # Excel limit
                    
                    # Write to sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
        except Exception as e:
            raise RuntimeError(f"Error exporting to Excel: {str(e)}")


# Convenience functions for common operations
def load_spectrum(file_path: Union[str, Path], **kwargs) -> SpectrumData:
    """Convenience function to load a single spectrum."""
    io_handler = SpectrumFileIO()
    return io_handler.load_spectrum(file_path, **kwargs)


def save_spectrum(spectrum: SpectrumData, file_path: Union[str, Path], **kwargs) -> None:
    """Convenience function to save a spectrum."""
    io_handler = SpectrumFileIO()
    return io_handler.save_spectrum(spectrum, file_path, **kwargs)


def validate_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to validate a spectrum file."""
    io_handler = SpectrumFileIO()
    return io_handler.validate_spectrum_file(file_path) 