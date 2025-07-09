#!/usr/bin/env python3
"""
Data Conversion Dialog for RamanLab
====================================

A flexible, modular dialog for various data conversion tools.
Designed for easy extension with new conversion capabilities.
"""

import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QPushButton, QLineEdit, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QGroupBox, QFormLayout, QCheckBox, QSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QSplitter, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont


class BaseConverter(ABC):
    """
    Abstract base class for data conversion tools.
    All converters should inherit from this class.
    """
    
    def __init__(self, parent=None):
        self.parent = parent
        self.name = "Base Converter"
        self.description = "Base class for data converters"
        self.input_extensions = []
        self.output_extensions = []
    
    @abstractmethod
    def create_ui(self, parent_widget: QWidget) -> QWidget:
        """Create and return the UI widget for this converter."""
        pass
    
    @abstractmethod
    def validate_input(self) -> tuple[bool, str]:
        """Validate input parameters. Returns (is_valid, error_message)."""
        pass
    
    @abstractmethod
    def convert(self, progress_callback=None) -> tuple[bool, str, List[str]]:
        """
        Perform the conversion.
        Returns (success, message, list_of_created_files).
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this converter."""
        return {
            'name': self.name,
            'description': self.description,
            'input_extensions': self.input_extensions,
            'output_extensions': self.output_extensions
        }


class LineScanConverter(BaseConverter):
    """
    Converter for splitting line scan Raman data into individual spectrum files.
    Handles files where row 1 = wavenumbers, column 1 = scan positions.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Line Scan Splitter"
        self.description = "Split line scan Raman data into individual spectrum files for batch processing"
        self.input_extensions = ['.txt', '.dat', '.csv']
        self.output_extensions = ['.txt']
        
        # UI components
        self.input_file_edit = None
        self.output_dir_edit = None
        self.base_filename_edit = None
        self.include_scan_number_checkbox = None
        self.delimiter_combo = None
        self.preview_table = None
        self.info_text = None
        
        # Data
        self.data = None
        self.wavenumbers = None
        self.scan_positions = None
        self.spectra_data = None
    
    def create_ui(self, parent_widget: QWidget) -> QWidget:
        """Create the UI for line scan conversion."""
        main_widget = QWidget(parent_widget)
        layout = QVBoxLayout(main_widget)
        
        # Header
        header_label = QLabel("Line Scan Raman Spectrum Splitter")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel(
            "Split multi-spectrum line scan files into individual two-column files.\n"
            "Input format: Row 1 = wavenumbers, Column 1 = scan positions, Data = intensities"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Input and Output sections side by side
        settings_layout = QHBoxLayout()
        
        # Input section
        input_group = QGroupBox("Input Settings")
        input_group.setMinimumWidth(350)  # Ensure consistent width
        input_layout = QFormLayout(input_group)
        
        # Input file selection
        input_file_layout = QHBoxLayout()
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Select line scan data file...")
        input_browse_btn = QPushButton("Browse")
        input_browse_btn.clicked.connect(self.browse_input_file)
        input_file_layout.addWidget(self.input_file_edit)
        input_file_layout.addWidget(input_browse_btn)
        input_layout.addRow("Input File:", input_file_layout)
        
        # Delimiter selection
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems(["Auto-detect", "Tab", "Comma", "Semicolon", "Space"])
        input_layout.addRow("Delimiter:", self.delimiter_combo)
        
        settings_layout.addWidget(input_group)
        
        # Output section
        output_group = QGroupBox("Output Settings")
        output_group.setMinimumWidth(350)  # Ensure consistent width
        output_layout = QFormLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(output_browse_btn)
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        # Base filename
        self.base_filename_edit = QLineEdit("spectrum")
        output_layout.addRow("Base Filename:", self.base_filename_edit)
        
        # Include scan number option
        self.include_scan_number_checkbox = QCheckBox("Include scan number in filenames")
        self.include_scan_number_checkbox.setChecked(True)
        output_layout.addRow("", self.include_scan_number_checkbox)
        
        settings_layout.addWidget(output_group)
        
        # Add the horizontal settings layout to main layout
        layout.addLayout(settings_layout)
        
        # Preview section - now with more vertical space
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview controls
        preview_controls = QHBoxLayout()
        load_preview_btn = QPushButton("Load & Preview")
        load_preview_btn.clicked.connect(self.load_and_preview)
        preview_controls.addWidget(load_preview_btn)
        preview_controls.addStretch()
        preview_layout.addLayout(preview_controls)
        
        # Info text - slightly larger now
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)  # Increased from 80
        self.info_text.setReadOnly(True)
        preview_layout.addWidget(self.info_text)
        
        # Preview table - much larger now to use the freed space
        self.preview_table = QTableWidget()
        self.preview_table.setMinimumHeight(250)  # Increased minimum height
        self.preview_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding  # Allow both horizontal and vertical expansion
        )
        preview_layout.addWidget(self.preview_table)
        
        layout.addWidget(preview_group)
        
        # Auto-populate output directory when input file is selected
        self.input_file_edit.textChanged.connect(self.auto_set_output_dir)
        
        layout.addStretch()
        return main_widget
    
    def browse_input_file(self):
        """Browse for input line scan file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Select Line Scan Data File",
            "",
            "Data Files (*.txt *.dat *.csv);;All Files (*)"
        )
        if file_path:
            self.input_file_edit.setText(file_path)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self.parent,
            "Select Output Directory"
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def auto_set_output_dir(self):
        """Auto-set output directory based on input file location."""
        input_file = self.input_file_edit.text().strip()
        if input_file and os.path.exists(input_file):
            input_path = Path(input_file)
            output_dir = input_path.parent / f"{input_path.stem}_split"
            self.output_dir_edit.setText(str(output_dir))
            
            # Auto-set base filename
            self.base_filename_edit.setText(input_path.stem)
    
    def load_and_preview(self):
        """Load the file and show preview information."""
        input_file = self.input_file_edit.text().strip()
        if not input_file or not os.path.exists(input_file):
            QMessageBox.warning(self.parent, "Warning", "Please select a valid input file.")
            return
        
        try:
            # Detect delimiter
            delimiter = self.detect_delimiter(input_file)
            
            # Load data
            self.data = pd.read_csv(input_file, delimiter=delimiter, index_col=0)
            
            # Extract components
            self.wavenumbers = np.array([float(col) for col in self.data.columns])
            self.scan_positions = self.data.index.values
            self.spectra_data = self.data.values
            
            # Update info
            info_text = f"File loaded successfully!\n"
            info_text += f"Number of spectra: {len(self.scan_positions)}\n"
            info_text += f"Wavenumber range: {self.wavenumbers.min():.1f} - {self.wavenumbers.max():.1f} cm⁻¹\n"
            info_text += f"Data points per spectrum: {len(self.wavenumbers)}\n"
            info_text += f"Detected delimiter: {repr(delimiter)}"
            
            self.info_text.setText(info_text)
            
            # Update preview table
            self.update_preview_table()
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to load file:\n{str(e)}")
            self.info_text.setText(f"Error loading file: {str(e)}")
    
    def detect_delimiter(self, file_path):
        """Detect the delimiter used in the file."""
        delimiter_selection = self.delimiter_combo.currentText()
        
        if delimiter_selection == "Tab":
            return '\t'
        elif delimiter_selection == "Comma":
            return ','
        elif delimiter_selection == "Semicolon":
            return ';'
        elif delimiter_selection == "Space":
            return ' '
        else:  # Auto-detect
            # Read first few lines to detect delimiter
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            
            # Try different delimiters
            for delimiter in ['\t', ',', ';', ' ']:
                if all(len(line.split(delimiter)) > 3 for line in first_lines if line):
                    return delimiter
            
            return '\t'  # Default fallback
    
    def update_preview_table(self):
        """Update the preview table with sample data."""
        if self.data is None:
            return
        
        # Show first few rows and columns
        preview_rows = min(5, len(self.scan_positions))
        preview_cols = min(8, len(self.wavenumbers))
        
        self.preview_table.setRowCount(preview_rows + 1)  # +1 for header
        self.preview_table.setColumnCount(preview_cols + 1)  # +1 for scan positions
        
        # Set headers
        headers = ["Scan Position"] + [f"{wn:.1f}" for wn in self.wavenumbers[:preview_cols]]
        self.preview_table.setHorizontalHeaderLabels(headers)
        
        # Add data
        for i in range(preview_rows):
            # Scan position
            self.preview_table.setItem(i, 0, QTableWidgetItem(str(self.scan_positions[i])))
            
            # Intensity data
            for j in range(preview_cols):
                intensity = self.spectra_data[i, j]
                self.preview_table.setItem(i, j + 1, QTableWidgetItem(f"{intensity:.1f}"))
        
        # Resize columns
        self.preview_table.resizeColumnsToContents()
    
    def validate_input(self) -> tuple[bool, str]:
        """Validate input parameters."""
        input_file = self.input_file_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        base_filename = self.base_filename_edit.text().strip()
        
        if not input_file:
            return False, "Please select an input file."
        
        if not os.path.exists(input_file):
            return False, "Input file does not exist."
        
        if not output_dir:
            return False, "Please select an output directory."
        
        if not base_filename:
            return False, "Please enter a base filename."
        
        if self.data is None:
            return False, "Please load and preview the file first."
        
        return True, "Input validation passed."
    
    def convert(self, progress_callback=None) -> tuple[bool, str, List[str]]:
        """Convert the line scan file to individual spectrum files."""
        try:
            output_dir = Path(self.output_dir_edit.text().strip())
            base_filename = self.base_filename_edit.text().strip()
            include_scan_number = self.include_scan_number_checkbox.isChecked()
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            created_files = []
            total_spectra = len(self.scan_positions)
            
            for i, scan_pos in enumerate(self.scan_positions):
                # Generate filename
                if include_scan_number:
                    filename = f"{base_filename}_scan_{scan_pos}.txt"
                else:
                    filename = f"{base_filename}_{i+1:03d}.txt"
                
                file_path = output_dir / filename
                
                # Create spectrum data
                spectrum_data = np.column_stack([self.wavenumbers, self.spectra_data[i]])
                
                # Write file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Wavenumber (cm-1)\tIntensity\n")
                    f.write(f"# Scan position: {scan_pos}\n")
                    for wn, intensity in spectrum_data:
                        f.write(f"{wn:.3f}\t{intensity:.6f}\n")
                
                created_files.append(str(file_path))
                
                # Update progress
                if progress_callback:
                    progress = int((i + 1) / total_spectra * 100)
                    progress_callback(progress)
            
            message = f"Successfully split {total_spectra} spectra into individual files.\n"
            message += f"Files saved to: {output_dir}"
            
            return True, message, created_files
            
        except Exception as e:
            return False, f"Conversion failed: {str(e)}", []


class WavelengthConverter(BaseConverter):
    """
    Converter for converting wavelength (nm) to wavenumbers (cm⁻¹).
    Handles CSV and TXT files with wavelength/intensity data.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Wavelength to Raman Shift"
        self.description = "Convert scattered wavelength (nm) to Raman shift (cm⁻¹) using laser wavelength"
        self.input_extensions = ['.csv', '.txt', '.dat']
        self.output_extensions = ['.txt']
        
        # UI components
        self.input_files_list = None
        self.output_dir_edit = None
        self.laser_wavelength_edit = None
        self.info_text = None
        self.preview_table = None
        
        # Data
        self.selected_files = []
    
    def create_ui(self, parent_widget: QWidget) -> QWidget:
        """Create the UI for wavelength conversion."""
        main_widget = QWidget(parent_widget)
        layout = QVBoxLayout(main_widget)
        
        # Header
        header_label = QLabel("Wavelength to Raman Shift Converter")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(12)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel(
            "Convert scattered wavelength (nm) to Raman shift (cm⁻¹) using laser wavelength.\n"
            "Formula: Raman Shift = (1/λ_laser - 1/λ_scattered) × 10⁷\n"
            "Select one or more CSV/TXT files. Output files will have '_rs' suffix."
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # File selection controls
        file_controls = QHBoxLayout()
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self.add_files)
        clear_files_btn = QPushButton("Clear All")
        clear_files_btn.clicked.connect(self.clear_files)
        
        file_controls.addWidget(add_files_btn)
        file_controls.addWidget(clear_files_btn)
        file_controls.addStretch()
        file_layout.addLayout(file_controls)
        
        # Selected files list
        self.input_files_list = QTableWidget()
        self.input_files_list.setColumnCount(2)
        self.input_files_list.setHorizontalHeaderLabels(["File Name", "Full Path"])
        self.input_files_list.horizontalHeader().setStretchLastSection(True)
        self.input_files_list.setMaximumHeight(150)
        file_layout.addWidget(self.input_files_list)
        
        layout.addWidget(file_group)
        
        # Conversion settings
        conversion_group = QGroupBox("Conversion Settings")
        conversion_layout = QFormLayout(conversion_group)
        
        # Laser wavelength input
        self.laser_wavelength_edit = QLineEdit("532")
        self.laser_wavelength_edit.setPlaceholderText("Enter laser wavelength in nm")
        conversion_layout.addRow("Laser Wavelength (nm):", self.laser_wavelength_edit)
        
        layout.addWidget(conversion_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory (default: same as input files)")
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(output_browse_btn)
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        layout.addWidget(output_group)
        
        # Preview section
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview controls
        preview_controls = QHBoxLayout()
        preview_btn = QPushButton("Preview First File")
        preview_btn.clicked.connect(self.preview_file)
        preview_controls.addWidget(preview_btn)
        preview_controls.addStretch()
        preview_layout.addLayout(preview_controls)
        
        # Info text
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(80)
        self.info_text.setReadOnly(True)
        preview_layout.addWidget(self.info_text)
        
        # Preview table
        self.preview_table = QTableWidget()
        self.preview_table.setMinimumHeight(150)
        self.preview_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        preview_layout.addWidget(self.preview_table)
        
        layout.addWidget(preview_group)
        
        layout.addStretch()
        return main_widget
    
    def add_files(self):
        """Add files for conversion."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self.parent,
            "Select Wavelength Data Files",
            "",
            "Data Files (*.csv *.txt *.dat);;All Files (*)"
        )
        
        if file_paths:
            for file_path in file_paths:
                if file_path not in [f['path'] for f in self.selected_files]:
                    self.selected_files.append({
                        'name': Path(file_path).name,
                        'path': file_path
                    })
            
            self.update_files_list()
    
    def clear_files(self):
        """Clear all selected files."""
        self.selected_files.clear()
        self.update_files_list()
        self.info_text.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
    
    def update_files_list(self):
        """Update the files list display."""
        self.input_files_list.setRowCount(len(self.selected_files))
        
        for i, file_info in enumerate(self.selected_files):
            name_item = QTableWidgetItem(file_info['name'])
            path_item = QTableWidgetItem(file_info['path'])
            
            self.input_files_list.setItem(i, 0, name_item)
            self.input_files_list.setItem(i, 1, path_item)
        
        self.input_files_list.resizeColumnsToContents()
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self.parent,
            "Select Output Directory"
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def preview_file(self):
        """Preview the first selected file."""
        if not self.selected_files:
            QMessageBox.warning(self.parent, "Warning", "Please select at least one file first.")
            return
        
        try:
            file_path = self.selected_files[0]['path']
            
            # Try to load the file
            if file_path.lower().endswith('.csv'):
                data = pd.read_csv(file_path)
            else:
                # Try tab-delimited first, then comma
                try:
                    data = pd.read_csv(file_path, delimiter='\t')
                except:
                    data = pd.read_csv(file_path, delimiter=',')
            
            # Identify wavelength and intensity columns
            wavelength_col = None
            intensity_col = None
            
            for col in data.columns:
                col_lower = col.lower()
                if 'wavelength' in col_lower or 'nm' in col_lower or 'lambda' in col_lower:
                    wavelength_col = col
                elif 'intensity' in col_lower or 'counts' in col_lower or 'signal' in col_lower:
                    intensity_col = col
            
            # If not found, assume first two columns
            if wavelength_col is None:
                wavelength_col = data.columns[0]
            if intensity_col is None and len(data.columns) > 1:
                intensity_col = data.columns[1]
            
            # Get laser wavelength
            try:
                laser_wavelength = float(self.laser_wavelength_edit.text().strip())
            except (ValueError, AttributeError):
                laser_wavelength = 532.0  # Default
            
            # Get scattered wavelength data and convert to Raman shift
            scattered_wavelengths = data[wavelength_col].values
            raman_shifts = (1/laser_wavelength - 1/scattered_wavelengths) * 10**7  # Convert to Raman shift cm⁻¹
            
            # Update info
            info_text = f"Preview: {Path(file_path).name}\n"
            info_text += f"Laser wavelength: {laser_wavelength:.1f} nm\n"
            info_text += f"Data points: {len(scattered_wavelengths)}\n"
            info_text += f"Scattered wavelength range: {scattered_wavelengths.min():.2f} - {scattered_wavelengths.max():.2f} nm\n"
            info_text += f"Raman shift range: {raman_shifts.min():.1f} - {raman_shifts.max():.1f} cm⁻¹\n"
            info_text += f"Columns: {wavelength_col} → Raman Shift, {intensity_col if intensity_col else 'N/A'}"
            
            self.info_text.setText(info_text)
            
            # Update preview table
            preview_rows = min(10, len(scattered_wavelengths))
            cols = 3 if intensity_col else 2
            self.preview_table.setRowCount(preview_rows)
            self.preview_table.setColumnCount(cols)
            
            headers = ["Scattered λ (nm)", "Raman Shift (cm⁻¹)"]
            if intensity_col:
                headers.append("Intensity")
            self.preview_table.setHorizontalHeaderLabels(headers)
            
            for i in range(preview_rows):
                # Scattered wavelength
                self.preview_table.setItem(i, 0, QTableWidgetItem(f"{scattered_wavelengths[i]:.3f}"))
                # Raman shift
                self.preview_table.setItem(i, 1, QTableWidgetItem(f"{raman_shifts[i]:.1f}"))
                # Intensity (if available)
                if intensity_col:
                    intensity_val = data[intensity_col].iloc[i]
                    self.preview_table.setItem(i, 2, QTableWidgetItem(f"{intensity_val}"))
            
            self.preview_table.resizeColumnsToContents()
            
        except Exception as e:
            error_msg = f"Failed to preview file: {str(e)}"
            self.info_text.setText(error_msg)
            QMessageBox.critical(self.parent, "Preview Error", error_msg)
    
    def validate_input(self) -> tuple[bool, str]:
        """Validate input parameters."""
        if not self.selected_files:
            return False, "Please select at least one input file."
        
        # Check if all files exist
        for file_info in self.selected_files:
            if not os.path.exists(file_info['path']):
                return False, f"File does not exist: {file_info['name']}"
        
        # Validate laser wavelength
        try:
            laser_wl = float(self.laser_wavelength_edit.text().strip())
            if laser_wl <= 0:
                return False, "Laser wavelength must be positive."
        except (ValueError, AttributeError):
            return False, "Please enter a valid laser wavelength."
        
        return True, "Input validation passed."
    
    def convert(self, progress_callback=None) -> tuple[bool, str, List[str]]:
        """Convert scattered wavelength files to Raman shift files."""
        try:
            output_dir = self.output_dir_edit.text().strip()
            laser_wavelength = float(self.laser_wavelength_edit.text().strip())
            created_files = []
            total_files = len(self.selected_files)
            
            for i, file_info in enumerate(self.selected_files):
                file_path = file_info['path']
                file_name = file_info['name']
                
                # Determine output directory
                if output_dir:
                    out_dir = Path(output_dir)
                else:
                    out_dir = Path(file_path).parent
                
                # Create output filename
                file_stem = Path(file_path).stem
                output_filename = f"{file_stem}_rs.txt"
                output_path = out_dir / output_filename
                
                # Load the file
                if file_path.lower().endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:
                    # Try tab-delimited first, then comma
                    try:
                        data = pd.read_csv(file_path, delimiter='\t')
                    except:
                        data = pd.read_csv(file_path, delimiter=',')
                
                # Identify columns
                wavelength_col = None
                intensity_col = None
                
                for col in data.columns:
                    col_lower = col.lower()
                    if 'wavelength' in col_lower or 'nm' in col_lower or 'lambda' in col_lower:
                        wavelength_col = col
                    elif 'intensity' in col_lower or 'counts' in col_lower or 'signal' in col_lower:
                        intensity_col = col
                
                # If not found, assume first two columns
                if wavelength_col is None:
                    wavelength_col = data.columns[0]
                if intensity_col is None and len(data.columns) > 1:
                    intensity_col = data.columns[1]
                
                # Convert scattered wavelengths to Raman shifts
                scattered_wavelengths = data[wavelength_col].values
                raman_shifts = (1/laser_wavelength - 1/scattered_wavelengths) * 10**7  # Convert to Raman shift cm⁻¹
                
                # Prepare output data
                if intensity_col:
                    intensities = data[intensity_col].values
                    output_data = np.column_stack([raman_shifts, intensities])
                    header = "Raman Shift (cm-1)\tIntensity"
                else:
                    output_data = raman_shifts.reshape(-1, 1)
                    header = "Raman Shift (cm-1)"
                
                # Write output file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"{header}\n")
                    f.write(f"# Converted from scattered wavelength: {file_name}\n")
                    f.write(f"# Laser wavelength: {laser_wavelength:.1f} nm\n")
                    f.write(f"# Scattered wavelength range: {scattered_wavelengths.min():.3f} - {scattered_wavelengths.max():.3f} nm\n")
                    f.write(f"# Raman shift range: {raman_shifts.min():.1f} - {raman_shifts.max():.1f} cm⁻¹\n")
                    
                    for row in output_data:
                        if len(row) == 2:
                            f.write(f"{row[0]:.3f}\t{row[1]:.6f}\n")
                        else:
                            f.write(f"{row[0]:.3f}\n")
                
                created_files.append(str(output_path))
                
                # Update progress
                if progress_callback:
                    progress = int((i + 1) / total_files * 100)
                    progress_callback(progress)
            
            message = f"Successfully converted {total_files} file(s) from scattered wavelength to Raman shift.\n"
            message += f"Laser wavelength: {laser_wavelength:.1f} nm\n"
            message += f"Output files saved with '_rs' suffix."
            
            return True, message, created_files
            
        except Exception as e:
            return False, f"Conversion failed: {str(e)}", []


class DataConversionDialog(QDialog):
    """
    Main dialog for data conversion tools.
    Provides a tabbed interface for different conversion capabilities.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RamanLab Data Conversion Tools")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        # Converter registry
        self.converters: Dict[str, BaseConverter] = {}
        self.tab_to_converter: Dict[int, BaseConverter] = {}  # Map tab index to converter
        
        # Setup UI
        self.setup_ui()
        
        # Register default converters
        self.register_converter("line_scan", LineScanConverter(self))
        self.register_converter("wavelength", WavelengthConverter(self))
        
        # Setup initial state
        self.current_converter = None
        
        # Select first tab if available
        if self.tab_widget.count() > 0:
            self.tab_widget.setCurrentIndex(0)
            self.on_tab_changed(0)
        
    def setup_ui(self):
        """Setup the main dialog UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Data Conversion Tools")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(16)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel(
            "Choose a conversion tool below to transform your data into formats compatible with RamanLab's analysis modules."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)
        
        # Tab widget for different converters
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)
        
        # Button box
        button_layout = QHBoxLayout()
        
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False)
        
        self.refresh_btn = QPushButton("Refresh Selection")
        self.refresh_btn.clicked.connect(self.refresh_converter_selection)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.convert_btn)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Connect tab change
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def register_converter(self, name: str, converter: BaseConverter):
        """Register a new converter."""
        self.converters[name] = converter
        
        # Create tab for this converter
        converter_widget = converter.create_ui(self.tab_widget)
        tab_index = self.tab_widget.addTab(converter_widget, converter.name)
        
        # Map tab index to converter
        self.tab_to_converter[tab_index] = converter
        
        # Update status
        self.status_text.append(f"Registered converter: {converter.name}")
    
    def on_tab_changed(self, index):
        """Handle tab change."""
        if index >= 0 and index in self.tab_to_converter:
            self.current_converter = self.tab_to_converter[index]
            self.convert_btn.setEnabled(True)
            
            # Show converter info
            info = self.current_converter.get_info()
            self.status_text.clear()
            self.status_text.append(f"Selected: {info['name']}")
            self.status_text.append(f"Description: {info['description']}")
            self.status_text.append(f"Input formats: {', '.join(info['input_extensions'])}")
        else:
            self.current_converter = None
            self.convert_btn.setEnabled(False)
            self.status_text.clear()
            self.status_text.append("No converter available for this tab.")
    
    def refresh_converter_selection(self):
        """Manually refresh the converter selection."""
        current_index = self.tab_widget.currentIndex()
        self.status_text.append(f"\n🔄 Refreshing converter selection...")
        self.status_text.append(f"Current tab index: {current_index}")
        self.status_text.append(f"Available converters: {list(self.tab_to_converter.keys())}")
        
        # Force re-selection
        self.on_tab_changed(current_index)
        
        if self.current_converter:
            self.status_text.append(f"✅ Successfully selected: {self.current_converter.name}")
        else:
            self.status_text.append(f"❌ Still no converter selected")
    
    def start_conversion(self):
        """Start the conversion process."""
        if not self.current_converter:
            # Debug information
            current_tab = self.tab_widget.currentIndex()
            total_tabs = self.tab_widget.count()
            available_converters = list(self.tab_to_converter.keys())
            
            error_msg = f"No converter selected.\n\n"
            error_msg += f"Debug Info:\n"
            error_msg += f"• Current tab index: {current_tab}\n"
            error_msg += f"• Total tabs: {total_tabs}\n"
            error_msg += f"• Available converter tabs: {available_converters}\n"
            error_msg += f"• Registered converters: {list(self.converters.keys())}"
            
            QMessageBox.warning(self, "No Converter Selected", error_msg)
            return
        
        # Validate input
        is_valid, error_message = self.current_converter.validate_input()
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_message)
            return
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.convert_btn.setEnabled(False)
        
        # Clear status
        self.status_text.clear()
        self.status_text.append("Starting conversion...")
        
        # Start conversion
        success, message, created_files = self.current_converter.convert(
            progress_callback=self.update_progress
        )
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.convert_btn.setEnabled(True)
        
        # Show results
        self.status_text.append("\n" + "="*50)
        if success:
            self.status_text.append("✅ CONVERSION SUCCESSFUL!")
            self.status_text.append(message)
            if created_files:
                self.status_text.append(f"\nCreated {len(created_files)} files:")
                for file_path in created_files[:5]:  # Show first 5 files
                    self.status_text.append(f"  • {Path(file_path).name}")
                if len(created_files) > 5:
                    self.status_text.append(f"  ... and {len(created_files) - 5} more files")
            
            # Show success dialog
            QMessageBox.information(self, "Success", "Conversion completed successfully!")
        else:
            self.status_text.append("❌ CONVERSION FAILED!")
            self.status_text.append(message)
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")
    
    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)


# Convenience function for easy importing
def launch_data_conversion_dialog(parent=None):
    """Launch the data conversion dialog."""
    dialog = DataConversionDialog(parent)
    return dialog.exec()


if __name__ == "__main__":
    # For testing
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = DataConversionDialog()
    dialog.show()
    sys.exit(app.exec()) 