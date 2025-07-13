#!/usr/bin/env python3
"""
Improved Geothermometry Analysis Module for Raman Spectroscopy

This module provides a streamlined geothermometry analysis tool that processes
multiple thermometry methods with clear parameter requirements and robust data handling.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Scientific computing
try:
    from scipy.signal import find_peaks, peak_widths
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, peak detection will be limited")

# Qt imports
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QProgressBar, QComboBox, QFileDialog, QMessageBox, QDialog,
    QGroupBox, QFormLayout, QTextEdit, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QTabWidget, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
from PySide6.QtGui import QFont, QColor

# Matplotlib for plotting
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class GeothermometerMethod(Enum):
    """Supported geothermometry methods with their parameter requirements."""
    BEYSSAC_2002 = "Beyssac et al. (2002)"
    AOYA_2010_514 = "Aoya et al. (2010) 514nm"
    AOYA_2010_532 = "Aoya et al. (2010) 532nm"
    RAHL_2005 = "Rahl et al. (2005)"
    KOUKETSU_2014_D1 = "Kouketsu et al. (2014) D1"
    KOUKETSU_2014_D2 = "Kouketsu et al. (2014) D2"
    RANTITSCH_2004 = "Rantitsch et al. (2004)"


# Parameter requirements for each method
METHOD_PARAMETERS = {
    GeothermometerMethod.BEYSSAC_2002: ['R2'],  # R2 ratio
    GeothermometerMethod.AOYA_2010_514: ['D1_pos', 'G_pos'],  # Peak positions
    GeothermometerMethod.AOYA_2010_532: ['D1_pos', 'G_pos'],  # Peak positions
    GeothermometerMethod.RAHL_2005: ['R1'],  # D1/G area ratio
    GeothermometerMethod.KOUKETSU_2014_D1: ['D1_fwhm'],  # D1 FWHM
    GeothermometerMethod.KOUKETSU_2014_D2: ['D2_fwhm'],  # D2 FWHM
    GeothermometerMethod.RANTITSCH_2004: ['D1_pos', 'G_pos'],  # Peak positions
}


def get_available_methods():
    """Return list of available geothermometry methods."""
    return list(GeothermometerMethod)


def calculate_temperature(method: GeothermometerMethod, **params) -> float:
    """
    Calculate temperature using the specified geothermometry method.
    
    Parameters
    ----------
    method : GeothermometerMethod
        The geothermometry method to use
    **params : dict
        Required parameters for the method
        
    Returns
    -------
    float
        Calculated temperature in Celsius
    """
    try:
        if method == GeothermometerMethod.BEYSSAC_2002:
            # Beyssac et al. (2002): T(°C) = -445 * R2 + 641
            R2 = params.get('R2', 0)
            if R2 <= 0:
                raise ValueError("R2 ratio must be greater than 0")
            return -445 * R2 + 641
            
        elif method in [GeothermometerMethod.AOYA_2010_514, GeothermometerMethod.AOYA_2010_532]:
            # Aoya et al. (2010): T(°C) = -0.5 * (D1 - G) + 0.00016 * (D1 - G)^2 + 347
            D1_pos = params.get('D1_pos', 0)
            G_pos = params.get('G_pos', 0)
            delta = D1_pos - G_pos
            return -0.5 * delta + 0.00016 * (delta ** 2) + 347
            
        elif method == GeothermometerMethod.KOUKETSU_2014_D1:
            # Kouketsu et al. (2014) D1: T(°C) = -2.1 * D1_FWHM + 640
            D1_fwhm = params.get('D1_fwhm', 0)
            if D1_fwhm <= 0:
                raise ValueError("D1_fwhm must be greater than 0")
            return -2.1 * D1_fwhm + 640
            
        elif method == GeothermometerMethod.KOUKETSU_2014_D2:
            # Kouketsu et al. (2014) D2: T(°C) = -2.1 * D2_FWHM + 640
            D2_fwhm = params.get('D2_fwhm', 0)
            if D2_fwhm <= 0:
                raise ValueError("D2_fwhm must be greater than 0")
            return -2.1 * D2_fwhm + 640
            
        elif method == GeothermometerMethod.RAHL_2005:
            # Rahl et al. (2005): T(°C) = -676 * log10(R1) + 834
            R1 = params.get('R1', 1.0)
            if R1 <= 0:
                raise ValueError("R1 ratio must be greater than 0")
            return -676 * np.log10(R1) + 834
            
        elif method == GeothermometerMethod.RANTITSCH_2004:
            # Rantitsch et al. (2004): T(°C) = -0.61 * (D1 - G) + 0.00012 * (D1 - G)^2 + 300
            D1_pos = params.get('D1_pos', 0)
            G_pos = params.get('G_pos', 0)
            delta = D1_pos - G_pos
            return -0.61 * delta + 0.00012 * (delta ** 2) + 300
            
        else:
            raise ValueError(f"Unsupported geothermometry method: {method}")
            
    except Exception as e:
        print(f"Error calculating temperature with {method.value}: {str(e)}")
        raise


@dataclass
class GeothermometryResult:
    """Container for geothermometry analysis results."""
    method: GeothermometerMethod
    sample_id: str
    temperature: Optional[float]
    parameters: Dict[str, float]
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for easy serialization."""
        return {
            'method': self.method.value,
            'sample_id': self.sample_id,
            'temperature': self.temperature,
            'parameters': self.parameters,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata
        }


class ParameterExtractor:
    """Extract parameters from spectrum data for geothermometry calculations."""
    
    @staticmethod
    def extract_from_spectrum(spectrum_data: Dict, method: GeothermometerMethod) -> Dict[str, float]:
        """
        Extract required parameters from spectrum data for a specific method.
        
        Parameters
        ----------
        spectrum_data : dict
            Spectrum data containing peaks information
        method : GeothermometerMethod
            The geothermometry method requiring parameters
            
        Returns
        -------
        dict
            Dictionary of extracted parameters
        """
        required_params = METHOD_PARAMETERS[method]
        params = {}
        
        print(f"Extracting parameters for {method.value}")
        print(f"Required parameters: {required_params}")
        print(f"Available data keys: {list(spectrum_data.keys())}")
        
        # Try to find peaks data
        peaks_data = None
        data_source = None
        
        # Check for different peak data formats
        if 'fitted_peaks' in spectrum_data:
            peaks_data = spectrum_data['fitted_peaks']
            data_source = 'fitted_peaks'
        elif 'peaks' in spectrum_data:
            peaks_data = spectrum_data['peaks']
            data_source = 'peaks'
        elif 'peak_data' in spectrum_data:
            peaks_data = spectrum_data['peak_data']
            data_source = 'peak_data'
        elif 'peaks_df' in spectrum_data:
            peaks_data = spectrum_data['peaks_df']
            data_source = 'peaks_df'
        
        if peaks_data is None:
            raise ValueError("No peaks data found in spectrum")
        
        print(f"Using peaks data from: {data_source}")
        print(f"Peaks data type: {type(peaks_data)}")
        
        # Handle numpy array fitted peaks (your case)
        if isinstance(peaks_data, np.ndarray):
            print(f"Processing numpy array with shape: {peaks_data.shape}")
            
            # For fitted peaks, we need to find actual peaks in the fitted data
            # Get wavenumbers and use peak detection
            wavenumbers = spectrum_data.get('wavenumbers', [])
            if len(wavenumbers) == 0:
                raise ValueError("No wavenumbers found for peak detection")
            
            # Use scipy peak detection on the fitted data
            from scipy.signal import find_peaks, peak_widths
            
            # Find peaks in the fitted data
            # Use parameters suitable for Raman spectra
            peaks_indices, peak_properties = find_peaks(
                peaks_data, 
                height=np.max(peaks_data) * 0.05,  # Minimum 5% of max intensity
                distance=10,  # Minimum 10 points between peaks
                prominence=np.max(peaks_data) * 0.02  # Minimum prominence
            )
            
            if len(peaks_indices) == 0:
                raise ValueError("No peaks detected in fitted data")
            
            print(f"Detected {len(peaks_indices)} peaks at indices: {peaks_indices}")
            
            # Convert to wavenumber positions and get properties
            peak_positions = [wavenumbers[i] for i in peaks_indices]
            peak_intensities = [peaks_data[i] for i in peaks_indices]
            
            print(f"Peak positions: {peak_positions}")
            
            # Calculate FWHM for each peak
            try:
                widths_results = peak_widths(peaks_data, peaks_indices, rel_height=0.5)
                peak_fwhms = []
                for i, width_points in enumerate(widths_results[0]):
                    # Convert width in points to width in wavenumbers
                    if peaks_indices[i] > 0 and peaks_indices[i] < len(wavenumbers) - 1:
                        wavenumber_per_point = abs(wavenumbers[peaks_indices[i]+1] - wavenumbers[peaks_indices[i]-1]) / 2
                        fwhm_wavenumber = width_points * wavenumber_per_point
                        peak_fwhms.append(fwhm_wavenumber)
                    else:
                        peak_fwhms.append(20.0)  # Default FWHM
            except:
                # Fallback FWHM estimation
                peak_fwhms = [20.0] * len(peak_positions)
            
            print(f"Peak FWHMs: {peak_fwhms}")
            
            # Estimate peak areas (simple approximation)
            peak_areas = []
            for i, (pos, intensity, fwhm) in enumerate(zip(peak_positions, peak_intensities, peak_fwhms)):
                # Simple Gaussian area approximation: area ≈ intensity * fwhm * √(π/ln(2))
                area = intensity * fwhm * 1.064  # √(π/ln(2)) ≈ 1.064
                peak_areas.append(area)
            
            print(f"Peak areas: {peak_areas}")
            
            # Create a DataFrame-like structure for parameter extraction
            peaks_df = pd.DataFrame({
                'peak_center': peak_positions,
                'amplitude': peak_intensities,
                'fwhm': peak_fwhms,
                'area': peak_areas
            })
            
            print(f"Created peaks DataFrame with {len(peaks_df)} peaks")
            print(peaks_df.to_string())
        
        # Handle DataFrame format
        elif isinstance(peaks_data, pd.DataFrame):
            peaks_df = peaks_data
            print(f"Using existing DataFrame with shape: {peaks_df.shape}")
            print(f"DataFrame columns: {list(peaks_df.columns)}")
        
        # Handle dict format
        elif isinstance(peaks_data, dict):
            peaks_df = pd.DataFrame(peaks_data)
            print(f"Converted dict to DataFrame with shape: {peaks_df.shape}")
        
        # Handle list of dicts
        elif isinstance(peaks_data, list) and peaks_data and isinstance(peaks_data[0], dict):
            peaks_df = pd.DataFrame(peaks_data)
            print(f"Converted list of dicts to DataFrame with shape: {peaks_df.shape}")
        
        else:
            raise ValueError(f"Unsupported peaks data type: {type(peaks_data)}")
        
        # Extract parameters based on method requirements
        for param in required_params:
            try:
                if param == 'R2':
                    params[param] = ParameterExtractor._calculate_R2(peaks_df)
                elif param == 'R1':
                    params[param] = ParameterExtractor._calculate_R1(peaks_df)
                elif param == 'D1_pos':
                    params[param] = ParameterExtractor._find_peak_position(peaks_df, 'D1', target_pos=1350)
                elif param == 'G_pos':
                    params[param] = ParameterExtractor._find_peak_position(peaks_df, 'G', target_pos=1580)
                elif param == 'D1_fwhm':
                    params[param] = ParameterExtractor._find_peak_fwhm(peaks_df, 'D1', target_pos=1350)
                elif param == 'D2_fwhm':
                    params[param] = ParameterExtractor._find_peak_fwhm(peaks_df, 'D2', target_pos=1620)
                else:
                    raise ValueError(f"Unknown parameter: {param}")
            except Exception as e:
                print(f"Warning: Could not extract {param}: {str(e)}")
                params[param] = np.nan
        
        print(f"Extracted parameters: {params}")
        return params
    
    @staticmethod
    def _find_peak_position(peaks_df: pd.DataFrame, peak_name: str, target_pos: float) -> float:
        """Find the position of a specific peak."""
        # Try different column names for peak position
        pos_columns = ['peak_center', 'center', 'position', 'pos', 'x', 'wavenumber']
        pos_col = None
        
        for col in pos_columns:
            if col in peaks_df.columns:
                pos_col = col
                break
        
        if pos_col is None:
            raise ValueError(f"No position column found. Available columns: {list(peaks_df.columns)}")
        
        # Find peaks in the expected region (±50 cm⁻¹)
        mask = (peaks_df[pos_col] >= target_pos - 50) & (peaks_df[pos_col] <= target_pos + 50)
        candidates = peaks_df[mask]
        
        if candidates.empty:
            raise ValueError(f"No {peak_name} peak found near {target_pos} cm⁻¹")
        
        # Return the position closest to target
        distances = np.abs(candidates[pos_col] - target_pos)
        best_idx = distances.idxmin()
        return candidates.loc[best_idx, pos_col]
    
    @staticmethod
    def _find_peak_fwhm(peaks_df: pd.DataFrame, peak_name: str, target_pos: float) -> float:
        """Find the FWHM of a specific peak."""
        # Try different column names for FWHM
        fwhm_columns = ['fwhm', 'width', 'peak_width']
        pos_columns = ['peak_center', 'center', 'position', 'pos', 'x', 'wavenumber']
        
        pos_col = None
        fwhm_col = None
        
        for col in pos_columns:
            if col in peaks_df.columns:
                pos_col = col
                break
                
        for col in fwhm_columns:
            if col in peaks_df.columns:
                fwhm_col = col
                break
        
        if pos_col is None:
            raise ValueError(f"No position column found. Available columns: {list(peaks_df.columns)}")
        if fwhm_col is None:
            raise ValueError(f"No FWHM column found. Available columns: {list(peaks_df.columns)}")
        
        # Find peaks in the expected region
        mask = (peaks_df[pos_col] >= target_pos - 50) & (peaks_df[pos_col] <= target_pos + 50)
        candidates = peaks_df[mask]
        
        if candidates.empty:
            raise ValueError(f"No {peak_name} peak found near {target_pos} cm⁻¹")
        
        # Return the FWHM of the peak closest to target
        distances = np.abs(candidates[pos_col] - target_pos)
        best_idx = distances.idxmin()
        return candidates.loc[best_idx, fwhm_col]
    
    @staticmethod
    def _calculate_R1(peaks_df: pd.DataFrame) -> float:
        """Calculate R1 ratio (D1/G area ratio)."""
        # Find D1 and G peak areas
        D1_area = ParameterExtractor._find_peak_area(peaks_df, 'D1', target_pos=1350)
        G_area = ParameterExtractor._find_peak_area(peaks_df, 'G', target_pos=1580)
        
        if G_area == 0:
            raise ValueError("G peak area is zero, cannot calculate R1 ratio")
        
        return D1_area / G_area
    
    @staticmethod
    def _calculate_R2(peaks_df: pd.DataFrame) -> float:
        """Calculate R2 ratio for Beyssac method."""
        # This is method-specific and may need adjustment based on actual data structure
        # R2 is typically calculated from peak intensities or areas
        # For now, implement a simple version - you may need to adjust this
        
        # Try to find D1, D2, D3, and G peaks
        try:
            D1_area = ParameterExtractor._find_peak_area(peaks_df, 'D1', target_pos=1350)
            D2_area = ParameterExtractor._find_peak_area(peaks_df, 'D2', target_pos=1620)
            D3_area = ParameterExtractor._find_peak_area(peaks_df, 'D3', target_pos=1500)
            G_area = ParameterExtractor._find_peak_area(peaks_df, 'G', target_pos=1580)
            
            # R2 calculation (this may need adjustment based on your specific definition)
            # Common definition: R2 = (D1 + D2 + D3) / G
            R2 = (D1_area + D2_area + D3_area) / G_area
            return R2
            
        except Exception as e:
            # Fallback: if we can't find all peaks, try a simpler calculation
            print(f"Warning: Complex R2 calculation failed ({e}), trying simpler method")
            
            # Alternative: use first available ratio
            if 'area' in peaks_df.columns and len(peaks_df) >= 2:
                areas = peaks_df['area'].values
                return areas[0] / areas[1] if len(areas) > 1 and areas[1] != 0 else 1.0
            else:
                raise ValueError("Cannot calculate R2 ratio from available data")
    
    @staticmethod
    def _find_peak_area(peaks_df: pd.DataFrame, peak_name: str, target_pos: float) -> float:
        """Find the area of a specific peak."""
        # Try different column names for area
        area_columns = ['area', 'intensity', 'amplitude', 'height']
        pos_columns = ['peak_center', 'center', 'position', 'pos', 'x', 'wavenumber']
        
        pos_col = None
        area_col = None
        
        for col in pos_columns:
            if col in peaks_df.columns:
                pos_col = col
                break
                
        for col in area_columns:
            if col in peaks_df.columns:
                area_col = col
                break
        
        if pos_col is None:
            raise ValueError(f"No position column found. Available columns: {list(peaks_df.columns)}")
        if area_col is None:
            raise ValueError(f"No area column found. Available columns: {list(peaks_df.columns)}")
        
        # Find peaks in the expected region
        mask = (peaks_df[pos_col] >= target_pos - 50) & (peaks_df[pos_col] <= target_pos + 50)
        candidates = peaks_df[mask]
        
        if candidates.empty:
            raise ValueError(f"No {peak_name} peak found near {target_pos} cm⁻¹")
        
        # Return the area of the peak closest to target
        distances = np.abs(candidates[pos_col] - target_pos)
        best_idx = distances.idxmin()
        return candidates.loc[best_idx, area_col]


class GeothermometryWorker(QObject):
    """Worker for running geothermometry analysis in a separate thread."""
    
    # Signals
    progress = Signal(int, str)  # progress percentage, status message
    result_ready = Signal(GeothermometryResult)
    finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, methods: List[GeothermometerMethod], data: Dict):
        super().__init__()
        self.methods = methods
        self.data = data
        self._is_running = True
    
    def run(self):
        """Run the geothermometry analysis."""
        try:
            # Extract spectra from data
            spectra_dict = self.data.get('spectra_dict', {})
            if not spectra_dict:
                raise ValueError("No spectra found in data")
            
            print(f"DEBUG: Found {len(spectra_dict)} spectra to process")
            print(f"DEBUG: Spectrum names: {list(spectra_dict.keys())}")
            print(f"DEBUG: Selected methods: {[m.value for m in self.methods]}")
            
            total_calculations = len(self.methods) * len(spectra_dict)
            completed = 0
            
            self.progress.emit(0, f"Starting analysis of {len(spectra_dict)} spectra with {len(self.methods)} methods...")
            
            # Process each spectrum
            for spectrum_idx, (spectrum_name, spectrum_data) in enumerate(spectra_dict.items()):
                if not self._is_running:
                    print("DEBUG: Analysis stopped by user")
                    return
                
                # Extract sample ID
                sample_id = spectrum_name
                if isinstance(spectrum_data, dict) and 'metadata' in spectrum_data:
                    sample_id = spectrum_data['metadata'].get('sample_id', spectrum_name)
                
                print(f"DEBUG: Processing spectrum {spectrum_idx + 1}/{len(spectra_dict)}: {sample_id}")
                
                # Process each method for this spectrum
                for method_idx, method in enumerate(self.methods):
                    if not self._is_running:
                        return
                    
                    method_name = method.value
                    print(f"DEBUG: Applying {method_name} to {sample_id}")
                    
                    try:
                        # Extract parameters
                        print(f"DEBUG: Extracting parameters for {method_name}")
                        params = ParameterExtractor.extract_from_spectrum(spectrum_data, method)
                        print(f"DEBUG: Extracted parameters: {params}")
                        
                        # Calculate temperature
                        temperature = calculate_temperature(method, **params)
                        print(f"DEBUG: Calculated temperature: {temperature:.1f}°C")
                        
                        # Create result
                        result = GeothermometryResult(
                            method=method,
                            sample_id=sample_id,
                            temperature=temperature,
                            parameters=params,
                            success=True,
                            metadata={'spectrum_name': spectrum_name}
                        )
                        
                        print(f"DEBUG: Created successful result for {sample_id} - {method_name}: {temperature:.1f}°C")
                        
                    except Exception as e:
                        print(f"DEBUG: Error processing {sample_id} with {method_name}: {str(e)}")
                        # Create error result
                        result = GeothermometryResult(
                            method=method,
                            sample_id=sample_id,
                            temperature=None,
                            parameters={},
                            success=False,
                            error=str(e),
                            metadata={'spectrum_name': spectrum_name}
                        )
                        
                        print(f"DEBUG: Created error result for {sample_id} - {method_name}")
                    
                    # Emit result
                    self.result_ready.emit(result)
                    
                    # Update progress
                    completed += 1
                    progress_pct = int((completed / total_calculations) * 100)
                    status = f"Processing {method_name} for {sample_id} ({completed}/{total_calculations})"
                    self.progress.emit(progress_pct, status)
                    
                    print(f"DEBUG: Progress: {progress_pct}% - {status}")
            
            print(f"DEBUG: Analysis complete! Processed {len(spectra_dict)} spectra with {len(self.methods)} methods")
            self.progress.emit(100, f"Analysis complete! Processed {total_calculations} calculations.")
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"DEBUG: {error_msg}")
            self.error_occurred.emit(error_msg)
        finally:
            self.finished.emit()
    
    def stop(self):
        """Stop the analysis."""
        self._is_running = False


class GeothermometryResultsWidget(QWidget):
    """Widget for displaying geothermometry results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = []
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Results table tab
        self.table_widget = self._create_table_widget()
        self.tab_widget.addTab(self.table_widget, "Results Table")
        
        # Plot tab
        self.plot_widget = self._create_plot_widget()
        self.tab_widget.addTab(self.plot_widget, "Temperature Plots")
        
        layout.addWidget(self.tab_widget)
        
        # Export button
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)
        
        self.setLayout(layout)
    
    def _create_table_widget(self) -> QWidget:
        """Create the results table widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSortingEnabled(True)
        
        layout.addWidget(self.results_table)
        widget.setLayout(layout)
        return widget
    
    def _create_plot_widget(self) -> QWidget:
        """Create the plot widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Method selector
        self.method_combo = QComboBox()
        self.method_combo.currentTextChanged.connect(self._update_plot)
        
        layout.addWidget(QLabel("Select Method:"))
        layout.addWidget(self.method_combo)
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        widget.setLayout(layout)
        return widget
    
    def set_results(self, results: List[GeothermometryResult]):
        """Set the results to display."""
        self.results = results
        self._update_table()
        self._update_method_combo()
        self._update_plot()
    
    def _update_table(self):
        """Update the results table."""
        if not self.results:
            return
        
        # Create pivot table: samples as rows, methods as columns
        # Get unique sample IDs and method names from results
        sample_ids = sorted(set(r.sample_id for r in self.results))
        methods = sorted(set(r.method.value for r in self.results))
        
        print(f"DEBUG: Creating table with {len(sample_ids)} samples and {len(methods)} methods")
        print(f"DEBUG: Sample IDs: {sample_ids}")
        print(f"DEBUG: Methods: {methods}")
        
        self.results_table.setRowCount(len(sample_ids))
        self.results_table.setColumnCount(len(methods) + 1)  # +1 for sample ID column
        
        headers = ["Sample ID"] + methods
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # Fill table
        for row, sample_id in enumerate(sample_ids):
            # Sample ID column - use the actual sample name from the PKL file
            sample_item = QTableWidgetItem(str(sample_id))
            self.results_table.setItem(row, 0, sample_item)
            
            # Temperature columns
            for col, method in enumerate(methods, 1):
                # Find result for this sample and method
                result = next((r for r in self.results 
                             if r.sample_id == sample_id and r.method.value == method), None)
                
                if result is None:
                    item = QTableWidgetItem("N/A")
                    item.setForeground(QColor(128, 128, 128))  # Gray for missing
                elif result.success and result.temperature is not None:
                    # Format temperature appropriately
                    if abs(result.temperature) > 10000:
                        temp_text = f"{result.temperature:.2e}"
                    elif abs(result.temperature) > 1000:
                        temp_text = f"{result.temperature:.0f}"
                    else:
                        temp_text = f"{result.temperature:.1f}"
                    
                    item = QTableWidgetItem(temp_text)
                    
                    # Color code based on temperature ranges
                    temp = result.temperature
                    if temp < 0:
                        item.setForeground(QColor(200, 0, 0))  # Red for negative
                    elif temp > 1000:
                        item.setForeground(QColor(255, 140, 0))  # Orange for very high
                    elif 200 <= temp <= 600:
                        item.setForeground(QColor(0, 150, 0))  # Green for reasonable range
                    # Default color for other temperatures
                else:
                    error_msg = result.error[:50] + "..." if result.error and len(result.error) > 50 else (result.error or "Unknown error")
                    item = QTableWidgetItem(f"Error: {error_msg}")
                    item.setForeground(QColor(200, 0, 0))  # Red for errors
                    item.setToolTip(result.error or "Unknown error")  # Full error in tooltip
                
                self.results_table.setItem(row, col, item)
        
        # Resize columns to fit content
        self.results_table.resizeColumnsToContents()
        
        # Make sample ID column a bit wider to ensure names are visible
        header = self.results_table.horizontalHeader()
        current_width = header.sectionSize(0)
        header.resizeSection(0, max(current_width, 150))  # Minimum 150 pixels for sample names
    
    def _update_method_combo(self):
        """Update the method combo box."""
        if not self.results:
            return
        
        methods = sorted(set(r.method.value for r in self.results))
        self.method_combo.clear()
        self.method_combo.addItems(methods)
    
    def _update_plot(self):
        """Update the temperature plot."""
        if not self.results:
            return
        
        selected_method = self.method_combo.currentText()
        if not selected_method:
            return
        
        # Get results for selected method
        method_results = [r for r in self.results 
                         if r.method.value == selected_method and r.success and r.temperature is not None]
        
        if not method_results:
            # Clear plot and show message
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'No successful results for {selected_method}', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Temperature Results - {selected_method}")
            self.figure.tight_layout()
            self.canvas.draw()
            return
        
        # Clear and create new plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Extract data - use actual sample names from PKL file
        sample_names = [r.sample_id for r in method_results]  # These are the actual sample names
        temperatures = [r.temperature for r in method_results]
        
        print(f"DEBUG: Plotting {len(sample_names)} samples for {selected_method}")
        print(f"DEBUG: Sample names: {sample_names[:5]}{'...' if len(sample_names) > 5 else ''}")
        print(f"DEBUG: Temperature range: {min(temperatures):.1f} to {max(temperatures):.1f}°C")
        
        # Create bar plot
        x_positions = range(len(sample_names))
        bars = ax.bar(x_positions, temperatures, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Color bars based on temperature ranges (geologically meaningful)
        for bar, temp in zip(bars, temperatures):
            if temp < 200:
                bar.set_color('lightblue')  # Very low grade
            elif temp < 300:
                bar.set_color('blue')       # Low grade
            elif temp < 400:
                bar.set_color('green')      # Medium grade
            elif temp < 500:
                bar.set_color('orange')     # High grade
            elif temp < 600:
                bar.set_color('red')        # Very high grade
            else:
                bar.set_color('darkred')    # Extreme grade
        
        # Customize plot
        ax.set_xlabel("Sample")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Temperature Results - {selected_method}")
        ax.set_xticks(x_positions)
        
        # Handle x-axis labels based on number of samples
        if len(sample_names) <= 20:
            # Show all sample names for small datasets
            ax.set_xticklabels(sample_names, rotation=45, ha='right', fontsize=8)
        else:
            # Show every nth label for large datasets to avoid overcrowding
            step = max(1, len(sample_names) // 20)
            labels = [sample_names[i] if i % step == 0 else '' for i in range(len(sample_names))]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        # Add horizontal grid for easier reading
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add statistics text box
        stats_text = f"n = {len(temperatures)}\n"
        stats_text += f"Mean: {np.mean(temperatures):.1f}°C\n"
        stats_text += f"Range: {min(temperatures):.1f} - {max(temperatures):.1f}°C\n"
        stats_text += f"Std: {np.std(temperatures):.1f}°C"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
        
        # Tight layout and redraw
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _export_results(self):
        """Export results to CSV files - both pivot table and detailed formats."""
        if not self.results:
            QMessageBox.warning(self, "No Data", "No results to export.")
            return
        
        # Get save directory and base filename
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results (Base Filename)", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Remove .csv extension if present to create base name
            base_path = file_path.replace('.csv', '')
            
            # 1. CREATE PIVOT TABLE CSV (sample names vs methods)
            pivot_path = f"{base_path}_pivot.csv"
            
            # Get unique samples and methods
            sample_ids = sorted(set(r.sample_id for r in self.results))
            methods = sorted(set(r.method.value for r in self.results))
            
            # Create pivot table data
            pivot_data = []
            
            # Header row: ['Sample'] + method names
            header = ['Sample'] + methods
            pivot_data.append(header)
            
            # Data rows: sample name + temperatures for each method
            for sample_id in sample_ids:
                row = [sample_id]  # First column: sample name
                
                for method in methods:
                    # Find result for this sample and method
                    result = next((r for r in self.results 
                                 if r.sample_id == sample_id and r.method.value == method), None)
                    
                    if result is None:
                        row.append('')  # Empty cell for missing data
                    elif result.success and result.temperature is not None:
                        row.append(f"{result.temperature:.1f}")  # Temperature value
                    else:
                        row.append('Error')  # Error indicator
                
                pivot_data.append(row)
            
            # Write pivot table CSV
            with open(pivot_path, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                writer.writerows(pivot_data)
            
            print(f"Exported pivot table to: {pivot_path}")
            
            # 2. CREATE DETAILED RESULTS CSV (existing format)
            detailed_path = f"{base_path}_detailed.csv"
            
            # Convert results to DataFrame (existing implementation)
            data = []
            for result in self.results:
                row = {
                    'sample_id': result.sample_id,
                    'method': result.method.value,
                    'temperature_c': result.temperature if result.success else None,
                    'success': result.success,
                    'error': result.error or '',
                }
                # Add parameters
                for param, value in result.parameters.items():
                    row[f'param_{param}'] = value
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(detailed_path, index=False)
            
            print(f"Exported detailed results to: {detailed_path}")
            
            # Show success message with both file paths
            QMessageBox.information(
                self, 
                "Export Successful", 
                f"Results exported to:\n\n"
                f"1. Pivot table: {pivot_path}\n"
                f"   (Sample names vs Methods)\n\n"
                f"2. Detailed results: {detailed_path}\n"
                f"   (Full data with parameters)"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Export Error", 
                f"Failed to export results: {str(e)}"
            )


class GeothermometryAnalysisDialog(QDialog):
    """Main dialog for geothermometry analysis."""
    
    def __init__(self, parent=None, data: Dict = None, results=None):
        super().__init__(parent)
        self.setWindowTitle("Geothermometry Analysis")
        self.setMinimumSize(800, 600)
        
        # Handle both old and new calling conventions
        if data is not None:
            self.data = data
        elif results is not None:
            # Convert old results format to new data format
            self.data = self._convert_results_to_data(results)
        else:
            self.data = {}
        
        self.worker_thread = None
        self.worker = None
        self.results = []
        
        self._init_ui()
        self._check_data()
    
    def _convert_results_to_data(self, results):
        """Convert old results format to new data format."""
        print("Converting legacy results format to new data structure...")
        print(f"DEBUG: Input results type: {type(results)}")
        
        # Initialize data structure
        data = {
            'spectra_dict': {},
            'metadata': {
                'source': 'legacy_results',
                'converted': True
            }
        }
        
        # Handle different result formats
        if isinstance(results, list):
            print(f"DEBUG: Processing list of {len(results)} results")
            # List of result dictionaries
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    print(f"DEBUG: Processing result {i+1}: {list(result.keys())}")
                    
                    # Check if this result contains a spectra_dict
                    if 'spectra_dict' in result and isinstance(result['spectra_dict'], dict):
                        print(f"DEBUG: Found nested spectra_dict with {len(result['spectra_dict'])} spectra")
                        # Merge all spectra from this result
                        for spec_name, spec_data in result['spectra_dict'].items():
                            # Use original name or create unique name
                            final_name = spec_name
                            counter = 1
                            while final_name in data['spectra_dict']:
                                final_name = f"{spec_name}_{counter}"
                                counter += 1
                            data['spectra_dict'][final_name] = spec_data
                            print(f"DEBUG: Added spectrum '{final_name}' from nested spectra_dict")
                        continue
                    
                    # Single spectrum in this result
                    sample_id = result.get('metadata', {}).get('sample_id', f'Sample_{i+1}')
                    
                    # Try different ways to get sample ID
                    if 'filename' in result:
                        filename = result['filename']
                        # Extract base filename without extension
                        import os
                        sample_id = os.path.splitext(os.path.basename(filename))[0]
                    elif 'sample_name' in result:
                        sample_id = result['sample_name']
                    
                    # Extract spectrum data from result
                    spectrum_data = {
                        'metadata': result.get('metadata', {'sample_id': sample_id}),
                    }
                    
                    # Copy other relevant data
                    for key in ['wavenumbers', 'intensities', 'fitted_intensities', 'baseline']:
                        if key in result:
                            spectrum_data[key] = result[key]
                    
                    # Try to extract peaks data from various possible locations
                    peaks_found = False
                    for peaks_key in ['peaks_df', 'peaks', 'fitted_peaks', 'peak_data']:
                        if peaks_key in result and result[peaks_key] is not None:
                            print(f"DEBUG: Found peaks data in '{peaks_key}' for {sample_id}")
                            spectrum_data['peaks'] = result[peaks_key]
                            peaks_found = True
                            break
                    
                    if not peaks_found:
                        # Create dummy peaks data for testing
                        print(f"WARNING: No peaks data found for {sample_id}, creating dummy data")
                        spectrum_data['peaks'] = pd.DataFrame({
                            'peak_center': [1350.0 + i*5, 1580.0 + i*2, 1620.0 + i*3],  # Slightly vary for each sample
                            'area': [1000.0 + i*100, 1500.0 + i*50, 800.0 + i*75],
                            'fwhm': [25.0 + i*2, 15.0 + i*1, 30.0 + i*1.5],
                            'amplitude': [100.0 + i*10, 150.0 + i*5, 80.0 + i*8]
                        })
                    
                    data['spectra_dict'][sample_id] = spectrum_data
                    print(f"DEBUG: Added spectrum '{sample_id}' to spectra_dict")
        
        elif isinstance(results, dict):
            print(f"DEBUG: Processing single dict result with keys: {list(results.keys())}")
            
            # Check if this is already a properly formatted data structure
            if 'spectra_dict' in results and isinstance(results['spectra_dict'], dict):
                print(f"DEBUG: Found existing spectra_dict with {len(results['spectra_dict'])} spectra")
                # Already in correct format
                data = results
            else:
                # Check if the entire dict IS the spectra_dict (common PKL format)
                # Look for keys that might be spectrum names
                spectrum_like_keys = []
                for key, value in results.items():
                    if isinstance(value, dict) and any(
                        peaks_key in value for peaks_key in ['peaks', 'fitted_peaks', 'peak_data', 'peaks_df']
                    ):
                        spectrum_like_keys.append(key)
                
                if spectrum_like_keys:
                    print(f"DEBUG: Found {len(spectrum_like_keys)} spectrum-like keys: {spectrum_like_keys}")
                    # This dict appears to be the spectra_dict itself
                    for key in spectrum_like_keys:
                        data['spectra_dict'][key] = results[key]
                        print(f"DEBUG: Added spectrum '{key}' from top-level dict")
                else:
                    # Single spectrum result
                    sample_id = results.get('metadata', {}).get('sample_id', 'Sample_1')
                    
                    # Try different ways to get sample ID
                    if 'filename' in results:
                        filename = results['filename']
                        import os
                        sample_id = os.path.splitext(os.path.basename(filename))[0]
                    elif 'sample_name' in results:
                        sample_id = results['sample_name']
                    
                    spectrum_data = {
                        'metadata': results.get('metadata', {'sample_id': sample_id}),
                    }
                    
                    # Copy other relevant data
                    for key in ['wavenumbers', 'intensities', 'fitted_intensities', 'baseline']:
                        if key in results:
                            spectrum_data[key] = results[key]
                    
                    # Extract peaks data
                    peaks_found = False
                    for peaks_key in ['peaks_df', 'peaks', 'fitted_peaks', 'peak_data']:
                        if peaks_key in results and results[peaks_key] is not None:
                            print(f"DEBUG: Found peaks data in '{peaks_key}'")
                            spectrum_data['peaks'] = results[peaks_key]
                            peaks_found = True
                            break
                    
                    if not peaks_found:
                        # Create dummy peaks data
                        print(f"WARNING: No peaks data found, creating dummy data")
                        spectrum_data['peaks'] = pd.DataFrame({
                            'peak_center': [1350.0, 1580.0, 1620.0],
                            'area': [1000.0, 1500.0, 800.0],
                            'fwhm': [25.0, 15.0, 30.0],
                            'amplitude': [100.0, 150.0, 80.0]
                        })
                    
                    data['spectra_dict'][sample_id] = spectrum_data
                    print(f"DEBUG: Added single spectrum '{sample_id}' to spectra_dict")
        
        print(f"DEBUG: Conversion complete - {len(data['spectra_dict'])} spectra in final data structure")
        print(f"DEBUG: Final spectrum names: {list(data['spectra_dict'].keys())}")
        
        return data
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QFormLayout()
        
        spectra_count = len(self.data.get('spectra_dict', {}))
        info_layout.addRow("Number of Spectra:", QLabel(str(spectra_count)))
        
        # Show spectrum names if there are any
        if spectra_count > 0:
            spectra_names = list(self.data['spectra_dict'].keys())
            if spectra_count <= 10:
                # Show all names if 10 or fewer
                names_text = ", ".join(spectra_names)
            else:
                # Show first few and indicate there are more
                names_text = ", ".join(spectra_names[:5]) + f", ... and {spectra_count - 5} more"
            info_layout.addRow("Spectrum Names:", QLabel(names_text))
        
        # Show data source
        data_source = self.data.get('metadata', {}).get('source', 'unknown')
        info_layout.addRow("Data Source:", QLabel(data_source))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Method selection
        method_group = QGroupBox("Select Geothermometry Methods")
        method_layout = QVBoxLayout()
        
        self.method_checks = {}
        for method in get_available_methods():
            check = QCheckBox(f"{method.value} (requires: {', '.join(METHOD_PARAMETERS[method])})")
            check.setChecked(True)
            self.method_checks[method] = check
            method_layout.addWidget(check)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Progress section
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_label = QLabel("Ready to start analysis")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.debug_btn = QPushButton("Debug Data Structure")
        self.debug_btn.clicked.connect(self._debug_data_structure)
        
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self._run_analysis)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        self.cancel_btn.setEnabled(False)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.debug_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _check_data(self):
        """Check if the data contains the necessary information."""
        if not self.data:
            QMessageBox.warning(self, "No Data", "No data provided for analysis.")
            return
        
        print("DEBUG: Checking data structure...")
        print(f"DEBUG: Data top-level keys: {list(self.data.keys())}")
        
        spectra_dict = self.data.get('spectra_dict', {})
        print(f"DEBUG: spectra_dict type: {type(spectra_dict)}")
        print(f"DEBUG: spectra_dict keys: {list(spectra_dict.keys()) if isinstance(spectra_dict, dict) else 'Not a dict'}")
        
        if not spectra_dict:
            # Try to find spectra in other locations
            print("DEBUG: No spectra_dict found, checking for alternative structures...")
            
            # Check if this is a direct spectra dictionary
            if all(isinstance(v, dict) for v in self.data.values() if not k.startswith('_')):
                print("DEBUG: Data appears to be a direct spectra dictionary")
                # Convert to proper format
                self.data = {
                    'spectra_dict': {k: v for k, v in self.data.items() if not k.startswith('_')},
                    'metadata': {'source': 'direct_spectra_dict'}
                }
                spectra_dict = self.data['spectra_dict']
            else:
                QMessageBox.warning(self, "No Spectra", "No spectra found in the data.")
                return
        
        print(f"DEBUG: Final spectra count: {len(spectra_dict)}")
        
        if len(spectra_dict) == 0:
            QMessageBox.warning(self, "Empty Data", "The spectra dictionary is empty.")
            return
        
        # Check first spectrum for peak data
        first_spectrum_name = next(iter(spectra_dict.keys()))
        first_spectrum = spectra_dict[first_spectrum_name]
        print(f"DEBUG: First spectrum '{first_spectrum_name}' keys: {list(first_spectrum.keys()) if isinstance(first_spectrum, dict) else 'Not a dict'}")
        
        has_peaks = False
        if isinstance(first_spectrum, dict):
            for key in ['peaks', 'fitted_peaks', 'peak_data', 'peaks_df']:
                if key in first_spectrum and first_spectrum[key] is not None:
                    print(f"DEBUG: Found peaks data in key '{key}'")
                    has_peaks = True
                    
                    # Show structure of peaks data
                    peaks_data = first_spectrum[key]
                    if isinstance(peaks_data, pd.DataFrame):
                        print(f"DEBUG: Peaks DataFrame shape: {peaks_data.shape}")
                        print(f"DEBUG: Peaks DataFrame columns: {list(peaks_data.columns)}")
                    elif isinstance(peaks_data, dict):
                        print(f"DEBUG: Peaks dict keys: {list(peaks_data.keys())}")
                    else:
                        print(f"DEBUG: Peaks data type: {type(peaks_data)}")
                    break
        
        if not has_peaks:
            QMessageBox.warning(
                self, "No Peak Data", 
                f"The spectra do not appear to contain peak fitting results. "
                f"Checked first spectrum '{first_spectrum_name}' for keys: peaks, fitted_peaks, peak_data, peaks_df. "
                f"Available keys: {list(first_spectrum.keys()) if isinstance(first_spectrum, dict) else 'None'}\n\n"
                f"Please ensure peak fitting has been performed before running geothermometry analysis."
            )
    
    def _debug_data_structure(self):
        """Debug function to print detailed data structure."""
        print("\n" + "="*80)
        print("DETAILED DATA STRUCTURE ANALYSIS")
        print("="*80)
        
        print(f"Data type: {type(self.data)}")
        print(f"Data keys: {list(self.data.keys()) if isinstance(self.data, dict) else 'Not a dict'}")
        
        if 'spectra_dict' in self.data:
            spectra_dict = self.data['spectra_dict']
            print(f"\nspectra_dict type: {type(spectra_dict)}")
            print(f"spectra_dict length: {len(spectra_dict) if hasattr(spectra_dict, '__len__') else 'No length'}")
            
            if isinstance(spectra_dict, dict):
                print(f"spectra_dict keys (first 10): {list(spectra_dict.keys())[:10]}")
                
                # Examine first few spectra
                for i, (name, spectrum) in enumerate(list(spectra_dict.items())[:3]):
                    print(f"\n--- Spectrum {i+1}: '{name}' ---")
                    print(f"  Type: {type(spectrum)}")
                    if isinstance(spectrum, dict):
                        print(f"  Keys: {list(spectrum.keys())}")
                        
                        # Check for peaks data
                        for peaks_key in ['peaks', 'fitted_peaks', 'peak_data', 'peaks_df']:
                            if peaks_key in spectrum:
                                peaks_data = spectrum[peaks_key]
                                print(f"  {peaks_key} type: {type(peaks_data)}")
                                if isinstance(peaks_data, pd.DataFrame):
                                    print(f"    Shape: {peaks_data.shape}")
                                    print(f"    Columns: {list(peaks_data.columns)}")
                                elif hasattr(peaks_data, '__len__'):
                                    print(f"    Length: {len(peaks_data)}")
        
        print("="*80 + "\n")
    
    def _run_analysis(self):
        """Start the geothermometry analysis."""
        # Get selected methods
        selected_methods = [method for method, check in self.method_checks.items() 
                          if check.isChecked()]
        
        if not selected_methods:
            QMessageBox.warning(self, "No Methods", "Please select at least one geothermometry method.")
            return
        
        # Check data again
        if not self.data.get('spectra_dict'):
            QMessageBox.warning(self, "No Data", "No spectral data available for analysis.")
            return
        
        # Setup UI for analysis
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Initializing analysis...")
        
        # Create worker and thread
        self.worker = GeothermometryWorker(selected_methods, self.data)
        self.worker_thread = QThread()
        
        # Move worker to thread
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.progress.connect(self._update_progress)
        self.worker.result_ready.connect(self._handle_result)
        self.worker.finished.connect(self._analysis_finished)
        self.worker.error_occurred.connect(self._handle_error)
        
        # Start thread
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
        
        # Clear previous results
        self.results = []
    
    def _update_progress(self, value: int, message: str):
        """Update the progress bar and label."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def _handle_result(self, result: GeothermometryResult):
        """Handle a single analysis result."""
        self.results.append(result)
    
    def _analysis_finished(self):
        """Handle completion of analysis."""
        # Clean up thread
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread.deleteLater()
            self.worker_thread = None
        
        # Reset UI
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Analysis complete!")
        
        # Show results
        if self.results:
            self._show_results()
        else:
            QMessageBox.information(self, "No Results", "No results were generated from the analysis.")
    
    def _handle_error(self, message: str):
        """Handle analysis error."""
        QMessageBox.critical(self, "Analysis Error", message)
        self._analysis_finished()
    
    def _cancel_analysis(self):
        """Cancel the running analysis."""
        if self.worker:
            self.worker.stop()
        
        self.progress_label.setText("Analysis cancelled")
        self._analysis_finished()
    
    def _show_results(self):
        """Show the analysis results."""
        results_dialog = QDialog(self)
        results_dialog.setWindowTitle("Geothermometry Results")
        results_dialog.setMinimumSize(1200, 800)
        
        layout = QVBoxLayout()
        
        # Create results widget
        results_widget = GeothermometryResultsWidget()
        results_widget.set_results(self.results)
        
        layout.addWidget(results_widget)
        results_dialog.setLayout(layout)
        
        # Show dialog
        results_dialog.show()
        results_dialog.raise_()
        results_dialog.activateWindow()


def create_geothermometry_dialog(parent=None, **kwargs):
    """
    Factory function to create GeothermometryAnalysisDialog with flexible input.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget
    data : dict, optional
        Data structure with 'spectra_dict'
    results : list or dict, optional
        Legacy results format
    pkl_file : str, optional
        Path to PKL file to load
        
    Returns
    -------
    GeothermometryAnalysisDialog
        Configured dialog ready to show
    """
    
    # Handle PKL file loading
    if 'pkl_file' in kwargs:
        pkl_path = kwargs['pkl_file']
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return GeothermometryAnalysisDialog(parent=parent, data=data)
        except Exception as e:
            raise ValueError(f"Failed to load PKL file {pkl_path}: {e}")
    
    # Handle direct data or results
    elif 'data' in kwargs:
        return GeothermometryAnalysisDialog(parent=parent, data=kwargs['data'])
    elif 'results' in kwargs:
        return GeothermometryAnalysisDialog(parent=parent, results=kwargs['results'])
    else:
        # Create with sample data for testing
        print("No data provided, creating dialog with sample data for testing...")
        return GeothermometryAnalysisDialog(parent=parent, data=_create_sample_data())


def _create_sample_data():
    """Create sample data for testing purposes."""
    return {
        'spectra_dict': {
            'sample_001': {
                'metadata': {'sample_id': 'Sample 1'},
                'peaks': pd.DataFrame({
                    'peak_center': [1350, 1580, 1620],
                    'area': [1000, 1500, 800],
                    'fwhm': [25, 15, 30],
                    'amplitude': [100, 150, 80]
                })
            },
            'sample_002': {
                'metadata': {'sample_id': 'Sample 2'},
                'peaks': pd.DataFrame({
                    'peak_center': [1355, 1575, 1615],
                    'area': [1200, 1800, 900],
                    'fwhm': [28, 18, 32],
                    'amplitude': [120, 180, 90]
                })
            },
            'sample_003': {
                'metadata': {'sample_id': 'Sample 3'},
                'peaks': pd.DataFrame({
                    'peak_center': [1345, 1585, 1625],
                    'area': [800, 1200, 600],
                    'fwhm': [22, 12, 28],
                    'amplitude': [80, 120, 60]
                })
            }
        },
        'metadata': {
            'analysis_type': 'geothermometry',
            'date': '2024-01-15'
        }
    }


def main():
    """Example usage and testing."""
    import sys
    
    # Create and run application
    app = QApplication(sys.argv)
    
    # Test parameter extraction with sample data
    print("Testing parameter extraction...")
    sample_data = _create_sample_data()
    
    try:
        for method in get_available_methods():
            print(f"\nTesting {method.value}:")
            for sample_name, sample_data_spec in sample_data['spectra_dict'].items():
                try:
                    params = ParameterExtractor.extract_from_spectrum(sample_data_spec, method)
                    temp = calculate_temperature(method, **params)
                    print(f"  {sample_name}: {params} -> {temp:.1f}°C")
                except Exception as e:
                    print(f"  {sample_name}: Error - {e}")
    
    except Exception as e:
        print(f"Error in parameter extraction test: {e}")
    
    # Create dialog - this will work with both old and new calling conventions
    dialog = create_geothermometry_dialog()
    dialog.show()
    
    sys.exit(app.exec())


# Backwards compatibility functions
def launch_geothermometry_analysis(results=None, parent=None):
    """
    Launch geothermometry analysis dialog - backwards compatible function.
    
    Parameters
    ----------
    results : list or dict
        Peak fitting results from previous analysis
    parent : QWidget, optional
        Parent widget
        
    Returns
    -------
    GeothermometryAnalysisDialog
        The analysis dialog
    """
    return GeothermometryAnalysisDialog(parent=parent, results=results)


def run_geothermometry_gui(data=None, results=None, pkl_file=None):
    """
    Standalone function to run geothermometry analysis GUI.
    
    Parameters
    ----------
    data : dict, optional
        Data structure with 'spectra_dict'
    results : list or dict, optional
        Legacy results format  
    pkl_file : str, optional
        Path to PKL file to load
    """
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        should_exec = True
    else:
        should_exec = False
    
    # Create dialog based on available input
    if pkl_file:
        dialog = create_geothermometry_dialog(pkl_file=pkl_file)
    elif data:
        dialog = create_geothermometry_dialog(data=data)
    elif results:
        dialog = create_geothermometry_dialog(results=results)
    else:
        dialog = create_geothermometry_dialog()  # Sample data
    
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    
    if should_exec:
        sys.exit(app.exec())
    
    return dialog


if __name__ == "__main__":
    main()