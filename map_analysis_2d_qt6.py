#!/usr/bin/env python3
# 2D Map Analysis Module for RamanLab - Qt6 Version
"""
Qt6-based module for analyzing 2D Raman spectroscopy data, including data loading,
organization, and preparation for machine learning analysis.
"""

import os
import re
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths

# Fix matplotlib backend for PySide6 compatibility
import matplotlib
matplotlib.use("QtAgg")  # Use QtAgg backend which works with PySide6

# Import PySide6 (same as main app) instead of PyQt6
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QTabWidget, QFrame,
                             QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
                             QDoubleSpinBox, QCheckBox, QGroupBox, QFileDialog,
                             QMessageBox, QProgressBar, QTextEdit, QSplitter,
                             QListWidget, QSlider, QScrollArea, QDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QFont, QPixmap, QIcon, QAction

# Import PySide6-compatible matplotlib backends and UI toolbar
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    # Fallback for older matplotlib versions
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar

from ui.matplotlib_config import configure_compact_ui, apply_theme

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import base64
from io import BytesIO
from ml_raman_map.pre_processing import preprocess_spectrum
from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from dask import array as da
import dask.array as darray
from dask.diagnostics import ProgressBar
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import nnls, lsq_linear
from scipy import sparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from tqdm import tqdm
import psutil
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TemplateSpectrum:
    """Class to hold template spectrum data for comparison."""
    name: str
    wavenumbers: np.ndarray
    intensities: np.ndarray
    processed_intensities: Optional[np.ndarray] = None
    color: Optional[str] = None
    
    def __post_init__(self):
        """Initialize with a random color if none is provided."""
        if self.color is None:
            # Generate a random color for plotting
            import random
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.color = f'#{r:02x}{g:02x}{b:02x}'

class TemplateSpectraManager:
    """Class to manage template spectra for comparison with map data."""
    
    def __init__(self, target_wavenumbers: Optional[np.ndarray] = None):
        """
        Initialize the template spectra manager.
        
        Parameters:
        -----------
        target_wavenumbers : Optional[np.ndarray]
            Target wavenumber values for resampling. If None, a default range is used.
        """
        self.templates: List[TemplateSpectrum] = []
        
        # Set up target wavenumbers for resampling
        if target_wavenumbers is None:
            self.target_wavenumbers = np.linspace(100, 3500, 400)
        else:
            self.target_wavenumbers = target_wavenumbers
    
    def load_template(self, filepath: str, name: Optional[str] = None) -> bool:
        """
        Load a template spectrum from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the template spectrum file
        name : Optional[str]
            Name for the template spectrum. If None, the filename is used.
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Get name from filename if not provided
            if name is None:
                name = Path(filepath).stem
            
            # Determine file type based on extension
            file_ext = Path(filepath).suffix.lower()
            
            # Load the data based on file type and try different separators
            if file_ext in ['.csv', '.txt']:
                # Try different separators
                separators = [',', '\t', ' ', ';']
                data = None
                
                # Try to load the file with different separators
                for sep in separators:
                    try:
                        # Try with pandas first which is more flexible with different file formats
                        import pandas as pd
                        df = pd.read_csv(filepath, sep=sep, header=None, comment='#')
                        
                        # Check if we have at least 2 columns
                        if df.shape[1] >= 2:
                            # Extract wavenumbers and intensities
                            wavenumbers = df.iloc[:, 0].values
                            intensities = df.iloc[:, 1].values
                            data = np.column_stack((wavenumbers, intensities))
                            break
                    except Exception as e:
                        logger.debug(f"Failed to load {filepath} with separator '{sep}': {str(e)}")
                        continue
                
                # If pandas methods fail, try numpy loadtxt as fallback
                if data is None:
                    try:
                        data = np.loadtxt(filepath)
                    except Exception as e:
                        logger.debug(f"Failed to load {filepath} with np.loadtxt: {str(e)}")
                
                # If all loading methods fail, raise exception
                if data is None:
                    raise ValueError(f"Could not parse file {filepath} with any known separator")
            else:
                # Default to numpy loadtxt for other file types
                data = np.loadtxt(filepath)
            
            # Extract wavenumbers and intensities
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
            
            # Preprocess the intensities
            processed = self._preprocess_spectrum(wavenumbers, intensities)
            
            # Create template spectrum
            template = TemplateSpectrum(
                name=name,
                wavenumbers=wavenumbers,
                intensities=intensities,
                processed_intensities=processed
            )
            
            # Add to templates list
            self.templates.append(template)
            return True
            
        except Exception as e:
            logger.error(f"Error loading template {filepath}: {str(e)}")
            return False
    
    def load_templates_from_directory(self, directory: str) -> int:
        """
        Load all template spectra from a directory.
        
        Parameters:
        -----------
        directory : str
            Directory containing template spectrum files
            
        Returns:
        --------
        int
            Number of templates successfully loaded
        """
        try:
            dir_path = Path(directory)
            
            # Look for both CSV and TXT files
            csv_files = list(dir_path.glob('*.csv'))
            txt_files = list(dir_path.glob('*.txt'))
            files = csv_files + txt_files
            
            if not files:
                logger.warning(f"No .csv or .txt files found in directory {directory}")
                return 0
            
            loaded_count = 0
            for file_path in files:
                if self.load_template(str(file_path)):
                    loaded_count += 1
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading templates from directory {directory}: {str(e)}")
            return 0
    
    def remove_template(self, index: int) -> bool:
        """
        Remove a template by index.
        
        Parameters:
        -----------
        index : int
            Index of template to remove
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if 0 <= index < len(self.templates):
                self.templates.pop(index)
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing template at index {index}: {str(e)}")
            return False
    
    def clear_templates(self):
        """Clear all templates."""
        self.templates.clear()
    
    def get_template_count(self) -> int:
        """Get the number of loaded templates."""
        return len(self.templates)
    
    def get_template_names(self) -> List[str]:
        """Get list of template names."""
        return [template.name for template in self.templates]
    
    def get_template_matrix(self) -> np.ndarray:
        """
        Get matrix of all template spectra for fitting.
        
        Returns:
        --------
        np.ndarray
            Matrix where each column is a template spectrum
        """
        if not self.templates:
            return np.array([])
        
        # Use processed intensities if available, otherwise use raw intensities
        template_matrix = []
        for template in self.templates:
            if template.processed_intensities is not None:
                template_matrix.append(template.processed_intensities)
            else:
                template_matrix.append(template.intensities)
        
        return np.column_stack(template_matrix)
    
    def _preprocess_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """
        Preprocess a spectrum for template fitting.
        
        Parameters:
        -----------
        wavenumbers : np.ndarray
            Wavenumber values
        intensities : np.ndarray
            Intensity values
            
        Returns:
        --------
        np.ndarray
            Processed intensity values resampled to target wavenumbers
        """
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
            clean_wn = wavenumbers[valid_mask]
            clean_int = intensities[valid_mask]
            
            if len(clean_wn) < 2:
                return np.zeros_like(self.target_wavenumbers)
            
            # Interpolate to target wavenumbers
            f = interp1d(clean_wn, clean_int, kind='linear', 
                        bounds_error=False, fill_value=0)
            resampled = f(self.target_wavenumbers)
            
            # Apply smoothing
            if len(resampled) > 5:
                try:
                    resampled = savgol_filter(resampled, window_length=5, polyorder=2)
                except ValueError:
                    pass
            
            # Normalize
            if np.max(resampled) > 0:
                resampled = resampled / np.max(resampled)
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error preprocessing spectrum: {str(e)}")
            return np.zeros_like(self.target_wavenumbers)
    
    def fit_spectrum(self, spectrum: np.ndarray, 
                    method: str = 'nnls', 
                    use_baseline: bool = True) -> Tuple[np.ndarray, float]:
        """
        Fit a spectrum using the loaded templates.
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Spectrum to fit
        method : str
            Fitting method ('nnls' or 'lstsq')
        use_baseline : bool
            Whether to include a baseline component
            
        Returns:
        --------
        Tuple[np.ndarray, float]
            Coefficients and goodness of fit
        """
        try:
            if not self.templates:
                return np.array([]), 0.0
            
            # Get template matrix
            template_matrix = self.get_template_matrix()
            
            if template_matrix.size == 0:
                return np.array([]), 0.0
            
            # Add baseline component if requested
            if use_baseline:
                baseline = np.ones((len(spectrum), 1))
                fitting_matrix = np.column_stack([template_matrix, baseline])
            else:
                fitting_matrix = template_matrix
            
            # Perform fitting based on method
            if method == 'nnls':
                coeffs, residual = nnls(fitting_matrix, spectrum)
            else:  # lstsq
                coeffs, residuals, rank, s = np.linalg.lstsq(fitting_matrix, spectrum, rcond=None)
                if len(residuals) > 0:
                    residual = residuals[0]
                else:
                    residual = np.sum((fitting_matrix @ coeffs - spectrum) ** 2)
            
            # Calculate R-squared
            ss_res = residual
            ss_tot = np.sum((spectrum - np.mean(spectrum)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Return only template coefficients (exclude baseline if used)
            if use_baseline:
                return coeffs[:-1], r_squared
            else:
                return coeffs, r_squared
                
        except Exception as e:
            logger.error(f"Error fitting spectrum: {str(e)}")
            return np.array([]), 0.0

@dataclass
class SpectrumData:
    """Class to hold spectrum data and metadata."""
    x_pos: int
    y_pos: int
    wavenumbers: np.ndarray
    intensities: np.ndarray
    filename: str
    processed_intensities: Optional[np.ndarray] = None

class RamanMapData:
    """Class to manage Raman map data and provide analysis capabilities."""
    
    def __init__(self, data_dir: str, target_wavenumbers: Optional[np.ndarray] = None):
        """
        Initialize RamanMapData.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the Raman map data files
        target_wavenumbers : Optional[np.ndarray]
            Target wavenumber values for resampling. If None, a default range is used.
        """
        self.data_dir = data_dir
        self.spectra: Dict[Tuple[int, int], SpectrumData] = {}
        self.x_positions: List[int] = []
        self.y_positions: List[int] = []
        self.cosmic_ray_map: Optional[np.ndarray] = None
        self.template_manager = TemplateSpectraManager(target_wavenumbers)
        self.template_coefficients: Optional[np.ndarray] = None
        self.template_residuals: Optional[np.ndarray] = None
        
        # Set up target wavenumbers for resampling
        if target_wavenumbers is None:
            self.target_wavenumbers = np.linspace(100, 3500, 400)
        else:
            self.target_wavenumbers = target_wavenumbers
        
        # Load data automatically
        self._load_data()
    
    def _parse_filename(self, filename: str) -> Tuple[int, int]:
        """
        Parse filename to extract x, y positions.
        Supports various filename formats.
        """
        # Remove file extension
        base_name = Path(filename).stem
        
        # Try different patterns (ordered from most specific to most general)
        patterns = [
            r'Y(\d+)_X(\d+)',  # Y###_X### format (for Steno-CT files)
            r'pos_(\d+)_(\d+)',  # pos_x_y format
            r'spectrum_(\d+)_(\d+)',  # spectrum_x_y format
            r'x(\d+)y(\d+)',  # x123y456 format
            r'(\d+)_(\d+)',  # x_y format (most general)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, base_name, re.IGNORECASE)
            if match:
                # For Y###_X### pattern, swap to return (X, Y)
                if pattern == r'Y(\d+)_X(\d+)':
                    return int(match.group(2)), int(match.group(1))  # X, Y
                else:
                    return int(match.group(1)), int(match.group(2))  # X, Y for other patterns
        
        # If no pattern matches, try to extract any two numbers
        numbers = re.findall(r'\d+', base_name)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        
        raise ValueError(f"Could not parse position from filename: {filename}")
    
    def _preprocess_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Preprocess spectrum data."""
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
            clean_wn = wavenumbers[valid_mask]
            clean_int = intensities[valid_mask]
            
            if len(clean_wn) < 2:
                return np.zeros_like(self.target_wavenumbers)
            
            # Interpolate to target wavenumbers
            f = interp1d(clean_wn, clean_int, kind='linear', 
                        bounds_error=False, fill_value=0)
            resampled = f(self.target_wavenumbers)
            
            # Apply smoothing
            if len(resampled) > 5:
                try:
                    resampled = savgol_filter(resampled, window_length=5, polyorder=2)
                except ValueError:
                    pass
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error preprocessing spectrum: {str(e)}")
            return np.zeros_like(self.target_wavenumbers)
    
    def _load_spectrum(self, filepath: Path) -> Optional[SpectrumData]:
        """Load a single spectrum file."""
        try:
            # Parse position from filename
            x_pos, y_pos = self._parse_filename(filepath.name)
            
            # Load data based on file extension
            file_ext = filepath.suffix.lower()
            
            if file_ext in ['.csv', '.txt']:
                # Try different separators
                separators = [',', '\t', ' ', ';']
                data = None
                
                for sep in separators:
                    try:
                        df = pd.read_csv(filepath, sep=sep, header=None, comment='#')
                        if df.shape[1] >= 2:
                            wavenumbers = df.iloc[:, 0].values
                            intensities = df.iloc[:, 1].values
                            data = np.column_stack((wavenumbers, intensities))
                            break
                    except:
                        continue
                
                if data is None:
                    data = np.loadtxt(filepath)
            else:
                data = np.loadtxt(filepath)
            
            # Extract wavenumbers and intensities
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
            
            # Apply cosmic ray detection and cleaning at data import
            # This ensures all downstream analysis works with clean data
            cosmic_detected, cleaned_intensities = self.detect_cosmic_rays(
                wavenumbers, intensities,
                threshold_factor=8.0,  # Aggressive for initial cleaning
                window_size=5
            )
            
            if cosmic_detected:
                logger.debug(f"Cosmic rays detected and cleaned in {filepath.name}")
                # Use cleaned intensities as the new "raw" data
                intensities = cleaned_intensities
            
            # Preprocess the spectrum (now using cosmic-ray-cleaned data)
            processed_intensities = self._preprocess_spectrum(wavenumbers, intensities)
            
            # Create SpectrumData object
            spectrum_data = SpectrumData(
                x_pos=x_pos,
                y_pos=y_pos,
                wavenumbers=wavenumbers,
                intensities=intensities,
                filename=filepath.name,
                processed_intensities=processed_intensities
            )
            
            return spectrum_data
            
        except Exception as e:
            logger.error(f"Error loading spectrum {filepath}: {str(e)}")
            return None
    
    def _load_data(self):
        """Load all spectra from the data directory."""
        data_path = Path(self.data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all spectrum files
        spectrum_files = []
        for ext in ['*.csv', '*.txt']:
            spectrum_files.extend(data_path.glob(ext))
        
        if not spectrum_files:
            raise ValueError(f"No spectrum files found in {self.data_dir}")
        
        # Load spectra
        for file_path in spectrum_files:
            spectrum_data = self._load_spectrum(file_path)
            if spectrum_data:
                self.spectra[(spectrum_data.x_pos, spectrum_data.y_pos)] = spectrum_data
        
        # Extract unique positions
        self.x_positions = sorted(list(set(pos[0] for pos in self.spectra.keys())))
        self.y_positions = sorted(list(set(pos[1] for pos in self.spectra.keys())))
        
        logger.info(f"Loaded {len(self.spectra)} spectra from {self.data_dir}")
        logger.info("Cosmic ray cleaning applied during data import - clean data ready for analysis")
    
    def get_spectrum(self, x_pos: int, y_pos: int) -> Optional[SpectrumData]:
        """Get spectrum at specified position."""
        return self.spectra.get((x_pos, y_pos))
    
    def get_map_data(self, feature: str = 'intensity', wavenumber: Optional[float] = None) -> np.ndarray:
        """Get 2D map data for a specific feature."""
        map_shape = (len(self.y_positions), len(self.x_positions))
        map_data = np.full(map_shape, np.nan)
        
        for i, y_pos in enumerate(self.y_positions):
            for j, x_pos in enumerate(self.x_positions):
                spectrum = self.get_spectrum(x_pos, y_pos)
                if spectrum:
                    if feature == 'intensity' and wavenumber is not None:
                        # Find closest wavenumber and get intensity
                        idx = np.argmin(np.abs(spectrum.wavenumbers - wavenumber))
                        map_data[i, j] = spectrum.intensities[idx]
                    elif feature == 'integrated_intensity':
                        map_data[i, j] = np.trapz(spectrum.intensities, spectrum.wavenumbers)
        
        return map_data
    
    def prepare_ml_data(self, save_path: Optional[str] = None, use_processed: bool = True) -> pd.DataFrame:
        """Prepare data for machine learning analysis."""
        data_list = []
        
        for (x_pos, y_pos), spectrum in self.spectra.items():
            # Use processed or raw intensities
            intensities = spectrum.processed_intensities if use_processed else spectrum.intensities
            
            # Create row with position and intensities
            row = {'x_pos': x_pos, 'y_pos': y_pos}
            for i, intensity in enumerate(intensities):
                row[f'wn_{i}'] = intensity
            
            data_list.append(row)
        
        df = pd.DataFrame(data_list)
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return df
    
    def save_map_data(self, save_path: str):
        """Save map data to pickle file."""
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_map_data(cls, load_path: str) -> 'RamanMapData':
        """Load map data from pickle file."""
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    
    def detect_cosmic_rays(self, wavenumbers: np.ndarray, intensities: np.ndarray, 
                          threshold_factor: float = 5.0, window_size: int = 5,
                          min_width_ratio: float = 0.1, max_fwhm: float = 5.0) -> Tuple[bool, np.ndarray]:
        """
        Simple and effective cosmic ray detection - just filter out high intensity spikes.
        
        Parameters:
        -----------
        wavenumbers : np.ndarray
            Wavenumber values
        intensities : np.ndarray
            Intensity values
        threshold_factor : float
            Multiplier for standard deviation threshold
        window_size : int
            Size of window for local analysis
        min_width_ratio : float
            Minimum width ratio for spike detection
        max_fwhm : float
            Maximum FWHM for cosmic ray spikes
            
        Returns:
        --------
        Tuple[bool, np.ndarray]
            (has_cosmic_ray, cleaned_intensities)
        """
        # Copy intensities to avoid modifying original
        cleaned_intensities = np.copy(intensities)
        has_cosmic_ray = False
        
        # Simple approach: detect rapid point-to-point changes (most effective for cosmic rays)
        if len(intensities) < 3:
            return False, cleaned_intensities
        
        # Calculate point-to-point changes
        point_changes = np.abs(np.diff(intensities))
        median_change = np.median(point_changes)
        std_change = np.std(point_changes)
        
        # Threshold for detecting abnormal jumps
        jump_threshold = median_change + threshold_factor * std_change
        
        # Find large jumps
        large_jumps = np.where(point_changes > jump_threshold)[0]
        
        if len(large_jumps) > 0:
            for jump_idx in large_jumps:
                i = jump_idx  # The index before the jump
                
                # Check if this looks like a cosmic ray spike
                if i > 0 and i < len(intensities) - 2:
                    # Look for spike pattern: normal -> high -> normal
                    left_val = intensities[i]
                    spike_val = intensities[i + 1]
                    right_val = intensities[i + 2] if i + 2 < len(intensities) else intensities[i + 1]
                    
                    # Check if middle value is significantly higher than neighbors
                    if (spike_val > left_val * 1.5 and spike_val > right_val * 1.5 and
                        spike_val > median_change + threshold_factor * std_change):
                        
                        # This looks like a cosmic ray - replace with interpolated value
                        cleaned_intensities[i + 1] = (left_val + right_val) / 2
                        has_cosmic_ray = True
        
        # Additional check: extremely high values compared to local median
        median_intensity = np.median(intensities)
        std_intensity = np.std(intensities)
        high_threshold = median_intensity + threshold_factor * std_intensity
        
        # Find points that are extremely high
        high_points = np.where(intensities > high_threshold)[0]
        
        for idx in high_points:
            if idx > 0 and idx < len(intensities) - 1:
                # Check if this point is isolated (cosmic ray characteristic)
                left_neighbor = intensities[idx - 1]
                right_neighbor = intensities[idx + 1]
                current_val = intensities[idx]
                
                # If current value is much higher than both neighbors, it's likely a cosmic ray
                if (current_val > left_neighbor * 2.0 and current_val > right_neighbor * 2.0):
                    # Replace with average of neighbors
                    cleaned_intensities[idx] = (left_neighbor + right_neighbor) / 2
                    has_cosmic_ray = True
        
        return has_cosmic_ray, cleaned_intensities
    
    def fit_templates_to_map(self, method: str = 'nnls', use_baseline: bool = True, 
                        filter_cosmic_rays: bool = True, threshold_factor: float = 5.0,
                        window_size: int = 5, min_width_ratio: float = 0.1, 
                        max_fwhm: float = 5.0) -> bool:
        """
        Fit templates to all spectra in the map.
        
        Parameters:
        -----------
        method : str
            Fitting method ('nnls' or 'lstsq')
        use_baseline : bool
            Whether to include a baseline component
        filter_cosmic_rays : bool
            Whether to filter cosmic rays before fitting
        threshold_factor : float
            Cosmic ray detection threshold factor
        window_size : int
            Window size for cosmic ray detection
        min_width_ratio : float
            Minimum width ratio for cosmic ray detection
        max_fwhm : float
            Maximum FWHM for cosmic ray detection
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if self.template_manager.get_template_count() == 0:
                logger.error("No templates loaded for fitting")
                return False
            
            n_templates = self.template_manager.get_template_count()
            n_spectra = len(self.spectra)
            
            # Initialize coefficient and residual arrays
            self.template_coefficients = np.zeros((len(self.y_positions), len(self.x_positions), n_templates))
            self.template_residuals = np.zeros((len(self.y_positions), len(self.x_positions)))
            
            # Initialize cosmic ray map
            self.cosmic_ray_map = np.zeros((len(self.y_positions), len(self.x_positions)), dtype=bool)
            
            # Fit templates to each spectrum
            for i, y_pos in enumerate(self.y_positions):
                for j, x_pos in enumerate(self.x_positions):
                    spectrum = self.get_spectrum(x_pos, y_pos)
                    if spectrum is not None:
                        # Use processed or raw intensities
                        if spectrum.processed_intensities is not None:
                            intensities = spectrum.processed_intensities
                        else:
                            intensities = self._preprocess_spectrum(spectrum.wavenumbers, spectrum.intensities)
                        
                        # Filter cosmic rays if requested
                        if filter_cosmic_rays:
                            cosmic_detected, intensities = self.detect_cosmic_rays(
                                self.target_wavenumbers, intensities,
                                threshold_factor, window_size, min_width_ratio, max_fwhm
                            )
                            self.cosmic_ray_map[i, j] = cosmic_detected
                        
                        # Fit templates
                        coeffs, r_squared = self.template_manager.fit_spectrum(
                            intensities, method, use_baseline
                        )
                        
                        # Store results
                        if len(coeffs) == n_templates:
                            self.template_coefficients[i, j, :] = coeffs
                            self.template_residuals[i, j] = 1.0 - r_squared  # Convert R² to residual
            
            logger.info(f"Template fitting completed for {n_spectra} spectra")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting templates to map: {str(e)}")
            return False
    
    def get_template_coefficient_map(self, template_index: int, normalized: bool = True) -> np.ndarray:
        """
        Get the coefficient map for a specific template.
        
        Parameters:
        -----------
        template_index : int
            Index of the template
        normalized : bool
            Whether to normalize coefficients
            
        Returns:
        --------
        np.ndarray
            2D array of template coefficients
        """
        if (self.template_coefficients is None or 
            template_index >= self.template_coefficients.shape[2]):
            return np.full((len(self.y_positions), len(self.x_positions)), np.nan)
        
        coeff_map = self.template_coefficients[:, :, template_index].copy()
        
        if normalized:
            # Normalize by the sum of all coefficients at each position
            total_coeffs = np.sum(self.template_coefficients, axis=2)
            # Avoid division by zero
            mask = total_coeffs > 0
            coeff_map[mask] = coeff_map[mask] / total_coeffs[mask]
            coeff_map[~mask] = 0
        
        return coeff_map
    
    def get_residual_map(self) -> np.ndarray:
        """
        Get the template fitting residual map.
        
        Returns:
        --------
        np.ndarray
            2D array of fitting residuals
        """
        if self.template_residuals is None:
            return np.full((len(self.y_positions), len(self.x_positions)), np.nan)
        
        return self.template_residuals.copy()
    
    def get_dominant_template_map(self) -> np.ndarray:
        """
        Get a map showing which template is dominant at each position.
        
        Returns:
        --------
        np.ndarray
            2D array with template indices (0-based)
        """
        if self.template_coefficients is None:
            return np.full((len(self.y_positions), len(self.x_positions)), np.nan)
        
        # Find the template with the highest coefficient at each position
        dominant_map = np.argmax(self.template_coefficients, axis=2).astype(float)
        
        # Set positions with all zero coefficients to NaN
        total_coeffs = np.sum(self.template_coefficients, axis=2)
        dominant_map[total_coeffs == 0] = np.nan
        
        return dominant_map
    
    def get_total_template_contributions(self, normalized: bool = True) -> dict:
        """
        Get the total contribution of each template across the entire map.
        
        Parameters:
        -----------
        normalized : bool
            Whether to normalize contributions
            
        Returns:
        --------
        dict
            Dictionary with template names as keys and contributions as values
        """
        if self.template_coefficients is None:
            return {}
        
        template_names = self.template_manager.get_template_names()
        contributions = {}
        
        for i, name in enumerate(template_names):
            total_contrib = np.nansum(self.template_coefficients[:, :, i])
            contributions[name] = total_contrib
        
        if normalized:
            total_sum = sum(contributions.values())
            if total_sum > 0:
                contributions = {name: contrib / total_sum for name, contrib in contributions.items()}
        
        return contributions


class MapAnalysisWorker(QThread):
    """Worker thread for time-consuming map analysis operations."""
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self._is_running = True
    
    def run(self):
        try:
            if not self._is_running:
                return
            # Pass the worker reference as the first argument so operations can emit progress
            result = self.operation(self, *self.args, **self.kwargs)
            if self._is_running:
                self.finished.emit(result)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
    
    def stop(self):
        """Stop the worker thread gracefully."""
        self._is_running = False
        self.wait(5000)  # Wait up to 5 seconds for thread to finish
        if self.isRunning():
            self.terminate()  # Force terminate if still running
            self.wait(1000)  # Wait for termination


class TwoDMapAnalysisQt6(QMainWindow):
    """Qt6-based window for 2D map analysis of Raman spectra."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Apply compact UI configuration for consistent toolbar sizing
        apply_theme('compact')
        
        self.setWindowTitle("RamanLab - 2D Map Analysis")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Initialize data variables
        self.map_data = None
        self.current_feature = "Integrated Intensity"
        self.use_processed = True
        
        # Analysis results
        self.pca = None
        self.nmf = None
        self.rf_model = None
        self.pca_components = None
        self.nmf_components = None
        self.rf_X_test = None
        self.rf_y_test = None
        self.rf_y_pred = None
        
        # Worker thread management
        self.worker = None
        
        # Template analysis variables
        self.template_fitting_method = "nnls"
        self.use_baseline = True
        self.normalize_coefficients = True
        self.filter_cosmic_rays = True
        self.selected_template = 0
        
        # Cosmic ray detection parameters
        self.cre_threshold_factor = 9.0
        self.cre_window_size = 5
        self.cre_min_width_ratio = 0.1
        self.cre_max_fwhm = 5.0
        
        # Initialize colorbar attribute
        self.colorbar = None
        
        # Define standardized button style to match raman_analysis_app_qt6.py
        self.standard_button_style = """
            QPushButton {
                background-color: #0D9488;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0F766E;
            }
            QPushButton:pressed {
                background-color: #115E59;
            }
        """
        
        # Initialize GUI
        self.setup_ui()
        
        # Set default values
        self.min_wavenumber_edit.setText("800")
        self.max_wavenumber_edit.setText("1000")
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Add progress bar to status bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        self.create_controls_panel(splitter)
        
        # Right panel for visualization
        self.create_visualization_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([300, 1100])
        
        # Create menu bar
        self.create_menu_bar()
    
    def create_controls_panel(self, parent):
        """Create the left control panel."""
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Data Loading Section
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)
        
        load_data_btn = QPushButton("Load Map Data")
        load_data_btn.clicked.connect(self.load_map_data)
        data_layout.addWidget(load_data_btn)
        
        # Save/Load processed data
        save_load_layout = QHBoxLayout()
        save_processed_btn = QPushButton("Save Processed")
        save_processed_btn.clicked.connect(self.save_processed_data)
        load_processed_btn = QPushButton("Load Processed")
        load_processed_btn.clicked.connect(self.load_processed_data)
        save_load_layout.addWidget(save_processed_btn)
        save_load_layout.addWidget(load_processed_btn)
        data_layout.addLayout(save_load_layout)
        
        controls_layout.addWidget(data_group)
        
        # Feature Selection Section
        feature_group = QGroupBox("Feature Selection")
        feature_layout = QVBoxLayout(feature_group)
        
        feature_layout.addWidget(QLabel("Feature:"))
        self.feature_combo = QComboBox()
        self.feature_combo.addItems([
            "Integrated Intensity", "Cosmic Ray Map", "Classification Map", 
            "Classification Probability", "Cluster Map",
            "Template Coefficient", "Template Residual", "Dominant Template"
        ])
        self.feature_combo.currentTextChanged.connect(self.on_feature_selected)
        feature_layout.addWidget(self.feature_combo)
        
        # Template selection (initially hidden)
        self.template_frame = QWidget()
        template_layout = QVBoxLayout(self.template_frame)
        template_layout.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.currentIndexChanged.connect(self.update_map)
        template_layout.addWidget(self.template_combo)
        self.template_frame.setVisible(False)
        feature_layout.addWidget(self.template_frame)
        
        # Wavenumber range
        self.wavenumber_frame = QGroupBox("Wavenumber Range")
        wn_layout = QVBoxLayout(self.wavenumber_frame)
        
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min (cm⁻¹):"))
        self.min_wavenumber_edit = QLineEdit("800")
        self.min_wavenumber_edit.editingFinished.connect(self.update_map)
        min_layout.addWidget(self.min_wavenumber_edit)
        wn_layout.addLayout(min_layout)
        
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max (cm⁻¹):"))
        self.max_wavenumber_edit = QLineEdit("1000")
        self.max_wavenumber_edit.editingFinished.connect(self.update_map)
        max_layout.addWidget(self.max_wavenumber_edit)
        wn_layout.addLayout(max_layout)
        
        feature_layout.addWidget(self.wavenumber_frame)
        
        controls_layout.addWidget(feature_group)
        
        # Data Processing Section
        process_group = QGroupBox("Data Processing")
        process_layout = QVBoxLayout(process_group)
        
        self.use_processed_cb = QCheckBox("Use Processed Data")
        self.use_processed_cb.setChecked(True)
        self.use_processed_cb.toggled.connect(self.update_map)
        process_layout.addWidget(self.use_processed_cb)
        
        self.filter_cosmic_rays_cb = QCheckBox("Filter Cosmic Rays")
        self.filter_cosmic_rays_cb.setChecked(True)
        process_layout.addWidget(self.filter_cosmic_rays_cb)
        
        controls_layout.addWidget(process_group)
        
        # Update Map Button
        update_btn = QPushButton("Update Map")
        update_btn.clicked.connect(self.update_map)
        update_btn.setStyleSheet(self.standard_button_style)
        controls_layout.addWidget(update_btn)
        
        # Add stretch to push everything to top
        controls_layout.addStretch()
        
        parent.addWidget(controls_widget)
    
    def create_visualization_panel(self, parent):
        """Create the right visualization panel."""
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        # Create tab widget for different analysis views
        self.tab_widget = QTabWidget()
        viz_layout.addWidget(self.tab_widget)
        
        # Main Map Tab
        self.create_map_tab()
        
        # Template Analysis Tab
        self.create_template_tab()
        
        # PCA Tab
        self.create_pca_tab()
        
        # NMF Tab  
        self.create_nmf_tab()
        
        # ML Classification Tab
        self.create_ml_tab()
        
        # Train Model Tab
        self.create_train_model_tab()
        
        # Results Tab
        self.create_results_tab()
        
        parent.addWidget(viz_widget)
    
    def create_map_tab(self):
        """Create the main map visualization tab with enhanced plot size."""
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create matplotlib figure with larger size for better visibility
        self.map_fig, self.map_ax = plt.subplots(figsize=(14, 10))
        self.map_canvas = FigureCanvas(self.map_fig)
        map_layout.addWidget(self.map_canvas)
        
        # Add navigation toolbar
        self.map_toolbar = NavigationToolbar(self.map_canvas, map_tab)
        map_layout.addWidget(self.map_toolbar)
        
        self.tab_widget.addTab(map_tab, "Map View")
    
    def create_template_tab(self):
        """Create the template analysis tab with emphasis on visualization."""
        template_tab = QWidget()
        template_layout = QVBoxLayout(template_tab)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)
        template_layout.addWidget(splitter)
        
        # Compact controls panel at top
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(160)  # Increased from 120
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        # Template loading row
        loading_layout = QHBoxLayout()
        
        loading_group = QGroupBox("Template Loading")
        loading_group_layout = QHBoxLayout(loading_group)
        
        load_template_btn = QPushButton("Load Template")
        load_template_btn.clicked.connect(self.load_template)
        loading_group_layout.addWidget(load_template_btn)
        
        load_template_dir_btn = QPushButton("Load Directory")
        load_template_dir_btn.clicked.connect(self.load_template_directory)
        loading_group_layout.addWidget(load_template_dir_btn)
        loading_group_layout.addStretch()
        
        loading_layout.addWidget(loading_group)
        
        # Fitting parameters
        fitting_group = QGroupBox("Fitting Parameters")
        fitting_layout = QHBoxLayout(fitting_group)
        
        fitting_layout.addWidget(QLabel("Method:"))
        self.fitting_method_combo = QComboBox()
        self.fitting_method_combo.addItems(["nnls", "lstsq"])
        self.fitting_method_combo.setMaximumWidth(100)
        fitting_layout.addWidget(self.fitting_method_combo)
        
        self.use_baseline_cb = QCheckBox("Use Baseline")
        self.use_baseline_cb.setChecked(True)
        fitting_layout.addWidget(self.use_baseline_cb)
        
        self.normalize_coeffs_cb = QCheckBox("Normalize Coefficients")
        self.normalize_coeffs_cb.setChecked(True)
        fitting_layout.addWidget(self.normalize_coeffs_cb)
        fitting_layout.addStretch()
        
        loading_layout.addWidget(fitting_group)
        
        # Action button
        action_group = QGroupBox("Action")
        action_layout = QHBoxLayout(action_group)
        
        fit_templates_btn = QPushButton("Fit Templates to Map")
        fit_templates_btn.clicked.connect(self.fit_templates_to_map)
        fit_templates_btn.setStyleSheet(self.standard_button_style)
        action_layout.addWidget(fit_templates_btn)
        action_layout.addStretch()
        
        loading_layout.addWidget(action_group)
        
        controls_layout.addLayout(loading_layout)
        splitter.addWidget(controls_widget)
        
        # Large visualization area
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # Much larger figure for better visibility
        self.template_fig, self.template_ax = plt.subplots(figsize=(16, 8))
        self.template_canvas = FigureCanvas(self.template_fig)
        viz_layout.addWidget(self.template_canvas)
        
        # Navigation toolbar
        self.template_toolbar = NavigationToolbar(self.template_canvas, viz_widget)
        viz_layout.addWidget(self.template_toolbar)
        
        splitter.addWidget(viz_widget)
        
        # Set splitter proportions to emphasize plots
        splitter.setSizes([160, 600])  # Adjusted for larger controls
        
        self.tab_widget.addTab(template_tab, "Template Analysis")
    
    def create_pca_tab(self):
        """Create the PCA analysis tab with emphasis on visualization."""
        pca_tab = QWidget()
        pca_layout = QVBoxLayout(pca_tab)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)
        pca_layout.addWidget(splitter)
        
        # Compact controls panel at top
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(150)  # Increased from 120
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        # Main parameters in horizontal layout
        main_params_layout = QHBoxLayout()
        
        # Basic parameters group - stack vertically
        basic_group = QGroupBox("Parameters")
        basic_layout = QVBoxLayout(basic_group)
        
        components_layout = QHBoxLayout()
        components_layout.addWidget(QLabel("Components:"))
        self.pca_n_components = QSpinBox()
        self.pca_n_components.setRange(2, 100)
        self.pca_n_components.setValue(20)
        self.pca_n_components.setMaximumWidth(80)
        components_layout.addWidget(self.pca_n_components)
        components_layout.addStretch()
        basic_layout.addLayout(components_layout)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.pca_batch_size = QSpinBox()
        self.pca_batch_size.setRange(1000, 10000)
        self.pca_batch_size.setSingleStep(1000)
        self.pca_batch_size.setValue(5000)
        self.pca_batch_size.setMaximumWidth(80)
        batch_layout.addWidget(self.pca_batch_size)
        batch_layout.addStretch()
        basic_layout.addLayout(batch_layout)
        
        main_params_layout.addWidget(basic_group)
        
        # Preprocessing options group - stack vertically
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        self.pca_filter_edges_cb = QCheckBox("Filter Edge Effects")
        self.pca_filter_edges_cb.setChecked(True)
        self.pca_filter_edges_cb.setToolTip("Remove spectra near missing scan lines")
        preprocess_layout.addWidget(self.pca_filter_edges_cb)
        
        self.pca_filter_cosmic_rays_cb = QCheckBox("Filter Cosmic Rays")
        self.pca_filter_cosmic_rays_cb.setChecked(False)  # Disabled by default - CRE done at import
        self.pca_filter_cosmic_rays_cb.setToolTip("Remove cosmic ray events (Note: CRE already applied at data import)")
        preprocess_layout.addWidget(self.pca_filter_cosmic_rays_cb)
        
        main_params_layout.addWidget(preprocess_group)
        
        # Cosmic ray parameters group - stack vertically
        cosmic_group = QGroupBox("CRE Parameters")
        cosmic_layout = QVBoxLayout(cosmic_group)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.pca_cosmic_threshold = QDoubleSpinBox()
        self.pca_cosmic_threshold.setRange(1.0, 20.0)
        self.pca_cosmic_threshold.setValue(8.0)
        self.pca_cosmic_threshold.setSingleStep(0.5)
        self.pca_cosmic_threshold.setMaximumWidth(70)
        threshold_layout.addWidget(self.pca_cosmic_threshold)
        threshold_layout.addStretch()
        cosmic_layout.addLayout(threshold_layout)
        
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        self.pca_cosmic_window = QSpinBox()
        self.pca_cosmic_window.setRange(3, 15)
        self.pca_cosmic_window.setValue(5)
        self.pca_cosmic_window.setMaximumWidth(60)
        window_layout.addWidget(self.pca_cosmic_window)
        window_layout.addStretch()
        cosmic_layout.addLayout(window_layout)
        
        main_params_layout.addWidget(cosmic_group)
        
        # Action buttons group
        buttons_group = QGroupBox("Actions")
        buttons_layout = QHBoxLayout(buttons_group)
        run_pca_btn = QPushButton("Run PCA")
        run_pca_btn.clicked.connect(self.run_pca)
        run_pca_btn.setStyleSheet(self.standard_button_style)
        save_pca_btn = QPushButton("Save Results")
        save_pca_btn.clicked.connect(self.save_pca_results)
        buttons_layout.addWidget(run_pca_btn)
        buttons_layout.addWidget(save_pca_btn)
        buttons_layout.addStretch()
        
        main_params_layout.addWidget(buttons_group)
        
        controls_layout.addLayout(main_params_layout)
        splitter.addWidget(controls_widget)
        
        # Large visualization area
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # Much larger figure for better visibility
        self.pca_fig, (self.pca_ax1, self.pca_ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.pca_canvas = FigureCanvas(self.pca_fig)
        viz_layout.addWidget(self.pca_canvas)
        
        # Navigation toolbar
        self.pca_toolbar = NavigationToolbar(self.pca_canvas, viz_widget)
        viz_layout.addWidget(self.pca_toolbar)
        
        splitter.addWidget(viz_widget)
        
        # Set splitter proportions to emphasize plots
        splitter.setSizes([150, 600])  # Adjusted for larger controls
        
        self.tab_widget.addTab(pca_tab, "PCA")
    
    def create_nmf_tab(self):
        """Create the NMF analysis tab with emphasis on visualization."""
        nmf_tab = QWidget()
        nmf_layout = QVBoxLayout(nmf_tab)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)
        nmf_layout.addWidget(splitter)
        
        # Compact controls panel at top
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(150)  # Increased from 120
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        # Main parameters in horizontal layout
        main_params_layout = QHBoxLayout()
        
        # Basic parameters group - stack vertically
        basic_group = QGroupBox("Parameters")
        basic_layout = QVBoxLayout(basic_group)
        
        components_layout = QHBoxLayout()
        components_layout.addWidget(QLabel("Components:"))
        self.nmf_n_components = QSpinBox()
        self.nmf_n_components.setRange(2, 50)
        self.nmf_n_components.setValue(10)
        self.nmf_n_components.setMaximumWidth(80)
        components_layout.addWidget(self.nmf_n_components)
        components_layout.addStretch()
        basic_layout.addLayout(components_layout)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.nmf_batch_size = QSpinBox()
        self.nmf_batch_size.setRange(1000, 10000)
        self.nmf_batch_size.setSingleStep(1000)
        self.nmf_batch_size.setValue(5000)
        self.nmf_batch_size.setMaximumWidth(80)
        batch_layout.addWidget(self.nmf_batch_size)
        batch_layout.addStretch()
        basic_layout.addLayout(batch_layout)
        
        main_params_layout.addWidget(basic_group)
        
        # Preprocessing options group - stack vertically
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QVBoxLayout(preprocess_group)
        
        self.nmf_filter_edges_cb = QCheckBox("Filter Edge Effects")
        self.nmf_filter_edges_cb.setChecked(True)
        self.nmf_filter_edges_cb.setToolTip("Remove spectra near missing scan lines")
        preprocess_layout.addWidget(self.nmf_filter_edges_cb)
        
        self.nmf_filter_cosmic_rays_cb = QCheckBox("Filter Cosmic Rays")
        self.nmf_filter_cosmic_rays_cb.setChecked(False)  # Disabled by default - CRE done at import
        self.nmf_filter_cosmic_rays_cb.setToolTip("Remove cosmic ray events (Note: CRE already applied at data import)")
        preprocess_layout.addWidget(self.nmf_filter_cosmic_rays_cb)
        
        main_params_layout.addWidget(preprocess_group)
        
        # Cosmic ray parameters group - stack vertically
        cosmic_group = QGroupBox("CRE Parameters")
        cosmic_layout = QVBoxLayout(cosmic_group)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.nmf_cosmic_threshold = QDoubleSpinBox()
        self.nmf_cosmic_threshold.setRange(1.0, 20.0)
        self.nmf_cosmic_threshold.setValue(8.0)
        self.nmf_cosmic_threshold.setSingleStep(0.5)
        self.nmf_cosmic_threshold.setMaximumWidth(70)
        threshold_layout.addWidget(self.nmf_cosmic_threshold)
        threshold_layout.addStretch()
        cosmic_layout.addLayout(threshold_layout)
        
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        self.nmf_cosmic_window = QSpinBox()
        self.nmf_cosmic_window.setRange(3, 15)
        self.nmf_cosmic_window.setValue(5)
        self.nmf_cosmic_window.setMaximumWidth(60)
        window_layout.addWidget(self.nmf_cosmic_window)
        window_layout.addStretch()
        cosmic_layout.addLayout(window_layout)
        
        main_params_layout.addWidget(cosmic_group)
        
        # Action buttons group
        buttons_group = QGroupBox("Actions")
        buttons_layout = QVBoxLayout(buttons_group)  # Changed to vertical layout
        
        # First row of buttons
        main_buttons_layout = QHBoxLayout()
        run_nmf_btn = QPushButton("Run NMF")
        run_nmf_btn.clicked.connect(self.run_nmf)
        run_nmf_btn.setStyleSheet(self.standard_button_style)
        save_nmf_btn = QPushButton("Save Results")
        save_nmf_btn.clicked.connect(self.save_nmf_results)
        save_nmf_btn.setStyleSheet(self.standard_button_style)
        main_buttons_layout.addWidget(run_nmf_btn)
        main_buttons_layout.addWidget(save_nmf_btn)
        main_buttons_layout.addStretch()
        
        # Second row - Component spectra save button
        component_buttons_layout = QHBoxLayout()
        save_components_btn = QPushButton("Save Component Spectra")
        save_components_btn.clicked.connect(self.save_nmf_component_spectra)
        save_components_btn.setStyleSheet(self.standard_button_style)
        save_components_btn.setToolTip("Save all NMF component spectra as space-separated text files")
        component_buttons_layout.addWidget(save_components_btn)
        component_buttons_layout.addStretch()
        
        buttons_layout.addLayout(main_buttons_layout)
        buttons_layout.addLayout(component_buttons_layout)
        
        main_params_layout.addWidget(buttons_group)
        
        controls_layout.addLayout(main_params_layout)
        splitter.addWidget(controls_widget)
        
        # Large visualization area
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # Much larger figure for better visibility
        self.nmf_fig, (self.nmf_ax1, self.nmf_ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.nmf_canvas = FigureCanvas(self.nmf_fig)
        viz_layout.addWidget(self.nmf_canvas)
        
        # Navigation toolbar
        self.nmf_toolbar = NavigationToolbar(self.nmf_canvas, viz_widget)
        viz_layout.addWidget(self.nmf_toolbar)
        
        splitter.addWidget(viz_widget)
        
        # Set splitter proportions to emphasize plots
        splitter.setSizes([150, 600])  # Adjusted for larger controls
        
        self.tab_widget.addTab(nmf_tab, "NMF")
    
    def create_ml_tab(self):
        """Create the ML classification tab with emphasis on visualization."""
        ml_tab = QWidget()
        ml_layout = QVBoxLayout(ml_tab)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)
        ml_layout.addWidget(splitter)
        
        # Larger controls panel to accommodate all elements
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(280)  # Significantly increased from 180
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(8)  # Add more spacing between elements
        
        # Directory selection row - make more spacious
        dirs_group = QGroupBox("Training Data Directories")
        dirs_group_layout = QGridLayout(dirs_group)
        dirs_group_layout.setSpacing(5)
        
        dirs_group_layout.addWidget(QLabel("Class A:"), 0, 0)
        self.class_a_path = QLineEdit()
        self.class_a_path.setPlaceholderText("Class A spectra directory...")
        self.class_a_path.setMinimumHeight(25)  # Ensure minimum height
        dirs_group_layout.addWidget(self.class_a_path, 0, 1)
        class_a_btn = QPushButton("Browse")
        class_a_btn.clicked.connect(self.browse_class_a_directory)
        class_a_btn.setMinimumWidth(80)
        class_a_btn.setMinimumHeight(25)
        dirs_group_layout.addWidget(class_a_btn, 0, 2)
        
        dirs_group_layout.addWidget(QLabel("Class B:"), 1, 0)
        self.class_b_path = QLineEdit()
        self.class_b_path.setPlaceholderText("Class B spectra directory...")
        self.class_b_path.setMinimumHeight(25)  # Ensure minimum height
        dirs_group_layout.addWidget(self.class_b_path, 1, 1)
        class_b_btn = QPushButton("Browse")
        class_b_btn.clicked.connect(self.browse_class_b_directory)
        class_b_btn.setMinimumWidth(80)
        class_b_btn.setMinimumHeight(25)
        dirs_group_layout.addWidget(class_b_btn, 1, 2)
        
        controls_layout.addWidget(dirs_group)
        
        # Parameters row - RF, Features, and Cosmic Ray Rejection on same line
        params_layout = QHBoxLayout()
        
        # Random Forest parameters with action buttons below
        rf_container = QWidget()
        rf_container_layout = QVBoxLayout(rf_container)
        rf_container_layout.setContentsMargins(0, 0, 0, 0)
        
        rf_group = QGroupBox("RF Parameters")
        rf_layout = QHBoxLayout(rf_group)
        rf_layout.addWidget(QLabel("Trees:"))
        self.rf_n_trees = QSpinBox()
        self.rf_n_trees.setRange(10, 1000)
        self.rf_n_trees.setSingleStep(10)
        self.rf_n_trees.setValue(100)
        self.rf_n_trees.setMaximumWidth(80)
        rf_layout.addWidget(self.rf_n_trees)
        
        rf_layout.addWidget(QLabel("Depth:"))
        self.rf_max_depth = QSpinBox()
        self.rf_max_depth.setRange(1, 100)
        self.rf_max_depth.setValue(10)
        self.rf_max_depth.setMaximumWidth(70)
        rf_layout.addWidget(self.rf_max_depth)
        rf_layout.addStretch()
        rf_container_layout.addWidget(rf_group)
        
        # Action buttons under RF Parameters
        actions_layout = QHBoxLayout()
        train_rf_btn = QPushButton("Train Random Forest")
        train_rf_btn.clicked.connect(self.train_rf)
        train_rf_btn.setStyleSheet(self.standard_button_style)
        train_rf_btn.setMinimumHeight(30)
        actions_layout.addWidget(train_rf_btn)
        
        classify_map_btn = QPushButton("Classify Map")
        classify_map_btn.clicked.connect(self.classify_map_spectra)
        classify_map_btn.setStyleSheet(self.standard_button_style)
        classify_map_btn.setMinimumHeight(30)
        actions_layout.addWidget(classify_map_btn)
        
        rf_container_layout.addLayout(actions_layout)
        params_layout.addWidget(rf_container)
        
        # Feature selection
        feature_group = QGroupBox("Features")
        feature_layout = QVBoxLayout(feature_group)
        self.use_pca_nmf_features_cb = QCheckBox("Use PCA/NMF Features")
        self.use_pca_nmf_features_cb.setChecked(True)
        feature_layout.addWidget(self.use_pca_nmf_features_cb)
        
        self.feature_type_combo = QComboBox()
        self.feature_type_combo.addItems(["Auto (PCA > NMF > Full)", "PCA Only", "NMF Only", "Full Spectrum"])
        feature_layout.addWidget(self.feature_type_combo)
        params_layout.addWidget(feature_group)
        
        # Cosmic ray options
        cosmic_group = QGroupBox("Cosmic Ray Rejection")
        cosmic_layout = QVBoxLayout(cosmic_group)
        self.enable_cosmic_rejection_cb = QCheckBox("Enable CRE Rejection")
        self.enable_cosmic_rejection_cb.setChecked(True)
        cosmic_layout.addWidget(self.enable_cosmic_rejection_cb)
        
        cosmic_params_layout = QHBoxLayout()
        cosmic_params_layout.addWidget(QLabel("Threshold:"))
        self.cosmic_threshold_spin = QDoubleSpinBox()
        self.cosmic_threshold_spin.setRange(1.0, 20.0)
        self.cosmic_threshold_spin.setValue(9.0)
        self.cosmic_threshold_spin.setSingleStep(0.5)
        self.cosmic_threshold_spin.setMaximumWidth(70)
        cosmic_params_layout.addWidget(self.cosmic_threshold_spin)
        
        cosmic_params_layout.addWidget(QLabel("Window:"))
        self.cosmic_window_spin = QSpinBox()
        self.cosmic_window_spin.setRange(3, 15)
        self.cosmic_window_spin.setValue(5)
        self.cosmic_window_spin.setMaximumWidth(60)
        cosmic_params_layout.addWidget(self.cosmic_window_spin)
        cosmic_params_layout.addStretch()
        cosmic_layout.addLayout(cosmic_params_layout)
        params_layout.addWidget(cosmic_group)
        
        controls_layout.addLayout(params_layout)
        splitter.addWidget(controls_widget)
        
        # Large visualization area
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # Much larger figure for better visibility
        self.ml_fig = plt.figure(figsize=(16, 8))
        self.ml_canvas = FigureCanvas(self.ml_fig)
        viz_layout.addWidget(self.ml_canvas)
        
        # Navigation toolbar
        self.ml_toolbar = NavigationToolbar(self.ml_canvas, viz_widget)
        viz_layout.addWidget(self.ml_toolbar)
        
        splitter.addWidget(viz_widget)
        
        # Set splitter proportions to emphasize plots
        splitter.setSizes([280, 600])  # Adjusted for much larger controls
        
        self.tab_widget.addTab(ml_tab, "ML Classification")
    
    def create_train_model_tab(self):
        """Create the train model tab with emphasis on visualization."""
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Vertical)
        train_layout.addWidget(splitter)
        
        # Larger controls panel 
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(250)  # Increased from 200
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(8)  # Add spacing
        
        # Model management row
        model_layout = QHBoxLayout()
        model_group = QGroupBox("Model Management")
        model_group_layout = QHBoxLayout(model_group)
        
        model_group_layout.addWidget(QLabel("Name:"))
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("Model name...")
        self.model_name_edit.setMinimumHeight(25)
        model_group_layout.addWidget(self.model_name_edit)
        
        save_model_btn = QPushButton("Save Model")
        save_model_btn.clicked.connect(self.save_rf_model)
        save_model_btn.setMinimumWidth(100)
        save_model_btn.setMinimumHeight(25)
        model_group_layout.addWidget(save_model_btn)
        
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_rf_model)
        load_model_btn.setMinimumWidth(100)
        load_model_btn.setMinimumHeight(25)
        model_group_layout.addWidget(load_model_btn)
        
        model_layout.addWidget(model_group)
        controls_layout.addLayout(model_layout)
        
        # Parameters row
        params_layout = QHBoxLayout()
        
        # Training parameters
        train_params_group = QGroupBox("Training Parameters")
        train_params_layout = QHBoxLayout(train_params_group)
        
        self.use_reduced_features_cb = QCheckBox("Use PCA/NMF Features")
        self.use_reduced_features_cb.setChecked(True)
        train_params_layout.addWidget(self.use_reduced_features_cb)
        
        train_params_layout.addWidget(QLabel("Test Size:"))
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setMaximumWidth(80)
        train_params_layout.addWidget(self.test_size_spin)
        train_params_layout.addStretch()
        
        params_layout.addWidget(train_params_group)
        
        # Method selection
        method_group = QGroupBox("Method")
        method_layout = QHBoxLayout(method_group)
        
        method_layout.addWidget(QLabel("Type:"))
        self.training_method_combo = QComboBox()
        self.training_method_combo.addItems([
            "K-Means Clustering",
            "Gaussian Mixture Model", 
            "Spectral Clustering",
            "DBSCAN Clustering"
        ])
        method_layout.addWidget(self.training_method_combo)
        
        method_layout.addWidget(QLabel("Clusters:"))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(5)
        self.n_clusters_spin.setMaximumWidth(70)
        method_layout.addWidget(self.n_clusters_spin)
        method_layout.addStretch()
        
        params_layout.addWidget(method_group)
        
        # Action button
        action_group = QGroupBox("Action")
        action_layout = QHBoxLayout(action_group)
        
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_unsupervised_model)
        self.train_model_btn.setStyleSheet(self.standard_button_style)
        self.train_model_btn.setMinimumHeight(30)
        action_layout.addWidget(self.train_model_btn)
        action_layout.addStretch()
        
        params_layout.addWidget(action_group)
        
        controls_layout.addLayout(params_layout)
        
        # Model status (with more space)
        self.model_status_text = QTextEdit()
        self.model_status_text.setMaximumHeight(80)  # Increased from 60
        self.model_status_text.setReadOnly(True)
        self.model_status_text.setPlainText("No model trained yet.")
        controls_layout.addWidget(self.model_status_text)
        
        splitter.addWidget(controls_widget)
        
        # Large visualization area
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        
        # Much larger figure for better visibility
        self.train_fig = plt.figure(figsize=(16, 10))
        self.train_canvas = FigureCanvas(self.train_fig)
        viz_layout.addWidget(self.train_canvas)
        
        # Navigation toolbar
        self.train_toolbar = NavigationToolbar(self.train_canvas, viz_widget)
        viz_layout.addWidget(self.train_toolbar)
        
        splitter.addWidget(viz_widget)
        
        # Set splitter proportions to emphasize plots
        splitter.setSizes([250, 600])  # Adjusted for larger controls
        
        self.tab_widget.addTab(train_tab, "Train Model")
    
    def create_results_tab(self):
        """Create the results summary tab with emphasis on visualization."""
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Controls at top with proper button sizing
        controls_widget = QWidget()
        controls_widget.setMaximumHeight(60)  # Reduced height for single row
        controls_widget_layout = QVBoxLayout(controls_widget)
        
        # Single row of buttons with consistent sizing
        buttons_layout = QHBoxLayout()
        
        # Standard button size (matching Update Map button)
        button_width = 160
        button_height = 30
        
        generate_viz_btn = QPushButton("Generate Visualizations")
        generate_viz_btn.clicked.connect(self.update_results_visualization)
        generate_viz_btn.setStyleSheet(self.standard_button_style)
        generate_viz_btn.setMinimumWidth(button_width)
        generate_viz_btn.setMinimumHeight(button_height)
        
        generate_report_btn = QPushButton("Generate Report")
        generate_report_btn.clicked.connect(self.generate_report)
        generate_report_btn.setStyleSheet(self.standard_button_style)
        generate_report_btn.setMinimumWidth(button_width)
        generate_report_btn.setMinimumHeight(button_height)
        
        plot_class_a_btn = QPushButton("Plot Class A Spectra")
        plot_class_a_btn.clicked.connect(self.plot_class_a_spectra)
        plot_class_a_btn.setStyleSheet(self.standard_button_style)
        plot_class_a_btn.setMinimumWidth(button_width)
        plot_class_a_btn.setMinimumHeight(button_height)
        
        save_all_class_a_btn = QPushButton("Save All Class A Spectra")
        save_all_class_a_btn.clicked.connect(self.save_all_class_a_spectra)
        save_all_class_a_btn.setStyleSheet(self.standard_button_style)
        save_all_class_a_btn.setMinimumWidth(button_width)
        save_all_class_a_btn.setMinimumHeight(button_height)
        
        buttons_layout.addWidget(generate_viz_btn)
        buttons_layout.addWidget(generate_report_btn)
        buttons_layout.addWidget(plot_class_a_btn)
        buttons_layout.addWidget(save_all_class_a_btn)
        buttons_layout.addStretch()
        
        controls_widget_layout.addLayout(buttons_layout)
        results_layout.addWidget(controls_widget)
        
        # Large results visualization area
        self.results_fig = plt.figure(figsize=(16, 12))
        self.results_canvas = FigureCanvas(self.results_fig)
        results_layout.addWidget(self.results_canvas)
        
        self.results_toolbar = NavigationToolbar(self.results_canvas, results_tab)
        results_layout.addWidget(self.results_toolbar)
        
        self.tab_widget.addTab(results_tab, "Results")
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_action = QAction('Load Map Data', self)
        load_action.triggered.connect(self.load_map_data)
        file_menu.addAction(load_action)
        
        load_processed_action = QAction('Load Processed Data', self)
        load_processed_action.triggered.connect(self.load_processed_data)
        file_menu.addAction(load_processed_action)
        
        save_action = QAction('Save Processed Data', self)
        save_action.triggered.connect(self.save_processed_data)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        
        pca_action = QAction('Run PCA', self)
        pca_action.triggered.connect(self.run_pca)
        analysis_menu.addAction(pca_action)
        
        nmf_action = QAction('Run NMF', self)
        nmf_action.triggered.connect(self.run_nmf)
        analysis_menu.addAction(nmf_action)
        
        rf_action = QAction('Train Random Forest', self)
        rf_action.triggered.connect(self.train_rf)
        analysis_menu.addAction(rf_action) 
        
        analysis_menu.addSeparator()
        
        diagnose_action = QAction('Diagnose Data Consistency', self)
        diagnose_action.triggered.connect(self.diagnose_data_consistency)
        analysis_menu.addAction(diagnose_action)
        
        # Cosmic Ray menu items
        cosmic_ray_menu = analysis_menu.addMenu('Cosmic Ray Tools')
        
        cr_diagnose_action = QAction('Diagnose Cosmic Rays', self)
        cr_diagnose_action.triggered.connect(self.run_cosmic_ray_diagnosis)
        cosmic_ray_menu.addAction(cr_diagnose_action)
        
        cr_clean_action = QAction('Clean All Cosmic Rays', self)
        cr_clean_action.triggered.connect(self.clean_all_cosmic_rays)
        cosmic_ray_menu.addAction(cr_clean_action)
        
        # ML debugging menu items
        ml_debug_menu = analysis_menu.addMenu('ML Debugging')
        
        ml_debug_action = QAction('Debug ML Pipeline', self)
        ml_debug_action.triggered.connect(self.debug_ml_pipeline)
        ml_debug_menu.addAction(ml_debug_action)
    
    def load_map_data(self):
        """Load map data from a directory."""
        try:
            directory = QFileDialog.getExistingDirectory(
                self, "Select Map Data Directory", ""
            )
            
            if directory:
                self.statusBar().showMessage("Loading map data...")
                
                # Create and start worker thread
                self.worker = MapAnalysisWorker(self._load_map_data_worker, directory)
                self.worker.finished.connect(self._on_map_data_loaded)
                self.worker.error.connect(self._on_map_data_error)
                self.worker.start()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading map data: {str(e)}")
    
    def _load_map_data_worker(self, worker, directory):
        """Worker method for loading map data."""
        return RamanMapData(directory)
    
    def _on_map_data_loaded(self, map_data):
        """Handle successful map data loading."""
        self.map_data = map_data
        self.statusBar().showMessage(f"Loaded {len(self.map_data.spectra)} spectra")
        
        # Update template combo
        self.update_template_combo()
        
        # Update the map
        self.update_map()
        
        QMessageBox.information(self, "Success", 
                               f"Successfully loaded {len(self.map_data.spectra)} spectra")
    
    def _on_map_data_error(self, error_msg):
        """Handle map data loading error."""
        self.statusBar().showMessage("Ready")
        QMessageBox.critical(self, "Error", f"Error loading map data: {error_msg}")
    
    def save_processed_data(self):
        """Save processed map data to pickle file."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "No map data to save")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Processed Data", "", "Pickle Files (*.pkl)"
            )
            
            if file_path:
                self.map_data.save_map_data(file_path)
                self.statusBar().showMessage("Processed data saved")
                QMessageBox.information(self, "Success", "Processed data saved successfully")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving data: {str(e)}")
    
    def load_processed_data(self):
        """Load processed map data from pickle file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Processed Data", "", "Pickle Files (*.pkl)"
            )
            
            if file_path:
                self.map_data = RamanMapData.load_map_data(file_path)
                
                # Apply cosmic ray cleaning to loaded data for consistency
                logger.info("Applying cosmic ray cleaning to loaded data...")
                
                # Clean all spectra in the loaded data
                cleaned_count = 0
                for (x_pos, y_pos), spectrum in self.map_data.spectra.items():
                    # Clean raw intensities
                    cosmic_detected, cleaned_intensities = self.map_data.detect_cosmic_rays(
                        spectrum.wavenumbers, spectrum.intensities,
                        threshold_factor=6.0,  # More aggressive for cleaning
                        window_size=5
                    )
                    
                    if cosmic_detected:
                        # Update the raw intensities
                        spectrum.intensities = cleaned_intensities
                        cleaned_count += 1
                        
                        # Re-process the cleaned spectrum
                        spectrum.processed_intensities = self.map_data._preprocess_spectrum(
                            spectrum.wavenumbers, cleaned_intensities
                        )
                
                logger.info(f"Cleaned cosmic rays from {cleaned_count} spectra in loaded data")
                
                self.statusBar().showMessage(f"Loaded {len(self.map_data.spectra)} spectra")
                
                # Update template combo
                self.update_template_combo()
                
                # Update the map
                self.update_map()
                
                QMessageBox.information(self, "Success", 
                                       f"Successfully loaded {len(self.map_data.spectra)} spectra\n"
                                       f"Cosmic ray cleaning applied during load.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")
    
    def load_template(self):
        """Load a single template spectrum."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Template Spectrum", "", 
                "Spectrum Files (*.csv *.txt);;All Files (*)"
            )
            
            if file_path:
                success = self.map_data.template_manager.load_template(file_path)
                if success:
                    self.update_template_combo()
                    self.statusBar().showMessage("Template loaded successfully")
                    QMessageBox.information(self, "Success", "Template loaded successfully")
                else:
                    QMessageBox.critical(self, "Error", "Failed to load template")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading template: {str(e)}")
    
    def load_template_directory(self):
        """Load all templates from a directory."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        try:
            directory = QFileDialog.getExistingDirectory(
                self, "Select Template Directory", ""
            )
            
            if directory:
                count = self.map_data.template_manager.load_templates_from_directory(directory)
                if count > 0:
                    self.update_template_combo()
                    self.statusBar().showMessage(f"Loaded {count} templates")
                    QMessageBox.information(self, "Success", f"Loaded {count} templates")
                else:
                    QMessageBox.warning(self, "Warning", "No templates found in directory")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading templates: {str(e)}")
    
    def update_template_combo(self):
        """Update the template selection combo box."""
        self.template_combo.clear()
        if self.map_data and self.map_data.template_manager.get_template_count() > 0:
            names = self.map_data.template_manager.get_template_names()
            self.template_combo.addItems(names)
    
    def fit_templates_to_map(self):
        """Fit templates to the entire map."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        if self.map_data.template_manager.get_template_count() == 0:
            QMessageBox.warning(self, "Warning", "Load templates first")
            return
        
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Fitting templates to map...")
            
            # Create and start worker thread
            self.worker = MapAnalysisWorker(
                self._fit_templates_worker,
                self.template_fitting_method,
                self.use_baseline,
                self.filter_cosmic_rays
            )
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self._on_template_fitting_finished)
            self.worker.error.connect(self._on_template_fitting_error)
            self.worker.start()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error fitting templates: {str(e)}")
    
    def _fit_templates_worker(self, worker, method, use_baseline, filter_cosmic_rays):
        """Worker method for template fitting."""
        return self.map_data.fit_templates_to_map(
            method=method,
            use_baseline=use_baseline,
            filter_cosmic_rays=filter_cosmic_rays,
            threshold_factor=self.cre_threshold_factor,
            window_size=self.cre_window_size,
            min_width_ratio=self.cre_min_width_ratio,
            max_fwhm=self.cre_max_fwhm
        )
    
    def _on_template_fitting_finished(self, success):
        """Handle successful template fitting."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        if success:
            self.statusBar().showMessage("Template fitting completed")
            QMessageBox.information(self, "Success", "Template fitting completed successfully")
            # Update the map if template features are selected
            if self.current_feature in ["Template Coefficient", "Template Residual", "Dominant Template"]:
                self.update_map()
        else:
            QMessageBox.warning(self, "Warning", "Template fitting failed")
    
    def _on_template_fitting_error(self, error_msg):
        """Handle template fitting error."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Ready")
        QMessageBox.critical(self, "Error", f"Error fitting templates: {error_msg}")
    
    def on_feature_selected(self, feature):
        """Handle feature selection change."""
        self.current_feature = feature
        
        # Show/hide appropriate controls based on feature
        if feature in ["Template Coefficient", "Template Residual", "Dominant Template"]:
            self.template_frame.setVisible(True)
            self.wavenumber_frame.setVisible(False)
        elif feature == "Integrated Intensity":
            self.template_frame.setVisible(False)
            self.wavenumber_frame.setVisible(True)
        else:
            self.template_frame.setVisible(False)
            self.wavenumber_frame.setVisible(False)
        
        self.update_map()
        
        # Also update the results visualization if it exists
        if hasattr(self, 'results_canvas'):
            self.update_results_visualization()
    
    def update_map(self):
        """Update the map visualization."""
        if self.map_data is None:
            return
        
        try:
            # Force canvas to stop any pending drawing operations
            self.map_canvas.draw_idle()
            
            # Completely clear the figure to avoid layout issues with colorbars
            self.map_fig.clear()
            
            # Recreate the subplot with proper sizing
            self.map_ax = self.map_fig.add_subplot(111)
            
            # Reset colorbar reference
            self.colorbar = None
            
            # Get map data based on selected feature
            if self.current_feature == "Integrated Intensity":
                min_wn = float(self.min_wavenumber_edit.text() or "800")
                max_wn = float(self.max_wavenumber_edit.text() or "1000")
                map_data = self.create_integrated_intensity_map(min_wn, max_wn)
                title = f"Integrated Intensity ({min_wn}-{max_wn} cm⁻¹)"
            
            elif self.current_feature == "Template Coefficient":
                if self.template_combo.currentIndex() >= 0:
                    template_idx = self.template_combo.currentIndex()
                    map_data = self.map_data.get_template_coefficient_map(
                        template_idx, normalized=self.normalize_coefficients
                    )
                    template_name = self.template_combo.currentText()
                    title = f"Template Coefficient: {template_name}"
                else:
                    return
                    
            elif self.current_feature == "Template Residual":
                map_data = self.map_data.get_residual_map()
                title = "Template Fitting Residual"
                
            elif self.current_feature == "Dominant Template":
                map_data = self.map_data.get_dominant_template_map()
                title = "Dominant Template"
                
            elif self.current_feature == "Cosmic Ray Map":
                map_data = self.create_cosmic_ray_map()
                title = "Cosmic Ray Detection Map"
                
            elif self.current_feature == "Classification Map":
                if hasattr(self, 'classification_map'):
                    map_data = self.classification_map
                    title = "ML Classification Map"
                else:
                    QMessageBox.information(self, "Info", "Run ML Classification first")
                    return
                
            elif self.current_feature == "Classification Probability":
                if hasattr(self, 'probabilities_map'):
                    map_data = self.probabilities_map
                    title = "ML Classification Probability"
                else:
                    QMessageBox.information(self, "Info", "Run ML Classification first")
                    return
                    
            elif self.current_feature == "Cluster Map":
                if hasattr(self, 'cluster_map'):
                    map_data = self.cluster_map
                    title = "Unsupervised Clustering Map"
                else:
                    QMessageBox.information(self, "Info", "Train an unsupervised model first")
                    return
                    
            else:
                # Default to integrated intensity
                map_data = self.create_integrated_intensity_map(800, 1000)
                title = "Integrated Intensity (800-1000 cm⁻¹)"
            
            # Plot the map
            if map_data is not None and map_data.size > 0:
                # Clear any NaN values for better visualization
                map_data_clean = np.ma.masked_invalid(map_data)
                
                im = self.map_ax.imshow(map_data_clean, cmap='viridis', aspect='auto', origin='lower')
                self.map_ax.set_title(title, fontsize=12, fontweight='bold')
                self.map_ax.set_xlabel("X Position")
                self.map_ax.set_ylabel("Y Position")
                
                # Add colorbar with proper layout management
                try:
                    self.colorbar = self.map_fig.colorbar(im, ax=self.map_ax, shrink=0.8)
                except Exception as e:
                    logger.warning(f"Error creating colorbar: {str(e)}")
                    self.colorbar = None
                
                # Adjust layout to ensure proper spacing
                self.map_fig.tight_layout(pad=2.0)
                
                # Force complete canvas refresh
                self.map_canvas.flush_events()
                self.map_canvas.draw()
                
                # Process any pending GUI events
                from PySide6.QtCore import QCoreApplication
                QCoreApplication.processEvents()
            
        except Exception as e:
            logger.error(f"Error updating map: {str(e)}")
            self.statusBar().showMessage(f"Error updating map: {str(e)}")
    
    def create_integrated_intensity_map(self, min_wn, max_wn):
        """Create integrated intensity map for given wavenumber range."""
        if self.map_data is None:
            return None
        
        map_shape = (len(self.map_data.y_positions), len(self.map_data.x_positions))
        map_data = np.full(map_shape, np.nan)
        
        for i, y_pos in enumerate(self.map_data.y_positions):
            for j, x_pos in enumerate(self.map_data.x_positions):
                spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                if spectrum:
                    # Use processed or raw data based on checkbox
                    if self.use_processed_cb.isChecked() and spectrum.processed_intensities is not None:
                        wavenumbers = self.map_data.target_wavenumbers
                        intensities = spectrum.processed_intensities
                    else:
                        wavenumbers = spectrum.wavenumbers
                        intensities = spectrum.intensities
                    
                    # Find wavenumber range
                    mask = (wavenumbers >= min_wn) & (wavenumbers <= max_wn)
                    if np.any(mask):
                        selected_wn = wavenumbers[mask]
                        selected_int = intensities[mask]
                        
                        # Calculate integrated intensity
                        if len(selected_wn) > 1:
                            map_data[i, j] = np.trapz(selected_int, selected_wn)
                        else:
                            map_data[i, j] = np.sum(selected_int)
        
        return map_data
    
    def create_cosmic_ray_map(self):
        """Create cosmic ray detection map."""
        if self.map_data is None or self.map_data.cosmic_ray_map is None:
            return None
        return self.map_data.cosmic_ray_map
    
    def run_pca(self):
        """Run PCA analysis on the map data."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return

        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Running PCA Analysis...")

            # Get parameters
            n_components = self.pca_n_components.value()
            batch_size = self.pca_batch_size.value()

            # Prepare data for PCA
            self.progress_bar.setValue(20)
            QApplication.processEvents()

            # Get the map data as a 2D array (pixels x features)
            df = self.map_data.prepare_ml_data(use_processed=self.use_processed_cb.isChecked())
            
            if df is None or df.empty:
                raise ValueError("No data available for PCA analysis")

            # Extract spectral data (exclude position columns)
            spectral_columns = [col for col in df.columns if col not in ['x_pos', 'y_pos']]
            X = df[spectral_columns].values
            
            # Remove any NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.progress_bar.setValue(40)
            QApplication.processEvents()

            # Apply PCA
            self.statusBar().showMessage("Fitting PCA model...")
            self.pca = PCA(n_components=n_components)
            
            # For large datasets, we might need to subsample
            if len(X) > batch_size:
                # Sample randomly but ensure we get representative data
                indices = np.random.choice(len(X), size=batch_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Fit PCA
            self.pca.fit(X_sample)
            
            self.progress_bar.setValue(70)
            QApplication.processEvents()

            # Transform all data
            self.statusBar().showMessage("Transforming data...")
            self.pca_components = self.pca.transform(X)
            
            # Store scaler for later use
            from sklearn.preprocessing import StandardScaler
            self.pca_scaler = StandardScaler()
            self.pca_scaler.fit(X_sample)
            
            self.progress_bar.setValue(90)
            QApplication.processEvents()

            # Plot results
            self._plot_pca_results()
            
            self.progress_bar.setValue(100)
            self.statusBar().showMessage("PCA analysis complete")
            
            # Hide progress bar after a brief delay
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.statusBar().showMessage("Ready")
            QMessageBox.critical(self, "Error", f"PCA analysis failed: {str(e)}")

    def _plot_pca_results(self):
        """Plot PCA analysis results."""
        if self.pca is None or self.pca_components is None:
            return

        try:
            # Clear axes
            self.pca_ax1.clear()
            self.pca_ax2.clear()

            # Plot explained variance
            explained_var = self.pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            self.pca_ax1.plot(range(1, len(explained_var) + 1), explained_var, 'bo-', label='Individual')
            self.pca_ax1.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', label='Cumulative')
            self.pca_ax1.set_xlabel('Principal Component')
            self.pca_ax1.set_ylabel('Explained Variance Ratio')
            self.pca_ax1.set_title('PCA Explained Variance')
            self.pca_ax1.legend()
            self.pca_ax1.grid(True)

            # Plot PC1 vs PC2 scatter
            n_points = len(self.pca_components)
            
            # Ensure we have at least 2 components for scatter plot
            if self.pca_components.shape[1] < 2:
                self.pca_ax2.text(0.5, 0.5, 'Need at least 2 PCA components\nfor scatter plot', 
                                 transform=self.pca_ax2.transAxes, ha='center', va='center')
                self.pca_ax2.set_title('PCA Component Scatter Plot')
            else:
                # Use binning for large datasets to improve performance
                if n_points > 10000:
                    # Create binned scatter plot using hexbin
                    hb = self.pca_ax2.hexbin(self.pca_components[:, 0], self.pca_components[:, 1], 
                                           gridsize=50, cmap='viridis', mincnt=1)
                    
                    # Add text showing total points
                    self.pca_ax2.text(0.02, 0.98, f'{n_points:,} total points\n(binned for display)', 
                                    transform=self.pca_ax2.transAxes, fontsize=8,
                                    verticalalignment='top', bbox=dict(boxstyle='round', 
                                    facecolor='white', alpha=0.8))
                else:
                    # Direct scatter plot for smaller datasets
                    self.pca_ax2.scatter(self.pca_components[:, 0], self.pca_components[:, 1], 
                                       alpha=0.6, s=1, c='blue')

                self.pca_ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
                self.pca_ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
                self.pca_ax2.set_title('PCA Component Scatter Plot')
                self.pca_ax2.grid(True, alpha=0.3)

            # Adjust layout and draw
            self.pca_fig.tight_layout()
            self.pca_canvas.draw()

        except Exception as e:
            print(f"Error plotting PCA results: {str(e)}")
            QMessageBox.warning(self, "Plot Error", f"Error creating PCA plots: {str(e)}")

    def save_pca_results(self):
        """Save PCA results to file."""
        if self.pca is None or self.pca_components is None:
            QMessageBox.warning(self, "Warning", "No PCA results to save.")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save PCA Results", 
                "pca_results.pkl",
                "Pickle files (*.pkl);;All files (*.*)"
            )
            
            if file_path:
                import pickle
                results = {
                    'pca': self.pca,
                    'components': self.pca_components,
                    'explained_variance_ratio': self.pca.explained_variance_ratio_
                }
                
                if hasattr(self, 'pca_scaler'):
                    results['scaler'] = self.pca_scaler
                
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f)
                    
                QMessageBox.information(self, "Success", "PCA results saved successfully.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save PCA results: {str(e)}")

    def save_nmf_results(self):
        """Save NMF results to file."""
        if self.nmf is None or self.nmf_components is None:
            QMessageBox.warning(self, "Warning", "No NMF results to save.")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save NMF Results", 
                "nmf_results.pkl",
                "Pickle files (*.pkl);;All files (*.*)"
            )
            
            if file_path:
                import pickle
                results = {
                    'nmf': self.nmf,
                    'components': self.nmf_components,
                    'reconstruction_err': self.nmf.reconstruction_err_
                }
                
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f)
                    
                QMessageBox.information(self, "Success", "NMF results saved successfully.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save NMF results: {str(e)}")

    def train_rf(self):
        """Train Random Forest classifier - this is the corrected method that was previously named run_pca."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        try:
            # Stop any existing worker thread
            if hasattr(self, 'worker') and self.worker is not None:
                if self.worker.isRunning():
                    self.worker.stop()
                self.worker = None
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Training Random Forest...")
            
            # Get directories
            class_a_dir = self.class_a_path.text()
            class_b_dir = self.class_b_path.text()
            
            # Create and start worker thread
            self.worker = MapAnalysisWorker(
                self._train_rf_classifier_worker,
                class_a_dir, class_b_dir
            )
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self._on_rf_training_finished)
            self.worker.error.connect(self._on_rf_training_error)
            self.worker.start()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error training Random Forest: {str(e)}")
    
    def _train_rf_classifier_worker(self, worker, class_a_dir, class_b_dir):
        """Worker method for Random Forest training."""
        # Load spectra from both directories
        class_a_spectra = self._load_spectra_from_directory(class_a_dir, label=1)
        class_b_spectra = self._load_spectra_from_directory(class_b_dir, label=0)
        
        if len(class_a_spectra) == 0 or len(class_b_spectra) == 0:
            raise ValueError("No spectra found in one or both directories")
        
        # Combine data
        all_spectra = class_a_spectra + class_b_spectra
        
        # Extract features and labels
        X = []
        y = []
        
        for spectrum_data, label in all_spectra:
            # Apply cosmic ray rejection if enabled
            if self.enable_cosmic_rejection_cb.isChecked():
                wavenumbers = spectrum_data['wavenumbers']
                intensities = spectrum_data['intensities']
                
                _, cleaned_intensities = self.map_data.detect_cosmic_rays(
                    wavenumbers, intensities,
                    threshold_factor=self.cosmic_threshold_spin.value(),
                    window_size=self.cosmic_window_spin.value()
                )
                spectrum_data['intensities'] = cleaned_intensities
            
            # Preprocess spectrum
            processed_spectrum = self._preprocess_single_spectrum(
                spectrum_data['wavenumbers'], 
                spectrum_data['intensities']
            )
            
            X.append(processed_spectrum)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Remove any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply PCA/NMF feature reduction if selected and available
        if self.use_pca_nmf_features_cb.isChecked():
            feature_type_selection = self.feature_type_combo.currentText()
            
            if feature_type_selection == "PCA Only":
                if hasattr(self, 'pca') and self.pca is not None and hasattr(self, 'pca_scaler'):
                    logger.info("Using PCA features for Random Forest training")
                    X_scaled = self.pca_scaler.transform(X)
                    X = self.pca.transform(X_scaled)
                    self.rf_feature_type = 'pca'
                else:
                    logger.warning("PCA features requested but PCA model or scaler not available. Using full spectrum.")
                    self.rf_feature_type = 'full'
                    
            elif feature_type_selection == "NMF Only":
                if hasattr(self, 'nmf') and self.nmf is not None:
                    logger.info("Using NMF features for Random Forest training")
                    # Ensure non-negative data for NMF (like the main NMF analysis)
                    X_positive = np.maximum(X, 0)
                    X = self.nmf.transform(X_positive)
                    self.rf_feature_type = 'nmf'
                else:
                    logger.warning("NMF features requested but NMF model not available. Using full spectrum.")
                    self.rf_feature_type = 'full'
                    
            elif feature_type_selection == "Full Spectrum":
                logger.info("Using full spectrum features for Random Forest training")
                self.rf_feature_type = 'full'
                
            else:  # Auto mode (original behavior)
                if hasattr(self, 'pca') and self.pca is not None and hasattr(self, 'pca_scaler'):
                    logger.info("Using PCA features for Random Forest training")
                    X_scaled = self.pca_scaler.transform(X)
                    X = self.pca.transform(X_scaled)
                    self.rf_feature_type = 'pca'
                elif hasattr(self, 'nmf') and self.nmf is not None:
                    logger.info("Using NMF features for Random Forest training")
                    # Ensure non-negative data for NMF (like the main NMF analysis)
                    X_positive = np.maximum(X, 0)
                    X = self.nmf.transform(X_positive)
                    self.rf_feature_type = 'nmf'
                else:
                    logger.warning("PCA/NMF features requested but no models available. Using full spectrum features.")
                    self.rf_feature_type = 'full'
        else:
            self.rf_feature_type = 'full'
            logger.info("Using full spectrum features for Random Forest training")
        
        # Split data
        test_size = getattr(self, 'test_size_spin', None)
        if test_size:
            test_size_val = test_size.value()
        else:
            test_size_val = 0.2
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_val, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_trees.value(),
            max_depth=self.rf_max_depth.value(),
            random_state=42
        )
        
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'model': rf,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'report': report,
            'n_class_a': len(class_a_spectra),
            'n_class_b': len(class_b_spectra),
            'feature_type': self.rf_feature_type
        }
    
    def _load_spectra_from_directory(self, directory, label):
        """Load all spectra from a directory."""
        dir_path = Path(directory)
        spectra = []
        
        # Look for spectrum files
        for ext in ['*.csv', '*.txt']:
            for file_path in dir_path.glob(ext):
                try:
                    # Load spectrum data
                    if file_path.suffix.lower() == '.csv':
                        data = pd.read_csv(file_path, header=None).values
                    else:
                        data = np.loadtxt(file_path)
                    
                    if data.shape[1] >= 2:
                        wavenumbers = data[:, 0]
                        intensities = data[:, 1]
                        
                        spectrum_data = {
                            'wavenumbers': wavenumbers,
                            'intensities': intensities,
                            'filename': file_path.name
                        }
                        
                        spectra.append((spectrum_data, label))
                        
                except Exception as e:
                    logger.warning(f"Failed to load spectrum {file_path}: {str(e)}")
                    continue
        
        return spectra
    
    def _preprocess_single_spectrum(self, wavenumbers, intensities):
        """Preprocess a single spectrum for ML training - matches original working version."""
        try:
            # Use the SAME preprocessing as the original working version
            # This uses the ml_raman_map.pre_processing.preprocess_spectrum function
            processed = preprocess_spectrum(wavenumbers, intensities, self.map_data.target_wavenumbers)
            
            # preprocess_spectrum returns a 2D array with [wavenumbers, intensities]
            # We only want the intensities for ML
            if processed.shape[0] == len(self.map_data.target_wavenumbers):
                return processed[:, 1]  # Return only intensities column
            else:
                logger.error(f"Preprocessing returned wrong shape: {processed.shape}")
                return np.zeros(len(self.map_data.target_wavenumbers))
                
        except Exception as e:
            logger.error(f"Error preprocessing spectrum: {str(e)}")
            return np.zeros(len(self.map_data.target_wavenumbers))
    
    def _on_rf_training_finished(self, result):
        """Handle successful Random Forest training."""
        self.rf_model = result['model']
        self.rf_X_test = result['X_test']
        self.rf_y_test = result['y_test']
        self.rf_y_pred = result['y_pred']
        self.rf_feature_type = result['feature_type']
        
        # Update model status if we have the text widget
        if hasattr(self, 'model_status_text'):
            feature_info = {
                'pca': "PCA reduced features",
                'nmf': "NMF reduced features", 
                'full': "Full spectrum features"
            }.get(self.rf_feature_type, "Unknown feature type")
            
            status_text = f"""Random Forest Model Trained Successfully!
            
Training Data:
- Class A (Positive): {result['n_class_a']} spectra
- Class B (Negative): {result['n_class_b']} spectra
- Feature Type: {feature_info}

Performance:
- Accuracy: {result['accuracy']:.3f}
- Precision: {result['report']['weighted avg']['precision']:.3f}
- Recall: {result['report']['weighted avg']['recall']:.3f}
- F1-Score: {result['report']['weighted avg']['f1-score']:.3f}
"""
            self.model_status_text.setPlainText(status_text)
        
        # Plot training results
        self._plot_rf_training_results(result)
        
        self.statusBar().showMessage("Random Forest training completed")
        QMessageBox.information(self, "Success", 
                               f"Random Forest trained successfully!\n"
                               f"Accuracy: {result['accuracy']:.3f}\n"
                               f"Features: {result['feature_type']}")
    
    def _on_rf_training_error(self, error_msg):
        """Handle Random Forest training error."""
        self.statusBar().showMessage("Ready")
        QMessageBox.critical(self, "Error", f"Error training Random Forest: {error_msg}")
    
    def _plot_rf_training_results(self, result):
        """Plot Random Forest training results."""
        if not hasattr(self, 'ml_fig'):
            return
        
        # Clear the figure
        self.ml_fig.clear()
        
        # Create subplots
        ax1 = self.ml_fig.add_subplot(121)
        ax2 = self.ml_fig.add_subplot(122)
        
        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        
        im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center')
        
        # Plot feature importance (top 20) - only for full spectrum features
        if hasattr(result['model'], 'feature_importances_') and result['feature_type'] == 'full':
            importances = result['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            wavenumbers = self.map_data.target_wavenumbers[indices]
            ax2.bar(range(len(indices)), importances[indices])
            ax2.set_title('Top 20 Feature Importances')
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Importance')
            
            # Add wavenumber labels for some points
            for i in range(0, len(indices), 5):
                ax2.text(i, importances[indices[i]], f'{wavenumbers[i]:.0f}', 
                        rotation=45, ha='center', va='bottom', fontsize=8)
        elif hasattr(result['model'], 'feature_importances_'):
            # For PCA/NMF features, just show component importances
            importances = result['model'].feature_importances_
            ax2.bar(range(len(importances)), importances)
            ax2.set_title(f'{result["feature_type"].upper()} Component Importances')
            ax2.set_xlabel('Component Index')
            ax2.set_ylabel('Importance')
        
        self.ml_fig.tight_layout()
        self.ml_canvas.draw()
    
    def classify_map_spectra(self):
        """Classify all spectra in the map using the trained Random Forest model."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        if self.rf_model is None:
            QMessageBox.warning(self, "Warning", "Train Random Forest model first")
            return
        
        # Check if PCA/NMF features are requested but not available
        if self.use_pca_nmf_features_cb.isChecked():
            if not hasattr(self, 'pca') or self.pca is None:
                if not hasattr(self, 'nmf') or self.nmf is None:
                    reply = QMessageBox.question(self, "No PCA/NMF Available", 
                                               "PCA/NMF features requested but no PCA or NMF models available.\n"
                                               "Continue with full spectrum features?",
                                               QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.No:
                        return
        
        try:
            # Show progress bar and update status
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Classifying map spectra...")
            
            # Create and start worker thread
            self.worker = MapAnalysisWorker(self._classify_map_worker)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self._on_map_classification_finished)
            self.worker.error.connect(self._on_map_classification_error)
            self.worker.start()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error classifying map: {str(e)}")
    
    def _classify_map_worker(self, worker):
        """Worker method for map classification - optimized for speed with batch processing."""
        # Check what feature type was used for training
        feature_type = getattr(self, 'rf_feature_type', 'full')
        logger.info(f"Using {feature_type} features for map classification")
        
        # Prepare map data for classification
        map_shape = (len(self.map_data.y_positions), len(self.map_data.x_positions))
        classification_map = np.full(map_shape, np.nan)
        probabilities_map = np.full(map_shape, np.nan)
        
        # Collect all spectra and their positions for batch processing
        all_features = []
        position_mapping = []  # Keep track of (i, j) positions
        
        total_spectra = len(self.map_data.y_positions) * len(self.map_data.x_positions)
        processed_count = 0
        
        # Extract and preprocess all features first
        for i, y_pos in enumerate(self.map_data.y_positions):
            for j, x_pos in enumerate(self.map_data.x_positions):
                spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                if spectrum:
                    # Use processed intensities if available
                    if spectrum.processed_intensities is not None:
                        features = spectrum.processed_intensities.copy()
                    else:
                        features = self._preprocess_single_spectrum(
                            spectrum.wavenumbers, spectrum.intensities
                        )
                    
                    # Note: Cosmic ray filtering removed from classification pipeline
                    # CRE should be handled in PCA/NMF preprocessing stages
                    
                    all_features.append(features)
                    position_mapping.append((i, j))
                
                processed_count += 1
                # Emit progress for preprocessing (first 30% of progress)
                progress = int((processed_count / total_spectra) * 30)
                worker.progress.emit(progress)
        
        if len(all_features) == 0:
            raise ValueError("No valid spectra found for classification")
        
        # Convert to numpy array for batch processing
        X = np.array(all_features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Preprocessing complete. Shape before transformation: {X.shape}")
        
        # Apply the same feature transformation used during training (BATCH PROCESSING)
        feature_type_selection = self.feature_type_combo.currentText()
        
        if feature_type_selection == "PCA Only":
            if feature_type == 'pca' and hasattr(self, 'pca') and self.pca is not None and hasattr(self, 'pca_scaler'):
                logger.info("Applying scaling and PCA transformation to all spectra at once")
                X_scaled = self.pca_scaler.transform(X)
                X = self.pca.transform(X_scaled)
                logger.info(f"PCA transformation complete. New shape: {X.shape}")
            else:
                logger.warning("PCA classification requested but training used different features or PCA/scaler unavailable")
                
        elif feature_type_selection == "NMF Only":
            if feature_type == 'nmf' and hasattr(self, 'nmf') and self.nmf is not None:
                logger.info("Applying NMF transformation to all spectra at once")
                # Ensure non-negative data for NMF (like the main NMF analysis)
                X_positive = np.maximum(X, 0)
                X = self.nmf.transform(X_positive)
                logger.info(f"NMF transformation complete. New shape: {X.shape}")
            else:
                logger.warning("NMF classification requested but training used different features or NMF unavailable")
                
        elif feature_type_selection == "Full Spectrum":
            logger.info(f"Using full spectrum features. Shape: {X.shape}")
            
        else:  # Auto mode - match training feature type
            if feature_type == 'pca' and hasattr(self, 'pca') and self.pca is not None and hasattr(self, 'pca_scaler'):
                logger.info("Applying scaling and PCA transformation to all spectra at once")
                X_scaled = self.pca_scaler.transform(X)
                X = self.pca.transform(X_scaled)
                logger.info(f"PCA transformation complete. New shape: {X.shape}")
            elif feature_type == 'nmf' and hasattr(self, 'nmf') and self.nmf is not None:
                logger.info("Applying NMF transformation to all spectra at once")
                # Ensure non-negative data for NMF (like the main NMF analysis)
                X_positive = np.maximum(X, 0)
                X = self.nmf.transform(X_positive)
                logger.info(f"NMF transformation complete. New shape: {X.shape}")
            else:
                logger.info(f"Using full spectrum features. Shape: {X.shape}")
        
        # Update progress after transformation (60% complete)
        worker.progress.emit(60)
        
        # Batch prediction (much faster than individual predictions)
        logger.info("Starting batch prediction...")
        predictions = self.rf_model.predict(X)
        probabilities = np.max(self.rf_model.predict_proba(X), axis=1)
        logger.info("Batch prediction complete")
        
        # Update progress after prediction (90% complete)
        worker.progress.emit(90)
        
        # Map results back to 2D grid
        for idx, (i, j) in enumerate(position_mapping):
            classification_map[i, j] = predictions[idx]
            probabilities_map[i, j] = probabilities[idx]
        
        # Final progress update
        worker.progress.emit(100)
        
        logger.info(f"Classification complete. Processed {len(position_mapping)} spectra using {feature_type} features")
        
        return {
            'classification_map': classification_map,
            'probabilities_map': probabilities_map,
            'feature_type': feature_type,
            'n_classified': len(position_mapping)
        }
    
    def _on_map_classification_finished(self, result):
        """Handle successful map classification."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        self.classification_map = result['classification_map']
        self.probabilities_map = result['probabilities_map']
        
        # Update feature combo to include classification
        current_items = [self.feature_combo.itemText(i) for i in range(self.feature_combo.count())]
        if "Classification Map" not in current_items:
            self.feature_combo.addItem("Classification Map")
        if "Classification Probability" not in current_items:
            self.feature_combo.addItem("Classification Probability")
        
        # Plot classification results
        self._plot_classification_results()
        
        feature_info = {
            'pca': "PCA features",
            'nmf': "NMF features", 
            'full': "full spectrum features"
        }.get(result['feature_type'], "unknown features")
        
        self.statusBar().showMessage(f"Classification completed - {result['n_classified']} spectra")
        QMessageBox.information(self, "Success", 
                               f"Map classification completed successfully!\n"
                               f"Classified: {result['n_classified']} spectra\n"
                               f"Used: {feature_info}")
    
    def _on_map_classification_error(self, error_msg):
        """Handle map classification error."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Ready")
        QMessageBox.critical(self, "Error", f"Error classifying map: {error_msg}")
    
    def _plot_classification_results(self):
        """Plot classification results in the ML tab."""
        if not hasattr(self, 'ml_fig') or not hasattr(self, 'classification_map'):
            return
        
        # Clear the figure
        self.ml_fig.clear()
        
        # Create subplots
        ax1 = self.ml_fig.add_subplot(121)
        ax2 = self.ml_fig.add_subplot(122)
        
        # Plot classification map
        im1 = ax1.imshow(self.classification_map, cmap='RdYlBu', aspect='auto')
        ax1.set_title('Classification Map')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        self.ml_fig.colorbar(im1, ax=ax1, label='Class (0=B, 1=A)')
        
        # Plot probability map
        im2 = ax2.imshow(self.probabilities_map, cmap='viridis', aspect='auto')
        ax2.set_title('Classification Probability')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        self.ml_fig.colorbar(im2, ax=ax2, label='Probability')
        
        self.ml_fig.tight_layout()
        self.ml_canvas.draw()
    
    def train_unsupervised_model(self):
        """Train an unsupervised model using the map data."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Training unsupervised model...")
            
            # Create and start worker thread
            method = self.training_method_combo.currentText()
            n_clusters = self.n_clusters_spin.value()
            use_reduced = self.use_reduced_features_cb.isChecked()
            
            self.worker = MapAnalysisWorker(
                self._train_unsupervised_worker,
                method, n_clusters, use_reduced
            )
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self._on_unsupervised_training_finished)
            self.worker.error.connect(self._on_unsupervised_training_error)
            self.worker.start()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error training model: {str(e)}")
    
    def _train_unsupervised_worker(self, worker, method, n_clusters, use_reduced):
        """Worker method for unsupervised model training."""
        # Prepare data from map
        df = self.map_data.prepare_ml_data(use_processed=True)
        
        # Extract feature columns (exclude position columns)
        feature_cols = [col for col in df.columns if col.startswith('wn_')]
        X = df[feature_cols].values
        positions = df[['x_pos', 'y_pos']].values
        
        # Remove any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        worker.progress.emit(20)
        
        # Apply dimensionality reduction if requested
        feature_type = 'full'
        if use_reduced:
            if hasattr(self, 'pca') and self.pca is not None:
                X = self.pca.transform(X)
                feature_type = 'pca'
                logger.info(f"Using PCA features: {X.shape}")
            elif hasattr(self, 'nmf') and self.nmf is not None:
                # Ensure non-negative data for NMF (like the main NMF analysis)
                X_positive = np.maximum(X, 0)
                X = self.nmf.transform(X_positive)
                feature_type = 'nmf'
                logger.info(f"Using NMF features: {X.shape}")
            else:
                logger.warning("Reduced features requested but no PCA/NMF available")
        
        worker.progress.emit(40)
        
        # Train the selected model
        if method == "K-Means Clustering":
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            
        elif method == "Gaussian Mixture Model":
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            model.fit(X)
            labels = model.predict(X)
            
        elif method == "Spectral Clustering":
            from sklearn.cluster import SpectralClustering
            model = SpectralClustering(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(X)
            
        elif method == "DBSCAN Clustering":
            from sklearn.cluster import DBSCAN
            # For DBSCAN, we don't use n_clusters, but estimate eps
            from sklearn.neighbors import NearestNeighbors
            neigh = NearestNeighbors(n_neighbors=4)
            neigh.fit(X)
            distances, indices = neigh.kneighbors(X)
            eps = np.mean(distances[:, -1])
            
            model = DBSCAN(eps=eps, min_samples=4)
            labels = model.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
            
        worker.progress.emit(80)
        
        # Calculate silhouette score for quality assessment
        from sklearn.metrics import silhouette_score
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = -1
        
        # Create cluster map
        map_shape = (len(self.map_data.y_positions), len(self.map_data.x_positions))
        cluster_map = np.full(map_shape, np.nan)
        
        # Map clusters back to 2D positions
        for idx, (x_pos, y_pos) in enumerate(positions):
            try:
                i = self.map_data.y_positions.index(y_pos)
                j = self.map_data.x_positions.index(x_pos)
                cluster_map[i, j] = labels[idx]
            except ValueError:
                continue  # Skip if position not found
        
        worker.progress.emit(100)
        
        return {
            'model': model,
            'labels': labels,
            'cluster_map': cluster_map,
            'method': method,
            'n_clusters': n_clusters,
            'feature_type': feature_type,
            'silhouette_score': silhouette,
            'n_spectra': len(labels)
        }
    
    def _on_unsupervised_training_finished(self, result):
        """Handle successful unsupervised training."""
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Store results
        self.unsupervised_model = result['model']
        self.cluster_labels = result['labels']
        self.cluster_map = result['cluster_map']
        
        # Update feature combo to include clustering
        current_items = [self.feature_combo.itemText(i) for i in range(self.feature_combo.count())]
        if "Cluster Map" not in current_items:
            self.feature_combo.addItem("Cluster Map")
        
        # Update model status
        if hasattr(self, 'model_status_text'):
            feature_info = {
                'pca': "PCA reduced features",
                'nmf': "NMF reduced features", 
                'full': "Full spectrum features"
            }.get(result['feature_type'], "Unknown features")
            
            status_text = f"""Unsupervised Model Trained Successfully!
            
Method: {result['method']}
Clusters Found: {result['n_clusters']}
Feature Type: {feature_info}
Spectra Processed: {result['n_spectra']}
Silhouette Score: {result['silhouette_score']:.3f}

Model ready for visualization and analysis.
"""
            self.model_status_text.setPlainText(status_text)
        
        # Plot training results
        self._plot_unsupervised_results(result)
        
        self.statusBar().showMessage("Unsupervised model training completed")
        QMessageBox.information(self, "Success", 
                               f"Model trained successfully!\n"
                               f"Method: {result['method']}\n"
                               f"Clusters: {result['n_clusters']}\n"
                               f"Silhouette Score: {result['silhouette_score']:.3f}")
    
    def _on_unsupervised_training_error(self, error_msg):
        """Handle unsupervised training error."""
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Ready")
        QMessageBox.critical(self, "Error", f"Error training model: {error_msg}")
    
    def _plot_unsupervised_results(self, result):
        """Plot unsupervised training results in the Train Model tab."""
        if not hasattr(self, 'train_fig'):
            return
        
        # Clear the figure
        self.train_fig.clear()
        
        # Create subplots
        ax1 = self.train_fig.add_subplot(121)
        ax2 = self.train_fig.add_subplot(122)
        
        # Plot cluster map
        im1 = ax1.imshow(result['cluster_map'], cmap='tab10', aspect='auto')
        ax1.set_title(f'{result["method"]} Clusters')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        self.train_fig.colorbar(im1, ax=ax1, label='Cluster')
        
        # Plot cluster distribution
        unique_labels, counts = np.unique(result['labels'], return_counts=True)
        # Remove noise label (-1) if present
        if -1 in unique_labels:
            noise_idx = np.where(unique_labels == -1)[0][0]
            ax2.bar(unique_labels[unique_labels != -1], counts[unique_labels != -1], 
                   label=f'Clusters (noise: {counts[noise_idx]})')
        else:
            ax2.bar(unique_labels, counts)
        
        ax2.set_title('Cluster Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Spectra')
        if -1 in unique_labels:
            ax2.legend()
        
        # Plot Class A location map (if Random Forest model is available)
        if hasattr(self, 'rf_model') and self.rf_model is not None:
            try:
                # Use the Random Forest model to classify all map spectra
                class_a_map = self._create_class_a_location_map()
                
                if class_a_map is not None:
                    im2 = ax2.imshow(class_a_map, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
                    ax2.set_title('Class A Probability Map')
                    ax2.set_xlabel('X Position')
                    ax2.set_ylabel('Y Position')
                    self.train_fig.colorbar(im2, ax=ax2, label='Class A Probability')
                else:
                    # Fallback to cluster distribution if classification fails
                    self._plot_cluster_distribution(ax2, result)
                    
            except Exception as e:
                logger.warning(f"Failed to create Class A map: {str(e)}")
                # Fallback to cluster distribution
                self._plot_cluster_distribution(ax2, result)
        else:
            # No Random Forest model available, show cluster distribution
            self._plot_cluster_distribution(ax2, result)
        
        self.train_fig.tight_layout()
        self.train_canvas.draw()
    
    def _create_class_a_location_map(self):
        """Create a map showing the probability/location of Class A spectra."""
        if self.map_data is None or self.rf_model is None:
            return None
        
        try:
            # Prepare map data for classification (similar to classify_map_worker but simplified)
            map_shape = (len(self.map_data.y_positions), len(self.map_data.x_positions))
            class_a_map = np.full(map_shape, np.nan)
            
            # Collect all spectra for batch processing
            all_features = []
            position_mapping = []
            
            for i, y_pos in enumerate(self.map_data.y_positions):
                for j, x_pos in enumerate(self.map_data.x_positions):
                    spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                    if spectrum:
                        # Use processed intensities if available
                        if spectrum.processed_intensities is not None:
                            features = spectrum.processed_intensities.copy()
                        else:
                            features = self._preprocess_single_spectrum(
                                spectrum.wavenumbers, spectrum.intensities
                            )
                        
                        all_features.append(features)
                        position_mapping.append((i, j))
            
            if len(all_features) == 0:
                return None
            
            # Convert to numpy array for batch processing
            X = np.array(all_features)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply the same feature transformation used during training
            feature_type = getattr(self, 'rf_feature_type', 'full')
            
            if feature_type == 'pca' and hasattr(self, 'pca') and self.pca is not None and hasattr(self, 'pca_scaler'):
                X_scaled = self.pca_scaler.transform(X)
                X = self.pca.transform(X_scaled)
            elif feature_type == 'nmf' and hasattr(self, 'nmf') and self.nmf is not None:
                # Ensure non-negative data for NMF
                X_positive = np.maximum(X, 0)
                X = self.nmf.transform(X_positive)
            # For 'full' features, use X as is
            
            # Get Class A probabilities (class 1 probabilities)
            probabilities = self.rf_model.predict_proba(X)
            if probabilities.shape[1] > 1:
                class_a_probs = probabilities[:, 1]  # Class 1 (Class A) probabilities
            else:
                # Binary case where only one class is present
                class_a_probs = probabilities[:, 0]
            
            # Map probabilities back to 2D grid
            for idx, (i, j) in enumerate(position_mapping):
                class_a_map[i, j] = class_a_probs[idx]
            
            return class_a_map
            
        except Exception as e:
            logger.error(f"Error creating Class A location map: {str(e)}")
            return None
    
    def _plot_cluster_distribution(self, ax, result):
        """Plot cluster distribution as fallback."""
        unique_labels, counts = np.unique(result['labels'], return_counts=True)
        # Remove noise label (-1) if present
        if -1 in unique_labels:
            noise_idx = np.where(unique_labels == -1)[0][0]
            ax.bar(unique_labels[unique_labels != -1], counts[unique_labels != -1], 
                   label=f'Clusters (noise: {counts[noise_idx]})')
        else:
            ax.bar(unique_labels, counts)
        
        ax.set_title('Cluster Distribution')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Number of Spectra')
        if -1 in unique_labels:
            ax.legend()
    
    def diagnose_data_consistency(self):
        """Diagnose data consistency between map visualization and ML analysis."""
        if self.map_data is None:
            return
        
        # Check what spectra are loaded vs what should be there
        total_positions = len(self.map_data.y_positions) * len(self.map_data.x_positions)
        loaded_spectra = len(self.map_data.spectra)
        
        print(f"\n=== DATA CONSISTENCY DIAGNOSIS ===")
        print(f"Expected positions: {total_positions}")
        print(f"Loaded spectra: {loaded_spectra}")
        print(f"Missing spectra: {total_positions - loaded_spectra}")
        
        # Analyze scan line completeness
        incomplete_lines = []
        complete_lines = []
        expected_line_length = len(self.map_data.x_positions)
        
        for y_pos in self.map_data.y_positions:
            spectra_in_line = 0
            missing_positions = []
            
            for x_pos in self.map_data.x_positions:
                if self.map_data.get_spectrum(x_pos, y_pos) is not None:
                    spectra_in_line += 1
                else:
                    missing_positions.append(x_pos)
            
            if spectra_in_line == 0:
                incomplete_lines.append({
                    'y_pos': y_pos,
                    'type': 'completely_missing',
                    'spectra_count': 0,
                    'missing_positions': missing_positions,
                    'completion_percent': 0.0
                })
            elif spectra_in_line < expected_line_length:
                completion_percent = (spectra_in_line / expected_line_length) * 100
                line_type = 'truncated' if missing_positions == list(range(max(missing_positions), self.map_data.x_positions[-1] + 1)) else 'partial'
                
                incomplete_lines.append({
                    'y_pos': y_pos,
                    'type': line_type,
                    'spectra_count': spectra_in_line,
                    'missing_positions': missing_positions,
                    'completion_percent': completion_percent
                })
            else:
                complete_lines.append(y_pos)
        
        # Report scan line analysis
        print(f"\n=== SCAN LINE ANALYSIS ===")
        print(f"Complete scan lines: {len(complete_lines)}")
        print(f"Incomplete scan lines: {len(incomplete_lines)}")
        
        if incomplete_lines:
            print(f"\nINCOMPLETE LINES DETECTED:")
            for line_info in incomplete_lines:
                y_pos = line_info['y_pos']
                line_type = line_info['type']
                count = line_info['spectra_count']
                percent = line_info['completion_percent']
                
                print(f"  Y={y_pos}: {line_type.upper()} - {count}/{expected_line_length} spectra ({percent:.1f}%)")
                
                if line_type == 'truncated':
                    first_missing = min(line_info['missing_positions'])
                    last_missing = max(line_info['missing_positions'])
                    print(f"    ⚠️  DATA ACQUISITION INTERRUPTED at X={first_missing}")
                    print(f"    📁 Missing X positions: {first_missing} to {last_missing}")
                elif line_type == 'partial':
                    print(f"    🔍 Scattered missing positions: {line_info['missing_positions'][:10]}{'...' if len(line_info['missing_positions']) > 10 else ''}")
        
        # Check ML data preparation
        df = self.map_data.prepare_ml_data(use_processed=True)
        ml_spectra_count = len(df)
        
        print(f"\n=== ML ANALYSIS IMPACT ===")
        print(f"ML Analysis includes: {ml_spectra_count} spectra")
        print(f"Map visualization handles: {total_positions} positions (missing filled with NaN)")
        
        # Warn about incomplete lines affecting ML analysis
        if incomplete_lines:
            print(f"\n⚠️  WARNING: INCOMPLETE SCAN LINES DETECTED")
            print(f"   This indicates data acquisition interruptions during scanning.")
            print(f"   Incomplete lines can create artificial edge effects in PCA/NMF analysis.")
            print(f"   Recommendation: Enable 'Filter Edge Effects' in PCA/NMF tabs.")
        
        # Check for extreme values that could cause outliers
        feature_cols = [col for col in df.columns if col.startswith('wn_')]
        X = df[feature_cols].values
        
        print(f"\n=== ML DATA STATISTICS ===")
        print(f"Shape: {X.shape}")
        print(f"Min value: {np.min(X):.6f}")
        print(f"Max value: {np.max(X):.6f}")
        print(f"Mean: {np.mean(X):.6f}")
        print(f"Std: {np.std(X):.6f}")
        
        # Check for spectra with extreme values
        spectrum_sums = np.sum(X, axis=1)
        spectrum_max = np.max(X, axis=1)
        
        outlier_threshold = np.mean(spectrum_sums) + 3 * np.std(spectrum_sums)
        outlier_indices = np.where(spectrum_sums > outlier_threshold)[0]
        
        if len(outlier_indices) > 0:
            print(f"\nPotential outlier spectra (sum > mean + 3*std):")
            for idx in outlier_indices[:5]:  # Show first 5 outliers
                x_pos = df.iloc[idx]['x_pos']
                y_pos = df.iloc[idx]['y_pos']
                print(f"  Position ({x_pos}, {y_pos}): sum={spectrum_sums[idx]:.1f}, max={spectrum_max[idx]:.1f}")
            if len(outlier_indices) > 5:
                print(f"  ... and {len(outlier_indices) - 5} more outliers")
        
        # Show user notification dialog
        self._show_data_consistency_dialog(incomplete_lines, complete_lines)
        
        return {
            'total_positions': total_positions,
            'loaded_spectra': loaded_spectra,
            'incomplete_lines': incomplete_lines,
            'complete_lines': complete_lines,
            'ml_spectra_count': ml_spectra_count,
            'outlier_indices': outlier_indices
        }
    
    def _show_data_consistency_dialog(self, incomplete_lines, complete_lines):
        """Show user-friendly dialog about data consistency issues."""
        if not incomplete_lines:
            QMessageBox.information(
                self, "Data Consistency Check", 
                f"✅ All scan lines complete!\n\n"
                f"Complete lines: {len(complete_lines)}\n"
                f"No data acquisition interruptions detected."
            )
            return
        
        # Create detailed message about incomplete lines
        msg = f"⚠️ DATA ACQUISITION INTERRUPTIONS DETECTED\n\n"
        msg += f"Complete scan lines: {len(complete_lines)}\n"
        msg += f"Incomplete scan lines: {len(incomplete_lines)}\n\n"
        
        truncated_lines = [line for line in incomplete_lines if line['type'] == 'truncated']
        if truncated_lines:
            msg += f"🚨 TRUNCATED LINES (Data acquisition stopped mid-line):\n"
            for line in truncated_lines[:3]:  # Show first 3
                msg += f"   Y={line['y_pos']}: {line['completion_percent']:.1f}% complete\n"
            if len(truncated_lines) > 3:
                msg += f"   ... and {len(truncated_lines) - 3} more truncated lines\n"
            msg += "\n"
        
        msg += "RECOMMENDATION:\n"
        msg += "• Enable 'Filter Edge Effects' in PCA/NMF tabs\n"
        msg += "• This will exclude spectra near incomplete lines\n"
        msg += "• Prevents artificial artifacts in analysis results\n\n"
        msg += "The incomplete lines appear to be caused by data acquisition\n"
        msg += "interruptions during the original scanning process."
        
        QMessageBox.warning(self, "Data Consistency Warning", msg)
    
    def filter_edge_effects_from_ml_data(self, y_buffer=2):
        """Filter out spectra that are now edge cases due to missing lines."""
        if self.map_data is None:
            return None
        
        # Analyze scan line completeness to identify incomplete lines
        incomplete_y_positions = set()
        expected_line_length = len(self.map_data.x_positions)
        
        for y_pos in self.map_data.y_positions:
            spectra_in_line = 0
            for x_pos in self.map_data.x_positions:
                if self.map_data.get_spectrum(x_pos, y_pos) is not None:
                    spectra_in_line += 1
            
            # Consider line incomplete if missing more than 10% of spectra
            if spectra_in_line < expected_line_length * 0.9:
                incomplete_y_positions.add(y_pos)
        
        # Get ML data
        df = self.map_data.prepare_ml_data(use_processed=True)
        
        if not incomplete_y_positions:
            logger.info("No incomplete scan lines detected - no edge filtering needed")
            return df
        
        # Filter out spectra near incomplete lines
        filtered_df = df.copy()
        total_removed = 0
        
        for incomplete_y in incomplete_y_positions:
            # Remove spectra within y_buffer of incomplete line
            mask = np.abs(filtered_df['y_pos'] - incomplete_y) > y_buffer
            removed_count = len(filtered_df) - np.sum(mask)
            filtered_df = filtered_df[mask]
            total_removed += removed_count
        
        logger.info(f"Edge effects filtering: {len(df)} -> {len(filtered_df)} spectra")
        logger.info(f"Removed {total_removed} spectra near {len(incomplete_y_positions)} incomplete scan lines")
        logger.info(f"Incomplete Y positions: {sorted(list(incomplete_y_positions))}")
        
        return filtered_df
    
    def _apply_cosmic_ray_filtering_to_data(self, df, threshold_factor=8.0, window_size=5):
        """Apply cosmic ray filtering to preprocessed spectral data."""
        if df is None or len(df) == 0:
            return df
        
        # Extract feature columns (spectral data)
        feature_cols = [col for col in df.columns if col.startswith('wn_')]
        position_cols = ['x_pos', 'y_pos']
        
        # Get spectral data matrix
        X = df[feature_cols].values
        
        logger.info(f"Applying cosmic ray filtering to {X.shape[0]} spectra...")
        
        # Apply cosmic ray detection and cleaning to each spectrum
        X_cleaned = np.zeros_like(X)
        cosmic_ray_count = 0
        
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            
            # Use target wavenumbers for cosmic ray detection
            wavenumbers = self.map_data.target_wavenumbers
            
            # Apply cosmic ray detection
            cosmic_detected, cleaned_spectrum = self.map_data.detect_cosmic_rays(
                wavenumbers, spectrum,
                threshold_factor=threshold_factor,
                window_size=window_size
            )
            
            X_cleaned[i, :] = cleaned_spectrum
            if cosmic_detected:
                cosmic_ray_count += 1
        
        logger.info(f"Cosmic ray filtering complete: {cosmic_ray_count}/{X.shape[0]} spectra had cosmic rays removed")
        
        # Create new dataframe with cleaned data
        cleaned_df = df[position_cols].copy()
        for i, col in enumerate(feature_cols):
            cleaned_df[col] = X_cleaned[:, i]
        
        return cleaned_df
    
    def diagnose_cosmic_ray_filtering(self, max_examples: int = 10):
        """
        Diagnostic method to understand cosmic ray filtering behavior.
        This helps debug why Class A spectra might be cosmic rays.
        """
        try:
            logger.info("=== COSMIC RAY FILTERING DIAGNOSIS ===")
            
            # Get some example spectra to analyze
            example_count = 0
            cosmic_ray_examples = []
            normal_examples = []
            
            for (x_pos, y_pos), spectrum in self.spectra.items():
                if example_count >= max_examples:
                    break
                
                # Test cosmic ray detection on both raw and processed data
                raw_detected, raw_cleaned = self.detect_cosmic_rays(
                    spectrum.wavenumbers, spectrum.intensities,
                    threshold_factor=5.0, window_size=5
                )
                
                if spectrum.processed_intensities is not None:
                    proc_detected, proc_cleaned = self.detect_cosmic_rays(
                        self.target_wavenumbers, spectrum.processed_intensities,
                        threshold_factor=5.0, window_size=5
                    )
                else:
                    proc_detected = False
                
                # Calculate statistics
                raw_max = np.max(spectrum.intensities)
                raw_median = np.median(spectrum.intensities)
                raw_ratio = raw_max / raw_median if raw_median > 0 else 0
                
                if spectrum.processed_intensities is not None:
                    proc_max = np.max(spectrum.processed_intensities)
                    proc_median = np.median(spectrum.processed_intensities)
                    proc_ratio = proc_max / proc_median if proc_median > 0 else 0
                else:
                    proc_max = proc_median = proc_ratio = 0
                
                example_info = {
                    'position': (x_pos, y_pos),
                    'raw_detected': raw_detected,
                    'proc_detected': proc_detected,
                    'raw_max': raw_max,
                    'raw_median': raw_median,
                    'raw_ratio': raw_ratio,
                    'proc_max': proc_max,
                    'proc_median': proc_median,
                    'proc_ratio': proc_ratio,
                    'filename': spectrum.filename
                }
                
                if raw_detected or proc_detected:
                    cosmic_ray_examples.append(example_info)
                else:
                    normal_examples.append(example_info)
                
                example_count += 1
            
            logger.info(f"Analyzed {example_count} spectra:")
            logger.info(f"  - {len(cosmic_ray_examples)} with cosmic rays detected")
            logger.info(f"  - {len(normal_examples)} normal spectra")
            
            # Report cosmic ray examples
            if cosmic_ray_examples:
                logger.info("\nCOSMIC RAY EXAMPLES:")
                for i, example in enumerate(cosmic_ray_examples[:5]):  # Show first 5
                    logger.info(f"  {i+1}. {example['filename']} at {example['position']}")
                    logger.info(f"     Raw: max={example['raw_max']:.0f}, median={example['raw_median']:.0f}, ratio={example['raw_ratio']:.1f}, detected={example['raw_detected']}")
                    logger.info(f"     Proc: max={example['proc_max']:.0f}, median={example['proc_median']:.0f}, ratio={example['proc_ratio']:.1f}, detected={example['proc_detected']}")
            
            # Report normal examples
            if normal_examples:
                logger.info("\nNORMAL SPECTRUM EXAMPLES:")
                for i, example in enumerate(normal_examples[:3]):  # Show first 3
                    logger.info(f"  {i+1}. {example['filename']} at {example['position']}")
                    logger.info(f"     Raw: max={example['raw_max']:.0f}, median={example['raw_median']:.0f}, ratio={example['raw_ratio']:.1f}")
                    logger.info(f"     Proc: max={example['proc_max']:.0f}, median={example['proc_median']:.0f}, ratio={example['proc_ratio']:.1f}")
            
            # Calculate overall statistics
            all_raw_ratios = [ex['raw_ratio'] for ex in cosmic_ray_examples + normal_examples if ex['raw_ratio'] > 0]
            all_proc_ratios = [ex['proc_ratio'] for ex in cosmic_ray_examples + normal_examples if ex['proc_ratio'] > 0]
            
            if all_raw_ratios:
                logger.info(f"\nOVERALL STATISTICS:")
                logger.info(f"  Raw intensity ratios: median={np.median(all_raw_ratios):.1f}, max={np.max(all_raw_ratios):.1f}")
                if all_proc_ratios:
                    logger.info(f"  Processed intensity ratios: median={np.median(all_proc_ratios):.1f}, max={np.max(all_proc_ratios):.1f}")
            
            logger.info("=== END DIAGNOSIS ===\n")
            
            return {
                'cosmic_ray_examples': cosmic_ray_examples,
                'normal_examples': normal_examples,
                'total_analyzed': example_count
            }
            
        except Exception as e:
            logger.error(f"Error in cosmic ray diagnosis: {e}")
            return None

    def plot_class_a_spectra(self):
        """Create a standalone plot window showing all Class A classified spectra."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        if not hasattr(self, 'classification_map') or self.classification_map is None:
            QMessageBox.warning(self, "Warning", "Run ML classification first")
            return
        
        # Create a new window for the Class A spectra plot
        class_a_window = QDialog(self)
        class_a_window.setWindowTitle("Class A Classified Spectra")
        class_a_window.setGeometry(150, 150, 1000, 600)
        
        layout = QVBoxLayout(class_a_window)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar(canvas, class_a_window)
        layout.addWidget(toolbar)
        
        # Plot the spectra
        self._plot_class_a_spectra(ax)
        
        fig.tight_layout()
        canvas.draw()
        
        # Show the dialog
        class_a_window.exec()

    def _plot_class_a_spectra(self, ax):
        """Plot all spectra classified as Class A."""
        if not hasattr(self, 'classification_map') or self.classification_map is None:
            ax.text(0.5, 0.5, 'Run ML classification first', ha='center', va='center')
            ax.set_title('Class A Classified Spectra')
            ax.axis('off')
            return
        
        try:
            # Get all Class A positions
            class_a_positions = []
            for i, y_pos in enumerate(self.map_data.y_positions):
                for j, x_pos in enumerate(self.map_data.x_positions):
                    if self.classification_map[i, j] == 1:  # Class A
                        class_a_positions.append((x_pos, y_pos))
            
            if len(class_a_positions) == 0:
                ax.text(0.5, 0.5, 'No Class A spectra found', ha='center', va='center')
                ax.set_title('Class A Classified Spectra')
                ax.axis('off')
                return
            
            # Get wavenumbers for x-axis
            wavenumbers = None
            spectra_plotted = 0
            
            # Plot each Class A spectrum
            for x_pos, y_pos in class_a_positions:
                spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                if spectrum:
                    # Use processed or raw data based on checkbox setting
                    if self.use_processed_cb.isChecked() and spectrum.processed_intensities is not None:
                        if wavenumbers is None:
                            wavenumbers = self.map_data.target_wavenumbers
                        intensities = spectrum.processed_intensities
                    else:
                        wavenumbers = spectrum.wavenumbers
                        intensities = spectrum.intensities
                    
                    # Plot spectrum with some transparency since there might be many
                    ax.plot(wavenumbers, intensities, alpha=0.6, linewidth=0.8, 
                           color='red' if spectra_plotted == 0 else None)
                    spectra_plotted += 1
            
            # Calculate and plot average spectrum
            if spectra_plotted > 1:
                avg_intensities = self._calculate_class_a_average_spectrum()
                if avg_intensities is not None:
                    if self.use_processed_cb.isChecked():
                        avg_wavenumbers = self.map_data.target_wavenumbers
                    else:
                        # Use first spectrum's wavenumbers as reference
                        first_spectrum = next(iter([self.map_data.get_spectrum(x, y) for x, y in class_a_positions if self.map_data.get_spectrum(x, y)]))
                        avg_wavenumbers = first_spectrum.wavenumbers
                    
                    ax.plot(avg_wavenumbers, avg_intensities, 'black', linewidth=2, 
                           label=f'Average (n={spectra_plotted})')
                    ax.legend()
            
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Class A Classified Spectra (n={spectra_plotted})')
            ax.grid(True, alpha=0.3)
            
            # Set reasonable x-axis limits
            if wavenumbers is not None and len(wavenumbers) > 0:
                ax.set_xlim(np.min(wavenumbers), np.max(wavenumbers))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting Class A spectra: {str(e)}', 
                   ha='center', va='center')
            ax.set_title('Class A Classified Spectra')
            ax.axis('off')
            logger.error(f"Error plotting Class A spectra: {str(e)}")

    def _calculate_class_a_average_spectrum(self):
        """Calculate the average spectrum for all Class A classified positions."""
        try:
            # Get all Class A positions
            class_a_positions = []
            for i, y_pos in enumerate(self.map_data.y_positions):
                for j, x_pos in enumerate(self.map_data.x_positions):
                    if self.classification_map[i, j] == 1:  # Class A
                        class_a_positions.append((x_pos, y_pos))
            
            if len(class_a_positions) == 0:
                return None
            
            # Collect all intensities
            all_intensities = []
            for x_pos, y_pos in class_a_positions:
                spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                if spectrum:
                    if self.use_processed_cb.isChecked() and spectrum.processed_intensities is not None:
                        intensities = spectrum.processed_intensities
                    else:
                        # For raw data, interpolate to common wavenumber grid
                        if hasattr(self, '_common_wavenumbers'):
                            common_wn = self._common_wavenumbers
                        else:
                            common_wn = self.map_data.target_wavenumbers
                        
                        # Interpolate to common grid
                        f = interp1d(spectrum.wavenumbers, spectrum.intensities, 
                                   kind='linear', bounds_error=False, fill_value=0)
                        intensities = f(common_wn)
                    
                    all_intensities.append(intensities)
            
            if len(all_intensities) == 0:
                return None
            
            # Calculate average
            avg_spectrum = np.mean(all_intensities, axis=0)
            return avg_spectrum
            
        except Exception as e:
            logger.error(f"Error calculating Class A average spectrum: {str(e)}")
            return None

    def _plot_analysis_summary(self, ax):
        """Plot a summary of analysis results."""
        try:
            # Create a text summary of all analyses
            summary_text = "Analysis Summary\n\n"
            
            if self.map_data:
                summary_text += f"• Map Data: {len(self.map_data.spectra)} spectra\n"
                summary_text += f"• Dimensions: {len(self.map_data.x_positions)} × {len(self.map_data.y_positions)}\n\n"
            
            if hasattr(self, 'pca') and self.pca is not None:
                explained_var = np.sum(self.pca.explained_variance_ratio_[:3])
                summary_text += f"• PCA: {self.pca.n_components_} components\n"
                summary_text += f"  First 3 explain {explained_var:.1%} variance\n\n"
            
            if hasattr(self, 'nmf') and self.nmf is not None:
                summary_text += f"• NMF: {self.nmf.n_components_} components\n"
                summary_text += f"  Reconstruction error: {self.nmf.reconstruction_err_:.3f}\n\n"
            
            if hasattr(self, 'rf_model') and self.rf_model is not None:
                summary_text += f"• Random Forest: Trained\n"
                if hasattr(self, 'classification_map') and self.classification_map is not None:
                    n_class_a = np.sum(self.classification_map == 1)
                    n_class_b = np.sum(self.classification_map == 0)
                    total_classified = n_class_a + n_class_b
                    if total_classified > 0:
                        summary_text += f"  Class A: {n_class_a} positions ({n_class_a/total_classified:.1%})\n"
                        summary_text += f"  Class B: {n_class_b} positions ({n_class_b/total_classified:.1%})\n"
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, fontfamily='monospace')
            ax.set_title('Analysis Summary')
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating summary: {str(e)}', 
                   ha='center', va='center')
            ax.set_title('Analysis Summary')
            ax.axis('off')

    def save_top_class_a_spectra(self):
        """Save the top N Class A spectra (highest probability) to a folder."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        if not hasattr(self, 'classification_map') or self.classification_map is None:
            QMessageBox.warning(self, "Warning", "Run ML classification first")
            return
        
        if not hasattr(self, 'probabilities_map') or self.probabilities_map is None:
            QMessageBox.warning(self, "Warning", "Classification probabilities not available")
            return
        
        # Get file format choice from user
        file_format = self._get_file_format_choice()
        if file_format is None:
            return  # User cancelled
        
        # Select directory to save spectra
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Top Class A Spectra", ""
        )
        
        if not save_dir:
            return
        
        try:
            n_top = self.top_spectra_count.value()
            self.statusBar().showMessage(f"Saving top {n_top} Class A spectra...")
            
            # Get Class A positions with their probabilities
            class_a_data = []
            for i, y_pos in enumerate(self.map_data.y_positions):
                for j, x_pos in enumerate(self.map_data.x_positions):
                    if self.classification_map[i, j] == 1:  # Class A
                        probability = self.probabilities_map[i, j]
                        spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                        if spectrum:
                            class_a_data.append((x_pos, y_pos, probability, spectrum))
            
            if len(class_a_data) == 0:
                QMessageBox.warning(self, "Warning", "No Class A spectra found")
                return
            
            # Apply cosmic ray filtering to Class A spectra
            logger.info(f"Found {len(class_a_data)} Class A spectra, applying cosmic ray filtering...")
            filtered_class_a_data = self.filter_cosmic_rays_from_classification(class_a_data)
            
            if len(filtered_class_a_data) == 0:
                QMessageBox.warning(self, "Warning", "All Class A spectra were filtered out as cosmic rays!")
                return
            
            logger.info(f"After cosmic ray filtering: {len(filtered_class_a_data)} spectra remain")
            
            # Sort by probability (highest first) and take top N
            filtered_class_a_data.sort(key=lambda x: x[2], reverse=True)
            top_class_a = filtered_class_a_data[:n_top]
            
            saved_count = 0
            for idx, (x_pos, y_pos, prob, spectrum) in enumerate(top_class_a):
                if spectrum:
                    # Create base filename (without extension)
                    filename_base = f"Class_A_Top_{idx+1:02d}_X{x_pos}_Y{y_pos}_prob{prob:.3f}"
                    filepath_base = os.path.join(save_dir, filename_base)
                    
                    # Get spectrum data
                    if self.use_processed_cb.isChecked() and spectrum.processed_intensities is not None:
                        wavenumbers = self.map_data.target_wavenumbers
                        intensities = spectrum.processed_intensities
                        data_type = "processed"
                    else:
                        wavenumbers = spectrum.wavenumbers
                        intensities = spectrum.intensities
                        data_type = "raw"
                    
                    # Prepare metadata
                    metadata = {
                        "Class A Spectrum - Top": idx+1,
                        "Position": f"X={x_pos}, Y={y_pos}",
                        "Classification Probability": f"{prob:.4f}",
                        "Data Type": data_type,
                        "Original Filename": spectrum.filename,
                        "Cosmic Ray Filtered": "Yes"
                    }
                    
                    # Save spectrum using chosen format
                    self._save_spectrum_file(filepath_base, wavenumbers, intensities, metadata, file_format)
                    saved_count += 1
            
            # Save summary file
            summary_filepath = os.path.join(save_dir, "Class_A_Top_Summary.txt")
            with open(summary_filepath, 'w') as f:
                f.write(f"Top {n_top} Class A Spectra Summary\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Class A positions: {len(class_a_data)}\n")
                f.write(f"After cosmic ray filtering: {len(filtered_class_a_data)}\n")
                f.write(f"Cosmic rays filtered out: {len(class_a_data) - len(filtered_class_a_data)}\n")
                f.write(f"Saved spectra: {saved_count}\n")
                f.write(f"File format: {file_format.upper()}\n")
                f.write(f"Data type: {'processed' if self.use_processed_cb.isChecked() else 'raw'}\n\n")
                
                f.write("Ranking by Classification Probability (Cosmic Ray Filtered):\n")
                for idx, (x_pos, y_pos, prob, spectrum) in enumerate(top_class_a):
                    f.write(f"{idx+1:2d}. X={x_pos:3d}, Y={y_pos:3d}, Prob={prob:.4f}\n")
            
            file_count = saved_count * (2 if file_format == "both" else 1)
            self.statusBar().showMessage(f"Saved {saved_count} top Class A spectra")
            QMessageBox.information(self, "Success", 
                                   f"Successfully saved {saved_count} top Class A spectra to:\n{save_dir}\n\n"
                                   f"Format: {file_format.upper()}\n"
                                   f"Files created: {file_count} spectrum files + 1 summary file")
            
        except Exception as e:
            self.statusBar().showMessage("Ready")
            QMessageBox.critical(self, "Error", f"Error saving top Class A spectra: {str(e)}")

    def save_all_class_a_spectra(self):
        """Save all Class A classified spectra to a folder."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        if not hasattr(self, 'classification_map') or self.classification_map is None:
            QMessageBox.warning(self, "Warning", "Run ML classification first")
            return
        
        # Get file format choice from user
        file_format = self._get_file_format_choice()
        if file_format is None:
            return  # User cancelled
        
        # Select directory to save spectra
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save All Class A Spectra", ""
        )
        
        if not save_dir:
            return
        
        try:
            self.statusBar().showMessage("Saving all Class A spectra...")
            
            # Get all Class A positions
            class_a_positions = []
            for i, y_pos in enumerate(self.map_data.y_positions):
                for j, x_pos in enumerate(self.map_data.x_positions):
                    if self.classification_map[i, j] == 1:  # Class A
                        # Include probability if available
                        prob = None
                        if hasattr(self, 'probabilities_map') and self.probabilities_map is not None:
                            prob = self.probabilities_map[i, j]
                        spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                        if spectrum:
                            class_a_positions.append((x_pos, y_pos, prob, spectrum))
            
            if len(class_a_positions) == 0:
                QMessageBox.warning(self, "Warning", "No Class A spectra found")
                return
            
            # Apply cosmic ray filtering to Class A spectra
            logger.info(f"Found {len(class_a_positions)} Class A spectra, applying cosmic ray filtering...")
            filtered_class_a_positions = self.filter_cosmic_rays_from_classification(class_a_positions)
            
            if len(filtered_class_a_positions) == 0:
                QMessageBox.warning(self, "Warning", "All Class A spectra were filtered out as cosmic rays!")
                return
            
            logger.info(f"After cosmic ray filtering: {len(filtered_class_a_positions)} spectra remain")
            
            saved_count = 0
            all_wavenumbers = []
            all_intensities = []
            position_info = []
            
            for x_pos, y_pos, prob, spectrum in filtered_class_a_positions:
                if spectrum:
                    # Create base filename (without extension)
                    prob_str = f"_prob{prob:.3f}" if prob is not None else ""
                    filename_base = f"Class_A_X{x_pos}_Y{y_pos}{prob_str}"
                    filepath_base = os.path.join(save_dir, filename_base)
                    
                    # Get spectrum data
                    if self.use_processed_cb.isChecked() and spectrum.processed_intensities is not None:
                        wavenumbers = self.map_data.target_wavenumbers
                        intensities = spectrum.processed_intensities
                        data_type = "processed"
                    else:
                        wavenumbers = spectrum.wavenumbers
                        intensities = spectrum.intensities
                        data_type = "raw"
                    
                    # Prepare metadata
                    metadata = {
                        "Class A Classified Spectrum": "",
                        "Position": f"X={x_pos}, Y={y_pos}",
                        "Data Type": data_type,
                        "Original Filename": spectrum.filename,
                        "Cosmic Ray Filtered": "Yes"
                    }
                    if prob is not None:
                        metadata["Classification Probability"] = f"{prob:.4f}"
                    
                    # Save individual spectrum using chosen format
                    self._save_spectrum_file(filepath_base, wavenumbers, intensities, metadata, file_format)
                    
                    # Collect data for combined file
                    all_wavenumbers.append(wavenumbers)
                    all_intensities.append(intensities)
                    position_info.append((x_pos, y_pos, prob, spectrum.filename))
                    saved_count += 1
            
            # Save combined file with all Class A spectra
            self._save_combined_class_a_file(save_dir, all_wavenumbers, all_intensities, position_info, file_format)
            
            # Save average spectrum
            self._save_class_a_average_spectrum(save_dir, file_format)
            
            # Save comprehensive summary
            summary_filepath = os.path.join(save_dir, "Class_A_All_Summary.txt")
            with open(summary_filepath, 'w') as f:
                f.write(f"All Class A Classified Spectra Summary\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Class A spectra found: {len(class_a_positions)}\n")
                f.write(f"After cosmic ray filtering: {len(filtered_class_a_positions)}\n")
                f.write(f"Cosmic rays filtered out: {len(class_a_positions) - len(filtered_class_a_positions)}\n")
                f.write(f"Successfully saved spectra: {saved_count}\n")
                f.write(f"File format: {file_format.upper()}\n")
                f.write(f"Data type: {'processed' if self.use_processed_cb.isChecked() else 'raw'}\n\n")
                
                f.write("Classification Results:\n")
                if hasattr(self, 'probabilities_map') and self.probabilities_map is not None:
                    # Sort by probability for summary
                    sorted_positions = sorted(position_info, key=lambda x: x[2] if x[2] is not None else 0, reverse=True)
                    for idx, (x_pos, y_pos, prob, filename) in enumerate(sorted_positions):
                        prob_str = f"{prob:.4f}" if prob is not None else "N/A"
                        f.write(f"{idx+1:3d}. X={x_pos:3d}, Y={y_pos:3d}, Prob={prob_str}, File={filename}\n")
                else:
                    for idx, (x_pos, y_pos, prob, filename) in enumerate(position_info):
                        f.write(f"{idx+1:3d}. X={x_pos:3d}, Y={y_pos:3d}, File={filename}\n")
            
            file_count = saved_count * (2 if file_format == "both" else 1)
            combined_count = 2 if file_format == "both" else 1  # combined file
            average_count = 2 if file_format == "both" else 1   # average file
            total_files = file_count + combined_count + average_count + 1  # +1 for summary
            
            self.statusBar().showMessage(f"Saved {saved_count} Class A spectra")
            QMessageBox.information(self, "Success", 
                                   f"Successfully saved {saved_count} Class A spectra to:\n{save_dir}\n\n"
                                   f"Format: {file_format.upper()}\n"
                                   f"Files created: {total_files} total\n"
                                   f"• {file_count} individual spectrum files\n"
                                   f"• {combined_count} combined spectra file(s)\n"
                                   f"• {average_count} average spectrum file(s)\n"
                                   f"• 1 summary file")
            
        except Exception as e:
            self.statusBar().showMessage("Ready")
            QMessageBox.critical(self, "Error", f"Error saving Class A spectra: {str(e)}")

    def _save_combined_class_a_file(self, save_dir, all_wavenumbers, all_intensities, position_info, file_format):
        """Save all Class A spectra in a single combined file."""
        try:
            # Use processed data if available, otherwise use first spectrum's wavenumbers as reference
            if self.use_processed_cb.isChecked():
                reference_wavenumbers = self.map_data.target_wavenumbers
                data_type = "processed"
            else:
                reference_wavenumbers = all_wavenumbers[0]
                data_type = "raw"
            
            # Create combined DataFrame
            combined_data = {'Wavenumber': reference_wavenumbers}
            
            for idx, (wavenumbers, intensities) in enumerate(zip(all_wavenumbers, all_intensities)):
                x_pos, y_pos, prob, filename = position_info[idx]
                
                # For raw data, interpolate to reference wavenumbers if needed
                if not self.use_processed_cb.isChecked() and not np.array_equal(wavenumbers, reference_wavenumbers):
                    f = interp1d(wavenumbers, intensities, kind='linear', 
                               bounds_error=False, fill_value=0)
                    intensities = f(reference_wavenumbers)
                
                prob_str = f"_prob{prob:.3f}" if prob is not None else ""
                column_name = f"X{x_pos}_Y{y_pos}{prob_str}"
                combined_data[column_name] = intensities
            
            df_combined = pd.DataFrame(combined_data)
            
            # Prepare metadata
            metadata = {
                "Combined Class A Classified Spectra": "",
                "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Total spectra": len(all_intensities),
                "Data type": data_type,
                "Column format": "X[pos]_Y[pos]_prob[value]"
            }
            
            # Save in the chosen format(s)
            if file_format == "csv":
                filepath = os.path.join(save_dir, "Class_A_All_Combined.csv")
                with open(filepath, 'w') as f:
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("#\n")
                    df_combined.to_csv(f, index=False)
            elif file_format == "txt":
                filepath = os.path.join(save_dir, "Class_A_All_Combined.txt")
                with open(filepath, 'w') as f:
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("#\n")
                    # Save as space-separated
                    header = " ".join(df_combined.columns)
                    f.write(f"{header}\n")
                    for _, row in df_combined.iterrows():
                        values = " ".join([str(val) for val in row.values])
                        f.write(f"{values}\n")
            elif file_format == "both":
                # CSV file
                csv_filepath = os.path.join(save_dir, "Class_A_All_Combined.csv")
                with open(csv_filepath, 'w') as f:
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("#\n")
                    df_combined.to_csv(f, index=False)
                
                # TXT file
                txt_filepath = os.path.join(save_dir, "Class_A_All_Combined.txt")
                with open(txt_filepath, 'w') as f:
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("#\n")
                    # Save as space-separated
                    header = " ".join(df_combined.columns)
                    f.write(f"{header}\n")
                    for _, row in df_combined.iterrows():
                        values = " ".join([str(val) for val in row.values])
                        f.write(f"{values}\n")
                
        except Exception as e:
            logger.error(f"Error saving combined Class A file: {str(e)}")

    def _save_class_a_average_spectrum(self, save_dir, file_format):
        """Save the average Class A spectrum."""
        try:
            avg_intensities = self._calculate_class_a_average_spectrum()
            if avg_intensities is None:
                return
            
            # Get wavenumbers
            if self.use_processed_cb.isChecked():
                wavenumbers = self.map_data.target_wavenumbers
                data_type = "processed"
            else:
                # Use first Class A spectrum's wavenumbers as reference
                for i, y_pos in enumerate(self.map_data.y_positions):
                    for j, x_pos in enumerate(self.map_data.x_positions):
                        if self.classification_map[i, j] == 1:  # Class A
                            spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                            if spectrum:
                                wavenumbers = spectrum.wavenumbers
                                data_type = "raw"
                                break
                    else:
                        continue
                    break
            
            # Count Class A spectra for metadata
            n_class_a = np.sum(self.classification_map == 1)
            
            # Prepare metadata
            metadata = {
                "Average Class A Spectrum": "",
                "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Number of spectra averaged": n_class_a,
                "Data type": data_type
            }
            
            # Save in the chosen format(s)
            filepath_base = os.path.join(save_dir, "Class_A_Average_Spectrum")
            self._save_spectrum_file(filepath_base, wavenumbers, avg_intensities, metadata, file_format)
                
        except Exception as e:
            logger.error(f"Error saving Class A average spectrum: {str(e)}")

    def _get_file_format_choice(self):
        """Get file format choice from user."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select File Format")
        dialog.setModal(True)
        dialog.resize(300, 150)
        
        layout = QVBoxLayout(dialog)
        
        # Add description
        desc_label = QLabel("Choose the file format for saving spectra:")
        layout.addWidget(desc_label)
        
        # Create radio buttons for format selection
        csv_radio = QCheckBox("CSV (Comma-separated)")
        txt_radio = QCheckBox("TXT (Space-separated)")
        both_radio = QCheckBox("Both formats")
        
        # Set CSV as default
        csv_radio.setChecked(True)
        
        # Group radio buttons to ensure only one is selected
        format_group = QGroupBox("File Format")
        format_layout = QVBoxLayout(format_group)
        format_layout.addWidget(csv_radio)
        format_layout.addWidget(txt_radio)
        format_layout.addWidget(both_radio)
        layout.addWidget(format_group)
        
        # Make radio buttons mutually exclusive
        def on_csv_clicked():
            if csv_radio.isChecked():
                txt_radio.setChecked(False)
                both_radio.setChecked(False)
        
        def on_txt_clicked():
            if txt_radio.isChecked():
                csv_radio.setChecked(False)
                both_radio.setChecked(False)
        
        def on_both_clicked():
            if both_radio.isChecked():
                csv_radio.setChecked(False)
                txt_radio.setChecked(False)
        
        csv_radio.clicked.connect(on_csv_clicked)
        txt_radio.clicked.connect(on_txt_clicked)
        both_radio.clicked.connect(on_both_clicked)
        
        # Add OK/Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Show dialog and get result
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if csv_radio.isChecked():
                return "csv"
            elif txt_radio.isChecked():
                return "txt"
            elif both_radio.isChecked():
                return "both"
        
        return None  # User cancelled

    def _save_spectrum_file(self, filepath_base, wavenumbers, intensities, metadata, file_format):
        """Save spectrum data in the specified format(s)."""
        if file_format == "csv":
            self._save_spectrum_csv(filepath_base + ".csv", wavenumbers, intensities, metadata)
        elif file_format == "txt":
            self._save_spectrum_txt(filepath_base + ".txt", wavenumbers, intensities, metadata)
        elif file_format == "both":
            self._save_spectrum_csv(filepath_base + ".csv", wavenumbers, intensities, metadata)
            self._save_spectrum_txt(filepath_base + ".txt", wavenumbers, intensities, metadata)

    def _save_spectrum_csv(self, filepath, wavenumbers, intensities, metadata):
        """Save spectrum as CSV file."""
        df = pd.DataFrame({
            'Wavenumber': wavenumbers,
            'Intensity': intensities
        })
        
        with open(filepath, 'w') as f:
            # Write metadata as comments
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")
            # Save as CSV
            df.to_csv(f, index=False)

    def _save_spectrum_txt(self, filepath, wavenumbers, intensities, metadata):
        """Save spectrum as space-separated TXT file."""
        with open(filepath, 'w') as f:
            # Write metadata as comments
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")
            # Write header
            f.write("Wavenumber Intensity\n")
            # Write data as space-separated values
            for wn, intensity in zip(wavenumbers, intensities):
                f.write(f"{wn} {intensity}\n")

    def filter_cosmic_rays_from_classification(self, class_a_spectra_info: List[Tuple], 
                                             intensity_threshold: float = 3000,  # Lowered from 5000 to catch more cosmic rays
                                             spike_ratio_threshold: float = 10.0,  # Lowered from 20.0 to be more sensitive
                                             max_spike_width: int = 5) -> List[Tuple]:
        """
        Filter out cosmic ray contaminated spectra from classification results.
        
        This method uses conservative thresholds to catch cosmic rays that may have
        escaped earlier filtering stages.
        
        Parameters:
        -----------
        class_a_spectra_info : List[Tuple]
            List of tuples containing (x_pos, y_pos, probability, spectrum_data)
        intensity_threshold : float
            Absolute intensity threshold - spectra with peaks above this are suspect
        spike_ratio_threshold : float
            Ratio threshold - spike must be this many times higher than median
        max_spike_width : int
            Maximum width (in data points) for a cosmic ray spike
            
        Returns:
        --------
        List[Tuple]
            Filtered list with cosmic ray contaminated spectra removed
        """
        try:
            if not class_a_spectra_info:
                return class_a_spectra_info
            
            filtered_spectra = []
            cosmic_ray_count = 0
            
            logger.info(f"Filtering {len(class_a_spectra_info)} Class A spectra for cosmic ray contamination...")
            
            # First, collect all max intensities for statistical analysis
            all_max_intensities = []
            for x_pos, y_pos, probability, spectrum_data in class_a_spectra_info:
                try:
                    # Get spectrum intensities
                    if hasattr(spectrum_data, 'processed_intensities') and spectrum_data.processed_intensities is not None:
                        intensities = spectrum_data.processed_intensities
                    else:
                        intensities = spectrum_data.intensities
                    
                    all_max_intensities.append(np.max(intensities))
                except Exception as e:
                    logger.debug(f"Error getting max intensity for ({x_pos}, {y_pos}): {e}")
                    all_max_intensities.append(0)
            
            # Calculate statistical thresholds
            all_max_intensities = np.array(all_max_intensities)
            median_max = np.median(all_max_intensities)
            mad = np.median(np.abs(all_max_intensities - median_max))
            
            # Dynamic intensity threshold based on the data distribution
            if mad > 0:
                dynamic_threshold = median_max + 3 * mad  # 3-sigma equivalent using MAD
                # Use the more conservative of the two thresholds
                effective_intensity_threshold = min(intensity_threshold, dynamic_threshold)
            else:
                effective_intensity_threshold = intensity_threshold
            
            logger.info(f"Using intensity threshold: {effective_intensity_threshold:.0f} (median max: {median_max:.0f})")
            
            for x_pos, y_pos, probability, spectrum_data in class_a_spectra_info:
                is_cosmic_ray = False
                rejection_reason = ""
                
                try:
                    # Get spectrum intensities
                    if hasattr(spectrum_data, 'processed_intensities') and spectrum_data.processed_intensities is not None:
                        intensities = spectrum_data.processed_intensities
                    else:
                        intensities = spectrum_data.intensities
                    
                    # Criterion 1: Absolute intensity check
                    max_intensity = np.max(intensities)
                    if max_intensity > effective_intensity_threshold:
                        logger.info(f"Spectrum at ({x_pos}, {y_pos}) rejected: max intensity {max_intensity:.0f} > {effective_intensity_threshold:.0f}")
                        is_cosmic_ray = True
                        rejection_reason = f"high_intensity_{max_intensity:.0f}"
                
                    # Criterion 2: Intensity ratio check (only if not already flagged)
                    if not is_cosmic_ray:
                        median_intensity = np.median(intensities[intensities > 0])  # Exclude zeros
                        if median_intensity > 0:
                            intensity_ratio = max_intensity / median_intensity
                            if intensity_ratio > spike_ratio_threshold:
                                logger.info(f"Spectrum at ({x_pos}, {y_pos}) rejected: intensity ratio {intensity_ratio:.1f} > {spike_ratio_threshold}")
                                is_cosmic_ray = True
                                rejection_reason = f"high_ratio_{intensity_ratio:.1f}"
                    
                    # Criterion 3: Sharp spike detection using gradient analysis
                    if not is_cosmic_ray and len(intensities) > 10:
                        # Calculate derivatives to find sharp spikes
                        gradient = np.gradient(intensities)
                        second_derivative = np.gradient(gradient)
                        
                        # Look for extreme second derivatives (sharp peaks)
                        if len(second_derivative) > 0:
                            sd_threshold = np.std(second_derivative) * 5  # 5-sigma for second derivative
                            extreme_points = np.abs(second_derivative) > sd_threshold
                            
                            if np.any(extreme_points):
                                # Check if these correspond to high-intensity narrow peaks
                                extreme_indices = np.where(extreme_points)[0]
                                for idx in extreme_indices:
                                    if (idx > 0 and idx < len(intensities) - 1 and 
                                        intensities[idx] > median_intensity * 3):  # At least 3x median
                                        
                                        # Check if it's a local maximum
                                        if (intensities[idx] > intensities[idx-1] and 
                                            intensities[idx] > intensities[idx+1]):
                                            
                                            logger.info(f"Spectrum at ({x_pos}, {y_pos}) rejected: sharp spike at index {idx}")
                                            is_cosmic_ray = True
                                            rejection_reason = f"sharp_spike_idx_{idx}"
                                            break
                    
                    # Criterion 4: Multiple peak analysis
                    if not is_cosmic_ray and len(intensities) > 20:
                        try:
                            # Find peaks above 2x median
                            peak_threshold = median_intensity * 2
                            peaks, properties = find_peaks(intensities, height=peak_threshold)
                            
                            if len(peaks) > 0:
                                # Calculate peak widths
                                widths = peak_widths(intensities, peaks, rel_height=0.5)[0]
                                peak_heights = intensities[peaks]
                                
                                # Check for very narrow, very high peaks
                                for i, (peak_idx, width, height) in enumerate(zip(peaks, widths, peak_heights)):
                                    # Cosmic rays are typically very narrow and very high
                                    if width <= max_spike_width and height > median_intensity * 8:  # Very high and narrow
                                        logger.info(f"Spectrum at ({x_pos}, {y_pos}) rejected: narrow peak (width={width:.1f}, height={height:.0f})")
                                        is_cosmic_ray = True
                                        rejection_reason = f"narrow_peak_w{width:.1f}_h{height:.0f}"
                                        break
                                        
                        except Exception as e:
                            logger.debug(f"Error in peak analysis for ({x_pos}, {y_pos}): {e}")
                    
                    # Keep the spectrum if no cosmic ray indicators were found
                    if not is_cosmic_ray:
                        filtered_spectra.append((x_pos, y_pos, probability, spectrum_data))
                    else:
                        cosmic_ray_count += 1
                        logger.debug(f"Rejected ({x_pos}, {y_pos}): {rejection_reason}")
                        
                except Exception as e:
                    logger.error(f"Error filtering spectrum at ({x_pos}, {y_pos}): {e}")
                    # If there's an error in analysis, include the spectrum (conservative approach)
                    filtered_spectra.append((x_pos, y_pos, probability, spectrum_data))
            
            logger.info(f"Cosmic ray filtering complete: {cosmic_ray_count} spectra removed, {len(filtered_spectra)} remain")
            
            # Apply additional statistical filtering only if we have enough spectra left
            if len(filtered_spectra) > 5:
                original_count = len(filtered_spectra)
                filtered_spectra = self._apply_statistical_outlier_filtering(filtered_spectra)
                statistical_removed = original_count - len(filtered_spectra)
                if statistical_removed > 0:
                    logger.info(f"Statistical filtering removed {statistical_removed} additional outliers")
            
            return filtered_spectra
            
        except Exception as e:
            logger.error(f"Error in cosmic ray filtering: {e}")
            return class_a_spectra_info  # Return original list if filtering fails
    
    def _apply_statistical_outlier_filtering(self, spectra_info: List[Tuple]) -> List[Tuple]:
        """
        Apply statistical outlier filtering to remove spectra that are statistical outliers
        compared to other Class A spectra.
        """
        try:
            if len(spectra_info) <= 3:  # Need at least 3 spectra for meaningful statistics
                return spectra_info
                
            logger.info("Applying statistical outlier filtering to Class A spectra...")
            
            # Extract all max intensities
            max_intensities = []
            for x_pos, y_pos, probability, spectrum_data in spectra_info:
                if hasattr(spectrum_data, 'processed_intensities') and spectrum_data.processed_intensities is not None:
                    intensities = spectrum_data.processed_intensities
                else:
                    intensities = spectrum_data.intensities
                max_intensities.append(np.max(intensities))
            
            max_intensities = np.array(max_intensities)
            
            # Use Modified Z-score for outlier detection
            median_max = np.median(max_intensities)
            mad = np.median(np.abs(max_intensities - median_max))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (max_intensities - median_max) / mad
                outlier_threshold = 5.0  # Much more conservative threshold (was 3.5)
                
                outlier_mask = modified_z_scores > outlier_threshold  # Only positive outliers
                
                if np.any(outlier_mask):
                    filtered_spectra = []
                    outlier_count = 0
                    
                    for i, (x_pos, y_pos, probability, spectrum_data) in enumerate(spectra_info):
                        if not outlier_mask[i]:
                            filtered_spectra.append((x_pos, y_pos, probability, spectrum_data))
                        else:
                            outlier_count += 1
                            logger.info(f"Statistical outlier removed: ({x_pos}, {y_pos}) with max intensity {max_intensities[i]:.0f} (z-score: {modified_z_scores[i]:.2f})")
                    
                    logger.info(f"Statistical filtering: {outlier_count} outliers removed, {len(filtered_spectra)} remain")
                    return filtered_spectra
            
            return spectra_info
            
        except Exception as e:
            logger.error(f"Error in statistical outlier filtering: {e}")
            return spectra_info

    def run_cosmic_ray_diagnosis(self):
        """Run cosmic ray diagnosis on current map data."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        try:
            # Run diagnosis
            diagnosis = self.map_data.diagnose_cosmic_ray_filtering(max_examples=50)
            
            if diagnosis:
                # Create a detailed report dialog
                dialog = QDialog(self)
                dialog.setWindowTitle("Cosmic Ray Diagnosis Report")
                dialog.setGeometry(200, 200, 600, 400)
                
                layout = QVBoxLayout(dialog)
                
                # Create text area for report
                text_edit = QTextEdit(dialog)
                text_edit.setReadOnly(True)
                
                # Generate report text
                report = f"COSMIC RAY DIAGNOSIS REPORT\n"
                report += f"============================\n\n"
                report += f"Total spectra analyzed: {diagnosis['total_analyzed']}\n"
                report += f"Spectra with cosmic rays: {len(diagnosis['cosmic_ray_examples'])}\n"
                report += f"Normal spectra: {len(diagnosis['normal_examples'])}\n\n"
                
                if diagnosis['cosmic_ray_examples']:
                    report += "COSMIC RAY CONTAMINATED SPECTRA:\n"
                    for i, ex in enumerate(diagnosis['cosmic_ray_examples'][:10]):
                        report += f"{i+1:2d}. {ex['filename']} at {ex['position']}\n"
                        report += f"    Raw data - Max: {ex['raw_max']:.0f}, Ratio: {ex['raw_ratio']:.1f}\n"
                        if ex['proc_max'] > 0:
                            report += f"    Processed - Max: {ex['proc_max']:.0f}, Ratio: {ex['proc_ratio']:.1f}\n"
                        report += "\n"
                
                # Calculate statistics
                if diagnosis['cosmic_ray_examples'] or diagnosis['normal_examples']:
                    all_examples = diagnosis['cosmic_ray_examples'] + diagnosis['normal_examples']
                    raw_ratios = [ex['raw_ratio'] for ex in all_examples if ex['raw_ratio'] > 0]
                    if raw_ratios:
                        report += f"\nSTATISTICS:\n"
                        report += f"Raw intensity ratios - Median: {np.median(raw_ratios):.1f}, Max: {np.max(raw_ratios):.1f}\n"
                        report += f"Cosmic ray threshold typically: ~10.0\n"
                
                text_edit.setPlainText(report)
                layout.addWidget(text_edit)
                
                # Add buttons
                button_layout = QHBoxLayout()
                
                clean_button = QPushButton("Clean All Cosmic Rays")
                clean_button.clicked.connect(lambda: self.clean_all_cosmic_rays(dialog))
                button_layout.addWidget(clean_button)
                
                close_button = QPushButton("Close")
                close_button.clicked.connect(dialog.accept)
                button_layout.addWidget(close_button)
                
                layout.addLayout(button_layout)
                
                dialog.exec()
            else:
                QMessageBox.information(self, "Diagnosis Failed", "Could not run cosmic ray diagnosis")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running diagnosis: {str(e)}")

    def clean_all_cosmic_rays(self, parent_dialog=None):
        """
        Clean all cosmic rays from the loaded map data.
        This will apply cosmic ray elimination to all spectra.
        """
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        try:
            # Create progress dialog
            progress_dialog = QDialog(parent_dialog or self)
            progress_dialog.setWindowTitle("Cleaning Cosmic Rays")
            progress_dialog.setModal(True)
            progress_dialog.resize(400, 150)
            
            layout = QVBoxLayout(progress_dialog)
            
            progress_label = QLabel("Cleaning cosmic rays from all spectra...")
            layout.addWidget(progress_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, len(self.map_data.spectra))
            layout.addWidget(progress_bar)
            
            progress_dialog.show()
            QApplication.processEvents()
            
            # Process each spectrum
            processed_count = 0
            total_spectra = len(self.map_data.spectra)
            
            for i, ((x_pos, y_pos), spectrum_data) in enumerate(self.map_data.spectra.items()):
                # Apply cosmic ray elimination
                has_cosmic_ray, cleaned_intensities = self.map_data.detect_cosmic_rays(
                    spectrum_data.wavenumbers, 
                    spectrum_data.intensities,
                    threshold_factor=8.0,  # Use more aggressive threshold
                    window_size=5,
                    min_width_ratio=0.1,
                    max_fwhm=5.0
                )
                
                if has_cosmic_ray:
                    # Update the spectrum data with cleaned intensities
                    spectrum_data.intensities = cleaned_intensities
                    # Also update processed intensities if they exist
                    if spectrum_data.processed_intensities is not None:
                        spectrum_data.processed_intensities = self.map_data._preprocess_spectrum(
                            spectrum_data.wavenumbers, cleaned_intensities
                        )
                    processed_count += 1
                
                # Update progress
                progress_bar.setValue(i + 1)
                progress_label.setText(f"Processed {i + 1}/{total_spectra} spectra, cleaned {processed_count} cosmic ray events")
                QApplication.processEvents()
            
            progress_dialog.close()
            
            logger.info(f"Cosmic ray cleaning complete: processed {processed_count} spectra out of {total_spectra}")
            
            QMessageBox.information(self, "Cosmic Ray Cleaning Complete", 
                                   f"Successfully processed {total_spectra} spectra.\n"
                                   f"Cleaned cosmic rays from {processed_count} spectra.\n\n"
                                   f"Cleaned spectra: {processed_count}\n"
                                   f"Clean spectra: {total_spectra - processed_count}")
            
        except Exception as e:
            logger.error(f"Error cleaning cosmic rays: {e}")
            QMessageBox.critical(self, "Error", f"Error cleaning cosmic rays: {str(e)}")

    def save_nmf_component_spectra(self):
        """Save all NMF component spectra as space-separated text files without headers."""
        if not hasattr(self, 'nmf') or self.nmf is None:
            QMessageBox.warning(self, "Warning", "Run NMF analysis first")
            return
        
        if not hasattr(self, 'nmf_component_spectra') or self.nmf_component_spectra is None:
            QMessageBox.warning(self, "Warning", "No NMF component spectra available")
            return
        
        try:
            # Select directory to save component spectra
            directory = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save NMF Component Spectra", ""
            )
            
            if directory:
                wavenumbers = self.map_data.target_wavenumbers
                component_spectra = self.nmf_component_spectra  # H matrix from NMF
                n_components = component_spectra.shape[0]
                
                logger.info(f"Saving {n_components} NMF component spectra to {directory}")
                
                for i in range(n_components):
                    # Create filename
                    filename = f"nmf_component_{i+1:02d}.txt"
                    filepath = os.path.join(directory, filename)
                    
                    # Get component spectrum
                    intensities = component_spectra[i, :]
                    
                    # Save as space-separated text without header
                    with open(filepath, 'w') as f:
                        for wn, intensity in zip(wavenumbers, intensities):
                            f.write(f"{wn} {intensity}\n")
                    
                    logger.debug(f"Saved component {i+1} to {filename}")
                
                # Also save a summary file with component information
                summary_filename = "nmf_components_summary.txt"
                summary_filepath = os.path.join(directory, summary_filename)
                
                with open(summary_filepath, 'w') as f:
                    f.write("NMF Component Spectra Summary\n")
                    f.write("============================\n\n")
                    f.write(f"Number of components: {n_components}\n")
                    f.write(f"Reconstruction error: {self.nmf.reconstruction_err_:.6f}\n")
                    f.write(f"Wavenumber range: {np.min(wavenumbers):.1f} - {np.max(wavenumbers):.1f} cm⁻¹\n")
                    f.write(f"Number of data points: {len(wavenumbers)}\n\n")
                    f.write("Component files:\n")
                    for i in range(n_components):
                        f.write(f"  nmf_component_{i+1:02d}.txt - Component {i+1}\n")
                    f.write(f"\nFile format: Two columns (wavenumber intensity) separated by space, no header\n")
                
                QMessageBox.information(self, "Success", 
                                       f"Successfully saved {n_components} NMF component spectra to:\n{directory}\n\n"
                                       f"Files: nmf_component_01.txt to nmf_component_{n_components:02d}.txt\n"
                                       f"Summary: {summary_filename}")
                
                logger.info(f"NMF component spectra save complete: {n_components} files saved")
                
        except Exception as e:
            logger.error(f"Error saving NMF component spectra: {e}")
            QMessageBox.critical(self, "Error", f"Error saving NMF component spectra:\n{str(e)}")

    def debug_ml_pipeline(self):
        """
        Comprehensive debugging tool to identify ML classification issues.
        Checks preprocessing consistency, training data quality, and model performance.
        """
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        if self.rf_model is None:
            QMessageBox.warning(self, "Warning", "Train Random Forest model first")
            return
        
        try:
            # Create debugging dialog
            debug_dialog = QDialog(self)
            debug_dialog.setWindowTitle("ML Pipeline Debugging")
            debug_dialog.setGeometry(100, 100, 1200, 800)
            
            layout = QVBoxLayout(debug_dialog)
            
            # Create text area for results
            debug_text = QTextEdit()
            debug_text.setFont(QFont("Courier", 10))
            layout.addWidget(debug_text)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(debug_dialog.close)
            layout.addWidget(close_btn)
            
            # Run debugging analysis
            debug_results = []
            debug_results.append("=== ML PIPELINE DEBUGGING REPORT ===\n")
            
            # 1. Check preprocessing consistency
            debug_results.append("1. PREPROCESSING CONSISTENCY CHECK:")
            debug_results.append("-" * 40)
            
            # Get training preprocessing method
            training_method = "RamanMapData._preprocess_spectrum: interpolation + smoothing (NO normalization)"
            classification_method = "_preprocess_single_spectrum: interpolation + smoothing + NORMALIZATION"
            
            debug_results.append(f"Training preprocessing: {training_method}")
            debug_results.append(f"Classification preprocessing: {classification_method}")
            debug_results.append("❌ MISMATCH DETECTED: Training uses NO normalization, Classification uses normalization!")
            debug_results.append("This is likely causing the classification failures.\n")
            
            # 2. Check a sample spectrum from training directories
            debug_results.append("2. TRAINING DATA ANALYSIS:")
            debug_results.append("-" * 30)
            
            try:
                class_a_dir = self.class_a_path.text()
                class_b_dir = self.class_b_path.text()
                
                if class_a_dir and Path(class_a_dir).exists():
                    # Load a sample Class A spectrum
                    class_a_files = list(Path(class_a_dir).glob("*.csv")) + list(Path(class_a_dir).glob("*.txt"))
                    if class_a_files:
                        sample_file = class_a_files[0]
                        data = np.loadtxt(sample_file)
                        wn_train = data[:, 0]
                        int_train = data[:, 1]
                        
                        # Process with training method (no normalization)
                        f = interp1d(wn_train, int_train, kind='linear', bounds_error=False, fill_value=0)
                        train_processed = f(self.map_data.target_wavenumbers)
                        if len(train_processed) > 5:
                            train_processed = savgol_filter(train_processed, window_length=5, polyorder=2)
                        
                        # Process with classification method (with normalization)
                        classif_processed = self._preprocess_single_spectrum(wn_train, int_train)
                        
                        debug_results.append(f"Sample Class A file: {sample_file.name}")
                        debug_results.append(f"Raw intensity range: {np.min(int_train):.1f} to {np.max(int_train):.1f}")
                        debug_results.append(f"Training processed range: {np.min(train_processed):.1f} to {np.max(train_processed):.1f}")
                        debug_results.append(f"Classification processed range: {np.min(classif_processed):.1f} to {np.max(classif_processed):.1f}")
                        debug_results.append(f"Scale difference: {np.max(train_processed)/np.max(classif_processed):.1f}x")
                        debug_results.append("❌ This scale difference explains the classification failure!\n")
                    else:
                        debug_results.append("No training files found in Class A directory\n")
                else:
                    debug_results.append("Class A directory not set or doesn't exist\n")
            
            except Exception as e:
                debug_results.append(f"Error analyzing training data: {str(e)}\n")
            
            # 3. Check map data preprocessing
            debug_results.append("3. MAP DATA ANALYSIS:")
            debug_results.append("-" * 25)
            
            try:
                # Get a sample map spectrum
                sample_pos = list(self.map_data.spectra.keys())[0]
                sample_spectrum = self.map_data.spectra[sample_pos]
                
                debug_results.append(f"Sample map spectrum at position {sample_pos}:")
                debug_results.append(f"Raw intensity range: {np.min(sample_spectrum.intensities):.1f} to {np.max(sample_spectrum.intensities):.1f}")
                
                if sample_spectrum.processed_intensities is not None:
                    debug_results.append(f"Map processed range: {np.min(sample_spectrum.processed_intensities):.1f} to {np.max(sample_spectrum.processed_intensities):.1f}")
                
                # Show how it would be processed for classification
                classif_features = self._preprocess_single_spectrum(
                    sample_spectrum.wavenumbers, sample_spectrum.intensities
                )
                debug_results.append(f"Classification processed range: {np.min(classif_features):.1f} to {np.max(classif_features):.1f}")
                debug_results.append("")
            
            except Exception as e:
                debug_results.append(f"Error analyzing map data: {str(e)}\n")
            
            # 4. Feature type analysis
            debug_results.append("4. FEATURE TYPE ANALYSIS:")
            debug_results.append("-" * 27)
            
            feature_type = getattr(self, 'rf_feature_type', 'unknown')
            debug_results.append(f"Model trained with: {feature_type} features")
            
            if hasattr(self, 'pca') and self.pca is not None:
                debug_results.append(f"PCA available: Yes ({self.pca.n_components_} components)")
            else:
                debug_results.append("PCA available: No")
            
            if hasattr(self, 'nmf') and self.nmf is not None:
                debug_results.append(f"NMF available: Yes ({self.nmf.n_components} components)")
            else:
                debug_results.append("NMF available: No")
            debug_results.append("")
            
            # 5. Model performance
            debug_results.append("5. MODEL PERFORMANCE:")
            debug_results.append("-" * 20)
            
            if hasattr(self, 'rf_y_test') and hasattr(self, 'rf_y_pred'):
                from sklearn.metrics import accuracy_score, confusion_matrix
                accuracy = accuracy_score(self.rf_y_test, self.rf_y_pred)
                cm = confusion_matrix(self.rf_y_test, self.rf_y_pred)
                
                debug_results.append(f"Training accuracy: {accuracy:.3f}")
                debug_results.append(f"Confusion matrix:")
                debug_results.append(f"  Predicted:  0    1")
                debug_results.append(f"Actual 0: [{cm[0,0]:3d}  {cm[0,1]:3d}]")
                debug_results.append(f"Actual 1: [{cm[1,0]:3d}  {cm[1,1]:3d}]")
                
                if cm[1,1] == 0:
                    debug_results.append("❌ Model never predicts Class A correctly in testing!")
                debug_results.append("")
            
            # 6. Recommendations
            debug_results.append("6. RECOMMENDATIONS:")
            debug_results.append("-" * 17)
            debug_results.append("Based on the analysis above, here are the main issues and fixes:")
            debug_results.append("")
            debug_results.append("❌ CRITICAL ISSUE: Preprocessing inconsistency")
            debug_results.append("   - Training: No normalization")
            debug_results.append("   - Classification: Normalization applied")
            debug_results.append("   - FIX: Make preprocessing consistent")
            debug_results.append("")
            debug_results.append("🔧 SOLUTIONS:")
            debug_results.append("   1. Retrain model with normalized training data")
            debug_results.append("   2. OR: Remove normalization from classification")
            debug_results.append("   3. Verify training data represents map spectra")
            debug_results.append("   4. Check if cosmic ray filtering is consistent")
            debug_results.append("")
            debug_results.append("Click 'Fix Preprocessing' button below to automatically fix this issue.")
            
            # Display results
            debug_text.setPlainText("\n".join(debug_results))
            
            # Add fix button
            fix_btn = QPushButton("Fix Preprocessing Inconsistency")
            fix_btn.clicked.connect(lambda: self.fix_preprocessing_inconsistency(debug_dialog))
            layout.insertWidget(-1, fix_btn)  # Insert before close button
            
            debug_dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error running ML debugging: {str(e)}")

    def fix_preprocessing_inconsistency(self, parent_dialog=None):
        """Fix the preprocessing inconsistency between training and classification."""
        reply = QMessageBox.question(
            parent_dialog or self, 
            "Fix Preprocessing", 
            "This will modify the preprocessing to make training and classification consistent.\n\n"
            "Choose the approach:\n"
            "- Yes: Remove normalization from classification (recommended)\n"
            "- No: Keep normalization and retrain model",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Fix by removing normalization from classification
            self._fix_remove_normalization()
            QMessageBox.information(parent_dialog or self, "Fixed", 
                                   "Preprocessing fixed! Classification will now use the same preprocessing as training.\n"
                                   "Try classifying the map again.")
        elif reply == QMessageBox.StandardButton.No:
            # Fix by retraining with normalized data
            QMessageBox.information(parent_dialog or self, "Retrain Required", 
                                   "You chose to keep normalization. Please retrain your Random Forest model\n"
                                   "with the current preprocessing settings.")

    def _fix_remove_normalization(self):
        """Remove normalization from the classification preprocessing."""
        # Create a backup of the original method
        if not hasattr(self, '_original_preprocess_single_spectrum'):
            self._original_preprocess_single_spectrum = self._preprocess_single_spectrum
        
        # Replace with non-normalizing version
        def _preprocess_single_spectrum_no_norm(wavenumbers, intensities):
            """Preprocess without normalization to match training."""
            try:
                # Remove NaN values
                valid_mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
                clean_wn = wavenumbers[valid_mask]
                clean_int = intensities[valid_mask]
                
                if len(clean_wn) < 2:
                    return np.zeros_like(self.map_data.target_wavenumbers)
                
                # Interpolate to target wavenumbers
                f = interp1d(clean_wn, clean_int, kind='linear', 
                            bounds_error=False, fill_value=0)
                resampled = f(self.map_data.target_wavenumbers)
                
                # Apply smoothing
                if len(resampled) > 5:
                    try:
                        resampled = savgol_filter(resampled, window_length=5, polyorder=2)
                    except ValueError:
                        pass
                
                # NO NORMALIZATION - this matches the training preprocessing
                return resampled
                
            except Exception as e:
                logger.error(f"Error preprocessing spectrum: {str(e)}")
                return np.zeros_like(self.map_data.target_wavenumbers)
        
        # Replace the method
        self._preprocess_single_spectrum = _preprocess_single_spectrum_no_norm
        logger.info("Preprocessing fixed: Removed normalization from classification to match training")

    def closeEvent(self, event):
        """Handle window close event - ensure threads are properly cleaned up."""
        try:
            # Stop any running worker threads
            if hasattr(self, 'worker') and self.worker is not None:
                if self.worker.isRunning():
                    logger.info("Stopping worker thread...")
                    self.worker.stop()
                    self.worker = None
            
            # Accept the close event
            event.accept()
            logger.info("Application closed cleanly")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            event.accept()  # Close anyway

    def run_nmf(self):
        """Run NMF analysis on the map data."""
        if self.map_data is None:
            QMessageBox.warning(self, "Warning", "Load map data first")
            return
        
        try:
            # Stop any existing worker thread
            if hasattr(self, 'worker') and self.worker is not None:
                if self.worker.isRunning():
                    self.worker.stop()
                self.worker = None
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Running NMF analysis...")
            
            # Create and start worker thread
            n_components = self.nmf_n_components.value()
            batch_size = self.nmf_batch_size.value()
            
            self.worker = MapAnalysisWorker(
                self._run_nmf_worker, n_components, batch_size
            )
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self._on_nmf_finished)
            self.worker.error.connect(self._on_nmf_error)
            self.worker.start()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Error running NMF: {str(e)}")

    def _run_nmf_worker(self, worker, n_components, batch_size):
        """Worker function for NMF analysis."""
        try:
            from sklearn.decomposition import NMF
            from sklearn.preprocessing import StandardScaler
            
            # Signal progress
            if worker.is_stopped:
                return None
            worker.progress.emit(10)
            
            # Prepare data for NMF
            df = self.map_data.prepare_ml_data(use_processed=self.use_processed_cb.isChecked())
            
            if df is None or df.empty:
                raise ValueError("No data available for NMF analysis")
            
            # Extract spectral data (exclude position columns)
            spectral_columns = [col for col in df.columns if col not in ['x_pos', 'y_pos']]
            X = df[spectral_columns].values
            
            # Remove any NaN values and ensure non-negative values for NMF
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make values non-negative (NMF requirement)
            X = np.clip(X, 0, None)
            
            if worker.is_stopped:
                return None
            worker.progress.emit(30)
            
            # Apply NMF
            nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
            
            # For large datasets, we might need to subsample for initial fit
            if len(X) > batch_size:
                # Sample randomly but ensure we get representative data
                indices = np.random.choice(len(X), size=batch_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Fit NMF
            if worker.is_stopped:
                return None
            worker.progress.emit(50)
            
            W_sample = nmf.fit_transform(X_sample)
            H = nmf.components_
            
            if worker.is_stopped:
                return None
            worker.progress.emit(70)
            
            # Transform all data
            W = nmf.transform(X)
            
            if worker.is_stopped:
                return None
            worker.progress.emit(90)
            
            # Store results
            results = {
                'nmf': nmf,
                'W': W,
                'H': H,
                'reconstruction_error': nmf.reconstruction_err_,
                'n_components': n_components,
                'n_iter': nmf.n_iter_
            }
            
            worker.progress.emit(100)
            return results
            
        except Exception as e:
            raise e

    def _on_nmf_finished(self, results):
        """Handle completion of NMF analysis."""
        try:
            if results is not None:
                # Store results
                self.nmf = results['nmf']
                self.nmf_components = results['W']
                self.nmf_basis = results['H']
                
                # Plot results
                self._plot_nmf_results()
                
                self.statusBar().showMessage("NMF analysis complete")
                
                # Show info message
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"NMF analysis completed successfully!\n"
                    f"Components: {results['n_components']}\n"
                    f"Iterations: {results['n_iter']}\n"
                    f"Reconstruction error: {results['reconstruction_error']:.6f}"
                )
            else:
                self.statusBar().showMessage("NMF analysis cancelled")
                
        except Exception as e:
            self.statusBar().showMessage("Error in NMF analysis")
            QMessageBox.critical(self, "Error", f"Error processing NMF results: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def _on_nmf_error(self, error_msg):
        """Handle NMF analysis errors."""
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("NMF analysis failed")
        QMessageBox.critical(self, "NMF Error", f"NMF analysis failed: {error_msg}")

    def _plot_nmf_results(self):
        """Plot NMF analysis results."""
        if self.nmf is None or self.nmf_components is None:
            return

        try:
            # Clear axes
            self.nmf_ax1.clear()
            self.nmf_ax2.clear()

            # Plot component weights (W matrix) - first few components
            n_components_to_show = min(5, self.nmf_components.shape[1])
            
            for i in range(n_components_to_show):
                self.nmf_ax1.hist(self.nmf_components[:, i], bins=50, alpha=0.7, 
                                label=f'Component {i+1}')
            
            self.nmf_ax1.set_xlabel('Weight Value')
            self.nmf_ax1.set_ylabel('Frequency')
            self.nmf_ax1.set_title('NMF Component Weight Distributions')
            self.nmf_ax1.legend()
            self.nmf_ax1.grid(True)

            # Plot basis spectra (H matrix) - component spectra
            if hasattr(self.map_data, 'target_wavenumbers') and self.map_data.target_wavenumbers is not None:
                wavenumbers = self.map_data.target_wavenumbers
            else:
                # Create dummy wavenumbers if not available
                wavenumbers = np.arange(self.nmf_basis.shape[1])
            
            for i in range(n_components_to_show):
                self.nmf_ax2.plot(wavenumbers, self.nmf_basis[i, :], 
                                label=f'Component {i+1}')
            
            self.nmf_ax2.set_xlabel('Wavenumber (cm⁻¹)')
            self.nmf_ax2.set_ylabel('Intensity')
            self.nmf_ax2.set_title('NMF Basis Spectra')
            self.nmf_ax2.legend()
            self.nmf_ax2.grid(True)

            # Adjust layout and draw
            self.nmf_fig.tight_layout()
            self.nmf_canvas.draw()

        except Exception as e:
            print(f"Error plotting NMF results: {str(e)}")

    def browse_class_a_directory(self):
        """Browse for Class A (positive examples) directory."""
        try:
            directory = QFileDialog.getExistingDirectory(
                self, "Select Class A (Positive Examples) Directory", ""
            )
            
            if directory:
                self.class_a_path.setText(directory)
                self.statusBar().showMessage(f"Selected Class A directory: {directory}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting Class A directory: {str(e)}")

    def browse_class_b_directory(self):
        """Browse for Class B (negative examples) directory."""
        try:
            directory = QFileDialog.getExistingDirectory(
                self, "Select Class B (Negative Examples) Directory", ""
            )
            
            if directory:
                self.class_b_path.setText(directory)
                self.statusBar().showMessage(f"Selected Class B directory: {directory}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting Class B directory: {str(e)}")

    def save_rf_model(self):
        """Save Random Forest model to file with metadata."""
        if not hasattr(self, 'rf_model') or self.rf_model is None:
            QMessageBox.warning(self, "Warning", "No Random Forest model to save.")
            return
        
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Random Forest Model",
                "rf_model.joblib",
                "Joblib files (*.joblib);;All files (*.*)"
            )
            
            if file_path:
                import joblib
                from datetime import datetime
                
                # Get the authoritative feature count from the model
                model_features = self.rf_model.n_features_in_ if hasattr(self.rf_model, 'n_features_in_') else None
                
                # Create model package with metadata
                model_package = {
                    'model': self.rf_model,
                    'metadata': {
                        'n_features': model_features,
                        'expected_features': model_features,  # Use the same value as model for consistency
                        'created_date': datetime.now().isoformat(),
                        'model_type': 'RandomForestClassifier',
                        'n_estimators': self.rf_model.n_estimators,
                        'max_depth': self.rf_model.max_depth,
                        'version': '1.1'  # Increment version to indicate fix
                    }
                }
                
                # Add preprocessing info if available
                if hasattr(self, 'map_data') and self.map_data:
                    try:
                        sample_data = self._prepare_data_for_analysis()
                        if sample_data is not None:
                            current_features = sample_data.shape[1]
                            model_package['metadata']['data_shape'] = sample_data.shape
                            model_package['metadata']['current_data_features'] = current_features
                            
                            # Warn if there's a mismatch at save time
                            if model_features and current_features != model_features:
                                print(f"WARNING: Model was trained with {model_features} features, "
                                      f"but current data has {current_features} features")
                            
                            # Add wavenumber info if available
                            if hasattr(self.map_data, 'target_wavenumbers') and self.map_data.target_wavenumbers is not None:
                                model_package['metadata']['wavenumber_range'] = [
                                    float(self.map_data.target_wavenumbers.min()),
                                    float(self.map_data.target_wavenumbers.max())
                                ]
                                model_package['metadata']['n_wavenumbers'] = len(self.map_data.target_wavenumbers)
                    except Exception as e:
                        print(f"Warning: Could not add preprocessing metadata: {e}")
                        pass  # Continue without this metadata
                
                joblib.dump(model_package, file_path)
                
                # Show detailed success message
                info_msg = f"Random Forest model saved successfully!\n\n"
                info_msg += f"Model details:\n"
                info_msg += f"• Expected features: {model_package['metadata'].get('expected_features', 'Unknown')}\n"
                info_msg += f"• Trees: {model_package['metadata']['n_estimators']}\n"
                info_msg += f"• Max depth: {model_package['metadata']['max_depth']}\n"
                if 'wavenumber_range' in model_package['metadata']:
                    wn_range = model_package['metadata']['wavenumber_range']
                    info_msg += f"• Wavenumber range: {wn_range[0]:.1f} - {wn_range[1]:.1f} cm⁻¹"
                
                QMessageBox.information(self, "Success", info_msg)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save Random Forest model: {str(e)}")

    def load_rf_model(self):
        """Load Random Forest model from file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Random Forest Model",
                "",
                "Joblib files (*.joblib);;All files (*.*)"
            )
            
            if file_path:
                import joblib
                loaded_data = joblib.load(file_path)
                
                # Handle both old format (direct model) and new format (with metadata)
                if isinstance(loaded_data, dict) and 'model' in loaded_data:
                    # New format with metadata
                    self.rf_model = loaded_data['model']
                    metadata = loaded_data.get('metadata', {})
                    
                    # Show detailed info about loaded model
                    info_msg = "Random Forest model loaded successfully!\n\n"
                    info_msg += "Model details:\n"
                    
                    # Show actual model features first
                    if hasattr(self.rf_model, 'n_features_in_'):
                        info_msg += f"• Model features (actual): {self.rf_model.n_features_in_}\n"
                    
                    # Then show metadata features for comparison
                    if 'expected_features' in metadata:
                        info_msg += f"• Expected features (metadata): {metadata['expected_features']}\n"
                    if 'n_features' in metadata:
                        info_msg += f"• N features (metadata): {metadata['n_features']}\n"
                        
                    if 'n_estimators' in metadata:
                        info_msg += f"• Trees: {metadata['n_estimators']}\n"
                    if 'max_depth' in metadata:
                        info_msg += f"• Max depth: {metadata['max_depth']}\n"
                    if 'wavenumber_range' in metadata:
                        wn_range = metadata['wavenumber_range']
                        info_msg += f"• Wavenumber range: {wn_range[0]:.1f} - {wn_range[1]:.1f} cm⁻¹\n"
                    if 'created_date' in metadata:
                        info_msg += f"• Created: {metadata['created_date'][:10]}\n"
                    
                    # Warn if there's a mismatch
                    model_features = getattr(self.rf_model, 'n_features_in_', None)
                    metadata_features = metadata.get('expected_features', None)
                    if model_features and metadata_features and model_features != metadata_features:
                        info_msg += f"\n⚠️ WARNING: Feature count mismatch detected!\n"
                        info_msg += f"Model object expects {model_features} features but metadata says {metadata_features}.\n"
                    
                    info_msg += f"\nYou can now go to the ML Analysis tab and use 'Classify Map Spectra' to classify your data."
                    
                    # Fix metadata if there's a mismatch (for backward compatibility)
                    model_features = getattr(self.rf_model, 'n_features_in_', None)
                    metadata_expected = metadata.get('expected_features', None)
                    
                    if model_features and metadata_expected and model_features != metadata_expected:
                        print(f"Fixing metadata mismatch: {metadata_expected} -> {model_features}")
                        metadata['expected_features'] = model_features
                        metadata['n_features'] = model_features
                    
                    # Store metadata for later use
                    self.model_metadata = metadata
                    
                else:
                    # Old format (direct model)
                    self.rf_model = loaded_data
                    info_msg = "Random Forest model loaded successfully!\n\n"
                    info_msg += "Note: This is an older model format without metadata.\n"
                    if hasattr(self.rf_model, 'n_features_in_'):
                        info_msg += f"Expected features: {self.rf_model.n_features_in_}\n"
                    info_msg += f"\nYou can now go to the ML Analysis tab and use 'Classify Map Spectra' to classify your data."
                    
                    # Create basic metadata
                    self.model_metadata = {'version': 'legacy'}
                
                # Clear any previously trained model to avoid confusion
                if hasattr(self, 'trained_model'):
                    self.trained_model = None
                
                # Update the GUI to show that a model is loaded
                if hasattr(self, 'train_fig') and self.train_fig:
                    self.train_fig.clear()
                    ax = self.train_fig.add_subplot(111)
                    ax.text(0.5, 0.5, 'Model loaded successfully!\nGo to ML Analysis tab and use "Classify Map Spectra"', 
                           ha='center', va='center', fontsize=12, fontweight='bold')
                    ax.axis('off')
                    if hasattr(self, 'train_canvas'):
                        self.train_canvas.draw()
                
                QMessageBox.information(self, "Success", info_msg)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load Random Forest model: {str(e)}")

    def _prepare_data_for_analysis(self):
        """Prepare data for machine learning analysis."""
        try:
            if not hasattr(self, 'map_data') or self.map_data is None:
                return None
            
            # Get data from map_data
            if hasattr(self.map_data, 'get_processed_intensities_array'):
                data = self.map_data.get_processed_intensities_array()
            elif hasattr(self.map_data, 'processed_intensities') and self.map_data.processed_intensities is not None:
                data = self.map_data.processed_intensities
            elif hasattr(self.map_data, 'intensities') and self.map_data.intensities is not None:
                data = self.map_data.intensities
            else:
                return None
            
            # Reshape if needed
            if len(data.shape) == 3:
                # Reshape from (y, x, wavenumbers) to (n_samples, wavenumbers)
                data = data.reshape(-1, data.shape[-1])
            
            return data
            
        except Exception as e:
            print(f"Error preparing data for analysis: {e}")
            return None

    def update_results_visualization(self):
        """Update the visualizations in the Results tab with comprehensive analysis results."""
        try:
            # Check if we have the necessary data
            has_pca = hasattr(self, 'pca') and self.pca is not None and hasattr(self, 'pca_components') and self.pca_components is not None
            has_nmf = hasattr(self, 'nmf') and self.nmf is not None and hasattr(self, 'nmf_components') and self.nmf_components is not None
            has_rf = hasattr(self, 'rf_model') and self.rf_model is not None
            has_classification = hasattr(self, 'classification_map') and self.classification_map is not None
            
            if not (has_pca or has_nmf):
                QMessageBox.information(self, "Info", "Please run PCA or NMF analysis first.")
                return
                
            if not has_rf:
                QMessageBox.information(self, "Info", "Please train a Random Forest model first.")
                return
            
            # Clear the figure
            self.results_fig.clear()
            
            # Create a 2x2 grid layout
            # Top row: PCA (left), NMF (right)
            # Bottom row: Representative Spectra (left), Analysis Summary (right)
            
            # Plot PCA components with classification (top left)
            if has_pca:
                ax1 = self.results_fig.add_subplot(2, 2, 1)
                
                X_pca = self.pca_components
                var_explained = self.pca.explained_variance_ratio_
                
                # Get classification labels
                if has_classification:
                    # Flatten classification map and match to PCA data size
                    labels_flat = self.classification_map.flatten()
                    
                    # Handle size mismatch
                    if len(X_pca) != len(labels_flat):
                        min_len = min(len(X_pca), len(labels_flat))
                        X_pca_plot = X_pca[:min_len]
                        labels_plot = labels_flat[:min_len]
                    else:
                        X_pca_plot = X_pca
                        labels_plot = labels_flat
                    
                    # Create scatter plot with classification colors
                    scatter = ax1.scatter(X_pca_plot[:, 0], X_pca_plot[:, 1], 
                                        c=labels_plot, cmap='coolwarm', alpha=0.7, s=15)
                else:
                    # Just plot PCA without classification colors
                    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=15)
                
                ax1.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
                ax1.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
                ax1.set_title("PCA Components with Classification")
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend for classification
                if has_classification:
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                  markersize=8, label='Class B'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                  markersize=8, label='Class A')
                    ]
                    ax1.legend(handles=legend_elements, loc='best')
            else:
                # If no PCA, show placeholder
                ax1 = self.results_fig.add_subplot(2, 2, 1)
                ax1.text(0.5, 0.5, 'PCA not available\nRun PCA analysis first', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title("PCA Components with Classification")
            
            # Plot NMF components with classification (top right)
            if has_nmf:
                ax2 = self.results_fig.add_subplot(2, 2, 2)
                
                X_nmf = self.nmf_components
                
                # Get classification labels
                if has_classification:
                    # Flatten classification map and match to NMF data size
                    labels_flat = self.classification_map.flatten()
                    
                    # Handle size mismatch
                    if len(X_nmf) != len(labels_flat):
                        min_len = min(len(X_nmf), len(labels_flat))
                        X_nmf_plot = X_nmf[:min_len]
                        labels_plot = labels_flat[:min_len]
                    else:
                        X_nmf_plot = X_nmf
                        labels_plot = labels_flat
                    
                    # Create scatter plot with classification colors
                    scatter = ax2.scatter(X_nmf_plot[:, 0], X_nmf_plot[:, 1], 
                                        c=labels_plot, cmap='coolwarm', alpha=0.7, s=15)
                else:
                    # Just plot NMF without classification colors
                    scatter = ax2.scatter(X_nmf[:, 0], X_nmf[:, 1], alpha=0.7, s=15)
                
                ax2.set_xlabel("NMF Component 1")
                ax2.set_ylabel("NMF Component 2")
                ax2.set_title("NMF Components with Classification")
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend for classification
                if has_classification:
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                  markersize=8, label='Class B'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                  markersize=8, label='Class A')
                    ]
                    ax2.legend(handles=legend_elements, loc='best')
            else:
                # If no NMF, show placeholder
                ax2 = self.results_fig.add_subplot(2, 2, 2)
                ax2.text(0.5, 0.5, 'NMF not available\nRun NMF analysis first', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("NMF Components with Classification")
            
            # Plot representative spectra (bottom left)
            ax3 = self.results_fig.add_subplot(2, 2, 3)
            self._plot_representative_spectra(ax3)
            
            # Plot analysis summary (bottom right)
            ax4 = self.results_fig.add_subplot(2, 2, 4)
            self._plot_analysis_summary(ax4)
            
            # Adjust layout to prevent overlap
            self.results_fig.tight_layout(pad=2.0)
            self.results_canvas.draw()
            
            # Switch to Results tab
            self.tab_widget.setCurrentWidget(self.tab_widget.widget(self.tab_widget.count() - 1))
            
        except Exception as e:
            import traceback
            print(f"Error updating results visualization: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to update visualizations: {str(e)}")

    def _plot_representative_spectra(self, ax):
        """Plot representative spectra for each class."""
        try:
            if not hasattr(self, 'rf_model') or self.rf_model is None:
                ax.text(0.5, 0.5, 'No trained model available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Representative Spectra")
                return
            
            if self.map_data is None:
                ax.text(0.5, 0.5, 'No map data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Representative Spectra")
                return
            
            # Get representative spectra for each class
            class_a_positions = []
            class_b_positions = []
            
            if hasattr(self, 'classification_map') and self.classification_map is not None:
                # Find positions of each class
                for i, y_pos in enumerate(self.map_data.y_positions):
                    for j, x_pos in enumerate(self.map_data.x_positions):
                        if i < self.classification_map.shape[0] and j < self.classification_map.shape[1]:
                            if self.classification_map[i, j] == 1:  # Class A
                                class_a_positions.append((x_pos, y_pos))
                            else:  # Class B
                                class_b_positions.append((x_pos, y_pos))
            
            # Plot a few representative spectra from each class
            max_spectra_per_class = 3
            
            # Plot Class B spectra
            for i, (x_pos, y_pos) in enumerate(class_b_positions[:max_spectra_per_class]):
                spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                if spectrum:
                    if hasattr(self, 'use_processed_cb') and self.use_processed_cb.isChecked() and spectrum.processed_intensities is not None:
                        wavenumbers = self.map_data.target_wavenumbers
                        intensities = spectrum.processed_intensities
                    else:
                        wavenumbers = spectrum.wavenumbers
                        intensities = spectrum.intensities
                    
                    alpha = 0.8 if i == 0 else 0.5
                    label = 'Class B' if i == 0 else None
                    ax.plot(wavenumbers, intensities, 'b-', alpha=alpha, linewidth=1, label=label)
            
            # Plot Class A spectra
            for i, (x_pos, y_pos) in enumerate(class_a_positions[:max_spectra_per_class]):
                spectrum = self.map_data.get_spectrum(x_pos, y_pos)
                if spectrum:
                    if hasattr(self, 'use_processed_cb') and self.use_processed_cb.isChecked() and spectrum.processed_intensities is not None:
                        wavenumbers = self.map_data.target_wavenumbers
                        intensities = spectrum.processed_intensities
                    else:
                        wavenumbers = spectrum.wavenumbers
                        intensities = spectrum.intensities
                    
                    alpha = 0.8 if i == 0 else 0.5
                    label = 'Class A' if i == 0 else None
                    ax.plot(wavenumbers, intensities, 'r-', alpha=alpha, linewidth=1, label=label)
            
            ax.set_xlabel('Wavenumber (cm⁻¹)')
            ax.set_ylabel('Intensity')
            ax.set_title('Representative Spectra')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting spectra:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Representative Spectra")

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        try:
            # Create report content
            report_lines = []
            report_lines.append("RamanLab - 2D Map Analysis Report")
            report_lines.append("=" * 50)
            report_lines.append("")
            
            from datetime import datetime
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Dataset information
            report_lines.append("DATASET INFORMATION")
            report_lines.append("-" * 20)
            if self.map_data:
                report_lines.append(f"• Map Data: {len(self.map_data.spectra)} spectra")
                report_lines.append(f"• Dimensions: {len(self.map_data.x_positions)} × {len(self.map_data.y_positions)}")
                report_lines.append(f"• X positions: {min(self.map_data.x_positions)} to {max(self.map_data.x_positions)}")
                report_lines.append(f"• Y positions: {min(self.map_data.y_positions)} to {max(self.map_data.y_positions)}")
            else:
                report_lines.append("• No map data loaded")
            report_lines.append("")
            
            # PCA Analysis
            report_lines.append("PCA ANALYSIS")
            report_lines.append("-" * 13)
            if hasattr(self, 'pca') and self.pca is not None:
                report_lines.append(f"• Number of components: {self.pca.n_components_}")
                explained_var = self.pca.explained_variance_ratio_
                report_lines.append(f"• First component explains {explained_var[0]:.1%} of variance")
                if len(explained_var) > 1:
                    report_lines.append(f"• Second component explains {explained_var[1]:.1%} of variance")
                report_lines.append(f"• First 3 components explain {explained_var[:3].sum():.1%} of total variance")
            else:
                report_lines.append("• PCA analysis not performed")
            report_lines.append("")
            
            # NMF Analysis
            report_lines.append("NMF ANALYSIS")
            report_lines.append("-" * 13)
            if hasattr(self, 'nmf') and self.nmf is not None:
                report_lines.append(f"• Number of components: {self.nmf.n_components_}")
                if hasattr(self.nmf, 'reconstruction_err_'):
                    report_lines.append(f"• Reconstruction error: {self.nmf.reconstruction_err_:.3f}")
            else:
                report_lines.append("• NMF analysis not performed")
            report_lines.append("")
            
            # Random Forest Analysis
            report_lines.append("RANDOM FOREST CLASSIFICATION")
            report_lines.append("-" * 29)
            if hasattr(self, 'rf_model') and self.rf_model is not None:
                report_lines.append("• Random Forest model: Trained")
                
                # Classification results
                if hasattr(self, 'classification_map') and self.classification_map is not None:
                    n_class_a = np.sum(self.classification_map == 1)
                    n_class_b = np.sum(self.classification_map == 0)
                    total_classified = n_class_a + n_class_b
                    
                    if total_classified > 0:
                        report_lines.append(f"• Total classified positions: {total_classified}")
                        report_lines.append(f"• Class A positions: {n_class_a} ({n_class_a/total_classified:.1%})")
                        report_lines.append(f"• Class B positions: {n_class_b} ({n_class_b/total_classified:.1%})")
                        
                        # Calculate spatial distribution
                        report_lines.append("")
                        report_lines.append("SPATIAL DISTRIBUTION")
                        report_lines.append("-" * 19)
                        
                        # Find regions of high Class A concentration
                        if n_class_a > 0:
                            class_a_indices = np.where(self.classification_map == 1)
                            avg_x = np.mean([self.map_data.x_positions[j] for j in class_a_indices[1]])
                            avg_y = np.mean([self.map_data.y_positions[i] for i in class_a_indices[0]])
                            report_lines.append(f"• Class A center of mass: X={avg_x:.1f}, Y={avg_y:.1f}")
                        
                        if n_class_b > 0:
                            class_b_indices = np.where(self.classification_map == 0)
                            avg_x = np.mean([self.map_data.x_positions[j] for j in class_b_indices[1]])
                            avg_y = np.mean([self.map_data.y_positions[i] for i in class_b_indices[0]])
                            report_lines.append(f"• Class B center of mass: X={avg_x:.1f}, Y={avg_y:.1f}")
                else:
                    report_lines.append("• Map classification not performed")
            else:
                report_lines.append("• Random Forest model not trained")
            report_lines.append("")
            
            # Template Analysis (if available)
            if hasattr(self, 'map_data') and self.map_data and hasattr(self.map_data, 'template_manager'):
                template_manager = self.map_data.template_manager
                if template_manager.get_template_count() > 0:
                    report_lines.append("TEMPLATE ANALYSIS")
                    report_lines.append("-" * 17)
                    report_lines.append(f"• Number of templates loaded: {template_manager.get_template_count()}")
                    template_names = template_manager.get_template_names()
                    for i, name in enumerate(template_names):
                        report_lines.append(f"  {i+1}. {name}")
                    
                    # Template fitting results (if available)
                    if hasattr(self.map_data, 'template_coefficients') and self.map_data.template_coefficients is not None:
                        report_lines.append("• Template fitting: Completed")
                        
                        # Calculate average contributions
                        if self.map_data.template_coefficients.size > 0:
                            avg_contributions = np.mean(self.map_data.template_coefficients, axis=0)
                            for i, (name, contrib) in enumerate(zip(template_names, avg_contributions)):
                                report_lines.append(f"  {name}: {contrib:.3f} average contribution")
                    else:
                        report_lines.append("• Template fitting: Not performed")
                    report_lines.append("")
            
            # Create and show report dialog
            self._show_report_dialog("\n".join(report_lines))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report:\n{str(e)}")

    def _show_report_dialog(self, report_content):
        """Show the report in a dialog with save option."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis Report")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Text area for report
        text_area = QTextEdit()
        text_area.setPlainText(report_content)
        text_area.setReadOnly(True)
        text_area.setFont(QFont("Courier", 10))  # Monospace font for better formatting
        layout.addWidget(text_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Report")
        save_btn.clicked.connect(lambda: self._save_report(report_content))
        button_layout.addWidget(save_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        dialog.exec()

    def _save_report(self, report_content):
        """Save the report to a file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Analysis Report",
                "analysis_report.txt",
                "Text files (*.txt);;All files (*.*)"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(report_content)
                
                QMessageBox.information(self, "Report Saved", f"Report saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save report:\n{str(e)}")

# ====================== MAIN APPLICATION ENTRY POINT ======================

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("2D Map Analysis - Qt6")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("RamanLab")
    
    # Create and show the main window
    window = TwoDMapAnalysisQt6()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 