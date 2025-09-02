"""File I/O operations for Raman map data."""

import os
import re
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from .spectrum_data import SpectrumData
from .cosmic_ray_detection import CosmicRayConfig, SimpleCosmicRayManager
from .template_management import TemplateSpectraManager

logger = logging.getLogger(__name__)


class RamanMapData:
    """Class to manage Raman map data and provide analysis capabilities."""
    
    def __init__(self, data_dir: str, target_wavenumbers: Optional[np.ndarray] = None, 
                 cosmic_ray_config: CosmicRayConfig = None, progress_callback=None):
        """
        Initialize RamanMapData.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the Raman map data files
        target_wavenumbers : Optional[np.ndarray]
            Target wavenumber values for resampling. If None, a default range is used.
        cosmic_ray_config : CosmicRayConfig
            Configuration for cosmic ray detection and removal
        progress_callback : callable, optional
            Function to call with progress updates (progress, message)
        """
        self.data_dir = data_dir
        self.spectra: Dict[Tuple[int, int], SpectrumData] = {}
        self.x_positions: List[int] = []
        self.y_positions: List[int] = []
        self.cosmic_ray_map: Optional[np.ndarray] = None
        
        # Initialize centralized cosmic ray manager
        self.cosmic_ray_manager = SimpleCosmicRayManager(cosmic_ray_config)
        
        self.template_manager = TemplateSpectraManager(target_wavenumbers)
        self.template_coefficients: Optional[np.ndarray] = None
        self.template_residuals: Optional[np.ndarray] = None
        
        # Target wavenumbers will be set intelligently
        self.target_wavenumbers = target_wavenumbers
        
        # Load data automatically
        self._load_data(progress_callback)
    
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
        
        # If no pattern matches, try to extract just two numbers
        numbers = re.findall(r'\d+', base_name)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        
        # Fallback: use hash of filename
        hash_val = hash(filename) % 1000
        logger.warning(f"Could not parse position from filename {filename}, using hash-based position {hash_val}")
        return hash_val, hash_val
    
    def _determine_target_wavenumbers_from_files(self, sample_files):
        """Determine target wavenumbers from a sample of data files."""
        for filepath in sample_files[:3]:  # Sample first 3 files
            try:
                # Load the first few files to determine wavenumber range
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
                                data = wavenumbers
                                break
                        except:
                            continue
                    
                    if data is None:
                        data = np.loadtxt(filepath)[:, 0]
                else:
                    data = np.loadtxt(filepath)[:, 0]
                
                if len(data) > 10:  # Reasonable amount of data
                    min_wn = np.min(data)
                    max_wn = np.max(data)
                    num_points = len(data)
                    
                    # Create target wavenumbers with appropriate resolution
                    self.target_wavenumbers = np.linspace(min_wn, max_wn, num_points)
                    logger.info(f"Determined target wavenumbers from {filepath.name}: "
                               f"{min_wn:.1f} to {max_wn:.1f} cm⁻¹ ({num_points} points)")
                    return
                    
            except Exception as e:
                logger.debug(f"Could not determine wavenumbers from {filepath}: {str(e)}")
                continue
        
        # Fallback if no files could be read
        self.target_wavenumbers = np.linspace(100, 3500, 400)
        logger.warning("Could not determine wavenumbers from files, using default range")
    
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
            
            # Apply centralized cosmic ray detection and cleaning at data import
            if self.cosmic_ray_manager.config.apply_during_load:
                spectrum_id = f"{filepath.name}_{x_pos}_{y_pos}"
                cosmic_detected, cleaned_intensities, detection_info = self.cosmic_ray_manager.detect_and_remove_cosmic_rays(
                    wavenumbers, intensities, spectrum_id
                )
                
                if cosmic_detected:
                    logger.debug(f"Cosmic rays detected and cleaned in {filepath.name}: "
                               f"{len(detection_info['cosmic_ray_indices'])} cosmic rays removed")
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
    
    def _load_data(self, progress_callback=None):
        """Load all spectra from the data directory."""
        try:
            data_path = Path(self.data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
            # Find all spectrum files, excluding macOS metadata files
            spectrum_files = []
            for ext in ['*.txt', '*.csv', '*.dat']:
                all_files = data_path.glob(ext)
                # Filter out macOS metadata files starting with '._'
                filtered_files = [f for f in all_files if not f.name.startswith('._')]
                spectrum_files.extend(filtered_files)
            
            if not spectrum_files:
                raise ValueError(f"No spectrum files found in {self.data_dir}")
            
            logger.info(f"Found {len(spectrum_files)} spectrum files")
            
            # Determine target wavenumbers if not provided
            if self.target_wavenumbers is None:
                self._determine_target_wavenumbers_from_files(spectrum_files)
            
            # Update template manager target wavenumbers
            self.template_manager.target_wavenumbers = self.target_wavenumbers
            
            # Load all spectra
            total_files = len(spectrum_files)
            loaded_count = 0
            
            for i, filepath in enumerate(spectrum_files):
                if progress_callback:
                    progress = int((i / total_files) * 100)
                    progress_callback(progress, f"Loading {filepath.name}")
                
                spectrum_data = self._load_spectrum(filepath)
                if spectrum_data is not None:
                    self.spectra[(spectrum_data.x_pos, spectrum_data.y_pos)] = spectrum_data
                    loaded_count += 1
            
            if loaded_count == 0:
                raise ValueError("No valid spectra could be loaded")
            
            # Extract unique positions and sort them
            positions = list(self.spectra.keys())
            self.x_positions = sorted(list(set(pos[0] for pos in positions)))
            self.y_positions = sorted(list(set(pos[1] for pos in positions)))
            
            logger.info(f"Successfully loaded {loaded_count} spectra")
            logger.info(f"Map dimensions: {len(self.x_positions)} x {len(self.y_positions)}")
            
            if progress_callback:
                progress_callback(100, f"Loaded {loaded_count} spectra")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_spectrum(self, x_pos: int, y_pos: int) -> Optional[SpectrumData]:
        """Get spectrum at specified position."""
        return self.spectra.get((x_pos, y_pos))
    
    def get_map_dimensions(self) -> Tuple[int, int]:
        """Get map dimensions (width, height)."""
        return len(self.x_positions), len(self.y_positions)
    
    @property
    def width(self) -> int:
        """Get map width."""
        return len(self.x_positions)
    
    @property 
    def height(self) -> int:
        """Get map height."""
        return len(self.y_positions)
    
    def get_processed_data_matrix(self) -> np.ndarray:
        """Get matrix of all processed spectra."""
        n_spectra = len(self.spectra)
        if n_spectra == 0:
            return np.array([])
        
        # Get the length from the first spectrum
        first_spectrum = next(iter(self.spectra.values()))
        if first_spectrum.processed_intensities is not None:
            n_points = len(first_spectrum.processed_intensities)
        else:
            n_points = len(self.target_wavenumbers)
        
        # Create matrix
        data_matrix = np.zeros((n_spectra, n_points))
        
        spectrum_index = 0
        for spectrum in self.spectra.values():
            if spectrum.processed_intensities is not None:
                data_matrix[spectrum_index, :] = spectrum.processed_intensities
            else:
                # Fallback to preprocessing on the fly
                processed = self._preprocess_spectrum(spectrum.wavenumbers, spectrum.intensities)
                data_matrix[spectrum_index, :] = processed
            spectrum_index += 1
        
        return data_matrix
    
    def get_position_list(self) -> List[Tuple[int, int]]:
        """Get list of (x, y) positions."""
        return list(self.spectra.keys()) 