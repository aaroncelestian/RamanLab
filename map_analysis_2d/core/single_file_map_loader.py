"""Loader for single-file 2D Raman map format.

This module handles Raman map data stored in a single text file where:
- Line 1: Wavenumber axis (tab-delimited)
- Lines 2+: X_position Y_position intensity1 intensity2 ... intensityN (tab-delimited)

Example format:
    100.093    103.274    106.457    ...    (wavenumbers)
    -5000    -5000    2575    2631    2623    ...    (X, Y, intensities)
    -5000    -4965.04    4249    4371    4389    ...
    ...
"""

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


class SingleFileRamanMapData:
    """Class to manage single-file Raman map data."""
    
    def __init__(self, filepath: str, target_wavenumbers: Optional[np.ndarray] = None,
                 cosmic_ray_config: CosmicRayConfig = None, progress_callback=None):
        """
        Initialize SingleFileRamanMapData.
        
        Parameters:
        -----------
        filepath : str
            Path to the single file containing all Raman map data
        target_wavenumbers : Optional[np.ndarray]
            Target wavenumber values for resampling. If None, uses file's wavenumbers.
        cosmic_ray_config : CosmicRayConfig
            Configuration for cosmic ray detection and removal
        progress_callback : callable, optional
            Function to call with progress updates (progress, message)
        """
        self.filepath = Path(filepath)
        self.spectra: Dict[Tuple[float, float], SpectrumData] = {}
        self.x_positions: List[float] = []
        self.y_positions: List[float] = []
        self.cosmic_ray_map: Optional[np.ndarray] = None
        
        # Initialize centralized cosmic ray manager
        self.cosmic_ray_manager = SimpleCosmicRayManager(cosmic_ray_config)
        
        self.template_manager = TemplateSpectraManager(target_wavenumbers)
        self.template_coefficients: Optional[np.ndarray] = None
        self.template_residuals: Optional[np.ndarray] = None
        
        # Target wavenumbers will be set from file
        self.target_wavenumbers = target_wavenumbers
        
        # Load data automatically
        self._load_data(progress_callback)
    
    def _preprocess_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Preprocess spectrum data."""
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
            clean_wn = wavenumbers[valid_mask]
            clean_int = intensities[valid_mask]
            
            if len(clean_wn) < 2:
                return np.zeros_like(self.target_wavenumbers)
            
            # If wavenumbers match target, no interpolation needed
            if np.allclose(clean_wn, self.target_wavenumbers):
                resampled = clean_int
            else:
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
    
    def _load_data(self, progress_callback=None):
        """Load all spectra from the single file."""
        try:
            if not self.filepath.exists():
                raise FileNotFoundError(f"File not found: {self.filepath}")
            
            logger.info(f"Loading single-file Raman map: {self.filepath}")
            
            if progress_callback:
                progress_callback(10, "Reading file...")
            
            # Read the entire file
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                raise ValueError("File must have at least 2 lines (wavenumbers + data)")
            
            # Parse wavenumber axis from first line
            wavenumber_line = lines[0].strip().split('\t')
            wavenumbers = np.array([float(x) for x in wavenumber_line if x])
            
            logger.info(f"Wavenumber range: {wavenumbers[0]:.1f} to {wavenumbers[-1]:.1f} cm⁻¹")
            logger.info(f"Number of wavenumber points: {len(wavenumbers)}")
            
            # Set target wavenumbers from file if not provided
            if self.target_wavenumbers is None:
                self.target_wavenumbers = wavenumbers
            
            # Update template manager
            self.template_manager.target_wavenumbers = self.target_wavenumbers
            
            if progress_callback:
                progress_callback(20, "Parsing spectra...")
            
            # Parse data lines
            total_lines = len(lines) - 1  # Exclude header
            loaded_count = 0
            
            for i, line in enumerate(lines[1:], start=1):
                if progress_callback and i % 50 == 0:
                    progress = 20 + int((i / total_lines) * 70)
                    progress_callback(progress, f"Loading spectrum {i}/{total_lines}")
                
                parts = line.strip().split('\t')
                if len(parts) < 3:  # Need at least X, Y, and one intensity
                    continue
                
                try:
                    # Extract position
                    x_pos = float(parts[0])
                    y_pos = float(parts[1])
                    
                    # Extract intensities
                    intensities = np.array([float(x) for x in parts[2:] if x])
                    
                    if len(intensities) != len(wavenumbers):
                        logger.warning(f"Line {i}: Expected {len(wavenumbers)} intensities, got {len(intensities)}")
                        continue
                    
                    # Apply cosmic ray detection if enabled
                    if self.cosmic_ray_manager.config.apply_during_load:
                        spectrum_id = f"pos_{x_pos}_{y_pos}"
                        cosmic_detected, cleaned_intensities, detection_info = \
                            self.cosmic_ray_manager.detect_and_remove_cosmic_rays(
                                wavenumbers, intensities, spectrum_id
                            )
                        
                        if cosmic_detected:
                            logger.debug(f"Cosmic rays detected at ({x_pos}, {y_pos}): "
                                       f"{len(detection_info['cosmic_ray_indices'])} spikes removed")
                            intensities = cleaned_intensities
                    
                    # Preprocess spectrum
                    processed_intensities = self._preprocess_spectrum(wavenumbers, intensities)
                    
                    # Create SpectrumData object
                    spectrum_data = SpectrumData(
                        x_pos=x_pos,
                        y_pos=y_pos,
                        wavenumbers=wavenumbers,
                        intensities=intensities,
                        filename=f"{self.filepath.name}_pos_{x_pos}_{y_pos}",
                        processed_intensities=processed_intensities
                    )
                    
                    self.spectra[(x_pos, y_pos)] = spectrum_data
                    loaded_count += 1
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line {i}: {str(e)}")
                    continue
            
            if loaded_count == 0:
                raise ValueError("No valid spectra could be loaded")
            
            # Extract unique positions and sort them
            positions = list(self.spectra.keys())
            self.x_positions = sorted(list(set(pos[0] for pos in positions)))
            self.y_positions = sorted(list(set(pos[1] for pos in positions)))
            
            logger.info(f"Successfully loaded {loaded_count} spectra")
            logger.info(f"Map dimensions: {len(self.x_positions)} X positions × {len(self.y_positions)} Y positions")
            logger.info(f"X range: {self.x_positions[0]} to {self.x_positions[-1]}")
            logger.info(f"Y range: {self.y_positions[0]} to {self.y_positions[-1]}")
            
            if progress_callback:
                progress_callback(100, f"Loaded {loaded_count} spectra")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_spectrum(self, x_pos: float, y_pos: float) -> Optional[SpectrumData]:
        """Get spectrum at specified position."""
        return self.spectra.get((x_pos, y_pos))
    
    def get_map_dimensions(self) -> Tuple[int, int]:
        """Get map dimensions (width, height)."""
        return len(self.x_positions), len(self.y_positions)
    
    @property
    def width(self) -> int:
        """Get map width (number of X positions)."""
        return len(self.x_positions)
    
    @property
    def height(self) -> int:
        """Get map height (number of Y positions)."""
        return len(self.y_positions)
    
    def get_processed_data_matrix(self) -> np.ndarray:
        """Get matrix of all processed spectra in row-major grid order.
        
        Returns spectra ordered by (y, x) position so that reshaping to 
        (n_y, n_x, n_wavenumbers) preserves spatial relationships.
        """
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
        
        # CRITICAL: Iterate in consistent row-major order (y, then x)
        # This ensures reshaping to (n_y, n_x) preserves spatial layout
        spectrum_index = 0
        for y in sorted(self.y_positions):
            for x in sorted(self.x_positions):
                if (x, y) in self.spectra:
                    spectrum = self.spectra[(x, y)]
                    if spectrum.processed_intensities is not None:
                        data_matrix[spectrum_index, :] = spectrum.processed_intensities
                    else:
                        # Fallback to preprocessing on the fly
                        processed = self._preprocess_spectrum(spectrum.wavenumbers, spectrum.intensities)
                        data_matrix[spectrum_index, :] = processed
                    spectrum_index += 1
        
        return data_matrix
    
    def get_position_list(self) -> List[Tuple[float, float]]:
        """Get list of (x, y) positions."""
        return list(self.spectra.keys())
    
    def get_grid_indices(self, x_pos: float, y_pos: float) -> Tuple[int, int]:
        """Convert physical position to grid indices."""
        try:
            x_idx = self.x_positions.index(x_pos)
            y_idx = self.y_positions.index(y_pos)
            return x_idx, y_idx
        except ValueError:
            return None, None
