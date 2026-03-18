"""Data structures for spectrum data."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


@dataclass
class SpectrumData:
    """Class to hold spectrum data and metadata."""
    x_pos: int
    y_pos: int
    wavenumbers: np.ndarray
    intensities: np.ndarray
    filename: str
    processed_intensities: Optional[np.ndarray] = None


class SimpleMapData:
    """
    Simple map data structure for imported binary map files.
    
    This class provides a minimal structure compatible with the Map Analysis
    loader, containing spectra organized by (x, y) position.
    """
    
    def __init__(self, spectra_dict: Dict[Tuple[int, int], SpectrumData], 
                 wavenumbers: np.ndarray):
        """
        Initialize SimpleMapData.
        
        Parameters:
        -----------
        spectra_dict : Dict[Tuple[int, int], SpectrumData]
            Dictionary mapping (x, y) positions to SpectrumData objects
        wavenumbers : np.ndarray
            Common wavenumber axis for all spectra
        """
        self.spectra = spectra_dict
        self.target_wavenumbers = wavenumbers
        self.wavenumbers = wavenumbers
        self.x_positions = sorted(set(k[0] for k in spectra_dict.keys()))
        self.y_positions = sorted(set(k[1] for k in spectra_dict.keys())) 