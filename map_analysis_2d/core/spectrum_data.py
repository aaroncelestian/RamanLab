"""Data structures for spectrum data."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpectrumData:
    """Class to hold spectrum data and metadata."""
    x_pos: int
    y_pos: int
    wavenumbers: np.ndarray
    intensities: np.ndarray
    filename: str
    processed_intensities: Optional[np.ndarray] = None 