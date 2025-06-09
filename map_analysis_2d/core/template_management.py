"""Template spectrum management for Raman analysis."""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

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
    
    def get_template_count(self) -> int:
        """Get the number of loaded templates."""
        return len(self.templates)
    
    def clear_templates(self):
        """Clear all loaded templates."""
        self.templates.clear()
    
    def remove_template(self, index: int) -> bool:
        """Remove a template by index."""
        try:
            if 0 <= index < len(self.templates):
                self.templates.pop(index)
                return True
            return False
        except Exception:
            return False
    
    def get_template_names(self) -> List[str]:
        """Get list of template names."""
        return [template.name for template in self.templates]
    
    def validate_and_fix_template_dimensions(self, expected_length: int):
        """Validate and fix template dimensions to match expected length."""
        for template in self.templates:
            if template.processed_intensities is not None:
                if len(template.processed_intensities) != expected_length:
                    # Reprocess with current target wavenumbers
                    template.processed_intensities = self._preprocess_spectrum(
                        template.wavenumbers, template.intensities
                    )
    
    def fit_spectrum(self, spectrum_intensities: np.ndarray, method: str = 'nnls', 
                    use_baseline: bool = True) -> tuple:
        """
        Fit templates to a spectrum.
        
        Parameters:
        -----------
        spectrum_intensities : np.ndarray
            Spectrum intensities to fit
        method : str
            Fitting method ('nnls' or 'lstsq')
        use_baseline : bool
            Whether to include a baseline component
            
        Returns:
        --------
        tuple
            (coefficients, r_squared)
        """
        try:
            template_matrix = self.get_template_matrix()
            
            if template_matrix.size == 0:
                return np.array([]), 0.0
            
            # Add baseline if requested
            if use_baseline:
                baseline = np.ones(len(spectrum_intensities))
                template_matrix = np.column_stack([template_matrix, baseline])
            
            # Fit using specified method
            if method == 'nnls':
                from scipy.optimize import nnls
                coeffs, residual = nnls(template_matrix, spectrum_intensities)
            else:  # lstsq
                coeffs, residuals, rank, s = np.linalg.lstsq(
                    template_matrix, spectrum_intensities, rcond=None
                )
                residual = residuals[0] if len(residuals) > 0 else 0
            
            # Calculate R-squared
            y_mean = np.mean(spectrum_intensities)
            ss_tot = np.sum((spectrum_intensities - y_mean) ** 2)
            ss_res = residual
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return coeffs, r_squared
            
        except Exception as e:
            logger.error(f"Error fitting spectrum: {str(e)}")
            return np.array([]), 0.0 