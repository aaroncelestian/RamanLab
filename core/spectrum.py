"""
Core Spectrum Module for Raman Polarization Analyzer

This module provides comprehensive spectrum processing functionality including:
- Synthetic spectrum generation from mineral data
- Peak fitting functions (Lorentzian, Gaussian, Voigt)
- Spectrum data manipulation and validation
- Noise handling and smoothing algorithms

Dependencies:
    - numpy: Numerical computations
    - scipy: Signal processing and curve fitting
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from scipy import ndimage
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings


@dataclass
class SpectrumData:
    """Data structure for Raman spectrum information."""
    name: str
    wavenumbers: np.ndarray
    intensities: np.ndarray
    source: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate spectrum data after initialization."""
        if self.metadata is None:
            self.metadata = {}
        
        # Validate arrays
        if len(self.wavenumbers) != len(self.intensities):
            raise ValueError("Wavenumbers and intensities must have the same length")
        
        # Convert to numpy arrays if needed
        self.wavenumbers = np.asarray(self.wavenumbers)
        self.intensities = np.asarray(self.intensities)


@dataclass
class PeakData:
    """Data structure for individual peak information."""
    position: float
    amplitude: float
    width: float
    area: float
    shape: str = "lorentzian"
    parameters: List[float] = None
    r_squared: float = 0.0
    
    def __post_init__(self):
        """Initialize parameters if not provided."""
        if self.parameters is None:
            if self.shape == "voigt":
                self.parameters = [self.amplitude, self.position, self.width, self.width]
            else:
                self.parameters = [self.amplitude, self.position, self.width]


class SpectrumProcessor:
    """
    Advanced spectrum processing class with comprehensive functionality.
    
    This class provides methods for spectrum generation, peak fitting,
    noise handling, and data manipulation for Raman spectroscopy.
    """
    
    def __init__(self):
        """Initialize the spectrum processor."""
        self.intensity_map = {
            'very_weak': 0.1,
            'weak': 0.3,
            'medium': 0.6,
            'strong': 0.8,
            'very_strong': 1.0
        }
    
    def generate_spectrum_from_mineral(self, 
                                     mineral_data: Dict[str, Any],
                                     wavenumber_range: Tuple[float, float] = (100, 1200),
                                     num_points: int = 2200,
                                     peak_width: float = 10.0,
                                     noise_level: float = 0.02) -> Optional[SpectrumData]:
        """
        Generate a synthetic Raman spectrum from mineral database data.
        
        Args:
            mineral_data: Dictionary containing mineral information including raman_modes
            wavenumber_range: Tuple of (min, max) wavenumber values
            num_points: Number of data points in the spectrum
            peak_width: Default peak width for Lorentzian peaks
            noise_level: Standard deviation of Gaussian noise to add
            
        Returns:
            SpectrumData object containing the generated spectrum, or None if failed
        """
        if 'raman_modes' not in mineral_data:
            warnings.warn("No Raman modes found in mineral data")
            return None
        
        try:
            # Create wavenumber range
            wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], num_points)
            intensities = np.zeros_like(wavenumbers)
            
            # Add peaks for each Raman mode
            for mode in mineral_data['raman_modes']:
                frequency = mode['frequency']
                intensity_label = mode.get('intensity', 'medium')
                
                # Convert intensity label to numerical value
                intensity = self.intensity_map.get(intensity_label, 0.5)
                
                # Add Lorentzian peak
                peak_intensities = self.lorentzian(wavenumbers, intensity, frequency, peak_width)
                intensities += peak_intensities
            
            # Add noise if requested
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, len(intensities))
                intensities += noise
            
            # Ensure non-negative intensities
            intensities = np.maximum(intensities, 0)
            
            # Create metadata
            metadata = {
                'generation_method': 'synthetic_from_mineral',
                'mineral_name': mineral_data.get('name', 'unknown'),
                'num_modes': len(mineral_data['raman_modes']),
                'peak_width': peak_width,
                'noise_level': noise_level,
                'wavenumber_range': wavenumber_range
            }
            
            return SpectrumData(
                name=f"Synthetic_{mineral_data.get('name', 'Unknown')}",
                wavenumbers=wavenumbers,
                intensities=intensities,
                source='synthetic',
                metadata=metadata
            )
            
        except Exception as e:
            warnings.warn(f"Error generating spectrum: {e}")
            return None
    
    def generate_synthetic_spectrum(self,
                                  peaks: List[Dict[str, float]],
                                  wavenumber_range: Tuple[float, float] = (100, 1200),
                                  num_points: int = 2200,
                                  noise_level: float = 0.02) -> SpectrumData:
        """
        Generate a synthetic spectrum from peak parameters.
        
        Args:
            peaks: List of dictionaries with 'position', 'amplitude', 'width', 'shape' keys
            wavenumber_range: Tuple of (min, max) wavenumber values
            num_points: Number of data points
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            SpectrumData object containing the generated spectrum
        """
        wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], num_points)
        intensities = np.zeros_like(wavenumbers)
        
        for peak in peaks:
            position = peak['position']
            amplitude = peak['amplitude']
            width = peak['width']
            shape = peak.get('shape', 'lorentzian')
            
            if shape == 'lorentzian':
                peak_intensities = self.lorentzian(wavenumbers, amplitude, position, width)
            elif shape == 'gaussian':
                peak_intensities = self.gaussian(wavenumbers, amplitude, position, width)
            elif shape == 'voigt':
                width_g = peak.get('width_g', width)
                peak_intensities = self.voigt(wavenumbers, amplitude, position, width, width_g)
            else:
                continue  # Skip unknown peak shapes
            
            intensities += peak_intensities
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(intensities))
            intensities += noise
        
        intensities = np.maximum(intensities, 0)
        
        metadata = {
            'generation_method': 'synthetic_from_peaks',
            'num_peaks': len(peaks),
            'noise_level': noise_level,
            'wavenumber_range': wavenumber_range
        }
        
        return SpectrumData(
            name="Synthetic_Spectrum",
            wavenumbers=wavenumbers,
            intensities=intensities,
            source='synthetic',
            metadata=metadata
        )
    
    def smooth_spectrum(self, spectrum: SpectrumData, 
                       method: str = 'savgol',
                       window_length: int = 11,
                       polyorder: int = 3) -> SpectrumData:
        """
        Apply smoothing to a spectrum.
        
        Args:
            spectrum: Input SpectrumData object
            method: Smoothing method ('savgol', 'gaussian', 'moving_average')
            window_length: Length of the smoothing window
            polyorder: Polynomial order for Savitzky-Golay filter
            
        Returns:
            New SpectrumData object with smoothed intensities
        """
        if method == 'savgol':
            # Ensure odd window length
            if window_length % 2 == 0:
                window_length += 1
            smoothed = signal.savgol_filter(spectrum.intensities, window_length, polyorder)
        
        elif method == 'gaussian':
            sigma = window_length / 4.0
            smoothed = ndimage.gaussian_filter1d(spectrum.intensities, sigma)
        
        elif method == 'moving_average':
            window = np.ones(window_length) / window_length
            smoothed = np.convolve(spectrum.intensities, window, mode='same')
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # Create new spectrum with smoothed data
        new_metadata = spectrum.metadata.copy()
        new_metadata['smoothing_method'] = method
        new_metadata['smoothing_params'] = {
            'window_length': window_length,
            'polyorder': polyorder if method == 'savgol' else None
        }
        
        return SpectrumData(
            name=f"{spectrum.name}_smoothed",
            wavenumbers=spectrum.wavenumbers.copy(),
            intensities=smoothed,
            source=spectrum.source,
            metadata=new_metadata
        )
    
    def normalize_spectrum(self, spectrum: SpectrumData, 
                          method: str = 'max') -> SpectrumData:
        """
        Normalize spectrum intensities.
        
        Args:
            spectrum: Input SpectrumData object
            method: Normalization method ('max', 'area', 'std', 'minmax')
            
        Returns:
            New SpectrumData object with normalized intensities
        """
        intensities = spectrum.intensities.copy()
        
        if method == 'max':
            max_intensity = np.max(intensities)
            if max_intensity > 0:
                intensities = intensities / max_intensity
        
        elif method == 'area':
            total_area = np.trapz(intensities, spectrum.wavenumbers)
            if total_area > 0:
                intensities = intensities / total_area
        
        elif method == 'std':
            std_intensity = np.std(intensities)
            if std_intensity > 0:
                intensities = (intensities - np.mean(intensities)) / std_intensity
        
        elif method == 'minmax':
            min_intensity = np.min(intensities)
            max_intensity = np.max(intensities)
            if max_intensity > min_intensity:
                intensities = (intensities - min_intensity) / (max_intensity - min_intensity)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Create new spectrum with normalized data
        new_metadata = spectrum.metadata.copy()
        new_metadata['normalization_method'] = method
        
        return SpectrumData(
            name=f"{spectrum.name}_normalized",
            wavenumbers=spectrum.wavenumbers.copy(),
            intensities=intensities,
            source=spectrum.source,
            metadata=new_metadata
        )
    
    def baseline_correct(self, spectrum: SpectrumData,
                        method: str = 'polynomial',
                        degree: int = 3) -> SpectrumData:
        """
        Apply baseline correction to spectrum.
        
        Args:
            spectrum: Input SpectrumData object
            method: Baseline correction method ('polynomial', 'linear')
            degree: Polynomial degree for polynomial baseline
            
        Returns:
            New SpectrumData object with baseline-corrected intensities
        """
        x = spectrum.wavenumbers
        y = spectrum.intensities
        
        if method == 'polynomial':
            # Fit polynomial to the spectrum
            coefficients = np.polyfit(x, y, degree)
            baseline = np.polyval(coefficients, x)
        
        elif method == 'linear':
            # Linear baseline from first to last point
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            baseline = y[0] + slope * (x - x[0])
        
        else:
            raise ValueError(f"Unknown baseline correction method: {method}")
        
        corrected_intensities = y - baseline
        corrected_intensities = np.maximum(corrected_intensities, 0)  # Ensure non-negative
        
        # Create new spectrum with corrected data
        new_metadata = spectrum.metadata.copy()
        new_metadata['baseline_correction'] = {
            'method': method,
            'degree': degree if method == 'polynomial' else None
        }
        
        return SpectrumData(
            name=f"{spectrum.name}_baseline_corrected",
            wavenumbers=spectrum.wavenumbers.copy(),
            intensities=corrected_intensities,
            source=spectrum.source,
            metadata=new_metadata
        )
    
    @staticmethod
    def lorentzian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """
        Lorentzian peak function.
        
        Args:
            x: Array of x values (wavenumbers)
            amplitude: Peak amplitude
            center: Peak center position
            width: Peak half-width at half-maximum
            
        Returns:
            Array of y values (intensities)
        """
        return amplitude / (1 + ((x - center) / width) ** 2)
    
    @staticmethod
    def gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """
        Gaussian peak function.
        
        Args:
            x: Array of x values (wavenumbers)
            amplitude: Peak amplitude
            center: Peak center position
            width: Peak standard deviation
            
        Returns:
            Array of y values (intensities)
        """
        return amplitude * np.exp(-((x - center) / width) ** 2)
    
    @staticmethod
    def voigt(x: np.ndarray, amplitude: float, center: float, 
              width_l: float, width_g: float) -> np.ndarray:
        """
        Simplified Voigt profile (convolution of Lorentzian and Gaussian).
        
        Args:
            x: Array of x values (wavenumbers)
            amplitude: Peak amplitude
            center: Peak center position
            width_l: Lorentzian width parameter
            width_g: Gaussian width parameter
            
        Returns:
            Array of y values (intensities)
        """
        # Simplified approximation - for full Voigt, use scipy.special.voigt_profile
        lorentz = 1 / (1 + ((x - center) / width_l) ** 2)
        gauss = np.exp(-((x - center) / width_g) ** 2)
        return amplitude * lorentz * gauss
    
    def find_peaks(self, spectrum: SpectrumData,
                   prominence: float = 0.1,
                   distance: int = 10,
                   height: float = 0.05) -> List[Dict[str, float]]:
        """
        Find peaks in a spectrum using scipy.signal.find_peaks.
        
        Args:
            spectrum: Input SpectrumData object
            prominence: Required prominence of peaks
            distance: Minimum distance between peaks (in data points)
            height: Minimum height of peaks
            
        Returns:
            List of dictionaries containing peak information
        """
        # Normalize intensities for peak finding
        normalized_intensities = spectrum.intensities / np.max(spectrum.intensities)
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            normalized_intensities,
            prominence=prominence,
            distance=distance,
            height=height
        )
        
        # Extract peak information
        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_data = {
                'position': spectrum.wavenumbers[peak_idx],
                'amplitude': spectrum.intensities[peak_idx],
                'prominence': properties['prominences'][i],
                'width_samples': properties.get('widths', [10])[i] if 'widths' in properties else 10,
                'index': peak_idx
            }
            peak_list.append(peak_data)
        
        return peak_list
    
    def fit_peak(self, spectrum: SpectrumData, peak_position: float,
                 peak_shape: str = 'lorentzian',
                 fit_window: float = 50.0) -> Optional[PeakData]:
        """
        Fit a single peak in the spectrum.
        
        Args:
            spectrum: Input SpectrumData object
            peak_position: Approximate peak center position
            peak_shape: Peak shape ('lorentzian', 'gaussian', 'voigt')
            fit_window: Window around peak center for fitting
            
        Returns:
            PeakData object with fit results, or None if fitting failed
        """
        try:
            # Define fitting window
            mask = (spectrum.wavenumbers >= peak_position - fit_window) & \
                   (spectrum.wavenumbers <= peak_position + fit_window)
            
            if np.sum(mask) < 5:  # Need minimum points for fitting
                return None
            
            x_fit = spectrum.wavenumbers[mask]
            y_fit = spectrum.intensities[mask]
            
            # Initial parameter guesses
            peak_idx = np.argmax(y_fit)
            amplitude_guess = y_fit[peak_idx]
            center_guess = x_fit[peak_idx]
            width_guess = 10.0
            
            # Define fitting function
            if peak_shape == 'lorentzian':
                fit_func = self.lorentzian
                p0 = [amplitude_guess, center_guess, width_guess]
            elif peak_shape == 'gaussian':
                fit_func = self.gaussian
                p0 = [amplitude_guess, center_guess, width_guess]
            elif peak_shape == 'voigt':
                fit_func = self.voigt
                p0 = [amplitude_guess, center_guess, width_guess, width_guess]
            else:
                raise ValueError(f"Unknown peak shape: {peak_shape}")
            
            # Perform curve fitting
            popt, pcov = curve_fit(fit_func, x_fit, y_fit, p0=p0, maxfev=1000)
            
            # Calculate R-squared
            y_pred = fit_func(x_fit, *popt)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate peak area (approximate)
            if peak_shape in ['lorentzian', 'gaussian']:
                area = np.pi * popt[0] * abs(popt[2])  # Approximate area
            else:
                area = np.trapz(y_pred, x_fit)  # Numerical integration
            
            return PeakData(
                position=popt[1],
                amplitude=popt[0],
                width=abs(popt[2]) if len(popt) > 2 else 10.0,
                area=area,
                shape=peak_shape,
                parameters=list(popt),
                r_squared=r_squared
            )
            
        except Exception as e:
            warnings.warn(f"Peak fitting failed: {e}")
            return None


# Convenience functions for easy access to static methods
def lorentzian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Lorentzian peak function - convenience wrapper."""
    return SpectrumProcessor.lorentzian(x, amplitude, center, width)


def gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Gaussian peak function - convenience wrapper."""
    return SpectrumProcessor.gaussian(x, amplitude, center, width)


def voigt(x: np.ndarray, amplitude: float, center: float, 
          width_l: float, width_g: float) -> np.ndarray:
    """Voigt peak function - convenience wrapper."""
    return SpectrumProcessor.voigt(x, amplitude, center, width_l, width_g) 