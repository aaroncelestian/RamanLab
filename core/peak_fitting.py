"""
Peak Fitting Module for Raman Polarization Analyzer.

This module contains all peak fitting algorithms and related mathematical functions.
Extracted from the main application to improve modularity and testability.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class PeakData:
    """Data class for storing peak fitting results."""
    position: float
    amplitude: float
    width: float
    shape: str
    parameters: np.ndarray
    covariance: np.ndarray
    r_squared: float
    original_position: float
    fit_quality: str
    gamma: Optional[float] = None  # For Voigt profiles


class PeakFitter:
    """
    Advanced peak fitting class with multiple peak shapes and algorithms.
    """
    
    def __init__(self):
        """Initialize the peak fitter."""
        self.available_shapes = ["Lorentzian", "Gaussian", "Voigt", "Pseudo-Voigt"]
        
    @staticmethod
    def lorentzian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """
        Lorentzian peak function.
        
        Parameters:
        -----------
        x : array_like
            Independent variable (wavenumber)
        amplitude : float
            Peak amplitude
        center : float
            Peak center position
        width : float
            Peak width (HWHM)
            
        Returns:
        --------
        np.ndarray
            Calculated intensities
        """
        width = abs(width) + 1e-10  # Prevent division by zero
        return amplitude * (width**2) / ((x - center)**2 + width**2)
    
    @staticmethod
    def gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """
        Gaussian peak function.
        
        Parameters:
        -----------
        x : array_like
            Independent variable (wavenumber)
        amplitude : float
            Peak amplitude
        center : float
            Peak center position
        width : float
            Peak width (standard deviation)
            
        Returns:
        --------
        np.ndarray
            Calculated intensities
        """
        width = abs(width) + 1e-10  # Prevent division by zero
        return amplitude * np.exp(-((x - center) / width) ** 2)
    
    @staticmethod
    def voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float) -> np.ndarray:
        """
        Voigt profile (convolution of Gaussian and Lorentzian).
        This is a simplified approximation for computational efficiency.
        
        Parameters:
        -----------
        x : array_like
            Independent variable (wavenumber)
        amplitude : float
            Peak amplitude
        center : float
            Peak center position
        sigma : float
            Gaussian width parameter
        gamma : float
            Lorentzian width parameter
            
        Returns:
        --------
        np.ndarray
            Calculated intensities
        """
        # Simplified Voigt approximation using Faddeeva function alternative
        sigma = abs(sigma) + 1e-10
        gamma = abs(gamma) + 1e-10
        
        # Simple approximation: weighted sum of Gaussian and Lorentzian
        gaussian_part = np.exp(-((x - center) / sigma) ** 2)
        lorentzian_part = (gamma**2) / ((x - center)**2 + gamma**2)
        
        # Combine with empirical weighting
        return amplitude * (0.3989423 * gaussian_part + 0.6366198 * lorentzian_part)
    
    @staticmethod
    def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, width: float, eta: float) -> np.ndarray:
        """
        Pseudo-Voigt profile (linear combination of Gaussian and Lorentzian).
        
        Parameters:
        -----------
        x : array_like
            Independent variable (wavenumber)
        amplitude : float
            Peak amplitude
        center : float
            Peak center position
        width : float
            Peak width
        eta : float
            Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
            
        Returns:
        --------
        np.ndarray
            Calculated intensities
        """
        eta = np.clip(eta, 0, 1)  # Ensure eta is between 0 and 1
        
        gaussian_part = PeakFitter.gaussian(x, 1.0, center, width)
        lorentzian_part = PeakFitter.lorentzian(x, 1.0, center, width)
        
        return amplitude * ((1 - eta) * gaussian_part + eta * lorentzian_part)
    
    def estimate_initial_parameters(self, x: np.ndarray, y: np.ndarray, 
                                  peak_position: float, shape: str) -> Tuple[List[float], List[Tuple[float, float]]]:
        """
        Estimate initial parameters for peak fitting.
        
        Parameters:
        -----------
        x : array_like
            Wavenumber array
        y : array_like
            Intensity array
        peak_position : float
            Approximate peak center
        shape : str
            Peak shape ("Lorentzian", "Gaussian", "Voigt", "Pseudo-Voigt")
            
        Returns:
        --------
        tuple
            (initial_parameters, parameter_bounds)
        """
        # Find peak index
        peak_idx = np.argmin(np.abs(x - peak_position))
        amplitude = y[peak_idx]
        
        # Estimate width from FWHM
        half_max = amplitude / 2
        left_idx = peak_idx
        right_idx = peak_idx
        
        # Find left half-maximum
        while left_idx > 0 and y[left_idx] > half_max:
            left_idx -= 1
        
        # Find right half-maximum
        while right_idx < len(y) - 1 and y[right_idx] > half_max:
            right_idx += 1
        
        estimated_fwhm = x[right_idx] - x[left_idx] if right_idx > left_idx else 10.0
        width = max(2.0, min(estimated_fwhm / 2, 50.0))
        
        # Estimate baseline
        baseline = min(np.min(y), np.mean([y[0], y[-1]]))
        amplitude_corrected = amplitude - baseline
        
        # Set initial parameters and bounds based on shape
        if shape == "Lorentzian" or shape == "Gaussian":
            initial_params = [amplitude_corrected, peak_position, width]
            bounds = ([0, x[0], 0.5], [amplitude_corrected * 3, x[-1], 100.0])
        elif shape == "Voigt":
            initial_params = [amplitude_corrected, peak_position, width, width/2]
            bounds = ([0, x[0], 0.5, 0.1], [amplitude_corrected * 3, x[-1], 100.0, 100.0])
        elif shape == "Pseudo-Voigt":
            initial_params = [amplitude_corrected, peak_position, width, 0.5]
            bounds = ([0, x[0], 0.5, 0.0], [amplitude_corrected * 3, x[-1], 100.0, 1.0])
        else:
            raise ValueError(f"Unknown peak shape: {shape}")
        
        return initial_params, bounds
    
    def fit_single_peak(self, x: np.ndarray, y: np.ndarray, peak_position: float, 
                       shape: str = "Lorentzian", window_width: float = 50.0) -> Optional[PeakData]:
        """
        Fit a single peak in the spectrum.
        
        Parameters:
        -----------
        x : array_like
            Wavenumber array
        y : array_like
            Intensity array
        peak_position : float
            Approximate peak center position
        shape : str
            Peak shape to fit
        window_width : float
            Fitting window width in wavenumber units
            
        Returns:
        --------
        PeakData or None
            Fitted peak data or None if fitting failed
        """
        try:
            # Select fitting window
            mask = np.abs(x - peak_position) <= window_width
            if np.sum(mask) < 5:
                print(f"Insufficient data points for fitting peak at {peak_position}")
                return None
            
            x_fit = x[mask]
            y_fit = y[mask]
            
            # Get initial parameters and bounds
            initial_params, bounds = self.estimate_initial_parameters(x_fit, y_fit, peak_position, shape)
            
            # Select fitting function
            if shape == "Lorentzian":
                fit_func = self.lorentzian
            elif shape == "Gaussian":
                fit_func = self.gaussian
            elif shape == "Voigt":
                fit_func = self.voigt
            elif shape == "Pseudo-Voigt":
                fit_func = self.pseudo_voigt
            else:
                raise ValueError(f"Unknown peak shape: {shape}")
            
            # Perform fitting with bounds
            try:
                popt, pcov = curve_fit(fit_func, x_fit, y_fit,
                                     p0=initial_params,
                                     bounds=bounds,
                                     maxfev=5000,
                                     method='trf')
            except Exception as bounded_error:
                print(f"Bounded fitting failed, trying unbounded: {bounded_error}")
                # Try without bounds
                popt, pcov = curve_fit(fit_func, x_fit, y_fit,
                                     p0=initial_params,
                                     maxfev=10000)
            
            # Calculate R-squared
            y_pred = fit_func(x_fit, *popt)
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r_squared = max(0, min(1, r_squared))
            
            # Validate parameters
            fitted_center = popt[1]
            fitted_amplitude = popt[0]
            fitted_width = popt[2]
            
            # Quality assessment
            center_deviation = abs(fitted_center - peak_position)
            if center_deviation > 50.0:
                print(f"Warning: Large center deviation for peak at {peak_position}")
            
            fit_quality = 'excellent' if r_squared > 0.95 else \
                         'good' if r_squared > 0.8 else \
                         'fair' if r_squared > 0.5 else 'poor'
            
            # Create result object
            peak_data = PeakData(
                position=fitted_center,
                amplitude=fitted_amplitude,
                width=fitted_width,
                shape=shape,
                parameters=popt,
                covariance=pcov,
                r_squared=r_squared,
                original_position=peak_position,
                fit_quality=fit_quality
            )
            
            # Add gamma parameter for Voigt
            if shape == "Voigt" and len(popt) > 3:
                peak_data.gamma = popt[3]
            
            return peak_data
            
        except Exception as e:
            print(f"Error fitting peak at {peak_position}: {e}")
            return None
    
    def fit_multiple_peaks(self, x: np.ndarray, y: np.ndarray, peak_positions: List[float],
                          shape: str = "Lorentzian", window_width: float = 50.0) -> List[PeakData]:
        """
        Fit multiple peaks in the spectrum.
        
        Parameters:
        -----------
        x : array_like
            Wavenumber array
        y : array_like
            Intensity array
        peak_positions : list
            List of approximate peak center positions
        shape : str
            Peak shape to fit
        window_width : float
            Fitting window width in wavenumber units
            
        Returns:
        --------
        list
            List of fitted PeakData objects
        """
        fitted_peaks = []
        
        for peak_pos in peak_positions:
            peak_data = self.fit_single_peak(x, y, peak_pos, shape, window_width)
            if peak_data is not None:
                fitted_peaks.append(peak_data)
        
        return fitted_peaks
    
    def calculate_peak_properties(self, peak_data: PeakData) -> Dict[str, float]:
        """
        Calculate additional peak properties from fitted parameters.
        
        Parameters:
        -----------
        peak_data : PeakData
            Fitted peak data
            
        Returns:
        --------
        dict
            Dictionary of calculated properties
        """
        properties = {}
        
        if peak_data.shape in ["Lorentzian", "Gaussian"]:
            # For simple shapes
            properties['fwhm'] = 2 * peak_data.width
            properties['area'] = np.pi * peak_data.amplitude * peak_data.width
            
        elif peak_data.shape == "Voigt":
            # For Voigt profile
            sigma = peak_data.width
            gamma = peak_data.gamma if peak_data.gamma else peak_data.width / 2
            
            # Approximate FWHM for Voigt profile
            fwhm_g = 2 * sigma * np.sqrt(2 * np.log(2))  # Gaussian component
            fwhm_l = 2 * gamma  # Lorentzian component
            
            # Empirical formula for Voigt FWHM
            properties['fwhm'] = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
            properties['area'] = np.pi * peak_data.amplitude * properties['fwhm'] / 2
            
        # Standard errors from covariance matrix
        if peak_data.covariance is not None:
            param_errors = np.sqrt(np.diag(peak_data.covariance))
            properties['position_error'] = param_errors[1] if len(param_errors) > 1 else 0
            properties['amplitude_error'] = param_errors[0] if len(param_errors) > 0 else 0
            properties['width_error'] = param_errors[2] if len(param_errors) > 2 else 0
        
        return properties


# Utility functions for peak analysis
def auto_find_peaks(x: np.ndarray, y: np.ndarray, 
                   height_threshold: float = 0.1, 
                   distance: float = 10.0) -> List[float]:
    """
    Automatically find peaks in a spectrum.
    
    Parameters:
    -----------
    x : array_like
        Wavenumber array
    y : array_like
        Intensity array
    height_threshold : float
        Minimum peak height as fraction of max intensity
    distance : float
        Minimum distance between peaks in wavenumber units
        
    Returns:
    --------
    list
        List of peak positions
    """
    from scipy.signal import find_peaks
    
    # Normalize intensities
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Convert distance to index units
    avg_spacing = np.mean(np.diff(x))
    distance_indices = int(distance / avg_spacing)
    
    # Find peaks
    peaks, _ = find_peaks(y_norm, 
                         height=height_threshold,
                         distance=distance_indices)
    
    return x[peaks].tolist()


def baseline_correct_spectrum(x: np.ndarray, y: np.ndarray, 
                            method: str = "polynomial", **kwargs) -> np.ndarray:
    """
    Apply baseline correction to spectrum.
    
    Parameters:
    -----------
    x : array_like
        Wavenumber array
    y : array_like
        Intensity array
    method : str
        Baseline correction method ("polynomial", "linear", "als")
    **kwargs : dict
        Method-specific parameters
        
    Returns:
    --------
    np.ndarray
        Baseline-corrected intensities
    """
    if method == "polynomial":
        degree = kwargs.get('degree', 2)
        # Fit polynomial to endpoints and minimum points
        edge_fraction = kwargs.get('edge_fraction', 0.1)
        n_edge = int(len(x) * edge_fraction)
        
        # Use edge points for baseline estimation
        x_baseline = np.concatenate([x[:n_edge], x[-n_edge:]])
        y_baseline = np.concatenate([y[:n_edge], y[-n_edge:]])
        
        # Fit polynomial
        poly_coeffs = np.polyfit(x_baseline, y_baseline, degree)
        baseline = np.polyval(poly_coeffs, x)
        
        return y - baseline
    
    elif method == "linear":
        # Simple linear baseline
        baseline = np.linspace(y[0], y[-1], len(y))
        return y - baseline
    
    else:
        print(f"Unknown baseline correction method: {method}")
        return y 