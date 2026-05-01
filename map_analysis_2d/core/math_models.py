from typing import List

import numpy as np

from core.peak_fitting import PeakFitter


DEFAULT_CURVE_FIT_MAXFEV = 5000


def get_peak_function(shape: str):
    """Returns the PeakFitter function corresponding to the shape."""
    if shape == "Lorentzian":
        return PeakFitter.lorentzian
    elif shape == "Gaussian":
        return PeakFitter.gaussian
    elif shape == "Pseudo-Voigt":
        return PeakFitter.pseudo_voigt
    else:
        raise ValueError(f"Unsupported peak shape: {shape}")

def get_num_params_for_shape(shape: str) -> int:
    """Returns the number of parameters required for a given peak shape."""
    if shape in ["Lorentzian", "Gaussian"]:
        return 3  # amplitude, center, width
    elif shape == "Pseudo-Voigt":
        return 4  # amplitude, center, width, eta
    else:
        raise ValueError(f"Unsupported peak shape: {shape}")

def create_multi_peak_model(shapes: List[str]):
    """
    Creates a dynamic multi-peak function based on the list of shapes.
    
    Parameters:
    -----------
    shapes : list of str
        List of shapes for each peak (e.g., ["Lorentzian", "Gaussian"]).
        
    Returns:
    --------
    callable
        A function f(x, *params) that returns the sum of the peaks.
    """
    peak_funcs = [get_peak_function(shape) for shape in shapes]
    params_per_peak = [get_num_params_for_shape(shape) for shape in shapes]
    
    def multi_peak_model(x, *params):
        expected = sum(params_per_peak)
        if len(params) != expected:
            raise ValueError(f"Expected {expected} parameters for {len(peak_funcs)}-peak model, got {len(params)}")
        y = np.zeros_like(x, dtype=float)
        param_idx = 0
        for i, func in enumerate(peak_funcs):
            n_params = params_per_peak[i]
            peak_params = params[param_idx:param_idx+n_params]
            y += func(x, *peak_params)
            param_idx += n_params
        return y
        
    return multi_peak_model

def compute_integrated_intensity(amplitude: float, width: float, shape: str, eta: float = 0.5) -> float:
    """Analytical area under a fitted peak derived from its stored parameters."""
    amp = float(amplitude)
    wid = abs(float(width))
    if amp < 0:
        return np.nan
    if shape == "Gaussian":
        return amp * wid * np.sqrt(np.pi)
    elif shape == "Lorentzian":
        return amp * wid * np.pi
    elif shape == "Pseudo-Voigt":
        eta = float(np.clip(eta, 0.0, 1.0))
        return amp * wid * ((1.0 - eta) * np.sqrt(np.pi) + eta * np.pi)
    return np.nan


def get_param_names(shapes: List[str]) -> List[str]:
    """
    Returns a list of parameter names for a given sequence of peak shapes.
    
    Parameters:
    -----------
    shapes : list of str
        List of shapes for each peak (e.g., ["Lorentzian", "Gaussian"]).
        
    Returns:
    --------
    list of str
        A list of parameter names, e.g. ['P1_Amp', 'P1_Cen', 'P1_Wid', 'P2_Amp', ...]
    """
    param_names = []
    allowed_shapes = {"Lorentzian", "Gaussian", "Pseudo-Voigt"}
    for i, shape in enumerate(shapes, 1):
        if shape not in allowed_shapes:
            raise ValueError(f"Unsupported peak shape at index {i}: {shape}")
        param_names.extend([f"P{i}_Amp", f"P{i}_Cen", f"P{i}_Wid"])
        if shape == "Pseudo-Voigt":
            param_names.append(f"P{i}_Eta")
    return param_names
