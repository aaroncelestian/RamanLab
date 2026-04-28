import numpy as np
from core.peak_fitting import PeakFitter

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

def create_multi_peak_model(shapes: list[str]):
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
        y = np.zeros_like(x, dtype=float)
        param_idx = 0
        for i, func in enumerate(peak_funcs):
            n_params = params_per_peak[i]
            peak_params = params[param_idx:param_idx+n_params]
            y += func(x, *peak_params)
            param_idx += n_params
        return y
        
    return multi_peak_model

def get_param_names(shapes: list[str]) -> list[str]:
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
    for i, shape in enumerate(shapes, 1):
        param_names.extend([f"P{i}_Amp", f"P{i}_Cen", f"P{i}_Wid"])
        if shape == "Pseudo-Voigt":
            param_names.append(f"P{i}_Eta")
    return param_names
