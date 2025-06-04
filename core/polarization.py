"""
Polarization Analysis Module for Raman Spectroscopy

This module provides comprehensive polarization analysis capabilities including:
- Depolarization ratio calculations
- Polarized spectrum generation from mineral databases
- Raman tensor analysis
- Symmetry classification based on polarization behavior

Author: RamanLab Development Team
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.signal import savgol_filter, find_peaks
from dataclasses import dataclass


@dataclass
class PolarizationData:
    """Container for polarized spectrum data."""
    wavenumbers: np.ndarray
    intensities: np.ndarray
    config: str  # e.g., 'xx', 'xy', 'yy', 'zz'
    filename: Optional[str] = None


@dataclass
class DepolarizationResult:
    """Results from depolarization ratio analysis."""
    wavenumbers: np.ndarray
    ratio: np.ndarray
    I_parallel: np.ndarray  # I_xx (parallel)
    I_perpendicular: np.ndarray  # I_xy (perpendicular)
    I_parallel_raw: np.ndarray
    I_perpendicular_raw: np.ndarray
    valid_mask: np.ndarray


@dataclass
class TensorAnalysisResult:
    """Results from Raman tensor analysis."""
    wavenumbers: np.ndarray
    tensor_elements: Dict[str, np.ndarray]
    configurations: List[str]


class PolarizationAnalyzer:
    """
    Comprehensive polarization analysis for Raman spectroscopy.
    
    This class provides methods for analyzing polarization-dependent Raman spectra,
    calculating depolarization ratios, generating synthetic polarized spectra,
    and performing symmetry analysis.
    """
    
    def __init__(self):
        """Initialize the polarization analyzer."""
        self.polarization_data: Dict[str, PolarizationData] = {}
        self.depolarization_results: Optional[DepolarizationResult] = None
        self.tensor_results: Optional[TensorAnalysisResult] = None
        
    def add_polarized_spectrum(self, config: str, wavenumbers: np.ndarray, 
                             intensities: np.ndarray, filename: Optional[str] = None) -> None:
        """
        Add a polarized spectrum to the analysis.
        
        Args:
            config: Polarization configuration (e.g., 'xx', 'xy', 'yy', 'zz')
            wavenumbers: Wavenumber array
            intensities: Intensity array
            filename: Optional filename for reference
        """
        self.polarization_data[config] = PolarizationData(
            wavenumbers=np.array(wavenumbers),
            intensities=np.array(intensities),
            config=config,
            filename=filename
        )
    
    def calculate_depolarization_ratios(self, smooth_data: bool = True, 
                                      resolution: int = 2000) -> DepolarizationResult:
        """
        Calculate depolarization ratios from polarized spectra.
        
        The depolarization ratio ρ = I_⊥ / I_∥ provides information about
        the symmetry of vibrational modes.
        
        Args:
            smooth_data: Whether to apply smoothing to reduce noise
            resolution: Number of points for interpolated common grid
            
        Returns:
            DepolarizationResult object containing analysis results
            
        Raises:
            ValueError: If required polarization configurations are missing
        """
        # Check for required configurations
        required_configs = ['xx', 'xy']  # Minimum for depolarization ratio
        available_configs = list(self.polarization_data.keys())
        
        if not all(config in available_configs for config in required_configs):
            raise ValueError(
                f"Need at least XX and XY configurations for depolarization ratio calculation.\n"
                f"Available: {', '.join([c.upper() for c in available_configs])}"
            )
        
        # Get data
        xx_data = self.polarization_data['xx']
        xy_data = self.polarization_data['xy']
        
        # Get common wavenumber range
        wavenumbers_xx = xx_data.wavenumbers
        wavenumbers_xy = xy_data.wavenumbers
        
        # Create common grid with higher resolution
        common_wavenumbers = np.linspace(
            max(np.min(wavenumbers_xx), np.min(wavenumbers_xy)),
            min(np.max(wavenumbers_xx), np.max(wavenumbers_xy)),
            resolution
        )
        
        # Interpolate intensities to common grid
        I_xx = np.interp(common_wavenumbers, wavenumbers_xx, xx_data.intensities)
        I_xy = np.interp(common_wavenumbers, wavenumbers_xy, xy_data.intensities)
        
        # Store raw data
        I_xx_raw = I_xx.copy()
        I_xy_raw = I_xy.copy()
        
        # Apply smoothing if requested
        if smooth_data:
            window_length = min(51, len(I_xx) // 10)  # Adaptive window length
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(5, window_length)  # Minimum window size
            
            if len(I_xx) > window_length:
                I_xx = savgol_filter(I_xx, window_length, 3)
                I_xy = savgol_filter(I_xy, window_length, 3)
        
        # Calculate depolarization ratio ρ = I_xy / I_xx
        # Use sophisticated approach to handle division by zero
        threshold = np.max(I_xx) * 0.01  # 1% of maximum intensity
        
        # Create mask for valid data points
        valid_mask = I_xx > threshold
        
        # Initialize ratio array
        depol_ratio = np.zeros_like(I_xx)
        
        # Calculate ratio only for valid points
        depol_ratio[valid_mask] = I_xy[valid_mask] / I_xx[valid_mask]
        
        # For invalid points, use interpolation from nearby valid points
        if np.any(~valid_mask):
            valid_indices = np.where(valid_mask)[0]
            invalid_indices = np.where(~valid_mask)[0]
            
            if len(valid_indices) > 1:
                depol_ratio[invalid_indices] = np.interp(
                    invalid_indices, valid_indices, depol_ratio[valid_indices]
                )
        
        # Clip ratio to reasonable range (0 to 2, theoretical max is 0.75 for Raman)
        depol_ratio = np.clip(depol_ratio, 0, 2)
        
        # Apply additional smoothing to the ratio if requested
        if smooth_data and len(depol_ratio) > window_length:
            depol_ratio = savgol_filter(depol_ratio, window_length, 2)
        
        # Create and store results
        self.depolarization_results = DepolarizationResult(
            wavenumbers=common_wavenumbers,
            ratio=depol_ratio,
            I_parallel=I_xx,
            I_perpendicular=I_xy,
            I_parallel_raw=I_xx_raw,
            I_perpendicular_raw=I_xy_raw,
            valid_mask=valid_mask
        )
        
        return self.depolarization_results
    
    def determine_raman_tensors(self) -> TensorAnalysisResult:
        """
        Determine Raman tensor elements from polarized measurements.
        
        This is a simplified implementation that requires multiple polarization
        configurations for full tensor determination.
        
        Returns:
            TensorAnalysisResult object containing tensor analysis
            
        Raises:
            ValueError: If insufficient polarization data is available
        """
        if not self.polarization_data:
            raise ValueError("No polarization data available for tensor analysis")
        
        available_configs = list(self.polarization_data.keys())
        
        # Check for minimum required configurations
        if len(available_configs) < 2:
            raise ValueError("Need at least 2 polarization configurations for tensor analysis")
        
        # Get reference wavenumbers from first configuration
        ref_config = available_configs[0]
        ref_data = self.polarization_data[ref_config]
        wavenumbers = ref_data.wavenumbers
        
        # Initialize tensor elements storage
        tensor_elements = {}
        
        # For each configuration, store the intensity as a tensor element
        for config in available_configs:
            data = self.polarization_data[config]
            # Interpolate to common wavenumber grid
            intensities = np.interp(wavenumbers, data.wavenumbers, data.intensities)
            tensor_elements[config] = intensities
        
        # Calculate derived quantities
        if 'xx' in available_configs and 'yy' in available_configs:
            # Isotropic component (for cubic crystals)
            I_iso = (tensor_elements['xx'] + tensor_elements['yy']) / 2
            tensor_elements['isotropic'] = I_iso
        
        if 'xx' in available_configs and 'xy' in available_configs:
            # Anisotropic component
            I_aniso = tensor_elements['xx'] - tensor_elements['xy']
            tensor_elements['anisotropic'] = I_aniso
        
        # Create and store results
        self.tensor_results = TensorAnalysisResult(
            wavenumbers=wavenumbers,
            tensor_elements=tensor_elements,
            configurations=available_configs
        )
        
        return self.tensor_results
    
    def classify_symmetries(self, crystal_system: str = "Unknown") -> Dict[str, Any]:
        """
        Classify vibrational modes based on polarization behavior.
        
        Args:
            crystal_system: Crystal system for theoretical predictions
            
        Returns:
            Dictionary containing symmetry classification results
            
        Raises:
            ValueError: If depolarization ratios haven't been calculated
        """
        if self.depolarization_results is None:
            raise ValueError("Calculate depolarization ratios first")
        
        results = {
            'crystal_system': crystal_system,
            'peaks': [],
            'theoretical_background': self._get_theoretical_background(),
            'expected_symmetries': self.get_expected_symmetries(crystal_system)
        }
        
        # Find peaks in the parallel spectrum for analysis
        I_parallel = self.depolarization_results.I_parallel
        wavenumbers = self.depolarization_results.wavenumbers
        ratios = self.depolarization_results.ratio
        
        peaks, _ = find_peaks(I_parallel, prominence=np.max(I_parallel) * 0.1, distance=20)
        
        for i, peak_idx in enumerate(peaks):
            wavenumber = wavenumbers[peak_idx]
            ratio = ratios[peak_idx]
            
            # Classify based on depolarization ratio
            classification = self._classify_peak_symmetry(ratio)
            
            peak_info = {
                'peak_number': i + 1,
                'wavenumber': wavenumber,
                'depolarization_ratio': ratio,
                'symmetry': classification['symmetry'],
                'polarization_type': classification['polarization_type']
            }
            results['peaks'].append(peak_info)
        
        return results
    
    def _classify_peak_symmetry(self, ratio: float) -> Dict[str, str]:
        """Classify peak symmetry based on depolarization ratio."""
        if ratio < 0.1:
            return {
                'symmetry': "Totally symmetric (A1, A1g)",
                'polarization_type': "Polarized"
            }
        elif 0.1 <= ratio < 0.5:
            return {
                'symmetry': "Partially polarized",
                'polarization_type': "Partially polarized"
            }
        elif 0.5 <= ratio < 0.75:
            return {
                'symmetry': "Depolarized (E, T2, etc.)",
                'polarization_type': "Depolarized"
            }
        else:
            return {
                'symmetry': "Highly depolarized",
                'polarization_type': "Highly depolarized"
            }
    
    def _get_theoretical_background(self) -> Dict[str, str]:
        """Get theoretical background information."""
        return {
            'definition': "Depolarization ratio ρ = I_⊥ / I_∥",
            'A1_totally_symmetric': "ρ ≈ 0 (polarized)",
            'E_doubly_degenerate': "ρ = 3/4 (depolarized)",
            'T2_triply_degenerate': "ρ = 3/4 (depolarized)"
        }
    
    def get_expected_symmetries(self, crystal_system: str) -> List[str]:
        """Get expected symmetries for different crystal systems."""
        symmetries = {
            "Cubic": ["A1g", "Eg", "T1g", "T2g", "A1u", "Eu", "T1u", "T2u"],
            "Tetragonal": ["A1g", "A2g", "B1g", "B2g", "Eg", "A1u", "A2u", "B1u", "B2u", "Eu"],
            "Orthorhombic": ["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
            "Hexagonal": ["A1g", "A2g", "B1g", "B2g", "E1g", "E2g", "A1u", "A2u", "B1u", "B2u", "E1u", "E2u"],
            "Trigonal": ["A1g", "A2g", "Eg", "A1u", "A2u", "Eu"],
            "Monoclinic": ["Ag", "Bg", "Au", "Bu"],
            "Triclinic": ["Ag", "Au"]
        }
        return symmetries.get(crystal_system, ["Unknown"])
    
    def clear_data(self) -> None:
        """Clear all polarization data and results."""
        self.polarization_data.clear()
        self.depolarization_results = None
        self.tensor_results = None


class PolarizedSpectrumGenerator:
    """
    Generator for synthetic polarized Raman spectra from mineral database data.
    
    This class provides methods to generate realistic polarized spectra
    based on mineral composition, crystal structure, and vibrational modes.
    """
    
    def __init__(self):
        """Initialize the spectrum generator."""
        pass
    
    def generate_polarized_spectrum(self, mineral_data: Dict[str, Any], 
                                  polarization_config: str,
                                  wavenumber_range: Tuple[float, float] = (100, 4000),
                                  num_points: int = 2000,
                                  peak_width: float = 5.0,
                                  intensity_scale: float = 1.0,
                                  noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a polarized Raman spectrum from mineral database data.
        
        Args:
            mineral_data: Dictionary containing mineral information
            polarization_config: Polarization configuration ('xx', 'xy', etc.)
            wavenumber_range: Tuple of (min, max) wavenumbers
            num_points: Number of points in the spectrum
            peak_width: Peak width parameter (FWHM)
            intensity_scale: Overall intensity scaling factor
            noise_level: Noise level as fraction of signal
            
        Returns:
            Tuple of (wavenumbers, intensities)
        """
        # Create wavenumber range
        wavenumbers = np.linspace(wavenumber_range[0], wavenumber_range[1], num_points)
        intensities = np.zeros_like(wavenumbers)
        
        # Check for modes data
        if 'modes' not in mineral_data or not mineral_data['modes']:
            return wavenumbers, intensities
        
        modes = mineral_data['modes']
        
        # Get polarization scaling factor
        polarization_factor = self.get_polarization_factor(mineral_data, polarization_config)
        
        # Process each mode
        for mode in modes:
            if isinstance(mode, (tuple, list)) and len(mode) >= 3:
                frequency = float(mode[0])  # Frequency in cm^-1
                symmetry = str(mode[1])     # Symmetry character
                base_intensity = float(mode[2])  # Relative intensity
                
                # Skip modes with zero or very low frequency
                if frequency < 50:
                    continue
                
                # Apply polarization-dependent intensity scaling
                polarized_intensity = base_intensity * polarization_factor * intensity_scale
                
                # Apply symmetry-dependent polarization rules
                polarized_intensity *= self.get_symmetry_polarization_factor(
                    symmetry, polarization_config
                )
                
                # Generate Lorentzian peak
                peak = polarized_intensity * (peak_width**2) / (
                    (wavenumbers - frequency)**2 + peak_width**2
                )
                intensities += peak
        
        # Normalize intensities
        if np.max(intensities) > 0:
            intensities = intensities / np.max(intensities)
        
        # Add realistic noise
        if noise_level > 0:
            intensities += noise_level * np.random.randn(len(intensities))
            intensities = np.maximum(intensities, 0)  # Ensure non-negative
        
        return wavenumbers, intensities
    
    def get_polarization_factor(self, mineral_data: Dict[str, Any], 
                              polarization_config: str) -> float:
        """
        Get polarization factor from dielectric tensor data.
        
        Args:
            mineral_data: Mineral database entry
            polarization_config: Polarization configuration
            
        Returns:
            Polarization scaling factor
        """
        # Check for different tensor data formats
        tensor_data = None
        
        for tensor_key in ['dielectric_tensor', 'n_tensor', 'dielectric_tensors']:
            if tensor_key in mineral_data:
                tensor_data = mineral_data[tensor_key]
                break
        
        if tensor_data is None:
            return 1.0  # Default factor if no tensor data
        
        # Extract tensor elements based on polarization configuration
        component_map = {
            'xx': 'xx', 'yy': 'yy', 'zz': 'zz',
            'xy': 'xy', 'yx': 'xy',
            'xz': 'xz', 'zx': 'xz', 
            'yz': 'yz', 'zy': 'yz'
        }
        
        component = component_map.get(polarization_config, 'xx')
        default_values = {
            'xx': 1.0, 'yy': 1.0, 'zz': 1.0,
            'xy': 0.5, 'xz': 0.5, 'yz': 0.5
        }
        
        return self.extract_tensor_element(
            tensor_data, component, default_values.get(component, 1.0)
        )
    
    def extract_tensor_element(self, tensor_data: Any, component: str, 
                             default: float = 1.0) -> float:
        """
        Extract a specific tensor element from various tensor data formats.
        
        Args:
            tensor_data: Tensor data in various formats
            component: Tensor component ('xx', 'xy', etc.)
            default: Default value if extraction fails
            
        Returns:
            Extracted tensor element value
        """
        try:
            # Handle numpy array format
            if isinstance(tensor_data, np.ndarray):
                if tensor_data.shape == (3, 3):
                    component_map = {
                        'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2),
                        'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)
                    }
                    if component in component_map:
                        i, j = component_map[component]
                        return abs(tensor_data[i, j])
            
            # Handle dictionary format
            elif isinstance(tensor_data, dict):
                if 'tensor' in tensor_data:
                    tensor_list = tensor_data['tensor']
                    for element in tensor_list:
                        if isinstance(element, list) and len(element) >= 3:
                            if element[1] == component:
                                return abs(float(element[2]))
            
            # Handle list format
            elif isinstance(tensor_data, list):
                for element in tensor_data:
                    if isinstance(element, dict):
                        if element.get('Component') == component:
                            return abs(float(element.get('X', default)))
                    elif isinstance(element, list) and len(element) >= 3:
                        if element[1] == component:
                            return abs(float(element[2]))
            
            return default
            
        except Exception:
            return default
    
    def get_symmetry_polarization_factor(self, symmetry: str, 
                                       polarization_config: str) -> float:
        """
        Get polarization factor based on vibrational mode symmetry.
        
        Args:
            symmetry: Vibrational mode symmetry label
            polarization_config: Polarization configuration
            
        Returns:
            Symmetry-dependent polarization factor
        """
        symmetry_lower = symmetry.lower()
        
        # Totally symmetric modes (A1, A1g) are strongly polarized
        if any(sym in symmetry_lower for sym in ['a1', 'ag']):
            if polarization_config in ['xx', 'yy', 'zz']:
                return 1.0  # Strong in parallel configurations
            else:
                return 0.1  # Weak in cross-polarized configurations
        
        # Doubly degenerate modes (E, Eg) show intermediate polarization
        elif any(sym in symmetry_lower for sym in ['e', 'eg']):
            if polarization_config in ['xx', 'yy', 'zz']:
                return 0.7
            else:
                return 0.5
        
        # Triply degenerate modes (T, T2g) are typically depolarized
        elif any(sym in symmetry_lower for sym in ['t', 't2', 't2g']):
            if polarization_config in ['xx', 'yy', 'zz']:
                return 0.5
            else:
                return 0.8
        
        # Default case
        return 0.6


def get_polarization_factor_simple(config: str) -> float:
    """
    Get a simple polarization factor for basic calculations.
    
    Args:
        config: Polarization configuration
        
    Returns:
        Simple polarization factor
    """
    factors = {
        'xx': 1.0, 'yy': 1.0, 'zz': 1.0,  # Parallel configurations
        'xy': 0.5, 'yx': 0.5,             # Cross-polarized
        'xz': 0.5, 'zx': 0.5,
        'yz': 0.5, 'zy': 0.5
    }
    return factors.get(config.lower(), 1.0) 