import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import voigt_profile
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

@dataclass
class RamanMode:
    """Data structure for a Raman mode with Grüneisen parameters"""
    name: str
    omega0: float  # Unstrained frequency (cm^-1)
    gamma_components: np.ndarray  # Grüneisen tensor components [γ1, γ2, γ3, γ4, γ5, γ6]
    intensity: float = 1.0
    natural_width: float = 2.0  # Natural linewidth (cm^-1)

class StrainRamanAnalyzer:
    """
    Class for refining strain tensors from Raman peak positions, widths, and asymmetries
    Based on Angel et al. (2019) phonon-mode Grüneisen tensor approach
    """
    
    def __init__(self, crystal_system: str = 'trigonal'):
        """
        Initialize with crystal system for symmetry constraints
        
        Parameters:
        crystal_system: str, one of ['cubic', 'tetragonal', 'trigonal', 'hexagonal', 
                       'orthorhombic', 'monoclinic', 'triclinic']
        """
        self.crystal_system = crystal_system.lower()
        self.modes = {}
        self.symmetry_constraints = self._get_symmetry_constraints()
        
    def _get_symmetry_constraints(self) -> Dict:
        """Return symmetry constraints for Grüneisen tensor components"""
        constraints = {
            'cubic': {'independent': [0], 'zeros': [3, 4, 5], 'equal_pairs': [(0, 1, 2)]},
            'tetragonal': {'independent': [0, 2], 'zeros': [3, 4, 5], 'equal_pairs': [(0, 1)]},
            'trigonal': {'independent': [0, 2], 'zeros': [3, 4, 5], 'equal_pairs': [(0, 1)]},
            'hexagonal': {'independent': [0, 2], 'zeros': [3, 4, 5], 'equal_pairs': [(0, 1)]},
            'orthorhombic': {'independent': [0, 1, 2], 'zeros': [3, 4, 5], 'equal_pairs': []},
            'monoclinic': {'independent': [0, 1, 2, 4], 'zeros': [3, 5], 'equal_pairs': []},
            'triclinic': {'independent': [0, 1, 2, 3, 4, 5], 'zeros': [], 'equal_pairs': []}
        }
        return constraints.get(self.crystal_system, constraints['triclinic'])
    
    def add_mode(self, mode: RamanMode):
        """Add a Raman mode with its Grüneisen parameters"""
        # Apply symmetry constraints to Grüneisen components
        gamma_constrained = self._apply_symmetry_constraints(mode.gamma_components)
        mode.gamma_components = gamma_constrained
        self.modes[mode.name] = mode
    
    def _apply_symmetry_constraints(self, gamma: np.ndarray) -> np.ndarray:
        """Apply crystal symmetry constraints to Grüneisen tensor components"""
        gamma_constrained = gamma.copy()
        
        # Set zero components
        for idx in self.symmetry_constraints['zeros']:
            gamma_constrained[idx] = 0.0
            
        # Apply equality constraints
        for equal_group in self.symmetry_constraints['equal_pairs']:
            if len(equal_group) > 1:
                avg_value = np.mean([gamma_constrained[i] for i in equal_group])
                for idx in equal_group:
                    gamma_constrained[idx] = avg_value
                    
        return gamma_constrained
    
    def calculate_frequency_shift(self, strain_vector: np.ndarray, mode_name: str) -> float:
        """
        Calculate frequency shift for a mode given strain vector
        
        Parameters:
        strain_vector: np.ndarray, strain components [ε1, ε2, ε3, ε4, ε5, ε6] (Voigt notation)
        mode_name: str, name of the mode
        
        Returns:
        float: Δω/ω0 (fractional frequency change)
        """
        if mode_name not in self.modes:
            raise ValueError(f"Mode {mode_name} not found")
            
        mode = self.modes[mode_name]
        # Equation (5) from the paper: -Δω/ω0 = γ·ε
        fractional_shift = -np.dot(mode.gamma_components, strain_vector)
        return fractional_shift
    
    def calculate_all_frequency_shifts(self, strain_vector: np.ndarray) -> Dict[str, float]:
        """Calculate frequency shifts for all modes"""
        shifts = {}
        for mode_name in self.modes:
            shifts[mode_name] = self.calculate_frequency_shift(strain_vector, mode_name)
        return shifts
    
    def strain_distribution_effect(self, strain_vector: np.ndarray, strain_gradient: np.ndarray, 
                                 mode_name: str) -> Tuple[float, float]:
        """
        Calculate peak broadening and asymmetry due to strain gradients
        
        Parameters:
        strain_vector: mean strain
        strain_gradient: strain gradient tensor (simplified as vector for implementation)
        mode_name: mode name
        
        Returns:
        Tuple[float, float]: (additional_width, asymmetry_parameter)
        """
        mode = self.modes[mode_name]
        
        # Peak broadening due to strain distribution
        # Simplified model: width ∝ |γ·∇ε|
        strain_variation = np.abs(np.dot(mode.gamma_components, strain_gradient))
        additional_width = strain_variation * mode.omega0
        
        # Asymmetry from non-linear strain effects
        # Higher-order terms can cause asymmetric broadening
        asymmetry = np.sum(mode.gamma_components * strain_gradient**2) * 0.1
        
        return additional_width, asymmetry
    
    def model_peak_shape(self, frequencies: np.ndarray, center: float, width: float, 
                        asymmetry: float = 0.0, intensity: float = 1.0) -> np.ndarray:
        """
        Model asymmetric peak shape with strain-induced broadening
        
        Parameters:
        frequencies: frequency array
        center: peak center
        width: peak width (FWHM)
        asymmetry: asymmetry parameter
        intensity: peak intensity
        
        Returns:
        np.ndarray: peak profile
        """
        # Use pseudo-Voigt with asymmetry correction
        sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to σ
        
        if asymmetry == 0:
            # Symmetric Voigt profile
            return intensity * voigt_profile(frequencies - center, sigma, sigma)
        else:
            # Asymmetric modification
            x_norm = (frequencies - center) / sigma
            profile = voigt_profile(frequencies - center, sigma, sigma)
            
            # Apply asymmetry using exponential modification
            asymmetry_factor = np.exp(asymmetry * x_norm)
            asymmetric_profile = profile * asymmetry_factor
            
            # Normalize to preserve integrated intensity
            norm_factor = intensity / np.trapz(asymmetric_profile, frequencies)
            return asymmetric_profile * norm_factor
    
    def forward_model(self, strain_params: np.ndarray, frequencies: np.ndarray, 
                     include_gradients: bool = True) -> np.ndarray:
        """
        Forward model: calculate complete Raman spectrum from strain parameters
        
        Parameters:
        strain_params: strain + gradient parameters
        frequencies: frequency array for spectrum
        include_gradients: whether to include strain gradient effects
        
        Returns:
        np.ndarray: modeled spectrum
        """
        n_strain = len(self.symmetry_constraints['independent']) + 1  # +1 for hydrostatic
        strain_vector = strain_params[:6]  # Always 6 components, but constrained
        
        if include_gradients and len(strain_params) > 6:
            strain_gradient = strain_params[6:12]
        else:
            strain_gradient = np.zeros(6)
        
        # Apply symmetry constraints
        strain_vector = self._apply_symmetry_constraints(strain_vector)
        
        spectrum = np.zeros_like(frequencies)
        
        for mode_name, mode in self.modes.items():
            # Calculate frequency shift
            fractional_shift = self.calculate_frequency_shift(strain_vector, mode_name)
            new_center = mode.omega0 * (1 + fractional_shift)
            
            # Calculate broadening and asymmetry
            if include_gradients:
                add_width, asymmetry = self.strain_distribution_effect(
                    strain_vector, strain_gradient, mode_name)
                total_width = mode.natural_width + add_width
            else:
                total_width = mode.natural_width
                asymmetry = 0.0
            
            # Add peak to spectrum
            peak = self.model_peak_shape(frequencies, new_center, total_width, 
                                       asymmetry, mode.intensity)
            spectrum += peak
            
        return spectrum
    
    def fit_strain_tensor(self, observed_frequencies: np.ndarray, observed_spectrum: np.ndarray,
                         initial_strain: Optional[np.ndarray] = None,
                         fit_gradients: bool = True,
                         weights: Optional[np.ndarray] = None) -> Dict:
        """
        Fit strain tensor to observed Raman spectrum
        
        Parameters:
        observed_frequencies: frequency array
        observed_spectrum: observed intensities
        initial_strain: initial guess for strain parameters
        fit_gradients: whether to fit strain gradients
        weights: weights for fitting (e.g., 1/σ for each point)
        
        Returns:
        Dict: fitting results including strain tensor, gradients, and statistics
        """
        n_strain_params = 6
        n_gradient_params = 6 if fit_gradients else 0
        n_total_params = n_strain_params + n_gradient_params
        
        if initial_strain is None:
            initial_strain = np.zeros(n_total_params)
        elif len(initial_strain) == 6 and fit_gradients:
            # If only strain components provided but gradients requested, pad with zeros
            initial_strain = np.concatenate([initial_strain, np.zeros(6)])
        elif len(initial_strain) != n_total_params:
            # Resize to correct length
            initial_strain = np.resize(initial_strain, n_total_params)
        
        if weights is None:
            weights = np.ones_like(observed_spectrum)
        
        def objective(params):
            model_spectrum = self.forward_model(params, observed_frequencies, fit_gradients)
            residuals = (observed_spectrum - model_spectrum) * weights
            return residuals
        
        # Bounds: reasonable strain values typically < 0.1 (10%)
        lower_bounds = [-0.1] * n_total_params
        upper_bounds = [0.1] * n_total_params
        bounds = (lower_bounds, upper_bounds)
        
        # Perform fitting
        result = least_squares(objective, initial_strain, bounds=bounds, 
                             method='trf', verbose=1)
        
        # Extract results
        fitted_strain = result.x[:6]
        fitted_gradients = result.x[6:12] if fit_gradients else np.zeros(6)
        
        # Apply symmetry constraints
        fitted_strain = self._apply_symmetry_constraints(fitted_strain)
        
        # Calculate statistics
        model_spectrum = self.forward_model(result.x, observed_frequencies, fit_gradients)
        residuals = observed_spectrum - model_spectrum
        chi_squared = np.sum((residuals * weights)**2)
        r_squared = 1 - np.sum(residuals**2) / np.sum((observed_spectrum - np.mean(observed_spectrum))**2)
        
        # Calculate uncertainties (simplified)
        try:
            # Approximate covariance from Jacobian
            jacobian = result.jac
            cov_matrix = np.linalg.inv(jacobian.T @ jacobian)
            param_errors = np.sqrt(np.diag(cov_matrix))
            strain_errors = param_errors[:6]
            gradient_errors = param_errors[6:12] if fit_gradients else np.zeros(6)
        except:
            strain_errors = np.full(6, np.nan)
            gradient_errors = np.full(6, np.nan)
        
        return {
            'strain_tensor': fitted_strain,
            'strain_gradients': fitted_gradients,
            'strain_errors': strain_errors,
            'gradient_errors': gradient_errors,
            'model_spectrum': model_spectrum,
            'residuals': residuals,
            'chi_squared': chi_squared,
            'r_squared': r_squared,
            'success': result.success,
            'message': result.message,
            'optimization_result': result
        }
    
    def analyze_peak_positions(self, observed_peaks: Dict[str, float]) -> Dict:
        """
        Simple strain analysis from peak positions only (following Angel et al.)
        
        Parameters:
        observed_peaks: Dict[mode_name, observed_frequency]
        
        Returns:
        Dict: strain analysis results
        """
        # Calculate frequency shifts
        shifts = {}
        for mode_name, obs_freq in observed_peaks.items():
            if mode_name in self.modes:
                omega0 = self.modes[mode_name].omega0
                shifts[mode_name] = (obs_freq - omega0) / omega0
        
        # Set up linear system: Δω/ω = -γ·ε
        mode_names = list(shifts.keys())
        n_modes = len(mode_names)
        n_independent_strains = len(self.symmetry_constraints['independent'])
        
        # Build design matrix
        G_matrix = np.zeros((n_modes, 6))  # Grüneisen matrix
        shift_vector = np.zeros(n_modes)
        
        for i, mode_name in enumerate(mode_names):
            G_matrix[i, :] = self.modes[mode_name].gamma_components
            shift_vector[i] = -shifts[mode_name]  # Note the negative sign
        
        # Apply symmetry constraints to reduce system
        # This is simplified - full implementation would properly handle constraints
        
        if n_modes >= n_independent_strains:
            # Overdetermined or exactly determined system
            strain_solution, residuals, rank, s = np.linalg.lstsq(G_matrix, shift_vector, rcond=None)
            
            # Apply symmetry constraints
            strain_solution = self._apply_symmetry_constraints(strain_solution)
            
            return {
                'strain_tensor': strain_solution,
                'residuals': residuals,
                'rank': rank,
                'mode_shifts': shifts,
                'success': rank >= n_independent_strains
            }
        else:
            return {
                'strain_tensor': np.full(6, np.nan),
                'success': False,
                'message': f'Insufficient modes: need {n_independent_strains}, got {n_modes}'
            }


def example_usage():
    """Example of how to use the StrainRamanAnalyzer"""
    
    # Create analyzer for quartz (trigonal)
    analyzer = StrainRamanAnalyzer('trigonal')
    
    # Add some quartz modes (from Table 3 in the paper)
    quartz_modes = [
        RamanMode('464', 464.8, np.array([0.60, 0.60, 1.19, 0, 0, 0]), intensity=100),
        RamanMode('207', 207.3, np.array([3.64, 3.64, 5.25, 0, 0, 0]), intensity=50),
        RamanMode('128', 128.1, np.array([1.21, 1.21, 2.69, 0, 0, 0]), intensity=30),
    ]
    
    for mode in quartz_modes:
        analyzer.add_mode(mode)
    
    # Example 1: Analyze peak positions
    observed_peaks = {
        '464': 469.2,  # 464 cm⁻¹ mode shifted to 469.2
        '207': 215.1,  # 207 cm⁻¹ mode shifted to 215.1
        '128': 134.5   # 128 cm⁻¹ mode shifted to 134.5
    }
    
    peak_analysis = analyzer.analyze_peak_positions(observed_peaks)
    print("Peak position analysis:")
    print(f"Strain tensor: {peak_analysis['strain_tensor']}")
    print(f"Success: {peak_analysis['success']}")
    
    # Example 2: Full spectrum fitting
    frequencies = np.linspace(100, 500, 1000)
    
    # Simulate a strained spectrum
    true_strain = np.array([0.02, 0.02, -0.01, 0, 0, 0])  # 2% compression in a,b; 1% extension in c
    true_gradients = np.array([0.001, 0.001, 0.0005, 0, 0, 0])  # Small gradients
    
    true_spectrum = analyzer.forward_model(
        np.concatenate([true_strain, true_gradients]), 
        frequencies, 
        include_gradients=True
    )
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(true_spectrum))
    observed_spectrum = true_spectrum + noise
    
    # Fit the spectrum
    fit_result = analyzer.fit_strain_tensor(
        frequencies, observed_spectrum, 
        fit_gradients=True
    )
    
    print("\nFull spectrum fitting:")
    print(f"True strain: {true_strain}")
    print(f"Fitted strain: {fit_result['strain_tensor']}")
    print(f"R²: {fit_result['r_squared']:.4f}")
    print(f"Success: {fit_result['success']}")
    
    return analyzer, fit_result

if __name__ == "__main__":
    analyzer, results = example_usage()