"""
Enhanced Strain Tensor Refinement for Chemical Strain Effects
============================================================

Extension of the basic strain refinement to handle:
1. Composition-dependent Grüneisen parameters
2. Jahn-Teller distortion effects  
3. Chemical disorder broadening
4. Multi-phase strain analysis

For applications like Li/H exchange in battery materials.
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import voigt_profile
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import warnings

@dataclass
class ChemicalRamanMode:
    """Extended Raman mode with composition-dependent properties"""
    name: str
    omega0_pure: float  # Frequency in pure end-member
    gamma_components_pure: np.ndarray  # Grüneisen tensor for pure phase
    intensity_pure: float = 1.0
    natural_width: float = 2.0
    
    # Chemical strain parameters
    composition_sensitivity: np.ndarray = None  # dγ/dx for composition x
    jahn_teller_coupling: float = 0.0  # Coupling to JT distortion
    defect_strain_radius: float = 5.0  # Angstroms - range of defect strain field

class ChemicalStrainAnalyzer:
    """
    Enhanced strain analyzer for chemical strain effects
    
    Handles:
    - Composition-dependent Grüneisen parameters
    - Jahn-Teller distortion coupling
    - Chemical disorder broadening
    - Multi-phase coexistence
    """
    
    def __init__(self, crystal_system: str = 'trigonal', 
                 composition_model: str = 'linear'):
        """
        Initialize with chemical strain capabilities
        
        Parameters:
        crystal_system: Base crystal symmetry
        composition_model: How properties vary with composition
                          'linear', 'vegard', 'non_linear'
        """
        self.crystal_system = crystal_system.lower()
        self.composition_model = composition_model
        self.modes = {}
        self.composition = 0.0  # Current composition (e.g., Li fraction)
        self.jahn_teller_parameter = 0.0  # JT distortion magnitude
        self.chemical_disorder = 0.0  # Degree of chemical disorder
        
        # Phase coexistence parameters
        self.phases = {}  # Multiple phases with different compositions
        self.phase_fractions = {}
        
    def set_composition(self, composition: float):
        """Set current composition (0 = pure A, 1 = pure B)"""
        self.composition = np.clip(composition, 0.0, 1.0)
        self._update_mode_parameters()
    
    def set_jahn_teller_distortion(self, jt_parameter: float):
        """Set Jahn-Teller distortion parameter"""
        self.jahn_teller_parameter = jt_parameter
        self._update_mode_parameters()
    
    def _update_mode_parameters(self):
        """Update mode parameters based on current composition and JT state"""
        for mode_name, mode in self.modes.items():
            if isinstance(mode, ChemicalRamanMode):
                # Update frequency with composition
                mode.omega0 = self._composition_dependent_frequency(mode)
                
                # Update Grüneisen parameters
                mode.gamma_components = self._composition_dependent_gruneisen(mode)
                
                # Update intensity (some modes may appear/disappear)
                mode.intensity = self._composition_dependent_intensity(mode)
    
    def _composition_dependent_frequency(self, mode: ChemicalRamanMode) -> float:
        """Calculate frequency as function of composition"""
        if self.composition_model == 'linear':
            # Simple linear interpolation
            return mode.omega0_pure * (1 + 0.1 * self.composition)  # Example: 10% shift
        
        elif self.composition_model == 'vegard':
            # Vegard's law with bowing parameter
            bowing = 0.05  # Bowing parameter
            return mode.omega0_pure * (1 + 0.1 * self.composition - 
                                     bowing * self.composition * (1 - self.composition))
        
        elif self.composition_model == 'non_linear':
            # Non-linear effects (e.g., from electronic structure changes)
            return mode.omega0_pure * (1 + 0.1 * self.composition + 
                                     0.02 * self.composition**2)
        
        return mode.omega0_pure
    
    def _composition_dependent_gruneisen(self, mode: ChemicalRamanMode) -> np.ndarray:
        """Calculate Grüneisen parameters as function of composition"""
        base_gamma = mode.gamma_components_pure.copy()
        
        if mode.composition_sensitivity is not None:
            # Linear composition dependence
            base_gamma += mode.composition_sensitivity * self.composition
        
        # Jahn-Teller coupling effects
        if mode.jahn_teller_coupling != 0:
            # JT distortion affects certain strain components more
            jt_modification = np.zeros(6)
            jt_modification[0] = mode.jahn_teller_coupling * self.jahn_teller_parameter  # ε₁
            jt_modification[1] = -mode.jahn_teller_coupling * self.jahn_teller_parameter  # ε₂
            base_gamma += jt_modification
        
        return base_gamma
    
    def _composition_dependent_intensity(self, mode: ChemicalRamanMode) -> float:
        """Calculate intensity as function of composition"""
        # Some modes may become forbidden or enhanced with composition
        base_intensity = mode.intensity_pure
        
        # Example: mode intensity varies with Li content
        composition_factor = 1.0 + 0.5 * self.composition
        
        return base_intensity * composition_factor
    
    def chemical_disorder_broadening(self, mode: ChemicalRamanMode) -> float:
        """Calculate additional broadening from chemical disorder"""
        if self.chemical_disorder == 0:
            return 0.0
        
        # Broadening proportional to composition variance and mode sensitivity
        composition_variance = self.composition * (1 - self.composition)  # Maximum at x=0.5
        
        # Mode-specific sensitivity to local environment
        sensitivity = np.linalg.norm(mode.composition_sensitivity) if mode.composition_sensitivity is not None else 1.0
        
        disorder_width = self.chemical_disorder * composition_variance * sensitivity * mode.omega0_pure
        
        return disorder_width
    
    def defect_strain_distribution(self, mode: ChemicalRamanMode, 
                                 defect_concentration: float) -> Tuple[float, np.ndarray]:
        """
        Calculate strain distribution from point defects
        
        Returns:
        Tuple[float, np.ndarray]: (additional_width, strain_distribution)
        """
        if defect_concentration == 0:
            return 0.0, np.array([])
        
        # Simplified model: defects create local strain fields
        # Real implementation would use elasticity theory
        
        # Average defect separation
        defect_separation = (1.0 / defect_concentration)**(1/3)  # Assuming 3D
        
        # Strain field strength (depends on size mismatch)
        if 'Li' in mode.name.upper():
            size_mismatch = 0.3  # Li+ vs H+ size difference (example)
        else:
            size_mismatch = 0.1  # Default
        
        # Local strain magnitude
        local_strain_magnitude = size_mismatch / (defect_separation / mode.defect_strain_radius)
        
        # Broadening from strain distribution
        strain_broadening = local_strain_magnitude * np.linalg.norm(mode.gamma_components_pure) * mode.omega0_pure
        
        # Generate strain distribution (for advanced modeling)
        n_points = 100
        strain_values = np.linspace(-local_strain_magnitude, local_strain_magnitude, n_points)
        strain_distribution = norm.pdf(strain_values, 0, local_strain_magnitude/3)
        
        return strain_broadening, strain_distribution
    
    def multi_phase_spectrum(self, frequencies: np.ndarray, 
                           strain_params: np.ndarray) -> np.ndarray:
        """
        Calculate spectrum with multiple coexisting phases
        
        For phase separation in Li/H exchange systems
        """
        total_spectrum = np.zeros_like(frequencies)
        
        if not self.phases:
            # Single phase - use standard calculation
            return self.forward_model(strain_params, frequencies)
        
        # Multi-phase calculation
        for phase_name, phase_data in self.phases.items():
            phase_composition = phase_data['composition']
            phase_fraction = self.phase_fractions.get(phase_name, 1.0)
            
            # Temporarily set composition for this phase
            old_composition = self.composition
            self.set_composition(phase_composition)
            
            # Calculate spectrum for this phase
            phase_spectrum = self.forward_model(strain_params, frequencies)
            total_spectrum += phase_fraction * phase_spectrum
            
            # Restore original composition
            self.set_composition(old_composition)
        
        return total_spectrum
    
    def fit_chemical_strain_tensor(self, observed_frequencies: np.ndarray, 
                                 observed_spectrum: np.ndarray,
                                 composition: float = None,
                                 fit_composition: bool = False,
                                 fit_jahn_teller: bool = False,
                                 defect_concentration: float = 0.0) -> Dict:
        """
        Enhanced fitting including chemical effects
        
        Parameters:
        composition: Known composition (if None, will be fitted)
        fit_composition: Whether to fit composition as parameter
        fit_jahn_teller: Whether to fit JT distortion
        defect_concentration: Concentration of point defects
        """
        
        # Set up parameters to fit
        n_strain_params = 6
        n_extra_params = 0
        
        if fit_composition:
            n_extra_params += 1
        if fit_jahn_teller:
            n_extra_params += 1
        
        n_total_params = n_strain_params + n_extra_params
        
        # Initial parameters
        initial_params = np.zeros(n_total_params)
        if composition is not None:
            self.set_composition(composition)
        
        # Parameter bounds
        lower_bounds = [-0.1] * n_strain_params  # Strain bounds
        upper_bounds = [0.1] * n_strain_params
        
        if fit_composition:
            lower_bounds.append(0.0)  # Composition bounds
            upper_bounds.append(1.0)
            initial_params[n_strain_params] = self.composition
        
        if fit_jahn_teller:
            lower_bounds.append(-0.1)  # JT parameter bounds
            upper_bounds.append(0.1)
            initial_params[-1] = self.jahn_teller_parameter
        
        bounds = (lower_bounds, upper_bounds)
        
        def objective(params):
            # Extract parameters
            strain_vector = params[:n_strain_params]
            param_idx = n_strain_params
            
            if fit_composition:
                self.set_composition(params[param_idx])
                param_idx += 1
            
            if fit_jahn_teller:
                self.set_jahn_teller_distortion(params[param_idx])
            
            # Calculate model spectrum
            model_spectrum = self.multi_phase_spectrum(observed_frequencies, strain_vector)
            
            # Add chemical disorder broadening
            if self.chemical_disorder > 0:
                # Convolve with disorder broadening (simplified)
                model_spectrum = self._apply_disorder_broadening(model_spectrum, observed_frequencies)
            
            residuals = observed_spectrum - model_spectrum
            return residuals
        
        # Perform fitting
        result = least_squares(objective, initial_params, bounds=bounds, method='trf')
        
        # Extract results
        fitted_strain = result.x[:n_strain_params]
        fitted_composition = result.x[n_strain_params] if fit_composition else self.composition
        fitted_jt = result.x[-1] if fit_jahn_teller else self.jahn_teller_parameter
        
        # Calculate final model
        final_model = self.multi_phase_spectrum(observed_frequencies, fitted_strain)
        
        return {
            'strain_tensor': fitted_strain,
            'composition': fitted_composition,
            'jahn_teller_parameter': fitted_jt,
            'model_spectrum': final_model,
            'success': result.success,
            'r_squared': self._calculate_r_squared(observed_spectrum, final_model),
            'chemical_effects': {
                'disorder_broadening': self.chemical_disorder,
                'defect_concentration': defect_concentration,
                'phase_fractions': self.phase_fractions.copy()
            }
        }
    
    def _apply_disorder_broadening(self, spectrum: np.ndarray, 
                                 frequencies: np.ndarray) -> np.ndarray:
        """Apply chemical disorder broadening to spectrum"""
        # Simplified convolution with Gaussian broadening
        disorder_width = self.chemical_disorder * 10  # cm⁻¹
        if disorder_width == 0:
            return spectrum
        
        # Create Gaussian kernel
        freq_step = frequencies[1] - frequencies[0]
        kernel_width = int(3 * disorder_width / freq_step)
        kernel_freqs = np.arange(-kernel_width, kernel_width + 1) * freq_step
        kernel = np.exp(-kernel_freqs**2 / (2 * disorder_width**2))
        kernel /= np.sum(kernel)
        
        # Convolve
        broadened_spectrum = np.convolve(spectrum, kernel, mode='same')
        return broadened_spectrum
    
    def _calculate_r_squared(self, observed: np.ndarray, model: np.ndarray) -> float:
        """Calculate R² value"""
        ss_res = np.sum((observed - model)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        return 1 - (ss_res / ss_tot)

# Example usage for Li/H exchange in battery materials
def battery_strain_analysis_example():
    """Example of chemical strain analysis for battery materials"""
    
    # Create analyzer for layered oxide (e.g., LiCoO2)
    analyzer = ChemicalStrainAnalyzer('hexagonal', 'vegard')
    
    # Define modes with chemical sensitivity
    # A1g mode sensitive to Li content
    a1g_mode = ChemicalRamanMode(
        name='A1g_CoO2',
        omega0_pure=595.0,  # cm⁻¹ for LiCoO2
        gamma_components_pure=np.array([1.2, 1.2, 0.8, 0, 0, 0]),
        composition_sensitivity=np.array([0.3, 0.3, -0.2, 0, 0, 0]),  # dγ/dx_Li
        jahn_teller_coupling=0.5,  # Couples to Co³⁺/Co⁴⁺ JT distortion
        intensity_pure=100.0
    )
    
    # Eg mode with different sensitivity
    eg_mode = ChemicalRamanMode(
        name='Eg_CoO2',
        omega0_pure=485.0,
        gamma_components_pure=np.array([0.8, 0.8, 1.5, 0, 0, 0]),
        composition_sensitivity=np.array([0.1, 0.1, 0.4, 0, 0, 0]),
        jahn_teller_coupling=0.2,
        intensity_pure=60.0
    )
    
    analyzer.modes['A1g'] = a1g_mode
    analyzer.modes['Eg'] = eg_mode
    
    # Set chemical state
    analyzer.set_composition(0.7)  # Li₀.₇CoO₂
    analyzer.set_jahn_teller_distortion(0.02)  # Small JT distortion
    analyzer.chemical_disorder = 0.1  # 10% disorder
    
    print("Chemical strain analysis for Li₀.₇CoO₂:")
    print(f"Composition: {analyzer.composition}")
    print(f"JT parameter: {analyzer.jahn_teller_parameter}")
    print(f"Chemical disorder: {analyzer.chemical_disorder}")
    
    return analyzer

if __name__ == "__main__":
    analyzer = battery_strain_analysis_example() 