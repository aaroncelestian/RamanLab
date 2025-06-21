"""
Spinel Raman Modes for LiMn2O4
==============================

Definition of Raman active modes in spinel LiMn2O4 structure and their
sensitivity to H/Li exchange and associated strain effects.

LiMn2O4 has cubic spinel structure (Fd3m) with Raman active modes:
- A1g: symmetric breathing mode 
- Eg: doubly degenerate bending mode
- T2g: triply degenerate stretching modes

Key effects during H/Li exchange:
1. Jahn-Teller distortion from Mn3+ (d4) formation
2. Changes in Mn-O bond lengths 
3. Symmetry breaking from local disorder
4. Peak broadening from compositional inhomogeneity
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path to import chemical_strain_enhancement
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chemical_strain_enhancement import ChemicalRamanMode

class SpinelRamanModes:
    """
    LiMn2O4 spinel Raman modes with H/Li exchange sensitivity
    
    Based on experimental data and DFT calculations for spinel structure.
    Includes composition-dependent Grüneisen parameters and Jahn-Teller coupling.
    """
    
    def __init__(self):
        """Initialize LiMn2O4 spinel modes"""
        self.modes = self._define_limn2o4_modes()
        self.crystal_system = 'cubic'  # Spinel structure
        self.space_group = 'Fd3m'
        
    def _define_limn2o4_modes(self) -> Dict[str, ChemicalRamanMode]:
        """Define the Raman active modes for LiMn2O4"""
        
        modes = {}
        
        # A1g mode - Symmetric breathing of MnO6 octahedra
        # Most sensitive to Mn-O bond length changes
        modes['A1g'] = ChemicalRamanMode(
            name='A1g_MnO6_breathing',
            omega0_pure=625.0,  # cm⁻¹ for pristine LiMn2O4
            gamma_components_pure=np.array([2.1, 2.1, 2.1, 0.0, 0.0, 0.0]),  # Isotropic for cubic
            intensity_pure=100.0,
            natural_width=3.0,
            
            # H/Li exchange effects
            composition_sensitivity=np.array([0.8, 0.8, 0.8, 0.0, 0.0, 0.0]),  # dγ/dx_H
            jahn_teller_coupling=1.2,  # Strong coupling due to Mn3+ JT distortion
            defect_strain_radius=6.0  # Angstroms
        )
        
        # Eg mode - Bending mode of MnO6 octahedra
        # Sensitive to octahedral distortion
        modes['Eg'] = ChemicalRamanMode(
            name='Eg_MnO6_bending',
            omega0_pure=580.0,  # cm⁻¹
            gamma_components_pure=np.array([1.5, 1.5, 1.8, 0.0, 0.0, 0.0]),
            intensity_pure=70.0,
            natural_width=4.0,
            
            composition_sensitivity=np.array([0.6, 0.6, 1.0, 0.0, 0.0, 0.0]),
            jahn_teller_coupling=0.8,  # Moderate coupling
            defect_strain_radius=5.5
        )
        
        # T2g mode 1 - First triply degenerate stretching mode
        # Sensitive to framework distortion
        modes['T2g_1'] = ChemicalRamanMode(
            name='T2g_1_framework',
            omega0_pure=480.0,  # cm⁻¹
            gamma_components_pure=np.array([1.2, 1.2, 1.0, 0.3, 0.3, 0.3]),
            intensity_pure=85.0,
            natural_width=5.0,
            
            composition_sensitivity=np.array([0.4, 0.4, 0.2, 0.1, 0.1, 0.1]),
            jahn_teller_coupling=0.5,
            defect_strain_radius=7.0
        )
        
        # T2g mode 2 - Second triply degenerate stretching mode
        # Often splits in distorted structure
        modes['T2g_2'] = ChemicalRamanMode(
            name='T2g_2_framework',
            omega0_pure=430.0,  # cm⁻¹
            gamma_components_pure=np.array([1.0, 1.0, 0.8, 0.2, 0.2, 0.2]),
            intensity_pure=60.0,
            natural_width=6.0,
            
            composition_sensitivity=np.array([0.3, 0.3, 0.1, 0.15, 0.15, 0.15]),
            jahn_teller_coupling=0.4,
            defect_strain_radius=6.5
        )
        
        # Additional modes that may appear with disorder/distortion
        
        # Li-O stretching mode (weak, appears with Li/H exchange)
        modes['Li_O'] = ChemicalRamanMode(
            name='Li_O_stretch',
            omega0_pure=280.0,  # cm⁻¹
            gamma_components_pure=np.array([0.5, 0.5, 0.3, 0.0, 0.0, 0.0]),
            intensity_pure=20.0,
            natural_width=8.0,
            
            composition_sensitivity=np.array([1.5, 1.5, 0.8, 0.0, 0.0, 0.0]),  # Strong H/Li sensitivity
            jahn_teller_coupling=0.1,
            defect_strain_radius=4.0
        )
        
        # Disorder-activated mode (forbidden in perfect spinel)
        modes['Disorder'] = ChemicalRamanMode(
            name='Disorder_activated',
            omega0_pure=515.0,  # cm⁻¹
            gamma_components_pure=np.array([0.8, 0.8, 0.6, 0.4, 0.4, 0.4]),
            intensity_pure=5.0,  # Very weak in pristine material
            natural_width=12.0,  # Broad due to disorder
            
            composition_sensitivity=np.array([0.2, 0.2, 0.1, 0.6, 0.6, 0.6]),  # Shear strain sensitive
            jahn_teller_coupling=0.6,
            defect_strain_radius=8.0
        )
        
        return modes
    
    def get_mode_names(self) -> List[str]:
        """Get list of all mode names"""
        return list(self.modes.keys())
    
    def get_primary_modes(self) -> List[str]:
        """Get primary modes (strongest intensity)"""
        return ['A1g', 'Eg', 'T2g_1', 'T2g_2']
    
    def get_composition_sensitive_modes(self) -> List[str]:
        """Get modes most sensitive to H/Li composition"""
        return ['A1g', 'Li_O', 'Eg']
    
    def get_jahn_teller_sensitive_modes(self) -> List[str]:
        """Get modes most sensitive to Jahn-Teller distortion"""
        return ['A1g', 'Eg', 'Disorder']
    
    def calculate_mode_splitting(self, jt_parameter: float) -> Dict[str, Tuple[float, float]]:
        """
        Calculate mode splitting due to Jahn-Teller distortion
        
        In tetragonal distortion (elongated octahedra):
        - Eg splits into A1g + B1g  
        - T2g splits into B2g + Eg
        
        Returns:
        Dict with mode names and (lower_freq, upper_freq) splitting
        """
        splitting = {}
        
        for mode_name, mode in self.modes.items():
            base_freq = mode.omega0_pure
            jt_shift = mode.jahn_teller_coupling * jt_parameter * base_freq
            
            if mode_name == 'Eg':
                # Eg splits into two components
                splitting[mode_name] = (base_freq - jt_shift, base_freq + jt_shift)
            elif mode_name.startswith('T2g'):
                # T2g splits into three components (simplified to two main ones)
                splitting[mode_name] = (base_freq - 0.7*jt_shift, base_freq + 0.5*jt_shift)
            else:
                # Other modes shift but don't split significantly
                splitting[mode_name] = (base_freq + jt_shift, base_freq + jt_shift)
        
        return splitting
    
    def get_temperature_effects(self, temperature: float) -> Dict[str, float]:
        """
        Calculate temperature-dependent frequency shifts
        
        Uses quasi-harmonic approximation with typical anharmonic parameters
        """
        # Reference temperature (room temperature)
        T0 = 298.0  # K
        
        temperature_shifts = {}
        
        for mode_name, mode in self.modes.items():
            # Typical anharmonic coefficient for oxide materials
            # Higher frequency modes have larger anharmonic effects
            alpha = -0.01 * (mode.omega0_pure / 500.0)  # cm⁻¹/K
            
            # Linear temperature dependence (simplified)
            freq_shift = alpha * (temperature - T0)
            temperature_shifts[mode_name] = freq_shift
        
        return temperature_shifts
    
    def estimate_disorder_broadening(self, composition_variance: float, 
                                   domain_size: float = 50.0) -> Dict[str, float]:
        """
        Estimate additional broadening from compositional disorder
        
        Parameters:
        composition_variance: Variance in local composition
        domain_size: Coherence length in Angstroms
        """
        broadening = {}
        
        for mode_name, mode in self.modes.items():
            # Broadening proportional to composition sensitivity
            if mode.composition_sensitivity is not None:
                sens_magnitude = np.linalg.norm(mode.composition_sensitivity)
            else:
                sens_magnitude = 0.1
            
            # Domain size effects (smaller domains = more broadening)
            size_factor = 50.0 / max(domain_size, 10.0)
            
            disorder_width = sens_magnitude * composition_variance * mode.omega0_pure * 0.01 * size_factor
            broadening[mode_name] = disorder_width
        
        return broadening

# Example usage and testing
if __name__ == "__main__":
    # Create spinel modes instance
    spinel = SpinelRamanModes()
    
    print("LiMn2O4 Spinel Raman Modes")
    print("=" * 40)
    
    for mode_name, mode in spinel.modes.items():
        print(f"\n{mode_name}:")
        print(f"  Frequency: {mode.omega0_pure:.1f} cm⁻¹")
        print(f"  Intensity: {mode.intensity_pure:.1f}")
        print(f"  Natural width: {mode.natural_width:.1f} cm⁻¹")
        print(f"  JT coupling: {mode.jahn_teller_coupling:.2f}")
    
    # Test mode splitting
    print("\n" + "=" * 40)
    print("Mode splitting with JT distortion (0.05):")
    splitting = spinel.calculate_mode_splitting(0.05)
    for mode, (low, high) in splitting.items():
        if abs(low - high) > 1.0:  # Only show significant splitting
            print(f"{mode}: {low:.1f} - {high:.1f} cm⁻¹")
    
    # Test disorder broadening
    print("\n" + "=" * 40)
    print("Disorder broadening (10% composition variance):")
    broadening = spinel.estimate_disorder_broadening(0.1)
    for mode, width in broadening.items():
        if width > 1.0:  # Only show significant broadening
            print(f"{mode}: +{width:.1f} cm⁻¹") 