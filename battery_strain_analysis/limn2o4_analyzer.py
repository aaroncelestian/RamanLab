"""
LiMn2O4 Strain Analyzer
======================

Specialized implementation of chemical strain analysis for LiMn2O4 spinel
battery materials undergoing H/Li exchange.

Key features:
1. Spinel-specific strain tensor analysis
2. Jahn-Teller distortion tracking
3. H/Li composition determination
4. Phase transition detection
5. Time series analysis capabilities
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, List, Tuple, Optional, Union
import warnings
import sys
import os

# Import parent modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chemical_strain_enhancement import ChemicalStrainAnalyzer
from battery_strain_analysis.spinel_modes import SpinelRamanModes

class LiMn2O4StrainAnalyzer(ChemicalStrainAnalyzer):
    """
    Specialized strain analyzer for LiMn2O4 spinel battery material
    
    Handles H/Li exchange effects including:
    - Jahn-Teller distortion from Mn3+ formation
    - Symmetry breaking and mode splitting
    - Composition-dependent strain evolution
    - Phase separation during cycling
    """
    
    def __init__(self, temperature: float = 298.0):
        """
        Initialize LiMn2O4 analyzer
        
        Parameters:
        temperature: Analysis temperature in Kelvin
        """
        # Initialize parent with cubic spinel parameters
        super().__init__('cubic', 'vegard') 
        
        # LiMn2O4 specific parameters
        self.temperature = temperature
        self.spinel_modes = SpinelRamanModes()
        
        # Copy spinel modes to parent analyzer
        for mode_name, mode in self.spinel_modes.modes.items():
            self.modes[mode_name] = mode
        
        # Battery-specific state variables
        self.lithium_content = 1.0  # x in LixMn2O4 (1.0 = fully lithiated)
        self.hydrogen_content = 0.0  # y in HyMn2O4 
        self.mn3_fraction = 0.0  # Fraction of Mn3+ (causes JT distortion)
        self.phase_separation = False
        self.cycling_history = []
        
        # Structural parameters
        self.lattice_parameter = 8.247  # Angstroms for LiMn2O4
        self.tetragonal_distortion = 0.0  # c/a - 1 for JT distortion
        
        # Analysis settings
        self.fit_jahn_teller = True
        self.fit_composition = True
        self.track_phase_separation = True
        
    def set_battery_state(self, li_content: float, h_content: float = None):
        """
        Set the current battery composition state
        
        Parameters:
        li_content: Lithium content x in LixMn2O4
        h_content: Hydrogen content y in HyMn2O4 (if None, calculated from charge balance)
        """
        self.lithium_content = np.clip(li_content, 0.0, 1.0)
        
        if h_content is None:
            # Assume charge balance: x_Li + y_H = 1
            self.hydrogen_content = max(0.0, 1.0 - self.lithium_content)
        else:
            self.hydrogen_content = np.clip(h_content, 0.0, 1.0)
        
        # Calculate Mn3+ fraction from electroneutrality
        # Li+ contributes +1, H+ contributes +1, Mn3+ and Mn4+ balance
        # For simplification: Mn3+ fraction ≈ 1 - Li_content (at constant total charge)
        self.mn3_fraction = max(0.0, 1.0 - self.lithium_content)
        
        # Update composition-dependent parameters
        total_content = self.lithium_content + self.hydrogen_content
        if total_content > 0:
            h_fraction = self.hydrogen_content / total_content
            self.set_composition(h_fraction)
        
        # Set Jahn-Teller parameter based on Mn3+ content
        # Mn3+ (d4) exhibits JT distortion, Mn4+ (d3) does not
        jt_strength = 0.1 * self.mn3_fraction  # Empirical scaling
        self.set_jahn_teller_distortion(jt_strength)
    
    def analyze_spectrum(self, frequencies: np.ndarray, 
                        intensities: np.ndarray,
                        known_composition: Dict = None,
                        background_subtract: bool = True) -> Dict:
        """
        Analyze a single Raman spectrum for strain and composition
        
        Parameters:
        frequencies: Wavenumber array (cm⁻¹)
        intensities: Raman intensity array
        known_composition: Optional known Li/H content {'Li': x, 'H': y}
        background_subtract: Whether to subtract polynomial background
        
        Returns:
        Dict with analysis results
        """
        
        # Preprocessing
        if background_subtract:
            intensities = self._subtract_background(frequencies, intensities)
        
        # Smooth spectrum
        if len(intensities) > 10:
            intensities = savgol_filter(intensities, 
                                      min(11, len(intensities)//4*2+1), 2)
        
        # Peak detection and assignment
        peaks = self._detect_and_assign_peaks(frequencies, intensities)
        
        # Set composition if known
        if known_composition:
            li_content = known_composition.get('Li', self.lithium_content)
            h_content = known_composition.get('H', self.hydrogen_content)
            self.set_battery_state(li_content, h_content)
            fit_composition = False
        else:
            fit_composition = self.fit_composition
        
        # Fit strain tensor and composition
        fit_result = self.fit_chemical_strain_tensor(
            frequencies, intensities,
            composition=None if fit_composition else self.composition,
            fit_composition=fit_composition,
            fit_jahn_teller=self.fit_jahn_teller
        )
        
        # Extract battery-specific results
        battery_results = self._extract_battery_parameters(fit_result, peaks)
        
        # Add peak analysis
        battery_results['peaks'] = peaks
        battery_results['preprocessing'] = {
            'background_subtracted': background_subtract,
            'smoothed': True
        }
        
        return battery_results
    
    def _subtract_background(self, frequencies: np.ndarray, 
                           intensities: np.ndarray) -> np.ndarray:
        """Subtract polynomial background from spectrum"""
        try:
            # Fit polynomial background (typically 2nd or 3rd order)
            coeffs = np.polyfit(frequencies, intensities, 3)
            background = np.polyval(coeffs, frequencies)
            return intensities - background
        except:
            warnings.warn("Background subtraction failed, returning original spectrum")
            return intensities
    
    def _detect_and_assign_peaks(self, frequencies: np.ndarray, 
                                intensities: np.ndarray) -> Dict:
        """
        Detect peaks and assign them to vibrational modes
        
        Returns:
        Dict with peak assignments and properties
        """
        # Find peaks
        peaks, properties = find_peaks(intensities, 
                                     height=np.max(intensities)*0.1,
                                     distance=len(frequencies)//50)
        
        peak_freqs = frequencies[peaks]
        peak_heights = intensities[peaks]
        
        # Assign peaks to modes based on expected frequencies
        assignments = {}
        mode_tolerances = {
            'A1g': 15.0,      # ±15 cm⁻¹ tolerance
            'Eg': 20.0,       # Broader tolerance for split modes
            'T2g_1': 15.0,
            'T2g_2': 20.0,
            'Li_O': 25.0,     # Broader for weak mode
            'Disorder': 30.0  # Very broad disorder-activated mode
        }
        
        for mode_name, mode in self.spinel_modes.modes.items():
            expected_freq = mode.omega0_pure
            tolerance = mode_tolerances.get(mode_name, 15.0)
            
            # Find closest peak within tolerance
            distances = np.abs(peak_freqs - expected_freq)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < tolerance:
                peak_idx = peaks[closest_idx]
                assignments[mode_name] = {
                    'frequency': peak_freqs[closest_idx],
                    'intensity': peak_heights[closest_idx],
                    'expected_frequency': expected_freq,
                    'shift': peak_freqs[closest_idx] - expected_freq,
                    'peak_index': peak_idx
                }
        
        return assignments
    
    def _extract_battery_parameters(self, fit_result: Dict, peaks: Dict) -> Dict:
        """Extract battery-specific parameters from fit results"""
        
        battery_params = fit_result.copy()
        
        # Calculate lattice strain
        strain_tensor = fit_result['strain_tensor']
        hydrostatic_strain = (strain_tensor[0] + strain_tensor[1] + strain_tensor[2]) / 3
        deviatoric_strain = np.sqrt(np.sum((strain_tensor[:3] - hydrostatic_strain)**2))
        
        battery_params['lattice_parameters'] = {
            'hydrostatic_strain': hydrostatic_strain,
            'deviatoric_strain': deviatoric_strain,
            'volume_strain': 3 * hydrostatic_strain,
            'tetragonal_distortion': strain_tensor[2] - strain_tensor[0]  # c/a distortion
        }
        
        # Estimate battery state from composition
        if 'composition' in fit_result:
            h_fraction = fit_result['composition']
            estimated_li = 1.0 - h_fraction  # Simplified assumption
            estimated_h = h_fraction
            
            battery_params['battery_state'] = {
                'lithium_content': estimated_li,
                'hydrogen_content': estimated_h,
                'mn3_fraction': max(0.0, 1.0 - estimated_li),
            }
        
        # Analyze mode behavior
        mode_analysis = {}
        for mode_name, peak_data in peaks.items():
            mode = self.spinel_modes.modes.get(mode_name)
            if mode:
                # Calculate broadening beyond natural width
                apparent_width = self._estimate_peak_width(peak_data)
                excess_broadening = max(0, apparent_width - mode.natural_width)
                
                mode_analysis[mode_name] = {
                    'frequency_shift': peak_data['shift'],
                    'relative_intensity': peak_data['intensity'] / mode.intensity_pure,
                    'excess_broadening': excess_broadening,
                    'strain_sensitivity': np.linalg.norm(mode.gamma_components_pure)
                }
        
        battery_params['mode_analysis'] = mode_analysis
        
        # Detect phase transitions
        battery_params['phase_analysis'] = self._analyze_phase_transitions(peaks)
        
        return battery_params
    
    def _estimate_peak_width(self, peak_data: Dict) -> float:
        """Estimate peak width from peak data (simplified)"""
        # This would normally require peak fitting
        # For now, return a reasonable estimate
        return 8.0  # cm⁻¹ typical for these materials
    
    def _analyze_phase_transitions(self, peaks: Dict) -> Dict:
        """Analyze indicators of phase transitions"""
        phase_indicators = {
            'jahn_teller_active': False,
            'symmetry_breaking': False,
            'phase_separation': False,
            'disorder_level': 0.0
        }
        
        # Check for Jahn-Teller activity (Eg mode splitting)
        if 'Eg' in peaks:
            eg_shift = abs(peaks['Eg']['shift'])
            if eg_shift > 10.0:  # Significant shift indicates JT distortion
                phase_indicators['jahn_teller_active'] = True
        
        # Check for symmetry breaking (disorder mode activation)
        if 'Disorder' in peaks:
            disorder_intensity = peaks['Disorder']['intensity']
            if disorder_intensity > 10.0:  # Arbitrary threshold
                phase_indicators['symmetry_breaking'] = True
                phase_indicators['disorder_level'] = disorder_intensity / 100.0
        
        # Check for mode splitting (would need more sophisticated analysis)
        broad_modes = 0
        for mode_name, peak_data in peaks.items():
            if peak_data.get('excess_broadening', 0) > 5.0:
                broad_modes += 1
        
        if broad_modes >= 2:
            phase_indicators['phase_separation'] = True
        
        return phase_indicators
    
    def analyze_time_series(self, time_data: List[Dict]) -> Dict:
        """
        Analyze time series of Raman spectra during H/Li exchange
        
        Parameters:
        time_data: List of dicts with keys 'time', 'frequencies', 'intensities'
                  and optionally 'composition'
        
        Returns:
        Dict with time-resolved analysis
        """
        
        results = {
            'time_points': [],
            'strain_evolution': [],
            'composition_evolution': [],
            'jt_evolution': [],
            'phase_transitions': [],
            'peak_tracking': {}
        }
        
        # Initialize peak tracking for main modes
        for mode_name in self.spinel_modes.get_primary_modes():
            results['peak_tracking'][mode_name] = {
                'frequencies': [],
                'intensities': [],
                'widths': []
            }
        
        # Analyze each time point
        for i, data_point in enumerate(time_data):
            time = data_point['time']
            frequencies = data_point['frequencies']
            intensities = data_point['intensities']
            known_comp = data_point.get('composition', None)
            
            # Analyze single spectrum
            analysis = self.analyze_spectrum(frequencies, intensities, known_comp)
            
            # Store results
            results['time_points'].append(time)
            results['strain_evolution'].append(analysis['strain_tensor'])
            results['composition_evolution'].append(analysis.get('composition', 0.5))
            results['jt_evolution'].append(analysis.get('jahn_teller_parameter', 0.0))
            results['phase_transitions'].append(analysis['phase_analysis'])
            
            # Track peak evolution
            for mode_name, peak_data in analysis.get('peaks', {}).items():
                if mode_name in results['peak_tracking']:
                    results['peak_tracking'][mode_name]['frequencies'].append(
                        peak_data['frequency'])
                    results['peak_tracking'][mode_name]['intensities'].append(
                        peak_data['intensity'])
                    results['peak_tracking'][mode_name]['widths'].append(
                        peak_data.get('excess_broadening', 0))
        
        # Convert lists to arrays for easier analysis
        for key in ['strain_evolution', 'composition_evolution', 'jt_evolution']:
            results[key] = np.array(results[key])
        
        # Calculate time derivatives and trends
        results['analysis_summary'] = self._summarize_time_series(results)
        
        return results
    
    def _summarize_time_series(self, results: Dict) -> Dict:
        """Summarize key trends in time series data"""
        
        summary = {}
        
        # Strain rate analysis
        if len(results['strain_evolution']) > 1:
            times = np.array(results['time_points'])
            strains = results['strain_evolution']
            
            # Calculate strain rates
            dt = np.diff(times)
            strain_rates = np.diff(strains, axis=0) / dt[:, np.newaxis]
            
            summary['max_strain_rate'] = np.max(np.abs(strain_rates))
            summary['average_strain_rate'] = np.mean(np.abs(strain_rates), axis=0)
        
        # Composition change rate
        if len(results['composition_evolution']) > 1:
            comp_rate = np.diff(results['composition_evolution']) / np.diff(results['time_points'])
            summary['composition_rate'] = np.mean(np.abs(comp_rate))
        
        # Phase transition detection
        jt_active_times = []
        for i, phase_data in enumerate(results['phase_transitions']):
            if phase_data.get('jahn_teller_active', False):
                jt_active_times.append(results['time_points'][i])
        
        if jt_active_times:
            summary['jt_transition_time'] = min(jt_active_times)
            summary['jt_active_duration'] = max(jt_active_times) - min(jt_active_times)
        
        # Peak degradation analysis
        for mode_name, tracking_data in results['peak_tracking'].items():
            if tracking_data['intensities']:
                intensities = np.array(tracking_data['intensities'])
                initial_intensity = intensities[0]
                final_intensity = intensities[-1]
                
                summary[f'{mode_name}_intensity_change'] = (final_intensity - initial_intensity) / initial_intensity
        
        return summary
    
    def forward_model(self, strain_params: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """
        Forward model to calculate Raman spectrum from strain parameters
        
        This method generates a synthetic Raman spectrum based on the current
        strain state and mode parameters.
        
        Parameters:
        strain_params: 6-component strain vector [ε₁₁, ε₂₂, ε₃₃, ε₁₂, ε₁₃, ε₂₃]
        frequencies: Frequency array (cm⁻¹)
        
        Returns:
        np.ndarray: Calculated spectrum intensities
        """
        
        spectrum = np.zeros_like(frequencies)
        
        # Calculate strain-induced frequency shifts for each mode
        for mode_name, mode in self.modes.items():
            if isinstance(mode, type(self.spinel_modes.modes['A1g'])):  # ChemicalRamanMode
                # Base frequency
                base_freq = mode.omega0_pure
                
                # Strain-induced frequency shift using Grüneisen parameters
                gamma_components = self._get_effective_gruneisen(mode)
                freq_shift = np.dot(gamma_components, strain_params) * base_freq
                
                # Calculate shifted frequency
                shifted_freq = base_freq + freq_shift
                
                # Calculate intensity (may depend on strain/composition)
                intensity = self._get_effective_intensity(mode)
                
                # Calculate linewidth (natural + strain broadening)
                linewidth = self._get_effective_linewidth(mode, strain_params)
                
                # Generate peak (Gaussian/Lorentzian profile)
                peak_spectrum = self._generate_peak(frequencies, shifted_freq, 
                                                  intensity, linewidth)
                
                spectrum += peak_spectrum
        
        return spectrum
    
    def _get_effective_gruneisen(self, mode) -> np.ndarray:
        """Get effective Grüneisen parameters for current state"""
        base_gamma = mode.gamma_components_pure.copy()
        
        # Add composition-dependent corrections
        if mode.composition_sensitivity is not None:
            base_gamma += mode.composition_sensitivity * self.composition
        
        # Add Jahn-Teller corrections
        if mode.jahn_teller_coupling != 0:
            jt_correction = np.zeros(6)
            jt_correction[0] = mode.jahn_teller_coupling * self.jahn_teller_parameter
            jt_correction[1] = -mode.jahn_teller_coupling * self.jahn_teller_parameter
            base_gamma += jt_correction
        
        return base_gamma
    
    def _get_effective_intensity(self, mode) -> float:
        """Get effective intensity for current state"""
        base_intensity = mode.intensity_pure
        
        # Composition-dependent intensity changes
        if hasattr(mode, 'name'):
            if 'Li_O' in mode.name:
                # Li-O mode weakens with delithiation
                base_intensity *= self.lithium_content
            elif 'Disorder' in mode.name:
                # Disorder mode strengthens with chemical disorder
                disorder_level = self.lithium_content * self.hydrogen_content * 4
                base_intensity *= disorder_level
        
        return base_intensity
    
    def _get_effective_linewidth(self, mode, strain_params: np.ndarray) -> float:
        """Get effective linewidth including strain broadening"""
        natural_width = mode.natural_width
        
        # Strain-induced broadening (simplified model)
        strain_magnitude = np.linalg.norm(strain_params)
        strain_broadening = strain_magnitude * 10.0  # Empirical scaling
        
        # Chemical disorder broadening
        disorder_level = self.lithium_content * self.hydrogen_content * 4
        disorder_broadening = disorder_level * 2.0
        
        total_width = natural_width + strain_broadening + disorder_broadening
        
        return max(total_width, 1.0)  # Minimum width
    
    def _generate_peak(self, frequencies: np.ndarray, center: float, 
                      intensity: float, width: float) -> np.ndarray:
        """Generate a peak profile (Gaussian)"""
        
        if intensity <= 0:
            return np.zeros_like(frequencies)
        
        # Gaussian peak
        peak = intensity * np.exp(-((frequencies - center) / width)**2)
        
        return peak

# Example usage and testing
if __name__ == "__main__":
    # Create analyzer
    analyzer = LiMn2O4StrainAnalyzer(temperature=298.0)
    
    print("LiMn2O4 Strain Analyzer")
    print("=" * 40)
    
    # Set initial state (fully lithiated)
    analyzer.set_battery_state(li_content=1.0, h_content=0.0)
    print(f"Initial state: Li{analyzer.lithium_content}H{analyzer.hydrogen_content}Mn2O4")
    print(f"Mn3+ fraction: {analyzer.mn3_fraction:.2f}")
    print(f"JT parameter: {analyzer.jahn_teller_parameter:.4f}")
    
    print("\nAvailable modes:")
    for mode_name in analyzer.spinel_modes.get_mode_names():
        mode = analyzer.spinel_modes.modes[mode_name]
        print(f"  {mode_name}: {mode.omega0_pure:.1f} cm⁻¹") 