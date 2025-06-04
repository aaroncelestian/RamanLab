import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class RamanDensityAnalyzer:
    """
    Quantitative density analysis of kidney stone Raman spectroscopy data
    for correlation with micro-CT measurements.
    """
    
    def __init__(self):
        # COM characteristic peaks (cm-1)
        self.com_peaks = {
            'main': 1462,      # Primary COM peak
            'secondary': 895,   # Secondary COM peak
            'tertiary': 1630   # Tertiary COM peak
        }
        
        # Organic reference regions (cm-1)
        self.organic_regions = {
            'ch_stretch': (2800, 3000),    # C-H stretching
            'amide_i': (1640, 1680),       # Protein Amide I
            'baseline': (400, 600)         # Low-frequency baseline
        }
        
        # Known densities (g/cm³) - Updated for bacterial biofilm analysis
        self.densities = {
            'com_crystal': 2.23,
            'organic_matrix': 1.0,      # Adjusted for bacteria-rich regions (1.0-1.2 g/cm³)
            'pure_bacteria': 1.0,       # Pure bacteria (0.95-1.05 g/cm³ depending on water content)
            'void_space': 0.0
        }
        
        # Bacterial biofilm density ranges for calibration
        self.biofilm_densities = {
            'pure_bacteria_range': (0.95, 1.05),           # Pure bacteria depending on water content
            'bacteria_rich_range': (1.0, 1.2),             # Bacteria + some mineral precipitation
            'mixed_organic_range': (1.0, 1.4),             # Mixed organic/bacterial matrix
            'crystalline_range': (2.0, 2.4)                # COM crystal range
        }
    
    def preprocess_spectrum(self, wavenumber, intensity):
        """
        Preprocess Raman spectrum with baseline correction and smoothing.
        
        Parameters:
        -----------
        wavenumber : array
            Wavenumber values (cm-1)
        intensity : array
            Raman intensity values
            
        Returns:
        --------
        wavenumber, corrected_intensity : arrays
        """
        # Remove cosmic rays using median filter
        intensity_smooth = signal.medfilt(intensity, kernel_size=3)
        
        # Asymmetric least squares baseline correction
        baseline = self._als_baseline(intensity_smooth, lam=1e6, p=0.01)
        corrected_intensity = intensity_smooth - baseline
        
        # Ensure non-negative values
        corrected_intensity = np.maximum(corrected_intensity, 0)
        
        return wavenumber, corrected_intensity
    
    def _als_baseline(self, y, lam=1e6, p=0.01, niter=10):
        """Asymmetric Least Squares baseline correction"""
        L = len(y)
        D = np.diff(np.eye(L), n=2, axis=0)
        w = np.ones(L)
        
        for i in range(niter):
            W = np.diag(w)
            Z = W + lam * D.T @ D
            z = np.linalg.solve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        
        return z
    
    def calculate_crystalline_density_index(self, wavenumber, intensity):
        """
        Calculate Crystalline Density Index (CDI) using peak height ratios.
        
        Parameters:
        -----------
        wavenumber : array
            Wavenumber values (cm-1)
        intensity : array
            Baseline-corrected Raman intensity
            
        Returns:
        --------
        cdi : float
            Crystalline Density Index (0-1)
        metrics : dict
            Detailed analysis metrics
        """
        # Find COM main peak intensity
        com_idx = np.argmin(np.abs(wavenumber - self.com_peaks['main']))
        com_intensity = intensity[com_idx]
        
        # Calculate baseline intensity in organic-dominated region
        baseline_mask = (wavenumber >= self.organic_regions['baseline'][0]) & \
                       (wavenumber <= self.organic_regions['baseline'][1])
        baseline_intensity = np.mean(intensity[baseline_mask])
        
        # Calculate peak-to-baseline ratio
        peak_height = com_intensity - baseline_intensity
        peak_height = max(peak_height, 0)  # Ensure non-negative
        
        # Normalize using empirical calibration from your whewellite standard
        # This value should be calibrated against your panel L data
        whewellite_reference = 1000  # Adjust based on your whewellite spectrum
        
        # Calculate CDI
        cdi = min(peak_height / whewellite_reference, 1.0)
        
        # Additional metrics for validation
        metrics = {
            'com_peak_height': peak_height,
            'baseline_intensity': baseline_intensity,
            'peak_width': self._calculate_peak_width(wavenumber, intensity, com_idx),
            'spectral_contrast': (com_intensity - baseline_intensity) / (com_intensity + baseline_intensity)
        }
        
        return cdi, metrics
    
    def _calculate_peak_width(self, wavenumber, intensity, peak_idx):
        """Calculate full width at half maximum of COM peak"""
        peak_intensity = intensity[peak_idx]
        half_max = peak_intensity / 2
        
        # Find left and right half-maximum points
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and intensity[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
            right_idx += 1
        
        return wavenumber[right_idx] - wavenumber[left_idx]
    
    def calculate_biofilm_density(self, cdi, biofilm_type='mixed'):
        """
        Convert CDI to apparent density specifically calibrated for bacterial biofilms.
        
        Parameters:
        -----------
        cdi : float or array
            Crystalline Density Index values
        biofilm_type : str
            Type of biofilm analysis ('pure', 'bacteria_rich', 'mixed')
            
        Returns:
        --------
        apparent_density : float or array
            Calculated apparent density (g/cm³) optimized for bacterial biofilms
        """
        if biofilm_type == 'pure':
            # Pure bacteria: 0.95-1.05 g/cm³
            min_density, max_density = self.biofilm_densities['pure_bacteria_range']
        elif biofilm_type == 'bacteria_rich':
            # Bacteria-rich regions: 1.0-1.2 g/cm³
            min_density, max_density = self.biofilm_densities['bacteria_rich_range']
        else:  # mixed
            # Mixed organic/bacterial matrix: 1.0-1.4 g/cm³
            min_density, max_density = self.biofilm_densities['mixed_organic_range']
        
        # Linear interpolation between biofilm baseline and COM crystal
        apparent_density = min_density + (self.densities['com_crystal'] - min_density) * cdi
        
        return apparent_density
    
    def calculate_apparent_density(self, cdi):
        """
        Convert CDI to apparent density for micro-CT correlation.
        
        Parameters:
        -----------
        cdi : float or array
            Crystalline Density Index values
            
        Returns:
        --------
        apparent_density : float or array
            Calculated apparent density (g/cm³)
        """
        # Linear mixing model - updated for bacterial biofilm baseline
        apparent_density = (self.densities['organic_matrix'] + 
                          (self.densities['com_crystal'] - self.densities['organic_matrix']) * cdi)
        
        return apparent_density
    
    def analyze_line_scan(self, positions, intensities, wavenumbers, biofilm_mode=False, biofilm_type='mixed'):
        """
        Analyze Raman line scan data to extract density profile.
        
        Parameters:
        -----------
        positions : array
            Spatial positions along line scan (μm)
        intensities : 2D array
            Intensity matrix (positions × wavenumbers)
        wavenumbers : array
            Wavenumber values (cm-1)
        biofilm_mode : bool
            Use bacterial biofilm calibration instead of standard calibration
        biofilm_type : str
            Type of biofilm analysis ('pure', 'bacteria_rich', 'mixed')
            
        Returns:
        --------
        density_profile : dict
            Spatial density analysis results
        """
        n_positions = len(positions)
        cdi_profile = np.zeros(n_positions)
        density_profile_data = np.zeros(n_positions)
        
        for i in range(n_positions):
            # Preprocess each spectrum
            wn, corrected_int = self.preprocess_spectrum(wavenumbers, intensities[i, :])
            
            # Calculate CDI
            cdi, _ = self.calculate_crystalline_density_index(wn, corrected_int)
            cdi_profile[i] = cdi
            
            # Convert to apparent density - choose calibration method
            if biofilm_mode:
                density_profile_data[i] = self.calculate_biofilm_density(cdi, biofilm_type)
            else:
                density_profile_data[i] = self.calculate_apparent_density(cdi)
        
        density_profile = {
            'positions': positions,
            'cdi_profile': cdi_profile,
            'density_profile': density_profile_data,
            'layer_classification': self._classify_layers(cdi_profile),
            'biofilm_mode': biofilm_mode,
            'biofilm_type': biofilm_type if biofilm_mode else None,
            'statistics': {
                'mean_density': np.mean(density_profile_data),
                'std_density': np.std(density_profile_data),
                'crystalline_fraction': np.mean(cdi_profile),
                'density_range': (np.min(density_profile_data), np.max(density_profile_data))
            }
        }
        
        return density_profile
    
    def analyze_biofilm_sample(self, positions, intensities, wavenumbers, biofilm_type='mixed'):
        """
        Specialized analysis for bacterial biofilm samples with appropriate density calibration.
        
        Parameters:
        -----------
        positions : array
            Spatial positions along line scan (μm)
        intensities : 2D array
            Intensity matrix (positions × wavenumbers)
        wavenumbers : array
            Wavenumber values (cm-1)
        biofilm_type : str
            Type of biofilm analysis ('pure', 'bacteria_rich', 'mixed')
            
        Returns:
        --------
        biofilm_analysis : dict
            Comprehensive biofilm analysis results with bacterial density calibration
        """
        # Use biofilm mode for density calculation
        density_profile = self.analyze_line_scan(positions, intensities, wavenumbers, 
                                                biofilm_mode=True, biofilm_type=biofilm_type)
        
        # Additional biofilm-specific statistics
        cdi_profile = density_profile['cdi_profile']
        density_data = density_profile['density_profile']
        
        # Calculate biofilm composition statistics
        bacterial_regions = np.sum(np.array(density_profile['layer_classification']) == 'bacterial')
        organic_regions = np.sum(np.array(density_profile['layer_classification']) == 'organic')
        crystalline_regions = np.sum(np.array(density_profile['layer_classification']) == 'crystalline')
        mixed_regions = np.sum(np.array(density_profile['layer_classification']) == 'mixed_crystalline')
        
        total_regions = len(density_profile['layer_classification'])
        
        biofilm_stats = {
            'composition_fractions': {
                'bacterial': bacterial_regions / total_regions,
                'organic': organic_regions / total_regions,
                'mixed_crystalline': mixed_regions / total_regions,
                'crystalline': crystalline_regions / total_regions
            },
            'density_in_biofilm_range': {
                'pure_bacteria': np.sum((density_data >= self.biofilm_densities['pure_bacteria_range'][0]) & 
                                      (density_data <= self.biofilm_densities['pure_bacteria_range'][1])) / total_regions,
                'bacteria_rich': np.sum((density_data >= self.biofilm_densities['bacteria_rich_range'][0]) & 
                                      (density_data <= self.biofilm_densities['bacteria_rich_range'][1])) / total_regions
            },
            'biofilm_quality_index': np.mean(cdi_profile[np.array(density_profile['layer_classification']) == 'bacterial'])
        }
        
        # Combine results
        biofilm_analysis = {**density_profile, 'biofilm_statistics': biofilm_stats}
        
        return biofilm_analysis
    
    def _classify_layers(self, cdi_profile, thresholds=None):
        """
        Classify layers as bacterial, organic, or crystalline based on CDI thresholds.
        Updated for bacterial biofilm analysis.
        """
        if thresholds is None:
            thresholds = {
                'bacterial': 0.2,     # Pure bacterial regions (low crystallinity)
                'organic': 0.5,       # Mixed organic/bacterial regions  
                'crystalline': 0.8    # High crystallinity threshold
            }
        
        classifications = []
        for cdi in cdi_profile:
            if cdi < thresholds['bacterial']:
                classifications.append('bacterial')
            elif cdi < thresholds['organic']:
                classifications.append('organic') 
            elif cdi < thresholds['crystalline']:
                classifications.append('mixed_crystalline')
            else:
                classifications.append('crystalline')
                
        return classifications
    
    def plot_density_analysis(self, density_profile, title="Kidney Stone Density Profile"):
        """
        Create comprehensive density analysis visualization.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        positions = density_profile['positions']
        
        # 1. CDI profile
        ax1.plot(positions, density_profile['cdi_profile'], 'b-', linewidth=2)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Classification threshold')
        ax1.set_xlabel('Position (μm)')
        ax1.set_ylabel('Crystalline Density Index')
        ax1.set_title('CDI Spatial Profile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Apparent density profile
        ax2.plot(positions, density_profile['density_profile'], 'g-', linewidth=2)
        ax2.axhline(y=self.densities['com_crystal'], color='b', linestyle=':', 
                   alpha=0.7, label='Pure COM density')
        ax2.axhline(y=self.densities['organic_matrix'], color='r', linestyle=':', 
                   alpha=0.7, label='Organic matrix density')
        
        # Add bacterial biofilm density reference ranges
        ax2.axhspan(self.biofilm_densities['pure_bacteria_range'][0], 
                   self.biofilm_densities['pure_bacteria_range'][1], 
                   alpha=0.2, color='cyan', label='Pure bacteria range')
        ax2.axhspan(self.biofilm_densities['bacteria_rich_range'][0], 
                   self.biofilm_densities['bacteria_rich_range'][1], 
                   alpha=0.2, color='orange', label='Bacteria-rich range')
                   
        ax2.set_xlabel('Position (μm)')
        ax2.set_ylabel('Apparent Density (g/cm³)')
        ax2.set_title('Calculated Density Profile (Bacterial Biofilm Calibrated)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Layer classification - updated for bacterial types
        color_map = {
            'bacterial': 'cyan',
            'organic': 'orange', 
            'mixed_crystalline': 'purple',
            'crystalline': 'blue'
        }
        colors = [color_map.get(layer, 'gray') for layer in density_profile['layer_classification']]
        ax3.scatter(positions, density_profile['cdi_profile'], c=colors, alpha=0.7)
        ax3.set_xlabel('Position (μm)')
        ax3.set_ylabel('CDI')
        ax3.set_title('Layer Classification (Bacterial Biofilm Analysis)')
        ax3.grid(True, alpha=0.3)
        
        # Add legend for layer classification
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='cyan', label='Bacterial'),
                          Patch(facecolor='orange', label='Organic'),
                          Patch(facecolor='purple', label='Mixed Crystalline'),
                          Patch(facecolor='blue', label='Crystalline')]
        ax3.legend(handles=legend_elements)
        
        # 4. Density histogram
        ax4.hist(density_profile['density_profile'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=density_profile['statistics']['mean_density'], 
                   color='red', linestyle='--', linewidth=2, label='Mean density')
        ax4.set_xlabel('Apparent Density (g/cm³)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Density Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        plt.show()
        
        return fig

# Example usage and calibration
def calibrate_with_whewellite(analyzer, whewellite_wavenumber, whewellite_intensity):
    """
    Calibrate the analyzer using whewellite reference spectrum (Panel L).
    """
    wn, corrected_int = analyzer.preprocess_spectrum(whewellite_wavenumber, whewellite_intensity)
    
    # Find the main COM peak in whewellite
    com_idx = np.argmin(np.abs(wn - analyzer.com_peaks['main']))
    
    # Calculate baseline
    baseline_mask = (wn >= analyzer.organic_regions['baseline'][0]) & \
                   (wn <= analyzer.organic_regions['baseline'][1])
    baseline_intensity = np.mean(corrected_int[baseline_mask])
    
    # Peak height for pure COM
    whewellite_peak_height = corrected_int[com_idx] - baseline_intensity
    
    print(f"Whewellite calibration peak height: {whewellite_peak_height:.2f}")
    print("Use this value to update the whewellite_reference parameter in calculate_crystalline_density_index()")
    
    return whewellite_peak_height

# Initialize analyzer
analyzer = RamanDensityAnalyzer()

