import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, integrate
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class MaterialConfigs:
    """Material configuration library for different types of Raman density analysis."""
    
    @staticmethod
    def get_kidney_stones_config():
        return {
            'name': 'Kidney Stones (COM)',
            'characteristic_peaks': {
                'main': 1462,      # Primary COM peak
                'secondary': 895,   # Secondary COM peak
                'tertiary': 1630   # Tertiary COM peak
            },
            'reference_regions': {
                'ch_stretch': (2800, 3000),    # C-H stretching
                'amide_i': (1640, 1680),       # Protein Amide I
                'baseline': (400, 600)         # Low-frequency baseline
            },
            'densities': {
                'crystalline': 2.23,    # COM crystal
                'matrix': 1.0,          # Organic matrix
                'low_density': 1.0      # Bacteria/organic
            },
            'density_ranges': {
                'low_range': (0.95, 1.05),      # Pure bacteria
                'medium_range': (1.0, 1.2),     # Bacteria + some mineral
                'mixed_range': (1.0, 1.4),      # Mixed organic matrix
                'crystalline_range': (2.0, 2.4) # COM crystal range
            },
            'reference_intensity': 1000,  # Whewellite reference
            'classification_thresholds': {
                'low': 0.2,       # Bacterial/organic
                'medium': 0.5,    # Mixed organic
                'high': 0.8       # Crystalline
            }
        }
    
    @staticmethod
    def get_quartz_config():
        return {
            'name': 'Quartz',
            'characteristic_peaks': {
                'main': 464,       # Primary quartz peak
                'secondary': 206,  # Secondary quartz peak
                'tertiary': 354    # Tertiary quartz peak
            },
            'reference_regions': {
                'low_freq': (100, 300),      # Low frequency region
                'fingerprint': (300, 600),   # Fingerprint region
                'baseline': (700, 900)       # Baseline region
            },
            'densities': {
                'crystalline': 2.65,   # Pure quartz
                'matrix': 2.0,         # Mixed silicate matrix
                'low_density': 1.5     # Amorphous/organic
            },
            'density_ranges': {
                'low_range': (1.4, 1.6),        # Amorphous silica
                'medium_range': (1.8, 2.2),     # Mixed crystalline
                'mixed_range': (2.0, 2.4),      # Polycrystalline
                'crystalline_range': (2.5, 2.7) # Pure quartz
            },
            'reference_intensity': 800,   # Pure quartz reference
            'classification_thresholds': {
                'low': 0.3,       # Amorphous
                'medium': 0.6,    # Mixed crystalline
                'high': 0.85      # Pure crystalline
            }
        }
    
    @staticmethod
    def get_feldspar_config():
        return {
            'name': 'Feldspar',
            'characteristic_peaks': {
                'main': 476,       # Primary feldspar peak
                'secondary': 507,  # Secondary peak
                'tertiary': 284    # Tertiary peak
            },
            'reference_regions': {
                'aluminosilicate': (400, 600),  # Al-Si framework
                'alkali': (200, 400),           # Alkali region
                'baseline': (800, 1000)        # Baseline region
            },
            'densities': {
                'crystalline': 2.56,   # Average feldspar density
                'matrix': 2.2,         # Mixed aluminosilicate
                'low_density': 1.8     # Glassy/amorphous
            },
            'density_ranges': {
                'low_range': (1.7, 1.9),        # Glassy feldspar
                'medium_range': (2.0, 2.3),     # Mixed crystalline
                'mixed_range': (2.2, 2.4),      # Polycrystalline
                'crystalline_range': (2.5, 2.6) # Pure feldspar
            },
            'reference_intensity': 600,   # Pure feldspar reference
            'classification_thresholds': {
                'low': 0.25,      # Glassy
                'medium': 0.55,   # Mixed
                'high': 0.8       # Crystalline
            }
        }
    
    @staticmethod
    def get_calcite_config():
        return {
            'name': 'Calcite',
            'characteristic_peaks': {
                'main': 1086,      # Primary calcite peak (ν1 CO3)
                'secondary': 282,  # Secondary peak
                'tertiary': 712    # ν4 CO3 peak
            },
            'reference_regions': {
                'carbonate': (1000, 1200),   # Carbonate stretching
                'lattice': (200, 400),       # Lattice modes
                'baseline': (600, 800)       # Baseline region
            },
            'densities': {
                'crystalline': 2.71,   # Pure calcite
                'matrix': 2.3,         # Mixed carbonate
                'low_density': 1.9     # Organic/amorphous
            },
            'density_ranges': {
                'low_range': (1.8, 2.0),        # Organic-rich
                'medium_range': (2.1, 2.4),     # Mixed carbonate
                'mixed_range': (2.3, 2.5),      # Polycrystalline
                'crystalline_range': (2.6, 2.8) # Pure calcite
            },
            'reference_intensity': 1200,  # Pure calcite reference
            'classification_thresholds': {
                'low': 0.3,       # Organic-rich
                'medium': 0.6,    # Mixed
                'high': 0.85      # Pure crystalline
            }
        }
    
    # Class variable to store custom configurations
    _custom_configs = {}
    
    @staticmethod
    def get_available_materials():
        """Get list of available material configurations."""
        built_in_materials = [
            'Kidney Stones (COM)',
            'Quartz',
            'Feldspar', 
            'Calcite',
            'Other (Custom)'
        ]
        # Add any custom materials that have been defined
        custom_materials = list(MaterialConfigs._custom_configs.keys())
        return built_in_materials + custom_materials
    
    @staticmethod
    def get_config(material_name):
        """Get configuration for specified material."""
        built_in_configs = {
            'Kidney Stones (COM)': MaterialConfigs.get_kidney_stones_config(),
            'Quartz': MaterialConfigs.get_quartz_config(),
            'Feldspar': MaterialConfigs.get_feldspar_config(),
            'Calcite': MaterialConfigs.get_calcite_config()
        }
        
        # Check built-in configs first
        if material_name in built_in_configs:
            return built_in_configs[material_name]
        
        # Check custom configs
        if material_name in MaterialConfigs._custom_configs:
            return MaterialConfigs._custom_configs[material_name]
        
        # Return default config if not found
        return MaterialConfigs.get_kidney_stones_config()
    
    @staticmethod
    def add_custom_material(material_name, config):
        """Add a custom material configuration."""
        MaterialConfigs._custom_configs[material_name] = config
    
    @staticmethod
    def get_custom_template():
        """Get a template for creating custom material configurations."""
        return {
            'name': 'Custom Material',
            'characteristic_peaks': {
                'main': 1000,      # Primary peak (cm-1)
                'secondary': 500,  # Secondary peak (cm-1)
                'tertiary': 1500   # Tertiary peak (cm-1)
            },
            'reference_regions': {
                'baseline': (200, 400),        # Baseline region
                'fingerprint': (400, 800),     # Fingerprint region  
                'high_freq': (800, 1200)       # High frequency region
            },
            'densities': {
                'crystalline': 2.5,    # Pure crystalline density (g/cm³)
                'matrix': 2.0,         # Mixed matrix density (g/cm³)
                'low_density': 1.5     # Low density/amorphous (g/cm³)
            },
            'density_ranges': {
                'low_range': (1.4, 1.6),        # Low density range
                'medium_range': (1.8, 2.2),     # Medium density range
                'mixed_range': (2.0, 2.4),      # Mixed range
                'crystalline_range': (2.4, 2.6) # Crystalline range
            },
            'reference_intensity': 800,   # Reference intensity for normalization
            'classification_thresholds': {
                'low': 0.3,       # Low crystallinity threshold
                'medium': 0.6,    # Medium crystallinity threshold
                'high': 0.85      # High crystallinity threshold
            }
        }


class RamanDensityAnalyzer:
    """
    Flexible quantitative density analysis for Raman spectroscopy data.
    Supports multiple material types with configurable parameters.
    """
    
    def __init__(self, material_type='Kidney Stones (COM)', custom_config=None):
        """
        Initialize density analyzer with specified material configuration.
        
        Parameters:
        -----------
        material_type : str
            Type of material to analyze ('Kidney Stones (COM)', 'Quartz', 'Feldspar', 'Calcite', 'Other (Custom)')
        custom_config : dict, optional
            Custom material configuration (only used when material_type is 'Other (Custom)')
        """
        self.material_type = material_type
        
        # Handle custom configuration
        if material_type == 'Other (Custom)' and custom_config is not None:
            self.config = custom_config
        else:
            self.config = MaterialConfigs.get_config(material_type)
        
        # Set up material-specific parameters
        self.characteristic_peaks = self.config['characteristic_peaks']
        self.reference_regions = self.config['reference_regions']
        self.densities = self.config['densities']
        self.density_ranges = self.config['density_ranges']
        self.reference_intensity = self.config['reference_intensity']
        self.classification_thresholds = self.config['classification_thresholds']
        
        # Legacy attribute names for backward compatibility (kidney stones)
        if material_type == 'Kidney Stones (COM)':
            self.com_peaks = self.characteristic_peaks
            self.organic_regions = self.reference_regions
            self.biofilm_densities = self.density_ranges
    
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
        Flexible for different material types.
        
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
        # Find main characteristic peak intensity
        main_peak = self.characteristic_peaks['main']
        peak_idx = np.argmin(np.abs(wavenumber - main_peak))
        
        # Ensure valid peak index
        if 0 <= peak_idx < len(intensity):
            peak_intensity = intensity[peak_idx]
        else:
            # Fallback to maximum intensity if peak not found
            peak_intensity = np.nanmax(intensity) if len(intensity) > 0 else 0.0
        
        # Ensure valid peak intensity
        if np.isnan(peak_intensity) or np.isinf(peak_intensity):
            peak_intensity = 0.0
        
        # Calculate baseline intensity in reference region
        baseline_region = list(self.reference_regions.values())[0]  # Use first reference region as baseline
        baseline_mask = (wavenumber >= baseline_region[0]) & (wavenumber <= baseline_region[1])
        
        # Ensure we have valid baseline data
        if np.any(baseline_mask):
            baseline_data = intensity[baseline_mask]
            if len(baseline_data) > 0 and not np.all(np.isnan(baseline_data)):
                baseline_intensity = np.nanmean(baseline_data)  # Use nanmean to handle any NaN values
            else:
                baseline_intensity = np.nanmin(intensity) if len(intensity) > 0 else 0.0
        else:
            # If no baseline region found, use minimum intensity
            baseline_intensity = np.nanmin(intensity) if len(intensity) > 0 else 0.0
        
        # Ensure valid baseline
        if np.isnan(baseline_intensity) or np.isinf(baseline_intensity):
            baseline_intensity = 0.0
        
        # Calculate peak-to-baseline ratio
        peak_height = peak_intensity - baseline_intensity
        peak_height = max(peak_height, 0)  # Ensure non-negative
        
        # Normalize using material-specific reference intensity with safe division
        if self.reference_intensity > 0 and not np.isnan(self.reference_intensity):
            cdi = min(peak_height / self.reference_intensity, 1.0)
        else:
            # Fallback: use a normalized intensity ratio
            max_intensity = np.max(intensity) if len(intensity) > 0 else 1.0
            if max_intensity > 0:
                cdi = min(peak_height / max_intensity, 1.0)
            else:
                cdi = 0.0
        
        # Ensure valid CDI value
        if np.isnan(cdi) or np.isinf(cdi) or cdi < 0:
            cdi = 0.0
        
        # Additional metrics for validation
        metrics = {
            'main_peak_height': peak_height,
            'baseline_intensity': baseline_intensity,
            'peak_width': self._calculate_peak_width(wavenumber, intensity, peak_idx),
            'spectral_contrast': (peak_intensity - baseline_intensity) / (peak_intensity + baseline_intensity + 1e-10),
            'material_type': self.material_type,
            'main_peak_position': main_peak
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
    
    def calculate_density_by_type(self, cdi, density_type='mixed'):
        """
        Convert CDI to apparent density using material-specific density ranges.
        
        Parameters:
        -----------
        cdi : float or array
            Crystalline Density Index values
        density_type : str
            Type of density analysis ('low', 'medium', 'mixed', 'crystalline')
            
        Returns:
        --------
        apparent_density : float or array
            Calculated apparent density (g/cm³)
        """
        # Get density range based on type
        if density_type == 'low':
            min_density, max_density = self.density_ranges['low_range']
        elif density_type == 'medium':
            min_density, max_density = self.density_ranges['medium_range']
        elif density_type == 'crystalline':
            min_density, max_density = self.density_ranges['crystalline_range']
        else:  # mixed (default)
            min_density, max_density = self.density_ranges['mixed_range']
        
        # Linear interpolation between baseline and crystalline
        apparent_density = min_density + (self.densities['crystalline'] - min_density) * cdi
        
        return apparent_density
    
    def calculate_biofilm_density(self, cdi, biofilm_type='mixed'):
        """
        Legacy method for backward compatibility (kidney stones).
        """
        if self.material_type == 'Kidney Stones (COM)':
            return self.calculate_density_by_type(cdi, biofilm_type)
        else:
            return self.calculate_density_by_type(cdi, 'mixed')
    
    def calculate_apparent_density(self, cdi):
        """
        Convert CDI to apparent density using linear mixing model.
        
        Parameters:
        -----------
        cdi : float or array
            Crystalline Density Index values
            
        Returns:
        --------
        apparent_density : float or array
            Calculated apparent density (g/cm³)
        """
        # Linear mixing model between low-density matrix and crystalline phase
        apparent_density = (self.densities['matrix'] + 
                          (self.densities['crystalline'] - self.densities['matrix']) * cdi)
        
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
        Classify layers based on CDI thresholds using material-specific configuration.
        """
        if thresholds is None:
            thresholds = self.classification_thresholds
        
        classifications = []
        for cdi in cdi_profile:
            if cdi < thresholds['low']:
                if self.material_type == 'Kidney Stones (COM)':
                    classifications.append('bacterial')
                else:
                    classifications.append('low')
            elif cdi < thresholds['medium']:
                classifications.append('organic') 
            elif cdi < thresholds['high']:
                classifications.append('mixed_crystalline')
            else:
                classifications.append('crystalline')
                
        return classifications
    
    def plot_density_analysis(self, density_profile, title=None):
        """
        Create comprehensive density analysis visualization.
        """
        # Set default title based on material type
        if title is None:
            title = f"{self.material_type} Density Profile Analysis"
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        positions = density_profile['positions']
        
        # 1. CDI profile
        ax1.plot(positions, density_profile['cdi_profile'], 'b-', linewidth=2)
        mid_threshold = (self.classification_thresholds['low'] + self.classification_thresholds['high']) / 2
        ax1.axhline(y=mid_threshold, color='r', linestyle='--', alpha=0.7, label='Classification threshold')
        ax1.set_xlabel('Position (μm)')
        ax1.set_ylabel('Crystalline Density Index')
        ax1.set_title('CDI Spatial Profile')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Apparent density profile
        ax2.plot(positions, density_profile['density_profile'], 'g-', linewidth=2)
        ax2.axhline(y=self.densities['crystalline'], color='b', linestyle=':', 
                   alpha=0.7, label=f'Pure {self.material_type.split()[0]} density')
        ax2.axhline(y=self.densities['matrix'], color='r', linestyle=':', 
                   alpha=0.7, label='Matrix density')
        
        # Add material-specific density reference ranges
        if self.material_type == 'Kidney Stones (COM)':
            # Special handling for kidney stones (biofilm ranges)
            ax2.axhspan(self.density_ranges['low_range'][0], 
                       self.density_ranges['low_range'][1], 
                       alpha=0.2, color='cyan', label='Bacterial range')
            ax2.axhspan(self.density_ranges['medium_range'][0], 
                       self.density_ranges['medium_range'][1], 
                       alpha=0.2, color='orange', label='Bacteria-rich range')
        else:
            # Generic material ranges
            ax2.axhspan(self.density_ranges['low_range'][0], 
                       self.density_ranges['low_range'][1], 
                       alpha=0.2, color='lightblue', label='Low density range')
            ax2.axhspan(self.density_ranges['crystalline_range'][0], 
                       self.density_ranges['crystalline_range'][1], 
                       alpha=0.2, color='lightgreen', label='Crystalline range')
                   
        ax2.set_xlabel('Position (μm)')
        ax2.set_ylabel('Apparent Density (g/cm³)')
        ax2.set_title(f'Calculated Density Profile ({self.material_type})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Layer classification
        if self.material_type == 'Kidney Stones (COM)':
            # Biofilm-specific colors
            color_map = {
                'bacterial': 'cyan',
                'organic': 'orange', 
                'mixed_crystalline': 'purple',
                'crystalline': 'blue'
            }
            legend_labels = ['Bacterial', 'Organic', 'Mixed Crystalline', 'Crystalline']
        else:
            # Generic material colors
            color_map = {
                'low': 'lightblue',
                'organic': 'orange', 
                'mixed_crystalline': 'purple',
                'crystalline': 'darkgreen'
            }
            legend_labels = ['Low Density', 'Matrix', 'Mixed Crystalline', 'Crystalline']
            
        colors = [color_map.get(layer, 'gray') for layer in density_profile['layer_classification']]
        ax3.scatter(positions, density_profile['cdi_profile'], c=colors, alpha=0.7)
        ax3.set_xlabel('Position (μm)')
        ax3.set_ylabel('CDI')
        ax3.set_title(f'Layer Classification ({self.material_type})')
        ax3.grid(True, alpha=0.3)
        
        # Add legend for layer classification
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=list(color_map.values())[i], label=legend_labels[i])
                          for i in range(len(legend_labels))]
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

