"""
Data Processor for RamanLab Cluster Analysis

This module contains data processing and preprocessing methods.
"""

import numpy as np
from sklearn.decomposition import NMF
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


class DataProcessor:
    """Data processor for Raman spectroscopy clustering analysis."""
    
    def __init__(self):
        self.nmf_model = None
        self.corundum_reference = None
    
    def apply_corundum_drift_correction(self, intensities, wavenumbers):
        """Apply corundum-based drift correction to spectra."""
        try:
            # Find corundum peak region (around 418 cm^-1)
            corundum_region = (wavenumbers >= 400) & (wavenumbers <= 440)
            
            if not np.any(corundum_region):
                print("Warning: Corundum region not found in wavenumber range")
                return intensities
            
            corrected_intensities = intensities.copy()
            
            for i, spectrum in enumerate(intensities):
                # Extract corundum peak intensity
                corundum_intensity = np.max(spectrum[corundum_region])
                
                if corundum_intensity > 0:
                    # Normalize to corundum peak
                    correction_factor = 1000.0 / corundum_intensity  # Target intensity of 1000
                    corrected_intensities[i] = spectrum * correction_factor
            
            print(f"Applied corundum drift correction to {len(intensities)} spectra")
            return corrected_intensities
            
        except Exception as e:
            print(f"Error in corundum drift correction: {str(e)}")
            return intensities
    
    def apply_nmf_phase_separation(self, intensities, wavenumbers, n_components=3):
        """Apply NMF-based phase separation to isolate corundum and other phases."""
        try:
            print(f"Applying NMF phase separation with {n_components} components...")
            
            # Apply NMF
            self.nmf_model = NMF(n_components=n_components, 
                                init='random', 
                                random_state=42,
                                max_iter=500)
            
            # Fit NMF model
            W = self.nmf_model.fit_transform(intensities)  # Weights (samples x components)
            H = self.nmf_model.components_  # Components (components x features)
            
            # Identify corundum component (highest intensity in corundum region)
            corundum_region = (wavenumbers >= 400) & (wavenumbers <= 440)
            corundum_scores = []
            
            for i in range(n_components):
                component_spectrum = H[i]
                corundum_intensity = np.max(component_spectrum[corundum_region])
                corundum_scores.append(corundum_intensity)
            
            corundum_idx = np.argmax(corundum_scores)
            
            # Remove corundum contribution
            corrected_intensities = intensities.copy()
            
            for i, spectrum in enumerate(intensities):
                # Reconstruct spectrum without corundum component
                corundum_contribution = W[i, corundum_idx] * H[corundum_idx]
                corrected_intensities[i] = spectrum - corundum_contribution
                
                # Ensure non-negative values
                corrected_intensities[i] = np.maximum(corrected_intensities[i], 0)
            
            print(f"NMF separation completed. Corundum component: {corundum_idx}")
            return corrected_intensities, H, W, corundum_idx
            
        except Exception as e:
            print(f"Error in NMF phase separation: {str(e)}")
            return intensities, None, None, None
    
    def apply_carbon_soot_preprocessing(self, intensities, wavenumbers):
        """Apply carbon-specific preprocessing for soot and carbonaceous materials."""
        try:
            print("Applying carbon soot optimization preprocessing...")
            
            processed_intensities = intensities.copy()
            
            for i, spectrum in enumerate(intensities):
                # 1. Baseline correction for carbon spectra
                processed_spectrum = self._baseline_correction_als(spectrum, wavenumbers)
                
                # 2. Smooth the spectrum to reduce noise
                processed_spectrum = uniform_filter1d(processed_spectrum, size=3)
                
                # 3. Normalize to D-band intensity (around 1350 cm^-1)
                d_band_region = (wavenumbers >= 1300) & (wavenumbers <= 1400)
                g_band_region = (wavenumbers >= 1500) & (wavenumbers <= 1600)
                
                if np.any(d_band_region):
                    d_band_intensity = np.max(processed_spectrum[d_band_region])
                    if d_band_intensity > 0:
                        processed_spectrum = processed_spectrum / d_band_intensity
                
                processed_intensities[i] = processed_spectrum
            
            print(f"Carbon soot preprocessing applied to {len(intensities)} spectra")
            return processed_intensities
            
        except Exception as e:
            print(f"Error in carbon soot preprocessing: {str(e)}")
            return intensities
    
    def _baseline_correction_als(self, spectrum, wavenumbers, lam=1e5, p=0.01):
        """Apply asymmetric least squares baseline correction."""
        try:
            from scipy.sparse import diags
            from scipy.sparse.linalg import spsolve
            
            L = len(spectrum)
            D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            
            # Iterative baseline correction
            for _ in range(10):
                W = diags(w, 0, shape=(L, L))
                Z = W + lam * D.dot(D.transpose())
                baseline = spsolve(Z, w * spectrum)
                w = p * (spectrum > baseline) + (1-p) * (spectrum < baseline)
            
            corrected_spectrum = spectrum - baseline
            return np.maximum(corrected_spectrum, 0)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error in baseline correction: {str(e)}")
            return spectrum
    
    def extract_vibrational_features(self, intensities, wavenumbers, exclude_regions=None):
        """Extract vibrational features from spectra with optional region exclusion."""
        try:
            features = []
            
            for spectrum in intensities:
                # Create mask for included regions
                mask = np.ones(len(wavenumbers), dtype=bool)
                
                if exclude_regions:
                    for region in exclude_regions:
                        start, end = region
                        region_mask = (wavenumbers >= start) & (wavenumbers <= end)
                        mask = mask & ~region_mask
                
                # Extract features from included regions only
                masked_spectrum = spectrum[mask]
                masked_wavenumbers = wavenumbers[mask]
                
                # Basic normalization
                if np.max(masked_spectrum) > 0:
                    normalized_spectrum = masked_spectrum / np.max(masked_spectrum)
                else:
                    normalized_spectrum = masked_spectrum
                
                features.append(normalized_spectrum)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error in vibrational feature extraction: {str(e)}")
            # Fallback to basic normalization
            features = []
            for spectrum in intensities:
                if np.max(spectrum) > 0:
                    normalized = spectrum / np.max(spectrum)
                else:
                    normalized = spectrum
                features.append(normalized)
            return np.array(features)
    
    def extract_carbon_specific_features(self, intensities, wavenumbers):
        """Extract carbon-specific features for soot and carbonaceous materials."""
        try:
            features = []
            
            # Define carbon-specific regions
            d_band_region = (wavenumbers >= 1300) & (wavenumbers <= 1400)
            g_band_region = (wavenumbers >= 1500) & (wavenumbers <= 1600)
            d2d_band_region = (wavenumbers >= 2600) & (wavenumbers <= 2800)
            
            for spectrum in intensities:
                carbon_features = []
                
                # D-band features
                if np.any(d_band_region):
                    d_band = spectrum[d_band_region]
                    carbon_features.extend([
                        np.max(d_band),
                        np.mean(d_band),
                        np.std(d_band)
                    ])
                else:
                    carbon_features.extend([0, 0, 0])
                
                # G-band features
                if np.any(g_band_region):
                    g_band = spectrum[g_band_region]
                    carbon_features.extend([
                        np.max(g_band),
                        np.mean(g_band),
                        np.std(g_band)
                    ])
                else:
                    carbon_features.extend([0, 0, 0])
                
                # 2D-band features
                if np.any(d2d_band_region):
                    d2_band = spectrum[d2d_band_region]
                    carbon_features.extend([
                        np.max(d2_band),
                        np.mean(d2_band),
                        np.std(d2_band)
                    ])
                else:
                    carbon_features.extend([0, 0, 0])
                
                # D/G ratio
                if np.any(d_band_region) and np.any(g_band_region):
                    d_max = np.max(spectrum[d_band_region])
                    g_max = np.max(spectrum[g_band_region])
                    d_g_ratio = d_max / g_max if g_max > 0 else 0
                else:
                    d_g_ratio = 0
                
                carbon_features.append(d_g_ratio)
                
                features.append(carbon_features)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error in carbon feature extraction: {str(e)}")
            # Fallback to basic features
            return self.extract_vibrational_features(intensities, wavenumbers)
    
    def normalize_spectra(self, intensities, method='area'):
        """Normalize spectra using specified method."""
        try:
            normalized_intensities = intensities.copy()
            
            if method == 'area':
                # Normalize by area under curve
                for i, spectrum in enumerate(intensities):
                    area = np.trapz(spectrum)
                    if area > 0:
                        normalized_intensities[i] = spectrum / area
                        
            elif method == 'max':
                # Normalize by maximum intensity
                for i, spectrum in enumerate(intensities):
                    max_intensity = np.max(spectrum)
                    if max_intensity > 0:
                        normalized_intensities[i] = spectrum / max_intensity
                        
            elif method == 'vector':
                # Normalize by vector norm
                for i, spectrum in enumerate(intensities):
                    norm = np.linalg.norm(spectrum)
                    if norm > 0:
                        normalized_intensities[i] = spectrum / norm
            
            return normalized_intensities
            
        except Exception as e:
            print(f"Error in spectrum normalization: {str(e)}")
            return intensities
    
    def detect_peaks(self, spectrum, wavenumbers, height_threshold=None, prominence_threshold=None):
        """Detect peaks in a spectrum."""
        try:
            # Set default thresholds if not provided
            if height_threshold is None:
                height_threshold = np.max(spectrum) * 0.1
            
            if prominence_threshold is None:
                prominence_threshold = np.max(spectrum) * 0.05
            
            # Find peaks
            peaks, properties = find_peaks(spectrum, 
                                        height=height_threshold,
                                        prominence=prominence_threshold)
            
            # Get peak wavenumbers and intensities
            peak_wavenumbers = wavenumbers[peaks]
            peak_intensities = spectrum[peaks]
            
            return {
                'peak_indices': peaks,
                'peak_wavenumbers': peak_wavenumbers,
                'peak_intensities': peak_intensities,
                'properties': properties
            }
            
        except Exception as e:
            print(f"Error in peak detection: {str(e)}")
            return {
                'peak_indices': np.array([]),
                'peak_wavenumbers': np.array([]),
                'peak_intensities': np.array([]),
                'properties': {}
            }
