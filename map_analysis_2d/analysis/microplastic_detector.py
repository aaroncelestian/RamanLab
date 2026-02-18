"""
Microplastic Detection Module for 2D Raman Map Analysis

This module provides specialized tools for detecting weak microplastic signals
in noisy Raman spectroscopy data, optimized for large-area scans with fast
acquisition times.
"""

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from typing import Dict, List, Tuple, Optional, Callable
import logging
from joblib import Parallel, delayed
import multiprocessing

logger = logging.getLogger(__name__)


class MicroplasticDetector:
    """
    Detector for microplastics in Raman spectroscopy maps.
    
    Optimized for:
    - Weak signals in high fluorescence backgrounds
    - Fast acquisition times (noisy data)
    - Large area scans (10k+ spectra)
    - Multiple plastic types simultaneously
    """
    
    # Characteristic Raman peaks for common plastics (cm⁻¹)
    PLASTIC_SIGNATURES = {
        'PE': {  # Polyethylene
            'name': 'Polyethylene',
            'peaks': [1060, 1130, 1295, 1440, 2850, 2880],
            'strong_peaks': [1130, 1295, 1440],
            'color': '#FF6B6B'
        },
        'PP': {  # Polypropylene
            'name': 'Polypropylene',
            'peaks': [810, 840, 970, 1150, 1330, 1460, 2840, 2870, 2950],
            'strong_peaks': [840, 1150, 1330, 1460],
            'color': '#4ECDC4'
        },
        'PS': {  # Polystyrene
            'name': 'Polystyrene',
            'peaks': [620, 1000, 1030, 1200, 1450, 1600, 3050],
            'strong_peaks': [1000, 1030, 1600],
            'color': '#45B7D1'
        },
        'PET': {  # Polyethylene terephthalate
            'name': 'PET',
            'peaks': [630, 860, 1095, 1180, 1290, 1615, 1730],
            'strong_peaks': [1095, 1290, 1615, 1730],
            'color': '#96CEB4'
        },
        'PVC': {  # Polyvinyl chloride
            'name': 'PVC',
            'peaks': [630, 690, 1100, 1330, 1430, 2850, 2910],
            'strong_peaks': [630, 690, 1430],
            'color': '#FFEAA7'
        },
        'PMMA': {  # Polymethyl methacrylate (Acrylic)
            'name': 'PMMA',
            'peaks': [600, 810, 990, 1450, 1730, 2950],
            'strong_peaks': [810, 1450, 1730],
            'color': '#DFE6E9'
        },
        'PA': {  # Polyamide (Nylon)
            'name': 'Nylon',
            'peaks': [930, 1080, 1130, 1270, 1440, 1640, 2850, 2930],
            'strong_peaks': [1080, 1440, 1640],
            'color': '#A29BFE'
        }
    }
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize microplastic detector.
        
        Args:
            database_path: Path to Raman database for reference matching
        """
        self.database_path = database_path
        self.database = None
        self.plastic_references = {}
        
    @property
    def reference_spectra(self):
        """Get reference spectra dictionary."""
        return self.plastic_references
    
    def load_plastic_references(self, database: Dict, chemical_family: str = 'Plastic'):
        """
        Load plastic reference spectra from database.
        
        Args:
            database: RamanLab database dictionary
            chemical_family: Chemical family to filter (default: 'Plastics')
        
        Returns:
            Number of reference spectra loaded
        """
        logger.info(f"Loading plastic references from database (family: {chemical_family})")
        logger.info(f"Database has {len(database)} total entries")
        
        self.database = database
        self.plastic_references = {}
        
        # Use flexible keyword matching (matches main_window.py filtering logic)
        plastic_keywords = ['plastic', 'polymer', 'polyethylene', 'polypropylene', 
                           'polystyrene', 'pet', 'pvc', 'pmma', 'nylon']
        
        # Debug: Track what families we see
        families_seen = set()
        sample_entries = []
        
        # Filter database for plastics
        for key, spectrum_data in database.items():
            metadata = spectrum_data.get('metadata', {})
            
            # Check multiple possible keys for chemical family (matches main_window.py)
            family = ''
            family_keys = ['chemical_family', 'Chemical_Family', 'CHEMICAL_FAMILY',
                          'Chemical Family', 'CHEMICAL FAMILY', 'chemical family',
                          'family', 'Family', 'FAMILY']
            
            for family_key in family_keys:
                if family_key in metadata:
                    family = metadata[family_key]
                    if family:
                        break
            
            family = family.lower() if family else ''
            
            if family:
                families_seen.add(family)
            
            # Store first few entries for debugging
            if len(sample_entries) < 3:
                sample_entries.append(f"{key}: family='{family}'")
            
            # Check if family contains any plastic-related keywords
            is_plastic = any(keyword in family for keyword in plastic_keywords)
            
            if is_plastic:
                # Extract wavenumbers and intensities as numpy arrays
                wavenumbers = np.array(spectrum_data.get('wavenumbers', []))
                intensities = np.array(spectrum_data.get('intensities', []))
                
                # Store as tuple (wavenumbers, intensities) for correlation
                if len(wavenumbers) > 0 and len(intensities) > 0:
                    self.plastic_references[key] = (wavenumbers, intensities)
                    logger.debug(f"Loaded reference: {key} ({len(wavenumbers)} points)")
        
        # Debug output
        logger.info(f"Sample database entries: {sample_entries}")
        logger.info(f"Chemical families found: {sorted(families_seen)}")
        logger.info(f"Loaded {len(self.plastic_references)} plastic reference spectra")
        return len(self.plastic_references)
    
    @staticmethod
    def baseline_als(intensities: np.ndarray, lam: float = 1e6, p: float = 0.001, 
                     niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction.
        
        Optimized for removing strong fluorescence backgrounds while preserving
        weak Raman peaks. Uses the same ALS algorithm as the main app's Process tab.
        
        Args:
            intensities: Raw intensity array
            lam: Smoothness parameter (higher = smoother baseline)
                 Typical range: 10^4 to 10^8
            p: Asymmetry parameter (lower = more aggressive removal)
               Typical range: 0.001 to 0.1
            niter: Number of iterations (default: 10)
                   Typical range: 5 to 20
            
        Returns:
            Baseline-corrected intensities
        """
        L = len(intensities)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * intensities)
            w = p * (intensities > z) + (1 - p) * (intensities < z)
        
        return intensities - z
    
    @staticmethod
    def enhance_peaks(intensities: np.ndarray, window_length: int = 11, 
                      polyorder: int = 3) -> np.ndarray:
        """
        Enhance weak peaks using Savitzky-Golay smoothing.
        
        Args:
            intensities: Baseline-corrected intensities
            window_length: Filter window length (must be odd)
            polyorder: Polynomial order for fitting
            
        Returns:
            Smoothed intensities with enhanced peaks
        """
        if window_length % 2 == 0:
            window_length += 1
        
        return signal.savgol_filter(intensities, window_length, polyorder)
    
    @staticmethod
    def gaussian(x: np.ndarray, amplitude: float, center: float, 
                 sigma: float, offset: float = 0) -> np.ndarray:
        """Gaussian peak function."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset
    
    @staticmethod
    def lorentzian(x: np.ndarray, amplitude: float, center: float, 
                   gamma: float, offset: float = 0) -> np.ndarray:
        """Lorentzian peak function."""
        return amplitude * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2) + offset
    
    @staticmethod
    def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, 
                     width: float, eta: float = 0.5, offset: float = 0) -> np.ndarray:
        """Pseudo-Voigt peak function (mix of Gaussian and Lorentzian)."""
        sigma = width / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        lorentzian = (width / 2) ** 2 / ((x - center) ** 2 + (width / 2) ** 2)
        return amplitude * (eta * lorentzian + (1 - eta) * gaussian) + offset
    
    def estimate_local_noise(self, intensities: np.ndarray, 
                             window_size: int = 50) -> np.ndarray:
        """
        Estimate local noise level using rolling MAD.
        
        Args:
            intensities: Intensity array
            window_size: Size of rolling window
            
        Returns:
            Array of local noise estimates
        """
        # Use rolling median absolute deviation
        half_window = window_size // 2
        n = len(intensities)
        noise = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window)
            window = intensities[start:end]
            median = np.median(window)
            noise[i] = np.median(np.abs(window - median)) * 1.4826  # Scale to std
        
        return noise
    
    def fit_peak(self, wavenumbers: np.ndarray, intensities: np.ndarray,
                 center_guess: float, window: float = 20.0,
                 model: str = 'pseudo_voigt') -> Optional[Dict]:
        """
        Fit a peak at the specified position.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected)
            center_guess: Initial guess for peak center (cm⁻¹)
            window: Fitting window half-width (cm⁻¹)
            model: Peak model ('gaussian', 'lorentzian', 'pseudo_voigt')
            
        Returns:
            Dictionary with fit results or None if fit failed
        """
        # Extract fitting region
        mask = np.abs(wavenumbers - center_guess) <= window
        if np.sum(mask) < 5:
            return None
        
        x = wavenumbers[mask]
        y = intensities[mask]
        
        # Initial parameter estimates
        offset_guess = np.percentile(y, 10)  # Use low percentile as baseline
        y_corrected = y - offset_guess
        amplitude_guess = np.max(y_corrected)
        
        if amplitude_guess <= 0:
            return None
        
        # Find actual peak position in window
        peak_idx = np.argmax(y_corrected)
        center_guess_refined = x[peak_idx]
        
        # Estimate width from FWHM
        half_max = amplitude_guess / 2
        above_half = y_corrected > half_max
        if np.sum(above_half) > 1:
            width_guess = np.abs(x[above_half][-1] - x[above_half][0])
            width_guess = max(2.0, min(width_guess, window))  # Clamp to reasonable range
        else:
            width_guess = 5.0  # Default width
        
        try:
            if model == 'gaussian':
                popt, pcov = curve_fit(
                    self.gaussian, x, y,
                    p0=[amplitude_guess, center_guess_refined, width_guess / 2.355, offset_guess],
                    bounds=([0, x.min(), 0.5, -np.inf], 
                           [np.inf, x.max(), window, np.inf]),
                    maxfev=1000
                )
                fitted = self.gaussian(x, *popt)
                amplitude, center, sigma, offset = popt
                fwhm = 2.355 * sigma
                
            elif model == 'lorentzian':
                popt, pcov = curve_fit(
                    self.lorentzian, x, y,
                    p0=[amplitude_guess, center_guess_refined, width_guess / 2, offset_guess],
                    bounds=([0, x.min(), 0.5, -np.inf], 
                           [np.inf, x.max(), window, np.inf]),
                    maxfev=1000
                )
                fitted = self.lorentzian(x, *popt)
                amplitude, center, gamma, offset = popt
                fwhm = 2 * gamma
                
            else:  # pseudo_voigt
                popt, pcov = curve_fit(
                    self.pseudo_voigt, x, y,
                    p0=[amplitude_guess, center_guess_refined, width_guess, 0.5, offset_guess],
                    bounds=([0, x.min(), 1.0, 0, -np.inf], 
                           [np.inf, x.max(), window * 2, 1, np.inf]),
                    maxfev=1000
                )
                fitted = self.pseudo_voigt(x, *popt)
                amplitude, center, fwhm, eta, offset = popt
            
            # Calculate fit quality metrics
            residuals = y - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate SNR
            noise_estimate = np.std(residuals)
            snr = amplitude / noise_estimate if noise_estimate > 0 else 0
            
            return {
                'amplitude': amplitude,
                'center': center,
                'fwhm': fwhm,
                'offset': offset,
                'r_squared': r_squared,
                'snr': snr,
                'fitted_curve': fitted,
                'x': x,
                'y': y,
                'residuals': residuals,
                'model': model
            }
            
        except (RuntimeError, ValueError) as e:
            logger.debug(f"Peak fit failed at {center_guess}: {e}")
            return None
    
    def detect_peaks_fast(self, wavenumbers: np.ndarray,
                          intensities: np.ndarray,
                          peak_positions: List[float],
                          tolerance: float = 15.0,
                          min_snr: float = 3.0) -> Tuple[int, float]:
        """
        FAST peak detection optimized for large datasets.
        
        Checks for actual peak SHAPE (local maximum above surroundings),
        not just positive intensity.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected)
            peak_positions: Expected peak positions (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            min_snr: Minimum signal-to-noise ratio (default 3.0 for selectivity)
            
        Returns:
            Tuple of (peaks_detected_count, average_snr)
        """
        # Global noise estimate from spectrum variance
        noise_level = np.median(np.abs(np.diff(intensities))) * 1.4826
        if noise_level <= 0:
            noise_level = np.std(intensities) * 0.1
        if noise_level <= 0:
            return 0, 0.0  # Can't estimate noise
        
        # Also get baseline level (median of spectrum)
        baseline = np.median(intensities)
        
        peaks_found = 0
        total_snr = 0.0
        
        for peak_pos in peak_positions:
            # Find indices within tolerance
            mask = np.abs(wavenumbers - peak_pos) <= tolerance
            if np.sum(mask) < 3:  # Need at least 3 points
                continue
            
            # Get region data
            region = intensities[mask]
            region_wn = wavenumbers[mask]
            
            # Find local maximum
            max_idx = np.argmax(region)
            max_val = region[max_idx]
            
            # Check if it's actually a LOCAL maximum (peak shape)
            # Must be higher than edges of the region
            edge_val = (region[0] + region[-1]) / 2
            prominence = max_val - edge_val
            
            # Also check it's above baseline
            height_above_baseline = max_val - baseline
            
            # SNR based on prominence (peak height above local background)
            snr = prominence / noise_level if noise_level > 0 else 0
            
            # Must have:
            # 1. Positive prominence (actual peak shape)
            # 2. SNR above threshold
            # 3. Peak above baseline
            if prominence > 0 and snr >= min_snr and height_above_baseline > noise_level:
                peaks_found += 1
                total_snr += snr
        
        avg_snr = total_snr / peaks_found if peaks_found > 0 else 0
        return peaks_found, avg_snr
    
    def detect_peaks_derivative_fast(self, wavenumbers: np.ndarray,
                                      intensities: np.ndarray,
                                      peak_positions: List[float],
                                      tolerance: float = 15.0,
                                      min_snr: float = 4.0) -> Tuple[int, float]:
        """
        FAST derivative-based peak detection for weak signals.
        
        Second derivative is more sensitive to peak shapes and less affected
        by sloping baselines. Optimized for speed with no curve fitting.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            peak_positions: Expected peak positions (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            min_snr: Minimum SNR in derivative space (default 4.0)
            
        Returns:
            Tuple of (peaks_detected_count, average_derivative_snr)
        """
        # Smooth slightly and compute second derivative (vectorized)
        smoothed = uniform_filter1d(intensities, size=5)
        second_deriv = np.gradient(np.gradient(smoothed))
        second_deriv = -second_deriv  # Invert so peaks are positive
        
        # Noise in derivative space - use MAD of derivative
        deriv_noise = np.median(np.abs(second_deriv - np.median(second_deriv))) * 1.4826
        if deriv_noise <= 0:
            deriv_noise = np.std(second_deriv) * 0.1
        if deriv_noise <= 0:
            return 0, 0.0
        
        peaks_found = 0
        total_snr = 0.0
        
        for peak_pos in peak_positions:
            mask = np.abs(wavenumbers - peak_pos) <= tolerance
            if np.sum(mask) < 3:
                continue
            
            # Check derivative in region
            region_deriv = second_deriv[mask]
            max_deriv = np.max(region_deriv)
            
            # Must be a local maximum in derivative (peak center)
            max_idx = np.argmax(region_deriv)
            # Check it's not at the edge (would indicate slope, not peak)
            if max_idx == 0 or max_idx == len(region_deriv) - 1:
                continue
            
            # SNR in derivative space
            snr = max_deriv / deriv_noise if deriv_noise > 0 else 0
            
            # Higher threshold for derivative detection
            if snr >= min_snr and max_deriv > 0:
                peaks_found += 1
                total_snr += snr
        
        avg_snr = total_snr / peaks_found if peaks_found > 0 else 0
        return peaks_found, avg_snr
    
    def detect_peaks_at_positions(self, wavenumbers: np.ndarray, 
                                  intensities: np.ndarray,
                                  peak_positions: List[float],
                                  tolerance: float = 15.0,
                                  min_snr: float = 2.0,
                                  min_r_squared: float = 0.3,
                                  use_fitting: bool = True) -> Dict[float, Dict]:
        """
        Detect and validate peaks at specific wavenumber positions.
        
        Uses peak fitting to distinguish real peaks from noise, with adaptive
        thresholding based on local noise estimation.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected)
            peak_positions: Expected peak positions (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            min_snr: Minimum signal-to-noise ratio for valid peak
            min_r_squared: Minimum R² for fit quality
            use_fitting: If True, use peak fitting; if False, use simple detection
            
        Returns:
            Dictionary mapping peak position to detection info
        """
        detected_peaks = {}
        
        # Estimate local noise across the spectrum
        local_noise = self.estimate_local_noise(intensities)
        
        for peak_pos in peak_positions:
            # Find wavenumbers within tolerance
            mask = np.abs(wavenumbers - peak_pos) <= tolerance
            if not np.any(mask):
                continue
            
            if use_fitting:
                # Try to fit a peak at this position
                fit_result = self.fit_peak(
                    wavenumbers, intensities, peak_pos, 
                    window=tolerance, model='pseudo_voigt'
                )
                
                if fit_result is not None:
                    # Validate the fit
                    if (fit_result['snr'] >= min_snr and 
                        fit_result['r_squared'] >= min_r_squared and
                        fit_result['amplitude'] > 0 and
                        fit_result['fwhm'] >= 2.0 and fit_result['fwhm'] <= 50.0):
                        
                        detected_peaks[peak_pos] = {
                            'intensity': fit_result['amplitude'],
                            'center': fit_result['center'],
                            'fwhm': fit_result['fwhm'],
                            'snr': fit_result['snr'],
                            'r_squared': fit_result['r_squared'],
                            'fitted': True
                        }
            else:
                # Simple detection (fallback for speed)
                region_intensities = intensities[mask]
                region_noise = local_noise[mask]
                
                max_idx = np.argmax(region_intensities)
                max_intensity = region_intensities[max_idx]
                local_noise_level = region_noise[max_idx]
                
                # Calculate local SNR
                snr = max_intensity / local_noise_level if local_noise_level > 0 else 0
                
                if snr >= min_snr:
                    detected_peaks[peak_pos] = {
                        'intensity': max_intensity,
                        'center': wavenumbers[mask][max_idx],
                        'snr': snr,
                        'fitted': False
                    }
        
        return detected_peaks
    
    def detect_all_peaks(self, wavenumbers: np.ndarray, intensities: np.ndarray,
                         min_snr: float = 2.5, min_prominence: float = None,
                         min_width: float = 2.0, max_width: float = 50.0) -> List[Dict]:
        """
        Detect all significant peaks in a spectrum using derivative analysis.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected)
            min_snr: Minimum signal-to-noise ratio
            min_prominence: Minimum peak prominence (auto-calculated if None)
            min_width: Minimum peak width in cm⁻¹
            max_width: Maximum peak width in cm⁻¹
            
        Returns:
            List of detected peak dictionaries
        """
        # Smooth the spectrum slightly to reduce noise
        smoothed = gaussian_filter1d(intensities, sigma=1.5)
        
        # Estimate noise level
        noise_level = np.median(np.abs(np.diff(intensities))) / 0.6745
        
        if min_prominence is None:
            min_prominence = 2.5 * noise_level
        
        # Calculate wavenumber spacing
        wn_spacing = np.abs(np.median(np.diff(wavenumbers)))
        min_distance = int(min_width / wn_spacing)
        
        # Find peaks using scipy
        peaks, properties = signal.find_peaks(
            smoothed,
            prominence=min_prominence,
            distance=max(1, min_distance),
            width=(min_width / wn_spacing, max_width / wn_spacing)
        )
        
        detected = []
        for i, peak_idx in enumerate(peaks):
            peak_wn = wavenumbers[peak_idx]
            peak_intensity = intensities[peak_idx]
            
            # Calculate local SNR
            local_noise = self.estimate_local_noise(intensities, window_size=30)
            snr = peak_intensity / local_noise[peak_idx] if local_noise[peak_idx] > 0 else 0
            
            if snr >= min_snr:
                # Try to fit the peak for better characterization
                fit_result = self.fit_peak(wavenumbers, intensities, peak_wn, 
                                          window=15.0, model='pseudo_voigt')
                
                if fit_result is not None and fit_result['r_squared'] > 0.3:
                    detected.append({
                        'position': fit_result['center'],
                        'intensity': fit_result['amplitude'],
                        'fwhm': fit_result['fwhm'],
                        'snr': fit_result['snr'],
                        'r_squared': fit_result['r_squared'],
                        'prominence': properties['prominences'][i]
                    })
                else:
                    # Use raw peak data
                    detected.append({
                        'position': peak_wn,
                        'intensity': peak_intensity,
                        'snr': snr,
                        'prominence': properties['prominences'][i]
                    })
        
        return detected
    
    def detect_weak_peaks_derivative(self, wavenumbers: np.ndarray, 
                                      intensities: np.ndarray,
                                      peak_positions: List[float],
                                      tolerance: float = 15.0,
                                      smoothing_sigma: float = 2.0) -> Dict[float, Dict]:
        """
        Detect weak peaks using second derivative analysis.
        
        Second derivative is more sensitive to peak shapes and less affected
        by sloping baselines, making it ideal for weak peak detection.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected preferred)
            peak_positions: Expected peak positions (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            smoothing_sigma: Gaussian smoothing sigma for noise reduction
            
        Returns:
            Dictionary mapping peak position to detection info
        """
        detected_peaks = {}
        
        # Smooth the spectrum
        smoothed = gaussian_filter1d(intensities, sigma=smoothing_sigma)
        
        # Calculate second derivative (negative peaks indicate positive peaks in original)
        # Use Savitzky-Golay for better derivative estimation
        wn_spacing = np.abs(np.median(np.diff(wavenumbers)))
        window = max(5, int(10 / wn_spacing))  # ~10 cm⁻¹ window
        if window % 2 == 0:
            window += 1
        
        try:
            second_deriv = signal.savgol_filter(smoothed, window, 3, deriv=2)
        except ValueError:
            # Fallback to simple second derivative
            second_deriv = np.gradient(np.gradient(smoothed))
        
        # Invert so peaks become positive
        second_deriv = -second_deriv
        
        # Estimate noise in second derivative
        deriv_noise = np.median(np.abs(second_deriv - np.median(second_deriv))) * 1.4826
        
        for peak_pos in peak_positions:
            mask = np.abs(wavenumbers - peak_pos) <= tolerance
            if not np.any(mask):
                continue
            
            region_deriv = second_deriv[mask]
            region_wn = wavenumbers[mask]
            region_int = intensities[mask]
            
            # Find maximum in second derivative (indicates peak center)
            max_idx = np.argmax(region_deriv)
            max_deriv = region_deriv[max_idx]
            
            # Calculate SNR in derivative space
            deriv_snr = max_deriv / deriv_noise if deriv_noise > 0 else 0
            
            # Also check original intensity
            intensity_at_peak = region_int[max_idx]
            
            # A peak is detected if derivative SNR is significant
            if deriv_snr >= 2.0:  # Lower threshold for weak peaks
                detected_peaks[peak_pos] = {
                    'center': region_wn[max_idx],
                    'intensity': intensity_at_peak,
                    'deriv_snr': deriv_snr,
                    'deriv_value': max_deriv,
                    'method': 'derivative'
                }
        
        return detected_peaks
    
    def matched_filter_detection(self, wavenumbers: np.ndarray,
                                  intensities: np.ndarray,
                                  peak_positions: List[float],
                                  expected_width: float = 10.0,
                                  tolerance: float = 15.0) -> Dict[float, Dict]:
        """
        Detect peaks using matched filtering with expected peak shape.
        
        Matched filtering is optimal for detecting known signals in noise.
        Uses a Gaussian template matched to expected Raman peak width.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected preferred)
            peak_positions: Expected peak positions (cm⁻¹)
            expected_width: Expected FWHM of peaks (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            
        Returns:
            Dictionary mapping peak position to detection info
        """
        detected_peaks = {}
        
        # Create matched filter template (Gaussian peak)
        wn_spacing = np.abs(np.median(np.diff(wavenumbers)))
        template_width = int(expected_width * 3 / wn_spacing)  # 3 sigma width
        if template_width % 2 == 0:
            template_width += 1
        template_width = max(5, template_width)
        
        # Create normalized Gaussian template
        sigma = expected_width / (2.355 * wn_spacing)  # Convert FWHM to sigma in samples
        template_x = np.arange(template_width) - template_width // 2
        template = np.exp(-0.5 * (template_x / sigma) ** 2)
        template = template / np.sqrt(np.sum(template ** 2))  # Normalize
        
        # Apply matched filter (cross-correlation)
        filtered = signal.correlate(intensities, template, mode='same')
        
        # Estimate noise in filtered signal
        noise_level = np.median(np.abs(filtered - np.median(filtered))) * 1.4826
        
        for peak_pos in peak_positions:
            mask = np.abs(wavenumbers - peak_pos) <= tolerance
            if not np.any(mask):
                continue
            
            region_filtered = filtered[mask]
            region_wn = wavenumbers[mask]
            region_int = intensities[mask]
            
            # Find maximum in filtered signal
            max_idx = np.argmax(region_filtered)
            max_filtered = region_filtered[max_idx]
            
            # Calculate SNR
            filter_snr = max_filtered / noise_level if noise_level > 0 else 0
            
            if filter_snr >= 2.5:  # Threshold for matched filter detection
                detected_peaks[peak_pos] = {
                    'center': region_wn[max_idx],
                    'intensity': region_int[max_idx],
                    'filter_snr': filter_snr,
                    'filter_response': max_filtered,
                    'method': 'matched_filter'
                }
        
        return detected_peaks
    
    def detect_peaks_ensemble(self, wavenumbers: np.ndarray,
                              intensities: np.ndarray,
                              peak_positions: List[float],
                              tolerance: float = 15.0,
                              min_methods: int = 2) -> Dict[float, Dict]:
        """
        Detect peaks using ensemble of methods for robust weak signal detection.
        
        Combines peak fitting, derivative analysis, and matched filtering.
        A peak is confirmed if detected by multiple methods.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected)
            peak_positions: Expected peak positions (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            min_methods: Minimum number of methods that must detect the peak
            
        Returns:
            Dictionary mapping peak position to combined detection info
        """
        # Run all detection methods
        fitting_results = self.detect_peaks_at_positions(
            wavenumbers, intensities, peak_positions,
            tolerance=tolerance, min_snr=1.5, min_r_squared=0.2, use_fitting=True
        )
        
        derivative_results = self.detect_weak_peaks_derivative(
            wavenumbers, intensities, peak_positions,
            tolerance=tolerance, smoothing_sigma=2.0
        )
        
        matched_results = self.matched_filter_detection(
            wavenumbers, intensities, peak_positions,
            expected_width=10.0, tolerance=tolerance
        )
        
        # Combine results
        combined = {}
        for peak_pos in peak_positions:
            methods_detected = 0
            combined_info = {'position': peak_pos, 'methods': []}
            
            if peak_pos in fitting_results:
                methods_detected += 1
                combined_info['methods'].append('fitting')
                combined_info['fitting'] = fitting_results[peak_pos]
                combined_info['intensity'] = fitting_results[peak_pos].get('intensity', 0)
                combined_info['snr'] = fitting_results[peak_pos].get('snr', 0)
            
            if peak_pos in derivative_results:
                methods_detected += 1
                combined_info['methods'].append('derivative')
                combined_info['derivative'] = derivative_results[peak_pos]
                if 'intensity' not in combined_info:
                    combined_info['intensity'] = derivative_results[peak_pos].get('intensity', 0)
                combined_info['deriv_snr'] = derivative_results[peak_pos].get('deriv_snr', 0)
            
            if peak_pos in matched_results:
                methods_detected += 1
                combined_info['methods'].append('matched_filter')
                combined_info['matched_filter'] = matched_results[peak_pos]
                if 'intensity' not in combined_info:
                    combined_info['intensity'] = matched_results[peak_pos].get('intensity', 0)
                combined_info['filter_snr'] = matched_results[peak_pos].get('filter_snr', 0)
            
            combined_info['methods_count'] = methods_detected
            
            # Only include if detected by minimum number of methods
            if methods_detected >= min_methods:
                combined[peak_pos] = combined_info
            elif methods_detected == 1 and min_methods <= 1:
                # For very weak signals, accept single method with high confidence
                if 'snr' in combined_info and combined_info['snr'] >= 3.0:
                    combined[peak_pos] = combined_info
                elif 'deriv_snr' in combined_info and combined_info['deriv_snr'] >= 4.0:
                    combined[peak_pos] = combined_info
                elif 'filter_snr' in combined_info and combined_info['filter_snr'] >= 4.0:
                    combined[peak_pos] = combined_info
        
        return combined
    
    def calculate_multi_window_score(self, wavenumbers: np.ndarray,
                                    intensities: np.ndarray,
                                    ref_wavenumbers: np.ndarray,
                                    ref_intensities: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate correlation score using multiple diagnostic frequency windows.
        EXACT COPY of proven algorithm from raman_analysis_app_qt6.py
        
        Args:
            wavenumbers: Query wavenumber array
            intensities: Query intensity array (baseline-corrected)
            ref_wavenumbers: Reference wavenumber array
            ref_intensities: Reference intensity array
            
        Returns:
            Tuple of (overall_score, window_scores_dict)
        """
        try:
            # Convert to numpy arrays with explicit float conversion
            ref_wavenumbers = np.array(ref_wavenumbers, dtype=float)
            ref_intensities = np.array(ref_intensities, dtype=float)
            
            # Check for empty or invalid data
            if len(ref_wavenumbers) == 0 or len(ref_intensities) == 0:
                return 0.0, {}
            
            # Define diagnostic windows for plastics (C-H, C-C, C=C vibrations)
            windows = [
                {"name": "C-H stretch", "range": (2800, 3100), "weight": 0.25},
                {"name": "C=C stretch", "range": (1600, 1700), "weight": 0.20},
                {"name": "C-H bend", "range": (1400, 1500), "weight": 0.20},
                {"name": "C-C stretch", "range": (1000, 1200), "weight": 0.20},
                {"name": "Fingerprint", "range": (600, 900), "weight": 0.15}
            ]
            
            total_weighted_score = 0.0
            total_weight = 0.0
            window_scores = {}
            
            for window in windows:
                window_start, window_end = window["range"]
                
                # Find overlapping region
                query_mask = (wavenumbers >= window_start) & (wavenumbers <= window_end)
                db_mask = (ref_wavenumbers >= window_start) & (ref_wavenumbers <= window_end)
                
                if not np.any(query_mask) or not np.any(db_mask):
                    window_scores[window["name"]] = 0.0
                    continue
                
                # Calculate actual overlap
                overlap_start = max(window_start, 
                                  max(wavenumbers[query_mask].min(), ref_wavenumbers[db_mask].min()))
                overlap_end = min(window_end,
                                min(wavenumbers[query_mask].max(), ref_wavenumbers[db_mask].max()))
                
                if overlap_end <= overlap_start:
                    window_scores[window["name"]] = 0.0
                    continue  # No meaningful overlap
                
                # Create common wavenumber grid for this window
                n_points = min(50, np.sum(query_mask), np.sum(db_mask))  # Adaptive resolution
                if n_points < 5:
                    window_scores[window["name"]] = 0.0
                    continue  # Too few points for reliable correlation
                
                common_wavenumbers = np.linspace(overlap_start, overlap_end, n_points)
                
                # Interpolate both spectra to common grid
                query_interp = np.interp(common_wavenumbers, wavenumbers, intensities)
                db_interp = np.interp(common_wavenumbers, ref_wavenumbers, ref_intensities)
                
                # Check for zero variance
                query_std = np.std(query_interp)
                db_std = np.std(db_interp)
                
                if query_std == 0 or db_std == 0:
                    window_scores[window["name"]] = 0.0
                    continue  # Skip windows with no variation
                
                # Normalize to zero mean, unit variance
                query_norm = (query_interp - np.mean(query_interp)) / query_std
                db_norm = (db_interp - np.mean(db_interp)) / db_std
                
                # Calculate correlation for this window
                correlation = np.corrcoef(query_norm, db_norm)[0, 1]
                window_similarity = abs(correlation)  # Use absolute correlation
                
                window_scores[window["name"]] = window_similarity
                # Add to weighted sum
                total_weighted_score += window_similarity * window["weight"]
                total_weight += window["weight"]
            
            if total_weight == 0:
                return 0.0, window_scores  # No valid windows
            
            # Return weighted average
            final_score = total_weighted_score / total_weight
            return max(0, min(1, final_score)), window_scores
            
        except Exception as e:
            logger.error(f"Error in multi-window score calculation: {e}")
            return 0.0, {}

    def calculate_plastic_score(self, wavenumbers: np.ndarray, 
                               intensities: np.ndarray,
                               plastic_type: str,
                               baseline_corrected: bool = False,
                               lam: float = 1e6,
                               p: float = 0.001,
                               niter: int = 10,
                               use_peak_fitting: bool = False,
                               min_snr: float = 1.5,
                               min_r_squared: float = 0.25,
                               use_ensemble: bool = False,
                               smoothing: int = 5) -> Tuple[float, Dict[str, float], Optional[str]]:
        """
        Calculate detection score for a specific plastic type.
        
        Uses a combination of reference correlation and peak fitting to detect
        weak plastic signals in noisy spectra.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            plastic_type: Type of plastic to detect
            baseline_corrected: If True, skip baseline correction
            lam: ALS lambda parameter (smoothness)
            p: ALS p parameter (asymmetry)
            niter: ALS iterations
            use_peak_fitting: If True, use peak fitting for validation
            min_snr: Minimum SNR for peak detection (lower = more sensitive)
            min_r_squared: Minimum R² for peak fit quality
            use_ensemble: If True, use ensemble detection (fitting + derivative + matched filter)
            smoothing: Smoothing window size (0 = no smoothing, 5-11 recommended)
            
        Returns:
            Tuple of (detection_score, window_scores_dict, matched_spectrum_name)
        """
        if plastic_type not in self.PLASTIC_SIGNATURES:
            return 0.0, {}, None
        
        # Apply baseline correction if needed
        if not baseline_corrected:
            intensities = self.baseline_als(intensities, lam=lam, p=p, niter=niter)
        
        # Apply smoothing to reduce noise (Savitzky-Golay filter preserves peak shapes)
        if smoothing > 0:
            window = smoothing if smoothing % 2 == 1 else smoothing + 1  # Must be odd
            window = max(5, min(window, 21))  # Clamp to reasonable range
            if len(intensities) > window:
                intensities = signal.savgol_filter(intensities, window, 3)
        
        # Use multi-window correlation if reference available
        # Try to find a matching reference spectrum by plastic type
        ref_spectrum = None
        matched_name = None
        plastic_name = self.PLASTIC_SIGNATURES[plastic_type]['name'].lower()
        
        # First try exact match with plastic_type code
        if plastic_type in self.reference_spectra:
            ref_spectrum = self.reference_spectra[plastic_type]
            matched_name = plastic_type
        else:
            # Try to find by matching plastic name in spectrum key
            for key, spectrum in self.reference_spectra.items():
                if plastic_name in key.lower() or plastic_type.lower() in key.lower():
                    ref_spectrum = spectrum
                    matched_name = key
                    logger.debug(f"Matched {plastic_type} to reference: {key}")
                    break
        
        # Calculate reference correlation score if available
        ref_score = 0.0
        window_scores = {}
        if ref_spectrum is not None:
            ref_wn, ref_int = ref_spectrum
            ref_score, window_scores = self.calculate_multi_window_score(
                wavenumbers, intensities, ref_wn, ref_int
            )
        
        # Calculate peak-based score
        expected_peaks = self.PLASTIC_SIGNATURES[plastic_type]['strong_peaks']
        all_peaks = self.PLASTIC_SIGNATURES[plastic_type]['peaks']
        
        if use_ensemble or use_peak_fitting:
            # SLOW methods - only for small datasets or validation
            if use_ensemble:
                detected_strong = self.detect_peaks_ensemble(
                    wavenumbers, intensities, expected_peaks,
                    tolerance=15, min_methods=1
                )
                detected_all = self.detect_peaks_ensemble(
                    wavenumbers, intensities, all_peaks,
                    tolerance=15, min_methods=1
                )
            else:
                detected_strong = self.detect_peaks_at_positions(
                    wavenumbers, intensities, expected_peaks, 
                    tolerance=15, min_snr=min_snr, min_r_squared=min_r_squared,
                    use_fitting=True
                )
                detected_all = self.detect_peaks_at_positions(
                    wavenumbers, intensities, all_peaks, 
                    tolerance=15, min_snr=min_snr, min_r_squared=min_r_squared,
                    use_fitting=True
                )
            strong_peak_count = len(detected_strong)
            all_peak_count = len(detected_all)
            avg_snr = np.mean([p.get('snr', 0) for p in detected_strong.values()]) if detected_strong else 0
        else:
            # FAST mode - use vectorized detection for large datasets
            # Combine intensity-based and derivative-based detection
            strong_count_int, avg_snr_int = self.detect_peaks_fast(
                wavenumbers, intensities, expected_peaks, 
                tolerance=15, min_snr=min_snr
            )
            all_count_int, _ = self.detect_peaks_fast(
                wavenumbers, intensities, all_peaks, 
                tolerance=15, min_snr=min_snr
            )
            
            # Also check with derivative (more sensitive to weak peaks)
            strong_count_deriv, avg_snr_deriv = self.detect_peaks_derivative_fast(
                wavenumbers, intensities, expected_peaks, tolerance=15
            )
            all_count_deriv, _ = self.detect_peaks_derivative_fast(
                wavenumbers, intensities, all_peaks, tolerance=15
            )
            
            # Take the better of the two methods
            strong_peak_count = max(strong_count_int, strong_count_deriv)
            all_peak_count = max(all_count_int, all_count_deriv)
            avg_snr = max(avg_snr_int, avg_snr_deriv)
        
        # Calculate peak-based scores
        strong_peak_ratio = strong_peak_count / len(expected_peaks) if expected_peaks else 0
        all_peak_ratio = all_peak_count / len(all_peaks) if all_peaks else 0
        
        # CRITICAL: Require minimum number of peaks for a valid detection
        # Single peak matches are likely false positives
        min_strong_peaks = 2  # Must have at least 2 strong peaks
        min_all_peaks = 3     # Must have at least 3 total peaks
        
        if strong_peak_count < min_strong_peaks and all_peak_count < min_all_peaks:
            # Not enough peaks - return zero score
            window_scores['strong_peaks_detected'] = strong_peak_count
            window_scores['strong_peaks_expected'] = len(expected_peaks)
            window_scores['all_peaks_detected'] = all_peak_count
            window_scores['all_peaks_expected'] = len(all_peaks)
            window_scores['avg_snr'] = avg_snr
            window_scores['peak_score'] = 0.0
            window_scores['ref_score'] = ref_score
            window_scores['rejection_reason'] = 'insufficient_peaks'
            return 0.0, window_scores, matched_name
        
        # Combine scores with weights
        # - Reference correlation is most reliable when available
        # - Strong peak detection is important for identification
        # - All peaks ratio provides additional confirmation
        # - Average SNR indicates signal quality
        
        if ref_spectrum is not None:
            # Have reference: weight correlation heavily
            peak_score = (
                0.5 * strong_peak_ratio +  # Strong peaks detected
                0.3 * all_peak_ratio +      # All peaks detected
                0.2 * min(1.0, avg_snr / 10.0)  # SNR quality (normalized to 10)
            )
            # Combine reference and peak scores
            final_score = 0.5 * ref_score + 0.5 * peak_score
        else:
            # No reference: rely on peak detection
            peak_score = (
                0.6 * strong_peak_ratio +   # Strong peaks detected
                0.3 * all_peak_ratio +     # All peaks detected
                0.1 * min(1.0, avg_snr / 10.0)  # SNR quality
            )
            final_score = peak_score
        
        # Add peak detection info to window_scores for diagnostics
        window_scores['strong_peaks_detected'] = strong_peak_count
        window_scores['strong_peaks_expected'] = len(expected_peaks)
        window_scores['all_peaks_detected'] = all_peak_count
        window_scores['all_peaks_expected'] = len(all_peaks)
        window_scores['avg_snr'] = avg_snr
        window_scores['peak_score'] = peak_score
        window_scores['ref_score'] = ref_score
        
        return final_score, window_scores, matched_name
    
    def correlate_with_reference(self, wavenumbers: np.ndarray,
                                 intensities: np.ndarray,
                                 reference_key: str,
                                 baseline_corrected: bool = False) -> float:
        """
        Calculate correlation with a reference spectrum from database.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            reference_key: Database key for reference spectrum
            baseline_corrected: Whether intensities are already baseline-corrected
            
        Returns:
            Correlation coefficient (0-1)
        """
        if self.plastic_references is None or reference_key not in self.plastic_references:
            return 0.0
        
        # Baseline correction if needed
        if not baseline_corrected:
            intensities = self.baseline_als(intensities)
        
        # Get reference spectrum
        ref_data = self.plastic_references[reference_key]
        ref_wavenumbers = ref_data.get('wavenumbers', np.array([]))
        ref_intensities = ref_data.get('intensities', np.array([]))
        
        # Interpolate to common wavenumber grid
        common_wn = wavenumbers
        ref_interp = np.interp(common_wn, ref_wavenumbers, ref_intensities)
        
        # Normalize both spectra
        intensities_norm = (intensities - np.mean(intensities)) / (np.std(intensities) + 1e-10)
        ref_norm = (ref_interp - np.mean(ref_interp)) / (np.std(ref_interp) + 1e-10)
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(intensities_norm, ref_norm)[0, 1]
        
        return max(0.0, correlation)  # Return 0 for negative correlations
    
    def _process_spectrum_batch(self, batch_indices: List[int], 
                                wavenumbers: np.ndarray,
                                intensity_map: np.ndarray,
                                plastic_types: List[str],
                                skip_baseline: bool = False,
                                lam: float = 1e6,
                                p: float = 0.001,
                                niter: int = 10,
                                use_peak_fitting: bool = False,
                                min_snr: float = 1.5,
                                min_r_squared: float = 0.25,
                                use_ensemble: bool = False,
                                smoothing: int = 5) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """Process a batch of spectra in parallel."""
        results = {ptype: [] for ptype in plastic_types}
        
        for idx in batch_indices:
            spectrum = intensity_map[idx]
            for ptype in plastic_types:
                score, window_scores, matched_name = self.calculate_plastic_score(
                    wavenumbers, spectrum, ptype, 
                    baseline_corrected=skip_baseline,  # If skip_baseline=True, don't do it again
                    lam=lam, p=p, niter=niter,
                    use_peak_fitting=use_peak_fitting,
                    min_snr=min_snr,
                    min_r_squared=min_r_squared,
                    use_ensemble=use_ensemble,
                    smoothing=smoothing
                )
                results[ptype].append((idx, score, window_scores, matched_name))
        
        return results
    
    def scan_map_for_plastics(self, wavenumbers: np.ndarray,
                             intensity_map: np.ndarray,
                             plastic_types: Optional[List[str]] = None,
                             threshold: float = 0.3,
                             progress_callback: Optional[Callable] = None,
                             n_jobs: int = -1,
                             fast_mode: bool = True,
                             lam: float = 1e6,
                             p: float = 0.001,
                             niter: int = 10,
                             use_peak_fitting: bool = False,
                             min_snr: float = 1.5,
                             min_r_squared: float = 0.25,
                             use_ensemble: bool = False,
                             smoothing: int = 5) -> Dict[str, np.ndarray]:
        """
        Scan entire 2D map for multiple plastic types using parallel processing.
        
        Args:
            wavenumbers: Wavenumber array
            intensity_map: 2D array (n_spectra, n_wavenumbers) or 3D array (x, y, wavenumbers)
            plastic_types: List of plastic types to detect (None = all)
            threshold: Minimum score threshold for detection
            progress_callback: Optional callback function(current, total, message)
            n_jobs: Number of parallel jobs (-1 = all cores)
            fast_mode: If True, uses simple baseline removal instead of ALS (much faster)
            lam: ALS lambda parameter (smoothness) - used when fast_mode=False
            p: ALS p parameter (asymmetry) - used when fast_mode=False
            niter: ALS iterations - used when fast_mode=False
            use_peak_fitting: If True, use curve fitting (SLOW - only for small datasets)
            min_snr: Minimum SNR for peak detection (lower = more sensitive)
            min_r_squared: Minimum R² for peak fit quality
            use_ensemble: If True, use ensemble detection (VERY SLOW - only for validation)
            smoothing: Smoothing window size (0=none, 5-11 recommended for noisy data)
            
        Returns:
            Dictionary mapping plastic type to 2D score map
        """
        if plastic_types is None:
            plastic_types = list(self.PLASTIC_SIGNATURES.keys())
        
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        logger.info(f"Scanning map for plastics: {plastic_types} using {n_cores} cores")
        
        # Import threading for progress tracking
        from threading import Lock
        
        # Get map dimensions
        if intensity_map.ndim == 2:
            # Flat array of spectra
            n_spectra = intensity_map.shape[0]
            score_maps = {ptype: np.zeros(n_spectra) for ptype in plastic_types}
            match_details = {}  # Store match info for each position
            
            # Create batches for parallel processing
            batch_size = max(100, n_spectra // (n_cores * 10))
            batches = [list(range(i, min(i + batch_size, n_spectra))) 
                      for i in range(0, n_spectra, batch_size)]
            
            logger.info(f"Processing {n_spectra} spectra in {len(batches)} batches using {n_cores} cores")
            
            if progress_callback:
                progress_callback(0, len(batches), f"Starting parallel processing with {n_cores} cores...")
            
            # Pre-process: Apply baseline correction if in fast mode
            if fast_mode:
                # Determine which baseline method to use based on parameters
                # If lam is very high and p is very low, use fast rolling ball
                # Otherwise use ALS
                use_rolling_ball = (lam >= 1e6 and p <= 0.001 and niter == 10)
                
                if use_rolling_ball:
                    # Fast rolling ball baseline (original fast method)
                    if progress_callback:
                        progress_callback(0, n_spectra, "Fast baseline removal (Rolling Ball)...")
                    
                    from scipy.ndimage import minimum_filter1d
                    
                    def remove_baseline_batch(indices):
                        """Remove baseline from a batch of spectra using rolling ball."""
                        for i in indices:
                            baseline = minimum_filter1d(intensity_map[i], size=100, mode='nearest')
                            intensity_map[i] = intensity_map[i] - baseline
                else:
                    # ALS baseline removal
                    if progress_callback:
                        progress_callback(0, n_spectra, f"ALS baseline removal (λ={lam:.0e}, p={p:.3f})...")
                    
                    def remove_baseline_batch(indices):
                        """Remove baseline from a batch of spectra using ALS."""
                        for i in indices:
                            # Apply ALS baseline correction
                            intensity_map[i] = MicroplasticDetector.baseline_als(
                                intensity_map[i], lam=lam, p=p, niter=niter
                            )
                
                # Create batches for baseline removal
                baseline_batch_size = max(100, n_spectra // (n_cores * 4))
                baseline_batches = [list(range(i, min(i + baseline_batch_size, n_spectra))) 
                                  for i in range(0, n_spectra, baseline_batch_size)]
                
                logger.info(f"Removing baseline from {n_spectra} spectra in {len(baseline_batches)} batches")
                
                # Initial progress message
                if progress_callback:
                    progress_callback(0, len(baseline_batches), 
                                    f"Starting ALS baseline on {n_spectra:,} spectra with {n_cores} cores...")
                
                # Process baseline removal in parallel
                completed_baseline = [0]
                baseline_lock = Lock()
                
                def baseline_with_progress(batch):
                    remove_baseline_batch(batch)
                    with baseline_lock:
                        completed_baseline[0] += 1
                        # Update every 5% or every 10 batches, whichever is more frequent
                        update_interval = max(1, min(10, len(baseline_batches) // 20))
                        if progress_callback and completed_baseline[0] % update_interval == 0:
                            prog = int((completed_baseline[0] / len(baseline_batches)) * 100)
                            spectra_done = completed_baseline[0] * baseline_batch_size
                            progress_callback(completed_baseline[0], len(baseline_batches), 
                                            f"ALS baseline: {prog}% ({spectra_done:,}/{n_spectra:,} spectra)")
                
                Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                    delayed(baseline_with_progress)(batch) for batch in baseline_batches
                )
                
                if progress_callback:
                    progress_callback(len(baseline_batches), len(baseline_batches), 
                                    "Baseline removal complete!")
            
            if progress_callback:
                progress_callback(0, len(batches), f"Detecting plastics in {len(batches)} batches using {n_cores} cores...")
            
            # Process batches in parallel using multiprocessing
            # Note: Progress updates not available during parallel execution
            batch_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(self._process_spectrum_batch)(
                    batch, wavenumbers, intensity_map, plastic_types, 
                    fast_mode, lam, p, niter,
                    use_peak_fitting, min_snr, min_r_squared, use_ensemble, smoothing
                ) for batch in batches
            )
            
            # Combine results
            if progress_callback:
                progress_callback(len(batches), len(batches), "Combining results...")
            
            for batch_result in batch_results:
                for ptype in plastic_types:
                    for idx, score, window_scores, matched_name in batch_result[ptype]:
                        score_maps[ptype][idx] = score
                        
                        # Store match details if score is above threshold and we have a match
                        if score >= threshold and matched_name is not None:
                            # Only store if this is the best match for this position
                            if idx not in match_details or score > match_details[idx]['score']:
                                match_details[idx] = {
                                    'plastic_type': ptype,
                                    'score': score,
                                    'matched_name': matched_name,
                                    'window_scores': window_scores
                                }
        
        else:
            # 3D map (x, y, wavenumbers) - flatten and process
            nx, ny, nw = intensity_map.shape
            n_spectra = nx * ny
            
            # Flatten to 2D for processing
            flat_map = intensity_map.reshape(n_spectra, nw)
            
            # Process using the 2D method
            flat_scores = self.scan_map_for_plastics(
                wavenumbers, flat_map, plastic_types, threshold, 
                progress_callback, n_jobs, fast_mode, lam, p, niter,
                use_peak_fitting, min_snr, min_r_squared, use_ensemble, smoothing
            )
            
            # Reshape back to 3D
            score_maps = {ptype: scores.reshape(nx, ny) 
                         for ptype, scores in flat_scores.items()}
        
        logger.info("Map scanning complete")
        if progress_callback:
            progress_callback(100, 100, "✅ Scanning complete!")
        
        # Add match details to results
        score_maps['_match_details'] = match_details
        
        return score_maps
    
    # ==================== TEMPLATE-BASED DETECTION ====================
    
    def load_plastic_templates_from_database(self, database: Dict, 
                                             plastic_types: Optional[List[str]] = None,
                                             max_per_type: int = 5) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]]:
        """
        Load curated plastic reference spectra from database for template matching.
        
        Args:
            database: RamanLab database dictionary
            plastic_types: List of plastic types to load (e.g., ['Polyethylene', 'Polypropylene'])
                          If None, loads common microplastic types
            max_per_type: Maximum number of reference spectra per plastic type
        
        Returns:
            Dictionary mapping plastic type to list of (wavenumbers, intensities, name) tuples
        """
        if plastic_types is None:
            plastic_types = [
                'Polyethylene', 'Polypropylene', 'Polystyrene', 'Polyester',
                'Polyethylene Terephthalate', 'Polyamide', 'Polycarbonate',
                'Polyurethane', 'Acrylic', 'PVC'
            ]
        
        logger.info(f"Loading plastic templates from database with {len(database)} entries")
        logger.info(f"Looking for plastic types: {plastic_types}")
        
        self.plastic_templates = {}
        
        for ptype in plastic_types:
            self.plastic_templates[ptype] = []
            ptype_lower = ptype.lower()
            
            # Find matching entries in database
            for key, spectrum_data in database.items():
                # Skip if we already have enough for this type
                if len(self.plastic_templates[ptype]) >= max_per_type:
                    break
                
                # Check metadata first (more reliable than key name)
                metadata = spectrum_data.get('metadata', {})
                
                # Check multiple possible keys for chemical family (matches main_window.py)
                chemical_family = ''
                family_keys = ['chemical_family', 'Chemical_Family', 'CHEMICAL_FAMILY',
                              'Chemical Family', 'CHEMICAL FAMILY', 'chemical family',
                              'family', 'Family', 'FAMILY']
                
                for family_key in family_keys:
                    if family_key in metadata:
                        chemical_family = metadata[family_key]
                        if chemical_family:
                            break
                
                chemical_family = chemical_family.lower() if chemical_family else ''
                mineral_name = metadata.get('mineral_name', '').lower()
                
                # Check if this is a plastic using flexible keyword matching
                # (matches the filtering logic in main_window.py)
                plastic_keywords = ['plastic', 'polymer', 'polyethylene', 'polypropylene', 
                                   'polystyrene', 'pet', 'pvc', 'pmma', 'nylon']
                is_plastic = any(keyword in chemical_family for keyword in plastic_keywords)
                
                matches_type = (ptype_lower in mineral_name or 
                               ptype_lower in key.lower() or
                               ptype_lower in chemical_family)
                
                if is_plastic and matches_type:
                    # Extract spectrum data
                    wavenumbers = np.array(spectrum_data.get('wavenumbers', []))
                    intensities = np.array(spectrum_data.get('intensities', []))
                    
                    if len(wavenumbers) > 100 and len(intensities) > 100:
                        # Normalize intensities
                        intensities = intensities - np.min(intensities)
                        if np.max(intensities) > 0:
                            intensities = intensities / np.max(intensities)
                        
                        self.plastic_templates[ptype].append((wavenumbers, intensities, key))
                        logger.debug(f"Loaded template: {key} for {ptype}")
            
            logger.info(f"Loaded {len(self.plastic_templates[ptype])} templates for {ptype}")
        
        # Remove empty types
        self.plastic_templates = {k: v for k, v in self.plastic_templates.items() if v}
        
        total = sum(len(v) for v in self.plastic_templates.values())
        logger.info(f"Total plastic templates loaded: {total}")
        
        return self.plastic_templates
    
    def template_match_spectrum(self, wavenumbers: np.ndarray, 
                                intensities: np.ndarray,
                                baseline_correct: bool = True,
                                smooth: bool = True,
                                smoothing_window: int = 7) -> Dict[str, Dict]:
        """
        Match a spectrum against all loaded plastic templates.
        
        Uses non-negative least squares fitting to find the best template match.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            baseline_correct: Apply baseline correction before matching
            smooth: Apply smoothing before matching
            smoothing_window: Savitzky-Golay smoothing window
        
        Returns:
            Dictionary with match results for each plastic type:
            {
                'Polyethylene': {
                    'best_match': 'Polyethylene 1. Blue Sphere',
                    'correlation': 0.85,
                    'residual': 0.15,
                    'coefficient': 1.2
                },
                ...
            }
        """
        if not hasattr(self, 'plastic_templates') or not self.plastic_templates:
            logger.warning("No plastic templates loaded. Call load_plastic_templates_from_database first.")
            return {}
        
        # Preprocess spectrum
        if baseline_correct:
            intensities = self.baseline_als(intensities)
        
        if smooth and len(intensities) > smoothing_window:
            window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
            intensities = signal.savgol_filter(intensities, window, 3)
        
        # Normalize
        intensities = intensities - np.min(intensities)
        if np.max(intensities) > 0:
            intensities = intensities / np.max(intensities)
        
        results = {}
        
        for ptype, templates in self.plastic_templates.items():
            best_corr = -1
            best_match = None
            best_residual = float('inf')
            best_coeff = 0
            
            for template_wn, template_int, template_name in templates:
                # Interpolate template to match spectrum wavenumbers
                # Find overlapping range
                wn_min = max(wavenumbers.min(), template_wn.min())
                wn_max = min(wavenumbers.max(), template_wn.max())
                
                if wn_max <= wn_min:
                    continue
                
                # Create common wavenumber grid
                mask = (wavenumbers >= wn_min) & (wavenumbers <= wn_max)
                common_wn = wavenumbers[mask]
                
                if len(common_wn) < 50:
                    continue
                
                # Interpolate both to common grid
                spec_interp = np.interp(common_wn, wavenumbers, intensities)
                template_interp = np.interp(common_wn, template_wn, template_int)
                
                # Use multiple metrics for robust matching with STRINGENT requirements
                
                # 1. Spectral Angle Mapper (SAM) - measures angle between vectors
                # More robust than correlation for spectral matching
                dot_product = np.dot(spec_interp, template_interp)
                norm_spec = np.linalg.norm(spec_interp)
                norm_template = np.linalg.norm(template_interp)
                
                if norm_spec > 1e-10 and norm_template > 1e-10:
                    cos_angle = dot_product / (norm_spec * norm_template)
                    cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
                    sam_score = cos_angle  # 1 = perfect match, 0 = orthogonal, -1 = opposite
                else:
                    sam_score = 0
                
                # 2. Normalized cross-correlation (more stringent than Pearson)
                spec_centered = spec_interp - np.mean(spec_interp)
                template_centered = template_interp - np.mean(template_interp)
                
                if np.std(spec_centered) > 1e-10 and np.std(template_centered) > 1e-10:
                    ncc = np.sum(spec_centered * template_centered) / (
                        np.sqrt(np.sum(spec_centered**2)) * np.sqrt(np.sum(template_centered**2))
                    )
                    ncc = max(0, ncc)  # Only positive correlations
                else:
                    ncc = 0
                
                # 3. Peak position matching - do the peaks align?
                # Find peaks in both spectra
                from scipy.signal import find_peaks
                spec_peaks, _ = find_peaks(spec_interp, prominence=0.1 * np.max(spec_interp))
                template_peaks, _ = find_peaks(template_interp, prominence=0.1 * np.max(template_interp))
                
                # Calculate peak overlap score
                if len(spec_peaks) > 0 and len(template_peaks) > 0:
                    # For each template peak, find closest spectrum peak
                    peak_distances = []
                    for tp in template_peaks:
                        if len(spec_peaks) > 0:
                            closest_dist = np.min(np.abs(spec_peaks - tp))
                            # Normalize by wavenumber range
                            peak_distances.append(closest_dist / len(common_wn))
                    
                    # Peak score: 1 if peaks align perfectly, 0 if far apart
                    avg_peak_dist = np.mean(peak_distances) if peak_distances else 1.0
                    peak_score = np.exp(-20 * avg_peak_dist)  # Strict penalty for misaligned peaks
                else:
                    peak_score = 0.0  # No peaks found
                
                # 4. Residual penalty - how well does template fit?
                if np.sum(template_interp**2) > 0:
                    scale = np.dot(spec_interp, template_interp) / np.sum(template_interp**2)
                    scale = max(0, scale)
                    fitted = scale * template_interp
                    residual_norm = np.sqrt(np.mean((spec_interp - fitted)**2))
                    # Convert to similarity score (0 = bad fit, 1 = perfect fit)
                    # Increased penalty from -5 to -10 for stricter matching
                    residual_score = np.exp(-10 * residual_norm)
                else:
                    residual_score = 0
                
                # Combined score with STRICT requirements:
                # SAM (30%) + NCC (25%) + Peak matching (25%) + Residual fit (20%)
                # All metrics must be reasonably high for a good match
                corr = 0.30 * max(0, sam_score) + 0.25 * ncc + 0.25 * peak_score + 0.20 * residual_score
                
                # Additional penalty: if any individual metric is too low, reduce overall score
                # This prevents cases where one high metric compensates for very low others
                min_metric = min(max(0, sam_score), ncc, peak_score, residual_score)
                if min_metric < 0.3:  # If any metric is below 0.3, apply penalty
                    corr *= (min_metric / 0.3)  # Scale down the score
                
                # Calculate residual (how well template fits)
                if np.sum(template_interp**2) > 0:
                    coeff = np.sum(spec_interp * template_interp) / np.sum(template_interp**2)
                    coeff = max(0, coeff)  # Non-negative
                    fitted = coeff * template_interp
                    residual = np.sqrt(np.mean((spec_interp - fitted)**2))
                else:
                    coeff = 0
                    residual = float('inf')
                
                if corr > best_corr:
                    best_corr = corr
                    best_match = template_name
                    best_residual = residual
                    best_coeff = coeff
            
            if best_match is not None:
                results[ptype] = {
                    'best_match': best_match,
                    'correlation': best_corr,
                    'residual': best_residual,
                    'coefficient': best_coeff
                }
        
        return results
    
    def scan_map_with_templates(self, wavenumbers: np.ndarray,
                                intensity_map: np.ndarray,
                                database: Dict,
                                plastic_types: Optional[List[str]] = None,
                                threshold: float = 0.5,
                                progress_callback: Optional[Callable] = None,
                                n_jobs: int = -1,
                                max_templates_per_type: int = 3) -> Dict[str, np.ndarray]:
        """
        Scan entire map using template matching against database plastic spectra.
        
        This is faster and more accurate than peak-based detection for noisy data.
        
        Args:
            wavenumbers: Wavenumber array
            intensity_map: 2D array (n_spectra, n_wavenumbers)
            database: RamanLab database dictionary
            plastic_types: List of plastic types to detect (None = common types)
            threshold: Minimum correlation threshold for detection
            progress_callback: Optional callback function(current, total, message)
            n_jobs: Number of parallel jobs (-1 = all cores)
            max_templates_per_type: Max reference spectra per plastic type
        
        Returns:
            Dictionary mapping plastic type to 2D correlation score map
        """
        # Load templates if not already loaded
        if not hasattr(self, 'plastic_templates') or not self.plastic_templates:
            if progress_callback:
                progress_callback(0, 100, "Loading plastic templates from database...")
            self.load_plastic_templates_from_database(database, plastic_types, max_templates_per_type)
        
        if not self.plastic_templates:
            logger.error("No plastic templates available")
            return {}
        
        n_spectra = intensity_map.shape[0]
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        
        logger.info(f"Template matching {n_spectra:,} spectra against {sum(len(v) for v in self.plastic_templates.values())} templates using {n_cores} cores")
        logger.info(f"Applying ALS baseline correction to each spectrum before matching")
        
        if progress_callback:
            progress_callback(0, 100, f"Template matching {n_spectra:,} spectra (with baseline correction)...")
        
        # Initialize score maps
        score_maps = {ptype: np.zeros(n_spectra) for ptype in self.plastic_templates.keys()}
        match_info = {}
        
        # Process in batches for parallel execution
        batch_size = max(100, n_spectra // (n_cores * 4))
        batches = [list(range(i, min(i + batch_size, n_spectra))) 
                   for i in range(0, n_spectra, batch_size)]
        
        def process_batch(batch_indices):
            """Process a batch of spectra."""
            batch_results = []
            for idx in batch_indices:
                spectrum = intensity_map[idx]
                matches = self.template_match_spectrum(wavenumbers, spectrum)
                batch_results.append((idx, matches))
            return batch_results
        
        # Run parallel processing
        all_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
            delayed(process_batch)(batch) for batch in batches
        )
        
        if progress_callback:
            progress_callback(80, 100, "Combining results...")
        
        # Combine results
        for batch_results in all_results:
            for idx, matches in batch_results:
                for ptype, match_data in matches.items():
                    corr = match_data['correlation']
                    score_maps[ptype][idx] = max(0, corr)  # Use correlation as score
                    
                    if corr >= threshold:
                        if idx not in match_info or corr > match_info[idx].get('correlation', 0):
                            match_info[idx] = {
                                'plastic_type': ptype,
                                'best_match': match_data['best_match'],
                                'correlation': corr,
                                'residual': match_data['residual']
                            }
        
        # Add match info to results
        score_maps['_match_info'] = match_info
        
        # Calculate detection statistics
        for ptype in self.plastic_templates.keys():
            detected = np.sum(score_maps[ptype] >= threshold)
            pct = 100 * detected / n_spectra
            logger.info(f"{ptype}: {detected:,} detections ({pct:.2f}%)")
        
        if progress_callback:
            progress_callback(100, 100, "✅ Template matching complete!")
        
        return score_maps
