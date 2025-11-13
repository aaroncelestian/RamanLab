"""
Microplastic Detection Module for 2D Raman Map Analysis

This module provides specialized tools for detecting weak microplastic signals
in noisy Raman spectroscopy data, optimized for large-area scans with fast
acquisition times.
"""

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
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
        
    def load_plastic_references(self, database: Dict, chemical_family: str = 'Plastics'):
        """
        Load plastic reference spectra from database.
        
        Args:
            database: RamanLab database dictionary
            chemical_family: Chemical family to filter (default: 'Plastics')
        """
        logger.info(f"Loading plastic references from database (family: {chemical_family})")
        
        self.database = database
        self.plastic_references = {}
        
        # Filter database for plastics
        for key, spectrum_data in database.items():
            metadata = spectrum_data.get('metadata', {})
            family = metadata.get('chemical_family', '')
            
            if family.lower() == chemical_family.lower():
                self.plastic_references[key] = spectrum_data
        
        logger.info(f"Loaded {len(self.plastic_references)} plastic reference spectra")
        return len(self.plastic_references)
    
    @staticmethod
    def baseline_als(intensities: np.ndarray, lam: float = 1e6, p: float = 0.001, 
                     niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline correction.
        
        Optimized for removing strong fluorescence backgrounds while preserving
        weak Raman peaks.
        
        Args:
            intensities: Raw intensity array
            lam: Smoothness parameter (higher = smoother baseline)
            p: Asymmetry parameter (lower = more aggressive removal)
            niter: Number of iterations
            
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
    
    def detect_peaks_at_positions(self, wavenumbers: np.ndarray, 
                                  intensities: np.ndarray,
                                  peak_positions: List[float],
                                  tolerance: float = 10.0) -> Dict[float, float]:
        """
        Detect peak intensities at specific wavenumber positions.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected)
            peak_positions: Expected peak positions (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            
        Returns:
            Dictionary mapping peak position to intensity
        """
        peak_intensities = {}
        
        for peak_pos in peak_positions:
            # Find wavenumbers within tolerance
            mask = np.abs(wavenumbers - peak_pos) <= tolerance
            if np.any(mask):
                # Get maximum intensity in this region
                peak_intensities[peak_pos] = np.max(intensities[mask])
            else:
                peak_intensities[peak_pos] = 0.0
        
        return peak_intensities
    
    def calculate_plastic_score(self, wavenumbers: np.ndarray,
                               intensities: np.ndarray,
                               plastic_type: str,
                               baseline_corrected: bool = False) -> float:
        """
        Calculate detection score for a specific plastic type.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            plastic_type: Plastic type code (e.g., 'PE', 'PP', 'PS')
            baseline_corrected: Whether intensities are already baseline-corrected
            
        Returns:
            Detection score (0-1, higher = more likely)
        """
        if plastic_type not in self.PLASTIC_SIGNATURES:
            logger.warning(f"Unknown plastic type: {plastic_type}")
            return 0.0
        
        # Baseline correction if needed
        if not baseline_corrected:
            intensities = self.baseline_als(intensities)
        
        # Get plastic signature
        signature = self.PLASTIC_SIGNATURES[plastic_type]
        strong_peaks = signature['strong_peaks']
        
        # Detect peaks
        peak_intensities = self.detect_peaks_at_positions(
            wavenumbers, intensities, strong_peaks
        )
        
        # Calculate score based on peak presence and intensity
        if len(peak_intensities) == 0:
            return 0.0
        
        # Normalize by number of expected peaks
        total_intensity = sum(peak_intensities.values())
        avg_intensity = total_intensity / len(strong_peaks)
        
        # Score is normalized average intensity (0-1 range)
        # Adjust threshold based on your data
        score = min(1.0, avg_intensity / 1000.0)  # Adjust denominator as needed
        
        return score
    
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
                                skip_baseline: bool = False) -> Dict[str, List[Tuple[int, float]]]:
        """Process a batch of spectra in parallel."""
        results = {ptype: [] for ptype in plastic_types}
        
        for idx in batch_indices:
            spectrum = intensity_map[idx]
            for ptype in plastic_types:
                score = self.calculate_plastic_score(
                    wavenumbers, spectrum, ptype, 
                    baseline_corrected=skip_baseline  # If skip_baseline=True, don't do it again
                )
                results[ptype].append((idx, score))
        
        return results
    
    def scan_map_for_plastics(self, wavenumbers: np.ndarray,
                             intensity_map: np.ndarray,
                             plastic_types: Optional[List[str]] = None,
                             threshold: float = 0.3,
                             progress_callback: Optional[Callable] = None,
                             n_jobs: int = -1,
                             fast_mode: bool = True) -> Dict[str, np.ndarray]:
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
            
        Returns:
            Dictionary mapping plastic type to 2D score map
        """
        if plastic_types is None:
            plastic_types = list(self.PLASTIC_SIGNATURES.keys())
        
        n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        logger.info(f"Scanning map for plastics: {plastic_types} using {n_cores} cores")
        
        # Get map dimensions
        if intensity_map.ndim == 2:
            # Flat array of spectra
            n_spectra = intensity_map.shape[0]
            score_maps = {ptype: np.zeros(n_spectra) for ptype in plastic_types}
            
            # Create batches for parallel processing
            batch_size = max(100, n_spectra // (n_cores * 10))
            batches = [list(range(i, min(i + batch_size, n_spectra))) 
                      for i in range(0, n_spectra, batch_size)]
            
            logger.info(f"Processing {n_spectra} spectra in {len(batches)} batches using {n_cores} cores")
            
            if progress_callback:
                progress_callback(0, len(batches), f"Starting parallel processing with {n_cores} cores...")
            
            # Pre-process: Apply fast baseline correction if in fast mode
            if fast_mode:
                if progress_callback:
                    progress_callback(0, len(batches), "Fast baseline removal...")
                # Simple baseline: subtract minimum rolling window
                from scipy.ndimage import minimum_filter1d
                for i in range(n_spectra):
                    baseline = minimum_filter1d(intensity_map[i], size=50, mode='nearest')
                    intensity_map[i] = intensity_map[i] - baseline
                    if i % 10000 == 0 and progress_callback:
                        prog = int((i / n_spectra) * 100)
                        progress_callback(i, n_spectra, f"Baseline removal: {prog}%")
            
            if progress_callback:
                progress_callback(0, len(batches), f"Detecting plastics in {len(batches)} batches...")
            
            # Process batches in parallel
            batch_results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                delayed(self._process_spectrum_batch)(
                    batch, wavenumbers, intensity_map, plastic_types, skip_baseline=fast_mode
                ) for batch in batches
            )
            
            # Combine results with progress updates
            for batch_idx, batch_result in enumerate(batch_results):
                if progress_callback:
                    progress_callback(batch_idx + 1, len(batches), 
                                    f"Processed {batch_idx + 1}/{len(batches)} batches")
                
                for ptype in plastic_types:
                    for idx, score in batch_result[ptype]:
                        score_maps[ptype][idx] = score
        
        else:
            # 3D map (x, y, wavenumbers) - flatten and process
            nx, ny, nw = intensity_map.shape
            n_spectra = nx * ny
            
            # Flatten to 2D for processing
            flat_map = intensity_map.reshape(n_spectra, nw)
            
            # Process using the 2D method
            flat_scores = self.scan_map_for_plastics(
                wavenumbers, flat_map, plastic_types, threshold, 
                progress_callback, n_jobs
            )
            
            # Reshape back to 3D
            score_maps = {ptype: scores.reshape(nx, ny) 
                         for ptype, scores in flat_scores.items()}
        
        logger.info("Map scanning complete")
        if progress_callback:
            progress_callback(100, 100, "✅ Scanning complete!")
        
        return score_maps
