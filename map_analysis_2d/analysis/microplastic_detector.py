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
        
    @property
    def reference_spectra(self):
        """Get reference spectra dictionary."""
        return self.plastic_references
    
    def load_plastic_references(self, database: Dict, chemical_family: str = 'Plastics'):
        """
        Load plastic reference spectra from database.
        
        Args:
            database: RamanLab database dictionary
            chemical_family: Chemical family to filter (default: 'Plastics')
        
        Returns:
            Number of reference spectra loaded
        """
        logger.info(f"Loading plastic references from database (family: {chemical_family})")
        
        self.database = database
        self.plastic_references = {}
        
        # Filter database for plastics
        for key, spectrum_data in database.items():
            metadata = spectrum_data.get('metadata', {})
            family = metadata.get('chemical_family', '')
            
            if family.lower() == chemical_family.lower():
                # Extract wavenumbers and intensities as numpy arrays
                wavenumbers = np.array(spectrum_data.get('wavenumbers', []))
                intensities = np.array(spectrum_data.get('intensities', []))
                
                # Store as tuple (wavenumbers, intensities) for correlation
                if len(wavenumbers) > 0 and len(intensities) > 0:
                    self.plastic_references[key] = (wavenumbers, intensities)
                    logger.debug(f"Loaded reference: {key} ({len(wavenumbers)} points)")
        
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
    
    def detect_peaks_at_positions(self, wavenumbers: np.ndarray, 
                                  intensities: np.ndarray,
                                  peak_positions: List[float],
                                  tolerance: float = 10.0,
                                  min_intensity: float = 50.0) -> Dict[float, float]:
        """
        Detect peak intensities at specific wavenumber positions.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array (baseline-corrected)
            peak_positions: Expected peak positions (cm⁻¹)
            tolerance: Wavenumber tolerance (±cm⁻¹)
            min_intensity: Minimum intensity to count as a peak
            
        Returns:
            Dictionary mapping peak position to intensity (only significant peaks)
        """
        peak_intensities = {}
        
        # Calculate noise level as median absolute deviation
        median_intensity = np.median(intensities)
        mad = np.median(np.abs(intensities - median_intensity))
        noise_threshold = median_intensity + 3 * mad  # 3-sigma threshold
        
        # Use the higher of min_intensity or noise threshold
        threshold = max(min_intensity, noise_threshold)
        
        for peak_pos in peak_positions:
            # Find wavenumbers within tolerance
            mask = np.abs(wavenumbers - peak_pos) <= tolerance
            if np.any(mask):
                # Get maximum intensity in this region
                max_intensity = np.max(intensities[mask])
                # Only count if significantly above threshold
                if max_intensity > threshold:
                    peak_intensities[peak_pos] = max_intensity
        
        return peak_intensities
    
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
                               niter: int = 10) -> Tuple[float, Dict[str, float]]:
        """
        Calculate detection score for a specific plastic type.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            plastic_type: Type of plastic to detect
            baseline_corrected: If True, skip baseline correction
            lam: ALS lambda parameter (smoothness)
            p: ALS p parameter (asymmetry)
            niter: ALS iterations
            
        Returns:
            Tuple of (detection_score, window_scores_dict)
        """
        if plastic_type not in self.PLASTIC_SIGNATURES:
            return 0.0, {}
        
        # Apply baseline correction if needed
        if not baseline_corrected:
            intensities = self.baseline_als(intensities, lam=lam, p=p, niter=niter)
        
        # Use multi-window correlation if reference available
        # Try to find a matching reference spectrum by plastic type
        ref_spectrum = None
        plastic_name = self.PLASTIC_SIGNATURES[plastic_type]['name'].lower()
        
        # First try exact match with plastic_type code
        if plastic_type in self.reference_spectra:
            ref_spectrum = self.reference_spectra[plastic_type]
        else:
            # Try to find by matching plastic name in spectrum key
            for key, spectrum in self.reference_spectra.items():
                if plastic_name in key.lower() or plastic_type.lower() in key.lower():
                    ref_spectrum = spectrum
                    logger.debug(f"Matched {plastic_type} to reference: {key}")
                    break
        
        if ref_spectrum is not None:
            ref_wn, ref_int = ref_spectrum
            score, window_scores = self.calculate_multi_window_score(
                wavenumbers, intensities, ref_wn, ref_int
            )
            return score, window_scores
        
        # Fallback: Peak detection at expected positions
        expected_peaks = self.PLASTIC_SIGNATURES[plastic_type]['strong_peaks']
        detected_peaks = self.detect_peaks_at_positions(
            wavenumbers, intensities, expected_peaks, tolerance=10
        )
        peak_score = len(detected_peaks) / len(expected_peaks)
        
        return peak_score, {}
    
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
                                niter: int = 10) -> Dict[str, List[Tuple[int, float, Dict]]]:
        """Process a batch of spectra in parallel."""
        results = {ptype: [] for ptype in plastic_types}
        
        for idx in batch_indices:
            spectrum = intensity_map[idx]
            for ptype in plastic_types:
                score, window_scores = self.calculate_plastic_score(
                    wavenumbers, spectrum, ptype, 
                    baseline_corrected=skip_baseline,  # If skip_baseline=True, don't do it again
                    lam=lam, p=p, niter=niter
                )
                results[ptype].append((idx, score, window_scores))
        
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
                             niter: int = 10) -> Dict[str, np.ndarray]:
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
                progress_callback(0, len(batches), f"Detecting plastics in {len(batches)} batches...")
            
            # Process batches with progress tracking
            detection_lock = Lock()
            completed_batches = [0]  # Use list to allow modification in nested function
            
            def process_with_progress(batch):
                result = self._process_spectrum_batch(
                    batch, wavenumbers, intensity_map, plastic_types, 
                    skip_baseline=fast_mode, lam=lam, p=p, niter=niter
                )
                with detection_lock:
                    completed_batches[0] += 1
                    if progress_callback and completed_batches[0] % max(1, len(batches) // 20) == 0:
                        progress_callback(completed_batches[0], len(batches), 
                                        f"Processed {completed_batches[0]}/{len(batches)} batches")
                return result
            
            # Process batches in parallel with progress
            batch_results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                delayed(process_with_progress)(batch) for batch in batches
            )
            
            # Combine results
            if progress_callback:
                progress_callback(len(batches), len(batches), "Combining results...")
            
            for batch_result in batch_results:
                for ptype in plastic_types:
                    for idx, score, window_scores in batch_result[ptype]:
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
                progress_callback, n_jobs, fast_mode, lam, p, niter
            )
            
            # Reshape back to 3D
            score_maps = {ptype: scores.reshape(nx, ny) 
                         for ptype, scores in flat_scores.items()}
        
        logger.info("Map scanning complete")
        if progress_callback:
            progress_callback(100, 100, "✅ Scanning complete!")
        
        return score_maps
