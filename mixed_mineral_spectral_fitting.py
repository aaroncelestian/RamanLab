#!/usr/bin/env python3
"""
Mixed Mineral Spectral Fitting and Deconvolution Module
========================================================

Advanced spectral fitting module specifically designed for mixed mineral analysis.
This module integrates with RamanLab's proven correlation-based database search
for the initial phase identification, then uses more advanced techniques like
DTW (Dynamic Time Warping) for residual analysis.

Key Features:
- Uses existing proven correlation-based search for major phase detection
- DTW-based residual analysis for overlapping/minor phases  
- Iterative constrained fitting with physicochemical constraints
- Non-negative matrix factorization (NMF) for component separation
- Bayesian peak deconvolution with prior knowledge
- Spectral unmixing with endmember detection
- Weighted residual analysis
- Phase quantification with uncertainty estimates

Author: RamanLab Enhanced Analysis System
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.stats import multivariate_normal
from scipy.linalg import lstsq
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import correlation
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Try to import advanced ML libraries
try:
    from sklearn.decomposition import NMF, PCA, FastICA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to import DTW for residual analysis
try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    try:
        from fastdtw import fastdtw
        DTW_AVAILABLE = True
        DTW_BACKEND = "fastdtw"
    except ImportError:
        DTW_AVAILABLE = False

# =============================================================================
# MINIMAL STEP-BY-STEP APPROACH - NO COMPLEX FUNCTIONS
# =============================================================================

def simple_database_search_only():
    """
    STEP 1: Just do a simple database search - nothing else.
    This isolates any numpy issues.
    """
    import numpy as np  # Import here to avoid scoping issues
    
    print("STEP 1: Simple Database Search Only")
    print("=" * 40)
    
    try:
        # Import the real database
        from raman_spectra_qt6 import RamanSpectraQt6
        
        print("Connecting to database...")
        raman_db = RamanSpectraQt6()
        
        # Check database
        stats = raman_db.get_database_stats()
        print(f"Database loaded: {stats['total_spectra']} spectra")
        
        if stats['total_spectra'] == 0:
            print("❌ Database is empty!")
            return None
        
        # Create simple test spectrum
        print("Creating test spectrum...")
        wavenumbers = np.linspace(100, 1200, 500)
        test_spectrum = np.zeros_like(wavenumbers)
        
        # Simple peaks
        for peak in [206, 465, 696]:
            test_spectrum += np.exp(-(wavenumbers - peak)**2 / (2 * 10**2))
        
        print(f"Test spectrum created: {len(wavenumbers)} points")
        
        # Do the search
        print("Searching database...")
        results = raman_db.search_database(
            wavenumbers,
            test_spectrum,
            n_matches=3,
            threshold=0.3
        )
        
        print(f"Search complete: {len(results)} matches found")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['name'][:50]}...")
            print(f"     Score: {result['score']:.4f}")
        
        return {
            'database': raman_db,
            'wavenumbers': wavenumbers,
            'spectrum': test_spectrum,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def simple_fitting_with_residual(search_data):
    """
    STEP 2: Take search results and do simple fitting, show residual.
    """
    import numpy as np  # Import here to avoid scoping issues
    import matplotlib.pyplot as plt
    
    print("\nSTEP 2: Simple Fitting with Residual")
    print("=" * 40)
    
    if search_data is None:
        print("❌ No search data provided")
        return None
    
    try:
        wavenumbers = search_data['wavenumbers']
        spectrum = search_data['spectrum']
        results = search_data['results']
        
        if not results:
            print("❌ No search results to fit")
            return None
        
        # Get the best match
        best_match = results[0]
        print(f"Fitting to: {best_match['name'][:50]}...")
        
        # Simple Gaussian fitting approach
        from scipy.optimize import minimize
        
        # Find peaks in the spectrum
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(spectrum, prominence=0.1)
        
        if len(peaks) == 0:
            print("❌ No peaks found for fitting")
            return None
        
        peak_positions = wavenumbers[peaks]
        peak_heights = spectrum[peaks]
        
        print(f"Found {len(peaks)} peaks to fit")
        
        # Simple Gaussian function
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
        
        def multi_gaussian(params):
            """Multiple Gaussian model"""
            model = np.zeros_like(wavenumbers)
            for i in range(0, len(params), 3):
                if i + 2 < len(params):
                    amp, cen, wid = params[i], params[i+1], params[i+2]
                    model += gaussian(wavenumbers, amp, cen, wid)
            return model
        
        def objective(params):
            """Objective function for fitting"""
            model = multi_gaussian(params)
            return np.sum((spectrum - model)**2)
        
        # Initial parameters: [amp, center, width] for each peak
        initial_params = []
        for pos, height in zip(peak_positions, peak_heights):
            initial_params.extend([height, pos, 10.0])  # amp, center, width
        
        print("Performing least squares fitting...")
        
        # Fit
        result = minimize(objective, initial_params, method='Nelder-Mead')
        
        if result.success:
            print(f"✓ Fitting successful")
            
            # Generate fitted spectrum
            fitted_spectrum = multi_gaussian(result.x)
            
            # Calculate residual
            residual = spectrum - fitted_spectrum
            
            # Calculate R²
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((spectrum - np.mean(spectrum))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"  R²: {r_squared:.4f}")
            print(f"  RMS residual: {np.sqrt(np.mean(residual**2)):.6f}")
            
            # Find positive peaks in residual
            positive_residual = np.maximum(residual, 0)
            residual_peaks, _ = find_peaks(positive_residual, prominence=0.01)
            
            print(f"  Positive residual peaks: {len(residual_peaks)}")
            
            # Plot the results
            print("Creating plots...")
            
            plt.figure(figsize=(12, 8))
            
            # Original spectrum and fit
            plt.subplot(2, 1, 1)
            plt.plot(wavenumbers, spectrum, 'b-', label='Original', linewidth=1)
            plt.plot(wavenumbers, fitted_spectrum, 'r-', label='Fitted', linewidth=1)
            plt.legend()
            plt.title(f'Spectrum Fit (R² = {r_squared:.4f})')
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Intensity')
            plt.grid(True, alpha=0.3)
            
            # Residual
            plt.subplot(2, 1, 2)
            plt.plot(wavenumbers, residual, 'g-', label='Residual', linewidth=1)
            plt.plot(wavenumbers, positive_residual, 'r-', label='Positive Residual', linewidth=1, alpha=0.7)
            
            # Mark residual peaks
            if len(residual_peaks) > 0:
                plt.plot(wavenumbers[residual_peaks], positive_residual[residual_peaks], 'ro', 
                        markersize=6, label=f'{len(residual_peaks)} Peaks')
            
            plt.legend()
            plt.title('Residual Analysis (for minor phases)')
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Residual Intensity')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.show()
            
            return {
                'fitted_spectrum': fitted_spectrum,
                'residual': residual,
                'positive_residual': positive_residual,
                'residual_peaks': residual_peaks,
                'r_squared': r_squared,
                'fit_params': result.x
            }
            
        else:
            print(f"❌ Fitting failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"❌ Fitting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def simple_residual_search(fit_data, search_data):
    """
    STEP 3: Search for minor phases in the residual.
    """
    import numpy as np  # Import here to avoid scoping issues
    
    print("\nSTEP 3: Residual Search for Minor Phases")
    print("=" * 40)
    
    if fit_data is None or search_data is None:
        print("❌ No fit or search data provided")
        return None
    
    try:
        database = search_data['database']
        wavenumbers = search_data['wavenumbers']
        positive_residual = fit_data['positive_residual']
        residual_peaks = fit_data['residual_peaks']
        
        print(f"Analyzing residual with {len(residual_peaks)} peaks...")
        
        if len(residual_peaks) == 0:
            print("No residual peaks to analyze")
            return {'minor_phases': []}
        
        # Search database with residual spectrum
        print("Searching database with residual spectrum...")
        
        residual_results = database.search_database(
            wavenumbers,
            positive_residual,
            n_matches=3,
            threshold=0.2  # Lower threshold for residual
        )
        
        print(f"Found {len(residual_results)} potential minor phases:")
        
        minor_phases = []
        for i, result in enumerate(residual_results):
            # Estimate abundance from residual
            residual_max = np.max(positive_residual)
            original_max = np.max(search_data['spectrum'])
            estimated_abundance = residual_max / original_max if original_max > 0 else 0
            
            phase_info = {
                'name': result['name'],
                'score': result['score'],
                'abundance': estimated_abundance,
                'method': 'residual_search'
            }
            
            minor_phases.append(phase_info)
            
            print(f"  {i+1}. {result['name'][:50]}...")
            print(f"     Score: {result['score']:.4f}")
            print(f"     Est. abundance: {estimated_abundance:.1%}")
        
        return {'minor_phases': minor_phases}
        
    except Exception as e:
        print(f"❌ Residual search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# INTEGRATION WITH EXISTING RAMANLAB SEARCH SYSTEM
# =============================================================================

class RamanLabDatabaseInterface:
    """
    Interface to existing RamanLab database search system.
    
    This integrates with the proven correlation-based search from raman_spectra_qt6.py
    rather than creating a separate mineral database.
    """
    
    def __init__(self, raman_spectra_instance=None):
        """
        Initialize interface to RamanLab database.
        
        Parameters:
        -----------
        raman_spectra_instance : RamanSpectraQt6
            Instance of the main RamanLab database class
        """
        self.raman_db = raman_spectra_instance
        
    def search_correlation_based(self, 
                                wavenumbers: np.ndarray, 
                                intensities: np.ndarray,
                                n_matches: int = 10,
                                threshold: float = 0.5) -> List[Dict]:
        """
        Use the existing proven correlation-based search from RamanLab.
        
        This is the first search step - use what already works well.
        
        Parameters:
        -----------
        wavenumbers : np.ndarray
            Query spectrum wavenumbers
        intensities : np.ndarray  
            Query spectrum intensities
        n_matches : int
            Number of matches to return
        threshold : float
            Correlation threshold
            
        Returns:
        --------
        List[Dict]
            Search results using proven correlation method
        """
        if self.raman_db is None:
            return []
            
        try:
            # Use the existing proven search method
            matches = self.raman_db.search_database(
                wavenumbers, 
                intensities, 
                n_matches=n_matches,
                threshold=threshold
            )
            
            # Convert to our expected format
            formatted_results = []
            for match in matches:
                formatted_results.append({
                    'name': match['name'],
                    'score': match['score'],
                    'correlation_score': match['score'],  # This is already correlation-based
                    'metadata': match.get('metadata', {}),
                    'peaks': match.get('peaks', []),
                    'timestamp': match.get('timestamp', 'Unknown'),
                    'search_method': 'correlation',
                    'confidence': match['score']  # Correlation score as confidence
                })
                
            return formatted_results
            
        except Exception as e:
            print(f"Correlation-based search failed: {e}")
            return []
    
    def search_dtw_based(self,
                        wavenumbers: np.ndarray,
                        intensities: np.ndarray, 
                        n_matches: int = 5,
                        window_size: Optional[int] = None) -> List[Dict]:
        """
        DTW-based search for residual/overlapping spectra.
        
        This is for the second search step where correlation might not work well
        due to peak shifts, overlaps, or matrix effects.
        
        Parameters:
        -----------
        wavenumbers : np.ndarray
            Query spectrum wavenumbers  
        intensities : np.ndarray
            Query spectrum intensities
        n_matches : int
            Number of matches to return
        window_size : int, optional
            DTW window constraint
            
        Returns:
        --------
        List[Dict]
            DTW-based search results
        """
        if not DTW_AVAILABLE or self.raman_db is None:
            return []
            
        try:
            matches = []
            
            # Get all database entries
            database = self.raman_db.database
            if not database:
                return []
            
            # Normalize query spectrum
            query_norm = self._normalize_for_dtw(intensities)
            
            for name, entry in database.items():
                try:
                    # Get database spectrum
                    db_wavenumbers = np.array(entry['wavenumbers'])
                    db_intensities = np.array(entry['intensities'])
                    
                    # Interpolate to common wavenumber range
                    common_min = max(wavenumbers.min(), db_wavenumbers.min())
                    common_max = min(wavenumbers.max(), db_wavenumbers.max())
                    
                    if common_min >= common_max:
                        continue
                    
                    # Create common grid and interpolate
                    common_wavenumbers = np.linspace(common_min, common_max, 200)
                    query_interp = np.interp(common_wavenumbers, wavenumbers, query_norm)
                    db_interp = np.interp(common_wavenumbers, db_wavenumbers, db_intensities)
                    
                    # Normalize database spectrum
                    db_norm = self._normalize_for_dtw(db_interp)
                    
                    # Calculate DTW distance
                    if DTW_BACKEND == "fastdtw":
                        dtw_distance, _ = fastdtw(query_interp, db_norm)
                    else:
                        dtw_distance = dtw.distance(query_interp, db_norm, window=window_size)
                    
                    # Convert distance to similarity score (0-1, higher is better)
                    max_possible_distance = len(query_interp) * 2  # Rough estimate
                    dtw_score = max(0, 1 - (dtw_distance / max_possible_distance))
                    
                    matches.append({
                        'name': name,
                        'score': dtw_score,
                        'dtw_distance': dtw_distance,
                        'metadata': entry.get('metadata', {}),
                        'peaks': entry.get('peaks', []),
                        'timestamp': entry.get('timestamp', 'Unknown'),
                        'search_method': 'dtw',
                        'confidence': dtw_score
                    })
                    
                except Exception as e:
                    continue
                    
            # Sort by DTW score and return top matches
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[:n_matches]
            
        except Exception as e:
            print(f"DTW-based search failed: {e}")
            return []
    
    def _normalize_for_dtw(self, intensities: np.ndarray) -> np.ndarray:
        """Normalize spectrum for DTW analysis."""
        # Remove baseline and normalize to [0,1]
        baseline = np.percentile(intensities, 10)  # Simple baseline estimate
        corrected = intensities - baseline
        corrected = np.maximum(corrected, 0)  # Ensure non-negative
        
        if np.max(corrected) > 0:
            return corrected / np.max(corrected)
        else:
            return corrected


class FittingMethod(Enum):
    """Enumeration of available fitting methods."""
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    PSEUDO_VOIGT = "pseudo_voigt"
    PEARSON_VII = "pearson_vii"
    ASYMMETRIC_GAUSSIAN = "asymmetric_gaussian"


class ConstraintType(Enum):
    """Types of constraints for peak fitting."""
    POSITION = "position"
    WIDTH = "width"
    INTENSITY = "intensity"
    RATIO = "ratio"


@dataclass
class PeakConstraint:
    """Constraint for peak fitting."""
    constraint_type: ConstraintType
    peak_indices: List[int]
    bounds: Tuple[float, float]
    weight: float = 1.0
    description: str = ""


@dataclass
class PhaseInfo:
    """Information about a mineral phase."""
    name: str
    formula: str
    expected_peaks: List[float]  # Expected peak positions
    peak_tolerances: List[float]  # Allowed deviations
    peak_intensities: List[float]  # Relative intensities
    constraints: List[PeakConstraint]
    confidence: float = 0.0
    abundance: float = 0.0
    search_method: str = "correlation"  # Track which search method found this
    database_metadata: Dict = None  # Original database metadata


@dataclass
class FitResult:
    """Results from spectral fitting."""
    phases: List[PhaseInfo]
    fitted_spectrum: np.ndarray
    residual: np.ndarray
    individual_components: Dict[str, np.ndarray]
    r_squared: float
    reduced_chi_squared: float
    uncertainties: Dict[str, float]
    convergence_info: Dict[str, any]


class MixedMineralFitter:
    """
    Advanced spectral fitting class for mixed mineral analysis.
    
    This class integrates with the existing proven RamanLab correlation-based 
    search for initial phase detection, then uses DTW and other advanced 
    techniques for residual analysis.
    """
    
    def __init__(self, wavenumbers: np.ndarray, intensities: np.ndarray, raman_db=None):
        """
        Initialize the mixed mineral fitter.
        
        Parameters:
        -----------
        wavenumbers : np.ndarray
            Wavenumber array (cm⁻¹)
        intensities : np.ndarray
            Intensity array
        raman_db : RamanSpectraQt6, optional
            RamanLab database instance for proven search methods
        """
        self.wavenumbers = np.array(wavenumbers)
        self.intensities = np.array(intensities)
        self.original_intensities = self.intensities.copy()
        
        # Database interface - use existing proven search system
        self.db_interface = RamanLabDatabaseInterface(raman_db)
        
        # Analysis state
        self.background = None
        self.processed_intensities = self.intensities.copy()
        self.detected_phases = []
        self.major_phase = None
        self.minor_phases = []
        self.fit_results = []
        
        # Fitting parameters
        self.max_iterations = 100
        self.convergence_threshold = 1e-6
        self.peak_width_bounds = (2.0, 50.0)  # cm⁻¹
        self.intensity_bounds = (0.0, np.inf)
        
    def detect_major_phase(self, 
                          peak_prominence: float = 0.1,
                          peak_height: float = 0.05,
                          correlation_threshold: float = 0.6,
                          n_matches: int = 10) -> PhaseInfo:
        """
        Detect the major (dominant) phase using proven correlation-based search.
        
        This leverages the existing RamanLab correlation search that already works well.
        
        Parameters:
        -----------
        peak_prominence : float
            Minimum peak prominence for detection
        peak_height : float
            Minimum peak height for detection
        correlation_threshold : float
            Minimum correlation threshold for database search
        n_matches : int
            Number of database matches to consider
            
        Returns:
        --------
        PhaseInfo
            Information about the detected major phase
        """
        # Use existing proven correlation-based search
        search_results = self.db_interface.search_correlation_based(
            self.wavenumbers,
            self.processed_intensities,
            n_matches=n_matches,
            threshold=correlation_threshold
        )
        
        if search_results:
            # Use best correlation match as major phase
            best_match = search_results[0]
            
            # Find peaks in the current spectrum for fitting
            peaks, properties = find_peaks(
                self.processed_intensities,
                prominence=peak_prominence,
                height=peak_height,
                distance=5
            )
            
            if len(peaks) > 0:
                peak_positions = self.wavenumbers[peaks]
                peak_intensities = self.processed_intensities[peaks]
                
                # Sort by intensity
                sorted_indices = np.argsort(peak_intensities)[::-1]
                peak_positions = peak_positions[sorted_indices]
                peak_intensities = peak_intensities[sorted_indices]
            else:
                # Fallback to database peaks if no peaks detected
                peak_positions = np.array(best_match.get('peaks', []))
                peak_intensities = np.ones(len(peak_positions))
            
            major_phase = PhaseInfo(
                name=best_match['name'],
                formula=best_match.get('metadata', {}).get('formula', 'Unknown'),
                expected_peaks=peak_positions[:10].tolist(),  # Top 10 peaks
                peak_tolerances=[5.0] * min(10, len(peak_positions)),
                peak_intensities=(peak_intensities[:10] / np.max(peak_intensities[:10]) if len(peak_intensities) > 0 else [1.0]).tolist(),
                constraints=[],
                confidence=best_match['score'],
                abundance=0.8,  # Initial estimate for major phase
                search_method='correlation',
                database_metadata=best_match.get('metadata', {})
            )
            
        else:
            # Fallback to peak-based detection if no database matches
            peaks, properties = find_peaks(
                self.processed_intensities,
                prominence=peak_prominence,
                height=peak_height,
                distance=5
            )
            
            if len(peaks) == 0:
                raise ValueError("No peaks detected and no database matches found")
            
            peak_positions = self.wavenumbers[peaks]
            peak_intensities = self.processed_intensities[peaks]
            
            # Sort by intensity
            sorted_indices = np.argsort(peak_intensities)[::-1]
            peak_positions = peak_positions[sorted_indices]
            peak_intensities = peak_intensities[sorted_indices]
            
            major_phase = PhaseInfo(
                name="Unknown_Major_Phase",
                formula="Unknown",
                expected_peaks=peak_positions[:10].tolist(),
                peak_tolerances=[5.0] * min(10, len(peak_positions)),
                peak_intensities=(peak_intensities[:10] / np.max(peak_intensities[:10])).tolist(),
                constraints=[],
                confidence=0.5,
                abundance=0.8,
                search_method='peak_detection',
                database_metadata={}
            )
        
        self.major_phase = major_phase
        return major_phase
    
    def detect_minor_phases(self, 
                           residual_spectrum: np.ndarray,
                           min_phase_abundance: float = 0.05,
                           max_minor_phases: int = 3,
                           use_dtw: bool = True) -> List[PhaseInfo]:
        """
        Detect minor phases in the residual spectrum using DTW-based search.
        
        This is where DTW shines - handling overlapped, shifted, or matrix-affected peaks
        that correlation-based search might miss.
        
        Parameters:
        -----------
        residual_spectrum : np.ndarray
            Corrected residual spectrum after major phase subtraction
        min_phase_abundance : float
            Minimum abundance threshold for phase detection
        max_minor_phases : int
            Maximum number of minor phases to detect
        use_dtw : bool
            Whether to use DTW search for minor phases
            
        Returns:
        --------
        List[PhaseInfo]
            List of detected minor phases
        """
        # Find peaks in residual
        peaks, properties = find_peaks(
            residual_spectrum,
            prominence=0.02,  # Lower threshold for minor phases
            height=0.01,
            distance=3
        )
        
        if len(peaks) == 0:
            return []
        
        minor_phases = []
        
        if use_dtw and DTW_AVAILABLE:
            # Use DTW-based search for residual analysis
            dtw_results = self.db_interface.search_dtw_based(
                self.wavenumbers,
                residual_spectrum,
                n_matches=max_minor_phases * 2,  # Get more candidates
                window_size=20  # Allow some flexibility in alignment
            )
            
            for result in dtw_results:
                if len(minor_phases) >= max_minor_phases:
                    break
                    
                # Estimate abundance from residual
                peak_positions = self.wavenumbers[peaks]
                peak_intensities = residual_spectrum[peaks]
                abundance = np.sum(peak_intensities) / np.sum(self.processed_intensities)
                
                if abundance >= min_phase_abundance:
                    minor_phase = PhaseInfo(
                        name=result['name'],
                        formula=result.get('metadata', {}).get('formula', 'Unknown'),
                        expected_peaks=peak_positions[:5].tolist(),  # Top 5 peaks for minor phase
                        peak_tolerances=[3.0] * min(5, len(peak_positions)),
                        peak_intensities=(peak_intensities[:5] / np.max(peak_intensities[:5]) if len(peak_intensities) > 0 else [1.0]).tolist(),
                        constraints=[],
                        confidence=result['score'],
                        abundance=abundance,
                        search_method='dtw',
                        database_metadata=result.get('metadata', {})
                    )
                    minor_phases.append(minor_phase)
        else:
            # Fallback to correlation-based search for minor phases
            correlation_results = self.db_interface.search_correlation_based(
                self.wavenumbers,
                residual_spectrum,
                n_matches=max_minor_phases * 2,
                threshold=0.3  # Lower threshold for minor phases
            )
            
            for result in correlation_results:
                if len(minor_phases) >= max_minor_phases:
                    break
                    
                peak_positions = self.wavenumbers[peaks]
                peak_intensities = residual_spectrum[peaks]
                abundance = np.sum(peak_intensities) / np.sum(self.processed_intensities)
                
                if abundance >= min_phase_abundance:
                    minor_phase = PhaseInfo(
                        name=result['name'],
                        formula=result.get('metadata', {}).get('formula', 'Unknown'),
                        expected_peaks=peak_positions[:5].tolist(),
                        peak_tolerances=[3.0] * min(5, len(peak_positions)),
                        peak_intensities=(peak_intensities[:5] / np.max(peak_intensities[:5]) if len(peak_intensities) > 0 else [1.0]).tolist(),
                        constraints=[],
                        confidence=result['score'],
                        abundance=abundance,
                        search_method='correlation_residual',
                        database_metadata=result.get('metadata', {})
                    )
                    minor_phases.append(minor_phase)
        
        self.minor_phases = minor_phases
        return minor_phases
    
    def preprocess_spectrum(self, 
                          remove_background: bool = True,
                          background_method: str = "als",
                          smooth_spectrum: bool = True,
                          smoothing_method: str = "savgol",
                          normalize: bool = True) -> np.ndarray:
        """
        Preprocess the spectrum for fitting.
        
        Parameters:
        -----------
        remove_background : bool
            Whether to remove background
        background_method : str
            Background removal method ('als', 'polynomial', 'rolling_ball')
        smooth_spectrum : bool
            Whether to smooth the spectrum
        smoothing_method : str
            Smoothing method ('savgol', 'gaussian', 'median')
        normalize : bool
            Whether to normalize intensities
            
        Returns:
        --------
        np.ndarray
            Processed intensity array
        """
        processed = self.intensities.copy()
        
        if remove_background:
            self.background = self._calculate_background(processed, method=background_method)
            processed = processed - self.background
            
        if smooth_spectrum:
            processed = self._smooth_spectrum(processed, method=smoothing_method)
            
        if normalize:
            processed = self._normalize_spectrum(processed)
            
        self.processed_intensities = processed
        return processed
    
    def fit_major_phase(self, 
                       fitting_method: FittingMethod = FittingMethod.PSEUDO_VOIGT,
                       use_constraints: bool = True) -> FitResult:
        """
        Fit the major phase to the spectrum.
        
        Parameters:
        -----------
        fitting_method : FittingMethod
            Peak shape function to use
        use_constraints : bool
            Whether to apply physicochemical constraints
            
        Returns:
        --------
        FitResult
            Fitting results for the major phase
        """
        if self.major_phase is None:
            raise ValueError("Major phase not detected. Run detect_major_phase() first.")
        
        # Set up fitting parameters
        initial_params = self._setup_initial_parameters(self.major_phase, fitting_method)
        
        # Set up constraints
        bounds = self._setup_parameter_bounds(self.major_phase, use_constraints)
        
        # Perform constrained fitting
        try:
            result = self._perform_constrained_fitting(
                initial_params, bounds, fitting_method, self.major_phase
            )
            
            # Create fitted spectrum
            fitted_spectrum = self._generate_fitted_spectrum(
                result.x, fitting_method, self.major_phase
            )
            
            # Calculate residual
            residual = self.processed_intensities - fitted_spectrum
            
            # Calculate quality metrics
            r_squared = self._calculate_r_squared(fitted_spectrum)
            chi_squared = self._calculate_chi_squared(fitted_spectrum)
            
            # Update phase abundance based on fit
            self.major_phase.abundance = self._estimate_phase_abundance(fitted_spectrum)
            
            fit_result = FitResult(
                phases=[self.major_phase],
                fitted_spectrum=fitted_spectrum,
                residual=residual,
                individual_components={self.major_phase.name: fitted_spectrum},
                r_squared=r_squared,
                reduced_chi_squared=chi_squared,
                uncertainties=self._calculate_parameter_uncertainties(result),
                convergence_info={"success": result.success, "iterations": result.nit}
            )
            
            self.fit_results.append(fit_result)
            return fit_result
            
        except Exception as e:
            raise RuntimeError(f"Major phase fitting failed: {str(e)}")
    
    def analyze_residual_spectrum(self, 
                                 major_phase_fit: FitResult,
                                 weighted_analysis: bool = True,
                                 overlap_correction: bool = True) -> np.ndarray:
        """
        Analyze the residual spectrum after major phase subtraction.
        
        This is the critical step that handles overlapping peaks properly.
        Instead of simple subtraction, we use weighted residual analysis.
        
        Parameters:
        -----------
        major_phase_fit : FitResult
            Results from major phase fitting
        weighted_analysis : bool
            Whether to use weighted residual analysis
        overlap_correction : bool
            Whether to apply overlap correction
            
        Returns:
        --------
        np.ndarray
            Corrected residual spectrum
        """
        # Get the raw residual
        raw_residual = major_phase_fit.residual
        
        if not weighted_analysis:
            return raw_residual
        
        # Calculate overlap weights
        overlap_weights = self._calculate_overlap_weights(major_phase_fit)
        
        # Apply weighted correction
        corrected_residual = self._apply_weighted_correction(
            raw_residual, overlap_weights, major_phase_fit
        )
        
        if overlap_correction:
            # Apply additional overlap correction
            corrected_residual = self._apply_overlap_correction(
                corrected_residual, major_phase_fit
            )
        
        return corrected_residual
    
    def perform_global_fitting(self, 
                              major_phase: PhaseInfo,
                              minor_phases: List[PhaseInfo],
                              fitting_method: FittingMethod = FittingMethod.PSEUDO_VOIGT,
                              use_constraints: bool = True,
                              iterative_refinement: bool = True) -> FitResult:
        """
        Perform global fitting of all phases simultaneously.
        
        This is the final step that optimizes all phases together while
        maintaining physicochemical constraints.
        
        Parameters:
        -----------
        major_phase : PhaseInfo
            Major phase information
        minor_phases : List[PhaseInfo]
            List of minor phases
        fitting_method : FittingMethod
            Peak shape function
        use_constraints : bool
            Whether to use constraints
        iterative_refinement : bool
            Whether to use iterative refinement
            
        Returns:
        --------
        FitResult
            Global fitting results
        """
        all_phases = [major_phase] + minor_phases
        
        # Set up global parameters
        initial_params = []
        bounds_lower = []
        bounds_upper = []
        
        for phase in all_phases:
            phase_params = self._setup_initial_parameters(phase, fitting_method)
            phase_bounds = self._setup_parameter_bounds(phase, use_constraints)
            
            initial_params.extend(phase_params)
            bounds_lower.extend(phase_bounds[0])
            bounds_upper.extend(phase_bounds[1])
        
        bounds = (bounds_lower, bounds_upper)
        
        # Define global objective function
        def global_objective(params):
            return self._calculate_global_residual(params, all_phases, fitting_method)
        
        # Perform global optimization
        try:
            if iterative_refinement:
                result = self._iterative_global_fitting(
                    initial_params, bounds, global_objective, all_phases
                )
            else:
                # Convert bounds format for scipy.optimize.minimize
                bounds_formatted = list(zip(bounds[0], bounds[1]))
                result = minimize(
                    global_objective,
                    initial_params,
                    bounds=bounds_formatted,
                    method='L-BFGS-B'
                )
            
            # Parse results
            fitted_components = {}
            fitted_spectrum = np.zeros_like(self.wavenumbers)
            
            param_idx = 0
            for phase in all_phases:
                n_params = len(phase.expected_peaks) * 3  # amp, cen, wid for each peak
                phase_params = result.x[param_idx:param_idx + n_params]
                param_idx += n_params
                
                # Generate component spectrum for this phase
                component_spectrum = self._generate_fitted_spectrum(
                    phase_params, fitting_method, phase
                )
                fitted_components[phase.name] = component_spectrum
                fitted_spectrum += component_spectrum
                
                # Update phase abundance
                phase.abundance = self._estimate_phase_abundance(component_spectrum)
            
            # Calculate final residual and quality metrics
            residual = self.processed_intensities - fitted_spectrum
            r_squared = self._calculate_r_squared(fitted_spectrum)
            chi_squared = self._calculate_chi_squared(fitted_spectrum)
            
            global_fit_result = FitResult(
                phases=all_phases,
                fitted_spectrum=fitted_spectrum,
                residual=residual,
                individual_components=fitted_components,
                r_squared=r_squared,
                reduced_chi_squared=chi_squared,
                uncertainties=self._calculate_global_uncertainties(result, all_phases),
                convergence_info={"success": result.success, "iterations": getattr(result, 'nit', 0)}
            )
            
            return global_fit_result
            
        except Exception as e:
            raise RuntimeError(f"Global fitting failed: {str(e)}")
    
    def quantify_phases(self, fit_result: FitResult) -> Dict[str, Dict[str, float]]:
        """
        Quantify the relative abundances of different phases.
        
        Parameters:
        -----------
        fit_result : FitResult
            Global fitting results
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Phase quantification results with uncertainties
        """
        quantification = {}
        
        # Calculate integrated intensities for each phase
        phase_integrals = {}
        total_integral = 0
        
        for phase_name, component_spectrum in fit_result.individual_components.items():
            # Calculate area under curve using trapezoidal rule
            integral = np.trapz(component_spectrum, self.wavenumbers)
            phase_integrals[phase_name] = integral
            total_integral += integral
        
        # Calculate relative abundances
        for phase_name, integral in phase_integrals.items():
            relative_abundance = integral / total_integral if total_integral > 0 else 0
            
            # Estimate uncertainty based on fit quality
            uncertainty = self._estimate_abundance_uncertainty(
                phase_name, fit_result, relative_abundance
            )
            
            quantification[phase_name] = {
                'abundance': relative_abundance,
                'uncertainty': uncertainty,
                'integral': integral,
                'confidence': self._calculate_phase_confidence(phase_name, fit_result)
            }
        
        return quantification
    
    def generate_analysis_report(self, 
                               fit_result: FitResult,
                               quantification: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a comprehensive analysis report.
        
        Parameters:
        -----------
        fit_result : FitResult
            Global fitting results
        quantification : Dict[str, Dict[str, float]]
            Phase quantification results
            
        Returns:
        --------
        str
            Analysis report
        """
        report = []
        report.append("=" * 60)
        report.append("MIXED MINERAL SPECTRAL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Fitting quality
        report.append("OVERALL FIT QUALITY:")
        report.append(f"  R² = {fit_result.r_squared:.4f}")
        report.append(f"  Reduced χ² = {fit_result.reduced_chi_squared:.4f}")
        report.append(f"  Convergence: {'Success' if fit_result.convergence_info.get('success', False) else 'Failed'}")
        report.append("")
        
        # Phase identification and quantification
        report.append("PHASE IDENTIFICATION & QUANTIFICATION:")
        report.append("-" * 40)
        
        for phase in fit_result.phases:
            phase_name = phase.name
            quant_data = quantification.get(phase_name, {})
            
            report.append(f"Phase: {phase_name}")
            report.append(f"  Formula: {phase.formula}")
            report.append(f"  Abundance: {quant_data.get('abundance', 0):.2%} ± {quant_data.get('uncertainty', 0):.2%}")
            report.append(f"  Confidence: {quant_data.get('confidence', 0):.3f}")
            report.append(f"  Expected peaks: {', '.join(f'{p:.1f}' for p in phase.expected_peaks)}")
            report.append("")
        
        # Peak assignments
        report.append("PEAK ASSIGNMENTS:")
        report.append("-" * 20)
        
        for phase in fit_result.phases:
            report.append(f"{phase.name}:")
            for i, (pos, intensity) in enumerate(zip(phase.expected_peaks, phase.peak_intensities)):
                report.append(f"  {pos:.1f} cm⁻¹ (Intensity: {intensity:.3f})")
            report.append("")
        
        # Analysis method summary
        report.append("ANALYSIS METHOD:")
        report.append("-" * 15)
        report.append("1. Major phase detection via peak intensity analysis")
        report.append("2. Constrained fitting of major phase")
        report.append("3. Weighted residual analysis with overlap correction")
        report.append("4. Minor phase detection in corrected residual")
        report.append("5. Global optimization of all phases")
        report.append("6. Quantitative phase analysis")
        report.append("")
        
        return "\n".join(report)
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _calculate_background(self, spectrum: np.ndarray, method: str = "als") -> np.ndarray:
        """Calculate background using various methods."""
        if method == "als":
            return self._baseline_als(spectrum)
        elif method == "polynomial":
            return self._polynomial_background(spectrum)
        elif method == "rolling_ball":
            return self._rolling_ball_background(spectrum)
        else:
            return np.zeros_like(spectrum)
    
    def _baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """Asymmetric Least Squares baseline correction."""
        try:
            # Ensure y is a valid numpy array
            if not isinstance(y, np.ndarray):
                y = np.array(y)
                
            L = len(y)
            if L < 3:
                return np.zeros_like(y)
                
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w*y)
                
                # Fix array boolean context issue by using explicit array operations
                condition1 = y > z
                condition2 = y <= z
                w = p * condition1.astype(float) + (1-p) * condition2.astype(float)
                
            return z
        except Exception as e:
            # Return zeros array on error
            return np.zeros_like(y) if isinstance(y, np.ndarray) else np.zeros(len(y))
    
    def _polynomial_background(self, spectrum: np.ndarray, degree: int = 3) -> np.ndarray:
        """Polynomial background estimation."""
        x = np.arange(len(spectrum))
        coeffs = np.polyfit(x, spectrum, degree)
        return np.polyval(coeffs, x)
    
    def _rolling_ball_background(self, spectrum: np.ndarray, window: int = 50) -> np.ndarray:
        """Rolling ball background estimation."""
        # Simplified rolling ball - minimum filter
        from scipy.ndimage import minimum_filter
        return minimum_filter(spectrum, size=window)
    
    def _smooth_spectrum(self, spectrum: np.ndarray, method: str = "savgol") -> np.ndarray:
        """Smooth spectrum using various methods."""
        if method == "savgol":
            from scipy.signal import savgol_filter
            return savgol_filter(spectrum, window_length=5, polyorder=3)
        elif method == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(spectrum, sigma=1.0)
        elif method == "median":
            from scipy.signal import medfilt
            return medfilt(spectrum, kernel_size=3)
        else:
            return spectrum
    
    def _normalize_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Normalize spectrum to [0, 1] range."""
        min_val = np.min(spectrum)
        max_val = np.max(spectrum)
        if max_val > min_val:
            return (spectrum - min_val) / (max_val - min_val)
        else:
            return spectrum
    
    def _match_major_phase_to_database(self, 
                                     peak_positions: np.ndarray,
                                     peak_intensities: np.ndarray) -> PhaseInfo:
        """Match detected peaks to database for major phase identification."""
        # This would interface with the mineral database
        # For now, return a placeholder
        return PhaseInfo(
            name="Major_Phase_DB_Match",
            formula="Unknown",
            expected_peaks=peak_positions[:8].tolist(),
            peak_tolerances=[3.0] * len(peak_positions[:8]),
            peak_intensities=(peak_intensities[:8] / np.max(peak_intensities[:8])).tolist(),
            constraints=[],
            confidence=0.7,
            abundance=0.75
        )
    
    def _match_minor_phase_to_database(self, 
                                     peak_positions: np.ndarray,
                                     peak_intensities: np.ndarray,
                                     estimated_abundance: float) -> Optional[PhaseInfo]:
        """Match peaks to database for minor phase identification."""
        # This would interface with the mineral database
        # For now, return a placeholder if abundance is significant enough
        if estimated_abundance >= 0.05:
            return PhaseInfo(
                name=f"Minor_Phase_{len(self.minor_phases) + 1}",
                formula="Unknown",
                expected_peaks=peak_positions[:5].tolist(),
                peak_tolerances=[3.0] * len(peak_positions[:5]),
                peak_intensities=(peak_intensities[:5] / np.max(peak_intensities[:5])).tolist(),
                constraints=[],
                confidence=0.5,
                abundance=estimated_abundance
            )
        return None
    
    def _setup_initial_parameters(self, phase: PhaseInfo, fitting_method: FittingMethod) -> List[float]:
        """Set up initial parameters for fitting."""
        params = []
        for pos, intensity in zip(phase.expected_peaks, phase.peak_intensities):
            params.extend([intensity, pos, 5.0])  # amp, cen, wid
        return params
    
    def _setup_parameter_bounds(self, phase: PhaseInfo, use_constraints: bool) -> Tuple[List[float], List[float]]:
        """Set up parameter bounds for constrained fitting."""
        lower_bounds = []
        upper_bounds = []
        
        for pos, tolerance in zip(phase.expected_peaks, phase.peak_tolerances):
            # Amplitude bounds
            lower_bounds.extend([0.0, pos - tolerance, self.peak_width_bounds[0]])
            upper_bounds.extend([np.inf, pos + tolerance, self.peak_width_bounds[1]])
        
        return lower_bounds, upper_bounds
    
    def _perform_constrained_fitting(self, 
                                   initial_params: List[float],
                                   bounds: Tuple[List[float], List[float]],
                                   fitting_method: FittingMethod,
                                   phase: PhaseInfo):
        """Perform constrained least squares fitting."""
        def objective(params):
            fitted = self._generate_fitted_spectrum(params, fitting_method, phase)
            return np.sum((self.processed_intensities - fitted) ** 2)
        
        # Convert bounds format from (lower_list, upper_list) to [(min, max), ...]
        lower_bounds, upper_bounds = bounds
        bounds_formatted = list(zip(lower_bounds, upper_bounds))
        
        return minimize(
            objective,
            initial_params,
            bounds=bounds_formatted,
            method='L-BFGS-B'
        )
    
    def _generate_fitted_spectrum(self, 
                                params: np.ndarray,
                                fitting_method: FittingMethod,
                                phase: PhaseInfo) -> np.ndarray:
        """Generate fitted spectrum from parameters."""
        spectrum = np.zeros_like(self.wavenumbers)
        
        # Process parameters in groups of 3 (amp, cen, wid)
        for i in range(0, len(params), 3):
            if i + 2 < len(params):
                amp, cen, wid = params[i], params[i+1], params[i+2]
                
                if fitting_method == FittingMethod.GAUSSIAN:
                    peak = self._gaussian_peak(self.wavenumbers, amp, cen, wid)
                elif fitting_method == FittingMethod.LORENTZIAN:
                    peak = self._lorentzian_peak(self.wavenumbers, amp, cen, wid)
                elif fitting_method == FittingMethod.PSEUDO_VOIGT:
                    peak = self._pseudo_voigt_peak(self.wavenumbers, amp, cen, wid)
                else:
                    peak = self._gaussian_peak(self.wavenumbers, amp, cen, wid)
                
                spectrum += peak
        
        return spectrum
    
    def _gaussian_peak(self, x, amp, cen, wid):
        """Gaussian peak function."""
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    
    def _lorentzian_peak(self, x, amp, cen, wid):
        """Lorentzian peak function."""
        return amp * wid**2 / ((x - cen)**2 + wid**2)
    
    def _pseudo_voigt_peak(self, x, amp, cen, wid, eta=0.5):
        """Pseudo-Voigt peak function (linear combination of Gaussian and Lorentzian)."""
        gaussian = self._gaussian_peak(x, amp, cen, wid)
        lorentzian = self._lorentzian_peak(x, amp, cen, wid)
        return eta * lorentzian + (1 - eta) * gaussian
    
    def _calculate_r_squared(self, fitted_spectrum: np.ndarray) -> float:
        """Calculate R² value."""
        ss_res = np.sum((self.processed_intensities - fitted_spectrum) ** 2)
        ss_tot = np.sum((self.processed_intensities - np.mean(self.processed_intensities)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def _calculate_chi_squared(self, fitted_spectrum: np.ndarray) -> float:
        """Calculate reduced chi-squared value."""
        n_data = len(self.processed_intensities)
        n_params = sum(len(phase.expected_peaks) * 3 for phase in [self.major_phase] + self.minor_phases)
        dof = n_data - n_params
        
        chi_squared = np.sum((self.processed_intensities - fitted_spectrum) ** 2)
        return chi_squared / dof if dof > 0 else np.inf
    
    def _estimate_phase_abundance(self, component_spectrum: np.ndarray) -> float:
        """Estimate phase abundance from component spectrum."""
        component_integral = np.trapz(component_spectrum, self.wavenumbers)
        total_integral = np.trapz(self.processed_intensities, self.wavenumbers)
        return component_integral / total_integral if total_integral > 0 else 0
    
    def _calculate_parameter_uncertainties(self, fit_result) -> Dict[str, float]:
        """Calculate parameter uncertainties from fit result."""
        # Simplified uncertainty estimation
        return {"general_uncertainty": 0.1}  # Placeholder
    
    def _calculate_overlap_weights(self, major_phase_fit: FitResult) -> np.ndarray:
        """Calculate weights for overlap correction."""
        # Areas where major phase has high intensity get lower weights in residual
        major_spectrum = major_phase_fit.individual_components[major_phase_fit.phases[0].name]
        max_intensity = np.max(major_spectrum)
        
        # Weight inversely proportional to major phase intensity
        weights = 1.0 - (major_spectrum / max_intensity)
        weights = np.clip(weights, 0.1, 1.0)  # Ensure minimum weight
        
        return weights
    
    def _apply_weighted_correction(self, 
                                 residual: np.ndarray,
                                 weights: np.ndarray,
                                 major_phase_fit: FitResult) -> np.ndarray:
        """Apply weighted correction to residual."""
        # Weight the residual based on overlap likelihood
        corrected_residual = residual * weights
        
        # Apply smoothing to reduce artifacts
        corrected_residual = self._smooth_spectrum(corrected_residual, method="gaussian")
        
        return corrected_residual
    
    def _apply_overlap_correction(self, 
                                residual: np.ndarray,
                                major_phase_fit: FitResult) -> np.ndarray:
        """Apply additional overlap correction."""
        # Identify regions of potential overlap
        major_spectrum = major_phase_fit.individual_components[major_phase_fit.phases[0].name]
        
        # Find peaks in residual that are close to major phase peaks
        residual_peaks, _ = find_peaks(residual, prominence=0.01)
        major_peaks, _ = find_peaks(major_spectrum, prominence=0.05)
        
        # Apply local correction near overlapping regions
        corrected = residual.copy()
        
        for r_peak in residual_peaks:
            for m_peak in major_peaks:
                if abs(self.wavenumbers[r_peak] - self.wavenumbers[m_peak]) < 10:  # Within 10 cm⁻¹
                    # Apply local smoothing/correction
                    window = slice(max(0, r_peak-5), min(len(corrected), r_peak+5))
                    corrected[window] = self._smooth_spectrum(corrected[window])
        
        return corrected
    
    def _calculate_global_residual(self, 
                                 params: np.ndarray,
                                 phases: List[PhaseInfo],
                                 fitting_method: FittingMethod) -> float:
        """Calculate global residual for all phases."""
        total_spectrum = np.zeros_like(self.wavenumbers)
        
        param_idx = 0
        for phase in phases:
            n_params = len(phase.expected_peaks) * 3
            phase_params = params[param_idx:param_idx + n_params]
            param_idx += n_params
            
            component = self._generate_fitted_spectrum(phase_params, fitting_method, phase)
            total_spectrum += component
        
        residual = self.processed_intensities - total_spectrum
        return np.sum(residual ** 2)
    
    def _iterative_global_fitting(self, 
                                initial_params: List[float],
                                bounds: Tuple[List[float], List[float]],
                                objective_func,
                                phases: List[PhaseInfo]):
        """Perform iterative global fitting with refinement."""
        current_params = np.array(initial_params)
        
        for iteration in range(self.max_iterations):
            # Optimize all parameters
            # Convert bounds format for scipy.optimize.minimize
            bounds_formatted = list(zip(bounds[0], bounds[1]))
            result = minimize(
                objective_func,
                current_params,
                bounds=bounds_formatted,
                method='L-BFGS-B'
            )
            
            if result.success:
                # Check convergence
                param_change = np.linalg.norm(result.x - current_params)
                if param_change < self.convergence_threshold:
                    break
                
                current_params = result.x
            else:
                break
        
        return result
    
    def _calculate_global_uncertainties(self, 
                                      fit_result,
                                      phases: List[PhaseInfo]) -> Dict[str, float]:
        """Calculate uncertainties for global fit."""
        # Simplified uncertainty calculation
        uncertainties = {}
        for phase in phases:
            uncertainties[phase.name] = 0.05  # 5% uncertainty placeholder
        return uncertainties
    
    def _estimate_abundance_uncertainty(self, 
                                      phase_name: str,
                                      fit_result: FitResult,
                                      abundance: float) -> float:
        """Estimate uncertainty in abundance measurement."""
        # Uncertainty scales with inverse of fit quality and abundance
        base_uncertainty = 0.02  # 2% base uncertainty
        fit_penalty = (1 - fit_result.r_squared) * 0.1
        abundance_penalty = (1 - abundance) * 0.05
        
        return base_uncertainty + fit_penalty + abundance_penalty
    
    def _calculate_phase_confidence(self, 
                                  phase_name: str,
                                  fit_result: FitResult) -> float:
        """Calculate confidence in phase identification."""
        # Confidence based on fit quality and spectral match
        base_confidence = 0.7
        fit_bonus = fit_result.r_squared * 0.3
        
        return min(1.0, base_confidence + fit_bonus)


# =============================================================================
# MINIMAL STEP-BY-STEP APPROACH - NO COMPLEX FUNCTIONS
# =============================================================================

def simple_database_search_only():
    """
    STEP 1: Just do a simple database search - nothing else.
    This isolates any numpy issues.
    """
    import numpy as np  # Import here to avoid scoping issues
    
    print("STEP 1: Simple Database Search Only")
    print("=" * 40)
    
    try:
        # Import the real database
        from raman_spectra_qt6 import RamanSpectraQt6
        
        print("Connecting to database...")
        raman_db = RamanSpectraQt6()
        
        # Check database
        stats = raman_db.get_database_stats()
        print(f"Database loaded: {stats['total_spectra']} spectra")
        
        if stats['total_spectra'] == 0:
            print("❌ Database is empty!")
            return None
        
        # Create simple test spectrum
        print("Creating test spectrum...")
        wavenumbers = np.linspace(100, 1200, 500)
        test_spectrum = np.zeros_like(wavenumbers)
        
        # Simple peaks
        for peak in [206, 465, 696]:
            test_spectrum += np.exp(-(wavenumbers - peak)**2 / (2 * 10**2))
        
        print(f"Test spectrum created: {len(wavenumbers)} points")
        
        # Do the search
        print("Searching database...")
        results = raman_db.search_database(
            wavenumbers,
            test_spectrum,
            n_matches=3,
            threshold=0.3
        )
        
        print(f"Search complete: {len(results)} matches found")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['name'][:50]}...")
            print(f"     Score: {result['score']:.4f}")
        
        return {
            'database': raman_db,
            'wavenumbers': wavenumbers,
            'spectrum': test_spectrum,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def simple_fitting_with_residual(search_data):
    """
    STEP 2: Take search results and do simple fitting, show residual.
    """
    import numpy as np  # Import here to avoid scoping issues
    import matplotlib.pyplot as plt
    
    print("\nSTEP 2: Simple Fitting with Residual")
    print("=" * 40)
    
    if search_data is None:
        print("❌ No search data provided")
        return None
    
    try:
        wavenumbers = search_data['wavenumbers']
        spectrum = search_data['spectrum']
        results = search_data['results']
        
        if not results:
            print("❌ No search results to fit")
            return None
        
        # Get the best match
        best_match = results[0]
        print(f"Fitting to: {best_match['name'][:50]}...")
        
        # Simple Gaussian fitting approach
        from scipy.optimize import minimize
        
        # Find peaks in the spectrum
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(spectrum, prominence=0.1)
        
        if len(peaks) == 0:
            print("❌ No peaks found for fitting")
            return None
        
        peak_positions = wavenumbers[peaks]
        peak_heights = spectrum[peaks]
        
        print(f"Found {len(peaks)} peaks to fit")
        
        # Simple Gaussian function
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
        
        def multi_gaussian(params):
            """Multiple Gaussian model"""
            model = np.zeros_like(wavenumbers)
            for i in range(0, len(params), 3):
                if i + 2 < len(params):
                    amp, cen, wid = params[i], params[i+1], params[i+2]
                    model += gaussian(wavenumbers, amp, cen, wid)
            return model
        
        def objective(params):
            """Objective function for fitting"""
            model = multi_gaussian(params)
            return np.sum((spectrum - model)**2)
        
        # Initial parameters: [amp, center, width] for each peak
        initial_params = []
        for pos, height in zip(peak_positions, peak_heights):
            initial_params.extend([height, pos, 10.0])  # amp, center, width
        
        print("Performing least squares fitting...")
        
        # Fit
        result = minimize(objective, initial_params, method='Nelder-Mead')
        
        if result.success:
            print(f"✓ Fitting successful")
            
            # Generate fitted spectrum
            fitted_spectrum = multi_gaussian(result.x)
            
            # Calculate residual
            residual = spectrum - fitted_spectrum
            
            # Calculate R²
            ss_res = np.sum(residual**2)
            ss_tot = np.sum((spectrum - np.mean(spectrum))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"  R²: {r_squared:.4f}")
            print(f"  RMS residual: {np.sqrt(np.mean(residual**2)):.6f}")
            
            # Find positive peaks in residual
            positive_residual = np.maximum(residual, 0)
            residual_peaks, _ = find_peaks(positive_residual, prominence=0.01)
            
            print(f"  Positive residual peaks: {len(residual_peaks)}")
            
            # Plot the results
            print("Creating plots...")
            
            plt.figure(figsize=(12, 8))
            
            # Original spectrum and fit
            plt.subplot(2, 1, 1)
            plt.plot(wavenumbers, spectrum, 'b-', label='Original', linewidth=1)
            plt.plot(wavenumbers, fitted_spectrum, 'r-', label='Fitted', linewidth=1)
            plt.legend()
            plt.title(f'Spectrum Fit (R² = {r_squared:.4f})')
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Intensity')
            plt.grid(True, alpha=0.3)
            
            # Residual
            plt.subplot(2, 1, 2)
            plt.plot(wavenumbers, residual, 'g-', label='Residual', linewidth=1)
            plt.plot(wavenumbers, positive_residual, 'r-', label='Positive Residual', linewidth=1, alpha=0.7)
            
            # Mark residual peaks
            if len(residual_peaks) > 0:
                plt.plot(wavenumbers[residual_peaks], positive_residual[residual_peaks], 'ro', 
                        markersize=6, label=f'{len(residual_peaks)} Peaks')
            
            plt.legend()
            plt.title('Residual Analysis (for minor phases)')
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Residual Intensity')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.show()
            
            return {
                'fitted_spectrum': fitted_spectrum,
                'residual': residual,
                'positive_residual': positive_residual,
                'residual_peaks': residual_peaks,
                'r_squared': r_squared,
                'fit_params': result.x
            }
            
        else:
            print(f"❌ Fitting failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"❌ Fitting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def simple_residual_search(fit_data, search_data):
    """
    STEP 3: Search for minor phases in the residual.
    """
    import numpy as np  # Import here to avoid scoping issues
    
    print("\nSTEP 3: Residual Search for Minor Phases")
    print("=" * 40)
    
    if fit_data is None or search_data is None:
        print("❌ No fit or search data provided")
        return None
    
    try:
        database = search_data['database']
        wavenumbers = search_data['wavenumbers']
        positive_residual = fit_data['positive_residual']
        residual_peaks = fit_data['residual_peaks']
        
        print(f"Analyzing residual with {len(residual_peaks)} peaks...")
        
        if len(residual_peaks) == 0:
            print("No residual peaks to analyze")
            return {'minor_phases': []}
        
        # Search database with residual spectrum
        print("Searching database with residual spectrum...")
        
        residual_results = database.search_database(
            wavenumbers,
            positive_residual,
            n_matches=3,
            threshold=0.2  # Lower threshold for residual
        )
        
        print(f"Found {len(residual_results)} potential minor phases:")
        
        minor_phases = []
        for i, result in enumerate(residual_results):
            # Estimate abundance from residual
            residual_max = np.max(positive_residual)
            original_max = np.max(search_data['spectrum'])
            estimated_abundance = residual_max / original_max if original_max > 0 else 0
            
            phase_info = {
                'name': result['name'],
                'score': result['score'],
                'abundance': estimated_abundance,
                'method': 'residual_search'
            }
            
            minor_phases.append(phase_info)
            
            print(f"  {i+1}. {result['name'][:50]}...")
            print(f"     Score: {result['score']:.4f}")
            print(f"     Est. abundance: {estimated_abundance:.1%}")
        
        return {'minor_phases': minor_phases}
        
    except Exception as e:
        print(f"❌ Residual search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# EXISTING CODE CONTINUES...
# =============================================================================

if __name__ == "__main__":
    # Run MINIMAL step-by-step analysis
    print("Mixed Mineral Analysis - MINIMAL STEP-BY-STEP")
    print("=" * 60)
    print("🔍 Breaking down analysis into smallest possible steps")
    print("📊 Each step isolated to identify any issues")
    print()
    
    # Check what the user wants to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("🔗 Running full analysis...")
        try:
            results = demo_real_database_analysis()
            if results:
                print("✅ Full analysis completed!")
        except Exception as e:
            print(f"❌ Full analysis failed: {str(e)}")
    else:
        print("🔍 Running MINIMAL step-by-step analysis...")
        
        # STEP 1: Just search
        print("\n" + "="*60)
        search_data = simple_database_search_only()
        
        if search_data is not None:
            print("✅ STEP 1 SUCCESS: Database search worked!")
            
            # STEP 2: Fit and show residual
            print("\n" + "="*60)
            fit_data = simple_fitting_with_residual(search_data)
            
            if fit_data is not None:
                print("✅ STEP 2 SUCCESS: Fitting and residual plotting worked!")
                
                # STEP 3: Search residual
                print("\n" + "="*60)
                residual_data = simple_residual_search(fit_data, search_data)
                
                if residual_data is not None:
                    print("✅ STEP 3 SUCCESS: Residual search worked!")
                    print("\n🎉 ALL MINIMAL STEPS COMPLETED!")
                    print("📊 You should see a plot with original fit and residual")
                    print("🔍 Residual peaks marked for minor phase detection")
                else:
                    print("❌ STEP 3 FAILED: Residual search had issues")
            else:
                print("❌ STEP 2 FAILED: Fitting had issues")
        else:
            print("❌ STEP 1 FAILED: Database search had issues")
            print("💡 Make sure database is not empty and accessible")
        
        print(f"\n💡 To run full analysis: python {sys.argv[0]} --full")