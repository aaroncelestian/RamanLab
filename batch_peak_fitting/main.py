#!/usr/bin/env python3
"""
Main Controller for Modular Batch Peak Fitting
Coordinates all components using the new Phase 2 modular architecture
Updated to use UIManager for clean separation of concerns
"""

import numpy as np
from PySide6.QtWidgets import QDialog, QVBoxLayout, QApplication, QMessageBox
from PySide6.QtCore import QObject, Signal, QThread, QTimer
import os
import sys
from pathlib import Path

# Add robust path handling for imports
def setup_import_paths():
    """Set up import paths to ensure core modules can be found."""
    # Get the RamanLab root directory
    current_file = Path(__file__)
    batch_peak_fitting_dir = current_file.parent
    ramanlab_root = batch_peak_fitting_dir.parent
    
    # Add RamanLab root to Python path if not already present
    ramanlab_root_str = str(ramanlab_root)
    if ramanlab_root_str not in sys.path:
        sys.path.insert(0, ramanlab_root_str)
        print(f"Added to Python path: {ramanlab_root_str}")
    
    return ramanlab_root

# Set up paths before any other imports
try:
    ramanlab_root = setup_import_paths()
    print(f"✅ Import paths configured. RamanLab root: {ramanlab_root}")
except Exception as e:
    print(f"⚠️  Warning: Could not configure import paths: {e}")

# Import local modules
try:
    from .core.data_processor import DataProcessor
    LOCAL_IMPORTS_AVAILABLE = True
    print("✅ Local imports successful")
except ImportError as e:
    print(f"⚠️  Warning: Local imports failed: {e}")
    LOCAL_IMPORTS_AVAILABLE = False

# Import core modules with robust error handling
try:
    from core.peak_fitting import PeakFitter, auto_find_peaks, baseline_correct_spectrum
    CORE_AVAILABLE = True
    print("✅ Core peak fitting imports successful")
except ImportError as e:
    print(f"⚠️  Warning: Core peak fitting not available: {e}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    CORE_AVAILABLE = False

# Import UI manager
try:
    from .ui.ui_manager import UIManager
    UI_MANAGER_AVAILABLE = True
    print("✅ UI Manager import successful")
except ImportError as e:
    print(f"⚠️  Warning: UI Manager not available: {e}")
    UI_MANAGER_AVAILABLE = False

# Create availability check function
def check_module_dependencies():
    """Check if all required modules and files are present."""
    print("\n🔍 Checking module dependencies...")
    
    required_files = [
        'core/__init__.py',
        'core/config_manager.py',
        'core/peak_fitting.py',
        'batch_peak_fitting/__init__.py',
        'batch_peak_fitting/main.py',
        'batch_peak_fitting/core/__init__.py',
        'batch_peak_fitting/core/data_processor.py',
        'batch_peak_fitting/ui/__init__.py',
        'batch_peak_fitting/ui/ui_manager.py'
    ]
    
    missing_files = []
    ramanlab_root = Path(__file__).parent.parent
    
    for file_path in required_files:
        full_path = ramanlab_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} required files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files are present")
        return True

# Run dependency check
check_result = check_module_dependencies()

# Module availability flags for graceful degradation
MODULE_STATUS = {
    'core': CORE_AVAILABLE,
    'local_imports': LOCAL_IMPORTS_AVAILABLE,
    'ui_manager': UI_MANAGER_AVAILABLE,
    'dependencies_complete': check_result
}

print(f"\n📊 Module Status Summary:")
for module, status in MODULE_STATUS.items():
    status_icon = "✅" if status else "❌"
    print(f"   {status_icon} {module}: {status}")

# Only proceed with class definitions if minimum requirements are met
if not LOCAL_IMPORTS_AVAILABLE:
    print("❌ Cannot continue: Local imports failed")
    
if not UI_MANAGER_AVAILABLE:
    print("❌ Cannot continue: UI Manager not available")


class BatchPeakFittingAdapter(QObject):
    """
    Adapter to bridge the interface between the centralized PeakFitter and the batch processing system.
    This maintains backward compatibility while using the centralized peak fitting routines.
    Enhanced with robust import handling and fallback functionality.
    """
    
    # Qt Signals for compatibility with existing batch system
    fitting_completed = Signal(dict)
    peaks_detected = Signal(np.ndarray)
    peaks_fitted = Signal(dict)
    background_calculated = Signal(np.ndarray)
    fitting_progress = Signal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Check if core functionality is available
        self.core_available = CORE_AVAILABLE
        
        # Use the centralized PeakFitter if available
        if self.core_available:
            try:
                self.core_peak_fitter = PeakFitter()
                print("✅ Centralized PeakFitter initialized")
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize PeakFitter: {e}")
                self.core_peak_fitter = None
                self.core_available = False
        else:
            self.core_peak_fitter = None
            print("❌ Core peak fitting not available - using fallback methods")
        
        # Default parameters for backward compatibility
        self.current_model = "Gaussian"
        self.background_method = "polynomial"
        self.peak_height = 0.15  # 15% of max intensity
        self.peak_distance = 20.0  # minimum distance between peaks
        self.peak_prominence = 0.20  # 20% of intensity range
        
        # ALS parameters for compatibility
        self.als_lambda = 1e5
        self.als_p = 0.01
        self.als_niter = 10
        
        # Store last results for compatibility
        self.last_background = None
        self.last_fit_results = []
        self.reference_peaks = []
    
    def set_model(self, model_name):
        """Set the peak fitting model (adapted interface)"""
        # Map batch system model names to centralized system names
        model_mapping = {
            "Gaussian": "Gaussian",
            "Lorentzian": "Lorentzian", 
            "Pseudo-Voigt": "Pseudo-Voigt",
            "Voigt": "Voigt",
            "Asymmetric Voigt": "Asymmetric Voigt"  # Now properly supported!
        }
        self.current_model = model_mapping.get(model_name, "Gaussian")
    
    def set_background_method(self, method):
        """Set background subtraction method"""
        # Extract method name from UI text if needed
        if "ALS" in method:
            self.background_method = "ALS"  # Use ALS directly
        elif "Linear" in method:
            self.background_method = "linear"
        elif "Polynomial" in method:
            self.background_method = "polynomial"
        else:
            self.background_method = "ALS"  # Default to ALS
    
    def set_peak_detection_parameters(self, height, distance, prominence):
        """Set peak detection parameters"""
        self.peak_height = height
        self.peak_distance = distance
        self.peak_prominence = prominence
    
    def set_als_parameters(self, lambda_val=None, p_val=None, niter_val=None):
        """Set ALS parameters (for compatibility)"""
        if lambda_val is not None:
            self.als_lambda = lambda_val
        if p_val is not None:
            self.als_p = p_val
        if niter_val is not None:
            self.als_niter = niter_val
    
    def find_peaks_auto(self, wavenumbers, intensities):
        """Automatically find peaks (adapted interface with fallback)"""
        try:
            if self.core_available and CORE_AVAILABLE:
                # Use centralized auto_find_peaks function
                peak_positions = auto_find_peaks(
                    wavenumbers, 
                    intensities,
                    height_threshold=self.peak_height,
                    distance=self.peak_distance
                )
                
                # Convert positions to indices for backward compatibility
                peak_indices = []
                for pos in peak_positions:
                    idx = np.argmin(np.abs(wavenumbers - pos))
                    peak_indices.append(idx)
                
                peak_indices = np.array(peak_indices, dtype=int)
                properties = {}  # Empty properties for compatibility
                
                # Emit signal for UI updates
                self.peaks_detected.emit(peak_indices)
                
                print(f"BatchPeakFittingAdapter: Found {len(peak_indices)} peaks using centralized detection")
                return peak_indices, properties
            else:
                # Fallback peak detection using scipy
                return self._fallback_find_peaks(wavenumbers, intensities)
            
        except Exception as e:
            print(f"Error in peak detection: {e}, falling back to scipy method")
            return self._fallback_find_peaks(wavenumbers, intensities)
    
    def _fallback_find_peaks(self, wavenumbers, intensities):
        """Fallback peak detection using scipy when core module is not available"""
        try:
            from scipy.signal import find_peaks
            
            # Calculate threshold values
            max_intensity = np.max(intensities)
            height_threshold = self.peak_height * max_intensity
            prominence_threshold = self.peak_prominence * max_intensity
            
            # Convert distance from wavenumber units to index units
            wn_spacing = np.mean(np.diff(wavenumbers))
            distance_indices = max(1, int(self.peak_distance / wn_spacing))
            
            # Find peaks using scipy
            peak_indices, properties = find_peaks(
                intensities, 
                height=height_threshold,
                distance=distance_indices,
                prominence=prominence_threshold
            )
            
            # Emit signal for UI updates
            self.peaks_detected.emit(peak_indices)
            
            print(f"BatchPeakFittingAdapter: Found {len(peak_indices)} peaks using fallback scipy detection")
            return peak_indices, properties
            
        except ImportError:
            print("❌ Error: scipy not available for fallback peak detection")
            return np.array([]), {}
        except Exception as e:
            print(f"❌ Error in fallback peak detection: {e}")
            return np.array([]), {}
    
    def calculate_background(self, wavenumbers, intensities):
        """Calculate background using the selected method with enhanced ALS"""
        try:
            print(f"BatchPeakFittingAdapter: Calculating background using method '{self.background_method}'")
            print(f"ALS parameters: lambda={self.als_lambda}, p={self.als_p}, niter={self.als_niter}")
            
            if "ALS" in self.background_method or self.background_method == "ALS":
                # Use enhanced ALS method directly - returns the actual baseline
                background = self._baseline_als(intensities, self.als_lambda, self.als_p, self.als_niter)
                print(f"ALS background calculated: min={np.min(background):.2f}, max={np.max(background):.2f}")
            else:
                # Use centralized baseline correction for other methods
                if self.background_method == "polynomial":
                    # For polynomial, fit to endpoints and return the fitted baseline
                    edge_fraction = 0.1
                    n_edge = int(len(wavenumbers) * edge_fraction)
                    
                    # Use edge points for baseline estimation
                    x_baseline = np.concatenate([wavenumbers[:n_edge], wavenumbers[-n_edge:]])
                    y_baseline = np.concatenate([intensities[:n_edge], intensities[-n_edge:]])
                    
                    # Fit polynomial
                    poly_coeffs = np.polyfit(x_baseline, y_baseline, 3)
                    background = np.polyval(poly_coeffs, wavenumbers)
                    print(f"Polynomial background calculated")
                
                elif self.background_method == "linear":
                    # Simple linear baseline
                    background = np.linspace(intensities[0], intensities[-1], len(intensities))
                    print(f"Linear background calculated")
                
                else:
                    print(f"Unknown background method: {self.background_method}, using ALS")
                    background = self._baseline_als(intensities, self.als_lambda, self.als_p, self.als_niter)
            
            self.last_background = background
            
            # Emit signal for UI updates
            self.background_calculated.emit(background)
            
            return background
            
        except Exception as e:
            print(f"Error calculating background: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros_like(intensities)
    
    def _baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares Smoothing for baseline correction.
        
        Parameters:
        -----------
        y : array-like
            Input spectrum.
        lam : float
            Smoothness parameter (default: 1e5).
        p : float
            Asymmetry parameter (default: 0.01).
        niter : int
            Number of iterations (default: 10).
            
        Returns:
        --------
        array-like
            Estimated baseline.
        """
        try:
            from scipy.sparse import csc_matrix
            from scipy.sparse.linalg import spsolve
            
            print(f"ALS: Processing {len(y)} points with lambda={lam}, p={p}, niter={niter}")
            
            # Ensure input is a numpy array
            y = np.array(y, dtype=float)
            L = len(y)
            
            # Handle edge cases
            if L < 3:
                print("ALS: Too few points, returning linear baseline")
                return np.linspace(y[0], y[-1], L)
            
            # Create second-order difference matrix
            D = csc_matrix(np.diff(np.eye(L), 2))
            
            # Initialize weights
            w = np.ones(L)
            
            print(f"ALS: Starting iterations...")
            for i in range(niter):
                # Create weight matrix
                W = csc_matrix((w, (np.arange(L), np.arange(L))))
                
                # Solve the linear system
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w * y)
                
                # Update weights asymmetrically
                w_new = p * (y > z) + (1 - p) * (y <= z)
                
                # Check for convergence
                weight_change = np.mean(np.abs(w_new - w))
                w = w_new
                
                if i % 5 == 0:  # Log every 5 iterations
                    print(f"ALS: Iteration {i+1}/{niter}, weight change: {weight_change:.6f}")
                
                # Early convergence check
                if weight_change < 1e-6 and i > 2:
                    print(f"ALS: Converged early at iteration {i+1}")
                    break
            
            # Ensure z is a proper numpy array
            z = np.array(z).flatten()
            
            print(f"ALS: Completed. Baseline range: {np.min(z):.2f} to {np.max(z):.2f}")
            
            return z
            
        except ImportError:
            print("Warning: scipy not available, using enhanced polynomial baseline")
            # Enhanced polynomial fallback
            x = np.arange(len(y))
            
            # Use robust polynomial fitting
            try:
                # Remove outliers for better polynomial fit
                median_y = np.median(y)
                mad = np.median(np.abs(y - median_y))
                outlier_mask = np.abs(y - median_y) < 3 * mad
                
                if np.sum(outlier_mask) > len(y) // 2:
                    # Fit polynomial to non-outlier points
                    coeffs = np.polyfit(x[outlier_mask], y[outlier_mask], 3)
                else:
                    # Fallback to simple polynomial
                    coeffs = np.polyfit(x, y, 2)
                
                baseline = np.polyval(coeffs, x)
                print(f"Polynomial fallback baseline: {np.min(baseline):.2f} to {np.max(baseline):.2f}")
                return baseline
                
            except Exception:
                print("Polynomial fallback failed, using linear baseline")
                return np.linspace(y[0], y[-1], len(y))
        except Exception as e:
            print(f"Error in ALS baseline calculation: {e}")
            import traceback
            traceback.print_exc()
            return np.linspace(y[0], y[-1], len(y))
    
    def fit_peaks(self, wavenumbers, intensities, peak_positions, apply_background=False):
        """
        Fit peaks using centralized peak fitting (adapted interface)
        
        Args:
            wavenumbers: Array of wavenumber values
            intensities: Array of intensity values  
            peak_positions: List/array of peak positions to fit
            apply_background: If True, apply background subtraction before fitting
        """
        try:
            if len(peak_positions) == 0:
                result = {"success": False, "error": "No peaks to fit"}
                self.fitting_completed.emit(result)
                self.peaks_fitted.emit(result)
                return result
            
            # FIXED: Only apply background correction if explicitly requested
            if apply_background:
                background = self.calculate_background(wavenumbers, intensities)
                fitting_intensities = intensities - background
                print("Peak fitting: Using background-corrected data")
            else:
                fitting_intensities = intensities.copy()
                print("Peak fitting: Using raw spectrum data (no background correction)")
            
            # Convert peak indices to positions if needed
            if len(peak_positions) > 0 and isinstance(peak_positions[0], (int, np.integer)):
                peak_positions_wavenumber = [wavenumbers[idx] for idx in peak_positions if 0 <= idx < len(wavenumbers)]
            else:
                peak_positions_wavenumber = list(peak_positions)
            
            # Use centralized peak fitting
            fitted_peaks = self.core_peak_fitter.fit_multiple_peaks(
                wavenumbers,
                fitting_intensities,
                peak_positions_wavenumber,
                shape=self.current_model,
                window_width=50.0
            )
            
            if len(fitted_peaks) == 0:
                result = {"success": False, "error": f"No peaks could be fitted from {len(peak_positions_wavenumber)} detected peaks. Try adjusting detection parameters or peak positions."}
                self.fitting_completed.emit(result)
                self.peaks_fitted.emit(result)
                return result
            
            print(f"Peak fitting: Successfully fitted {len(fitted_peaks)} out of {len(peak_positions_wavenumber)} detected peaks")
            
            # Convert results to format expected by batch system
            fit_params = []
            for peak in fitted_peaks:
                # Each peak: [amplitude, center, width, ...]
                params = [peak.amplitude, peak.position, peak.width]
                # Add extra parameters for complex models if needed
                if self.current_model == "Pseudo-Voigt":
                    eta = getattr(peak, 'eta', 0.5)  # default eta value
                    params.append(eta)
                elif self.current_model == "Voigt":
                    gamma = getattr(peak, 'gamma', peak.width / 2)  # default gamma value
                    params.append(gamma)
                elif self.current_model == "Asymmetric Voigt":
                    gamma = getattr(peak, 'gamma', peak.width / 2)  # default gamma value
                    alpha = getattr(peak, 'alpha', 0.0)  # default alpha value
                    params.extend([gamma, alpha])
                fit_params.extend(params)
            
            # Calculate overall R-squared
            total_fitted = np.zeros_like(wavenumbers)
            for peak in fitted_peaks:
                if self.current_model == "Gaussian":
                    peak_curve = self.core_peak_fitter.gaussian(wavenumbers, peak.amplitude, peak.position, peak.width)
                elif self.current_model == "Lorentzian":
                    peak_curve = self.core_peak_fitter.lorentzian(wavenumbers, peak.amplitude, peak.position, peak.width)
                elif self.current_model == "Pseudo-Voigt":
                    eta = getattr(peak, 'eta', 0.5)  # default eta value
                    peak_curve = self.core_peak_fitter.pseudo_voigt(wavenumbers, peak.amplitude, peak.position, peak.width, eta)
                elif self.current_model == "Voigt":
                    gamma = getattr(peak, 'gamma', peak.width / 2)  # default gamma value
                    peak_curve = self.core_peak_fitter.voigt(wavenumbers, peak.amplitude, peak.position, peak.width, gamma)
                elif self.current_model == "Asymmetric Voigt":
                    gamma = getattr(peak, 'gamma', peak.width / 2)  # default gamma value
                    alpha = getattr(peak, 'alpha', 0.0)  # default alpha value
                    peak_curve = self.core_peak_fitter.asymmetric_voigt(wavenumbers, peak.amplitude, peak.position, peak.width, gamma, alpha)
                else:
                    # For other models, use the fitted curve approximation
                    peak_curve = peak.amplitude * np.exp(-((wavenumbers - peak.position) / peak.width) ** 2)
                total_fitted += peak_curve
            
            # Calculate R-squared using the correct data
            ss_res = np.sum((fitting_intensities - total_fitted) ** 2)
            ss_tot = np.sum((fitting_intensities - np.mean(fitting_intensities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r_squared = max(0.0, min(1.0, r_squared))
            
            # Store results
            self.last_fit_results = fitted_peaks
            
            # Calculate visualization data
            individual_peak_curves = []
            for peak in fitted_peaks:
                if self.current_model == "Gaussian":
                    peak_curve = self.core_peak_fitter.gaussian(wavenumbers, peak.amplitude, peak.position, peak.width)
                elif self.current_model == "Lorentzian":
                    peak_curve = self.core_peak_fitter.lorentzian(wavenumbers, peak.amplitude, peak.position, peak.width)
                elif self.current_model == "Pseudo-Voigt":
                    eta = getattr(peak, 'eta', 0.5)
                    peak_curve = self.core_peak_fitter.pseudo_voigt(wavenumbers, peak.amplitude, peak.position, peak.width, eta)
                elif self.current_model == "Voigt":
                    gamma = getattr(peak, 'gamma', peak.width / 2)
                    peak_curve = self.core_peak_fitter.voigt(wavenumbers, peak.amplitude, peak.position, peak.width, gamma)
                else:
                    # Fallback to Gaussian
                    peak_curve = self.core_peak_fitter.gaussian(wavenumbers, peak.amplitude, peak.position, peak.width)
                individual_peak_curves.append(peak_curve)
            
            # Calculate residuals
            residuals = fitting_intensities - total_fitted
            
            result = {
                "success": True,
                "fit_params": fit_params,
                "r_squared": r_squared,
                "model": self.current_model,
                "n_peaks": len(fitted_peaks),
                "peak_positions": peak_positions_wavenumber,
                "fitted_peaks": fitted_peaks,  # Include detailed results
                "fitted_curve": total_fitted,
                "individual_peaks": individual_peak_curves,
                "residuals": residuals
            }
            
            # Emit signals for UI updates
            self.fitting_completed.emit(result)
            self.peaks_fitted.emit(result)  # Also emit peaks_fitted for components that expect it
            
            print(f"BatchPeakFittingAdapter: Successfully fitted {len(fitted_peaks)} peaks with R² = {r_squared:.4f}")
            return result
            
        except Exception as e:
            print(f"Error in centralized peak fitting: {e}")
            result = {"success": False, "error": str(e)}
            self.fitting_completed.emit(result)
            self.peaks_fitted.emit(result)
            return result
    
    def detect_peaks(self, wavenumbers, intensities):
        """Simple peak detection interface for batch processing"""
        peak_indices, _ = self.find_peaks_auto(wavenumbers, intensities)
        return peak_indices if len(peak_indices) > 0 else None
    
    def set_reference_peaks(self, peak_positions, fit_params=None):
        """Set reference peaks for batch processing"""
        self.reference_peaks = peak_positions.copy() if peak_positions is not None else []
        # Note: fit_params not directly used in centralized system but stored for compatibility
    
    def clear_reference_peaks(self):
        """Clear reference peaks"""
        self.reference_peaks = []
    
    def reset(self):
        """Reset the peak fitter state"""
        self.last_background = None
        self.last_fit_results = []
        self.reference_peaks = []
    
    def get_status(self):
        """Get current status (for compatibility)"""
        return {
            'model': self.current_model,
            'background_method': self.background_method,
            'n_reference_peaks': len(self.reference_peaks),
            'last_fit_successful': len(self.last_fit_results) > 0
        }
    
    def get_parameters(self):
        """Get current fitting parameters"""
        return {
            "model": self.current_model,
            "background_method": self.background_method,
            "peak_height": self.peak_height,
            "peak_distance": self.peak_distance,
            "peak_prominence": self.peak_prominence,
            "als_lambda": self.als_lambda,
            "als_p": self.als_p,
            "als_niter": self.als_niter
        }


class BatchPeakFittingMainController(QDialog):
    """
    Main controller for the modular batch peak fitting system.
    Orchestrates all components using clean dependency injection.
    
    Phase 2 Architecture:
    - UIManager handles all interface coordination
    - DataProcessor handles file operations and spectrum management
    - PeakFitter handles mathematical operations (now using centralized module)
    - MainController orchestrates component interactions
    """
    
    # Signals for external communication
    processing_started = Signal()
    processing_finished = Signal(dict)
    spectrum_changed = Signal(dict)
    
    def __init__(self, parent=None, wavenumbers=None, intensities=None):
        super().__init__(parent)
        
        # Initialize core components
        self.data_processor = DataProcessor()
        # REFACTORED: Use adapter for centralized peak fitting
        self.peak_fitter = BatchPeakFittingAdapter()
        self.ui_manager = UIManager()
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.current_batch_results = []
        
        # Initialize with provided data if available
        if wavenumbers is not None and intensities is not None:
            self.data_processor.set_current_spectrum(wavenumbers, intensities)
        
        # Setup the complete interface
        self.setup_application()
        
        print("MainController: Initialization complete with centralized peak fitting")  # Debug
    
    def setup_application(self):
        """Setup the complete application with modular architecture"""
        # Configure main dialog
        self.setWindowTitle("Batch Peak Fitting - Modular Architecture v2.0")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Initialize UI Manager with core components
        self.ui_manager.initialize(
            self.data_processor,
            self.peak_fitter,
            self  # MainController reference
        )
        
        # Setup main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create and add UI manager's main widget
        ui_widget = self.ui_manager.setup_ui(self)
        main_layout.addWidget(ui_widget)
        
        # Connect component signals
        self._connect_component_signals()
        
        print("MainController: Application setup complete")  # Debug
    
    def _connect_component_signals(self):
        """Connect signals between all components"""
        
        # Connect UI Manager to Main Controller
        self.ui_manager.action_requested.connect(self._handle_ui_action)
        self.ui_manager.status_updated.connect(self._handle_status_update)
        
        # Connect Data Processor signals
        self.data_processor.spectrum_loaded.connect(self._on_spectrum_loaded)
        self.data_processor.spectra_list_changed.connect(self._on_spectra_list_changed)
        self.data_processor.current_spectrum_changed.connect(self._on_current_spectrum_changed)
        self.data_processor.file_operation_completed.connect(self._on_file_operation_completed)
        
        # Connect Peak Fitter signals
        self.peak_fitter.fitting_completed.connect(self._on_fitting_completed)
        self.peak_fitter.peaks_detected.connect(self._on_peaks_detected)
        self.peak_fitter.background_calculated.connect(self._on_background_calculated)
        self.peak_fitter.fitting_progress.connect(self._on_fitting_progress)
        
        # Connect Visualization Manager signals if available
        viz_manager = self.ui_manager.get_visualization_manager()
        if viz_manager:
            viz_manager.plot_updated.connect(self._on_plot_updated)
            viz_manager.plot_clicked.connect(self._on_plot_clicked)
            viz_manager.visualization_ready.connect(self._on_visualization_ready)
        
        print("MainController: Component signals connected")  # Debug
    
    def _handle_ui_action(self, action_name, parameters):
        """Handle action requests from the UI"""
        source_tab = parameters.get('source_tab', 'Unknown')
        print(f"MainController: Handling action '{action_name}' from {source_tab}")  # Debug
        
        try:
            # Route actions to appropriate handlers
            if action_name.startswith('files_') or action_name in ['add_files', 'remove_files']:
                self._handle_file_action(action_name, parameters)
            elif action_name.startswith('spectrum_') or action_name.startswith('navigate_') or action_name == 'load_file':
                self._handle_spectrum_action(action_name, parameters)
            elif action_name.startswith('peak_') or action_name.startswith('detect_') or action_name.startswith('fit_') or action_name.startswith('add_manual_') or action_name.startswith('delete_manual_') or action_name.startswith('clear_manual_') or action_name.startswith('remove_manual_') or action_name.startswith('set_reference') or action_name.startswith('clear_reference') or action_name.startswith('enable_interactive'):
                self._handle_peak_action(action_name, parameters)
            elif action_name.startswith('background_') or action_name.startswith('apply_') or action_name.startswith('preview_') or action_name.startswith('clear_'):
                self._handle_background_action(action_name, parameters)
            elif action_name.startswith('batch_') or action_name == 'start_batch_processing' or action_name == 'stop_batch_processing':
                self._handle_batch_action(action_name, parameters)
            elif action_name.startswith('export_') or action_name.startswith('save_') or action_name.startswith('load_'):
                self._handle_io_action(action_name, parameters)
            elif action_name.startswith('session_'):
                self._handle_session_action(action_name, parameters)
            elif action_name.startswith('plot_') or action_name.startswith('visualization_'):
                self._handle_visualization_action(action_name, parameters)
            else:
                self._handle_generic_action(action_name, parameters)
                
        except Exception as e:
            print(f"MainController: Error handling action '{action_name}': {e}")  # Debug
            self._show_error_message(f"Error handling {action_name}", str(e))
    
    def _handle_file_action(self, action_name, parameters):
        """Handle file-related actions"""
        if action_name == 'add_files':
            # Add files to data processor
            files = parameters.get('files', [])
            if files:
                files_added = self.data_processor.add_files(files)
                print(f"MainController: Added {files_added} files")
                self.ui_manager.update_all_tabs_from_data_processor()
            
        elif action_name == 'remove_files':
            # Remove files from data processor
            files_to_remove = parameters.get('files', [])
            if files_to_remove:
                # Convert file paths to indices
                indices_to_remove = []
                for file_path in files_to_remove:
                    file_list = self.data_processor.get_file_list()
                    # Find by basename since UI shows basename but stores full path
                    for i, full_path in enumerate(file_list):
                        if os.path.basename(full_path) == file_path or full_path == file_path:
                            indices_to_remove.append(i)
                
                if indices_to_remove:
                    files_removed = self.data_processor.remove_files(indices_to_remove)
                    print(f"MainController: Removed {files_removed} files")
                    self.ui_manager.update_all_tabs_from_data_processor()
            
        elif action_name == 'files_added':
            # Files were added, update UI
            self.ui_manager.update_all_tabs_from_data_processor()
            
        elif action_name == 'files_removed':
            # Files were removed, update UI
            self.ui_manager.update_all_tabs_from_data_processor()
    
    def _handle_spectrum_action(self, action_name, parameters):
        """Handle spectrum-related actions"""
        if action_name == 'spectrum_selected':
            index = parameters.get('index', 0)
            success = self.data_processor.load_spectrum(index)
            if not success:
                self._show_error_message("Spectrum Loading", f"Failed to load spectrum at index {index}")
                
        elif action_name == 'spectrum_navigated':
            # Navigation was handled by data processor, just update UI
            self.ui_manager.update_all_tabs_from_data_processor()
            
        elif action_name == 'navigate_first':
            success = self.data_processor.navigate_spectrum(0)  # 0 = first
            if success:
                self.ui_manager.update_all_tabs_from_data_processor()
                
        elif action_name == 'navigate_previous':
            success = self.data_processor.navigate_spectrum(-1)  # -1 = previous
            if success:
                self.ui_manager.update_all_tabs_from_data_processor()
                
        elif action_name == 'navigate_next':
            success = self.data_processor.navigate_spectrum(1)  # 1 = next
            if success:
                self.ui_manager.update_all_tabs_from_data_processor()
                
        elif action_name == 'navigate_last':
            success = self.data_processor.navigate_spectrum(-2)  # -2 = last
            if success:
                self.ui_manager.update_all_tabs_from_data_processor()
                
        elif action_name == 'load_file':
            file_path = parameters.get('file_path')
            if file_path:
                # Find the index of this file and load it
                file_list = self.data_processor.get_file_list()
                for i, full_path in enumerate(file_list):
                    if os.path.basename(full_path) == file_path or full_path == file_path:
                        success = self.data_processor.load_spectrum(i)
                        if success:
                            self.ui_manager.update_all_tabs_from_data_processor()
                        else:
                            self._show_error_message("File Loading", f"Failed to load {file_path}")
                        break
    
    def _handle_peak_action(self, action_name, parameters):
        """Handle peak-related actions"""
        if action_name == 'detect_peaks':
            # Get current spectrum data
            spectrum_data = self.data_processor.get_current_spectrum()
            if spectrum_data:
                # Extract detection parameters (now properly converted from UI)
                height = parameters.get('height', 0.15)  # 15% of max intensity
                distance = parameters.get('distance', 20)
                prominence = parameters.get('prominence', 0.20)  # 20% of intensity range
                
                # Set detection parameters in peak fitter
                self.peak_fitter.set_peak_detection_parameters(height, distance, prominence)
                
                # Perform peak detection
                peaks, properties = self.peak_fitter.find_peaks_auto(
                    spectrum_data['wavenumbers'],
                    spectrum_data['intensities']
                )
                
                if len(peaks) > 0:
                    # Update data processor with detected peaks
                    self.data_processor.peaks = peaks
                    
                    # Update peak count in UI
                    peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
                    if peaks_tab and hasattr(peaks_tab, 'peak_count_label'):
                        peaks_tab.peak_count_label.setText(f"Peaks found: {len(peaks)}")
                        # Also update fitting status if the method exists
                        if hasattr(peaks_tab, '_update_fitting_status'):
                            peaks_tab._update_fitting_status(len(peaks))
                    
                    # OPTIMIZED: Only update current spectrum plot for peak detection
                    viz_manager = self.ui_manager.get_visualization_manager()
                    if viz_manager:
                        viz_manager.update_plot("current_spectrum")
                    
                    print(f"MainController: Detected {len(peaks)} peaks")
                else:
                    print("MainController: No peaks detected with current parameters")
                    # Update UI to show no peaks found
                    peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
                    if peaks_tab and hasattr(peaks_tab, 'peak_count_label'):
                        peaks_tab.peak_count_label.setText("Peaks found: 0")
                        if hasattr(peaks_tab, '_update_fitting_status'):
                            peaks_tab._update_fitting_status(0)
            else:
                self._show_error_message("Peak Detection", "No spectrum data available")
                
        elif action_name == 'fit_peaks':
            # Fit peaks to current spectrum
            spectrum_data = self.data_processor.get_current_spectrum()
            if spectrum_data and len(spectrum_data.get('peaks', [])) > 0:
                # Get peak positions from data processor
                peak_positions = spectrum_data['peaks']
                
                # Get background correction option from parameters
                apply_background = parameters.get('apply_background', False)
                
                # Perform peak fitting
                result = self.peak_fitter.fit_peaks(
                    spectrum_data['wavenumbers'],
                    spectrum_data['intensities'],
                    peak_positions,
                    apply_background=apply_background
                )
                
                if result.get('success', False):
                    # Update data processor with fitting results
                    self.data_processor.fit_params = result.get('fit_params', [])
                    self.data_processor.fit_result = result
                    
                    # Calculate fitted curves for visualization
                    fitted_peaks = result.get('fitted_peaks', [])
                    if fitted_peaks and len(fitted_peaks) > 0:
                        # Calculate total fitted curve and individual peak curves
                        wavenumbers = spectrum_data['wavenumbers']
                        total_fitted = np.zeros_like(wavenumbers)
                        individual_peaks = []
                        
                        for peak in fitted_peaks:
                            # Calculate individual peak curve based on peak shape
                            if peak.shape == "Gaussian":
                                peak_curve = self.peak_fitter.core_peak_fitter.gaussian(wavenumbers, peak.amplitude, peak.position, peak.width)
                            elif peak.shape == "Lorentzian":
                                peak_curve = self.peak_fitter.core_peak_fitter.lorentzian(wavenumbers, peak.amplitude, peak.position, peak.width)
                            elif peak.shape == "Pseudo-Voigt":
                                eta = getattr(peak, 'eta', 0.5)
                                peak_curve = self.peak_fitter.core_peak_fitter.pseudo_voigt(wavenumbers, peak.amplitude, peak.position, peak.width, eta)
                            elif peak.shape == "Voigt":
                                gamma = getattr(peak, 'gamma', peak.width / 2)
                                peak_curve = self.peak_fitter.core_peak_fitter.voigt(wavenumbers, peak.amplitude, peak.position, peak.width, gamma)
                            else:
                                # Fallback to Gaussian
                                peak_curve = self.peak_fitter.core_peak_fitter.gaussian(wavenumbers, peak.amplitude, peak.position, peak.width)
                            
                            individual_peaks.append(peak_curve)
                            total_fitted += peak_curve
                        
                        # Calculate residuals
                        fitting_intensities = spectrum_data['intensities']
                        if apply_background and self.peak_fitter.last_background is not None:
                            fitting_intensities = fitting_intensities - self.peak_fitter.last_background
                        
                        residuals = fitting_intensities - total_fitted
                        
                        # Update fit_result with visualization data
                        result['fitted_curve'] = total_fitted
                        result['individual_peaks'] = individual_peaks
                        result['residuals'] = residuals
                        self.data_processor.fit_result = result
                    
                    # OPTIMIZED: Only update current spectrum plot for peak fitting
                    viz_manager = self.ui_manager.get_visualization_manager()
                    if viz_manager:
                        viz_manager.update_plot("current_spectrum")
                    
                    print(f"MainController: Peak fitting successful with R² = {result.get('r_squared', 0):.4f}")
                else:
                    print(f"MainController: Peak fitting failed: {result.get('error', 'Unknown error')}")
                    self._show_error_message("Peak Fitting", f"Fitting failed: {result.get('error', 'Unknown error')}")
            else:
                self._show_error_message("Peak Fitting", "No peaks detected. Please detect peaks first.")
                
        elif action_name == 'clear_peaks':
            # Clear both automatic and manual peaks
            self.data_processor.peaks = np.array([], dtype=int)
            self.data_processor.clear_manual_peaks()
            
            # Update peak count in UI
            peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
            if peaks_tab and hasattr(peaks_tab, 'peak_count_label'):
                peaks_tab.peak_count_label.setText("Peaks found: 0")
            
            # OPTIMIZED: Only update current spectrum plot for clearing peaks
            viz_manager = self.ui_manager.get_visualization_manager()
            if viz_manager:
                viz_manager.update_plot("current_spectrum")
            
        elif action_name == 'set_reference':
            # Set current peaks as reference for batch processing
            spectrum_data = self.data_processor.get_current_spectrum()
            if spectrum_data:
                all_peaks = self.data_processor.get_all_peaks()
                self.peak_fitter.set_reference_peaks(all_peaks, spectrum_data.get('fit_params'))
                print(f"MainController: Set {len(all_peaks)} peaks as reference")
            
        elif action_name == 'clear_reference':
            self.peak_fitter.clear_reference_peaks()
            print("MainController: Reference peaks cleared")
            
        elif action_name == 'clear_manual_peaks':
            self.data_processor.clear_manual_peaks()
            
            # Update peak count in UI
            peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
            if peaks_tab and hasattr(peaks_tab, 'peak_count_label'):
                all_peaks = self.data_processor.get_all_peaks()
                peaks_tab.peak_count_label.setText(f"Peaks found: {len(all_peaks)}")
            
            # OPTIMIZED: Only update current spectrum plot for manual peak changes
            viz_manager = self.ui_manager.get_visualization_manager()
            if viz_manager:
                viz_manager.update_plot("current_spectrum")
        
        elif action_name == 'add_manual_peak':
            wavenumber = parameters.get('wavenumber')
            if wavenumber is not None:
                # Get current spectrum to convert wavenumber to index
                spectrum_data = self.data_processor.get_current_spectrum()
                if spectrum_data and len(spectrum_data.get('wavenumbers', [])) > 0:
                    wavenumbers = spectrum_data['wavenumbers']
                    # Find closest index to the specified wavenumber
                    closest_index = np.argmin(np.abs(wavenumbers - wavenumber))
                    self.data_processor.add_manual_peak(closest_index)
                    
                    # Update peak count in UI
                    peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
                    if peaks_tab and hasattr(peaks_tab, 'peak_count_label'):
                        all_peaks = self.data_processor.get_all_peaks()
                        peaks_tab.peak_count_label.setText(f"Peaks found: {len(all_peaks)}")
                    
                    # Update manual peaks list in UI
                    if peaks_tab and hasattr(peaks_tab, '_refresh_manual_peaks_display'):
                        peaks_tab._refresh_manual_peaks_display()
                    
                    # Update visualization
                    viz_manager = self.ui_manager.get_visualization_manager()
                    if viz_manager:
                        viz_manager.update_plot("current_spectrum")
                    
                    print(f"MainController: Added manual peak at wavenumber {wavenumber:.1f} cm⁻¹ (index {closest_index})")
                    
        elif action_name == 'delete_manual_peaks':
            indices = parameters.get('indices', [])
            if indices:
                for index in sorted(indices, reverse=True):  # Remove from highest index first
                    if hasattr(self.data_processor, 'manual_peaks') and index < len(self.data_processor.manual_peaks):
                        peak_index = self.data_processor.manual_peaks[index]
                        self.data_processor.remove_manual_peak(peak_index)
                
                # Update UI
                peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
                if peaks_tab and hasattr(peaks_tab, '_refresh_manual_peaks_display'):
                    peaks_tab._refresh_manual_peaks_display()
                
                # Update visualization
                viz_manager = self.ui_manager.get_visualization_manager()
                if viz_manager:
                    viz_manager.update_plot("current_spectrum")
                
                print(f"MainController: Deleted {len(indices)} manual peaks")
        
        elif action_name == 'remove_manual_peak':
            peak_index = parameters.get('peak_index')
            if peak_index is not None:
                self.data_processor.remove_manual_peak(peak_index)
                
                # Update peak count in UI
                peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
                if peaks_tab and hasattr(peaks_tab, 'peak_count_label'):
                    all_peaks = self.data_processor.get_all_peaks()
                    peaks_tab.peak_count_label.setText(f"Peaks found: {len(all_peaks)}")
                
                # Update manual peaks list in UI
                spectrum_data = self.data_processor.get_current_spectrum()
                if spectrum_data and peaks_tab and hasattr(peaks_tab, '_update_manual_peaks_list'):
                    manual_peaks = spectrum_data.get('manual_peaks', [])
                    wavenumbers = spectrum_data.get('wavenumbers', [])
                    peaks_tab._update_manual_peaks_list(manual_peaks, wavenumbers)
                
                # OPTIMIZED: Only update current spectrum plot for manual peak changes
                viz_manager = self.ui_manager.get_visualization_manager()
                if viz_manager:
                    viz_manager.update_plot("current_spectrum")
                
                print(f"MainController: Removed manual peak at index {peak_index}")
                
        elif action_name == 'enable_interactive_mode':
            enabled = parameters.get('enabled', False)
            # This could be used to enable/disable plot click handling
            viz_manager = self.ui_manager.get_visualization_manager()
            if viz_manager:
                # Enable or disable interactive mode in visualization manager
                # (implementation would depend on visualization manager capabilities)
                print(f"MainController: Interactive mode {'enabled' if enabled else 'disabled'}")
    
    def _handle_background_action(self, action_name, parameters):
        """Handle background-related actions"""
        spectrum_data = self.data_processor.get_current_spectrum()
        if not spectrum_data:
            self._show_error_message("Background Processing", "No spectrum data available")
            return
            
        if action_name == 'apply_background':
            # Apply background subtraction
            background = self.peak_fitter.calculate_background(
                spectrum_data['wavenumbers'],
                spectrum_data['intensities']
            )
            if background is not None:
                self.data_processor.apply_background_subtraction(background)
                print("MainController: Background subtraction applied")  # Debug
                
        elif action_name == 'preview_background':
            # Preview background subtraction
            background = self.peak_fitter.calculate_background(
                spectrum_data['wavenumbers'],
                spectrum_data['intensities']
            )
            if background is not None:
                self.data_processor.preview_background_subtraction(background)
                print("MainController: Background preview updated")  # Debug
                
        elif action_name == 'clear_background':
            self.data_processor.clear_background_subtraction()
            
        elif action_name == 'background_method_changed':
            method = parameters.get('method', 'ALS')
            # Pass the full method text to peak_fitter for proper parsing
            self.peak_fitter.set_background_method(method)
            # Trigger live background update (now optimized)
            self._update_live_background()
            
        elif action_name == 'peak_model_changed':
            model = parameters.get('model', 'Gaussian')
            self.peak_fitter.set_model(model)
        
        elif action_name == 'peak_detection_params_changed':
            # Handle live parameter updates for peak detection
            height = parameters.get('height')
            distance = parameters.get('distance') 
            prominence = parameters.get('prominence')
            
            # Update peak fitter parameters directly
            current_params = self.peak_fitter.get_parameters()
            if height is not None:
                current_params['peak_height'] = height  # height is already in 0-1 range
            if distance is not None:
                current_params['peak_distance'] = distance
            if prominence is not None:
                current_params['peak_prominence'] = prominence  # prominence is already in 0-1 range
            
            self.peak_fitter.set_peak_detection_parameters(
                current_params.get('peak_height', 0.2),
                current_params.get('peak_distance', 20),
                current_params.get('peak_prominence', 0.3)
            )
            # NOTE: No plot update needed for parameter changes unless detection is run
        
        elif action_name in ['background_params_changed', 'background_parameters_changed']:
            # Handle live background parameter updates (both individual and centralized widget signals)
            param_type = parameters.get('param_type')
            
            if param_type == 'lambda':
                lambda_val = parameters.get('lambda', 100000)
                self.peak_fitter.set_als_parameters(lambda_val, None, None)
                self._update_live_background()
            elif param_type == 'p':
                p_val = parameters.get('p', 0.01)
                self.peak_fitter.set_als_parameters(None, p_val, None)
                self._update_live_background()
            elif param_type == 'niter':
                niter_val = parameters.get('niter', 10)
                self.peak_fitter.set_als_parameters(None, None, niter_val)
                self._update_live_background()
            else:
                # Handle centralized widget parameters (all at once)
                lambda_val = parameters.get('lambda')
                p_val = parameters.get('p')
                niter_val = parameters.get('niter')
                
                if lambda_val is not None or p_val is not None or niter_val is not None:
                    self.peak_fitter.set_als_parameters(lambda_val, p_val, niter_val)
                    self._update_live_background()
    
    def _handle_batch_action(self, action_name, parameters):
        """Handle batch processing actions"""
        if action_name == 'start_batch_processing':
            if self.is_processing:
                self._show_error_message("Batch Processing", "Batch processing is already running")
                return
                
            # Get file list from data processor
            file_list = self.data_processor.get_file_list()
            if not file_list:
                self._show_error_message("Batch Processing", "No files loaded for processing")
                return
            
            # Start batch processing
            self._start_batch_processing(parameters)
            
        elif action_name == 'stop_batch_processing':
            self._stop_batch_processing()
    
    def _handle_io_action(self, action_name, parameters):
        """Handle input/output actions"""
        if action_name == 'export_results':
            export_format = parameters.get('format', 'csv')
            self._export_results(export_format, parameters)
            
        elif action_name.startswith('save_') or action_name.startswith('load_'):
            # Handle session save/load actions
            self._handle_session_action(action_name, parameters)
    
    def _handle_session_action(self, action_name, parameters):
        """Handle session management actions"""
        if action_name == 'save_session':
            session_data = parameters.get('data', {})
            prompt_name = parameters.get('prompt_name', False)
            self._save_session(session_data, prompt_name)
            
        elif action_name == 'load_session':
            session_name = parameters.get('session_name', '')
            self._load_session(session_name)
            
        elif action_name == 'delete_session':
            session_name = parameters.get('session_name', '')
            self._delete_session(session_name)
            
        elif action_name == 'new_session':
            self._new_session()
    
    def _handle_visualization_action(self, action_name, parameters):
        """Handle visualization-related actions"""
        viz_manager = self.ui_manager.get_visualization_manager()
        if not viz_manager:
            print("MainController: Visualization manager not available")  # Debug
            return
        
        if action_name == 'plot_update':
            plot_type = parameters.get('plot_type', 'current_spectrum')
            viz_manager.update_plot(plot_type)
            
        elif action_name == 'plot_refresh_all':
            viz_manager.update_all_plots()
            
        elif action_name == 'plot_export':
            plot_type = parameters.get('plot_type', 'current_spectrum')
            file_path = parameters.get('file_path', None)
            if file_path:
                plot_component = viz_manager.get_plot_component(plot_type)
                if plot_component:
                    plot_component.export_plot(file_path)
                    
        elif action_name == 'plot_settings_changed':
            plot_type = parameters.get('plot_type', 'current_spectrum')
            settings = parameters.get('settings', {})
            viz_manager.set_plot_settings(plot_type, settings)
            
        elif action_name == 'visualization_clear_all':
            viz_manager.clear_all_plots()
            
        else:
            print(f"MainController: Unknown visualization action '{action_name}'")  # Debug
    
    def _handle_generic_action(self, action_name, parameters):
        """Handle generic actions that don't fit other categories"""
        print(f"MainController: Generic action '{action_name}' not implemented yet")  # Debug
    
    def _handle_status_update(self, message):
        """Handle status updates from UI components"""
        print(f"Status: {message}")  # Debug - could be connected to a status bar
    
    # Core processing methods
    
    def _start_batch_processing(self, options):
        """Start batch processing with given options"""
        file_list = self.data_processor.get_file_list()
        
        print(f"MainController: Starting batch processing of {len(file_list)} files")  # Debug
        
        try:
            self.is_processing = True
            successful = 0
            failed = 0
            
            # Get batch tab for live updates
            batch_tab = self.ui_manager.get_tab_by_name('Batch')
            if batch_tab:
                batch_tab.reset_batch_counters()
            
            for i, file_path in enumerate(file_list):
                if not self.is_processing:  # Check for stop request
                    break
                
                # Update progress
                current_file = os.path.basename(file_path)
                if batch_tab:
                    batch_tab.update_batch_progress(i + 1, len(file_list), current_file)
                
                # Process the spectrum
                try:
                    # Load spectrum
                    self.data_processor.switch_spectrum(i)
                    spectrum_data = self.data_processor.get_current_spectrum()
                    
                    if not spectrum_data:
                        if batch_tab:
                            batch_tab.update_batch_result(i + 1, current_file, False, "Failed to load spectrum")
                        failed += 1
                        continue
                    
                    # Apply background subtraction if requested
                    if options.get('auto_background', True):
                        background = self.peak_fitter.calculate_background(
                            spectrum_data['wavenumbers'],
                            spectrum_data['intensities']
                        )
                        if background is not None:
                            self.data_processor.apply_background_subtraction(background)
                    
                    # Detect/fit peaks
                    peaks_found = False
                    detected_peaks = np.array([], dtype=int)
                    if options.get('auto_peaks', True):
                        peaks = self.peak_fitter.detect_peaks(
                            spectrum_data['wavenumbers'],
                            spectrum_data['intensities']
                        )
                        if peaks is not None and len(peaks) > 0:
                            peaks_found = True
                            detected_peaks = peaks
                    
                    # Fit peaks (only if we have detected peaks)
                    if peaks_found and len(detected_peaks) > 0:
                        # For batch processing, use the same background setting as the user preference
                        # Could be made configurable per batch in future
                        result = self.peak_fitter.fit_peaks(
                            spectrum_data['wavenumbers'],
                            spectrum_data['intensities'],
                            detected_peaks,
                            apply_background=False  # Default to raw spectrum for batch processing
                        )
                        
                        if result.get('success', False):
                            # Store result
                            self.data_processor.fit_params = result.get('fit_params', [])
                            self.data_processor.fit_result = result
                            result_data = {
                                'file_path': file_path,
                                'peak_count': len(detected_peaks),
                                'r_squared': result.get('r_squared', 0),
                                'fit_params': result.get('fit_params', []),
                                'success': True
                            }
                            if batch_tab:
                                batch_tab.update_batch_result(i + 1, current_file, True, f"R² = {result.get('r_squared', 0):.3f}")
                            successful += 1
                        else:
                            result_data = {
                                'file_path': file_path,
                                'peak_count': 0,
                                'success': False,
                                'error': result.get('error', 'Fitting failed')
                            }
                            if batch_tab:
                                batch_tab.update_batch_result(i + 1, current_file, False, result.get('error', 'Fitting failed'))
                            failed += 1
                    else:
                        result_data = {
                            'file_path': file_path,
                            'peak_count': 0,
                            'success': False,
                            'error': 'No peaks detected'
                        }
                        if batch_tab:
                            batch_tab.update_batch_result(i + 1, current_file, False, "No peaks detected")
                        failed += 1
                    
                    # Store result in batch results
                    self.data_processor.add_batch_result(result_data)
                    
                    # Update results tab with detailed data
                    results_tab = self.ui_manager.get_tab_by_name('Results')
                    if results_tab:
                        results_tab._add_result_to_table({
                            'file_name': current_file,
                            'file_path': file_path,
                            'success': result_data['success'],
                            'peak_count': result_data['peak_count'],
                            'r_squared': result_data.get('r_squared', 0),
                            'processing_time': 0.0,  # Could be added if needed
                            'notes': result_data.get('error', '') if not result_data['success'] else f"{result_data['peak_count']} peaks fitted"
                        })
                    
                    # Force GUI update
                    QApplication.processEvents()
                    
                except Exception as e:
                    error_msg = f"Processing error: {str(e)}"
                    if batch_tab:
                        batch_tab.update_batch_result(i + 1, current_file, False, error_msg)
            
            # Final update
            if batch_tab:
                batch_tab.finish_batch_processing(len(file_list), successful, failed)
            
            print(f"Batch processing completed: {successful} successful, {failed} failed")
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            if batch_tab:
                batch_tab.finish_batch_processing(0, 0, 0)
        finally:
            self.is_processing = False
    
    def _stop_batch_processing(self):
        """Stop batch processing"""
        self.is_processing = False
        print("MainController: Batch processing stopped")  # Debug
    
    def _finish_batch_processing(self):
        """Complete batch processing"""
        self.is_processing = False
        
        # Calculate summary statistics
        total_processed = len(self.current_batch_results)
        successful = sum(1 for r in self.current_batch_results if r['success'])
        failed = total_processed - successful
        
        print(f"MainController: Batch processing complete - {successful}/{total_processed} successful")  # Debug
        
        # Update batch tab
        batch_tab = self.ui_manager.get_tab_by_name('Batch')
        if batch_tab:
            batch_tab.finish_batch_processing(total_processed, successful, failed)
        
        # OPTIMIZED: Update plots that depend on batch results
        self.ui_manager.update_batch_visualizations()
        
        # Emit completion signal
        self.processing_finished.emit({
            'total': total_processed,
            'successful': successful,
            'failed': failed,
            'results': self.current_batch_results
        })
    
    # Event handlers for component signals
    
    def _on_spectrum_loaded(self, spectrum_data):
        """Handle spectrum loading"""
        # Debug output reduced to prevent performance issues during file loading
        # print(f"MainController: Spectrum loaded with {len(spectrum_data.get('wavenumbers', []))} points")
        self.ui_manager.update_all_tabs_from_data_processor()
        
        # Note: update_all_tabs_from_data_processor() now uses optimized visualization updates
        
        self.spectrum_changed.emit(spectrum_data)
    
    def _on_spectra_list_changed(self, file_list):
        """Handle spectra list changes"""
        # Debug output reduced to prevent performance issues during file loading
        # print(f"MainController: Spectra list changed - {len(file_list)} files")
        self.ui_manager.update_all_tabs_from_data_processor()
    
    def _on_current_spectrum_changed(self, index):
        """Handle current spectrum change"""
        # Debug output reduced to prevent performance issues during spectrum selection
        # print(f"MainController: Current spectrum changed to index {index}")
        self.ui_manager.update_all_tabs_from_data_processor()
    
    def _on_file_operation_completed(self, operation_type, success, message):
        """Handle file operation completion"""
        print(f"MainController: File operation '{operation_type}' - {'Success' if success else 'Failed'}: {message}")  # Debug
    
    def _on_fitting_completed(self, result):
        """Handle peak fitting completion"""
        print(f"MainController: Peak fitting completed - {'Success' if result.get('success', False) else 'Failed'}")  # Debug
        self.ui_manager.update_all_tabs_from_peak_fitter()
    
    def _on_peaks_detected(self, peaks):
        """Handle peak detection"""
        print(f"MainController: {len(peaks)} peaks detected")  # Debug
        self.ui_manager.update_all_tabs_from_peak_fitter()
    
    def _on_background_calculated(self, background):
        """Handle background calculation"""
        print(f"MainController: Background calculated with {len(background)} points")  # Debug
        
        # Update the spectrum data with the new background for preview
        self.data_processor.preview_background_subtraction(background)
        
        # Update the visualization to show the new background
        viz_manager = self.ui_manager.get_visualization_manager()
        if viz_manager:
            viz_manager.update_current_spectrum_only()
    
    def _on_fitting_progress(self, progress_info):
        """Handle fitting progress updates"""
        # Could update progress bars if needed
        pass
    
    def _on_plot_updated(self, plot_type):
        """Handle plot update events"""
        # Debug output removed to prevent performance issues during live updates
        # print(f"MainController: Plot '{plot_type}' updated")
    
    def _on_plot_clicked(self, plot_type, x, y):
        """Handle plot click events"""
        print(f"MainController: Plot '{plot_type}' clicked at ({x:.2f}, {y:.2f})")  # Debug
        
        # Handle specific plot interactions
        if plot_type == "current_spectrum":
            self._handle_spectrum_plot_click(x, y)
    
    def _on_visualization_ready(self):
        """Handle visualization manager ready event"""
        print("MainController: Visualization system ready")  # Debug
    
    def _handle_spectrum_plot_click(self, x, y):
        """Handle clicks on the spectrum plot to add manual peaks"""
        print(f"MainController: Spectrum click at wavenumber {x:.2f}, intensity {y:.2f}")
        
        # Get current spectrum data
        spectrum_data = self.data_processor.get_current_spectrum()
        if not spectrum_data or len(spectrum_data.get('wavenumbers', [])) == 0:
            return
        
        # Find the closest wavenumber index
        wavenumbers = spectrum_data['wavenumbers']
        closest_index = np.argmin(np.abs(wavenumbers - x))
        
        # Add manual peak
        self.data_processor.add_manual_peak(closest_index)
        
        # Update peak count in UI
        peaks_tab = self.ui_manager.get_tab_by_name('Peaks')
        if peaks_tab and hasattr(peaks_tab, 'peak_count_label'):
            all_peaks = self.data_processor.get_all_peaks()
            peaks_tab.peak_count_label.setText(f"Peaks found: {len(all_peaks)}")
        
        # Update manual peaks list in UI
        spectrum_data = self.data_processor.get_current_spectrum()
        if spectrum_data and peaks_tab and hasattr(peaks_tab, '_update_manual_peaks_list'):
            manual_peaks = spectrum_data.get('manual_peaks', [])
            wavenumbers = spectrum_data.get('wavenumbers', [])
            peaks_tab._update_manual_peaks_list(manual_peaks, wavenumbers)
        
        # OPTIMIZED: Only update current spectrum plot for manual peak addition
        viz_manager = self.ui_manager.get_visualization_manager()
        if viz_manager:
            viz_manager.update_current_spectrum_only()
        
        print(f"MainController: Added manual peak at index {closest_index} (wavenumber {wavenumbers[closest_index]:.2f})")
    
    # Utility methods
    
    def _export_results(self, export_format, parameters):
        """Export results in specified format"""
        print(f"MainController: Exporting results in {export_format} format")  # Debug
        
        if export_format == 'csv':
            self.data_processor.export_results_to_csv(self.current_batch_results)
        elif export_format == 'comprehensive':
            self.data_processor.export_comprehensive_results(self.current_batch_results)
        else:
            print(f"MainController: Export format '{export_format}' not implemented")  # Debug
    
    def _save_session(self, session_data, prompt_name):
        """Save current session"""
        print(f"MainController: Saving session (prompt_name={prompt_name})")  # Debug
        # Session management would be implemented here
    
    def _load_session(self, session_name):
        """Load session by name"""
        print(f"MainController: Loading session '{session_name}'")  # Debug
        # Session management would be implemented here
    
    def _delete_session(self, session_name):
        """Delete session by name"""
        print(f"MainController: Deleting session '{session_name}'")  # Debug
        # Session management would be implemented here
    
    def _new_session(self):
        """Start a new session"""
        print("MainController: Starting new session")  # Debug
        # Reset all components to default state
        self.data_processor.reset()
        self.peak_fitter.reset()
        self.ui_manager.reset_all_tabs()
        self.current_batch_results = []
    
    def _show_error_message(self, title, message):
        """Show error message to user"""
        QMessageBox.critical(self, title, message)
    
    def _show_info_message(self, title, message):
        """Show info message to user"""
        QMessageBox.information(self, title, message)
    
    def _update_live_background(self):
        """Update background calculation live when parameters change"""
        spectrum_data = self.data_processor.get_current_spectrum()
        if spectrum_data and len(spectrum_data.get('wavenumbers', [])) > 0:
            # Calculate background with current method
            background = self.peak_fitter.calculate_background(
                spectrum_data['wavenumbers'],
                spectrum_data['intensities']
            )
            
            if background is not None:
                # Preview the background (don't apply yet)
                self.data_processor.preview_background_subtraction(background)
                
                # OPTIMIZED: Only update current spectrum plot for background changes
                viz_manager = self.ui_manager.get_visualization_manager()
                if viz_manager:
                    viz_manager.update_current_spectrum_only()
                
                # Debug output reduced for performance during live background updates
                # print("MainController: Live background updated")
    

    # Public interface methods
    
    def get_current_spectrum_data(self):
        """Get current spectrum data"""
        return self.data_processor.get_current_spectrum()
    
    def get_batch_results(self):
        """Get current batch results"""
        return self.current_batch_results.copy()
    
    def get_component_status(self):
        """Get status of all components"""
        return {
            'data_processor': self.data_processor.get_status(),
            'peak_fitter': self.peak_fitter.get_status(),
            'ui_manager': self.ui_manager.get_ui_state_summary(),
            'is_processing': self.is_processing
        }


# Convenience function for launching the application
def launch_batch_peak_fitting(parent=None, wavenumbers=None, intensities=None):
    """
    Launch the modular batch peak fitting application with robust error handling.
    
    Args:
        parent: Parent widget (optional)
        wavenumbers: Initial wavenumber data (optional)
        intensities: Initial intensity data (optional)
    
    Returns:
        BatchPeakFittingMainController instance or None if critical dependencies missing
    """
    # Check module availability before launching
    print(f"\n🚀 Launching Batch Peak Fitting...")
    print(f"📊 Module availability check:")
    for module, status in MODULE_STATUS.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {module}")
    
    # Check minimum requirements
    if not LOCAL_IMPORTS_AVAILABLE or not UI_MANAGER_AVAILABLE:
        error_msg = (
            "❌ Cannot launch Batch Peak Fitting: Critical modules missing\n\n"
            "Missing modules:\n"
        )
        if not LOCAL_IMPORTS_AVAILABLE:
            error_msg += "• Local batch peak fitting modules\n"
        if not UI_MANAGER_AVAILABLE:
            error_msg += "• UI Manager\n"
        
        error_msg += (
            "\nPossible solutions:\n"
            "• Ensure you're running from the RamanLab root directory\n"
            "• Check that all batch_peak_fitting subdirectories exist\n"
            "• Verify Python path includes RamanLab root\n"
            "• Run: python -c \"import sys; print(sys.path[0])\""
        )
        
        print(error_msg)
        
        if parent:
            QMessageBox.critical(parent, "Module Error", error_msg)
        
        return None
    
    # Warn about degraded functionality if core modules unavailable
    if not CORE_AVAILABLE:
        warning_msg = (
            "⚠️  Warning: Core peak fitting modules not available\n\n"
            "The application will run with reduced functionality:\n"
            "• Basic peak detection using scipy\n"
            "• Limited background correction methods\n"
            "• Some advanced features may not work\n\n"
            "For full functionality, ensure the 'core' directory is accessible."
        )
        print(warning_msg)
        
        if parent:
            QMessageBox.warning(parent, "Reduced Functionality", warning_msg)
    
    try:
        # Check if we're running within an existing QApplication
        existing_app = QApplication.instance()
        app_created = False
        
        # Create QApplication if none exists and no parent (standalone mode)
        if not existing_app and not parent:
            import sys
            app = QApplication(sys.argv)
            app_created = True
        
        # Create the main controller
        print("🔧 Initializing main controller...")
        controller = BatchPeakFittingMainController(parent, wavenumbers, intensities)
        
        # Show the dialog
        controller.show()
        print("✅ Batch Peak Fitting launched successfully!")
        
        # If we created our own app, run the event loop
        if app_created:
            controller.exec()
        
        return controller
        
    except Exception as e:
        error_msg = f"❌ Failed to launch Batch Peak Fitting: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        if parent:
            QMessageBox.critical(parent, "Launch Error", error_msg)
        
        return None 