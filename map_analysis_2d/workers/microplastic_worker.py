"""
Worker thread for microplastic detection to avoid blocking UI.
"""

from PySide6.QtCore import QThread, Signal
import numpy as np


class MicroplasticDetectionWorker(QThread):
    """
    Worker thread for running microplastic detection without blocking UI.
    
    Signals:
        progress_updated: (int, int, str) - current, total, message
        detection_complete: (dict) - results dictionary
        detection_failed: (str) - error message
    """
    
    progress_updated = Signal(int, int, str)
    detection_complete = Signal(dict)
    detection_failed = Signal(str)
    
    def __init__(self, detector, wavenumbers, intensities, params, database=None):
        super().__init__()
        self.detector = detector
        self.wavenumbers = wavenumbers
        self.intensities = intensities
        self.params = params
        self.database = database  # For template matching
        self._is_running = True
    
    def stop(self):
        """Stop the detection process."""
        self._is_running = False
    
    def run(self):
        """Run the detection in a separate thread."""
        try:
            # Create progress callback that emits signals
            def progress_callback(current, total, message):
                if self._is_running:
                    self.progress_updated.emit(current, total, message)
            
            # Check detection method
            method = self.params.get('method', 'Hybrid (Recommended)')
            
            # Template Matching mode - uses database plastic spectra
            if 'Template Matching' in method and self.database is not None:
                progress_callback(0, 100, "Using Template Matching mode...")
                
                # Map UI plastic type codes to full names for template matching
                plastic_type_map = {
                    'PE': 'Polyethylene',
                    'PP': 'Polypropylene', 
                    'PS': 'Polystyrene',
                    'PET': 'Polyethylene Terephthalate',
                    'PVC': 'PVC',
                    'PMMA': 'Acrylic',
                    'PA': 'Polyamide'
                }
                
                # Convert plastic type codes to full names
                plastic_types = []
                for code in self.params.get('plastic_types', []):
                    if code in plastic_type_map:
                        plastic_types.append(plastic_type_map[code])
                
                if not plastic_types:
                    plastic_types = None  # Use default types
                
                baseline_params = self.params.get('baseline', {})
                results = self.detector.scan_map_with_templates(
                    wavenumbers=self.wavenumbers,
                    intensity_map=self.intensities,
                    database=self.database,
                    plastic_types=plastic_types,
                    threshold=self.params['threshold'],
                    progress_callback=progress_callback,
                    n_jobs=-1,  # Used for baseline batching; correlation uses BLAS multi-core
                    max_templates_per_type=5,
                    baseline_method=baseline_params.get('method', 'rolling_ball'),
                    lam=baseline_params.get('lam', 1e6),
                    p=baseline_params.get('p', 0.001),
                    niter=baseline_params.get('niter', 10),
                    snip_iterations=baseline_params.get('iterations', 40),
                    smoothing_window=self.params.get('enhancement_window', 11),
                )
                
                if self._is_running:
                    self.detection_complete.emit(results)
                return
            
            # Standard detection modes (Peak-based, Database Correlation, Hybrid)
            baseline_params = self.params.get('baseline', {})
            baseline_method = baseline_params.get('method', 'rolling_ball')
            
            results = self.detector.scan_map_for_plastics(
                wavenumbers=self.wavenumbers,
                intensity_map=self.intensities,
                plastic_types=self.params['plastic_types'],
                threshold=self.params['threshold'],
                progress_callback=progress_callback,
                n_jobs=-1,  # Will be capped appropriately in detector
                fast_mode=True,
                baseline_method=baseline_method,
                lam=baseline_params.get('lam', 1e6),
                p=baseline_params.get('p', 0.001),
                niter=baseline_params.get('niter', 10),
                snip_iterations=baseline_params.get('iterations', 40),
            )
            
            if self._is_running:
                self.detection_complete.emit(results)
        
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.detection_failed.emit(error_msg)
