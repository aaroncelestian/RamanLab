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
    
    def __init__(self, detector, wavenumbers, intensities, params):
        super().__init__()
        self.detector = detector
        self.wavenumbers = wavenumbers
        self.intensities = intensities
        self.params = params
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
            
            # Run detection with baseline correction
            baseline_params = self.params.get('baseline', {})
            baseline_method = baseline_params.get('method', 'rolling_ball')
            
            # Choose mode based on baseline method
            # fast_mode=True: Uses fast rolling ball baseline (very fast, ~30 sec for 82k)
            # fast_mode=False: Uses full ALS baseline per-spectrum (slow, minutes for 82k)
            if baseline_method == 'rolling_ball':
                # Use original fast rolling ball method
                use_fast_mode = True
                lam, p, niter = 1e6, 0.001, 10  # Dummy values, not used in fast mode
            else:
                # Use ALS with specified parameters
                use_fast_mode = True  # Still use fast mode but with ALS in pre-processing
                lam = baseline_params.get('lam', 1e6)
                p = baseline_params.get('p', 0.001)
                niter = baseline_params.get('niter', 10)
            
            results = self.detector.scan_map_for_plastics(
                wavenumbers=self.wavenumbers,
                intensity_map=self.intensities,
                plastic_types=self.params['plastic_types'],
                threshold=self.params['threshold'],
                progress_callback=progress_callback,
                n_jobs=-1,
                fast_mode=use_fast_mode,
                lam=lam,
                p=p,
                niter=niter
            )
            
            if self._is_running:
                self.detection_complete.emit(results)
        
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.detection_failed.emit(error_msg)
