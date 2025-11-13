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
            
            # Run detection
            results = self.detector.scan_map_for_plastics(
                wavenumbers=self.wavenumbers,
                intensity_map=self.intensities,
                plastic_types=self.params['plastic_types'],
                threshold=self.params['threshold'],
                progress_callback=progress_callback,
                n_jobs=-1,
                fast_mode=True
            )
            
            if self._is_running:
                self.detection_complete.emit(results)
        
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.detection_failed.emit(error_msg)
