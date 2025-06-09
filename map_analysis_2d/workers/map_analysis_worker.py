"""Background worker threads for map analysis operations."""

from PySide6.QtCore import QThread, Signal


class MapAnalysisWorker(QThread):
    """Worker thread for time-consuming map analysis operations."""
    progress = Signal(int)
    progress_message = Signal(int, str)  # progress percentage and message
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self._is_running = True
    
    @property
    def is_stopped(self):
        """Check if the worker has been stopped."""
        return not self._is_running
    
    def run(self):
        try:
            if not self._is_running:
                return
            # Pass the worker reference as the first argument so operations can emit progress
            result = self.operation(self, *self.args, **self.kwargs)
            if self._is_running:
                self.finished.emit(result)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
    
    def stop(self):
        """Stop the worker thread gracefully."""
        self._is_running = False
        self.wait(5000)  # Wait up to 5 seconds for thread to finish
        if self.isRunning():
            self.terminate()  # Force terminate if still running
            self.wait(1000)  # Wait for termination 