#!/usr/bin/env python3
"""
Window Focus Manager for RamanLab
Handles window focus issues after file dialogs and other operations
"""

import platform
from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import QTimer, QObject, Signal


class WindowFocusManager(QObject):
    """
    Manages window focus restoration after file dialogs and other operations.
    
    This class provides utilities to restore window focus after operations that
    can cause the main window to lose focus and go to the background.
    """
    
    focus_restored = Signal(QWidget)
    
    def __init__(self):
        super().__init__()
        self.system = platform.system().lower()
        
    def restore_window_focus(self, window, delay_ms=100):
        """
        Restore focus to a window after a brief delay.
        
        Args:
            window: The QWidget/QMainWindow to restore focus to
            delay_ms: Delay in milliseconds before restoring focus
        """
        if window is None:
            return
            
        # Use a timer to delay focus restoration to avoid Qt timing issues
        QTimer.singleShot(delay_ms, lambda: self._perform_focus_restoration(window))
    
    def _perform_focus_restoration(self, window):
        """
        Perform the actual focus restoration with platform-specific optimizations.
        
        Args:
            window: The QWidget/QMainWindow to restore focus to
        """
        try:
            if not window or not window.isVisible():
                return
                
            # Ensure the window is not minimized
            if window.isMinimized():
                window.showNormal()
            
            # Platform-specific focus restoration
            if self.system == "darwin":  # macOS
                self._restore_focus_macos(window)
            elif self.system == "windows":  # Windows
                self._restore_focus_windows(window)
            else:  # Linux and others
                self._restore_focus_linux(window)
                
            # Emit signal that focus was restored
            self.focus_restored.emit(window)
            
        except Exception as e:
            print(f"Warning: Failed to restore window focus: {e}")
    
    def _restore_focus_macos(self, window):
        """macOS-specific focus restoration."""
        # macOS requires specific sequence for proper focus restoration
        window.raise_()
        window.activateWindow()
        window.setFocus()
        
        # Additional macOS-specific fix: force application to foreground
        QApplication.processEvents()
        
        # Sometimes on macOS we need a second attempt
        QTimer.singleShot(50, lambda: self._secondary_focus_macos(window))
    
    def _secondary_focus_macos(self, window):
        """Secondary focus attempt for macOS."""
        try:
            if window and window.isVisible():
                window.raise_()
                window.activateWindow()
        except:
            pass
    
    def _restore_focus_windows(self, window):
        """Windows-specific focus restoration."""
        # Windows focus restoration sequence
        window.setWindowState(window.windowState() & ~window.windowState().WindowMinimized)
        window.raise_()
        window.activateWindow()
        window.setFocus()
        
        # Process events to ensure changes take effect
        QApplication.processEvents()
    
    def _restore_focus_linux(self, window):
        """Linux-specific focus restoration."""
        # Linux focus restoration (similar to Windows but with additional steps)
        window.raise_()
        window.activateWindow()
        window.setFocus()
        
        # Process events
        QApplication.processEvents()
        
        # Additional attempt for some Linux window managers
        QTimer.singleShot(25, lambda: self._secondary_focus_linux(window))
    
    def _secondary_focus_linux(self, window):
        """Secondary focus attempt for Linux."""
        try:
            if window and window.isVisible():
                window.raise_()
                window.activateWindow()
        except:
            pass


# Global instance for easy access
_focus_manager = None

def get_focus_manager():
    """Get the global WindowFocusManager instance."""
    global _focus_manager
    if _focus_manager is None:
        _focus_manager = WindowFocusManager()
    return _focus_manager


def restore_window_focus_after_dialog(window, delay_ms=100):
    """
    Convenience function to restore window focus after a dialog.
    
    This should be called after file dialogs (save/open) or other operations
    that might cause the main window to lose focus.
    
    Args:
        window: The QWidget/QMainWindow to restore focus to
        delay_ms: Delay in milliseconds before restoring focus
    
    Example:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "*.txt")
        if file_path:
            # ... save file ...
            restore_window_focus_after_dialog(self)
    """
    focus_manager = get_focus_manager()
    focus_manager.restore_window_focus(window, delay_ms)


def create_focus_restoring_file_dialog():
    """
    Create a context manager for file dialogs that automatically restores focus.
    
    Example:
        with create_focus_restoring_file_dialog() as dialog:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "*.txt")
            if file_path:
                # ... save file ...
                # Focus will be automatically restored
    """
    return FocusRestoringFileDialog()


class FocusRestoringFileDialog:
    """
    Context manager that automatically restores focus after file dialogs.
    """
    
    def __init__(self):
        self.parent_window = None
        self.focus_manager = get_focus_manager()
    
    def __enter__(self):
        # Try to find the active window
        app = QApplication.instance()
        if app:
            self.parent_window = app.activeWindow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore focus to the parent window
        if self.parent_window:
            self.focus_manager.restore_window_focus(self.parent_window)


# Decorator for methods that might cause focus loss
def restore_focus_after(delay_ms=100):
    """
    Decorator that automatically restores focus to the window after method execution.
    
    Args:
        delay_ms: Delay in milliseconds before restoring focus
    
    Example:
        @restore_focus_after(150)
        def save_spectrum(self):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save", "", "*.txt")
            # ... save logic ...
            # Focus will be automatically restored
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Execute the original method
            result = func(self, *args, **kwargs)
            
            # Restore focus to self (assuming self is a QWidget)
            if hasattr(self, 'isVisible') and callable(self.isVisible):
                restore_window_focus_after_dialog(self, delay_ms)
            
            return result
        return wrapper
    return decorator


# Platform-specific utilities
def is_macos():
    """Check if running on macOS."""
    return platform.system().lower() == "darwin"

def is_windows():
    """Check if running on Windows."""
    return platform.system().lower() == "windows"

def is_linux():
    """Check if running on Linux."""
    return platform.system().lower() == "linux"


if __name__ == "__main__":
    # Test the focus manager
    print("Window Focus Manager")
    print(f"Platform: {platform.system()}")
    print(f"macOS: {is_macos()}")
    print(f"Windows: {is_windows()}")
    print(f"Linux: {is_linux()}")
    
    # Create a test instance
    focus_manager = get_focus_manager()
    print(f"Focus manager created: {focus_manager}") 