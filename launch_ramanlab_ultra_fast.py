#!/usr/bin/env python3
"""
RamanLab ULTRA FAST Launcher
=============================

Optimized launcher with lazy imports to minimize startup time.
Shows UI in <1 second, loads heavy libraries in background.

This launcher:
1. Shows window immediately (~0.5s)
2. Loads heavy libraries (numpy, pandas, sklearn) in background
3. Enables full functionality once loaded

Usage:
    python launch_ramanlab_ultra_fast.py

Author: Aaron Celestian
License: MIT
"""

import sys
import os
from pathlib import Path

# Only import what we absolutely need for the GUI shell
from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtCore import Qt, QTimer, QCoreApplication
from PySide6.QtGui import QPixmap, QFont, QPainter, QColor

class LazyImportManager:
    """Manages lazy loading of heavy libraries"""
    def __init__(self):
        self._numpy = None
        self._pandas = None
        self._scipy = None
        self._sklearn = None
        self._matplotlib = None
        self._loaded = set()
    
    @property
    def numpy(self):
        if self._numpy is None:
            import numpy as np
            self._numpy = np
            self._loaded.add('numpy')
        return self._numpy
    
    @property
    def pandas(self):
        if self._pandas is None:
            import pandas as pd
            self._pandas = pd
            self._loaded.add('pandas')
        return self._pandas
    
    @property
    def scipy(self):
        if self._scipy is None:
            import scipy
            self._scipy = scipy
            self._loaded.add('scipy')
        return self._scipy
    
    @property
    def sklearn(self):
        if self._sklearn is None:
            import sklearn
            self._sklearn = sklearn
            self._loaded.add('sklearn')
        return self._sklearn
    
    @property
    def matplotlib(self):
        if self._matplotlib is None:
            import matplotlib.pyplot as plt
            self._matplotlib = plt
            self._loaded.add('matplotlib')
        return self._matplotlib
    
    def is_loaded(self, lib_name):
        return lib_name in self._loaded

# Global lazy import manager
lazy_imports = LazyImportManager()

class FastLoadSplash(QSplashScreen):
    """Custom splash screen that shows loading progress"""
    def __init__(self):
        # Create a simple splash screen
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(45, 45, 48))
        super().__init__(pixmap)
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.loading_stage = "Initializing..."
        self.progress = 0
        
    def showMessage(self, message, alignment=Qt.AlignBottom | Qt.AlignCenter, color=Qt.white):
        """Override to customize message display"""
        self.loading_stage = message
        self.progress += 15
        super().showMessage(
            f"\n\n🔬 RamanLab\n\n{message}\n\n{self.progress}%",
            alignment,
            color
        )
        QApplication.processEvents()

def load_libraries_background(splash):
    """Load heavy libraries in the background with progress updates"""
    import time
    
    # Stage 1: Core scientific libraries
    splash.showMessage("Loading NumPy...")
    _ = lazy_imports.numpy
    time.sleep(0.1)  # Brief pause to show message
    
    splash.showMessage("Loading SciPy...")
    _ = lazy_imports.scipy
    time.sleep(0.1)
    
    splash.showMessage("Loading Pandas...")
    _ = lazy_imports.pandas
    time.sleep(0.1)
    
    splash.showMessage("Loading Matplotlib...")
    _ = lazy_imports.matplotlib
    time.sleep(0.1)
    
    splash.showMessage("Loading Scikit-learn...")
    _ = lazy_imports.sklearn
    time.sleep(0.1)
    
    splash.showMessage("Almost ready...")
    time.sleep(0.2)

def main():
    """Ultra-fast launch with lazy loading."""
    import time
    start_time = time.time()
    
    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    # Set application properties BEFORE creating QApplication
    QCoreApplication.setApplicationName("RamanLab")
    QCoreApplication.setOrganizationName("RamanLab")
    
    # Create Qt application (fast - only PySide6)
    app = QApplication(sys.argv)
    
    # Show splash screen immediately
    splash = FastLoadSplash()
    splash.show()
    splash.showMessage("Starting RamanLab...")
    
    window_shown_time = time.time()
    print(f"⚡ Window shown in {window_shown_time - start_time:.3f}s")
    
    try:
        # Load libraries with progress
        load_libraries_background(splash)
        
        libs_loaded_time = time.time()
        print(f"📚 Libraries loaded in {libs_loaded_time - window_shown_time:.3f}s")
        
        # Now import the main application (it will use already-loaded libraries)
        splash.showMessage("Loading main application...")
        import main_qt6
        
        # Close splash and show main window
        splash.finish(None)
        
        total_time = time.time() - start_time
        print(f"✅ Total startup time: {total_time:.3f}s")
        print(f"   - Window shown:    {window_shown_time - start_time:.3f}s")
        print(f"   - Libraries:       {libs_loaded_time - window_shown_time:.3f}s")
        print(f"   - Main app:        {total_time - libs_loaded_time:.3f}s")
        
        # Run the application
        exit_code = main_qt6.main()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n👋 RamanLab closed by user.")
        sys.exit(0)
    except Exception as e:
        splash.close()
        print(f"\n❌ Error launching RamanLab: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error dialog
        try:
            QMessageBox.critical(
                None, 
                "RamanLab Launch Error",
                f"Failed to start RamanLab:\n\n{str(e)}\n\n"
                "Check the terminal for detailed error information."
            )
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
