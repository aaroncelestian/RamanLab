#!/usr/bin/env python3
"""
RamanLab Qt6 Version - Main Application Entry Point
Qt6 conversion of the Raman Spectrum Analysis Tool
"""

import sys
import importlib
import platform
from pathlib import Path

# Qt6 imports
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QIcon

# Version info
from version import __version__, __author__, __copyright__

# Import the Qt6 version of the main app
from raman_analysis_app_qt6 import RamanAnalysisAppQt6

# Import update checker for optional startup checking
# TEMPORARILY DISABLED to isolate threading issues
# try:
#     from core.update_checker import check_for_updates, UPDATE_CHECKER_AVAILABLE
#     UPDATE_CHECKER_AVAILABLE = UPDATE_CHECKER_AVAILABLE
# except ImportError:
UPDATE_CHECKER_AVAILABLE = False


def setup_application():
    """Set up the Qt application with proper configuration."""
    # Set application properties before creating QApplication
    QCoreApplication.setApplicationName("RamanLab")
    QCoreApplication.setApplicationVersion(__version__)
    QCoreApplication.setOrganizationName("RamanLab")
    QCoreApplication.setOrganizationDomain("ramanlab.org")
    
    # Create the application
    app = QApplication(sys.argv)
    
    # Set application icon based on platform
    icon_set = False
    
    # Try platform-specific icons first
    if platform.system() == "Darwin":  # macOS
        icon_path = Path("RamanLab_icon.icns")
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            icon_set = True
    elif platform.system() == "Windows":
        icon_path = Path("RamanLab_icon.ico") 
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            icon_set = True
    
    # Fallback to existing icons if new ones aren't available
    if not icon_set:
        fallback_icons = ["RamanLab_icon.icns", "RamanLab_icon.ico", "RamanLab_custom.icns"]
        for icon_name in fallback_icons:
            icon_path = Path(icon_name)
            if icon_path.exists():
                app.setWindowIcon(QIcon(str(icon_path)))
                break
    
    # High-DPI scaling is automatic in Qt6, no need to explicitly enable these deprecated attributes
    # app.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # Deprecated in Qt6
    # app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # Deprecated in Qt6
    
    return app


def main():
    """
    Main function to run the RamanLab Qt6 application.
    """
    try:
        # Set up the application
        app = setup_application()
        
        # Create the main window
        main_window = RamanAnalysisAppQt6()
        
        # Show the main window
        main_window.show()
        
        # Center the window on screen
        main_window.center_on_screen()
        
        # Optional: Check for updates silently on startup (non-blocking)
        # DISABLED: Automatic startup check causes threading issues
        # Users can still check manually via Help → Check for Updates
        # if UPDATE_CHECKER_AVAILABLE:
        #     from PySide6.QtCore import QTimer
        #     def delayed_update_check():
        #         check_for_updates(parent=main_window, show_no_update=False)
        #     QTimer.singleShot(3000, delayed_update_check)
        
        # Start the event loop
        return app.exec()
        
    except ImportError as e:
        # Handle missing dependencies gracefully
        error_msg = f"""
        Missing required dependency: {str(e)}
        
        Please install the required packages:
        pip install -r requirements_qt6.txt
        
        Or install the core requirements:
        pip install PySide6 numpy matplotlib scipy pandas
        """
        
        # Try to show a simple error dialog
        try:
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "Dependency Error", error_msg)
        except:
            print(error_msg)
        
        return 1
        
    except Exception as e:
        # Handle other errors
        error_msg = f"Failed to start RamanLab: {str(e)}"
        print(error_msg)
        
        try:
            QMessageBox.critical(None, "Startup Error", error_msg)
        except:
            pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main()) 