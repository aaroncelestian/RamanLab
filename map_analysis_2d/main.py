#!/usr/bin/env python3
"""
Entry point for the 2D Map Analysis application.

This module provides the main entry point for running the modular
2D Raman spectroscopy map analysis application.
"""

import sys
import logging
from pathlib import Path

# Add the package to the path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('map_analysis.log')
        ]
    )

def main():
    """Main entry point for the application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting 2D Map Analysis application")
    
    try:
        # Import Qt application
        from PySide6.QtWidgets import QApplication
        
        # Import our main window
        from map_analysis_2d.ui import MapAnalysisMainWindow
        
        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("RamanLab - 2D Map Analysis")
        app.setApplicationVersion("2.0.0")
        
        # Create and show main window
        main_window = MapAnalysisMainWindow()
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Run the application
        return app.exec()
        
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        print(f"✗ Import error: {e}")
        print("Make sure PySide6 is installed: pip install PySide6 matplotlib")
        return 1
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"✗ Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 