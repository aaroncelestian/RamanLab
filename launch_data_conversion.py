#!/usr/bin/env python3
"""
Standalone launcher for the Data Conversion Dialog
For testing and standalone use.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from PySide6.QtWidgets import QApplication
    from core.data_conversion_dialog import DataConversionDialog
    
    def main():
        """Launch the data conversion dialog."""
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("RamanLab Data Conversion")
        app.setApplicationVersion("1.0")
        
        # Create and show dialog
        dialog = DataConversionDialog()
        dialog.show()
        
        # Run application
        sys.exit(app.exec())
    
    if __name__ == "__main__":
        print("🔄 Launching RamanLab Data Conversion Tools...")
        main()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure PySide6 is installed: pip install PySide6")
    input("Press Enter to exit...")
except Exception as e:
    print(f"❌ Error: {e}")
    input("Press Enter to exit...") 