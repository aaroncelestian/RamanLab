#!/usr/bin/env python3
"""
RamanLab Fast Launcher
======================

Fast launcher script for RamanLab that skips dependency checking for speed.
Use this for desktop shortcuts after dependencies have been verified once.

Usage:
    python launch_ramanlab_fast.py

Author: Aaron Celestian
Version: 1.0.2
License: MIT
"""

import sys
import os
from pathlib import Path

def main():
    """Launch RamanLab without dependency checking for faster startup."""
    print("🔬 RamanLab v1.0.2 'DTW Performance Enhancement' - Fast Launch")
    
    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    main_app = script_dir / "main_qt6.py"
    
    # Check if main application exists
    if not main_app.exists():
        print("❌ Error: Main application not found!")
        print(f"Expected location: {main_app}")
        sys.exit(1)
    
    # Launch the main application directly
    print("🚀 Launching RamanLab...")
    try:
        # Change to the script directory to ensure relative paths work
        os.chdir(script_dir)
        
        # Add the script directory to Python path to allow imports
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Import and run the main application directly
        import main_qt6
        exit_code = main_qt6.main()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n👋 RamanLab closed by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching RamanLab: {e}")
        print("💡 If this persists, try running 'python launch_ramanlab.py' for diagnostic info.")
        # Try to show error in a dialog if possible
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication(sys.argv)
            QMessageBox.critical(None, "RamanLab Launch Error", 
                               f"Failed to start RamanLab:\n\n{str(e)}\n\n"
                               "Try running 'python launch_ramanlab.py' from terminal for more info.")
            app.quit()
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main() 