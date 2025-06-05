#!/usr/bin/env python3
"""
Standalone launcher for the Raman Database Browser.
"""

import sys
import os

# Add the current directory to the Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    """Launch the Raman Database Browser in standalone mode."""
    try:
        from raman_database_browser import RamanDatabaseGUI
        
        # Create the standalone application
        app = RamanDatabaseGUI()
        app.run()
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure raman_database_browser.py is in the same directory.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error launching Raman Database Browser: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 