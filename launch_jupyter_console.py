#!/usr/bin/env python3
"""
Simple launcher for the RamanLab Advanced Jupyter Console
========================================================

This script provides a simple way to launch the advanced Jupyter console
for testing and development purposes.
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the advanced Jupyter console."""
    try:
        # Add the current directory to the Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Import and launch the console
        from advanced_jupyter_console import main as launch_console
        launch_console()
        
    except ImportError as e:
        print(f"Error: Could not import advanced_jupyter_console: {e}")
        print("Make sure advanced_jupyter_console.py is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching console: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 