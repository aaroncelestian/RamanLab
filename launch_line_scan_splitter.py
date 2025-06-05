#!/usr/bin/env python3
"""
Launcher for Line Scan Raman Spectrum Splitter
Simple script to launch the GUI interface for splitting line scan data.
"""

import sys
import os

# Add current directory to path so we can import the splitter
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from line_scan_splitter import LineScanSplitterGUI
    
    print("Starting Line Scan Raman Spectrum Splitter...")
    gui = LineScanSplitterGUI()
    gui.run()
    
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    print("Make sure all dependencies are installed (numpy, pandas, tkinter)")
    input("Press Enter to exit...")
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to exit...") 