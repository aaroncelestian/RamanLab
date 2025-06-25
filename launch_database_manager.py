#!/usr/bin/env python3
"""
Launch script for RamanLab Database Manager GUI
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from database_manager_gui import main
    main()
except ImportError as e:
    print(f"Error importing database manager: {e}")
    print("Make sure you have PySide6 installed: pip install PySide6")
    sys.exit(1)
except Exception as e:
    print(f"Error running database manager: {e}")
    sys.exit(1) 