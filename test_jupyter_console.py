#!/usr/bin/env python3
"""
Test script for the RamanLab Advanced Jupyter Console
====================================================

This script tests the Jupyter console functionality and helps debug issues.
"""

import sys
import os
from pathlib import Path

def test_jupyter_imports():
    """Test if Jupyter components are available."""
    print("Testing Jupyter imports...")
    
    try:
        from qtconsole.rich_jupyter_widget import RichJupyterWidget
        print("✅ RichJupyterWidget imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RichJupyterWidget: {e}")
        return False
    
    try:
        from qtconsole.manager import QtKernelManager
        print("✅ QtKernelManager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import QtKernelManager: {e}")
        return False
    
    try:
        from jupyter_client import find_connection_file
        print("✅ jupyter_client imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import jupyter_client: {e}")
        return False
    
    return True

def test_qt_imports():
    """Test if Qt components are available."""
    print("\nTesting Qt imports...")
    
    try:
        from PySide6.QtWidgets import QApplication
        print("✅ PySide6 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PySide6: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic console functionality."""
    print("\nTesting basic console functionality...")
    
    try:
        from PySide6.QtWidgets import QApplication
        from advanced_jupyter_console import AdvancedJupyterConsole
        
        # Create application
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # Try to create console (without showing)
        console = AdvancedJupyterConsole()
        print("✅ Console created successfully")
        
        # Clean up
        console.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create console: {e}")
        return False

def main():
    """Run all tests."""
    print("RamanLab Jupyter Console Test Suite")
    print("=" * 40)
    
    # Test imports
    jupyter_ok = test_jupyter_imports()
    qt_ok = test_qt_imports()
    
    if not jupyter_ok:
        print("\n❌ Jupyter components not available. Install with:")
        print("pip install qtconsole jupyter-client ipykernel")
        return
    
    if not qt_ok:
        print("\n❌ Qt components not available. Install with:")
        print("pip install PySide6")
        return
    
    # Test functionality
    console_ok = test_basic_functionality()
    
    if console_ok:
        print("\n✅ All tests passed! Console should work properly.")
    else:
        print("\n❌ Console test failed. Check error messages above.")

if __name__ == '__main__':
    main() 