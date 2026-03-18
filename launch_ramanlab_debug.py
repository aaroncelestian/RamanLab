#!/usr/bin/env python3
"""
RamanLab Debug Launcher
=======================

Diagnostic launcher that captures all errors and shows detailed information.
Use this when RamanLab crashes immediately on startup.

Usage:
    python launch_ramanlab_debug.py

This will:
- Show Python environment details
- Capture all import errors
- Log full error tracebacks
- Keep the window open to see errors
"""

import sys
import os
import traceback
from pathlib import Path

def print_separator():
    print("=" * 70)

def check_environment():
    """Print detailed environment information."""
    print_separator()
    print("PYTHON ENVIRONMENT")
    print_separator()
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent.absolute()}")
    print()

def check_critical_imports():
    """Check if critical packages can be imported."""
    print_separator()
    print("CHECKING CRITICAL IMPORTS")
    print_separator()
    
    critical_packages = [
        ('PySide6', 'PySide6.QtWidgets'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('h5py', 'h5py'),
    ]
    
    all_ok = True
    for display_name, import_name in critical_packages:
        try:
            __import__(import_name)
            print(f"✅ {display_name}: OK")
        except ImportError as e:
            print(f"❌ {display_name}: FAILED - {e}")
            all_ok = False
        except Exception as e:
            print(f"⚠️  {display_name}: ERROR - {e}")
            all_ok = False
    
    print()
    return all_ok

def main():
    """Launch RamanLab with full error capture."""
    print_separator()
    print("RamanLab Debug Launcher")
    print_separator()
    print()
    
    # Show environment
    check_environment()
    
    # Check imports
    imports_ok = check_critical_imports()
    
    if not imports_ok:
        print("⚠️  WARNING: Some critical imports failed!")
        print("The application may not start correctly.")
        print()
        input("Press ENTER to continue anyway, or Ctrl+C to exit...")
        print()
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    
    print_separator()
    print("LAUNCHING RAMANLAB")
    print_separator()
    print()
    
    try:
        # Import version
        try:
            from version import __version__, __release_name__
            print(f"🔬 RamanLab v{__version__} '{__release_name__}'")
        except ImportError as e:
            print(f"⚠️  Could not import version info: {e}")
            print("🔬 RamanLab (version unknown)")
        
        print()
        print("Importing main application...")
        
        # Import main_qt6
        try:
            import main_qt6
            print("✅ main_qt6 imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import main_qt6: {e}")
            print()
            print("Full traceback:")
            traceback.print_exc()
            print()
            input("Press ENTER to exit...")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error importing main_qt6: {e}")
            print()
            print("Full traceback:")
            traceback.print_exc()
            print()
            input("Press ENTER to exit...")
            sys.exit(1)
        
        print()
        print("Starting application...")
        print()
        
        # Run the application
        exit_code = main_qt6.main()
        
        print()
        print(f"Application exited with code: {exit_code}")
        
        if exit_code != 0:
            print()
            print("⚠️  Application exited with error code")
            input("Press ENTER to exit...")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print()
        print("👋 Cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        print()
        print_separator()
        print("❌ FATAL ERROR")
        print_separator()
        print()
        print(f"Error: {e}")
        print()
        print("Full traceback:")
        print_separator()
        traceback.print_exc()
        print_separator()
        print()
        print("💡 TROUBLESHOOTING:")
        print("  1. Check that all dependencies are installed:")
        print("     python check_dependencies.py")
        print()
        print("  2. Try installing missing packages:")
        print("     pip install -r requirements_qt6.txt")
        print()
        print("  3. If using conda, try:")
        print("     conda install -c conda-forge PySide6 numpy scipy matplotlib pandas h5py")
        print()
        print("  4. Check WINDOWS_SETUP.md for detailed troubleshooting")
        print()
        
        # Try to show error in dialog
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            QMessageBox.critical(
                None, 
                "RamanLab Startup Error",
                f"RamanLab failed to start:\n\n{str(e)}\n\n"
                f"See the console window for full error details.\n\n"
                f"Run 'python check_dependencies.py' to diagnose issues."
            )
        except:
            pass
        
        print()
        input("Press ENTER to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
