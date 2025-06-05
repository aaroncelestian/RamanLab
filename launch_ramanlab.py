#!/usr/bin/env python3
"""
RamanLab Launcher
=================

Simple launcher script for RamanLab with dependency checking.

Usage:
    python launch_ramanlab.py

Author: Aaron Celestian
Version: 1.0.0
License: MIT
"""

import sys
import os
from pathlib import Path

def main():
    """Launch RamanLab with dependency checking."""
    print("🔬 RamanLab v1.0.0 'Debut'")
    print("=" * 40)
    
    # Get the script directory
    script_dir = Path(__file__).parent.absolute()
    main_app = script_dir / "main_qt6.py"
    dependency_checker = script_dir / "check_dependencies.py"
    
    # Check if main application exists
    if not main_app.exists():
        print("❌ Error: Main application not found!")
        print(f"Expected location: {main_app}")
        sys.exit(1)
    
    # Check dependencies first if checker exists
    if dependency_checker.exists():
        print("🔍 Checking dependencies...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, str(dependency_checker)], 
                                  capture_output=True, text=True)
            
            # Check for critical missing dependencies in the output
            critical_missing = []
            # Check if both Qt6 frameworks are missing (either PySide6 or PyQt6 is sufficient)
            pyside6_missing = "❌ PySide6: Not installed" in result.stdout
            pyqt6_missing = "❌ PyQt6: Not installed" in result.stdout
            if pyside6_missing and pyqt6_missing:
                critical_missing.append("Qt6 GUI Framework (PySide6 or PyQt6)")
            if "❌ scipy: Not installed" in result.stdout:
                critical_missing.append("scipy")
            if "❌ numpy: Not installed" in result.stdout:
                critical_missing.append("numpy")
            if "❌ matplotlib: Not installed" in result.stdout:
                critical_missing.append("matplotlib")
            
            if critical_missing:
                print("❌ Critical dependencies missing!")
                print("Missing required packages:")
                for pkg in critical_missing:
                    print(f"   - {pkg}")
                print("\n💡 Install missing dependencies with:")
                if "Qt6 GUI Framework" in str(critical_missing):
                    print("   pip install PySide6")
                if any(pkg in str(critical_missing) for pkg in ["scipy", "numpy", "matplotlib"]):
                    print("   pip install scipy numpy matplotlib pandas")
                print("\n🔧 Or install all requirements:")
                print("   pip install -r requirements_qt6.txt")
                sys.exit(1)
            else:
                print("✅ All critical dependencies satisfied!")
                
        except Exception as e:
            print(f"⚠️  Could not run dependency check: {e}")
            print("Proceeding with launch anyway...")
    
    # Launch the main application
    print("🚀 Launching RamanLab...")
    try:
        # Change to the script directory to ensure relative paths work
        os.chdir(script_dir)
        
        # Add the script directory to Python path to allow imports
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        # Import and run the main application
        import main_qt6
        exit_code = main_qt6.main()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n👋 RamanLab closed by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error launching RamanLab: {e}")
        print(f"📍 Please check the main application file: {main_app}")
        sys.exit(1)

if __name__ == "__main__":
    main() 