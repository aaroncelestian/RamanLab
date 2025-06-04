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
            
            # If dependency check fails, show the output and exit
            if result.returncode != 0:
                print("❌ Dependency check failed!")
                print(result.stdout)
                print(result.stderr)
                print("\n💡 Please install missing dependencies before running RamanLab.")
                sys.exit(1)
            else:
                print("✅ All dependencies satisfied!")
                
        except Exception as e:
            print(f"⚠️  Could not run dependency check: {e}")
            print("Proceeding with launch anyway...")
    
    # Launch the main application
    print("🚀 Launching RamanLab...")
    try:
        # Change to the script directory to ensure relative paths work
        os.chdir(script_dir)
        
        # Import and run the main application
        exec(open(main_app).read())
        
    except KeyboardInterrupt:
        print("\n👋 RamanLab closed by user.")
    except Exception as e:
        print(f"\n❌ Error launching RamanLab: {e}")
        print(f"📍 Please check the main application file: {main_app}")
        sys.exit(1)

if __name__ == "__main__":
    main() 