#!/usr/bin/env python3
"""
Legacy raman_polarization_analyzer.py stub  
===========================================
This file is a stub for the old raman_polarization_analyzer.py module.
In RamanLab Qt6, this functionality has been moved to raman_polarization_analyzer_qt6.py.

For Qt6 compatibility, use:
    python raman_polarization_analyzer_qt6.py
"""

print("⚠️  Warning: raman_polarization_analyzer.py is a legacy stub.")
print("   Use 'python raman_polarization_analyzer_qt6.py' instead.")

if __name__ == "__main__":
    print("🚀 Launching Qt6 Polarization Analyzer...")
    try:
        import subprocess
        import sys
        subprocess.run([sys.executable, "raman_polarization_analyzer_qt6.py"])
    except Exception as e:
        print(f"❌ Failed to launch Qt6 version: {e}")
        print("   Try running: python raman_polarization_analyzer_qt6.py")
