#!/usr/bin/env python3
"""
Legacy raman_spectra.py stub
============================
This file is a stub for the old raman_spectra.py module.
In RamanLab Qt6, this functionality has been moved to raman_spectra_qt6.py.

For Qt6 compatibility, use:
    from raman_spectra_qt6 import RamanSpectraQt6
"""

print("⚠️  Warning: raman_spectra.py is a legacy stub.")
print("   Use 'from raman_spectra_qt6 import RamanSpectraQt6' instead.")

# Stub class for backward compatibility
class RamanSpectra:
    def __init__(self):
        print("⚠️  Using legacy RamanSpectra stub. Please update to RamanSpectraQt6")
        try:
            from raman_spectra_qt6 import RamanSpectraQt6
            self._modern_db = RamanSpectraQt6()
        except ImportError:
            print("❌ Could not import RamanSpectraQt6")
            self._modern_db = None
    
    def __getattr__(self, name):
        if self._modern_db:
            return getattr(self._modern_db, name)
        else:
            raise AttributeError(f"'{name}' not available - RamanSpectraQt6 import failed")
