#!/usr/bin/env python3
"""
Legacy mineral_database.py stub
===============================
This file is a stub for the old mineral_database.py module.
In RamanLab Qt6, this functionality has been moved to core/database.py.

For Qt6 compatibility, use:
    from core.database import MineralDatabase
"""

print("⚠️  Warning: mineral_database.py is a legacy stub.")
print("   Use 'from core.database import MineralDatabase' instead.")

# Stub class for backward compatibility
class MineralDatabase:
    def __init__(self):
        print("⚠️  Using legacy MineralDatabase stub. Please update to core.database.MineralDatabase")
        from core.database import MineralDatabase as ModernMineralDatabase
        self._modern_db = ModernMineralDatabase()
    
    def __getattr__(self, name):
        return getattr(self._modern_db, name)
