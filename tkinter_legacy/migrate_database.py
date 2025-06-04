#!/usr/bin/env python3
"""
Database Migration Utility - Import tkinter database to Qt6 format
"""

import os
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

# Import Qt6 database class
from raman_spectra_qt6 import RamanSpectraQt6


def migrate_database():
    """Migrate the original tkinter database to Qt6 format."""
    
    # Path to original database
    original_db_path = "raman_database.pkl"  # Now in the same directory
    
    print("🔄 Starting database migration...")
    print(f"📂 Source: {original_db_path}")
    
    if not os.path.exists(original_db_path):
        print("❌ Original database not found!")
        return False
    
    try:
        # Load original database
        print("📖 Loading original database...")
        with open(original_db_path, 'rb') as f:
            original_db = pickle.load(f)
        
        print(f"✅ Original database loaded with {len(original_db)} entries")
        
        # Create Qt6 database instance
        qt6_db = RamanSpectraQt6()
        print(f"📁 Qt6 database will be created at: {qt6_db.db_path}")
        
        # Migrate each spectrum
        success_count = 0
        error_count = 0
        
        for name, entry in original_db.items():
            try:
                # Skip metadata entries
                if name.startswith('__'):
                    continue
                
                print(f"🔄 Migrating: {name}")
                
                # Extract data
                wavenumbers = entry.get('wavenumbers', [])
                intensities = entry.get('intensities', [])
                metadata = entry.get('metadata', {})
                
                # Convert to numpy arrays if needed
                if not isinstance(wavenumbers, np.ndarray):
                    wavenumbers = np.array(wavenumbers)
                if not isinstance(intensities, np.ndarray):
                    intensities = np.array(intensities)
                
                # Add timestamp if not present
                if 'timestamp' not in metadata:
                    metadata['timestamp'] = datetime.now().isoformat()
                
                # Add to Qt6 database
                success = qt6_db.add_to_database(
                    name=name,
                    wavenumbers=wavenumbers,
                    intensities=intensities,
                    metadata=metadata
                )
                
                if success:
                    success_count += 1
                    print(f"✅ Successfully migrated: {name}")
                else:
                    error_count += 1
                    print(f"❌ Failed to migrate: {name}")
                    
            except Exception as e:
                error_count += 1
                print(f"❌ Error migrating {name}: {str(e)}")
        
        # Print summary
        print("\n" + "="*50)
        print("📊 Migration Summary:")
        print(f"✅ Successfully migrated: {success_count} spectra")
        print(f"❌ Failed migrations: {error_count} spectra")
        print(f"📁 New database location: {qt6_db.db_path}")
        
        # Get database stats
        stats = qt6_db.get_database_stats()
        print(f"💾 Database size: {stats['database_size']}")
        print(f"📈 Average data points: {stats['avg_data_points']:.0f}")
        
        print("\n🎉 Migration completed!")
        print("You can now use the Qt6 Database Browser to explore your data.")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        return False


if __name__ == "__main__":
    migrate_database() 