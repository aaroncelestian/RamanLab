#!/usr/bin/env python3
"""
Database Optimization Script
Converts large pickle to SQLite for faster startup
"""

import pickle
import sqlite3
import json
from pathlib import Path
import time

def convert_pickle_to_sqlite(pickle_file, sqlite_file):
    """Convert pickle database to SQLite for faster loading"""
    print(f"Loading pickle database: {pickle_file}")
    start = time.time()
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    load_time = time.time() - start
    print(f"  Loaded in {load_time:.2f}s")
    print(f"  Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"  Dictionary with {len(data)} keys")
        # Show sample keys
        sample_keys = list(data.keys())[:5]
        print(f"  Sample keys: {sample_keys}")
    elif isinstance(data, list):
        print(f"  List with {len(data)} items")
    
    print(f"\nTo optimize, we need to understand the data structure.")
    print(f"Please examine the pickle file structure and create appropriate SQLite schema.")
    
    return data

def create_lazy_loader():
    """Create a lazy loading wrapper for the database"""
    code = '''
class LazyDatabase:
    """Lazy loader for raman_database.pkl - only loads when accessed"""
    def __init__(self, db_path='raman_database.pkl'):
        self.db_path = db_path
        self._data = None
        self._loaded = False
    
    def load(self):
        """Load the database on first access"""
        if not self._loaded:
            import pickle
            import time
            print(f"Loading database from {self.db_path}...")
            start = time.time()
            with open(self.db_path, 'rb') as f:
                self._data = pickle.load(f)
            elapsed = time.time() - start
            print(f"Database loaded in {elapsed:.2f}s")
            self._loaded = True
        return self._data
    
    def __getitem__(self, key):
        return self.load()[key]
    
    def __contains__(self, key):
        return key in self.load()
    
    def get(self, key, default=None):
        return self.load().get(key, default)
    
    def keys(self):
        return self.load().keys()
    
    def values(self):
        return self.load().values()
    
    def items(self):
        return self.load().items()

# Usage in your app:
# Instead of:
#   with open('raman_database.pkl', 'rb') as f:
#       database = pickle.load(f)
#
# Use:
#   database = LazyDatabase('raman_database.pkl')
#   # Database only loads when you actually access it
'''
    
    output_file = Path(__file__).parent / 'lazy_database_loader.py'
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"\nCreated lazy loader: {output_file}")
    print("Import this in your main app to defer database loading until needed.")

def analyze_database_usage():
    """Analyze which parts of the database are actually used at startup"""
    print("\n" + "="*80)
    print("Database Usage Analysis")
    print("="*80)
    print("\nTo optimize startup, we need to know:")
    print("1. What data is accessed immediately at startup?")
    print("2. What data can be loaded on-demand?")
    print("3. Can the database be split into smaller chunks?")
    print("\nRecommendations:")
    print("• Use lazy loading (load only when accessed)")
    print("• Split database into: startup_data.pkl (small) + full_database.pkl (large)")
    print("• Convert to SQLite with indexes for fast queries")
    print("• Use memory-mapped files for large datasets")
    print("• Consider HDF5 for scientific data (faster than pickle)")

if __name__ == "__main__":
    print("="*80)
    print("RamanLab Database Optimization")
    print("="*80)
    
    pickle_file = Path(__file__).parent / 'raman_database.pkl'
    
    if pickle_file.exists():
        # Analyze the database
        data = convert_pickle_to_sqlite(pickle_file, None)
        
        # Create lazy loader
        create_lazy_loader()
        
        # Provide analysis
        analyze_database_usage()
    else:
        print(f"Database file not found: {pickle_file}")
