#!/usr/bin/env python3
"""
Database Access Checker for RamanLab
==========================================

This script verifies that the database files can be found and loaded
after applying the path fixes.

Usage: python check_database_access.py
"""

import os
import pickle
import sys
from pathlib import Path

def check_file_access(filename):
    """Check if a file exists and can be read."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    print(f"\nChecking {filename}:")
    print(f"  Path: {filepath}")
    
    # Check existence
    if not os.path.exists(filepath):
        print(f"  ❌ File not found!")
        return False
    
    print(f"  ✅ File exists")
    
    # Check file size
    size = os.path.getsize(filepath)
    if size == 0:
        print(f"  ⚠️  File is empty (0 bytes)")
        return False
    
    print(f"  ✅ File size: {size:,} bytes ({size/1024/1024:.1f} MB)")
    
    # Check read permissions
    if not os.access(filepath, os.R_OK):
        print(f"  ❌ File is not readable (permission denied)")
        return False
    
    print(f"  ✅ File is readable")
    
    # Try to load pickle file
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            print(f"  ✅ Successfully loaded as dictionary with {len(data)} entries")
            if data:
                # Show a sample key
                sample_key = list(data.keys())[0]
                print(f"  ℹ️  Sample entry: '{sample_key}'")
        else:
            print(f"  ⚠️  Loaded data is not a dictionary (type: {type(data)})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load pickle file: {e}")
        return False

def check_database_imports():
    """Check if database modules can import and initialize."""
    print("\nChecking database module imports:")
    
    # Test mineral database import
    try:
        from mineral_database import MineralDatabase
        db = MineralDatabase()
        minerals = db.get_minerals()
        print(f"  ✅ MineralDatabase: {len(minerals)} minerals loaded")
    except Exception as e:
        print(f"  ❌ MineralDatabase import failed: {e}")
    
    # Test raman spectra import
    try:
        from raman_spectra import RamanSpectra
        raman = RamanSpectra()
        print(f"  ✅ RamanSpectra: Initialized successfully")
    except Exception as e:
        print(f"  ❌ RamanSpectra import failed: {e}")

def main():
    """Main function to check database access."""
    print("RamanLab Database Access Checker")
    print("======================================")
    
    # Database files to check
    database_files = [
        'mineral_modes.pkl',
        'RamanLab_Database_20250602.pkl'
    ]
    
    all_good = True
    
    for db_file in database_files:
        if not check_file_access(db_file):
            all_good = False
    
    # Check module imports
    check_database_imports()
    
    print("\n" + "="*50)
    
    if all_good:
        print("🎉 All database files are accessible!")
        print("\nThe application should work correctly now.")
    else:
        print("⚠️  Some database files have issues.")
        print("\nTroubleshooting tips:")
        print("  1. Make sure you copied ALL files from the original installation")
        print("  2. Check that .pkl files have the correct permissions")
        print("  3. Verify files weren't corrupted during transfer")
        print("  4. Try running the database_path_fix.py script again")
    
    print("\nTo start the application:")
    print("  python main.py")

if __name__ == "__main__":
    main() 