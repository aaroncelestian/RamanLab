#!/usr/bin/env python3
"""
Startup Performance Profiler for RamanLab
Measures import times and identifies bottlenecks
"""

import time
import sys
import importlib.util
from pathlib import Path

def time_import(module_name, file_path=None):
    """Time how long it takes to import a module"""
    start = time.time()
    try:
        if file_path:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            __import__(module_name)
        elapsed = time.time() - start
        return elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return elapsed, str(e)

def profile_startup():
    """Profile the startup time of RamanLab components"""
    print("="*80)
    print("RamanLab Startup Performance Profile")
    print("="*80)
    
    # Common scientific libraries
    print("\n1. Core Python Libraries:")
    libraries = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'PySide6',
        'sklearn',
    ]
    
    total_lib_time = 0
    for lib in libraries:
        elapsed, error = time_import(lib)
        total_lib_time += elapsed
        status = "✓" if error is None else "✗"
        print(f"  {status} {lib:20s} {elapsed:6.3f}s" + (f" - {error}" if error else ""))
    
    print(f"\n  Total library import time: {total_lib_time:.3f}s")
    
    # Check for pickle loading
    print("\n2. Large Data Files:")
    data_files = [
        'raman_database.pkl',
        'mineral_modes.pkl',
    ]
    
    for file in data_files:
        file_path = Path(__file__).parent / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"  • {file:30s} {size_mb:8.2f} MB")
            
            # Time loading the pickle
            start = time.time()
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                elapsed = time.time() - start
                print(f"    Load time: {elapsed:.3f}s")
            except Exception as e:
                print(f"    Load failed: {e}")
    
    # Check main application files
    print("\n3. Main Application Files:")
    app_files = [
        'raman_analysis_app_qt6.py',
        'raman_cluster_analysis_qt6.py',
        'peak_fitting_qt6.py',
    ]
    
    for file in app_files:
        file_path = Path(__file__).parent / file
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  • {file:40s} {size_kb:8.2f} KB")
    
    # Check for slow file operations
    print("\n4. File System Performance:")
    test_dir = Path(__file__).parent
    
    # Count files
    start = time.time()
    file_count = len(list(test_dir.glob('*.py')))
    elapsed = time.time() - start
    print(f"  • Listing {file_count} Python files: {elapsed:.3f}s")
    
    # Test read speed
    test_file = test_dir / 'raman_analysis_app_qt6.py'
    if test_file.exists():
        start = time.time()
        with open(test_file, 'r') as f:
            content = f.read()
        elapsed = time.time() - start
        size_kb = len(content) / 1024
        speed_mb_s = (size_kb / 1024) / elapsed if elapsed > 0 else 0
        print(f"  • Reading {size_kb:.1f} KB file: {elapsed:.3f}s ({speed_mb_s:.1f} MB/s)")
    
    # Check for iCloud sync conflicts
    print("\n5. iCloud Sync Issues:")
    conflict_files = list(test_dir.glob("*conflicted copy*"))
    if conflict_files:
        print(f"  ⚠️  Found {len(conflict_files)} iCloud conflict files!")
        print(f"  This indicates active iCloud syncing which slows file operations")
        for f in conflict_files[:5]:
            print(f"    • {f.name}")
        if len(conflict_files) > 5:
            print(f"    ... and {len(conflict_files) - 5} more")
    else:
        print(f"  ✓ No iCloud conflicts detected")
    
    # Check if directory is in iCloud
    if 'Mobile Documents' in str(test_dir):
        print(f"\n  ⚠️  WARNING: This directory appears to be in iCloud!")
        print(f"  Path: {test_dir}")
    
    print("\n" + "="*80)
    print("Recommendations:")
    print("="*80)
    
    recommendations = []
    
    if total_lib_time > 2.0:
        recommendations.append("• Library imports are slow (>2s). Consider lazy loading.")
    
    # Check for large pickle files
    db_file = Path(__file__).parent / 'raman_database.pkl'
    if db_file.exists() and db_file.stat().st_size > 50 * 1024**2:
        recommendations.append("• raman_database.pkl is very large (>50MB). Consider:")
        recommendations.append("  - Using SQLite or HDF5 instead of pickle")
        recommendations.append("  - Lazy loading only needed data")
        recommendations.append("  - Compressing the database")
    
    if conflict_files:
        recommendations.append("• Disable iCloud sync for this working directory:")
        recommendations.append("  - Right-click folder → 'Remove Download' or move outside iCloud")
    
    if not recommendations:
        recommendations.append("✓ No major issues detected. Check Activity Monitor during startup.")
    
    for rec in recommendations:
        print(rec)
    
    print("\n")

if __name__ == "__main__":
    profile_startup()
