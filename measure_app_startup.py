#!/usr/bin/env python3
"""
Measure actual RamanLab application startup time
"""

import time
import sys
from pathlib import Path

print("="*80)
print("RamanLab Application Startup Timing")
print("="*80)

total_start = time.time()

# Phase 1: Import standard libraries
print("\nPhase 1: Standard Library Imports")
phase_start = time.time()
import os
import pickle
import json
phase1_time = time.time() - phase_start
print(f"  Time: {phase1_time:.3f}s")

# Phase 2: Import scientific libraries
print("\nPhase 2: Scientific Libraries")
phase_start = time.time()

import numpy as np
numpy_time = time.time() - phase_start
print(f"  numpy: {numpy_time:.3f}s")

phase_start = time.time()
import scipy
scipy_time = time.time() - phase_start
print(f"  scipy: {scipy_time:.3f}s")

phase_start = time.time()
import pandas as pd
pandas_time = time.time() - phase_start
print(f"  pandas: {pandas_time:.3f}s")

phase_start = time.time()
import matplotlib
matplotlib_time = time.time() - phase_start
print(f"  matplotlib: {matplotlib_time:.3f}s")

phase_start = time.time()
import matplotlib.pyplot as plt
pyplot_time = time.time() - phase_start
print(f"  matplotlib.pyplot: {pyplot_time:.3f}s")

phase_start = time.time()
from sklearn import cluster, decomposition
sklearn_time = time.time() - phase_start
print(f"  sklearn: {sklearn_time:.3f}s")

phase2_time = numpy_time + scipy_time + pandas_time + matplotlib_time + pyplot_time + sklearn_time
print(f"  Total scientific libs: {phase2_time:.3f}s")

# Phase 3: Import PySide6
print("\nPhase 3: PySide6 GUI Framework")
phase_start = time.time()
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QApplication
pyside_time = time.time() - phase_start
print(f"  PySide6: {pyside_time:.3f}s")

# Phase 4: Load database
print("\nPhase 4: Database Loading")
db_file = Path(__file__).parent / 'raman_database.pkl'
if db_file.exists():
    size_mb = db_file.stat().st_size / (1024**2)
    print(f"  Database size: {size_mb:.2f} MB")
    
    phase_start = time.time()
    with open(db_file, 'rb') as f:
        database = pickle.load(f)
    db_time = time.time() - phase_start
    print(f"  Load time: {db_time:.3f}s")
    
    # Analyze database structure
    if isinstance(database, dict):
        print(f"  Structure: Dictionary with {len(database)} entries")
    elif isinstance(database, list):
        print(f"  Structure: List with {len(database)} items")
else:
    db_time = 0
    print(f"  Database not found")

# Phase 5: Create QApplication
print("\nPhase 5: Initialize Qt Application")
phase_start = time.time()
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
qt_init_time = time.time() - phase_start
print(f"  QApplication init: {qt_init_time:.3f}s")

# Total time
total_time = time.time() - total_start

print("\n" + "="*80)
print("STARTUP TIME BREAKDOWN")
print("="*80)
print(f"Phase 1 - Standard libs:     {phase1_time:6.3f}s ({phase1_time/total_time*100:5.1f}%)")
print(f"Phase 2 - Scientific libs:   {phase2_time:6.3f}s ({phase2_time/total_time*100:5.1f}%)")
print(f"Phase 3 - PySide6:           {pyside_time:6.3f}s ({pyside_time/total_time*100:5.1f}%)")
print(f"Phase 4 - Database:          {db_time:6.3f}s ({db_time/total_time*100:5.1f}%)")
print(f"Phase 5 - Qt init:           {qt_init_time:6.3f}s ({qt_init_time/total_time*100:5.1f}%)")
print("-" * 80)
print(f"TOTAL STARTUP TIME:          {total_time:6.3f}s")
print("="*80)

# Identify bottlenecks
print("\nBOTTLENECKS:")
phases = [
    ("Database loading", db_time),
    ("Scientific libraries", phase2_time),
    ("PySide6", pyside_time),
    ("Qt initialization", qt_init_time),
    ("Standard libraries", phase1_time),
]
phases.sort(key=lambda x: x[1], reverse=True)

for i, (name, time_val) in enumerate(phases[:3], 1):
    if time_val > 0.1:
        print(f"  {i}. {name}: {time_val:.3f}s")

print("\nOPTIMIZATION STRATEGIES:")
if db_time > 0.3:
    print(f"  • Implement lazy database loading (save ~{db_time:.2f}s)")
    print(f"    - Only load database when user accesses it")
    print(f"    - Or split into small startup cache + full database")

if phase2_time > 1.0:
    print(f"  • Use lazy imports for scientific libraries")
    print(f"    - Import only when features are used")

if pyside_time > 0.2:
    print(f"  • PySide6 import is acceptable (<0.2s is good)")

print(f"\n  Potential startup time with optimizations: ~{total_time - db_time:.2f}s")
print("="*80)
