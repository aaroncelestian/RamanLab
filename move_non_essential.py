#!/usr/bin/env python3
import os
import shutil

# Define core files to keep
core_files = [
    'raman_analysis_app.py',
    'peak_fitting.py',
    'batch_peak_fitting.py',
    'raman_polarization_analyzer.py',
    'raman_group_analysis.py',
    'mineral_database.py',
    'raman_spectra.py',
    'mineral_database.pkl',
    'mineral_modes.pkl',
    'raman_database.pkl',
    '__init__.py',
    'database_helpers.py',
    'check_dependencies.py',
    'move_non_essential.py',  # Keep this script
    'README.md'  # Keep the readme file
]

# Get all files in the current directory
all_files = [f for f in os.listdir('.') if os.path.isfile(f)]

# Destination folder
dest_folder = 'non_essential_files'

# Ensure destination folder exists
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Move non-essential files
moved_files = []
for file in all_files:
    if file not in core_files and not file.startswith('.'):
        try:
            shutil.move(file, os.path.join(dest_folder, file))
            moved_files.append(file)
        except Exception as e:
            print(f"Error moving {file}: {e}")

print(f"Moved {len(moved_files)} non-essential files to {dest_folder}/")
print("Core files kept in the main directory.") 