#!/usr/bin/env python3
"""
Database Path Fix for RamanLab
====================================

This script fixes database path issues that occur when copying the application
between different operating systems (Mac/Linux to Windows).

The issue occurs because some parts of the code use relative paths while others
use absolute paths relative to the script location. This causes "database not found"
errors when running on different systems.

Usage:
------
python database_path_fix.py

This will:
1. Scan all Python files for database path references
2. Update them to use consistent, platform-independent paths
3. Create backup copies of modified files
"""

import os
import sys
import shutil
import re
from pathlib import Path

def get_script_directory():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def backup_file(filepath):
    """Create a backup copy of a file."""
    backup_path = filepath + '.backup'
    shutil.copy2(filepath, backup_path)
    print(f"Created backup: {backup_path}")

def fix_database_paths():
    """Fix database path issues in Python files."""
    script_dir = get_script_directory()
    
    # Database files that should be in the same directory as the scripts
    database_files = [
        'mineral_modes.pkl',
        'raman_database.pkl'
    ]
    
    # Check if database files exist
    missing_files = []
    for db_file in database_files:
        db_path = os.path.join(script_dir, db_file)
        if not os.path.exists(db_path):
            missing_files.append(db_file)
    
    if missing_files:
        print("⚠️  Warning: The following database files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nMake sure you've copied ALL files from the original installation.")
        print("The application may not work properly without these files.\n")
    
    # Files to fix and their patterns
    files_to_fix = {
        'mineral_database.py': [
            {
                'pattern': r'self\.database_path = database_path or os\.path\.join\(os\.path\.dirname\(__file__\), "mineral_modes\.pkl"\)',
                'replacement': 'self.database_path = database_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")'
            }
        ],
        'raman_spectra.py': [
            {
                'pattern': r'self\.db_path = "raman_database\.pkl"',
                'replacement': 'self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raman_database.pkl")'
            }
        ],
        'raman_polarization_analyzer.py': [
            {
                'pattern': r'database_path = os\.path\.join\(os\.path\.dirname\(__file__\), "mineral_modes\.pkl"\)',
                'replacement': 'database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")'
            },
            {
                'pattern': r'db_path = "mineral_modes\.pkl"',
                'replacement': 'db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")'
            }
        ],
        'load_calcite3.py': [
            {
                'pattern': r'db_path = "mineral_modes\.pkl"',
                'replacement': 'db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineral_modes.pkl")'
            }
        ]
    }
    
    print("🔧 Fixing database path issues...")
    print("=" * 50)
    
    fixed_files = []
    
    for filename, patterns in files_to_fix.items():
        filepath = os.path.join(script_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filename}")
            continue
        
        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        # Apply each pattern fix
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            replacement = pattern_info['replacement']
            
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
                print(f"✅ Fixed pattern in {filename}")
        
        # Write the fixed content back
        if modified:
            # Create backup first
            backup_file(filepath)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            fixed_files.append(filename)
            print(f"✅ Updated {filename}")
        else:
            print(f"ℹ️  No changes needed for {filename}")
    
    return fixed_files

def add_os_import():
    """Ensure os module is imported in files that need it."""
    script_dir = get_script_directory()
    
    files_needing_os = ['raman_spectra.py']
    
    for filename in files_needing_os:
        filepath = os.path.join(script_dir, filename)
        
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if os is already imported
        if re.search(r'^import os', content, re.MULTILINE):
            continue
        
        # Add os import after other imports
        lines = content.split('\n')
        import_section_end = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_section_end = i
        
        # Insert os import
        if import_section_end > 0:
            lines.insert(import_section_end + 1, 'import os')
            content = '\n'.join(lines)
            
            # Create backup and write
            backup_file(filepath)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Added 'import os' to {filename}")

def main():
    """Main function to fix database path issues."""
    print("RamanLab Database Path Fix")
    print("================================")
    print()
    print("This script will fix database path issues that occur when")
    print("copying the application between different operating systems.")
    print()
    
    # Add os imports where needed
    add_os_import()
    
    # Fix database paths
    fixed_files = fix_database_paths()
    
    print()
    print("=" * 50)
    print("🎉 Database path fix completed!")
    print()
    
    if fixed_files:
        print("Modified files:")
        for filename in fixed_files:
            print(f"  ✅ {filename}")
        print()
        print("Backup files created with .backup extension")
    else:
        print("No files needed modification.")
    
    print()
    print("💡 Tips:")
    print("  - Make sure all .pkl database files are in the same directory as the Python scripts")
    print("  - If you still get 'database not found' errors, check file permissions")
    print("  - You can restore backups by removing .backup extension if needed")
    print()
    print("The application should now work correctly on Windows!")

if __name__ == "__main__":
    main() 