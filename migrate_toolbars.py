#!/usr/bin/env python3
"""
Toolbar Migration Script for RamanLab PySide6

This script automatically updates PySide6 files to use the compact toolbar
configuration from the ui module.
"""

import re
import os
from pathlib import Path

def migrate_toolbar_imports(file_path):
    """Migrate a single file to use compact toolbars."""
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    original_content = content
    
    # Pattern 1: Replace NavigationToolbar2QT imports
    pattern1 = r'from matplotlib\.backends\.backend_qtagg import NavigationToolbar2QT as NavigationToolbar'
    replacement1 = 'from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar'
    content = re.sub(pattern1, replacement1, content)
    
    pattern2 = r'from matplotlib\.backends\.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar'
    replacement2 = 'from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar'
    content = re.sub(pattern2, replacement2, content)
    
    # Pattern 3: Update try/except import blocks
    try_except_pattern = r'try:\s*\n\s*from matplotlib\.backends\.backend_qtagg import FigureCanvasQTAgg as FigureCanvas\s*\n\s*from matplotlib\.backends\.backend_qtagg import NavigationToolbar2QT as NavigationToolbar\s*\nexcept ImportError:\s*\n\s*from matplotlib\.backends\.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas\s*\n\s*from matplotlib\.backends\.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar'
    
    try_except_replacement = '''try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar'''
    
    content = re.sub(try_except_pattern, try_except_replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Add UI configuration import if not present
    if 'from ui.matplotlib_config import' in content and 'configure_compact_ui' not in content:
        # Find the ui.matplotlib_config import line and add configure_compact_ui
        ui_import_pattern = r'(from ui\.matplotlib_config import [^\\n]*)'
        def add_config_import(match):
            imports = match.group(1)
            if 'configure_compact_ui' not in imports and 'apply_theme' not in imports:
                if imports.endswith(')'):
                    # Multi-line import
                    return imports
                else:
                    # Single line import - add apply_theme
                    return imports + ', apply_theme'
            return imports
        
        content = re.sub(ui_import_pattern, add_config_import, content)
    
    # Add apply_theme call to __init__ methods if not present
    if 'class ' in content and 'apply_theme(' not in content:
        # Find __init__ method and add apply_theme call
        init_pattern = r'(def __init__\(self[^)]*\):\s*\n\s*super\(\).__init__\([^)]*\)\s*\n)'
        def add_theme_call(match):
            init_part = match.group(1)
            return init_part + '        \n        # Apply compact UI configuration for consistent toolbar sizing\n        apply_theme(\'compact\')\n        \n'
        
        content = re.sub(init_pattern, add_theme_call, content)
    
    # Write back if changes were made
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated {file_path}")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    else:
        print(f"ℹ️ No changes needed for {file_path}")
        return True

def main():
    """Main migration function."""
    print("🚀 Starting RamanLab Toolbar Migration")
    print("=" * 50)
    
    # List of Qt6 files to migrate
    qt6_files = [
        'batch_peak_fitting_qt6.py',
        'raman_cluster_analysis_qt6.py', 
        'multi_spectrum_manager_qt6.py',
        'raman_polarization_analyzer_modular_qt6.py',
        'database_browser_qt6.py',
        'peak_fitting_qt6.py'
    ]
    
    # Optional backup files (lower priority)
    backup_files = [
        'mineral_modes_browser_qt6_backup.py'
    ]
    
    # Migrate high priority files
    print("📋 Processing high priority files...")
    success_count = 0
    total_count = 0
    
    for file_path in qt6_files:
        if os.path.exists(file_path):
            total_count += 1
            if migrate_toolbar_imports(file_path):
                success_count += 1
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print()
    print("📋 Processing backup files...")
    
    for file_path in backup_files:
        if os.path.exists(file_path):
            total_count += 1
            if migrate_toolbar_imports(file_path):
                success_count += 1
        else:
            print(f"ℹ️ Backup file not found: {file_path}")
    
    print()
    print("=" * 50)
    print(f"🎯 Migration Complete: {success_count}/{total_count} files processed successfully")
    
    if success_count == total_count:
        print("✅ All files migrated successfully!")
        print()
        print("Next steps:")
        print("1. Test each migrated application")
        print("2. Verify toolbar sizing is consistent")
        print("3. Check for any import errors")
        print("4. Update any remaining files manually if needed")
    else:
        print("⚠️ Some files had issues during migration")
        print("Please check the error messages above and fix manually if needed")

if __name__ == "__main__":
    main() 