#!/usr/bin/env python3
"""
Script to upload missing files to RamanLab GitHub repository
Excludes large database files and cache directories
"""

import os
import subprocess
import sys
from pathlib import Path

def run_git_command(command):
    """Run a git command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {command}")
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"   Error: {e.stderr.strip()}")
        return False

def check_git_status():
    """Check if we're in a git repository"""
    try:
        subprocess.run(['git', 'status'], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Not in a git repository or git not available")
        return False

def main():
    print("🚀 RamanLab GitHub Upload Script")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not check_git_status():
        return False
    
    # Core application files (Priority 1)
    print("\n📁 Adding Core Application Files...")
    core_files = [
        "main_qt6.py",
        "raman_analysis_app_qt6.py",
        "raman_spectra_qt6.py",
        "requirements_qt6.txt",
        "check_dependencies.py",
        "launch_ramanlab.py",
        "install_desktop_icon.py",
        "version.py",
        "cross_platform_utils.py",
        "database_helpers.py"
    ]
    
    for file in core_files:
        if os.path.exists(file):
            run_git_command(f"git add {file}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # Major feature files (Priority 2)
    print("\n🔧 Adding Major Feature Files...")
    feature_files = [
        "peak_fitting_qt6.py",
        "map_analysis_2d_qt6.py",
        "raman_cluster_analysis_qt6.py",
        "mineral_modes_browser_qt6.py",
        "database_browser_qt6.py",
        "raman_polarization_analyzer_qt6.py",
        "raman_polarization_analyzer_modular_qt6.py",
        "multi_spectrum_manager_qt6.py",
        "spectrum_viewer_qt6.py",
        "database_path_fix.py",
        "utf8_encoding_fix.py",
        "comprehensive_utf8_fix.py"
    ]
    
    for file in feature_files:
        if os.path.exists(file):
            run_git_command(f"git add {file}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # Supporting modules and directories (Priority 3)
    print("\n📂 Adding Supporting Modules...")
    directories = [
        "ui/",
        "core/",
        "parsers/",
        "ml_raman_map/",
        "utils/",
        "analysis/"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            # Add directory but exclude __pycache__
            run_git_command(f"git add {directory} --ignore-errors")
        else:
            print(f"⚠️  Directory not found: {directory}")
    
    # Additional useful files
    print("\n📄 Adding Additional Files...")
    additional_files = [
        "launch_raman_database_browser.py",
        "launch_line_scan_splitter.py",
        "make_mac_style_icon.py",
        "install_trilogy_dependencies.py",
        "migrate_toolbars.py",
        "rename_project.py",
        "fix_tensor_physics.py",
        "tensor_physics_patch.py",
        "missing_methods.py",
        "strain_analysis_example.py",
        "example_advanced_cluster_analysis.py",
        "check_database_access.py"
    ]
    
    for file in additional_files:
        if os.path.exists(file):
            run_git_command(f"git add {file}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # Test files
    print("\n🧪 Adding Test Files...")
    test_files = [
        "test_widget_with_anatase.py",
        "test_cif_loading.py",
        "test_crystal_structure.py"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            run_git_command(f"git add {file}")
        else:
            print(f"⚠️  File not found: {file}")
    
    # Update .gitignore to exclude database files and cache
    print("\n🚫 Updating .gitignore...")
    gitignore_content = """
# Database files (too large for git)
*.pkl
*.sqlite
*.db

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
raman_env/
venv/
env/

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Large data files
Density/
sample_spectra/
__exampleData/

# Backup files
*_backup.py
docs_backup/

# Package builds
packages/
build/
dist/
*.egg-info/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# Log files
*.log

# Configuration files that might contain sensitive data
*.json
!hey_classification_config.json

# CSV data files (can be large)
*.csv
!sample_output.csv
!sample_results.csv
"""
    
    try:
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content.strip())
        run_git_command("git add .gitignore")
    except Exception as e:
        print(f"⚠️  Could not update .gitignore: {e}")
    
    # Show status
    print("\n📊 Current Git Status:")
    subprocess.run(['git', 'status', '--short'], check=False)
    
    print("\n" + "=" * 50)
    print("✅ File addition complete!")
    print("\n📝 Next steps:")
    print("1. Review the changes: git status")
    print("2. Commit the changes: git commit -m 'Add missing core files and modules'")
    print("3. Push to GitHub: git push origin main")
    print("\n💡 Note: Large database files (.pkl, .sqlite) have been excluded")
    print("   Users can download them separately as mentioned in the README")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 