#!/usr/bin/env python3
"""
RamanLab Dependency Update Script
=================================
This script helps users update their RamanLab dependencies to the latest versions.
It checks for outdated packages and provides options to update them.

Usage:
    python update_dependencies.py           # Interactive mode
    python update_dependencies.py --all     # Update all packages
    python update_dependencies.py --jupyter # Update only Jupyter packages
    python update_dependencies.py --core    # Update only core packages

Features:
- Safe dependency updates with backup recommendations
- Interactive mode for selective updates
- Jupyter console integration updates
- Core package updates
- Virtual environment detection and recommendations

Updated: 2025-01-26
"""

import subprocess
import sys
import argparse
import importlib.util
from pathlib import Path

try:
    from importlib.metadata import version as get_version, PackageNotFoundError
except ImportError:
    import pkg_resources
    def get_version(package_name):
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None

def check_virtual_environment():
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  WARNING: You're not in a virtual environment!")
        print("   It's recommended to use a virtual environment for RamanLab.")
        print("   To create one:")
        print("   python -m venv ramanlab_env")
        print("   source ramanlab_env/bin/activate  # Windows: ramanlab_env\\Scripts\\activate")
        print()
        
        response = input("Continue without virtual environment? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted. Please set up a virtual environment first.")
            return False
    else:
        print("✅ Running in virtual environment")
    
    return True

def get_package_version_safe(package_name):
    """Safely get package version."""
    try:
        return get_version(package_name)
    except (PackageNotFoundError, Exception):
        return None

def run_pip_command(command):
    """Run a pip command and return success status."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def update_core_packages():
    """Update core RamanLab packages."""
    print("\n" + "="*60)
    print("UPDATING CORE PACKAGES")
    print("="*60)
    
    core_packages = [
        "PySide6",
        "numpy", 
        "scipy",
        "matplotlib",
        "pandas",
        "seaborn",
        "pillow",
        "openpyxl",
        "fastdtw",
        "tqdm",
        "psutil",
        "requests",
        "packaging",
        "pyperclip",
        "scikit-learn",
        "joblib",
        "dask",
        "chardet"
    ]
    
    print("Updating core packages...")
    command = f"pip install --upgrade {' '.join(core_packages)}"
    
    print(f"Running: {command}")
    success, output = run_pip_command(command)
    
    if success:
        print("✅ Core packages updated successfully!")
        print(output)
    else:
        print("❌ Failed to update core packages!")
        print(output)
        return False
    
    return True

def update_jupyter_packages():
    """Update Jupyter console packages."""
    print("\n" + "="*60)
    print("UPDATING JUPYTER CONSOLE PACKAGES")
    print("="*60)
    
    jupyter_packages = [
        "qtconsole",
        "jupyter-client", 
        "ipykernel"
    ]
    
    print("Updating Jupyter packages...")
    command = f"pip install --upgrade {' '.join(jupyter_packages)}"
    
    print(f"Running: {command}")
    success, output = run_pip_command(command)
    
    if success:
        print("✅ Jupyter packages updated successfully!")
        print("🐍 Interactive console features are now available!")
        print(output)
    else:
        print("❌ Failed to update Jupyter packages!")
        print(output)
        return False
    
    return True

def update_optional_packages():
    """Update optional advanced packages."""
    print("\n" + "="*60)
    print("UPDATING OPTIONAL PACKAGES")
    print("="*60)
    
    optional_packages = [
        "umap-learn",
        "pymatgen", 
        "reportlab",
        "pyinstaller"
    ]
    
    print("Updating optional packages...")
    print("Note: Some packages may fail to install - this is normal.")
    
    for package in optional_packages:
        print(f"\nUpdating {package}...")
        command = f"pip install --upgrade {package}"
        success, output = run_pip_command(command)
        
        if success:
            print(f"✅ {package} updated successfully!")
        else:
            print(f"⚠️  {package} failed to update (this is optional)")
    
    return True

def update_all_packages():
    """Update all RamanLab packages."""
    print("\n" + "="*60)
    print("UPDATING ALL PACKAGES")
    print("="*60)
    
    print("This will update all RamanLab dependencies from requirements_qt6.txt")
    
    if not Path("requirements_qt6.txt").exists():
        print("❌ requirements_qt6.txt not found!")
        print("   Please run this script from the RamanLab directory.")
        return False
    
    command = "pip install --upgrade -r requirements_qt6.txt"
    print(f"Running: {command}")
    
    success, output = run_pip_command(command)
    
    if success:
        print("✅ All packages updated successfully!")
        print(output)
    else:
        print("❌ Some packages failed to update!")
        print(output)
        print("\nTrying core packages only...")
        return update_core_packages()
    
    return True

def show_package_status():
    """Show current package versions."""
    print("\n" + "="*60)
    print("CURRENT PACKAGE STATUS")
    print("="*60)
    
    packages_to_check = [
        ("PySide6", "Qt6 GUI Framework"),
        ("numpy", "Numerical computing"),
        ("scipy", "Scientific computing"),
        ("matplotlib", "Plotting"),
        ("pandas", "Data analysis"),
        ("scikit-learn", "Machine learning"),
        ("qtconsole", "Jupyter console"),
        ("jupyter-client", "Jupyter client"),
        ("ipykernel", "IPython kernel")
    ]
    
    for package, description in packages_to_check:
        version = get_package_version_safe(package)
        if version:
            print(f"✅ {package}: {version} - {description}")
        else:
            print(f"❌ {package}: Not installed - {description}")

def interactive_update():
    """Interactive update mode."""
    print("\n" + "="*60)
    print("INTERACTIVE UPDATE MODE")
    print("="*60)
    
    print("What would you like to update?")
    print("1. Core packages only (recommended)")
    print("2. Jupyter console packages (new!)")
    print("3. All packages (core + optional)")
    print("4. Show current package status")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            return update_core_packages()
        elif choice == "2":
            return update_jupyter_packages()
        elif choice == "3":
            return update_all_packages()
        elif choice == "4":
            show_package_status()
            continue
        elif choice == "5":
            print("Goodbye!")
            return True
        else:
            print("Invalid choice. Please enter 1-5.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update RamanLab dependencies")
    parser.add_argument("--all", action="store_true", help="Update all packages")
    parser.add_argument("--core", action="store_true", help="Update core packages only")
    parser.add_argument("--jupyter", action="store_true", help="Update Jupyter packages only")
    parser.add_argument("--status", action="store_true", help="Show package status")
    
    args = parser.parse_args()
    
    print("RamanLab Dependency Update Script")
    print("=================================")
    print()
    
    # Check virtual environment
    if not check_virtual_environment():
        return
    
    # Handle command line arguments
    if args.status:
        show_package_status()
        return
    elif args.all:
        success = update_all_packages()
    elif args.core:
        success = update_core_packages()
    elif args.jupyter:
        success = update_jupyter_packages()
    else:
        success = interactive_update()
    
    if success:
        print("\n✅ Update completed successfully!")
        print("🚀 You can now run RamanLab with the latest features!")
        print("\nTo verify the installation, run:")
        print("   python check_dependencies.py")
        print("\nTo test the new Jupyter console:")
        print("   python launch_jupyter_console.py")
    else:
        print("\n❌ Update failed!")
        print("Please check the error messages above.")
        print("You can try updating packages individually:")
        print("   pip install --upgrade PySide6")
        print("   pip install --upgrade qtconsole jupyter-client ipykernel")

if __name__ == "__main__":
    main() 