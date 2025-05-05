#!/usr/bin/env python3
"""
Dependency Checker for ClaritySpectra
This script checks if your Python environment has all required dependencies,
reports their versions, and provides installation instructions if needed.
"""

import importlib
import platform
import sys
import pkg_resources
import subprocess
import os

def check_python_version():
    """Check if Python version meets requirements."""
    current_version = sys.version_info
    print(f"Python Version: {platform.python_version()}")
    
    # The code doesn't specify a minimum version, but let's use 3.6 as a safe minimum
    if current_version < (3, 6):
        print("⚠️ Warning: Python 3.6+ recommended for this application")
        print("  To update Python:")
        print("  - Windows: Download the latest installer from python.org")
        print("  - macOS: Use 'brew install python' or download from python.org")
        print("  - Linux: Use your distribution's package manager (apt, yum, etc.)")
    else:
        print("✅ Python version is sufficient")
    
    return current_version >= (3, 6)

def check_package(package_name, min_version=None):
    """Check if a package is installed and get its version."""
    try:
        package = importlib.import_module(package_name)
        version = "Unknown"
        
        try:
            version = pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            # Some packages don't follow standard version reporting
            version = getattr(package, "__version__", "Unknown")
        
        # Special case for tkinter which doesn't have a standard version reporting
        if package_name == "tkinter" and version == "Unknown":
            version = getattr(package, "TkVersion", "Unknown")
        
        if min_version and version != "Unknown":
            if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                print(f"⚠️ {package_name}: Installed but outdated (version: {version}, required: {min_version}+)")
                return False, version
        
        print(f"✅ {package_name}: Installed (version: {version})")
        return True, version
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False, None

def get_reportlab_status():
    """Check if reportlab is installed (for PDF generation)."""
    try:
        import reportlab
        version = pkg_resources.get_distribution("reportlab").version
        print(f"✅ reportlab: Installed (version: {version}) - PDF export available")
        return True, version
    except ImportError:
        print("ℹ️ reportlab: Not installed - PDF export will not be available")
        return False, None

def get_tensorflow_status():
    """Check if TensorFlow is installed (for deep learning)."""
    try:
        import tensorflow as tf
        version = pkg_resources.get_distribution("tensorflow").version
        print(f"✅ TensorFlow: Installed (version: {version}) - Deep learning available")
        return True, version
    except ImportError:
        print("ℹ️ TensorFlow: Not installed - Deep learning will not be available")
        return False, None

def get_keras_status():
    """Check if Keras is installed (for deep learning)."""
    try:
        import keras
        version = pkg_resources.get_distribution("keras").version
        print(f"✅ Keras: Installed (version: {version}) - Deep learning available")
        return True, version
    except ImportError:
        print("ℹ️ Keras: Not installed - Deep learning will not be available")
        return False, None

def get_sklearn_status():
    """Check if scikit-learn is installed (for ML search)."""
    try:
        import sklearn
        version = pkg_resources.get_distribution("scikit-learn").version
        print(f"✅ scikit-learn: Installed (version: {version}) - ML search available")
        return True, version
    except ImportError:
        print("ℹ️ scikit-learn: Not installed - ML search will not be available")
        return False, None

def suggest_install_command(missing_packages):
    """Suggest pip install command for missing packages."""
    if missing_packages:
        install_commands = {
            "numpy": "pip install numpy",
            "matplotlib": "pip install matplotlib",
            "scipy": "pip install scipy",
            "pandas": "pip install pandas",
            "tkinter": "Tkinter is included with Python, but may need additional installation:",
            "reportlab": "pip install reportlab",
            "scikit-learn": "pip install scikit-learn"
        }
        
        print("\nInstallation instructions for missing packages:")
        
        for package in missing_packages:
            print(f"\n{package}:")
            
            # Handle tkinter specially
            if package == "tkinter":
                print("  " + install_commands[package])
                print("  - Windows: Python installer should include tkinter")
                print("  - macOS: 'brew install python-tk' or reinstall Python with tkinter")
                print("  - Debian/Ubuntu: 'sudo apt-get install python3-tk'")
                print("  - Fedora: 'sudo dnf install python3-tkinter'")
            else:
                print("  " + install_commands[package])
                
                # Add extra notes for certain packages
                if package == "numpy":
                    print("  For optimized installation: pip install numpy --no-binary :all:")
                elif package == "scipy":
                    print("  May require compiler tools. For binary installation: pip install scipy --only-binary=scipy")
                elif package == "scikit-learn":
                    print("  Depends on numpy and scipy. Install with: pip install scikit-learn")
        
        # General command for all missing packages (except tkinter)
        regular_packages = [p for p in missing_packages if p != "tkinter"]
        if regular_packages:
            cmd = "pip install " + " ".join(regular_packages)
            print("\nCombined installation command for all missing packages (except tkinter):")
            print(cmd)
            print("\nFor a virtual environment installation (recommended):")
            print("python -m venv raman_env")
            print("source raman_env/bin/activate  # On Windows: raman_env\\Scripts\\activate")
            print(cmd)

def check_package_versions(package_name, min_version=None):
    """Check if the installed package meets the minimum version requirements."""
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
        if min_version:
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                print(f"⚠️ {package_name} version {installed_version} is below the recommended {min_version}")
                print(f"  To upgrade: pip install --upgrade {package_name}")
                return False
        return True
    except (pkg_resources.DistributionNotFound, ImportError):
        return False

def main():
    """Main function to check all dependencies."""
    print("=" * 60)
    print("ClaritySpectra - Dependency Checker")
    print("=" * 60)
    print("\nChecking Python version...")
    python_ok = check_python_version()
    
    print("\nChecking required packages...")
    
    # Core packages required by the application with minimum recommended versions
    required_packages = [
        ("numpy", "1.16.0"), 
        ("matplotlib", "3.0.0"), 
        ("scipy", "1.2.0"), 
        ("pandas", "0.25.0"),
        ("tkinter", None)  # tkinter doesn't follow standard versioning
    ]
    
    missing_packages = []
    outdated_packages = []
    
    # Check core packages
    for package, min_version in required_packages:
        installed, version = check_package(package, min_version)
        if not installed:
            missing_packages.append(package)
        elif min_version and version != "Unknown":
            if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                outdated_packages.append((package, version, min_version))
    
    # Check optional packages with special handling
    print("\nChecking optional packages...")
    reportlab_ok, reportlab_version = get_reportlab_status()
    if not reportlab_ok:
        missing_packages.append("reportlab")
    
    tensorflow_ok, tensorflow_version = get_tensorflow_status()
    if not tensorflow_ok:
        missing_packages.append("tensorflow")
    
    keras_ok, keras_version = get_keras_status()
    if not keras_ok:
        missing_packages.append("keras")
    
    sklearn_ok, sklearn_version = get_sklearn_status()
    if not sklearn_ok:
        missing_packages.append("scikit-learn")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not python_ok:
        print("⚠️ Python version may be too old")
    
    if missing_packages:
        print(f"❌ Missing {len(missing_packages)} required package(s): {', '.join(missing_packages)}")
    
    if outdated_packages:
        print(f"⚠️ {len(outdated_packages)} package(s) need updating:")
        for package, current, minimum in outdated_packages:
            print(f"  - {package}: current {current}, recommended {minimum}+")
    
    if not missing_packages and not outdated_packages:
        print("✅ All required packages are installed with sufficient versions!")
    
    # Installation instructions
    if missing_packages or outdated_packages:
        # For missing packages
        suggest_install_command(missing_packages)
        
        # For outdated packages
        if outdated_packages:
            print("\nUpgrade commands for outdated packages:")
            for package, current, minimum in outdated_packages:
                print(f"pip install --upgrade {package}")
            
            all_outdated = " ".join([p[0] for p in outdated_packages])
            print(f"\nCombined upgrade command: pip install --upgrade {all_outdated}")
        
    # Additional notes about optional features
    print("\nAdditional notes:")
    if not reportlab_ok:
        print("- PDF export functionality will not be available")
    if not tensorflow_ok:
        print("- Deep learning functionality will not be available")
    if not keras_ok:
        print("- Deep learning functionality will not be available")
    if not sklearn_ok:
        print("- Machine learning search functionality will not be available")
    
    # Check for database files
    print("\nChecking for required data files...")
    
    required_files = [
        "raman_database.pkl",
        "RRUFF_Export_with_Hey_Classification.csv"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}: Found")
        else:
            print(f"ℹ️ {file}: Not found - will be created on first run or needs to be provided")
            if file == "RRUFF_Export_with_Hey_Classification.csv":
                print("  This file is required for mineral classification and may need to be downloaded")
                print("  from RRUFF database (https://rruff.info/) or created manually.")
    
    print("\n" + "=" * 60)
    
    # Final recommendation
    if missing_packages or outdated_packages:
        print("\nRECOMMENDATION:")
        print("It's recommended to set up a virtual environment for this application:")
        print("1. Create a virtual environment:")
        print("   python -m venv raman_env")
        print("2. Activate the environment:")
        print("   - Windows: raman_env\\Scripts\\activate")
        print("   - macOS/Linux: source raman_env/bin/activate")
        print("3. Install/upgrade all required packages:")
        
        all_needed = list(set([p for p in missing_packages if p != "tkinter"] + 
                            [p[0] for p in outdated_packages]))
        if all_needed:
            print(f"   pip install {' '.join(all_needed)}")
        
        if "tkinter" in missing_packages:
            print("4. Install tkinter according to your operating system (see instructions above)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
