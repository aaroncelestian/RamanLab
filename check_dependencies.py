#!/usr/bin/env python3
"""
Dependency Checker for RamanLab Qt6 Version 2.6.3
==================================================
This script checks if your Python environment has all required dependencies
for RamanLab Qt6, reports their versions, and provides installation instructions if needed.

Features:
- Qt6 GUI Framework (PySide6/PyQt6)
- Core Scientific Computing
- Advanced Raman Analysis
- Multi-Spectrum Management
- Cross-Platform Compatibility
"""

import importlib
import platform
import sys
import subprocess
import os
from version import __version__  # Import version from version.py

# Use modern importlib.metadata when available, fallback to pkg_resources
try:
    from importlib.metadata import version as get_version, PackageNotFoundError
    def get_package_version(package_name):
        try:
            return get_version(package_name)
        except PackageNotFoundError:
            return None
    def parse_version_func(version_str):
        # Simple version comparison - split by dots and compare numerically
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        return version_tuple
    parse_version = parse_version_func
except ImportError:
    # Fallback to pkg_resources (deprecated)
    import pkg_resources
    def get_package_version(package_name):
        try:
            return pkg_resources.get_distribution(package_name).version
        except pkg_resources.DistributionNotFound:
            return None
    parse_version = pkg_resources.parse_version

def check_python_version():
    """Check if Python version meets requirements."""
    current_version = sys.version_info
    print(f"Python Version: {platform.python_version()}")
    
    # Minimum Python 3.8 for Qt6 compatibility
    if current_version < (3, 8):
        print("❌ Python 3.8+ is required for Qt6 compatibility")
        print("  To update Python:")
        print("  - Windows: Download the latest installer from python.org")
        print("  - macOS: Use 'brew install python' or download from python.org")
        print("  - Linux: Use your distribution's package manager (apt, yum, etc.)")
        return False
    elif current_version < (3, 9):
        print("✅ Python version is sufficient (3.9+ recommended for optimal performance)")
    else:
        print("✅ Python version is excellent")
    
    return current_version >= (3, 8)

def check_package(package_name, min_version=None, component_info=None):
    """Check if a package is installed and get its version."""
    try:
        package = importlib.import_module(package_name)
        version = "Unknown"
        
        # Try to get version using modern approach first
        version = get_package_version(package_name)
        if version is None:
            # Some packages don't follow standard version reporting
            version = getattr(package, "__version__", "Unknown")
        
        # Special case for tkinter which doesn't have a standard version reporting
        if package_name == "tkinter" and version in [None, "Unknown"]:
            version = getattr(package, "TkVersion", "Unknown")
        
        component_suffix = f" ({component_info})" if component_info else ""
        
        if min_version and version not in [None, "Unknown"]:
            try:
                # Simple version comparison for basic cases
                def version_tuple(v):
                    return tuple(map(int, (v.split("."))))
                if version_tuple(str(version)) < version_tuple(min_version):
                    print(f"⚠️ {package_name}: Installed but outdated (version: {version}, required: {min_version}+){component_suffix}")
                    return False, version
            except:
                # Fallback for version comparison issues
                pass
        
        print(f"✅ {package_name}: Installed (version: {version}){component_suffix}")
        return True, version
    except ImportError:
        component_suffix = f" - {component_info}" if component_info else ""
        print(f"❌ {package_name}: Not installed{component_suffix}")
        return False, None

def check_qt6_framework():
    """Check Qt6 GUI framework availability."""
    print("\n" + "="*60)
    print("QT6 GUI FRAMEWORK CHECK")
    print("="*60)
    
    pyside6_ok = False
    pyqt6_ok = False
    
    # Check PySide6 (recommended)
    try:
        import PySide6
        from PySide6 import QtCore, QtWidgets, QtGui
        version = get_package_version("PySide6") or "Unknown"
        print(f"✅ PySide6: Installed (version: {version}) - Official Qt6 bindings (recommended)")
        pyside6_ok = True
    except ImportError:
        print("❌ PySide6: Not installed - Official Qt6 bindings")
    
    # Check PyQt6 (alternative)
    try:
        import PyQt6
        from PyQt6 import QtCore, QtWidgets, QtGui
        version = get_package_version("PyQt6") or "Unknown"
        print(f"✅ PyQt6: Installed (version: {version}) - Alternative Qt6 bindings")
        pyqt6_ok = True
    except ImportError:
        print("❌ PyQt6: Not installed - Alternative Qt6 bindings")
    
    if pyside6_ok or pyqt6_ok:
        print("\n✅ Qt6 GUI Framework: Available")
        if pyside6_ok and pyqt6_ok:
            print("   Note: Both PySide6 and PyQt6 are installed. PySide6 is recommended.")
        return True
    else:
        print("\n❌ Qt6 GUI Framework: NOT AVAILABLE")
        print("   RamanLab Qt6 requires either PySide6 or PyQt6")
        return False

def check_component_specific_dependencies(qt6_ok, sklearn_ok, emcee_ok):
    """Check dependencies specific to each RamanLab component."""
    print("\n" + "="*60)
    print("COMPONENT-SPECIFIC DEPENDENCY ANALYSIS")
    print("="*60)
    
    component_status = {
        'gui': {'available': qt6_ok, 'missing': []},
        'analysis': {'available': True, 'missing': []},
        'advanced': {'available': True, 'missing': []},
        'machine_learning': {'available': sklearn_ok, 'missing': []}
    }
    
    # GUI Framework
    print("\n🖥️ GUI Framework (Core Interface):")
    if not qt6_ok:
        component_status['gui']['missing'].append("PySide6 or PyQt6")
        print("   ❌ Qt6 framework required for GUI interface")
    else:
        print("   ✅ Qt6 framework available")
    
    # Core Analysis
    print("\n🔬 Core Analysis (Spectrum Processing):")
    print("   Dependencies: numpy, scipy, matplotlib, pandas")
    print("   ✅ Core analysis functionality available")
    
    # Advanced Analysis
    print("\n🧠 Advanced Analysis (Machine Learning & MCMC):")
    if not sklearn_ok:
        component_status['advanced']['missing'].append("scikit-learn")
        print("   ⚠️  Machine learning features limited without scikit-learn")
    else:
        print("   ✅ Machine learning features available")
    
    if not emcee_ok:
        component_status['advanced']['missing'].append("emcee")
        print("   ⚠️  MCMC sampling not available without emcee")
    else:
        print("   ✅ MCMC sampling available")
    
    # Multi-Spectrum Management
    print("\n📊 Multi-Spectrum Management:")
    print("   Dependencies: Core packages + Qt6")
    if qt6_ok:
        print("   ✅ Multi-spectrum manager fully functional")
    else:
        print("   ❌ Multi-spectrum manager requires Qt6")
    
    return component_status

def get_emcee_status():
    """Check if emcee is installed (for MCMC sampling)."""
    try:
        import emcee
        version = get_package_version("emcee") or "Unknown"
        print(f"✅ emcee: Installed (version: {version}) - MCMC sampling available")
        return True, version
    except ImportError:
        print("❌ emcee: Not installed - MCMC sampling will not be available")
        return False, None

def get_sklearn_status():
    """Check if scikit-learn is installed with specific components."""
    try:
        import sklearn
        version = get_package_version("scikit-learn") or "Unknown"
        
        # Check specific components needed
        components_available = []
        components_missing = []
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            components_available.append("Gaussian Processes")
        except ImportError:
            components_missing.append("Gaussian Processes")
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            components_available.append("Ensemble Methods")
        except ImportError:
            components_missing.append("Ensemble Methods")
        
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.cluster import DBSCAN
            components_available.append("Clustering")
        except ImportError:
            components_missing.append("Clustering")
        
        print(f"✅ scikit-learn: Installed (version: {version})")
        if components_available:
            print(f"   Available: {', '.join(components_available)}")
        if components_missing:
            print(f"   Missing: {', '.join(components_missing)}")
        
        return True, version, components_available, components_missing
    except ImportError:
        print("❌ scikit-learn: Not installed - Advanced analysis features will not be available")
        return False, None, [], []

def get_reportlab_status():
    """Check if reportlab is installed (for PDF generation)."""
    try:
        import reportlab
        version = get_package_version("reportlab") or "Unknown"
        print(f"✅ reportlab: Installed (version: {version}) - PDF export available")
        return True, version
    except ImportError:
        print("ℹ️ reportlab: Not installed - PDF export will not be available")
        return False, None

def get_tensorflow_status():
    """Check if TensorFlow is installed (for deep learning)."""
    try:
        import tensorflow as tf
        version = get_package_version("tensorflow") or "Unknown"
        print(f"✅ TensorFlow: Installed (version: {version}) - Deep learning available")
        return True, version
    except ImportError:
        print("ℹ️ TensorFlow: Not installed - Deep learning will not be available")
        return False, None

def get_keras_status():
    """Check if Keras is installed (for deep learning)."""
    try:
        import keras
        version = get_package_version("keras") or "Unknown"
        print(f"✅ Keras: Installed (version: {version}) - Deep learning available")
        return True, version
    except ImportError:
        print("ℹ️ Keras: Not installed - Deep learning will not be available")
        return False, None

def get_pymatgen_status():
    """Check if pymatgen is installed (for crystallographic analysis)."""
    try:
        import pymatgen
        version = get_package_version("pymatgen") or "Unknown"
        print(f"✅ pymatgen: Installed (version: {version}) - Advanced crystallographic analysis available")
        return True, version
    except ImportError:
        print("ℹ️ pymatgen: Not installed - Advanced crystallographic analysis will not be available")
        return False, None

def get_pyinstaller_status():
    """Check if PyInstaller is installed (for creating standalone executables)."""
    try:
        import PyInstaller
        version = get_package_version("pyinstaller") or "Unknown"
        print(f"✅ PyInstaller: Installed (version: {version}) - Executable creation available")
        return True, version
    except ImportError:
        print("ℹ️ PyInstaller: Not installed - Standalone executable creation will not be available")
        return False, None

def suggest_install_command(missing_packages, component_status):
    """Suggest pip install command for missing packages with component-specific guidance."""
    if missing_packages:
        install_commands = {
            "numpy": "pip install numpy",
            "matplotlib": "pip install matplotlib",
            "scipy": "pip install scipy",
            "pandas": "pip install pandas",
            "PySide6": "pip install PySide6",
            "PyQt6": "pip install PyQt6",
            "tkinter": "Tkinter is included with Python, but may need additional installation:",
            "reportlab": "pip install reportlab",
            "scikit-learn": "pip install scikit-learn",
            "emcee": "pip install emcee",
            "seaborn": "pip install seaborn",
            "mplcursors": "pip install mplcursors",
            "openpyxl": "pip install openpyxl",
            "PIL": "pip install pillow",
            "fastdtw": "pip install fastdtw",
            "tensorflow": "pip install tensorflow",
            "keras": "pip install keras",
            "pymatgen": "pip install pymatgen",
            "pyinstaller": "pip install pyinstaller"
        }
        
        print("\n" + "="*60)
        print("INSTALLATION RECOMMENDATIONS")
        print("="*60)
        
        # Component-specific recommendations
        print("\n📋 Component-Specific Installation Recommendations:")
        
        if not component_status['gui']['available']:
            print("\n🖥️ For GUI Framework (REQUIRED):")
            print("   pip install PySide6  # Recommended (LGPL license)")
            print("   # OR")
            print("   pip install PyQt6   # Alternative (GPL/Commercial license)")
        
        if not component_status['analysis']['available'] or component_status['analysis']['missing']:
            print("\n🔬 For Core Analysis:")
            analysis_packages = [p for p in component_status['analysis']['missing'] if p != "tkinter"]
            if analysis_packages:
                print(f"   pip install {' '.join(analysis_packages)}")
        
        if not component_status['advanced']['available'] or component_status['advanced']['missing']:
            print("\n🧠 For Advanced Analysis:")
            advanced_packages = [p for p in component_status['advanced']['missing'] if p != "tkinter"]
            if advanced_packages:
                print(f"   pip install {' '.join(advanced_packages)}")
        
        print("\n📦 Individual Package Installation:")
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
                if package == "PySide6":
                    print("  Recommended Qt6 bindings with LGPL license")
                elif package == "PyQt6":
                    print("  Alternative Qt6 bindings with GPL/Commercial license")
                elif package == "scikit-learn":
                    print("  Essential for advanced analysis features")
                elif package == "emcee":
                    print("  Required for MCMC sampling functionality")
        
        # Comprehensive installation commands
        print("\n🔧 Comprehensive Installation Commands:")
        
        # Essential packages
        essential_packages = ["PySide6", "numpy", "matplotlib", "scipy", "pandas"]
        essential_missing = [p for p in missing_packages if p in essential_packages]
        if essential_missing:
            print("\nEssential packages (minimum functionality):")
            print(f"pip install {' '.join(essential_missing)}")
        
        # Full functionality
        full_packages = ["seaborn", "pillow", "mplcursors", "reportlab", "openpyxl", "fastdtw", "scikit-learn", "emcee"]
        full_missing = [p for p in missing_packages if p in full_packages]
        if full_missing:
            print("\nFull functionality:")
            print(f"pip install {' '.join(full_missing)}")
        
        # All packages
        regular_packages = [p for p in missing_packages if p not in ["tkinter"]]
        if regular_packages:
            print("\nComplete installation (all features):")
            print(f"pip install {' '.join(regular_packages)}")
        
        print("\n🐍 Virtual Environment Setup (Recommended):")
        print("python -m venv ramanlab_env")
        print("source ramanlab_env/bin/activate  # On Windows: ramanlab_env\\Scripts\\activate")
        print("pip install -r requirements_qt6.txt")

def main():
    """Main function to check all dependencies."""
    print("=" * 70)
    print(f"RamanLab v{__version__} - Qt6 Dependency Checker")
    print("=" * 70)
    print("\nChecking Python version...")
    python_ok = check_python_version()
    
    # Check Qt6 GUI framework first (most critical)
    qt6_ok = check_qt6_framework()
    
    print("\nChecking core packages...")
    
    # Core packages required by the application with updated minimum versions
    required_packages = [
        ("numpy", "1.19.0"), 
        ("matplotlib", "3.3.0"), 
        ("scipy", "1.6.0"), 
        ("pandas", "1.2.0"),
        ("tkinter", None),  # tkinter doesn't follow standard versioning
        ("seaborn", "0.11.0"),
        ("mplcursors", "0.5.0"),
        ("fastdtw", "0.3.4"),
        ("openpyxl", "3.0.0"),
        ("PIL", "8.0.0")  # Pillow is imported as PIL
    ]
    
    missing_packages = []
    outdated_packages = []
    
    # Check core packages
    for package, min_version in required_packages:
        installed, version = check_package(package, min_version)
        if not installed:
            missing_packages.append(package)
        elif min_version and version not in [None, "Unknown"]:
            try:
                # Simple version comparison for basic cases
                def version_tuple(v):
                    return tuple(map(int, (v.split("."))))
                if version_tuple(str(version)) < version_tuple(min_version):
                    outdated_packages.append((package, version, min_version))
            except:
                pass  # Skip version comparison issues
    
    # Add Qt6 to missing packages if not available
    if not qt6_ok:
        missing_packages.append("PySide6")  # Default recommendation
    
    # Check advanced packages
    print("\nChecking advanced packages...")
    emcee_ok, emcee_version = get_emcee_status()
    if not emcee_ok:
        missing_packages.append("emcee")
    
    sklearn_ok, sklearn_version, sklearn_components, sklearn_missing = get_sklearn_status()
    if not sklearn_ok:
        missing_packages.append("scikit-learn")
    
    # Check component-specific dependencies
    component_status = check_component_specific_dependencies(qt6_ok, sklearn_ok, emcee_ok)
    
    # Check optional packages
    print("\nChecking optional packages...")
    reportlab_ok, reportlab_version = get_reportlab_status()
    if not reportlab_ok:
        missing_packages.append("reportlab")
    
    tensorflow_ok, tensorflow_version = get_tensorflow_status()
    keras_ok, keras_version = get_keras_status()
    pymatgen_ok, pymatgen_version = get_pymatgen_status()
    pyinstaller_ok, pyinstaller_version = get_pyinstaller_status()
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 70)
    
    if not python_ok:
        print("❌ Python version is too old for Qt6")
    
    # Component availability summary
    print("\n📊 Component Availability Summary:")
    components = [
        ("🖥️ GUI Framework", component_status['gui']['available']),
        ("🔬 Core Analysis", component_status['analysis']['available']),
        ("🧠 Advanced Analysis", component_status['advanced']['available']),
        ("📊 Multi-Spectrum Manager", qt6_ok)
    ]
    
    for component_name, available in components:
        status = "✅ AVAILABLE" if available else "❌ LIMITED/UNAVAILABLE"
        print(f"   {component_name}: {status}")
    
    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} package(s): {', '.join(missing_packages)}")
    
    if outdated_packages:
        print(f"\n⚠️ {len(outdated_packages)} package(s) need updating:")
        for package, current, minimum in outdated_packages:
            print(f"  - {package}: current {current}, recommended {minimum}+")
    
    if not missing_packages and not outdated_packages and qt6_ok:
        print("\n✅ All required packages are installed with sufficient versions!")
        print("🎉 RamanLab Qt6 is ready to use!")
    
    # Installation instructions
    if missing_packages or outdated_packages:
        suggest_install_command(missing_packages, component_status)
        
        # For outdated packages
        if outdated_packages:
            print("\n🔄 Upgrade commands for outdated packages:")
            for package, current, minimum in outdated_packages:
                print(f"pip install --upgrade {package}")
            
            all_outdated = " ".join([p[0] for p in outdated_packages])
            print(f"\nCombined upgrade command: pip install --upgrade {all_outdated}")
    
    # Feature availability summary
    print("\n" + "=" * 70)
    print("FEATURE AVAILABILITY SUMMARY")
    print("=" * 70)
    
    # Check if core packages are available (numpy, scipy, matplotlib, pandas)
    core_packages_needed = ["numpy", "scipy", "matplotlib", "pandas"]
    core_analysis_available = all(p not in missing_packages for p in core_packages_needed)
    
    features = [
        ("Qt6 GUI Framework", qt6_ok),
        ("Core Raman Analysis", core_analysis_available),
        ("Multi-Spectrum Management", qt6_ok),
        ("PDF Export", reportlab_ok),
        ("MCMC Sampling", emcee_ok),
        ("Machine Learning", sklearn_ok),
        ("Deep Learning", tensorflow_ok and keras_ok),
        ("Advanced Crystallography", pymatgen_ok),
        ("Executable Creation", pyinstaller_ok)
    ]
    
    for feature_name, available in features:
        status = "✅ Available" if available else "❌ Not Available"
        print(f"   {feature_name}: {status}")
    
    # Check for database files
    print("\n" + "=" * 70)
    print("DATA FILES CHECK")
    print("=" * 70)
    
    required_files = [
        ("raman_database.pkl", "Core mineral database"),
        ("RamanLab_Database_20250602.sqlite", "SQLite mineral database"),
        ("RRUFF_Hey_Index.csv", "RRUFF mineral classification data"),
        ("mineral_modes.pkl", "Mineral mode database")
    ]
    
    for file, description in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # Size in MB
            print(f"✅ {file}: Found ({size:.1f} MB) - {description}")
        else:
            print(f"ℹ️ {file}: Not found - {description}")
    
    print("\n" + "=" * 70)
    
    # Final recommendation
    if missing_packages or outdated_packages or not qt6_ok:
        print("\n🎯 FINAL RECOMMENDATION:")
        print("Set up a virtual environment for optimal RamanLab Qt6 experience:")
        print("\n1. Create and activate virtual environment:")
        print("   python -m venv ramanlab_env")
        print("   source ramanlab_env/bin/activate  # Windows: ramanlab_env\\Scripts\\activate")
        print("\n2. Install requirements:")
        print("   pip install -r requirements_qt6.txt")
        print("\n3. Verify installation:")
        print("   python check_dependencies.py")
        
        if "tkinter" in missing_packages:
            print("\n4. Install tkinter according to your operating system (see instructions above)")
    else:
        print("\n🎉 CONGRATULATIONS!")
        print("Your environment is fully configured for RamanLab Qt6!")
        print("All components are available with full functionality.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
