#!/usr/bin/env python3
"""
Dependency Checker for RamanLab Qt6 Version 1.0.2
==================================================
This script checks if your Python environment has all required dependencies
for RamanLab Qt6, reports their versions, and provides installation instructions if needed.

Features:
- Qt6 GUI Framework (PySide6/PyQt6)
- Core Scientific Computing Stack
- Advanced Raman Analysis Capabilities
- Machine Learning and AI Features
- Cross-Platform Compatibility
- Professional Reporting and Export

Updated: 2025-01-26
Release: DTW Performance Enhancement (v1.0.2)
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
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    
    # Minimum Python 3.8 for Qt6 compatibility
    if current_version < (3, 8):
        print("❌ Python 3.8+ is required for Qt6 compatibility")
        print("  Current version is too old for RamanLab v1.0.2")
        print("  To update Python:")
        print("  - Windows: Download from python.org or use Microsoft Store")
        print("  - macOS: Use 'brew install python' or download from python.org")
        print("  - Linux: Use your distribution's package manager")
        return False
    elif current_version < (3, 9):
        print("✅ Python version is sufficient (3.9+ recommended for optimal performance)")
    else:
        print("✅ Python version is excellent for RamanLab")
    
    return current_version >= (3, 8)

def check_package(package_name, min_version=None, component_info=None, import_name=None):
    """Check if a package is installed and get its version."""
    # Use different import name if specified
    actual_import = import_name or package_name
    
    try:
        package = importlib.import_module(actual_import)
        version = "Unknown"
        
        # Try to get version using modern approach first
        version = get_package_version(package_name)
        if version is None:
            # Some packages don't follow standard version reporting
            version = getattr(package, "__version__", "Unknown")
        
        # Special cases for packages that don't follow standard version reporting
        if actual_import == "tkinter" and version in [None, "Unknown"]:
            version = getattr(package, "TkVersion", "Unknown")
        elif actual_import == "PIL" and version in [None, "Unknown"]:
            version = getattr(package, "__version__", "Unknown")
        
        component_suffix = f" ({component_info})" if component_info else ""
        
        if min_version and version not in [None, "Unknown"]:
            try:
                # Simple version comparison for basic cases
                def version_tuple(v):
                    return tuple(map(int, (str(v).split("."))))
                if version_tuple(str(version)) < version_tuple(min_version):
                    print(f"⚠️ {package_name}: Outdated (version: {version}, required: {min_version}+){component_suffix}")
                    return False, version
            except Exception:
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
    print("\n" + "="*70)
    print("QT6 GUI FRAMEWORK CHECK")
    print("="*70)
    
    pyside6_ok = False
    pyqt6_ok = False
    
    # Check PySide6 (recommended)
    try:
        import PySide6
        from PySide6 import QtCore, QtWidgets, QtGui
        version = get_package_version("PySide6") or "Unknown"
        print(f"✅ PySide6: Installed (version: {version}) - Official Qt6 bindings (RECOMMENDED)")
        pyside6_ok = True
        
        # Check Qt version
        try:
            qt_version = QtCore.qVersion()
            print(f"   Qt Framework Version: {qt_version}")
        except:
            pass
            
    except ImportError:
        print("❌ PySide6: Not installed - Official Qt6 bindings (RECOMMENDED)")
    
    # Check PyQt6 (alternative)
    try:
        import PyQt6
        from PyQt6 import QtCore, QtWidgets, QtGui
        version = get_package_version("PyQt6") or "Unknown"
        status = "✅ PyQt6: Installed" if not pyside6_ok else "ℹ️ PyQt6: Also installed"
        print(f"{status} (version: {version}) - Alternative Qt6 bindings")
        pyqt6_ok = True
    except ImportError:
        print("❌ PyQt6: Not installed - Alternative Qt6 bindings")
    
    if pyside6_ok or pyqt6_ok:
        print("\n✅ Qt6 GUI Framework: Available")
        if pyside6_ok and pyqt6_ok:
            print("   📋 Note: Both PySide6 and PyQt6 are installed. PySide6 is recommended for RamanLab.")
        elif pyside6_ok:
            print("   🎯 Using PySide6 (recommended)")
        else:
            print("   ⚠️ Using PyQt6 (consider switching to PySide6 for better compatibility)")
        return True
    else:
        print("\n❌ Qt6 GUI Framework: NOT AVAILABLE")
        print("   🚨 CRITICAL: RamanLab Qt6 requires either PySide6 or PyQt6")
        print("   📦 Install with: pip install PySide6")
        return False

def check_core_scientific_stack():
    """Check core scientific computing packages."""
    print("\n" + "="*70)
    print("CORE SCIENTIFIC COMPUTING STACK")
    print("="*70)
    
    # Updated core packages with minimum versions from requirements_qt6.txt
    core_packages = [
        ("numpy", "1.21.0", "Numerical computations and arrays"),
        ("scipy", "1.7.0", "Scientific computing and optimization"), 
        ("matplotlib", "3.5.0", "Plotting and visualization"),
        ("pandas", "1.3.0", "Data manipulation and analysis"),
        ("seaborn", "0.11.0", "Statistical visualization"),
        ("pillow", "8.0.0", "Image processing", "PIL"),  # PIL is the import name
        ("openpyxl", "3.0.0", "Excel file support"),
        ("fastdtw", "0.3.4", "Dynamic time warping"),
        ("tqdm", "4.60.0", "Progress indicators"),
        ("psutil", "5.8.0", "System utilities"),
        ("scikit-learn", "1.0.0", "Machine learning", "sklearn"),
        ("dask", "2021.0.0", "Parallel computing")
    ]
    
    missing_packages = []
    outdated_packages = []
    all_ok = True
    
    for package_info in core_packages:
        if len(package_info) == 4:
            package, min_version, description, import_name = package_info
        else:
            package, min_version, description = package_info
            import_name = None
            
        installed, version = check_package(package, min_version, description, import_name)
        if not installed:
            missing_packages.append(package)
            all_ok = False
        elif min_version and version not in [None, "Unknown"]:
            try:
                def version_tuple(v):
                    return tuple(map(int, (str(v).split("."))))
                if version_tuple(str(version)) < version_tuple(min_version):
                    outdated_packages.append((package, version, min_version))
                    all_ok = False
            except:
                pass  # Skip version comparison issues
    
    return all_ok, missing_packages, outdated_packages

def check_optional_advanced_packages():
    """Check optional but recommended advanced packages."""
    print("\n" + "="*70)
    print("ADVANCED & OPTIONAL PACKAGES")
    print("="*70)
    
    optional_packages = [
        ("pymatgen", "2022.0.0", "Advanced crystallography and materials analysis"),
        ("reportlab", "3.5.0", "PDF report generation"),
        ("pyinstaller", "5.0.0", "Standalone executable creation"),
        ("emcee", "3.0.0", "MCMC sampling for Bayesian analysis"),
        ("tensorflow", "2.12.0", "Deep learning framework"),
        ("umap-learn", "0.5.0", "UMAP dimensionality reduction")
    ]
    
    available_features = []
    missing_optional = []
    
    for package, min_version, description in optional_packages:
        import_name = "umap" if package == "umap-learn" else package
        installed, version = check_package(package, min_version, description, import_name)
        if installed:
            available_features.append(package)
        else:
            missing_optional.append(package)
    
    return available_features, missing_optional

def check_component_availability(qt6_ok, core_ok, advanced_features):
    """Analyze component availability based on dependencies."""
    print("\n" + "="*70)
    print("RAMANLAB COMPONENT AVAILABILITY")
    print("="*70)
    
    components = {
        "🖥️ Core GUI Interface": qt6_ok,
        "🔬 Spectrum Analysis": core_ok,
        "📊 Data Visualization": core_ok,
        "🎯 Peak Fitting": core_ok,
        "⚡ Batch Processing": core_ok,
        "🧠 Machine Learning": core_ok,  # scikit-learn is now required
        "🗺️ 2D Map Analysis": core_ok,
        "📈 Group Analysis": core_ok,
        "🔍 Polarization Analysis": "pymatgen" in advanced_features,
        "📄 PDF Reports": "reportlab" in advanced_features,
        "🤖 Deep Learning": "tensorflow" in advanced_features,
        "📦 Executable Creation": "pyinstaller" in advanced_features
    }
    
    available_count = 0
    total_count = len(components)
    
    for component, available in components.items():
        status = "✅ AVAILABLE" if available else "❌ LIMITED/UNAVAILABLE"
        print(f"   {component}: {status}")
        if available:
            available_count += 1
    
    print(f"\n📈 Overall Availability: {available_count}/{total_count} components functional")
    
    if available_count == total_count:
        print("🎉 EXCELLENT: All RamanLab features are available!")
    elif available_count >= total_count * 0.8:
        print("✅ GOOD: Most RamanLab features are available!")
    elif available_count >= total_count * 0.6:
        print("⚠️ PARTIAL: Core RamanLab features are available!")
    else:
        print("❌ LIMITED: Many RamanLab features are unavailable!")
    
    return components

def check_system_resources():
    """Check system resources and performance indicators."""
    print("\n" + "="*70)
    print("SYSTEM RESOURCES & PERFORMANCE")
    print("="*70)
    
    try:
        import psutil
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"💾 RAM: {memory_gb:.1f} GB total, {memory.percent}% used")
        
        if memory_gb >= 8:
            print("   ✅ Excellent memory for large dataset analysis")
        elif memory_gb >= 4:
            print("   ✅ Sufficient memory for basic analysis")
        else:
            print("   ⚠️ Limited memory - may affect large dataset performance")
        
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"⚙️ CPU: {cpu_count} cores, {cpu_percent}% current usage")
        
        # Disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        print(f"💽 Disk: {disk_free_gb:.1f} GB free space")
        
        if disk_free_gb >= 5:
            print("   ✅ Sufficient disk space for databases and results")
        elif disk_free_gb >= 2:
            print("   ✅ Adequate disk space for basic operation")
        else:
            print("   ⚠️ Limited disk space - consider cleanup")
            
    except ImportError:
        print("❌ psutil not available - cannot check system resources")

def check_data_files():
    """Check for database and data files."""
    print("\n" + "="*70)
    print("DATA FILES & DATABASES")
    print("="*70)
    
    required_files = [
        ("raman_database.pkl", "Core mineral Raman database"),
        ("RamanLab_Database_20250602.sqlite", "SQLite mineral database"),
        ("RRUFF_Hey_Index.csv", "RRUFF mineral classification data"),
        ("mineral_modes.pkl", "Mineral vibrational modes database")
    ]
    
    optional_files = [
        ("hey_classification_config.json", "Classification configuration"),
        ("saved_models/", "Machine learning models directory"),
        ("__exampleData/", "Example data directory")
    ]
    
    files_found = 0
    total_files = len(required_files)
    
    print("Required Database Files:")
    for filename, description in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024*1024)  # Size in MB
            print(f"✅ {filename}: Found ({size:.1f} MB) - {description}")
            files_found += 1
        else:
            print(f"❌ {filename}: Missing - {description}")
    
    print("\nOptional Files:")
    for filename, description in optional_files:
        if os.path.exists(filename):
            if os.path.isdir(filename):
                print(f"✅ {filename}: Found (directory) - {description}")
            else:
                size = os.path.getsize(filename) / 1024  # Size in KB
                print(f"✅ {filename}: Found ({size:.1f} KB) - {description}")
        else:
            print(f"ℹ️ {filename}: Not found - {description}")
    
    if files_found == total_files:
        print("\n✅ All required database files are present!")
    elif files_found > 0:
        print(f"\n⚠️ {files_found}/{total_files} required database files found")
        print("   Some databases are missing - download from provided links")
    else:
        print("\n❌ No database files found - download required for full functionality")

def suggest_installation_commands(missing_core, missing_optional, outdated_packages):
    """Suggest installation commands for missing packages."""
    print("\n" + "="*70)
    print("INSTALLATION RECOMMENDATIONS")
    print("="*70)
    
    if not missing_core and not missing_optional and not outdated_packages:
        print("🎉 No installation needed - all packages are up to date!")
        return
    
    # Core packages installation
    if missing_core:
        print("🚨 CRITICAL - Install missing core packages:")
        print("pip install " + " ".join(missing_core))
        print("\nOr install all requirements:")
        print("pip install -r requirements_qt6.txt")
    
    # Optional packages
    if missing_optional:
        print("\n📦 OPTIONAL - Advanced features (install as needed):")
        for package in missing_optional:
            if package == "tensorflow":
                print(f"pip install {package}  # Large download - only if using deep learning")
            elif package == "umap-learn":
                print(f"pip install {package}  # For advanced group analysis visualization")
            elif package == "emcee":
                print(f"pip install {package}  # For MCMC statistical analysis")
            else:
                print(f"pip install {package}")
    
    # Outdated packages
    if outdated_packages:
        print("\n🔄 UPDATE - Upgrade outdated packages:")
        for package, current, minimum in outdated_packages:
            print(f"pip install --upgrade {package}  # {current} -> {minimum}+")
    
    # Virtual environment recommendation
    print("\n🐍 RECOMMENDED - Virtual Environment Setup:")
    print("python -m venv ramanlab_env")
    print("source ramanlab_env/bin/activate  # Windows: ramanlab_env\\Scripts\\activate")
    print("pip install -r requirements_qt6.txt")
    print("python check_dependencies.py  # Verify installation")

def main():
    """Main function to check all dependencies."""
    print("=" * 70)
    print(f"🔬 RamanLab v{__version__} - Qt6 Dependency Checker")
    print("=" * 70)
    print("Cross-platform Raman Spectrum Analysis Tool")
    print("Checking system compatibility and dependencies...")
    print(f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check Qt6 GUI framework (most critical)
    qt6_ok = check_qt6_framework()
    
    # Check core scientific stack
    core_ok, missing_core, outdated_core = check_core_scientific_stack()
    
    # Check optional advanced packages
    advanced_features, missing_optional = check_optional_advanced_packages()
    
    # Analyze component availability
    components = check_component_availability(qt6_ok, core_ok, advanced_features)
    
    # Check system resources 
    check_system_resources()
    
    # Check data files
    check_data_files()
    
    # Combine all missing and outdated packages
    all_missing = missing_core + (["PySide6"] if not qt6_ok else [])
    all_outdated = outdated_core
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL ASSESSMENT")
    print("="*70)
    
    # System compatibility
    if not python_ok:
        print("❌ INCOMPATIBLE: Python version too old for RamanLab v1.0.2")
        print("   Please update Python to 3.8+ (3.9+ recommended)")
        return
    
    # Core functionality
    if qt6_ok and core_ok:
        print("✅ READY: RamanLab core functionality is available!")
        print("   All essential features can be used")
    elif qt6_ok:
        print("⚠️ PARTIAL: GUI available but some analysis features limited")
    elif core_ok:
        print("❌ LIMITED: Analysis packages available but no GUI framework")
    else:
        print("❌ NOT READY: Critical dependencies missing")
    
    # Feature summary
    total_components = len(components)
    available_components = sum(1 for available in components.values() if available)
    completion_percent = (available_components / total_components) * 100
    
    print(f"\n📊 Feature Availability: {available_components}/{total_components} ({completion_percent:.0f}%)")
    
    if completion_percent >= 90:
        print("🎉 EXCELLENT: Nearly all RamanLab features available!")
    elif completion_percent >= 70:
        print("✅ GOOD: Most RamanLab features available!")
    elif completion_percent >= 50:
        print("⚠️ BASIC: Core RamanLab features available!")
    else:
        print("❌ LIMITED: Many features unavailable!")
    
    # Installation recommendations
    if all_missing or all_outdated:
        suggest_installation_commands(all_missing, missing_optional, all_outdated)
    else:
        print("\n🎉 CONGRATULATIONS!")
        print("Your environment is fully configured for RamanLab v1.0.2!")
        print("Ready to launch: python launch_ramanlab.py")
    
    print("\n" + "="*70)
    print("For help and documentation: https://github.com/aaroncelestian/RamanLab")
    print("=" * 70)

if __name__ == "__main__":
    main()
