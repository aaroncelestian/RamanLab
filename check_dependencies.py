#!/usr/bin/env python3
"""
Dependency Checker for ClaritySpectra - Complete Trilogy Edition
================================================================
This script checks if your Python environment has all required dependencies
for the complete ClaritySpectra crystal orientation optimization trilogy,
reports their versions, and provides installation instructions if needed.

Supports:
- Stage 1: Enhanced Individual Peak Optimization
- Stage 2: Probabilistic Bayesian Framework  
- Stage 3: Advanced Multi-Objective Bayesian Optimization
"""

import importlib
import platform
import sys
import pkg_resources
import subprocess
import os
from version import __version__  # Import version from version.py

def check_python_version():
    """Check if Python version meets requirements."""
    current_version = sys.version_info
    print(f"Python Version: {platform.python_version()}")
    
    # Minimum Python 3.6 for modern features
    if current_version < (3, 6):
        print("⚠️ Warning: Python 3.6+ recommended for this application")
        print("  To update Python:")
        print("  - Windows: Download the latest installer from python.org")
        print("  - macOS: Use 'brew install python' or download from python.org")
        print("  - Linux: Use your distribution's package manager (apt, yum, etc.)")
        return False
    elif current_version < (3, 8):
        print("✅ Python version is sufficient (3.8+ recommended for optimal performance)")
    else:
        print("✅ Python version is excellent")
    
    return current_version >= (3, 6)

def check_package(package_name, min_version=None, stage_info=None):
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
        
        stage_suffix = f" ({stage_info})" if stage_info else ""
        
        if min_version and version != "Unknown":
            try:
                if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                    print(f"⚠️ {package_name}: Installed but outdated (version: {version}, required: {min_version}+){stage_suffix}")
                    return False, version
            except:
                # Fallback for version comparison issues
                pass
        
        print(f"✅ {package_name}: Installed (version: {version}){stage_suffix}")
        return True, version
    except ImportError:
        stage_suffix = f" - {stage_info}" if stage_info else ""
        print(f"❌ {package_name}: Not installed{stage_suffix}")
        return False, None

def check_stage_specific_dependencies_with_status(emcee_ok, sklearn_ok):
    """Check dependencies specific to each optimization stage with known status."""
    print("\n" + "="*60)
    print("STAGE-SPECIFIC DEPENDENCY ANALYSIS")
    print("="*60)
    
    stage_status = {
        'stage1': {'available': True, 'missing': []},
        'stage2': {'available': True, 'missing': []},
        'stage3': {'available': True, 'missing': []}
    }
    
    # Stage 1: Enhanced Individual Peak Optimization
    print("\n🚀 Stage 1 (Enhanced Individual Peak Optimization):")
    print("   Dependencies: Core packages only")
    
    stage1_deps = [
        ("numpy", "1.16.0"),
        ("scipy", "1.2.0"),
        ("matplotlib", "3.0.0"),
        ("tkinter", None)
    ]
    
    for package, min_version in stage1_deps:
        installed, version = check_package(package, min_version, "Stage 1 core")
        if not installed:
            stage_status['stage1']['available'] = False
            stage_status['stage1']['missing'].append(package)
    
    # Stage 2: Probabilistic Bayesian Framework
    print("\n🧠 Stage 2 (Probabilistic Bayesian Framework):")
    print("   Dependencies: Core packages + emcee + scikit-learn")
    
    if not emcee_ok:
        stage_status['stage2']['available'] = False
        stage_status['stage2']['missing'].append("emcee")
        print("   ⚠️  Stage 2 will run with reduced functionality (no MCMC sampling)")
    else:
        print("   ✅ MCMC sampling available")
    
    if not sklearn_ok:
        stage_status['stage2']['missing'].append("scikit-learn")
        print("   ⚠️  Stage 2 will run with reduced clustering functionality")
    else:
        print("   ✅ Clustering functionality available")
    
    # Stage 3: Advanced Multi-Objective Bayesian Optimization
    print("\n🌟 Stage 3 (Advanced Multi-Objective Bayesian Optimization):")
    print("   Dependencies: Core packages + scikit-learn + emcee")
    
    if not sklearn_ok:
        stage_status['stage3']['available'] = False
        stage_status['stage3']['missing'].append("scikit-learn")
        print("   ❌ Stage 3 requires scikit-learn for Gaussian Processes")
    else:
        print("   ✅ Gaussian Processes available")
    
    if not emcee_ok:
        stage_status['stage3']['missing'].append("emcee")
        print("   ⚠️  Stage 3 will run with reduced MCMC functionality")
    else:
        print("   ✅ MCMC functionality available")
    
    # Additional Stage 3 specific checks
    if sklearn_ok:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.ensemble import RandomForestRegressor
            print("   ✅ Advanced ensemble methods available")
        except ImportError:
            print("   ❌ Advanced Stage 3 features not available (sklearn components missing)")
            stage_status['stage3']['available'] = False
    
    return stage_status

def get_emcee_status():
    """Check if emcee is installed (for MCMC sampling in Stages 2 & 3)."""
    try:
        import emcee
        version = pkg_resources.get_distribution("emcee").version
        print(f"✅ emcee: Installed (version: {version}) - MCMC sampling available for Stages 2 & 3")
        return True, version
    except ImportError:
        print("❌ emcee: Not installed - MCMC sampling will not be available in Stages 2 & 3")
        return False, None

def get_sklearn_status():
    """Check if scikit-learn is installed with specific components for Stages 2 & 3."""
    try:
        import sklearn
        version = pkg_resources.get_distribution("scikit-learn").version
        
        # Check specific components needed for stages
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
        print("❌ scikit-learn: Not installed - Advanced optimization features will not be available")
        return False, None, [], []

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

def get_pymatgen_status():
    """Check if pymatgen is installed (for crystallographic analysis)."""
    try:
        import pymatgen
        version = pkg_resources.get_distribution("pymatgen").version
        print(f"✅ pymatgen: Installed (version: {version}) - Advanced crystallographic analysis available")
        return True, version
    except ImportError:
        print("ℹ️ pymatgen: Not installed - Advanced crystallographic analysis will not be available")
        return False, None

def get_pyinstaller_status():
    """Check if PyInstaller is installed (for creating standalone executables)."""
    try:
        import PyInstaller
        version = pkg_resources.get_distribution("pyinstaller").version
        print(f"✅ PyInstaller: Installed (version: {version}) - Executable creation available")
        return True, version
    except ImportError:
        print("ℹ️ PyInstaller: Not installed - Standalone executable creation will not be available")
        return False, None

def suggest_install_command(missing_packages, stage_status):
    """Suggest pip install command for missing packages with stage-specific guidance."""
    if missing_packages:
        install_commands = {
            "numpy": "pip install numpy",
            "matplotlib": "pip install matplotlib",
            "scipy": "pip install scipy",
            "pandas": "pip install pandas",
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
        
        # Stage-specific recommendations
        print("\n📋 Stage-Specific Installation Recommendations:")
        
        if not stage_status['stage1']['available']:
            print("\n🚀 For Stage 1 (Enhanced Optimization):")
            stage1_packages = [p for p in stage_status['stage1']['missing'] if p != "tkinter"]
            if stage1_packages:
                print(f"   pip install {' '.join(stage1_packages)}")
        
        if not stage_status['stage2']['available'] or stage_status['stage2']['missing']:
            print("\n🧠 For Stage 2 (Probabilistic Bayesian):")
            stage2_packages = [p for p in stage_status['stage2']['missing'] if p != "tkinter"]
            if stage2_packages:
                print(f"   pip install {' '.join(stage2_packages)}")
            if 'emcee' in stage_status['stage2']['missing']:
                print("   ⚠️  emcee is critical for Stage 2 MCMC functionality")
        
        if not stage_status['stage3']['available'] or stage_status['stage3']['missing']:
            print("\n🌟 For Stage 3 (Advanced Multi-Objective):")
            stage3_packages = [p for p in stage_status['stage3']['missing'] if p != "tkinter"]
            if stage3_packages:
                print(f"   pip install {' '.join(stage3_packages)}")
            if 'scikit-learn' in stage_status['stage3']['missing']:
                print("   ⚠️  scikit-learn is critical for Stage 3 Gaussian Processes")
        
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
                if package == "numpy":
                    print("  For optimized installation: pip install numpy")
                elif package == "scipy":
                    print("  May require compiler tools. For binary installation: pip install scipy")
                elif package == "scikit-learn":
                    print("  Critical for Stages 2 & 3. Install with: pip install scikit-learn")
                elif package == "emcee":
                    print("  Critical for Stages 2 & 3 MCMC. Install with: pip install emcee")
                elif package == "fastdtw":
                    print("  May require Cython. If installation fails, try: pip install Cython first")
                elif package == "pymatgen":
                    print("  Large package with many dependencies. Install with: pip install pymatgen")
                elif package == "tensorflow":
                    print("  Consider installing tensorflow-cpu for lower resource usage: pip install tensorflow-cpu")
        
        # Comprehensive installation commands
        print("\n🔧 Comprehensive Installation Commands:")
        
        # Core functionality
        core_packages = ["numpy", "matplotlib", "scipy", "pandas", "seaborn", "PIL", "mplcursors", "reportlab", "openpyxl", "fastdtw"]
        core_missing = [p for p in missing_packages if p in core_packages]
        if core_missing:
            print("\nCore functionality (all stages with basic features):")
            print(f"pip install {' '.join(core_missing)}")
        
        # Advanced functionality
        advanced_packages = ["scikit-learn", "emcee"]
        advanced_missing = [p for p in missing_packages if p in advanced_packages]
        if advanced_missing:
            print("\nAdvanced functionality (Stages 2 & 3 full features):")
            print(f"pip install {' '.join(advanced_missing)}")
        
        # All packages
        regular_packages = [p for p in missing_packages if p != "tkinter"]
        if regular_packages:
            print("\nComplete installation (all features):")
            print(f"pip install {' '.join(regular_packages)}")
        
        print("\n🐍 Virtual Environment Setup (Recommended):")
        print("python -m venv clarityspectra_env")
        print("source clarityspectra_env/bin/activate  # On Windows: clarityspectra_env\\Scripts\\activate")
        if regular_packages:
            print(f"pip install {' '.join(regular_packages)}")

def main():
    """Main function to check all dependencies."""
    print("=" * 70)
    print(f"ClaritySpectra v{__version__} - Complete Trilogy Dependency Checker")
    print("=" * 70)
    print("\nChecking Python version...")
    python_ok = check_python_version()
    
    print("\nChecking core packages...")
    
    # Core packages required by the application with minimum recommended versions
    required_packages = [
        ("numpy", "1.16.0"), 
        ("matplotlib", "3.0.0"), 
        ("scipy", "1.2.0"), 
        ("pandas", "0.25.0"),
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
        elif min_version and version != "Unknown":
            try:
                if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                    outdated_packages.append((package, version, min_version))
            except:
                pass  # Skip version comparison issues
    
    # Check advanced optimization packages first
    print("\nChecking advanced optimization packages...")
    emcee_ok, emcee_version = get_emcee_status()
    if not emcee_ok:
        missing_packages.append("emcee")
    
    sklearn_ok, sklearn_version, sklearn_components, sklearn_missing = get_sklearn_status()
    if not sklearn_ok:
        missing_packages.append("scikit-learn")
    
    # Check stage-specific dependencies with the sklearn status
    stage_status = check_stage_specific_dependencies_with_status(emcee_ok, sklearn_ok)
    
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
        print("⚠️ Python version may be too old")
    
    # Stage availability summary
    print("\n📊 Stage Availability Summary:")
    stages = [
        ("🚀 Stage 1 (Enhanced)", stage_status['stage1']['available']),
        ("🧠 Stage 2 (Probabilistic)", stage_status['stage2']['available']),
        ("🌟 Stage 3 (Advanced)", stage_status['stage3']['available'])
    ]
    
    for stage_name, available in stages:
        status = "✅ AVAILABLE" if available else "❌ LIMITED/UNAVAILABLE"
        print(f"   {stage_name}: {status}")
    
    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} package(s): {', '.join(missing_packages)}")
    
    if outdated_packages:
        print(f"\n⚠️ {len(outdated_packages)} package(s) need updating:")
        for package, current, minimum in outdated_packages:
            print(f"  - {package}: current {current}, recommended {minimum}+")
    
    if not missing_packages and not outdated_packages:
        print("\n✅ All required packages are installed with sufficient versions!")
        print("🎉 Complete trilogy functionality available!")
    
    # Installation instructions
    if missing_packages or outdated_packages:
        suggest_install_command(missing_packages, stage_status)
        
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
    
    features = [
        ("Core Optimization", not missing_packages or all(p in ["tensorflow", "keras", "pymatgen", "pyinstaller"] for p in missing_packages)),
        ("PDF Export", reportlab_ok),
        ("MCMC Sampling (Stages 2 & 3)", emcee_ok),
        ("Gaussian Processes (Stage 3)", sklearn_ok),
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
        ("RRUFF_Export_with_Hey_Classification.csv", "RRUFF mineral classification data"),
        ("mineral_modes.pkl", "Mineral mode database")
    ]
    
    for file, description in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # Size in MB
            print(f"✅ {file}: Found ({size:.1f} MB) - {description}")
        else:
            print(f"ℹ️ {file}: Not found - {description}")
            if file == "RRUFF_Export_with_Hey_Classification.csv":
                print("  This file may need to be downloaded from RRUFF database (https://rruff.info/)")
    
    print("\n" + "=" * 70)
    
    # Final recommendation
    if missing_packages or outdated_packages:
        print("\n🎯 FINAL RECOMMENDATION:")
        print("Set up a virtual environment for optimal ClaritySpectra experience:")
        print("\n1. Create and activate virtual environment:")
        print("   python -m venv clarityspectra_env")
        print("   source clarityspectra_env/bin/activate  # Windows: clarityspectra_env\\Scripts\\activate")
        print("\n2. Install requirements:")
        print("   pip install -r requirements.txt")
        print("\n3. Verify installation:")
        print("   python check_dependencies.py")
        
        if "tkinter" in missing_packages:
            print("\n4. Install tkinter according to your operating system (see instructions above)")
    else:
        print("\n🎉 CONGRATULATIONS!")
        print("Your environment is fully configured for ClaritySpectra complete trilogy!")
        print("All optimization stages (1, 2, and 3) are available with full functionality.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
