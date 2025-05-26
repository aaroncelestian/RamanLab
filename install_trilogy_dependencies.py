#!/usr/bin/env python3
"""
ClaritySpectra Trilogy Dependencies Installer
=============================================

This script helps install the complete dependencies for ClaritySpectra's
crystal orientation optimization trilogy (Stages 1, 2, and 3).

Usage:
    python install_trilogy_dependencies.py [option]

Options:
    --core      Install core dependencies only (Stage 1 + basic functionality)
    --advanced  Install advanced dependencies (Stages 2 & 3 full functionality)
    --complete  Install all dependencies including optional features
    --check     Check current installation status
"""

import subprocess
import sys
import os
import argparse

def run_pip_install(packages, description=""):
    """Run pip install for a list of packages."""
    if not packages:
        return True
    
    print(f"\n{'='*60}")
    print(f"Installing {description}")
    print(f"{'='*60}")
    print(f"Packages: {', '.join(packages)}")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Installation successful!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed with exit code {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def install_core_dependencies():
    """Install core dependencies for basic functionality and Stage 1."""
    packages = [
        "numpy>=1.16.0",
        "matplotlib>=3.0.0", 
        "scipy>=1.2.0",
        "pandas>=0.25.0",
        "seaborn>=0.11.0",
        "pillow>=8.0.0",
        "mplcursors>=0.5.0",
        "reportlab>=3.5.0",
        "openpyxl>=3.0.0",
        "fastdtw>=0.3.4"
    ]
    
    return run_pip_install(packages, "Core Dependencies (Stage 1 + Basic Functionality)")

def install_advanced_dependencies():
    """Install advanced dependencies for Stages 2 & 3."""
    packages = [
        "scikit-learn>=0.21.0",
        "emcee>=3.0.0"
    ]
    
    return run_pip_install(packages, "Advanced Dependencies (Stages 2 & 3)")

def install_optional_dependencies():
    """Install optional dependencies for enhanced features."""
    packages = [
        "tensorflow>=2.12.0",
        "keras>=2.12.0",
        "pymatgen>=2022.0.0",
        "pyinstaller>=5.0.0"
    ]
    
    return run_pip_install(packages, "Optional Dependencies (Enhanced Features)")

def check_installation():
    """Check current installation status."""
    print("\n" + "="*60)
    print("CHECKING CURRENT INSTALLATION STATUS")
    print("="*60)
    
    try:
        # Import and run the dependency checker
        import check_dependencies
        check_dependencies.main()
    except ImportError:
        print("❌ check_dependencies.py not found. Running basic check...")
        basic_check()

def basic_check():
    """Basic dependency check if check_dependencies.py is not available."""
    packages_to_check = [
        ("numpy", "Core"),
        ("matplotlib", "Core"),
        ("scipy", "Core"),
        ("pandas", "Core"),
        ("tkinter", "Core"),
        ("seaborn", "Core"),
        ("PIL", "Core"),  # Pillow is imported as PIL
        ("mplcursors", "Core"),
        ("reportlab", "Core"),
        ("openpyxl", "Core"),
        ("fastdtw", "Core"),
        ("sklearn", "Advanced (Stage 2 & 3)"),
        ("emcee", "Advanced (Stage 2 & 3)"),
        ("tensorflow", "Optional"),
        ("keras", "Optional"),
        ("pymatgen", "Optional"),
        ("PyInstaller", "Optional")
    ]
    
    print("\nPackage Status:")
    print("-" * 50)
    
    for package, category in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package:<15} - {category}")
        except ImportError:
            print(f"❌ {package:<15} - {category}")

def check_tkinter():
    """Check tkinter availability and provide installation instructions."""
    print("\n" + "="*60)
    print("TKINTER CHECK")
    print("="*60)
    
    try:
        import tkinter
        print("✅ tkinter is available")
        return True
    except ImportError:
        print("❌ tkinter is not available")
        print("\ntkinter Installation Instructions:")
        print("- Windows: Usually included with Python installer")
        print("- macOS: 'brew install python-tk' or reinstall Python with tkinter")
        print("- Debian/Ubuntu: 'sudo apt-get install python3-tk'")
        print("- Fedora/RHEL: 'sudo dnf install python3-tkinter'")
        return False

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="Install ClaritySpectra Trilogy Dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_trilogy_dependencies.py --core      # Core dependencies only
  python install_trilogy_dependencies.py --advanced  # Core + advanced
  python install_trilogy_dependencies.py --complete  # Everything
  python install_trilogy_dependencies.py --check     # Check status
        """
    )
    
    parser.add_argument("--core", action="store_true", 
                       help="Install core dependencies (Stage 1 + basic functionality)")
    parser.add_argument("--advanced", action="store_true",
                       help="Install advanced dependencies (Stages 2 & 3)")
    parser.add_argument("--complete", action="store_true",
                       help="Install all dependencies including optional features")
    parser.add_argument("--check", action="store_true",
                       help="Check current installation status")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ClaritySpectra Trilogy Dependencies Installer")
    print("="*70)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment (consider using one)")
    
    success = True
    
    if args.check:
        check_installation()
        check_tkinter()
        return
    
    if not any([args.core, args.advanced, args.complete]):
        print("\n🤔 No installation option specified. What would you like to install?")
        print("\nOptions:")
        print("1. Core dependencies only (Stage 1 + basic functionality)")
        print("2. Advanced dependencies (Stages 2 & 3 full functionality)")  
        print("3. Complete installation (all features)")
        print("4. Check current status")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                if choice == "1":
                    args.core = True
                    break
                elif choice == "2":
                    args.advanced = True
                    break
                elif choice == "3":
                    args.complete = True
                    break
                elif choice == "4":
                    check_installation()
                    check_tkinter()
                    return
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\n\nInstallation cancelled.")
                return
    
    # Install based on selected options
    if args.core or args.advanced or args.complete:
        print(f"\n🚀 Starting installation...")
        
        # Always install core dependencies first
        if not install_core_dependencies():
            success = False
        
        # Install advanced dependencies if requested
        if (args.advanced or args.complete) and success:
            if not install_advanced_dependencies():
                success = False
        
        # Install optional dependencies if complete installation requested
        if args.complete and success:
            if not install_optional_dependencies():
                print("⚠️  Some optional dependencies failed to install, but core functionality should work")
    
    # Check tkinter
    check_tkinter()
    
    # Final status check
    print("\n" + "="*70)
    print("INSTALLATION SUMMARY")
    print("="*70)
    
    if success:
        print("✅ Installation completed successfully!")
        
        if args.core:
            print("🚀 Stage 1 (Enhanced) optimization is now available")
        if args.advanced or args.complete:
            print("🧠 Stage 2 (Probabilistic) optimization is now available")
            print("🌟 Stage 3 (Advanced) optimization is now available")
        if args.complete:
            print("🎉 Complete trilogy with all optional features installed")
            
        print("\nNext steps:")
        print("1. Verify installation: python install_trilogy_dependencies.py --check")
        print("2. Run ClaritySpectra: python raman_polarization_analyzer.py")
        
    else:
        print("❌ Installation encountered errors")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure pip is up to date: python -m pip install --upgrade pip")
        print("3. Try installing packages individually")
        print("4. Check for system-specific requirements")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main() 