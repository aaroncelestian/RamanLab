#!/usr/bin/env python3
"""
h5py Diagnostic Script for RamanLab
====================================
This script checks h5py availability and provides detailed diagnostic information.
"""

import sys
import os
import subprocess

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)

def check_python_info():
    """Display Python interpreter information."""
    print_header("PYTHON ENVIRONMENT")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python prefix: {sys.prefix}")
    
    # Check if this is conda/miniconda
    if 'conda' in sys.executable.lower() or 'anaconda' in sys.executable.lower():
        print("\n⚠️  CONDA/MINICONDA DETECTED")
        print("You are using a conda-based Python environment.")
        
        # Try to get conda env name
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        print(f"Conda environment: {conda_env}")

def check_h5py_import():
    """Try to import h5py and report results."""
    print_header("H5PY IMPORT TEST")
    
    try:
        import h5py
        print("✅ SUCCESS: h5py imported successfully!")
        print(f"   Version: {h5py.__version__}")
        print(f"   Location: {h5py.__file__}")
        
        # Check HDF5 library version
        try:
            print(f"   HDF5 library version: {h5py.version.hdf5_version}")
        except:
            pass
        
        return True
        
    except ImportError as e:
        print("❌ FAILED: h5py could not be imported")
        print(f"   Error: {e}")
        return False
        
    except Exception as e:
        print("❌ FAILED: Unexpected error importing h5py")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error: {e}")
        return False

def check_pip_list():
    """Check if h5py is in pip list."""
    print_header("PIP PACKAGE CHECK")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "h5py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ h5py found in pip packages:")
            print(result.stdout)
            return True
        else:
            print("❌ h5py NOT found in pip packages")
            print("\nTrying 'pip list' to search for h5py...")
            
            # Try pip list
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if "h5py" in result.stdout.lower():
                print("⚠️  h5py appears in pip list but 'pip show' failed")
                for line in result.stdout.split('\n'):
                    if 'h5py' in line.lower():
                        print(f"   {line}")
            else:
                print("❌ h5py not found in pip list at all")
            
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  pip command timed out")
        return False
    except Exception as e:
        print(f"⚠️  Error running pip: {e}")
        return False

def check_conda_list():
    """Check if h5py is in conda list (if conda is available)."""
    if 'conda' not in sys.executable.lower() and 'anaconda' not in sys.executable.lower():
        return False
    
    print_header("CONDA PACKAGE CHECK")
    
    try:
        result = subprocess.run(
            ["conda", "list", "h5py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "h5py" in result.stdout:
            print("✅ h5py found in conda packages:")
            print(result.stdout)
            return True
        else:
            print("❌ h5py NOT found in conda packages")
            return False
            
    except FileNotFoundError:
        print("⚠️  conda command not found")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  conda command timed out")
        return False
    except Exception as e:
        print(f"⚠️  Error running conda: {e}")
        return False

def provide_recommendations(h5py_works, in_pip, in_conda, is_conda_env):
    """Provide installation recommendations based on diagnostic results."""
    print_header("RECOMMENDATIONS")
    
    if h5py_works:
        print("✅ h5py is working correctly!")
        print("   No action needed.")
        return
    
    print("❌ h5py is NOT working in this Python environment")
    print()
    
    if is_conda_env:
        print("📋 RECOMMENDED SOLUTION (Conda Environment):")
        print()
        print("1. Install h5py using conda (recommended for conda environments):")
        print(f"   conda install -c conda-forge h5py")
        print()
        print("2. Alternative - use pip:")
        print(f"   {sys.executable} -m pip install --no-cache-dir h5py")
        print()
        
    else:
        print("📋 RECOMMENDED SOLUTION (Standard Python):")
        print()
        print("1. Upgrade pip first:")
        print(f"   {sys.executable} -m pip install --upgrade pip setuptools wheel")
        print()
        print("2. Install h5py:")
        print(f"   {sys.executable} -m pip install --no-cache-dir h5py")
        print()
    
    print("3. After installation, run this script again to verify:")
    print(f"   {sys.executable} check_h5py.py")
    print()
    
    # Windows-specific advice
    if sys.platform == 'win32':
        print("⚠️  WINDOWS USERS:")
        print("   If you get DLL errors after installing h5py:")
        print("   1. Install Microsoft Visual C++ Redistributable (x64)")
        print("      Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   2. Restart your computer")
        print("   3. Reinstall h5py")
        print()

def main():
    """Run all diagnostic checks."""
    print("="*70)
    print("RamanLab h5py Diagnostic Tool")
    print("="*70)
    
    # Check Python environment
    check_python_info()
    
    # Determine if conda environment
    is_conda_env = 'conda' in sys.executable.lower() or 'anaconda' in sys.executable.lower()
    
    # Try to import h5py
    h5py_works = check_h5py_import()
    
    # Check package managers
    in_pip = check_pip_list()
    in_conda = check_conda_list() if is_conda_env else False
    
    # Provide recommendations
    provide_recommendations(h5py_works, in_pip, in_conda, is_conda_env)
    
    print_header("SUMMARY")
    print(f"Python: {sys.executable}")
    print(f"h5py import: {'✅ WORKS' if h5py_works else '❌ FAILED'}")
    print(f"h5py in pip: {'✅ YES' if in_pip else '❌ NO'}")
    if is_conda_env:
        print(f"h5py in conda: {'✅ YES' if in_conda else '❌ NO'}")
    print()
    
    if not h5py_works:
        print("⚠️  ACTION REQUIRED: Follow the recommendations above to install h5py")
        print()
        print("💡 TIP: You can run 'install_h5py.py' to automatically install h5py")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
