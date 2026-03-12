#!/usr/bin/env python3
"""
Automatic h5py Installer for RamanLab
======================================
This script automatically installs h5py into the current Python environment.
"""

import sys
import os
import subprocess
import time

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)

def is_conda_environment():
    """Check if running in a conda environment."""
    return 'conda' in sys.executable.lower() or 'anaconda' in sys.executable.lower()

def check_h5py_already_installed():
    """Check if h5py is already working."""
    try:
        import h5py
        print(f"✅ h5py {h5py.__version__} is already installed and working!")
        return True
    except ImportError:
        return False

def install_with_conda():
    """Install h5py using conda."""
    print_header("INSTALLING H5PY WITH CONDA")
    print("Running: conda install -c conda-forge h5py -y")
    print()
    
    try:
        result = subprocess.run(
            ["conda", "install", "-c", "conda-forge", "h5py", "-y"],
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ conda command not found")
        print("   Falling back to pip installation...")
        return False
    except Exception as e:
        print(f"❌ Error running conda: {e}")
        print("   Falling back to pip installation...")
        return False

def install_with_pip():
    """Install h5py using pip."""
    print_header("INSTALLING H5PY WITH PIP")
    
    # Step 1: Upgrade pip
    print("Step 1/3: Upgrading pip, setuptools, and wheel...")
    print(f"Running: {sys.executable} -m pip install --upgrade pip setuptools wheel")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print("⚠️  Warning: pip upgrade had issues, continuing anyway...")
        else:
            print("✅ pip upgraded successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Error upgrading pip: {e}")
        print("   Continuing with h5py installation...")
    
    print()
    time.sleep(1)
    
    # Step 2: Uninstall existing h5py (if any)
    print("Step 2/3: Removing any existing h5py installation...")
    print(f"Running: {sys.executable} -m pip uninstall -y h5py")
    print()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "h5py"],
            capture_output=True,
            text=True
        )
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️  Note: {e}")
    
    print()
    time.sleep(1)
    
    # Step 3: Install h5py
    print("Step 3/3: Installing h5py...")
    print(f"Running: {sys.executable} -m pip install --no-cache-dir h5py")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "h5py"],
            capture_output=False,
            text=True
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error installing h5py: {e}")
        return False

def verify_installation():
    """Verify that h5py was installed successfully."""
    print_header("VERIFYING INSTALLATION")
    print("Testing h5py import...")
    print()
    
    try:
        import h5py
        print("✅ SUCCESS! h5py imported successfully")
        print(f"   Version: {h5py.__version__}")
        print(f"   Location: {h5py.__file__}")
        
        try:
            print(f"   HDF5 library: {h5py.version.hdf5_version}")
        except:
            pass
        
        return True
        
    except ImportError as e:
        print("❌ FAILED: h5py still cannot be imported")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print("❌ FAILED: Unexpected error")
        print(f"   Error: {e}")
        return False

def main():
    """Main installation routine."""
    print("="*70)
    print("RamanLab h5py Automatic Installer")
    print("="*70)
    print()
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print()
    
    # Check if already installed
    if check_h5py_already_installed():
        print()
        print("No installation needed!")
        return 0
    
    print("❌ h5py is not currently available")
    print()
    
    # Determine installation method
    is_conda = is_conda_environment()
    
    if is_conda:
        print("📦 Conda environment detected")
        print("   Will try conda installation first, then pip if needed")
    else:
        print("📦 Standard Python environment detected")
        print("   Will use pip installation")
    
    print()
    input("Press ENTER to begin installation (or Ctrl+C to cancel)...")
    
    # Try installation
    success = False
    
    if is_conda:
        print()
        print("Attempting conda installation...")
        success = install_with_conda()
        
        if not success:
            print()
            print("Conda installation failed or unavailable, trying pip...")
            success = install_with_pip()
    else:
        success = install_with_pip()
    
    print()
    
    if not success:
        print_header("INSTALLATION FAILED")
        print("❌ Could not install h5py automatically")
        print()
        print("Please try manual installation:")
        print()
        if is_conda:
            print("  conda install -c conda-forge h5py")
        else:
            print(f"  {sys.executable} -m pip install h5py")
        print()
        
        if sys.platform == 'win32':
            print("⚠️  WINDOWS USERS:")
            print("   If you see DLL errors, you may need:")
            print("   1. Microsoft Visual C++ Redistributable (x64)")
            print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("   2. Restart your computer after installing")
            print()
        
        return 1
    
    # Verify installation
    print()
    time.sleep(1)
    
    if verify_installation():
        print_header("INSTALLATION COMPLETE")
        print("🎉 h5py has been successfully installed!")
        print()
        print("You can now:")
        print("  • Import HDF5/MAPX files in RamanLab")
        print("  • Use all h5py-dependent features")
        print()
        print("Next steps:")
        print("  1. Restart RamanLab if it's currently running")
        print("  2. Try importing an HDF5 file again")
        print()
        return 0
    else:
        print_header("VERIFICATION FAILED")
        print("⚠️  h5py was installed but cannot be imported")
        print()
        print("This may indicate:")
        print("  • Missing system libraries (Windows: VC++ Redistributable)")
        print("  • Python environment issues")
        print("  • Corrupted installation")
        print()
        print("Recommended actions:")
        print("  1. Restart your computer")
        print("  2. Run this script again")
        print("  3. Check WINDOWS_SETUP.md for detailed troubleshooting")
        print()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
