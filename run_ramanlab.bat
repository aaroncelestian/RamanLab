@echo off
REM RamanLab Launcher Script for Windows
REM This script verifies the Python environment and h5py availability before launching

echo ========================================
echo RamanLab Windows Launcher
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not found in PATH
    echo Please install Python 3.8+ from python.org or Microsoft Store
    echo.
    pause
    exit /b 1
)

echo Python found:
python -c "import sys; print(f'  {sys.executable}')"
echo.

REM Check for h5py specifically (common Windows issue)
echo Checking h5py availability...
python -c "import h5py; print(f'  h5py {h5py.__version__} - OK')" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: h5py is not available in this Python environment
    echo This is required for HDF5/MAPX file import functionality.
    echo.
    echo To install h5py, run:
    echo   python -m pip install --upgrade pip
    echo   python -m pip install --no-cache-dir h5py
    echo.
    echo Or with conda:
    echo   conda install -c conda-forge h5py
    echo.
    echo Press any key to continue anyway, or Ctrl+C to exit and install h5py...
    pause
) else (
    echo   h5py is available
)

echo.
echo Starting RamanLab...
echo.

REM Launch RamanLab with debug launcher for better error capture
python launch_ramanlab_debug.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo RamanLab exited with an error.
    echo.
    echo See error details above.
    echo.
    pause
    exit /b 1
)
