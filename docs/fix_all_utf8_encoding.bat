@echo off
REM RamanLab Comprehensive UTF-8 Encoding Fix for Windows
REM ==========================================================
REM 
REM This batch file will fix UTF-8 encoding issues across ALL modules:
REM - 2D Map Analysis
REM - Batch Peak Fitting  
REM - Peak Fitting
REM - Polarization Analyzer
REM
REM Simply double-click this file to run the comprehensive fix.

echo.
echo RamanLab Comprehensive UTF-8 Encoding Fix
echo ===============================================
echo.
echo This will fix UTF-8 encoding issues in ALL modules:
echo - 2D Map Analysis
echo - Batch Peak Fitting
echo - Peak Fitting
echo - Polarization Analyzer
echo.
echo Fixing UTF-8 encoding issues for cross-platform compatibility...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python and try again.
    echo.
    pause
    exit /b 1
)

REM Run the comprehensive UTF-8 encoding fix script
python comprehensive_utf8_fix.py

REM Check if fix was successful
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Comprehensive UTF-8 encoding fix completed!
    echo ========================================
    echo.
    echo All modules should now work correctly on Windows:
    echo ✓ 2D Map Analysis - Fixed file reading for map data
    echo ✓ Batch Peak Fitting - Fixed spectrum file loading
    echo ✓ Peak Fitting - Fixed individual spectrum files
    echo ✓ Polarization Analyzer - Fixed spectrum imports
    echo.
    echo You can now use all features without UTF-8 encoding errors.
    echo.
) else (
    echo.
    echo ERROR: Comprehensive UTF-8 encoding fix failed.
    echo Please check the error messages above.
    echo.
)

echo Press any key to continue...
pause >nul 