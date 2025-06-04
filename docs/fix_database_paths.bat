@echo off
REM RamanLab Database Path Fix for Windows
REM ============================================
REM 
REM This batch file will fix database path issues when copying
REM RamanLab from Mac/Linux to Windows.
REM
REM Simply double-click this file to run the fix.

echo.
echo RamanLab Database Path Fix
echo ================================
echo.
echo Fixing database path issues for Windows compatibility...
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

REM Run the fix script
python database_path_fix.py

REM Check if fix was successful
if %errorlevel% equ 0 (
    echo.
    echo ===================================
    echo Fix completed successfully!
    echo ===================================
    echo.
    echo You can now run the application with:
    echo   python main.py
    echo.
) else (
    echo.
    echo ERROR: Fix script failed.
    echo Please check the error messages above.
    echo.
)

echo Press any key to continue...
pause >nul 