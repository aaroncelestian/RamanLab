@echo off
REM RamanLab UTF-8 Encoding Fix for Windows
REM =============================================
REM 
REM This batch file will fix UTF-8 encoding issues when opening
REM map files for 2D map analysis on Windows.
REM
REM Simply double-click this file to run the fix.

echo.
echo RamanLab UTF-8 Encoding Fix
echo =================================
echo.
echo Fixing UTF-8 encoding issues for 2D map analysis...
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

REM Run the UTF-8 encoding fix script
python utf8_encoding_fix.py

REM Check if fix was successful
if %errorlevel% equ 0 (
    echo.
    echo ===================================
    echo UTF-8 encoding fix completed!
    echo ===================================
    echo.
    echo The 2D map analysis should now work correctly on Windows.
    echo You can now open map files without UTF-8 encoding errors.
    echo.
) else (
    echo.
    echo ERROR: UTF-8 encoding fix failed.
    echo Please check the error messages above.
    echo.
)

echo Press any key to continue...
pause >nul 