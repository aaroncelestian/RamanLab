# RamanLab Startup Performance Guide

## Problem: Slow Desktop Launch vs Fast Command Line Launch

When launching RamanLab from the desktop icon, the application takes much longer to start compared to running `python main_qt6.py` directly from the command line.

## Root Cause Analysis

The desktop icon launches through `launch_ramanlab.py`, which performs comprehensive dependency checking before starting the application. This includes:

1. **Importing multiple packages** (PySide6, PyQt6, scipy, numpy, matplotlib, etc.)
2. **Running subprocess calls** to check package versions
3. **Analyzing component-specific dependencies**
4. **Generating detailed compatibility reports**

When you run `python main_qt6.py` directly, it bypasses all these checks and launches immediately.

## Solutions Implemented

### 1. Fast Launcher (Recommended for Desktop Icons)
- **File**: `launch_ramanlab_fast.py`
- **Purpose**: Skips all dependency checks for maximum speed
- **Use case**: Desktop shortcuts, after dependencies have been verified once

### 2. Smart Caching (Enhanced Original Launcher)
- **File**: `launch_ramanlab.py` (updated)
- **Features**:
  - Caches dependency check results for 24 hours
  - Adds `--fast` flag to skip checks
  - Adds `--force-check` flag to force recheck
- **Use case**: Command line usage with smart dependency management

### 3. Updated Desktop Icon
- **Automatically uses** `launch_ramanlab_fast.py` for desktop shortcuts
- **Falls back** to `launch_ramanlab.py` if fast launcher is not available

## Usage Options

### Desktop Launch (Fastest)
The desktop icon now automatically uses the fast launcher. Just click the icon!

### Command Line Options

```bash
# Direct launch (fastest, no dependency checks)
python main_qt6.py

# Fast launcher (no dependency checks)
python launch_ramanlab_fast.py

# Smart launcher with caching (default)
python launch_ramanlab.py

# Force fast mode (skip checks)
python launch_ramanlab.py --fast

# Force dependency recheck
python launch_ramanlab.py --force-check
```

## Performance Comparison

| Launch Method | Typical Startup Time | Dependency Checks |
|---------------|---------------------|-------------------|
| `python main_qt6.py` | ~2-3 seconds | None |
| `launch_ramanlab_fast.py` | ~2-3 seconds | None |
| `launch_ramanlab.py --fast` | ~2-3 seconds | None |
| `launch_ramanlab.py` (cached) | ~3-4 seconds | Skipped |
| `launch_ramanlab.py` (first run) | ~8-15 seconds | Full check |

## Recommendations

1. **For Daily Use**: Use the desktop icon (now uses fast launcher)
2. **For Development**: Use `python main_qt6.py` for immediate testing
3. **After Installing New Packages**: Run `python launch_ramanlab.py --force-check` once
4. **For Troubleshooting**: Use `python launch_ramanlab.py` for diagnostic information

## When to Use Each Launcher

### Use Fast Launcher When:
- ✅ You've already verified dependencies work
- ✅ You want maximum startup speed
- ✅ You're using the application regularly
- ✅ You're launching from desktop/applications menu

### Use Regular Launcher When:
- 🔍 First time setup
- 🔍 After installing new Python packages
- 🔍 Troubleshooting startup issues
- 🔍 You want to see current package versions

## Technical Details

The dependency checker (`check_dependencies.py`) performs these operations:
- Imports ~15+ scientific Python packages
- Runs `importlib.metadata.version()` for each package
- Checks Qt6 framework availability (PySide6/PyQt6)
- Analyzes component-specific dependencies
- Generates compatibility reports

While valuable for diagnostics, these checks add 5-12 seconds to startup time depending on your system's performance and the number of installed packages.

## Troubleshooting

If the fast launcher fails to start:
1. Run `python launch_ramanlab.py` to check for dependency issues
2. Install missing packages with `pip install -r requirements_qt6.txt`
3. Use `python launch_ramanlab.py --force-check` to verify installation

The desktop icon will automatically fall back to the regular launcher if the fast launcher is not available. 