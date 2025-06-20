# PySide6-Only Migration Guide

## Overview

RamanLab has been migrated to use **PySide6 exclusively** as the Qt binding for Python. This decision was made to standardize on the official Qt for Python binding, which provides better licensing terms and is actively maintained by The Qt Company.

## Key Changes Made

### 1. Qt Binding Standardization
- **Removed**: All PyQt5 and PyQt6 imports and fallbacks
- **Adopted**: PySide6 as the single Qt binding
- **Rationale**: PySide6 is the official Qt for Python binding with LGPL licensing

### 2. Updated Import Statements
```python
# OLD - Multiple Qt binding support
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
except ImportError:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *

# NEW - PySide6 only
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
```

### 3. Signal/Slot Updates
- **Changed**: `pyqtSignal` → `Signal` (PySide6 naming convention)
- **Example**: `progress_updated = Signal(int)` instead of `progress_updated = pyqtSignal(int)`

### 4. Matplotlib Backend Updates
- **Updated**: Matplotlib backend to use `QtAgg` instead of `Qt5Agg`
- **Import**: `matplotlib.backends.backend_qtagg` instead of `backend_qt5agg`

## Files Updated

### Core Modules
- `polarization_ui/visualization_3d.py` - 3D visualization widget
- `polarization_ui/orientation_optimizer_widget.py` - Orientation optimization
- `raman_polarization_analyzer_qt6.py` - Main analyzer (already using PySide6)

### Demo and Test Files
- `demo_3d_visualization.py` - Updated imports and error messages
- `launch_orientation_optimizer.py` - Updated dependency instructions

### Documentation
- Updated installation instructions to reference PySide6
- Updated error messages to suggest PySide6 installation

## Installation Requirements

### Required Package
```bash
pip install PySide6
```

### Full Dependencies
```bash
pip install PySide6 matplotlib numpy scipy
pip install scikit-learn emcee  # optional for advanced features
```

## Benefits of PySide6-Only Approach

### 1. **Licensing Clarity**
- LGPL licensing allows commercial use without GPL restrictions
- No licensing complexity from mixing different Qt bindings

### 2. **Maintenance Simplification**
- Single codebase path reduces complexity
- No conditional imports or version-specific code
- Easier debugging and testing

### 3. **Official Support**
- PySide6 is officially maintained by The Qt Company
- Better long-term support and compatibility
- Regular updates and bug fixes

### 4. **Performance**
- Native Qt6 performance improvements
- Better memory management
- Improved graphics rendering

## Migration Verification

### Testing Imports
```python
# Test core functionality
from PySide6.QtWidgets import QApplication
from polarization_ui.visualization_3d import Advanced3DVisualizationWidget
from polarization_ui.orientation_optimizer_widget import OrientationOptimizerWidget

print("✓ All PySide6 imports successful")
```

### Running Applications
```bash
# Test 3D visualization demo
python demo_3d_visualization.py

# Test orientation optimizer
python launch_orientation_optimizer.py

# Test main analyzer
python raman_polarization_analyzer_qt6.py
```

## Compatibility Notes

### Qt Constants
- PySide6 uses different constant naming (e.g., `Qt.Orientation.Horizontal`)
- All slider orientations and alignment constants updated

### Matplotlib Integration
- Uses `backend_qtagg` for Qt6 integration
- Automatic backend detection and configuration

### Thread Signals
- `Signal` instead of `pyqtSignal`
- Same functionality, different naming convention

## Future Development

### New Code Guidelines
1. **Always use PySide6** - Never import PyQt5 or PyQt6
2. **Use Signal** - For thread communication
3. **Use QtAgg backend** - For matplotlib integration
4. **Test imports** - Verify PySide6 compatibility

### Memory Management
- Created memory rule: "Always use PySide6 for Qt applications, never PyQt5 or PyQt6"
- This rule is enforced in the AI assistant's memory system

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure PySide6 is installed (`pip install PySide6`)
2. **Backend Issues**: Check matplotlib backend configuration
3. **Signal Errors**: Use `Signal` instead of `pyqtSignal`

### Debug Commands
```bash
# Check PySide6 installation
python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"

# Check matplotlib backend
python -c "from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg; print('Matplotlib backend OK')"
```

## Summary

The migration to PySide6-only provides:
- ✅ Simplified, maintainable codebase
- ✅ Clear licensing for commercial use
- ✅ Official Qt Company support
- ✅ Modern Qt6 features and performance
- ✅ Consistent development experience

This migration ensures RamanLab remains modern, legally compliant, and technically robust for future development. 