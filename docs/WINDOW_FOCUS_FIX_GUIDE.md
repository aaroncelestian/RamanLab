# Window Focus Fix Guide for RamanLab

## Problem Description

Users experience a window focus issue where the main RamanLab window goes to the background after file save operations. This is a common Qt application issue that occurs when file dialogs (`QFileDialog.getSaveFileName`, `QFileDialog.getOpenFileName`, etc.) are used.

## Root Cause

The issue occurs because:
1. **File dialogs are modal**: They temporarily take focus away from the main window
2. **Platform differences**: Different operating systems handle window focus differently after modal dialogs
3. **Qt timing issues**: The window focus restoration happens before the dialog is fully closed
4. **Missing focus restoration**: Qt doesn't automatically restore focus to the parent window

## Solution Overview

We've implemented a comprehensive `WindowFocusManager` in `core/window_focus_manager.py` that provides:

- **Platform-specific focus restoration** (macOS, Windows, Linux)
- **Delayed focus restoration** to avoid Qt timing issues
- **Multiple restoration attempts** for stubborn cases
- **Fallback mechanisms** if the focus manager isn't available
- **Easy-to-use utilities** for common scenarios

## Quick Fix for Existing Code

### Method 1: Simple Focus Restoration (Recommended)

Add this code after any file dialog operation:

```python
# After QFileDialog.getSaveFileName or similar
if file_path:
    try:
        # ... your save logic ...
        
        # Restore window focus after file dialog
        try:
            from core.window_focus_manager import restore_window_focus_after_dialog
            restore_window_focus_after_dialog(self)
        except ImportError:
            # Fallback if focus manager not available
            self.raise_()
            self.activateWindow()
            
    except Exception as e:
        # ... error handling ...
```

### Method 2: Using the Decorator (For New Code)

For new methods, you can use the decorator:

```python
from core.window_focus_manager import restore_focus_after

@restore_focus_after(150)  # 150ms delay
def save_spectrum(self):
    """Save the current spectrum."""
    file_path, _ = QFileDialog.getSaveFileName(self, "Save", "", "*.txt")
    if file_path:
        # ... save logic ...
        # Focus will be automatically restored
```

### Method 3: Using Context Manager (Advanced)

For complex dialog sequences:

```python
from core.window_focus_manager import create_focus_restoring_file_dialog

def save_multiple_files(self):
    """Save multiple files with automatic focus restoration."""
    with create_focus_restoring_file_dialog():
        file_path1, _ = QFileDialog.getSaveFileName(self, "Save File 1", "", "*.txt")
        if file_path1:
            # ... save file 1 ...
            
        file_path2, _ = QFileDialog.getSaveFileName(self, "Save File 2", "", "*.csv")
        if file_path2:
            # ... save file 2 ...
            
        # Focus will be automatically restored when exiting the context
```

## Files That Need Updating

Here are the key files that contain file dialog operations and should be updated:

### Core Application Files
- ✅ `raman_analysis_app_qt6.py` - **FIXED**
- ✅ `raman_cluster_analysis_qt6.py` - **FIXED**
- `spectrum_viewer_qt6.py`
- `multi_spectrum_manager_qt6.py`
- `peak_fitting_qt6.py`
- `raman_polarization_analyzer_qt6.py`

### Map Analysis Files
- `map_analysis_2d/ui/main_window.py`
- `map_analysis_2d/ui/dialogs/quantitative_calibration_dialog.py`

### Database Files
- `database_browser_qt6.py`
- `database_manager_gui.py`

### Advanced Analysis Files
- `advanced_analysis/density_gui_launcher.py`
- `advanced_analysis/raman_density_analysis.py`

### Utility Files
- `core/settings_dialog.py`
- `core/update_checker.py`

## Implementation Steps

### Step 1: Import the Focus Manager

At the top of your file, add the import (inside the function to avoid import errors):

```python
# Inside the function where you use file dialogs
try:
    from core.window_focus_manager import restore_window_focus_after_dialog
    restore_window_focus_after_dialog(self)
except ImportError:
    # Fallback if focus manager not available
    self.raise_()
    self.activateWindow()
```

### Step 2: Add Focus Restoration After File Operations

Add the focus restoration code after successful file operations:

```python
def save_something(self):
    """Save something to file."""
    file_path, _ = QFileDialog.getSaveFileName(self, "Save", "", "*.txt")
    
    if file_path:
        try:
            # ... your save logic here ...
            
            # Show success message
            QMessageBox.information(self, "Success", f"Saved to {file_path}")
            
            # IMPORTANT: Add focus restoration here
            try:
                from core.window_focus_manager import restore_window_focus_after_dialog
                restore_window_focus_after_dialog(self)
            except ImportError:
                # Fallback if focus manager not available
                self.raise_()
                self.activateWindow()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
```

### Step 3: Test the Fix

1. **Open RamanLab**
2. **Load some data**
3. **Try to save something** (spectrum, plot, analysis results, etc.)
4. **Verify the main window stays in focus** after the save dialog closes

## Platform-Specific Considerations

### macOS
- Requires multiple focus restoration attempts
- Uses `raise_()`, `activateWindow()`, and `setFocus()` sequence
- May need secondary attempts after 50ms delay

### Windows
- Handles window state changes differently
- Uses `setWindowState()` to ensure window isn't minimized
- Requires `QApplication.processEvents()` for proper timing

### Linux
- Similar to Windows but with additional attempts
- Different window managers may behave differently
- Uses secondary focus attempt after 25ms delay

## Testing

Create a simple test script to verify the fix:

```python
#!/usr/bin/env python3
"""Test script for window focus fix."""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from core.window_focus_manager import restore_window_focus_after_dialog

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window Focus Test")
        self.setGeometry(100, 100, 400, 300)
        
        # Create UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Test button
        test_btn = QPushButton("Test Save Dialog")
        test_btn.clicked.connect(self.test_save_dialog)
        layout.addWidget(test_btn)
    
    def test_save_dialog(self):
        """Test save dialog with focus restoration."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Test Save", "", "Text files (*.txt)")
        
        if file_path:
            QMessageBox.information(self, "Success", f"Would save to: {file_path}")
            
            # Restore focus
            try:
                restore_window_focus_after_dialog(self)
                print("✅ Focus restoration applied")
            except Exception as e:
                print(f"❌ Focus restoration failed: {e}")
                # Fallback
                self.raise_()
                self.activateWindow()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
```

## Troubleshooting

### Issue: Focus manager import fails
**Solution**: Make sure `core/window_focus_manager.py` exists and is in the Python path. The fallback code should still work.

### Issue: Focus restoration doesn't work on some platforms
**Solution**: The focus manager includes platform-specific code. If issues persist, try increasing the delay:

```python
restore_window_focus_after_dialog(self, delay_ms=200)  # Increase delay
```

### Issue: Multiple dialogs interfere with each other
**Solution**: Use the context manager approach for complex dialog sequences.

## Migration Checklist

For each file containing file dialogs:

- [ ] Identify all `QFileDialog.getSaveFileName()` calls
- [ ] Identify all `QFileDialog.getOpenFileName()` calls  
- [ ] Identify all `QFileDialog.getExistingDirectory()` calls
- [ ] Add focus restoration after successful operations
- [ ] Test the fix on your platform
- [ ] Add fallback code for import errors

## Benefits of This Fix

1. **Better User Experience**: Windows stay in focus after save operations
2. **Cross-Platform Compatibility**: Works on macOS, Windows, and Linux
3. **Robust Implementation**: Includes fallbacks and error handling
4. **Easy to Apply**: Simple code addition to existing functions
5. **Non-Breaking**: Doesn't change existing functionality, only improves it

## Future Enhancements

Potential improvements for the focus manager:

1. **Auto-detection**: Automatically detect file dialog usage and apply fixes
2. **Configuration**: Allow users to configure focus behavior
3. **Metrics**: Track focus restoration success rates
4. **Integration**: Build into custom dialog classes

---

## Summary

The window focus issue is now resolved with a comprehensive solution that:

- ✅ **Fixes the immediate problem** of windows going to background after save operations
- ✅ **Works across all platforms** (macOS, Windows, Linux)
- ✅ **Provides multiple implementation options** (simple, decorator, context manager)
- ✅ **Includes robust error handling** and fallbacks
- ✅ **Is easy to apply** to existing code

Apply the fixes gradually across the codebase, starting with the most commonly used save operations, and test thoroughly on your platform. 