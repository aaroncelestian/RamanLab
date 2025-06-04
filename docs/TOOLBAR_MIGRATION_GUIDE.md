# Toolbar Migration Guide for RamanLab PySide6

This guide explains how to systematically apply the compact toolbar sizing methods from the `ui/` folder to all PySide6 code in RamanLab.

## 🎯 Current Status

The `ui/matplotlib_config.py` module provides excellent compact toolbar implementations:
- **CompactNavigationToolbar**: 30% smaller icons, 32px height, professional styling
- **MiniNavigationToolbar**: Ultra-compact with essential tools only  
- **Global configuration**: Optimized matplotlib settings for embedded plots

## 🔧 Migration Steps

### Step 1: Update Imports

**Replace this pattern:**
```python
try:
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
```

**With this:**
```python
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar
```

### Step 2: Choose Appropriate Toolbar Size

For different contexts, choose the right toolbar:

```python
from ui.matplotlib_config import (
    CompactNavigationToolbar,    # Default choice - 30% smaller
    MiniNavigationToolbar,       # For small interfaces
    get_toolbar_class           # Dynamic selection
)

# Standard approach
self.toolbar = CompactNavigationToolbar(self.canvas, self)

# For small interfaces
self.toolbar = MiniNavigationToolbar(self.canvas, self)

# Dynamic sizing based on context
ToolbarClass = get_toolbar_class('compact')  # or 'mini', 'standard'
self.toolbar = ToolbarClass(self.canvas, self)
```

### Step 3: Apply Global Configuration

Add to the beginning of your main application classes:

```python
from ui.matplotlib_config import configure_compact_ui, apply_theme

# In your __init__ method:
configure_compact_ui()  # Apply global compact settings

# Or use themed approach:
apply_theme('compact')    # Professional compact (recommended)
apply_theme('minimal')    # Ultra-minimal for small screens  
apply_theme('publication') # High-quality for publications
```

## 📋 Files to Update

### High Priority (Main Applications)
- [x] `ui/spectrum_analysis.py` ✅ Already updated
- [ ] `raman_analysis_app_qt6.py`
- [ ] `map_analysis_2d_qt6.py` 
- [ ] `batch_peak_fitting_qt6.py`
- [ ] `raman_cluster_analysis_qt6.py`
- [ ] `mineral_modes_browser_qt6.py`
- [ ] `spectrum_viewer_qt6.py`
- [ ] `multi_spectrum_manager_qt6.py`

### Medium Priority (Specialized Tools)
- [ ] `raman_polarization_analyzer_modular_qt6.py`
- [ ] `database_browser_qt6.py`
- [ ] `peak_fitting_qt6.py`

### Low Priority (Backup/Test Files)
- [ ] `mineral_modes_browser_qt6_backup.py`
- [ ] Test files and examples

## 🎨 Best Practices

### 1. Consistent Sizing Strategy
```python
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Apply global theme first
        from ui.matplotlib_config import apply_theme
        apply_theme('compact')
        
        # Then create toolbars
        self.setup_plots()
    
    def setup_plots(self):
        from ui.matplotlib_config import CompactNavigationToolbar
        
        # All plots use compact toolbar for consistency
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
```

### 2. Context-Aware Sizing
```python
def create_toolbar(self, canvas, parent, context='main'):
    """Create toolbar with appropriate size for context."""
    from ui.matplotlib_config import get_toolbar_class
    
    # Choose size based on available space
    if context == 'popup' or context == 'small_widget':
        size = 'mini'
    elif context == 'main' or context == 'analysis':
        size = 'compact'
    else:
        size = 'standard'
    
    ToolbarClass = get_toolbar_class(size)
    return ToolbarClass(canvas, parent)
```

### 3. Responsive Design
```python
def update_toolbar_size(self):
    """Adjust toolbar size based on window dimensions."""
    window_width = self.width()
    
    if window_width < 800:
        toolbar_class = MiniNavigationToolbar
    elif window_width < 1200:
        toolbar_class = CompactNavigationToolbar
    else:
        toolbar_class = NavigationToolbar2QT  # Full size for large screens
    
    # Recreate toolbar with new size
    old_toolbar = self.toolbar
    self.toolbar = toolbar_class(self.canvas, self)
    
    # Replace in layout
    layout = old_toolbar.parent().layout()
    layout.replaceWidget(old_toolbar, self.toolbar)
    old_toolbar.deleteLater()
```

## 🚀 Automated Migration Script

You can create a migration script to update multiple files:

```python
import re
import os
from pathlib import Path

def migrate_toolbar_imports(file_path):
    """Migrate a single file to use compact toolbars."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace import pattern
    old_pattern = r'from matplotlib\.backends\.backend_[^\\n]+ import NavigationToolbar2QT as NavigationToolbar'
    new_pattern = 'from ui.matplotlib_config import CompactNavigationToolbar as NavigationToolbar'
    
    content = re.sub(old_pattern, new_pattern, content)
    
    # Add global configuration if not present
    if 'configure_compact_ui()' not in content and 'class ' in content:
        # Find first class definition and add configuration
        class_match = re.search(r'class\s+\w+[^:]*:', content)
        if class_match:
            # Add import and configuration
            imports_end = content.find('\n\n')
            if imports_end != -1:
                content = (content[:imports_end] + 
                          '\nfrom ui.matplotlib_config import configure_compact_ui\n' +
                          content[imports_end:])
    
    with open(file_path, 'w') as f:
        f.write(content)

# Usage
qt6_files = [
    'raman_analysis_app_qt6.py',
    'map_analysis_2d_qt6.py', 
    'batch_peak_fitting_qt6.py',
    # ... add other files
]

for file_path in qt6_files:
    if os.path.exists(file_path):
        migrate_toolbar_imports(file_path)
        print(f"✅ Updated {file_path}")
```

## ✅ Verification Checklist

After migration, verify each file has:

- [ ] Import from `ui.matplotlib_config` instead of matplotlib backends
- [ ] Global configuration applied (`configure_compact_ui()` or `apply_theme()`)
- [ ] Consistent toolbar sizing across all plots
- [ ] No toolbar sizing conflicts or inconsistencies
- [ ] Proper error handling for import fallbacks

## 🎯 Expected Results

After migration, you should see:
- **30% smaller toolbar icons** across all applications
- **Reduced toolbar height** (32px instead of ~40px)
- **Consistent professional styling** with hover effects
- **Better space utilization** in your UI layouts
- **Unified appearance** across all RamanLab modules

## 🔍 Testing

Test each migrated file by:
1. Running the application
2. Verifying toolbar appears correctly sized
3. Testing all toolbar functions (zoom, pan, save, etc.)
4. Checking for any visual inconsistencies
5. Confirming hover effects and styling work properly

---

This systematic approach ensures all your PySide6 applications have consistent, professional-looking compact toolbars while maintaining full functionality. 