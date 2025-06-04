# Matplotlib Toolbar Size Configuration for RamanLab

This document explains how to customize the matplotlib toolbar size in your RamanLab application.

## 🎯 Quick Solutions

### 1. **Automatic (Recommended)**
The application now uses compact toolbars by default. The `ui/matplotlib_config.py` module automatically configures all matplotlib plots with:
- **30% smaller icons**
- **Reduced toolbar height** (32px instead of ~40px)
- **Professional styling** with hover effects
- **Optimized fonts** for better readability

### 2. **Manual Size Control**
You can manually adjust toolbar sizes by importing different toolbar classes:

```python
from ui.matplotlib_config import CompactNavigationToolbar, MiniNavigationToolbar

# Use compact toolbar (30% smaller)
toolbar = CompactNavigationToolbar(canvas, parent)

# Use mini toolbar (40% smaller, fewer buttons)  
toolbar = MiniNavigationToolbar(canvas, parent)
```

### 3. **Global Theme Configuration**
Apply different themes across your entire application:

```python
from ui.matplotlib_config import apply_theme

# Ultra-compact for small screens
apply_theme('minimal')

# Professional compact (default)
apply_theme('compact') 

# High-quality for publications
apply_theme('publication')
```

## 📏 Toolbar Size Comparison

| Toolbar Type | Icon Size | Height | Best For |
|-------------|-----------|---------|----------|
| **Standard** | 100% | ~40px | Large screens |
| **Compact** | 70% | 32px | **Most uses** |
| **Mini** | 70% | 28px | Small interfaces |

## 🔧 Advanced Customization

### Custom Icon Sizes
```python
class CustomToolbar(CompactNavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        # Make even smaller (50% of original)
        self.setIconSize(self.iconSize() * 0.5)
        self.setMaximumHeight(24)
```

### Per-Module Configuration
```python
# In your UI module
from ui.matplotlib_config import get_toolbar_class

# Choose size based on widget context
ToolbarClass = get_toolbar_class('mini')  # or 'compact', 'standard'
self.toolbar = ToolbarClass(self.canvas, self)
```

### Dynamic Size Adjustment
```python
# Adjust size based on screen DPI or window size
import sys
from PySide6.QtWidgets import QApplication

app = QApplication.instance()
screen = app.primaryScreen()
dpi = screen.logicalDotsPerInch()

if dpi > 150:  # High DPI display
    toolbar_size = 'mini'
else:
    toolbar_size = 'compact'
```

## 🎨 Styling Options

### Toolbar Colors
```python
toolbar.setStyleSheet("""
    QToolBar {
        background-color: #2c3e50;  /* Dark theme */
        border-bottom: 1px solid #34495e;
    }
    QToolButton {
        color: white;
    }
    QToolButton:hover {
        background-color: #34495e;
    }
""")
```

### Icon-Only Mode
```python
class IconOnlyToolbar(CompactNavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        # Hide text labels, show only icons
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
```

## 📱 Responsive Design

### Auto-sizing Based on Window Size
```python
def update_toolbar_size(self):
    """Automatically adjust toolbar size based on window dimensions."""
    window_width = self.width()
    
    if window_width < 800:
        self.toolbar = MiniNavigationToolbar(self.canvas, self)
    elif window_width < 1200:
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
    else:
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
```

## 🛠️ Troubleshooting

### Problem: Toolbar still looks large
**Solution:** Ensure you're importing from the correct module:
```python
# ❌ Wrong
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

# ✅ Correct  
from ui.matplotlib_config import CompactNavigationToolbar
```

### Problem: Icons are blurry on high-DPI displays
**Solution:** Increase the DPI setting:
```python
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150  # or higher
```

### Problem: Toolbar doesn't fit theme
**Solution:** Apply a consistent theme:
```python
from ui.matplotlib_config import apply_theme
apply_theme('compact')  # Ensure consistent styling
```

## 📋 File Locations

- **Main config:** `ui/matplotlib_config.py` - Central configuration
- **Spectrum module:** `ui/spectrum_analysis.py` - Uses CompactNavigationToolbar
- **Peak fitting:** `ui/peak_fitting.py` - Configured automatically
- **Crystal structure:** `ui/crystal_structure.py` - Configured automatically

## ✅ Current Status

Your RamanLab application now automatically uses:
- ✅ **CompactNavigationToolbar** in Spectrum Analysis module
- ✅ **Global matplotlib configuration** for consistent sizing
- ✅ **Professional styling** with hover effects
- ✅ **Optimized fonts** for better readability
- ✅ **Responsive design** ready for different screen sizes

The toolbar should now be approximately **30% smaller** than before while maintaining full functionality! 