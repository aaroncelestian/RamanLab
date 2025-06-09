# MATPLOTLIB COLORBAR SHRINKAGE - PERMANENT SOLUTION

## The Problem
Matplotlib colorbars, when added with the default `figure.colorbar(mappable, ax=ax)` method, cause the main plot to shrink because they "steal" space from the main axes. This is a persistent issue across all matplotlib-based applications.

## PERMANENT RULE: Always Use the No-Shrink Solution

### For All New Code:
**ALWAYS use the `add_colorbar_no_shrink()` function from `ui/matplotlib_config.py`**

```python
from ui.matplotlib_config import add_colorbar_no_shrink

# Instead of:
# colorbar = fig.colorbar(mappable, ax=ax)  # BAD - causes shrinkage

# Use:
colorbar = add_colorbar_no_shrink(fig, mappable, ax)  # GOOD - no shrinkage
```

### Technical Implementation
The solution uses `mpl_toolkits.axes_grid1.make_axes_locatable` to create dedicated space for the colorbar:

```python
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
colorbar = figure.colorbar(mappable, cax=cax)
```

## Why This Works
1. **Dedicated Space**: Creates a separate axes specifically for the colorbar
2. **No Stealing**: The main plot axes remain unchanged
3. **Consistent Layout**: Predictable spacing and sizing
4. **Robust**: Works across different matplotlib versions and backends

## Fallback Hierarchy
The `add_colorbar_no_shrink()` function implements multiple fallbacks:

1. **Primary**: `make_axes_locatable` method (best)
2. **Secondary**: `fraction` and `pad` parameters
3. **Tertiary**: `shrink` parameter
4. **Final**: Return None if all methods fail

## Where to Apply This Rule

### ✅ MUST Use No-Shrink Solution:
- All 2D map visualizations
- Heatmaps and intensity plots
- Classification/clustering result plots
- Any plot where colorbar is essential

### Examples in RamanLab:
- `MapPlotWidget.plot_map()` ✅ FIXED
- Results tab comprehensive plots
- PCA/NMF visualization colorbars
- ML classification maps

## Migration Guide for Existing Code

### Before (Problematic):
```python
im = ax.imshow(data, cmap='viridis')
cbar = fig.colorbar(im, ax=ax)  # Causes shrinkage
```

### After (Fixed):
```python
im = ax.imshow(data, cmap='viridis')
cbar = add_colorbar_no_shrink(fig, im, ax)  # No shrinkage
```

## Testing the Fix
To verify the fix works:
1. Create a plot with colorbar
2. Resize the window
3. Check that the main plot maintains its proportions
4. Verify colorbar appears correctly positioned

## Memory Rule
**"Never use fig.colorbar() directly - always use add_colorbar_no_shrink()"**

This rule should be applied to:
- All new matplotlib plotting code
- Any existing code that exhibits colorbar shrinkage
- Code reviews - check for direct colorbar usage

## Dependencies
- Requires: `matplotlib` with `mpl_toolkits.axes_grid1`
- Fallback: Built-in matplotlib colorbar methods
- Compatible: All matplotlib versions 3.0+ 