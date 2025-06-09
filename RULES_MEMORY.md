# RamanLab Development Rules & Memory

## Matplotlib Rules

### 🔥 CRITICAL: Colorbar Shrinkage Fix
**PERMANENT RULE**: Never use `fig.colorbar()` directly - always use `add_colorbar_no_shrink()`

```python
# ❌ WRONG - causes plot shrinkage
colorbar = fig.colorbar(mappable, ax=ax)

# ✅ CORRECT - no shrinkage
from ui.matplotlib_config import add_colorbar_no_shrink
colorbar = add_colorbar_no_shrink(fig, mappable, ax)
```

**Why**: Matplotlib colorbars steal space from main plot, causing shrinkage. Our solution uses `make_axes_locatable` to create dedicated colorbar space.

**Files**: 
- Implementation: `ui/matplotlib_config.py` 
- Documentation: `MATPLOTLIB_COLORBAR_RULES.md`
- Applied to: `map_analysis_2d/ui/plotting_widgets.py`

---

## UI Design Rules

### Plot Configuration
- When making plots, always use the matplotlib_config.py file
- Apply compact themes for embedded plots
- Use consistent color schemes across the application

### Map View Controls
- Keep duplicate controls issue in mind - only show controls in left panel
- Wavenumber range integration: 100 cm⁻¹ width, 50 cm⁻¹ stepping
- Display scaling separate from integration controls

---

## Code Organization Rules

### Import Structure
- Always check for required imports in each module
- Handle missing dependencies gracefully with fallbacks
- Use try/except for optional features

### Signal Connections
- Connect all control panel signals in `on_tab_changed` method
- Ensure proper signal cleanup when switching tabs
- Test signal connections during development

---

## Application Architecture

### Control Panels
- Use left panel for primary controls
- Avoid duplicate control panels in different locations
- Implement proper tab-based control switching

### Data Integration
- PKL file compatibility: handle old module names with ModuleCompatibilityUnpickler
- Support both processed and raw data display
- Implement proper data validation before plotting

---

## Development Workflow

### Testing Requirements
- Test colorbar implementation on all plotting functions
- Verify no plot shrinkage occurs with window resizing
- Check control panel functionality across all tabs

### Documentation
- Update rules when persistent issues are solved
- Document permanent solutions for future reference
- Maintain compatibility notes for cross-module usage 