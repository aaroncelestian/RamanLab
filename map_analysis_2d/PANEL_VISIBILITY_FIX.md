# Map View Panel Visibility Fix

## Issue Description

The panels in the map view table were no longer visible due to a runtime error:

```
RuntimeError: Internal C++ object (MapViewControlPanel) already deleted.
```

This error occurred when switching between tabs in the main application window.

## Root Cause

The issue was caused by a widget lifecycle management problem in the `on_tab_changed` method in `main_window.py`:

1. When switching tabs, `clear_dynamic_sections()` was called to remove old control panels
2. This method called `deleteLater()` on the cached `MapViewControlPanel`, scheduling its deletion
3. However, the Python reference was still stored in `self.control_panel_cache['map_view']`
4. When returning to the Map View tab, the code tried to reuse the cached panel whose C++ object had been deleted

## Solution

The fix involved modifying the `on_tab_changed` method to:

1. **Check widget validity before reuse**: Before using a cached control panel, we now check if the underlying C++ object is still valid by trying to access a basic property (`isVisible()`).

2. **Handle deleted widgets gracefully**: If a `RuntimeError` is caught (indicating the C++ object was deleted), we remove the invalid reference from the cache and create a new panel.

3. **Prevent deletion of cached widgets**: Before calling `clear_dynamic_sections()`, we manually remove cached control panels from the layout to prevent them from being scheduled for deletion.

## Code Changes

### In `ui/main_window.py` - `on_tab_changed` method:

```python
def on_tab_changed(self, index: int):
    """Handle tab changes to update control panel."""
    # Remove cached control panels from layout before clearing sections
    if hasattr(self, 'control_panel_cache'):
        for cached_widget in self.control_panel_cache.values():
            try:
                # Try to remove from layout if it's still valid
                if cached_widget.parent() == self.controls_panel.content_widget:
                    self.controls_panel.main_layout.removeWidget(cached_widget)
            except (RuntimeError, AttributeError):
                # Widget was already deleted or doesn't have expected structure
                pass
    
    self.controls_panel.clear_dynamic_sections()
    
    if index == 0:  # Map View
        cached_panel = self.control_panel_cache.get('map_view')
        # Check if cached panel exists and is still valid (not deleted)
        if cached_panel is not None:
            try:
                # Try to access a basic property to check if C++ object is still valid
                _ = cached_panel.isVisible()
                control_panel = cached_panel
            except RuntimeError:
                # C++ object was deleted, remove from cache
                del self.control_panel_cache['map_view']
                cached_panel = None
        
        if cached_panel is None:
            # Create new control panel and set up connections...
```

## Benefits

- **Robust widget management**: The application now gracefully handles cases where cached widgets are deleted
- **Preserved functionality**: The caching mechanism still works for performance, but with proper validation
- **Error prevention**: The `RuntimeError: Internal C++ object already deleted` error is completely prevented
- **Backward compatibility**: All existing functionality remains unchanged

## Testing

The fix can be tested by:
1. Loading map data
2. Switching between different tabs (Map View → Template → Map View)
3. Verifying that the Map View control panel remains visible and functional
4. Checking that no runtime errors occur in the console

The panels should now remain visible and functional when switching between tabs. 