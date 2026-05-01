# Results Panel Implementation Plan

## Overview

Implementation of a permanent results panel in the Peak Fitting interface that displays:
1. Overall statistics after fitting completion (total area, count, mean, etc.)
2. Detailed information about a specific pixel when clicking on the map
3. Automatic updates and always visible

**Status:** Planning
**Priority:** High
**Estimated Time:** 15-20 hours
**Target Version:** 2.1.0

## User Requirements

> "Когда появляется постоянная панель с результатами, которые ты предложил, то есть в ней будет показываться всегда, после того как фитинг произошел, общее количество, общая площадь, а также будет значение точечные. То есть, когда я тыкаю мышкой на определенную точку карты, должны показываться значения в этой точке."

### Key Features
- Permanent panel showing results after peak fitting
- Overall statistics: total area, total count, averages
- Click on map point → show values for that specific pixel
- Always visible during peak fitting workflow

## Architecture

### Panel Location
- **Position:** Right side of Peak Fitting interface
- **Implementation:** QDockWidget (allows moving/hiding)
- **Size:** Minimum width 250px, recommended 300-350px
- **State:** Dockable, resizable, closable via View menu

### Panel Structure

```
┌─ Results Summary ──────────────────┐
│ Overall Statistics                  │
│ ├─ Total Area: 123,456.78          │
│ ├─ Fitted Pixels: 2,500/2,500      │
│ ├─ Success Rate: 98.2%             │
│ ├─ Mean Area: 49.38 ± 12.45        │
│ └─ Median Area: 47.82              │
│                                     │
│ Per-Peak Statistics:                │
│ ├─ Peak 1 Total: 95,234.56         │
│ └─ Peak 2 Total: 28,222.22         │
│                                     │
│ [Export Summary] [Copy to Clipboard]│
├─────────────────────────────────────┤
│ Selected Pixel Details              │
│ Position: (X=10, Y=15)              │
│                                     │
│ Integrated Intensities:             │
│ ├─ Peak 1: 45.23                   │
│ ├─ Peak 2: 12.15                   │
│ └─ Total: 57.38                    │
│                                     │
│ Fit Parameters:                     │
│ ├─ P1_Amp: 1234.5                  │
│ ├─ P1_Cen: 520.3                   │
│ ├─ P1_Wid: 8.2                     │
│ └─ ...                              │
│                                     │
│ Fit Quality:                        │
│ ├─ R²: 0.9845                      │
│ └─ Status: ✓ Success               │
│                                     │
│ [Show Spectrum] [Export Point Data] │
└─────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Base Panel Structure (2-3 hours)

**Goal:** Create the basic panel widget and integrate it into main window

**Tasks:**
- [ ] Create new file `map_analysis_2d/ui/results_panel.py`
- [ ] Implement `ResultsPanel(QDockWidget)` class
- [ ] Create two main sections:
  - `OverallStatsWidget` - overall statistics section
  - `PixelDetailsWidget` - pixel details section
- [ ] Integrate panel into `main_window.py` as QDockWidget
- [ ] Add View menu option to show/hide panel
- [ ] Basic styling and layout

**Files to modify:**
- New: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/main_window.py`

**Code Structure:**
```python
class ResultsPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Results Summary", parent)
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)

        # Create sections
        self.overall_stats = OverallStatsWidget()
        self.pixel_details = PixelDetailsWidget()

        # Add to layout
        self.layout.addWidget(self.overall_stats)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        self.layout.addWidget(separator)
        self.layout.addWidget(self.pixel_details)

        self.setWidget(self.main_widget)
```

### Phase 2: Overall Statistics Display (2-3 hours)

**Goal:** Compute and display overall statistics after peak fitting

**Tasks:**
- [ ] Create `compute_overall_statistics()` function
- [ ] Calculate key metrics:
  - Total integrated area (sum of all Total_IntInt)
  - Number of fitted pixels / total pixels
  - Success rate (% of successful fits based on R²)
  - Mean, median, std dev for Total_IntInt
  - Per-peak totals (if multiple peaks)
- [ ] Implement `update_overall_stats(results_dict)` method
- [ ] Format numbers appropriately (thousands separators, scientific notation)
- [ ] Connect to peak fitting worker completion signal (decorate slot with `@Slot` — worker runs on QThread, UI updates must arrive on the main thread via Qt's signal/slot mechanism)
- [ ] Handle edge cases (no data, all failed fits, etc.)

**Files to modify:**
- New: `map_analysis_2d/core/statistics.py`
- Modified: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/main_window.py`

**Statistics Computation:**
```python
def compute_overall_statistics(map_data, peak_fitting_results):
    """Compute overall statistics from peak fitting results."""
    stats = {
        'total_area': 0.0,
        'mean_area': 0.0,
        'median_area': 0.0,
        'std_area': 0.0,
        'fitted_count': 0,
        'total_count': len(map_data.spectra),
        'success_rate': 0.0,
        'per_peak_totals': {}
    }

    # Extract data
    total_areas = []
    shapes = peak_fitting_results['peak_shapes']
    map_params = peak_fitting_results['map_parameters']

    for pos_key, spectrum in map_data.spectra.items():
        # Compute total IntInt for this pixel
        total_int = 0.0
        all_valid = True

        for i, shape in enumerate(shapes, 1):
            amp = map_params.get(f'P{i}_Amp', {}).get(pos_key, np.nan)
            wid = map_params.get(f'P{i}_Wid', {}).get(pos_key, np.nan)
            eta = map_params.get(f'P{i}_Eta', {}).get(pos_key, 0.5)

            if np.isfinite(amp) and np.isfinite(wid):
                ii = compute_integrated_intensity(amp, wid, shape, eta)
                total_int += ii

                # Per-peak tracking
                peak_key = f'Peak {i}'
                if peak_key not in stats['per_peak_totals']:
                    stats['per_peak_totals'][peak_key] = 0.0
                stats['per_peak_totals'][peak_key] += ii
            else:
                all_valid = False

        if all_valid:
            total_areas.append(total_int)
            stats['fitted_count'] += 1

    # Calculate statistics
    if total_areas:
        stats['total_area'] = np.sum(total_areas)
        stats['mean_area'] = np.mean(total_areas)
        stats['median_area'] = np.median(total_areas)
        stats['std_area'] = np.std(total_areas)
        stats['success_rate'] = (stats['fitted_count'] / stats['total_count']) * 100

    return stats
```

### Phase 3: Pixel Details on Map Click (3-4 hours)

**Goal:** Show detailed information when user clicks on a pixel in the map

**Tasks:**
- [ ] Implement map click event handler in main_window
- [ ] Connect matplotlib event (`button_press_event`) to handler
- [ ] Convert click coordinates to map pixel position
- [ ] Find closest pixel with data
- [ ] Implement `update_pixel_details(pos, results)` method
- [ ] Display pixel information:
  - Coordinates (X, Y)
  - Integrated intensity per peak
  - Total integrated intensity
  - All fit parameters (Amp, Cen, Wid, Eta)
  - R-squared value
  - Fit status (Success/Warning/Error)
- [ ] Add visual marker on map for selected pixel
- [ ] Clear previous selection marker when new pixel is clicked

**Files to modify:**
- Modified: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/main_window.py`

**Click Handler Implementation:**
```python
def _connect_map_click_handler(self):
    """Connect click handler to peak fitting map."""
    if hasattr(self, 'peak_fitting_plot_widget'):
        canvas = self.peak_fitting_plot_widget.canvas
        self._map_click_cid = canvas.mpl_connect(
            'button_press_event', self._on_peak_fitting_map_click
        )

def _on_peak_fitting_map_click(self, event):
    """Handle click on peak fitting map."""
    if not event.inaxes or self.peak_fitting_results is None:
        return

    x_click = event.xdata
    y_click = event.ydata

    # Find closest pixel
    closest_pos = self._find_closest_pixel(x_click, y_click)

    if closest_pos:
        # Update results panel
        self.results_panel.update_pixel_details(
            closest_pos,
            self.peak_fitting_results,
            self.map_data
        )

        # Add visual marker
        self._highlight_selected_pixel(closest_pos)

def _find_closest_pixel(self, x, y):
    """Find the closest pixel to clicked coordinates."""
    min_dist = float('inf')
    closest_pos = None

    for pos_key in self.map_data.spectra.keys():
        px, py = pos_key
        dist = np.sqrt((px - x)**2 + (py - y)**2)
        if dist < min_dist:
            min_dist = dist
            closest_pos = pos_key

    return closest_pos

def _highlight_selected_pixel(self, pos):
    """Add visual marker for selected pixel on map."""
    if hasattr(self, '_selected_pixel_marker') and self._selected_pixel_marker:
        self._selected_pixel_marker.remove()

    ax = self.peak_fitting_plot_widget.ax
    x, y = pos
    self._selected_pixel_marker = ax.plot(
        x, y, 'r*', markersize=15, markeredgecolor='white',
        markeredgewidth=1.5, zorder=100
    )[0]
    self.peak_fitting_plot_widget.canvas.draw_idle()
```

### Phase 4: Export and Copy Functionality (1-2 hours)

**Goal:** Allow users to export and copy statistics

**Tasks:**
- [ ] Implement "Export Summary" button
  - Save statistics to .txt file
  - Include all overall statistics
  - Timestamp and map filename
- [ ] Implement "Copy to Clipboard" button
  - Format statistics as text
  - Copy to system clipboard
- [ ] Implement "Show Spectrum" button (for selected pixel)
  - Open spectrum in separate plot window
  - Show fit overlay
- [ ] Implement "Export Point Data" button
  - Export selected pixel data to CSV
  - Include all parameters and integrated intensities

**Files to modify:**
- Modified: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/main_window.py`

**Export Format:**
```
RamanLab Peak Fitting Results Summary
Generated: 2026-05-01 19:35:00
Map: silicon_sample_map.h5

OVERALL STATISTICS
==================
Total Integrated Area: 123,456.78
Fitted Pixels: 2,500 / 2,500
Success Rate: 98.2%
Mean Area: 49.38 ± 12.45
Median Area: 47.82

PER-PEAK STATISTICS
===================
Peak 1 Total: 95,234.56
Peak 2 Total: 28,222.22
```

### Phase 5: Visual Improvements (1-2 hours)

**Goal:** Enhance visual appearance and usability

**Tasks:**
- [ ] Add color coding (green for good fits, yellow/red for poor)
- [ ] Add icons for sections and buttons
- [ ] Create mini histogram for area distribution
- [ ] Add tooltips with additional information
- [ ] Implement number format toggle (scientific notation on/off)
- [ ] Add collapsible sections
- [ ] Improve spacing and alignment
- [ ] Add loading indicator during computation

**Files to modify:**
- Modified: `map_analysis_2d/ui/results_panel.py`
- New: `map_analysis_2d/ui/widgets/` (optional widget submodules)

### Phase 6: State Persistence (1 hour)

**Goal:** Remember panel state between sessions

**Tasks:**
- [ ] Save panel position and size to config
- [ ] Save panel visibility state (shown/hidden)
- [ ] Save format preferences (scientific notation, etc.)
- [ ] Restore state on startup
- [ ] Use existing ConfigManager

**Files to modify:**
- Modified: `map_analysis_2d/ui/main_window.py`
- Modified: `map_analysis_2d/ui/results_panel.py`

**Config Storage:**
```python
# Save state
cfg.set('results_panel.visible', self.results_panel.isVisible())
cfg.set('results_panel.width', self.results_panel.width())
cfg.set('results_panel.floating', self.results_panel.isFloating())
cfg.set('results_panel.scientific_notation', self.results_panel.use_scientific)

# Restore state
if cfg.get('results_panel.visible', True):
    self.results_panel.show()
```

### Phase 7: Testing and Documentation (2-3 hours)

**Goal:** Ensure robustness and document the feature

**Tasks:**
- [ ] Test with real silicon Raman data
- [ ] Test with different map sizes (small, medium, large)
- [ ] Test edge cases:
  - No successful fits
  - Single pixel map
  - Multiple peak shapes
  - Missing parameters
- [ ] Performance testing on large maps (10,000+ pixels)
- [ ] Add docstrings to all new functions
- [ ] Update README.md with new feature description
- [ ] Create user guide section in docs
- [ ] Add screenshots

**Files to modify:**
- Modified: `README.md`
- New: `docs/user_guide/results_panel.md` (optional)

## File Structure

```
map_analysis_2d/
├── ui/
│   ├── __init__.py              # Update exports
│   ├── results_panel.py         # NEW: Main panel class
│   ├── widgets/                 # NEW: (optional) Widget submodules
│   │   ├── __init__.py
│   │   ├── overall_stats.py    # NEW: Overall statistics widget
│   │   └── pixel_details.py    # NEW: Pixel details widget
│   └── main_window.py          # MODIFIED: Panel integration
└── core/
    ├── __init__.py              # Update exports
    └── statistics.py            # NEW: Statistics utilities

docs/
└── features/
    ├── README.md                # NEW: Features index
    └── results-panel-implementation-plan.md  # This document
```

## Technical Requirements

### Dependencies
- PySide6 (Qt6) - already included
- NumPy - already included
- Matplotlib - already included
- No new dependencies required

### Python Compatibility
- Python 3.10+ (per repository standard)
- Use type hints where appropriate
- Follow existing code style

### Performance Considerations
- Statistics computation: O(n) where n = number of pixels
- Click handling: O(n) for finding closest pixel (consider spatial index if slow)
- Memory: Minimal additional memory (statistics dict only)
- UI updates: Use Qt signals/slots for async updates

### Error Handling
- Handle missing/incomplete fit results gracefully
- Validate pixel position before access
- Catch and log matplotlib event errors
- Display user-friendly error messages

## Testing Strategy

### Unit Tests
- Statistics computation accuracy
- Coordinate transformation
- Data formatting functions

### Integration Tests
- Panel initialization with main window
- Event handler connections
- State persistence

### Manual Tests
- Click accuracy on different map sizes
- Export file generation
- Copy to clipboard functionality
- Visual appearance on different platforms (Windows, macOS, Linux)

## Time Estimates

| Phase | Description | Time Estimate |
|-------|-------------|---------------|
| 1 | Base Panel Structure | 2-3 hours |
| 2 | Overall Statistics | 2-3 hours |
| 3 | Pixel Details on Click | 3-4 hours |
| 4 | Export/Copy Functionality | 1-2 hours |
| 5 | Visual Improvements | 1-2 hours |
| 6 | State Persistence | 1 hour |
| 7 | Testing & Documentation | 2-3 hours |
| **Total** | **Full Implementation** | **15-20 hours** |

### Minimum Viable Product (MVP)
Phases 1-3 only: **8-10 hours**
- Basic panel with overall statistics and pixel click details
- No export functionality yet
- Basic styling only

## Prioritization

### Must Have (Critical)
- ✅ Phase 1: Base panel structure
- ✅ Phase 2: Overall statistics display
- ✅ Phase 3: Pixel details on click

### Should Have (Important)
- ⚠️ Phase 4: Export and copy functionality
- ⚠️ Phase 6: State persistence

### Nice to Have (Optional)
- ➕ Phase 5: Enhanced visualization
- ➕ Phase 7: Comprehensive testing

## Success Criteria

### Functional Requirements
- [ ] Panel displays after peak fitting completion
- [ ] Shows total area and per-peak statistics
- [ ] Click on map updates pixel details
- [ ] All statistics are mathematically correct
- [ ] Export functions work as expected

### User Experience
- [ ] Panel is intuitive and easy to read
- [ ] Response to clicks is immediate (<100ms)
- [ ] Panel doesn't obstruct map view
- [ ] Numbers are formatted clearly

### Code Quality
- [ ] Follows existing code style
- [ ] Properly documented
- [ ] No performance degradation
- [ ] Error handling implemented

## Future Enhancements

Potential improvements for future versions:
- Multi-pixel selection (shift+click for range)
- Statistics filtering (by R² threshold, area range)
- Time-series analysis (if multiple maps)
- Compare statistics between different fitting runs
- Export statistics to Excel with charts
- Integration with database for historical tracking

## References

- Related Issue: TBD (to be created)
- Related PR: TBD
- Related Commits:
  - b67bebd - Add integrated intensity export and map session persistence
- User Request: See issue comments

## Notes

- This feature builds on the integrated intensity export functionality added in commit b67bebd
- The `compute_integrated_intensity()` function from `core/math_models.py` will be reused
- Panel should be disabled/hidden when no peak fitting results are available
- Consider accessibility (screen readers, keyboard navigation) in future iterations

---

**Document Version:** 1.0
**Last Updated:** 2026-05-01
**Author:** Implementation planning session
**Status:** Ready for implementation
