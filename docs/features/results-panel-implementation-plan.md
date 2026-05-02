# Results Panel Implementation Plan

## Overview

Implementation of a permanent results panel in the Peak Fitting interface that displays:
1. Per-peak total integrated areas across the whole map, with a grand total and Copy button
2. Detailed per-pixel information (integrated area, center, width per peak + R²) when clicking on the map
3. Auto-shows the fitted spectrum in the bottom pane on click

**Status:** Planning — decisions finalized
**Priority:** High
**Estimated Time:** 12-16 hours
**Target Version:** 2.1.0

## User Requirements

> "Когда появляется постоянная панель с результатами, которые ты предложил, то есть в ней будет показываться всегда, после того как фитинг произошел, общее количество, общая площадь, а также будет значение точечные. То есть, когда я тыкаю мышкой на определенную точку карты, должны показываться значения в этой точке."

### Key Features
- Permanent panel showing results after peak fitting completes
- Overall statistics: per-peak total integrated areas + grand total (easy to copy)
- Click on map point → show integrated area, center, width per peak + R² for that pixel
- Click auto-shows the fitted spectrum (raw + total fit + individual peak components) in bottom pane
- Always visible during peak fitting workflow; grayed-out placeholder before first fit

## Architecture

### Panel Location
- **Position:** Bottom of the existing left sidebar (`MapPeakFittingControlPanel`)
- **Implementation:** `QGroupBox` embedded in the existing `QScrollArea` — no dock widget needed
- **Collapsible:** No — the scroll area handles overflow
- **State:** Always expanded; cleared when fitting config changes

### Rationale for QGroupBox over QDockWidget
The left sidebar already has empty space below the existing controls (due to `addStretch()`). Embedding there avoids adding a floating dock widget and keeps the UI consistent with the existing control panel pattern.

### Panel Structure

```text
┌─ Results Summary ──────────────────┐
│ [grayed placeholder until fit runs] │
│                                     │
│ Peak 1 total area:  45,231.42       │
│ Peak 2 total area:  28,110.20       │
│ ─────────────────────────────────── │
│ Total:              73,341.62  [Copy]│
├─────────────────────────────────────┤
│ Selected Pixel                      │
│ [click a map pixel to see details]  │
│                                     │
│ Position: (X=12.5 μm, Y=8.0 μm)    │
│                                     │
│ Peak 1:  area 45.23 | cen 520.3 | wid 8.2  │
│ Peak 2:  area 12.15 | cen 480.1 | wid 5.4  │
│                                     │
│ R²:  0.98                           │
└─────────────────────────────────────┘
```

**Failed pixel state:**
```text
│ Selected Pixel                      │
│ Position: (X=12.5 μm, Y=8.0 μm)    │
│ Fit failed for this pixel           │
```
(Bottom spectrum pane shows raw spectrum only, no fit overlay.)

## Design Decisions (finalized 2026-05-01)

| Decision | Choice | Rationale |
|---|---|---|
| Panel widget type | `QGroupBox` in left sidebar | Existing empty space; no dock complexity |
| Update trigger | Click only (not hover) | Low overhead; sufficient for analysis |
| Overall stats | Per-peak total areas + grand total | Physically meaningful; per-peak useful for separating contributions |
| Copy button | Next to grand total only | Most common copy target |
| Per-pixel: coordinates | Spatial (μm) from LabSpec metadata; fall back to array indices | Directly relatable to physical sample |
| Per-pixel: parameters shown | Integrated area + center + width per peak, R² | Amplitude and eta omitted — not actionable in results view |
| Spectrum on click | Auto-show in bottom pane | No manual "Show Spectrum" button needed |
| Spectrum content | Raw data + total fit curve + individual peak components | Deconvolved view confirms peak separation |
| Failed pixel | "Fit failed" message + raw spectrum in bottom pane | Honest; lets user see why the fit failed |
| Panel state (no results) | Grayed-out placeholder text | Always visible so user knows it exists |
| Panel state (config change) | Clear immediately | Stale numbers in a results panel are worse than empty |
| Collapsible | No | Scroll area handles overflow |
| Number format (panel) | Fixed 2 decimal places | Easier to compare visually |
| Number format (CSV) | Full precision | For downstream analysis |
| Export format | Per-pixel CSV: X, Y, Peak1_area, Peak1_center, Peak1_width, …, R² | Full spatial grid for downstream analysis |
| New file | `map_analysis_2d/ui/results_panel.py` | Panel has its own state; keeps control_panels.py focused |

## Implementation Phases

### Phase 1: Base Panel Structure (2-3 hours)

**Goal:** Create the panel widget as a `QGroupBox` and embed it in `MapPeakFittingControlPanel`

**Tasks:**
- [ ] Create new file `map_analysis_2d/ui/results_panel.py`
- [ ] Implement `ResultsPanel(QGroupBox)` class with two sections:
  - `OverallStatsWidget` — overall statistics section
  - `PixelDetailsWidget` — pixel details section
- [ ] Append panel to `MapPeakFittingControlPanel` layout in `control_panels.py`
- [ ] Show grayed-out placeholder text in both sections on init

**Files to modify:**
- New: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/control_panels.py`

**Code Structure:**
```python
class ResultsPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Results Summary", parent)
        layout = QVBoxLayout(self)

        self.overall_stats = OverallStatsWidget()
        self.pixel_details = PixelDetailsWidget()

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)

        layout.addWidget(self.overall_stats)
        layout.addWidget(separator)
        layout.addWidget(self.pixel_details)

    def clear(self):
        self.overall_stats.clear()
        self.pixel_details.clear()
```

### Phase 2: Overall Statistics Display (2-3 hours)

**Goal:** Compute and display per-peak totals + grand total after fitting completes

**Tasks:**
- [ ] Create `compute_overall_statistics()` function in `map_analysis_2d/core/statistics.py`
- [ ] Calculate per-peak total integrated areas (using `compute_integrated_intensity()`)
- [ ] Calculate grand total (sum of all per-peak totals across all pixels)
- [ ] Implement `update_overall_stats(results_dict)` method
- [ ] Format numbers: fixed 2 decimal places with thousands separator
- [ ] Add Copy button next to grand total (copies plain number to clipboard)
- [ ] Connect to peak fitting worker completion signal (`@Slot` — worker on QThread)
- [ ] Clear stats when fitting config changes (wavenumber range, number of peaks)

**Files to modify:**
- New: `map_analysis_2d/core/statistics.py`
- Modified: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/main_window.py`

**Statistics Computation:**
```python
def compute_overall_statistics(peak_fitting_results):
    """Compute per-peak and grand total integrated areas."""
    n_peaks = peak_fitting_results['n_peaks']
    shapes = peak_fitting_results['peak_shapes']
    map_params = peak_fitting_results['map_parameters']

    per_peak_totals = {f'Peak {i}': 0.0 for i in range(1, n_peaks + 1)}

    for pos_key in peak_fitting_results['positions']:
        for i, shape in enumerate(shapes, 1):
            amp = map_params.get(f'P{i}_Amp', {}).get(pos_key, np.nan)
            wid = map_params.get(f'P{i}_Wid', {}).get(pos_key, np.nan)
            eta = map_params.get(f'P{i}_Eta', {}).get(pos_key, 0.5)

            if np.isfinite(amp) and np.isfinite(wid):
                ii = compute_integrated_intensity(amp, wid, shape, eta)
                per_peak_totals[f'Peak {i}'] += ii

    grand_total = sum(per_peak_totals.values())
    return {'per_peak_totals': per_peak_totals, 'grand_total': grand_total}
```

### Phase 3: Pixel Details on Map Click (3-4 hours)

**Goal:** Show per-pixel details and auto-display fitted spectrum when user clicks a map pixel

**Tasks:**
- [ ] Connect matplotlib `button_press_event` on the peak fitting map canvas
- [ ] Convert click `(xdata, ydata)` to spatial coordinates → nearest pixel index
- [ ] Use spatial coordinates from LabSpec metadata when available; fall back to array indices
- [ ] Display in panel: position (μm), area + center + width per peak (fixed 2 dp), R²
- [ ] For failed pixels: show "Fit failed for this pixel" with coordinates
- [ ] Auto-show bottom spectrum pane with:
  - Successful pixel: raw spectrum + total fit curve + individual peak components (dashed)
  - Failed pixel: raw spectrum only, no fit overlay
- [ ] Add/update visual marker (red star) on map for selected pixel
- [ ] Clear pixel details when fitting config changes

**Files to modify:**
- Modified: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/main_window.py`

**Click Handler Implementation:**
```python
def _connect_peak_fitting_click_handler(self):
    canvas = self.peak_fitting_plot_widget.map_widget.canvas
    self._peak_fitting_click_cid = canvas.mpl_connect(
        'button_press_event', self._on_peak_fitting_map_click
    )

def _on_peak_fitting_map_click(self, event):
    if not event.inaxes or self.peak_fitting_results is None:
        return

    pos = self._find_closest_pixel(event.xdata, event.ydata)
    if pos is None:
        return

    # Update results panel
    self.peak_fitting_control_panel.results_panel.update_pixel_details(
        pos, self.peak_fitting_results, self.map_data
    )

    # Auto-show spectrum in bottom pane
    self._show_peak_fitting_spectrum(pos)

    # Update map marker
    self._highlight_peak_fitting_pixel(pos)

def _find_closest_pixel(self, x, y):
    """Find nearest pixel to clicked coordinates using spatial positions."""
    positions = self.peak_fitting_results['positions']  # list of (x, y) tuples
    if not positions:
        return None
    coords = np.array(positions)
    dists = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
    return positions[int(np.argmin(dists))]
```

### Phase 4: Export to CSV (1-2 hours)

**Goal:** Export full per-pixel fitting results as a CSV file

**Tasks:**
- [ ] Add "Export Results CSV" button to the results panel
- [ ] On click: open file dialog, write CSV with one row per pixel
- [ ] Columns: X (μm), Y (μm), Peak1_area, Peak1_center, Peak1_width, [Peak2_...], ..., R², fit_status
- [ ] Use full float precision in CSV (not the 2 dp display format)
- [ ] Skip or mark NaN rows for failed pixels

**Files to modify:**
- Modified: `map_analysis_2d/ui/results_panel.py`
- Modified: `map_analysis_2d/ui/main_window.py`

**CSV Format:**
```
X_um,Y_um,Peak1_area,Peak1_center,Peak1_width,Peak2_area,Peak2_center,Peak2_width,R2,status
12.5,8.0,45.234567,520.312,8.215,12.154321,480.1,5.432,0.9845,success
12.5,8.5,,,,,,,nan,failed
```

### Phase 5: Testing and Documentation (2-3 hours)

**Goal:** Ensure robustness and document the feature

**Tasks:**
- [ ] Test with real silicon Raman data (1-peak and 2-peak fits)
- [ ] Test edge cases:
  - No successful fits (all-failed map)
  - Single pixel map
  - Mixed success/failure map
  - LabSpec file without spatial coordinates (array index fallback)
- [ ] Test Copy button copies correct value to clipboard
- [ ] Test CSV export column headers match number of fitted peaks
- [ ] Performance test on large maps (10,000+ pixels)
- [ ] Update README.md with new feature description

**Files to modify:**
- New: `map_analysis_2d/test_results_panel.py`
- Modified: `README.md`

## File Structure

```text
map_analysis_2d/
├── ui/
│   ├── __init__.py              # Update exports
│   ├── results_panel.py         # NEW: ResultsPanel, OverallStatsWidget, PixelDetailsWidget
│   ├── control_panels.py        # MODIFIED: append ResultsPanel to MapPeakFittingControlPanel
│   └── main_window.py           # MODIFIED: click handler, spectrum display, export wiring
└── core/
    ├── __init__.py              # Update exports
    └── statistics.py            # NEW: compute_overall_statistics()

docs/
└── features/
    └── results-panel-implementation-plan.md  # This document
```

## Technical Requirements

### Dependencies
- PySide6 (Qt6) — already included
- NumPy — already included
- Matplotlib — already included
- No new dependencies required

### Python Compatibility
- Python 3.10+ (per repository standard)
- Use type hints where appropriate
- Follow existing code style

### Performance Considerations
- Statistics computation: O(n) where n = number of pixels; runs once after fitting
- Click handling: O(n) for nearest-pixel search; acceptable for typical map sizes (<10,000 pixels); consider `scipy.spatial.KDTree` if slow on very large maps
- Spectrum display: reuses existing `SplitMapSpectrumWidget` bottom pane

### Error Handling
- Handle missing/incomplete fit results gracefully — show "Fit failed" for bad pixels
- Validate pixel position before data access
- Catch and log matplotlib event errors
- Clear panel immediately on fitting config change (never show stale data)

## Time Estimates

| Phase | Description | Time Estimate |
|-------|-------------|---------------|
| 1 | Base Panel Structure (QGroupBox) | 2-3 hours |
| 2 | Overall Statistics + Copy button | 2-3 hours |
| 3 | Pixel Details on Click + Spectrum | 3-4 hours |
| 4 | Export to CSV | 1-2 hours |
| 5 | Testing & Documentation | 2-3 hours |
| **Total** | | **10-15 hours** |

### Minimum Viable Product (MVP)
Phases 1-3: **7-10 hours**
- Panel with per-peak totals, Copy button, and per-pixel click details
- Auto-show spectrum on click
- No CSV export yet

## Success Criteria

### Functional Requirements
- [ ] Panel displays grayed placeholder before first fit; populates after fitting completes
- [ ] Shows per-peak total areas and grand total after fitting
- [ ] Copy button copies grand total to clipboard
- [ ] Click on map pixel updates per-pixel section with area, center, width, R²
- [ ] Click auto-shows fitted spectrum (raw + total + components) in bottom pane
- [ ] Failed pixel shows "Fit failed" message and raw spectrum only
- [ ] Spatial coordinates shown in μm when available; array indices as fallback
- [ ] Panel clears when fitting config changes
- [ ] CSV export contains one row per pixel at full precision

### User Experience
- [ ] Response to map clicks is immediate (<100ms)
- [ ] Numbers formatted with 2 decimal places and thousands separator
- [ ] Panel does not obstruct map view (embedded in existing sidebar)

### Code Quality
- [ ] Follows existing code style (PySide6, Qt signals/slots, `@Slot` decorator)
- [ ] No performance degradation on maps up to 10,000 pixels
- [ ] Error handling for all edge cases

## Future Enhancements

- Multi-pixel selection (shift+click for range statistics)
- Statistics filtering (by R² threshold)
- Export statistics to Excel with charts
- Integration with database for historical tracking

## References

- Related Commits:
  - b67bebd — Add integrated intensity export and map session persistence
  - cc6069ec — Merge feature/map-peak-fitting (base implementation)
- `compute_integrated_intensity()` in `map_analysis_2d/core/math_models.py` — reused for area calculation

## Notes

- `QGroupBox` chosen over `QDockWidget`: existing `QScrollArea` sidebar has space below controls; simpler integration
- Panel state persistence (position, floating) not needed since panel is embedded, not dockable
- The `compute_integrated_intensity()` function from `core/math_models.py` is reused for all area calculations
- Spectrum display reuses the existing hidden bottom pane in `SplitMapSpectrumWidget`

---

**Document Version:** 2.0
**Last Updated:** 2026-05-01
**Author:** Implementation planning session with user review
**Status:** Ready for implementation
