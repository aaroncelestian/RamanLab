# Epic: Permanent Results Panel for Peak Fitting

## 🎯 Overview

Implement a permanent results panel in the Peak Fitting interface that displays overall statistics and interactive pixel-specific details.

**Status:** 📋 Planning
**Priority:** 🔴 High
**Target Version:** 2.1.0
**Estimated Effort:** 15-20 hours

## 📖 Full Implementation Plan

See detailed implementation plan: [docs/features/results-panel-implementation-plan.md](results-panel-implementation-plan.md)

## ✨ Key Features

### 1. Overall Statistics Display
After peak fitting completion, automatically show:
- ✅ Total integrated area across entire map
- ✅ Number of successfully fitted pixels
- ✅ Mean, median, standard deviation
- ✅ Per-peak statistics (when multiple peaks)
- ✅ Success rate

### 2. Interactive Pixel Details
Click on any point on the map to see:
- ✅ Pixel coordinates (X, Y)
- ✅ Integrated intensity per peak
- ✅ All fit parameters (amplitude, center, width)
- ✅ Fit quality (R-squared)
- ✅ Fit status (success/warning/error)

### 3. Export & Copy
- ✅ Export statistics summary to text file
- ✅ Copy statistics to clipboard
- ✅ Export individual pixel data
- ✅ Show spectrum for selected pixel

## 🏗️ Implementation Phases

### Phase 1: Base Panel Structure ⏱️ 2-3h
- [ ] Create `map_analysis_2d/ui/results_panel.py`
- [ ] Implement `ResultsPanel` as QDockWidget
- [ ] Integrate into main window
- [ ] Add View menu toggle

**Subtask Issue:** #TBD

### Phase 2: Overall Statistics Display ⏱️ 2-3h
- [ ] Create `map_analysis_2d/core/statistics.py`
- [ ] Implement statistics computation
- [ ] Connect to peak fitting completion
- [ ] Display formatted results

**Subtask Issue:** #TBD

### Phase 3: Pixel Details on Click ⏱️ 3-4h
- [ ] Implement map click event handler
- [ ] Convert click coordinates to pixel position
- [ ] Update panel with pixel details
- [ ] Add visual marker on map

**Subtask Issue:** #TBD

### Phase 4: Export & Copy Functionality ⏱️ 1-2h
- [ ] Export summary to .txt file
- [ ] Copy to clipboard
- [ ] Show spectrum for selected pixel
- [ ] Export pixel data

**Subtask Issue:** #TBD

### Phase 5: Visual Improvements ⏱️ 1-2h
- [ ] Color coding for fit quality
- [ ] Icons and better styling
- [ ] Mini histogram for distribution
- [ ] Tooltips

**Subtask Issue:** #TBD

### Phase 6: State Persistence ⏱️ 1h
- [ ] Save panel position/size to config
- [ ] Save visibility state
- [ ] Restore on startup

**Subtask Issue:** #TBD

### Phase 7: Testing & Documentation ⏱️ 2-3h
- [ ] Test with real data
- [ ] Test edge cases
- [ ] Performance testing on large maps
- [ ] Update README
- [ ] Add user guide

**Subtask Issue:** #TBD

## 🎨 UI Mockup

```text
┌─ Results Summary ──────────────────┐
│ Overall Statistics                  │
│ ├─ Total Area: 123,456.78          │
│ ├─ Fitted Pixels: 2,500/2,500      │
│ ├─ Success Rate: 98.2%             │
│ └─ Mean Area: 49.38 ± 12.45        │
│                                     │
│ [Export Summary] [Copy]             │
├─────────────────────────────────────┤
│ Selected Pixel Details              │
│ Position: (X=10, Y=15)              │
│                                     │
│ Integrated Intensities:             │
│ ├─ Peak 1: 45.23                   │
│ └─ Peak 2: 12.15                   │
│                                     │
│ [Show Spectrum] [Export]            │
└─────────────────────────────────────┘
```

## 📋 Minimum Viable Product (MVP)

**Phases 1-3 (8-10 hours):**
- Base panel structure
- Overall statistics display
- Pixel details on click

This provides core functionality without export features or visual polish.

## ✅ Success Criteria

### Functional
- [ ] Panel displays correctly after peak fitting
- [ ] Statistics are mathematically accurate
- [ ] Map clicks update pixel details immediately
- [ ] Export functions work correctly

### User Experience
- [ ] Intuitive and easy to read
- [ ] Fast response to clicks (<100ms)
- [ ] Doesn't obstruct map view
- [ ] Clear number formatting

### Code Quality
- [ ] Follows existing code style
- [ ] Properly documented
- [ ] No performance degradation
- [ ] Error handling implemented

## 🔗 Related

- **User Request:** [Original conversation]
- **Related Commit:** b67bebd - Add integrated intensity export and map session persistence
- **Implementation Plan:** [docs/features/results-panel-implementation-plan.md](results-panel-implementation-plan.md)

## 💡 Future Enhancements

Ideas for future versions:
- Multi-pixel selection (shift+click)
- Statistics filtering by R² threshold
- Compare different fitting runs
- Export to Excel with charts
- Historical tracking via database

---

## 📝 Implementation Notes

As you work on this feature:
1. Create separate issues for each phase as you start them
2. Link commits using `Refs #XXX` (this issue number)
3. Update this issue description if requirements change
4. Check off tasks as completed
5. Update the implementation plan document if needed

## 🏷️ Labels

`feature`, `enhancement`, `peak-fitting`, `ui`, `results-panel`
