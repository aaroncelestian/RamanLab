# Peak Fitting Code Unification Proposal

## Current Problem: Massive Code Duplication

The RamanLab codebase has **severe code duplication** across peak fitting tools:

### Duplicated Functions Found:
- `fit_peaks()` - **12+ implementations**
- `gaussian()` - **14+ implementations** 
- `lorentzian()` - **12+ implementations**
- `pseudo_voigt()` - **11+ implementations**
- `baseline_als()` - **12+ implementations**
- UI handlers (`update_lambda_label()`, `on_bg_method_changed()`) - **8+ implementations**

### Files with Duplicated Code:
- `batch_peak_fitting_qt6.py` (~9000 lines)
- `peak_fitting_qt6.py` (~3800 lines)
- `raman_polarization_analyzer_qt6.py`
- Multiple backup files with more duplications

## Solution: Use Existing Centralized Infrastructure

**Good news**: The codebase already has excellent centralized implementations that are being ignored!

### Existing Centralized Code:
- ✅ `core/peak_fitting.py` - Complete peak fitting module with proper typing
- ✅ `batch_peak_fitting/core/peak_fitter.py` - Modern QObject-based implementation
- ✅ `core/peak_fitting_ui.py` - **NEW** unified UI components (created today)

## Refactoring Plan

### Phase 1: Replace Duplicated Math Functions

**Current (Duplicated):**
```python
# In batch_peak_fitting_qt6.py (line 4585)
def gaussian(self, x, amp, cen, wid):
    return amp * np.exp(-((x - cen) / wid) ** 2)

# In peak_fitting_qt6.py (line 2011)  
def gaussian(self, x, amp, cen, wid):
    return amp * np.exp(-((x - cen) / wid) ** 2)

# In 12+ other files...
```

**Proposed (Unified):**
```python
# Use existing core/peak_fitting.py
from core.peak_fitting import PeakFitter

# In any tool:
peak_fitter = PeakFitter()
result = peak_fitter.gaussian(x, amp, cen, wid)
```

### Phase 2: Replace Duplicated UI Controls

**Current (Duplicated):**
```python
# Duplicated in every tool
def update_lambda_label(self):
    self.lambda_label.setText(f"λ: {self.lambda_slider.value():.0e}")

def on_bg_method_changed(self):
    method = self.bg_method_combo.currentText()
    # 50+ lines of identical code...
```

**Proposed (Unified):**
```python
# Use new core/peak_fitting_ui.py
from core.peak_fitting_ui import UnifiedPeakFittingWidget

class MyTool(QDialog):
    def setup_ui(self):
        # Replace 500+ lines of control creation with:
        self.peak_fitting_widget = UnifiedPeakFittingWidget()
        self.peak_fitting_widget.peaks_fitted.connect(self.on_peaks_fitted)
        layout.addWidget(self.peak_fitting_widget)
```

### Phase 3: Refactor Main Tools

#### Before: batch_peak_fitting_qt6.py Structure
```
BatchPeakFittingQt6 (9000+ lines)
├── Duplicated math functions (200+ lines)
├── Duplicated UI controls (800+ lines)  
├── Duplicated background methods (600+ lines)
├── Business logic (7000+ lines)
└── Duplicated plotting code (400+ lines)
```

#### After: Refactored Structure
```
BatchPeakFittingQt6 (3000 lines)  # 70% reduction!
├── UnifiedPeakFittingWidget (from core/)
├── Business logic (2500 lines)
└── Tool-specific features (500 lines)
```

## Implementation Timeline

### Week 1: Foundation
- ✅ Create `core/peak_fitting_ui.py` (DONE)
- ✅ Fix import issues and dependencies
- Test unified components in isolation

### Week 2: Refactor peak_fitting_qt6.py
- Replace duplicated math functions with `core.peak_fitting`
- Replace UI controls with `UnifiedPeakFittingWidget`
- Test functionality parity

### Week 3: Refactor batch_peak_fitting_qt6.py  
- Replace massive duplicated sections
- Integrate with existing batch processing logic
- Test batch operations

### Week 4: Other Tools
- Update `raman_polarization_analyzer_qt6.py`
- Update other tools using peak fitting
- Clean up backup files with duplications

## Benefits

### 1. **Massive Code Reduction**
- `batch_peak_fitting_qt6.py`: 9000 → 3000 lines (70% reduction)
- `peak_fitting_qt6.py`: 3800 → 1500 lines (60% reduction)
- **Total**: ~15,000 lines of duplicated code eliminated

### 2. **Consistent User Experience**
- Identical controls across all tools
- Same parameter ranges and behavior
- Unified keyboard shortcuts and interactions

### 3. **Better Maintainability**
- Fix bug once, fixes everywhere
- Add new peak model once, available everywhere  
- Consistent testing and validation

### 4. **Easier Development**
- New tools just import unified components
- No need to reimplement peak fitting from scratch
- Focus on business logic, not infrastructure

## Example: Before vs After

### Before (Current Code):
```python
# In batch_peak_fitting_qt6.py - 150 lines just for ALS controls!
def create_background_smoothing_tab(self):
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    # Background method combo
    self.bg_method_combo = QComboBox()
    self.bg_method_combo.addItems([...])  # 20 lines
    
    # Lambda controls
    self.lambda_slider = QSlider(Qt.Horizontal)
    self.lambda_slider.setRange(...)  # 10 lines
    self.lambda_label = QLabel(...)   # 5 lines
    
    # P controls  
    self.p_slider = QSlider(Qt.Horizontal)
    # ... 50 more lines of identical UI setup
    
    # Signal connections
    self.lambda_slider.valueChanged.connect(self.update_lambda_label)
    # ... 20 more lines of identical connections
    
    return tab
    
def update_lambda_label(self):
    # 10 lines of identical code
    
def on_bg_method_changed(self):
    # 50 lines of identical code
```

### After (Refactored Code):
```python
# In refactored batch_peak_fitting_qt6.py - 5 lines total!
def create_background_tab(self):
    from core.peak_fitting_ui import BackgroundControlsWidget
    
    self.bg_controls = BackgroundControlsWidget()
    self.bg_controls.parameters_changed.connect(self.on_background_changed)
    return self.bg_controls

def on_background_changed(self):
    # Just handle the business logic, UI is handled centrally
    self.recalculate_backgrounds()
```

## Recommendation

**This refactoring should be the #1 priority** for RamanLab code quality improvement. The amount of duplicated code is unsustainable and creates:

1. **User confusion** - Different tools behave differently
2. **Bug multiplication** - Same bug exists in 12+ places  
3. **Development inefficiency** - Changes must be made 12+ times
4. **Testing complexity** - Must test same functionality 12+ times

The existing `core/peak_fitting.py` infrastructure is already excellent - we just need to use it!

## Next Steps

1. **Immediate**: Test the new `core/peak_fitting_ui.py` components
2. **This week**: Start refactoring `peak_fitting_qt6.py` as proof of concept
3. **Next week**: Apply lessons learned to `batch_peak_fitting_qt6.py`
4. **Cleanup**: Remove all the backup files with duplicated code

This will make RamanLab significantly more maintainable and provide a much better user experience. 