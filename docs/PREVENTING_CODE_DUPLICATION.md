# Preventing Code Duplication in RamanLab

## 🚨 **Why Duplication is Bad**
- **File bloat**: 13,804 → 9,344 lines (32% reduction in batch_peak_fitting_qt6.py)
- **Bug multiplication**: Fix a bug in one place, it still exists in the duplicate
- **Maintenance nightmare**: Changes need to be made in multiple places
- **Memory waste**: Larger files, slower loading
- **Merge conflicts**: Duplicated code creates complex git conflicts

## ✅ **Prevention Strategies**

### 1. **Module Structure**
```
core/
├── base_classes/           # Abstract base classes
│   ├── analysis_module.py  # Base class for all analysis modules
│   └── qt_dialog.py       # Base class for all Qt dialogs
├── mixins/                # Reusable functionality
│   ├── state_management.py
│   ├── file_operations.py
│   └── plotting_mixin.py
└── utils/                 # Shared utilities
    ├── data_processing.py
    ├── peak_fitting.py
    └── background_removal.py
```

### 2. **Base Classes Instead of Copy-Paste**
```python
# ✅ GOOD: Create a base class
class BaseAnalysisDialog(QDialog):
    """Base class for all analysis dialogs"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_state_management()
        self.setup_common_ui()
    
    def setup_state_management(self):
        """Common state management setup"""
        if STATE_MANAGER_AVAILABLE:
            register_module(self.module_name, self)
            self.add_state_management_ui()
    
    def create_session_tab(self):
        """Standard session tab for all modules"""
        # Implementation here
        pass

# ✅ GOOD: Inherit instead of duplicating
class BatchPeakFittingQt6(BaseAnalysisDialog):
    module_name = 'batch_peak_fitting'
    
    def __init__(self, parent, wavenumbers=None, intensities=None):
        super().__init__(parent)
        # Only module-specific code here
```

### 3. **Mixins for Shared Functionality**
```python
# ✅ GOOD: Create mixins for common behavior
class PlottingMixin:
    """Shared plotting functionality"""
    
    def setup_matplotlib_canvas(self):
        """Standard matplotlib setup"""
        pass
    
    def create_toolbar(self):
        """Standard toolbar creation"""
        pass

class FileOperationsMixin:
    """Shared file operations"""
    
    def add_files(self):
        """Standard file adding dialog"""
        pass
    
    def export_results(self):
        """Standard results export"""
        pass

# ✅ GOOD: Use multiple inheritance
class BatchPeakFittingQt6(BaseAnalysisDialog, PlottingMixin, FileOperationsMixin):
    pass
```

### 4. **Configuration-Driven UI**
```python
# ✅ GOOD: Define UI structure in config
TAB_CONFIG = {
    'file_selection': {
        'title': 'File Selection',
        'widgets': ['file_list', 'navigation', 'status']
    },
    'peak_controls': {
        'title': 'Peak Controls', 
        'subtabs': ['background', 'detection']
    },
    'session': {
        'title': '📋 Session',
        'auto_add': True  # Always add session tab
    }
}

def create_tabs_from_config(self, config):
    """Create tabs based on configuration"""
    for tab_id, tab_config in config.items():
        # Create tab based on config
        pass
```

## 🛠️ **Refactoring Tools**

### 1. **Extract Common Code**
Use our duplicate detector:
```bash
python check_duplicates.py
```

### 2. **Git Hooks** 
Pre-commit hook automatically checks for:
- Duplicate class definitions
- Duplicate methods within files
- Files over 5000 lines
- Similar code blocks

### 3. **IDE Settings**
Configure your IDE to:
- **Show warnings** for duplicate code
- **Limit file size** warnings at 3000 lines
- **Auto-format** to prevent whitespace differences hiding duplicates

## 📋 **Development Workflow**

### Before Adding New Features:
1. ✅ Check if similar functionality exists
2. ✅ Look for base classes to inherit from
3. ✅ Consider creating mixins for reusable parts
4. ✅ Run `python check_duplicates.py`

### Before Committing:
1. ✅ Pre-commit hook runs automatically
2. ✅ Review file sizes (`wc -l *.py`)
3. ✅ Check for TODO comments about refactoring

### Code Review Checklist:
- [ ] No duplicate classes
- [ ] No duplicate methods within files
- [ ] New functionality uses existing base classes
- [ ] Common patterns extracted to mixins
- [ ] Files under 3000 lines (soft limit)

## 🚀 **Migration Plan**

### Phase 1: Create Base Classes
- [ ] `BaseAnalysisDialog` with common Qt setup
- [ ] `BaseStateManagement` mixin
- [ ] `BasePlotting` mixin

### Phase 2: Refactor Existing Modules
- [ ] Batch Peak Fitting ✅ DONE
- [ ] Polarization Analyzer
- [ ] Cluster Analysis  
- [ ] Map Analysis

### Phase 3: Automated Enforcement
- [ ] Pre-commit hooks ✅ DONE
- [ ] CI/CD pipeline checks
- [ ] Weekly duplication reports

## 🎯 **Success Metrics**
- **File size reduction**: Target <3000 lines per file
- **Duplication detection**: Zero tolerance for duplicate classes/methods
- **Code reuse**: 80% of new dialogs inherit from base classes
- **Developer efficiency**: Faster development through reusable components

Remember: **"Don't Repeat Yourself (DRY)" - but also "Don't Repeat Others (DRO)"** 