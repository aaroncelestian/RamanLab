# RamanLab Application State Management Architecture

## Executive Summary

This document presents a comprehensive design for implementing application state management in RamanLab. The system will enable users to save and restore complete application sessions including window layouts, loaded data, analysis results, and UI preferences.

## Core Design Principles

1. **Modular Architecture**: Each component (main app, map analysis, polarization analyzer, etc.) manages its own state
2. **Progressive Enhancement**: State management can be added incrementally without breaking existing functionality
3. **Graceful Degradation**: System continues to work even if some state restoration fails
4. **User Control**: Users can choose when to save/restore sessions with automatic options available
5. **Performance Conscious**: Large data objects are handled efficiently with references and lazy loading

## System Architecture Overview

### 1. Central State Manager (`ApplicationStateManager`)

The heart of the system that coordinates all state operations:

```python
class ApplicationStateManager:
    """Central coordinator for application state management"""
    
    Key Responsibilities:
    - Register/unregister stateful modules
    - Coordinate save/restore operations across modules
    - Manage session files and directories
    - Handle auto-save and crash recovery
    - Provide progress feedback to users
```

### 2. Module Interface (`StatefulModule`)

Abstract base class that all stateful components implement:

```python
class StatefulModule(ABC):
    """Interface for components that can save/restore state"""
    
    Required Methods:
    - save_state() -> Dict[str, Any]
    - restore_state(state_data: Dict) -> bool
    - validate_state(state_data: Dict) -> bool
    - get_state_metadata() -> Dict
```

### 3. Storage Structure

```
~/.ramanlab/sessions/
├── current_session.json          # Active session metadata
├── auto_save/                     # Crash recovery saves
│   ├── auto_save_20240101_143022.json
│   └── data/                      # Large data objects
├── named_sessions/                # User-saved sessions
│   ├── my_analysis.json
│   ├── my_analysis_data/
│   └── important_results.json
└── templates/                     # Session templates
    └── default_workspace.json
```

## State Categories

### 1. Window & UI State
- **Window Geometry**: Position, size, maximized/minimized state
- **Layout Configuration**: Splitter positions, panel sizes
- **Tab States**: Active tabs, tab order
- **Toolbar States**: Visible toolbars, button states
- **View Settings**: Zoom levels, color schemes

### 2. Data State
- **Loaded Spectra**: Current spectrum data and file paths
- **Processed Data**: Background-subtracted, smoothed data
- **Analysis Results**: PCA, NMF, ML classification results
- **Template Libraries**: Loaded template spectra
- **Database State**: Search results, database connections

### 3. Analysis Parameters
- **Processing Settings**: Background subtraction parameters, smoothing settings
- **Peak Detection**: Threshold values, algorithm choices
- **Classification**: Model parameters, training data references
- **Visualization**: Plot settings, color maps, axis ranges

## Module-Specific Implementation

### 1. Main Analysis Window State Handler

```python
class MainAnalysisAppState(StatefulModule):
    """Handles state for the main RamanAnalysisAppQt6 window"""
    
    State Components:
    - Current spectrum data (wavenumbers, intensities)
    - Processing parameters (background, smoothing)
    - Window layout (splitters, tabs)
    - Database search history
    - UI preferences
    
    Implementation:
    def save_state(self) -> Dict[str, Any]:
        return {
            "window": self._save_window_state(),
            "spectrum": self._save_spectrum_data(),
            "processing": self._save_processing_params(),
            "database": self._save_database_state(),
            "ui": self._save_ui_preferences()
        }
```

### 2. Map Analysis State Handler

```python
class MapAnalysisState(StatefulModule):
    """Handles state for 2D Map Analysis window"""
    
    State Components:
    - Loaded map data (positions, intensities, wavenumbers)
    - PCA/NMF analysis results
    - Template fitting results
    - ML classification models and results
    - Visualization settings
    
    Special Considerations:
    - Large array data saved as separate files
    - Model objects serialized with pickle
    - Template libraries stored efficiently
```

### 3. Polarization Analyzer State Handler

```python
class PolarizationAnalyzerState(StatefulModule):
    """Handles state for polarization analysis tools"""
    
    State Components:
    - Crystal structure data
    - Orientation optimization results
    - 3D visualization settings
    - Analysis parameters
    - Calculation results
```

## Data Serialization Strategy

### 1. Lightweight Data (JSON)
- UI preferences and settings
- Analysis parameters
- File paths and metadata
- Session information

### 2. Medium Data (Pickle)
- Analysis results objects
- Template libraries
- Configuration objects
- Search results

### 3. Heavy Data (Optimized Binary)
- NumPy arrays → `.npy` files (compressed for large arrays)
- Pandas DataFrames → `.parquet` files
- Large datasets → HDF5 format
- Images → PNG with metadata

### 4. Smart Reference System

Instead of storing large data directly in JSON:
```json
{
    "intensities": {
        "type": "numpy_array",
        "file_path": "session_data/intensities_20240101.npy",
        "shape": [1000, 2048],
        "dtype": "float64",
        "size_mb": 15.6,
        "compressed": true
    }
}
```

## Session Management Features

### 1. Auto-Save System
- **Frequency**: Configurable (default: 5 minutes)
- **Trigger**: Data changes, analysis completion
- **Storage**: Separate auto-save directory
- **Cleanup**: Keep last 10 auto-saves
- **Background**: Non-blocking operation

### 2. Named Sessions
```python
# User creates named session
session_manager.save_session("my_important_analysis")

# Lists all saved sessions with metadata
sessions = session_manager.list_sessions()
# Returns: [{"name": "my_analysis", "timestamp": "2024-01-01T14:30:22", 
#           "modules": ["main_app", "map_analysis"], "size_mb": 45.2}]

# Load specific session
session_manager.load_session("my_important_analysis")
```

### 3. Crash Recovery
```python
def check_crash_recovery():
    # Detect if app crashed by checking:
    # 1. Auto-save files newer than clean shutdown marker
    # 2. Presence of .last_clean_shutdown file
    
    if crash_detected:
        show_recovery_dialog()
        # Options: Restore, Ignore, View Details
```

### 4. Session Browser
```python
class SessionBrowserDialog(QDialog):
    """Comprehensive session management interface"""
    
    Features:
    - List all sessions with previews
    - Show session contents and size
    - Rename, delete, export sessions
    - Import sessions from other machines
    - Set session descriptions and tags
```

## Implementation Timeline

### Phase 1: Foundation (2-3 weeks)
**Deliverables:**
- [ ] Core state management classes
- [ ] Abstract StatefulModule interface
- [ ] Basic JSON session storage
- [ ] Window state save/restore for main app
- [ ] Simple auto-save mechanism

**Key Files:**
- `core/state_management/__init__.py`
- `core/state_management/stateful_module.py`
- `core/state_management/state_manager.py`
- `core/state_management/window_state.py`

### Phase 2: Main Application Integration (2-3 weeks)
**Deliverables:**
- [ ] MainAnalysisAppState implementation
- [ ] Spectrum data state management
- [ ] Processing parameter persistence
- [ ] Database state handling
- [ ] Basic UI integration (Save/Load menu)

**Key Files:**
- `state_handlers/main_app_state.py`
- Integration in `raman_analysis_app_qt6.py`

### Phase 3: Advanced Modules (3-4 weeks)
**Deliverables:**
- [ ] Map analysis state management
- [ ] Polarization analyzer integration
- [ ] Template and model persistence
- [ ] Cross-module dependency handling
- [ ] Data optimization (compression, lazy loading)

**Key Files:**
- `state_handlers/map_analysis_state.py`
- `state_handlers/polarization_state.py`
- `core/state_management/data_state.py`

### Phase 4: Polish & Advanced Features (2-3 weeks)
**Deliverables:**
- [ ] Session browser interface
- [ ] Crash recovery system
- [ ] Performance optimization
- [ ] Comprehensive error handling
- [ ] User documentation

## Error Handling Strategy

### 1. Graceful Degradation
```python
def restore_application_state(session_data):
    successful_modules = []
    failed_modules = []
    
    for module_id, module_data in session_data["modules"].items():
        try:
            if module.restore_state(module_data):
                successful_modules.append(module_id)
            else:
                failed_modules.append(module_id)
        except Exception as e:
            log_error(f"Module {module_id} failed: {e}")
            failed_modules.append(module_id)
    
    # Show user summary of what was restored
    show_restoration_summary(successful_modules, failed_modules)
```

### 2. Validation & Compatibility
```python
def validate_session(session_data):
    # Check version compatibility
    if not is_version_compatible(session_data["version"]):
        return False
    
    # Validate module states
    for module_id, module_data in session_data["modules"].items():
        if not validate_module_state(module_data):
            return False
    
    # Check file dependencies
    if not check_file_dependencies(session_data):
        return False
    
    return True
```

### 3. Recovery Options
```python
class RecoveryDialog(QDialog):
    """Dialog for handling recovery situations"""
    
    Options:
    - "Restore All": Attempt to restore everything
    - "Restore Partial": Only restore validated modules
    - "Start Fresh": Ignore session and start clean
    - "Show Details": Display specific error information
```

## Performance Considerations

### 1. Lazy Loading
```python
class LazyDataReference:
    """Reference to data that's loaded only when accessed"""
    
    def __init__(self, file_path, metadata):
        self.file_path = file_path
        self.metadata = metadata
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            self._data = self.load_data()
        return self._data
```

### 2. Compression
- Automatic compression for arrays > 1MB
- ZIP-based session packages for sharing
- Configurable compression levels

### 3. Background Operations
```python
class StateWorker(QThread):
    """Background thread for state operations"""
    
    progress_updated = Signal(int)  # 0-100
    status_updated = Signal(str)
    
    def run(self):
        # Perform save/load operations without blocking UI
```

## User Experience Design

### 1. Transparent Operation
- State automatically saved on major operations
- Quick startup with session restoration
- Progress indicators for long operations
- Background saves don't interrupt work

### 2. User Control
```python
# Menu: Session > Save Session As...
def save_session_dialog():
    name, ok = QInputDialog.getText(self, "Save Session", "Session name:")
    if ok and name:
        self.state_manager.save_application_state(name)

# Menu: Session > Load Session...
def load_session_dialog():
    dialog = SessionBrowserDialog(self)
    if dialog.exec() == QDialog.Accepted:
        session_name = dialog.selected_session()
        self.state_manager.restore_application_state(session_name)
```

### 3. Recovery Experience
```python
def startup_recovery_check():
    if crash_recovery_available():
        reply = QMessageBox.question(
            None, "Session Recovery",
            "RamanLab detected a previous session that wasn't properly saved. "
            "Would you like to restore your work?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Save,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            restore_crash_session()
        elif reply == QMessageBox.Save:
            save_crash_session_as_named()
```

## Testing Strategy

### 1. Unit Tests
- Individual module state save/restore
- Serialization/deserialization accuracy
- State validation functions
- Error handling edge cases

### 2. Integration Tests
- Cross-module state consistency
- Session lifecycle (save → load → verify)
- Auto-save and recovery mechanisms
- UI integration points

### 3. Performance Tests
- Large dataset handling (>100MB maps)
- Memory usage during operations
- Startup time with various session sizes
- Concurrent save/load operations

### 4. User Acceptance Tests
- Real-world workflow scenarios
- Recovery from various failure modes
- UI usability and discoverability
- Cross-platform compatibility

## Implementation Benefits

### For Users:
1. **Productivity**: Never lose work, continue exactly where left off
2. **Confidence**: Automatic crash recovery provides safety net
3. **Collaboration**: Share complete analysis sessions with colleagues
4. **Organization**: Named sessions help organize different projects

### For Development:
1. **Modularity**: Each component manages its own state independently
2. **Extensibility**: New modules easily add state management
3. **Maintainability**: Clear interfaces and separation of concerns
4. **Testability**: Individual components can be tested in isolation

## Future Enhancements

### 1. Cloud Integration
- Sync sessions across devices
- Collaborative sessions
- Version control for sessions

### 2. Advanced Features
- Session templates for common workflows
- Automated session tagging and search
- Session analytics and usage patterns
- Integration with external data sources

### 3. Performance Optimizations
- Incremental saves (only changed data)
- Differential session storage
- Predictive loading based on usage patterns

This architecture provides a robust, scalable foundation for application state management that will significantly enhance the RamanLab user experience while maintaining system reliability and performance. 