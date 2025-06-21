# RamanLab Application State Management Architecture

## Overview

This document outlines the architecture for comprehensive application state management in RamanLab, enabling users to save and restore complete application sessions including window layouts, data, analysis results, and UI preferences.

## Core Architecture Components

### 1. Central State Manager (`core/state_management/`)

The `ApplicationStateManager` serves as the central coordinator for all state operations:

```python
class ApplicationStateManager:
    """Central coordinator for all application state management"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        self.settings = QSettings("RamanLab", "RamanLab")
        self.registered_modules = {}
        self.session_dir = Path.home() / ".ramanlab" / "sessions"
        self.auto_save_timer = QTimer()
        
    def register_module(self, module_id: str, module: StatefulModule)
    def save_application_state(self, session_name: Optional[str] = None) -> bool
    def restore_application_state(self, session_name: Optional[str] = None) -> bool
```

### 2. Module State Interface

All modules implement the `StatefulModule` abstract base class:

```python
class StatefulModule(ABC):
    @abstractmethod
    def save_state(self) -> Dict[str, Any]
    @abstractmethod
    def restore_state(self, state_data: Dict[str, Any]) -> bool
    @abstractmethod
    def validate_state(self, state_data: Dict[str, Any]) -> bool
    @abstractmethod
    def get_state_metadata(self) -> Dict[str, Any]
```

### 3. Storage Architecture

```
~/.ramanlab/sessions/
├── current_session.json          # Active session metadata
├── auto_save/                     # Automatic crash recovery
│   ├── auto_save_20240101_143022.json
│   └── data/                      # Auto-saved data objects
├── named_sessions/                # User-saved sessions
│   ├── my_analysis_session.json
│   ├── my_analysis_session_data/
│   └── backup_20240101.json
└── templates/                     # Session templates
    └── default_workspace.json
```

## State Categories

### 1. Window State
- Window geometry and position
- Splitter positions and panel layouts
- Tab widget states and active tabs
- Toolbar configurations
- Menu states and customizations

### 2. Data State
- Currently loaded spectra and file paths
- Processed data (background subtraction, smoothing)
- Analysis results (PCA, NMF, ML models)
- Template libraries and fitting results
- Database entries and search results

### 3. Analysis Parameters
- Peak detection settings
- Background subtraction parameters
- Smoothing configurations
- Classification thresholds
- Template fitting parameters

### 4. UI Preferences
- Plot zoom levels and axis ranges
- Color schemes and visualization settings
- Control panel configurations
- Filter settings and search parameters

## Module-Specific State Handlers

### 1. Main Analysis Window (`MainAnalysisAppState`)
```python
def save_state(self) -> Dict[str, Any]:
    return {
        "window": self.window_manager.save_window_state(self.app),
        "current_spectrum": self._save_spectrum_state(),
        "processing_parameters": self._save_processing_parameters(),
        "database_state": self._save_database_state(),
        "ui_preferences": self._save_ui_preferences()
    }
```

### 2. Map Analysis Window (`MapAnalysisState`)
```python
def save_state(self) -> Dict[str, Any]:
    return {
        "window": self.window_manager.save_window_state(self.window),
        "map_data": self._save_map_data(),
        "analysis_results": self._save_analysis_results(),
        "templates": self._save_templates(),
        "ui_settings": self._save_ui_settings()
    }
```

### 3. Polarization Analyzer (`PolarizationAnalyzerState`)
```python
def save_state(self) -> Dict[str, Any]:
    return {
        "window": self.window_state,
        "crystal_structure": self._save_crystal_structure(),
        "orientation_data": self._save_orientation_data(),
        "analysis_parameters": self._save_analysis_parameters(),
        "3d_visualization": self._save_3d_state()
    }
```

## Data Serialization Strategy

### 1. Lightweight Data (JSON)
- UI preferences and settings
- Analysis parameters
- File paths and metadata
- Session information

### 2. Medium Data (Pickle)
- Analysis results
- Template libraries
- Search results
- Configuration objects

### 3. Heavy Data (Binary Formats)
- Numpy arrays → `.npy` files
- Pandas DataFrames → `.parquet` files
- Large datasets → HDF5 format
- Images → PNG/JPEG with metadata

### 4. Smart Data References
```python
# Instead of storing large arrays directly in JSON:
{
    "intensities": {
        "type": "numpy_array",
        "file_path": "data/intensities_20240101.npy",
        "shape": [1000, 2048],
        "dtype": "float64",
        "size_mb": 15.6
    }
}
```

## Session Management Features

### 1. Auto-Save System
- Automatic saves every 5 minutes
- Crash recovery detection
- Background saving without UI blocking
- Cleanup of old auto-saves

### 2. Named Sessions
- User-defined session names
- Session metadata and descriptions
- Session templates for common workflows
- Import/export session capability

### 3. Session Browser
```python
class SessionBrowserDialog(QDialog):
    """Dialog for managing saved sessions"""
    
    Features:
    - List all saved sessions with metadata
    - Preview session contents
    - Load, rename, delete sessions
    - Export/import session files
```

### 4. Crash Recovery
```python
def check_for_crash_recovery(self) -> bool:
    """Detect if application crashed and offer recovery"""
    # Check for auto-save files newer than clean shutdown
    # Offer recovery options to user
    # Validate recovery data before restoration
```

## Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-2)
1. Create base state management classes
2. Implement module registration system
3. Basic window state save/restore
4. JSON-based session storage

### Phase 2: Main Application (Weeks 3-4)
1. Integrate main analysis window
2. Spectrum data state management
3. Processing parameter persistence
4. Basic UI integration

### Phase 3: Advanced Modules (Weeks 5-6)
1. Map analysis state management
2. Polarization analyzer integration
3. Template and model persistence
4. Cross-module data sharing

### Phase 4: Polish & Features (Weeks 7-8)
1. Auto-save and crash recovery
2. Session browser and management
3. Performance optimization
4. Comprehensive testing

## Error Handling & Validation

### 1. State Validation
```python
class StateValidator:
    @staticmethod
    def validate_session(session_data: Dict[str, Any]) -> bool:
        # Check required fields
        # Validate module compatibility
        # Verify data integrity
        # Check file dependencies
```

### 2. Graceful Degradation
- Partial state recovery when some modules fail
- Fallback to default values for missing data
- User notification of recovery issues
- Option to continue with partial state

### 3. Backup Strategies
- Multiple auto-save versions
- Session file versioning
- Automatic backup before major operations
- Manual backup/restore options

## Performance Considerations

### 1. Lazy Loading
- Load large datasets only when needed
- Reference-based data management
- Progressive loading for UI responsiveness

### 2. Compression
- Compress large arrays automatically
- ZIP-based session packages
- Selective compression based on data size

### 3. Memory Management
- Unload unused data objects
- Configurable memory limits
- Garbage collection integration

## User Experience Design

### 1. Transparent Operation
- Automatic state saving without user intervention
- Quick startup with session restoration
- Background operations don't block UI

### 2. User Control
- Manual save/load options
- Session management interface
- Customizable auto-save intervals

### 3. Recovery Experience
- Clear recovery options after crashes
- Preview of recoverable data
- User choice in recovery strategies

## Testing Strategy

### 1. Unit Tests
- Individual module state operations
- Serialization/deserialization
- Validation functions
- Error handling

### 2. Integration Tests
- Cross-module state consistency
- Session lifecycle management
- Recovery mechanisms
- UI integration

### 3. Performance Tests
- Large dataset handling
- Memory usage monitoring
- Startup/shutdown timing
- Concurrent operation handling

This architecture provides a robust, scalable foundation for application state management that enhances user productivity while maintaining system reliability and performance. 