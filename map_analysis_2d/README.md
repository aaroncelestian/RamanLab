# 2D Map Analysis - Modular Structure

This directory contains the modularized version of the 2D Map Analysis application for RamanLab.

## Directory Structure

```
map_analysis_2d/
├── __init__.py                    # Main package initialization
├── main.py                        # Entry point (50-100 lines)
├── README.md                      # This file
├── core/                          # Core functionality
│   ├── __init__.py
│   ├── spectrum_data.py           # Data classes (SpectrumData)
│   ├── cosmic_ray_detection.py    # Cosmic ray detection and removal
│   ├── template_management.py     # Template spectrum management
│   └── file_io.py                 # File I/O and RamanMapData class
├── analysis/                      # Analysis algorithms
│   ├── __init__.py
│   ├── pca_analysis.py           # PCA analysis (PCAAnalyzer class)
│   ├── nmf_analysis.py           # NMF analysis (NMFAnalyzer class) 
│   └── ml_classification.py      # ML classification (RandomForestAnalyzer, UnsupervisedAnalyzer)
├── ui/                           # User interface components
│   ├── __init__.py
│   ├── main_window.py            # Main application window (to be implemented)
│   ├── plotting_widgets.py       # Plotting widgets (to be implemented)
│   └── dialogs.py                # Dialog windows (to be implemented)
└── workers/                      # Background worker threads
    ├── __init__.py
    └── map_analysis_worker.py    # Worker thread for long operations
```

## Implemented Modules

### Core Modules (✅ Complete)

- **`spectrum_data.py`**: Contains the `SpectrumData` dataclass for holding spectrum information
- **`cosmic_ray_detection.py`**: Complete cosmic ray detection and removal system with `CosmicRayConfig` and `SimpleCosmicRayManager`
- **`template_management.py`**: Template spectrum management with `TemplateSpectrum` and `TemplateSpectraManager`
- **`file_io.py`**: File I/O operations and the main `RamanMapData` class for managing map data

### Workers (✅ Complete)

- **`map_analysis_worker.py`**: Background worker thread for time-consuming operations

### Analysis Modules (✅ Complete)

These modules contain the analysis algorithms extracted from the original monolithic file:

- **`pca_analysis.py`**: Principal Component Analysis with `PCAAnalyzer` class (156 lines)
- **`nmf_analysis.py`**: Non-negative Matrix Factorization with `NMFAnalyzer` class (282 lines)
- **`ml_classification.py`**: Machine learning classification with `RandomForestAnalyzer`, `UnsupervisedAnalyzer`, and `SpectrumLoader` classes (342 lines)

**Features implemented**:
- PCA with batch processing, data transformation, and save/load functionality
- NMF with stability improvements, memory management, and robust error handling
- Random Forest classification with feature transformation support
- K-means clustering with standardization and silhouette scoring
- Spectrum loading utilities with cosmic ray filtering support

### UI Modules (🚧 To be implemented)

These modules will contain the user interface components:

- **`main_window.py`**: Main application window (`TwoDMapAnalysisQt6` class)
- **`plotting_widgets.py`**: Matplotlib plotting widgets and visualization components
- **`dialogs.py`**: Dialog windows for various operations

## Usage

### Running the Application

```python
# Option 1: Run as module
python -m map_analysis_2d.main

# Option 2: Import and run
from map_analysis_2d import main
main()
```

### Using Core Components

```python
from map_analysis_2d.core import (
    SpectrumData,
    CosmicRayConfig, 
    SimpleCosmicRayManager,
    TemplateSpectraManager,
    RamanMapData
)

# Create cosmic ray detection configuration
config = CosmicRayConfig(enabled=True, absolute_threshold=1000)

# Initialize cosmic ray manager
cr_manager = SimpleCosmicRayManager(config)

# Load map data
map_data = RamanMapData("/path/to/data", cosmic_ray_config=config)

# Use template manager
template_manager = TemplateSpectraManager()
template_manager.load_template("/path/to/template.txt", "Template 1")
```

## Benefits of Modular Structure

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Reusability**: Core components can be used in other applications
4. **Scalability**: New features can be added without affecting existing code
5. **Collaboration**: Multiple developers can work on different modules simultaneously

## Migration Status

- ✅ **Core functionality**: Fully extracted and tested (664 + 240 + 300 + 40 lines)
- ✅ **Worker threads**: Extracted and functional (40 lines)
- ✅ **Analysis algorithms**: Fully extracted and tested (156 + 282 + 342 lines)
- ✅ **UI components**: Modular UI structure created with main window integration (400+ lines)
- 🚧 **Complete UI migration**: Need to extract remaining UI methods from original file
- 🚧 **Full integration**: Need to connect all analysis methods to new UI

## Testing

Run the test scripts to verify the modular structure:

```bash
# Test core modules
python test_modules.py

# Test analysis modules  
python test_analysis_modules.py
```

These will test that all modules can be imported and basic functionality works correctly.

## Next Steps

1. ✅ ~~Extract analysis algorithms from the original file~~ **COMPLETED**
2. ✅ ~~Extract UI components and create the main window~~ **COMPLETED**
3. Extract remaining UI methods (cosmic ray controls, template analysis, etc.)
4. Complete NMF and ML classification UI integration
5. Add comprehensive error handling and logging
6. Add unit tests for UI components
7. Create documentation for each module
8. Optimize imports and dependencies

## Step 2 Completion Summary

**✅ UI Components Successfully Extracted:**

- **Main Window Structure**: Created `MapAnalysisMainWindow` with modular design
- **Plotting System**: Matplotlib widgets with Qt6 integration for maps and spectra
- **Control Panels**: Dynamic panels that change based on analysis type
- **Base Widgets**: Reusable components for consistent UI design
- **Integration**: Working PCA analysis with visualization

**Key Achievements:**
- Replaced monolithic UI with modular, maintainable components
- Implemented tab-based interface with dynamic control panels
- Created working map visualization with spectrum interaction
- Integrated PCA analysis from Step 1 with new UI
- Established foundation for remaining analysis integrations

**Files Created:**
- `ui/main_window.py` (280 lines) - Main application window
- `ui/plotting_widgets.py` (183 lines) - Matplotlib plotting components  
- `ui/control_panels.py` (100+ lines) - Analysis control panels
- `ui/base_widgets.py` (250+ lines) - Reusable UI utilities

The modular UI foundation is now complete and ready for the remaining analysis integrations! 