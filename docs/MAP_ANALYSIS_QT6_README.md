# 2D Map Analysis - Qt6 Conversion

## 🎯 **Overview**
This is the Qt6 version of the comprehensive 2D Raman map analysis tool, converted from the original tkinter implementation. The tool provides sophisticated analysis capabilities for 2D Raman spectroscopy data including data loading, preprocessing, machine learning analysis, and visualization.

## 🔄 **Conversion Summary**

### **What Was Converted**
- **Original**: `map_analysis_2d.py` (7,431 lines, tkinter-based)
- **New**: `map_analysis_2d_qt6.py` (Qt6-based with modern UI)
- **Test Suite**: `test_map_analysis_qt6.py` (comprehensive testing)

### **Key Improvements in Qt6 Version**
1. **Modern UI Framework**: Converted from tkinter to PyQt6 for better performance and appearance
2. **Threaded Operations**: All heavy computations run in background threads to prevent UI freezing
3. **Enhanced User Experience**: Better layouts, groupboxes, and responsive design
4. **Improved Error Handling**: Comprehensive error messages and user feedback
5. **Menu System**: Professional menu bar with common actions
6. **Status Bar**: Real-time status updates during operations

## 🏗️ **Architecture Overview**

### **Core Components**

#### **1. Data Management Classes**
- `TemplateSpectrum`: Dataclass for individual template spectra
- `TemplateSpectraManager`: Manages collection of template spectra with preprocessing
- `SpectrumData`: Dataclass for individual map spectrum data
- `RamanMapData`: Main data container with analysis capabilities

#### **2. GUI Components**
- `TwoDMapAnalysisQt6`: Main Qt6 application window
- `MapAnalysisWorker`: QThread for background processing
- Multiple tabs for different analysis types

#### **3. Analysis Features**
- **Data Loading**: Directory-based map loading with pickle support for fast re-loading
- **Template Analysis**: Template spectrum fitting with NNLS/least-squares methods
- **PCA Analysis**: Principal Component Analysis with visualization
- **NMF Analysis**: Non-negative Matrix Factorization
- **ML Classification**: Random Forest classification (extensible framework)
- **Cosmic Ray Detection**: Sophisticated filtering algorithms
- **Report Generation**: PDF reports with comprehensive analysis results

## 🎨 **User Interface**

### **Layout Structure**
```
Main Window
├── Menu Bar (File, Analysis)
├── Status Bar
└── Main Panel (Horizontal Splitter)
    ├── Left Panel: Controls (300px width)
    │   ├── Data Loading
    │   ├── Feature Selection
    │   ├── Data Processing Options
    │   └── Template Analysis
    └── Right Panel: Visualization Tabs
        ├── Map View
        ├── PCA
        ├── NMF  
        ├── ML Classification
        └── Results
```

### **Control Panels**

#### **Data Loading Section**
- Load Map Data from directory
- Save/Load processed data as pickle files
- Fast loading for previously processed datasets

#### **Feature Selection**
- Integrated Intensity (with wavenumber range)
- Template Coefficient maps
- Template Residual maps
- Dominant Template maps
- Cosmic Ray detection maps

#### **Data Processing Options**
- Use processed vs. raw data toggle
- Cosmic ray filtering controls
- Sensitivity adjustment for cosmic ray detection

#### **Template Analysis**
- Load individual template spectra
- Load entire template directories
- Fit templates to map data
- Multiple fitting methods (NNLS, least squares)

### **Analysis Tabs**

#### **1. Map View Tab**
- Real-time 2D map visualization
- Multiple colormap options
- Interactive navigation with matplotlib toolbar
- Dynamic feature switching

#### **2. PCA Tab**
- Configurable number of components (2-100)
- Batch size control for large datasets
- Explained variance plots
- Principal component score plots
- Save/load PCA results

#### **3. NMF Tab**
- Configurable number of components (2-50)
- Component spectra visualization
- Mixing coefficient analysis
- Save/load NMF results

#### **4. ML Classification Tab**
- Random Forest parameter configuration
- Feature importance analysis
- Model training and evaluation
- Save/load trained models
- Export classification results

#### **5. Results Tab**
- Comprehensive visualization summary
- Combined analysis results
- PDF report generation
- Export capabilities

## ⚙️ **Technical Features**

### **Threading and Performance**
- All heavy computations use `QThread` for non-blocking operations
- Progress reporting for long-running tasks
- Efficient memory management for large datasets
- Batch processing capabilities

### **Data Format Support**
- **Input**: CSV, TXT files with various separators
- **Output**: Pickle files for processed data, PDF reports
- **Export**: Multiple formats for analysis results

### **Error Handling**
- Comprehensive exception handling throughout
- User-friendly error messages
- Graceful degradation for invalid data
- Logging system for debugging

### **Extensibility**
- Modular architecture for easy feature addition
- Plugin-style template management
- Configurable analysis parameters
- Extensible ML framework

## 📋 **Workflow Guide**

### **1. Data Loading**
1. Click "Load Map Data" to select directory containing spectrum files
2. Files should follow naming convention: `x_y.csv` or similar patterns
3. Or load previously processed data from pickle files

### **2. Template Analysis** (Optional)
1. Load template spectra (individual files or directories)
2. Configure fitting parameters (method, baseline, cosmic ray filtering)
3. Run "Fit Templates to Map"
4. Visualize template coefficient maps

### **3. PCA Analysis**
1. Switch to PCA tab
2. Set number of components and batch size
3. Click "Run PCA"
4. View explained variance and score plots
5. Save results if needed

### **4. NMF Analysis**
1. Switch to NMF tab
2. Configure parameters
3. Run analysis
4. Examine component spectra and mixing coefficients

### **5. Results and Reporting**
1. Go to Results tab
2. Click "Generate Visualizations" for summary plots
3. Click "Generate Report" for comprehensive PDF report

## 🔧 **Installation and Setup**

### **Dependencies**
```python
# Core Qt6 and GUI
PyQt6
matplotlib

# Scientific computing
numpy
pandas
scipy
scikit-learn
dask

# Data processing
joblib
pickle
tqdm

# Optional: for preprocessing module
ml_raman_map  # Custom preprocessing module
```

### **Running the Application**
```bash
# Test the application
python test_map_analysis_qt6.py

# Run the main application
python map_analysis_2d_qt6.py
```

## 🎯 **Key Differences from Tkinter Version**

### **Advantages of Qt6 Version**
1. **Performance**: Better handling of large datasets
2. **Threading**: Non-blocking UI with background processing
3. **Appearance**: Modern, professional look
4. **Responsiveness**: Real-time updates and feedback
5. **Menu System**: Standard application menus
6. **Error Handling**: Better user experience with errors

### **Maintained Features**
- All original analysis capabilities
- Same data formats and file structures
- Compatible with existing processed data files
- Identical analysis algorithms

### **Enhanced Features**
- Real-time status updates
- Progress indication for long operations
- Better parameter validation
- Improved visualization controls
- Professional reporting system

## 🧪 **Testing**

The `test_map_analysis_qt6.py` script provides comprehensive testing:

1. **Application Startup Test**: Verifies GUI initialization
2. **Data Structures Test**: Tests core data classes
3. **GUI Components Test**: Validates user interface elements

Run tests with:
```bash
python test_map_analysis_qt6.py
```

## 📈 **Performance Considerations**

### **Memory Management**
- Efficient numpy array operations
- Dask integration for large dataset processing
- Garbage collection optimization
- Memory usage monitoring

### **Processing Speed**
- Batch processing for large maps
- Parallel template fitting
- Optimized PCA/NMF algorithms
- Caching of processed results

### **User Experience**
- Non-blocking operations with progress feedback
- Responsive UI during computations
- Intelligent defaults for parameters
- Quick preview capabilities

## 🔮 **Future Enhancements**

### **Planned Features**
1. **Advanced ML Models**: Support for more classification algorithms
2. **Real-time Processing**: Live analysis during data acquisition
3. **3D Visualization**: Volume rendering for stack datasets
4. **Batch Processing**: Automated analysis of multiple maps
5. **Plugin System**: External analysis modules
6. **Cloud Integration**: Remote processing capabilities

### **UI Improvements**
1. **Dark Mode**: Theme selection
2. **Customizable Layouts**: User-configurable interface
3. **Keyboard Shortcuts**: Power user features
4. **Tooltips**: Better help system
5. **Undo/Redo**: Action history

## 📝 **Developer Notes**

### **Code Organization**
- Clean separation of data classes and GUI
- Consistent error handling patterns
- Comprehensive documentation
- Type hints throughout
- Logging integration

### **Extending the Application**
1. **Adding New Analysis Methods**: Extend the worker thread system
2. **New Visualization Types**: Add tabs to the main widget
3. **Custom Data Formats**: Extend the loading methods
4. **Additional Export Formats**: Enhance the reporting system

## 🎉 **Conclusion**

The Qt6 version of the 2D Map Analysis tool successfully modernizes the original tkinter application while maintaining all core functionality. The new architecture provides better performance, user experience, and extensibility for future enhancements.

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Status**: ✅ READY FOR USE  
**Files**: `map_analysis_2d_qt6.py`, `test_map_analysis_qt6.py` 