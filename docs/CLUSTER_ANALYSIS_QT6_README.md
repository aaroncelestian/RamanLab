# Raman Cluster Analysis Qt6 Conversion

## Overview

The `raman_cluster_analysis_qt6.py` module is a complete Qt6 conversion of the original tkinter-based `raman_cluster_analysis.py` module. This provides comprehensive cluster analysis capabilities for Raman spectroscopy data.

## Features

### Data Import
- **Folder Import**: Batch import of spectrum files from a selected folder
- **Main App Import**: Import current spectrum from the main application
- **Database Import**: Import spectra directly from the database (placeholder)
- **Append Data**: Add additional spectra to existing datasets (placeholder)

### Clustering Analysis
- **Hierarchical Clustering**: Ward, complete, average, and single linkage methods
- **Distance Metrics**: Euclidean, cosine, and correlation distances
- **Configurable Parameters**: Number of clusters (2-20)
- **Feature Extraction**: Automated vibrational feature extraction from spectra

### Visualizations
- **Dendrogram**: Interactive hierarchical clustering dendrogram with customizable orientation and sample limits
- **Heatmap**: Cluster-sorted intensity heatmap with multiple colormaps and normalization options
- **Scatter Plot**: PCA-based 2D visualization of clusters (UMAP support if available)
- **PCA Components**: Analysis of principal components with variance explained

### Analysis Results
- **Cluster Statistics**: Detailed breakdown of cluster sizes and percentages
- **PCA Analysis**: Explained variance ratios for principal components
- **Export Capabilities**: Export results as text or CSV files

### Refinement Tools (Placeholders)
- **Interactive Refinement**: Split and merge clusters interactively
- **Undo/Redo**: Track and reverse refinement operations
- **Selection Tools**: Select clusters for refinement operations

## Integration

### Main Application
The cluster analysis is integrated into the main Qt6 application (`raman_analysis_app_qt6.py`) through the Advanced tab. Click the "Cluster Analysis" button to launch the module.

### Launch Function
```python
from raman_cluster_analysis_qt6 import launch_cluster_analysis
cluster_window = launch_cluster_analysis(parent, raman_app)
```

## Usage

### Basic Workflow
1. **Import Data**: Use the Import tab to load spectra from files or the main application
2. **Configure Clustering**: Set parameters in the Clustering tab
3. **Run Analysis**: Execute clustering algorithm
4. **Visualize Results**: Explore dendrograms, heatmaps, and scatter plots
5. **Export Results**: Save analysis results and visualizations

### File Formats Supported
- `.txt` files (space/tab delimited)
- `.csv` files (comma delimited)
- `.dat` files (space delimited)
- `.asc` files (ASCII format)

### Configuration Options
- **Wavenumber Column**: Specify which column contains wavenumber data (default: 0)
- **Intensity Column**: Specify which column contains intensity data (default: 1)

## Technical Details

### Dependencies
- PySide6 (Qt6 framework)
- NumPy (numerical arrays)
- SciPy (clustering algorithms)
- Scikit-learn (PCA, preprocessing)
- Matplotlib (plotting)
- Pandas (data handling)
- Seaborn (enhanced plotting)
- UMAP (optional, for UMAP visualization)

### Architecture
- **Main Class**: `RamanClusterAnalysisQt6` - Main window class inheriting from `QMainWindow`
- **Tab Structure**: Six main tabs for different functionality areas
- **Data Storage**: Centralized `cluster_data` dictionary for all analysis data
- **Visualization**: Separate matplotlib figures for each plot type

### Key Improvements over Tkinter Version
- **Modern UI**: Clean, modern Qt6 interface with better layout management
- **Responsive Design**: Better handling of window resizing and layout
- **Enhanced Controls**: Improved form layouts and control grouping
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Status Updates**: Progress bars and status messages for long operations

## Testing

### Direct Testing
Run the test script to launch the cluster analysis module with mock data:
```bash
python test_cluster_analysis_qt6.py
```

### Integration Testing
Launch the main application and access cluster analysis through the Advanced tab:
```bash
python raman_analysis_app_qt6.py
```

## Future Enhancements

### Planned Features
1. **Database Integration**: Complete implementation of database import functionality
2. **Advanced Refinement**: Full implementation of cluster splitting and merging
3. **Additional Clustering Methods**: K-means, spectral clustering, DBSCAN
4. **Feature Engineering**: More sophisticated feature extraction methods
5. **Interactive Plots**: Click-to-select functionality in visualizations
6. **Batch Processing**: Process multiple datasets simultaneously

### Performance Optimizations
- **Large Dataset Handling**: Optimize for datasets with thousands of spectra
- **Memory Management**: Better memory usage for large clustering operations
- **Background Processing**: Move heavy computations to background threads

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **File Format Issues**: Check column indices match your data format
3. **Memory Issues**: Reduce dataset size for very large collections
4. **Visualization Problems**: Update matplotlib and Qt6 to latest versions

### Debug Mode
Set environment variable for debugging:
```bash
export QT_LOGGING_RULES="*.debug=true"
```

## License

This module is part of the RamanLab project and follows the same licensing terms. 