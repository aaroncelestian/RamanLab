# RamanLab Mixture Analysis GUI

## Overview
The RamanLab Mixture Analysis GUI provides an advanced interface for spectral mixture analysis using iterative decomposition techniques. This tool integrates seamlessly with the RamanLab ecosystem and uses the shared matplotlib configuration for consistent visualization.

## Features

### Core Functionality
- **Advanced Spectral Decomposition**: Uses iterative methods to identify mineral components in mixed spectra
- **Database Integration**: Leverages the complete RamanLab mineral database for component matching
- **Intelligent Preprocessing**: Automatic background subtraction and peak detection
- **Statistical Analysis**: Comprehensive uncertainty estimation and quality metrics
- **Export Integration**: Results are automatically saved to the RamanLab batch database

### User Interface
- **Query Spectrum Loading**: Simple file loading with automatic format detection
- **Component Library Management**: Browse and load pure component spectra from the database
- **Real-time Progress**: Live updates during analysis with detailed status information
- **Interactive Results**: Detailed breakdown of identified components with confidence intervals
- **Visualization**: Professional plots using the RamanLab matplotlib configuration

### Integration with Search Results
🆕 **NEW**: The mixture analysis tool is now directly integrated with the search functionality in the main RamanLab application:

1. **Search Tab Integration**: After performing a database search in the main application, search results now include a "Mixture Analysis" button
2. **Automatic Query Loading**: When launched from search results, the mixture analysis tool automatically loads your query spectrum
3. **Guided Component Selection**: The tool provides helpful hints about using your search results as potential mixture components
4. **Seamless Workflow**: Search for similar minerals, then immediately analyze your spectrum as a potential mixture of those components

#### How to Use the Integration
1. Load a spectrum in the main RamanLab application
2. Go to the "Search" tab and perform a database search (Basic or Advanced)
3. In the search results window, click the "Mixture Analysis" button
4. The mixture analysis tool opens with your query spectrum already loaded
5. Use the database browser to load components from your search results as potential mixture components

## Technical Details

### Analysis Engine
- **Algorithm**: Enhanced iterative spectral decomposition with statistical validation
- **Database Format**: Full compatibility with RamanLab's HDF5 and Parquet database formats
- **Background Handling**: Multiple baseline correction methods (ALS, polynomial, spline)
- **Peak Detection**: Automated peak finding with manual override capabilities

### File Formats
- **Input**: Standard Raman formats (.txt, .csv, .spc, etc.)
- **Output**: Text reports, PNG plots, HDF5 batch results
- **Database**: RamanLab-compatible mineral spectrum database

### Dependencies
- **Qt Framework**: PySide6 for modern GUI
- **Scientific Computing**: NumPy, SciPy for analysis
- **Visualization**: Matplotlib with RamanLab styling
- **Data Handling**: H5py, Pandas for database operations

## Installation and Setup

### Prerequisites
```bash
pip install PySide6 numpy scipy matplotlib h5py pandas
```

### Launch Methods
1. **Standalone**: Run `python launch_mixture_analysis.py`
2. **From Main App**: Use the "Mixture Analysis" button in the Advanced tab
3. **From Search Results**: Click "Mixture Analysis" in search results window

### Configuration
The tool automatically uses the RamanLab matplotlib configuration file (`polarization_ui/matplotlib_config.py`) for consistent styling across the ecosystem.

## Usage Workflow

### Basic Analysis
1. **Load Query**: Import your unknown/mixed spectrum
2. **Browse Components**: Use the database browser to find potential pure components
3. **Add Components**: Load 2-6 pure component spectra
4. **Run Analysis**: Execute the mixture analysis algorithm
5. **Review Results**: Examine component percentages and fit quality
6. **Export**: Save results to the batch database or as text files

### Advanced Features
- **Parameter Tuning**: Adjust convergence criteria and iteration limits
- **Quality Control**: Review residuals and goodness-of-fit statistics
- **Batch Processing**: Analyze multiple spectra with the same component set
- **Uncertainty Estimation**: Statistical confidence intervals for all results

## Integration Architecture

The mixture analysis tool is built as a modular component that integrates with:
- **State Management**: Uses the universal state manager for session persistence
- **Database System**: Shared access to the mineral spectrum database
- **Visualization**: Consistent matplotlib styling and color schemes
- **Search System**: Direct integration with database search functionality
- **Export System**: Automatic saving to the batch results database

## Performance Notes

- **Database Size**: Optimized for databases with 1000+ mineral spectra
- **Analysis Speed**: Typical analysis completes in 30-60 seconds
- **Memory Usage**: Efficient handling of large spectral datasets
- **Concurrency**: Background processing to keep UI responsive

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Database Connection**: Verify RamanLab database is accessible
3. **File Loading**: Check spectrum file format compatibility
4. **Analysis Convergence**: Adjust parameters for difficult mixtures

### Error Messages
- **"No database found"**: Initialize RamanLab database first
- **"Convergence failed"**: Try different component combinations
- **"Import failed"**: Check file format and encoding

## Future Enhancements

- **Machine Learning**: Integration with ML-based classification
- **Real-time Analysis**: Live mixture analysis for streaming data
- **Advanced Algorithms**: Additional decomposition methods
- **Enhanced Visualization**: 3D component space plotting

---
*Part of the RamanLab Suite - Advanced Raman Spectroscopy Analysis Tools* 