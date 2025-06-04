# Batch Peak Fitting Qt6 Documentation

## Overview

The Batch Peak Fitting module (`batch_peak_fitting_qt6.py`) provides a comprehensive Qt6-based interface for processing multiple Raman spectra with consistent peak fitting parameters. This tool is essential for analyzing series of measurements or samples where you need to track how peak characteristics change across different conditions.

## Features

### 🔄 **Batch Processing Capabilities**
- Process multiple spectrum files with consistent parameters
- Use one spectrum as a reference template for all others
- Automatic peak position tracking across spectra
- Progress monitoring and error handling
- Export results to CSV format

### 📊 **Advanced Visualization**
- **Current Spectrum View**: Interactive display of individual spectra with peak fits
- **Trends Analysis**: Track how peak parameters (position, amplitude, width) change across spectra
- **Waterfall Plot**: Stacked view of multiple spectra for easy comparison
- Real-time plotting with matplotlib integration

### 🔍 **Peak Analysis Tools**
- Asymmetric Least Squares (ALS) background subtraction
- Interactive peak detection with adjustable parameters
- Multiple peak models: Gaussian, Lorentzian, Pseudo-Voigt, Asymmetric Voigt
- Individual peak parameter analysis and visualization

### 📁 **File Management**
- Support for multiple file formats (.txt, .csv)
- Robust file loading with multiple parsing strategies
- Drag-and-drop file selection
- Navigation controls for browsing loaded spectra

## Integration with RamanLab

The batch peak fitting functionality is seamlessly integrated into the main RamanLab application through the Peak Fitting module:

1. **Access Point**: Available in the Advanced Peak Fitting window
2. **Location**: Deconvolution tab → Batch Processing section
3. **Launch Button**: "🔄 Launch Batch Peak Fitting"

## User Interface

### Main Window Layout

The batch peak fitting window is organized into two main areas:

#### Left Panel - Control Tabs
1. **File Selection**
   - Add/remove spectrum files
   - File navigation controls
   - Current file status display

2. **Peak Controls**
   - Background subtraction parameters
   - Peak detection settings (height, distance thresholds)
   - Peak model selection and fitting

3. **Batch Processing**
   - Reference spectrum management
   - Batch execution controls
   - Progress log and status updates

4. **Results**
   - Peak visibility controls for trends
   - Display options (boundaries, grid lines)
   - Export functionality

#### Right Panel - Visualization Tabs
1. **Current Spectrum**
   - Main spectrum plot with background and peaks
   - Residuals plot showing fit quality
   - Individual peak components

2. **Trends**
   - Peak position trends across spectra
   - Amplitude and width variations
   - R² values for fit quality assessment

3. **Waterfall**
   - Stacked spectrum display
   - Adjustable spacing and skip parameters
   - Overview of entire dataset

## Workflow

### Step 1: File Management
```
1. Click "Add Files" to load spectrum files
2. Double-click files in the list to view them
3. Use navigation buttons to browse between spectra
4. Remove unwanted files using "Remove Selected"
```

### Step 2: Peak Fitting Setup
```
1. Select a representative spectrum
2. Adjust background subtraction parameters (λ, p)
3. Click "Subtract Background"
4. Set peak detection parameters:
   - Height threshold (% of max intensity)
   - Minimum distance between peaks
5. Click "Find Peaks" to detect peaks automatically
6. Select appropriate peak model (Gaussian/Lorentzian)
7. Click "Fit Peaks" to perform the fit
```

### Step 3: Batch Processing
```
1. With a well-fitted spectrum displayed, click "Set as Reference"
2. This saves the peak positions and background parameters
3. Click "Apply to All" to process all loaded files
4. Monitor progress in the log window
5. View results in the Trends and Waterfall tabs
```

### Step 4: Results Analysis
```
1. Use peak visibility controls to select which peaks to analyze
2. View trends plots to see parameter variations
3. Check R² values for fit quality
4. Export results to CSV for further analysis
```

## Technical Details

### File Format Support
The module supports various file formats through multiple parsing strategies:

- **Strategy 1**: numpy.loadtxt for simple space/tab-delimited files
- **Strategy 2**: pandas CSV reader with automatic delimiter detection
- **Strategy 3**: Manual parsing for complex or non-standard formats

### Peak Fitting Models

#### Gaussian
```python
f(x) = amp * exp(-((x - center) / width)²)
```

#### Lorentzian
```python
f(x) = amp / (1 + ((x - center) / width)²)
```

#### Multi-Peak Model
Combines multiple individual peaks:
```python
f(x) = Σ peak_i(x)
```

### Background Subtraction (ALS)
The Asymmetric Least Squares method uses:
- **λ (lambda)**: Smoothness parameter (1e3 - 1e8)
- **p**: Asymmetry parameter (0.001 - 0.1)
- **Iterations**: Number of optimization iterations (5-50)

### Export Format
Results are exported as CSV files with columns:
- Spectrum index and filename
- Fit success status
- R² values
- Peak parameters (position, amplitude, width) for each peak
- Error messages for failed fits

## Error Handling

The module includes comprehensive error handling:

1. **File Loading Errors**: Gracefully handles corrupted or incompatible files
2. **Fitting Failures**: Continues processing other files if individual fits fail
3. **Parameter Validation**: Ensures realistic parameter bounds
4. **Progress Monitoring**: Allows user to stop batch processing if needed

## Performance Considerations

- **Memory Usage**: Efficiently manages large datasets by processing files sequentially
- **Processing Speed**: Optimized curve fitting with reasonable iteration limits
- **UI Responsiveness**: Progress updates and ability to cancel long operations
- **Visualization**: Efficient matplotlib rendering with reasonable data limits

## Dependencies

### Required Packages
- PySide6 (Qt6 GUI framework)
- numpy (numerical computations)
- scipy (optimization and signal processing)
- matplotlib (plotting and visualization)
- pandas (data handling and export)

### Optional Packages
- chardet (encoding detection for problematic files)

## Examples

### Basic Batch Processing Workflow
```python
# Launch from command line or integrate into larger application
from batch_peak_fitting_qt6 import launch_batch_peak_fitting

# Launch with optional initial data
launch_batch_peak_fitting(parent_window, wavenumbers, intensities)
```

### Programmatic Access
```python
from batch_peak_fitting_qt6 import BatchPeakFittingQt6

# Create dialog directly
dialog = BatchPeakFittingQt6(parent, wavenumbers, intensities)
dialog.exec()
```

## Best Practices

### Data Preparation
1. Ensure all spectra have similar wavenumber ranges
2. Apply consistent preprocessing (e.g., cosmic ray removal) before batch fitting
3. Choose a high-quality reference spectrum with clear, well-defined peaks

### Parameter Selection
1. Start with conservative peak detection parameters
2. Use the same background subtraction parameters for all spectra
3. Choose appropriate peak models based on your spectral features

### Quality Control
1. Monitor R² values in the trends plot
2. Check for systematic shifts in peak positions
3. Investigate outliers or failed fits individually

### Results Analysis
1. Export data early and often during analysis
2. Use external tools (R, Python, Excel) for advanced statistical analysis
3. Validate trends with known experimental conditions

## Troubleshooting

### Common Issues

**Problem**: Peaks not detected properly
- **Solution**: Adjust height and distance thresholds in Peak Controls

**Problem**: Fitting fails for many spectra
- **Solution**: Choose a better reference spectrum or adjust fitting parameters

**Problem**: Trends show unexpected behavior
- **Solution**: Check for systematic experimental effects or preprocessing issues

**Problem**: Export fails
- **Solution**: Ensure write permissions and sufficient disk space

### Performance Issues

**Large datasets (>100 spectra)**:
- Process in smaller batches
- Use skip parameters in waterfall plots
- Export results frequently

**Memory issues**:
- Close other applications
- Process files sequentially rather than loading all at once

## Future Enhancements

Potential future additions:
1. **Advanced Statistical Analysis**: Built-in statistical tests for trends
2. **Automated Peak Assignment**: Machine learning-based peak identification
3. **Advanced Export Options**: Support for other formats (HDF5, MATLAB)
4. **Cloud Processing**: Support for processing large datasets in the cloud
5. **Real-time Processing**: Monitor folders for new files and process automatically

## Support and Contact

For technical support or feature requests, please refer to the main RamanLab documentation or contact the development team.

---

*This documentation is for Batch Peak Fitting Qt6 module version 1.0*
*Last updated: December 2024* 