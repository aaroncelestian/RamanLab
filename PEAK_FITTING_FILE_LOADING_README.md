# Peak Fitting Qt6 - File Loading Capability

The Peak Fitting Qt6 app now includes the ability to open spectrum files directly without needing to go back to the main RamanLab application.

## New Features

### 📁 File Menu
- **Open Spectrum File** (Ctrl+O): Load spectrum files in various formats
- **Recent Files**: Quick access to recently opened files (placeholder for future enhancement)
- **Export Results** (Ctrl+S): Export analysis results to CSV/TXT
- **Close** (Ctrl+W): Close the application

### 📊 Analysis Menu
- **Reset Analysis**: Clear all analysis data
- **Clear All Peaks**: Remove all detected/manual peaks
- **Apply Background Subtraction**: Apply current background settings
- **Fit Peaks** (Ctrl+F): Perform peak fitting

### 📈 Status Bar
Displays real-time information:
- Number of data points
- Wavenumber range
- Intensity range
- Number of detected peaks
- File size (when loaded from file)

### 🏷️ Window Title
Updates to show:
- Current filename (when loaded from file)
- Number of data points (when passed from main app)
- "No data loaded" status

## Supported File Formats

- **.txt** - Tab or space delimited text files
- **.csv** - Comma separated values
- **.dat** - Data files
- **.asc** - ASCII files
- **.spc** - Spectrum files

### File Format Requirements
- **Two-column format**: wavenumber, intensity
- **Delimiters**: Tab, comma, space, or semicolon
- **Headers**: Automatically detected and skipped
- **Comments**: Lines starting with #, %, or ; are ignored
- **Data**: Must be numeric values

## Usage

### 1. Launch as Standalone Application
```bash
# Method 1: Direct execution
python peak_fitting_qt6.py

# Method 2: Using test script
python test_peak_fitting_file_loading.py
```

### 2. Launch from Main RamanLab App
```python
from peak_fitting_qt6 import launch_spectral_deconvolution

# With existing data
launch_spectral_deconvolution(parent, wavenumbers, intensities)

# Without data (will show file loading interface)
launch_spectral_deconvolution(parent)
```

### 3. Loading Files
1. Use **File → Open Spectrum File** or press **Ctrl+O**
2. Select your spectrum file from the dialog
3. The app will automatically:
   - Load and validate the data
   - Update the window title and status bar
   - Reset any previous analysis state
   - Display the spectrum

### 4. Workflow Example
1. Launch the app (standalone or from main app)
2. Open a spectrum file (**File → Open** or **Ctrl+O**)
3. Adjust background subtraction parameters
4. Detect peaks (automatic) or select manually (interactive mode)
5. Fit peaks using your preferred model
6. Export results (**File → Export Results** or **Ctrl+S**)

## Error Handling

### File Loading Errors
- **File not found**: Clear error message with file path
- **Invalid format**: Details about what went wrong
- **No valid data**: Information about data requirements
- **Permission errors**: File access issues

### Missing Dependencies
- If file loading utilities are not available, the Open File menu item is disabled
- Clear messages indicate missing dependencies

## Testing

Use the included test script to explore features:

```bash
python test_peak_fitting_file_loading.py
```

Options:
1. **Standalone app** - Launch with no data for file loading
2. **Sample data** - Launch with generated test spectrum
3. **File formats** - Show supported format information
4. **All tests** - Combined information and launch options

## Integration with RamanLab

The file loading capability is fully integrated with existing RamanLab functionality:

- **Backward Compatible**: Existing code continues to work unchanged
- **File Utilities**: Uses RamanLab's existing file loading infrastructure
- **UI Consistency**: Matches RamanLab's UI patterns and styling
- **Error Handling**: Consistent with RamanLab's error reporting

## Dependencies

- **PySide6**: Qt6 Python bindings for UI
- **numpy**: Numerical computations
- **scipy**: Scientific computing (peak fitting, signal processing)
- **pandas**: Data manipulation (for export functionality)
- **matplotlib**: Plotting and visualization
- **RamanLab utils**: File loading utilities

## Example Files

### Sample Text File Format
```
# Raman spectrum data
# Wavenumber (cm-1)    Intensity
200.0    12.3
201.0    13.1
202.0    14.5
...
```

### Sample CSV Format
```
Wavenumber,Intensity
200.0,12.3
201.0,13.1
202.0,14.5
...
```

## Future Enhancements

- **Recent Files**: Remember and quickly access recently opened files
- **File Validation**: Enhanced file format detection and validation
- **Batch Loading**: Load multiple spectrum files at once
- **Save Session**: Save and restore complete analysis sessions
- **Custom File Readers**: Support for additional spectrum file formats

## Troubleshooting

### Common Issues

1. **"File loading not available"**
   - Install missing dependencies: `pip install pandas numpy scipy`
   - Check that `utils/file_loaders.py` exists

2. **"No valid data found"**
   - Ensure file has two numeric columns
   - Check that data isn't all headers/comments
   - Verify delimiter is standard (tab, comma, space)

3. **Import errors**
   - Run from the RamanLab root directory
   - Check Python path includes RamanLab modules

### Performance Tips

- **Large files**: Files with >100,000 points may load slowly
- **Memory usage**: Peak fitting scales with number of peaks
- **Plot updates**: Disable grid/legend for faster rendering with large datasets

---

*This feature enhances the Peak Fitting Qt6 app's usability by allowing direct file access, making it more convenient for users who want to analyze spectrum files without launching the full RamanLab application.* 