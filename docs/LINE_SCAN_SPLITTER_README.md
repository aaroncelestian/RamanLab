# Line Scan Raman Spectrum Splitter

This tool solves the problem of importing line scan Raman spectroscopy data into the RamanLab batch processing system. Instead of modifying the existing batch processing code (which led to missing methods), this helper utility splits multi-spectrum line scan files into individual spectrum files that can be imported normally.

## Problem Solved

Line scan Raman data files typically have this format:
- **Row 1**: Raman shift values (wavenumbers) 
- **Column 1**: Scan numbers/positions
- **Data matrix**: Each column represents intensity data for one spectrum

The RamanLab batch processing system expects individual files with two columns (wavenumber, intensity). This splitter bridges that gap.

## Features

- **GUI Interface**: Easy-to-use graphical interface
- **Batch Processing**: Handles any number of spectra in a single file
- **Flexible Input**: Supports various delimiters (tab, comma, semicolon, space)
- **Robust Parsing**: Automatically detects file format and encoding
- **Customizable Output**: Configure base filename and include scan numbers
- **Preview Mode**: View file information before splitting
- **Error Handling**: Comprehensive error checking and user feedback

## Files Included

1. **`line_scan_splitter.py`** - Main module with core functionality and GUI
2. **`launch_line_scan_splitter.py`** - Simple launcher script for the GUI
3. **`LINE_SCAN_SPLITTER_README.md`** - This documentation file

## How to Use

### Method 1: GUI Interface (Recommended)

1. **Launch the GUI**:
   ```bash
   python launch_line_scan_splitter.py
   ```
   Or directly:
   ```bash
   python line_scan_splitter.py
   ```

2. **Load your line scan file**:
   - Click "Browse" next to "Input File"
   - Select your line scan data file (e.g., `PiC_C2_maps_10.txt`)
   - The output directory will auto-populate

3. **Preview the data**:
   - Click "Load & Preview" to see file information
   - Check the number of spectra and wavenumber range

4. **Configure options**:
   - Set base filename (default: "spectrum")
   - Choose whether to include scan numbers in filenames

5. **Split the file**:
   - Click "Split Files" to create individual spectrum files
   - Files will be saved in the specified output directory

### Method 2: Programmatic Use

```python
from line_scan_splitter import split_line_scan_file

# Split a line scan file
created_files = split_line_scan_file(
    input_file='sample_spectra/PiC_C2_maps_10.txt',
    output_directory='sample_spectra/split_spectra',
    base_filename='PiC_C2_maps_10',
    include_scan_number=True
)

print(f"Created {len(created_files)} individual spectrum files")
```

### Method 3: Class-based Approach

```python
from line_scan_splitter import LineScanSplitter

splitter = LineScanSplitter()

# Load file
if splitter.load_line_scan_file('your_file.txt'):
    # Get information
    info = splitter.get_info()
    print(f"Found {info['num_spectra']} spectra")
    
    # Split files
    created_files = splitter.split_to_individual_files(
        'output_directory',
        'base_name'
    )
```

## Example Workflow

1. **Start with line scan file** (`PiC_C2_maps_10.txt`):
   ```
   	201.303	203.15	204.999	...	(wavenumbers)
   -10	316.5	329	328.5	...	(spectrum 1)
   -9	306.5	308	303.5	...	(spectrum 2)
   ...
   ```

2. **Run the splitter** to create individual files:
   ```
   PiC_C2_maps_10_scan_-10.0.txt
   PiC_C2_maps_10_scan_-9.0.txt
   PiC_C2_maps_10_scan_-8.0.txt
   ...
   ```

3. **Each output file** has standard two-column format:
   ```
   Wavenumber (cm-1)	Intensity
   # Scan number: -10.0
   201.303	316.500000
   203.150	329.000000
   204.999	328.500000
   ...
   ```

4. **Import into batch processing**:
   - Open RamanLab batch processing
   - Use "Add Files" to select all the split spectrum files
   - Process normally with peak fitting, analysis, etc.

## Output File Format

Each split file contains:
- **Header line**: Column names and scan number comment
- **Two columns**: Wavenumber (cm⁻¹) and Intensity
- **Tab-delimited**: Compatible with existing import routines
- **Proper precision**: Wavenumbers to 3 decimal places, intensities to 6

## Benefits of This Approach

1. **No code modification**: Existing batch processing code remains unchanged
2. **No missing methods**: Avoids integration complexity that caused previous issues
3. **Reusable**: Works with any line scan format that follows the standard convention
4. **Maintainable**: Simple, focused tool that's easy to debug and extend
5. **Future-proof**: Can handle different line scan formats by updating just this tool

## Dependencies

- Python 3.6+
- numpy
- pandas  
- tkinter (usually included with Python)
- pathlib (Python 3.4+)

## Error Handling

The tool handles various error conditions:
- File encoding issues
- Different delimiter formats
- Malformed data
- Permission errors
- Invalid file paths

## Testing

The tool has been tested with:
- Tab-delimited line scan files
- Multiple spectra (22 spectra confirmed working)
- Large wavenumber ranges (201-1799 cm⁻¹)
- Various scan number formats

## Integration with RamanLab

After splitting:
1. Individual files are in the format expected by `batch_peak_fitting.py`
2. Files can be imported using the existing "Add Files" functionality
3. All existing features work normally (peak fitting, background subtraction, etc.)
4. Results can be analyzed using all existing tools

## Future Enhancements

Potential improvements for the future:
- Support for different line scan orientations
- Automatic detection of header formats
- Batch processing of multiple line scan files
- Integration into main RamanLab interface
- Support for other spectroscopy formats (IR, UV-Vis, etc.)

## Troubleshooting

**File won't load**: 
- Check file encoding (try UTF-8, ASCII)
- Verify data format (ensure row 1 = wavenumbers, column 1 = scan numbers)
- Check for special characters in file path

**Wrong number of spectra**:
- Verify first row contains wavenumbers only
- Check for empty rows in data

**Permission errors**:
- Ensure write access to output directory
- Try different output location

**Import errors in Python**:
- Install missing dependencies: `pip install numpy pandas`
- Check Python version (3.6+ required)

## Contact

For issues specific to this splitter tool, check the error messages and ensure your data follows the expected format. For general RamanLab issues, refer to the main documentation. 