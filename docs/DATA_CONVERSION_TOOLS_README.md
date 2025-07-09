# RamanLab Data Conversion Tools

A flexible, modular system for converting various data formats into RamanLab-compatible formats. The system is designed for easy extension with new conversion capabilities.

## 🎯 **Overview**

The Data Conversion Tools provide a unified interface for converting different types of spectroscopic data files into formats that can be processed by RamanLab's analysis modules. The system uses a modular architecture that makes it easy to add new conversion tools.

## 🚀 **How to Access**

### Method 1: From RamanLab Main Application (Recommended)
1. Open `raman_analysis_app_qt6.py` 
2. Go to the **Advanced** tab
3. Click **"Advanced Data Conversion"** button

### Method 2: Standalone Launcher
```bash
python launch_data_conversion.py
```

### Method 3: Direct Import
```python
from core.data_conversion_dialog import DataConversionDialog
dialog = DataConversionDialog()
dialog.exec()
```

## 🔧 **Available Converters**

### 1. Line Scan Splitter

**Purpose**: Split multi-spectrum line scan files into individual two-column files for batch processing.

**Input Format**: 
- Row 1: Wavenumbers (cm⁻¹)
- Column 1: Scan positions/numbers  
- Data Matrix: Intensity values

**Example Input** (`horiba_compact_linescan.txt`):
```
        201.303  203.15   204.999  ...
-10     316.5    329      328.5    ...
-9      306.5    308      303.5    ...
-8      298.2    301      295.8    ...
...
```

**Output**: Individual files with standard two-column format:
```
Wavenumber (cm-1)    Intensity
# Scan position: -10.0
201.303              316.500000
203.150              329.000000
204.999              328.500000
...
```

**Features**:
- ✅ Auto-detects delimiters (tab, comma, semicolon, space)
- ✅ Preview functionality before conversion
- ✅ Customizable output filenames
- ✅ Optional scan number inclusion in filenames
- ✅ Comprehensive error handling
- ✅ Progress tracking

**Usage Workflow**:
1. Select input line scan file
2. Choose output directory (auto-populated)
3. Set base filename
4. Click "Load & Preview" to validate data
5. Configure options as needed
6. Click "Convert" to create individual files

### 2. Wavelength to Raman Shift Converter

**Purpose**: Convert scattered wavelength data (nm) to Raman shift (cm⁻¹) for Raman spectroscopy analysis.

**Physics Background**: In Raman spectroscopy, the Raman shift represents the energy difference between the incident laser and the scattered light. This requires knowing the laser wavelength to calculate the shift from the laser frequency.

**Input Format**: 
- CSV or TXT files with scattered wavelength/intensity data
- Laser wavelength setting (e.g., 532 nm, 785 nm, 1064 nm)
- Automatic column detection (wavelength, intensity)
- Supports multiple file formats and delimiters

**Example Input** (`scattered_wavelength_data.csv`):
```
Wavelength,Intensity
544.228,886
544.246,855
544.263,864
...
```

**Output**: Tab-delimited TXT files with '_rs' suffix:
```
Raman Shift (cm-1)    Intensity
# Converted from scattered wavelength: scattered_wavelength_data.csv
# Laser wavelength: 532.0 nm
# Scattered wavelength range: 544.228 - 567.330 nm
# Raman shift range: 421.3 - 1165.8 cm⁻¹
421.321    886.000000
427.156    855.000000
...
```

**Features**:
- ✅ **Configurable laser wavelength**: Set your actual laser wavelength (defaults to 532 nm)
- ✅ **Multiple file processing**: Select one or more files for batch conversion
- ✅ **Automatic column detection**: Finds wavelength and intensity columns automatically
- ✅ **Flexible input**: Supports CSV, TXT, and DAT files
- ✅ **Physics accuracy**: Uses formula: Raman Shift = (1/λ_laser - 1/λ_scattered) × 10⁷
- ✅ **Preserve metadata**: Includes laser wavelength and conversion information in output headers
- ✅ **Smart naming**: Automatically appends '_rs' suffix to output files
- ✅ **Preview functionality**: View conversion results before processing

**Usage Workflow**:
1. Enter your laser wavelength in nm (e.g., 532, 785, 1064)
2. Click "Add Files" to select scattered wavelength data files
3. Optional: Set custom output directory (default: same as input files)
4. Click "Preview First File" to validate data and conversion
5. Click "Convert" to create Raman shift files with '_rs' suffix

**Conversion Formula**: `Raman Shift (cm⁻¹) = (1/λ_laser - 1/λ_scattered) × 10⁷`

**Note**: This is specifically for Raman spectroscopy data. If you need absolute wavenumber conversion (not Raman shift), please request a separate converter.

## 🏗️ **Architecture**

### Modular Design

The system uses a plugin-like architecture:

```
DataConversionDialog (Main Interface)
├── BaseConverter (Abstract Base Class)
├── LineScanConverter (Line Scan Implementation) 
├── [Future Converter 1]
├── [Future Converter 2]
└── [...]
```

### Key Components

1. **`DataConversionDialog`**: Main tabbed interface
2. **`BaseConverter`**: Abstract base class for all converters
3. **`LineScanConverter`**: Implementation for line scan splitting
4. **Converter Registry**: System for registering new converters

### File Structure
```
core/
├── data_conversion_dialog.py    # Main implementation
└── ...

docs/
├── DATA_CONVERSION_TOOLS_README.md    # This file
└── ...

launch_data_conversion.py    # Standalone launcher
```

## 🔌 **Adding New Converters**

The system is designed for easy extension. To add a new converter:

### Step 1: Create Converter Class
```python
class MyNewConverter(BaseConverter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "My Converter"
        self.description = "Description of what it does"
        self.input_extensions = ['.xyz', '.abc']
        self.output_extensions = ['.txt']
    
    def create_ui(self, parent_widget):
        # Create and return UI widget
        pass
    
    def validate_input(self):
        # Validate user inputs
        return True, "Validation passed"
    
    def convert(self, progress_callback=None):
        # Perform the conversion
        return True, "Success message", ["file1.txt", "file2.txt"]
```

### Step 2: Register Converter
```python
# In DataConversionDialog.__init__()
self.register_converter("my_converter", MyNewConverter(self))
```

### Required Methods
- **`create_ui()`**: Build the user interface
- **`validate_input()`**: Check if inputs are valid
- **`convert()`**: Perform the actual conversion
- **`get_info()`**: Return converter metadata (inherited)

## 🎨 **User Interface Features**

### Optimized Layout Design
- **Side-by-side Configuration**: Input and Output settings arranged horizontally for efficient use of screen space
- **Enhanced Preview Area**: File preview section expanded to utilize freed vertical space
- **Better Workflow**: Logical arrangement reduces scrolling and improves user experience
- **Responsive Design**: Components scale appropriately with window resizing

### Tabbed Interface
- Each converter gets its own tab
- Easy navigation between different tools
- Consistent layout and styling
- Optimized space utilization with side-by-side input/output sections

### Progress Tracking
- Real-time progress bars during conversion
- Status messages and logging
- Error reporting with detailed messages

### File Management
- Smart file browsing with appropriate filters
- Auto-population of output directories
- Enhanced preview functionality with larger viewing area
- Side-by-side input and output configuration for better workflow

### Results Display
- Success/failure notifications
- List of created files
- Detailed status messages

## 🔍 **Error Handling**

The system includes comprehensive error handling:

- **Input Validation**: Checks files, paths, and parameters
- **File Format Detection**: Robust delimiter and encoding detection
- **Progress Reporting**: Clear feedback during long operations
- **Graceful Failures**: Informative error messages with suggestions

## 📊 **Example Usage Scenarios**

### Scenario 1: Horiba Line Scan Processing
1. You have a `horiba_compact_linescan.txt` file with 50 spectra
2. Use Line Scan Splitter to create 50 individual `.txt` files
3. Import these files into RamanLab's batch processing system
4. Run peak fitting, background correction, etc. on all spectra

### Scenario 2: Custom Data Format (Future)
1. Implement a new converter for your specific instrument
2. Register it with the system
3. Users can access it through the same unified interface

## 🚀 **Integration with RamanLab**

### Batch Processing Workflow
```
Line Scan File → Data Conversion → Individual Files → Batch Processing → Analysis
```

### Compatible Modules
- ✅ Batch Peak Fitting (`peak_fitting_qt6.py`)
- ✅ Background Correction
- ✅ Peak Detection and Analysis
- ✅ Export to various formats

## 🛠️ **Technical Details**

### Dependencies
- **PySide6**: UI framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **pathlib**: File path handling

### File Formats Supported
- **Input**: `.txt`, `.dat`, `.csv` (delimiter auto-detection)
- **Output**: Two-column tab-delimited `.txt` files

### Performance
- Handles large files efficiently
- Progress reporting for long operations
- Memory-efficient processing

## 📋 **Testing**

### Manual Testing
```bash
# Test standalone dialog
python launch_data_conversion.py

# Test import
python -c "from core.data_conversion_dialog import DataConversionDialog; print('✅ Import successful')"
```

### Integration Testing
1. Launch main RamanLab application
2. Go to Advanced tab
3. Click "Advanced Data Conversion"
4. Test line scan conversion with sample data

## 🔮 **Future Enhancements**

### Planned Converters
- **FTIR Data Converter**: Convert FTIR formats to Raman-compatible
- **Energy to Wavenumber**: Convert photon energy (eV) to wavenumbers (cm⁻¹)
- **Frequency Converter**: Convert frequencies (Hz, THz) to wavenumbers
- **Format Standardizer**: Convert between different spectroscopic formats
- **Metadata Extractor**: Extract and preserve instrument metadata
- **Unit Normalizer**: Standardize intensity units across datasets

### Architecture Improvements
- **Plugin System**: Dynamic loading of converters
- **Configuration Saving**: Remember user preferences
- **Batch Mode**: Command-line interface for automation
- **Export Templates**: Customizable output formats

## 💡 **Tips and Best Practices**

### For Users
1. **Always preview** your data before conversion
2. **Check file paths** and permissions
3. **Use descriptive filenames** for easier organization
4. **Backup original data** before conversion

### For Developers
1. **Follow the BaseConverter interface** exactly
2. **Implement comprehensive error handling**
3. **Provide clear user feedback**
4. **Test with various file formats**
5. **Document your converter thoroughly**

## 🐛 **Troubleshooting**

### Common Issues

**"Could not import data conversion tools"**
- Ensure `core/data_conversion_dialog.py` exists
- Check Python path and module imports

**"File won't load"**
- Verify file format and encoding
- Check delimiter detection
- Ensure proper file permissions

**"Conversion failed"**
- Check output directory permissions
- Verify sufficient disk space
- Validate input data format

**"Preview shows wrong data"**
- Try different delimiter settings
- Check if file has headers or metadata rows
- Verify file encoding (UTF-8 recommended)

### Getting Help
1. Check error messages in the status panel
2. Try the standalone launcher for isolated testing
3. Verify file formats match expected input
4. Check file permissions and disk space

## 📜 **Version History**

- **v1.1** (Current): Added Wavelength to Raman Shift Converter
  - Scattered wavelength (nm) to Raman shift (cm⁻¹) conversion for Raman spectroscopy
  - Configurable laser wavelength input (defaults to 532 nm)
  - Multiple file batch processing
  - Automatic column detection
  - Physics accuracy with proper Raman shift formula implementation
- **v1.0**: Initial implementation with Line Scan Splitter
  - Modular architecture
  - Tabbed interface
  - Comprehensive error handling
  - Progress tracking
  - Integration with Advanced tab

---

🎉 **The Data Conversion Tools are now ready to handle your line scan data and are designed to grow with your future conversion needs!** 