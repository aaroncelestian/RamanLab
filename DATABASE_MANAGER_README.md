# RamanLab Database Manager GUI

A comprehensive GUI tool for managing RamanLab database files, designed to help users validate, repair, and maintain their Raman spectroscopy databases.

## Features

### 🗄️ Database File Management
- **File Selection**: Easy browsing and selection of database files
- **Auto-Detection**: Automatically finds database files in the current directory
- **Path Validation**: Ensures database files are in the correct location
- **Status Indicators**: Visual feedback on database file status

### 🔍 Database Validation
- **Entry Counting**: Reports the number of entries in each database
- **Format Validation**: Checks if databases are in the correct format
- **Integrity Checks**: Validates database structure and readability
- **Sample Entry Display**: Shows sample entries from each database
- **Browser Compatibility**: Tests if database_browser_qt6.py can read the files

### 🔧 Database Repair
- **Automatic Backup**: Creates backups before making changes
- **Path Correction**: Fixes database path references in Python files
- **File Movement**: Moves databases to the correct location
- **Backup Restoration**: Restores databases from backup copies

### ⚙️ Additional Features
- **Multi-threaded Validation**: Non-blocking database validation
- **Comprehensive Logging**: Detailed logs of all operations
- **Progress Tracking**: Real-time progress indicators
- **Settings Management**: Configurable validation parameters

## Required Database Files

The tool manages these critical RamanLab database files:

### RamanLab_Database_20250602.sqlite
- **Purpose**: Main Raman spectra database (SQLite format)
- **Content**: Experimental and reference Raman spectra
- **Size**: Typically 200-500 MB
- **Structure**: SQLite database with tables including:
  - `spectra` table with spectrum data
  - Wavenumber arrays and intensity arrays
  - Metadata (mineral names, formulas, classifications, etc.)

### mineral_modes.pkl
- **Purpose**: Mineral vibrational modes database (PKL format)
- **Content**: Calculated Raman modes for various minerals
- **Size**: Typically 6-10 MB
- **Structure**: Dictionary with mineral data including:
  - Crystal system information
  - Vibrational modes with frequencies and symmetries
  - Point group and space group data

## Installation & Usage

### Prerequisites
```bash
pip install PySide6 numpy matplotlib scipy pandas
```

### Running the Database Manager

#### Option 1: Direct execution
```bash
python database_manager_gui.py
```

#### Option 2: Using the launcher
```bash
python launch_database_manager.py
```

### Using the GUI

#### 1. Database Files Tab
- Click "Browse..." to select each database file
- Use "Auto-Detect Databases" to find files automatically
- Click "Validate All Databases" for quick validation
- Use "Fix Database Paths" to correct path issues

#### 2. Validation Tab
- Click "Run Full Validation" for comprehensive checks
- Click "Test Database Browser" to verify compatibility
- View detailed validation results in the text area

#### 3. Repair Tab
- Enable "Create backup before repair" (recommended)
- Use "Create Backup" to manually backup databases
- Use "Move Databases" to relocate files to correct directory
- Use "Restore Backup" to recover from previous backups

#### 4. Settings & Help Tab
- Configure RamanLab directory location
- Adjust validation parameters
- View help documentation

## Common Issues & Solutions

### Database Not Found Errors
**Problem**: Application can't find database files
**Solution**: 
1. Use "Auto-Detect Databases" to locate files
2. Manually browse for files if auto-detection fails
3. Use "Move Databases" to put files in correct location

### Path Reference Issues
**Problem**: Python files have incorrect database paths
**Solution**: 
1. Click "Fix Database Paths" to update path references
2. The tool automatically updates these files:
   - `mineral_database.py`
   - `raman_spectra.py`
   - `raman_polarization_analyzer.py`

### Corrupted Database Files
**Problem**: Database files are unreadable or corrupted
**Solution**:
1. Check validation results for specific error messages
2. Restore from backup if available
3. Contact support for database recovery assistance

### Browser Compatibility Issues
**Problem**: database_browser_qt6.py can't read databases
**Solution**:
1. Run "Test Database Browser" to identify issues
2. Ensure all required dependencies are installed
3. Verify database file permissions

## Validation Results Interpretation

### Status Indicators
- ✅ **Green checkmark**: Database is valid and readable
- ❌ **Red X**: Database has issues or is not found
- ❓ **Question mark**: Database not yet validated

### Validation Report Fields
- **File exists**: Whether the file is found at the specified path
- **Readable**: Whether the file can be opened and read
- **Format valid**: Whether the file contains expected data structure
- **Entry count**: Number of entries/records in the database
- **File size**: Size of the database file in bytes
- **Sample entries**: First few entry names for verification

## Backup Strategy

The tool implements a comprehensive backup strategy:

### Automatic Backups
- Created before any file modifications
- Timestamped directory names: `backup_YYYYMMDD_HHMMSS`
- Includes all selected database files

### Manual Backups
- Create backups at any time using "Create Backup" button
- Stored in timestamped directories for easy identification

### Backup Restoration
- Select any backup directory to restore files
- Files are copied (not moved) to preserve backups
- Updates GUI to reflect restored file locations

## Troubleshooting

### Import Errors
If you get PySide6 import errors:
```bash
pip install PySide6
```

### Permission Errors
If you can't write to the RamanLab directory:
- Run with administrator privileges (Windows)
- Use `sudo` (macOS/Linux)
- Change directory permissions

### Memory Issues
For very large databases:
- Close other applications to free memory
- Increase validation sample size gradually
- Consider splitting large databases

## Advanced Features

### Multi-threaded Validation
- Database validation runs in background threads
- GUI remains responsive during validation
- Progress bars show real-time status

### Logging System
- All operations are logged with timestamps
- Logs appear in multiple tabs for easy access
- Detailed error messages for troubleshooting

### Settings Persistence
- Application remembers your preferences
- Directory locations are saved between sessions
- Validation parameters are configurable

## Getting Help

If you encounter issues:

1. Check the validation results for specific error messages
2. Review the logs in the various tabs
3. Try the "Test Database Browser" function
4. Ensure all dependencies are installed
5. Verify file permissions and directory access

## Technical Details

### Supported File Formats
- **Primary**: Pickle files (.pkl)
- **Backup**: All file types for manual operations

### Database Structure Requirements
- Root level must be a Python dictionary
- Keys should be string identifiers
- Values can be complex nested structures

### Path Fixing Patterns
The tool updates these specific patterns in Python files:
- Relative `__file__` references to absolute paths
- Hardcoded database filenames to full paths
- Platform-specific path separators

## Version History

### v1.0
- Initial GUI implementation
- Multi-threaded validation
- Comprehensive backup system
- Database browser compatibility testing
- Automatic path fixing functionality 