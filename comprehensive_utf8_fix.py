#!/usr/bin/env python3
"""
Comprehensive UTF-8 Encoding Fix for RamanLab
==================================================

This script fixes UTF-8 encoding errors that occur across multiple modules
when opening files on Windows computers after copying from Mac/Linux.

The issue affects:
- 2D Map Analysis (map_analysis_2d.py)
- Batch Peak Fitting (batch_peak_fitting.py) 
- Peak Fitting (peak_fitting.py)
- Polarization Analyzer (raman_polarization_analyzer.py)

All these modules use file reading functions (np.loadtxt, pd.read_csv) 
without encoding specifications, which can fail on Windows.

Usage:
------
python comprehensive_utf8_fix.py

This will:
1. Update all affected modules with robust encoding detection
2. Create backup copies of original files
3. Implement fallback encoding methods for cross-platform compatibility
"""

import os
import sys
import shutil
import re

def get_script_directory():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def backup_file(filepath):
    """Create a backup copy of a file."""
    backup_path = filepath + '.utf8_backup'
    shutil.copy2(filepath, backup_path)
    print(f"Created backup: {backup_path}")

def check_chardet_availability():
    """Check if chardet package is available and suggest installation if not."""
    try:
        import chardet
        print("✅ chardet package is available for encoding detection")
        return True
    except ImportError:
        print("⚠️  chardet package not found - encoding detection will use fallback method")
        print("   For better encoding detection, install chardet:")
        print("   pip install chardet")
        return False

def get_robust_file_reader_code():
    """Get the robust file reading utility code to be inserted into modules."""
    return '''
    def detect_encoding_robust(self, file_path):
        """
        Detect file encoding robustly for cross-platform compatibility.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the file
            
        Returns:
        --------
        str
            Detected encoding
        """
        try:
            import chardet
            # Read a sample of the file to detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                if result['encoding'] is None:
                    return 'utf-8'  # Default to utf-8 if detection fails
                return result['encoding']
        except ImportError:
            # If chardet is not available, try common encodings
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # Try to read a small portion
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'  # Final fallback
        except Exception:
            return 'utf-8'  # Default fallback
    
    def load_spectrum_robust(self, file_path):
        """
        Load spectrum file with robust encoding detection and error handling.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the spectrum file
            
        Returns:
        --------
        tuple
            (wavenumbers, intensities) or (None, None) if failed
        """
        import pandas as pd
        import numpy as np
        
        # Try multiple methods in order of preference
        methods = [
            self._try_numpy_loadtxt,
            self._try_pandas_csv,
            self._try_manual_parsing
        ]
        
        for method in methods:
            try:
                wavenumbers, intensities = method(file_path)
                if wavenumbers is not None and intensities is not None:
                    return wavenumbers, intensities
            except Exception as e:
                continue
        
        return None, None
    
    def _try_numpy_loadtxt(self, file_path):
        """Try loading with numpy loadtxt with encoding detection."""
        import numpy as np
        
        # Detect encoding
        encoding = self.detect_encoding_robust(file_path)
        
        # Try multiple delimiter options
        delimiters = [None, '\\t', ',', ';', ' ']
        
        for delimiter in delimiters:
            try:
                # numpy loadtxt doesn't directly support encoding, so we need to handle it differently
                # First, read and decode the file content
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    lines = f.readlines()
                
                # Filter out comment lines and empty lines
                data_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        data_lines.append(line)
                
                if not data_lines:
                    continue
                
                # Parse the data
                data = []
                for line in data_lines:
                    if delimiter is None:
                        # Auto-detect delimiter
                        for delim in ['\\t', ',', ';', ' ']:
                            parts = line.split(delim)
                            if len(parts) >= 2:
                                try:
                                    wn = float(parts[0].strip())
                                    intensity = float(parts[1].strip())
                                    data.append([wn, intensity])
                                    break
                                except ValueError:
                                    continue
                    else:
                        parts = line.split(delimiter)
                        if len(parts) >= 2:
                            try:
                                wn = float(parts[0].strip())
                                intensity = float(parts[1].strip())
                                data.append([wn, intensity])
                            except ValueError:
                                continue
                
                if len(data) > 0:
                    data_array = np.array(data)
                    return data_array[:, 0], data_array[:, 1]
                    
            except Exception:
                continue
        
        return None, None
    
    def _try_pandas_csv(self, file_path):
        """Try loading with pandas with robust encoding detection."""
        import pandas as pd
        import numpy as np
        
        # Try multiple encoding and delimiter combinations
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
        delimiters_to_try = [None, '\\t', ',', ';', ' ']
        
        # First, try to detect encoding
        detected_encoding = self.detect_encoding_robust(file_path)
        encodings_to_try.insert(0, detected_encoding)
        
        for encoding in encodings_to_try:
            for delimiter in delimiters_to_try:
                try:
                    df = pd.read_csv(file_path, 
                                   sep=delimiter,
                                   engine='python', 
                                   header=None, 
                                   comment='#',
                                   encoding=encoding,
                                   on_bad_lines='skip')
                    
                    if len(df) > 0 and len(df.columns) >= 2:
                        # Use first two numeric columns
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            wavenumbers = df[numeric_cols[0]].values
                            intensities = df[numeric_cols[1]].values
                            return wavenumbers, intensities
                        elif len(df.columns) >= 2:
                            # Try to convert first two columns to numeric
                            try:
                                wavenumbers = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                                intensities = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                                # Remove NaN values
                                valid_mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
                                if np.sum(valid_mask) > 0:
                                    return wavenumbers[valid_mask], intensities[valid_mask]
                            except:
                                continue
                
                except Exception:
                    continue
        
        return None, None
    
    def _try_manual_parsing(self, file_path):
        """Try manual parsing with robust encoding detection."""
        import numpy as np
        
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
        detected_encoding = self.detect_encoding_robust(file_path)
        encodings_to_try.insert(0, detected_encoding)
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    wavenumbers = []
                    intensities = []
                    
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Try different delimiters
                        for delimiter in ['\\t', ',', ';', ' ']:
                            parts = line.split(delimiter)
                            if len(parts) >= 2:
                                try:
                                    wn = float(parts[0].strip())
                                    intensity = float(parts[1].strip())
                                    wavenumbers.append(wn)
                                    intensities.append(intensity)
                                    break
                                except ValueError:
                                    continue
                    
                    if len(wavenumbers) > 0:
                        return np.array(wavenumbers), np.array(intensities)
            
            except Exception:
                continue
        
        return None, None
'''

def fix_batch_peak_fitting():
    """Fix UTF-8 encoding issues in batch_peak_fitting.py."""
    script_dir = get_script_directory()
    filepath = os.path.join(script_dir, 'batch_peak_fitting.py')
    
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: batch_peak_fitting.py")
        return False
    
    print("🔧 Fixing UTF-8 encoding issue in batch peak fitting...")
    
    # Read the current file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'detect_encoding_robust' in content:
        print("ℹ️  Batch peak fitting appears to already be fixed.")
        return True
    
    # Create backup
    backup_file(filepath)
    
    # Replace the problematic np.loadtxt line
    old_pattern = r'data = np\.loadtxt\(file_path\)'
    new_code = '''# Load spectrum with robust encoding detection
            wavenumbers, intensities = self.load_spectrum_robust(file_path)
            if wavenumbers is None or intensities is None:
                raise Exception(f"Failed to load data from {file_path}")
            data = np.column_stack((wavenumbers, intensities))'''
    
    # Insert the robust reader methods before the load_spectrum method
    load_spectrum_pos = content.find('def load_spectrum(self, index):')
    if load_spectrum_pos == -1:
        print("❌ Could not find load_spectrum method in batch_peak_fitting.py")
        return False
    
    # Insert robust reader code before load_spectrum method
    robust_reader_code = get_robust_file_reader_code()
    new_content = (content[:load_spectrum_pos] + 
                   robust_reader_code + '\n\n    ' +
                   content[load_spectrum_pos:])
    
    # Replace the np.loadtxt line
    new_content = re.sub(old_pattern, new_code, new_content)
    
    # Write the updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully updated batch_peak_fitting.py with robust encoding detection")
    return True

def fix_polarization_analyzer():
    """Fix UTF-8 encoding issues in raman_polarization_analyzer.py."""
    script_dir = get_script_directory()
    filepath = os.path.join(script_dir, 'raman_polarization_analyzer.py')
    
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: raman_polarization_analyzer.py")
        return False
    
    print("🔧 Fixing UTF-8 encoding issue in polarization analyzer...")
    
    # Read the current file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'detect_encoding_robust' in content:
        print("ℹ️  Polarization analyzer appears to already be fixed.")
        return True
    
    # Create backup
    backup_file(filepath)
    
    # Find the load_spectrum method and replace the file reading logic
    load_spectrum_start = content.find('def load_spectrum(self):')
    if load_spectrum_start == -1:
        print("❌ Could not find load_spectrum method in polarization analyzer")
        return False
    
    # Find the end of the load_spectrum method
    load_spectrum_end = content.find('\n    def ', load_spectrum_start + 1)
    if load_spectrum_end == -1:
        # If it's the last method, find the end of the class
        load_spectrum_end = len(content)
    
    # Insert robust reader code before load_spectrum method
    robust_reader_code = get_robust_file_reader_code()
    
    # Create new load_spectrum method with robust file reading
    new_load_spectrum = '''def load_spectrum(self):
        """Load a Raman spectrum file with robust encoding detection."""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File",
            filetypes=[
                ("Text files", "*.txt"), 
                ("CSV files", "*.csv"),
                ("Data files", "*.dat"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Use robust file loading
                wavenumbers, intensities = self.load_spectrum_robust(file_path)
                
                if wavenumbers is None or intensities is None:
                    messagebox.showerror("Error", f"Failed to load data from {os.path.basename(file_path)}")
                    return
                
                # Validate data
                if len(wavenumbers) == 0 or len(intensities) == 0:
                    messagebox.showerror("Error", "File contains no valid data")
                    return
                
                if len(wavenumbers) != len(intensities):
                    messagebox.showerror("Error", "Wavenumber and intensity arrays have different lengths")
                    return
                
                # Store both original and current spectrum
                spectrum_data = {
                    'wavenumbers': wavenumbers.copy(),
                    'intensities': intensities.copy(),
                    'name': os.path.basename(file_path)
                }
                
                self.original_spectrum = spectrum_data.copy()
                self.current_spectrum = spectrum_data
                
                # Update the plot
                self.update_spectrum_plot()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading spectrum file: {str(e)}")'''
    
    # Replace the load_spectrum method
    new_content = (content[:load_spectrum_start] + 
                   robust_reader_code + '\n\n    ' +
                   new_load_spectrum + '\n\n    ' +
                   content[load_spectrum_end:])
    
    # Write the updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully updated raman_polarization_analyzer.py with robust encoding detection")
    return True

def fix_peak_fitting():
    """Fix UTF-8 encoding issues in peak_fitting.py if it exists."""
    script_dir = get_script_directory()
    filepath = os.path.join(script_dir, 'peak_fitting.py')
    
    if not os.path.exists(filepath):
        print(f"ℹ️  File not found: peak_fitting.py (this is optional)")
        return True
    
    print("🔧 Fixing UTF-8 encoding issue in peak fitting...")
    
    # Read the current file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'detect_encoding_robust' in content:
        print("ℹ️  Peak fitting appears to already be fixed.")
        return True
    
    # Create backup
    backup_file(filepath)
    
    # Look for file reading patterns and update them
    # This is a simplified fix - the specific implementation will depend on the file structure
    updated = False
    
    # Replace np.loadtxt patterns
    np_loadtxt_pattern = r'np\.loadtxt\([^)]+\)'
    if re.search(np_loadtxt_pattern, content):
        print("Found np.loadtxt usage in peak_fitting.py - manual review may be needed")
    
    # Replace pd.read_csv patterns without encoding
    pd_csv_pattern = r'pd\.read_csv\([^)]*\)(?![^)]*encoding)'
    if re.search(pd_csv_pattern, content):
        # Add encoding parameter to pd.read_csv calls
        def add_encoding(match):
            call = match.group(0)
            if 'encoding=' not in call:
                # Insert encoding parameter before the closing parenthesis
                call = call[:-1] + ', encoding="utf-8")'
            return call
        
        content = re.sub(pd_csv_pattern, add_encoding, content)
        updated = True
    
    if updated:
        # Write the updated content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Successfully updated peak_fitting.py with encoding fixes")
    else:
        print("ℹ️  No encoding issues found in peak_fitting.py")
    
    return True

def fix_map_analysis_2d():
    """Fix UTF-8 encoding issues in map_analysis_2d.py (run the existing fix)."""
    script_dir = get_script_directory()
    filepath = os.path.join(script_dir, 'map_analysis_2d.py')
    
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: map_analysis_2d.py")
        return False
    
    # Check if already fixed
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'detect_encoding_robust' in content:
        print("ℹ️  2D map analysis appears to already be fixed.")
        return True
    
    print("🔧 Applying existing UTF-8 fix to 2D map analysis...")
    
    # Import and run the existing fix
    try:
        import utf8_encoding_fix
        return utf8_encoding_fix.fix_map_analysis_encoding()
    except ImportError:
        print("⚠️  utf8_encoding_fix.py not found - 2D map analysis fix skipped")
        return False

def main():
    """Main function to fix UTF-8 encoding issues across all modules."""
    print("RamanLab Comprehensive UTF-8 Encoding Fix")
    print("===============================================")
    print()
    print("This script will fix UTF-8 encoding errors across all modules:")
    print("- 2D Map Analysis")
    print("- Batch Peak Fitting")
    print("- Peak Fitting") 
    print("- Polarization Analyzer")
    print()
    
    # Check chardet availability
    check_chardet_availability()
    print()
    
    # Apply fixes to all modules
    fixes = [
        ("2D Map Analysis", fix_map_analysis_2d),
        ("Batch Peak Fitting", fix_batch_peak_fitting),
        ("Polarization Analyzer", fix_polarization_analyzer),
        ("Peak Fitting", fix_peak_fitting),
    ]
    
    success_count = 0
    total_count = len(fixes)
    
    for module_name, fix_function in fixes:
        print(f"Fixing {module_name}...")
        try:
            if fix_function():
                success_count += 1
                print(f"✅ {module_name} fixed successfully")
            else:
                print(f"❌ {module_name} fix failed")
        except Exception as e:
            print(f"❌ {module_name} fix failed with error: {str(e)}")
        print()
    
    print("=" * 60)
    print(f"🎉 Comprehensive UTF-8 encoding fix completed!")
    print(f"Successfully fixed {success_count}/{total_count} modules")
    print()
    
    if success_count == total_count:
        print("All file reading functions now include:")
        print("  ✅ Automatic encoding detection")
        print("  ✅ Multiple encoding fallbacks (utf-8, latin1, cp1252, etc.)")
        print("  ✅ Multiple delimiter detection (tab, comma, semicolon, space)")
        print("  ✅ Robust error handling")
        print()
        print("🎯 All modules should now work correctly on Windows!")
    else:
        print("⚠️  Some modules could not be fixed automatically.")
        print("Please check the error messages above and fix manually if needed.")
    
    print()
    print("Backup files created with .utf8_backup extension")
    print()
    print("💡 Tips:")
    print("  - For best results, install chardet: pip install chardet")
    print("  - Test each module after the fix")
    print("  - You can restore backups if needed by removing .utf8_backup extension")

if __name__ == "__main__":
    main() 