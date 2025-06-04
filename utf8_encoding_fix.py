#!/usr/bin/env python3
"""
UTF-8 Encoding Fix for RamanLab 2D Map Analysis
====================================================

This script fixes UTF-8 encoding errors that occur when opening map files
for 2D map analysis on Windows computers.

The issue occurs because the 2D map analysis code uses pandas.read_csv()
without specifying encoding parameters, which fails when files have
different encodings (common when transferring files between Mac/Linux and Windows).

Usage:
------
python utf8_encoding_fix.py

This will:
1. Update the map_analysis_2d.py file to use robust encoding detection
2. Create a backup of the original file
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
    backup_path = filepath + '.backup'
    shutil.copy2(filepath, backup_path)
    print(f"Created backup: {backup_path}")

def fix_map_analysis_encoding():
    """Fix the UTF-8 encoding issue in map_analysis_2d.py."""
    script_dir = get_script_directory()
    filepath = os.path.join(script_dir, 'map_analysis_2d.py')
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: map_analysis_2d.py")
        print("Make sure you're running this script in the RamanLab directory.")
        return False
    
    print("🔧 Fixing UTF-8 encoding issue in 2D map analysis...")
    
    # Read the current file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the file has already been fixed
    if 'detect_encoding_robust' in content:
        print("ℹ️  File appears to already be fixed.")
        return True
    
    # Create backup
    backup_file(filepath)
    
    # Define the new robust file reading function
    robust_reader_code = '''
    def detect_encoding_robust(self, file_path):
        """
        Detect file encoding robustly for cross-platform compatibility.
        
        Parameters:
        -----------
        file_path : Path or str
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
    
    def read_spectrum_file_robust(self, filepath):
        """
        Read spectrum file with robust encoding detection and error handling.
        
        Parameters:
        -----------
        filepath : Path
            Path to the spectrum file
            
        Returns:
        --------
        tuple
            (wavenumbers, intensities) or (None, None) if failed
        """
        import pandas as pd
        import numpy as np
        
        # Try multiple encoding and delimiter combinations
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
        delimiters_to_try = [None, '\\t', ',', ';', ' ']  # None lets pandas auto-detect
        
        # First, try to detect encoding
        detected_encoding = self.detect_encoding_robust(filepath)
        encodings_to_try.insert(0, detected_encoding)  # Try detected encoding first
        
        for encoding in encodings_to_try:
            for delimiter in delimiters_to_try:
                try:
                    # Try pandas read_csv with current encoding/delimiter combination
                    df = pd.read_csv(filepath, 
                                   sep=delimiter,
                                   engine='python', 
                                   header=None, 
                                   comment='#',
                                   encoding=encoding,
                                   on_bad_lines='skip',
                                   names=['wavenumber', 'intensity'])
                    
                    if len(df) > 0 and len(df.columns) >= 2:
                        wavenumbers = df['wavenumber'].values
                        intensities = df['intensity'].values
                        
                        # Validate data
                        if len(wavenumbers) > 0 and len(intensities) > 0:
                            # Check for numeric data
                            if np.issubdtype(wavenumbers.dtype, np.number) and np.issubdtype(intensities.dtype, np.number):
                                return wavenumbers, intensities
                
                except Exception as e:
                    continue  # Try next combination
        
        # If pandas fails, try manual parsing
        for encoding in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=encoding, errors='replace') as f:
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
    
    # Find the _load_spectrum method and replace it
    old_method_pattern = r'def _load_spectrum\(self, filepath: Path\) -> Optional\[SpectrumData\]:(.*?)(?=def|\Z)'
    
    new_method = '''def _load_spectrum(self, filepath: Path) -> Optional[SpectrumData]:
        """
        Load and preprocess a single spectrum file with robust encoding detection.
        
        Parameters:
        -----------
        filepath : Path
            Path to the spectrum file
                
        Returns:
        --------
        Optional[SpectrumData]
            SpectrumData object if successful, None if failed
        """
        try:
            # Parse filename for position
            x_pos, y_pos = self._parse_filename(filepath.name)
            
            # Use robust file reading
            wavenumbers, intensities = self.read_spectrum_file_robust(filepath)
            
            if wavenumbers is None or intensities is None:
                print(f"Failed to read data from {filepath}")
                return None
            
            # Simple preprocessing without cosmic ray detection for initial load
            processed_intensities = self._preprocess_spectrum(wavenumbers, intensities)
            
            # Create spectrum data
            spectrum_data = SpectrumData(
                x_pos=x_pos,
                y_pos=y_pos,
                wavenumbers=wavenumbers,
                intensities=intensities,
                filename=filepath.name,
                processed_intensities=processed_intensities
            )
            
            # Flag for later cosmic ray detection
            spectrum_data.has_cosmic_ray = False
            
            return spectrum_data
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None'''
    
    # Insert the robust reader methods before the _load_spectrum method
    load_spectrum_pos = content.find('def _load_spectrum(self, filepath: Path)')
    if load_spectrum_pos == -1:
        print("❌ Could not find _load_spectrum method in the file.")
        return False
    
    # Find the end of the _load_spectrum method
    method_start = load_spectrum_pos
    method_end = content.find('\n    def ', method_start + 1)
    if method_end == -1:
        method_end = len(content)
    
    # Replace the method
    new_content = (content[:method_start] + 
                   robust_reader_code + '\n' +
                   new_method + '\n\n' +
                   content[method_end:])
    
    # Write the updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully updated map_analysis_2d.py with robust encoding detection")
    return True

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

def main():
    """Main function to fix UTF-8 encoding issues."""
    print("RamanLab UTF-8 Encoding Fix for 2D Map Analysis")
    print("====================================================")
    print()
    print("This script will fix UTF-8 encoding errors when opening map files")
    print("for 2D map analysis on Windows computers.")
    print()
    
    # Check chardet availability
    check_chardet_availability()
    print()
    
    # Fix the encoding issue
    success = fix_map_analysis_encoding()
    
    print()
    print("=" * 50)
    
    if success:
        print("🎉 UTF-8 encoding fix completed successfully!")
        print()
        print("The 2D map analysis should now work correctly on Windows.")
        print("The fix includes:")
        print("  ✅ Automatic encoding detection")
        print("  ✅ Multiple encoding fallbacks (utf-8, latin1, cp1252, etc.)")
        print("  ✅ Multiple delimiter detection (tab, comma, semicolon, space)")
        print("  ✅ Robust error handling")
        print()
        print("Backup file created with .backup extension")
    else:
        print("❌ Fix failed. Please check the error messages above.")
    
    print()
    print("💡 Tips:")
    print("  - The fix handles most common encoding issues automatically")
    print("  - If you still have issues, try installing chardet: pip install chardet")
    print("  - You can restore the backup if needed by removing .backup extension")

if __name__ == "__main__":
    main() 