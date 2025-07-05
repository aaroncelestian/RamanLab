"""
Data Processing Engine for Batch Peak Fitting
Handles file loading, spectrum management, and data operations
Extracted from the original batch_peak_fitting_qt6.py for better modularity
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from PySide6.QtCore import QObject, Signal

# Optional import for encoding detection
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False


class DataProcessor(QObject):
    """
    Core data processing engine handling file loading and spectrum management
    Separated from UI for better testability and maintainability
    """
    
    # Signals for communication with UI
    spectrum_loaded = Signal(dict)  # Emits spectrum data when loaded
    spectra_list_changed = Signal(list)  # Emits updated file list
    current_spectrum_changed = Signal(int)  # Emits new current index
    file_operation_completed = Signal(str, bool, str)  # (operation, success, message)
    
    def __init__(self):
        super().__init__()
        
        # File management
        self.spectra_files = []
        self.current_spectrum_index = 0
        
        # Current spectrum data
        self.wavenumbers = np.array([])
        self.intensities = np.array([])
        self.original_intensities = np.array([])
        
        # Peak data
        self.peaks = np.array([], dtype=int)
        self.manual_peaks = np.array([], dtype=int)
        
        # Fitting data
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.residuals = None
        
        # Batch results storage
        self.batch_results = []
    
    def add_files(self, file_paths=None):
        """
        Add spectrum files for batch processing
        Returns: number of files added
        """
        if file_paths is None:
            return 0
        
        files_added = 0
        for file_path in file_paths:
            if file_path not in self.spectra_files:
                self.spectra_files.append(file_path)
                files_added += 1
        
        if files_added > 0:
            self.spectra_list_changed.emit(self.spectra_files.copy())
            
            # Load first spectrum if this is the first batch of files
            if len(self.spectra_files) == files_added:
                self.load_spectrum(0)
        
        return files_added
    
    def remove_files(self, indices):
        """
        Remove files by indices
        Returns: number of files removed
        """
        if not indices:
            return 0
        
        # Sort indices in reverse order to maintain validity during removal
        sorted_indices = sorted(indices, reverse=True)
        files_removed = 0
        
        for index in sorted_indices:
            if 0 <= index < len(self.spectra_files):
                self.spectra_files.pop(index)
                files_removed += 1
        
        if files_removed > 0:
            # Adjust current index if necessary
            if self.current_spectrum_index >= len(self.spectra_files):
                self.current_spectrum_index = max(0, len(self.spectra_files) - 1)
            
            self.spectra_list_changed.emit(self.spectra_files.copy())
            
            # Load a spectrum if files remain, otherwise clear
            if self.spectra_files:
                self.load_spectrum(self.current_spectrum_index)
            else:
                self.clear_current_spectrum()
        
        return files_removed
    
    def navigate_spectrum(self, direction):
        """
        Navigate through spectra
        direction: 0=first, -2=last, -1=previous, 1=next
        Returns: True if navigation successful
        """
        if not self.spectra_files:
            return False
        
        if direction == 0:  # First
            new_index = 0
        elif direction == -2:  # Last
            new_index = len(self.spectra_files) - 1
        elif direction == -1:  # Previous
            new_index = max(0, self.current_spectrum_index - 1)
        elif direction == 1:  # Next
            new_index = min(len(self.spectra_files) - 1, self.current_spectrum_index + 1)
        else:
            return False
        
        if new_index != self.current_spectrum_index:
            return self.load_spectrum(new_index)
        
        return True
    
    def load_spectrum(self, index):
        """
        Load a spectrum by index
        Returns: True if successful, False otherwise
        """
        if not (0 <= index < len(self.spectra_files)):
            return False
        
        file_path = self.spectra_files[index]
        
        try:
            # Check if we have batch results for this file
            batch_result = self._find_batch_result(file_path)
            
            if batch_result:
                self._load_from_batch_result(batch_result, index)
            else:
                self._load_from_file(file_path, index)
            
            # Emit spectrum loaded signal
            spectrum_data = {
                'wavenumbers': self.wavenumbers,
                'intensities': self.intensities,
                'original_intensities': self.original_intensities,
                'peaks': self.peaks,
                'manual_peaks': self.manual_peaks,
                'fit_params': self.fit_params,
                'background': self.background,
                'residuals': self.residuals,
                'file_path': file_path,
                'index': index
            }
            self.spectrum_loaded.emit(spectrum_data)
            self.current_spectrum_changed.emit(index)
            
            return True
            
        except Exception as e:
            print(f"Error loading spectrum {index}: {e}")
            return False
    
    def _find_batch_result(self, file_path):
        """Find batch result for a given file path"""
        if not self.batch_results:
            return None
        
        for result in self.batch_results:
            # Safely handle fit_failed flag
            fit_failed = result.get('fit_failed', True)
            
            # Handle numpy array boolean context - convert array to scalar boolean
            if isinstance(fit_failed, np.ndarray):
                if fit_failed.size == 0:
                    fit_failed = True  # Empty array means failed
                elif fit_failed.size == 1:
                    fit_failed = bool(fit_failed.item())  # Single element - extract scalar
                else:
                    fit_failed = bool(fit_failed.any())  # Multiple elements - check if any are True
            elif fit_failed is None:
                fit_failed = True
            else:
                fit_failed = bool(fit_failed)  # Ensure it's a proper boolean
            
            if result['file'] == file_path and not fit_failed:
                return result
        
        return None
    
    def _load_from_batch_result(self, batch_result, index):
        """Load spectrum data from batch result"""
        self.wavenumbers = np.array(batch_result['wavenumbers'])
        self.intensities = np.array(batch_result['intensities'])
        self.original_intensities = np.array(batch_result['original_intensities'])
        
        # Handle background safely
        if batch_result.get('background') is not None:
            self.background = np.array(batch_result['background'])
        else:
            self.background = None
        
        # Handle peaks safely
        reference_peaks = batch_result.get('reference_peaks', None)
        peaks = batch_result.get('peaks', None)
        
        if (reference_peaks is not None and 
            isinstance(reference_peaks, (list, np.ndarray)) and len(reference_peaks) > 0):
            self.peaks = np.array(reference_peaks)
        elif (peaks is not None and 
              isinstance(peaks, (list, np.ndarray)) and len(peaks) > 0):
            self.peaks = np.array(peaks)
        else:
            self.peaks = np.array([], dtype=int)
        
        # Handle fit parameters safely
        self.fit_params = batch_result.get('fit_params', [])
        if self.fit_params is None:
            self.fit_params = []
        
        self.fit_result = True
        
        # Handle residuals safely
        if batch_result.get('residuals') is not None:
            self.residuals = np.array(batch_result['residuals'])
        else:
            self.residuals = None
        
        # Clear manual peaks when loading fitted data
        self.manual_peaks = np.array([], dtype=int)
        self.current_spectrum_index = index
    
    def _load_from_file(self, file_path, index):
        """Load spectrum data from file"""
        data = self.load_spectrum_robust(file_path)
        
        if data is not None:
            self.wavenumbers, self.intensities = data
            self.original_intensities = self.intensities.copy()
            self.current_spectrum_index = index
            
            # Reset peak data for new spectrum
            self.peaks = np.array([], dtype=int)
            self.manual_peaks = np.array([], dtype=int)
            self.fit_params = []
            self.fit_result = None
            self.background = None
            self.residuals = None
        else:
            raise Exception(f"Could not load data from {file_path}")
    
    def load_spectrum_robust(self, file_path):
        """
        Robust spectrum loading with multiple strategies
        Returns: tuple (wavenumbers, intensities) or None if failed
        """
        try:
            # Strategy 1: Try numpy loadtxt
            try:
                data = np.loadtxt(file_path)
                if data.ndim == 2 and data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
                elif data.ndim == 1 and len(data) >= 2:
                    # Single row case
                    return np.array([0]), np.array([data[0]])
            except:
                pass
            
            # Strategy 2: Try pandas with automatic delimiter detection
            try:
                df = pd.read_csv(file_path, sep=None, engine='python', comment='#')
                if len(df.columns) >= 2:
                    return df.iloc[:, 0].values, df.iloc[:, 1].values
            except:
                pass
            
            # Strategy 3: Try with different encodings
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python', 
                                   encoding=encoding, comment='#')
                    if len(df.columns) >= 2:
                        return df.iloc[:, 0].values, df.iloc[:, 1].values
                except:
                    continue
            
            # Strategy 4: Detect encoding and manual parsing
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    
                # Use chardet if available, otherwise try common encodings
                if CHARDET_AVAILABLE:
                    detected = chardet.detect(raw_data)
                    encoding = detected['encoding'] or 'utf-8'
                else:
                    # Fallback: try to detect encoding manually
                    encoding = 'utf-8'
                    # Try to decode with utf-8 first
                    try:
                        raw_data.decode('utf-8')
                        encoding = 'utf-8'
                    except UnicodeDecodeError:
                        # If utf-8 fails, try common alternatives
                        for fallback_encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                            try:
                                raw_data.decode(fallback_encoding)
                                encoding = fallback_encoding
                                break
                            except UnicodeDecodeError:
                                continue
                
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                
                wavenumbers = []
                intensities = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('//'):
                        try:
                            # Try different separators
                            separators = ['\t', ',', ';', ' ', '  ', '   ']
                            parts = None
                            
                            for sep in separators:
                                if sep in line:
                                    parts = line.split(sep)
                                    break
                            
                            if parts is None:
                                parts = line.split()
                            
                            # Clean empty strings
                            parts = [p.strip() for p in parts if p.strip()]
                            
                            if len(parts) >= 2:
                                wavenumber = float(parts[0])
                                intensity = float(parts[1])
                                wavenumbers.append(wavenumber)
                                intensities.append(intensity)
                                
                        except (ValueError, IndexError):
                            continue
                
                if len(wavenumbers) > 0:
                    return np.array(wavenumbers), np.array(intensities)
                    
            except Exception as e:
                print(f"Encoding detection failed for {file_path}: {e}")
            
            # Strategy 5: Last resort - try to read as binary and convert
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Try to decode with different encodings
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        text = content.decode(encoding)
                        lines = text.split('\n')
                        
                        wavenumbers = []
                        intensities = []
                        
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                try:
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        wavenumber = float(parts[0])
                                        intensity = float(parts[1])
                                        wavenumbers.append(wavenumber)
                                        intensities.append(intensity)
                                except (ValueError, IndexError):
                                    continue
                        
                        if len(wavenumbers) > 0:
                            return np.array(wavenumbers), np.array(intensities)
                            
                    except UnicodeDecodeError:
                        continue
                        
            except Exception as e:
                print(f"Binary reading failed for {file_path}: {e}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
        return None
    
    def clear_current_spectrum(self):
        """Clear current spectrum data"""
        self.wavenumbers = np.array([])
        self.intensities = np.array([])
        self.original_intensities = np.array([])
        self.peaks = np.array([], dtype=int)
        self.manual_peaks = np.array([], dtype=int)
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.residuals = None
        
        # Emit empty spectrum
        spectrum_data = {
            'wavenumbers': self.wavenumbers,
            'intensities': self.intensities,
            'original_intensities': self.original_intensities,
            'peaks': self.peaks,
            'manual_peaks': self.manual_peaks,
            'fit_params': self.fit_params,
            'background': self.background,
            'residuals': self.residuals,
            'file_path': None,
            'index': -1
        }
        self.spectrum_loaded.emit(spectrum_data)
    
    def set_current_spectrum(self, wavenumbers, intensities):
        """Set current spectrum data directly"""
        self.wavenumbers = np.array(wavenumbers)
        self.intensities = np.array(intensities)
        self.original_intensities = self.intensities.copy()
        
        # Reset derived data
        self.peaks = np.array([], dtype=int)
        self.manual_peaks = np.array([], dtype=int)
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.residuals = None
        
        # Emit spectrum loaded signal
        spectrum_data = {
            'wavenumbers': self.wavenumbers,
            'intensities': self.intensities,
            'original_intensities': self.original_intensities,
            'peaks': self.peaks,
            'manual_peaks': self.manual_peaks,
            'fit_params': self.fit_params,
            'background': self.background,
            'residuals': self.residuals,
            'file_path': None,
            'index': -1
        }
        self.spectrum_loaded.emit(spectrum_data)
    
    def update_spectrum_data(self, **kwargs):
        """Update current spectrum data with new values"""
        if 'wavenumbers' in kwargs:
            self.wavenumbers = np.array(kwargs['wavenumbers'])
        if 'intensities' in kwargs:
            self.intensities = np.array(kwargs['intensities'])
        if 'original_intensities' in kwargs:
            self.original_intensities = np.array(kwargs['original_intensities'])
        if 'peaks' in kwargs:
            self.peaks = np.array(kwargs['peaks'])
        if 'manual_peaks' in kwargs:
            self.manual_peaks = np.array(kwargs['manual_peaks'])
        if 'fit_params' in kwargs:
            self.fit_params = kwargs['fit_params']
        if 'fit_result' in kwargs:
            self.fit_result = kwargs['fit_result']
        if 'background' in kwargs:
            self.background = kwargs['background']
        if 'residuals' in kwargs:
            self.residuals = kwargs['residuals']
    
    def get_current_spectrum(self):
        """Get current spectrum data including all analysis results"""
        if not self.has_spectrum_data():
            return None
        
        spectrum_data = {
            'wavenumbers': self.wavenumbers.copy() if len(self.wavenumbers) > 0 else np.array([]),
            'intensities': self.intensities.copy() if len(self.intensities) > 0 else np.array([]),
            'original_intensities': self.original_intensities.copy() if len(self.original_intensities) > 0 else np.array([]),
            'peaks': self.peaks.copy() if len(self.peaks) > 0 else np.array([], dtype=int),
            'manual_peaks': self.manual_peaks.copy() if len(self.manual_peaks) > 0 else np.array([], dtype=int),
            'fit_params': self.fit_params.copy() if self.fit_params else [],
            'background': self.background.copy() if self.background is not None else None,
            'residuals': self.residuals.copy() if self.residuals is not None else None,
            'file_path': self.get_current_file_path(),
            'index': self.current_spectrum_index,
            'fit_result': self.fit_result  # Include complete fitting results
        }
        
        return spectrum_data
    
    def get_file_list(self):
        """Get list of loaded files"""
        return self.spectra_files.copy()
    
    def get_current_file_path(self):
        """Get current file path"""
        if 0 <= self.current_spectrum_index < len(self.spectra_files):
            return self.spectra_files[self.current_spectrum_index]
        return None
    
    def get_current_file_name(self):
        """Get current file name"""
        file_path = self.get_current_file_path()
        if file_path:
            return os.path.basename(file_path)
        return "None"
    
    def get_file_status(self):
        """Get file status string"""
        if self.spectra_files:
            filename = self.get_current_file_name()
            return f"File {self.current_spectrum_index + 1} of {len(self.spectra_files)}: {filename}"
        else:
            return "No files loaded"
    
    def has_spectrum_data(self):
        """Check if current spectrum has data"""
        return len(self.wavenumbers) > 0 and len(self.intensities) > 0
    
    def add_manual_peak(self, peak_index):
        """Add a manual peak"""
        if peak_index not in self.manual_peaks and 0 <= peak_index < len(self.intensities):
            self.manual_peaks = np.append(self.manual_peaks, peak_index)
            self.manual_peaks = np.sort(self.manual_peaks)
    
    def remove_manual_peak(self, peak_index):
        """Remove a manual peak"""
        if peak_index in self.manual_peaks:
            self.manual_peaks = self.manual_peaks[self.manual_peaks != peak_index]
    
    def clear_manual_peaks(self):
        """Clear all manual peaks"""
        self.manual_peaks = np.array([], dtype=int)
    
    def get_all_peaks(self):
        """Get combined automatic and manual peaks"""
        if len(self.manual_peaks) > 0:
            # Ensure both arrays are the same dtype before concatenation
            peaks_int = self.peaks.astype(int)
            manual_peaks_int = self.manual_peaks.astype(int)
            all_peaks = np.concatenate([peaks_int, manual_peaks_int])
            return np.unique(np.sort(all_peaks)).astype(int)
        return self.peaks.astype(int).copy()
    
    def set_batch_results(self, results):
        """Set batch processing results"""
        self.batch_results = results
    
    def add_batch_result(self, result):
        """Add a single batch result"""
        self.batch_results.append(result)
    
    def get_batch_results(self):
        """Get batch processing results"""
        return self.batch_results.copy() if self.batch_results else []
    
    def clear_batch_results(self):
        """Clear batch processing results"""
        self.batch_results = []
    
    # Additional methods for Phase 2 architecture compatibility
    
    def load_spectrum_from_path(self, file_path):
        """Load spectrum directly from file path"""
        try:
            result = self.load_spectrum_robust(file_path)
            if result is not None:
                wavenumbers, intensities = result
                self.set_current_spectrum(wavenumbers, intensities)
                self.file_operation_completed.emit("load_spectrum", True, f"Loaded {os.path.basename(file_path)}")
                return True
            else:
                self.file_operation_completed.emit("load_spectrum", False, f"Failed to load {os.path.basename(file_path)}")
                return False
        except Exception as e:
            self.file_operation_completed.emit("load_spectrum", False, f"Error loading {file_path}: {str(e)}")
            return False
    
    def apply_background_subtraction(self, background):
        """Apply background subtraction to current spectrum"""
        if background is not None and len(self.intensities) == len(background):
            self.intensities = self.intensities - background
            self.background = background
            print("DataProcessor: Background subtraction applied")
            
            # Emit updated spectrum
            spectrum_data = self.get_current_spectrum()
            self.spectrum_loaded.emit(spectrum_data)
            
            self.file_operation_completed.emit("apply_background", True, "Background subtraction applied")
        else:
            self.file_operation_completed.emit("apply_background", False, "Invalid background data")
    
    def preview_background_subtraction(self, background):
        """Preview background subtraction without applying"""
        if background is not None and len(self.intensities) == len(background):
            # Store the background for preview but don't apply it
            self.background = background
            print("DataProcessor: Background preview updated")
            self.file_operation_completed.emit("preview_background", True, "Background preview updated")
        else:
            self.file_operation_completed.emit("preview_background", False, "Invalid background data")
    
    def clear_background_subtraction(self):
        """Clear background subtraction and restore original intensities"""
        if len(self.original_intensities) > 0:
            self.intensities = self.original_intensities.copy()
            self.background = None
            print("DataProcessor: Background subtraction cleared")
            
            # Emit updated spectrum
            spectrum_data = self.get_current_spectrum()
            self.spectrum_loaded.emit(spectrum_data)
            
            self.file_operation_completed.emit("clear_background", True, "Background cleared")
        else:
            self.file_operation_completed.emit("clear_background", False, "No original intensities available")
    
    def export_results_to_csv(self, results):
        """Export results to CSV format"""
        try:
            import pandas as pd
            from datetime import datetime
            
            # Convert results to DataFrame
            df_data = []
            for result in results:
                row = {
                    'file_name': result.get('file_name', ''),
                    'success': result.get('success', False),
                    'peaks_found': len(result.get('peak_positions', [])),
                    'r_squared': result.get('r_squared', 0.0),
                    'processing_time': result.get('processing_time', 0.0),
                    'notes': result.get('notes', '')
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_results_{timestamp}.csv"
            
            df.to_csv(filename, index=False)
            print(f"DataProcessor: Results exported to {filename}")
            self.file_operation_completed.emit("export_csv", True, f"Results exported to {filename}")
            
        except Exception as e:
            print(f"DataProcessor: Error exporting to CSV: {e}")
            self.file_operation_completed.emit("export_csv", False, f"Export failed: {str(e)}")
    
    def export_comprehensive_results(self, results):
        """Export comprehensive results with detailed information"""
        try:
            from datetime import datetime
            import json
            
            # Create comprehensive data structure
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_files': len(results),
                'successful_fits': sum(1 for r in results if r.get('success', False)),
                'results': results
            }
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"DataProcessor: Comprehensive results exported to {filename}")
            self.file_operation_completed.emit("export_comprehensive", True, f"Comprehensive results exported to {filename}")
            
        except Exception as e:
            print(f"DataProcessor: Error exporting comprehensive results: {e}")
            self.file_operation_completed.emit("export_comprehensive", False, f"Export failed: {str(e)}")
    
    def reset(self):
        """Reset data processor to default state"""
        self.spectra_files = []
        self.current_spectrum_index = -1
        self.wavenumbers = np.array([])
        self.intensities = np.array([])
        self.original_intensities = np.array([])
        self.peaks = np.array([], dtype=int)
        self.manual_peaks = np.array([], dtype=int)
        self.fit_params = []
        self.fit_result = None
        self.background = None
        self.residuals = None
        self.batch_results = []
        
        print("DataProcessor: Reset to defaults")
        
        # Emit signals
        self.spectra_list_changed.emit([])
        self.current_spectrum_changed.emit(-1)
        self.file_operation_completed.emit("reset", True, "DataProcessor reset to defaults")
    
    def get_status(self):
        """Get current status of data processor"""
        return {
            'files_loaded': len(self.spectra_files),
            'current_index': self.current_spectrum_index,
            'has_spectrum': len(self.wavenumbers) > 0,
            'has_background': self.background is not None,
            'peaks_count': len(self.peaks),
            'manual_peaks_count': len(self.manual_peaks),
            'batch_results_count': len(self.batch_results)
        } 