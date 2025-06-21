"""
Time Series Processor for Battery Strain Analysis
=================================================

Handles time-resolved Raman spectroscopy data during H/Li exchange in battery materials.
Provides data loading, preprocessing, and analysis coordination for time series experiments.

Key features:
1. Load time series data from various formats
2. Synchronize spectral data with electrochemical measurements
3. Handle data preprocessing and quality control
4. Coordinate strain analysis across time points
5. Export results for visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import json

class TimeSeriesProcessor:
    """
    Process time-resolved Raman data for battery strain analysis
    
    Handles data loading, synchronization, and preprocessing for
    time series analysis of H/Li exchange in battery materials.
    """
    
    def __init__(self, data_directory: Union[str, Path] = None):
        """
        Initialize time series processor
        
        Parameters:
        data_directory: Directory containing time series data files
        """
        self.data_directory = Path(data_directory) if data_directory else None
        self.time_series_data = []
        self.metadata = {}
        self.electrochemical_data = None
        
        # Data quality settings
        self.min_intensity_threshold = 10.0  # Minimum peak intensity
        self.max_noise_level = 0.1  # Maximum noise/signal ratio
        self.interpolation_method = 'linear'
        
    def load_time_series(self, file_pattern: str = "*.txt",
                        time_column: str = 'time',
                        frequency_column: str = 'wavenumber',
                        intensity_column: str = 'intensity') -> bool:
        """
        Load time series Raman data from files
        
        Parameters:
        file_pattern: Glob pattern for data files
        time_column: Column name for time data
        frequency_column: Column name for frequency/wavenumber
        intensity_column: Column name for intensity
        
        Returns:
        bool: Success status
        """
        
        if not self.data_directory or not self.data_directory.exists():
            warnings.warn("Data directory not found")
            return False
        
        # Find data files
        data_files = sorted(self.data_directory.glob(file_pattern))
        
        if not data_files:
            warnings.warn(f"No files found matching pattern: {file_pattern}")
            return False
        
        print(f"Loading {len(data_files)} data files...")
        
        self.time_series_data = []
        
        for i, file_path in enumerate(data_files):
            try:
                # Load single file
                data_point = self._load_single_file(file_path, 
                                                  time_column, 
                                                  frequency_column, 
                                                  intensity_column)
                
                if data_point is not None:
                    data_point['file_index'] = i
                    data_point['file_path'] = str(file_path)
                    self.time_series_data.append(data_point)
                
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.time_series_data)} data points")
        
        # Sort by time
        self.time_series_data.sort(key=lambda x: x['time'])
        
        return len(self.time_series_data) > 0
    
    def _load_single_file(self, file_path: Path, 
                         time_col: str, freq_col: str, int_col: str) -> Optional[Dict]:
        """Load a single data file"""
        
        try:
            # Try different file formats
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.txt', '.dat']:
                # Try tab-delimited first, then space-delimited
                try:
                    df = pd.read_csv(file_path, sep='\t')
                except:
                    df = pd.read_csv(file_path, sep=r'\s+')
            else:
                # Generic attempt
                df = pd.read_csv(file_path)
            
            # Extract time (could be from filename if not in data)
            if time_col in df.columns:
                time = df[time_col].iloc[0]  # Assume constant time per file
            else:
                # Try to extract from filename
                time = self._extract_time_from_filename(file_path)
            
            # Get frequency and intensity data
            frequencies = df[freq_col].values
            intensities = df[int_col].values
            
            # Quality check
            if len(frequencies) < 10 or np.max(intensities) < self.min_intensity_threshold:
                return None
            
            return {
                'time': time,
                'frequencies': frequencies,
                'intensities': intensities,
                'n_points': len(frequencies)
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _extract_time_from_filename(self, file_path: Path) -> float:
        """Extract time from filename (various formats)"""
        
        filename = file_path.stem
        
        # Common patterns
        import re
        
        # Pattern: filename_123.45s.txt or filename_123.45min.txt
        time_patterns = [
            r'(\d+\.?\d*)s(?:ec)?',  # seconds
            r'(\d+\.?\d*)m(?:in)?',  # minutes
            r'(\d+\.?\d*)h(?:r|our)?',  # hours
            r't(\d+\.?\d*)',  # t123.45
            r'time(\d+\.?\d*)',  # time123.45
            r'(\d+\.?\d*)'  # just numbers
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                time_value = float(match.group(1))
                
                # Convert to seconds
                if 'min' in pattern:
                    time_value *= 60
                elif 'h' in pattern:
                    time_value *= 3600
                
                return time_value
        
        # Default: use file index as time
        return 0.0
    
    def load_electrochemical_data(self, ec_file: Union[str, Path],
                                time_column: str = 'time',
                                voltage_column: str = 'voltage',
                                current_column: str = 'current') -> bool:
        """
        Load electrochemical data to correlate with Raman measurements
        
        Parameters:
        ec_file: Electrochemical data file
        time_column: Time column name
        voltage_column: Voltage column name  
        current_column: Current column name
        """
        
        try:
            ec_path = Path(ec_file)
            
            if ec_path.suffix.lower() == '.csv':
                df = pd.read_csv(ec_path)
            else:
                df = pd.read_csv(ec_path, sep='\t')
            
            self.electrochemical_data = {
                'time': df[time_column].values,
                'voltage': df[voltage_column].values,
                'current': df[current_column].values
            }
            
            print(f"Loaded electrochemical data: {len(df)} points")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load electrochemical data: {e}")
            return False
    
    def synchronize_with_electrochemistry(self, tolerance: float = 60.0) -> None:
        """
        Synchronize Raman data with electrochemical measurements
        
        Parameters:
        tolerance: Time tolerance for synchronization (seconds)
        """
        
        if not self.electrochemical_data:
            warnings.warn("No electrochemical data loaded")
            return
        
        ec_times = self.electrochemical_data['time']
        ec_voltages = self.electrochemical_data['voltage']
        ec_currents = self.electrochemical_data['current']
        
        # Interpolation functions
        voltage_interp = interp1d(ec_times, ec_voltages, 
                                bounds_error=False, fill_value='extrapolate')
        current_interp = interp1d(ec_times, ec_currents,
                                bounds_error=False, fill_value='extrapolate')
        
        # Add electrochemical data to each Raman measurement
        for data_point in self.time_series_data:
            raman_time = data_point['time']
            
            # Find closest electrochemical measurement
            time_diffs = np.abs(ec_times - raman_time)
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= tolerance:
                # Interpolate electrochemical values
                data_point['voltage'] = float(voltage_interp(raman_time))
                data_point['current'] = float(current_interp(raman_time))
                data_point['ec_sync'] = True
            else:
                data_point['ec_sync'] = False
                warnings.warn(f"No electrochemical data within {tolerance}s of Raman measurement at t={raman_time}s")
    
    def estimate_composition_from_voltage(self, voltage_composition_map: Dict = None) -> None:
        """
        Estimate Li/H composition from electrochemical voltage
        
        Parameters:
        voltage_composition_map: Mapping from voltage to composition
                               If None, uses typical LiMn2O4 values
        """
        
        if voltage_composition_map is None:
            # Typical LiMn2O4 voltage vs composition (simplified)
            # These are approximate values - should be calibrated for specific system
            voltage_composition_map = {
                4.2: {'Li': 0.1, 'H': 0.0},  # Highly delithiated
                4.0: {'Li': 0.5, 'H': 0.0},  
                3.8: {'Li': 0.8, 'H': 0.0},
                3.5: {'Li': 1.0, 'H': 0.0},  # Fully lithiated
                # H exchange typically at lower voltages
                3.0: {'Li': 0.8, 'H': 0.2},
                2.5: {'Li': 0.5, 'H': 0.5},
                2.0: {'Li': 0.2, 'H': 0.8},
            }
        
        # Create interpolation functions
        voltages = sorted(voltage_composition_map.keys())
        li_contents = [voltage_composition_map[v]['Li'] for v in voltages]
        h_contents = [voltage_composition_map[v]['H'] for v in voltages]
        
        li_interp = interp1d(voltages, li_contents, bounds_error=False, 
                           fill_value=(li_contents[0], li_contents[-1]))
        h_interp = interp1d(voltages, h_contents, bounds_error=False,
                          fill_value=(h_contents[0], h_contents[-1]))
        
        # Estimate composition for each data point
        for data_point in self.time_series_data:
            if 'voltage' in data_point:
                voltage = data_point['voltage']
                
                data_point['composition'] = {
                    'Li': float(li_interp(voltage)),
                    'H': float(h_interp(voltage))
                }
    
    def preprocess_spectra(self, smooth: bool = True, 
                          normalize: bool = True,
                          background_subtract: bool = True) -> None:
        """
        Preprocess all spectra in the time series
        
        Parameters:
        smooth: Apply Savitzky-Golay smoothing
        normalize: Normalize intensities
        background_subtract: Subtract polynomial background
        """
        
        for data_point in self.time_series_data:
            frequencies = data_point['frequencies']
            intensities = data_point['intensities']
            
            # Background subtraction
            if background_subtract:
                try:
                    coeffs = np.polyfit(frequencies, intensities, 3)
                    background = np.polyval(coeffs, frequencies)
                    intensities = intensities - background
                    intensities = np.maximum(intensities, 0)  # Remove negative values
                except:
                    warnings.warn("Background subtraction failed for one spectrum")
            
            # Smoothing
            if smooth and len(intensities) > 10:
                window_size = min(11, len(intensities)//4*2+1)
                intensities = savgol_filter(intensities, window_size, 2)
            
            # Normalization
            if normalize:
                max_intensity = np.max(intensities)
                if max_intensity > 0:
                    intensities = intensities / max_intensity
            
            # Update data
            data_point['intensities'] = intensities
            data_point['preprocessed'] = True
    
    def get_time_series_data(self) -> List[Dict]:
        """Get the time series data"""
        return self.time_series_data
    
    def export_data(self, output_file: Union[str, Path], format: str = 'json') -> bool:
        """
        Export time series data
        
        Parameters:
        output_file: Output file path
        format: Export format ('json', 'csv', 'hdf5')
        """
        
        output_path = Path(output_file)
        
        try:
            if format.lower() == 'json':
                # Convert numpy arrays to lists for JSON serialization
                export_data = []
                for data_point in self.time_series_data:
                    export_point = data_point.copy()
                    export_point['frequencies'] = data_point['frequencies'].tolist()
                    export_point['intensities'] = data_point['intensities'].tolist()
                    export_data.append(export_point)
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == 'csv':
                # Create a CSV with metadata and peak data
                rows = []
                for data_point in self.time_series_data:
                    row = {
                        'time': data_point['time'],
                        'n_points': data_point['n_points'],
                        'max_intensity': np.max(data_point['intensities']),
                        'file_path': data_point.get('file_path', ''),
                    }
                    
                    # Add electrochemical data if available
                    if 'voltage' in data_point:
                        row['voltage'] = data_point['voltage']
                        row['current'] = data_point['current']
                    
                    # Add composition if available
                    if 'composition' in data_point:
                        row['Li_content'] = data_point['composition']['Li']
                        row['H_content'] = data_point['composition']['H']
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
            
            print(f"Data exported to {output_path}")
            return True
            
        except Exception as e:
            warnings.warn(f"Export failed: {e}")
            return False
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for the time series"""
        
        if not self.time_series_data:
            return {}
        
        times = [dp['time'] for dp in self.time_series_data]
        n_points = [dp['n_points'] for dp in self.time_series_data]
        max_intensities = [np.max(dp['intensities']) for dp in self.time_series_data]
        
        summary = {
            'n_spectra': len(self.time_series_data),
            'time_range': (min(times), max(times)),
            'average_spectrum_length': np.mean(n_points),
            'intensity_range': (min(max_intensities), max(max_intensities)),
            'has_electrochemical_data': any('voltage' in dp for dp in self.time_series_data),
            'has_composition_data': any('composition' in dp for dp in self.time_series_data),
            'preprocessed': any(dp.get('preprocessed', False) for dp in self.time_series_data)
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Test the time series processor
    processor = TimeSeriesProcessor()
    
    print("Time Series Processor for Battery Strain Analysis")
    print("=" * 50)
    
    # Would normally load real data
    print("Processor initialized")
    print("Use load_time_series() to load experimental data")
    
    # Show example data structure
    example_data = {
        'time': 0.0,
        'frequencies': np.linspace(200, 700, 1000),
        'intensities': np.random.normal(100, 10, 1000),
        'voltage': 3.8,
        'composition': {'Li': 0.8, 'H': 0.2}
    }
    
    print("\nExample data point structure:")
    for key, value in example_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: array with {len(value)} points")
        else:
            print(f"  {key}: {value}") 