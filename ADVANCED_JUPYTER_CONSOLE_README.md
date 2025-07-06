# Advanced Jupyter Console for RamanLab

## Overview

The Advanced Jupyter Console is a powerful standalone application that provides a full IPython/Jupyter environment for advanced analysis of RamanLab batch processing data. This feature bridges the gap between RamanLab's GUI-based analysis and the flexibility of Python scripting.

## Features

### 🐍 Full Jupyter Experience
- **Interactive Python Console**: Complete IPython/Jupyter functionality with syntax highlighting
- **Rich Output**: Supports matplotlib plots, pandas DataFrames, images, and HTML
- **Tab Completion**: Intelligent auto-completion based on live objects
- **Magic Commands**: Access to IPython magic commands (`%matplotlib`, `%timeit`, etc.)
- **Help System**: Built-in help with `?` and `??` syntax

### 📊 Seamless Data Integration
- **Automatic Data Loading**: Batch processing results automatically converted to pandas DataFrames
- **Pre-configured Environment**: NumPy, pandas, and matplotlib pre-loaded and configured
- **Spectral Data Access**: Full access to raw spectra, backgrounds, and fitted peaks
- **Metadata Preservation**: All processing parameters and statistics available

### 🔧 Advanced Analysis Capabilities
- **Custom Scripting**: Write and execute custom analysis scripts
- **Statistical Analysis**: Perform advanced statistics on peak parameters
- **Data Visualization**: Create custom plots and visualizations
- **Feature Engineering**: Extract custom features for machine learning
- **Batch Operations**: Process multiple spectra programmatically

## Installation

### Required Dependencies

For full functionality, install the Jupyter console components:

```bash
pip install qtconsole jupyter-client ipykernel
```

### Optional Dependencies

For enhanced functionality:

```bash
pip install matplotlib seaborn scikit-learn scipy
```

## Usage

### From RamanLab GUI

1. **Process Batch Data**: Run batch peak fitting in RamanLab and export to pickle file
2. **Open Advanced Tab**: Navigate to the "Advanced" tab
3. **Select Data**: Use "Select Pickle File" in the Data Management section to choose your pickle file
4. **Launch Console**: Click "🐍 Advanced Jupyter Console"
5. **Analyze Data**: Use the pre-loaded pandas DataFrames for analysis

### Standalone Launch

```bash
# Launch empty console
python advanced_jupyter_console.py

# Launch with existing pickle file
python advanced_jupyter_console.py batch_results.pkl

# Use the simple launcher
python launch_jupyter_console.py
```

## Data Structure

When batch processing data is loaded, the following variables are available:

### `summary_df` - Summary Statistics DataFrame
```python
summary_df.head()
#   file_index    filename  n_peaks  total_r2  processing_time
# 0          0  sample1.txt        3      0.98             2.1
# 1          1  sample2.txt        2      0.95             1.8
```

### `peaks_df` - Peak Parameters DataFrame
```python
peaks_df.head()
#   file_index    filename  peak_index  position  height  width    area    r2
# 0          0  sample1.txt           0    1085.2    1500   12.5  245.3  0.99
# 1          0  sample1.txt           1    1125.8    1200   15.2  298.7  0.97
```

### `spectra_dict` - Spectral Data Dictionary
```python
# Access spectral data for a specific file
spectrum = spectra_dict['sample1.txt']
wavenumbers = spectrum['wavenumbers']
intensities = spectrum['intensities']
background = spectrum['background']
```

### `batch_data` - Raw Batch Results
```python
# Access raw batch processing results
raw_data = batch_data[0]  # First spectrum
print(raw_data.keys())
```

## Example Analysis Scripts

### Basic Statistics
```python
# Summary statistics
print("Dataset Overview:")
print(f"Total spectra: {len(summary_df)}")
print(f"Total peaks: {len(peaks_df)}")
print(f"Average peaks per spectrum: {peaks_df.groupby('filename').size().mean():.1f}")

# Peak position statistics
print("\nPeak Position Statistics:")
print(peaks_df['position'].describe())
```

### Peak Distribution Analysis
```python
# Plot peak position histogram
plt.figure(figsize=(10, 6))
plt.hist(peaks_df['position'], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Peak Position (cm⁻¹)')
plt.ylabel('Frequency')
plt.title('Peak Position Distribution')
plt.grid(True, alpha=0.3)
plt.show()
```

### Spectral Visualization
```python
# Plot multiple spectra
plt.figure(figsize=(12, 8))
for i, (filename, spectrum) in enumerate(spectra_dict.items()):
    if i < 5:  # Plot first 5 spectra
        plt.plot(spectrum['wavenumbers'], spectrum['intensities'], 
                label=filename, alpha=0.7)
        
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title('Spectral Overlay')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Peak Correlation Analysis
```python
# Analyze peak correlations
import seaborn as sns

# Create pivot table for peak positions
peak_pivot = peaks_df.pivot_table(
    values='position', 
    index='filename', 
    columns='peak_index', 
    fill_value=np.nan
)

# Calculate correlation matrix
correlation_matrix = peak_pivot.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Peak Position Correlation Matrix')
plt.show()
```

### Quality Assessment
```python
# Analyze fit quality
plt.figure(figsize=(12, 5))

# Plot R² distribution
plt.subplot(1, 2, 1)
plt.hist(summary_df['total_r2'].dropna(), bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Total R²')
plt.ylabel('Frequency')
plt.title('Fit Quality Distribution')
plt.grid(True, alpha=0.3)

# Plot processing time vs number of peaks
plt.subplot(1, 2, 2)
plt.scatter(summary_df['n_peaks'], summary_df['processing_time'], alpha=0.6)
plt.xlabel('Number of Peaks')
plt.ylabel('Processing Time (s)')
plt.title('Processing Time vs Peak Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Feature Engineering
```python
# Extract custom features
features = []
for filename, spectrum in spectra_dict.items():
    # Calculate spectral features
    total_intensity = np.sum(spectrum['intensities'])
    max_intensity = np.max(spectrum['intensities'])
    spectral_range = np.max(spectrum['wavenumbers']) - np.min(spectrum['wavenumbers'])
    
    # Get peak statistics for this spectrum
    peaks_subset = peaks_df[peaks_df['filename'] == filename]
    n_peaks = len(peaks_subset)
    avg_peak_height = peaks_subset['height'].mean() if n_peaks > 0 else 0
    peak_width_std = peaks_subset['width'].std() if n_peaks > 0 else 0
    
    features.append({
        'filename': filename,
        'total_intensity': total_intensity,
        'max_intensity': max_intensity,
        'spectral_range': spectral_range,
        'n_peaks': n_peaks,
        'avg_peak_height': avg_peak_height,
        'peak_width_std': peak_width_std
    })

# Convert to DataFrame
features_df = pd.DataFrame(features)
print("Custom Features:")
print(features_df.head())
```

## Fallback Mode

If Jupyter components are not available, the console falls back to a simple code editor with basic Python execution. While less feature-rich, it still provides:

- Syntax highlighting
- Code execution
- Access to all data variables
- Basic output display

## Integration with RamanLab

The console is designed to work seamlessly with RamanLab's workflow:

1. **Data Pipeline**: Automatically receives batch processing results
2. **Matplotlib Configuration**: Uses RamanLab's matplotlib styling
3. **Pandas Integration**: Leverages RamanLab's preference for pandas data management
4. **Independent Operation**: Runs as a separate process for safety and stability

## Troubleshooting

### Common Issues

1. **Jupyter Not Available**: Install required packages with `pip install qtconsole jupyter-client ipykernel`
2. **Data Not Loading**: Ensure batch processing has been completed in RamanLab
3. **Plots Not Showing**: Use `%matplotlib inline` or `plt.show()` in the console
4. **Memory Issues**: Process data in chunks for very large datasets

### Performance Tips

- Use `.head()` and `.sample()` for quick data exploration
- Leverage pandas' built-in statistical functions
- Use matplotlib's figure management for multiple plots
- Save intermediate results to avoid recomputation

## Future Enhancements

Planned features include:

- **Session Save/Restore**: Save and restore console sessions
- **Custom Analysis Templates**: Pre-built analysis workflows
- **Database Integration**: Direct connection to RamanLab databases
- **Export Tools**: Enhanced data export capabilities
- **Collaborative Features**: Share analysis scripts and results

## Contributing

This feature is part of the RamanLab ecosystem. Contributions are welcome for:

- New analysis examples
- Performance improvements
- Additional data processing utilities
- Documentation enhancements

---

*This console provides the flexibility of Python scripting while maintaining the ease of use that makes RamanLab accessible to researchers at all levels.* 