# Single-File 2D Raman Map Import

## Overview

RamanLab now supports importing 2D Raman maps stored in a single text file format. This is particularly useful for data exported from certain Raman spectrometers that combine all spatial positions and spectra into one file.

## File Format

The single-file format is a tab-delimited text file with the following structure:

```
wavenumber1    wavenumber2    wavenumber3    ...    wavenumberN
X1    Y1    intensity1_1    intensity1_2    intensity1_3    ...    intensity1_N
X2    Y2    intensity2_1    intensity2_2    intensity2_3    ...    intensity2_N
X3    Y3    intensity3_1    intensity3_2    intensity3_3    ...    intensity3_N
...
```

### Format Details

- **Line 1**: Wavenumber axis (Raman shift values in cm⁻¹)
- **Lines 2+**: Each line represents one spectrum with:
  - Column 1: X position (spatial coordinate)
  - Column 2: Y position (spatial coordinate)
  - Columns 3+: Intensity values corresponding to each wavenumber

### Example

```
100.093    103.274    106.457    109.636    ...
-5000    -5000    2575    2631    2623    2685    ...
-5000    -4965.04    4249    4371    4389    4493    ...
-5000    -4930.07    4803    4857    4969    5173    ...
```

This example shows:
- Wavenumbers starting at ~100 cm⁻¹
- X positions at -5000 (constant in this case)
- Y positions varying from -5000 to higher values
- Intensity values for each (X, Y) position

## Usage

### 1. Basic Loading and Visualization

Use the demo script to load and visualize your map:

```bash
python demo_single_file_map.py
```

**What it does:**
- Loads the single-file map
- Displays map dimensions and statistics
- Shows sample spectra
- Creates intensity maps
- Saves data matrix for further analysis

**Output:**
- Map overview with dimensions
- Sample spectrum plot
- Integrated intensity map
- Intensity map at specific wavenumber
- Saved data matrix (`demo_data/map_data_matrix.npy`)

### 2. Cluster Analysis

Launch the cluster analysis GUI with your map data:

```bash
python launch_cluster_analysis_single_file.py
```

**Features available:**
- K-means clustering
- Hierarchical clustering
- PCA (Principal Component Analysis)
- NMF (Non-negative Matrix Factorization)
- UMAP dimensionality reduction
- Spatial cluster visualization

### 3. Programmatic Usage

```python
from map_analysis_2d.core.single_file_map_loader import SingleFileRamanMapData
from map_analysis_2d.core.cosmic_ray_detection import CosmicRayConfig

# Configure cosmic ray removal (optional)
cosmic_config = CosmicRayConfig(
    apply_during_load=True,  # Enable cosmic ray removal
    enabled=True,
    absolute_threshold=1000,
    neighbor_ratio=5.0
)

# Load the map
map_data = SingleFileRamanMapData(
    filepath="path/to/your/map_file.txt",
    cosmic_ray_config=cosmic_config,
    progress_callback=lambda p, m: print(f"[{p}%] {m}")
)

# Access map information
print(f"Map dimensions: {map_data.width} × {map_data.height}")
print(f"Total spectra: {len(map_data.spectra)}")
print(f"Wavenumber range: {map_data.target_wavenumbers[0]:.1f} to {map_data.target_wavenumbers[-1]:.1f} cm⁻¹")

# Get a specific spectrum
spectrum = map_data.get_spectrum(x_pos=-5000, y_pos=-4965.04)
if spectrum:
    print(f"Spectrum at ({spectrum.x_pos}, {spectrum.y_pos})")
    print(f"Intensities shape: {spectrum.intensities.shape}")

# Get all data as a matrix for analysis
data_matrix = map_data.get_processed_data_matrix()
print(f"Data matrix shape: {data_matrix.shape}")  # (n_spectra, n_wavenumbers)

# Get position list
positions = map_data.get_position_list()
print(f"Positions: {positions[:5]}...")  # First 5 positions
```

## Map Dimensions

The loader automatically detects:
- **Number of unique X positions** (map width)
- **Number of unique Y positions** (map height)
- **Total number of spectra** (width × height)
- **Wavenumber range and resolution**

### Example Output

```
Map dimensions: 287 X positions × 4 Y positions
Total spectra: 1148
X range: -5000.0 to -4895.1
Y range: -5000.0 to 5000.0
Wavenumber range: 100.1 to 1998.3 cm⁻¹
Spectral points: 672
```

## Data Processing

### Automatic Processing Steps

1. **File Parsing**: Reads tab-delimited format
2. **Position Extraction**: Identifies X, Y coordinates
3. **Spectrum Interpolation**: Resamples to target wavenumbers (if needed)
4. **Smoothing**: Applies Savitzky-Golay filter (window=5, order=2)
5. **Cosmic Ray Removal** (optional): Detects and removes cosmic ray spikes
6. **Data Matrix Creation**: Organizes spectra for cluster analysis

### Cosmic Ray Removal

Enable cosmic ray detection during loading:

```python
cosmic_config = CosmicRayConfig(
    apply_during_load=True,      # Enable during load
    enabled=True,                 # Enable detection
    absolute_threshold=1000,      # Intensity threshold
    neighbor_ratio=5.0,           # Ratio to neighbors
    removal_range=3,              # Points to remove around spike
    enable_intelligent_noise=True # Add realistic noise to repaired regions
)
```

## Integration with Existing Tools

### Map Analysis 2D

The loaded data is compatible with the `map_analysis_2d` module:

```python
# After loading with SingleFileRamanMapData
from map_analysis_2d.analysis.pca_analysis import perform_pca
from map_analysis_2d.analysis.ml_classification import MLClassifier

# Perform PCA
data_matrix = map_data.get_processed_data_matrix()
pca_results = perform_pca(data_matrix, n_components=5)

# Use ML classification
classifier = MLClassifier()
# ... train and classify
```

### Cluster Analysis

The data integrates seamlessly with `raman_cluster_analysis_qt6.py`:

```python
from raman_cluster_analysis_qt6 import RamanClusterAnalysisQt6

# Load map data
map_data = SingleFileRamanMapData(filepath="your_map.txt")

# Get data for cluster analysis
data_matrix = map_data.get_processed_data_matrix()
wavenumbers = map_data.target_wavenumbers
positions = map_data.get_position_list()

# Create cluster analysis window
window = RamanClusterAnalysisQt6()
window.wavenumbers = wavenumbers
window.intensities = data_matrix
window.filenames = [f"pos_{x}_{y}" for x, y in positions]
```

## Troubleshooting

### File Not Loading

**Issue**: File not found or cannot be read

**Solutions**:
- Check file path is correct
- Ensure file is tab-delimited (not space or comma)
- Verify file has at least 2 lines (header + data)

### Wrong Number of Columns

**Issue**: Warning about mismatched intensity count

**Solutions**:
- Check that all data rows have the same number of columns
- Verify wavenumber header has correct number of values
- Look for missing or extra tab characters

### Memory Issues

**Issue**: Large files causing memory problems

**Solutions**:
- Process in chunks (modify loader)
- Reduce spectral resolution by resampling
- Use subset of data for initial analysis

### Visualization Issues

**Issue**: Maps look incorrect or distorted

**Solutions**:
- Check X, Y position ordering
- Verify grid dimensions match expectations
- Use `aspect='auto'` in imshow for non-square grids

## File Format Variations

The loader can handle:
- **Different delimiters**: Tab (default), comma, space
- **Variable spacing**: Multiple spaces/tabs between values
- **Comments**: Lines starting with `#` are ignored
- **Missing values**: Handled gracefully with NaN

## Performance

### Loading Speed

- **Small maps** (< 100 spectra): < 1 second
- **Medium maps** (100-1000 spectra): 1-5 seconds
- **Large maps** (1000-10000 spectra): 5-30 seconds

### Memory Usage

Approximate memory requirements:
- **Data storage**: ~8 bytes × n_spectra × n_wavenumbers
- **Example**: 1000 spectra × 1000 points = ~8 MB
- **With processing**: ~2-3× raw data size

## Advanced Features

### Custom Wavenumber Resampling

```python
# Define custom target wavenumbers
target_wn = np.linspace(200, 2000, 500)  # 500 points from 200-2000 cm⁻¹

map_data = SingleFileRamanMapData(
    filepath="your_map.txt",
    target_wavenumbers=target_wn
)
```

### Progress Monitoring

```python
def my_progress(progress, message):
    print(f"Progress: {progress}% - {message}")
    # Could update a GUI progress bar here

map_data = SingleFileRamanMapData(
    filepath="your_map.txt",
    progress_callback=my_progress
)
```

### Accessing Raw vs Processed Data

```python
spectrum = map_data.get_spectrum(x_pos, y_pos)

# Raw data (as loaded from file)
raw_intensities = spectrum.intensities

# Processed data (smoothed, resampled)
processed_intensities = spectrum.processed_intensities
```

## Example Workflows

### Workflow 1: Quick Visualization

```bash
# 1. Load and visualize
python demo_single_file_map.py

# 2. View the generated plots
# 3. Check saved data matrix
```

### Workflow 2: Cluster Analysis

```bash
# 1. Launch cluster analysis
python launch_cluster_analysis_single_file.py

# 2. In the GUI:
#    - Run PCA to see variance
#    - Try K-means with different cluster numbers
#    - Visualize cluster maps
#    - Export results
```

### Workflow 3: Custom Analysis

```python
# Load data
from map_analysis_2d.core.single_file_map_loader import SingleFileRamanMapData
import numpy as np
import matplotlib.pyplot as plt

map_data = SingleFileRamanMapData("your_map.txt")

# Custom analysis: Find peak at specific wavenumber
target_wn = 1000  # cm⁻¹
wn_idx = np.argmin(np.abs(map_data.target_wavenumbers - target_wn))

# Create intensity map at this wavenumber
intensity_map = np.zeros((map_data.height, map_data.width))
for i, y in enumerate(map_data.y_positions):
    for j, x in enumerate(map_data.x_positions):
        spectrum = map_data.get_spectrum(x, y)
        if spectrum:
            intensity_map[i, j] = spectrum.processed_intensities[wn_idx]

# Plot
plt.imshow(intensity_map, cmap='hot', aspect='auto')
plt.colorbar(label='Intensity')
plt.title(f'Intensity at {map_data.target_wavenumbers[wn_idx]:.1f} cm⁻¹')
plt.show()
```

## API Reference

### SingleFileRamanMapData

**Constructor**:
```python
SingleFileRamanMapData(
    filepath: str,
    target_wavenumbers: Optional[np.ndarray] = None,
    cosmic_ray_config: CosmicRayConfig = None,
    progress_callback: callable = None
)
```

**Properties**:
- `width`: Number of X positions
- `height`: Number of Y positions
- `x_positions`: List of unique X coordinates
- `y_positions`: List of unique Y coordinates
- `target_wavenumbers`: Wavenumber axis
- `spectra`: Dictionary of SpectrumData objects

**Methods**:
- `get_spectrum(x_pos, y_pos)`: Get spectrum at position
- `get_map_dimensions()`: Get (width, height) tuple
- `get_processed_data_matrix()`: Get all spectra as matrix
- `get_position_list()`: Get list of (x, y) positions
- `get_grid_indices(x_pos, y_pos)`: Convert position to grid indices

## See Also

- [Map Analysis 2D Documentation](./MAP_ANALYSIS_2D_README.md)
- [Cluster Analysis Guide](./CLUSTER_ANALYSIS_README.md)
- [Cosmic Ray Detection](./COSMIC_RAY_DETECTION.md)
