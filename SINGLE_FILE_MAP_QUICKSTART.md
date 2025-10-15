# Single-File 2D Raman Map - Quick Start Guide

## Your File: `Cymato_LA57434-2(94)GI01.txt`

### File Analysis Results ✓

**Format Detected:**
- Single-file 2D Raman map (tab-delimited)
- Line 1: Wavenumber axis (672 points)
- Lines 2-1149: Spectra with X, Y positions + intensities

**Map Dimensions:**
- **1,148 total spectra**
- **4 Y positions** (from -5000 to 5000)
- **287 X positions** (from -5000 to -4895.1)
- **Grid: 287 (wide) × 4 (tall)**

**Spectral Range:**
- Wavenumbers: 100.1 to 1998.3 cm⁻¹
- 672 data points per spectrum

## Quick Start (3 Steps)

### Step 1: Visualize Your Map

```bash
python demo_single_file_map.py
```

**This will:**
- Load all 1,148 spectra
- Show map overview and dimensions
- Display sample spectra
- Create intensity maps
- Save data matrix for analysis

**Expected output:**
- 4 plots showing map structure and data quality
- Saved file: `demo_data/map_data_matrix.npy`

### Step 2: Run Cluster Analysis

```bash
python launch_cluster_analysis_single_file.py
```

**This will:**
- Load your map into the cluster analysis GUI
- Enable PCA, K-means, hierarchical clustering
- Allow spatial visualization of clusters

**What you can do:**
1. Click "PCA Analysis" to see spectral variance
2. Try "K-means Clustering" with 3-5 clusters
3. Visualize cluster spatial distribution
4. Export results

### Step 3: Custom Analysis (Optional)

```python
from map_analysis_2d.core.single_file_map_loader import SingleFileRamanMapData

# Load your map
map_data = SingleFileRamanMapData("demo_data/Cymato_LA57434-2(94)GI01.txt")

# Access data
print(f"Loaded {len(map_data.spectra)} spectra")
print(f"Map: {map_data.width} × {map_data.height}")

# Get specific spectrum
spectrum = map_data.get_spectrum(-5000, -4965.04)
print(f"Spectrum intensities: {spectrum.intensities.shape}")

# Get all data for analysis
data_matrix = map_data.get_processed_data_matrix()
print(f"Data matrix: {data_matrix.shape}")  # (1148, 672)
```

## What Was Created

### New Files

1. **`map_analysis_2d/core/single_file_map_loader.py`**
   - Parser for single-file format
   - Handles your specific data structure
   - Compatible with existing RamanLab tools

2. **`demo_single_file_map.py`**
   - Demo script to visualize your map
   - Creates overview plots
   - Saves data matrix

3. **`launch_cluster_analysis_single_file.py`**
   - Launches cluster analysis GUI
   - Pre-loads your map data
   - Ready for clustering

4. **`docs/SINGLE_FILE_MAP_IMPORT.md`**
   - Complete documentation
   - API reference
   - Advanced usage examples

## Your Map Structure

```
Cymato_LA57434-2(94)GI01.txt
├── Line 1: Wavenumbers (672 values)
│   100.093  103.274  106.457  ...  1998.3
│
└── Lines 2-1149: Spectra (1148 total)
    X_pos    Y_pos    I₁    I₂    I₃    ...    I₆₇₂
    -5000   -5000    2575  2631  2623  ...
    -5000   -4965    4249  4371  4389  ...
    -5000   -4930    4803  4857  4969  ...
    ...
```

**Grid Layout:**
```
Y positions (4):  -5000, -4965.04, -4930.07, -4895.1
X positions (287): -5000 to -4895.1 (287 steps)

Total: 4 × 287 = 1,148 spectra
```

## Next Steps

### For Map Visualization
Run `demo_single_file_map.py` to see:
- Map dimensions and statistics
- Sample spectra from different positions
- Integrated intensity maps
- Intensity distribution at specific wavenumbers

### For Cluster Analysis
Run `launch_cluster_analysis_single_file.py` to:
- Identify spectral groups/phases
- Visualize spatial distribution of clusters
- Perform PCA to see variance
- Export cluster assignments

### For Custom Analysis
Use the `SingleFileRamanMapData` class to:
- Access individual spectra by position
- Create custom intensity maps
- Perform peak fitting on regions
- Export data in different formats

## Troubleshooting

**If plots don't show:**
- Make sure matplotlib backend is configured
- Try adding `plt.show()` at the end of scripts

**If loading is slow:**
- Normal for 1,148 spectra (~5-10 seconds)
- Progress updates will show loading status

**If cluster analysis doesn't launch:**
- Ensure PySide6 is installed: `pip install PySide6`
- Check Qt dependencies

## File Format Notes

Your file uses:
- **Tab-delimited** format (✓ detected correctly)
- **First line** = wavenumber axis
- **Subsequent lines** = X, Y, intensities
- **No header row** (data starts immediately)

This format is now fully supported by RamanLab!

## Questions?

See full documentation in `docs/SINGLE_FILE_MAP_IMPORT.md` for:
- Detailed API reference
- Advanced usage examples
- Integration with other RamanLab modules
- Cosmic ray removal options
- Custom preprocessing workflows
