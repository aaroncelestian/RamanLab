# Cluster Export Features for RamanLab

## Overview

The RamanLab cluster analysis now includes comprehensive export functionality that allows you to analyze and visualize your clustered spectra in multiple ways. These features are available in the **Visualization tab → Scatter Plot** section.

## Export Options

### 1. Export to Folders
**Purpose**: Export each cluster's individual spectra to separate folders for detailed analysis.

**What it does**:
- Creates a separate folder for each cluster (e.g., `Cluster_0`, `Cluster_1`, etc.)
- Exports each spectrum as a separate `.txt` file with metadata
- Includes a summary file for each cluster
- Preserves original filenames and metadata

**Use case**: When you want to:
- Analyze individual spectra within each cluster
- Import specific clusters into other analysis software
- Perform detailed peak fitting on cluster members
- Compare individual spectra within and between clusters

**Output structure**:
```
Export_Directory/
├── Cluster_0/
│   ├── spectrum_001.txt
│   ├── spectrum_002.txt
│   ├── ...
│   └── cluster_0_summary.txt
├── Cluster_1/
│   ├── spectrum_021.txt
│   ├── spectrum_022.txt
│   ├── ...
│   └── cluster_1_summary.txt
└── ...
```

### 2. Export Summed Spectra ⭐ **ENHANCED**
**Purpose**: Create high-quality single spectra for each cluster by combining all member spectra with advanced signal processing.

**What it does**:
- **Creates individual `.txt` files** for each cluster's summed spectrum
- **Advanced preprocessing pipeline**:
  - Baseline correction using polynomial fitting
  - Area normalization for consistent scaling
  - Savitzky-Golay smoothing for noise reduction
  - Outlier rejection for quality control
- **Signal-to-noise improvement** of ~√N (where N = number of spectra)
- **Professional visualization** with individual spectra (transparent) and summed spectrum (bold)
- **Metadata preservation** with processing details and statistics

**Use case**: When you want to:
- **Get a single, high-quality spectrum** representing each cluster
- **Achieve maximum signal-to-noise ratio** for peak identification
- **Create publication-ready spectra** with minimal noise
- **Perform detailed spectral analysis** on cluster representatives
- **Compare cluster characteristics** with optimal spectral quality

**Features**:
- **Individual spectrum files**: Each cluster gets its own `.txt` file with the summed spectrum
- **Processing metadata**: Complete documentation of preprocessing steps
- **Quality metrics**: SNR improvement calculations and statistics
- **Visualization**: Professional plots showing individual spectra and the final summed result
- **Error bars**: Standard deviation information for uncertainty assessment

**Output files**:
```
Export_Directory/
├── cluster_0_summed_spectrum.txt
├── cluster_1_summed_spectrum.txt
├── cluster_2_summed_spectrum.txt
└── cluster_summed_spectra.png (visualization)
```

**Processing pipeline**:
1. **Baseline correction**: Polynomial fitting removes instrumental drift
2. **Area normalization**: Ensures consistent intensity scaling across spectra
3. **Noise reduction**: Savitzky-Golay smoothing preserves peak shapes
4. **Outlier rejection**: Removes spectra that deviate significantly from the cluster
5. **Final averaging**: Creates the optimal summed spectrum
6. **Quality assessment**: Calculates SNR improvement and uncertainty

### 3. Export Cluster Overview
**Purpose**: Create a comprehensive overview document with multiple analysis views.

**What it does**:
- Creates a 2x2 grid layout with different analysis perspectives
- Shows all clusters overlaid in one plot
- Displays cluster size distribution as a bar chart
- Includes detailed views of individual clusters
- Provides statistical summary information

**Use case**: When you want to:
- Get a complete overview of your clustering results
- Present results in meetings or publications
- Understand cluster size distribution
- Compare cluster characteristics at a glance

**Layout**:
- **Top Left**: All clusters overlaid with confidence intervals
- **Top Right**: Cluster size distribution bar chart
- **Bottom Left**: Detailed view of first cluster
- **Bottom Right**: Additional cluster details

## How to Use

### Step 1: Prepare Your Data
1. Import your spectral data (from files, database, or main app)
2. Run clustering analysis
3. Navigate to **Visualization tab → Scatter Plot**

### Step 2: Access Export Features
The export buttons are located at the bottom of the scatter plot tab:
- **Export to Folders**: Select a directory to save cluster folders
- **Export Summed Spectra**: Choose directory for files + filename for plot
- **Export Cluster Overview**: Save comprehensive overview document

### Step 3: Choose Export Options
Each export method provides different file format options:
- **PNG**: High-quality raster format (300 DPI)
- **PDF**: Vector format for publications
- **Folders**: Text files with metadata headers
- **Summed spectra**: Individual `.txt` files with processing details

## Technical Details

### File Formats
- **Spectrum files**: Tab-separated text with metadata headers
- **Summary files**: Human-readable text with cluster statistics
- **Summed spectra**: High-quality processed spectra with uncertainty data
- **Plots**: High-resolution PNG (300 DPI) or vector PDF

### Signal Processing Pipeline (Summed Spectra)
1. **Baseline Correction**: 
   - Polynomial fitting (degree 3) removes instrumental drift
   - Ensures baseline starts at zero
   
2. **Normalization**:
   - Area normalization for consistent intensity scaling
   - Preserves relative peak intensities
   
3. **Noise Reduction**:
   - Savitzky-Golay smoothing (window size 5-7)
   - Preserves peak shapes while reducing noise
   
4. **Quality Control**:
   - Outlier rejection (2σ threshold)
   - Removes spectra that deviate significantly
   
5. **Final Processing**:
   - Weighted averaging of remaining spectra
   - Final smoothing for optimal signal quality

### Signal-to-Noise Improvement
The summed spectra provide significant SNR improvement:
- **Theoretical improvement**: ~√N where N = number of spectra
- **Practical improvement**: 2-5x for typical cluster sizes
- **Quality metrics**: Displayed on plots and in file headers

### Metadata Preservation
All export methods preserve:
- Original filenames
- Sample IDs and descriptions
- Cluster assignments
- Export timestamps
- Analysis parameters
- Processing details (for summed spectra)

### Matplotlib Configuration
The export functions use the RamanLab matplotlib configuration for:
- Professional styling
- Publication-quality output
- Consistent color schemes
- Proper figure sizing and layout

## Example Use Cases

### Research Publication
1. Use **Export Summed Spectra** for high-quality cluster representatives
2. Use **Export Cluster Overview** for the main figure
3. Use **Export to Folders** to provide supplementary data

### Quality Control
1. Use **Export Summed Spectra** to assess cluster quality
2. Use **Export to Folders** to manually inspect individual spectra
3. Use **Export Cluster Overview** to identify problematic clusters

### Collaboration
1. Use **Export Summed Spectra** to share high-quality cluster representatives
2. Use **Export to Folders** to share specific clusters with colleagues
3. Use **Export Cluster Overview** for progress reports

## Tips and Best Practices

### For Best Results
1. **Use consistent preprocessing**: Ensure all spectra are processed the same way
2. **Check cluster quality**: Use silhouette analysis before exporting
3. **Choose appropriate colormaps**: Use distinguishable colors for different clusters
4. **Review metadata**: Ensure filenames and descriptions are meaningful
5. **Validate summed spectra**: Check that the processing pipeline works for your data

### Signal Processing Optimization
1. **Baseline correction**: Adjust polynomial degree if needed (3 is usually optimal)
2. **Smoothing parameters**: Increase window size for noisier data
3. **Outlier threshold**: Adjust 2σ threshold based on your data quality
4. **Normalization method**: Area normalization works well for most cases

### File Organization
1. **Use descriptive export directories**: Include date and analysis name
2. **Keep original data**: Always maintain backups of raw data
3. **Document analysis parameters**: Note clustering settings and preprocessing steps
4. **Archive summed spectra**: These represent significant processing effort

### Troubleshooting
- **Large datasets**: Export may take time for datasets with many spectra
- **Memory usage**: Close other applications if exporting large datasets
- **File permissions**: Ensure write permissions in export directory
- **Disk space**: Check available space before large exports
- **Processing errors**: Check that scipy is available for advanced processing

## Integration with Other RamanLab Features

The export functionality integrates seamlessly with:
- **Peak fitting**: Import summed spectra for detailed peak analysis
- **Database management**: Use summed spectra for database updates
- **Polarization analysis**: Analyze cluster representatives for orientation effects
- **Time series analysis**: Export clusters for temporal analysis

## Future Enhancements

Planned improvements include:
- **Batch export options**: Export multiple analysis results at once
- **Custom plot templates**: User-defined export layouts
- **Statistical summaries**: Automated cluster statistics
- **Integration with external software**: Direct export to common analysis packages
- **Advanced processing options**: User-configurable signal processing parameters

---

*For questions or feature requests, please refer to the RamanLab documentation or contact the development team.* 