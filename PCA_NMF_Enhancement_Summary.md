# PCA & NMF Analysis Enhancement Summary

## Overview
Enhanced the 2D Map Analysis application with comprehensive PCA and NMF visualization in a 2x2 plot layout, including clustering analysis and re-run functionality.

## New Features

### 1. Enhanced Plotting Widget (`PCANMFPlotWidget`)
- **Location**: `map_analysis_2d/ui/plotting_widgets.py`
- **Layout**: 2x2 subplot grid showing:
  - **Top Left**: PCA Explained Variance (individual and cumulative)
  - **Top Right**: PCA Clustering Results (scatter plot of first 2 components)
  - **Bottom Left**: NMF Components (spectral signatures)
  - **Bottom Right**: NMF Clustering Results (scatter plot of first 2 components)

### 2. Automatic Clustering Integration
- **PCA Clustering**: K-means clustering on principal components
- **NMF Clustering**: K-means clustering on NMF component weights
- **Visualization**: Color-coded scatter plots with cluster legends
- **Default**: 3 clusters (configurable)

### 3. Re-run Functionality
- **PCA Re-run**: Updates clustering without re-computing PCA
- **NMF Re-run**: Updates clustering without re-computing NMF
- **UI Controls**: Green "Re-run" buttons in control panel
- **Efficiency**: Preserves analysis results, only updates visualization

### 4. Enhanced Control Panel
- **Location**: `map_analysis_2d/ui/control_panels.py`
- **New Signals**: `rerun_pca_requested`, `rerun_nmf_requested`
- **New Buttons**: 
  - "Re-run PCA (Update Clustering)"
  - "Re-run NMF (Update Clustering)"

## Technical Implementation

### Key Methods Added

#### Main Window (`map_analysis_2d/ui/main_window.py`)
```python
def perform_pca_clustering(self, pca_results, n_clusters=3)
def perform_nmf_clustering(self, nmf_results, n_clusters=3)
def rerun_pca_analysis(self)
def rerun_nmf_analysis(self)
def update_clustering_parameters(self, pca_clusters=3, nmf_clusters=3)
```

#### Plotting Widget (`map_analysis_2d/ui/plotting_widgets.py`)
```python
def plot_pca_results(self, pca_results, pca_clusters=None)
def plot_nmf_results(self, nmf_results, nmf_clusters=None, wavenumbers=None)
def setup_subplots(self)
def clear_plot(self)
```

### Data Flow
1. **Initial Analysis**: Run PCA/NMF → Perform clustering → Display in 2x2 layout
2. **Re-run**: Use cached results → Re-perform clustering → Update plots
3. **Persistence**: Results stored for multiple re-runs with different parameters

## User Workflow

### Running Analysis
1. Load map data
2. Navigate to "PCA & NMF Analysis" tab
3. Set parameters in control panel
4. Click "Run PCA Analysis" or "Run NMF Analysis"
5. View comprehensive results in 2x2 plot

### Re-running Analysis
1. After initial analysis, click "Re-run PCA (Update Clustering)" or "Re-run NMF (Update Clustering)"
2. Plots update with new clustering results
3. Can be repeated multiple times for different clustering parameters

## Visualization Features

### PCA Plots
- **Explained Variance**: Bar plot with percentage labels
- **Clustering Scatter**: PC1 vs PC2 with cluster colors
- **Legend**: Shows cluster assignments
- **Axes Labels**: Include variance percentages

### NMF Plots
- **Component Signatures**: Line plots of spectral components
- **Clustering Scatter**: Component 1 vs Component 2 with cluster colors
- **Wavenumber Support**: Uses actual wavenumbers when available
- **Color Coding**: Consistent cluster colors across plots

## Error Handling
- **Graceful Degradation**: Shows error messages for failed analyses
- **Data Validation**: Checks for sufficient components for scatter plots
- **Fallback Options**: Displays informative messages when data is insufficient

## Testing
- **Test Script**: `test_pca_nmf_plotting.py`
- **Sample Data**: Creates synthetic spectroscopic data
- **Verification**: Tests both PCA and NMF with clustering
- **Visual Confirmation**: Opens plotting window for manual inspection

## Dependencies
- **Existing**: numpy, matplotlib, scikit-learn, PySide6
- **New**: None (uses existing dependencies)

## Configuration
- **Matplotlib Config**: Uses existing `matplotlib_config.py` for consistent styling
- **Clustering Parameters**: Default 3 clusters, easily configurable
- **Plot Layout**: 2x2 grid with proper spacing and labels

## Benefits
1. **Comprehensive View**: All PCA/NMF information in one view
2. **Interactive Analysis**: Easy re-running with different parameters
3. **Clustering Insights**: Automatic pattern detection in component space
4. **Efficient Workflow**: No need to re-compute expensive analyses
5. **Professional Visualization**: Publication-ready plots with proper labeling

## Future Enhancements
- **Configurable Clustering**: UI controls for cluster count
- **Alternative Clustering**: DBSCAN, hierarchical clustering options
- **Export Options**: Save individual plots or combined figure
- **3D Visualization**: Optional 3D scatter plots for more components 