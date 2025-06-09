# NMF Analysis Integration Guide

## Overview

Non-negative Matrix Factorization (NMF) has been successfully integrated into the RamanLab 2D Map Analysis interface. This feature allows users to decompose Raman spectroscopy maps into meaningful spectral components and visualize their spatial distributions.

## Features Implemented

### 1. **NMF Analysis Engine**
- **Location**: `map_analysis_2d/analysis/nmf_analysis.py`
- **Core functionality**: Robust NMF implementation with extensive data validation
- **Key features**:
  - Automatic data cleaning and preprocessing
  - Memory-efficient batch processing for large datasets
  - Fallback mechanisms for stability
  - Comprehensive error handling

### 2. **Enhanced UI Integration**
- **Control Panel**: Complete NMF parameter controls in the NMF tab
- **Visualization**: Multi-panel comprehensive results display
- **Map Integration**: NMF components appear as selectable features in the Map View
- **Menu Integration**: Analysis, View, and Save/Load options in menu bar

### 3. **Advanced Visualization**
The NMF results tab displays:
- **Component Spectra**: Individual spectral signatures (H matrix)
- **Component Contributions**: Bar chart showing average contribution levels
- **Analysis Statistics**: Detailed metrics and convergence information
- **Component Correlation**: Heatmap showing inter-component relationships

### 4. **Map Visualization**
- NMF components automatically added to Map View feature dropdown
- Each component can be visualized as a spatial distribution map
- Custom colormap (`inferno`) for optimal component visualization
- Real-time switching between different components

### 5. **Data Management**
- **Save/Load NMF Results**: Preserve analysis results for future sessions
- **Export Functionality**: Export component maps, spectra, and metadata
- **PKL Integration**: NMF results preserved in map PKL files

## User Interface Guide

### Getting Started

1. **Load Map Data**: Start by loading your Raman map data or PKL file
2. **Navigate to NMF Tab**: Click on "NMF Analysis" tab
3. **Set Parameters**: Adjust analysis parameters as needed
4. **Run Analysis**: Click "Run NMF Analysis" button

### NMF Control Panel Parameters

#### Basic Parameters
- **Components** (2-20, default: 5): Number of NMF components to extract
- **Max Iterations** (100-1000, default: 200): Maximum iterations for convergence
- **Random State** (0-999, default: 42): Random seed for reproducible results

#### Advanced Options
- **Batch Size** (500-10000, default: 2000): Maximum samples for fitting (useful for large datasets)
- **Solver**: Choose between:
  - `mu`: Multiplicative Update (more stable, default)
  - `cd`: Coordinate Descent (faster convergence)

#### Analysis Info Panel
- Real-time display of analysis information
- Updates with results after successful analysis
- Guidance for next steps

### Using NMF Results

#### Visualization Tab
After running NMF analysis, the results tab shows:
1. **Component Spectra Plot**: Spectral signatures of each component
2. **Average Contributions**: Bar chart of component importance
3. **Analysis Statistics**: Key metrics and convergence info
4. **Correlation Matrix**: Inter-component relationships

#### Map View Integration
1. Switch to "Map View" tab
2. In the feature dropdown, select "NMF Component X" where X is the component number
3. The map displays the spatial distribution of that component

#### Spectrum View
- Click on any position in the map to view the spectrum
- Individual component contributions are overlaid when available

### Save and Load Operations

#### From Control Panel
- **Save NMF Results**: Saves current analysis for later loading
- **Load NMF Results**: Loads previously saved analysis

#### From Menu Bar
- **Analysis → Save NMF Results**: Same as control panel option
- **Analysis → Load NMF Results**: Same as control panel option

#### Export Options
- **File → Export Results**: Exports complete analysis results including:
  - Component maps as CSV files
  - Component spectra matrix
  - Component contributions matrix
  - Analysis metadata
  - High-resolution plots

## Technical Implementation

### Core Algorithm Features

#### Data Validation and Preprocessing
```python
# Automatic handling of:
- NaN and infinite values → converted to 0
- Negative values → made non-negative (NMF requirement)
- Zero-variance features → removed
- Empty spectra → filtered out
```

#### Memory Management
```python
# Efficient processing for large datasets:
- Batch processing for fitting
- Progressive transformation
- Memory-safe array operations
```

#### Robust Convergence
```python
# Multiple fallback strategies:
- Primary solver attempt
- Fallback with reduced parameters
- Graceful error handling
```

### UI Architecture

#### Control Panel System
- Dynamic control panel loading based on active tab
- Signal-slot architecture for loose coupling
- Parameter validation and tooltips

#### Visualization Integration
- GridSpec-based multi-panel layouts
- Automatic colormap selection
- Interactive map features
- Real-time plot updates

#### Data Flow
```
User Input → Control Panel → Main Window → NMF Analyzer → Results → Visualization
                ↓                                                      ↓
         Save/Load System                                    Map Integration
```

## Best Practices

### For Users

1. **Start with Default Parameters**: The defaults work well for most datasets
2. **Monitor Reconstruction Error**: Lower values indicate better fit
3. **Check Component Correlation**: High correlation may indicate over-fitting
4. **Validate Results**: Compare component spectra with known reference spectra
5. **Use Map View**: Spatial distribution helps interpret component meaning

### For Developers

1. **Error Handling**: All NMF operations include comprehensive error handling
2. **Logging**: Detailed logging for troubleshooting and optimization
3. **Memory Efficiency**: Batch processing prevents memory issues
4. **User Feedback**: Progress indicators and status updates
5. **Data Validation**: Extensive validation before analysis

## Troubleshooting

### Common Issues

#### "No valid spectra found for NMF analysis"
- **Cause**: All spectra are empty or invalid
- **Solution**: Check data loading and cosmic ray processing

#### "NMF failed: Matrix contains non-finite values"
- **Cause**: Data preprocessing failed
- **Solution**: Check for extremely large values or data corruption

#### "Component index X out of range"
- **Cause**: Trying to visualize non-existent component
- **Solution**: Re-run analysis or check component count

#### Memory Issues with Large Datasets
- **Solution**: Reduce batch size in Advanced Options
- **Alternative**: Use data subsampling in map loading

### Performance Optimization

#### For Large Datasets (>10,000 spectra):
1. Reduce batch size to 1000-2000
2. Consider using fewer components initially
3. Use coordinate descent solver (`cd`)
4. Monitor memory usage

#### For Slow Convergence:
1. Increase max iterations
2. Try different random state values
3. Check data quality and preprocessing
4. Consider different solver options

## Integration Points

### With Other Analysis Modules

#### Template Analysis
- NMF components can inform template selection
- Template fitting can validate NMF results

#### PCA Comparison
- Run both PCA and NMF for comprehensive analysis
- Compare explained variance and component interpretability

#### Cosmic Ray Detection
- Use processed spectra for best NMF results
- Cosmic ray artifacts can interfere with component identification

### Data Export Integration
- NMF results included in comprehensive data exports
- Compatible with external analysis software
- Preserves spatial coordinate information

## Future Enhancements

### Planned Features
1. **Interactive Component Selection**: Click-to-select components in plots
2. **Component Naming**: User-defined names for identified components
3. **Reference Library Integration**: Auto-comparison with spectral databases
4. **Advanced Solvers**: Additional NMF algorithm options
5. **Parallel Processing**: Multi-core optimization for large datasets

### API Extensions
```python
# Planned API enhancements:
nmf_analyzer.fit_transform_map(map_data, spatial_weights=True)
nmf_analyzer.compare_with_references(reference_library)
nmf_analyzer.optimize_components(criteria='chemical_meaning')
```

## Testing and Validation

### Automated Tests
- **Unit Tests**: Individual component testing
- **Integration Tests**: Full workflow validation
- **Performance Tests**: Memory and speed benchmarks
- **Synthetic Data Tests**: Controlled validation scenarios

### Validation Results
✓ Successfully processes synthetic multi-component data
✓ Correctly identifies known spectral signatures
✓ Handles edge cases and error conditions
✓ Maintains performance with large datasets
✓ UI integration works seamlessly

## Conclusion

The NMF analysis integration provides a powerful tool for Raman map decomposition with:
- **Robust Implementation**: Handles real-world data challenges
- **Intuitive Interface**: Easy-to-use controls and visualization
- **Comprehensive Results**: Multiple analysis perspectives
- **Integration**: Seamless workflow with other analysis tools
- **Flexibility**: Customizable parameters for different use cases

This implementation establishes a solid foundation for advanced spectral analysis workflows in RamanLab. 