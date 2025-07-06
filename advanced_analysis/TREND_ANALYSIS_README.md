# 📈 Trend Analysis with Confidence Intervals - Documentation

## Overview

The **Trend Analysis** module provides powerful sequential analysis capabilities for Raman density measurements, enabling you to discover patterns and trends in how density and crystallinity change across scan sequences with rigorous statistical confidence intervals.

## 🎯 Key Features

### Statistical Analysis
- **Linear regression** with confidence intervals
- **Prediction intervals** for uncertainty quantification
- **Statistical significance testing** (p-values)
- **R-squared** goodness-of-fit metrics
- **Robust trend detection** with customizable sensitivity

### Data Processing
- **Multiple sequence ordering** options (alphabetical, numerical, timestamp)
- **Savitzky-Golay smoothing** for noise reduction
- **Outlier-resistant** analysis methods
- **Flexible parameter selection** (CDI, density, peak properties)

### Visualization
- **Multi-parameter trend plots** with confidence bands
- **Color-coded trend significance** (increasing/decreasing/stable/uncertain)
- **Interactive parameter display** with statistical metrics
- **Publication-ready plots** with customizable aesthetics

### Export Capabilities
- **CSV export** of trend statistics and confidence intervals
- **Detailed data export** including residuals and prediction intervals
- **High-resolution plot export** (PNG, PDF, SVG)
- **Comprehensive metadata** preservation

## 🚀 Quick Start

### 1. Load Your Data
```python
# Start the density analysis GUI
python density_gui_launcher.py

# Or use the demo with simulated data
python demo_trend_analysis.py
```

### 2. Prepare Batch Data
1. Load your spectra in the **Standard Analysis** tab
2. Run **batch density analysis** first
3. This creates the dataset for trend analysis

### 3. Configure Trend Analysis
Switch to the **Trend Analysis** tab and configure:

- **Sequence Order**: How to sort your spectra
  - `Filename (alphabetical)`: Standard alphabetical sorting
  - `Filename (numerical)`: Extract numbers from filenames
  - `Timestamp (if available)`: Use file modification times
  - `Manual ordering`: Keep original order

- **Analysis Parameters**:
  - `Confidence Level`: 50% to 99% (default: 95%)
  - `Smoothing Window`: 1-20 points (default: 3)
  - `Trend Sensitivity`: 0.01-0.50 slope threshold (default: 0.05)

- **Parameters to Analyze**:
  - ✓ CDI (Crystalline Density Index)
  - ✓ Specialized Density
  - ✓ Peak Height
  - ✓ Peak Width

### 4. Run Analysis
Click **"Analyze Trends"** to perform:
- Linear regression with confidence intervals
- Statistical significance testing
- Trend classification
- Comprehensive visualization

## 📊 Understanding Results

### Trend Classifications
The system automatically classifies trends as:

- **🟢 INCREASING**: Significant positive slope (p < 0.05, slope > sensitivity)
- **🔴 DECREASING**: Significant negative slope (p < 0.05, slope < -sensitivity)
- **🔵 STABLE**: No significant trend (|slope| < sensitivity)
- **🟠 UNCERTAIN**: Slope detected but not statistically significant

### Statistical Metrics
For each parameter, you get:

```
🔍 CDI (Crystalline Density Index):
   • Trend: INCREASING
   • Slope: 0.012345 ± 0.001234
   • R²: 0.8567
   • P-value: 0.0023
   • 95% CI: [0.009876, 0.014567]
   • ↗️ Significant INCREASING trend detected
```

### Confidence Intervals
- **Slope Confidence Interval**: Range of likely slope values
- **Prediction Intervals**: Uncertainty bands around fitted line
- **Customizable confidence levels**: Typically 95% (α = 0.05)

## 🔬 Scientific Applications

### Material Processing Studies
```python
# Example: Annealing study
sequence_order = "Timestamp (if available)"  # Time-based analysis
parameters = ["CDI", "Specialized Density"]  # Track crystallization
confidence_level = 0.99  # High confidence for publication
```

### Spatial Analysis
```python
# Example: Across-sample mapping
sequence_order = "Filename (numerical)"  # Position-based ordering
parameters = ["CDI", "Peak Height", "Peak Width"]  # Full characterization
smoothing_window = 5  # Smooth spatial variations
```

### Quality Control
```python
# Example: Process monitoring
sequence_order = "Filename (alphabetical)"  # Production order
trend_sensitivity = 0.02  # Sensitive drift detection
parameters = ["Specialized Density"]  # Key quality metric
```

## 📈 Advanced Analysis Options

### Sequence Ordering Strategies

#### Numerical Filename Extraction
For files like `sample_001.txt`, `sample_002.txt`:
- Automatically extracts numbers for proper ordering
- Handles zero-padding and mixed formats
- Fallback to alphabetical if no numbers found

#### Timestamp Analysis
For time-series studies:
- Uses file modification times when available
- Extracts dates from filenames (YYYY-MM-DD, YYYYMMDD)
- Enables temporal trend analysis

#### Manual Ordering
For custom sequences:
- Preserves original data loading order
- Useful for pre-sorted datasets
- Maintains experimental sequence

### Statistical Considerations

#### Sample Size Requirements
- **Minimum**: 3 spectra (basic analysis)
- **Recommended**: 10+ spectra (reliable statistics)
- **Optimal**: 20+ spectra (robust confidence intervals)

#### Trend Sensitivity Tuning
- **High sensitivity (0.01-0.02)**: Detect subtle changes
- **Medium sensitivity (0.03-0.05)**: Balanced detection
- **Low sensitivity (0.10+)**: Only major trends

#### Confidence Level Selection
- **90%**: Quick screening, broader intervals
- **95%**: Standard scientific reporting
- **99%**: Conservative, narrow intervals

## 🛠️ Technical Implementation

### Statistical Methods
```python
# Linear regression with scipy.stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Confidence intervals using t-distribution
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha/2, n - 2)
confidence_interval = slope ± t_critical * std_err

# Prediction intervals
pred_se = sqrt(mse * (1 + 1/n + (x - x_mean)²/sxx))
prediction_interval = fitted ± t_critical * pred_se
```

### Smoothing Algorithm
```python
# Savitzky-Golay filter for noise reduction
if window_size > 1:
    smoothed = signal.savgol_filter(data, window_size, polynomial_order)
```

### Trend Detection Logic
```python
# Multi-criteria trend classification
if abs(slope) < sensitivity:
    trend = 'stable'
elif slope > sensitivity and p_value < 0.05:
    trend = 'increasing'
elif slope < -sensitivity and p_value < 0.05:
    trend = 'decreasing'
else:
    trend = 'uncertain'  # Slope present but not significant
```

## 📁 Export Formats

### Summary Results (CSV)
```csv
parameter,trend,slope,r_squared,p_value,ci_lower,ci_upper,n_spectra
CDI,increasing,0.012345,0.8567,0.0023,0.009876,0.014567,25
Specialized Density,increasing,0.045678,0.7834,0.0045,0.032145,0.059211,25
```

### Detailed Data (CSV)
```csv
sequence_index,filename,CDI_raw,CDI_smoothed,CDI_fitted,CDI_residual,CDI_pred_lower,CDI_pred_upper
0,spectrum_001.txt,0.345,0.347,0.342,0.003,0.320,0.364
1,spectrum_002.txt,0.356,0.358,0.354,0.002,0.332,0.376
```

## 🎨 Visualization Features

### Multi-Panel Layout
- Automatic subplot arrangement based on selected parameters
- Color-coded trend significance
- Statistical information overlay
- Confidence interval shading

### Customizable Elements
- Point colors and sizes
- Line styles and transparency
- Grid and axis formatting
- Title and label customization

### Interactive Features
- Zoom and pan capabilities
- Data point tooltips
- Legend toggle
- Export options

## 🔧 Troubleshooting

### Common Issues

#### "No batch data available"
**Solution**: Run batch density analysis first in the Standard Analysis tab

#### "Insufficient data for trend analysis"
**Solution**: Ensure you have at least 3 spectra with valid density results

#### "No significant trends detected"
**Solutions**:
- Reduce trend sensitivity threshold
- Check sequence ordering (might be scrambled)
- Verify data quality and range

#### "Statistical calculation errors"
**Solutions**:
- Check for NaN or infinite values in data
- Ensure data ranges are reasonable
- Try different smoothing window sizes

### Performance Optimization

#### Large Datasets (>50 spectra)
- Increase smoothing window for noise reduction
- Consider subsampling for initial exploration
- Use timestamp ordering for time-series data

#### Noisy Data
- Increase smoothing window (5-10 points)
- Reduce trend sensitivity
- Check original spectral quality

## 📚 Example Workflows

### Crystallization Study
```python
# Configuration for crystallization monitoring
confidence_level = 0.95
smoothing_window = 3
trend_sensitivity = 0.03
sequence_order = "Timestamp (if available)"
parameters = ["CDI", "Specialized Density"]

# Expected results: Increasing trends with high confidence
```

### Spatial Mapping
```python
# Configuration for spatial trend analysis
confidence_level = 0.90  # Slightly relaxed for exploration
smoothing_window = 5     # Smooth spatial noise
trend_sensitivity = 0.05
sequence_order = "Filename (numerical)"  # Position-based
parameters = ["CDI", "Peak Height"]

# Look for spatial gradients or zones
```

### Quality Control
```python
# Configuration for process monitoring
confidence_level = 0.99  # High confidence for decisions
smoothing_window = 2     # Minimal smoothing
trend_sensitivity = 0.02 # Sensitive detection
sequence_order = "Filename (alphabetical)"  # Production order
parameters = ["Specialized Density"]

# Detect process drift early
```

## 🏆 Best Practices

### Data Preparation
1. **Consistent naming**: Use systematic filename conventions
2. **Quality control**: Remove obviously bad spectra first
3. **Sequence verification**: Verify correct ordering before analysis

### Statistical Analysis
1. **Choose appropriate confidence levels**: 95% for most applications
2. **Adjust sensitivity**: Start with default, then fine-tune
3. **Multiple parameters**: Analyze several parameters for comprehensive understanding

### Interpretation
1. **Consider physical meaning**: Statistical significance ≠ practical significance
2. **Check assumptions**: Linear trends may not capture all behavior
3. **Validate results**: Cross-check with other analytical methods

### Reporting
1. **Document parameters**: Include all analysis settings
2. **Report confidence intervals**: Not just point estimates
3. **Show raw data**: Include scatter plots with fitted lines
4. **Statistical metrics**: Report R², p-values, and sample sizes

## 🆕 Version History

### v2.0.0 - Trend Analysis Release
- **New**: Complete trend analysis module with confidence intervals
- **New**: Multiple sequence ordering options
- **New**: Statistical significance testing
- **New**: Advanced visualization with confidence bands
- **New**: Comprehensive export capabilities
- **Enhanced**: Tab-based interface for better organization
- **Enhanced**: Integration with existing batch processing

## 🤝 Contributing

Found a bug or have a feature request? Please report issues or contribute improvements to help make this tool even better for the Raman spectroscopy community!

## 📖 References

1. **Linear Regression Theory**: Draper & Smith, "Applied Regression Analysis"
2. **Confidence Intervals**: Montgomery et al., "Introduction to Linear Regression Analysis"
3. **Time Series Analysis**: Box & Jenkins, "Time Series Analysis"
4. **Savitzky-Golay Filtering**: Savitzky & Golay, "Smoothing and Differentiation of Data"

---

*Happy analyzing! 🔬📊* 