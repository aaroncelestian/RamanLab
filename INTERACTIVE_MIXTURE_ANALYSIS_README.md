# RamanLab Interactive Mixture Analysis

## Overview

The Interactive Mixture Analysis is a sophisticated, user-guided approach to analyzing Raman spectra containing multiple mineral phases. Unlike fully automated methods, this system leverages expert knowledge through an iterative workflow that builds synthetic spectra component by component.

## Key Features

- **Expert-Guided Analysis**: User selects matching peaks rather than relying on automated algorithms
- **Real-time Visualization**: Live overlay of user data with database references
- **Interactive Peak Selection**: Click-to-select peaks in overlay plots
- **Pseudo-Voigt Peak Fitting**: High-quality peak fitting for synthetic spectrum generation
- **Iterative Refinement**: Search residuals to find additional components
- **Comprehensive Tracking**: Full history of iterations and fit quality evolution

## Why This Approach?

This interactive method addresses several limitations of fully automated mixture analysis:

1. **Limited Training Data**: Machine learning requires large datasets that don't exist for Raman mineral databases
2. **Expert Knowledge**: Leverages geological and spectroscopic expertise in component identification
3. **Unbiased Analysis**: Reduces false positives by requiring user confirmation of matches
4. **Transparency**: Every decision is visible and controllable by the user
5. **Quality Control**: Real-time feedback on fit quality and residual evolution

## Installation & Launch

### Prerequisites
```bash
pip install PySide6 matplotlib numpy scipy scikit-learn
```

### Launch Methods

**Option 1: Direct Launch**
```bash
python launch_interactive_mixture_analysis.py
```

**Option 2: From RamanLab**
- Add to main RamanLab interface as a module launcher
- Access through Tools → Interactive Mixture Analysis

## Workflow

### Step 1: Data Loading
1. **Load Spectrum Data**: Import your Raman spectrum file (.txt, .csv, .dat)
2. **Use Demo Data**: Try the built-in demo mixture (Quartz + Calcite + Feldspar)

### Step 2: Initial Database Search
1. Click **"Search Database"** to find the top 10 matches
2. Review correlation coefficients for each match
3. Select the best match from the results list

### Step 3: Interactive Peak Selection
1. **Overlay Visualization**: Your data (black) overlays with selected reference (red)
2. **Click to Select**: Click on peaks in the overlay plot where your data matches the reference
3. **Visual Feedback**: Selected peaks appear as blue dashed lines
4. **Clear if Needed**: Use "Clear Selected Peaks" to start over

### Step 4: Peak Fitting
1. Click **"Fit Selected Peaks"** to fit pseudo-Voigt profiles
2. System fits peaks at your selected positions
3. Fitted component is added to synthetic spectrum
4. Residual is automatically calculated and updated

### Step 5: Residual Analysis
1. **Search Residual**: Click "Search on Residual" to find matches to remaining signal
2. **Repeat Process**: Select new match, pick peaks, fit, and continue
3. **Iterative Refinement**: Each iteration improves the overall fit

### Step 6: Analysis Finalization
1. **Monitor Progress**: Watch R² evolution and cumulative synthetic spectrum
2. **Components Summary**: Review all fitted components in the summary table
3. **Finalize**: Click "Finalize Analysis" for complete results

## Interface Layout

### Left Panel: Controls
- **Data Loading**: Import or use demo data
- **Search Controls**: Database search and residual search
- **Search Results**: Top 10 matches with correlation scores
- **Peak Selection**: Selected peaks display and controls
- **Analysis Control**: Iteration tracking and finalization
- **Components Summary**: Table of all fitted components
- **Status Log**: Real-time analysis messages

### Right Panel: Interactive Plots

**Top Left: Spectrum Overlay & Peak Selection**
- User data (black line)
- Selected reference (red line)  
- Selected peaks (blue dashed lines)
- Click to select peaks

**Top Right: Current Residual**
- Shows remaining signal after subtracting fitted components
- Used for subsequent database searches

**Bottom Left: Cumulative Synthetic Spectrum**
- Original data (black line)
- Built-up synthetic spectrum (blue line)
- Shows overall fit quality

**Bottom Right: Fit Quality Evolution**
- R² value progression through iterations
- Visual feedback on analysis improvement

## Analysis Strategy

### Best Practices

1. **Start with Major Phases**: Look for strongest, most obvious matches first
2. **Peak Selection Quality**: Choose clear, unambiguous peaks that match well
3. **Residual Monitoring**: Watch residual plot to ensure meaningful signal remains
4. **Iteration Limit**: Typically 3-5 components are sufficient for most mixtures
5. **Final R² Target**: Aim for R² > 0.95 for high-quality fits

### Common Workflows

**Simple Binary Mixture**:
1. Search → Select major phase → Fit 2-3 peaks
2. Search residual → Select minor phase → Fit 1-2 peaks
3. Finalize

**Complex Multi-phase Mixture**:
1. Search → Select dominant phase → Fit main peaks
2. Search residual → Select secondary phase → Fit peaks
3. Search residual → Select tertiary phase → Fit peaks
4. Continue until R² > 0.95 or residual is noise-level
5. Finalize

## Technical Details

### Peak Fitting Algorithm
- **Profile Type**: Pseudo-Voigt (50:50 Gaussian:Lorentzian mix)
- **Parameters**: [amplitude, center, σ_gaussian, γ_lorentzian] per peak
- **Constraints**: Physical bounds on parameters (non-negative amplitudes, reasonable widths)
- **Optimization**: Scipy curve_fit with bounded least squares

### Database Integration
- **RamanLab Database**: Automatically loads main RamanLab mineral database
- **Fallback**: Test database if RamanLab database unavailable
- **Search Algorithm**: Correlation-based matching with wavenumber interpolation
- **Performance**: Optimized for interactive response times

### Statistical Metrics
- **R² Calculation**: Coefficient of determination for fit quality
- **RMS Residual**: Root mean square of fitting residuals
- **Component Tracking**: Individual component R² and peak count
- **Iteration History**: Complete record of analysis progression

## Output & Results

### Analysis Results Include:
- **Component List**: All identified minerals with fit statistics
- **Peak Positions**: Fitted peak centers for each component
- **Fit Quality**: Individual and overall R² values
- **Synthetic Spectrum**: Complete built-up synthetic spectrum
- **Residual Analysis**: Final residual for quality assessment

### Export Capabilities:
- **Results Summary**: Text summary of complete analysis
- **Plot Export**: All plots can be saved as images
- **Component Data**: Access to fitted parameters and spectra

## Integration with RamanLab

### Database Compatibility
- Uses existing RamanLab mineral database (RamanLab_Database_20250602.pkl)
- Compatible with mineral_modes.pkl for vibrational mode information
- Maintains consistency with other RamanLab modules

### Styling Integration
- Applies RamanLab matplotlib configuration for consistent appearance
- Uses RamanLab color schemes and font choices
- Matches styling of other RamanLab interfaces

### Future Enhancements
- **Batch Processing**: Apply interactive templates to multiple spectra
- **Component Library**: Save successful component fits for reuse
- **Uncertainty Quantification**: Bootstrap analysis of component confidence
- **Machine Learning**: Train models on successful interactive analyses

## Troubleshooting

### Common Issues

**"No matches found"**
- Check wavenumber range overlap with database
- Ensure spectrum is properly normalized
- Try searching with broader correlation thresholds

**"Peak fitting failed"**
- Select fewer peaks (2-3 recommended)
- Ensure selected peaks are clear and well-defined
- Check that peaks actually match between spectra

**"Poor fit quality"**
- More iterations may be needed
- Check residual for remaining signal
- Verify peak selections are accurate

**"Interface not responding"**
- Large databases may take time to search
- Progress dialog shows search status
- Cancel and restart if needed

### Performance Tips
- **Demo Data**: Use demo data to learn the interface
- **Peak Selection**: Be selective - quality over quantity
- **Residual Monitoring**: Stop when residual approaches noise level
- **Component Limit**: Rarely need more than 5 components

## Example Analysis

### Demo Mixture: Quartz + Calcite + Feldspar

1. **Load Demo Data**: Click "Use Demo Data"
2. **Search Database**: Should find Quartz as top match (correlation ~0.8)
3. **Select Quartz**: Click on Quartz in results list
4. **Pick Peaks**: Click on peaks at ~465 cm⁻¹ and ~207 cm⁻¹
5. **Fit Peaks**: R² should be ~0.6-0.7
6. **Search Residual**: Should find Calcite as top match
7. **Select Calcite**: Click on Calcite in results
8. **Pick Peaks**: Click on peaks at ~1086 cm⁻¹ and ~712 cm⁻¹
9. **Fit Peaks**: Cumulative R² should improve to ~0.8-0.9
10. **Search Residual**: May find Feldspar or other minor components
11. **Repeat**: Continue until satisfied with fit quality
12. **Finalize**: Review complete analysis results

Expected final result: 3 components with overall R² > 0.9

---

## Support

For questions or issues with the Interactive Mixture Analysis:
1. Check this README for common solutions
2. Review the status log for error messages
3. Try the demo data to verify functionality
4. Ensure all dependencies are properly installed

This interactive approach represents a significant advance in mixture analysis capabilities, combining the power of automated database searching with the expertise and judgment of human operators. 