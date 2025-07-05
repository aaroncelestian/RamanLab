# Raman Polarization Analyzer - Qt6 Implementation

## Overview

The Raman Polarization Analyzer is a comprehensive Qt6-based application for analyzing Raman spectra with a focus on polarization-dependent measurements and crystal orientation determination. This tool integrates seamlessly with the main RamanLab application and can also be used as a standalone analyzer.

## Features

### Core Functionality
- **Spectrum Analysis**: Load, visualize, and analyze Raman spectra
- **Peak Fitting**: Interactive peak selection and fitting with multiple peak shapes (Lorentzian, Gaussian, Voigt)
- **Polarization Analysis**: Calculate depolarization ratios and polarization characteristics
- **Crystal Structure**: Load CIF files and extract crystallographic information
- **Tensor Analysis**: Calculate and visualize Raman tensors for each vibrational mode
- **Orientation Optimization**: Multi-stage optimization for crystal orientation determination

### Integration Features
- **Main App Integration**: Launches from the Advanced tab of `raman_analysis_app_qt6.py`
- **Automatic Data Transfer**: Current spectrum from main app is automatically loaded
- **Full Database Integration**: Access to comprehensive mineral modes database with 461+ calculated spectra
- **Multiple Database Support**: Supports mineral_modes.pkl, RamanLab_Database_20250602.pkl, and other formats
- **Professional CIF Support**: Uses pymatgen for advanced crystallographic file parsing

## Usage

### Launch from Main Application
1. Open `raman_analysis_app_qt6.py`
2. Load a spectrum in the main application
3. Go to the "Advanced" tab
4. Click "Polarization Analysis" button
5. The polarization analyzer will open with your spectrum pre-loaded

### Standalone Usage
```bash
python raman_polarization_analyzer_qt6.py
```

## Workflow

### 1. Spectrum Analysis Tab
- **Load Spectrum**: Import spectrum files (TXT, CSV, DAT formats)
- **Enhanced Database Search**: Search through 461+ calculated mineral spectra from the mineral modes database
- **Multi-field Search**: Search by mineral name, chemical formula, crystal system, or space group
- **Crystal System Filter**: Filter search results by crystal system (Cubic, Hexagonal, etc.)
- **Detailed Information**: View formula, crystal system, and number of Raman modes for each mineral
- **Database Refresh**: Reload the mineral database to get latest updates
- **Visualization**: View current and imported spectra with automatic normalization

### 2. Peak Fitting Tab
- **Peak Selection**: Toggle peak selection mode and click on peaks
- **Fitting Options**: Choose between Lorentzian, Gaussian, or Voigt peak shapes
- **Reference Mineral**: Select reference mineral for comparison
- **Interactive Fitting**: Click on spectrum to select peaks, then fit with chosen shape

### 3. Polarization Tab
- **Crystal System**: Select appropriate crystal system (Cubic, Hexagonal, etc.)
- **Polarization Angles**: Set incident and scattered polarization angles
- **Analysis**: Calculate depolarization ratios based on crystal system and peak properties

### 4. Crystal Structure Tab
- **CIF Loading**: Load crystal structure from CIF files
- **Automatic Detection**: Crystal system is automatically determined from lattice parameters
- **Structure Information**: View detailed crystallographic information
- **Professional Parsing**: Uses pymatgen when available, falls back to simplified parser

### 5. Tensor Analysis Tab
- **Tensor Calculation**: Generate Raman tensors for each fitted peak
- **Crystal System Dependence**: Tensors calculated based on crystal symmetry
- **Property Analysis**: Calculate tensor eigenvalues, anisotropy, and invariants
- **Detailed Results**: View tensor matrices and properties in organized dialog

### 6. Orientation Optimization Tab
- **Multi-Stage Optimization**:
  - Stage 1: Peak intensity analysis for initial estimates
  - Stage 2: Angular dependence fitting around preliminary orientation
  - Stage 3: Local optimization refinement using scipy.minimize
- **Progress Tracking**: Real-time progress indication with cancellation option
- **Comprehensive Results**: Detailed optimization metrics and convergence information

## Implementation Details

### Peak Fitting
- **Interactive Selection**: Mouse-click interface for peak selection
- **Multiple Shapes**: Support for Lorentzian, Gaussian, and simplified Voigt profiles
- **Robust Fitting**: Error handling and parameter validation

### Polarization Analysis
```python
# Depolarization ratio calculation based on crystal system
if crystal_system == "Cubic":
    depol_ratio = 0.0  # Fully polarized A1g modes
elif crystal_system == "Hexagonal":
    if freq < 500:  # E2g modes
        depol_ratio = 0.75  # Depolarized
    else:  # A1g modes
        depol_ratio = 0.0  # Polarized
```

### Tensor Generation
- **Symmetry-Based**: Tensors generated according to crystal system symmetry
- **Physical Properties**: Calculation of anisotropy, trace, determinant
- **Eigenvalue Analysis**: Complete eigenvalue/eigenvector decomposition

### CIF File Support
- **Professional Parsing**: Uses pymatgen when available for complete CIF support
- **Fallback Parser**: Simplified parser for basic lattice parameters
- **Automatic Classification**: Crystal system determination from lattice parameters

### Orientation Optimization
- **Grid Search**: Initial coarse grid search over Euler angles
- **Local Refinement**: Fine-tuning using scipy optimization
- **Multi-Stage Approach**: Progressive refinement for robust results

## Dependencies

### Required
- PySide6 (Qt6 bindings)
- numpy
- scipy
- matplotlib
- pathlib
- datetime

### Optional
- pymatgen (for professional CIF parsing)

## File Structure

```
raman_polarization_analyzer_qt6.py     # Main application file
├── Class: RamanPolarizationAnalyzerQt6
│   ├── UI Setup Methods
│   │   ├── init_ui()
│   │   ├── create_tabs()
│   │   └── setup_*_tab() methods
│   ├── Core Functionality
│   │   ├── load_mineral_database()
│   │   ├── generate_spectrum_from_mineral()
│   │   └── update_*_plot() methods
│   ├── File Operations
│   │   ├── load_spectrum()
│   │   ├── save_spectrum()
│   │   └── export_plot()
│   ├── Peak Fitting
│   │   ├── toggle_peak_selection()
│   │   ├── fit_peaks()
│   │   └── peak shape functions
│   ├── Polarization Analysis
│   │   ├── calculate_polarization()
│   │   └── show_polarization_results()
│   ├── Crystal Structure
│   │   ├── load_cif_file()
│   │   ├── load_cif_with_pymatgen()
│   │   └── load_cif_simple()
│   ├── Tensor Analysis
│   │   ├── calculate_raman_tensors()
│   │   ├── generate_raman_tensor()
│   │   └── analyze_tensor_properties()
│   └── Orientation Optimization
│       ├── run_orientation_optimization()
│       ├── optimize_stage1_intensity_analysis()
│       ├── optimize_stage2_angular_fitting()
│       └── optimize_stage3_refinement()
```

## Integration with Main Application

The polarization analyzer integrates with `raman_analysis_app_qt6.py` through the `launch_polarization_analysis()` method:

```python
def launch_polarization_analysis(self):
    """Launch polarization analysis tool."""
    try:
        from raman_polarization_analyzer_qt6 import RamanPolarizationAnalyzerQt6
        
        # Create and show the polarization analyzer window
        self.polarization_analyzer = RamanPolarizationAnalyzerQt6()
        
        # Transfer current spectrum if available
        if self.current_wavenumbers is not None and self.current_intensities is not None:
            spectrum_data = {
                'name': self.spectrum_file_path or 'Current Spectrum',
                'wavenumbers': self.current_wavenumbers,
                'intensities': self.processed_intensities or self.current_intensities,
                'source': 'main_app'
            }
            self.polarization_analyzer.current_spectrum = spectrum_data
            self.polarization_analyzer.update_spectrum_plot()
        
        self.polarization_analyzer.show()
```

## Future Enhancements

### Planned Features
- **3D Visualization**: Interactive 3D crystal structure and tensor visualization
- **Stress/Strain Analysis**: Advanced mechanical property analysis
- **Enhanced Tensor Physics**: More sophisticated tensor calculations
- **Machine Learning Integration**: AI-assisted peak identification and orientation optimization
- **Export Capabilities**: Enhanced data export and reporting features

### Technical Improvements
- **Performance Optimization**: Faster orientation optimization algorithms
- **Enhanced CIF Support**: Extended crystallographic file format support
- **Advanced Plotting**: Interactive 3D tensor visualization
- **Database Expansion**: Larger mineral database with more reference spectra

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install PySide6 numpy scipy matplotlib pymatgen
   ```

2. **CIF Loading Issues**: Install pymatgen for full CIF support
   ```bash
   pip install pymatgen
   ```

3. **Performance Issues**: For large datasets, consider reducing optimization grid resolution

### Error Messages
- "Please load a spectrum first": Load spectrum in Spectrum Analysis tab before proceeding
- "Please fit peaks first": Use Peak Fitting tab to select and fit peaks before advanced analysis
- "Please load crystal structure first": Load CIF file in Crystal Structure tab for orientation optimization

## Contributing

When contributing to this module:
1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new methods
3. Include error handling and user feedback
4. Test integration with the main application
5. Update this README with new features

## License

This module is part of the RamanLab suite and follows the same licensing terms as the main application. 