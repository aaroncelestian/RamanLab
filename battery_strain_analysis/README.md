# LiMn2O4 Battery Strain Analysis System

A comprehensive implementation of chemical strain analysis for LiMn2O4 spinel battery materials undergoing H/Li exchange. This system provides advanced Raman spectroscopy analysis capabilities to track structural changes, phase transitions, and strain evolution in battery materials.

## Overview

This package extends the base `chemical_strain_enhancement.py` to provide specialized analysis for LiMn2O4 battery materials, focusing on:

- **H/Li exchange effects** - Track composition changes during battery operation
- **Jahn-Teller distortion** - Monitor Mn³⁺ formation and associated structural distortions
- **Strain tensor analysis** - Quantify mechanical strain in the crystal lattice
- **Time series analysis** - Process time-resolved Raman spectroscopy data
- **Phase transition detection** - Identify structural phase changes
- **Comprehensive visualization** - Generate publication-quality plots and reports

## Key Features

### 🔬 **Advanced Raman Analysis**
- Spinel-specific mode definitions for LiMn2O4
- Composition-dependent Grüneisen parameters
- Jahn-Teller coupling effects
- Chemical disorder broadening
- Temperature-dependent frequency shifts

### ⚡ **Battery-Specific Capabilities**
- H/Li exchange tracking
- Mn³⁺/Mn⁴⁺ ratio determination
- Electrochemical synchronization
- State-of-charge correlation
- Cycling degradation analysis

### 📊 **Comprehensive Visualization**
- Time series strain evolution plots
- 3D strain tensor visualization
- Peak tracking and mode splitting
- Phase transition mapping
- Correlation analysis

### 🔄 **Time Series Processing**
- Multi-format data loading (.txt, .csv, .dat)
- Background subtraction and smoothing
- Noise filtering and quality control
- Electrochemical data synchronization
- Automated peak detection and assignment

## Installation & Dependencies

```bash
# Required packages
pip install numpy scipy matplotlib pandas seaborn
```

Make sure you have the base `chemical_strain_enhancement.py` file in the parent directory.

## Quick Start

```python
from battery_strain_analysis import LiMn2O4StrainAnalyzer, StrainVisualizer

# Initialize analyzer
analyzer = LiMn2O4StrainAnalyzer(temperature=298.0)

# Set battery composition
analyzer.set_battery_state(li_content=0.8, h_content=0.2)

# Analyze a single spectrum
results = analyzer.analyze_spectrum(frequencies, intensities)

# Visualize results
visualizer = StrainVisualizer()
fig = visualizer.plot_strain_tensor_3d(results['strain_tensor'])
```

## Module Structure

```
battery_strain_analysis/
├── __init__.py                 # Package initialization
├── spinel_modes.py            # LiMn2O4 Raman mode definitions
├── limn2o4_analyzer.py        # Main strain analysis class
├── time_series_processor.py   # Time series data handling
├── strain_visualization.py    # Plotting and visualization
├── demo_limn2o4_analysis.py   # Complete demo script
└── README.md                  # This file
```

## Core Classes

### 1. `SpinelRamanModes`
Defines Raman active modes for LiMn2O4 spinel structure:
- **A1g**: Symmetric breathing mode (625 cm⁻¹)
- **Eg**: Octahedral bending mode (580 cm⁻¹) 
- **T2g**: Framework stretching modes (480, 430 cm⁻¹)
- **Li_O**: Li-O stretch (280 cm⁻¹)
- **Disorder**: Disorder-activated mode (515 cm⁻¹)

### 2. `LiMn2O4StrainAnalyzer`
Main analysis class providing:
- Strain tensor fitting
- Composition determination
- Jahn-Teller parameter calculation
- Phase transition detection
- Time series analysis

### 3. `TimeSeriesProcessor`
Handles time-resolved data:
- Multi-format file loading
- Data preprocessing
- Electrochemical synchronization
- Quality control

### 4. `StrainVisualizer`
Comprehensive visualization tools:
- Time series overview plots
- 3D strain tensor visualization
- Peak evolution tracking
- Phase transition maps
- Correlation matrices

## Usage Examples

### Time Series Analysis

```python
# Load and process time series data
from battery_strain_analysis import TimeSeriesProcessor, LiMn2O4StrainAnalyzer

# Initialize components
processor = TimeSeriesProcessor("data/time_series/")
analyzer = LiMn2O4StrainAnalyzer()

# Load data
processor.load_time_series("*.txt")
time_series_data = processor.get_time_series_data()

# Run analysis
results = analyzer.analyze_time_series(time_series_data)

# Results contain:
# - strain_evolution: Strain tensor vs time
# - composition_evolution: H/Li content vs time  
# - jt_evolution: Jahn-Teller parameter vs time
# - peak_tracking: Peak frequencies vs time
# - phase_transitions: Phase indicators vs time
```

### Visualization and Reporting

```python
from battery_strain_analysis import StrainVisualizer

visualizer = StrainVisualizer()

# Create comprehensive report
visualizer.create_analysis_report(results, "strain_analysis_report/")

# Individual plots
overview_fig = visualizer.plot_time_series_overview(results)
strain_3d_fig = visualizer.plot_strain_tensor_3d(final_strain)
```

## Physical Interpretation

### Strain Tensor Components
- **ε₁₁, ε₂₂, ε₃₃**: Normal strains along crystal axes
- **ε₁₂, ε₁₃, ε₂₃**: Shear strains
- **Hydrostatic strain**: Volume change
- **Deviatoric strain**: Shape change

### Jahn-Teller Effects
- **Mn³⁺ formation**: Creates d⁴ electronic configuration
- **Tetragonal distortion**: Elongation of MnO₆ octahedra
- **Mode splitting**: Eg → A1g + B1g
- **Frequency shifts**: Generally softening with increasing distortion

### Composition Indicators
- **Peak frequency shifts**: Composition-dependent bond lengths
- **Intensity changes**: Li-O modes weaken with delithiation
- **Broadening**: Chemical disorder increases with mixed occupancy
- **New mode activation**: Disorder-activated modes appear

## Demo Script

Run the complete demonstration:

```bash
cd battery_strain_analysis
python demo_limn2o4_analysis.py
```

This will:
1. Generate synthetic time series data
2. Run complete strain analysis
3. Create visualizations
4. Generate a comprehensive report
5. Save all results to `limn2o4_analysis_results/`

## Expected Output

The analysis produces:
- **Strain evolution plots** showing tensor components vs time
- **Composition tracking** showing H/Li content changes
- **Peak evolution** showing frequency shifts and broadening
- **3D strain visualization** showing strain ellipsoids
- **Phase transition maps** identifying structural changes
- **Quantitative results** including strain rates and transition times

## Experimental Data Format

The system accepts various file formats:

### Single Spectrum Files (.txt, .csv)
```
# Wavenumber (cm⁻¹)    Intensity
200.0                  1250.5
201.0                  1245.2
...
```

### Time Series Files
- Multiple files: `spectrum_t000.txt`, `spectrum_t060.txt`, etc.
- With time in filename or as column in data
- Automatic time extraction from filenames

### Electrochemical Data (.csv)
```
time,voltage,current
0.0,4.15,0.1
60.0,4.10,0.1
...
```

## Applications

This system is designed for:
- **Battery research** - Understanding degradation mechanisms
- **Materials science** - Studying phase transitions
- **Electrochemistry** - Correlating structure with performance
- **Quality control** - Monitoring production consistency
- **Method development** - Advanced Raman analysis techniques

## Scientific Background

### LiMn2O4 Spinel Structure
- Space group: Fd3m (cubic)
- Li in tetrahedral sites (8a)
- Mn in octahedral sites (16d)
- O in 32e sites

### H/Li Exchange Process
1. **Delithiation**: Li⁺ removal → Mn³⁺ → Mn⁴⁺
2. **Protonation**: H⁺ insertion → charge compensation
3. **Structural changes**: Jahn-Teller distortion → symmetry breaking
4. **Phase evolution**: Cubic → tetragonal → phase separation

### Strain Mechanisms
- **Chemical strain**: Composition-dependent lattice parameters
- **Coherency strain**: Lattice mismatch between phases
- **Thermal strain**: Temperature-dependent expansion
- **Defect strain**: Point defect relaxation

## Contributing

To extend the system:
1. Add new modes to `spinel_modes.py`
2. Implement new analysis methods in `limn2o4_analyzer.py`
3. Create additional visualization tools in `strain_visualization.py`
4. Update tests and documentation

## References

1. Thackeray, M. M., et al. "Lithium insertion into manganese spinels." *Materials Research Bulletin* 18.4 (1983): 461-472.
2. Ammundsen, B., et al. "Local structure and vertex sharing in LiMn2O4 electrode materials." *Journal of the Electrochemical Society* 149.4 (2002): A431-A436.
3. Julien, C. M., et al. "Raman spectra of birnessite manganese dioxides." *Solid State Ionics* 159.3-4 (2003): 345-356.

## License

This project is part of the RamanLab package. See the main LICENSE file for details.

## Contact

For questions or contributions, please contact the RamanLab development team.

---

*Built with ❤️ for the battery research community* 