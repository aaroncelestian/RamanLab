# Crystal Structure Visualization Widget

## Overview

The Crystal Structure Widget provides comprehensive 3D visualization and analysis of crystal structures loaded from CIF (Crystallographic Information File) format files. It uses crystal coordinate systems rather than Cartesian coordinates for authentic crystallographic representation.

## Features

### File Operations
- **Load CIF File**: Import crystal structures from standard CIF format files
- **Export Structure**: Export structure data and analysis results
- **Save View**: Save current 3D visualization as high-resolution images

### 3D Visualization
- **Crystal Coordinates**: Displays structures in crystal coordinate system (a, b, c axes)
- **Interactive Rotation**: Manual control with elevation and azimuth sliders
- **Auto-Rotation**: Automatic rotation for presentations
- **View Presets**: Quick views along crystal axes (a, b, c)

### Structural Analysis
- **Bond Calculation**: Automatic bond detection using pymatgen's CrystalNN algorithm
- **Supercell Generation**: Create supercells up to 5x5x5 for extended structures
- **Bond Distance Analysis**: Configurable bond cutoff distances

### Visualization Controls
- **Atom Scaling**: Adjustable atom sizes for clarity
- **Bond Thickness**: Customizable bond line thickness
- **Color Schemes**: Element-based coloring with extensible schemes
- **Display Options**: Toggle unit cell outline, crystal axes, and bonds

## Usage

### Basic Loading
1. Click "Load CIF File" in the File Operations panel
2. Select a CIF file from your system
3. The structure will automatically load and display in 3D
4. Bonds are calculated automatically using default parameters

### Customizing Display
- **Atom Size**: Use the atom size slider to make atoms larger/smaller
- **Bond Thickness**: Adjust bond line thickness for better visibility  
- **Show/Hide Elements**: Toggle unit cell, axes, and bonds on/off
- **Color Scheme**: Choose from element-based coloring options

### Interactive Navigation
- **Manual Rotation**: Use elevation and azimuth sliders for precise control
- **View Presets**: Click a/b/c axis buttons for standard crystallographic views
- **Auto-Rotate**: Enable for continuous rotation (useful for presentations)
- **Reset View**: Return to default viewing angle

### Advanced Analysis
- **Bond Cutoff**: Adjust maximum bond distance for bond detection
- **Supercell**: Create larger unit cells to see extended structure patterns
- **Statistics**: View real-time atom counts, bond statistics, and lattice parameters

## Technical Details

### Dependencies
- **pymatgen**: Required for CIF parsing and crystallographic analysis
- **matplotlib**: For 3D visualization with interactive controls
- **PySide6**: Qt6 interface framework
- **numpy**: Numerical computations

### Crystal Coordinate System
The widget uses authentic crystal coordinates where:
- **a-axis**: First lattice vector direction
- **b-axis**: Second lattice vector direction  
- **c-axis**: Third lattice vector direction

This preserves the true crystallographic relationships unlike Cartesian representations.

### Bond Calculation
Bonds are calculated using pymatgen's CrystalNN (Crystal Neural Network) algorithm, which:
- Considers local coordination environments
- Uses machine learning for accurate bond detection
- Respects crystallographic constraints
- Provides distance and bond type information

## Example Usage with Anatase

The included `anatase.cif` file demonstrates TiO2 (titanium dioxide) in the anatase crystal structure:

```
Formula: TiO2
Space Group: I 41/a m d  
Crystal System: Tetragonal
Lattice Parameters:
  a = 3.7845 Å
  b = 3.7845 Å  
  c = 9.5143 Å
  α = β = γ = 90°
```

Key features to explore:
- **Ti-O Bonds**: Octahedral coordination of titanium
- **Layer Structure**: Characteristic layered arrangement
- **Supercell**: Create 2x2x1 supercell to see extended structure
- **c-axis View**: Shows the layered nature clearly

## Integration

The widget integrates seamlessly with the main Raman Polarization Analyzer:
- **Structure Data**: Shares crystal system information with other tabs
- **Bond Information**: Provides coordination data for Raman analysis
- **Orientation**: Crystal structure informs polarization calculations

## Troubleshooting

### Common Issues
1. **"pymatgen not available"**: Install with `pip install pymatgen`
2. **CIF Loading Errors**: Ensure CIF file is valid and properly formatted
3. **No Bonds Displayed**: Adjust bond cutoff distance or check structure validity
4. **Slow Rendering**: Reduce supercell size or atom scaling for complex structures

### Performance Tips
- Start with 1x1x1 supercell for initial viewing
- Use smaller atom scales for dense structures
- Disable bonds temporarily for very large structures
- Save complex views as images rather than keeping them rendered

## Future Enhancements

Planned features include:
- Additional color schemes (coordination number, oxidation state)
- Thermal ellipsoid representation
- Miller plane visualization
- Export to common 3D formats
- Advanced bond filtering options 