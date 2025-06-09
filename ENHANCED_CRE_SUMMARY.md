# Enhanced Cosmic Ray Elimination (CRE) with Shape Analysis

## Overview

The cosmic ray elimination system has been significantly enhanced to better distinguish between cosmic ray events and legitimate Raman peaks using sophisticated shape analysis. This addresses the critical issue where strong Raman peaks could be mistakenly identified as cosmic rays based on intensity thresholding alone.

## Key Problem Addressed

**Original Issue**: Cosmic ray elimination algorithms that rely solely on intensity thresholds can incorrectly remove strong Raman peaks, leading to loss of important spectral information.

**Solution**: Implement shape-based discrimination that analyzes the characteristic differences between cosmic rays and Raman peaks.

## Shape Analysis Characteristics

### Cosmic Rays Typically Have:
- **Very narrow peaks** (FWHM ≤ 3 data points)
- **Extreme sharpness** (high intensity-to-width ratio)
- **Symmetric shape** (low asymmetry factor)
- **Steep intensity gradients** on both sides
- **Isolated spikes** with dramatic intensity differences from neighbors

### Raman Peaks Typically Have:
- **Broader peaks** (FWHM > 3 data points)
- **Gradual intensity changes** (lower sharpness ratio)
- **May be asymmetric** (higher asymmetry factor)
- **Gentler intensity gradients**
- **More gradual transitions** to baseline

## New Configuration Parameters

The `CosmicRayConfig` class now includes advanced shape analysis parameters:

```python
@dataclass
class CosmicRayConfig:
    # Existing parameters
    enabled: bool = True
    absolute_threshold: float = 1500
    neighbor_ratio: float = 10.0
    apply_during_load: bool = True
    
    # New shape analysis parameters
    max_cosmic_fwhm: float = 3.0          # Maximum FWHM for cosmic rays (data points)
    min_sharpness_ratio: float = 5.0      # Minimum ratio of peak height to width
    max_asymmetry_factor: float = 0.3     # Maximum asymmetry for cosmic rays
    gradient_threshold: float = 200.0     # Minimum gradient for cosmic ray edges
    enable_shape_analysis: bool = True    # Enable/disable shape analysis
```

## Enhanced Detection Algorithm

The new algorithm uses a multi-criteria approach:

1. **Traditional Intensity Checks**: Absolute threshold and neighbor ratio analysis
2. **Shape Analysis**: Four-parameter shape characterization
3. **Decision Logic**: Requires at least 3 out of 4 shape criteria to classify as cosmic ray

### Shape Metrics Calculated:

1. **FWHM (Full Width at Half Maximum)**: Measures peak width
2. **Sharpness Ratio**: Peak height divided by width
3. **Asymmetry Factor**: Measures left-right symmetry
4. **Intensity Gradient**: Average gradient on peak edges

## User Interface Enhancements

### New UI Controls Added:
- **Shape Analysis Enable/Disable**: Toggle for the new functionality
- **Max FWHM**: Control for maximum cosmic ray width
- **Min Sharpness**: Control for minimum sharpness ratio
- **Max Asymmetry**: Control for maximum asymmetry tolerance
- **Min Gradient**: Control for minimum edge gradient
- **Shape Analysis Diagnosis**: Button to analyze peak shapes in current spectrum

### Enhanced Statistics Display:
- **False Positive Prevention Rate**: Shows how many peaks were saved from misclassification
- **Shape Analysis Configuration**: Displays current shape analysis parameters
- **Detailed Peak Classification**: Shows reasoning for each peak classification

## Benefits of the Enhanced System

### 1. **Improved Accuracy**
- Reduces false positive removal of strong Raman peaks
- Maintains high sensitivity to true cosmic ray events
- Uses multiple independent criteria for robust classification

### 2. **Better Preservation of Spectral Features**
- Strong, narrow Raman peaks are preserved
- Asymmetric Raman peaks are correctly identified
- Broad Raman features remain intact

### 3. **Diagnostic Capabilities**
- Detailed shape analysis for parameter tuning
- Peak-by-peak classification reasoning
- Visual feedback on classification decisions

### 4. **Backward Compatibility**
- Shape analysis can be disabled for traditional behavior
- Existing workflows remain unchanged
- Gradual adoption possible

## Usage Examples

### Basic Usage with Shape Analysis:
```python
# Create configuration with shape analysis
config = CosmicRayConfig(
    enabled=True,
    absolute_threshold=1500,
    neighbor_ratio=10.0,
    enable_shape_analysis=True,
    max_cosmic_fwhm=3.0,
    min_sharpness_ratio=5.0,
    max_asymmetry_factor=0.3,
    gradient_threshold=200.0
)

# Apply cosmic ray elimination
cre_manager = SimpleCosmicRayManager(config)
has_cosmic_rays, cleaned_spectrum, info = cre_manager.detect_and_remove_cosmic_rays(
    wavenumbers, intensities, "spectrum_id"
)
```

### Shape Analysis Diagnosis:
```python
# Analyze peak shapes for parameter tuning
diagnosis = cre_manager.diagnose_peak_shape(wavenumbers, intensities)

for peak in diagnosis['peak_details']:
    print(f"Peak at {peak['wavenumber']:.1f} cm⁻¹:")
    print(f"  Classification: {peak['classification']}")
    print(f"  Reason: {peak['reason']}")
    print(f"  FWHM: {peak['shape_metrics']['fwhm']:.1f}")
    print(f"  Sharpness: {peak['shape_metrics']['sharpness_ratio']:.1f}")
```

## Performance Considerations

- **Computational Overhead**: Minimal additional processing time
- **Memory Usage**: No significant increase
- **Scalability**: Suitable for large map datasets
- **Real-time Processing**: Compatible with live data acquisition

## Parameter Tuning Guidelines

### For Different Sample Types:

**High-Quality Crystals** (sharp Raman peaks):
- Increase `max_cosmic_fwhm` to 4.0-5.0
- Decrease `min_sharpness_ratio` to 3.0-4.0

**Powder Samples** (broader peaks):
- Keep default `max_cosmic_fwhm` at 3.0
- Keep default `min_sharpness_ratio` at 5.0

**Noisy Spectra**:
- Increase `gradient_threshold` to 300-500
- Decrease `max_asymmetry_factor` to 0.2

## Testing and Validation

The enhanced system has been tested with:
- Synthetic spectra with known cosmic rays and Raman peaks
- Real experimental data with various sample types
- Edge cases with very strong Raman peaks
- Different detector types and acquisition parameters

## Future Enhancements

Potential improvements for future versions:
- Machine learning-based classification
- Adaptive parameter adjustment
- Multi-scale shape analysis
- Integration with peak fitting algorithms

## Conclusion

The enhanced cosmic ray elimination system with shape analysis provides a significant improvement in the accuracy and reliability of cosmic ray detection while preserving important spectral features. The system maintains backward compatibility while offering powerful new capabilities for advanced users.

The shape-based approach represents a major step forward in automated spectral preprocessing, reducing the need for manual intervention and improving the quality of downstream analysis results. 