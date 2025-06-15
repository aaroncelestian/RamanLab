# Quantitative Analysis Guide for Raman Spectroscopy

## Overview

This guide shows how to solve your specific quantitative analysis problems using a **robust multi-method approach** that combines the strengths of different analysis methods while minimizing their individual weaknesses.

## Your Current Issues & Solutions

### 1. **Template Analysis (Overestimation)**
- **Problem**: Too many spectra "fit" the template, leading to overestimation
- **Solution**: Use template results as primary signal, but weight by fit quality (R²) and apply confidence thresholds

### 2. **NMF Analysis (Underestimation)**  
- **Problem**: Scale mismatch, completely underestimates component presence
- **Solution**: Use NMF as confidence booster rather than primary detection method

### 3. **ML Classification (Class Imbalance)**
- **Problem**: Gets components reversed, too few detections due to training data mismatch
- **Solution**: Use ML probabilities to validate other methods, with lower weight if detection rate is extreme

### 4. **Hybrid Methods (Scale Mismatch)**
- **Problem**: Current hybrid implementation suffers from scale alignment issues
- **Solution**: Intelligent weighted combination with automatic method weight adjustment

## Implementation Steps

### Step 1: Run the Demonstration
```bash
cd map_analysis_2d
python demo_quantitative_analysis.py
```

This shows you exactly how the method works with simulated data that matches your issues.

### Step 2: Integrate with Your Existing Workflow

Add this to your main analysis script:

```python
from integrate_quantitative_analysis import QuantitativeAnalysisIntegrator

# After you've run your existing analyses (template, NMF, ML)
integrator = QuantitativeAnalysisIntegrator(main_window=your_main_window)

# Extract all available results
extraction_results = integrator.auto_extract_all_results()
print(f"Available methods: {list(extraction_results.keys())}")

# Analyze your component of interest
result = integrator.analyze_component_by_name(
    component_name="Polypropylene",
    template_name="your_template_name",  # Use your actual template name
    nmf_component_index=2,  # Use the NMF component that best represents your material
    ml_class_name="plastic1",  # Use your ML class name
    confidence_threshold=0.3  # Adjust this to control sensitivity
)

if result:
    # Get quantitative statistics
    stats = result.statistics
    print(f"Component detected in {stats['detection_percentage']:.1f}% of pixels")
    print(f"Average percentage: {stats['mean_percentage_detected']:.1f}%")
    
    # Create maps for visualization
    maps = integrator.create_analysis_maps(result, your_map_shape)
    
    # maps now contains:
    # - maps['intensity']: Component intensity map
    # - maps['confidence']: Confidence map
    # - maps['percentage']: Percentage map  
    # - maps['detection']: Binary detection map
```

### Step 3: Adjust Parameters for Your Data

The key parameters to tune:

1. **Confidence Threshold** (0.1 - 0.8)
   - Lower values: More sensitive detection, more pixels detected
   - Higher values: More conservative detection, fewer false positives
   - Start with 0.3 and adjust based on results

2. **Method Selection**
   - Use all available methods for best results
   - If one method is clearly unreliable, exclude it
   - The algorithm automatically weights methods based on quality

3. **Component Identification**
   - Template: Use your best-fitting template
   - NMF: Try different components to find the one that best represents your material
   - ML: Use the class that represents your target material

## Expected Results

Based on the demonstration, you should see:

### **Quantitative Metrics**
- **Detection Percentage**: Reliable estimate of how much of your map contains the component
- **Average Component Percentage**: Quantitative concentration estimates with uncertainty bounds
- **Confidence Scores**: Reliability measure for each pixel
- **Method Contributions**: See how each method contributes to the final result

### **Improved Maps**
- **Intensity Map**: Shows component strength across the map
- **Confidence Map**: Shows reliability of detection
- **Percentage Map**: Shows estimated component concentration
- **Detection Map**: Binary map of where component is detected

### **Quality Metrics**
- Automatic method weighting based on data quality
- Agreement scores between methods
- Statistical summaries with uncertainty estimates

## Troubleshooting Common Issues

### Issue: "No analysis results available"
**Solution**: Make sure you've run template fitting, NMF, or ML classification first.

### Issue: "Method weights are unbalanced"
**Solution**: 
- Check if your template fitting has good R² values
- Try different NMF components
- Verify ML training data quality

### Issue: "Detection rate is too high/low"
**Solution**:
- Adjust confidence threshold
- Check individual method performance
- Verify template quality and NMF component selection

### Issue: "Percentages seem unrealistic"
**Solution**:
- The percentages are relative to the strongest detections in your dataset
- Consider them as "relative concentration" rather than absolute percentages
- Use the confidence scores to filter reliable detections

## Advanced Usage

### Custom Method Weights
```python
# If you want to manually adjust method weights
analyzer = RobustQuantitativeAnalyzer()

# Analyze with custom logic
result = analyzer.analyze_component(
    component_name="YourComponent",
    template_index=0,
    nmf_component=2,
    target_class_index=1
)
```

### Batch Analysis
```python
# Analyze multiple components
components_to_analyze = [
    {"name": "Polypropylene", "template": "PP_template", "nmf": 2, "ml": "plastic1"},
    {"name": "Background", "template": None, "nmf": 0, "ml": "background2"}
]

results = []
for comp in components_to_analyze:
    result = integrator.analyze_component_by_name(**comp)
    if result:
        results.append(result)

# Generate comprehensive report
summary = integrator.generate_analysis_summary(results)
print(summary)
```

## Integration with Your UI

The `QuantitativeAnalysisIntegrator` is designed to work with your existing UI. Here's how to add a new analysis option:

1. **Add Menu Item**: Create a "Quantitative Analysis" option in your analysis menu
2. **Extract Results**: The integrator automatically finds your existing analysis results
3. **User Dialog**: Show options for the user to select which methods to combine
4. **Results Display**: Add the quantitative maps to your existing map dropdown
5. **Summary Report**: Display the statistical summary in a dialog

## Key Benefits

1. **Addresses Your Specific Issues**:
   - Reduces template method false positives
   - Improves NMF underestimation
   - Handles ML class imbalance

2. **Provides Reliable Quantification**:
   - Percentage estimates with uncertainty bounds
   - Confidence measures for each detection
   - Statistical summaries

3. **Maintains Existing Workflow**:
   - Uses your existing analysis results
   - Integrates with current UI
   - No need to rerun analyses

4. **Adaptive Quality Assessment**:
   - Automatically weights methods based on data quality
   - Accounts for method agreement/disagreement
   - Provides diagnostic information

## Next Steps

1. **Test with your data**: Run the integration script with your actual results
2. **Optimize parameters**: Adjust confidence threshold and method selection
3. **Validate results**: Compare with your known standards or manual inspection
4. **Integrate with UI**: Add quantitative analysis option to your interface

This approach should give you the reliable quantitative information you need for identifying component presence and estimating concentrations in your Raman maps! 