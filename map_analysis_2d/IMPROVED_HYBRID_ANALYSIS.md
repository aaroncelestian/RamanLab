# Improved Hybrid Analysis Guide

## Problem Overview

You were experiencing misalignment between your Template Extracted Maps (which work well) and the NMF-Template Hybrid maps (which appeared sparse and didn't show the expected patterns). The original hybrid analysis was essentially just showing NMF component intensities rather than truly combining the two methods.

## Solution: Improved Hybrid Method

The new implementation creates a true hybrid analysis that:

1. **Uses template fitting results as the primary signal** (ensuring similarity to your working template maps)
2. **Applies NMF as a confidence booster** (enhancing regions where both methods agree)
3. **Weights results by fitting quality** (R-squared values)

## How to Use

### 1. Prerequisites
- Run template fitting first (your existing workflow)
- Run NMF analysis
- Both need to be completed before hybrid analysis

### 2. Access the Improved Hybrid Map
- The new map appears as **"Improved: Hybrid Template Map"** in your map features dropdown
- Look for it in the hybrid analysis results after running the analysis

### 3. Enhanced NMF Component Map
- The existing **"Enhanced: NMF Component (Log Scale)"** map has also been improved
- It now automatically detects when template results are available and enhances the NMF visualization

## Mathematical Approach

```
For each pixel:
if template_strength > 0:
    hybrid_intensity = template_strength × (1 + nmf_boost) × (1 + r_squared)
    where nmf_boost = nmf_intensity / 95th_percentile_nmf
else:
    hybrid_intensity = nmf_intensity × 0.1  # Reduced NMF-only signal
```

## Expected Results

The improved hybrid maps should now:

✅ **Look similar to your template extraction maps** (spatial patterns preserved)
✅ **Show enhanced confidence** where NMF and template fitting agree
✅ **Suppress false positives** from pure NMF analysis
✅ **Maintain template-based detection quality** while gaining NMF validation

## Comparison: Old vs New

| Aspect | Old Method | New Method |
|--------|------------|------------|
| Primary Signal | NMF component intensity | Template fitting coefficients |
| Template Role | Gating only (threshold) | Primary signal source |
| NMF Role | Main visualization | Confidence enhancement |
| Result Similarity | Poor alignment with template maps | Good alignment with template maps |
| False Positives | High (from NMF artifacts) | Low (template-based) |

## Testing the Method

Run the test script to see the logic in action:

```bash
cd map_analysis_2d
python test_improved_hybrid.py
```

This will show you:
- How the hybrid combination works mathematically
- Comparison between old and new approaches
- Expected improvements in different scenarios

## Troubleshooting

**Q: The hybrid map still doesn't look right**
- Ensure template fitting has run successfully
- Check that your polypropylene template is properly named (contains 'pp', 'polyprop', or 'extract')
- Verify NMF analysis completed with multiple components

**Q: No "Improved: Hybrid Template Map" option**
- Make sure both template fitting and NMF analysis have been completed
- Run the hybrid analysis dialog to populate the map features

**Q: Map appears blank or mostly zero**
- Check that template fitting found significant signals
- Verify NMF component 3 contains relevant data
- Try using a different NMF component in the hybrid analysis dialog

## Key Benefits

1. **Better Alignment**: Maps now look similar to your working template extraction
2. **Enhanced Confidence**: Strong template detections get boosted when NMF agrees
3. **Reduced Artifacts**: Pure NMF false positives are suppressed
4. **Quality Weighting**: Better fits get more emphasis in the final result
5. **Fallback Behavior**: Graceful handling when one method fails 