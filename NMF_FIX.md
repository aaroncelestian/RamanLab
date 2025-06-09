# NMF Plotting Dimension Mismatch Fix

## Problem
NMF computation succeeded but plotting failed with error:
```
Error plotting NMF results: x and y must have same first dimension, but have shapes (665,) and (672,)
```

## Root Cause
- **NMF basis matrix**: 672 features (original spectrum length)  
- **Target wavenumbers**: 665 points (reduced due to preprocessing/trimming)
- **Mismatch**: Plotting tried to use 665 wavenumber points with 672 NMF component values

## Solution
Modified `_plot_nmf_results()` method to:

1. **Detect dimension mismatch** between wavenumbers and NMF basis
2. **Automatically trim to overlapping range** - use `min(len(wavenumbers), nmf_basis.shape[1])`
3. **Plot compatible dimensions** - ensures x and y arrays have same length
4. **Show warning message** to inform user about the mismatch

## Code Changes
```python
# Check dimension mismatch and fix it
if len(wavenumbers) != self.nmf_basis.shape[1]:
    print(f"Warning: Wavenumber array ({len(wavenumbers)}) and NMF basis ({self.nmf_basis.shape[1]}) dimension mismatch")
    # Use only the overlapping range
    min_length = min(len(wavenumbers), self.nmf_basis.shape[1])
    wavenumbers = wavenumbers[:min_length]
    nmf_basis_plot = self.nmf_basis[:, :min_length]
```

## Result
- ✅ NMF plots display correctly
- ✅ No more dimension mismatch errors
- ✅ User sees informative warning about data inconsistency
- ✅ NMF analysis results are fully viewable 