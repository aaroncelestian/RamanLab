# Peak Search Enhancement Documentation

## Overview

The advanced search functionality in RamanLab has been significantly enhanced to properly handle peak position searches. The previous implementation had several issues that made peak-based searches less effective than intended.

## Issues Fixed

### 1. **Peak Filtering vs Peak Scoring Separation**

**Previous Problem**: When you specified peak positions in advanced search, they were only used for filtering the database (finding spectra that contain those peaks), but not for ranking/scoring the results.

**Solution**: The system now uses user-specified peak positions for both filtering AND scoring, making spectra with better peak matches rank higher in results.

### 2. **Peak Algorithm Enhancement**

**Previous Problem**: The "peak" algorithm only compared automatically detected peaks from your current spectrum with detected peaks in database spectra, ignoring the peak positions you specified in the search form.

**Solution**: The `calculate_peak_score` method now:
- Uses your specified peak positions as the primary reference when available
- Gives higher scores to database spectra that have peaks matching your specified positions
- Uses a tolerance-based matching system with enhanced scoring weights

### 3. **Peak-Only Search Mode Removed**

**Previous Problem**: When only peak positions were specified, the system entered a "peak-only" mode that bypassed similarity scoring entirely, assigning all matching spectra a score of 1.0.

**Solution**: All searches now use proper similarity scoring, but with enhanced peak-based scoring when peak positions are specified.

## How Peak Search Now Works

### 1. **Peak Position Filtering**

The `spectrum_passes_filters` method filters the database to only include spectra that have peaks within the specified tolerance of your target peak positions.

```python
# Example: Search for peaks at 1000, 1500 cm⁻¹ with ±10 cm⁻¹ tolerance
# Only spectra with peaks between 990-1010 AND 1490-1510 cm⁻¹ will be considered
```

### 2. **Enhanced Peak Scoring**

The `calculate_peak_score` method now:

- **Uses your specified peaks as reference**: Instead of comparing detected peaks, it uses your specified peak positions
- **Interpolates intensities**: Gets intensity values at your specified positions from both query and database spectra
- **Weighted scoring**: Emphasizes peak position matching (80%) over intensity similarity (20%) for user-specified peaks
- **Tolerance-based matching**: Finds the closest peak in database spectra within tolerance
- **Bonus scoring**: Gives extra points for high peak match ratios

### 3. **Algorithm Selection Impact**

**Peak Algorithm**: 
- Uses enhanced peak scoring with your specified positions
- Best for searches where specific peak positions are most important

**Combined Algorithm**: 
- Uses correlation (20%) + DTW (30%) + peak matching (50%) when peaks are specified
- Uses correlation (30%) + DTW (70%) when no peaks are specified
- Best for comprehensive searches

**Correlation/DTW Algorithms**: 
- Use peak positions for filtering only, then apply their respective similarity measures
- Best for overall spectral shape matching

## Usage Recommendations

### 1. **For Peak Position Searches**

```
Peak Positions: 1000, 1500, 2000
Tolerance: ±15 cm⁻¹
Algorithm: Peak
```

This will:
1. Filter database for spectra with peaks at those positions (±15 cm⁻¹)
2. Score results based on how well the peaks match
3. Rank results with better peak matches higher

### 2. **For Comprehensive Searches with Peak Emphasis**

```
Peak Positions: 1000, 1500
Tolerance: ±10 cm⁻¹
Algorithm: Combined
```

This will:
1. Filter database for spectra with those peaks
2. Use 50% peak scoring + 30% DTW + 20% correlation
3. Provide balanced results considering both peaks and overall spectral shape

### 3. **For Shape Matching with Peak Pre-filtering**

```
Peak Positions: 1000, 1500
Algorithm: DTW or Correlation
```

This will:
1. Filter database for spectra with those peaks
2. Use DTW/correlation scoring on the filtered results
3. Focus on overall spectral similarity among peak-containing spectra

## Database Peak Storage Format

The system handles multiple peak storage formats:

1. **Preferred Format**: `{"peaks": {"wavenumbers": numpy_array}}`
2. **Legacy Format**: Peak indices that get converted to wavenumbers
3. **Direct Values**: List of wavenumber values

## Technical Details

### Enhanced Scoring Formula

For user-specified peaks:
```
score = 0.8 × peak_match_ratio + 0.2 × intensity_similarity
```

With bonus:
```
if peak_match_ratio >= 0.8:
    score = min(1.0, score × 1.1)  # 10% bonus
```

### Peak Matching Logic

1. For each specified peak position
2. Find closest peak in database spectrum
3. If distance ≤ tolerance, count as match
4. Calculate position and intensity similarity
5. Apply scoring weights based on search type

## Benefits

1. **Peak positions are now prominent**: Spectra with better peak matches rank higher
2. **Flexible search modes**: Choose algorithm based on what's most important
3. **Proper scoring**: No more artificial 1.0 scores for all peak matches
4. **Enhanced filtering**: Database is pre-filtered for efficiency
5. **Backward compatibility**: Still works with automatically detected peaks when no positions are specified

## Example Search Results

**Before Enhancement**:
- Peak search would return random order with all scores = 1.0
- No distinction between good and poor peak matches

**After Enhancement**:
- Results ranked by peak matching quality
- Scores reflect actual similarity (0.0 - 1.0 range)
- Better peak matches appear first in results 