# Gabor Feature Map Sparsification - Implementation Summary

## Problem Addressed
- **Before**: 97-99% of cells were above threshold for all orientations
- **Issue**: Produced sparse, unreliable spike trains dominated by intrinsic dynamics
- **Result**: Orientation maps didn't match Gabor filter orientations

## Solution Implemented

### 1. Enhanced Sparsification Pipeline (`gabor_extractor.py`)

Added `_sparsify_grid()` method with 4-step pipeline:

```python
def _sparsify_grid(self, grid):
    # Step 1: Z-score normalization
    z_scores = (grid - mean) / std
    
    # Step 2: Clip negative values (suppress below-average)
    z_scores = np.clip(z_scores, 0, None)
    
    # Step 3: Rescale to [0, 3]
    normalized = (z_scores / z_max) * 3.0
    
    # Step 4: Percentile thresholding (keep top 20%)
    threshold = np.percentile(normalized[normalized > 0], 80)
    normalized[normalized < threshold] = 0
```

### 2. Orientation Competition (`_apply_orientation_competition()`)

Implements softmax competition across orientations at each spatial location:

```python
def _apply_orientation_competition(self, features):
    # Stack all orientations
    stacked = np.stack([features[ori] for ori in orientations], axis=0)
    
    # Apply softmax with temperature=0.5 for sharpness
    exp_values = np.exp(stacked / temperature)
    softmax = exp_values / (exp_values.sum(axis=0, keepdims=True) + 1e-8)
    
    # Multiply by original magnitudes
    sharpened = softmax * stacked
```

### 3. Comprehensive Diagnostics (`_print_diagnostics()`)

Automatically prints on frame 0 and 10:
- Active cell percentages per orientation
- Value histograms (before/after normalization)
- Orientation dominance statistics
- Cross-orientation comparisons

### 4. Enhanced Test Script (`test_static_image.py`)

Added three detailed statistics functions:
- `print_gabor_statistics()` - Shows active cell percentages, histograms, distribution
- `print_spike_statistics()` - Spike counts, timing distributions, neuron coverage
- `print_v1_statistics()` - Layer-wise firing rates, activity levels

## Expected Results

### Before Changes
```
Active cells per orientation:
  0°:   97-99% above threshold
  45°:  97-99% above threshold
  90°:  97-99% above threshold
  135°: 97-99% above threshold
```

### After Changes
```
Active cells per orientation:
  0°:   10-30% above threshold (sparse)
  45°:  10-30% above threshold (sparse)
  90°:  10-30% above threshold (sparse)
  135°: 10-30% above threshold (sparse)

With clear dominance at appropriate locations:
  0° dominant where horizontal edges exist
  90° dominant where vertical edges exist
  etc.
```

## Files Modified

1. **gabor_extractor.py**
   - Modified `_create_retinotopic_grid()` to call sparsification
   - Added `_sparsify_grid()` method (z-score + percentile thresholding)
   - Added `_apply_orientation_competition()` method (softmax across orientations)
   - Added `_print_diagnostics()` method (comprehensive statistics)
   - Updated `extract_features()` to accept verbose flag and apply competition

2. **pipeline.py**
   - Updated Gabor feature extraction call to enable verbose on frames 0 and 10
   - Passes `apply_orientation_competition=True` by default

3. **test_static_image.py**
   - Added `print_gabor_statistics()` function
   - Added `print_spike_statistics()` function
   - Added `print_v1_statistics()` function
   - Enhanced main() to display all detailed statistics

## How to Test

Run the static image test:
```bash
python test_static_image.py
```

Expected output will show:
1. **Gabor diagnostics** automatically printed (frame 0)
2. **Detailed statistics** for each pipeline stage
3. **Active cell percentages** dropped to 10-30% range
4. **Orientation selectivity** clearly visible in spike trains
5. **Visual confirmation** in the displayed plots

## Key Parameters

- **Percentile threshold**: 80th percentile (keeps top 20% of cells)
- **Z-score clipping**: Negative values → 0
- **Rescale range**: [0, 3]
- **Softmax temperature**: 0.5 (sharper competition)

## Success Criteria

Active cells drop from ~99% to ~10-30%
Spike counts scale with Gabor strength
Decoder orientation maps align with Gabor outputs
Clear orientation dominance at appropriate spatial locations
Histograms show sparse distribution with strong peaks

## Next Steps

1. Run `test_static_image.py` to verify improvements
2. Check the diagnostic output for confirmation
3. If needed, tune parameters:
   - Percentile threshold (currently 80)
   - Softmax temperature (currently 0.5)
   - Rescale range (currently [0, 3])

