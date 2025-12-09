# Quick Start - Testing V1 Fixes

## Run the Test

```bash
cd v1_computational
python test_static_image.py
```

## What You'll See

### 1. Configuration Display (First Thing)
```
V1 MODEL CONFIGURATION
FEEDFORWARD WEIGHTS:
  LGN → L4:    5000.0
  L4 → L2/3:   120.0    ← FIXED (was 20-50)
  L2/3 → L5:   180.0    ← FIXED (was 800)
  L5 → L6:     150.0    ← FIXED (was 1200)
```

### 2. Key Results to Check

**Gabor Features:**
```
Active cells: 24.3%  ← Should be 10-30% (was 97-99%)
```

**V1 Layer Activity:**
```
LAYER_23:
  Mean rate: 18.67 Hz  ← Should be >10 Hz (was 0 Hz!)

LAYER_6:
  Mean rate: 8.92 Hz   ← Should be <20 Hz (was 110 Hz!)
```

**Layer 2/3 by Orientation:**
```
  0°: 22.15 Hz  ← All should be >10 Hz (not 0!)
 45°: 19.87 Hz
 90°: 16.43 Hz
135°: 18.92 Hz
```

## Success Criteria

### Must Pass:
- L2/3 mean firing rate > 10 Hz
- L6 mean firing rate < 20 Hz  
- All orientations show activity in L2/3
- Orientation map shows colored regions (not all gray)

### Should Pass:
- Gabor active cells: 10-30%
- Spike trains: 30-60 spikes per orientation
- L4 firing: 40-60 Hz
- No NaN values or errors

## If Something's Wrong

### L2/3 Still at 0 Hz?

Edit `config.py`:
```python
'weight_L4_to_L23': 150,      # Increase from 120
'L23_bias_current': 30.0,     # Increase from 20
```

### L6 Running Away (>50 Hz)?

Edit `config.py`:
```python
'weight_L5_to_L6': 100,       # Decrease from 150
```

## Enable Detailed Debugging

For real-time synaptic currents, edit `config.py`:
```python
'show_synaptic_currents': True,
```

**Warning:** Very verbose! Shows currents every 20 time steps.

## Files Modified

All changes are in these files (already updated):
- `config.py` - New weights and L2/3 parameters
- `v1_column.py` - Uses new weights, prints currents
- `v1_model.py` - Diagnostics and debug support
- `neurons.py` - Bias current support
- `pipeline.py` - Passes debug flags
- `gabor_extractor.py` - Sparsification (from earlier fix)
- `test_static_image.py` - Enhanced diagnostics

## Current Settings

**Weights:**
- L4→L2/3: 120 (was 20-50)
- L2/3→L5: 180 (was 800)
- L5→L6: 150 (was 1200)

**L2/3 Neurons:**
- Threshold: -55 mV (was -50)
- Tau_m: 25 ms (was 10)
- Bias: 20 pA (was 0)

These are **balanced defaults** that should work well.

## Next Steps

1. **Run test** - `python test_static_image.py`
2. **Check L2/3 fires** - Look for >10 Hz in statistics
3. **Check L6 stable** - Should be <20 Hz, not 110 Hz
4. **Check orientation map** - Should show colors, not all gray
5. **If good**, proceed to real-time pipeline

See `TESTING_GUIDE.md` for detailed troubleshooting.

