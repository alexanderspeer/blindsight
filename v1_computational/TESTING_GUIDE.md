# V1 Model Testing Guide

## Quick Start

### Run the Static Image Test
```bash
python test_static_image.py
```

This test will:
1. Display the current V1 configuration (weights, L2/3 parameters)
2. Process a test image with oriented edges (0°, 45°, 90°, 135°)
3. Print detailed statistics for every pipeline stage
4. Show orientation maps and visualizations

## What to Look For

### 1. Configuration Display (at start)
```
V1 MODEL CONFIGURATION
FEEDFORWARD WEIGHTS:
  LGN → L4:    5000.0
  L4 → L2/3:   120.0    ← Should be 100-150 (not 20!)
  L2/3 → L5:   180.0    ← Should be 150-250 (not 800!)
  L5 → L6:     150.0    ← Should be 100-200 (not 1200!)

LAYER 2/3 PARAMETERS:
  Threshold:    -55.0 mV (default: -50.0 mV)  ← Lower = easier to fire
  Tau_m:        25.0 ms (default: 10.0 ms)    ← Longer = better integration
  Bias current: 20.0 pA                        ← Keeps neurons ready
```

### 2. Gabor Feature Statistics
```
GABOR FEATURE MAPS - DETAILED STATISTICS
0° Orientation:
  Active cells:
    > 0.01: 35/144 (24.3%)   ← Should be ~10-30%, not 97%!
```

**Success criteria:**
- Active cells: 10-30% per orientation (not 97-99%)
- Clear sparsity pattern

### 3. Spike Encoding Statistics
```
SPIKE ENCODING - DETAILED STATISTICS
0° Orientation:
  Total spikes: 43          ← Should be 30-60
  Unique neurons: 38/144 (26.4%)
```

**Success criteria:**
- 30-60 spikes per orientation
- 20-40% of neurons spike

### 4. V1 Layer Activity (CRITICAL!)
```
V1 LAYER ACTIVITY - DETAILED STATISTICS
LAYER_4:
  Mean rate: 45.23 Hz       ← Should be 40-60 Hz

LAYER_23:
  Mean rate: 18.67 Hz       ← Should be >10 Hz (NOT 0 Hz!)
  Active neurons:
    > 1 Hz:  85/576 (14.8%) ← Should be >10%

LAYER_5:
  Mean rate: 12.34 Hz       ← Should be 5-25 Hz

LAYER_6:
  Mean rate: 8.92 Hz        ← Should be <20 Hz (NOT 110 Hz!)
```

**Success criteria:**
- L2/3 mean rate > 10 Hz (was 0 Hz before fix)
- L6 mean rate < 20 Hz (was 110 Hz before fix)
- All layers show some activity

### 5. Layer 2/3 by Orientation
```
LAYER 2/3 BY ORIENTATION:
    0°: 22.15 Hz (active: 28/144)
   45°: 19.87 Hz (active: 24/144)
   90°: 16.43 Hz (active: 21/144)
  135°: 18.92 Hz (active: 26/144)
```

**Success criteria:**
- All orientations show >10 Hz (not just one!)
- Rates vary based on image content
- Active neuron counts are reasonable

### 6. Frame 10 Diagnostics
On frame 10, you'll see comprehensive diagnostics:
```
V1 LAYER CONNECTIVITY DIAGNOSTICS
CONNECTION WEIGHTS:
  LGN → L4:    5000.0
  L4 → L2/3:   120.0
  L2/3 → L5:   180.0
  L5 → L6:     150.0

LAYER 2/3 SYNAPTIC CURRENTS (by orientation):
    0°: Exc=42.15 pA (max=156.23), Nonzero=38/144, V_m=-58.34 mV, Near_thresh=12
   45°: Exc=38.92 pA (max=142.67), Nonzero=35/144, V_m=-59.12 mV, Near_thresh=10
```

**What to check:**
- Exc currents are >20 pA mean (not near 0!)
- Some neurons have Nonzero currents
- V_m is closer to threshold (-55 mV)
- Some neurons are Near_thresh

## Enabling Verbose Debugging

### Synaptic Current Tracing
To see real-time synaptic currents during simulation:

**Edit `config.py`:**
```python
DEBUG_CONFIG = {
    ...
    'show_synaptic_currents': True,  # Change from False
    ...
}
```

**Output during simulation:**
```
SYNAPTIC CURRENTS DURING STIMULUS:
  [  0° @  55.0ms] L2/3_syn= 42.15 pA, L5_syn= 12.34 pA, L6_syn=  8.67 pA
  [ 45° @  55.0ms] L2/3_syn= 38.92 pA, L5_syn= 15.23 pA, L6_syn=  9.12 pA
  [ 90° @  55.0ms] L2/3_syn= 35.67 pA, L5_syn= 11.89 pA, L6_syn=  7.45 pA
  [135° @  55.0ms] L2/3_syn= 40.23 pA, L5_syn= 13.56 pA, L6_syn=  8.90 pA
```

**Warning:** This produces A LOT of output (printed every 20 time steps for 100ms stimulus = many lines).

## Troubleshooting

### Problem: L2/3 Still Shows 0 Hz

**Check:**
1. Is `weight_L4_to_L23` set correctly? (Should be 100-150)
2. Are L2/3 parameters applied? (Check config printout)
3. Do spikes reach L4? (Check spike encoding statistics)

**Fix:**
```python
# In config.py, try increasing:
'weight_L4_to_L23': 150,      # Increase from 120
'L23_bias_current': 30.0,     # Increase from 20
'L23_v_threshold': -57.0,     # Lower from -55
```

### Problem: L6 Shows Runaway (>50 Hz)

**Check:**
1. Is `weight_L5_to_L6` set correctly? (Should be 100-200)
2. Is only one orientation affected? (Check per-orientation stats)

**Fix:**
```python
# In config.py, try decreasing:
'weight_L5_to_L6': 100,       # Decrease from 150
```

### Problem: All Layers Silent

**Check:**
1. Are Gabor features sparse? (Should be 10-30% active)
2. Do spike trains generate? (Should be 30-60 spikes)
3. Does L4 fire? (Should be 40-60 Hz)

**If L4 is silent:**
```python
# In config.py:
'lgn_to_ss4_weight': 6000.0,  # Increase from 5000
```

### Problem: Orientation Map Shows All Gray ("No Response")

**This means L2/3 is silent!** The decoder uses L2/3 firing rates.

**Fix:** Follow "L2/3 Still Shows 0 Hz" section above.

## Parameter Tuning Quick Reference

### Conservative (Safe, Moderate Activity)
```python
'weight_L4_to_L23': 100,
'weight_L23_to_L5': 150,
'weight_L5_to_L6': 100,
'L23_v_threshold': -55.0,
'L23_tau_membrane': 25.0,
'L23_bias_current': 20.0,
```

### Balanced (Recommended, Current Settings)
```python
'weight_L4_to_L23': 120,
'weight_L23_to_L5': 180,
'weight_L5_to_L6': 150,
'L23_v_threshold': -55.0,
'L23_tau_membrane': 25.0,
'L23_bias_current': 20.0,
```

### Aggressive (High Activity, Risk of Runaway)
```python
'weight_L4_to_L23': 150,
'weight_L23_to_L5': 250,
'weight_L5_to_L6': 200,
'L23_v_threshold': -57.0,
'L23_tau_membrane': 30.0,
'L23_bias_current': 30.0,
```

## Expected Timeline

A successful test run should take ~5-10 seconds and produce:

1. Configuration printout (2 seconds)
2. Pipeline initialization (2 seconds)
3. Processing with diagnostics (2-3 seconds)
4. Visualization display (waiting for keypress)

Total output: ~500-1000 lines of diagnostics (without synaptic current tracing).

## Success Checklist

Before moving to real-time pipeline:

- [ ] Configuration shows updated weights (120/180/150)
- [ ] Gabor features show 10-30% active cells
- [ ] Spike trains generate 30-60 spikes per orientation
- [ ] L4 fires at 40-60 Hz
- [ ] **L2/3 fires at >10 Hz** (critical!)
- [ ] L6 stays under 20 Hz (no runaway)
- [ ] Orientation map shows colored regions (not all gray)
- [ ] Frame 10 diagnostics confirm proper connectivity
- [ ] No NaN or error messages

If all checks pass, the model is ready for real-time testing!

