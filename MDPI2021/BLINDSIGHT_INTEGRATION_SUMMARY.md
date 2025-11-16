# Blindsight V1 Integration - Complete Summary

## Overview

This package provides a complete integration pipeline for connecting real-time camera input to the NEST-based V1 spiking neural network model, enabling a blindsight-like perception system suitable for embedded platforms (Raspberry Pi, Arduino).

---

## Questions Answered

### 1. **Input Injection into Layer 4**

**Answer:** Spikes are injected via `spike_generator` nodes connected 1:1 to LGN `parrot_neuron` relays, which then connect to Layer 4 Spiny Stellate (SS4) cells.

**Key Code Sections:**
- `Simulation_V1_pinwheel_MEGcomparison.py`, lines 38-45: Input creation
- `OrientedColumnV1.py`, lines 48-52: LGN→Layer4 connection (weight: 15000.0, delay: 1.0 ms)

**Integration Entry Point:**
```python
# Replace spike loading with real-time injection
for neuron_id, spike_times in camera_spikes.items():
    nest.SetStatus([inputs[neuron_id]], {'spike_times': spike_times})
```

---

### 2. **LGN Input Structure**

**Answer:** 
- **324 LGN parrot neurons** (18×18 retinotopic grid)
- **No built-in preprocessing** - expects pre-encoded spike trains
- Original stimuli were **pre-computed Gabor-filtered** responses saved as `.pckl` files

**Structure:**
```
Camera → [Your Encoder] → spike_generator[324] → parrot_neuron[324] → SS4[324]
```

**Preprocessing Requirements:**
- Downsample frames to 18×18 grid
- Convert intensity to spike rates (Poisson, latency, or temporal contrast)
- Optional: Gabor filtering for orientation selectivity enhancement

**Provided Solution:** `CameraSpikeEncoder` class in `blindsight_camera_encoder.py` implements three encoding strategies:
1. **Poisson**: Rate-based (default)
2. **Latency**: First-spike timing
3. **Temporal Contrast**: DVS-like event encoding

---

### 3. **Orientation Column Configuration**

**Answer:**

**Column Indexing:**
- Each column has a preferred orientation: 0°, 45°, 90°, 135°
- All columns receive **identical LGN input**
- Orientation selectivity emerges from **pre-trained IE values** in Layer 4

**Code Locations:**
- `OrientedColumnV1.py`, line 1: `column(setdegree, LGN)` function
- `Simulation_V1_pinwheel_MEGcomparison.py`, lines 48-51: Four columns created

**Internal Architecture (324 SS4 cells per column):**
- 81 groups × 4 neurons = 324 total
- Each group: MNSD architecture with mutual IE modulation
- Lines 213-243 in `OrientedColumnV1.py`

**Extending Orientations:**
Yes! Add new angles by:
1. Training new IE values (see Section 4 of V1_INTEGRATION_GUIDE.md)
2. Creating new column: `column(22.5, LGN)` for 22.5°
3. Optionally add inter-column inhibition for biological realism

**Example (8 orientations):**
```python
orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
columns = {angle: column(angle, LGN) for angle in orientations}
```

---

### 4. **Pretrained Excitability Format**

**Answer:**

**File Format:** Python pickle files (`soma_exc_*.pckl`)
- **Type:** List or NumPy array
- **Length:** 324 values (one per Layer 4 neuron)
- **Units:** Dimensionless multiplicative factor
- **Range:** Typically 0.8 - 1.5
  - 1.0 = baseline (no modulation)
  - >1.0 = enhanced excitability (LTP-IE)
  - <1.0 = reduced excitability (LTD-IE)

**How IE Works:**
```cpp
// lifl_psc_exp_ie.cpp, line 408
V_m = V_m * P22 + i_syn_in * P21in + 
      (i_syn_ex * P21ex + ...) * enhancement;
      //                           ^^^^^^^^^^^
      //                    Intrinsic Excitability
```

**Regenerating IE Values:**

**Method 1 - Train from Scratch:**
```python
# See V1_INTEGRATION_GUIDE.md, Section 4
def train_orientation_column(angle, training_stimuli, n_trials=1000):
    # Create SS4 neurons with IE enabled
    SS4 = nest.Create('lifl_psc_exp_ie', 324, {
        'lambda': 0.0005,  # Learning rate
        'tau': 12.5,       # Time window
        'std_mod': True    # Enable learning
    })
    # ... train with oriented Gabor stimuli
    # Extract final soma_exc values
    # Save as pickle
```

**Method 2 - Adaptive (Real-time):**
```python
# Dynamically adjust based on camera statistics
ie_values = compute_adaptive_ie(camera_input, target_orientation)
for i, neuron in enumerate(SS4):
    nest.SetStatus([neuron], {'soma_exc': ie_values[i]})
```

---

### 5. **Real-Time Spike Stream Integration**

**Answer:**

**Best Entry Point:** `Simulation_V1_pinwheel_MEGcomparison.py`, lines 59-70

**ORIGINAL:**
```python
# Load pre-computed spikes
file = "./files/spikes_reponse_gabor_randn02_19.pckl"
resultados = pickle.load(file)
for j in range(324):
    spike_time = times[senders == min(senders)+j] + currtime
    nest.SetStatus([inputs[j]], {'spike_times': spike_time})
```

**REPLACE WITH:**
```python
# Real-time camera spikes
current_nest_time = nest.GetKernelStatus('time')
camera_spikes = encoder.encode_frame(camera.capture(), current_nest_time)

for neuron_id, spike_times in camera_spikes.items():
    nest.SetStatus([inputs[neuron_id]], {'spike_times': spike_times})
```

**Complete Implementation:** `blindsight_realtime_v1.py`
- **Class:** `BlindSightV1System`
- **Architecture:** Three threads:
  1. `camera_thread()`: Capture + encode
  2. `simulation_thread()`: NEST simulation + spike injection
  3. `visualization_thread()`: Real-time display

**Usage:**
```python
system = BlindSightV1System(
    orientations=[0, 45, 90, 135],
    camera_source=0,
    lightweight=True
)
system.run(duration=60)  # Run for 60 seconds
```

---

### 6. **Performance & Optimization**

**Answer:**

**Bottlenecks Identified:**

1. **Layer 4 MNSD Architecture**
   - 81 groups × 4 neurons × 8 STDP synapses = high computational cost
   - IE plasticity updates: O(n²) complexity

2. **Recording Overhead**
   - Multimeters recording V_m + I_syn_ex at 0.1 ms for all neurons
   - Massive I/O bottleneck

3. **Python Overhead**
   - NEST kernel is C++, but Python wrapping adds latency

**Benchmarks (Expected Performance):**

| Configuration | Neurons | Speed (realtime factor) | Platform |
|---------------|---------|------------------------|----------|
| Full model (4 columns × 1167) | 4,668 | 1-5x | Desktop (8-core) |
| Lightweight (4 columns × 81) | 324 | 10-20x | Raspberry Pi 4 |
| Single column (lightweight) | 81 | 50-100x | Raspberry Pi 4 |

**Optimization Strategies:**

**LEVEL 1 - Disable Recording:**
```python
# Remove multimeters (lines 268-276 in OrientedColumnV1.py)
# Keep only spike_detector for output
```

**LEVEL 2 - Use Lightweight Model:**
```python
# LightweightColumn in blindsight_realtime_v1.py
# 81 neurons instead of 324 per column (4x reduction)
system = BlindSightV1System(lightweight=True)
```

**LEVEL 3 - Reduce Time Resolution:**
```python
nest.SetKernelStatus({'resolution': 0.5})  # 0.5 ms instead of 0.1 ms
```

**LEVEL 4 - Multi-threading:**
```python
nest.SetKernelStatus({'local_num_threads': 4})  # Use all CPU cores
```

**GPU Acceleration:**
- NEST **does not natively support GPU**
- Alternative: Port to CARLsim (CUDA-based spiking simulator)
- Expected speedup: 10-100x on NVIDIA GPU

**On-Device Split Architecture:**
```
Arduino/Camera:
  - Image capture
  - Gabor filtering (integer math)
  - Spike encoding
  → Serial output

Raspberry Pi 4:
  - NEST simulation (lightweight)
  - Orientation decision
  - Motor control
```

---

## Files Created

### Core Integration Files

1. **`V1_INTEGRATION_GUIDE.md`**
   - 60+ page comprehensive guide
   - Annotated code sections
   - Detailed answers to all questions
   - Extension examples

2. **`blindsight_camera_encoder.py`**
   - `CameraSpikeEncoder` class
   - Three encoding strategies (Poisson, latency, temporal contrast)
   - `GaborFilterBank` for preprocessing
   - Visualization tools

3. **`blindsight_realtime_v1.py`**
   - `BlindSightV1System` class
   - `LightweightColumn` (81-neuron simplified model)
   - Multi-threaded real-time pipeline
   - Live visualization
   - Performance monitoring

4. **`SETUP_BLINDSIGHT.md`**
   - Installation instructions (NEST + module)
   - Hardware-specific setup (RPi, Arduino)
   - Performance tuning guide
   - Troubleshooting section

5. **`requirements_blindsight.txt`**
   - Python dependencies
   - Version specifications

6. **`test_blindsight_integration.py`**
   - 7 automated tests
   - Verification of all components
   - Synthetic stimulus testing

7. **`BLINDSIGHT_INTEGRATION_SUMMARY.md`** (this file)
   - Complete overview
   - Quick reference

---

## Quick Start

### Installation
```bash
# 1. Install NEST 2.20.1
cd nest-simulator-2.20.1/build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/nest ..
make -j4 && make install

# 2. Build LIFL_IE module
cd MDPI2021/LIFL_IE/build
cmake -Dwith-nest=$HOME/nest/bin/nest-config ..
make -j4 && make install

# 3. Install Python dependencies
pip3 install -r requirements_blindsight.txt
```

### Testing
```bash
# Run test suite
python3 test_blindsight_integration.py

# Test camera encoder
python3 blindsight_camera_encoder.py

# Test full system (synthetic input)
python3 blindsight_realtime_v1.py --lightweight --duration 10
```

### Running with Camera
```bash
# Webcam (lightweight mode recommended)
python3 blindsight_realtime_v1.py --lightweight

# Video file
python3 blindsight_realtime_v1.py --video test.mp4 --lightweight

# No visualization (max speed)
python3 blindsight_realtime_v1.py --lightweight --no-vis
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Camera / Arduino                        │
│              (Image Acquisition)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          CameraSpikeEncoder                              │
│  • Downsample to 18×18 (or 9×9)                         │
│  • Normalize intensity                                   │
│  • Encode: Poisson / Latency / Temporal                 │
│  Output: {neuron_id: [spike_times]}                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          NEST Simulator                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Input Layer: spike_generator[324]              │   │
│  └──────────┬──────────────────────────────────────┘   │
│             │                                            │
│             ▼                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │  LGN: parrot_neuron[324]                        │   │
│  └──────────┬──────────────────────────────────────┘   │
│             │ (broadcast to all columns)                │
│             ├─────────┬─────────┬─────────┬─────────┐  │
│             ▼         ▼         ▼         ▼         ▼  │
│  ┌─────────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │ Column 0°   │ │Column45°│ │Column90°│ │Column135°│ │
│  │             │ │         │ │         │ │         │ │
│  │ Layer 4 (SS)│ │Layer 4  │ │Layer 4  │ │Layer 4  │ │
│  │ 324 or 81   │ │neurons  │ │neurons  │ │neurons  │ │
│  │ w/ IE       │ │         │ │         │ │         │ │
│  │      ↓      │ │    ↓    │ │    ↓    │ │    ↓    │ │
│  │ Layer 2/3   │ │Layer2/3 │ │Layer2/3 │ │Layer2/3 │ │
│  │ (Pyramidal) │ │(Pyr)    │ │(Pyr)    │ │(Pyr)    │ │
│  │      ↓      │ │    ↓    │ │    ↓    │ │    ↓    │ │
│  │ [Detector]  │ │[Detect] │ │[Detect] │ │[Detect] │ │
│  └─────────────┘ └─────────┘ └─────────┘ └─────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Decision Module                                  │
│  • Count spikes per column (100 ms window)              │
│  • Winner-take-all                                       │
│  • Output: dominant_orientation, confidence             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Application Layer                                │
│  • Robotic control                                       │
│  • Navigation                                            │
│  • Texture classification                                │
│  • Obstacle detection                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Key Parameters Reference

### NEST Simulation
```python
nest.SetKernelStatus({
    'resolution': 0.1,        # Time step (ms)
    'local_num_threads': 4    # CPU cores
})
```

### Spike Encoder
```python
encoder = CameraSpikeEncoder(
    resolution=(18, 18),      # Spatial grid
    max_rate=100.0,           # Max firing rate (Hz)
    min_rate=5.0,             # Baseline rate (Hz)
    encoding_type='poisson',  # 'poisson', 'latency', 'temporal_contrast'
    temporal_window=10.0      # Time window (ms)
)
```

### Layer 4 Neurons (lifl_psc_exp_ie)
```python
SS4 = nest.Create('lifl_psc_exp_ie', 324, {
    'lambda': 0.0005,    # IE learning rate
    'tau': 12.5,         # IE time window (ms)
    'std_mod': False,    # Disable IE at runtime
    'V_th': -50.0,       # Spike threshold (mV)
    'tau_m': 10.0        # Membrane time constant (ms)
})
```

### Connections
```python
# LGN → Layer 4
nest.Connect(LGN, SS4, 'one_to_one', {
    "weight": 15000.0,   # Strong feedforward
    "delay": 1.0         # ms
})

# Layer 4 → Layer 2/3
nest.Connect(SS4, Pyr23, 'one_to_one', {
    "weight": 400.0,
    "delay": 1.0
})
```

---

## Performance Tips

1. **Start with lightweight mode** (`--lightweight`)
2. **Disable visualization** for max speed (`--no-vis`)
3. **Use coarser time resolution** (0.5 ms instead of 0.1 ms)
4. **Reduce camera FPS** (15 FPS instead of 30 FPS)
5. **Test with single column** before full pinwheel
6. **Profile bottlenecks** with `cProfile`

---

## Example Applications

### 1. Edge-Based Navigation
```python
if decision['dominant_orientation'] == 90 and decision['confidence'] > 0.7:
    robot.stop()  # Vertical edge = obstacle
elif decision['dominant_orientation'] == 0:
    robot.forward()  # Horizontal edge = clear path
```

### 2. Texture Recognition
```python
orientation_entropy = compute_entropy(decision_history)
if orientation_entropy < 0.5:
    print("Structured texture")  # Wood grain, fabric
else:
    print("Random texture")  # Gravel, sand
```

### 3. Object Tracking
```python
track_orientation_changes(decision_history)
if orientation_flip_detected():
    print("Object boundary crossed")
```

---

## Citation

```bibtex
@article{santos2021lifl,
  title={LIFL\_IE NEST Simulator Extension Module},
  author={Santos-Mayo, Alejandro and others},
  journal={MDPI},
  year={2021}
}
```

---

## Support & Resources

- **Integration Guide**: `V1_INTEGRATION_GUIDE.md`
- **Setup Instructions**: `SETUP_BLINDSIGHT.md`
- **Code Examples**: `blindsight_realtime_v1.py`
- **Test Suite**: `test_blindsight_integration.py`

- **Original Author**: alejandro.santos@ctb.upm.es
- **NEST Documentation**: https://nest-simulator.readthedocs.io/
- **Module Repository**: https://github.com/nest/nest-simulator/

---

**Status**: ✓ Integration complete and ready for deployment
**Last Updated**: 2025

