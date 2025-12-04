# V1 Computational Architecture

## Overview

This document describes the computational V1 model architecture, which exactly replicates the MDPI2021 V1 orientation column structure.

## Model Structure

### Orientation Columns

The model contains **4 orientation-selective columns**:
- 0° (horizontal)
- 45° (diagonal /)
- 90° (vertical)
- 135° (diagonal \)

Each column processes visual input preferentially for its orientation.

### Layer Architecture (Per Column)

#### Layer 4 (Input Layer)
- **Spiny Stellate Cells**: 324 neurons (18x18 retinotopic grid)
  - Receive direct input from LGN (spike trains from Gabor filters)
  - Strong synaptic weight: 15,000
  - Main input gateway to V1
  
- **Inhibitory Interneurons**: 65 neurons
  - Provide lateral inhibition
  - Sharpen orientation selectivity
  - Indegree: 32 from SS, 6 recurrent

#### Layer 2/3 (Output Layer)
- **Pyramidal Cells**: 324 neurons (18x18 grid)
  - Primary output of V1
  - Strong recurrent connections (indegree: 36)
  - Orientation-selective responses
  - Projects to higher visual areas
  
- **Inhibitory Interneurons**: 65 neurons
  - Balance excitation
  - Indegree: 35 from pyramidal, 8 recurrent

#### Layer 5 (Deep Layer)
- **Pyramidal Cells**: 81 neurons (9x9 grid)
  - Receives from Layer 2/3 (indegree: 15)
  - Projects to Layer 6
  - Reduced spatial resolution
  - Recurrent connections (indegree: 10)
  
- **Inhibitory Interneurons**: 16 neurons

#### Layer 6 (Deepest Layer)
- **Pyramidal Cells**: 243 neurons (9x27 grid)
  - Receives from Layer 5 (indegree: 20)
  - Feedback to thalamus (not modeled)
  - Recurrent connections (indegree: 20)
  
- **Inhibitory Interneurons**: 49 neurons

### Total Neuron Count

Per column: **1,167 neurons**
- Layer 4: 389 (324 SS + 65 Inh)
- Layer 2/3: 389 (324 Pyr + 65 Inh)
- Layer 5: 97 (81 Pyr + 16 Inh)
- Layer 6: 292 (243 Pyr + 49 Inh)

Total across 4 columns: **4,668 neurons**

## Connectivity

### Within-Layer Connections

**Layer 4:**
- SS → SS: None (no recurrent)
- SS → Inh: indegree=32, weight=100
- Inh → SS: indegree=6, weight=-100
- Inh → Inh: indegree=6, weight=-100

**Layer 2/3:**
- Pyr → Pyr: indegree=36, weight=100 (strong recurrence)
- Pyr → Inh: indegree=35, weight=100
- Inh → Pyr: indegree=8, weight=-100
- Inh → Inh: indegree=8, weight=-100

**Layer 5:**
- Pyr → Pyr: indegree=10, weight=100
- Pyr → Inh: indegree=30, weight=100
- Inh → Pyr: indegree=8, weight=-100
- Inh → Inh: indegree=8, weight=-100

**Layer 6:**
- Pyr → Pyr: indegree=20, weight=100
- Pyr → Inh: indegree=32, weight=100
- Inh → Pyr: indegree=6, weight=-100
- Inh → Inh: indegree=6, weight=-100

### Between-Layer Connections

**Layer 4 → Layer 2/3:**
- Special "polychrony detection" architecture
- Groups of 4 SS cells connect to groups of 4 Pyramidal cells
- Detects synchronous spike patterns
- Weight: 100

**Layer 2/3 → Layer 5:**
- Feedforward projection
- Indegree: 15
- Weight: 100

**Layer 5 → Layer 6:**
- Feedforward projection
- Indegree: 20
- Weight: 100

### Between-Column Connections

In this model: **None**

Each orientation column operates independently. In biological V1, there are horizontal connections between columns that could be added for:
- Cross-orientation suppression
- Contour integration
- Context modulation

## Neuron Model

### Leaky Integrate-and-Fire (LIF)

Membrane potential dynamics:
```
dV/dt = (-(V - V_rest) + I_total) / tau_m
```

Where:
- `V`: Membrane potential
- `V_rest`: Resting potential (-65 mV)
- `tau_m`: Membrane time constant (10 ms)
- `I_total`: Total synaptic current

### Synaptic Currents

**Excitatory:**
```
I_ex(t) = I_ex * exp(-t / tau_syn_ex)
```
- `tau_syn_ex`: 2 ms

**Inhibitory:**
```
I_in(t) = I_in * exp(-t / tau_syn_in)
```
- `tau_syn_in`: 2 ms

### Spike Generation

When `V >= V_threshold` (-50 mV):
1. Spike emitted
2. V reset to `V_reset` (-65 mV)
3. Refractory period (2 ms)

### Background Activity

Poisson noise simulates spontaneous synaptic bombardment:
- Layer 2/3: 1,721,500 Hz
- Layer 5: 1,740,000 Hz
- Layer 6: 1,700,000 Hz
- Inhibitory: 1,750,000 Hz
- Weight: 5.0

These high rates represent the summed activity of thousands of background synapses.

## Input Processing

### 1. Gabor Feature Extraction

- **Filters**: 4 orientations (0°, 45°, 90°, 135°)
- **Parameters**: 
  - Wavelength: 10 pixels
  - Sigma: 5 pixels
  - Gamma: 0.5
- **Output**: 18x18 grid per orientation

### 2. Spike Encoding (Latency Coding)

Strong features → Early spikes:
```
latency = max_latency - feature_strength * (max_latency - min_latency)
```
- Min latency: 43 ms (strong features)
- Max latency: 200 ms (weak features)
- Spike start: 50 ms

### 3. V1 Processing

**Timeline:**
1. **Warmup** (0-400 ms): Spontaneous activity only
2. **Stimulus** (400-600 ms): LGN spikes injected
3. **Analysis**: Measure firing rates during stimulus

### 4. Output Decoding

From Layer 2/3 firing rates:
- Extract preferred orientation at each grid position
- Weight by response strength
- Generate orientation map

## Comparison to MDPI2021

| Feature | MDPI2021 (NEST) | Computational |
|---------|----------------|---------------|
| **Neurons per column** | 1,167 | 1,167 ✓ |
| **Layer 4 SS** | 324 | 324 ✓ |
| **Layer 2/3 Pyr** | 324 | 324 ✓ |
| **Layer 5 Pyr** | 81 | 81 ✓ |
| **Layer 6 Pyr** | 243 | 243 ✓ |
| **Recurrent indegree L2/3** | 36 | 36 ✓ |
| **LGN weight** | 15,000 | 15,000 ✓ |
| **Poisson rates** | Matched | Matched ✓ |
| **Simulation engine** | NEST | NumPy |
| **Time step** | 0.1 ms | 0.1 ms ✓ |
| **Neuron model** | lifl_psc_exp_ie | LIF (simplified) |

## Performance

**Simulation time per frame:**
- Warmup: 400 ms @ 0.1 ms steps = 4,000 steps
- Stimulus: 200 ms @ 0.1 ms steps = 2,000 steps
- **Total: 6,000 simulation steps**

Wall-clock time: ~50 seconds per frame on typical CPU

**Speedup options:**
1. Reduce stimulus time (200ms → 100ms): 2x faster
2. Skip warmup for continuous video: 1.7x faster
3. Increase time step (0.1ms → 0.5ms): 5x faster (less accurate)

## Biological Realism

**What's realistic:**
- ✅ Layer structure (4, 2/3, 5, 6)
- ✅ Neuron counts (scaled from real V1)
- ✅ Connectivity patterns
- ✅ Orientation selectivity
- ✅ Retinotopic organization
- ✅ Spike-based processing
- ✅ Background activity

**What's simplified:**
- ⚠️ Neuron model (LIF vs detailed biophysics)
- ⚠️ No between-column connections
- ⚠️ No feedback from higher areas
- ⚠️ No temporal dynamics beyond single frame
- ⚠️ Simplified synaptic dynamics

## Extensions

Possible enhancements:
1. **Lateral connections** between columns
2. **Feedback** from higher visual areas
3. **Temporal integration** across frames
4. **Additional cell types** (Martinotti, Chandelier, etc.)
5. **Dendritic computation**
6. **Synaptic plasticity** (STDP, BCM)
7. **More detailed neuron models** (HH, AdEx)

## References

- MDPI2021 repository: `OrientedColumnV1.py`
- Simulation script: `Simulation_V1_pinwheel_MEGcomparison.py`
- NEST neuron models: `lifl_psc_exp_ie`, `aeif_psc_exp_peak`

