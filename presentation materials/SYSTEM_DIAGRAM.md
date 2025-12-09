# V1 Vision System - Visual Architecture Diagrams

## Complete System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RASPBERRY PI CAMERA                                   │
│                    320×240 @ 15 FPS via TCP/H.264                           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Raw video stream
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PREPROCESSING                                Time: ~1-2 ms        │
│  ─────────────────────────                                                  │
│  • Convert BGR → Grayscale                                                  │
│  • Gaussian blur (3×3 kernel, σ=auto)                                      │
│  • Contrast normalization (cv2.NORM_MINMAX)                                │
│  • Output: 320×240 normalized grayscale                                    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Preprocessed frame
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: GABOR FEATURE EXTRACTION                   Time: ~5-10 ms        │
│  ──────────────────────────────────                                         │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Gabor 0° │  │ Gabor 45°│  │ Gabor 90°│  │Gabor 135°│                  │
│  │ (horiz.) │  │(diagonal)│  │ (vert.)  │  │(diagonal)│                  │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘                  │
│        │             │             │             │                          │
│        ▼             ▼             ▼             ▼                          │
│  [Conv 320×240] [Conv 320×240] [Conv 320×240] [Conv 320×240]              │
│        │             │             │             │                          │
│        ▼             ▼             ▼             ▼                          │
│  ┌────────────────────────────────────────────────────┐                    │
│  │   RETINOTOPIC GRID CREATION (12×12)                │                    │
│  │   • Receptive field size: 20×20 pixels             │                    │
│  │   • Overlap: 50%                                   │                    │
│  │   • Response: MAX(|filtered values|) in RF         │                    │
│  └────────────────────────────────────────────────────┘                    │
│        │             │             │             │                          │
│        ▼             ▼             ▼             ▼                          │
│    [12×12]       [12×12]       [12×12]       [12×12]                       │
│   144 values    144 values    144 values    144 values                     │
│                                                                              │
│  Total: 576 feature neurons (144 per orientation)                          │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Feature grids
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: SPIKE ENCODER (Latency Coding)            Time: ~1-2 ms          │
│  ─────────────────────────────────────────                                  │
│                                                                              │
│  For each of 144 neurons in each orientation:                              │
│  ┌───────────────────────────────────────────────────┐                     │
│  │ IF feature_strength > threshold (0.5):            │                     │
│  │   latency = 100ms - (strength × 100ms)            │                     │
│  │   latency += gaussian_noise(0, 0.3ms)             │                     │
│  │   spike_time = latency                            │                     │
│  │   RECORD: (neuron_id, spike_time)                 │                     │
│  └───────────────────────────────────────────────────┘                     │
│                                                                              │
│  Strong features (0.8-1.0) → Early spikes (0-20ms)                         │
│  Medium features (0.5-0.8) → Mid spikes (20-50ms)                          │
│  Weak features (<0.5)     → No spikes (below threshold)                    │
│                                                                              │
│  Output per orientation:                                                    │
│    neuron_ids:  [3, 7, 12, 15, 18, ...]                                   │
│    spike_times: [5.2, 12.8, 23.1, 31.5, 47.2, ...] ms                     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Spike trains (4 orientations)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: V1 SIMULATION (Core)                    Time: ~100-200 ms        │
│  ──────────────────────────────                                             │
│                                                                              │
│  ╔═══════════════════╗  ╔═══════════════════╗                              │
│  ║   COLUMN 0°       ║  ║   COLUMN 45°      ║                              │
│  ╠═══════════════════╣  ╠═══════════════════╣                              │
│  ║ Layer 2/3 Output  ║  ║ Layer 2/3 Output  ║                              │
│  ║ ┌───────────────┐ ║  ║ ┌───────────────┐ ║                              │
│  ║ │ 144 Pyramidal │ ║  ║ │ 144 Pyramidal │ ║  ← Primary Output            │
│  ║ │ 65 Inhibitory │ ║  ║ │ 65 Inhibitory │ ║                              │
│  ║ └───────────────┘ ║  ║ └───────────────┘ ║                              │
│  ║        ▲          ║  ║        ▲          ║                              │
│  ║ ┌──────┴────────┐ ║  ║ ┌──────┴────────┐ ║                              │
│  ║ │ Layer 4 Input │ ║  ║ │ Layer 4 Input │ ║                              │
│  ║ │ 144 Sp.Stell. │ ║  ║ │ 144 Sp.Stell. │ ║  ← Receives LGN spikes       │
│  ║ │ 65 Inhibitory │ ║  ║ │ 65 Inhibitory │ ║                              │
│  ║ └───────────────┘ ║  ║ └───────────────┘ ║                              │
│  ║        ▲          ║  ║        ▲          ║                              │
│  ║ [LGN spikes 0°]   ║  ║ [LGN spikes 45°]  ║                              │
│  ╚═══════════════════╝  ╚═══════════════════╝                              │
│                                                                              │
│  ╔═══════════════════╗  ╔═══════════════════╗                              │
│  ║   COLUMN 90°      ║  ║   COLUMN 135°     ║                              │
│  ╠═══════════════════╣  ╠═══════════════════╣                              │
│  ║ Layer 2/3 Output  ║  ║ Layer 2/3 Output  ║                              │
│  ║ ┌───────────────┐ ║  ║ ┌───────────────┐ ║                              │
│  ║ │ 144 Pyramidal │ ║  ║ │ 144 Pyramidal │ ║                              │
│  ║ │ 65 Inhibitory │ ║  ║ │ 65 Inhibitory │ ║                              │
│  ║ └───────────────┘ ║  ║ └───────────────┘ ║                              │
│  ║        ▲          ║  ║        ▲          ║                              │
│  ║ ┌──────┴────────┐ ║  ║ ┌──────┴────────┐ ║                              │
│  ║ │ Layer 4 Input │ ║  ║ │ Layer 4 Input │ ║                              │
│  ║ │ 144 Sp.Stell. │ ║  ║ │ 144 Sp.Stell. │ ║                              │
│  ║ │ 65 Inhibitory │ ║  ║ │ 65 Inhibitory │ ║                              │
│  ║ └───────────────┘ ║  ║ └───────────────┘ ║                              │
│  ║        ▲          ║  ║        ▲          ║                              │
│  ║ [LGN spikes 90°]  ║  ║ [LGN spikes 135°] ║                              │
│  ╚═══════════════════╝  ╚═══════════════════╝                              │
│                                                                              │
│  Each column also has Layer 5 (81 neurons) and Layer 6 (243 neurons)       │
│  Total: 807 neurons × 4 columns = 3,228 neurons                            │
│                                                                              │
│  Simulation: 300 time steps × 0.5ms = 150ms                                │
│    • Warmup: 50ms (spontaneous activity)                                   │
│    • Stimulus: 100ms (process spikes)                                      │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ Firing rates (144 × 4 orientations)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: DECODER & VISUALIZATION                   Time: ~5-10 ms         │
│  ──────────────────────────────────                                         │
│                                                                              │
│  Read Layer 2/3 firing rates from all 4 columns:                           │
│  ┌──────────────────────────────────────────────────┐                      │
│  │ For each grid position (i, j) in 12×12:         │                      │
│  │   rate_0   = column_0_layer23[i×12+j]           │                      │
│  │   rate_45  = column_45_layer23[i×12+j]          │                      │
│  │   rate_90  = column_90_layer23[i×12+j]          │                      │
│  │   rate_135 = column_135_layer23[i×12+j]         │                      │
│  │                                                   │                      │
│  │   preferred = argmax(rate_0, rate_45, ...)      │  ← Winner-take-all    │
│  │   strength = max(rate_0, rate_45, ...)          │                      │
│  │                                                   │                      │
│  │   orientation_map[i, j] = preferred             │                      │
│  │   strength_map[i, j] = strength                 │                      │
│  └──────────────────────────────────────────────────┘                      │
│                                                                              │
│  Visualizations:                                                            │
│  • Orientation map (color-coded: R=0°, G=45°, B=90°, Y=135°)              │
│  • Edge map (oriented line segments)                                       │
│  • Layer activity heatmaps                                                 │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │  DISPLAY OUTPUT  │
                          │  1920×1080 panel │
                          └──────────────────┘
```

## Single Neuron Detail: LIF Dynamics

```
┌───────────────────────────────────────────────────────────────┐
│                 LEAKY INTEGRATE-AND-FIRE NEURON                │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  State Variables:                                             │
│  • V_membrane = -65.0 mV (membrane potential)                 │
│  • I_syn_ex = 0.0 (excitatory synaptic current)              │
│  • I_syn_in = 0.0 (inhibitory synaptic current)              │
│  • refractory_counter = 0 ms                                  │
│                                                                │
│  ┌────────────────────────────────────────────────┐          │
│  │        MEMBRANE POTENTIAL DYNAMICS              │          │
│  │                                                 │          │
│  │  dV/dt = (-(V - V_rest) + I_syn_ex - I_syn_in) │          │
│  │          ────────────────────────────────────── │          │
│  │                    τ_membrane                   │          │
│  │                                                 │          │
│  │  Numerical integration (Euler):                │          │
│  │    V(t+Δt) = V(t) + dV/dt × Δt                │          │
│  │    Δt = 0.5 ms                                 │          │
│  └────────────────────────────────────────────────┘          │
│                                                                │
│  ┌────────────────────────────────────────────────┐          │
│  │      SYNAPTIC CURRENT DYNAMICS                  │          │
│  │                                                 │          │
│  │  I_syn(t+Δt) = I_syn(t) × exp(-Δt/τ_syn)      │          │
│  │                                                 │          │
│  │  When spike received:                          │          │
│  │    I_syn_ex += weight (if weight > 0)          │          │
│  │    I_syn_in += |weight| (if weight < 0)        │          │
│  └────────────────────────────────────────────────┘          │
│                                                                │
│  ┌────────────────────────────────────────────────┐          │
│  │         SPIKE GENERATION                        │          │
│  │                                                 │          │
│  │  IF V ≥ V_threshold (-50 mV):                 │          │
│  │    1. Emit spike                               │          │
│  │    2. V ← V_reset (-65 mV)                    │          │
│  │    3. refractory_counter ← 2 ms                │          │
│  │    4. Propagate to connected neurons           │          │
│  └────────────────────────────────────────────────┘          │
│                                                                │
│  Parameters:                                                  │
│  • V_rest = -65 mV                                           │
│  • V_threshold = -50 mV                                      │
│  • V_reset = -65 mV                                          │
│  • τ_membrane = 10 ms                                        │
│  • τ_syn_ex = 2 ms                                           │
│  • τ_syn_in = 2 ms                                           │
│  • refractory_period = 2 ms                                  │
└───────────────────────────────────────────────────────────────┘
```

## Column Architecture Detail

```
┌─────────────────────────────────────────────────────────────────┐
│           SINGLE ORIENTATION COLUMN (807 neurons)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LAYER 6 (Deepest)                  292 neurons          │  │
│  │  • 243 Pyramidal (feedback to thalamus)                  │  │
│  │  • 49 Inhibitory                                         │  │
│  │  Grid: 9×27 (non-square for biological accuracy)         │  │
│  └────────────────────────────▲─────────────────────────────┘  │
│                                │                                │
│                          Feedforward                            │
│                          (indegree=20,                          │
│                           weight=50)                            │
│                                │                                │
│  ┌────────────────────────────┴─────────────────────────────┐  │
│  │  LAYER 5 (Subcortical Output)      97 neurons           │  │
│  │  • 81 Pyramidal (to superior colliculus, pons)          │  │
│  │  • 16 Inhibitory                                         │  │
│  │  Grid: 9×9                                               │  │
│  │  Recurrent: indegree=10, weight=0 (DISABLED)            │  │
│  └────────────────────────────▲─────────────────────────────┘  │
│                                │                                │
│                          Feedforward                            │
│                          (indegree=15,                          │
│                           weight=50)                            │
│                                │                                │
│  ┌────────────────────────────┴─────────────────────────────┐  │
│  │  LAYER 2/3 (Primary Output) ★      209 neurons          │  │
│  │  • 144 Pyramidal (to V2, V4, MT, other cortical areas)  │  │
│  │  • 65 Inhibitory                                         │  │
│  │  Grid: 12×12 (retinotopic)                              │  │
│  │  Recurrent: indegree=36, weight=0 (DISABLED)            │  │
│  │                                                           │  │
│  │  ★ This is the primary output we decode                 │  │
│  └────────────────────────────▲─────────────────────────────┘  │
│                                │                                │
│                          Feedforward                            │
│                          (polychrony groups:                    │
│                           4 L4 SS → 4 L2/3 Pyr,                │
│                           weight=50)                            │
│                                │                                │
│  ┌────────────────────────────┴─────────────────────────────┐  │
│  │  LAYER 4 (Primary Input) ★         209 neurons          │  │
│  │  • 144 Spiny Stellate (main input)                      │  │
│  │  • 65 Inhibitory                                         │  │
│  │  Grid: 12×12 (retinotopic)                              │  │
│  │  No recurrence in SS cells                               │  │
│  │                                                           │  │
│  │  ★ This is where LGN spikes arrive                      │  │
│  └────────────────────────────▲─────────────────────────────┘  │
│                                │                                │
│                          LGN INPUT                              │
│                          (weight=5000.0)                        │
│                          Very strong!                           │
│                                │                                │
│                   [Spike trains from encoder]                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Connectivity Summary:
─────────────────────
FEEDFORWARD PATH:
  LGN → L4 SS → L2/3 Pyr → L5 Pyr → L6 Pyr

LATERAL (within-layer):
  • L4: SS ↔ Inhibitory (DISABLED: weight=0)
  • L2/3: Pyramidal ↔ Inhibitory (DISABLED: weight=0)
  • L5: Pyramidal ↔ Inhibitory (DISABLED: weight=0)
  • L6: Pyramidal ↔ Inhibitory (DISABLED: weight=0)

RECURRENT (within population):
  • L2/3 Pyramidal: indegree=36 (DISABLED: weight=0)
  • L5 Pyramidal: indegree=10 (DISABLED: weight=0)
  • L6 Pyramidal: indegree=20 (DISABLED: weight=0)

Note: Lateral and recurrent connections are disabled in current version
      for debugging. Only feedforward path is active.
```

## Spike Timing Example

```
TIME EVOLUTION OF A SINGLE NEURON
─────────────────────────────────

t = 0.0 ms
│ Warmup phase begins
│ V = -65.0 mV (at rest)
│ I_syn_ex = 0
│ No spikes
│
├─────────────────────────────────────────────
│ [50 ms of spontaneous activity]
├─────────────────────────────────────────────
│
t = 50.0 ms
│ Stimulus begins
│ V = -64.8 mV (small fluctuations from noise)
│ I_syn_ex = 0
│ Waiting for LGN input...
│
t = 73.5 ms
│ ★ LGN SPIKE ARRIVES! (weight = 5000.0)
│ I_syn_ex += 5000.0
│ I_syn_ex = 5000.0
│
t = 74.0 ms (dt = 0.5ms later)
│ dV/dt = (0 + 5000.0) / 10.0 = 500.0 mV/ms
│ V = -65.0 + 500.0 × 0.5 = 185.0 mV
│ 
│ ★★★ V > threshold! SPIKE! ★★★
│ • Reset V = -65.0 mV
│ • Refractory = 2.0 ms
│ • Propagate spike to Layer 2/3 (weight=50.0 each)
│
│ I_syn_ex = 5000.0 × exp(-0.5/2.0)
│          = 5000.0 × 0.7788
│          = 3894.0
│
t = 74.5 ms
│ In refractory period
│ V = -65.0 mV (held at V_rest)
│ I_syn_ex = 3894.0 × exp(-0.5/2.0) = 3032.0
│
t = 75.0 ms
│ In refractory period
│ V = -65.0 mV
│ I_syn_ex = 3032.0 × exp(-0.5/2.0) = 2361.0
│
t = 76.0 ms
│ Out of refractory period
│ V = -65.0 mV
│ I_syn_ex = 1431.0 (decayed)
│ Now sensitive to new inputs...
│
├─────────────────────────────────────────────
│ [Continue until t = 150ms]
├─────────────────────────────────────────────
│
t = 150.0 ms
│ Stimulus phase ends
│ Calculate firing rate:
│   Spikes: [74.0, 89.2, 112.5, 138.7] ms
│   Count: 4 spikes in 100ms window
│   Rate: 4 / 0.1s = 40 Hz
│
└─────────────────────────────────────────────
```

## Orientation Map Generation

```
DECODER: WINNER-TAKE-ALL AT EACH LOCATION
──────────────────────────────────────────

Grid position (0, 0):
┌──────────────────────────────────┐
│ Column 0°:   rate = 15 Hz        │
│ Column 45°:  rate = 8 Hz         │
│ Column 90°:  rate = 42 Hz   ★    │ ← Winner!
│ Column 135°: rate = 11 Hz        │
└──────────────────────────────────┘
Result: orientation_map[0,0] = 90°
        strength_map[0,0] = 42 Hz
        color = BLUE

Grid position (0, 1):
┌──────────────────────────────────┐
│ Column 0°:   rate = 67 Hz   ★    │ ← Winner!
│ Column 45°:  rate = 23 Hz        │
│ Column 90°:  rate = 12 Hz        │
│ Column 135°: rate = 19 Hz        │
└──────────────────────────────────┘
Result: orientation_map[0,1] = 0°
        strength_map[0,1] = 67 Hz
        color = RED

... repeat for all 144 positions ...

FINAL OUTPUT (12×12 grid):
┌─────────────────────────────────────┐
│  0° 45° 90° 135° 0° ...             │
│ [R] [G] [B] [Y] [R] ...             │
│  └── colors represent orientations  │
│                                      │
│  Brightness = normalized strength   │
│  Dark = weak response               │
│  Bright = strong response           │
└─────────────────────────────────────┘

Upscaled to 360×360 for display
```

## Performance Breakdown Visualization

```
TIMING PER FRAME (Total: ~330 ms)
─────────────────────────────────

Preprocessing         ▌ 1-2 ms (0.5%)
Gabor Extraction      ▌▌▌ 5-10 ms (2%)
Spike Encoding        ▌ 1-2 ms (0.5%)
V1 Simulation         ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌ 100-200 ms (60%)
Decoding              ▌▌▌ 5-10 ms (2%)
Visualization         ▌▌▌▌▌▌ 20-40 ms (10%)
Other                 ▌▌▌ ~10-20 ms (5%)

0ms     50ms    100ms   150ms   200ms   250ms   300ms   350ms
├───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│                       │       V1 BOTTLENECK    │       │

FPS: ~3 frames/second

V1 SIMULATION BREAKDOWN (200 ms):
────────────────────────────────
• Warmup: 50ms (100 time steps)
  - Let network settle
  - Spontaneous activity
  - Run only on first frame

• Stimulus: 100ms (200 time steps)
  - Process LGN spikes
  - Update 3,228 neurons × 200 times
  - Propagate spikes through connections
  - Main computational load

• Each time step:
  - Check for incoming spikes
  - Update membrane potentials (3,228 equations)
  - Check for threshold crossings
  - Propagate new spikes
  - Decay synaptic currents
```

## System Scale Visualization

```
COMPUTATIONAL SCALE
───────────────────

Per Frame Processing:
┌─────────────────────────────────────────┐
│ Pixels processed:     76,800            │
│ Gabor convolutions:   4                 │
│ Feature neurons:      576               │
│ Spikes generated:     ~200-400          │
│ V1 neurons:           3,228             │
│ Time steps:           300               │
│ Neuron updates:       968,400           │
│ Synaptic connections: ~100,000          │
│ Spike propagations:   ~50,000           │
│                                          │
│ Total operations:     ~324 million      │
└─────────────────────────────────────────┘

Biological Comparison:
┌─────────────────────────────────────────┐
│                  Model    |    Real V1  │
├─────────────────────────────────────────┤
│ Neurons:         3,228    |  150 million│
│ Synapses:        100K     |  100 billion│
│ Processing time: 150ms    |  ~50ms      │
│ Accuracy:        Good     |  Perfect    │
└─────────────────────────────────────────┘

Data Flow Size:
┌─────────────────────────────────────────┐
│ Stage 1: 320×240×3 = 230 KB             │
│ Stage 2: 12×12×4 = 576 floats = 2.3 KB │
│ Stage 3: ~300 spikes × 2 = 2.4 KB      │
│ Stage 4: 3,228 rates × 1 = 12.9 KB     │
│ Stage 5: 12×12×2 = 1.2 KB              │
│                                          │
│ Total memory: ~3 MB                     │
└─────────────────────────────────────────┘
```

## Code Organization Map

```
v1_computational/
│
├── realtime_pipeline.py ──────► Entry point
│   └── main()
│       └── calls ↓
│
├── pipeline.py ───────────────► Orchestrator
│   ├── __init__()
│   │   ├── Creates: GaborFeatureExtractor
│   │   ├── Creates: SpikeEncoder
│   │   ├── Creates: ComputationalV1Model
│   │   └── Creates: V1Decoder
│   │
│   ├── process_frame()
│   │   ├── 1. _preprocess_frame()
│   │   ├── 2. gabor_extractor.extract_features()
│   │   ├── 3. spike_encoder.encode_features()
│   │   ├── 4. v1_model.run_stimulus()
│   │   └── 5. decoder.decode_v1_output()
│   │
│   └── run_on_video_stream()
│       └── Loop: process_frame() for each frame
│
├── gabor_extractor.py ────────► Stage 1
│   ├── __init__(): Create filter bank
│   ├── extract_features(): Apply filters
│   └── _create_retinotopic_grid(): 12×12 grid
│
├── spike_encoder.py ──────────► Stage 2
│   ├── encode_features_to_spikes()
│   ├── _latency_encoding(): Strong → early
│   └── _rate_encoding(): Strong → many
│
├── v1_model.py ───────────────► Stage 3 (Main)
│   ├── __init__()
│   │   └── Creates 4 × V1OrientationColumn
│   │
│   ├── inject_spike_trains(): Route to columns
│   │
│   ├── run_stimulus()
│   │   ├── Warmup loop (50ms)
│   │   ├── Inject spikes
│   │   ├── Stimulus loop (100ms)
│   │   │   └── column.update() for each column
│   │   └── get_results()
│   │
│   └── get_results(): Collect firing rates
│
├── v1_column.py ──────────────► Single Column
│   ├── __init__()
│   │   ├── Creates: layer_4_ss (144 neurons)
│   │   ├── Creates: layer_4_inh (65 neurons)
│   │   ├── Creates: layer_23_pyr (144 neurons)
│   │   ├── Creates: layer_23_inh (65 neurons)
│   │   ├── Creates: layer_5_pyr (81 neurons)
│   │   ├── Creates: layer_5_inh (16 neurons)
│   │   ├── Creates: layer_6_pyr (243 neurons)
│   │   ├── Creates: layer_6_inh (49 neurons)
│   │   └── Sets up connections
│   │
│   ├── inject_lgn_spikes(): L4 input
│   │
│   ├── update(): One time step
│   │   ├── Update Layer 4
│   │   ├── Propagate spikes
│   │   ├── Update Layer 2/3
│   │   ├── Propagate spikes
│   │   ├── Update Layer 5
│   │   ├── Propagate spikes
│   │   └── Update Layer 6
│   │
│   └── get_layer_firing_rates(): Extract results
│
├── neurons.py ────────────────► Neuron Models
│   ├── LIFNeuron
│   │   ├── update(): Integrate & fire
│   │   └── receive_spike(): Add current
│   │
│   └── NeuronPopulation
│       ├── update(): Update all neurons
│       ├── add_recurrent_connections()
│       └── get_firing_rates()
│
├── v1_decoder.py ─────────────► Stage 4
│   ├── decode_v1_output()
│   │   ├── _create_orientation_map(): Winner-take-all
│   │   ├── _create_strength_map(): Max responses
│   │   └── _visualize_orientation_map(): Colors
│   │
│   └── visualize_layer_activity(): Heatmaps
│
└── config.py ─────────────────► Parameters
    ├── VIDEO_CONFIG: Camera settings
    ├── GRID_CONFIG: 12×12 grid
    ├── GABOR_CONFIG: Filter parameters
    ├── SPIKE_CONFIG: Encoding settings
    ├── V1_ARCHITECTURE: Neuron counts, weights
    └── DEBUG_CONFIG: Logging settings
```

---

## Summary Statistics

**Total System Complexity:**
- 5 processing stages
- 3,228 neurons
- 8 neuron populations per column
- 4 orientation columns
- ~100,000 synaptic connections
- 300 simulation time steps
- ~324 million operations per frame
- ~3 FPS processing speed
- Biologically accurate neuron dynamics
- Real-time video processing

**Key Innovation:** Not computer vision algorithms, but actual computational neuroscience - simulating how the brain sees.

