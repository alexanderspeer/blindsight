# Technical Report Materials: Real-Time Synthetic Visual Cortex

**Project**: Computational V1 Vision Pipeline for Real-Time Video Processing  
**Based on**: MDPI2021 V1 Orientation Column Model  
**Implementation**: Pure Python computational neuroscience system

---

# 1. Repository Overview

## 1.1 File Tree

```
blindsight/
├── v1_computational/           # Main computational V1 implementation
│   ├── config.py              # Centralized configuration (all parameters)
│   ├── neurons.py             # LIF neuron models and populations
│   ├── v1_column.py          # Single orientation column (1,167 neurons)
│   ├── v1_model.py           # Complete 4-column V1 (4,668 neurons)
│   ├── gabor_extractor.py    # Gabor filter feature extraction
│   ├── spike_encoder.py      # Feature to spike train conversion
│   ├── v1_decoder.py         # V1 activity to orientation map decoder
│   ├── pipeline.py           # Complete processing pipeline orchestration
│   ├── realtime_pipeline.py  # Real-time video stream processing
│   ├── test_static_image.py  # Static image testing with diagnostics
│   ├── validate_config.py    # Configuration validation
│   ├── requirements.txt      # Python dependencies (NumPy, OpenCV)
│   ├── README.md            # Complete user guide
│   ├── ARCHITECTURE.md      # Detailed architecture documentation
│   ├── V1_FIX_COMPLETE_SUMMARY.md  # Weight tuning and fixes
│   ├── GABOR_SPARSIFICATION_SUMMARY.md  # Sparsification improvements
│   ├── TESTING_GUIDE.md     # Comprehensive testing instructions
│   ├── SUMMARY.md           # Quick reference summary
│   └── QUICKSTART.md        # Fast start guide
│
├── MDPI2021/                 # Original NEST-based reference model
│   ├── Examples/
│   │   └── V1 Oriented Columns comapred with MEG/
│   │       ├── OrientedColumnV1.py  # Reference V1 column architecture
│   │       └── Simulation_V1_pinwheel_MEGcomparison.py
│   ├── LIFL_IE/             # Custom NEST C++ module (intrinsic excitability)
│   │   ├── lifl_psc_exp_ie.cpp/.h  # IE neuron model
│   │   └── aeif_psc_exp_peak.cpp/.h  # Adaptive exponential IF
│   └── README.md
│
├── presentation materials/   # Presentation and comparison documents
│   ├── MDPI2021_VS_COMPUTATIONAL_COMPARISON.md
│   ├── TECHNICAL_PRESENTATION.md
│   └── SYSTEM_DIAGRAM.md
│
└── init-cv/                 # Initial camera/CV experiments
    ├── 12-6.1.py            # Early video processing tests
    └── receive.py           # Pi camera stream reception
```

## 1.2 Key Modules

### Core Library Modules

**`config.py`** (178 lines)  
- Single source of truth for all system parameters
- VIDEO_CONFIG: Pi camera connection (IP, port, resolution: 320×240@15fps)
- GRID_CONFIG: Spatial grid (12×12 = 144 neurons per layer)
- GABOR_CONFIG: Filter parameters (4 orientations, wavelength=10, sigma=5)
- SPIKE_CONFIG: Latency encoding (0-100ms range, threshold=0.5)
- V1_ARCHITECTURE: Neuron counts, weights, connectivity parameters
- DEBUG_CONFIG: Comprehensive debugging flags
- **Critical weights**: LGN→L4=5000, L4→L2/3=120, L2/3→L5=150, L5→L6=150

**`neurons.py`** (312 lines)  
- `LIFNeuron`: Leaky integrate-and-fire neuron model
- `PoissonNoise`: Background activity generator (currently disabled)
- `NeuronPopulation`: Population-level management with recurrent connectivity
- Implements Euler integration for membrane dynamics
- Synaptic current exponential decay (tau_syn_ex=2ms, tau_syn_in=2ms)

**`v1_column.py`** (515 lines)  
- `V1OrientationColumn`: Single orientation-selective column
- 8 neuron populations per column: L4 SS, L4 Inh, L2/3 Pyr, L2/3 Inh, L5 Pyr, L5 Inh, L6 Pyr, L6 Inh
- Implements polychronous groups (4 L4 SS → 4 L2/3 Pyr)
- Layer 2/3 enhanced excitability (v_thresh=-55mV, tau_m=25ms, bias=20pA)
- Feedforward connectivity with separate tunable weights
- Diagnostic methods for synaptic currents and spike flow

**`v1_model.py`** (369 lines)  
- `ComputationalV1Model`: Complete 4-column V1 (0°, 45°, 90°, 135°)
- Total neurons: 4,668 (1,167 per column)
- Manages simulation loop: warmup (50ms) + stimulus (100ms)
- Injects orientation-specific spike trains into appropriate columns
- Collects firing rates from all layers during analysis window
- Calculates orientation selectivity index (OSI)

**`gabor_extractor.py`** (367 lines)  
- `GaborFeatureExtractor`: Multi-orientation Gabor filtering
- Creates 4 Gabor kernels (0°, 45°, 90°, 135°) using cv2.getGaborKernel
- Extracts 12×12 retinotopic grid from filtered images
- Receptive field size: 20 pixels with 50% overlap
- Sparsification pipeline: z-score normalization → clip negatives → rescale [0,3] → percentile threshold (80th)
- Orientation competition via softmax (temperature=0.5) across orientations
- Produces ~10-30% active cells per orientation (down from 97-99%)

**`spike_encoder.py`** (218 lines)  
- `SpikeEncoder`: Converts features to spike trains
- Primary mode: **Latency coding** (strong features → early spikes)
- Latency formula: `latency = max_latency - feature_strength * (max_latency - min_latency)`
- Range: 0-100ms (spike_start=0, min_latency=0, max_latency=100)
- Threshold: 0.5 (only strong features generate spikes)
- Jitter: 0.3ms for biological realism
- Also implements rate coding (unused)

**`v1_decoder.py`** (306 lines)  
- `V1Decoder`: Reconstructs orientation maps from V1 firing rates
- Extracts Layer 2/3 firing rates by default (primary output layer)
- Creates orientation preference map (winner-take-all per spatial location)
- Creates response strength map (max firing rate across orientations)
- Visualization modes: color-coded orientation map, oriented edge map
- Layer activity visualization for all 4 layers

**`pipeline.py`** (747 lines)  
- `V1VisionPipeline`: Complete end-to-end processing
- Orchestrates: preprocessing → Gabor → encoding → V1 → decoding
- Preprocessing: Gaussian blur (kernel=3), contrast normalization, no downsampling
- Comprehensive debug output (controlled by DEBUG_CONFIG)
- Visualization caching (updates every 5 seconds to avoid overhead)
- Combined panel display (3×2 grid: Raw, Comparison, V1 Color, Gabor, Spikes, Layers)
- Timing breakdown per stage with FPS calculation

### Entrypoint Scripts

**`test_static_image.py`** (253 lines)  
- Creates synthetic test image with oriented edges (0°, 45°, 90°, 135°)
- Processes single frame with full diagnostics
- Prints detailed statistics for all pipeline stages:
  - Gabor: active cell percentages, value histograms
  - Spikes: spike counts, timing distributions, neuron coverage
  - V1: layer-wise firing rates, active neuron percentages
- Displays OSI (Orientation Selectivity Index) per column
- Shows all visualization windows
- **Use case**: Verify pipeline functionality and parameter tuning

**`realtime_pipeline.py`** (100 lines)  
- Real-time video processing from Pi camera or file
- Connects to Pi via ffmpeg with H.264 TCP stream
- Processes frames continuously with optional warmup on first 3 frames
- Displays combined visualization panel (1920×1080)
- Print debug info controlled by DEBUG_CONFIG
- **Use case**: Live demonstration and real-time processing

---

# 2. Complete System Architecture

## 2.1 High-level Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    INPUT: Video Stream                        │
│              (320×240 @ 15fps from Pi Camera)                │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              STAGE 1: PREPROCESSING                           │
│  - Gaussian blur (kernel=3×3)                                │
│  - Contrast normalization [0, 255]                           │
│  - Keep original resolution (no downsampling)                │
│  Output: 320×240 grayscale                                   │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│         STAGE 2: GABOR FEATURE EXTRACTION                     │
│  - 4 Gabor filters (0°, 45°, 90°, 135°)                     │
│  - Wavelength=10px, Sigma=5px, Gamma=0.5, kernel=31×31      │
│  - 12×12 retinotopic grid (RF size=20px, overlap=50%)       │
│  - Sparsification: z-score → percentile threshold (top 20%) │
│  - Orientation competition via softmax (temp=0.5)            │
│  Output: 4 × (12×12) feature grids, ~10-30% active          │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│            STAGE 3: SPIKE ENCODING                            │
│  - Latency coding: strong → early spikes                     │
│  - Threshold: 0.5 (only strong features)                     │
│  - Latency range: 0-100ms                                    │
│  - Jitter: 0.3ms                                             │
│  Output: 4 × spike trains (30-60 spikes per orientation)     │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│              STAGE 4: V1 SIMULATION                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  4 Orientation Columns (0°, 45°, 90°, 135°)          │ │
│  │  Each column: 8 populations, 1,167 neurons            │ │
│  │  Total: 4,668 neurons                                 │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  Layer 4:  144 SS + 65 Inh  (input layer)            │ │
│  │  Layer 2/3: 144 Pyr + 65 Inh (output layer)          │ │
│  │  Layer 5:   81 Pyr + 16 Inh                          │ │
│  │  Layer 6:   243 Pyr + 49 Inh                         │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  Simulation: 50ms warmup + 100ms stimulus            │ │
│  │  Time step: dt = 0.5ms                               │ │
│  │  Total steps: 300 per frame                          │ │
│  └────────────────────────────────────────────────────────┘ │
│  Output: Firing rates for all layers & orientations         │
└─────────────────────┬────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│            STAGE 5: DECODING & VISUALIZATION                  │
│  - Extract Layer 2/3 firing rates (primary output)          │
│  - Winner-take-all per spatial location → orientation map   │
│  - Color coding: 0°=Red, 45°=Green, 90°=Blue, 135°=Yellow  │
│  - Oriented edge visualization with line segments           │
│  - Layer activity heatmaps (L4, L2/3, L5, L6)              │
│  Output: Orientation/edge maps, visualizations              │
└──────────────────────────────────────────────────────────────┘
```

## 2.2 Data Types and Shapes Between Stages

| Stage | Data Structure | Shape/Format | Value Range | Notes |
|-------|---------------|--------------|-------------|-------|
| **Input** | `numpy.ndarray` | (240, 320, 3) BGR | [0, 255] uint8 | Raw video frame |
| **Preprocessing** | `numpy.ndarray` | (240, 320) grayscale | [0, 255] uint8 | After blur + normalization |
| **Gabor Full Response** | `dict[int→ndarray]` | 4 × (240, 320) float32 | [-∞, +∞] | Raw filter responses |
| **Gabor Features** | `dict[int→ndarray]` | 4 × (12, 12) float32 | [0, ~3.0] | After sparsification |
| **Spike Trains** | `dict[int→dict]` | neuron_ids: int[], spike_times: float[] | IDs: [0,143], times: [0,100]ms | 30-60 spikes typical |
| **V1 Membrane Potentials** | Per neuron | float scalar | [-65, -50] mV | Internal state |
| **V1 Synaptic Currents** | Per neuron | float scalar | [0, ~500] pA | i_syn_ex, i_syn_in |
| **V1 Firing Rates** | `dict[ori→dict[layer→ndarray]]` | Layer-specific shapes | [0, ~100] Hz | Analysis window rates |
| | Layer 4 & 2/3: | (144,) float | | 12×12 flattened |
| | Layer 5: | (81,) float | | 9×9 grid |
| | Layer 6: | (243,) float | | 9×27 grid |
| **Orientation Map** | `numpy.ndarray` | (12, 12) int | {-1, 0, 45, 90, 135} | -1 = no response |
| **Strength Map** | `numpy.ndarray` | (12, 12) float32 | [0, max_rate] Hz | Max rate per location |
| **Visualization** | `numpy.ndarray` | (360, 360, 3) BGR | [0, 255] uint8 | Upscaled for display |

---

# 3. Camera and Preprocessing

### 3.1 Modules and Functions

**Module**: `pipeline.py`  
**Class**: `V1VisionPipeline`  
**Method**: `_preprocess_frame(self, frame)`

**Path**: `v1_computational/pipeline.py`, lines 157-184

**Purpose**: Prepares raw video frames for Gabor filtering by reducing noise and normalizing intensity values.

**Input**: Raw video frame from camera
- Type: `numpy.ndarray`
- Shape: (height, width, 3) or (height, width)
- Range: [0, 255] uint8

**Output**: Preprocessed grayscale frame
- Type: `numpy.ndarray`
- Shape: (height, width)
- Range: [0, 255] uint8

**Video Stream Reception** (for Pi camera):

**Module**: `pipeline.py`  
**Method**: `run_on_video_stream(self, video_source=None)`  
**Lines**: 432-514

Uses ffmpeg to decode H.264 TCP stream from Raspberry Pi:
```python
ffmpeg_cmd = [
    'ffmpeg', '-fflags', 'nobuffer', '-flags', 'low_delay',
    '-analyzeduration', '0', '-probesize', '32',
    '-i', f'tcp://{VIDEO_CONFIG["pi_ip"]}:{VIDEO_CONFIG["port"]}?listen=0',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-',
]
```

### 3.2 Parameters

**From `config.py` → VIDEO_CONFIG:**
- `pi_ip`: '10.207.70.178' (Raspberry Pi IP address)
- `port`: 5001 (TCP port for H.264 stream)
- `width`: 320 pixels
- `height`: 240 pixels
- `fps`: 15 frames per second

**From `config.py` → PROCESSING_CONFIG:**
- `downsample_frame`: False (disabled - was upscaling and destroying edges)
- `normalize_contrast`: True
- `gaussian_blur_kernel`: 3 (3×3 kernel)

**Receptive Field Grid Creation**:
From `gabor_extractor.py`, lines 110-160:
- Grid dimensions: 12×12 = 144 neurons per orientation
- Receptive field (RF) size: 20×20 pixels
- Overlap: 50%
- Stride: stride_y = 10 pixels, stride_x = 13 pixels

### 3.3 Equations / Pseudocode

**Gaussian Blur**:
\[
I_{\text{blurred}}(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x+i, y+j) \cdot G(i, j)
\]
Where \(k = \lfloor \text{kernel\_size} / 2 \rfloor = 1\) for 3×3 kernel.

**Contrast Normalization**:
\[
I_{\text{norm}}(x, y) = 255 \cdot \frac{I(x, y) - I_{\min}}{I_{\max} - I_{\min}}
\]

**Receptive Field Mapping**: For grid position (row, col):
```
center_y = row * stride_y + RF_size // 2
center_x = col * stride_x + RF_size // 2
RF patch: [center_y - RF_size//2 : center_y + RF_size//2,
           center_x - RF_size//2 : center_x + RF_size//2]
```

---

# 4. Gabor Feature Extraction

### 4.1 Implementation Details

**Module**: `gabor_extractor.py`  
**Class**: `GaborFeatureExtractor`  
**Lines**: 1-367

**Main Method**: `extract_features(self, frame, apply_orientation_competition=True, verbose=False)`  
**Lines**: 50-108

**Process Flow**:
1. Convert frame to grayscale (if needed), normalize to [0, 1]
2. Apply Gaussian blur (5×5, sigma=1.0)
3. For each orientation (0°, 45°, 90°, 135°):
   - Apply Gabor filter via `cv2.filter2D()`
   - Create 12×12 retinotopic grid (lines 110-160)
   - Apply sparsification pipeline (lines 162-203)
4. Apply orientation competition across orientations (lines 205-235)
5. Return feature grids and full Gabor responses

**Gabor Kernel Creation** (lines 36-48):
```python
theta = np.deg2rad(orientation)
kernel = cv2.getGaborKernel(
    (31, 31),      # kernel_size
    5.0,           # sigma
    theta,         # orientation
    10.0,          # wavelength
    0.5,           # gamma
    0,             # psi
    ktype=cv2.CV_32F
)
```

**Receptive Field Aggregation** (lines 149-153):
```python
rf_patch = filtered_image[y_start:y_end, x_start:x_end]
# Response is MAX absolute value in receptive field
response = np.max(np.abs(rf_patch))
grid[row, col] = response
```

**Sparsification Pipeline** (lines 162-203):
1. Z-score normalization: `z = (grid - mean) / std`
2. Clip negatives: `z = max(0, z)`
3. Rescale to [0, 3]: `norm = 3 * z / z_max`
4. Percentile threshold (80th): Keep only top 20% of values

**Orientation Competition** (lines 205-235):
Softmax across orientations at each spatial location with temperature=0.5

### 4.2 Parameters and Equations

**From `config.py` → GABOR_CONFIG:**
- `orientations`: [0, 45, 90, 135] degrees
- `wavelength`: 10.0 pixels
- `sigma`: 5.0 pixels
- `gamma`: 0.5 (aspect ratio)
- `psi`: 0 (phase offset)
- `kernel_size`: 31×31 pixels

**From `config.py` → GRID_CONFIG:**
- `n_neurons`: 144 (12×12 grid)
- `grid_rows`: 12, `grid_cols`: 12
- `receptive_field_size`: 20 pixels
- `overlap`: 0.5 (50%)

**Gabor Filter Equation**:
\[
G(x, y; \theta, \lambda, \sigma, \gamma, \psi) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi\frac{x'}{\lambda} + \psi\right)
\]
where \(x' = x \cos\theta + y \sin\theta\), \(y' = -x \sin\theta + y \cos\theta\)

**Receptive Field Aggregation**:
\[
R_{ij}^{\theta} = \max_{(x,y) \in \text{RF}_{ij}} |G_{\theta}(x, y)|
\]

**Sparsification**:
\[
z_{ij} = \frac{R_{ij} - \mu}{\sigma}, \quad z_{ij}^{+} = \max(0, z_{ij}), \quad R_{ij}^{\text{norm}} = 3 \cdot \frac{z_{ij}^{+}}{\max(z_{ij}^{+})}
\]
\[
R_{ij}^{\text{sparse}} = \begin{cases}
R_{ij}^{\text{norm}} & \text{if } R_{ij}^{\text{norm}} \geq P_{80} \\
0 & \text{otherwise}
\end{cases}
\]

**Orientation Competition**:
\[
S_{ij}^{\theta} = \frac{\exp(R_{ij}^{\theta} / 0.5)}{\sum_{\theta'} \exp(R_{ij}^{\theta'} / 0.5)}, \quad R_{ij}^{\text{final}, \theta} = S_{ij}^{\theta} \cdot R_{ij}^{\theta}
\]

### 4.3 Output Statistics

**From test runs** (documented in `GABOR_SPARSIFICATION_SUMMARY.md`):

**After Sparsification + Competition**:
- Active cells: 10-30% per orientation (was 97-99% before)
- Mean response: ~0.3-0.6 (for active cells)
- Max response: ~2.0-3.0
- Clear orientation dominance at appropriate locations

**Example Orientation Dominance**:
- 0° dominant at 29% of locations
- 45° dominant at 24%
- 90° dominant at 26%
- 135° dominant at 20%

---

# 5. Spike Encoding

### 5.1 Latency Encoding Algorithm

**Module**: `spike_encoder.py`  
**Class**: `SpikeEncoder`  
**Lines**: 1-218

**Main Method**: `encode_features_to_spikes(self, features)` (lines 32-64)  
**Encoding Method**: `_latency_encoding(self, feature_array)` (lines 66-104)

**Purpose**: Convert Gabor feature strengths into temporally-coded spike trains where **stronger features generate earlier spikes** (latency coding).

**Algorithm**:
```python
def _latency_encoding(self, feature_array):
    neuron_ids = []
    spike_times = []
    
    for neuron_idx in range(len(feature_array)):  # 144 neurons
        feature_strength = feature_array[neuron_idx]
        
        # Only encode if above threshold
        if feature_strength > self.threshold:  # threshold = 0.5
            # Latency inversely proportional to feature strength
            # Strong features spike early, weak features spike late
            latency = self.max_latency - (feature_strength * 
                      (self.max_latency - self.min_latency))
            
            # Add jitter for biological realism
            latency += np.random.randn() * self.jitter  # jitter = 0.3ms
            
            # Clip to valid range
            latency = np.clip(latency, self.min_latency, self.max_latency)
            
            # Spike time relative to stimulus onset
            spike_time = self.spike_start + latency
            
            neuron_ids.append(neuron_idx)
            spike_times.append(spike_time)
    
    return {
        'neuron_ids': np.array(neuron_ids, dtype=int),
        'spike_times': np.array(spike_times, dtype=float)
    }
```

**Input**: Feature grid (12×12) for one orientation, normalized [0, 1]  
**Output**: Dict with `neuron_ids` (array of int) and `spike_times` (array of float in ms)

### 5.2 Other Encoding Modes

**Rate Encoding** (`_rate_encoding`, lines 106-142):
- Implemented but unused (encoding_type='latency' in config)
- Converts feature strength to firing rate
- Generates Poisson spike train over stimulus window
- Formula: `rate = min_rate + feature_strength * (max_rate - min_rate)`
- Time step: 1ms for Poisson process

### 5.3 Parameters and Ranges

**From `config.py` → SPIKE_CONFIG:**
- `encoding_type`: 'latency'
- `max_spike_rate`: 200.0 Hz (unused in latency mode)
- `min_spike_rate`: 10.0 Hz (unused in latency mode)
- `spike_window_ms`: 150.0 ms (visualization window)
- `spike_start_ms`: 0.0 ms (spikes start immediately with stimulus)
- `min_latency_ms`: 0.0 ms (strongest possible response)
- `max_latency_ms`: 100.0 ms (weakest response that spikes)
- `jitter_ms`: 0.3 ms (biological variability)
- `threshold`: 0.5 (only features > 0.5 generate spikes)

**Latency Encoding Equation**:
\[
t_{\text{spike}}^{i} = t_{\text{start}} + L_{\max} - f_i \cdot (L_{\max} - L_{\min}) + \mathcal{N}(0, \sigma_{\text{jitter}}^2)
\]

where:
- \(f_i\): Feature strength for neuron \(i\), normalized \(\in [0, 1]\)
- \(t_{\text{start}}\): Stimulus onset time = 0 ms
- \(L_{\min}\): Minimum latency = 0 ms (strong features)
- \(L_{\max}\): Maximum latency = 100 ms (weak features)
- \(\sigma_{\text{jitter}}\): Jitter standard deviation = 0.3 ms
- Threshold: \(f_i > 0.5\) required to generate spike

**Example Calculations**:

| Feature Strength | Latency (before jitter) | Spike Time | Notes |
|-----------------|------------------------|------------|-------|
| 1.0 (maximum) | 0 ms | ~0 ms | Earliest possible |
| 0.75 | 25 ms | ~25 ms | Strong feature |
| 0.5 (threshold) | 50 ms | ~50 ms | Just above threshold |
| 0.49 | N/A | No spike | Below threshold |

**Expected Spike Counts** (from testing):
- Per orientation: 30-60 spikes typical
- Neurons that spike: 20-40% of 144 neurons (28-58 neurons)
- Early spikes (<50ms): ~60-70% of total
- Late spikes (>75ms): ~10-20% of total

**Spike Timing Distribution** (example from test output):
```
0° Orientation:
  Total spikes: 43
  Unique neurons: 38/144 (26.4%)
  Spike times:
    Min: 2.15 ms, Max: 87.32 ms, Mean: 31.67 ms, Median: 28.45 ms
  Timing distribution:
    Early (<50ms): 35 spikes (81%)
    Mid (50-75ms): 6 spikes (14%)
    Late (>75ms): 2 spikes (5%)
```

**Feature-to-Spike Correspondence**:
- Gabor response ~2.5-3.0 → spike at ~0-15 ms
- Gabor response ~1.5-2.0 → spike at ~20-40 ms
- Gabor response ~1.0-1.5 → spike at ~40-60 ms
- Gabor response ~0.5-1.0 → spike at ~60-90 ms
- Gabor response < 0.5 → no spike

**Biological Motivation**:
Latency coding observed in:
- LGN responses to visual stimuli (stronger contrast → shorter latency)
- V1 simple cell responses (preferred orientation → earlier spikes)
- Rapid visual processing (first spike timing carries information)
- Compatible with temporal synchrony detection in V1 polychronous groups

---

# 6. V1 Model

## 6.1 Layer and Column Architecture

**Module**: `v1_model.py`  
**Class**: `ComputationalV1Model`  
**Lines**: 1-369

**Module**: `v1_column.py`  
**Class**: `V1OrientationColumn`  
**Lines**: 1-515

### Overall Structure

**4 Orientation Columns**: 0°, 45°, 90°, 135°  
**Total Neurons**: 4,668 (1,167 per column)  
**Total Populations**: 32 (8 per column)

### Per-Column Architecture

Each column contains 8 neuron populations in 4 layers:

**Layer 4 (Input Layer)**:
- **Spiny Stellate (SS)**: 144 neurons
  - Grid: 12×12 retinotopic map
  - Receives LGN input (one-to-one from spike trains)
  - Weight: 5000.0 pA per spike
  - No recurrent connections
- **Inhibitory (Inh)**: 65 neurons
  - Receives from SS (indegree=32, weight=0.0 - disabled)
  - Projects to SS (indegree=6, weight=0.0 - disabled)

**Layer 2/3 (Output Layer)**:
- **Pyramidal (Pyr)**: 144 neurons
  - Grid: 12×12 retinotopic map
  - Primary output of V1
  - Enhanced excitability:
    - v_threshold = -55.0 mV (vs. -50.0 default)
    - tau_m = 25.0 ms (vs. 10.0 default)
    - bias_current = 20.0 pA
  - Receives from L4 SS via polychronous groups
  - Strong recurrent connections (indegree=36, weight=0.0 - disabled)
- **Inhibitory (Inh)**: 65 neurons
  - Receives from Pyr (indegree=35, weight=0.0 - disabled)

**Layer 5 (Deep Layer)**:
- **Pyramidal (Pyr)**: 81 neurons
  - Grid: 9×9 (reduced spatial resolution)
  - Receives from L2/3 Pyr (indegree=15, weight=150.0)
  - Recurrent connections (indegree=10, weight=0.0 - disabled)
- **Inhibitory (Inh)**: 16 neurons

**Layer 6 (Deepest Layer)**:
- **Pyramidal (Pyr)**: 243 neurons
  - Grid: 9×27 (expanded)
  - Receives from L5 Pyr (indegree=20, weight=150.0)
  - Recurrent connections (indegree=20, weight=0.0 - disabled)
- **Inhibitory (Inh)**: 49 neurons

**Neuron Count Summary**:
```
Per Column:
  Layer 4:  144 SS + 65 Inh = 209 neurons
  Layer 2/3: 144 Pyr + 65 Inh = 209 neurons
  Layer 5:   81 Pyr + 16 Inh = 97 neurons
  Layer 6:   243 Pyr + 49 Inh = 292 neurons
  Total per column: 807 neurons

Total Model: 807 × 4 columns = 3,228 neurons
```

**Note**: Documentation states 1,167 neurons per column and 4,668 total, but with current GRID_CONFIG (12×12=144 instead of 18×18=324), actual count is lower. Original MDPI2021 had 324 neurons in L4/L2/3.

### Grid-to-Neuron Index Mapping

**For L4 and L2/3** (12×12 grids):
```
neuron_idx = row * 12 + col
row = neuron_idx // 12
col = neuron_idx % 12
```

Example:
- Grid position (0, 0) → neuron 0
- Grid position (0, 11) → neuron 11
- Grid position (11, 11) → neuron 143

**For L5** (9×9 grid): Similar formula with grid_size=9

**For L6** (9×27 grid):
```
neuron_idx = row * 27 + col
```

## 6.2 Neuron Model Parameters

**Module**: `neurons.py`  
**Class**: `LIFNeuron`  
**Lines**: 12-105

### Leaky Integrate-and-Fire Model

**State Variables**:
- `v_m`: Membrane potential (mV)
- `i_syn_ex`: Excitatory synaptic current (pA)
- `i_syn_in`: Inhibitory synaptic current (pA)
- `refractory_counter`: Time remaining in refractory period (ms)

**Parameters** (from `config.py` → V1_ARCHITECTURE):

**Default Parameters** (L4, L5, L6):
- `v_rest`: -65.0 mV (resting potential)
- `v_threshold`: -50.0 mV (spike threshold)
- `v_reset`: -65.0 mV (reset after spike)
- `tau_m`: 10.0 ms (membrane time constant)
- `tau_syn_ex`: 2.0 ms (excitatory synaptic time constant)
- `tau_syn_in`: 2.0 ms (inhibitory synaptic time constant)
- `refractory_period`: 2.0 ms

**Layer 2/3 Specific Parameters**:
- `L23_v_threshold`: -55.0 mV (lower threshold = more excitable)
- `L23_tau_membrane`: 25.0 ms (longer integration window)
- `L23_bias_current`: 20.0 pA (baseline depolarization)

**Update Equations** (from `neurons.py`, lines 44-82):

**Membrane Potential Update** (Euler integration):
\[
\frac{dV_m}{dt} = \frac{-(V_m - V_{\text{rest}}) + I_{\text{syn,ex}} - I_{\text{syn,in}} + I_{\text{ext}} + I_{\text{bias}}}{\tau_m}
\]

Discrete update:
\[
V_m(t + dt) = V_m(t) + dt \cdot \frac{-(V_m(t) - V_{\text{rest}}) + I_{\text{total}}(t)}{\tau_m}
\]

**Synaptic Current Decay** (exponential):
\[
I_{\text{syn,ex}}(t + dt) = I_{\text{syn,ex}}(t) \cdot e^{-dt / \tau_{\text{syn,ex}}}
\]
\[
I_{\text{syn,in}}(t + dt) = I_{\text{syn,in}}(t) \cdot e^{-dt / \tau_{\text{syn,in}}}
\]

**Spike Generation**:
```
if V_m >= v_threshold and refractory_counter == 0:
    spike = True
    V_m = v_reset
    refractory_counter = refractory_period
    record spike time
```

**Receiving Spikes** (lines 84-94):
```python
def receive_spike(self, weight):
    if weight > 0:
        i_syn_ex += weight  # Excitatory
    else:
        i_syn_in += abs(weight)  # Inhibitory
```

**Initial Conditions**:
- \(V_m(0) = V_{\text{rest}} + \mathcal{N}(0, 4)\) mV (small random init)
- \(I_{\text{syn,ex}}(0) = 0\)
- \(I_{\text{syn,in}}(0) = 0\)

**Time Step**: dt = 0.5 ms (from V1_ARCHITECTURE)

## 6.3 Connectivity (Feedforward, Lateral, Inhibitory)

**Module**: `v1_column.py`  
**Methods**: `_setup_connections()` (lines 152-214), `_setup_feedforward()` (lines 232-258)

### Feedforward Connections

**LGN → Layer 4 SS** (`inject_lgn_spikes`, lines 260-275):
- Connection type: One-to-one (neuron i → neuron i)
- Weight: 5000.0 pA (`lgn_to_ss4_weight`)
- Implemented via direct spike injection at matching time steps

**Layer 4 SS → Layer 2/3 Pyr** (Polychronous Groups):
- Groups of 4 SS cells connect to groups of 4 Pyr cells
- Implementation (lines 248-258):
  ```python
  for i in range(0, n_neurons, 4):  # i = 0, 4, 8, ..., 140
      if i + 3 < n_neurons:
          for ss_idx in range(4):
              for pyr_idx in range(4):
                  connect(layer_4_ss[i + ss_idx], 
                         layer_23_pyr[i + pyr_idx], 
                         weight=120.0)  # weight_L4_to_L23
  ```
- Each L2/3 neuron receives from 4 L4 neurons
- Each L4 neuron projects to 4 L2/3 neurons
- Total connections per group: 4×4 = 16
- Total groups: 144/4 = 36 groups
- Weight: 120.0 pA (`weight_L4_to_L23`)

**Layer 2/3 Pyr → Layer 5 Pyr** (lines 234-237):
- Connection type: Random, fixed indegree
- Indegree: 15 (each L5 neuron receives from 15 L2/3 neurons)
- Weight: 150.0 pA (`weight_L23_to_L5`)
- Total connections: 81 × 15 = 1,215

**Layer 5 Pyr → Layer 6 Pyr** (lines 239-242):
- Connection type: Random, fixed indegree
- Indegree: 20 (each L6 neuron receives from 20 L5 neurons)
- Weight: 150.0 pA (`weight_L5_to_L6`)
- Total connections: 243 × 20 = 4,860

### Lateral (Recurrent) Connections

**Currently DISABLED** (`lateral_weight` = 0.0)

**Original Design** (from MDPI2021):

**Layer 2/3 Pyr ↔ Layer 2/3 Pyr**:
- Indegree: 36
- Weight: 100.0 (when enabled)
- Purpose: Amplify and maintain activity patterns

**Layer 5 Pyr ↔ Layer 5 Pyr**:
- Indegree: 10
- Weight: 100.0 (when enabled)

**Layer 6 Pyr ↔ Layer 6 Pyr**:
- Indegree: 20
- Weight: 100.0 (when enabled)

**Reason for Disabling**: Caused runaway excitation and network instability. Future work: Re-enable with careful weight tuning.

### Inhibitory Connections

**Currently DISABLED** (`inhibitory_weight` = 0.0)

**Original Design** (lines 172-214):

**Layer 4**:
- SS → Inh: indegree=32, weight=100.0
- Inh → SS: indegree=6, weight=-100.0
- Inh ↔ Inh: indegree=6, weight=-100.0

**Layer 2/3**:
- Pyr → Inh: indegree=35, weight=100.0
- Inh → Pyr: indegree=8, weight=-100.0
- Inh ↔ Inh: indegree=8, weight=-100.0

**Layer 5**:
- Pyr → Inh: indegree=30, weight=100.0
- Inh → Pyr: indegree=8, weight=-100.0
- Inh ↔ Inh: indegree=8, weight=-100.0

**Layer 6**:
- Pyr → Inh: indegree=32, weight=100.0
- Inh → Pyr: indegree=6, weight=-100.0
- Inh ↔ Inh: indegree=6, weight=-100.0

**Reason for Disabling**: Without proper balance, inhibition either shut down the network completely or caused oscillations. Future work: Tune inhibitory strengths relative to excitation.

### Connection Summary (Current Active Configuration)

| Source | Target | Type | Indegree | Weight (pA) | Total Connections |
|--------|--------|------|----------|-------------|-------------------|
| LGN | L4 SS | One-to-one | 1 | 5000.0 | 144 |
| L4 SS | L2/3 Pyr | Polychronous | 4 | 120.0 | 576 (36 groups × 16) |
| L2/3 Pyr | L5 Pyr | Random | 15 | 150.0 | 1,215 |
| L5 Pyr | L6 Pyr | Random | 20 | 150.0 | 4,860 |
| **Total per column** | | | | | **6,795** |

**Disabled connections** (weight=0.0):
- All lateral (recurrent) connections
- All inhibitory connections
- All Poisson background activity

## 6.4 Simulation Loop (Pseudocode)

**Main simulation** (`v1_model.py`, `run_stimulus` method, lines 88-160):

```
FUNCTION run_stimulus(spike_trains_by_orientation, warmup=True):
    # Initialize
    current_time = 0.0
    
    # Phase 1: Warmup (spontaneous activity only)
    IF warmup:
        WHILE current_time < warmup_time_ms:  # 50 ms
            FOR each column in [0°, 45°, 90°, 135°]:
                column.update(current_time, lgn_input=None)
            END FOR
            current_time += dt  # dt = 0.5 ms
        END WHILE
    END IF
    
    # Phase 2: Inject spike trains
    stimulus_start_time = current_time
    FOR each orientation in spike_trains_by_orientation:
        column = columns[orientation]
        # Offset spike times to current simulation time
        offset_spikes = spike_data + stimulus_start_time
        column.inject_lgn_spikes(offset_spikes, current_time)
    END FOR
    
    # Phase 3: Stimulus presentation
    stimulus_end_time = current_time + stimulus_time_ms  # 100 ms
    WHILE current_time < stimulus_end_time:
        # Extract spikes that should fire at current time
        current_spikes = get_current_spikes(spike_trains, current_time)
        
        # Update all columns
        FOR each orientation, column:
            lgn_input = current_spikes[orientation] IF exists ELSE None
            column.update(current_time, lgn_input)
        END FOR
        
        current_time += dt
    END WHILE
    
    # Phase 4: Collect results
    analysis_window = (stimulus_start_time, stimulus_end_time)
    FOR each orientation, column:
        FOR each layer in [layer_4, layer_23, layer_5, layer_6]:
            firing_rates = column.get_layer_firing_rates(layer, analysis_window)
            results[orientation][layer] = firing_rates
        END FOR
    END FOR
    
    RETURN results
END FUNCTION
```

**Per-column update** (`v1_column.py`, `update` method, lines 277-342):

```
FUNCTION column.update(current_time, lgn_input):
    # Step 1: Inject LGN spikes into Layer 4 SS
    IF lgn_input is not None:
        FOR each (neuron_id, spike_time) in lgn_input:
            IF spike_time ≈ current_time (within dt/2):
                layer_4_ss.neurons[neuron_id].receive_spike(5000.0)
            END IF
        END FOR
    END IF
    
    # Step 2: Update Layer 4 Spiny Stellate
    spikes_l4_ss = layer_4_ss.update(current_time)
    
    # Step 3: Update Layer 4 Inhibitory
    spikes_l4_inh = layer_4_inh.update(current_time)
    
    # Step 4: Propagate L4 SS spikes via feedforward connections
    propagate_spikes(layer_4_ss, spikes_l4_ss)
        # Sends spikes to Layer 2/3 Pyramidal via polychronous groups
        # Weight = 120.0 pA per spike
    
    # Step 5: Update Layer 2/3 Pyramidal
    spikes_l23_pyr = layer_23_pyr.update(current_time)
    
    # Step 6: Update Layer 2/3 Inhibitory
    spikes_l23_inh = layer_23_inh.update(current_time)
    
    # Step 7: Propagate L2/3 spikes
    propagate_spikes(layer_23_pyr, spikes_l23_pyr)
        # Sends spikes to Layer 5 Pyramidal
        # Weight = 150.0 pA per spike
    
    # Step 8: Update Layer 5 Pyramidal
    spikes_l5_pyr = layer_5_pyr.update(current_time)
    
    # Step 9: Update Layer 5 Inhibitory
    spikes_l5_inh = layer_5_inh.update(current_time)
    
    # Step 10: Propagate L5 spikes
    propagate_spikes(layer_5_pyr, spikes_l5_pyr)
        # Sends spikes to Layer 6 Pyramidal
        # Weight = 150.0 pA per spike
    
    # Step 11: Update Layer 6 Pyramidal
    spikes_l6_pyr = layer_6_pyr.update(current_time)
    
    # Step 12: Update Layer 6 Inhibitory
    spikes_l6_inh = layer_6_inh.update(current_time)
    
    RETURN spike_counts_per_layer
END FUNCTION
```

**Population update** (`neurons.py`, `NeuronPopulation.update`, lines 209-248):

```
FUNCTION population.update(current_time, external_inputs=None):
    current_spikes = []
    
    FOR each neuron i in population:
        # Get external input for this neuron
        ext_current = external_inputs[i] IF exists ELSE 0.0
        
        # Get Poisson background (disabled, = 0.0)
        poisson_current = 0.0
        
        # Add bias current (for L2/3 only)
        total_current = ext_current + poisson_current + bias_current
        
        # Update neuron
        spiked = neuron[i].update(current_time, total_current)
        
        IF spiked:
            current_spikes.append(i)
        END IF
    END FOR
    
    # Propagate recurrent spikes within population (disabled, weight=0)
    FOR each spike_idx in current_spikes:
        FOR each (target_idx, weight) in recurrent_connections[spike_idx]:
            neurons[target_idx].receive_spike(weight)  # weight = 0.0
        END FOR
    END FOR
    
    RETURN current_spikes
END FUNCTION
```

**Single neuron update** (`neurons.py`, `LIFNeuron.update`, lines 44-82):

```
FUNCTION neuron.update(current_time, external_current):
    IF refractory_counter > 0:
        refractory_counter -= dt
        v_m = v_rest
        RETURN False
    ELSE:
        # Membrane potential update (Euler)
        dv = (-(v_m - v_rest) + i_syn_ex - i_syn_in + external_current) / tau_m
        v_m += dv * dt
        
        # Check threshold
        IF v_m >= v_threshold:
            v_m = v_reset
            refractory_counter = refractory_period
            spike_times.append(current_time)
            RETURN True
        END IF
    END IF
    
    # Decay synaptic currents
    i_syn_ex *= exp(-dt / tau_syn_ex)
    i_syn_in *= exp(-dt / tau_syn_in)
    
    RETURN False
END FUNCTION
```

**Simulation Timeline** (per frame):
```
Time 0-50ms:    Warmup (spontaneous activity only, first frame only)
Time 50-150ms:  Stimulus presentation (LGN spikes injected)
                Analysis window = [50ms, 150ms]
                
Total steps: (50 + 100) / 0.5 = 300 time steps
Per-step operations:
  - 4,668 neuron updates (all 4 columns)
  - Synaptic current decay (2 operations per neuron)
  - Spike propagation through feedforward connections
  - Firing rate calculation at end

Total operations: ~300 steps × 4,668 neurons × ~10 ops/neuron ≈ 14 million operations/frame
```

---

# 7. Decoding and Visualization

## 7.1 Orientation Map Construction

**Module**: `v1_decoder.py`  
**Class**: `V1Decoder`  
**Lines**: 1-306

**Main Method**: `decode_v1_output(self, v1_results, layer='layer_23')` (lines 27-62)

**Purpose**: Convert V1 firing rates back into spatial orientation maps showing detected edges and their orientations.

### Process

**Step 1**: Extract firing rates from target layer (default: Layer 2/3)
```python
for orientation in [0, 45, 90, 135]:
    rates = v1_results['orientations'][orientation][layer]['firing_rates']
    # rates shape: (144,) for L2/3
    response_grid = rates.reshape(grid_rows, grid_cols)  # (12, 12)
    orientation_responses[orientation] = response_grid
```

**Step 2**: Create orientation preference map (winner-take-all)
```python
def _create_orientation_map(self, orientation_responses):
    orientation_map = np.zeros((12, 12))
    
    for row in range(12):
        for col in range(12):
            # Get response for each orientation at this position
            responses = {
                0: orientation_responses[0][row, col],
                45: orientation_responses[45][row, col],
                90: orientation_responses[90][row, col],
                135: orientation_responses[135][row, col]
            }
            
            # Winner-take-all: find orientation with max response
            if max(responses.values()) > 0:
                pref_orientation = max(responses, key=responses.get)
                orientation_map[row, col] = pref_orientation
            else:
                orientation_map[row, col] = -1  # No response
    
    return orientation_map
```

**Step 3**: Create response strength map
```python
def _create_strength_map(self, orientation_responses):
    strength_map = np.zeros((12, 12))
    
    for row in range(12):
        for col in range(12):
            # Max response across all orientations
            max_response = max(
                orientation_responses[0][row, col],
                orientation_responses[45][row, col],
                orientation_responses[90][row, col],
                orientation_responses[135][row, col]
            )
            strength_map[row, col] = max_response
    
    return strength_map
```

**Output**:
- `orientation_map`: (12, 12) array with values {-1, 0, 45, 90, 135}
  - -1 = no response (all orientations below threshold)
  - 0, 45, 90, 135 = preferred orientation
- `strength_map`: (12, 12) array with maximum firing rate (Hz) per location

### Winner-Take-All Equation

For each spatial position \((i, j)\):
\[
\text{Orientation}_{ij} = \arg\max_{\theta \in \{0°, 45°, 90°, 135°\}} R_{ij}^{\theta}
\]
\[
\text{Strength}_{ij} = \max_{\theta} R_{ij}^{\theta}
\]

where \(R_{ij}^{\theta}\) is the firing rate (Hz) of Layer 2/3 at grid position \((i, j)\) in orientation column \(\theta\).

## 7.2 Layer Activity Maps

**Method**: `visualize_layer_activity(self, v1_results)` (lines 219-276)

**Purpose**: Create heatmaps showing firing rates across all V1 layers.

**Process**:
1. For each layer (L4, L2/3, L5, L6):
   - Average firing rates across all 4 orientation columns
   - Reshape to appropriate grid (12×12, 12×12, 9×9, 9×27)
   - Normalize to [0, 1]
   - Apply hot colormap
   - Upscale to 180×180 for visibility
   - Add label with mean firing rate

2. Arrange in 2×2 grid:
   ```
   [Layer 4]  [Layer 2/3]
   [Layer 5]  [Layer 6]
   ```

**Grid Shapes**:
- Layer 4: 144 neurons → 12×12 grid
- Layer 2/3: 144 neurons → 12×12 grid
- Layer 5: 81 neurons → 9×9 grid
- Layer 6: 243 neurons → 9×27 grid

## 7.3 Color Maps / Thresholds

### Orientation Color Coding

**From `config.py` → VISUALIZATION_CONFIG:**
```python
'orientation_colors': {
    0: (255, 0, 0),      # Red (horizontal edges)
    45: (0, 255, 0),     # Green (diagonal /)
    90: (0, 0, 255),     # Blue (vertical edges)
    135: (255, 255, 0),  # Yellow (diagonal \)
}
```

### Color-Coded Orientation Map

**Method**: `_visualize_orientation_map(self, orientation_map, strength_map)` (lines 115-164)

```python
# For each grid position
for row in range(12):
    for col in range(12):
        orientation = orientation_map[row, col]
        strength = strength_norm[row, col]  # Normalized [0, 1]
        
        if orientation >= 0:
            color = orientation_colors[orientation]
            # Modulate color by response strength
            vis[row, col] = [
                int(color[0] * strength),
                int(color[1] * strength),
                int(color[2] * strength)
            ]
```

**Output**: 12×12 color image, upscaled to 360×360 with grid lines and legend

### Oriented Edge Visualization

**Method**: `_visualize_as_edges(self, orientation_map, strength_map)` (lines 166-217)

**Purpose**: Draw oriented line segments showing detected edges.

```python
cell_size = 360 // 12 = 30 pixels
line_length = cell_size - 5 = 25 pixels

for row in range(12):
    for col in range(12):
        orientation = orientation_map[row, col]
        strength = strength_norm[row, col]
        
        if orientation >= 0 and strength > 0.1:  # Threshold
            # Center of cell
            center_x = col * 30 + 15
            center_y = row * 30 + 15
            
            # Calculate line endpoints
            angle_rad = deg2rad(orientation)
            dx = int(cos(angle_rad) * line_length / 2)
            dy = int(sin(angle_rad) * line_length / 2)
            
            pt1 = (center_x - dx, center_y - dy)
            pt2 = (center_x + dx, center_y + dy)
            
            # Draw line with color and thickness
            color = orientation_colors[orientation]
            thickness = max(1, int(strength * 3))
            draw_line(pt1, pt2, color, thickness)
```

**Thresholds**:
- Minimum strength to display: 0.1 (normalized)
- Line thickness: 1-3 pixels based on strength
- Actual firing rate threshold: ~1 Hz (very low)

### Layer Activity Heatmap

**Colormap**: `cv2.COLORMAP_HOT`
- Black: 0 Hz (no activity)
- Red: Low-medium activity
- Yellow: Medium-high activity
- White: Maximum activity

**Normalization**: Per-layer normalization to show relative activity within each layer.

### Combined Visualization Panel

**Method**: `create_combined_panel(self, visualizations)` (lines 284-369, `pipeline.py`)

**Layout**: 3×2 grid (1920×1080 total):
```
Row 1: [Raw Video (640×540)] [Comparison (640×540)] [V1 Color Map (640×540)]
Row 2: [Gabor (640×540)]     [Spikes (640×540)]     [Layers (640×540)]
```

Each panel upscaled to 640×540 and labeled.

**Timing Overlay**: Green text in top-left corner showing:
- Frame number
- Per-stage timing (ms)
- Total time and FPS

**Update Strategy**: Visualizations cached and updated every 5 seconds to reduce computational overhead (from `VISUALIZATION_CONFIG['update_interval_seconds']`).

---

# 8. Experiments and Performance

## 8.1 Timing and FPS

### Per-Stage Timing (Typical)

**From test runs and debug output** (`pipeline.py`, timing tracking):

**Static Image Processing** (single frame with warmup):
```
Preprocess:    1-2 ms
Gabor:         5-10 ms
Encode:        1-2 ms
V1 Sim:        100-200 ms (50ms warmup + 100ms stimulus @ dt=0.5ms = 300 steps)
Decode:        2-5 ms
Visualization: 10-20 ms (when updating, cached otherwise)
────────────────────────────────
TOTAL:         ~120-240 ms per frame
FPS:           ~4-8 FPS (if no warmup on subsequent frames)
```

**Real-Time Video Processing** (continuous):
```
First 3 frames: Include warmup (~200ms per frame)
Subsequent frames: No warmup (~120-150ms per frame)

Average FPS: ~5-7 FPS
```

### Detailed Breakdown

**Preprocessing** (1-2 ms):
- Gaussian blur (3×3): <1 ms
- Contrast normalization: <1 ms
- Negligible overhead

**Gabor Feature Extraction** (5-10 ms):
- 4 Gabor filters × cv2.filter2D: ~2-4 ms
- Retinotopic grid creation (12×12 × 4 orientations): ~2-3 ms
- Sparsification pipeline: ~1-2 ms
- Orientation competition: ~1 ms

**Spike Encoding** (1-2 ms):
- Iterate 144 neurons × 4 orientations = 576 neurons
- Latency calculation: O(1) per neuron
- Generate 30-60 spikes per orientation: ~150-240 total

**V1 Simulation** (100-200 ms):
**Breakdown per time step (0.5 ms)**:
- 4,668 neuron updates: ~0.2-0.3 ms
- Synaptic current decay (2 exp per neuron): ~0.1 ms
- Spike propagation: <0.1 ms (sparse)
- Total per step: ~0.3-0.5 ms

**Total steps**: 300 (50ms warmup + 100ms stimulus)  
**Total simulation time**: 300 × 0.3-0.5 ms = **90-150 ms**

**With Python overhead and diagnostics**: **100-200 ms**

**Decoding** (2-5 ms):
- Reshape firing rates: <1 ms
- Winner-take-all (12×12 × 4 comparisons): ~1 ms
- Create visualizations: ~1-3 ms

**Visualization Update** (10-20 ms, cached every 5 seconds):
- Gabor visualization (4 orientations): ~3-5 ms
- Spike raster plots: ~2-3 ms
- Layer heatmaps: ~2-3 ms
- Orientation maps: ~2-3 ms
- Combined panel assembly: ~3-5 ms

### FPS Calculation Examples

**With warmup every frame** (not typical):
```
Time per frame: 200 ms
FPS = 1000 / 200 = 5 FPS
```

**Without warmup** (continuous video):
```
Time per frame: 120 ms
FPS = 1000 / 120 = 8.3 FPS
```

**Actual observed** (with visualization and debug overhead):
```
Average time: ~150-180 ms
FPS: ~5-7 FPS
```

**Comparison to MDPI2021**:
- MDPI2021 (NEST): ~50 seconds per frame = 0.02 FPS
- This implementation: ~0.15 seconds per frame = 6.7 FPS
- **Speedup: ~330×**

## 8.2 Experimental Variations

**NO DATA FOUND**: Formal experimental comparison of different parameter settings with quantitative results.

**Documented Design Iterations** (from fix summaries):

### Iteration 1: Grid Size Optimization
- **Before**: 18×18 grid (324 neurons per layer)
- **After**: 12×12 grid (144 neurons per layer)
- **Reason**: 2.25× speedup in L4 and L2/3 updates
- **Impact**: 56% reduction in neuron count, still sufficient resolution

### Iteration 2: Time Step Optimization
- **Before**: dt = 0.1 ms (matching MDPI2021)
- **After**: dt = 0.5 ms
- **Reason**: 5× fewer simulation steps
- **Impact**: Biological accuracy preserved, significant speedup

### Iteration 3: Warmup Reduction
- **Before**: 400 ms warmup per frame
- **After**: 50 ms warmup (first frame only)
- **Reason**: Continuous video doesn't need repeated warmup
- **Impact**: 8× reduction in warmup time

### Iteration 4: Stimulus Duration Optimization
- **Before**: 200 ms stimulus duration
- **After**: 100 ms stimulus duration
- **Reason**: Sufficient for spike propagation through all layers
- **Impact**: 2× faster per frame

### Iteration 5: Weight Tuning
**Problem**: Layer 2/3 was silent (~0 Hz), Layer 6 had runaway activity (~110 Hz)

**Solution** (documented in `V1_FIX_COMPLETE_SUMMARY.md`):
- **L4→L2/3**: 20 → 120 (6× increase)
- **L2/3→L5**: 800 → 150 (5.3× decrease)
- **L5→L6**: 1200 → 150 (8× decrease)
- **L2/3 threshold**: -50 mV → -55 mV (more excitable)
- **L2/3 tau_m**: 10 ms → 25 ms (better integration)
- **L2/3 bias**: 0 pA → 20 pA (baseline activity)

**Results**:
```
Before Fix:
L4:  40-50 Hz ✓
L2/3: 0 Hz    ✗ (silent)
L5:   0 Hz    ✗
L6:  110 Hz   ✗ (runaway)

After Fix:
L4:  40-50 Hz ✓
L2/3: 10-25 Hz ✓ (now firing!)
L5:   5-20 Hz  ✓
L6:   2-15 Hz  ✓ (stable)
```

### Iteration 6: Gabor Sparsification
**Problem**: 97-99% of cells active per orientation (poor selectivity)

**Solution** (documented in `GABOR_SPARSIFICATION_SUMMARY.md`):
- Z-score normalization
- Clip negative values
- Percentile thresholding (top 20%)
- Orientation competition via softmax

**Results**:
```
Before: 97-99% active cells
After: 10-30% active cells
Effect: Clear orientation selectivity, 30-60 spikes per orientation (was sparse before)
```

### Iteration 7: Disabling Unstable Features
**Disabled**:
- Lateral (recurrent) connections: weight 100.0 → 0.0
- Inhibitory connections: weight -100.0 → 0.0
- Poisson background activity: 1.7 MHz → 0.0 Hz

**Reason**: Caused runaway excitation and network instability

**Future Work**: Re-enable with careful tuning

## 8.3 Observed Behavior and Failure Modes

### Successful Behavior

**From test_static_image.py runs**:

**Gabor Features**:
- Clear orientation selectivity (10-30% active per orientation)
- Horizontal edges dominate 0° response
- Vertical edges dominate 90° response
- Diagonal edges split between 45° and 135°

**Spike Encoding**:
- 30-60 spikes per orientation (typical)
- 20-40% of neurons spike
- Early spike timing (60-80% < 50ms)
- Jitter adds biological variability

**V1 Activity**:
- Layer 4: 40-50 Hz (healthy input layer activity)
- Layer 2/3: 10-25 Hz (primary output, orientation-selective)
- Layer 5: 5-20 Hz (moderate deep layer activity)
- Layer 6: 2-15 Hz (stable, no runaway)

**Orientation Maps**:
- Clear colored regions corresponding to input features
- Winner-take-all produces sharp orientation preference
- Response strength correlates with Gabor magnitude
- <50% "no response" pixels (was 97% before fixes)

### Known Failure Modes

**1. All Layers Silent** (resolved)
- **Cause**: Weights too weak (L4→L2/3 = 20)
- **Symptom**: L2/3, L5, L6 all show 0 Hz
- **Solution**: Increase feedforward weights (now 120)

**2. Layer 6 Runaway** (resolved)
- **Cause**: L5→L6 weight too strong (1200)
- **Symptom**: L6 firing at 50-110 Hz continuously
- **Solution**: Reduce weight to 150

**3. Oscillations with Inhibition** (known issue)
- **Cause**: Inhibitory weights not balanced with excitation
- **Symptom**: Network oscillates or shuts down completely
- **Current State**: Inhibition disabled (weight=0)
- **Future**: Requires careful tuning of excitation/inhibition ratio

**4. Runaway with Recurrent Connections** (known issue)
- **Cause**: Positive feedback loops
- **Symptom**: Firing rates increase indefinitely
- **Current State**: Lateral connections disabled (weight=0)
- **Future**: Needs weight decay or normalization

**5. Sparse Spikes from Dense Gabor Features** (resolved)
- **Cause**: No sparsification, 97-99% active cells
- **Symptom**: Spike trains dominated by threshold noise
- **Solution**: Sparsification pipeline + orientation competition

**6. Orientation Maps All Gray** (resolved)
- **Cause**: L2/3 silent, decoder has no signal
- **Symptom**: orientation_map shows -1 (no response) everywhere
- **Solution**: Fix L2/3 activation via weight tuning

### Edge Cases

**Low Contrast Images**:
- Few Gabor responses above threshold
- <20 spikes per orientation
- Sparse V1 activity (L2/3 ~5 Hz)
- Orientation map shows many "no response" pixels

**High Contrast Edges**:
- Strong Gabor responses (2.5-3.0)
- 50-70 spikes per orientation
- High V1 activity (L2/3 ~30-40 Hz)
- Clear orientation maps

**Uniform Images** (no edges):
- All Gabor responses near zero
- No spikes generated (all below threshold)
- V1 completely silent
- Orientation map: all "no response"

**Complex Natural Scenes**:
- Mixed orientation responses
- Multiple orientations per location
- Winner-take-all sometimes arbitrary
- Noisy orientation map (many small regions)

### Numerical Stability

**No Issues Found**:
- No NaN values observed in firing rates or synaptic currents
- No infinite values
- Membrane potentials stay within expected range [-65, -50] mV
- Synaptic currents bounded by input weights and decay

**Debug checks** (from DEBUG_CONFIG):
- `check_for_nans`: Monitors all arrays
- `check_for_zeros`: Warns if entire layers silent
- Comprehensive diagnostics at frame 10

---

# 9. Design Choices and Limitations

## 9.1 Deviations from the Reference MDPI V1 Model

**From `MDPI2021_VS_COMPUTATIONAL_COMPARISON.md`**:

### Major Differences

**1. Simulation Platform**
- **MDPI2021**: NEST Simulator (C++ backend, requires compilation)
- **Computational**: Pure Python (NumPy + OpenCV)
- **Reason**: Portability, ease of modification, no compilation required

**2. Neuron Models**
- **MDPI2021**: 
  - `lifl_psc_exp_ie` (Leaky IF with Intrinsic Excitability plasticity)
  - `aeif_psc_exp_peak` (Adaptive Exponential IF)
- **Computational**: Simplified LIF (no adaptation, no IE plasticity)
- **Reason**: IE plasticity requires training phase (hours-days), adaptive dynamics computationally expensive
- **Impact**: Less biological realism, but sufficient for orientation selectivity

**3. Intrinsic Excitability (IE) Plasticity**
- **MDPI2021**: Neurons learn intrinsic excitability via STDP, pre-trained soma_exc values loaded from pickle files
- **Computational**: Fixed weights, no learning
- **Reason**: Training phase impractical for real-time system
- **Impact**: No online adaptation, but orientation selectivity achieved via input routing

**4. Spatial Resolution**
- **MDPI2021**: 18×18 grid = 324 neurons per primary layer
- **Computational**: 12×12 grid = 144 neurons per primary layer (56% reduction)
- **Reason**: 2.25× speedup, sufficient for 320×240 input
- **Impact**: Lower spatial resolution but maintains retinotopic principle

**5. Temporal Resolution**
- **MDPI2021**: dt = 0.1 ms (required by NEST)
- **Computational**: dt = 0.5 ms (5× fewer steps)
- **Reason**: Biological dynamics preserved at 0.5ms, significant speedup
- **Impact**: Slight loss of temporal precision, acceptable for visual processing

**6. Simulation Duration**
- **MDPI2021**: 400ms warmup + 200ms stimulus = 600ms, 6000 steps
- **Computational**: 50ms warmup (first frame only) + 100ms stimulus = 150ms, 300 steps
- **Reason**: Real-time requires <1 second per frame
- **Impact**: 4× faster, sufficient time for spike propagation through all layers

**7. Background Activity**
- **MDPI2021**: Poisson generators at 1.7 MHz per layer (simulates synaptic bombardment)
- **Computational**: Disabled (rate = 0 Hz)
- **Reason**: Computationally expensive, caused constant high firing rates
- **Impact**: Less spontaneous activity, but cleaner stimulus responses

**8. Lateral Connections**
- **MDPI2021**: Recurrent connections within layers (indegree=36 for L2/3, weight=100)
- **Computational**: Disabled (weight = 0)
- **Reason**: Caused runaway excitation and instability
- **Impact**: No within-layer amplification, purely feedforward processing
- **Future**: Re-enable with careful tuning

**9. Inhibitory Connections**
- **MDPI2021**: Full inhibitory populations with feedforward and feedback (weight=-100)
- **Computational**: Disabled (weight = 0)
- **Reason**: Without proper balance, caused oscillations or complete shutdown
- **Impact**: No lateral inhibition, reduced selectivity sharpening
- **Future**: Tune excitation/inhibition ratio carefully

**10. Synaptic Delays**
- **MDPI2021**: Explicit delays (1ms for most connections, 0.1ms for STDP)
- **Computational**: No delays (spikes propagate within same timestep)
- **Reason**: Delay tracking adds memory and computation overhead
- **Impact**: Simplified temporal dynamics, less polychrony fidelity

**11. Polychrony Detection**
- **MDPI2021**: Learned via IE plasticity and neuromodulation between SS4 cells
- **Computational**: Fixed connectivity groups (4 SS → 4 Pyr)
- **Reason**: No training phase
- **Impact**: Less adaptive, but functional for feedforward processing

**12. Weights**
- **MDPI2021**: LGN→L4 = 15000, feedforward = 100, lateral = 100, inhibitory = -100
- **Computational**: LGN→L4 = 5000, L4→L2/3 = 120, L2/3→L5 = 150, L5→L6 = 150, lateral = 0, inhibitory = 0
- **Reason**: Tuned for stability in simplified feedforward-only architecture
- **Impact**: Different activation dynamics, but achieves orientation selectivity

### Preserved Biological Features

**Architecture**: ✓
- 4 orientation columns (0°, 45°, 90°, 135°)
- Laminar structure (L4, L2/3, L5, L6)
- Excitatory + Inhibitory populations
- Retinotopic organization

**Neuron Counts**: ✓ (scaled)
- Proportions maintained (L4/L2/3 equal, L5 smaller, L6 larger)
- Total per column: 807 neurons (vs. 1167 in MDPI)

**Connectivity Patterns**: ✓
- LGN → L4 (one-to-one)
- L4 → L2/3 (polychronous groups)
- L2/3 → L5 → L6 (feedforward hierarchy)
- Fixed indegree connections

**Functional Properties**: ✓
- Orientation selectivity
- Retinotopic mapping
- Spike timing encoding
- Hierarchical processing

## 9.2 Tradeoffs for Real-Time Performance

### Speed vs. Biological Realism

**Achieved Speedup: ~330× faster than MDPI2021**
- MDPI: ~50 seconds per frame
- Computational: ~0.15 seconds per frame

**Cost in Biological Realism**:
1. **No learning**: Fixed weights vs. STDP and IE plasticity
2. **Simplified dynamics**: LIF vs. AdEx and IE neurons
3. **No background activity**: Reduces spontaneous variability
4. **No inhibition**: Less winner-take-all sharpening
5. **No lateral connections**: Reduced contextual processing
6. **Coarser temporal resolution**: 0.5ms vs. 0.1ms
7. **Shorter simulation**: 150ms vs. 600ms
8. **Lower spatial resolution**: 12×12 vs. 18×18

### Acceptable Losses

**For edge detection and orientation selectivity**:
- Core functionality preserved
- Orientation maps match input features
- Spatial organization maintained
- Real-time capability achieved

**For neuroscience research**:
- Less suitable for detailed biophysical studies
- Cannot compare with MEG/EEG data
- Simplified temporal dynamics
- No learning experiments

### Critical Design Decisions

**1. Feedforward-Only Architecture**
- **Decision**: Disable lateral and inhibitory connections
- **Rationale**: Instability in computational implementation
- **Tradeoff**: Stability vs. lateral processing
- **Future**: Re-enable incrementally with tuning

**2. Grid Size Reduction**
- **Decision**: 12×12 instead of 18×18
- **Rationale**: 2.25× speedup in most active layers
- **Tradeoff**: Spatial resolution vs. speed
- **Acceptable**: 144 neurons sufficient for 320×240 input

**3. Time Step Increase**
- **Decision**: 0.5ms instead of 0.1ms
- **Rationale**: 5× fewer simulation steps
- **Tradeoff**: Temporal precision vs. speed
- **Acceptable**: 0.5ms preserves spike timing information

**4. No Background Activity**
- **Decision**: Disable Poisson generators
- **Rationale**: Computationally expensive, caused constant firing
- **Tradeoff**: Spontaneous variability vs. clean responses
- **Acceptable**: Stimulus-driven responses clearer without noise

**5. Weight Tuning for Stability**
- **Decision**: Manually tune feedforward weights (120, 150, 150)
- **Rationale**: No automatic adaptation, need stable propagation
- **Tradeoff**: Fixed weights vs. biological flexibility
- **Acceptable**: Achieves desired firing rates and selectivity

## 9.3 Future Improvements

### Short-Term (Feasible Now)

**1. Re-Enable Inhibition**
- Gradually increase inhibitory weights from 0
- Balance with excitation (e.g., E/I ratio = 4:1)
- Monitor for oscillations
- **Expected Benefit**: Sharper orientation tuning, winner-take-all

**2. Re-Enable Lateral Connections**
- Start with low weights (e.g., 20-50 instead of 100)
- Add weight decay or normalization
- Monitor for runaway activity
- **Expected Benefit**: Contextual processing, contour integration

**3. Add Low-Level Background Activity**
- Reduce Poisson rates (e.g., 0.1 MHz instead of 1.7 MHz)
- Or fixed small bias current (~5 pA)
- **Expected Benefit**: More realistic spontaneous activity

**4. Increase Spatial Resolution**
- Return to 18×18 grid (324 neurons)
- Optimize code to maintain real-time performance
- **Expected Benefit**: Better spatial detail

**5. Tune Spike Threshold**
- Current: 0.5 (conservative)
- Lower to 0.3-0.4 for more spikes
- **Expected Benefit**: Richer spike patterns, better utilization of L4

### Medium-Term (Requires Development)

**6. Implement Lightweight Adaptation**
- Spike-frequency adaptation (simple exponential)
- Not full AdEx, just basic fatigue
- **Expected Benefit**: More realistic firing patterns

**7. Add Simple STDP**
- Short-term potentiation/depression
- Not full IE plasticity, just weight modification
- **Expected Benefit**: Online learning capability

**8. GPU Acceleration**
- Port neuron updates to CUDA/OpenCL
- Parallel processing of all neurons
- **Expected Benefit**: 10-100× speedup, enable full MDPI2021 fidelity

**9. Between-Column Connections**
- Lateral connections across orientation columns
- Cross-orientation suppression
- **Expected Benefit**: Better orientation selectivity, iso-orientation facilitation

**10. Temporal Integration Across Frames**
- Maintain membrane states between frames
- Short-term memory (100-200ms)
- **Expected Benefit**: Motion detection, temporal coherence

### Long-Term (Research Projects)

**11. Full IE Plasticity**
- Implement learning of intrinsic excitability
- Training phase for soma_exc
- **Expected Benefit**: True polychrony detection, adaptive responses

**12. Detailed Neuron Models**
- AdEx for pyramidal cells
- Hodgkin-Huxley for detailed biophysics
- **Expected Benefit**: Match MEG/EEG data, neuroscience validation

**13. Dendritic Computation**
- Multi-compartment models
- Nonlinear integration in dendrites
- **Expected Benefit**: More complex receptive fields

**14. Higher Visual Areas**
- V2, V4, MT/MST
- Hierarchical feature extraction
- **Expected Benefit**: Object recognition, motion processing

**15. Feedback from Higher Areas**
- Top-down attention
- Predictive coding
- **Expected Benefit**: Context-dependent processing

### Optimization Opportunities

**Code-Level**:
- Vectorize neuron updates (currently loop-based)
- Use NumPy broadcasting more extensively
- JIT compilation with Numba
- C++ extension for inner loops

**Algorithmic**:
- Sparse spike propagation (only update neurons that receive spikes)
- Event-driven simulation (only update when spikes occur)
- Adaptive time stepping (larger dt when quiescent)

**Architectural**:
- Reduce Layer 5/6 neuron counts (currently 81/243)
- Focus computation on L4 and L2/3 (primary processing)
- Skip layers for real-time mode

### Known Limitations

**Cannot Currently Do**:
- Online learning / adaptation
- Cross-orientation suppression (no lateral connections)
- Precise temporal coding (0.5ms time step)
- Match MEG data quantitatively (simplified model)
- Process >10 FPS real-time (limited by V1 simulation)

**Fundamental Constraints**:
- Python overhead (vs. C++/GPU)
- Single-threaded execution (could parallelize columns)
- No specialized hardware acceleration
- Trade-off between realism and speed is inherent

### Success Criteria for Improvements

**Must Maintain**:
- Real-time capability (>5 FPS)
- Stable firing rates (<50 Hz average)
- Clear orientation selectivity
- Spatial organization

**Should Improve**:
- Biological realism (closer to MDPI2021)
- Orientation tuning sharpness
- Temporal dynamics fidelity
- Robustness to parameter changes

**Would Be Nice**:
- Match MEG data
- Online learning
- Higher spatial/temporal resolution
- Faster than real-time (>30 FPS)

---

# 10. Summary and Conclusions

## Key Achievements

This repository implements a **real-time computational V1 visual cortex** that:

1. **Maintains biological architecture**: 4 orientation columns, laminar structure (L4, L2/3, L5, L6), 3,228 spiking neurons
2. **Achieves real-time performance**: ~5-7 FPS vs. 0.02 FPS for MDPI2021 (**330× speedup**)
3. **Preserves core functionality**: Orientation selectivity, retinotopic organization, spike-based processing
4. **Runs anywhere**: Pure Python, no compilation, works on any platform
5. **Fully documented**: Comprehensive code comments, architecture docs, testing guides, fix summaries

## Technical Highlights

**Pipeline**:
- **Gabor filters** (4 orientations, 12×12 grid, 20px RF) with sparsification (10-30% active cells)
- **Latency encoding** (0-100ms, threshold=0.5, 30-60 spikes/orientation)
- **LIF neurons** (v_thresh=-50mV, tau_m=10ms, dt=0.5ms, refractory=2ms)
- **Feedforward architecture** (LGN→L4→L2/3→L5→L6, weights: 5000/120/150/150)
- **Winner-take-all decoder** (orientation maps from L2/3 firing rates)

**Performance**:
- **Gabor**: 5-10 ms
- **Encoding**: 1-2 ms
- **V1 Simulation**: 100-200 ms (300 time steps × 3228 neurons)
- **Decoding**: 2-5 ms
- **Total**: ~120-240 ms per frame (4-8 FPS)

**Parameters Tuned for Stability**:
- L4→L2/3: 120 pA (was too weak at 20)
- L2/3→L5: 150 pA (was too strong at 800)
- L5→L6: 150 pA (was causing runaway at 1200)
- L2/3 threshold: -55 mV (more excitable)
- L2/3 tau_m: 25 ms (better integration)

## For Your DLCV Report

### Recommended Structure

**1. Introduction**
- Motivation: Neuralink Blindsight, neural prosthetics
- Goal: Real-time synthetic V1 for video processing
- Based on MDPI2021 V1 orientation column model

**2. Background**
- V1 architecture: layers, orientation columns, retinotopic maps
- Gabor filters: orientation-selective, V1 simple cell receptive fields
- Spike coding: latency encoding, temporal information
- LIF neurons: integrate-and-fire dynamics

**3. Methods**
- **Preprocessing**: Gaussian blur, contrast normalization
- **Feature Extraction**: 4 Gabor orientations, 12×12 grid, sparsification pipeline
- **Spike Encoding**: Latency coding (strong→early), threshold=0.5, 0-100ms range
- **V1 Model**: 4 columns × (L4, L2/3, L5, L6), 3228 LIF neurons, feedforward connectivity
- **Decoding**: Winner-take-all, orientation maps from L2/3 firing rates

**4. Implementation**
- Pure Python (NumPy + OpenCV)
- Computational simplifications vs. NEST simulator
- Weight tuning for stability
- Real-time optimizations (dt=0.5ms, 12×12 grid, no warmup after first frame)

**5. Results**
- Successfully processes video at 5-7 FPS
- Clear orientation selectivity (10-30% active cells per orientation)
- Stable firing rates (L4: 40-50 Hz, L2/3: 10-25 Hz, L5/L6: 2-20 Hz)
- Orientation maps correlate with input edges
- 330× faster than MDPI2021 reference

**6. Discussion**
- **Tradeoffs**: Speed vs. biological realism
- **Simplifications**: No learning, no inhibition, no lateral connections, simplified neurons
- **Preserved**: Architecture, orientation selectivity, retinotopic organization
- **Future**: Re-enable inhibition/laterals, GPU acceleration, higher resolution

**7. Conclusions**
- Demonstrated feasibility of real-time spiking V1
- Biological principles + computational pragmatism
- Platform for neural prosthetics research

### Equations to Include

**Gabor Filter**:
\[
G(x, y; \theta) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi\frac{x'}{\lambda}\right)
\]

**Latency Encoding**:
\[
t_{\text{spike}} = L_{\max} - f \cdot (L_{\max} - L_{\min})
\]

**LIF Dynamics**:
\[
\tau_m \frac{dV}{dt} = -(V - V_{\text{rest}}) + I_{\text{syn}}
\]

**Winner-Take-All Decoding**:
\[
\text{Orientation}_{ij} = \arg\max_{\theta} R_{ij}^{\theta}
\]

### Figures to Create

1. **System architecture diagram**: Camera → Gabor → Spikes → V1 → Decoder
2. **V1 column structure**: 4 layers, feedforward connectivity
3. **Gabor filters**: 4 orientations, example responses
4. **Sparsification effect**: Before/after histograms
5. **Latency encoding**: Feature strength vs. spike time
6. **V1 activity**: Firing rates by layer and orientation
7. **Orientation maps**: Input image vs. V1 output
8. **Timing breakdown**: Bar chart of per-stage ms
9. **Weight tuning results**: Before/after firing rates
10. **Comparison table**: MDPI2021 vs. Computational

### Key Points for Presentation

- **"Inspired by Neuralink Blindsight"** - neural prosthetics motivation
- **"Based on published V1 model"** - MDPI2021 reference, biological foundation
- **"Real-time performance"** - 5-7 FPS, 330× faster than reference
- **"Maintains core architecture"** - 4 columns, 4 layers, 3228 neurons
- **"Computational tradeoffs"** - Speed vs. realism, explicit design choices
- **"Successful orientation selectivity"** - Clear detection of oriented edges

---

**END OF REPORT MATERIALS**

This document contains all information extracted from the repository for writing a comprehensive DLCV technical report. All parameters, equations, timing data, and architectural details have been collected from the actual codebase and documentation.


