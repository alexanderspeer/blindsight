# V1 Vision Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RASPBERRY PI CAMERA                         │
│                    (1280x720 @ 30fps)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ TCP Stream (H.264)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VIDEO RECEIVER                             │
│                    (FFmpeg → Raw BGR)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ Frame (H×W×3)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                GABOR FEATURE EXTRACTOR                          │
│  ┌────────────┬────────────┬────────────┬────────────┐         │
│  │   0° Gabor │  45° Gabor │  90° Gabor │ 135° Gabor │         │
│  └────────────┴────────────┴────────────┴────────────┘         │
│         Applied to 324 Receptive Fields (18×18 grid)            │
└────────────────────────────┬────────────────────────────────────┘
                             │ Features (324 neurons × 4 orientations)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPIKE ENCODER                                │
│                                                                 │
│  Rate Coding    Latency Coding      Hybrid Coding              │
│  ───────────    ──────────────      ─────────────              │
│  More spikes    Earlier spike       Both combined              │
│  = stronger     = stronger          = richer                   │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ Spike Trains (324 neurons)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   V1 MODEL (NEST Simulator)                     │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ 0° Column │  │ 45° Column│  │ 90° Column│  │135° Column│   │
│  ├───────────┤  ├───────────┤  ├───────────┤  ├───────────┤   │
│  │ Layer 2/3 │  │ Layer 2/3 │  │ Layer 2/3 │  │ Layer 2/3 │   │
│  │  324 Pyr  │  │  324 Pyr  │  │  324 Pyr  │  │  324 Pyr  │   │
│  ├───────────┤  ├───────────┤  ├───────────┤  ├───────────┤   │
│  │ Layer 4   │  │ Layer 4   │  │ Layer 4   │  │ Layer 4   │   │
│  │  324 SS   │  │  324 SS   │  │  324 SS   │  │  324 SS   │   │
│  ├───────────┤  ├───────────┤  ├───────────┤  ├───────────┤   │
│  │ Layer 5   │  │ Layer 5   │  │ Layer 5   │  │ Layer 5   │   │
│  │   81 Pyr  │  │   81 Pyr  │  │   81 Pyr  │  │   81 Pyr  │   │
│  ├───────────┤  ├───────────┤  ├───────────┤  ├───────────┤   │
│  │ Layer 6   │  │ Layer 6   │  │ Layer 6   │  │ Layer 6   │   │
│  │  243 Pyr  │  │  243 Pyr  │  │  243 Pyr  │  │  243 Pyr  │   │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘   │
│                                                                 │
│  + Inhibitory interneurons in each layer                        │
│  + STDP synaptic plasticity                                     │
│  + Intrinsic excitability (IE) in Layer 4                       │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ V1 Spikes (all layers, all columns)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        V1 DECODER                               │
│                                                                 │
│  1. Calculate Orientation Selectivity                           │
│     → Which orientation fires most at each location?            │
│                                                                 │
│  2. Generate Visualizations:                                    │
│     • Orientation Map (color-coded preferences)                 │
│     • Activity Map (neural firing rates)                        │
│     • Edge Reconstruction (oriented line segments)              │
│                                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ Decoded Visual Representations
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION                              │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Raw    │  │  Gabor   │  │  Spikes  │  │V1 Output │       │
│  │  Video   │  │ Features │  │  Raster  │  │   Maps   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Details

### 1. Video Input
- **Source**: Raspberry Pi Camera Module
- **Format**: H.264 over TCP
- **Resolution**: 1280×720 (configurable)
- **FPS**: 30 (configurable)

### 2. Feature Extraction
- **Method**: Gabor filters (4 orientations)
- **Receptive Fields**: 324 (18×18 spatial grid)
- **Output**: 324×4 response matrix
- **Processing Time**: ~10-30ms

### 3. Spike Encoding
- **Input**: Feature responses (324×4)
- **Output**: Spike trains (neuron IDs + spike times)
- **Encoding Options**:
  - **Rate**: Response strength → spike count
  - **Latency**: Response strength → spike timing
  - **Hybrid**: Both rate and latency
- **Processing Time**: ~1-5ms

### 4. V1 Simulation
- **Simulator**: NEST (Network Simulator for Spiking Neural Networks)
- **Architecture**: 4 orientation columns
- **Total Neurons**: ~4,800 (including inhibitory)
- **Key Features**:
  - Spike latency mechanism
  - Intrinsic excitability plasticity
  - STDP (spike-timing-dependent plasticity)
- **Simulation Duration**: 200ms per frame
- **Processing Time**: ~50-150ms

### 5. Decoding
- **Input**: V1 spike trains (all layers, all columns)
- **Analysis**:
  - Count spikes per orientation column
  - Calculate preferred orientation per spatial location
  - Compute orientation selectivity index
- **Output**: Visual reconstructions
- **Processing Time**: ~5-15ms

### 6. Visualization
- **Windows**: 4 simultaneous displays
- **Update Rate**: Configurable (1-30 FPS)
- **Overlays**: Grid, statistics, FPS counter

## Component Dependencies

```
config.py
    ↓
    ├─→ gabor_feature_extractor.py
    │       ↓
    ├─→ spike_encoder.py
    │       ↓
    ├─→ v1_model_interface.py ←─┐
    │       ↓                    │
    ├─→ v1_decoder.py            │
    │       ↓                    │
    ├─→ visualization.py         │
    │       ↓                    │
    └─→ realtime_pipeline.py     │
            (orchestrates all)   │
                                 │
        MDPI2021/                │
        └─→ OrientedColumnV1.py ─┘
```

## Key Design Decisions

### Why 324 Neurons?
- Matches V1 model architecture (18×18 grid)
- Provides good spatial coverage
- Computationally feasible for real-time

### Why 4 Orientations?
- Biological V1 has continuous orientation tuning
- 4 orientations (0°, 45°, 90°, 135°) balance:
  - Computational efficiency
  - Orientation coverage
  - Model complexity

### Why Gabor Filters?
- Biologically inspired (model V1 simple cells)
- Mathematically tractable
- Well-suited for edge/orientation detection
- Proven effective in computer vision

### Why Spike Encoding?
- V1 model requires spike trains as input
- Biologically realistic communication
- Preserves temporal information
- Enables spike-timing-dependent mechanisms

### Why NEST Simulator?
- Industry-standard for spiking neural networks
- Efficient simulation of large networks
- Supports custom neuron models (LIFL_IE)
- Active development and community

## Performance Considerations

### Bottlenecks
1. **V1 Simulation** (~50-150ms) - largest bottleneck
   - Depends on number of spikes
   - More spikes = longer simulation
2. **Gabor Filtering** (~10-30ms)
   - Depends on image resolution
   - 4 convolutions per frame
3. **Frame Reception** (~5-10ms)
   - Network latency
   - FFmpeg decoding

### Optimization Strategies
1. **Downsample frames** before processing
2. **Process every Nth frame** (skip frames)
3. **Adjust spike encoding** (fewer spikes = faster simulation)
4. **Reduce grid resolution** (fewer than 324 neurons)

### Typical Performance
- **Full pipeline**: 70-200ms per frame
- **Achievable FPS**: 5-15 FPS
- **With frame skipping**: Up to 30 FPS display

## Biological Inspiration

### Real V1 Cortex
- **Neurons**: ~140 million in human V1
- **Layers**: 6 distinct layers (we model all)
- **Columns**: Orientation columns (we model 4)
- **Plasticity**: STDP and intrinsic excitability (included)
- **Receptive Fields**: Small, overlapping (modeled)

### Our Model
- **Scale**: ~5,000 neurons (0.004% of real V1)
- **Purpose**: Functional demonstration, not full simulation
- **Key Features**: Preserves essential V1 properties
  - Orientation selectivity ✓
  - Retinotopic organization ✓
  - Layered structure ✓
  - Spike timing dynamics ✓

## Extension Points

### Add Color Processing
- Create separate pathways for R, G, B
- Model color-opponent cells
- Integrate in V1 or add V2 area

### Add Motion Detection
- Implement temporal filtering
- Model MT/V5 area
- Track moving edges

### Add Attention
- Implement top-down modulation
- Selective enhancement of RFs
- Priority-based processing

### Add Learning
- Train receptive fields on data
- Adapt to input statistics
- Implement unsupervised learning

### Add Higher Areas
- V2: More complex features
- V4: Shape processing
- IT: Object recognition

