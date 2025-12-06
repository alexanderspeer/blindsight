# Project Structure

Complete overview of the V1 Vision Pipeline file organization.

## File Tree

```
blindsight/
│
├── init-cv/                                  # Original video receiving scripts
│   ├── receive.py                            # Basic Pi video receiver
│   └── cv_receive_edges_luminance.py         # Example with CV processing
│
├── MDPI2021/                                 # V1 cortex model (external)
│   ├── README.md
│   ├── LIFL_IE/                              # NEST custom neuron module
│   │   ├── lifl_psc_exp_ie.cpp               # Neuron model implementation
│   │   ├── lifl_psc_exp_ie.h                 # Header
│   │   ├── CMakeLists.txt                    # Build configuration
│   │   └── ...
│   └── Examples/
│       └── V1 Oriented Columns comapred with MEG/
│           ├── OrientedColumnV1.py           # V1 column definition
│           ├── Simulation_V1_pinwheel_MEGcomparison.py
│           └── files/                        # Pre-trained weights
│               ├── soma_exc_0.pckl
│               ├── soma_exc_45.pckl
│               ├── soma_exc_90.pckl
│               └── soma_exc_135.pckl
│
└── v1_vision/                                # ← OUR PIPELINE (THIS DIRECTORY)
    │
    ├── README.md                             # Full documentation
    ├── QUICKSTART.md                         # 5-minute setup guide
    ├── ARCHITECTURE.md                       # System design details
    ├── PROJECT_STRUCTURE.md                  # This file
    ├── requirements.txt                      # Python dependencies
    │
    ├── __init__.py                           # Package initialization
    ├── config.py                             # ⚙️ All configuration parameters
    │
    ├── gabor_feature_extractor.py            # 🔍 Extract oriented edges
    ├── spike_encoder.py                      # ⚡ Convert to spike trains
    ├── v1_model_interface.py                 # 🧠 V1 cortex simulation
    ├── v1_decoder.py                         # 🎨 Reconstruct visual output
    ├── visualization.py                      # 📊 Display utilities
    │
    ├── realtime_pipeline.py                  # ▶️ MAIN SCRIPT - Run this!
    └── test_static_image.py                  # 🧪 Test with static images
```

## Core Files (Required)

### 1. **config.py** ⚙️
**Purpose**: Central configuration for entire pipeline
**Contains**:
- Video stream settings (IP, port, resolution)
- Spatial grid parameters (18×18 = 324 neurons)
- Gabor filter settings (orientations, wavelength, sigma)
- Spike encoding parameters (rate/latency/hybrid)
- V1 model settings (simulation time, layers)
- Visualization options (which windows to show)
- Performance tuning (downsampling, frame skipping)

**Edit this first** to configure for your setup!

### 2. **gabor_feature_extractor.py** 🔍
**Purpose**: Extract orientation-selective visual features
**Key Class**: `GaborFeatureExtractor`
**Methods**:
- `extract_features(frame)` → Returns responses for 324 neurons × 4 orientations
- `visualize_gabor_responses()` → Creates 2×2 grid of filtered images
- `visualize_receptive_fields()` → Draws grid on frame

**What it does**: Applies Gabor filters (edge detectors) at 4 orientations to 324 spatial locations.

### 3. **spike_encoder.py** ⚡
**Purpose**: Convert visual features to neural spike trains
**Key Class**: `SpikeEncoder`
**Methods**:
- `encode(features)` → Generates spike trains
- `format_for_nest()` → Formats for NEST simulator

**Encoding Types**:
- **Rate**: More spikes for stronger features
- **Latency**: Earlier spikes for stronger features
- **Hybrid**: Both combined

### 4. **v1_model_interface.py** 🧠
**Purpose**: Interface to NEST-based V1 cortex model
**Key Class**: `V1ModelInterface`
**Methods**:
- `setup_model()` → Creates 4 orientation columns
- `inject_spikes(spike_trains)` → Injects input
- `run_simulation()` → Runs NEST simulation
- `get_output()` → Returns V1 spikes
- `calculate_orientation_selectivity()` → Analyzes responses

**What it does**: Simulates ~5,000 cortical neurons processing visual input.

### 5. **v1_decoder.py** 🎨
**Purpose**: Reconstruct visual representations from V1 spikes
**Key Class**: `V1Decoder`
**Methods**:
- `decode(v1_output)` → Creates visualizations
- `create_orientation_map()` → Color-coded orientation preferences
- `create_activity_map()` → Neural activity heatmap
- `reconstruct_edges()` → Oriented line segments

**What it does**: Interprets V1 output and creates human-readable visualizations.

### 6. **visualization.py** 📊
**Purpose**: Display utilities and performance monitoring
**Key Classes**:
- `SpikeRasterPlot` → Real-time spike visualization
- `PipelineMonitor` → FPS and timing statistics
- `MultiWindowDisplay` → Manages multiple windows

### 7. **realtime_pipeline.py** ▶️
**Purpose**: MAIN SCRIPT - Orchestrates entire pipeline
**Key Class**: `V1VisionPipeline`
**What it does**:
1. Receives video from Pi
2. Extracts Gabor features
3. Encodes to spikes
4. Runs V1 simulation
5. Decodes output
6. Displays all stages

**Run this to start the pipeline!**

### 8. **test_static_image.py** 🧪
**Purpose**: Test pipeline with static images (no Pi needed)
**What it does**:
- Tests each component independently
- Useful for debugging
- Can use generated test images or your own

**Run this first** to verify installation!

## Documentation Files

### **README.md**
Complete documentation:
- Architecture overview
- Setup instructions
- Configuration guide
- Troubleshooting
- Technical details

### **QUICKSTART.md**
Get running in 5 minutes:
- Installation steps
- Quick test
- Common issues

### **ARCHITECTURE.md**
System design:
- Data flow diagrams
- Component interactions
- Performance analysis
- Extension points

### **PROJECT_STRUCTURE.md** (this file)
File organization and purpose of each component.

## Configuration Files

### **requirements.txt**
Python package dependencies:
```
numpy
opencv-python
nest-simulator
matplotlib (optional)
```

### **__init__.py**
Makes v1_vision a Python package. Exports main classes.

## Usage Patterns

### Quick Test (No Pi)
```bash
python test_static_image.py
```

### Full Pipeline
```bash
# 1. Edit config.py (set Pi IP)
# 2. Start Pi camera stream
# 3. Run pipeline
python realtime_pipeline.py
```

### Import as Package
```python
from v1_vision import (
    GaborFeatureExtractor,
    SpikeEncoder,
    V1ModelInterface,
    V1Decoder
)

# Use components individually
extractor = GaborFeatureExtractor()
features = extractor.extract_features(my_image)
```

## External Dependencies

### MDPI2021 V1 Model
- Located in `../MDPI2021/`
- Contains pre-trained cortical column
- Requires NEST simulator
- Custom LIFL_IE neuron module

**Must be compiled separately!** See `MDPI2021/LIFL_IE/` for instructions.

### Pre-trained Weights
Located in: `MDPI2021/Examples/V1 Oriented Columns comapred with MEG/files/`
- `soma_exc_0.pckl` → 0° orientation column weights
- `soma_exc_45.pckl` → 45° orientation column weights  
- `soma_exc_90.pckl` → 90° orientation column weights
- `soma_exc_135.pckl` → 135° orientation column weights

These files contain trained intrinsic excitability parameters.

## Typical Workflow

### Development
1. Edit `config.py` → adjust parameters
2. Run `test_static_image.py` → verify changes
3. Run `realtime_pipeline.py` → test on video
4. Iterate

### Deployment
1. Configure Pi IP in `config.py`
2. Start Pi camera stream
3. Run `realtime_pipeline.py`
4. Monitor performance (FPS display)

### Debugging
1. Use `test_static_image.py` with known images
2. Check each stage output window
3. Review spike counts and firing rates
4. Adjust encoding parameters in `config.py`

## File Relationships

```
config.py (parameters)
    ↓
gabor_feature_extractor.py (uses config)
    ↓
spike_encoder.py (uses config)
    ↓
v1_model_interface.py (uses config, imports OrientedColumnV1.py)
    ↓
v1_decoder.py (uses config)
    ↓
visualization.py (uses config)
    ↓
realtime_pipeline.py (orchestrates all, uses config)
```

All components read from `config.py`, allowing centralized parameter tuning.

## Size Reference

| File | Lines | Purpose |
|------|-------|---------|
| config.py | 94 | Configuration |
| gabor_feature_extractor.py | 247 | Feature extraction |
| spike_encoder.py | 244 | Spike encoding |
| v1_model_interface.py | 247 | V1 simulation |
| v1_decoder.py | 235 | Output decoding |
| visualization.py | 290 | Display utilities |
| realtime_pipeline.py | 313 | Main integration |
| test_static_image.py | 207 | Testing |
| **Total** | **~1,877 lines** | Complete pipeline |

## Getting Help

1. **Setup issues** → See QUICKSTART.md
2. **Understanding system** → See ARCHITECTURE.md
3. **Configuration** → See README.md
4. **Code questions** → Read inline comments (all files well-documented)
5. **Performance** → See ARCHITECTURE.md performance section

## Next Steps

1. ✅ Read QUICKSTART.md
2. ✅ Install dependencies (`pip install -r requirements.txt`)
3. ✅ Test with static image (`python test_static_image.py`)
4. ✅ Configure Pi IP in `config.py`
5. ✅ Run full pipeline (`python realtime_pipeline.py`)
6. ✅ Experiment with parameters in `config.py`
7. ✅ Read ARCHITECTURE.md for deeper understanding

Enjoy exploring biologically-inspired vision! 🧠👁️

