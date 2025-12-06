# V1 Vision Pipeline

A biologically-inspired real-time visual processing system that mimics the primary visual cortex (V1). This pipeline processes video from a Raspberry Pi camera through Gabor filters, converts features to spike trains, simulates V1 cortex processing using a spiking neural network, and reconstructs visual representations.

## Architecture Overview

```
Raspberry Pi Camera
        ↓
   Video Stream (TCP)
        ↓
  [Gabor Feature Extraction] ← 4 orientations (0°, 45°, 90°, 135°)
        ↓
  [Spike Encoding] ← Converts features to spike trains
        ↓
  [V1 Model (NEST)] ← 324 neurons × 4 orientation columns
        ↓
  [V1 Decoder] ← Reconstructs orientation maps
        ↓
   Visualization
```

## Components

### 1. **config.py**
Configuration file for all pipeline parameters:
- Video stream settings (IP, resolution, FPS)
- Spatial grid configuration (18×18 = 324 neurons)
- Gabor filter parameters
- Spike encoding settings
- V1 model parameters
- Visualization options

### 2. **gabor_feature_extractor.py**
Extracts orientation-selective features using Gabor filters:
- Applies 4 Gabor filters at different orientations
- Organizes 324 receptive fields in spatial grid
- Returns responses for each neuron at each orientation

### 3. **spike_encoder.py**
Converts visual features to spike trains:
- Supports 3 encoding schemes:
  - **Rate coding**: More spikes for stronger responses
  - **Latency coding**: Earlier spikes for stronger responses (biologically realistic)
  - **Hybrid coding**: Combines both approaches
- Formats output for NEST simulator

### 4. **v1_model_interface.py**
Interface to the NEST-based V1 cortex model:
- Creates 4 orientation columns (0°, 45°, 90°, 135°)
- Each column: 324 neurons across 6 cortical layers
- Injects spike trains and runs simulation
- Extracts output spikes and calculates orientation selectivity

### 5. **v1_decoder.py**
Reconstructs visual representations from V1 output:
- Creates orientation preference maps
- Generates activity heatmaps
- Reconstructs edge maps using oriented line segments
- Color-codes orientations

### 6. **visualization.py**
Visualization utilities:
- Spike raster plots
- Performance monitoring (FPS, processing time)
- Multi-window display management

### 7. **realtime_pipeline.py**
Main integration script that ties everything together.

## Prerequisites

### Hardware
- Raspberry Pi with camera module
- Computer with network connection to Pi

### Software
```bash
# Python packages
pip install opencv-python numpy nest-simulator

# NEST simulator custom module (LIFL_IE)
# Must be compiled and installed - see MDPI2021/LIFL_IE/
```

## Setup

### 1. Start Video Stream on Raspberry Pi

SSH into your Raspberry Pi and run:

```bash
rpicam-vid \
    --codec h264 \
    --profile high \
    --level 4.2 \
    --width 1280 \
    --height 720 \
    --framerate 30 \
    --bitrate 8000000 \
    --inline \
    --intra 30 \
    --timeout 0 \
    --listen \
    -o tcp://0.0.0.0:5001
```

### 2. Update Configuration

Edit `config.py` and set your Pi's IP address:

```python
VIDEO_CONFIG = {
    'pi_ip': 'YOUR_PI_IP_HERE',  # e.g., '10.207.55.64'
    'port': 5001,
    # ...
}
```

### 3. Run Pipeline

```bash
cd v1_vision
python realtime_pipeline.py
```

## Display Windows

When running, you'll see 4 windows:

1. **Raw Video Stream**: Original video with receptive field grid overlay
2. **Gabor Features**: 2×2 grid showing responses at 4 orientations
3. **Spike Train Raster**: Real-time plot of generated spikes
4. **V1 Output**: Three views:
   - Orientation Map (color-coded preferred orientations)
   - Activity Map (neural activity heatmap)
   - Edge Reconstruction (oriented line segments)

## Configuration Options

### Spike Encoding

Choose encoding type in `config.py`:

```python
SPIKE_CONFIG = {
    'encoding_type': 'latency',  # 'rate', 'latency', or 'hybrid'
    # ...
}
```

- **Latency**: More biologically realistic, matches V1 model's spike latency mechanism
- **Rate**: Traditional rate coding
- **Hybrid**: Combines both for richer encoding

### Performance Tuning

For faster processing:

```python
PROCESSING_CONFIG = {
    'downsample_frame': True,
    'downsample_width': 640,
    'downsample_height': 360,
}

VISUALIZATION_CONFIG = {
    'update_interval_frames': 2,  # Process every 2nd frame
}
```

### Grid Resolution

Adjust spatial resolution by changing grid size (must result in 324 neurons):

```python
GRID_CONFIG = {
    'grid_rows': 18,  # 18×18 = 324
    'grid_cols': 18,
}
```

## Understanding the Output

### Orientation Map Colors
- 🔴 **Red**: 0° (horizontal edges)
- 🟢 **Green**: 45° (diagonal edges ↗)
- 🔵 **Blue**: 90° (vertical edges)
- 🟡 **Yellow**: 135° (diagonal edges ↖)

### Activity Map
- **Hot colors (red/yellow)**: High neural activity
- **Cool colors (blue/purple)**: Low neural activity

### Edge Reconstruction
- Colored line segments show detected edges
- Line orientation matches preferred orientation
- Line length indicates response strength

## Troubleshooting

### "Could not connect to video stream"
- Verify Pi is on network: `ping YOUR_PI_IP`
- Check rpicam-vid is running on Pi
- Verify firewall allows port 5001

### "Could not load LIFL_IE module"
- NEST custom module must be compiled
- See `MDPI2021/LIFL_IE/` for installation
- Check NEST installation: `python -c "import nest; print(nest.__version__)"`

### Slow Performance
- Reduce frame resolution in config
- Increase `update_interval_frames`
- Enable frame downsampling
- Close unnecessary windows in VISUALIZATION_CONFIG

### "Model not initialized"
- NEST module failed to load
- Check V1 model path in config matches actual location

## File Structure

```
blindsight/
├── init-cv/
│   ├── receive.py                    # Basic video receiver
│   └── cv_receive_edges_luminance.py # Example CV processing
├── MDPI2021/                         # V1 cortex model
│   ├── LIFL_IE/                      # NEST custom module
│   └── Examples/
│       └── V1 Oriented Columns.../
│           ├── OrientedColumnV1.py   # Column definition
│           └── files/                # Pre-trained weights
└── v1_vision/                        # This pipeline
    ├── README.md
    ├── config.py
    ├── gabor_feature_extractor.py
    ├── spike_encoder.py
    ├── v1_model_interface.py
    ├── v1_decoder.py
    ├── visualization.py
    └── realtime_pipeline.py          # ← RUN THIS
```

## Technical Details

### V1 Model Architecture
- **Input**: 324 LGN neurons (parrot neurons relaying retinal spikes)
- **Layer 4**: 324 Spiny Stellate cells with intrinsic excitability
- **Layers 2/3**: 324 Pyramidal cells (primary output)
- **Layer 5**: 81 Pyramidal cells
- **Layer 6**: 243 Pyramidal cells
- **Inhibitory interneurons** in each layer
- **4 columns** tuned to different orientations
- **STDP** (spike-timing-dependent plasticity) between neurons
- **Intrinsic excitability** plasticity in layer 4

### Gabor Filters
Gabor filters are biologically-motivated edge detectors that model V1 simple cells:
- Combine Gaussian envelope with sinusoidal grating
- Selective for orientation, spatial frequency, and phase
- Each neuron has a localized receptive field

### Spike Encoding
The model uses **spike timing** for information encoding:
- Strong visual responses → early/frequent spikes
- Weak responses → late/sparse spikes or no spikes
- Spike latency mechanism in model allows precise temporal coding

## Performance

Typical performance on modern hardware:
- **Gabor extraction**: ~10-30ms
- **Spike encoding**: ~1-5ms
- **V1 simulation**: ~50-150ms (depends on spike count)
- **Decoding**: ~5-15ms
- **Total**: ~70-200ms per frame (~5-15 FPS)

For real-time operation at 30 FPS, set `update_interval_frames` to skip frames.

## Citation

This pipeline uses the V1 cortex model from:

**A bio-inspired model of the early visual system, based on parallel spike-sequence detection, showing orientation selectivity**

MDPI2021 Repository: NEST Simulator Extension Module containing a LeakyIFwithLatency neuron model expressing Intrinsic Excitability (IE) plasticity.

Contact: alejandro.santos@ctb.upm.es

## Controls

- **'q'**: Quit pipeline
- Window management: Resize/move windows as needed

## Future Enhancements

Possible extensions:
- Color processing (separate channels)
- Motion detection (temporal filtering)
- Higher-level visual areas (V2, V4, IT)
- Attention mechanisms
- Learning/adaptation of receptive fields
- Save/replay recorded sessions

