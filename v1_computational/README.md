# **git add v1_computational / git commit /  git subtree push --prefix=v1_computational school main - **

# Computational V1 Vision Pipeline

## Architecture

### Exact V1 Replication

 replicates the MDPI2021 V1 structure:

**4 Orientation Columns** (0°, 45°, 90°, 135°)

Each column contains:
- **Layer 4**: 324 Spiny Stellate (input) + 65 Inhibitory neurons
- **Layer 2/3**: 324 Pyramidal (output) + 65 Inhibitory neurons
- **Layer 5**: 81 Pyramidal + 16 Inhibitory neurons
- **Layer 6**: 243 Pyramidal + 49 Inhibitory neurons

**Total: ~4,800 neurons** with realistic connectivity and dynamics

### Processing Pipeline

```
Video Frame
    ↓
Gabor Feature Extraction (18x18 grid, 4 orientations)
    ↓
Spike Encoding (Latency coding: strong features → early spikes)
    ↓
V1 Model (4 orientation columns, 8 layers each)
    ↓
Orientation Map Reconstruction (detected edges/orientations)
```

### Installation

```bash
cd /Users/alexanderspeer/Desktop/blindsight/v1_computational
pip install -r requirements.txt
```

### Test with Static Image

```bash
python test_static_image.py
```
1. Create a test image with oriented edges
2. Process through the complete V1 pipeline
3. Display all intermediate stages
4. Show performance metrics

### Real-time Processing

**From Raspberry Pi Camera:**
```bash
python realtime_pipeline.py
```

**From Video File:**
```bash
python realtime_pipeline.py path/to/video.mp4
```

## What You'll See

### Visualization Windows

1. **Comparison View**: Input image vs V1 orientation map
2. **Gabor Features**: 4 orientation-selective filters (0°, 45°, 90°, 135°)
3. **Spike Trains**: Raster plots showing neural spikes
4. **Layer Activity**: Firing rates across all V1 layers

### V1 Output

The reconstructed image is an **orientation/edge map** showing:
- Detected edge orientations (color-coded)
- Response strengths (brightness)
- Spatial organization (18x18 grid matching input)

## File Structure

```
v1_computational/
├── config.py              # Central configuration
├── neurons.py             # LIF neuron models
├── v1_column.py          # Single orientation column
├── v1_model.py           # Complete 4-column V1
├── gabor_extractor.py    # Gabor feature extraction
├── spike_encoder.py      # Feature → spike conversion
├── v1_decoder.py         # V1 → orientation map
├── pipeline.py           # Main pipeline orchestration
├── test_static_image.py  # Static image test
├── realtime_pipeline.py  # Real-time video processing
└── requirements.txt      # Python dependencies
```

## Configuration

If you want to edit
Edit `config.py` to modify:
- Video source settings
- Gabor filter parameters
- Spike encoding parameters
- V1 architecture (neuron counts, connections)
- Visualization options

## Performance

**Typical timing per frame:**
- Preprocessing: ~1-2 ms
- Gabor extraction: ~5-10 ms
- Spike encoding: ~1-2 ms
- V1 simulation: ~100-200 ms (400ms warmup + 200ms stimulus)
- Decoding: ~5-10 ms

**Total: ~300-400 ms per frame (~2-3 FPS)**

For faster performance:
- Reduce `stimulus_time_ms` in config
- Skip `warmup_time_ms` for continuous video
- Reduce grid size (18x18 → smaller)


### Helping to Understand the Orientation Map

Colors indicate preferred orientation:
- **Red**: 0° (horizontal)
- **Green**: 45° (diagonal /)
- **Blue**: 90° (vertical)
- **Yellow**: 135° (diagonal \)

## Differences from NEST Version

| Feature | NEST Version | Computational Version |
|---------|--------------|----------------------|
| **Neurons** | Full NEST simulation | LIF computational model |
| **Synapses** | NEST synapse models | Direct weight application |
| **Dynamics** | Differential equations | Simplified integration |
| **Speed** | Slower (~minutes) | Fast (~300ms per frame) |
| **Dependencies** | NEST + C++ module | NumPy + OpenCV only |
| **Compatibility** | NEST 3.x required | Works anywhere |
| **Output** | Same orientation maps | Same orientation maps |

## References

Santos-Mayo, Alejandro, Stephan Moratti, Javier de Echegaray, and Gianluca Susi. 
“A Model of the Early Visual System Based on Parallel Spike-Sequence Detection, 
Showing Orientation Selectivity.” *Biology* 10, no. 8 (2021): 801. 
https://doi.org/10.3390/biology10080801.


