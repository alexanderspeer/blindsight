# Quick Start Guide

## Installation (30 seconds)

```bash
cd /Users/alexanderspeer/Desktop/blindsight/v1_computational

# Create virtual environment (first time only)
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

That's it! No NEST, no CMake, no compilation.

**Quick activation next time:**
```bash
source activate.sh
```

## Test (1 minute)

```bash
# Make sure venv is activated first!
source venv/bin/activate

# Run test
python test_static_image.py
```

You'll see:
- ✅ V1 model initialization (~5 seconds)
- ✅ Image processing through pipeline (~0.3 seconds)
- ✅ 4 visualization windows with results
- ✅ Performance metrics and timing

Press any key to close.

## What You're Seeing

### Window 1: Input vs V1 Output
- **Left**: Original test image with oriented edges
- **Right**: V1 orientation map (colored line segments)
  - Red = horizontal edges
  - Blue = vertical edges
  - Green/Yellow = diagonal edges

### Window 2: Gabor Features
- 4 orientation filters (0°, 45°, 90°, 135°)
- Heatmaps showing orientation-selective responses
- This is what V1 neurons "see"

### Window 3: Spike Trains
- Raster plots for each orientation column
- Each dot = neural spike
- X-axis = time (43-243 ms)
- Y-axis = neuron ID (0-323)
- More spikes = stronger feature

### Window 4: Layer Activity
- Firing rates across V1 layers (4, 2/3, 5, 6)
- Layer 2/3 is the main output layer
- Hotter colors = higher activity

## Run on Video

**Test video file:**
```bash
python realtime_pipeline.py /path/to/video.mp4
```

**Raspberry Pi camera:**
```bash
# 1. On Pi, start streaming:
ffmpeg -f v4l2 -i /dev/video0 -listen 1 -f mpegts tcp://0.0.0.0:5001

# 2. On this computer:
python realtime_pipeline.py
```

Press 'q' to quit.

## Customize

Edit `config.py`:

**Change video resolution:**
```python
VIDEO_CONFIG = {
    'width': 640,   # Smaller = faster
    'height': 480,
}
```

**Speed up simulation:**
```python
V1_ARCHITECTURE = {
    'warmup_time_ms': 0,      # Skip warmup for video
    'stimulus_time_ms': 100,  # Reduce from 200ms
}
```

**Adjust spike encoding:**
```python
SPIKE_CONFIG = {
    'threshold': 0.2,  # Higher = fewer spikes
}
```

## Next Steps

1. ✅ Test works? → Try real-time video
2. ✅ Want faster? → Reduce `stimulus_time_ms`
3. ✅ Understand architecture? → Read `README.md`
4. ✅ Modify behavior? → Edit `config.py`

## Troubleshooting

**Import errors?**
```bash
pip install numpy opencv-python
```

**Slow performance?**
- Normal! V1 simulation takes ~300ms per frame
- Reduce `stimulus_time_ms` in `config.py` for speed
- Or disable `warmup=True` in pipeline

**Windows don't appear?**
- Check OpenCV: `python -c "import cv2; print(cv2.__version__)"`
- macOS: May need to allow screen recording permissions

## What's Next?

This computational V1 is:
- ✅ **Fast**: No NEST compilation, runs immediately
- ✅ **Complete**: Same architecture as MDPI2021
- ✅ **Portable**: Works on any system with Python
- ✅ **Functional**: Real orientation selectivity

The output is an **orientation/edge map** - exactly what V1 extracts from visual input!

