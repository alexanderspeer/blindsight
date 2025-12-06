# Quick Start Guide

Get the V1 Vision Pipeline running in 5 minutes!

## Step 1: Install Dependencies

```bash
cd v1_vision
pip install -r requirements.txt
```

**Important**: The NEST LIFL_IE module must be compiled separately. See `../MDPI2021/LIFL_IE/` for instructions.

## Step 2: Test with Static Image (No Pi Required)

Before setting up the video stream, test the pipeline with a static image:

```bash
python test_static_image.py
```

This will:
- Generate a test image with oriented bars
- Process it through the entire pipeline
- Display 5 windows showing each stage

**Expected output**: You should see oriented edges detected and color-coded by orientation.

## Step 3: Setup Raspberry Pi Video Stream

On your **Raspberry Pi**, run:

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

## Step 4: Configure Pipeline

Edit `config.py` and set your Pi's IP address:

```python
VIDEO_CONFIG = {
    'pi_ip': '10.207.55.64',  # ← CHANGE THIS to your Pi's IP
    'port': 5001,
    # ...
}
```

To find your Pi's IP:
```bash
# On the Pi:
hostname -I
```

## Step 5: Run Real-time Pipeline

```bash
python realtime_pipeline.py
```

You should see:
1. **Raw Video Stream** - with 18×18 grid overlay
2. **Gabor Features** - 4 orientation filters
3. **Spike Trains** - real-time neural spikes
4. **V1 Output** - orientation map + edge reconstruction

Press **'q'** to quit.

## Troubleshooting

### "Could not connect to video stream"

Check Pi connection:
```bash
ping YOUR_PI_IP
```

Verify rpicam-vid is running on Pi (you should see camera LED on).

### "Could not load LIFL_IE module"

The NEST custom module needs to be compiled:
```bash
cd ../MDPI2021/LIFL_IE
# Follow compilation instructions in that directory
```

### "Pipeline is slow / low FPS"

In `config.py`, adjust:
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

## Understanding the Output

### Colors in Orientation Map:
- 🔴 **Red** = 0° (horizontal)
- 🟢 **Green** = 45° (diagonal /)
- 🔵 **Blue** = 90° (vertical)
- 🟡 **Yellow** = 135° (diagonal \)

Brighter colors = stronger responses.

### Activity Map:
- **Hot (red/yellow)** = High activity
- **Cool (blue/purple)** = Low activity

### Edge Reconstruction:
- Colored lines show detected edges
- Line angle = detected orientation
- Line brightness = response strength

## What's Happening?

1. **Gabor Filters** extract oriented edges (mimics V1 simple cells)
2. **Spike Encoder** converts visual features to neural spikes
3. **V1 Model** processes spikes through 4 orientation-selective columns
4. **Decoder** reconstructs what V1 "sees"

## Next Steps

- Try different encoding modes in `config.py` (`'rate'`, `'latency'`, `'hybrid'`)
- Adjust Gabor filter parameters to tune sensitivity
- Point camera at objects with clear edges for best results
- Read `README.md` for detailed documentation

## Example Commands

Test with your own image:
```bash
python test_static_image.py path/to/your/image.jpg
```

Generate different test patterns:
```python
# Edit test_static_image.py, line 134:
test_pipeline(image_path=None, image_type='edges')  # or 'checkerboard'
```

## Support

For issues:
1. Check `README.md` troubleshooting section
2. Verify NEST installation: `python -c "import nest; print(nest.__version__)"`
3. Test components individually using `test_static_image.py`

Enjoy exploring biologically-inspired vision! 🧠👁️

