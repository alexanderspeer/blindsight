# Blindsight V1 System Setup Guide

## Quick Start (Raspberry Pi 4 / Linux)

### 1. Install NEST Simulator

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake libgsl-dev python3-dev python3-pip

# Download NEST 2.20.1 (compatible with this module)
wget https://github.com/nest/nest-simulator/archive/v2.20.1.tar.gz
tar -xzf v2.20.1.tar.gz
cd nest-simulator-2.20.1

# Build NEST
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/nest \
      -Dwith-python=3 \
      -Dwith-gsl=ON \
      ..
make -j4
make install

# Add to path
echo 'export PATH=$HOME/nest/bin:$PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$HOME/nest/lib/python3.7/site-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Build LIFL_IE Module

```bash
cd /path/to/MDPI2021/LIFL_IE

# Create build directory
mkdir build && cd build

# Configure
cmake -Dwith-nest=$HOME/nest/bin/nest-config ..

# Build and install
make -j4
make install
```

### 3. Install Python Dependencies

```bash
pip3 install -r requirements_blindsight.txt
```

### 4. Test Installation

```python
python3
>>> import nest
>>> nest.Install('LIFL_IEmodule')
>>> print(nest.Models())  # Should include 'lifl_psc_exp_ie'
```

### 5. Run Blindsight System

```bash
# Test with webcam
python3 blindsight_realtime_v1.py --lightweight

# Test with video file
python3 blindsight_realtime_v1.py --video test_video.mp4 --lightweight

# Run for 60 seconds
python3 blindsight_realtime_v1.py --lightweight --duration 60
```

---

## Hardware-Specific Setup

### Raspberry Pi + PiCamera

```bash
# Enable camera
sudo raspi-config
# Navigate to: Interfacing Options > Camera > Enable

# Install PiCamera2
sudo apt-get install -y python3-picamera2

# Modify blindsight_realtime_v1.py to use PiCamera2
# (See commented section in _init_camera method)
```

### Arduino + Serial Camera

```bash
# Install PySerial
pip3 install pyserial

# Find Arduino port
ls /dev/ttyACM* /dev/ttyUSB*

# Test connection
python3 -c "import serial; s = serial.Serial('/dev/ttyACM0', 115200); print(s.readline())"
```

**Arduino Sketch** (spike_camera_encoder.ino):

```cpp
// Simple Arduino spike encoder
// Reads analog camera sensor, sends spikes via serial

const int CAMERA_PIN = A0;
const int N_NEURONS = 81;
const int THRESHOLD = 512;

void setup() {
  Serial.begin(115200);
  pinMode(CAMERA_PIN, INPUT);
}

void loop() {
  unsigned long timestamp = millis();
  
  for (int neuron_id = 0; neuron_id < N_NEURONS; neuron_id++) {
    int intensity = analogRead(CAMERA_PIN);
    
    // Simple threshold spiking
    if (intensity > THRESHOLD) {
      // Send: neuron_id,timestamp
      Serial.print(neuron_id);
      Serial.print(",");
      Serial.println(timestamp);
      
      delay(1);  // Simple refractory period
    }
  }
  
  delay(33);  // ~30 FPS
}
```

---

## Performance Tuning

### For Raspberry Pi 4 (4GB)

**Lightweight Mode (Recommended):**
- 81 neurons per column × 4 orientations = 324 neurons total
- Expected speed: ~10-20x realtime
- Suitable for real-time operation

```python
system = BlindSightV1System(
    orientations=[0, 45, 90, 135],
    lightweight=True  # Use this on RPi
)
```

**Full Mode:**
- 1167 neurons per column × 4 orientations = 4668 neurons
- Expected speed: ~1-5x realtime
- May lag on RPi 4, use desktop for development

### Multi-threading Optimization

```python
# In blindsight_realtime_v1.py, adjust threads:
nest.SetKernelStatus({
    'local_num_threads': 4  # Set to number of CPU cores
})
```

### Reduce Visualization Overhead

```bash
# Disable visualization for max speed
python3 blindsight_realtime_v1.py --lightweight --no-vis
```

---

## Troubleshooting

### NEST Module Not Found

```python
# Manually install module path
import nest
nest.Install('/path/to/MDPI2021/LIFL_IE/build/LIFL_IEmodule')
```

### Slow Simulation

**Diagnosis:**
```python
# Add profiling to simulation_thread
import cProfile
cProfile.run('system.run(duration=10)')
```

**Solutions:**
1. Reduce resolution: `nest.SetKernelStatus({'resolution': 0.5})  # 0.5 ms instead of 0.1 ms`
2. Decrease camera FPS: `self.camera.set(cv2.CAP_PROP_FPS, 15)  # 15 FPS instead of 30`
3. Use single orientation column for testing

### Camera Not Opening

```python
# List available cameras
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
```

### Import Errors

```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Add local modules
export PYTHONPATH=$PWD:$PYTHONPATH
```

---

## Advanced: Custom Spike Encoding

### Implement Your Own Encoder

```python
# In blindsight_camera_encoder.py

class CustomSpikeEncoder(CameraSpikeEncoder):
    def encode_custom(self, frame, current_time):
        """
        Implement your custom encoding strategy
        E.g., DVS-like event camera, predictive coding, etc.
        """
        # Your algorithm here
        spikes = {}
        
        # Example: Spike on motion detection
        if self.prev_frame is not None:
            diff = cv2.absdiff(frame, self.prev_frame)
            motion_pixels = np.where(diff > threshold)
            
            for y, x in zip(*motion_pixels):
                neuron_id = y * self.resolution[1] + x
                spikes[neuron_id] = [current_time]
        
        self.prev_frame = frame.copy()
        return spikes
```

### Use with V1 System

```python
# Replace encoder in blindsight_realtime_v1.py
system.encoder = CustomSpikeEncoder(...)
```

---

## Benchmarking

```bash
# Run benchmark
python3 -c "
from blindsight_realtime_v1 import BlindSightV1System
import time

system = BlindSightV1System(lightweight=True, enable_visualization=False)
start = time.time()
system.run(duration=10)
elapsed = time.time() - start

print(f'Processed {system.frame_count} frames in {elapsed:.2f}s')
print(f'Average FPS: {system.frame_count / elapsed:.1f}')
"
```

---

## Example Applications

### 1. Robotic Navigation

```python
# Orient robot based on dominant edge orientation
def control_robot(decision):
    angle = decision['dominant_orientation']
    confidence = decision['confidence']
    
    if confidence > 0.7:
        if angle in [0, 180]:
            # Horizontal edge → move forward
            motor_control.forward()
        elif angle in [45, 135]:
            # Diagonal → turn
            motor_control.turn(angle)
        elif angle == 90:
            # Vertical edge → stop or reorient
            motor_control.stop()
```

### 2. Texture Classification

```python
# Classify textures based on orientation distribution
def classify_texture(decision_history):
    orientations = [d['dominant_orientation'] for d in decision_history[-100:]]
    
    # Calculate entropy
    from scipy.stats import entropy
    hist, _ = np.histogram(orientations, bins=4)
    texture_entropy = entropy(hist)
    
    if texture_entropy < 0.5:
        return "structured"  # Dominant orientation
    else:
        return "random"  # Uniform distribution
```

### 3. Obstacle Detection

```python
# Detect vertical edges (potential obstacles)
def detect_obstacle(decision):
    if decision['dominant_orientation'] == 90 and decision['confidence'] > 0.6:
        print("OBSTACLE DETECTED - Vertical edge ahead")
        return True
    return False
```

---

## Citation

If you use this integration for research, please cite the original paper:

```bibtex
@article{santos2021lifl,
  title={LIFL\_IE NEST Simulator Extension Module},
  author={Santos-Mayo, Alejandro and others},
  journal={MDPI},
  year={2021},
  note={GitHub: github.com/alejandrosantmayo/MDPI2021}
}
```

---

## Support

- **Original Module**: alejandro.santos@ctb.upm.es
- **Integration Issues**: Check GitHub issues
- **NEST Documentation**: https://nest-simulator.readthedocs.io/

