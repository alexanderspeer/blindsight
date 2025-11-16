# V1 Spiking Visual Cortex Integration Guide
## For Blindsight-like Embedded Vision System

---

## 1. INPUT INJECTION INTO LAYER 4

### **Current Architecture**

```python
# Simulation_V1_pinwheel_MEGcomparison.py, lines 38-45
GCells = 324  # Number of receptive fields (18x18 grid)
inputs = nest.Create("spike_generator", GCells)  # ← SPIKE INJECTION POINT
LGN = nest.Create('parrot_neuron', GCells)      # LGN relay neurons
nest.Connect(inputs, LGN, 'one_to_one')         # 1:1 mapping

# OrientedColumnV1.py, lines 48-52
nest.Connect(LGN, SS4, {'rule': 'one_to_one'}, {
    "weight": 15000.0,      # Strong feedforward drive
    "delay": lgn2v1_delay   # 1.0 ms LGN→V1 delay
})
# SS4 = 324 Spiny Stellate cells in Layer 4
```

### **Key Functions:**

```python
# FUNCTION: column(setdegree, LGN)
# Location: OrientedColumnV1.py, line 1
# Purpose: Creates single orientation column
# Parameters:
#   - setdegree: Orientation preference (0, 45, 90, 135)
#   - LGN: Array of 324 parrot neurons receiving retinal input

# CRITICAL CONNECTION: Line 50-52
# This is where external spikes enter Layer 4
```

### **For Your Integration:**

**REPLACE:** `nest.Create("spike_generator", 324)` with your camera spike stream

**METHOD 1 - Offline (Batch Processing):**
```python
# After feature extraction on RPi/Arduino
for neuron_idx in range(324):
    spike_times = your_camera_spike_train[neuron_idx]  # List of times in ms
    nest.SetStatus([inputs[neuron_idx]], {'spike_times': spike_times})
```

**METHOD 2 - Real-time (Recommended for blindsight):**
```python
# Use dynamic spike injection during simulation
def inject_camera_spikes(inputs, camera_buffer, current_time):
    """
    Inject spikes from camera buffer into NEST simulation
    
    Args:
        inputs: NEST spike_generator array (324 units)
        camera_buffer: Queue with (neuron_id, spike_time) tuples
        current_time: Current NEST simulation time (ms)
    """
    for neuron_id, spike_time in camera_buffer.get_recent_spikes():
        nest.SetStatus([inputs[neuron_id]], {
            'spike_times': [current_time + spike_time]
        })
```

---

## 2. LGN INPUT STRUCTURE

### **Current Preprocessing Pipeline**

```python
# Simulation_V1_pinwheel_MEGcomparison.py, lines 59-70

# STEP 1: Load pre-computed Gabor-filtered spike responses
file = "./files/spikes_reponse_gabor_randn02_19.pckl"
with open(file, 'rb') as f:
    resultados = pickle.load(f)
    senders = resultados['senders']  # Neuron IDs (0-323)
    times = resultados['times']      # Spike times (ms)

# STEP 2: Inject spikes into spike generators
currtime = nest.GetKernelStatus('time')
for j in range(324):
    spike_time = times[senders == min(senders)+j] + currtime
    nest.SetStatus([inputs[j]], {'spike_times': spike_time})
```

### **Spatial Layout:**
- **324 LGN cells** = 18×18 grid (retinotopic mapping)
- Each orientation column receives **same LGN input**
- Orientation selectivity comes from **pre-trained IE values** in Layer 4

### **NO Built-in Gabor Filtering:**
The model expects **already spike-encoded** input. Original preprocessing was:
1. Gabor filter bank (0°, 45°, 90°, 135°)
2. Temporal dynamics simulation
3. Spike encoding (rate-to-spike conversion)
4. Saved as `.pckl` files

### **For Your Integration:**

You need to implement a **spike encoder** on your embedded system:

```python
# Example spike encoder for camera input
import numpy as np

class CameraSpikeEncoder:
    def __init__(self, resolution=(18, 18), max_rate=100.0):
        """
        resolution: Spatial grid (matches 324 = 18×18)
        max_rate: Maximum firing rate (Hz)
        """
        self.resolution = resolution
        self.max_rate = max_rate
        self.n_neurons = resolution[0] * resolution[1]
        
    def encode_frame(self, frame, dt=0.1):
        """
        Convert camera frame to spike trains
        
        Args:
            frame: (H, W) grayscale image, values 0-255
            dt: Time step (ms), matches NEST resolution
            
        Returns:
            spike_dict: {neuron_id: [spike_times]}
        """
        # Downsample to 18×18
        downsampled = cv2.resize(frame, self.resolution, 
                                interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        intensity = downsampled.astype(float) / 255.0
        
        # Poisson spike generation
        rates = intensity * self.max_rate  # Hz
        prob_spike = rates * (dt / 1000.0)  # Probability per time step
        
        spikes = {}
        for idx in range(self.n_neurons):
            y, x = divmod(idx, self.resolution[1])
            if np.random.rand() < prob_spike[y, x]:
                spikes[idx] = [nest.GetKernelStatus('time') + dt]
                
        return spikes
```

---

## 3. ORIENTATION COLUMN CONFIGURATION

### **Indexing Structure:**

```python
# OrientedColumnV1.py, line 1
def column(setdegree, LGN):
    # setdegree ∈ {0, 45, 90, 135}
    
    # Line 199-210: Load pre-trained IE values
    file = f"./files/soma_exc_15_{setdegree}.pckl"
    SS4_soma_exc = pickle.load(file)  # Array[324] of IE values
    
    for i in range(324):
        nest.SetStatus([SS4[i]], {'soma_exc': SS4_soma_exc[i]})
```

### **Column Organization:**

```python
# Simulation_V1_pinwheel_MEGcomparison.py, lines 48-51
# 4 columns created, all receive SAME LGN input
Detector0, ...   = column(0, LGN)    # 0° preference
Detector45, ...  = column(45, LGN)   # 45° preference
Detector90, ...  = column(90, LGN)   # 90° preference
Detector135, ... = column(135, LGN)  # 135° preference
```

### **Layer 4 Internal Structure (324 SS cells):**

```python
# Lines 213-243: MNSD architecture in groups of 4
# 81 groups × 4 cells = 324 total

for j in range(0, 324, 36):     # 9 macro-groups
    for i in range(0, 18, 2):   # 9 micro-groups per macro
        # Each micro-group: 4 neurons with mutual connections
        # Cell indices: [i+j, i+j+1, i+j+18, i+j+19]
        
        # Set mutual neuromodulation (IE plasticity)
        nest.SetStatus([SS4[i+j]], {
            'stimulator': [SS4[i+j+1], SS4[i+j+18]]
        })
        # ... (3 more cells with similar cross-connectivity)
        
        # STDP connections between group members
        nest.Connect([SS4[i+j]], [SS4[i+j+1]], 
                    {"model": "stdp_synapse", 'delay': 0.1})
```

### **Extending to Other Orientations:**

```python
# To add 22.5°, 67.5°, 112.5°, 157.5° (8 total orientations)

# STEP 1: Train new IE values (see Section 4)
# STEP 2: Add columns
orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
columns = {}

for angle in orientations:
    columns[angle] = column(angle, LGN)
    
# STEP 3: Inter-column inhibition (biologically plausible)
for angle1 in orientations:
    for angle2 in orientations:
        if angle1 != angle2:
            # Connect inhibitory neurons between columns
            angle_diff = abs(angle1 - angle2)
            inhibition_strength = -50.0 * np.exp(-angle_diff/45.0)
            
            nest.Connect(columns[angle1].In4, columns[angle2].SS4,
                        {'rule': 'fixed_indegree', 'indegree': 5},
                        {"weight": inhibition_strength, "delay": 1.0})
```

---

## 4. PRETRAINED EXCITABILITY FORMAT

### **File Structure:**

```python
# soma_exc_*.pckl format inspection
import pickle

with open('./files/soma_exc_0.pckl', 'rb') as f:
    soma_exc = pickle.load(f, encoding='latin1')
    
print(type(soma_exc))    # <class 'list'> or <class 'numpy.ndarray'>
print(len(soma_exc))     # 324
print(soma_exc[0:5])     # [1.0234, 1.0891, 0.9876, 1.1234, 1.0456]
```

### **Units & Interpretation:**

```cpp
// lifl_psc_exp_ie.cpp, line 408
// IE value modulates excitatory input:
S_.V_m_ = S_.V_m_ * V_.P22_ + S_.i_syn_in_ * V_.P21in_ 
        + (S_.i_syn_ex_ * V_.P21ex_ + ...) * S_.enhancement;
        //                                     ↑
        //                            Intrinsic Excitability
```

**Range:** Typically 0.8 - 1.5
- **1.0** = baseline (no modulation)
- **> 1.0** = enhanced excitability (LTP-IE)
- **< 1.0** = reduced excitability (LTD-IE)

### **Regenerating IE Values:**

**METHOD 1 - Train from Scratch:**

```python
def train_orientation_column(angle, training_stimuli, n_trials=1000):
    """
    Train Layer 4 IE values for specific orientation
    
    Args:
        angle: Preferred orientation (degrees)
        training_stimuli: List of Gabor-filtered spike trains
        n_trials: Number of training iterations
    """
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': 0.1})
    
    # Create LGN input
    inputs = nest.Create("spike_generator", 324)
    LGN = nest.Create('parrot_neuron', 324)
    nest.Connect(inputs, LGN, 'one_to_one')
    
    # Create Layer 4 with IE enabled
    SS4 = nest.Create('lifl_psc_exp_ie', 324, {
        'lambda': 0.0005,   # IE learning rate
        'tau': 12.5,        # IE time window
        'std_mod': True     # Enable IE plasticity
    })
    nest.Connect(LGN, SS4, 'one_to_one', {"weight": 15000.0, "delay": 1.0})
    
    # Set up MNSD architecture (groups of 4)
    for j in range(0, 324, 36):
        for i in range(0, 18, 2):
            nest.SetStatus([SS4[i+j]], {
                'stimulator': [SS4[i+j+1], SS4[i+j+18]]
            })
            # ... (set up other 3 cells)
            
            nest.Connect([SS4[i+j]], [SS4[i+j+1]], 
                        {"model": "stdp_synapse", 'delay': 0.1})
            # ... (other connections)
    
    # Create soma_exc recorder
    soma_monitor = nest.Create('multimeter', 
                               params={'record_from': ['soma_exc']})
    nest.Connect(soma_monitor, SS4)
    
    # Training loop
    for trial in range(n_trials):
        # Select stimulus at preferred angle + noise
        stim_angle = angle + np.random.randn() * 5.0
        spike_train = generate_gabor_spikes(stim_angle)
        
        # Inject spikes
        for neuron_id in range(324):
            nest.SetStatus([inputs[neuron_id]], {
                'spike_times': spike_train[neuron_id]
            })
        
        # Simulate
        nest.Simulate(1000)  # 1 second per trial
        
        if trial % 100 == 0:
            print(f"Trial {trial}/{n_trials}")
    
    # Extract final IE values
    final_soma_exc = []
    for neuron in SS4:
        status = nest.GetStatus([neuron])[0]
        final_soma_exc.append(status['soma_exc'])
    
    # Save
    with open(f'soma_exc_{angle}.pckl', 'wb') as f:
        pickle.dump(final_soma_exc, f)
    
    return final_soma_exc
```

**METHOD 2 - Adaptive Modulation (Real-time):**

```python
def adaptive_ie_modulation(camera_input, target_orientation):
    """
    Dynamically adjust IE values based on camera input
    Useful for online adaptation in blindsight system
    """
    # Calculate orientation energy in camera input
    orientations = [0, 45, 90, 135]
    orientation_energy = compute_orientation_energy(camera_input)
    
    # Boost IE for neurons tuned to detected orientation
    ie_boost = {}
    for angle in orientations:
        similarity = np.exp(-((angle - target_orientation) % 180)**2 / 1000)
        ie_boost[angle] = 1.0 + 0.2 * similarity  # Max 20% boost
    
    return ie_boost
```

---

## 5. ENTRY POINTS FOR REAL-TIME INTEGRATION

### **Recommended Architecture:**

```python
# blindsight_realtime_v1.py

import nest
import numpy as np
import threading
import queue
from picamera2 import Picamera2  # RPi camera
# OR
# import serial  # Arduino camera via serial

class BlindSightV1System:
    def __init__(self, orientations=[0, 45, 90, 135]):
        self.orientations = orientations
        self.spike_queue = queue.Queue(maxsize=1000)
        self.simulation_running = False
        
        # Initialize NEST
        nest.ResetKernel()
        nest.SetKernelStatus({
            'resolution': 0.1,      # 0.1 ms time steps
            'local_num_threads': 4  # Multi-threading
        })
        
        # Create input layer
        self.inputs = nest.Create("spike_generator", 324)
        self.LGN = nest.Create('parrot_neuron', 324)
        nest.Connect(self.inputs, self.LGN, 'one_to_one')
        
        # Create orientation columns
        self.columns = {}
        for angle in orientations:
            self.columns[angle] = self._create_column(angle)
        
        # Recording devices
        self.spike_detectors = {}
        for angle in orientations:
            detector = nest.Create('spike_detector')
            nest.Connect(self.columns[angle]['Pyr23'], detector)
            self.spike_detectors[angle] = detector
    
    def _create_column(self, angle):
        """Wrapper around column() function"""
        from OrientedColumnV1 import column
        return column(angle, self.LGN)
    
    def camera_thread(self):
        """
        Separate thread for camera acquisition
        Produces spike-encoded frames → spike_queue
        """
        camera = Picamera2()
        encoder = CameraSpikeEncoder(resolution=(18, 18))
        
        while self.simulation_running:
            # Capture frame
            frame = camera.capture_array()
            
            # Convert to spikes
            spikes = encoder.encode_frame(frame)
            
            # Queue for injection
            self.spike_queue.put({
                'time': nest.GetKernelStatus('time'),
                'spikes': spikes
            })
            
            # Frame rate control (e.g., 30 FPS = 33.3 ms)
            time.sleep(0.033)
    
    def simulation_thread(self):
        """
        Main simulation thread
        Consumes spikes from queue and runs NEST
        """
        dt_simulation = 10.0  # Simulate 10 ms chunks
        
        while self.simulation_running:
            # Get recent spikes from queue
            if not self.spike_queue.empty():
                spike_data = self.spike_queue.get()
                
                # Inject spikes into NEST
                for neuron_id, spike_times in spike_data['spikes'].items():
                    nest.SetStatus([self.inputs[neuron_id]], {
                        'spike_times': spike_times
                    })
            
            # Run simulation
            nest.Simulate(dt_simulation)
            
            # Read out responses (every 100 ms)
            if nest.GetKernelStatus('time') % 100 < dt_simulation:
                self.process_responses()
    
    def process_responses(self):
        """
        Extract orientation responses and make decisions
        For blindsight: feed to higher-level decision network
        """
        responses = {}
        for angle, detector in self.spike_detectors.items():
            spikes = nest.GetStatus([detector])[0]['events']
            recent_spikes = spikes['times'][spikes['times'] > 
                           nest.GetKernelStatus('time') - 100]
            responses[angle] = len(recent_spikes)
        
        # Winner-take-all
        dominant_orientation = max(responses, key=responses.get)
        
        # Send to motor control / decision module
        self.output_decision(dominant_orientation, responses)
    
    def output_decision(self, orientation, responses):
        """
        Interface to external modules (e.g., robotic control)
        """
        # Example: Serial output to Arduino motor controller
        decision = {
            'orientation': orientation,
            'confidence': responses[orientation] / sum(responses.values()),
            'timestamp': nest.GetKernelStatus('time')
        }
        print(f"Detected: {orientation}°, Confidence: {decision['confidence']:.2f}")
    
    def run(self, duration=60000):  # 60 seconds
        """Start real-time processing"""
        self.simulation_running = True
        
        # Start threads
        cam_thread = threading.Thread(target=self.camera_thread)
        sim_thread = threading.Thread(target=self.simulation_thread)
        
        cam_thread.start()
        sim_thread.start()
        
        # Wait for duration
        time.sleep(duration / 1000.0)
        
        self.simulation_running = False
        cam_thread.join()
        sim_thread.join()


# USAGE
if __name__ == "__main__":
    system = BlindSightV1System(orientations=[0, 45, 90, 135])
    system.run(duration=60000)  # Run for 60 seconds
```

### **Key Modifications to Original Scripts:**

```python
# ORIGINAL (Simulation_V1_pinwheel_MEGcomparison.py)
exec('file = "./files/spikes_reponse_gabor_randn02_19.pckl"')
# ↓↓↓ REPLACE WITH ↓↓↓

# NEW (Real-time)
current_spikes = camera_spike_buffer.get_latest()
for neuron_id, spike_times in current_spikes.items():
    nest.SetStatus([inputs[neuron_id]], {'spike_times': spike_times})
```

---

## 6. PERFORMANCE BOTTLENECKS & OPTIMIZATION

### **Current Performance Characteristics:**

```python
# Full V1 model size:
# 4 columns × (324 SS4 + 324 Pyr23 + 81 Pyr5 + 243 Pyr6 + 195 Inh)
# = 4 × 1167 = 4668 neurons
# + ~50,000 synapses per column = ~200,000 total synapses

# Simulation speed (single core, 0.1 ms resolution):
# ~100 ms biological time per 1 second real time
# For real-time: Need 1000x speedup or model reduction
```

### **Bottleneck Analysis:**

1. **Layer 4 MNSD architecture** (Lines 213-243 in OrientedColumnV1.py)
   - 81 groups × 4 neurons = 324 neurons
   - Each group: 8 STDP synapses + cross-modulation
   - **Computational cost:** O(n²) for IE plasticity updates

2. **Recording devices:**
   - Multimeters recording at 0.1 ms → massive I/O
   - Lines 268-276: Recording V_m + I_syn_ex for all neurons

3. **Python overhead:**
   - NEST kernel is C++, but Python wrapping adds latency

### **Optimization Strategies:**

**LEVEL 1 - Reduce Recording:**

```python
# BEFORE (OrientedColumnV1.py, lines 268-276)
Multimeter = nest.Create('multimeter',
    params={'withtime': True, 'record_from': ['V_m', 'I_syn_ex'], 
            'interval': 0.1})
nest.Connect(Multimeter, Pyr23)  # All 324 neurons
nest.Connect(Multimeter, SS4)    # All 324 neurons

# AFTER (only record spike times)
Spikes = nest.Create('spike_detector')
nest.Connect(Pyr23, Spikes)  # Only output layer
# Remove membrane potential recording entirely
```

**LEVEL 2 - Simplify Layer 4:**

```python
def lightweight_column(angle, LGN):
    """
    Simplified column for embedded deployment
    Removes MNSD architecture, keeps orientation selectivity
    """
    # Layer 4: 81 neurons instead of 324 (4x reduction)
    SS4 = nest.Create('lifl_psc_exp_ie', 81, {
        'std_mod': False,  # Disable IE plasticity at runtime
        'lambda': 0.0,
        'tau': 12.5
    })
    
    # Load pre-trained IE values (downsampled)
    soma_exc = load_downsampled_ie(angle, target_size=81)
    for i, neuron in enumerate(SS4):
        nest.SetStatus([neuron], {'soma_exc': soma_exc[i]})
    
    # Simplified LGN→L4 connection (4:1 pooling)
    for i in range(81):
        lgn_sources = LGN[i*4:(i+1)*4]  # 4 LGN cells per SS4 cell
        nest.Connect(lgn_sources, [SS4[i]], 'all_to_one',
                    {"weight": 3750.0, "delay": 1.0})  # 15000/4
    
    # Simplified L4→L2/3 (no MNSD groups)
    Pyr23 = nest.Create('aeif_psc_exp_peak', 81, {...})
    nest.Connect(SS4, Pyr23, 'one_to_one', {"weight": 400.0})
    
    return SS4, Pyr23
```

**LEVEL 3 - GPU Acceleration:**

NEST doesn't natively support GPU, but you can:

```python
# Option A: Use CARLsim (GPU spiking simulator)
# - Reimplement lifl_psc_exp_ie in CUDA
# - 10-100x speedup possible

# Option B: Hybrid CPU-GPU pipeline
import cupy as cp  # GPU NumPy

def gpu_orientation_pooling(spike_counts_per_column):
    """
    Offload orientation decision to GPU
    Useful if running multiple trials in parallel
    """
    spike_counts_gpu = cp.array(spike_counts_per_column)
    orientation_energy = cp.fft.fft2(spike_counts_gpu)
    dominant = cp.argmax(cp.abs(orientation_energy))
    return cp.asnumpy(dominant)
```

**LEVEL 4 - On-Device Processing:**

```python
# RPi + Arduino Split:

# Arduino (low-power):
# - Camera capture (320×240 @ 30 FPS)
# - Gabor filtering (integer arithmetic)
# - Spike encoding (Poisson threshold)
# → Serial output: neuron_id, spike_time

# Raspberry Pi 4 (4-core ARM):
# - NEST simulation (simplified 81-neuron columns)
# - Orientation decision (CPU)
# - Motor control output

# Example Arduino spike encoder:
/*
void loop() {
    camera.readFrame(frame);
    for (int i = 0; i < 324; i++) {
        int intensity = downsample_receptive_field(frame, i);
        if (poisson_spike(intensity)) {
            Serial.print(i); Serial.print(",");
            Serial.print(millis()); Serial.print("\n");
        }
    }
}
*/
```

### **Benchmarking:**

```python
import time

def benchmark_v1_model(n_columns=4, n_neurons_per_column=324, 
                      simulation_time=1000):
    """
    Test simulation speed
    """
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': 0.1})
    
    # Create model
    inputs = nest.Create("spike_generator", n_neurons_per_column)
    LGN = nest.Create('parrot_neuron', n_neurons_per_column)
    nest.Connect(inputs, LGN, 'one_to_one')
    
    columns = []
    for angle in [0, 45, 90, 135][:n_columns]:
        col = column(angle, LGN)
        columns.append(col)
    
    # Inject random spikes
    for i in range(n_neurons_per_column):
        spikes = np.random.uniform(0, simulation_time, 50)
        nest.SetStatus([inputs[i]], {'spike_times': spikes.tolist()})
    
    # Time simulation
    start = time.time()
    nest.Simulate(simulation_time)
    elapsed = time.time() - start
    
    speedup = simulation_time / elapsed  # Biological time / real time
    
    print(f"Columns: {n_columns}")
    print(f"Neurons: {n_columns * n_neurons_per_column}")
    print(f"Simulated: {simulation_time} ms")
    print(f"Real time: {elapsed:.2f} s")
    print(f"Speedup: {speedup:.1f}x")
    print(f"Real-time capable: {speedup >= 1.0}")
    
    return speedup

# Run benchmarks
benchmark_v1_model(n_columns=1, simulation_time=1000)  # Single column
benchmark_v1_model(n_columns=4, simulation_time=1000)  # Full pinwheel
```

---

## 7. RECOMMENDED INTEGRATION PIPELINE

### **Phase 1: Offline Development**

```
RPi Camera → Record Video → Preprocess (Python) → 
  Generate Spikes → Save .pckl → Test with existing V1 model
```

### **Phase 2: Near-Real-Time**

```
RPi Camera → Frame Buffer → Spike Encoder (NumPy) →
  Batch Inject → NEST (10 ms chunks) → Orientation Decision
```

### **Phase 3: Embedded Real-Time**

```
Arduino Camera → Serial Spikes → RPi NEST (Simplified) →
  Orientation Output → Motor Control / Higher Decision
```

### **Complete Example Pipeline:**

```python
# complete_blindsight_pipeline.py

import nest
import numpy as np
import cv2
import serial
from queue import Queue
from threading import Thread

class BlindSightPipeline:
    def __init__(self):
        # Camera interface
        self.serial_port = serial.Serial('/dev/ttyACM0', 115200)
        self.spike_queue = Queue()
        
        # Initialize V1 model
        self.init_nest_model()
        
    def init_nest_model(self):
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': 0.1})
        
        # Lightweight model (81 neurons per column)
        self.inputs = nest.Create("spike_generator", 81)
        self.LGN = nest.Create('parrot_neuron', 81)
        nest.Connect(self.inputs, self.LGN, 'one_to_one')
        
        self.columns = {}
        for angle in [0, 45, 90, 135]:
            self.columns[angle] = lightweight_column(angle, self.LGN)
    
    def arduino_listener(self):
        """Read spikes from Arduino via serial"""
        while True:
            line = self.serial_port.readline().decode().strip()
            if ',' in line:
                neuron_id, spike_time = map(int, line.split(','))
                self.spike_queue.put((neuron_id, spike_time))
    
    def nest_simulator(self):
        """Run NEST simulation"""
        while True:
            # Collect spikes from queue
            spike_buffer = {}
            while not self.spike_queue.empty():
                neuron_id, spike_time = self.spike_queue.get()
                if neuron_id not in spike_buffer:
                    spike_buffer[neuron_id] = []
                spike_buffer[neuron_id].append(spike_time)
            
            # Inject into NEST
            for neuron_id, times in spike_buffer.items():
                nest.SetStatus([self.inputs[neuron_id]], 
                              {'spike_times': times})
            
            # Simulate 10 ms
            nest.Simulate(10.0)
            
            # Process responses
            self.make_decision()
    
    def make_decision(self):
        """Orientation detection and output"""
        # Read spike detectors
        responses = {}
        for angle in [0, 45, 90, 135]:
            spikes = nest.GetStatus([self.columns[angle]['detector']])[0]['events']
            responses[angle] = len(spikes['times'])
        
        # Winner-take-all
        if sum(responses.values()) > 0:
            winner = max(responses, key=responses.get)
            confidence = responses[winner] / sum(responses.values())
            
            # Output decision
            self.serial_port.write(f"ORIENT:{winner},{confidence}\n".encode())
    
    def run(self):
        # Start threads
        Thread(target=self.arduino_listener, daemon=True).start()
        Thread(target=self.nest_simulator, daemon=True).start()
        
        # Keep alive
        while True:
            time.sleep(1)

if __name__ == "__main__":
    pipeline = BlindSightPipeline()
    pipeline.run()
```

---

## SUMMARY OF KEY INTEGRATION POINTS

| Component | File | Lines | Modification Required |
|-----------|------|-------|----------------------|
| **Spike Input** | `Simulation_V1_pinwheel_MEGcomparison.py` | 38-45, 59-70 | Replace `.pckl` loading with real-time queue |
| **LGN Layer** | `Simulation_V1_pinwheel_MEGcomparison.py` | 44 | `parrot_neuron` stays same, feed from camera |
| **Layer 4 Connection** | `OrientedColumnV1.py` | 48-52 | Keep 1:1 mapping, adjust weight if needed |
| **IE Values** | `OrientedColumnV1.py` | 199-210 | Pre-load trained values, or disable plasticity |
| **Column Creation** | `OrientedColumnV1.py` | 1-278 | Simplify to `lightweight_column()` for speed |
| **Recording** | `OrientedColumnV1.py` | 268-276 | Remove multimeters, keep only spike detectors |
| **Decision Output** | `Simulation_V1_pinwheel_MEGcomparison.py` | 75-83 | Add real-time decision function |

---

## NEXT STEPS

1. **Test Current Model**: Run `Simulation_V1_pinwheel_MEGcomparison.py` to verify installation
2. **Create Spike Encoder**: Implement `CameraSpikeEncoder` for your camera
3. **Benchmark**: Use `benchmark_v1_model()` to assess speed
4. **Simplify**: If too slow, switch to `lightweight_column()`
5. **Integrate Hardware**: Implement `BlindSightPipeline` with your camera
6. **Tune Parameters**: Adjust weights, IE values for your visual environment

**Contact**: alejandro.santos@ctb.upm.es (original author)

