"""
Configuration for V1 Vision Pipeline
All parameters for video streaming, feature extraction, spike encoding, and visualization
"""

# ==================== VIDEO STREAM SETTINGS ====================
VIDEO_CONFIG = {
    'pi_ip': '10.207.55.64',
    'port': 5001,
    'width': 1280,
    'height': 720,
    'fps': 30,
}

# ==================== SPATIAL GRID SETTINGS ====================
# The V1 model expects 324 neurons organized spatially
GRID_CONFIG = {
    'n_neurons': 324,           # Total number of ganglion cells
    'grid_rows': 18,            # 18x18 = 324
    'grid_cols': 18,
    'receptive_field_size': 64,  # Pixels per receptive field (approx)
    'overlap': 0.5,             # Overlap between adjacent receptive fields
}

# ==================== GABOR FILTER SETTINGS ====================
GABOR_CONFIG = {
    'orientations': [0, 45, 90, 135],  # Degrees - matches V1 model columns
    'wavelength': 10.0,         # Wavelength of sinusoidal factor (pixels)
    'sigma': 5.0,               # Gaussian envelope standard deviation
    'gamma': 0.5,               # Spatial aspect ratio
    'psi': 0,                   # Phase offset
    'kernel_size': 31,          # Size of Gabor kernel (should be odd)
}

# ==================== SPIKE ENCODING SETTINGS ====================
SPIKE_CONFIG = {
    'encoding_type': 'latency',  # 'rate', 'latency', or 'hybrid'
    
    # Rate coding parameters
    'max_spike_rate': 200.0,    # Hz
    'min_spike_rate': 10.0,     # Hz
    'spike_window_ms': 150.0,   # Time window for spiking (ms)
    'spike_start_ms': 50.0,     # When spikes start (ms)
    
    # Latency coding parameters
    'min_latency_ms': 43.0,     # Minimum spike latency (from example data)
    'max_latency_ms': 200.0,    # Maximum spike latency
    
    # Noise and jitter
    'jitter_ms': 0.3,           # Random jitter to add to spike times (ms)
    'threshold': 0.1,           # Minimum Gabor response to generate spikes
}

# ==================== V1 MODEL SETTINGS ====================
V1_CONFIG = {
    'nest_resolution': 0.1,     # NEST time resolution (ms)
    'warmup_time_ms': 400,      # Initialization time before stimulus
    'stimulus_time_ms': 200,    # Duration of stimulus presentation
    'columns': [0, 45, 90, 135], # Orientation columns to create
    
    # Which layers to record from
    'record_layers': {
        'layer_23': True,       # Primary output layer for orientation selectivity
        'layer_4': True,        # Input layer (spiny stellate cells)
        'layer_5': True,
        'layer_6': True,
    },
    
    # Path to V1 model files
    'model_path': '../MDPI2021/Examples/V1 Oriented Columns comapred with MEG',
}

# ==================== VISUALIZATION SETTINGS ====================
VISUALIZATION_CONFIG = {
    'display_raw': True,            # Show raw video from Pi
    'display_preprocessed': True,   # Show Gabor-filtered video
    'display_spikes': True,         # Show spike raster plots
    'display_v1_output': True,      # Show V1 reconstruction
    
    'window_names': {
        'raw': 'Raw Video Stream',
        'gabor': 'Gabor Features (4 Orientations)',
        'spikes': 'Spike Train Raster',
        'v1': 'V1 Output (Orientation Map)',
    },
    
    'spike_plot_duration_ms': 250,  # Duration to show in spike raster
    'update_interval_frames': 1,    # Process every N frames (1 = every frame)
    
    # Color maps for different orientations
    'orientation_colors': {
        0: (255, 0, 0),      # Red for 0°
        45: (0, 255, 0),     # Green for 45°
        90: (0, 0, 255),     # Blue for 90°
        135: (255, 255, 0),  # Yellow for 135°
    }
}

# ==================== PROCESSING SETTINGS ====================
PROCESSING_CONFIG = {
    'downsample_frame': True,       # Downsample before processing
    'downsample_width': 640,        # Smaller size for faster processing
    'downsample_height': 360,
    'normalize_contrast': True,     # Normalize image contrast
    'gaussian_blur_kernel': 3,      # Pre-smoothing (0 to disable)
}

# ==================== PERFORMANCE SETTINGS ====================
PERFORMANCE_CONFIG = {
    'show_fps': True,               # Display FPS counter
    'profile_stages': True,         # Time each processing stage
    'save_outputs': False,          # Save processed frames to disk
    'output_dir': './outputs',      # Where to save outputs
}

