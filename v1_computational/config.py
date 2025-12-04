"""
Configuration for Computational V1 Vision Pipeline
"""

# ==================== VIDEO STREAM SETTINGS ====================
VIDEO_CONFIG = {
    'pi_ip': '10.207.70.178',
    'port': 5001,
    'width': 1280,
    'height': 720,
    'fps': 30,
}

# ==================== SPATIAL GRID SETTINGS ====================
GRID_CONFIG = {
    'n_neurons': 324,           # Total number per layer (18x18 grid)
    'grid_rows': 18,
    'grid_cols': 18,
    'receptive_field_size': 64,
    'overlap': 0.5,
}

# ==================== GABOR FILTER SETTINGS ====================
GABOR_CONFIG = {
    'orientations': [0, 45, 90, 135],  # Matches V1 model
    'wavelength': 10.0,
    'sigma': 5.0,
    'gamma': 0.5,
    'psi': 0,
    'kernel_size': 31,
}

# ==================== SPIKE ENCODING SETTINGS ====================
SPIKE_CONFIG = {
    'encoding_type': 'latency',
    'max_spike_rate': 200.0,
    'min_spike_rate': 10.0,
    'spike_window_ms': 150.0,
    'spike_start_ms': 50.0,
    'min_latency_ms': 43.0,
    'max_latency_ms': 200.0,
    'jitter_ms': 0.3,
    'threshold': 0.1,
}

# ==================== V1 ARCHITECTURE (matches MDPI2021 exactly) ====================
V1_ARCHITECTURE = {
    # Number of neurons per layer (per orientation column)
    'layer_4_ss': 324,          # Spiny Stellate cells (input layer)
    'layer_4_inh': 65,          # Inhibitory interneurons
    'layer_23_pyr': 324,        # Pyramidal cells
    'layer_23_inh': 65,         # Inhibitory interneurons
    'layer_5_pyr': 81,          # Pyramidal cells
    'layer_5_inh': 16,          # Inhibitory interneurons
    'layer_6_pyr': 243,         # Pyramidal cells
    'layer_6_inh': 49,          # Inhibitory interneurons
    
    # Connection weights (from MDPI2021 model)
    'lgn_to_ss4_weight': 15000.0,
    'lateral_weight': 100.0,
    'inhibitory_weight': -100.0,
    'feedforward_weight': 100.0,
    
    # Connection probabilities
    'layer_23_recurrent_indegree': 36,
    'layer_5_recurrent_indegree': 10,
    'layer_6_recurrent_indegree': 20,
    
    # Neuron parameters (from MDPI2021)
    'v_rest': -65.0,            # mV
    'v_threshold': -50.0,       # mV
    'v_reset': -65.0,           # mV
    'tau_membrane': 10.0,       # ms
    'tau_syn_ex': 2.0,          # ms
    'tau_syn_in': 2.0,          # ms
    'refractory_period': 2.0,   # ms
    
    # Simulation parameters
    'dt': 0.1,                  # Time step (ms)
    'warmup_time_ms': 400,
    'stimulus_time_ms': 200,
    
    # Noise (Poisson background activity from MDPI2021)
    'poisson_rate_l23': 1721500.0,
    'poisson_rate_l5': 1740000.0,
    'poisson_rate_l6': 1700000.0,
    'poisson_rate_inh': 1750000.0,
    'poisson_weight': 5.0,
}

# ==================== PROCESSING SETTINGS ====================
PROCESSING_CONFIG = {
    'downsample_frame': True,
    'downsample_width': 640,
    'downsample_height': 360,
    'normalize_contrast': True,
    'gaussian_blur_kernel': 3,
}

# ==================== VISUALIZATION SETTINGS ====================
VISUALIZATION_CONFIG = {
    'display_raw': True,
    'display_preprocessed': True,
    'display_spikes': True,
    'display_v1_output': True,
    
    'window_names': {
        'raw': 'Raw Video Stream',
        'gabor': 'Gabor Features (4 Orientations)',
        'spikes': 'Input Spike Trains',
        'v1': 'V1 Output (Orientation Map)',
        'layers': 'V1 Layer Activity'
    },
    
    'spike_plot_duration_ms': 250,
    'update_interval_frames': 1,
    
    'orientation_colors': {
        0: (255, 0, 0),      # Red
        45: (0, 255, 0),     # Green
        90: (0, 0, 255),     # Blue
        135: (255, 255, 0),  # Yellow
    }
}

# ==================== PERFORMANCE SETTINGS ====================
PERFORMANCE_CONFIG = {
    'show_fps': True,
    'profile_stages': True,
    'save_outputs': False,
    'output_dir': './outputs',
}

