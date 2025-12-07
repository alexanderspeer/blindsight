"""
Configuration for Computational V1 Vision Pipeline
"""

# ==================== VIDEO STREAM SETTINGS ====================
VIDEO_CONFIG = {
    'pi_ip': '10.207.70.178',
    'port': 5001,
    'width': 320,           # Actual Pi camera resolution
    'height': 240,          # Actual Pi camera resolution
    'fps': 15,              # Actual Pi framerate
}

# ==================== SPATIAL GRID SETTINGS ====================
GRID_CONFIG = {
    'n_neurons': 144,           # Total number per layer (12x12 grid) - OPTIMIZED
    'grid_rows': 12,            # Reduced from 18 for speed
    'grid_cols': 12,            # Reduced from 18 for speed
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
    'threshold': 0.5,  # INCREASED from 0.1 - only strong features should spike!
}

# ==================== V1 ARCHITECTURE (matches MDPI2021 exactly) ====================
V1_ARCHITECTURE = {
    # Number of neurons per layer (per orientation column)
    # Note: layer_4_ss and layer_23_pyr use GRID_CONFIG['n_neurons'] = 144
    'layer_4_ss': 144,          # Spiny Stellate cells (input layer) - UPDATED for 12x12 grid
    'layer_4_inh': 65,          # Inhibitory interneurons
    'layer_23_pyr': 144,        # Pyramidal cells - UPDATED for 12x12 grid
    'layer_23_inh': 65,         # Inhibitory interneurons
    'layer_5_pyr': 81,          # Pyramidal cells
    'layer_5_inh': 16,          # Inhibitory interneurons
    'layer_6_pyr': 243,         # Pyramidal cells
    'layer_6_inh': 49,          # Inhibitory interneurons
    
    # Connection weights (from MDPI2021 model)
    'lgn_to_ss4_weight': 15000.0,
    'lateral_weight': 0.0,        # DISABLED FOR DEBUG - was 100.0 (testing if recurrence causes runaway)
    'inhibitory_weight': 0.0,     # DISABLED FOR DEBUG - was -100.0
    'feedforward_weight': 10.0,  # Keep feedforward enabled
    
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
    
    # Simulation parameters - OPTIMIZED FOR REAL-TIME
    'dt': 0.5,                  # Time step (ms) - increased from 0.1 for 5x speedup
    'warmup_time_ms': 50,       # Reduced from 400 for 8x speedup
    'stimulus_time_ms': 100,    # Increased to 100ms to allow more neuron activity
    
    # Noise (Poisson background activity from MDPI2021)
    # DISABLED FOR DEBUGGING - was causing constant 1200+ Hz firing in all layers
    'poisson_rate_l23': 0.0,
    'poisson_rate_l5': 0.0,
    'poisson_rate_l6': 0.0,
    'poisson_rate_inh': 0.0,
    'poisson_weight': 5.0,
}

# ==================== PROCESSING SETTINGS ====================
PROCESSING_CONFIG = {
    'downsample_frame': False,  # DISABLED - was upscaling 320x240 to 640x360, destroying edges!
    'downsample_width': 320,    # Keep original resolution
    'downsample_height': 240,   # Keep original resolution
    'normalize_contrast': True,
    'gaussian_blur_kernel': 3,
}

# ==================== VISUALIZATION SETTINGS ====================
VISUALIZATION_CONFIG = {
    'display_raw': True,
    'display_preprocessed': False,  # Disabled - Pi video already small
    'display_spikes': True,
    'display_v1_output': True,
    'combined_display': True,       # Show all in one panel
    'fullscreen': False,            # Set to True for fullscreen mode
    'update_interval_seconds': 5.0, # Update visualizations every 5 seconds
    
    'window_names': {
        'raw': 'Raw Video Stream',
        'gabor': 'Gabor Features (4 Orientations)',
        'spikes': 'Input Spike Trains',
        'v1': 'V1 Output (Orientation Map)',
        'layers': 'V1 Layer Activity',
        'combined': 'V1 Vision Pipeline - Real-Time'
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

# ==================== DEBUG SETTINGS ====================
DEBUG_CONFIG = {
    'enabled': True,                    # Master debug switch
    'print_every_n_frames': 1,          # Print debug info every N frames
    'show_gabor_stats': True,           # Gabor filter response statistics
    'show_spike_stats': True,           # Spike encoding statistics
    'show_v1_layer_stats': True,        # V1 layer firing rate statistics
    'show_decoder_stats': True,         # Decoder output statistics
    'show_array_shapes': True,          # Show shapes of all arrays
    'show_distributions': True,         # Show min/max/mean/std of arrays
    'check_for_nans': True,             # Check for NaN values
    'check_for_zeros': True,            # Check if arrays are all zeros
}

