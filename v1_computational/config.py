"""
Configuration for Computational V1 Vision Pipeline
"""

# VIDEO STREAM SETTINGS
VIDEO_CONFIG = {
    'pi_ip': '10.207.70.178',
    'port': 5001,
    'width': 320,
    'height': 240,
    'fps': 15,
}

# SPATIAL GRID SETTINGS
GRID_CONFIG = {
    'n_neurons': 144,
    'grid_rows': 12,
    'grid_cols': 12,
    'receptive_field_size': 20,
    'overlap': 0.5,
}

# GABOR FILTER SETTINGS
GABOR_CONFIG = {
    'orientations': [0, 45, 90, 135],
    'wavelength': 10.0,
    'sigma': 5.0,
    'gamma': 0.5,
    'psi': 0,
    'kernel_size': 31,
}

# SPIKE ENCODING SETTINGS
SPIKE_CONFIG = {
    'encoding_type': 'latency',
    'max_spike_rate': 200.0,
    'min_spike_rate': 10.0,
    'spike_window_ms': 150.0,
    'spike_start_ms': 0.0,
    'min_latency_ms': 0.0,
    'max_latency_ms': 100.0,
    'jitter_ms': 0.3,
    'threshold': 0.5,
}

# V1 ARCHITECTURE
V1_ARCHITECTURE = {
    'layer_4_ss': 144,
    'layer_4_inh': 65,
    'layer_23_pyr': 144,
    'layer_23_inh': 65,
    'layer_5_pyr': 81,
    'layer_5_inh': 16,
    'layer_6_pyr': 243,
    'layer_6_inh': 49,
    
    'lgn_to_ss4_weight': 5000.0,
    'lateral_weight': 0.0,
    'inhibitory_weight': 0.0,
    
    'weight_L4_to_L23': 120.0,
    'weight_L23_to_L5': 150.0,
    'weight_L5_to_L6': 150.0,
    
    'V1_L4_TO_L23': 120.0,
    'V1_L23_TO_L5': 150.0,
    'V1_L5_TO_L6': 150.0, 
    
    'layer_23_recurrent_indegree': 36,
    'layer_5_recurrent_indegree': 10,
    'layer_6_recurrent_indegree': 20,
    
    'v_rest': -65.0,
    'v_threshold': -50.0,
    'v_reset': -65.0,
    'tau_membrane': 10.0,
    'tau_syn_ex': 2.0,
    'tau_syn_in': 2.0,
    'refractory_period': 2.0,
    
    'L23_v_threshold': -55.0,
    'L23_tau_membrane': 25.0,
    'L23_bias_current': 20.0,
    
    'dt': 0.5,
    'warmup_time_ms': 50,
    'stimulus_time_ms': 100,
    
    'poisson_rate_l23': 0.0,
    'poisson_rate_l5': 0.0,
    'poisson_rate_l6': 0.0,
    'poisson_rate_inh': 0.0,
    'poisson_weight': 5.0,
}

# PROCESSING SETTINGS
PROCESSING_CONFIG = {
    'downsample_frame': False,
    'downsample_width': 320,
    'downsample_height': 240,
    'normalize_contrast': True,
    'gaussian_blur_kernel': 3,
}

# VISUALIZATION SETTINGS
VISUALIZATION_CONFIG = {
    'display_raw': True,
    'display_preprocessed': False,
    'display_spikes': True,
    'display_v1_output': True,
    'combined_display': True,
    'fullscreen': False,
    'update_interval_seconds': 5.0,
    
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
        0: (255, 0, 0),
        45: (0, 255, 0),
        90: (0, 0, 255),
        135: (255, 255, 0),
    }
}

# PERFORMANCE SETTINGS
PERFORMANCE_CONFIG = {
    'show_fps': True,
    'profile_stages': True,
    'save_outputs': False,
    'output_dir': './outputs',
}

# DEBUG SETTINGS
DEBUG_CONFIG = {
    'enabled': True,
    'print_every_n_frames': 1,
    'show_gabor_stats': True,
    'show_spike_stats': True,
    'show_v1_layer_stats': True,
    'show_decoder_stats': True,
    'show_array_shapes': True,
    'show_distributions': True,
    'check_for_nans': True,
    'check_for_zeros': True,
    'show_synaptic_currents': False,
}

