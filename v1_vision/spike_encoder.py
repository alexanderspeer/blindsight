"""
Spike Encoder
Converts Gabor feature responses to spike trains for V1 model input
"""

import numpy as np
from config import SPIKE_CONFIG


class SpikeEncoder:
    """
    Encodes visual features as spike trains
    Supports multiple encoding schemes: rate coding, latency coding, or hybrid
    """
    
    def __init__(self, encoding_type=None):
        self.encoding_type = encoding_type or SPIKE_CONFIG['encoding_type']
        self.n_neurons = 324
        
    def encode(self, features, current_time_ms=0):
        """
        Convert Gabor feature responses to spike trains
        
        Args:
            features: dict from GaborFeatureExtractor.extract_features()
                - 'responses': (324, 4) array of responses
                - 'max_response': (324,) array of max response per neuron
            current_time_ms: Current simulation time offset
            
        Returns:
            spike_trains: dict with 'senders' and 'times' arrays
                - senders: array of neuron IDs
                - times: array of spike times (ms)
        """
        if self.encoding_type == 'rate':
            return self._rate_coding(features, current_time_ms)
        elif self.encoding_type == 'latency':
            return self._latency_coding(features, current_time_ms)
        elif self.encoding_type == 'hybrid':
            return self._hybrid_coding(features, current_time_ms)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def _rate_coding(self, features, current_time_ms):
        """
        Rate coding: Higher responses generate more spikes
        """
        max_responses = features['max_response']
        
        spike_senders = []
        spike_times = []
        
        spike_window = SPIKE_CONFIG['spike_window_ms']
        spike_start = current_time_ms + SPIKE_CONFIG['spike_start_ms']
        min_rate = SPIKE_CONFIG['min_spike_rate']
        max_rate = SPIKE_CONFIG['max_spike_rate']
        threshold = SPIKE_CONFIG['threshold']
        jitter = SPIKE_CONFIG['jitter_ms']
        
        for neuron_id in range(self.n_neurons):
            response = max_responses[neuron_id]
            
            # Skip if below threshold
            if response < threshold:
                continue
            
            # Map response to spike rate (linear mapping)
            # Normalize response to [0, 1]
            response_norm = np.clip(response, 0, 1)
            
            # Calculate firing rate
            rate_hz = min_rate + (max_rate - min_rate) * response_norm
            
            # Convert rate to number of spikes in window
            n_spikes = int((rate_hz / 1000.0) * spike_window)
            n_spikes = max(1, n_spikes)  # At least 1 spike if above threshold
            
            # Generate spike times uniformly in window
            spike_intervals = spike_window / n_spikes
            
            for spike_idx in range(n_spikes):
                spike_time = spike_start + (spike_idx * spike_intervals)
                
                # Add jitter
                spike_time += np.random.randn() * jitter
                
                spike_senders.append(neuron_id)
                spike_times.append(spike_time)
        
        return {
            'senders': np.array(spike_senders, dtype=np.int32),
            'times': np.array(spike_times, dtype=np.float32)
        }
    
    def _latency_coding(self, features, current_time_ms):
        """
        Latency coding: Higher responses generate earlier spikes
        More biologically plausible and matches V1 model's spike latency mechanism
        """
        max_responses = features['max_response']
        
        spike_senders = []
        spike_times = []
        
        min_latency = SPIKE_CONFIG['min_latency_ms']
        max_latency = SPIKE_CONFIG['max_latency_ms']
        threshold = SPIKE_CONFIG['threshold']
        jitter = SPIKE_CONFIG['jitter_ms']
        
        for neuron_id in range(self.n_neurons):
            response = max_responses[neuron_id]
            
            # Skip if below threshold
            if response < threshold:
                continue
            
            # Map response to latency (inverse relationship)
            # Higher response -> shorter latency (earlier spike)
            response_norm = np.clip(response, 0, 1)
            
            # Inverse mapping: strong response -> min_latency
            latency = max_latency - (max_latency - min_latency) * response_norm
            
            spike_time = current_time_ms + latency
            
            # Add jitter
            spike_time += np.random.randn() * jitter
            
            spike_senders.append(neuron_id)
            spike_times.append(spike_time)
        
        return {
            'senders': np.array(spike_senders, dtype=np.int32),
            'times': np.array(spike_times, dtype=np.float32)
        }
    
    def _hybrid_coding(self, features, current_time_ms):
        """
        Hybrid coding: Combines latency and rate
        Strong responses: early spike + multiple spikes
        Weak responses: late spike + few spikes
        """
        max_responses = features['max_response']
        
        spike_senders = []
        spike_times = []
        
        min_latency = SPIKE_CONFIG['min_latency_ms']
        max_latency = SPIKE_CONFIG['max_latency_ms']
        min_rate = SPIKE_CONFIG['min_spike_rate']
        max_rate = SPIKE_CONFIG['max_spike_rate']
        spike_window = SPIKE_CONFIG['spike_window_ms']
        threshold = SPIKE_CONFIG['threshold']
        jitter = SPIKE_CONFIG['jitter_ms']
        
        for neuron_id in range(self.n_neurons):
            response = max_responses[neuron_id]
            
            if response < threshold:
                continue
            
            response_norm = np.clip(response, 0, 1)
            
            # First spike timing (latency coded)
            first_spike_latency = max_latency - (max_latency - min_latency) * response_norm
            first_spike_time = current_time_ms + first_spike_latency + np.random.randn() * jitter
            
            spike_senders.append(neuron_id)
            spike_times.append(first_spike_time)
            
            # Additional spikes (rate coded)
            rate_hz = min_rate + (max_rate - min_rate) * response_norm
            n_additional_spikes = int((rate_hz / 1000.0) * spike_window * 0.5)  # Fewer additional spikes
            
            if n_additional_spikes > 0:
                # Subsequent spikes after first spike
                for spike_idx in range(n_additional_spikes):
                    isi = (1000.0 / rate_hz)  # Inter-spike interval
                    spike_time = first_spike_time + (spike_idx + 1) * isi + np.random.randn() * jitter
                    
                    # Don't exceed window
                    if spike_time < current_time_ms + max_latency:
                        spike_senders.append(neuron_id)
                        spike_times.append(spike_time)
        
        return {
            'senders': np.array(spike_senders, dtype=np.int32),
            'times': np.array(spike_times, dtype=np.float32)
        }
    
    def get_spike_statistics(self, spike_trains):
        """
        Calculate statistics about generated spike trains
        Useful for debugging and visualization
        """
        if len(spike_trains['senders']) == 0:
            return {
                'n_spikes': 0,
                'n_active_neurons': 0,
                'mean_rate': 0,
                'time_range': (0, 0)
            }
        
        n_spikes = len(spike_trains['senders'])
        n_active_neurons = len(np.unique(spike_trains['senders']))
        
        time_range = (np.min(spike_trains['times']), np.max(spike_trains['times']))
        duration_ms = time_range[1] - time_range[0]
        
        if duration_ms > 0:
            mean_rate = (n_spikes / n_active_neurons / duration_ms) * 1000.0
        else:
            mean_rate = 0
        
        return {
            'n_spikes': n_spikes,
            'n_active_neurons': n_active_neurons,
            'mean_rate': mean_rate,
            'time_range': time_range
        }
    
    def format_for_nest(self, spike_trains, neuron_id_offset=0):
        """
        Format spike trains for NEST simulator
        
        Args:
            spike_trains: dict with 'senders' and 'times'
            neuron_id_offset: Offset to add to neuron IDs (if needed for NEST GIDs)
            
        Returns:
            List of dicts, one per neuron: [{'neuron_id': X, 'spike_times': [t1, t2, ...]}, ...]
        """
        nest_format = []
        
        if len(spike_trains['senders']) == 0:
            return nest_format
        
        unique_neurons = np.unique(spike_trains['senders'])
        
        for neuron_id in unique_neurons:
            # Get all spike times for this neuron
            mask = spike_trains['senders'] == neuron_id
            spike_times = spike_trains['times'][mask]
            
            # Sort spike times
            spike_times = np.sort(spike_times)
            
            nest_format.append({
                'neuron_id': int(neuron_id) + neuron_id_offset,
                'spike_times': spike_times.tolist()
            })
        
        return nest_format

