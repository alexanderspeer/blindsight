"""
Spike Encoder
Converts visual features into spike trains for V1 model input
"""

import numpy as np
from config import SPIKE_CONFIG, GRID_CONFIG


class SpikeEncoder:
    """
    Encodes visual features into spike trains
    Uses latency coding: stronger features -> earlier spikes
    Matches the encoding used in MDPI2021 (spikes_reponse_gabor_randn02_19.pckl)
    """
    
    def __init__(self):
        """Initialize spike encoder"""
        self.encoding_type = SPIKE_CONFIG['encoding_type']
        self.min_latency = SPIKE_CONFIG['min_latency_ms']
        self.max_latency = SPIKE_CONFIG['max_latency_ms']
        self.spike_start = SPIKE_CONFIG['spike_start_ms']
        self.spike_window = SPIKE_CONFIG['spike_window_ms']
        self.jitter = SPIKE_CONFIG['jitter_ms']
        self.threshold = SPIKE_CONFIG['threshold']
        self.n_neurons = GRID_CONFIG['n_neurons']
        
        print("Spike encoder initialized")
        print(f"   Encoding: {self.encoding_type}")
        print(f"   Latency range: {self.min_latency}-{self.max_latency} ms")
    
    def encode_features_to_spikes(self, features):
        """
        Convert Gabor features to spike trains
        
        Args:
            features: Dict mapping orientation -> 18x18 feature grid
            
        Returns:
            Dict mapping orientation -> spike data
                spike data = {'neuron_ids': array, 'spike_times': array}
        """
        spike_trains = {}
        
        for orientation, feature_grid in features.items():
            feature_array = feature_grid.flatten()
            
            feature_array = np.clip(feature_array, 0, None)
            if feature_array.max() > 0:
                feature_array = feature_array / feature_array.max()
            
            if self.encoding_type == 'latency':
                spike_data = self._latency_encoding(feature_array)
            elif self.encoding_type == 'rate':
                spike_data = self._rate_encoding(feature_array)
            else:
                spike_data = self._latency_encoding(feature_array)
            
            spike_trains[orientation] = spike_data
        
        return spike_trains
    
    def _latency_encoding(self, feature_array):
        """
        Latency coding: stronger features -> earlier spikes
        Matches MDPI2021 encoding strategy
        
        Args:
            feature_array: Array of feature strengths (324 neurons)
            
        Returns:
            Dict with 'neuron_ids' and 'spike_times'
        """
        neuron_ids = []
        spike_times = []
        
        for neuron_idx in range(len(feature_array)):
            feature_strength = feature_array[neuron_idx]
            
            if feature_strength > self.threshold:
                latency = self.max_latency - (feature_strength * (self.max_latency - self.min_latency))
                latency += np.random.randn() * self.jitter
                latency = np.clip(latency, self.min_latency, self.max_latency)
                spike_time = self.spike_start + latency
                
                neuron_ids.append(neuron_idx)
                spike_times.append(spike_time)
        
        return {
            'neuron_ids': np.array(neuron_ids, dtype=int),
            'spike_times': np.array(spike_times, dtype=float)
        }
    
    def _rate_encoding(self, feature_array):
        """
        Rate coding: stronger features -> more spikes
        
        Args:
            feature_array: Array of feature strengths (324 neurons)
            
        Returns:
            Dict with 'neuron_ids' and 'spike_times'
        """
        neuron_ids = []
        spike_times = []
        
        dt = 1.0
        
        for neuron_idx in range(len(feature_array)):
            feature_strength = feature_array[neuron_idx]
            
            if feature_strength > self.threshold:
                firing_rate = (SPIKE_CONFIG['min_spike_rate'] + 
                              feature_strength * (SPIKE_CONFIG['max_spike_rate'] - 
                                                 SPIKE_CONFIG['min_spike_rate']))
                
                t = self.spike_start
                while t < self.spike_start + self.spike_window:
                    if np.random.rand() < (firing_rate / 1000.0) * dt:
                        neuron_ids.append(neuron_idx)
                        spike_times.append(t + np.random.randn() * self.jitter)
                    t += dt
        
        return {
            'neuron_ids': np.array(neuron_ids, dtype=int),
            'spike_times': np.array(spike_times, dtype=float)
        }
    
    def visualize_spike_trains(self, spike_trains, frame_shape):
        """
        Visualize spike trains as raster plot
        
        Args:
            spike_trains: Dict mapping orientation -> spike data
            frame_shape: Shape for output visualization
            
        Returns:
            Visualization image
        """
        import cv2
        
        plots = []
        
        for orientation in sorted(spike_trains.keys()):
            spike_data = spike_trains[orientation]
            
            plot_height = 300
            plot_width = 600
            plot = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
            
            if len(spike_data['neuron_ids']) > 0:
                min_time = self.spike_start
                max_time = self.spike_start + self.spike_window
                
                times = spike_data['spike_times']
                neurons = spike_data['neuron_ids']
                
                mask = (times >= min_time) & (times <= max_time)
                times = times[mask]
                neurons = neurons[mask]
                
                if len(times) > 0:
                    x_coords = ((times - min_time) / (max_time - min_time) * (plot_width - 20) + 10).astype(int)
                    y_coords = (neurons / self.n_neurons * (plot_height - 40) + 20).astype(int)
                    
                    for x, y in zip(x_coords, y_coords):
                        cv2.circle(plot, (x, y), 1, (0, 0, 0), -1)
            
            cv2.putText(plot, f'{orientation} deg ({len(spike_data["neuron_ids"])} spikes)',
                       (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(plot, f'{self.spike_start:.0f}ms',
                       (10, plot_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(plot, f'{self.spike_start + self.spike_window:.0f}ms',
                       (plot_width - 60, plot_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            cv2.line(plot, (10, 20), (10, plot_height - 20), (0, 0, 0), 1)
            cv2.line(plot, (10, plot_height - 20), (plot_width - 10, plot_height - 20), (0, 0, 0), 1)
            
            plots.append(plot)
        
        if len(plots) >= 4:
            top_row = np.hstack([plots[0], plots[1]])
            bottom_row = np.hstack([plots[2], plots[3]])
            combined = np.vstack([top_row, bottom_row])
        elif len(plots) >= 2:
            combined = np.vstack(plots)
        elif len(plots) == 1:
            combined = plots[0]
        else:
            combined = np.ones((300, 600, 3), dtype=np.uint8) * 255
        
        return combined

