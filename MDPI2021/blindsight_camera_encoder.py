#!/usr/bin/env python3
"""
Camera Spike Encoder for Blindsight V1 Integration
Converts visual input to spike trains compatible with NEST V1 model

Author: Integration wrapper for MDPI2021 V1 model
"""

import numpy as np
import cv2
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class CameraSpikeEncoder:
    """
    Encodes camera frames into spike trains for V1 model input
    
    Supports multiple encoding strategies:
    - Poisson: Rate-based probabilistic spiking
    - Latency: First-spike timing encodes intensity
    - Temporal_Contrast: Spike on intensity changes
    """
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (18, 18),
                 max_rate: float = 100.0,
                 min_rate: float = 5.0,
                 encoding_type: str = 'poisson',
                 temporal_window: float = 10.0):
        """
        Args:
            resolution: Spatial grid (18×18 = 324 neurons matches V1 model)
            max_rate: Maximum firing rate in Hz
            min_rate: Baseline firing rate in Hz
            encoding_type: 'poisson', 'latency', or 'temporal_contrast'
            temporal_window: Time window for spike encoding (ms)
        """
        self.resolution = resolution
        self.n_neurons = resolution[0] * resolution[1]
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.encoding_type = encoding_type
        self.temporal_window = temporal_window
        
        # State for temporal encoding
        self.prev_frame = None
        self.last_spike_times = defaultdict(lambda: -np.inf)
        
        # Camera calibration
        self.brightness_mean = 128
        self.brightness_std = 50
        
    def calibrate(self, frames: List[np.ndarray], n_samples: int = 100):
        """
        Calibrate encoder based on sample frames from camera
        
        Args:
            frames: List of sample frames
            n_samples: Number of frames to use for calibration
        """
        intensities = []
        for frame in frames[:n_samples]:
            downsampled = cv2.resize(frame, self.resolution, 
                                   interpolation=cv2.INTER_AREA)
            intensities.append(downsampled.flatten())
        
        all_intensities = np.concatenate(intensities)
        self.brightness_mean = np.mean(all_intensities)
        self.brightness_std = np.std(all_intensities)
        
        print(f"Calibrated: mean={self.brightness_mean:.1f}, std={self.brightness_std:.1f}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess camera frame
        
        Args:
            frame: Input frame (H, W) or (H, W, 3)
            
        Returns:
            processed: Normalized intensity grid (18, 18)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Downsample to V1 input grid
        downsampled = cv2.resize(gray, self.resolution, 
                               interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = (downsampled.astype(float) - self.brightness_mean) / self.brightness_std
        normalized = np.clip(normalized, -2, 2)  # ±2 std
        normalized = (normalized + 2) / 4  # Map to [0, 1]
        
        return normalized
    
    def encode_poisson(self, 
                       intensity: np.ndarray, 
                       current_time: float, 
                       dt: float = 0.1) -> Dict[int, List[float]]:
        """
        Poisson spike encoding
        
        Args:
            intensity: Normalized intensity grid (18, 18)
            current_time: Current simulation time (ms)
            dt: Time step (ms), should match NEST resolution
            
        Returns:
            spikes: Dict mapping neuron_id → [spike_times]
        """
        # Calculate firing rates
        rates = self.min_rate + intensity * (self.max_rate - self.min_rate)
        
        # Probability of spike in this time step
        spike_prob = rates * (dt / 1000.0)  # Convert to probability
        
        # Generate spikes
        spikes = {}
        for neuron_id in range(self.n_neurons):
            y, x = divmod(neuron_id, self.resolution[1])
            if np.random.rand() < spike_prob[y, x]:
                spikes[neuron_id] = [current_time]
        
        return spikes
    
    def encode_latency(self, 
                       intensity: np.ndarray, 
                       current_time: float,
                       window: float = None) -> Dict[int, List[float]]:
        """
        First-spike latency encoding
        Higher intensity → earlier spike within time window
        
        Args:
            intensity: Normalized intensity grid (18, 18)
            current_time: Current simulation time (ms)
            window: Time window for encoding (ms)
            
        Returns:
            spikes: Dict mapping neuron_id → [spike_time]
        """
        if window is None:
            window = self.temporal_window
        
        # Latency: high intensity → short delay
        latency = window * (1.0 - intensity)  # 0 to window ms
        
        spikes = {}
        for neuron_id in range(self.n_neurons):
            y, x = divmod(neuron_id, self.resolution[1])
            spike_time = current_time + latency[y, x]
            spikes[neuron_id] = [spike_time]
        
        return spikes
    
    def encode_temporal_contrast(self, 
                                 intensity: np.ndarray, 
                                 current_time: float,
                                 threshold: float = 0.1) -> Dict[int, List[float]]:
        """
        Temporal contrast encoding (DVS-like)
        Spike on intensity changes exceeding threshold
        
        Args:
            intensity: Normalized intensity grid (18, 18)
            current_time: Current simulation time (ms)
            threshold: Minimum change to trigger spike
            
        Returns:
            spikes: Dict mapping neuron_id → [spike_time]
        """
        spikes = {}
        
        if self.prev_frame is not None:
            # Calculate intensity change
            delta = intensity - self.prev_frame
            
            # ON spikes (brightness increase)
            on_mask = delta > threshold
            # OFF spikes (brightness decrease)
            off_mask = delta < -threshold
            
            for neuron_id in range(self.n_neurons):
                y, x = divmod(neuron_id, self.resolution[1])
                
                # Check if enough time since last spike (refractory period)
                time_since_last = current_time - self.last_spike_times[neuron_id]
                if time_since_last < 2.0:  # 2 ms refractory period
                    continue
                
                if on_mask[y, x] or off_mask[y, x]:
                    spikes[neuron_id] = [current_time]
                    self.last_spike_times[neuron_id] = current_time
        
        self.prev_frame = intensity.copy()
        return spikes
    
    def encode_frame(self, 
                    frame: np.ndarray, 
                    current_time: float,
                    dt: float = 0.1) -> Dict[int, List[float]]:
        """
        Main encoding function - dispatches to appropriate encoder
        
        Args:
            frame: Camera frame
            current_time: Current simulation time (ms)
            dt: Time step (ms)
            
        Returns:
            spikes: Dict mapping neuron_id → [spike_times]
        """
        # Preprocess
        intensity = self.preprocess_frame(frame)
        
        # Encode based on type
        if self.encoding_type == 'poisson':
            return self.encode_poisson(intensity, current_time, dt)
        elif self.encoding_type == 'latency':
            return self.encode_latency(intensity, current_time)
        elif self.encoding_type == 'temporal_contrast':
            return self.encode_temporal_contrast(intensity, current_time)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def visualize_spikes(self, 
                        spikes: Dict[int, List[float]], 
                        current_time: float,
                        window: float = 10.0) -> np.ndarray:
        """
        Create visualization of spike activity
        
        Args:
            spikes: Spike dictionary
            current_time: Current time
            window: Time window to visualize (ms)
            
        Returns:
            vis: Visualization image (18, 18)
        """
        spike_count = np.zeros(self.resolution)
        
        for neuron_id, spike_times in spikes.items():
            y, x = divmod(neuron_id, self.resolution[1])
            recent_spikes = [t for t in spike_times 
                           if current_time - t < window]
            spike_count[y, x] = len(recent_spikes)
        
        # Normalize for visualization
        if spike_count.max() > 0:
            vis = (spike_count / spike_count.max() * 255).astype(np.uint8)
        else:
            vis = np.zeros(self.resolution, dtype=np.uint8)
        
        return vis


class GaborFilterBank:
    """
    Optional: Gabor filter bank for orientation-selective preprocessing
    Mimics LGN→V1 receptive field structure
    """
    
    def __init__(self, 
                 orientations: List[float] = [0, 45, 90, 135],
                 wavelength: float = 4.0,
                 bandwidth: float = 1.0):
        """
        Args:
            orientations: List of preferred orientations (degrees)
            wavelength: Gabor wavelength (pixels)
            bandwidth: Spatial frequency bandwidth
        """
        self.orientations = orientations
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.filters = self._create_filters()
    
    def _create_filters(self) -> Dict[float, np.ndarray]:
        """Create Gabor kernels for each orientation"""
        filters = {}
        ksize = 7  # Kernel size
        sigma = self.wavelength / (np.pi * self.bandwidth)
        
        for angle_deg in self.orientations:
            angle_rad = np.deg2rad(angle_deg)
            kernel = cv2.getGaborKernel(
                (ksize, ksize), 
                sigma, 
                angle_rad, 
                self.wavelength, 
                0.5,  # Aspect ratio
                0,    # Phase offset
                ktype=cv2.CV_32F
            )
            filters[angle_deg] = kernel
        
        return filters
    
    def filter_frame(self, frame: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Apply Gabor filters to frame
        
        Args:
            frame: Input frame (H, W)
            
        Returns:
            responses: Dict mapping orientation → filtered response
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        responses = {}
        for angle, kernel in self.filters.items():
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            # Rectify (keep positive responses)
            responses[angle] = np.maximum(filtered, 0)
        
        return responses


# Example usage and testing
if __name__ == "__main__":
    print("Testing Camera Spike Encoder")
    
    # Create encoder
    encoder = CameraSpikeEncoder(
        resolution=(18, 18),
        max_rate=100.0,
        encoding_type='poisson'
    )
    
    # Test with synthetic frame
    test_frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    
    # Add oriented edge
    cv2.line(test_frame, (50, 50), (270, 190), 255, 3)
    
    # Encode
    current_time = 0.0
    spikes = encoder.encode_frame(test_frame, current_time)
    
    print(f"Generated {len(spikes)} spiking neurons")
    print(f"Total spikes: {sum(len(times) for times in spikes.values())}")
    
    # Visualize
    vis = encoder.visualize_spikes(spikes, current_time)
    vis_upscaled = cv2.resize(vis, (180, 180), interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("Original", test_frame)
    cv2.imshow("Spike Activity", vis_upscaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Test Gabor filtering
    print("\nTesting Gabor Filter Bank")
    gabor = GaborFilterBank()
    responses = gabor.filter_frame(test_frame)
    
    for angle, response in responses.items():
        print(f"{angle}°: max response = {response.max():.2f}")

