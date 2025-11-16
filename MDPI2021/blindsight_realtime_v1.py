#!/usr/bin/env python3
"""
Real-time Blindsight V1 System
Integrates camera input with NEST V1 orientation column model

Author: Integration wrapper for MDPI2021 V1 model
"""

import nest
import numpy as np
import cv2
import time
import threading
from queue import Queue, Empty
from collections import defaultdict
from typing import Dict, List, Optional
import sys

# Import local modules
from blindsight_camera_encoder import CameraSpikeEncoder
sys.path.append('./Examples/V1 Oriented Columns comapred with MEG/')
from OrientedColumnV1 import column


class LightweightColumn:
    """
    Simplified V1 column for real-time embedded deployment
    Reduces neuron count from 324 to 81 per layer
    """
    
    def __init__(self, 
                 orientation: float, 
                 lgn_inputs: List[int],
                 enable_ie_plasticity: bool = False):
        """
        Args:
            orientation: Preferred orientation (0, 45, 90, 135)
            lgn_inputs: List of NEST node IDs for LGN inputs (81 neurons)
            enable_ie_plasticity: Whether to enable IE learning at runtime
        """
        self.orientation = orientation
        self.enable_ie = enable_ie_plasticity
        
        # Create Layer 4 (Spiny Stellate cells)
        self.SS4 = nest.Create('lifl_psc_exp_ie', 81, {
            'I_e': 0.0,
            'V_m': -70.0,
            'E_L': -65.0,
            'V_th': -50.0,
            'V_reset': -65.0,
            'C_m': 250.0,
            'tau_m': 10.0,
            'tau_syn_ex': 2.0,
            'tau_syn_in': 2.0,
            't_ref': 2.0,
            'std_mod': enable_ie_plasticity,
            'lambda': 0.0005 if enable_ie_plasticity else 0.0,
            'tau': 12.5,
        })
        
        # Load pre-trained IE values (downsampled from 324 to 81)
        try:
            soma_exc = self._load_downsampled_ie(orientation)
            for i, neuron in enumerate(self.SS4):
                nest.SetStatus([neuron], {'soma_exc': soma_exc[i]})
        except FileNotFoundError:
            print(f"Warning: No pre-trained IE values for {orientation}°, using defaults")
        
        # Connect LGN → Layer 4 (4:1 pooling from 324 to 81)
        for i in range(81):
            # Each SS4 neuron receives from 4 LGN neurons
            lgn_sources = lgn_inputs[i*4:(i+1)*4] if len(lgn_inputs) == 324 else [lgn_inputs[i]]
            nest.Connect(lgn_sources, [self.SS4[i]], 'all_to_one',
                        {"weight": 3750.0 if len(lgn_inputs) == 324 else 15000.0, 
                         "delay": 1.0})
        
        # Create Layer 2/3 (Pyramidal cells) - output layer
        self.Pyr23 = nest.Create('aeif_psc_exp_peak', 81, {
            'I_e': 0.0,
            'V_m': -70.0,
            'E_L': -70.0,
            'V_th': -50.0,
            'V_reset': -55.0,
            'C_m': 250.0,
            'tau_syn_ex': 2.0,
            'tau_syn_in': 2.0,
            't_ref': 2.0,
            'g_L': 980.0
        })
        
        # Add background noise
        poisson_noise = nest.Create('poisson_generator', 1)
        nest.SetStatus(poisson_noise, {'rate': 1000000.0})  # Adjust rate
        nest.Connect(poisson_noise, self.Pyr23, 'all_to_all', {"weight": 5.0})
        
        # Connect Layer 4 → Layer 2/3
        nest.Connect(self.SS4, self.Pyr23, 'one_to_one', 
                    {"weight": 400.0, "delay": 1.0})
        
        # Create spike detector for output
        self.spike_detector = nest.Create('spike_detector')
        nest.Connect(self.Pyr23, self.spike_detector)
    
    def _load_downsampled_ie(self, orientation: float) -> List[float]:
        """Load and downsample pre-trained IE values"""
        import pickle
        
        filename = f"./Examples/V1 Oriented Columns comapred with MEG/files/soma_exc_{int(orientation)}.pckl"
        with open(filename, 'rb') as f:
            full_ie = pickle.load(f, encoding='latin1')
        
        # Downsample from 324 to 81 (every 4th value)
        downsampled = [full_ie[i] for i in range(0, 324, 4)]
        return downsampled[:81]
    
    def get_spike_count(self, time_window: float = 100.0) -> int:
        """
        Get number of spikes in recent time window
        
        Args:
            time_window: Time window to count spikes (ms)
            
        Returns:
            count: Number of spikes
        """
        events = nest.GetStatus([self.spike_detector])[0]['events']
        current_time = nest.GetKernelStatus('time')
        recent_spikes = events['times'][events['times'] > current_time - time_window]
        return len(recent_spikes)


class BlindSightV1System:
    """
    Complete real-time V1 orientation detection system
    """
    
    def __init__(self,
                 orientations: List[float] = [0, 45, 90, 135],
                 camera_source: int = 0,
                 lightweight: bool = True,
                 enable_visualization: bool = True):
        """
        Args:
            orientations: List of orientation preferences (degrees)
            camera_source: Camera index or video file path
            lightweight: Use simplified 81-neuron columns vs full 324
            enable_visualization: Show live visualization
        """
        self.orientations = orientations
        self.camera_source = camera_source
        self.lightweight = lightweight
        self.enable_vis = enable_visualization
        
        # Threading control
        self.running = False
        self.spike_queue = Queue(maxsize=1000)
        self.decision_queue = Queue(maxsize=100)
        
        # Initialize components
        self._init_camera()
        self._init_encoder()
        self._init_nest_model()
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = None
        self.decision_history = []
        
    def _init_camera(self):
        """Initialize camera capture"""
        self.camera = cv2.VideoCapture(self.camera_source)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.camera_source}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera initialized: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
              f"{self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ "
              f"{self.camera.get(cv2.CAP_PROP_FPS)} FPS")
    
    def _init_encoder(self):
        """Initialize spike encoder"""
        n_neurons = 81 if self.lightweight else 324
        grid_size = int(np.sqrt(n_neurons))
        
        self.encoder = CameraSpikeEncoder(
            resolution=(grid_size, grid_size),
            max_rate=100.0,
            min_rate=5.0,
            encoding_type='poisson',
            temporal_window=10.0
        )
        
        # Calibration with initial frames
        print("Calibrating encoder...")
        calibration_frames = []
        for _ in range(30):
            ret, frame = self.camera.read()
            if ret:
                calibration_frames.append(frame)
        self.encoder.calibrate(calibration_frames)
    
    def _init_nest_model(self):
        """Initialize NEST V1 model"""
        print("Initializing NEST simulation...")
        nest.ResetKernel()
        nest.SetKernelStatus({
            'resolution': 0.1,  # 0.1 ms time step
            'local_num_threads': 4,  # Multi-threading
            'print_time': False
        })
        
        # Create input layer
        n_neurons = 81 if self.lightweight else 324
        self.inputs = nest.Create("spike_generator", n_neurons)
        self.LGN = nest.Create('parrot_neuron', n_neurons)
        nest.Connect(self.inputs, self.LGN, 'one_to_one')
        
        # Create orientation columns
        print(f"Creating {len(self.orientations)} orientation columns "
              f"({'lightweight' if self.lightweight else 'full'} mode)...")
        
        self.columns = {}
        for angle in self.orientations:
            if self.lightweight:
                self.columns[angle] = LightweightColumn(angle, self.LGN)
            else:
                # Use full model from original paper
                col_data = column(angle, self.LGN)
                self.columns[angle] = {
                    'detector': col_data[0],
                    'Pyr23': col_data[4]
                }
        
        print("NEST model initialized")
    
    def camera_thread(self):
        """Camera capture and spike encoding thread"""
        print("Camera thread started")
        frame_interval = 1.0 / 30.0  # 30 FPS
        
        while self.running:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Encode to spikes
            current_nest_time = nest.GetKernelStatus('time')
            spikes = self.encoder.encode_frame(frame, current_nest_time)
            
            # Queue for injection
            try:
                self.spike_queue.put({
                    'time': current_nest_time,
                    'spikes': spikes,
                    'frame': frame  # For visualization
                }, timeout=0.01)
            except:
                pass  # Drop frame if queue full
            
            self.frame_count += 1
            
            # Maintain frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)
        
        print("Camera thread stopped")
    
    def simulation_thread(self):
        """NEST simulation thread"""
        print("Simulation thread started")
        dt_simulation = 10.0  # Simulate in 10 ms chunks
        decision_interval = 100.0  # Make decision every 100 ms
        last_decision_time = 0.0
        
        while self.running:
            loop_start = time.time()
            
            # Get spikes from camera
            spike_buffer = defaultdict(list)
            try:
                while True:
                    spike_data = self.spike_queue.get_nowait()
                    for neuron_id, spike_times in spike_data['spikes'].items():
                        spike_buffer[neuron_id].extend(spike_times)
                    
                    # Save frame for visualization
                    self.current_frame = spike_data['frame']
            except Empty:
                pass
            
            # Inject spikes into NEST
            for neuron_id, spike_times in spike_buffer.items():
                if spike_times:
                    nest.SetStatus([self.inputs[neuron_id]], 
                                 {'spike_times': spike_times})
            
            # Run simulation
            nest.Simulate(dt_simulation)
            
            # Make decision periodically
            current_time = nest.GetKernelStatus('time')
            if current_time - last_decision_time >= decision_interval:
                decision = self._make_decision()
                self.decision_queue.put(decision)
                last_decision_time = current_time
            
            # Performance monitoring
            sim_speed = dt_simulation / (time.time() - loop_start)
            if self.frame_count % 100 == 0 and self.frame_count > 0:
                fps = self.frame_count / (time.time() - self.start_time)
                print(f"FPS: {fps:.1f}, Sim speed: {sim_speed:.1f}x realtime")
        
        print("Simulation thread stopped")
    
    def _make_decision(self) -> Dict:
        """Extract orientation responses and make decision"""
        responses = {}
        
        for angle in self.orientations:
            if self.lightweight:
                count = self.columns[angle].get_spike_count(time_window=100.0)
            else:
                events = nest.GetStatus([self.columns[angle]['detector']])[0]['events']
                current_time = nest.GetKernelStatus('time')
                recent = events['times'][events['times'] > current_time - 100]
                count = len(recent)
            
            responses[angle] = count
        
        # Winner-take-all
        total_spikes = sum(responses.values())
        if total_spikes > 0:
            winner = max(responses, key=responses.get)
            confidence = responses[winner] / total_spikes
        else:
            winner = None
            confidence = 0.0
        
        decision = {
            'timestamp': nest.GetKernelStatus('time'),
            'dominant_orientation': winner,
            'confidence': confidence,
            'responses': responses,
            'total_activity': total_spikes
        }
        
        self.decision_history.append(decision)
        return decision
    
    def visualization_thread(self):
        """Visualization thread"""
        if not self.enable_vis:
            return
        
        print("Visualization thread started")
        
        while self.running:
            try:
                # Get latest decision
                decision = None
                try:
                    while True:
                        decision = self.decision_queue.get_nowait()
                except Empty:
                    pass
                
                if decision and hasattr(self, 'current_frame'):
                    # Create visualization
                    vis_frame = self.current_frame.copy()
                    
                    # Draw orientation indicator
                    if decision['dominant_orientation'] is not None:
                        self._draw_orientation_overlay(
                            vis_frame, 
                            decision['dominant_orientation'],
                            decision['confidence']
                        )
                    
                    # Draw response bars
                    self._draw_response_bars(vis_frame, decision['responses'])
                    
                    # Show frame
                    cv2.imshow("Blindsight V1 - Real-time", vis_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                
                time.sleep(0.033)  # ~30 FPS visualization
                
            except Exception as e:
                print(f"Visualization error: {e}")
        
        cv2.destroyAllWindows()
        print("Visualization thread stopped")
    
    def _draw_orientation_overlay(self, frame, angle, confidence):
        """Draw orientation indicator on frame"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        length = 80
        
        # Convert angle to line coordinates
        angle_rad = np.deg2rad(angle)
        end_x = int(center[0] + length * np.cos(angle_rad))
        end_y = int(center[1] + length * np.sin(angle_rad))
        
        # Draw line
        color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        thickness = int(2 + 3 * confidence)
        cv2.line(frame, center, (end_x, end_y), color, thickness)
        
        # Draw text
        text = f"{angle:.0f}deg ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_response_bars(self, frame, responses):
        """Draw response bars for all orientations"""
        h, w = frame.shape[:2]
        bar_height = 10
        max_width = 100
        
        y_offset = h - 20
        max_response = max(responses.values()) if responses else 1
        
        for i, (angle, response) in enumerate(sorted(responses.items())):
            y = y_offset - i * (bar_height + 5)
            bar_width = int((response / max_response) * max_width) if max_response > 0 else 0
            
            # Draw bar
            cv2.rectangle(frame, (10, y), (10 + bar_width, y + bar_height),
                         (255, 100, 0), -1)
            
            # Draw label
            cv2.putText(frame, f"{angle:.0f}", (115, y + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self, duration: Optional[float] = None):
        """
        Start real-time processing
        
        Args:
            duration: Run duration in seconds (None = infinite)
        """
        print("Starting Blindsight V1 System...")
        self.running = True
        self.start_time = time.time()
        
        # Start threads
        threads = [
            threading.Thread(target=self.camera_thread, daemon=True),
            threading.Thread(target=self.simulation_thread, daemon=True),
            threading.Thread(target=self.visualization_thread, daemon=True)
        ]
        
        for t in threads:
            t.start()
        
        # Wait for duration or Ctrl+C
        try:
            if duration is None:
                while self.running:
                    time.sleep(1)
            else:
                time.sleep(duration)
                self.running = False
        except KeyboardInterrupt:
            print("\nStopping...")
            self.running = False
        
        # Wait for threads
        for t in threads:
            t.join(timeout=2.0)
        
        # Cleanup
        self.camera.release()
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print performance statistics"""
        if not self.decision_history:
            return
        
        print("\n=== Performance Statistics ===")
        print(f"Total frames: {self.frame_count}")
        print(f"Total decisions: {len(self.decision_history)}")
        
        total_time = time.time() - self.start_time
        print(f"Average FPS: {self.frame_count / total_time:.1f}")
        
        # Orientation statistics
        orientations_detected = [d['dominant_orientation'] 
                               for d in self.decision_history 
                               if d['dominant_orientation'] is not None]
        
        if orientations_detected:
            print("\nOrientation detections:")
            from collections import Counter
            counts = Counter(orientations_detected)
            for angle, count in sorted(counts.items()):
                pct = 100 * count / len(orientations_detected)
                print(f"  {angle}°: {count} ({pct:.1f}%)")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Blindsight V1 Real-time System')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, default=None,
                       help='Video file path (overrides --camera)')
    parser.add_argument('--lightweight', action='store_true',
                       help='Use lightweight 81-neuron columns')
    parser.add_argument('--duration', type=float, default=None,
                       help='Run duration in seconds (default: infinite)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Initialize system
    camera_source = args.video if args.video else args.camera
    
    system = BlindSightV1System(
        orientations=[0, 45, 90, 135],
        camera_source=camera_source,
        lightweight=args.lightweight,
        enable_visualization=not args.no_vis
    )
    
    # Run
    system.run(duration=args.duration)

