"""
Real-time V1 Vision Pipeline
Main script that integrates video stream, feature extraction, spike encoding,
V1 model simulation, and visualization
"""

import cv2
import subprocess
import numpy as np
import time
import sys

from config import (
    VIDEO_CONFIG, PROCESSING_CONFIG, 
    VISUALIZATION_CONFIG, PERFORMANCE_CONFIG
)
from gabor_feature_extractor import GaborFeatureExtractor
from spike_encoder import SpikeEncoder
from v1_model_interface import V1ModelInterface
from v1_decoder import V1Decoder
from visualization import (
    SpikeRasterPlot, PipelineMonitor, 
    MultiWindowDisplay, draw_text_with_background
)


class V1VisionPipeline:
    """
    Main pipeline class that orchestrates the entire V1 vision processing
    """
    
    def __init__(self):
        print("=" * 60)
        print("Initializing V1 Vision Pipeline")
        print("=" * 60)
        
        # Initialize components
        print("\n[1/6] Initializing Gabor feature extractor...")
        self.gabor_extractor = GaborFeatureExtractor()
        
        print("[2/6] Initializing spike encoder...")
        self.spike_encoder = SpikeEncoder()
        
        print("[3/6] Initializing V1 model interface...")
        self.v1_model = V1ModelInterface()
        
        print("[4/6] Setting up V1 model architecture...")
        self.v1_model.setup_model()
        
        print("[5/6] Initializing decoder...")
        self.v1_decoder = V1Decoder()
        
        print("[6/6] Initializing visualization...")
        self.spike_plotter = SpikeRasterPlot()
        self.monitor = PipelineMonitor()
        self.display = MultiWindowDisplay()
        
        # State
        self.frame_count = 0
        self.simulation_time = 0
        self.running = False
        
        print("\n✓ Pipeline initialized successfully!")
        print("=" * 60)
    
    def setup_video_stream(self):
        """
        Setup FFmpeg video stream from Raspberry Pi
        """
        print("\nConnecting to Raspberry Pi video stream...")
        print(f"  IP: {VIDEO_CONFIG['pi_ip']}")
        print(f"  Port: {VIDEO_CONFIG['port']}")
        print(f"  Resolution: {VIDEO_CONFIG['width']}x{VIDEO_CONFIG['height']}")
        
        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-i", f"tcp://{VIDEO_CONFIG['pi_ip']}:{VIDEO_CONFIG['port']}?listen=0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        
        try:
            self.video_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Suppress FFmpeg output
                bufsize=VIDEO_CONFIG['width'] * VIDEO_CONFIG['height'] * 3
            )
            print("✓ Video stream connected!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to video stream: {e}")
            return False
    
    def read_frame(self):
        """Read a frame from the video stream"""
        buffer_size = VIDEO_CONFIG['width'] * VIDEO_CONFIG['height'] * 3
        
        try:
            raw = self.video_process.stdout.read(buffer_size)
            
            if len(raw) != buffer_size:
                return None
            
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (VIDEO_CONFIG['height'], VIDEO_CONFIG['width'], 3)
            ).copy()
            
            return frame
            
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def process_frame(self, frame):
        """
        Process a single frame through the entire pipeline
        
        Returns:
            dict with all processing results and visualizations
        """
        results = {}
        stage_start = time.time()
        
        # Optional: Downsample for faster processing
        if PROCESSING_CONFIG['downsample_frame']:
            frame_processed = cv2.resize(
                frame,
                (PROCESSING_CONFIG['downsample_width'], 
                 PROCESSING_CONFIG['downsample_height'])
            )
        else:
            frame_processed = frame
        
        # Stage 1: Extract Gabor features
        stage_start = time.time()
        features = self.gabor_extractor.extract_features(frame_processed)
        gabor_viz = self.gabor_extractor.visualize_gabor_responses(features['gabor_images'])
        stage_time = (time.time() - stage_start) * 1000
        if PERFORMANCE_CONFIG['profile_stages']:
            self.monitor.update_stage_time('Gabor', stage_time)
        results['features'] = features
        results['gabor_viz'] = gabor_viz
        
        # Stage 2: Encode to spikes
        stage_start = time.time()
        spike_trains = self.spike_encoder.encode(features, self.simulation_time)
        spike_stats = self.spike_encoder.get_spike_statistics(spike_trains)
        stage_time = (time.time() - stage_start) * 1000
        if PERFORMANCE_CONFIG['profile_stages']:
            self.monitor.update_stage_time('Spike Encoding', stage_time)
        results['spike_trains'] = spike_trains
        results['spike_stats'] = spike_stats
        
        # Stage 3: Run through V1 model
        stage_start = time.time()
        spike_trains_nest = self.spike_encoder.format_for_nest(spike_trains)
        self.v1_model.inject_spikes(spike_trains_nest)
        
        # Run simulation (only stimulus period, warmup already done)
        self.v1_model.run_simulation(warmup=False)
        
        # Get V1 output
        v1_output = self.v1_model.get_output()
        selectivity = self.v1_model.calculate_orientation_selectivity(v1_output)
        stage_time = (time.time() - stage_start) * 1000
        if PERFORMANCE_CONFIG['profile_stages']:
            self.monitor.update_stage_time('V1 Simulation', stage_time)
        results['v1_output'] = v1_output
        results['selectivity'] = selectivity
        
        # Stage 4: Decode V1 output
        stage_start = time.time()
        decoded = self.v1_decoder.decode(v1_output, selectivity)
        v1_viz = self.v1_decoder.create_combined_visualization(decoded, frame)
        stage_time = (time.time() - stage_start) * 1000
        if PERFORMANCE_CONFIG['profile_stages']:
            self.monitor.update_stage_time('Decoding', stage_time)
        results['decoded'] = decoded
        results['v1_viz'] = v1_viz
        
        # Stage 5: Create spike raster plot
        spike_viz = self.spike_plotter.plot(spike_trains, self.simulation_time)
        results['spike_viz'] = spike_viz
        
        return results
    
    def run(self):
        """
        Main processing loop
        """
        # Setup video stream
        if not self.setup_video_stream():
            print("\nPlease ensure:")
            print("1. Raspberry Pi is powered on and connected to network")
            print("2. rpicam-vid is running on the Pi (see receive.py for command)")
            return
        
        print("\n" + "=" * 60)
        print("Starting real-time processing...")
        print("Press 'q' to quit")
        print("=" * 60 + "\n")
        
        self.running = True
        self.display.create_windows()
        
        try:
            while self.running:
                frame_start_time = time.time()
                
                # Read frame
                frame = self.read_frame()
                if frame is None:
                    print("Lost connection to video stream")
                    break
                
                self.frame_count += 1
                
                # Process every Nth frame
                if self.frame_count % VISUALIZATION_CONFIG['update_interval_frames'] == 0:
                    # Process frame
                    results = self.process_frame(frame)
                    
                    # Prepare raw frame with overlay
                    raw_display = frame.copy()
                    
                    # Add receptive field grid
                    raw_display = self.gabor_extractor.visualize_receptive_fields(
                        raw_display, show_all=False
                    )
                    
                    # Add info text
                    info_text = f"Frame: {self.frame_count} | " \
                               f"Spikes: {results['spike_stats']['n_spikes']} | " \
                               f"Active neurons: {results['spike_stats']['n_active_neurons']}"
                    
                    raw_display = draw_text_with_background(
                        raw_display, info_text, (10, 30)
                    )
                    
                    # Add performance stats if enabled
                    if PERFORMANCE_CONFIG['show_fps']:
                        raw_display = self.monitor.create_stats_overlay(raw_display)
                    
                    # Update displays
                    self.display.update_displays(
                        raw_frame=raw_display,
                        gabor_frame=results['gabor_viz'],
                        spike_frame=results['spike_viz'],
                        v1_frame=results['v1_viz']
                    )
                    
                    # Print status
                    print(f"Frame {self.frame_count}: "
                          f"{results['spike_stats']['n_spikes']} spikes, "
                          f"{results['spike_stats']['n_active_neurons']} active neurons, "
                          f"FPS: {self.monitor.get_fps():.1f}")
                
                else:
                    # Just display raw frame
                    if VISUALIZATION_CONFIG['display_raw']:
                        cv2.imshow(self.display.window_names['raw'], frame)
                
                # Calculate frame time
                frame_time = (time.time() - frame_start_time) * 1000
                self.monitor.update_frame_time(frame_time)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    self.running = False
                    break
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        except Exception as e:
            print(f"\n\nError in processing loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        if hasattr(self, 'video_process'):
            self.video_process.terminate()
        
        self.display.destroy_windows()
        
        print("✓ Cleanup complete")
        print("\nPipeline statistics:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Average FPS: {self.monitor.get_fps():.1f}")


def main():
    """Entry point"""
    print("\n" + "=" * 60)
    print(" " * 15 + "V1 Vision Pipeline")
    print(" " * 10 + "Biologically-Inspired Visual Processing")
    print("=" * 60)
    
    # Create and run pipeline
    pipeline = V1VisionPipeline()
    pipeline.run()
    
    print("\nGoodbye!\n")


if __name__ == "__main__":
    main()

