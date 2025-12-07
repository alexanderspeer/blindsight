"""
Complete V1 Vision Pipeline
Processes video through Gabor extraction -> Spike encoding -> V1 model -> Reconstruction
"""

import numpy as np
import cv2
import time
from gabor_extractor import GaborFeatureExtractor
from spike_encoder import SpikeEncoder
from v1_model import ComputationalV1Model
from v1_decoder import V1Decoder
from config import VIDEO_CONFIG, PROCESSING_CONFIG, VISUALIZATION_CONFIG


class V1VisionPipeline:
    """
    Complete pipeline from video to V1 output
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        print("🚀 Initializing V1 Vision Pipeline...")
        print("=" * 60)
        
        # Create components
        self.gabor_extractor = GaborFeatureExtractor()
        self.spike_encoder = SpikeEncoder()
        self.v1_model = ComputationalV1Model(dt=PROCESSING_CONFIG.get('v1_dt', 0.5))
        self.decoder = V1Decoder()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        
        # Visualization update tracking
        self.last_viz_update = 0
        self.viz_update_interval = VISUALIZATION_CONFIG.get('update_interval_seconds', 5.0)
        self.cached_visualization = None
        
        print("\n" + "=" * 60)
        print("✅ Pipeline ready!")
        print(f"   Visualization updates: Every {self.viz_update_interval}s")
        print("=" * 60)
    
    def process_frame(self, frame, warmup=False):
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input video frame
            warmup: Whether to run V1 warmup period
            
        Returns:
            Dict with all intermediate and final outputs
        """
        t_start = time.time()
        
        # 1. Preprocess frame
        t1 = time.time()
        processed_frame = self._preprocess_frame(frame)
        t_preprocess = time.time() - t1
        
        # 2. Extract Gabor features
        t1 = time.time()
        features, gabor_responses = self.gabor_extractor.extract_features(processed_frame)
        t_gabor = time.time() - t1
        
        # 3. Encode to spike trains
        t1 = time.time()
        spike_trains = self.spike_encoder.encode_features_to_spikes(features)
        t_encode = time.time() - t1
        
        # 4. Run V1 simulation
        t1 = time.time()
        v1_results = self.v1_model.run_stimulus(spike_trains, warmup=warmup)
        t_v1 = time.time() - t1
        
        # 5. Decode V1 output
        t1 = time.time()
        v1_output = self.decoder.decode_v1_output(v1_results, layer='layer_23')
        t_decode = time.time() - t1
        
        t_total = time.time() - t_start
        
        # Track performance
        self.frame_count += 1
        
        # Debug: Check if neurons are firing
        if self.frame_count % 10 == 0:  # Every 10 frames
            for orientation in [0, 45, 90, 135]:
                l23_rate = v1_results['orientations'][orientation]['layer_23']['mean_rate']
                l4_rate = v1_results['orientations'][orientation]['layer_4']['mean_rate']
                if self.frame_count == 10:
                    print(f"[Debug] {orientation}deg - Layer 4: {l4_rate:.1f} Hz, Layer 2/3: {l23_rate:.1f} Hz")
        
        return {
            'original_frame': frame,
            'processed_frame': processed_frame,
            'gabor_features': features,
            'gabor_responses': gabor_responses,
            'spike_trains': spike_trains,
            'v1_results': v1_results,
            'v1_output': v1_output,
            'timing': {
                'preprocess': t_preprocess,
                'gabor': t_gabor,
                'encode': t_encode,
                'v1': t_v1,
                'decode': t_decode,
                'total': t_total
            }
        }
    
    def _preprocess_frame(self, frame):
        """
        Preprocess video frame
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Downsample if configured
        if PROCESSING_CONFIG['downsample_frame']:
            frame = cv2.resize(
                frame,
                (PROCESSING_CONFIG['downsample_width'],
                 PROCESSING_CONFIG['downsample_height'])
            )
        
        # Apply Gaussian blur
        if PROCESSING_CONFIG['gaussian_blur_kernel'] > 0:
            kernel_size = PROCESSING_CONFIG['gaussian_blur_kernel']
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        # Normalize contrast
        if PROCESSING_CONFIG['normalize_contrast']:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        
        return frame
    
    def visualize_pipeline(self, results, force_update=False):
        """
        Create comprehensive visualization of pipeline stages
        
        Args:
            results: Pipeline results dict
            force_update: Force visualization update regardless of timer
            
        Returns:
            Dict with visualization images
        """
        # Check if we should update visualizations
        current_time = time.time()
        if not force_update and self.cached_visualization is not None:
            if (current_time - self.last_viz_update) < self.viz_update_interval:
                # Return cached visualization with updated timing only
                cached = self.cached_visualization.copy()
                if 'timing' in results:
                    cached['timing_text'] = self._create_timing_display(results['timing'])
                return cached
        
        # Time to update visualizations
        self.last_viz_update = current_time
        
        visualizations = {}
        
        # 1. Raw video (always show for real-time feedback)
        if VISUALIZATION_CONFIG['display_raw']:
            visualizations['raw'] = results['original_frame'].copy()
        
        # 2. Gabor features
        gabor_vis = self.gabor_extractor.visualize_features(
            results['gabor_features'],
            results['gabor_responses']
        )
        visualizations['gabor'] = gabor_vis
        
        # 3. Grid responses
        grid_vis = self.gabor_extractor.visualize_grid_responses(
            results['gabor_features']
        )
        visualizations['grid'] = grid_vis
        
        # 4. Spike trains
        if VISUALIZATION_CONFIG['display_spikes']:
            spike_vis = self.spike_encoder.visualize_spike_trains(
                results['spike_trains'],
                results['processed_frame'].shape
            )
            visualizations['spikes'] = spike_vis
        
        # 5. V1 output (orientation map)
        if VISUALIZATION_CONFIG['display_v1_output']:
            visualizations['v1_color'] = results['v1_output']['visualization_color']
            visualizations['v1_edges'] = results['v1_output']['visualization_edges']
        
        # 6. Layer activity
        layer_vis = self.decoder.visualize_layer_activity(results['v1_results'])
        visualizations['layers'] = layer_vis
        
        # 7. Comparison view
        comparison = self.decoder.create_comparison_view(
            results['original_frame'],
            results['v1_output']
        )
        visualizations['comparison'] = comparison
        
        # 8. Add timing info
        if 'timing' in results:
            visualizations['timing_text'] = self._create_timing_display(results['timing'])
        
        # Cache for next frames
        self.cached_visualization = visualizations.copy()
        
        return visualizations
    
    def _create_timing_display(self, timing):
        """
        Create text overlay with timing information
        
        Args:
            timing: Dict with timing info
            
        Returns:
            String with formatted timing info
        """
        lines = [
            f"Frame: {self.frame_count}",
            f"Preprocess: {timing['preprocess']*1000:.1f} ms",
            f"Gabor: {timing['gabor']*1000:.1f} ms",
            f"Encoding: {timing['encode']*1000:.1f} ms",
            f"V1 Simulation: {timing['v1']*1000:.1f} ms",
            f"Decoding: {timing['decode']*1000:.1f} ms",
            f"Total: {timing['total']*1000:.1f} ms",
            f"FPS: {1.0/timing['total']:.2f}",
        ]
        return "\n".join(lines)
    
    def create_combined_panel(self, visualizations):
        """
        Combine all visualizations into one labeled full-screen panel
        
        Args:
            visualizations: Dict with visualization images
            
        Returns:
            Combined panel image
        """
        # Define panel layout (3x2 grid)
        # Row 1: Raw Video | Comparison | V1 Color
        # Row 2: Gabor    | Spikes     | Layers
        
        # Resize all components to larger size for full screen (1920×1080)
        panel_h, panel_w = 540, 640  # Each panel 640×540, total 1920×1080
        
        # Prepare panels
        panels = {}
        
        # Raw video
        if 'raw' in visualizations:
            raw = cv2.resize(visualizations['raw'], (panel_w, panel_h))
            raw = self._add_label(raw, "RAW VIDEO INPUT")
            panels['raw'] = raw
        else:
            panels['raw'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # Comparison
        if 'comparison' in visualizations:
            comp = cv2.resize(visualizations['comparison'], (panel_w, panel_h))
            panels['comparison'] = comp
        else:
            panels['comparison'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # V1 Color map
        if 'v1_color' in visualizations:
            v1c = cv2.resize(visualizations['v1_color'], (panel_w, panel_h))
            v1c = self._add_label(v1c, "V1 ORIENTATION MAP")
            panels['v1_color'] = v1c
        else:
            panels['v1_color'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # Gabor features
        if 'gabor' in visualizations:
            gabor = cv2.resize(visualizations['gabor'], (panel_w, panel_h))
            gabor = self._add_label(gabor, "GABOR FILTERS (0/45/90/135)")
            panels['gabor'] = gabor
        else:
            panels['gabor'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # Spike trains
        if 'spikes' in visualizations:
            spikes = cv2.resize(visualizations['spikes'], (panel_w, panel_h))
            spikes = self._add_label(spikes, "SPIKE TRAINS (Latency Coded)")
            panels['spikes'] = spikes
        else:
            panels['spikes'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # Layer activity
        if 'layers' in visualizations:
            layers = cv2.resize(visualizations['layers'], (panel_w, panel_h))
            layers = self._add_label(layers, "V1 LAYER ACTIVITY (4/2/3/5/6)")
            panels['layers'] = layers
        else:
            panels['layers'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        # Combine into rows
        row1 = np.hstack([panels['raw'], panels['comparison'], panels['v1_color']])
        row2 = np.hstack([panels['gabor'], panels['spikes'], panels['layers']])
        
        # Combine rows
        combined = np.vstack([row1, row2])
        
        # Add timing overlay
        if 'timing_text' in visualizations:
            y_pos = 40
            for line in visualizations['timing_text'].split('\n'):
                # Add black background for readability
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(combined, (8, y_pos - 28), (text_size[0] + 20, y_pos + 8), (0, 0, 0), -1)
                cv2.putText(combined, line, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_pos += 35
        
        return combined
    
    def _add_label(self, image, label):
        """Add a label at the top of an image"""
        img = image.copy()
        # Add black background for label (larger for bigger display)
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, label, (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img
    
    def display_visualizations(self, visualizations):
        """
        Display all visualizations in windows
        
        Args:
            visualizations: Dict with visualization images
        """
        if VISUALIZATION_CONFIG.get('combined_display', True):
            # Show combined panel in one window
            combined = self.create_combined_panel(visualizations)
            window_name = VISUALIZATION_CONFIG['window_names'].get('combined', 'V1 Vision Pipeline')
            
            # Create window with option for fullscreen
            if not hasattr(self, '_window_created'):
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if VISUALIZATION_CONFIG.get('fullscreen', False):
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    # Set explicit window size (1920x1080)
                    cv2.resizeWindow(window_name, 1920, 1080)
                self._window_created = True
            
            cv2.imshow(window_name, combined)
        else:
            # Original multi-window display
            window_positions = {
                'comparison': (0, 0),
                'gabor': (730, 0),
                'spikes': (0, 400),
                'layers': (730, 400),
            }
            
            for name, img in visualizations.items():
                if name in window_positions:
                    window_name = VISUALIZATION_CONFIG['window_names'].get(name, name)
                    cv2.imshow(window_name, img)
                    
                    # Set window position
                    if name in window_positions:
                        x, y = window_positions[name]
                        cv2.moveWindow(window_name, x, y)
            
            # Add timing overlay to comparison window
            if 'timing_text' in visualizations and 'comparison' in visualizations:
                timing_img = visualizations['comparison'].copy()
                y_pos = 50
                for line in visualizations['timing_text'].split('\n'):
                    cv2.putText(timing_img, line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_pos += 20
                cv2.imshow(VISUALIZATION_CONFIG['window_names'].get('comparison', 'Comparison'), timing_img)
    
    def run_on_video_stream(self, video_source=None):
        """
        Run pipeline on video stream (Pi camera or file)
        
        Args:
            video_source: Video source (None for Pi camera, or file path)
        """
        if video_source is None:
            # Connect to Pi camera via ffmpeg (matches receive.py exactly)
            ffmpeg_cmd = [
                'ffmpeg',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-analyzeduration', '0',
                '-probesize', '32',
                '-i', f'tcp://{VIDEO_CONFIG["pi_ip"]}:{VIDEO_CONFIG["port"]}?listen=0',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-',
            ]
            import subprocess
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=VIDEO_CONFIG['width'] * VIDEO_CONFIG['height'] * 3
            )
            
            print(f"🎥 Connecting to Pi stream at {VIDEO_CONFIG['pi_ip']}:{VIDEO_CONFIG['port']}...")
            print(f"   Make sure Pi is running: rpicam-vid -o tcp://0.0.0.0:5001 --listen")
            
            # Read frames
            frame_size = VIDEO_CONFIG['width'] * VIDEO_CONFIG['height'] * 3
            
            try:
                while True:
                    raw_frame = process.stdout.read(frame_size)
                    if len(raw_frame) != frame_size:
                        continue  # Skip incomplete frames
                    
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        (VIDEO_CONFIG['height'], VIDEO_CONFIG['width'], 3)
                    )
                    
                    # Process frame (warmup on first few frames)
                    warmup = (self.frame_count < 3)
                    results = self.process_frame(frame, warmup=warmup)
                    
                    # Visualize
                    visualizations = self.visualize_pipeline(results)
                    self.display_visualizations(visualizations)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            finally:
                process.terminate()
                cv2.destroyAllWindows()
        
        else:
            # Run on video file
            cap = cv2.VideoCapture(video_source)
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame (warmup on first few frames)
                    warmup = (self.frame_count < 3)
                    results = self.process_frame(frame, warmup=warmup)
                    
                    # Visualize
                    visualizations = self.visualize_pipeline(results)
                    self.display_visualizations(visualizations)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            finally:
                cap.release()
                cv2.destroyAllWindows()
    
    def reset(self):
        """Reset pipeline state"""
        self.v1_model.reset()
        self.frame_count = 0
        self.start_time = None

