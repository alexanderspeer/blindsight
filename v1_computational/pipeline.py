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
from config import VIDEO_CONFIG, PROCESSING_CONFIG, VISUALIZATION_CONFIG, DEBUG_CONFIG


class V1VisionPipeline:
    """
    Complete pipeline from video to V1 output
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        print("Initializing V1 Vision Pipeline...")
        
        self.debug_enabled = DEBUG_CONFIG['enabled']
        self.debug_interval = DEBUG_CONFIG['print_every_n_frames']
        
        self.gabor_extractor = GaborFeatureExtractor()
        self.spike_encoder = SpikeEncoder()
        self.v1_model = ComputationalV1Model(
            dt=PROCESSING_CONFIG.get('v1_dt', 0.5),
            debug_synaptic_currents=DEBUG_CONFIG.get('show_synaptic_currents', False)
        )
        self.decoder = V1Decoder()
        
        self.frame_count = 0
        self.start_time = None
        
        self.last_viz_update = 0
        self.viz_update_interval = VISUALIZATION_CONFIG.get('update_interval_seconds', 5.0)
        self.cached_visualization = None
        
        print("\nPipeline ready!")
        print(f"   Visualization updates: Every {self.viz_update_interval}s")
    
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
        should_debug = self.debug_enabled and (self.frame_count % self.debug_interval == 0)
        
        if should_debug:
            print("\n")
            print(f"DEBUG FRAME {self.frame_count}")
        
        t1 = time.time()
        processed_frame = self._preprocess_frame(frame)
        t_preprocess = time.time() - t1
        
        if should_debug:
            self._debug_preprocessing(frame, processed_frame)
        
        t1 = time.time()
        verbose = should_debug or (self.frame_count in [0, 10])
        features, gabor_responses = self.gabor_extractor.extract_features(
            processed_frame, 
            apply_orientation_competition=True,
            verbose=verbose
        )
        t_gabor = time.time() - t1
        
        if should_debug:
            self._debug_gabor_features(features, gabor_responses)
        
        t1 = time.time()
        spike_trains = self.spike_encoder.encode_features_to_spikes(features)
        t_encode = time.time() - t1
        
        if should_debug:
            self._debug_spike_trains(spike_trains)
        
        t1 = time.time()
        v1_results = self.v1_model.run_stimulus(spike_trains, warmup=warmup)
        t_v1 = time.time() - t1
        
        if should_debug:
            self._debug_v1_results(v1_results)
        
        t1 = time.time()
        v1_output = self.decoder.decode_v1_output(v1_results, layer='layer_23')
        t_decode = time.time() - t1
        
        if should_debug:
            self._debug_decoder_output(v1_output)
        
        t_total = time.time() - t_start
        
        # Track performance
        self.frame_count += 1
        
        if should_debug:
            print("\n" + "-"*80)
            print(f"TIMING SUMMARY (Frame {self.frame_count-1}):")
            print(f"  Preprocess: {t_preprocess*1000:.2f} ms")
            print(f"  Gabor:      {t_gabor*1000:.2f} ms")
            print(f"  Encode:     {t_encode*1000:.2f} ms")
            print(f"  V1 Sim:     {t_v1*1000:.2f} ms")
            print(f"  Decode:     {t_decode*1000:.2f} ms")
            print(f"  TOTAL:      {t_total*1000:.2f} ms ({1.0/t_total:.2f} FPS)")
            print("\n")
        
        # summary at frame 10
        if self.frame_count == 10:
            self._print_frame_10_summary(v1_results, spike_trains)
            self.v1_model.print_layer_diagnostics()
        
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
        if PROCESSING_CONFIG['downsample_frame']:
            frame = cv2.resize(
                frame,
                (PROCESSING_CONFIG['downsample_width'],
                 PROCESSING_CONFIG['downsample_height'])
            )
        
        if PROCESSING_CONFIG['gaussian_blur_kernel'] > 0:
            kernel_size = PROCESSING_CONFIG['gaussian_blur_kernel']
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
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
        current_time = time.time()
        if not force_update and self.cached_visualization is not None:
            if (current_time - self.last_viz_update) < self.viz_update_interval:
                cached = self.cached_visualization.copy()
                if 'timing' in results:
                    cached['timing_text'] = self._create_timing_display(results['timing'])
                return cached
        
        self.last_viz_update = current_time
        
        visualizations = {}
        
        if VISUALIZATION_CONFIG['display_raw']:
            visualizations['raw'] = results['original_frame'].copy()
        
        gabor_vis = self.gabor_extractor.visualize_features(
            results['gabor_features'],
            results['gabor_responses']
        )
        visualizations['gabor'] = gabor_vis
        
        grid_vis = self.gabor_extractor.visualize_grid_responses(
            results['gabor_features']
        )
        visualizations['grid'] = grid_vis
        
        if VISUALIZATION_CONFIG['display_spikes']:
            spike_vis = self.spike_encoder.visualize_spike_trains(
                results['spike_trains'],
                results['processed_frame'].shape
            )
            visualizations['spikes'] = spike_vis
        
        if VISUALIZATION_CONFIG['display_v1_output']:
            visualizations['v1_color'] = results['v1_output']['visualization_color']
            visualizations['v1_edges'] = results['v1_output']['visualization_edges']
        
        layer_vis = self.decoder.visualize_layer_activity(results['v1_results'])
        visualizations['layers'] = layer_vis
        
        comparison = self.decoder.create_comparison_view(
            results['original_frame'],
            results['v1_output']
        )
        visualizations['comparison'] = comparison
        
        if 'timing' in results:
            visualizations['timing_text'] = self._create_timing_display(results['timing'])
        
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
        
        if 'raw' in visualizations:
            raw = cv2.resize(visualizations['raw'], (panel_w, panel_h))
            raw = self._add_label(raw, "RAW VIDEO INPUT")
            panels['raw'] = raw
        else:
            panels['raw'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        if 'comparison' in visualizations:
            comp = cv2.resize(visualizations['comparison'], (panel_w, panel_h))
            panels['comparison'] = comp
        else:
            panels['comparison'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        if 'v1_color' in visualizations:
            v1c = cv2.resize(visualizations['v1_color'], (panel_w, panel_h))
            v1c = self._add_label(v1c, "V1 ORIENTATION MAP")
            panels['v1_color'] = v1c
        else:
            panels['v1_color'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        if 'gabor' in visualizations:
            gabor = cv2.resize(visualizations['gabor'], (panel_w, panel_h))
            gabor = self._add_label(gabor, "GABOR FILTERS (0/45/90/135)")
            panels['gabor'] = gabor
        else:
            panels['gabor'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        if 'spikes' in visualizations:
            spikes = cv2.resize(visualizations['spikes'], (panel_w, panel_h))
            spikes = self._add_label(spikes, "SPIKE TRAINS (Latency Coded)")
            panels['spikes'] = spikes
        else:
            panels['spikes'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        if 'layers' in visualizations:
            layers = cv2.resize(visualizations['layers'], (panel_w, panel_h))
            layers = self._add_label(layers, "V1 LAYER ACTIVITY (4/2/3/5/6)")
            panels['layers'] = layers
        else:
            panels['layers'] = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        row1 = np.hstack([panels['raw'], panels['comparison'], panels['v1_color']])
        row2 = np.hstack([panels['gabor'], panels['spikes'], panels['layers']])
        combined = np.vstack([row1, row2])
        
        if 'timing_text' in visualizations:
            y_pos = 40
            for line in visualizations['timing_text'].split('\n'):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(combined, (8, y_pos - 28), (text_size[0] + 20, y_pos + 8), (0, 0, 0), -1)
                cv2.putText(combined, line, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_pos += 35
        
        return combined
    
    def _add_label(self, image, label):
        """Add a label at the top of an image"""
        img = image.copy()
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
            combined = self.create_combined_panel(visualizations)
            window_name = VISUALIZATION_CONFIG['window_names'].get('combined', 'V1 Vision Pipeline')
            
            if not hasattr(self, '_window_created'):
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                if VISUALIZATION_CONFIG.get('fullscreen', False):
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.resizeWindow(window_name, 1920, 1080)
                self._window_created = True
            
            cv2.imshow(window_name, combined)
        else:
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
            
            print(f"Connecting to Pi stream at {VIDEO_CONFIG['pi_ip']}:{VIDEO_CONFIG['port']}...")
            print(f"   Make sure Pi is running: rpicam-vid -o tcp://0.0.0.0:5001 --listen")
            
            frame_size = VIDEO_CONFIG['width'] * VIDEO_CONFIG['height'] * 3
            
            try:
                while True:
                    raw_frame = process.stdout.read(frame_size)
                    if len(raw_frame) != frame_size:
                        continue
                    
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        (VIDEO_CONFIG['height'], VIDEO_CONFIG['width'], 3)
                    )
                    
                    warmup = (self.frame_count < 3)
                    results = self.process_frame(frame, warmup=warmup)
                    visualizations = self.visualize_pipeline(results)
                    self.display_visualizations(visualizations)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            finally:
                process.terminate()
                cv2.destroyAllWindows()
        
        else:
            cap = cv2.VideoCapture(video_source)
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    warmup = (self.frame_count < 3)
                    results = self.process_frame(frame, warmup=warmup)
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
    
    def _debug_preprocessing(self, original, processed):
        """Debug preprocessing stage"""
        if not DEBUG_CONFIG['show_array_shapes']:
            return
        
        print("\nPREPROCESSING:")
        print(f"  Input shape: {original.shape}")
        print(f"  Output shape: {processed.shape}")
        
        if DEBUG_CONFIG['show_distributions']:
            print(f"  Input range: [{original.min():.2f}, {original.max():.2f}], mean={original.mean():.2f}")
            print(f"  Output range: [{processed.min():.2f}, {processed.max():.2f}], mean={processed.mean():.2f}")
        
        if DEBUG_CONFIG['check_for_nans']:
            if np.isnan(processed).any():
                print("  WARNING: NaN values detected in preprocessed frame!")
    
    def _debug_gabor_features(self, features, gabor_responses):
        """Debug Gabor feature extraction"""
        if not DEBUG_CONFIG['show_gabor_stats']:
            return
        
        print("\nGABOR FEATURE EXTRACTION:")
        
        for orientation in sorted(features.keys()):
            feature_grid = features[orientation]
            gabor_response = gabor_responses[orientation]
            
            print(f"\n  Orientation {orientation}°:")
            
            if DEBUG_CONFIG['show_array_shapes']:
                print(f"    Response shape: {gabor_response.shape}")
                print(f"    Feature grid shape: {feature_grid.shape}")
            
            if DEBUG_CONFIG['show_distributions']:
                print(f"    Gabor response - min: {gabor_response.min():.4f}, max: {gabor_response.max():.4f}, "
                      f"mean: {gabor_response.mean():.4f}, std: {gabor_response.std():.4f}")
                print(f"    Feature grid - min: {feature_grid.min():.4f}, max: {feature_grid.max():.4f}, "
                      f"mean: {feature_grid.mean():.4f}, std: {feature_grid.std():.4f}")
            
            if DEBUG_CONFIG['check_for_nans']:
                if np.isnan(feature_grid).any():
                    print(f"    WARNING: NaN values in feature grid!")
            
            if DEBUG_CONFIG['check_for_zeros']:
                if feature_grid.max() == 0:
                    print(f"    WARNING: Feature grid is all zeros!")
                else:
                    non_zero_pct = (feature_grid > 0.01).sum() / feature_grid.size * 100
                    print(f"    Active cells: {non_zero_pct:.1f}% above threshold")
    
    def _debug_spike_trains(self, spike_trains):
        """Debug spike encoding"""
        if not DEBUG_CONFIG['show_spike_stats']:
            return
        
        print("\nSPIKE ENCODING:")
        
        total_spikes = 0
        for orientation in sorted(spike_trains.keys()):
            spike_data = spike_trains[orientation]
            n_spikes = len(spike_data['neuron_ids'])
            total_spikes += n_spikes
            
            print(f"\n  Orientation {orientation}°:")
            print(f"    Number of spikes: {n_spikes}")
            
            if n_spikes > 0:
                print(f"    Neurons that spiked: {len(np.unique(spike_data['neuron_ids']))}/144")
                print(f"    Spike times - min: {spike_data['spike_times'].min():.2f} ms, "
                      f"max: {spike_data['spike_times'].max():.2f} ms, "
                      f"mean: {spike_data['spike_times'].mean():.2f} ms")
                
                # Show distribution of spike times
                early_spikes = (spike_data['spike_times'] < 100).sum()
                mid_spikes = ((spike_data['spike_times'] >= 100) & (spike_data['spike_times'] < 150)).sum()
                late_spikes = (spike_data['spike_times'] >= 150).sum()
                print(f"    Spike timing: early (<100ms): {early_spikes}, "
                      f"mid (100-150ms): {mid_spikes}, late (>150ms): {late_spikes}")
            else:
                print(f"    WARNING: No spikes generated for this orientation!")
            
            if DEBUG_CONFIG['check_for_nans']:
                if np.isnan(spike_data['spike_times']).any():
                    print(f"    WARNING: NaN values in spike times!")
        
        print(f"\n  Total spikes across all orientations: {total_spikes}")
        if total_spikes == 0:
            print("  WARNING: No spikes generated at all! Check Gabor features and thresholds!")
    
    def _debug_v1_results(self, v1_results):
        """Debug V1 simulation results"""
        if not DEBUG_CONFIG['show_v1_layer_stats']:
            return
        
        print("\nV1 SIMULATION:")
        
        for orientation in sorted(v1_results['orientations'].keys()):
            orient_data = v1_results['orientations'][orientation]
            
            print(f"\n  Orientation {orientation}°:")
            
            for layer_name in ['layer_4', 'layer_23', 'layer_5', 'layer_6']:
                layer_data = orient_data[layer_name]
                firing_rates = layer_data['firing_rates']
                mean_rate = layer_data['mean_rate']
                
                print(f"    {layer_name}:")
                print(f"      Shape: {firing_rates.shape}")
                print(f"      Mean firing rate: {mean_rate:.2f} Hz")
                
                if DEBUG_CONFIG['show_distributions']:
                    print(f"      Min: {firing_rates.min():.2f} Hz, Max: {firing_rates.max():.2f} Hz, "
                          f"Std: {firing_rates.std():.2f} Hz")
                
                if DEBUG_CONFIG['check_for_zeros']:
                    active_neurons = (firing_rates > 1.0).sum()
                    total_neurons = firing_rates.size
                    print(f"      Active neurons (>1 Hz): {active_neurons}/{total_neurons} "
                          f"({active_neurons/total_neurons*100:.1f}%)")
                
                if DEBUG_CONFIG['check_for_nans']:
                    if np.isnan(firing_rates).any():
                        print(f"      WARNING: NaN values in firing rates!")
        
        # Summary statistics
        print("\n  Cross-orientation comparison (Layer 2/3 mean rates):")
        for orientation in sorted(v1_results['orientations'].keys()):
            rate = v1_results['orientations'][orientation]['layer_23']['mean_rate']
            print(f"    {orientation}°: {rate:.2f} Hz")
    
    def _print_frame_10_summary(self, v1_results, spike_trains):
        """Print comprehensive summary after frame 10"""
        print("\n")
        print("FRAME 10 COMPREHENSIVE SUMMARY")
        
        # Spike encoding summary
        print("\nSPIKE ENCODING SUMMARY:")
        total_spikes = sum(len(spike_trains[ori]['neuron_ids']) for ori in spike_trains)
        print(f"  Total spikes generated: {total_spikes}")
        for ori in sorted(spike_trains.keys()):
            n_spikes = len(spike_trains[ori]['neuron_ids'])
            if n_spikes > 0:
                min_t = spike_trains[ori]['spike_times'].min()
                max_t = spike_trains[ori]['spike_times'].max()
                mean_t = spike_trains[ori]['spike_times'].mean()
                print(f"  {ori}deg: {n_spikes} spikes, times: {min_t:.1f}-{max_t:.1f}ms (mean={mean_t:.1f})")
            else:
                print(f"  {ori}deg: 0 spikes")
        
        # V1 layer summary
        print("\nV1 LAYER FIRING RATES (across all orientations):")
        for layer_name in ['layer_4', 'layer_23', 'layer_5', 'layer_6']:
            all_rates = []
            for ori in v1_results['orientations']:
                rates = v1_results['orientations'][ori][layer_name]['firing_rates']
                all_rates.extend(rates)
            all_rates = np.array(all_rates)
            
            active_pct = (all_rates > 1.0).sum() / len(all_rates) * 100
            print(f"  {layer_name}:")
            print(f"    Mean: {all_rates.mean():.2f} Hz, Median: {np.median(all_rates):.2f} Hz")
            print(f"    Min: {all_rates.min():.2f} Hz, Max: {all_rates.max():.2f} Hz")
            print(f"    Active (>1 Hz): {active_pct:.1f}%")
        
        # Orientation selectivity
        print("\nORIENTATION SELECTIVITY (Layer 2/3 mean rates):")
        for ori in sorted(v1_results['orientations'].keys()):
            rate = v1_results['orientations'][ori]['layer_23']['mean_rate']
            print(f"  {ori}deg: {rate:.2f} Hz")
        
        # Configuration reminder
        print("\nCURRENT CONFIGURATION:")
        from config import GRID_CONFIG, SPIKE_CONFIG, V1_ARCHITECTURE
        print(f"  RF size: {GRID_CONFIG['receptive_field_size']} px")
        print(f"  Spike timing: {SPIKE_CONFIG['spike_start_ms']}-{SPIKE_CONFIG['max_latency_ms']} ms")
        print(f"  LGN→L4 weight: {V1_ARCHITECTURE['lgn_to_ss4_weight']}")
        print(f"  Feedforward weight: {V1_ARCHITECTURE['feedforward_weight']}")
        print(f"  Spike threshold: {SPIKE_CONFIG['threshold']}")
        
        print("\n")
    
    def _debug_decoder_output(self, v1_output):
        """Debug decoder output"""
        if not DEBUG_CONFIG['show_decoder_stats']:
            return
        
        print("\nDECODER OUTPUT:")
        
        orientation_map = v1_output['orientation_map']
        strength_map = v1_output['strength_map']
        
        if DEBUG_CONFIG['show_array_shapes']:
            print(f"  Orientation map shape: {orientation_map.shape}")
            print(f"  Strength map shape: {strength_map.shape}")
        
        if DEBUG_CONFIG['show_distributions']:
            print(f"  Strength map - min: {strength_map.min():.2f}, max: {strength_map.max():.2f}, "
                  f"mean: {strength_map.mean():.2f}, std: {strength_map.std():.2f}")
        
        # Count preferred orientations
        print("\n  Preferred orientation distribution:")
        for orientation in [0, 45, 90, 135]:
            count = (orientation_map == orientation).sum()
            percentage = count / orientation_map.size * 100
            print(f"    {orientation}°: {count} pixels ({percentage:.1f}%)")
        
        no_response = (orientation_map == -1).sum()
        if no_response > 0:
            print(f"    No response: {no_response} pixels ({no_response/orientation_map.size*100:.1f}%)")
        
        if DEBUG_CONFIG['check_for_zeros']:
            if strength_map.max() == 0:
                print("  WARNING: Strength map is all zeros!")
        
        if DEBUG_CONFIG['check_for_nans']:
            if np.isnan(strength_map).any():
                print("  WARNING: NaN values in strength map!")
        
        # Check if decoder output makes sense
        active_pixels = (strength_map > 0).sum()
        print(f"\n  Active pixels (strength > 0): {active_pixels}/{strength_map.size} "
              f"({active_pixels/strength_map.size*100:.1f}%)")

