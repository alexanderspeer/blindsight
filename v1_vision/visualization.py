"""
Visualization Utilities
Helper functions for displaying spike trains, activity, and pipeline status
"""

import numpy as np
import cv2
from config import VISUALIZATION_CONFIG, SPIKE_CONFIG


class SpikeRasterPlot:
    """Creates real-time spike raster plots"""
    
    def __init__(self, n_neurons=324, duration_ms=250):
        self.n_neurons = n_neurons
        self.duration_ms = duration_ms
        self.width = 800
        self.height = 400
        
    def plot(self, spike_trains, current_time_ms=0):
        """
        Create a spike raster plot
        
        Args:
            spike_trains: dict with 'senders' and 'times' arrays
            current_time_ms: Current time offset
            
        Returns:
            Image with raster plot
        """
        # Create blank image
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        if len(spike_trains['senders']) == 0:
            # No spikes to plot
            cv2.putText(img, "No spikes", (self.width // 2 - 50, self.height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
            return img
        
        # Filter spikes in time window
        time_window_start = current_time_ms
        time_window_end = current_time_ms + self.duration_ms
        
        mask = (spike_trains['times'] >= time_window_start) & \
               (spike_trains['times'] <= time_window_end)
        
        senders = spike_trains['senders'][mask]
        times = spike_trains['times'][mask]
        
        if len(senders) == 0:
            cv2.putText(img, "No spikes in window", (self.width // 2 - 100, self.height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            return img
        
        # Plot spikes
        margin_left = 60
        margin_right = 20
        margin_top = 40
        margin_bottom = 40
        
        plot_width = self.width - margin_left - margin_right
        plot_height = self.height - margin_top - margin_bottom
        
        # Map neuron IDs to y-positions
        neuron_to_y = {}
        active_neurons = np.unique(senders)
        n_active = len(active_neurons)
        
        for idx, neuron_id in enumerate(sorted(active_neurons)):
            y_pos = margin_top + int((idx / max(n_active - 1, 1)) * plot_height)
            neuron_to_y[neuron_id] = y_pos
        
        # Map time to x-position
        for sender, spike_time in zip(senders, times):
            # X position based on time
            time_rel = spike_time - time_window_start
            x_pos = margin_left + int((time_rel / self.duration_ms) * plot_width)
            
            # Y position based on neuron
            y_pos = neuron_to_y[sender]
            
            # Draw spike tick
            cv2.line(img, (x_pos, y_pos - 2), (x_pos, y_pos + 2), (0, 0, 0), 1)
        
        # Draw axes
        # Y-axis
        cv2.line(img, (margin_left, margin_top), 
                (margin_left, self.height - margin_bottom), (0, 0, 0), 2)
        # X-axis
        cv2.line(img, (margin_left, self.height - margin_bottom),
                (self.width - margin_right, self.height - margin_bottom), (0, 0, 0), 2)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # Y-axis label
        cv2.putText(img, "Neuron ID", (5, margin_top + plot_height // 2),
                   font, font_scale, (0, 0, 0), 1)
        
        # X-axis label
        cv2.putText(img, f"Time (ms): {time_window_start:.1f} - {time_window_end:.1f}",
                   (margin_left + 10, self.height - 10),
                   font, font_scale, (0, 0, 0), 1)
        
        # Title
        cv2.putText(img, f"Spike Raster ({len(senders)} spikes, {n_active} neurons)",
                   (margin_left, 25), font, 0.6, (0, 0, 0), 1)
        
        # Tick marks on Y-axis
        if n_active > 0:
            for tick in [0, n_active // 2, n_active - 1]:
                if tick < len(active_neurons):
                    neuron_id = sorted(active_neurons)[tick]
                    y_pos = neuron_to_y[neuron_id]
                    cv2.line(img, (margin_left - 5, y_pos), (margin_left, y_pos), (0, 0, 0), 1)
                    cv2.putText(img, str(neuron_id), (5, y_pos + 5),
                               font, 0.4, (0, 0, 0), 1)
        
        return img


class PipelineMonitor:
    """Monitors and displays pipeline performance metrics"""
    
    def __init__(self):
        self.frame_times = []
        self.stage_times = {}
        self.max_history = 30
        
    def update_frame_time(self, elapsed_ms):
        """Update frame processing time"""
        self.frame_times.append(elapsed_ms)
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)
    
    def update_stage_time(self, stage_name, elapsed_ms):
        """Update processing time for a specific stage"""
        if stage_name not in self.stage_times:
            self.stage_times[stage_name] = []
        
        self.stage_times[stage_name].append(elapsed_ms)
        if len(self.stage_times[stage_name]) > self.max_history:
            self.stage_times[stage_name].pop(0)
    
    def get_fps(self):
        """Calculate current FPS"""
        if not self.frame_times:
            return 0.0
        avg_time_ms = np.mean(self.frame_times)
        if avg_time_ms > 0:
            return 1000.0 / avg_time_ms
        return 0.0
    
    def create_stats_overlay(self, frame):
        """
        Add performance statistics overlay to frame
        
        Args:
            frame: Frame to add overlay to
            
        Returns:
            Frame with stats overlay
        """
        overlay = frame.copy()
        
        # Create semi-transparent background
        stats_height = 120
        stats_bg = np.zeros((stats_height, frame.shape[1], 3), dtype=np.uint8)
        stats_bg[:] = (0, 0, 0)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        
        y_offset = 20
        line_height = 20
        
        # FPS
        fps = self.get_fps()
        cv2.putText(stats_bg, f"FPS: {fps:.1f}", (10, y_offset),
                   font, font_scale, color, 1)
        
        # Average frame time
        if self.frame_times:
            avg_time = np.mean(self.frame_times)
            cv2.putText(stats_bg, f"Frame Time: {avg_time:.1f} ms", (10, y_offset + line_height),
                       font, font_scale, color, 1)
        
        # Stage times
        y_offset += line_height * 2
        for stage_name, times in self.stage_times.items():
            if times:
                avg_time = np.mean(times)
                cv2.putText(stats_bg, f"{stage_name}: {avg_time:.1f} ms",
                           (10, y_offset), font, font_scale, color, 1)
                y_offset += line_height
        
        # Blend with frame
        alpha = 0.7
        result = overlay.copy()
        result[0:stats_height, :] = cv2.addWeighted(
            overlay[0:stats_height, :], 1 - alpha,
            stats_bg, alpha, 0
        )
        
        return result


class MultiWindowDisplay:
    """Manages multiple display windows for the pipeline"""
    
    def __init__(self):
        self.window_names = VISUALIZATION_CONFIG['window_names']
        self.windows_created = False
        
    def create_windows(self):
        """Create all display windows"""
        if VISUALIZATION_CONFIG['display_raw']:
            cv2.namedWindow(self.window_names['raw'], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_names['raw'], 640, 360)
        
        if VISUALIZATION_CONFIG['display_preprocessed']:
            cv2.namedWindow(self.window_names['gabor'], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_names['gabor'], 800, 400)
        
        if VISUALIZATION_CONFIG['display_spikes']:
            cv2.namedWindow(self.window_names['spikes'], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_names['spikes'], 800, 400)
        
        if VISUALIZATION_CONFIG['display_v1_output']:
            cv2.namedWindow(self.window_names['v1'], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_names['v1'], 1000, 500)
        
        self.windows_created = True
    
    def update_displays(self, raw_frame=None, gabor_frame=None, 
                       spike_frame=None, v1_frame=None):
        """
        Update all display windows
        
        Args:
            raw_frame: Raw video frame
            gabor_frame: Gabor-filtered visualization
            spike_frame: Spike raster plot
            v1_frame: V1 output visualization
        """
        if not self.windows_created:
            self.create_windows()
        
        if raw_frame is not None and VISUALIZATION_CONFIG['display_raw']:
            cv2.imshow(self.window_names['raw'], raw_frame)
        
        if gabor_frame is not None and VISUALIZATION_CONFIG['display_preprocessed']:
            cv2.imshow(self.window_names['gabor'], gabor_frame)
        
        if spike_frame is not None and VISUALIZATION_CONFIG['display_spikes']:
            cv2.imshow(self.window_names['spikes'], spike_frame)
        
        if v1_frame is not None and VISUALIZATION_CONFIG['display_v1_output']:
            cv2.imshow(self.window_names['v1'], v1_frame)
    
    def destroy_windows(self):
        """Close all windows"""
        cv2.destroyAllWindows()
        self.windows_created = False


def draw_text_with_background(img, text, position, font_scale=0.6, 
                              thickness=1, text_color=(255, 255, 255), 
                              bg_color=(0, 0, 0)):
    """
    Draw text with a background rectangle for better visibility
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    padding = 5
    cv2.rectangle(img, 
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + baseline + padding),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
    
    return img

