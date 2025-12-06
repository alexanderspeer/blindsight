"""
V1 Decoder
Reconstructs visual representation from V1 cortex output
"""

import numpy as np
import cv2
from config import VISUALIZATION_CONFIG, GRID_CONFIG


class V1Decoder:
    """
    Decodes V1 spike responses back into visual representation
    Creates orientation maps and edge reconstructions
    """
    
    def __init__(self):
        self.grid_rows = GRID_CONFIG['grid_rows']
        self.grid_cols = GRID_CONFIG['grid_cols']
        self.orientations = [0, 45, 90, 135]
        self.orientation_colors = VISUALIZATION_CONFIG['orientation_colors']
        
    def decode(self, v1_output, selectivity_data):
        """
        Decode V1 output to create visual representations
        
        Args:
            v1_output: Output from V1ModelInterface.get_output()
            selectivity_data: Output from V1ModelInterface.calculate_orientation_selectivity()
            
        Returns:
            dict with multiple visualizations:
                - 'orientation_map': Color-coded orientation preference map
                - 'activity_map': Activity intensity map
                - 'edge_reconstruction': Reconstructed edge map
        """
        preferred_ori = selectivity_data['preferred_orientation']
        selectivity = selectivity_data['selectivity_index']
        firing_rates = selectivity_data['firing_rates']
        
        # Create orientation preference map
        orientation_map = self._create_orientation_map(
            preferred_ori, selectivity, firing_rates
        )
        
        # Create activity intensity map
        activity_map = self._create_activity_map(firing_rates)
        
        # Reconstruct edges using oriented filters
        edge_reconstruction = self._reconstruct_edges(
            preferred_ori, firing_rates
        )
        
        return {
            'orientation_map': orientation_map,
            'activity_map': activity_map,
            'edge_reconstruction': edge_reconstruction
        }
    
    def _create_orientation_map(self, preferred_ori, selectivity, firing_rates):
        """
        Create a color-coded map of orientation preferences
        Each spatial location colored by its preferred orientation
        """
        # Create grid image
        cell_size = 40  # Pixels per grid cell in visualization
        img_height = self.grid_rows * cell_size
        img_width = self.grid_cols * cell_size
        
        orientation_map = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Calculate total activity per neuron
        total_activity = np.zeros(324)
        for ori in self.orientations:
            total_activity += firing_rates[ori]
        
        for idx in range(324):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Get cell boundaries
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            
            # Get preferred orientation and activity
            pref_ori_idx = preferred_ori[idx]
            pref_ori_deg = self.orientations[pref_ori_idx]
            activity = total_activity[idx]
            sel_idx = selectivity[idx]
            
            # Get color for this orientation
            color = self.orientation_colors[pref_ori_deg]
            
            # Modulate intensity by activity and selectivity
            intensity = np.clip(activity / 20.0, 0, 1) * np.clip(sel_idx * 2, 0, 1)
            color_scaled = tuple(int(c * intensity) for c in color)
            
            # Fill cell
            cv2.rectangle(orientation_map, (x1, y1), (x2, y2), color_scaled, -1)
            
            # Draw grid lines
            cv2.rectangle(orientation_map, (x1, y1), (x2, y2), (50, 50, 50), 1)
        
        # Add legend
        legend_height = 60
        legend = np.zeros((legend_height, img_width, 3), dtype=np.uint8)
        
        legend_cell_width = img_width // 4
        for idx, ori in enumerate(self.orientations):
            x1 = idx * legend_cell_width
            x2 = (idx + 1) * legend_cell_width
            color = self.orientation_colors[ori]
            cv2.rectangle(legend, (x1, 0), (x2, legend_height), color, -1)
            
            # Add text
            text = f"{ori}°"
            cv2.putText(legend, text, (x1 + 10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine map and legend
        full_map = np.vstack([orientation_map, legend])
        
        return full_map
    
    def _create_activity_map(self, firing_rates):
        """
        Create a heatmap of overall neural activity
        """
        cell_size = 40
        img_height = self.grid_rows * cell_size
        img_width = self.grid_cols * cell_size
        
        # Calculate total activity per neuron
        total_activity = np.zeros(324)
        for ori in self.orientations:
            total_activity += firing_rates[ori]
        
        # Reshape to grid
        activity_grid = total_activity.reshape(self.grid_rows, self.grid_cols)
        
        # Normalize
        if np.max(activity_grid) > 0:
            activity_grid = activity_grid / np.max(activity_grid)
        
        # Scale to image size
        activity_map = cv2.resize(activity_grid, (img_width, img_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Convert to heatmap
        activity_map = (activity_map * 255).astype(np.uint8)
        activity_map = cv2.applyColorMap(activity_map, cv2.COLORMAP_JET)
        
        return activity_map
    
    def _reconstruct_edges(self, preferred_ori, firing_rates):
        """
        Reconstruct edge map using oriented line segments
        """
        cell_size = 40
        img_height = self.grid_rows * cell_size
        img_width = self.grid_cols * cell_size
        
        reconstruction = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Calculate total activity per neuron
        total_activity = np.zeros(324)
        for ori in self.orientations:
            total_activity += firing_rates[ori]
        
        for idx in range(324):
            if total_activity[idx] < 1:  # Skip inactive neurons
                continue
            
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Center of cell
            cy = int((row + 0.5) * cell_size)
            cx = int((col + 0.5) * cell_size)
            
            # Preferred orientation
            pref_ori_idx = preferred_ori[idx]
            angle_deg = self.orientations[pref_ori_idx]
            angle_rad = np.deg2rad(angle_deg)
            
            # Line length proportional to activity
            length = min(cell_size * 0.8, total_activity[idx] * 2)
            
            # Calculate line endpoints
            dx = length * np.cos(angle_rad) / 2
            dy = length * np.sin(angle_rad) / 2
            
            pt1 = (int(cx - dx), int(cy - dy))
            pt2 = (int(cx + dx), int(cy + dy))
            
            # Color based on orientation
            color = self.orientation_colors[angle_deg]
            
            # Draw line
            cv2.line(reconstruction, pt1, pt2, color, 2, cv2.LINE_AA)
        
        return reconstruction
    
    def create_combined_visualization(self, decoded_output, original_frame=None):
        """
        Combine multiple visualizations into one display
        
        Args:
            decoded_output: Output from decode()
            original_frame: Optional original frame to include
            
        Returns:
            Combined visualization image
        """
        ori_map = decoded_output['orientation_map']
        activity_map = decoded_output['activity_map']
        edge_recon = decoded_output['edge_reconstruction']
        
        # Resize to common height
        target_height = 400
        
        def resize_keep_aspect(img, target_h):
            h, w = img.shape[:2]
            aspect = w / h
            target_w = int(target_h * aspect)
            return cv2.resize(img, (target_w, target_h))
        
        ori_map_resized = resize_keep_aspect(ori_map, target_height)
        activity_map_resized = resize_keep_aspect(activity_map, target_height)
        edge_recon_resized = resize_keep_aspect(edge_recon, target_height)
        
        # Stack horizontally
        combined = np.hstack([
            ori_map_resized,
            activity_map_resized,
            edge_recon_resized
        ])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Orientation Map", (10, 30), 
                   font, 0.7, (255, 255, 255), 2)
        
        w1 = ori_map_resized.shape[1]
        cv2.putText(combined, "Activity Map", (w1 + 10, 30), 
                   font, 0.7, (255, 255, 255), 2)
        
        w2 = w1 + activity_map_resized.shape[1]
        cv2.putText(combined, "Edge Reconstruction", (w2 + 10, 30), 
                   font, 0.7, (255, 255, 255), 2)
        
        return combined
    
    def create_detailed_analysis(self, v1_output, selectivity_data):
        """
        Create detailed analysis plots
        
        Returns:
            dict with analysis images
        """
        # Firing rate distribution
        firing_rates = selectivity_data['firing_rates']
        
        # Create bar chart of firing rates per orientation
        chart_height = 300
        chart_width = 600
        chart = np.ones((chart_height, chart_width, 3), dtype=np.uint8) * 255
        
        # Calculate mean firing rates
        mean_rates = {}
        for ori in self.orientations:
            mean_rates[ori] = np.mean(firing_rates[ori])
        
        max_rate = max(mean_rates.values()) if mean_rates else 1
        
        bar_width = chart_width // (len(self.orientations) + 1)
        for idx, ori in enumerate(self.orientations):
            rate = mean_rates[ori]
            bar_height = int((rate / max_rate) * (chart_height - 50)) if max_rate > 0 else 0
            
            x1 = (idx + 1) * bar_width - bar_width // 3
            x2 = (idx + 1) * bar_width + bar_width // 3
            y1 = chart_height - 30
            y2 = y1 - bar_height
            
            color = self.orientation_colors[ori]
            cv2.rectangle(chart, (x1, y2), (x2, y1), color, -1)
            cv2.rectangle(chart, (x1, y2), (x2, y1), (0, 0, 0), 2)
            
            # Label
            cv2.putText(chart, f"{ori}°", (x1, y1 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Title
        cv2.putText(chart, "Mean Firing Rate by Orientation", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return {
            'firing_rate_chart': chart
        }

