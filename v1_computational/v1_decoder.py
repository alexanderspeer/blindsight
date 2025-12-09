"""
V1 Decoder
Reconstructs visual output (orientation/edge map) from V1 responses
"""

import numpy as np
import cv2
from config import GABOR_CONFIG, GRID_CONFIG, VISUALIZATION_CONFIG


class V1Decoder:
    """
    Decodes V1 neural responses into visual output
    Creates orientation/edge map showing detected orientations
    """
    
    def __init__(self):
        """Initialize decoder"""
        self.orientations = GABOR_CONFIG['orientations']
        self.grid_rows = GRID_CONFIG['grid_rows']
        self.grid_cols = GRID_CONFIG['grid_cols']
        self.orientation_colors = VISUALIZATION_CONFIG['orientation_colors']
        
        print("V1 Decoder initialized")
        print(f"   Output: Orientation/Edge Map ({self.grid_rows}x{self.grid_cols})")
    
    def decode_v1_output(self, v1_results, layer='layer_23'):
        """
        Decode V1 responses into orientation map
        
        Args:
            v1_results: Results from V1 model simulation
            layer: Which layer to decode from (default 'layer_23')
            
        Returns:
            Dict with orientation map and visualization
        """
        orientation_responses = {}
        for orientation in self.orientations:
            rates = v1_results['orientations'][orientation][layer]['firing_rates']
            response_grid = rates.reshape(self.grid_rows, self.grid_cols)
            orientation_responses[orientation] = response_grid
        
        orientation_map = self._create_orientation_map(orientation_responses)
        strength_map = self._create_strength_map(orientation_responses)
        
        vis_color = self._visualize_orientation_map(orientation_map, strength_map)
        vis_edges = self._visualize_as_edges(orientation_map, strength_map)
        
        return {
            'orientation_map': orientation_map,
            'strength_map': strength_map,
            'visualization_color': vis_color,
            'visualization_edges': vis_edges,
            'orientation_responses': orientation_responses
        }
    
    def _create_orientation_map(self, orientation_responses):
        """
        Create map showing preferred orientation at each location
        
        Args:
            orientation_responses: Dict mapping orientation -> 18x18 grid
            
        Returns:
            18x18 array with preferred orientation at each position
        """
        orientation_map = np.zeros((self.grid_rows, self.grid_cols))
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                responses = {}
                for orientation in self.orientations:
                    responses[orientation] = orientation_responses[orientation][row, col]
                
                if max(responses.values()) > 0:
                    pref_orientation = max(responses, key=responses.get)
                    orientation_map[row, col] = pref_orientation
                else:
                    orientation_map[row, col] = -1
        
        return orientation_map
    
    def _create_strength_map(self, orientation_responses):
        """
        Create map showing response strength at each location
        
        Args:
            orientation_responses: Dict mapping orientation -> 18x18 grid
            
        Returns:
            18x18 array with max response strength at each position
        """
        strength_map = np.zeros((self.grid_rows, self.grid_cols))
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                max_response = 0
                for orientation in self.orientations:
                    response = orientation_responses[orientation][row, col]
                    max_response = max(max_response, response)
                strength_map[row, col] = max_response
        
        return strength_map
    
    def _visualize_orientation_map(self, orientation_map, strength_map):
        """
        Create color-coded orientation map visualization
        
        Args:
            orientation_map: NxN preferred orientation map (N=grid_rows)
            strength_map: NxN response strength map
            
        Returns:
            Color visualization image
        """
        vis = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.uint8)
        strength_norm = strength_map / (strength_map.max() + 1e-6)
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                orientation = orientation_map[row, col]
                strength = strength_norm[row, col]
                
                if orientation >= 0 and orientation in self.orientation_colors:
                    color = self.orientation_colors[orientation]
                    vis[row, col] = [
                        int(color[0] * strength),
                        int(color[1] * strength),
                        int(color[2] * strength)
                    ]
        
        vis_large = cv2.resize(vis, (360, 360), interpolation=cv2.INTER_NEAREST)
        
        grid_spacing = 360 // self.grid_rows
        for i in range(0, 360, grid_spacing):
            cv2.line(vis_large, (i, 0), (i, 360), (128, 128, 128), 1)
            cv2.line(vis_large, (0, i), (360, i), (128, 128, 128), 1)
        
        legend_y = 10
        for orientation in sorted(self.orientations):
            color = self.orientation_colors[orientation]
            cv2.rectangle(vis_large, (10, legend_y), (30, legend_y + 15), color, -1)
            cv2.putText(vis_large, f'{orientation} deg', (35, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            legend_y += 20
        
        return vis_large
    
    def _visualize_as_edges(self, orientation_map, strength_map):
        """
        Visualize as oriented edge map (line segments)
        
        Args:
            orientation_map: NxN preferred orientation map (N=grid_rows)
            strength_map: NxN response strength map
            
        Returns:
            Edge visualization image
        """
        vis = np.zeros((360, 360, 3), dtype=np.uint8)
        strength_norm = strength_map / (strength_map.max() + 1e-6)
        
        cell_size = 360 // self.grid_rows
        line_length = cell_size - 5
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                orientation = orientation_map[row, col]
                strength = strength_norm[row, col]
                
                if orientation >= 0 and strength > 0.1:
                    center_x = col * cell_size + cell_size // 2
                    center_y = row * cell_size + cell_size // 2
                    
                    angle_rad = np.deg2rad(orientation)
                    dx = int(np.cos(angle_rad) * line_length / 2)
                    dy = int(np.sin(angle_rad) * line_length / 2)
                    
                    pt1 = (center_x - dx, center_y - dy)
                    pt2 = (center_x + dx, center_y + dy)
                    
                    color = self.orientation_colors.get(orientation, (255, 255, 255))
                    thickness = max(1, int(strength * 3))
                    
                    cv2.line(vis, pt1, pt2, color, thickness)
        
        cv2.putText(vis, 'V1 Orientation Map', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def visualize_layer_activity(self, v1_results):
        """
        Visualize activity across all V1 layers
        
        Args:
            v1_results: Results from V1 model simulation
            
        Returns:
            Visualization showing all layers
        """
        layers = ['layer_4', 'layer_23', 'layer_5', 'layer_6']
        
        vis_list = []
        
        for layer_name in layers:
            all_responses = []
            for orientation in self.orientations:
                rates = v1_results['orientations'][orientation][layer_name]['firing_rates']
                all_responses.append(rates)
            
            avg_response = np.mean(all_responses, axis=0)
            
            n_neurons = len(avg_response)
            if layer_name == 'layer_4' or layer_name == 'layer_23':
                grid_size = int(np.sqrt(n_neurons))
                response_grid = avg_response.reshape(grid_size, grid_size)
            elif layer_name == 'layer_5':
                response_grid = avg_response.reshape(9, 9)
            elif layer_name == 'layer_6':
                response_grid = avg_response.reshape(9, 27)
            
            if response_grid.max() > 0:
                vis = response_grid / response_grid.max()
            else:
                vis = response_grid
            
            vis = (vis * 255).astype(np.uint8)
            vis = cv2.resize(vis, (180, 180), interpolation=cv2.INTER_NEAREST)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)
            
            mean_rate = np.mean(avg_response)
            cv2.putText(vis_color, f'{layer_name} ({mean_rate:.1f} Hz)',
                       (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            vis_list.append(vis_color)
        
        top_row = np.hstack([vis_list[0], vis_list[1]])
        bottom_row = np.hstack([vis_list[2], vis_list[3]])
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def create_comparison_view(self, original_frame, v1_output):
        """
        Create side-by-side comparison of input and V1 output
        
        Args:
            original_frame: Original input frame
            v1_output: V1 decoder output
            
        Returns:
            Combined comparison image
        """
        original_resized = cv2.resize(original_frame, (360, 360))
        v1_vis = v1_output['visualization_edges']
        combined = np.hstack([original_resized, v1_vis])
        
        cv2.putText(combined, 'Input', (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, 'V1 Output', (370, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined

