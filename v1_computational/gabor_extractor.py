"""
Gabor Feature Extraction
Extracts orientation-selective features from video frames
"""

import numpy as np
import cv2
from config import GABOR_CONFIG, GRID_CONFIG


class GaborFeatureExtractor:
    """
    Extracts Gabor features for multiple orientations
    Creates a retinotopic map of orientation responses
    """
    
    def __init__(self):
        """Initialize Gabor filter bank"""
        self.orientations = GABOR_CONFIG['orientations']
        self.wavelength = GABOR_CONFIG['wavelength']
        self.sigma = GABOR_CONFIG['sigma']
        self.gamma = GABOR_CONFIG['gamma']
        self.psi = GABOR_CONFIG['psi']
        self.kernel_size = GABOR_CONFIG['kernel_size']
        
        # Grid configuration
        self.grid_rows = GRID_CONFIG['grid_rows']
        self.grid_cols = GRID_CONFIG['grid_cols']
        self.n_neurons = GRID_CONFIG['n_neurons']
        self.receptive_field_size = GRID_CONFIG['receptive_field_size']
        self.overlap = GRID_CONFIG['overlap']
        
        # Create Gabor kernels
        print("🔬 Creating Gabor filter bank...")
        self.kernels = {}
        for orientation in self.orientations:
            theta = np.deg2rad(orientation)
            kernel = cv2.getGaborKernel(
                (self.kernel_size, self.kernel_size),
                self.sigma,
                theta,
                self.wavelength,
                self.gamma,
                self.psi,
                ktype=cv2.CV_32F
            )
            self.kernels[orientation] = kernel
            print(f"   {orientation}° filter created")
    
    def extract_features(self, frame):
        """
        Extract Gabor features from a frame
        
        Args:
            frame: Input image (grayscale or BGR)
            
        Returns:
            Dict mapping orientation -> feature map (18x18 grid)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Extract features for each orientation
        features = {}
        gabor_responses = {}
        
        for orientation in self.orientations:
            # Apply Gabor filter
            filtered = cv2.filter2D(gray, cv2.CV_32F, self.kernels[orientation])
            gabor_responses[orientation] = filtered
            
            # Create retinotopic grid (18x18 neurons)
            feature_grid = self._create_retinotopic_grid(filtered)
            features[orientation] = feature_grid
        
        return features, gabor_responses
    
    def _create_retinotopic_grid(self, filtered_image):
        """
        Create 18x18 grid of neural responses from filtered image
        Each neuron has a receptive field in the image
        
        Args:
            filtered_image: Gabor-filtered image
            
        Returns:
            18x18 array of response strengths
        """
        h, w = filtered_image.shape
        
        # Calculate receptive field positions with overlap
        stride_y = int(h / self.grid_rows * (1.0 - self.overlap))
        stride_x = int(w / self.grid_cols * (1.0 - self.overlap))
        
        if stride_y == 0:
            stride_y = 1
        if stride_x == 0:
            stride_x = 1
        
        rf_size = self.receptive_field_size
        
        # Extract responses
        grid = np.zeros((self.grid_rows, self.grid_cols))
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Receptive field center
                center_y = min(row * stride_y + rf_size // 2, h - rf_size // 2)
                center_x = min(col * stride_x + rf_size // 2, w - rf_size // 2)
                
                # Extract receptive field
                y_start = max(0, center_y - rf_size // 2)
                y_end = min(h, center_y + rf_size // 2)
                x_start = max(0, center_x - rf_size // 2)
                x_end = min(w, center_x + rf_size // 2)
                
                rf_patch = filtered_image[y_start:y_end, x_start:x_end]
                
                # Response is the MAX absolute response in receptive field
                # FIX: Use max instead of mean for better selectivity
                # Use absolute value to capture both ON and OFF responses
                response = np.max(np.abs(rf_patch))
                
                grid[row, col] = response
        
        # FIX: Normalize grid to expand dynamic range
        grid_min, grid_max = grid.min(), grid.max()
        if grid_max > grid_min:
            grid = (grid - grid_min) / (grid_max - grid_min) * 3.0
        
        return grid
    
    def visualize_features(self, features, gabor_responses):
        """
        Create visualization of Gabor features
        
        Args:
            features: Dict mapping orientation -> 18x18 grid
            gabor_responses: Dict mapping orientation -> full Gabor response
            
        Returns:
            Visualization image
        """
        # Create 2x2 grid of Gabor responses
        vis_list = []
        
        for orientation in self.orientations:
            response = gabor_responses[orientation]
            
            # Normalize for visualization
            vis = np.abs(response)
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-6)
            vis = (vis * 255).astype(np.uint8)
            
            # Apply colormap
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            
            # Add label
            cv2.putText(vis_color, f'{orientation} deg', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 2)
            
            vis_list.append(vis_color)
        
        # Arrange in 2x2 grid
        top_row = np.hstack([vis_list[0], vis_list[1]])
        bottom_row = np.hstack([vis_list[2], vis_list[3]])
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def visualize_grid_responses(self, features):
        """
        Visualize 18x18 grid responses for all orientations
        
        Args:
            features: Dict mapping orientation -> 18x18 grid
            
        Returns:
            Visualization image
        """
        vis_list = []
        
        for orientation in self.orientations:
            grid = features[orientation]
            
            # Upscale grid for visibility
            grid_vis = cv2.resize(grid, (180, 180), interpolation=cv2.INTER_NEAREST)
            
            # Normalize
            grid_vis = (grid_vis - grid_vis.min()) / (grid_vis.max() - grid_vis.min() + 1e-6)
            grid_vis = (grid_vis * 255).astype(np.uint8)
            
            # Apply colormap
            grid_color = cv2.applyColorMap(grid_vis, cv2.COLORMAP_HOT)
            
            # Add label
            cv2.putText(grid_color, f'{orientation} deg', 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            vis_list.append(grid_color)
        
        # Arrange in 2x2 grid
        top_row = np.hstack([vis_list[0], vis_list[1]])
        bottom_row = np.hstack([vis_list[2], vis_list[3]])
        combined = np.vstack([top_row, bottom_row])
        
        return combined

