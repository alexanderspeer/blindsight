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
        print("Creating Gabor filter bank...")
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
    
    def extract_features(self, frame, apply_orientation_competition=True, verbose=False):
        """
        Extract Gabor features from a frame
        
        Args:
            frame: Input image (grayscale or BGR)
            apply_orientation_competition: If True, apply softmax competition across orientations
            verbose: If True, print diagnostic information
            
        Returns:
            Dict mapping orientation -> feature map (18x18 grid)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        gray = gray.astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        features = {}
        gabor_responses = {}
        pre_norm_stats = {}
        
        for orientation in self.orientations:
            filtered = cv2.filter2D(gray, cv2.CV_32F, self.kernels[orientation])
            gabor_responses[orientation] = filtered
            
            feature_grid = self._create_retinotopic_grid(filtered)
            features[orientation] = feature_grid
            
            if verbose:
                pre_norm_stats[orientation] = {
                    'mean': feature_grid.mean(),
                    'std': feature_grid.std(),
                    'max': feature_grid.max(),
                    'min': feature_grid.min(),
                    'active_pct': (feature_grid > 0.01).sum() / feature_grid.size * 100
                }
        
        if apply_orientation_competition:
            features = self._apply_orientation_competition(features)
        
        if verbose:
            self._print_diagnostics(features, pre_norm_stats, apply_orientation_competition)
        
        return features, gabor_responses
    
    def _create_retinotopic_grid(self, filtered_image):
        """
        Create 18x18 grid of neural responses from filtered image
        Each neuron has a receptive field in the image
        
        Args:
            filtered_image: Gabor-filtered image
            
        Returns:
            18x18 array of response strengths (sparsified and normalized)
        """
        h, w = filtered_image.shape
        
        stride_y = int(h / self.grid_rows * (1.0 - self.overlap))
        stride_x = int(w / self.grid_cols * (1.0 - self.overlap))
        
        if stride_y == 0:
            stride_y = 1
        if stride_x == 0:
            stride_x = 1
        
        rf_size = self.receptive_field_size
        grid = np.zeros((self.grid_rows, self.grid_cols))
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                center_y = min(row * stride_y + rf_size // 2, h - rf_size // 2)
                center_x = min(col * stride_x + rf_size // 2, w - rf_size // 2)
                
                y_start = max(0, center_y - rf_size // 2)
                y_end = min(h, center_y + rf_size // 2)
                x_start = max(0, center_x - rf_size // 2)
                x_end = min(w, center_x + rf_size // 2)
                
                rf_patch = filtered_image[y_start:y_end, x_start:x_end]
                response = np.max(np.abs(rf_patch))
                
                grid[row, col] = response
        
        grid = self._sparsify_grid(grid)
        
        return grid
    
    def _sparsify_grid(self, grid):
        """Sparsify feature grid to improve orientation selectivity"""
        mean = grid.mean()
        std = grid.std()
        
        if std < 1e-8:
            return np.zeros_like(grid)
        
        z_scores = (grid - mean) / std
        z_scores = np.clip(z_scores, 0, None)
        
        z_max = z_scores.max()
        if z_max > 0:
            normalized = (z_scores / z_max) * 3.0
        else:
            normalized = z_scores
        
        if normalized.max() > 0:
            threshold = np.percentile(normalized[normalized > 0], 80)
            normalized[normalized < threshold] = 0
        
        return normalized
    
    def _apply_orientation_competition(self, features):
        """Apply softmax competition across orientations"""
        orientations = sorted(features.keys())
        stacked = np.stack([features[ori] for ori in orientations], axis=0)
        
        temperature = 0.5
        exp_values = np.exp(stacked / temperature)
        softmax = exp_values / (exp_values.sum(axis=0, keepdims=True) + 1e-8)
        
        sharpened = softmax * stacked
        
        result = {}
        for i, ori in enumerate(orientations):
            result[ori] = sharpened[i]
        
        return result
    
    def _print_diagnostics(self, features, pre_norm_stats, competition_applied):
        """
        Print diagnostic information about feature maps
        
        Args:
            features: Final feature maps
            pre_norm_stats: Statistics before orientation competition
            competition_applied: Whether orientation competition was applied
        """
        print("\nGABOR FEATURE DIAGNOSTICS")
        
        if pre_norm_stats:
            print("\nBEFORE ORIENTATION COMPETITION:")
            for ori in sorted(pre_norm_stats.keys()):
                stats = pre_norm_stats[ori]
                print(f"  {ori:3d}°: mean={stats['mean']:.3f}, max={stats['max']:.3f}, "
                      f"active={stats['active_pct']:.1f}%")
        
        print(f"\nAFTER {'COMPETITION' if competition_applied else 'SPARSIFICATION'}:")
        for ori in sorted(features.keys()):
            grid = features[ori]
            active_pct = (grid > 0.01).sum() / grid.size * 100
            print(f"  {ori:3d}°: mean={grid.mean():.3f}, max={grid.max():.3f}, "
                  f"active={active_pct:.1f}%")
        
        # Histogram of values
        print("\nVALUE HISTOGRAMS (after processing):")
        for ori in sorted(features.keys()):
            grid = features[ori]
            bins = [0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0]
            hist, _ = np.histogram(grid, bins=bins)
            print(f"  {ori:3d}°: ", end="")
            for i, count in enumerate(hist):
                if i < len(bins) - 1:
                    print(f"[{bins[i]:.2f}-{bins[i+1]:.2f}):{count:3d} ", end="")
            print()
        
        # Cross-orientation comparison at each location
        print("\nORIENTATION DOMINANCE:")
        orientations = sorted(features.keys())
        stacked = np.stack([features[ori] for ori in orientations], axis=0)
        winner_indices = np.argmax(stacked, axis=0)
        
        for i, ori in enumerate(orientations):
            winner_count = (winner_indices == i).sum()
            winner_pct = winner_count / winner_indices.size * 100
            print(f"  {ori:3d}° is dominant at {winner_count} locations ({winner_pct:.1f}%)")
        
        print("\n")
    
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

