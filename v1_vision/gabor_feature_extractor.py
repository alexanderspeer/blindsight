"""
Gabor Feature Extractor
Applies Gabor filters at multiple orientations to extract edge/orientation features
"""

import cv2
import numpy as np
from config import GABOR_CONFIG, GRID_CONFIG, PROCESSING_CONFIG


class GaborFeatureExtractor:
    """
    Extracts orientation-selective features using Gabor filters
    Organizes responses into a spatial grid matching V1 model input (324 neurons)
    """
    
    def __init__(self):
        self.orientations = GABOR_CONFIG['orientations']
        self.n_neurons = GRID_CONFIG['n_neurons']
        self.grid_rows = GRID_CONFIG['grid_rows']
        self.grid_cols = GRID_CONFIG['grid_cols']
        
        # Pre-compute Gabor kernels for each orientation
        self.gabor_kernels = self._create_gabor_kernels()
        
        # Calculate receptive field positions
        self.receptive_fields = None  # Will be set based on actual frame size
        
    def _create_gabor_kernels(self):
        """Create Gabor filter kernels for each orientation"""
        kernels = {}
        ksize = GABOR_CONFIG['kernel_size']
        
        for theta_deg in self.orientations:
            theta = np.deg2rad(theta_deg)
            kernel = cv2.getGaborKernel(
                ksize=(ksize, ksize),
                sigma=GABOR_CONFIG['sigma'],
                theta=theta,
                lambd=GABOR_CONFIG['wavelength'],
                gamma=GABOR_CONFIG['gamma'],
                psi=GABOR_CONFIG['psi'],
                ktype=cv2.CV_32F
            )
            kernels[theta_deg] = kernel
        
        return kernels
    
    def _initialize_receptive_fields(self, frame_height, frame_width):
        """
        Calculate center positions for each of the 324 receptive fields
        Arranged in an 18x18 grid
        """
        receptive_fields = []
        
        # Calculate spacing between receptive field centers
        row_step = frame_height / self.grid_rows
        col_step = frame_width / self.grid_cols
        
        # Calculate receptive field size (with overlap)
        rf_height = int(row_step * (1 + GRID_CONFIG['overlap']))
        rf_width = int(col_step * (1 + GRID_CONFIG['overlap']))
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Center of receptive field
                center_y = int((row + 0.5) * row_step)
                center_x = int((col + 0.5) * col_step)
                
                # Boundaries (with bounds checking)
                y_start = max(0, center_y - rf_height // 2)
                y_end = min(frame_height, center_y + rf_height // 2)
                x_start = max(0, center_x - rf_width // 2)
                x_end = min(frame_width, center_x + rf_width // 2)
                
                receptive_fields.append({
                    'center': (center_x, center_y),
                    'bounds': (y_start, y_end, x_start, x_end),
                    'row': row,
                    'col': col
                })
        
        return receptive_fields
    
    def preprocess_frame(self, frame):
        """Preprocess frame before Gabor filtering"""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Optional: Gaussian blur to reduce noise
        if PROCESSING_CONFIG['gaussian_blur_kernel'] > 0:
            ksize = PROCESSING_CONFIG['gaussian_blur_kernel']
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        
        # Optional: Normalize contrast
        if PROCESSING_CONFIG['normalize_contrast']:
            gray = cv2.equalizeHist(gray)
        
        # Convert to float
        gray = gray.astype(np.float32) / 255.0
        
        return gray
    
    def extract_features(self, frame):
        """
        Extract Gabor features from frame
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            features: dict with keys:
                - 'responses': array of shape (324, 4) - responses for each neuron at each orientation
                - 'max_orientation': array of shape (324,) - preferred orientation index for each neuron
                - 'max_response': array of shape (324,) - maximum response across orientations
                - 'gabor_images': dict of filtered images for each orientation (for visualization)
        """
        # Preprocess
        gray = self.preprocess_frame(frame)
        
        # Initialize receptive fields if first frame
        if self.receptive_fields is None:
            self.receptive_fields = self._initialize_receptive_fields(
                gray.shape[0], gray.shape[1]
            )
        
        # Apply Gabor filters for each orientation
        gabor_responses = {}
        gabor_images = {}
        
        for orientation, kernel in self.gabor_kernels.items():
            # Convolve with Gabor kernel
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            
            # Take absolute value (energy)
            filtered = np.abs(filtered)
            
            gabor_responses[orientation] = filtered
            gabor_images[orientation] = filtered
        
        # Extract response for each receptive field
        neuron_responses = np.zeros((self.n_neurons, len(self.orientations)))
        
        for idx, rf in enumerate(self.receptive_fields):
            y1, y2, x1, x2 = rf['bounds']
            
            for ori_idx, orientation in enumerate(self.orientations):
                # Extract patch
                patch = gabor_responses[orientation][y1:y2, x1:x2]
                
                # Average response in receptive field (could also use max)
                response = np.mean(patch)
                neuron_responses[idx, ori_idx] = response
        
        # Find preferred orientation and max response for each neuron
        max_orientation_idx = np.argmax(neuron_responses, axis=1)
        max_response = np.max(neuron_responses, axis=1)
        
        return {
            'responses': neuron_responses,
            'max_orientation': max_orientation_idx,
            'max_response': max_response,
            'gabor_images': gabor_images,
            'receptive_fields': self.receptive_fields
        }
    
    def visualize_gabor_responses(self, gabor_images):
        """
        Create a visualization of Gabor filter responses
        Returns a single image with 4 orientations in a 2x2 grid
        """
        orientations = list(gabor_images.keys())
        
        # Normalize each image for visualization
        normalized = []
        for ori in orientations:
            img = gabor_images[ori]
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_norm = img_norm.astype(np.uint8)
            # Convert to BGR for color visualization
            img_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
            normalized.append(img_bgr)
        
        # Arrange in 2x2 grid
        top_row = np.hstack([normalized[0], normalized[1]])
        bottom_row = np.hstack([normalized[2], normalized[3]])
        grid = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, '0 deg', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(grid, '45 deg', (grid.shape[1]//2 + 10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(grid, '90 deg', (10, grid.shape[0]//2 + 30), font, 1, (0, 255, 0), 2)
        cv2.putText(grid, '135 deg', (grid.shape[1]//2 + 10, grid.shape[0]//2 + 30), font, 1, (0, 255, 0), 2)
        
        return grid
    
    def visualize_receptive_fields(self, frame, show_all=False):
        """
        Draw receptive field grid on frame
        
        Args:
            frame: Input frame to draw on
            show_all: If True, draw all RFs. If False, draw only grid lines
        """
        if self.receptive_fields is None:
            return frame
        
        overlay = frame.copy()
        
        if show_all:
            # Draw each receptive field
            for rf in self.receptive_fields:
                y1, y2, x1, x2 = rf['bounds']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
        else:
            # Just draw grid lines
            h, w = frame.shape[:2]
            row_step = h / self.grid_rows
            col_step = w / self.grid_cols
            
            for i in range(1, self.grid_rows):
                y = int(i * row_step)
                cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 1)
            
            for j in range(1, self.grid_cols):
                x = int(j * col_step)
                cv2.line(overlay, (x, 0), (x, h), (0, 255, 0), 1)
        
        return overlay

