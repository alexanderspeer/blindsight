"""
Test V1 Pipeline with Static Image
Useful for debugging and understanding the pipeline without video stream
"""

import cv2
import numpy as np
import sys
import time

from gabor_feature_extractor import GaborFeatureExtractor
from spike_encoder import SpikeEncoder
from v1_model_interface import V1ModelInterface
from v1_decoder import V1Decoder
from visualization import SpikeRasterPlot


def create_test_image(image_type='edges'):
    """
    Create a test image with known patterns
    
    Args:
        image_type: 'edges', 'bars', or 'checkerboard'
    """
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    
    if image_type == 'edges':
        # Vertical edge
        img[:, 160:240, :] = 255
        # Horizontal edge
        img[120:180, :, :] = 255
        # Diagonal edge
        for i in range(360):
            j = int(i * 640 / 360)
            if j < 640:
                cv2.line(img, (j-10, i), (j+10, i), (255, 255, 255), 2)
    
    elif image_type == 'bars':
        # Oriented bars
        # Horizontal bars (0°)
        for y in range(0, 180, 40):
            cv2.rectangle(img, (0, y), (160, y+20), (255, 255, 255), -1)
        
        # 45° bars
        for i in range(-360, 360, 40):
            pt1 = (160, i)
            pt2 = (320, i+200)
            cv2.line(img, pt1, pt2, (255, 255, 255), 15)
        
        # Vertical bars (90°)
        for x in range(320, 480, 40):
            cv2.rectangle(img, (x, 0), (x+20, 360), (255, 255, 255), -1)
        
        # 135° bars
        for i in range(-360, 640, 40):
            pt1 = (i, 0)
            pt2 = (i+200, 200)
            cv2.line(img, pt1, pt2, (255, 255, 255), 15)
    
    elif image_type == 'checkerboard':
        # Checkerboard pattern
        square_size = 40
        for i in range(0, 360, square_size):
            for j in range(0, 640, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    cv2.rectangle(img, (j, i), 
                                (j+square_size, i+square_size), 
                                (255, 255, 255), -1)
    
    return img


def test_pipeline(image_path=None, image_type='bars'):
    """
    Test the pipeline with a static image
    
    Args:
        image_path: Path to image file, or None to use generated test image
        image_type: Type of generated test image if image_path is None
    """
    print("=" * 60)
    print("V1 Pipeline Static Image Test")
    print("=" * 60)
    
    # Load or create image
    if image_path:
        print(f"\nLoading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return
    else:
        print(f"\nGenerating test image: {image_type}")
        img = create_test_image(image_type)
    
    print(f"Image size: {img.shape}")
    
    # Initialize components
    print("\nInitializing components...")
    gabor_extractor = GaborFeatureExtractor()
    spike_encoder = SpikeEncoder()
    v1_model = V1ModelInterface()
    v1_decoder = V1Decoder()
    spike_plotter = SpikeRasterPlot()
    
    print("Setting up V1 model...")
    v1_model.setup_model()
    
    # Process image
    print("\n" + "-" * 60)
    print("Processing image through pipeline...")
    print("-" * 60)
    
    # Stage 1: Extract Gabor features
    print("\n[1/5] Extracting Gabor features...")
    start = time.time()
    features = gabor_extractor.extract_features(img)
    elapsed = (time.time() - start) * 1000
    print(f"  ✓ Complete ({elapsed:.1f} ms)")
    print(f"  Responses shape: {features['responses'].shape}")
    print(f"  Max response: {np.max(features['max_response']):.3f}")
    
    # Create Gabor visualization
    gabor_viz = gabor_extractor.visualize_gabor_responses(features['gabor_images'])
    
    # Stage 2: Encode to spikes
    print("\n[2/5] Encoding to spike trains...")
    start = time.time()
    spike_trains = spike_encoder.encode(features, current_time_ms=0)
    elapsed = (time.time() - start) * 1000
    spike_stats = spike_encoder.get_spike_statistics(spike_trains)
    print(f"  ✓ Complete ({elapsed:.1f} ms)")
    print(f"  Generated {spike_stats['n_spikes']} spikes")
    print(f"  Active neurons: {spike_stats['n_active_neurons']}/324")
    print(f"  Mean rate: {spike_stats['mean_rate']:.1f} Hz")
    
    # Stage 3: Run V1 simulation
    print("\n[3/5] Running V1 simulation...")
    start = time.time()
    spike_trains_nest = spike_encoder.format_for_nest(spike_trains)
    v1_model.inject_spikes(spike_trains_nest)
    v1_model.run_simulation(warmup=True)
    elapsed = (time.time() - start) * 1000
    print(f"  ✓ Complete ({elapsed:.1f} ms)")
    
    # Get output
    v1_output = v1_model.get_output()
    selectivity = v1_model.calculate_orientation_selectivity(v1_output)
    
    # Print orientation statistics
    print("\n  V1 Orientation Statistics:")
    for ori in [0, 45, 90, 135]:
        rate = selectivity['firing_rates'][ori]
        mean_rate = np.mean(rate)
        active = np.sum(rate > 0)
        print(f"    {ori:3d}°: {mean_rate:5.1f} spikes/neuron (avg), {active:3d}/324 active")
    
    # Stage 4: Decode output
    print("\n[4/5] Decoding V1 output...")
    start = time.time()
    decoded = v1_decoder.decode(v1_output, selectivity)
    elapsed = (time.time() - start) * 1000
    print(f"  ✓ Complete ({elapsed:.1f} ms)")
    
    # Stage 5: Create visualizations
    print("\n[5/5] Creating visualizations...")
    v1_viz = v1_decoder.create_combined_visualization(decoded, img)
    spike_viz = spike_plotter.plot(spike_trains, 0)
    analysis = v1_decoder.create_detailed_analysis(v1_output, selectivity)
    
    print("  ✓ Complete")
    
    # Display results
    print("\n" + "=" * 60)
    print("Displaying results (press any key to close)...")
    print("=" * 60)
    
    # Add receptive field grid to original
    img_with_grid = gabor_extractor.visualize_receptive_fields(img, show_all=False)
    
    # Create windows
    cv2.namedWindow('1. Original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('2. Gabor Features', cv2.WINDOW_NORMAL)
    cv2.namedWindow('3. Spike Trains', cv2.WINDOW_NORMAL)
    cv2.namedWindow('4. V1 Output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('5. Firing Rate Analysis', cv2.WINDOW_NORMAL)
    
    # Position windows
    cv2.resizeWindow('1. Original Image', 640, 360)
    cv2.resizeWindow('2. Gabor Features', 800, 400)
    cv2.resizeWindow('3. Spike Trains', 800, 400)
    cv2.resizeWindow('4. V1 Output', 1000, 400)
    cv2.resizeWindow('5. Firing Rate Analysis', 600, 300)
    
    # Show images
    cv2.imshow('1. Original Image', img_with_grid)
    cv2.imshow('2. Gabor Features', gabor_viz)
    cv2.imshow('3. Spike Trains', spike_viz)
    cv2.imshow('4. V1 Output', v1_viz)
    cv2.imshow('5. Firing Rate Analysis', analysis['firing_rate_chart'])
    
    print("\nPress any key in any window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✓ Test complete!")


def main():
    """Entry point"""
    if len(sys.argv) > 1:
        # Use provided image
        image_path = sys.argv[1]
        test_pipeline(image_path=image_path)
    else:
        # Use generated test image
        print("\nNo image provided. Using generated test image.")
        print("Usage: python test_static_image.py [path/to/image.jpg]")
        print("\nGenerating test image with oriented bars...")
        test_pipeline(image_path=None, image_type='bars')


if __name__ == "__main__":
    main()

