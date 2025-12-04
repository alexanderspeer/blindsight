"""
Test V1 Pipeline with Static Image
Quick test to verify all components work correctly
"""

import cv2
import numpy as np
from pipeline import V1VisionPipeline


def create_test_image():
    """Create a test image with oriented edges"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw oriented lines
    # Horizontal lines (0 degrees)
    for y in range(50, 150, 20):
        cv2.line(img, (50, y), (200, y), (255, 255, 255), 2)
    
    # Vertical lines (90 degrees)
    for x in range(250, 350, 20):
        cv2.line(img, (x, 50), (x, 200), (255, 255, 255), 2)
    
    # Diagonal lines (45 degrees)
    for offset in range(-100, 100, 20):
        cv2.line(img, (400 + offset, 50), (550 + offset, 200), (255, 255, 255), 2)
    
    # Diagonal lines (-45 degrees / 135 degrees)
    for offset in range(-100, 100, 20):
        cv2.line(img, (400 + offset, 250), (550 + offset, 400), (255, 255, 255), 2)
    
    # Add some shapes
    cv2.rectangle(img, (50, 250), (150, 350), (255, 255, 255), 2)
    cv2.circle(img, (300, 300), 50, (255, 255, 255), 2)
    
    # Add text
    cv2.putText(img, 'V1 Test Image', (50, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def main():
    """Test pipeline with static image"""
    print("\n" + "=" * 60)
    print("🧪 V1 COMPUTATIONAL MODEL - STATIC IMAGE TEST")
    print("=" * 60 + "\n")
    
    # Create test image
    print("📷 Creating test image...")
    test_image = create_test_image()
    
    # Initialize pipeline
    pipeline = V1VisionPipeline()
    
    # Process image
    print("\n⚙️  Processing image through V1 pipeline...")
    print("-" * 60)
    results = pipeline.process_frame(test_image, warmup=True)
    
    # Print timing
    print("\n⏱️  Performance:")
    for stage, time_ms in results['timing'].items():
        print(f"   {stage:12s}: {time_ms*1000:7.2f} ms")
    
    # Print V1 activity
    print("\n🧠 V1 Activity:")
    for orientation in [0, 45, 90, 135]:
        layer_23 = results['v1_results']['orientations'][orientation]['layer_23']
        print(f"   {orientation:3d}° column: {layer_23['mean_rate']:.2f} Hz (Layer 2/3)")
    
    # Calculate OSI
    print("\n📊 Orientation Selectivity:")
    osi_values = pipeline.v1_model.calculate_orientation_selectivity_index(results['v1_results'])
    for orientation, osi_data in osi_values.items():
        print(f"   {orientation:3d}° column: OSI = {osi_data['mean_osi']:.3f}")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    visualizations = pipeline.visualize_pipeline(results)
    
    # Display
    print("\n✅ Displaying results (press any key to close)...\n")
    pipeline.display_visualizations(visualizations)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

