"""
Real-time V1 Vision Pipeline
Processes video stream from Raspberry Pi camera in real-time
"""

import cv2
import sys
from pipeline import V1VisionPipeline
from config import VIDEO_CONFIG


def main():
    """Run real-time V1 vision pipeline"""
    print("\n" + "=" * 60)
    print("🧠 COMPUTATIONAL V1 VISION - REAL-TIME MODE")
    print("=" * 60 + "\n")
    
    # Check if video file specified
    video_source = None
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        print(f"📹 Video source: {video_source}")
    else:
        print(f"📡 Connecting to Raspberry Pi camera...")
        print(f"   IP: {VIDEO_CONFIG['pi_ip']}")
        print(f"   Port: {VIDEO_CONFIG['port']}")
        print(f"   Resolution: {VIDEO_CONFIG['width']}x{VIDEO_CONFIG['height']}")
    
    # Initialize pipeline
    pipeline = V1VisionPipeline()
    
    # Run pipeline
    print("\n🚀 Starting real-time processing...")
    print("   Press 'q' to quit\n")
    
    try:
        pipeline.run_on_video_stream(video_source)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print(f"✅ Pipeline stopped. Processed {pipeline.frame_count} frames.")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

