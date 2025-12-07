"""
Real-time V1 Vision Pipeline
Processes video stream from Raspberry Pi camera in real-time
"""
'''
rpicam-vid \
    --codec h264 \
    --profile baseline \
    --level 4.2 \
    --width 320 \
    --height 240 \
    --framerate 15 \
    --bitrate 1500000 \
    --intra 15 \
    --inline \
    --low-latency \
    --libav-video-codec-opts "tune=zerolatency;preset=ultrafast" \
    --sharpness 1.2 \
    --contrast 1.1 \
    --denoise off \
    --exposure sport \
    --shutter 8000 \
    --awb custom \
    --awbgains 1.0,1.0 \
    --timeout 0 \
    --buffer-count 2 \
    --flush \
    -n \
    --listen \
    -o tcp://0.0.0.0:5001
'''
import cv2
import sys
from pipeline import V1VisionPipeline
from config import VIDEO_CONFIG, GRID_CONFIG, V1_ARCHITECTURE, DEBUG_CONFIG


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
    
    print(f"\n⚡ OPTIMIZATIONS ENABLED:")
    print(f"   Grid: {GRID_CONFIG['grid_rows']}x{GRID_CONFIG['grid_cols']} = {GRID_CONFIG['n_neurons']} neurons")
    print(f"   Time step: {V1_ARCHITECTURE['dt']} ms")
    print(f"   Simulation: {V1_ARCHITECTURE['warmup_time_ms']}ms warmup + {V1_ARCHITECTURE['stimulus_time_ms']}ms stimulus")
    
    # Show debug configuration
    if DEBUG_CONFIG['enabled']:
        print(f"\n🐛 DEBUG MODE ENABLED:")
        print(f"   Printing debug info every {DEBUG_CONFIG['print_every_n_frames']} frame(s)")
        print(f"   Gabor stats: {DEBUG_CONFIG['show_gabor_stats']}")
        print(f"   Spike stats: {DEBUG_CONFIG['show_spike_stats']}")
        print(f"   V1 layer stats: {DEBUG_CONFIG['show_v1_layer_stats']}")
        print(f"   Decoder stats: {DEBUG_CONFIG['show_decoder_stats']}")
        print(f"   Array shapes: {DEBUG_CONFIG['show_array_shapes']}")
        print(f"   Distributions: {DEBUG_CONFIG['show_distributions']}")
        print(f"   Check NaNs: {DEBUG_CONFIG['check_for_nans']}")
        print(f"   Check zeros: {DEBUG_CONFIG['check_for_zeros']}")
    else:
        print(f"\n   Debug mode: OFF")
    
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

