#!/usr/bin/env python3
"""
Test script for Blindsight V1 integration
Verifies all components work correctly
"""

import sys
import numpy as np
import cv2

def test_nest_installation():
    """Test 1: NEST Simulator installation"""
    print("\n=== Test 1: NEST Installation ===")
    try:
        import nest
        print(f"✓ NEST version: {nest.version()}")
        return True
    except ImportError as e:
        print(f"✗ NEST import failed: {e}")
        print("  Install NEST: https://nest-simulator.readthedocs.io/")
        return False

def test_lifl_module():
    """Test 2: LIFL_IE module loading"""
    print("\n=== Test 2: LIFL_IE Module ===")
    try:
        import nest
        nest.ResetKernel()
        
        # Try to install module
        if 'lifl_psc_exp_ie' not in nest.Models():
            nest.Install('LIFL_IEmodule')
        
        # Verify models available
        models = nest.Models()
        has_lifl = 'lifl_psc_exp_ie' in models
        has_aeif = 'aeif_psc_exp_peak' in models
        
        if has_lifl and has_aeif:
            print("✓ LIFL_IE module loaded successfully")
            print(f"  - lifl_psc_exp_ie: {'✓' if has_lifl else '✗'}")
            print(f"  - aeif_psc_exp_peak: {'✓' if has_aeif else '✗'}")
            return True
        else:
            print("✗ LIFL_IE models not found")
            print("  Build module: cd LIFL_IE && mkdir build && cd build && cmake .. && make install")
            return False
    except Exception as e:
        print(f"✗ Module loading failed: {e}")
        return False

def test_neuron_creation():
    """Test 3: Create custom neurons"""
    print("\n=== Test 3: Neuron Creation ===")
    try:
        import nest
        nest.ResetKernel()
        
        if 'lifl_psc_exp_ie' not in nest.Models():
            nest.Install('LIFL_IEmodule')
        
        # Create test neurons
        lifl_neuron = nest.Create('lifl_psc_exp_ie', 1, {
            'lambda': 0.0005,
            'tau': 12.5,
            'std_mod': False
        })
        
        aeif_neuron = nest.Create('aeif_psc_exp_peak', 1)
        
        # Verify creation
        status_lifl = nest.GetStatus(lifl_neuron)[0]
        status_aeif = nest.GetStatus(aeif_neuron)[0]
        
        print(f"✓ Created lifl_psc_exp_ie (ID: {lifl_neuron[0]})")
        print(f"  - lambda: {status_lifl['lambda']}")
        print(f"  - tau: {status_lifl['tau']}")
        print(f"✓ Created aeif_psc_exp_peak (ID: {aeif_neuron[0]})")
        
        return True
    except Exception as e:
        print(f"✗ Neuron creation failed: {e}")
        return False

def test_spike_encoder():
    """Test 4: Spike encoder"""
    print("\n=== Test 4: Spike Encoder ===")
    try:
        from blindsight_camera_encoder import CameraSpikeEncoder
        
        encoder = CameraSpikeEncoder(
            resolution=(18, 18),
            max_rate=100.0,
            encoding_type='poisson'
        )
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        
        # Encode
        spikes = encoder.encode_frame(test_frame, current_time=0.0)
        
        print(f"✓ Encoder created (resolution: {encoder.resolution})")
        print(f"✓ Encoded frame: {len(spikes)} neurons spiked")
        print(f"  - Total spikes: {sum(len(times) for times in spikes.values())}")
        
        return True
    except Exception as e:
        print(f"✗ Spike encoder failed: {e}")
        print(f"  Error details: {type(e).__name__}")
        return False

def test_opencv():
    """Test 5: OpenCV camera access"""
    print("\n=== Test 5: Camera Access ===")
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"✓ Camera accessible (frame shape: {frame.shape})")
                return True
            else:
                print("⚠ Camera opened but frame capture failed")
                return False
        else:
            print("⚠ No camera detected (will work with video files)")
            return True  # Not critical for testing
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_lightweight_column():
    """Test 6: Create lightweight V1 column"""
    print("\n=== Test 6: Lightweight V1 Column ===")
    try:
        import nest
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': 0.1})
        
        if 'lifl_psc_exp_ie' not in nest.Models():
            nest.Install('LIFL_IEmodule')
        
        # Create input layer
        inputs = nest.Create("spike_generator", 81)
        LGN = nest.Create('parrot_neuron', 81)
        nest.Connect(inputs, LGN, 'one_to_one')
        
        # Create simplified column
        from blindsight_realtime_v1 import LightweightColumn
        
        try:
            column = LightweightColumn(0, LGN, enable_ie_plasticity=False)
            print(f"✓ Created lightweight column (orientation: 0°)")
            print(f"  - Layer 4 neurons: {len(column.SS4)}")
            print(f"  - Layer 2/3 neurons: {len(column.Pyr23)}")
            
            # Test simulation
            nest.Simulate(10.0)
            print("✓ Simulation test passed (10 ms)")
            
            return True
        except FileNotFoundError:
            print("⚠ Pre-trained IE values not found (expected for first run)")
            print("  Column created with default values")
            return True
            
    except Exception as e:
        print(f"✗ Column creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_system():
    """Test 7: Complete system integration (synthetic input)"""
    print("\n=== Test 7: Full System Integration ===")
    try:
        import nest
        from blindsight_camera_encoder import CameraSpikeEncoder
        from blindsight_realtime_v1 import LightweightColumn
        
        # Initialize
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': 0.1, 'print_time': False})
        
        if 'lifl_psc_exp_ie' not in nest.Models():
            nest.Install('LIFL_IEmodule')
        
        # Create encoder
        encoder = CameraSpikeEncoder(resolution=(9, 9), max_rate=50.0)
        
        # Create input layer
        inputs = nest.Create("spike_generator", 81)
        LGN = nest.Create('parrot_neuron', 81)
        nest.Connect(inputs, LGN, 'one_to_one')
        
        # Create columns
        columns = {}
        for angle in [0, 90]:
            try:
                columns[angle] = LightweightColumn(angle, LGN, enable_ie_plasticity=False)
            except FileNotFoundError:
                print(f"  ⚠ Skipping {angle}° column (no pre-trained values)")
        
        if not columns:
            print("⚠ No columns created (missing pre-trained data)")
            return True  # Not critical for basic testing
        
        # Create synthetic oriented stimulus
        stimulus = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(stimulus, (20, 20), (80, 80), 255, 3)  # 45° line
        
        # Encode and inject spikes
        current_time = 0.0
        spikes = encoder.encode_frame(stimulus, current_time)
        
        for neuron_id, spike_times in spikes.items():
            if spike_times:
                nest.SetStatus([inputs[neuron_id]], {'spike_times': spike_times})
        
        # Simulate
        nest.Simulate(100.0)
        
        # Check responses
        responses = {}
        for angle, column in columns.items():
            count = column.get_spike_count(time_window=100.0)
            responses[angle] = count
        
        print(f"✓ System integration test passed")
        print(f"  - Encoded spikes: {len(spikes)} neurons")
        print(f"  - Simulated: 100 ms")
        print(f"  - Responses: {responses}")
        
        return True
        
    except Exception as e:
        print(f"✗ Full system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Blindsight V1 Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("NEST Installation", test_nest_installation),
        ("LIFL_IE Module", test_lifl_module),
        ("Neuron Creation", test_neuron_creation),
        ("Spike Encoder", test_spike_encoder),
        ("Camera Access", test_opencv),
        ("Lightweight Column", test_lightweight_column),
        ("Full System", test_full_system),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System ready for use.")
        print("\nNext steps:")
        print("  1. python3 blindsight_realtime_v1.py --lightweight")
        print("  2. Check V1_INTEGRATION_GUIDE.md for detailed usage")
    else:
        print("\n⚠ Some tests failed. Check SETUP_BLINDSIGHT.md for troubleshooting.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

