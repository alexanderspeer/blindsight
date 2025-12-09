"""
Test V1 Pipeline with Static Image
Quick test to verify all components work correctly
"""

import cv2
import numpy as np
from pipeline import V1VisionPipeline
from config import V1_ARCHITECTURE, DEBUG_CONFIG


def create_test_image():
    """Create a test image with oriented edges"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # horizontal lines
    for y in range(50, 150, 20):
        cv2.line(img, (50, y), (200, y), (255, 255, 255), 2)
    
    # vertical lines
    for x in range(250, 350, 20):
        cv2.line(img, (x, 50), (x, 200), (255, 255, 255), 2)
    
    # diagonal lines at 45 degrees
    for offset in range(-100, 100, 20):
        cv2.line(img, (400 + offset, 50), (550 + offset, 200), (255, 255, 255), 2)
    
    # diagonal lines at 135 degrees
    for offset in range(-100, 100, 20):
        cv2.line(img, (400 + offset, 250), (550 + offset, 400), (255, 255, 255), 2)
    
    cv2.rectangle(img, (50, 250), (150, 350), (255, 255, 255), 2)
    cv2.circle(img, (300, 300), 50, (255, 255, 255), 2)
    cv2.putText(img, 'V1 Test Image', (50, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def print_gabor_statistics(features):
    """Print detailed Gabor feature statistics"""
    print("\nGABOR FEATURE MAPS - DETAILED STATISTICS")
    
    for orientation in sorted(features.keys()):
        grid = features[orientation]
        print(f"\n{orientation}° Orientation:")
        print(f"  Shape: {grid.shape}")
        print(f"  Min: {grid.min():.4f}, Max: {grid.max():.4f}")
        print(f"  Mean: {grid.mean():.4f}, Std: {grid.std():.4f}")
        print(f"  Median: {np.median(grid):.4f}")
        
        # count active cells
        total_cells = grid.size
        active_001 = (grid > 0.01).sum()
        active_01 = (grid > 0.1).sum()
        active_05 = (grid > 0.5).sum()
        active_10 = (grid > 1.0).sum()
        
        print(f"  Active cells:")
        print(f"    > 0.01: {active_001}/{total_cells} ({active_001/total_cells*100:.1f}%)")
        print(f"    > 0.1:  {active_01}/{total_cells} ({active_01/total_cells*100:.1f}%)")
        print(f"    > 0.5:  {active_05}/{total_cells} ({active_05/total_cells*100:.1f}%)")
        print(f"    > 1.0:  {active_10}/{total_cells} ({active_10/total_cells*100:.1f}%)")
        
        # value distribution
        bins = [0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, np.inf]
        hist, _ = np.histogram(grid, bins=bins)
        print(f"  Value distribution:")
        for i in range(len(hist)):
            if i < len(bins) - 1:
                print(f"    [{bins[i]:.2f} - {bins[i+1]:.2f}): {hist[i]} cells")


def print_spike_statistics(spike_trains):
    """Print detailed spike train statistics"""
    print("\nSPIKE ENCODING - DETAILED STATISTICS")
    
    total_spikes = 0
    for orientation in sorted(spike_trains.keys()):
        spike_data = spike_trains[orientation]
        n_spikes = len(spike_data['neuron_ids'])
        total_spikes += n_spikes
        
        print(f"\n{orientation}° Orientation:")
        print(f"  Total spikes: {n_spikes}")
        
        if n_spikes > 0:
            unique_neurons = len(np.unique(spike_data['neuron_ids']))
            print(f"  Unique neurons: {unique_neurons}/144 ({unique_neurons/144*100:.1f}%)")
            print(f"  Spike times:")
            print(f"    Min: {spike_data['spike_times'].min():.2f} ms")
            print(f"    Max: {spike_data['spike_times'].max():.2f} ms")
            print(f"    Mean: {spike_data['spike_times'].mean():.2f} ms")
            print(f"    Median: {np.median(spike_data['spike_times']):.2f} ms")
            
            # timing breakdown
            early = (spike_data['spike_times'] < 100).sum()
            mid = ((spike_data['spike_times'] >= 100) & (spike_data['spike_times'] < 150)).sum()
            late = (spike_data['spike_times'] >= 150).sum()
            print(f"  Timing distribution:")
            print(f"    Early (<100ms): {early} ({early/n_spikes*100:.1f}%)")
            print(f"    Mid (100-150ms): {mid} ({mid/n_spikes*100:.1f}%)")
            print(f"    Late (>150ms): {late} ({late/n_spikes*100:.1f}%)")
        else:
            print(f"  WARNING: No spikes generated!")
    
    print(f"\n  TOTAL SPIKES: {total_spikes}")


def print_v1_statistics(v1_results):
    """Print detailed V1 layer statistics"""
    print("\nV1 LAYER ACTIVITY - DETAILED STATISTICS")
    
    # stats for each layer
    for layer_name in ['layer_4', 'layer_23', 'layer_5', 'layer_6']:
        print(f"\n{layer_name.upper()}:")
        
        all_rates = []
        for orientation in sorted(v1_results['orientations'].keys()):
            rates = v1_results['orientations'][orientation][layer_name]['firing_rates']
            all_rates.extend(rates)
        
        all_rates = np.array(all_rates)
        
        print(f"  Total neurons: {len(all_rates)}")
        print(f"  Mean rate: {all_rates.mean():.2f} Hz")
        print(f"  Median rate: {np.median(all_rates):.2f} Hz")
        print(f"  Min: {all_rates.min():.2f} Hz, Max: {all_rates.max():.2f} Hz")
        print(f"  Std: {all_rates.std():.2f} Hz")
        
        # active neuron counts
        active_1 = (all_rates > 1.0).sum()
        active_5 = (all_rates > 5.0).sum()
        active_10 = (all_rates > 10.0).sum()
        print(f"  Active neurons:")
        print(f"    > 1 Hz:  {active_1}/{len(all_rates)} ({active_1/len(all_rates)*100:.1f}%)")
        print(f"    > 5 Hz:  {active_5}/{len(all_rates)} ({active_5/len(all_rates)*100:.1f}%)")
        print(f"    > 10 Hz: {active_10}/{len(all_rates)} ({active_10/len(all_rates)*100:.1f}%)")
    
    # layer 2/3 by orientation
    print(f"\nLAYER 2/3 BY ORIENTATION:")
    for orientation in sorted(v1_results['orientations'].keys()):
        rate = v1_results['orientations'][orientation]['layer_23']['mean_rate']
        rates = v1_results['orientations'][orientation]['layer_23']['firing_rates']
        active = (rates > 1.0).sum()
        print(f"  {orientation:3d}°: {rate:5.2f} Hz (active: {active}/{len(rates)})")


def print_v1_configuration():
    """Print current V1 model configuration"""
    print("\nV1 MODEL CONFIGURATION")
    
    print("\nFEEDFORWARD WEIGHTS:")
    print(f"  LGN → L4:    {V1_ARCHITECTURE['lgn_to_ss4_weight']:.1f}")
    print(f"  L4 → L2/3:   {V1_ARCHITECTURE['weight_L4_to_L23']:.1f}")
    print(f"  L2/3 → L5:   {V1_ARCHITECTURE['weight_L23_to_L5']:.1f}")
    print(f"  L5 → L6:     {V1_ARCHITECTURE['weight_L5_to_L6']:.1f}")
    
    print("\nLAYER 2/3 PARAMETERS:")
    print(f"  Threshold:    {V1_ARCHITECTURE['L23_v_threshold']:.1f} mV (default: {V1_ARCHITECTURE['v_threshold']:.1f} mV)")
    print(f"  Tau_m:        {V1_ARCHITECTURE['L23_tau_membrane']:.1f} ms (default: {V1_ARCHITECTURE['tau_membrane']:.1f} ms)")
    print(f"  Bias current: {V1_ARCHITECTURE['L23_bias_current']:.1f} pA")
    
    print("\nDEBUG SETTINGS:")
    print(f"  Synaptic currents: {DEBUG_CONFIG.get('show_synaptic_currents', False)}")
    print(f"  V1 layer stats:    {DEBUG_CONFIG.get('show_v1_layer_stats', False)}")
    


def main():
    """Test pipeline with static image"""
    print("\nV1 COMPUTATIONAL MODEL - STATIC IMAGE TEST\n")
    
    print_v1_configuration()
    
    print("\nCreating test image...")
    test_image = create_test_image()
    
    print("\nInitializing pipeline...")
    pipeline = V1VisionPipeline()
    
    print("\nProcessing image...")
    results = pipeline.process_frame(test_image, warmup=True)
    
    print_gabor_statistics(results['gabor_features'])
    print_spike_statistics(results['spike_trains'])
    print_v1_statistics(results['v1_results'])
    
    print("\nPERFORMANCE TIMING")
    for stage, time_ms in results['timing'].items():
        print(f"  {stage:12s}: {time_ms*1000:7.2f} ms")
    print(f"  Total FPS: {1.0/results['timing']['total']:.2f}")
    
    print("\nORIENTATION SELECTIVITY INDEX (OSI)")
    osi_values = pipeline.v1_model.calculate_orientation_selectivity_index(results['v1_results'])
    for orientation, osi_data in osi_values.items():
        print(f"  {orientation:3d}° column: OSI = {osi_data['mean_osi']:.3f}")
    
    print("\nGenerating visualizations...")
    visualizations = pipeline.visualize_pipeline(results)
    
    print("Displaying results (press any key to close)...\n")
    pipeline.display_visualizations(visualizations)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nTest complete!")
    
    print("\nTo enable synaptic current debugging, edit config.py:")
    print("  'show_synaptic_currents': True")
    print("  (warning: very verbose output)\n")


if __name__ == '__main__':
    main()

