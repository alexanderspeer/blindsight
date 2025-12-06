"""
V1 Model Interface
Wrapper for the NEST-based V1 cortex model from MDPI2021
"""

import sys
import os
import numpy as np
from config import V1_CONFIG

# Add V1 model path to Python path
model_path = os.path.join(os.path.dirname(__file__), V1_CONFIG['model_path'])
sys.path.insert(0, model_path)


class V1ModelInterface:
    """
    Interface to the V1 cortex model
    Handles initialization, spike injection, simulation, and output collection
    """
    
    def __init__(self):
        self.nest = None
        self.initialized = False
        self.columns = {}
        self.inputs = None
        self.lgn = None
        self.detectors = {}
        self.simulation_time = 0
        
        # Import NEST and check for custom module
        self._initialize_nest()
        
    def _initialize_nest(self):
        """Initialize NEST simulator and load custom module"""
        try:
            import nest
            import nest.raster_plot
            
            # Check if custom module is available
            if 'lifl_psc_exp_ie' not in nest.Models():
                print("Installing LIFL_IE module...")
                nest.Install('LIFL_IEmodule')
            
            self.nest = nest
            print("✓ NEST simulator initialized")
            
        except ImportError as e:
            raise RuntimeError(
                f"Could not import NEST simulator: {e}\n"
                "Make sure NEST is installed: pip install nest-simulator"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load LIFL_IE module: {e}\n"
                f"Make sure the module is compiled and in the correct path"
            )
    
    def setup_model(self):
        """
        Setup the V1 model architecture
        Creates 4 orientation columns (0°, 45°, 90°, 135°)
        """
        if self.initialized:
            print("Model already initialized. Resetting...")
            self.nest.ResetKernel()
        
        # Set NEST parameters
        self.nest.ResetKernel()
        self.nest.SetKernelStatus({'resolution': V1_CONFIG['nest_resolution']})
        
        # Create input spike generators (324 ganglion cells)
        n_cells = 324
        self.inputs = self.nest.Create("spike_generator", n_cells)
        
        # Create LGN layer (parrot neurons - relay spikes from retina)
        self.lgn = self.nest.Create('parrot_neuron', n_cells)
        self.nest.Connect(self.inputs, self.lgn, 'one_to_one')
        
        # Import the column function
        try:
            from OrientedColumnV1 import column
        except ImportError:
            raise RuntimeError(
                f"Could not import OrientedColumnV1 from {model_path}\n"
                "Make sure the V1 model files are in the correct location"
            )
        
        # Create columns for each orientation
        print("Creating V1 orientation columns...")
        for orientation in V1_CONFIG['columns']:
            print(f"  Creating {orientation}° column...")
            
            # column() returns: Detector, Spikes, Multimeter, SomaMultimeter, 
            #                   Pyr23, SS4, Pyr5, Pyr6, In23, In4, In5, In6
            result = column(orientation, self.lgn)
            
            self.columns[orientation] = {
                'detector': result[0],      # All spikes
                'spikes': result[1],        # Pyramidal spikes only
                'multimeter': result[2],    # Voltage recordings
                'soma_multimeter': result[3],  # IE parameter
                'pyr_23': result[4],        # Layer 2/3 pyramidal cells
                'ss_4': result[5],          # Layer 4 spiny stellate
                'pyr_5': result[6],         # Layer 5 pyramidal
                'pyr_6': result[7],         # Layer 6 pyramidal
                'inh_23': result[8],        # Layer 2/3 inhibitory
                'inh_4': result[9],         # Layer 4 inhibitory
                'inh_5': result[10],        # Layer 5 inhibitory
                'inh_6': result[11],        # Layer 6 inhibitory
            }
        
        self.initialized = True
        self.simulation_time = 0
        
        print(f"✓ V1 model initialized with {len(self.columns)} orientation columns")
    
    def inject_spikes(self, spike_trains_nest):
        """
        Inject spike trains into the model
        
        Args:
            spike_trains_nest: List of dicts from SpikeEncoder.format_for_nest()
                [{'neuron_id': X, 'spike_times': [t1, t2, ...]}, ...]
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        current_time = self.nest.GetKernelStatus('time')
        
        # Set spike times for each input neuron
        for neuron_data in spike_trains_nest:
            neuron_id = neuron_data['neuron_id']
            spike_times = neuron_data['spike_times']
            
            if neuron_id >= len(self.inputs):
                continue  # Skip invalid neuron IDs
            
            # Offset spike times by current simulation time
            offset_times = [t + current_time for t in spike_times]
            
            # Set spike times for this neuron
            self.nest.SetStatus([self.inputs[neuron_id]], 
                               {'spike_times': offset_times})
    
    def run_simulation(self, warmup=True, stimulus_duration=None):
        """
        Run the simulation
        
        Args:
            warmup: If True, run warmup period first
            stimulus_duration: Duration of stimulus (ms), default from config
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        # Warmup period (initialize network with noise)
        if warmup and self.simulation_time == 0:
            warmup_time = V1_CONFIG['warmup_time_ms']
            print(f"Running warmup ({warmup_time}ms)...")
            self.nest.Simulate(warmup_time)
            self.simulation_time += warmup_time
        
        # Stimulus presentation
        if stimulus_duration is None:
            stimulus_duration = V1_CONFIG['stimulus_time_ms']
        
        print(f"Running simulation ({stimulus_duration}ms)...")
        self.nest.Simulate(stimulus_duration)
        self.simulation_time += stimulus_duration
    
    def get_output(self, start_time=None):
        """
        Get spike output from all columns
        
        Args:
            start_time: Only return spikes after this time (ms)
                       If None, use time at start of last stimulus
        
        Returns:
            dict with structure:
            {
                orientation: {
                    'layer_23': {'senders': [...], 'times': [...]},
                    'layer_4': {...},
                    'layer_5': {...},
                    'layer_6': {...},
                    'all': {...}  # All spikes from this column
                }
            }
        """
        if not self.initialized:
            raise RuntimeError("Model not initialized.")
        
        if start_time is None:
            # Use time at start of last stimulus
            start_time = self.simulation_time - V1_CONFIG['stimulus_time_ms']
        
        output = {}
        
        for orientation, column_data in self.columns.items():
            # Get spikes from pyramidal cells only (from 'spikes' detector)
            spike_events = self.nest.GetStatus(column_data['spikes'])[0]['events']
            
            # Filter by time
            mask = spike_events['times'] > start_time
            senders = spike_events['senders'][mask]
            times = spike_events['times'][mask]
            
            # Separate by layer
            layer_data = {}
            
            # Layer 2/3
            pyr23_mask = np.isin(senders, column_data['pyr_23'])
            layer_data['layer_23'] = {
                'senders': senders[pyr23_mask],
                'times': times[pyr23_mask]
            }
            
            # Layer 5
            pyr5_mask = np.isin(senders, column_data['pyr_5'])
            layer_data['layer_5'] = {
                'senders': senders[pyr5_mask],
                'times': times[pyr5_mask]
            }
            
            # Layer 6
            pyr6_mask = np.isin(senders, column_data['pyr_6'])
            layer_data['layer_6'] = {
                'senders': senders[pyr6_mask],
                'times': times[pyr6_mask]
            }
            
            # Layer 4 (from full detector since SS cells aren't in 'spikes')
            all_events = self.nest.GetStatus(column_data['detector'])[0]['events']
            mask_4 = all_events['times'] > start_time
            senders_4 = all_events['senders'][mask_4]
            times_4 = all_events['times'][mask_4]
            ss4_mask = np.isin(senders_4, column_data['ss_4'])
            layer_data['layer_4'] = {
                'senders': senders_4[ss4_mask],
                'times': times_4[ss4_mask]
            }
            
            # All spikes from this column
            layer_data['all'] = {
                'senders': senders,
                'times': times
            }
            
            output[orientation] = layer_data
        
        return output
    
    def calculate_orientation_selectivity(self, output):
        """
        Calculate orientation selectivity index for each spatial location
        
        Args:
            output: Output from get_output()
            
        Returns:
            dict with:
                'preferred_orientation': (324,) array of preferred orientations
                'selectivity_index': (324,) array of OSI values
                'firing_rates': dict of {orientation: (324,) array}
        """
        # Count spikes in Layer 2/3 for each orientation
        firing_rates = {}
        
        for orientation in self.columns.keys():
            layer23_data = output[orientation]['layer_23']
            
            # Count spikes per neuron (Layer 2/3 has 324 neurons)
            counts = np.zeros(324)
            
            if len(layer23_data['senders']) > 0:
                # Map GIDs to 0-323 index
                pyr23_gids = self.columns[orientation]['pyr_23']
                gid_to_idx = {gid: idx for idx, gid in enumerate(pyr23_gids)}
                
                for sender in layer23_data['senders']:
                    if sender in gid_to_idx:
                        counts[gid_to_idx[sender]] += 1
            
            firing_rates[orientation] = counts
        
        # Stack into array: (4, 324)
        rate_array = np.stack([firing_rates[ori] for ori in sorted(self.columns.keys())])
        
        # Find preferred orientation
        preferred_orientation = np.argmax(rate_array, axis=0)
        
        # Calculate OSI (simplified version)
        max_rate = np.max(rate_array, axis=0)
        mean_rate = np.mean(rate_array, axis=0)
        
        # Avoid division by zero
        selectivity_index = np.zeros(324)
        mask = (max_rate + mean_rate) > 0
        selectivity_index[mask] = (max_rate[mask] - mean_rate[mask]) / (max_rate[mask] + mean_rate[mask])
        
        return {
            'preferred_orientation': preferred_orientation,
            'selectivity_index': selectivity_index,
            'firing_rates': firing_rates
        }
    
    def reset(self):
        """Reset the simulation"""
        if self.nest is not None:
            self.nest.ResetKernel()
        self.initialized = False
        self.simulation_time = 0
        print("✓ V1 model reset")

