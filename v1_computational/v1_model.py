"""
Complete Computational V1 Model
Combines 4 orientation columns (0°, 45°, 90°, 135°) exactly like MDPI2021
"""

import numpy as np
from v1_column import V1OrientationColumn
from config import V1_ARCHITECTURE, GABOR_CONFIG, GRID_CONFIG


class ComputationalV1Model:
    """
    Full V1 Model with 4 orientation-selective columns
    Replicates Simulation_V1_pinwheel_MEGcomparison.py
    """
    
    def __init__(self, dt=0.1):
        """
        Args:
            dt: Time step (ms)
        """
        self.dt = dt
        self.orientations = GABOR_CONFIG['orientations']  # [0, 45, 90, 135]
        
        print("🧠 Creating Computational V1 Model...")
        print(f"   4 orientation columns: {self.orientations}°")
        print(f"   Time step: {dt} ms")
        
        # Create orientation columns
        self.columns = {}
        for orientation in self.orientations:
            print(f"\n📊 Building {orientation}° column...")
            self.columns[orientation] = V1OrientationColumn(orientation, dt)
        
        # Simulation state
        self.current_time = 0.0
        self.warmup_time = V1_ARCHITECTURE['warmup_time_ms']
        self.stimulus_time = V1_ARCHITECTURE['stimulus_time_ms']
        
        # Stimulus tracking
        self.stimulus_active = False
        self.stimulus_start_time = 0.0
        
        print("\n✅ V1 Model created successfully!")
        self._print_stats()
    
    def _print_stats(self):
        """Print model statistics"""
        total_neurons = 0
        for orientation, column in self.columns.items():
            n = (V1_ARCHITECTURE['layer_4_ss'] + V1_ARCHITECTURE['layer_4_inh'] +
                 V1_ARCHITECTURE['layer_23_pyr'] + V1_ARCHITECTURE['layer_23_inh'] +
                 V1_ARCHITECTURE['layer_5_pyr'] + V1_ARCHITECTURE['layer_5_inh'] +
                 V1_ARCHITECTURE['layer_6_pyr'] + V1_ARCHITECTURE['layer_6_inh'])
            total_neurons += n
        
        print(f"\n📈 Total neurons: {total_neurons}")
        print(f"   Per column: {total_neurons // 4}")
        print(f"   Layer 4 SS (input): {GRID_CONFIG['n_neurons']} neurons/column")
        print(f"   Layer 2/3 (output): {GRID_CONFIG['n_neurons']} neurons/column")
    
    def inject_spike_trains(self, spike_trains_by_orientation):
        """
        Inject spike trains into appropriate orientation columns
        
        Args:
            spike_trains_by_orientation: Dict mapping orientation -> spike data
                Each spike data is dict with 'neuron_ids' and 'spike_times'
        """
        for orientation, spike_data in spike_trains_by_orientation.items():
            if orientation in self.columns:
                # Offset spike times by current simulation time
                offset_spike_data = {
                    'neuron_ids': spike_data['neuron_ids'],
                    'spike_times': spike_data['spike_times'] + self.current_time
                }
                self.columns[orientation].inject_lgn_spikes(
                    offset_spike_data,
                    self.current_time
                )
    
    def run_stimulus(self, spike_trains_by_orientation, warmup=True):
        """
        Run a complete stimulus presentation
        
        Args:
            spike_trains_by_orientation: Dict mapping orientation -> spike trains
            warmup: Whether to run warmup period before stimulus
            
        Returns:
            Dict with simulation results
        """
        # Reset time
        self.current_time = 0.0
        
        # Warmup period (spontaneous activity)
        # Only print on first run to avoid console spam
        if warmup and not hasattr(self, '_warmup_done'):
            print(f"⏱️  Warmup: {self.warmup_time} ms...")
            self._warmup_done = True
        
        if warmup:
            while self.current_time < self.warmup_time:
                for column in self.columns.values():
                    column.update(self.current_time)
                self.current_time += self.dt
        
        # Inject spike trains
        self.inject_spike_trains(spike_trains_by_orientation)
        
        # Stimulus period
        self.stimulus_start_time = self.current_time
        stimulus_end_time = self.current_time + self.stimulus_time
        
        while self.current_time < stimulus_end_time:
            # Get spike trains for this time point
            current_spikes = self._get_current_spikes(
                spike_trains_by_orientation,
                self.current_time
            )
            
            # Update all columns
            for orientation, column in self.columns.items():
                lgn_input = current_spikes.get(orientation, None)
                column.update(self.current_time, lgn_input)
            
            self.current_time += self.dt
        
        # Collect results
        results = self.get_results()
        
        return results
    
    def _get_current_spikes(self, spike_trains, current_time):
        """
        Extract spikes that should fire at current time from all spike trains
        
        Args:
            spike_trains: Dict mapping orientation -> spike data
            current_time: Current simulation time
            
        Returns:
            Dict mapping orientation -> current spike data
        """
        current_spikes = {}
        
        for orientation, spike_data in spike_trains.items():
            # Find spikes within current time step
            mask = np.abs(spike_data['spike_times'] - 
                         (current_time - self.stimulus_start_time)) < self.dt/2
            
            if np.any(mask):
                current_spikes[orientation] = {
                    'neuron_ids': spike_data['neuron_ids'][mask],
                    'spike_times': np.full(np.sum(mask), current_time)
                }
        
        return current_spikes
    
    def get_results(self):
        """
        Extract results from all columns
        
        Returns:
            Dict with firing rates and spikes for each orientation and layer
        """
        # Time window for analysis (during stimulus presentation)
        analysis_window = (
            self.stimulus_start_time,
            self.stimulus_start_time + self.stimulus_time
        )
        
        results = {
            'orientations': {},
            'time_window': analysis_window,
            'dt': self.dt
        }
        
        for orientation, column in self.columns.items():
            # Get firing rates from each layer
            layer_23_rates = column.get_layer_firing_rates('layer_23', analysis_window)
            layer_4_rates = column.get_layer_firing_rates('layer_4', analysis_window)
            layer_5_rates = column.get_layer_firing_rates('layer_5', analysis_window)
            layer_6_rates = column.get_layer_firing_rates('layer_6', analysis_window)
            
            # Get spike times
            layer_23_spikes = column.get_layer_output('layer_23', analysis_window)
            layer_4_spikes = column.get_layer_output('layer_4', analysis_window)
            
            results['orientations'][orientation] = {
                'layer_23': {
                    'firing_rates': layer_23_rates,
                    'spikes': layer_23_spikes,
                    'mean_rate': np.mean(layer_23_rates)
                },
                'layer_4': {
                    'firing_rates': layer_4_rates,
                    'spikes': layer_4_spikes,
                    'mean_rate': np.mean(layer_4_rates)
                },
                'layer_5': {
                    'firing_rates': layer_5_rates,
                    'mean_rate': np.mean(layer_5_rates)
                },
                'layer_6': {
                    'firing_rates': layer_6_rates,
                    'mean_rate': np.mean(layer_6_rates)
                }
            }
        
        return results
    
    def calculate_orientation_selectivity_index(self, results):
        """
        Calculate Orientation Selectivity Index (OSI) for each neuron
        OSI = (R_pref - R_orth) / (R_pref + R_orth)
        
        Args:
            results: Results dict from get_results()
            
        Returns:
            Dict with OSI values for each orientation
        """
        osi_values = {}
        
        orientations = list(self.columns.keys())
        
        for i, orientation in enumerate(orientations):
            # Get responses from this orientation
            pref_response = results['orientations'][orientation]['layer_23']['firing_rates']
            
            # Get orthogonal orientation (90° away)
            orth_idx = (i + 2) % 4
            orth_orientation = orientations[orth_idx]
            orth_response = results['orientations'][orth_orientation]['layer_23']['firing_rates']
            
            # Calculate OSI
            denominator = pref_response + orth_response + 1e-6
            osi = (pref_response - orth_response) / denominator
            
            osi_values[orientation] = {
                'osi': osi,
                'mean_osi': np.mean(osi),
                'median_osi': np.median(osi)
            }
        
        return osi_values
    
    def get_orientation_map(self, results, layer='layer_23'):
        """
        Create spatial orientation preference map
        
        Args:
            results: Results dict from get_results()
            layer: Which layer to use ('layer_23' by default)
            
        Returns:
            2D array (NxN) with preferred orientation at each position
        """
        grid_size = GRID_CONFIG['grid_rows']  # Use config grid size
        n_neurons = GRID_CONFIG['n_neurons']
        orientation_map = np.zeros((grid_size, grid_size))
        response_strength_map = np.zeros((grid_size, grid_size))
        
        # For each position in the grid
        for neuron_idx in range(n_neurons):
            row = neuron_idx // grid_size
            col = neuron_idx % grid_size
            
            # Get responses from all orientations at this position
            responses = {}
            for orientation in self.orientations:
                rate = results['orientations'][orientation][layer]['firing_rates'][neuron_idx]
                responses[orientation] = rate
            
            # Find preferred orientation (highest response)
            pref_orientation = max(responses, key=responses.get)
            max_response = responses[pref_orientation]
            
            orientation_map[row, col] = pref_orientation
            response_strength_map[row, col] = max_response
        
        return {
            'orientation_map': orientation_map,
            'response_strength': response_strength_map
        }
    
    def reset(self):
        """Reset all columns"""
        for column in self.columns.values():
            column.reset()
        self.current_time = 0.0
        self.stimulus_start_time = 0.0
        self.stimulus_active = False

