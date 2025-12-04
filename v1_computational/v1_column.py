"""
V1 Orientation Column
Computational implementation replicating the MDPI2021 V1 column structure
"""

import numpy as np
from neurons import NeuronPopulation, LIFNeuron
from config import V1_ARCHITECTURE


class V1OrientationColumn:
    """
    Single orientation-selective column in V1
    Replicates the exact architecture from OrientedColumnV1.py
    
    Layers:
        - Layer 4: Spiny Stellate cells (input) + Inhibitory
        - Layer 2/3: Pyramidal cells + Inhibitory
        - Layer 5: Pyramidal cells + Inhibitory
        - Layer 6: Pyramidal cells + Inhibitory
    """
    
    def __init__(self, preferred_orientation, dt=0.1):
        """
        Args:
            preferred_orientation: Preferred orientation (0, 45, 90, or 135 degrees)
            dt: Time step (ms)
        """
        self.preferred_orientation = preferred_orientation
        self.dt = dt
        
        # Neuron parameters
        neuron_params = {
            'v_rest': V1_ARCHITECTURE['v_rest'],
            'v_threshold': V1_ARCHITECTURE['v_threshold'],
            'v_reset': V1_ARCHITECTURE['v_reset'],
            'tau_m': V1_ARCHITECTURE['tau_membrane'],
            'tau_syn_ex': V1_ARCHITECTURE['tau_syn_ex'],
            'tau_syn_in': V1_ARCHITECTURE['tau_syn_in'],
            'refractory_period': V1_ARCHITECTURE['refractory_period'],
        }
        
        print(f"  Creating {preferred_orientation}° column layers...")
        
        # Layer 4: Spiny Stellate cells (324 neurons - main input layer)
        self.layer_4_ss = NeuronPopulation(
            V1_ARCHITECTURE['layer_4_ss'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_l23'],  # Uses same as L23
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        # Layer 4: Inhibitory interneurons (65 neurons)
        self.layer_4_inh = NeuronPopulation(
            V1_ARCHITECTURE['layer_4_inh'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_inh'],
            poisson_weight=4.9,
            dt=dt
        )
        
        # Layer 2/3: Pyramidal cells (324 neurons - primary output)
        self.layer_23_pyr = NeuronPopulation(
            V1_ARCHITECTURE['layer_23_pyr'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_l23'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        # Layer 2/3: Inhibitory interneurons (65 neurons)
        self.layer_23_inh = NeuronPopulation(
            V1_ARCHITECTURE['layer_23_inh'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_inh'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        # Layer 5: Pyramidal cells (81 neurons)
        self.layer_5_pyr = NeuronPopulation(
            V1_ARCHITECTURE['layer_5_pyr'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_l5'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        # Layer 5: Inhibitory interneurons (16 neurons)
        self.layer_5_inh = NeuronPopulation(
            V1_ARCHITECTURE['layer_5_inh'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_inh'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        # Layer 6: Pyramidal cells (243 neurons)
        self.layer_6_pyr = NeuronPopulation(
            V1_ARCHITECTURE['layer_6_pyr'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_l6'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        # Layer 6: Inhibitory interneurons (49 neurons)
        self.layer_6_inh = NeuronPopulation(
            V1_ARCHITECTURE['layer_6_inh'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_inh'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        # Inter-layer connections storage
        self.feedforward_connections = {}  # (pre_layer, pre_idx) -> [(post_layer, post_idx, weight), ...]
        
        # Setup connectivity
        self._setup_connections()
        self._setup_feedforward()
        
    def _setup_connections(self):
        """Setup recurrent connections within each layer (from MDPI2021)"""
        # Layer 2/3 recurrent (indegree=36)
        self.layer_23_pyr.add_recurrent_connections(
            V1_ARCHITECTURE['layer_23_recurrent_indegree'],
            V1_ARCHITECTURE['lateral_weight']
        )
        
        # Layer 5 recurrent (indegree=10)
        self.layer_5_pyr.add_recurrent_connections(
            V1_ARCHITECTURE['layer_5_recurrent_indegree'],
            V1_ARCHITECTURE['lateral_weight']
        )
        
        # Layer 6 recurrent (indegree=20)
        self.layer_6_pyr.add_recurrent_connections(
            V1_ARCHITECTURE['layer_6_recurrent_indegree'],
            V1_ARCHITECTURE['lateral_weight']
        )
        
        # Layer 4 SS -> Layer 4 Inhibitory (indegree=32)
        self._connect_populations(
            self.layer_4_ss, self.layer_4_inh, 32, V1_ARCHITECTURE['lateral_weight']
        )
        
        # Layer 4 Inhibitory -> Layer 4 SS (indegree=6, inhibitory)
        self._connect_populations(
            self.layer_4_inh, self.layer_4_ss, 6, V1_ARCHITECTURE['inhibitory_weight']
        )
        
        # Layer 4 Inhibitory recurrent
        self.layer_4_inh.add_recurrent_connections(6, V1_ARCHITECTURE['inhibitory_weight'])
        
        # Layer 2/3 Pyramidal -> Layer 2/3 Inhibitory (indegree=35)
        self._connect_populations(
            self.layer_23_pyr, self.layer_23_inh, 35, V1_ARCHITECTURE['lateral_weight']
        )
        
        # Layer 2/3 Inhibitory -> Layer 2/3 Pyramidal (indegree=8, inhibitory)
        self._connect_populations(
            self.layer_23_inh, self.layer_23_pyr, 8, V1_ARCHITECTURE['inhibitory_weight']
        )
        
        # Layer 2/3 Inhibitory recurrent
        self.layer_23_inh.add_recurrent_connections(8, V1_ARCHITECTURE['inhibitory_weight'])
        
        # Layer 5 connections
        self._connect_populations(
            self.layer_5_pyr, self.layer_5_inh, 30, V1_ARCHITECTURE['lateral_weight']
        )
        self._connect_populations(
            self.layer_5_inh, self.layer_5_pyr, 8, V1_ARCHITECTURE['inhibitory_weight']
        )
        self.layer_5_inh.add_recurrent_connections(8, V1_ARCHITECTURE['inhibitory_weight'])
        
        # Layer 6 connections
        self._connect_populations(
            self.layer_6_pyr, self.layer_6_inh, 32, V1_ARCHITECTURE['lateral_weight']
        )
        self._connect_populations(
            self.layer_6_inh, self.layer_6_pyr, 6, V1_ARCHITECTURE['inhibitory_weight']
        )
        self.layer_6_inh.add_recurrent_connections(6, V1_ARCHITECTURE['inhibitory_weight'])
    
    def _connect_populations(self, pre_pop, post_pop, indegree, weight):
        """Create connections between two populations"""
        for post_idx in range(post_pop.n_neurons):
            # Random presynaptic neurons
            pre_indices = np.random.choice(
                pre_pop.n_neurons,
                size=min(indegree, pre_pop.n_neurons),
                replace=False
            )
            
            for pre_idx in pre_indices:
                key = (id(pre_pop), pre_idx)
                if key not in self.feedforward_connections:
                    self.feedforward_connections[key] = []
                self.feedforward_connections[key].append((id(post_pop), post_idx, weight))
    
    def _setup_feedforward(self):
        """Setup feedforward connections between layers"""
        # Layer 2/3 -> Layer 5 (indegree=15)
        self._connect_populations(
            self.layer_23_pyr, self.layer_5_pyr, 15, V1_ARCHITECTURE['feedforward_weight']
        )
        
        # Layer 5 -> Layer 6 (indegree=20)
        self._connect_populations(
            self.layer_5_pyr, self.layer_6_pyr, 20, V1_ARCHITECTURE['feedforward_weight']
        )
        
        # Layer 4 SS -> Layer 2/3 Pyramidal (groups of 4 SS -> 4 Pyr)
        # This implements the polychrony detection from MDPI2021
        for i in range(0, 324, 4):
            if i + 3 < 324:
                # Each group of 4 SS cells connects to corresponding 4 Pyr cells
                for ss_idx in range(4):
                    for pyr_idx in range(4):
                        key = (id(self.layer_4_ss), i + ss_idx)
                        if key not in self.feedforward_connections:
                            self.feedforward_connections[key] = []
                        self.feedforward_connections[key].append(
                            (id(self.layer_23_pyr), i + pyr_idx, V1_ARCHITECTURE['feedforward_weight'])
                        )
    
    def inject_lgn_spikes(self, spike_data, current_time):
        """
        Inject LGN spikes into Layer 4 Spiny Stellate cells
        
        Args:
            spike_data: Dict with 'neuron_ids' and 'spike_times'
            current_time: Current simulation time
        """
        # Find spikes that should arrive at current time
        for neuron_id, spike_time in zip(spike_data['neuron_ids'], spike_data['spike_times']):
            if abs(spike_time - current_time) < self.dt/2:  # Within current time step
                if neuron_id < len(self.layer_4_ss.neurons):
                    # Strong input to SS cells (weight=15000 from MDPI2021)
                    self.layer_4_ss.neurons[neuron_id].receive_spike(
                        V1_ARCHITECTURE['lgn_to_ss4_weight']
                    )
    
    def update(self, current_time, lgn_input=None):
        """
        Update all layers for one time step
        
        Args:
            current_time: Current simulation time (ms)
            lgn_input: LGN spike data to inject
            
        Returns:
            Dict with spikes from each layer
        """
        # Inject LGN input into Layer 4
        if lgn_input is not None:
            self.inject_lgn_spikes(lgn_input, current_time)
        
        # Update Layer 4 Spiny Stellate
        spikes_l4_ss = self.layer_4_ss.update(current_time)
        
        # Update Layer 4 Inhibitory
        spikes_l4_inh = self.layer_4_inh.update(current_time)
        
        # Propagate Layer 4 SS spikes to other layers via feedforward connections
        self._propagate_spikes(self.layer_4_ss, spikes_l4_ss)
        
        # Update Layer 2/3 Pyramidal
        spikes_l23_pyr = self.layer_23_pyr.update(current_time)
        
        # Update Layer 2/3 Inhibitory
        spikes_l23_inh = self.layer_23_inh.update(current_time)
        
        # Propagate Layer 2/3 spikes
        self._propagate_spikes(self.layer_23_pyr, spikes_l23_pyr)
        
        # Update Layer 5 Pyramidal
        spikes_l5_pyr = self.layer_5_pyr.update(current_time)
        
        # Update Layer 5 Inhibitory
        spikes_l5_inh = self.layer_5_inh.update(current_time)
        
        # Propagate Layer 5 spikes
        self._propagate_spikes(self.layer_5_pyr, spikes_l5_pyr)
        
        # Update Layer 6 Pyramidal
        spikes_l6_pyr = self.layer_6_pyr.update(current_time)
        
        # Update Layer 6 Inhibitory
        spikes_l6_inh = self.layer_6_inh.update(current_time)
        
        return {
            'layer_4_ss': spikes_l4_ss,
            'layer_4_inh': spikes_l4_inh,
            'layer_23_pyr': spikes_l23_pyr,
            'layer_23_inh': spikes_l23_inh,
            'layer_5_pyr': spikes_l5_pyr,
            'layer_5_inh': spikes_l5_inh,
            'layer_6_pyr': spikes_l6_pyr,
            'layer_6_inh': spikes_l6_inh,
        }
    
    def _propagate_spikes(self, source_pop, spike_indices):
        """
        Propagate spikes from source population through feedforward connections
        
        Args:
            source_pop: Source neuron population
            spike_indices: List of neuron indices that spiked
        """
        source_id = id(source_pop)
        
        for spike_idx in spike_indices:
            key = (source_id, spike_idx)
            if key in self.feedforward_connections:
                for target_pop_id, target_idx, weight in self.feedforward_connections[key]:
                    # Find target population
                    target_pop = self._get_population_by_id(target_pop_id)
                    if target_pop and target_idx < len(target_pop.neurons):
                        target_pop.neurons[target_idx].receive_spike(weight)
    
    def _get_population_by_id(self, pop_id):
        """Get population object by its id"""
        populations = [
            self.layer_4_ss, self.layer_4_inh,
            self.layer_23_pyr, self.layer_23_inh,
            self.layer_5_pyr, self.layer_5_inh,
            self.layer_6_pyr, self.layer_6_inh
        ]
        for pop in populations:
            if id(pop) == pop_id:
                return pop
        return None
    
    def get_layer_output(self, layer_name, time_window):
        """
        Get spikes from a specific layer
        
        Args:
            layer_name: 'layer_4', 'layer_23', 'layer_5', or 'layer_6'
            time_window: (start_ms, end_ms)
            
        Returns:
            Dict with spike data
        """
        if layer_name == 'layer_4':
            return self.layer_4_ss.get_all_spikes(time_window)
        elif layer_name == 'layer_23':
            return self.layer_23_pyr.get_all_spikes(time_window)
        elif layer_name == 'layer_5':
            return self.layer_5_pyr.get_all_spikes(time_window)
        elif layer_name == 'layer_6':
            return self.layer_6_pyr.get_all_spikes(time_window)
        return {'neuron_ids': np.array([]), 'spike_times': np.array([])}
    
    def get_layer_firing_rates(self, layer_name, time_window):
        """
        Get firing rates from a specific layer
        
        Args:
            layer_name: 'layer_4', 'layer_23', 'layer_5', or 'layer_6'
            time_window: (start_ms, end_ms)
            
        Returns:
            Array of firing rates
        """
        if layer_name == 'layer_4':
            return self.layer_4_ss.get_firing_rates(time_window)
        elif layer_name == 'layer_23':
            return self.layer_23_pyr.get_firing_rates(time_window)
        elif layer_name == 'layer_5':
            return self.layer_5_pyr.get_firing_rates(time_window)
        elif layer_name == 'layer_6':
            return self.layer_6_pyr.get_firing_rates(time_window)
        return np.array([])
    
    def reset(self):
        """Reset all neurons in column"""
        self.layer_4_ss.reset()
        self.layer_4_inh.reset()
        self.layer_23_pyr.reset()
        self.layer_23_inh.reset()
        self.layer_5_pyr.reset()
        self.layer_5_inh.reset()
        self.layer_6_pyr.reset()
        self.layer_6_inh.reset()

