"""
V1 Orientation Column
Computational implementation replicating the MDPI2021 V1 column structure
"""

import numpy as np
from neurons import NeuronPopulation, LIFNeuron
from config import V1_ARCHITECTURE, GRID_CONFIG


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
        
        self.debug_step_counter = 0
        
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
        
        n_grid_neurons = GRID_CONFIG['n_neurons']
        self.layer_4_ss = NeuronPopulation(
            n_grid_neurons,
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_l23'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt
        )
        
        self.layer_4_inh = NeuronPopulation(
            V1_ARCHITECTURE['layer_4_inh'],
            neuron_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_inh'],
            poisson_weight=4.9,
            dt=dt
        )
        
        # layer 2/3 uses different params
        l23_params = neuron_params.copy()
        l23_params['v_threshold'] = V1_ARCHITECTURE['L23_v_threshold']
        l23_params['tau_m'] = V1_ARCHITECTURE['L23_tau_membrane']
        
        self.layer_23_pyr = NeuronPopulation(
            n_grid_neurons,
            l23_params,
            poisson_rate=V1_ARCHITECTURE['poisson_rate_l23'],
            poisson_weight=V1_ARCHITECTURE['poisson_weight'],
            dt=dt,
            bias_current=V1_ARCHITECTURE['L23_bias_current']
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
        
        self.feedforward_connections = {}
        
        self._setup_connections()
        self._setup_feedforward()
        
        print(f"\nDEBUG COLUMN WEIGHTS ({preferred_orientation}°):")
        print(f"  L4→L2/3 = {V1_ARCHITECTURE['weight_L4_to_L23']}")
        print(f"  L2/3→L5 = {V1_ARCHITECTURE['weight_L23_to_L5']}")
        print(f"  L5→L6   = {V1_ARCHITECTURE['weight_L5_to_L6']}")
        
        print(f"DEBUG THRESHOLDS ({preferred_orientation}°):")
        print(f"  L4 thr  = {neuron_params['v_threshold']} mV")
        print(f"  L2/3 thr= {l23_params['v_threshold']} mV")
        print(f"  L5 thr  = {neuron_params['v_threshold']} mV")
        print(f"  L6 thr  = {neuron_params['v_threshold']} mV")
        print(f"  L2/3 tau= {l23_params['tau_m']} ms")
        print(f"  L2/3 bias={V1_ARCHITECTURE['L23_bias_current']} pA")
        print()
        
    def _setup_connections(self):
        """Setup recurrent connections within each layer"""
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
        
        self._connect_populations(
            self.layer_4_ss, self.layer_4_inh, 32, V1_ARCHITECTURE['lateral_weight']
        )
        self._connect_populations(
            self.layer_4_inh, self.layer_4_ss, 6, V1_ARCHITECTURE['inhibitory_weight']
        )
        self.layer_4_inh.add_recurrent_connections(6, V1_ARCHITECTURE['inhibitory_weight'])
        
        self._connect_populations(
            self.layer_23_pyr, self.layer_23_inh, 35, V1_ARCHITECTURE['lateral_weight']
        )
        self._connect_populations(
            self.layer_23_inh, self.layer_23_pyr, 8, V1_ARCHITECTURE['inhibitory_weight']
        )
        self.layer_23_inh.add_recurrent_connections(8, V1_ARCHITECTURE['inhibitory_weight'])
        
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
        self._connect_populations(
            self.layer_23_pyr, self.layer_5_pyr, 15, V1_ARCHITECTURE['weight_L23_to_L5']
        )
        
        self._connect_populations(
            self.layer_5_pyr, self.layer_6_pyr, 20, V1_ARCHITECTURE['weight_L5_to_L6']
        )
        
        n_grid_neurons = GRID_CONFIG['n_neurons']
        for i in range(0, n_grid_neurons, 4):
            if i + 3 < n_grid_neurons:
                for ss_idx in range(4):
                    for pyr_idx in range(4):
                        key = (id(self.layer_4_ss), i + ss_idx)
                        if key not in self.feedforward_connections:
                            self.feedforward_connections[key] = []
                        self.feedforward_connections[key].append(
                            (id(self.layer_23_pyr), i + pyr_idx, V1_ARCHITECTURE['weight_L4_to_L23'])
                        )
    
    def inject_lgn_spikes(self, spike_data, current_time):
        """
        Inject LGN spikes into Layer 4 Spiny Stellate cells
        
        Args:
            spike_data: Dict with 'neuron_ids' and 'spike_times'
            current_time: Current simulation time
        """
        for neuron_id, spike_time in zip(spike_data['neuron_ids'], spike_data['spike_times']):
            if abs(spike_time - current_time) < self.dt/2:
                if neuron_id < len(self.layer_4_ss.neurons):
                    self.layer_4_ss.neurons[neuron_id].receive_spike(
                        V1_ARCHITECTURE['lgn_to_ss4_weight']
                    )
    
    def update(self, current_time, lgn_input=None, debug_print=False):
        """
        Update all layers for one time step
        
        Args:
            current_time: Current simulation time (ms)
            lgn_input: LGN spike data to inject
            debug_print: If True, print synaptic current statistics
            
        Returns:
            Dict with spikes from each layer
        """
        if lgn_input is not None:
            self.inject_lgn_spikes(lgn_input, current_time)
        
        spikes_l4_ss = self.layer_4_ss.update(current_time)
        spikes_l4_inh = self.layer_4_inh.update(current_time)
        self._propagate_spikes(self.layer_4_ss, spikes_l4_ss)
        
        if debug_print and self.debug_step_counter % 20 == 0:
            self._print_layer_currents(current_time)
            self._print_spike_counts_and_flow(spikes_l4_ss, current_time)
        
        spikes_l23_pyr = self.layer_23_pyr.update(current_time)
        spikes_l23_inh = self.layer_23_inh.update(current_time)
        self._propagate_spikes(self.layer_23_pyr, spikes_l23_pyr)
        
        spikes_l5_pyr = self.layer_5_pyr.update(current_time)
        spikes_l5_inh = self.layer_5_inh.update(current_time)
        self._propagate_spikes(self.layer_5_pyr, spikes_l5_pyr)
        
        spikes_l6_pyr = self.layer_6_pyr.update(current_time)
        spikes_l6_inh = self.layer_6_inh.update(current_time)
        
        self.debug_step_counter += 1
        
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
    
    def _print_layer_currents(self, current_time):
        """Print synaptic currents and activity for debugging"""
        l4_currents = [n.i_syn_ex for n in self.layer_4_ss.neurons]
        l23_currents = [n.i_syn_ex for n in self.layer_23_pyr.neurons]
        l5_currents = [n.i_syn_ex for n in self.layer_5_pyr.neurons]
        l6_currents = [n.i_syn_ex for n in self.layer_6_pyr.neurons]
        
        l23_voltages = [n.v_m for n in self.layer_23_pyr.neurons]
        
        print(f"\nDEBUG SYNAPTIC CURRENTS ({self.preferred_orientation}° @ {current_time:.1f}ms):")
        print(f"  L4 input avg  = {np.mean(l4_currents):.3f} pA (max={np.max(l4_currents):.3f})")
        print(f"  L2/3 input avg= {np.mean(l23_currents):.3f} pA (max={np.max(l23_currents):.3f})")
        print(f"  L5 input avg  = {np.mean(l5_currents):.3f} pA (max={np.max(l5_currents):.3f})")
        print(f"  L6 input avg  = {np.mean(l6_currents):.3f} pA (max={np.max(l6_currents):.3f})")
        print(f"  L2/3 V_m avg  = {np.mean(l23_voltages):.2f} mV (threshold={V1_ARCHITECTURE['L23_v_threshold']:.2f})")
    
    def _print_spike_counts_and_flow(self, l4_spikes, current_time):
        """Print spike counts and synaptic drive between layers"""
        time_window = (current_time - 10.0, current_time)
        
        l4_spike_data = self.layer_4_ss.get_all_spikes(time_window)
        l23_spike_data = self.layer_23_pyr.get_all_spikes(time_window)
        l5_spike_data = self.layer_5_pyr.get_all_spikes(time_window)
        l6_spike_data = self.layer_6_pyr.get_all_spikes(time_window)
        
        l4_count = len(l4_spike_data['neuron_ids'])
        l23_count = len(l23_spike_data['neuron_ids'])
        l5_count = len(l5_spike_data['neuron_ids'])
        l6_count = len(l6_spike_data['neuron_ids'])
        
        print(f"\nDEBUG SPIKES ({self.preferred_orientation}°, last 10ms):")
        print(f"  L4 spikes  = {l4_count}")
        print(f"  L2/3 spikes= {l23_count}")
        print(f"  L5 spikes  = {l5_count}")
        print(f"  L6 spikes  = {l6_count}")
        
        l4_to_l23_drive = l4_count * V1_ARCHITECTURE['weight_L4_to_L23']
        l23_to_l5_drive = l23_count * V1_ARCHITECTURE['weight_L23_to_L5']
        l5_to_l6_drive = l5_count * V1_ARCHITECTURE['weight_L5_to_L6']
        
        print(f"\nDEBUG FLOW ({self.preferred_orientation}°):")
        print(f"  L4→L2/3 mean drive = {l4_to_l23_drive:.3f} (spikes×{V1_ARCHITECTURE['weight_L4_to_L23']})")
        print(f"  L2/3→L5 mean drive = {l23_to_l5_drive:.3f} (spikes×{V1_ARCHITECTURE['weight_L23_to_L5']})")
        print(f"  L5→L6 mean drive   = {l5_to_l6_drive:.3f} (spikes×{V1_ARCHITECTURE['weight_L5_to_L6']})")
    
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
    
    def get_synaptic_current_stats(self, layer_name='layer_23'):
        """
        Get statistics about synaptic currents in a layer
        
        Args:
            layer_name: 'layer_23', 'layer_5', or 'layer_6'
            
        Returns:
            Dict with current statistics
        """
        if layer_name == 'layer_23':
            neurons = self.layer_23_pyr.neurons
        elif layer_name == 'layer_5':
            neurons = self.layer_5_pyr.neurons
        elif layer_name == 'layer_6':
            neurons = self.layer_6_pyr.neurons
        else:
            return {}
        
        exc_currents = [n.i_syn_ex for n in neurons]
        inh_currents = [n.i_syn_in for n in neurons]
        voltages = [n.v_m for n in neurons]
        
        return {
            'exc_current_mean': np.mean(exc_currents),
            'exc_current_max': np.max(exc_currents),
            'exc_current_nonzero': np.sum(np.array(exc_currents) > 0.01),
            'inh_current_mean': np.mean(inh_currents),
            'voltage_mean': np.mean(voltages),
            'voltage_max': np.max(voltages),
            'neurons_near_threshold': np.sum(np.array(voltages) > -55.0),
            'neurons_total': len(neurons)
        }
    
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

