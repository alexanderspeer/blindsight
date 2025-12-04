"""
Computational Neuron Models
Fast computational implementations of spiking neurons
Replicates the behavior of NEST neurons without the overhead
"""

import numpy as np
from config import V1_ARCHITECTURE


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron (computational implementation)
    Replicates iaf_psc_alpha and lifl_psc_exp_ie behavior
    """
    
    def __init__(self, neuron_id, v_rest=-65.0, v_threshold=-50.0, 
                 v_reset=-65.0, tau_m=10.0, tau_syn_ex=2.0, tau_syn_in=2.0,
                 refractory_period=2.0, dt=0.1):
        
        self.neuron_id = neuron_id
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.tau_syn_ex = tau_syn_ex
        self.tau_syn_in = tau_syn_in
        self.refractory_period = refractory_period
        self.dt = dt
        
        # State variables
        self.v_m = v_rest + np.random.randn() * 2.0  # Add noise to initial voltage
        self.i_syn_ex = 0.0  # Excitatory synaptic current
        self.i_syn_in = 0.0  # Inhibitory synaptic current
        self.refractory_counter = 0
        
        # Spike history
        self.spike_times = []
        self.last_spike_time = -1000.0
        
        # Enhancement factor (for intrinsic excitability simulation)
        self.enhancement = 1.0
        
    def update(self, current_time, external_current=0.0, poisson_input=0.0):
        """
        Update neuron state for one time step
        
        Args:
            current_time: Current simulation time (ms)
            external_current: External input current
            poisson_input: Background Poisson noise
            
        Returns:
            True if neuron spiked, False otherwise
        """
        spiked = False
        
        # Refractory period
        if self.refractory_counter > 0:
            self.refractory_counter -= self.dt
            self.v_m = self.v_rest
            
        else:
            # Update membrane potential (Leaky integration)
            dv = (-(self.v_m - self.v_rest) + 
                  (self.i_syn_ex - self.i_syn_in + external_current + poisson_input) * self.enhancement) / self.tau_m
            
            self.v_m += dv * self.dt
            
            # Check for spike
            if self.v_m >= self.v_threshold:
                self.v_m = self.v_reset
                self.refractory_counter = self.refractory_period
                self.spike_times.append(current_time)
                self.last_spike_time = current_time
                spiked = True
        
        # Decay synaptic currents
        self.i_syn_ex *= np.exp(-self.dt / self.tau_syn_ex)
        self.i_syn_in *= np.exp(-self.dt / self.tau_syn_in)
        
        return spiked
    
    def receive_spike(self, weight):
        """
        Receive a spike from a presynaptic neuron
        
        Args:
            weight: Synaptic weight (positive for excitatory, negative for inhibitory)
        """
        if weight > 0:
            self.i_syn_ex += weight
        else:
            self.i_syn_in += abs(weight)
    
    def reset(self):
        """Reset neuron to initial state"""
        self.v_m = self.v_rest + np.random.randn() * 2.0
        self.i_syn_ex = 0.0
        self.i_syn_in = 0.0
        self.refractory_counter = 0
        self.spike_times = []
        self.last_spike_time = -1000.0
        self.enhancement = 1.0


class PoissonNoise:
    """
    Poisson background activity generator
    Simulates background synaptic bombardment
    """
    
    def __init__(self, rate_hz, weight, dt=0.1):
        """
        Args:
            rate_hz: Firing rate in Hz
            weight: Synaptic weight
            dt: Time step (ms)
        """
        self.rate_hz = rate_hz
        self.weight = weight
        self.dt = dt
        
        # Convert rate to probability per time step
        self.spike_prob = (rate_hz / 1000.0) * dt
    
    def sample(self):
        """
        Sample Poisson process for one time step
        
        Returns:
            Current contribution (weight if spike, 0 otherwise)
        """
        if np.random.rand() < self.spike_prob:
            return self.weight
        return 0.0


class NeuronPopulation:
    """
    Population of neurons with connectivity
    Represents one layer or group in V1
    """
    
    def __init__(self, n_neurons, neuron_params, poisson_rate=0, poisson_weight=0, dt=0.1):
        """
        Args:
            n_neurons: Number of neurons in population
            neuron_params: Dict with neuron parameters
            poisson_rate: Background Poisson rate (Hz)
            poisson_weight: Weight of Poisson inputs
            dt: Time step (ms)
        """
        self.n_neurons = n_neurons
        self.dt = dt
        
        # Create neurons
        self.neurons = [
            LIFNeuron(
                neuron_id=i,
                v_rest=neuron_params.get('v_rest', -65.0),
                v_threshold=neuron_params.get('v_threshold', -50.0),
                v_reset=neuron_params.get('v_reset', -65.0),
                tau_m=neuron_params.get('tau_m', 10.0),
                tau_syn_ex=neuron_params.get('tau_syn_ex', 2.0),
                tau_syn_in=neuron_params.get('tau_syn_in', 2.0),
                refractory_period=neuron_params.get('refractory_period', 2.0),
                dt=dt
            )
            for i in range(n_neurons)
        ]
        
        # Poisson noise
        self.poisson = None
        if poisson_rate > 0:
            self.poisson = PoissonNoise(poisson_rate, poisson_weight, dt)
        
        # Recurrent connections (sparse)
        self.recurrent_connections = {}
        
        # Spike buffer
        self.current_spikes = []
    
    def add_recurrent_connections(self, indegree, weight):
        """
        Add random recurrent connections within population
        
        Args:
            indegree: Number of incoming connections per neuron
            weight: Connection weight
        """
        for post_idx in range(self.n_neurons):
            # Random presynaptic neurons
            pre_indices = np.random.choice(
                self.n_neurons, 
                size=min(indegree, self.n_neurons-1), 
                replace=False
            )
            # Remove self-connections
            pre_indices = pre_indices[pre_indices != post_idx]
            
            for pre_idx in pre_indices:
                if pre_idx not in self.recurrent_connections:
                    self.recurrent_connections[pre_idx] = []
                self.recurrent_connections[pre_idx].append((post_idx, weight))
    
    def update(self, current_time, external_inputs=None):
        """
        Update all neurons for one time step
        
        Args:
            current_time: Current simulation time (ms)
            external_inputs: Dict mapping neuron_id -> current
            
        Returns:
            List of neuron indices that spiked
        """
        if external_inputs is None:
            external_inputs = {}
        
        self.current_spikes = []
        
        # Update each neuron
        for i, neuron in enumerate(self.neurons):
            # External input
            ext_current = external_inputs.get(i, 0.0)
            
            # Poisson background
            poisson_current = self.poisson.sample() if self.poisson else 0.0
            
            # Update neuron
            spiked = neuron.update(current_time, ext_current, poisson_current)
            
            if spiked:
                self.current_spikes.append(i)
        
        # Propagate recurrent spikes
        for spike_idx in self.current_spikes:
            if spike_idx in self.recurrent_connections:
                for post_idx, weight in self.recurrent_connections[spike_idx]:
                    self.neurons[post_idx].receive_spike(weight)
        
        return self.current_spikes
    
    def get_spike_times(self, neuron_idx):
        """Get spike times for a specific neuron"""
        if neuron_idx < len(self.neurons):
            return self.neurons[neuron_idx].spike_times
        return []
    
    def get_all_spikes(self, time_window=None):
        """
        Get all spikes from this population
        
        Args:
            time_window: (start_ms, end_ms) or None for all spikes
            
        Returns:
            Dict with 'neuron_ids' and 'spike_times' arrays
        """
        neuron_ids = []
        spike_times = []
        
        for neuron_idx, neuron in enumerate(self.neurons):
            for spike_time in neuron.spike_times:
                if time_window is None or \
                   (spike_time >= time_window[0] and spike_time <= time_window[1]):
                    neuron_ids.append(neuron_idx)
                    spike_times.append(spike_time)
        
        return {
            'neuron_ids': np.array(neuron_ids),
            'spike_times': np.array(spike_times)
        }
    
    def get_firing_rates(self, time_window):
        """
        Calculate firing rate for each neuron in time window
        
        Args:
            time_window: (start_ms, end_ms)
            
        Returns:
            Array of firing rates (Hz) for each neuron
        """
        rates = np.zeros(self.n_neurons)
        duration_s = (time_window[1] - time_window[0]) / 1000.0
        
        if duration_s <= 0:
            return rates
        
        for neuron_idx, neuron in enumerate(self.neurons):
            spike_count = sum(
                1 for t in neuron.spike_times 
                if time_window[0] <= t <= time_window[1]
            )
            rates[neuron_idx] = spike_count / duration_s
        
        return rates
    
    def reset(self):
        """Reset all neurons"""
        for neuron in self.neurons:
            neuron.reset()
        self.current_spikes = []

