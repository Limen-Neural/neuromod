// Deleting mining/HFT scripts and reward struct from this repo. I am moving all mining related code to the mining repo. 
// Taking out references to 'mining' in this file as well.  So I am deleting any references to 'mining' in engine.rs as well. 
// Removing any references to 'HftReward' in traits.rs as well. The 'pub use traits::HftReward;' line from this file as well. 

use rand::Rng; // For stochastic encoding and initialization
use serde::{Deserialize, Serialize}; // For easy serialization of network state (weights, modulators, etc.) for checkpointing and analysis

use super::lif::LifNeuron; // Importing the LIF neuron struct to use in the SpikingNetwork
use super::izhikevich::IzhikevichNeuron; // Importing the Izhikevich neuron struct to use in the SpikingNetwork
use super::rm_stdp::*; // Importing R-STDP related structs and constants
use super::modulators::NeuroModulators; // Importing the neuromodulator struct to use in the SpikingNetwork

/// L1 synaptic weight budget per neuron (total weight sum target).
const WEIGHT_BUDGET: f32 = 2.0; // This encourages competition among synapses and prevents runaway excitation

// This is the core of the system, integrating LIF neurons, Izhikevich neurons
#[derive(Default, Serialize, Deserialize)]
pub struct SpikingNetwork {
    // Bank 1: LIF Neurons (Fast, Reactive)
    pub neurons: Vec<LifNeuron>,
    // Bank 2: Izhikevich Neurons (Complex, Adaptive)
    pub iz_neurons: Vec<IzhikevichNeuron>,
    // Global Neuromodulators
    pub modulators: NeuroModulators,
    /// Global step counter for STDP timing
    pub global_step: i64,
    /// Pre-synaptic spike times for each input channel (for STDP)
    pub input_spike_times: Vec<i64>,
    /// Per-channel exponential moving average of input stimuli
    pub predictive_state: [f32; crate::NUM_INPUT_CHANNELS],
}

impl SpikingNetwork {
    /// Create a new spiking network with default configuration
    pub fn new() -> Self {
        let mut neurons: Vec<LifNeuron> = (0..16).map(|_| {
            let mut n = LifNeuron::new();
            n.weights = vec![1.5; crate::NUM_INPUT_CHANNELS];
            n.last_spike_time = -1;
            n
        }).collect();

        let mut rng = rand::thread_rng();

        // Initialize neurons with bear/bull pairs
        for i in 0..14 {
            let ch = i / 2;
            let neuron = &mut neurons[i];

            // Set primary channel weight
            neuron.weights[ch] = 0.8 + (rng.r#gen::<f32>() * 0.4);

            // Differentiated thresholds
            if i % 2 == 0 {
                // Bear neurons: conservative threshold
                neuron.threshold = 0.10 + (rng.r#gen::<f32>() * 0.04);
            } else {
                // Bull neurons: sensitive threshold
                neuron.threshold = 0.06 + (rng.r#gen::<f32>() * 0.04);
            }
            neuron.base_threshold = neuron.threshold;
        }

        // N14: Coincidence Detector
        neurons[14].threshold = 0.50;
        neurons[14].base_threshold = 0.50;
        // N15: Global Inhibitory Interneuron
        neurons[15].threshold = 0.50;
        neurons[15].base_threshold = 0.50;

        Self {
            neurons,
            iz_neurons: vec![IzhikevichNeuron::new_regular_spiking(); 5],
            modulators: NeuroModulators::default(),
            global_step: 0,
            input_spike_times: vec![-1; crate::NUM_INPUT_CHANNELS],
            predictive_state: [0.0; crate::NUM_INPUT_CHANNELS],
        }
    }

    /// Step the network with input stimuli and modulators
    /// 
    /// # Arguments
    /// * `stimuli` - Input stimulus values for each channel
    /// * `modulators` - Current neuromodulator levels
    /// 
    /// Returns: Vector of neuron indices that spiked
    pub fn step(&mut self, stimuli: &[f32; crate::NUM_INPUT_CHANNELS], modulators: &NeuroModulators) -> Vec<usize> {
        self.global_step += 1;
        self.modulators = *modulators;

        let stress_multiplier = (1.0 - self.modulators.cortisol).max(0.1);
        let learning_rate = 0.5 * self.modulators.dopamine;

        // Update neuron parameters based on modulators
        for neuron in &mut self.neurons {
            let target_decay = 0.15 - (0.05 * self.modulators.acetylcholine);
            neuron.decay_rate = target_decay;

            let global_target = 0.20 - (0.05 * self.modulators.dopamine) + (0.15 * self.modulators.cortisol);
            let target_threshold = (global_target + if neuron.last_spike { 0.005 } else { -0.001 })
                .clamp(0.05, 0.50);
            neuron.threshold += (target_threshold - neuron.threshold) * learning_rate;
            neuron.threshold = neuron.threshold.clamp(0.05, 0.50);
        }

        // Predictive error coding
        const PRED_ALPHA: f32 = 0.1;
        const PRED_ERR_WEIGHT: f32 = 0.5;
        let mut pred_errors = [0.0_f32; crate::NUM_INPUT_CHANNELS];
        
        for ch in 0..crate::NUM_INPUT_CHANNELS {
            let s = stimuli[ch].abs().clamp(0.0, 1.0);
            pred_errors[ch] = (s - self.predictive_state[ch]).abs();
            self.predictive_state[ch] = PRED_ALPHA * s + (1.0 - PRED_ALPHA) * self.predictive_state[ch];
        }

        // Record input spike times
        let mut rng = rand::thread_rng();
        for (ch, &s) in stimuli.iter().enumerate() {
            let abs_s = s.abs().clamp(0.0, 1.0);
            if abs_s > 0.01 && rng.r#gen::<f32>() < abs_s {
                self.input_spike_times[ch] = self.global_step;
            }
        }

        // Integration phase
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let ch = i % crate::NUM_INPUT_CHANNELS;
            let is_bull = i % 2 == 1;
            let delta = stimuli[ch];
            let polarity_match = if is_bull { delta > 0.0 } else { delta < 0.0 };
            
            let mut total_current = 0.0;
            if polarity_match {
                let stim = delta.abs().clamp(0.0, 1.0);
                let surprise = PRED_ERR_WEIGHT * pred_errors[ch];
                total_current = neuron.weights[ch] * (stim + surprise) * 0.45 * stress_multiplier;
            }
            
            neuron.integrate(total_current);
        }

        // Evaluation phase - check for spikes
        let mut spike_ids = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if let Some(_peak_v) = neuron.check_fire() {
                neuron.last_spike = true;
                neuron.last_spike_time = self.global_step;
                spike_ids.push(i);
            } else {
                neuron.last_spike = false;
            }
        }

        // Lateral inhibition
        if !spike_ids.is_empty() {
            const INHIBITION_STRENGTH: f32 = 0.05;
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                if !spike_ids.contains(&i) {
                    neuron.membrane_potential = (neuron.membrane_potential - INHIBITION_STRENGTH).max(0.0);
                }
            }

// Removed this block of code since we are removing mining related code from this repo.  So I am deleting any references to 'mining' in engine.rs as well.
        }

        // STDP learning
        let dopamine_lr = learning_rate;
        self.apply_stdp(stimuli, dopamine_lr);

        // Synaptic scaling
        for neuron in &mut self.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > 1e-6 {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    *w *= scale;
                    *w = w.clamp(RM_STDP_W_MIN, RM_STDP_W_MAX);
                }
            }
        }

        // Step Izhikevich adaptive bank.
        // Drive with mean LIF activity modulated by dopamine, so the adaptive
        // bank reflects the overall network excitement level.
        let lif_mean = if !self.neurons.is_empty() {
            let sum: f32 = self.neurons.iter().map(|n| n.membrane_potential).sum();
            sum / self.neurons.len() as f32
        } else {
            0.0
        };
        let iz_drive = (lif_mean * 20.0 + self.modulators.dopamine * 5.0)
            .clamp(0.0, 15.0);
        for iz in &mut self.iz_neurons {
            iz.step(iz_drive);
        }

        spike_ids
    }

    /// Apply STDP learning rule
    fn apply_stdp(&mut self, _stimuli: &[f32; crate::NUM_INPUT_CHANNELS], dopamine_lr: f32) {
        if dopamine_lr < 1e-6 {
            return;
        }

        let input_times = self.input_spike_times.clone();

        for neuron in &mut self.neurons {
            if neuron.last_spike_time < 0 {
                continue;
            }

            for (ch, &pre_time) in input_times.iter().enumerate() {
                if ch >= neuron.weights.len() || pre_time < 0 {
                    continue;
                }

                let post_time = neuron.last_spike_time;
                if post_time < 0 {
                    continue;
                }

                let delta_t = (post_time - pre_time) as f32;

                let dw = if delta_t >= 0.0 {
                    RM_STDP_A_PLUS * (-delta_t / RM_STDP_TAU_PLUS).exp()
                } else {
                    -RM_STDP_A_MINUS * (delta_t / RM_STDP_TAU_MINUS).exp()
                };

                neuron.weights[ch] = (neuron.weights[ch] + dw * dopamine_lr)
                    .clamp(RM_STDP_W_MIN, RM_STDP_W_MAX);
            }
        }
    }

    /// Get current membrane potentials for all neurons
    pub fn get_membrane_potentials(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.membrane_potential).collect()
    }

    /// Get current thresholds for all neurons
    pub fn get_thresholds(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.threshold).collect()
    }

    /// Reset network to initial state
    pub fn reset(&mut self) {
        self.global_step = 0;
        self.input_spike_times = vec![-1; crate::NUM_INPUT_CHANNELS];
        self.predictive_state = [0.0; crate::NUM_INPUT_CHANNELS];
        
        for neuron in &mut self.neurons {
            neuron.membrane_potential = 0.0;
            neuron.last_spike = false;
            neuron.last_spike_time = -1;
        }
        
        self.modulators = NeuroModulators::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = SpikingNetwork::new();
        assert_eq!(network.neurons.len(), 16);
        assert_eq!(network.iz_neurons.len(), 5);
        assert_eq!(network.global_step, 0);
    }

    #[test]
    fn test_network_step() {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5; crate::NUM_INPUT_CHANNELS];
        let modulators = NeuroModulators::default();
        
        let spikes = network.step(&stimuli, &modulators);
        assert_eq!(network.global_step, 1);
        // Should have some neurons potentially spiking
        assert!(spikes.len() <= 16);
    }

    #[test]
    fn test_membrane_potentials() {
        let network = SpikingNetwork::new();
        let potentials = network.get_membrane_potentials();
        assert_eq!(potentials.len(), 16);
        // All should start at 0
        for &p in &potentials {
            assert_eq!(p, 0.0);
        }
    }
}
