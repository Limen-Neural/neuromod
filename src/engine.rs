use rand::Rng;
use serde::{Deserialize, Serialize};

use super::izhikevich::IzhikevichNeuron;
use super::lif::LifNeuron;
use super::modulators::NeuroModulators;
use super::rm_stdp::*;

/// L1 synaptic weight budget per neuron (total weight sum target).
const WEIGHT_BUDGET: f32 = 2.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    InputLenMismatch { expected: usize, got: usize },
}

/// Core network type integrating LIF and Izhikevich neurons.
#[derive(Serialize, Deserialize)]
pub struct SpikingNetwork {
    /// Bank 1: LIF neurons.
    pub neurons: Vec<LifNeuron>,
    /// Bank 2: Izhikevich neurons.
    pub iz_neurons: Vec<IzhikevichNeuron>,
    /// Global neuromodulators.
    pub modulators: NeuroModulators,
    /// Global step counter for STDP timing.
    pub global_step: i64,
    /// Number of input channels expected by `step`.
    pub num_channels: usize,
    /// Pre-synaptic spike times for each input channel.
    pub input_spike_times: Vec<i64>,
    /// Per-channel exponential moving average of input stimuli.
    pub predictive_state: Vec<f32>,
}

impl SpikingNetwork {
    /// Create the default network (16 LIF, 5 Izhikevich, 16 channels).
    pub fn new() -> Self {
        Self::with_dimensions(16, 5, crate::NUM_INPUT_CHANNELS)
    }

    /// Create a dynamically sized network.
    pub fn with_dimensions(num_lif: usize, num_izh: usize, num_channels: usize) -> Self {
        let neurons: Vec<LifNeuron> = (0..num_lif)
            .map(|_| {
                let mut n = LifNeuron::new();
                n.weights = vec![0.0; num_channels];
                n.last_spike_time = -1;
                n
            })
            .collect();

        Self {
            neurons,
            iz_neurons: vec![IzhikevichNeuron::new_regular_spiking(); num_izh],
            modulators: NeuroModulators::default(),
            global_step: 0,
            num_channels,
            input_spike_times: vec![-1; num_channels],
            predictive_state: vec![0.0; num_channels],
        }
    }

    /// Step the network with input stimuli and modulators.
    pub fn step(
        &mut self,
        stimuli: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<Vec<usize>, StepError> {
        if stimuli.len() != self.num_channels {
            return Err(StepError::InputLenMismatch {
                expected: self.num_channels,
                got: stimuli.len(),
            });
        }

        self.global_step += 1;
        self.modulators = *modulators;

        let stress_multiplier = (1.0 - self.modulators.cortisol).max(0.1);
        let learning_rate = 0.5 * self.modulators.dopamine;

        for neuron in &mut self.neurons {
            let target_decay = 0.15 - (0.05 * self.modulators.acetylcholine);
            neuron.decay_rate = target_decay;

            let global_target =
                0.20 - (0.05 * self.modulators.dopamine) + (0.15 * self.modulators.cortisol);
            let target_threshold =
                (global_target + if neuron.last_spike { 0.005 } else { -0.001 }).clamp(0.05, 0.50);
            neuron.threshold += (target_threshold - neuron.threshold) * learning_rate;
            neuron.threshold = neuron.threshold.clamp(0.05, 0.50);
        }

        const PRED_ALPHA: f32 = 0.1;
        const PRED_ERR_WEIGHT: f32 = 0.5;
        let mut pred_errors = vec![0.0_f32; self.num_channels];

        for ch in 0..self.num_channels {
            let s = stimuli[ch].abs().clamp(0.0, 1.0);
            pred_errors[ch] = (s - self.predictive_state[ch]).abs();
            self.predictive_state[ch] = PRED_ALPHA * s + (1.0 - PRED_ALPHA) * self.predictive_state[ch];
        }

        let mut rng = rand::rng();
        for (ch, &s) in stimuli.iter().enumerate() {
            let abs_s = s.abs().clamp(0.0, 1.0);
            if abs_s > 0.01 && rng.random::<f32>() < abs_s {
                self.input_spike_times[ch] = self.global_step;
            }
        }

        for neuron in &mut self.neurons {
            let mut total_current = 0.0;
            for ch in 0..self.num_channels {
                if ch >= neuron.weights.len() {
                    continue;
                }
                let stim = stimuli[ch].abs().clamp(0.0, 1.0);
                let surprise = PRED_ERR_WEIGHT * pred_errors[ch];
                total_current += neuron.weights[ch] * (stim + surprise);
            }
            total_current *= 0.45 * stress_multiplier;
            neuron.integrate(total_current);
        }

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

        if !spike_ids.is_empty() {
            const INHIBITION_STRENGTH: f32 = 0.05;
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                if !spike_ids.contains(&i) {
                    neuron.membrane_potential =
                        (neuron.membrane_potential - INHIBITION_STRENGTH).max(0.0);
                }
            }
        }

        self.apply_stdp(learning_rate);

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

        let lif_mean = if !self.neurons.is_empty() {
            let sum: f32 = self.neurons.iter().map(|n| n.membrane_potential).sum();
            sum / self.neurons.len() as f32
        } else {
            0.0
        };

        let iz_drive = (lif_mean * 20.0 + self.modulators.dopamine * 5.0).clamp(0.0, 15.0);
        for iz in &mut self.iz_neurons {
            iz.step(iz_drive);
        }

        Ok(spike_ids)
    }

    fn apply_stdp(&mut self, dopamine_lr: f32) {
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

                neuron.weights[ch] =
                    (neuron.weights[ch] + dw * dopamine_lr).clamp(RM_STDP_W_MIN, RM_STDP_W_MAX);
            }
        }
    }

    /// Get current membrane potentials for all neurons.
    pub fn get_membrane_potentials(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.membrane_potential).collect()
    }

    /// Get current thresholds for all neurons.
    pub fn get_thresholds(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.threshold).collect()
    }

    /// Reset network to initial state.
    pub fn reset(&mut self) {
        self.global_step = 0;
        self.input_spike_times = vec![-1; self.num_channels];
        self.predictive_state = vec![0.0; self.num_channels];

        for neuron in &mut self.neurons {
            neuron.membrane_potential = 0.0;
            neuron.last_spike = false;
            neuron.last_spike_time = -1;
        }

        self.modulators = NeuroModulators::default();
    }
}

impl Default for SpikingNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation_defaults() {
        let network = SpikingNetwork::new();
        assert_eq!(network.neurons.len(), 16);
        assert_eq!(network.iz_neurons.len(), 5);
        assert_eq!(network.num_channels, 16);
        assert_eq!(network.global_step, 0);
    }

    #[test]
    fn test_network_creation_dynamic() {
        let network = SpikingNetwork::with_dimensions(518, 5, 518);
        assert_eq!(network.neurons.len(), 518);
        assert_eq!(network.iz_neurons.len(), 5);
        assert_eq!(network.num_channels, 518);
        assert_eq!(network.input_spike_times.len(), 518);
        assert_eq!(network.predictive_state.len(), 518);
        assert_eq!(network.neurons[0].weights.len(), 518);
    }

    #[test]
    fn test_default_matches_new() {
        let a = SpikingNetwork::new();
        let b = SpikingNetwork::default();
        assert_eq!(a.neurons.len(), b.neurons.len());
        assert_eq!(a.iz_neurons.len(), b.iz_neurons.len());
        assert_eq!(a.num_channels, b.num_channels);
    }

    #[test]
    fn test_network_step() {
        let mut network = SpikingNetwork::new();
        let stimuli = vec![0.5; network.num_channels];
        let modulators = NeuroModulators::default();

        let spikes = network.step(&stimuli, &modulators).expect("valid input length should pass");
        assert_eq!(network.global_step, 1);
        assert!(spikes.len() <= network.neurons.len());
    }

    #[test]
    fn test_step_input_mismatch_returns_error_and_preserves_state() {
        let mut network = SpikingNetwork::new();
        let modulators = NeuroModulators::default();
        let wrong = vec![0.5; network.num_channels - 1];
        let before_step = network.global_step;
        let before_pred = network.predictive_state.clone();
        let before_mod = network.modulators;

        let result = network.step(&wrong, &modulators);

        assert_eq!(
            result,
            Err(StepError::InputLenMismatch {
                expected: network.num_channels,
                got: network.num_channels - 1
            })
        );
        assert_eq!(network.global_step, before_step);
        assert_eq!(network.predictive_state, before_pred);
        assert_eq!(network.modulators, before_mod);
    }

    #[test]
    fn test_membrane_potentials() {
        let network = SpikingNetwork::new();
        let potentials = network.get_membrane_potentials();
        assert_eq!(potentials.len(), 16);
        for &p in &potentials {
            assert_eq!(p, 0.0);
        }
    }
}
