//! Classical (unmodulated) Hebbian STDP.
//!
//! Donald O. Hebb (1949): *"When an axon of cell A is near enough to excite
//! a cell B and repeatedly or persistently takes part in firing it, some
//! growth process or metabolic change takes place in one or both cells such
//! that A's efficiency, as one of the cells firing B, is increased."*
//!
//! This module implements the classic Spike-Timing-Dependent Plasticity (STDP)
//! rule without any reward or neuromodulator multiplier.  It is the unmodulated
//! foundation that `rm_stdp` extends with dopamine gating.
//!
//! Reference:
//! - Hebb, D.O. (1949). *The Organization of Behavior*. Wiley.

use crate::izhikevich::IzhikevichNeuron;

/// Hyperparameters for the classical STDP learning rule.
#[derive(Debug, Clone, Copy)]
pub struct StdpParams {
    /// Maximum LTP (long-term potentiation) amplitude.
    pub a_plus: f32,
    /// Maximum LTD (long-term depression) amplitude.
    pub a_minus: f32,
    /// LTP time constant (steps).
    pub tau_plus: f32,
    /// LTD time constant (steps).
    pub tau_minus: f32,
    /// Minimum synaptic weight.
    pub w_min: f32,
    /// Maximum synaptic weight.
    pub w_max: f32,
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_min: 0.0,
            w_max: 2.0,
        }
    }
}

/// Apply pure Hebbian STDP and return the updated synaptic weight.
///
/// - Pre fires before post (`delta_t > 0`) → LTP (weight increase).
/// - Post fires before pre (`delta_t < 0`) → LTD (weight decrease).
///
/// `delta_t = post_spike_time − pre_spike_time`
pub fn apply_classical_stdp(
    pre_spike_time: i64,
    post_spike_time: i64,
    current_weight: f32,
    params: &StdpParams,
) -> f32 {
    let delta_t = post_spike_time - pre_spike_time;
    let weight_change = if delta_t > 0 {
        params.a_plus * (-delta_t as f32 / params.tau_plus).exp()
    } else if delta_t < 0 {
        -params.a_minus * (delta_t as f32 / params.tau_minus).exp()
    } else {
        0.0
    };
    (current_weight + weight_change).clamp(params.w_min, params.w_max)
}

/// Minimal Izhikevich network with classical Hebbian STDP weights.
///
/// Demonstrates how `apply_classical_stdp` integrates into a network loop.
/// In a production setting the STDP call would live inside the main step loop
/// where pre/post spike times are tracked per synapse.
pub struct HebbianIzhikevichNetwork {
    pub neurons: Vec<IzhikevichNeuron>,
    /// Flat synaptic weight matrix: `weights[pre * N + post]`.
    pub weights: Vec<f32>,
    pub stdp_params: StdpParams,
}

impl HebbianIzhikevichNetwork {
    /// Create a fully-connected network of `num_neurons` regular-spiking neurons.
    pub fn new(num_neurons: usize) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| IzhikevichNeuron::new_regular_spiking())
            .collect();
        let weights = vec![0.5f32; num_neurons * num_neurons];
        Self { neurons, weights, stdp_params: StdpParams::default() }
    }

    /// Update the synapse from `pre_index` → `post_index` using classical STDP.
    pub fn update_weights(&mut self, pre_index: usize, post_index: usize) {
        let n = self.neurons.len();
        let pre_t  = self.neurons[pre_index].last_spike_time;
        let post_t = self.neurons[post_index].last_spike_time;
        let w = self.weights[pre_index * n + post_index];
        self.weights[pre_index * n + post_index] =
            apply_classical_stdp(pre_t, post_t, w, &self.stdp_params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltp_when_pre_before_post() {
        let params = StdpParams::default();
        let w0 = 0.5;
        let w1 = apply_classical_stdp(0, 5, w0, &params);
        assert!(w1 > w0, "Pre before post should potentiate (LTP)");
    }

    #[test]
    fn test_ltd_when_post_before_pre() {
        let params = StdpParams::default();
        let w0 = 0.5;
        let w1 = apply_classical_stdp(5, 0, w0, &params);
        assert!(w1 < w0, "Post before pre should depress (LTD)");
    }

    #[test]
    fn test_no_change_simultaneous_spikes() {
        let params = StdpParams::default();
        let w0 = 0.5;
        let w1 = apply_classical_stdp(3, 3, w0, &params);
        assert_eq!(w1, w0, "Simultaneous spikes should produce no weight change");
    }

    #[test]
    fn test_weight_clamped_to_bounds() {
        let params = StdpParams::default();
        // Drive weight toward max
        let mut w = 1.99;
        for _ in 0..100 { w = apply_classical_stdp(0, 1, w, &params); }
        assert!(w <= params.w_max, "Weight should not exceed w_max");
        // Drive weight toward min
        let mut w = 0.01;
        for _ in 0..100 { w = apply_classical_stdp(1, 0, w, &params); }
        assert!(w >= params.w_min, "Weight should not go below w_min");
    }

    #[test]
    fn test_hebbian_network_update() {
        let mut net = HebbianIzhikevichNetwork::new(3);
        // Step neurons to produce spike times
        for t in 0..50i64 { net.neurons[0].step_with_time(10.0, t); }
        for t in 0..50i64 { net.neurons[1].step_with_time(10.0, t + 5); }
        let w_before = net.weights[0 * 3 + 1];
        net.update_weights(0, 1);
        // Weight should change if both neurons have fired
        let w_after = net.weights[0 * 3 + 1];
        assert_ne!(w_before, w_after, "Weight should update after neurons have spiked");
    }
}
