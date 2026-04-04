//! Lapicque (1907) Integrate-and-Fire neuron model — the biological root of all
//! spiking neuron models.
//!
//! The simplest possible model that captures the "integrate and fire" behaviour
//! of real neurons: the membrane potential integrates incoming current and fires
//! a spike the moment it crosses a threshold, after which it resets.
//!
//! Equation:
//! ```text
//! dv/dt = −v/τ + I(t)
//! ```
//! When `v ≥ threshold`, emit a spike and reset `v = 0`.
//!
//! Reference:
//! - Lapicque, L. (1907). Recherches quantitatives sur l'excitation électrique des
//!   nerfs traitée comme une polarisation. *J. Physiol. Pathol. Gén.*, 9, 620–635.

use serde::{Deserialize, Serialize};

/// Lapicque (1907) pure Integrate-and-Fire neuron.
///
/// This is the original neuron model — a single variable (membrane potential)
/// that integrates input, leaks toward rest, and resets after a spike.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LapicqueNeuron {
    /// Current membrane potential (dimensionless).
    pub membrane_potential: f32,
    /// Passive leak rate per step (fraction of potential lost).
    pub decay_rate: f32,
    /// Firing threshold.
    pub threshold: f32,
    /// Resting threshold (used for dynamic modulation).
    pub base_threshold: f32,
    /// Whether the neuron fired on the last step.
    pub last_spike: bool,
    /// Synaptic weights for each input channel.
    pub weights: Vec<f32>,
    /// Timestep of the most recent spike (-1 = never).
    pub last_spike_time: i64,
}

impl Default for LapicqueNeuron {
    fn default() -> Self {
        Self {
            membrane_potential: 0.0,
            decay_rate: 0.15,
            threshold: 0.02,
            base_threshold: 0.02,
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
        }
    }
}

impl LapicqueNeuron {
    /// Create a new Lapicque neuron with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Integrate one step of incoming stimulus and apply passive leak.
    ///
    /// Implements: `v ← (v + stimulus) × (1 − decay_rate)`
    pub fn integrate(&mut self, stimulus: f32) {
        self.membrane_potential += stimulus;
        self.membrane_potential *= 1.0 - self.decay_rate;
    }

    /// Check whether the neuron fires this step.
    ///
    /// If `membrane_potential ≥ threshold`, resets the potential to 0,
    /// records the spike time, and returns `true`.
    pub fn check_for_spike(&mut self, current_time: i64) -> bool {
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0;
            self.last_spike = true;
            self.last_spike_time = current_time;
            true
        } else {
            self.last_spike = false;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_spike_without_input() {
        let mut n = LapicqueNeuron::new();
        for t in 0..100 {
            n.integrate(0.0);
            assert!(!n.check_for_spike(t), "should not spike without input");
        }
    }

    #[test]
    fn test_fires_with_sufficient_input() {
        let mut n = LapicqueNeuron::new();
        let mut fired = false;
        for t in 0..1000 {
            n.integrate(0.05);
            if n.check_for_spike(t) {
                fired = true;
                break;
            }
        }
        assert!(fired, "Lapicque neuron should fire with sustained suprathreshold input");
    }

    #[test]
    fn test_reset_after_spike() {
        let mut n = LapicqueNeuron::new();
        n.membrane_potential = 1.0; // force above threshold
        n.check_for_spike(0);
        assert_eq!(n.membrane_potential, 0.0, "potential should reset to 0 after spike");
    }

    #[test]
    fn test_spike_time_recorded() {
        let mut n = LapicqueNeuron::new();
        n.membrane_potential = 1.0;
        n.check_for_spike(42);
        assert_eq!(n.last_spike_time, 42);
    }

    #[test]
    fn test_leak_reduces_potential() {
        let mut n = LapicqueNeuron::new();
        n.membrane_potential = 1.0;
        n.integrate(0.0);
        assert!(n.membrane_potential < 1.0, "leak should reduce membrane potential");
    }
}
