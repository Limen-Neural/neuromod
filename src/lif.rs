use rand::RngExt;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoissonEncoder {
    pub num_steps: usize,
}

impl PoissonEncoder {
    pub fn new(steps: usize) -> Self {
        Self { num_steps: steps }
    }

    /// Encodes a normalized value (0.0 - 1.0) into a temporal spike train.
    ///
    /// PHYSICS ANALOGY:
    /// This acts like a "Geiger Counter" for your data.
    /// High Intensity (Molarity/Voltage) = High Click Rate (Spikes).
    pub fn encode(&self, input: f32) -> Vec<u8> {
        let mut rng = rand::rng();
        let mut spikes = Vec::with_capacity(self.num_steps);

        // Clamp input to ensure probability is valid (0% to 100%)
        let probability = input.clamp(0.0, 1.0);

        for _ in 0..self.num_steps {
            // Stochastic firing:
            // If the random number (0.0-1.0) is LESS than our intensity, we spike.
            // This mimics the noise inherent in quantum/chemical systems.
            //
            // Handle exact 0.0 / 1.0 without RNG to make edge-case behavior
            // explicit and avoid RNG calls for deterministic paths.
            let fire = if probability <= 0.0 {
                false
            } else if probability >= 1.0 {
                true
            } else {
                rng.random_range(0.0..1.0) < probability
            };
            spikes.push(u8::from(fire));
        }
        spikes
    }
}

/// This struct simulates the physical properties of a biological neuron.
///
/// CIRCUIT ANALOGY (RC Circuit):
/// - Membrane Potential = Voltage across a Capacitor.
/// - Decay Rate = Current leakage through a Resistor.
/// - Threshold = Breakdown voltage of a component (like a Diode or Spark Gap).
/// - Weights = Resistor values on each input trace (synaptic strength).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LifNeuron {
    pub membrane_potential: f32, // Current charge state
    pub decay_rate: f32,         // How fast it "forgets" (Leak)
    pub threshold: f32,          // Limit to trigger an action potential
    /// Resting threshold baseline used for dynamic threshold modulation
    /// without losing the original calibrated value.
    #[serde(default)]
    pub base_threshold: f32,
    pub last_spike: bool, // Tracks if it fired in the last step
    /// Synaptic weights — one per input channel.
    /// These are learned via STDP during training.
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Timestep of the most recent spike (for STDP delta-t calculation).
    /// Uses a global step counter maintained by the engine.
    #[serde(default)]
    pub last_spike_time: i64,
}

impl Default for LifNeuron {
    fn default() -> Self {
        Self {
            membrane_potential: 0.0,
            decay_rate: 0.15,
            threshold: 0.02, // Aggressively lowered threshold
            base_threshold: 0.02,
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
        }
    }
}

impl LifNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    /// The Core Logic Step:
    /// 1. Add Input (Integration)
    /// 2. Lose Charge (Leak)
    pub fn integrate(&mut self, stimulus: f32) {
        // CHARGE: Add input stimulus to current state
        self.membrane_potential += stimulus;

        // LEAK: Passive decay over time (Simulates real-world signal loss)
        self.membrane_potential -= self.membrane_potential * self.decay_rate;
    }

    /// Check if the neuron should fire.
    /// If yes, captures the peak potential, then performs a hard reset (Refractory Period).
    /// Returns `Some(peak_potential)` on a spike, `None` otherwise.
    /// Capturing before reset lets debug logs show the actual firing voltage, not the post-reset 0.0.
    pub fn check_fire(&mut self) -> Option<f32> {
        if self.membrane_potential >= self.threshold {
            let peak = self.membrane_potential; // Capture BEFORE reset
            self.membrane_potential = 0.0; // Hard reset after spike
            return Some(peak);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_neuron_has_expected_initial_state() {
        let neuron = LifNeuron::new();
        assert_eq!(neuron.membrane_potential, 0.0);
        assert_eq!(neuron.decay_rate, 0.15);
        assert_eq!(neuron.threshold, 0.02);
        assert_eq!(neuron.base_threshold, 0.02);
        assert!(!neuron.last_spike);
        assert!(neuron.weights.is_empty());
        assert_eq!(neuron.last_spike_time, -1);
    }

    #[test]
    fn integrate_charges_then_leaks() {
        let mut neuron = LifNeuron::new();
        neuron.integrate(1.0);
        let expected = 1.0 - 1.0 * neuron.decay_rate;
        assert!((neuron.membrane_potential - expected).abs() < 1e-6);
    }

    #[test]
    fn integrate_accumulates_over_multiple_calls() {
        let mut neuron = LifNeuron::new();
        neuron.integrate(0.5);
        let after_first = neuron.membrane_potential;
        let expected_after_second = after_first + 0.5 - (after_first + 0.5) * neuron.decay_rate;
        neuron.integrate(0.5);
        assert!((neuron.membrane_potential - expected_after_second).abs() < 1e-6);
    }

    #[test]
    fn check_fire_below_threshold_returns_none_and_leaves_potential_unchanged() {
        let mut neuron = LifNeuron::new();
        neuron.membrane_potential = neuron.threshold - 0.01;
        let before = neuron.membrane_potential;

        assert_eq!(neuron.check_fire(), None);
        assert_eq!(neuron.membrane_potential, before);
    }

    #[test]
    fn check_fire_at_or_above_threshold_fires_and_hard_resets() {
        let mut neuron = LifNeuron::new();

        neuron.membrane_potential = neuron.threshold;
        let expected_peak_exact = neuron.membrane_potential;
        let fired_exact = neuron.check_fire();
        assert_eq!(fired_exact, Some(expected_peak_exact));
        assert_eq!(neuron.membrane_potential, 0.0);

        neuron.membrane_potential = neuron.threshold + 0.05;
        let expected_peak_above = neuron.membrane_potential;
        let fired_above = neuron.check_fire();
        assert_eq!(fired_above, Some(expected_peak_above));
        assert_eq!(neuron.membrane_potential, 0.0);
    }

    #[test]
    fn poisson_encoder_zero_input_yields_all_zero_spike_train() {
        let encoder = PoissonEncoder::new(50);
        let spikes = encoder.encode(0.0);
        assert_eq!(spikes.len(), 50);
        assert!(spikes.iter().all(|&s| s == 0));
    }

    #[test]
    fn poisson_encoder_negative_input_clamps_to_zero_spikes() {
        let encoder = PoissonEncoder::new(20);
        let spikes = encoder.encode(-5.0);
        assert!(spikes.iter().all(|&s| s == 0));
    }

    #[test]
    fn poisson_encoder_full_intensity_input_yields_all_ones() {
        let encoder = PoissonEncoder::new(50);
        let spikes = encoder.encode(1.0);
        assert_eq!(spikes.len(), 50);
        assert!(spikes.iter().all(|&s| s == 1));
    }

    #[test]
    fn poisson_encoder_output_length_matches_num_steps() {
        for steps in [0, 1, 10, 100] {
            let encoder = PoissonEncoder::new(steps);
            assert_eq!(encoder.encode(0.5).len(), steps);
        }
    }
}
