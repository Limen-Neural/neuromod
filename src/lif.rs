//! # Leaky integrate-and-fire (LIF) neurons
//!
//! Primary bank of [`crate::SpikingNetwork`]: each [`LifNeuron`] has a membrane
//! potential, threshold, decay, and a vector of synaptic weights (one per input
//! channel), each weight paired with an [`EligibilityTrace`]. The engine
//! integrates, fires, applies lateral inhibition, then decays and accumulates
//! the traces and converts them into weight changes under a dopamine gate.
//!
//! [`PoissonEncoder`] is a small helper that turns a scalar intensity into a
//! binary spike train (Bernoulli trials). It is **not** required by
//! `SpikingNetwork::step` (the engine encodes stimuli itself), but is useful in
//! demos and tests. [`PoissonEncoder::encode_with_rng`] accepts a caller RNG so
//! a seeded stream can reproduce a train; [`PoissonEncoder::encode`] keeps the
//! thread-local convenience wrapper.
//!
//! For the classical Lapicque root model, see [`crate::lapicque`]. For the
//! secondary engine bank, see [`crate::izhikevich`].

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

use crate::rm_stdp::EligibilityTrace;

const LIF_BASE_THRESHOLD: f32 = 0.02;

fn default_base_threshold() -> f32 {
    LIF_BASE_THRESHOLD
}

fn never_spiked() -> i64 {
    -1
}

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
    ///
    /// Draws from the thread-local RNG ([`rand::rng`]). For a reproducible
    /// train, use [`Self::encode_with_rng`].
    pub fn encode(&self, input: f32) -> Vec<u8> {
        self.encode_with_rng(input, &mut rand::rng())
    }

    /// Encode using a caller-injected RNG for the Bernoulli trials.
    ///
    /// Same contract as [`Self::encode`]: `input` is clamped to `[0, 1]`, a
    /// zero probability yields an all-zero train, and a probability of `1.0`
    /// yields an all-ones train without drawing from `rng`. Intermediate
    /// probabilities consume one uniform draw per step from the same `rng`.
    ///
    /// The generator is not stored on [`PoissonEncoder`] and is not serialized.
    pub fn encode_with_rng<R: Rng + ?Sized>(&self, input: f32, rng: &mut R) -> Vec<u8> {
        let mut spikes = Vec::with_capacity(self.num_steps);

        // Clamp input to ensure probability is valid (0% to 100%)
        let probability = input.clamp(0.0, 1.0);

        for _ in 0..self.num_steps {
            // Bernoulli trial: spike if U(0,1) < intensity.
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
    #[serde(default = "default_base_threshold")]
    pub base_threshold: f32,
    pub last_spike: bool, // Tracks if it fired in the last step
    /// Synaptic weights — one per input channel.
    /// These are learned via STDP during training.
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Timestep of the most recent spike (for STDP delta-t calculation).
    ///
    /// Shares the engine's discrete step unit: `-1` means this neuron has never
    /// spiked; a non-negative value is the [`crate::SpikingNetwork::global_step`]
    /// at which it last fired (`1..=i64::MAX` on a live network).
    #[serde(default = "never_spiked")]
    pub last_spike_time: i64,
    /// Per-synapse eligibility traces — one per input channel, indexed exactly
    /// like [`Self::weights`].
    ///
    /// The engine decays these every step and accumulates a pre/post
    /// coincidence on the step a spike occurs, then converts them into weight
    /// changes when dopamine is present. Empty on a neuron built outside the
    /// engine (and on pre-0.6 deserialized state); the engine resizes it to
    /// match `weights` before use.
    #[serde(default)]
    pub eligibility: Vec<EligibilityTrace>,
}

impl Default for LifNeuron {
    fn default() -> Self {
        Self {
            membrane_potential: 0.0,
            decay_rate: 0.15,
            threshold: LIF_BASE_THRESHOLD, // Aggressively lowered threshold
            base_threshold: LIF_BASE_THRESHOLD,
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
            eligibility: Vec::new(),
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
    use rand::SeedableRng;
    use rand::rngs::StdRng;

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
        assert!(neuron.eligibility.is_empty());
    }

    #[test]
    fn missing_checkpoint_sentinels_match_constructor_defaults() {
        let mut value = serde_json::to_value(LifNeuron::default()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("base_threshold");
        object.remove("last_spike_time");

        let restored: LifNeuron = serde_json::from_value(value).unwrap();
        assert_eq!(restored.base_threshold, LifNeuron::new().base_threshold);
        assert_eq!(restored.last_spike_time, LifNeuron::new().last_spike_time);
    }

    #[test]
    fn explicit_checkpoint_sentinels_are_preserved() {
        let mut value = serde_json::to_value(LifNeuron::default()).unwrap();
        value["base_threshold"] = serde_json::json!(0.37);
        value["last_spike_time"] = serde_json::json!(42);

        let restored: LifNeuron = serde_json::from_value(value).unwrap();
        assert_eq!(restored.base_threshold, 0.37);
        assert_eq!(restored.last_spike_time, 42);
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

    fn poisson_rng_hash(spikes: &[u8]) -> u64 {
        spikes.iter().fold(0xC0FF_EE01_u64, |h, &s| {
            h.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(s as u64)
        })
    }

    #[test]
    fn poisson_encoder_rng_injection_is_reproducible() {
        let encoder = PoissonEncoder::new(256);
        let mut rng_a = StdRng::seed_from_u64(0xA11CE);
        let mut rng_b = StdRng::seed_from_u64(0xA11CE);
        let spikes_a = encoder.encode_with_rng(0.4, &mut rng_a);
        let spikes_b = encoder.encode_with_rng(0.4, &mut rng_b);
        assert_eq!(spikes_a, spikes_b);
        let hash = poisson_rng_hash(&spikes_a);
        println!("poisson seeded trace hash: {hash:#018x}");
        assert_eq!(hash, poisson_rng_hash(&spikes_b));
    }

    #[test]
    fn poisson_encoder_different_rng_seeds_diverge() {
        let encoder = PoissonEncoder::new(512);
        let mut rng_a = StdRng::seed_from_u64(1);
        let mut rng_b = StdRng::seed_from_u64(2);
        let spikes_a = encoder.encode_with_rng(0.5, &mut rng_a);
        let spikes_b = encoder.encode_with_rng(0.5, &mut rng_b);
        let mismatches = spikes_a
            .iter()
            .zip(spikes_b.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            mismatches > 50,
            "independent seeds should disagree often at p=0.5, got {mismatches} / 512"
        );
    }

    #[test]
    fn poisson_encoder_rng_stream_is_not_recreated_per_step() {
        let encoder = PoissonEncoder::new(32);
        let mut rng = StdRng::seed_from_u64(99);
        let first = encoder.encode_with_rng(0.5, &mut rng);
        let second = encoder.encode_with_rng(0.5, &mut rng);

        let mut reseeded = StdRng::seed_from_u64(99);
        let reseeded_first = encoder.encode_with_rng(0.5, &mut reseeded);
        let mut reseeded_again = StdRng::seed_from_u64(99);
        let reseeded_second = encoder.encode_with_rng(0.5, &mut reseeded_again);

        assert_eq!(first, reseeded_first);
        assert_ne!(
            second, reseeded_second,
            "continuing the same stream must not match a fresh seed"
        );
    }
}
