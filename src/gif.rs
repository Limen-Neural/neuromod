//! Generalized Integrate-and-Fire (GIF) neuron model.
//!
//! The GIF neuron extends the classic Leaky Integrate-and-Fire dynamics with a
//! spike-driven adaptation variable that both raises the effective firing
//! threshold and exerts a hyperpolarizing pull on the membrane. It also uses a
//! *soft reset* — subtracting a fraction of the effective threshold rather than
//! clamping to zero — which preserves supra-threshold drive across spikes.
//!
//! Equations (per discrete step):
//! ```text
//! w        ← w · adaptation_decay
//! v        ← v · leak + I · drive_scale − w · adaptation_coupling
//! θ_eff    = base_threshold + w · adaptation_scale
//! if v ≥ θ_eff:
//!     emit spike
//!     v    ← v − θ_eff · reset_ratio           (soft reset)
//!     w    ← w + adaptation_increment
//! ```
//!
//! References:
//! - Mensi, S., Naud, R., Pozzorini, C., Avermann, M., Petersen, C. C. H., &
//!   Gerstner, W. (2012). Parameter extraction and classification of three
//!   cortical neuron types reveals two distinct adaptation mechanisms.
//!   *J. Neurophysiol.*, 107(6), 1756–1775.
//! - Pozzorini, C., Mensi, S., Hagens, O., Naud, R., Koch, C., & Gerstner, W.
//!   (2015). Automated high-throughput characterization of single neurons by
//!   means of simplified spiking models. *PLoS Comput. Biol.*, 11(6), e1004275.
//!
//! The default parameters mirror the production configuration extracted from
//! the author's `corinth-canal` spike-to-embedding pipeline (see
//! `SparseGifHiddenLayer` in that crate's `funnel.rs`). They are a good
//! starting point for ternary-spike driven hidden layers; tune for other
//! regimes.

use serde::{Deserialize, Serialize};

/// Single Generalized Integrate-and-Fire neuron with spike-triggered
/// adaptation and soft reset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GifNeuron {
    /// Current membrane potential `v` (dimensionless).
    pub membrane_potential: f32,
    /// Spike-triggered adaptation variable `w` (dimensionless).
    pub adaptation: f32,
    /// Passive membrane retention per step (`v ← v · leak`).
    pub leak: f32,
    /// Scaling applied to incoming stimulus before integration.
    pub drive_scale: f32,
    /// Current effective firing threshold (runtime-mutable for neuromodulation).
    pub threshold: f32,
    /// Resting threshold baseline, used as the `θ_0` term for the effective
    /// threshold computation and as a restore point for dynamic modulation.
    #[serde(default)]
    pub base_threshold: f32,
    /// How strongly `w` inflates the effective threshold (`θ_eff = θ_0 + w · adaptation_scale`).
    pub adaptation_scale: f32,
    /// Exponential decay applied to `w` every step (`w ← w · adaptation_decay`).
    pub adaptation_decay: f32,
    /// Hyperpolarizing coupling pulling the membrane down each step
    /// proportionally to `w`.
    pub adaptation_coupling: f32,
    /// Jump added to `w` on each emitted spike.
    pub adaptation_increment: f32,
    /// Fraction of the effective threshold subtracted from `v` on a spike
    /// (soft reset — 0.0 keeps `v` untouched, 1.0 subtracts full `θ_eff`).
    pub reset_ratio: f32,
    /// Whether the neuron fired on the last step.
    pub last_spike: bool,
    /// Synaptic weights — one per input channel. Populated by the caller or
    /// the engine.
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Timestep of the most recent spike (`-1` = never fired).
    #[serde(default)]
    pub last_spike_time: i64,
}

impl Default for GifNeuron {
    fn default() -> Self {
        Self {
            membrane_potential: 0.0,
            adaptation: 0.0,
            leak: 0.92,
            drive_scale: 0.75,
            threshold: 0.65,
            base_threshold: 0.65,
            adaptation_scale: 0.22,
            adaptation_decay: 0.94,
            adaptation_coupling: 0.05,
            adaptation_increment: 1.0,
            reset_ratio: 0.35,
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
        }
    }
}

impl GifNeuron {
    /// Create a new GIF neuron with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Integrate one timestep: decay adaptation, then update the membrane with
    /// leak, scaled drive, and adaptation-current coupling.
    pub fn integrate(&mut self, stimulus: f32) {
        self.adaptation *= self.adaptation_decay;
        self.membrane_potential = self.membrane_potential * self.leak
            + stimulus * self.drive_scale
            - self.adaptation * self.adaptation_coupling;
    }

    /// Check whether the neuron fires this step against its effective
    /// threshold. On spike: performs a soft reset on the membrane, increments
    /// `w`, and records the spike time.
    pub fn check_for_spike(&mut self, current_time: i64) -> bool {
        let theta = self.base_threshold + self.adaptation * self.adaptation_scale;
        if self.membrane_potential >= theta {
            self.membrane_potential -= theta * self.reset_ratio;
            self.adaptation += self.adaptation_increment;
            self.last_spike = true;
            self.last_spike_time = current_time;
            true
        } else {
            self.last_spike = false;
            false
        }
    }

    /// Reset the neuron's dynamic state (membrane and adaptation) without
    /// disturbing the learned weights or calibrated thresholds.
    pub fn reset(&mut self) {
        self.membrane_potential = 0.0;
        self.adaptation = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_spike_without_input() {
        let mut n = GifNeuron::new();
        for t in 0..200 {
            n.integrate(0.0);
            assert!(
                !n.check_for_spike(t),
                "GIF neuron should not spike without input"
            );
        }
        assert!(n.membrane_potential.abs() < 1e-6);
        assert!(n.adaptation.abs() < 1e-6);
    }

    #[test]
    fn test_fires_with_sufficient_input() {
        let mut n = GifNeuron::new();
        let mut fired = false;
        for t in 0..200 {
            n.integrate(0.9);
            if n.check_for_spike(t) {
                fired = true;
                break;
            }
        }
        assert!(
            fired,
            "GIF neuron should fire with sustained suprathreshold input"
        );
    }

    #[test]
    fn test_adaptation_increases_after_spike() {
        let mut n = GifNeuron::new();
        n.membrane_potential = 10.0; // force above threshold
        let spiked = n.check_for_spike(0);
        assert!(spiked, "forced high membrane should produce a spike");
        assert!(
            n.adaptation > 0.0,
            "adaptation should accumulate after a spike (got {})",
            n.adaptation
        );
    }

    #[test]
    fn test_soft_reset_not_hard_zero() {
        let mut n = GifNeuron::new();
        n.membrane_potential = 10.0;
        let before = n.membrane_potential;
        let spiked = n.check_for_spike(0);
        assert!(spiked);
        assert!(
            n.membrane_potential < before,
            "membrane should be reduced after spike"
        );
        assert!(
            n.membrane_potential > 0.0,
            "soft reset should leave residual potential (got {}), not clamp to 0",
            n.membrane_potential
        );
    }

    #[test]
    fn test_adaptation_raises_effective_threshold() {
        let mut fresh = GifNeuron::new();
        let mut adapted = GifNeuron::new();
        adapted.adaptation = 5.0; // pre-load heavy adaptation

        let mut fresh_spikes = 0usize;
        let mut adapted_spikes = 0usize;
        for t in 0..200 {
            fresh.integrate(0.9);
            if fresh.check_for_spike(t) {
                fresh_spikes += 1;
            }
            adapted.integrate(0.9);
            if adapted.check_for_spike(t) {
                adapted_spikes += 1;
            }
        }
        assert!(
            fresh_spikes > adapted_spikes,
            "pre-adapted neuron ({adapted_spikes} spikes) should fire less than fresh neuron ({fresh_spikes} spikes) under identical drive"
        );
    }
}
