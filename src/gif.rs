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
//! θ_eff    = θ_0 + w · adaptation_scale
//! if v ≥ θ_eff:
//!     emit spike
//!     v    ← v − θ_eff · reset_ratio           (soft reset)
//!     w    ← w + adaptation_increment
//! ```
//!
//! `θ_0` is [`GifParams::base_threshold`]. On [`GifNeuron`] the live `θ_0` is
//! [`GifNeuron::threshold`] — the runtime-mutable neuromodulation knob —
//! while [`GifNeuron::base_threshold`] is the restore point, matching
//! [`crate::LifNeuron`]. [`GifNeuron::params`] feeds the live value into the
//! shared `GifParams` arithmetic so both representations stay one equation.
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
//!
//! The dynamics live on [`GifParams`], a plain-old-data parameter block shared
//! by the single-neuron [`GifNeuron`] and the structure-of-arrays
//! [`crate::gif_layer::SparseGifHiddenLayer`]. Both paths therefore execute the
//! *same* arithmetic in the same order; there is no second copy of the
//! equations to drift.

use serde::{Deserialize, Serialize};

/// Default passive membrane retention per step.
pub const GIF_LEAK: f32 = 0.92;
/// Default scaling applied to incoming stimulus before integration.
pub const GIF_DRIVE_SCALE: f32 = 0.75;
/// Default resting firing threshold `θ_0`.
pub const GIF_BASE_THRESHOLD: f32 = 0.65;
/// Default coupling from adaptation into the effective threshold.
pub const GIF_ADAPTATION_SCALE: f32 = 0.22;
/// Default per-step exponential decay of the adaptation variable.
pub const GIF_ADAPTATION_DECAY: f32 = 0.94;
/// Default hyperpolarizing coupling from adaptation into the membrane.
pub const GIF_ADAPTATION_COUPLING: f32 = 0.05;
/// Default jump added to the adaptation variable on each spike.
pub const GIF_ADAPTATION_INCREMENT: f32 = 1.0;
/// Default fraction of the effective threshold removed by the soft reset.
pub const GIF_RESET_RATIO: f32 = 0.35;

/// Parameter block for the Generalized Integrate-and-Fire dynamics.
///
/// This is the *shared* definition of the GIF equations. [`GifNeuron`] holds
/// per-neuron state alongside its own copy of these parameters;
/// [`crate::gif_layer::SparseGifHiddenLayer`] holds one `GifParams` for a whole
/// bank of neurons whose state lives in parallel arrays. Keeping the arithmetic
/// here is what makes those two representations numerically identical.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GifParams {
    /// Passive membrane retention per step (`v ← v · leak`).
    pub leak: f32,
    /// Scaling applied to incoming stimulus before integration.
    pub drive_scale: f32,
    /// Resting / live threshold baseline `θ_0`.
    ///
    /// This is the only `θ_0` the shared arithmetic has. A
    /// [`crate::gif_layer::SparseGifHiddenLayer`] stores it once for the whole
    /// bank. [`GifNeuron::params`] copies the neuron's runtime
    /// [`GifNeuron::threshold`] here so firing uses the live knob rather than
    /// the restore point.
    pub base_threshold: f32,
    /// How strongly `w` inflates the effective threshold.
    pub adaptation_scale: f32,
    /// Exponential decay applied to `w` every step.
    pub adaptation_decay: f32,
    /// Hyperpolarizing coupling pulling the membrane down proportionally to `w`.
    pub adaptation_coupling: f32,
    /// Jump added to `w` on each emitted spike.
    pub adaptation_increment: f32,
    /// Fraction of the effective threshold subtracted from `v` on a spike.
    pub reset_ratio: f32,
}

impl Default for GifParams {
    fn default() -> Self {
        Self {
            leak: GIF_LEAK,
            drive_scale: GIF_DRIVE_SCALE,
            base_threshold: GIF_BASE_THRESHOLD,
            adaptation_scale: GIF_ADAPTATION_SCALE,
            adaptation_decay: GIF_ADAPTATION_DECAY,
            adaptation_coupling: GIF_ADAPTATION_COUPLING,
            adaptation_increment: GIF_ADAPTATION_INCREMENT,
            reset_ratio: GIF_RESET_RATIO,
        }
    }
}

impl GifParams {
    /// Effective firing threshold `θ_eff = θ_0 + w · adaptation_scale`.
    ///
    /// `θ_0` is [`Self::base_threshold`]. Callers going through [`GifNeuron`]
    /// get the live [`GifNeuron::threshold`] here via [`GifNeuron::params`].
    #[inline]
    pub fn effective_threshold(&self, adaptation: f32) -> f32 {
        self.base_threshold + adaptation * self.adaptation_scale
    }

    /// Advance one integration step in place: decay adaptation, then update the
    /// membrane with leak, scaled drive, and adaptation-current coupling.
    ///
    /// State is passed by reference rather than owned so that a
    /// structure-of-arrays bank can call this on `membrane[i]` / `adaptation[i]`
    /// without materialising a per-neuron struct.
    #[inline]
    pub fn integrate(&self, membrane: &mut f32, adaptation: &mut f32, stimulus: f32) {
        *adaptation *= self.adaptation_decay;
        *membrane = *membrane * self.leak + stimulus * self.drive_scale
            - *adaptation * self.adaptation_coupling;
    }

    /// Apply the threshold test in place. On spike: soft-reset the membrane and
    /// increment the adaptation variable. Returns whether a spike was emitted.
    #[inline]
    pub fn check_for_spike(&self, membrane: &mut f32, adaptation: &mut f32) -> bool {
        let theta = self.effective_threshold(*adaptation);
        if *membrane >= theta {
            *membrane -= theta * self.reset_ratio;
            *adaptation += self.adaptation_increment;
            true
        } else {
            false
        }
    }
}

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
    /// Live firing threshold `θ_0` (runtime-mutable for neuromodulation).
    ///
    /// This is the value [`Self::check_for_spike`] reads. Adaptation still
    /// inflates it: `θ_eff = threshold + w · adaptation_scale`. Seeded from
    /// [`Self::base_threshold`] at construction; neuromodulation should move
    /// this field, not the restore point. Do not pass it to
    /// [`crate::apply_neuromodulation`]: that helper clamps to `0.05..=0.50`,
    /// below the GIF default of [`GIF_BASE_THRESHOLD`].
    pub threshold: f32,
    /// Resting threshold baseline — the restore point for dynamic modulation.
    ///
    /// Construction copies this into [`Self::threshold`]. Subsequent writes
    /// here do not change firing until the caller (or a modulator) also
    /// updates `threshold`, matching [`crate::LifNeuron`].
    #[serde(default)]
    pub base_threshold: f32,
    /// How strongly `w` inflates the effective threshold
    /// (`θ_eff = threshold + w · adaptation_scale`).
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
        Self::from_params(GifParams::default())
    }
}

impl GifNeuron {
    /// Create a new GIF neuron with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a resting neuron from a shared [`GifParams`] block.
    ///
    /// Both [`Self::threshold`] (live `θ_0`) and [`Self::base_threshold`]
    /// (restore point) are seeded from `params.base_threshold`.
    pub fn from_params(params: GifParams) -> Self {
        Self {
            membrane_potential: 0.0,
            adaptation: 0.0,
            leak: params.leak,
            drive_scale: params.drive_scale,
            threshold: params.base_threshold,
            base_threshold: params.base_threshold,
            adaptation_scale: params.adaptation_scale,
            adaptation_decay: params.adaptation_decay,
            adaptation_coupling: params.adaptation_coupling,
            adaptation_increment: params.adaptation_increment,
            reset_ratio: params.reset_ratio,
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
        }
    }

    /// Snapshot this neuron's live dynamics parameters as a shared [`GifParams`].
    ///
    /// [`GifParams::base_threshold`] is filled from [`Self::threshold`], not
    /// from [`Self::base_threshold`]. The restore point is a `GifNeuron`
    /// concern; the shared arithmetic only has one `θ_0`, and firing must
    /// use the runtime-mutable knob.
    pub fn params(&self) -> GifParams {
        GifParams {
            leak: self.leak,
            drive_scale: self.drive_scale,
            base_threshold: self.threshold,
            adaptation_scale: self.adaptation_scale,
            adaptation_decay: self.adaptation_decay,
            adaptation_coupling: self.adaptation_coupling,
            adaptation_increment: self.adaptation_increment,
            reset_ratio: self.reset_ratio,
        }
    }

    /// Integrate one timestep: decay adaptation, then update the membrane with
    /// leak, scaled drive, and adaptation-current coupling.
    pub fn integrate(&mut self, stimulus: f32) {
        let params = self.params();
        params.integrate(&mut self.membrane_potential, &mut self.adaptation, stimulus);
    }

    /// Check whether the neuron fires this step against its effective
    /// threshold (`threshold + adaptation * adaptation_scale`). On spike:
    /// performs a soft reset on the membrane, increments `w`, and records
    /// the spike time.
    pub fn check_for_spike(&mut self, current_time: i64) -> bool {
        let params = self.params();
        let fired = params.check_for_spike(&mut self.membrane_potential, &mut self.adaptation);
        self.last_spike = fired;
        if fired {
            self.last_spike_time = current_time;
        }
        fired
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
    fn test_params_round_trip() {
        let params = GifParams::default();
        assert_eq!(GifNeuron::from_params(params).params(), params);
        assert_eq!(GifNeuron::new().params(), GifParams::default());
    }

    #[test]
    fn test_neuron_and_params_dynamics_agree_bitwise() {
        // `GifNeuron` delegates to `GifParams`; a structure-of-arrays bank uses
        // the same call. Agreement holds while `threshold == base_threshold`
        // (the default). Mutating the live knob is covered separately.
        let params = GifParams::default();
        let mut neuron = GifNeuron::new();
        let (mut v, mut w) = (0.0f32, 0.0f32);

        for t in 0..100i64 {
            let stimulus = if t % 4 == 0 { 1.1 } else { 0.05 };
            neuron.integrate(stimulus);
            let neuron_fired = neuron.check_for_spike(t);

            params.integrate(&mut v, &mut w, stimulus);
            let params_fired = params.check_for_spike(&mut v, &mut w);

            assert_eq!(neuron_fired, params_fired, "spike mismatch at t={t}");
            assert_eq!(neuron.membrane_potential, v, "membrane mismatch at t={t}");
            assert_eq!(neuron.adaptation, w, "adaptation mismatch at t={t}");
        }
    }

    #[test]
    fn mutating_threshold_changes_when_the_neuron_fires() {
        // Default θ_0 is GIF_BASE_THRESHOLD; adaptation is 0 so θ_eff == threshold.
        let mut at_rest = GifNeuron::default();
        at_rest.membrane_potential = at_rest.base_threshold;
        assert!(
            at_rest.check_for_spike(0),
            "membrane at default θ_0 must fire"
        );

        let mut raised = GifNeuron::default();
        raised.membrane_potential = raised.base_threshold;
        raised.threshold = raised.base_threshold + 0.25;
        assert!(
            !raised.check_for_spike(0),
            "raising threshold above the membrane must suppress the spike"
        );

        let mut lowered = GifNeuron::default();
        lowered.membrane_potential = lowered.base_threshold - 0.2;
        lowered.threshold = lowered.base_threshold - 0.2;
        assert!(
            lowered.check_for_spike(0),
            "lowering threshold to the membrane must produce a spike"
        );
    }

    #[test]
    fn base_threshold_is_restore_point_not_live_firing_knob() {
        let mut neuron = GifNeuron::default();
        neuron.membrane_potential = neuron.threshold;
        neuron.base_threshold = 1.0e6;
        assert!(
            neuron.check_for_spike(0),
            "mutating base_threshold must not by itself change firing"
        );
        assert_eq!(
            neuron.params().base_threshold,
            neuron.threshold,
            "params() must snapshot the live threshold, not the restore point"
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
