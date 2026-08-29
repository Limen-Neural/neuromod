//! R-STDP (Reward-modulated Spike-Timing-Dependent Plasticity) parameters.
//!
//! This module holds the R-STDP constants, [`RmStdpConfig`], and the
//! [`EligibilityTrace`] building block that the engine learns through.
//!
//! R-STDP in one sentence: an eligibility trace accumulates a "memory" of
//! recent pre/post spike-timing coincidences, then a reward signal (dopamine)
//! converts that trace into a weight change.
//!
//! **Live engine path:** `SpikingNetwork::apply_stdp` (`src/engine.rs`) keeps
//! one [`EligibilityTrace`] per synapse in [`crate::LifNeuron::eligibility`]. Every
//! step it decays each trace and accumulates a coincidence on the step where the
//! post neuron or the pre channel actually spiked — **regardless of dopamine**.
//! Only the trace → weight conversion is dopamine-gated. Weight updates therefore
//! flow *through* the trace, not around it, so a coincidence recorded during a
//! reward-free step can still be paid out when reward arrives later.
//!
//! Decision record (wire, not demote): [ADR 002][adr].
//!
//! [adr]: https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/002-wire-eligibility-traces.md
//!
//! ANALOGY: Hebb's Rule on a timer — "neurons that fire together wire
//! together," but only if the timing *and* the reward are right.

use serde::{Deserialize, Serialize};

/// LTP (potentiation) time constant, in steps.
pub const RM_STDP_TAU_PLUS: f32 = 20.0;
/// LTD (depression) time constant, in steps.
pub const RM_STDP_TAU_MINUS: f32 = 20.0;
/// Maximum LTP amplitude.
pub const RM_STDP_A_PLUS: f32 = 0.01;
/// Maximum LTD amplitude (slightly stronger than LTP for stability).
pub const RM_STDP_A_MINUS: f32 = 0.012;
/// Minimum synaptic weight (no negative/inhibitory weights yet).
pub const RM_STDP_W_MIN: f32 = 0.0;
/// Maximum synaptic weight (prevents runaway excitation).
pub const RM_STDP_W_MAX: f32 = 2.0;
/// Default eligibility-trace decay time constant, in steps.
///
/// Longer than the LTP/LTD kernels ([`RM_STDP_TAU_PLUS`] / [`RM_STDP_TAU_MINUS`]):
/// the kernel decides *how much* credit a coincidence earns, the trace decides
/// *how long* that credit stays claimable before reward arrives.
pub const RM_STDP_TAU_ELIGIBILITY: f32 = 50.0;
/// Default learning rate for converting an eligibility trace into a weight change.
pub const RM_STDP_REWARD_LR: f32 = 0.05;

const _: () = assert!(RM_STDP_W_MIN < RM_STDP_W_MAX);
const _: () = assert!(RM_STDP_A_MINUS >= RM_STDP_A_PLUS);
const _: () = assert!(RM_STDP_TAU_ELIGIBILITY > 0.0);
const _: () = assert!(RM_STDP_REWARD_LR > 0.0);

/// Eligibility trace for a single synapse.
///
/// Accumulates on pre/post spike timing and decays exponentially over time;
/// positive values favor potentiation (LTP), negative values favor depression
/// (LTD). Each synapse holds its own trace instance — the engine keeps one per
/// input channel in [`crate::LifNeuron::eligibility`].
///
/// ```
/// use neuromod::EligibilityTrace;
///
/// let mut trace = EligibilityTrace::new(50.0);
/// assert_eq!(trace.value, 0.0);
///
/// // Pre fired one step before post: potentiation.
/// trace.accumulate(1.0);
/// assert!(trace.value > 0.0);
///
/// // The imprint fades unless reward converts it into a weight change.
/// let peak = trace.value;
/// trace.decay();
/// assert!(trace.value < peak);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EligibilityTrace {
    /// Current trace value.
    pub value: f32,
    /// Decay time constant, in steps. Typical values are 50-100.
    pub tau: f32,
}

impl Default for EligibilityTrace {
    fn default() -> Self {
        Self::new(RM_STDP_TAU_ELIGIBILITY)
    }
}

/// R-STDP hyperparameters.
///
/// Held by `SpikingNetwork::stdp_config`; see
/// [`SpikingNetwork::set_rm_stdp_config`](crate::SpikingNetwork::set_rm_stdp_config)
/// to change it on a live network.
///
/// `w_min` / `w_max` take precedence over the engine's L1 weight budget: the
/// renormalization pass in `step` scales toward the budget and then clamps, so
/// narrowing this range leaves each neuron's weight sum off budget by however
/// much the clamp binds. The defaults cannot bind (weights are non-negative and
/// `w_max` equals the budget), so the budget holds exactly under them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RmStdpConfig {
    /// Eligibility trace decay time constant, in steps. Typical values are 50-100.
    pub tau_eligibility: f32,
    /// Learning rate for converting an eligibility trace into a weight change
    /// when a reward signal arrives. Typical values are 0.01-0.1.
    pub reward_lr: f32,
    /// Minimum weight (no negative/inhibitory weights yet).
    ///
    /// Bounds weight *updates* — the clamp in `apply_stdp` and the L1
    /// renormalization pass. It is not a floor imposed on untouched weights: a
    /// network starts with blank (zero) weights by design, and renormalization
    /// skips a neuron whose weights still sum to ~0, so a positive `w_min`
    /// never seeds one.
    pub w_min: f32,
    /// Maximum weight (prevents runaway excitation).
    pub w_max: f32,
}

impl Default for RmStdpConfig {
    fn default() -> Self {
        Self {
            tau_eligibility: RM_STDP_TAU_ELIGIBILITY,
            reward_lr: RM_STDP_REWARD_LR,
            w_min: RM_STDP_W_MIN,
            w_max: RM_STDP_W_MAX,
        }
    }
}

impl RmStdpConfig {
    /// Weight bounds as an ordered `(min, max)` pair, safe to hand to
    /// [`f32::clamp`].
    ///
    /// `w_min` and `w_max` are public fields, so a caller can leave them
    /// reversed or non-finite — which would make `clamp` panic. This falls back
    /// to [`RM_STDP_W_MIN`] / [`RM_STDP_W_MAX`] in that case rather than taking
    /// the engine down mid-step.
    ///
    /// ```
    /// use neuromod::RmStdpConfig;
    ///
    /// let sane = RmStdpConfig { w_min: 0.2, w_max: 1.5, ..RmStdpConfig::default() };
    /// assert_eq!(sane.weight_bounds(), (0.2, 1.5));
    ///
    /// let reversed = RmStdpConfig { w_min: 1.5, w_max: 0.2, ..RmStdpConfig::default() };
    /// assert_eq!(reversed.weight_bounds(), (0.0, 2.0));
    /// ```
    pub fn weight_bounds(&self) -> (f32, f32) {
        if !self.w_min.is_finite() || !self.w_max.is_finite() || self.w_min > self.w_max {
            return (RM_STDP_W_MIN, RM_STDP_W_MAX);
        }
        (self.w_min, self.w_max)
    }

    /// Reward learning rate, guarded against a non-finite value.
    ///
    /// `reward_lr` is public too, and a `NaN` rate is worse than a bad clamp: it
    /// poisons a weight on the first rewarded step, and the L1 renormalization
    /// pass then skips that neuron forever, because a `NaN` total is not
    /// `> 1e-6`. The corruption would never clear. Falls back to
    /// [`RM_STDP_REWARD_LR`].
    ///
    /// A finite negative rate is passed through — inverting the sign of learning
    /// is a legitimate (if unusual) choice, unlike `NaN`.
    ///
    /// ```
    /// use neuromod::RmStdpConfig;
    ///
    /// let poisoned = RmStdpConfig { reward_lr: f32::NAN, ..RmStdpConfig::default() };
    /// assert_eq!(poisoned.effective_reward_lr(), 0.05);
    ///
    /// let tuned = RmStdpConfig { reward_lr: 0.02, ..RmStdpConfig::default() };
    /// assert_eq!(tuned.effective_reward_lr(), 0.02);
    /// ```
    pub fn effective_reward_lr(&self) -> f32 {
        if self.reward_lr.is_finite() {
            self.reward_lr
        } else {
            RM_STDP_REWARD_LR
        }
    }

    /// Eligibility time constant, guarded against a non-finite or non-positive
    /// value.
    ///
    /// Both failure modes are silent rather than loud: a `NaN` tau degrades
    /// [`EligibilityTrace::decay`] to `exp(-1/f32::EPSILON) == 0`, erasing every
    /// banked trace on the next step, and `+∞` gives `exp(-0) == 1`, disabling
    /// decay altogether. Either falls back to [`RM_STDP_TAU_ELIGIBILITY`].
    ///
    /// ```
    /// use neuromod::RmStdpConfig;
    ///
    /// let slow = RmStdpConfig { tau_eligibility: 200.0, ..RmStdpConfig::default() };
    /// assert_eq!(slow.effective_tau_eligibility(), 200.0);
    ///
    /// for bad in [f32::NAN, f32::INFINITY, 0.0, -10.0] {
    ///     let config = RmStdpConfig { tau_eligibility: bad, ..RmStdpConfig::default() };
    ///     assert_eq!(config.effective_tau_eligibility(), 50.0);
    /// }
    /// ```
    pub fn effective_tau_eligibility(&self) -> f32 {
        if self.tau_eligibility.is_finite() && self.tau_eligibility > 0.0 {
            self.tau_eligibility
        } else {
            RM_STDP_TAU_ELIGIBILITY
        }
    }
}

impl EligibilityTrace {
    /// Create a zeroed trace with decay time constant `tau` (in steps).
    pub const fn new(tau: f32) -> Self {
        Self { value: 0.0, tau }
    }

    /// Spike-timing kernel for one pre/post coincidence, with
    /// `delta_t = t_post - t_pre` measured in steps.
    ///
    /// - `delta_t >= 0` (pre fired first, and so may have caused the post
    ///   spike): potentiation, `+A₊·exp(−Δt/τ₊)`.
    /// - `delta_t < 0` (post fired first, so pre cannot have caused it):
    ///   depression, `−A₋·exp(Δt/τ₋)`.
    ///
    /// Magnitude is largest at `delta_t == 0` and falls off exponentially as
    /// the two spikes drift apart.
    pub fn kernel(delta_t: f32) -> f32 {
        if delta_t >= 0.0 {
            RM_STDP_A_PLUS * (-delta_t / RM_STDP_TAU_PLUS).exp()
        } else {
            -RM_STDP_A_MINUS * (delta_t / RM_STDP_TAU_MINUS).exp()
        }
    }

    /// Record one pre/post coincidence, adding [`Self::kernel`] to the trace.
    ///
    /// Call this only on a step where a spike actually occurred. Re-applying it
    /// every step from a stale spike pair inflates the trace toward
    /// `tau × kernel` and turns one coincidence into sustained learning.
    pub fn accumulate(&mut self, delta_t: f32) {
        self.value += Self::kernel(delta_t);
    }

    /// Decay the trace by one step (assumes dt = 1 unit).
    ///
    /// A non-finite or non-positive `tau` falls back to
    /// [`RM_STDP_TAU_ELIGIBILITY`]. Clamping to `f32::EPSILON` instead would
    /// make `NaN` erase the trace outright (`exp(-1/ε) == 0`), and `+∞` would
    /// leave `exp(-0) == 1`, disabling decay — both silent corruptions of
    /// banked credit rather than honest degradation.
    pub fn decay(&mut self) {
        let tau = if self.tau.is_finite() && self.tau > 0.0 {
            self.tau
        } else {
            RM_STDP_TAU_ELIGIBILITY
        };
        self.value *= (-1.0 / tau).exp();
    }

    /// Clear the accumulated value, keeping [`Self::tau`].
    pub fn reset(&mut self) {
        self.value = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_at_zero_with_requested_tau() {
        let trace = EligibilityTrace::new(75.0);
        assert_eq!(trace.value, 0.0);
        assert_eq!(trace.tau, 75.0);
    }

    #[test]
    fn default_trace_uses_default_eligibility_tau() {
        let trace = EligibilityTrace::default();
        assert_eq!(trace.value, 0.0);
        assert_eq!(trace.tau, RM_STDP_TAU_ELIGIBILITY);
    }

    #[test]
    fn default_config_matches_published_constants() {
        let config = RmStdpConfig::default();
        assert_eq!(config.tau_eligibility, RM_STDP_TAU_ELIGIBILITY);
        assert_eq!(config.reward_lr, RM_STDP_REWARD_LR);
        assert_eq!(config.w_min, RM_STDP_W_MIN);
        assert_eq!(config.w_max, RM_STDP_W_MAX);
    }

    #[test]
    fn weight_bounds_pass_through_a_sane_configuration() {
        let config = RmStdpConfig {
            w_min: 0.25,
            w_max: 1.75,
            ..RmStdpConfig::default()
        };
        assert_eq!(config.weight_bounds(), (0.25, 1.75));
    }

    #[test]
    fn weight_bounds_fall_back_when_reversed_or_non_finite() {
        let fallback = (RM_STDP_W_MIN, RM_STDP_W_MAX);
        for (w_min, w_max) in [
            (1.5, 0.2),
            (f32::NAN, 1.0),
            (0.0, f32::NAN),
            (f32::NEG_INFINITY, f32::INFINITY),
        ] {
            let config = RmStdpConfig {
                w_min,
                w_max,
                ..RmStdpConfig::default()
            };
            let (lo, hi) = config.weight_bounds();
            assert_eq!((lo, hi), fallback);
            // The contract that matters: `clamp` must not panic on the result.
            assert!(0.5_f32.clamp(lo, hi).is_finite());
        }
    }

    #[test]
    fn effective_reward_lr_passes_through_finite_rates() {
        for rate in [0.0, 0.02, RM_STDP_REWARD_LR, -0.03] {
            let config = RmStdpConfig {
                reward_lr: rate,
                ..RmStdpConfig::default()
            };
            assert_eq!(config.effective_reward_lr(), rate);
        }
    }

    #[test]
    fn effective_reward_lr_falls_back_for_non_finite_rates() {
        for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let config = RmStdpConfig {
                reward_lr: rate,
                ..RmStdpConfig::default()
            };
            assert_eq!(config.effective_reward_lr(), RM_STDP_REWARD_LR);
        }
    }

    #[test]
    fn kernel_potentiates_when_pre_precedes_post() {
        let dw = EligibilityTrace::kernel(5.0);
        assert!(dw > 0.0);
        let expected = RM_STDP_A_PLUS * (-5.0_f32 / RM_STDP_TAU_PLUS).exp();
        assert!((dw - expected).abs() < 1e-9);
    }

    #[test]
    fn kernel_depresses_when_post_precedes_pre() {
        let dw = EligibilityTrace::kernel(-5.0);
        assert!(dw < 0.0);
        let expected = -RM_STDP_A_MINUS * (-5.0_f32 / RM_STDP_TAU_MINUS).exp();
        assert!((dw - expected).abs() < 1e-9);
    }

    #[test]
    fn kernel_is_strongest_at_coincident_spikes() {
        assert!((EligibilityTrace::kernel(0.0) - RM_STDP_A_PLUS).abs() < 1e-9);
        assert!(EligibilityTrace::kernel(1.0) < EligibilityTrace::kernel(0.0));
        assert!(EligibilityTrace::kernel(20.0) < EligibilityTrace::kernel(1.0));
        assert!(EligibilityTrace::kernel(-1.0) > EligibilityTrace::kernel(-0.5));
    }

    #[test]
    fn accumulate_adds_the_kernel_to_the_trace() {
        let mut trace = EligibilityTrace::new(50.0);
        trace.accumulate(3.0);
        assert!((trace.value - EligibilityTrace::kernel(3.0)).abs() < 1e-9);
    }

    #[test]
    fn accumulate_compounds_repeated_coincidences() {
        let mut trace = EligibilityTrace::new(50.0);
        trace.accumulate(0.0);
        let after_first = trace.value;
        trace.accumulate(0.0);
        assert!(trace.value > after_first);
        assert!((trace.value - 2.0 * RM_STDP_A_PLUS).abs() < 1e-9);
    }

    #[test]
    fn accumulate_can_flip_a_potentiated_trace_negative() {
        let mut trace = EligibilityTrace::new(50.0);
        trace.accumulate(0.0);
        assert!(trace.value > 0.0);
        // LTD amplitude exceeds LTP amplitude, so a coincident depression wins.
        trace.accumulate(-0.0001);
        assert!(trace.value < 0.0);
    }

    #[test]
    fn reset_clears_value_but_keeps_tau() {
        let mut trace = EligibilityTrace::new(60.0);
        trace.accumulate(0.0);
        trace.reset();
        assert_eq!(trace.value, 0.0);
        assert_eq!(trace.tau, 60.0);
    }

    #[test]
    fn decay_scales_value_by_exp_neg_inv_tau() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: 50.0,
        };
        let expected_factor = (-1.0_f32 / 50.0).exp();

        trace.decay();

        assert!((trace.value - expected_factor).abs() < 1e-6);
    }

    #[test]
    fn decay_applied_repeatedly_compounds_toward_zero() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: 50.0,
        };
        let factor = (-1.0_f32 / 50.0).exp();

        for _ in 0..5 {
            trace.decay();
        }

        let expected = factor.powi(5);
        assert!((trace.value - expected).abs() < 1e-5);
        assert!(trace.value < 1.0);
    }

    #[test]
    fn decay_preserves_sign_for_negative_values() {
        let mut trace = EligibilityTrace {
            value: -1.0,
            tau: 50.0,
        };
        trace.decay();
        assert!(trace.value < 0.0);
    }

    #[test]
    fn decay_with_non_finite_tau_falls_back_to_the_default() {
        let expected = (-1.0_f32 / RM_STDP_TAU_ELIGIBILITY).exp();
        for bad_tau in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -10.0] {
            let mut trace = EligibilityTrace {
                value: 1.0,
                tau: bad_tau,
            };
            trace.decay();
            assert!(
                (trace.value - expected).abs() < 1e-6,
                "tau {bad_tau} should decay at the default rate, got {}",
                trace.value
            );
            // Neither erased outright nor frozen in place.
            assert!(trace.value > 0.0 && trace.value < 1.0);
        }
    }

    #[test]
    fn decay_with_zero_tau_does_not_panic_or_diverge() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: 0.0,
        };
        trace.decay();
        assert!(trace.value.is_finite());
        assert!(trace.value >= 0.0);
    }

    #[test]
    fn decay_with_negative_tau_does_not_panic_or_diverge() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: -10.0,
        };
        trace.decay();
        assert!(trace.value.is_finite());
        assert!(trace.value >= 0.0);
    }
}
