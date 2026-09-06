//! # Neuromodulators and domain-agnostic reward hooks
//!
//! Four scalar modulators (dopamine, serotonin, acetylcholine, norepinephrine)
//! with exponential decay and helpers. [`NeuroModulators`] is the snapshot
//! passed into [`crate::SpikingNetwork::step`] each tick.
//!
//! [`SignalProfile`] maps unitless external signals into those levels.
//! [`GenericReward`] / [`UnitReward`] let downstream crates shape rewards without
//! hard-coding domain logic in this crate.
//!
//! [`apply_neuromodulation`] tweaks weight/threshold slices without owning a
//! full network — useful for tests and simple pipelines.
//!
//! ## Unit conventions
//!
//! This crate is **unit-agnostic on the input side and dimensionless on the
//! output side**:
//!
//! - **Outputs.** Every [`NeuroModulators`] field is a dimensionless level, and
//!   `0.0..=1.0` is the intended range — the only unit contract in the module.
//!   [`NeuroModulators::from_signals`] and [`NeuroModulators::decay`] keep levels
//!   inside it for finite inputs. The `add_*` / `boost_*` helpers clamp only the
//!   **upper** bound, so passing a negative amount drives a level below `0.0`;
//!   keeping those amounts non-negative is the caller's job.
//! - **Inputs.** The four signal channels accepted by
//!   [`NeuroModulators::from_signals`] (thermal, power, throughput, timing) carry
//!   **no unit of their own**. `neuromod` never assumes degrees, watts, hertz, or
//!   samples; the caller picks the unit and states it once, in the
//!   [`SignalProfile`].
//! - **Profiles.** Each [`SignalProfile`] field is expressed in the *same unit as
//!   the channel it pairs with*, so every ratio inside `from_signals` cancels to a
//!   dimensionless number. Mixing units within a channel (a threshold in °C
//!   against a signal in °F) is the caller's bug, not something this crate can
//!   detect.
//!
//! [`SignalProfile::default`] is the neutral profile for callers who already
//! normalize their signals into `0.0..=1.0`. See
//! [docs/signal-units.md](https://github.com/Limen-Neural/neuromod/blob/main/docs/signal-units.md)
//! for the full channel-by-channel table and worked calibration examples.

use serde::{Deserialize, Serialize};

const DOPAMINE_DECAY: f32 = 0.95;
const SEROTONIN_DECAY: f32 = 0.92;
const ACETYLCHOLINE_DECAY: f32 = 0.99;
const NOREPINEPHRINE_DECAY: f32 = 0.90;

/// Calibration for mapping external signals into neuromodulator levels.
///
/// A profile is the single place a caller declares what its signal channels
/// *mean*. `neuromod` treats the channels as bare `f32`s: the profile supplies
/// the reference values that turn them into dimensionless
/// [`NeuroModulators`] levels in `0.0..=1.0`.
///
/// # Units
///
/// Every field below is in the **same unit as the channel it scales**, so each
/// ratio in [`NeuroModulators::from_signals`] is dimensionless. There is no
/// implied SI unit anywhere in this struct.
///
/// | Field | Channel | Meaning |
/// |-------|---------|---------|
/// | [`throughput_scale`](Self::throughput_scale) | throughput | throughput that maps to dopamine `1.0` |
/// | [`stability_target`](Self::stability_target) | throughput | throughput considered perfectly stable (serotonin `1.0`) |
/// | [`thermal_threshold`](Self::thermal_threshold) | thermal | onset of thermal stress; `2 x threshold` saturates it |
/// | [`power_baseline`](Self::power_baseline) | power | power at which stress starts accumulating |
/// | [`power_scale`](Self::power_scale) | power | excess over baseline that saturates power stress |
/// | [`timing_scale`](Self::timing_scale) | timing | timing value that maps to acetylcholine `1.0` |
///
/// # Choosing a profile
///
/// - Signals already normalized to `0.0..=1.0` — use [`SignalProfile::default`].
/// - Signals in physical units — construct the struct directly with the
///   reference values of *your* domain. All fields are public; there is no
///   builder and no hidden state.
///
/// ```rust
/// use neuromod::{NeuroModulators, SignalProfile};
///
/// // Thermal in °C, power in W, throughput in items/s, timing in samples.
/// let profile = SignalProfile {
///     throughput_scale: 500.0,
///     thermal_threshold: 80.0,
///     power_baseline: 120.0,
///     power_scale: 40.0,
///     timing_scale: 1024.0,
///     stability_target: 400.0,
/// };
///
/// let mods = NeuroModulators::from_signals(&profile, 88.0, 130.0, 250.0, 512.0);
/// assert!((0.0..=1.0).contains(&mods.dopamine));
/// assert!((0.0..=1.0).contains(&mods.norepinephrine));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SignalProfile {
    /// Throughput value that maps to dopamine `1.0`, in throughput units
    /// (default: `1.0`). Larger values make a given throughput less rewarding.
    pub throughput_scale: f32,
    /// Thermal value above which thermal stress starts accumulating, in thermal
    /// units (default: `0.5`). Stress saturates at twice this value.
    pub thermal_threshold: f32,
    /// Power value at which power stress starts accumulating, in power units
    /// (default: `0.0`).
    pub power_baseline: f32,
    /// Excess over [`power_baseline`](Self::power_baseline) that saturates power
    /// stress, in power units (default: `1.0`).
    pub power_scale: f32,
    /// Timing value that maps to acetylcholine `1.0`, in timing units
    /// (default: `1.0`).
    pub timing_scale: f32,
    /// Throughput considered perfectly stable, in throughput units
    /// (default: `1.0`). Serotonin falls off linearly with the **raw**
    /// deviation from this target — see [`NeuroModulators::from_signals`].
    pub stability_target: f32,
}

impl Default for SignalProfile {
    /// Neutral, unitless profile for callers whose signals are already
    /// normalized to `0.0..=1.0`.
    ///
    /// Every scale is `1.0`, so each channel passes through unchanged (modulo
    /// clamping); `thermal_threshold` is `0.5`, so the upper half of a
    /// normalized thermal channel maps onto the full stress range.
    fn default() -> Self {
        Self {
            throughput_scale: 1.0,
            thermal_threshold: 0.5,
            power_baseline: 0.0,
            power_scale: 1.0,
            timing_scale: 1.0,
            stability_target: 1.0,
        }
    }
}

impl SignalProfile {
    /// Legacy hardware-calibrated profile kept for pre-0.5 callers.
    ///
    /// # Deprecated
    ///
    /// Calibration constants describe a *deployment*, not neuron dynamics, so
    /// they belong to the consuming crate. This constructor is a domain fossil:
    /// its numbers only mean anything for one historical device (thermal in °C,
    /// power in W, timing in samples), and `neuromod` cannot check that a caller
    /// feeds it those units.
    ///
    /// Nothing is removed in 0.6 — this still returns exactly the values it
    /// always has. To migrate, copy the literal into your own crate:
    ///
    /// ```rust
    /// use neuromod::SignalProfile;
    ///
    /// let profile = SignalProfile {
    ///     throughput_scale: 0.0105,
    ///     thermal_threshold: 83.0,
    ///     power_baseline: 400.0,
    ///     power_scale: 50.0,
    ///     timing_scale: 2640.0,
    ///     stability_target: 1.05,
    /// };
    /// # let _ = profile;
    /// ```
    ///
    /// Note the wart this profile makes visible: `throughput_scale` (`0.0105`)
    /// and `stability_target` (`1.05`) disagree by two orders of magnitude, so
    /// the two throughput-derived channels are never informative at the same
    /// time. Dopamine saturates at any throughput at or above `0.0105`, while
    /// serotonin is non-zero only within `0.5` of `1.05` (roughly `0.55..1.55`).
    /// Across the range where dopamine still varies, serotonin is pinned at
    /// `0.0`; across the range where serotonin varies, dopamine is already
    /// pinned at `1.0`. New profiles should keep the two within the same range.
    #[deprecated(
        since = "0.6.0",
        note = "deployment calibration belongs to the consuming crate; construct `SignalProfile { .. }` directly (the legacy literal is in this method's docs)"
    )]
    pub fn hardware_calibrated() -> Self {
        Self {
            throughput_scale: 0.0105,
            thermal_threshold: 83.0,
            power_baseline: 400.0,
            power_scale: 50.0,
            timing_scale: 2640.0,
            stability_target: 1.05,
        }
    }
}

/// Domain-agnostic observation bag for reward computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub signals: Vec<f32>,
}

impl Observation {
    pub fn from_slice(signals: &[f32]) -> Self {
        Self {
            signals: signals.to_vec(),
        }
    }
}

/// Generic reward interface for domain-specific implementations in downstream crates.
pub trait GenericReward {
    fn compute_reward(&self, observation: &Observation) -> f32;
}

/// Mean-signal reward for tests and simple pipelines.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnitReward;

impl GenericReward for UnitReward {
    fn compute_reward(&self, observation: &Observation) -> f32 {
        if observation.signals.is_empty() {
            0.0
        } else {
            observation.signals.iter().sum::<f32>() / observation.signals.len() as f32
        }
    }
}

/// Neuromodulator system for reward-modulated learning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NeuroModulators {
    pub dopamine: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub norepinephrine: f32,
}

impl Default for NeuroModulators {
    fn default() -> Self {
        Self {
            dopamine: 0.0,
            serotonin: 0.0,
            acetylcholine: 0.0,
            norepinephrine: 0.0,
        }
    }
}

impl NeuroModulators {
    /// Map four external signal channels into modulator levels using `profile`.
    ///
    /// # Units
    ///
    /// The four signals are **unitless as far as this crate is concerned**: each
    /// one only has meaning relative to the matching [`SignalProfile`] field,
    /// which must be expressed in the same unit. For finite inputs every result
    /// is dimensionless and clamped to `0.0..=1.0`; see *Edge cases* for what
    /// `NaN` does.
    ///
    /// | Signal | Paired profile field(s) | Drives |
    /// |--------|-------------------------|--------|
    /// | `thermal_signal` | [`thermal_threshold`](SignalProfile::thermal_threshold) | norepinephrine (with power) |
    /// | `power_signal` | [`power_baseline`](SignalProfile::power_baseline), [`power_scale`](SignalProfile::power_scale) | norepinephrine (with thermal) |
    /// | `throughput_signal` | [`throughput_scale`](SignalProfile::throughput_scale), [`stability_target`](SignalProfile::stability_target) | dopamine, serotonin |
    /// | `timing_signal` | [`timing_scale`](SignalProfile::timing_scale) | acetylcholine |
    ///
    /// # Mapping
    ///
    /// ```text
    /// dopamine       = clamp(throughput / throughput_scale)
    /// acetylcholine  = clamp(timing / timing_scale)
    /// serotonin      = clamp(1 - 2 * |throughput - stability_target|)
    /// norepinephrine = max(thermal_stress, power_stress)
    ///   thermal_stress = clamp((thermal - thermal_threshold) / thermal_threshold)   [0 below threshold]
    ///   power_stress   = clamp((power - power_baseline) / power_scale)
    /// ```
    ///
    /// where `clamp(x)` is `x.clamp(0.0, 1.0)`, so negative signals read as `0.0`
    /// and over-range signals saturate rather than wrap.
    ///
    /// # Edge cases
    ///
    /// - A divisor within [`f32::EPSILON`] of zero yields `0.0` for that term
    ///   instead of an infinity or `NaN`. A `NaN` profile field takes the same
    ///   path, because the guard compares `abs() > f32::EPSILON`.
    /// - **`NaN` signals are not sanitized.** `f32::clamp` returns `NaN` for a
    ///   `NaN` input, so a `NaN` throughput reaches `dopamine` and `serotonin`,
    ///   and a `NaN` timing reaches `acetylcholine`. `norepinephrine` is the
    ///   exception and reads `0.0`: the thermal comparison is false for `NaN`,
    ///   and `f32::max` discards a `NaN` operand. Validate signals upstream if
    ///   they can be `NaN`.
    /// - `thermal_signal` at or below `thermal_threshold` contributes no stress;
    ///   stress saturates at `2 * thermal_threshold`.
    /// - **Serotonin is the one channel that is not scale-normalized.** The
    ///   deviation from `stability_target` is measured in raw throughput units
    ///   with a fixed half-width of `0.5`, so a profile whose throughput unit is
    ///   much larger or smaller than `1.0` pins serotonin to `0.0`. Keep
    ///   `stability_target` and `throughput_scale` in the same range, or feed a
    ///   pre-normalized throughput channel. This is preserved behavior, not a
    ///   recommendation.
    ///
    /// ```rust
    /// use neuromod::{NeuroModulators, SignalProfile};
    ///
    /// // Normalized signals in 0.0..=1.0 against the neutral profile.
    /// let mods = NeuroModulators::from_signals(&SignalProfile::default(), 0.75, 0.4, 1.0, 0.6);
    /// assert_eq!(mods.dopamine, 1.0); // throughput / 1.0, saturated
    /// assert_eq!(mods.serotonin, 1.0); // exactly on the stability target
    /// assert!((mods.acetylcholine - 0.6).abs() < 1e-6);
    /// assert!((mods.norepinephrine - 0.5).abs() < 1e-6); // thermal 0.75 vs threshold 0.5
    /// ```
    pub fn from_signals(
        profile: &SignalProfile,
        thermal_signal: f32,
        power_signal: f32,
        throughput_signal: f32,
        timing_signal: f32,
    ) -> Self {
        let safe_div = |num: f32, den: f32| -> f32 {
            if den.abs() > f32::EPSILON {
                num / den
            } else {
                0.0
            }
        };

        let dopamine = safe_div(throughput_signal, profile.throughput_scale).clamp(0.0, 1.0);

        let thermal_stress = if thermal_signal > profile.thermal_threshold {
            safe_div(
                thermal_signal - profile.thermal_threshold,
                profile.thermal_threshold,
            )
            .clamp(0.0, 1.0)
        } else {
            0.0
        };
        let power_stress =
            safe_div(power_signal - profile.power_baseline, profile.power_scale).clamp(0.0, 1.0);
        let norepinephrine = thermal_stress.max(power_stress);

        let stability_dev = (throughput_signal - profile.stability_target).abs();
        let serotonin = (1.0 - stability_dev * 2.0).clamp(0.0, 1.0);

        let acetylcholine = safe_div(timing_signal, profile.timing_scale).clamp(0.0, 1.0);

        Self {
            dopamine,
            serotonin,
            acetylcholine,
            norepinephrine,
        }
    }

    /// Apply natural decay (homeostasis).
    pub fn decay(&mut self) {
        self.dopamine = (self.dopamine * DOPAMINE_DECAY).max(0.0);
        self.serotonin = (self.serotonin * SEROTONIN_DECAY).max(0.0);
        self.acetylcholine = (self.acetylcholine * ACETYLCHOLINE_DECAY).max(0.0);
        self.norepinephrine = (self.norepinephrine * NOREPINEPHRINE_DECAY).max(0.0);
    }

    /// Add dopamine reward.
    pub fn add_reward(&mut self, amount: f32) {
        self.dopamine = (self.dopamine + amount).min(1.0);
    }

    /// Add serotonin (mood/stability).
    pub fn add_serotonin(&mut self, amount: f32) {
        self.serotonin = (self.serotonin + amount).min(1.0);
    }

    /// Boost acetylcholine for focus.
    pub fn boost_focus(&mut self, amount: f32) {
        self.acetylcholine = (self.acetylcholine + amount).min(1.0);
    }

    /// Add norepinephrine (arousal/stress).
    pub fn add_norepinephrine(&mut self, amount: f32) {
        self.norepinephrine = (self.norepinephrine + amount).min(1.0);
    }

    /// Apply reward from a generic reward source.
    pub fn apply_reward<R: GenericReward>(&mut self, reward: &R, observation: &Observation) {
        self.add_reward(reward.compute_reward(observation));
    }

    /// Check if system is under high arousal/stress.
    pub fn is_aroused(&self) -> bool {
        self.norepinephrine > 0.7
    }

    /// Check if system is in reward state.
    pub fn is_rewarded(&self) -> bool {
        self.dopamine >= 0.5
    }

    /// Check if system is focused.
    pub fn is_focused(&self) -> bool {
        self.acetylcholine > 0.6
    }

    /// Check if system is in a calm/stable state.
    pub fn is_calm(&self) -> bool {
        self.serotonin > 0.6
    }
}

/// Apply neuromodulator effects to synaptic weights and firing thresholds.
pub fn apply_neuromodulation(
    modulators: &NeuroModulators,
    weights: &mut [f32],
    thresholds: &mut [f32],
) {
    let learning_rate = 0.5 * modulators.dopamine;
    let stress_multiplier = (1.0 - modulators.norepinephrine).max(0.1);
    let focus_scale = 1.0 + 0.05 * modulators.acetylcholine;

    for w in weights.iter_mut() {
        *w *= stress_multiplier * focus_scale;
    }

    let global_target = 0.20 - (0.05 * modulators.dopamine) + (0.15 * modulators.norepinephrine)
        - (0.05 * modulators.serotonin);

    for t in thresholds.iter_mut() {
        *t += (global_target - *t) * learning_rate;
        *t = t.clamp(0.05, 0.50);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulators_default() {
        let mods = NeuroModulators::default();
        assert_eq!(mods.dopamine, 0.0);
        assert_eq!(mods.serotonin, 0.0);
        assert_eq!(mods.acetylcholine, 0.0);
        assert_eq!(mods.norepinephrine, 0.0);
    }

    #[test]
    fn test_from_signals_default_profile() {
        let profile = SignalProfile::default();
        let mods = NeuroModulators::from_signals(&profile, 0.2, 0.1, 0.8, 0.9);
        assert!(mods.dopamine > 0.0);
        assert!(mods.acetylcholine > 0.0);
        assert!(mods.serotonin >= 0.0);
    }

    /// The legacy literal documented on `hardware_calibrated` as the migration
    /// target. Kept next to the deprecated constructor so the two cannot drift.
    const LEGACY_HARDWARE_PROFILE: SignalProfile = SignalProfile {
        throughput_scale: 0.0105,
        thermal_threshold: 83.0,
        power_baseline: 400.0,
        power_scale: 50.0,
        timing_scale: 2640.0,
        stability_target: 1.05,
    };

    #[test]
    #[allow(deprecated)]
    fn test_hardware_calibrated_matches_documented_migration_literal() {
        assert_eq!(
            SignalProfile::hardware_calibrated(),
            LEGACY_HARDWARE_PROFILE
        );
    }

    #[test]
    fn test_from_signals_legacy_hardware_profile() {
        let mods =
            NeuroModulators::from_signals(&LEGACY_HARDWARE_PROFILE, 75.0, 300.0, 0.05, 2640.0);
        assert!(mods.dopamine > 0.0);
        assert!(mods.acetylcholine > 0.0);
    }

    #[test]
    fn test_from_signals_outputs_are_dimensionless_and_clamped() {
        let profile = SignalProfile::default();
        // Wildly out-of-range and negative inputs must still land in 0.0..=1.0.
        for (thermal, power, throughput, timing) in [(1e6, 1e6, 1e6, 1e6), (-1e6, -1e6, -1e6, -1e6)]
        {
            let mods = NeuroModulators::from_signals(&profile, thermal, power, throughput, timing);
            for level in [
                mods.dopamine,
                mods.serotonin,
                mods.acetylcholine,
                mods.norepinephrine,
            ] {
                assert!((0.0..=1.0).contains(&level), "level out of range: {level}");
            }
        }
    }

    #[test]
    fn test_from_signals_zero_scales_do_not_produce_nan() {
        let profile = SignalProfile {
            throughput_scale: 0.0,
            thermal_threshold: 0.0,
            power_baseline: 0.0,
            power_scale: 0.0,
            timing_scale: 0.0,
            stability_target: 0.0,
        };
        let mods = NeuroModulators::from_signals(&profile, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(mods.dopamine, 0.0);
        assert_eq!(mods.acetylcholine, 0.0);
        assert_eq!(mods.norepinephrine, 0.0);
        assert!(mods.serotonin.is_finite());
    }

    #[test]
    fn test_add_helpers_clamp_only_the_upper_bound() {
        // Documented caveat: `add_*` / `boost_*` cap at 1.0 but not at 0.0.
        let mut mods = NeuroModulators::default();
        mods.add_reward(-0.5);
        mods.add_serotonin(-0.5);
        mods.boost_focus(-0.5);
        mods.add_norepinephrine(-0.5);
        assert_eq!(mods.dopamine, -0.5);
        assert_eq!(mods.serotonin, -0.5);
        assert_eq!(mods.acetylcholine, -0.5);
        assert_eq!(mods.norepinephrine, -0.5);

        // `decay` does clamp low, so a negative level recovers to 0.0.
        mods.decay();
        assert_eq!(mods.dopamine, 0.0);
        assert_eq!(mods.serotonin, 0.0);
        assert_eq!(mods.acetylcholine, 0.0);
        assert_eq!(mods.norepinephrine, 0.0);
    }

    #[test]
    fn test_from_signals_propagates_nan_signals_except_norepinephrine() {
        // Documented caveat: `f32::clamp` passes `NaN` through unchanged.
        let profile = SignalProfile::default();
        let nan = f32::NAN;

        let throughput = NeuroModulators::from_signals(&profile, 0.0, 0.0, nan, 0.0);
        assert!(throughput.dopamine.is_nan());
        assert!(throughput.serotonin.is_nan());

        let timing = NeuroModulators::from_signals(&profile, 0.0, 0.0, 0.0, nan);
        assert!(timing.acetylcholine.is_nan());

        // Norepinephrine is the exception: the thermal comparison is false for
        // `NaN`, and `f32::max` discards a `NaN` operand.
        assert_eq!(
            NeuroModulators::from_signals(&profile, nan, 0.0, 0.0, 0.0).norepinephrine,
            0.0
        );
        assert_eq!(
            NeuroModulators::from_signals(&profile, 0.0, nan, 0.0, 0.0).norepinephrine,
            0.0
        );
    }

    #[test]
    fn test_from_signals_nan_profile_field_hits_the_divisor_guard() {
        let profile = SignalProfile {
            throughput_scale: f32::NAN,
            ..SignalProfile::default()
        };
        assert_eq!(
            NeuroModulators::from_signals(&profile, 0.0, 0.0, 1.0, 0.0).dopamine,
            0.0
        );
    }

    #[test]
    fn test_legacy_profile_dopamine_and_serotonin_bands_do_not_overlap() {
        // Guards the documented table: dopamine saturates from 0.0105 upward,
        // while serotonin is non-zero only within 0.5 of the 1.05 target.
        let read = |throughput| {
            NeuroModulators::from_signals(&LEGACY_HARDWARE_PROFILE, 0.0, 0.0, throughput, 0.0)
        };

        // Where dopamine still varies, serotonin is pinned at 0.0.
        let low = read(0.005);
        assert!(low.dopamine > 0.0 && low.dopamine < 1.0);
        assert_eq!(low.serotonin, 0.0);

        // Where serotonin varies, dopamine is already saturated.
        let on_target = read(1.05);
        assert_eq!(on_target.dopamine, 1.0);
        assert_eq!(on_target.serotonin, 1.0);

        let near = read(1.0);
        assert_eq!(near.dopamine, 1.0);
        assert!((near.serotonin - 0.9).abs() < 1e-6);

        // Outside the serotonin band, dopamine stays saturated.
        let far = read(1.55);
        assert_eq!(far.dopamine, 1.0);
        assert_eq!(far.serotonin, 0.0);
    }

    #[test]
    fn test_from_signals_thermal_threshold_semantics() {
        let profile = SignalProfile {
            thermal_threshold: 80.0,
            ..SignalProfile::default()
        };
        // At or below the threshold: no thermal stress.
        let at = NeuroModulators::from_signals(&profile, 80.0, 0.0, 0.0, 0.0);
        assert_eq!(at.norepinephrine, 0.0);
        // Halfway to saturation (threshold + 0.5 * threshold).
        let mid = NeuroModulators::from_signals(&profile, 120.0, 0.0, 0.0, 0.0);
        assert!((mid.norepinephrine - 0.5).abs() < 1e-6);
        // Twice the threshold saturates.
        let hot = NeuroModulators::from_signals(&profile, 160.0, 0.0, 0.0, 0.0);
        assert_eq!(hot.norepinephrine, 1.0);
    }

    #[test]
    fn test_decay() {
        let mut mods = NeuroModulators {
            dopamine: 1.0,
            serotonin: 1.0,
            acetylcholine: 1.0,
            norepinephrine: 1.0,
        };

        mods.decay();

        assert!(mods.dopamine < 1.0);
        assert!(mods.serotonin < 1.0);
        assert!(mods.acetylcholine < 1.0);
        assert!(mods.norepinephrine < 1.0);
    }

    #[test]
    fn test_reward_and_arousal() {
        let mut mods = NeuroModulators::default();

        mods.add_reward(0.5);
        assert_eq!(mods.dopamine, 0.5);
        assert!(mods.is_rewarded());

        mods.add_norepinephrine(0.8);
        assert_eq!(mods.norepinephrine, 0.8);
        assert!(mods.is_aroused());

        mods.boost_focus(0.7);
        assert_eq!(mods.acetylcholine, 0.7);
        assert!(mods.is_focused());

        mods.add_serotonin(0.7);
        assert_eq!(mods.serotonin, 0.7);
        assert!(mods.is_calm());
    }

    #[test]
    fn test_clamping() {
        let mut mods = NeuroModulators::default();

        mods.add_reward(2.0);
        assert_eq!(mods.dopamine, 1.0);

        mods.add_norepinephrine(2.0);
        assert_eq!(mods.norepinephrine, 1.0);

        mods.boost_focus(2.0);
        assert_eq!(mods.acetylcholine, 1.0);

        mods.add_serotonin(2.0);
        assert_eq!(mods.serotonin, 1.0);
    }

    #[test]
    fn test_unit_reward() {
        let reward = UnitReward;
        let obs = Observation::from_slice(&[0.2, 0.8]);
        let mut mods = NeuroModulators::default();
        mods.apply_reward(&reward, &obs);
        assert!((mods.dopamine - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_neuromodulation() {
        let mods = NeuroModulators {
            dopamine: 0.8,
            serotonin: 0.2,
            acetylcholine: 0.5,
            norepinephrine: 0.3,
        };
        let mut weights = vec![1.0, 1.0];
        let mut thresholds = vec![0.20, 0.20];
        apply_neuromodulation(&mods, &mut weights, &mut thresholds);
        assert_ne!(weights[0], 1.0);
        assert!(thresholds[0] >= 0.05 && thresholds[0] <= 0.50);
    }
}
