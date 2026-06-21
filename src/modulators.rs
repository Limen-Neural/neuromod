use serde::{Deserialize, Serialize};

const DOPAMINE_DECAY: f32 = 0.95;
const SEROTONIN_DECAY: f32 = 0.92;
const ACETYLCHOLINE_DECAY: f32 = 0.99;
const NOREPINEPHRINE_DECAY: f32 = 0.90;

/// Configuration for mapping external signals into neuromodulator levels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SignalProfile {
    /// Divisor for normalizing throughput into dopamine (default: 1.0).
    pub throughput_scale: f32,
    /// Unitless threshold above which thermal input contributes stress.
    pub thermal_threshold: f32,
    /// Baseline power/load signal before stress accumulation.
    pub power_baseline: f32,
    /// Divisor for power signal stress scaling.
    pub power_scale: f32,
    /// Divisor for normalizing timing input into acetylcholine.
    pub timing_scale: f32,
    /// Target throughput level for stability (serotonin) computation.
    pub stability_target: f32,
}

impl Default for SignalProfile {
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
    /// Legacy hardware-calibrated profile for callers migrating from pre-0.5 APIs.
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
    /// Create neuromodulators from generic external signals using `profile` scaling.
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

    #[test]
    fn test_from_signals_hardware_calibrated() {
        let profile = SignalProfile::hardware_calibrated();
        let mods = NeuroModulators::from_signals(&profile, 75.0, 300.0, 0.05, 2640.0);
        assert!(mods.dopamine > 0.0);
        assert!(mods.acetylcholine > 0.0);
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
        assert!(weights[0] != 1.0);
        assert!(thresholds[0] >= 0.05 && thresholds[0] <= 0.50);
    }
}
