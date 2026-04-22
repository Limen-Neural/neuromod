
use serde::{Deserialize, Serialize};

// Decay constants (per step)
const DOPAMINE_DECAY: f32 = 0.95;
const CORTISOL_DECAY: f32 = 0.90;
const ACETYLCHOLINE_DECAY: f32 = 0.99;

/// Neuromodulator system for reward-modulated learning
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NeuroModulators {
    pub dopamine: f32,
    pub cortisol: f32,
    pub acetylcholine: f32,
    pub tempo: f32,
    /// Auxiliary dopamine-like channel for external reward shaping.
    pub aux_dopamine: f32,
}

impl Default for NeuroModulators {
    fn default() -> Self {
        Self {
            dopamine: 0.0,
            cortisol: 0.0,
            acetylcholine: 0.0,
            tempo: 0.0,
            aux_dopamine: 0.0,
        }
    }
}

impl NeuroModulators {
    /// Create neuromodulators from generic external signals.
    /// 
    /// # Arguments
    /// * `thermal_signal` - Thermal stress proxy
    /// * `power_signal` - Power/load stress proxy
    /// * `throughput_signal` - Throughput-like signal (generic performance proxy)
    /// * `timing_signal` - Timing/frequency proxy
    pub fn from_signals(
        thermal_signal: f32,
        power_signal: f32,
        throughput_signal: f32,
        timing_signal: f32,
    ) -> Self {
        // DOPAMINE: Proportional to throughput-like signal.
        let dopamine = (throughput_signal / 0.0105).clamp(0.3, 1.0);

        // CORTISOL: Stress from thermal or load spikes.
        let thermal_stress: f32 = if thermal_signal > 1.0 {
            ((thermal_signal - 83.0) / 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let power_stress = ((power_signal - 400.0) / 50.0).clamp(0.0, 1.0);
        let cortisol = thermal_stress.max(power_stress).max(0.0);

        // ACETYLCHOLINE: Signal stability proxy (focus-like state).
        let stability_dev = (1.05_f32 - 1.0_f32).abs();
        let acetylcholine = (1.0_f32 - stability_dev * 2.0_f32).clamp(0.4_f32, 1.0_f32);

        // TEMPO: Timing-driven temporal scaling.
        let tempo = (timing_signal / 2640.0).clamp(0.5, 2.0);

        Self {
            dopamine,
            cortisol,
            acetylcholine,
            tempo,
            aux_dopamine: 0.0,
        }
    }

    /// Apply natural decay (homeostasis)
    pub fn decay(&mut self) {
        self.dopamine = (self.dopamine * DOPAMINE_DECAY).max(0.0);
        self.cortisol = (self.cortisol * CORTISOL_DECAY).max(0.0);
        self.acetylcholine = (self.acetylcholine * ACETYLCHOLINE_DECAY).max(0.0);
        self.aux_dopamine = (self.aux_dopamine * DOPAMINE_DECAY).max(0.0);
    }

    /// Add dopamine reward
    pub fn add_reward(&mut self, amount: f32) {
        self.dopamine = (self.dopamine + amount).min(1.0);
    }

    /// Add cortisol stress
    pub fn add_stress(&mut self, amount: f32) {
        self.cortisol = (self.cortisol + amount).min(1.0);
    }

    /// Boost acetylcholine for focus
    pub fn boost_focus(&mut self, amount: f32) {
        self.acetylcholine = (self.acetylcholine + amount).min(1.0);
    }

    /// Set tempo directly
    pub fn set_tempo(&mut self, tempo: f32) {
        self.tempo = tempo.clamp(0.0, 2.0);
    }

    /// Add reward to the auxiliary dopamine channel.
    pub fn add_aux_reward(&mut self, amount: f32) {
        self.aux_dopamine = (self.aux_dopamine + amount).min(1.0);
    }

    /// Check if system is under high stress
    pub fn is_stressed(&self) -> bool {
        self.cortisol > 0.7
    }

    /// Check if system is in reward state
    pub fn is_rewarded(&self) -> bool {
        self.dopamine >= 0.5
    }

    /// Check if system is focused
    pub fn is_focused(&self) -> bool {
        self.acetylcholine > 0.6
    }

    /// Check if the auxiliary reward channel is active.
    pub fn is_aux_rewarded(&self) -> bool {
        self.aux_dopamine > 0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulators_default() {
        let mods = NeuroModulators::default();
        assert_eq!(mods.dopamine, 0.0);
        assert_eq!(mods.cortisol, 0.0);
        assert_eq!(mods.acetylcholine, 0.0);
        assert_eq!(mods.tempo, 0.0);
    }

    #[test]
    fn test_from_signals() {
        let mods = NeuroModulators::from_signals(75.0, 300.0, 0.05, 2640.0);
        assert!(mods.dopamine > 0.0);
        assert!(mods.acetylcholine > 0.0);
        assert_eq!(mods.tempo, 1.0);
    }

    #[test]
    fn test_decay() {
        let mut mods = NeuroModulators::default();
        mods.dopamine = 1.0;
        mods.cortisol = 1.0;
        mods.acetylcholine = 1.0;
        
        mods.decay();
        
        assert!(mods.dopamine < 1.0);
        assert!(mods.cortisol < 1.0);
        assert!(mods.acetylcholine < 1.0);
    }

    #[test]
    fn test_reward_and_stress() {
        let mut mods = NeuroModulators::default();
        
        mods.add_reward(0.5);
        assert_eq!(mods.dopamine, 0.5);
        assert!(mods.is_rewarded());
        
        mods.add_stress(0.8);
        assert_eq!(mods.cortisol, 0.8);
        assert!(mods.is_stressed());
        
        mods.boost_focus(0.7);
        assert_eq!(mods.acetylcholine, 0.7);
        assert!(mods.is_focused());
    }

    #[test]
    fn test_clamping() {
        let mut mods = NeuroModulators::default();
        
        // Test overflow clamping
        mods.add_reward(2.0);
        assert_eq!(mods.dopamine, 1.0);
        
        mods.add_stress(2.0);
        assert_eq!(mods.cortisol, 1.0);
        
        mods.boost_focus(2.0);
        assert_eq!(mods.acetylcholine, 1.0);
        
        // Test tempo clamping
        mods.set_tempo(3.0);
        assert_eq!(mods.tempo, 2.0);
        
        mods.set_tempo(-1.0);
        assert_eq!(mods.tempo, 0.0);
    }
}
