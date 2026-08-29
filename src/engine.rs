//! # Engine — LIF + Izhikevich `SpikingNetwork`
//!
//! Topology-neutral simulation core. One network owns:
//!
//! - a **LIF** bank (`neurons`) driven by multi-channel stimuli and STDP,
//! - an **Izhikevich** bank (`iz_neurons`) driven from mean LIF membrane potential
//!   + dopamine (not a second full STDP pipeline),
//! - a [`NeuroModulators`] snapshot updated each step.
//!
//! Construction is blank weights / no domain topology:
//! [`SpikingNetwork::new`] (16 / 5 / 16) or [`SpikingNetwork::with_dimensions`].
//!
//! For classical Hebbian STDP on a small Izhikevich network, see
//! [`crate::hebbian`] — that path is separate from this engine.
//!
//! Plasticity: live reward-modulated updates run inside `step` via `apply_stdp`.
//! Every step decays and accumulates one [`crate::EligibilityTrace`] per synapse;
//! dopamine gates only the trace → weight conversion, so a coincidence recorded
//! while reward was absent can still be paid out later (see [`crate::rm_stdp`]
//! and [`RmStdpConfig`]).

use rand::RngExt;
use serde::{Deserialize, Serialize};

use super::izhikevich::IzhikevichNeuron;
use super::lif::LifNeuron;
use super::modulators::NeuroModulators;
use super::rm_stdp::*;

/// L1 synaptic weight budget per neuron (total weight sum target).
const WEIGHT_BUDGET: f32 = 2.0;

/// Errors from [`SpikingNetwork::step`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    /// `stimuli.len()` did not match the network's `num_channels`.
    InputLenMismatch { expected: usize, got: usize },
}

/// Topology-neutral network: LIF bank + Izhikevich bank + neuromodulators.
///
/// Only these two neuron types are wired here. Other models in the crate are
/// standalone (see crate root docs).
#[derive(Serialize, Deserialize)]
pub struct SpikingNetwork {
    /// Bank 1: LIF neurons.
    pub neurons: Vec<LifNeuron>,
    /// Bank 2: Izhikevich neurons.
    pub iz_neurons: Vec<IzhikevichNeuron>,
    /// Global neuromodulators.
    pub modulators: NeuroModulators,
    /// Global step counter for STDP timing.
    pub global_step: i64,
    /// Number of input channels expected by `step`.
    pub num_channels: usize,
    /// Pre-synaptic spike times for each input channel.
    pub input_spike_times: Vec<i64>,
    /// Per-channel exponential moving average of input stimuli.
    pub predictive_state: Vec<f32>,
    /// R-STDP hyperparameters: eligibility-trace decay, the reward learning
    /// rate used to convert traces into weight changes, and the weight bounds
    /// enforced by `apply_stdp` and the L1 renormalization pass.
    ///
    /// Assigning this field directly leaves existing traces on their previous
    /// `tau`; use [`SpikingNetwork::set_rm_stdp_config`] to update both.
    #[serde(default)]
    pub stdp_config: RmStdpConfig,
}

impl SpikingNetwork {
    /// Create the default network (16 LIF, 5 Izhikevich, 16 channels).
    pub fn new() -> Self {
        Self::with_dimensions(16, 5, crate::NUM_INPUT_CHANNELS)
    }

    /// Create a dynamically sized network.
    pub fn with_dimensions(num_lif: usize, num_izh: usize, num_channels: usize) -> Self {
        let stdp_config = RmStdpConfig::default();
        let neurons: Vec<LifNeuron> = (0..num_lif)
            .map(|_| {
                let mut n = LifNeuron::new();
                n.weights = vec![0.0; num_channels];
                n.eligibility = vec![
                    EligibilityTrace::new(stdp_config.effective_tau_eligibility());
                    num_channels
                ];
                n.last_spike_time = -1;
                n
            })
            .collect();

        Self {
            neurons,
            iz_neurons: vec![IzhikevichNeuron::new_regular_spiking(); num_izh],
            modulators: NeuroModulators::default(),
            global_step: 0,
            num_channels,
            input_spike_times: vec![-1; num_channels],
            predictive_state: vec![0.0; num_channels],
            stdp_config,
        }
    }

    /// Replace the R-STDP hyperparameters, re-`tau`-ing every existing
    /// eligibility trace so traces and config stay consistent.
    ///
    /// The config is **normalized on the way in**: each field passes through its
    /// guard ([`RmStdpConfig::weight_bounds`],
    /// [`RmStdpConfig::effective_reward_lr`],
    /// [`RmStdpConfig::effective_tau_eligibility`]), so a reversed or non-finite
    /// value is replaced by the published default rather than stored and worked
    /// around later. Assigning [`Self::stdp_config`] directly bypasses this; the
    /// engine still reads through the same guards, so it stays safe either way.
    ///
    /// Accumulated trace *values* are preserved — only the decay time constant
    /// changes. Use [`Self::reset`] to clear them.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromod::{RmStdpConfig, SpikingNetwork};
    ///
    /// let mut net = SpikingNetwork::with_dimensions(4, 1, 4);
    /// net.set_rm_stdp_config(RmStdpConfig {
    ///     tau_eligibility: 100.0,
    ///     reward_lr: 0.02,
    ///     ..RmStdpConfig::default()
    /// });
    ///
    /// assert_eq!(net.neurons[0].eligibility[0].tau, 100.0);
    /// ```
    pub fn set_rm_stdp_config(&mut self, config: RmStdpConfig) {
        let (w_min, w_max) = config.weight_bounds();
        let tau = config.effective_tau_eligibility();
        self.stdp_config = RmStdpConfig {
            tau_eligibility: tau,
            reward_lr: config.effective_reward_lr(),
            w_min,
            w_max,
        };

        for neuron in &mut self.neurons {
            for trace in &mut neuron.eligibility {
                trace.tau = tau;
            }
        }
    }

    /// Advance the network by one discrete time step.
    ///
    /// # Contract
    ///
    /// - `stimuli.len()` must equal [`Self::num_channels`], else
    ///   [`StepError::InputLenMismatch`].
    /// - Returns the indices of **LIF** neurons that fired this step (Izhikevich
    ///   spikes are not listed in the return value).
    ///
    /// # Order of work
    ///
    /// 1. Store `modulators` and derive stress / learning rates.
    /// 2. Recompute LIF targets from neuromodulators: assign `decay_rate`
    ///    directly; soft-update `threshold` toward its target (learning-rate blend).
    /// 3. Update per-channel predictive EMA and surprise (`pred_errors`).
    /// 4. For each channel with `|stimuli| > 0.01`, run a Bernoulli trial
    ///    with probability `clamp(|stimuli|, 0.0, 1.0)` and stamp
    ///    `input_spike_times` on success.
    /// 5. Integrate each LIF neuron (weighted stimuli + surprise), then `check_fire`.
    /// 6. Lateral inhibition on non-firing LIF cells if anyone spiked.
    /// 7. R-STDP on LIF weights (`apply_stdp`): decay and accumulate every
    ///    [`crate::EligibilityTrace`] regardless of dopamine, then convert traces
    ///    into weight changes only when the dopamine-derived learning rate is
    ///    above ≈ 0.
    /// 8. Renormalize LIF weights toward an L1 budget, then clamp to the
    ///    [`RmStdpConfig`] bounds. Applies only to a neuron whose weights already
    ///    sum above `1e-6`; a blank neuron stays blank rather than being scaled
    ///    up to the budget, and a synapse at exactly zero is left alone so a
    ///    positive `w_min` cannot conjure a connection on an unrewarded step.
    ///    **Bounds take precedence over the budget.** Under the default bounds
    ///    the clamp provably cannot bind — weights are non-negative and `w_max`
    ///    equals the budget — so the L1 sum lands on budget exactly. A binding
    ///    bound is still enforced, leaving the sum off budget in whichever
    ///    direction it binds.
    /// 9. Drive each Izhikevich neuron from mean LIF membrane potential + dopamine.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuromod::{NeuroModulators, SpikingNetwork, StepError};
    ///
    /// let mut net = SpikingNetwork::with_dimensions(8, 2, 4);
    /// let modulators = NeuroModulators::default();
    ///
    /// // Wrong length → structured error
    /// assert!(matches!(
    ///     net.step(&[0.1, 0.2], &modulators),
    ///     Err(StepError::InputLenMismatch { expected: 4, got: 2 })
    /// ));
    ///
    /// let spikes = net.step(&[0.5; 4], &modulators).expect("length matches");
    /// assert!(spikes.iter().all(|&i| i < 8));
    /// ```
    pub fn step(
        &mut self,
        stimuli: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<Vec<usize>, StepError> {
        if stimuli.len() != self.num_channels {
            return Err(StepError::InputLenMismatch {
                expected: self.num_channels,
                got: stimuli.len(),
            });
        }

        self.global_step += 1;
        self.modulators = *modulators;

        let stress_multiplier = (1.0 - self.modulators.norepinephrine).max(0.1);
        let learning_rate = 0.5 * self.modulators.dopamine;

        for neuron in &mut self.neurons {
            let target_decay = 0.15 - (0.05 * self.modulators.acetylcholine);
            neuron.decay_rate = target_decay;

            let global_target = 0.20 - (0.05 * self.modulators.dopamine)
                + (0.15 * self.modulators.norepinephrine)
                - (0.05 * self.modulators.serotonin);
            let target_threshold =
                (global_target + if neuron.last_spike { 0.005 } else { -0.001 }).clamp(0.05, 0.50);
            neuron.threshold += (target_threshold - neuron.threshold) * learning_rate;
            neuron.threshold = neuron.threshold.clamp(0.05, 0.50);
        }

        const PRED_ALPHA: f32 = 0.1;
        const PRED_ERR_WEIGHT: f32 = 0.5;
        let mut pred_errors = vec![0.0_f32; self.num_channels];

        for ch in 0..self.num_channels {
            let s = stimuli[ch].abs().clamp(0.0, 1.0);
            pred_errors[ch] = (s - self.predictive_state[ch]).abs();
            self.predictive_state[ch] =
                PRED_ALPHA * s + (1.0 - PRED_ALPHA) * self.predictive_state[ch];
        }

        let mut rng = rand::rng();
        for (ch, &s) in stimuli.iter().enumerate() {
            let abs_s = s.abs().clamp(0.0, 1.0);
            if abs_s > 0.01 && rng.random_range(0.0..1.0) < abs_s {
                self.input_spike_times[ch] = self.global_step;
            }
        }

        for neuron in &mut self.neurons {
            let mut total_current = 0.0;
            for ch in 0..self.num_channels {
                if ch >= neuron.weights.len() {
                    continue;
                }
                let stim = stimuli[ch].abs().clamp(0.0, 1.0);
                let surprise = PRED_ERR_WEIGHT * pred_errors[ch];
                total_current += neuron.weights[ch] * (stim + surprise);
            }
            total_current *= 0.45 * stress_multiplier;
            neuron.integrate(total_current);
        }

        let mut spike_ids = Vec::new();
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            if let Some(_peak_v) = neuron.check_fire() {
                neuron.last_spike = true;
                neuron.last_spike_time = self.global_step;
                spike_ids.push(i);
            } else {
                neuron.last_spike = false;
            }
        }

        if !spike_ids.is_empty() {
            const INHIBITION_STRENGTH: f32 = 0.05;
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                if !spike_ids.contains(&i) {
                    neuron.membrane_potential =
                        (neuron.membrane_potential - INHIBITION_STRENGTH).max(0.0);
                }
            }
        }

        self.apply_stdp(learning_rate);

        // Scale toward the L1 budget, then enforce the configured bounds. The
        // bounds win where the two disagree: under the defaults they cannot
        // bind here, so the budget holds exactly; a narrowed range is honored
        // and leaves the sum off budget. See the `step` contract, item 8.
        let (w_min, w_max) = self.stdp_config.weight_bounds();
        for neuron in &mut self.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > 1e-6 {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    // A synapse at exactly zero is unconnected. Rescaling and
                    // capping the connected ones is this pass's job, but a
                    // positive `w_min` must not conjure a connection here: this
                    // loop runs every step, dopamine or not, and an unrewarded
                    // step must leave weights alone. Learning raises a synapse
                    // to the floor in `apply_stdp`, which is reward-gated.
                    if *w == 0.0 {
                        continue;
                    }
                    *w *= scale;
                    *w = w.clamp(w_min, w_max);
                }
            }
        }

        let lif_mean = if !self.neurons.is_empty() {
            let sum: f32 = self.neurons.iter().map(|n| n.membrane_potential).sum();
            sum / self.neurons.len() as f32
        } else {
            0.0
        };

        let iz_drive = (lif_mean * 20.0 + self.modulators.dopamine * 5.0).clamp(0.0, 15.0);
        for iz in &mut self.iz_neurons {
            iz.step(iz_drive);
        }

        Ok(spike_ids)
    }

    /// Reward-modulated STDP over the per-synapse eligibility traces.
    ///
    /// Runs on every step. Traces decay and accumulate independently of
    /// `dopamine_lr`; only the trace → weight conversion is gated on it, which is
    /// what lets reward arriving *after* a coincidence still pay for it.
    ///
    /// A coincidence is recorded once, on the step it happens — when the post
    /// neuron fired now (`Δt = t_post − t_pre ≥ 0`, potentiation) or when the pre
    /// channel fired now after an earlier post spike (`Δt < 0`, depression).
    /// Re-applying the kernel every step from stale `last_spike_time` values
    /// would inflate one spike pair into sustained learning.
    fn apply_stdp(&mut self, dopamine_lr: f32) {
        let now = self.global_step;
        let config = self.stdp_config;
        let (w_min, w_max) = config.weight_bounds();
        let reward_lr = config.effective_reward_lr();
        let rewarding = dopamine_lr >= 1e-6;
        let input_times = &self.input_spike_times;

        for neuron in &mut self.neurons {
            // Pre-0.6 deserialized state carries no traces, and a caller may have
            // resized `weights` by hand; keep the two vectors index-compatible.
            if neuron.eligibility.len() != neuron.weights.len() {
                neuron.eligibility.resize(
                    neuron.weights.len(),
                    EligibilityTrace::new(config.effective_tau_eligibility()),
                );
            }

            let post_time = neuron.last_spike_time;

            for (ch, &pre_time) in input_times.iter().enumerate().take(neuron.weights.len()) {
                let trace = &mut neuron.eligibility[ch];

                // An untouched trace decays to itself; skip the `exp` so a blank
                // or unrewarded network stays cheap at large channel counts.
                if trace.value != 0.0 {
                    trace.decay();
                }

                if pre_time >= 0
                    && post_time >= 0
                    && (post_time == now || (pre_time == now && post_time < pre_time))
                {
                    trace.accumulate((post_time - pre_time) as f32);
                }

                if rewarding && trace.value != 0.0 {
                    let dw = reward_lr * dopamine_lr * trace.value;
                    neuron.weights[ch] = (neuron.weights[ch] + dw).clamp(w_min, w_max);
                }
            }
        }
    }

    /// Get current membrane potentials for all neurons.
    pub fn get_membrane_potentials(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.membrane_potential).collect()
    }

    /// Get current thresholds for all neurons.
    pub fn get_thresholds(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.threshold).collect()
    }

    /// Reset network to initial state.
    pub fn reset(&mut self) {
        self.global_step = 0;
        self.input_spike_times = vec![-1; self.num_channels];
        self.predictive_state = vec![0.0; self.num_channels];

        for neuron in &mut self.neurons {
            neuron.membrane_potential = 0.0;
            neuron.last_spike = false;
            neuron.last_spike_time = -1;
            for trace in &mut neuron.eligibility {
                trace.reset();
            }
        }

        self.modulators = NeuroModulators::default();
    }
}

impl Default for SpikingNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation_defaults() {
        let network = SpikingNetwork::new();
        assert_eq!(network.neurons.len(), 16);
        assert_eq!(network.iz_neurons.len(), 5);
        assert_eq!(network.num_channels, 16);
        assert_eq!(network.global_step, 0);
    }

    #[test]
    fn test_network_creation_dynamic() {
        let network = SpikingNetwork::with_dimensions(518, 5, 518);
        assert_eq!(network.neurons.len(), 518);
        assert_eq!(network.iz_neurons.len(), 5);
        assert_eq!(network.num_channels, 518);
        assert_eq!(network.input_spike_times.len(), 518);
        assert_eq!(network.predictive_state.len(), 518);
        assert_eq!(network.neurons[0].weights.len(), 518);
        assert_eq!(network.neurons[0].eligibility.len(), 518);
        assert_eq!(network.stdp_config, RmStdpConfig::default());
        assert_eq!(
            network.neurons[0].eligibility[0].tau,
            RmStdpConfig::default().tau_eligibility
        );
    }

    #[test]
    fn test_default_matches_new() {
        let a = SpikingNetwork::new();
        let b = SpikingNetwork::default();
        assert_eq!(a.neurons.len(), b.neurons.len());
        assert_eq!(a.iz_neurons.len(), b.iz_neurons.len());
        assert_eq!(a.num_channels, b.num_channels);
    }

    #[test]
    fn test_network_step() {
        let mut network = SpikingNetwork::new();
        let stimuli = vec![0.5; network.num_channels];
        let modulators = NeuroModulators::default();

        let spikes = network
            .step(&stimuli, &modulators)
            .expect("valid input length should pass");
        assert_eq!(network.global_step, 1);
        assert!(spikes.len() <= network.neurons.len());
    }

    #[test]
    fn test_step_input_mismatch_returns_error_and_preserves_state() {
        let mut network = SpikingNetwork::new();
        let modulators = NeuroModulators::default();
        let wrong = vec![0.5; network.num_channels - 1];
        let before_step = network.global_step;
        let before_pred = network.predictive_state.clone();
        let before_mod = network.modulators;

        let result = network.step(&wrong, &modulators);

        assert_eq!(
            result,
            Err(StepError::InputLenMismatch {
                expected: network.num_channels,
                got: network.num_channels - 1
            })
        );
        assert_eq!(network.global_step, before_step);
        assert_eq!(network.predictive_state, before_pred);
        assert_eq!(network.modulators, before_mod);
    }

    #[test]
    fn test_membrane_potentials() {
        let network = SpikingNetwork::new();
        let potentials = network.get_membrane_potentials();
        assert_eq!(potentials.len(), 16);
        for &p in &potentials {
            assert_eq!(p, 0.0);
        }
    }

    // --- Reward-gated STDP over eligibility traces (GH#72 / GH#73 / GH#74) ---

    /// Four LIF neurons over four channels, weights seeded so the L1
    /// renormalization pass in `step` is an exact no-op (`sum == WEIGHT_BUDGET`).
    /// That isolates weight movement caused by learning from weight movement
    /// caused by rescaling.
    fn rstdp_test_network() -> SpikingNetwork {
        const CHANNELS: usize = 4;
        let mut network = SpikingNetwork::with_dimensions(4, 1, CHANNELS);
        let seed = WEIGHT_BUDGET / CHANNELS as f32;
        for neuron in &mut network.neurons {
            neuron.weights = vec![seed; CHANNELS];
        }
        network
    }

    /// Channels 0 and 1 spike on every step (`|s| = 1.0` always wins the
    /// Bernoulli trial); channels 2 and 3 never spike (`|s| <= 0.01` skips the
    /// trial entirely). No RNG outcome is left to chance.
    const DRIVEN_AND_SILENT: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

    #[test]
    fn test_traces_accumulate_without_dopamine_but_weights_hold() {
        let mut network = rstdp_test_network();
        let before = network.neurons[0].weights.clone();
        let no_reward = NeuroModulators::default();
        assert_eq!(no_reward.dopamine, 0.0);

        for _ in 0..25 {
            network
                .step(&DRIVEN_AND_SILENT, &no_reward)
                .expect("length matches");
        }

        // Learning is gated off, so not one weight moved...
        assert_eq!(network.neurons[0].weights, before);
        // ...but the driven synapses still banked the coincidences.
        assert!(
            network.neurons[0].eligibility[0].value > 0.0,
            "driven channel should hold a positive trace, got {}",
            network.neurons[0].eligibility[0].value
        );
        assert_eq!(network.neurons[0].eligibility[2].value, 0.0);
    }

    #[test]
    fn test_dopamine_converts_traces_into_weight_change() {
        let mut network = rstdp_test_network();
        let seed = network.neurons[0].weights[0];
        let reward = NeuroModulators {
            dopamine: 0.8,
            ..Default::default()
        };

        for _ in 0..25 {
            network
                .step(&DRIVEN_AND_SILENT, &reward)
                .expect("length matches");
        }

        for neuron in &network.neurons {
            assert!(
                neuron.weights[0] > seed && neuron.weights[1] > seed,
                "driven synapses should potentiate: {:?}",
                neuron.weights
            );
            assert!(
                neuron.weights[2] < seed && neuron.weights[3] < seed,
                "silent synapses should lose share of the L1 budget: {:?}",
                neuron.weights
            );
            assert!(neuron.eligibility[0].value > 0.0);
        }
    }

    #[test]
    fn test_weight_change_scales_with_dopamine_level() {
        let run = |dopamine: f32| {
            let mut network = rstdp_test_network();
            let modulators = NeuroModulators {
                dopamine,
                ..Default::default()
            };
            for _ in 0..25 {
                network
                    .step(&DRIVEN_AND_SILENT, &modulators)
                    .expect("length matches");
            }
            network.neurons[0].weights[0]
        };

        let weak = run(0.2);
        let strong = run(0.9);
        assert!(
            strong > weak,
            "more dopamine must buy more learning: {strong} vs {weak}"
        );
    }

    #[test]
    fn test_reward_pays_out_the_banked_trace_not_just_the_latest_spike() {
        let reward = NeuroModulators {
            dopamine: 0.8,
            ..Default::default()
        };

        // Bank ten steps of coincidences with reward switched off, then reward once.
        let mut banked = rstdp_test_network();
        let no_reward = NeuroModulators::default();
        for _ in 0..10 {
            banked
                .step(&DRIVEN_AND_SILENT, &no_reward)
                .expect("length matches");
        }
        let before_reward = banked.neurons[0].weights[0];
        banked
            .step(&DRIVEN_AND_SILENT, &reward)
            .expect("length matches");
        let banked_gain = banked.neurons[0].weights[0] - before_reward;

        // Identical reward on an identical step, with nothing banked behind it.
        let mut fresh = rstdp_test_network();
        let fresh_seed = fresh.neurons[0].weights[0];
        fresh
            .step(&DRIVEN_AND_SILENT, &reward)
            .expect("length matches");
        let fresh_gain = fresh.neurons[0].weights[0] - fresh_seed;

        assert!(banked_gain > 0.0 && fresh_gain > 0.0);
        assert!(
            banked_gain > 5.0 * fresh_gain,
            "the accumulated trace, not the latest coincidence alone, must drive \
             the update: {banked_gain} vs {fresh_gain}"
        );
    }

    #[test]
    fn test_trace_converts_on_a_step_with_no_new_coincidence() {
        // Two channels, weights summing to the L1 budget so renormalization
        // cannot manufacture the difference this test looks for.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![WEIGHT_BUDGET / 2.0; 2];
        let seed = network.neurons[0].weights[0];
        // Credit earned earlier, on steps this network has no memory of beyond
        // the trace itself.
        network.neurons[0].eligibility[0].value = 0.5;

        // Zero stimuli: no pre spike is stamped, and with no drive (and no
        // prediction error to be surprised by) the neuron cannot fire either.
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };
        network.step(&[0.0, 0.0], &reward).expect("length matches");

        assert_eq!(network.input_spike_times[0], -1, "no pre spike");
        assert_eq!(network.neurons[0].last_spike_time, -1, "no post spike");
        assert!(
            network.neurons[0].eligibility[0].value < 0.5,
            "the trace should have decayed, not grown"
        );
        assert!(
            network.neurons[0].weights[0] > seed,
            "deferred credit: reward converts the banked trace with no new spikes"
        );
        assert!(network.neurons[0].weights[0] > network.neurons[0].weights[1]);
    }

    #[test]
    fn test_post_before_pre_drives_depression_and_respects_w_min() {
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        // Zero weights mean zero drive, so the neuron cannot fire and
        // `last_spike_time` keeps the post spike we plant here — strictly before
        // the pre spike that channel 0 emits on the step below.
        network.neurons[0].weights = vec![0.0, 0.0];
        network.neurons[0].last_spike_time = 0;
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        network.step(&[1.0, 0.0], &reward).expect("length matches");

        assert_eq!(network.neurons[0].last_spike_time, 0, "must not have fired");
        assert!(
            network.neurons[0].eligibility[0].value < 0.0,
            "post-before-pre is depression, got {}",
            network.neurons[0].eligibility[0].value
        );
        assert_eq!(
            network.neurons[0].weights[0],
            network.stdp_config.weight_bounds().0,
            "depression must clamp at w_min, not go negative"
        );
    }

    #[test]
    fn test_one_spike_pair_is_counted_once_not_re_accumulated() {
        let mut network = SpikingNetwork::with_dimensions(1, 1, 1);
        network.neurons[0].weights = vec![0.0];
        network.neurons[0].last_spike_time = 0;
        let no_reward = NeuroModulators::default();

        // Step 1 stamps a pre spike; the planted post spike at t=0 precedes it.
        network.step(&[1.0], &no_reward).expect("length matches");
        let after_event = network.neurons[0].eligibility[0].value;
        assert!(after_event < 0.0);

        // Step 2 has no stimulus, so neither side spikes: the stale pair must
        // not be re-counted, leaving pure decay toward zero.
        network.step(&[0.0], &no_reward).expect("length matches");
        let after_quiet = network.neurons[0].eligibility[0].value;

        assert!(
            after_quiet > after_event && after_quiet < 0.0,
            "expected decay toward zero, got {after_event} -> {after_quiet}"
        );
    }

    #[test]
    fn test_reset_clears_eligibility_traces() {
        let mut network = rstdp_test_network();
        let reward = NeuroModulators {
            dopamine: 0.8,
            ..Default::default()
        };
        for _ in 0..10 {
            network
                .step(&DRIVEN_AND_SILENT, &reward)
                .expect("length matches");
        }
        assert!(network.neurons[0].eligibility[0].value > 0.0);

        network.reset();

        for neuron in &network.neurons {
            assert!(neuron.eligibility.iter().all(|t| t.value == 0.0));
            // `tau` survives a reset; only the accumulated value is cleared.
            assert!(
                neuron
                    .eligibility
                    .iter()
                    .all(|t| t.tau == network.stdp_config.tau_eligibility)
            );
        }
    }

    #[test]
    fn test_set_rm_stdp_config_normalizes_what_it_stores() {
        let mut network = rstdp_test_network();

        network.set_rm_stdp_config(RmStdpConfig {
            tau_eligibility: f32::NAN,
            reward_lr: f32::INFINITY,
            w_min: 1.5,
            w_max: 0.2, // reversed
        });

        // The setter installs guarded values rather than storing nonsense and
        // working around it at every read.
        assert_eq!(network.stdp_config, RmStdpConfig::default());
        for neuron in &network.neurons {
            assert!(
                neuron
                    .eligibility
                    .iter()
                    .all(|t| t.tau == RmStdpConfig::default().tau_eligibility)
            );
        }
    }

    #[test]
    fn test_set_rm_stdp_config_retaus_existing_traces() {
        let mut network = rstdp_test_network();
        let config = RmStdpConfig {
            tau_eligibility: 100.0,
            reward_lr: 0.02,
            w_min: 0.1,
            w_max: 1.5,
        };

        network.set_rm_stdp_config(config);

        assert_eq!(network.stdp_config, config);
        for neuron in &network.neurons {
            assert!(neuron.eligibility.iter().all(|t| t.tau == 100.0));
        }
    }

    #[test]
    fn test_default_bounds_keep_the_l1_sum_on_budget() {
        let mut network = rstdp_test_network();
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        for _ in 0..40 {
            network
                .step(&DRIVEN_AND_SILENT, &reward)
                .expect("length matches");
        }

        for neuron in &network.neurons {
            let total: f32 = neuron.weights.iter().sum();
            assert!(
                (total - WEIGHT_BUDGET).abs() < 1e-4,
                "the default clamp cannot bind, so the budget holds: got {total}"
            );
        }
    }

    #[test]
    fn test_configured_bounds_take_precedence_over_the_l1_budget() {
        // Four channels capped at 0.4 cannot reach the budget of 2.0 — the
        // documented precedence is that the bound wins and the sum sits under.
        let mut network = rstdp_test_network();
        network.set_rm_stdp_config(RmStdpConfig {
            w_max: 0.4,
            ..RmStdpConfig::default()
        });
        let reward = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        for _ in 0..40 {
            network
                .step(&DRIVEN_AND_SILENT, &reward)
                .expect("length matches");
        }

        for neuron in &network.neurons {
            assert!(
                neuron.weights.iter().all(|&w| w <= 0.4 + 1e-6),
                "weights must respect the configured w_max: {:?}",
                neuron.weights
            );
            let total: f32 = neuron.weights.iter().sum();
            assert!(
                total < WEIGHT_BUDGET,
                "a binding w_max holds the L1 sum under budget: got {total}"
            );
        }
    }

    #[test]
    fn test_non_finite_reward_lr_cannot_poison_weights() {
        // A NaN weight would survive forever: renormalization skips a neuron
        // whose total is not `> 1e-6`, and `NaN > 1e-6` is false.
        let mut network = rstdp_test_network();
        network.stdp_config.reward_lr = f32::NAN;
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        for _ in 0..10 {
            network
                .step(&DRIVEN_AND_SILENT, &reward)
                .expect("length matches");
        }

        for neuron in &network.neurons {
            assert!(
                neuron.weights.iter().all(|w| w.is_finite()),
                "a non-finite reward_lr must not poison weights: {:?}",
                neuron.weights
            );
        }
    }

    #[test]
    fn test_positive_w_min_does_not_seed_a_blank_network() {
        // Blank weights are the documented neutral initialization. Bounds gate
        // weight *updates*; they do not fabricate synaptic weight where the
        // network deliberately has none.
        let mut network = SpikingNetwork::with_dimensions(2, 1, 4);
        network.set_rm_stdp_config(RmStdpConfig {
            w_min: 0.1,
            ..RmStdpConfig::default()
        });
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        for _ in 0..5 {
            network.step(&[1.0; 4], &reward).expect("length matches");
        }

        for neuron in &network.neurons {
            assert!(
                neuron.weights.iter().all(|&w| w == 0.0),
                "a positive w_min must not seed blank weights: {:?}",
                neuron.weights
            );
        }
    }

    #[test]
    fn test_positive_w_min_does_not_seed_untouched_zero_weights() {
        // A partially connected neuron: channel 0 carries the whole budget,
        // channel 1 is unconnected. The nonzero total clears the `> 1e-6` guard,
        // so this reaches the clamp that a fully blank network never does.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![WEIGHT_BUDGET, 0.0];
        network.set_rm_stdp_config(RmStdpConfig {
            w_min: 0.1,
            ..RmStdpConfig::default()
        });
        let no_reward = NeuroModulators::default();

        network
            .step(&[0.0, 0.0], &no_reward)
            .expect("length matches");

        assert_eq!(
            network.neurons[0].weights[1], 0.0,
            "an unrewarded step must not conjure a connection out of w_min"
        );
        assert_eq!(network.neurons[0].weights[0], WEIGHT_BUDGET);
    }

    #[test]
    fn test_non_finite_tau_eligibility_neither_erases_nor_freezes_traces() {
        for bad_tau in [f32::NAN, f32::INFINITY, 0.0, -5.0] {
            let mut network = rstdp_test_network();
            network.set_rm_stdp_config(RmStdpConfig {
                tau_eligibility: bad_tau,
                ..RmStdpConfig::default()
            });
            let no_reward = NeuroModulators::default();

            for _ in 0..10 {
                network
                    .step(&DRIVEN_AND_SILENT, &no_reward)
                    .expect("length matches");
            }

            // The default tau is installed instead, so credit is neither wiped
            // (NaN) nor held forever (+inf): it banks like any other run.
            let trace = network.neurons[0].eligibility[0].value;
            assert!(
                trace > 0.0 && trace.is_finite(),
                "tau {bad_tau} should fall back to the default, got trace {trace}"
            );
            assert_eq!(
                network.neurons[0].eligibility[0].tau,
                RmStdpConfig::default().tau_eligibility
            );
        }
    }

    #[test]
    fn test_pre_0_6_state_without_new_fields_loads_and_steps() {
        let mut network = SpikingNetwork::with_dimensions(2, 1, 3);
        for neuron in &mut network.neurons {
            neuron.weights = vec![0.4; 3];
        }

        // Strip the fields added in 0.6 to mimic a checkpoint written before
        // eligibility traces were wired in.
        let mut state = serde_json::to_value(&network).expect("network serializes");
        let object = state.as_object_mut().expect("network is a JSON object");
        object.remove("stdp_config");
        for neuron in object["neurons"].as_array_mut().expect("neurons array") {
            neuron
                .as_object_mut()
                .expect("neuron is a JSON object")
                .remove("eligibility");
        }

        let mut restored: SpikingNetwork =
            serde_json::from_value(state).expect("pre-0.6 state still deserializes");
        assert!(restored.neurons[0].eligibility.is_empty());
        assert_eq!(restored.stdp_config, RmStdpConfig::default());

        let reward = NeuroModulators {
            dopamine: 0.7,
            ..Default::default()
        };
        for _ in 0..5 {
            restored
                .step(&[0.9, 0.9, 0.9], &reward)
                .expect("length matches");
        }

        assert_eq!(restored.neurons[0].eligibility.len(), 3);
        assert!(
            restored.neurons[0]
                .eligibility
                .iter()
                .all(|t| t.tau == RmStdpConfig::default().tau_eligibility)
        );
    }
}
