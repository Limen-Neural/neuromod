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

use core::fmt;

use rand::RngExt;
use serde::{Deserialize, Serialize};

use super::izhikevich::IzhikevichNeuron;
use super::lif::LifNeuron;
use super::modulators::NeuroModulators;
use super::rm_stdp::*;

/// L1 synaptic weight budget per neuron (total weight sum target).
const WEIGHT_BUDGET: f32 = 2.0;

/// Classification of a non-finite `f32` rejected by [`SpikingNetwork::step`].
///
/// The error reports the class, not the payload: every NaN (any sign / quiet
/// bit) is [`Self::Nan`], and the infinities are distinguished by sign. Finite
/// values, including signed zero and [`f32::MAX`] / [`f32::MIN`], are not
/// represented here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteClass {
    /// IEEE-754 NaN (any payload or sign).
    Nan,
    /// Positive infinity.
    PosInfinity,
    /// Negative infinity.
    NegInfinity,
}

impl NonFiniteClass {
    /// Classify `x` as NaN, `+∞`, or `−∞`.
    ///
    /// Returns `None` when `x` is finite.
    pub const fn classify(x: f32) -> Option<Self> {
        if x.is_nan() {
            Some(Self::Nan)
        } else if x.is_infinite() {
            if x.is_sign_positive() {
                Some(Self::PosInfinity)
            } else {
                Some(Self::NegInfinity)
            }
        } else {
            None
        }
    }
}

impl fmt::Display for NonFiniteClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nan => f.write_str("NaN"),
            Self::PosInfinity => f.write_str("+inf"),
            Self::NegInfinity => f.write_str("-inf"),
        }
    }
}

/// Named neuromodulator field rejected by [`SpikingNetwork::step`].
///
/// Variant order matches [`NeuroModulators`] field order and the preflight
/// scan: dopamine, serotonin, acetylcholine, norepinephrine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModulatorField {
    /// [`NeuroModulators::dopamine`].
    Dopamine,
    /// [`NeuroModulators::serotonin`].
    Serotonin,
    /// [`NeuroModulators::acetylcholine`].
    Acetylcholine,
    /// [`NeuroModulators::norepinephrine`].
    Norepinephrine,
}

impl fmt::Display for ModulatorField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dopamine => f.write_str("dopamine"),
            Self::Serotonin => f.write_str("serotonin"),
            Self::Acetylcholine => f.write_str("acetylcholine"),
            Self::Norepinephrine => f.write_str("norepinephrine"),
        }
    }
}

/// Errors from [`SpikingNetwork::step`].
///
/// Every variant is returned **before** the step mutates network state or
/// draws from the random-number generator (RNG). Adding a variant is a
/// source-level break for exhaustive `match`es: handle
/// [`Self::NonFiniteStimulus`] and [`Self::NonFiniteModulator`], or use a
/// `_` wildcard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepError {
    /// `stimuli.len()` did not match the network's `num_channels`.
    InputLenMismatch { expected: usize, got: usize },
    /// A stimulus sample was NaN or infinite.
    NonFiniteStimulus { index: usize, class: NonFiniteClass },
    /// A neuromodulator field was NaN or infinite.
    NonFiniteModulator {
        field: ModulatorField,
        class: NonFiniteClass,
    },
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputLenMismatch { expected, got } => {
                write!(f, "expected {expected} input channels, got {got}")
            }
            Self::NonFiniteStimulus { index, class } => {
                write!(f, "non-finite stimulus at index {index}: {class}")
            }
            Self::NonFiniteModulator { field, class } => {
                write!(f, "non-finite modulator {field}: {class}")
            }
        }
    }
}

impl core::error::Error for StepError {}

/// Length, then every stimulus, then each modulator field. Returns on the first
/// problem; never allocates. Callers must invoke this before any mutation or
/// RNG draw so a rejected step is a no-op.
fn validate_step_inputs(
    stimuli: &[f32],
    num_channels: usize,
    modulators: &NeuroModulators,
) -> Result<(), StepError> {
    if stimuli.len() != num_channels {
        return Err(StepError::InputLenMismatch {
            expected: num_channels,
            got: stimuli.len(),
        });
    }

    for (index, &value) in stimuli.iter().enumerate() {
        if let Some(class) = NonFiniteClass::classify(value) {
            return Err(StepError::NonFiniteStimulus { index, class });
        }
    }

    for (field, value) in [
        (ModulatorField::Dopamine, modulators.dopamine),
        (ModulatorField::Serotonin, modulators.serotonin),
        (ModulatorField::Acetylcholine, modulators.acetylcholine),
        (ModulatorField::Norepinephrine, modulators.norepinephrine),
    ] {
        if let Some(class) = NonFiniteClass::classify(value) {
            return Err(StepError::NonFiniteModulator { field, class });
        }
    }

    Ok(())
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
    /// - Every stimulus and every modulator field must be **finite**. `NaN`,
    ///   `+∞`, and `−∞` return [`StepError::NonFiniteStimulus`] (with the
    ///   offending index) or [`StepError::NonFiniteModulator`] (with the
    ///   field name) plus the [`NonFiniteClass`]. Finite signed values,
    ///   including `0.0` / `-0.0` and [`f32::MAX`] / [`f32::MIN`], still
    ///   pass through the existing `abs().clamp(0.0, 1.0)` magnitude path.
    /// - **Failure-atomic:** any `Err` is returned before `global_step`
    ///   increments, before the modulator snapshot is stored, before
    ///   predictive state / membranes / traces / weights change, and before
    ///   any random-number generator (RNG) draw. A rejected step is a no-op.
    /// - Preflight is a single linear pass over the stimulus slice plus the
    ///   four modulator fields and allocates nothing.
    /// - Returns the indices of **LIF** neurons that fired this step (Izhikevich
    ///   spikes are not listed in the return value).
    ///
    /// # Order of work
    ///
    /// 0. Validate length and finiteness (`validate_step_inputs`); return
    ///    without mutating on failure.
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
    /// use neuromod::{NeuroModulators, NonFiniteClass, SpikingNetwork, StepError};
    ///
    /// let mut net = SpikingNetwork::with_dimensions(8, 2, 4);
    /// let modulators = NeuroModulators::default();
    ///
    /// // Wrong length → structured error, network untouched
    /// assert!(matches!(
    ///     net.step(&[0.1, 0.2], &modulators),
    ///     Err(StepError::InputLenMismatch { expected: 4, got: 2 })
    /// ));
    ///
    /// // Non-finite stimulus → indexed class, still a no-op
    /// assert!(matches!(
    ///     net.step(&[0.1, f32::NAN, 0.2, 0.3], &modulators),
    ///     Err(StepError::NonFiniteStimulus {
    ///         index: 1,
    ///         class: NonFiniteClass::Nan
    ///     })
    /// ));
    ///
    /// let spikes = net.step(&[0.5; 4], &modulators).expect("finite, length matches");
    /// assert!(spikes.iter().all(|&i| i < 8));
    /// ```
    pub fn step(
        &mut self,
        stimuli: &[f32],
        modulators: &NeuroModulators,
    ) -> Result<Vec<usize>, StepError> {
        validate_step_inputs(stimuli, self.num_channels, modulators)?;

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
                    // to the floor in `apply_stdp`, and only on a step where it
                    // actually applies an update.
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

                // A non-finite trace carries no credit, and paying it out would
                // poison the weight — `clamp` preserves NaN, and the L1 pass then
                // skips this neuron forever because a NaN total is never
                // `> 1e-6`. Clear it so the synapse can learn again. `value` is
                // public and deserializable, so this is reachable without ever
                // going through `accumulate`.
                if !trace.value.is_finite() {
                    trace.reset();
                } else if trace.value != 0.0 {
                    // An untouched trace decays to itself; skip the `exp` so a
                    // blank or unrewarded network stays cheap at large channel
                    // counts.
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
                    let w = neuron.weights[ch];
                    // The bounds must never move a weight on their own. Writing
                    // unconditionally lets them do exactly that on a synapse
                    // left at exactly zero -- unconnected, the state the L1 pass
                    // below is careful to preserve -- in two ways:
                    //
                    // - `reward_lr = 0.0` disables conversion, yet a `dw` of
                    //   zero still clamps the synapse up to a positive `w_min`.
                    // - Depression (`dw < 0`) has nothing to take away, and its
                    //   negative result clamps up to `w_min` too, connecting a
                    //   synapse by weakening it.
                    //
                    // Potentiation is the one update that may bring a synapse
                    // online, and it still lands on the floor where `w_min`
                    // binds. Everywhere else the L1 pass re-clamps each step.
                    if dw > 0.0 || (dw < 0.0 && w != 0.0) {
                        neuron.weights[ch] = (w + dw).clamp(w_min, w_max);
                    }
                }
            }

            // A caller can hand a neuron more weights than the network has
            // channels. Those synapses have no input that could spike, so they
            // only ever decay — but the loop above stops at the last channel,
            // which would leave a planted or deserialized trace frozen there
            // forever. Empty in the usual case, where the two lengths agree.
            for trace in neuron.eligibility.iter_mut().skip(input_times.len()) {
                if !trace.value.is_finite() {
                    trace.reset();
                } else if trace.value != 0.0 {
                    trace.decay();
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
        let before = capture_network(&network);

        let result = network.step(&wrong, &modulators);

        assert_eq!(
            result,
            Err(StepError::InputLenMismatch {
                expected: network.num_channels,
                got: network.num_channels - 1
            })
        );
        assert_network_unchanged(&network, &before);
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
    fn test_zero_reward_rate_leaves_an_unconnected_synapse_at_zero() {
        // `reward_lr = 0.0` turns trace conversion off. A positive `w_min` must
        // not then stand in for it: with no update applied there is nothing to
        // clamp, and raising an unconnected synapse to the floor would let a
        // disabled learning rate change connectivity. The renormalization pass
        // would then scale that fabricated weight toward the budget.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![0.0, WEIGHT_BUDGET];
        network.set_rm_stdp_config(RmStdpConfig {
            reward_lr: 0.0,
            w_min: 0.1,
            ..RmStdpConfig::default()
        });
        // Credit is banked, so the conversion branch is entered every step.
        network.neurons[0].eligibility[0].value = 0.5;

        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };
        for _ in 0..5 {
            network.step(&[0.0, 0.0], &reward).expect("length matches");
        }

        assert!(
            network.neurons[0].eligibility[0].value > 0.0,
            "the trace should still be banked, so the branch really ran"
        );
        assert_eq!(
            network.neurons[0].weights[0], 0.0,
            "a zero learning rate applied no update, so the floor must not connect this synapse"
        );
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
    fn test_rewarded_depression_does_not_connect_a_zero_weight_synapse() {
        // Depression on an unconnected synapse: `(0.0 + dw).clamp(w_min, w_max)`
        // with a negative `dw` and a positive floor returns `w_min`, so paying
        // out a negative trace would *create* the connection depression is
        // supposed to weaken. Skipping a zero update is not enough here -- `dw`
        // is genuinely non-zero, it just points the wrong way.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![0.0, WEIGHT_BUDGET];
        network.set_rm_stdp_config(RmStdpConfig {
            w_min: 0.1,
            ..RmStdpConfig::default()
        });
        // Zero weight on channel 0 means no drive from it, so the neuron cannot
        // fire and keeps the post spike planted here -- strictly before the pre
        // spike channel 0 emits on the step below.
        network.neurons[0].last_spike_time = 0;
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        network.step(&[1.0, 0.0], &reward).expect("length matches");

        assert_eq!(network.neurons[0].last_spike_time, 0, "must not have fired");
        assert!(
            network.neurons[0].eligibility[0].value < 0.0,
            "expected a depressing trace, got {}",
            network.neurons[0].eligibility[0].value
        );
        assert_eq!(
            network.neurons[0].weights[0], 0.0,
            "depression must not lift an unconnected synapse to the floor"
        );
    }
    #[test]
    fn test_depression_clamps_a_connected_synapse_at_w_min() {
        // The floor still binds wherever an update actually applies: a
        // *connected* synapse depressed past `w_min` stops there instead of
        // going negative. This is the other half of the test above, which
        // covers the synapse that must not be connected in the first place.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![0.001, WEIGHT_BUDGET];
        network.neurons[0].eligibility[0].value = -0.5;
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        // Zero stimuli: nothing spikes, so the banked trace is the only input
        // to the update and its sign is not in doubt.
        network.step(&[0.0, 0.0], &reward).expect("length matches");

        assert_eq!(
            network.neurons[0].weights[0],
            network.stdp_config.weight_bounds().0,
            "depression must clamp at w_min, not go negative"
        );
    }
    #[test]
    fn test_traces_past_the_last_channel_still_decay() {
        // Two channels, but a caller widened this neuron to four synapses. The
        // extra two have no input that could ever spike, so decay is all they
        // can do — and the per-channel loop stops before reaching them, which
        // left a planted trace frozen there across every step.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![WEIGHT_BUDGET / 4.0; 4];
        network.neurons[0].eligibility = vec![EligibilityTrace::new(50.0); 4];
        network.neurons[0].eligibility[3].value = 0.5;

        let no_reward = NeuroModulators::default();
        for _ in 0..5 {
            network
                .step(&[0.0, 0.0], &no_reward)
                .expect("length matches");
        }

        let orphan = network.neurons[0].eligibility[3].value;
        assert!(
            orphan > 0.0 && orphan < 0.5,
            "a trace past the last channel should decay toward zero, got {orphan}"
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
    fn test_non_finite_trace_value_cannot_poison_weights() {
        // `EligibilityTrace::value` is public and deserializable, so a NaN can
        // arrive without ever passing through `accumulate`.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![1.0, 1.0];
        network.neurons[0].eligibility[0].value = f32::NAN;
        network.neurons[0].eligibility[1].value = f32::INFINITY;
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        for _ in 0..5 {
            network.step(&[1.0, 1.0], &reward).expect("length matches");
        }

        assert!(
            network.neurons[0].weights.iter().all(|w| w.is_finite()),
            "a non-finite trace must not poison weights: {:?}",
            network.neurons[0].weights
        );
        assert!(
            network.neurons[0]
                .eligibility
                .iter()
                .all(|t| t.value.is_finite()),
            "the trace itself should be cleared, not left non-finite"
        );
    }

    #[test]
    fn test_negative_w_min_cannot_make_weights_inhibitory() {
        // Ordered and finite, but inhibitory — unsupported by this crate. Left
        // unguarded, a rewarded depression drives weights negative and the L1
        // pass then skips the neuron forever, since a negative total never
        // satisfies its `> 1e-6` guard.
        let mut network = SpikingNetwork::with_dimensions(1, 1, 2);
        network.neurons[0].weights = vec![0.5, 0.5];
        network.set_rm_stdp_config(RmStdpConfig {
            w_min: -2.0,
            w_max: -1.0,
            ..RmStdpConfig::default()
        });
        let reward = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };

        for _ in 0..5 {
            network.step(&[1.0, 1.0], &reward).expect("length matches");
        }

        for neuron in &network.neurons {
            assert!(
                neuron.weights.iter().all(|&w| w >= 0.0),
                "weights must stay excitatory: {:?}",
                neuron.weights
            );
        }
        assert_eq!(network.stdp_config.weight_bounds(), (0.0, 2.0));
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

    // --- Non-finite step ingress (LIM-1226) ---

    /// Full serialized snapshot so a rejected step cannot hide a mutation in
    /// any serde-visible field (`global_step`, membranes, traces, weights,
    /// predictive state, Izhikevich bank, …).
    fn capture_network(network: &SpikingNetwork) -> serde_json::Value {
        serde_json::to_value(network).expect("network serializes")
    }

    fn assert_network_unchanged(network: &SpikingNetwork, before: &serde_json::Value) {
        assert_eq!(
            &capture_network(network),
            before,
            "rejected step must be a no-op on every serialized field"
        );
    }

    fn restored_blank_network(channels: usize) -> SpikingNetwork {
        serde_json::from_value(
            serde_json::to_value(SpikingNetwork::with_dimensions(2, 1, channels))
                .expect("blank network serializes"),
        )
        .expect("round-trip restores a network")
    }

    /// Four LIF / four channels with L1-neutral weights, driven once at `|s| =
    /// 1.0` so Bernoulli outcomes are deterministic while still leaving
    /// non-trivial spike, trace, and predictive state to protect.
    fn ingress_test_network() -> SpikingNetwork {
        let mut network = rstdp_test_network();
        let mods = NeuroModulators {
            dopamine: 0.4,
            serotonin: 0.1,
            acetylcholine: 0.2,
            norepinephrine: 0.3,
        };
        network
            .step(&[1.0; 4], &mods)
            .expect("finite warm-up must succeed");
        network
    }

    const NON_FINITE_CASES: [(f32, NonFiniteClass); 3] = [
        (f32::NAN, NonFiniteClass::Nan),
        (f32::INFINITY, NonFiniteClass::PosInfinity),
        (f32::NEG_INFINITY, NonFiniteClass::NegInfinity),
    ];

    #[test]
    fn test_non_finite_class_distinguishes_nan_and_signed_infinities() {
        assert_eq!(NonFiniteClass::classify(0.0), None);
        assert_eq!(NonFiniteClass::classify(-0.0), None);
        assert_eq!(NonFiniteClass::classify(f32::MAX), None);
        assert_eq!(NonFiniteClass::classify(f32::MIN), None);
        assert_eq!(
            NonFiniteClass::classify(f32::NAN),
            Some(NonFiniteClass::Nan)
        );
        assert_eq!(
            NonFiniteClass::classify(-f32::NAN),
            Some(NonFiniteClass::Nan)
        );
        assert_eq!(
            NonFiniteClass::classify(f32::INFINITY),
            Some(NonFiniteClass::PosInfinity)
        );
        assert_eq!(
            NonFiniteClass::classify(f32::NEG_INFINITY),
            Some(NonFiniteClass::NegInfinity)
        );
    }

    #[test]
    fn test_non_finite_stimulus_rejected_atomically_for_each_class() {
        let mods = NeuroModulators::default();
        for (bad, class) in NON_FINITE_CASES {
            let mut network = ingress_test_network();
            let before = capture_network(&network);
            let mut stimuli = [0.5_f32; 4];
            stimuli[2] = bad;

            assert_eq!(
                network.step(&stimuli, &mods),
                Err(StepError::NonFiniteStimulus { index: 2, class })
            );
            assert_network_unchanged(&network, &before);
        }
    }

    #[test]
    fn test_non_finite_modulator_rejected_atomically_for_each_field() {
        let fields = [
            ModulatorField::Dopamine,
            ModulatorField::Serotonin,
            ModulatorField::Acetylcholine,
            ModulatorField::Norepinephrine,
        ];
        for field in fields {
            for (bad, class) in NON_FINITE_CASES {
                let mut network = ingress_test_network();
                let before = capture_network(&network);
                let mut mods = NeuroModulators {
                    dopamine: 0.4,
                    serotonin: 0.1,
                    acetylcholine: 0.2,
                    norepinephrine: 0.3,
                };
                match field {
                    ModulatorField::Dopamine => mods.dopamine = bad,
                    ModulatorField::Serotonin => mods.serotonin = bad,
                    ModulatorField::Acetylcholine => mods.acetylcholine = bad,
                    ModulatorField::Norepinephrine => mods.norepinephrine = bad,
                }

                assert_eq!(
                    network.step(&[0.5; 4], &mods),
                    Err(StepError::NonFiniteModulator { field, class })
                );
                assert_network_unchanged(&network, &before);
            }
        }
    }

    #[test]
    fn test_non_finite_late_in_a_long_stimulus_slice_is_a_complete_preflight() {
        const CHANNELS: usize = 518;
        let mut network = SpikingNetwork::with_dimensions(4, 1, CHANNELS);
        for neuron in &mut network.neurons {
            neuron.weights = vec![WEIGHT_BUDGET / CHANNELS as f32; CHANNELS];
        }
        let mods = NeuroModulators::default();
        let before = capture_network(&network);

        let mut stimuli = vec![0.25_f32; CHANNELS];
        stimuli[CHANNELS - 1] = f32::NAN;

        assert_eq!(
            network.step(&stimuli, &mods),
            Err(StepError::NonFiniteStimulus {
                index: CHANNELS - 1,
                class: NonFiniteClass::Nan
            })
        );
        assert_network_unchanged(&network, &before);
    }

    #[test]
    fn test_rejected_step_then_valid_step_matches_control_network() {
        // Control never sees the invalid call. Treatment rejects it, then both
        // take the same finite step. `|s| = 1.0` makes the Bernoulli outcome
        // deterministic so the serialized states can match without a
        // caller-owned RNG (LIM-1221). Because `rand::rng()` is reached only
        // after preflight, the rejected call also cannot consume a draw.
        let mut treatment = ingress_test_network();
        let mut control = ingress_test_network();
        assert_eq!(capture_network(&treatment), capture_network(&control));

        let mods = NeuroModulators {
            dopamine: 0.4,
            serotonin: 0.1,
            acetylcholine: 0.2,
            norepinephrine: 0.3,
        };
        let mut bad = [1.0_f32; 4];
        bad[3] = f32::INFINITY;
        assert_eq!(
            treatment.step(&bad, &mods),
            Err(StepError::NonFiniteStimulus {
                index: 3,
                class: NonFiniteClass::PosInfinity
            })
        );
        assert_eq!(capture_network(&treatment), capture_network(&control));

        let spikes_t = treatment
            .step(&[1.0; 4], &mods)
            .expect("finite retry must succeed");
        let spikes_c = control
            .step(&[1.0; 4], &mods)
            .expect("control step must succeed");
        assert_eq!(spikes_t, spikes_c);
        assert_eq!(capture_network(&treatment), capture_network(&control));
    }

    #[test]
    fn test_length_mismatch_wins_over_non_finite_samples() {
        let mut network = SpikingNetwork::with_dimensions(2, 1, 4);
        let before = capture_network(&network);
        let mods = NeuroModulators {
            dopamine: f32::NAN,
            ..Default::default()
        };

        assert_eq!(
            network.step(&[f32::NAN, 0.1], &mods),
            Err(StepError::InputLenMismatch {
                expected: 4,
                got: 2
            })
        );
        assert_network_unchanged(&network, &before);
    }

    #[test]
    fn test_stimulus_error_wins_over_modulator_error() {
        let mut network = SpikingNetwork::with_dimensions(2, 1, 4);
        let before = capture_network(&network);
        let mods = NeuroModulators {
            dopamine: f32::NAN,
            ..Default::default()
        };
        let stimuli = [0.1, f32::NEG_INFINITY, 0.2, 0.3];

        assert_eq!(
            network.step(&stimuli, &mods),
            Err(StepError::NonFiniteStimulus {
                index: 1,
                class: NonFiniteClass::NegInfinity
            })
        );
        assert_network_unchanged(&network, &before);
    }

    #[test]
    fn test_finite_signed_and_extreme_stimuli_still_step() {
        let mods = NeuroModulators::default();
        for value in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            -0.5,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
        ] {
            let mut network = SpikingNetwork::with_dimensions(2, 1, 4);
            network
                .step(&[value; 4], &mods)
                .unwrap_or_else(|_| panic!("{value} is finite and must be accepted"));
            assert_eq!(network.global_step, 1);
            assert!(
                network.predictive_state.iter().all(|s| s.is_finite()),
                "finite input must not poison predictive state: {:?}",
                network.predictive_state
            );
        }
    }

    #[test]
    fn test_serde_json_overflow_stimuli_are_rejected() {
        // JSON has no Inf token. A magnitude past `f32::MAX` (~3.4e38) is still
        // a finite JSON/`f64` number and deserializes to `+inf` as `f32`.
        let pos_inf: f32 = serde_json::from_str("1e39").expect("f32 overflow is +inf");
        assert_eq!(
            NonFiniteClass::classify(pos_inf),
            Some(NonFiniteClass::PosInfinity)
        );

        let mut restored = restored_blank_network(3);
        let before = capture_network(&restored);
        let inf_stimuli: Vec<f32> =
            serde_json::from_str("[0.1, 1e39, 0.2]").expect("stimulus frame deserializes");
        assert_eq!(
            restored.step(&inf_stimuli, &NeuroModulators::default()),
            Err(StepError::NonFiniteStimulus {
                index: 1,
                class: NonFiniteClass::PosInfinity
            })
        );
        assert_network_unchanged(&restored, &before);
    }

    #[test]
    fn test_serde_json_overflow_modulators_are_rejected() {
        let neg_inf: f32 = serde_json::from_str("-1e39").expect("f32 overflow is -inf");
        assert_eq!(
            NonFiniteClass::classify(neg_inf),
            Some(NonFiniteClass::NegInfinity)
        );

        let mut restored = restored_blank_network(3);
        let before = capture_network(&restored);
        let inf_mods: NeuroModulators = serde_json::from_str(
            r#"{"dopamine":0.0,"serotonin":-1e39,"acetylcholine":0.0,"norepinephrine":0.0}"#,
        )
        .expect("modulator snapshot deserializes");
        assert_eq!(
            restored.step(&[0.1, 0.2, 0.3], &inf_mods),
            Err(StepError::NonFiniteModulator {
                field: ModulatorField::Serotonin,
                class: NonFiniteClass::NegInfinity
            })
        );
        assert_network_unchanged(&restored, &before);
    }

    #[test]
    fn test_serde_bit_pattern_nan_modulator_is_rejected() {
        // JSON has no NaN token; reconstitute it from a serde-decoded IEEE-754
        // bit pattern so the invalid payload still arrived through Deserialize.
        let nan_bits: u32 =
            serde_json::from_str("2143289344").expect("canonical quiet NaN payload");
        let nan = f32::from_bits(nan_bits);
        assert!(
            nan.is_nan(),
            "serde-decoded bits must be NaN, got {nan} from {nan_bits}"
        );

        let mut restored = restored_blank_network(3);
        let before = capture_network(&restored);
        let nan_mods: NeuroModulators = serde_json::from_value(serde_json::json!({
            "dopamine": 0.0,
            "serotonin": 0.0,
            "acetylcholine": 0.0,
            "norepinephrine": 0.0
        }))
        .expect("finite modulator object deserializes");
        let nan_mods = NeuroModulators {
            acetylcholine: nan,
            ..nan_mods
        };
        assert_eq!(
            restored.step(&[0.1, 0.2, 0.3], &nan_mods),
            Err(StepError::NonFiniteModulator {
                field: ModulatorField::Acetylcholine,
                class: NonFiniteClass::Nan
            })
        );
        assert_network_unchanged(&restored, &before);
    }

    #[test]
    fn test_step_error_display_names_index_field_and_class() {
        assert_eq!(
            StepError::InputLenMismatch {
                expected: 16,
                got: 2
            }
            .to_string(),
            "expected 16 input channels, got 2"
        );
        assert_eq!(
            StepError::NonFiniteStimulus {
                index: 7,
                class: NonFiniteClass::Nan
            }
            .to_string(),
            "non-finite stimulus at index 7: NaN"
        );
        assert_eq!(
            StepError::NonFiniteModulator {
                field: ModulatorField::Norepinephrine,
                class: NonFiniteClass::PosInfinity
            }
            .to_string(),
            "non-finite modulator norepinephrine: +inf"
        );
        assert_eq!(NonFiniteClass::NegInfinity.to_string(), "-inf");
    }
}
