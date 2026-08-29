//! # neuromod — Spiking neural network primitives
//!
//! Biologically grounded SNN building blocks for Rust: a topology-neutral
//! [`SpikingNetwork`] engine (LIF + Izhikevich banks), generic neuromodulators,
//! classical STDP helpers, and reward-modulated STDP types.
//!
//! Aimed at **SNN / neuroscience readers learning Rust**: equations and
//! engine contracts first; idiomatic APIs second.
//!
//! ## Requirements
//!
//! - **Rust 1.97.1+** (MSRV; also `rust-version` in `Cargo.toml` and
//!   [`rust-toolchain.toml`](https://github.com/Limen-Neural/neuromod/blob/main/rust-toolchain.toml)).
//! - Edition **2024**.
//! - **CI-tested platforms:** Linux, macOS, and Windows (GitHub Actions matrix).
//!
//! ## Syllabus (reading order on docs.rs)
//!
//! 1. This page — engine vs standalone honesty and a quick start.
//! 2. [`engine`] — [`SpikingNetwork`] and the per-tick [`SpikingNetwork::step`] contract.
//! 3. [`lif`] / [`izhikevich`] — the two banks the engine actually wires.
//! 4. [`modulators`] — dopamine / serotonin / acetylcholine / norepinephrine.
//! 5. [`rm_stdp`] / [`hebbian`] — plasticity: reward-modulated eligibility traces
//!    (wired into the engine) and classical Hebbian STDP (standalone).
//! 6. Standalone models ([`lapicque`], [`gif`], [`fitzhugh_nagumo`], [`hodgkin_huxley`])
//!    for research use outside the engine.
//!
//! ## Engine vs standalone models
//!
//! - **`SpikingNetwork`** wires **LIF** and **Izhikevich** neuron banks only
//!   (`with_dimensions(num_lif, num_izh, num_channels)`).
//! - **Standalone** types (`LapicqueNeuron`, `GifNeuron`, `FitzHughNagumoNeuron`,
//!   `HodgkinHuxleyNeuron`, …) are usable on their own; they are not alternate
//!   engine banks.
//! - Plasticity: `SpikingNetwork` learns through `EligibilityTrace` / `RmStdpConfig`.
//!   Each step decays and accumulates one trace per synapse regardless of dopamine;
//!   dopamine gates only the trace → weight conversion, so reward can arrive after
//!   the coincidence it pays for. Classical Hebbian STDP utilities are separate and
//!   unmodulated.
//!
//! ## Features
//!
//! - Topology-neutral, dynamically sized `SpikingNetwork`
//! - Neuromodulators: dopamine, serotonin, acetylcholine, norepinephrine
//! - Reward-modulated STDP over per-synapse eligibility traces
//!
//! ```rust
//! use neuromod::{NeuroModulators, SpikingNetwork};
//!
//! let mut network = SpikingNetwork::new();
//! let stimuli = [0.5f32; 16];
//! let modulators = NeuroModulators::default();
//! let output = network.step(&stimuli, &modulators).unwrap();
//! println!("Neurons that fired: {output:?}");
//!
//! // Or build dynamically for larger architectures.
//! let mut large = SpikingNetwork::with_dimensions(518, 5, 518);
//! let large_input = vec![0.25f32; 518];
//! let _ = large.step(&large_input, &modulators).unwrap();
//! ```
pub mod engine;
pub mod fitzhugh_nagumo;
pub mod gif;
pub mod hebbian;
pub mod hodgkin_huxley;
pub mod izhikevich;
pub mod lapicque;
pub mod lif;
pub mod modulators;
pub mod rm_stdp;

pub use engine::{SpikingNetwork, StepError};
pub use fitzhugh_nagumo::FitzHughNagumoNeuron;
pub use gif::GifNeuron;
pub use hebbian::{HebbianIzhikevichNetwork, StdpParams, apply_classical_stdp};
pub use hodgkin_huxley::HodgkinHuxleyNeuron;
pub use izhikevich::IzhikevichNeuron;
pub use lapicque::LapicqueNeuron;
pub use lif::LifNeuron;
pub use modulators::{
    GenericReward, NeuroModulators, Observation, SignalProfile, UnitReward, apply_neuromodulation,
};
pub use rm_stdp::{EligibilityTrace, RmStdpConfig};

/// Number of input channels supported by default.
pub const NUM_INPUT_CHANNELS: usize = 16;
