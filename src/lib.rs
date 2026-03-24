//! # Neuromod - Reward-Modulated Spiking Neural Networks
//!
//! A lightweight, focused Rust crate for neuromorphic computing with
//! reward-modulated spiking neural networks. Designed for high-frequency
//! trading (HFT) applications and FPGA deployment.
//!
//! ## Provenance
//!
//! Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor
//! repository (closed-source). The LIF/Izhikevich network, STDP, and neuromodulator
//! system ran in production for Dynex/Quai/Qubic mining optimization before being
//! published as a standalone crate on crates.io.
//!
//! ## Features
//!
//! - **LIF Neurons**: Fast, reactive leaky integrate-and-fire neurons
//! - **Izhikevich Neurons**: Complex, adaptive neuron dynamics  
//! - **STDP Learning**: Spike-timing-dependent plasticity with reward modulation
//! - **Neuromodulators**: Dopamine, cortisol, acetylcholine, and tempo control
//! - **FPGA Support**: Hardware acceleration ready
//! - **HFT Optimized**: Built for real-time trading applications
//!
//! ## Quick Start
//!
//! ```rust
//! use neuromod::{SpikingNetwork, NeuroModulators};
//!
//! let mut network = SpikingNetwork::new();
//! let stimuli = [0.5f32; 16]; // 16-channel input
//! let modulators = NeuroModulators::default();
//!
//! // Step the network
//! let spikes = network.step(&stimuli, &modulators);
//! let signal = network.bear_bull_signal();
//! println!("Spikes: {:?}  Net sentiment: {}", spikes, signal.net());
//! ```

pub mod lif;
pub mod izhikevich;
pub mod stdp;
pub mod modulators;
pub mod engine;
pub mod mining;
pub mod traits;

// Re-export main types for convenience
pub use lif::LifNeuron;
pub use izhikevich::IzhikevichNeuron;
pub use modulators::NeuroModulators;
pub use engine::{SpikingNetwork, BearBullSignal};
pub use mining::MiningReward;
pub use traits::HftReward;

/// Number of input channels supported by default
pub const NUM_INPUT_CHANNELS: usize = 16;
