//! # Neuromod - Reward-Modulated Spiking Neural Networks
//!
//! A lightweight, focused Rust crate for neuromorphic computing with
//! reward-modulated spiking neural networks.
//!
//! ## Provenance
//!
//! Extracted from Eagle-Lander, the author's own private neuromorphic supervisor
//! repository (closed-source). The LIF/Izhikevich network, STDP, and neuromodulator
//! system ran in production before being published as a standalone crate on crates.io.
//!
//! ## Features
//!
//! - **LIF Neurons**: Fast, reactive leaky integrate-and-fire neurons
//! - **Izhikevich Neurons**: Complex, adaptive neuron dynamics  
//! - **Reward STDP Learning**: Spike-timing-dependent plasticity with reward modulation
//! - **Neuromodulators**: Dopamine, cortisol, acetylcholine, and tempo control
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
pub mod lif;
pub mod izhikevich;
pub mod rm_stdp;
pub mod modulators;
pub mod engine;
// Godfathers of Neuroscience
pub mod lapicque;
pub mod hebbian;
pub mod hodgkin_huxley;
pub mod fitzhugh_nagumo;
pub mod gif;

// Re-export main types for convenience
pub use lif::LifNeuron;
pub use izhikevich::IzhikevichNeuron;
pub use modulators::NeuroModulators;
pub use engine::{SpikingNetwork, StepError};
pub use rm_stdp::{EligibilityTrace, RmStdpConfig};
pub use lapicque::LapicqueNeuron;
pub use hebbian::{apply_classical_stdp, HebbianIzhikevichNetwork, StdpParams};
pub use hodgkin_huxley::HodgkinHuxleyNeuron;
pub use fitzhugh_nagumo::FitzHughNagumoNeuron;
pub use gif::GifNeuron;

/// Number of input channels supported by default
pub const NUM_INPUT_CHANNELS: usize = 16;
