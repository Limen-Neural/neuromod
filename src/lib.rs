//! # Neuromod - Reward-Modulated Spiking Neural Networks
//!
//! A lightweight Rust crate for neuromorphic computing with reward-modulated
//! spiking neural networks and generic neuromodulator control.
//!
//! ## Features
//!
//! - **LIF Neurons**: Fast, reactive leaky integrate-and-fire neurons
//! - **Izhikevich Neurons**: Complex, adaptive neuron dynamics
//! - **Reward STDP Learning**: Spike-timing-dependent plasticity with reward modulation
//! - **Neuromodulators**: Dopamine, serotonin, acetylcholine, and norepinephrine
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
