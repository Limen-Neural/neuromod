//! # Neuromod - Reward-Modulated Spiking Neural Networks
//!
//! A lightweight, focused Rust crate for neuromorphic computing with
//! reward-modulated spiking neural networks.
//!
//! ## Provenance
//!
//! Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor
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
//! ## For Rust new comers think of lib.rs as like a house.  libs.rs is like a front door - it tells Rust which rooms (modules) are in the house and which furniture (types/functions) to make available to visitors.  
//! ## So when you import the crate, you can access the furniture that lib.rs has chosen to show you.  The modules are like different rooms in the house where the actual work happens.  
//! ## If you want to change how a specific piece of furniture works, you go into the room (module) where it's made and change it there.  
//! ## But if you want to add a new piece of furniture or a new room, you also go into lib.rs and tell it about your new creation so that visitors can see it when they come in.  
//! ## So lib.rs is like the blueprint and directory for the whole crate, while the modules are where the actual code lives and does its thing.
//!
//! Note: I already made my own mining repo so I am deleting any mining related code from this repo.  So the mining reward struct and algorithm will be deleted from this repo.  
//! -- I am keeping it in the mining repo.  So I am deleting the mining module from this repo as well.  So I am deleting the 'mod mining;' line from this file as well.  
//! -- I am also deleting the 'pub use mining::MiningReward;' line from this file as well.  So I am also deleting the 'use mining::MiningReward;' line from engine.rs as well.  
//! -- So I am also deleting any references to 'MiningReward' in engine.rs as well.  So I am also deleting any references to 'mining' in engine.rs as well.  So I am also deleting any references to 'mining' in traits.rs as well.  
//! -- So I am also deleting any references to 'HftReward' in traits.rs as well.  So I am also deleting the 'pub use traits::HftReward;' line from this file as well.
//!
//! ```rust
//! use neuromod::{SpikingNetwork, NeuroModulators};
//!
//! let mut network = SpikingNetwork::new();
//! let stimuli = [0.5f32; 16]; // 16-channel input
//! let modulators = NeuroModulators::default();
//! // Simulate one step of the network with the given stimuli and modulators
//! let output = network.step(&stimuli, &modulators);
//! println!("Neurons that fired: {:?}", output);
//! ```
pub mod lif;
pub mod izhikevich;
pub mod rm_stdp;  // change from 'stdp' to 'rm_stdp' to reflect reward modulation
pub mod modulators;
pub mod engine;
// Deleting the mining module as per the new plan
pub mod traits;

// Re-export main types for convenience
pub use lif::LifNeuron;
pub use izhikevich::IzhikevichNeuron;
pub use modulators::NeuroModulators;
pub use engine::{SpikingNetwork}; // Re-exporting the SpikingNetwork struct for external use
// Deleting the mining reward struct from this repo as well.  So I am deleting the
// Deleting this line as well since we are removing mining related code from this repo
// Deleting HFT trait to make this crate more focused on neuromodulated spiking networks and less on specific applications like HFT
pub use rm_stdp::{EligibilityTrace, RmStdpConfig}; // Re-exporting the RmStdpConfig struct for external use

/// Number of input channels supported by default
pub const NUM_INPUT_CHANNELS: usize = 16;

