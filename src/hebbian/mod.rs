//! Classical Hebbian learning module.
//!
//! Honors Donald O. Hebb (1949): "Neurons that fire together wire together."
//! This module implements pure (unmodulated) Spike-Timing-Dependent Plasticity
//! (STDP), the biological root that the rest of this crate's reward-modulated
//! STDP builds upon.

pub mod classical;

pub use classical::{apply_classical_stdp, HebbianIzhikevichNetwork, StdpParams};
