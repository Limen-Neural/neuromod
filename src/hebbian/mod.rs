//! Hebbian Learning Module
//!
//! Pure (classical) Hebbian plasticity honoring Donald O. Hebb (1949).
//! This is the biological root that Spikenaut's reward-modulated STDP builds upon.

pub mod classical;

pub use classical::{apply_classical_stdp, HebbianIzhikevichNetwork, StdpParams};

