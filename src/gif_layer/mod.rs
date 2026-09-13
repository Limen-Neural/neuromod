//! Sparse Generalized Integrate-and-Fire (GIF) hidden layer.
//!
//! ## Provenance
//!
//! This module is an **upstream port** of the reusable GIF hidden-layer
//! implementation that previously lived in the author's `rmems/corinth-canal`
//! repository (`src/funnel.rs`, `SparseGifHiddenLayer`). It was promoted into
//! `neuromod` under Limen-Neural/neuromod issue **#101** so that the canonical
//! dynamics live in the dynamics crate and downstream consumers can eventually
//! delete their local copy.
//!
//! Only the *layer* is ported. The application-side orchestration that wrapped
//! it upstream (telemetry funnels, checkpoint parsing, encoding schemas, signed
//! split-bank bridging) is deliberately **out of scope** and stays downstream —
//! see [`docs/neuromod-boundary-matrix.md`](https://github.com/Limen-Neural/neuromod/blob/main/docs/neuromod-boundary-matrix.md).
//!
//! ### Parity-fixture caveat
//!
//! The reference implementation was not available while this port was written,
//! so the regression fixtures in this module are **internal goldens**: values
//! produced by this implementation and pinned so that any future change to the
//! dynamics, the topology generator, or the traversal order is caught. They are
//! *not* cross-repo bit-parity fixtures. True parity vectors captured from
//! `corinth-canal` must be added in a follow-up once that repository is
//! available for side-by-side comparison (issue #101 follow-up).
//!
//! ## Design
//!
//! - **Structure-of-arrays state.** Membrane potential, adaptation, and last
//!   spike time live in three parallel `Vec`s indexed by neuron, not in a
//!   `Vec<GifNeuron>`. The dynamics themselves are shared with the
//!   single-neuron model through [`GifParams`](crate::gif::GifParams), so the
//!   two representations compute identical values.
//! - **CSR topology.** Sparse fan-in is stored as a compressed row structure
//!   (`offsets`, `sources`, `weights`) owned by the layer.
//! - **Deterministic generation.** Topology and initial weights come from a
//!   seeded SplitMix64 stream with a per-neuron sub-stream. The same seed and
//!   the same shape always produce the same layer, on any platform, with no
//!   dependence on wall clock, hashing, or iteration order.
//! - **Deterministic execution.** [`SparseGifHiddenLayer::run`] is a plain
//!   sequential fold over time steps; there is no threading and no reduction
//!   whose order could vary, so identical input yields bit-identical output.
//!
//! ## Neuromodulation
//!
//! This layer intentionally does **not** consume [`crate::NeuroModulators`].
//! A GIF hidden layer is a pure integrate-and-fire structure: the ported
//! surface has no reward signal, no eligibility trace, and no dopamine gate,
//! and wiring one in would change the numerics a parity port is supposed to
//! preserve. Callers that want modulation apply it themselves between steps,
//! through [`SparseGifHiddenLayer::weights_mut`] for synaptic strength and
//! [`SparseGifHiddenLayer::params_mut`] for the shared dynamics.
//!
//! [`crate::modulators::apply_neuromodulation`] is deliberately not used here:
//! it takes a per-neuron `&mut [f32]` of thresholds, and this layer keeps a
//! single shared [`GifParams`](crate::gif::GifParams) for the whole bank rather
//! than one threshold per neuron, so there is no slice to hand it.
//!
//! ## Example
//!
//! ```rust
//! use neuromod::gif_layer::{SparseGifHiddenLayer, SparseGifLayerConfig};
//!
//! let config = SparseGifLayerConfig {
//!     num_inputs: 32,
//!     num_neurons: 8,
//!     fan_in: 4,
//!     seed: 0xC0FF_EE01,
//!     ..Default::default()
//! };
//! let mut layer = SparseGifHiddenLayer::new(&config).unwrap();
//!
//! // A 12-step spike train of 32 channels each.
//! let train: Vec<Vec<f32>> = (0..12).map(|t| vec![(t % 2) as f32; 32]).collect();
//! let raster = layer.run(&train).unwrap();
//!
//! assert_eq!(raster.num_steps(), 12);
//! assert_eq!(raster.num_neurons(), 8);
//! ```

mod config;
mod error;
mod layer;
mod raster;
mod rng;

pub use config::SparseGifLayerConfig;
pub use error::GifLayerError;
pub use layer::SparseGifHiddenLayer;
pub use raster::SpikeRaster;

/// Default fan-in used by [`SparseGifLayerConfig::default`].
pub const GIF_LAYER_DEFAULT_FAN_IN: usize = 16;
/// Default lower bound of the initial synaptic weight range.
pub const GIF_LAYER_DEFAULT_W_MIN: f32 = 0.0;
/// Default upper bound of the initial synaptic weight range.
pub const GIF_LAYER_DEFAULT_W_MAX: f32 = 1.0;

/// Largest addressable input-channel count.
///
/// CSR sources are stored as `u32` to halve the topology's footprint, so a
/// channel index has to fit in one. Construction rejects anything wider rather
/// than letting the cast wrap.
pub(crate) const MAX_INPUTS: usize = u32::MAX as usize;
