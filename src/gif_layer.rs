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
//!   single-neuron model through [`GifParams`], so the two representations
//!   compute identical values.
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
//! single shared [`GifParams`] for the whole bank rather than one threshold per
//! neuron, so there is no slice to hand it.
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

use serde::{Deserialize, Serialize};

use crate::gif::GifParams;

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
const MAX_INPUTS: usize = u32::MAX as usize;

/// Deterministic SplitMix64 generator.
///
/// Used instead of a `rand` RNG so that generated topology is reproducible
/// across `rand` releases and across platforms: the whole state transition is
/// wrapping integer arithmetic with fixed constants.
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Sub-stream for one neuron. Mixing the index into the seed (rather than
    /// consuming a shared stream) keeps each neuron's topology independent of
    /// how many neurons precede it, so growing a layer does not reshuffle it.
    fn for_neuron(seed: u64, neuron: usize) -> Self {
        Self::new(seed ^ (neuron as u64).wrapping_add(1).wrapping_mul(Self::GAMMA))
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, bound)`, rejection-sampled so the distribution is
    /// unbiased (a bare `%` would over-weight small values).
    fn next_bounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0, "next_bounded requires a positive bound");
        let zone = (u64::MAX / bound) * bound;
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % bound;
            }
        }
    }

    /// Uniform `f32` in `[0, 1)` with 24 bits of mantissa — the full precision
    /// an `f32` can represent in that interval, and exactly representable, so
    /// the value does not depend on rounding mode.
    fn next_unit(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / (1u32 << 24) as f32;
        ((self.next_u64() >> 40) as f32) * SCALE
    }
}

/// Errors produced when building or driving a [`SparseGifHiddenLayer`].
///
/// `PartialEq` only — [`GifLayerError::InvalidWeightRange`] carries `f32`
/// bounds, which may be `NaN` (that is one of the ways a range becomes
/// invalid), so the type cannot honestly be `Eq`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GifLayerError {
    /// `fan_in` exceeded the number of available input channels.
    FanInExceedsInputs {
        /// Requested fan-in.
        fan_in: usize,
        /// Channels actually available.
        num_inputs: usize,
    },
    /// The initial weight range was reversed or non-finite.
    InvalidWeightRange {
        /// Requested lower bound.
        min: f32,
        /// Requested upper bound.
        max: f32,
    },
    /// A stimulus frame did not match the layer's input width.
    InputLenMismatch {
        /// Channels the layer expects.
        expected: usize,
        /// Channels supplied.
        got: usize,
    },
    /// The caller-owned spike buffer did not match the layer's neuron count.
    ///
    /// Distinct from [`Self::InputLenMismatch`] so the message names neurons
    /// rather than input channels — the two widths are unrelated, and reporting
    /// a neuron count as a channel count sends readers to the wrong end of the
    /// call.
    OutputLenMismatch {
        /// Neurons the layer expects to write.
        expected: usize,
        /// Buffer length supplied.
        got: usize,
    },
    /// An explicit topology referenced an input channel that does not exist.
    SourceOutOfRange {
        /// Offending neuron index.
        neuron: usize,
        /// Offending source channel.
        source: usize,
        /// Channels actually available.
        num_inputs: usize,
    },
    /// More input channels were requested than a CSR source index can address.
    ///
    /// Sources are stored as `u32` to keep the topology compact, so the channel
    /// count is capped at [`u32::MAX`]. Without this guard a larger count would
    /// truncate silently — channel `2^32` would alias to channel `0`.
    TooManyInputs {
        /// Channels requested.
        num_inputs: usize,
        /// Largest addressable channel count.
        max: usize,
    },
    /// A deserialized layer failed its internal consistency checks.
    ///
    /// The derived `Deserialize` cannot enforce the CSR/SoA length invariants,
    /// so they are validated on the way in: a checkpoint that violates them
    /// would otherwise panic later while indexing during
    /// [`SparseGifHiddenLayer::step`].
    MalformedCheckpoint {
        /// Which invariant was violated.
        detail: &'static str,
    },
}

impl core::fmt::Display for GifLayerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FanInExceedsInputs { fan_in, num_inputs } => write!(
                f,
                "fan_in {fan_in} exceeds the {num_inputs} available input channels"
            ),
            Self::InvalidWeightRange { min, max } => {
                write!(
                    f,
                    "invalid weight range ({min}, {max}): expected finite min <= max"
                )
            }
            Self::InputLenMismatch { expected, got } => {
                write!(f, "expected {expected} input channels, got {got}")
            }
            Self::OutputLenMismatch { expected, got } => {
                write!(
                    f,
                    "expected a spike buffer of {expected} neurons, got {got}"
                )
            }
            Self::SourceOutOfRange {
                neuron,
                source,
                num_inputs,
            } => write!(
                f,
                "neuron {neuron} references input channel {source}, but only {num_inputs} exist"
            ),
            Self::TooManyInputs { num_inputs, max } => write!(
                f,
                "{num_inputs} input channels exceeds the addressable maximum of {max}"
            ),
            Self::MalformedCheckpoint { detail } => {
                write!(f, "malformed serialized layer: {detail}")
            }
        }
    }
}

impl core::error::Error for GifLayerError {}

/// Construction parameters for a [`SparseGifHiddenLayer`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseGifLayerConfig {
    /// Number of input channels the layer reads.
    pub num_inputs: usize,
    /// Number of GIF neurons in the layer.
    pub num_neurons: usize,
    /// Synapses per neuron. Must be `<= num_inputs`. `0` builds a
    /// topology-free layer that only ever sees zero drive.
    pub fan_in: usize,
    /// Seed for the deterministic topology and weight generator.
    pub seed: u64,
    /// Inclusive-exclusive range for the initial uniform synaptic weights.
    pub weight_range: (f32, f32),
    /// Shared GIF dynamics for every neuron in the bank.
    pub params: GifParams,
}

impl Default for SparseGifLayerConfig {
    fn default() -> Self {
        Self {
            num_inputs: crate::NUM_INPUT_CHANNELS,
            num_neurons: crate::NUM_INPUT_CHANNELS,
            fan_in: GIF_LAYER_DEFAULT_FAN_IN,
            seed: 0,
            weight_range: (GIF_LAYER_DEFAULT_W_MIN, GIF_LAYER_DEFAULT_W_MAX),
            params: GifParams::default(),
        }
    }
}

/// Spike output of a batched [`SparseGifHiddenLayer::run`].
///
/// Stored as one flat row-major `bool` buffer (`step * num_neurons + neuron`)
/// rather than a `Vec<Vec<bool>>`, matching the layer's structure-of-arrays
/// storage and keeping a whole raster in one allocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpikeRaster {
    num_steps: usize,
    num_neurons: usize,
    spikes: Vec<bool>,
}

impl SpikeRaster {
    /// Number of time steps recorded.
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Number of neurons per step.
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Spike flags for one time step, or `None` if `step` is out of range.
    pub fn step(&self, step: usize) -> Option<&[bool]> {
        if step >= self.num_steps {
            return None;
        }
        let lo = step * self.num_neurons;
        Some(&self.spikes[lo..lo + self.num_neurons])
    }

    /// Indices of the neurons that fired at `step`.
    pub fn fired_at(&self, step: usize) -> Vec<usize> {
        self.step(step)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(i, &fired)| fired.then_some(i))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total spikes across the whole raster.
    pub fn total_spikes(&self) -> usize {
        self.spikes.iter().filter(|&&s| s).count()
    }

    /// Spike count per neuron over the whole raster.
    pub fn per_neuron_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.num_neurons];
        for row in 0..self.num_steps {
            let lo = row * self.num_neurons;
            for (neuron, count) in counts.iter_mut().enumerate() {
                if self.spikes[lo + neuron] {
                    *count += 1;
                }
            }
        }
        counts
    }

    /// Flat row-major view of the raster (`step * num_neurons + neuron`).
    pub fn as_flat(&self) -> &[bool] {
        &self.spikes
    }
}

/// A bank of GIF neurons with sparse, layer-owned fan-in.
///
/// See the [module documentation](self) for provenance, determinism guarantees,
/// and the neuromodulation decision.
///
/// # Example
///
/// ```rust
/// use neuromod::gif_layer::{SparseGifHiddenLayer, SparseGifLayerConfig};
///
/// let mut layer = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
///     num_inputs: 8,
///     num_neurons: 4,
///     fan_in: 3,
///     seed: 7,
///     ..Default::default()
/// })
/// .unwrap();
///
/// let fired = layer.step(&[1.0; 8]).unwrap();
/// assert!(fired.len() <= 4);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SparseGifHiddenLayerRepr")]
pub struct SparseGifHiddenLayer {
    num_inputs: usize,
    num_neurons: usize,
    seed: u64,
    params: GifParams,

    // --- CSR fan-in topology (len num_neurons + 1 / nnz / nnz) ---
    fan_in_offsets: Vec<usize>,
    fan_in_sources: Vec<u32>,
    weights: Vec<f32>,

    // --- Structure-of-arrays neuron state (len num_neurons each) ---
    membrane: Vec<f32>,
    adaptation: Vec<f32>,
    last_spike_time: Vec<i64>,

    step_count: i64,
}

/// Deserialization mirror of [`SparseGifHiddenLayer`].
///
/// The public type is `#[serde(try_from = ...)]` this struct so every decoded
/// checkpoint passes the CSR/SoA invariant checks in the `TryFrom` impl before
/// it can be observed. Field names and order match the public type exactly, so
/// the wire format is unchanged and existing checkpoints still load.
#[derive(Deserialize)]
#[serde(rename = "SparseGifHiddenLayer")]
struct SparseGifHiddenLayerRepr {
    num_inputs: usize,
    num_neurons: usize,
    seed: u64,
    params: GifParams,
    fan_in_offsets: Vec<usize>,
    fan_in_sources: Vec<u32>,
    weights: Vec<f32>,
    membrane: Vec<f32>,
    adaptation: Vec<f32>,
    last_spike_time: Vec<i64>,
    step_count: i64,
}

impl TryFrom<SparseGifHiddenLayerRepr> for SparseGifHiddenLayer {
    type Error = GifLayerError;

    /// Enforce every invariant `step_into` relies on when indexing.
    ///
    /// Checked in the order a reader would: addressability, then the SoA bank
    /// widths, then the CSR row structure, then the payload lengths the row
    /// offsets imply, and finally that each source names a real channel.
    fn try_from(repr: SparseGifHiddenLayerRepr) -> Result<Self, Self::Error> {
        let malformed = |detail| GifLayerError::MalformedCheckpoint { detail };

        if repr.num_inputs > MAX_INPUTS {
            return Err(GifLayerError::TooManyInputs {
                num_inputs: repr.num_inputs,
                max: MAX_INPUTS,
            });
        }

        let n = repr.num_neurons;
        if repr.membrane.len() != n || repr.adaptation.len() != n {
            return Err(malformed("membrane/adaptation length != num_neurons"));
        }
        if repr.last_spike_time.len() != n {
            return Err(malformed("last_spike_time length != num_neurons"));
        }

        if repr.fan_in_offsets.len() != n + 1 {
            return Err(malformed("fan_in_offsets length != num_neurons + 1"));
        }
        if repr.fan_in_offsets[0] != 0 {
            return Err(malformed("fan_in_offsets does not start at 0"));
        }
        if repr.fan_in_offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(malformed("fan_in_offsets is not non-decreasing"));
        }

        // Indexing `sources[start..end]` is only in bounds if the final offset
        // is exactly the payload length; a shorter payload panics, a longer one
        // silently strands synapses.
        let nnz = repr.fan_in_offsets[n];
        if repr.fan_in_sources.len() != nnz || repr.weights.len() != nnz {
            return Err(malformed(
                "fan_in_sources/weights length != final fan_in_offset",
            ));
        }

        if repr
            .fan_in_sources
            .iter()
            .any(|&s| s as usize >= repr.num_inputs)
        {
            return Err(malformed("a CSR source references a nonexistent channel"));
        }

        Ok(Self {
            num_inputs: repr.num_inputs,
            num_neurons: repr.num_neurons,
            seed: repr.seed,
            params: repr.params,
            fan_in_offsets: repr.fan_in_offsets,
            fan_in_sources: repr.fan_in_sources,
            weights: repr.weights,
            membrane: repr.membrane,
            adaptation: repr.adaptation,
            last_spike_time: repr.last_spike_time,
            step_count: repr.step_count,
        })
    }
}

impl SparseGifHiddenLayer {
    /// Build a layer with deterministically generated sparse fan-in.
    ///
    /// Each neuron draws `fan_in` *distinct* input channels from its own
    /// SplitMix64 sub-stream (a partial Fisher–Yates shuffle), sorts them
    /// ascending for a canonical CSR row, then draws one uniform weight per
    /// synapse in that same order.
    ///
    /// # Errors
    ///
    /// [`GifLayerError::FanInExceedsInputs`] if `fan_in > num_inputs`, and
    /// [`GifLayerError::InvalidWeightRange`] if the weight range is reversed or
    /// non-finite.
    pub fn new(config: &SparseGifLayerConfig) -> Result<Self, GifLayerError> {
        let SparseGifLayerConfig {
            num_inputs,
            num_neurons,
            fan_in,
            seed,
            weight_range: (w_min, w_max),
            params,
        } = *config;

        if num_inputs > MAX_INPUTS {
            return Err(GifLayerError::TooManyInputs {
                num_inputs,
                max: MAX_INPUTS,
            });
        }
        if fan_in > num_inputs {
            return Err(GifLayerError::FanInExceedsInputs { fan_in, num_inputs });
        }
        if !w_min.is_finite() || !w_max.is_finite() || w_min > w_max {
            return Err(GifLayerError::InvalidWeightRange {
                min: w_min,
                max: w_max,
            });
        }

        // Only a capacity hint; saturating keeps a pathological shape from
        // panicking here in debug builds before allocation fails on its own.
        let nnz = num_neurons.saturating_mul(fan_in);
        let mut fan_in_offsets = Vec::with_capacity(num_neurons + 1);
        let mut fan_in_sources = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);

        // Reused across neurons: the candidate pool and the swap journal that
        // restores it in O(fan_in) instead of O(num_inputs) per neuron.
        let mut pool: Vec<u32> = (0..num_inputs as u32).collect();
        let mut journal: Vec<usize> = Vec::with_capacity(fan_in);

        fan_in_offsets.push(0);
        for neuron in 0..num_neurons {
            let mut rng = SplitMix64::for_neuron(seed, neuron);

            journal.clear();
            for k in 0..fan_in {
                let j = k + rng.next_bounded((num_inputs - k) as u64) as usize;
                pool.swap(k, j);
                journal.push(j);
            }

            let row_start = fan_in_sources.len();
            fan_in_sources.extend_from_slice(&pool[..fan_in]);
            fan_in_sources[row_start..].sort_unstable();

            for _ in 0..fan_in {
                // Interpolate in f64: `w_max - w_min` overflows to `inf` for a
                // finite range wider than f32::MAX (e.g. -3e38..3e38), which
                // would install `inf`/`NaN` synapses from inputs that passed
                // the finiteness check above. f64 spans any f32 range exactly,
                // and `next_unit` is an exact 24-bit value, so every range that
                // did work keeps its previous weights bit for bit.
                let unit = f64::from(rng.next_unit());
                let w = f64::from(w_min) + (f64::from(w_max) - f64::from(w_min)) * unit;
                weights.push(w as f32);
            }

            // Undo the partial shuffle so the next neuron starts from the same
            // canonical pool; without this, topology would depend on the draws
            // of every preceding neuron.
            for (k, &j) in journal.iter().enumerate().rev() {
                pool.swap(k, j);
            }

            fan_in_offsets.push(fan_in_sources.len());
        }

        Ok(Self {
            num_inputs,
            num_neurons,
            seed,
            params,
            fan_in_offsets,
            fan_in_sources,
            weights,
            membrane: vec![0.0; num_neurons],
            adaptation: vec![0.0; num_neurons],
            last_spike_time: vec![-1; num_neurons],
            step_count: 0,
        })
    }

    /// Build a layer from an explicit, caller-supplied topology.
    ///
    /// `rows[n]` lists the `(source_channel, weight)` synapses of neuron `n`.
    /// Rows may have different lengths, including zero. Sources are stored in
    /// the order given, so a caller porting an existing topology keeps its
    /// traversal order exactly.
    ///
    /// # Errors
    ///
    /// [`GifLayerError::SourceOutOfRange`] if any source is `>= num_inputs`,
    /// and [`GifLayerError::TooManyInputs`] if `num_inputs` exceeds what a
    /// `u32` CSR source can address.
    pub fn from_topology(
        num_inputs: usize,
        params: GifParams,
        rows: &[Vec<(usize, f32)>],
    ) -> Result<Self, GifLayerError> {
        // Checked before the per-source bound test below: that test compares
        // against `num_inputs` as a `usize`, so without this guard a source
        // above u32::MAX would pass it and then wrap on the `as u32` cast,
        // silently aliasing to a low channel.
        if num_inputs > MAX_INPUTS {
            return Err(GifLayerError::TooManyInputs {
                num_inputs,
                max: MAX_INPUTS,
            });
        }

        let num_neurons = rows.len();
        let nnz: usize = rows.iter().map(Vec::len).sum();
        let mut fan_in_offsets = Vec::with_capacity(num_neurons + 1);
        let mut fan_in_sources = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);

        fan_in_offsets.push(0);
        for (neuron, row) in rows.iter().enumerate() {
            for &(source, weight) in row {
                if source >= num_inputs {
                    return Err(GifLayerError::SourceOutOfRange {
                        neuron,
                        source,
                        num_inputs,
                    });
                }
                fan_in_sources.push(source as u32);
                weights.push(weight);
            }
            fan_in_offsets.push(fan_in_sources.len());
        }

        Ok(Self {
            num_inputs,
            num_neurons,
            seed: 0,
            params,
            fan_in_offsets,
            fan_in_sources,
            weights,
            membrane: vec![0.0; num_neurons],
            adaptation: vec![0.0; num_neurons],
            last_spike_time: vec![-1; num_neurons],
            step_count: 0,
        })
    }

    /// Number of input channels the layer reads.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Number of neurons in the bank.
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Total number of synapses across the layer.
    pub fn num_synapses(&self) -> usize {
        self.fan_in_sources.len()
    }

    /// Seed the topology was generated from (`0` for [`Self::from_topology`]).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Steps executed since construction or the last [`Self::reset`].
    pub fn step_count(&self) -> i64 {
        self.step_count
    }

    /// Shared GIF dynamics parameters.
    pub fn params(&self) -> &GifParams {
        &self.params
    }

    /// Mutable access to the shared dynamics — the hook for external threshold
    /// or leak modulation.
    pub fn params_mut(&mut self) -> &mut GifParams {
        &mut self.params
    }

    /// Fan-in of one neuron as `(sources, weights)`, or `None` if out of range.
    pub fn fan_in_of(&self, neuron: usize) -> Option<(&[u32], &[f32])> {
        if neuron >= self.num_neurons {
            return None;
        }
        let lo = self.fan_in_offsets[neuron];
        let hi = self.fan_in_offsets[neuron + 1];
        Some((&self.fan_in_sources[lo..hi], &self.weights[lo..hi]))
    }

    /// All synaptic weights in CSR order.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Mutable weights in CSR order — the hook for external plasticity.
    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Membrane potentials, indexed by neuron.
    pub fn membrane(&self) -> &[f32] {
        &self.membrane
    }

    /// Adaptation variables, indexed by neuron.
    pub fn adaptation(&self) -> &[f32] {
        &self.adaptation
    }

    /// Step index of each neuron's most recent spike (`-1` = never fired).
    pub fn last_spike_time(&self) -> &[i64] {
        &self.last_spike_time
    }

    /// Clear dynamic state (membrane, adaptation, spike history, step counter)
    /// while keeping topology and weights intact.
    pub fn reset(&mut self) {
        self.membrane.fill(0.0);
        self.adaptation.fill(0.0);
        self.last_spike_time.fill(-1);
        self.step_count = 0;
    }

    /// Advance one time step and return the indices of the neurons that fired.
    ///
    /// # Errors
    ///
    /// [`GifLayerError::InputLenMismatch`] if `stimuli.len() != num_inputs()`.
    pub fn step(&mut self, stimuli: &[f32]) -> Result<Vec<usize>, GifLayerError> {
        let mut spikes = vec![false; self.num_neurons];
        self.step_into(stimuli, &mut spikes)?;
        Ok(spikes
            .iter()
            .enumerate()
            .filter_map(|(i, &fired)| fired.then_some(i))
            .collect())
    }

    /// Advance one time step, writing spike flags into a caller-owned buffer.
    ///
    /// This is the allocation-free entry point used by [`Self::run`].
    ///
    /// # Errors
    ///
    /// [`GifLayerError::InputLenMismatch`] if `stimuli.len() != num_inputs()`,
    /// or [`GifLayerError::OutputLenMismatch`] if `spikes.len() != num_neurons()`.
    pub fn step_into(&mut self, stimuli: &[f32], spikes: &mut [bool]) -> Result<(), GifLayerError> {
        if stimuli.len() != self.num_inputs {
            return Err(GifLayerError::InputLenMismatch {
                expected: self.num_inputs,
                got: stimuli.len(),
            });
        }
        if spikes.len() != self.num_neurons {
            return Err(GifLayerError::OutputLenMismatch {
                expected: self.num_neurons,
                got: spikes.len(),
            });
        }

        // Copied out of `self` so the shared parameter block can be read while
        // the state arrays are mutably borrowed.
        let params = self.params;
        let now = self.step_count;

        for (neuron, spike) in spikes.iter_mut().enumerate() {
            let lo = self.fan_in_offsets[neuron];
            let hi = self.fan_in_offsets[neuron + 1];
            // Sequential, fixed-order accumulation: floating-point addition is
            // not associative, so the traversal order is part of the contract.
            let drive: f32 = self.fan_in_sources[lo..hi]
                .iter()
                .zip(&self.weights[lo..hi])
                .map(|(&source, &weight)| weight * stimuli[source as usize])
                .sum();

            params.integrate(
                &mut self.membrane[neuron],
                &mut self.adaptation[neuron],
                drive,
            );
            let fired =
                params.check_for_spike(&mut self.membrane[neuron], &mut self.adaptation[neuron]);
            *spike = fired;
            if fired {
                self.last_spike_time[neuron] = now;
            }
        }

        self.step_count += 1;
        Ok(())
    }

    /// Run a batch of spike-train frames and collect the output raster.
    ///
    /// `spike_train[t]` is one frame of `num_inputs()` channel activations.
    /// State carries across frames and across successive `run` calls; call
    /// [`Self::reset`] between independent trials.
    ///
    /// Generic over `AsRef<[f32]>` so both `&[Vec<f32>]` and `&[&[f32]]` work.
    ///
    /// # Errors
    ///
    /// [`GifLayerError::InputLenMismatch`] on the first frame whose width does
    /// not match. Frames before it have already been applied — `run` is not
    /// transactional; validate up front or `reset` after an error.
    pub fn run<S: AsRef<[f32]>>(
        &mut self,
        spike_train: &[S],
    ) -> Result<SpikeRaster, GifLayerError> {
        let num_steps = spike_train.len();
        let mut spikes = vec![false; num_steps * self.num_neurons];
        let mut frame_out = vec![false; self.num_neurons];

        for (t, frame) in spike_train.iter().enumerate() {
            self.step_into(frame.as_ref(), &mut frame_out)?;
            let lo = t * self.num_neurons;
            spikes[lo..lo + self.num_neurons].copy_from_slice(&frame_out);
        }

        Ok(SpikeRaster {
            num_steps,
            num_neurons: self.num_neurons,
            spikes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gif::GifNeuron;

    fn config(
        num_inputs: usize,
        num_neurons: usize,
        fan_in: usize,
        seed: u64,
    ) -> SparseGifLayerConfig {
        SparseGifLayerConfig {
            num_inputs,
            num_neurons,
            fan_in,
            seed,
            ..Default::default()
        }
    }

    fn ramp_train(num_steps: usize, num_inputs: usize) -> Vec<Vec<f32>> {
        (0..num_steps)
            .map(|t| {
                (0..num_inputs)
                    .map(|c| ((t + c) % 5) as f32 * 0.25)
                    .collect()
            })
            .collect()
    }

    // --- structure -------------------------------------------------------

    #[test]
    fn csr_shape_is_consistent() {
        let layer = SparseGifHiddenLayer::new(&config(32, 8, 4, 1)).unwrap();
        assert_eq!(layer.num_synapses(), 32);
        assert_eq!(layer.weights().len(), layer.num_synapses());
        assert_eq!(layer.membrane().len(), 8);
        assert_eq!(layer.adaptation().len(), 8);
        assert_eq!(layer.last_spike_time().len(), 8);
        for n in 0..8 {
            let (sources, weights) = layer.fan_in_of(n).unwrap();
            assert_eq!(sources.len(), 4);
            assert_eq!(weights.len(), 4);
        }
        assert!(layer.fan_in_of(8).is_none());
    }

    #[test]
    fn fan_in_sources_are_distinct_sorted_and_in_range() {
        let layer = SparseGifHiddenLayer::new(&config(24, 16, 6, 0xDEAD_BEEF)).unwrap();
        for n in 0..layer.num_neurons() {
            let (sources, _) = layer.fan_in_of(n).unwrap();
            assert!(
                sources.windows(2).all(|w| w[0] < w[1]),
                "row {n} not strictly ascending: {sources:?}"
            );
            assert!(sources.iter().all(|&s| (s as usize) < 24));
        }
    }

    #[test]
    fn generated_weights_lie_in_range() {
        let mut cfg = config(16, 16, 8, 5);
        cfg.weight_range = (-0.25, 0.75);
        let layer = SparseGifHiddenLayer::new(&cfg).unwrap();
        assert!(layer.weights().iter().all(|&w| (-0.25..0.75).contains(&w)));
    }

    // --- determinism -----------------------------------------------------

    #[test]
    fn same_seed_same_layer() {
        let a = SparseGifHiddenLayer::new(&config(64, 32, 8, 42)).unwrap();
        let b = SparseGifHiddenLayer::new(&config(64, 32, 8, 42)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_different_topology() {
        let a = SparseGifHiddenLayer::new(&config(64, 32, 8, 42)).unwrap();
        let b = SparseGifHiddenLayer::new(&config(64, 32, 8, 43)).unwrap();
        assert_ne!(a, b, "distinct seeds must not collapse to the same layer");
    }

    #[test]
    fn topology_prefix_is_stable_when_the_layer_grows() {
        // Per-neuron sub-streams mean neuron n's fan-in must not depend on how
        // many neurons follow it.
        let small = SparseGifHiddenLayer::new(&config(48, 4, 5, 9)).unwrap();
        let large = SparseGifHiddenLayer::new(&config(48, 40, 5, 9)).unwrap();
        for n in 0..small.num_neurons() {
            assert_eq!(small.fan_in_of(n), large.fan_in_of(n), "row {n} drifted");
        }
    }

    #[test]
    fn run_is_reproducible() {
        let train = ramp_train(40, 32);
        let mut a = SparseGifHiddenLayer::new(&config(32, 12, 6, 7)).unwrap();
        let mut b = SparseGifHiddenLayer::new(&config(32, 12, 6, 7)).unwrap();
        assert_eq!(a.run(&train).unwrap(), b.run(&train).unwrap());
        // ... and a reset layer reproduces its own first pass.
        let first = {
            let mut c = SparseGifHiddenLayer::new(&config(32, 12, 6, 7)).unwrap();
            c.run(&train).unwrap()
        };
        a.reset();
        assert_eq!(a.run(&train).unwrap(), first);
    }

    #[test]
    fn run_matches_stepwise_execution() {
        let train = ramp_train(25, 20);
        let mut batched = SparseGifHiddenLayer::new(&config(20, 10, 4, 3)).unwrap();
        let mut stepped = SparseGifHiddenLayer::new(&config(20, 10, 4, 3)).unwrap();

        let raster = batched.run(&train).unwrap();
        for (t, frame) in train.iter().enumerate() {
            assert_eq!(stepped.step(frame).unwrap(), raster.fired_at(t), "step {t}");
        }
        assert_eq!(batched.membrane(), stepped.membrane());
        assert_eq!(batched.adaptation(), stepped.adaptation());
    }

    #[test]
    fn run_accepts_slice_frames() {
        let owned = ramp_train(6, 8);
        let borrowed: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
        let mut a = SparseGifHiddenLayer::new(&config(8, 4, 2, 11)).unwrap();
        let mut b = SparseGifHiddenLayer::new(&config(8, 4, 2, 11)).unwrap();
        assert_eq!(a.run(&owned).unwrap(), b.run(&borrowed).unwrap());
    }

    // --- parity with the single-neuron model -----------------------------

    #[test]
    fn layer_matches_gif_neuron_exactly() {
        // A one-neuron, one-synapse layer must reproduce `GifNeuron` bit for
        // bit — this is what makes `GifParams` the single source of truth.
        let params = GifParams::default();
        let layer_rows = vec![vec![(0usize, 1.0f32)]];
        let mut layer = SparseGifHiddenLayer::from_topology(1, params, &layer_rows).unwrap();
        let mut neuron = GifNeuron::new();

        for t in 0..120i64 {
            let stimulus = if t % 3 == 0 { 0.9 } else { 0.1 };
            let layer_fired = !layer.step(&[stimulus]).unwrap().is_empty();

            neuron.integrate(stimulus);
            let neuron_fired = neuron.check_for_spike(t);

            assert_eq!(layer_fired, neuron_fired, "spike mismatch at t={t}");
            assert_eq!(layer.membrane()[0], neuron.membrane_potential, "v at t={t}");
            assert_eq!(layer.adaptation()[0], neuron.adaptation, "w at t={t}");
            assert_eq!(layer.last_spike_time()[0], neuron.last_spike_time);
        }
    }

    // --- edge cases ------------------------------------------------------

    #[test]
    fn zero_fan_in_never_fires() {
        let mut layer = SparseGifHiddenLayer::new(&config(16, 4, 0, 1)).unwrap();
        assert_eq!(layer.num_synapses(), 0);
        let raster = layer.run(&ramp_train(50, 16)).unwrap();
        assert_eq!(raster.total_spikes(), 0);
        assert!(layer.membrane().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn zero_input_never_fires() {
        let mut layer = SparseGifHiddenLayer::new(&config(16, 8, 8, 2)).unwrap();
        let train = vec![vec![0.0f32; 16]; 60];
        assert_eq!(layer.run(&train).unwrap().total_spikes(), 0);
    }

    #[test]
    fn fully_dense_fan_in_selects_every_channel() {
        let layer = SparseGifHiddenLayer::new(&config(12, 6, 12, 4)).unwrap();
        for n in 0..6 {
            let (sources, _) = layer.fan_in_of(n).unwrap();
            let expected: Vec<u32> = (0..12).collect();
            assert_eq!(sources, expected.as_slice(), "dense row {n}");
        }
    }

    #[test]
    fn single_neuron_single_input() {
        let mut layer = SparseGifHiddenLayer::new(&config(1, 1, 1, 8)).unwrap();
        assert_eq!(layer.fan_in_of(0).unwrap().0, &[0]);
        let raster = layer.run(&vec![vec![5.0f32]; 10]).unwrap();
        assert_eq!(raster.num_neurons(), 1);
        assert!(raster.total_spikes() > 0, "strong drive should fire");
    }

    #[test]
    fn empty_layer_and_empty_train_are_benign() {
        let mut layer = SparseGifHiddenLayer::new(&config(8, 0, 0, 1)).unwrap();
        let raster = layer.run(&ramp_train(5, 8)).unwrap();
        assert_eq!(raster.num_neurons(), 0);
        assert_eq!(raster.total_spikes(), 0);

        let mut normal = SparseGifHiddenLayer::new(&config(8, 3, 2, 1)).unwrap();
        let empty: Vec<Vec<f32>> = Vec::new();
        let raster = normal.run(&empty).unwrap();
        assert_eq!(raster.num_steps(), 0);
        assert_eq!(normal.step_count(), 0);
    }

    #[test]
    fn reset_clears_state_but_not_topology() {
        let mut layer = SparseGifHiddenLayer::new(&config(16, 5, 4, 6)).unwrap();
        let before = layer.weights().to_vec();
        layer.run(&ramp_train(30, 16)).unwrap();
        layer.reset();
        assert_eq!(layer.step_count(), 0);
        assert!(layer.membrane().iter().all(|&v| v == 0.0));
        assert!(layer.adaptation().iter().all(|&w| w == 0.0));
        assert!(layer.last_spike_time().iter().all(|&t| t == -1));
        assert_eq!(layer.weights(), before.as_slice());
    }

    // --- errors ----------------------------------------------------------

    #[test]
    fn fan_in_larger_than_inputs_is_rejected() {
        assert_eq!(
            SparseGifHiddenLayer::new(&config(4, 2, 5, 0)).unwrap_err(),
            GifLayerError::FanInExceedsInputs {
                fan_in: 5,
                num_inputs: 4
            }
        );
    }

    #[test]
    fn invalid_weight_range_is_rejected() {
        let mut cfg = config(8, 2, 2, 0);
        cfg.weight_range = (1.0, 0.0);
        assert!(matches!(
            SparseGifHiddenLayer::new(&cfg),
            Err(GifLayerError::InvalidWeightRange { .. })
        ));
        cfg.weight_range = (0.0, f32::NAN);
        assert!(matches!(
            SparseGifHiddenLayer::new(&cfg),
            Err(GifLayerError::InvalidWeightRange { .. })
        ));
    }

    #[test]
    fn wrong_frame_width_is_rejected() {
        let mut layer = SparseGifHiddenLayer::new(&config(8, 2, 2, 0)).unwrap();
        assert_eq!(
            layer.step(&[0.0; 7]).unwrap_err(),
            GifLayerError::InputLenMismatch {
                expected: 8,
                got: 7
            }
        );
        assert!(layer.run(&[vec![0.0f32; 3]]).is_err());
    }

    #[test]
    fn out_of_range_explicit_source_is_rejected() {
        let rows = vec![vec![(0usize, 1.0f32)], vec![(9usize, 1.0f32)]];
        assert_eq!(
            SparseGifHiddenLayer::from_topology(4, GifParams::default(), &rows).unwrap_err(),
            GifLayerError::SourceOutOfRange {
                neuron: 1,
                source: 9,
                num_inputs: 4
            }
        );
    }

    #[test]
    fn error_display_is_informative() {
        let msg = GifLayerError::FanInExceedsInputs {
            fan_in: 5,
            num_inputs: 4,
        }
        .to_string();
        assert!(msg.contains('5') && msg.contains('4'));
    }

    // --- serde -----------------------------------------------------------

    #[test]
    fn layer_round_trips_through_json() {
        let mut layer = SparseGifHiddenLayer::new(&config(16, 6, 4, 21)).unwrap();
        layer.run(&ramp_train(15, 16)).unwrap();
        let json = serde_json::to_string(&layer).unwrap();
        let restored: SparseGifHiddenLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(layer, restored);

        // And the restored layer continues the trajectory identically.
        let more = ramp_train(10, 16);
        let mut restored = restored;
        assert_eq!(layer.run(&more).unwrap(), restored.run(&more).unwrap());
    }

    // --- malformed-checkpoint rejection ----------------------------------
    //
    // `Deserialize` is derived over private CSR/SoA vectors whose lengths have
    // to agree, and nothing in the wire format enforces that. Before the
    // `try_from` shim each of these decoded into a layer that panicked on the
    // next `step` while indexing. They must now fail at decode instead.

    /// Serialize a good layer, corrupt one field in the JSON, and decode.
    fn decode_corrupted(
        mutate: impl FnOnce(&mut serde_json::Value),
    ) -> Result<SparseGifHiddenLayer, serde_json::Error> {
        let layer = SparseGifHiddenLayer::new(&config(8, 3, 3, 5)).unwrap();
        let mut v: serde_json::Value = serde_json::to_value(&layer).unwrap();
        mutate(&mut v);
        serde_json::from_value(v)
    }

    #[test]
    fn rejects_checkpoint_with_short_soa_bank() {
        for field in ["membrane", "adaptation", "last_spike_time"] {
            let err = decode_corrupted(|v| {
                v[field].as_array_mut().unwrap().pop();
            })
            .unwrap_err();
            assert!(
                err.to_string().contains("malformed serialized layer"),
                "{field} truncation should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn rejects_checkpoint_with_bad_csr_offsets() {
        // Wrong length.
        assert!(
            decode_corrupted(|v| {
                v["fan_in_offsets"].as_array_mut().unwrap().pop();
            })
            .is_err()
        );

        // Does not start at zero.
        assert!(decode_corrupted(|v| { v["fan_in_offsets"][0] = 1.into() }).is_err());

        // Not non-decreasing — this is the one that would index backwards.
        // Offsets are [0, 3, 6, 9]; 7 > 6 makes row 1 end before it starts.
        // (Lowering an offset instead would still be valid CSR: [0, 0, 6, 9]
        // just describes an empty first row.)
        assert!(decode_corrupted(|v| { v["fan_in_offsets"][1] = 7.into() }).is_err());
    }

    #[test]
    fn rejects_checkpoint_whose_payload_disagrees_with_offsets() {
        for field in ["fan_in_sources", "weights"] {
            let err = decode_corrupted(|v| {
                v[field].as_array_mut().unwrap().pop();
            })
            .unwrap_err();
            assert!(
                err.to_string().contains("malformed serialized layer"),
                "{field} truncation should be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn rejects_checkpoint_sourcing_a_nonexistent_channel() {
        // num_inputs is 8, so channel 99 does not exist.
        let err = decode_corrupted(|v| v["fan_in_sources"][0] = 99.into()).unwrap_err();
        assert!(err.to_string().contains("nonexistent channel"), "{err}");
    }

    #[test]
    fn a_valid_checkpoint_still_decodes() {
        // Guard against the validator being so strict it rejects good input.
        assert!(decode_corrupted(|_| {}).is_ok());
    }

    // --- numeric and addressability guards -------------------------------

    #[test]
    fn wide_but_finite_weight_range_stays_finite() {
        // `w_max - w_min` overflows f32 here (6e38 > f32::MAX), which used to
        // yield `inf` weights — and `inf * 0.0` = `NaN` — from a range that
        // passes the finiteness check. Interpolating in f64 keeps every weight
        // inside the requested bounds.
        let layer = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
            num_inputs: 8,
            num_neurons: 4,
            fan_in: 3,
            seed: 11,
            weight_range: (-3.0e38, 3.0e38),
            ..Default::default()
        })
        .unwrap();

        assert!(
            layer.weights().iter().all(|w| w.is_finite()),
            "non-finite weight from a finite range: {:?}",
            layer.weights()
        );
        assert!(
            layer
                .weights()
                .iter()
                .all(|&w| (-3.0e38..=3.0e38).contains(&w)),
            "weight escaped the requested range"
        );
    }

    #[test]
    fn wrong_sized_spike_buffer_reports_neurons_not_channels() {
        // num_inputs is 8 and num_neurons is 3, so a bad spike buffer must not
        // be described as an input-width problem -- the two widths differ and
        // the old message sent readers to the wrong argument.
        let mut layer = SparseGifHiddenLayer::new(&config(8, 3, 3, 5)).unwrap();
        let mut spikes = [false; 2];
        let err = layer.step_into(&[0.5; 8], &mut spikes).unwrap_err();
        assert_eq!(
            err,
            GifLayerError::OutputLenMismatch {
                expected: 3,
                got: 2
            }
        );
        assert!(err.to_string().contains("spike buffer"), "{err}");
    }

    #[test]
    fn rejects_more_input_channels_than_a_u32_source_can_address() {
        // Empty rows, so this allocates nothing: the guard must fire on the
        // declared width alone. Without it, `source as u32` would wrap and a
        // channel above u32::MAX would alias onto a low one.
        let err = SparseGifHiddenLayer::from_topology(MAX_INPUTS + 1, GifParams::default(), &[])
            .unwrap_err();
        assert_eq!(
            err,
            GifLayerError::TooManyInputs {
                num_inputs: MAX_INPUTS + 1,
                max: MAX_INPUTS,
            }
        );
        assert!(err.to_string().contains("addressable maximum"));
    }

    // --- golden regression fixtures --------------------------------------
    //
    // These are INTERNAL goldens produced by this implementation, not
    // cross-repo parity vectors from `corinth-canal` (see the module docs).
    // They pin the topology generator, the CSR traversal order, and the GIF
    // arithmetic together: any of the three drifting will fail here.

    #[test]
    fn golden_topology_fixture() {
        let layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
        let rows: Vec<Vec<u32>> = (0..4)
            .map(|n| layer.fan_in_of(n).unwrap().0.to_vec())
            .collect();
        assert_eq!(
            rows,
            vec![
                vec![11, 13, 15],
                vec![3, 10, 15],
                vec![6, 10, 11],
                vec![1, 3, 14],
            ]
        );
    }

    #[test]
    fn golden_weight_fixture() {
        let layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
        let expected: [f32; 12] = [
            0.249_479_71,
            0.710_527_5,
            0.974_420_7,
            0.850_541_5,
            0.567_846_83,
            0.039_654_434,
            0.028_471_59,
            0.159_303_13,
            0.461_927_65,
            0.108_223_14,
            0.395_182_2,
            0.187_865_02,
        ];
        for (i, (&got, &want)) in layer.weights().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "weight {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn golden_raster_fixture() {
        let mut layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
        let raster = layer.run(&ramp_train(20, 16)).unwrap();
        assert_eq!(raster.per_neuron_counts(), vec![12, 10, 4, 5]);
        assert_eq!(raster.total_spikes(), 31);
        assert_eq!(raster.fired_at(5), vec![0, 3]);
        assert_eq!(raster.fired_at(19), vec![1, 3]);
    }

    #[test]
    fn golden_state_fixture() {
        let mut layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
        layer.run(&ramp_train(20, 16)).unwrap();

        let membrane: [f32; 4] = [1.865_292_9, 1.185_997_7, 1.030_233_5, 0.742_578];
        let adaptation: [f32; 4] = [6.510_236_3, 5.604_505_5, 2.015_107_6, 2.864_713];
        for (i, (&got, &want)) in layer.membrane().iter().zip(membrane.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "membrane {i}: got {got}, want {want}"
            );
        }
        for (i, (&got, &want)) in layer.adaptation().iter().zip(adaptation.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "adaptation {i}: got {got}, want {want}"
            );
        }
        assert_eq!(layer.step_count(), 20);
    }
}
