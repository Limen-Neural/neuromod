use serde::{Deserialize, Serialize};

use crate::gif::GifParams;

use super::numeric::{Transition, first_non_finite, invalid_parameter, validate_params};
use super::rng::SplitMix64;
use super::{GifLayerError, MAX_INPUTS, SparseGifLayerConfig, SpikeRaster};

/// A bank of GIF neurons with sparse, layer-owned fan-in.
///
/// See the [module documentation](super) for provenance, determinism guarantees,
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
#[derive(Debug, PartialEq, Serialize, Deserialize)]
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

    // Empty between calls, including errors, so scratch never affects equality.
    #[serde(skip)]
    transitions: Vec<Transition>,
}

impl Clone for SparseGifHiddenLayer {
    fn clone(&self) -> Self {
        Self {
            num_inputs: self.num_inputs,
            num_neurons: self.num_neurons,
            seed: self.seed,
            params: self.params,
            fan_in_offsets: self.fan_in_offsets.clone(),
            fan_in_sources: self.fan_in_sources.clone(),
            weights: self.weights.clone(),
            membrane: self.membrane.clone(),
            adaptation: self.adaptation.clone(),
            last_spike_time: self.last_spike_time.clone(),
            step_count: self.step_count,
            // Cloning an empty Vec would discard its reserved capacity and
            // make the clone's first step allocate.
            transitions: Vec::with_capacity(self.num_neurons),
        }
    }
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

impl SparseGifHiddenLayerRepr {
    /// Enforce every invariant the layer relies on after decoding.
    ///
    /// Split in two because the checks answer different questions: whether the
    /// arrays *line up* ([`Self::validate_shape`], which is what keeps indexing
    /// in bounds), and whether the values they hold are ones this crate could
    /// have produced ([`Self::validate_values`]).
    fn validate(&self) -> Result<(), GifLayerError> {
        self.validate_shape()?;
        self.validate_values()
    }

    /// Lengths and CSR structure: everything `step_into` indexes through.
    ///
    /// Checked in the order a reader would: addressability, then the SoA bank
    /// widths, then the CSR row structure, then the payload lengths the row
    /// offsets imply.
    fn validate_shape(&self) -> Result<(), GifLayerError> {
        let malformed = |detail| GifLayerError::MalformedCheckpoint { detail };

        if self.num_inputs > MAX_INPUTS {
            return Err(GifLayerError::TooManyInputs {
                num_inputs: self.num_inputs,
                max: MAX_INPUTS,
            });
        }

        let n = self.num_neurons;
        if self.membrane.len() != n || self.adaptation.len() != n {
            return Err(malformed("membrane/adaptation length != num_neurons"));
        }
        if self.last_spike_time.len() != n {
            return Err(malformed("last_spike_time length != num_neurons"));
        }

        if self.fan_in_offsets.len() != n + 1 {
            return Err(malformed("fan_in_offsets length != num_neurons + 1"));
        }
        if self.fan_in_offsets[0] != 0 {
            return Err(malformed("fan_in_offsets does not start at 0"));
        }
        if self.fan_in_offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(malformed("fan_in_offsets is not non-decreasing"));
        }

        // Indexing `sources[start..end]` is only in bounds if the final offset
        // is exactly the payload length; a shorter payload panics, a longer one
        // silently strands synapses.
        let nnz = self.fan_in_offsets[n];
        if self.fan_in_sources.len() != nnz || self.weights.len() != nnz {
            return Err(malformed(
                "fan_in_sources/weights length != final fan_in_offset",
            ));
        }

        Ok(())
    }

    /// Values the layer could actually have produced.
    ///
    /// Runs after [`Self::validate_shape`], so every bank is already known to
    /// be the right length; these checks are about content, not layout.
    fn validate_values(&self) -> Result<(), GifLayerError> {
        let malformed = |detail| GifLayerError::MalformedCheckpoint { detail };

        if self.weights.iter().any(|value| !value.is_finite()) {
            return Err(malformed("weights contains a non-finite value"));
        }
        if self.membrane.iter().any(|value| !value.is_finite()) {
            return Err(malformed("membrane contains a non-finite value"));
        }
        if self.adaptation.iter().any(|value| !value.is_finite()) {
            return Err(malformed("adaptation contains a non-finite value"));
        }

        if let Some((_, detail, _)) = invalid_parameter(&self.params) {
            return Err(malformed(detail));
        }

        if self
            .fan_in_sources
            .iter()
            .any(|&s| s as usize >= self.num_inputs)
        {
            return Err(malformed("a CSR source references a nonexistent channel"));
        }

        // The counter is monotonic from 0, so a negative value never came from
        // this crate and would make `last_spike_time` comparisons meaningless.
        // (Exhaustion at i64::MAX is a separate, recoverable case: `step_into`
        // reports it rather than rejecting the whole checkpoint, so a saturated
        // layer can still be inspected.)
        if self.step_count < 0 {
            return Err(malformed("step_count is negative"));
        }

        // Timestamps are `-1` (never fired) or the step index fired on, which
        // is recorded before the counter advances — so a real one is always
        // below `step_count`. The bank's *length* is checked above, but a
        // value the layer could never have produced still gets through, and
        // `step_count - last_spike_time` over a bogus entry can overflow.
        if self
            .last_spike_time
            .iter()
            .any(|&t| t < -1 || t >= self.step_count)
        {
            return Err(malformed(
                "a last_spike_time is below -1 or not strictly in the past",
            ));
        }

        Ok(())
    }
}

impl TryFrom<SparseGifHiddenLayerRepr> for SparseGifHiddenLayer {
    type Error = GifLayerError;

    /// Validate the decoded fields, then move them into the layer.
    ///
    /// The checks themselves live in [`SparseGifHiddenLayerRepr::validate`].
    fn try_from(repr: SparseGifHiddenLayerRepr) -> Result<Self, Self::Error> {
        repr.validate()?;

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
            transitions: Vec::with_capacity(repr.num_neurons),
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
    /// non-finite. [`GifLayerError::NonFiniteParam`] identifies the first
    /// non-finite shared parameter. Finite signed parameters remain supported.
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

        validate_params(&params)?;

        let (fan_in_offsets, fan_in_sources, weights) =
            Self::generate_topology(num_inputs, num_neurons, fan_in, seed, w_min, w_max);

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
            transitions: Vec::with_capacity(num_neurons),
        })
    }

    /// Draw the CSR fan-in topology and initial weights for [`Self::new`].
    ///
    /// Returns `(offsets, sources, weights)`. Split out of `new` so the
    /// constructor reads as validate-then-build; all the determinism-critical
    /// mechanics live here.
    ///
    /// Callers must have validated `fan_in <= num_inputs`, `num_inputs <=
    /// MAX_INPUTS`, and a finite ordered weight range — this drives the
    /// generator directly and does no checking of its own.
    fn generate_topology(
        num_inputs: usize,
        num_neurons: usize,
        fan_in: usize,
        seed: u64,
        w_min: f32,
        w_max: f32,
    ) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
        // Only a capacity hint; saturating keeps a pathological shape from
        // panicking here in debug builds before allocation fails on its own.
        let nnz = num_neurons.saturating_mul(fan_in);
        let mut fan_in_offsets = Vec::with_capacity(num_neurons + 1);
        let mut fan_in_sources = Vec::with_capacity(nnz);
        let mut weights = Vec::with_capacity(nnz);

        // Reused across neurons: the candidate pool and the swap journal that
        // restores it in O(fan_in) instead of O(num_inputs) per neuron.
        //
        // A zero fan-in layer never samples the pool, so skip building it:
        // otherwise a topology-free layer still allocates one `u32` per input
        // channel, which is pure waste at a wide input width.
        let mut pool: Vec<u32> = if fan_in == 0 {
            Vec::new()
        } else {
            (0..num_inputs as u32).collect()
        };
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
                weights.push(Self::draw_weight(&mut rng, w_min, w_max));
            }

            // Undo the partial shuffle so the next neuron starts from the same
            // canonical pool; without this, topology would depend on the draws
            // of every preceding neuron.
            for (k, &j) in journal.iter().enumerate().rev() {
                pool.swap(k, j);
            }

            fan_in_offsets.push(fan_in_sources.len());
        }

        (fan_in_offsets, fan_in_sources, weights)
    }

    /// Draw one synaptic weight uniformly from the half-open `[w_min, w_max)`.
    ///
    /// Consumes exactly one value from `rng` on every path, so the seeded
    /// stream does not depend on which branch is taken.
    ///
    /// Two corrections sit on top of the plain interpolation, both narrow:
    ///
    /// - `w_max - w_min` overflows to `inf` for a finite range wider than
    ///   `f32::MAX` (e.g. `-3e38..3e38`), and `inf * 0.0` is `NaN`, so that
    ///   range would install non-finite synapses despite passing the
    ///   finiteness check in [`Self::new`]. Only that case falls back to f64.
    ///   The fallback is deliberately *not* applied to ordinary ranges: f64
    ///   rounds once where the f32 expression rounds three times, so the two
    ///   disagree by 1 ULP on roughly 30% of random ranges, and routing
    ///   everything through f64 would silently reseed existing layers. (The
    ///   default `0.0..1.0` agrees exactly — span is `1.0`, offset `0.0`, so
    ///   all three f32 roundings are exact — which is why the golden fixtures
    ///   alone do not catch the difference.)
    /// - `unit` is always below 1, but rounding can still carry the result up
    ///   to exactly `w_max`, breaking the documented half-open range. That
    ///   needs bounds only a few ULPs apart (with adjacent floats any unit
    ///   above 0.5 rounds up), so stepping back one representable value costs
    ///   nothing real: a 2M-draw sweep over random ranges produced zero
    ///   results at or above `w_max`. Skipped for a degenerate range, where
    ///   the half-open interval is empty and `w_min` is the only answer.
    fn draw_weight(rng: &mut SplitMix64, w_min: f32, w_max: f32) -> f32 {
        let unit = rng.next_unit();
        let span = w_max - w_min;

        let w = if span.is_finite() {
            w_min + span * unit
        } else {
            (f64::from(w_min) + (f64::from(w_max) - f64::from(w_min)) * f64::from(unit)) as f32
        };

        if w >= w_max && w_min < w_max {
            w_max.next_down()
        } else {
            w
        }
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
    /// `u32` CSR source can address. [`GifLayerError::NonFiniteParam`] rejects
    /// non-finite dynamics, and [`GifLayerError::NonFiniteWeight`] identifies
    /// a non-finite weight by its flat CSR index (rows concatenated in order).
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

        validate_params(&params)?;

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
                if let Some(class) = crate::NonFiniteClass::classify(weight) {
                    return Err(GifLayerError::NonFiniteWeight {
                        index: weights.len(),
                        class,
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
            transitions: Vec::with_capacity(num_neurons),
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
    /// or leak modulation. All fields must be finite before the next step;
    /// invalid edits are rejected by stepping, not silently repaired.
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
    ///
    /// Values must be finite before the next step. A rejected step preserves
    /// invalid edits for inspection; the caller must correct them before retry.
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
    /// This convenience API allocates a spike buffer and the returned indices.
    /// Prefer [`Self::step_into`] with a reused output buffer in hot loops.
    ///
    /// # Errors
    ///
    /// The length, numeric, and counter errors documented by [`Self::step_into`].
    /// Every rejection leaves the layer unchanged.
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
    /// Preferred hot-loop entry point: reuse the output buffer for allocation-free
    /// stepping, including the first call after construction, cloning, or decoding.
    /// Each candidate transition is computed and validated once in fixed arithmetic
    /// order, then committed only after every neuron passes. Scratch storage is
    /// reserved outside stepping and omitted from checkpoints. Complexity remains
    /// O(inputs + synapses + neurons), with O(neurons) extra scratch space.
    ///
    /// # Errors
    ///
    /// Validation order is input length, output length, counter exhaustion,
    /// parameters (GifParams declaration order), all input channels, then
    /// candidate transitions in neuron order. A non-finite drive first reports
    /// an invalid weight in that row, if any, before arithmetic overflow. Non-finite
    /// parameters, weights, and stimuli produce [`GifLayerError::NonFiniteParam`],
    /// [`GifLayerError::NonFiniteWeight`], and [`GifLayerError::NonFiniteInput`].
    /// Finite values are not clamped or restricted to physiological ranges.
    ///
    /// [`GifLayerError::NumericOverflow`] rejects any non-finite drive, decayed
    /// adaptation, integrated membrane, effective threshold, or post-spike state.
    /// This includes overflow from finite operands. **Every error preserves all
    /// layer fields and the entire caller output buffer**, even if a later neuron
    /// fails. Correct invalid inputs/edits before retrying; no repair is implicit.
    ///
    /// ```rust
    /// use neuromod::{GifLayerError, GifParams, NonFiniteClass, SparseGifHiddenLayer};
    /// let mut layer = SparseGifHiddenLayer::from_topology(
    ///     1, GifParams::default(), &[vec![(0, 1.0)]],
    /// ).unwrap();
    /// let before = layer.clone();
    /// let mut output = [true];
    /// assert_eq!(layer.step_into(&[f32::NAN], &mut output),
    ///     Err(GifLayerError::NonFiniteInput { index: 0, class: NonFiniteClass::Nan }));
    /// assert_eq!(layer, before);
    /// assert_eq!(output, [true]);
    /// layer.step_into(&[1.0], &mut output).unwrap();
    /// ```
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

        // Checked before any state is touched: a restored checkpoint can carry
        // a counter at i64::MAX, and incrementing it below would panic in debug
        // or wrap to i64::MIN in release, corrupting every later
        // `last_spike_time` comparison. Failing here leaves the layer untouched
        // rather than half-stepped.
        let next_step_count =
            self.step_count
                .checked_add(1)
                .ok_or(GifLayerError::StepCounterExhausted {
                    step_count: self.step_count,
                })?;

        self.validate_numeric_inputs(stimuli)?;
        for neuron in 0..self.num_neurons {
            match self.validate_transition(neuron, stimuli) {
                Ok(next) => self.transitions.push(next),
                Err(error) => {
                    self.transitions.clear();
                    return Err(error);
                }
            }
        }

        // Draining retains capacity and restores empty scratch before returning.
        for (neuron, (spike, next)) in spikes
            .iter_mut()
            .zip(self.transitions.drain(..))
            .enumerate()
        {
            self.membrane[neuron] = next.membrane;
            self.adaptation[neuron] = next.adaptation;
            *spike = next.fired;
            if next.fired {
                self.last_spike_time[neuron] = self.step_count;
            }
        }

        self.step_count = next_step_count;
        Ok(())
    }

    /// Check all mutable ingress, even stimuli not referenced by the topology.
    fn validate_numeric_inputs(&self, stimuli: &[f32]) -> Result<(), GifLayerError> {
        validate_params(&self.params)?;
        if let Some((index, class)) = first_non_finite(stimuli) {
            return Err(GifLayerError::NonFiniteInput { index, class });
        }
        Ok(())
    }

    /// Non-finite weights necessarily poison the drive with finite stimuli
    /// (including inf * 0 -> NaN). Diagnose their CSR index only on failure,
    /// avoiding an extra synapse traversal on the successful path.
    fn validate_transition(
        &self,
        neuron: usize,
        stimuli: &[f32],
    ) -> Result<Transition, GifLayerError> {
        let next = self.transition(neuron, stimuli);
        if !next.drive_is_finite() {
            let lo = self.fan_in_offsets[neuron];
            let hi = self.fan_in_offsets[neuron + 1];
            if let Some((index, class)) = first_non_finite(&self.weights[lo..hi]) {
                return Err(GifLayerError::NonFiniteWeight {
                    index: lo + index,
                    class,
                });
            }
        }
        next.validate(neuron)?;
        Ok(next)
    }

    /// Fixed-order CSR accumulation and shared GIF dynamics, without mutation.
    fn transition(&self, neuron: usize, stimuli: &[f32]) -> Transition {
        let lo = self.fan_in_offsets[neuron];
        let hi = self.fan_in_offsets[neuron + 1];
        let drive: f32 = self.fan_in_sources[lo..hi]
            .iter()
            .zip(&self.weights[lo..hi])
            .map(|(&source, &weight)| weight * stimuli[source as usize])
            .sum();
        Transition::compute(
            &self.params,
            self.membrane[neuron],
            self.adaptation[neuron],
            drive,
        )
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
    /// Propagates any [`Self::step_into`] error on the first invalid frame.
    /// That frame leaves state unchanged; preceding frames remain applied.
    /// The whole batch is not transactional. An empty batch does not step or
    /// validate mutable parameters/weights. `reset` clears dynamics but does not
    /// repair parameters or weights; correct those edits before retrying.
    pub fn run<S: AsRef<[f32]>>(
        &mut self,
        spike_train: &[S],
    ) -> Result<SpikeRaster, GifLayerError> {
        let num_steps = spike_train.len();
        // Checked: an overflowing product panics in debug, and in release wraps
        // to a short allocation that then panics when a row is copied into it.
        let raster_len =
            num_steps
                .checked_mul(self.num_neurons)
                .ok_or(GifLayerError::RasterTooLarge {
                    num_steps,
                    num_neurons: self.num_neurons,
                })?;
        let mut spikes = vec![false; raster_len];
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
#[path = "layer_tests.rs"]
mod tests;
