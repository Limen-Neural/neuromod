use serde::{Deserialize, Serialize};

use crate::gif::GifParams;

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

impl SparseGifHiddenLayerRepr {
    /// Enforce every invariant `step_into` relies on when indexing.
    ///
    /// Checked in the order a reader would: addressability, then the SoA bank
    /// widths, then the CSR row structure, then the payload lengths the row
    /// offsets imply, and finally that each source names a real channel.
    fn validate(&self) -> Result<(), GifLayerError> {
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
                // Drawn once, outside the branch: both paths must consume
                // exactly one value or the seeded stream would diverge.
                let unit = rng.next_unit();
                let span = w_max - w_min;

                // `span` overflows to `inf` for a finite range wider than
                // f32::MAX (e.g. -3e38..3e38), and `inf * 0.0` is `NaN`, so
                // that range would install non-finite synapses despite passing
                // the finiteness check in `new`. Only that case falls back to
                // f64.
                //
                // The fallback is deliberately NOT applied to ordinary ranges.
                // f64 interpolation rounds once where the f32 expression rounds
                // three times, so the two disagree by 1 ULP on roughly 30% of
                // random ranges. Routing every range through f64 would silently
                // reseed existing layers. (The default 0.0..1.0 range happens to
                // agree exactly -- span is 1.0 and the offset is 0.0, so all
                // three f32 roundings are exact -- which is why the golden
                // fixtures alone do not catch the difference.)
                let w = if span.is_finite() {
                    w_min + span * unit
                } else {
                    (f64::from(w_min) + (f64::from(w_max) - f64::from(w_min)) * f64::from(unit))
                        as f32
                };

                // `unit` is always below 1, but the rounding above can still
                // carry the result up to exactly `w_max`, breaking the
                // half-open range `weight_range` documents. It takes bounds
                // only a few ULPs apart -- with adjacent floats any unit above
                // 0.5 rounds up -- so stepping back one representable value
                // costs nothing for real ranges: a 2M-draw sweep over random
                // ranges produced zero results at or above `w_max`. Skipped
                // when the range is degenerate (`w_min == w_max`), where the
                // half-open interval is empty and `w_min` is the only answer.
                let w = if w >= w_max && w_min < w_max {
                    w_max.next_down()
                } else {
                    w
                };
                weights.push(w);
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

        self.step_count = next_step_count;
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
