/// Errors produced when building or driving a
/// [`SparseGifHiddenLayer`](super::SparseGifHiddenLayer).
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
    /// [`SparseGifHiddenLayer::step`](super::SparseGifHiddenLayer::step).
    MalformedCheckpoint {
        /// Which invariant was violated.
        detail: &'static str,
    },
    /// The monotonic step counter cannot advance another step.
    ///
    /// Reached only by a restored checkpoint carrying a counter at
    /// [`i64::MAX`]; incrementing it would panic in debug builds and wrap to
    /// [`i64::MIN`] in release, corrupting every later `last_spike_time`
    /// comparison. Reported before any neuron state is mutated.
    StepCounterExhausted {
        /// The counter that cannot be advanced.
        step_count: i64,
    },
    /// A batched run's raster would not fit in memory addressing.
    ///
    /// `num_steps * num_neurons` is the flat raster length; an overflowing
    /// product would panic in debug and, in release, wrap to a short
    /// allocation that then panics when a row is written into it.
    RasterTooLarge {
        /// Steps requested.
        num_steps: usize,
        /// Neurons per step.
        num_neurons: usize,
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
            Self::StepCounterExhausted { step_count } => {
                write!(f, "step counter is exhausted at {step_count}")
            }
            Self::RasterTooLarge {
                num_steps,
                num_neurons,
            } => write!(
                f,
                "a {num_steps}-step raster of {num_neurons} neurons exceeds the addressable size"
            ),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_is_informative() {
        let msg = GifLayerError::FanInExceedsInputs {
            fan_in: 5,
            num_inputs: 4,
        }
        .to_string();
        assert!(msg.contains('5') && msg.contains('4'));
    }
}
