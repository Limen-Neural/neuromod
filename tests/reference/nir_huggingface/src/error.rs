//! Error taxonomy for the offline NIR -> neuromod interop harness.
//!
//! Failures are split into three disjoint kinds so a smoke test can assert on
//! *why* a handoff failed without string-matching:
//!
//! * [`HandoffError::Load`] — reading or structurally validating a `.nir`
//!   fixture failed (carries the offending path and the underlying
//!   [`nir_rs::NirError`] as its [`std::error::Error::source`]).
//! * [`HandoffError::UnsupportedMapping`] — the graph loaded fine but a node
//!   cannot be faithfully represented as a neuromod primitive (bad shapes,
//!   non-finite / non-positive thresholds, a non-zero reset potential, …).
//! * [`HandoffError::Runtime`] — a mapped IF bank misbehaved while stepping
//!   through the *real* neuromod integrate / spike path (wrong input length, a
//!   non-finite value, or a spiking element that did not hard-reset to `0`).
//!
//! The [`RuntimeCause`] carried by [`HandoffError::Runtime`] reuses
//! [`neuromod::NonFiniteClass`] (NaN / `+inf` / `-inf`) for the non-finite case
//! rather than redefining that classification here.

use std::path::PathBuf;

use neuromod::NonFiniteClass;

/// Why a [`HandoffError::Runtime`] was raised for a specific element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RuntimeCause {
    /// The supplied input vector length did not match the neuron bank size.
    ///
    /// `index` on the enclosing [`HandoffError::Runtime`] is `0` for this cause.
    InputLenMismatch {
        /// Expected length (the neuron bank size).
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// A value was NaN or infinite: either an input sample (detected before any
    /// neuron mutation) or a membrane potential observed after stepping.
    NonFinite(NonFiniteClass),
    /// A neuron reported a spike but its membrane potential was not reset to
    /// `0` afterwards, violating the hard-reset IF contract.
    ResetViolation {
        /// The membrane potential observed after the spike (should be `0.0`).
        potential: f32,
    },
}

impl std::fmt::Display for RuntimeCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputLenMismatch { expected, got } => {
                write!(f, "input length {got} does not match bank size {expected}")
            }
            Self::NonFinite(class) => write!(f, "non-finite value ({class})"),
            Self::ResetViolation { potential } => write!(
                f,
                "spiking element did not reset to 0 (membrane_potential = {potential})"
            ),
        }
    }
}

/// A faithful, test-only NIR -> neuromod handoff failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum HandoffError {
    /// Reading or structurally validating a `.nir` fixture failed.
    Load {
        /// The fixture path that could not be loaded.
        path: PathBuf,
        /// The underlying `nir-rs` failure (I/O, decode, or structure).
        source: nir_rs::NirError,
    },
    /// A node cannot be faithfully mapped onto a neuromod primitive.
    UnsupportedMapping {
        /// The graph node name (map key in `graph.nodes`).
        node: String,
        /// The upstream wire `type` string (`NirNode::type_name`).
        type_name: &'static str,
        /// Why the mapping is unsupported.
        reason: String,
    },
    /// A mapped IF bank misbehaved on the real neuromod step path.
    Runtime {
        /// The graph node name the bank was mapped from.
        node: String,
        /// The offending element index within the bank.
        index: usize,
        /// The timestep at which the violation was observed.
        step: i64,
        /// What went wrong.
        cause: RuntimeCause,
    },
}

impl std::fmt::Display for HandoffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load { path, .. } => {
                write!(f, "failed to load NIR fixture `{}`", path.display())
            }
            Self::UnsupportedMapping {
                node,
                type_name,
                reason,
            } => write!(
                f,
                "unsupported mapping for node `{node}` ({type_name}): {reason}"
            ),
            Self::Runtime {
                node,
                index,
                step,
                cause,
            } => write!(
                f,
                "runtime error in node `{node}` element {index} at step {step}: {cause}"
            ),
        }
    }
}

impl std::error::Error for HandoffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Load { source, .. } => Some(source),
            Self::UnsupportedMapping { .. } | Self::Runtime { .. } => None,
        }
    }
}
