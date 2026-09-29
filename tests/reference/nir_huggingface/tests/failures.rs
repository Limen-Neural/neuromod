//! Failure-category smoke tests.
//!
//! Every faithful-handoff failure is one of three disjoint kinds. These tests
//! provoke each kind and assert the exact variant / cause:
//!
//! * `HandoffError::Load` — a missing path and a non-HDF5 file.
//! * `HandoffError::UnsupportedMapping` — an in-memory `IF` with a non-zero
//!   `v_reset` and one with a shape mismatch.
//! * `HandoffError::Runtime` — NaN / `+inf` / `-inf` / wrong-length inputs to
//!   `step`, each with the matching cause and no mutation of neuron state.
//!
//! This is an interoperability *smoke* and does NOT claim bit-identical
//! equivalence with NeuroCUDA.

use std::path::Path;

use neuromod::NonFiniteClass;
use nir_huggingface_interop::{HandoffError, IfHandoff, RuntimeCause, load_fixture};
use nir_rs::Tensor;
use nir_rs::nodes::If;

/// Build an in-memory `IF` node from flat vectors sharing one shape.
fn if_node(r: Vec<f32>, v_threshold: Vec<f32>, v_reset: Option<Vec<f32>>) -> If {
    If {
        r: Tensor::from_f32([r.len()], r).unwrap(),
        v_threshold: Tensor::from_f32([v_threshold.len()], v_threshold).unwrap(),
        v_reset: v_reset.map(|values| Tensor::from_f32([values.len()], values).unwrap()),
        metadata: Default::default(),
    }
}

/// Assert that mapping `node` fails as an unsupported mapping.
fn assert_unsupported(node: &If, what: &str) {
    let err = IfHandoff::from_node("if_bad", node).expect_err(what);
    assert!(
        matches!(err, HandoffError::UnsupportedMapping { .. }),
        "expected UnsupportedMapping, got {err:?}"
    );
}

// --- Load failures ---------------------------------------------------------

#[test]
fn load_missing_path_is_load_error() {
    let missing = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("this_file_does_not_exist.nir");
    let err = load_fixture(&missing).expect_err("loading a missing path must fail");
    assert!(
        matches!(err, HandoffError::Load { .. }),
        "expected Load, got {err:?}"
    );
}

#[test]
fn load_non_hdf5_file_is_load_error() {
    // Write a small non-HDF5 file under the per-test target tmp dir.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join("not_an_hdf5_file.nir");
    std::fs::write(&path, b"not an hdf5 file").expect("write scratch file");

    let err = load_fixture(&path).expect_err("loading a non-HDF5 file must fail");
    assert!(
        matches!(err, HandoffError::Load { .. }),
        "expected Load, got {err:?}"
    );

    let _ = std::fs::remove_file(&path);
}

// --- UnsupportedMapping failures -------------------------------------------

#[test]
fn non_zero_v_reset_is_unsupported_mapping() {
    // Valid r / v_threshold, but v_reset is non-zero -> not a hard reset to 0.
    let node = if_node(vec![1.0, 1.0], vec![1.0, 1.0], Some(vec![0.1, 0.0]));
    assert_unsupported(&node, "non-zero v_reset must be rejected");
}

#[test]
fn shape_mismatch_is_unsupported_mapping() {
    // r shape [2] but v_threshold shape [3] -> shapes disagree.
    let node = if_node(vec![1.0, 1.0], vec![1.0, 1.0, 1.0], None);
    assert_unsupported(&node, "shape mismatch must be rejected");
}

// --- Runtime failures ------------------------------------------------------

/// A valid, small two-element IF handoff, advanced one sub-threshold step so
/// "unchanged after failure" is a meaningful, non-zero assertion.
fn warmed_handoff() -> IfHandoff {
    let node = if_node(vec![1.0, 1.0], vec![1.0, 1.0], None);
    let mut handoff = IfHandoff::from_node("if_runtime", &node).expect("valid IF maps");
    handoff.step(&[0.3, 0.4], 0).expect("valid warm-up step");
    handoff
}

/// The membrane potentials of a bank, in order.
fn potentials(handoff: &IfHandoff) -> Vec<f32> {
    handoff
        .bank()
        .iter()
        .map(|neuron| neuron.membrane_potential)
        .collect()
}

#[test]
fn non_finite_inputs_are_runtime_errors_without_mutation() {
    for (bad, expected) in [
        (f32::NAN, NonFiniteClass::Nan),
        (f32::INFINITY, NonFiniteClass::PosInfinity),
        (f32::NEG_INFINITY, NonFiniteClass::NegInfinity),
    ] {
        let mut handoff = warmed_handoff();
        let before = potentials(&handoff);

        // Put the bad value in the second element to prove the scan runs over
        // the whole input before any mutation.
        let err = handoff
            .step(&[0.1, bad], 1)
            .expect_err("non-finite input must fail");
        match err {
            HandoffError::Runtime {
                index,
                cause: RuntimeCause::NonFinite(class),
                ..
            } => {
                assert_eq!(class, expected, "non-finite class for {bad}");
                assert_eq!(index, 1, "offending element index");
            }
            other => panic!("expected Runtime NonFinite, got {other:?}"),
        }

        assert_eq!(
            potentials(&handoff),
            before,
            "neuron state must be unchanged after a failed step ({bad})"
        );
    }
}

#[test]
fn wrong_length_input_is_runtime_error_without_mutation() {
    let mut handoff = warmed_handoff();
    let before = potentials(&handoff);

    let err = handoff
        .step(&[0.5], 1)
        .expect_err("wrong-length input must fail");
    match err {
        HandoffError::Runtime {
            cause: RuntimeCause::InputLenMismatch { expected, got },
            ..
        } => {
            assert_eq!(expected, 2, "expected bank size");
            assert_eq!(got, 1, "supplied length");
        }
        other => panic!("expected Runtime InputLenMismatch, got {other:?}"),
    }

    assert_eq!(
        potentials(&handoff),
        before,
        "neuron state must be unchanged after a length-mismatch step"
    );
}
