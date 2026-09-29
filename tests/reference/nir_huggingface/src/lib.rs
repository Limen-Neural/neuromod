//! Offline NIR -> neuromod interoperability smoke harness.
//!
//! This is a **test-only** crate that is deliberately isolated from the
//! `neuromod` library (it has its own empty `[workspace]` table and is excluded
//! from neuromod's package and workspace). It loads the two vendored
//! Hugging Face-derived `.nir` fixtures through `nir-rs` and maps the subset of
//! graph semantics that `neuromod` can represent (IF neurons) into neuromod
//! primitives, while explicitly classifying everything else as an
//! adapter/runtime concern that stays *outside* `neuromod`.
//!
//! FEAT-001 scaffolds the crate: it vendors the fixtures with provenance and a
//! SHA-256 checksum test, and exposes a manifest-relative fixture-path helper.
//! FEAT-002 adds the adapter: [`load_fixture`] loads and structurally validates
//! a `.nir` graph, [`classify_graph`] sorts every node into a [`NodeRole`], and
//! [`IfHandoff`] maps `IF` neurons into a neuromod `LapicqueNeuron` bank and
//! steps them through the real neuromod integrate / spike path.
//!
//! # Boundary
//!
//! This crate is the *only* place NIR loading, tensor reading, node
//! classification, and the IF handoff live. The `neuromod` library gains no
//! `nir-rs` / `hdf5` dependency and no import, tensor-math, or graph-runtime
//! responsibility. NIR scheduling, the `Affine` / `Conv2d` / `AvgPool2d` /
//! `Flatten` kernels, and graph wiring stay downstream of neuromod (see
//! `docs/neuromod-boundary-matrix.md`); here they are only *classified*, never
//! executed.

mod classify;
mod error;
mod handoff;

pub use classify::{
    AdapterKind, NodeRecord, NodeRole, classify, classify_graph, report_covers_all_nodes,
};
pub use error::{HandoffError, RuntimeCause};
pub use handoff::IfHandoff;

use std::path::{Path, PathBuf};

use nir_rs::NirGraph;

/// Load and structurally validate a `.nir` fixture.
///
/// Reads the graph with [`nir_rs::io::read`], then runs
/// [`NirGraph::validate_structure`]. Either failure is mapped to
/// [`HandoffError::Load`] carrying the fixture `path` and the underlying
/// [`nir_rs::NirError`] as the error source.
///
/// # Errors
///
/// [`HandoffError::Load`] if the file cannot be read/decoded, or if the loaded
/// graph fails structural validation.
pub fn load_fixture(path: impl AsRef<Path>) -> Result<NirGraph, HandoffError> {
    let path = path.as_ref();
    let graph = nir_rs::io::read(path).map_err(|source| HandoffError::Load {
        path: path.to_path_buf(),
        source,
    })?;
    graph
        .validate_structure()
        .map_err(|source| HandoffError::Load {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(graph)
}

/// Absolute path to the vendored fixtures directory.
///
/// Paths are resolved from `CARGO_MANIFEST_DIR` (this crate's root), NOT the
/// process working directory, so tests and examples find the fixtures no matter
/// where `cargo` is invoked from.
#[must_use]
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Absolute path to a named fixture inside [`fixtures_dir`].
#[must_use]
pub fn fixture_path(file_name: &str) -> PathBuf {
    fixtures_dir().join(file_name)
}

/// The MLP MNIST fixture file name.
pub const MLP_FIXTURE: &str = "neurocuda_mlp_mnist.nir";

/// The CNN N-MNIST fixture file name.
pub const CNN_FIXTURE: &str = "neurocuda_cnn_nmnist.nir";

/// Expected SHA-256 of [`MLP_FIXTURE`] (see `MANIFEST.toml` / the NOTICE file).
pub const MLP_SHA256: &str = "fc0b1a1e0c4caeb9d1f7be8700de0212a76ec5f441cae13038887411fd9a1ef0";

/// Expected SHA-256 of [`CNN_FIXTURE`] (see `MANIFEST.toml` / the NOTICE file).
pub const CNN_SHA256: &str = "972b45984094606b83b5b19173a653524550a5aa416f8a46398baa4250c34f2f";

/// Compute the lowercase hex SHA-256 digest of a byte slice.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored fixtures must match the provenance-pinned SHA-256 values.
    /// Paths are resolved from `CARGO_MANIFEST_DIR`, not the working directory.
    #[test]
    fn vendored_fixtures_match_expected_sha256() {
        for (file_name, expected) in [(MLP_FIXTURE, MLP_SHA256), (CNN_FIXTURE, CNN_SHA256)] {
            let path = fixture_path(file_name);
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()));
            let actual = sha256_hex(&bytes);
            assert_eq!(
                actual,
                expected,
                "SHA-256 mismatch for {} (expected {expected}, got {actual})",
                path.display(),
            );
        }
    }
}
