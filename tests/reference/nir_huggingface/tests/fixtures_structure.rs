//! Fixture-load and node-classification smoke tests.
//!
//! These load the two vendored Hugging Face-derived `.nir` fixtures **offline**
//! from disk (no Hugging Face Hub access) and assert their structure and the
//! classification every node receives. They exercise the real `nir-rs` HDF5
//! reader plus this harness's [`load_fixture`] and [`classify_graph`] paths.
//!
//! This is an interoperability *smoke*: it checks that HF-derived NIR graphs
//! load and that neuromod can faithfully receive their IF neurons. It does NOT
//! claim bit-identical numerical equivalence with NeuroCUDA.

use std::collections::HashMap;

use nir_huggingface_interop::{
    AdapterKind, CNN_FIXTURE, MLP_FIXTURE, NodeRecord, NodeRole, classify_graph, fixture_path,
    load_fixture, report_covers_all_nodes,
};

/// Count nodes by their wire `type` string.
fn type_counts(graph: &nir_rs::NirGraph) -> HashMap<&'static str, usize> {
    let mut counts = HashMap::new();
    for node in graph.nodes.values() {
        *counts.entry(node.type_name()).or_insert(0) += 1;
    }
    counts
}

#[test]
fn mlp_fixture_loads_with_expected_structure() {
    let graph = load_fixture(fixture_path(MLP_FIXTURE)).expect("MLP fixture loads offline");

    assert_eq!(graph.version.as_deref(), Some("1.0.8"), "MLP NIR version");
    assert_eq!(graph.nodes.len(), 7, "MLP node count");
    assert_eq!(graph.edges.len(), 6, "MLP edge count");

    let counts = type_counts(&graph);
    assert_eq!(counts.get("Affine").copied().unwrap_or(0), 3, "MLP Affine");
    assert_eq!(counts.get("IF").copied().unwrap_or(0), 2, "MLP IF");
    assert_eq!(counts.get("Input").copied().unwrap_or(0), 1, "MLP Input");
    assert_eq!(counts.get("Output").copied().unwrap_or(0), 1, "MLP Output");
}

#[test]
fn cnn_fixture_loads_with_expected_structure() {
    let graph = load_fixture(fixture_path(CNN_FIXTURE)).expect("CNN fixture loads offline");

    assert_eq!(graph.version.as_deref(), Some("1.0.8"), "CNN NIR version");
    assert_eq!(graph.nodes.len(), 13, "CNN node count");
    assert_eq!(graph.edges.len(), 12, "CNN edge count");

    let counts = type_counts(&graph);
    assert_eq!(counts.get("Conv2d").copied().unwrap_or(0), 3, "CNN Conv2d");
    assert_eq!(counts.get("IF").copied().unwrap_or(0), 3, "CNN IF");
    assert_eq!(
        counts.get("AvgPool2d").copied().unwrap_or(0),
        3,
        "CNN AvgPool2d"
    );
    assert_eq!(
        counts.get("Flatten").copied().unwrap_or(0),
        1,
        "CNN Flatten"
    );
    assert_eq!(counts.get("Affine").copied().unwrap_or(0), 1, "CNN Affine");
    assert_eq!(counts.get("Input").copied().unwrap_or(0), 1, "CNN Input");
    assert_eq!(counts.get("Output").copied().unwrap_or(0), 1, "CNN Output");
}

/// A non-IF `AdapterConcern` node's kind matches its wire type.
fn assert_adapter_kind(fixture: &str, record: &NodeRecord, kind: AdapterKind, is_if: bool) {
    assert!(
        !is_if,
        "{fixture}: IF node `{}` must not be an AdapterConcern",
        record.name
    );
    let expected = match record.wire_type {
        "Affine" | "Conv2d" => AdapterKind::TensorKernel,
        "AvgPool2d" => AdapterKind::Pooling,
        "Flatten" => AdapterKind::Reshape,
        other => panic!("{fixture}: unexpected AdapterConcern wire type {other}"),
    };
    assert_eq!(
        kind, expected,
        "{fixture}: node `{}` ({}) adapter kind",
        record.name, record.wire_type
    );
}

/// Every node is classified; only IF nodes are `SupportedNeuron`, every other
/// node gets a non-supported role, and no node is `Unsupported`.
fn assert_classification(fixture: &str) {
    let graph = load_fixture(fixture_path(fixture)).expect("fixture loads offline");
    let report = classify_graph(&graph);

    assert!(
        report_covers_all_nodes(&graph, &report),
        "{fixture}: classification must cover every node exactly once"
    );

    for record in &report {
        let is_if = record.wire_type == "IF";
        match &record.role {
            NodeRole::SupportedNeuron => assert!(
                is_if,
                "{fixture}: node `{}` ({}) is SupportedNeuron but not IF",
                record.name, record.wire_type
            ),
            NodeRole::GraphBoundary => assert!(
                matches!(record.wire_type, "Input" | "Output"),
                "{fixture}: node `{}` ({}) is GraphBoundary but not Input/Output",
                record.name,
                record.wire_type
            ),
            NodeRole::AdapterConcern { kind } => {
                assert_adapter_kind(fixture, record, *kind, is_if);
            }
            NodeRole::Unsupported(type_name) => panic!(
                "{fixture}: node `{}` ({}) classified Unsupported; every node in these fixtures must map to a known role",
                record.name, type_name
            ),
        }

        // Restate the acceptance criterion directly: every non-IF node has a
        // non-supported role.
        if !is_if {
            assert!(
                !matches!(record.role, NodeRole::SupportedNeuron),
                "{fixture}: non-IF node `{}` ({}) must not be SupportedNeuron",
                record.name,
                record.wire_type
            );
        }
    }
}

#[test]
fn mlp_every_node_classified_and_none_unsupported() {
    assert_classification(MLP_FIXTURE);
}

#[test]
fn cnn_every_node_classified_and_none_unsupported() {
    assert_classification(CNN_FIXTURE);
}
