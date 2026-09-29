//! Node classification for a loaded NIR graph.
//!
//! Every node in a graph is sorted into exactly one [`NodeRole`]. Only the IF
//! neuron ([`NodeRole::SupportedNeuron`]) is mapped into a neuromod primitive by
//! this harness; everything else is recorded either as a graph boundary port or
//! as an *adapter concern* that stays **downstream** of neuromod (see the
//! boundary note in `lib.rs` and `docs/neuromod-boundary-matrix.md`).

use nir_rs::{NirGraph, NirNode};

/// The kind of adapter/runtime concern a non-neuron, non-boundary node
/// represents. These are deliberately *not* implemented in neuromod; they name
/// the responsibility that a full NIR runtime would own downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    /// Dense/convolutional tensor kernels (`Affine`, `Conv2d`).
    TensorKernel,
    /// Spatial pooling (`AvgPool2d`).
    Pooling,
    /// Shape reshaping (`Flatten`).
    Reshape,
}

impl std::fmt::Display for AdapterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TensorKernel => f.write_str("tensor-kernel"),
            Self::Pooling => f.write_str("pooling"),
            Self::Reshape => f.write_str("reshape"),
        }
    }
}

/// The role a NIR node plays with respect to the neuromod handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeRole {
    /// A neuron this harness maps into a neuromod `LapicqueNeuron` bank (`IF`).
    SupportedNeuron,
    /// A virtual graph boundary port (`Input`, `Output`).
    GraphBoundary,
    /// A downstream adapter concern (tensor kernel, pooling, reshape).
    AdapterConcern {
        /// Which class of downstream concern this node is.
        kind: AdapterKind,
    },
    /// Any other wire type: not mapped, carries its wire `type` string.
    Unsupported(&'static str),
}

/// Classify a single NIR node into its [`NodeRole`].
#[must_use]
pub fn classify(node: &NirNode) -> NodeRole {
    match node {
        NirNode::If(_) => NodeRole::SupportedNeuron,
        NirNode::Input(_) | NirNode::Output(_) => NodeRole::GraphBoundary,
        NirNode::Affine(_) | NirNode::Conv2d(_) => NodeRole::AdapterConcern {
            kind: AdapterKind::TensorKernel,
        },
        NirNode::AvgPool2d(_) => NodeRole::AdapterConcern {
            kind: AdapterKind::Pooling,
        },
        NirNode::Flatten(_) => NodeRole::AdapterConcern {
            kind: AdapterKind::Reshape,
        },
        other => NodeRole::Unsupported(other.type_name()),
    }
}

/// A single per-node classification record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRecord {
    /// The node name (map key in `graph.nodes`).
    pub name: String,
    /// The upstream wire `type` string (`NirNode::type_name`).
    pub wire_type: &'static str,
    /// The role assigned by [`classify`].
    pub role: NodeRole,
}

/// Classify every node in `graph`, in insertion order.
///
/// The returned vector has exactly one record per entry in `graph.nodes`; no
/// node is dropped or merged. This is asserted by
/// [`report_covers_all_nodes`].
#[must_use]
pub fn classify_graph(graph: &NirGraph) -> Vec<NodeRecord> {
    graph
        .nodes
        .iter()
        .map(|(name, node)| NodeRecord {
            name: name.clone(),
            wire_type: node.type_name(),
            role: classify(node),
        })
        .collect()
}

/// Whether `report` covers every node in `graph` exactly once.
///
/// Every node name in `graph.nodes` must appear in `report` exactly once with a
/// matching wire type, and `report` must carry no extra or duplicate names. A
/// length match alone is insufficient: a report that repeats one node while
/// omitting another (e.g. `[a, a]` for a graph `{a, b}`) also has the right
/// length and each record resolves, so duplicates are rejected explicitly by
/// tracking seen names and requiring the deduplicated set to cover every node.
#[must_use]
pub fn report_covers_all_nodes(graph: &NirGraph, report: &[NodeRecord]) -> bool {
    if report.len() != graph.nodes.len() {
        return false;
    }
    let mut seen = std::collections::HashSet::with_capacity(report.len());
    for record in report {
        // Reject a name that does not exist or whose wire type disagrees.
        let matches = graph
            .get(&record.name)
            .is_some_and(|node| node.type_name() == record.wire_type);
        if !matches {
            return false;
        }
        // Reject duplicates: each node may be reported only once.
        if !seen.insert(record.name.as_str()) {
            return false;
        }
    }
    // With no duplicates and matching length, every graph node is covered once.
    seen.len() == graph.nodes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nir_rs::nodes::Input;

    fn two_node_graph() -> NirGraph {
        let mut g = NirGraph::new();
        g.insert_node(
            "a",
            NirNode::Input(Input {
                shape: vec![1],
                metadata: Default::default(),
            }),
        )
        .unwrap();
        g.insert_node(
            "b",
            NirNode::Output(nir_rs::nodes::Output {
                shape: vec![1],
                metadata: Default::default(),
            }),
        )
        .unwrap();
        g
    }

    #[test]
    fn classify_graph_covers_all_nodes() {
        let g = two_node_graph();
        let report = classify_graph(&g);
        assert!(report_covers_all_nodes(&g, &report));
    }

    #[test]
    fn duplicate_record_omitting_a_node_is_rejected() {
        // `[a, a]` has the right length (2) and each record resolves to a real
        // node, but it omits `b`. The "exactly once" contract must reject it.
        let g = two_node_graph();
        let a = NodeRecord {
            name: "a".to_owned(),
            wire_type: "Input",
            role: NodeRole::GraphBoundary,
        };
        let duplicated = vec![a.clone(), a];
        assert!(
            !report_covers_all_nodes(&g, &duplicated),
            "a report that repeats one node and omits another must not pass"
        );
    }

    #[test]
    fn wrong_length_report_is_rejected() {
        let g = two_node_graph();
        let report = vec![NodeRecord {
            name: "a".to_owned(),
            wire_type: "Input",
            role: NodeRole::GraphBoundary,
        }];
        assert!(!report_covers_all_nodes(&g, &report));
    }
}
