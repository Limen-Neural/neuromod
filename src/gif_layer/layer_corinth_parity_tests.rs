//! Cross-repository GIF bit-parity test.
//!
//! This test replays a fixture captured from the author's `corinth-canal`
//! spike-to-embedding pipeline at the *pinned* source commit
//! `8e54e234ac005dd84e4ad2bedbf9f5bceb082355` (its `funnel.rs`, recorded by the
//! SHA-256 asserted below) and checks that this crate's
//! [`SparseGifHiddenLayer`] reproduces every spike ID and every final
//! membrane/adaptation f32 *bit for bit*, with no tolerance.
//!
//! "Every spike ID" is literal: the fixture carries a per-step fired-ID oracle
//! (`per_step_fired_ids`) for all 512 steps, and the replay asserts the exact
//! ascending fired-ID list on each step. Per-step spike counts and the
//! deterministically selected raster rows are still checked, but they are
//! redundant safety nets over the exhaustive per-step ID comparison, not the
//! guarantee itself.
//!
//! # Scope
//!
//! The parity contract is the **shared GIF dynamics only**: the per-step
//! integrate/threshold/soft-reset arithmetic and the fixed-order fan-in
//! accumulation. Because f32 addition is not associative, the drive sum must be
//! accumulated in exactly the edge order Corinth used, so the fixture stores
//! each neuron's fan-in row in Corinth edge order and the test rebuilds the
//! topology with [`SparseGifHiddenLayer::from_topology`], which preserves the
//! given row order.
//!
//! # Explicitly excluded
//!
//! The topology **generator** is out of scope. Corinth's fixed fan-in index
//! formula differs *intentionally* from this crate's per-neuron SplitMix64
//! generator; parity is established by reconstructing Corinth's explicit
//! topology through `from_topology`, never by matching generators. The internal
//! generator + CSR-traversal + arithmetic goldens live separately in
//! `layer_golden_tests.rs`.
//!
//! # Offline
//!
//! The fixture is embedded at compile time via `include_str!`; this test has no
//! network or runtime dependency on `corinth-canal`.

use super::*;
use serde::Deserialize;

/// The committed Corinth parity fixture, embedded at compile time.
const FIXTURE_JSON: &str = include_str!("../../tests/reference/corinth_gif_parity.json");

// Pinned provenance the fixture must carry. A drift here means the vectors were
// regenerated against a different Corinth revision and the parity claim below
// no longer describes what was actually audited.
const PINNED_CORINTH_COMMIT: &str = "8e54e234ac005dd84e4ad2bedbf9f5bceb082355";
const PINNED_FUNNEL_SHA256: &str =
    "10192537a1a096fc8ec8a9a87740b643624b64c9fc3f3408faf06653b6694b47";

// Audited case dimensions.
const EXPECT_NUM_STEPS: usize = 512;
const EXPECT_NUM_NEURONS: usize = 2048;
const EXPECT_NUM_INPUTS: usize = 2048;
const EXPECT_FAN_IN: usize = 4;

/// Private deserialization mirror of the fixture. Local to this module so no
/// public API is introduced.
#[derive(Deserialize)]
struct Fixture {
    provenance: Provenance,
    dimensions: Dimensions,
    counts: Counts,
    param_bits: ParamBits,
    input_masks_hex: Vec<String>,
    per_step_mask_index: Vec<usize>,
    per_step_spike_count: Vec<usize>,
    /// Exhaustive per-step fired-ID oracle: `per_step_fired_ids[t]` is the
    /// ascending list of hidden-neuron IDs that fired at step `t`, for every
    /// one of the 512 steps. This is what lets the replay verify the exact
    /// spike ID (not merely the count) on all steps.
    per_step_fired_ids: Vec<Vec<usize>>,
    selected_raster_rows: Vec<RasterRow>,
    /// One flat row per neuron: `[src0, wbits0, src1, wbits1, ...]` in Corinth
    /// edge order (NOT sorted).
    topology_edge_order: Vec<Vec<u64>>,
    final_membrane_bits: Vec<u32>,
    final_adaptation_bits: Vec<u32>,
}

#[derive(Deserialize)]
struct Provenance {
    corinth_source_commit: String,
    corinth_funnel_rs_sha256: String,
}

#[derive(Deserialize)]
struct Dimensions {
    num_steps: usize,
    num_neurons: usize,
    num_inputs: usize,
    fan_in: usize,
}

#[derive(Deserialize)]
struct Counts {
    total_spikes: usize,
}

/// GIF parameter defaults as f32 `to_bits()` patterns.
#[derive(Deserialize)]
struct ParamBits {
    leak: u32,
    drive_scale: u32,
    base_threshold: u32,
    adaptation_scale: u32,
    adaptation_decay: u32,
    adaptation_coupling: u32,
    adaptation_increment: u32,
    reset_ratio: u32,
}

#[derive(Deserialize)]
struct RasterRow {
    step: usize,
    fired_ids: Vec<usize>,
    spike_count: usize,
}

/// Decode a little-endian hex mask into a REUSABLE dense 0/1 f32 buffer.
///
/// The mask encodes 2048 channels as 256 bytes (512 hex chars); channel `i`
/// lives in byte `i / 8`, bit `i % 8` (little-endian within each byte). The
/// caller owns `buffer` and this reuses it across all 512 steps: it is cleared
/// to `0.0` and only the active channels are set to `1.0`.
fn fill_dense_from_hex(hex: &str, buffer: &mut [f32]) {
    buffer.fill(0.0);
    let bytes = hex.as_bytes();
    // Each channel maps to one hex nibble pair; walk channels and consult the
    // corresponding bit rather than parsing the whole string into a Vec<u8>.
    for (channel, slot) in buffer.iter_mut().enumerate() {
        let byte_index = channel / 8;
        let bit_index = channel % 8;
        // byte_index -> two hex chars, high nibble first.
        let hi = hex_nibble(bytes[byte_index * 2]);
        let lo = hex_nibble(bytes[byte_index * 2 + 1]);
        let byte = (hi << 4) | lo;
        if (byte >> bit_index) & 1 == 1 {
            *slot = 1.0;
        }
    }
}

/// Decode one lowercase-hex ASCII digit to its 0..16 value.
fn hex_nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("invalid hex digit in input mask: {c:#x}"),
    }
}

/// Assert the fixture carries the pinned Corinth provenance. A drift here means
/// the vectors were regenerated against a different Corinth revision and the
/// parity claim no longer describes what was actually audited.
fn assert_provenance(fixture: &Fixture) {
    assert_eq!(
        fixture.provenance.corinth_source_commit, PINNED_CORINTH_COMMIT,
        "fixture pins a different corinth-canal source commit"
    );
    assert_eq!(
        fixture.provenance.corinth_funnel_rs_sha256, PINNED_FUNNEL_SHA256,
        "fixture pins a different funnel.rs SHA-256"
    );
}

/// Assert the audited-case dimensions and that every fixture array is sized as
/// those dimensions claim before we start indexing them.
fn assert_dimensions(fixture: &Fixture) {
    assert_eq!(fixture.dimensions.num_steps, EXPECT_NUM_STEPS);
    assert_eq!(fixture.dimensions.num_neurons, EXPECT_NUM_NEURONS);
    assert_eq!(fixture.dimensions.num_inputs, EXPECT_NUM_INPUTS);
    assert_eq!(fixture.dimensions.fan_in, EXPECT_FAN_IN);

    let num_steps = fixture.dimensions.num_steps;
    let num_neurons = fixture.dimensions.num_neurons;

    assert_eq!(fixture.topology_edge_order.len(), num_neurons);
    assert_eq!(fixture.final_membrane_bits.len(), num_neurons);
    assert_eq!(fixture.final_adaptation_bits.len(), num_neurons);
    assert_eq!(fixture.per_step_mask_index.len(), num_steps);
    assert_eq!(fixture.per_step_spike_count.len(), num_steps);
    assert_eq!(
        fixture.per_step_fired_ids.len(),
        num_steps,
        "per_step_fired_ids must cover every step"
    );
}

/// Assert the exhaustive fired-ID oracle is internally consistent with the
/// per-step counts and the selected raster rows it supersedes, before we replay
/// anything.
fn assert_oracle_self_consistent(fixture: &Fixture) {
    let num_steps = fixture.dimensions.num_steps;
    for step in 0..num_steps {
        let fired_ids = &fixture.per_step_fired_ids[step];
        assert_eq!(
            fired_ids.len(),
            fixture.per_step_spike_count[step],
            "per_step_fired_ids[{step}] length disagrees with per_step_spike_count[{step}]"
        );
        assert!(
            fired_ids.windows(2).all(|w| w[0] < w[1]),
            "per_step_fired_ids[{step}] is not strictly ascending: {fired_ids:?}"
        );
    }
    for row in &fixture.selected_raster_rows {
        assert_eq!(
            fixture.per_step_fired_ids[row.step], row.fired_ids,
            "selected_raster_rows[{}] disagrees with per_step_fired_ids",
            row.step
        );
    }
}

/// Assert this crate's `GifParams::default()` are the exact f32 constants
/// Corinth ran with; compare bit patterns, not approximate values, and return
/// the params so the caller can rebuild the topology with them.
fn assert_param_bits(fixture: &Fixture) -> GifParams {
    let params = GifParams::default();
    assert_eq!(params.leak.to_bits(), fixture.param_bits.leak, "param leak");
    assert_eq!(
        params.drive_scale.to_bits(),
        fixture.param_bits.drive_scale,
        "param drive_scale"
    );
    assert_eq!(
        params.base_threshold.to_bits(),
        fixture.param_bits.base_threshold,
        "param base_threshold"
    );
    assert_eq!(
        params.adaptation_scale.to_bits(),
        fixture.param_bits.adaptation_scale,
        "param adaptation_scale"
    );
    assert_eq!(
        params.adaptation_decay.to_bits(),
        fixture.param_bits.adaptation_decay,
        "param adaptation_decay"
    );
    assert_eq!(
        params.adaptation_coupling.to_bits(),
        fixture.param_bits.adaptation_coupling,
        "param adaptation_coupling"
    );
    assert_eq!(
        params.adaptation_increment.to_bits(),
        fixture.param_bits.adaptation_increment,
        "param adaptation_increment"
    );
    assert_eq!(
        params.reset_ratio.to_bits(),
        fixture.param_bits.reset_ratio,
        "param reset_ratio"
    );
    params
}

/// Reconstruct the per-neuron fan-in topology in fixture (Corinth edge) order.
///
/// Each flat row is `[src0, wbits0, src1, wbits1, ...]`. We keep the pairs in
/// the order given and decode weights via `f32::from_bits`. DO NOT sort:
/// fixed-order f32 accumulation is part of the bit contract, and
/// `from_topology` preserves the row order we hand it.
fn rebuild_topology(fixture: &Fixture) -> Vec<Vec<(usize, f32)>> {
    fixture
        .topology_edge_order
        .iter()
        .enumerate()
        .map(|(neuron, flat)| {
            assert_eq!(
                flat.len(),
                fixture.dimensions.fan_in * 2,
                "topology row {neuron} is not fan_in (src,wbits) pairs"
            );
            (0..fixture.dimensions.fan_in)
                .map(|edge| {
                    let source = flat[edge * 2] as usize;
                    let weight = f32::from_bits(flat[edge * 2 + 1] as u32);
                    (source, weight)
                })
                .collect()
        })
        .collect()
}

/// Replay all 512 frames through `layer` and assert bit parity on every step.
///
/// Corinth drives each neuron by summing the weights of its *active* input
/// edges. Converting each frame to a dense 0/1 vector and letting `step` sum
/// `weight * stimuli[source]` is bit-equivalent because in f32
/// `weight * 1.0 == weight` and `weight * 0.0 == 0.0` exactly (no rounding),
/// and the per-row accumulation order matches Corinth's because the topology
/// row order was preserved when the layer was rebuilt. So the dense sum equals
/// Corinth's active-index sum term for term, in the same order.
fn replay_and_assert(layer: &mut SparseGifHiddenLayer, fixture: &Fixture) {
    let num_steps = fixture.dimensions.num_steps;
    let num_inputs = fixture.dimensions.num_inputs;

    let mut dense = vec![0.0f32; num_inputs];
    // Selected raster rows are keyed by step; resolve each as we reach it.
    let mut selected_by_step: std::collections::HashMap<usize, &RasterRow> =
        std::collections::HashMap::with_capacity(fixture.selected_raster_rows.len());
    for row in &fixture.selected_raster_rows {
        selected_by_step.insert(row.step, row);
    }

    let mut running_total_spikes = 0usize;
    for step in 0..num_steps {
        let mask_index = fixture.per_step_mask_index[step];
        let hex = &fixture.input_masks_hex[mask_index];
        fill_dense_from_hex(hex, &mut dense);

        let fired = layer.step(&dense).expect("step must not error");

        // Per-step spike count, no tolerance.
        let want_count = fixture.per_step_spike_count[step];
        assert_eq!(
            fired.len(),
            want_count,
            "spike-count divergence at step {step}: got {}, want {want_count}",
            fired.len()
        );
        running_total_spikes += fired.len();

        // Every-step fired-ID parity, no tolerance. This is the core bit-parity
        // guarantee: a regression that returned a wrong neuron ID while keeping
        // the count intact would slip past a count-only check, so compare the
        // exact fired-ID list on ALL 512 steps, not just the selected ones. The
        // fixture stores IDs ascending and `step` returns them ascending, so
        // this reports the first divergent step and the exact IDs involved.
        let want_fired = &fixture.per_step_fired_ids[step];
        assert_eq!(
            &fired, want_fired,
            "fired-ID divergence at step {step}: got {fired:?}, want {want_fired:?}"
        );

        // Selected fired-ID rows remain valid and NOTICE-referenced; assert
        // they still agree at their steps (indices ascending from `step`).
        if let Some(row) = selected_by_step.get(&step) {
            assert_eq!(
                row.spike_count, want_count,
                "fixture is internally inconsistent at selected step {step}"
            );
            assert_eq!(
                fired, row.fired_ids,
                "fired-ID divergence at selected step {step}: got {fired:?}, want {:?}",
                row.fired_ids
            );
        }
    }

    // Sanity: the replayed spike total matches the recorded count.
    assert_eq!(
        running_total_spikes, fixture.counts.total_spikes,
        "total spike count divergence"
    );

    // --- step counter -------------------------------------------------------
    assert_eq!(
        layer.step_count(),
        num_steps as i64,
        "step_count() must equal the number of replayed frames"
    );
}

/// Assert the final membrane and adaptation banks match the fixture for every
/// neuron, exact bits.
fn assert_final_state(layer: &SparseGifHiddenLayer, fixture: &Fixture) {
    for (i, &value) in layer.membrane().iter().enumerate() {
        let got = value.to_bits();
        let want = fixture.final_membrane_bits[i];
        assert_eq!(
            got, want,
            "final membrane divergence at neuron {i}: got bits {got:#010x}, want bits {want:#010x}"
        );
    }
    for (i, &value) in layer.adaptation().iter().enumerate() {
        let got = value.to_bits();
        let want = fixture.final_adaptation_bits[i];
        assert_eq!(
            got, want,
            "final adaptation divergence at neuron {i}: got bits {got:#010x}, want bits {want:#010x}"
        );
    }
}

#[test]
fn corinth_gif_parity_is_bit_exact() {
    let fixture: Fixture =
        serde_json::from_str(FIXTURE_JSON).expect("corinth_gif_parity.json must be valid JSON");

    // --- provenance: the vectors describe the pinned Corinth revision -------
    assert_provenance(&fixture);

    // --- dimensions + internal array-size consistency -----------------------
    assert_dimensions(&fixture);

    // --- exhaustive fired-ID oracle self-consistency ------------------------
    assert_oracle_self_consistent(&fixture);

    // --- default GifParams bits ---------------------------------------------
    let params = assert_param_bits(&fixture);

    // --- reconstruct topology in fixture (Corinth edge) order ---------------
    let rows = rebuild_topology(&fixture);
    let mut layer =
        SparseGifHiddenLayer::from_topology(fixture.dimensions.num_inputs, params, &rows)
            .expect("reconstructing Corinth topology must succeed");

    // --- replay all 512 frames via a REUSABLE dense 0/1 buffer --------------
    replay_and_assert(&mut layer, &fixture);

    // --- final membrane / adaptation, every neuron, exact bits -------------
    assert_final_state(&layer, &fixture);
}
