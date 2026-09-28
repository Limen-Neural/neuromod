// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Offline generator for `tests/reference/corinth_gif_parity.json`.
//
// It runs the audited 512-step x 2048-neuron case through the UNMODIFIED
// `SparseGifHiddenLayer::run` dynamics vendored (arithmetic-identical) from the
// pinned `rmems/corinth-canal` revision, then serializes a compact parity
// fixture: provenance, dimensions/counts, GIF parameter f32 bit patterns,
// deduplicated hex input masks + per-step mask indices, per-step fired IDs and
// counts, selected raster rows, ordered per-neuron fan-in topology (in Corinth
// EDGE ORDER), and the final membrane/adaptation banks as f32 bit patterns.
//
// Determinism: no RNG, no wall clock, no hashing, no iteration over unordered
// containers. Running it twice produces byte-identical output.
//
// Regenerate with:
//   cd tests/reference/corinth_gif
//   cargo run --release > ../corinth_gif_parity.json
// (see corinth_gif_parity.NOTICE.md).

mod funnel_vendored;

use funnel_vendored::{FUNNEL_HIDDEN_NEURONS, FUNNEL_INPUT_NEURONS, GIF_FAN_IN, SparseGifHiddenLayer};
use serde_json::{Map, Value, json};

// ---- Audited-case dimensions -------------------------------------------------
const NUM_STEPS: usize = 512;

// ---- Provenance constants (see corinth_gif_parity.NOTICE.md) -----------------
const CORINTH_SOURCE_COMMIT: &str = "8e54e234ac005dd84e4ad2bedbf9f5bceb082355";
const CORINTH_FUNNEL_RS_SHA256: &str =
    "10192537a1a096fc8ec8a9a87740b643624b64c9fc3f3408faf06653b6694b47";
const HISTORICAL_AUDITED_NEUROMOD_COMMIT: &str =
    "263ec19e807454eac943681993623989fb986cc6";
const CORINTH_REPO: &str = "https://github.com/rmems/corinth-canal";
const CORINTH_LICENSE: &str = "Apache-2.0 OR MIT";
const CORINTH_COPYRIGHT: &str = "Copyright (c) 2026 Raul Montoya Cardenas and contributors";

/// Deterministic 512-step input spike train over the 2048-input space.
///
/// Rule (documented verbatim in the NOTICE): input index `i` is ACTIVE at
/// step `t` iff `(i * 31 + t * 17) % 23 == 0`. This is a fixed integer
/// congruence with no RNG or state, so it is reproducible on any platform. It
/// yields a moderately sparse, time-varying activation that drives the audited
/// dynamics across all 512 steps.
fn deterministic_input_train() -> Vec<Vec<usize>> {
    (0..NUM_STEPS)
        .map(|t| {
            (0..FUNNEL_INPUT_NEURONS)
                .filter(|&i| (i * 31 + t * 17) % 23 == 0)
                .collect::<Vec<usize>>()
        })
        .collect()
}

/// Encode an active-index list over `FUNNEL_INPUT_NEURONS` channels as a
/// lowercase hex bit-string (little-endian bit order within each byte:
/// channel `i` -> byte `i / 8`, bit `i % 8`). Length is
/// `FUNNEL_INPUT_NEURONS / 8` bytes = 256 bytes = 512 hex chars.
fn mask_to_hex(active: &[usize]) -> String {
    let mut bytes = vec![0u8; FUNNEL_INPUT_NEURONS / 8];
    for &i in active {
        if i < FUNNEL_INPUT_NEURONS {
            bytes[i / 8] |= 1u8 << (i % 8);
        }
    }
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

/// Deterministic selected-step rule for exported raster rows: every 64th step
/// (0, 64, 128, ..., 448) plus the final step (NUM_STEPS - 1 = 511).
fn selected_steps() -> Vec<usize> {
    let mut steps: Vec<usize> = (0..NUM_STEPS).step_by(64).collect();
    let last = NUM_STEPS - 1;
    if !steps.contains(&last) {
        steps.push(last);
    }
    steps
}

fn main() {
    // 1. Build the input train and run the UNMODIFIED vendored dynamics.
    let input_train = deterministic_input_train();
    let mut layer = SparseGifHiddenLayer::new();
    let (spike_train, _potentials, _iz) = layer.run(&input_train);

    // 2. Deduplicate input masks: build a mask table (unique hex strings) plus a
    //    per-step index into that table. The rule is deterministic so many steps
    //    share a mask; storing each unique mask once keeps the fixture compact.
    let mut mask_table: Vec<String> = Vec::new();
    let mut mask_index_of: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut per_step_mask_index: Vec<usize> = Vec::with_capacity(NUM_STEPS);
    for step in &input_train {
        let hex = mask_to_hex(step);
        let idx = *mask_index_of.entry(hex.clone()).or_insert_with(|| {
            mask_table.push(hex.clone());
            mask_table.len() - 1
        });
        per_step_mask_index.push(idx);
    }

    // 3. Per-step spike counts + total. Full per-step fired-ID frames are NOT
    //    dumped (that would be 512 full rasters); the fired IDs are preserved at
    //    the deterministically selected steps (see `selected_raster_rows`) and
    //    the total + per-step counts pin the aggregate spike behavior. FEAT-002
    //    reconstructs the full raster by replaying the fixture through neuromod.
    let mut per_step_spike_count: Vec<usize> = Vec::with_capacity(NUM_STEPS);
    let mut total_spikes: usize = 0;
    for fired in &spike_train {
        per_step_spike_count.push(fired.len());
        total_spikes += fired.len();
    }

    // 4. Selected raster rows (fired IDs at the documented selected steps).
    let sel = selected_steps();
    let selected_rows: Vec<Value> = sel
        .iter()
        .map(|&t| {
            json!({
                "step": t,
                "fired_ids": spike_train[t].iter().map(|&id| json!(id)).collect::<Vec<_>>(),
                "spike_count": spike_train[t].len(),
            })
        })
        .collect();

    // 5. Ordered per-neuron fan-in topology in Corinth EDGE ORDER (indices[0..4],
    //    NOT sorted), as (source, weight_bits) rows. `weight_bits` are f32
    //    `to_bits()` so the fixture carries exact bit patterns, not decimals.
    //    Each row is encoded as a FLAT integer array in edge order:
    //    [source_0, weight_bits_0, source_1, weight_bits_1, ...] with
    //    GIF_FAN_IN (source, weight_bits) pairs. This keeps the ordering
    //    (source then its weight, edge 0..fan_in) unambiguous while staying
    //    compact. `weight_bits` are f32 `to_bits()` (exact bit patterns).
    let mut topology: Vec<Value> = Vec::with_capacity(FUNNEL_HIDDEN_NEURONS);
    for hidden in 0..FUNNEL_HIDDEN_NEURONS {
        let indices = layer.weight_indices(hidden);
        let values = layer.weight_values(hidden);
        let mut row: Vec<Value> = Vec::with_capacity(GIF_FAN_IN * 2);
        for edge in 0..GIF_FAN_IN {
            row.push(json!(indices[edge]));
            row.push(json!(values[edge].to_bits()));
        }
        topology.push(Value::Array(row));
    }

    // 6. GIF parameter defaults as f32 BIT patterns. These are the exact
    //    constants the vendored `SparseGifHiddenLayer` uses (fields + the two
    //    literals `0.05` adaptation_coupling and `1.0` adaptation_increment in
    //    the `run` loop). They match neuromod `GifParams::default()` bit-for-bit.
    let mut param_bits = Map::new();
    param_bits.insert("leak".into(), json!(0.92f32.to_bits()));
    param_bits.insert("drive_scale".into(), json!(0.75f32.to_bits()));
    param_bits.insert("base_threshold".into(), json!(0.65f32.to_bits()));
    param_bits.insert("adaptation_scale".into(), json!(0.22f32.to_bits()));
    param_bits.insert("adaptation_decay".into(), json!(0.94f32.to_bits()));
    param_bits.insert("adaptation_coupling".into(), json!(0.05f32.to_bits()));
    param_bits.insert("adaptation_increment".into(), json!(1.0f32.to_bits()));
    param_bits.insert("reset_ratio".into(), json!(0.35f32.to_bits()));

    // 7. Final membrane / adaptation banks as f32 bit patterns for all neurons.
    let final_membrane_bits: Vec<Value> = layer
        .membrane()
        .iter()
        .map(|v| json!(v.to_bits()))
        .collect();
    let final_adaptation_bits: Vec<Value> = layer
        .adaptation()
        .iter()
        .map(|v| json!(v.to_bits()))
        .collect();

    // 8. Assemble the fixture.
    let fixture = json!({
        "provenance": {
            "corinth_repo": CORINTH_REPO,
            "corinth_source_commit": CORINTH_SOURCE_COMMIT,
            "corinth_funnel_rs_sha256": CORINTH_FUNNEL_RS_SHA256,
            "historical_audited_neuromod_commit": HISTORICAL_AUDITED_NEUROMOD_COMMIT,
            "license": CORINTH_LICENSE,
            "copyright": CORINTH_COPYRIGHT,
            "generated_by": "tests/reference/corinth_gif (corinth-gif-parity-gen)",
            "input_rule": "input index i is active at step t iff (i*31 + t*17) % 23 == 0",
            "selected_step_rule": "every 64th step (0,64,...,448) plus the final step 511",
            "mask_hex_encoding": "little-endian bits within each byte: channel i -> byte i/8, bit i%8; 256 bytes = 512 hex chars over 2048 channels",
            "topology_encoding": "topology_edge_order[n] is a flat array [src0,wbits0,src1,wbits1,...] of fan_in (source, weight_bits) pairs in Corinth edge order (NOT sorted); weight_bits are f32 to_bits()",
        },
        "dimensions": {
            "num_steps": NUM_STEPS,
            "num_neurons": FUNNEL_HIDDEN_NEURONS,
            "num_inputs": FUNNEL_INPUT_NEURONS,
            "fan_in": GIF_FAN_IN,
        },
        "counts": {
            "total_spikes": total_spikes,
            "num_unique_input_masks": mask_table.len(),
        },
        "param_bits": Value::Object(param_bits),
        "input_masks_hex": mask_table,
        "per_step_mask_index": per_step_mask_index,
        "per_step_spike_count": per_step_spike_count,
        "selected_raster_rows": selected_rows,
        "topology_edge_order": topology,
        "final_membrane_bits": final_membrane_bits,
        "final_adaptation_bits": final_adaptation_bits,
    });

    // Compact, deterministic serialization (no trailing newline variance):
    // serde_json orders map keys as inserted for `Value::Object`? No — serde_json
    // preserves insertion order only with the `preserve_order` feature; by default
    // it sorts `Map` keys via BTreeMap, which is ALSO fully deterministic. Either
    // way the output is stable across runs.
    let out = serde_json::to_string(&fixture).expect("serialize fixture");
    print!("{out}");
}
