//! Parameter and dynamics smoke tests for the NIR -> neuromod IF handoff.
//!
//! These build [`IfHandoff`] banks from the vendored fixtures' IF nodes, assert
//! the mapped parameters (bank sizes, threshold bit patterns, resistance, reset)
//! and then step at least one bank through the **real** neuromod
//! `integrate` / `check_for_spike` path, comparing spike trains and post-step
//! membrane potentials against an inline, pure-arithmetic forward-Euler
//! reference.
//!
//! This is a *smoke* test of interoperability, not an equivalence proof: it does
//! NOT claim bit-identical numerical equivalence with NeuroCUDA. It only checks
//! that HF-derived IF parameters map into neuromod and that the neuromod IF step
//! matches the documented forward-Euler contract (Assumption 5: forward Euler,
//! `dt = 1`, `stimulus = r * I`, `v0 = 0`, spike on `v >= threshold`, hard reset
//! to `0`).

use nir_huggingface_interop::{CNN_FIXTURE, IfHandoff, MLP_FIXTURE, fixture_path, load_fixture};
use nir_rs::NirNode;
use nir_rs::nodes::If;

/// Fetch a named IF node from a fixture, or panic if it is missing / not an IF.
fn if_node<'a>(graph: &'a nir_rs::NirGraph, name: &str) -> &'a If {
    match graph.get(name) {
        Some(NirNode::If(if_node)) => if_node,
        Some(other) => panic!("node `{name}` is {}, not IF", other.type_name()),
        None => panic!("node `{name}` not found"),
    }
}

/// Build an [`IfHandoff`] for a named IF node in a fixture.
fn handoff(fixture: &str, node: &str) -> IfHandoff {
    let graph = load_fixture(fixture_path(fixture)).expect("fixture loads offline");
    IfHandoff::from_node(node, if_node(&graph, node)).expect("IF node maps into neuromod")
}

#[test]
fn mlp_if_banks_map_with_expected_parameters() {
    let graph = load_fixture(fixture_path(MLP_FIXTURE)).expect("MLP fixture loads offline");

    let if1 = IfHandoff::from_node("if1", if_node(&graph, "if1")).expect("MLP if1 maps");
    let if2 = IfHandoff::from_node("if2", if_node(&graph, "if2")).expect("MLP if2 maps");

    // Bank size: MLP if1 has 256 neurons (r shape [256]).
    assert_eq!(if1.len(), 256, "MLP if1 bank size");
    assert_eq!(if1.shape(), &[256], "MLP if1 shape");

    // Threshold bit patterns (exact IEEE-754 f32 bits, no float compare).
    assert_eq!(
        if1.bank()[0].threshold.to_bits(),
        0x409f_2a4d,
        "MLP if1 v_threshold[0] bits"
    );
    assert_eq!(
        if2.bank()[0].threshold.to_bits(),
        0x4071_05cb,
        "MLP if2 v_threshold[0] bits"
    );

    // r[0] == 1.0 exactly.
    assert_eq!(if1.r()[0], 1.0, "MLP if1 r[0]");

    // Reset value is 0: neuromod maps the (absent / zero) reset to v0 = 0, and
    // decay_rate 0 gives a pure integrate-and-fire step.
    assert_eq!(
        if1.bank()[0].membrane_potential,
        0.0,
        "MLP if1 initial/reset potential"
    );
    assert_eq!(if1.bank()[0].decay_rate, 0.0, "MLP if1 decay_rate");
    // threshold and base_threshold agree.
    assert_eq!(
        if1.bank()[0].threshold,
        if1.bank()[0].base_threshold,
        "MLP if1 threshold == base_threshold"
    );
}

#[test]
fn cnn_if1_bank_size_and_threshold() {
    let graph = load_fixture(fixture_path(CNN_FIXTURE)).expect("CNN fixture loads offline");
    let if1 = IfHandoff::from_node("if1", if_node(&graph, "if1")).expect("CNN if1 maps");

    // Bank size: 32 * 34 * 34 = 36992 (r shape [32, 34, 34]).
    assert_eq!(if1.len(), 32 * 34 * 34, "CNN if1 bank size");
    assert_eq!(if1.shape(), &[32, 34, 34], "CNN if1 shape");
    assert_eq!(
        if1.bank()[0].threshold.to_bits(),
        0x3ff1_b57e,
        "CNN if1 v_threshold[0] bits"
    );
}

/// Pure-arithmetic forward-Euler IF reference (Assumption 5), computed inline so
/// the test does not lean on the harness's own step implementation for the
/// oracle. Returns per-step spike trains and final membrane potentials.
struct EulerRef {
    /// `spikes[step][element]`.
    spikes: Vec<Vec<bool>>,
    /// Final membrane potential per element.
    potentials: Vec<f32>,
}

fn euler_reference(r: &[f32], thresholds: &[f32], currents: &[f32], steps: usize) -> EulerRef {
    let n = r.len();
    let mut v = vec![0.0f32; n];
    let mut spikes = Vec::with_capacity(steps);
    for _ in 0..steps {
        let mut row = Vec::with_capacity(n);
        for i in 0..n {
            v[i] += r[i] * currents[i];
            let fired = v[i] >= thresholds[i];
            if fired {
                v[i] = 0.0;
            }
            row.push(fired);
        }
        spikes.push(row);
    }
    EulerRef {
        spikes,
        potentials: v,
    }
}

#[test]
fn mlp_if1_steps_match_forward_euler_reference() {
    let graph = load_fixture(fixture_path(MLP_FIXTURE)).expect("MLP fixture loads offline");
    let mut handoff = IfHandoff::from_node("if1", if_node(&graph, "if1")).expect("MLP if1 maps");

    let n = handoff.len();
    let r: Vec<f32> = handoff.r().to_vec();
    let thresholds: Vec<f32> = handoff
        .bank()
        .iter()
        .map(|neuron| neuron.threshold)
        .collect();

    // Deterministic, finite, per-element constant currents. Chosen so that
    // r[i] * I[i] is a clean fraction of the threshold and v never lands
    // exactly on the threshold at any step (the neuron fires on v >= thr, so
    // exact equality would make the analytic first-spike step ambiguous).
    // step = thr[i] / 2.71 gives a non-terminating increment per step.
    let currents: Vec<f32> = (0..n)
        .map(|i| {
            let ri = r[i];
            assert!(ri != 0.0, "r[{i}] must be non-zero for a clean current");
            (thresholds[i] / 2.71) / ri
        })
        .collect();

    let steps = 8usize;
    let reference = euler_reference(&r, &thresholds, &currents, steps);

    let mut got_spikes = Vec::with_capacity(steps);
    for t in 0..steps {
        let row = handoff
            .step(&currents, t as i64)
            .expect("MLP if1 steps without runtime error");
        got_spikes.push(row);
    }

    assert_eq!(
        got_spikes, reference.spikes,
        "MLP if1 spike trains must match the forward-Euler reference"
    );

    for (i, neuron) in handoff.bank().iter().enumerate() {
        assert_eq!(
            neuron.membrane_potential.to_bits(),
            reference.potentials[i].to_bits(),
            "MLP if1 element {i} post-step membrane potential"
        );
    }

    // Analytic first-spike step for a few selected elements. With v starting
    // at 0 and a constant per-step increment `d = r*I`, the neuron first
    // reaches `v >= threshold` at step ceil(threshold / d) (1-indexed), i.e.
    // spike index ceil(thr / d) - 1 in the 0-indexed train.
    for &i in &[0usize, 1, 7, 42, 128, 255] {
        let d = r[i] * currents[i];
        let analytic_step_1indexed = (thresholds[i] / d).ceil() as usize;
        assert!(analytic_step_1indexed >= 1);
        let first_spike = (0..steps).find(|&t| got_spikes[t][i]);
        if analytic_step_1indexed <= steps {
            assert_eq!(
                first_spike,
                Some(analytic_step_1indexed - 1),
                "MLP if1 element {i}: first-spike step should be ceil(thr/(r*I)) = {analytic_step_1indexed}"
            );
        } else {
            assert_eq!(
                first_spike, None,
                "MLP if1 element {i}: should not spike within {steps} steps"
            );
        }
    }
}

/// The remaining IF banks (MLP if2, CNN if1/if2/if3) just need to step without
/// panicking and without a `Runtime` error under finite currents. CNN if1 has
/// ~37k neurons, so keep the step count tiny.
#[test]
fn other_if_banks_step_without_runtime_error() {
    // (fixture, node, steps).
    let cases = [
        (MLP_FIXTURE, "if2", 4usize),
        (CNN_FIXTURE, "if1", 2usize),
        (CNN_FIXTURE, "if2", 2usize),
        (CNN_FIXTURE, "if3", 2usize),
    ];

    for (fixture, node, steps) in cases {
        let mut handoff = handoff(fixture, node);
        let n = handoff.len();
        assert!(n > 0, "{fixture} {node} bank should be non-empty");

        // A small finite constant current for every element.
        let currents = vec![0.25f32; n];
        for t in 0..steps {
            let result = handoff.step(&currents, t as i64);
            assert!(
                result.is_ok(),
                "{fixture} {node} step {t} returned an error: {:?}",
                result.err()
            );
        }
    }
}
