use std::time::Instant;

use neuromod::{NeuroModulators, SeedableRng, SpikingNetwork, StdRng};

const CHANNELS: usize = 16;
const LIF_NEURONS: usize = 16;
const IZH_NEURONS: usize = 5;
const WEIGHT_BUDGET: f32 = 2.0;
const L1_EPSILON: f32 = 1.0e-4;

#[derive(Debug, PartialEq, Eq)]
struct EngineCapacities {
    neurons: usize,
    iz_neurons: usize,
    input_spike_times: usize,
    predictive_state: usize,
    weights: Vec<usize>,
    eligibility: Vec<usize>,
}

impl EngineCapacities {
    fn capture(network: &SpikingNetwork) -> Self {
        Self {
            neurons: network.neurons.capacity(),
            iz_neurons: network.iz_neurons.capacity(),
            input_spike_times: network.input_spike_times.capacity(),
            predictive_state: network.predictive_state.capacity(),
            weights: network
                .neurons
                .iter()
                .map(|neuron| neuron.weights.capacity())
                .collect(),
            eligibility: network
                .neurons
                .iter()
                .map(|neuron| neuron.eligibility.capacity())
                .collect(),
        }
    }
}

fn linux_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

fn run_engine_soak(steps: usize) {
    let mut network = SpikingNetwork::with_dimensions(LIF_NEURONS, IZH_NEURONS, CHANNELS);
    // Start off-budget so a no-op learning/normalize path cannot vacuously pass.
    for neuron in &mut network.neurons {
        neuron.weights.fill(0.0);
    }
    let initial_weights: Vec<Vec<f32>> = network
        .neurons
        .iter()
        .map(|neuron| neuron.weights.clone())
        .collect();

    let initial_capacities = EngineCapacities::capture(&network);
    let report_rss = std::env::var_os("NEUROMOD_SOAK_RSS").is_some();
    let initial_rss = report_rss.then(linux_rss_kib).flatten();
    let modulators = NeuroModulators {
        dopamine: 0.5,
        ..NeuroModulators::default()
    };
    let mut rng = StdRng::seed_from_u64(0x5A17_1338);
    let mut stimuli = [0.0; CHANNELS];
    let started = Instant::now();

    for step in 0..steps {
        for (channel, value) in stimuli.iter_mut().enumerate() {
            *value = usize::from((step + channel) % 4 == 0) as f32;
        }
        network
            .step_with_rng(&stimuli, &modulators, &mut rng)
            .expect("fixed-width finite inputs must step");
    }

    let elapsed = started.elapsed();
    eprintln!("engine soak: {steps} steps in {elapsed:?}");
    if report_rss {
        eprintln!(
            "engine soak RSS diagnostic (not asserted): initial={initial_rss:?} KiB final={:?} KiB",
            linux_rss_kib()
        );
    }

    assert_eq!(network.global_step, steps as i64);
    assert_eq!(EngineCapacities::capture(&network), initial_capacities);
    assert!(
        network
            .predictive_state
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(network.neurons.iter().all(|neuron| {
        neuron.membrane_potential.is_finite()
            && neuron.decay_rate.is_finite()
            && neuron.threshold.is_finite()
            && neuron.base_threshold.is_finite()
            && neuron.weights.iter().all(|weight| weight.is_finite())
            && neuron
                .eligibility
                .iter()
                .all(|trace| trace.value.is_finite() && trace.tau.is_finite())
    }));
    assert!(network.iz_neurons.iter().all(|neuron| {
        neuron.v.is_finite()
            && neuron.u.is_finite()
            && neuron.a.is_finite()
            && neuron.b.is_finite()
            && neuron.c.is_finite()
            && neuron.d.is_finite()
    }));

    assert!(
        network
            .neurons
            .iter()
            .zip(initial_weights.iter())
            .any(|(neuron, before)| neuron.weights.as_slice() != before.as_slice()),
        "rewarded soak must change at least one weight (non-vacuous learning path)"
    );

    for (index, neuron) in network.neurons.iter().enumerate() {
        let l1: f32 = neuron.weights.iter().map(|weight| weight.abs()).sum();
        assert!(
            (l1 - WEIGHT_BUDGET).abs() <= L1_EPSILON,
            "neuron {index} L1 sum {l1} exceeded budget epsilon {L1_EPSILON}"
        );
    }
}

#[test]
fn soak_engine_10k_steps() {
    run_engine_soak(10_000);
}

#[test]
#[ignore = "million-step reliability gate; run with `cargo test soak -- --ignored --nocapture`"]
fn soak_engine_million_steps() {
    run_engine_soak(1_000_000);
}

