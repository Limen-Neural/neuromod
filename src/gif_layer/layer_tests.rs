use super::*;
use crate::gif::GifNeuron;
use std::time::Instant;

fn config(num_inputs: usize, num_neurons: usize, fan_in: usize, seed: u64) -> SparseGifLayerConfig {
    SparseGifLayerConfig {
        num_inputs,
        num_neurons,
        fan_in,
        seed,
        ..Default::default()
    }
}

fn ramp_train(num_steps: usize, num_inputs: usize) -> Vec<Vec<f32>> {
    (0..num_steps)
        .map(|t| {
            (0..num_inputs)
                .map(|c| ((t + c) % 5) as f32 * 0.25)
                .collect()
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
struct LayerCapacities {
    fan_in_offsets: usize,
    fan_in_sources: usize,
    weights: usize,
    membrane: usize,
    adaptation: usize,
    last_spike_time: usize,
}

impl LayerCapacities {
    fn capture(layer: &SparseGifHiddenLayer) -> Self {
        Self {
            fan_in_offsets: layer.fan_in_offsets.capacity(),
            fan_in_sources: layer.fan_in_sources.capacity(),
            weights: layer.weights.capacity(),
            membrane: layer.membrane.capacity(),
            adaptation: layer.adaptation.capacity(),
            last_spike_time: layer.last_spike_time.capacity(),
        }
    }
}

fn run_gif_layer_soak(steps: usize) {
    const INPUTS: usize = 16;
    const NEURONS: usize = 8;
    let mut layer = SparseGifHiddenLayer::new(&config(INPUTS, NEURONS, 4, 0x5A17_1338)).unwrap();
    let initial_capacities = LayerCapacities::capture(&layer);
    let mut stimuli = [0.0; INPUTS];
    let mut spikes = [false; NEURONS];
    let started = Instant::now();

    for step in 0..steps {
        for (channel, value) in stimuli.iter_mut().enumerate() {
            *value = usize::from((step + channel) % 4 == 0) as f32;
        }
        layer.step_into(&stimuli, &mut spikes).unwrap();
    }

    eprintln!("GIF layer soak: {steps} steps in {:?}", started.elapsed());
    assert_eq!(layer.step_count(), steps as i64);
    assert_eq!(LayerCapacities::capture(&layer), initial_capacities);
    assert!(layer.weights().iter().all(|value| value.is_finite()));
    assert!(layer.membrane().iter().all(|value| value.is_finite()));
    assert!(layer.adaptation().iter().all(|value| value.is_finite()));
}

#[test]
fn soak_gif_layer_10k_steps() {
    run_gif_layer_soak(10_000);
}

#[test]
#[ignore = "million-step reliability gate; run with `cargo test soak -- --ignored --nocapture`"]
fn soak_gif_layer_million_steps() {
    run_gif_layer_soak(1_000_000);
}

// --- structure -------------------------------------------------------

#[test]
fn csr_shape_is_consistent() {
    let layer = SparseGifHiddenLayer::new(&config(32, 8, 4, 1)).unwrap();
    assert_eq!(layer.num_synapses(), 32);
    assert_eq!(layer.weights().len(), layer.num_synapses());
    assert_eq!(layer.membrane().len(), 8);
    assert_eq!(layer.adaptation().len(), 8);
    assert_eq!(layer.last_spike_time().len(), 8);
    for n in 0..8 {
        let (sources, weights) = layer.fan_in_of(n).unwrap();
        assert_eq!(sources.len(), 4);
        assert_eq!(weights.len(), 4);
    }
    assert!(layer.fan_in_of(8).is_none());
}

#[test]
fn fan_in_sources_are_distinct_sorted_and_in_range() {
    let layer = SparseGifHiddenLayer::new(&config(24, 16, 6, 0xDEAD_BEEF)).unwrap();
    for n in 0..layer.num_neurons() {
        let (sources, _) = layer.fan_in_of(n).unwrap();
        assert!(
            sources.windows(2).all(|w| w[0] < w[1]),
            "row {n} not strictly ascending: {sources:?}"
        );
        assert!(sources.iter().all(|&s| (s as usize) < 24));
    }
}

#[test]
fn generated_weights_lie_in_range() {
    let mut cfg = config(16, 16, 8, 5);
    cfg.weight_range = (-0.25, 0.75);
    let layer = SparseGifHiddenLayer::new(&cfg).unwrap();
    assert!(layer.weights().iter().all(|&w| (-0.25..0.75).contains(&w)));
}

// --- determinism -----------------------------------------------------

#[test]
fn same_seed_same_layer() {
    let a = SparseGifHiddenLayer::new(&config(64, 32, 8, 42)).unwrap();
    let b = SparseGifHiddenLayer::new(&config(64, 32, 8, 42)).unwrap();
    assert_eq!(a, b);
}

#[test]
fn different_seed_different_topology() {
    let a = SparseGifHiddenLayer::new(&config(64, 32, 8, 42)).unwrap();
    let b = SparseGifHiddenLayer::new(&config(64, 32, 8, 43)).unwrap();

    // Compare the generated arrays, not the layers themselves. `PartialEq`
    // includes the `seed` field, so `assert_ne!(a, b)` would pass even for a
    // generator that ignored the seed entirely.
    let rows = |l: &SparseGifHiddenLayer| -> Vec<Vec<u32>> {
        (0..l.num_neurons())
            .map(|n| l.fan_in_of(n).unwrap().0.to_vec())
            .collect()
    };
    assert_ne!(
        rows(&a),
        rows(&b),
        "distinct seeds must produce distinct fan-in topology"
    );
    assert_ne!(
        a.weights(),
        b.weights(),
        "distinct seeds must produce distinct initial weights"
    );
}

#[test]
fn topology_prefix_is_stable_when_the_layer_grows() {
    // Per-neuron sub-streams mean neuron n's fan-in must not depend on how
    // many neurons follow it.
    let small = SparseGifHiddenLayer::new(&config(48, 4, 5, 9)).unwrap();
    let large = SparseGifHiddenLayer::new(&config(48, 40, 5, 9)).unwrap();
    for n in 0..small.num_neurons() {
        assert_eq!(small.fan_in_of(n), large.fan_in_of(n), "row {n} drifted");
    }
}

#[test]
fn run_is_reproducible() {
    let train = ramp_train(40, 32);
    let mut a = SparseGifHiddenLayer::new(&config(32, 12, 6, 7)).unwrap();
    let mut b = SparseGifHiddenLayer::new(&config(32, 12, 6, 7)).unwrap();
    assert_eq!(a.run(&train).unwrap(), b.run(&train).unwrap());
    // ... and a reset layer reproduces its own first pass.
    let first = {
        let mut c = SparseGifHiddenLayer::new(&config(32, 12, 6, 7)).unwrap();
        c.run(&train).unwrap()
    };
    a.reset();
    assert_eq!(a.run(&train).unwrap(), first);
}

#[test]
fn run_matches_stepwise_execution() {
    let train = ramp_train(25, 20);
    let mut batched = SparseGifHiddenLayer::new(&config(20, 10, 4, 3)).unwrap();
    let mut stepped = SparseGifHiddenLayer::new(&config(20, 10, 4, 3)).unwrap();

    let raster = batched.run(&train).unwrap();
    for (t, frame) in train.iter().enumerate() {
        assert_eq!(stepped.step(frame).unwrap(), raster.fired_at(t), "step {t}");
    }
    assert_eq!(batched.membrane(), stepped.membrane());
    assert_eq!(batched.adaptation(), stepped.adaptation());
}

#[test]
fn run_accepts_slice_frames() {
    let owned = ramp_train(6, 8);
    let borrowed: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
    let mut a = SparseGifHiddenLayer::new(&config(8, 4, 2, 11)).unwrap();
    let mut b = SparseGifHiddenLayer::new(&config(8, 4, 2, 11)).unwrap();
    assert_eq!(a.run(&owned).unwrap(), b.run(&borrowed).unwrap());
}

// --- parity with the single-neuron model -----------------------------

#[test]
fn layer_matches_gif_neuron_exactly() {
    // A one-neuron, one-synapse layer must reproduce `GifNeuron` bit for
    // bit — this is what makes `GifParams` the single source of truth.
    let params = GifParams::default();
    let layer_rows = vec![vec![(0usize, 1.0f32)]];
    let mut layer = SparseGifHiddenLayer::from_topology(1, params, &layer_rows).unwrap();
    let mut neuron = GifNeuron::new();

    for t in 0..120i64 {
        let stimulus = if t % 3 == 0 { 0.9 } else { 0.1 };
        let layer_fired = !layer.step(&[stimulus]).unwrap().is_empty();

        neuron.integrate(stimulus);
        let neuron_fired = neuron.check_for_spike(t);

        assert_eq!(layer_fired, neuron_fired, "spike mismatch at t={t}");
        assert_eq!(layer.membrane()[0], neuron.membrane_potential, "v at t={t}");
        assert_eq!(layer.adaptation()[0], neuron.adaptation, "w at t={t}");
        assert_eq!(layer.last_spike_time()[0], neuron.last_spike_time);
    }
}

// --- edge cases ------------------------------------------------------

#[test]
fn zero_fan_in_never_fires() {
    let mut layer = SparseGifHiddenLayer::new(&config(16, 4, 0, 1)).unwrap();
    assert_eq!(layer.num_synapses(), 0);
    let raster = layer.run(&ramp_train(50, 16)).unwrap();
    assert_eq!(raster.total_spikes(), 0);
    assert!(layer.membrane().iter().all(|&v| v == 0.0));
}

#[test]
fn zero_input_never_fires() {
    let mut layer = SparseGifHiddenLayer::new(&config(16, 8, 8, 2)).unwrap();
    let train = vec![vec![0.0f32; 16]; 60];
    assert_eq!(layer.run(&train).unwrap().total_spikes(), 0);
}

#[test]
fn fully_dense_fan_in_selects_every_channel() {
    let layer = SparseGifHiddenLayer::new(&config(12, 6, 12, 4)).unwrap();
    for n in 0..6 {
        let (sources, _) = layer.fan_in_of(n).unwrap();
        let expected: Vec<u32> = (0..12).collect();
        assert_eq!(sources, expected.as_slice(), "dense row {n}");
    }
}

#[test]
fn single_neuron_single_input() {
    let mut layer = SparseGifHiddenLayer::new(&config(1, 1, 1, 8)).unwrap();
    assert_eq!(layer.fan_in_of(0).unwrap().0, &[0]);
    let raster = layer.run(&vec![vec![5.0f32]; 10]).unwrap();
    assert_eq!(raster.num_neurons(), 1);
    assert!(raster.total_spikes() > 0, "strong drive should fire");
}

#[test]
fn empty_layer_and_empty_train_are_benign() {
    let mut layer = SparseGifHiddenLayer::new(&config(8, 0, 0, 1)).unwrap();
    let raster = layer.run(&ramp_train(5, 8)).unwrap();
    assert_eq!(raster.num_neurons(), 0);
    assert_eq!(raster.total_spikes(), 0);

    let mut normal = SparseGifHiddenLayer::new(&config(8, 3, 2, 1)).unwrap();
    let empty: Vec<Vec<f32>> = Vec::new();
    let raster = normal.run(&empty).unwrap();
    assert_eq!(raster.num_steps(), 0);
    assert_eq!(normal.step_count(), 0);
}

#[test]
fn reset_clears_state_but_not_topology() {
    let mut layer = SparseGifHiddenLayer::new(&config(16, 5, 4, 6)).unwrap();
    let before = layer.weights().to_vec();
    layer.run(&ramp_train(30, 16)).unwrap();
    layer.reset();
    assert_eq!(layer.step_count(), 0);
    assert!(layer.membrane().iter().all(|&v| v == 0.0));
    assert!(layer.adaptation().iter().all(|&w| w == 0.0));
    assert!(layer.last_spike_time().iter().all(|&t| t == -1));
    assert_eq!(layer.weights(), before.as_slice());
}

// --- errors ----------------------------------------------------------

#[test]
fn fan_in_larger_than_inputs_is_rejected() {
    assert_eq!(
        SparseGifHiddenLayer::new(&config(4, 2, 5, 0)).unwrap_err(),
        GifLayerError::FanInExceedsInputs {
            fan_in: 5,
            num_inputs: 4
        }
    );
}

#[test]
fn invalid_weight_range_is_rejected() {
    let mut cfg = config(8, 2, 2, 0);
    cfg.weight_range = (1.0, 0.0);
    assert!(matches!(
        SparseGifHiddenLayer::new(&cfg),
        Err(GifLayerError::InvalidWeightRange { .. })
    ));
    cfg.weight_range = (0.0, f32::NAN);
    assert!(matches!(
        SparseGifHiddenLayer::new(&cfg),
        Err(GifLayerError::InvalidWeightRange { .. })
    ));
}

#[test]
fn wrong_frame_width_is_rejected() {
    let mut layer = SparseGifHiddenLayer::new(&config(8, 2, 2, 0)).unwrap();
    assert_eq!(
        layer.step(&[0.0; 7]).unwrap_err(),
        GifLayerError::InputLenMismatch {
            expected: 8,
            got: 7
        }
    );
    assert!(layer.run(&[vec![0.0f32; 3]]).is_err());
}

#[test]
fn out_of_range_explicit_source_is_rejected() {
    let rows = vec![vec![(0usize, 1.0f32)], vec![(9usize, 1.0f32)]];
    assert_eq!(
        SparseGifHiddenLayer::from_topology(4, GifParams::default(), &rows).unwrap_err(),
        GifLayerError::SourceOutOfRange {
            neuron: 1,
            source: 9,
            num_inputs: 4
        }
    );
}

// --- numeric and addressability guards -------------------------------

#[test]
fn wide_but_finite_weight_range_stays_finite() {
    // `w_max - w_min` overflows f32 here (6e38 > f32::MAX), which used to
    // yield `inf` weights — and `inf * 0.0` = `NaN` — from a range that
    // passes the finiteness check. Interpolating in f64 keeps every weight
    // inside the requested bounds.
    let layer = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
        num_inputs: 8,
        num_neurons: 4,
        fan_in: 3,
        seed: 11,
        weight_range: (-3.0e38, 3.0e38),
        ..Default::default()
    })
    .unwrap();

    assert!(
        layer.weights().iter().all(|w| w.is_finite()),
        "non-finite weight from a finite range: {:?}",
        layer.weights()
    );
    assert!(
        layer
            .weights()
            .iter()
            .all(|&w| (-3.0e38..=3.0e38).contains(&w)),
        "weight escaped the requested range"
    );
}

#[test]
fn wrong_sized_spike_buffer_reports_neurons_not_channels() {
    // num_inputs is 8 and num_neurons is 3, so a bad spike buffer must not
    // be described as an input-width problem -- the two widths differ and
    // the old message sent readers to the wrong argument.
    let mut layer = SparseGifHiddenLayer::new(&config(8, 3, 3, 5)).unwrap();
    let mut spikes = [false; 2];
    let err = layer.step_into(&[0.5; 8], &mut spikes).unwrap_err();
    assert_eq!(
        err,
        GifLayerError::OutputLenMismatch {
            expected: 3,
            got: 2
        }
    );
    assert!(err.to_string().contains("spike buffer"), "{err}");
}

#[test]
fn rejects_more_input_channels_than_a_u32_source_can_address() {
    // Empty rows, so this allocates nothing: the guard must fire on the
    // declared width alone. Without it, `source as u32` would wrap and a
    // channel above u32::MAX would alias onto a low one.
    let err =
        SparseGifHiddenLayer::from_topology(MAX_INPUTS + 1, GifParams::default(), &[]).unwrap_err();
    assert_eq!(
        err,
        GifLayerError::TooManyInputs {
            num_inputs: MAX_INPUTS + 1,
            max: MAX_INPUTS,
        }
    );
    assert!(err.to_string().contains("addressable maximum"));
}

#[test]
fn exhausted_step_counter_is_reported_without_mutating_state() {
    // Reachable only from a restored checkpoint. Incrementing past i64::MAX
    // would panic in debug and wrap to i64::MIN in release, which would make
    // every later `last_spike_time` comparison meaningless.
    let mut layer = SparseGifHiddenLayer::new(&config(8, 3, 3, 5)).unwrap();
    let mut json: serde_json::Value = serde_json::to_value(&layer).unwrap();
    json["step_count"] = serde_json::json!(i64::MAX);
    layer = serde_json::from_value(json).unwrap();

    // Snapshot every mutable bank, not just the membrane: a partial mutation
    // is exactly the failure this test exists to catch, so checking one array
    // would let the other two regress unnoticed.
    let membrane_before = layer.membrane().to_vec();
    let adaptation_before = layer.adaptation().to_vec();
    let spike_times_before = layer.last_spike_time().to_vec();

    let err = layer.step(&[1.0; 8]).unwrap_err();

    assert_eq!(
        err,
        GifLayerError::StepCounterExhausted {
            step_count: i64::MAX
        }
    );
    assert!(err.to_string().contains("exhausted"));
    // The step must fail before touching neuron state, not half-way through.
    assert_eq!(layer.membrane(), &membrane_before[..]);
    assert_eq!(layer.adaptation(), &adaptation_before[..]);
    assert_eq!(layer.last_spike_time(), &spike_times_before[..]);
    assert_eq!(layer.step_count(), i64::MAX);
}

#[test]
fn zero_fan_in_does_not_allocate_a_candidate_pool() {
    // A topology-free layer never samples the pool, so a wide input width must
    // not cost one u32 per channel. Constructing succeeds and stays empty.
    let layer = SparseGifHiddenLayer::new(&config(1_000_000, 4, 0, 7)).unwrap();
    assert_eq!(layer.num_synapses(), 0);
    assert!(layer.weights().is_empty());
    for n in 0..layer.num_neurons() {
        assert!(layer.fan_in_of(n).unwrap().0.is_empty());
    }
}

#[test]
fn generated_weights_stay_below_the_exclusive_upper_bound() {
    // `weight_range` documents a half-open interval, but `unit < 1` is not
    // enough on its own: with bounds one ULP apart, any unit above 0.5 rounds
    // the product up to exactly `w_max`. Seed 2 draws 0.5956, which did.
    let w_min = 1.0f32;
    let w_max = f32::from_bits(w_min.to_bits() + 1);
    let layer = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
        num_inputs: 1,
        num_neurons: 1,
        fan_in: 1,
        seed: 2,
        weight_range: (w_min, w_max),
        ..Default::default()
    })
    .unwrap();

    for &w in layer.weights() {
        assert!(
            w >= w_min && w < w_max,
            "weight {w:e} escaped the half-open range [{w_min:e}, {w_max:e})"
        );
    }

    // A degenerate range has an empty half-open interval, so the single
    // representable answer is allowed through rather than stepped below the
    // floor.
    let degenerate = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
        num_inputs: 4,
        num_neurons: 2,
        fan_in: 2,
        seed: 3,
        weight_range: (0.25, 0.25),
        ..Default::default()
    })
    .unwrap();
    assert!(degenerate.weights().iter().all(|&w| w == 0.25));
}

#[path = "layer_api_tests.rs"]
mod api_tests;

#[path = "layer_serde_tests.rs"]
mod serde_tests;

#[path = "layer_golden_tests.rs"]
mod golden_tests;
