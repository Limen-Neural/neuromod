use super::*;

// --- public accessor surface -----------------------------------------

#[test]
fn accessors_report_the_configured_shape() {
    let layer = SparseGifHiddenLayer::new(&config(16, 5, 4, 0xBEEF)).unwrap();
    assert_eq!(layer.num_inputs(), 16);
    assert_eq!(layer.num_neurons(), 5);
    assert_eq!(layer.seed(), 0xBEEF);
    assert_eq!(layer.num_synapses(), 5 * 4);
    assert_eq!(layer.params(), &GifParams::default());
    assert_eq!(layer.step_count(), 0);

    // `from_topology` has no generator, so it reports a zero seed.
    let explicit =
        SparseGifHiddenLayer::from_topology(3, GifParams::default(), &[vec![(0, 0.5)]]).unwrap();
    assert_eq!(explicit.seed(), 0);
    assert_eq!(explicit.num_inputs(), 3);
}

#[test]
fn params_mut_changes_firing() {
    // The documented modulation hook. Assert it actually reaches the dynamics
    // rather than merely being reachable: a threshold far above any attainable
    // membrane value must silence a layer that otherwise fires.
    let mut loud = SparseGifHiddenLayer::new(&config(8, 4, 4, 21)).unwrap();
    let fired_before: usize = loud
        .run(&ramp_train(40, 8))
        .unwrap()
        .per_neuron_counts()
        .iter()
        .sum();
    assert!(fired_before > 0, "fixture must fire to be meaningful");

    let mut quiet = SparseGifHiddenLayer::new(&config(8, 4, 4, 21)).unwrap();
    quiet.params_mut().base_threshold = 1.0e6;
    let fired_after: usize = quiet
        .run(&ramp_train(40, 8))
        .unwrap()
        .per_neuron_counts()
        .iter()
        .sum();
    assert_eq!(fired_after, 0, "params_mut did not reach the dynamics");
}

#[test]
fn weights_mut_changes_drive() {
    // The other documented modulation hook: zeroing every synapse removes all
    // drive, so the membrane cannot move off its resting value.
    let mut layer = SparseGifHiddenLayer::new(&config(8, 4, 4, 21)).unwrap();
    for w in layer.weights_mut() {
        *w = 0.0;
    }
    layer.run(&ramp_train(20, 8)).unwrap();
    assert!(
        layer.membrane().iter().all(|&v| v == 0.0),
        "zeroed weights still produced drive: {:?}",
        layer.membrane()
    );
}

#[test]
fn as_flat_agrees_with_the_per_step_views() {
    // The flat buffer is row-major `step * num_neurons + neuron`; check that
    // contract against the accessors built on top of it rather than just
    // touching the getter.
    let mut layer = SparseGifHiddenLayer::new(&config(8, 4, 3, 77)).unwrap();
    let raster = layer.run(&ramp_train(12, 8)).unwrap();

    let flat = raster.as_flat();
    assert_eq!(flat.len(), raster.num_steps() * raster.num_neurons());
    assert_eq!(flat.iter().filter(|&&b| b).count(), raster.total_spikes());

    for t in 0..raster.num_steps() {
        let lo = t * raster.num_neurons();
        assert_eq!(
            &flat[lo..lo + raster.num_neurons()],
            raster.step(t).unwrap()
        );

        let expected: Vec<usize> = raster
            .step(t)
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(n, &f)| f.then_some(n))
            .collect();
        assert_eq!(raster.fired_at(t), expected);
    }

    // Out-of-range steps are None / empty, not a panic.
    assert!(raster.step(raster.num_steps()).is_none());
    assert!(raster.fired_at(raster.num_steps()).is_empty());
}

/// Names every [`GifLayerError`] variant in an exhaustive match.
///
/// This is the compile-time guard: adding a variant makes the match
/// non-exhaustive and breaks the build here, forcing it into
/// [`all_error_variants`] below. The array on its own could not do that — it
/// is hand-written, so a new variant would simply be missing from it.
#[expect(
    clippy::match_same_arms,
    reason = "one arm per variant is the point; collapsing them defeats the guard"
)]
fn assert_error_variants_exhaustive(e: &GifLayerError) {
    match e {
        GifLayerError::FanInExceedsInputs { .. } => {}
        GifLayerError::InvalidWeightRange { .. } => {}
        GifLayerError::InputLenMismatch { .. } => {}
        GifLayerError::OutputLenMismatch { .. } => {}
        GifLayerError::SourceOutOfRange { .. } => {}
        GifLayerError::TooManyInputs { .. } => {}
        GifLayerError::MalformedCheckpoint { .. } => {}
        GifLayerError::StepCounterExhausted { .. } => {}
        GifLayerError::RasterTooLarge { .. } => {}
    }
}

/// One instance of every [`GifLayerError`] variant.
///
/// Kept honest by [`assert_error_variants_exhaustive`] and by the array's own
/// length: a new variant fails to compile there, and widening this array
/// without adding an entry fails to compile here.
fn all_error_variants() -> [GifLayerError; 9] {
    [
        GifLayerError::FanInExceedsInputs {
            fan_in: 5,
            num_inputs: 4,
        },
        GifLayerError::InvalidWeightRange {
            min: 1.0,
            max: -1.0,
        },
        GifLayerError::InputLenMismatch {
            expected: 8,
            got: 3,
        },
        GifLayerError::OutputLenMismatch {
            expected: 4,
            got: 2,
        },
        GifLayerError::SourceOutOfRange {
            neuron: 1,
            source: 9,
            num_inputs: 4,
        },
        GifLayerError::TooManyInputs {
            num_inputs: MAX_INPUTS + 1,
            max: MAX_INPUTS,
        },
        GifLayerError::MalformedCheckpoint {
            detail: "some invariant",
        },
        GifLayerError::StepCounterExhausted {
            step_count: i64::MAX,
        },
        GifLayerError::RasterTooLarge {
            num_steps: 3,
            num_neurons: 4,
        },
    ]
}

#[test]
fn every_error_variant_renders_a_distinct_message() {
    let variants = all_error_variants();
    let messages: Vec<String> = variants.iter().map(ToString::to_string).collect();
    for (v, m) in variants.iter().zip(&messages) {
        assert_error_variants_exhaustive(v);
        assert!(!m.is_empty(), "{v:?} rendered an empty message");
    }

    // Distinct messages: a copy-paste arm that reported the wrong variant
    // would collide here.
    let mut unique = messages.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        messages.len(),
        "duplicate messages: {messages:?}"
    );
}
