use super::*;

#[test]
fn constructors_reject_non_finite_parameters_and_weights() {
    let params = GifParams {
        leak: f32::NAN,
        ..Default::default()
    };
    assert!(
        SparseGifHiddenLayer::new(&SparseGifLayerConfig {
            params,
            ..config(1, 1, 1, 176)
        })
        .is_err()
    );
    assert!(SparseGifHiddenLayer::from_topology(1, params, &[vec![(0, 1.0)]]).is_err());
    assert!(
        SparseGifHiddenLayer::from_topology(1, GifParams::default(), &[vec![(0, f32::INFINITY)]])
            .is_err()
    );
}

#[test]
fn nan_stimulus_is_rejected_without_poisoning_the_layer() {
    let mut layer =
        SparseGifHiddenLayer::from_topology(1, GifParams::default(), &[vec![(0, 1.0)]]).unwrap();
    let before = layer.clone();
    let mut spikes = [true];
    assert!(layer.step_into(&[f32::NAN], &mut spikes).is_err());
    assert_eq!(layer, before);
    assert_eq!(spikes, [true]);
}

#[test]
fn finite_drive_overflow_is_rejected_atomically() {
    let mut layer = SparseGifHiddenLayer::from_topology(
        1,
        GifParams::default(),
        &[vec![(0, 1.0)], vec![(0, f32::MAX)]],
    )
    .unwrap();
    let before = layer.clone();
    let mut spikes = [false, true];
    assert!(layer.step_into(&[2.0], &mut spikes).is_err());
    assert_eq!(layer, before);
    assert_eq!(spikes, [false, true]);
}

use crate::NonFiniteClass;

const NON_FINITE: [(f32, NonFiniteClass); 3] = [
    (f32::NAN, NonFiniteClass::Nan),
    (f32::INFINITY, NonFiniteClass::PosInfinity),
    (f32::NEG_INFINITY, NonFiniteClass::NegInfinity),
];

type ParamCase = (&'static str, fn(&mut GifParams, f32));

fn parameter_cases() -> [ParamCase; 8] {
    [
        ("leak", |p, v| p.leak = v),
        ("drive_scale", |p, v| p.drive_scale = v),
        ("base_threshold", |p, v| p.base_threshold = v),
        ("adaptation_scale", |p, v| p.adaptation_scale = v),
        ("adaptation_decay", |p, v| p.adaptation_decay = v),
        ("adaptation_coupling", |p, v| p.adaptation_coupling = v),
        ("adaptation_increment", |p, v| p.adaptation_increment = v),
        ("reset_ratio", |p, v| p.reset_ratio = v),
    ]
}

/// JSON covers all structural fields; bits preserve signed zero and NaN payloads.
fn fingerprint(layer: &SparseGifHiddenLayer) -> (serde_json::Value, Vec<u32>) {
    let p = layer.params();
    let params = [
        p.leak,
        p.drive_scale,
        p.base_threshold,
        p.adaptation_scale,
        p.adaptation_decay,
        p.adaptation_coupling,
        p.adaptation_increment,
        p.reset_ratio,
    ];
    let bits = layer
        .weights()
        .iter()
        .chain(layer.membrane())
        .chain(layer.adaptation())
        .chain(&params)
        .map(|v| v.to_bits())
        .collect();
    (serde_json::to_value(layer).unwrap(), bits)
}

/// Exercise both stepping surfaces; a later-neuron error must retain every bit
/// and the caller's pre-existing mixed output flags.
fn assert_rejected(layer: &mut SparseGifHiddenLayer, input: &[f32], expected: GifLayerError) {
    let before = fingerprint(layer);
    let mut output: Vec<bool> = (0..layer.num_neurons()).map(|i| i % 2 == 0).collect();
    let saved_output = output.clone();
    assert_eq!(
        (
            layer.step_into(input, &mut output),
            fingerprint(layer),
            output
        ),
        (Err(expected), before.clone(), saved_output),
    );
    assert_eq!(
        (layer.step(input), fingerprint(layer)),
        (Err(expected), before)
    );
}

/// A repaired or restored layer must both succeed and match its valid control.
fn assert_same_step(
    layer: &mut SparseGifHiddenLayer,
    control: &mut SparseGifHiddenLayer,
    input: &[f32],
) {
    let actual = layer.step(input).expect("valid retry/continuation");
    let expected = control.step(input).expect("valid control");
    assert_eq!(
        (actual, fingerprint(layer)),
        (expected, fingerprint(control))
    );
}

fn small_layer() -> SparseGifHiddenLayer {
    SparseGifHiddenLayer::from_topology(
        3,
        GifParams::default(),
        &[vec![(0, 0.5)], vec![(1, 0.25), (0, 0.75)]],
    )
    .unwrap()
}

#[test]
fn every_constructor_parameter_rejects_each_non_finite_class() {
    for (field, set) in parameter_cases() {
        for (value, class) in NON_FINITE {
            let mut params = GifParams::default();
            set(&mut params, value);
            let expected = GifLayerError::NonFiniteParam { field, class };
            assert_eq!(
                SparseGifHiddenLayer::new(&SparseGifLayerConfig {
                    params,
                    ..config(3, 2, 2, 176)
                })
                .unwrap_err(),
                expected
            );
            assert_eq!(
                SparseGifHiddenLayer::from_topology(3, params, &[]).unwrap_err(),
                expected
            );
        }
    }
}

#[test]
fn explicit_weights_report_the_flat_csr_index() {
    for (value, class) in NON_FINITE {
        let rows = [vec![], vec![(0, 1.0)], vec![(1, 0.5), (0, value)]];
        assert_eq!(
            SparseGifHiddenLayer::from_topology(3, GifParams::default(), &rows).unwrap_err(),
            GifLayerError::NonFiniteWeight { index: 2, class }
        );
    }
}

#[test]
fn every_mutable_parameter_is_rechecked_and_can_be_repaired() {
    for (field, set) in parameter_cases() {
        for (value, class) in NON_FINITE {
            let mut layer = small_layer();
            let mut control = layer.clone();
            set(layer.params_mut(), value);
            assert_rejected(
                &mut layer,
                &[1.0; 3],
                GifLayerError::NonFiniteParam { field, class },
            );
            *layer.params_mut() = GifParams::default();
            assert_same_step(&mut layer, &mut control, &[1.0; 3]);
        }
    }
}

#[test]
fn injected_weights_are_rejected_even_with_zero_drive_and_retry_matches_control() {
    for (value, class) in NON_FINITE
        .into_iter()
        .chain([(f32::from_bits(0x7fc0_0176), NonFiniteClass::Nan)])
    {
        let mut layer = small_layer();
        let mut control = layer.clone();
        layer.weights_mut()[2] = value;
        assert_rejected(
            &mut layer,
            &[0.0; 3],
            GifLayerError::NonFiniteWeight { index: 2, class },
        );
        layer.weights_mut()[2] = 0.75;
        assert_same_step(&mut layer, &mut control, &[1.0; 3]);
    }
}

#[test]
fn all_stimuli_are_checked_including_unused_channels_and_empty_layers() {
    for (value, class) in NON_FINITE {
        for neurons in [0, 2] {
            let mut layer = SparseGifHiddenLayer::new(&config(3, neurons, 0, 176)).unwrap();
            assert_rejected(
                &mut layer,
                &[0.0, 0.0, value],
                GifLayerError::NonFiniteInput { index: 2, class },
            );
            let mut control = layer.clone();
            assert_same_step(&mut layer, &mut control, &[0.0; 3]);
        }
    }
}

/// Plant finite, serde-valid state so each arithmetic stage can overflow
/// independently. Private-state edits here model an accepted checkpoint.
fn overflow_case(stage: &str) -> (SparseGifHiddenLayer, Vec<f32>) {
    let mut layer = small_layer();
    let mut input = vec![0.0; 3];
    match stage {
        "drive" => {
            layer.weights_mut()[2] = f32::MAX;
            input[0] = 2.0;
        }
        "decayed_adaptation" => {
            layer.adaptation[1] = f32::MAX;
            layer.params_mut().adaptation_decay = 2.0;
        }
        "integrated_membrane" => {
            layer.membrane[1] = f32::MAX;
            layer.params_mut().leak = 2.0;
        }
        "threshold" => {
            layer.adaptation[1] = f32::MAX;
            layer.params_mut().adaptation_scale = 2.0;
        }
        "membrane" => {
            layer.params_mut().base_threshold = -f32::MAX;
            layer.params_mut().reset_ratio = 2.0;
        }
        "adaptation" => {
            layer.adaptation[1] = f32::MAX;
            layer.params_mut().adaptation_scale = 0.0;
            layer.params_mut().adaptation_coupling = 0.0;
            layer.params_mut().adaptation_increment = f32::MAX;
            input[0] = 2.0;
        }
        _ => unreachable!(),
    }
    (layer, input)
}

#[test]
fn overflow_at_each_transition_stage_preserves_state_output_and_allows_retry() {
    for stage in [
        "drive",
        "decayed_adaptation",
        "integrated_membrane",
        "threshold",
        "membrane",
        "adaptation",
    ] {
        let (mut layer, input) = overflow_case(stage);
        let json = serde_json::to_string(&layer).unwrap();
        layer = serde_json::from_str(&json).expect("all starting values are finite");
        let neuron = if stage == "membrane" { 0 } else { 1 };
        assert_rejected(
            &mut layer,
            &input,
            GifLayerError::NumericOverflow {
                neuron,
                stage,
                class: NonFiniteClass::PosInfinity,
            },
        );
        let restored: SparseGifHiddenLayer =
            serde_json::from_str(&serde_json::to_string(&layer).unwrap()).unwrap();
        assert_eq!(fingerprint(&layer), fingerprint(&restored));
        layer.reset();
        *layer.params_mut() = GifParams::default();
        layer.weights_mut().fill(0.25);
        let mut control = small_layer();
        control.weights_mut().fill(0.25);
        assert_same_step(&mut layer, &mut control, &[1.0; 3]);
    }
}

#[test]
fn finite_product_sum_and_negative_overflow_are_rejected() {
    for weight in [f32::MAX, -f32::MAX] {
        let mut layer = SparseGifHiddenLayer::from_topology(
            1,
            GifParams::default(),
            &[vec![(0, weight), (0, weight)]],
        )
        .unwrap();
        let class = if weight > 0.0 {
            NonFiniteClass::PosInfinity
        } else {
            NonFiniteClass::NegInfinity
        };
        assert_rejected(
            &mut layer,
            &[1.0],
            GifLayerError::NumericOverflow {
                neuron: 0,
                stage: "drive",
                class,
            },
        );
    }
}

#[test]
fn run_keeps_successful_prefix_but_rejects_the_failing_frame() {
    let mut layer = small_layer();
    let mut expected = layer.clone();
    expected.step(&[0.5; 3]).unwrap();
    let result = layer.run(&[[0.5; 3], [f32::NAN; 3], [1.0; 3]]);
    assert_eq!(
        result.unwrap_err(),
        GifLayerError::NonFiniteInput {
            index: 0,
            class: NonFiniteClass::Nan
        }
    );
    assert_eq!(fingerprint(&layer), fingerprint(&expected));
    assert_same_step(&mut layer, &mut expected, &[1.0; 3]);
}

#[test]
fn finite_signed_parameters_and_constructor_states_round_trip_and_continue() {
    for (_, set) in parameter_cases() {
        let mut params = GifParams::default();
        set(&mut params, -0.25);
        let generated = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
            params,
            weight_range: (-0.5, 0.5),
            ..config(3, 2, 2, 176)
        })
        .unwrap();
        let explicit =
            SparseGifHiddenLayer::from_topology(3, params, &[vec![(0, -0.5)], vec![]]).unwrap();
        for mut layer in [generated, explicit] {
            let mut restored: SparseGifHiddenLayer =
                serde_json::from_str(&serde_json::to_string(&layer).unwrap()).unwrap();
            for _ in 0..3 {
                assert_same_step(&mut layer, &mut restored, &[0.5; 3]);
            }
        }
    }
}

#[test]
fn existing_length_errors_precede_numeric_validation() {
    let mut layer = small_layer();
    layer.params_mut().leak = f32::NAN;
    let before = fingerprint(&layer);
    let mut output = [true];
    assert_eq!(
        layer.step_into(&[], &mut output),
        Err(GifLayerError::InputLenMismatch {
            expected: 3,
            got: 0
        })
    );
    assert_eq!(
        layer.step_into(&[0.0; 3], &mut output),
        Err(GifLayerError::OutputLenMismatch {
            expected: 2,
            got: 1
        })
    );
    assert_eq!((fingerprint(&layer), output), (before, [true]));
}

#[test]
fn counter_exhaustion_precedes_numeric_validation() {
    let mut layer = small_layer();
    layer.params_mut().leak = f32::NAN;
    layer.step_count = i64::MAX;
    assert_rejected(
        &mut layer,
        &[0.0; 3],
        GifLayerError::StepCounterExhausted {
            step_count: i64::MAX,
        },
    );
}

#[test]
fn finite_products_can_cancel_to_nan_but_must_not_commit() {
    let mut layer = SparseGifHiddenLayer::from_topology(
        1,
        GifParams::default(),
        &[vec![(0, f32::MAX), (0, -f32::MAX)]],
    )
    .unwrap();
    assert_rejected(
        &mut layer,
        &[2.0],
        GifLayerError::NumericOverflow {
            neuron: 0,
            stage: "drive",
            class: NonFiniteClass::Nan,
        },
    );
}

#[test]
fn finite_drive_scaling_overflow_is_rejected_before_a_reset_can_hide_it() {
    let params = GifParams {
        drive_scale: f32::MAX,
        reset_ratio: 0.0,
        ..Default::default()
    };
    let mut layer = SparseGifHiddenLayer::from_topology(1, params, &[vec![(0, 1.0)]]).unwrap();
    assert_rejected(
        &mut layer,
        &[2.0],
        GifLayerError::NumericOverflow {
            neuron: 0,
            stage: "integrated_membrane",
            class: NonFiniteClass::PosInfinity,
        },
    );
}

#[test]
fn poisoned_mutable_weights_remain_inspectable_but_cannot_decode_as_a_valid_layer() {
    let mut layer = small_layer();
    layer.weights_mut()[2] = f32::INFINITY;
    let json = serde_json::to_string(&layer).unwrap();
    assert!(serde_json::from_str::<SparseGifHiddenLayer>(&json).is_err());
    assert_rejected(
        &mut layer,
        &[0.0; 3],
        GifLayerError::NonFiniteWeight {
            index: 2,
            class: NonFiniteClass::PosInfinity,
        },
    );
    assert!(layer.weights()[2].is_infinite());
}
