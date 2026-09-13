use super::*;

// --- golden regression fixtures --------------------------------------
//
// These are INTERNAL goldens produced by this implementation, not
// cross-repo parity vectors from `corinth-canal` (see the module docs).
// They pin the topology generator, the CSR traversal order, and the GIF
// arithmetic together: any of the three drifting will fail here.

#[test]
fn golden_topology_fixture() {
    let layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
    let rows: Vec<Vec<u32>> = (0..4)
        .map(|n| layer.fan_in_of(n).unwrap().0.to_vec())
        .collect();
    assert_eq!(
        rows,
        vec![
            vec![11, 13, 15],
            vec![3, 10, 15],
            vec![6, 10, 11],
            vec![1, 3, 14],
        ]
    );
}

#[test]
fn golden_weight_fixture() {
    let layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
    let expected: [f32; 12] = [
        0.249_479_71,
        0.710_527_5,
        0.974_420_7,
        0.850_541_5,
        0.567_846_83,
        0.039_654_434,
        0.028_471_59,
        0.159_303_13,
        0.461_927_65,
        0.108_223_14,
        0.395_182_2,
        0.187_865_02,
    ];
    for (i, (&got, &want)) in layer.weights().iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "weight {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn golden_raster_fixture() {
    let mut layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
    let raster = layer.run(&ramp_train(20, 16)).unwrap();
    assert_eq!(raster.per_neuron_counts(), vec![12, 10, 4, 5]);
    assert_eq!(raster.total_spikes(), 31);
    assert_eq!(raster.fired_at(5), vec![0, 3]);
    assert_eq!(raster.fired_at(19), vec![1, 3]);
}

#[test]
fn golden_state_fixture() {
    let mut layer = SparseGifHiddenLayer::new(&config(16, 4, 3, 0xA5A5_A5A5)).unwrap();
    layer.run(&ramp_train(20, 16)).unwrap();

    let membrane: [f32; 4] = [1.865_292_9, 1.185_997_7, 1.030_233_5, 0.742_578];
    let adaptation: [f32; 4] = [6.510_236_3, 5.604_505_5, 2.015_107_6, 2.864_713];
    for (i, (&got, &want)) in layer.membrane().iter().zip(membrane.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "membrane {i}: got {got}, want {want}"
        );
    }
    for (i, (&got, &want)) in layer.adaptation().iter().zip(adaptation.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "adaptation {i}: got {got}, want {want}"
        );
    }
    assert_eq!(layer.step_count(), 20);
}
