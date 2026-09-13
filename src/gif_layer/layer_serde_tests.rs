use super::*;

// --- serde -----------------------------------------------------------

#[test]
fn layer_round_trips_through_json() {
    let mut layer = SparseGifHiddenLayer::new(&config(16, 6, 4, 21)).unwrap();
    layer.run(&ramp_train(15, 16)).unwrap();
    let json = serde_json::to_string(&layer).unwrap();
    let restored: SparseGifHiddenLayer = serde_json::from_str(&json).unwrap();
    assert_eq!(layer, restored);

    // And the restored layer continues the trajectory identically.
    let more = ramp_train(10, 16);
    let mut restored = restored;
    assert_eq!(layer.run(&more).unwrap(), restored.run(&more).unwrap());
}

// --- malformed-checkpoint rejection ----------------------------------
//
// `Deserialize` is derived over private CSR/SoA vectors whose lengths have
// to agree, and nothing in the wire format enforces that. Before the
// `try_from` shim each of these decoded into a layer that panicked on the
// next `step` while indexing. They must now fail at decode instead.

/// Serialize a good layer, corrupt one field in the JSON, and decode.
fn decode_corrupted(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> Result<SparseGifHiddenLayer, serde_json::Error> {
    let layer = SparseGifHiddenLayer::new(&config(8, 3, 3, 5)).unwrap();
    let mut v: serde_json::Value = serde_json::to_value(&layer).unwrap();
    mutate(&mut v);
    serde_json::from_value(v)
}

#[test]
fn rejects_checkpoint_with_short_soa_bank() {
    for field in ["membrane", "adaptation", "last_spike_time"] {
        let err = decode_corrupted(|v| {
            v[field].as_array_mut().unwrap().pop();
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("malformed serialized layer"),
            "{field} truncation should be rejected, got: {err}"
        );
    }
}

#[test]
fn rejects_checkpoint_with_bad_csr_offsets() {
    // Wrong length.
    assert!(
        decode_corrupted(|v| {
            v["fan_in_offsets"].as_array_mut().unwrap().pop();
        })
        .is_err()
    );

    // Does not start at zero.
    assert!(decode_corrupted(|v| { v["fan_in_offsets"][0] = 1.into() }).is_err());

    // Not non-decreasing — this is the one that would index backwards.
    // Offsets are [0, 3, 6, 9]; 7 > 6 makes row 1 end before it starts.
    // (Lowering an offset instead would still be valid CSR: [0, 0, 6, 9]
    // just describes an empty first row.)
    assert!(decode_corrupted(|v| { v["fan_in_offsets"][1] = 7.into() }).is_err());
}

#[test]
fn rejects_checkpoint_whose_payload_disagrees_with_offsets() {
    for field in ["fan_in_sources", "weights"] {
        let err = decode_corrupted(|v| {
            v[field].as_array_mut().unwrap().pop();
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("malformed serialized layer"),
            "{field} truncation should be rejected, got: {err}"
        );
    }
}

#[test]
fn rejects_checkpoint_sourcing_a_nonexistent_channel() {
    // num_inputs is 8, so channel 99 does not exist.
    let err = decode_corrupted(|v| v["fan_in_sources"][0] = 99.into()).unwrap_err();
    assert!(err.to_string().contains("nonexistent channel"), "{err}");
}

#[test]
fn a_valid_checkpoint_still_decodes() {
    // Guard against the validator being so strict it rejects good input.
    assert!(decode_corrupted(|_| {}).is_ok());
}

/// Serialize a real raster, corrupt one field, and decode.
fn decode_corrupted_raster(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> Result<SpikeRaster, serde_json::Error> {
    let mut layer = SparseGifHiddenLayer::new(&config(8, 3, 3, 5)).unwrap();
    let raster = layer.run(&ramp_train(6, 8)).unwrap();
    let mut v: serde_json::Value = serde_json::to_value(&raster).unwrap();
    mutate(&mut v);
    serde_json::from_value(v)
}

#[test]
fn rejects_raster_whose_buffer_disagrees_with_its_shape() {
    // `step()` bounds-checks the step index but then slices
    // `spikes[lo..lo + num_neurons]`, so a short buffer panics there
    // instead of failing at decode.
    let err = decode_corrupted_raster(|v| {
        v["spikes"].as_array_mut().unwrap().pop();
    })
    .unwrap_err();
    assert!(
        err.to_string().contains("spikes length"),
        "short buffer should be rejected, got: {err}"
    );

    // Overflowing the product must not wrap into a length that matches.
    let err = decode_corrupted_raster(|v| {
        v["num_steps"] = (usize::MAX / 2 + 1).into();
        v["num_neurons"] = 4.into();
    })
    .unwrap_err();
    assert!(err.to_string().contains("overflow"), "{err}");
}

#[test]
fn a_valid_raster_still_decodes() {
    assert!(decode_corrupted_raster(|_| {}).is_ok());
}
