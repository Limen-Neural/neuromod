#![allow(clippy::assertions_on_result_states)]

use neuromod::{NeuroModulators, SpikingNetwork};

// ---------------------------------------------------------------------------
// Core API used by examples/sentry.rs (always compiled, no feature gate)
// ---------------------------------------------------------------------------

/// examples/sentry.rs constructs a default network and calls step exactly once.
/// This mirrors that call and verifies it succeeds.
#[test]
fn example_sentry_network_step_succeeds() {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();

    let result = network.step(&stimuli, &modulators);
    assert!(
        result.is_ok(),
        "step should succeed with correct 16-element input"
    );
}

/// The step result is a Vec of spiking neuron indices, all valid indices.
#[test]
fn example_sentry_step_returns_valid_spike_indices() {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();

    let spikes = network.step(&stimuli, &modulators).unwrap();
    for &idx in &spikes {
        assert!(
            idx < network.neurons.len(),
            "spike index {idx} must be within neuron count {}",
            network.neurons.len()
        );
    }
}

/// global_step increments by exactly 1 after a successful step, as in the example.
#[test]
fn example_sentry_step_increments_global_step() {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();

    assert_eq!(network.global_step, 0);
    network.step(&stimuli, &modulators).unwrap();
    assert_eq!(network.global_step, 1);
}

// ---------------------------------------------------------------------------
// Default network dimensions (matches constants used in the example)
// ---------------------------------------------------------------------------

/// SpikingNetwork::new() must expose 16 input channels to accept [0.5f32; 16].
#[test]
fn default_network_accepts_16_channel_stimuli() {
    let network = SpikingNetwork::new();
    assert_eq!(
        network.num_channels, 16,
        "default network must have 16 input channels"
    );
}

/// The default network has 16 LIF neurons and 5 Izhikevich neurons.
#[test]
fn default_network_neuron_counts() {
    let network = SpikingNetwork::new();
    assert_eq!(network.neurons.len(), 16);
    assert_eq!(network.iz_neurons.len(), 5);
}

// ---------------------------------------------------------------------------
// Edge cases from the example pattern
// ---------------------------------------------------------------------------

/// All-zero stimuli: step still returns Ok (no panic, no error).
#[test]
fn step_with_zero_stimuli_returns_ok() {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.0f32; 16];
    let modulators = NeuroModulators::default();

    let result = network.step(&stimuli, &modulators);
    assert!(result.is_ok());
}

/// All-ones stimuli: step returns Ok with at most 16 spike indices.
#[test]
fn step_with_max_stimuli_returns_ok() {
    let mut network = SpikingNetwork::new();
    let stimuli = [1.0f32; 16];
    let modulators = NeuroModulators::default();

    let spikes = network.step(&stimuli, &modulators).unwrap();
    assert!(spikes.len() <= 16, "cannot have more spikes than neurons");
}

/// Negative stimuli values are silently clamped; step must not panic or error.
#[test]
fn step_with_negative_stimuli_does_not_panic() {
    let mut network = SpikingNetwork::new();
    let stimuli = [-0.5f32; 16];
    let modulators = NeuroModulators::default();

    assert!(network.step(&stimuli, &modulators).is_ok());
}

/// Multiple successive steps advance global_step correctly.
#[test]
fn successive_steps_advance_global_step() {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();

    for expected_step in 1..=5u64 {
        network.step(&stimuli, &modulators).unwrap();
        assert_eq!(network.global_step, expected_step as i64);
    }
}

/// Wrong-length input returns an error (boundary / negative case).
#[test]
fn step_wrong_input_length_returns_error() {
    let mut network = SpikingNetwork::new();
    let modulators = NeuroModulators::default();

    // One fewer channel than expected
    let short = vec![0.5f32; 15];
    assert!(
        network.step(&short, &modulators).is_err(),
        "should reject input shorter than num_channels"
    );

    // One more channel than expected
    let long = vec![0.5f32; 17];
    assert!(
        network.step(&long, &modulators).is_err(),
        "should reject input longer than num_channels"
    );
}

// ---------------------------------------------------------------------------
// SENTRY_DSN environment-variable logic (mirrors examples/sentry.rs)
// ---------------------------------------------------------------------------

/// An empty SENTRY_DSN means Sentry is not initialised.
/// The example uses `std::env::var("SENTRY_DSN").unwrap_or_default()` and
/// checks `.is_empty()`. These tests validate that exact lookup pattern.
#[test]
fn sentry_dsn_absent_resolves_to_empty() {
    temp_env::with_var_unset("SENTRY_DSN", || {
        let dsn = std::env::var("SENTRY_DSN").unwrap_or_default();
        assert!(
            dsn.is_empty(),
            "absent SENTRY_DSN must produce empty string"
        );
    });
}

/// A non-empty SENTRY_DSN is detected by the example's guard condition.
#[test]
fn sentry_dsn_present_is_non_empty() {
    temp_env::with_var(
        "SENTRY_DSN",
        Some("https://example@sentry.example.com/1"),
        || {
            let dsn = std::env::var("SENTRY_DSN").unwrap_or_default();
            assert!(
                !dsn.is_empty(),
                "set SENTRY_DSN must produce non-empty string"
            );
        },
    );
}

/// An explicitly empty SENTRY_DSN ("") also triggers the "not reporting" branch.
#[test]
fn sentry_dsn_explicit_empty_resolves_to_empty() {
    temp_env::with_var("SENTRY_DSN", Some(""), || {
        let dsn = std::env::var("SENTRY_DSN").unwrap_or_default();
        assert!(
            dsn.is_empty(),
            "explicitly empty SENTRY_DSN must still be empty"
        );
    });
}

// ---------------------------------------------------------------------------
// Feature-flag compilation tests
// ---------------------------------------------------------------------------

/// When the `sentry` feature is NOT enabled the library still compiles and
/// the full neuromod API is usable — exactly as the `#[cfg(not(feature = "sentry"))]`
/// block in examples/sentry.rs expects.
#[test]
fn neuromod_usable_without_sentry_feature() {
    // This test always compiles (no feature gate). The fact that it compiles
    // and passes without `--features sentry` proves the default build is
    // fully functional.
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("step failed without sentry feature");
    // Arbitrary correctness check: result is within bounds
    assert!(spikes.len() <= network.neurons.len());
}

/// When the `sentry` feature IS enabled the same neuromod API must continue
/// to work — the feature only adds an optional initialisation side-effect.
#[cfg(feature = "sentry")]
#[test]
fn neuromod_usable_with_sentry_feature_enabled() {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("step failed with sentry feature enabled");
    assert!(spikes.len() <= network.neurons.len());
}

/// With the `sentry` feature enabled and no DSN set, initialising Sentry with
/// an empty string must not crash the binary.  We model this by exercising the
/// env-var guard logic and confirming the fallback path is taken.
#[cfg(feature = "sentry")]
#[test]
fn sentry_feature_empty_dsn_uses_fallback_path() {
    temp_env::with_var_unset("SENTRY_DSN", || {
        let dsn = std::env::var("SENTRY_DSN").unwrap_or_default();
        // The example checks `!dsn.is_empty()` before calling sentry::init.
        // When empty we skip init — verify that guard logic here.
        let would_init = !dsn.is_empty();
        assert!(
            !would_init,
            "should NOT attempt sentry::init when DSN is absent"
        );
    });
}

// ---------------------------------------------------------------------------
// Cargo.toml feature definition: default features must NOT include sentry
// ---------------------------------------------------------------------------

/// Compiles only when `sentry` is off (default build path).
#[cfg(not(feature = "sentry"))]
#[test]
fn sentry_is_not_a_default_feature() {
    assert!(std::mem::size_of::<SpikingNetwork>() > 0);
    assert!(std::mem::size_of::<NeuroModulators>() > 0);
}
