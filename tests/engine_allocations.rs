use neuromod::{NeuroModulators, SeedableRng, SpikingNetwork, StdRng};

/// Returning spikes may allocate, but a firing-heavy step must not grow its
/// output geometrically or allocate a second mask of the same neuron bank.
#[test]
fn step_allocation_budget_is_independent_of_bank_size() {
    let mut within_budget = true;
    for size in [16, 64, 512] {
        for frozen in [false, true] {
            within_budget &= steady_state_within_budget(size, frozen);
        }
    }
    assert!(
        within_budget,
        "step allocation budget exceeded; see counts above"
    );
}

/// Counts allocations over 100 dense-firing steps of one bank size and mode,
/// prints the count, and reports whether it fits that mode's budget.
fn steady_state_within_budget(size: usize, frozen: bool) -> bool {
    let mut network = SpikingNetwork::with_dimensions(size, 5, size);
    for neuron in &mut network.neurons {
        neuron.weights.fill(2.0 / size as f32);
    }
    let stimuli = vec![1.0; size];
    let modulators = NeuroModulators::default();
    let mut rng = StdRng::seed_from_u64(190);
    let step = |network: &mut SpikingNetwork, rng: &mut StdRng| {
        if frozen {
            network.step_frozen_with_rng(&stimuli, &modulators, rng)
        } else {
            network.step_with_rng(&stimuli, &modulators, rng)
        }
        .unwrap()
    };
    // Initialize all runtime paths before counting steady-state work.
    step(&mut network, &mut rng);

    let allocations = allocation_counter::measure(|| {
        for _ in 0..100 {
            let spikes = step(&mut network, &mut rng);
            assert_eq!(spikes.len(), size);
            assert!(spikes.iter().copied().eq(0..size));
        }
    });
    let budget = if frozen { 300 } else { 200 };
    eprintln!(
        "{size}x{size} frozen={frozen}: {} allocations/100 steps",
        allocations.count_total
    );
    assert_eq!(allocations.count_current, 0, "step buffers must be freed");
    allocations.count_total <= budget
}
