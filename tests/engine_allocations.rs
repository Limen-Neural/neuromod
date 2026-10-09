use neuromod::{NeuroModulators, SeedableRng, SpikingNetwork, StdRng};

/// Returning spikes may allocate, but a firing-heavy step must not grow its
/// output geometrically or allocate a second mask of the same neuron bank.
#[test]
fn step_allocation_budget_is_independent_of_bank_size() {
    let mut within_budget = true;
    for size in [16, 64, 512] {
        for frozen in [false, true] {
            let mut network = SpikingNetwork::with_dimensions(size, 5, size);
            for neuron in &mut network.neurons {
                neuron.weights.fill(2.0 / size as f32);
            }
            let stimuli = vec![1.0; size];
            let modulators = NeuroModulators::default();
            let mut rng = StdRng::seed_from_u64(190);
            // Initialize all runtime paths before counting steady-state work.
            if frozen {
                network
                    .step_frozen_with_rng(&stimuli, &modulators, &mut rng)
                    .unwrap();
            } else {
                network
                    .step_with_rng(&stimuli, &modulators, &mut rng)
                    .unwrap();
            }

            let allocations = allocation_counter::measure(|| {
                for _ in 0..100 {
                    let spikes = if frozen {
                        network.step_frozen_with_rng(&stimuli, &modulators, &mut rng)
                    } else {
                        network.step_with_rng(&stimuli, &modulators, &mut rng)
                    }
                    .unwrap();
                    assert_eq!(spikes.len(), size);
                    assert!(spikes.iter().copied().eq(0..size));
                }
            });
            let budget = if frozen { 300 } else { 200 };
            eprintln!(
                "{size}x{size} frozen={frozen}: {} allocations/100 steps",
                allocations.count_total
            );
            within_budget &= allocations.count_total <= budget;
            assert_eq!(allocations.count_current, 0, "step buffers must be freed");
        }
    }
    assert!(
        within_budget,
        "step allocation budget exceeded; see counts above"
    );
}
