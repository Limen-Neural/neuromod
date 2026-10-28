use criterion::measurement::WallTime;
use criterion::{
    BatchSize, Bencher, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group,
    criterion_main,
};
use neuromod::rm_stdp::{
    EligibilityTrace, RM_STDP_A_MINUS, RM_STDP_A_PLUS, RM_STDP_TAU_MINUS, RM_STDP_TAU_PLUS,
    RmStdpConfig,
};
use neuromod::{
    HebbianIzhikevichNetwork, NeuroModulators, SeedableRng, SpikingNetwork, StdRng, StdpParams,
    apply_classical_stdp,
};
use std::hint::black_box;

fn bench_classical_stdp(c: &mut Criterion) {
    let params = StdpParams::default();

    c.bench_function("classical_stdp_ltp", |b| {
        b.iter(|| {
            apply_classical_stdp(
                black_box(0),
                black_box(5),
                black_box(0.5),
                black_box(&params),
            );
        });
    });

    c.bench_function("classical_stdp_ltd", |b| {
        b.iter(|| {
            apply_classical_stdp(
                black_box(5),
                black_box(0),
                black_box(0.5),
                black_box(&params),
            );
        });
    });
}

fn bench_eligibility_trace_decay(c: &mut Criterion) {
    c.bench_function("eligibility_trace_decay", |b| {
        let mut trace = EligibilityTrace {
            value: 0.5,
            tau: 50.0,
        };
        b.iter(|| {
            trace.decay();
        });
    });
}

fn bench_eligibility_trace_accumulate(c: &mut Criterion) {
    c.bench_function("eligibility_trace_accumulate_ltp", |b| {
        let mut trace = EligibilityTrace::new(50.0);
        b.iter(|| {
            trace.accumulate(black_box(5.0));
            // Observe the result: `accumulate` only writes `self.value`, so
            // without this the kernel and the add are dead stores and the
            // bench times an empty loop.
            black_box(trace.value);
            trace.value = 0.0;
        });
    });

    c.bench_function("eligibility_trace_accumulate_ltd", |b| {
        let mut trace = EligibilityTrace::new(50.0);
        b.iter(|| {
            trace.accumulate(black_box(-5.0));
            black_box(trace.value);
            trace.value = 0.0;
        });
    });
}

/// The wired path: decay, accumulate, then convert the trace into a weight
/// change under the dopamine gate — one synapse's worth of `apply_stdp`.
///
/// Batched rather than carried across iterations. Every conversion here is a
/// potentiation, so a persistent `weight` climbs to `w_max` within a couple of
/// hundred iterations and stays pinned: from then on `clamp` returns the bound
/// and the bench times a saturated synapse instead of a converting one. Each
/// sample starts from a mid-range weight and a representative banked trace, so
/// the measured work is the same on the first sample and the last.
fn bench_reward_conversion(c: &mut Criterion) {
    let config = RmStdpConfig::default();

    c.bench_function("rm_stdp_trace_to_weight", |b| {
        b.iter_batched(
            || {
                (
                    EligibilityTrace {
                        value: 0.5,
                        tau: config.tau_eligibility,
                    },
                    0.5_f32,
                )
            },
            |(mut trace, weight)| {
                trace.decay();
                trace.accumulate(black_box(1.0));
                let dw = config.reward_lr * black_box(0.45_f32) * trace.value;
                (weight + dw).clamp(config.w_min, config.w_max)
            },
            BatchSize::SmallInput,
        );
    });
}

/// End-to-end engine step, which now carries trace bookkeeping for every
/// synapse. Rewarded and unrewarded are separate: only the rewarded path pays
/// for the trace -> weight conversion. `preserve` is the
/// `RmStdpConfig::renormalize_weights = false` opt-out at dopamine 0.0: the
/// same per-step work as `unrewarded` minus the L1 renormalization scan, so
/// the gap between the two isolates the pass's cost.
///
/// The network is warmed to steady state before timing. `apply_stdp` skips both
/// the decay `exp` and the weight conversion while a trace is still exactly
/// zero, so a freshly built network makes the first samples cheaper than every
/// later one — timing from a blank state would measure that transient rather
/// than the per-step cost of a running network. Warming up is preferable to
/// rebuilding per iteration here: `SpikingNetwork` is not `Clone`, and steady
/// state is the condition worth reporting.
fn bench_engine_step_with_traces(c: &mut Criterion) {
    const WARMUP_STEPS: usize = 200; // >> tau_eligibility, so traces settle

    let mut group = c.benchmark_group("engine_step");

    for (label, dopamine, renormalize) in [
        ("unrewarded", 0.0_f32, true),
        ("rewarded", 0.9, true),
        ("preserve", 0.0, false),
    ] {
        group.bench_function(label, |b| {
            let mut network = SpikingNetwork::with_dimensions(64, 5, 64);
            network.stdp_config.renormalize_weights = renormalize;
            for neuron in &mut network.neurons {
                neuron.weights = vec![2.0 / 64.0; 64];
            }
            let modulators = NeuroModulators {
                dopamine,
                ..Default::default()
            };
            let stimuli = vec![0.5_f32; 64];

            for _ in 0..WARMUP_STEPS {
                let _ = network.step(&stimuli, &modulators);
            }

            b.iter(|| {
                // Observe the returned spike list: `step` mutates `network`, but
                // dropping its result lets the optimizer elide the work that
                // builds that list.
                black_box(network.step(black_box(&stimuli), black_box(&modulators)))
            });
        });
    }

    group.finish();
}

/// Deterministic banked-credit workload: every trace stays nonzero, without
/// spike-timing kernels obscuring the cost of decay. Restore values outside
/// the timed step so long measurements cannot underflow into the zero fast path.
fn bench_engine_banked_trace_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_banked_trace_decay");
    register_banked_trace_decay(&mut group, "unrewarded", 0.0, 64);
    register_banked_trace_decay(&mut group, "rewarded", 0.9, 64);
    register_banked_trace_decay(&mut group, "unrewarded", 0.0, 512);
    register_banked_trace_decay(&mut group, "rewarded", 0.9, 512);
    group.finish();
}

fn register_banked_trace_decay(
    group: &mut BenchmarkGroup<WallTime>,
    label: &'static str,
    dopamine: f32,
    size: usize,
) {
    group.bench_with_input(BenchmarkId::new(label, size), &size, |b, &size| {
        measure_banked_trace_decay(b, size, dopamine);
    });
}

fn measure_banked_trace_decay(b: &mut Bencher<'_>, size: usize, dopamine: f32) {
    let mut network = prepared_banked_network(size);
    let stimuli = vec![0.0; size];
    let modulators = NeuroModulators {
        dopamine,
        ..Default::default()
    };
    b.iter_custom(|iterations| time_banked_steps(&mut network, &stimuli, &modulators, iterations));
}

fn prepared_banked_network(size: usize) -> SpikingNetwork {
    let mut network = SpikingNetwork::with_dimensions(size, 5, size);
    let weight = 2.0 / size as f32;
    for neuron in &mut network.neurons {
        neuron.weights.fill(weight);
    }
    network
}

fn time_banked_steps(
    network: &mut SpikingNetwork,
    stimuli: &[f32],
    modulators: &NeuroModulators,
    iterations: u64,
) -> std::time::Duration {
    let mut elapsed = std::time::Duration::ZERO;
    for _ in 0..iterations {
        elapsed += one_banked_step(network, stimuli, modulators);
    }
    elapsed
}

fn one_banked_step(
    network: &mut SpikingNetwork,
    stimuli: &[f32],
    modulators: &NeuroModulators,
) -> std::time::Duration {
    restore_alternating_traces(network);
    let start = std::time::Instant::now();
    black_box(network.step(black_box(stimuli), black_box(modulators)))
        .expect("finite inputs with matching dimensions");
    // Observe decay writes before the next iteration resets them.
    black_box(&network.neurons);
    start.elapsed()
}

fn restore_alternating_traces(network: &mut SpikingNetwork) {
    for neuron in &mut network.neurons {
        write_alternating_trace_values(&mut neuron.eligibility);
    }
}

fn write_alternating_trace_values(traces: &mut [EligibilityTrace]) {
    for (channel, trace) in traces.iter_mut().enumerate() {
        trace.value = [0.5_f32, -0.25][channel % 2];
    }
}

/// Seeded, steady-state normal/frozen steps across bank sizes. Binary stimuli
/// and nonzero weights exercise dense spike output rather than empty lists.
fn bench_engine_step_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_step_allocations");
    for size in [16, 64, 512] {
        for frozen in [false, true] {
            let label = if frozen { "frozen" } else { "normal" };
            group.bench_with_input(BenchmarkId::new(label, size), &size, |b, &size| {
                let mut network = SpikingNetwork::with_dimensions(size, 5, size);
                for neuron in &mut network.neurons {
                    neuron.weights.fill(2.0 / size as f32);
                }
                let stimuli = vec![1.0; size];
                let modulators = NeuroModulators::default();
                let mut rng = StdRng::seed_from_u64(190);
                let mut step = || {
                    if frozen {
                        network.step_frozen_with_rng(
                            black_box(&stimuli),
                            black_box(&modulators),
                            &mut rng,
                        )
                    } else {
                        network.step_with_rng(black_box(&stimuli), black_box(&modulators), &mut rng)
                    }
                    .unwrap()
                };
                for _ in 0..200 {
                    black_box(step());
                }
                b.iter(|| black_box(step()));
            });
        }
    }
    group.finish();
}

fn channel_drive_stimuli(steps: usize) -> Vec<Vec<f32>> {
    (0..steps)
        .map(|step| {
            (0..64)
                .map(|ch| [0.009, -0.25, 0.5, 1.5][(ch + step) % 4])
                .collect()
        })
        .collect()
}

/// Fixed seeded batches make before/after runs process identical spike and
/// learning workloads. Checkpoint restoration and RNG construction are untimed.
fn bench_engine_channel_drive(c: &mut Criterion) {
    const STEPS: usize = 32;
    let stimuli = channel_drive_stimuli(STEPS);
    let mut group = c.benchmark_group("engine_channel_drive");
    group.throughput(Throughput::Elements(STEPS as u64));

    for (label, dopamine) in [("unrewarded", 0.0_f32), ("rewarded", 0.9)] {
        let mut network = SpikingNetwork::with_dimensions(64, 5, 64);
        for neuron in &mut network.neurons {
            neuron.weights.fill(2.0 / 64.0);
        }
        let modulators = NeuroModulators {
            dopamine,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(191);
        for step in 0..200 {
            network
                .step_with_rng(&stimuli[step % STEPS], &modulators, &mut rng)
                .unwrap();
        }
        let checkpoint = serde_json::to_string(&network).unwrap();

        group.bench_function(label, |b| {
            b.iter_batched_ref(
                || {
                    (
                        serde_json::from_str::<SpikingNetwork>(&checkpoint).unwrap(),
                        StdRng::seed_from_u64(192),
                    )
                },
                |(network, rng)| {
                    for input in &stimuli {
                        black_box(
                            network
                                .step_with_rng(black_box(input), black_box(&modulators), rng)
                                .unwrap(),
                        );
                    }
                    black_box(&*network);
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

fn bench_stdp_weight_update(c: &mut Criterion) {
    let params = StdpParams::default();

    c.bench_function("stdp_weight_update", |b| {
        let mut weight = 0.5;
        let pre_time = 0i64;
        let post_time = 5i64;

        b.iter(|| {
            weight = apply_classical_stdp(
                black_box(pre_time),
                black_box(post_time),
                black_box(weight),
                black_box(&params),
            );
        });
    });
}

fn bench_hebbian_network_update(c: &mut Criterion) {
    let mut network = HebbianIzhikevichNetwork::new(10);

    // Simulate some spikes
    for t in 0..50i64 {
        network.neurons[0].step_with_time(10.0, t);
        network.neurons[1].step_with_time(10.0, t + 5);
    }

    c.bench_function("hebbian_network_update", |b| {
        b.iter(|| {
            network.update_weights(black_box(0), black_box(1));
        });
    });
}

fn bench_stdp_delta_t_calculation(c: &mut Criterion) {
    c.bench_function("stdp_delta_t_calculation", |b| {
        b.iter(|| {
            let pre_time = black_box(0i64);
            let post_time = black_box(5i64);
            let delta_t = (post_time - pre_time) as f32;

            let dw = if delta_t >= 0.0 {
                RM_STDP_A_PLUS * (-delta_t / RM_STDP_TAU_PLUS).exp()
            } else {
                -RM_STDP_A_MINUS * (delta_t / RM_STDP_TAU_MINUS).exp()
            };
            black_box(dw);
        });
    });
}

fn bench_stdp_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdp_network_size");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut network = HebbianIzhikevichNetwork::new(size);
            b.iter(|| {
                for pre in 0..size {
                    for post in 0..size {
                        network.update_weights(pre, post);
                    }
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_classical_stdp,
    bench_eligibility_trace_decay,
    bench_eligibility_trace_accumulate,
    bench_reward_conversion,
    bench_engine_step_with_traces,
    bench_engine_banked_trace_decay,
    bench_engine_step_allocations,
    bench_engine_channel_drive,
    bench_stdp_weight_update,
    bench_hebbian_network_update,
    bench_stdp_delta_t_calculation,
    bench_stdp_scaling
);
criterion_main!(benches);
