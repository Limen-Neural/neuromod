use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use neuromod::rm_stdp::{
    EligibilityTrace, RM_STDP_A_MINUS, RM_STDP_A_PLUS, RM_STDP_TAU_MINUS, RM_STDP_TAU_PLUS,
    RmStdpConfig,
};
use neuromod::{
    HebbianIzhikevichNetwork, NeuroModulators, SpikingNetwork, StdpParams, apply_classical_stdp,
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
/// for the trace -> weight conversion.
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

    for (label, dopamine) in [("unrewarded", 0.0_f32), ("rewarded", 0.9)] {
        group.bench_function(label, |b| {
            let mut network = SpikingNetwork::with_dimensions(64, 5, 64);
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
    bench_stdp_weight_update,
    bench_hebbian_network_update,
    bench_stdp_delta_t_calculation,
    bench_stdp_scaling
);
criterion_main!(benches);
