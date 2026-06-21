use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use neuromod::{NeuroModulators, SpikingNetwork};

fn bench_network_step_baseline(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();

    c.bench_function("network_step_baseline", |b| {
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels");
        });
    });
}

fn bench_network_step_with_dopamine(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators {
        dopamine: 0.8,
        ..Default::default()
    };

    c.bench_function("network_step_with_dopamine", |b| {
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels");
        });
    });
}

fn bench_network_step_with_norepinephrine(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators {
        norepinephrine: 0.5,
        ..Default::default()
    };

    c.bench_function("network_step_with_norepinephrine", |b| {
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels");
        });
    });
}

fn bench_network_step_with_acetylcholine(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators {
        acetylcholine: 0.8,
        ..Default::default()
    };

    c.bench_function("network_step_with_acetylcholine", |b| {
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels");
        });
    });
}

fn bench_network_step_with_all_modulators(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators {
        dopamine: 0.8,
        norepinephrine: 0.3,
        acetylcholine: 0.7,
        serotonin: 0.5,
    };

    c.bench_function("network_step_with_all_modulators", |b| {
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels");
        });
    });
}

fn bench_modulator_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulator_comparison");

    group.bench_function("baseline", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let modulators = NeuroModulators::default();
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels")
        });
    });

    group.bench_function("high_dopamine", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let modulators = NeuroModulators {
            dopamine: 0.9,
            ..Default::default()
        };
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels")
        });
    });

    group.bench_function("high_norepinephrine", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let modulators = NeuroModulators {
            norepinephrine: 0.9,
            ..Default::default()
        };
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels")
        });
    });

    group.bench_function("high_acetylcholine", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let modulators = NeuroModulators {
            acetylcholine: 0.9,
            ..Default::default()
        };
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels")
        });
    });

    group.bench_function("all_active", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let modulators = NeuroModulators {
            dopamine: 0.7,
            norepinephrine: 0.3,
            acetylcholine: 0.7,
            serotonin: 0.6,
        };
        b.iter(|| {
            network
                .step(black_box(&stimuli), black_box(&modulators))
                .expect("stimuli length must match network channels")
        });
    });

    group.finish();
}

fn bench_dopamine_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dopamine_scaling");

    for dopamine in [0.0, 0.2, 0.5, 0.8, 1.0].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(dopamine),
            dopamine,
            |b, &dopamine| {
                let mut network = SpikingNetwork::new();
                let stimuli = [0.5f32; 16];
                let modulators = NeuroModulators {
                    dopamine,
                    ..Default::default()
                };
                b.iter(|| {
                    network
                        .step(black_box(&stimuli), black_box(&modulators))
                        .expect("stimuli length must match network channels")
                });
            },
        );
    }

    group.finish();
}

fn bench_modulator_decay(c: &mut Criterion) {
    let modulators = NeuroModulators {
        dopamine: 1.0,
        serotonin: 1.0,
        acetylcholine: 1.0,
        norepinephrine: 1.0,
    };

    c.bench_function("modulator_decay", |b| {
        let mut mods = modulators;
        b.iter(|| {
            mods.decay();
        });
    });
}

fn bench_modulator_operations(c: &mut Criterion) {
    c.bench_function("modulator_add_reward", |b| {
        let mut modulators = NeuroModulators::default();
        b.iter(|| {
            modulators.add_reward(black_box(0.5));
        });
    });

    c.bench_function("modulator_add_norepinephrine", |b| {
        let mut modulators = NeuroModulators::default();
        b.iter(|| {
            modulators.add_norepinephrine(black_box(0.5));
        });
    });

    c.bench_function("modulator_boost_focus", |b| {
        let mut modulators = NeuroModulators::default();
        b.iter(|| {
            modulators.boost_focus(black_box(0.5));
        });
    });
}

criterion_group!(
    benches,
    bench_network_step_baseline,
    bench_network_step_with_dopamine,
    bench_network_step_with_norepinephrine,
    bench_network_step_with_acetylcholine,
    bench_network_step_with_all_modulators,
    bench_modulator_comparison,
    bench_dopamine_scaling,
    bench_modulator_decay,
    bench_modulator_operations
);
criterion_main!(benches);
