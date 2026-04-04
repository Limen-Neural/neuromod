use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuromod::{SpikingNetwork, NeuroModulators};

fn bench_network_step_baseline(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let modulators = NeuroModulators::default();
    
    c.bench_function("network_step_baseline", |b| {
        b.iter(|| {
            network.step(black_box(&stimuli), black_box(&modulators));
        });
    });
}

fn bench_network_step_with_dopamine(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let mut modulators = NeuroModulators::default();
    modulators.dopamine = 0.8;
    
    c.bench_function("network_step_with_dopamine", |b| {
        b.iter(|| {
            network.step(black_box(&stimuli), black_box(&modulators));
        });
    });
}

fn bench_network_step_with_cortisol(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let mut modulators = NeuroModulators::default();
    modulators.cortisol = 0.5;
    
    c.bench_function("network_step_with_cortisol", |b| {
        b.iter(|| {
            network.step(black_box(&stimuli), black_box(&modulators));
        });
    });
}

fn bench_network_step_with_acetylcholine(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let mut modulators = NeuroModulators::default();
    modulators.acetylcholine = 0.8;
    
    c.bench_function("network_step_with_acetylcholine", |b| {
        b.iter(|| {
            network.step(black_box(&stimuli), black_box(&modulators));
        });
    });
}

fn bench_network_step_with_all_modulators(c: &mut Criterion) {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5f32; 16];
    let mut modulators = NeuroModulators::default();
    modulators.dopamine = 0.8;
    modulators.cortisol = 0.3;
    modulators.acetylcholine = 0.7;
    modulators.tempo = 1.5;
    
    c.bench_function("network_step_with_all_modulators", |b| {
        b.iter(|| {
            network.step(black_box(&stimuli), black_box(&modulators));
        });
    });
}

fn bench_modulator_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("modulator_comparison");
    
    // Baseline (no modulators)
    group.bench_function("baseline", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let modulators = NeuroModulators::default();
        b.iter(|| network.step(black_box(&stimuli), black_box(&modulators)));
    });
    
    // High dopamine (reward state)
    group.bench_function("high_dopamine", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let mut modulators = NeuroModulators::default();
        modulators.dopamine = 0.9;
        b.iter(|| network.step(black_box(&stimuli), black_box(&modulators)));
    });
    
    // High cortisol (stress state)
    group.bench_function("high_cortisol", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let mut modulators = NeuroModulators::default();
        modulators.cortisol = 0.9;
        b.iter(|| network.step(black_box(&stimuli), black_box(&modulators)));
    });
    
    // High acetylcholine (focus state)
    group.bench_function("high_acetylcholine", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let mut modulators = NeuroModulators::default();
        modulators.acetylcholine = 0.9;
        b.iter(|| network.step(black_box(&stimuli), black_box(&modulators)));
    });
    
    // All modulators active
    group.bench_function("all_active", |b| {
        let mut network = SpikingNetwork::new();
        let stimuli = [0.5f32; 16];
        let mut modulators = NeuroModulators::default();
        modulators.dopamine = 0.7;
        modulators.cortisol = 0.3;
        modulators.acetylcholine = 0.7;
        modulators.tempo = 1.2;
        b.iter(|| network.step(black_box(&stimuli), black_box(&modulators)));
    });
    
    group.finish();
}

fn bench_dopamine_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dopamine_scaling");
    
    for dopamine in [0.0, 0.2, 0.5, 0.8, 1.0].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(dopamine), dopamine, |b, &dopamine| {
            let mut network = SpikingNetwork::new();
            let stimuli = [0.5f32; 16];
            let mut modulators = NeuroModulators::default();
            modulators.dopamine = dopamine;
            b.iter(|| network.step(black_box(&stimuli), black_box(&modulators)));
        });
    }
    
    group.finish();
}

fn bench_modulator_decay(c: &mut Criterion) {
    let mut modulators = NeuroModulators::default();
    modulators.dopamine = 1.0;
    modulators.cortisol = 1.0;
    modulators.acetylcholine = 1.0;
    
    c.bench_function("modulator_decay", |b| {
        let mut mods = modulators.clone();
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
    
    c.bench_function("modulator_add_stress", |b| {
        let mut modulators = NeuroModulators::default();
        b.iter(|| {
            modulators.add_stress(black_box(0.5));
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
    bench_network_step_with_cortisol,
    bench_network_step_with_acetylcholine,
    bench_network_step_with_all_modulators,
    bench_modulator_comparison,
    bench_dopamine_scaling,
    bench_modulator_decay,
    bench_modulator_operations
);
criterion_main!(benches);
