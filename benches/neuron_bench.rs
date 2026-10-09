use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use neuromod::{
    FitzHughNagumoNeuron, HodgkinHuxleyNeuron, IzhikevichNeuron, LapicqueNeuron, LifNeuron,
    SparseGifHiddenLayer, SparseGifLayerConfig,
}; // Import neuron types
use std::hint::black_box;

/// Benchmark LIF neuron integration
fn bench_lif_integrate(c: &mut Criterion) {
    let mut neuron = LifNeuron::new();
    // Black box to prevent optimization

    // Benchmark function
    c.bench_function("lif_integrate", |b| {
        b.iter(|| {
            // Iterate benchmark
            neuron.integrate(black_box(0.5)); // Black box to prevent optimization
        });
    });
}

fn bench_lif_check_fire(c: &mut Criterion) {
    c.bench_function("lif_check_fire/firing", |b| {
        b.iter_batched_ref(
            || {
                let mut neuron = LifNeuron::new();
                neuron.membrane_potential = 0.03;
                neuron
            },
            |neuron| {
                let spike = neuron.check_fire();
                black_box((spike, neuron.membrane_potential));
            },
            // Keep input batch size small enough to remain cache-resident and avoid memory bandwidth bottlenecks.
            BatchSize::LargeInput,
        );
    });
}

fn bench_lif_full_step(c: &mut Criterion) {
    // Benchmark function
    let mut neuron = LifNeuron::new(); // Create LIF neuron

    c.bench_function("lif_full_step", |b| {
        // Benchmark function
        b.iter(|| {
            // Iterate benchmark
            neuron.integrate(black_box(0.5)); // Integrate
            let _ = neuron.check_fire(); // Check if neuron fires
        });
    });
}

fn bench_izhikevich_step(c: &mut Criterion) {
    let mut neuron = IzhikevichNeuron::new_regular_spiking();

    c.bench_function("izhikevich_step", |b| {
        b.iter(|| {
            neuron.step(black_box(10.0));
        });
    });
}

fn bench_lapicque_step(c: &mut Criterion) {
    // Benchmark function
    let mut neuron = LapicqueNeuron::new(); // Create Lapicque neuron

    c.bench_function("lapicque_step", |b| {
        // Benchmark function
        b.iter(|| {
            // Iterate benchmark
            neuron.integrate(black_box(10.0)); // Integrate
            let _ = neuron.check_for_spike(black_box(0)); // Check for spike
        });
    });
}

fn bench_hodgkin_huxley_step(c: &mut Criterion) {
    let mut neuron = HodgkinHuxleyNeuron::new();

    c.bench_function("hodgkin_huxley_step", |b| {
        b.iter(|| {
            neuron.step(black_box(10.0), black_box(0.05));
        });
    });
}

fn bench_fitzhugh_nagumo_step(c: &mut Criterion) {
    // Benchmark function
    let mut neuron = FitzHughNagumoNeuron::new(); // Create FitzHugh-Nagumo neuron

    c.bench_function("fitzhugh_nagumo_step", |b| {
        // Benchmark function
        b.iter(|| {
            // Iterate benchmark
            neuron.step(black_box(10.0), black_box(0.5)); // Step
        });
    });
}

fn bench_neuron_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_types");

    group.bench_function("LIF", |b| {
        let mut neuron = LifNeuron::new();
        b.iter(|| {
            neuron.integrate(black_box(0.5)); // Integrate
            let _ = neuron.check_fire(); // Check if neuron fires
        });
    });

    group.bench_function("Izhikevich", |b| {
        // Benchmark function
        let mut neuron = IzhikevichNeuron::new_regular_spiking(); // Create Izhikevich neuron
        b.iter(|| {
            // Iterate benchmark
            neuron.step(black_box(10.0)); // Step
        });
    });

    group.bench_function("Lapicque", |b| {
        // Benchmark function
        let mut neuron = LapicqueNeuron::new(); // Create Lapicque neuron
        b.iter(|| {
            // Iterate benchmark
            neuron.integrate(black_box(10.0)); // Integrate
            let _ = neuron.check_for_spike(black_box(0)); // Check for spike
        });
    });

    group.finish(); // Finish benchmark group
}

fn bench_sparse_gif_layer(c: &mut Criterion) {
    let stimuli: Vec<f32> = (0..64).map(|channel| (channel % 2) as f32).collect();
    let mut group = c.benchmark_group("sparse_gif_layer");
    for neurons in [8, 64, 256] {
        let layer = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
            num_inputs: stimuli.len(),
            num_neurons: neurons,
            fan_in: 16,
            seed: 192,
            ..Default::default()
        })
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("step_into", neurons),
            &layer,
            |b, layer| {
                let mut layer = layer.clone();
                let mut spikes = vec![false; neurons];
                for _ in 0..200 {
                    layer.step_into(&stimuli, &mut spikes).unwrap();
                }
                b.iter(|| {
                    layer
                        .step_into(black_box(&stimuli), black_box(&mut spikes))
                        .unwrap();
                    black_box(&spikes);
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("step", neurons), &layer, |b, layer| {
            let mut layer = layer.clone();
            for _ in 0..200 {
                layer.step(&stimuli).unwrap();
            }
            b.iter(|| black_box(layer.step(black_box(&stimuli)).unwrap()));
        });
    }
    group.finish();
}

// Criterion benchmark group
criterion_group!(
    benches,
    bench_lif_integrate,
    bench_lif_check_fire,
    bench_lif_full_step,
    bench_izhikevich_step,
    bench_lapicque_step,
    bench_hodgkin_huxley_step,
    bench_fitzhugh_nagumo_step,
    bench_neuron_comparison,
    bench_sparse_gif_layer
);
criterion_main!(benches);
