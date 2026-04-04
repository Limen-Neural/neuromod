use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use neuromod::{
    LifNeuron, IzhikevichNeuron, LapicqueNeuron, HodgkinHuxleyNeuron, 
    FitzHughNagumoNeuron, SpikingNetwork, NeuroModulators
};

fn bench_neuron_memory_size(c: &mut Criterion) {
    c.bench_function("lif_neuron_size", |b| {
        b.iter(|| {
            let neuron = LifNeuron::new();
            black_box(std::mem::size_of_val(&neuron));
        });
    });
    
    c.bench_function("izhikevich_neuron_size", |b| {
        b.iter(|| {
            let neuron = IzhikevichNeuron::new_regular_spiking();
            black_box(std::mem::size_of_val(&neuron));
        });
    });
    
    c.bench_function("lapicque_neuron_size", |b| {
        b.iter(|| {
            let neuron = LapicqueNeuron::new();
            black_box(std::mem::size_of_val(&neuron));
        });
    });
    
    c.bench_function("hodgkin_huxley_neuron_size", |b| {
        b.iter(|| {
            let neuron = HodgkinHuxleyNeuron::new();
            black_box(std::mem::size_of_val(&neuron));
        });
    });
    
    c.bench_function("fitzhugh_nagumo_neuron_size", |b| {
        b.iter(|| {
            let neuron = FitzHughNagumoNeuron::new();
            black_box(std::mem::size_of_val(&neuron));
        });
    });
}

fn bench_network_memory_overhead(c: &mut Criterion) {
    c.bench_function("spiking_network_size", |b| {
        b.iter(|| {
            let network = SpikingNetwork::new();
            black_box(std::mem::size_of_val(&network));
        });
    });
    
    c.bench_function("neuromodulators_size", |b| {
        b.iter(|| {
            let modulators = NeuroModulators::default();
            black_box(std::mem::size_of_val(&modulators));
        });
    });
}

fn bench_network_allocation(c: &mut Criterion) {
    c.bench_function("network_allocation", |b| {
        b.iter(|| {
            let _network = SpikingNetwork::new();
        });
    });
}

fn bench_neuron_vector_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_vector_allocation");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let neurons: Vec<LifNeuron> = (0..size).map(|_| LifNeuron::new()).collect();
                black_box(neurons);
            });
        });
    }
    
    group.finish();
}

fn bench_weights_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("weights_allocation");
    
    for size in [16, 64, 256, 1024].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let weights: Vec<f32> = vec![0.5; size];
                black_box(weights);
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_neuron_memory_size,
    bench_network_memory_overhead,
    bench_network_allocation,
    bench_neuron_vector_allocation,
    bench_weights_allocation
);
criterion_main!(benches);
