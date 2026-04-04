use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuromod::{apply_classical_stdp, StdpParams, HebbianIzhikevichNetwork};
use neuromod::rm_stdp::{EligibilityTrace, RM_STDP_A_PLUS, RM_STDP_A_MINUS, RM_STDP_TAU_PLUS, RM_STDP_TAU_MINUS};

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
    bench_stdp_weight_update,
    bench_hebbian_network_update,
    bench_stdp_delta_t_calculation,
    bench_stdp_scaling
);
criterion_main!(benches);
