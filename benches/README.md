# Neuromod Benchmarks

This directory contains criterion-based benchmarks for the neuromod crate, designed to answer key performance questions about SNN training and plasticity.

## Running Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run a specific benchmark suite:
```bash
cargo bench --bench neuron_bench
cargo bench --bench stdp_bench
cargo bench --bench memory_bench
cargo bench --bench modulation_bench
```

Generate HTML reports (automatically generated in `target/criterion/`):
```bash
cargo bench -- --save-baseline main
```

Compare with a previous baseline:
```bash
cargo bench -- --baseline main
```

## Benchmark Suites

### 1. Neuron Benchmarks (`neuron_bench.rs`)

**Answers: "How fast is one neuron step?"**

Benchmarks individual neuron model performance:
- `lif_integrate` - LIF neuron integration step
- `lif_check_fire` - LIF neuron spike detection
- `lif_full_step` - Complete LIF neuron step (integrate + check)
- `izhikevich_step` - Izhikevich neuron step
- `lapicque_step` - Lapicque neuron step
- `hodgkin_huxley_step` - Hodgkin-Huxley neuron step
- `fitzhugh_nagumo_step` - FitzHugh-Nagumo neuron step
- `neuron_types` - Comparison across all neuron types

### 2. STDP Benchmarks (`stdp_bench.rs`)

**Answers: "How fast is STDP update?"**

Benchmarks synaptic plasticity operations:
- `classical_stdp_ltp` - Classical STDP with pre-before-post timing (LTP)
- `classical_stdp_ltd` - Classical STDP with post-before-pre timing (LTD)
- `eligibility_trace_decay` - Eligibility trace decay operation
- `stdp_weight_update` - Complete weight update cycle
- `hebbian_network_update` - Hebbian network weight update
- `stdp_delta_t_calculation` - Spike timing difference calculation
- `stdp_network_size` - STDP scaling with network size (10, 50, 100, 200 neurons)

### 3. Memory Benchmarks (`memory_bench.rs`)

**Answers: "What's the memory overhead?"**

Benchmarks memory usage and allocation:
- `*_neuron_size` - Size of each neuron type struct
- `spiking_network_size` - Size of the complete network
- `neuromodulators_size` - Size of modulator struct
- `network_allocation` - Network initialization time
- `neuron_vector_allocation` - Vector allocation scaling (10, 50, 100, 500, 1000 neurons)
- `weights_allocation` - Weight vector allocation scaling (16, 64, 256, 1024 weights)

### 4. Modulation Benchmarks (`modulation_bench.rs`)

**Answers: "How does reward modulation affect performance?"**

Benchmarks neuromodulator impact on network performance:
- `network_step_baseline` - Network step without modulators
- `network_step_with_dopamine` - Network step with high dopamine (reward)
- `network_step_with_cortisol` - Network step with high cortisol (stress)
- `network_step_with_acetylcholine` - Network step with high acetylcholine (focus)
- `network_step_with_all_modulators` - Network step with all modulators active
- `modulator_comparison` - Direct comparison of modulator states
- `dopamine_scaling` - Performance scaling with dopamine levels (0.0 to 1.0)
- `modulator_decay` - Modulator decay operation
- `modulator_operations` - Individual modulator operations (add_reward, add_stress, boost_focus)

## Interpreting Results

### Neuron Step Performance
- **LIF neurons** should be the fastest (simplest model)
- **Izhikevich** adds complexity but remains fast
- **Hodgkin-Huxley** is most computationally expensive (biophysically detailed)
- **FitzHugh-Nagumo** offers a middle ground with 2D dynamics

### STDP Performance
- LTP/LTD operations should be sub-microsecond
- Eligibility trace decay is a simple multiplication
- Network STDP scales quadratically with neuron count (fully connected)

### Memory Overhead
- LIF neurons: ~48-64 bytes (depending on weights)
- Izhikevich neurons: ~32 bytes (compact state)
- Hodgkin-Huxley: ~120 bytes (multiple gating variables)
- Network overhead dominated by weight matrices

### Modulation Impact
- Baseline performance: reference point
- Dopamine: enables learning (may add small overhead)
- Cortisol: stress modulation (minimal overhead)
- Acetylcholine: affects decay rates (minimal overhead)
- Combined modulators: should show minimal cumulative overhead

## Performance Targets

For a production-ready SNN library:
- **Neuron step**: < 100 ns for LIF, < 500 ns for Izhikevich
- **STDP update**: < 50 ns per synapse
- **Network step**: < 1 µs per neuron (including STDP)
- **Memory**: < 1 KB per neuron (including weights)
- **Modulation overhead**: < 5% of baseline step time

## Continuous Benchmarking

To track performance over time:
```bash
# Save baseline
cargo bench -- --save-baseline main

# After changes, compare
cargo bench -- --baseline main
```

Criterion will generate comparison reports showing performance regressions or improvements.
