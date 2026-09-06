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

> **IDE users:** these targets are declared with `harness = false` (required for
> Criterion's own runner to execute at all). Run them via a terminal or a plain
> **Cargo** run configuration. Don't use a "Run Test" / "Debug Test" gutter
> action — that expects the structured libtest protocol, which `harness = false`
> targets don't emit, and will fail with an IDE-side error (e.g. RustRover's
> "test frame quit unexpectedly") even though `cargo bench` itself succeeds.

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
- `eligibility_trace_accumulate_ltp` - Kernel evaluation and accumulation for a pre-before-post pair
- `eligibility_trace_accumulate_ltd` - Same for post-before-pre (depression)
- `rm_stdp_trace_to_weight` - One synapse's decay -> accumulate -> dopamine-gated conversion
- `engine_step/unrewarded` / `engine_step/rewarded` - Full `SpikingNetwork::step` (64 LIF over 64 channels) carrying trace bookkeeping, with the dopamine gate shut and open
- `stdp_weight_update` - Complete weight update cycle
- `hebbian_network_update` - Hebbian network weight update
- `stdp_delta_t_calculation` - Spike timing difference calculation
- `stdp_network_size` - STDP scaling with network size (10, 50, 100, 200 neurons)

### 3. Memory Benchmarks (`memory_bench.rs`)

**Answers: "What's the memory overhead?"**

Benchmarks memory usage and allocation:
- `*_neuron_size`, `spiking_network_size`, `neuromodulators_size` - time a construct-then-`size_of_val` call; at this scale the reported nanoseconds/picoseconds reflect construction and call overhead, not the struct's byte size. See [Measured Baseline](#measured-baseline) below for the actual `size_of::<T>()` byte counts, taken separately.
- `network_allocation` - Network initialization time
- `neuron_vector_allocation` - Vector allocation scaling (10, 50, 100, 500, 1000 neurons)
- `weights_allocation` - Weight vector allocation scaling (16, 64, 256, 1024 weights)

### 4. Modulation Benchmarks (`modulation_bench.rs`)

**Answers: "How does reward modulation affect performance?"**

Benchmarks neuromodulator impact on network performance:
- `network_step_baseline` - Network step without modulators
- `network_step_with_dopamine` - Network step with high dopamine (reward)
- `network_step_with_norepinephrine` - Network step with high norepinephrine (stress/arousal)
- `network_step_with_acetylcholine` - Network step with high acetylcholine (focus)
- `network_step_with_all_modulators` - Network step with all modulators active
- `modulator_comparison` - Direct comparison of modulator states
- `dopamine_scaling` - Performance scaling with dopamine levels (0.0 to 1.0)
- `modulator_decay` - Modulator decay operation
- `modulator_operations` - Individual modulator operations (add_reward, add_norepinephrine, boost_focus)

## Measured Baseline

This crate makes no unqualified "high-performance" claim. The timing tables below are
what the benches in this directory actually report; the byte-size table in the "Memory"
section further down is a separate `size_of::<T>()` measurement, not a bench result (see
that section for how it was taken). All of it is a single-machine snapshot, not a
performance SLA or a cross-crate comparison — re-run the suite yourself with `cargo bench`
before relying on any of this for a decision.

- **Commit:** `5a6ac98`
- **Date:** 2026-09-06
- **Toolchain:** `rustc 1.97.1` (pinned in `rust-toolchain.toml`), `profile.bench` inherits
  `profile.release` (`opt-level = 3`, `lto = true`, `codegen-units = 1`)
- **Machine:** 4-core Intel Xeon @ 2.10GHz cloud CI container (not an isolated benchmarking
  rig — expect run-to-run noise; several benchmarks below report outliers)
- **Command:** `cargo bench --all-features --bench <name> -- --save-baseline gh77-2026-09-06`
- Times are Criterion's reported point estimate — the middle value of its
  `[lower estimate upper]` confidence-interval output. This is not a statistical median,
  and the interval is not guaranteed to be symmetric around it.

### Neuron step (`neuron_bench.rs`)

| Benchmark | Point estimate |
| --- | --- |
| `lif_check_fire` | 0.55 ns |
| `lapicque_step` | 1.64 ns |
| `lif_integrate` | 2.35 ns |
| `lif_full_step` | 4.64 ns |
| `izhikevich_step` | 10.05 ns |
| `fitzhugh_nagumo_step` | 321 ns |
| `hodgkin_huxley_step` | 627 ns |

On this run the ranking is LIF < Izhikevich < FitzHugh-Nagumo < Hodgkin-Huxley, consistent
with their relative model complexity: LIF integrates a single membrane-potential state,
Izhikevich and FitzHugh-Nagumo each track two coupled state variables, and Hodgkin-Huxley
tracks four (membrane potential plus three gating variables `m`, `h`, `n`).

### STDP / plasticity (`stdp_bench.rs`)

| Benchmark | Point estimate |
| --- | --- |
| `classical_stdp_ltp` | 0.58 ns |
| `classical_stdp_ltd` | 0.61 ns |
| `eligibility_trace_accumulate_ltp` | 2.89 ns |
| `eligibility_trace_accumulate_ltd` | 3.22 ns |
| `stdp_delta_t_calculation` | 3.37 ns |
| `eligibility_trace_decay` | 36.5 ns |
| `stdp_weight_update` | 7.11 ns |
| `hebbian_network_update` | 13.3 ns |
| `rm_stdp_trace_to_weight` | 8.26 ns |

Full `SpikingNetwork::step` with trace bookkeeping (64 LIF neurons, 64 channels, steady
state after 200 warm-up steps):

| Benchmark | Point estimate |
| --- | --- |
| `engine_step/unrewarded` (dopamine = 0.0) | 35.6 µs |
| `engine_step/rewarded` (dopamine = 0.9) | 51.2 µs |

`stdp_network_size` (`HebbianIzhikevichNetwork::update_weights` over every pre/post pair,
fully connected):

| Network size | Point estimate |
| --- | --- |
| 10 | 134 ns |
| 50 | 3.14 µs |
| 100 | 10.9 µs |
| 200 | 42.3 µs |

10 → 50 neurons (5x) costs ~23x; 50 → 100 (2x) costs ~3.5x; 100 → 200 (2x) costs ~3.9x —
close enough to O(n²) to call the fully-connected update quadratic in neuron count, as the
loop structure implies.

### Memory (`memory_bench.rs`)

`size_of::<T>()` byte counts for the `x86_64-unknown-linux-gnu` target (layout can vary by
target; measured separately from the timing benchmarks below, which time a constructor +
`size_of_val` call rather than reporting a byte size):

| Type | Size |
| --- | --- |
| `NeuroModulators` | 16 bytes |
| `FitzHughNagumoNeuron` | 20 bytes |
| `IzhikevichNeuron` | 32 bytes |
| `HodgkinHuxleyNeuron` | 48 bytes |
| `LapicqueNeuron` | 56 bytes |
| `LifNeuron` | 80 bytes |
| `SpikingNetwork` (empty banks) | 144 bytes |

`LifNeuron` and `SpikingNetwork` hold `Vec` fields (`weights`, `eligibility`, the neuron
banks themselves); the sizes above are the fixed struct layout only (each `Vec` is a
pointer/len/cap triple) and exclude heap-allocated contents, which scale with channel and
neuron count.

Allocation timing:

| Benchmark | Point estimate | Throughput |
| --- | --- | --- |
| `network_allocation` (`SpikingNetwork::new()`, 16 LIF + 5 Izhikevich, 16 channels) | 1.01 µs | — |
| `neuron_vector_allocation/10` | 22.1 ns | 452 Melem/s |
| `neuron_vector_allocation/50` | 133 ns | 375 Melem/s |
| `neuron_vector_allocation/100` | 234 ns | 428 Melem/s |
| `neuron_vector_allocation/500` | 954 ns | 524 Melem/s |
| `neuron_vector_allocation/1000` | 2.59 µs | 387 Melem/s |
| `weights_allocation/16` | 10.6 ns | 1.51 Gelem/s |
| `weights_allocation/64` | 11.3 ns | 5.66 Gelem/s |
| `weights_allocation/256` | 17.0 ns | 15.1 Gelem/s |
| `weights_allocation/1024` | 73.3 ns | 14.0 Gelem/s |

### Modulation (`modulation_bench.rs`)

Full `SpikingNetwork::step` (default `new()`: 16 LIF, 5 Izhikevich, 16 channels):

| Benchmark | Point estimate |
| --- | --- |
| `network_step_baseline` (no modulators) | 1.24 µs |
| `network_step_with_dopamine` (0.8) | 1.27 µs |
| `network_step_with_norepinephrine` (0.5) | 1.28 µs |
| `network_step_with_acetylcholine` (0.8) | 1.28 µs |
| `network_step_with_all_modulators` | 1.37 µs |

`modulator_comparison` (each variant builds its own fresh network) and `dopamine_scaling`
land in the same 1.05–1.38 µs band without a clean monotonic trend, and several of the
groups above report 7–22% outliers — on this machine, single-run modulator overhead is not
cleanly separable from sample-to-sample noise. Treat "modulators add roughly low-double-digit-percent
overhead over baseline, or less" as the honest read of this data, not a precise percentage,
and do not treat 5%-or-any-other threshold as a target this baseline demonstrates.

Modulator primitive operations (not full network steps):

| Benchmark | Point estimate |
| --- | --- |
| `modulator_add_reward` | 1.83 ns |
| `modulator_boost_focus` | 1.86 ns |
| `modulator_add_norepinephrine` | 2.16 ns |
| `modulator_decay` | 36.4 ns |

## Continuous Benchmarking

To track performance over time on your own machine:
```bash
# Save baseline
cargo bench -- --save-baseline main

# After changes, compare
cargo bench -- --baseline main
```

Criterion will generate comparison reports showing performance regressions or
improvements. Criterion's `--save-baseline` output lives under the git-ignored
`target/criterion/`, so it never lands in this repo — the *Measured Baseline* table
above is the durable, reviewable record; regenerate it (and update the commit/date
above) whenever a change to `src/` is expected to move these numbers materially.
