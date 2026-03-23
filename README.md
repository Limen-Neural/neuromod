# Neuromod - Reward-Modulated Spiking Neural Networks

[![Crates.io](https://img.shields.io/crates/v/neuromod.svg)](https://crates.io/crates/neuromod)
[![Docs.rs](https://img.shields.io/badge/docs.rs-neuromod-blue.svg)](https://docs.rs/neuromod)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL_3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub](https://img.shields.io/badge/GitHub-rmems/neuromod-black.svg)](https://github.com/rmems/neuromod)

**v0.2.1** — Now with lean **mining_dopamine** reward signal.

A lightweight, zero-unsafe Rust crate for neuromorphic computing. Designed as the official Rust backend for **Spikenaut-v2** — the 16-channel neuromorphic HFT + FPGA system.

## Features

- LIF + Izhikevich neurons
- Reward-modulated STDP learning
- Full neuromodulator system (dopamine, cortisol, acetylcholine, tempo, **mining_dopamine**)
- Lean MiningReward EMA calculation (no heavy dependencies)
- Sub-1 µs modulator updates
- ~1.6 KB memory footprint
- no_std + Q8.8 fixed-point FPGA .mem export ready
- jlrs zero-copy interop for Julia training

## Quick Start

```rust
use neuromod::{SpikingNetwork, NeuroModulators, MiningReward, HftReward};

let mut network = SpikingNetwork::new();

// 16-channel telemetry stimuli
let stimuli = [0.5f32; 16];

// Create modulators + mining reward
let mut reward = MiningReward::new();
let mining_dopamine = reward.compute(hashrate, power_draw, gpu_temp);

let modulators = NeuroModulators {
    dopamine: 0.7,
    cortisol: 0.3,
    acetylcholine: 0.6,
    tempo: 1.0,
    mining_dopamine,  // ← new in v0.2.1
};

let spikes = network.step(&stimuli, &modulators);
```

## Architecture

### Neuron Banks (16 channels)
- 8 bear/bull asset pairs (DNX, QUAI, QUBIC, KASPA, XMR, OCEAN, VERUS + thermal)
- Coincidence detector + global inhibitor

### Neuromodulator System
- **Dopamine** – market/sync reward
- **Cortisol** – stress/inhibition
- **Acetylcholine** – focus/SNR
- **Tempo** – clock scaling
- **mining_dopamine** (v0.2.1) – EMA-smoothed mining efficiency reward

### High-Frequency Trading
```rust
// Real-time market processing
let market_data = get_market_data();
let stimuli = normalize_market_data(&market_data);
let spikes = network.step(&stimuli, &modulators);

// Execute trades based on neural spikes
for &neuron_id in &spikes {
    if neuron_id < 14 { // Trading neurons only
        execute_trade(neuron_id);
    }
}
```

### Hardware Monitoring
```rust
// Create modulators from GPU telemetry
let modulators = NeuroModulators::from_telemetry(
    gpu_temp,
    power_draw,
    hashrate,
    gpu_clock
);

// Network adapts to hardware conditions
let spikes = network.step(&stimuli, &modulators);

// Check stress levels
if modulators.is_stressed() {
    reduce_mining_intensity();
}
```

## Performance

- **Latency**: < 1μs per network step
- **Memory**: ~2KB for full 16-neuron network
- **Throughput**: > 1M steps/second on single core
- **Deterministic**: No allocations in hot path

## FPGA Integration

The architecture is designed for FPGA deployment with:
- Fixed-point arithmetic support
- Parallel neuron evaluation
- Hardware STDP implementation
- Low-latency spike propagation

## License

Licensed under the GNU General Public License, Version 3.0 ([GPL-3.0](LICENSE) or https://www.gnu.org/licenses/gpl-3.0)

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

- **GitHub**: https://github.com/rmems/neuromod
- **Crates.io**: https://crates.io/crates/neuromod

---

*Built for the Spikenaut HFT system - neuromorphic computing for real-time trading*
