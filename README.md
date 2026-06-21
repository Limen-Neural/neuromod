# neuromod

A generalized Rust library for spiking neural networks (SNNs), centered on biologically grounded neuron models, neuromodulation, and plasticity.

`neuromod` is designed to be a reusable core: topology-neutral at initialization, dynamically sizable at runtime, and strict about input shape validation.

## Highlights

- Dynamic network sizing with `SpikingNetwork::with_dimensions(...)`
- Backward-compatible default constructor: `SpikingNetwork::new()`
- Strict step contract: `Result<Vec<usize>, StepError>`
- Neutral initialization (blank synaptic weights; no hardcoded domain topology)
- Generic neuromodulators: dopamine, serotonin, acetylcholine, norepinephrine
- `GenericReward` trait for domain-specific reward shaping in downstream crates
- Canonical neuron models included:
  - Lapicque
  - LIF
  - GIF (Generalized Integrate-and-Fire)
  - Izhikevich
  - FitzHugh-Nagumo
  - Hodgkin-Huxley
- Classical Hebbian STDP utilities and reward-modulated learning components

## Installation

```toml
[dependencies]
neuromod = "0.5.0"
```

## Quick Start

```rust
use neuromod::{NeuroModulators, SpikingNetwork};

fn main() {
    let mut network = SpikingNetwork::new(); // default: 16 LIF, 5 Izh, 16 channels
    let stimuli = [0.5_f32; 16];
    let modulators = NeuroModulators::default();

    let spikes = network.step(&stimuli, &modulators).unwrap();
    println!("Spiking neuron indices: {spikes:?}");
}
```

## Dynamic Dimensions

```rust
use neuromod::{NeuroModulators, SpikingNetwork};

fn main() {
    let mut network = SpikingNetwork::with_dimensions(518, 5, 518);
    let modulators = NeuroModulators::default();
    let stimuli = vec![0.25_f32; 518];

    let spikes = network.step(&stimuli, &modulators).unwrap();
    println!("Spike count: {}", spikes.len());
}
```

## Step Errors (Shape Validation)

`step` validates that `stimuli.len() == num_channels` and returns an error on mismatch.

```rust
use neuromod::{NeuroModulators, SpikingNetwork, StepError};

fn main() {
    let mut network = SpikingNetwork::with_dimensions(32, 4, 32);
    let modulators = NeuroModulators::default();
    let bad_stimuli = vec![0.1_f32; 31];

    match network.step(&bad_stimuli, &modulators) {
        Ok(_) => unreachable!("expected length mismatch"),
        Err(StepError::InputLenMismatch { expected, got }) => {
            println!("InputLenMismatch: expected {expected}, got {got}");
        }
    }
}
```

## Neuromodulators

`NeuroModulators` supports direct control, signal-derived initialization via `SignalProfile`, and generic reward shaping.

```rust
use neuromod::{
    apply_neuromodulation, GenericReward, NeuroModulators, Observation, SignalProfile, UnitReward,
};

fn main() {
    let profile = SignalProfile::default();
    let mut mods = NeuroModulators::from_signals(&profile, 0.2, 0.1, 0.8, 0.9);

    mods.add_reward(0.2);
    mods.add_norepinephrine(0.1);
    mods.boost_focus(0.3);
    mods.add_serotonin(0.4);
    mods.decay();

    let reward = UnitReward;
    let obs = Observation::from_slice(&[0.5, 0.7]);
    mods.apply_reward(&reward, &obs);

    let mut weights = vec![1.0, 0.8];
    let mut thresholds = vec![0.20, 0.25];
    apply_neuromodulation(&mods, &mut weights, &mut thresholds);

    println!(
        "dopamine={:.3}, serotonin={:.3}, ne={:.3}",
        mods.dopamine, mods.serotonin, mods.norepinephrine
    );
}
```

For legacy hardware-calibrated signal mapping, use `SignalProfile::hardware_calibrated()`.

## Included Components

- `SpikingNetwork`, `StepError`
- `NeuroModulators`, `SignalProfile`, `Observation`, `GenericReward`, `UnitReward`
- `apply_neuromodulation`
- Neuron models:
  - `LifNeuron`
  - `GifNeuron`
  - `IzhikevichNeuron`
  - `LapicqueNeuron`
  - `FitzHughNagumoNeuron`
  - `HodgkinHuxleyNeuron`
- Learning/plasticity:
  - `apply_classical_stdp`, `StdpParams`, `HebbianIzhikevichNetwork`
  - `EligibilityTrace`, `RmStdpConfig`

## Examples

Run included examples:

```bash
cargo run --example basic
cargo run --example rstdp_demo
```

## Development

```bash
cargo check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo bench --no-run
```

## License

GPL-3.0

## Coverage

[![codecov](https://codecov.io/gh/Limen-Neural/neuromod/branch/main/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/neuromod)

## Links

- Crates.io: https://crates.io/crates/neuromod
- Docs.rs: https://docs.rs/neuromod
- Repository: https://github.com/Limen-Neural/neuromod
