# neuromod

A generalized Rust library for spiking neural networks (SNNs), centered on biologically grounded neuron models, neuromodulation, and plasticity.

`neuromod` is designed to be a reusable core: topology-neutral at initialization, dynamically sizable at runtime, and strict about input shape validation.

## Highlights

- Dynamic network sizing with `SpikingNetwork::with_dimensions(...)`
- Backward-compatible default constructor: `SpikingNetwork::new()`
- Strict step contract: `Result<Vec<usize>, StepError>`
- Neutral initialization (blank synaptic weights; no hardcoded domain topology)
- Canonical neuron models included:
  - Lapicque
  - LIF
  - Izhikevich
  - FitzHugh-Nagumo
  - Hodgkin-Huxley
- Classical Hebbian STDP utilities and reward-modulated learning components

## Installation

```toml
[dependencies]
neuromod = "0.3"
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

`NeuroModulators` supports both direct control and signal-derived initialization.

```rust
use neuromod::NeuroModulators;

fn main() {
    // (thermal_signal, power_signal, throughput_signal, timing_signal)
    let mut mods = NeuroModulators::from_signals(75.0, 300.0, 0.05, 2640.0);

    mods.add_reward(0.2);
    mods.add_stress(0.1);
    mods.boost_focus(0.3);
    mods.add_aux_reward(0.4);
    mods.decay();

    println!("dopamine={:.3}, aux={:.3}", mods.dopamine, mods.aux_dopamine);
}
```

## Included Components

- `SpikingNetwork`, `StepError`
- `NeuroModulators`
- Neuron models:
  - `LifNeuron`
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
cargo bench --no-run
```

## License

GPL-3.0

## Links

- Crates.io: https://crates.io/crates/neuromod
- Docs.rs: https://docs.rs/neuromod
- Repository: https://github.com/Limen-Neural/neuromod
