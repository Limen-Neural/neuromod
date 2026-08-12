# neuromod

[![Crates.io](https://img.shields.io/crates/v/neuromod.svg)](https://crates.io/crates/neuromod)
[![docs.rs](https://docs.rs/neuromod/badge.svg)](https://docs.rs/neuromod)
[![License](https://img.shields.io/crates/l/neuromod.svg)](https://github.com/Limen-Neural/neuromod#license)
[![codecov](https://codecov.io/gh/Limen-Neural/neuromod/branch/main/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/neuromod)

Biologically grounded spiking neural network (SNN) primitives in Rust: a topology-neutral `SpikingNetwork` engine, generic neuromodulators, STDP building blocks, and standalone neuron models.

`neuromod` is a reusable core library: topology-neutral at initialization, dynamically sizable at runtime, and strict about input shape validation. Dual-licensed MIT OR Apache-2.0.

## Highlights

- Dynamic network sizing with `SpikingNetwork::with_dimensions(...)`
- Backward-compatible default constructor: `SpikingNetwork::new()`
- Strict step contract: `Result<Vec<usize>, StepError>`
- Neutral initialization (blank synaptic weights; no hardcoded domain topology)
- Generic neuromodulators: dopamine, serotonin, acetylcholine, norepinephrine
- `GenericReward` trait for domain-specific reward shaping in downstream crates
- Classical Hebbian STDP utilities and reward-modulated STDP types (`EligibilityTrace`, `RmStdpConfig`)

### Engine (`SpikingNetwork`)

The network engine integrates **two** neuron banks only:

- **LIF** (`LifNeuron`) — primary bank sized by `num_lif`
- **Izhikevich** (`IzhikevichNeuron`) — secondary bank sized by `num_izh`

Default construction: 16 LIF, 5 Izhikevich, 16 input channels.

### Standalone neuron models

These types ship in the crate for research and composition, but are **not** wired as alternate banks inside `SpikingNetwork`:

- Lapicque (`LapicqueNeuron`)
- GIF — Generalized Integrate-and-Fire (`GifNeuron`)
- FitzHugh–Nagumo (`FitzHughNagumoNeuron`)
- Hodgkin–Huxley (`HodgkinHuxleyNeuron`)

Use them directly; use `HebbianIzhikevichNetwork` for a small classical-STDP Izhikevich helper separate from `SpikingNetwork`.

## Requirements

| | |
|--|--|
| **MSRV** | **Rust 1.97.1** (`rust-version` in `Cargo.toml`) |
| **Edition** | 2024 |
| **Pin** | [`rust-toolchain.toml`](rust-toolchain.toml) (channel `1.97.1`) |
| **CI platforms** | **Linux**, **macOS**, and **Windows** (GitHub Actions matrix: `ubuntu-latest`, `macos-latest`, `windows-latest`) |

CI installs the same toolchain on each OS. Keep `Cargo.toml` `rust-version`, `rust-toolchain.toml`, and the version string in `.github/workflows/ci.yml` identical (the CI job fails if they drift).

## Installation

```toml
[dependencies]
neuromod = "0.5.2"
```

Links: [crates.io](https://crates.io/crates/neuromod) · [docs.rs](https://docs.rs/neuromod) · [repository](https://github.com/Limen-Neural/neuromod)

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

- Engine: `SpikingNetwork`, `StepError` (LIF + Izhikevich banks)
- Neuromodulation: `NeuroModulators`, `SignalProfile`, `Observation`, `GenericReward`, `UnitReward`, `apply_neuromodulation`
- Engine neuron types: `LifNeuron`, `IzhikevichNeuron`
- Standalone neuron types: `GifNeuron`, `LapicqueNeuron`, `FitzHughNagumoNeuron`, `HodgkinHuxleyNeuron`
- Learning/plasticity:
  - Classical: `apply_classical_stdp`, `StdpParams`, `HebbianIzhikevichNetwork`
  - Reward-modulated building blocks: `EligibilityTrace`, `RmStdpConfig`

## Architecture & Boundaries

`neuromod` is the core library layer for neuron dynamics, generic neuromodulation, and foundational plasticity primitives.

See the full planning documents:

- [Org Modularization Standards](docs/org-modularization.md) — workstream index (#35–#43), cross-cutting git/build/beads standards, and audit commands.
- [neuromod Boundary Matrix](docs/neuromod-boundary-matrix.md) — runtime/deployment role, owns/does-not-own, allowed/forbidden dependencies vs. limbic-critic, brainstem-daemon, axon-encoder, synaptic-mesh, silicon-bridge, Spikenaut-Hardware, plasticity-lab, etc. (LIM-9).
- [ADR 001: Shared traits live in neuromod](docs/adr/001-traits-in-neuromod.md) — why traits are hosted here.

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

# Coverage (matches CI; see codecov.yml)
cargo install cargo-llvm-cov
cargo llvm-cov --all-features --lcov --output-path lcov.info
# HTML report: cargo llvm-cov --all-features --html

# Full CI-like validation
cargo install cargo-hack --locked
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo hack check --feature-powerset --exclude-no-default-features --keep-going
```

## Observability

`neuromod` publishes test coverage to Codecov. Error monitoring belongs in **application** binaries (depend on the `sentry` crate there), not in this library.

### Codecov

[![codecov](https://codecov.io/gh/Limen-Neural/neuromod/branch/main/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/neuromod)

- Configuration: [`codecov.yml`](codecov.yml)
- Workflow: [`.github/workflows/coverage.yml`](.github/workflows/coverage.yml)

Local coverage (also listed under [Development](#development)):

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --all-features --lcov --output-path lcov.info
# HTML report: cargo llvm-cov --all-features --html
```

- View the dashboard at [Codecov](https://codecov.io/gh/Limen-Neural/neuromod).
- Open `target/llvm-cov/html/index.html` after running the HTML report locally.
- CI runs the `coverage.yml` workflow on every PR and push to `main`.


## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0])
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT])

at your option.

## CI & Automation

This repository uses a comprehensive CI setup for speed, quality, security, and observability:

- **Core CI** (`.github/workflows/ci.yml`): matrix over **Linux / macOS / Windows** (`ubuntu-latest`, `macos-latest`, `windows-latest`). On every OS: MSRV toolchain, `clippy`, and build. When `dorny/paths-filter` detects rust-relevant path changes (`src/`, `tests/`, `examples/`, `benches/`, `Cargo.toml` / `Cargo.lock`): tests via `cargo-nextest` on every OS, and feature-matrix testing (`cargo-hack`) on Linux only. Always on Linux: `fmt` and domain-agnostic docs check. Uses `Swatinem/rust-cache` for faster feedback.
- **Qodana** (`.github/workflows/qodana_code_quality.yml`): JetBrains code-quality scans on every PR/push to `main` and `releases/*`; results are published to Qodana Cloud.
- **Codecov** (`.github/workflows/coverage.yml`): `cargo-llvm-cov` + Test Analytics (stable JUnit via pinned nextest). See [Observability](#observability) for local usage and report links.
- **reviewdog** (`.github/workflows/reviewdog.yml`): Inline PR comments for clippy and rustfmt.
- **Security scanning**:
  - CodeQL (`.github/workflows/codeql.yml`)
  - `rustsec/audit-check` + Trivy (`.github/workflows/audit.yml`)
- **Dependencies**: Dependabot (`.github/dependabot.yml`) for Cargo, GitHub Actions, Docker.
- **Docker** (`.github/workflows/docker.yml`, `Dockerfile`): Reproducible builds.
  Local usage:
  ```bash
  # Runtime image (example binaries only — no cargo toolchain)
  docker build -t neuromod:runtime .
  docker run --rm neuromod:runtime ls /usr/local/bin

  # Run tests inside the builder stage (has Rust + source)
  docker build --target builder -t neuromod:builder .
  docker run --rm neuromod:builder cargo test --all-features --quiet
  ```

## Links

- Crates.io: https://crates.io/crates/neuromod
- Docs.rs: https://docs.rs/neuromod
- Repository: https://github.com/Limen-Neural/neuromod
