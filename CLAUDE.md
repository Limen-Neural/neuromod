# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Identity

You are a Rust contributor to `neuromod`, a foundational spiking neural network (SNN) neuron-dynamics crate in the Limen-Neural ecosystem. Prefer concrete commands and file references over speculation.

For the full agent brief (repo map, boundaries, PR conventions), read [AGENTS.md](AGENTS.md) before structural changes. This file focuses on commands and architecture.

## Project

`neuromod` is a Rust library crate (edition 2024). It implements biologically grounded SNN primitives: neuron models, a topology-neutral `SpikingNetwork` engine, generic `NeuroModulators`, and reward-modulated plasticity building blocks. It is the core library layer of the Limen-Neural ecosystem.

Ownership rules: [docs/neuromod-boundary-matrix.md](docs/neuromod-boundary-matrix.md) and [docs/org-modularization.md](docs/org-modularization.md).

**Off-limits in this crate:** no async, networking, or hardware-specific code. No mining, trading, high-frequency trading (HFT), or crypto domain logic. Downstream crates (`axon-encoder`, `synaptic-mesh`, `limbic-critic`, `plasticity-lab`, `corpus-ipc`, `brainstem-daemon`, `silicon-bridge`, `Spikenaut-Hardware`) own everything outside neuron dynamics, neuromodulation, and plasticity.

## Commands

```bash
# Build
cargo build
cargo build --all-features

# Test (unit + doctests; crate-level example in src/lib.rs runs as a doctest)
cargo test
cargo test --all-features

# Single test
cargo test <test_name>
cargo test --package neuromod <module>::tests::<test_name>

# Lint / format (CI fails on any warning)
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings

# Feature-powerset matrix (as run in CI)
cargo hack check --feature-powerset --exclude-no-default-features --keep-going

# Coverage (matches codecov.yml)
cargo llvm-cov --all-features --lcov --output-path lcov.info

# Benchmarks (Criterion; compile-only smoke)
cargo bench --no-run --all-features

# Examples
cargo run --example basic
cargo run --example basic_lif
cargo run --example hebbian_learning
cargo run --example rstdp_demo
```

Before pushing changes that touch `src/`, `benches/`, `examples/`, `tests/`, or `Cargo.toml`, run the full gate in [REVIEW.md](REVIEW.md). That gate covers fmt, clippy, build, test, examples smoke, docs domain-hygiene grep, and public-API regression greps.

The toolchain is pinned in [rust-toolchain.toml](rust-toolchain.toml) (1.97.1). A matching `.devcontainer/` is available (`devcontainer up --workspace-folder .`).

## Architecture

### Two neuron banks driven by one engine

`SpikingNetwork` (`src/engine.rs`) is the central struct.

It owns two parallel neuron banks: leaky integrate-and-fire (LIF) neurons (`neurons: Vec<LifNeuron>`) and Izhikevich neurons (`iz_neurons: Vec<IzhikevichNeuron>`).

It also holds a `NeuroModulators` snapshot, a `global_step` counter, and per-channel spike-timing-dependent plasticity (STDP) / prediction state (`input_spike_times`, `predictive_state`).

Construction is topology-neutral. `new()` is the legacy default (16 LIF, 5 Izhikevich, 16 channels). `with_dimensions(num_lif, num_izh, num_channels)` builds arbitrary sizes with blank synaptic weights. No domain topology is hardcoded.

`SpikingNetwork::step(stimuli, modulators)` is the normal per-tick entry point for the default engine. Prefer it for full-network simulation; call lower-level neuron APIs only when testing or embedding a single model. Order of work inside `step`:

1. Validate `stimuli.len() == num_channels`, else `Err(StepError::InputLenMismatch)`.
2. Recompute per-neuron `decay_rate`/`threshold` targets from the current `NeuroModulators` (dopamine/serotonin/acetylcholine/norepinephrine each pull thresholds/decay in different directions — see formulas in `engine.rs`).
3. Update `predictive_state` (exponential moving average (EMA) per channel) and derive `pred_errors` ("surprise") that boost synaptic drive.
4. Stochastically encode `stimuli` into `input_spike_times` (Poisson-style, probability proportional to stimulus magnitude).
5. Integrate LIF membrane potentials, fire (`check_fire`), apply lateral inhibition to non-firing LIF neurons.
6. Apply reward-modulated STDP (`apply_stdp`, gated by `dopamine`-derived `learning_rate`; skipped entirely if learning rate ≈ 0).
7. Re-normalize each neuron's weights to `WEIGHT_BUDGET` (L1 budget) and clamp to `RM_STDP_W_MIN..RM_STDP_W_MAX`.
8. Drive the Izhikevich bank from the mean LIF membrane potential + dopamine (`iz_drive`), independent of the LIF spike/STDP pipeline.

Returns the indices of LIF neurons that fired this step.

### Two separate STDP implementations — do not conflate them

- **Classical/unmodulated Hebbian STDP** — `src/hebbian/classical.rs` (`apply_classical_stdp`, `StdpParams`, `HebbianIzhikevichNetwork`). Pure Hebb's rule, no reward gating; the "biological root."
- **Reward-modulated STDP (R-STDP)** — constants and `EligibilityTrace`/`RmStdpConfig` types live in `src/rm_stdp.rs`. The live per-step learning rule is inlined in `SpikingNetwork::apply_stdp` (`src/engine.rs`), gated by dopamine.
- The R-STDP engine path was reconstructed after going missing; `EligibilityTrace` is not yet wired into `apply_stdp`. Weight updates currently happen directly rather than via eligibility-trace-then-reward-conversion. Do not assume eligibility traces are live.

### Neuromodulators are domain-agnostic by design

`NeuroModulators` (`src/modulators.rs`) is a four-field struct (dopamine/serotonin/acetylcholine/norepinephrine) with exponential `decay()` and `add_*`/`boost_*`/`is_*` helpers.

Domain signals (thermal, power, throughput, timing) map into modulator levels via `SignalProfile` and `NeuroModulators::from_signals(...)`. `SignalProfile::default()` is unitless/neutral.

Unit convention (full details in [docs/signal-units.md](docs/signal-units.md)):

- Modulator levels are dimensionless. `0.0..=1.0` is the intended range, not an enforced invariant.
- Input channels carry no unit.
- Each `SignalProfile` field uses the same unit as the channel it scales.

`SignalProfile::hardware_calibrated()` is **deprecated since 0.6.0**. It still compiles and returns unchanged values. Deployment calibration belongs downstream. To migrate, copy the literal documented on the method.

Domain-specific reward shaping is downstream via the `GenericReward` trait. `UnitReward` is the only shipped implementation, intended for tests and simple consumer pipelines. `apply_neuromodulation` applies a `NeuroModulators` snapshot to weight/threshold slices without needing `SpikingNetwork`.

### Neuron models are standalone structs, not a shared trait

Each neuron model (`LifNeuron`, `GifNeuron`, `IzhikevichNeuron`, `LapicqueNeuron`, `FitzHughNagumoNeuron`, `HodgkinHuxleyNeuron`) is its own `Serialize`/`Deserialize` struct under `src/`, with its own `integrate`/`step`/`check_fire`-style API. There is no shared `Neuron` trait (see [docs/adr/001-traits-in-neuromod.md](docs/adr/001-traits-in-neuromod.md)).

Only `LifNeuron` and `IzhikevichNeuron` are wired into `SpikingNetwork`. The others are standalone building blocks for downstream consumers and examples.

### Serialization

`SpikingNetwork` and its neuron banks derive `Serialize`/`Deserialize` (serde) for checkpointing. Fields added after the initial release use `#[serde(default)]` (for example `LifNeuron::weights`, `base_threshold`, `last_spike_time`) so older serialized states still load.

### Observability

This crate does not ship a `sentry` feature. Downstream binaries that want Sentry should depend on the `sentry` crate directly and initialize it in the application.
