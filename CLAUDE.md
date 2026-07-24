# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`neuromod` is a Rust library crate (edition 2024) implementing biologically grounded spiking neural network (SNN) primitives: neuron models, a topology-neutral `SpikingNetwork` engine, generic `NeuroModulators`, and reward-modulated plasticity building blocks. It is the core library layer of the Limen-Neural ecosystem — see `AGENTS.md` for the full agent brief (identity, repo map, boundaries, PR conventions) and `docs/neuromod-boundary-matrix.md` / `docs/org-modularization.md` for cross-repo ownership rules. Read `AGENTS.md` before making structural changes; this file focuses on commands and architecture.

**Off-limits in this crate:** no async, networking, or hardware-specific code; no mining/trading/HFT/crypto domain logic. Downstream crates (`axon-encoder`, `synaptic-mesh`, `limbic-critic`, `plasticity-lab`, `corpus-ipc`, `brainstem-daemon`, `silicon-bridge`, `Spikenaut-Hardware`) own everything outside neuron dynamics/neuromodulation/plasticity.

## Commands

```bash
# Build
cargo build                    # default (no optional features)
cargo build --all-features     # includes `sentry`

# Test (unit + doctests; crate-level example in src/lib.rs runs as a doctest)
cargo test
cargo test --all-features      # also runs tests/sentry_integration.rs (16 tests)

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
SENTRY_DSN=https://...@... cargo run --example sentry --features sentry
```

Before pushing changes touching `src/`, `benches/`, `examples/`, `tests/`, or `Cargo.toml`, run the full gate in `REVIEW.md` (fmt, clippy, build, test, examples smoke, docs domain-hygiene grep, regression grep for the public API surface).

The toolchain is pinned in `rust-toolchain.toml` (1.97.1); a matching `.devcontainer/` is available (`devcontainer up --workspace-folder .`).

## Architecture

### Two neuron banks driven by one engine

`SpikingNetwork` (`src/engine.rs`) is the central struct. It owns two parallel neuron banks — `neurons: Vec<LifNeuron>` and `iz_neurons: Vec<IzhikevichNeuron>` — plus a `NeuroModulators` snapshot, a `global_step` counter, and per-channel STDP/prediction state (`input_spike_times`, `predictive_state`). It is constructed topology-neutral: `new()` gives the legacy default (16 LIF, 5 Izhikevich, 16 channels), `with_dimensions(num_lif, num_izh, num_channels)` builds arbitrary sizes with blank synaptic weights — no domain topology is hardcoded.

`SpikingNetwork::step(stimuli, modulators)` is the single per-tick entry point and always runs, in order:

1. Validate `stimuli.len() == num_channels`, else `Err(StepError::InputLenMismatch)`.
2. Recompute per-neuron `decay_rate`/`threshold` targets from the current `NeuroModulators` (dopamine/serotonin/acetylcholine/norepinephrine each pull thresholds/decay in different directions — see the formulas inline in `engine.rs`).
3. Update `predictive_state` (EMA per channel) and derive `pred_errors` ("surprise") that boost synaptic drive.
4. Stochastically encode `stimuli` into `input_spike_times` (Poisson-style, probability ∝ stimulus magnitude).
5. Integrate LIF membrane potentials, fire (`check_fire`), apply lateral inhibition to non-firing LIF neurons.
6. Apply reward-modulated STDP (`apply_stdp`, gated by `dopamine`-derived `learning_rate`; skipped entirely if learning rate ≈ 0).
7. Re-normalize each neuron's weights to `WEIGHT_BUDGET` (L1 budget) and clamp to `RM_STDP_W_MIN..RM_STDP_W_MAX`.
8. Drive the Izhikevich bank from the mean LIF membrane potential + dopamine (`iz_drive`), independent of the LIF spike/STDP pipeline.

Returns the indices of LIF neurons that fired this step.

### Two separate STDP implementations — don't conflate them

- **Classical/unmodulated Hebbian STDP** — `src/hebbian/classical.rs` (`apply_classical_stdp`, `StdpParams`, `HebbianIzhikevichNetwork`). Pure Hebb's rule, no reward gating; the "biological root."
- **Reward-modulated STDP (R-STDP)** — constants and `EligibilityTrace`/`RmStdpConfig` types live in `src/rm_stdp.rs`, but the actual per-step learning rule that consumes them is inlined in `SpikingNetwork::apply_stdp` (`src/engine.rs`), gated by dopamine. Per `rm_stdp.rs`'s own comments, this reward-modulated path was reconstructed after being found missing from the original codebase — `EligibilityTrace` exists but is not yet wired into `apply_stdp`; weight updates currently happen directly rather than via eligibility-trace-then-reward-conversion. Be aware of this gap before assuming eligibility traces are live.

### Neuromodulators are domain-agnostic by design

`NeuroModulators` (`src/modulators.rs`) is a plain 4-tuple (dopamine/serotonin/acetylcholine/norepinephrine) with its own exponential `decay()` and `add_*`/`boost_*`/`is_*` helpers. Domain signals (thermal, power, throughput, timing) are mapped into modulator levels via `SignalProfile` + `NeuroModulators::from_signals(...)` — `SignalProfile::default()` is unitless/neutral, `SignalProfile::hardware_calibrated()` is a legacy pre-0.5 profile kept for migration. Reward shaping for a specific domain is implemented downstream via the `GenericReward` trait (`UnitReward` is the only in-crate impl, used for tests). `apply_neuromodulation` applies a `NeuroModulators` snapshot to arbitrary weight/threshold slices independent of `SpikingNetwork`.

### Neuron models are standalone structs, not a shared trait

Each neuron model (`LifNeuron`, `GifNeuron`, `IzhikevichNeuron`, `LapicqueNeuron`, `FitzHughNagumoNeuron`, `HodgkinHuxleyNeuron`) is its own `Serialize`/`Deserialize` struct in its own file under `src/`, with its own `integrate`/`step`/`check_fire`-style API — there is no shared `Neuron` trait unifying them (see `docs/adr/001-traits-in-neuromod.md` for why shared traits are hosted in this crate at all). Only `LifNeuron` and `IzhikevichNeuron` are wired into `SpikingNetwork`; the others are standalone building blocks for downstream consumers/examples.

### Serialization

`SpikingNetwork` and its neuron banks derive `Serialize`/`Deserialize` (serde) for checkpointing; fields added after the initial release use `#[serde(default)]` (e.g. `LifNeuron::weights`, `base_threshold`, `last_spike_time`) to stay backward-compatible with older serialized states.

### Optional `sentry` feature

Off by default (`features = []`). When enabled, adds error/panic reporting (`sentry` crate, rustls transport) — see `examples/sentry.rs` and `tests/sentry_integration.rs`. Requires `pkg-config`/`libssl-dev` system deps only when building with this feature.
