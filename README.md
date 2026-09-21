# neuromod

[![CI](https://github.com/Limen-Neural/neuromod/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/Limen-Neural/neuromod/actions/workflows/ci.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Limen-Neural/neuromod/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/neuromod)
[![Code Quality: Codacy](https://img.shields.io/badge/code%20quality-Codacy-222f29?logo=codacy)](https://app.codacy.com/gh/Limen-Neural/neuromod/dashboard)
[![Maintainability: Qlty](https://qlty.sh/gh/Limen-Neural/projects/neuromod/maintainability.svg)](https://qlty.sh/gh/Limen-Neural/projects/neuromod)
[![crates.io](https://img.shields.io/crates/v/neuromod.svg?label=crates.io)](https://crates.io/crates/neuromod)
[![docs.rs](https://docs.rs/neuromod/badge.svg)](https://docs.rs/neuromod)
[![License](https://img.shields.io/crates/l/neuromod.svg)](https://github.com/Limen-Neural/neuromod#license)

[Quick Start](#quick-start) · [Examples](#examples) · [Migration Notes](#migration-notes) · [Changelog](CHANGELOG.md) · [Validation](#maintainer-release-sequence)

Biologically grounded spiking neural network (SNN) primitives in Rust: a topology-neutral `SpikingNetwork` engine, generic neuromodulators, STDP building blocks, and standalone neuron models.

`neuromod` is a reusable core library: topology-neutral at initialization, dynamically sizable at runtime, and strict about input shape and finiteness validation. Dual-licensed MIT OR Apache-2.0.

## Highlights

- Dynamic network sizing with `SpikingNetwork::with_dimensions(...)`
- Backward-compatible default constructor: `SpikingNetwork::new()`
- Strict step contract: `Result<Vec<usize>, StepError>`
- Neutral initialization (blank synaptic weights; no hardcoded domain topology)
- Generic neuromodulators: dopamine, serotonin, acetylcholine, norepinephrine
- `GenericReward` trait for domain-specific reward shaping in downstream crates
- Reward-modulated STDP wired into the engine: per-synapse `EligibilityTrace` accumulation with a dopamine-gated payout, tuned by `RmStdpConfig`
- Classical (unmodulated) Hebbian STDP utilities for the biological root case
- Caller-injected RNG on the live stochastic paths (`SpikingNetwork::step_with_rng`, `PoissonEncoder::encode_with_rng`) so a seeded stream can replay a run

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

### Sparse GIF hidden layer

`SparseGifHiddenLayer` is a structure-of-arrays bank of GIF neurons with deterministic sparse fan-in and batched execution — a layer abstraction rather than a single neuron, and also **not** an engine bank.

- **SoA state:** membrane, adaptation, and last-spike-time live in parallel `Vec`s indexed by neuron, never a `Vec<GifNeuron>`.
- **CSR topology:** the layer owns its `(offsets, sources, weights)` fan-in.
- **Deterministic:** topology and initial weights come from a seeded SplitMix64 stream with a per-neuron sub-stream, so the same `SparseGifLayerConfig` — seed, shape, `weight_range`, and `params` alike — always yields the same layer; `run()` is a sequential fold with no threading, so the same input always yields the same raster.
- **Shared dynamics:** both the layer and `GifNeuron` execute the equations on `GifParams`, so the two representations agree bit for bit (pinned by a test).

```rust
use neuromod::{SparseGifHiddenLayer, SparseGifLayerConfig};

let mut layer = SparseGifHiddenLayer::new(&SparseGifLayerConfig {
    num_inputs: 32,
    num_neurons: 8,
    fan_in: 6,
    seed: 0xC0FF_EE01,
    ..Default::default()
})
.unwrap();

let train: Vec<Vec<f32>> = (0..24).map(|t| vec![(t % 3) as f32 * 0.4; 32]).collect();
let raster = layer.run(&train).unwrap();
println!("{:?}", raster.per_neuron_counts());
```

The layer takes no `NeuroModulators`: a GIF hidden layer is a pure integrate-and-fire structure with no reward signal. Callers that want modulation apply it themselves between steps — `weights_mut()` for synaptic strength, `params_mut()` for the shared dynamics (`base_threshold` and friends). `apply_neuromodulation` is **not** usable here: it expects a per-neuron `&mut [f32]` threshold slice, whereas this layer holds one shared `GifParams` for the whole bank rather than a threshold per neuron.

This module is an upstream port of the equivalent layer from the author's `corinth-canal` repository (issue #101). Its regression fixtures are internal goldens produced by this implementation, not independent cross-repository bit-parity vectors. Committed cross-repository parity fixtures remain a follow-up in the existing v0.7 issue #144. See `cargo run --example sparse_gif_layer`.

## Requirements

| | |
|--|--|
| **MSRV** | **Rust 1.98.1** (`rust-version` in `Cargo.toml`) |
| **Edition** | 2024 |
| **Pin** | [`rust-toolchain.toml`](rust-toolchain.toml) (channel `1.98.1`) |
| **CI platforms** | **Linux**, **macOS**, and **Windows** (GitHub Actions matrix: `ubuntu-latest`, `macos-latest`, `windows-latest`) |

CI installs the same toolchain on each OS. Keep `Cargo.toml` `rust-version`, `rust-toolchain.toml`, and the version string in `.github/workflows/ci.yml` identical (the CI job fails if they drift).

## Installation

```toml
[dependencies]
neuromod = "0.6.0"
```

Browser, Web Worker, and other supported JavaScript-hosted
`wasm32-unknown-unknown` consumers opt into the upstream getrandom backend with
`neuromod = { version = "0.6.0", features = ["wasm-js"] }`. Non-Web WASM
consumers should leave this feature disabled and choose the entropy backend for
their final application. See [the RNG guide](https://github.com/Limen-Neural/neuromod/blob/main/docs/rng.md#webassembly-entropy-backends).

> This README describes the in-repository `0.6.0` candidate. Check
> [crates.io](https://crates.io/crates/neuromod) for registry availability; before maintainer
> publication, the in-repository examples use the local source.

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

### Reproducible steps

The only live stochastic work inside `step` is Bernoulli encoding of
`input_spike_times`. Keep using `step` when you do not care about the stream.
For replay, inject one caller RNG and reuse it for the whole run:

```rust
use neuromod::{NeuroModulators, SeedableRng, SpikingNetwork, StdRng};

fn main() {
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5_f32; 16];
    let modulators = NeuroModulators::default();
    let mut rng = StdRng::seed_from_u64(0xC0FF_EE01);

    let spikes = network
        .step_with_rng(&stimuli, &modulators, &mut rng)
        .unwrap();
    println!("Spiking neuron indices: {spikes:?}");
}
```

The generator is re-exported from this crate (`StdRng`, `SeedableRng`), so the
example above does not need a direct `rand` dependency. It is not stored on
`SpikingNetwork` and is not part of a serde checkpoint. With the same `StdRng`
implementation (this crate's `rand` version and target), the same seed + same
inputs/state replays a run **from the start**. Resuming a mid-run checkpoint
needs the generator's advanced state, not only the original seed; see
[docs/rng.md](https://github.com/Limen-Neural/neuromod/blob/main/docs/rng.md).

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

## Step Errors

`step` validates the call **before** mutating the network or drawing from the random-number generator (RNG). A length mismatch, a non-finite input, or an exhausted tick counter returns a structured [`StepError`](https://docs.rs/neuromod/latest/neuromod/enum.StepError.html) and leaves every field unchanged (failure-atomic no-op). Finite signed values still go through the existing `abs().clamp` magnitude path.

`global_step` is a discrete tick counter in **steps** (not wall-clock time), range `0..=i64::MAX`. Spike timestamps (`LifNeuron::last_spike_time`, `input_spike_times`) use the same unit; `-1` is the sentinel for no recorded spike. A restored checkpoint sitting at `i64::MAX` (or with a negative counter) still deserializes — `step` then returns `StepCounterExhausted` instead of panicking in debug, wrapping in release, or stamping the `-1` sentinel. Call `reset()` to start a new epoch; the engine will not renumber a live network for you.

```rust
use neuromod::{NeuroModulators, NonFiniteClass, SpikingNetwork, StepError};

fn main() {
    let mut network = SpikingNetwork::with_dimensions(32, 4, 32);
    let modulators = NeuroModulators::default();
    let bad_stimuli = vec![0.1_f32; 31];

    match network.step(&bad_stimuli, &modulators) {
        Ok(_) => unreachable!("expected a structured error"),
        Err(StepError::InputLenMismatch { expected, got }) => {
            println!("InputLenMismatch: expected {expected}, got {got}");
        }
        Err(StepError::NonFiniteStimulus { index, class }) => {
            println!("NonFiniteStimulus at {index}: {class:?}");
        }
        Err(StepError::NonFiniteModulator { field, class }) => {
            println!("NonFiniteModulator {field:?}: {class:?}");
        }
        Err(StepError::StepCounterExhausted { global_step }) => {
            println!("step counter cannot advance from {global_step}");
        }
    }

    let mut nonfinite = vec![0.1_f32; 32];
    nonfinite[31] = f32::NAN;
    assert!(matches!(
        network.step(&nonfinite, &modulators),
        Err(StepError::NonFiniteStimulus {
            index: 31,
            class: NonFiniteClass::Nan
        })
    ));

    network.global_step = i64::MAX;
    assert!(matches!(
        network.step(&[0.0; 32], &modulators),
        Err(StepError::StepCounterExhausted {
            global_step: i64::MAX
        })
    ));
    assert_eq!(network.global_step, i64::MAX);
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

### Signal units

`neuromod` is unit-agnostic on the input side and dimensionless on the output side:

- Every `NeuroModulators` level is a dimensionless value; `0.0..=1.0` is the intended range, kept by `from_signals` and `decay()` for finite inputs (negative `add_*` amounts and `NaN` signals are the documented exceptions).
- The four `from_signals` channels (thermal, power, throughput, timing) carry no unit of their own.
- Each `SignalProfile` field is expressed in the same unit as the channel it scales, so the caller declares its units exactly once, in the profile.

`SignalProfile::default()` is the neutral profile for signals already normalized to `0.0..=1.0`. For physical units, construct the struct directly — all fields are public. See [docs/signal-units.md](https://github.com/Limen-Neural/neuromod/blob/main/docs/signal-units.md) for the channel table, the exact mapping formulas, and the serotonin caveat.

### Migrating off `hardware_calibrated()`

`SignalProfile::hardware_calibrated()` is **deprecated since 0.6.0** and still returns the same values; nothing is removed or renamed. Deployment calibration belongs to the consuming crate, so copy the literal into your own code:

```rust
use neuromod::SignalProfile;

let profile = SignalProfile {
    throughput_scale: 0.0105,
    thermal_threshold: 83.0,
    power_baseline: 400.0,
    power_scale: 50.0,
    timing_scale: 2640.0,
    stability_target: 1.05,
};
```

## Reward-Modulated STDP

> **New in 0.6.0.** `EligibilityTrace`, `RmStdpConfig`, `SpikingNetwork::set_rm_stdp_config`,
> and `LifNeuron::eligibility` do not exist in `0.5.2`, so the code below will not compile
> against the last published release — see [Installation](#installation) and
> [Migration Notes](#migration-notes).

`SpikingNetwork` learns *through* eligibility traces, not around them. Each `LifNeuron`
carries one `EligibilityTrace` per input channel, indexed like `weights`:

1. Every step, each trace decays and — on the step a spike actually occurs — accumulates the
   pre/post timing kernel. This happens **whether or not dopamine is present**.
2. Dopamine gates only the payout: `w += reward_lr × dopamine_lr × trace`, clamped to the
   `RmStdpConfig` bounds.

Splitting it that way is what buys credit assignment: reward can arrive several steps after
the coincidence it pays for, and still find the credit waiting.

```rust
use neuromod::{NeuroModulators, RmStdpConfig, SpikingNetwork};

fn main() {
    let mut network = SpikingNetwork::with_dimensions(4, 1, 4);
    for neuron in &mut network.neurons {
        neuron.weights = vec![0.5; 4]; // sums to the engine's L1 weight budget
    }

    // Bank coincidences with reward switched off: traces grow, weights do not.
    let unrewarded = NeuroModulators::default();
    let stimuli = [1.0, 1.0, 0.0, 0.0];
    for _ in 0..10 {
        network.step(&stimuli, &unrewarded).unwrap();
    }
    println!("trace: {:.4}", network.neurons[0].eligibility[0].value); // > 0
    println!("weight: {:.4}", network.neurons[0].weights[0]); // still 0.5

    // Reward converts the banked trace into a weight change.
    let rewarded = NeuroModulators { dopamine: 0.9, ..Default::default() };
    for _ in 0..10 {
        network.step(&stimuli, &rewarded).unwrap();
    }
    println!("weight: {:.4}", network.neurons[0].weights[0]); // driven synapse potentiated

    // Retune decay, payout rate, and weight bounds at any time.
    network.set_rm_stdp_config(RmStdpConfig { tau_eligibility: 100.0, ..Default::default() });
}
```

Proof, not promise: the behavior above is covered by unit and multi-step tests in
`src/rm_stdp.rs` and `src/engine.rs` (including a pre-0.6 checkpoint that deserializes
without the trace fields and keeps stepping), and `cargo run --example rstdp_demo` prints
the real trace and weight numbers. Rationale for wiring the types in rather than demoting
them: [ADR 002](https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/002-wire-eligibility-traces.md).

## Migration Notes

### 0.6.0 — missing neuron checkpoint fields use constructor sentinels

Self-describing checkpoints that omit `LifNeuron::last_spike_time` or
`GifNeuron::last_spike_time` now restore `-1` (never fired), while omitted
`base_threshold` fields restore the model's resting threshold (`0.02` for LIF and
`0.65` for GIF). Explicit serialized values remain unchanged. For compatibility
with checkpoints written before the field existed, an omitted
`IzhikevichNeuron::last_spike_time` also restores `-1` rather than failing to
deserialize.

### 0.6.0 — `StepError` names non-finite ingress

`SpikingNetwork::step` now rejects `NaN` / `±∞` stimuli and modulator fields
before mutating anything. Exhaustive `match`es on `StepError` must handle the
new `StepError::NonFiniteStimulus` and `StepError::NonFiniteModulator` arms
(or a `_` wildcard). Length mismatch is still checked first. Finite signed
inputs are unchanged.

### 0.6.0 — eligibility traces wired into the engine

**Serialized state survives in self-describing formats.** `LifNeuron::eligibility` and
`SpikingNetwork::stdp_config` are `#[serde(default)]`, and `apply_stdp` resizes a missing
trace vector, so a 0.5.x checkpoint written with a format that names its fields — JSON,
YAML, TOML, RON, map-encoded MessagePack — still deserializes and steps. This is covered by
`test_pre_0_6_state_without_new_fields_loads_and_steps`, which strips both fields from
serialized JSON and drives the restored network.

`#[serde(default)]` cannot help positional binary formats such as `bincode` or `postcard`:
they encode a struct as a bare sequence of fields, so old bytes hit end-of-input before the
new fields are reached. If you checkpoint with one of those, re-serialize from 0.5.x before
upgrading, or read through a versioned wrapper of your own.

**Struct literals need updating.** Both types have public fields and are not
`#[non_exhaustive]`, so adding a field is a source-level break: any downstream
`LifNeuron { .. }` or `SpikingNetwork { .. }` literal that spells out every field now fails
to compile. Fill the remainder from the constructor or `Default`:

```rust
use neuromod::LifNeuron;

// Before (0.5.x) — breaks in 0.6
// let neuron = LifNeuron {
//     membrane_potential: 0.0,
//     decay_rate: 0.15,
//     threshold: 0.02,
//     base_threshold: 0.02,
//     last_spike: false,
//     weights: vec![0.0; 16],
//     last_spike_time: -1,
// };

// After — forward-compatible with future field additions
let neuron = LifNeuron {
    weights: vec![0.0; 16],
    ..LifNeuron::new()
};
```

Callers that already build through `LifNeuron::new()`, `SpikingNetwork::new()`, or
`SpikingNetwork::with_dimensions(..)` need no change.

**Weight trajectories change.** Updates now flow through a decaying eligibility trace
instead of being recomputed from raw spike times each step, so a 0.6 run will not reproduce
0.5.x weights on the same inputs. Learning gained memory: reward arriving after a
coincidence still pays for it.

**Weight bounds moved into `RmStdpConfig`.** `RM_STDP_W_MIN` / `RM_STDP_W_MAX` remain public
and are the defaults. Bounds take precedence over the engine's L1 weight budget: `step`
scales toward the budget and then clamps, so a binding bound leaves the sum **off** budget in
whichever direction it binds — a lowered `w_max` caps weights and leaves the sum short, while
a raised `w_min` lifts weights after scaling and can push the sum past it. The defaults cannot
bind, so the budget holds exactly under them.

### 0.6.0 — `StepError::StepCounterExhausted`

**Exhaustive matches on `StepError` need a new arm.** `SpikingNetwork::step` now returns
`StepError::StepCounterExhausted { global_step }` when incrementing `global_step` would
overflow `i64::MAX`. Debug and release share this behavior. A rejected tick is atomic (no
RNG draw, no neuromodulator snapshot, no membrane or trace updates).

**Checkpoints at `i64::MAX` still load.** Serde does not reject an exhausted or negative
counter; validation is on `step`, so a saturated network can be inspected. Call `reset()` to continue
from tick 0 — the engine will not renumber timestamps for you. Normal (non-exhausted)
checkpoints are unchanged.

```rust
use neuromod::{NeuroModulators, SpikingNetwork, StepError};

let mut net = SpikingNetwork::new();
net.global_step = i64::MAX;
assert!(matches!(
    net.step(&[0.0; 16], &NeuroModulators::default()),
    Err(StepError::StepCounterExhausted { global_step: i64::MAX })
));
```

### 0.6.0 — numerical integration and reset corrections

**Izhikevich trajectories change from 0.5.x and earlier 0.6.0 candidates.** Each documented
voltage substep is again a 0.5-ms Euler increment, so exact voltages and `step_with_time` spike
schedules can differ. The public API and serialized shape are unchanged.

**FitzHugh–Nagumo and Hodgkin–Huxley now consume the requested duration.** Finite positive
durations are integrated completely in bounded substeps, including short durations and a final
remainder. Zero, negative, `NaN`, and either infinity return `false` without mutating state.
Scientific trajectories can therefore differ where a previous call was rounded, truncated, or
ignored; no signature or serialized shape changed.

**FitzHugh–Nagumo reset selects a bounded, validated nullcline intersection.**
`FitzHughNagumoNeuron::reset` selects a zero-input solution of the original equations
`v - v^3 / 3 - w = 0` and `v + a - b * w = 0` through a root search that is bounded. An
approximate root is accepted only when its rounded `f32` state passes residual checks in both the
implemented arithmetic and the original equations; `b = 0` remains supported directly. Finite
coefficients therefore do not guarantee that the bounded search finds a representable, validated
state: when it finds no such pair, reset writes `NaN` to both `v` and `w`, which is not proof that
none exists. `epsilon` is intentionally outside this unscaled equilibrium geometry, so it cannot
hide a missed nullcline intersection. Different selected roots can change reset values and later
trajectories. Selection is not guaranteed unique, and the selected root is not necessarily stable;
this correction does not guarantee RK4 stability for arbitrary parameters, inputs, or timesteps.

**`FitzHughNagumoNeuron::is_excitable` now requires strict local linear stability.** The method
evaluates the Jacobian at its selected finite zero-input root in `f64` arithmetic and returns
`true` only when both coefficients are finite, `trace < 0`, and `determinant > 0`. A selected
saddle or neutral boundary, a failed/non-finite root search, or a non-finite Jacobian coefficient
therefore returns `false`. This is a local linear classification of the selected root: `false`
does not establish global oscillation, nonlinear instability, or any other global behavior. The
public signature, serialization, parameters, and root-selection behavior are unchanged, though
callers whose selected equilibrium is a saddle or zero-determinant boundary now receive the
corrected `false` result.

**Hodgkin–Huxley spike booleans now use an above-rest action-potential threshold.** `step`
reports a strict upward crossing from below to at or above relative `65 mV` for
`RelativeToRest`, or absolute `0 mV` for `Absolute`. A genuine first squid action potential now
reports `true`; small near-rest oscillations no longer do. This detector-only correction does
not alter continuous integration state or recalibrate model dynamics.

**`SpikingNetwork::reset` clears episode state without discarding configuration.** It resets
both neuron banks' dynamic state and spike history, including every LIF eligibility-trace value,
the engine clock, inputs, predictive state, and the neuromodulator snapshot. Current LIF
thresholds and decay rates, configured neuron
parameters, and learned LIF weights are preserved.

**Classical Hebbian updates require real spike history at both endpoints.**
`HebbianIzhikevichNetwork::update_weights` now leaves a weight unchanged until both timestamps
are nonnegative. `apply_classical_stdp` keeps its existing public timing kernel.

### 0.6.0 — Hodgkin–Huxley voltage conventions and checkpoints

`VoltageConvention::{RelativeToRest, Absolute}` is public and
`HodgkinHuxleyNeuron::voltage_convention` identifies the coordinate system shared by `v`,
`e_na`, `e_k`, and `e_l`. The constructors retain their numeric presets: the squid preset uses
rest `0`, reversals `(115, -12, 10.6)`, and `6.3 °C`; the cortical preset uses rest `-65`,
reversals `(50, -77, -54.387)`, and `37 °C`. Temperature now affects Q10 kinetics only; it does
not select a voltage coordinate.

**Struct literals and JSON change.** Add `voltage_convention` to exhaustive
`HodgkinHuxleyNeuron` literals, or build from a constructor. JSON serializes the field as
`"relative_to_rest"` or `"absolute"`. Legacy JSON that omits it is accepted only when the parsed
reversal tuple is exactly `(115, -12, 10.6)` or `(50, -77, -54.387)`; custom legacy records must
add the field. Explicit `null` or malformed conventions are rejected. Positional checkpoints
such as bincode/postcard need migration or re-encoding because they do not name fields.

**Changing the field does not convert values.** To move between conventions, transform `v` and
all three reversal potentials together: subtract 65 mV to move relative values to absolute
coordinates, or add 65 mV for the reverse. Deserialization preserves stored numeric values and
does not silently repair mixed-coordinate state.

## Included Components

- Engine: `SpikingNetwork`, `StepError`, `NonFiniteClass`, `ModulatorField` (LIF + Izhikevich banks)
- Neuromodulation: `NeuroModulators`, `SignalProfile`, `Observation`, `GenericReward`, `UnitReward`, `apply_neuromodulation`
- Engine neuron types: `LifNeuron`, `IzhikevichNeuron`
- Stochastic helpers: `lif::PoissonEncoder` (`encode` / `encode_with_rng`); re-exported `Rng`, `SeedableRng`, `StdRng` for caller-injected streams
- Standalone neuron types: `GifNeuron`, `GifParams`, `LapicqueNeuron`, `FitzHughNagumoNeuron`, `HodgkinHuxleyNeuron`
- Standalone layer: `SparseGifHiddenLayer`, `SparseGifLayerConfig`, `SpikeRaster`, `GifLayerError`
- Learning/plasticity:
  - Classical (unmodulated): `apply_classical_stdp`, `StdpParams`, `HebbianIzhikevichNetwork`
  - Reward-modulated, wired into `SpikingNetwork`: `EligibilityTrace`, `RmStdpConfig`,
    `LifNeuron::eligibility`, `SpikingNetwork::set_rm_stdp_config`

## Architecture & Boundaries

`neuromod` is the core library layer for neuron dynamics, generic neuromodulation, and foundational plasticity primitives.

See the full planning documents:

- [Org Modularization Standards](https://github.com/Limen-Neural/neuromod/blob/main/docs/org-modularization.md) — workstream index (#35–#43), cross-cutting git/build/beads standards, and audit commands.
- [neuromod Boundary Matrix](https://github.com/Limen-Neural/neuromod/blob/main/docs/neuromod-boundary-matrix.md) — runtime/deployment role, owns/does-not-own, allowed/forbidden dependencies vs. limbic-critic, brainstem-daemon, axon-encoder, synaptic-mesh, silicon-bridge, Spikenaut-Hardware, plasticity-lab, etc. (LIM-9).
- [ADR 001: Shared traits live in neuromod](https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/001-traits-in-neuromod.md) — why traits are hosted here.
- [ADR 002: Wire eligibility traces into the engine](https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/002-wire-eligibility-traces.md) — why R-STDP is wired rather than demoted, and what changed in the learning path.
- [RNG inventory](https://github.com/Limen-Neural/neuromod/blob/main/docs/rng.md) — live stochastic paths, caller-injected RNG variants, and deterministic surfaces.

## Examples

In-repo examples use the **local** crate (clone this repository):

```bash
cargo run --example basic
cargo run --example rstdp_demo
cargo run --example sparse_gif_layer
```

### Standalone crates.io demo

Outsiders who are not on the Limen git graph can depend only on crates.io. The runnable package is [`examples/crates-io-standalone`](https://github.com/Limen-Neural/neuromod/tree/main/examples/crates-io-standalone) — a detached Cargo workspace so it cannot pick up this path crate.

```bash
cd examples/crates-io-standalone
cargo run
```

Or start a binary anywhere with this `Cargo.toml` (no `git =`, no `path =`):

```toml
[package]
name = "neuromod-crates-io-demo"
version = "0.1.0"
edition = "2024"

[dependencies]
neuromod = "0.5"
```

`neuromod = "0.5"` intentionally resolves the published 0.5.x line. This registry-only demo
does not validate this repository's 0.6.0 APIs (wired R-STDP and `SparseGifHiddenLayer`); check
crates.io for later registry availability. Before maintainer publication, use the in-repository
examples above, which resolve the local source.

## Maintainer release sequence

1. After every correctness fix and documentation change is merged, set the intended release date
   in the 0.6.0 changelog heading and commit it. The date records release intent; it does not
   claim registry publication. Then start from that clean, exact final SHA and run the final-gate
   checklist, including `cargo package --locked`, `cargo publish --locked --dry-run`, the
   independent unpacked archive-consumer checks, archive checksum capture, and exact-SHA CI
   evidence. Before packaging, run this release-content guard:

   ```bash
   python3 - <<'PY'
   from datetime import date
   from pathlib import Path
   import re

   readme = Path("README.md").read_text()
   changelog = Path("CHANGELOG.md").read_text()

   def require(condition, message):
       if not condition:
           raise SystemExit(message)

   for name, text in (("README.md", readme), ("CHANGELOG.md", changelog)):
       require(
           not re.search(r"^(<<<<<<<|=======|\|\|\|\|\|\|\||>>>>>>>)", text, re.M),
           f"conflict marker in {name}",
       )

   headings = re.findall(r"^## \[0\.6\.0\].*$", changelog, re.M)
   require(len(headings) == 1, f"expected one 0.6.0 heading: {headings}")
   dated_heading = re.fullmatch(r"## \[0\.6\.0\] - (\d{4}-\d{2}-\d{2})", headings[0])
   require(dated_heading is not None, f"invalid 0.6.0 heading: {headings[0]}")
   try:
       release_date = date.fromisoformat(dated_heading.group(1))
   except ValueError as error:
       raise SystemExit(f"invalid 0.6.0 date: {error}") from error
   print(f"ok: release documents have no conflict markers and one dated 0.6.0 heading ({release_date})")
   PY
   ```

2. Obtain separate explicit authorization before creating a release tag or running
   `cargo publish --locked`. A tag alone does not publish the crate.
3. After the authorized tag and publication, verify the registry version and archive metadata
   directly before describing 0.6.0 as published.

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

[![codecov](https://codecov.io/gh/Limen-Neural/neuromod/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/neuromod)

- Configuration: [`codecov.yml`](https://github.com/Limen-Neural/neuromod/blob/main/codecov.yml)
- Workflow: [`.github/workflows/coverage.yml`](https://github.com/Limen-Neural/neuromod/blob/main/.github/workflows/coverage.yml)
- Dashboard: [codecov.io/gh/Limen-Neural/neuromod](https://codecov.io/gh/Limen-Neural/neuromod)

The badge links to Codecov's test coverage report.
**Uploads** need the repository secret **`CODECOV_TOKEN`** (tokenless uploads return
HTTP 400 for this org). The coverage workflow passes that token and sets
`fail_ci_if_error: false`, so a missing/stale token does **not** fail CI—only the
badge may stay `unknown` until the secret is correct. After a successful upload on
`main`, the badge shows a coverage %.

Local coverage (also listed under [Development](#development)):

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --all-features --lcov --output-path lcov.info
# HTML report: cargo llvm-cov --all-features --html
```

- Open `target/llvm-cov/html/index.html` after running the HTML report locally.
- CI runs the `coverage.yml` workflow on every PR and push to `main`.


## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0])
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT])

at your option.

## CI & Automation

This repository uses a comprehensive CI setup for speed, quality, security, and observability:

- **Core CI** (`.github/workflows/ci.yml`): runs on every pull request and push to `main`, and can be started with `workflow_dispatch`. The **Linux / macOS / Windows** matrix (`ubuntu-latest`, `macos-latest`, `windows-latest`) runs the pinned MSRV toolchain, `clippy`, build, and `cargo test --locked --all-features` (unit tests and doctests) unconditionally. Linux additionally runs the release test suite; overflow-checks-off, feature-powerset, and browser-WASM regressions; formatting; strict domain-agnostic rustdoc; dependency audit; debug and release example smokes; benchmark compilation and Criterion runtime smoke execution; package validation; and debug/release registry-only outsider-demo smokes. The same release-candidate path applies to source, workflow, toolchain, and documentation changes. Uses `Swatinem/rust-cache` for faster feedback.
- **Codecov** (`.github/workflows/coverage.yml`): `cargo-llvm-cov` + Test Analytics (stable JUnit via pinned nextest). See [Observability](#observability) for local usage and report links.
- **reviewdog** (`.github/workflows/reviewdog.yml`): Inline PR comments for clippy and rustfmt.
- **Security scanning**:
  - CodeQL (`.github/workflows/codeql.yml`)
  - `rustsec/audit-check` + Trivy (`.github/workflows/audit.yml`)
- **Dependencies**: Dependabot (`.github/dependabot.yml`) for Cargo and GitHub Actions.

## Links

- Crates.io: https://crates.io/crates/neuromod
- Docs.rs: https://docs.rs/neuromod
- Repository: https://github.com/Limen-Neural/neuromod
