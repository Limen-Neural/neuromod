# RNG inventory

> Scope: live stochastic public paths in `neuromod`, and the caller-injected
> RNG variants that make them reproducible.
> Tracked by [LIM-1221](https://linear.app/rpd-34/issue/LIM-1221/featrng-inject-caller-rng-into-stochastic-neuromod-dynamics).

`neuromod` does not own a global seed, a cryptographic generator, or sensory
encoding (that last belongs to `axon-encoder`). Callers that need replay pass
their own `&mut impl rand::Rng` into the injected variants below and keep that
one stream for the whole run.

## WebAssembly entropy backends

Browser, Web Worker, and other supported JavaScript-hosted
`wasm32-unknown-unknown` applications must explicitly enable Neuromod's
`neuromod/wasm-js` feature:

```toml
[dependencies]
neuromod = { version = "0.6", features = ["wasm-js"] }
```

This selects getrandom's upstream-supported `wasm_js` backend. It remains
opt-in because not every `wasm32-unknown-unknown` host provides JavaScript
bindings. Non-Web WASM applications should leave `wasm-js` disabled and select
an entropy backend appropriate for their final application.

## Live stochastic paths

These are the only public APIs that draw from a `rand` generator during
dynamics:

| Path | Convenience wrapper | Injected variant | What is random |
|------|---------------------|------------------|----------------|
| [`SpikingNetwork::{step, step_frozen}`](../src/engine.rs) | `step` and `step_frozen` use [`rand::rng`](https://docs.rs/rand/latest/rand/fn.rng.html) (thread-local) | [`SpikingNetwork::{step_with_rng, step_frozen_with_rng}`](../src/engine.rs) | Bernoulli encoding of `input_spike_times` for channels with `\|stimuli\| > 0.01` |
| [`lif::PoissonEncoder::encode`](../src/lif.rs) | `encode` uses the thread-local RNG | [`PoissonEncoder::encode_with_rng`](../src/lif.rs) | One Bernoulli trial per output step at the clamped intensity |

The injected methods take `rng: &mut impl Rng` (`R: Rng + ?Sized`, so
`&mut dyn Rng` also works). `Rng`, `SeedableRng`, and `StdRng` are re-exported
from this crate so a downstream `neuromod` dependency is enough to seed a
stream. Pass the same `&mut` generator on every step of a run; the injected
path does not construct, reseed, or store an RNG per neuron or per step.

For an identical starting network, input, modulator snapshot, and RNG state,
`step_frozen_with_rng` makes the same Bernoulli decisions and consumes the same
draws as `step_with_rng`. It executes the normal pipeline and then restores
plasticity-controlled state; this restoration neither draws nor rewinds the
caller's generator.

A rejected `step_with_rng` or `step_frozen_with_rng` call returns before any
draw, so invalid input does not advance the caller stream.

## Deterministic public paths

These public surfaces do not draw from `rand` during execution:

| Path | Notes |
|------|--------|
| `LifNeuron::{integrate, check_fire}` | Analog RC step; no encoding. |
| `IzhikevichNeuron::{step, step_with_time}` | Deterministic quadratic integrate-and-fire. |
| `GifNeuron` / `GifParams` | Deterministic GIF equations. |
| `SparseGifHiddenLayer::{step, step_into, run}` | Sequential fold; identical input → bit-identical raster. Topology and initial weights are generated at **construction** from `SparseGifLayerConfig::seed` via an internal SplitMix64 (not `rand`), with a per-neuron sub-stream. That seed is part of the config, not a live dynamics RNG. |
| `LapicqueNeuron` | Deterministic. |
| `FitzHughNagumoNeuron` | Deterministic. |
| `HodgkinHuxleyNeuron` | Deterministic. |
| `apply_classical_stdp` / `HebbianIzhikevichNetwork` | Deterministic given spike times. |
| Engine R-STDP (`apply_stdp`, `EligibilityTrace`, `RmStdpConfig`) | Deterministic given `input_spike_times` and neuron spike times. Stochasticity reaches this path only through the Bernoulli encoding above. |
| `NeuroModulators`, `SignalProfile`, `apply_neuromodulation`, `UnitReward` | Deterministic maps. |

## Checkpoints

`SpikingNetwork`, `LifNeuron`, and `PoissonEncoder` do not store an RNG.
Serde snapshots therefore do not capture the random stream.

- **Replay from the start:** persist the original seed with the initial network
  state. The same seed plus the same inputs reproduces the run for a given
  `rand` version and target (`StdRng` is not a portable bitstream across
  `rand` upgrades).
- **Resume a mid-run checkpoint:** persist the generator's *advanced* state
  (not only the original seed), or replay every prior draw from that seed
  before continuing. Reconstructing `StdRng` from the starting seed alone
  rewinds the stream, so later Bernoulli draws and R-STDP updates diverge from
  the uninterrupted run.

`SparseGifHiddenLayer` serializes the generated topology and SoA state, not a
live generator. Reconstructing from `SparseGifLayerConfig` (including `seed`)
reproduces the same initial topology.
