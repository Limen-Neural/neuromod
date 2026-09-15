# RNG inventory

> Scope: live stochastic public paths in `neuromod`, and the caller-injected
> RNG variants that make them reproducible.
> Tracked by [LIM-1221](https://linear.app/rpd-34/issue/LIM-1221/featrng-inject-caller-rng-into-stochastic-neuromod-dynamics).

`neuromod` does not own a global seed, a cryptographic generator, or sensory
encoding (that last belongs to `axon-encoder`). Callers that need replay pass
their own `&mut impl rand::Rng` into the injected variants below and keep that
one stream for the whole run.

## Live stochastic paths

These are the only public APIs that draw from a `rand` generator during
dynamics:

| Path | Convenience wrapper | Injected variant | What is random |
|------|---------------------|------------------|----------------|
| [`SpikingNetwork::step`](../src/engine.rs) | `step` uses [`rand::rng`](https://docs.rs/rand/latest/rand/fn.rng.html) (thread-local) | [`SpikingNetwork::step_with_rng`](../src/engine.rs) | Bernoulli encoding of `input_spike_times` for channels with `\|stimuli\| > 0.01` |
| [`lif::PoissonEncoder::encode`](../src/lif.rs) | `encode` uses the thread-local RNG | [`PoissonEncoder::encode_with_rng`](../src/lif.rs) | One Bernoulli trial per output step at the clamped intensity |

`step` and `encode` remain source-compatible. The injected methods take
`rng: &mut impl rand::Rng` (`R: Rng + ?Sized` in the signature so `&mut dyn Rng`
also works). Pass the same `&mut` generator on every step of a run; the
injected path does not construct, reseed, or store an RNG per neuron or per
step.

A length-mismatch error from `step_with_rng` returns before any draw, so a
rejected step does not advance the caller stream.

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
Serde snapshots therefore do not capture the random stream. Replay requires
the caller to persist their own seed (or generator state) alongside the
network checkpoint.

`SparseGifHiddenLayer` serializes the generated topology and SoA state, not a
live generator. Reconstructing from `SparseGifLayerConfig` (including `seed`)
reproduces the same initial topology.
