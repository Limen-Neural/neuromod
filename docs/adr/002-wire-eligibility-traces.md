# ADR 002: Wire eligibility traces into the engine (do not demote)

**Status:** Accepted

**Date:** 2026-08-29

## Context

`neuromod` shipped `EligibilityTrace` and `RmStdpConfig` as public reward-modulated STDP
(R-STDP) types, and the README listed them under "reward-modulated STDP types". Neither type
was consumed by the live learning path: `SpikingNetwork::apply_stdp` recomputed a
spike-timing kernel inline and wrote the result straight into `LifNeuron::weights`. The
eligibility half of R-STDP — accumulate a trace of recent pre/post coincidences, then let a
reward signal convert that trace into a weight change — existed only as an unused building
block.

That is the honesty gap tracked by GH#72 (epic), GH#73 (wire or demote), and GH#74 (prove it
with tests): the public surface promised R-STDP, the engine delivered dopamine-gated direct
STDP. Every source doc that disclosed the gap (`src/rm_stdp.rs`, `src/engine.rs`,
`src/lib.rs`, `CLAUDE.md`) had to repeat the caveat, and downstream crates could not tell
which half was real without reading the engine.

Two ways to close it: make the promise true, or shrink the promise.

## Decision

Wire `EligibilityTrace` and `RmStdpConfig` into the live engine path. Do not demote them to
experimental.

- `LifNeuron` gains `eligibility: Vec<EligibilityTrace>`, one trace per input channel,
  indexed exactly like `weights`.
- `SpikingNetwork` gains an `stdp_config: RmStdpConfig` field holding the trace time
  constant, the reward learning rate, and the weight bounds.
- `SpikingNetwork::apply_stdp` decays and accumulates traces **every step, independent of
  dopamine**, and gates only the trace → weight conversion on dopamine.
- Both new fields are `#[serde(default)]`, so pre-0.6 checkpoints still deserialize; the
  engine resizes a missing or stale trace vector to match `weights` before use.

Trace accumulation is **event-gated**: a coincidence is recorded only on the step where the
post neuron or the pre channel actually spiked, never re-recorded from a stale pair on
subsequent steps. A single spike pair therefore leaves a single decaying imprint, which is
what an eligibility trace means. Re-applying the kernel every step from persistent
`last_spike_time` values would inflate the trace to roughly `tau x kernel` for a pair that
fired once — learning that looks real but is an artifact of the loop, exactly the "silent
no-op that looks like learning" GH#73 rules out.

## Consequences

- **Behavior change (breaking).** Weight updates now flow through a decaying trace instead of
  being recomputed from raw spike times each step. Same-input runs will not reproduce pre-0.6
  weight trajectories. Pre-1.0 policy bumps the minor version (`0.5.x` -> `0.6.0`).
- Learning has memory. A coincidence that happened while dopamine was zero can still be
  converted into a weight change several steps later when reward arrives — the credit
  assignment R-STDP exists to provide, and the reason dopamine no longer short-circuits the
  whole plasticity pass.
- `apply_stdp` now runs every step rather than early-returning when dopamine is ~0. Traces
  that are still exactly zero skip both decay and conversion, so a blank or unrewarded
  network stays cheap.
- Weight bounds come from `stdp_config` rather than the `RM_STDP_W_MIN` / `RM_STDP_W_MAX`
  constants directly, in `apply_stdp` and in the L1 renormalization pass. The constants
  remain public and are the config defaults.
- `stdp_config` is a public field like every other field on `SpikingNetwork`. Assigning it
  directly does not re-`tau` traces that already exist, so
  `SpikingNetwork::set_rm_stdp_config` is provided to keep the two consistent.
- Docs no longer carry the "not yet wired" caveat, and the R-STDP claim is backed by tests
  (`src/rm_stdp.rs`, `src/engine.rs`) and by `examples/rstdp_demo.rs`, which prints real
  trace and weight numbers instead of narrating them.

## Alternatives Considered

- **Demote to experimental** (`#[doc(hidden)]`, feature gate, or move to an `experimental`
  module) — Rejected. It closes the honesty gap by deleting the feature rather than
  delivering it, and the v0.6 milestone is "product completeness". The types are small, the
  wiring is contained in one private method, and reward-modulated plasticity is the point of
  a crate named `neuromod`.
- **Keep the inline rule and consume traces in parallel** (traces updated but not driving
  weights) — Rejected. That is the current state with extra bookkeeping: two plasticity
  stories, one of them decorative.
- **Accumulate the kernel every step from `last_spike_time`** (the mechanically simpler
  wiring) — Rejected for the inflation reason above.
- **Add `RmStdpConfig` as a `with_dimensions` parameter** — Rejected. It breaks every
  existing call site for a value that has a sane default; `set_rm_stdp_config` is additive.

## References

- Epic: #72 (close the R-STDP / eligibility gap)
- Children: #73 (wire or demote), #74 (test the reward-gated path)
- Milestone: neuromod v0.6 — Product completeness
- Implementation: `src/rm_stdp.rs`, `src/lif.rs`, `src/engine.rs`
- Proof: unit and multi-step tests in `src/rm_stdp.rs` and `src/engine.rs`; `examples/rstdp_demo.rs`
- Prior art on the split between engine and standalone building blocks: `docs/adr/001-traits-in-neuromod.md`
