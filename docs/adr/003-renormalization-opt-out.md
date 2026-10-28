# ADR 003: Per-step LIF weight renormalization — on by default, explicit opt-out

**Status:** Accepted

**Date:** 2026-10-27 (supersedes the 2026-10-23 document-only decision drafted on this branch)

## Context

`SpikingNetwork::renormalize_weights` (`src/engine.rs`) runs after `apply_stdp` on every
Normal step: an `O(num_lif × num_channels)` sum, a scale toward the L1 budget
(`WEIGHT_BUDGET`), and a clamp to the `RmStdpConfig` bounds. It is one of several
per-synapse linear scans in the step, alongside trace decay, coincidence accumulation,
integration, and reward conversion.

Two questions shaped this record.

**Is the per-step pass worth its cost?** On steps where the dopamine-derived learning rate is
below `1e-6`, `apply_stdp` still decays and accumulates traces but writes no weights, so the
renormalization pass can only reach weights someone else changed. Because `LifNeuron::weights`
is a public field and every self-describing checkpoint format deserializes straight into it,
"someone else" includes any caller who edits or restores weights between steps. Skipping the
pass on those dopamine-silent steps would therefore change a policing behavior that the
`step` contract documents (rustdoc, "Order of work", item 9) and the soak gate asserts after
every step (`tests/soak.rs`, L1 within `1e-4` of the budget).

Measured cost (GH#193 / LIM-1507; 64 LIF × 64 channels, steady state after 200 warm-up steps,
rustc 1.98.1, Criterion `--warm-up-time 1 --measurement-time 3 --sample-size 30`, two runs per
build on the shared bench VM described in `benches/README.md`):

- Pass unconditional (then current): `engine_step/unrewarded` 41.4 / 43.5 µs,
  `engine_step/rewarded` 50.2 / 48.4 µs.
- Pass skipped when the dopamine-derived learning rate is < `1e-6`:
  `engine_step/unrewarded` 37.4 / 38.8 µs, `engine_step/rewarded` 46.9 / 47.4 µs.

The unrewarded mean fell ~10%, but the rewarded control — whose per-step work is identical,
because dopamine 0.9 keeps the pass on — also fell ~4%, and a no-change rebuild moved that same
control by −3.5% on its own. Between-build noise on this VM is ±3–5%. Correcting for it, the
pass costs roughly 2 µs of a ~41 µs unrewarded step (≈5%). That answers the performance
question: the pass is cheap, and the default is not worth weakening to chase 5%.

**Can a caller run with chosen weights that are never rescaled?** RM-2352 (sparse,
desynchronised firing in the hero network) needs exactly that: a neuron meant to listen to one
channel wants a diagonal weight of its own choosing, and the per-step L1 pass rescales any
total that is not the budget. The workaround used so far is planting a diagonal of exactly the
budget (2.0) — workable, but rigid: it forces a listening neuron to spend its whole budget on
one channel and leaves no way to express anything else.

## Decision

Keep the pass on every Normal step by default — behavior bit-identical to before — and add an
explicit opt-out for callers who need un-rescaled weights:

1. `RmStdpConfig::renormalize_weights: bool`, default `true`, deserializing a checkpoint that
   predates the field to `true` so existing state keeps the contract it was written under.
   When `false`, `step` never renormalizes — rewarded and unrewarded steps alike.
2. Expose the pass as `SpikingNetwork::renormalize_weights()`, so an opt-out caller can police
   the public `weights` field on demand (after a hand edit or a checkpoint load) with one
   `O(num_lif × num_channels)` call instead of one per step.
3. `engine_step/preserve` joins `engine_step/unrewarded` / `engine_step/rewarded` in the
   Criterion suite so the opt-out's per-step saving is measured, not asserted.

Why the default stays on: `LifNeuron::weights` is public and checkpoints deserialize straight
into it, so the engine cannot know weights changed without scanning them. The pass is what
upholds the documented step contract (on-budget weights after every Normal step, which
`tests/soak.rs` asserts per step) against caller edits and restored checkpoints — not just
against STDP writes.

Why the opt-out is safe for those who choose it: weight updates from `apply_stdp` are
individually clamped to the `RmStdpConfig` bounds, so learning alone cannot push a weight past
`w_max`. The pass is the only enforcement of the *budget*; opting out is an explicit statement
that the caller owns weight magnitudes.

This supersedes the 2026-10-23 draft decision recorded earlier on this branch ("keep the pass
unconditional; document the cost as intentional; no switch"). Document-only cannot satisfy the
RM-2352 requirement to run with caller-chosen weights that are not rescaled. The measurement
from that draft is retained above and is what justifies keeping the default unchanged.

## Consequences

- The default path is bit-identical: goldens, the soak gate's per-step L1 assertion, and the
  on-budget invariant tests pass untouched.
- Opt-out callers' weights survive every Normal step un-rescaled; the on-demand
  `renormalize_weights()` call restores budget + bounds policing in a single pass. Zero
  weights stay zero and blank neurons stay blank in both the per-step pass and the on-demand
  call (the `> 1e-6` total guard and the zero-skip in the write loop).
- A dopamine-silent default step still polices caller-edited, off-budget weights — locked by
  `test_dopamine_silent_step_renormalizes_caller_edited_weights` (`src/engine.rs`).
- Checkpoints written before the field existed deserialize with `renormalize_weights: true`
  and keep renormalizing; no migration is needed. The field follows the crate's
  missing-field-takes-default serde convention for self-describing formats.
- Adding the field to `RmStdpConfig` breaks exhaustive struct literals; migration is
  `renormalize_weights: ...` or `..RmStdpConfig::default()`.
- A weights-dirty flag remains impossible (public-field edits bypass any flag), and a
  dopamine-level-dependent skip remains rejected: whether weights are policed must not depend
  on whether reward happens to be present on a given step.

## Alternatives Considered

- **Skip the pass when the dopamine-derived learning rate is < `1e-6`** — Rejected. Saves ~5%
  of an unrewarded step (measured above) and silently weakens the step-9 contract:
  caller-edited or restored off-budget weights would persist through unrewarded stretches,
  and behavior would differ between rewarded and unrewarded steps for the same weights.
- **Weights-dirty flag** — Rejected. The flag can record `apply_stdp`'s own writes, but
  `LifNeuron::weights` is public: direct edits and deserialization bypass any setter, so the
  flag either misses them (the same silent weakening) or must start and stay dirty (saving
  nothing).
- **Document-only, no switch (the 2026-10-23 draft)** — Superseded. It answered the
  performance question but cannot satisfy RM-2352's requirement to run with caller-chosen
  weights that are not rescaled; the diagonal-of-exactly-the-budget workaround stays
  necessary.
- **Frozen mode (`step_frozen` / `step_frozen_with_rng`) instead of a switch** — Kept as a
  separate tool, not a substitute. It skips all plasticity changes and preserves learning
  state bit-for-bit, which is right for held-out evaluation but wrong for a network that must
  keep learning with un-rescaled weights.
- **Explicit opt-out, default unchanged** — Accepted. The default keeps the public per-step
  budget contract; opting out is an explicit statement that the caller owns weight magnitudes,
  and the on-demand pass keeps the budget machinery available when it is wanted.

## References

- Issue: [#193](https://github.com/Limen-Neural/neuromod/issues/193) /
  [LIM-1507](https://linear.app/rpd-34/issue/LIM-1507/perfengine-decide-whether-renormalize-lif-weights-should-run-on)
  — "decide whether renormalize_lif_weights should run on unrewarded steps"; decision
  comment 2026-10-27.
- Requirement driving the opt-out:
  [RM-2352](https://linear.app/rpd-34/issue/RM-2352/featwasm-sparse-desynchronised-firing-in-the-hero-network)
  (sparse, desynchronised firing in the hero network), tracked under
  [#203](https://github.com/Limen-Neural/neuromod/issues/203); the per-neuron input API
  stays a separate issue
  ([#208](https://github.com/Limen-Neural/neuromod/issues/208)) that this decision does not
  close.
- Contract: `SpikingNetwork::step` rustdoc, "Order of work", item 9 (`src/engine.rs`)
- Per-step budget assertion: `tests/soak.rs` (`assert_l1_budget` after every step)
- Measurement notes: `benches/README.md`, `engine_step` section
- Prior engine decision records:
  [ADR 001](https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/001-traits-in-neuromod.md),
  [ADR 002](https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/002-wire-eligibility-traces.md)
