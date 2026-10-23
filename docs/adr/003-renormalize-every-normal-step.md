# ADR 003: Renormalize LIF weights on every Normal step (do not skip dopamine-silent steps)

**Status:** Accepted

**Date:** 2026-10-23

## Context

`SpikingNetwork::renormalize_lif_weights` (`src/engine.rs`) runs after `apply_stdp` on every
Normal step: an `O(num_lif × num_channels)` sum, a scale toward the L1 budget
(`WEIGHT_BUDGET`), and a clamp to the `RmStdpConfig` bounds. It is one of several
per-synapse linear scans in the step, alongside trace decay, coincidence accumulation,
integration, and reward conversion.

The pass is not "needed" for the engine's own writes on every step: when the dopamine-derived
learning rate is below `1e-6`, `apply_stdp` still decays and accumulates traces but writes no
weights, so on such a step the renormalization pass can only reach weights someone else
changed. Because `LifNeuron::weights` is a public field and every self-describing checkpoint
format deserializes straight into it, "someone else" includes any caller who edits or restores
weights between steps. Skipping the pass on those dopamine-silent steps would therefore change
a policing behavior that the `step` contract documents (rustdoc, "Order of work", item 9) and
the soak gate asserts after every step (`tests/soak.rs`, L1 within `1e-4` of the budget):
after each Normal step, every non-blank neuron's weights are on the L1 budget or bound-limited
— including caller-edited ones, not just STDP-written ones.

Measured cost (GH#193 / LIM-1507; 64 LIF × 64 channels, steady state after 200 warm-up steps,
rustc 1.98.1, Criterion `--warm-up-time 1 --measurement-time 3 --sample-size 30`, two runs per
build on the shared bench VM described in `benches/README.md`):

- Pass unconditional (current): `engine_step/unrewarded` 41.4 / 43.5 µs, `engine_step/rewarded`
  50.2 / 48.4 µs.
- Pass skipped when the dopamine-derived learning rate is < `1e-6`:
  `engine_step/unrewarded` 37.4 / 38.8 µs, `engine_step/rewarded` 46.9 / 47.4 µs.

The unrewarded mean fell ~10%, but the rewarded control — whose per-step work is identical,
because dopamine 0.9 keeps the pass on — also fell ~4%, and a no-change rebuild moved that same
control by −3.5% on its own. Between-build noise on this VM is ±3–5%. Correcting for it, the
pass costs roughly 2 µs of a ~41 µs unrewarded step (≈5%). This matches the issue's
observation that unrewarded and rewarded steps sit only ~7% apart: trace decay and
renormalization dominate both, and the trace → weight conversion is essentially the only
difference between them.

## Decision

Keep `renormalize_lif_weights` unconditional on every Normal step. Treat its per-step cost as
the intentional price of the invariant it enforces, and record that rationale where the
question arises: the `step` contract (item 9), the pass's rustdoc, this ADR, and the
`engine_step` notes in `benches/README.md`. Lock the behavior with a contract test: a
dopamine-silent step must still re-normalize caller-edited, off-budget weights.

- The engine cannot soundly track dirtiness. A "weights changed" flag only sees writes the
  engine itself made; direct public-field edits and deserialized checkpoints are invisible to
  it, so the flag either misses them (silently weakening the invariant) or must assume dirty
  (saving nothing).
- The invariant is public and per-step, not an internal optimization detail. The `step`
  rustdoc, the `RmStdpConfig` docs, and `tests/soak.rs` all state or depend on it.
- The saving is bounded and small. The corrected cost is ~5% of an unrewarded step; even the
  uncorrected −10% point estimate is barely outside the ±3–5% between-build noise band. A
  caller who must not pay the plasticity phase at all already has the frozen mode
  (`step_frozen` / `step_frozen_with_rng`), which skips STDP and renormalization entirely.

## Consequences

- No behavior or performance change ships with this decision. The pass, its contract, and its
  cost stay as they are; the decision is recorded so the next reader does not re-litigate it
  as an oversight.
- A dopamine-silent step still scans — and, where a neuron's weights are off budget, rescales
  — every neuron's weights. A caller who plants off-budget weights between steps sees them
  pulled back to budget on the next Normal step, rewarded or not. A blank neuron stays blank
  and an exactly-zero synapse stays zero (the `> 1e-6` total guard and the zero-skip in the
  write loop).
- On a steady-state dopamine-silent step the pass is a read-only scan: weights left on budget
  by the previous pass scale by `WEIGHT_BUDGET / total ≈ 1.0` and re-clamp to values already
  in bounds.
- Reversing this decision later requires making `LifNeuron::weights` private (a breaking API
  change with a pre-1.0 minor bump and README migration notes) or an explicit, documented
  contract change to the per-step budget guarantee plus a soak-gate update. Neither is a
  performance patch.

## Alternatives Considered

- **Skip the pass when the dopamine-derived learning rate is < `1e-6`** — Rejected. It saves
  ~5% of an unrewarded step (measured above) and silently weakens the step-9 contract:
  caller-edited or restored off-budget weights would persist through unrewarded stretches,
  and the soak gate's per-step L1 assertion would hold only while incidental spike dynamics
  happen to trigger reward conversion.
- **Weights-dirty flag** — Rejected. The flag can record `apply_stdp`'s own writes, but
  `LifNeuron::weights` is public: direct edits and deserialization bypass any setter, so the
  flag either misses them (the same silent weakening) or must start and stay dirty (saving
  nothing). Making the field private to fix that is a breaking API change far larger than the
  ~5% it buys.
- **Explicit opt-in renormalization** (a `RmStdpConfig` switch, or moving the pass out of
  `step` into a public method callers invoke by hand) — Rejected. Defaulting to the current
  per-step pass saves nothing; defaulting to the cheap path removes homeostatic L1 scaling
  from R-STDP learning dynamics, a semantic change to what the rule does rather than a cost
  adjustment. An opt-in switch also duplicates what the frozen mode already provides for
  evaluation-time stepping.
- **Document the per-step cost as intentional** — Accepted. The pass is the enforcement cost
  of a public invariant over state the engine cannot track; it is proportional to synapse
  count, and its measured share of the step is small.

## References

- Issue: [#193](https://github.com/Limen-Neural/neuromod/issues/193) /
  [LIM-1507](https://linear.app/rpd-34/issue/LIM-1507/perfengine-decide-whether-renormalize-lif-weights-should-run-on)
  — "decide whether renormalize_lif_weights should run on unrewarded steps"
- Contract: `SpikingNetwork::step` rustdoc, "Order of work", item 9 (`src/engine.rs`)
- Per-step budget assertion: `tests/soak.rs` (`assert_l1_budget` after every step)
- Measurement notes: `benches/README.md`, `engine_step` section
- Prior engine decision records:
  [ADR 001](https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/001-traits-in-neuromod.md),
  [ADR 002](https://github.com/Limen-Neural/neuromod/blob/main/docs/adr/002-wire-eligibility-traces.md)
