# Frozen Stepping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add frozen evaluation stepping that advances the exact `SpikingNetwork` runtime pipeline while restoring every plasticity-controlled field bit-for-bit.

**Architecture:** Both public stepping families enter one private step implementation. Frozen mode captures the persistent modulators, R-STDP configuration, and each LIF neuron's threshold/decay/weight/eligibility state after validation but before mutation; it executes the normal dynamics and learning pipeline, then restores that snapshot while retaining clocks, spike history, membranes, predictive state, inhibition, and Izhikevich dynamics.

**Tech Stack:** Rust 1.98.1, `rand` 0.10, inline unit tests, rustdoc, Cargo quality gates.

**Spec:** Linear LIM-1423 / GitHub issue #161, “feat(engine): add frozen stepping for held-out evaluation.”

## Global Constraints

- Frozen stepping must advance input-spike RNG decisions, membrane/model dynamics, clock, spikes, timestamps, predictive state, and inhibition.
- Frozen stepping must preserve weights, eligibility traces, thresholds, learned/adaptive decay state, persistent modulators, and all other plasticity-controlled state bit-for-bit.
- `step_frozen_with_rng` must consume the same caller-owned RNG stream as `step_with_rng` for the same accepted input and must consume no draws for rejected input.
- Frozen and normal stepping must use one runtime pipeline; do not duplicate STDP or add a downstream workaround.
- Public rustdoc must explain why zero dopamine is not equivalent to frozen evaluation.

---

### Task 1: Specify the frozen stepping contract with failing tests

**Files:**
- Modify: `src/engine.rs` (inline `tests` module)

**Interfaces:**
- Consumes: existing `SpikingNetwork`, `NeuroModulators`, `StdRng`, and `StepError` APIs.
- Produces: executable expectations for `step_frozen(&[f32], &NeuroModulators)` and `step_frozen_with_rng(&[f32], &NeuroModulators, &mut R)`.

- [x] **Step 1: Add bitwise plasticity snapshots**

  Add test-only helpers that capture `f32::to_bits()` for persistent modulators, `stdp_config`, each LIF neuron's `decay_rate`, `threshold`, `base_threshold`, weights, and every eligibility trace's `value` and `tau`. Include Izhikevich parameters `a`, `b`, `c`, and `d` as the non-dynamic model configuration guard.

- [x] **Step 2: Add the one-step and multi-step preservation tests**

  Construct a connected network with high dopamine, planted non-zero eligibility, non-default threshold/decay/config values, and a distinct pre-existing modulator snapshot. Assert the captured frozen snapshot is exactly equal after one accepted frozen step and after a multi-step sequence.

- [x] **Step 3: Add runtime-equivalence and observability tests**

  Run normal and frozen twins from identical state and identical seeded RNGs. Restore the normal twin's frozen fields between steps, then assert identical returned spike IDs, `global_step`, input timestamps, predictive state, LIF membrane/spike/timestamp state, and Izhikevich voltage/recovery/timestamp state. Require at least one observable LIF spike and changed runtime fields.

- [x] **Step 4: Add RNG/error-contract tests**

  Assert accepted normal/frozen calls leave identically seeded caller RNGs at the same next `u64`, and rejected frozen calls neither consume RNG nor mutate the network.

- [x] **Step 5: Verify RED**

  Run:

  ```bash
  cargo test engine::tests::frozen -- --nocapture
  ```

  Expected: compilation fails because `step_frozen` and `step_frozen_with_rng` do not exist.

### Task 2: Implement one shared normal/frozen stepping pipeline

**Files:**
- Modify: `src/engine.rs`

**Interfaces:**
- Consumes: the existing validated step sequence and test contract from Task 1.
- Produces: public `step_frozen` and `step_frozen_with_rng`; private `PlasticitySnapshot` capture/restore and one mode-aware step implementation.

- [x] **Step 1: Add the private snapshot**

  Define private snapshot types equivalent to:

  ```rust
  struct PlasticitySnapshot {
      modulators: NeuroModulators,
      stdp_config: RmStdpConfig,
      lif: Vec<LifPlasticitySnapshot>,
  }

  struct LifPlasticitySnapshot {
      decay_rate: f32,
      threshold: f32,
      base_threshold: f32,
      weights: Vec<f32>,
      eligibility: Vec<EligibilityTrace>,
  }
  ```

  Capture by copying scalar state and cloning owned vectors; restore by assignment so every IEEE-754 payload/sign bit and vector shape is retained exactly.

- [x] **Step 2: Refactor the step core**

  Move the existing `step_with_rng` body into one private mode-aware method. Perform input and counter preflight once, capture only in frozen mode, execute the unchanged runtime/plasticity pipeline, then restore the snapshot before returning spike IDs. Keep `step_with_rng` as a thin normal-mode wrapper.

- [x] **Step 3: Add public frozen wrappers**

  `step_frozen` supplies `rand::rng()` and `step_frozen_with_rng<R: Rng + ?Sized>` forwards the caller's generator. Neither stores or reseeds a generator.

- [x] **Step 4: Verify GREEN**

  Run:

  ```bash
  cargo test engine::tests::frozen -- --nocapture
  cargo test engine::tests::frozen_step_rejection_is_atomic_and_does_not_consume_rng -- --nocapture
  ```

  Expected: all new targeted tests pass.

### Task 3: Document and qualify the public API

**Files:**
- Modify: `src/engine.rs`
- Modify: `src/lib.rs`
- Modify: `docs/rng.md`

**Interfaces:**
- Consumes: completed public frozen stepping methods.
- Produces: docs.rs-visible semantics and RNG inventory coverage.

- [x] **Step 1: Write public rustdoc**

  Document that frozen stepping is for held-out evaluation, executes the normal transient dynamics, restores plasticity-controlled state, and differs from zero dopamine because zero reward still allows eligibility decay/accumulation and acetylcholine-driven decay retuning.

- [x] **Step 2: Update crate and RNG documentation**

  List frozen wrappers beside normal wrappers and state that injected normal/frozen calls consume identical RNG decisions for identical accepted inputs.

- [x] **Step 3: Run formatting and complete quality gates**

  Run:

  ```bash
  cargo fmt --check
  cargo test --all-features
  cargo clippy --all-targets --all-features -- -D warnings
  cargo hack check --feature-powerset --exclude-no-default-features --keep-going
  ```

  Expected: every command exits successfully with no warnings.

  Local result: formatting, all-feature tests, strict Clippy, and explicit
  checks for the default, all-feature, no-default, and `wasm-js` combinations
  passed. `cargo-hack` was not installed locally, so the equivalent explicit
  feature checks were run and the exact `cargo hack` gate is deferred to CI.

- [x] **Step 4: Review the diff and commit**

  Confirm only LIM-1423 files changed, then commit with:

  ```bash
  git add src/engine.rs src/lib.rs docs/rng.md docs/superpowers/plans/2026-09-20-frozen-stepping.md
  git commit -m "feat(engine): add frozen evaluation stepping" \
    -m "Co-authored-by: Codex <noreply@openai.com>"
  ```
