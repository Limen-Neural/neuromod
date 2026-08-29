# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **R-STDP is wired into the engine.** `LifNeuron` gains `eligibility: Vec<EligibilityTrace>`
  (one trace per input channel, indexed like `weights`) and `SpikingNetwork` gains
  `stdp_config: RmStdpConfig`. `SpikingNetwork::apply_stdp` now decays and accumulates those
  traces on every step and converts them into weight changes under the dopamine gate, so
  reward arriving *after* a coincidence still pays for it (#72, #73).
- `SpikingNetwork::set_rm_stdp_config` — replace the R-STDP hyperparameters and re-`tau`
  existing traces in one call.
- `EligibilityTrace::new` / `kernel` / `accumulate` / `reset`, plus `Default` impls for
  `EligibilityTrace` and `RmStdpConfig`, and the `RM_STDP_TAU_ELIGIBILITY` /
  `RM_STDP_REWARD_LR` constants behind those defaults.
- `RmStdpConfig::weight_bounds` — ordered `(min, max)` accessor that falls back to
  `RM_STDP_W_MIN` / `RM_STDP_W_MAX` when the public bound fields are left reversed or
  non-finite, so a hand-edited config cannot panic `f32::clamp` inside `step`.
- `RmStdpConfig::effective_reward_lr` — same guard for a non-finite `reward_lr`. A `NaN`
  rate would poison a weight on the first rewarded step and never clear, because the L1
  renormalization pass skips any neuron whose total is not `> 1e-6` and `NaN > 1e-6` is
  false. Finite negative rates still pass through.
- `RmStdpConfig::effective_tau_eligibility` — same guard for `tau_eligibility`, and
  `EligibilityTrace::decay` now falls back the same way. A `NaN` tau used to erase every
  banked trace on the next step (`exp(-1/f32::EPSILON) == 0`) and `+∞` disabled decay
  entirely; both now degrade to `RM_STDP_TAU_ELIGIBILITY`.
- Tests proving the reward-gated path: traces accumulate with dopamine off while weights
  hold, dopamine converts the banked trace (and more dopamine buys more learning),
  post-before-pre depresses and clamps at `w_min`, one spike pair is counted once, and a
  pre-0.6 checkpoint without the new fields still deserializes and steps (#74).
- Benchmarks for trace accumulation, the trace → weight conversion, and rewarded vs
  unrewarded engine steps (`benches/stdp_bench.rs`).
- [ADR 002](docs/adr/002-wire-eligibility-traces.md) recording the wire-vs-demote decision.

### Changed

- **Breaking (targets 0.6.0):** weight updates flow through a decaying eligibility trace
  instead of being recomputed from raw spike times each step, so same-input runs will not
  reproduce pre-0.6 weight trajectories. `apply_stdp` no longer early-returns when dopamine
  is ~0 — only the trace → weight conversion is gated. Weight bounds now come from
  `stdp_config` rather than the `RM_STDP_W_MIN` / `RM_STDP_W_MAX` constants directly, in both
  `apply_stdp` and the L1 renormalization pass (the constants remain public and are the
  defaults) (#72, #73).
- **Breaking (source, targets 0.6.0):** `LifNeuron` and `SpikingNetwork` have public fields
  and are not `#[non_exhaustive]`, so the new `eligibility` / `stdp_config` fields break
  downstream struct literals that spell out every field. Fill the remainder from
  `..LifNeuron::new()` / `..Default::default()`, or build through the constructors. Serialized
  state written by 0.5.x stays *deserializable* in self-describing formats (JSON, YAML, RON,
  map-encoded MessagePack) via `#[serde(default)]`; positional binary formats such as
  `bincode` / `postcard` cannot use those defaults and will not load pre-0.6 bytes. The
  serialized shape does change — checkpoints written by 0.6 carry `eligibility` and
  `stdp_config`, so they will not load into 0.5.x. See the README migration notes (#72, #73).
- Weight bounds now take documented precedence over the engine's L1 weight budget: `step`
  scales toward the budget and then clamps, so a narrowed `w_min` / `w_max` leaves the sum
  off budget. The defaults cannot bind, so the budget still holds exactly under them. The
  renormalization pass leaves a synapse at exactly zero alone, so a positive `w_min` cannot
  conjure a connection on an unrewarded step; learning raises a synapse to the floor in the
  reward-gated `apply_stdp` instead.
- `examples/rstdp_demo.rs` prints real trace and weight numbers read back from the network
  instead of narrating hardcoded claims about learning.
- Docs no longer carry the "eligibility traces are not wired" caveat: crate root, `engine`,
  `lif`, and `rm_stdp` rustdoc, `README.md`, `CLAUDE.md`, and the `REVIEW.md` regression
  guards describe (and guard) the wired path.
- **Docker:** CI pushes example runtime images to Docker Hub and **GHCR** (`ghcr.io/limen-neural/neuromod` with SHA, version, and `latest` tags) so the image appears under GitHub org packages; README documents pull URLs.
- **Docker verify:** PR job asserts example binaries exist in the runtime image; publish job requires full `X.Y.Z` crate version for tags.
- README crates.io / docs.rs badges stay version-agnostic (latest); install pin documents **0.5.2**.

## [0.5.2] - 2026-08-12

Docs, CI, and packaging hygiene for the v0.5.2 milestone. Publish with `cargo publish` after this tag lands.

### Removed

- Optional `sentry` Cargo feature, `examples/sentry.rs`, `tests/sentry_integration.rs`, and `.github/workflows/sentry-release.yml`. Observability belongs in application binaries (depend on `sentry` there); the library no longer pulls or demos Sentry (#88, #78).

### Changed

- Stop tracking `.cubic/wiki` in git; ignore `.cubic/` and exclude it from the crates.io package so regenerated cubic wiki is not a documentation source of truth (#90).
- **Rustdoc textbook spine:** crate-root syllabus, module docs for `engine` / `lif` / `izhikevich` / `modulators`, and expanded `SpikingNetwork::step` contract (pipeline order, STDP honesty, doctest) (#89).
- **MSRV policy:** README Requirements + crate docs state Rust **1.97.1**; CI verifies `Cargo.toml` / `rust-toolchain.toml` / workflow pin stay identical (#79).
- **CI OS matrix:** GitHub Actions core CI runs build/clippy/tests on Linux, macOS, and Windows (`ubuntu-latest`, `macos-latest`, `windows-latest`); README Requirements and CI section document the three-platform matrix and the `paths-filter` gate for nextest/cargo-hack (#94, #95).
- Domain-agnostic docs check builds with `cargo doc --all-features --no-deps` before the forbidden-term scan.
- **Codecov:** official badge markdown with graph token; CI uploads via `CODECOV_TOKEN` (tokenless returns HTTP 400 for this org); upload steps use `fail_ci_if_error: false` so coverage remains non-blocking; coverage workflow declares `permissions: contents: read`.

## [0.5.1] - 2026-08-08

### Changed

- **Docs / packaging honesty:** honest crates.io `description` (no unsubstantiated “high-performance”); README badges and dual-license presentation; crate and README docs state that `SpikingNetwork` wires **LIF + Izhikevich** only, with other models available as standalone types (#66, #67, #69, #70).
- `src/hodgkin_huxley.rs`: stripped verbose AI-generated commentary and consolidated repeated RK4/gating calculations.
- `src/lif.rs`: shortened Poisson-spike generation comment.
- `tests/sentry_integration.rs`: trimmed redundant feature-gate prose.

### Fixed

- Fixed `HodgkinHuxleyNeuron::derivatives` to shift the membrane potential back to the HH relative convention before evaluating α/β gating rates for cortical (mammalian) parameterizations, matching the shift already used by `steady_state_gating_mammalian` and `reset`.

## [0.5.0] - 2026-07-30

Published to [crates.io](https://crates.io/crates/neuromod/0.5.0).

### Added

- **Generic neuromodulator API** — `NeuroModulators` now exposes `dopamine`, `serotonin`, `acetylcholine`, and `norepinephrine`
- **`SignalProfile`** — configurable mapping from external signals to modulator levels (neutral defaults; optional `hardware_calibrated()` for legacy callers)
- **`GenericReward` trait** and **`Observation`** — domain-agnostic reward shaping interface for downstream crates
- **`UnitReward`** — simple mean-signal reward implementation for tests and demos
- **`apply_neuromodulation`** — standalone function to apply modulator effects to weight and threshold slices
- **GitHub Actions CI** — `fmt`, `clippy`, `build`, and `test` on push/PR to `main`
- Documentation for org modularization program:
  - `docs/org-modularization.md` (standards, workstream index for #35–#43, git/build/beads rules, audit commands)
  - `docs/adr/001-traits-in-neuromod.md` (records decision to host shared traits in neuromod)
  - `docs/neuromod-boundary-matrix.md` (runtime/deployment boundary matrix per LIM-9 / #11 / #25)
- `CLAUDE.md` architecture/commands reference for AI coding agents.
- Unit tests for `EligibilityTrace::decay` (`src/rm_stdp.rs`) and `LifNeuron`/`PoissonEncoder` (`src/lif.rs`), previously untested.
- `REVIEW.md` regression guard against the Criterion benchmark harness reverting to `harness = true`.

### Fixed

- **Benchmarks:** all four `[[bench]]` targets (`neuron_bench`, `stdp_bench`, `memory_bench`, `modulation_bench`) had `harness = true` in `Cargo.toml`, which left Cargo's default libtest harness attached instead of Criterion's runner — `cargo bench` silently reported `running 0 tests` instead of executing any benchmark. Fixed to `harness = false`.
- Removed first-person development-log commentary from `src/rm_stdp.rs` doc comments that leaked into `cargo doc`/docs.rs output.
- **Sentry example:** dropped the info-level `capture_message` probe (issue noise) and set `environment` from `SENTRY_ENVIRONMENT` (default `development`); non_exhaustive `ClientOptions` builder update for sentry 0.49.
- **PoissonEncoder:** full-intensity (probability 1.0) and zero-intensity paths no longer depend on floating-point RNG bounds (avoids flaky all-ones encoding).
- **`.gitignore`:** restructured AI-tool ignores so selected `.kilo` / `.mimocode` / `.devin` paths can be force-committed.

### Changed

- **License:** switched from GPL-3.0 to dual MIT/Apache-2.0 for maximum adoption and ecosystem health.
- README now links the new Architecture & Boundaries docs.
- Added `homepage` and `documentation` fields to `Cargo.toml` for crates.io/docs.rs metadata.
- Explicit `[profile.bench]` inherits `release` so Criterion runs with the same LTO/codegen settings.
- Dropped unused direct `serde_json` dependency (serialization uses `serde` derives only).
- **Breaking:** removed `cortisol`, `tempo`, and `aux_dopamine` fields from `NeuroModulators`
- **Breaking:** `from_signals` now requires a `&SignalProfile` as its first argument
- **Breaking:** `add_stress` renamed to `add_norepinephrine`; `is_stressed` renamed to `is_aroused`
- Replaced `"spikenaut"` crates.io keyword with `"neuromodulation"`
- Documentation and crate-level docs are now domain-agnostic

### Removed

- Domain-specific mining/HFT metadata from changelog and public documentation
- Eagle-Lander provenance from crate docs

## [0.4.0] - 2026-05-01

### Changed

- Topology-neutral network initialization with dynamic sizing via `SpikingNetwork::with_dimensions`
- Strict input validation via `StepError::InputLenMismatch`

## [0.3.0] - 2026-04-01

### Added

- Extended neuron model library (Lapicque, GIF, Hodgkin-Huxley, FitzHugh-Nagumo)
- Classical Hebbian STDP utilities

## [0.1.0] - 2026-02-01

### Added

- Initial release: LIF/Izhikevich network, reward-modulated STDP, neuromodulator system
