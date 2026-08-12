# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

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
- **Codecov:** coverage/test-result uploads fail the job on error (no silent “unknown” badge); README documents repo activation + `CODECOV_TOKEN` for `Limen-Neural/neuromod`.

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
