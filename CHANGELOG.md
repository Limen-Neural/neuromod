# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

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
- **Sentry example:** dropped the info-level `capture_message` probe (issue noise) and set `environment` from `SENTRY_ENVIRONMENT` (default `development`).

### Changed

- **License:** switched from GPL-3.0 to dual MIT/Apache-2.0 for maximum adoption and ecosystem health.
- README now links the new Architecture & Boundaries docs.
- Added `homepage` and `documentation` fields to `Cargo.toml` for crates.io/docs.rs metadata.
- Explicit `[profile.bench]` inherits `release` so Criterion runs with the same LTO/codegen settings.
- Dropped unused direct `serde_json` dependency (serialization uses `serde` derives only).

## [0.5.0] - 2026-06-20

### Added

- **Generic neuromodulator API** — `NeuroModulators` now exposes `dopamine`, `serotonin`, `acetylcholine`, and `norepinephrine`
- **`SignalProfile`** — configurable mapping from external signals to modulator levels (neutral defaults; optional `hardware_calibrated()` for legacy callers)
- **`GenericReward` trait** and **`Observation`** — domain-agnostic reward shaping interface for downstream crates
- **`UnitReward`** — simple mean-signal reward implementation for tests and demos
- **`apply_neuromodulation`** — standalone function to apply modulator effects to weight and threshold slices
- **GitHub Actions CI** — `fmt`, `clippy`, `build`, and `test` on push/PR to `main`

### Changed

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
