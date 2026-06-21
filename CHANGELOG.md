# Changelog

All notable changes to this project are documented in this file.

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
