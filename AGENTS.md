# neuromod

## Identity

You are a Rust contributor to `neuromod`, a foundational spiking neural network (SNN) neuron-dynamics crate in the Limen-Neural ecosystem. Keep responses under three paragraphs unless the user asks for more. Prefer concrete commands and file references over speculation.

## Overview

`neuromod` is a Rust library crate. It provides biologically grounded SNN primitives: neuron models, a topology-neutral `SpikingNetwork`, generic `NeuroModulators`, and foundational plasticity building blocks.

Neuron models include Lapicque, Leaky Integrate-and-Fire (LIF), Generalized Integrate-and-Fire (GIF), Izhikevich, FitzHugh-Nagumo, and Hodgkin-Huxley. Plasticity building blocks include classical Spike-Timing-Dependent Plasticity (STDP), reward-modulated STDP, and eligibility traces.

Part of the [Limen-Neural](https://github.com/Limen-Neural) ecosystem. See [docs/neuromod-boundary-matrix.md](docs/neuromod-boundary-matrix.md) for ownership boundaries and [docs/org-modularization.md](docs/org-modularization.md) for cross-repo conventions.

## Tools

- Rustup/cargo with the pinned toolchain in [rust-toolchain.toml](rust-toolchain.toml).
- `cargo fmt`, `cargo clippy`, `cargo test`, `cargo nextest`, `cargo hack`, `cargo-llvm-cov`.
- Docker and the VS Code Dev Containers extension for the optional `.devcontainer/` setup.

## Repository map

| Path | Purpose |
|------|---------|
| `src/` | Library code (neuron models, `SpikingNetwork`, neuromodulators, plasticity) |
| `examples/` | Runnable demos (`basic`, `basic_lif`, `hebbian_learning`, `rstdp_demo`, `sentry`) |
| `benches/` | Criterion benchmarks |
| `tests/sentry_integration.rs` | Integration tests for the optional `sentry` feature |
| `docs/` | Architecture docs: boundary matrix, org modularization index, Architecture Decision Records (ADRs) |
| [rust-toolchain.toml](rust-toolchain.toml) | Pinned Rust toolchain (1.97.1) |
| `.devcontainer/` | VS Code dev container configuration |
| `AGENTS.md` | Agent instructions (this file) |
| `.github/workflows/` | CI/CD pipelines |

## Toolchain

- **Edition:** 2024
- **Pinned toolchain:** `1.97.1` (see [rust-toolchain.toml](rust-toolchain.toml))
- **System dependencies (only for `sentry` feature):** `pkg-config`, `libssl-dev`

## Build, test, and examples

```bash
# Build and test
cargo build                   # default build
cargo build --all-features    # includes sentry
cargo test                    # unit + doc tests
cargo test --all-features     # includes sentry integration tests
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check

# Examples
cargo run --example basic
cargo run --example basic_lif
cargo run --example hebbian_learning
cargo run --example rstdp_demo

# With the optional sentry feature:
SENTRY_DSN=https://...@... cargo run --example sentry --features sentry
```

## Code style

- Formatting: `cargo fmt`
- Linting: `cargo clippy --all-targets --all-features -- -D warnings`
- Comments: only for non-obvious invariants or design rationale
- Unsafe: avoid; use `temp-env` for environment-variable tests
- Pin third-party GitHub Actions to immutable commit SHAs

## Testing notes

- Unit tests: inline `#[cfg(test)]` modules (48 at last count)
- Integration tests: `tests/sentry_integration.rs` (16 tests)
- Doc tests: before committing, run `cargo test` so the crate-level example in `src/lib.rs` runs as a doctest
- Feature matrix: CI runs `cargo hack check --feature-powerset --exclude-no-default-features --keep-going`
- Optional `sentry` feature is off by default; enable with `--features sentry`

## Dev container

Open in VS Code with the Dev Containers extension or run:

```bash
devcontainer up --workspace-folder .
```

The container is `rust:1.97.1-slim-bookworm`. It adds `pkg-config` and `libssl-dev` for `sentry`. `cargo fetch` runs on first create. The `vscode` user owns the toolchain, so `cargo` commands and component installs work from the terminal. `cargo-llvm-cov` is also supported.

## Boundaries

- **Owns:** `src/`, `Cargo.toml`, `README.md`, `AGENTS.md`, review quality gate document, [rust-toolchain.toml](rust-toolchain.toml), `.devcontainer/`
- **Does not own:** sensory encoding (`axon-encoder`), topology/wiring (`synaptic-mesh`), reward shaping (`limbic-critic`), training loops (`plasticity-lab`), IPC (`corpus-ipc`), runtime daemon (`brainstem-daemon`), hardware export (`silicon-bridge`, `Spikenaut-Hardware`)
- **Off-limits:** no mining/trading/high-frequency trading (HFT)/crypto domain logic; no async, networking, or hardware-specific code in the core library

## PR conventions

- Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`
- One concern per PR
- Breaking public API changes bump the minor version for pre-1.0 (`0.X.0` -> `0.(X+1).0`) and update the README migration notes
- Resolve all CI checks and review threads before merge unless a maintainer explicitly approves an override
