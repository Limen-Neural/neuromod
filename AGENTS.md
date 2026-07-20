# neuromod

A Rust library crate providing biologically grounded spiking neural network (SNN) primitives: neuron models (Lapicque, LIF, GIF, Izhikevich, FitzHugh-Nagumo, Hodgkin-Huxley), a topology-neutral `SpikingNetwork`, generic `NeuroModulators`, and foundational plasticity building blocks (classical STDP, reward-modulated STDP, eligibility traces).

Part of the [Limen-Neural](https://github.com/Limen-Neural) ecosystem. See [`docs/neuromod-boundary-matrix.md`](docs/neuromod-boundary-matrix.md) for ownership boundaries and [`docs/org-modularization.md`](docs/org-modularization.md) for cross-repo conventions.

## Repository map

| Path | Purpose |
|------|---------|
| `src/` | Library code (neuron models, `SpikingNetwork`, neuromodulators, plasticity) |
| `examples/` | Runnable demos (`basic`, `basic_lif`, `hebbian_learning`, `rstdp_demo`, `sentry`) |
| `benches/` | Criterion benchmarks (`neuron_bench`, `stdp_bench`, `memory_bench`, `modulation_bench`) |
| `tests/sentry_integration.rs` | Integration tests covering the optional `sentry` feature |
| `docs/` | Architecture docs: boundary matrix, org modularization index, ADR |
| `rust-toolchain.toml` | Pinned Rust toolchain (1.97.1) |
| `.devcontainer/` | VS Code dev container configuration |
| `REVIEW.md` | Local review quality gate |
| `AGENTS.md` | Agent instructions (this file) |
| `.github/workflows/` | CI/CD pipelines |

## Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| `serde` 1.0 | required | Serialization of core types |
| `serde_json` 1.0 | required | JSON serialization helpers |
| `rand` 0.10 | required | Stochastic elements in neuron models |
| `sentry` 0.48 | optional | Error monitoring (feature-gated) |
| `criterion` 0.8 | dev | Criterion benchmarks |
| `temp-env` 0.3 | dev | Safe environment variable manipulation in tests |

## Toolchain

- **Edition:** 2024
- **Pinned toolchain:** `1.97.1` (see `rust-toolchain.toml`)
- **System dependencies (only for `sentry` feature):** `pkg-config`, `libssl-dev`

## Build & test

```bash
cargo build                   # default build
cargo build --all-features    # includes sentry
cargo test                    # unit + doc tests
cargo test --all-features     # includes sentry integration tests
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Running examples

```bash
cargo run --example basic
cargo run --example basic_lif
cargo run --example hebbian_learning
cargo run --example rstdp_demo

# With the optional sentry feature:
SENTRY_DSN=https://...@... cargo run --example sentry --features sentry
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `sentry` | off | Enables `examples/sentry.rs` and pulls in `sentry` error reporting |

## CI workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | push/PR to `main` | fmt, clippy, build, `cargo nextest`, `cargo-hack` feature matrix, doc domain check |
| `coverage.yml` | push/PR to `main` | `cargo-llvm-cov`, Codecov upload, JUnit via nextest |
| `docker.yml` | push/PR to `main` | Multi-stage Docker build and example binary verification |
| `audit.yml` | push/PR/weekly | `rustsec` audit + Trivy filesystem scan |
| `codeql.yml` | push/PR/weekly | CodeQL static analysis for Rust |
| `reviewdog.yml` | PR to `main` | Inline clippy/rustfmt comments |
| `sentry-release.yml` | tag push `v*` / manual | Creates Sentry release |

## Code style

- **Formatting:** `cargo fmt` (rustfmt)
- **Linting:** `cargo clippy --all-targets --all-features -- -D warnings`
- **Comments:** Avoid comments that restate what the code does. Use comments only for non-obvious invariants or design rationale.
- **Unsafe:** Avoid. Use `temp-env` for environment-variable tests in Edition 2024.
- **Third-party GitHub Actions:** Pin to immutable commit SHAs (Aikido policy already enforced in workflows).

## Testing

- **Unit tests:** Inline `#[cfg(test)]` modules in `src/**/*.rs` (48 tests at last count)
- **Integration tests:** `tests/sentry_integration.rs` (16 tests)
- **Doc tests:** Crate-level example in `src/lib.rs`
- **Feature matrix:** CI runs `cargo hack check --feature-powerset --exclude-no-default-features --keep-going`

## Dev container

Open the repo in VS Code with the Dev Containers extension or use the CLI:

```bash
devcontainer up --workspace-folder .
```

The container is based on `rust:1.97.1-slim-bookworm`, adds `pkg-config` and `libssl-dev` for the `sentry` feature, and runs `cargo fetch` on first create. The non-root `vscode` user owns the Rust toolchain so `cargo` commands, component installs, and the optional `cargo-llvm-cov` workflow work from the terminal.

## Boundaries

- **Owns:** `src/`, `Cargo.toml`, `README.md`, `AGENTS.md`, `REVIEW.md`, `rust-toolchain.toml`, `.devcontainer/`
- **Does not own:** sensory encoding (`axon-encoder`), topology/wiring (`synaptic-mesh`), reward shaping (`limbic-critic`), training loops (`plasticity-lab`), IPC (`corpus-ipc`), runtime daemon (`brainstem-daemon`), hardware export (`silicon-bridge`, `Spikenaut-Hardware`)
- **Off-limits:** no mining/trading/HFT/crypto domain logic; no async, networking, or hardware-specific code in the core library

## PR conventions

- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`
- Keep changes scoped to one concern per PR.
- Breaking public API changes bump the minor version for pre-1.0 (`0.X.0` → `0.(X+1).0`) and update the README migration notes.
- All CI checks must pass and all review threads must be resolved before merge.
