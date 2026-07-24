# Local Review Quality Gate

Run these commands before claiming a PR is ready, especially when touching `src/`, `Cargo.toml`, public APIs, or CI.

## When to run

- Before every push that changes `src/`, `benches/`, `examples/`, `tests/`, or `Cargo.toml`
- After resolving merges with `main`
- Before requesting review or merge

## Mandatory commands

```bash
# Formatting (exit 0 with no output means clean)
cargo fmt --check

# Lint
cargo clippy --all-targets --all-features -- -D warnings

# Build
cargo build --all-features

# Tests (unit + sentry integration + doctests)
cargo test --all-features
```

## Optional CI-equivalent matrix

If the optional tools are not installed:

```bash
cargo install cargo-nextest --locked
cargo install cargo-hack --locked
cargo install cargo-llvm-cov --locked
```

Then run:

```bash
# CI uses nextest for JUnit output and speed
cargo nextest run --all-features --no-fail-fast

# Feature-powerset build check (no running tests)
cargo hack check --feature-powerset --exclude-no-default-features --keep-going

# Coverage (matches coverage.yml)
cargo llvm-cov --all-features --lcov --output-path lcov.info
```

## Examples smoke

```bash
cargo run --example basic
cargo run --example basic_lif
cargo run --example hebbian_learning
cargo run --example rstdp_demo

# Optional sentry example (no DSN needed for compilation smoke)
cargo run --example sentry --features sentry

# Release-mode smoke
cargo run --example basic --release
cargo run --example basic_lif --release
cargo run --example hebbian_learning --release
cargo run --example rstdp_demo --release
cargo run --example sentry --release --features sentry
```

## Benchmarks smoke

```bash
# Compile benchmarks without running long measurements
cargo bench --no-run --all-features

# Benchmarks use harness = false (Criterion's own runner), so run them via
# `cargo bench --bench <name>` in a terminal or a plain Cargo run
# configuration in your IDE -- not a "Run Test" gutter action, which expects
# the structured libtest protocol these targets don't emit.
```

## Docs and domain hygiene

```bash
# Build docs; then confirm they remain domain-agnostic
cargo doc --all-features --no-deps
! grep -riE 'spikenaut|\bhft\b|\bmining\b|\bcrypto\b|eagle-lander' target/doc/neuromod/
```

## Docker smoke

```bash
# Runtime image (example binaries only)
docker build -t neuromod:runtime .
docker run --rm neuromod:runtime ls /usr/local/bin

# Builder stage (has Rust toolchain, runs the test suite)
docker build --target builder -t neuromod:builder .
docker run --rm neuromod:builder cargo test --all-features --quiet
```

## Regression guards

Verify the core public API surface has not been silently removed:

```bash
grep -R 'pub struct SpikingNetwork\|pub enum StepError' src/
grep -R 'pub struct LifNeuron\|pub struct GifNeuron\|pub struct IzhikevichNeuron\|pub struct LapicqueNeuron\|pub struct FitzHughNagumoNeuron\|pub struct HodgkinHuxleyNeuron' src/
grep -R 'pub struct NeuroModulators\|pub struct SignalProfile\|pub struct Observation' src/
grep -R 'pub trait GenericReward\|pub struct UnitReward' src/
grep -R 'pub fn apply_classical_stdp\|pub fn apply_neuromodulation' src/
grep -R 'pub struct EligibilityTrace\|pub struct RmStdpConfig' src/
```

Verify Criterion benchmarks aren't silently reverted to the default libtest harness (causes `cargo bench` to report `running 0 tests` instead of executing benchmarks):

```bash
! grep -n 'harness = true' Cargo.toml
```

## Diff hygiene

```bash
git fetch origin main
git diff --stat origin/main...HEAD

# No IDE or local tooling directories should be tracked
git ls-files .idea .kilo .kilocode .mimocode  # must print nothing
```

## Pass criteria

- `cargo fmt --check` is silent and exits 0
- `cargo clippy --all-targets --all-features -- -D warnings` reports zero warnings
- `cargo build --all-features` succeeds
- `cargo test --all-features` reports all unit tests (48), sentry integration tests (16), and doctests (1) passing
- Examples run without panic
- `cargo doc` domain-agnostic grep finds no forbidden terms in `target/doc/neuromod/`
- Docker builder image compiles and tests pass
- `git diff origin/main...HEAD` contains only intentional changes
