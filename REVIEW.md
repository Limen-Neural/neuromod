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

## Local Qodana (before push)

CLI: `qodana` (see `qodana.yaml`; CI twin is `.github/workflows/qodana_code_quality.yml`).

```bash
qodana scan --project-dir . --linter qodana-rust --print-problems --save-report
```

Optional Cloud upload if `QODANA_TOKEN` is set locally (never commit the token):

```bash
export QODANA_ENDPOINT=https://qodana.cloud
qodana scan --project-dir . --linter qodana-rust --print-problems --save-report
```

Do not push while Qodana reports new actionable defects on `src/`, `examples/`, `benches/`, or `tests/`.

**Known noise (do not block on these alone):**

- `DuplicatedCode` in multi-stage RK4 integrators (`fitzhugh_nagumo`, `hodgkin_huxley`) — intentional stage structure.
- `RsBorrowChecker` / `RsTypeCheck` false positives from incomplete Cargo project load in the Docker linter (host `cargo check` / `clippy -D warnings` is authoritative).
- Prefer fixing real hits: unused deps (`CargoUnusedDependency`), clippy-equivalent nits.

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

Verify Criterion benchmarks aren't silently reverted to the default libtest harness (causes `cargo bench` to report `running 0 tests` instead of executing benchmarks). Every `[[bench]]` must explicitly set `harness = false` — omitting the key is as bad as setting `true`:

```bash
! grep -n 'harness = true' Cargo.toml
# Fail if any [[bench]] lacks an explicit harness = false in the following lines
python3 - <<'PY'
from pathlib import Path
text = Path("Cargo.toml").read_text()
blocks = text.split("[[bench]]")[1:]
assert blocks, "expected at least one [[bench]] target"
for i, block in enumerate(blocks, 1):
    # Only the next table section belongs to this bench target
    section = block.split("\n[")[0]
    has_harness = any(
        line.strip().startswith("harness") and "=" in line and "false" in line
        for line in section.split("\n")
        if not line.strip().startswith("#")
    )
    assert has_harness, f"[[bench]] #{i} missing active harness = false assignment"
print(f"ok: {len(blocks)} [[bench]] targets declare harness = false")
PY
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
