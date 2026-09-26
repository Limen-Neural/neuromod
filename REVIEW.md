# Local Review Quality Gate

Run these commands before claiming a PR is ready, especially when touching `src/`, `Cargo.toml`, public APIs, or CI.

## When to run

- Before every push that changes `src/`, `benches/`, `examples/`, `tests/`, or `Cargo.toml`
- After resolving merges with `main`
- Before requesting review or merge

## MSRV pin rule

`Cargo.toml` `rust-version`, `rust-toolchain.toml` `channel`, and the toolchain string in `.github/workflows/ci.yml` must stay **identical**. CI enforces this; do not bump one without the others.

## Mandatory commands

```bash
# Formatting (exit 0 with no output means clean)
cargo fmt --check

# Lint
cargo clippy --all-targets --all-features -- -D warnings

# Build
cargo build --all-features

# Tests (unit + doctests)
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

# Outsider crates.io-only demo (detached workspace; GH#82)
# Must resolve neuromod from the registry, not this path crate.
( cd examples/crates-io-standalone && cargo run )


# Release-mode smoke
cargo run --example basic --release
cargo run --example basic_lif --release
cargo run --example hebbian_learning --release
cargo run --example rstdp_demo --release
( cd examples/crates-io-standalone && cargo run --release )
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

## Long-horizon soak tests

The always-on 10,000-step variants run with the normal test suite. Run the
explicitly ignored million-step engine and sparse GIF layer gates with:

```bash
cargo test soak -- --ignored --nocapture
```

Both variants use fixed topology and deterministic binary (`0` or `1`) inputs.
They assert exact counters and finite numeric state. The engine test starts with
nonzero weights, exercises reward-modulated learning and normalization, and
checks the documented default-bounds precedence contract by requiring each
neuron's weight L1 sum to remain within `1e-4` of the `2.0` budget.

The capacity checks cover persistent allocations only. For `SpikingNetwork`
these are both neuron banks, input spike times, predictive state, and every LIF
neuron's weights and eligibility traces. For `SparseGifHiddenLayer` they are the
CSR offsets, sources, and weights plus membrane, adaptation, and spike-time
banks. Temporary per-step outputs are deliberately excluded.

On Linux, `NEUROMOD_SOAK_RSS=1 cargo test soak -- --ignored --nocapture` prints
best-effort `/proc/self/status` RSS samples. RSS is diagnostic only and is never
asserted because allocator and operating-system behavior is not portable.

Recorded locally on 2026-09-22 with rustc 1.98.1 in the repository dev profile:

| Variant | 10,000 steps | 1,000,000 steps |
| --- | ---: | ---: |
| `SpikingNetwork` | 69 ms | 7.15 s |
| `SparseGifHiddenLayer` | 1.59 ms | 110 ms |

These correctness tests are distinct from `benches/memory_bench.rs`: that
Criterion suite measures fixed object layout and construction/allocation
latency, not persistent capacity growth across steps or process RSS.

## Docs and domain hygiene

```bash
# Build docs; then confirm they remain domain-agnostic
cargo doc --all-features --no-deps
! grep -riE 'spikenaut|\bhft\b|\bmining\b|\bcrypto\b|eagle-lander' target/doc/neuromod/
```

## Regression guards

Verify the core public API surface has not been silently removed:

```bash
grep -R 'pub struct SpikingNetwork\|pub enum StepError' src/
grep -q 'pub fn step_with_rng' src/ \
  && grep -q 'pub fn encode_with_rng' src/
grep -R 'pub struct LifNeuron\|pub struct GifNeuron\|pub struct IzhikevichNeuron\|pub struct LapicqueNeuron\|pub struct FitzHughNagumoNeuron\|pub struct HodgkinHuxleyNeuron' src/
grep -R 'pub struct NeuroModulators\|pub struct SignalProfile\|pub struct Observation' src/
grep -R 'pub trait GenericReward\|pub struct UnitReward' src/
grep -R 'pub fn apply_classical_stdp\|pub fn apply_neuromodulation' src/
grep -R 'pub struct EligibilityTrace\|pub struct RmStdpConfig' src/
```

R-STDP must stay **wired into the engine**, not merely exported (GH#72; see
[ADR 002](docs/adr/002-wire-eligibility-traces.md)). These guard against a refactor that
quietly reverts to an inline rule and leaves the eligibility types decorative:

Chained with `&&` so the block exits non-zero on the **first** missing invariant. Run as
separate commands, an early failure would be masked by a later success and the guard would
report a pass on a half-unwired engine. Decay and accumulation are checked separately rather
than as one alternation, so dropping either one fails the guard.

```bash
grep -q 'pub eligibility: Vec<EligibilityTrace>' src/lif.rs \
  && grep -q 'pub stdp_config: RmStdpConfig' src/engine.rs \
  && grep -q 'trace.decay()' src/engine.rs \
  && grep -q 'trace.accumulate(' src/engine.rs \
  && grep -q 'pub fn set_rm_stdp_config' src/engine.rs \
  && echo "ok: R-STDP still wired into the engine"
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
- `cargo test --all-features` reports all unit tests and doctests passing
- Examples run without panic
- `cargo doc` domain-agnostic grep finds no forbidden terms in `target/doc/neuromod/`
- `git diff origin/main...HEAD` contains only intentional changes
