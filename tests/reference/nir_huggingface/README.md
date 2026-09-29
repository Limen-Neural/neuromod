# `nir_huggingface_interop` — offline NIR -> neuromod interop smoke

A **test-only, boundary-isolated** harness crate that loads two Hugging
Face-derived NIR graphs from disk and maps their integrate-and-fire (`IF`)
neurons onto neuromod's `LapicqueNeuron`. It exists to smoke-test that
HF-derived NIR models can be *received* by neuromod, not to re-implement a NIR
runtime.

This crate is **not** part of the `neuromod` crate. It has its own empty
`[workspace]` table (making it a workspace root of its own) and is listed in
neuromod's `[package].exclude` and `[workspace].exclude`, exactly like the
sibling `tests/reference/corinth_gif/` generator. `neuromod` gains **no**
`nir-rs` / `hdf5` dependency and no import, tensor-math, or graph-runtime code.

Provenance, pinned Hugging Face revisions, converter version, SHA-256 checksums,
and license/attribution for the two vendored `.nir` files live in the sibling
[`../nir_huggingface_interop.NOTICE.md`](../nir_huggingface_interop.NOTICE.md).

## Adapter boundary

This harness owns everything on the NIR side of the boundary; `neuromod` stays a
pure computation library (see [`docs/neuromod-boundary-matrix.md`](../../../docs/neuromod-boundary-matrix.md)).

| Responsibility | Owner |
|----------------|-------|
| Read `.nir` (HDF5) and structurally validate | this harness (via `nir-rs`) |
| Classify every node into a role | this harness |
| Read / flatten tensors, map `IF` -> `LapicqueNeuron` bank | this harness |
| Forward-Euler `integrate` + spike/reset per step | `neuromod::LapicqueNeuron` (called by this harness) |
| NIR graph **scheduling** (edge order, timing) | **downstream** (not implemented here) |
| `Affine` / `Conv2d` kernels, `AvgPool2d` pooling, `Flatten` reshape | **downstream** (not implemented here) |
| Graph **wiring** (routing tensors between nodes) | **downstream** (not implemented here) |

Only the `IF` neuron is mapped into neuromod. Every other node is *classified*
(so nothing is silently dropped) but never *executed* by this harness.

## IF mapping conventions (Assumption 5)

A NIR `IF` node is a non-leaky integrate-and-fire neuron with a resistance `r`
and firing threshold `v_threshold` (and an optional `v_reset`). It is mapped
element-wise onto a bank of `LapicqueNeuron`s under these fixed conventions:

- **Integration:** forward Euler, `dt = 1`.
- **Stimulus:** element `i` receives `stimulus = r[i] * I[i]`, where `I` is the
  per-step input current.
- **Initial state:** membrane potential `v0 = 0`.
- **Spiking:** the neuron fires when `v >= threshold`.
- **Reset:** hard reset to `v = 0` on spike. An absent `v_reset` is treated as
  `0`; a **non-zero** `v_reset` is rejected as an unsupported mapping.

neuromod's `LapicqueNeuron::integrate` applies `v <- (v + stimulus) * (1 - decay_rate)`.
Setting `decay_rate = 0` reduces this to exactly `v += stimulus`, which is a pure
(non-leaky) integrate-and-fire step — precisely the NIR `IF` semantics (`IF` has
no `tau`, so there is no leak). `threshold` and `base_threshold` are set to
`v_threshold[i]`, and the hard reset is performed by
`LapicqueNeuron::check_for_spike`.

This is an interoperability **smoke**. It checks that HF-derived IF parameters
load and step through the real neuromod path and match this documented
forward-Euler contract. It does **not** claim bit-identical numerical
equivalence with the original NeuroCUDA forward pass.

## Node classification table

Every node in a loaded graph is sorted into exactly one role:

| Wire type(s) | `NodeRole` | Notes |
|--------------|------------|-------|
| `IF` | `SupportedNeuron` | mapped into a `LapicqueNeuron` bank |
| `Input`, `Output` | `GraphBoundary` | virtual graph ports |
| `Affine`, `Conv2d` | `AdapterConcern { kind: TensorKernel }` | downstream tensor kernel |
| `AvgPool2d` | `AdapterConcern { kind: Pooling }` | downstream pooling |
| `Flatten` | `AdapterConcern { kind: Reshape }` | downstream reshape |
| any other | `Unsupported(type_name)` | catch-all; carries the wire `type` |

The two vendored fixtures classify with **no** `Unsupported` node:

- **`neurocuda_mlp_mnist.nir`** — 7 nodes / 6 edges: 1 `Input`, 3 `Affine`,
  2 `IF`, 1 `Output`.
- **`neurocuda_cnn_nmnist.nir`** — 13 nodes / 12 edges: 1 `Input`, 3 `Conv2d`,
  3 `IF`, 3 `AvgPool2d`, 1 `Flatten`, 1 `Affine`, 1 `Output`.

## Failure taxonomy

Handoff failures are split into three disjoint `HandoffError` kinds so tests can
assert *why* a handoff failed:

- **`Load`** — reading or structurally validating a `.nir` file failed.
- **`UnsupportedMapping`** — the graph loaded but a node cannot be faithfully
  represented (shape disagreement, non-finite / non-positive threshold, non-zero
  reset, …).
- **`Runtime`** — a mapped IF bank misbehaved while stepping (wrong input
  length, a non-finite value classified via `neuromod::NonFiniteClass`, or a
  spiking element that did not hard-reset to `0`).

## Running the smoke locally

```sh
cargo test --locked --manifest-path tests/reference/nir_huggingface/Cargo.toml
```

The harness reads `.nir` (HDF5) files through `nir-rs`'s `hdf5` feature. It
depends on `hdf5-metno` with `static` + `zlib` so a **vendored** libhdf5 is
compiled from source (no system libhdf5 required); the first build compiles that
C library and is minutes-long. That vendored build needs **CMake >= 3.26**. If
your system `cmake` is older (some sandboxes ship 3.22), install a newer one and
put it first on `PATH` for the harness `cargo` commands, e.g.:

```sh
pip install 'cmake>=3.29,<4'
export PATH="$(python -c 'import cmake,os;print(os.path.join(os.path.dirname(cmake.__file__),"data","bin"))'):$PATH"
cargo test --locked --manifest-path tests/reference/nir_huggingface/Cargo.toml
```

This CMake step is a **local** convenience only. In CI the Ubuntu runner uses
its bundled CMake and installs **nothing**: the vendored static libhdf5 build
keeps the step hermetic (see the "NIR HuggingFace interop smoke" step in
[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)). No Hugging Face
Hub download happens in either case: loading is fully offline from the vendored
fixtures, resolved from `CARGO_MANIFEST_DIR`.

## What stays downstream

NIR graph **scheduling**, the `Affine` / `Conv2d` / `AvgPool2d` / `Flatten`
kernels, and graph **wiring** are deliberately **not** implemented here or in
`neuromod`. They belong to a downstream NIR runtime/adapter. This harness only
loads, classifies, and hands off the `IF` neurons.
