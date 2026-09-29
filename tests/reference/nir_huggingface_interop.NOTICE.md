# Hugging Face NIR -> neuromod interop fixtures — provenance and attribution

The two `.nir` files vendored under
`tests/reference/nir_huggingface/fixtures/` are **Hugging Face-derived NIR
graphs** used by neuromod's offline NIR -> neuromod interoperability smoke
harness (`tests/reference/nir_huggingface/`). They carry **zero runtime or
network dependency on Hugging Face**: the files are frozen into the repository
and the harness that loads them is a standalone crate that is **not** part of
the neuromod build, test, or published-package paths.

CI performs **no Hugging Face Hub downloads**. Loading is fully offline from the
vendored files; the harness resolves them from `CARGO_MANIFEST_DIR`.

## Upstream `nir-rs` provenance

These fixtures were introduced upstream in the `nir-rs` project and are consumed
here at the crates.io release `nir-rs = "=0.4.4"` (the harness does **not**
depend on any local `nir-rs` path).

- **Repository:** <https://github.com/Limen-Neural/nir-rs>
- **Fixture work:** issue [#44](https://github.com/Limen-Neural/nir-rs/issues/44)
  and PR [#49](https://github.com/Limen-Neural/nir-rs/pull/49) — **complete /
  merged**. The two fixtures are vendored, checksummed, documented, and
  exercised by `nir-rs` CI.

## Hugging Face source checkpoints (pinned revisions)

The `.nir` files are **new artifacts**, produced by converting public Hugging
Face PyTorch SNN checkpoints with the official Python
[`nir`](https://pypi.org/project/nir/) writer (`nir.write`). They are **not**
byte-identical Hub downloads. Trained weights are copied from the pinned
revisions; graph topology follows the NeuroCUDA hub architectures.

| Fixture | HF repository | HF revision | Architecture |
|---------|---------------|-------------|--------------|
| `neurocuda_mlp_mnist.nir` | [`Krishnav1234/neurocuda-mlp-mnist-snn`](https://huggingface.co/Krishnav1234/neurocuda-mlp-mnist-snn) | `5a2422453d0a1672f5d1dec2ea73d54196a07d85` | MLP 784->256 IF ->256 IF ->10 Affine |
| `neurocuda_cnn_nmnist.nir` | [`Krishnav1234/neurocuda-cnn-nmnist-snn`](https://huggingface.co/Krishnav1234/neurocuda-cnn-nmnist-snn) | `1ee6ba2f500a4584ef05a77b44016beea2491879` | CNN 2x34x34 Conv/IF/AvgPool x3 -> Flatten -> Affine |

## Converter

- **Writer package:** Python `nir` (neuromorphs/NIR)
- **Writer version:** **1.0.8** (`nir.write`, gzip)
- **Embedded `/version`:** `1.0.8`, matching `nir_rs::io::DEFAULT_NIR_VERSION`.

## SHA-256 checksums

The harness asserts these in `vendored_fixtures_match_expected_sha256`
(paths resolved from `CARGO_MANIFEST_DIR`):

```text
fc0b1a1e0c4caeb9d1f7be8700de0212a76ec5f441cae13038887411fd9a1ef0  neurocuda_mlp_mnist.nir
972b45984094606b83b5b19173a653524550a5aa416f8a46398baa4250c34f2f  neurocuda_cnn_nmnist.nir
```

## License / attribution

The Hugging Face model cards for both pinned checkpoints declare **MIT**. The
converted `.nir` files contain those trained weights and are therefore also
distributed under MIT.

> **SPDX-License-Identifier: MIT**
>
> Copyright (c) 2026 NeuroCUDA.

[`tests/reference/nir_huggingface/LICENSE-NeuroCUDA`](nir_huggingface/LICENSE-NeuroCUDA)
reproduces the copyright and permission notice from the NeuroCUDA source
repository; that notice applies to both converted checkpoint fixtures. neuromod
itself is licensed `MIT OR Apache-2.0`; the two are compatible.

## Scope

This is a **load / inspect / handoff** smoke, **not** a bit-identical oracle
against the original NeuroCUDA forward pass. NIR graph scheduling, the Affine /
Conv2d / AvgPool2d / Flatten kernels, and graph wiring stay **downstream** in
this harness (or a future adapter crate) and are never added to `neuromod`.
