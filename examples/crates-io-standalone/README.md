# crates.io standalone demo

Minimal outsider onboarding binary: depends on published [`neuromod`](https://crates.io/crates/neuromod) from crates.io only. No git path, no sibling Limen crates.

This package is a **separate Cargo workspace** so it cannot accidentally resolve the parent path crate.

## Run

From this directory:

```bash
cargo run
```

Copy the `Cargo.toml` + `src/main.rs` into any empty cargo binary to reproduce outside this repository.

## Version pin

`neuromod = "0.5"` intentionally resolves the published 0.5.x line. This demo is a registry-only compatibility smoke and does not validate the in-tree 0.6.0 candidate (including R-STDP and the sparse GIF layer). Check [crates.io](https://crates.io/crates/neuromod) for registry availability. Before maintainer publication, in-repo examples such as `rstdp_demo` and `sparse_gif_layer` use the local crate and are not crates.io-only.
