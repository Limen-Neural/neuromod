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

`neuromod = "0.5"` tracks the latest **published** 0.5.x release. The in-tree library is 0.6.0 (R-STDP, sparse GIF layer) and is not on crates.io until that tag is published. In-repo examples such as `rstdp_demo` and `sparse_gif_layer` use the local crate and are not crates.io-only.
