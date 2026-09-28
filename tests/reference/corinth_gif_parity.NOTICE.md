# Corinth GIF parity fixture — provenance and attribution

`corinth_gif_parity.json` is a bit-exact reference fixture captured from the
**upstream Corinth GIF hidden-layer dynamics**, for use by neuromod's offline
`SparseGifHiddenLayer` parity test. It carries **zero runtime or network
dependency on Corinth**: the values are frozen into JSON and the generator that
produced them (`tests/reference/corinth_gif/`) is a standalone crate that is
**not** part of the neuromod build, test, or published-package paths.

## Upstream source

- **Repository:** <https://github.com/rmems/corinth-canal>
- **Pinned commit:** `8e54e234ac005dd84e4ad2bedbf9f5bceb082355`
- **File:** `src/funnel.rs` (`SparseGifHiddenLayer`)
- **`src/funnel.rs` SHA-256 (unmodified, pinned revision):**
  `10192537a1a096fc8ec8a9a87740b643624b64c9fc3f3408faf06653b6694b47`

  Reproduce the hash:

  ```sh
  git -C <corinth-canal> show \
    8e54e234ac005dd84e4ad2bedbf9f5bceb082355:src/funnel.rs | sha256sum
  ```

- **Historical audited neuromod commit** (the neuromod revision whose GIF layer
  was verified spike- and bit-identical against this Corinth source under
  matched explicit topology and dense 0/1 frame conversion):
  `263ec19e807454eac943681993623989fb986cc6`.

## License / attribution

Upstream `corinth-canal` is dual-licensed:

> **SPDX-License-Identifier: Apache-2.0 OR MIT**
>
> Copyright (c) 2026 Raul Montoya Cardenas and contributors.

The vendored dynamics in `tests/reference/corinth_gif/src/funnel_vendored.rs`
retain that SPDX header. neuromod itself is licensed `MIT OR Apache-2.0`; the
two are compatible.

## What was vendored, and the exact accessor patch

The generator does **not** depend on the whole `corinth-canal` crate (that pulls
in CUDA/`cust`, `sentry`, `rustls`, `memmap2`, etc., which are not offline
viable). It vendors **only** the `SparseGifHiddenLayer` structure generation
(`new()`) and step loop (`run()`) from the pinned `src/funnel.rs`. The upstream
telemetry/encoder/bridge orchestration (`TelemetryFunnel`,
`SignedSplitBankBridge`, `TelemetryEncoder`, `active_neuron_indices`, and the
`#[cfg(test)]` module) is intentionally omitted; it is not on the audited
dynamics path.

**The arithmetic of `new()` and `run()` is byte-for-byte identical to the pinned
source.** The only changes are **additive, read-only accessors** that expose
already-computed state for serialization. They perform no computation and change
no field, no loop, and no numeric expression. The exact patch, relative to the
upstream `impl SparseGifHiddenLayer`:

```diff
     pub fn run(
         &mut self,
         input_spike_train: &[Vec<usize>],
     ) -> (Vec<Vec<usize>>, Vec<f32>, Vec<f32>) {
         // ... unchanged upstream body ...
     }

-    pub fn reset(&mut self) {
-        self.membrane.fill(0.0);
-        self.adaptation.fill(0.0);
-    }
-
-    pub fn state_activity(&self) -> bool {
-        self.membrane.iter().any(|value| value.abs() > 1e-6)
-            || self.adaptation.iter().any(|value| value.abs() > 1e-6)
-    }
+    // PATCH: read-only accessors (additive; no arithmetic changed).
+    // `reset`/`state_activity` are dropped (unused by the generator);
+    // the four accessors below expose already-computed state.
+
+    /// PATCH: final membrane potentials, indexed by neuron.
+    pub fn membrane(&self) -> &[f32] {
+        &self.membrane
+    }
+
+    /// PATCH: final adaptation variables, indexed by neuron.
+    pub fn adaptation(&self) -> &[f32] {
+        &self.adaptation
+    }
+
+    /// PATCH: per-neuron fan-in source indices in Corinth EDGE ORDER
+    /// (`indices[0..GIF_FAN_IN]`, the exact order `run` reads them — NOT sorted).
+    pub fn weight_indices(&self, hidden: usize) -> &[usize; GIF_FAN_IN] {
+        &self.weight_indices[hidden]
+    }
+
+    /// PATCH: per-neuron fan-in weights in the same edge order.
+    pub fn weight_values(&self, hidden: usize) -> &[f32; GIF_FAN_IN] {
+        &self.weight_values[hidden]
+    }
```

`GIF_FAN_IN` is also made `pub` (upstream `const GIF_FAN_IN: usize = 4;` → `pub
const GIF_FAN_IN: usize = 4;`) so the generator can size the exported rows. This
too changes no arithmetic.

## The audited case

- **512** steps × **2048** hidden neurons, **2048** input channels, fan-in
  **4**.
- Total output spikes across the run: **8357** (recorded in `counts.total_spikes`).

### Deterministic input rule

The 512-step input spike train is a fixed integer congruence with no RNG, wall
clock, or hashing:

> Input index `i` is **active** at step `t` iff `(i * 31 + t * 17) % 23 == 0`.

Because the rule is periodic mod 23, the 512 per-step input masks deduplicate to
**23** unique masks. The fixture stores those 23 masks once
(`input_masks_hex`, each a 512-hex-char little-endian bit-string over 2048
channels: channel `i` → byte `i / 8`, bit `i % 8`) plus a `per_step_mask_index`
mapping each of the 512 steps to its mask.

### Deterministic selected-step rule

Raster rows (fired IDs at chosen steps) are exported for:

> every 64th step (`0, 64, 128, 192, 256, 320, 384, 448`) plus the final step
> (`511`).

Fired IDs are recorded for **every** step in `per_step_fired_ids` (a per-step
ascending list of hidden-neuron IDs), so the parity test can verify the exact
spike ID on all 512 steps, not merely the spike count. Per-step spike **counts**
are also recorded for all 512 steps, and the selected steps above are retained
as `selected_raster_rows` (a documented subset that must agree with
`per_step_fired_ids` at those steps). `per_step_fired_ids` is the exhaustive
oracle and is internally consistent with both `per_step_spike_count` and
`selected_raster_rows`.

## Parity ordering requirement (why edge order matters)

neuromod's `SparseGifHiddenLayer::step_into` accumulates each neuron's drive as
a **fixed-order f32 sum over its fan-in row, in row order**. f32 addition is not
associative, so the row order is part of the numeric contract. Corinth sums over
edges `0..4` in its `indices` array order, which is **distinct but not sorted**
(the generator's `cursor` wraps around the input space). Therefore
`topology_edge_order[n]` records each neuron's fan-in **in Corinth edge order**,
not sorted. The neuromod parity test must rebuild topology via
`SparseGifHiddenLayer::from_topology(num_inputs, GifParams::default(), &rows)`,
which preserves the given row order, to obtain bit-identical results.

`topology_edge_order[n]` is a flat integer array
`[src0, wbits0, src1, wbits1, ...]` of `fan_in` `(source, weight_bits)` pairs in
that order; `weight_bits` are the f32 weights via `f32::to_bits()`.

## Parameter bit patterns

`param_bits` records all 8 GIF defaults as `f32::to_bits()`. They match
neuromod `GifParams::default()` bit-for-bit (`GIF_LEAK 0.92`,
`GIF_DRIVE_SCALE 0.75`, `GIF_BASE_THRESHOLD 0.65`, `GIF_ADAPTATION_SCALE 0.22`,
`GIF_ADAPTATION_DECAY 0.94`, `GIF_ADAPTATION_COUPLING 0.05`,
`GIF_ADAPTATION_INCREMENT 1.0`, `GIF_RESET_RATIO 0.35`). `final_membrane_bits`
and `final_adaptation_bits` hold the end-of-run state for all 2048 neurons, also
as `f32::to_bits()`.

## Regeneration

The generator is a standalone crate with its own empty `[workspace]` table, so
`cargo` invoked inside neuromod never absorbs it. From the neuromod repo root:

```sh
cd tests/reference/corinth_gif
cargo run --locked --offline --release > ../corinth_gif_parity.json
```

The generator's `Cargo.lock` pins its dependency graph. The CI workflow runs
the generator in offline mode and compares its output byte-for-byte with the
committed fixture. Running it twice produces byte-identical output.

### Built-in provenance self-check

Before emitting the fixture, the generator verifies the vendored dynamics
against the pinned upstream source and aborts (nonzero exit, no stdout) on any
drift:

1. **Upstream identity.** It embeds a verbatim, byte-identical copy of the
   pinned upstream `src/funnel.rs`
   (`tests/reference/corinth_gif/src/funnel_upstream_pinned.rs.txt`, included as
   raw text so it is never compiled) and recomputes its SHA-256 with the
   generator's small self-contained implementation, checked against standard
   SHA-256 test vectors. That digest must equal `CORINTH_FUNNEL_RS_SHA256`
   (`10192537…`), which is the hash of the **unmodified whole** upstream file.
   Refresh the embedded copy with
   `git -C <corinth-canal> show 8e54e234ac005dd84e4ad2bedbf9f5bceb082355:src/funnel.rs`;
   never edit the recorded hash unless the pinned commit is being re-audited.
2. **Vendored fidelity.** Because `funnel_vendored.rs` carries local, non-
   arithmetic additions (SPDX header, read-only accessors, `pub const`), it
   cannot reproduce that whole-file hash. The generator instead extracts the
   **complete** `new()` and `run()` function bodies (the contiguous source spans
   that carry the audited GIF arithmetic) out of the SHA-verified upstream text
   (via a deterministic balanced-brace scan anchored inside the
   `impl SparseGifHiddenLayer` block) and asserts each full body appears
   character-for-character inside `funnel_vendored.rs`. Because the upstream text
   is already proven authentic by check (1), proving the vendored file contains
   those full bodies verbatim establishes that every character of the vendored
   audited arithmetic came from the pinned upstream. Any drift anywhere inside
   `new()`/`run()`, not just at a couple of anchor lines, is therefore caught
   even though the vendored file is not byte-identical to the whole upstream.

The generator depends only on `serde_json`, which is already a neuromod
development dependency; provenance hashing adds no external crate. The check
does **not** re-run upstream Corinth (which pulls in CUDA/`sentry`/`rustls` and
is not offline-viable); it verifies the pinned bytes and that the vendored
arithmetic is a faithful excerpt of them.

## Static analysis scope

`tests/reference/corinth_gif/` is a standalone, workspace- and package-excluded
dev tool whose whole purpose is to vendor a **byte-/arithmetic-identical** copy
of the pinned upstream `funnel.rs` (`funnel_vendored.rs`). That byte identity is
the provenance guarantee, so the vendored copy must never be restructured to
satisfy a complexity or duplication metric. The directory is therefore excluded
from the repository's static-analysis tools that support in-repo exclusion:

- **Codacy** — added to `exclude_paths` in `.codacy.yml`.
- **DeepSource** — added to `exclude_patterns` in `.deepsource.toml`.

**CodeScene** has no documented in-repo file that excludes content from
analysis; its file/content exclusion is a project-level (dashboard) setting
("Specify the content to exclude from your analysis"). Rather than commit a
config file that would silently do nothing, the exclusion of
`tests/reference/corinth_gif/` from CodeScene must be applied maintainer-side in
the CodeScene project configuration. Until then, the remaining CodeScene "Code
Health" flags on `funnel_vendored.rs` (`SparseGifHiddenLayer::new`/`run`: Bumpy
Road / Deep Nested Complexity) are inherent to the verbatim vendored upstream
copy and cannot be resolved in-repo without breaking the provenance guarantee;
they require a maintainer-side CodeScene suppression, not a code change.
