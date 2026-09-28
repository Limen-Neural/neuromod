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

The full 512-frame output raster is **not** dumped; per-step spike **counts**
are recorded for all 512 steps, and fired IDs are recorded at the selected steps
above. The complete raster is reconstructed by replaying the fixture through
neuromod (FEAT-002).

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
cargo run --release > ../corinth_gif_parity.json
```

The generator is fully deterministic: running it twice produces byte-identical
output. It depends only on `serde_json` and resolves offline against a populated
Cargo registry cache. `sha2` was not added as a dependency (not present in the
offline cache); the funnel SHA-256 above is verified out-of-band with
`sha256sum` and pinned in both this NOTICE and the generator source.
