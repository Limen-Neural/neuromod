// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// VENDORED, UNMODIFIED-ARITHMETIC copy of the `SparseGifHiddenLayer` dynamics
// from the pinned `rmems/corinth-canal` revision
// `8e54e234ac005dd84e4ad2bedbf9f5bceb082355`, file `src/funnel.rs`
// (SHA-256 of the full original file:
//  10192537a1a096fc8ec8a9a87740b643624b64c9fc3f3408faf06653b6694b47).
//
// Copyright (c) 2026 Raul Montoya Cardenas and contributors.
//
// Only the `SparseGifHiddenLayer` structure-generation (`new`) and step loop
// (`run`) needed to reproduce the audited 512-step parity case are vendored
// here. The upstream telemetry/encoder/bridge orchestration
// (`TelemetryFunnel`, `SignedSplitBankBridge`, `TelemetryEncoder`,
// `active_neuron_indices`, the `#[cfg(test)]` module) is intentionally omitted:
// it is not on the audited dynamics path and would drag in `crate::telemetry`
// and `crate::types`, which are not offline-viable.
//
// The arithmetic of `new()` and `run()` is byte-for-byte identical to the
// pinned source. The ONLY changes are additive, read-only accessors marked
// `// PATCH:` below, which expose already-computed state for serialization and
// change no computation. See `corinth_gif_parity.NOTICE.md` for the exact patch.

pub const FUNNEL_INPUT_NEURONS: usize = 2048;
pub const FUNNEL_HIDDEN_NEURONS: usize = 2048;
pub const GIF_FAN_IN: usize = 4;
const GIF_IZ_NEURONS: usize = 5;

#[derive(Debug, Clone)]
pub struct SparseGifHiddenLayer {
    weight_indices: Vec<[usize; GIF_FAN_IN]>,
    weight_values: Vec<[f32; GIF_FAN_IN]>,
    membrane: Vec<f32>,
    adaptation: Vec<f32>,
    leak: f32,
    drive_scale: f32,
    threshold_base: f32,
    adaptation_scale: f32,
    adaptation_decay: f32,
    reset_ratio: f32,
}

impl SparseGifHiddenLayer {
    pub fn new() -> Self {
        let mut weight_indices = Vec::with_capacity(FUNNEL_HIDDEN_NEURONS);
        let mut weight_values = Vec::with_capacity(FUNNEL_HIDDEN_NEURONS);

        for hidden in 0..FUNNEL_HIDDEN_NEURONS {
            let tuned_negative = hidden % 2 == 1;
            let mut indices = [0usize; GIF_FAN_IN];
            let mut values = [0.0f32; GIF_FAN_IN];
            let mut cursor = (hidden * 11 + 3) % FUNNEL_INPUT_NEURONS;

            for edge in 0..GIF_FAN_IN {
                while indices[..edge].contains(&cursor) {
                    cursor = (cursor + 5) % FUNNEL_INPUT_NEURONS;
                }

                indices[edge] = cursor;

                let positive_bank = cursor % 4 < 2;
                let preference = if tuned_negative {
                    if positive_bank { -1.0 } else { 1.0 }
                } else if positive_bank {
                    1.0
                } else {
                    -1.0
                };
                let phase = ((hidden * 37 + edge * 19 + cursor * 13) % 97) as f32 / 96.0;
                values[edge] = preference * (0.35 + phase * 0.4);
                cursor = (cursor + 7 + hidden % 3) % FUNNEL_INPUT_NEURONS;
            }

            weight_indices.push(indices);
            weight_values.push(values);
        }

        Self {
            weight_indices,
            weight_values,
            membrane: vec![0.0; FUNNEL_HIDDEN_NEURONS],
            adaptation: vec![0.0; FUNNEL_HIDDEN_NEURONS],
            leak: 0.92,
            drive_scale: 0.75,
            threshold_base: 0.65,
            adaptation_scale: 0.22,
            adaptation_decay: 0.94,
            reset_ratio: 0.35,
        }
    }

    pub fn run(
        &mut self,
        input_spike_train: &[Vec<usize>],
    ) -> (Vec<Vec<usize>>, Vec<f32>, Vec<f32>) {
        let mut spike_train = Vec::with_capacity(input_spike_train.len());
        let mut active = [false; FUNNEL_INPUT_NEURONS];

        for step in input_spike_train {
            active.fill(false);
            for &idx in step {
                if idx < FUNNEL_INPUT_NEURONS {
                    active[idx] = true;
                }
            }

            let mut step_spikes = Vec::new();
            for hidden in 0..FUNNEL_HIDDEN_NEURONS {
                self.adaptation[hidden] *= self.adaptation_decay;

                let mut drive = 0.0f32;
                let indices = &self.weight_indices[hidden];
                let values = &self.weight_values[hidden];
                for edge in 0..GIF_FAN_IN {
                    if active[indices[edge]] {
                        drive += values[edge];
                    }
                }

                self.membrane[hidden] = self.membrane[hidden] * self.leak
                    + drive * self.drive_scale
                    - self.adaptation[hidden] * 0.05;

                let threshold =
                    self.threshold_base + self.adaptation[hidden] * self.adaptation_scale;
                if self.membrane[hidden] >= threshold {
                    step_spikes.push(hidden);
                    self.membrane[hidden] -= threshold * self.reset_ratio;
                    self.adaptation[hidden] += 1.0;
                }
            }

            spike_train.push(step_spikes);
        }

        let potentials = self
            .membrane
            .iter()
            .map(|value| (value / (self.threshold_base * 2.0)).clamp(0.0, 1.0))
            .collect();

        (spike_train, potentials, vec![0.0; GIF_IZ_NEURONS])
    }

    // ---------------------------------------------------------------------
    // PATCH: read-only accessors (additive; no arithmetic changed)
    //
    // The upstream struct keeps every field private and exposes only
    // `run`/`reset`/`state_activity`. The parity generator needs to read the
    // final membrane/adaptation banks and the per-neuron fan-in in the exact
    // edge order the `run` loop consumes it. These accessors return references
    // to already-computed state; they perform no computation of their own.
    // ---------------------------------------------------------------------

    /// PATCH: final membrane potentials, indexed by neuron.
    pub fn membrane(&self) -> &[f32] {
        &self.membrane
    }

    /// PATCH: final adaptation variables, indexed by neuron.
    pub fn adaptation(&self) -> &[f32] {
        &self.adaptation
    }

    /// PATCH: per-neuron fan-in source indices in Corinth EDGE ORDER
    /// (`indices[0..GIF_FAN_IN]`, i.e. the exact order `run` reads them —
    /// NOT sorted).
    pub fn weight_indices(&self, hidden: usize) -> &[usize; GIF_FAN_IN] {
        &self.weight_indices[hidden]
    }

    /// PATCH: per-neuron fan-in weights in the same edge order as
    /// [`Self::weight_indices`].
    pub fn weight_values(&self, hidden: usize) -> &[f32; GIF_FAN_IN] {
        &self.weight_values[hidden]
    }
}

impl Default for SparseGifHiddenLayer {
    fn default() -> Self {
        Self::new()
    }
}
