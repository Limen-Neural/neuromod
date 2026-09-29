// SPDX-License-Identifier: Apache-2.0 OR MIT
use crate::telemetry::TelemetryEncoder;
use crate::types::TelemetrySnapshot;

pub const FUNNEL_INPUT_NEURONS: usize = 2048;
pub const FUNNEL_HIDDEN_NEURONS: usize = 2048;
const TELEMETRY_CHANNELS: usize = 4;
const BANK_WIDTH: usize = 2;
const GIF_FAN_IN: usize = 4;
const GIF_IZ_NEURONS: usize = 5;

pub fn active_neuron_indices<T>(spikes: &[T]) -> Vec<usize>
where
    T: PartialEq + Default,
{
    let zero = T::default();

    spikes
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| (value != &zero).then_some(idx))
        .collect()
}
#[derive(Debug, Clone, PartialEq)]
pub struct FunnelActivity {
    pub ternary_events: [i8; 4],
    pub input_spike_train: Vec<Vec<usize>>,
    pub spike_train: Vec<Vec<usize>>,
    pub potentials: Vec<f32>,
    pub iz_potentials: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct SignedSplitBankBridge;

impl SignedSplitBankBridge {
    pub fn new() -> Self {
        Self
    }

    pub fn active_bank(&self, channel: usize, spike: i8) -> Option<[usize; BANK_WIDTH]> {
        if channel >= TELEMETRY_CHANNELS {
            return None;
        }

        let channel_base = channel * (BANK_WIDTH * 2);
        match spike {
            1 => Some([channel_base, channel_base + 1]),
            -1 => Some([channel_base + 2, channel_base + 3]),
            _ => None,
        }
    }

    pub fn expand(&self, ternary: [i8; 4], snn_steps: usize) -> Vec<Vec<usize>> {
        let mut spike_train = vec![Vec::new(); snn_steps];
        for (channel, spike) in ternary.into_iter().enumerate() {
            let Some(bank) = self.active_bank(channel, spike) else {
                continue;
            };
            for (step_idx, step_spikes) in spike_train.iter_mut().enumerate().take(snn_steps) {
                if step_idx % 2 == 0 {
                    step_spikes.push(bank[0]);
                    step_spikes.push(bank[1]);
                } else {
                    step_spikes.push(bank[1]);
                    step_spikes.push(bank[0]);
                }
            }
        }
        spike_train
    }
}

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

    pub fn reset(&mut self) {
        self.membrane.fill(0.0);
        self.adaptation.fill(0.0);
    }

    pub fn state_activity(&self) -> bool {
        self.membrane.iter().any(|value| value.abs() > 1e-6)
            || self.adaptation.iter().any(|value| value.abs() > 1e-6)
    }
}

impl Default for SparseGifHiddenLayer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TelemetryFunnel {
    encoder: TelemetryEncoder,
    bridge: SignedSplitBankBridge,
    hidden: SparseGifHiddenLayer,
    snn_steps: usize,
}

impl TelemetryFunnel {
    pub fn new(thresholds: [f32; 4], snn_steps: usize) -> Self {
        Self {
            encoder: TelemetryEncoder::new(thresholds),
            bridge: SignedSplitBankBridge::new(),
            hidden: SparseGifHiddenLayer::new(),
            snn_steps,
        }
    }

    pub fn encode_snapshot(&mut self, snap: &TelemetrySnapshot) -> FunnelActivity {
        let ternary_events = self.encoder.encode(snap);
        let input_spike_train = self.bridge.expand(ternary_events, self.snn_steps);
        let (spike_train, potentials, iz_potentials) = self.hidden.run(&input_spike_train);

        FunnelActivity {
            ternary_events,
            input_spike_train,
            spike_train,
            potentials,
            iz_potentials,
        }
    }

    pub fn reset(&mut self) {
        self.hidden.reset();
    }

    pub fn hidden_state_active(&self) -> bool {
        self.hidden.state_activity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        gpu_temp_c: f32,
        gpu_power_w: f32,
        cpu_tctl_c: f32,
        cpu_package_power_w: f32,
    ) -> TelemetrySnapshot {
        TelemetrySnapshot {
            gpu_temp_c,
            gpu_power_w,
            cpu_tctl_c,
            cpu_package_power_w,
            timestamp_ms: 0,
        }
    }

    #[test]
    fn signed_split_bank_mapping_uses_distinct_neurons() {
        let bridge = SignedSplitBankBridge::new();
        assert_eq!(bridge.active_bank(0, 1), Some([0, 1]));
        assert_eq!(bridge.active_bank(0, -1), Some([2, 3]));
        assert_eq!(bridge.active_bank(3, 1), Some([12, 13]));
        assert_eq!(bridge.active_bank(3, -1), Some([14, 15]));
    }

    #[test]
    fn zero_events_produce_silent_input_spike_train() {
        let bridge = SignedSplitBankBridge::new();
        let spike_train = bridge.expand([0, 0, 0, 0], 6);
        assert_eq!(spike_train.len(), 6);
        assert!(spike_train.iter().all(Vec::is_empty));
    }

    #[test]
    fn hidden_layer_output_has_expected_shape() {
        let bridge = SignedSplitBankBridge::new();
        let mut hidden = SparseGifHiddenLayer::new();
        let input = bridge.expand([1, -1, 0, 1], 20);
        let (spike_train, potentials, iz_potentials) = hidden.run(&input);

        assert_eq!(spike_train.len(), 20);
        assert_eq!(potentials.len(), FUNNEL_HIDDEN_NEURONS);
        assert_eq!(iz_potentials.len(), GIF_IZ_NEURONS);
    }

    #[test]
    fn hidden_layer_is_deterministic_from_fresh_state() {
        let bridge = SignedSplitBankBridge::new();
        let input = bridge.expand([1, 0, -1, 1], 12);
        let mut hidden_a = SparseGifHiddenLayer::new();
        let mut hidden_b = SparseGifHiddenLayer::new();

        let output_a = hidden_a.run(&input);
        let output_b = hidden_b.run(&input);

        assert_eq!(output_a, output_b);
    }

    #[test]
    fn adaptive_threshold_state_accumulates_under_repeated_drive() {
        let bridge = SignedSplitBankBridge::new();
        let input = bridge.expand([1, 1, 1, 1], 20);
        let mut hidden = SparseGifHiddenLayer::new();
        let _ = hidden.run(&input);

        assert!(hidden.state_activity());
        assert!(hidden.adaptation.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn telemetry_funnel_emits_hidden_activity_after_threshold_crossing() {
        let mut funnel = TelemetryFunnel::new([1.0, 5.0, 1.0, 5.0], 20);
        let _ = funnel.encode_snapshot(&snapshot(60.0, 260.0, 77.0, 153.0));
        let activity = funnel.encode_snapshot(&snapshot(61.0, 266.0, 78.0, 160.0));

        assert_eq!(activity.ternary_events, [1, 1, 1, 1]);
        assert_eq!(activity.input_spike_train.len(), 20);
        assert_eq!(activity.potentials.len(), FUNNEL_HIDDEN_NEURONS);
        assert_eq!(activity.iz_potentials.len(), GIF_IZ_NEURONS);
    }
}
