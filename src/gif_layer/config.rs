use serde::{Deserialize, Serialize};

use crate::gif::GifParams;

use super::{GIF_LAYER_DEFAULT_FAN_IN, GIF_LAYER_DEFAULT_W_MAX, GIF_LAYER_DEFAULT_W_MIN};

/// Construction parameters for a
/// [`SparseGifHiddenLayer`](super::SparseGifHiddenLayer).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseGifLayerConfig {
    /// Number of input channels the layer reads.
    pub num_inputs: usize,
    /// Number of GIF neurons in the layer.
    pub num_neurons: usize,
    /// Synapses per neuron. Must be `<= num_inputs`. `0` builds a
    /// topology-free layer that only ever sees zero drive.
    pub fan_in: usize,
    /// Seed for the deterministic topology and weight generator.
    pub seed: u64,
    /// Inclusive-exclusive range for the initial uniform synaptic weights.
    pub weight_range: (f32, f32),
    /// Shared GIF dynamics for every neuron in the bank.
    pub params: GifParams,
}

impl Default for SparseGifLayerConfig {
    fn default() -> Self {
        Self {
            num_inputs: crate::NUM_INPUT_CHANNELS,
            num_neurons: crate::NUM_INPUT_CHANNELS,
            fan_in: GIF_LAYER_DEFAULT_FAN_IN,
            seed: 0,
            weight_range: (GIF_LAYER_DEFAULT_W_MIN, GIF_LAYER_DEFAULT_W_MAX),
            params: GifParams::default(),
        }
    }
}
