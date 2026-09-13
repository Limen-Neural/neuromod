use serde::{Deserialize, Serialize};

use super::GifLayerError;

/// Spike output of a batched [`SparseGifHiddenLayer::run`](super::SparseGifHiddenLayer::run).
///
/// Stored as one flat row-major `bool` buffer (`step * num_neurons + neuron`)
/// rather than a `Vec<Vec<bool>>`, matching the layer's structure-of-arrays
/// storage and keeping a whole raster in one allocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SpikeRasterRepr")]
pub struct SpikeRaster {
    pub(super) num_steps: usize,
    pub(super) num_neurons: usize,
    pub(super) spikes: Vec<bool>,
}

/// Deserialization mirror of [`SpikeRaster`].
///
/// Same rationale as `SparseGifHiddenLayerRepr`: the derived `Deserialize`
/// cannot enforce `spikes.len() == num_steps * num_neurons`, and
/// [`SpikeRaster::step`] only bounds-checks the step index before slicing, so a
/// short buffer panics there rather than failing at decode.
#[derive(Deserialize)]
#[serde(rename = "SpikeRaster")]
struct SpikeRasterRepr {
    num_steps: usize,
    num_neurons: usize,
    spikes: Vec<bool>,
}

impl TryFrom<SpikeRasterRepr> for SpikeRaster {
    type Error = GifLayerError;

    fn try_from(repr: SpikeRasterRepr) -> Result<Self, Self::Error> {
        // Checked, not saturating: this product is the exact length the buffer
        // must have, so a wrapped value would be compared against and could
        // spuriously match a buffer that is nothing like the right size.
        let expected = repr.num_steps.checked_mul(repr.num_neurons).ok_or(
            GifLayerError::MalformedCheckpoint {
                detail: "num_steps * num_neurons overflows",
            },
        )?;

        if repr.spikes.len() != expected {
            return Err(GifLayerError::MalformedCheckpoint {
                detail: "spikes length != num_steps * num_neurons",
            });
        }

        Ok(Self {
            num_steps: repr.num_steps,
            num_neurons: repr.num_neurons,
            spikes: repr.spikes,
        })
    }
}

impl SpikeRaster {
    /// Number of time steps recorded.
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Number of neurons per step.
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Spike flags for one time step, or `None` if `step` is out of range.
    pub fn step(&self, step: usize) -> Option<&[bool]> {
        if step >= self.num_steps {
            return None;
        }
        let lo = step * self.num_neurons;
        Some(&self.spikes[lo..lo + self.num_neurons])
    }

    /// Indices of the neurons that fired at `step`.
    pub fn fired_at(&self, step: usize) -> Vec<usize> {
        self.step(step)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(i, &fired)| fired.then_some(i))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total spikes across the whole raster.
    pub fn total_spikes(&self) -> usize {
        self.spikes.iter().filter(|&&s| s).count()
    }

    /// Spike count per neuron over the whole raster.
    pub fn per_neuron_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.num_neurons];
        for row in 0..self.num_steps {
            let lo = row * self.num_neurons;
            for (neuron, count) in counts.iter_mut().enumerate() {
                if self.spikes[lo + neuron] {
                    *count += 1;
                }
            }
        }
        counts
    }

    /// Flat row-major view of the raster (`step * num_neurons + neuron`).
    pub fn as_flat(&self) -> &[bool] {
        &self.spikes
    }
}
