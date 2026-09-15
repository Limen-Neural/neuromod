//! Sparse GIF Hidden Layer Example
//!
//! Builds a deterministic structure-of-arrays bank of Generalized
//! Integrate-and-Fire neurons with sparse fan-in, runs a batched spike train
//! through it, and shows that the same seed reproduces the same output.
//!
//! Run with: cargo run --example sparse_gif_layer

use neuromod::gif_layer::{SparseGifHiddenLayer, SparseGifLayerConfig};

const NUM_INPUTS: usize = 32;
const NUM_NEURONS: usize = 8;
const NUM_STEPS: usize = 24;

/// A simple deterministic stimulus: each channel carries a travelling ramp.
fn spike_train() -> Vec<Vec<f32>> {
    (0..NUM_STEPS)
        .map(|t| {
            (0..NUM_INPUTS)
                .map(|c| ((t + c) % 5) as f32 * 0.25)
                .collect()
        })
        .collect()
}

fn main() {
    println!("=== Sparse GIF Hidden Layer Example ===\n");

    let config = SparseGifLayerConfig {
        num_inputs: NUM_INPUTS,
        num_neurons: NUM_NEURONS,
        fan_in: 6,
        seed: 0xC0FF_EE01,
        ..Default::default()
    };

    let mut layer = SparseGifHiddenLayer::new(&config).expect("valid layer configuration");

    println!(
        "Layer: {} inputs -> {} neurons, {} synapses (fan-in {}), seed 0x{:X}\n",
        layer.num_inputs(),
        layer.num_neurons(),
        layer.num_synapses(),
        config.fan_in,
        layer.seed(),
    );

    println!("Deterministic sparse fan-in (source channels per neuron):");
    for neuron in 0..layer.num_neurons() {
        let (sources, weights) = layer.fan_in_of(neuron).expect("neuron in range");
        let weight_sum: f32 = weights.iter().sum();
        println!("  neuron {neuron}: sources {sources:?}  (weight sum {weight_sum:.3})");
    }

    // Batched execution over the whole spike train in one call.
    let train = spike_train();
    let raster = layer.run(&train).expect("frame widths match the layer");

    println!("\nRan {} steps. Spikes per neuron:", raster.num_steps());
    for (neuron, count) in raster.per_neuron_counts().iter().enumerate() {
        let rate = *count as f32 / raster.num_steps() as f32;
        println!("  neuron {neuron}: {count:2} spikes  (rate {rate:.2})");
    }
    println!("  total: {} spikes", raster.total_spikes());

    println!("\nFinal structure-of-arrays state:");
    println!("  membrane:   {:.3?}", layer.membrane());
    println!("  adaptation: {:.3?}", layer.adaptation());

    // Same seed + same input => same output, with no wall-clock or
    // thread-order dependence anywhere in the path.
    let mut twin = SparseGifHiddenLayer::new(&config).expect("valid layer configuration");
    let twin_raster = twin.run(&train).expect("frame widths match the layer");
    assert_eq!(raster, twin_raster);
    println!("\nReproducibility check: identical seed reproduced the raster exactly.");

    println!("\n=== Simulation Complete ===");
    println!("This demonstrates:");
    println!("  • Structure-of-arrays state (parallel Vecs, not per-neuron structs)");
    println!("  • Layer-owned deterministic sparse fan-in topology");
    println!("  • Batched run() over a whole spike train");
    println!("  • GIF dynamics shared with the single-neuron GifNeuron model");
}
