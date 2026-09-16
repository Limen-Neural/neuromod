//! Minimal outsider onboarding demo: `neuromod` from crates.io, no git path.
//!
//! ```bash
//! cargo run
//! ```
use neuromod::{LifNeuron, NeuroModulators, SpikingNetwork};

fn main() {
    println!("=== neuromod crates.io standalone demo ===");

    // Deterministic single-neuron path (no Poisson encoding).
    let mut neuron = LifNeuron::new();
    println!(
        "LIF: threshold={:.4}, decay={:.4}",
        neuron.threshold, neuron.decay_rate
    );
    for step in 0..12 {
        let stimulus = if (3..8).contains(&step) { 0.08 } else { 0.01 };
        neuron.integrate(stimulus);
        match neuron.check_fire() {
            Some(peak) => println!("  step {step:2}: stimulus={stimulus:.3} SPIKE peak={peak:.4}"),
            None => println!(
                "  step {step:2}: stimulus={stimulus:.3} V={:.4}",
                neuron.membrane_potential
            ),
        }
    }

    // Engine contract: default 16 LIF / 5 Izhikevich / 16 channels.
    let mut network = SpikingNetwork::new();
    let stimuli = [0.5_f32; 16];
    let modulators = NeuroModulators::default();
    println!(
        "Network: {} LIF, {} Izhikevich, {} channels",
        network.neurons.len(),
        network.iz_neurons.len(),
        network.num_channels
    );
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!("step() fired LIF indices: {spikes:?}");
}
