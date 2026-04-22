//! Basic example of using the neuromod library
use neuromod::{SpikingNetwork, NeuroModulators};

fn main() {  
    println!("=== Neuromod Basic Example ===");
    
    // Create network
    let mut network = SpikingNetwork::new();
    println!("✓ Created spiking network with {} neurons", network.neurons.len());
    
    // Create input stimuli (16 channels)
    let stimuli = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.7,
                   0.6, 0.2, 0.8, 0.3, 0.5, 0.1, 0.9, 0.4];
    
    println!("✓ Created input stimuli: {:?}", &stimuli);
    
    // Create neuromodulators
    let modulators = NeuroModulators::default();
    println!("✓ Created neuromodulators: dopamine={:.2}, cortisol={:.2}, ach={:.2}, tempo={:.2}",
             modulators.dopamine, modulators.cortisol, modulators.acetylcholine, modulators.tempo);
    
    // Step the network
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!("✓ Network step completed");
    println!("  Neurons that spiked: {:?}", spikes);
    
    // Get membrane potentials
    let potentials = network.get_membrane_potentials();
    println!("  Membrane potentials: {:?}", potentials.iter().take(8).collect::<Vec<_>>());
    
    // Get thresholds
    let thresholds = network.get_thresholds();
    println!("  Thresholds: {:?}", thresholds.iter().take(8).collect::<Vec<_>>());
    
    println!("✓ Example completed successfully!");
}
