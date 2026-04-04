//! Basic LIF Neuron Example
//! 
//! This example demonstrates the simplest possible use of neuromod:
//! a single LIF neuron that responds to input stimuli and fires spikes.
//! 
//! Run with: cargo run --example basic_lif

use neuromod::LifNeuron;

fn main() {
    println!("=== Basic LIF Neuron Example ===\n");
    
    // Create a single LIF neuron with default parameters
    let mut neuron = LifNeuron::new();
    
    println!("Initial neuron state:");
    println!("  Membrane potential: {:.4}", neuron.membrane_potential);
    println!("  Threshold: {:.4}", neuron.threshold);
    println!("  Decay rate: {:.4}\n", neuron.decay_rate);
    
    // Simulate multiple time steps with varying input
    println!("Simulating 20 time steps with pulsed input:\n");
    
    for step in 0..20 {
        // Create a pulsed input: high for steps 5-10, low otherwise
        let stimulus = if (5..10).contains(&step) { 0.08 } else { 0.01 };
        
        // Integrate the input
        neuron.integrate(stimulus);
        
        // Check if neuron fires
        if let Some(peak) = neuron.check_fire() {
            println!("Step {:2}: Input={:.3} → SPIKE! (peak potential: {:.4})", 
                     step, stimulus, peak);
        } else {
            println!("Step {:2}: Input={:.3} → Potential: {:.4}", 
                     step, stimulus, neuron.membrane_potential);
        }
    }
    
    println!("\n=== Simulation Complete ===");
    println!("This demonstrates:");
    println!("  • LIF neuron integrates input over time");
    println!("  • Passive leak causes potential to decay");
    println!("  • Spike occurs when potential exceeds threshold");
    println!("  • After spike, potential resets to 0");
}
