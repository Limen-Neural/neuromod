//! Reward-Modulated STDP (R-STDP) Demo
//! 
//! This example demonstrates reward-modulated spike-timing-dependent plasticity
//! using the full SpikingNetwork with neuromodulators. The network learns only
//! when rewarded (dopamine), allowing credit assignment in reinforcement learning.
//! 
//! Run with: cargo run --example rstdp_demo

use neuromod::{SpikingNetwork, NeuroModulators};

fn main() {
    println!("=== Reward-Modulated STDP Demo ===\n");
    
    // Create the spiking network (16 LIF neurons + 5 Izhikevich neurons)
    let mut network = SpikingNetwork::new();
    
    println!("Network initialized:");
    println!("  LIF neurons: {}", network.neurons.len());
    println!("  Izhikevich neurons: {}", network.iz_neurons.len());
    println!("  Input channels: {}\n", 16);
    
    // Create input stimuli (16 channels)
    let stimuli = [0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.7,
                   0.6, 0.2, 0.8, 0.3, 0.5, 0.1, 0.9, 0.4];
    
    println!("Input stimuli (first 8 channels): {:?}", &stimuli[..8]);
    
    // Create neuromodulators
    let mut modulators = NeuroModulators::default();
    println!("\nInitial modulators:");
    println!("  Dopamine: {:.2} (reward signal)", modulators.dopamine);
    println!("  Cortisol: {:.2} (stress signal)", modulators.cortisol);
    println!("  Acetylcholine: {:.2} (focus signal)", modulators.acetylcholine);
    println!("  Tempo: {:.2} (time scaling)\n", modulators.tempo);
    
    // Run simulation with different modulator states
    println!("=== Simulation Scenarios ===\n");
    
    // Scenario 1: No reward (baseline)
    println!("--- Scenario 1: No Reward (Baseline) ---");
    modulators = NeuroModulators::default();
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!("  Modulators: dopamine={:.2}, cortisol={:.2}", 
             modulators.dopamine, modulators.cortisol);
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: DISABLED (no dopamine)\n");
    
    // Scenario 2: Reward state (high dopamine)
    println!("--- Scenario 2: Reward State (High Dopamine) ---");
    modulators.dopamine = 0.9;
    modulators.cortisol = 0.1;
    modulators.acetylcholine = 0.7;
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!("  Modulators: dopamine={:.2}, cortisol={:.2}, ach={:.2}", 
             modulators.dopamine, modulators.cortisol, modulators.acetylcholine);
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: ENABLED (dopamine > 0.5)");
    println!("  Learning rate: {:.3}", 0.5 * modulators.dopamine);
    
    // Show some weight changes
    println!("  Sample weights (neuron 0, first 8 channels):");
    for (ch, &w) in network.neurons[0].weights.iter().take(8).enumerate() {
        println!("    Channel {}: {:.4}", ch, w);
    }
    println!();
    
    // Scenario 3: Stress state (high cortisol)
    println!("--- Scenario 3: Stress State (High Cortisol) ---");
    modulators = NeuroModulators::default();
    modulators.dopamine = 0.2;
    modulators.cortisol = 0.8;
    modulators.acetylcholine = 0.3;
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!("  Modulators: dopamine={:.2}, cortisol={:.2}, ach={:.2}", 
             modulators.dopamine, modulators.cortisol, modulators.acetylcholine);
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: REDUCED (low dopamine)");
    println!("  Stress multiplier: {:.3} (1.0 - cortisol)", 
             (1.0 - modulators.cortisol).max(0.1));
    println!();
    
    // Scenario 4: Focus state (high acetylcholine)
    println!("--- Scenario 4: Focus State (High Acetylcholine) ---");
    modulators = NeuroModulators::default();
    modulators.dopamine = 0.6;
    modulators.cortisol = 0.1;
    modulators.acetylcholine = 0.9;
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!("  Modulators: dopamine={:.2}, cortisol={:.2}, ach={:.2}", 
             modulators.dopamine, modulators.cortisol, modulators.acetylcholine);
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: ENABLED");
    println!("  Decay rate adjustment: {:.3} (reduced for better memory)", 
             0.15 - 0.05 * modulators.acetylcholine);
    println!();
    
    // Demonstrate modulator operations
    println!("=== Modulator Operations Demo ===\n");
    
    let mut mods = NeuroModulators::default();
    
    println!("Adding reward (+0.5 dopamine):");
    mods.add_reward(0.5);
    println!("  Dopamine: {:.2}", mods.dopamine);
    
    println!("\nAdding stress (+0.4 cortisol):");
    mods.add_stress(0.4);
    println!("  Cortisol: {:.2}", mods.cortisol);
    
    println!("\nBoosting focus (+0.6 acetylcholine):");
    mods.boost_focus(0.6);
    println!("  Acetylcholine: {:.2}", mods.acetylcholine);
    
    println!("\nSetting tempo to 1.5:");
    mods.set_tempo(1.5);
    println!("  Tempo: {:.2}", mods.tempo);
    
    println!("\nApplying decay (homeostasis):");
    mods.decay();
    println!("  After decay - Dopamine: {:.2}, Cortisol: {:.2}, Ach: {:.2}",
             mods.dopamine, mods.cortisol, mods.acetylcholine);
    
    println!("\n=== Demo Complete ===");
    println!("Key takeaways:");
    println!("  • Dopamine enables STDP learning (credit assignment)");
    println!("  • Cortisol reduces network sensitivity (stress response)");
    println!("  • Acetylcholine adjusts decay rates (focus/memory)");
    println!("  • Modulators can be computed from environment signals");
    println!("  • Decay provides homeostasis (modulators return to baseline)");
}
