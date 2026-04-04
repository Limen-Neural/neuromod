//! Hebbian Learning Example with Classical STDP
//! 
//! This example demonstrates classical Hebbian spike-timing-dependent plasticity
//! using Lapicque neurons. The network learns to strengthen connections between
//! neurons that fire together ("neurons that fire together wire together").
//! 
//! Run with: cargo run --example hebbian_learning

use neuromod::{LapicqueNeuron, apply_classical_stdp, StdpParams};

fn main() {
    println!("=== Hebbian Learning with Classical STDP ===\n");
    
    // Create two Lapicque neurons (pre-synaptic and post-synaptic)
    let mut pre_neuron = LapicqueNeuron::new();
    let mut post_neuron = LapicqueNeuron::new();
    
    // Initialize synaptic weight
    let mut synaptic_weight = 0.5;
    
    // Set up STDP parameters
    let stdp_params = StdpParams::default();
    
    println!("Initial state:");
    println!("  Synaptic weight: {:.4}", synaptic_weight);
    println!("  STDP parameters: A+={:.3}, A-={:.3}, τ+={:.1}, τ-={:.1}\n",
             stdp_params.a_plus, stdp_params.a_minus, 
             stdp_params.tau_plus, stdp_params.tau_minus);
    
    // Simulate learning over multiple trials
    println!("Running 5 learning trials:\n");
    
    for trial in 0..5 {
        println!("--- Trial {} ---", trial + 1);
        
        // Reset neurons
        pre_neuron.membrane_potential = 0.0;
        post_neuron.membrane_potential = 0.0;
        
        let mut pre_spike_time: i64 = -1;
        let mut post_spike_time: i64 = -1;
        
        // Simulate 50 time steps
        for step in 0..50 {
            // Pre-neuron gets strong input at step 10
            let pre_input = if step == 10 { 0.1 } else { 0.0 };
            pre_neuron.integrate(pre_input);
            
            // Post-neuron gets input from pre-synapse (weighted) at step 15
            let post_input = if step == 15 { synaptic_weight * 0.1 } else { 0.0 };
            post_neuron.integrate(post_input);
            
            // Check for spikes
            if pre_neuron.check_for_spike(step) {
                pre_spike_time = step;
                println!("  Step {:2}: Pre-neuron SPIKES", step);
            }
            
            if post_neuron.check_for_spike(step) {
                post_spike_time = step;
                println!("  Step {:2}: Post-neuron SPIKES", step);
            }
        }
        
        // Apply STDP if both neurons fired
        if pre_spike_time >= 0 && post_spike_time >= 0 {
            println!("  Applying STDP: pre_time={}, post_time={}, Δt={}",
                     pre_spike_time, post_spike_time, post_spike_time - pre_spike_time);
            
            synaptic_weight = apply_classical_stdp(
                pre_spike_time,
                post_spike_time,
                synaptic_weight,
                &stdp_params,
            );
            
            println!("  Updated synaptic weight: {:.4}", synaptic_weight);
        } else {
            println!("  No STDP update (both neurons must fire)");
        }
        
        println!("  Final weight: {:.4}\n", synaptic_weight);
    }
    
    println!("=== Learning Complete ===");
    println!("Final synaptic weight: {:.4}", synaptic_weight);
    println!("\nThis demonstrates:");
    println!("  • Pre-synaptic neuron fires first (causality)");
    println!("  • Post-synaptic neuron fires shortly after");
    println!("  • STDP strengthens the weight (LTP)");
    println!("  • Repeated trials increase synaptic strength");
    println!("\nNote: If post fires before pre, weight would decrease (LTD)");
}
