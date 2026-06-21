//! Reward-Modulated STDP (R-STDP) Demo
//!
//! Demonstrates reward-modulated spike-timing-dependent plasticity using
//! `SpikingNetwork` and generic neuromodulators.
//!
//! Run with: cargo run --example rstdp_demo

use neuromod::{NeuroModulators, Observation, SpikingNetwork, UnitReward};

fn main() {
    println!("=== Reward-Modulated STDP Demo ===\n");

    let mut network = SpikingNetwork::new();

    println!("Network initialized:");
    println!("  LIF neurons: {}", network.neurons.len());
    println!("  Izhikevich neurons: {}", network.iz_neurons.len());
    println!("  Input channels: {}\n", 16);

    let stimuli = [
        0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.7, 0.6, 0.2, 0.8, 0.3, 0.5, 0.1, 0.9, 0.4,
    ];

    println!("Input stimuli (first 8 channels): {:?}", &stimuli[..8]);

    let mut modulators = NeuroModulators::default();
    println!("\nInitial modulators:");
    println!("  Dopamine: {:.2} (reward signal)", modulators.dopamine);
    println!(
        "  Norepinephrine: {:.2} (arousal/stress signal)",
        modulators.norepinephrine
    );
    println!(
        "  Acetylcholine: {:.2} (focus signal)",
        modulators.acetylcholine
    );
    println!(
        "  Serotonin: {:.2} (stability signal)\n",
        modulators.serotonin
    );

    println!("=== Simulation Scenarios ===\n");

    println!("--- Scenario 1: No Reward (Baseline) ---");
    modulators = NeuroModulators::default();
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!(
        "  Modulators: dopamine={:.2}, norepinephrine={:.2}",
        modulators.dopamine, modulators.norepinephrine
    );
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: DISABLED (no dopamine)\n");

    println!("--- Scenario 2: Reward State (High Dopamine) ---");
    modulators = NeuroModulators {
        dopamine: 0.9,
        norepinephrine: 0.1,
        acetylcholine: 0.7,
        ..Default::default()
    };
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!(
        "  Modulators: dopamine={:.2}, norepinephrine={:.2}, ach={:.2}",
        modulators.dopamine, modulators.norepinephrine, modulators.acetylcholine
    );
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: ENABLED (dopamine > 0.5)");
    println!("  Learning rate: {:.3}", 0.5 * modulators.dopamine);

    println!("  Sample weights (neuron 0, first 8 channels):");
    for (ch, &w) in network.neurons[0].weights.iter().take(8).enumerate() {
        println!("    Channel {}: {:.4}", ch, w);
    }
    println!();

    println!("--- Scenario 3: Stress State (High Norepinephrine) ---");
    modulators = NeuroModulators {
        dopamine: 0.2,
        norepinephrine: 0.8,
        acetylcholine: 0.3,
        ..Default::default()
    };
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!(
        "  Modulators: dopamine={:.2}, norepinephrine={:.2}, ach={:.2}",
        modulators.dopamine, modulators.norepinephrine, modulators.acetylcholine
    );
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: REDUCED (low dopamine)");
    println!(
        "  Stress multiplier: {:.3} (1.0 - norepinephrine)",
        (1.0 - modulators.norepinephrine).max(0.1)
    );
    println!();

    println!("--- Scenario 4: Focus State (High Acetylcholine) ---");
    modulators = NeuroModulators {
        dopamine: 0.6,
        norepinephrine: 0.1,
        acetylcholine: 0.9,
        serotonin: 0.5,
    };
    let spikes = network
        .step(&stimuli, &modulators)
        .expect("stimuli length must match network channels");
    println!(
        "  Modulators: dopamine={:.2}, norepinephrine={:.2}, ach={:.2}",
        modulators.dopamine, modulators.norepinephrine, modulators.acetylcholine
    );
    println!("  Neurons spiked: {}", spikes.len());
    println!("  STDP learning: ENABLED");
    println!(
        "  Decay rate adjustment: {:.3} (reduced for better memory)",
        0.15 - 0.05 * modulators.acetylcholine
    );
    println!();

    println!("=== Modulator Operations Demo ===\n");

    let mut mods = NeuroModulators::default();

    println!("Adding reward (+0.5 dopamine):");
    mods.add_reward(0.5);
    println!("  Dopamine: {:.2}", mods.dopamine);

    println!("\nAdding norepinephrine (+0.4):");
    mods.add_norepinephrine(0.4);
    println!("  Norepinephrine: {:.2}", mods.norepinephrine);

    println!("\nBoosting focus (+0.6 acetylcholine):");
    mods.boost_focus(0.6);
    println!("  Acetylcholine: {:.2}", mods.acetylcholine);

    println!("\nAdding serotonin (+0.5):");
    mods.add_serotonin(0.5);
    println!("  Serotonin: {:.2}", mods.serotonin);

    let reward = UnitReward;
    let observation = Observation::from_slice(&stimuli);
    mods.apply_reward(&reward, &observation);
    println!(
        "\nApplied GenericReward (UnitReward): dopamine={:.2}",
        mods.dopamine
    );

    println!("\nApplying decay (homeostasis):");
    mods.decay();
    println!(
        "  After decay - Dopamine: {:.2}, Norepinephrine: {:.2}, Ach: {:.2}, Serotonin: {:.2}",
        mods.dopamine, mods.norepinephrine, mods.acetylcholine, mods.serotonin
    );

    println!("\n=== Demo Complete ===");
    println!("Key takeaways:");
    println!("  • Dopamine enables STDP learning (credit assignment)");
    println!("  • Norepinephrine reduces network sensitivity (stress response)");
    println!("  • Acetylcholine adjusts decay rates (focus/memory)");
    println!("  • Serotonin stabilizes firing thresholds");
    println!("  • GenericReward allows domain-specific reward shaping upstream");
    println!("  • Decay provides homeostasis (modulators return to baseline)");
}
