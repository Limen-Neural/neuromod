//! Reward-Modulated STDP (R-STDP) Demo
//!
//! Walks the wired reward-modulated plasticity path in `SpikingNetwork`:
//! per-synapse eligibility traces accumulate spike-timing coincidences on
//! every step, and dopamine converts those traces into weight changes.
//!
//! The weight and eligibility reports are read back out of the network after
//! `step`, not narrated from constants. Setup lines, stimuli, and modulator
//! levels print the inputs used to drive it.
//!
//! Run with: cargo run --example rstdp_demo

use neuromod::{NeuroModulators, Observation, RmStdpConfig, SpikingNetwork, UnitReward};

/// Channels 0 and 1 fire on every step; channels 2 and 3 stay silent.
const STIMULI: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

/// Print neuron 0's synaptic weights beside the eligibility traces that drive them.
fn report(network: &SpikingNetwork, label: &str) {
    let neuron = &network.neurons[0];
    print!("  {label:<22}");
    for ch in 0..network.num_channels {
        print!(
            " | ch{ch}: w={:.4} e={:+.4}",
            neuron.weights[ch], neuron.eligibility[ch].value
        );
    }
    println!();
}

/// The reward state Scenarios 2 and 5 both drive with.
fn rewarded_state() -> NeuroModulators {
    NeuroModulators {
        dopamine: 0.9,
        norepinephrine: 0.1,
        acetylcholine: 0.7,
        ..Default::default()
    }
}

/// Four LIF neurons over four channels, every synapse seeded at an equal share
/// of the engine's L1 weight budget so the renormalization pass starts neutral
/// and any drift below is learning.
fn build_network(seed: f32) -> SpikingNetwork {
    let mut network = SpikingNetwork::with_dimensions(4, 2, STIMULI.len());
    for neuron in &mut network.neurons {
        neuron.weights = vec![seed; STIMULI.len()];
    }
    network
}

fn print_setup(network: &SpikingNetwork, config: &RmStdpConfig, seed: f32) {
    println!("Network initialized:");
    println!("  LIF neurons:        {}", network.neurons.len());
    println!("  Izhikevich neurons: {}", network.iz_neurons.len());
    println!("  Input channels:     {}", network.num_channels);
    println!("  Seed weight:        {seed:.4} per synapse");
    println!("\nR-STDP configuration (`RmStdpConfig::default()`):");
    println!("  tau_eligibility: {:.1} steps", config.tau_eligibility);
    println!("  reward_lr:       {:.3}", config.reward_lr);
    println!(
        "  weight bounds:   [{:.1}, {:.1}]",
        config.w_min, config.w_max
    );
    println!("\nStimuli: {STIMULI:?}  (channels 0-1 driven, 2-3 silent)\n");
}

/// Dopamine off: the driven synapses bank credit they are not paid for.
fn scenario_no_reward(network: &mut SpikingNetwork, config: &RmStdpConfig, seed: f32) {
    println!("--- Scenario 1: No Reward — credit is earned, not yet paid ---");
    let no_reward = NeuroModulators::default();
    println!("  dopamine={:.2}\n", no_reward.dopamine);
    report(network, "before");
    for _ in 0..10 {
        network
            .step(&STIMULI, &no_reward)
            .expect("stimuli length must match network channels");
    }
    report(network, "after 10 steps");
    println!(
        "  Weights held at {:.4}, still the {seed:.4} they were seeded with: the\n  \
         dopamine gate is shut. The driven synapses did bank eligibility (e > 0)\n  \
         — that credit stays claimable for roughly {:.0} steps.\n",
        network.neurons[0].weights[0], config.tau_eligibility
    );
}

/// Dopamine on: the banked traces convert into weight changes.
fn scenario_reward(network: &mut SpikingNetwork, config: &RmStdpConfig) {
    println!("--- Scenario 2: Reward State (High Dopamine) — the trace is cashed in ---");
    let rewarded = rewarded_state();
    println!(
        "  dopamine={:.2}, norepinephrine={:.2}, ach={:.2}",
        rewarded.dopamine, rewarded.norepinephrine, rewarded.acetylcholine
    );
    let dopamine_lr = 0.5 * rewarded.dopamine;
    println!(
        "  dopamine_lr: {dopamine_lr:.3}  (trace payout factor \
         reward_lr x dopamine_lr = {:.4})\n",
        config.reward_lr * dopamine_lr
    );
    let before = network.neurons[0].weights.clone();
    for _ in 0..10 {
        network
            .step(&STIMULI, &rewarded)
            .expect("stimuli length must match network channels");
    }
    report(network, "after 10 steps");
    print!("  Weight deltas:        ");
    for (ch, (&now, &then)) in network.neurons[0]
        .weights
        .iter()
        .zip(before.iter())
        .enumerate()
    {
        print!(" | ch{ch}: {:+.4}", now - then);
    }
    println!();
    println!(
        "  The trace payout itself is reward_lr x dopamine_lr x eligibility, applied\n  \
         per synapse. These printed deltas are that payout *after* the L1\n  \
         renormalization pass — which is why channels 2 and 3 moved at all despite\n  \
         holding no trace: the driven synapses took share of a fixed budget.\n"
    );
}

fn scenario_stress(network: &mut SpikingNetwork) {
    println!("--- Scenario 3: Stress State (High Norepinephrine) ---");
    let stressed = NeuroModulators {
        dopamine: 0.2,
        norepinephrine: 0.8,
        acetylcholine: 0.3,
        ..Default::default()
    };
    let spikes = network
        .step(&STIMULI, &stressed)
        .expect("stimuli length must match network channels");
    println!(
        "  dopamine={:.2}, norepinephrine={:.2}, ach={:.2}",
        stressed.dopamine, stressed.norepinephrine, stressed.acetylcholine
    );
    println!("  Neurons spiked: {}", spikes.len());
    println!(
        "  Stress multiplier: {:.3} (1.0 - norepinephrine) — input drive is damped,\n  \
         and the smaller learning rate slows the trace payout without stopping\n  \
         accumulation.\n",
        (1.0 - stressed.norepinephrine).max(0.1)
    );
}

fn scenario_focus(network: &mut SpikingNetwork) {
    println!("--- Scenario 4: Focus State (High Acetylcholine) ---");
    let focused = NeuroModulators {
        dopamine: 0.6,
        norepinephrine: 0.1,
        acetylcholine: 0.9,
        serotonin: 0.5,
    };
    let spikes = network
        .step(&STIMULI, &focused)
        .expect("stimuli length must match network channels");
    println!(
        "  dopamine={:.2}, norepinephrine={:.2}, ach={:.2}, serotonin={:.2}",
        focused.dopamine, focused.norepinephrine, focused.acetylcholine, focused.serotonin
    );
    println!("  Neurons spiked: {}", spikes.len());
    println!(
        "  Membrane decay rate: {:.3} (reduced for better memory)",
        network.neurons[0].decay_rate
    );
    println!(
        "  Firing threshold:    {:.3}\n",
        network.neurons[0].threshold
    );
}

/// A/B the reconfigured network against an otherwise identical twin left on the
/// defaults, so the effect of `RmStdpConfig` is measured rather than asserted.
fn scenario_slow_traces(network: &mut SpikingNetwork) {
    println!("--- Scenario 5: Slower Traces via `RmStdpConfig` ---");
    // Snapshot the network before touching its config. The twin inherits this
    // exact state and keeps the defaults, then sees the identical stimuli and
    // modulators below — so the only thing separating the two runs is
    // `RmStdpConfig`. `SpikingNetwork` is not `Clone`, but it is serde
    // round-trippable, which is the same mechanism used for checkpointing.
    let mut default_twin: SpikingNetwork =
        serde_json::from_str(&serde_json::to_string(&network).expect("network must serialize"))
            .expect("snapshot must deserialize");

    let entry_weight = network.neurons[0].weights[0];
    let entry_trace = network.neurons[0].eligibility[0].value;

    network.set_rm_stdp_config(RmStdpConfig {
        tau_eligibility: 200.0,
        reward_lr: 0.02,
        ..RmStdpConfig::default()
    });
    println!(
        "  tau_eligibility {:.1} -> {:.1}, reward_lr {:.3} -> {:.3}",
        default_twin.stdp_config.tau_eligibility,
        network.stdp_config.tau_eligibility,
        default_twin.stdp_config.reward_lr,
        network.stdp_config.reward_lr
    );
    println!(
        "  ch0 trace across the switch: {entry_trace:+.4} -> {:+.4} (kept), now\n  \
         decaying with tau={:.1}\n",
        network.neurons[0].eligibility[0].value, network.neurons[0].eligibility[0].tau
    );

    // Same 10 rewarded steps Scenario 2 used, run on both networks.
    let rewarded = rewarded_state();
    for _ in 0..10 {
        network
            .step(&STIMULI, &rewarded)
            .expect("stimuli length must match network channels");
        default_twin
            .step(&STIMULI, &rewarded)
            .expect("stimuli length must match network channels");
    }
    report(network, "slow config");
    report(&default_twin, "default config");

    let slow_gain = network.neurons[0].weights[0] - entry_weight;
    let default_gain = default_twin.neurons[0].weights[0] - entry_weight;
    // Fraction of a trace discarded per step, read off each network's live tau.
    let per_step_loss = |tau: f32| 100.0 * (1.0 - (-1.0 / tau).exp());
    println!(
        "  Over those 10 steps ch0 gained {slow_gain:+.4} under the slow config \
         against\n  {default_gain:+.4} under the defaults: a net weight-gain \
         ratio of {:.2}x. That is\n  an end-to-end figure, not a payout factor \
         — both config changes and the L1\n  pass feed into it. `reward_lr` \
         alone accounts for {:.2}x; the slow config's\n  larger trace pulls the \
         other way.\n  \
         That trace is {:+.4} against {:+.4}, since the longer tau discards \
         {:.2}%\n  of it per step instead of {:.2}%. The 4x there is the ratio \
         of decay\n  constants, not of trace values — both networks keep taking \
         in fresh\n  coincidences every step, so the gap compounds rather than \
         arriving at 4x.\n",
        slow_gain / default_gain,
        network.stdp_config.reward_lr / default_twin.stdp_config.reward_lr,
        network.neurons[0].eligibility[0].value,
        default_twin.neurons[0].eligibility[0].value,
        per_step_loss(network.stdp_config.tau_eligibility),
        per_step_loss(default_twin.stdp_config.tau_eligibility),
    );
}

/// The modulator helpers, independent of the engine.
fn modulator_operations() {
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
    let observation = Observation::from_slice(&STIMULI);
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
}

fn print_takeaways() {
    println!("\n=== Demo Complete ===");
    println!("Key takeaways:");
    println!("  • Eligibility traces accumulate every step, dopamine or not");
    println!("  • Dopamine gates only the trace -> weight conversion (credit assignment)");
    println!("  • Reward can therefore arrive after the coincidence it pays for");
    println!("  • Norepinephrine reduces network sensitivity (stress response)");
    println!("  • Acetylcholine adjusts decay rates (focus/memory)");
    println!("  • Serotonin stabilizes firing thresholds");
    println!("  • `RmStdpConfig` tunes trace decay, reward rate, and weight bounds");
    println!("  • GenericReward allows domain-specific reward shaping upstream");
    println!("  • Decay provides homeostasis (modulators return to baseline)");
}

fn main() {
    println!("=== Reward-Modulated STDP Demo ===\n");

    let seed = 2.0 / STIMULI.len() as f32;
    let mut network = build_network(seed);
    let config = RmStdpConfig::default();
    print_setup(&network, &config, seed);

    println!("=== Simulation Scenarios ===\n");
    scenario_no_reward(&mut network, &config, seed);
    scenario_reward(&mut network, &config);
    scenario_stress(&mut network);
    scenario_focus(&mut network);
    scenario_slow_traces(&mut network);

    modulator_operations();
    print_takeaways();
}
