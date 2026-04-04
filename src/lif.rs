// Credit: Majority edited by author in newest version.  Grok 4.20 Expert with the help of researching and tutor by Qwen Coder 3.6 on fixing bugs and Gemini 3.1 for fixing the last bit of bug.

use rand::Rng; // For stochastic spike generation in PoissonEncoder
use serde::{Deserialize, Serialize}; // For serialization of neuron states, useful for saving/loading models and debugging

#[derive(Clone, Debug, Serialize, Deserialize)] // A simple fixed-point representation for synaptic weights (stub for demonstration)
pub struct PoissonEncoder { // This struct encodes continuous input values into temporal spike trains using a Poisson process, which is a common method for simulating the stochastic nature of neuronal firing. The `num_steps` field determines how many time steps the encoder will produce for each input value, allowing for temporal encoding of information.
    pub num_steps: usize,// Number of time steps to encode the input into a spike train
}

impl PoissonEncoder { // Constructor for creating a new PoissonEncoder with a specified number of steps
    pub fn new(steps: usize) -> Self {  // Constructor for creating a new PoissonEncoder with a specified number of steps
        Self { num_steps: steps } // Constructor for creating a new PoissonEncoder with a specified number of steps
    }

    /// Encodes a normalized value (0.0 - 1.0) into a temporal spike train.
    /// 
    /// PHYSICS ANALOGY:
    /// This acts like a "Geiger Counter" for your data.
    /// High Intensity (Molarity/Voltage) = High Click Rate (Spikes).
    pub fn encode(&self, input: f32) -> Vec<u8> { // Encodes a normalized value (0.0 - 1.0) into a temporal spike train, where the number of spikes is proportional to the input intensity. This simulates how biological neurons convert continuous stimuli into discrete spikes, with higher input values leading to more frequent firing. The output is a vector of binary values (0s and 1s) representing the presence or absence of a spike at each time step.
        let mut rng = rand::thread_rng(); // Create a random number generator for stochastic spike generation
        let mut spikes = Vec::with_capacity(self.num_steps); // Pre-allocate a vector to hold the spike train, improving performance by avoiding dynamic resizing during encoding
        
        // Clamp input to ensure probability is valid (0% to 100%)
        let probability = input.clamp(0.0, 1.0); // Clamp input to ensure probability is valid (0% to 100%), preventing invalid probabilities that could arise from out-of-range inputs. This ensures that the encoding process remains stable and produces meaningful spike trains even if the input is not perfectly normalized.

        for _ in 0..self.num_steps { // For each time step, generate a random number and compare it to the input intensity to decide whether to emit a spike. This simulates the inherent noise in biological systems, where even a constant stimulus can lead to variable firing patterns due to stochastic processes at the molecular level.
            // Stochastic firing: 
            // If the random number (0.0-1.0) is LESS than our intensity, we spike.
            // This mimics the noise inherent in quantum/chemical systems.
            if rng.r#gen::<f32>() < probability { // Stochastic firing: If the random number (0.0-1.0) is LESS than our intensity, we spike. This mimics the noise inherent in quantum/chemical systems, where the exact timing of spikes can vary even for the same input due to underlying stochastic processes.
                spikes.push(1); // Emit a spike (1) if the random number is less than the input intensity, indicating that the neuron has fired in this time step. The use of `rng.gen::<f32>()` generates a random floating-point number between 0.0 and 1.0, which is then compared to the clamped input value to determine whether a spike occurs.
            } else { // Otherwise, emit no spike (0)
                spikes.push(0); // Emit no spike (0) if the random number is greater than or equal to the input intensity, indicating that the neuron did not fire in this time step. This allows for a probabilistic encoding of the input, where higher input values lead to more frequent spikes, but there is still variability in the exact timing of those spikes due to the stochastic nature of the encoding process.
            }
        }
        spikes
    }
}

/// This struct simulates the physical properties of a biological neuron.
/// 
/// CIRCUIT ANALOGY (RC Circuit):
/// - Membrane Potential = Voltage across a Capacitor.
/// - Decay Rate = Current leakage through a Resistor.
/// - Threshold = Breakdown voltage of a component (like a Diode or Spark Gap).
/// - Weights = Resistor values on each input trace (synaptic strength).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LifNeuron {
    pub membrane_potential: f32, // Current charge state
    pub decay_rate: f32,         // How fast it "forgets" (Leak)
    pub threshold: f32,          // Limit to trigger an action potential
    /// Resting threshold — stored so N15 (global inhibitory interneuron) can
    /// modulate `threshold` dynamically via Vth(t) = base_threshold + w_inhib·S15(t)
    /// and decay back without losing the original calibrated value.
    #[serde(default)]
    pub base_threshold: f32,
    pub last_spike: bool,        // Tracks if it fired in the last step
    /// Synaptic weights — one per input channel.
    /// These are learned via STDP during training.
    #[serde(default)]
    pub weights: Vec<f32>,
    /// Timestep of the most recent spike (for STDP delta-t calculation).
    /// Uses a global step counter maintained by the engine.
    #[serde(default)]
    pub last_spike_time: i64,
}

impl Default for LifNeuron {
    fn default() -> Self {
        Self {
            membrane_potential: 0.0,
            decay_rate: 0.15,
            threshold: 0.02,  // Aggressively lowered threshold
            base_threshold: 0.02,
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
        }
    }
}

impl LifNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    /// The Core Logic Step:
    /// 1. Add Input (Integration)
    /// 2. Lose Charge (Leak)
    pub fn integrate(&mut self, stimulus: f32) {
        // CHARGE: Add input stimulus to current state
        self.membrane_potential += stimulus;
        
        // LEAK: Passive decay over time (Simulates real-world signal loss)
        self.membrane_potential -= self.membrane_potential * self.decay_rate;
    }

    /// Check if the neuron should fire.
    /// If yes, captures the peak potential, then performs a hard reset (Refractory Period).
    /// Returns `Some(peak_potential)` on a spike, `None` otherwise.
    /// Capturing before reset lets debug logs show the actual firing voltage, not the post-reset 0.0.
    pub fn check_fire(&mut self) -> Option<f32> {
        if self.membrane_potential >= self.threshold {
            let peak = self.membrane_potential; // Capture BEFORE reset
            self.membrane_potential = 0.0;       // Hard reset after spike
            return Some(peak);
        }
        None
    }
}
