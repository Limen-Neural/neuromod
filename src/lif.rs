use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoissonEncoder {
    pub num_steps: usize,
}

impl PoissonEncoder {
    pub fn new(steps: usize) -> Self {
        Self { num_steps: steps }
    }

    /// Encodes a normalized value (0.0 - 1.0) into a temporal spike train.
    /// 
    /// PHYSICS ANALOGY:
    /// This acts like a "Geiger Counter" for your data.
    /// High Intensity (Molarity/Voltage) = High Click Rate (Spikes).
    pub fn encode(&self, input: f32) -> Vec<u8> {
        let mut rng = rand::rng();
        let mut spikes = Vec::with_capacity(self.num_steps);
        
        // Clamp input to ensure probability is valid (0% to 100%)
        let probability = input.clamp(0.0, 1.0);

        for _ in 0..self.num_steps {
            // Stochastic firing: 
            // If the random number (0.0-1.0) is LESS than our intensity, we spike.
            // This mimics the noise inherent in quantum/chemical systems.
            if rng.random::<f32>() < probability {
                spikes.push(1);
            } else {
                spikes.push(0);
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
    /// Resting threshold baseline used for dynamic threshold modulation
    /// without losing the original calibrated value.
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
