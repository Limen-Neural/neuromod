/// The actual STDP learning rule was missing from the original codebase. Which was orginally too massive for me to manually extract.  So I had AI Agent cdoer break it down from original codebase into smaller pieces. So I am now making the proper changes.
/// Orginally the algorithm lived in engine.rs.  The orginal codebase had plascticity but this was missing it. So weights were static and never updated in this codebase, disconnected from learning.
/// The weights don't change immediately instead it records an eligibility trace of memory.  Once a reward signal arrives (dopamine), you convert the eligibility trace into actual weight changes.
/// 
/// R-STDP (Reward based Spike-Timing-Dependent Plasticity) parameters.
///
/// ANALOGY: This is the "learning rule" — like Hebb's Rule on a timer.
/// "Neurons that fire together wire together" but only if the timing is right.
pub const RM_STDP_TAU_PLUS: f32 = 20.0;   // LTP time constant (ms / steps)
pub const RM_STDP_TAU_MINUS: f32 = 20.0;  // LTD time constant (ms / steps)
pub const RM_STDP_A_PLUS: f32 = 0.01;     // Max LTP amplitude
pub const RM_STDP_A_MINUS: f32 = 0.012;   // Max LTD amplitude (slightly stronger → stability)
pub const RM_STDP_W_MIN: f32 = 0.0;       // Minimum weight (no negative / inhibitory yet)
pub const RM_STDP_W_MAX: f32 = 2.0;       // Maximum weight (prevents runaway excitation)

// Newly added struct to track the state of a single synapse's eligibility trace
pub struct EligibilityTrace {
    /// The current value of the eligibility trace, which accumulates based on spike timing
    pub value: f32, // The value can be positive (LTP) or negative (LTD) depending on the timing of pre/post spikes
    /// The time constant that determines how quickly the eligibility trace decays
    pub tau: f32, // tau dictates over how fast it decays, so typical values are 50-100ms/steps
    // Each synapse would have its own eligibility trace instance, which gets updated based on pre/post spike timing and decays over time.
}

// Holds Rm-STDP hyperparameters
pub struct RmStdpConfig {
    /// Eligibility trace decay time constant (ms / steps)
    pub tau_eligibility: f32, // Determines how long the eligibility trace lasts before it decays back to zero. Typical values are 50-100ms/steps.
    /// LTP time constant (ms / steps)
    pub reward_lr: f32, // This dicates the learning rate for converting the eligibility trace into actual weight changes when a reward signal arrives. Typical values are 0.01-0.1.
    /// Weight clipping bounds
    pub w_min: f32, // Plays a role in preventing runaway excitation or complete silencing. Typical values are 0.0 (no negative weights) to 1.0 or 2.0 (allowing some potentiation).
    // Minimum weight (no negative / inhibitory yet)
    pub w_max: f32, // Maximum weight (prevents runaway excitation)
}

// Makes the trace decay each step on 'EligibilityTrace' struct
impl EligibilityTrace { // Call this method each time step to decay the eligibility trace
    pub fn decay(&mut self) { // Exponential decay of the eligibility trace over time
        self.value *= (-1.0 / self.tau).exp(); // Exponential decay based on tau
    }
}