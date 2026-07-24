//! R-STDP (Reward-modulated Spike-Timing-Dependent Plasticity) parameters.
//!
//! Weights don't change immediately on a spike pair; instead an eligibility
//! trace accumulates a "memory" of recent pre/post spike-timing coincidences.
//! When a reward signal (dopamine) arrives, that trace is converted into an
//! actual weight change.
//!
//! ANALOGY: This is the "learning rule" — like Hebb's Rule on a timer.
//! "Neurons that fire together wire together," but only if the timing is right.

/// LTP (potentiation) time constant, in steps.
pub const RM_STDP_TAU_PLUS: f32 = 20.0;
/// LTD (depression) time constant, in steps.
pub const RM_STDP_TAU_MINUS: f32 = 20.0;
/// Maximum LTP amplitude.
pub const RM_STDP_A_PLUS: f32 = 0.01;
/// Maximum LTD amplitude (slightly stronger than LTP for stability).
pub const RM_STDP_A_MINUS: f32 = 0.012;
/// Minimum synaptic weight (no negative/inhibitory weights yet).
pub const RM_STDP_W_MIN: f32 = 0.0;
/// Maximum synaptic weight (prevents runaway excitation).
pub const RM_STDP_W_MAX: f32 = 2.0;

const _: () = assert!(RM_STDP_W_MIN < RM_STDP_W_MAX);
const _: () = assert!(RM_STDP_A_MINUS >= RM_STDP_A_PLUS);

/// Eligibility trace for a single synapse.
///
/// Accumulates based on pre/post spike timing and decays exponentially over
/// time; positive values favor potentiation (LTP), negative values favor
/// depression (LTD). Each synapse holds its own trace instance.
pub struct EligibilityTrace {
    /// Current trace value.
    pub value: f32,
    /// Decay time constant, in steps. Typical values are 50-100.
    pub tau: f32,
}

/// R-STDP hyperparameters.
pub struct RmStdpConfig {
    /// Eligibility trace decay time constant, in steps. Typical values are 50-100.
    pub tau_eligibility: f32,
    /// Learning rate for converting an eligibility trace into a weight change
    /// when a reward signal arrives. Typical values are 0.01-0.1.
    pub reward_lr: f32,
    /// Minimum weight (no negative/inhibitory weights yet).
    pub w_min: f32,
    /// Maximum weight (prevents runaway excitation).
    pub w_max: f32,
}

impl EligibilityTrace {
    /// Decay the trace by one step.
    pub fn decay(&mut self) {
        // Guard against non-positive tau, which would cause division by zero
        // or exponential growth instead of decay.
        let tau = self.tau.max(f32::EPSILON);
        self.value *= (-1.0 / tau).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_scales_value_by_exp_neg_inv_tau() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: 50.0,
        };
        let expected_factor = (-1.0_f32 / 50.0).exp();

        trace.decay();

        assert!((trace.value - expected_factor).abs() < 1e-6);
    }

    #[test]
    fn decay_applied_repeatedly_compounds_toward_zero() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: 50.0,
        };
        let factor = (-1.0_f32 / 50.0).exp();

        for _ in 0..5 {
            trace.decay();
        }

        let expected = factor.powi(5);
        assert!((trace.value - expected).abs() < 1e-5);
        assert!(trace.value < 1.0);
    }

    #[test]
    fn decay_preserves_sign_for_negative_values() {
        let mut trace = EligibilityTrace {
            value: -1.0,
            tau: 50.0,
        };
        trace.decay();
        assert!(trace.value < 0.0);
    }

    #[test]
    fn decay_with_zero_tau_does_not_panic_or_diverge() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: 0.0,
        };
        trace.decay();
        assert!(trace.value.is_finite());
        assert!(trace.value >= 0.0);
    }

    #[test]
    fn decay_with_negative_tau_does_not_panic_or_diverge() {
        let mut trace = EligibilityTrace {
            value: 1.0,
            tau: -10.0,
        };
        trace.decay();
        assert!(trace.value.is_finite());
        assert!(trace.value >= 0.0);
    }
}
