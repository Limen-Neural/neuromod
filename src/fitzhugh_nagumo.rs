//! FitzHugh-Nagumo neuron model (1961) — the classic 2D relaxation oscillator.
//!
//! A simplified reduction of the Hodgkin-Huxley model that captures the essential
//! excitable dynamics with two variables: a fast voltage-like activator `v` and a
//! slow recovery variable `w`.  Despite its simplicity it exhibits threshold
//! behaviour, refractoriness, and oscillatory firing under sustained input.
//!
//! Equations:
//! ```text
//! dv/dt = v − v³/3 − w + I_app
//! dw/dt = ε · (v + a − b·w)
//! ```
//!
//! References:
//! - FitzHugh, R. (1961). Impulses and physiological states in theoretical
//!   models of nerve membrane. *Biophys. J.*, 1(6), 445–466.
//! - Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse transmission
//!   line simulating nerve axon. *Proc. IRE*, 50(10), 2061–2070.

use serde::{Deserialize, Serialize};

/// FitzHugh-Nagumo 2D neuron oscillator.
///
/// A minimal excitable system that captures the qualitative dynamics of spiking
/// neurons with far fewer parameters than Hodgkin-Huxley.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct FitzHughNagumoNeuron {
    /// Membrane potential (fast activator, dimensionless).
    pub v: f32,
    /// Recovery variable (slow adaptation, dimensionless).
    pub w: f32,
    /// Timescale separation: ε ≪ 1 means recovery is slow.
    pub epsilon: f32,
    /// Shift of the recovery nullcline.
    pub a: f32,
    /// Slope of the recovery nullcline.
    pub b: f32,
}

impl FitzHughNagumoNeuron {
    /// Standard FitzHugh-Nagumo neuron in the excitable regime.
    ///
    /// Default parameters (a=0.7, b=0.8, ε=0.08) place the fixed point on
    /// the stable branch of the cubic nullcline; the neuron fires action
    /// potentials only when driven above threshold.
    pub fn new() -> Self {
        let a = 0.7;
        let b = 0.8;
        let epsilon = 0.08;
        let (v0, w0) = Self::resting_state(a, b, 0.0);
        Self { v: v0, w: w0, epsilon, a, b }
    }

    /// Neuron in the oscillatory (tonic spiking) regime.
    ///
    /// Setting `a` near zero places the fixed point on the unstable middle branch,
    /// producing spontaneous limit-cycle oscillations even without input.
    pub fn new_oscillatory() -> Self {
        let a = -0.1;
        let b = 0.5;
        let epsilon = 0.08;
        let (v0, w0) = Self::resting_state(a, b, 0.0);
        Self { v: v0 + 0.1, w: w0, epsilon, a, b }
    }

    /// Neuron with stronger adaptation (higher ε → faster recovery).
    pub fn new_adaptive() -> Self {
        let a = 0.7;
        let b = 0.5;
        let epsilon = 0.12;
        let (v0, w0) = Self::resting_state(a, b, 0.0);
        Self { v: v0, w: w0, epsilon, a, b }
    }

    /// Compute the resting fixed point (nullcline intersection) via Newton's method.
    fn resting_state(a: f32, b: f32, i_app: f32) -> (f32, f32) {
        let mut v = 0.0f32;
        for _ in 0..50 {
            let f = v * v * v / 3.0 + (1.0 / b - 1.0) * v + (a / b - i_app);
            let df = v * v + (1.0 / b - 1.0);
            if df.abs() < 1e-12 { break; }
            let dv = f / df;
            v -= dv;
            if dv.abs() < 1e-10 { break; }
        }
        let w = v - v * v * v / 3.0 + i_app;
        (v, w)
    }

    fn dv_dt(&self, v: f32, w: f32, i_app: f32) -> f32 {
        v - v * v * v / 3.0 - w + i_app
    }

    fn dw_dt(&self, v: f32, w: f32) -> f32 {
        self.epsilon * (v + self.a - self.b * w)
    }

    /// Simulate one timestep using 4th-order Runge-Kutta (RK4).
    ///
    /// Returns `true` if V crossed above +1.0 (the spike threshold) from below.
    /// Internally subdivides `dt` into sub-steps of 0.05 for numerical stability.
    pub fn step(&mut self, i_app: f32, dt: f32) -> bool {
        let sub_dt = 0.05f32;
        let n_steps = (dt / sub_dt).round() as usize;
        if n_steps == 0 { return false; }

        let mut fired = false;
        let v_threshold: f32 = 1.0;

        for _ in 0..n_steps {
            let v_before = self.v;
            let half = sub_dt / 2.0;

            let (k1_v, k1_w) = (self.dv_dt(self.v, self.w, i_app), self.dw_dt(self.v, self.w));
            let (k2_v, k2_w) = (
                self.dv_dt(self.v + half * k1_v, self.w + half * k1_w, i_app),
                self.dw_dt(self.v + half * k1_v, self.w + half * k1_w),
            );
            let (k3_v, k3_w) = (
                self.dv_dt(self.v + half * k2_v, self.w + half * k2_w, i_app),
                self.dw_dt(self.v + half * k2_v, self.w + half * k2_w),
            );
            let (k4_v, k4_w) = (
                self.dv_dt(self.v + sub_dt * k3_v, self.w + sub_dt * k3_w, i_app),
                self.dw_dt(self.v + sub_dt * k3_v, self.w + sub_dt * k3_w),
            );

            self.v += (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
            self.w += (sub_dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w);

            if v_before < v_threshold && self.v >= v_threshold {
                fired = true;
            }
        }

        fired
    }

    /// Reset the neuron to its resting state (zero input).
    pub fn reset(&mut self) {
        let (v0, w0) = Self::resting_state(self.a, self.b, 0.0);
        self.v = v0;
        self.w = w0;
    }

    /// v-nullcline: w = v − v³/3 + I (useful for phase-plane analysis).
    pub fn v_nullcline(&self, v: f32, i_app: f32) -> f32 {
        v - v * v * v / 3.0 + i_app
    }

    /// w-nullcline: w = (v + a) / b (useful for phase-plane analysis).
    pub fn w_nullcline(&self, v: f32) -> f32 {
        (v + self.a) / self.b
    }

    /// Returns `true` if the neuron is in the excitable (stable fixed-point) regime.
    ///
    /// Stability is determined by the Hopf bifurcation condition: the trace of the
    /// Jacobian at the fixed point must be negative, i.e. `v*² > 1 − ε·b`.
    pub fn is_excitable(&self) -> bool {
        let (v_fp, _) = Self::resting_state(self.a, self.b, 0.0);
        v_fp * v_fp > 1.0 - self.epsilon * self.b
    }

    /// Approximate firing frequency under constant input (spikes per unit time).
    ///
    /// Returns `None` if no spikes are detected over `total_time`.
    pub fn firing_rate(&self, i_app: f32, total_time: f32) -> Option<f32> {
        let mut neuron = self.clone();
        let dt = 0.1f32;
        let n_steps = (total_time / dt).round() as usize;
        let spike_count = (0..n_steps).filter(|_| neuron.step(i_app, dt)).count();
        if spike_count == 0 { None } else { Some(spike_count as f32 / total_time) }
    }
}

impl Default for FitzHughNagumoNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resting_state_is_stable_without_input() {
        let mut fhn = FitzHughNagumoNeuron::new();
        for _ in 0..1000 { fhn.step(0.0, 0.5); }
        let (v_ss, w_ss) = FitzHughNagumoNeuron::resting_state(fhn.a, fhn.b, 0.0);
        assert!((fhn.v - v_ss).abs() < 0.1, "V should stay near resting state");
        assert!((fhn.w - w_ss).abs() < 0.1, "W should stay near resting state");
    }

    #[test]
    fn test_fires_with_sufficient_current() {
        let mut fhn = FitzHughNagumoNeuron::new();
        let fired = (0..5000).any(|_| fhn.step(0.7, 0.5));
        assert!(fired, "FHN neuron should fire with 0.7 sustained input");
    }

    #[test]
    fn test_no_spike_with_weak_input() {
        let mut fhn = FitzHughNagumoNeuron::new();
        for _ in 0..2000 { fhn.step(0.1, 0.5); }
        assert!(fhn.v < 1.0, "Neuron should remain subthreshold with weak input");
    }

    #[test]
    fn test_reset_restores_state() {
        let mut fhn = FitzHughNagumoNeuron::new();
        for _ in 0..5000 { fhn.step(1.0, 0.5); }
        fhn.reset();
        let (v0, w0) = FitzHughNagumoNeuron::resting_state(fhn.a, fhn.b, 0.0);
        assert!((fhn.v - v0).abs() < 1e-6, "After reset, V should return to resting state");
        assert!((fhn.w - w0).abs() < 1e-6, "After reset, W should return to resting state");
    }

    #[test]
    fn test_oscillatory_regime_spontaneous_firing() {
        let mut fhn = FitzHughNagumoNeuron::new_oscillatory();
        let fired = (0..10000).any(|_| fhn.step(0.0, 0.5));
        assert!(fired, "Oscillatory FHN should fire spontaneously");
    }

    #[test]
    fn test_firing_rate_increases_with_input() {
        let fhn = FitzHughNagumoNeuron::new();
        let rate_low = fhn.firing_rate(0.5, 500.0).unwrap_or(0.0);
        let rate_high = fhn.firing_rate(1.0, 500.0).unwrap_or(0.0);
        assert!(rate_high > rate_low, "Higher input should produce higher firing rate");
    }

    #[test]
    fn test_nullclines_intersect_at_fixed_point() {
        let fhn = FitzHughNagumoNeuron::new();
        let (v_fp, w_fp) = FitzHughNagumoNeuron::resting_state(fhn.a, fhn.b, 0.0);
        let v_nc_fp = fhn.v_nullcline(v_fp, 0.0);
        assert!((v_nc_fp - w_fp).abs() < 1e-6, "Nullclines should intersect at the fixed point");
    }

    #[test]
    fn test_excitable_regime_detection() {
        let excitable = FitzHughNagumoNeuron::new();
        assert!(excitable.is_excitable(), "Default FHN should be excitable");

        let oscillatory = FitzHughNagumoNeuron::new_oscillatory();
        assert!(!oscillatory.is_excitable(), "Oscillatory FHN should not be excitable");
    }
}
