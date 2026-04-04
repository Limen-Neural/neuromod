//! Hodgkin-Huxley neuron model (1952) — the biophysical gold standard.
//!
//! Based on voltage-clamp experiments of the squid giant axon, this model
//! explicitly represents sodium (Na⁺), potassium (K⁺), and leak currents
//! through voltage-gated ion channels. It captures the full biophysics of the
//! action potential: the rapid Na⁺ upstroke, K⁺ repolarisation, and the
//! refractory period caused by channel inactivation.
//!
//! Equations:
//! ```text
//! C_m · dV/dt = I_app − g_Na·m³·h·(V − E_Na) − g_K·n⁴·(V − E_K) − g_L·(V − E_L)
//! dx/dt = α_x(V)·(1 − x) − β_x(V)·x   for x ∈ {m, h, n}
//! ```
//!
//! Reference:
//! - Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane
//!   current and its application to conduction and excitation in nerve.
//!   *J. Physiol.*, 117(4), 500–544. doi:10.1113/jphysiol.1952.sp004764

use serde::{Deserialize, Serialize};

/// Squid giant axon Hodgkin-Huxley neuron model.
///
/// Uses physiological units: mV for voltage, ms for time, µA/cm² for current,
/// mS/cm² for conductance.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HodgkinHuxleyNeuron {
    /// Membrane potential (mV, squid convention: rest = 0 mV).
    pub v: f32,
    /// Na⁺ activation gating variable (fast).
    pub m: f32,
    /// Na⁺ inactivation gating variable (slow).
    pub h: f32,
    /// K⁺ activation gating variable (slow).
    pub n: f32,

    /// Na⁺ reversal potential (mV from rest).
    pub e_na: f32,
    /// K⁺ reversal potential (mV from rest).
    pub e_k: f32,
    /// Leak reversal potential (mV from rest).
    pub e_l: f32,

    /// Maximum Na⁺ conductance (mS/cm²).
    pub g_na: f32,
    /// Maximum K⁺ conductance (mS/cm²).
    pub g_k: f32,
    /// Leak conductance (mS/cm²).
    pub g_l: f32,

    /// Membrane capacitance (µF/cm²).
    pub c_m: f32,
    /// Temperature (°C) — scales gating kinetics via Q₁₀.
    pub temperature: f32,
}

impl HodgkinHuxleyNeuron {
    /// Squid giant axon at 6.3 °C (original HH 1952 conditions).
    pub fn new() -> Self {
        let (e_na, e_k, e_l) = (115.0, -12.0, 10.6);
        let (g_na, g_k, g_l) = (120.0, 36.0, 0.3);
        let c_m = 1.0;
        let temperature = 6.3;
        let v_rest = Self::find_resting_potential(e_na, e_k, e_l, g_na, g_k, g_l);
        let (m0, h0, n0) = Self::steady_state_gating(v_rest);
        Self { v: v_rest, m: m0, h: h0, n: n0, e_na, e_k, e_l, g_na, g_k, g_l, c_m, temperature }
    }

    /// Mammalian cortical neuron at 37 °C (faster kinetics).
    pub fn new_cortical() -> Self {
        let (e_na, e_k, e_l) = (115.0, -12.0, 10.6);
        let (g_na, g_k, g_l) = (120.0, 36.0, 0.3);
        let c_m = 1.0;
        let temperature = 37.0;
        let v_rest = Self::find_resting_potential(e_na, e_k, e_l, g_na, g_k, g_l);
        let (m0, h0, n0) = Self::steady_state_gating(v_rest);
        Self { v: v_rest, m: m0, h: h0, n: n0, e_na, e_k, e_l, g_na, g_k, g_l, c_m, temperature }
    }

    /// Solve for the resting potential where net ionic current = 0.
    fn find_resting_potential(e_na: f32, e_k: f32, e_l: f32, g_na: f32, g_k: f32, g_l: f32) -> f32 {
        let mut v = 0.0f32;
        for _ in 0..100 {
            let (m, h, n) = Self::steady_state_gating(v);
            let f = g_na * m.powi(3) * h * (v - e_na)
                  + g_k  * n.powi(4)        * (v - e_k)
                  + g_l                     * (v - e_l);
            let dv = 0.01f32;
            let (m2, h2, n2) = Self::steady_state_gating(v + dv);
            let f2 = g_na * m2.powi(3) * h2 * (v + dv - e_na)
                   + g_k  * n2.powi(4)         * (v + dv - e_k)
                   + g_l                        * (v + dv - e_l);
            let df = (f2 - f) / dv;
            if df.abs() < 1e-12 { break; }
            let step = f / df;
            v -= step;
            if step.abs() < 1e-8 { break; }
        }
        v
    }

    // --- Gating variable rate functions (Hodgkin-Huxley 1952) ---

    /// Q₁₀ temperature scaling factor (φ = 3^((T − 6.3) / 10)).
    fn phi(&self) -> f32 {
        3.0f32.powf((self.temperature - 6.3) / 10.0)
    }

    fn alpha_m(v: f32) -> f32 {
        if (v + 10.0).abs() < 1e-6 { 1.0 }
        else { 0.1 * (v + 10.0) / (1.0 - (-0.1 * (v + 10.0)).exp()) }
    }

    fn beta_m(v: f32) -> f32 { 4.0 * (-v / 18.0).exp() }

    fn alpha_h(v: f32) -> f32 { 0.07 * (-v / 20.0).exp() }

    fn beta_h(v: f32) -> f32 { 1.0 / (1.0 + (-0.1 * (v + 30.0)).exp()) }

    fn alpha_n(v: f32) -> f32 {
        if (v + 10.0).abs() < 1e-6 { 0.1 }
        else { 0.01 * (v + 10.0) / (1.0 - (-0.1 * (v + 10.0)).exp()) }
    }

    fn beta_n(v: f32) -> f32 { 0.125 * (-v / 80.0).exp() }

    /// Steady-state gating values: x_∞ = α_x / (α_x + β_x).
    fn steady_state_gating(v: f32) -> (f32, f32, f32) {
        let am = Self::alpha_m(v); let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v); let bh = Self::beta_h(v);
        let an = Self::alpha_n(v); let bn = Self::beta_n(v);
        (am / (am + bm), ah / (ah + bh), an / (an + bn))
    }

    fn gating_derivs(&self) -> (f32, f32, f32) {
        let phi = self.phi();
        let v = self.v;
        let dm = phi * (Self::alpha_m(v) * (1.0 - self.m) - Self::beta_m(v) * self.m);
        let dh = phi * (Self::alpha_h(v) * (1.0 - self.h) - Self::beta_h(v) * self.h);
        let dn = phi * (Self::alpha_n(v) * (1.0 - self.n) - Self::beta_n(v) * self.n);
        (dm, dh, dn)
    }

    fn voltage_deriv(&self, i_app: f32) -> f32 {
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
        let i_k  = self.g_k  * self.n.powi(4)          * (self.v - self.e_k);
        let i_l  = self.g_l                             * (self.v - self.e_l);
        (i_app - i_na - i_k - i_l) / self.c_m
    }

    // --- RK4 helper stages ---

    fn rk4_stage1(&self, i_app: f32) -> (f32, f32, f32, f32) {
        let (dm, dh, dn) = self.gating_derivs();
        (self.voltage_deriv(i_app), dm, dh, dn)
    }

    fn rk4_stage(&self, i_app: f32, dt: f32, kv: f32, km: f32, kh: f32, kn: f32) -> (f32, f32, f32, f32) {
        let v = self.v + dt * kv;
        let m = (self.m + dt * km).clamp(0.0, 1.0);
        let h = (self.h + dt * kh).clamp(0.0, 1.0);
        let n = (self.n + dt * kn).clamp(0.0, 1.0);
        let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
        let i_k  = self.g_k  * n.powi(4)     * (v - self.e_k);
        let i_l  = self.g_l                   * (v - self.e_l);
        let dv = (i_app - i_na - i_k - i_l) / self.c_m;
        let phi = self.phi();
        let dm = phi * (Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m);
        let dh = phi * (Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h);
        let dn = phi * (Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n);
        (dv, dm, dh, dn)
    }

    /// Simulate one timestep using 4th-order Runge-Kutta (RK4).
    ///
    /// Returns `true` if V crossed above 0 mV (squid convention rest = 0 mV).
    /// Internally uses sub-steps of 0.01 ms (0.001 ms above 20 °C) for stability.
    pub fn step(&mut self, i_app: f32, dt_ms: f32) -> bool {
        let sub_dt = if self.temperature > 20.0 { 0.001 } else { 0.01 };
        let n_steps = (dt_ms / sub_dt).round() as usize;
        if n_steps == 0 { return false; }

        let mut fired = false;

        for _ in 0..n_steps {
            let v_before = self.v;
            let half = sub_dt / 2.0;

            let (k1_v, k1_m, k1_h, k1_n) = self.rk4_stage1(i_app);
            let (k2_v, k2_m, k2_h, k2_n) = self.rk4_stage(i_app, half, k1_v, k1_m, k1_h, k1_n);
            let (k3_v, k3_m, k3_h, k3_n) = self.rk4_stage(i_app, half, k2_v, k2_m, k2_h, k2_n);
            let (k4_v, k4_m, k4_h, k4_n) = self.rk4_stage(i_app, sub_dt, k3_v, k3_m, k3_h, k3_n);

            self.v += (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
            self.m = (self.m + (sub_dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)).clamp(0.0, 1.0);
            self.h = (self.h + (sub_dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)).clamp(0.0, 1.0);
            self.n = (self.n + (sub_dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)).clamp(0.0, 1.0);

            if v_before < 0.0 && self.v >= 0.0 {
                fired = true;
            }
        }

        fired
    }

    /// Reset the neuron to its resting state.
    pub fn reset(&mut self) {
        let v_rest = Self::find_resting_potential(self.e_na, self.e_k, self.e_l, self.g_na, self.g_k, self.g_l);
        let (m0, h0, n0) = Self::steady_state_gating(v_rest);
        self.v = v_rest; self.m = m0; self.h = h0; self.n = n0;
    }

    /// Total ionic currents at the current state (µA/cm²): (I_Na, I_K, I_leak).
    pub fn ionic_currents(&self) -> (f32, f32, f32) {
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
        let i_k  = self.g_k  * self.n.powi(4)          * (self.v - self.e_k);
        let i_l  = self.g_l                             * (self.v - self.e_l);
        (i_na, i_k, i_l)
    }

    /// Membrane time constant τ = C_m / g_L (ms).
    pub fn membrane_time_constant(&self) -> f32 {
        self.c_m / self.g_l
    }
}

impl Default for HodgkinHuxleyNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resting_state_is_stable() {
        let hh = HodgkinHuxleyNeuron::new();
        let (m_ss, h_ss, n_ss) = HodgkinHuxleyNeuron::steady_state_gating(hh.v);
        assert!((hh.m - m_ss).abs() < 1e-6);
        assert!((hh.h - h_ss).abs() < 1e-6);
        assert!((hh.n - n_ss).abs() < 1e-6);
    }

    #[test]
    fn test_fires_with_sufficient_current() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let fired = (0..5000).any(|_| hh.step(10.0, 0.05));
        assert!(fired, "HH neuron should fire with 10 µA/cm² sustained input");
    }

    #[test]
    fn test_no_spike_at_rest() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let v_rest = hh.v;
        for _ in 0..1000 { hh.step(0.0, 0.05); }
        assert!(
            (hh.v - v_rest).abs() < 1.0,
            "Neuron should remain near rest without input (V={:.2}, rest={:.2})", hh.v, v_rest
        );
    }

    #[test]
    fn test_reset_restores_state() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let v_rest = hh.v;
        for _ in 0..5000 { hh.step(15.0, 0.05); }
        hh.reset();
        assert!(
            (hh.v - v_rest).abs() < 1.0,
            "After reset V should be near resting; got V={}, rest={}", hh.v, v_rest
        );
    }

    #[test]
    fn test_gating_variables_bounded() {
        let mut hh = HodgkinHuxleyNeuron::new();
        for _ in 0..5000 {
            hh.step(20.0, 0.05);
            assert!((0.0..=1.0).contains(&hh.m), "m should be in [0,1]");
            assert!((0.0..=1.0).contains(&hh.h), "h should be in [0,1]");
            assert!((0.0..=1.0).contains(&hh.n), "n should be in [0,1]");
        }
    }

    #[test]
    fn test_cortical_neuron_temperature() {
        let hh = HodgkinHuxleyNeuron::new_cortical();
        assert_eq!(hh.temperature, 37.0);
        assert!(hh.v.abs() < 10.0, "Resting potential should be near the HH rest point");
    }

    #[test]
    fn test_ionic_currents_at_rest() {
        let hh = HodgkinHuxleyNeuron::new();
        let (i_na, i_k, i_l) = hh.ionic_currents();
        let net = i_na + i_k + i_l;
        assert!(net.abs() < 0.01, "Net ionic current at rest should be near zero (got {net})");
    }
}
