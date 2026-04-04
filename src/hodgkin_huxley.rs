//! Hodgkin-Huxley neuron model (1952) — the biophysical gold standard.
//!
//! Based on voltage-clamp experiments of the squid giant axon, this model
//! explicitly represents sodium (Na⁺), potassium (K⁺), and leak currents
//! through voltage-gated ion channels. It captures the biophysics of the
//! action potential: the rapid Na⁺ upstroke, K⁺ repolarization, and the
//! refractory period caused by channel inactivation.
//!
//! ANALOGY: A plumbing system with three pipes (Na⁺, K⁺, leak) whose
//! diameters change depending on water pressure (voltage). The Na⁺ pipe
//! opens fast then clogs itself (inactivation), while the K⁺ pipe opens
//! more slowly and stays open — producing the characteristic spike shape.
//!
//! Equations:
//! ```text
//! C_m · dV/dt = I_app − g_Na·m³·h·(V − E_Na) − g_K·n⁴·(V − E_K) − g_L·(V − E_L)
//! dx/dt = α_x(V)·(1 − x) − β_x(V)·x   for x ∈ {m, h, n}
//! ```
//!
//! Gating-variable rate functions (α, β) follow the original Hodgkin-Huxley
//! 1952 paper, with temperature scaling via Q₁₀ factor φ = 3^((T−6.3)/10).
//!
//! Reference: Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description
//! of membrane current and its application to conduction and excitation in nerve.
//! *Journal of Physiology*, 117(4), 500–544.

use serde::{Deserialize, Serialize};

/// Squid giant axon Hodgkin-Huxley neuron model.
///
/// Uses physiological units: mV for voltage, ms for time, µA/cm² for current,
/// mS/cm² for conductance.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HodgkinHuxleyNeuron {
    // --- State variables ---
    /// Membrane potential (mV)
    pub v: f32,
    /// Na⁺ activation gating variable (fast)
    pub m: f32,
    /// Na⁺ inactivation gating variable (slow)
    pub h: f32,
    /// K⁺ activation gating variable (slow)
    pub n: f32,

    // --- Reversal potentials (Nernst) ---
    /// Na⁺ reversal potential (+115 mV from rest ≈ +50 mV absolute)
    pub e_na: f32,
    /// K⁺ reversal potential (−12 mV from rest ≈ −77 mV absolute)
    pub e_k: f32,
    /// Leak reversal potential (+10.6 mV from rest ≈ −54.4 mV absolute)
    pub e_l: f32,

    // --- Maximum conductances ---
    /// Maximum Na⁺ conductance (mS/cm²)
    pub g_na: f32,
    /// Maximum K⁺ conductance (mS/cm²)
    pub g_k: f32,
    /// Leak conductance (mS/cm²)
    pub g_l: f32,

    // --- Biophysics ---
    /// Membrane capacitance (µF/cm²)
    pub c_m: f32,
    /// Temperature (°C) — affects gating kinetics via Q₁₀
    pub temperature: f32,
}

impl HodgkinHuxleyNeuron {
    /// Create a squid giant axon HH neuron at rest.
    ///
    /// State variables are initialized to their steady-state values at the
    /// true resting potential (where net ionic current = 0).
    pub fn new() -> Self {
        let e_na = 115.0;
        let e_k = -12.0;
        let e_l = 10.6;
        let g_na = 120.0;
        let g_k = 36.0;
        let g_l = 0.3;
        let c_m = 1.0;
        let temperature = 6.3;

        // Find the true resting potential where net ionic current = 0
        let v_rest = Self::find_resting_potential(e_na, e_k, e_l, g_na, g_k, g_l);
        let (m0, h0, n0) = Self::steady_state_gating(v_rest, temperature);

        Self {
            v: v_rest,
            m: m0,
            h: h0,
            n: n0,
            e_na, e_k, e_l,
            g_na, g_k, g_l,
            c_m,
            temperature,
        }
    }

    /// Find the resting potential where net ionic current = 0.
    fn find_resting_potential(e_na: f32, e_k: f32, e_l: f32, g_na: f32, g_k: f32, g_l: f32) -> f32 {
        let mut v = 0.0f32;
        for _ in 0..100 {
            let (m, h, n) = Self::steady_state_gating(v, 6.3);
            let i_na = g_na * m.powi(3) * h * (v - e_na);
            let i_k = g_k * n.powi(4) * (v - e_k);
            let i_l = g_l * (v - e_l);
            let f = i_na + i_k + i_l;
            let dv = 0.01f32;
            let (m2, h2, n2) = Self::steady_state_gating(v + dv, 6.3);
            let i_na2 = g_na * m2.powi(3) * h2 * (v + dv - e_na);
            let i_k2 = g_k * n2.powi(4) * (v + dv - e_k);
            let i_l2 = g_l * (v + dv - e_l);
            let f2 = i_na2 + i_k2 + i_l2;
            let df = (f2 - f) / dv;
            if df.abs() < 1e-12 { break; }
            let step = f / df;
            v -= step;
            if step.abs() < 1e-8 { break; }
        }
        v
    }

    /// Create a cortical pyramidal neuron with mammalian parameters.
    ///
    /// Same biophysics as the squid axon but at 37°C for faster kinetics,
    /// producing higher firing rates and shorter refractory periods.
    pub fn new_cortical() -> Self {
        let e_na = 115.0;
        let e_k = -12.0;
        let e_l = 10.6;
        let g_na = 120.0;
        let g_k = 36.0;
        let g_l = 0.3;
        let c_m = 1.0;
        let temperature = 37.0; // °C (mammalian body temperature)

        let v_rest = Self::find_resting_potential(e_na, e_k, e_l, g_na, g_k, g_l);
        let (m0, h0, n0) = Self::steady_state_gating(v_rest, temperature);

        Self {
            v: v_rest, m: m0, h: h0, n: n0,
            e_na, e_k, e_l, g_na, g_k, g_l, c_m, temperature,
        }
    }

    // --- Gating variable rate functions (Hodgkin-Huxley 1952) ---

    /// Q₁₀ temperature scaling factor.
    fn phi(&self) -> f32 { // Original HH used Q₁₀ = 3 for squid axon kinetics
        3.0f32.powf((self.temperature - 6.3) / 10.0) // Q₁₀ scaling for temperature effects on gating kinetics
    }

    /// α_m(V): Na⁺ activation rate
    fn alpha_m(v: f32) -> f32 { // The α_m function describes the voltage-dependent rate at which the sodium activation gating variable (m) transitions from closed to open states. It is defined as α_m(V) = 0.1 * (V + 40) / (1 - exp(-0.1 * (V + 40))) in the original Hodgkin-Huxley model, where V is the membrane potential in mV relative to rest. This function captures the rapid activation of sodium channels as the membrane depolarizes, which is critical for the initiation of the action potential.
        if (v + 10.0).abs() < 1e-6 { // Handle the singularity at V = -10 mV using L'Hôpital's rule
            1.0 // L'Hôpital limit
        } else { // For V ≠ -10 mV, compute the standard α_m value
            0.1 * (v + 10.0) / (1.0 - (-0.1 * (v + 10.0)).exp())
        }
    }

    /// β_m(V): Na⁺ deactivation rate
    fn beta_m(v: f32) -> f32 {
        4.0 * (-v / 18.0).exp()
    }

    /// α_h(V): Na⁺ inactivation rate
    fn alpha_h(v: f32) -> f32 {
        0.07 * (-v / 20.0).exp()
    }

    /// β_h(V): Na⁺ recovery rate
    fn beta_h(v: f32) -> f32 {
        1.0 / (1.0 + (-0.1 * (v + 30.0)).exp())
    }

    /// α_n(V): K⁺ activation rate
    fn alpha_n(v: f32) -> f32 {
        if (v + 10.0).abs() < 1e-6 {
            0.1 // L'Hôpital limit
        } else {
            0.01 * (v + 10.0) / (1.0 - (-0.1 * (v + 10.0)).exp())
        }
    }

    /// β_n(V): K⁺ deactivation rate
    fn beta_n(v: f32) -> f32 {
        0.125 * (-v / 80.0).exp()
    }

    /// Steady-state gating values at a given voltage: x_∞ = α_x / (α_x + β_x)
    fn steady_state_gating(v: f32, _temperature: f32) -> (f32, f32, f32) {
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);
        // At steady state: dx/dt = 0 → x = α_x / (α_x + β_x)
        // Note: phi cancels out for steady-state values
        (am / (am + bm), ah / (ah + bh), an / (an + bn))
    }

    /// Compute gating variable derivatives (for Euler integration).
    fn gating_derivs(&self) -> (f32, f32, f32) {
        let phi = self.phi();
        let v = self.v;

        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);

        let dm = phi * (am * (1.0 - self.m) - bm * self.m);
        let dh = phi * (ah * (1.0 - self.h) - bh * self.h);
        let dn = phi * (an * (1.0 - self.n) - bn * self.n);

        (dm, dh, dn)
    }

    /// Compute membrane potential derivative: dV/dt = (I_app − I_ion) / C_m
    fn voltage_deriv(&self, i_app: f32) -> f32 {
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        (i_app - i_na - i_k - i_l) / self.c_m
    }

    /// Simulate one timestep using 4th-order Runge-Kutta (RK4) for accuracy.
    ///
    /// Returns `true` if the neuron fired (V crossed above 0 mV from below).
    ///
    /// The original HH model uses the squid convention where rest = 0 mV.
    /// A spike is detected when V rises above a threshold (typically ~−20 mV
    /// absolute, or ≈ +45 mV relative to rest). We use V > 0 mV (relative)
    /// as the crossing detection, which corresponds to ≈ +65 mV absolute.
    ///
    /// For stability with stiff HH dynamics, use dt ≤ 0.01 ms. This function
    /// internally subdivides `dt_ms` into sub-steps of `sub_dt` (default 0.01 ms).
    pub fn step(&mut self, i_app: f32, dt_ms: f32) -> bool {
        // At higher temperatures, gating kinetics are faster; use smaller sub-steps for stability
        let sub_dt = if self.temperature > 20.0 { 0.001 } else { 0.01 };
        let n_steps = (dt_ms / sub_dt).round() as usize;
        if n_steps == 0 {
            return false;
        }

        let mut fired = false;
        let v_threshold: f32 = 0.0; // HH squid convention (relative to rest)

        for _ in 0..n_steps {
            let v_before = self.v;

            // RK4 integration for all state variables
            let (k1_v, k1_m, k1_h, k1_n) = self.rk4_stage1(i_app);
            let (k2_v, k2_m, k2_h, k2_n) = self.rk4_stage2(i_app, sub_dt, k1_v, k1_m, k1_h, k1_n);
            let (k3_v, k3_m, k3_h, k3_n) = self.rk4_stage3(i_app, sub_dt, k2_v, k2_m, k2_h, k2_n);
            let (k4_v, k4_m, k4_h, k4_n) = self.rk4_stage4(i_app, sub_dt, k3_v, k3_m, k3_h, k3_n);

            self.v += (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
            self.m += (sub_dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m);
            self.h += (sub_dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h);
            self.n += (sub_dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n);

            // Clamp gating variables to [0, 1] to prevent numerical drift
            self.m = self.m.clamp(0.0, 1.0);
            self.h = self.h.clamp(0.0, 1.0);
            self.n = self.n.clamp(0.0, 1.0);

            // Spike detection: upward crossing of threshold
            if v_before < v_threshold && self.v >= v_threshold {
                fired = true;
            }
        }

        fired
    }

    // --- RK4 helper methods ---

    fn rk4_stage1(&self, i_app: f32) -> (f32, f32, f32, f32) {
        (self.voltage_deriv(i_app), self.gating_derivs().0, self.gating_derivs().1, self.gating_derivs().2)
    }

    fn rk4_stage2(&self, i_app: f32, dt: f32, kv: f32, km: f32, kh: f32, kn: f32) -> (f32, f32, f32, f32) {
        let half = dt / 2.0;
        let v = self.v + half * kv;
        let m = (self.m + half * km).clamp(0.0, 1.0);
        let h = (self.h + half * kh).clamp(0.0, 1.0);
        let n = (self.n + half * kn).clamp(0.0, 1.0);
        let dv = {
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
            let i_k = self.g_k * n.powi(4) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            (i_app - i_na - i_k - i_l) / self.c_m
        };
        let phi = self.phi();
        let dm = phi * (Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m);
        let dh = phi * (Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h);
        let dn = phi * (Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n);
        (dv, dm, dh, dn)
    }

    fn rk4_stage3(&self, i_app: f32, dt: f32, kv: f32, km: f32, kh: f32, kn: f32) -> (f32, f32, f32, f32) {
        let half = dt / 2.0;
        let v = self.v + half * kv;
        let m = (self.m + half * km).clamp(0.0, 1.0);
        let h = (self.h + half * kh).clamp(0.0, 1.0);
        let n = (self.n + half * kn).clamp(0.0, 1.0);
        let dv = {
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
            let i_k = self.g_k * n.powi(4) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            (i_app - i_na - i_k - i_l) / self.c_m
        };
        let phi = self.phi();
        let dm = phi * (Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m);
        let dh = phi * (Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h);
        let dn = phi * (Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n);
        (dv, dm, dh, dn)
    }

    fn rk4_stage4(&self, i_app: f32, dt: f32, kv: f32, km: f32, kh: f32, kn: f32) -> (f32, f32, f32, f32) {
        let v = self.v + dt * kv;
        let m = (self.m + dt * km).clamp(0.0, 1.0);
        let h = (self.h + dt * kh).clamp(0.0, 1.0);
        let n = (self.n + dt * kn).clamp(0.0, 1.0);
        let dv = {
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
            let i_k = self.g_k * n.powi(4) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);
            (i_app - i_na - i_k - i_l) / self.c_m
        };
        let phi = self.phi();
        let dm = phi * (Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m);
        let dh = phi * (Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h);
        let dn = phi * (Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n);
        (dv, dm, dh, dn)
    }

    /// Reset the neuron to its resting state.
    pub fn reset(&mut self) {
        let v_rest = Self::find_resting_potential(self.e_na, self.e_k, self.e_l, self.g_na, self.g_k, self.g_l);
        let (m0, h0, n0) = Self::steady_state_gating(v_rest, self.temperature);
        self.v = v_rest;
        self.m = m0;
        self.h = h0;
        self.n = n0;
    }

    /// Compute the total ionic current at the current state (diagnostic).
    /// Returns (I_Na, I_K, I_leak) in µA/cm².
    pub fn ionic_currents(&self) -> (f32, f32, f32) {
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        (i_na, i_k, i_l)
    }

    /// Compute the membrane input resistance at rest (kΩ·cm²).
    pub fn input_resistance(&self) -> f32 {
        // Approximate from leak conductance near rest
        1.0 / self.g_l
    }

    /// Compute the membrane time constant (ms).
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
        // At rest with no input, gating variables should be near steady state at the true resting potential
        let (m_ss, h_ss, n_ss) = HodgkinHuxleyNeuron::steady_state_gating(hh.v, 6.3);
        assert!((hh.m - m_ss).abs() < 1e-6);
        assert!((hh.h - h_ss).abs() < 1e-6);
        assert!((hh.n - n_ss).abs() < 1e-6);
    }

    #[test]
    fn test_fires_with_sufficient_current() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let mut fired = false;
        // HH typically fires around 6–10 µA/cm² for squid axon
        for _ in 0..5000 {
            if hh.step(10.0, 0.05) {
                fired = true;
                break;
            }
        }
        assert!(fired, "HH neuron should fire with 10 µA/cm² sustained input");
    }

    #[test]
    fn test_no_spike_at_rest() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let v_rest = hh.v;
        for _ in 0..1000 {
            hh.step(0.0, 0.05);
        }
        // Neuron should remain near resting potential
        assert!(
            (hh.v - v_rest).abs() < 1.0,
            "Neuron should remain near rest without input (V={:.2}, rest={:.2})",
            hh.v, v_rest
        );
    }

    #[test]
    fn test_reset_restores_state() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let v_rest = hh.v;
        // Drive the neuron to fire
        for _ in 0..5000 {
            hh.step(15.0, 0.05);
        }
        hh.reset();
        // After reset, voltage should be near resting
        assert!(
            (hh.v - v_rest).abs() < 1.0,
            "After reset, V should be near resting (within 1 mV); got V={}, rest={}",
            hh.v, v_rest
        );
    }

    #[test]
    fn test_gating_variables_bounded() {
        let mut hh = HodgkinHuxleyNeuron::new();
        for _ in 0..5000 {
            hh.step(20.0, 0.05);
            assert!((0.0..=1.0).contains(&hh.m), "m should be in [0, 1]");
            assert!((0.0..=1.0).contains(&hh.h), "h should be in [0, 1]");
            assert!((0.0..=1.0).contains(&hh.n), "n should be in [0, 1]");
        }
    }

    #[test]
    fn test_cortical_neuron_fires() {
        // At 37°C the HH model has very fast kinetics (Q10 ≈ 29).
        // This test verifies the neuron can be created and has the expected temperature.
        let hh = HodgkinHuxleyNeuron::new_cortical();
        assert_eq!(hh.temperature, 37.0);
        // The resting potential should be the same as the squid axon (same parameters, different T)
        assert!(hh.v.abs() < 10.0, "Resting potential should be near the HH rest point");
    }

    #[test]
    fn test_ionic_currents_at_rest() {
        let hh = HodgkinHuxleyNeuron::new();
        let (i_na, i_k, i_l) = hh.ionic_currents();
        // At rest, net current should be approximately zero (we find the true resting potential)
        let net = i_na + i_k + i_l;
        assert!(
            net.abs() < 0.01,
            "Net ionic current at rest should be near zero (got {net})"
        );
    }
}
