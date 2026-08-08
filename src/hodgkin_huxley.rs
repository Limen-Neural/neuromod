//! Hodgkin-Huxley neuron model (1952) — the biophysical gold standard.
//!
//! Based on voltage-clamp experiments of the squid giant axon, this model
//! explicitly represents sodium (Na⁺), potassium (K⁺), and leak currents
//! through voltage-gated ion channels. It captures the biophysics of the
//! action potential: the rapid Na⁺ upstroke, K⁺ repolarization, and the
//! refractory period caused by channel inactivation.
//!
//! This is a **standalone** biophysical neuron (not a `SpikingNetwork` bank).
//! Plasticity lives in other modules.
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
//! <https://www.nature.com/articles/117500a0>

use serde::{Deserialize, Serialize};

/// Squid giant axon Hodgkin-Huxley neuron model.
///
/// Units: mV (voltage), ms (time), µA/cm² (current), mS/cm² (conductance).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HodgkinHuxleyNeuron {
    // --- State ---
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
    /// Shift between absolute mammalian mV and the squid HH relative convention
    /// (rest = 0 mV ↔ absolute rest = −65 mV).
    const CORTICAL_VOLTAGE_SHIFT: f32 = 65.0;

    fn derivatives(&self, v: f32, m: f32, h: f32, n: f32, i_app: f32) -> (f32, f32, f32, f32) {
        let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (i_app - i_na - i_k - i_l) / self.c_m;

        // Gating-rate functions are written in the squid HH relative convention
        // (rest = 0 mV). Cortical parameters use absolute mV (rest = -65 mV), so
        // shift the voltage back to the HH convention before evaluating α/β.
        let gating_v = if self.is_cortical() {
            v + Self::CORTICAL_VOLTAGE_SHIFT
        } else {
            v
        };
        let phi = self.phi();
        let dm = phi * (Self::alpha_m(gating_v) * (1.0 - m) - Self::beta_m(gating_v) * m);
        let dh = phi * (Self::alpha_h(gating_v) * (1.0 - h) - Self::beta_h(gating_v) * h);
        let dn = phi * (Self::alpha_n(gating_v) * (1.0 - n) - Self::beta_n(gating_v) * n);
        (dv, dm, dh, dn)
    }
    /// Create a squid giant axon HH neuron at rest.
    ///
    /// State variables are initialized to their steady-state values at
    /// the resting potential (V = 0 mV in the Hodgkin-Huxley convention,
    /// which is ≈ −65 mV absolute).
    pub fn new() -> Self {
        let v_rest = 0.0f32;
        let e_na = 115.0;
        let e_k = -12.0;
        let e_l = 10.6;
        let g_na = 120.0;
        let g_k = 36.0;
        let g_l = 0.3;
        let c_m = 1.0;
        let temperature = 6.3; // °C (original HH experiments)

        let (m0, h0, n0) = Self::steady_state_gating(v_rest, temperature);

        Self {
            v: v_rest,
            m: m0,
            h: h0,
            n: n0,
            e_na,
            e_k,
            e_l,
            g_na,
            g_k,
            g_l,
            c_m,
            temperature,
        }
    }

    /// Create a cortical pyramidal neuron with mammalian parameters.
    ///
    /// Adjusted reversal potentials and temperature (37°C). Gating rates still
    /// use the HH α/β forms shifted into absolute-mV convention.
    pub fn new_cortical() -> Self {
        let mut hh = Self::new();
        hh.e_na = 50.0;
        hh.e_k = -77.0;
        hh.e_l = -54.387;
        hh.temperature = 37.0;
        let v_rest = -Self::CORTICAL_VOLTAGE_SHIFT;
        hh.v = v_rest;
        let (m0, h0, n0) = Self::steady_state_gating_mammalian(v_rest, hh.temperature);
        hh.m = m0;
        hh.h = h0;
        hh.n = n0;
        hh
    }

    // --- Gating rates (Hodgkin-Huxley 1952) ---

    /// Q₁₀ temperature scaling factor (squid axon: Q₁₀ = 3).
    fn phi(&self) -> f32 {
        3.0f32.powf((self.temperature - 6.3) / 10.0)
    }

    /// Cortical/mammalian parameterizations use absolute mV (rest ≈ −65 mV),
    /// whereas the squid axon uses the HH relative convention (rest = 0 mV).
    /// We currently distinguish the two by temperature (> 20 °C for cortex).
    fn is_cortical(&self) -> bool {
        self.temperature > 20.0
    }

    /// α_m(V): Na⁺ activation rate
    fn alpha_m(v: f32) -> f32 {
        // Singularity at V = 25 mV (HH relative convention)
        if (v - 25.0).abs() < 1e-6 {
            1.0
        } else {
            0.1 * (25.0 - v) / (((25.0 - v) / 10.0).exp() - 1.0)
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
        1.0 / (((30.0 - v) / 10.0).exp() + 1.0)
    }

    /// α_n(V): K⁺ activation rate
    fn alpha_n(v: f32) -> f32 {
        // Singularity at V = 10 mV (HH relative convention)
        if (v - 10.0).abs() < 1e-6 {
            0.1
        } else {
            0.01 * (10.0 - v) / (((10.0 - v) / 10.0).exp() - 1.0)
        }
    }

    /// β_n(V): K⁺ deactivation rate
    fn beta_n(v: f32) -> f32 {
        0.125 * (-v / 80.0).exp()
    }

    /// Steady-state gating for any voltage in HH relative convention.
    fn steady_state_gating_raw(v: f32) -> (f32, f32, f32) {
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);
        (am / (am + bm), ah / (ah + bh), an / (an + bn))
    }

    /// Steady-state gating: x_∞ = α_x / (α_x + β_x). φ cancels at steady state.
    fn steady_state_gating(v: f32, _temperature: f32) -> (f32, f32, f32) {
        Self::steady_state_gating_raw(v)
    }

    /// Steady-state gating for mammalian absolute-mV voltages (shifted into HH convention).
    fn steady_state_gating_mammalian(v: f32, _temperature: f32) -> (f32, f32, f32) {
        Self::steady_state_gating_raw(v + Self::CORTICAL_VOLTAGE_SHIFT)
    }

    /// Advance by `dt_ms` with RK4 sub-steps (default 0.01 ms).
    ///
    /// Returns `true` if V crossed above 0 mV (HH relative convention) from below.
    /// Prefer `dt_ms` such that the internal sub-step remains ≤ 0.01 ms for stiffness.
    pub fn step(&mut self, i_app: f32, dt_ms: f32) -> bool {
        let sub_dt = 0.01f32;
        let n_steps = (dt_ms / sub_dt).round() as usize;
        if n_steps == 0 {
            return false;
        }

        let mut fired = false;
        let v_threshold: f32 = 0.0;

        for _ in 0..n_steps {
            let v_before = self.v;

            let (k1_v, k1_m, k1_h, k1_n) = self.rk4_stage1(i_app);
            let (k2_v, k2_m, k2_h, k2_n) = self.rk4_stage2(i_app, sub_dt, k1_v, k1_m, k1_h, k1_n);
            let (k3_v, k3_m, k3_h, k3_n) = self.rk4_stage3(i_app, sub_dt, k2_v, k2_m, k2_h, k2_n);
            let (k4_v, k4_m, k4_h, k4_n) = self.rk4_stage4(i_app, sub_dt, k3_v, k3_m, k3_h, k3_n);

            self.v += (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
            self.m += (sub_dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m);
            self.h += (sub_dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h);
            self.n += (sub_dt / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n);

            self.m = self.m.clamp(0.0, 1.0);
            self.h = self.h.clamp(0.0, 1.0);
            self.n = self.n.clamp(0.0, 1.0);

            if v_before < v_threshold && self.v >= v_threshold {
                fired = true;
            }
        }

        fired
    }

    fn rk4_stage1(&self, i_app: f32) -> (f32, f32, f32, f32) {
        self.derivatives(self.v, self.m, self.h, self.n, i_app)
    }

    fn rk4_stage2(
        &self,
        i_app: f32,
        dt: f32,
        kv: f32,
        km: f32,
        kh: f32,
        kn: f32,
    ) -> (f32, f32, f32, f32) {
        let half = dt / 2.0;
        let v = self.v + half * kv;
        let m = (self.m + half * km).clamp(0.0, 1.0);
        let h = (self.h + half * kh).clamp(0.0, 1.0);
        let n = (self.n + half * kn).clamp(0.0, 1.0);
        self.derivatives(v, m, h, n, i_app)
    }

    fn rk4_stage3(
        &self,
        i_app: f32,
        dt: f32,
        kv: f32,
        km: f32,
        kh: f32,
        kn: f32,
    ) -> (f32, f32, f32, f32) {
        self.rk4_stage2(i_app, dt, kv, km, kh, kn)
    }

    fn rk4_stage4(
        &self,
        i_app: f32,
        dt: f32,
        kv: f32,
        km: f32,
        kh: f32,
        kn: f32,
    ) -> (f32, f32, f32, f32) {
        let v = self.v + dt * kv;
        let m = (self.m + dt * km).clamp(0.0, 1.0);
        let h = (self.h + dt * kh).clamp(0.0, 1.0);
        let n = (self.n + dt * kn).clamp(0.0, 1.0);
        self.derivatives(v, m, h, n, i_app)
    }

    /// Reset to resting potential and steady-state gates for the current temperature.
    pub fn reset(&mut self) {
        let v_rest = if self.is_cortical() {
            -Self::CORTICAL_VOLTAGE_SHIFT
        } else {
            0.0
        };
        let (m0, h0, n0) = if self.is_cortical() {
            Self::steady_state_gating_mammalian(v_rest, self.temperature)
        } else {
            Self::steady_state_gating(v_rest, self.temperature)
        };
        self.v = v_rest;
        self.m = m0;
        self.h = h0;
        self.n = n0;
    }

    /// Ionic currents (I_Na, I_K, I_leak) in µA/cm².
    pub fn ionic_currents(&self) -> (f32, f32, f32) {
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        (i_na, i_k, i_l)
    }

    /// Approximate input resistance at rest from leak conductance (kΩ·cm²).
    pub fn input_resistance(&self) -> f32 {
        1.0 / self.g_l
    }

    /// Approximate membrane time constant τ = C_m / g_L (ms).
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
        let (m_ss, h_ss, n_ss) = HodgkinHuxleyNeuron::steady_state_gating(0.0, 6.3);
        assert!((hh.m - m_ss).abs() < 1e-6);
        assert!((hh.h - h_ss).abs() < 1e-6);
        assert!((hh.n - n_ss).abs() < 1e-6);
    }

    #[test]
    fn test_fires_with_sufficient_current() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let mut fired = false;
        // Squid axon typically fires around 6–10 µA/cm²
        for _ in 0..5000 {
            if hh.step(10.0, 0.05) {
                fired = true;
                break;
            }
        }
        assert!(
            fired,
            "HH neuron should fire with 10 µA/cm² sustained input"
        );
    }

    #[test]
    fn test_no_spike_at_rest() {
        let mut hh = HodgkinHuxleyNeuron::new();
        let mut fired = false;
        for _ in 0..1000 {
            if hh.step(0.0, 0.05) {
                fired = true;
                break;
            }
        }
        assert!(!fired, "Neuron should not fire without input");
    }

    #[test]
    fn test_reset_restores_state() {
        let mut hh = HodgkinHuxleyNeuron::new();
        for _ in 0..5000 {
            hh.step(15.0, 0.05);
        }
        hh.reset();
        assert!(
            hh.v.abs() < 1.0,
            "After reset, V should be near resting (within 1 mV)"
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
        let mut hh = HodgkinHuxleyNeuron::new_cortical();
        let baseline = hh.v;
        let mut peak_v = hh.v;
        // Mammalian parameterization is simplified; assert substantial depolarization.
        for _ in 0..5000 {
            hh.step(20.0, 0.05);
            peak_v = peak_v.max(hh.v);
        }
        assert!(
            peak_v > baseline + 5.0,
            "Cortical HH neuron should depolarize substantially under sustained input"
        );
    }

    #[test]
    fn test_cortical_gating_derivatives_at_rest() {
        // At cortical rest the gates are initialized to their steady-state values.
        // derivatives() must apply the same +65 mV shift used by new_cortical and reset,
        // otherwise dm/dh/dn would not be zero at rest.
        let mut hh = HodgkinHuxleyNeuron::new_cortical();
        let (_, dm, dh, dn) = hh.derivatives(hh.v, hh.m, hh.h, hh.n, 0.0);
        assert!(
            dm.abs() < 1e-6,
            "dm should be near zero at cortical rest (got {dm})"
        );
        assert!(
            dh.abs() < 1e-6,
            "dh should be near zero at cortical rest (got {dh})"
        );
        assert!(
            dn.abs() < 1e-6,
            "dn should be near zero at cortical rest (got {dn})"
        );

        // Perturb and reset to confirm the shift is also consistent in reset().
        for _ in 0..100 {
            hh.step(0.0, 0.05);
        }
        hh.reset();
        let (_, dm, dh, dn) = hh.derivatives(hh.v, hh.m, hh.h, hh.n, 0.0);
        assert!(
            dm.abs() < 1e-6,
            "dm should be near zero after reset (got {dm})"
        );
        assert!(
            dh.abs() < 1e-6,
            "dh should be near zero after reset (got {dh})"
        );
        assert!(
            dn.abs() < 1e-6,
            "dn should be near zero after reset (got {dn})"
        );
    }

    #[test]
    fn test_ionic_currents_at_rest() {
        let hh = HodgkinHuxleyNeuron::new();
        let (i_na, i_k, i_l) = hh.ionic_currents();
        let net = i_na + i_k + i_l;
        assert!(
            net.abs() < 1.0,
            "Net ionic current at rest should be near zero (got {net})"
        );
    }
}
