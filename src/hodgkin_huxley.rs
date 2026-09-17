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

/// Voltage coordinates shared by the membrane and reversal potentials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoltageConvention {
    /// Original HH coordinates: nominal rest is 0 mV.
    RelativeToRest,
    /// Absolute coordinates: nominal rest is −65 mV.
    Absolute,
}

/// Squid giant axon Hodgkin-Huxley neuron model.
///
/// Units: mV (voltage), ms (time), µA/cm² (current), mS/cm² (conductance).
///
/// JSON includes `voltage_convention` as `"relative_to_rest"` or `"absolute"`.
/// Legacy JSON without this field is accepted only for the exact constructor
/// reversal tuples `(115, -12, 10.6)` and `(50, -77, -54.387)`, respectively,
/// regardless of temperature. Custom legacy tuples must supply the field.
/// Explicit null or invalid conventions are rejected; decoding never shifts state.
/// The added field requires updating source struct literals and may break
/// positional binary serialization formats.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(try_from = "HodgkinHuxleyWire")]
pub struct HodgkinHuxleyNeuron {
    /// Coordinates of `v`, `e_na`, `e_k`, and `e_l`, independent of temperature.
    /// Changing this field does not convert state: callers must also shift all
    /// four voltages consistently (subtract 65 mV when changing to absolute).
    pub voltage_convention: VoltageConvention,
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

// Omission permits legacy inference, while an explicit null must remain an error.
#[derive(Deserialize)]
struct HodgkinHuxleyWire {
    #[serde(default, deserialize_with = "deserialize_present_convention")]
    voltage_convention: Option<VoltageConvention>,
    v: f32,
    m: f32,
    h: f32,
    n: f32,
    e_na: f32,
    e_k: f32,
    e_l: f32,
    g_na: f32,
    g_k: f32,
    g_l: f32,
    c_m: f32,
    temperature: f32,
}

fn deserialize_present_convention<'de, D>(
    deserializer: D,
) -> Result<Option<VoltageConvention>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    VoltageConvention::deserialize(deserializer).map(Some)
}

impl TryFrom<HodgkinHuxleyWire> for HodgkinHuxleyNeuron {
    type Error = &'static str;

    fn try_from(wire: HodgkinHuxleyWire) -> Result<Self, Self::Error> {
        let voltage_convention = match wire.voltage_convention {
            Some(convention) => convention,
            None => match (wire.e_na, wire.e_k, wire.e_l) {
                (115.0, -12.0, 10.6) => VoltageConvention::RelativeToRest,
                (50.0, -77.0, -54.387) => VoltageConvention::Absolute,
                _ => {
                    return Err(
                        "custom reversal potentials require an explicit voltage_convention",
                    );
                }
            },
        };
        Ok(Self {
            voltage_convention,
            v: wire.v,
            m: wire.m,
            h: wire.h,
            n: wire.n,
            e_na: wire.e_na,
            e_k: wire.e_k,
            e_l: wire.e_l,
            g_na: wire.g_na,
            g_k: wire.g_k,
            g_l: wire.g_l,
            c_m: wire.c_m,
            temperature: wire.temperature,
        })
    }
}

impl HodgkinHuxleyNeuron {
    /// Shift between absolute mammalian mV and the squid HH relative convention
    /// (rest = 0 mV ↔ absolute rest = −65 mV).
    const ABSOLUTE_VOLTAGE_SHIFT: f32 = 65.0;

    fn derivatives(&self, v: f32, m: f32, h: f32, n: f32, i_app: f32) -> (f32, f32, f32, f32) {
        let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
        let i_k = self.g_k * n.powi(4) * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (i_app - i_na - i_k - i_l) / self.c_m;

        let gating_v = self.relative_voltage(v);
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

        let (m0, h0, n0) = Self::steady_state_gating(v_rest);

        Self {
            voltage_convention: VoltageConvention::RelativeToRest,
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
        hh.voltage_convention = VoltageConvention::Absolute;
        let v_rest = -Self::ABSOLUTE_VOLTAGE_SHIFT;
        hh.v = v_rest;
        let (m0, h0, n0) = Self::steady_state_gating(hh.relative_voltage(v_rest));
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

    fn relative_voltage(&self, v: f32) -> f32 {
        match self.voltage_convention {
            VoltageConvention::RelativeToRest => v,
            VoltageConvention::Absolute => v + Self::ABSOLUTE_VOLTAGE_SHIFT,
        }
    }

    fn resting_voltage(&self) -> f32 {
        match self.voltage_convention {
            VoltageConvention::RelativeToRest => 0.0,
            VoltageConvention::Absolute => -Self::ABSOLUTE_VOLTAGE_SHIFT,
        }
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
    fn steady_state_gating(v: f32) -> (f32, f32, f32) {
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);
        (am / (am + bm), ah / (ah + bh), an / (an + bn))
    }

    /// Advance by `dt_ms` with RK4 sub-steps (default 0.01 ms).
    ///
    /// Returns `true` if V crossed above 0 mV (HH relative convention) from below.
    /// Covers the full finite positive `dt_ms` with sub-steps no larger than 0.01 ms.
    /// Zero, negative, and non-finite durations return `false` without mutation.
    /// Runtime scales with duration; extremely large durations are impractical.
    /// A crossing in any sub-step is retained in the returned result.
    pub fn step(&mut self, i_app: f32, dt_ms: f32) -> bool {
        if !dt_ms.is_finite() || dt_ms <= 0.0 {
            return false;
        }
        if let Some(half) = Self::split_duration(dt_ms) {
            let first = self.step(i_app, half);
            let second = self.step(i_app, half);
            return first || second;
        }
        let mut remaining = f64::from(dt_ms);

        let mut fired = false;
        let v_threshold: f32 = 0.0;

        while remaining > 0.0 {
            let sub_dt = remaining.min(f64::from(0.01f32)) as f32;
            remaining -= f64::from(sub_dt);
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

    // Keep subtraction within f64's exact range for f32 durations. Large
    // intervals are halved exactly, with at most 135 recursive stack frames;
    // unlike a cast step count or repeated subtraction from a huge float,
    // this cannot overflow or stop making progress.
    fn split_duration(duration: f32) -> Option<f32> {
        (duration > 0.01f32 * 65_536.0).then_some(duration * 0.5)
    }

    /// Reset to nominal rest in the selected coordinates and steady-state gates.
    /// Temperature and all other model parameters are preserved. Q₁₀ cancels
    /// from the steady-state gate values.
    pub fn reset(&mut self) {
        let v_rest = self.resting_voltage();
        let (m0, h0, n0) = Self::steady_state_gating(self.relative_voltage(v_rest));
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
    fn convention_legacy_json_infers_only_reversals_and_preserves_state() {
        for (preset, convention, name, rest, reversals) in [
            (
                HodgkinHuxleyNeuron::new(),
                VoltageConvention::RelativeToRest,
                "relative_to_rest",
                0.0,
                (115.0, -12.0, 10.6),
            ),
            (
                HodgkinHuxleyNeuron::new_cortical(),
                VoltageConvention::Absolute,
                "absolute",
                -65.0,
                (50.0, -77.0, -54.387),
            ),
        ] {
            assert_eq!(preset.voltage_convention, convention);
            assert_eq!(preset.v, rest);
            assert_eq!((preset.e_na, preset.e_k, preset.e_l), reversals);
            for temperature in [6.3, 20.0, 20.001, 37.0] {
                let mut hh = preset.clone();
                hh.temperature = temperature;
                hh.v += 17.0;
                hh.m = 0.7;
                hh.h = 0.2;
                hh.n = 0.4;
                let expected = serde_json::to_value(&hh).unwrap();
                assert_eq!(expected["voltage_convention"], name);
                let mut legacy = expected.clone();
                legacy.as_object_mut().unwrap().remove("voltage_convention");
                let decoded: HodgkinHuxleyNeuron = serde_json::from_value(legacy).unwrap();
                assert_eq!(serde_json::to_value(decoded).unwrap(), expected);
            }
        }
    }

    #[test]
    fn convention_explicit_json_wins_and_round_trips_custom_state() {
        for convention in [
            VoltageConvention::Absolute,
            VoltageConvention::RelativeToRest,
        ] {
            for mut hh in [
                HodgkinHuxleyNeuron::new(),
                HodgkinHuxleyNeuron::new_cortical(),
            ] {
                hh.voltage_convention = convention;
                for custom in [false, true] {
                    if custom {
                        hh.e_na = 123.0;
                        hh.e_k = -23.0;
                        hh.e_l = 4.0;
                    }
                    let wire = serde_json::to_value(&hh).unwrap();
                    let decoded: HodgkinHuxleyNeuron =
                        serde_json::from_value(wire.clone()).unwrap();
                    assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
                }
            }
        }
    }

    #[test]
    fn convention_json_retains_required_and_duplicate_field_validation() {
        let hh = HodgkinHuxleyNeuron::new();
        let serialized = serde_json::to_string(&hh).unwrap();
        let duplicate = serialized.replacen('{', "{\"voltage_convention\":\"absolute\",", 1);
        assert!(serde_json::from_str::<HodgkinHuxleyNeuron>(&duplicate).is_err());
        let duplicate_v = serialized.replacen('{', "{\"v\":0,", 1);
        assert!(serde_json::from_str::<HodgkinHuxleyNeuron>(&duplicate_v).is_err());
        let value = serde_json::to_value(&hh).unwrap();
        for field in [
            "v",
            "m",
            "h",
            "n",
            "e_na",
            "e_k",
            "e_l",
            "g_na",
            "g_k",
            "g_l",
            "c_m",
            "temperature",
        ] {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                serde_json::from_value::<HodgkinHuxleyNeuron>(missing).is_err(),
                "{field}"
            );
        }
        let mut extra = value.clone();
        extra["unrelated_metadata"] = serde_json::json!(true);
        let decoded: HodgkinHuxleyNeuron = serde_json::from_value(extra).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), value);
    }

    #[test]
    fn convention_shifted_custom_state_has_equivalent_dynamics() {
        for temperature in [6.3, 19.999, 20.001, 37.0] {
            let mut relative = HodgkinHuxleyNeuron {
                v: 7.0,
                m: 0.2,
                h: 0.4,
                n: 0.3,
                e_na: 110.0,
                e_k: -15.0,
                e_l: 8.0,
                temperature,
                ..HodgkinHuxleyNeuron::new()
            };
            let mut absolute = relative.clone();
            absolute.voltage_convention = VoltageConvention::Absolute;
            absolute.v -= 65.0;
            absolute.e_na -= 65.0;
            absolute.e_k -= 65.0;
            absolute.e_l -= 65.0;
            let (r_na, r_k, r_l) = relative.ionic_currents();
            let (a_na, a_k, a_l) = absolute.ionic_currents();
            for (r, a) in [r_na, r_k, r_l].into_iter().zip([a_na, a_k, a_l]) {
                assert!((r - a).abs() < 2e-5);
            }
            let (r_v, r_m, r_h, r_n) = relative.rk4_stage1(2.0);
            let (a_v, a_m, a_h, a_n) = absolute.rk4_stage1(2.0);
            for (r, a) in [r_v, r_m, r_h, r_n].into_iter().zip([a_v, a_m, a_h, a_n]) {
                assert!((r - a).abs() < 2e-5);
            }
            relative.step(2.0, 0.137);
            absolute.step(2.0, 0.137);
            assert!((relative.v - (absolute.v + 65.0)).abs() < 2e-5);
            for (r, a) in [relative.m, relative.h, relative.n]
                .into_iter()
                .zip([absolute.m, absolute.h, absolute.n])
            {
                assert!((r - a).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn convention_temperature_only_scales_kinetics() {
        // Independently evaluate the published rates at relative V=7 mV.
        let v = 7.0_f64;
        let rates = [
            0.1 * (25.0 - v) / ((25.0 - v) / 10.0).exp_m1() * 0.8 - 4.0 * (-v / 18.0).exp() * 0.2,
            0.07 * (-v / 20.0).exp() * 0.6 - 0.4 / (((30.0 - v) / 10.0).exp() + 1.0),
            0.01 * (10.0 - v) / ((10.0 - v) / 10.0).exp_m1() * 0.7
                - 0.125 * (-v / 80.0).exp() * 0.3,
        ];
        for mut hh in [
            HodgkinHuxleyNeuron::new(),
            HodgkinHuxleyNeuron::new_cortical(),
        ] {
            hh.v += 7.0;
            hh.m = 0.2;
            hh.h = 0.4;
            hh.n = 0.3;
            let baseline_dv = hh.derivatives(hh.v, hh.m, hh.h, hh.n, 2.0).0;
            for temperature in [6.3, 19.999, 20.0, 20.001, 21.0, 37.0] {
                hh.temperature = temperature;
                let (dv, dm, dh, dn) = hh.derivatives(hh.v, hh.m, hh.h, hh.n, 2.0);
                assert_eq!(dv, baseline_dv);
                let phi = 3.0_f64.powf((f64::from(temperature) - 6.3) / 10.0);
                for (actual, rate) in [dm, dh, dn].into_iter().zip(rates) {
                    assert!(
                        (f64::from(actual) / phi - rate).abs() < 2e-7,
                        "T={temperature}, actual={actual}, unscaled expected={rate}"
                    );
                }
            }
        }
    }

    #[test]
    fn convention_reset_preserves_coordinates_across_temperature_boundary() {
        for preset in [
            HodgkinHuxleyNeuron::new(),
            HodgkinHuxleyNeuron::new_cortical(),
        ] {
            for temperature in [6.3, 19.999, 20.0, 20.001, 21.0, 37.0] {
                let mut hh = preset.clone();
                hh.temperature = temperature;
                let expected = serde_json::to_value(&hh).unwrap();
                hh.v += 15.0;
                hh.m = 0.8;
                hh.h = 0.1;
                hh.n = 0.7;
                hh.reset();
                assert_eq!(serde_json::to_value(&hh).unwrap(), expected);
                // f64 evaluation of alpha/(alpha+beta) at relative zero.
                for (actual, expected) in
                    [hh.m, hh.h, hh.n]
                        .into_iter()
                        .zip([0.0529324853, 0.5961207535, 0.3176769141])
                {
                    assert!((f64::from(actual) - expected).abs() < 1e-7);
                }
            }
        }
    }

    #[test]
    fn convention_rejects_malformed_explicit_json() {
        for bad in [
            serde_json::Value::Null,
            serde_json::json!("cortical"),
            serde_json::json!(3),
            serde_json::json!({}),
        ] {
            let mut wire = serde_json::to_value(HodgkinHuxleyNeuron::new()).unwrap();
            wire["voltage_convention"] = bad;
            assert!(serde_json::from_value::<HodgkinHuxleyNeuron>(wire).is_err());
        }
    }

    #[test]
    fn convention_rejects_ambiguous_legacy_json() {
        for preset in [
            HodgkinHuxleyNeuron::new(),
            HodgkinHuxleyNeuron::new_cortical(),
        ] {
            for field in ["e_na", "e_k", "e_l"] {
                let mut wire = serde_json::to_value(&preset).unwrap();
                wire.as_object_mut().unwrap().remove("voltage_convention");
                let value = wire[field].as_f64().unwrap() as f32;
                wire[field] = serde_json::json!(f32::from_bits(value.to_bits() + 1));
                assert!(serde_json::from_value::<HodgkinHuxleyNeuron>(wire).is_err());
            }
        }
    }

    #[test]
    fn duration_recursive_step_integrates_both_halves_after_early_spike() {
        let duration = 0.01f32 * 131_072.0;
        let mut whole = HodgkinHuxleyNeuron::default();
        let mut halves = whole.clone();

        let first_fired = halves.step(10.0, duration / 2.0);
        assert!(
            first_fired,
            "the first half must exercise early spike retention"
        );
        let midpoint_voltage = halves.v;
        let second_fired = halves.step(10.0, duration / 2.0);
        assert!((halves.v - midpoint_voltage).abs() > 1e-6);

        assert_eq!(whole.step(10.0, duration), first_fired || second_fired);
        assert_eq!(whole.v, halves.v);
        assert_eq!(whole.m, halves.m);
        assert_eq!(whole.h, halves.h);
        assert_eq!(whole.n, halves.n);
    }

    #[test]
    fn duration_large_schedule_halves_exactly_and_terminates() {
        let mut duration = f32::MAX;
        let mut depth = 0;
        while let Some(half) = HodgkinHuxleyNeuron::split_duration(duration) {
            assert!(half > 0.0 && half < duration);
            assert_eq!(2.0 * f64::from(half), f64::from(duration));
            duration = half;
            depth += 1;
            assert!(depth <= 135);
        }
        assert!(duration > 0.0);
        assert!(HodgkinHuxleyNeuron::split_duration(0.1).is_none());
    }

    #[test]
    fn duration_retains_early_spike_and_integrates_after_it() {
        let mut neuron = HodgkinHuxleyNeuron {
            v: -0.001,
            ..HodgkinHuxleyNeuron::default()
        };
        let mut reference = neuron.clone();
        assert!(neuron.step(10.0, 0.02));
        assert!(reference.step(10.0, 0.01));
        assert!(!reference.step(10.0, 0.01));
        assert!((neuron.v - reference.v).abs() < 1e-6);
    }

    #[test]
    fn duration_small_positive_advances_state() {
        let mut neuron = HodgkinHuxleyNeuron::default();
        let before = neuron.v;
        neuron.step(10.0, 0.001);
        assert!(neuron.v > before);
    }

    #[test]
    fn duration_nonmultiple_matches_fine_subdivision() {
        for duration in [0.006, 0.06, 0.137] {
            let mut coarse = HodgkinHuxleyNeuron::default();
            let mut fine = coarse.clone();
            coarse.step(10.0, duration);
            for _ in 0..128 {
                fine.step(10.0, duration / 128.0);
            }
            assert!(
                (coarse.v - fine.v).abs() < 2e-5,
                "duration={duration}, coarse={}, fine={}",
                coarse.v,
                fine.v
            );
            assert!((coarse.m - fine.m).abs() < 2e-5);
            assert!((coarse.h - fine.h).abs() < 2e-5);
            assert!((coarse.n - fine.n).abs() < 2e-5);
        }
    }

    #[test]
    fn duration_nonpositive_and_nonfinite_leave_state_unchanged() {
        for duration in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let original = HodgkinHuxleyNeuron::default();
            let mut neuron = original.clone();
            assert!(!neuron.step(10.0, duration));
            assert_eq!(neuron.v.to_bits(), original.v.to_bits());
            assert_eq!(neuron.m.to_bits(), original.m.to_bits());
            assert_eq!(neuron.h.to_bits(), original.h.to_bits());
            assert_eq!(neuron.n.to_bits(), original.n.to_bits());
        }
    }

    #[test]
    fn test_resting_state_is_stable() {
        let hh = HodgkinHuxleyNeuron::new();
        let (m_ss, h_ss, n_ss) = HodgkinHuxleyNeuron::steady_state_gating(0.0);
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
