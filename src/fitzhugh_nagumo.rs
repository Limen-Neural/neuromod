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
        Self {
            v: v0,
            w: w0,
            epsilon,
            a,
            b,
        }
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
        Self {
            v: v0 + 0.1,
            w: w0,
            epsilon,
            a,
            b,
        }
    }

    /// Neuron with stronger adaptation (higher ε → faster recovery).
    pub fn new_adaptive() -> Self {
        let a = 0.7;
        let b = 0.5;
        let epsilon = 0.12;
        let (v0, w0) = Self::resting_state(a, b, 0.0);
        Self {
            v: v0,
            w: w0,
            epsilon,
            a,
            b,
        }
    }

    /// Select a nullcline intersection with zero-start Newton, then bisection.
    /// The selected root need not be unique, nearest zero, or stable.
    fn resting_state(a: f32, b: f32, i_app: f32) -> (f32, f32) {
        let invalid = (f32::NAN, f32::NAN);
        if !a.is_finite() || !b.is_finite() || !i_app.is_finite() {
            return invalid;
        }
        let (a, b, i_app) = (f64::from(a), f64::from(b), f64::from(i_app));
        if b == 0.0 {
            return Self::resting_candidate(a, b, i_app, -a).unwrap_or(invalid);
        }
        let p = 1.0 / b - 1.0;
        let q = a / b - i_app;
        Self::newton_resting_voltage(p, q)
            .and_then(|v| Self::resting_candidate(a, b, i_app, v))
            .or_else(|| Self::bracketed_resting_state(a, b, i_app, p, q))
            .unwrap_or(invalid)
    }

    fn resting_candidate(a: f64, b: f64, i_app: f64, v: f64) -> Option<(f32, f32)> {
        let cubic_w = v - v * v * v / 3.0 + i_app;
        let rounded_v = v as f32;
        if Self::valid_resting_pair(a, b, i_app, rounded_v, cubic_w as f32) {
            return Some((rounded_v, cubic_w as f32));
        }
        // Near an outer cubic root, cancellation can erase a representable
        // recovery value. Try the recovery nullcline, still checking both ODEs.
        if b != 0.0 {
            let recovery_w = ((v + a) / b) as f32;
            if Self::valid_resting_pair(a, b, i_app, rounded_v, recovery_w) {
                return Some((rounded_v, recovery_w));
            }
        }
        None
    }

    fn valid_resting_pair(a: f64, b: f64, i_app: f64, v: f32, w: f32) -> bool {
        if !v.is_finite() || !w.is_finite() {
            return false;
        }
        // A valid f64 intersection can still overflow the implemented f32
        // right-hand sides. Preserve their evaluation order when checking it.
        let voltage_rhs = v - v * v * v / 3.0 - w + i_app as f32;
        let recovery_rhs = v + a as f32 - b as f32 * w;
        if !voltage_rhs.is_finite() || !recovery_rhs.is_finite() {
            return false;
        }
        let (v, w) = (f64::from(v), f64::from(w));
        let cubic = v * v * v / 3.0;
        let rv = v - cubic - w + i_app;
        let rw = v + a - b * w;
        let tolerance = 16.0 * f64::from(f32::EPSILON);
        // Validate both original equations after rounding, without epsilon
        // scaling: epsilon=0 must not hide a missed nullcline intersection.
        rv.abs() <= tolerance * (v.abs() + cubic.abs() + w.abs() + i_app.abs())
            && rw.abs() <= tolerance * (v.abs() + a.abs() + (b * w).abs())
    }

    fn resting_residual(v: f64, p: f64, q: f64) -> f64 {
        v * v * v / 3.0 + p * v + q
    }

    fn resting_converged(v: f64, p: f64, q: f64, f: f64) -> bool {
        f.is_finite()
            && (f == 0.0
                || f.abs()
                    <= 32.0 * f64::EPSILON * ((v * v * v / 3.0).abs() + (p * v).abs() + q.abs()))
    }

    fn newton_resting_voltage(p: f64, q: f64) -> Option<f64> {
        let mut v = 0.0;
        for _ in 0..50 {
            let f = Self::resting_residual(v, p, q);
            // In particular, keep the exact middle root when q=0.
            if Self::resting_converged(v, p, q, f) {
                return Some(v);
            }
            let derivative = v * v + p;
            if derivative == 0.0 || !derivative.is_finite() {
                return None;
            }
            let next = v - f / derivative;
            if !next.is_finite() || next == v {
                return None;
            }
            v = next;
        }
        None
    }

    fn bracketed_resting_state(a: f64, b: f64, i_app: f64, p: f64, q: f64) -> Option<(f32, f32)> {
        // At this radius the cubic dominates both remaining terms, giving
        // opposite endpoint signs even when Newton starts at a zero derivative.
        let radius = 1.0 + (6.0 * p.abs()).sqrt().max((6.0 * q.abs()).cbrt());
        let (mut low, mut high) = (-radius, radius);
        let (f_low, f_high) = (
            Self::resting_residual(low, p, q),
            Self::resting_residual(high, p, q),
        );
        if !f_low.is_finite() || !f_high.is_finite() || f_low > 0.0 || f_high < 0.0 {
            return None;
        }
        for (endpoint, f) in [(low, f_low), (high, f_high)] {
            if f == 0.0
                && let Some(result) = Self::resting_candidate(a, b, i_app, endpoint)
            {
                return Some(result);
            }
        }
        for _ in 0..512 {
            let middle = low + (high - low) / 2.0;
            let f = Self::resting_residual(middle, p, q);
            if Self::resting_converged(middle, p, q, f)
                && let Some(result) = Self::resting_candidate(a, b, i_app, middle)
            {
                return Some(result);
            }
            // A small cubic residual alone is insufficient: keep refining
            // after a rounded candidate misses either original nullcline.
            if middle == low || middle == high {
                break;
            }
            if f < 0.0 {
                low = middle;
            } else {
                high = middle;
            }
        }
        None
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
    /// Covers the full finite positive `dt` with sub-steps no larger than 0.05.
    /// Zero, negative, and non-finite durations return `false` without mutation.
    /// Runtime scales with duration; extremely large durations are impractical.
    /// A crossing in any sub-step is retained in the returned result.
    pub fn step(&mut self, i_app: f32, dt: f32) -> bool {
        if !dt.is_finite() || dt <= 0.0 {
            return false;
        }
        if let Some(half) = Self::split_duration(dt) {
            let first = self.step(i_app, half);
            let second = self.step(i_app, half);
            return first || second;
        }
        let mut remaining = f64::from(dt);

        let mut fired = false;
        let v_threshold: f32 = 1.0;

        while remaining > 0.0 {
            let sub_dt = remaining.min(f64::from(0.05f32)) as f32;
            remaining -= f64::from(sub_dt);
            let v_before = self.v;
            let half = sub_dt / 2.0;

            let (k1_v, k1_w) = (
                self.dv_dt(self.v, self.w, i_app),
                self.dw_dt(self.v, self.w),
            );
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

    // Keep subtraction within f64's exact range for f32 durations. Large
    // intervals are halved exactly, with at most 135 recursive stack frames;
    // unlike a cast step count or repeated subtraction from a huge float,
    // this cannot overflow or stop making progress.
    fn split_duration(duration: f32) -> Option<f32> {
        (duration > 0.05f32 * 65_536.0).then_some(duration * 0.5)
    }

    /// Reset to a selected zero-input nullcline intersection.
    ///
    /// Uses Newton iteration starting at zero with a bracketed fallback. With
    /// multiple roots, the selected intersection is not necessarily stable.
    /// Supports `b = 0` directly. Nonfinite `a`/`b`, unrepresentable results,
    /// failure to validate both nullclines, or nonfinite evaluation of the
    /// implemented `f32` voltage/unscaled recovery right-hand sides set both
    /// `v` and `w` to NaN. The intersection does not depend on `epsilon`.
    /// This does not guarantee numerical stability for arbitrary parameters,
    /// inputs, or timesteps.
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
    ///
    /// For `b = 0`, the nullcline is the vertical line `v = -a`; this graph
    /// representation is undefined and retains floating-point division by zero.
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
        if spike_count == 0 {
            None
        } else {
            Some(spike_count as f32 / total_time)
        }
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

    fn assert_equilibrium(a: f32, b: f32, current: f32, v: f32, w: f32) {
        assert!(
            v.is_finite() && w.is_finite(),
            "nonfinite equilibrium: {v}, {w}"
        );
        let (a, b, current, v, w) = (
            f64::from(a),
            f64::from(b),
            f64::from(current),
            f64::from(v),
            f64::from(w),
        );
        let cubic = v.powi(3) / 3.0;
        let rv = v - cubic - w + current;
        let rw = v + a - b * w;
        let tolerance = 16.0 * f64::from(f32::EPSILON);
        assert!(
            rv.abs() <= tolerance * (v.abs() + cubic.abs() + w.abs() + current.abs()),
            "voltage residual {rv}"
        );
        assert!(
            rw.abs() <= tolerance * (v.abs() + a.abs() + (b * w).abs()),
            "recovery residual {rw}"
        );
    }

    #[test]
    fn resting_degenerate_and_adjacent_derivatives_find_intersection() {
        for b in [
            1.0,
            f32::from_bits(1.0f32.to_bits() - 1),
            f32::from_bits(1.0f32.to_bits() + 1),
        ] {
            let mut neuron = FitzHughNagumoNeuron {
                b,
                epsilon: 0.0,
                ..Default::default()
            };
            neuron.reset();
            assert_equilibrium(neuron.a, b, 0.0, neuron.v, neuron.w);
            assert!((f64::from(neuron.v) - (-3.0 * f64::from(neuron.a)).cbrt()).abs() < 2e-6);
        }
    }

    #[test]
    fn resting_preserves_exact_zero_branch() {
        for b in [1.0, 20.0] {
            let mut neuron = FitzHughNagumoNeuron {
                a: 0.0,
                b,
                ..Default::default()
            };
            neuron.reset();
            assert_eq!((neuron.v, neuron.w), (0.0, 0.0));
        }
    }

    #[test]
    fn resting_zero_b_and_nonzero_current_match_analytic_roots() {
        for current in [0.0, 0.2] {
            for b in [0.0, 1.0] {
                let a = 0.7;
                let (v, w) = FitzHughNagumoNeuron::resting_state(a, b, current);
                assert_equilibrium(a, b, current, v, w);
                let expected = if b == 0.0 {
                    -f64::from(a)
                } else {
                    (3.0 * (f64::from(current) - f64::from(a))).cbrt()
                };
                assert!((f64::from(v) - expected).abs() < 2e-7);
            }
        }
    }

    #[test]
    fn resting_constructor_references_and_oscillatory_perturbation() {
        for (mut neuron, v, w, perturbation) in [
            (
                FitzHughNagumoNeuron::default(),
                -1.199408031928253,
                -0.6242600455092783,
                0.0,
            ),
            (
                FitzHughNagumoNeuron::new_adaptive(),
                -1.0327898582069097,
                -0.6655797402556773,
                0.0,
            ),
            (
                FitzHughNagumoNeuron::new_oscillatory(),
                0.19743464012202916,
                0.19486927726382605,
                0.1,
            ),
        ] {
            assert!((f64::from(neuron.v) - v - perturbation).abs() < 2e-7);
            assert!((f64::from(neuron.w) - w).abs() < 2e-7);
            neuron.reset();
            assert!((f64::from(neuron.v) - v).abs() < 2e-7);
            assert_equilibrium(neuron.a, neuron.b, 0.0, neuron.v, neuron.w);
        }
    }

    #[test]
    fn resting_rejects_nonfinite_and_unrepresentable_results() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for (a, b, current) in [
                (invalid, 0.8, 0.0),
                (0.7, invalid, 0.0),
                (0.7, 0.8, invalid),
            ] {
                let (v, w) = FitzHughNagumoNeuron::resting_state(a, b, current);
                assert!(v.is_nan() && w.is_nan());
            }
        }
        for (a, b) in [(f32::MAX, 0.0), (f32::INFINITY, 0.8)] {
            let mut neuron = FitzHughNagumoNeuron {
                a,
                b,
                ..Default::default()
            };
            neuron.reset();
            assert!(neuron.v.is_nan() && neuron.w.is_nan());
            assert!(!neuron.is_excitable());
        }
    }

    #[test]
    fn resting_rejects_equilibrium_that_overflows_implemented_voltage_derivative() {
        let mut neuron = FitzHughNagumoNeuron {
            a: -8.0e12,
            b: 0.0,
            ..Default::default()
        };
        neuron.reset();
        assert!(neuron.v.is_nan() && neuron.w.is_nan());
    }

    #[test]
    fn resting_large_representable_equilibrium_survives_positive_step() {
        let mut neuron = FitzHughNagumoNeuron {
            a: -1_099_511_627_776.0, // -2^40: its cube remains representable in f32.
            b: 0.0,
            ..Default::default()
        };
        neuron.reset();
        assert_equilibrium(neuron.a, neuron.b, 0.0, neuron.v, neuron.w);
        assert_eq!(neuron.v, -neuron.a);
        assert!(!neuron.step(0.0, 0.01));
        assert!(neuron.v.is_finite() && neuron.w.is_finite());
        assert_equilibrium(neuron.a, neuron.b, 0.0, neuron.v, neuron.w);
    }

    #[test]
    fn resting_finite_coefficients_with_unrepresentable_recovery_fail_closed() {
        // Recovery requires w=(v+MAX)/MIN_POSITIVE. Any representable w
        // forces v near -MAX, whose voltage-nullcline value is unrepresentable.
        let mut neuron = FitzHughNagumoNeuron {
            a: f32::MAX,
            b: f32::MIN_POSITIVE,
            ..Default::default()
        };
        neuron.reset();
        assert!(neuron.v.is_nan() && neuron.w.is_nan());
        assert!(!neuron.is_excitable());
    }

    #[test]
    fn resting_large_negative_b_retains_a_representable_intersection() {
        let mut neuron = FitzHughNagumoNeuron {
            a: -3.009_265_5e-36,
            b: -34_359_738_368.0,
            ..Default::default()
        };
        neuron.reset();
        assert_equilibrium(neuron.a, neuron.b, 0.0, neuron.v, neuron.w);
        // For negligible a, the outer roots approach ±sqrt(3*(1-1/b)).
        assert!((f64::from(neuron.v) - (-1.732_050_807_594_082_3)).abs() < 2e-6);
        assert!((f64::from(neuron.w) - 5.040_931_304_666_685e-11).abs() < 2e-16);
    }

    #[test]
    fn resting_large_positive_b_avoids_cubic_recovery_cancellation() {
        let mut neuron = FitzHughNagumoNeuron {
            a: -1.036_468e-32,
            b: 1.652_763_9e38,
            ..Default::default()
        };
        neuron.reset();
        assert_equilibrium(neuron.a, neuron.b, 0.0, neuron.v, neuron.w);
        assert!((f64::from(neuron.v).abs() - 3.0f64.sqrt()).abs() < 2e-6);
        // Either outer root gives a representable subnormal near ±sqrt(3)/b.
        assert!((f64::from(neuron.w).abs() - 1.047_972_34e-38).abs() < 2e-44);
    }

    #[test]
    fn resting_newton_cycle_falls_back_to_valid_intersection() {
        // 3*F(v)=v^3-2*v+2 makes zero-start Newton cycle between 0 and 1.
        // Independent bisection reference for its sole real root.
        let (v, w) = FitzHughNagumoNeuron::resting_state(2.0, 3.0, 0.0);
        assert_equilibrium(2.0, 3.0, 0.0, v, w);
        assert!((f64::from(v) - (-1.7692923542386314)).abs() < 2e-7);
    }

    #[test]
    fn resting_handles_negative_b_and_small_coefficients() {
        for (a, b) in [
            (0.7, -0.8),
            (f32::MIN_POSITIVE, 1.0),
            (0.7, f32::MIN_POSITIVE),
        ] {
            let (v, w) = FitzHughNagumoNeuron::resting_state(a, b, 0.0);
            assert_equilibrium(a, b, 0.0, v, w);
        }
    }

    #[test]
    fn duration_recursive_step_integrates_both_halves_after_early_spike() {
        let duration = 0.05f32 * 131_072.0;
        let mut whole = FitzHughNagumoNeuron::default();
        let mut halves = whole.clone();

        let first_fired = halves.step(0.7, duration / 2.0);
        assert!(
            first_fired,
            "the first half must exercise early spike retention"
        );
        let midpoint_voltage = halves.v;
        let second_fired = halves.step(0.7, duration / 2.0);
        assert!((halves.v - midpoint_voltage).abs() > 1e-6);

        assert_eq!(whole.step(0.7, duration), first_fired || second_fired);
        assert_eq!(whole.v, halves.v);
        assert_eq!(whole.w, halves.w);
    }

    #[test]
    fn duration_large_schedule_halves_exactly_and_terminates() {
        let mut duration = f32::MAX;
        let mut depth = 0;
        while let Some(half) = FitzHughNagumoNeuron::split_duration(duration) {
            assert!(half > 0.0 && half < duration);
            assert_eq!(2.0 * f64::from(half), f64::from(duration));
            duration = half;
            depth += 1;
            assert!(depth <= 135);
        }
        assert!(duration > 0.0);
        assert!(FitzHughNagumoNeuron::split_duration(0.1).is_none());
    }

    #[test]
    fn duration_retains_early_spike_and_integrates_after_it() {
        let mut neuron = FitzHughNagumoNeuron {
            v: 0.99,
            w: 0.0,
            ..FitzHughNagumoNeuron::default()
        };
        let mut reference = neuron.clone();
        assert!(neuron.step(0.7, 0.1));
        assert!(reference.step(0.7, 0.05));
        assert!(!reference.step(0.7, 0.05));
        assert!((neuron.v - reference.v).abs() < 1e-6);
    }

    #[test]
    fn duration_small_positive_advances_state() {
        let mut neuron = FitzHughNagumoNeuron::default();
        let before = neuron.v;
        neuron.step(0.7, 0.01);
        assert!(neuron.v > before);
    }

    #[test]
    fn duration_nonmultiple_matches_fine_subdivision() {
        for duration in [0.006, 0.06, 0.137] {
            let mut coarse = FitzHughNagumoNeuron::default();
            let mut fine = coarse.clone();
            coarse.step(0.7, duration);
            for _ in 0..128 {
                fine.step(0.7, duration / 128.0);
            }
            assert!(
                (coarse.v - fine.v).abs() < 2e-5,
                "duration={duration}, coarse={}, fine={}",
                coarse.v,
                fine.v
            );
            assert!((coarse.w - fine.w).abs() < 2e-5);
        }
    }

    #[test]
    fn duration_nonpositive_and_nonfinite_leave_state_unchanged() {
        for duration in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let original = FitzHughNagumoNeuron::default();
            let mut neuron = original.clone();
            assert!(!neuron.step(0.7, duration));
            assert_eq!(neuron.v.to_bits(), original.v.to_bits());
            assert_eq!(neuron.w.to_bits(), original.w.to_bits());
        }
    }

    #[test]
    fn test_resting_state_is_stable_without_input() {
        let mut fhn = FitzHughNagumoNeuron::new();
        for _ in 0..1000 {
            fhn.step(0.0, 0.5);
        }
        let (v_ss, w_ss) = FitzHughNagumoNeuron::resting_state(fhn.a, fhn.b, 0.0);
        assert!(
            (fhn.v - v_ss).abs() < 0.1,
            "V should stay near resting state"
        );
        assert!(
            (fhn.w - w_ss).abs() < 0.1,
            "W should stay near resting state"
        );
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
        for _ in 0..2000 {
            fhn.step(0.1, 0.5);
        }
        assert!(
            fhn.v < 1.0,
            "Neuron should remain subthreshold with weak input"
        );
    }

    #[test]
    fn test_reset_restores_state() {
        let mut fhn = FitzHughNagumoNeuron::new();
        for _ in 0..5000 {
            fhn.step(1.0, 0.5);
        }
        fhn.reset();
        let (v0, w0) = FitzHughNagumoNeuron::resting_state(fhn.a, fhn.b, 0.0);
        assert!(
            (fhn.v - v0).abs() < 1e-6,
            "After reset, V should return to resting state"
        );
        assert!(
            (fhn.w - w0).abs() < 1e-6,
            "After reset, W should return to resting state"
        );
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
        assert!(
            rate_high > rate_low,
            "Higher input should produce higher firing rate"
        );
    }

    #[test]
    fn test_nullclines_intersect_at_fixed_point() {
        let fhn = FitzHughNagumoNeuron::new();
        let (v_fp, w_fp) = FitzHughNagumoNeuron::resting_state(fhn.a, fhn.b, 0.0);
        let v_nc_fp = fhn.v_nullcline(v_fp, 0.0);
        assert!(
            (v_nc_fp - w_fp).abs() < 1e-6,
            "Nullclines should intersect at the fixed point"
        );
    }

    #[test]
    fn test_excitable_regime_detection() {
        let excitable = FitzHughNagumoNeuron::new();
        assert!(excitable.is_excitable(), "Default FHN should be excitable");

        let oscillatory = FitzHughNagumoNeuron::new_oscillatory();
        assert!(
            !oscillatory.is_excitable(),
            "Oscillatory FHN should not be excitable"
        );
    }
}
