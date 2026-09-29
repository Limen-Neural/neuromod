//! IF neuron -> neuromod `LapicqueNeuron` handoff.
//!
//! # Assumption 5 (forward-Euler IF conventions)
//!
//! A NIR `IF` node is mapped element-wise onto a bank of neuromod
//! [`LapicqueNeuron`]s under these fixed conventions:
//!
//! * **Integration:** forward Euler with `dt = 1`.
//! * **Stimulus:** element `i` receives `stimulus = r[i] * I[i]`, where `I` is
//!   the per-step input current and `r` is the IF node's resistance tensor.
//! * **Initial state:** membrane potential `v = 0`.
//! * **Spiking:** the neuron fires when `v >= threshold`.
//! * **Reset:** hard reset to `v = 0` on spike.
//!
//! neuromod's [`LapicqueNeuron::integrate`] applies
//! `v <- (v + stimulus) * (1 - decay_rate)`; setting `decay_rate = 0` reduces
//! this to exactly `v += stimulus`. That is a pure (non-leaky)
//! integrate-and-fire step, which is precisely the NIR `IF` semantics (`IF` has
//! no `tau`, so there is no leak). The threshold and reset map directly onto
//! [`LapicqueNeuron::threshold`] / [`LapicqueNeuron::base_threshold`] and the
//! hard reset performed by [`LapicqueNeuron::check_for_spike`].
//!
//! Only the `IF` neuron is mapped here. NIR scheduling, the `Affine` / `Conv2d`
//! / `AvgPool2d` / `Flatten` kernels, and graph wiring are downstream adapter
//! concerns (see `docs/neuromod-boundary-matrix.md`); this harness classifies
//! them but does not execute them.

use neuromod::{LapicqueNeuron, NonFiniteClass};
use nir_rs::nodes::If;
use nir_rs::types::TensorData;

use crate::error::{HandoffError, RuntimeCause};

/// A bank of neuromod neurons mapped from a single NIR `IF` node, together with
/// the per-element resistance and the source node shape.
#[derive(Debug, Clone)]
pub struct IfHandoff {
    /// The graph node name this bank was mapped from (used in error messages).
    node: String,
    /// One [`LapicqueNeuron`] per IF element, in row-major order.
    bank: Vec<LapicqueNeuron>,
    /// Per-element resistance `r`, in row-major order.
    r: Vec<f32>,
    /// The (agreed) shape of the IF tensors, outer-to-inner (C-order).
    shape: Vec<usize>,
}

/// Read a tensor payload as a flat row-major `Vec<f32>`.
///
/// Only `f32` payloads are accepted: `LapicqueNeuron` is `f32`-only, and
/// narrowing an arbitrary `f64` NIR parameter to `f32` is lossy. A tiny nonzero
/// `f64` `v_reset` could round to `0.0` and slip past the reset check, and a
/// large `f64` threshold could round enough to shift the spike step. Rather
/// than silently narrow and claim faithful support, `f64` (and integer /
/// boolean) payloads are rejected as an unsupported mapping. The vendored HF
/// fixtures are `f32`, so this rejects only genuinely out-of-scope inputs.
fn tensor_as_f32(
    node: &str,
    field: &str,
    tensor: &nir_rs::Tensor,
) -> Result<Vec<f32>, HandoffError> {
    // Shape product must equal the data length. This holds by Tensor
    // construction, but we require it explicitly for defensiveness.
    let shape_product: usize = tensor.shape().iter().product();
    let data = tensor.data();
    if shape_product != data.len() {
        return Err(HandoffError::UnsupportedMapping {
            node: node.to_owned(),
            type_name: "IF",
            reason: format!(
                "`{field}` shape product {shape_product} != data length {}",
                data.len()
            ),
        });
    }
    match data {
        TensorData::F32(values) => Ok(values.clone()),
        TensorData::F64(_) | TensorData::I64(_) | TensorData::Bool(_) => {
            Err(HandoffError::UnsupportedMapping {
                node: node.to_owned(),
                type_name: "IF",
                reason: format!(
                    "`{field}` has dtype {:?}; only f32 is faithfully supported \
                     (f64 -> f32 narrowing is lossy)",
                    tensor.dtype()
                ),
            })
        }
    }
}

/// Build an [`HandoffError::UnsupportedMapping`] for the named `IF` node.
fn unsupported(node: &str, reason: String) -> HandoffError {
    HandoffError::UnsupportedMapping {
        node: node.to_owned(),
        type_name: "IF",
        reason,
    }
}

/// A parameter tensor's shape must equal `r`'s shape (`shape`).
fn check_shape_matches(
    name: &str,
    field: &str,
    tensor: &nir_rs::Tensor,
    shape: &[usize],
) -> Result<(), HandoffError> {
    if tensor.shape() != shape {
        return Err(unsupported(
            name,
            format!(
                "`{field}` shape {:?} != `r` shape {shape:?}",
                tensor.shape()
            ),
        ));
    }
    Ok(())
}

/// The first `(index, value)` pair whose value is not finite, if any.
fn first_non_finite(values: &[f32]) -> Option<(usize, f32)> {
    values
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, v)| !v.is_finite())
}

/// An absent `v_reset` implies all-zeros; a present one must match `shape` and
/// be entirely zero (only a hard reset to `0` is supported).
fn check_v_reset_all_zero(name: &str, node: &If, shape: &[usize]) -> Result<(), HandoffError> {
    let Some(v_reset_tensor) = node.v_reset.as_ref() else {
        return Ok(());
    };
    check_shape_matches(name, "v_reset", v_reset_tensor, shape)?;
    let v_reset = tensor_as_f32(name, "v_reset", v_reset_tensor)?;
    if let Some((i, value)) = v_reset.iter().copied().enumerate().find(|&(_, v)| v != 0.0) {
        return Err(unsupported(
            name,
            format!("`v_reset[{i}]` = {value} is non-zero; only hard reset to 0 is supported"),
        ));
    }
    Ok(())
}

/// Every resistance value must be finite.
fn check_r_finite(name: &str, r: &[f32]) -> Result<(), HandoffError> {
    if let Some((i, value)) = first_non_finite(r) {
        return Err(unsupported(
            name,
            format!("`r[{i}]` = {value} is non-finite"),
        ));
    }
    Ok(())
}

/// Every threshold must be finite and strictly positive.
fn check_thresholds(name: &str, v_threshold: &[f32]) -> Result<(), HandoffError> {
    if let Some((i, value)) = first_non_finite(v_threshold) {
        return Err(unsupported(
            name,
            format!("`v_threshold[{i}]` = {value} is non-finite"),
        ));
    }
    if let Some((i, value)) = v_threshold
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, v)| v <= 0.0)
    {
        return Err(unsupported(
            name,
            format!("`v_threshold[{i}]` = {value} is non-positive"),
        ));
    }
    Ok(())
}

/// A single non-leaky (`decay_rate = 0`) integrate-and-fire neuron whose
/// threshold and baseline are `threshold`, starting at `v = 0` with no weights.
fn lapicque_if_neuron(threshold: f32) -> LapicqueNeuron {
    let mut neuron = LapicqueNeuron::new();
    neuron.decay_rate = 0.0;
    neuron.threshold = threshold;
    neuron.base_threshold = threshold;
    neuron.membrane_potential = 0.0;
    neuron.weights = Vec::new();
    neuron
}

impl IfHandoff {
    /// Map a NIR `IF` node into a neuromod neuron bank.
    ///
    /// # Errors
    ///
    /// [`HandoffError::UnsupportedMapping`] if:
    /// * the shapes of `r`, `v_threshold`, and (when present) `v_reset` do not
    ///   agree, or a shape product does not equal its data length;
    /// * `r` or `v_threshold` contains a non-finite value;
    /// * any threshold is non-positive (`<= 0`);
    /// * `v_reset` (or, when absent, the implied all-zero reset) contains a
    ///   non-zero element.
    pub fn from_node(name: impl Into<String>, node: &If) -> Result<Self, HandoffError> {
        let name = name.into();

        let r = tensor_as_f32(&name, "r", &node.r)?;
        let v_threshold = tensor_as_f32(&name, "v_threshold", &node.v_threshold)?;
        let shape = node.r.shape().to_vec();

        // Each check is delegated to a small helper so this constructor stays a
        // flat, readable sequence of validations rather than nested branches.
        check_shape_matches(&name, "v_threshold", &node.v_threshold, &shape)?;
        check_v_reset_all_zero(&name, node, &shape)?;
        check_r_finite(&name, &r)?;
        check_thresholds(&name, &v_threshold)?;

        let bank = v_threshold.iter().map(|&t| lapicque_if_neuron(t)).collect();

        Ok(Self {
            node: name,
            bank,
            r,
            shape,
        })
    }

    /// The number of neurons (IF elements) in the bank.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bank.len()
    }

    /// Whether the bank is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bank.is_empty()
    }

    /// The source node name.
    #[must_use]
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Per-element resistance `r`, in row-major order.
    #[must_use]
    pub fn r(&self) -> &[f32] {
        &self.r
    }

    /// The agreed IF tensor shape (outer-to-inner / C-order).
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Read-only view of the mapped neuron bank.
    #[must_use]
    pub fn bank(&self) -> &[LapicqueNeuron] {
        &self.bank
    }

    /// Advance every neuron by one forward-Euler step and return the spike
    /// vector.
    ///
    /// For each element `i`: `integrate(r[i] * inputs[i])`, then
    /// `check_for_spike(t)`. See the module-level Assumption 5 notes.
    ///
    /// The step is **two-phase and atomic**: every stimulus (`r[i] * inputs[i]`)
    /// and the resulting projected membrane potential are computed and
    /// validated for *all* elements before any neuron is mutated. If any
    /// element would fail, the whole step returns an error with the bank left
    /// untouched. This also closes an overflow gap: a finite `r[i] * inputs[i]`
    /// (or projected `v`) that overflows to `±inf` is caught here, before
    /// `check_for_spike` could interpret `+inf` as a spike and hard-reset it to
    /// `0.0`, which would otherwise hide the non-finite value from a
    /// post-mutation check.
    ///
    /// # Errors
    ///
    /// [`HandoffError::Runtime`] if:
    /// * `inputs.len()` does not equal the bank size (checked before any
    ///   mutation);
    /// * an input sample is non-finite (checked before any mutation);
    /// * a stimulus `r[i] * inputs[i]` or the projected membrane potential is
    ///   non-finite, e.g. from overflow (checked before any mutation); or
    /// * a neuron reports a spike but did not hard-reset to `0`.
    pub fn step(&mut self, inputs: &[f32], t: i64) -> Result<Vec<bool>, HandoffError> {
        // Two phases keep the step atomic: validate the whole input first
        // (touching no state), then mutate only once every element is known
        // finite. See each helper for details.
        self.validate_step_inputs(inputs, t)?;
        self.apply_step(inputs, t)
    }

    /// Build the [`HandoffError::Runtime`] this bank reports at step `t`.
    fn runtime_error(&self, index: usize, t: i64, cause: RuntimeCause) -> HandoffError {
        HandoffError::Runtime {
            node: self.node.clone(),
            index,
            step: t,
            cause,
        }
    }

    /// Phase 1: length + per-element finiteness of the input, the stimulus
    /// `r[i] * inputs[i]`, and the projected membrane potential
    /// (`v + stimulus`, since `decay_rate == 0`). Mutates nothing.
    ///
    /// Validating the projected potential here catches an overflow to `±inf`
    /// before phase 2's `check_for_spike` could read it as a spike and reset it
    /// to `0.0`, which would hide the non-finite value.
    fn validate_step_inputs(&self, inputs: &[f32], t: i64) -> Result<(), HandoffError> {
        if inputs.len() != self.bank.len() {
            return Err(self.runtime_error(
                0,
                t,
                RuntimeCause::InputLenMismatch {
                    expected: self.bank.len(),
                    got: inputs.len(),
                },
            ));
        }

        for (i, (&input, neuron)) in inputs.iter().zip(self.bank.iter()).enumerate() {
            let stimulus = self.r[i] * input;
            let projected = neuron.membrane_potential + stimulus;
            let class = NonFiniteClass::classify(input)
                .or_else(|| NonFiniteClass::classify(stimulus))
                .or_else(|| NonFiniteClass::classify(projected));
            if let Some(class) = class {
                return Err(self.runtime_error(i, t, RuntimeCause::NonFinite(class)));
            }
        }
        Ok(())
    }

    /// Phase 2: every input validated finite, so integrate and spike each
    /// neuron, asserting a spiking neuron hard-reset to `0`.
    fn apply_step(&mut self, inputs: &[f32], t: i64) -> Result<Vec<bool>, HandoffError> {
        let mut spikes = Vec::with_capacity(self.bank.len());
        for (i, neuron) in self.bank.iter_mut().enumerate() {
            neuron.integrate(self.r[i] * inputs[i]);
            let fired = neuron.check_for_spike(t);

            let potential = neuron.membrane_potential;
            if fired && potential != 0.0 {
                return Err(self.runtime_error(i, t, RuntimeCause::ResetViolation { potential }));
            }
            spikes.push(fired);
        }
        Ok(spikes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nir_rs::Tensor;

    fn if_node(r: Vec<f32>, v_threshold: Vec<f32>, v_reset: Option<Vec<f32>>) -> If {
        let shape = vec![r.len()];
        If {
            r: Tensor::from_f32(shape.clone(), r).unwrap(),
            v_threshold: Tensor::from_f32(shape.clone(), v_threshold).unwrap(),
            v_reset: v_reset.map(|values| Tensor::from_f32(shape.clone(), values).unwrap()),
            metadata: Default::default(),
        }
    }

    #[test]
    fn maps_if_bank_with_zero_decay() {
        let node = if_node(vec![1.0, 2.0], vec![0.5, 1.0], None);
        let handoff = IfHandoff::from_node("if0", &node).unwrap();
        assert_eq!(handoff.len(), 2);
        assert_eq!(handoff.r(), &[1.0, 2.0]);
        assert_eq!(handoff.shape(), &[2]);
        for (neuron, threshold) in handoff.bank().iter().zip([0.5, 1.0]) {
            assert_eq!(neuron.decay_rate, 0.0);
            assert_eq!(neuron.threshold, threshold);
            assert_eq!(neuron.base_threshold, threshold);
            assert_eq!(neuron.membrane_potential, 0.0);
            assert!(neuron.weights.is_empty());
        }
    }

    #[test]
    fn step_integrates_and_spikes() {
        let node = if_node(vec![1.0], vec![1.0], None);
        let mut handoff = IfHandoff::from_node("if0", &node).unwrap();
        // 0.4 -> below threshold, no spike.
        assert_eq!(handoff.step(&[0.4], 0).unwrap(), vec![false]);
        // +0.7 -> 1.1 >= 1.0, spikes and resets.
        assert_eq!(handoff.step(&[0.7], 1).unwrap(), vec![true]);
        assert_eq!(handoff.bank()[0].membrane_potential, 0.0);
    }

    #[test]
    fn rejects_non_positive_threshold() {
        let node = if_node(vec![1.0], vec![0.0], None);
        assert!(matches!(
            IfHandoff::from_node("if0", &node),
            Err(HandoffError::UnsupportedMapping { .. })
        ));
    }

    #[test]
    fn rejects_non_zero_reset() {
        let node = if_node(vec![1.0], vec![1.0], Some(vec![0.5]));
        assert!(matches!(
            IfHandoff::from_node("if0", &node),
            Err(HandoffError::UnsupportedMapping { .. })
        ));
    }

    /// Step `handoff` with inputs that must fail, asserting the error is a
    /// `Runtime` and no neuron's state was touched. Returns the error for the
    /// caller to inspect the cause.
    fn assert_runtime_step_leaves_bank_untouched(
        handoff: &mut IfHandoff,
        inputs: &[f32],
        t: i64,
    ) -> HandoffError {
        let err = handoff.step(inputs, t).unwrap_err();
        assert!(
            matches!(err, HandoffError::Runtime { .. }),
            "expected Runtime, got {err:?}"
        );
        for neuron in handoff.bank() {
            assert_eq!(neuron.membrane_potential, 0.0, "no mutation on failure");
        }
        err
    }

    #[test]
    fn step_rejects_bad_input_before_mutation() {
        // Wrong-length input: Runtime InputLenMismatch, bank untouched.
        let node = if_node(vec![1.0, 1.0], vec![1.0, 1.0], None);
        let mut handoff = IfHandoff::from_node("if0", &node).unwrap();
        let err = assert_runtime_step_leaves_bank_untouched(&mut handoff, &[0.5], 0);
        assert!(matches!(
            err,
            HandoffError::Runtime {
                cause: RuntimeCause::InputLenMismatch {
                    expected: 2,
                    got: 1
                },
                ..
            }
        ));

        // Non-finite input: Runtime NonFinite(Nan) at the offending step.
        let err = assert_runtime_step_leaves_bank_untouched(&mut handoff, &[f32::NAN, 0.0], 3);
        assert!(matches!(
            err,
            HandoffError::Runtime {
                cause: RuntimeCause::NonFinite(NonFiniteClass::Nan),
                step: 3,
                ..
            }
        ));
    }

    #[test]
    fn rejects_f64_dtype_as_unsupported() {
        // f64 -> f32 narrowing is lossy, so an f64 IF node is not faithfully
        // supported and must be rejected rather than silently cast.
        let node = If {
            r: Tensor::from_f64(vec![1], vec![1.0]).unwrap(),
            v_threshold: Tensor::from_f64(vec![1], vec![1.0]).unwrap(),
            v_reset: None,
            metadata: Default::default(),
        };
        assert!(matches!(
            IfHandoff::from_node("if_f64", &node),
            Err(HandoffError::UnsupportedMapping { .. })
        ));
    }

    #[test]
    fn step_detects_finite_input_overflow_without_false_spike() {
        // A finite resistance times a finite input can overflow to +inf. The
        // two-phase step must catch this as a NonFinite runtime error rather
        // than let check_for_spike see +inf, "spike", and reset to 0.
        let node = if_node(vec![1e30], vec![1.0], None);
        let mut handoff = IfHandoff::from_node("if_of", &node).unwrap();
        let err = handoff.step(&[1e30], 5).unwrap_err();
        assert!(
            matches!(
                err,
                HandoffError::Runtime {
                    cause: RuntimeCause::NonFinite(NonFiniteClass::PosInfinity),
                    step: 5,
                    ..
                }
            ),
            "expected +inf overflow to be a runtime error, got {err:?}"
        );
        // No mutation: the neuron did not "spike" and reset.
        assert_eq!(handoff.bank()[0].membrane_potential, 0.0);
    }

    #[test]
    fn step_is_atomic_across_the_bank() {
        // Element 0 is valid; element 1 overflows. The whole step must fail and
        // leave element 0 unmutated (no partial application).
        let node = if_node(vec![1.0, 1e30], vec![1.0, 1.0], None);
        let mut handoff = IfHandoff::from_node("if_atomic", &node).unwrap();
        let err = handoff.step(&[0.5, 1e30], 0).unwrap_err();
        assert!(matches!(
            err,
            HandoffError::Runtime {
                index: 1,
                cause: RuntimeCause::NonFinite(NonFiniteClass::PosInfinity),
                ..
            }
        ));
        // Element 0 must be untouched despite being validated before element 1.
        assert_eq!(handoff.bank()[0].membrane_potential, 0.0);
        assert_eq!(handoff.bank()[1].membrane_potential, 0.0);
    }
}
