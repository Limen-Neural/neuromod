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
/// Fixtures are `f32`, but `f64` payloads are accepted and cast to `f32`.
/// Integer / boolean payloads are rejected as an unsupported mapping.
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
        TensorData::F64(values) => Ok(values.iter().map(|&v| v as f32).collect()),
        TensorData::I64(_) | TensorData::Bool(_) => Err(HandoffError::UnsupportedMapping {
            node: node.to_owned(),
            type_name: "IF",
            reason: format!("`{field}` has non-float dtype {:?}", tensor.dtype()),
        }),
    }
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
        let unsupported = |reason: String| HandoffError::UnsupportedMapping {
            node: name.clone(),
            type_name: "IF",
            reason,
        };

        // Shapes must agree across r / v_threshold / v_reset.
        if node.v_threshold.shape() != shape.as_slice() {
            return Err(unsupported(format!(
                "`v_threshold` shape {:?} != `r` shape {shape:?}",
                node.v_threshold.shape()
            )));
        }

        // v_reset: absent means all-zeros; present must match shape and be all-zero.
        if let Some(v_reset_tensor) = node.v_reset.as_ref() {
            if v_reset_tensor.shape() != shape.as_slice() {
                return Err(unsupported(format!(
                    "`v_reset` shape {:?} != `r` shape {shape:?}",
                    v_reset_tensor.shape()
                )));
            }
            let v_reset = tensor_as_f32(&name, "v_reset", v_reset_tensor)?;
            if let Some((i, value)) = v_reset.iter().copied().enumerate().find(|&(_, v)| v != 0.0) {
                return Err(unsupported(format!(
                    "`v_reset[{i}]` = {value} is non-zero; only hard reset to 0 is supported"
                )));
            }
        }

        // Reject non-finite r and non-finite / non-positive thresholds.
        if let Some((i, value)) = r.iter().copied().enumerate().find(|&(_, v)| !v.is_finite()) {
            return Err(unsupported(format!("`r[{i}]` = {value} is non-finite")));
        }
        for (i, &threshold) in v_threshold.iter().enumerate() {
            if !threshold.is_finite() {
                return Err(unsupported(format!(
                    "`v_threshold[{i}]` = {threshold} is non-finite"
                )));
            }
            if threshold <= 0.0 {
                return Err(unsupported(format!(
                    "`v_threshold[{i}]` = {threshold} is non-positive"
                )));
            }
        }

        let bank = v_threshold
            .iter()
            .map(|&threshold| {
                let mut neuron = LapicqueNeuron::new();
                neuron.decay_rate = 0.0;
                neuron.threshold = threshold;
                neuron.base_threshold = threshold;
                neuron.membrane_potential = 0.0;
                neuron.weights = Vec::new();
                neuron
            })
            .collect();

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
    /// # Errors
    ///
    /// [`HandoffError::Runtime`] if:
    /// * `inputs.len()` does not equal the bank size (checked before any
    ///   mutation);
    /// * an input sample is non-finite (checked before any mutation);
    /// * after a step, a membrane potential is non-finite; or
    /// * a neuron reports a spike but did not hard-reset to `0`.
    pub fn step(&mut self, inputs: &[f32], t: i64) -> Result<Vec<bool>, HandoffError> {
        // Length check before touching any neuron state.
        if inputs.len() != self.bank.len() {
            return Err(HandoffError::Runtime {
                node: self.node.clone(),
                index: 0,
                step: t,
                cause: RuntimeCause::InputLenMismatch {
                    expected: self.bank.len(),
                    got: inputs.len(),
                },
            });
        }

        // Scan for non-finite inputs before mutating any neuron state.
        for (i, &input) in inputs.iter().enumerate() {
            if let Some(class) = NonFiniteClass::classify(input) {
                return Err(HandoffError::Runtime {
                    node: self.node.clone(),
                    index: i,
                    step: t,
                    cause: RuntimeCause::NonFinite(class),
                });
            }
        }

        let mut spikes = Vec::with_capacity(self.bank.len());
        for (i, neuron) in self.bank.iter_mut().enumerate() {
            neuron.integrate(self.r[i] * inputs[i]);
            let fired = neuron.check_for_spike(t);

            let potential = neuron.membrane_potential;
            if let Some(class) = NonFiniteClass::classify(potential) {
                return Err(HandoffError::Runtime {
                    node: self.node.clone(),
                    index: i,
                    step: t,
                    cause: RuntimeCause::NonFinite(class),
                });
            }
            if fired && potential != 0.0 {
                return Err(HandoffError::Runtime {
                    node: self.node.clone(),
                    index: i,
                    step: t,
                    cause: RuntimeCause::ResetViolation { potential },
                });
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

    #[test]
    fn step_rejects_length_mismatch_before_mutation() {
        let node = if_node(vec![1.0, 1.0], vec![1.0, 1.0], None);
        let mut handoff = IfHandoff::from_node("if0", &node).unwrap();
        let err = handoff.step(&[0.5], 0).unwrap_err();
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
        // No mutation happened.
        assert_eq!(handoff.bank()[0].membrane_potential, 0.0);
    }

    #[test]
    fn step_rejects_non_finite_input() {
        let node = if_node(vec![1.0], vec![1.0], None);
        let mut handoff = IfHandoff::from_node("if0", &node).unwrap();
        let err = handoff.step(&[f32::NAN], 3).unwrap_err();
        assert!(matches!(
            err,
            HandoffError::Runtime {
                cause: RuntimeCause::NonFinite(NonFiniteClass::Nan),
                step: 3,
                ..
            }
        ));
        assert_eq!(handoff.bank()[0].membrane_potential, 0.0);
    }
}
