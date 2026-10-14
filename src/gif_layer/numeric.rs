use super::GifLayerError;
use crate::{GifParams, NonFiniteClass};

/// One field table keeps constructor, mutable-state, and checkpoint rules aligned.
pub(super) fn invalid_parameter(
    p: &GifParams,
) -> Option<(&'static str, &'static str, NonFiniteClass)> {
    let fields = [
        ("leak", "params.leak is non-finite", p.leak),
        (
            "drive_scale",
            "params.drive_scale is non-finite",
            p.drive_scale,
        ),
        (
            "base_threshold",
            "params.base_threshold is non-finite",
            p.base_threshold,
        ),
        (
            "adaptation_scale",
            "params.adaptation_scale is non-finite",
            p.adaptation_scale,
        ),
        (
            "adaptation_decay",
            "params.adaptation_decay is non-finite",
            p.adaptation_decay,
        ),
        (
            "adaptation_coupling",
            "params.adaptation_coupling is non-finite",
            p.adaptation_coupling,
        ),
        (
            "adaptation_increment",
            "params.adaptation_increment is non-finite",
            p.adaptation_increment,
        ),
        (
            "reset_ratio",
            "params.reset_ratio is non-finite",
            p.reset_ratio,
        ),
    ];
    fields.into_iter().find_map(|(field, detail, value)| {
        NonFiniteClass::classify(value).map(|class| (field, detail, class))
    })
}

/// Reject invalid live parameters without changing their stored values.
pub(super) fn validate_params(params: &GifParams) -> Result<(), GifLayerError> {
    match invalid_parameter(params) {
        Some((field, _, class)) => Err(GifLayerError::NonFiniteParam { field, class }),
        None => Ok(()),
    }
}

/// Report the earliest invalid value, including unused channels and zero-drive weights.
pub(super) fn first_non_finite(values: &[f32]) -> Option<(usize, NonFiniteClass)> {
    values
        .iter()
        .enumerate()
        .find_map(|(index, &value)| NonFiniteClass::classify(value).map(|class| (index, class)))
}

/// Candidate transition: inspect pre-reset values too, so a threshold/reset
/// cannot conceal overflow. Uses the shared GifParams arithmetic.
#[derive(Debug, PartialEq)]
pub(super) struct Transition {
    drive: f32,
    decayed_adaptation: f32,
    integrated_membrane: f32,
    threshold: f32,
    pub membrane: f32,
    pub adaptation: f32,
    pub fired: bool,
}

impl Transition {
    pub(super) fn compute(
        params: &GifParams,
        mut membrane: f32,
        mut adaptation: f32,
        drive: f32,
    ) -> Self {
        params.integrate(&mut membrane, &mut adaptation, drive);
        let decayed_adaptation = adaptation;
        let integrated_membrane = membrane;
        let threshold = params.effective_threshold(adaptation);
        let fired = params.check_for_spike(&mut membrane, &mut adaptation);
        Self {
            drive,
            decayed_adaptation,
            integrated_membrane,
            threshold,
            membrane,
            adaptation,
            fired,
        }
    }

    pub(super) fn drive_is_finite(&self) -> bool {
        self.drive.is_finite()
    }

    pub(super) fn validate(&self, neuron: usize) -> Result<(), GifLayerError> {
        for (stage, value) in [
            ("drive", self.drive),
            ("decayed_adaptation", self.decayed_adaptation),
            ("integrated_membrane", self.integrated_membrane),
            ("threshold", self.threshold),
            ("membrane", self.membrane),
            ("adaptation", self.adaptation),
        ] {
            if let Some(class) = NonFiniteClass::classify(value) {
                return Err(GifLayerError::NumericOverflow {
                    neuron,
                    stage,
                    class,
                });
            }
        }
        Ok(())
    }
}
