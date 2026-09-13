//! Private implementation details for generic filter sample types.

use crate::error::Result;
use num_complex::{Complex32, ComplexFloat};
use std::f32::consts::PI;

mod private {
    use super::Complex32;

    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for Complex32 {}
}

/// Numeric sample or coefficient type supported by generic filter APIs.
///
/// This trait is sealed. Yagi currently supports `f32` and `Complex32`.
pub trait Sample: Default + ComplexFloat<Real = f32> + From<f32> + private::Sealed {
    /// Convert an FFT-domain processing result to an output sample.
    fn from_complex32(value: Complex32) -> Self;

    /// Modulate a real half-band prototype coefficient for a center frequency.
    fn halfband_coefficient(prototype_coefficient: f32, time_offset: f32, center_frequency: f32) -> Self;

    /// Design coefficients for a notch filter.
    fn notch_coefficients(filter_delay: usize, stopband_attenuation: f32, center_frequency: f32) -> Result<Vec<Self>>;
}

impl Sample for f32 {
    fn from_complex32(value: Complex32) -> Self {
        value.re
    }

    fn halfband_coefficient(prototype_coefficient: f32, time_offset: f32, center_frequency: f32) -> Self {
        2.0 * prototype_coefficient * (2.0 * PI * time_offset * center_frequency).cos()
    }

    fn notch_coefficients(filter_delay: usize, stopband_attenuation: f32, center_frequency: f32) -> Result<Vec<Self>> {
        crate::filter::fir::design::fir_design_notch(filter_delay, center_frequency, stopband_attenuation)
    }
}

impl Sample for Complex32 {
    fn from_complex32(value: Complex32) -> Self {
        value
    }

    fn halfband_coefficient(prototype_coefficient: f32, time_offset: f32, center_frequency: f32) -> Self {
        let phase = 2.0 * PI * time_offset * center_frequency;
        2.0 * prototype_coefficient * Complex32::new(phase.cos(), phase.sin())
    }

    fn notch_coefficients(filter_delay: usize, stopband_attenuation: f32, center_frequency: f32) -> Result<Vec<Self>> {
        let coefficients = crate::filter::fir::design::fir_design_notch(filter_delay, 0.0, stopband_attenuation)?;
        Ok(coefficients
            .into_iter()
            .enumerate()
            .map(|(i, coefficient)| {
                let phase = 2.0 * PI * center_frequency * (i as f32 - filter_delay as f32);
                Complex32::new(coefficient, 0.0) * Complex32::from_polar(1.0, phase)
            })
            .collect())
    }
}
