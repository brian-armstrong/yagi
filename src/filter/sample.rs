//! Private implementation details for generic filter sample types.

use crate::error::{Error, Result};
use num_complex::{Complex32, ComplexFloat};
use std::f32::consts::PI;

mod private {
    use super::Complex32;

    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for Complex32 {}
}

/// Sample or coefficient type accepted by the generic filters.
///
/// The trait is sealed. It is implemented for `f32` and `Complex32` only.
pub trait Sample: Default + ComplexFloat<Real = f32> + From<f32> + private::Sealed {
    /// Converts a complex intermediate result to this sample type.
    ///
    /// Conversion to `f32` discards the imaginary part.
    fn from_complex32(value: Complex32) -> Self;

    /// Shifts a real half-band prototype coefficient to `center_frequency`.
    ///
    /// The coefficient is scaled by two and modulated by a carrier at
    /// `center_frequency` cycles per sample, evaluated `time_offset` samples
    /// from the center tap. A real result uses a cosine carrier, so its
    /// passband appears at both `center_frequency` and `-center_frequency`. A
    /// complex result uses `exp(j*phase)` and shifts the passband only to
    /// `center_frequency`.
    fn halfband_coefficient(prototype_coefficient: f32, time_offset: f32, center_frequency: f32) -> Self;

    /// Designs a notch filter with `2 * filter_delay + 1` taps.
    ///
    /// The notch is at `center_frequency` cycles per sample, and
    /// `stopband_attenuation` is in dB. Real coefficients notch both
    /// `center_frequency` and `-center_frequency`. Complex coefficients notch
    /// only `center_frequency`.
    ///
    /// # Errors
    ///
    /// Returns an error when `filter_delay` is outside `1..=1000`,
    /// `stopband_attenuation` is nonfinite, zero or negative, or `center_frequency` is
    /// nonfinite or outside `[-0.5, 0.5]`.
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
        if !(-0.5..=0.5).contains(&center_frequency) {
            return Err(Error::Config(format!("notch frequency ({}) out of range [-0.5,0.5]", center_frequency)));
        }
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
