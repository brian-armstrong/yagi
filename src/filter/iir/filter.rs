#![allow(clippy::excessive_precision)]
use super::design;
use super::second_order_section::IirSecondOrderSection;
use crate::buffer::Window;
use crate::dotprod::{DotProd, DotProduct};
use crate::error::{Error, Result};
use crate::filter::Sample;
use core::f32;
use num_complex::{Complex32, ComplexFloat};

// References:
//  [Pintelon:1990] Rik Pintelon and Johan Schoukens, "Real-Time
//      Integration and Differentiation of Analog Signals by Means of
//      Digital Filtering," IEEE Transactions on Instrumentation and
//      Measurement, vol 39 no. 6, December 1990.

/// IIR filter type (Normal or Second-Order Sections)
#[derive(Debug, Clone, Copy, PartialEq)]
enum IirFilterType {
    Norm,
    Sos,
}

/// Designs coefficients for a prototype IIR filter.
///
/// # Arguments
///
/// - `order`: filter order. Must be nonzero.
/// - `cutoff_frequency`: in cycles per sample, in `(0, 0.5)`. It marks an
///   approximately -3 dB point for Butterworth and Bessel, the passband edge
///   for Chebyshev I and elliptic, and the stopband edge for Chebyshev II.
/// - `center_frequency`: in cycles per sample, in `[0, 0.5]`. Only band-pass
///   and band-stop designs use it, but it is range-checked for every design.
/// - `passband_ripple`: in dB. Must be finite and positive. Used by Chebyshev I
///   and elliptic, but checked for every shape.
/// - `stopband_attenuation`: in dB. Must be finite and positive. Used by
///   Chebyshev II and elliptic, but checked for every shape. For elliptic
///   designs, it must exceed `passband_ripple`.
///
/// For band-pass and band-stop designs, `cutoff_frequency` is one band edge and
/// `center_frequency` sets the other, as described by
/// [`iir_design`](crate::filter::iir::design::iir_design).
///
/// The result holds the numerator coefficients, the denominator coefficients,
/// and the number of second-order sections. The section count is zero when
/// `format` selects a transfer function. Coefficients are in ascending powers of
/// `z^-1`, with three per section in the cascade form.
///
/// # Errors
///
/// Returns an error when `order` is zero, a frequency or ripple parameter is out
/// of range, or a design step fails.
///
/// # Panics
///
/// Panics when a very large `order` overflows the coefficient-length arithmetic
/// and overflow checks are enabled. Without overflow checks, the wrapped length
/// gives a failed or invalid design.
pub fn iir_filter_design_prototype<Coeff>(
    filter_shape: design::IirFilterShape,
    band_type: design::IirBandType,
    format: design::IirFormat,
    order: usize,
    cutoff_frequency: f32,
    center_frequency: f32,
    passband_ripple: f32,
    stopband_attenuation: f32,
) -> Result<(Vec<Coeff>, Vec<Coeff>, usize)>
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    // filter length
    let mut n = order;

    if band_type == design::IirBandType::Bandpass || band_type == design::IirBandType::Bandstop {
        n *= 2;
    }

    let r = n % 2; // odd/even order
    let l = (n - r) / 2; // filter semi-length

    let h_len = if format == design::IirFormat::SecondOrderSections { 3 * (l + r) } else { n + 1 };
    let mut b = vec![0.0; h_len];
    let mut a = vec![0.0; h_len];

    design::iir_design(
        filter_shape,
        band_type,
        format,
        order,
        cutoff_frequency,
        center_frequency,
        passband_ripple,
        stopband_attenuation,
        &mut b,
        &mut a,
    )?;

    let b = b.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    let a = a.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

    let nsos = if format == design::IirFormat::SecondOrderSections { l + r } else { 0 };

    Ok((b, a, nsos))
}

/// Designs a low-pass Butterworth filter as second-order sections.
///
/// `fc` is the approximately -3 dB point in cycles per sample and must lie in
/// `(0, 0.5)`. The result has the layout described by
/// [`iir_filter_design_prototype`].
///
/// # Errors
///
/// Returns an error when `order` is zero, `fc` is outside `(0, 0.5)`, or a
/// design step fails.
pub fn iir_filter_design_lowpass<Coeff>(order: usize, fc: f32) -> Result<(Vec<Coeff>, Vec<Coeff>, usize)>
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    iir_filter_design_prototype(
        design::IirFilterShape::Butter,
        design::IirBandType::Lowpass,
        design::IirFormat::SecondOrderSections,
        order,
        fc,
        0.0,
        0.1,
        60.0,
    )
}

/// Designs a fixed eighth-order integrating filter.
///
/// The result is four second-order sections, taken from the digital integrator
/// of Pintelon and Schoukens (1990). One pole sits at `z = 1`, so a sustained DC
/// component in the input makes the output grow without bound.
pub fn iir_filter_design_integrator<Coeff>() -> (Vec<Coeff>, Vec<Coeff>, usize)
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    // integrator digital zeros/poles/gain, [Pintelon:1990] Table II
    //
    // zeros, digital, integrator
    #[rustfmt::skip]
    let zdi = [
        Complex32::from(-1.175839),
        3.371020 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -125.1125),
        3.371020 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  125.1125),
        4.549710 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -80.96404),
        4.549710 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   80.96404),
        5.223966 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -40.09347),
        5.223966 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   40.09347),
        Complex32::from(5.443743),
    ];
    // poles, digital, integrator
    #[rustfmt::skip]
    let pdi = [
        Complex32::from(-0.5805235),
        0.2332021 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -114.0968),
        0.2332021 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  114.0968),
        0.1814755 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -66.33969),
        0.1814755 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   66.33969),
        0.1641457 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -21.89539),
        0.1641457 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   21.89539),
        Complex32::from(1.0),
    ];
    // gain, digital, integrator (slight adjustment added for proper gain)
    let kdi = Complex32::from(-1.89213380759321e-05 / 0.9695401191711425781);

    // second-order sections
    let mut b = vec![0.0; 12];
    let mut a = vec![0.0; 12];
    design::iir_design_d2sos(&zdi, &pdi, kdi, &mut b, &mut a)
        .expect("built-in integrator roots form four second-order sections");

    let b = b.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    let a = a.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

    (b, a, 4)
}

/// Designs a fixed eighth-order differentiating filter.
///
/// The result is four second-order sections, taken from the digital
/// differentiator of Pintelon and Schoukens (1990). One zero sits at `z = 1`, so
/// the filter rejects DC.
pub fn iir_filter_design_differentiator<Coeff>() -> (Vec<Coeff>, Vec<Coeff>, usize)
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    // differentiator digital zeros/poles/gain, [Pintelon:1990] Table IV
    //
    // zeros, digital, differentiator
    #[rustfmt::skip]
    let zdd = [
        Complex32::from(-1.702575),
        5.877385 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -221.4063),
        5.877385 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  221.4063),
        4.197421 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -144.5972),
        4.197421 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  144.5972),
        5.350284 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -66.88802),
        5.350284 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   66.88802),
        Complex32::from(1.0),
    ];
    #[rustfmt::skip]
    let pdd = [
        Complex32::from(-0.8476936),
        0.2990781 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -125.5188),
        0.2990781 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  125.5188),
        0.2232427 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -81.52326),
        0.2232427 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   81.52326),
        0.1958670 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -40.51510),
        0.1958670 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   40.51510),
        Complex32::from(0.1886088),
    ];
    // gain, digital, differentiator (slight adjustment added for proper gain)
    let kdd = Complex32::from(2.09049284907492e-05 / 1.033477783203125000);

    // second-order sections
    let mut b = vec![0.0; 12];
    let mut a = vec![0.0; 12];
    design::iir_design_d2sos(&zdd, &pdd, kdd, &mut b, &mut a)
        .expect("built-in differentiator roots form four second-order sections");

    let b = b.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    let a = a.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

    (b, a, 4)
}

/// Designs a first-order DC-blocking filter.
///
/// The result is a transfer function with numerator `[1, -1]` and denominator
/// `[1, alpha - 1]`, so the section count is zero.
///
/// ```text
///          1 -          z^-1
///  H(z) = ------------------
///          1 - (1-alpha)z^-1
/// ```
///
/// The pole sits at `1 - alpha`. A small `alpha` gives a narrow notch around DC.
/// The filter is stable only for `0 < alpha < 2`. At `alpha == 2` the pole is on
/// the unit circle, and larger values are unstable. Only nonfinite, zero and
/// negative values are rejected. Use [`iir_filter_design_dc_blocker_scale`] for the
/// matching output scale.
///
/// # Errors
///
/// Returns an error when `alpha` is nonfinite, zero or negative.
pub fn iir_filter_design_dc_blocker<Coeff>(alpha: f32) -> Result<(Vec<Coeff>, Vec<Coeff>, usize)>
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Config("DC-blocking filter bandwidth must be finite and greater than zero".into()));
    }

    let bf = [1.0, -1.0];
    let af = [1.0, -1.0 + alpha];

    let b = bf.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    let a = af.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

    Ok((b, a, 0))
}

/// Computes the output scale for [`iir_filter_design_dc_blocker`].
///
/// The scale is `sqrt(1 - alpha)`. It is zero at `alpha == 1`, and any larger
/// `alpha` gives a `NaN` scale. Neither case is rejected, so only
/// `0 < alpha < 1` gives a usable scale.
///
/// # Errors
///
/// Returns an error when `alpha` is nonfinite, zero or negative.
pub fn iir_filter_design_dc_blocker_scale<Coeff>(alpha: f32) -> Result<Coeff>
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Config("DC-blocking filter bandwidth must be finite and greater than zero".into()));
    }

    Ok(Coeff::from((1.0 - alpha).sqrt()).unwrap())
}

/// Designs a second-order active-lag loop filter for a phase-locked loop.
///
/// `natural_frequency` is in radians per sample, and `damping_factor` is
/// dimensionless. A damping factor of `1/sqrt(2)` is a common choice. The result
/// is one second-order section.
///
/// # Errors
///
/// Returns an error when `natural_frequency`, `damping_factor`, or `loop_gain`
/// is nonfinite, zero or negative.
pub fn iir_filter_design_pll<Coeff>(
    natural_frequency: f32,
    damping_factor: f32,
    loop_gain: f32,
) -> Result<(Vec<Coeff>, Vec<Coeff>, usize)>
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    if !natural_frequency.is_finite() || natural_frequency <= 0.0 {
        return Err(Error::Config("PLL natural frequency must be finite and greater than zero".into()));
    }
    if !damping_factor.is_finite() || damping_factor <= 0.0 {
        return Err(Error::Config("PLL damping factor must be finite and greater than zero".into()));
    }
    if !loop_gain.is_finite() || loop_gain <= 0.0 {
        return Err(Error::Config("PLL loop gain must be finite and greater than zero".into()));
    }

    let mut bf = [0.0; 3];
    let mut af = [0.0; 3];
    design::iir_design_pll_active_lag(natural_frequency, damping_factor, loop_gain, &mut bf, &mut af)?;

    let b = bf.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    let a = af.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

    Ok((b, a, 1))
}

/// Evaluates a transfer function at `fc` cycles per sample, including `scale`.
///
/// Coefficients are in ascending powers of `z^-1`. The result is
/// `scale * B / A`, where `B = sum(b[i] * exp(-j*2*pi*fc*i))` and `A` is
/// the same sum over `a`.
///
/// Neither `fc` nor the coefficient lengths are checked. A zero denominator sum
/// gives a nonfinite response.
pub fn iir_filter_freqresponse<Coeff>(b: &[Coeff], a: &[Coeff], scale: Coeff, fc: f32) -> Complex32
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    let mut hb = Complex32::default();
    let mut ha = Complex32::default();

    for (i, &bi) in b.iter().enumerate() {
        hb += bi.into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * i as f32);
    }

    for (i, &ai) in a.iter().enumerate() {
        ha += ai.into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * i as f32);
    }

    let h = hb / ha;

    h * scale.into()
}

/// Evaluates a cascade of second-order sections at `fc` cycles per sample,
/// including `scale`.
///
/// `b` and `a` hold three coefficients per section in ascending powers of
/// `z^-1`. The result is the product of the section responses. With empty
/// slices the result is `scale`. A zero section denominator gives a
/// nonfinite response.
///
/// # Panics
///
/// Panics when the coefficient slices differ in length or their length is not
/// a multiple of three.
pub fn iir_filter_freqresponse_sos<Coeff>(b: &[Coeff], a: &[Coeff], scale: Coeff, fc: f32) -> Complex32
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    assert_eq!(b.len(), a.len(), "section coefficient lengths must match");
    assert!(b.len().is_multiple_of(3), "section coefficient length must be divisible by three");

    let mut h;

    h = Complex32::from(1.0);

    for i in 0..b.len() / 3 {
        let hb = b[3 * i].into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * 0.0)
            + b[3 * i + 1].into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * 1.0)
            + b[3 * i + 2].into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * 2.0);

        let ha = a[3 * i].into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * 0.0)
            + a[3 * i + 1].into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * 1.0)
            + a[3 * i + 2].into() * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * fc * 2.0);

        h *= hb / ha;
    }

    h * scale.into()
}

/// Evaluates the power response of a transfer function in dB, including `scale`.
///
/// `fc` is in cycles per sample. The result is `10 * log10(|H|^2)`, where `H` is
/// the response from [`iir_filter_freqresponse`]. A response null gives negative
/// infinity, and a zero denominator gives a nonfinite result.
pub fn iir_filter_get_psd<Coeff>(b: &[Coeff], a: &[Coeff], scale: Coeff, fc: f32) -> f32
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    let h = iir_filter_freqresponse(b, a, scale, fc);
    10.0 * (h * h.conj()).re.log10()
}

/// Evaluates the power response of a second-order-section cascade in dB,
/// including `scale`.
///
/// `fc` is in cycles per sample. The result is `10 * log10(|H|^2)`, where `H` is
/// the response from [`iir_filter_freqresponse_sos`]. A response null gives
/// negative infinity.
///
/// # Panics
///
/// Panics when the coefficient slices differ in length or their length is not
/// a multiple of three.
pub fn iir_filter_get_psd_sos<Coeff>(b: &[Coeff], a: &[Coeff], scale: Coeff, fc: f32) -> f32
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    let h = iir_filter_freqresponse_sos(b, a, scale, fc);
    10.0 * (h * h.conj()).re.log10()
}

/// Estimates the group delay of a transfer function in samples.
///
/// `fc` is in cycles per sample. Only the real parts of the coefficients are
/// used, so the estimate does not apply to a filter with complex coefficients.
/// Near a response null the estimate falls back to zero, as described by
/// [`iir_group_delay`](crate::filter::iir::design::iir_group_delay).
///
/// # Errors
///
/// Returns an error when `b` or `a` is empty, or when `fc` is nonfinite or outside
/// `[-0.5, 0.5]`.
pub fn iir_filter_groupdelay<Coeff>(b: &[Coeff], a: &[Coeff], fc: f32) -> Result<f32>
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    let mut b_re = vec![0.0; b.len()];
    let mut a_re = vec![0.0; a.len()];
    for i in 0..b.len() {
        b_re[i] = b[i].re();
    }
    for i in 0..a.len() {
        a_re[i] = a[i].re();
    }
    let groupdelay = design::iir_group_delay(&b_re, &a_re, fc)?;

    Ok(groupdelay)
}

/// Estimates the group delay of a second-order-section cascade in samples.
///
/// `fc` is in cycles per sample. `b` and `a` hold three coefficients per
/// section, and the estimate is the sum of the section delays. Only the real
/// parts of the coefficients are used. With empty slices the result is zero, and
/// `fc` is not checked.
///
/// # Errors
///
/// Returns an error when `fc` is nonfinite or outside `[-0.5, 0.5]`, or when
/// the coefficient slices differ in length or their length is not a multiple
/// of three.
pub fn iir_filter_groupdelay_sos<Coeff>(b: &[Coeff], a: &[Coeff], fc: f32) -> Result<f32>
where
    Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    if b.len() != a.len() || !b.len().is_multiple_of(3) {
        return Err(Error::Config("section coefficient lengths must match and be divisible by three".into()));
    }
    let mut groupdelay = 0.0;

    for i in 0..b.len() / 3 {
        let b_re: [f32; 3] = [b[3 * i].re(), b[3 * i + 1].re(), b[3 * i + 2].re()];
        let a_re: [f32; 3] = [a[3 * i].re(), a[3 * i + 1].re(), a[3 * i + 2].re()];

        groupdelay += design::iir_group_delay(&b_re, &a_re, fc)?;
    }

    Ok(groupdelay)
}

/// Infinite impulse response filter.
///
/// The filter runs in one of two forms. A transfer-function filter, built with
/// [`Self::new`], applies one numerator and one denominator directly. A cascade,
/// built with [`Self::new_sos`], runs a chain of [`IirSecondOrderSection`]s. The
/// design constructors pick the form for you.
///
/// Every output is multiplied by [`Self::scale`], which starts at one. A new
/// filter starts with zero-valued history. Use [`Self::execute`] to process one
/// sample at a time, or [`Self::execute_block`] to process many.
#[derive(Debug, Clone)]
pub struct IirFilter<T, Coeff = T> {
    b: Vec<Coeff>,             // numerator (feed-forward coefficients)
    dpb: DotProduct<T, Coeff>, // numerator dot product
    a: Vec<Coeff>,             // denominator (feed-back coefficients)
    dpa: DotProduct<T, Coeff>, // denominator dot product
    v: Window<T>,              // internal filter state (buffer)
    n: usize,                  // filter length (order+1)
    nb: usize,                 // numerator length
    na: usize,                 // denominator length

    filter_type: IirFilterType,

    qsos: Vec<IirSecondOrderSection<T, Coeff>>, // second-order sections filters

    scale: Coeff, // output scaling factor
}

impl<T, Coeff> IirFilter<T, Coeff>
where
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    Coeff: Sample + std::ops::Mul<T, Output = T> + Into<Complex32>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a transfer-function filter.
    ///
    /// Coefficients are in ascending powers of `z^-1`, and the numerator and
    /// denominator may have different lengths. Both are normalized by the first
    /// denominator coefficient. A zero there is not rejected and makes every
    /// coefficient nonfinite.
    ///
    /// High-order transfer functions can be numerically fragile in `f32`. Use
    /// [`Self::new_sos`] when second-order sections are available.
    ///
    /// # Errors
    ///
    /// Returns an error when either coefficient slice is empty.
    pub fn new(numerator_coefficients: &[Coeff], denominator_coefficients: &[Coeff]) -> Result<Self> {
        let b = numerator_coefficients;
        let a = denominator_coefficients;

        if b.is_empty() {
            return Err(Error::Config("numerator length cannot be zero".into()));
        }
        if a.is_empty() {
            return Err(Error::Config("denominator length cannot be zero".into()));
        }

        let nb = b.len();
        let na = a.len();
        let n = na.max(nb);

        let mut filter = IirFilter {
            b: b.to_vec(),
            dpb: DotProduct::new_rev(b)?,
            a: a.to_vec(),
            dpa: DotProduct::new_rev(a)?,
            v: Window::new(n)?,
            n,
            nb,
            na,
            filter_type: IirFilterType::Norm,
            qsos: Vec::new(),
            scale: Coeff::one(),
        };

        // normalize coefficients to a[0]
        let a0 = filter.a[0];
        for i in 0..filter.nb {
            filter.b[i] = filter.b[i] / a0;
        }
        for i in 0..filter.na {
            filter.a[i] = filter.a[i] / a0;
        }
        filter.dpb.set_coefficients_rev(&filter.b)?;
        filter.dpa.set_coefficients_rev(&filter.a)?;

        filter.reset();
        Ok(filter)
    }

    /// Creates a cascade of second-order sections.
    ///
    /// Each slice holds three coefficients per section, in ascending powers of
    /// `z^-1`. Section `i` uses elements `3 * i` through `3 * i + 2` of each slice,
    /// and the sections run in that order. Each section is normalized by its own
    /// first denominator coefficient. A zero there is not rejected and makes that
    /// section's coefficients nonfinite.
    ///
    /// # Errors
    ///
    /// Returns an error when the slices are empty, differ in length, or have a
    /// length that is not a multiple of three.
    pub fn new_sos(numerator_coefficients: &[Coeff], denominator_coefficients: &[Coeff]) -> Result<Self> {
        if numerator_coefficients.is_empty() {
            return Err(Error::Config("filter must have at least one second-order section".into()));
        }
        if numerator_coefficients.len() != denominator_coefficients.len() {
            return Err(Error::Config(format!(
                "numerator and denominator coefficient lengths must match (got {} and {})",
                numerator_coefficients.len(),
                denominator_coefficients.len()
            )));
        }
        if !numerator_coefficients.len().is_multiple_of(3) {
            return Err(Error::Config(format!(
                "second-order-section coefficient length must be divisible by 3 (got {})",
                numerator_coefficients.len()
            )));
        }

        let num_sections = numerator_coefficients.len() / 3;
        let b = numerator_coefficients;
        let a = denominator_coefficients;

        let mut filter = IirFilter::<T, Coeff> {
            b: b.to_vec(),
            dpb: DotProduct::new_rev(b)?,
            a: a.to_vec(),
            dpa: DotProduct::new_rev(a)?,
            v: Window::new(1)?,
            n: num_sections * 2,
            nb: 0,
            na: 0,
            filter_type: IirFilterType::Sos,
            qsos: Vec::with_capacity(num_sections),
            scale: Coeff::one(),
        };

        for i in 0..num_sections {
            let bt = [b[3 * i], b[3 * i + 1], b[3 * i + 2]];
            let at = [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
            filter.qsos.push(IirSecondOrderSection::<T, Coeff>::new(&bt, &at)?);
        }

        filter.set_scale(Coeff::one());
        Ok(filter)
    }

    /// Creates a filter from a prototype design.
    ///
    /// The parameters mean the same as in [`iir_filter_design_prototype`]. `format`
    /// chooses between a transfer function and a cascade of second-order sections.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`iir_filter_design_prototype`].
    ///
    /// # Panics
    ///
    /// Panics under the same overflow conditions as
    /// [`iir_filter_design_prototype`].
    pub fn new_prototype(
        filter_shape: design::IirFilterShape,
        band_type: design::IirBandType,
        format: design::IirFormat,
        order: usize,
        cutoff_frequency: f32,
        center_frequency: f32,
        passband_ripple: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        // filter length
        let (b, a, _num_sections) = iir_filter_design_prototype(
            filter_shape,
            band_type,
            format,
            order,
            cutoff_frequency,
            center_frequency,
            passband_ripple,
            stopband_attenuation,
        )?;

        let filter = if format == design::IirFormat::SecondOrderSections {
            IirFilter::<T, Coeff>::new_sos(&b, &a)?
        } else {
            IirFilter::<T, Coeff>::new(&b, &a)?
        };
        Ok(filter)
    }

    /// Creates a low-pass Butterworth filter as a cascade of second-order sections.
    ///
    /// `cutoff_frequency` is the approximately -3 dB point in cycles per sample and
    /// must lie in `(0, 0.5)`.
    ///
    /// # Errors
    ///
    /// Returns an error when `order` is zero, `cutoff_frequency` is outside
    /// `(0, 0.5)`, or a design step fails.
    pub fn new_lowpass(order: usize, cutoff_frequency: f32) -> Result<Self> {
        let (b, a, _num_sections) = iir_filter_design_lowpass(order, cutoff_frequency)?;
        Self::new_sos(&b, &a)
    }

    /// Creates a fixed eighth-order integrator as four second-order sections.
    ///
    /// One pole sits at `z = 1`, so a sustained DC component in the input makes the
    /// output grow without bound. See [`iir_filter_design_integrator`] for the
    /// design.
    ///
    pub fn new_integrator() -> Self {
        let (b, a, _num_sections) = iir_filter_design_integrator();
        Self::new_sos(&b, &a).expect("built-in integrator coefficients form valid sections")
    }

    /// Creates a fixed eighth-order differentiator as four second-order sections.
    ///
    /// One zero sits at `z = 1`, so the filter rejects DC. See
    /// [`iir_filter_design_differentiator`] for the design.
    ///
    pub fn new_differentiator() -> Self {
        let (b, a, _num_sections) = iir_filter_design_differentiator();
        Self::new_sos(&b, &a).expect("built-in differentiator coefficients form valid sections")
    }

    /// Creates a first-order DC-blocking filter with its output scale set.
    ///
    /// ```text
    ///          1 -          z^-1
    ///  H(z) = ------------------
    ///          1 - (1-alpha)z^-1
    /// ```
    ///
    /// The pole sits at `1 - alpha`, and a small `alpha` gives a narrow notch around
    /// DC. The output scale is `sqrt(1 - alpha)`. That scale is zero at
    /// `alpha == 1` and `NaN` above it, and neither case is rejected. Only
    /// `0 < alpha < 1` gives a usable filter.
    ///
    /// # Errors
    ///
    /// Returns an error when `alpha` is nonfinite, zero or negative.
    pub fn new_dc_blocker(alpha: f32) -> Result<Self> {
        let (b, a, _nsos) = iir_filter_design_dc_blocker(alpha)?;
        let scale = iir_filter_design_dc_blocker_scale(alpha)?;
        let mut filter = Self::new(&b, &a)?;
        filter.set_scale(scale);
        Ok(filter)
    }

    /// Creates a second-order active-lag loop filter for a phase-locked loop.
    ///
    /// `natural_frequency` is in radians per sample, and `damping_factor` is
    /// dimensionless. The filter is a single second-order section, so
    /// [`Self::set_coefficients`] can update it later.
    ///
    /// # Errors
    ///
    /// Returns an error when `natural_frequency`, `damping_factor`, or `loop_gain`
    /// is nonfinite, zero or negative.
    pub fn new_pll(natural_frequency: f32, damping_factor: f32, loop_gain: f32) -> Result<Self> {
        let (b, a, _num_sections) = iir_filter_design_pll(natural_frequency, damping_factor, loop_gain)?;
        Self::new_sos(&b, &a)
    }

    /// Clears the filter history to zeros.
    ///
    /// The coefficients and output scale are unchanged.
    pub fn reset(&mut self) {
        if self.filter_type == IirFilterType::Sos {
            for sos in &mut self.qsos {
                sos.reset();
            }
        } else {
            self.v.reset();
        }
    }

    /// Sets the factor applied to every output sample.
    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale;
    }

    /// Returns the factor applied to every output sample.
    pub fn scale(&self) -> Coeff {
        self.scale
    }

    /// Replaces the coefficients without clearing the filter history.
    ///
    /// For a transfer-function filter, both slices must have the same lengths as the
    /// current coefficients. The new coefficients are normalized by the first
    /// denominator coefficient. A zero there is not rejected and makes every
    /// coefficient nonfinite.
    ///
    /// A cascade with one section also accepts new coefficients. It uses the first
    /// three elements of each slice and ignores the rest. A cascade with more than
    /// one section cannot be updated.
    ///
    /// # Errors
    ///
    /// Returns an error when a transfer-function filter receives slices of the wrong
    /// length, a single-section cascade receives a slice shorter than three, or
    /// the cascade has more than one section. The filter is unchanged on error.
    pub fn set_coefficients(
        &mut self,
        numerator_coefficients: &[Coeff],
        denominator_coefficients: &[Coeff],
    ) -> Result<()> {
        let b = numerator_coefficients;
        let a = denominator_coefficients;

        if self.filter_type == IirFilterType::Sos {
            if self.qsos.len() != 1 {
                return Err(Error::Config("cannot update a cascade with more than one section".into()));
            }
            if b.len() < 3 || a.len() < 3 {
                return Err(Error::Config(
                    "a second-order section requires three numerator and denominator coefficients".into(),
                ));
            }
            self.qsos[0].set_coefficients(&[b[0], b[1], b[2]], &[a[0], a[1], a[2]])?;
            self.b.copy_from_slice(&b[..3]);
            self.a.copy_from_slice(&a[..3]);
            return Ok(());
        }

        if b.len() != self.nb || a.len() != self.na {
            return Err(Error::Config("coefficient vector lengths must match filter order".into()));
        }

        self.b = b.to_vec();
        self.a = a.to_vec();

        let a0 = self.a[0];
        for i in 0..self.nb {
            self.b[i] = self.b[i] / a0;
        }
        for i in 0..self.na {
            self.a[i] = self.a[i] / a0;
        }
        self.dpb.set_coefficients_rev(&self.b)?;
        self.dpa.set_coefficients_rev(&self.a)?;
        Ok(())
    }

    /// Returns copies of the stored numerator and denominator coefficients.
    ///
    /// For a transfer-function filter, they are normalized by the first denominator
    /// coefficient. For a cascade, they are the section coefficients as passed to
    /// [`Self::new_sos`] or [`Self::set_coefficients`], three per section and not
    /// normalized.
    pub fn coeffs(&self) -> (Vec<Coeff>, Vec<Coeff>) {
        (self.b.clone(), self.a.clone())
    }

    // execute normal iir filter using traditional numerator/denominator
    // form (not second-order sections form)
    //  _x      :   input sample
    //  _y      :   output sample
    fn execute_norm(&mut self, x: T) -> T {
        // advance buffer

        self.v.push(T::default());

        // compute new v[0]
        // n - na accounts for unequal numerator/denominator sizes
        // if they're the same, this starts at index 0, so no effect
        let v0 = self.dpa.execute(&self.v.read()[self.n - self.na..]);
        self.v.set(self.n - 1, x - v0);

        let y = self.dpb.execute(&self.v.read()[self.n - self.nb..]);
        y * self.scale
    }

    #[inline]
    fn execute_sos_const<const NSOS: usize>(qsos: &mut [IirSecondOrderSection<T, Coeff>], mut x: T) -> T {
        let qsos: &mut [IirSecondOrderSection<T, Coeff>; NSOS] = qsos.try_into().unwrap();
        for sos in qsos {
            x = sos.execute(x);
        }
        x
    }

    #[inline]
    fn execute_sos_dynamic(qsos: &mut [IirSecondOrderSection<T, Coeff>], mut x: T) -> T {
        for sos in qsos {
            x = sos.execute(x);
        }
        x
    }

    // execute filter using second-order sections form
    //  _x      :   input sample
    //  _y      :   output sample
    fn execute_sos(&mut self, x: T) -> T {
        let y = match self.qsos.len() {
            1 => Self::execute_sos_const::<1>(&mut self.qsos, x),
            2 => Self::execute_sos_const::<2>(&mut self.qsos, x),
            3 => Self::execute_sos_const::<3>(&mut self.qsos, x),
            4 => Self::execute_sos_const::<4>(&mut self.qsos, x),
            5 => Self::execute_sos_const::<5>(&mut self.qsos, x),
            6 => Self::execute_sos_const::<6>(&mut self.qsos, x),
            7 => Self::execute_sos_const::<7>(&mut self.qsos, x),
            8 => Self::execute_sos_const::<8>(&mut self.qsos, x),
            _ => Self::execute_sos_dynamic(&mut self.qsos, x),
        };
        y * self.scale
    }

    /// Filters one sample and returns one output sample.
    pub fn execute(&mut self, input: T) -> T {
        match self.filter_type {
            IirFilterType::Norm => self.execute_norm(input),
            IirFilterType::Sos => self.execute_sos(input),
        }
    }

    fn execute_sos_block_const<const NSOS: usize>(&mut self, x: &[T], y: &mut [T]) {
        let qsos: &mut [IirSecondOrderSection<T, Coeff>; NSOS] = self.qsos.as_mut_slice().try_into().unwrap();
        for (x_s, y_s) in x.iter().zip(y.iter_mut()) {
            let mut value = *x_s;
            for sos in qsos.iter_mut() {
                value = sos.execute(value);
            }
            *y_s = value * self.scale;
        }
    }

    fn execute_sos_block_dynamic(&mut self, x: &[T], y: &mut [T]) {
        for (x_s, y_s) in x.iter().zip(y.iter_mut()) {
            *y_s = Self::execute_sos_dynamic(&mut self.qsos, *x_s) * self.scale;
        }
    }

    /// Filters a block of samples and returns `input.len()`.
    ///
    /// `output` must be at least as long as `input`, and extra output is left
    /// unchanged. The results match calling [`Self::execute`] on each sample in
    /// turn.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The filter and
    /// `output` are unchanged on error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output block length must be at least the input length".into()));
        }

        match self.filter_type {
            IirFilterType::Norm => {
                for (x_s, y_s) in input.iter().zip(output.iter_mut()) {
                    *y_s = self.execute_norm(*x_s);
                }
            }
            IirFilterType::Sos => match self.qsos.len() {
                1 => self.execute_sos_block_const::<1>(input, output),
                2 => self.execute_sos_block_const::<2>(input, output),
                3 => self.execute_sos_block_const::<3>(input, output),
                4 => self.execute_sos_block_const::<4>(input, output),
                5 => self.execute_sos_block_const::<5>(input, output),
                6 => self.execute_sos_block_const::<6>(input, output),
                7 => self.execute_sos_block_const::<7>(input, output),
                8 => self.execute_sos_block_const::<8>(input, output),
                _ => self.execute_sos_block_dynamic(input, output),
            },
        }

        Ok(input.len())
    }

    /// Returns the filter length.
    ///
    /// The meaning differs by form. For a transfer-function filter, it is the longer
    /// of the two coefficient counts, which is the order plus one. For a cascade, it
    /// is twice the number of sections.
    pub fn length(&self) -> usize {
        self.n
    }

    /// Evaluates the frequency response at `fc` cycles per sample, including the
    /// output scale.
    pub fn freqresponse(&self, fc: f32) -> Complex32 {
        if self.filter_type == IirFilterType::Norm {
            iir_filter_freqresponse(&self.b, &self.a, self.scale, fc)
        } else {
            iir_filter_freqresponse_sos(&self.b, &self.a, self.scale, fc)
        }
    }

    /// Evaluates the power response in dB at `fc` cycles per sample, including the
    /// output scale.
    ///
    /// A response null gives negative infinity.
    pub fn psd(&self, fc: f32) -> f32 {
        if self.filter_type == IirFilterType::Norm {
            iir_filter_get_psd(&self.b, &self.a, self.scale, fc)
        } else {
            iir_filter_get_psd_sos(&self.b, &self.a, self.scale, fc)
        }
    }

    /// Estimates the group delay in samples at `fc` cycles per sample.
    ///
    /// The output scale and the imaginary parts of the coefficients are ignored.
    ///
    /// # Errors
    ///
    /// Returns an error when `fc` is nonfinite or outside `[-0.5, 0.5]`.
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        if self.filter_type == IirFilterType::Norm {
            iir_filter_groupdelay(&self.b, &self.a, fc)
        } else {
            iir_filter_groupdelay_sos(&self.b, &self.a, fc)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::SpectralPeriodogram;
    use crate::random::randnf;
    use crate::utility::test_helpers::{validate_block_execute, validate_psd_spectrum, PsdRegion};
    use approx::assert_abs_diff_eq;
    use design::{IirBandType, IirFilterShape, IirFormat};
    use test_macro::autotest_annotate;

    #[test]
    fn test_iirfilt_freqresponse_delay_phase() {
        // Three samples of delay: H(1/4) = exp(-j*3*pi/2) = j.
        let b = [0.0, 0.0, 0.0, 1.0];
        let a = [1.0];
        let b_sos = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let a_sos = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut q = IirFilter::<f32, f32>::new(&b, &a).unwrap();
        let mut q_sos = IirFilter::<f32, f32>::new_sos(&b_sos, &a_sos).unwrap();
        q.set_scale(2.0);
        q_sos.set_scale(2.0);

        for fc in [-0.25, 0.25] {
            for h in [
                iir_filter_freqresponse(&b, &a, 2.0, fc),
                iir_filter_freqresponse_sos(&b_sos, &a_sos, 2.0, fc),
                q.freqresponse(fc),
                q_sos.freqresponse(fc),
            ] {
                assert_abs_diff_eq!(h.re, 0.0, epsilon = 1e-6);
                assert_abs_diff_eq!(h.im, 8.0 * fc, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_iirfilt_freqresponse_complex_power() {
        // H(z) = 1 / ((1 - 0.5j*z^-1) * (1 - 0.25*z^-1)).
        // The first factor gives gains of 2 and 2/3 at +1/4 and -1/4.
        let one = Complex32::new(1.0, 0.0);
        let zero = Complex32::new(0.0, 0.0);
        let b = [one];
        let a = [one, Complex32::new(-0.25, -0.5), Complex32::new(0.0, 0.125)];
        let b_sos = [one, zero, zero, one, zero, zero];
        let a_sos = [one, Complex32::new(0.0, -0.5), zero, one, Complex32::new(-0.25, 0.0), zero];
        let scale = Complex32::new(0.75, -0.25);

        for (fc, gain) in [(-0.25, 2.0 / 3.0), (0.25, 2.0)] {
            let expected = scale * gain / Complex32::new(1.0, fc);
            let expected_psd = 10.0 * expected.norm_sqr().log10();
            for h in [
                iir_filter_freqresponse(&b, &a, scale, fc),
                iir_filter_freqresponse_sos(&b_sos, &a_sos, scale, fc),
            ] {
                assert_abs_diff_eq!(h.re, expected.re, epsilon = 1e-6);
                assert_abs_diff_eq!(h.im, expected.im, epsilon = 1e-6);
            }
            assert_abs_diff_eq!(iir_filter_get_psd(&b, &a, scale, fc), expected_psd, epsilon = 1e-5);
            assert_abs_diff_eq!(iir_filter_get_psd_sos(&b_sos, &a_sos, scale, fc), expected_psd, epsilon = 1e-5);

            for mut q in [
                IirFilter::<Complex32, Complex32>::new(&b, &a).unwrap(),
                IirFilter::<Complex32, Complex32>::new_sos(&b_sos, &a_sos).unwrap(),
            ] {
                q.set_scale(scale);
                let h = q.freqresponse(fc);
                assert_abs_diff_eq!(h.re, expected.re, epsilon = 1e-6);
                assert_abs_diff_eq!(h.im, expected.im, epsilon = 1e-6);
                assert_abs_diff_eq!(q.psd(fc), expected_psd, epsilon = 1e-5);

                // Check the reported response against steady-state filtering.
                let tone = [one, Complex32::new(0.0, 1.0), -one, Complex32::new(0.0, -1.0)];
                for i in 0..64 {
                    let x = if fc > 0.0 { tone[i % 4] } else { tone[i % 4].conj() };
                    let y = q.execute(x);
                    if i >= 32 {
                        let measured = y / x;
                        assert_abs_diff_eq!(measured.re, expected.re, epsilon = 1e-5);
                        assert_abs_diff_eq!(measured.im, expected.im, epsilon = 1e-5);
                    }
                }
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_integrator)]
    fn test_iirfilt_integrator() {
        // options
        let num_ones = 10;
        let num_samples = 40;

        // allocate memory for data arrays
        let mut buf_0 = vec![0.0f32; num_samples]; // filter input
        let mut buf_1 = vec![0.0f32; num_samples]; // filter output

        // generate input signal
        for i in 0..num_samples {
            buf_0[i] = if i < num_ones { 1.0 } else { 0.0 };
        }

        // create integrator and run on sample data
        let mut q = IirFilter::<f32, f32>::new_integrator();
        q.execute_block(&buf_0, &mut buf_1).unwrap();

        // check that last value matches expected
        assert_abs_diff_eq!(buf_1[num_samples - 1], num_ones as f32, epsilon = 0.01);
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_differentiator)]
    fn test_iirfilt_differentiator() {
        // options
        let num_samples = 400;

        // allocate memory for data arrays
        let mut buf_0 = vec![0.0f32; num_samples]; // filter input
        let mut buf_1 = vec![0.0f32; num_samples]; // filter output

        // generate input signal
        for i in 0..num_samples {
            buf_0[i] = i as f32;
        }

        // create differentiator and run on sample data
        let mut q = IirFilter::<f32, f32>::new_differentiator();
        q.execute_block(&buf_0, &mut buf_1).unwrap();

        // check that derivative is equal to 1
        assert_abs_diff_eq!(buf_1[num_samples - 1], 1.0f32, epsilon = 0.01);
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_dcblock)]
    fn test_iirfilt_dcblock() {
        // options
        let n = 400000; // number of output samples to analyze
        let alpha = 0.2f32; // forgetting factor
        let nfft = 1200; // number of bins in transform
        let tol = 0.7f32; // error tolerance [dB]

        // create base object
        let mut filter = IirFilter::<Complex32, f32>::new_dc_blocker(alpha).unwrap();

        // create and configure objects
        let mut q = SpectralPeriodogram::new(nfft, crate::math::WindowType::Hann, nfft / 2, nfft / 4).unwrap();

        // start running input through filter
        for _ in 0..n {
            // generate noise input
            let x = Complex32::new(randnf(), randnf()) * f32::consts::FRAC_1_SQRT_2;

            // apply filter
            let y = filter.execute(x);
            // run samples through the spgram object
            q.push(y);
        }

        // verify result
        let psd = q.psd();
        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: -0.500, fmax: -0.200, pmin: -tol, pmax:   tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin: -0.002, fmax:  0.002, pmin: -tol, pmax: -20.0, test_lo: false, test_hi: true },
            PsdRegion { fmin:  0.200, fmax:  0.500, pmin: -tol, pmax:   tol, test_lo:  true, test_hi: true },
        ];
        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    fn testbench_iirfilt_copy(format: IirFormat) {
        // create base object
        let mut q0 = IirFilter::<Complex32, f32>::new_prototype(
            IirFilterShape::Ellip,
            IirBandType::Lowpass,
            format,
            9,
            0.2,
            0.0,
            0.1,
            60.0,
        )
        .unwrap();

        // start running input through filter
        let num_samples = 80;
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let _y0 = q0.execute(v);
        }

        // copy filter
        let mut q1 = q0.clone();

        // continue running through both filters
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let y0 = q0.execute(v);
            let y1 = q1.execute(v);

            // compare result
            assert_eq!(y0, y1);
        }

        // Rust automatically handles destruction of objects when they go out of scope
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_copy_tf)]
    fn test_iirfilt_copy_tf() {
        testbench_iirfilt_copy(IirFormat::TransferFunction);
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_copy_sos)]
    fn test_iirfilt_copy_sos() {
        testbench_iirfilt_copy(IirFormat::SecondOrderSections);
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_config)]
    fn test_iirfilt_config() {
        // test copying/creating invalid objects
        assert!(IirFilter::<Complex32, f32>::new(&[], &[]).is_err());
        assert!(IirFilter::<Complex32, f32>::new(&[0.0], &[]).is_err());
        assert!(IirFilter::<Complex32, f32>::new(&[], &[1.0]).is_err());
        assert!(IirFilter::<Complex32, f32>::new_sos(&[], &[]).is_err());
        assert!(IirFilter::<Complex32, f32>::new_sos(&[1.0, 0.0, 0.0], &[1.0, 0.0]).is_err());
        assert!(IirFilter::<Complex32, f32>::new_sos(&[1.0, 0.0], &[1.0, 0.0]).is_err());

        // create valid object
        let mut filter = IirFilter::<Complex32, f32>::new_lowpass(7, 0.1).unwrap();

        // check properties
        filter.set_scale(7.22);
        let scale = filter.scale();
        assert_eq!(scale, 7.22);
        assert_eq!(filter.length(), 8); // 7+1

        // Rust automatically handles destruction of objects when they go out of scope
    }

    #[test]
    fn test_iirfilt_pll_accepts_all_positive_design_parameters() {
        assert!(IirFilter::<f32, f32>::new_pll(0.0, 0.5, 1000.0).is_err());
        assert!(IirFilter::<f32, f32>::new_pll(0.5, 0.0, 1000.0).is_err());
        assert!(IirFilter::<f32, f32>::new_pll(0.5, 0.5, 0.0).is_err());

        // allow any natural frequency > 0
        let (numerator, denominator, num_sections) = iir_filter_design_pll::<f32>(2.0, 1.5, 1000.0).unwrap();
        assert_eq!(num_sections, 1);
        assert!(numerator.iter().all(|coefficient| coefficient.is_finite()));
        assert!(denominator.iter().all(|coefficient| coefficient.is_finite()));
        assert!(IirFilter::<f32, f32>::new_pll(2.0, 1.5, 1000.0).is_ok());
    }

    #[test]
    fn test_iirfilt_sos_block_const_matches() {
        let x = (0..32)
            .map(|i| Complex32::new(i as f32 * 0.03125 - 0.4, 0.25 - i as f32 * 0.015625))
            .collect::<Vec<_>>();

        for nsos in 1..=9 {
            let mut b = Vec::with_capacity(3 * nsos);
            let mut a = Vec::with_capacity(3 * nsos);
            for i in 0..nsos {
                let offset = i as f32 * 0.001;
                b.extend_from_slice(&[0.7 + offset, 0.2 - offset, 0.05]);
                a.extend_from_slice(&[1.0, -0.15 + offset, 0.03]);
            }

            let mut sample_filter = IirFilter::<Complex32, f32>::new_sos(&b, &a).unwrap();
            sample_filter.set_scale(0.875);
            let mut block_filter = sample_filter.clone();

            let expected = x.iter().map(|&v| sample_filter.execute(v)).collect::<Vec<_>>();
            let mut actual = vec![Complex32::default(); x.len()];
            block_filter.execute_block(&x, &mut actual).unwrap();

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_iirfilt_numerator_longer_than_denominator() {
        let mut filter = IirFilter::<f32, f32>::new(&[1.0, 2.0, 3.0], &[1.0]).unwrap();
        let mut output = [0.0; 4];
        filter.execute_block(&[1.0, 0.0, 0.0, 0.0], &mut output).unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_iirfilt_denominator_longer_than_numerator() {
        let mut filter = IirFilter::<f32, f32>::new(&[1.0], &[1.0, -0.5]).unwrap();
        let mut output = [0.0; 4];
        filter.execute_block(&[1.0, 0.0, 0.0, 0.0], &mut output).unwrap();
        assert_eq!(output, [1.0, 0.5, 0.25, 0.125]);

        filter.set_coefficients(&[2.0], &[2.0, -0.5]).unwrap();
        filter.reset();
        filter.execute_block(&[1.0, 0.0, 0.0, 0.0], &mut output).unwrap();
        assert_eq!(output, [1.0, 0.25, 0.0625, 0.015625]);
    }

    #[test]
    #[autotest_annotate(autotest_iir_groupdelay_n3)]
    fn test_iir_groupdelay_n3() {
        // create coefficients array
        let b = [0.20657210, 0.41314420, 0.20657210];
        let a = [1.00000000, -0.36952737, 0.19581573];

        let tol = 1e-3f32;

        // create testing vectors
        let fc = [0.000, 0.125, 0.250, 0.375];
        let g0 = [0.973248939389634, 1.366481121240365, 1.227756735863196, 0.651058521306726];

        // run tests
        for i in 0..4 {
            let g = design::iir_group_delay(&b, &a, fc[i]).unwrap();
            assert_abs_diff_eq!(g, g0[i], epsilon = tol);
        }

        // create filter
        let filter = IirFilter::<f32, f32>::new(&b, &a).unwrap();

        // run tests again
        for i in 0..4 {
            let g = filter.groupdelay(fc[i]).unwrap();
            assert_abs_diff_eq!(g, g0[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iir_groupdelay_n8)]
    fn test_iir_groupdelay_n8() {
        // create coefficients arrays (7th-order Butterworth)
        #[rustfmt::skip]
        let b = [
            0.00484212,
            0.03389481,
            0.10168444,
            0.16947407,
            0.16947407,
            0.10168444,
            0.03389481,
            0.00484212,
        ];

        #[rustfmt::skip]
        let a = [
             1.00000000,
            -1.38928008,
             1.67502367,
            -1.05389738,
             0.50855154,
            -0.14482945,
             0.02625222,
            -0.00202968,
        ];

        let tol = 1e-3f32;

        // create testing vectors
        #[rustfmt::skip]
        let fc = [
            0.00000,
            0.06250,
            0.12500,
            0.18750,
            0.25000,
            0.31250,
            0.37500,
        ];

        #[rustfmt::skip]
        let g0 = [
            3.09280801068444,
            3.30599360247944,
            4.18341028373046,
            7.71934054380586,
            4.34330109915390,
            2.60203085226210,
            1.97868452107144,
        ];

        // run tests
        for i in 0..7 {
            let g = design::iir_group_delay(&b, &a, fc[i]).unwrap();
            assert_abs_diff_eq!(g, g0[i], epsilon = tol);
        }

        //
        // test with iir filter (tf)
        //

        // create filter
        let filter = IirFilter::<f32, f32>::new(&b, &a).unwrap();

        // run tests again
        for i in 0..7 {
            let g = filter.groupdelay(fc[i]).unwrap();
            assert_abs_diff_eq!(g, g0[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iir_groupdelay_sos_n8)]
    fn test_iir_groupdelay_sos_n8() {
        // create coefficients arrays (7th-order Butterworth)
        #[rustfmt::skip]
        let b = [
            0.00484212, 0.00968423, 0.00484212,
            1.00000000, 2.00000000, 1.00000000,
            1.00000000, 2.00000000, 1.00000000,
            1.00000000, 1.00000000, 0.00000000,
        ];

        #[rustfmt::skip]
        let a = [
            1.00000000, -0.33283597, 0.07707999,
            1.00000000, -0.38797498, 0.25551325,
            1.00000000, -0.51008475, 0.65066898,
            1.00000000, -0.15838444, 0.00000000,
        ];

        let tol = 1e-3f32;

        // create testing vectors
        #[rustfmt::skip]
        let fc = [
            0.00000,
            0.06250,
            0.12500,
            0.18750,
            0.25000,
            0.31250,
            0.37500,
        ];

        #[rustfmt::skip]
        let g0 = [
            3.09280801068444,
            3.30599360247944,
            4.18341028373046,
            7.71934054380586,
            4.34330109915390,
            2.60203085226210,
            1.97868452107144,
        ];

        // create filter
        let filter = IirFilter::<f32, f32>::new_sos(&b, &a).unwrap();

        // run tests
        for i in 0..7 {
            let g = filter.groupdelay(fc[i]).unwrap();
            assert_abs_diff_eq!(g, g0[i], epsilon = tol);
        }
    }

    include!("test_data_filter.rs");

    // autotest helper function
    //  _b      :   filter coefficients (numerator)
    //  _a      :   filter coefficients (denominator)
    //  _h_len  :   filter coefficients length
    //  _x      :   input array
    //  _x_len  :   input array length
    //  _y      :   output array
    //  _y_len  :   output array length
    fn iirfilt_rrrf_test(b: &[f32], a: &[f32], x: &[f32], y: &[f32]) -> () {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = IirFilter::<f32, f32>::new(b, a).unwrap();

        // allocate memory for output
        let mut y_test = vec![0.0; y.len()];

        // compute output
        q.execute_block(x, &mut y_test).unwrap();

        // Compare results
        for (y_i, y_test_i) in y.iter().zip(y_test.iter()) {
            assert_abs_diff_eq!(*y_i, *y_test_i, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_rrrf_h3x64)]
    fn test_iirfilt_rrrf_h3x64() {
        iirfilt_rrrf_test(
            &IIRFILT_RRRF_DATA_H3X64_B,
            &IIRFILT_RRRF_DATA_H3X64_A,
            &IIRFILT_RRRF_DATA_H3X64_X,
            &IIRFILT_RRRF_DATA_H3X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_rrrf_h5x64)]
    fn test_iirfilt_rrrf_h5x64() {
        iirfilt_rrrf_test(
            &IIRFILT_RRRF_DATA_H5X64_B,
            &IIRFILT_RRRF_DATA_H5X64_A,
            &IIRFILT_RRRF_DATA_H5X64_X,
            &IIRFILT_RRRF_DATA_H5X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_rrrf_h7x64)]
    fn test_iirfilt_rrrf_h7x64() {
        iirfilt_rrrf_test(
            &IIRFILT_RRRF_DATA_H7X64_B,
            &IIRFILT_RRRF_DATA_H7X64_A,
            &IIRFILT_RRRF_DATA_H7X64_X,
            &IIRFILT_RRRF_DATA_H7X64_Y,
        );
    }

    // autotest helper function
    //  _b      :   filter coefficients (numerator)
    //  _a      :   filter coefficients (denominator)
    //  _h_len  :   filter coefficients length
    //  _x      :   input array
    //  _x_len  :   input array length
    //  _y      :   output array
    //  _y_len  :   output array length
    fn iirfilt_crcf_test(b: &[f32], a: &[f32], x: &[Complex32], y: &[Complex32]) -> () {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = IirFilter::<Complex32, f32>::new(b, a).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];

        // compute output
        q.execute_block(x, &mut y_test).unwrap();

        // Compare results
        for (y_i, y_test_i) in y.iter().zip(y_test.iter()) {
            assert_abs_diff_eq!(y_i.re, y_test_i.re, epsilon = tol);
            assert_abs_diff_eq!(y_i.im, y_test_i.im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_crcf_h3x64)]
    fn test_iirfilt_crcf_h3x64() {
        iirfilt_crcf_test(
            &IIRFILT_CRCF_DATA_H3X64_B,
            &IIRFILT_CRCF_DATA_H3X64_A,
            &IIRFILT_CRCF_DATA_H3X64_X,
            &IIRFILT_CRCF_DATA_H3X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_crcf_h5x64)]
    fn test_iirfilt_crcf_h5x64() {
        iirfilt_crcf_test(
            &IIRFILT_CRCF_DATA_H5X64_B,
            &IIRFILT_CRCF_DATA_H5X64_A,
            &IIRFILT_CRCF_DATA_H5X64_X,
            &IIRFILT_CRCF_DATA_H5X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_crcf_h7x64)]
    fn test_iirfilt_crcf_h7x64() {
        iirfilt_crcf_test(
            &IIRFILT_CRCF_DATA_H7X64_B,
            &IIRFILT_CRCF_DATA_H7X64_A,
            &IIRFILT_CRCF_DATA_H7X64_X,
            &IIRFILT_CRCF_DATA_H7X64_Y,
        );
    }

    // autotest helper function
    //  _b      :   filter coefficients (numerator)
    //  _a      :   filter coefficients (denominator)
    //  _h_len  :   filter coefficients length
    //  _x      :   input array
    //  _x_len  :   input array length
    //  _y      :   output array
    //  _y_len  :   output array length
    fn iirfilt_cccf_test(b: &[Complex32], a: &[Complex32], x: &[Complex32], y: &[Complex32]) -> () {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = IirFilter::<Complex32, Complex32>::new(b, a).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];

        // compute output
        q.execute_block(x, &mut y_test).unwrap();

        // Compare results
        for (y_i, y_test_i) in y.iter().zip(y_test.iter()) {
            assert_abs_diff_eq!(y_i.re, y_test_i.re, epsilon = tol);
            assert_abs_diff_eq!(y_i.im, y_test_i.im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_cccf_h3x64)]
    fn test_iirfilt_cccf_h3x64() {
        iirfilt_cccf_test(
            &IIRFILT_CCCF_DATA_H3X64_B,
            &IIRFILT_CCCF_DATA_H3X64_A,
            &IIRFILT_CCCF_DATA_H3X64_X,
            &IIRFILT_CCCF_DATA_H3X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_cccf_h5x64)]
    fn test_iirfilt_cccf_h5x64() {
        iirfilt_cccf_test(
            &IIRFILT_CCCF_DATA_H5X64_B,
            &IIRFILT_CCCF_DATA_H5X64_A,
            &IIRFILT_CCCF_DATA_H5X64_X,
            &IIRFILT_CCCF_DATA_H5X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_cccf_h7x64)]
    fn test_iirfilt_cccf_h7x64() {
        iirfilt_cccf_test(
            &IIRFILT_CCCF_DATA_H7X64_B,
            &IIRFILT_CCCF_DATA_H7X64_A,
            &IIRFILT_CCCF_DATA_H7X64_X,
            &IIRFILT_CCCF_DATA_H7X64_Y,
        );
    }

    #[test]
    fn test_iirfilt_execute_block_contract() {
        let q = IirFilter::<f32, f32>::new(&[0.25, 0.25], &[1.0, -0.5]).unwrap();
        validate_block_execute(&q, &[1.0, -0.5, 0.25], -9.0, IirFilter::execute_block);
    }
}
