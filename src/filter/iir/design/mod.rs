mod bessel;
mod butterworth;
mod chebyshev_type_i;
mod chebyshev_type_ii;
mod elliptic;
mod pll;

pub use bessel::*;
pub use butterworth::*;
pub use chebyshev_type_i::*;
pub use chebyshev_type_ii::*;
pub use elliptic::*;
pub use pll::*;

use crate::error::{Error, Result};
use crate::math::{poly_expandbinomial_pm, poly_expandroots, polyf_findroots};
use num_complex::Complex32;
use std::f32::consts::PI;

//
// iir (infinite impulse response) filter design
//
// References
//  [Constantinides:1967] A. G. Constantinides, "Frequency
//      Transformations for Digital Filters." IEEE Electronic
//      Letters, vol. 3, no. 11, pp 487-489, 1967.
//

/// Analog low-pass prototype used for IIR design.
///
/// The shape decides what the design frequency `fc` means. Butterworth and
/// Bessel are near -3 dB at `fc`. Chebyshev type I and elliptic filters reach
/// the edge of their `ap` passband ripple there. Chebyshev type II reaches its
/// `as_` stopband attenuation at `fc`, so its passband ends below `fc`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirFilterShape {
    /// Maximally flat passband.
    Butter,
    /// Chebyshev type I with passband ripple.
    Cheby1,
    /// Chebyshev type II with stopband ripple.
    Cheby2,
    /// Elliptic response with passband and stopband ripple.
    Ellip,
    /// Bessel response with approximately flat group delay near zero frequency.
    Bessel,
}

/// Frequency transformation applied to an IIR prototype.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirBandType {
    /// Retain the low-pass prototype.
    Lowpass,
    /// Transform the prototype to high-pass.
    Highpass,
    /// Transform the prototype to band-pass.
    Bandpass,
    /// Transform the prototype to band-stop.
    Bandstop,
}

/// Coefficient representation produced by IIR design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirFormat {
    /// One numerator and denominator polynomial.
    TransferFunction,
    /// A cascade of three-coefficient sections in `[b0, b1, b2]` order.
    SecondOrderSections,
}

/// Groups `z[..n]` into complex conjugate pairs and writes them to `p`.
///
/// Two values pair when their real parts and their negated imaginary parts
/// agree within `tol`. A value whose imaginary part is below `tol` in
/// magnitude counts as real. Pairs come first, ordered by increasing real
/// part with the negative imaginary member first. Real values follow in
/// increasing order.
///
/// ```text
/// z                p
/// 10 + 3j          -3 - 4j
///  5 + 0j          -3 + 4j
/// -3 + 4j          10 - 3j
/// 10 - 3j          10 + 3j
///  3 + 0j           3 + 0j
/// -3 - 4j           5 + 0j
/// ```
///
/// A complex value with no partner is dropped, and the tail of `p` it would
/// have filled is left unchanged.
///
/// # Errors
///
/// Returns an error when `tol` is negative.
fn find_conjugate_pairs(z: &[Complex32], n: usize, tol: f32, p: &mut [Complex32]) -> Result<()> {
    // validate input
    if tol < 0.0 {
        return Err(Error::Range("tolerance must be positive".into()));
    }

    // keep track of which elements have been paired
    let mut paired = vec![false; n];
    let mut num_pairs = 0;
    let mut k = 0;

    for i in 0..n {
        // ignore value if already paired, or if imaginary
        // component is less than tolerance
        if paired[i] || z[i].im.abs() < tol {
            continue;
        }

        for j in i + 1..n {
            // ignore value if already paired, or if imaginary
            // component is less than tolerance
            if j == i || paired[j] || z[j].im.abs() < tol {
                continue;
            }

            if (z[i].im + z[j].im).abs() < tol && (z[i].re - z[j].re).abs() < tol {
                // found complex conjugate pair
                p[k] = z[i];
                p[k + 1] = z[j];
                paired[i] = true;
                paired[j] = true;
                num_pairs += 1;
                k += 2;
                break;
            }
        }
    }

    if k > n {
        return Err(Error::Range("invalid derived order".into()));
    }

    // sort through remaining unpaired values and ensure
    // they are purely real
    for i in 0..n {
        if paired[i] {
            continue;
        }

        if z[i].im.abs() < tol {
            p[k] = z[i];
            paired[i] = true;
            k += 1;
        }
    }

    // clean up result
    //  * force pairs to be perfect conjugates with
    //    negative imaginary component first
    //  * complex conjugate pairs are ordered by
    //    increasing real component
    //  * pure-real elements are ordered by increasing
    //    value
    find_conjugate_pairs_cleanup(p, n, num_pairs)
}

/// Tidies the output of [`find_conjugate_pairs`].
///
/// `p[..2 * num_pairs]` holds the pairs and the rest of `p[..n]` holds the
/// real values. Each pair is made an exact conjugate pair with the negative
/// imaginary member first. The pairs are then sorted by increasing real part,
/// and the real values by increasing value.
fn find_conjugate_pairs_cleanup(p: &mut [Complex32], n: usize, num_pairs: usize) -> Result<()> {
    // force pairs to be perfect conjugates with
    // negative imaginary component first
    for i in 0..num_pairs {
        p[2 * i] = if p[2 * i + 1].im < 0.0 { p[2 * i + 1] } else { p[2 * i + 1].conj() };
        p[2 * i + 1] = p[2 * i].conj();
    }

    // sort conjugate pairs
    for i in 0..num_pairs {
        for j in (i + 1..num_pairs).rev() {
            if p[2 * (j - 1)].re > p[2 * j].re {
                // swap pairs
                p.swap(2 * (j - 1), 2 * j);

                p.swap(2 * (j - 1) + 1, 2 * j + 1);
            }
        }
    }

    // sort pure-real elements
    for i in 2 * num_pairs..n {
        for j in (i + 1..n).rev() {
            if p[j - 1].re > p[j].re {
                p.swap(j - 1, j);
            }
        }
    }

    Ok(())
}

/// Computes the bilinear-transform prewarp factor.
///
/// `fc` is the cutoff in cycles per sample. `f0` is the center frequency in
/// cycles per sample, and only band-pass and band-stop transformations use
/// it. Neither frequency is validated. The factor is non-negative. For
/// band-pass and band-stop designs, either band edge may be supplied as `fc`.
pub fn iir_design_freqprewarp(btype: IirBandType, fc: f32, f0: f32) -> f32 {
    // See [Constantinides:1967]
    let m = match btype {
        IirBandType::Lowpass => (PI * fc).tan(),
        IirBandType::Highpass => -(PI * fc).cos() / (PI * fc).sin(),
        IirBandType::Bandpass => ((2.0 * PI * fc).cos() - (2.0 * PI * f0).cos()) / (2.0 * PI * fc).sin(),
        IirBandType::Bandstop => (2.0 * PI * fc).sin() / ((2.0 * PI * fc).cos() - (2.0 * PI * f0).cos()),
    };
    m.abs()
}

/// Applies a bilinear transform to analog zeros, poles, and gain.
///
/// `m` is the prewarp factor from [`iir_design_freqprewarp`]. `zd` and `pd`
/// receive one entry for every analog pole, and `kd` receives the digital
/// gain. A filter with fewer zeros than poles gets its missing zeros at -1.
/// Analog zeros beyond the pole count are ignored.
///
/// `ka` is the gain of the digital filter at DC, not the analog gain factor.
/// The analog filter's own DC gain does not carry over. For the analog filter
/// `2 / (s + 2)`, whose DC gain is one, passing `ka = 2` gives a digital DC
/// gain of two.
///
/// An analog root at `1 / m`, or an analog zero at zero, produces nonfinite
/// roots or gain without an error.
///
/// # Errors
///
/// Returns an error when `zd` or `pd` is shorter than `pa`. The outputs are
/// unchanged on error.
pub fn iir_design_bilinear_a2d(
    za: &[Complex32],
    pa: &[Complex32],
    ka: Complex32,
    m: f32,
    zd: &mut [Complex32],
    pd: &mut [Complex32],
    kd: &mut Complex32,
) -> Result<()> {
    let nza = za.len();
    let npa = pa.len();

    if zd.len() < npa || pd.len() < npa {
        return Err(Error::Range("Output arrays too small".into()));
    }

    let mut kd_temp = ka;
    for i in 0..npa {
        // compute digital zeros
        if i < nza {
            zd[i] = (1.0 + za[i] * m) / (1.0 - za[i] * m);
        } else {
            zd[i] = Complex32::new(-1.0, 0.0);
        }

        // compute digital poles
        pd[i] = (1.0 + pa[i] * m) / (1.0 - pa[i] * m);

        // compute digital gain
        kd_temp *= (1.0 - pd[i]) / (1.0 - zd[i]);
    }

    *kd = kd_temp;

    Ok(())
}

/// Applies a bilinear transform to analog transfer-function polynomials.
///
/// `b` and `a` hold ascending powers of `s`, with `b_order + 1` and
/// `a_order + 1` coefficients. Both outputs receive `a_order + 1`
/// coefficients in ascending powers of `z^-1`, normalized so `ad[0]` is one.
///
/// The substitution is `s = (1 - z^-1) / (m * (1 + z^-1))`, using the same
/// prewarp factor as [`iir_design_bilinear_a2d`]. `m` must be nonzero, and
/// the transformed denominator's constant coefficient must be nonzero.
/// Unlike the root-based transform, this preserves the analog gain factor.
///
/// # Errors
///
/// Returns an error when `a_order` is zero, `b_order` exceeds `a_order`, `m`
/// is zero or nonfinite, an order count overflows, an input or output slice
/// is too short, or the transformed denominator's constant coefficient is
/// zero.
pub fn iir_design_bilinear_z(
    b: &[Complex32],
    b_order: usize,
    a: &[Complex32],
    a_order: usize,
    m: f32,
    bd: &mut [Complex32],
    ad: &mut [Complex32],
) -> Result<()> {
    if a_order == 0 {
        return Err(Error::Range("denominator order must be greater than zero".into()));
    }
    if b_order > a_order {
        return Err(Error::Range("numerator order cannot be higher than denominator".into()));
    }
    if !m.is_finite() || m == 0.0 {
        return Err(Error::Config("bilinear prewarp factor must be finite and nonzero".into()));
    }

    let nb = b_order.checked_add(1).ok_or_else(|| Error::Range("numerator order overflow".into()))?;
    let na = a_order.checked_add(1).ok_or_else(|| Error::Range("denominator order overflow".into()))?;
    if b.len() < nb || a.len() < na || bd.len() < na || ad.len() < na {
        return Err(Error::Config("bilinear transform coefficient slices are too short".into()));
    }

    // clear output arrays
    for i in 0..na {
        bd[i] = Complex32::new(0.0, 0.0);
        ad[i] = Complex32::new(0.0, 0.0);
    }

    // temporary polynomial: (1 + 1/z)^(n-i) * (1 - 1/z)^i
    let mut poly_1pz = vec![Complex32::new(0.0, 0.0); na];

    let mut mk = Complex32::new(1.0, 0.0);

    // substitute s and multiply denominator by (1+1/z)^a_order
    for i in 0..na {
        // expand the polynomial (1+x)^(a_order-i) * (1-x)^i
        poly_expandbinomial_pm(a_order - i, i, &mut poly_1pz)?;

        // accumulate polynomial coefficients
        for j in 0..na {
            ad[j] += a[i] * mk * poly_1pz[j];
        }

        // update multiplier
        mk /= m;
    }

    if ad[0] == Complex32::new(0.0, 0.0) {
        return Err(Error::Config("transformed denominator has zero constant coefficient".into()));
    }

    // substitute s and multiply numerator by (1+1/z)^a_order
    mk = Complex32::new(1.0, 0.0);
    for i in 0..nb {
        // expand the polynomial (1+x)^(a_order-i) * (1-x)^i
        poly_expandbinomial_pm(a_order - i, i, &mut poly_1pz)?;

        // accumulate polynomial coefficients
        for j in 0..na {
            bd[j] += b[i] * mk * poly_1pz[j];
        }

        // update multiplier
        mk /= m;
    }

    // normalize by ad[0]
    let a0_inv = 1.0 / ad[0];
    for i in 0..na {
        bd[i] *= a0_inv;
        ad[i] *= a0_inv;
    }

    Ok(())
}

/// Converts digital zeros, poles, and gain to a transfer function.
///
/// `zd` and `pd` supply `n` zeros and poles. `b` and `a` receive `n + 1`
/// coefficients each, in ascending powers of `z^-1`. Only the real part of
/// each expanded coefficient is kept, so the zeros and poles should be real
/// or come in conjugate pairs. With `n` zero, both outputs are set to zero.
///
/// # Errors
///
/// Returns an error when `n + 1` overflows, `zd` or `pd` is shorter than `n`,
/// or `b` or `a` is shorter than `n + 1`. The outputs are unchanged on error.
pub fn iir_design_d2tf(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    k: Complex32,
    b: &mut [f32],
    a: &mut [f32],
) -> Result<()> {
    let roots = n.checked_add(1).ok_or_else(|| Error::Range("filter order overflows".into()))?;
    let zd = zd.get(..n).ok_or_else(|| Error::Range("zero slice is shorter than filter order".into()))?;
    let pd = pd.get(..n).ok_or_else(|| Error::Range("pole slice is shorter than filter order".into()))?;
    let b = b
        .get_mut(..roots)
        .ok_or_else(|| Error::Range("numerator slice is shorter than filter order plus one".into()))?;
    let a = a
        .get_mut(..roots)
        .ok_or_else(|| Error::Range("denominator slice is shorter than filter order plus one".into()))?;
    let mut q = vec![Complex32::new(0.0, 0.0); roots];

    // expand poles
    poly_expandroots(pd, &mut q)?;
    for i in 0..=n {
        a[i] = q[n - i].re;
    }

    // expand zeros
    poly_expandroots(zd, &mut q)?;
    for i in 0..=n {
        b[i] = (q[n - i] * k).re;
    }

    Ok(())
}

/// Converts digital zeros, poles, and gain to second-order sections.
///
/// `zd` and `pd` supply `n` zeros and poles. The result has `(n + 1) / 2`
/// sections, and `b` and `a` receive three coefficients per section in
/// ascending powers of `z^-1`. An odd order ends with a first-order section,
/// whose last numerator and denominator coefficients are zero.
///
/// The zeros and poles should be real or come in conjugate pairs, matched
/// within `1e-6`. A complex root without a partner is not reported. It
/// yields sections that do not represent the inputs.
///
/// Every numerator section is scaled by the same factor, chosen so the
/// factors multiply to the magnitude of the real part of `k`. The sign of `k`
/// goes to the first section. The imaginary part of `k` is ignored.
///
/// # Errors
///
/// Returns an error when `n` is zero, the section count overflows, or any
/// input or output slice is too short. Outputs are unchanged for these errors.
pub fn iir_design_d2sos(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    k: Complex32,
    b: &mut [f32],
    a: &mut [f32],
) -> Result<()> {
    if n == 0 {
        return Err(Error::Config("filter order must be greater than zero".into()));
    }
    let sections = n / 2 + n % 2;
    let output_len =
        sections.checked_mul(3).ok_or_else(|| Error::Range("section coefficient count overflow".into()))?;
    if zd.len() < n || pd.len() < n || b.len() < output_len || a.len() < output_len {
        return Err(Error::Config("second-order section slices are too short".into()));
    }
    let tol = 1e-6f32;

    // find/group complex conjugate pairs (zeros)
    let mut zp = vec![Complex32::new(0.0, 0.0); n];
    if find_conjugate_pairs(zd, n, tol, &mut zp).is_err() {
        return Err(Error::Internal("could not associate complex pairs (zeros)".into()));
    }

    // find/group complex conjugate pairs (poles)
    let mut pp = vec![Complex32::new(0.0, 0.0); n];
    if find_conjugate_pairs(pd, n, tol, &mut pp).is_err() {
        return Err(Error::Internal("could not associate complex pairs (poles)".into()));
    }

    // _n = 2*L + r
    let r = n % 2; // odd/even order
    let l = (n - r) / 2; // filter semi-length

    for i in 0..l {
        let p0 = -pp[2 * i];
        let p1 = -pp[2 * i + 1];

        let z0 = -zp[2 * i];
        let z1 = -zp[2 * i + 1];

        // expand complex pole pairs
        a[3 * i] = 1.0;
        a[3 * i + 1] = (p0 + p1).re;
        a[3 * i + 2] = (p0 * p1).re;

        // expand complex zero pairs
        b[3 * i] = 1.0;
        b[3 * i + 1] = (z0 + z1).re;
        b[3 * i + 2] = (z0 * z1).re;
    }

    // add remaining zero/pole pair if order is odd
    if r == 1 {
        let p0 = -pp[n - 1];
        let z0 = -zp[n - 1];

        // expand complex pole pair
        a[3 * l] = 1.0;
        a[3 * l + 1] = p0.re;
        a[3 * l + 2] = 0.0;

        // expand complex zero pair
        b[3 * l] = 1.0;
        b[3 * l + 1] = z0.re;
        b[3 * l + 2] = 0.0;
    }

    // distribute gain equally amongst all feed-forward coefficients
    let k = k.re;
    let sgn = if k < 0.0 { -1.0 } else { 1.0 };
    let g = (k * sgn).powf(1.0 / (l + r) as f32);

    // adjust gain of first element
    for i in 0..l + r {
        b[3 * i] *= g;
        b[3 * i + 1] *= g;
        b[3 * i + 2] *= g;
    }

    // apply sign to first section (handle case where gain is negative)
    b[0] *= sgn;
    b[1] *= sgn;
    b[2] *= sgn;

    Ok(())
}

/// Transforms digital low-pass zeros and poles to high-pass.
///
/// The transform negates each root. `zdt` and `pdt` each receive `n`
/// transformed roots.
///
/// # Errors
///
/// Returns an error when any input or output slice is shorter than `n`.
/// Outputs are unchanged on error.
pub fn iir_design_lp2hp(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    zdt: &mut [Complex32],
    pdt: &mut [Complex32],
) -> Result<()> {
    if zd.len() < n || pd.len() < n || zdt.len() < n || pdt.len() < n {
        return Err(Error::Config("low-pass to high-pass root slices are too short".into()));
    }
    for i in 0..n {
        zdt[i] = -zd[i];
        pdt[i] = -pd[i];
    }

    Ok(())
}

/// Transforms digital low-pass zeros and poles to band-pass.
///
/// `f0` is the center frequency in cycles per sample, in `[0, 0.5]`.
/// Each input root becomes two output roots, so `zdt` and `pdt` each receive
/// `2 * n` roots.
///
/// # Errors
///
/// Returns an error when `f0` is nonfinite or out of range, `2 * n` overflows,
/// `zd` or `pd` is shorter than `n`, or `zdt` or `pdt` is shorter than `2 * n`.
/// Outputs are unchanged on error.
pub fn iir_design_lp2bp(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    f0: f32,
    zdt: &mut [Complex32],
    pdt: &mut [Complex32],
) -> Result<()> {
    if !(0.0..=0.5).contains(&f0) {
        return Err(Error::Config("center frequency must be in [0,0.5]".into()));
    }
    let output_len = n.checked_mul(2).ok_or_else(|| Error::Range("band-pass root count overflow".into()))?;
    if zd.len() < n || pd.len() < n || zdt.len() < output_len || pdt.len() < output_len {
        return Err(Error::Config("low-pass to band-pass root slices are too short".into()));
    }
    let c0 = (2.0 * PI * f0).cos();

    // transform zeros, poles using quadratic formula
    for i in 0..n {
        let t0 = 1.0 + zd[i];
        (zdt[2 * i], zdt[2 * i + 1]) = (
            0.5 * (c0 * t0 + (c0 * c0 * t0 * t0 - 4.0 * zd[i]).sqrt()),
            0.5 * (c0 * t0 - (c0 * c0 * t0 * t0 - 4.0 * zd[i]).sqrt()),
        );

        let t0 = 1.0 + pd[i];
        (pdt[2 * i], pdt[2 * i + 1]) = (
            0.5 * (c0 * t0 + (c0 * c0 * t0 * t0 - 4.0 * pd[i]).sqrt()),
            0.5 * (c0 * t0 - (c0 * c0 * t0 * t0 - 4.0 * pd[i]).sqrt()),
        );
    }

    Ok(())
}

/// Designs a digital IIR filter from an analog prototype.
///
/// # Arguments
///
/// - `ftype`: the analog prototype.
/// - `btype`: the frequency transformation.
/// - `format`: the coefficient layout, a transfer function or second-order
///   sections.
/// - `n`: the prototype order. Must be nonzero.
/// - `fc`: a band edge in cycles per sample, in `(0, 0.5)`. For low-pass and
///   high-pass designs it is the edge that [`IirFilterShape`] describes:
///   about -3 dB for Butterworth and Bessel, the passband edge for Chebyshev
///   type I and elliptic, and the stopband edge for Chebyshev type II.
/// - `f0`: the center frequency for band-pass and band-stop designs, in
///   cycles per sample. Must lie in `[0, 0.5]` for every band type, including
///   those that ignore it.
/// - `ap`: passband ripple in dB. Must be finite and positive. Used by
///   Chebyshev type I and elliptic.
/// - `as_`: stopband attenuation in dB. Must be finite and positive. Used by
///   Chebyshev type II and elliptic. An elliptic design also needs `as_` greater than
///   `ap`.
/// - `b`, `a`: output coefficients in ascending powers of `z^-1`, sized as
///   described below.
///
/// For band-pass and band-stop designs, `fc` is one band edge rather than a
/// bandwidth or half-width, and `f0` places the other edge. Either edge may
/// be supplied as `fc`. For desired edges `f1` and `f2`, set
/// `f0 = acos(cos(pi * (f1 + f2)) / cos(pi * (f2 - f1))) / (2 * pi)`. Both
/// edges must lie in `(0, 0.5)`, and the `acos` argument must lie in
/// `[-1, 1]`. The edges need not be symmetric about `f0`.
///
/// Band-pass and band-stop transformations double the order. A transfer
/// function needs `n + 1` coefficients in each of `b` and `a` for low-pass
/// and high-pass, or `2 * n + 1` for band-pass and band-stop. Second-order
/// sections need three coefficients per section, which is
/// `3 * ((n + 1) / 2)` for low-pass and high-pass, or `3 * n` for band-pass
/// and band-stop.
///
/// A successful return does not verify that the filter meets its ripple
/// specification.
///
/// # Errors
///
/// Returns an error when `n` is zero, a parameter is out of range, an elliptic
/// design has `as_ <= ap`, an order count overflows, a design step fails, or
/// either output slice is too short for the selected format.
pub fn iir_design(
    ftype: IirFilterShape,
    btype: IirBandType,
    format: IirFormat,
    n: usize,
    fc: f32,
    f0: f32,
    ap: f32,
    as_: f32,
    b: &mut [f32],
    a: &mut [f32],
) -> Result<()> {
    // validate input
    if !(fc > 0.0 && fc < 0.5) {
        return Err(Error::Config("cutoff frequency out of range".into()));
    }
    if !(0.0..=0.5).contains(&f0) {
        return Err(Error::Config("center frequency out of range".into()));
    }
    if !ap.is_finite() || ap <= 0.0 {
        return Err(Error::Config("pass-band ripple out of range".into()));
    }
    if !as_.is_finite() || as_ <= 0.0 {
        return Err(Error::Config("stop-band ripple out of range".into()));
    }
    if ftype == IirFilterShape::Ellip && as_ <= ap {
        return Err(Error::Config("elliptic stop-band attenuation must exceed pass-band ripple".into()));
    }
    if n == 0 {
        return Err(Error::Config("filter order must be > 0".into()));
    }
    let band = matches!(btype, IirBandType::Bandpass | IirBandType::Bandstop);
    let required_output = match (format, band) {
        (IirFormat::TransferFunction, false) => n.checked_add(1),
        (IirFormat::TransferFunction, true) => n.checked_mul(2).and_then(|order| order.checked_add(1)),
        (IirFormat::SecondOrderSections, false) => (n / 2 + n % 2).checked_mul(3),
        (IirFormat::SecondOrderSections, true) => n.checked_mul(3),
    }
    .ok_or_else(|| Error::Range("filter coefficient count overflow".into()))?;
    if b.len() < required_output || a.len() < required_output {
        return Err(Error::Config("filter coefficient output slices are too short".into()));
    }

    let mut order = n;

    // analog poles/zeros/gain
    let mut pa = vec![Complex32::new(0.0, 0.0); order]; // analog poles
    let mut za = vec![Complex32::new(0.0, 0.0); order]; // analog zeros
    let mut ka = Complex32::new(0.0, 0.0); // analog gain
    let k0; // nominal digital gain

    // derived values
    let r = n % 2; // odd/even filter order

    // compute zeros and poles of analog prototype
    match ftype {
        IirFilterShape::Butter => {
            // Butterworth filter design : no zeros, n poles
            k0 = Complex32::new(1.0, 0.0);
            if iir_design_butter_analog(order, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (butterworth)".into()));
            }
        }
        IirFilterShape::Cheby1 => {
            // Cheby-I filter design : no zeros, n poles, pass-band ripple
            let epsilon = (10.0f32.powf(ap / 10.0) - 1.0).sqrt();
            k0 = if r == 1 {
                Complex32::new(1.0, 0.0)
            } else {
                Complex32::new(1.0 / (1.0 + epsilon * epsilon).sqrt(), 0.0)
            };
            if iir_design_cheby1_analog(order, epsilon, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (cheby1)".into()));
            }
        }
        IirFilterShape::Cheby2 => {
            // Cheby-II filter design : n-r zeros, n poles, stop-band ripple
            let epsilon = 10.0f32.powf(-as_ / 20.0);
            k0 = Complex32::new(1.0, 0.0);
            if iir_design_cheby2_analog(order, epsilon, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (cheby2)".into()));
            }
        }
        IirFilterShape::Ellip => {
            // elliptic filter design : n-r zeros, n poles, pass/stop-band ripple
            let gp = 10.0f32.powf(-ap / 20.0);
            let gs = 10.0f32.powf(-as_ / 20.0);
            let ep = (1.0 / (gp * gp) - 1.0).sqrt();
            let es = (1.0 / (gs * gs) - 1.0).sqrt();
            k0 = if r == 1 { Complex32::new(1.0, 0.0) } else { Complex32::new(1.0 / (1.0 + ep * ep).sqrt(), 0.0) };
            if iir_design_ellip_analog(order, ep, es, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (elliptic)".into()));
            }
        }
        IirFilterShape::Bessel => {
            // Bessel filter design : no zeros, n poles
            k0 = Complex32::new(1.0, 0.0);
            if iir_design_bessel_analog(order, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (bessel)".into()));
            }
        }
    }

    // complex digital poles/zeros/gain
    // NOTE: allocated double the filter order to cover band-pass, band-stop cases
    let mut zd = vec![Complex32::new(0.0, 0.0); 2 * order];
    let mut pd = vec![Complex32::new(0.0, 0.0); 2 * order];
    let mut kd = Complex32::new(0.0, 0.0);

    let m = iir_design_freqprewarp(btype, fc, f0);

    if iir_design_bilinear_a2d(&za, &pa, k0, m, &mut zd, &mut pd, &mut kd).is_err() {
        return Err(Error::Internal("could not perform bilinear z-transform".into()));
    }

    // negate zeros, poles for high-pass and band-stop cases
    if btype == IirBandType::Highpass || btype == IirBandType::Bandstop {
        let mut pd_tmp = vec![Complex32::new(0.0, 0.0); 2 * order];
        let mut zd_tmp = vec![Complex32::new(0.0, 0.0); 2 * order];

        if iir_design_lp2hp(&zd, &pd, order, &mut zd_tmp, &mut pd_tmp).is_err() {
            return Err(Error::Internal("could not perform high-pass transformation".into()));
        }

        zd.copy_from_slice(&zd_tmp);
        pd.copy_from_slice(&pd_tmp);
    }

    // transform zeros, poles in band-pass, band-stop cases
    // NOTE: this also doubles the filter order
    if btype == IirBandType::Bandpass || btype == IirBandType::Bandstop {
        let mut zd1 = vec![Complex32::new(0.0, 0.0); 2 * n];
        let mut pd1 = vec![Complex32::new(0.0, 0.0); 2 * n];

        // run zeros, poles low-pass -> band-pass transform
        if iir_design_lp2bp(&zd, &pd, order, f0, &mut zd1, &mut pd1).is_err() {
            return Err(Error::Internal("could not perform band-pass transformation".into()));
        }

        // copy transformed zeros, poles
        zd.copy_from_slice(&zd1);
        pd.copy_from_slice(&pd1);

        // update parameters; filter order doubles which changes the
        // number of second-order sections and forces there to never
        // be any remainder (r=0 always).
        order *= 2;
    }

    if format == IirFormat::TransferFunction {
        // convert complex digital poles/zeros/gain into transfer
        // function : H(z) = B(z) / A(z)
        // where length(B,A) = low/high-pass ? _n + 1 : 2*_n + 1
        if iir_design_d2tf(&zd, &pd, order, kd, b, a).is_err() {
            return Err(Error::Internal("could not perform transfer function expansion".into()));
        }
    } else {
        // convert complex digital poles/zeros/gain into second-
        // order sections form :
        // H(z) = prod { (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2) }
        // where size(B,A) = low|high-pass  : [3]x[L+r]
        //                   band-pass|stop : [3]x[2*L]
        if iir_design_d2sos(&zd, &pd, order, kd, b, a).is_err() {
            return Err(Error::Internal("could not perform second-order sections expansion".into()));
        }
    }

    Ok(())
}

/// Checks whether an IIR denominator has no poles outside the unit circle.
///
/// `a[..n]` holds the denominator in ascending powers of `z^-1`, so its order
/// is `n - 1`. Stability depends only on the denominator.
///
/// This is not a strict stability test. A pole on the unit circle counts as
/// stable, so `true` does not rule out a marginally stable filter.
///
/// # Errors
///
/// Returns an error when `n < 2`, `a` is shorter than `n`, or root finding
/// fails.
pub fn iir_design_is_stable(a: &[f32], n: usize) -> Result<bool> {
    if n < 2 {
        return Err(Error::Config("filter order too low".into()));
    }
    if a.len() < n {
        return Err(Error::Config("denominator slice is shorter than filter order".into()));
    }

    let mut a_hat = vec![0.0; n];
    for i in 0..n {
        a_hat[i] = a[n - i - 1];
    }

    let mut roots = vec![Complex32::new(0.0, 0.0); n - 1];
    if polyf_findroots(&a_hat, &mut roots).is_err() {
        return Err(Error::Internal("could not find roots of polynomial".into()));
    }

    for root in &roots[..n - 1] {
        if root.norm() > 1.0 {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Estimates the group delay of an IIR filter, in samples, at `fc` cycles per
/// sample.
///
/// `b` and `a` are in ascending powers of `z^-1`. The calculation divides by a
/// term that scales with `b` and `a` and falls near zero close to a null in the
/// response. When its magnitude is below `1e-5`, the function returns zero
/// instead. That zero is a fallback, not the true delay. The threshold is
/// absolute, so scaling the coefficients down also trips it. A one-sample delay
/// reports 1, but the same filter with `b` scaled by `1e-6` reports 0.
///
/// # Errors
///
/// Returns an error when `b` or `a` is empty, or when `fc` is nonfinite or outside
/// `[-0.5, 0.5]`.
pub fn iir_group_delay(b: &[f32], a: &[f32], fc: f32) -> Result<f32> {
    // validate input
    if b.is_empty() {
        return Err(Error::Config("iir_group_delay(), numerator length must be greater than zero".to_string()));
    }
    if a.is_empty() {
        return Err(Error::Config("iir_group_delay(), denominator length must be greater than zero".to_string()));
    }
    if !(-0.5..=0.5).contains(&fc) {
        return Err(Error::Config("iir_group_delay(), _fc must be in [-0.5,0.5]".to_string()));
    }

    // compute c = conv(b,fliplr(a))
    //         c(z) = b(z)*a(1/z)*z^(-_na)
    let nc = a.len() + b.len() - 1;
    let mut c = vec![0.0; nc];

    for i in 0..a.len() {
        for j in 0..b.len() {
            let sum = a[a.len() - i - 1] * b[j];
            c[i + j] += sum;
        }
    }

    // compute
    //      sum(c[i] * exp(j 2 pi fc i) * i)
    //      --------------------------------
    //      sum(c[i] * exp(j 2 pi fc i))
    let mut t0 = Complex32::new(0.0, 0.0);
    let mut t1 = Complex32::new(0.0, 0.0);
    for (i, &ci) in c.iter().enumerate() {
        let c0 = ci * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * i as f32);
        t0 += c0 * i as f32;
        t1 += c0;
    }

    // prevent divide-by-zero (check magnitude for tolerance range)
    let tol = 1e-5;
    if t1.norm() < tol {
        return Ok(0.0);
    }

    // return result, scaled by length of denominator
    Ok((t0 / t1).re - (a.len() - 1) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::IirFilter;
    use crate::utility::test_helpers::{validate_psd_iirfilt, PsdRegion};
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    fn assert_iirdes_sos_poles_stable(a: &[f32]) {
        for section in a.chunks_exact(3) {
            // Solve a0*z^2 + a1*z + a2 = 0 independently of the root finder.
            let discriminant = Complex32::new(section[1] * section[1] - 4.0 * section[0] * section[2], 0.0);
            for sign in [-1.0, 1.0] {
                let pole = (-section[1] + sign * discriminant.sqrt()) / (2.0 * section[0]);
                assert!(pole.norm() < 1.0, "unstable pole: {pole}");
            }
        }
    }

    #[test]
    fn test_iirdes_freqprewarp_stability() {
        for btype in [IirBandType::Highpass, IirBandType::Bandpass, IirBandType::Bandstop] {
            let num_coeffs = if btype == IirBandType::Highpass { 6 } else { 12 };
            for fc in [0.1, 0.2, 0.3, 0.4] {
                let f0 = 0.25;
                let mut b = vec![0.0; num_coeffs];
                let mut a = b.clone();
                iir_design(
                    IirFilterShape::Butter,
                    btype,
                    IirFormat::SecondOrderSections,
                    4,
                    fc,
                    f0,
                    1.0,
                    60.0,
                    &mut b,
                    &mut a,
                )
                .unwrap();
                assert_iirdes_sos_poles_stable(&a);
            }
        }
    }

    #[test]
    fn test_iirdes_freqprewarp_band_edge_equivalence() {
        for (f1, f2) in [(0.1, 0.3), (0.2, 0.4)] {
            let f0 = ((PI * (f1 + f2)).cos() / (PI * (f2 - f1)).cos()).acos() / (2.0 * PI);
            for btype in [IirBandType::Bandpass, IirBandType::Bandstop] {
                for format in [IirFormat::TransferFunction, IirFormat::SecondOrderSections] {
                    let num_coeffs = if format == IirFormat::TransferFunction { 9 } else { 12 };
                    let mut b_lower = vec![0.0; num_coeffs];
                    let mut a_lower = b_lower.clone();
                    let mut b_upper = b_lower.clone();
                    let mut a_upper = b_lower.clone();
                    for (fc, b, a) in [(f1, &mut b_lower, &mut a_lower), (f2, &mut b_upper, &mut a_upper)] {
                        iir_design(IirFilterShape::Butter, btype, format, 4, fc, f0, 1.0, 60.0, b, a).unwrap();
                    }
                    for i in 0..num_coeffs {
                        assert_abs_diff_eq!(b_lower[i], b_upper[i], epsilon = 1e-5);
                        assert_abs_diff_eq!(a_lower[i], a_upper[i], epsilon = 1e-5);
                    }
                    if format == IirFormat::SecondOrderSections {
                        assert_iirdes_sos_poles_stable(&a_lower);
                        assert_iirdes_sos_poles_stable(&a_upper);
                    }
                }
            }
        }
    }

    #[test]
    fn test_iirdes_bilinear_z_first_order() {
        // Closed forms for 1/(s+1) and (s+3)/(s+2).
        for m in [0.25, 1.0, 2.0] {
            for (b, a, expected_b, expected_a) in [
                (vec![1.0], [1.0, 1.0], [m / (m + 1.0); 2], [1.0, (m - 1.0) / (m + 1.0)]),
                (
                    vec![3.0, 1.0],
                    [2.0, 1.0],
                    [(3.0 * m + 1.0) / (2.0 * m + 1.0), (3.0 * m - 1.0) / (2.0 * m + 1.0)],
                    [1.0, (2.0 * m - 1.0) / (2.0 * m + 1.0)],
                ),
            ] {
                let b: Vec<_> = b.into_iter().map(|x| Complex32::new(x, 0.0)).collect();
                let a = a.map(|x| Complex32::new(x, 0.0));
                let mut bd = [Complex32::new(0.0, 0.0); 2];
                let mut ad = bd;
                iir_design_bilinear_z(&b, b.len() - 1, &a, 1, m, &mut bd, &mut ad).unwrap();
                for i in 0..2 {
                    assert_abs_diff_eq!(bd[i].re, expected_b[i], epsilon = 1e-6);
                    assert_abs_diff_eq!(ad[i].re, expected_a[i], epsilon = 1e-6);
                    assert_abs_diff_eq!(bd[i].im, 0.0, epsilon = 1e-6);
                    assert_abs_diff_eq!(ad[i].im, 0.0, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_iirdes_bilinear_z_matches_a2d() {
        for n in 1..=5 {
            let mut za = Vec::new();
            let mut pa = Vec::new();
            let mut ka = Complex32::new(0.0, 0.0);
            iir_design_butter_analog(n, &mut za, &mut pa, &mut ka).unwrap();
            let mut a = vec![Complex32::new(0.0, 0.0); n + 1];
            poly_expandroots(&pa, &mut a).unwrap();
            // The root-based transform expects DC gain; choose unity for both.
            let b = [a[0]];

            for m in [0.25, 0.75, 1.5] {
                let mut bd = vec![Complex32::new(0.0, 0.0); n + 1];
                let mut ad = bd.clone();
                iir_design_bilinear_z(&b, 0, &a, n, m, &mut bd, &mut ad).unwrap();

                let mut zd = vec![Complex32::new(0.0, 0.0); n];
                let mut pd = zd.clone();
                let mut kd = Complex32::new(0.0, 0.0);
                iir_design_bilinear_a2d(&za, &pa, ka, m, &mut zd, &mut pd, &mut kd).unwrap();
                let mut b_ref = vec![0.0; n + 1];
                let mut a_ref = b_ref.clone();
                iir_design_d2tf(&zd, &pd, n, kd, &mut b_ref, &mut a_ref).unwrap();

                for i in 0..=n {
                    assert_abs_diff_eq!(bd[i].re, b_ref[i], epsilon = 1e-5);
                    assert_abs_diff_eq!(ad[i].re, a_ref[i], epsilon = 1e-5);
                    assert_abs_diff_eq!(bd[i].im, 0.0, epsilon = 1e-5);
                    assert_abs_diff_eq!(ad[i].im, 0.0, epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_iirdes_bilinear_z_rejects_invalid_orders() {
        let one = Complex32::new(1.0, 0.0);
        for (b_order, a_order) in [(0, 0), (2, 1)] {
            let mut bd = [one; 3];
            let mut ad = bd;
            assert!(iir_design_bilinear_z(&[], b_order, &[], a_order, 1.0, &mut bd, &mut ad).is_err());
            assert_eq!(bd, [one; 3]);
            assert_eq!(ad, [one; 3]);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cplxpair_n6)]
    fn test_iirdes_cplxpair_n6() {
        let tol = 1e-8f32;

        let r = [
            Complex32::new(0.980066577841242, 0.198669330795061),
            Complex32::new(5.000000000000000, 0.000000000000000),
            Complex32::new(-0.416146836547142, 0.909297426825682),
            Complex32::new(0.980066577841242, -0.198669330795061),
            Complex32::new(0.300000000000000, 0.000000000000000),
            Complex32::new(-0.416146836547142, -0.909297426825682),
        ];

        let mut p = [Complex32::new(0.0, 0.0); 6];

        let ptest = [
            Complex32::new(-0.416146836547142, -0.909297426825682),
            Complex32::new(-0.416146836547142, 0.909297426825682),
            Complex32::new(0.980066577841242, -0.198669330795061),
            Complex32::new(0.980066577841242, 0.198669330795061),
            Complex32::new(0.300000000000000, 0.000000000000000),
            Complex32::new(5.000000000000000, 0.000000000000000),
        ];

        // compute complex pairs
        find_conjugate_pairs(&r, 6, 1e-6f32, &mut p).unwrap();

        // run test
        for i in 0..6 {
            assert_abs_diff_eq!(p[i].re, ptest[i].re, epsilon = tol);
            assert_abs_diff_eq!(p[i].im, ptest[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cplxpair_n20)]
    fn test_iirdes_cplxpair_n20() {
        let tol = 1e-8f32;

        let r = [
            Complex32::new(-0.340396183901119, 1.109902927794652),
            Complex32::new(1.148964416793990, 0.000000000000000),
            Complex32::new(0.190037889511651, 0.597517076404221),
            Complex32::new(-0.340396183901119, -1.109902927794652),
            Complex32::new(0.890883293686046, 0.000000000000000),
            Complex32::new(-0.248338528396292, -0.199390430636670),
            Complex32::new(0.190037889511651, -0.597517076404221),
            Complex32::new(0.003180396218998, 0.000000000000000),
            Complex32::new(0.261949046540733, -0.739400953405199),
            Complex32::new(0.261949046540733, 0.739400953405199),
            Complex32::new(0.309342570837113, 0.000000000000000),
            Complex32::new(0.035516103001236, 0.000000000000000),
            Complex32::new(-0.184159864176452, -0.240335024546875),
            Complex32::new(-0.485244526317243, 0.452251520655749),
            Complex32::new(-0.485244526317243, -0.452251520655749),
            Complex32::new(-0.581633365450190, 0.000000000000000),
            Complex32::new(-0.248338528396292, 0.199390430636670),
            Complex32::new(-0.184159864176452, 0.240335024546875),
            Complex32::new(1.013685316242435, 0.000000000000000),
            Complex32::new(-0.089598596934739, 0.000000000000000),
        ];

        let mut p = [Complex32::new(0.0, 0.0); 20];

        let ptest = [
            Complex32::new(-0.485244526317243, -0.452251520655749),
            Complex32::new(-0.485244526317243, 0.452251520655749),
            Complex32::new(-0.340396183901119, -1.109902927794652),
            Complex32::new(-0.340396183901119, 1.109902927794652),
            Complex32::new(-0.248338528396292, -0.199390430636670),
            Complex32::new(-0.248338528396292, 0.199390430636670),
            Complex32::new(-0.184159864176452, -0.240335024546875),
            Complex32::new(-0.184159864176452, 0.240335024546875),
            Complex32::new(0.190037889511651, -0.597517076404221),
            Complex32::new(0.190037889511651, 0.597517076404221),
            Complex32::new(0.261949046540733, -0.739400953405199),
            Complex32::new(0.261949046540733, 0.739400953405199),
            Complex32::new(-0.581633365450190, 0.000000000000000),
            Complex32::new(-0.089598596934739, 0.000000000000000),
            Complex32::new(0.003180396218998, 0.000000000000000),
            Complex32::new(0.035516103001236, 0.000000000000000),
            Complex32::new(0.309342570837113, 0.000000000000000),
            Complex32::new(0.890883293686046, 0.000000000000000),
            Complex32::new(1.013685316242435, 0.000000000000000),
            Complex32::new(1.148964416793990, 0.000000000000000),
        ];

        // compute complex pairs
        find_conjugate_pairs(&r, 20, 1e-6f32, &mut p).unwrap();

        // run test
        for i in 0..20 {
            assert_abs_diff_eq!(p[i].re, ptest[i].re, epsilon = tol);
            assert_abs_diff_eq!(p[i].im, ptest[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_dzpk2sosf)]
    fn test_iirdes_dzpk2sosf() {
        const N: usize = 4;
        let fc = 0.25f32;

        let mut za = vec![Complex32::new(0.0, 0.0); N];
        let mut pa = vec![Complex32::new(0.0, 0.0); N];
        let mut ka = Complex32::new(0.0, 0.0);
        crate::filter::iir::design::iir_design_butter_analog(N, &mut za, &mut pa, &mut ka).unwrap();

        let mut zd = [Complex32::new(0.0, 0.0); N];
        let mut pd = [Complex32::new(0.0, 0.0); N];
        let mut kd = Complex32::new(0.0, 0.0);
        let m = 1.0 / (std::f32::consts::PI * fc).tan();
        crate::filter::iir::design::iir_design_bilinear_a2d(&za, &pa, ka, m, &mut zd, &mut pd, &mut kd).unwrap();

        let l = if N % 2 == 1 { (N + 1) / 2 } else { N / 2 };
        let mut b = vec![0.0f32; 3 * l];
        let mut a = vec![0.0f32; 3 * l];

        iir_design_d2sos(&zd, &pd, N, kd, &mut b, &mut a).unwrap();

        // TODO there are no tests here
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_isstable_n2_yes)]
    fn test_iirdes_isstable_n2_yes() {
        // initialize pre-determined coefficient array
        // for 2^nd-order low-pass Butterworth filter
        // with cutoff frequency 0.25
        let a = [1.0f32, 0.0f32, 0.171572875253810f32];
        let stable = iir_design_is_stable(&a, 3).unwrap();
        assert!(stable);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_isstable_n2_no)]
    fn test_iirdes_isstable_n2_no() {
        // initialize unstable filter
        let a = [1.0f32, 0.0f32, 1.171572875253810f32];
        let stable = iir_design_is_stable(&a, 3).unwrap();
        assert!(!stable);
    }

    #[test]
    fn test_iirdes_isstable_n1() {
        assert!(iir_design_is_stable(&[1.0, -0.5], 2).unwrap());
        assert!(!iir_design_is_stable(&[1.0, -1.5], 2).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_butter_2)]
    fn test_iirdes_butter_2() {
        // design butterworth filter
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        iir_design(
            IirFilterShape::Butter,
            IirBandType::Lowpass,
            IirFormat::TransferFunction,
            2,       // order
            0.25f32, // fc, normalized cut-off frequency
            0.0f32,  // f0, center frequency (ignored for low-pass filter)
            1.0f32,  // Ap, pass-band ripple (ignored for Butterworth)
            40.0f32, // As, stop-band attenuation (ignored for Butterworth)
            &mut b,
            &mut a,
        )
        .unwrap();

        // initialize pre-determined coefficient array
        // for 2^nd-order low-pass Butterworth filter
        // with cutoff frequency 0.25
        let a_test = [1.0f32, 0.0f32, 0.171572875253810f32];
        let b_test = [0.292893218813452f32, 0.585786437626905f32, 0.292893218813452f32];

        // Ensure data are equal to within tolerance
        let tol = 1e-6f32; // error tolerance
        for i in 0..3 {
            assert_abs_diff_eq!(b[i], b_test[i], epsilon = tol);
            assert_abs_diff_eq!(a[i], a_test[i], epsilon = tol);
        }
    }

    // Test infinite impulse response filter design responses
    //
    //  H(z)
    //   ^          fc     fs
    //   |           |     |
    //   |************. . . . . . . . . . H0
    //   |/\/\/\/\/\/\  *
    //   |************\  *. . . . . . . . H1
    //   |           * \  *
    //   |           *  \  *
    //   |           *   \  ************* H2
    //   |           *    \ /^\ /^\ /^\ /|
    //   |           *     |   |   |   | |
    //   0           fc    fs            0.5

    fn test_iirdes_ellip_lowpass(n: usize, fc: f32, fs: f32, ap: f32, as_: f32) {
        let tol = 1e-3; // error tolerance [dB], yes, that's dB
        let nfft = 800; // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::<f32, f32>::new_prototype(
            IirFilterShape::Ellip,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            ap,
            as_,
        )
        .unwrap();

        let h0 = 0.0;
        let h1 = -ap;
        let h2 = -as_;

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: 0.0, fmax:  fc, pmin: h1 - tol, pmax: h0 + tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin:  fs, fmax: 0.5, pmin:      0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_lowpass_0)]
    fn test_iirdes_ellip_lowpass_0() {
        test_iirdes_ellip_lowpass(5, 0.20, 0.30, 1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_lowpass_1)]
    fn test_iirdes_ellip_lowpass_1() {
        test_iirdes_ellip_lowpass(5, 0.05, 0.09, 1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_lowpass_2)]
    fn test_iirdes_ellip_lowpass_2() {
        test_iirdes_ellip_lowpass(5, 0.20, 0.43, 1.0, 100.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_lowpass_3)]
    fn test_iirdes_ellip_lowpass_3() {
        test_iirdes_ellip_lowpass(5, 0.20, 0.40, 0.1, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_lowpass_4)]
    fn test_iirdes_ellip_lowpass_4() {
        test_iirdes_ellip_lowpass(15, 0.35, 0.37, 0.1, 120.0);
    }

    fn test_iirdes_cheby1_lowpass(n: usize, fc: f32, fs: f32, ap: f32) {
        let tol = 1e-3; // error tolerance [dB], yes, that's dB
        let nfft = 800; // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::<f32, f32>::new_prototype(
            IirFilterShape::Cheby1,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            ap,
            60.0,
        )
        .unwrap();

        let h0 = 0.0;
        let h1 = -ap;
        let h2 = -60.0;

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: 0.0, fmax:  fc, pmin: h1 - tol, pmax: h0 + tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin:  fs, fmax: 0.5, pmin:      0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby1_lowpass_0)]
    fn test_iirdes_cheby1_lowpass_0() {
        test_iirdes_cheby1_lowpass(5, 0.20, 0.36, 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby1_lowpass_1)]
    fn test_iirdes_cheby1_lowpass_1() {
        test_iirdes_cheby1_lowpass(5, 0.05, 0.14, 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby1_lowpass_2)]
    fn test_iirdes_cheby1_lowpass_2() {
        test_iirdes_cheby1_lowpass(5, 0.20, 0.36, 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby1_lowpass_3)]
    fn test_iirdes_cheby1_lowpass_3() {
        test_iirdes_cheby1_lowpass(5, 0.20, 0.40, 0.1);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby1_lowpass_4)]
    fn test_iirdes_cheby1_lowpass_4() {
        test_iirdes_cheby1_lowpass(15, 0.35, 0.38, 0.1);
    }

    fn test_iirdes_cheby2_lowpass(n: usize, fp: f32, fc: f32, as_: f32) {
        let tol = 1e-3; // error tolerance [dB], yes, that's dB
        let nfft = 800; // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::new_prototype(
            IirFilterShape::Cheby2,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            0.1,
            as_,
        )
        .unwrap();

        let h0 = 0.0;
        let h1 = -3.0;
        let h2 = -as_;

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: 0.0, fmax:  fp, pmin: h1 - tol, pmax: h0 + tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin:  fc, fmax: 0.5, pmin:      0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby2_lowpass_0)]
    fn test_iirdes_cheby2_lowpass_0() {
        test_iirdes_cheby2_lowpass(5, 0.08, 0.20, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby2_lowpass_1)]
    fn test_iirdes_cheby2_lowpass_1() {
        test_iirdes_cheby2_lowpass(5, 0.02, 0.05, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby2_lowpass_2)]
    fn test_iirdes_cheby2_lowpass_2() {
        test_iirdes_cheby2_lowpass(5, 0.07, 0.20, 70.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby2_lowpass_3)]
    fn test_iirdes_cheby2_lowpass_3() {
        test_iirdes_cheby2_lowpass(5, 0.09, 0.20, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_cheby2_lowpass_4)]
    fn test_iirdes_cheby2_lowpass_4() {
        test_iirdes_cheby2_lowpass(15, 0.30, 0.35, 70.0);
    }

    fn test_iirdes_butter_lowpass(n: usize, fc: f32, fs: f32) {
        let tol = 1e-3; // error tolerance [dB], yes, that's dB
        let nfft = 800; // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::new_prototype(
            IirFilterShape::Butter,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            1.0,
            60.0,
        )
        .unwrap();

        let h0 = 0.0;
        let h1 = -3.0;
        let h2 = -60.0;

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: 0.0, fmax: 0.98 * fc, pmin: h1 - tol, pmax: h0 + tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin:  fs, fmax:       0.5, pmin:      0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_butter_lowpass_0)]
    fn test_iirdes_butter_lowpass_0() {
        test_iirdes_butter_lowpass(5, 0.20, 0.40);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_butter_lowpass_1)]
    fn test_iirdes_butter_lowpass_1() {
        test_iirdes_butter_lowpass(5, 0.05, 0.19);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_butter_lowpass_2)]
    fn test_iirdes_butter_lowpass_2() {
        test_iirdes_butter_lowpass(5, 0.20, 0.40);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_butter_lowpass_3)]
    fn test_iirdes_butter_lowpass_3() {
        test_iirdes_butter_lowpass(5, 0.20, 0.40);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_butter_lowpass_4)]
    fn test_iirdes_butter_lowpass_4() {
        test_iirdes_butter_lowpass(15, 0.35, 0.41);
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_highpass)]
    fn test_iirdes_ellip_highpass() {
        let n = 9;
        let fc = 0.2;
        let ap = 0.1;
        let as_ = 60.0;

        let tol = 1e-3;
        let nfft = 800;

        let q = IirFilter::new_prototype(
            IirFilterShape::Ellip,
            IirBandType::Highpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            ap,
            as_,
        )
        .unwrap();

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin:   -0.5, fmax:   -fc, pmin: -ap - tol, pmax:        tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin: -0.184, fmax: 0.184, pmin:       0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin:     fc, fmax:   0.5, pmin: -ap - tol, pmax:        tol, test_lo:  true, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_bandpass)]
    fn test_iirdes_ellip_bandpass() {
        let n = 9;
        let fc = 0.3;
        let f0 = 0.35;
        let ap = 0.1;
        let as_ = 60.0;

        let tol = 1e-3;
        let nfft = 2400;

        let q = IirFilter::new_prototype(
            IirFilterShape::Ellip,
            IirBandType::Bandpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            f0,
            ap,
            as_,
        )
        .unwrap();

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin:   -0.5, fmax: -0.396, pmin:       0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.388, fmax: -0.301, pmin: -ap - tol, pmax:        tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin: -0.293, fmax:  0.293, pmin:       0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin:  0.301, fmax:  0.388, pmin: -ap - tol, pmax:        tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin:  0.396, fmax:    0.5, pmin:       0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_ellip_bandstop)]
    fn test_iirdes_ellip_bandstop() {
        let n = 9;
        let fc = 0.3;
        let f0 = 0.35;
        let ap = 0.1;
        let as_ = 60.0;

        let tol = 1e-3;
        let nfft = 2400;

        let q = IirFilter::new_prototype(
            IirFilterShape::Ellip,
            IirBandType::Bandstop,
            IirFormat::SecondOrderSections,
            n,
            fc,
            f0,
            ap,
            as_,
        )
        .unwrap();

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin:   -0.5, fmax: -0.391, pmin: -ap - tol, pmax:        tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin: -0.387, fmax: -0.306, pmin:       0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.298, fmax:  0.298, pmin: -ap - tol, pmax:        tol, test_lo:  true, test_hi: true },
            PsdRegion { fmin:  0.306, fmax:  0.387, pmin:       0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin:  0.391, fmax:    0.5, pmin: -ap - tol, pmax:        tol, test_lo:  true, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirdes_bessel)]
    fn test_iirdes_bessel() {
        let n = 9;
        let fc = 0.1;
        let nfft = 960;

        let q = IirFilter::new_prototype(
            IirFilterShape::Bessel,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            1.0,
            60.0,
        )
        .unwrap();

        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin:   -0.5, fmax: -0.305, pmin:  0.0, pmax: -60.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.095, fmax:  0.095, pmin: -3.0, pmax:   0.1, test_lo:  true, test_hi: true },
            PsdRegion { fmin:  0.305, fmax:    0.5, pmin:  0.0, pmax: -60.0, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }
}
