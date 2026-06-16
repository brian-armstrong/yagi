mod bessel;
mod butter;
mod cheby1;
mod cheby2;
mod ellip;
mod pll;

pub use bessel::*;
pub use butter::*;
pub use cheby1::*;
pub use cheby2::*;
pub use ellip::*;
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

/// IIR filter design shape
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirFilterShape {
    Butter,
    Cheby1,
    Cheby2,
    Ellip,
    Bessel,
}

/// IIR filter band type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirBandType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

/// IIR filter format
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirFormat {
    TransferFunction,
    SecondOrderSections,
}


/// Sorts array _z of complex numbers into complex conjugate pairs to
/// within a tolerance. Conjugate pairs are ordered by increasing real
/// component with the negative imaginary element first. All pure-real
/// elements are placed at the end of the array.
///
/// Example:
///      v:              liquid_cplxpair(v):
///      10 + j*3        -3 - j*4
///       5 + j*0         3 + j*4
///      -3 + j*4        10 - j*3
///      10 - j*3        10 + j*3
///       3 + j*0         3 + j*0
///      -3 + j*4         5 + j*0
/// 
/// # Arguments
/// 
/// * `z` - complex array (size _n)
/// * `n` - number of elements in _z
/// * `tol` - tolerance for finding complex pairs
/// * `p` - resulting pairs, pure real values of _z at end
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
        
        for j in i+1..n {
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
    

/// post-process cleanup used with liquid_cplxpair
///
/// once pairs have been identified and ordered, this method
/// will clean up the result by ensuring the following:
///  * pairs are perfect conjugates
///  * pairs have negative imaginary component first
///  * pairs are ordered by increasing real component
///  * pure-real elements are ordered by increasing value
///
/// # Arguments
/// 
/// * `p` - pre-processed complex array [size: _n x 1]
/// * `n` - array length
/// * `num_pairs` - number of complex conjugate pairs
fn find_conjugate_pairs_cleanup(p: &mut [Complex32], n: usize, num_pairs: usize) -> Result<()> {
    // force pairs to be perfect conjugates with
    // negative imaginary component first
    for i in 0..num_pairs {
        p[2*i+0] = if p[2*i+1].im < 0.0 { p[2*i+1] } else { p[2*i+1].conj() };
        p[2*i+1] = p[2*i+0].conj();
    }

    // sort conjugate pairs
    for i in 0..num_pairs {
        for j in (i+1..num_pairs).rev() {
            if p[2*(j-1)].re > p[2*j].re {
                // swap pairs
                let temp = p[2*(j-1)];    
                p[2*(j-1)] = p[2*j];
                p[2*j] = temp;

                let temp = p[2*(j-1)+1];
                p[2*(j-1)+1] = p[2*j+1];
                p[2*j+1] = temp;
            }
        }
    }

    // sort pure-real elements
    for i in 2*num_pairs..n {   
        for j in (i+1..n).rev() {
            if p[j-1].re > p[j].re {
                let temp = p[j-1];
                p[j-1] = p[j];
                p[j] = temp;
            }
        }
    }   

    Ok(())
}

/// Compute frequency pre-warping factor.
/// 
/// # Arguments
/// 
/// * `btype` - band type (e.g. `IirBandType::Lowpass`)
/// * `fc` - low-pass cutoff frequency
/// * `f0` - center frequency (band-pass|stop cases only)
/// 
/// # Returns
/// 
/// Frequency pre-warping factor
pub fn iir_design_freqprewarp(btype: IirBandType, fc: f32, f0: f32) -> f32 {
    // See [Constantinides:1967]
    match btype {
        IirBandType::Lowpass => (PI * fc).tan(),
        IirBandType::Highpass => -(PI * fc).cos() / (PI * fc).sin(),
        IirBandType::Bandpass => {
            ((2.0 * PI * fc).cos() - (2.0 * PI * f0).cos()) / (2.0 * PI * fc).sin()
        }
        IirBandType::Bandstop => {
            (2.0 * PI * fc).sin() / ((2.0 * PI * fc).cos() - (2.0 * PI * f0).cos())
        }
    }
}

/// convert analog zeros, poles, gain to digital zeros, poles gain
/// 
/// The filter order is characterized by the number of analog
/// poles.  The analog filter may have up to pa.len() zeros.
/// The number of digital zeros and poles is equal to pa.len().
/// 
/// # Arguments
/// 
/// * `za` - analog zeros
/// * `pa` - analog poles
/// * `ka` - nominal gain (NOTE: this does not necessarily carry over from analog gain)
/// * `m` - frequency pre-warping factor
/// * `zd` - digital zeros (output)
/// * `pd` - digital poles (output)
/// * `kd` - digital gain (output)
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

/// compute bilinear z-transform using polynomial expansion in numerator and
/// denominator
/// <pre>
///          b[0] + b[1]*s + ... + b[nb]*s^(nb-1)
/// H(s) =   ------------------------------------
///          a[0] + a[1]*s + ... + a[na]*s^(na-1)
/// </pre>
/// computes H(z) = H( s -> _m*(z-1)/(z+1) ) and expands as
/// <pre>
///          bd[0] + bd[1]*z^-1 + ... + bd[nb]*z^-n
/// H(z) =   --------------------------------------
///          ad[0] + ad[1]*z^-1 + ... + ad[nb]*z^-m
/// </pre>
/// 
/// # Arguments
/// 
/// * `b` - numerator array, [size: _b_order+1]
/// * `b_order` - polynomial order of _b
/// * `a` - denominator array, [size: _a_order+1]
/// * `a_order` - polynomial order of _a
/// * `m` - bilateral warping factor
/// * `bd` - output digital filter numerator, [size: _b_order+1]
/// * `ad` - output digital filter numerator, [size: _a_order+1]
pub fn iir_design_bilinear_z(
    b: &[Complex32],
    b_order: usize,
    a: &[Complex32],
    a_order: usize,
    m: f32,
    bd: &mut [Complex32],
    ad: &mut [Complex32],
) -> Result<()> {
    if b_order > a_order {
        return Err(Error::Range("numerator order cannot be higher than denominator".into()));
    }

    let nb = b_order + 1;
    let na = a_order + 1;

    // clear output arrays
    for i in 0..na {
        bd[i] = Complex32::new(0.0, 0.0);
        ad[i] = Complex32::new(0.0, 0.0);
    }

    // temporary polynomial: (1 + 1/z)^(k) * (1 - 1/z)^(n-k)
    let mut poly_1pz = vec![Complex32::new(0.0, 0.0); na];

    let mut mk = Complex32::new(1.0, 0.0);

    // multiply denominator by ((1-1/z)/(1+1/z))^na and expand
    for i in 0..na {
        // expand the polynomial (1+x)^i * (1-x)^(_a_order-i)
        poly_expandbinomial_pm(a_order,
                               a_order-i,
                               &mut poly_1pz);

        // accumulate polynomial coefficients
        for j in 0..na {
            ad[j] += a[i] * mk * poly_1pz[j];
        }

        // update multiplier
        mk *= m;
    }

    // multiply numerator by ((1-1/z)/(1+1/z))^na and expand
    mk = Complex32::new(1.0, 0.0);
    for i in 0..nb {
        // expand the polynomial (1+x)^i * (1-x)^(_a_order-i)
        poly_expandbinomial_pm(a_order,
                               a_order-i,
                               &mut poly_1pz);  

        // accumulate polynomial coefficients
        for j in 0..na {
            bd[j] += b[i] * mk * poly_1pz[j];
        }

        // update multiplier
        mk *= m;
    }

    // normalize by a[0]
    let a0_inv = 1.0 / ad[0];
    for i in 0..na {
        bd[i] *= a0_inv;
        ad[i] *= a0_inv;
    }

    Ok(())
}

/// convert discrete z/p/k form to transfer function form
/// 
/// # Arguments
/// 
/// * `zd` - digital zeros (length: _n)
/// * `pd` - digital poles (length: _n)
/// * `n` - filter order
/// * `k` - digital gain
/// * `b` - output numerator (length: _n+1)
/// * `a` - output denominator (length: _n+1)
pub fn iir_design_d2tf(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    k: Complex32,
    b: &mut [f32],
    a: &mut [f32],
) -> Result<()> {
    let mut q = vec![Complex32::new(0.0, 0.0); n + 1];

    // expand poles
    poly_expandroots(pd, n, &mut q);
    for i in 0..=n {
        a[i] = q[n-i].re;
    }   

    // expand zeros
    poly_expandroots(zd, n, &mut q);
    for i in 0..=n {
        b[i] = (q[n-i] * k).re;
    }

    Ok(())
}

/// convert discrete z/p/k form to second-order sections form
/// 
/// L is the number of sections in the cascade:
///      r = n % 2
///      L = (n - r) / 2;
/// 
/// # Arguments
/// 
/// * `zd` - digital zeros (length: n)
/// * `pd` - digital poles (length: n)
/// * `n` - number of poles, zeros
/// * `kd` - gain
/// * `b` - output numerator matrix (size (L+r) x 3)
/// * `a` - output denominator matrix (size (L+r) x 3)
pub fn iir_design_d2sos(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    k: Complex32,
    b: &mut [f32],
    a: &mut [f32],
) -> Result<()> {
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
    let r = n % 2;        // odd/even order
    let l = (n - r) / 2;    // filter semi-length

    for i in 0..l {
        let p0 = -pp[2*i+0];
        let p1 = -pp[2*i+1];

        let z0 = -zp[2*i+0];
        let z1 = -zp[2*i+1];

        // expand complex pole pairs
        a[3*i+0] = 1.0;
        a[3*i+1] = (p0 + p1).re;
        a[3*i+2] = (p0 * p1).re;

        // expand complex zero pairs
        b[3*i+0] = 1.0;
        b[3*i+1] = (z0 + z1).re;
        b[3*i+2] = (z0 * z1).re;
    }

    // add remaining zero/pole pair if order is odd
    if r == 1 {
        let p0 = -pp[n-1];
        let z0 = -zp[n-1];

        // expand complex pole pair
        a[3*l+0] = 1.0;
        a[3*l+1] = p0.re;
        a[3*l+2] = 0.0;

        // expand complex zero pair
        b[3*l+0] = 1.0;
        b[3*l+1] = z0.re;
        b[3*l+2] = 0.0;
    }

    // distribute gain equally amongst all feed-forward coefficients
    let k   = k.re;
    let sgn = if k < 0.0 { -1.0 } else { 1.0 };
    let g   = (k * sgn).powf(1.0 / (l+r) as f32);

    // adjust gain of first element
    for i in 0..l+r {
        b[3*i+0] *= g;
        b[3*i+1] *= g;
        b[3*i+2] *= g;
    }

    // apply sign to first section (handle case where gain is negative)
    b[0] *= sgn;
    b[1] *= sgn;
    b[2] *= sgn;

    Ok(())
}

/// digital z/p/k low-pass to high-pass transformation
/// 
/// # Arguments
/// 
/// * `zd` - digital zeros (low-pass prototype)
/// * `pd` - digital poles (low-pass prototype)
/// * `n` - low-pass filter order
/// * `zdt` - output digital zeros transformed [length: n]
/// * `pdt` - output digital poles transformed [length: n]
pub fn iir_design_lp2hp(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    zdt: &mut [Complex32],
    pdt: &mut [Complex32],
) -> Result<()> {
    for i in 0..n {
        zdt[i] = -zd[i];
        pdt[i] = -pd[i];
    }

    Ok(())
}

/// digital z/p/k low-pass to band-pass transformation
/// 
/// # Arguments
/// 
/// * `zd` - digital zeros (low-pass prototype)
/// * `pd` - digital poles (low-pass prototype)
/// * `n` - low-pass filter order
/// * `f0` - center frequency
/// * `zdt` - output digital zeros transformed [length: 2*n]
/// * `pdt` - output digital poles transformed [length: 2*n]
pub fn iir_design_lp2bp(
    zd: &[Complex32],
    pd: &[Complex32],
    n: usize,
    f0: f32,
    zdt: &mut [Complex32],
    pdt: &mut [Complex32],
) -> Result<()> {
    let c0 = (2.0 * PI * f0).cos(); 

    // transform zeros, poles using quadratic formula
    for i in 0..n {
        let t0 = 1.0 + zd[i];
        zdt[2*i+0] = 0.5 * (c0 * t0 + (c0*c0*t0*t0 - 4.0 * zd[i]).sqrt());
        zdt[2*i+1] = 0.5 * (c0 * t0 - (c0*c0*t0*t0 - 4.0 * zd[i]).sqrt());

        let t0 = 1.0 + pd[i];
        pdt[2*i+0] = 0.5 * (c0 * t0 + (c0*c0*t0*t0 - 4.0 * pd[i]).sqrt());
        pdt[2*i+1] = 0.5 * (c0 * t0 - (c0*c0*t0*t0 - 4.0 * pd[i]).sqrt());
    }

    Ok(())
}

/// IIR filter design template
/// 
/// # Arguments
/// 
/// * `ftype` - filter type (e.g. IirFilterShape::Butter)
/// * `btype` - band type (e.g. IirBandType::Bandpass)
/// * `format` - coefficients format (e.g. IirFormat::Sos)
/// * `n` - filter order
/// * `fc` - low-pass prototype cut-off frequency
/// * `f0` - center frequency (band-pass, band-stop)
/// * `ap` - pass-band ripple in dB
/// * `as` - stop-band ripple in dB
/// * `b` - output numerator
/// * `a` - output denominator
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
    if fc <= 0.0 || fc >= 0.5 {
        return Err(Error::Config("cutoff frequency out of range".into()));
    }
    if f0 < 0.0 || f0 > 0.5 {
        return Err(Error::Config("center frequency out of range".into()));
    }
    if ap <= 0.0 {
        return Err(Error::Config("pass-band ripple out of range".into()));
    }
    if as_ <= 0.0 {
        return Err(Error::Config("stop-band ripple out of range".into()));
    }
    if n == 0 {
        return Err(Error::Config("filter order must be > 0".into()));
    }

    let mut order = n;

    // analog poles/zeros/gain
    let mut pa = vec![Complex32::new(0.0, 0.0); order];  // analog poles
    let mut za = vec![Complex32::new(0.0, 0.0); order];  // analog zeros
    let mut ka = Complex32::new(0.0, 0.0);               // analog gain
    let k0;                                              // nominal digital gain

    // derived values
    let r = n % 2;        // odd/even filter order

    // compute zeros and poles of analog prototype
    match ftype {
        IirFilterShape::Butter => {
            // Butterworth filter design : no zeros, n poles
            k0 = Complex32::new(1.0, 0.0);
            if iir_design_butter_analog(order, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (butterworth)".into()));
            }
        },
        IirFilterShape::Cheby1 => {
            // Cheby-I filter design : no zeros, n poles, pass-band ripple
            let epsilon = (10.0f32.powf(ap / 10.0) - 1.0).sqrt();
            k0 = if r == 1 { Complex32::new(1.0, 0.0) } else { Complex32::new(1.0 / (1.0 + epsilon * epsilon).sqrt(), 0.0) };
            if iir_design_cheby1_analog(order, epsilon, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (cheby1)".into()));
            }
        },
        IirFilterShape::Cheby2 => {
            // Cheby-II filter design : n-r zeros, n poles, stop-band ripple
            let epsilon = 10.0f32.powf(-as_ / 20.0);
            k0 = Complex32::new(1.0, 0.0);
            if iir_design_cheby2_analog(order, epsilon, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (cheby2)".into()));
            }
        },
        IirFilterShape::Ellip => {
            // elliptic filter design : n-r zeros, n poles, pass/stop-band ripple
            let gp = 10.0f32.powf(-ap / 20.0);
            let gs = 10.0f32.powf(-as_ / 20.0);
            let ep = (1.0 / (gp*gp) - 1.0).sqrt();
            let es = (1.0 / (gs*gs) - 1.0).sqrt();
            k0 = if r == 1 { Complex32::new(1.0, 0.0) } else { Complex32::new(1.0 / (1.0 + ep*ep).sqrt(), 0.0) };
            if iir_design_ellip_analog(order, ep, es, &mut za, &mut pa, &mut ka).is_err() {
                return Err(Error::Internal("could not design analog filter (elliptic)".into()));
            }
        },
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
    let mut zd = vec![Complex32::new(0.0, 0.0); 2*order];
    let mut pd = vec![Complex32::new(0.0, 0.0); 2*order];
    let mut kd = Complex32::new(0.0, 0.0);

    let m = iir_design_freqprewarp(btype, fc, f0);

    if iir_design_bilinear_a2d(&za, &pa, k0, m, &mut zd, &mut pd, &mut kd).is_err() {
        return Err(Error::Internal("could not perform bilinear z-transform".into()));
    }

    // negate zeros, poles for high-pass and band-stop cases
    if btype == IirBandType::Highpass || btype == IirBandType::Bandstop {
        let mut pd_tmp = vec![Complex32::new(0.0, 0.0); 2*order];
        let mut zd_tmp = vec![Complex32::new(0.0, 0.0); 2*order];

        if iir_design_lp2hp(&zd, &pd, order, &mut zd_tmp, &mut pd_tmp).is_err() {
            return Err(Error::Internal("could not perform high-pass transformation".into()));
        }

        zd.copy_from_slice(&zd_tmp);
        pd.copy_from_slice(&pd_tmp);
    }

    // transform zeros, poles in band-pass, band-stop cases 
    // NOTE: this also doubles the filter order
    if btype == IirBandType::Bandpass || btype == IirBandType::Bandstop {
        let mut zd1 = vec![Complex32::new(0.0, 0.0); 2*n];
        let mut pd1 = vec![Complex32::new(0.0, 0.0); 2*n];

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
        order = 2*order;
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

/// Checks stability of iir filter
/// 
/// # Arguments
/// 
/// * `b` - feed-forward coefficients [size: n x 1]
/// * `a` - feed-back coefficients [size: n x 1]
/// * `n` - number of coefficients
/// 
/// # Returns
/// 
/// `true` if the filter is stable, `false` otherwise
pub fn iir_design_is_stable(
    _b: &[f32],
    a: &[f32],
    n: usize,
) -> Result<bool> {
    if n < 2 {
        return Err(Error::Config("filter order too low".into()));
    }

    let mut a_hat = vec![0.0; n];
    for i in 0..n {
        a_hat[i] = a[n-i-1];
    }

    let mut roots = vec![Complex32::new(0.0, 0.0); n-1];
    if polyf_findroots(&a_hat, n, &mut roots).is_err() {
        return Err(Error::Internal("could not find roots of polynomial".into()));
    }

    for i in 0..n-1 {
        if roots[i].norm() > 1.0 {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Compute group delay for an IIR filter
/// 
/// # Arguments
/// 
/// * `b` - filter coefficients array (numerator), [size: nb x 1]
/// * `nb` - filter length (numerator)
/// * `a` - filter coefficients array (denominator), [size: na x 1]
/// * `na` - filter length (denominator)
/// * `fc` - frequency at which delay is evaluated (-0.5 < fc < 0.5)
/// 
/// # Returns
/// 
/// Group delay at the specified frequency
pub fn iir_group_delay(b: &[f32], a: &[f32], fc: f32) -> Result<f32> {
    // validate input
    if b.is_empty() {
        return Err(Error::Config("iir_group_delay(), numerator length must be greater than zero".to_string()));
    }   
    if a.is_empty() {
        return Err(Error::Config("iir_group_delay(), denominator length must be greater than zero".to_string()));
    }   
    if fc < -0.5 || fc > 0.5 {
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
    let mut c0: Complex32;
    for i in 0..nc {
        c0  = c[i] * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * i as f32);
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
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;
    use crate::filter::iir::iirfilt::IirFilter;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_iirfilt};

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
        let a = [
            1.0f32,
            0.0f32,
            0.171572875253810f32
        ];
        let b = [
            0.292893218813452f32,
            0.585786437626905f32,
            0.292893218813452f32
        ];
    
        let stable = iir_design_is_stable(&b, &a, 3).unwrap();
        assert!(stable);
    }
    
    #[test]
    #[autotest_annotate(autotest_iirdes_isstable_n2_no)]
    fn test_iirdes_isstable_n2_no() {
        // initialize unstable filter
        let a = [
            1.0f32,
            0.0f32,
            1.171572875253810f32
        ];
        let b = [
            0.292893218813452f32,
            0.585786437626905f32,
            0.292893218813452f32
        ];
    
        let stable = iir_design_is_stable(&b, &a, 3).unwrap();
        assert!(!stable);
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
            2,        // order
            0.25f32,  // fc, normalized cut-off frequency
            0.0f32,   // f0, center frequency (ignored for low-pass filter)
            1.0f32,   // Ap, pass-band ripple (ignored for Butterworth)
            40.0f32,  // As, stop-band attenuation (ignored for Butterworth)
            &mut b,
            &mut a
        ).unwrap();

        // initialize pre-determined coefficient array
        // for 2^nd-order low-pass Butterworth filter
        // with cutoff frequency 0.25
        let a_test = [1.0f32, 0.0f32, 0.171572875253810f32];
        let b_test = [0.292893218813452f32, 0.585786437626905f32, 0.292893218813452f32];

        // Ensure data are equal to within tolerance
        let tol = 1e-6f32;  // error tolerance
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
        let tol = 1e-3;  // error tolerance [dB], yes, that's dB
        let nfft = 800;    // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::<f32, f32>::new_prototype(
            IirFilterShape::Ellip,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            ap,
            as_
        ).unwrap();
        
        let h0 = 0.0;
        let h1 = -ap;
        let h2 = -as_;

        let regions = [
            PsdRegion { fmin: 0.0, fmax: fc, pmin: h1 - tol, pmax: h0 + tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: fs, fmax: 0.5, pmin: 0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
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
        let tol = 1e-3;  // error tolerance [dB], yes, that's dB
        let nfft = 800;    // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::<f32, f32>::new_prototype(
            IirFilterShape::Cheby1,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            ap,
            60.0
        ).unwrap();
        
        let h0 = 0.0;
        let h1 = -ap;
        let h2 = -60.0;

        let regions = [
            PsdRegion { fmin: 0.0, fmax: fc, pmin: h1 - tol, pmax: h0 + tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: fs, fmax: 0.5, pmin: 0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
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
        let tol = 1e-3;  // error tolerance [dB], yes, that's dB
        let nfft = 800;    // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::new_prototype(
            IirFilterShape::Cheby2,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            0.1,
            as_
        ).unwrap();
        
        let h0 = 0.0;
        let h1 = -3.0;
        let h2 = -as_;

        let regions = [
            PsdRegion { fmin: 0.0, fmax: fp, pmin: h1 - tol, pmax: h0 + tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: fc, fmax: 0.5, pmin: 0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
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
        let tol = 1e-3;  // error tolerance [dB], yes, that's dB
        let nfft = 800;    // number of points to evaluate

        // design filter from prototype
        let q = IirFilter::new_prototype(
            IirFilterShape::Butter,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            n,
            fc,
            0.0,
            1.0,
            60.0
        ).unwrap();
        
        let h0 = 0.0;
        let h1 = -3.0;
        let h2 = -60.0;

        let regions = [
            PsdRegion { fmin: 0.0, fmax: 0.98 * fc, pmin: h1 - tol, pmax: h0 + tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: fs, fmax: 0.5, pmin: 0.0, pmax: h2 + tol, test_lo: false, test_hi: true },
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
            as_
        ).unwrap();

        let regions = [
            PsdRegion { fmin: -0.5, fmax: -fc, pmin: -ap - tol, pmax: tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.184, fmax: 0.184, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: fc, fmax: 0.5, pmin: -ap - tol, pmax: tol, test_lo: true, test_hi: true },
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
            as_
        ).unwrap();

        let regions = [
            PsdRegion { fmin: -0.5, fmax: -0.396, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.388, fmax: -0.301, pmin: -ap - tol, pmax: tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.293, fmax: 0.293, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: 0.301, fmax: 0.388, pmin: -ap - tol, pmax: tol, test_lo: true, test_hi: true },   
            PsdRegion { fmin: 0.396, fmax: 0.5, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
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
            as_
        ).unwrap();

        let regions = [
            PsdRegion { fmin: -0.5, fmax: -0.391, pmin: -ap - tol, pmax: tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.387, fmax: -0.306, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.298, fmax: 0.298, pmin: -ap - tol, pmax: tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.306, fmax: 0.387, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: 0.391, fmax: 0.5, pmin: -ap - tol, pmax: tol, test_lo: true, test_hi: true },
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
            60.0
        ).unwrap();

        let regions = [
            PsdRegion { fmin: -0.5, fmax: -0.305, pmin: 0.0, pmax: -60.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.095, fmax: 0.095, pmin: -3.0, pmax: 0.1, test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.305, fmax: 0.5, pmin: 0.0, pmax: -60.0, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_iirfilt(&q, nfft, &regions).unwrap());
    }
}