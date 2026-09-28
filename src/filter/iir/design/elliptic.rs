use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::PI;

//
// Elliptic filter design
//
//  [Orfanidis:2006] Sophocles J. Orfanidis, "Lecture Notes on
//      Elliptic Filter Design." 2006
//  [Orfanidis:2005] Sophocles J. Orfanidis, source code available
//      on www.ece.rutgers.edu/~orfanidi/hpeq

/// Applies `n` Landen transformations to the elliptic modulus `k`.
///
/// Each step computes `k[i] = (1 - k'[i-1]) / (1 + k'[i-1])`, where
/// `k'[i-1] = sqrt(1 - k[i-1]^2)`. The resulting sequence of decreasing moduli
/// is written to `v[..n]`.
fn landenf(k: f32, n: usize, v: &mut [f32]) -> Result<()> {
    let mut k = k;
    for vi in &mut v[..n] {
        let kp = (1.0 - k * k).sqrt();
        k = (1.0 - kp) / (1.0 + kp);
        *vi = k;
    }
    Ok(())
}

/// Computes the complete elliptic integrals `K(k)` and `K(k')`.
///
/// `k_out` receives `K(k)` and `kp_out` receives `K(k')`, where
/// `k' = sqrt(1 - k^2)` is the complementary modulus. Each integral uses `n`
/// Landen transformations. A modulus too close to zero or one for `f32` to
/// resolve uses an asymptotic approximation instead.
fn ellipkf(k: f32, n: usize, k_out: &mut f32, kp_out: &mut f32) -> Result<()> {
    let kmin = 4e-4f32;
    let kmax = (1.0 - kmin * kmin).sqrt();

    let kp = (1.0 - k * k).sqrt();

    // Floating point resolution limits the range of the
    // computation of K on input _k [Orfanidis:2005]
    if k > kmax {
        let l = -(0.25 * kp).ln();
        *k_out = l + 0.25 * (l - 1.0) * kp * kp;
    } else {
        let mut v = vec![0.0; n];
        landenf(k, n, &mut v)?;
        *k_out = PI * 0.5;
        for &vi in &v[..n] {
            *k_out *= 1.0 + vi;
        }
    };

    if k < kmin {
        let l = -(0.25 * k).ln();
        *kp_out = l + 0.25 * (l - 1.0) * k * k;
    } else {
        let mut vp = vec![0.0; n];
        landenf(kp, n, &mut vp)?;
        *kp_out = PI * 0.5;
        for &vpi in &vp[..n] {
            *kp_out *= 1.0 + vpi;
        }
    }

    Ok(())
}

/// Solves the degree equation for the elliptic modulus of an order-`n` filter.
///
/// `k1_in` is the ratio of the passband to stopband ripple factors,
/// `ep / es`. The integrals and series use `n_iter` terms.
fn ellipdegf(n: f32, k1_in: f32, n_iter: usize) -> Result<f32> {
    let mut k1 = k1_in;
    let mut k1p = k1_in;
    ellipkf(k1_in, n_iter, &mut k1, &mut k1p)?;

    let q1 = (-PI * k1p / k1).exp();
    let q = q1.powf(1.0 / n);

    let mut b = 0.0;
    for m in 0..n_iter {
        b += q.powf((m * (m + 1)) as f32);
    }
    let mut a = 0.0;
    for m in 1..n_iter {
        a += q.powf((m * m) as f32);
    }

    let g = b / (1.0 + 2.0 * a);
    let k = 4.0 * q.sqrt() * g * g;

    Ok(k)
}

/// Evaluates the Jacobian elliptic function `cd(u * K, k)`.
///
/// `u` is a complex argument in units of the quarter period `K = K(k)`. The
/// modulus `k` lies in `[0, 1)`. The result uses `n` Landen transformations.
fn ellip_cdf(u: Complex32, k: f32, n: usize) -> Complex32 {
    let mut wn = (u * PI * 0.5).cos();
    let mut v = vec![0.0; n];
    landenf(k, n, &mut v).unwrap();
    for i in (1..=n).rev() {
        wn = (1.0 + v[i - 1]) * wn / (1.0 + v[i - 1] * wn * wn);
    }
    wn
}

/// Evaluates the Jacobian elliptic function `sn(u * K, k)`.
///
/// The argument, modulus, and iteration count follow [`ellip_cdf`].
fn ellip_snf(u: Complex32, k: f32, n: usize) -> Complex32 {
    let mut wn = (u * PI * 0.5).sin();
    let mut v = vec![0.0; n];
    landenf(k, n, &mut v).unwrap();
    for i in (1..=n).rev() {
        wn = (1.0 + v[i - 1]) * wn / (1.0 + v[i - 1] * wn * wn);
    }
    wn
}

/// Inverts [`ellip_cdf`].
///
/// Returns `u`, in units of the quarter period `K(k)`, such that
/// `cd(u * K, k) = w`. The modulus `k` lies in `[0, 1)`. The result uses `n`
/// Landen transformations.
fn ellip_acdf(w: Complex32, k: f32, n: usize) -> Complex32 {
    let mut v = vec![0.0; n];
    landenf(k, n, &mut v).unwrap();

    let mut w = w;
    for i in 0..n {
        let v1 = if i == 0 { k } else { v[i - 1] };
        w = w / (1.0 + (1.0 - w * w * v1 * v1).sqrt()) * 2.0 / (1.0 + v[i]);
    }

    w.acos() * 2.0 / PI
}

/// Inverts [`ellip_snf`].
///
/// Returns `u`, in units of the quarter period `K(k)`, such that
/// `sn(u * K, k) = w`. The arguments follow [`ellip_acdf`].
fn ellip_asnf(w: Complex32, k: f32, n: usize) -> Complex32 {
    Complex32::new(1.0, 0.0) - ellip_acdf(w, k, n)
}

/// Designs an analog low-pass elliptic prototype.
///
/// `ep` and `es` are the passband and stopband ripple factors, linear values
/// rather than decibels. For a gain of `g` dB below unity, the factor is
/// `sqrt(10^(g / 10) - 1)`. Unit analog frequency is the passband edge.
///
/// `pa` receives `n` poles in conjugate pairs, and an odd order puts its one
/// real pole last. `za` receives `n - n % 2` zeros in conjugate pairs. `ka`
/// receives the prototype gain.
///
/// The inputs are not validated. `n` should be positive, and the ripple
/// factors should be positive with `ep < es`. Other values can
/// produce nonfinite poles, zeros, or gain without an error.
///
/// The elliptic functions use a fixed number of iterations, so a successful
/// return does not guarantee that the design meets its ripple specification.
pub fn iir_design_ellip_analog(
    n: usize,
    ep: f32,
    es: f32,
    za: &mut Vec<Complex32>,
    pa: &mut Vec<Complex32>,
    ka: &mut Complex32,
) -> Result<()> {
    let fp = 1.0 / (2.0 * PI); // pass-band cutoff
    let fs = 1.1 * fp; // stop-band cutoff

    // number of iterations for elliptic integral
    // approximations
    let n_iter = 7;

    let wp = 2.0 * PI * fp;
    let ws = 2.0 * PI * fs;

    let k = wp / ws;
    let k1 = ep / es;

    let mut k_ellip = 0.0;
    let mut k1_ellip = 0.0;
    let mut kp = 0.0;
    let mut k1p = 0.0;

    ellipkf(k, n_iter, &mut k_ellip, &mut kp)?;
    ellipkf(k1, n_iter, &mut k1_ellip, &mut k1p)?;

    let n = n as f32;

    let k = ellipdegf(n, k1, n_iter)?;

    let l = (n / 2.0).floor() as usize;
    let r = (n as usize) % 2;
    let mut u = vec![0.0; l];
    for (i, ui) in u.iter_mut().enumerate() {
        let t = (i + 1) as f32;
        *ui = (2.0 * t - 1.0) / n;
    }
    let mut zeta = vec![Complex32::new(0.0, 0.0); l];
    for i in 0..l {
        zeta[i] = ellip_cdf(Complex32::new(u[i], 0.0), k, n_iter);
    }
    let mut za_tmp = vec![Complex32::new(0.0, 0.0); l];
    for i in 0..l {
        za_tmp[i] = Complex32::new(0.0, 1.0) * wp / (k * zeta[i]);
    }
    let v0 = -Complex32::new(0.0, 1.0) * ellip_asnf(Complex32::new(0.0, 1.0) / ep, k1, n_iter) / n;

    let mut pa_tmp = vec![Complex32::new(0.0, 0.0); l];
    for i in 0..l {
        pa_tmp[i] = wp
            * Complex32::new(0.0, 1.0)
            * ellip_cdf(Complex32::new(u[i], 0.0) - Complex32::new(0.0, 1.0) * v0, k, n_iter);
    }
    let pa0 = wp * Complex32::new(0.0, 1.0) * ellip_snf(Complex32::new(0.0, 1.0) * v0, k, n_iter);

    za.clear();
    pa.clear();

    for &p in &pa_tmp[..l] {
        pa.push(p);
        pa.push(p.conj());
    }

    if r != 0 {
        pa.push(pa0);
    }

    if pa.len() != n as usize {
        return Err(Error::Internal("Invalid derived order (poles)".into()));
    }

    for &z in &za_tmp[..l] {
        za.push(z);
        za.push(z.conj());
    }

    if za.len() != 2 * l {
        return Err(Error::Internal("Invalid derived order (zeros)".into()));
    }

    *ka = if r == 1 { Complex32::new(1.0, 0.0) } else { Complex32::new(1.0 / (1.0 + ep * ep).sqrt(), 0.0) };
    for &p in &pa[..n as usize] {
        *ka *= p;
    }
    for &z in &za[..2 * l] {
        *ka /= z;
    }

    Ok(())
}
