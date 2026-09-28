use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::PI;

/// Designs an analog low-pass Chebyshev type I prototype.
///
/// `ep` is the passband ripple factor, a linear value rather than decibels.
/// A ripple of `ap` dB corresponds to `ep = sqrt(10^(ap / 10) - 1)`.
///
/// A Chebyshev type I prototype has no zeros, so `za` is cleared. `pa`
/// receives `n` poles in conjugate pairs, and an odd order puts its one real
/// pole last. `ka` receives the prototype gain.
///
/// `ep` is not validated and must be positive. Zero or negative values
/// produce nonfinite poles or gain without an error.
///
/// # Errors
///
/// Returns an error when `n` is zero.
pub fn iir_design_cheby1_analog(
    n: usize,
    ep: f32,
    za: &mut Vec<Complex32>,
    pa: &mut Vec<Complex32>,
    ka: &mut Complex32,
) -> Result<()> {
    // validate input
    if n == 0 {
        return Err(Error::Config("filter order must be greater than zero".to_string()));
    }

    za.clear();
    pa.clear();

    // temporary values
    let t0 = (1.0 + 1.0 / (ep * ep)).sqrt();
    let tp = (t0 + 1.0 / ep).powf(1.0 / n as f32);
    let tm = (t0 - 1.0 / ep).powf(1.0 / n as f32);

    let b = 0.5 * (tp + tm); // ellipse major axis
    let a = 0.5 * (tp - tm); // ellipse minor axis

    // filter order variables
    let r = n % 2; // odd order?
    let l = (n - r) / 2; // half order

    // compute poles
    for i in 0..l {
        let theta = (2.0 * (i + 1) as f32 + n as f32 - 1.0) * PI / (2.0 * n as f32);
        pa.push(Complex32::new(a * theta.cos(), -b * theta.sin()));
        pa.push(Complex32::new(a * theta.cos(), b * theta.sin()));
    }

    // if filter order is odd, there is an additional pole on the
    // real axis
    if r == 1 {
        pa.push(Complex32::new(-a, 0.0));
    }

    // ensure we have written exactly n poles
    assert_eq!(pa.len(), n);

    // compute analog gain (ignored in digital conversion)
    *ka = if r == 1 { Complex32::new(1.0, 0.0) } else { Complex32::new((1.0 + ep * ep).sqrt().recip(), 0.0) };
    for &p in &pa[..n] {
        *ka *= p;
    }

    Ok(())
}
