use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f64::consts::PI;

use libm::{pow, sqrt};

/// Compute analog zeros, poles, gain of low-pass Chebyshev Type II filter, grouping
/// complex conjugates together. If the filter order is odd, the single real pole
/// is at the end of the array.
///
/// # Arguments
///
/// * `n` - filter order
/// * `es` - epsilon, related to stop-band ripple
/// * `za` - output analog zeros [length: 2*floor(_n/2)]
/// * `pa` - output analog poles [length: _n]
/// * `ka` - output analog gain
pub fn iir_design_cheby2_analog(
    n: usize,
    es: f32,
    za: &mut Vec<Complex32>,
    pa: &mut Vec<Complex32>,
    ka: &mut Complex32,
) -> Result<()> {
    // validate input
    if n == 0 {
        return Err(Error::Config("filter order must be greater than zero".into()));
    }

    za.clear();
    pa.clear();

    // temporary values
    let es_wide: f64 = es as f64;
    let es_squared: f64 = es_wide * es_wide;
    let t0: f64 = sqrt(1.0 + 1.0 / es_squared);
    let tp: f64 = pow(t0 + 1.0 / es_wide, 1.0 / n as f64);
    let tm: f64 = pow(t0 - 1.0 / es_wide, 1.0 / n as f64);

    let b = 0.5 * (tp + tm); // ellipse major axis
    let a = 0.5 * (tp - tm); // ellipse minor axis

    // filter order variables
    let r = n % 2; // odd order?
    let l = (n - r) / 2; // half order

    // compute poles
    for i in 0..l {
        let theta = (2 * (i + 1) + n - 1) as f64 * PI / (2 * n) as f64;
        pa.push(Complex32::new((a * theta.cos()) as f32, (-b * theta.sin()) as f32).inv());
        pa.push(Complex32::new((a * theta.cos()) as f32, (b * theta.sin()) as f32).inv());
    }

    // if filter order is odd, there is an additional pole on the
    // real axis
    if r == 1 {
        pa.push(Complex32::new(-a as f32, 0.0).inv());
    }

    // ensure we have written exactly n poles
    assert_eq!(pa.len(), n);

    // compute zeros
    for i in 0..l {
        let theta = 0.5 * PI * (2 * (i + 1) - 1) as f64 / n as f64;
        za.push(-Complex32::new(0.0, theta.cos() as f32).inv());
        za.push(Complex32::new(0.0, theta.cos() as f32).inv());
    }

    // ensure we have written exactly 2*L poles
    assert_eq!(za.len(), 2 * l);

    // compute analog gain (ignored in digital conversion)
    *ka = Complex32::new(1.0, 0.0);
    for &p in &pa[..n] {
        *ka *= p;
    }
    for &z in &za[..2 * l] {
        *ka /= z;
    }

    Ok(())
}
