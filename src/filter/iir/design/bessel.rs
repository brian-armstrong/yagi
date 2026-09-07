use crate::error::{Error, Result};
use libm::lgammaf;
use num_complex::Complex32;
use std::f32::consts::LN_2;

//
// Bessel filter design
//
// References:
//  [Bianchi:2007] G. Bianchi and R. Sorrentino, "Electronic Filter Simulation
//      and Design." New York: McGraw-Hill, 2007.
//  [Orchard:1965] H. J. Orchard, "The Roots of the Maximally Flat-Delay
//      Polynomials." IEEE Transactions on Circuit Theory, September, 1965.
//

/// Compute analog zeros, poles, gain of low-pass Bessel filter, grouping complex
/// conjugates together. If the filter order is odd, the single real pole is at
/// the end of the array. There are no zeros for the analog Bessel filter. The
/// gain is unity.
///
/// # Arguments
///
/// * `n` - filter order
/// * `za` - output analog zeros [length: 0]
/// * `pa` - output analog poles [length: _n]
/// * `ka` - output analog gain
pub fn iir_design_bessel_analog(
    n: usize,
    za: &mut Vec<Complex32>,
    pa: &mut Vec<Complex32>,
    ka: &mut Complex32,
) -> Result<()> {
    // roots are computed with order n+1 so we must use a longer array to
    // prevent out-of-bounds write on the provided pa array
    let mut tmp_pa = vec![Complex32::new(0.0, 0.0); n + 1];

    za.clear();
    pa.clear();

    // compute poles (roots to Bessel polynomial)
    fpoly_bessel_roots(n + 1, &mut tmp_pa)?;

    for i in 0..n {
        pa.push(tmp_pa[i]);
    }

    // analog Bessel filter prototype has no zeros

    // The analog Bessel filter's 3-dB cut-off frequency is a
    // non-linear function of its order.  This frequency can
    // be approximated from [Bianchi:2007] (1.67), pp. 33.
    // Re-normalize poles by (approximated) 3-dB frequency.
    let w3db = ((2 * n - 1) as f32 * 2.0f32.ln()).sqrt();
    for p in pa.iter_mut().take(n) {
        *p /= w3db;
    }

    // set gain
    *ka = Complex32::new(1.0, 0.0);
    for p in pa.iter().take(n) {
        *ka *= p;
    }

    Ok(())
}

// TODO verify if we want this
#[allow(dead_code)]
fn fpoly_bessel(n: usize, p: &mut [f32]) -> Result<()> {
    let n = n - 1;
    for k in 0..n + 1 {
        let t0 = lgammaf((2 * n - k + 1) as f32);
        let t1 = lgammaf((n - k + 1) as f32);
        let t2 = lgammaf((k + 1) as f32);
        let t3 = LN_2 * (n - k) as f32;

        p[k] = (t0 - t1 - t2 - t3).exp().round();
    }
    Ok(())
}

fn fpoly_bessel_roots(n: usize, roots: &mut [Complex32]) -> Result<()> {
    fpoly_bessel_roots_orchard(n, roots)
}

fn fpoly_bessel_roots_orchard(n: usize, roots: &mut [Complex32]) -> Result<()> {
    let mut r0 = vec![Complex32::new(0.0, 0.0); n];
    let mut r1 = vec![Complex32::new(0.0, 0.0); n];
    let mut r_hat = vec![Complex32::new(0.0, 0.0); n];

    for i in 1..n {
        let p = i % 2; // order is odd?
        let l = (i + p) / 2;

        if i == 1 {
            r1[0] = Complex32::new(-1.0, 0.0);
            r_hat[0] = Complex32::new(-1.0, 0.0);
        } else if i == 2 {
            r1[0] = Complex32::new(-1.0, 0.0);
            r_hat[0] = Complex32::new(-1.5, 0.5 * 3.0f32.sqrt());
        } else {
            // use previous 2 sets of roots to estimate this set
            if p == 1 {
                // odd order : one real root on negative imaginary axis
                r_hat[0] = Complex32::new(2.0 * r1[0].re - r0[0].re, 0.0);
            } else {
                // even order
                r_hat[0] = 2.0 * r1[0] - r0[0].conj();
            }

            // linear extrapolation of roots of L_{k-2} and L_{k-1} for
            // new root estimate in L_{k}
            for j in 1..l {
                r_hat[j] = 2.0 * r1[j - p] - r0[j - 1];
            }

            for j in 0..l {
                let (x, y) = (r_hat[j].re, r_hat[j].im);
                let (x_hat, y_hat) = fpoly_bessel_roots_orchard_recursion(i, x, y)?;
                r_hat[j] = Complex32::new(x_hat, y_hat);
            }
        }

        // copy roots:  roots(L_{k+1}) -> roots(L_{k+2))
        //              roots(L_{k})   -> roots(L_{k+1))
        r0[..(l - p)].copy_from_slice(&r1[..(l - p)]);
        r1[..l].copy_from_slice(&r_hat[..l]);
    }

    // copy results to output
    let p = n % 2;
    let l = (n - p) / 2;
    for i in 0..l {
        let p = l - i - 1;
        roots[2 * i] = r_hat[p];
        roots[2 * i + 1] = r_hat[p].conj();
    }

    // if order is odd, copy single real root last
    if p == 1 {
        roots[n - 1] = r_hat[0];
    }

    Ok(())
}

fn fpoly_bessel_roots_orchard_recursion(n: usize, x: f32, y: f32) -> Result<(f32, f32)> {
    if n < 2 {
        return Err(Error::Config("n < 2".into()));
    }

    let mut x = x as f64;
    let mut y = y as f64;

    for _ in 0..50 {
        let mut u0 = 1.0;
        let mut u1 = 1.0 + x;
        let mut v0 = 0.0;
        let mut v1 = y;

        let mut u2 = 0.0;
        let mut v2 = 0.0;

        // compute u_r, v_r
        for i in 2..=n {
            u2 = (2 * i - 1) as f64 * u1 + (x * x - y * y) * u0 - 2.0 * x * y * v0;
            v2 = (2 * i - 1) as f64 * v1 + (x * x - y * y) * v0 + 2.0 * x * y * u0;

            // if not on last iteration, update u0, v0, u1, v1
            if i < n {
                u0 = u1;
                v0 = v1;
                u1 = u2;
                v1 = v2;
            }
        }

        // compute derivatives
        let u2p = u2 - x * u1 + y * v1;
        let v2p = v2 - x * v1 - y * u1;

        // update roots
        let g = u2p * u2p + v2p * v2p;
        if g == 0.0 {
            break;
        }

        // For larger order n, the step values dx and dy will be the
        // evaluation of the ratio of two large numbers which can prevent
        // the algorithm from converging for finite machine precision.
        let dx = -(u2p * u2 + v2p * v2) / g;
        let dy = -(u2p * v2 - v2p * u2) / g;
        x += dx;
        y += dy;
    }

    Ok((x as f32, y as f32))
}
