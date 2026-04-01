use std::f32::consts::PI;
use num_complex::Complex;
use rand::Rng;
use libm::erff;

use crate::error::{Error, Result};

/// Gauss
pub fn randnf() -> f32 {
    let mut rng = rand::thread_rng();

    // generate two uniform random numbers
    let u1: f32 = loop {
        let u = rng.gen::<f32>();
        if u != 0.0 {
            break u;
        }
    };
    let u2: f32 = rng.gen();

    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin()
}

pub fn awgn(x: &mut f32, nstd: f32) {
    *x += randnf() * nstd;
}

/// Complex Gauss
pub fn crandnf() -> Complex<f32> {
    let mut rng = rand::thread_rng();

    // generate two uniform random numbers
    let u1: f32 = loop {
        let u = rng.gen::<f32>();
        if u != 0.0 {
            break u;
        }
    };
    let u2: f32 = rng.gen();

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    Complex::from_polar(r, theta)
}

pub fn cawgn(x: &mut Complex<f32>, nstd: f32) {
    *x += crandnf() * nstd * 0.707106781186547;
}

/// Gauss random number probability distribution function
pub fn randnf_pdf(x: f32, eta: f32, sig: f32) -> Result<f32> {
    if sig <= 0.0 {
        return Err(Error::Config("standard deviation must be greater than zero".to_string()));
    }

    let t = x - eta;
    let s2 = sig * sig;
    Ok((-t * t / (2.0 * s2)).exp() / (2.0 * PI * s2).sqrt())
}

/// Gauss random number cumulative distribution function
pub fn randnf_cdf(x: f32, eta: f32, sig: f32) -> Result<f32> {
    if sig <= 0.0 {
        return Err(Error::Config("standard deviation must be greater than zero".to_string()));
    }

    Ok(0.5 + 0.5 * erff((x - eta) / (sig * 2.0_f32.sqrt())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;

    #[test]
    #[autotest_annotate(autotest_randnf)]
    fn test_randnf() {
        const NUM_TRIALS: usize = 100_000;
        const TOL: f32 = 0.1;

        let mut m1: f32 = 0.0;
        let mut m2: f32 = 0.0;

        for _ in 0..NUM_TRIALS {
            let x = randnf();
            m1 += x;
            m2 += x * x;
        }

        m1 /= NUM_TRIALS as f32;
        m2 = (m2 / NUM_TRIALS as f32) - m1 * m1;

        println!("m1 = {:.6} (expected 0.0)", m1);
        println!("m2 = {:.6} (expected 1.0)", m2);

        assert_relative_eq!(m1, 0.0, epsilon = TOL);
        assert_relative_eq!(m2, 1.0, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_crandnf)]
    fn test_crandnf() {
        const NUM_TRIALS: usize = 100_000;
        const TOL: f32 = 0.1;

        let mut m1: Complex32 = Complex32::new(0.0, 0.0);
        let mut m2: f32 = 0.0;

        for _ in 0..NUM_TRIALS {
            let x = crandnf();
            m1 += x;
            m2 += x.norm_sqr();
        }

        let n = 2.0 * NUM_TRIALS as f32;
        m1 /= n;
        m2 = (m2 / n) - m1.norm_sqr();

        println!("m1 = {:.6} + j*{:.6} (expected 0+j*0)", m1.re, m1.im);
        println!("m2 = {:.6} (expected 1.0)", m2);

        assert_relative_eq!(m1.re, 0.0, epsilon = TOL);
        assert_relative_eq!(m1.im, 0.0, epsilon = TOL);
        assert_relative_eq!(m2, 1.0, epsilon = TOL);
    }
}