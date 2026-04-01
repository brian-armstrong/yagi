use crate::error::{Error, Result};
use crate::math::gamma::{gammaf, lowergammaf};
use crate::random::randf;

pub fn randgammaf(alpha: f32, beta: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    let n = alpha.floor() as usize;
    let delta = alpha - n as f32;

    // generate x' ~ Gamma(n,1)
    let mut x_n = 0.0;
    for _ in 0..n {
        let u = randf();
        x_n -= u.ln();
    }

    // generate x'' ~ Gamma(delta,1) using rejection method
    let x_delta = randgammaf_delta(delta)?;

    Ok(beta * (x_delta + x_n))
}

/// Gamma distribution probability distribution function
///          x^(a-1) exp{-x/b)
///  f(x) = -------------------
///            Gamma(a) b^a
///  where
///      a = alpha, a > 0
///      b = beta,  b > 0
///      Gamma(z) = regular gamma function
///      x >= 0
pub fn randgammaf_pdf(x: f32, alpha: f32, beta: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    if x <= 0.0 {
        return Ok(0.0);
    }

    let t0 = x.powf(alpha - 1.0);
    let t1 = (-x / beta).exp();
    let t2 = gammaf(alpha);
    let t3 = beta.powf(alpha);

    Ok((t0 * t1) / (t2 * t3))
}

// Gamma distribution cumulative distribution function
//  F(x) = gamma(a,x/b) / Gamma(a)
//  where
//      a = alpha,  alpha > 0
//      b = beta,   beta > 0
//      gamma(a,z) = lower incomplete gamma function
//      Gamma(z)   = regular gamma function
//
pub fn randgammaf_cdf(x: f32, alpha: f32, beta: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    if x <= 0.0 {
        return Ok(0.0);
    }

    Ok(lowergammaf(alpha, x / beta) / gammaf(alpha))
}

// generate x ~ Gamma(delta,1)
fn randgammaf_delta(delta: f32) -> Result<f32> {
    // validate input
    if delta < 0.0 || delta >= 1.0 {
        return Err(Error::Config("delta must be in [0,1)".to_string()));
    }

    // initialization
    let delta_inv = 1.0 / delta;
    let e = std::f32::consts::E;
    let v0_ = e / (e + delta);

    loop {
        // step 2
        let v0 = randf();
        let v1 = randf();
        let v2 = randf();

        let (xi, eta) = if v2 <= v0_ {
            // step 4
            let xi = v1.powf(delta_inv);
            let eta = v0 * xi.powf(delta - 1.0);
            (xi, eta)
        } else {
            // step 5
            let xi = 1.0 - v1.ln();
            let eta = v0 * (-xi).exp();
            (xi, eta)
        };

        // step 6
        if eta <= xi.powf(delta - 1.0) * (-xi).exp() {
            return Ok(xi);
        }
    }
}