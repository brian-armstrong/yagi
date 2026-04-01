use crate::error::{Error, Result};

const LEVINSON_MAXORDER: usize = 256;

/// Compute the linear prediction coefficients for an input signal _x
///
/// # Arguments
/// * `x`: input signal [size: n x 1]
/// * `p`: prediction filter order
///
/// # Returns
/// * `a`: prediction filter [size: p+1 x 1]
/// * `g`: prediction error variance [size: p+1 x 1]
pub fn design_lpc(x: &[f32], p: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = x.len();

    // validate input
    if p > n {
        return Err(Error::Config("prediction filter length cannot exceed input signal length".into()));
    }

    // compute auto-correlation with lags
    let mut r = vec![0.0; p + 1];

    for i in 0..=p {
        let lag = i;
        r[i] = x[lag..].iter().zip(&x[..n - lag]).map(|(&a, &b)| a * b).sum();
    }

    // solve the Toeplitz inversion using Levinson-Durbin recursion
    levinson(&r, p)
}

/// Solve the Yule-Walker equations using Levinson-Durbin recursion
/// for _symmetric_ autocorrelation
///
/// # Arguments
/// * `r`: autocorrelation array [size: p+1 x 1]
/// * `p`: filter order
///
/// # Returns
/// * `a`: output coefficients [size: p+1 x 1]
/// * `e`: error variance [size: p+1 x 1]
///
/// # Notes
/// By definition a`[0]` = 1.0
pub fn levinson(r: &[f32], p: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    // check allocation length
    if p > LEVINSON_MAXORDER {
        return Err(Error::Config(format!("filter order ({}) exceeds maximum ({})", p, LEVINSON_MAXORDER)));
    }

    // allocate arrays
    let mut a0 = vec![0.0; p + 1]; // temporary coefficients array, index [n]
    let mut a1 = vec![0.0; p + 1]; // temporary coefficients array, index [n-1]
    let mut e = vec![0.0; p + 1];  // prediction error
    let mut k = vec![0.0; p + 1];  // reflection coefficients

    // initialize
    k[0] = 1.0;
    e[0] = r[0];

    a0[0] = 1.0;
    a1[0] = 1.0;

    for n in 1..=p {
        let mut q = 0.0;
        for i in 0..n {
            q += a0[i] * r[n - i];
        }

        k[n] = -q / e[n - 1];
        e[n] = e[n - 1] * (1.0 - k[n] * k[n]);

        // compute new coefficients
        for i in 0..n {
            a1[i] = a0[i] + k[n] * a0[n - i];
        }

        a1[n] = k[n];

        // copy temporary vector a1 to a0
        a0.copy_from_slice(&a1);
    }

    Ok((a1, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::msequence::MSequence;
    use crate::filter::iir::IirFilter;
    use test_macro::autotest_annotate;
    
    fn lpc_test_harness(n: usize, p: usize, fc: f32, tol: f32) -> Result<()> {
        // create filter
        let mut lowpass = IirFilter::<f32, f32>::new_lowpass(7, fc)?;
    
        // allocate memory for arrays
        let mut y = vec![0.0; n];    // input sequence (filtered noise)
    
        // generate input signal (filtered noise)
        let mut ms = MSequence::create_default(15)?;
        for i in 0..n {
            // rough, but simple uniform random variable
            let v = ms.generate_symbol(10) as f32 / 1023.0 - 0.5;

            // filter result
            y[i] = lowpass.execute(v);
        }
    
        // compute lpc coefficients
        let (a_hat, _g_hat) = design_lpc(&y, p)?;

        // create linear prediction filter
        let mut a_lpc = vec![0.0; p+1];
        let mut b_lpc = a_hat.iter().map(|&a| -a).collect::<Vec<f32>>();
        a_lpc[0] = 1.0;
        b_lpc[0] = 0.0;
        let mut lpc = IirFilter::<f32, f32>::new(&b_lpc, &a_lpc)?;

        // compute prediction error over random sequence
        let mut rmse = 0.0;
        let n_error = 5000;
        for _ in 0..n_error {
            // generate input
            let v = ms.generate_symbol(10) as f32 / 1023.0 - 0.5;

            // run filters
            let s0 = lowpass.execute(v);
            let s1 = lpc.execute(s0);

            // compute error
            rmse += (s0 - s1) * (s0 - s1);
        }
        rmse = 10.0 * (rmse / n_error as f32).log10();
    
        println!("lpc test: n={}, p={}, rmse={:.2e} (tol={:.2e})", n, p, rmse, tol);
    
        // Check RMSE
        assert!(rmse < tol, "RMSE ({:.2e}) exceeds threshold ({:.2e})", rmse, tol);
    
        Ok(())
    }
    
    #[test]
    #[autotest_annotate(autotest_lpc_p4)]
    fn test_lpc_p4() -> Result<()> {
        lpc_test_harness(200, 4, 0.020, -40.0)
    }
    
    #[test]
    #[autotest_annotate(autotest_lpc_p6)]
    fn test_lpc_p6() -> Result<()> {
        lpc_test_harness(400, 6, 0.028, -40.0)
    }
    
    #[test]
    #[autotest_annotate(autotest_lpc_p8)]
    fn test_lpc_p8() -> Result<()> {
        lpc_test_harness(600, 8, 0.035, -40.0)
    }
    
    #[test]
    #[autotest_annotate(autotest_lpc_p10)]
    fn test_lpc_p10() -> Result<()> {
        lpc_test_harness(800, 10, 0.050, -40.0)
    }
    
    #[test]
    #[autotest_annotate(autotest_lpc_p16)]
    fn test_lpc_p16() -> Result<()> {
        lpc_test_harness(1600, 16, 0.055, -40.0)
    }
    
    #[test]
    #[autotest_annotate(autotest_lpc_p32)]
    fn test_lpc_p32() -> Result<()> {
        lpc_test_harness(3200, 24, 0.065, -40.0)
    }
}