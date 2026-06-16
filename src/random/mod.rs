// Random module
// Current state:
// - Autotests matching for rng functions
// - Distribution tests turned off for nakm, ricek, uniform
// - Some replacement may be possible from stdlib or other crates
// - Module orginization still WIP

pub mod exp;
pub mod gamma;
pub mod nakm;
pub mod normal;
pub mod ricek;
pub mod scramble;
pub mod uniform;
pub mod weib;

pub use exp::*;
pub use gamma::*;
pub use nakm::*;
pub use normal::*;
pub use ricek::*;
pub use uniform::*;
pub use weib::*;


#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;
    use crate::error::Result;

    #[test]
    #[autotest_annotate(autotest_random_config)]
    fn test_random_config() {
        // Exponential: lambda out of range
        assert!(randexpf(-1.0).is_err());
        assert!(randexpf_pdf(0.0, -1.0).is_err());
        assert!(randexpf_cdf(0.0, -1.0).is_err());
        // Exponential: pdf, cdf with valid input, but negative variable
        assert_eq!(randexpf_pdf(-2.0, 2.3).unwrap(), 0.0);
        assert_eq!(randexpf_cdf(-2.0, 2.3).unwrap(), 0.0);

        // Gamma: parameters out of range (alpha)
        assert!(randgammaf(-1.0, 1.0).is_err());
        assert!(randgammaf_pdf(0.0, -1.0, 1.0).is_err());
        assert!(randgammaf_cdf(0.0, -1.0, 1.0).is_err());
        // Gamma: parameters out of range (beta)
        assert!(randgammaf(1.0, -1.0).is_err());
        assert!(randgammaf_pdf(0.0, 1.0, -1.0).is_err());
        assert!(randgammaf_cdf(0.0, 1.0, -1.0).is_err());
        // Gamma: delta function parameter out of range
        // TODO not a public function, maybe test
        //    assert!(randgammaf_delta(-1.0).is_err());
        // Gamma: pdf, cdf with valid input, but negative variable
        assert_eq!(randgammaf_pdf(-2.0, 1.2, 2.3).unwrap(), 0.0);
        assert_eq!(randgammaf_cdf(-2.0, 1.2, 2.3).unwrap(), 0.0);

        // Nakagami-m: parameters out of range (m)
        assert!(randnakmf(0.2, 1.0).is_err());
        assert!(randnakmf_pdf(0.0, 0.2, 1.0).is_err());
        assert!(randnakmf_cdf(0.0, 0.2, 1.0).is_err());
        // Nakagami-m: parameters out of range (omega)
        assert!(randnakmf(1.0, -1.0).is_err());
        assert!(randnakmf_pdf(0.0, 1.0, -1.0).is_err());
        assert!(randnakmf_cdf(0.0, 1.0, -1.0).is_err());
        // Nakagami-m: pdf, cdf with valid input, but negative variable
        assert_eq!(randnakmf_pdf(-2.0, 1.2, 2.3).unwrap(), 0.0);
        assert_eq!(randnakmf_cdf(-2.0, 1.2, 2.3).unwrap(), 0.0);
    }

    // Helper functions for histogram operations
    fn support_histogram_add(value: f32, bins: &mut [f32], num_bins: usize, vmin: f32, vmax: f32) -> usize {
        if value < vmin || value > vmax {
            return 0;
        }
        let vstep = (vmax - vmin) / num_bins as f32;
        let mut indexf = (value - vmin) / vstep;
        if indexf < 0.0 {
            indexf = 0.0;
        }
        if indexf >= num_bins as f32 {
            indexf = (num_bins - 1) as f32;
        }
        let index = indexf as usize;
        bins[index] += 1.0;
        index
    }
    
    fn support_histogram_normalize(bins: &mut [f32], num_bins: usize, num_trials: usize, vmin: f32, vmax: f32) -> f32 {
        let vstep = (vmax - vmin) / num_bins as f32;
        let area = num_trials as f32 * vstep;
        for bin in bins.iter_mut() {
            *bin /= area;
        }
        area
    }
    
    fn support_histogram_validate(bins: &[f32], pdf: impl Fn(f32) -> Result<f32>, cdf: impl Fn(f32) -> Result<f32>, num_bins: usize, num_trials: usize, vmin: f32, vmax: f32, tol: f32) {
        const NUM_PDF_STEPS: usize = 20;
        let mut bins_normalized = bins.to_vec();
        support_histogram_normalize(&mut bins_normalized, num_bins, num_trials, vmin, vmax);
    
        let vstep = (vmax - vmin) / num_bins as f32;
        for i in 0..num_bins {
            let mut pdf_avg = 0.0;
            for j in 0..NUM_PDF_STEPS {
                let x = vmin + (i as f32 + (j as f32 / NUM_PDF_STEPS as f32)) * vstep;
                pdf_avg += pdf(x).unwrap() / NUM_PDF_STEPS as f32;
            }
            // println!("bin {:?}, range: {:?}, normalized: {:?}, pdf_avg: {:?}", i, (vmin + i as f32 * vstep, vmin + (i + 1) as f32 * vstep), bins_normalized[i], pdf_avg);
            assert_abs_diff_eq!(bins_normalized[i], pdf_avg, epsilon = tol);
        }
    
        let mut accum = cdf(vmin).unwrap();
        for i in 1..num_bins {
            let right = vmin + i as f32 * vstep;
            accum += bins_normalized[i-1] * vstep;
            // println!("accum: {:?}, cdf(right): {:?}", accum, cdf(right).unwrap());
            let cdf_val = cdf(right).unwrap();
            assert_abs_diff_eq!(accum, cdf_val, epsilon = tol);
        }
    }
    
    #[test]
    #[autotest_annotate(autotest_distribution_randnf)]
    fn test_distribution_randnf() {
        let num_trials = 10000000;
        let eta = 0.0;
        let sig = 1.0;
        let tol = 0.1;
    
        let num_bins = 31;
        let mut bins = vec![0.0; num_bins];
        let vmin = -3.0;
        let vmax = 3.0;
    
        // compute histogram
        for _ in 0..num_trials {
            let v = randnf() * sig + eta;
            support_histogram_add(v, &mut bins, num_bins, vmin, vmax);
        }

        let pdf = |v| randnf_pdf(v, eta, sig);
        let cdf = |v| randnf_cdf(v, eta, sig);
    
        // validate distributions
        support_histogram_validate(&bins, pdf, cdf, num_bins, num_trials, vmin, vmax, tol);
    }
    
    #[test]
    fn test_distribution_randexpf() {
        let num_trials = 10000000;
        let lambda = 1.3;
        let tol = 0.1;
    
        let num_bins = 21;
        let mut bins = vec![0.0; num_bins];
        let vmin = -1.0;
        let vmax = 6.0;
    
        // compute histogram
        for _ in 0..num_trials {
            let v = randexpf(lambda).unwrap();
            support_histogram_add(v, &mut bins, num_bins, vmin, vmax);
        }

        let pdf = |v| randexpf_pdf(v, lambda);
        let cdf = |v| randexpf_cdf(v, lambda);
    
        // validate distributions
        support_histogram_validate(&bins, pdf, cdf, num_bins, num_trials, vmin, vmax, tol);
    }

    #[test]
    fn test_distribution_randgammaf() {
        let num_trials = 1000000;
        let alpha = 2.5;
        let beta = 1.0;
        let tol = 0.1;

        let num_bins = 21;
        let mut bins = vec![0.0; num_bins];
        let vmin = -1.0;
        let vmax = 9.0;

        // compute histogram
        for _ in 0..num_trials {
            let v = randgammaf(alpha, beta).unwrap();
            support_histogram_add(v, &mut bins, num_bins, vmin, vmax);
        }

        let pdf = |v| randgammaf_pdf(v, alpha, beta);
        let cdf = |v| randgammaf_cdf(v, alpha, beta);
    
        // validate distributions
        support_histogram_validate(&bins, pdf, cdf, num_bins, num_trials, vmin, vmax, tol);
    }

    // #[test]
    // fn test_distribution_randnakmf() {
    //     let num_trials = 10000000;
    //     let m = 2.0;
    //     let omega = 1.0;
    //     let tol = 0.1;

    //     let num_bins = 28;
    //     let mut bins = vec![0.0; num_bins];
    //     let vmin = -1.0;
    //     let vmax = 2.5;

    //     // compute histogram
    //     for _ in 0..num_trials {
    //         let v = randnakmf(m, omega).unwrap();
    //         support_histogram_add(v, &mut bins, num_bins, vmin, vmax);
    //     }

    //     let pdf = |v| randnakmf_pdf(v, m, omega);
    //     let cdf = |v| randnakmf_cdf(v, m, omega);
    
    //     // validate distributions
    //     support_histogram_validate(&bins, pdf, cdf, num_bins, num_trials, vmin, vmax, tol);
    // }

    // #[test]
    // fn test_distribution_randricekf() {
    //     let num_trials = 5000000;
    //     let k = 2.0;
    //     let sigma = 1.0;
    //     let tol = 0.1;

    //     let num_bins = 24;
    //     let mut bins = vec![0.0; num_bins];
    //     let vmin = -1.0;
    //     let vmax = 5.0;

    //     // compute histogram
    //     for _ in 0..num_trials {
    //         let v = randricekf(k, sigma).unwrap();
    //         support_histogram_add(v, &mut bins, num_bins, vmin, vmax);
    //     }

    //     let pdf = |v| randricekf_pdf(v, k, sigma);
    //     let cdf = |v| randricekf_cdf(v, k, sigma);
    
    //     // validate distributions
    //     support_histogram_validate(&bins, pdf, cdf, num_bins, num_trials, vmin, vmax, tol);
    // }

    // #[test]
    // fn test_distribution_randweibf() {  
    //     let num_trials = 1000000;
    //     let alpha = 1.0;
    //     let beta = 2.0;
    //     let gamma = 6.0;
    //     let tol = 0.1;

    //     let num_bins = 21;
    //     let mut bins = vec![0.0; num_bins];
    //     let vmin = -1.0;
    //     let vmax = 9.0;

    //     // compute histogram
    //     for _ in 0..num_trials {
    //         let v = randweibf(alpha, beta, gamma).unwrap();
    //         support_histogram_add(v, &mut bins, num_bins, vmin, vmax);
    //     }

    //     let pdf = |v| randweibf_pdf(v, alpha, beta, gamma);
    //     let cdf = |v| randweibf_cdf(v, alpha, beta, gamma);
    
    //     // validate distributions
    //     support_histogram_validate(&bins, pdf, cdf, num_bins, num_trials, vmin, vmax, tol);
    // }

    // #[test]
    // fn test_distribution_randuf() {
    //     let num_trials = 248000;
    //     let a = 10.0;
    //     let b = 25.0;
    //     let tol = 0.05;

    //     let num_bins = 59;
    //     let mut bins = vec![0.0; num_bins];
    //     let vmin = 5.0;
    //     let vmax = 35.0;

    //     // compute histogram
    //     for _ in 0..num_trials {
    //         let v = randuf(a, b).unwrap();
    //         support_histogram_add(v, &mut bins, num_bins, vmin, vmax);
    //     }

    //     let pdf = |v| randuf_pdf(v, a, b);
    //     let cdf = |v| randuf_cdf(v, a, b);
    
    //     // validate distributions
    //     support_histogram_validate(&bins, pdf, cdf, num_bins, num_trials, vmin, vmax, tol);
    // }
}