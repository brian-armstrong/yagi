use std::f32::consts::PI;

const LOWERGAMMA_MIN_ITERATIONS: usize = 50;
const LOWERGAMMA_MAX_ITERATIONS: usize = 1000;

/// log(Gamma(z))
pub fn lngammaf(z: f32) -> f32 {
    if z <= 0.0 {
        panic!("lngammaf(), undefined for z <= 0");
    } else if z < 10.0 {
        // Use recursive formula:
        // gamma(z+1) = z * gamma(z)
        // therefore:
        // log(Gamma(z)) = log(gamma(z+1)) - ln(z)
        lngammaf(z + 1.0) - z.ln()
    } else {
        // high value approximation
        let mut g = 0.5 * ((2.0 * PI).ln() - z.ln());
        g += z * ((z + (1.0 / (12.0 * z - 0.1 / z))).ln() - 1.0);
        g
    }
}

/// Gamma(z)
pub fn gammaf(z: f32) -> f32 {
    if z < 0.0 {
        // use identities
        // (1) gamma(z)*gamma(-z) = -pi / (z*sin(pi*z))
        // (2) z*gamma(z) = gamma(1+z)
        //
        // therefore:
        // gamma(z) = pi / ( gamma(1-z) * sin(pi*z) )
        let t0 = gammaf(1.0 - z);
        let t1 = (PI * z).sin();
        if t0 == 0.0 || t1 == 0.0 {
            panic!("gammaf(), divide by zero");
        }
        PI / (t0 * t1)
    } else {
        lngammaf(z).exp()
    }
}

/// ln(gamma(z,alpha)) : lower incomplete gamma function
pub fn lnlowergammaf(z: f32, alpha: f32) -> f32 {
    let t0 = z * alpha.ln();
    let t1 = lngammaf(z);
    let t2 = -alpha;
    let mut t3 = 0.0;

    let log_alpha = alpha.ln();
    let mut tmax = 0.0;
    let mut t = 0.0;
    let mut tprime;

    for k in 0..LOWERGAMMA_MAX_ITERATIONS {
        // retain previous value for t
        tprime = t;

        // compute log( alpha^k / Gamma(z + k + 1) )
        //         = k*log(alpha) - lnGamma(z + k + 1)
        t = k as f32 * log_alpha - lngammaf(z + k as f32 + 1.0);

        // accumulate e^t
        t3 += t.exp();

        // check premature stopping criteria
        if k == 0 || t > tmax {
            tmax = t;
        }

        // conditions:
        //  1. minimum number of iterations met
        //  2. surpassed inflection point: k*log(alpha) - log(Gamma(z+k+1))
        //     has an inverted parabolic shape
        //  3. sufficiently beyond peak
        if k > LOWERGAMMA_MIN_ITERATIONS && tprime > t && (tmax - t) > 20.0 {
            break;
        }
    }

    t0 + t1 + t2 + t3.ln()
}

/// ln(Gamma(z,alpha)) : upper incomplete gamma function
pub fn lnuppergammaf(z: f32, alpha: f32) -> f32 {
    (gammaf(z) - lowergammaf(z, alpha)).ln()
}

/// gamma(z,alpha) : lower incomplete gamma function
pub fn lowergammaf(z: f32, alpha: f32) -> f32 {
    lnlowergammaf(z, alpha).exp()
}

/// Gamma(z,alpha) : upper incomplete gamma function
pub fn uppergammaf(z: f32, alpha: f32) -> f32 {
    lnuppergammaf(z, alpha).exp()
}

/// compute n!
pub fn factorialf(n: u32) -> f32 {
    gammaf((n + 1) as f32).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_gamma)]
    fn test_gamma() {
        const EPSILON: f32 = 1e-5;
        let test_vectors: [(f32, f32); 12] = [
            (0.0001, 9999.42288323161),
            (0.001, 999.423772484595),
            (0.01, 99.4325851191505),
            (0.1, 9.51350769866873),
            (0.2, 4.59084371199880),
            (0.5, 1.77245385090552),
            (1.5, 0.886226925452758),
            (2.5, 1.329340388179140),
            (3.2, 2.42396547993537),
            (4.1, 6.81262286301667),
            (5.3, 38.0779764499523),
            (12.0, 39916800.0000000),
        ];

        for &(z, expected_g) in &test_vectors {
            let gamma = gammaf(z);
            let error = (gamma - expected_g).abs() / expected_g.abs();
            assert!(error < EPSILON);
        }
    }

    #[test]
    #[autotest_annotate(autotest_lngamma)]
    fn test_lngamma() {
        const EPSILON: f32 = 1e-4;
        let test_vectors = [
            (1.00000000000000e-05, 1.15129196928958e+01),
            (1.47910838816821e-05, 1.11214774616959e+01),
            (2.18776162394955e-05, 1.07300339056431e+01),
            (3.23593656929628e-05, 1.03385883900717e+01),
            (4.78630092322638e-05, 9.94713997633970e+00),
            (7.07945784384137e-05, 9.55568727630758e+00),
            (1.04712854805090e-04, 9.16422823723390e+00),
            (1.54881661891248e-04, 8.77275982391398e+00),
            (2.29086765276777e-04, 8.38127754918765e+00),
            (3.38844156139203e-04, 7.98977478095072e+00),
            (5.01187233627273e-04, 7.59824172030200e+00),
            (7.41310241300918e-04, 7.20666389700363e+00),
            (1.09647819614319e-03, 6.81501995916639e+00),
            (1.62181009735893e-03, 6.42327843686104e+00),
            (2.39883291901949e-03, 6.03139302698778e+00),
            (3.54813389233576e-03, 5.63929577576287e+00),
            (5.24807460249773e-03, 5.24688733606614e+00),
            (7.76247116628692e-03, 4.85402329836329e+00),
            (1.14815362149688e-02, 4.46049557831785e+00),
            (1.69824365246175e-02, 4.06600834803635e+00),
            (2.51188643150958e-02, 3.67014984368726e+00),
            (3.71535229097173e-02, 3.27236635981559e+00),
            (5.49540873857625e-02, 2.87195653880158e+00),
            (8.12830516164099e-02, 2.46812982675138e+00),
            (1.20226443461741e-01, 2.06022544058646e+00),
            (1.77827941003892e-01, 1.64828757901128e+00),
            (2.63026799189538e-01, 1.23436563201614e+00),
            (3.89045144994281e-01, 8.25181176502332e-01),
            (5.75439937337158e-01, 4.37193579132034e-01),
            (8.51138038202378e-01, 1.05623142071343e-01),
            (1.25892541179417e+00, -1.00254418080515e-01),
            (1.86208713666287e+00, -5.19895823734552e-02),
            (2.75422870333817e+00, 4.78681466346387e-01),
            (4.07380277804113e+00, 1.88523210546678e+00),
            (6.02559586074358e+00, 4.83122059829788e+00),
            (8.91250938133746e+00, 1.04177681572532e+01),
            (1.31825673855641e+01, 2.04497048921129e+01),
            (1.94984459975805e+01, 3.78565107279246e+01),
            (2.88403150312661e+01, 6.73552537656878e+01),
            (4.26579518801593e+01, 1.16490742768456e+02),
            (6.30957344480194e+01, 1.97262133863497e+02),
            (9.33254300796992e+01, 3.28659150940827e+02),
            (1.38038426460289e+02, 5.40606126998515e+02),
        ];

        for &(input, expected) in &test_vectors {
            assert_abs_diff_eq!(lngammaf(input), expected, epsilon = EPSILON);
        }

        // test very large numbers
        assert_abs_diff_eq!(lngammaf(140.0), 550.278651724286, epsilon = EPSILON);
        assert_abs_diff_eq!(lngammaf(150.0), 600.009470555327, epsilon = EPSILON);
        assert_abs_diff_eq!(lngammaf(160.0), 650.409682895655, epsilon = EPSILON);
        assert_abs_diff_eq!(lngammaf(170.0), 701.437263808737, epsilon = EPSILON);
    }

    #[test]
    #[autotest_annotate(autotest_uppergamma)]
    fn test_uppergamma() {
        const EPSILON: f32 = 1e-3;
        let test_vectors = [
            (2.1, 0.001, 1.04649),
            (2.1, 0.01, 1.04646),
            (2.1, 0.1, 1.04295),
            (2.1, 0.2, 1.03231),
            (2.1, 0.3, 1.01540),
            (2.1, 0.4, 0.993237),
            (2.1, 0.5, 0.966782),
            (2.1, 0.6, 0.936925),
            (2.1, 0.7, 0.904451),
            (2.1, 0.8, 0.870053),
            (2.1, 0.9, 0.834330),
            (2.1, 1.0, 0.797796),
            (2.1, 2.0, 0.455589),
            (2.1, 3.0, 0.229469),
            (2.1, 4.0, 0.107786),
            (2.1, 5.0, 0.0484292),
            (2.1, 6.0, 0.0211006),
            (2.1, 7.0, 0.00898852),
            (2.1, 8.0, 0.00376348),
            (2.1, 9.0, 0.00155445),
            (2.1, 10.0, 0.000635002),
        ];

        for &(a, x, expected) in &test_vectors {
            assert_abs_diff_eq!(uppergammaf(a, x), expected, epsilon = EPSILON);
        }
    }

    #[test]
    #[autotest_annotate(autotest_factorial)]
    fn test_factorial() {
        const EPSILON: f32 = 1e-3;
        let test_vectors = [
            (0, 1.0),
            (1, 1.0),
            (2, 2.0),
            (3, 6.0),
            (4, 24.0),
            (5, 120.0),
            (6, 720.0),
        ];

        for &(n, expected) in &test_vectors {
            assert_abs_diff_eq!(factorialf(n), expected, epsilon = EPSILON);
        }
    }

}