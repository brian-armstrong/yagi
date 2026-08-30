use std::f32;
use crate::math::gamma;

const NUM_BESSELI_ITERATIONS: usize = 64;
const NUM_BESSELJ_ITERATIONS: usize = 128;
const NUM_BESSELJ0_ITERATIONS: usize = 16;

/// log(I_v(z)) : log Modified Bessel function of the first kind
pub fn lnbesselif(nu: f32, z: f32) -> f32 {
    // special case: check for zeros; besseli_nu(0) = (nu = 0 ? 1 : 0)
    if z == 0.0 {
        return if nu == 0.0 { 0.0 } else { f32::NEG_INFINITY };
    }

    // special case: nu = 1/2, besseli(z) = sqrt(2/pi*z)*sinh(z)
    if nu == 0.5 {
        return 0.5 * (2.0 / (f32::consts::PI * z)).ln() + z.sinh().ln();
    }

    // low signal approximation
    if z < 1e-3 * (nu + 1.0).sqrt() {
        return -gamma::lngammaf(nu + 1.0) + nu * (0.5 * z).ln();
    }

    let t0 = nu * (0.5 * z).ln();
    let mut y = 0.0;

    for k in 0..NUM_BESSELI_ITERATIONS {
        // compute log( (z^2/4)^k )
        let t1 = 2.0 * k as f32 * (0.5 * z).ln();

        // compute: log( k! * Gamma(nu + k +1) )
        let t2 = gamma::lngammaf(k as f32 + 1.0);
        let t3 = gamma::lngammaf(nu + k as f32 + 1.0);

        // accumulate y
        y += (t1 - t2 - t3).exp();
    }

    t0 + y.ln()
}

/// I_v(z) : Modified Bessel function of the first kind
pub fn besselif(nu: f32, z: f32) -> f32 {
    // special case: check for zeros; besseli_nu(0) = (nu = 0 ? 1 : 0)
    if z == 0.0 {
        return if nu == 0.0 { 1.0 } else { 0.0 };
    }

    // special case: nu = 1/2, besseli(z) = sqrt(2/pi*z)*sinh(z)
    if nu == 0.5 {
        return (2.0 / (f32::consts::PI * z)).sqrt() * z.sinh();
    }

    // low signal approximation
    if z < 1e-3 * (nu + 1.0).sqrt() {
        return (0.5 * z).powf(nu) / gamma::gammaf(nu + 1.0);
    }

    // derive from logarithmic expansion
    lnbesselif(nu, z).exp()
}

/// I_0(z) : Modified bessel function of the first kind (order zero)
pub fn besseli0f(z: f32) -> f32 {
    besselif(0.0, z)
}

/// J_v(z) : Bessel function of the first kind
pub fn besseljf(nu: f32, z: f32) -> f32 {
    // special case: check for zeros; besselj_nu(0) = (nu = 0 ? 1 : 0)
    if z == 0.0 {
        return if nu == 0.0 { 1.0 } else { 0.0 };
    }

    // low signal approximation
    if z < 1e-3 * (nu + 1.0).sqrt() {
        return (0.5 * z).powf(nu) / gamma::gammaf(nu + 1.0);
    }

    let mut j = 0.0;
    let abs_nu = nu.abs();

    for k in 0..NUM_BESSELJ_ITERATIONS {
        // compute: (2k + |nu|)
        let t0 = 2.0 * k as f32 + abs_nu;

        // compute: (2k + |nu|)*log(z)
        let t1 = t0 * z.ln();

        // compute: (2k + |nu|)*log(2)
        let t2 = t0 * 2.0f32.ln();

        // compute: log(Gamma(k+1))
        let t3 = gamma::lngammaf(k as f32 + 1.0);

        // compute: log(Gamma(|nu|+k+1))
        let t4 = gamma::lngammaf(abs_nu + k as f32 + 1.0);

        // accumulate J
        let term = (t1 - t2 - t3 - t4).exp();
        j += if k % 2 == 0 { term } else { -term };
    }

    j
}

/// J_0(z) : Bessel function of the first kind (order zero)
pub fn besselj0f(z: f32) -> f32 {
    // large signal approximation, see
    // Gross, F. B "New Approximations to J0 and J1 Bessel Functions,"
    //   IEEE Trans. on Antennas and Propagation, vol. 43, no. 8,
    //   August, 1995
    if z.abs() > 10.0 {
        return (2.0 / (f32::consts::PI * z.abs())).sqrt() * (z.abs() - f32::consts::FRAC_PI_4).cos();
    }

    let mut y = 0.0;
    for k in 0..NUM_BESSELJ0_ITERATIONS {
        let t = (z / 2.0).powi(k as i32) / gamma::gammaf(k as f32 + 1.0);
        y += if k % 2 == 0 { t * t } else { -t * t };
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;
    
    #[test]
    #[autotest_annotate(autotest_lnbesselif)]
    fn test_lnbesselif() {
        const EPSILON: f32 = 1e-5;
        assert_abs_diff_eq!(lnbesselif(0.0, 0.0), 0.0, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(0.0, 0.1), 0.00249843923387607, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(0.1, 7.1), 5.21933724549090, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(0.3, 2.1), 0.853008130814754, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(0.9, 9.3), 7.23414120004177, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(1.0, 0.1), -2.99448253386220, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(1.7, 0.01), -9.44195081753909, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(1.8, 1e-3), -14.1983271298778, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(1.9, 8.7), 6.49469148684252, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(4.9, 0.01), -30.5795429642925, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(7.4, 9.3), 4.33486237261960, epsilon = EPSILON);
    
        // test large values of nu
        assert_abs_diff_eq!(lnbesselif(20.0, 3.0), -34.1194307343208, epsilon = EPSILON);
        assert_abs_diff_eq!(lnbesselif(30.0, 3.0), -62.4217845317278, epsilon = EPSILON);
        // commented out (same as liquid)
        // assert_abs_diff_eq!(lnbesselif(35.0, 3.0), -77.8824494916507, epsilon = EPSILON);
        // assert_abs_diff_eq!(lnbesselif(38.0, 3.0), -87.5028737258841, epsilon = EPSILON);
        // assert_abs_diff_eq!(lnbesselif(39.0, 3.0), -90.7624095618186, epsilon = EPSILON);
        // assert_abs_diff_eq!(lnbesselif(40.0, 3.0), -94.0471931331690, epsilon = EPSILON);
        // assert_abs_diff_eq!(lnbesselif(80.0, 3.0), -241.208142562073, epsilon = EPSILON);
        // assert_abs_diff_eq!(lnbesselif(140.0, 3.0), -498.439222461430, epsilon = EPSILON);
        
    }
    
    #[test]
    #[autotest_annotate(autotest_besselif)]
    fn test_besselif() {
        const EPSILON: f32 = 1e-3;
        assert_abs_diff_eq!(besselif(0.0, 0.0), 1.0, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.0, 0.1), 1.00250156293410, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.0, 0.2), 1.01002502779515, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.0, 0.5), 1.06348337074132, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.0, 1.0), 1.26606587775201, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.0, 2.0), 2.27958530233607, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.0, 3.0), 4.88079258586503, epsilon = EPSILON);
    
        assert_abs_diff_eq!(besselif(0.5, 0.0), 0.0, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.5, 0.1), 0.252733984600132, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.5, 0.2), 0.359208417583362, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.5, 0.5), 0.587993086790417, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.5, 1.0), 0.937674888245489, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.5, 2.0), 2.046236863089057, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(0.5, 3.0), 4.614822903407577, epsilon = EPSILON);
    
        assert_abs_diff_eq!(besselif(1.3, 0.0), 0.0, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(1.3, 0.1), 0.017465030873157, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(1.3, 0.2), 0.043144293848607, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(1.3, 0.5), 0.145248507279042, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(1.3, 1.0), 0.387392350983796, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(1.3, 2.0), 1.290819215135879, epsilon = EPSILON);
        assert_abs_diff_eq!(besselif(1.3, 3.0), 3.450680420553085, epsilon = EPSILON);
    }
    
    #[test]
    #[autotest_annotate(autotest_besseli0f)]
    fn test_besseli0f() {
        const EPSILON: f32 = 1e-3;
        assert_abs_diff_eq!(besseli0f(0.0), 1.0, epsilon = EPSILON);
        assert_abs_diff_eq!(besseli0f(0.1), 1.00250156293410, epsilon = EPSILON);
        assert_abs_diff_eq!(besseli0f(0.2), 1.01002502779515, epsilon = EPSILON);
        assert_abs_diff_eq!(besseli0f(0.5), 1.06348337074132, epsilon = EPSILON);
        assert_abs_diff_eq!(besseli0f(1.0), 1.26606587775201, epsilon = EPSILON);
        assert_abs_diff_eq!(besseli0f(2.0), 2.27958530233607, epsilon = EPSILON);
        assert_abs_diff_eq!(besseli0f(3.0), 4.88079258586503, epsilon = EPSILON);
    }
    
    #[test]
    #[autotest_annotate(autotest_besseljf)]
    fn test_besseljf() {
        const EPSILON: f32 = 1e-3;
        assert_abs_diff_eq!(besseljf(0.0, 0.0), 1.000000000000000, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 0.1), 0.997501562066040, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 0.2), 0.990024972239576, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 0.5), 0.938469807240813, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 1.0), 0.765197686557967, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 2.0), 0.223890779141236, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 3.0), -0.260051954901934, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 4.0), -0.397149809863847, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 6.0), 0.150645257250997, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.0, 8.0), 0.171650807137554, epsilon = EPSILON);
    
        assert_abs_diff_eq!(besseljf(0.5, 0.0), 0.000000000000000, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 0.1), 0.251892940326001, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 0.2), 0.354450744211402, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 0.5), 0.540973789934529, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 1.0), 0.671396707141804, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 2.0), 0.513016136561828, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 3.0), 0.065008182877376, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 4.0), -0.301920513291637, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 6.0), -0.091015409523068, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(0.5, 8.0), 0.279092808570990, epsilon = EPSILON);
    
        assert_abs_diff_eq!(besseljf(1.7, 0.0), 0.000000000000000, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 0.1), 0.003971976455203, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 0.2), 0.012869169735073, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 0.5), 0.059920175825578, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 1.0), 0.181417665056645, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 2.0), 0.437811462130677, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 3.0), 0.494432522734784, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 4.0), 0.268439400467270, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 6.0), -0.308175744215833, epsilon = EPSILON);
        assert_abs_diff_eq!(besseljf(1.7, 8.0), -0.001102600927987, epsilon = EPSILON);
    }
    
    #[test]
    #[autotest_annotate(autotest_besselj0f)]
    fn test_besselj0f() {
        const EPSILON: f32 = 1e-3;
        assert_abs_diff_eq!(besselj0f(0.0), 1.0, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(0.1), 0.997501562066040, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(0.2), 0.990024972239576, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(0.5), 0.938469807240813, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(1.0), 0.765197686557967, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(2.0), 0.223890779141236, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(2.5), -0.048383776468199, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(3.0), -0.260051954901934, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(3.5), -0.380127739987263, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(4.0), -0.397149809863848, epsilon = EPSILON);
        assert_abs_diff_eq!(besselj0f(4.5), -0.320542508985121, epsilon = EPSILON);
    }
}