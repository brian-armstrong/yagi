// Math module
// Current state:
// - Autotests matching and passing
// - ready to use

pub mod bessel;
pub mod gamma;
pub mod windows;
pub mod poly;
pub mod modarith;
pub mod complex;

pub use self::bessel::*;
pub use self::gamma::*;
pub use self::windows::*;
pub use self::poly::*;
pub use self::modarith::*;

use std::f32::consts::{PI, SQRT_2};
use libm::{erff, erfcf};

use crate::error::{Error, Result};

// Q-function
pub fn qf(z: f32) -> f32 {
    0.5 * (1.0 - erff(z / SQRT_2))
}

// Marcum Q-function
pub fn marcumqf(m: i32, alpha: f32, beta: f32) -> f32 {
    // Use approximation [Helstrom:1992] (Eq. 25)
    // Q_M(a,b) ~ erfc(x),
    //   x = (b-a-M)/sigma^2,
    //   sigma = M + 2a

    let sigma = m as f32 + 2.0 * alpha;
    let x = (beta - alpha - m as f32) / (sigma * sigma);
    erfcf(x)
}

// Marcum Q-function (M=1)
pub fn marcumq1f(alpha: f32, beta: f32) -> f32 {
    const NUM_MARCUMQ1_ITERATIONS: usize = 64;

    let t0 = (-0.5 * (alpha * alpha + beta * beta)).exp();
    let mut t1 = 1.0;
    let a_div_b = alpha / beta;
    let a_mul_b = alpha * beta;

    let mut y = 0.0;
    for k in 0..NUM_MARCUMQ1_ITERATIONS {
        // accumulate y
        y += t1 * besselif(k as f32, a_mul_b);

        // update t1
        t1 *= a_div_b;
    }

    t0 * y
}

// compute sinc(x) = sin(pi*x) / (pi*x)
pub fn sincf(x: f32) -> f32 {
    if x.abs() < 0.01 {
        (PI * x / 2.0).cos() * (PI * x / 4.0).cos() * (PI * x / 8.0).cos()
    } else {
        (PI * x).sin() / (PI * x)
    }
}

pub fn sincd(x: f64) -> f64 {
    if x.abs() < 0.01 {
        (std::f64::consts::PI * x / 2.0).cos() * (std::f64::consts::PI * x / 4.0).cos() * (std::f64::consts::PI * x / 8.0).cos()
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

// next power of 2 : y = ceil(log2(_x))
pub fn nextpow2(mut x: u32) -> Result<u32> {
    if x == 0 {
        return Err(Error::Value("nextpow2(), input must be greater than zero".to_owned()));
    }

    x -= 1;
    let mut n = 0;
    while x > 0 {
        x >>= 1;
        n += 1;
    }
    Ok(n)
}

// (n choose k) = n! / ( k! (n-k)! )
pub fn nchoosek(n: u32, k: u32) -> Result<f32> {
    if k > n {
        return Err(Error::Value(("invalid input: k cannot exceed n").to_owned()));
    } else if k == 0 || k == n {
        return Ok(1.0);
    }

    // take advantage of symmetry and take larger value
    let k = if k < n / 2 { n - k } else { k };

    // use lngamma() function when n is large
    if n > 12 {
        let t0 = lngammaf(n as f32 + 1.0);
        let t1 = lngammaf((n - k) as f32 + 1.0);
        let t2 = lngammaf(k as f32 + 1.0);

        return Ok((t0 - t1 - t2).exp().round() as f32);
    }

    // old method
    let mut rnum = 1.0;
    let mut rden = 1.0;
    for i in (k + 1)..=n {
        rnum *= i as f32;
    }
    for i in 1..=(n - k) {
        rden *= i as f32;
    }
    Ok(rnum / rden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_Q)]
    fn test_q() {
        let tol = 1e-6f32;
        assert_relative_eq!(qf(-4.0), 0.999968329, epsilon = tol);
        assert_relative_eq!(qf(-3.0), 0.998650102, epsilon = tol);
        assert_relative_eq!(qf(-2.0), 0.977249868, epsilon = tol);
        assert_relative_eq!(qf(-1.0), 0.841344746, epsilon = tol);
        assert_relative_eq!(qf( 0.0), 0.5,         epsilon = tol);
        assert_relative_eq!(qf( 1.0), 0.158655254, epsilon = tol);
        assert_relative_eq!(qf( 2.0), 0.022750132, epsilon = tol);
        assert_relative_eq!(qf( 3.0), 0.001349898, epsilon = tol);
        assert_relative_eq!(qf( 4.0), 0.000031671, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_sincf)]
    fn test_sincf() {
        let tol = 1e-3f32;
        assert_relative_eq!(sincf(0.0), 1.0, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_nextpow2)]
    fn test_nextpow2() {
        assert_eq!(nextpow2(1).unwrap(),  0);
        assert_eq!(nextpow2(2).unwrap(),  1);
        assert_eq!(nextpow2(3).unwrap(),  2);
        assert_eq!(nextpow2(4).unwrap(),  2);
        assert_eq!(nextpow2(5).unwrap(),  3);
        assert_eq!(nextpow2(6).unwrap(),  3);
        assert_eq!(nextpow2(7).unwrap(),  3);
        assert_eq!(nextpow2(8).unwrap(),  3);
        assert_eq!(nextpow2(9).unwrap(),  4);
        assert_eq!(nextpow2(10).unwrap(), 4);
        assert_eq!(nextpow2(11).unwrap(), 4);
        assert_eq!(nextpow2(12).unwrap(), 4);
        assert_eq!(nextpow2(13).unwrap(), 4);
        assert_eq!(nextpow2(14).unwrap(), 4);
        assert_eq!(nextpow2(15).unwrap(), 4);
        assert_eq!(nextpow2(67).unwrap(), 7);
        assert_eq!(nextpow2(179).unwrap(), 8);
        assert_eq!(nextpow2(888).unwrap(), 10);
    }

    #[test]
    #[autotest_annotate(autotest_nchoosek)]
    fn test_nchoosek() {
        const EPSILON: f32 = 1e-3;
        let test_vectors = [
            (6, 0, 1),
            (6, 1, 6),
            (6, 2, 15),
            (6, 3, 20),
            (6, 4, 15),
            (6, 5, 6),
            (6, 6, 1),
            (7, 0, 1),
            (7, 1, 7),
            (7, 2, 21),
            (7, 3, 35),
            (7, 4, 35),
            (7, 5, 21),
            (7, 6, 7),
            (7, 7, 1),
        ];

        for &(n, k, expected) in &test_vectors {
            assert_relative_eq!(nchoosek(n, k).unwrap(), expected as f32, epsilon = EPSILON);
        }

        // test very large numbers
        assert_relative_eq!(nchoosek(124, 5).unwrap(), 225150024.0, epsilon = 5000.0);
    }

    #[test]
    #[autotest_annotate(autotest_math_config)]
    fn test_math_config() {
        assert!(nextpow2(0).is_err());
        assert!(nchoosek(4, 5).is_err());
        // assert!(std::panic::catch_unwind(|| lngammaf(-1.0)).is_err());
        assert!(gcd(12, 0).is_err());
        assert!(gcd( 0,12).is_err());
        assert!(gcd( 0, 0).is_err());
    }
}