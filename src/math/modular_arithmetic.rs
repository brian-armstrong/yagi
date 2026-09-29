//! Small integer number-theory routines.
//!
//! These routines use trial division and direct repeated multiplication.

use crate::error::{Error, Result};

use num_traits::{One, Zero};

const MAX_FACTORS: usize = 40;

/// Returns whether `n` is prime.
///
/// Zero and one are not prime. Trial division stops at `sqrt(n)`, so any
/// `u32` is checked quickly.
pub fn is_prime(n: u32) -> bool {
    // check base cases (0, 1, 2, 3, divisible by 2, divisible by 3)
    if n <= 1 {
        return false;
    } else if n <= 3 {
        return true;
    } else if n.is_multiple_of(2) {
        return false; // divisible by 2
    } else if n.is_multiple_of(3) {
        return false; // divisible by 3
    }

    let mut r = 5;
    while r <= n / r {
        if n.is_multiple_of(r) || n.is_multiple_of(r + 2) {
            return false;
        }
        r += 6;
    }
    true
}

/// Appends the prime factors of `n` to `factors` and returns how many it
/// appended.
///
/// Factors are appended in ascending order, and a repeated factor appears
/// once for each multiplicity. Nothing is appended when `n` is less than two.
/// Each factor is found by counting up from two, so the time grows with the
/// largest prime factor, and a large prime `n` takes billions of steps.
///
/// This never returns an error.
pub fn factor(n: u32, factors: &mut Vec<u32>) -> Result<usize> {
    let mut num = n;
    let mut num_factors = 0;

    while num > 1 && num_factors < MAX_FACTORS {
        for k in 2..=num {
            if num.is_multiple_of(k) {
                // k factors n; append to list
                factors.push(k);
                num_factors += 1;
                num /= k;
                break;
            }
        }
    }

    if num > 1 && num_factors == MAX_FACTORS {
        return Err(Error::NoConvergence("could not factor number in MAX_FACTORS numbers".to_string()));
    }

    Ok(num_factors)
}

/// Appends the distinct prime factors of `n` to `factors` and returns how
/// many it appended.
///
/// Factors are appended in ascending order. Nothing is appended when `n` is
/// less than two. As with [`factor`], the time grows with the largest prime
/// factor of `n`.
///
/// This never returns an error.
///
/// # Warning
///
/// `factors` must start empty. When it already holds values, repeated
/// factors are compared against the wrong entries and can be appended more
/// than once.
pub fn unique_factor(n: u32, factors: &mut Vec<u32>) -> Result<usize> {
    let mut num = n;
    let mut num_factors = 0;

    while num > 1 && num_factors < MAX_FACTORS {
        for k in 2..=num {
            if num.is_multiple_of(k) {
                // k factors n; append to list
                if num_factors == 0 || factors[num_factors - 1] != k {
                    factors.push(k);
                    num_factors += 1;
                }

                num /= k;

                break;
            }
        }
    }

    if num > 1 && num_factors == MAX_FACTORS {
        return Err(Error::NoConvergence("could not factor number in MAX_FACTORS numbers".to_string()));
    }

    Ok(num_factors)
}

/// Returns the greatest common divisor of two positive integers.
///
/// Common factors are found by trial division up to the smaller input, so the
/// time grows with its value.
///
/// # Errors
///
/// Returns an error when either input is zero.
pub fn gcd<T>(mut p: T, mut q: T) -> Result<T>
where
    T: Copy
        + std::cmp::PartialEq
        + std::cmp::PartialOrd
        + std::ops::DivAssign
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::ops::Rem<Output = T>
        + Zero
        + One,
{
    // check base cases
    if p == T::zero() || q == T::zero() {
        return Err(Error::Value("input cannot be zero".to_string()));
    } else if p == T::one() || q == T::one() {
        return Ok(T::one());
    } else if p == q {
        return Ok(p);
    } else if p < q {
        return gcd(q, p);
    }

    // dumb, slow method
    let zero = T::zero();
    let one = T::one();
    let two = T::one() + T::one();
    let mut gcd = one;
    let mut r = two; // root
    while r <= q {
        while p % r == zero && q % r == zero {
            p /= r;
            q /= r;
            gcd *= r;
        }
        r += if r == two { one } else { two };
    }

    Ok(gcd)
}

/// Returns `base^exp mod n`.
///
/// This multiplies once per unit of `exp`, so the time grows with `exp`. An
/// `exp` of zero returns one, even when `n` is one.
///
/// # Errors
///
/// Returns an error when `n` is zero.
pub fn modpow(base: u32, exp: u32, n: u32) -> Result<u32> {
    if n == 0 {
        return Err(Error::Value("modulus must be greater than zero".to_string()));
    }

    let mut c = 1u64;
    let base = u64::from(base);
    let modulus = u64::from(n);
    for _ in 0..exp {
        c = (c * base) % modulus;
    }
    Ok(c as u32)
}

/// Returns the smallest primitive root of a prime `n`.
///
/// The smallest primitive root of two is taken to be one. Each candidate is
/// tested with [`modpow`] at exponents up to `(n - 1) / 2`, so the time
/// grows with `n`.
///
/// # Errors
///
/// Returns an error when `n` is not prime.
pub fn primitive_root_prime(n: u32) -> Result<u32> {
    if n < 2 || !is_prime(n) {
        return Err(Error::Value("input must be prime".to_string()));
    }
    if n == 2 {
        return Ok(1);
    }

    // find unique factors of n-1
    let mut unique_factors = Vec::with_capacity(MAX_FACTORS);
    let num_unique_factors = unique_factor(n - 1, &mut unique_factors)?;

    // search for minimum integer for which
    //   g^( (n-1)/m ) != 1 (mod n)
    // for all unique roots 'm'
    for g in 2..n {
        let mut is_root = true;
        for &factor in &unique_factors[0..num_unique_factors] {
            let e = (n - 1) / factor;
            if modpow(g, e, n)? == 1 {
                // not a root
                is_root = false;
                break;
            }
        }

        if is_root {
            return Ok(g);
        }
    }

    // This should never happen for prime n
    Err(Error::Value("No primitive root found for prime".to_string()))
}

/// Returns Euler's totient of `x`.
///
/// The totient is the number of positive integers up to `x` that are coprime
/// with `x`. A zero `x` returns zero. As with [`factor`], the time grows with
/// the largest prime factor of `x`.
pub fn totient(x: u32) -> u32 {
    let mut t = x;
    let mut n = x;
    let mut p = 0;
    loop {
        let mut k = 2;
        while k <= n {
            if n.is_multiple_of(k) {
                n /= k;

                if p != k {
                    // factor is unique
                    t /= k;
                    t *= k - 1;
                }
                p = k;
                break;
            }
            k += 1;
        }

        if n < 2 {
            break;
        }
    }

    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_prime_small)]
    fn test_prime_small() {
        for (n, &expected_is_prime) in IS_PRIME_ARRAY.iter().enumerate() {
            assert_eq!(is_prime(n as u32), expected_is_prime != 0);
        }
    }

    #[test]
    fn test_prime_large() {
        assert!(is_prime(4_294_967_291));
    }

    #[test]
    #[autotest_annotate(autotest_factors)]
    fn test_factors() {
        const FACTORS_280: [u32; 5] = [2, 2, 2, 5, 7];
        const FACTORS_280_UNIQUE: [u32; 3] = [2, 5, 7];

        let mut factors = Vec::with_capacity(MAX_FACTORS);
        let num_factors = factor(280, &mut factors).unwrap();
        assert_eq!(num_factors, 5);
        assert_eq!(factors, FACTORS_280);

        let mut factors = Vec::with_capacity(MAX_FACTORS);
        let num_unique_factors = unique_factor(280, &mut factors).unwrap();
        assert_eq!(num_unique_factors, 3);
        assert_eq!(factors, FACTORS_280_UNIQUE);
    }

    #[test]
    #[autotest_annotate(autotest_totient)]
    fn test_totient() {
        assert_eq!(totient(9), 6);
        assert_eq!(totient(20), 8);
        assert_eq!(totient(100), 40);
        assert_eq!(totient(1200), 320);
        assert_eq!(totient(1201), 1200);
    }

    fn testbench_gcd(gcd_expected: u32, p: u32, q: u32) {
        let p = p * gcd_expected;
        let q = q * gcd_expected;
        let gcd_test = gcd(p, q).unwrap();

        assert_eq!(gcd_test, gcd_expected, "gcd({}, {}) = {} (expected {})", p, q, gcd_test, gcd_expected);
    }

    #[test]
    #[autotest_annotate(autotest_gcd_one)]
    fn test_gcd_one() {
        testbench_gcd(1, 2, 3);
        testbench_gcd(1, 2, 5);
        testbench_gcd(1, 2, 7);
        testbench_gcd(1, 7, 2);
        testbench_gcd(1, 17, 19);
        testbench_gcd(1, 23, 31);
        testbench_gcd(1, 2 * 2 * 2 * 2 * 2, 3 * 5 * 7 * 19);
    }

    #[test]
    #[autotest_annotate(autotest_gcd_edge_cases)]
    fn test_gcd_edge_cases() {
        testbench_gcd(1, 1, 1);
        testbench_gcd(1, 1, 2);
        testbench_gcd(1, 1, 2400);
        testbench_gcd(12345, 1, 1); // P==Q
        testbench_gcd((1 << 17) - 1, 1, 1); // P==Q
    }

    #[test]
    #[autotest_annotate(autotest_gcd_base)]
    fn test_gcd_base() {
        testbench_gcd(2 * 2 * 3 * 5 * 7, 2 * 3, 17);
        testbench_gcd(2 * 2 * 3 * 5 * 7, 2 * 3, 17 * 17);
        testbench_gcd(2 * 2 * 3 * 5 * 7, 2 * 3, 17 * 17 * 17);
        testbench_gcd(11, 3, 1);
        testbench_gcd(3, 127, 131);
        testbench_gcd(131, 127, 3);
        testbench_gcd(127, 131, 3);
    }

    #[test]
    fn test_gcd_traits() {
        let p: u32 = 15;
        let q: u32 = 3;
        assert_eq!(gcd(p, q).unwrap(), 3);

        let p: i32 = 28;
        let q: i32 = 14;
        assert_eq!(gcd(p, q).unwrap(), 14);

        let p: usize = 105;
        let q: usize = 140;
        assert_eq!(gcd(p, q).unwrap(), 35);
    }

    #[test]
    fn test_modpow() {
        assert_eq!(modpow(2, 3, 5), Ok(3));
        assert_eq!(modpow(3, 4, 7), Ok(4));
        assert!(modpow(2, 3, 0).is_err());

        let base = u32::MAX / 2;
        let modulus = u32::MAX;
        let expected = (u64::from(base) * u64::from(base) % u64::from(modulus)) as u32;
        assert_eq!(modpow(base, 2, modulus), Ok(expected));
    }

    #[test]
    fn test_primitive_root_prime() {
        assert_eq!(primitive_root_prime(2), Ok(1));
        assert_eq!(primitive_root_prime(5).unwrap(), 2);
        assert_eq!(primitive_root_prime(7).unwrap(), 3);
        assert!(primitive_root_prime(0).is_err());
        assert!(primitive_root_prime(1).is_err());
        assert!(primitive_root_prime(4).is_err());
    }

    const IS_PRIME_ARRAY: [u8; 2500] = [
        0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
}
