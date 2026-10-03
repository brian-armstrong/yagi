//! Chromosome object for genetic algorithm search.

use crate::error::{Error, Result};
use rand::Rng;

const CHROMOSOME_MAX_SIZE: u32 = 32;

/// Fixed list of integer traits, each stored in its own number of bits, for
/// use with [`GeneticSearch`](crate::optim::GeneticSearch).
///
/// A trait of `b` bits normally holds an integer in `[0, 2^b)`. The traits
/// can also be read as fractions, and the bits of all traits taken together,
/// first trait first, are what mutation and crossover act on. New traits
/// start at zero.
///
/// Trait widths must be in `1..=32` bits. Fractions use the full `[0, 1]`
/// range: zero maps to zero and one maps to the largest value for the width.
#[doc(alias = "chromosome")]
#[derive(Clone)]
pub struct Chromosome {
    num_traits: usize,
    bits_per_trait: Vec<u32>,
    max_value: Vec<u64>,
    traits: Vec<u64>,
    num_bits: u32,
}

impl Chromosome {
    /// Creates a chromosome with one trait per entry of `bits_per_trait`.
    ///
    /// # Errors
    ///
    /// Returns an error when `bits_per_trait` is empty, a width is outside
    /// `1..=32`, or the total bit count exceeds `u32::MAX`.
    pub fn new(bits_per_trait: &[u32]) -> Result<Self> {
        // validate input
        let num_traits = bits_per_trait.len();
        if num_traits == 0 {
            return Err(Error::Config("must have at least one trait".into()));
        }
        for &bits in bits_per_trait {
            if bits == 0 || bits > CHROMOSOME_MAX_SIZE {
                return Err(Error::Config(format!("bits/trait must be in 1..={}", CHROMOSOME_MAX_SIZE)));
            }
        }

        // copy/initialize values
        let mut max_value = Vec::with_capacity(num_traits);
        let mut num_bits = 0u32;
        for &bits in bits_per_trait {
            max_value.push(1u64 << bits);
            num_bits =
                num_bits.checked_add(bits).ok_or_else(|| Error::Config("total bit count exceeds u32::MAX".into()))?;
        }

        Ok(Self {
            num_traits,
            bits_per_trait: bits_per_trait.to_vec(),
            max_value,
            traits: vec![0; num_traits],
            num_bits,
        })
    }

    /// Creates a chromosome of `num_traits` traits that all have the same
    /// width.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_traits` is zero, `bits_per_trait` is
    /// outside `1..=32`, or the total bit count exceeds `u32::MAX`.
    pub fn new_uniform(num_traits: usize, bits_per_trait: u32) -> Result<Self> {
        // validate input
        if num_traits == 0 {
            return Err(Error::Config("must have at least one trait".into()));
        }
        if bits_per_trait == 0 || bits_per_trait > CHROMOSOME_MAX_SIZE {
            return Err(Error::Config("bits per trait must be in 1..=32".into()));
        }

        if num_traits > (u32::MAX / bits_per_trait) as usize {
            return Err(Error::Config("total bit count exceeds u32::MAX".into()));
        }
        Self::new(&vec![bits_per_trait; num_traits])
    }

    /// Copies the trait values of `parent` into this chromosome.
    ///
    /// # Errors
    ///
    /// Returns an error when the trait widths differ. This chromosome is
    /// unchanged on error.
    pub fn copy_from(&mut self, parent: &Chromosome) -> Result<()> {
        if self.bits_per_trait != parent.bits_per_trait {
            return Err(Error::Config("chromosome trait widths must match".into()));
        }
        self.traits.copy_from_slice(&parent.traits);
        Ok(())
    }

    /// Returns the number of traits.
    pub fn num_traits(&self) -> usize {
        self.num_traits
    }

    /// Returns the total width of all traits, in bits.
    pub fn num_bits(&self) -> u32 {
        self.num_bits
    }

    /// Sets every trait to zero.
    pub fn reset(&mut self) {
        for t in self.traits.iter_mut() {
            *t = 0;
        }
    }

    /// Sets the traits from integers, one per trait.
    ///
    /// Each value must be below `2^b` for its trait's width `b`. The slice
    /// must contain exactly one value per trait.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice length differs from the trait count or
    /// any value is too large. The chromosome is unchanged on error.
    pub fn set_values(&mut self, values: &[u64]) -> Result<()> {
        if values.len() != self.num_traits {
            return Err(Error::Config("expected one value per chromosome trait".into()));
        }
        for (index, &value) in values.iter().enumerate() {
            if value >= self.max_value[index] {
                return Err(Error::Range("chromosome trait value exceeds maximum".into()));
            }
        }
        self.traits.copy_from_slice(values);
        Ok(())
    }

    /// Sets the traits from fractions in `[0, 1]`, one per trait.
    ///
    /// Each value is rounded to the nearest integer in `0..=2^b - 1` for its
    /// trait's width `b`. The slice must contain exactly one value per trait.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice length differs from the trait count or
    /// any fraction is outside `[0, 1]` (including NaN). The chromosome is
    /// unchanged on error.
    pub fn set_fractions(&mut self, fractions: &[f32]) -> Result<()> {
        if fractions.len() != self.num_traits {
            return Err(Error::Config("expected one fraction per chromosome trait".into()));
        }
        for &fraction in fractions {
            if !(0.0..=1.0).contains(&fraction) {
                return Err(Error::Range("chromosome fraction must be in [0, 1]".into()));
            }
        }
        for (index, &fraction) in fractions.iter().enumerate() {
            // quantize in f64 so a 32-bit trait's upper endpoint remains exact.
            let max = self.max_value[index] - 1;
            self.traits[index] = ((fraction as f64) * (max as f64)).round() as u64;
        }
        Ok(())
    }

    /// Flips one bit of the chromosome.
    ///
    /// Bits are numbered across all traits together, starting from the most
    /// significant bit of the first trait.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is not below [`Self::num_bits`].
    pub fn mutate(&mut self, index: u32) -> Result<()> {
        if index >= self.num_bits {
            return Err(Error::Range("chromosome_mutate(), maximum index exceeded".into()));
        }

        // search for the trait containing the indexed bit
        let mut t = 0u32;
        for i in 0..self.num_traits {
            let b = self.bits_per_trait[i];
            if t == index {
                self.traits[i] ^= 1u64 << (b - 1);
                return Ok(());
            } else if t > index {
                self.traits[i - 1] ^= 1u64 << (t - index - 1);
                return Ok(());
            } else {
                t += b;
            }
        }

        self.traits[self.num_traits - 1] ^= 1u64 << (t - index - 1);
        Ok(())
    }

    /// Overwrites this chromosome with bits from two parents.
    ///
    /// Bits before position `threshold`, counted as in [`Self::mutate`], come
    /// from `first_parent` and the rest from `second_parent`. Both parents
    /// must have the same trait widths as this chromosome.
    ///
    /// # Errors
    ///
    /// Returns an error when `threshold` is above [`Self::num_bits`] or either
    /// parent's trait widths differ. The chromosome is unchanged on error.
    pub fn crossover(&mut self, first_parent: &Chromosome, second_parent: &Chromosome, threshold: u32) -> Result<()> {
        if threshold > self.num_bits {
            return Err(Error::Range("chromosome_crossover(), maximum index exceeded".into()));
        }
        if self.bits_per_trait != first_parent.bits_per_trait || self.bits_per_trait != second_parent.bits_per_trait {
            return Err(Error::Config("chromosome trait widths must match".into()));
        }

        // find crossover point
        let mut t = 0u32;
        let mut i = 0;
        while i < self.num_traits {
            if t >= threshold {
                break;
            } else {
                t += self.bits_per_trait[i];
            }

            // child gets first parent's traits up until threshold is reached
            self.traits[i] = first_parent.traits[i];
            i += 1;
        }

        // determine if trait is split
        let rem = t - threshold;
        if rem > 0 {
            // split trait on remainder
            let b = self.bits_per_trait[i - 1];
            let mask1 = ((1u64 << (b - rem)) - 1) << rem;
            let mask2 = (1u64 << rem) - 1;
            self.traits[i - 1] = (first_parent.traits[i - 1] & mask1) | (second_parent.traits[i - 1] & mask2);
        }

        // finish crossover
        while i < self.num_traits {
            // child gets second parent's traits beyond threshold
            self.traits[i] = second_parent.traits[i];
            i += 1;
        }
        Ok(())
    }

    /// Sets every trait to a uniformly random value that fits its width,
    /// drawn from the thread-local random number generator.
    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.num_traits {
            // random bits masked to the trait's width (max_value[i] is 2^bits)
            self.traits[i] = rng.gen::<u64>() & (self.max_value[i] - 1);
        }
    }

    /// Returns trait `index` as a fraction, its value divided by `2^b - 1` for
    /// width `b`, or `None` when `index` is out of range.
    pub fn fraction(&self, index: usize) -> Option<f32> {
        self.traits.get(index).map(|&value| value as f32 / (self.max_value[index] - 1) as f32)
    }

    /// Returns the integer value of trait `index`, or `None` when `index` is
    /// out of range.
    pub fn value(&self, index: usize) -> Option<u64> {
        self.traits.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_chromosome_config)]
    fn chromosome_config() {
        // invalid configurations
        let bits_per_trait_invalid = [6, 6, 6, 6, 6, 6, 6, 1000];
        let bits_per_trait_valid = [6, 6, 6, 6, 6, 6, 6, 32];
        assert!(Chromosome::new(&bits_per_trait_invalid).is_err());
        assert!(Chromosome::new(&[]).is_err()); // zero traits
        assert!(Chromosome::new_uniform(0, 12).is_err()); // too few traits
        assert!(Chromosome::new_uniform(8, 0).is_err()); // bits per trait too small
        assert!(Chromosome::new_uniform(8, 99).is_err()); // bits per trait too large

        // create prototype chromosome using basic method
        let _prototype = Chromosome::new_uniform(20, 5).unwrap();

        // create prototype chromosome using more specific method
        let mut prototype = Chromosome::new(&bits_per_trait_valid).unwrap();
        prototype.reset();

        // test initialization
        // invalid because first trait is only 6 bits
        let values_invalid = [999, 12, 11, 13, 63, 17, 3, 123456789];
        let values_valid = [0, 12, 11, 13, 63, 17, 3, 123456789];
        assert!(prototype.set_values(&values_invalid).is_err());
        prototype.set_values(&values_valid).unwrap();
        assert_eq!(prototype.value(999), None);
        assert_eq!(prototype.fraction(999), None);
        assert!(prototype.mutate(999).is_err());
        let p_clone = prototype.clone();
        assert!(prototype.crossover(&p_clone, &p_clone, 999).is_err());

        // check individual values
        assert_eq!(prototype.value(0), Some(0));
        assert_eq!(prototype.value(1), Some(12));
        assert_eq!(prototype.value(2), Some(11));
        assert_eq!(prototype.value(3), Some(13));
        assert_eq!(prototype.value(4), Some(63));
        assert_eq!(prototype.value(5), Some(17));
        assert_eq!(prototype.value(6), Some(3));
        assert_eq!(prototype.value(7), Some(123456789));

        // test initialization (float values)
        let valuesf_invalid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 999.0];
        let valuesf_valid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        assert!(prototype.set_fractions(&valuesf_invalid).is_err());
        prototype.set_fractions(&valuesf_valid).unwrap();

        // check individual values
        assert_abs_diff_eq!(prototype.fraction(0).unwrap(), 0.0, epsilon = 0.02);
        assert_abs_diff_eq!(prototype.fraction(1).unwrap(), 0.1, epsilon = 0.02);
        assert_abs_diff_eq!(prototype.fraction(2).unwrap(), 0.2, epsilon = 0.02);
        assert_abs_diff_eq!(prototype.fraction(3).unwrap(), 0.3, epsilon = 0.02);
        assert_abs_diff_eq!(prototype.fraction(4).unwrap(), 0.4, epsilon = 0.02);
        assert_abs_diff_eq!(prototype.fraction(5).unwrap(), 0.5, epsilon = 0.02);
        assert_abs_diff_eq!(prototype.fraction(6).unwrap(), 0.6, epsilon = 0.02);
        assert_abs_diff_eq!(prototype.fraction(7).unwrap(), 0.7, epsilon = 0.02);
    }

    #[test]
    fn chromosome_rejects_invalid_widths() {
        assert!(Chromosome::new(&[0]).is_err());
        assert!(Chromosome::new(&[1, 33]).is_err());
        assert!(Chromosome::new_uniform(1, 33).is_err());
        assert!(Chromosome::new_uniform(1, 0).is_err());
        assert!(Chromosome::new_uniform((u32::MAX / 32) as usize + 1, 32).is_err());
        assert_eq!(Chromosome::new(&[1, 32]).unwrap().num_bits(), 33);
    }

    #[test]
    fn chromosome_queries_distinguish_zero_from_missing_trait() {
        let chromosome = Chromosome::new(&[1]).unwrap();
        assert_eq!(chromosome.value(0), Some(0));
        assert_eq!(chromosome.fraction(0), Some(0.0));
        assert_eq!(chromosome.value(1), None);
        assert_eq!(chromosome.fraction(1), None);
    }

    #[test]
    fn chromosome_setters_validate_before_writing() {
        let mut chromosome = Chromosome::new(&[3, 4]).unwrap();
        chromosome.set_values(&[2, 7]).unwrap();
        for values in [&[0][..], &[0, 1, 2][..], &[0, 16][..]] {
            assert!(chromosome.set_values(values).is_err());
            assert_eq!((chromosome.value(0), chromosome.value(1)), (Some(2), Some(7)));
        }
        for fractions in [&[0.0][..], &[0.0, 0.5, 1.0][..], &[0.0, f32::NAN][..], &[0.0, -0.1][..], &[0.0, 1.1][..]] {
            assert!(chromosome.set_fractions(fractions).is_err());
            assert_eq!((chromosome.value(0), chromosome.value(1)), (Some(2), Some(7)));
        }
    }

    #[test]
    fn chromosome_fractions_include_both_endpoints() {
        let mut chromosome = Chromosome::new(&[3, 3, 32]).unwrap();
        chromosome.set_fractions(&[0.0, 0.5, 1.0]).unwrap();
        assert_eq!(chromosome.value(0), Some(0));
        assert_eq!(chromosome.value(1), Some(4));
        assert_eq!(chromosome.value(2), Some(u32::MAX as u64));
        assert_eq!(chromosome.fraction(0), Some(0.0));
        assert_abs_diff_eq!(chromosome.fraction(1).unwrap(), 4.0 / 7.0);
        assert_eq!(chromosome.fraction(2), Some(1.0));

        // Every representable value of a small trait survives read/write.
        let mut small = Chromosome::new(&[3]).unwrap();
        for expected in 0..=7 {
            small.set_values(&[expected]).unwrap();
            let fraction = small.fraction(0).unwrap();
            small.set_fractions(&[fraction]).unwrap();
            assert_eq!(small.value(0), Some(expected));
        }
    }

    #[test]
    fn chromosome_copy_and_crossover_reject_mismatched_layouts() {
        let mut chromosome = Chromosome::new(&[2, 3]).unwrap();
        chromosome.set_values(&[1, 5]).unwrap();
        let shorter = Chromosome::new(&[2]).unwrap();
        let different_widths = Chromosome::new(&[3, 2]).unwrap();
        let matching = chromosome.clone();
        assert!(chromosome.copy_from(&shorter).is_err());
        assert!(chromosome.copy_from(&different_widths).is_err());
        assert!(chromosome.crossover(&shorter, &matching, 2).is_err());
        assert!(chromosome.crossover(&matching, &different_widths, 2).is_err());
        assert_eq!((chromosome.value(0), chromosome.value(1)), (Some(1), Some(5)));
    }
}
