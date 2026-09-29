// continuous phase frequency-shift keying modulator

use crate::error::{Error, Result};
use crate::filter::fir::design::fir_design_gmsktx;
use crate::filter::FirInterpolationFilter;
use num_complex::Complex32;
use std::f32::consts::PI;

/// Frequency pulse shape for [`CpfskModulator`] and
/// [`CpfskDemodulator`](super::CpfskDemodulator).
///
/// The pulse sets how each symbol's frequency change is spread over time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpfskFilterType {
    /// Constant frequency for one symbol.
    Square,
    /// Raised-cosine frequency pulse lasting one symbol.
    RcosFull,
    /// Raised-cosine frequency pulse lasting two symbols.
    RcosPartial,
    /// Gaussian frequency pulse, as in Gaussian minimum-shift keying.
    ///
    /// `excess_bandwidth` is its bandwidth-time product.
    Gmsk,
}

/// Continuous-phase frequency-shift keying modulator.
///
/// Each symbol `s` of `M = 2^bits_per_symbol` changes the phase by
/// `pi * modulation_index * (2*s - M + 1)` radians. The pulse shape sets
/// how that change is spread over time. The output has unit magnitude and
/// continuous phase.
///
/// A [`CpfskDemodulator`](super::CpfskDemodulator) with the same settings
/// recovers the symbols when `modulation_index * (M - 1)` is below one. Above
/// that, the outermost symbols are misread. The demodulated symbols lag by the
/// sum of the two objects' [`Self::delay`] values, apart from the multi-level
/// raised-cosine modes noted on
/// [`CpfskDemodulator::delay`](super::CpfskDemodulator::delay).
#[derive(Clone, Debug)]
#[doc(alias = "Cpfskmod")]
pub struct CpfskModulator {
    bps: usize, // bits per symbol
    k: usize,   // samples per symbol
    beta: f32,  // filter bandwidth parameter
    h: f32,     // modulation index
    filter_type: CpfskFilterType,
    m_size: usize,       // constellation size (M = 2^bps)
    symbol_delay: usize, // transmit filter delay [symbols]

    // pulse-shaping filter
    interp: FirInterpolationFilter<f32, f32>,

    // phase integrator
    phase_interp: Vec<f32>, // phase interpolation buffer
    b0: f32,                // integrator coefficients
    b1: f32,
    a1: f32,
    v0: f32, // integrator state
    v1: f32,
}

impl CpfskModulator {
    /// Creates a CPFSK modulator with its phase at zero.
    ///
    /// # Arguments
    ///
    /// - `modulation_index`: scales the phase change given on the type. Must
    ///   be positive. A value of 0.5 with one bit per symbol is
    ///   minimum-shift keying.
    /// - `samples_per_symbol`: even, and at least two.
    /// - `filter_delay`: length of the Gaussian pulse on each side of its
    ///   center, in symbols. Only [`CpfskFilterType::Gmsk`] uses it, but it
    ///   must be nonzero for every shape.
    /// - `excess_bandwidth`: bandwidth-time product of the Gaussian pulse, in
    ///   `(0, 1]`. Only [`CpfskFilterType::Gmsk`] uses it, but it is checked
    ///   for every shape.
    ///
    /// # Errors
    ///
    /// Returns an error when `bits_per_symbol` or `filter_delay` is zero,
    /// `modulation_index` is zero or negative, `samples_per_symbol` is less
    /// than two or odd, or `excess_bandwidth` is outside `(0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics when `bits_per_symbol` is at least the width of `usize` and
    /// overflow checks are enabled. Without overflow checks, the shift wraps
    /// and the constellation size is wrong.
    pub fn new(
        bits_per_symbol: usize,
        modulation_index: f32,
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
        filter_type: CpfskFilterType,
    ) -> Result<Self> {
        // validate input
        if bits_per_symbol == 0 {
            return Err(Error::Config("bits/symbol must be greater than 0".into()));
        }
        if modulation_index <= 0.0 {
            return Err(Error::Config("modulation index must be greater than 0".into()));
        }
        if samples_per_symbol < 2 || !samples_per_symbol.is_multiple_of(2) {
            return Err(Error::Config("samples/symbol must be greater than 2 and even".into()));
        }
        if filter_delay == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if excess_bandwidth <= 0.0 || excess_bandwidth > 1.0 {
            return Err(Error::Config("filter roll-off must be in (0,1]".into()));
        }

        let bps = bits_per_symbol;
        let h = modulation_index;
        let k = samples_per_symbol;
        let m = filter_delay;
        let beta = excess_bandwidth;

        let m_size = 1 << bps;

        // create object depending upon input type
        let (b0, b1, ht_len, symbol_delay) = match filter_type {
            CpfskFilterType::Square => {
                // modify integrator for square pulse
                (0.0, 1.0, k, 1)
            }
            CpfskFilterType::RcosFull => {
                // rcos full
                (0.5, 0.5, k, 1)
            }
            CpfskFilterType::RcosPartial => {
                // TODO: adjust response based on 'm'
                (0.5, 0.5, 3 * k, 2)
            }
            CpfskFilterType::Gmsk => {
                // gmsk
                (0.5, 0.5, 2 * k * m + k + 1, m + 1)
            }
        };

        // create pulse-shaping filter and scale by modulation index
        //
        // f64 for stability
        let mut ht = cpfskmod_firdes(k, m, beta, filter_type, ht_len)?;
        let scale = std::f64::consts::PI * h as f64;
        for coeff in ht.iter_mut() {
            *coeff = (*coeff as f64 * scale) as f32;
        }

        let interp = FirInterpolationFilter::new(k, &ht[..ht_len])?;
        let phase_interp = vec![0.0; k];

        let mut q = Self {
            bps,
            k,
            beta,
            h,
            filter_type,
            m_size,
            symbol_delay,
            interp,
            phase_interp,
            b0,
            b1,
            a1: -1.0,
            v0: 0.0,
            v1: 0.0,
        };

        q.reset();
        Ok(q)
    }

    /// Creates a minimum-shift keying modulator.
    ///
    /// This is binary CPFSK with a modulation index of 0.5 and a square
    /// pulse.
    ///
    /// # Errors
    ///
    /// Returns an error when `samples_per_symbol` is less than two or odd.
    pub fn new_msk(samples_per_symbol: usize) -> Result<Self> {
        Self::new(1, 0.5, samples_per_symbol, 1, 1.0, CpfskFilterType::Square)
    }

    /// Creates a Gaussian minimum-shift keying modulator.
    ///
    /// This is binary CPFSK with a modulation index of 0.5 and a Gaussian
    /// pulse. `filter_delay` is the pulse length on each side of its center,
    /// in symbols.
    ///
    /// # Errors
    ///
    /// Returns an error when `samples_per_symbol` is less than two or odd,
    /// `filter_delay` is zero, or `bandwidth_time_product` is outside
    /// `(0, 1]`.
    pub fn new_gmsk(samples_per_symbol: usize, filter_delay: usize, bandwidth_time_product: f32) -> Result<Self> {
        Self::new(1, 0.5, samples_per_symbol, filter_delay, bandwidth_time_product, CpfskFilterType::Gmsk)
    }

    /// Clears the pulse-shaping history and sets the phase to zero.
    pub fn reset(&mut self) {
        self.interp.reset();
        self.v0 = 0.0;
        self.v1 = 0.0;
    }

    /// Returns the number of bits per symbol.
    pub fn bits_per_symbol(&self) -> usize {
        self.bps
    }

    /// Returns the modulation index.
    pub fn modulation_index(&self) -> f32 {
        self.h
    }

    /// Returns the number of samples per symbol.
    pub fn samples_per_symbol(&self) -> usize {
        self.k
    }

    /// Returns the delay the modulator adds, in symbols.
    ///
    /// This is 1 for [`CpfskFilterType::Square`] and
    /// [`CpfskFilterType::RcosFull`], 2 for [`CpfskFilterType::RcosPartial`]
    /// and `filter_delay + 1` for [`CpfskFilterType::Gmsk`].
    pub fn delay(&self) -> usize {
        self.symbol_delay
    }

    /// Returns `excess_bandwidth` as given to the constructor.
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Returns the pulse shape.
    pub fn filter_type(&self) -> CpfskFilterType {
        self.filter_type
    }

    /// Modulates one symbol into [`Self::samples_per_symbol`] samples.
    ///
    /// The samples go to the start of `output`, and the rest of `output` is
    /// left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `symbol` is `2^bits_per_symbol` or more, or
    /// `output` is shorter than [`Self::samples_per_symbol`]. The modulator
    /// and `output` are unchanged on error.
    pub fn modulate(&mut self, symbol: usize, output: &mut [Complex32]) -> Result<()> {
        if symbol >= self.m_size {
            return Err(Error::Range(format!("input symbol ({}) exceeds maximum ({})", symbol, self.m_size)));
        }
        if output.len() < self.k {
            return Err(Error::Range(format!(
                "output buffer length ({}) must be at least samples/symbol ({})",
                output.len(),
                self.k
            )));
        }

        // run interpolator
        let v = 2.0 * symbol as f32 - self.m_size as f32 + 1.0;
        self.interp.execute(v, &mut self.phase_interp)?;

        // integrate phase state
        for i in 0..self.k {
            // push phase through integrator
            self.v0 = self.phase_interp[i] - self.v1 * self.a1;
            let theta = self.v0 * self.b0 + self.v1 * self.b1;
            self.v1 = self.v0;

            // constrain state
            if self.v1 > 2.0 * PI {
                self.v1 -= 2.0 * PI;
            }
            if self.v1 < -2.0 * PI {
                self.v1 += 2.0 * PI;
            }

            // compute output
            output[i] = Complex32::from_polar(1.0, theta);
        }

        Ok(())
    }
}

/// Design transmit filter for CPFSK modulator
fn cpfskmod_firdes(k: usize, m: usize, beta: f32, filter_type: CpfskFilterType, ht_len: usize) -> Result<Vec<f32>> {
    let mut ht = vec![0.0; ht_len];

    match filter_type {
        CpfskFilterType::Square => {
            // square pulse
            if ht_len != k {
                return Err(Error::Config("invalid filter length (square)".into()));
            }
            for coeff in ht.iter_mut() {
                *coeff = 1.0;
            }
        }
        CpfskFilterType::RcosFull => {
            // full-response raised-cosine pulse
            if ht_len != k {
                return Err(Error::Config("invalid filter length (rcos full)".into()));
            }
            for (i, hti) in ht.iter_mut().enumerate() {
                *hti = 1.0 - (2.0 * PI * i as f32 / ht_len as f32).cos();
            }
        }
        CpfskFilterType::RcosPartial => {
            // partial-response raised-cosine pulse
            if ht_len != 3 * k {
                return Err(Error::Config("invalid filter length (rcos partial)".into()));
            }
            // initialize with zeros (already done)
            // adding raised-cosine pulse with half-symbol delay
            for i in 0..(2 * k) {
                ht[i + k / 2] = 1.0 - (2.0 * PI * i as f32 / (2 * k) as f32).cos();
            }
        }
        CpfskFilterType::Gmsk => {
            // Gauss minimum-shift keying pulse
            if ht_len != 2 * k * m + k + 1 {
                return Err(Error::Config("invalid filter length (gmsk)".into()));
            }
            // adding Gauss pulse with half-symbol delay
            let gmsk_coeffs = fir_design_gmsktx(k, m, beta, 0.0)?;
            for (i, &c) in gmsk_coeffs.iter().enumerate() {
                ht[k / 2 + i] = c;
            }
        }
    }

    // normalize pulse area to unity
    let ht_sum: f32 = ht.iter().sum();
    if ht_sum.abs() > 1e-10 {
        for coeff in ht.iter_mut() {
            *coeff /= ht_sum;
        }
    }

    Ok(ht)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cpfskmod_create() {
        // valid creation
        let result = CpfskModulator::new(2, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_ok());

        // invalid bps
        let result = CpfskModulator::new(0, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid modulation index
        let result = CpfskModulator::new(2, 0.0, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid k (odd)
        let result = CpfskModulator::new(2, 0.5, 3, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid k (too small)
        let result = CpfskModulator::new(2, 0.5, 1, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid m
        let result = CpfskModulator::new(2, 0.5, 4, 0, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid beta
        let result = CpfskModulator::new(2, 0.5, 4, 3, 0.0, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        let result = CpfskModulator::new(2, 0.5, 4, 3, 1.5, CpfskFilterType::Gmsk);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpfskmod_msk() {
        let result = CpfskModulator::new_msk(4);
        assert!(result.is_ok());
        let mod_ = result.unwrap();
        assert_eq!(mod_.bits_per_symbol(), 1);
        assert_abs_diff_eq!(mod_.modulation_index(), 0.5);
        assert_eq!(mod_.filter_type(), CpfskFilterType::Square);
    }

    #[test]
    fn test_cpfskmod_gmsk() {
        let result = CpfskModulator::new_gmsk(4, 3, 0.35);
        assert!(result.is_ok());
        let mod_ = result.unwrap();
        assert_eq!(mod_.bits_per_symbol(), 1);
        assert_abs_diff_eq!(mod_.modulation_index(), 0.5);
        assert_eq!(mod_.filter_type(), CpfskFilterType::Gmsk);
    }

    #[test]
    fn test_cpfskmod_modulate() -> Result<()> {
        let mut mod_ = CpfskModulator::new(2, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk)?;
        let mut y = vec![Complex32::new(0.0, 0.0); 4];

        // test valid symbol
        assert!(mod_.modulate(0, &mut y).is_ok());
        assert!(mod_.modulate(1, &mut y).is_ok());
        assert!(mod_.modulate(2, &mut y).is_ok());
        assert!(mod_.modulate(3, &mut y).is_ok());

        // test invalid symbol
        assert!(mod_.modulate(4, &mut y).is_err());

        // test buffer too short
        let mut y_short = vec![Complex32::new(0.0, 0.0); 2];
        assert!(mod_.modulate(0, &mut y_short).is_err());

        Ok(())
    }

    #[test]
    fn test_cpfskmod_output_unit_amplitude() -> Result<()> {
        let mut mod_ = CpfskModulator::new_gmsk(4, 3, 0.35)?;
        let mut y = vec![Complex32::new(0.0, 0.0); 4];

        // modulate some symbols and verify output has unit amplitude
        for s in 0..2 {
            mod_.modulate(s, &mut y)?;
            for sample in &y {
                assert_abs_diff_eq!(sample.norm(), 1.0, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_cpfskmod_gmsk_matches_liquid_over_long_run() -> Result<()> {
        let (k, m) = (4, 3);
        let mut q = CpfskModulator::new_gmsk(k, m, 0.35)?;
        let mut y = vec![Complex32::new(0.0, 0.0); k];

        // (symbol index, sample index, re, im)
        let expected = [
            (50usize, 0usize, 0.468684524f32, 0.883365631f32),
            (50, 2, 0.707106829, 0.707106769),
            (200, 0, 0.468684524, 0.883365631),
            (200, 3, 0.839165568, 0.543875992),
            (399, 0, 0.883365631, 0.468684465),
            (399, 2, 0.707106829, 0.707106769),
        ];

        let mut checked = 0;
        for i in 0..400 {
            q.modulate((i * 7 + 3) % 2, &mut y)?;
            for &(si, sj, re, im) in expected.iter() {
                if si == i {
                    assert_abs_diff_eq!(y[sj].re, re, epsilon = 2e-6);
                    assert_abs_diff_eq!(y[sj].im, im, epsilon = 2e-6);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, expected.len(), "not every reference sample was checked");

        Ok(())
    }

    #[test]
    fn test_cpfskmod_copy() -> Result<()> {
        // options
        let bps = 3;
        let h = 0.71;
        let k = 4;
        let m = 3;
        let beta = 0.35;
        let filter_type = CpfskFilterType::Gmsk;

        // create modulator
        let mut mod_orig = CpfskModulator::new(bps, h, k, m, beta, filter_type)?;

        let num_symbols = 80;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); k];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); k];
        let mut ms = crate::sequence::MaximalLengthSequence::from_degree(7)?;

        // run original object
        for _ in 0..num_symbols {
            let s = ms.generate_symbol(bps as u32) as usize;
            mod_orig.modulate(s, &mut buf_orig)?;
        }

        // copy object
        let mut mod_copy = mod_orig.clone();

        // run through both objects and compare
        for _ in 0..num_symbols {
            let s = ms.generate_symbol(bps as u32) as usize;
            mod_orig.modulate(s, &mut buf_orig)?;
            mod_copy.modulate(s, &mut buf_copy)?;

            for j in 0..k {
                assert_abs_diff_eq!(buf_orig[j].re, buf_copy[j].re, epsilon = 1e-6);
                assert_abs_diff_eq!(buf_orig[j].im, buf_copy[j].im, epsilon = 1e-6);
            }
        }

        Ok(())
    }
}
