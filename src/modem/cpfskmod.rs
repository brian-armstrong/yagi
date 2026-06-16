// continuous phase frequency-shift keying modulator

use std::f32::consts::PI;
use num_complex::Complex32;
use crate::error::{Error, Result};
use crate::filter::{FirInterpolationFilter, fir_design_gmsktx};

/// CPFSK filter type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpfskFilterType {
    /// Square pulse
    Square,
    /// Raised-cosine (full response)
    RcosFull,
    /// Raised-cosine (partial response)
    RcosPartial,
    /// Gauss minimum-shift keying pulse
    Gmsk,
}

/// Continuous-phase frequency-shift keying modulator
#[derive(Clone, Debug)]
pub struct Cpfskmod {
    bps: usize,              // bits per symbol
    k: usize,                // samples per symbol
    beta: f32,               // filter bandwidth parameter
    h: f32,                  // modulation index
    filter_type: CpfskFilterType,
    m_size: usize,           // constellation size (M = 2^bps)
    symbol_delay: usize,     // transmit filter delay [symbols]

    // pulse-shaping filter
    interp: FirInterpolationFilter<f32, f32>,

    // phase integrator
    phase_interp: Vec<f32>,  // phase interpolation buffer
    b0: f32,                 // integrator coefficients
    b1: f32,
    a1: f32,
    v0: f32,                 // integrator state
    v1: f32,
}

impl Cpfskmod {
    /// Create CPFSK modulator object (frequency modulator)
    ///
    /// # Arguments
    ///
    /// * `bps` - bits per symbol, bps > 0
    /// * `h` - modulation index, h > 0
    /// * `k` - samples/symbol, k > 1, k even
    /// * `m` - filter delay (symbols), m > 0
    /// * `beta` - filter bandwidth parameter, 0 < beta <= 1
    /// * `filter_type` - filter type (e.g. CpfskFilterType::Square)
    pub fn new(
        bps: usize,
        h: f32,
        k: usize,
        m: usize,
        beta: f32,
        filter_type: CpfskFilterType,
    ) -> Result<Self> {
        // validate input
        if bps == 0 {
            return Err(Error::Config("bits/symbol must be greater than 0".into()));
        }
        if h <= 0.0 {
            return Err(Error::Config("modulation index must be greater than 0".into()));
        }
        if k < 2 || (k % 2) != 0 {
            return Err(Error::Config("samples/symbol must be greater than 2 and even".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if beta <= 0.0 || beta > 1.0 {
            return Err(Error::Config("filter roll-off must be in (0,1]".into()));
        }

        let m_size = 1 << bps;

        // create object depending upon input type
        let (b0, b1, ht_len, symbol_delay) = match filter_type {
            CpfskFilterType::Square => {
                // modify integrator for square pulse
                (0.0, 1.0, k, 1)
            }
            CpfskFilterType::RcosFull => {
                (0.5, 0.5, k, 1)
            }
            CpfskFilterType::RcosPartial => {
                // TODO: adjust response based on 'm'
                (0.5, 0.5, 3 * k, 2)
            }
            CpfskFilterType::Gmsk => {
                (0.5, 0.5, 2 * k * m + k + 1, m + 1)
            }
        };

        // create pulse-shaping filter and scale by modulation index
        let mut ht = cpfskmod_firdes(k, m, beta, filter_type, ht_len)?;
        for coeff in ht.iter_mut() {
            *coeff *= PI * h;
        }

        let interp = FirInterpolationFilter::new(k, &ht, ht_len)?;
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

    /// Create modulator object for minimum-shift keying
    ///
    /// # Arguments
    ///
    /// * `k` - samples/symbol, k > 1, k even
    pub fn new_msk(k: usize) -> Result<Self> {
        Self::new(1, 0.5, k, 1, 1.0, CpfskFilterType::Square)
    }

    /// Create modulator object for Gaussian minimum-shift keying
    ///
    /// # Arguments
    ///
    /// * `k` - samples/symbol, k > 1, k even
    /// * `m` - filter delay (symbols), m > 0
    /// * `bt` - bandwidth-time factor, 0 < bt <= 1
    pub fn new_gmsk(k: usize, m: usize, bt: f32) -> Result<Self> {
        Self::new(1, 0.5, k, m, bt, CpfskFilterType::Gmsk)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.interp.reset();
        self.v0 = 0.0;
        self.v1 = 0.0;
    }

    /// Get modulator's number of bits per symbol
    pub fn get_bits_per_symbol(&self) -> usize {
        self.bps
    }

    /// Get modulator's modulation index
    pub fn get_modulation_index(&self) -> f32 {
        self.h
    }

    /// Get modulator's number of samples per symbol
    pub fn get_samples_per_symbol(&self) -> usize {
        self.k
    }

    /// Get modulator's filter delay [symbols]
    pub fn get_delay(&self) -> usize {
        self.symbol_delay
    }

    /// Get modulator's bandwidth parameter
    pub fn get_beta(&self) -> f32 {
        self.beta
    }

    /// Get modulator's filter type
    pub fn get_type(&self) -> CpfskFilterType {
        self.filter_type
    }

    /// Modulate sample
    ///
    /// # Arguments
    ///
    /// * `s` - input symbol
    /// * `y` - output sample array [size: k x 1]
    pub fn modulate(&mut self, s: usize, y: &mut [Complex32]) -> Result<()> {
        if s >= self.m_size {
            return Err(Error::Range(format!(
                "input symbol ({}) exceeds maximum ({})",
                s, self.m_size
            )));
        }
        if y.len() < self.k {
            return Err(Error::Range(format!(
                "output buffer length ({}) must be at least samples/symbol ({})",
                y.len(), self.k
            )));
        }

        // run interpolator
        let v = 2.0 * s as f32 - self.m_size as f32 + 1.0;
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
            y[i] = Complex32::from_polar(1.0, theta);
        }

        Ok(())
    }
}

/// Design transmit filter for CPFSK modulator
fn cpfskmod_firdes(
    k: usize,
    m: usize,
    beta: f32,
    filter_type: CpfskFilterType,
    ht_len: usize,
) -> Result<Vec<f32>> {
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
            for i in 0..ht_len {
                ht[i] = 1.0 - (2.0 * PI * i as f32 / ht_len as f32).cos();
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
        let result = Cpfskmod::new(2, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_ok());

        // invalid bps
        let result = Cpfskmod::new(0, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid modulation index
        let result = Cpfskmod::new(2, 0.0, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid k (odd)
        let result = Cpfskmod::new(2, 0.5, 3, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid k (too small)
        let result = Cpfskmod::new(2, 0.5, 1, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid m
        let result = Cpfskmod::new(2, 0.5, 4, 0, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid beta
        let result = Cpfskmod::new(2, 0.5, 4, 3, 0.0, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        let result = Cpfskmod::new(2, 0.5, 4, 3, 1.5, CpfskFilterType::Gmsk);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpfskmod_msk() {
        let result = Cpfskmod::new_msk(4);
        assert!(result.is_ok());
        let mod_ = result.unwrap();
        assert_eq!(mod_.get_bits_per_symbol(), 1);
        assert_abs_diff_eq!(mod_.get_modulation_index(), 0.5);
        assert_eq!(mod_.get_type(), CpfskFilterType::Square);
    }

    #[test]
    fn test_cpfskmod_gmsk() {
        let result = Cpfskmod::new_gmsk(4, 3, 0.35);
        assert!(result.is_ok());
        let mod_ = result.unwrap();
        assert_eq!(mod_.get_bits_per_symbol(), 1);
        assert_abs_diff_eq!(mod_.get_modulation_index(), 0.5);
        assert_eq!(mod_.get_type(), CpfskFilterType::Gmsk);
    }

    #[test]
    fn test_cpfskmod_modulate() -> Result<()> {
        let mut mod_ = Cpfskmod::new(2, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk)?;
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
        let mut mod_ = Cpfskmod::new_gmsk(4, 3, 0.35)?;
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
    fn test_cpfskmod_copy() -> Result<()> {
        // options
        let bps = 3;
        let h = 0.71;
        let k = 4;
        let m = 3;
        let beta = 0.35;
        let filter_type = CpfskFilterType::Gmsk;

        // create modulator
        let mut mod_orig = Cpfskmod::new(bps, h, k, m, beta, filter_type)?;

        let num_symbols = 80;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); k];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); k];
        let mut ms = crate::sequence::MSequence::create_default(7)?;

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
