//
// ofdmframegen.rs
//
// OFDM frame generator
//

use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::multichannel::ofdmframe::{
    ofdmframe_init_s0, ofdmframe_init_s1, OfdmFrameConfig, SubcarrierCounts, SubcarrierType,
};
use crate::sequence::MSequence;
use num_complex::Complex32;

/// OFDM frame generator
///
/// writes the PLCP preamble (two S0 symbols and one S1 symbol), then any
/// number of data symbols, then a tail. each symbol carries a cyclic prefix,
/// and adjacent symbols are cross-faded over `taper_len` samples to soften the
/// spectral splatter at symbol boundaries.
#[derive(Clone, Debug)]
pub struct OfdmFrameGen {
    num_subcarriers: usize,          // number of subcarriers
    cp_len: usize,                   // cyclic prefix length
    p: Vec<SubcarrierType>,          // subcarrier allocation

    // tapering/transition
    taper_len: usize,                // number of samples in tapering window/overlap
    taper: Vec<f32>,                 // tapering window
    postfix: Vec<Complex32>,         // overlapping symbol buffer

    // constants
    counts: SubcarrierCounts,        // number of null/pilot/data subcarriers

    // scaling factors
    g_data: f32,

    // transform object
    ifft: Fft<f32>,                  // ifft object
    x_freq: Vec<Complex32>,          // frequency-domain buffer
    x_time: Vec<Complex32>,          // time-domain buffer

    // PLCP short
    s0_time: Vec<Complex32>,         // short sequence (time)

    // PLCP long
    s1_time: Vec<Complex32>,         // long sequence (time)

    // pilot sequence
    ms_pilot: MSequence,
}

impl OfdmFrameGen {
    /// Create an OFDM frame generator from a validated configuration.
    pub fn new(config: &OfdmFrameConfig) -> Result<Self> {
        let num_subcarriers = config.num_subcarriers();
        let cp_len = config.cp_len();
        let taper_len = config.taper_len();
        let p = config.allocation().to_vec();
        let counts = config.counts();

        // allocate memory for transform objects
        let x_freq = vec![Complex32::new(0.0, 0.0); num_subcarriers];
        let x_time = vec![Complex32::new(0.0, 0.0); num_subcarriers];
        let ifft = Fft::new(num_subcarriers, Direction::Backward);

        // allocate memory for PLCP arrays
        let mut s0_freq = vec![Complex32::new(0.0, 0.0); num_subcarriers];
        let mut s0_time = vec![Complex32::new(0.0, 0.0); num_subcarriers];
        let mut s1_freq = vec![Complex32::new(0.0, 0.0); num_subcarriers];
        let mut s1_time = vec![Complex32::new(0.0, 0.0); num_subcarriers];
        ofdmframe_init_s0(&p, &mut s0_freq, &mut s0_time)?;
        ofdmframe_init_s1(&p, &mut s1_freq, &mut s1_time)?;

        // create tapering window and transition buffer
        let mut taper = vec![0.0f32; taper_len];
        let postfix = vec![Complex32::new(0.0, 0.0); taper_len];
        for (i, t) in taper.iter_mut().enumerate() {
            let s = (i as f32 + 0.5) / taper_len as f32;
            let g = (std::f32::consts::FRAC_PI_2 * s).sin();
            *t = g * g;
        }

        // compute scaling factor
        let g_data = 1.0 / ((counts.pilot + counts.data) as f32).sqrt();

        // set pilot sequence
        let ms_pilot = MSequence::create_default(8)?;

        Ok(Self {
            num_subcarriers,
            cp_len,
            p,
            taper_len,
            taper,
            postfix,
            counts,
            g_data,
            ifft,
            x_freq,
            x_time,
            s0_time,
            s1_time,
            ms_pilot,
        })
    }

    /// number of subcarriers
    pub fn num_subcarriers(&self) -> usize {
        self.num_subcarriers
    }

    /// cyclic prefix length
    pub fn cp_len(&self) -> usize {
        self.cp_len
    }

    /// taper (symbol overlap) length
    pub fn taper_len(&self) -> usize {
        self.taper_len
    }

    /// number of null subcarriers
    pub fn num_null(&self) -> usize {
        self.counts.null
    }

    /// number of pilot subcarriers
    pub fn num_pilot(&self) -> usize {
        self.counts.pilot
    }

    /// number of data subcarriers
    pub fn num_data(&self) -> usize {
        self.counts.data
    }

    /// length of each written symbol, in samples
    pub fn symbol_len(&self) -> usize {
        self.num_subcarriers + self.cp_len
    }

    pub fn reset(&mut self) {
        self.ms_pilot.reset();

        // clear internal postfix buffer
        self.postfix.fill(Complex32::new(0.0, 0.0));
    }

    /// write first PLCP short sequence 'symbol' to buffer
    ///
    /// ```text
    ///  |<- 2*cp->|<-       M       ->|<-       M       ->|
    ///  |         |                   |                   |
    ///      +-----+-------------------+-------------------+
    ///     /      |                   |                   |
    ///    /  ..s0 |        s0         |        s0         |
    ///   /        |                   |                   |
    ///  +---------+-------------------+-------------------+-----> time
    ///  |                        |                        |
    ///  |<-        s0[a]       ->|<-        s0[b]       ->|
    ///  |        M + cp_len      |        M + cp_len      |
    /// ```
    pub fn write_s0a(&mut self, y: &mut [Complex32]) -> Result<()> {
        if y.len() != self.symbol_len() {
            return Err(Error::Config(format!(
                "ofdmframegen_write_s0a(), output must be {} samples",
                self.symbol_len()
            )));
        }

        let m = self.num_subcarriers;
        // reduce 2*cp_len first so the subtraction cannot go negative in our
        //    unsigned arithmetic below
        let shift = (2 * self.cp_len) % m;
        for i in 0..m + self.cp_len {
            let k = (i + m - shift) % m;
            y[i] = self.s0_time[k];
        }

        // apply tapering window
        for i in 0..self.taper_len {
            y[i] *= self.taper[i];
        }
        Ok(())
    }

    /// write second PLCP short sequence 'symbol' to buffer
    pub fn write_s0b(&mut self, y: &mut [Complex32]) -> Result<()> {
        if y.len() != self.symbol_len() {
            return Err(Error::Config(format!(
                "ofdmframegen_write_s0b(), output must be {} samples",
                self.symbol_len()
            )));
        }

        let m = self.num_subcarriers;
        for i in 0..m + self.cp_len {
            let k = (i + m - self.cp_len) % m;
            y[i] = self.s0_time[k];
        }

        // copy postfix (first 'taper_len' samples of s0 symbol)
        self.postfix.copy_from_slice(&self.s0_time[..self.taper_len]);
        Ok(())
    }

    /// write PLCP long sequence symbol to buffer
    pub fn write_s1(&mut self, y: &mut [Complex32]) -> Result<()> {
        if y.len() != self.symbol_len() {
            return Err(Error::Config(format!(
                "ofdmframegen_write_s1(), output must be {} samples",
                self.symbol_len()
            )));
        }

        // copy S1 symbol to output, adding cyclic prefix and tapering window
        self.x_time.copy_from_slice(&self.s1_time);
        self.gensymbol(y);
        Ok(())
    }

    /// write OFDM symbol
    ///   x : input symbols, [size: num_subcarriers]
    ///   y : output samples, [size: num_subcarriers + cp_len]
    ///
    /// `x` is indexed by natural fft bin; only the data subcarriers are read.
    pub fn write_symbol(&mut self, x: &[Complex32], y: &mut [Complex32]) -> Result<()> {
        if x.len() != self.num_subcarriers {
            return Err(Error::Config(format!(
                "ofdmframegen_writesymbol(), input must be {} symbols",
                self.num_subcarriers
            )));
        }

        if y.len() != self.symbol_len() {
            return Err(Error::Config(format!(
                "ofdmframegen_writesymbol(), output must be {} samples",
                self.symbol_len()
            )));
        }

        // move frequency data to internal buffer
        let m = self.num_subcarriers;
        for i in 0..m {
            // start at mid-point (effective fftshift)
            let k = (i + m / 2) % m;

            self.x_freq[k] = match self.p[k] {
                // disabled subcarrier
                SubcarrierType::Null => Complex32::new(0.0, 0.0),
                // pilot subcarrier
                SubcarrierType::Pilot => {
                    let s = if self.ms_pilot.advance() != 0 { 1.0 } else { -1.0 };
                    Complex32::new(s * self.g_data, 0.0)
                }
                // data subcarrier
                SubcarrierType::Data => x[k] * self.g_data,
            };
        }

        // execute transform
        self.ifft.run(&self.x_freq, &mut self.x_time);

        // copy result to output, adding cyclic prefix and tapering window
        self.gensymbol(y);
        Ok(())
    }

    /// write tail to output, [size: taper_len]
    pub fn write_tail(&mut self, buffer: &mut [Complex32]) -> Result<()> {
        if buffer.len() != self.taper_len {
            return Err(Error::Config(format!(
                "ofdmframegen_writetail(), output must be {} samples",
                self.taper_len
            )));
        }

        // write tail to output, applying tapering window
        for i in 0..self.taper_len {
            buffer[i] = self.postfix[i] * self.taper[self.taper_len - i - 1];
        }
        Ok(())
    }

    /// generate symbol (add cyclic prefix/postfix, overlap)
    ///
    /// ```text
    ///  ->|   |<- taper_len
    ///    +   +-----+-------------------+
    ///     \ /      |                   |
    ///      X       |      symbol       |
    ///     / \      |                   |
    ///    +---+-----+-------------------+----> time
    ///    |         |                   |
    ///    |<- cp  ->|<-       M       ->|
    /// ```
    ///
    /// reads `self.x_time` and the previous symbol's postfix, writes the
    /// symbol to `buffer` and this symbol's postfix back to `self.postfix`.
    fn gensymbol(&mut self, buffer: &mut [Complex32]) {
        let m = self.num_subcarriers;

        // copy input symbol with cyclic prefix to output symbol
        buffer[..self.cp_len].copy_from_slice(&self.x_time[m - self.cp_len..]);
        buffer[self.cp_len..self.cp_len + m].copy_from_slice(&self.x_time);

        // apply tapering window to over-lapping regions
        for i in 0..self.taper_len {
            buffer[i] *= self.taper[i];
            buffer[i] += self.postfix[i] * self.taper[self.taper_len - i - 1];
        }

        // copy post-fix to output (first 'taper_len' samples of input symbol)
        self.postfix
            .copy_from_slice(&self.x_time[..self.taper_len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    fn new_framegen(
        num_subcarriers: usize,
        cp_len: usize,
        taper_len: usize,
        allocation: Option<&[SubcarrierType]>,
    ) -> Result<OfdmFrameGen> {
        let config = OfdmFrameConfig::new(num_subcarriers, cp_len, taper_len, allocation)?;
        OfdmFrameGen::new(&config)
    }

    #[test]
    #[autotest_annotate(autotest_ofdmframegen_config)]
    fn test_ofdmframegen_config() {
        // check invalid function calls
        assert!(new_framegen(0, 16, 4, None).is_err()); // too few subcarriers
        assert!(new_framegen(7, 16, 4, None).is_err()); // too few subcarriers
        assert!(new_framegen(65, 16, 4, None).is_err()); // odd-length subcarriers
        assert!(new_framegen(64, 66, 4, None).is_err()); // cp length too large
        assert!(new_framegen(64, 16, 24, None).is_err()); // taper > cyclic prefix

        // create proper object and test configurations
        let config = OfdmFrameConfig::new(64, 16, 4, None).unwrap();
        let _q = OfdmFrameGen::new(&config).unwrap();
    }

    #[test]
    fn test_buffer_length_validation() {
        let m = 64;
        let cp_len = 16;
        let taper_len = 4;
        let mut q = new_framegen(m, cp_len, taper_len, None).unwrap();

        let x = vec![Complex32::new(0.0, 0.0); m];
        let mut wrong = vec![Complex32::new(0.0, 0.0); m];
        let mut right = vec![Complex32::new(0.0, 0.0); m + cp_len];

        assert!(q.write_s0a(&mut wrong).is_err());
        assert!(q.write_s0b(&mut wrong).is_err());
        assert!(q.write_s1(&mut wrong).is_err());
        assert!(q.write_symbol(&x, &mut wrong).is_err());
        assert!(q.write_symbol(&x[..m - 1], &mut right).is_err());

        let mut short_tail = vec![Complex32::new(0.0, 0.0); taper_len - 1];
        assert!(q.write_tail(&mut short_tail).is_err());
    }

    #[test]
    fn test_large_cyclic_prefix() {
        let m = 64;
        for cp_len in [0usize, 1, 2, 16, 31, 32, 33, 48, 63, 64] {
            let mut q = new_framegen(m, cp_len, 0, None).unwrap();
            let mut y = vec![Complex32::new(0.0, 0.0); m + cp_len];

            q.write_s0a(&mut y).unwrap();
            // the output must be a permutation of s0, not garbage
            let energy: f32 = y.iter().map(|v| v.norm_sqr()).sum();
            assert!(energy > 0.0, "cp_len={} produced no energy", cp_len);
            assert!(
                y.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
                "cp_len={} produced non-finite samples",
                cp_len
            );

            q.write_s0b(&mut y).unwrap();
            q.write_s1(&mut y).unwrap();
            let x = vec![Complex32::new(0.5, 0.5); m];
            q.write_symbol(&x, &mut y).unwrap();
        }

        // beyond M is still rejected
        assert!(new_framegen(m, m + 1, 0, None).is_err());
    }

    #[test]
    fn test_s0a_wrapping() {
        let m = 64;
        for (cp_len, want) in [
            (16usize, [32usize, 33, 34, 35]),
            (32, [0, 1, 2, 3]),
            (33, [62, 63, 0, 1]),
            (34, [60, 61, 62, 63]),
        ] {
            let mut q = new_framegen(m, cp_len, 0, None).unwrap();
            let mut y = vec![Complex32::new(0.0, 0.0); m + cp_len];
            q.write_s0a(&mut y).unwrap();

            // s0_time is what write_s0a indexes. compare the first few samples.
            for (i, &k) in want.iter().enumerate() {
                assert_eq!(
                    y[i], q.s0_time[k],
                    "cp_len={} sample {} should be s0[{}]",
                    cp_len, i, k
                );
            }
        }
    }

    #[test]
    fn test_taper_window_symmetry() {
        for taper_len in [1usize, 2, 4, 8, 16] {
            let q = new_framegen(64, 16, taper_len, None).unwrap();
            for i in 0..taper_len {
                assert_abs_diff_eq!(
                    q.taper[i] + q.taper[taper_len - i - 1],
                    1.0,
                    epsilon = 1e-6
                );
            }
            for i in 1..taper_len {
                assert!(q.taper[i] > q.taper[i - 1], "taper not increasing");
            }
        }
    }

    #[test]
    fn test_cyclic_prefix_is_symbol_tail() {
        let m = 64;
        let cp_len = 16;
        let mut q = new_framegen(m, cp_len, 0, None).unwrap();

        let x = vec![Complex32::new(1.0, -0.5); m];
        let mut y = vec![Complex32::new(0.0, 0.0); m + cp_len];
        q.write_symbol(&x, &mut y).unwrap();

        for i in 0..cp_len {
            assert_eq!(y[i], y[m + i], "cp sample {} should equal tail", i);
        }
    }

    #[test]
    fn test_s0_symbols_are_half_periodic() {
        let m = 64;
        let cp_len = 16;
        let mut q = new_framegen(m, cp_len, 0, None).unwrap();

        let mut y = vec![Complex32::new(0.0, 0.0); m + cp_len];
        q.write_s0a(&mut y).unwrap();
        for i in 0..m / 2 {
            assert_abs_diff_eq!(y[i].re, y[i + m / 2].re, epsilon = 1e-5);
            assert_abs_diff_eq!(y[i].im, y[i + m / 2].im, epsilon = 1e-5);
        }

        q.write_s0b(&mut y).unwrap();
        for i in 0..m / 2 {
            assert_abs_diff_eq!(y[i].re, y[i + m / 2].re, epsilon = 1e-5);
            assert_abs_diff_eq!(y[i].im, y[i + m / 2].im, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_null_subcarriers_carry_no_energy() {
        let m = 64;
        let mut q = new_framegen(m, 16, 0, None).unwrap();

        let x = vec![Complex32::new(100.0, 100.0); m];
        let mut y = vec![Complex32::new(0.0, 0.0); m + 16];
        q.write_symbol(&x, &mut y).unwrap();

        for k in 0..m {
            if q.p[k] == SubcarrierType::Null {
                assert_eq!(q.x_freq[k], Complex32::new(0.0, 0.0), "null bin {}", k);
            }
        }
    }

    #[test]
    fn test_custom_allocation() {
        let m = 64;

        // a user allocation must be honored and its length checked
        let mut p = vec![SubcarrierType::Data; m];
        p[0] = SubcarrierType::Pilot;
        p[1] = SubcarrierType::Pilot;
        p[2] = SubcarrierType::Null;

        let q = new_framegen(m, 16, 4, Some(&p)).unwrap();
        assert_eq!(q.num_pilot(), 2);
        assert_eq!(q.num_null(), 1);
        assert_eq!(q.num_data(), m - 3);

        // wrong length is rejected
        assert!(new_framegen(m, 16, 4, Some(&p[..m - 1])).is_err());

        // an allocation that fails validation is rejected
        let bad = vec![SubcarrierType::Null; m];
        assert!(new_framegen(m, 16, 4, Some(&bad)).is_err());
    }
}
