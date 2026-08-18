//
// CVSD: continuously variable slope delta
//

use crate::error::{Error, Result};
use crate::filter::IirFilter;

/// continuously variable slope delta encoder/decoder
///
/// A CVSD codec transmits one bit per sample. For each sample, the codec can
/// either add (1) or subtract (0) one step size's worth of amplitude to the
/// signal. The step size adapts, growing when the output is all-ones or all-zeros
/// and shrinking otherwise.
///
/// # Signal conditioning
///
/// Pre-emphasis/de-emphasis filters boost the high-frequency contributions
/// of the signal to improve intelligibility.
#[derive(Debug, Clone)]
pub struct Cvsd {
    num_bits: usize,
    bitref: u8,             // historical bit reference
    bitmask: u8,            // historical bit reference mask
    ref_: f32,              // internal reference

    zeta: f32,              // delta step factor
    delta: f32,             // current step size
    delta_min: f32,         // minimum delta
    delta_max: f32,         // maximum delta

    alpha: f32,             // pre-/de-emphasis filter coefficient
    beta: f32,              // DC-blocking coefficient (decoder)
    // `None` when signal conditioning is disabled
    filters: Option<Filters>,
}

#[derive(Debug, Clone)]
struct Filters {
    prefilt: IirFilter<f32, f32>,  // pre-emphasis filter (encoder)
    postfilt: IirFilter<f32, f32>, // de-emphasis filter (decoder)
}

impl Cvsd {
    /// create cvsd object
    ///
    ///  num_bits   :   number of adjacent bits to observe
    ///  zeta       :   slope adjustment multiplier
    pub fn new(num_bits: usize, zeta: f32) -> Result<Self> {
        if num_bits == 0 {
            return Err(Error::Config("cvsd num_bits must be positive".into()));
        }
        if zeta <= 1.0 {
            return Err(Error::Config("cvsd zeta must be greater than 1".into()));
        }
        if num_bits > 8 {
            return Err(Error::Config("cvsd num_bits must be no more than 8".into()));
        }

        Ok(Self {
            num_bits,
            bitref: 0,
            bitmask: ((1u32 << num_bits) - 1) as u8,
            ref_: 0.0,
            zeta,
            delta: 0.01,
            delta_min: 0.01,
            delta_max: 1.0,
            alpha: 0.0,
            beta: 0.0,
            filters: None,
        })
    }

    /// enable signal conditioning
    pub fn with_conditioning(
        mut self,
        alpha: f32,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(Error::Config("cvsd alpha must be in [0,1]".into()));
        }

        // DC-blocking parameter
        let beta = 0.99;

        // design pre-emphasis filter
        let b_pre = [1.0, -alpha];
        let a_pre = [1.0, 0.0];
        let prefilt = IirFilter::new(&b_pre, &a_pre)?;

        // design post-emphasis filter
        let b_post = [1.0, -1.0, 0.0];
        let a_post = [1.0, -(alpha + beta), alpha * beta];
        let postfilt = IirFilter::new(&b_post, &a_post)?;

        let filters = Some(Filters { prefilt, postfilt });

        self.alpha = alpha;
        self.beta = beta;
        self.filters = filters;

        Ok(self)
    }

    /// number of adjacent bits observed
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// slope adjustment multiplier
    pub fn zeta(&self) -> f32 {
        self.zeta
    }

    /// pre-/de-emphasis filter coefficient
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// DC-blocking coefficient used by the decoder's de-emphasis filter
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// is signal conditioning applied?
    pub fn conditioning(&self) -> bool {
        self.filters.is_some()
    }

    // advance the bit history and adapt the step size
    fn step(&mut self, bit: u8) {
        // shift last value into buffer
        self.bitref <<= 1;
        self.bitref |= bit & 0x01;
        self.bitref &= self.bitmask;

        // update delta
        if self.bitref == 0 || self.bitref == self.bitmask {
            self.delta *= self.zeta; // increase delta
        } else {
            self.delta /= self.zeta; // decrease delta
        }

        // limit delta
        self.delta = self.delta.min(self.delta_max);
        self.delta = self.delta.max(self.delta_min);

        // update reference
        self.ref_ += if bit & 0x01 != 0 { self.delta } else { -self.delta };

        // limit reference
        self.ref_ = self.ref_.clamp(-1.0, 1.0);
    }

    /// encode single sample
    pub fn encode(&mut self, audio_sample: f32) -> u8 {
        // push audio sample through pre-filter
        let y = match &mut self.filters {
            Some(f) => f.prefilt.execute(audio_sample),
            None => audio_sample,
        };

        // determine output value
        let bit = if self.ref_ > y { 0 } else { 1 };

        self.step(bit);

        bit
    }

    /// decode single sample
    pub fn decode(&mut self, bit: u8) -> f32 {
        self.step(bit);

        // push reference value through post-filter
        match &mut self.filters {
            Some(f) => f.postfilt.execute(self.ref_),
            None => self.ref_,
        }
    }

    /// encode 8 samples
    pub fn encode8(&mut self, audio: &[f32]) -> Result<u8> {
        if audio.len() < 8 {
            return Err(Error::Config("cvsd encode8 requires 8 samples".into()));
        }

        let mut data = 0u8;
        for &sample in &audio[..8] {
            data <<= 1;
            data |= self.encode(sample);
        }
        Ok(data)
    }

    /// decode 8 samples
    pub fn decode8(&mut self, data: u8, audio: &mut [f32]) -> Result<()> {
        if audio.len() < 8 {
            return Err(Error::Config("cvsd decode8 requires room for 8 samples".into()));
        }

        for (i, sample) in audio[..8].iter_mut().enumerate() {
            let bit = (data >> (8 - i - 1)) & 0x01;
            *sample = self.decode(bit);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    // check RMS error
    #[test]
    #[autotest_annotate(autotest_cvsd_rmse_sine)]
    fn test_cvsd_rmse_sine() {
        let n = 256;
        let nbits = 3;
        let zeta = 1.5;
        let alpha = 0.90;

        // create cvsd codecs
        let mut cvsd_encoder = Cvsd::new(nbits, zeta).unwrap().with_conditioning(alpha).unwrap();
        let mut cvsd_decoder = Cvsd::new(nbits, zeta).unwrap().with_conditioning(alpha).unwrap();
        // no print check
        assert_eq!(cvsd_encoder.num_bits(), nbits);
        assert_eq!(cvsd_encoder.zeta(), zeta);
        assert_eq!(cvsd_encoder.alpha(), alpha);

        let mut phi = 0.0f32;
        let dphi = 0.1f32;
        let mut rmse = 0.0f32;
        for _ in 0..n {
            let x = 0.5 * phi.sin();
            let b = cvsd_encoder.encode(x);
            let y = cvsd_decoder.decode(b);

            rmse += (x - y) * (x - y);
            phi += dphi;
        }

        let rmse = 10.0 * (rmse / n as f32).log10();
        assert!(rmse < -20.0, "rmse = {} dB", rmse);
    }

    // check RMS error running in blocks of 8 samples
    #[test]
    #[autotest_annotate(autotest_cvsd_rmse_sine8)]
    fn test_cvsd_rmse_sine8() {
        let n = 256;
        let nbits = 3;
        let zeta = 1.5;
        let alpha = 0.90;

        let mut cvsd_encoder = Cvsd::new(nbits, zeta).unwrap().with_conditioning(alpha).unwrap();
        let mut cvsd_decoder = Cvsd::new(nbits, zeta).unwrap().with_conditioning(alpha).unwrap();

        let mut phi = 0.0f32;
        let dphi = 0.1f32;
        let mut buf_0 = [0.0f32; 8];
        let mut buf_1 = [0.0f32; 8];
        let mut rmse = 0.0f32;
        for _ in 0..n {
            // generate tone
            for j in 0..8 {
                buf_0[j] = 0.5 * phi.sin();
                phi += dphi;
            }
            // encode/decode
            let byte = cvsd_encoder.encode8(&buf_0).unwrap();
            cvsd_decoder.decode8(byte, &mut buf_1).unwrap();

            // accumulate RMS error
            for j in 0..8 {
                rmse += (buf_0[j] - buf_1[j]) * (buf_0[j] - buf_1[j]);
            }
        }

        let rmse = 10.0 * (rmse / (n * 8) as f32).log10();
        assert!(rmse < -20.0, "rmse = {} dB", rmse);
    }

    #[test]
    #[autotest_annotate(autotest_cvsd_invalid_config)]
    fn test_cvsd_invalid_config() {
        // test invalid configuration to new()
        assert!(Cvsd::new(0, 2.0).is_err()); // too few bits
        assert!(Cvsd::new(2, 1.0).is_err()); // zeta too small
        assert!(Cvsd::new(2, 0.5).is_err()); // zeta too small
        assert!(Cvsd::new(2, 2.0).unwrap().with_conditioning(-1.0).is_err()); // alpha too small
        assert!(Cvsd::new(2, 2.0).unwrap().with_conditioning(2.0).is_err()); // alpha too large

        // liquid silently allows num_bits too big, but we don't
        assert!(Cvsd::new(9, 2.0).is_err());
    }

    #[test]
    fn test_cvsd_encode8_matches_encode() {
        let mut single = Cvsd::new(3, 1.5).unwrap().with_conditioning(0.9).unwrap();
        let mut block = Cvsd::new(3, 1.5).unwrap().with_conditioning(0.9).unwrap();

        let mut phi = 0.0f32;
        for _ in 0..32 {
            let mut buf = [0.0f32; 8];
            for s in buf.iter_mut() {
                *s = 0.5 * phi.sin();
                phi += 0.1;
            }

            // pack the single-sample bits the way encode8 does
            let mut expected = 0u8;
            for &s in buf.iter() {
                expected <<= 1;
                expected |= single.encode(s);
            }

            assert_eq!(block.encode8(&buf).unwrap(), expected);
        }
    }

    #[test]
    fn test_cvsd_decode8_matches_decode() {
        let mut single = Cvsd::new(3, 1.5).unwrap().with_conditioning(0.9).unwrap();
        let mut block = Cvsd::new(3, 1.5).unwrap().with_conditioning(0.9).unwrap();

        let mut buf = [0.0f32; 8];
        for byte in [0x00u8, 0xff, 0xa5, 0x5a, 0x01, 0x80, 0x3c] {
            block.decode8(byte, &mut buf).unwrap();
            for (i, &y) in buf.iter().enumerate() {
                let bit = (byte >> (8 - i - 1)) & 0x01;
                assert_eq!(single.decode(bit), y);
            }
        }
    }

    #[test]
    fn test_cvsd_reference_bounded() {
        for &(nbits, zeta) in &[(1usize, 1.5f32), (3, 1.5), (8, 2.0)] {
            let mut q = Cvsd::new(nbits, zeta).unwrap();
            // all-ones drives delta to its maximum and the reference to +1
            for _ in 0..200 {
                let y = q.decode(1);
                assert!(y <= 1.0 + 1e-6, "nbits={}: {}", nbits, y);
            }
            assert!((q.decode(1) - 1.0).abs() < 1e-6);

            for _ in 0..400 {
                let y = q.decode(0);
                assert!(y >= -1.0 - 1e-6, "nbits={}: {}", nbits, y);
            }
            assert!((q.decode(0) + 1.0).abs() < 1e-6);
        }
    }

}
