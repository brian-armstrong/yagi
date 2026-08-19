//
// channel.rs
//
// Generic channel emulator: applies multipath, shadowing, carrier offset, and
// additive white Gaussian noise to a complex baseband signal.
//

use num_complex::Complex32;

use crate::channel::noise::NoiseSource;
use crate::error::{Error, Result};
use crate::filter::{FirFilter, IirFilter};
use crate::nco::{Osc, OscScheme};
use crate::sequence::MSequence;

/// maximum multipath filter length liquid accepts
const MAX_MULTIPATH_LEN: usize = 1000;

/// Channel emulator
///
/// Impairments are opt-in. Any enabled impairments are applied in a fixed order
/// regardless of the order the `add_*` methods are called: multipath,
/// then shadowing, then carrier offset, then AWGN.
#[derive(Debug, Clone)]
pub struct Channel {
    // additive white Gauss noise
    enabled_awgn: bool,          // AWGN enabled?
    gamma: f32,                  // channel gain
    nstd: f32,                   // noise standard deviation
    noise_floor: f32,            // noise floor density [dB]
    snr: f32,                    // signal-to-noise ratio [dB]

    // carrier offset
    enabled_carrier: bool,       // carrier offset enabled?
    dphi: f32,                   // carrier frequency offset [radians/sample]
    phi: f32,                    // carrier phase offset [radians]
    nco: Osc,                    // oscillator

    // multi-path channel
    enabled_multipath: bool,     // enable multi-path channel filter?
    channel_filter: FirFilter<Complex32, Complex32>, // multi-path channel filter
    h: Vec<Complex32>,           // multi-path channel filter coefficients

    // shadowing channel
    enabled_shadowing: bool,     // enable shadowing?
    shadowing_filter: Option<IirFilter<f32, f32>>, // shadowing filter object
    shadowing_std: f32,          // shadowing standard deviation
    shadowing_fd: f32,           // shadowing Doppler frequency

    noise: NoiseSource,          // Gaussian source for AWGN and shadowing
}

impl Channel {
    /// create channel object with default parameters
    ///
    /// all impairments start disabled
    pub fn new() -> Self {
        let h = vec![Complex32::new(1.0, 0.0)];
        Self {
            enabled_awgn: false,
            gamma: 1.0,
            nstd: 0.0,
            noise_floor: 0.0,
            snr: 0.0,

            enabled_carrier: false,
            dphi: 0.0,
            phi: 0.0,
            nco: Osc::new(OscScheme::Vco),

            enabled_multipath: false,
            channel_filter: FirFilter::new(&h).expect("unit filter is valid"),
            h,

            enabled_shadowing: false,
            shadowing_filter: None,
            shadowing_std: 0.0,
            shadowing_fd: 0.0,

            noise: NoiseSource::Global,
        }
    }

    /// create channel object drawing noise from a seeded generator
    pub fn new_seeded(seed: u64) -> Self {
        let mut q = Self::new();
        q.noise = NoiseSource::seeded(seed);
        q
    }

    /// reset internal state of the channel
    ///
    /// clears the multipath filter history and the oscillator phase
    pub fn reset(&mut self) {
        self.channel_filter.reset();
        self.nco.reset();
        self.nco.set_frequency(self.dphi);
        self.nco.set_phase(self.phi);
        if let Some(f) = self.shadowing_filter.as_mut() {
            f.reset();
        }
    }

    /// apply additive white Gaussian noise impairment
    ///
    /// # Arguments
    ///
    /// * `noise_floor` - noise floor power spectral density [dB]
    /// * `snr` - signal-to-noise ratio [dB]
    pub fn add_awgn(&mut self, noise_floor: f32, snr: f32) {
        // enable module
        self.enabled_awgn = true;

        self.noise_floor = noise_floor;
        self.snr = snr;

        // set values appropriately
        self.nstd = 10.0f32.powf(noise_floor / 20.0);
        self.gamma = 10.0f32.powf((self.snr + self.noise_floor) / 20.0);
    }

    /// apply carrier offset impairment
    ///
    /// # Arguments
    ///
    /// * `frequency` - carrier frequency offset [radians/sample]
    /// * `phase` - carrier phase offset [radians]
    pub fn add_carrier_offset(&mut self, frequency: f32, phase: f32) {
        // enable module
        self.enabled_carrier = true;

        // carrier frequency/phase offsets
        self.dphi = frequency;
        self.phi = phase;

        // set values appropriately
        self.nco.set_frequency(self.dphi);
        self.nco.set_phase(self.phi);
    }

    /// apply specific multi-path channel impairment coefficients
    ///
    /// # Arguments
    ///
    /// * `h` - channel coefficients
    pub fn add_multipath(&mut self, h: &[Complex32]) -> Result<()> {
        if h.len() == 0 {
            return Err(Error::Mode(
                "channel_add_multipath(), filter length is zero".into(),
            ));
        }
        if h.len() > MAX_MULTIPATH_LEN {
            return Err(Error::Mode(
                "channel_add_multipath(), filter length exceeds maximum".into(),
            ));
        }

        // enable module
        self.enabled_multipath = true;

        self.h.resize(h.len(), Complex32::new(0.0, 0.0));
        self.h.copy_from_slice(h);

        // re-create channel filter
        self.channel_filter = FirFilter::new(&self.h)?;
        Ok(())
    }

    /// apply random multi-path channel impairment cofficients
    ///
    /// # Arguments
    ///
    /// * `h_len` - number of channel coefficients
    pub fn add_multipath_random(&mut self, h_len: usize) -> Result<()> {
        if h_len == 0 {
            return Err(Error::Mode(
                "channel_add_multipath_random(), filter length is zero".into(),
            ));
        }
        if h_len > MAX_MULTIPATH_LEN {
            return Err(Error::Mode(
                "channel_add_multipath_random(), filter length exceeds maximum".into(),
            ));
        }

        // enable module
        self.enabled_multipath = true;

        self.h.resize(h_len, Complex32::new(0.0, 0.0));

        // generate random coefficients using m-sequence generator
        self.h[0] = Complex32::new(1.0, 0.0);
        let mut ms = MSequence::create_default(14)?;
        for i in 1..h_len {
            let vi = ms.generate_symbol(8) as f32 / 256.0 - 0.5;
            let vq = ms.generate_symbol(8) as f32 / 256.0 - 0.5;
            self.h[i] = Complex32::new(vi, vq) * 0.5;
        }

        // re-create channel filter
        self.channel_filter = FirFilter::new(&self.h)?;
        Ok(())
    }

    /// apply slowly-varying shadowing impairment
    ///
    /// # Arguments
    ///
    /// * `sigma` - std. deviation for log-normal shadowing
    /// * `fd` - Doppler frequency, `fd` in (0, 0.5)
    pub fn add_shadowing(&mut self, sigma: f32, fd: f32) -> Result<()> {
        if self.enabled_shadowing {
            return Err(Error::Mode(
                "channel_add_shadowing(), shadowing already enabled".into(),
            ));
        }
        if sigma <= 0.0 {
            return Err(Error::Mode(
                "channel_add_shadowing(), standard deviation less than or equal to zero".into(),
            ));
        }
        if fd <= 0.0 || fd >= 0.5 {
            return Err(Error::Mode(
                "channel_add_shadowing(), Doppler frequency must be in (0,0.5)".into(),
            ));
        }

        // enable module
        self.enabled_shadowing = true;

        self.shadowing_std = sigma;
        self.shadowing_fd = fd;

        // single-pole low-pass filter shaping the shadowing process
        let alpha = self.shadowing_fd;
        let a = [1.0, alpha - 1.0];
        let b = [alpha, 0.0];
        self.shadowing_filter = Some(IirFilter::new(&b, &a)?);
        Ok(())
    }

    /// apply channel impairments on a single input sample
    ///
    /// # Arguments
    ///
    /// * `x` - input sample
    ///
    /// # Returns
    ///
    /// The impaired output sample
    pub fn execute(&mut self, x: Complex32) -> Complex32 {
        // apply filter
        let mut r = if self.enabled_multipath {
            self.channel_filter.push(x);
            self.channel_filter.execute()
        } else {
            x
        };

        // apply shadowing if enabled
        if self.enabled_shadowing {
            let n = self.noise.randnf() * self.shadowing_std;
            let mut g = self
                .shadowing_filter
                .as_mut()
                .expect("filter exists when shadowing is enabled")
                .execute(n);
            g /= self.shadowing_fd * 6.9;
            g = 10.0f32.powf(g / 20.0);
            r *= g;
        }

        // apply carrier if enabled
        if self.enabled_carrier {
            r = self.nco.mix_up(r);
            self.nco.step();
        }

        // apply AWGN if enabled
        if self.enabled_awgn {
            r *= self.gamma;
            let n = Complex32::new(self.noise.randnf(), self.noise.randnf());
            r += n * self.nstd * std::f32::consts::FRAC_1_SQRT_2;
        }

        r
    }

    /// apply channel impairments on a block of samples
    ///
    /// # Arguments
    ///
    /// * `x` - input array
    /// * `y` - output array, same length as `x`
    pub fn execute_block(&mut self, x: &[Complex32], y: &mut [Complex32]) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::Config(
                "channel_execute_block(), input and output lengths must match".into(),
            ));
        }
        // apply channel effects on each input sample
        for (x_i, y_i) in x.iter().zip(y.iter_mut()) {
            *y_i = self.execute(*x_i);
        }
        Ok(())
    }

    /// get the multipath channel coefficients currently in use
    pub fn get_multipath(&self) -> &[Complex32] {
        &self.h
    }
}

impl Default for Channel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    // oracle literals are pasted verbatim from liquid's %.9e output
    #![allow(clippy::excessive_precision)]

    use super::*;
    use approx::assert_abs_diff_eq;

    /// deterministic test signal, matching the C cross-check harness
    fn test_signal(n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                let t = i as f32;
                Complex32::new(
                    (0.3 * t + 0.01 * t * t).cos(),
                    (0.17 * t - 0.02 * t * t).sin(),
                )
            })
            .collect()
    }

    #[test]
    fn test_channel_passthrough() {
        // a channel with nothing enabled must not alter the signal at all
        let x = test_signal(32);
        let mut y = vec![Complex32::new(0.0, 0.0); x.len()];
        let mut q = Channel::new();
        q.execute_block(&x, &mut y).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn test_channel_multipath_matches_liquid() {
        // oracle: liquid channel_cccf with these taps on the signal above
        let h = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.5, -0.3),
            Complex32::new(-0.2, 0.1),
            Complex32::new(0.05, 0.07),
        ];
        let expected = [
            Complex32::new(1.000000000e+00, 0.000000000e+00),
            Complex32::new(1.452333570e+00, -1.505618691e-01),
            Complex32::new(1.123093963e+00, 1.460995376e-01),
        ];

        let x = test_signal(32);
        let mut y = vec![Complex32::new(0.0, 0.0); x.len()];
        let mut q = Channel::new();
        q.add_multipath(&h).unwrap();
        q.execute_block(&x, &mut y).unwrap();

        for (i, e) in expected.iter().enumerate() {
            assert_abs_diff_eq!(y[i].re, e.re, epsilon = 1e-6);
            assert_abs_diff_eq!(y[i].im, e.im, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_channel_random_taps_match_liquid() {
        // the "random" taps come from an m-sequence, so they are reproducible
        // oracle captured from liquid's generator
        let expected = [
            Complex32::new(1.0, 0.0),
            Complex32::new(-8.398437500e-02, -1.093750000e-01),
            Complex32::new(1.757812500e-02, 2.539062500e-02),
            Complex32::new(4.492187500e-02, -8.203125000e-02),
            Complex32::new(-2.050781250e-01, -6.640625000e-02),
            Complex32::new(-2.050781250e-01, 2.148437500e-02),
            Complex32::new(-1.093750000e-01, 2.421875000e-01),
            Complex32::new(1.796875000e-01, 1.875000000e-01),
        ];

        let mut q = Channel::new();
        q.add_multipath_random(expected.len()).unwrap();
        assert_eq!(q.get_multipath(), &expected);
    }

    #[test]
    fn test_channel_carrier_offset_is_a_rotation() {
        // a carrier offset must preserve magnitude and advance phase linearly
        let dphi = 0.03f32;
        let phi = 1.2f32;
        let n = 64;
        let x = vec![Complex32::new(1.0, 0.0); n];
        let mut y = vec![Complex32::new(0.0, 0.0); n];

        let mut q = Channel::new();
        q.add_carrier_offset(dphi, phi);
        q.execute_block(&x, &mut y).unwrap();

        for (i, v) in y.iter().enumerate() {
            assert_abs_diff_eq!(v.norm(), 1.0, epsilon = 1e-5);
            let want = phi + dphi * i as f32;
            let err = (v.arg() - want).sin(); // compare modulo 2*pi
            assert_abs_diff_eq!(err, 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_channel_awgn_gain_and_noise_power() {
        let noise_floor = -40.0f32;
        let snr = 20.0f32;
        let n = 200_000;

        let mut q = Channel::new_seeded(0x5eed);
        q.add_awgn(noise_floor, snr);

        // drive with zeros to isolate the noise term
        let zeros = vec![Complex32::new(0.0, 0.0); n];
        let mut y = vec![Complex32::new(0.0, 0.0); n];
        q.execute_block(&zeros, &mut y).unwrap();

        let noise_power: f32 = y.iter().map(|v| v.norm_sqr()).sum::<f32>() / n as f32;
        let expected_noise = 10.0f32.powf(noise_floor / 10.0);
        assert_abs_diff_eq!(noise_power, expected_noise, epsilon = 0.05 * expected_noise);

        // measured SNR of a unit-power signal should come back as `snr`
        let ones = vec![Complex32::new(1.0, 0.0); n];
        let mut y2 = vec![Complex32::new(0.0, 0.0); n];
        q.execute_block(&ones, &mut y2).unwrap();
        let gamma = 10.0f32.powf((snr + noise_floor) / 20.0);
        let signal_power = gamma * gamma;
        let snr_hat = 10.0 * (signal_power / noise_power).log10();
        assert_abs_diff_eq!(snr_hat, snr, epsilon = 0.5);
    }

    #[test]
    fn test_channel_seeded_is_reproducible() {
        let x = test_signal(64);
        let mut y0 = vec![Complex32::new(0.0, 0.0); x.len()];
        let mut y1 = vec![Complex32::new(0.0, 0.0); x.len()];

        for (seed, y) in [(7u64, &mut y0), (7u64, &mut y1)] {
            let mut q = Channel::new_seeded(seed);
            q.add_awgn(-30.0, 15.0);
            q.add_shadowing(1.0, 0.1).unwrap();
            q.execute_block(&x, y).unwrap();
        }
        assert_eq!(y0, y1);

        let mut y2 = vec![Complex32::new(0.0, 0.0); x.len()];
        let mut q = Channel::new_seeded(8);
        q.add_awgn(-30.0, 15.0);
        q.add_shadowing(1.0, 0.1).unwrap();
        q.execute_block(&x, &mut y2).unwrap();
        assert_ne!(y0, y2);
    }

    #[test]
    fn test_channel_shadowing_varies_slowly() {
        let n = 20_000;
        let x = vec![Complex32::new(1.0, 0.0); n];
        let mut y = vec![Complex32::new(0.0, 0.0); n];

        let mut q = Channel::new_seeded(42);
        q.add_shadowing(3.0, 0.01).unwrap();
        q.execute_block(&x, &mut y).unwrap();

        // gains in dB (input is unit magnitude, so |y| is the gain)
        let g: Vec<f32> = y.iter().map(|v| 20.0 * v.norm().log10()).collect();
        let spread = {
            let mean = g.iter().sum::<f32>() / n as f32;
            (g.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32).sqrt()
        };
        let step = {
            let d: Vec<f32> = g.windows(2).map(|w| w[1] - w[0]).collect();
            (d.iter().map(|v| v * v).sum::<f32>() / d.len() as f32).sqrt()
        };
        assert!(
            step < 0.5 * spread,
            "shadowing should be slow: step rms {step} vs spread {spread}"
        );
    }

    #[test]
    fn test_channel_rejects_invalid_config() {
        let mut q = Channel::new();
        assert!(q.add_multipath_random(0).is_err());
        assert!(q.add_multipath_random(MAX_MULTIPATH_LEN + 1).is_err());

        assert!(q.add_shadowing(0.0, 0.1).is_err());
        assert!(q.add_shadowing(-1.0, 0.1).is_err());
        assert!(q.add_shadowing(1.0, 0.0).is_err());
        assert!(q.add_shadowing(1.0, 0.5).is_err());

        // enabling shadowing twice is rejected
        assert!(q.add_shadowing(1.0, 0.1).is_ok());
        assert!(q.add_shadowing(1.0, 0.1).is_err());
    }

    #[test]
    fn test_channel_execute_block_length_mismatch() {
        let mut q = Channel::new();
        let x = vec![Complex32::new(0.0, 0.0); 8];
        let mut y = vec![Complex32::new(0.0, 0.0); 4];
        assert!(q.execute_block(&x, &mut y).is_err());
    }

    #[test]
    fn test_channel_reset_restores_initial_state() {
        let h = [
            Complex32::new(0.8, 0.1),
            Complex32::new(-0.4, 0.25),
            Complex32::new(0.15, -0.05),
        ];
        let x = test_signal(32);
        let mut y0 = vec![Complex32::new(0.0, 0.0); x.len()];
        let mut y1 = vec![Complex32::new(0.0, 0.0); x.len()];

        let mut q = Channel::new();
        q.add_carrier_offset(-0.017, -0.4);
        q.add_multipath(&h).unwrap();
        q.execute_block(&x, &mut y0).unwrap();
        q.reset();
        q.execute_block(&x, &mut y1).unwrap();
        assert_eq!(y0, y1);
    }
}