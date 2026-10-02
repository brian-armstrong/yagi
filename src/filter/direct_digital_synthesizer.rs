// direct digital synthesizer (up/down-converter)

use std::f32::consts::PI;

use num_complex::Complex32;

use crate::error::{Error, Result};
use crate::filter::fir::design::estimate_req_filter_len;
use crate::filter::HalfBandResampler;
use crate::nco::{Nco, NcoBackend};

/// Complex frequency converter that changes the sample rate by a power of two.
///
/// The converter pairs an oscillator with a chain of half-band filters. Each
/// stage doubles or halves the sample rate, so the rate changes by a factor of
/// `2^num_stages` between the low-rate and high-rate sides.
///
/// [`Self::interp_execute`] shifts a low-rate signal up to the center
/// frequency and raises its sample rate. [`Self::decim_execute`] does the
/// reverse. Both directions share the oscillator phase and filter history, so
/// use one direction per synthesizer.
///
/// The half-band stages use complex coefficients to select the shifted band
/// without passing its negative-frequency image.
#[derive(Clone, Debug)]
#[doc(alias = "Dds")]
pub struct DirectDigitalSynthesizer {
    // user-defined parameters
    num_stages: usize,

    // derived values
    rate: usize,

    // halfband decimation/interpolation stages
    halfband_resamp: Vec<HalfBandResampler<Complex32, Complex32>>,

    m: Vec<usize>,

    // internal buffers
    buffer0: Vec<Complex32>,
    buffer1: Vec<Complex32>,

    // low-rate mixing stage
    ncox: Nco,

    // down-converter scaling factor
    zeta: f32,
    scale: Complex32,
}

impl DirectDigitalSynthesizer {
    /// Creates a direct digital synthesizer.
    ///
    /// # Arguments
    ///
    /// - `num_stages`: number of half-band stages, at most 20. The rate
    ///   changes by `2^num_stages`.
    /// - `center_frequency`: in cycles per high-rate sample, in `[-0.5, 0.5]`.
    /// - `bandwidth`: relative to the low sample rate, in `(0, 1)`. It sets the
    ///   transition width of each half-band stage rather than a passband edge
    ///   of the final output.
    /// - `stopband_attenuation`: in dB. Must be finite and non-negative.
    ///
    /// With zero stages, the converter mixes and scales one sample without
    /// filtering or changing the sample rate.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_stages` is greater than 20,
    /// `center_frequency` or `bandwidth` is nonfinite or outside its range,
    /// `stopband_attenuation` is nonfinite or negative, or a half-band stage cannot be
    /// designed.
    pub fn new(num_stages: usize, center_frequency: f32, bandwidth: f32, stopband_attenuation: f32) -> Result<Self> {
        // error checking
        if num_stages > 20 {
            return Err(Error::Config(format!("number of stages {} exceeds reasonable maximum (20)", num_stages)));
        }
        if !(-0.5..=0.5).contains(&center_frequency) {
            return Err(Error::Config(format!("frequency {} is out of range [-0.5,0.5]", center_frequency)));
        }
        if !(bandwidth > 0.0 && bandwidth < 1.0) {
            return Err(Error::Config(format!("bandwidth {} is out of range (0,1)", bandwidth)));
        }
        if !stopband_attenuation.is_finite() || stopband_attenuation < 0.0 {
            return Err(Error::Config(format!(
                "stop-band suppression {} must be finite and non-negative",
                stopband_attenuation
            )));
        }

        let rate = 1 << num_stages;

        // allocate memory for filter properties
        let mut fc_vec = vec![0.0f32; num_stages];
        let mut ft_vec = vec![0.0f32; num_stages];
        let mut as_vec = vec![0.0f32; num_stages];
        let mut m_vec = vec![0usize; num_stages];

        let mut fc_current = 0.5 * (1 << num_stages) as f32 * center_frequency;
        let mut bw_current = bandwidth;

        // TODO : compute/set filter bandwidths, lengths appropriately
        for i in 0..num_stages {
            fc_vec[i] = fc_current;
            while fc_vec[i] > 0.5 {
                fc_vec[i] -= 1.0;
            }
            while fc_vec[i] < -0.5 {
                fc_vec[i] += 1.0;
            }

            // compute transition bandwidth
            ft_vec[i] = 0.5 * (1.0 - bw_current);
            if ft_vec[i] > 0.45 {
                ft_vec[i] = 0.45;
            }
            as_vec[i] = stopband_attenuation;

            // compute (estimate) required filter length
            m_vec[i] = estimate_req_filter_len(ft_vec[i], as_vec[i])?;

            // update carrier, bandwidth parameters
            fc_current *= 0.5;
            bw_current *= 0.5;
        }

        // allocate memory for buffering
        let buffer0 = vec![Complex32::new(0.0, 0.0); rate];
        let buffer1 = vec![Complex32::new(0.0, 0.0); rate];

        // allocate memory for resampler pointers and create objects
        let mut halfband_resamp = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            halfband_resamp.push(HalfBandResampler::new(m_vec[i], fc_vec[i], as_vec[i])?);
        }

        // set down-converter scaling factor
        let zeta = 1.0 / (rate as f32);
        let scale = Complex32::new(1.0, 0.0);

        // create NCO and set frequency
        let mut ncox = Nco::new(NcoBackend::InterpolatedLookupTable);
        // TODO : ensure range is in [-pi,pi]
        ncox.set_frequency(2.0 * PI * (rate as f32) * center_frequency);

        Ok(Self { num_stages, rate, halfband_resamp, m: m_vec, buffer0, buffer1, ncox, zeta, scale })
    }

    /// Clears the filter history and returns the oscillator phase to zero.
    ///
    /// The scale is unchanged.
    pub fn reset(&mut self) {
        // reset internal filter state variables
        for resamp in &mut self.halfband_resamp {
            resamp.reset();
        }
        self.ncox.set_phase(0.0);
    }

    /// Sets the complex factor applied to the signal in both directions.
    pub fn set_scale(&mut self, scale: Complex32) {
        self.scale = scale;
    }

    /// Returns the complex factor applied to the signal in both directions.
    pub fn scale(&self) -> Complex32 {
        self.scale
    }

    /// Returns the number of half-band stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Returns the ratio of the high sample rate to the low sample rate.
    ///
    /// This is `2^num_stages()`.
    pub fn rate(&self) -> usize {
        self.rate
    }

    /// Returns the delay through the up-converter, in high-rate samples.
    pub fn delay_interp(&self) -> usize {
        let mut delay = 0usize;
        for i in 0..self.num_stages {
            delay *= 2;
            delay += 2 * self.m[i];
        }
        delay
    }

    /// Returns the delay through the down-converter, in low-rate samples.
    pub fn delay_decim(&self) -> f32 {
        let mut delay = 0.0f32;
        for i in 0..self.num_stages {
            delay *= 0.5;
            delay += self.m[self.num_stages - i - 1] as f32 - 0.5;
        }
        delay
    }

    /// Down-converts [`Self::rate`] high-rate samples to one low-rate sample.
    ///
    /// The output is divided by [`Self::rate`] and then multiplied by the
    /// scale. Samples beyond the first [`Self::rate`] are ignored.
    ///
    /// With zero stages, `input[0]` is mixed down and scaled directly.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` is shorter than [`Self::rate`]. The
    /// oscillator and filter history are unchanged on error.
    pub fn decim_execute(&mut self, input: &[Complex32]) -> Result<Complex32> {
        if input.len() < self.rate {
            return Err(Error::Config("decimation input length too small for synthesizer rate".into()));
        }
        let mut b0 = &mut self.buffer0;
        let mut b1 = &mut self.buffer1;

        // iterate through each stage
        for s in 0..self.num_stages {
            let k = 1 << (self.num_stages - s - 1);
            let g = self.num_stages - s - 1;

            // first stage reads the input directly, later stages read the prior buffer
            let src: &[Complex32] = if s == 0 { input } else { b0 };

            for i in 0..k {
                b1[i] = self.halfband_resamp[g].decim_execute(&src[2 * i..2 * i + 2])?;
            }

            std::mem::swap(&mut b0, &mut b1);
        }

        // output value
        let y = if self.num_stages == 0 { input[0] } else { b0[0] };

        // increment NCO
        let y = self.ncox.mix_down(y);
        self.ncox.step();

        // set output, normalizing by scaling factor
        Ok(y * self.zeta * self.scale)
    }

    /// Up-converts one low-rate sample to [`Self::rate`] high-rate samples.
    ///
    /// The input is multiplied by the scale before mixing. `output` must hold
    /// at least [`Self::rate`] samples, and extra output is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than [`Self::rate`]. The
    /// oscillator, filter history and `output` are unchanged on error.
    pub fn interp_execute(&mut self, input: Complex32, output: &mut [Complex32]) -> Result<()> {
        if output.len() < self.rate {
            return Err(Error::Config("interpolation output length too small for synthesizer rate".into()));
        }
        // increment NCO
        let input = input * self.scale;
        let input = self.ncox.mix_up(input);
        self.ncox.step();

        let mut b0 = self.buffer0.as_mut_slice();
        let mut b1 = self.buffer1.as_mut_slice();

        b0[0] = input;

        // iterate through each stage
        for s in 0..self.num_stages {
            let k = 1 << s;

            for i in 0..k {
                self.halfband_resamp[s].interp_execute(b0[i], &mut b1[2 * i..2 * i + 2])?;
            }

            std::mem::swap(&mut b0, &mut b1);
        }

        // copy output data
        output[..self.rate].copy_from_slice(&b0[..self.rate]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    use crate::filter::fir::design::fir_design_kaiser;
    use crate::framing::ArbitraryRateSymbolStream;
    use crate::utility::test_helpers::{validate_psd_signal, PsdRegion};

    #[test]
    fn test_dds_rejects_nonfinite_design_parameters() {
        let outside = f32::from_bits(0.5f32.to_bits() + 1);
        for stages in [0, 1] {
            for frequency in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -outside, outside] {
                assert!(matches!(DirectDigitalSynthesizer::new(stages, frequency, 0.1, 60.0), Err(Error::Config(_))));
            }
            for bandwidth in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0] {
                assert!(matches!(DirectDigitalSynthesizer::new(stages, 0.0, bandwidth, 60.0), Err(Error::Config(_))));
            }
            for attenuation in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -f32::EPSILON] {
                assert!(matches!(DirectDigitalSynthesizer::new(stages, 0.0, 0.1, attenuation), Err(Error::Config(_))));
            }
        }
        for frequency in [-0.5, 0.0, 0.5] {
            assert!(DirectDigitalSynthesizer::new(1, frequency, 0.1, 60.0).is_ok());
            // No filter is designed with zero stages, so zero attenuation and
            // bandwidths immediately inside the open interval remain valid.
            for bandwidth in [f32::from_bits(1), f32::from_bits(1.0f32.to_bits() - 1)] {
                assert!(DirectDigitalSynthesizer::new(0, frequency, bandwidth, 0.0).is_ok());
            }
        }
    }

    #[test]
    fn test_dds_zero_stage_decimation() {
        let scale = Complex32::new(2.0, -0.5);
        let samples = [Complex32::new(1.0, -2.0), Complex32::new(-3.0, 0.5), Complex32::new(0.0, 0.0)];
        for fc in [0.0, 0.125, -0.2] {
            let mut q = DirectDigitalSynthesizer::new(0, fc, 0.1, 60.0).unwrap();
            q.set_scale(scale);
            for _ in 0..2 {
                for i in 0..24 {
                    let baseband = samples[i % samples.len()];
                    let carrier = Complex32::from_polar(1.0, 2.0 * PI * fc * i as f32);
                    let input = [baseband * carrier, Complex32::new(99.0, -99.0)];
                    let output = q.decim_execute(&input).unwrap();
                    let expected = baseband * scale;
                    approx::assert_abs_diff_eq!(output.re, expected.re, epsilon = 1e-4);
                    approx::assert_abs_diff_eq!(output.im, expected.im, epsilon = 1e-4);
                }
                q.reset();
            }
        }
    }

    #[test]
    fn test_dds_nonzero_center_rejects_image() {
        let rate = 4;
        let num_input = 512;
        let mut output = vec![Complex32::new(0.0, 0.0); rate * num_input];

        for center in [0.2f32, -0.2] {
            let mut q = DirectDigitalSynthesizer::new(2, center, 0.1, 60.0).unwrap();
            for chunk in output.chunks_exact_mut(rate) {
                q.interp_execute(Complex32::new(1.0, 0.0), chunk).unwrap();
            }

            let measure = |frequency: f32| {
                let settled = &output[output.len() / 2..];
                settled
                    .iter()
                    .enumerate()
                    .map(|(i, &sample)| sample * Complex32::from_polar(1.0, -2.0 * PI * frequency * i as f32))
                    .sum::<Complex32>()
                    / settled.len() as f32
            };

            let image = if center > 0.0 { center - 0.5 } else { center + 0.5 };
            assert!((measure(center).norm() - 1.0).abs() < 0.05);
            assert!(measure(image).norm() < 0.02);

            for (frequency, expected_magnitude) in [(center, 1.0), (image, 0.0)] {
                let mut down = DirectDigitalSynthesizer::new(2, center, 0.1, 60.0).unwrap();
                let mut sum = Complex32::new(0.0, 0.0);
                for i in 0..num_input {
                    let input: [Complex32; 4] = std::array::from_fn(|j| {
                        Complex32::from_polar(1.0, 2.0 * PI * frequency * (rate * i + j) as f32)
                    });
                    let sample = down.decim_execute(&input).unwrap();
                    if i >= num_input / 2 {
                        sum += sample;
                    }
                }
                let magnitude = (sum / (num_input / 2) as f32).norm();
                assert!((magnitude - expected_magnitude).abs() < 0.05);
            }
        }
    }

    fn testbench_dds_cccf(num_stages: usize, fc: f32, as_: f32) {
        let tol = 1.0f32; // error tolerance [dB]
        let bw = 0.1f32; // original pulse bandwidth
        let m = 40usize; // pulse semi-length
        let r = 1 << num_stages; // resampling rate (output/input)

        // create resampler
        let mut q = DirectDigitalSynthesizer::new(num_stages, fc, bw, as_).unwrap();
        q.set_scale(Complex32::new(1.0 / r as f32, 0.0));

        let delay_interp = q.delay_interp();
        let delay_decim = q.delay_decim();
        let h_len = 2 * r * m + 1; // pulse length
        let num_samples = h_len + delay_interp + delay_decim as usize + 8;

        let mut buf_0: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); num_samples]; // input
        let mut buf_1: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); num_samples * r]; // interpolated
        let mut buf_2: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); num_samples]; // decimated

        // generate the baseband signal (filter pulse)
        let w = 0.36 * bw; // pulse bandwidth
        let h = fir_design_kaiser(h_len, w, as_, 0.0).unwrap();
        for i in 0..num_samples {
            buf_0[i] = if i < h_len { Complex32::new(2.0 * w * h[i], 0.0) } else { Complex32::new(0.0, 0.0) };
        }

        // run interpolation (up-conversion) stage
        for i in 0..num_samples {
            q.interp_execute(buf_0[i], &mut buf_1[r * i..r * i + r]).unwrap();
        }

        // clear DDS object
        q.reset();
        q.set_scale(Complex32::new(r as f32, 0.0));

        // run decimation (down-conversion) stage
        for i in 0..num_samples {
            buf_2[i] = q.decim_execute(&buf_1[r * i..r * i + r]).unwrap();
        }

        // verify input spectrum
        let regions_orig = vec![
            PsdRegion { fmin: -0.5, fmax: -0.6 * bw, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.3 * bw, fmax: 0.3 * bw, pmin: -1.0, pmax: 1.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.6 * bw, fmax: 0.5, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signal(&buf_0, &regions_orig).unwrap());

        // verify interpolated spectrum
        let f1 = fc - 0.6 * bw / r as f32;
        let f2 = fc - 0.3 * bw / r as f32;
        let f3 = fc + 0.3 * bw / r as f32;
        let f4 = fc + 0.6 * bw / r as f32;
        let regions_interp = vec![
            PsdRegion { fmin: -0.5, fmax: f1, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: f2, fmax: f3, pmin: -1.0, pmax: 1.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: f4, fmax: 0.5, pmin: 0.0, pmax: -as_ + tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signal(&buf_1, &regions_interp).unwrap());

        // verify decimated spectrum (using same regions as original)
        assert!(validate_psd_signal(&buf_2, &regions_orig).unwrap());
    }

    // test different configurations
    #[test]
    #[autotest_annotate(autotest_dds_cccf_0)]
    fn test_dds_cccf_0() {
        testbench_dds_cccf(1, 0.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_dds_cccf_1)]
    fn test_dds_cccf_1() {
        testbench_dds_cccf(2, 0.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_dds_cccf_2)]
    fn test_dds_cccf_2() {
        testbench_dds_cccf(3, 0.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_dds_config)]
    fn test_dds_config() {
        // check that object returns None for invalid configurations
        assert!(DirectDigitalSynthesizer::new(50, 0.0, 0.1, 60.0).is_err()); // num stages out of range
        assert!(DirectDigitalSynthesizer::new(2, 0.7, 0.1, 60.0).is_err()); // fc out of range
        assert!(DirectDigitalSynthesizer::new(2, -0.7, 0.1, 60.0).is_err()); // fc out of range
        assert!(DirectDigitalSynthesizer::new(2, 0.2, 1.4, 60.0).is_err()); // bw out of range
        assert!(DirectDigitalSynthesizer::new(2, 0.2, -1.4, 60.0).is_err()); // bw out of range
        assert!(DirectDigitalSynthesizer::new(2, 0.2, 0.1, -1.0).is_err()); // as out of range

        // create proper object and test configurations
        let mut q = DirectDigitalSynthesizer::new(2, 0.0, 0.2, 60.0).unwrap();

        // test setting/getting properties
        q.set_scale(Complex32::new(2.0, -3.0));
        let scale = q.scale();
        assert_eq!(scale, Complex32::new(2.0, -3.0));
    }

    // copy object
    #[test]
    #[autotest_annotate(autotest_dds_copy)]
    fn test_dds_copy() {
        let num_stages = 3usize;
        let r = 1usize << num_stages; // resampling rate (input/output)

        // create resampler
        let mut q0 = DirectDigitalSynthesizer::new(num_stages, 0.1234, 0.4321, 60.0).unwrap();
        q0.set_scale(Complex32::new(0.72280, 0.0));

        // create generator with default parameters
        let mut gen = ArbitraryRateSymbolStream::new().unwrap();

        // generate samples and push through resampler
        let mut buf = vec![Complex32::new(0.0, 0.0); r];
        for _ in 0..10 {
            // generate block of samples
            gen.write_samples(&mut buf).unwrap();

            // resample
            let _ = q0.decim_execute(&buf).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run samples through both resamplers in parallel
        for _ in 0..60 {
            // generate block of samples
            gen.write_samples(&mut buf).unwrap();

            // resample
            let y0 = q0.decim_execute(&buf).unwrap();
            let y1 = q1.decim_execute(&buf).unwrap();

            // compare output
            assert_eq!(y0, y1);
        }
    }
}
