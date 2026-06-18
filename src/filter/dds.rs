// direct digital synthesizer (up/down-converter)

use std::f32::consts::PI;

use num_complex::Complex32;

use crate::error::{Error, Result};
use crate::filter::estimate_req_filter_len;
use crate::filter::resamp2::Resamp2;
use crate::nco::{Osc, OscScheme};

#[derive(Clone, Debug)]
pub struct Dds {
    // user-defined parameters
    num_stages: usize,

    // derived values
    rate: usize,

    // halfband decimation/interpolation stages
    halfband_resamp: Vec<Resamp2<Complex32, f32>>,

    m: Vec<usize>,

    // internal buffers
    buffer0: Vec<Complex32>,
    buffer1: Vec<Complex32>,

    // low-rate mixing stage
    ncox: Osc,

    // down-converter scaling factor
    zeta: f32,
    scale: Complex32,
}

impl Dds {
    pub fn new(num_stages: usize, fc: f32, bw: f32, as_: f32) -> Result<Self> {
        // error checking
        if num_stages > 20 {
            return Err(Error::Config(format!(
                "number of stages {} exceeds reasonable maximum (20)",
                num_stages
            )));
        }
        if fc > 0.5 || fc < -0.5 {
            return Err(Error::Config(format!(
                "frequency {} is out of range [-0.5,0.5]",
                fc
            )));
        }
        if bw <= 0.0 || bw >= 1.0 {
            return Err(Error::Config(format!(
                "bandwidth {} is out of range (0,1)",
                bw
            )));
        }
        if as_ < 0.0 {
            return Err(Error::Config(format!(
                "stop-band suppression {} must be greater than zero",
                as_
            )));
        }

        let rate = 1 << num_stages;

        // allocate memory for filter properties
        let mut fc_vec = vec![0.0f32; num_stages];
        let mut ft_vec = vec![0.0f32; num_stages];
        let mut as_vec = vec![0.0f32; num_stages];
        let mut m_vec = vec![0usize; num_stages];

        let mut fc_current = 0.5 * (1 << num_stages) as f32 * fc;
        let mut bw_current = bw;

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
            as_vec[i] = as_;

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
            halfband_resamp.push(Resamp2::new(m_vec[i], fc_vec[i], as_vec[i])?);
        }

        // set down-converter scaling factor
        let zeta = 1.0 / (rate as f32);
        let scale = Complex32::new(1.0, 0.0);

        // create NCO and set frequency
        let mut ncox = Osc::new(OscScheme::Vco);
        // TODO : ensure range is in [-pi,pi]
        ncox.set_frequency(2.0 * PI * (rate as f32) * fc);

        Ok(Self {
            num_stages,
            rate,
            halfband_resamp,
            m: m_vec,
            buffer0,
            buffer1,
            ncox,
            zeta,
            scale,
        })
    }

    pub fn reset(&mut self) {
        // reset internal filter state variables
        for resamp in &mut self.halfband_resamp {
            resamp.reset();
        }
        self.ncox.set_phase(0.0);
    }

    pub fn set_scale(&mut self, scale: Complex32) {
        self.scale = scale;
    }

    pub fn get_scale(&self) -> Complex32 {
        self.scale
    }

    pub fn get_num_stages(&self) -> usize {
        self.num_stages
    }

    pub fn get_rate(&self) -> usize {
        self.rate
    }

    pub fn get_delay_interp(&self) -> usize {
        let mut delay = 0usize;
        for i in 0..self.num_stages {
            delay *= 2;
            delay += 2 * self.m[i];
        }
        delay
    }

    pub fn get_delay_decim(&self) -> f32 {
        let mut delay = 0.0f32;
        for i in 0..self.num_stages {
            delay *= 0.5;
            delay += self.m[self.num_stages - i - 1] as f32 - 0.5;
        }
        delay
    }

    pub fn decim_execute(&mut self, x: &[Complex32]) -> Result<Complex32> {
        let mut b0 = &mut self.buffer0;
        let mut b1 = &mut self.buffer1;

        // iterate through each stage
        for s in 0..self.num_stages {
            let k = 1 << (self.num_stages - s - 1);
            let g = self.num_stages - s - 1;

            // first stage reads the input directly, later stages read the prior buffer
            let src: &[Complex32] = if s == 0 { x } else { b0 };

            for i in 0..k {
                b1[i] = self.halfband_resamp[g].decim_execute(&src[2 * i..2 * i + 2])?;
            }

            std::mem::swap(&mut b0, &mut b1);
        }

        // output value
        let y = b0[0];

        // increment NCO
        let y = self.ncox.mix_down(y);
        self.ncox.step();

        // set output, normalizing by scaling factor
        Ok(y * self.zeta * self.scale)
    }

    pub fn interp_execute(&mut self, x: Complex32, y: &mut [Complex32]) -> Result<()> {
        // increment NCO
        let x = x * self.scale;
        let x = self.ncox.mix_up(x);
        self.ncox.step();

        let mut b0 = self.buffer0.as_mut_slice();
        let mut b1 = self.buffer1.as_mut_slice();

        b0[0] = x;

        // iterate through each stage
        for s in 0..self.num_stages {
            let k = 1 << s;

            for i in 0..k {
                self.halfband_resamp[s].interp_execute(b0[i], &mut b1[2 * i..2 * i + 2])?;
            }

            std::mem::swap(&mut b0, &mut b1);
        }

        // copy output data
        y[..self.rate].copy_from_slice(&b0[..self.rate]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    use crate::filter::fir_design_kaiser;
    use crate::framing::symstreamr::SymStreamR;
    use crate::utility::test_helpers::{validate_psd_signal, PsdRegion};

    fn testbench_dds_cccf(num_stages: usize, fc: f32, as_: f32) {
        let tol = 1.0f32; // error tolerance [dB]
        let bw = 0.1f32; // original pulse bandwidth
        let m = 40usize; // pulse semi-length
        let r = 1 << num_stages; // resampling rate (output/input)

        // create resampler
        let mut q = Dds::new(num_stages, fc, bw, as_).unwrap();
        q.set_scale(Complex32::new(1.0 / r as f32, 0.0));

        let delay_interp = q.get_delay_interp();
        let delay_decim = q.get_delay_decim();
        let h_len = 2 * r * m + 1; // pulse length
        let num_samples = h_len + delay_interp + delay_decim as usize + 8;

        let mut buf_0: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); num_samples]; // input
        let mut buf_1: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); num_samples * r]; // interpolated
        let mut buf_2: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); num_samples]; // decimated

        // generate the baseband signal (filter pulse)
        let w = 0.36 * bw; // pulse bandwidth
        let h = fir_design_kaiser(h_len, w, as_, 0.0).unwrap();
        for i in 0..num_samples {
            buf_0[i] = if i < h_len {
                Complex32::new(2.0 * w * h[i], 0.0)
            } else {
                Complex32::new(0.0, 0.0)
            };
        }

        // run interpolation (up-conversion) stage
        for i in 0..num_samples {
            q.interp_execute(buf_0[i], &mut buf_1[r * i..r * i + r])
                .unwrap();
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
        assert!(Dds::new(50, 0.0, 0.1, 60.0).is_err()); // num stages out of range
        assert!(Dds::new(2, 0.7, 0.1, 60.0).is_err()); // fc out of range
        assert!(Dds::new(2, -0.7, 0.1, 60.0).is_err()); // fc out of range
        assert!(Dds::new(2, 0.2, 1.4, 60.0).is_err()); // bw out of range
        assert!(Dds::new(2, 0.2, -1.4, 60.0).is_err()); // bw out of range
        assert!(Dds::new(2, 0.2, 0.1, -1.0).is_err()); // as out of range

        // create proper object and test configurations
        let mut q = Dds::new(2, 0.0, 0.2, 60.0).unwrap();

        // test setting/getting properties
        q.set_scale(Complex32::new(2.0, -3.0));
        let scale = q.get_scale();
        assert_eq!(scale, Complex32::new(2.0, -3.0));
    }

    // copy object
    #[test]
    #[autotest_annotate(autotest_dds_copy)]
    fn test_dds_copy() {
        let num_stages = 3usize;
        let r = 1usize << num_stages; // resampling rate (input/output)

        // create resampler
        let mut q0 = Dds::new(num_stages, 0.1234, 0.4321, 60.0).unwrap();
        q0.set_scale(Complex32::new(0.72280, 0.0));

        // create generator with default parameters
        let mut gen = SymStreamR::new().unwrap();

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
