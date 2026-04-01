use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter::{self, FirPfbFilter, FirFilterShape};
use crate::filter::iir::IirFilterSos;
use num_complex::ComplexFloat;


#[derive(Clone, Debug)]
pub struct Symsync<T> {
    k: usize,           // samples/symbol (input)
    k_out: usize,       // samples/symbol (output)

    npfb: usize,        // number of filters in symsync
    mf: FirPfbFilter<T, f32>, // matched filter
    dmf: FirPfbFilter<T, f32>,// derivative matched filter
    b: usize,           // filterbank index
    bf: f32,            // filterbank index (fractional)
    tau: f32,           // fractional sample offset
    tau_decim: f32,     // fractional sample offset (decimated)

    rate: f32,          // internal resampling rate
    del: f32,           // fractional delay step

    q: f32,             // timing error
    q_hat: f32,         // filtered timing error
    decim_counter: usize, // decimation counter
    pll: IirFilterSos<f32>, // loop filter
    rate_adjustment: f32, // rate adjustment factor
    is_locked: bool,    // synchronizer locked flag
}

impl<T> Symsync<T>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + From<f32> + std::ops::Mul<f32, Output = T> + Default,
    [f32]: DotProd<T, Output = T>,
{
    pub fn new(k: usize, m: usize, h: &[f32], h_len: usize) -> Result<Self> {
        if k < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if m == 0 {
            return Err(Error::Config("number of filters must be greater than 0".into()));
        }
        if h_len == 0 {
            return Err(Error::Config("filter length must be greater than 0".into()));
        }
        if (h_len - 1) % m != 0 {
            return Err(Error::Config("filter length must be of the form: h_len = m*k + 1".into()));
        }

        // same as set_output_rate(1)
        let k_out = 1;
        let rate = k as f32 / k_out as f32;
        let del = rate;

        let npfb = m;

        let mut dh = vec![0.0f32; h_len];
        let mut hdh_max = 0.0;
        for i in 0..h_len {
            if i == 0 {
                dh[i] = h[i + 1] - h[h_len - 1];
            } else if i == h_len - 1 {
                dh[i] = h[0] - h[i - 1];
            } else {
                dh[i] = h[i + 1] - h[i - 1];
            }

            if (h[i] * dh[i]).abs() > hdh_max || i == 0 {
                hdh_max = (h[i] * dh[i]).abs();
            }
        }

        for i in 0..h_len {
            dh[i] *= 0.06f32 / hdh_max;
        }

        let mf = FirPfbFilter::new(npfb, h, h_len)?;
        let dmf = FirPfbFilter::new(npfb, &dh, h_len)?;

        let a_coeff = [1.0, 0.0, 0.0];
        let b_coeff = [0.0, 0.0, 0.0];
        let pll = IirFilterSos::new(&b_coeff, &a_coeff)?;

        let mut q = Self {
            k,
            k_out,
            npfb,
            mf,
            dmf,
            b: 0,
            bf: 0.0,
            tau: 0.0,
            tau_decim: 0.0,
            rate,
            del,
            q: 0.0,
            q_hat: 0.0,
            decim_counter: 0,
            pll,
            rate_adjustment: 0.0f32,
            is_locked: false,
        };

        q.reset();
        q.set_lf_bw(0.01f32)?;
        q.unlock();

        Ok(q)
    }

    pub fn new_rnyquist(ftype: FirFilterShape, k: usize, m: usize, beta: f32, num_filters: usize) -> Result<Self> {
        if k < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(Error::Config("excess bandwidth factor must be in [0,1]".into()));
        }
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than 0".into()));
        }

        let h_len = 2 * num_filters * k * m + 1;

        let h = filter::fir_design_prototype(ftype, k * num_filters, m, beta, 0.0)?;

        Self::new(k, num_filters, &h, h_len)
    }

    pub fn new_kaiser(k: usize, m: usize, beta: f32, num_filters: usize) -> Result<Self> {
        if k < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if beta <= 0.0 || beta > 1.0 {
            return Err(Error::Config("excess bandwidth factor must be in [0,1]".into()));
        }
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than 0".into()));
        }

        let h_len = 2 * num_filters * k * m + 1;

        let fc = 0.75f32;
        let as_ = 40.0f32;
        let mut h = filter::fir_design_kaiser(h_len, fc / (k as f32 * num_filters as f32), as_, 0.0)?;

        for c in h.iter_mut() {
            *c *= 2.0 * fc;
        }

        Self::new(k, num_filters, &h, h_len)
    }

    pub fn reset(&mut self) {
        self.mf.reset();
        self.rate = self.k as f32 / self.k_out as f32;
        self.del = self.rate;
        self.b = 0;
        self.bf = 0.0;
        self.tau = 0.0;
        self.tau_decim = 0.0;
        self.q = 0.0;
        self.q_hat = 0.0;
        self.decim_counter = 0;
        self.pll.reset();
    }

    pub fn lock(&mut self) {
        self.is_locked = true;
    }

    pub fn unlock(&mut self) {
        self.is_locked = false;
    }

    pub fn is_locked(&self) -> bool {
        self.is_locked
    }

    pub fn set_output_rate(&mut self, k_out: usize) -> Result<()> {
        if k_out == 0 {
            return Err(Error::Config("output rate must be greater than 0".into()));
        }
        self.k_out = k_out;
        self.rate = self.k as f32 / self.k_out as f32;
        self.del = self.rate;
        Ok(())
    }

    pub fn set_fractional_rate(&mut self, rate: f32) -> Result<()> {
        if rate <= 0.0 {
            return Err(Error::Config("rate must be greater than 0".into()));
        }
        self.rate = rate;
        self.del = self.rate;
        Ok(())
    }

    pub fn set_lf_bw(&mut self, _bandwidth: f32) -> Result<()> {
        if _bandwidth < 0.0 || _bandwidth > 1.0 {
            return Err(Error::Config("bandwidth must be in [0,1]".into()));
        }

        let alpha = 1.0 - _bandwidth;
        let beta = 0.22 * _bandwidth;
        let a = 0.5;
        let b = 0.495;

        let b_coeff = [beta, 0.0, 0.0];
        let a_coeff = [1.0 - a * alpha, -b * alpha, 0.0];

        self.pll.set_coefficients(&b_coeff, &a_coeff)?;

        self.rate_adjustment = 0.5 * _bandwidth;
        Ok(())
    }

    pub fn get_tau(&self) -> f32 {
        self.tau_decim
    }

    pub fn get_rate(&self) -> f32 {
        self.rate
    }

    pub fn get_del(&self) -> f32 {
        self.del
    }

    pub fn get_q(&self) -> f32 {
        self.q
    }

    pub fn get_q_hat(&self) -> f32 {
        self.q_hat
    }

    pub fn execute(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        let mut ny = 0;

        for &xi in x.iter() {
            let k = self.step(xi, &mut y[ny..])?;
            ny += k;
        }

        Ok(ny)
    }

    fn step(&mut self, x: T, y: &mut [T]) -> Result<usize> {
        self.mf.push(x);
        self.dmf.push(x);

        let mut mf: T;
        let mut dmf: T;
        let mut ny = 0;

        while self.b < self.npfb {
            mf = self.mf.execute(self.b)?;
            y[ny] = mf / (self.k as f32).into();

            if self.decim_counter == self.k_out {
                self.decim_counter = 0;

                if self.is_locked {
                    continue;
                }

                dmf = self.dmf.execute(self.b)?;
                self.advance_internal_loop(mf, dmf);
                self.tau_decim = self.tau;
            }

            self.decim_counter += 1;
            self.tau += self.del;
            self.bf = self.tau * self.npfb as f32;
            self.b = self.bf.round() as usize;
            ny += 1;
        }

        self.tau -= 1.0;
        self.bf -= self.npfb as f32;
        self.b -= self.npfb;

        Ok(ny)
    }

    fn advance_internal_loop(&mut self, mf: T, dmf: T) {
        self.q = (mf.conj() * dmf).re();
        self.q = self.q.clamp(-1.0, 1.0);

        self.q_hat = self.pll.execute(self.q);

        self.rate += self.rate_adjustment * self.q_hat;
        self.del = self.rate + self.q_hat;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use num_complex::Complex32;
    use crate::sequence::MSequence;
    use crate::filter::FirInterpolationFilter;
    use crate::filter::resampler::resamp::Resamp;
    use crate::random::randnf;

    #[test]
    #[autotest_annotate(autotest_symsync_copy)]
    fn test_symsync_copy() {
        // create base object
        let mut q0 = Symsync::<Complex32>::new_rnyquist(
            FirFilterShape::Arkaiser,
            5,
            7,
            0.25,
            64
        ).unwrap();
        q0.set_lf_bw(0.02).unwrap();

        // run samples through filter
        // NOTE: we don't care that the input is noise; just that both objects
        //       produce the same output
        let buf_len = 640;
        let mut buf = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); buf_len];

        for i in 0..buf_len {
            buf[i] = Complex32::new(randnf(), randnf());
        }

        let _nw_0 = q0.execute(&buf, &mut buf_0).unwrap();

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters and check equality
        for i in 0..buf_len {
            buf[i] = Complex32::new(randnf(), randnf());
        }

        let nw_0 = q0.execute(&buf, &mut buf_0).unwrap();
        let nw_1 = q1.execute(&buf, &mut buf_1).unwrap();

        // check that the same number of samples were written
        assert_eq!(nw_0, nw_1);

        // check output sample values
        assert_eq!(buf_0[..nw_0], buf_1[..nw_1]);

        // check other internal properties
        assert_eq!(q0.get_tau(), q1.get_tau());
    }

    // test errors and invalid configuration
    #[test]
    #[autotest_annotate(autotest_symsync_config)]
    fn test_symsync_config() {
        // test copying/creating invalid objects
        // assert!(Symsync::<Complex32>::copy(&None).is_err());

        assert!(Symsync::<Complex32>::new(0, 12, &[], 48).is_err()); // k is too small
        assert!(Symsync::<Complex32>::new(2, 0, &[], 48).is_err()); // M is too small
        assert!(Symsync::<Complex32>::new(2, 12, &[], 0).is_err()); // h_len is too small
        assert!(Symsync::<Complex32>::new(2, 12, &[], 47).is_err()); // h_len is not divisible by M

        assert!(Symsync::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 0, 12, 0.2, 48).is_err()); // k is too small
        assert!(Symsync::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 2, 0, 0.2, 48).is_err()); // m is too small
        assert!(Symsync::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 2, 12, 7.2, 48).is_err()); // beta is too large
        assert!(Symsync::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 2, 12, 0.2, 0).is_err()); // M is too small

        assert!(Symsync::<Complex32>::new_kaiser(0, 12, 0.2, 48).is_err()); // k is too small
        assert!(Symsync::<Complex32>::new_kaiser(2, 0, 0.2, 48).is_err()); // m is too small
        assert!(Symsync::<Complex32>::new_kaiser(2, 12, 7.2, 48).is_err()); // beta is too large
        assert!(Symsync::<Complex32>::new_kaiser(2, 12, 0.2, 0).is_err()); // M is too small

        // create valid object
        let mut q = Symsync::<Complex32>::new_kaiser(2, 12, 0.2, 48).unwrap();
        // assert!(q.print().is_ok());

        // check lock state
        q.lock();
        assert!(q.is_locked());
        q.unlock();
        assert!(!q.is_locked());

        // check invalid properties
        assert!(q.set_output_rate(0).is_err());
        assert!(q.set_lf_bw(-1.0).is_err());
    }

    fn symsync_crcf_test(method: &str, _k: usize, _m: usize, _beta: f32, _tau: f32, _rate: f32) {
        // options
        let tol: f32 = 0.2;    // error tolerance
        let k: usize = _k;      // samples/symbol (input)
        let m: usize = _m;      // filter delay (symbols)
        let beta: f32 = _beta;   // filter excess bandwidth factor
        let num_filters: usize = 32;       // number of filters in the bank

        let num_symbols_init: usize = 200;  // number of initial symbols
        let num_symbols_test: usize = 100;  // number of testing symbols

        // transmit filter type
        let ftype_tx = if method == "rnyquist" {
            FirFilterShape::Arkaiser
        } else {
            FirFilterShape::Kaiser
        };

        let bt: f32 = 0.02;               // loop filter bandwidth
        let mut tau: f32 = _tau;                // fractional symbol offset
        let rate: f32 = _rate;               // resampled rate

        // derived values
        let num_symbols = num_symbols_init + num_symbols_test;
        let num_samples = k * num_symbols;
        let num_samples_resamp = (num_samples as f32 * rate * 1.1).ceil() as u32 + 4;
        
        // compute delay
        while tau < 0.0 { tau += 1.0; }    // ensure positive tau
        let g = k as f32 * tau;                // number of samples offset
        let ds = g.floor() as i32;               // additional symbol delay
        let mut dt = g - ds as f32;     // fractional sample offset
        if dt > 0.5 {                // force dt to be in [0.5,0.5]
            dt -= 1.0;
        }

        // allocate arrays
        let mut s = vec![Complex32::new(0.0, 0.0); num_symbols as usize];       // data symbols
        let mut x = vec![Complex32::new(0.0, 0.0); num_samples as usize];       // interpolated samples
        let mut y = vec![Complex32::new(0.0, 0.0); num_samples_resamp as usize];// resampled data (resamp_crcf)
        let mut z = vec![Complex32::new(0.0, 0.0); num_symbols as usize + 64];  // synchronized symbols

        // generate pseudo-random QPSK symbols
        // NOTE: by using an m-sequence generator this sequence will be identical
        //       each time this test is run
        let mut ms = MSequence::create_default(10).unwrap();
        for i in 0..num_symbols as usize {
            let si = ms.generate_symbol(1);
            let sq = ms.generate_symbol(1);
            s[i] = Complex32::new(
                if si == 0 { 1.0 } else { -1.0 } * std::f32::consts::FRAC_1_SQRT_2,
                if sq == 0 { 1.0 } else { -1.0 } * std::f32::consts::FRAC_1_SQRT_2
            );
        }

        // 
        // create and run interpolator
        //

        // design interpolating filter
        let mut interp = FirInterpolationFilter::<Complex32>::new_prototype(ftype_tx, k as usize, m, beta, dt).unwrap();

        // interpolate block of samples
        interp.execute_block(&s[..num_symbols as usize], &mut x[..num_samples as usize]).unwrap();

        // 
        // run resampler
        //

        // create resampler
        let resamp_len = 10 * k; // resampling filter semi-length (filter delay)
        let resamp_bw = 0.45;        // resampling filter bandwidth
        let resamp_as = 60.0;        // resampling filter stop-band attenuation
        let resamp_npfb = 64;  // number of filters in bank
        let mut resamp = Resamp::<Complex32>::new(rate, resamp_len, resamp_bw, resamp_as, resamp_npfb).unwrap();

        // run resampler on block
        let ny = resamp.execute_block(&x[..num_samples as usize], &mut y[..]).unwrap();

        // 
        // create and run symbol synchronizer
        //

        // create symbol synchronizer
        let mut sync = if method == "rnyquist" {
            Symsync::<Complex32>::new_rnyquist(ftype_tx, k, m, beta, num_filters).unwrap()
        } else {
            Symsync::<Complex32>::new_kaiser(k, m, beta, num_filters).unwrap()
        };

        // set loop filter bandwidth
        sync.set_lf_bw(bt).unwrap();

        // execute on entire block of samples
        let nz = sync.execute(&y[..ny], &mut z).unwrap();

        // compute total delay through system
        // (initial filter + resampler + matched filter)
        let delay = m + 10 + m;

        // compare (and print) results
        for i in (nz - num_symbols_test as usize)..nz {
            // compute error
            let err = (z[i] - s[i - delay as usize]).norm();
            
            // assert that error is below tolerance
            assert!(err < tol, "Error {} exceeds tolerance {} at index {}", err, tol, i);
        }
    }

    // autotest scenarios (root-Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_0)]
    fn symsync_crcf_scenario_0() { symsync_crcf_test("rnyquist", 2, 7, 0.35,  0.00, 1.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_1)]
    fn symsync_crcf_scenario_1() { symsync_crcf_test("rnyquist", 2, 7, 0.35, -0.25, 1.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_2)]
    fn symsync_crcf_scenario_2() { symsync_crcf_test("rnyquist", 2, 7, 0.35, -0.25, 1.0001 ); }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_3)]
    fn symsync_crcf_scenario_3() { symsync_crcf_test("rnyquist", 2, 7, 0.35, -0.25, 0.9999 ); }

    // autotest scenarios (Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_4)]
    fn symsync_crcf_scenario_4() { symsync_crcf_test("nyquist", 2, 7, 0.35,  0.00, 1.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_5)]
    fn symsync_crcf_scenario_5() { symsync_crcf_test("nyquist", 2, 7, 0.35, -0.25, 1.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_6)]
    fn symsync_crcf_scenario_6() { symsync_crcf_test("nyquist", 2, 7, 0.35, -0.25, 1.0001 ); }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_7)]
    fn symsync_crcf_scenario_7() { symsync_crcf_test("nyquist", 2, 7, 0.35, -0.25, 0.9999 ); }

    fn symsync_rrrf_test(method: &str,
                         k: usize,
                         m: usize,
                         beta: f32,
                         tau: f32,
                         rate: f32,
                         expected_rate: f32)
    {
        // options
        let tol        = 0.20f32;   // error tolerance
        let num_filters= 32;       // number of filters in the bank

        let num_symbols_init = 400;  // number of initial symbols
        let num_symbols_test = 100;  // number of testing symbols

        // transmit filter type
        let ftype_tx = if method == "rnyquist" {
            FirFilterShape::Arkaiser
        } else {
            FirFilterShape::Kaiser
        };

        let bt    = 0.01f32;               // loop filter bandwidth
        let mut tau   = tau;                // fractional symbol offset
        let rate  = rate;               // resampled rate

        // derived values
        let num_symbols = num_symbols_init + num_symbols_test;
        let num_samples = k * num_symbols;
        let num_samples_resamp = (num_samples as f32 * rate * 1.1f32).ceil() as usize + 4;
        
        // compute delay
        while tau < 0.0 { tau += 1.0; }    // ensure positive tau
        let g = k as f32 * tau;                // number of samples offset
        let ds = g.floor() as i32;               // additional symbol delay
        let mut dt = g - ds as f32;     // fractional sample offset
        if dt > 0.5 {                // force dt to be in [0.5,0.5]
            dt -= 1.0;
        }

        // allocate arrays
        let mut s = vec![0.0f32; num_symbols];           // data symbols
        let mut x = vec![0.0f32; num_samples];           // interpolated samples
        let mut y = vec![0.0f32; num_samples_resamp];    // resampled data (resamp_rrrf)
        let mut z = vec![0.0f32; num_symbols + 64];      // synchronized symbols

        // generate pseudo-random BPSK symbols
        // NOTE: by using an m-sequence generator this sequence will be identical
        //       each time this test is run
        let mut ms = MSequence::create_default(10).unwrap();
        for i in 0..num_symbols {
            s[i] = if ms.generate_symbol(1) == 0 { 1.0 } else { -1.0 };
        }

        // 
        // create and run interpolator
        //

        // design interpolating filter
        let mut interp = FirInterpolationFilter::<f32>::new_prototype(ftype_tx, k as usize, m, beta, dt).unwrap();

        // interpolate block of samples
        interp.execute_block(&s[..num_symbols], &mut x).unwrap();

        // 
        // run resampler
        //

        // create resampler
        let resamp_len = 10 * k; // resampling filter semi-length (filter delay)
        let resamp_bw = 0.45f32;        // resampling filter bandwidth
        let resamp_as = 60.0f32;        // resampling filter stop-band attenuation
        let resamp_npfb = 64;  // number of filters in bank
        let mut resamp = Resamp::<f32>::new(rate, resamp_len, resamp_bw, resamp_as, resamp_npfb).unwrap();

        // run resampler on block
        let ny = resamp.execute_block(&x[..num_samples], &mut y).unwrap();

        // 
        // create and run symbol synchronizer
        //

        // create symbol synchronizer
        let mut sync = if method == "rnyquist" {
            Symsync::<f32>::new_rnyquist(ftype_tx, k, m, beta, num_filters).unwrap()
        } else {
            Symsync::<f32>::new_kaiser(k, m, beta, num_filters).unwrap()
        };

        // set loop filter bandwidth
        sync.set_lf_bw(bt).unwrap();

        // set fractional rate
        sync.set_fractional_rate(expected_rate).unwrap();

        // execute on entire block of samples
        let nz = sync.execute(&y[..ny], &mut z).unwrap();

        // compute total delay through system
        // (initial filter + resampler + matched filter)
        let delay = m + 10 + m;

        // compare (and print) results
        for i in (nz - num_symbols_test)..nz {
            // compute error
            let err = (z[i] - s[i - delay]).abs();
            
            // assert that error is below tolerance
            assert!(err < tol, "Error {} exceeds tolerance {}", err, tol);
        }
    }

    // autotest scenarios (root-Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_0)]
    fn symsync_rrrf_scenario_0() { symsync_rrrf_test("rnyquist", 2, 7, 0.35,  0.00, 1.0, 2.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_1)]
    fn symsync_rrrf_scenario_1() { symsync_rrrf_test("rnyquist", 2, 7, 0.35, -0.25, 1.0, 2.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_2)]
    fn symsync_rrrf_scenario_2() { symsync_rrrf_test("rnyquist", 2, 7, 0.35, -0.25, 1.0001, 2.0 ); }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_3)]
    fn symsync_rrrf_scenario_3() { symsync_rrrf_test("rnyquist", 2, 7, 0.35, -0.25, 0.9999, 2.0 ); }

    // autotest scenarios (Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_4)]
    fn symsync_rrrf_scenario_4() { symsync_rrrf_test("nyquist", 2, 7, 0.35,  0.00, 1.0, 2.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_5)]
    fn symsync_rrrf_scenario_5() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 1.0, 2.0    ); }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_6)]
    fn symsync_rrrf_scenario_6() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 1.0001, 2.0 ); }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_7)]
    fn symsync_rrrf_scenario_7() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.9999, 2.0 ); }

    #[test]
    fn symsync_rrrf_scenario_8() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.998, 1.996 ); }

    #[test]
    fn symsync_rrrf_scenario_9() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.998, 1.994 ); }

    #[test]
    fn symsync_rrrf_scenario_10() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.998, 1.998 ); }

    #[test]
    fn symsync_rrrf_scenario_11() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.99, 1.98 ); }

    #[test]
    fn symsync_rrrf_scenario_12() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.99, 1.981 ); }

    #[test]
    fn symsync_rrrf_scenario_13() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.99, 1.979 ); }

    #[test]
    fn symsync_rrrf_scenario_14() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.98, 1.96 ); }

    #[test]
    fn symsync_rrrf_scenario_15() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.98, 1.962 ); }

    #[test]
    fn symsync_rrrf_scenario_16() { symsync_rrrf_test("nyquist", 2, 7, 0.35, -0.25, 0.98, 1.958 ); }
}