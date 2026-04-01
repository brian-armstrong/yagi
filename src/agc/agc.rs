use crate::error::{Error, Result};
use num_complex::ComplexFloat;

// Default AGC loop bandwidth
const AGC_DEFAULT_BW: f32 = 1e-2;

#[derive(Clone, Debug)]
pub struct Agc<T> {
    g: f32,
    scale: f32,
    bandwidth: f32,
    alpha: f32,
    y2_prime: f32,
    is_locked: bool,
    squelch_mode: AgcSquelchMode,
    squelch_threshold: f32,
    squelch_timeout: usize,
    squelch_timer: usize,
    _p: std::marker::PhantomData<T>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AgcSquelchMode {
    Disabled,
    Enabled,
    Rise,
    SignalHi,
    Fall,
    SignalLo,
    Timeout,
}

impl<T> Agc<T>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + From<f32> + std::ops::Mul<f32, Output = T>,
{
    pub fn new() -> Self {
        let mut agc = Self {
            g: 0.0,
            scale: 0.0,
            bandwidth: 0.0,
            alpha: 0.0,
            y2_prime: 0.0,
            is_locked: false,
            squelch_mode: AgcSquelchMode::Disabled,
            squelch_threshold: 0.0,
            squelch_timeout: 0,
            squelch_timer: 0,
            _p: std::marker::PhantomData,
        };
        agc.set_bandwidth(AGC_DEFAULT_BW).unwrap();
        agc.reset();
        agc.squelch_disable();
        agc.squelch_set_threshold(0.0);
        agc.squelch_set_timeout(100);
        agc.set_scale(1.0).unwrap();
        agc
    }

    pub fn reset(&mut self) {
        self.g = 1.0;
        self.y2_prime = 1.0;
        self.unlock();
        self.squelch_mode = if self.squelch_mode == AgcSquelchMode::Disabled {
            AgcSquelchMode::Disabled
        } else {
            AgcSquelchMode::Enabled
        };
    }

    pub fn execute(&mut self, x: T) -> Result<T> {
        let y = x * self.g;
        let y2 = (y * y.conj()).re();

        self.y2_prime = (1.0 - self.alpha) * self.y2_prime + self.alpha * y2;

        if self.is_locked {
            return Ok(y);
        }

        if self.y2_prime > 1e-6.into() {
            self.g *= (-0.5 * self.alpha * self.y2_prime.ln()).exp();
        }

        self.g = self.g.min(1e6.into());
        self.squelch_update_mode()?;

        Ok(y * self.scale)
    }

    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        for (x_i, y_i) in x.iter().zip(y.iter_mut()) {
            *y_i = self.execute(*x_i)?;
        }
        Ok(())
    }

    pub fn execute_input_block(&mut self, x: &[T]) -> Result<()> {
        for x_i in x.iter() {
            self.execute(*x_i)?;
        }
        Ok(())
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

    pub fn get_bandwidth(&self) -> f32 {
        self.bandwidth
    }

    pub fn set_bandwidth(&mut self, bt: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&bt) {
            return Err(Error::Config("bandwidth must be in [0, 1]".into()));
        }
        self.bandwidth = bt;
        self.alpha = bt;
        Ok(())
    }

    pub fn get_signal_level(&self) -> f32 {
        1.0 / self.g
    }

    pub fn set_signal_level(&mut self, x2: f32) -> Result<()> {
        if x2 <= 0.0 {
            return Err(Error::Config("signal level must be greater than zero".into()));
        }
        self.g = 1.0 / x2;
        self.y2_prime = 1.0;
        Ok(())
    }

    pub fn get_rssi(&self) -> f32 {
        -20.0 * self.g.log10()
    }

    pub fn set_rssi(&mut self, rssi: f32) -> Result<()> {
        self.g = 10.0f32.powf(-rssi / 20.0);
        self.g = self.g.max(1e-16);
        self.y2_prime = 1.0;
        Ok(())
    }

    pub fn get_gain(&self) -> f32 {
        self.g
    }

    pub fn set_gain(&mut self, gain: f32) -> Result<()> {
        if gain <= 0.0 {
            return Err(Error::Config("gain must be greater than zero".into()));
        }
        self.g = gain;
        Ok(())
    }

    pub fn get_scale(&self) -> f32 {
        self.scale
    }

    pub fn set_scale(&mut self, scale: f32) -> Result<()> {
        if scale <= 0.0 {
            return Err(Error::Config("scale must be greater than zero".into()));
        }
        self.scale = scale;
        Ok(())
    }

    pub fn init(&mut self, x: &[T]) -> Result<()> {
        if x.is_empty() {
            return Err(Error::Config("number of samples must be greater than zero".into()));
        }
        let x2: f32 = x.iter().map(|&xi| (xi * xi.conj()).re()).sum::<f32>();
        let x2 = (x2 / x.len() as f32).sqrt() + 1e-16;
        self.set_signal_level(x2)
    }

    pub fn squelch_enable(&mut self) {
        self.squelch_mode = AgcSquelchMode::Enabled;
    }

    pub fn squelch_disable(&mut self) {
        self.squelch_mode = AgcSquelchMode::Disabled;
    }

    pub fn squelch_is_enabled(&self) -> bool {
        self.squelch_mode != AgcSquelchMode::Disabled
    }

    pub fn squelch_set_threshold(&mut self, threshold: f32) {
        self.squelch_threshold = threshold;
    }

    pub fn squelch_get_threshold(&self) -> f32 {
        self.squelch_threshold
    }

    pub fn squelch_set_timeout(&mut self, timeout: usize) {
        self.squelch_timeout = timeout;
    }

    pub fn squelch_get_timeout(&self) -> usize {
        self.squelch_timeout
    }

    pub fn squelch_get_status(&self) -> AgcSquelchMode {
        self.squelch_mode
    }

    fn squelch_update_mode(&mut self) -> Result<()> {
        let threshold_exceeded = self.get_rssi() > self.squelch_threshold;

        self.squelch_mode = match self.squelch_mode {
            AgcSquelchMode::Enabled => {
                if threshold_exceeded { AgcSquelchMode::Rise } else { AgcSquelchMode::Enabled }
            },
            AgcSquelchMode::Rise => {
                if threshold_exceeded { AgcSquelchMode::SignalHi } else { AgcSquelchMode::Fall }
            },
            AgcSquelchMode::SignalHi => {
                if threshold_exceeded { AgcSquelchMode::SignalHi } else { AgcSquelchMode::Fall }
            },
            AgcSquelchMode::Fall => {
                self.squelch_timer = self.squelch_timeout;
                if threshold_exceeded {
                    AgcSquelchMode::SignalHi
                } else {
                    AgcSquelchMode::SignalLo
                }
            },
            AgcSquelchMode::SignalLo => {
                self.squelch_timer -= 1;
                if self.squelch_timer == 0 {
                    AgcSquelchMode::Timeout
                } else if threshold_exceeded {
                    AgcSquelchMode::SignalHi
                } else {
                    AgcSquelchMode::SignalLo
                }
            },
            AgcSquelchMode::Timeout => AgcSquelchMode::Enabled,
            AgcSquelchMode::Disabled => AgcSquelchMode::Disabled,
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;
    use num_complex::Complex32;

    #[test]
    #[autotest_annotate(autotest_agc_crcf_dc_gain_control)]
    fn test_agc_crcf_dc_gain_control() {
        // set parameters
        let gamma = 0.1f32;     // nominal signal level
        let bt    = 0.1f32;     // bandwidth-time product
        let tol   = 0.001f32;   // error tolerance

        // create AGC object and initialize
        let mut q = Agc::<Complex32>::new();
        q.set_bandwidth(bt).unwrap();

        let x = num_complex::Complex32::new(gamma, 0.0);    // input sample
        let mut y = num_complex::Complex32::new(0.0, 0.0);  // output sample
        for _ in 0..256 {
            y = q.execute(x).unwrap();
        }
        
        // Check results
        assert_relative_eq!(y.re, 1.0f32, epsilon = tol);
        assert_relative_eq!(y.im, 0.0f32, epsilon = tol);
        assert_relative_eq!(q.get_gain(), 1.0f32 / gamma, epsilon = tol);

        // explicitly set gain and check result
        q.set_gain(1.0f32).unwrap();
        assert_eq!(q.get_gain(), 1.0f32);

        // AGC object will be automatically destroyed when it goes out of scope
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_scale)]
    fn test_agc_crcf_scale() {
        // set parameters
        let scale = 4.0f32;     // output scale (independent of AGC loop)
        let tol   = 0.001f32;   // error tolerance

        // create AGC object and initialize
        let mut q = Agc::new();
        q.set_bandwidth(0.1f32).unwrap();
        q.set_scale(scale).unwrap();
        assert_eq!(q.get_scale(), scale);

        let x = Complex32::new(0.1f32, 0.0); // input sample
        let mut y = Complex32::new(0.0, 0.0); // output sample
        for _ in 0..256 {
            y = q.execute(x).unwrap();
        }
        
        // Check results
        assert_relative_eq!(y.re, scale, epsilon = tol);
        assert_relative_eq!(y.im, 0.0f32, epsilon = tol);
    }

    // Test AC gain control
    #[test]
    #[autotest_annotate(autotest_agc_crcf_ac_gain_control)]
    fn test_agc_crcf_ac_gain_control() {
        // set parameters
        let gamma = 0.1f32;             // nominal signal level
        let bt    = 0.1f32;             // bandwidth-time product
        let tol   = 0.001f32;           // error tolerance
        let dphi  = 0.1f32;             // NCO frequency

        // create AGC object and initialize
        let mut q = Agc::new();
        q.set_bandwidth(bt).unwrap();

        let mut x;
        for i in 0..256 {
            x = gamma * Complex32::from_polar(1.0, i as f32 * dphi);
            let _y = q.execute(x).unwrap();
        }

        if cfg!(test) {
            println!("gamma : {:.8}, rssi : {:.8}", gamma, q.get_signal_level());
        }

        // Check results
        assert_relative_eq!(q.get_gain(), 1.0f32 / gamma, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_rssi_sinusoid)]
    fn test_agc_crcf_rssi_sinusoid() {
        // set parameters
        let gamma = 0.3f32;         // nominal signal level
        let bt    = 0.05f32;        // agc bandwidth
        let tol   = 0.001f32;       // error tolerance

        // signal properties
        let dphi = 0.1f32;          // signal frequency

        // create AGC object and initialize
        let mut q = Agc::new();
        q.set_bandwidth(bt).unwrap();

        for i in 0..512 {
            // generate sample (complex sinusoid)
            let x = gamma * Complex32::from_polar(1.0, dphi * i as f32);

            // execute agc
            let _y = q.execute(x).unwrap();
        }

        // get received signal strength indication
        let rssi = q.get_signal_level();

        if cfg!(test) {
            println!("gamma : {:.8}, rssi : {:.8}", gamma, rssi);
        }

        // Check results
        assert_relative_eq!(rssi, gamma, epsilon = tol);
    }

    // Test RSSI on noise input
    #[test]
    #[autotest_annotate(autotest_agc_crcf_rssi_noise)]
    fn test_agc_crcf_rssi_noise() {
        use std::f32::consts::FRAC_1_SQRT_2;

        // set parameters
        let gamma = -30.0f32;   // nominal signal level [dB]
        let bt    =  2e-3f32;   // agc bandwidth
        let tol   =  1.0f32;    // error tolerance [dB]

        // signal properties
        let nstd = 10f32.powf(gamma / 20.0);

        // create AGC object and initialize
        let mut q = Agc::<Complex32>::new();
        q.set_bandwidth(bt).unwrap();

        for _ in 0..8000 {
            // generate sample (circular complex noise)
            let x = nstd * Complex32::new(randnf(), randnf()) * FRAC_1_SQRT_2;

            // execute agc
            let _y = q.execute(x).unwrap();
        }

        // get received signal strength indication
        let rssi = q.get_rssi();

        if cfg!(test) {
            println!("gamma : {:.8}, rssi : {:.8}", gamma, rssi);
        }

        // Check results
        assert_relative_eq!(rssi, gamma, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_squelch)]
    fn test_agc_crcf_squelch() {
        use std::f32::consts::PI;

        // create agc object, set loop bandwidth, and initialize parameters
        let mut q = Agc::<Complex32>::new();
        q.set_bandwidth(0.25).unwrap();
        q.set_signal_level(1e-3f32).unwrap();     // initial guess at starting signal level

        // initialize squelch functionality
        assert!(!q.squelch_is_enabled());
        q.squelch_enable();             // enable squelch
        q.squelch_set_threshold(-50.0); // threshold for detection [dB]
        q.squelch_set_timeout(100);     // timeout for hysteresis
        assert!(q.squelch_is_enabled());
        assert_eq!(q.squelch_get_threshold(), -50.0);
        assert_eq!(q.squelch_get_timeout(), 100);

        // run agc
        let num_samples = 2000; // total number of samples to run
        for i in 0..num_samples {
            // generate signal, applying tapering window appropriately
            let gamma = if i < 500 {
                1e-3f32
            } else if i < 550 {
                1e-3f32 + (1e-2f32 - 1e-3f32) * (0.5f32 - 0.5f32 * ((PI * (i - 500) as f32) / 50.0f32).cos())
            } else if i < 1450 {
                1e-2f32
            } else if i < 1500 {
                1e-3f32 + (1e-2f32 - 1e-3f32) * (0.5f32 + 0.5f32 * ((PI * (i - 1450) as f32) / 50.0f32).cos())
            } else {
                1e-3f32
            };
            let x = gamma * Complex32::from_polar(1.0, 2.0 * PI * 0.0193f32 * i as f32);

            // apply gain
            let _y = q.execute(x).unwrap();

            // get squelch mode
            let mode = q.squelch_get_status();

            // check certain conditions based on sample input (assuming 2000 samples)
            match i {
                0 => assert_eq!(mode, AgcSquelchMode::Enabled),
                500 => assert_eq!(mode, AgcSquelchMode::Enabled),
                600 => assert_eq!(mode, AgcSquelchMode::SignalHi),
                1400 => assert_eq!(mode, AgcSquelchMode::SignalHi),
                1500 => assert_eq!(mode, AgcSquelchMode::SignalLo),
                1600 => assert_eq!(mode, AgcSquelchMode::Enabled),
                1900 => assert_eq!(mode, AgcSquelchMode::Enabled),
                _ => {}
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_lock)]
    fn test_agc_lock() {
        // set parameters
        let gamma = 0.1f32;     // nominal signal level
        let tol   = 0.01f32;    // error tolerance

        // create AGC object and initialize buffers for block processing
        let mut q = Agc::<Complex32>::new();
        q.set_bandwidth(0.1).unwrap();
        let buf_0 = vec![Complex32::new(gamma, 0.0); 4];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); 4];

        // basic tests
        assert_relative_eq!(q.get_bandwidth(), 0.1f32);
        // assert!(q.print().is_ok());
        q.set_rssi(0.0).unwrap();

        // lock AGC and show it is not tracking
        assert_relative_eq!(q.get_rssi(), 0.0, max_relative = tol);  // base signal level is 0 dB
        assert!(!q.is_locked());         // not locked
        q.lock();
        assert!(q.is_locked());          // locked
        for _ in 0..256 {
            q.execute_block(&buf_0, &mut buf_1).unwrap();
        }
        assert_relative_eq!(q.get_rssi(), 0.0, max_relative = tol);  // signal level has not changed

        // unlock AGC and show it is tracking
        q.unlock();
        assert!(!q.is_locked());         // unlocked
        q.init(&buf_0).unwrap();
        // agc tracks to signal level
        assert_relative_eq!(q.get_rssi(), 20.0 * gamma.log10(), epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_invalid_config)]
    fn test_agc_invalid_config() {
        // create main object and check invalid configurations
        let mut q = Agc::<Complex32>::new();

        // invalid bandwidths
        assert!(q.set_bandwidth(-1.0).is_err());
        assert!(q.set_bandwidth(2.0).is_err());

        // invalid gains
        assert!(q.set_gain(0.0).is_err());
        assert!(q.set_gain(-1.0).is_err());

        // invalid signal levels
        assert!(q.set_signal_level(0.0).is_err());
        assert!(q.set_signal_level(-1.0).is_err());

        // invalid scale values
        assert!(q.set_scale(0.0).is_err());
        assert!(q.set_scale(-1.0).is_err());

        // initialize gain on input array, but array has length 0
        let empty: Vec<Complex32> = Vec::new();
        assert!(q.init(&empty).is_err());
    }

    // copy test
    #[test]
    #[autotest_annotate(autotest_agc_crcf_copy)]
    fn test_agc_copy() {
        // create base object and initialize
        let mut q0 = Agc::<Complex32>::new();
        q0.set_bandwidth(0.01234).unwrap();

        // start running input through AGC
        let n = 32;
        for _ in 0..n {
            let x = Complex32::new(randnf(), randnf());
            let _y = q0.execute(x).unwrap();
        }

        // copy AGC
        let mut q1 = q0.clone();

        // continue running through both AGCs
        for _ in 0..n {
            // run AGCs in parallel
            let x = Complex32::new(randnf(), randnf());
            let y0 = q0.execute(x).unwrap();
            let y1 = q1.execute(x).unwrap();
            assert_eq!(y0, y1);
        }
    }
}