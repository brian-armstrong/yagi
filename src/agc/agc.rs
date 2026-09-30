use crate::error::{Error, Result};
use num_complex::ComplexFloat;

// Default AGC loop bandwidth
const AGC_DEFAULT_BW: f32 = 1e-2;

/// Sample-by-sample automatic gain control.
///
/// The controller multiplies each input sample by a linear gain and adjusts
/// that gain so the smoothed output power moves toward one. A fixed output
/// scale, [`Self::scale`], is then applied while tracking is unlocked. The
/// usual workflow is to acquire a signal, then call [`Self::lock`] to hold
/// the gain for the rest of the burst.
///
/// The gain is capped at `1e6` (120 dB) during tracking. The received signal
/// strength, [`Self::rssi`], is derived from the gain rather than measured
/// from the input, so it follows the gain loop and lags a change in input
/// level.
///
/// The optional squelch reports threshold crossings through
/// [`Self::squelch_status`]. It does not mute the output.
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

/// State of the optional AGC squelch.
///
/// The state advances once per [`Agc::execute`] call while the AGC is
/// unlocked, and holds while it is locked. Enable it with
/// [`Agc::squelch_enable`] and read it with [`Agc::squelch_status`]. Squelch
/// does not change the returned samples.
///
/// From `Enabled`, a signal passes through `Rise` to `SignalHi`. When it
/// drops below the threshold it passes through `Fall` to `SignalLo`, and
/// after the timeout it reaches `Timeout` and returns to `Enabled`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AgcSquelchMode {
    /// Squelch tracking is disabled.
    Disabled,
    /// Squelch is waiting for RSSI to rise above the threshold.
    Enabled,
    /// RSSI crossed above the threshold on the most recent sample.
    ///
    /// Only `Enabled` leads here. A signal that returns during `Fall` or
    /// `SignalLo` goes straight to `SignalHi`, so `Rise` does not mark the
    /// start of every burst.
    Rise,
    /// RSSI remains above the threshold.
    SignalHi,
    /// RSSI crossed below the threshold on the most recent sample, after
    /// `Rise` or `SignalHi`.
    Fall,
    /// RSSI is below the threshold and the timeout is counting down.
    SignalLo,
    /// The timeout expired. The next unlocked sample returns to `Enabled`.
    Timeout,
}

impl<T> Agc<T>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + From<f32> + std::ops::Mul<f32, Output = T>,
{
    /// Creates an AGC with the default settings.
    ///
    /// The gain and output scale start at one and the bandwidth at `0.01`.
    /// Squelch starts disabled, with a threshold of 0 dB and a timeout of 100
    /// samples.
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

    /// Resets the gain to one and unlocks tracking.
    ///
    /// The smoothed power estimate also returns to one. The bandwidth, scale,
    /// and squelch threshold and timeout are unchanged. An enabled squelch
    /// returns to [`AgcSquelchMode::Enabled`].
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

    /// Applies the gain to one sample and updates the gain.
    ///
    /// The returned sample uses the gain from before the update. While
    /// unlocked, the AGC updates its gain and squelch state and returns the
    /// sample multiplied by [`Self::scale`]. While locked, the gain and
    /// squelch state hold, and the output is not multiplied by the scale. This
    /// never returns an error.
    ///
    /// # Panics
    ///
    /// Panics when the squelch timeout is zero, an unlocked call runs in
    /// [`AgcSquelchMode::SignalLo`], and overflow checks are enabled. Without
    /// overflow checks, the squelch timer wraps and `Timeout` is effectively
    /// never reported.
    pub fn execute(&mut self, input: T) -> Result<T> {
        let y = input * self.g;
        let y2 = (y * y.conj()).re();

        self.y2_prime = (1.0 - self.alpha) * self.y2_prime + self.alpha * y2;

        if self.is_locked {
            return Ok(y);
        }

        if self.y2_prime > 1e-6_f32 {
            self.g *= (-0.5 * self.alpha * self.y2_prime.ln()).exp();
        }

        self.g = self.g.min(1e6_f32);
        self.squelch_update_mode()?;

        Ok(y * self.scale)
    }

    /// Processes a block of samples and writes one output per input.
    ///
    /// The results match calling [`Self::execute`] on each sample. Extra
    /// output is left unchanged. Returns `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The AGC and
    /// `output` are unchanged on error.
    ///
    /// # Panics
    ///
    /// Panics under the same squelch timer condition as [`Self::execute`].
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output block length must be at least the input length".into()));
        }
        for (x_i, y_i) in input.iter().zip(output.iter_mut()) {
            *y_i = self.execute(*x_i)?;
        }
        Ok(input.len())
    }

    /// Processes a block of samples and discards the output.
    ///
    /// The gain and squelch state change as they would from calling
    /// [`Self::execute`] on each sample, which trains the gain before the
    /// output is needed. This never returns an error.
    ///
    /// # Panics
    ///
    /// Panics under the same squelch timer condition as [`Self::execute`].
    pub fn execute_input_block(&mut self, input: &[T]) -> Result<()> {
        for x_i in input.iter() {
            self.execute(*x_i)?;
        }
        Ok(())
    }

    /// Holds the gain and squelch state at their current values.
    ///
    /// The smoothed power estimate still updates while locked.
    ///
    /// # Warning
    ///
    /// Locked output is not multiplied by [`Self::scale`]. With a scale other
    /// than one, the output amplitude changes by a factor of `1 / scale` at
    /// the moment of locking, even though the gain is held.
    pub fn lock(&mut self) {
        self.is_locked = true;
    }

    /// Resumes gain and squelch tracking.
    pub fn unlock(&mut self) {
        self.is_locked = false;
    }

    /// Returns whether the gain is held by [`Self::lock`].
    pub fn is_locked(&self) -> bool {
        self.is_locked
    }

    /// Returns the gain-tracking bandwidth.
    pub fn bandwidth(&self) -> f32 {
        self.bandwidth
    }

    /// Sets the gain-tracking bandwidth.
    ///
    /// The bandwidth is the per-sample smoothing coefficient of both the
    /// power estimate and the gain update. Larger values adapt more quickly
    /// and smooth the measured power less. Zero freezes the gain.
    ///
    /// # Errors
    ///
    /// Returns an error when `bandwidth` is outside `[0, 1]`. The bandwidth
    /// is unchanged on error.
    pub fn set_bandwidth(&mut self, bandwidth: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&bandwidth) {
            return Err(Error::Config("bandwidth must be in [0, 1]".into()));
        }
        self.bandwidth = bandwidth;
        self.alpha = bandwidth;
        Ok(())
    }

    /// Returns the estimated input magnitude, which is the reciprocal of the
    /// gain.
    pub fn signal_level(&self) -> f32 {
        1.0 / self.g
    }

    /// Sets the estimated input magnitude.
    ///
    /// The level is linear. The gain becomes its reciprocal, and the smoothed
    /// power estimate returns to one.
    ///
    /// # Errors
    ///
    /// Returns an error when `signal_level` is zero or negative. The AGC is
    /// unchanged on error.
    pub fn set_signal_level(&mut self, signal_level: f32) -> Result<()> {
        if signal_level <= 0.0 {
            return Err(Error::Config("signal level must be greater than zero".into()));
        }
        self.g = 1.0 / signal_level;
        self.y2_prime = 1.0;
        Ok(())
    }

    /// Returns the received signal strength in dB relative to a unit input
    /// magnitude.
    ///
    /// The value is `-20 * log10(gain)`. Tracking caps the gain at `1e6`, so
    /// the RSSI is at least -120 dB after an unlocked sample. The gain setters
    /// and [`Self::init`] can set a lower value, which holds until the next
    /// unlocked sample.
    pub fn rssi(&self) -> f32 {
        -20.0 * self.g.log10()
    }

    /// Sets the received signal strength in dB relative to a unit input
    /// magnitude.
    ///
    /// The gain becomes `max(10^(-rssi / 20), 1e-16)`, and the smoothed power
    /// estimate returns to one. This never returns an error.
    pub fn set_rssi(&mut self, rssi: f32) -> Result<()> {
        self.g = 10.0f32.powf(-rssi / 20.0);
        self.g = self.g.max(1e-16);
        self.y2_prime = 1.0;
        Ok(())
    }

    /// Returns the linear gain applied to the next input sample.
    pub fn gain(&self) -> f32 {
        self.g
    }

    /// Sets the linear gain applied to the next input sample.
    ///
    /// Unlike [`Self::set_signal_level`], this leaves the smoothed power
    /// estimate unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `gain` is zero or negative. The gain is unchanged
    /// on error.
    pub fn set_gain(&mut self, gain: f32) -> Result<()> {
        if gain <= 0.0 {
            return Err(Error::Config("gain must be greater than zero".into()));
        }
        self.g = gain;
        Ok(())
    }

    /// Returns the linear scale applied to unlocked output.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Sets the linear scale applied to unlocked output.
    ///
    /// The AGC drives the unscaled output toward unit magnitude, so the
    /// scaled output settles near `scale`. Locked output is not scaled. See
    /// [`Self::lock`].
    ///
    /// # Errors
    ///
    /// Returns an error when `scale` is zero or negative. The scale is
    /// unchanged on error.
    pub fn set_scale(&mut self, scale: f32) -> Result<()> {
        if scale <= 0.0 {
            return Err(Error::Config("scale must be greater than zero".into()));
        }
        self.scale = scale;
        Ok(())
    }

    /// Sets the initial gain from the RMS magnitude of `x`.
    ///
    /// The signal level becomes the RMS magnitude plus `1e-16`, so an
    /// all-zero block still gives a finite gain. The smoothed power estimate
    /// returns to one. The bandwidth, scale, lock state and squelch state are
    /// unchanged.
    ///
    /// The squared magnitudes are summed in `f32`, which overflows once the
    /// sum passes about `3.4e38`, as it does for a single sample of `1e20`.
    /// The gain then becomes zero and stays there, so prescale a high-level
    /// block before calling this.
    ///
    /// # Errors
    ///
    /// Returns an error when `x` is empty. The AGC is unchanged on error.
    pub fn init(&mut self, x: &[T]) -> Result<()> {
        if x.is_empty() {
            return Err(Error::Config("number of samples must be greater than zero".into()));
        }
        let x2: f32 = x.iter().map(|&xi| (xi * xi.conj()).re()).sum::<f32>();
        let x2 = (x2 / x.len() as f32).sqrt() + 1e-16;
        self.set_signal_level(x2)
    }

    /// Enables the squelch and sets its state to
    /// [`AgcSquelchMode::Enabled`].
    pub fn squelch_enable(&mut self) {
        self.squelch_mode = AgcSquelchMode::Enabled;
    }

    /// Disables the squelch.
    pub fn squelch_disable(&mut self) {
        self.squelch_mode = AgcSquelchMode::Disabled;
    }

    /// Returns whether the squelch is enabled.
    pub fn squelch_is_enabled(&self) -> bool {
        self.squelch_mode != AgcSquelchMode::Disabled
    }

    /// Sets the squelch threshold, in dB on the same scale as [`Self::rssi`].
    ///
    /// The signal counts as present when the RSSI is strictly greater than
    /// the threshold. Because the RSSI follows the gain loop, a crossing lags
    /// the change in input level. For a small bandwidth the lag is on the
    /// order of `1 / bandwidth` samples, so allow for it when choosing the
    /// threshold and timeout.
    pub fn squelch_set_threshold(&mut self, threshold: f32) {
        self.squelch_threshold = threshold;
    }

    /// Returns the squelch threshold in dB.
    pub fn squelch_threshold(&self) -> f32 {
        self.squelch_threshold
    }

    /// Sets how many samples the squelch stays in
    /// [`AgcSquelchMode::SignalLo`] before [`AgcSquelchMode::Timeout`].
    ///
    /// A signal that returns on the sample that uses up the timer still gives
    /// `Timeout`. Use a nonzero timeout, since zero makes [`Self::execute`]
    /// panic in `SignalLo` when overflow checks are enabled.
    pub fn squelch_set_timeout(&mut self, timeout: usize) {
        self.squelch_timeout = timeout;
    }

    /// Returns the squelch timeout in samples.
    pub fn squelch_timeout(&self) -> usize {
        self.squelch_timeout
    }

    /// Returns the current squelch state.
    pub fn squelch_status(&self) -> AgcSquelchMode {
        self.squelch_mode
    }

    fn squelch_update_mode(&mut self) -> Result<()> {
        let threshold_exceeded = self.rssi() > self.squelch_threshold;

        self.squelch_mode = match self.squelch_mode {
            AgcSquelchMode::Enabled => {
                if threshold_exceeded {
                    AgcSquelchMode::Rise
                } else {
                    AgcSquelchMode::Enabled
                }
            }
            AgcSquelchMode::Rise => {
                if threshold_exceeded {
                    AgcSquelchMode::SignalHi
                } else {
                    AgcSquelchMode::Fall
                }
            }
            AgcSquelchMode::SignalHi => {
                if threshold_exceeded {
                    AgcSquelchMode::SignalHi
                } else {
                    AgcSquelchMode::Fall
                }
            }
            AgcSquelchMode::Fall => {
                self.squelch_timer = self.squelch_timeout;
                if threshold_exceeded {
                    AgcSquelchMode::SignalHi
                } else {
                    AgcSquelchMode::SignalLo
                }
            }
            AgcSquelchMode::SignalLo => {
                self.squelch_timer -= 1;
                if self.squelch_timer == 0 {
                    AgcSquelchMode::Timeout
                } else if threshold_exceeded {
                    AgcSquelchMode::SignalHi
                } else {
                    AgcSquelchMode::SignalLo
                }
            }
            AgcSquelchMode::Timeout => AgcSquelchMode::Enabled,
            AgcSquelchMode::Disabled => AgcSquelchMode::Disabled,
        };

        Ok(())
    }
}

impl<T> Default for Agc<T>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + From<f32> + std::ops::Mul<f32, Output = T>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use crate::utility::test_helpers::validate_block_execute;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_agc_crcf_dc_gain_control)]
    fn test_agc_crcf_dc_gain_control() {
        // set parameters
        let gamma = 0.1f32; // nominal signal level
        let bt = 0.1f32; // bandwidth-time product
        let tol = 0.001f32; // error tolerance

        // create AGC object and initialize
        let mut q = Agc::<Complex32>::new();
        q.set_bandwidth(bt).unwrap();

        let x = num_complex::Complex32::new(gamma, 0.0); // input sample
        let mut y = num_complex::Complex32::new(0.0, 0.0); // output sample
        for _ in 0..256 {
            y = q.execute(x).unwrap();
        }

        // Check results
        assert_abs_diff_eq!(y.re, 1.0f32, epsilon = tol);
        assert_abs_diff_eq!(y.im, 0.0f32, epsilon = tol);
        assert_abs_diff_eq!(q.gain(), 1.0f32 / gamma, epsilon = tol);

        // explicitly set gain and check result
        q.set_gain(1.0f32).unwrap();
        assert_eq!(q.gain(), 1.0f32);

        // AGC object will be automatically destroyed when it goes out of scope
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_scale)]
    fn test_agc_crcf_scale() {
        // set parameters
        let scale = 4.0f32; // output scale (independent of AGC loop)
        let tol = 0.001f32; // error tolerance

        // create AGC object and initialize
        let mut q = Agc::new();
        q.set_bandwidth(0.1f32).unwrap();
        q.set_scale(scale).unwrap();
        assert_eq!(q.scale(), scale);

        let x = Complex32::new(0.1f32, 0.0); // input sample
        let mut y = Complex32::new(0.0, 0.0); // output sample
        for _ in 0..256 {
            y = q.execute(x).unwrap();
        }

        // Check results
        assert_abs_diff_eq!(y.re, scale, epsilon = tol);
        assert_abs_diff_eq!(y.im, 0.0f32, epsilon = tol);
    }

    // Test AC gain control
    #[test]
    #[autotest_annotate(autotest_agc_crcf_ac_gain_control)]
    fn test_agc_crcf_ac_gain_control() {
        // set parameters
        let gamma = 0.1f32; // nominal signal level
        let bt = 0.1f32; // bandwidth-time product
        let tol = 0.001f32; // error tolerance
        let dphi = 0.1f32; // NCO frequency

        // create AGC object and initialize
        let mut q = Agc::new();
        q.set_bandwidth(bt).unwrap();

        let mut x;
        for i in 0..256 {
            x = gamma * Complex32::from_polar(1.0, i as f32 * dphi);
            let _y = q.execute(x).unwrap();
        }

        if cfg!(test) {
            println!("gamma : {:.8}, rssi : {:.8}", gamma, q.signal_level());
        }

        // Check results
        assert_abs_diff_eq!(q.gain(), 1.0f32 / gamma, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_rssi_sinusoid)]
    fn test_agc_crcf_rssi_sinusoid() {
        // set parameters
        let gamma = 0.3f32; // nominal signal level
        let bt = 0.05f32; // agc bandwidth
        let tol = 0.001f32; // error tolerance

        // signal properties
        let dphi = 0.1f32; // signal frequency

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
        let rssi = q.signal_level();

        if cfg!(test) {
            println!("gamma : {:.8}, rssi : {:.8}", gamma, rssi);
        }

        // Check results
        assert_abs_diff_eq!(rssi, gamma, epsilon = tol);
    }

    // Test RSSI on noise input
    #[test]
    #[autotest_annotate(autotest_agc_crcf_rssi_noise)]
    fn test_agc_crcf_rssi_noise() {
        use std::f32::consts::FRAC_1_SQRT_2;

        // set parameters
        let gamma = -30.0f32; // nominal signal level [dB]
        let bt = 2e-3f32; // agc bandwidth
        let tol = 1.0f32; // error tolerance [dB]

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
        let rssi = q.rssi();

        if cfg!(test) {
            println!("gamma : {:.8}, rssi : {:.8}", gamma, rssi);
        }

        // Check results
        assert_abs_diff_eq!(rssi, gamma, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_agc_crcf_squelch)]
    fn test_agc_crcf_squelch() {
        use std::f32::consts::PI;

        // create agc object, set loop bandwidth, and initialize parameters
        let mut q = Agc::<Complex32>::new();
        q.set_bandwidth(0.25).unwrap();
        q.set_signal_level(1e-3f32).unwrap(); // initial guess at starting signal level

        // initialize squelch functionality
        assert!(!q.squelch_is_enabled());
        q.squelch_enable(); // enable squelch
        q.squelch_set_threshold(-50.0); // threshold for detection [dB]
        q.squelch_set_timeout(100); // timeout for hysteresis
        assert!(q.squelch_is_enabled());
        assert_eq!(q.squelch_threshold(), -50.0);
        assert_eq!(q.squelch_timeout(), 100);

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
            let mode = q.squelch_status();

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
        let gamma = 0.1f32; // nominal signal level
        let tol = 0.01f32; // error tolerance

        // create AGC object and initialize buffers for block processing
        let mut q = Agc::<Complex32>::new();
        q.set_bandwidth(0.1).unwrap();
        let buf_0 = vec![Complex32::new(gamma, 0.0); 4];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); 4];

        // basic tests
        assert_abs_diff_eq!(q.bandwidth(), 0.1f32);
        // assert!(q.print().is_ok());
        q.set_rssi(0.0).unwrap();

        // lock AGC and show it is not tracking
        assert_abs_diff_eq!(q.rssi(), 0.0, epsilon = tol); // base signal level is 0 dB
        assert!(!q.is_locked()); // not locked
        q.lock();
        assert!(q.is_locked()); // locked
        for _ in 0..256 {
            q.execute_block(&buf_0, &mut buf_1).unwrap();
        }
        assert_abs_diff_eq!(q.rssi(), 0.0, epsilon = tol); // signal level has not changed

        // unlock AGC and show it is tracking
        q.unlock();
        assert!(!q.is_locked()); // unlocked
        q.init(&buf_0).unwrap();
        // agc tracks to signal level
        assert_abs_diff_eq!(q.rssi(), 20.0 * gamma.log10(), epsilon = tol);
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

    #[test]
    fn test_agc_execute_block_contract() {
        let q = Agc::<f32>::new();
        validate_block_execute(&q, &[0.25, -0.5, 1.0], -9.0, Agc::execute_block);
    }
}
