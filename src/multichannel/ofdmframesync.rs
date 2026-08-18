//
// ofdmframesync.rs
//
// OFDM frame synchronizer
//

use crate::buffer::Window;
use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::math::{poly_fit, poly_val};
use crate::matrix::matrix_linsolve;
use crate::multichannel::ofdmframe::{
    ofdmframe_init_s0, ofdmframe_init_s1, OfdmFrameConfig, SubcarrierType,
};
use crate::nco::{unwrap_phase, Osc, OscScheme};
use crate::sequence::MSequence;
use num_complex::Complex32;

/// channel estimator used at acquisition
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqGainMethod {
    /// fit |H| and arg(H) separately with a polynomial of this order.
    /// cheap, and marginally the best option on a nearly-flat channel,
    /// but it cannot represent more than roughly order/2 cycles of
    /// ripple across the band
    Poly { order: usize },
    /// fit this many time-domain taps by least squares. exact whenever the tap
    /// count covers the channel's delay spread, at the cost of admitting more
    /// noise per tap.
    Dft { num_taps: usize },
}

/// Runtime state for the acquisition channel estimator.
enum EqGainState {
    Poly { order: usize },
    Dft(DftEqGainState),
}

impl EqGainState {
    fn method(&self) -> EqGainMethod {
        match self {
            Self::Poly { order } => EqGainMethod::Poly { order: *order },
            Self::Dft(state) => EqGainMethod::Dft { num_taps: state.num_taps },
        }
    }

    fn estimate(&mut self, allocation: &[SubcarrierType], gain: &mut [Complex32]) -> Result<()> {
        match self {
            Self::Poly { order } => Self::estimate_poly(allocation, gain, *order),
            Self::Dft(state) => state.estimate(gain),
        }
    }

    /// Estimate complex equalizer gain using separate polynomial fits for
    /// magnitude and unwrapped phase.
    fn estimate_poly(
        allocation: &[SubcarrierType],
        gain: &mut [Complex32],
        order: usize,
    ) -> Result<()> {
        let m = gain.len();
        let m2 = m / 2;
        let n = allocation.iter().filter(|&&p| p != SubcarrierType::Null).count();
        let order = order.min(n - 1).min(10);

        let mut x_freq = vec![0.0f32; n];
        let mut y_abs = vec![0.0f32; n];
        let mut y_arg = vec![0.0f32; n];
        let mut p_abs = vec![0.0f32; order + 1];
        let mut p_arg = vec![0.0f32; order + 1];

        let mut idx = 0;
        for i in 0..m {
            // start at mid-point (effective fftshift)
            let k = (i + m2) % m;

            if allocation[k] != SubcarrierType::Null {
                if idx == n {
                    return Err(Error::Internal(
                        "ofdmframesync_estimate_eqgain_poly(), pilot subcarrier mismatch".into(),
                    ));
                }
                x_freq[idx] = if k > m2 { k as f32 - m as f32 } else { k as f32 };
                x_freq[idx] /= m as f32;
                y_abs[idx] = gain[k].norm();
                y_arg[idx] = gain[k].arg();
                idx += 1;
            }
        }

        if idx != n {
            return Err(Error::Internal(
                "ofdmframesync_estimate_eqgain_poly(), pilot subcarrier mismatch".into(),
            ));
        }

        unwrap_phase(&mut y_arg);
        poly_fit(&x_freq, &y_abs, n, &mut p_abs, order + 1)?;
        poly_fit(&x_freq, &y_arg, n, &mut p_arg, order + 1)?;

        for i in 0..m {
            let mut freq = if i > m2 { i as f32 - m as f32 } else { i as f32 };
            freq /= m as f32;
            let a = poly_val(&p_abs, order + 1, freq);
            let theta = poly_val(&p_arg, order + 1, freq);
            gain[i] = if allocation[i] == SubcarrierType::Null {
                Complex32::new(0.0, 0.0)
            } else {
                Complex32::from_polar(a, theta)
            };
        }

        Ok(())
    }
}

/// Persistent state for the time-domain channel fit.
struct DftEqGainState {
    num_taps: usize,
    active: Vec<usize>,
    basis: Vec<Complex32>,
    gram: Vec<Complex32>,
    rhs: Vec<Complex32>,
    taps: Vec<Complex32>,
    solve_scratch: Vec<Complex32>,
}

impl DftEqGainState {
    /// diagonal loading for the time-domain channel fit, as a fraction of the
    /// number of observations.
    const EQGAIN_RIDGE: f32 = 1e-3;

    /// number of *negative*-delay taps the time-domain channel fit reserves.
    const EQGAIN_LEAD_TAPS: usize = 2;

    fn new(num_subcarriers: usize, num_taps: usize, allocation: &[SubcarrierType]) -> Self {
        let active = allocation
            .iter()
            .enumerate()
            .filter_map(|(k, &p)| (p != SubcarrierType::Null).then_some(k))
            .collect();

        // basis[t * M + k] = exp(-j 2 pi k * delay(t) / M), where the
        // tap delays run from -lead upward so a slightly early sampling
        // instant stays representable. always preserve at least one tap
        // for positive delay
        let lead = Self::EQGAIN_LEAD_TAPS.min(num_subcarriers).min(num_taps - 1);
        let mut basis = vec![Complex32::new(0.0, 0.0); num_taps * num_subcarriers];
        for t in 0..num_taps {
            let delay = (t + num_subcarriers - lead) % num_subcarriers;
            for k in 0..num_subcarriers {
                let theta = -2.0 * std::f32::consts::PI * (k as f32) * (delay as f32)
                    / (num_subcarriers as f32);
                basis[t * num_subcarriers + k] = Complex32::from_polar(1.0, theta);
            }
        }

        Self {
            num_taps,
            active,
            basis,
            gram: vec![Complex32::new(0.0, 0.0); num_taps * num_taps],
            rhs: vec![Complex32::new(0.0, 0.0); num_taps],
            taps: vec![Complex32::new(0.0, 0.0); num_taps],
            solve_scratch: vec![Complex32::new(0.0, 0.0); num_taps * (num_taps + 1)],
        }
    }

    /// Fit time-domain taps to the channel response on the active carriers.
    fn estimate(&mut self, gain: &mut [Complex32]) -> Result<()> {
        // fit a finite-length channel impulse response using selected columns of the
        // DFT matrix, then reconstruct the per-subcarrier channel response.
        // van de Beek et al., "On Channel Estimation in OFDM Systems," VTC 1995
        let m = gain.len();
        let n = self.active.len();
        let taps = self.num_taps;

        debug_assert_eq!(self.basis.len(), taps * m);
        debug_assert_eq!(self.gram.len(), taps * taps);
        debug_assert_eq!(self.rhs.len(), taps);
        debug_assert_eq!(self.taps.len(), taps);
        debug_assert_eq!(self.solve_scratch.len(), taps * (taps + 1));

        self.gram.fill(Complex32::new(0.0, 0.0));
        self.rhs.fill(Complex32::new(0.0, 0.0));

        for &k in &self.active {
            let gk = gain[k];
            for r in 0..taps {
                let ar = self.basis[r * m + k];
                self.rhs[r] += ar.conj() * gk;
                for c in 0..taps {
                    self.gram[r * taps + c] += ar.conj() * self.basis[c * m + k];
                }
            }
        }

        // the guard bands leave gaps in k, so the highest taps can be poorly
        // observed. this bounds the solution rather than letting an
        // ill-conditioned system amplify noise.
        let ridge = Self::EQGAIN_RIDGE * n as f32;
        for r in 0..taps {
            self.gram[r * taps + r] += Complex32::new(ridge, 0.0);
        }

        matrix_linsolve(&mut self.gram, taps, &self.rhs, &mut self.taps, Some(&mut self.solve_scratch))?;

        gain.fill(Complex32::new(0.0, 0.0));
        for &k in &self.active {
            for t in 0..taps {
                gain[k] += self.taps[t] * self.basis[t * m + k];
            }
        }

        Ok(())
    }

    fn default_num_taps(cp_len: usize, num_active: usize) -> usize {
        cp_len.max(2).min(num_active)
    }
}

/// Sample buffering, frequency correction, and transform scratch shared by
/// preamble acquisition and payload reception.
struct OfdmRxFrontend {
    fft: Fft<f32>,
    x_freq: Vec<Complex32>,
    x_time: Vec<Complex32>,
    input_buffer: Window<Complex32>,
    nco: Osc,
}

impl OfdmRxFrontend {
    fn new(config: &OfdmFrameConfig) -> Result<Self> {
        let m = config.num_subcarriers();
        Ok(Self {
            fft: Fft::new(m, Direction::Forward),
            x_freq: vec![Complex32::new(0.0, 0.0); m],
            x_time: vec![Complex32::new(0.0, 0.0); m],
            input_buffer: Window::new(config.symbol_len())?,
            nco: Osc::new(OscScheme::Nco),
        })
    }

    fn reset(&mut self) {
        self.nco.reset();
    }
}

/// State of the S0/S1 preamble acquisition procedure.
#[derive(Clone, Copy, Debug)]
enum AcquisitionStage {
    SeekPlcp { timer: usize },
    PlcpShort0 { timer: usize },
    PlcpShort1 { timer: usize, s0_metric: Complex32 },
    PlcpLong { timer: usize, attempts: usize, previous_half_detected: bool },
}

enum AcquisitionStatus {
    Pending,
    Acquired { payload_timer: usize, backoff: usize },
}

/// Detection, timing/CFO acquisition, and initial channel estimation from the
/// complete S0/S1 preamble.
struct OfdmFrameAcquisition {
    stage: AcquisitionStage,
    m_s0: usize,
    m_s1: usize,
    s0_freq: Vec<Complex32>,
    s0_time: Vec<Complex32>,
    s1_freq: Vec<Complex32>,
    g0: f32,
    g0a: Vec<Complex32>,
    g0b: Vec<Complex32>,
    gain: Vec<Complex32>,
    backoff: usize,
    backoff_phase: Vec<Complex32>,
    detect_threshold: f32,
    sync_threshold: f32,
    eqgain: EqGainState,
}

impl OfdmFrameAcquisition {
    fn new(config: &OfdmFrameConfig) -> Result<Self> {
        let m = config.num_subcarriers();
        let cp_len = config.cp_len();
        let allocation = config.allocation();
        let counts = config.counts();

        let mut s0_freq = vec![Complex32::new(0.0, 0.0); m];
        let mut s0_time = vec![Complex32::new(0.0, 0.0); m];
        let mut s1_freq = vec![Complex32::new(0.0, 0.0); m];
        let mut s1_time = vec![Complex32::new(0.0, 0.0); m];
        let m_s0 = ofdmframe_init_s0(allocation, &mut s0_freq, &mut s0_time)?;
        let m_s1 = ofdmframe_init_s1(allocation, &mut s1_freq, &mut s1_time)?;

        let backoff = cp_len.min(2);
        let phi = backoff as f32 * 2.0 * std::f32::consts::PI / m as f32;
        let backoff_phase =
            (0..m).map(|i| Complex32::from_polar(1.0, i as f32 * phi)).collect();
        let active = counts.pilot + counts.data;
        let taps = DftEqGainState::default_num_taps(cp_len, active);

        Ok(Self {
            stage: AcquisitionStage::SeekPlcp { timer: 0 },
            m_s0,
            m_s1,
            s0_freq,
            s0_time,
            s1_freq,
            g0: 1.0,
            g0a: vec![Complex32::new(0.0, 0.0); m],
            g0b: vec![Complex32::new(0.0, 0.0); m],
            gain: vec![Complex32::new(0.0, 0.0); m],
            backoff,
            backoff_phase,
            detect_threshold: if m > 44 { 0.35 } else { 0.35 + 0.01 * (44 - m) as f32 },
            sync_threshold: if m > 44 { 0.30 } else { 0.30 + 0.01 * (44 - m) as f32 },
            eqgain: EqGainState::Dft(DftEqGainState::new(m, taps, allocation)),
        })
    }

    fn reset(&mut self) {
        self.stage = AcquisitionStage::SeekPlcp { timer: 0 };
    }

    fn is_frame_open(&self) -> bool {
        !matches!(self.stage, AcquisitionStage::SeekPlcp { .. })
    }

    fn should_mix_down(&self) -> bool {
        self.is_frame_open()
    }

    fn get_rssi(&self) -> f32 {
        // TODO this should be recomputed during other parts of acquisition, not just seekplcp
        -10.0 * self.g0.log10()
    }

    fn set_eqgain_method(&mut self, config: &OfdmFrameConfig, method: EqGainMethod) -> Result<()> {
        let m = config.num_subcarriers();
        let counts = config.counts();
        let active = counts.pilot + counts.data;

        self.eqgain = match method {
            EqGainMethod::Poly { order } => {
                if order == 0 {
                    return Err(Error::Config(
                        "ofdmframesync, polynomial order must be at least 1".into(),
                    ));
                }
                EqGainState::Poly { order }
            }
            EqGainMethod::Dft { num_taps } => {
                if num_taps == 0 {
                    return Err(Error::Config(
                        "ofdmframesync, tap count must be at least 1".into(),
                    ));
                }
                if num_taps > active || num_taps > m {
                    return Err(Error::Config(format!(
                        "ofdmframesync, tap count {} exceeds {} usable subcarriers",
                        num_taps,
                        active.min(m)
                    )));
                }
                EqGainState::Dft(DftEqGainState::new(m, num_taps, config.allocation()))
            }
        };
        Ok(())
    }

    fn eqgain_method(&self) -> EqGainMethod {
        self.eqgain.method()
    }

    fn execute(
        &mut self,
        config: &OfdmFrameConfig,
        frontend: &mut OfdmRxFrontend,
    ) -> Result<AcquisitionStatus> {
        match self.stage {
            AcquisitionStage::SeekPlcp { .. } => self.execute_seekplcp(config, frontend),
            AcquisitionStage::PlcpShort0 { .. } => self.execute_s0a(config, frontend),
            AcquisitionStage::PlcpShort1 { .. } => self.execute_s0b(config, frontend),
            AcquisitionStage::PlcpLong { .. } => self.execute_s1(config, frontend),
        }
    }

    fn execute_seekplcp(
        &mut self,
        config: &OfdmFrameConfig,
        frontend: &mut OfdmRxFrontend,
    ) -> Result<AcquisitionStatus> {
        let AcquisitionStage::SeekPlcp { mut timer } = self.stage else {
            unreachable!("seek handler called outside seek state")
        };

        timer += 1;

        if timer < config.num_subcarriers() {
            self.stage = AcquisitionStage::SeekPlcp { timer };
            return Ok(AcquisitionStatus::Pending);
        }


        let m = config.num_subcarriers();
        let cp_len = config.cp_len();
        timer = 0;

        // estimate gain
        let rc = frontend.input_buffer.read();
        // avoid division by 0
        let mut g = 1.0e-9f32;
        for sample in &rc[cp_len..m + cp_len] {
            g += sample.re * sample.re + sample.im * sample.im;
        }
        g = m as f32 / g;

        // estimate s0 gain
        frontend.x_time.copy_from_slice(&rc[cp_len..cp_len + m]);
        self.estimate_gain_s0(config, frontend, 0);

        let mut s_hat = self.s0_metrics(config, 0);
        s_hat *= g;
        let tau_hat = s_hat.arg() * (m / 2) as f32 / (2.0 * std::f32::consts::PI);
        self.g0 = g;

        if s_hat.norm() > self.detect_threshold {
            // set timer appropriately
            let dt = tau_hat.round() as i32;
            timer = ((m as i32 + dt) as usize) % (m / 2);
            timer += m; // add delay to help ensure good S0 estimate
            self.stage = AcquisitionStage::PlcpShort0 { timer };
        } else {
            self.stage = AcquisitionStage::SeekPlcp { timer };
        }
        Ok(AcquisitionStatus::Pending)
    }

    fn execute_s0a(
        &mut self,
        config: &OfdmFrameConfig,
        frontend: &mut OfdmRxFrontend,
    ) -> Result<AcquisitionStatus> {
        let AcquisitionStage::PlcpShort0 { mut timer } = self.stage else {
            unreachable!("S0a handler called outside S0a state")
        };

        timer += 1;

        if timer < config.num_subcarriers() / 2 {
            self.stage = AcquisitionStage::PlcpShort0 { timer };
            return Ok(AcquisitionStatus::Pending);
        }

        let m = config.num_subcarriers();
        let cp_len = config.cp_len();
        let rc = frontend.input_buffer.read();

        // TODO : re-estimate nominal gain

        // estimate s0 gain
        frontend.x_time.copy_from_slice(&rc[cp_len..cp_len + m]);
        self.estimate_gain_s0(config, frontend, 0);

        let s0_metric = self.s0_metrics(config, 0) * self.g0;
        self.stage = AcquisitionStage::PlcpShort1 { timer: 0, s0_metric };
        Ok(AcquisitionStatus::Pending)
    }

    fn execute_s0b(
        &mut self,
        config: &OfdmFrameConfig,
        frontend: &mut OfdmRxFrontend,
    ) -> Result<AcquisitionStatus> {
        let AcquisitionStage::PlcpShort1 { mut timer, s0_metric } = self.stage else {
            unreachable!("S0b handler called outside S0b state")
        };

        timer += 1;

        if timer < config.num_subcarriers() / 2 {
            self.stage = AcquisitionStage::PlcpShort1 { timer, s0_metric };
            return Ok(AcquisitionStatus::Pending);
        }

        let m = config.num_subcarriers();
        let m2 = m / 2;
        let cp_len = config.cp_len();

        // reset timer
        timer = m + cp_len - self.backoff;
        let rc = frontend.input_buffer.read();

        // estimate S0 gain
        frontend.x_time.copy_from_slice(&rc[cp_len..cp_len + m]);
        self.estimate_gain_s0(config, frontend, 1);
        let mut s_hat = self.s0_metrics(config, 1);
        s_hat *= self.g0;

        // re-adjust timer accordingly
        let tau_prime = (s0_metric + s_hat).arg() * m2 as f32 / (2.0 * std::f32::consts::PI);
        timer = (timer as i32 - tau_prime.round() as i32) as usize;

        // reborrow input buffer
        let rc = frontend.input_buffer.read();

        // compute carrier frequency offset estimate using ML method
        let mut t0 = Complex32::new(0.0, 0.0);
        for i in 0..m2 {
            t0 += rc[i].conj()
                * self.s0_time[i]
                * rc[i + m2]
                * self.s0_time[i + m2].conj();
        }

        // set NCO frequency
        frontend.nco.set_frequency(t0.arg() / m2 as f32);

        self.stage = AcquisitionStage::PlcpLong {
            timer,
            attempts: 0,
            previous_half_detected: false,
        };

        Ok(AcquisitionStatus::Pending)
    }

    fn execute_s1(
        &mut self,
        config: &OfdmFrameConfig,
        frontend: &mut OfdmRxFrontend,
    ) -> Result<AcquisitionStatus> {
        let AcquisitionStage::PlcpLong {
            mut timer,
            mut attempts,
            previous_half_detected,
        } = self.stage
        else {
            unreachable!("S1 handler called outside S1 state")
        };

        timer = timer.wrapping_sub(1);
        if timer > 0 {
            self.stage = AcquisitionStage::PlcpLong { timer, attempts, previous_half_detected };
            return Ok(AcquisitionStatus::Pending);
        }

        attempts += 1;

        let m = config.num_subcarriers();
        let m2 = m / 2;
        let cp_len = config.cp_len();

        // estimate S1 gain
        // TODO : add backoff in gain estimation
        let rc = frontend.input_buffer.read();
        frontend.x_time.copy_from_slice(&rc[cp_len..cp_len + m]);
        self.estimate_gain_s1(config, frontend);

        // compute detector output
        let mut g_hat = Complex32::new(0.0, 0.0);
        for i in 0..m {
            g_hat += self.gain[(i + 1) % m] * self.gain[i].conj();
        }
        g_hat /= self.m_s1 as f32; // normalize output
        g_hat *= self.g0;
        // rotate by complex phasor relative to timing backoff
        g_hat *= Complex32::from_polar(
            1.0,
            self.backoff as f32 * 2.0 * std::f32::consts::PI / m as f32,
        );

        // check conditions for g_hat:
        //  1. magnitude should be large (near unity) when aligned
        //  2. phase should be very near zero (time aligned)
        let phase_limit = 0.1 * std::f32::consts::PI;
        let magnitude_ok = g_hat.norm() > self.sync_threshold;
        let detected = magnitude_ok && g_hat.arg().abs() < phase_limit;

        // when the prefix is at least half a symbol long, it can cause ambiguity
        // for the symbol detection. require it to be found in the previous window
        // in order for this window to achieve detection
        let precursor_ok = cp_len < m2 || previous_half_detected;

        // calculate the prefix check for the next window. the prefix is rotated
        // by pi.
        let this_half_detected =
            magnitude_ok && (std::f32::consts::PI - g_hat.arg().abs()).abs() < phase_limit;

        if detected && precursor_ok {
            // normalize gain by subcarriers, apply timing backoff correction
            let counts = config.counts();
            let scale = m as f32 / ((counts.pilot + counts.data) as f32).sqrt();
            for i in 0..m {
                self.gain[i] *= scale; // gain due to relative subcarrier allocation
                self.gain[i] *= self.backoff_phase[i]; // timing backoff correction
            }
            self.estimate_eqgain(config)?;
            return Ok(AcquisitionStatus::Acquired {
                payload_timer: m + cp_len + self.backoff,
                backoff: self.backoff,
            });
        }

        if attempts == 16 {
            // ran out of attempts, reset to head
            frontend.nco.reset();
            self.stage = AcquisitionStage::SeekPlcp { timer: m2 };
            return Ok(AcquisitionStatus::Pending);
        }

        // wait another half symbol
        self.stage = AcquisitionStage::PlcpLong {
            timer: m2,
            attempts,
            previous_half_detected: this_half_detected,
        };
        Ok(AcquisitionStatus::Pending)
    }

    fn s0_metrics(&self, config: &OfdmFrameConfig, which: usize) -> Complex32 {
        let gain = if which == 0 { &self.g0a } else { &self.g0b };
        let m = config.num_subcarriers();

        // compute timing estimate, accumulate phase difference across
        // gains on subsequent pilot subcarriers (note that all the odd
        // subcarriers are null)
        let mut metric = Complex32::new(0.0, 0.0);
        for i in (0..m).step_by(2) {
            metric += gain[(i + 2) % m] * gain[i].conj();
        }
        metric / self.m_s0 as f32 // normalize output
    }

    fn estimate_gain_s0(
        &mut self,
        config: &OfdmFrameConfig,
        frontend: &mut OfdmRxFrontend,
        which: usize,
    ) {
        let m = config.num_subcarriers();

        // compute fft of x_time
        frontend.fft.run(&frontend.x_time, &mut frontend.x_freq);

        let scale = (self.m_s0 as f32).sqrt() / m as f32;
        let destination = if which == 0 { &mut self.g0a } else { &mut self.g0b };

        for i in 0..m {
            destination[i] = if config.allocation()[i] != SubcarrierType::Null && i % 2 == 0 {
                frontend.x_freq[i] * self.s0_freq[i].conj() * scale
            } else {
                Complex32::new(0.0, 0.0)
            };
        }
    }

    fn estimate_gain_s1(&mut self, config: &OfdmFrameConfig, frontend: &mut OfdmRxFrontend) {
        let m = config.num_subcarriers();

        // compute fft of x_time
        frontend.fft.run(&frontend.x_time, &mut frontend.x_freq);

        let scale = (self.m_s1 as f32).sqrt() / m as f32;

        for i in 0..m {
            self.gain[i] = if config.allocation()[i] != SubcarrierType::Null {
                frontend.x_freq[i] * self.s1_freq[i].conj() * scale
            } else {
                Complex32::new(0.0, 0.0)
            };
        }
    }

    fn estimate_eqgain(&mut self, config: &OfdmFrameConfig) -> Result<()> {
        self.eqgain.estimate(config.allocation(), &mut self.gain)
    }
}

/// Payload timing, equalization, and pilot-aided phase/timing tracking.
struct PayloadReceiver {
    timer: usize,
    symbol_index: usize,
    backoff: usize,
    equalizer: Vec<Complex32>,
    active: Vec<(usize, f32)>,
    nulls: Vec<usize>,
    pilot_bins: Vec<usize>,
    pilot_frequencies: Vec<f32>,
    pilot_phases: Vec<f32>,
    pilot_sequence: MSequence,
    phase_offset: f32,
    phase_slope: f32,
}

impl PayloadReceiver {
    fn new(config: &OfdmFrameConfig) -> Result<Self> {
        let m = config.num_subcarriers();
        let m2 = m / 2;
        let allocation = config.allocation();

        let mut active = Vec::new();
        let mut nulls = Vec::new();
        for (k, &kind) in allocation.iter().enumerate() {
            if kind == SubcarrierType::Null {
                nulls.push(k);
            } else {
                let frequency = if k > m2 { k as f32 - m as f32 } else { k as f32 };
                active.push((k, frequency));
            }
        }

        let mut pilot_bins = Vec::new();
        let mut pilot_frequencies = Vec::new();
        for i in 0..m {
            // keep pilots ordered from negative to positive frequency so phase
            // unwrapping follows adjacent points across the occupied band
            let k = (i + m2) % m;
            if allocation[k] == SubcarrierType::Pilot {
                pilot_bins.push(k);
                pilot_frequencies.push(if k > m2 { k as f32 - m as f32 } else { k as f32 });
            }
        }
        let pilot_phases = vec![0.0; pilot_bins.len()];

        Ok(Self {
            timer: 0,
            symbol_index: 0,
            backoff: 0,
            equalizer: vec![Complex32::new(0.0, 0.0); m],
            active,
            nulls,
            pilot_bins,
            pilot_frequencies,
            pilot_phases,
            pilot_sequence: MSequence::create_default(8)?,
            phase_offset: 0.0,
            phase_slope: 0.0,
        })
    }

    fn reset(&mut self) {
        self.timer = 0;
        self.symbol_index = 0;
        self.backoff = 0;
        self.pilot_sequence.reset();
        self.phase_offset = 0.0;
        self.phase_slope = 0.0;
    }

    fn begin(
        &mut self,
        timer: usize,
        backoff: usize,
        backoff_phase: &[Complex32],
        channel_gain: &[Complex32],
    ) {
        self.reset();
        self.timer = timer;
        self.backoff = backoff;
        // compute composite gain
        for ((equalizer, &phase), &gain) in
            self.equalizer.iter_mut().zip(backoff_phase).zip(channel_gain)
        {
            *equalizer = phase / gain;
        }
    }

    fn execute(&mut self, config: &OfdmFrameConfig, frontend: &mut OfdmRxFrontend) -> Result<bool> {
        self.timer -= 1;
        if self.timer != 0 {
            return Ok(false);
        }

        let m = config.num_subcarriers();
        let lo = config.cp_len() - self.backoff;
        let rc = frontend.input_buffer.read();

        // run fft
        frontend.x_time.copy_from_slice(&rc[lo..lo + m]);
        frontend.fft.run(&frontend.x_time, &mut frontend.x_freq);

        // apply equalizer gain
        for (x, &gain) in frontend.x_freq.iter_mut().zip(&self.equalizer) {
            *x *= gain;
        }

        self.correct_phase_and_timing(&mut frontend.x_freq, &mut frontend.nco)?;

        self.timer = config.symbol_len();
        self.symbol_index += 1;
        Ok(true)
    }

    fn correct_phase_and_timing(&mut self, subcarriers: &mut [Complex32], nco: &mut Osc) -> Result<()> {
        let n = self.pilot_bins.len();
        if n < 2 {
            return Err(Error::Internal(
                "ofdmframesync_rxsymbol(), at least two pilot subcarriers are required".into(),
            ));
        }

        // find pilot phase received vs actual
        for (phase, &k) in self.pilot_phases.iter_mut().zip(&self.pilot_bins) {
            let pilot = if self.pilot_sequence.advance() != 0 { 1.0 } else { -1.0 };
            *phase = (subcarriers[k] * pilot).arg();
        }
        unwrap_phase(&mut self.pilot_phases);

        // closed-form first-order least squares for timing offset/phase error
        // least-squares line y = offset + slope*x:
        // slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²), offset = (Σy - slope*Σx) / n.
        let nf = n as f32;
        let sum_x: f32 = self.pilot_frequencies.iter().sum();
        let sum_y: f32 = self.pilot_phases.iter().sum();
        let sum_xx: f32 = self.pilot_frequencies.iter().map(|x| x * x).sum();
        let sum_xy: f32 =
            self.pilot_frequencies.iter().zip(&self.pilot_phases).map(|(x, y)| x * y).sum();
        let denominator = nf * sum_xx - sum_x * sum_x;
        if denominator == 0.0 {
            return Err(Error::Internal(
                "ofdmframesync_rxsymbol(), pilot frequencies are degenerate".into(),
            ));
        }
        let mut slope = (nf * sum_xy - sum_x * sum_y) / denominator;
        let offset = (sum_y - slope * sum_x) / nf;

        // filter slope estimate (timing offset).
        let alpha = 0.3;
        slope = alpha * slope + (1.0 - alpha) * self.phase_slope;
        self.phase_slope = slope;

        for &(k, frequency) in &self.active {
            subcarriers[k] *= Complex32::from_polar(1.0, -(offset + slope * frequency));
        }
        for &k in &self.nulls {
            subcarriers[k] = Complex32::new(0.0, 0.0);
        }

        if self.symbol_index > 0 {
            let mut phase_error = offset - self.phase_offset;
            while phase_error > std::f32::consts::PI {
                phase_error -= 2.0 * std::f32::consts::PI;
            }
            while phase_error < -std::f32::consts::PI {
                phase_error += 2.0 * std::f32::consts::PI;
            }
            nco.adjust_frequency(1e-3 * phase_error);
        }
        self.phase_offset = offset;

        Ok(())
    }
}


/// One equalized OFDM payload symbol recovered by [`OfdmFrameSync::execute`].
///
/// Both slices borrow the synchronizer's internal buffers and remain valid
/// until its next mutable operation.
#[derive(Clone, Copy, Debug)]
pub struct OfdmFrameSyncSymbol<'a> {
    /// Equalized subcarrier values, indexed by natural FFT bin.
    pub subcarriers: &'a [Complex32],
    /// Subcarrier allocation corresponding to [`Self::subcarriers`].
    pub allocation: &'a [SubcarrierType],
}

/// Result of pushing a block of samples through an [`OfdmFrameSync`].
#[derive(Clone, Copy, Debug)]
pub struct OfdmFrameSyncOutput<'a> {
    /// Number of input samples consumed. Resume with `&input[consumed..]`.
    pub consumed: usize,
    /// Recovered payload symbol, if one became available.
    pub symbol: Option<OfdmFrameSyncSymbol<'a>>,
}

/// Result of copying recovered symbols with
/// [`OfdmFrameSync::execute_symbols_into`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OfdmFrameSyncBlockOutput {
    /// Number of input samples consumed. Resume with `&input[consumed..]`.
    pub consumed: usize,
    /// Number of complete OFDM symbols copied to the destination.
    pub symbols_written: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameSyncMode {
    Acquiring,
    Receiving,
}

/// OFDM frame synchronizer
///
/// Consumes a sample stream and returns each recovered payload symbol.
///
/// [`Self::execute`] stops after recovering one symbol so the caller can
/// process it and decide whether the logical frame is complete. The caller
/// must call [`Self::reset`] after the final symbol before feeding the next
/// frame.
pub struct OfdmFrameSync {
    config: OfdmFrameConfig,
    frontend: OfdmRxFrontend,
    acquisition: OfdmFrameAcquisition,
    payload: PayloadReceiver,
    mode: FrameSyncMode,
}

impl OfdmFrameSync {
    /// Create an OFDM frame synchronizer from a validated configuration.
    pub fn new(config: &OfdmFrameConfig) -> Result<Self> {
        let mut synchronizer = Self {
            config: config.clone(),
            frontend: OfdmRxFrontend::new(config)?,
            acquisition: OfdmFrameAcquisition::new(config)?,
            payload: PayloadReceiver::new(config)?,
            mode: FrameSyncMode::Acquiring,
        };
        synchronizer.reset();
        Ok(synchronizer)
    }

    /// number of subcarriers
    pub fn num_subcarriers(&self) -> usize {
        self.config.num_subcarriers()
    }

    /// number of null subcarriers
    pub fn num_null(&self) -> usize {
        self.config.counts().null
    }

    /// number of pilot subcarriers
    pub fn num_pilot(&self) -> usize {
        self.config.counts().pilot
    }

    /// number of data subcarriers
    pub fn num_data(&self) -> usize {
        self.config.counts().data
    }

    pub fn reset(&mut self) {
        self.frontend.reset();
        self.acquisition.reset();
        self.payload.reset();
        self.mode = FrameSyncMode::Acquiring;
    }

    /// true once a frame has been detected and is being received
    pub fn is_frame_open(&self) -> bool {
        match self.mode {
            FrameSyncMode::Acquiring => self.acquisition.is_frame_open(),
            FrameSyncMode::Receiving => true,
        }
    }

    /// get receiver RSSI
    pub fn get_rssi(&self) -> f32 {
        // TODO: see note in acquisition.get_rssi()
        self.acquisition.get_rssi()
    }

    /// get receiver carrier frequency offset estimate
    pub fn get_cfo(&self) -> f32 {
        self.frontend.nco.get_frequency()
    }

    /// set receiver carrier frequency offset estimate
    pub fn set_cfo(&mut self, cfo: f32) {
        self.frontend.nco.set_frequency(cfo);
    }

    /// Select the channel estimator used when the frame is acquired
    ///
    /// Defaults to [`EqGainMethod::Dft`] with a tap count derived from the
    /// cyclic prefix. [`EqGainMethod::Poly`] with order 4 reproduces
    /// liquid's behavior, and is marginally better on a nearly-flat channel
    /// where the extra taps only admit noise. It also has a smaller computational
    /// complexity.
    pub fn set_eqgain_method(&mut self, method: EqGainMethod) -> Result<()> {
        self.acquisition.set_eqgain_method(&self.config, method)
    }

    /// the channel estimator in use
    pub fn eqgain_method(&self) -> EqGainMethod {
        self.acquisition.eqgain_method()
    }

    /// Push samples through the synchronizer until one payload symbol is
    /// recovered or the input is exhausted.
    ///
    /// When a symbol is returned, resume with `&x[output.consumed..]` after the
    /// borrowed symbol is no longer needed. If it completes the logical frame,
    /// call [`Self::reset`] before resuming so the next preamble can be found.
    pub fn execute<'a>(&'a mut self, x: &[Complex32]) -> Result<OfdmFrameSyncOutput<'a>> {
        for (i, &xi) in x.iter().enumerate() {
            let mix_down = match self.mode {
                FrameSyncMode::Acquiring => self.acquisition.should_mix_down(),
                FrameSyncMode::Receiving => true,
            };
            let sample = if mix_down {
                let sample = self.frontend.nco.mix_down(xi);
                self.frontend.nco.step();
                sample
            } else {
                xi
            };
            self.frontend.input_buffer.push(sample);

            let symbol_ready = match self.mode {
                FrameSyncMode::Acquiring => {
                    match self.acquisition.execute(&self.config, &mut self.frontend)? {
                        AcquisitionStatus::Pending => false,
                        AcquisitionStatus::Acquired { payload_timer, backoff } => {
                            self.payload.begin(
                                payload_timer,
                                backoff,
                                &self.acquisition.backoff_phase,
                                &self.acquisition.gain,
                            );
                            self.mode = FrameSyncMode::Receiving;
                            false
                        }
                    }
                }
                FrameSyncMode::Receiving => self.payload.execute(&self.config, &mut self.frontend)?,
            };

            if symbol_ready {
                return Ok(OfdmFrameSyncOutput {
                    consumed: i + 1,
                    symbol: Some(OfdmFrameSyncSymbol {
                        subcarriers: &self.frontend.x_freq,
                        allocation: self.config.allocation(),
                    }),
                });
            }
        }

        Ok(OfdmFrameSyncOutput { consumed: x.len(), symbol: None })
    }

    /// Push samples through the synchronizer and copy as many recovered payload
    /// symbols as will fit in `symbols`.
    ///
    /// The destination is a flat array of complete symbols and its length must
    /// be a multiple of [`Self::num_subcarriers`]. Symbol `i` is written to
    /// `symbols[i*M..(i+1)*M]`. Processing stops when the input is exhausted or
    /// the destination is full. An empty destination consumes no input.
    ///
    /// The synchronizer does not know the logical frame length. Callers must
    /// limit the destination to the number of symbols remaining in the frame
    /// and call [`Self::reset`] after the final one; otherwise samples from the
    /// next frame can be interpreted as additional payload symbols.
    pub fn execute_symbols_into(
        &mut self,
        x: &[Complex32],
        symbols: &mut [Complex32],
    ) -> Result<OfdmFrameSyncBlockOutput> {
        let m = self.num_subcarriers();
        if symbols.len() % m != 0 {
            return Err(Error::Config(format!(
                "ofdmframesync_execute_symbols_into(), output length must be a multiple of {}",
                m
            )));
        }

        let capacity = symbols.len() / m;
        let mut consumed = 0;
        let mut symbols_written = 0;

        while consumed < x.len() && symbols_written < capacity {
            let output = self.execute(&x[consumed..])?;
            consumed += output.consumed;

            let Some(symbol) = output.symbol else {
                break;
            };
            let lo = symbols_written * m;
            symbols[lo..lo + m].copy_from_slice(symbol.subcarriers);
            symbols_written += 1;
        }

        Ok(OfdmFrameSyncBlockOutput { consumed, symbols_written })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multichannel::{ofdmframe_init_default_sctype, OfdmFrameGen};
    use crate::random::{crandnf, randf};
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

    fn new_framesync(
        num_subcarriers: usize,
        cp_len: usize,
        taper_len: usize,
        allocation: Option<&[SubcarrierType]>,
    ) -> Result<OfdmFrameSync> {
        let config = OfdmFrameConfig::new(num_subcarriers, cp_len, taper_len, allocation)?;
        OfdmFrameSync::new(&config)
    }

    fn make_frame(
        fg: &mut OfdmFrameGen,
        m: usize,
        cp_len: usize,
        num_data: usize,
        x: &[Complex32],
    ) -> Vec<Complex32> {
        let sym = m + cp_len;
        let mut y = vec![Complex32::new(0.0, 0.0); (3 + num_data) * sym];
        let mut n = 0;
        fg.write_s0a(&mut y[n..n + sym]).unwrap();
        n += sym;
        fg.write_s0b(&mut y[n..n + sym]).unwrap();
        n += sym;
        fg.write_s1(&mut y[n..n + sym]).unwrap();
        n += sym;
        for _ in 0..num_data {
            fg.write_symbol(x, &mut y[n..n + sym]).unwrap();
            n += sym;
        }
        assert_eq!(n, y.len());
        y
    }

    fn collect_symbols(fs: &mut OfdmFrameSync, samples: &[Complex32]) -> Vec<Vec<Complex32>> {
        let mut consumed = 0;
        let mut symbols = Vec::new();
        while consumed < samples.len() {
            let (n, symbol) = {
                let output = fs.execute(&samples[consumed..]).unwrap();
                (
                    output.consumed,
                    output.symbol.map(|symbol| {
                        assert_eq!(symbol.allocation.len(), symbol.subcarriers.len());
                        symbol.subcarriers.to_vec()
                    }),
                )
            };
            assert!(n > 0, "execute must make progress on non-empty input");
            consumed += n;
            if let Some(symbol) = symbol {
                symbols.push(symbol);
            }
        }
        symbols
    }

    fn ofdmframesync_acquire_test(num_subcarriers: usize, cp_len: usize, taper_len: usize) {
        // options
        let m = num_subcarriers;
        let tol = 1e-2f32; // error tolerance

        let dphi = 1.0 / m as f32; // carrier frequency offset

        // subcarrier allocation (initialize to default)
        let mut p = vec![SubcarrierType::default(); m];
        ofdmframe_init_default_sctype(&mut p).unwrap();

        // derived values
        let num_samples = (3 + 1) * (m + cp_len);

        // create synthesizer/analyzer objects
        let config = OfdmFrameConfig::new(m, cp_len, taper_len, Some(&p)).unwrap();
        let mut fg = OfdmFrameGen::new(&config).unwrap();
        let mut fs = OfdmFrameSync::new(&config).unwrap();

        let mut y = vec![Complex32::new(0.0, 0.0); num_samples];

        // assemble full frame
        let sym = m + cp_len;
        let mut n = 0;

        // write first S0 symbol
        fg.write_s0a(&mut y[n..n + sym]).unwrap();
        n += sym;

        // write second S0 symbol
        fg.write_s0b(&mut y[n..n + sym]).unwrap();
        n += sym;

        // write S1 symbol
        fg.write_s1(&mut y[n..n + sym]).unwrap();
        n += sym;

        // generate data symbol (random)
        let x: Vec<Complex32> = (0..m)
            .map(|_| Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * randf()))
            .collect();

        // write data symbol
        fg.write_symbol(&x, &mut y[n..n + sym]).unwrap();
        n += sym;

        // validate frame length
        assert_eq!(n, num_samples);

        // add carrier offset
        for (i, v) in y.iter_mut().enumerate() {
            *v *= Complex32::from_polar(1.0, dphi * i as f32);
        }

        // run receiver
        let symbols = collect_symbols(&mut fs, &y);

        // check output
        let recovered = symbols.last().expect("receiver should recover a symbol");
        for i in 0..m {
            if p[i] == SubcarrierType::Data {
                let d = x[i] - recovered[i];
                let e = (d * d.conj()).re;
                assert_abs_diff_eq!(e.abs(), 0.0, epsilon = tol);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_ofdmframesync_acquire_n64)]
    fn test_ofdmframesync_acquire_n64() {
        ofdmframesync_acquire_test(64, 8, 0);
    }

    #[test]
    #[autotest_annotate(autotest_ofdmframesync_acquire_n128)]
    fn test_ofdmframesync_acquire_n128() {
        ofdmframesync_acquire_test(128, 16, 0);
    }

    #[test]
    #[autotest_annotate(autotest_ofdmframesync_acquire_n256)]
    fn test_ofdmframesync_acquire_n256() {
        ofdmframesync_acquire_test(256, 32, 0);
    }

    #[test]
    #[autotest_annotate(autotest_ofdmframesync_acquire_n512)]
    fn test_ofdmframesync_acquire_n512() {
        ofdmframesync_acquire_test(512, 64, 0);
    }

    #[test]
    fn test_ofdmframesync_acquire_long_prefixes() {
        for (m, cp_len) in [(32, 16), (32, 20), (64, 32), (64, 40), (128, 64), (128, 80)] {
            ofdmframesync_acquire_test(m, cp_len, 0);
        }
    }

    #[test]
    #[autotest_annotate(autotest_ofdmframesync_config)]
    fn test_ofdmframesync_config() {
        // check invalid function calls
        assert!(new_framesync(0, 16, 4, None).is_err());
        assert!(new_framesync(7, 16, 4, None).is_err());
        assert!(new_framesync(65, 16, 4, None).is_err());
        assert!(new_framesync(64, 66, 4, None).is_err());
        assert!(new_framesync(64, 16, 24, None).is_err());

        // create proper object and test configurations
        let config = OfdmFrameConfig::new(64, 16, 4, None).unwrap();
        let mut q = OfdmFrameSync::new(&config).unwrap();

        assert!(!q.is_frame_open());
        q.set_cfo(0.0);
        assert_eq!(q.get_cfo(), 0.0);
    }

    #[test]
    fn test_ofdmframesync_detection_thresholds_scale_with_subcarriers() {
        let big = new_framesync(64, 8, 0, None).unwrap();
        assert_abs_diff_eq!(big.acquisition.detect_threshold, 0.35, epsilon = 1e-6);
        assert_abs_diff_eq!(big.acquisition.sync_threshold, 0.30, epsilon = 1e-6);

        let small = new_framesync(32, 8, 0, None).unwrap();
        assert_abs_diff_eq!(small.acquisition.detect_threshold, 0.47, epsilon = 1e-6);
        assert_abs_diff_eq!(small.acquisition.sync_threshold, 0.42, epsilon = 1e-6);
    }

    fn ofdmframesync_test_cfo_noise(
            epsilon: f32,
            snr_db: f32,
            num_data: usize,
            pad: usize,
        ) {
        let m = 64;
        let cp_len = 8;
        let dphi = 2.0 * std::f32::consts::PI * epsilon / m as f32;
        let nstd = 10.0f32.powf(-snr_db / 20.0);

        let mut p = vec![SubcarrierType::Null; m];
        ofdmframe_init_default_sctype(&mut p).unwrap();
        let mut fg = new_framegen(m, cp_len, 0, Some(&p)).unwrap();
        let mut fs = new_framesync(m, cp_len, 0, Some(&p)).unwrap();

        assert_eq!(fs.get_cfo(), 0.0, "cfo starts at zero");

        let x: Vec<Complex32> = (0..m)
            .map(|_| Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * randf()))
            .collect();

        let frame = make_frame(&mut fg, m, cp_len, num_data, &x);

        // pad the front with silence to test rejection of false acquisitions
        let mut y = vec![Complex32::new(0.0, 0.0); frame.len() + pad];
        y[pad..].copy_from_slice(&frame);

        // apply noise and cfo
        for (i, v) in y.iter_mut().enumerate() {
            *v *= Complex32::from_polar(1.0, dphi * i as f32);
            *v += crandnf() * nstd * std::f32::consts::FRAC_1_SQRT_2;
        }

        let symbols = collect_symbols(&mut fs, &y);

        assert!(!symbols.is_empty(), "no symbol recovered");
        assert!(symbols.len() == num_data, "wrong number of symbols received");

        assert_abs_diff_eq!(fs.get_cfo(), dphi, epsilon = 1e-2);

        let mut err_0 = 0.0f32;
        let mut err_last = 0.0f32;
        let mut c = 0;
        for i in 0..m {
            if p[i] == SubcarrierType::Data {
                err_0 += (x[i] - symbols[0][i]).norm_sqr();
                err_last += (x[i] - symbols[symbols.len() - 1][i]).norm_sqr();
                c += 1;
            }
        }
        let rmse_0 = (err_0 / c as f32).sqrt();
        assert!(
            rmse_0 < nstd + 0.5,
            "payload first rmse {rmse_0} at snr {snr_db} dB ({nstd})"
        );
        let rmse_last = (err_last / c as f32).sqrt();
        assert!(
            rmse_last < nstd + 0.5,
            "payload last rmse {rmse_last} at snr {snr_db} dB ({nstd})"
        );
    }

    #[test]
    fn test_ofdmframesync_cfo_noise() {
        for epsilon in [0.2f32, 0.3f32, 0.4f32] {
            for snr_db in [60.0f32, 30.0, 20.0, 10.0] {
                for num_data in [1, 2, 4, 8, 16, 32] {
                    for pad in [0, 3, 7, 19, 251, 1021] {
                        ofdmframesync_test_cfo_noise(epsilon, snr_db, num_data, pad);
                    }
                }
            }
        }
    }

    fn ofdmframesync_test_delay(
        delay: usize,
        gain: f32,
        method: EqGainMethod,
        rmse_max: f32,
    ) {
        let m = 64;
        let cp_len = 16;

        let mut p = vec![SubcarrierType::Null; m];
        ofdmframe_init_default_sctype(&mut p).unwrap();

        let payload: Vec<Complex32> = (0..m)
            .map(|i| {
                let a = if i % 2 == 0 { 1.0 } else { -1.0 };
                let b = if (i / 5) % 2 == 0 { 1.0 } else { -1.0 };
                Complex32::new(
                    a * std::f32::consts::FRAC_1_SQRT_2,
                    b * std::f32::consts::FRAC_1_SQRT_2,
                )
            })
            .collect();

        let test_case_rmse = |delay: usize, gain: f32, method: EqGainMethod| -> Option<f32> {
            let mut fg = new_framegen(m, cp_len, 0, Some(&p)).unwrap();
            let mut fs = new_framesync(m, cp_len, 0, Some(&p)).unwrap();
            fs.set_eqgain_method(method).unwrap();

            let y = make_frame(&mut fg, m, cp_len, 1, &payload);
            let a = Complex32::from_polar(gain, 0.8);
            let mut ch = vec![Complex32::new(0.0, 0.0); y.len() + delay];
            // apply a delay
            for i in 0..ch.len() {
                if i < y.len() {
                    ch[i] = y[i] + if i >= delay { a * y[i - delay] } else { Complex32::new(0.0, 0.0) };
                } else {
                    ch[i] = a * y[i - delay];
                }
            }

            let symbols = collect_symbols(&mut fs, &ch);

            let last_symbol = symbols.last()?;
            let mut err = 0.0f32;
            let mut n = 0;
            for i in 0..m {
                if p[i] == SubcarrierType::Data {
                    err += (payload[i] - last_symbol[i]).norm_sqr();
                    n += 1;
                }
            }
            Some((err / n as f32).sqrt())
        };

        let rmse = test_case_rmse(delay, gain, method).expect("test case failed to acquire");
        let method_s = match method {
            EqGainMethod::Poly { order } => format!("poly-{order}"),
            EqGainMethod::Dft { num_taps } => format!("dft-{num_taps}")
        };
        let delay_s = if gain < 1.0 { "delay" } else { "lead" };
        assert!(rmse < rmse_max, "ofdmframesync method ({}) {} {}: rmse actual {} >= expected {}",
          method_s, delay_s, delay, rmse, rmse_max);
    }

    #[test]
    fn test_ofdmframesync_poly_4_delay_flat() {
        ofdmframesync_test_delay(0, 0.0, EqGainMethod::Poly { order: 4 }, 2e-6);
    }

    #[test]
    fn test_ofdmframesync_poly_8_delay_flat() {
        ofdmframesync_test_delay(0, 0.0, EqGainMethod::Poly { order: 8 }, 6e-4);
    }

    #[test]
    fn test_ofdmframesync_dft_delay_flat() {
        ofdmframesync_test_delay(0, 0.0, EqGainMethod::Dft { num_taps: 16 }, 4e-3);
    }

    #[test]
    fn test_ofdmframesync_poly_4_delay_1() {
        ofdmframesync_test_delay(1, 0.35, EqGainMethod::Poly { order: 4 }, 6e-3);
    }

    #[test]
    fn test_ofdmframesync_poly_8_delay_1() {
        ofdmframesync_test_delay(1, 0.35, EqGainMethod::Poly { order: 8 }, 4e-4);
    }

    #[test]
    fn test_ofdmframesync_dft_delay_1() {
        ofdmframesync_test_delay(1, 0.35, EqGainMethod::Dft { num_taps: 16 }, 3e-3);
    }

    #[test]
    fn test_ofdmframesync_poly_4_lead_1() {
        ofdmframesync_test_delay(1, 3.0, EqGainMethod::Poly { order: 4 }, 5e-3);
    }

    #[test]
    fn test_ofdmframesync_poly_8_lead_1() {
        ofdmframesync_test_delay(1, 3.0, EqGainMethod::Poly { order: 8 }, 2e-3);
    }

    #[test]
    fn test_ofdmframesync_dft_lead_1() {
        ofdmframesync_test_delay(1, 3.0, EqGainMethod::Dft { num_taps: 16 }, 3e-3);
    }

    #[test]
    fn test_ofdmframesync_poly_4_delay_3() {
        ofdmframesync_test_delay(3, 0.35, EqGainMethod::Poly { order: 4 }, 4e-1);
    }

    #[test]
    fn test_ofdmframesync_poly_8_delay_3() {
        ofdmframesync_test_delay(3, 0.35, EqGainMethod::Poly { order: 8 }, 7e-2);
    }

    #[test]
    fn test_ofdmframesync_dft_delay_3() {
        ofdmframesync_test_delay(3, 0.35, EqGainMethod::Dft { num_taps: 16 }, 3e-3);
    }

    #[test]
    fn test_ofdmframesync_poly_4_lead_3() {
        ofdmframesync_test_delay(3, 3.0, EqGainMethod::Poly { order: 4 }, 3e-1);
    }

    #[test]
    fn test_ofdmframesync_poly_8_lead_3() {
        ofdmframesync_test_delay(3, 3.0, EqGainMethod::Poly { order: 8 }, 8e-2);
    }

    #[test]
    fn test_ofdmframesync_dft_lead_3() {
        // there's only 2 lead taps, so dft lead-3 blows up
        ofdmframesync_test_delay(3, 3.0, EqGainMethod::Dft { num_taps: 16 }, 2e-1);
    }

    #[test]
    fn test_ofdmframesync_poly_4_delay_8() {
        ofdmframesync_test_delay(8, 0.35, EqGainMethod::Poly { order: 4 }, 5e-1);
    }

    #[test]
    fn test_ofdmframesync_poly_8_delay_8() {
        ofdmframesync_test_delay(8, 0.35, EqGainMethod::Poly { order: 8 }, 5e-1);
    }

    #[test]
    fn test_ofdmframesync_dft_delay_8() {
        ofdmframesync_test_delay(8, 0.35, EqGainMethod::Dft { num_taps: 16 }, 4e-3);
    }

    #[test]
    fn test_execute_symbols_into_fills_destination_and_exposes_remainder() {
        let m = 64;
        let cp_len = 8;
        let sym = m + cp_len;

        let mut p = vec![SubcarrierType::Null; m];
        ofdmframe_init_default_sctype(&mut p).unwrap();
        let mut fg = new_framegen(m, cp_len, 0, Some(&p)).unwrap();
        let mut fs = new_framesync(m, cp_len, 0, Some(&p)).unwrap();

        let x: Vec<Complex32> = (0..m)
            .map(|i| {
                let re = if i % 2 == 0 { 0.5 } else { -0.5 };
                Complex32::new(re, 0.5)
            })
            .collect();
        let y = make_frame(&mut fg, m, cp_len, 3, &x);

        // validate the output shape before touching synchronizer state.
        let mut malformed = vec![Complex32::new(0.0, 0.0); m + 1];
        assert!(fs.execute_symbols_into(&y, &mut malformed).is_err());
        assert!(!fs.is_frame_open());

        let empty = fs.execute_symbols_into(&y, &mut []).unwrap();
        assert_eq!(empty, OfdmFrameSyncBlockOutput { consumed: 0, symbols_written: 0 });

        let mut first = vec![Complex32::new(0.0, 0.0); 2 * m];
        let first_result = fs.execute_symbols_into(&y, &mut first).unwrap();
        assert_eq!(first_result.symbols_written, 2);
        assert!(first_result.consumed < y.len());

        for recovered in first.chunks_exact(m) {
            for i in 0..m {
                if p[i] == SubcarrierType::Data {
                    assert!((recovered[i] - x[i]).norm() < 0.02, "subcarrier {i}");
                }
            }
        }

        let mut last = vec![Complex32::new(0.0, 0.0); m];
        let last_result = fs.execute_symbols_into(&y[first_result.consumed..], &mut last).unwrap();
        assert_eq!(last_result.symbols_written, 1);
        assert_eq!(last_result.consumed, sym);
        assert_eq!(first_result.consumed + last_result.consumed, y.len());
    }
}

