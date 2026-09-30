//
// ofdmframesync.rs
//
// OFDM frame synchronizer
//

use super::ofdm_frame::{ofdmframe_init_s0, ofdmframe_init_s1, OfdmFrameConfig, SubcarrierType};
use crate::buffer::Window;
use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::math::{poly_fit, poly_val};
use crate::matrix::{ColMatrixMut, LeastSquaresWorkspace, MatrixMut, MatrixRef};
use crate::nco::{unwrap_phase, Nco, NcoBackend};
use crate::sequence::MaximalLengthSequence;
use num_complex::Complex32;

/// Method an [`OfdmFrameSynchronizer`] uses to smooth its channel estimate.
///
/// The synchronizer measures the channel on each pilot and data subcarrier of
/// the third preamble symbol. The method fits a smooth response to those
/// measurements, which reduces noise, and the fit sets the equalizer for the
/// rest of the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EqGainMethod {
    /// Fits the magnitude and the unwrapped phase across frequency with
    /// separate polynomials of degree `order`.
    ///
    /// `order` must be at least one. An order above ten, or above one less
    /// than the number of pilot and data subcarriers, is reduced to that
    /// limit. A low order can miss frequency-selective fading. A high order
    /// over pilot and data subcarriers that span a narrow band can make the
    /// fit fail.
    Poly { order: usize },
    /// Fits a channel impulse response of `num_taps` taps by regularized
    /// least squares.
    ///
    /// `num_taps` runs from one to the number of pilot and data subcarriers.
    /// More taps can follow a longer delay spread but also fit more noise.
    /// The first two taps sit before the timing estimate, so channel energy
    /// that arrives slightly ahead of it is still fitted. With two taps only
    /// the first does, and with one tap neither does.
    Dft { num_taps: usize },
}

/// Runtime state for the acquisition channel estimator.
enum EqGainState {
    Poly { order: usize },
    Dft(Box<DftEqGainState>),
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
    fn estimate_poly(allocation: &[SubcarrierType], gain: &mut [Complex32], order: usize) -> Result<()> {
        let m = gain.len();
        let m2 = m / 2;
        let n = allocation.iter().filter(|&&p| p != SubcarrierType::Null).count();
        let order = order.min(n - 1).min(10);

        // the order can approach the sample count, making the system unstable in f32
        let mut x_freq = vec![0.0f64; n];
        let mut y_abs = vec![0.0f64; n];
        let mut y_arg = vec![0.0f32; n];
        let mut p_abs = vec![0.0f64; order + 1];
        let mut p_arg = vec![0.0f64; order + 1];

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
                x_freq[idx] = if k > m2 { k as f64 - m as f64 } else { k as f64 };
                x_freq[idx] /= m as f64;
                y_abs[idx] = gain[k].norm() as f64;
                y_arg[idx] = gain[k].arg();
                idx += 1;
            }
        }

        if idx != n {
            return Err(Error::Internal("ofdmframesync_estimate_eqgain_poly(), pilot subcarrier mismatch".into()));
        }

        unwrap_phase(&mut y_arg);
        let y_arg: Vec<f64> = y_arg.into_iter().map(f64::from).collect();
        poly_fit(&x_freq, &y_abs, &mut p_abs)?;
        poly_fit(&x_freq, &y_arg, &mut p_arg)?;

        for i in 0..m {
            let mut freq = if i > m2 { i as f64 - m as f64 } else { i as f64 };
            freq /= m as f64;
            let a = poly_val(&p_abs, freq) as f32;
            let theta = poly_val(&p_arg, freq) as f32;
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
    num_subcarriers: usize,
    num_taps: usize,
    active: Vec<usize>,
    basis: Vec<Complex32>,
    least_squares: LeastSquaresWorkspace<Complex32>,
    rhs: Vec<Complex32>,
    taps: Vec<Complex32>,
}

impl DftEqGainState {
    /// diagonal loading for the time-domain channel fit, as a fraction of the
    /// number of observations.
    const EQGAIN_RIDGE: f32 = 1e-3;

    /// number of *negative*-delay taps the time-domain channel fit reserves.
    const EQGAIN_LEAD_TAPS: usize = 2;

    fn new(num_subcarriers: usize, num_taps: usize, allocation: &[SubcarrierType]) -> Result<Self> {
        if allocation.len() != num_subcarriers {
            return Err(Error::Range(
                "OFDM channel-estimator allocation length must match the subcarrier count".into(),
            ));
        }
        if num_taps == 0 {
            return Err(Error::Range("OFDM channel estimator must have at least one tap".into()));
        }

        let active: Vec<_> = allocation
            .iter()
            .enumerate()
            .filter_map(|(k, &p)| (p != SubcarrierType::Null).then_some(k))
            .collect();
        if num_taps > active.len() || num_taps > num_subcarriers {
            return Err(Error::Range("OFDM channel-estimator tap count exceeds the usable subcarrier count".into()));
        }

        // basis[(k, t)] = exp(-j 2 pi k * delay(t) / M), where the
        // tap delays run from -lead upward so a slightly early sampling
        // instant stays representable. always preserve at least one tap
        // for positive delay
        let lead = Self::EQGAIN_LEAD_TAPS.min(num_subcarriers).min(num_taps - 1);
        let basis_len = num_subcarriers
            .checked_mul(num_taps)
            .ok_or_else(|| Error::Range("OFDM channel-estimator basis dimensions overflow".into()))?;
        // pick subcarrier-major access since that's what estimate() wants
        let mut basis = vec![Complex32::new(0.0, 0.0); basis_len];
        let mut basis_view = MatrixMut::new(&mut basis, num_subcarriers, num_taps)?;
        for t in 0..num_taps {
            let delay = (t + num_subcarriers - lead) % num_subcarriers;
            for k in 0..num_subcarriers {
                let theta = -2.0 * std::f32::consts::PI * (k as f32) * (delay as f32) / (num_subcarriers as f32);
                basis_view[(k, t)] = Complex32::from_polar(1.0, theta);
            }
        }

        // create a reusable least-squares solver for our equalizer. the ridge-
        // regularized design matrix stacks the DFT basis (sampled on the active
        // subcarriers) over sqrt(lambda) * I, solving the system
        //   [ A         ] taps = [ gain ]
        //   [ sqrt(l)*I ]        [ 0    ]
        // for 6 active subcarriers and 4 taps (augmented_rows = 10):
        //
        //              tap 0      tap 1      tap 2      tap 3
        //            +-----------------------------------------+
        //   obs 0    | A[0][0]   A[0][1]   A[0][2]   A[0][3]   |   <- DFT basis at active[0]
        //   obs 1    | A[1][0]   A[1][1]   A[1][2]   A[1][3]   |                   active[1]
        //   obs 2    | A[2][0]   A[2][1]   A[2][2]   A[2][3]   |                   active[2]
        //   obs 3    | A[3][0]   A[3][1]   A[3][2]   A[3][3]   |                   active[3]
        //   obs 4    | A[4][0]   A[4][1]   A[4][2]   A[4][3]   |                   active[4]
        //   obs 5    | A[5][0]   A[5][1]   A[5][2]   A[5][3]   |                   active[5]
        //            | sqrt(l)   0         0         0         |   <- ridge block = sqrt(lambda) * I
        //            | 0         sqrt(l)   0         0         |
        //            | 0         0         sqrt(l)   0         |
        //            | 0         0         0         sqrt(l)   |
        //            +-----------------------------------------+
        let observations = active.len();
        let augmented_rows = observations
            .checked_add(num_taps)
            .ok_or_else(|| Error::Range("OFDM channel-estimator augmented dimensions overflow".into()))?;
        let factor_len = augmented_rows
            .checked_mul(num_taps)
            .ok_or_else(|| Error::Range("OFDM channel-estimator augmented dimensions overflow".into()))?;
        let ridge = Self::EQGAIN_RIDGE * observations as f32;
        let ridge_scale = ridge.sqrt();
        let mut factors = vec![Complex32::new(0.0, 0.0); factor_len];
        let mut matrix = ColMatrixMut::new(&mut factors, augmented_rows, num_taps)?;
        for tap in 0..num_taps {
            let column = matrix.column_mut(tap);
            for (value, &subcarrier) in column[..observations].iter_mut().zip(&active) {
                *value = basis_view[(subcarrier, tap)];
            }
            column[observations + tap] = Complex32::new(ridge_scale, 0.0);
        }
        let least_squares = LeastSquaresWorkspace::from_design_matrix(matrix.as_ref())?;

        Ok(Self {
            num_subcarriers,
            num_taps,
            active,
            basis,
            least_squares,
            rhs: vec![Complex32::new(0.0, 0.0); augmented_rows],
            taps: vec![Complex32::new(0.0, 0.0); num_taps],
        })
    }

    /// Fit time-domain taps to the channel response on the active carriers.
    fn estimate(&mut self, gain: &mut [Complex32]) -> Result<()> {
        if gain.len() != self.num_subcarriers {
            return Err(Error::Range("OFDM channel-estimator gain length must match the subcarrier count".into()));
        }

        // fit a finite-length channel impulse response using selected columns of the
        // DFT matrix, then reconstruct the per-subcarrier channel response.
        // van de Beek et al., "On Channel Estimation in OFDM Systems," VTC 1995
        let m = gain.len();
        let n = self.active.len();
        let taps = self.num_taps;

        debug_assert_eq!(self.basis.len(), m * taps);
        debug_assert_eq!(self.rhs.len(), n + taps);
        debug_assert_eq!(self.taps.len(), taps);

        for (value, &subcarrier) in self.rhs[..n].iter_mut().zip(&self.active) {
            *value = gain[subcarrier];
        }
        self.rhs[n..].fill(Complex32::new(0.0, 0.0));
        self.least_squares.solve(&self.rhs, &mut self.taps)?;

        // reconstruct each subcarrier's response from the taps
        let basis = MatrixRef::new(&self.basis, m, taps)?;
        gain.fill(Complex32::new(0.0, 0.0));
        for &k in &self.active {
            let mut sum = Complex32::new(0.0, 0.0);
            for (&tap, &basis_value) in self.taps.iter().zip(basis.row(k)) {
                sum += tap * basis_value;
            }
            gain[k] = sum;
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
    nco: Nco,
}

impl OfdmRxFrontend {
    fn new(config: &OfdmFrameConfig) -> Result<Self> {
        let m = config.num_subcarriers();
        Ok(Self {
            fft: Fft::new(m, Direction::Forward),
            x_freq: vec![Complex32::new(0.0, 0.0); m],
            x_time: vec![Complex32::new(0.0, 0.0); m],
            input_buffer: Window::new(config.symbol_len())?,
            nco: Nco::new(NcoBackend::LookupTable),
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
        let backoff_phase = (0..m).map(|i| Complex32::from_polar(1.0, i as f32 * phi)).collect();
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
            eqgain: EqGainState::Dft(Box::new(DftEqGainState::new(m, taps, allocation)?)),
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

    fn rssi(&self) -> f32 {
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
                    return Err(Error::Config("ofdmframesync, polynomial order must be at least 1".into()));
                }
                EqGainState::Poly { order }
            }
            EqGainMethod::Dft { num_taps } => {
                if num_taps == 0 {
                    return Err(Error::Config("ofdmframesync, tap count must be at least 1".into()));
                }
                if num_taps > active || num_taps > m {
                    return Err(Error::Config(format!(
                        "ofdmframesync, tap count {} exceeds {} usable subcarriers",
                        num_taps,
                        active.min(m)
                    )));
                }
                EqGainState::Dft(Box::new(DftEqGainState::new(m, num_taps, config.allocation())?))
            }
        };
        Ok(())
    }

    fn eqgain_method(&self) -> EqGainMethod {
        self.eqgain.method()
    }

    fn execute(&mut self, config: &OfdmFrameConfig, frontend: &mut OfdmRxFrontend) -> Result<AcquisitionStatus> {
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

    fn execute_s0a(&mut self, config: &OfdmFrameConfig, frontend: &mut OfdmRxFrontend) -> Result<AcquisitionStatus> {
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

    fn execute_s0b(&mut self, config: &OfdmFrameConfig, frontend: &mut OfdmRxFrontend) -> Result<AcquisitionStatus> {
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
            t0 += rc[i].conj() * self.s0_time[i] * rc[i + m2] * self.s0_time[i + m2].conj();
        }

        // set NCO frequency
        frontend.nco.set_frequency(t0.arg() / m2 as f32);

        self.stage = AcquisitionStage::PlcpLong { timer, attempts: 0, previous_half_detected: false };

        Ok(AcquisitionStatus::Pending)
    }

    fn execute_s1(&mut self, config: &OfdmFrameConfig, frontend: &mut OfdmRxFrontend) -> Result<AcquisitionStatus> {
        let AcquisitionStage::PlcpLong { mut timer, mut attempts, previous_half_detected } = self.stage else {
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
        g_hat *= Complex32::from_polar(1.0, self.backoff as f32 * 2.0 * std::f32::consts::PI / m as f32);

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
        let this_half_detected = magnitude_ok && (std::f32::consts::PI - g_hat.arg().abs()).abs() < phase_limit;

        if detected && precursor_ok {
            // normalize gain by subcarriers, apply timing backoff correction
            let counts = config.counts();
            let scale = m as f32 / ((counts.pilot + counts.data) as f32).sqrt();
            for i in 0..m {
                self.gain[i] *= scale; // gain due to relative subcarrier allocation
                self.gain[i] *= self.backoff_phase[i]; // timing backoff correction
            }
            self.estimate_eqgain(config)?;
            return Ok(AcquisitionStatus::Acquired { payload_timer: m + cp_len + self.backoff, backoff: self.backoff });
        }

        if attempts == 16 {
            // ran out of attempts, reset to head
            frontend.nco.reset();
            self.stage = AcquisitionStage::SeekPlcp { timer: m2 };
            return Ok(AcquisitionStatus::Pending);
        }

        // wait another half symbol
        self.stage = AcquisitionStage::PlcpLong { timer: m2, attempts, previous_half_detected: this_half_detected };
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

    fn estimate_gain_s0(&mut self, config: &OfdmFrameConfig, frontend: &mut OfdmRxFrontend, which: usize) {
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
    pilot_sequence: MaximalLengthSequence,
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
            pilot_sequence: MaximalLengthSequence::from_degree(8)?,
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

    fn begin(&mut self, timer: usize, backoff: usize, backoff_phase: &[Complex32], channel_gain: &[Complex32]) {
        self.reset();
        self.timer = timer;
        self.backoff = backoff;
        // compute composite gain
        for ((equalizer, &phase), &gain) in self.equalizer.iter_mut().zip(backoff_phase).zip(channel_gain) {
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

    fn correct_phase_and_timing(&mut self, subcarriers: &mut [Complex32], nco: &mut Nco) -> Result<()> {
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
        let sum_xy: f32 = self.pilot_frequencies.iter().zip(&self.pilot_phases).map(|(x, y)| x * y).sum();
        let denominator = nf * sum_xx - sum_x * sum_x;
        if denominator == 0.0 {
            return Err(Error::Internal("ofdmframesync_rxsymbol(), pilot frequencies are degenerate".into()));
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

/// One equalized payload symbol returned by [`OfdmFrameSynchronizer::execute`].
///
/// The slices borrow the synchronizer.
#[derive(Clone, Copy, Debug)]
#[doc(alias = "OfdmFrameSyncSymbol")]
pub struct OfdmFrameSynchronizerSymbol<'a> {
    /// Equalized subcarrier values, in natural FFT-bin order.
    ///
    /// Data values come back at the scale they were given to the generator.
    /// Null subcarriers are zero.
    pub subcarriers: &'a [Complex32],
    /// Role of each entry of [`Self::subcarriers`].
    pub allocation: &'a [SubcarrierType],
}

/// Result of pushing a block of samples through an [`OfdmFrameSynchronizer`].
#[derive(Clone, Copy, Debug)]
#[doc(alias = "OfdmFrameSyncOutput")]
pub struct OfdmFrameSynchronizerOutput<'a> {
    /// Number of input samples consumed. Resume with `&input[consumed..]`.
    pub consumed: usize,
    /// Recovered payload symbol, or `None` when the input ran out first.
    pub symbol: Option<OfdmFrameSynchronizerSymbol<'a>>,
}

/// Result of copying recovered symbols with
/// [`OfdmFrameSynchronizer::execute_symbols_into`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(alias = "OfdmFrameSyncBlockOutput")]
pub struct OfdmFrameSynchronizerBlockOutput {
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

/// Receives frames from an [`OfdmFrameGenerator`] in a stream of samples.
///
/// The synchronizer searches the stream for the first two preamble symbols.
/// When it finds them, it estimates the symbol timing and the carrier
/// frequency offset, and mixes the rest of the frame down by that offset. The
/// third preamble symbol fixes the timing and gives the channel estimate,
/// which sets the equalizer. Each payload symbol is then equalized, and its
/// pilots correct the remaining phase and timing drift.
///
/// The frame length is not part of the preamble, so the synchronizer cannot
/// tell when a frame ends. [`Self::execute`] stops after each symbol so the
/// caller can decide whether the frame is complete. Call [`Self::reset`]
/// after the last symbol, or the synchronizer keeps returning symbols from
/// whatever follows.
///
/// If the third preamble symbol is not found within 16 tries, spaced half a
/// symbol apart, the synchronizer drops the detection and resumes the search.
///
/// [`OfdmFrameGenerator`]: super::OfdmFrameGenerator
#[doc(alias = "OfdmFrameSync")]
pub struct OfdmFrameSynchronizer {
    config: OfdmFrameConfig,
    frontend: OfdmRxFrontend,
    acquisition: OfdmFrameAcquisition,
    payload: PayloadReceiver,
    mode: FrameSyncMode,
}

impl OfdmFrameSynchronizer {
    /// Creates a synchronizer for the frame layout in `config`.
    ///
    /// The channel estimator starts as [`EqGainMethod::Dft`] with
    /// `max(cp_len, 2)` taps, capped at the number of pilot and data
    /// subcarriers.
    ///
    /// # Errors
    ///
    /// Returns an error when no pilot or data subcarrier sits at an even bin
    /// index. The first preamble symbols use only even bins.
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

    /// Returns the number of subcarriers.
    pub fn num_subcarriers(&self) -> usize {
        self.config.num_subcarriers()
    }

    /// Returns the number of null subcarriers.
    pub fn num_null(&self) -> usize {
        self.config.counts().null
    }

    /// Returns the number of pilot subcarriers.
    pub fn num_pilot(&self) -> usize {
        self.config.counts().pilot
    }

    /// Returns the number of data subcarriers.
    pub fn num_data(&self) -> usize {
        self.config.counts().data
    }

    /// Drops the current frame and resumes the preamble search.
    ///
    /// The carrier offset returns to zero. The channel estimator method and
    /// [`Self::rssi`] are kept.
    pub fn reset(&mut self) {
        self.frontend.reset();
        self.acquisition.reset();
        self.payload.reset();
        self.mode = FrameSyncMode::Acquiring;
    }

    /// Returns whether a frame has been detected.
    ///
    /// This stays true through the payload until [`Self::reset`]. It returns
    /// to false early only when the third preamble symbol is not found.
    pub fn is_frame_open(&self) -> bool {
        match self.mode {
            FrameSyncMode::Acquiring => self.acquisition.is_frame_open(),
            FrameSyncMode::Receiving => true,
        }
    }

    /// Returns the received signal strength, in dB relative to unit power.
    ///
    /// While searching, the synchronizer measures the average input power
    /// over one symbol every [`Self::num_subcarriers`] samples. The value is
    /// not updated once a frame is detected, so during a frame it holds the
    /// level at detection. A new synchronizer reports zero.
    pub fn rssi(&self) -> f32 {
        // TODO: see note in acquisition.rssi()
        self.acquisition.rssi()
    }

    /// Returns the carrier frequency offset being removed, in radians per
    /// sample.
    ///
    /// A positive value means the input is above its nominal frequency and is
    /// mixed down by this amount. The value lies in `(-pi, pi]`. It is set
    /// from the first two preamble symbols, which resolve offsets up to one
    /// subcarrier spacing, `2 * pi / num_subcarriers`, in either direction.
    /// Larger offsets alias. The pilots then adjust it slowly during the
    /// payload. It is zero after [`Self::reset`].
    pub fn cfo(&self) -> f32 {
        self.frontend.nco.frequency()
    }

    /// Sets the carrier frequency offset to remove, in radians per sample.
    ///
    /// The value wraps modulo `2 * pi`. It takes effect while a frame is open.
    /// Detecting a frame replaces it with a new estimate, so it is useful
    /// only for adjusting the offset during the payload.
    ///
    /// # Warning
    ///
    /// A nonzero value set before a frame is detected corrupts that frame's
    /// offset estimate. The offset is measured on preamble samples that were
    /// already partly mixed down by this value.
    pub fn set_cfo(&mut self, carrier_frequency_offset: f32) {
        self.frontend.nco.set_frequency(carrier_frequency_offset);
    }

    /// Selects the method that smooths the channel estimate.
    ///
    /// A polynomial fit can suit a nearly flat channel. The tap fit can follow
    /// a longer delay spread. The method is used for the next channel
    /// estimate. It does not change the equalizer of a frame already being
    /// received.
    ///
    /// # Errors
    ///
    /// Returns an error when the polynomial order is zero or the tap count is
    /// zero or more than the number of pilot and data subcarriers. The method
    /// is unchanged on error.
    pub fn set_eqgain_method(&mut self, method: EqGainMethod) -> Result<()> {
        self.acquisition.set_eqgain_method(&self.config, method)
    }

    /// Returns the method that smooths the channel estimate.
    ///
    /// A polynomial order is reported as set, before any reduction.
    pub fn eqgain_method(&self) -> EqGainMethod {
        self.acquisition.eqgain_method()
    }

    /// Processes samples until one payload symbol is recovered or the input
    /// runs out.
    ///
    /// When a symbol is returned, finish with it and resume with
    /// `&input[output.consumed..]`. If it was the last symbol of the frame,
    /// call [`Self::reset`] before resuming.
    ///
    /// # Errors
    ///
    /// Returns an error when the channel fit at the third preamble symbol
    /// fails. Nonfinite samples cause this. So can a high
    /// [`EqGainMethod::Poly`] order when the pilot and data subcarriers span a
    /// narrow band. Earlier samples have been consumed by then, and the count
    /// is lost.
    pub fn execute<'a>(&'a mut self, input: &[Complex32]) -> Result<OfdmFrameSynchronizerOutput<'a>> {
        for (i, &xi) in input.iter().enumerate() {
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
                FrameSyncMode::Acquiring => match self.acquisition.execute(&self.config, &mut self.frontend)? {
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
                },
                FrameSyncMode::Receiving => self.payload.execute(&self.config, &mut self.frontend)?,
            };

            if symbol_ready {
                return Ok(OfdmFrameSynchronizerOutput {
                    consumed: i + 1,
                    symbol: Some(OfdmFrameSynchronizerSymbol {
                        subcarriers: &self.frontend.x_freq,
                        allocation: self.config.allocation(),
                    }),
                });
            }
        }

        Ok(OfdmFrameSynchronizerOutput { consumed: input.len(), symbol: None })
    }

    /// Processes samples and copies recovered payload symbols into
    /// `symbols` until it is full or the input runs out.
    ///
    /// With `M` subcarriers, symbol `i` is written to
    /// `symbols[i*M..(i+1)*M]`, so the length of `symbols` must be a multiple
    /// of `M`. An empty `symbols` consumes no input.
    ///
    /// Size `symbols` to no more than the symbols left in the frame, and call
    /// [`Self::reset`] after the last one. Otherwise samples after the frame
    /// are returned as more payload symbols.
    ///
    /// # Errors
    ///
    /// Returns an error when the length of `symbols` is not a multiple of `M`.
    /// The synchronizer and `symbols` are unchanged in that case. Also returns
    /// an error under the same condition as [`Self::execute`]. Earlier symbols
    /// may have been copied by then, and the counts are lost.
    pub fn execute_symbols_into(
        &mut self,
        input: &[Complex32],
        symbols: &mut [Complex32],
    ) -> Result<OfdmFrameSynchronizerBlockOutput> {
        let m = self.num_subcarriers();
        if !symbols.len().is_multiple_of(m) {
            return Err(Error::Config(format!(
                "ofdmframesync_execute_symbols_into(), output length must be a multiple of {}",
                m
            )));
        }

        let capacity = symbols.len() / m;
        let mut consumed = 0;
        let mut symbols_written = 0;

        while consumed < input.len() && symbols_written < capacity {
            let output = self.execute(&input[consumed..])?;
            consumed += output.consumed;

            let Some(symbol) = output.symbol else {
                break;
            };
            let lo = symbols_written * m;
            symbols[lo..lo + m].copy_from_slice(symbol.subcarriers);
            symbols_written += 1;
        }

        Ok(OfdmFrameSynchronizerBlockOutput { consumed, symbols_written })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{matrix_linsolve, MatrixRef};
    use crate::multichannel::{ofdmframe_init_default_sctype, OfdmFrameGenerator};
    use crate::random::randf;
    use approx::assert_abs_diff_eq;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use test_macro::autotest_annotate;

    fn new_framegen(
        num_subcarriers: usize,
        cp_len: usize,
        taper_len: usize,
        allocation: Option<&[SubcarrierType]>,
    ) -> Result<OfdmFrameGenerator> {
        let config = OfdmFrameConfig::new(num_subcarriers, cp_len, taper_len, allocation)?;
        OfdmFrameGenerator::new(&config)
    }

    fn new_framesync(
        num_subcarriers: usize,
        cp_len: usize,
        taper_len: usize,
        allocation: Option<&[SubcarrierType]>,
    ) -> Result<OfdmFrameSynchronizer> {
        let config = OfdmFrameConfig::new(num_subcarriers, cp_len, taper_len, allocation)?;
        OfdmFrameSynchronizer::new(&config)
    }

    fn make_frame(
        fg: &mut OfdmFrameGenerator,
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

    fn collect_symbols(fs: &mut OfdmFrameSynchronizer, samples: &[Complex32]) -> Vec<Vec<Complex32>> {
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
        let mut fg = OfdmFrameGenerator::new(&config).unwrap();
        let mut fs = OfdmFrameSynchronizer::new(&config).unwrap();

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
        let x: Vec<Complex32> =
            (0..m).map(|_| Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * randf())).collect();

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
        let mut q = OfdmFrameSynchronizer::new(&config).unwrap();

        assert!(!q.is_frame_open());
        q.set_cfo(0.0);
        assert_eq!(q.cfo(), 0.0);
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

    #[test]
    fn test_dft_eqgain_rejects_invalid_shapes_and_preserves_gain_on_error() {
        let allocation = vec![SubcarrierType::Data; 8];
        assert!(DftEqGainState::new(8, 0, &allocation).is_err());
        assert!(DftEqGainState::new(8, 9, &allocation).is_err());
        assert!(DftEqGainState::new(8, 2, &allocation[..7]).is_err());

        let mut state = DftEqGainState::new(8, 2, &allocation).unwrap();
        let mut short_gain = vec![Complex32::new(42.0, -7.0); 7];
        let original = short_gain.clone();
        assert!(state.estimate(&mut short_gain).is_err());
        assert_eq!(short_gain, original);

        let mut nonfinite_gain = vec![Complex32::new(1.0, 0.0); 8];
        nonfinite_gain[3].re = f32::NAN;
        let original = nonfinite_gain.clone();
        assert!(state.estimate(&mut nonfinite_gain).is_err());
        for (&actual, &expected) in nonfinite_gain.iter().zip(&original) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }

    fn estimate_dft_with_normal_equations(state: &DftEqGainState, gain: &[Complex32]) -> Vec<Complex32> {
        let m = state.num_subcarriers;
        let taps = state.num_taps;
        let basis = MatrixRef::new(&state.basis, m, taps).unwrap();
        let mut gram = vec![Complex32::new(0.0, 0.0); taps * taps];
        let mut rhs = vec![Complex32::new(0.0, 0.0); taps];
        for &subcarrier in &state.active {
            let basis_row = basis.row(subcarrier);
            for row in 0..taps {
                let left = basis_row[row];
                rhs[row] += left.conj() * gain[subcarrier];
                for column in 0..taps {
                    gram[row * taps + column] += left.conj() * basis_row[column];
                }
            }
        }
        let ridge = DftEqGainState::EQGAIN_RIDGE * state.active.len() as f32;
        for diagonal in 0..taps {
            gram[diagonal * taps + diagonal] += Complex32::new(ridge, 0.0);
        }

        let mut coefficients = vec![Complex32::new(0.0, 0.0); taps];
        matrix_linsolve(MatrixRef::new(&gram, taps, taps).unwrap(), &rhs, &mut coefficients, None).unwrap();
        let mut output = vec![Complex32::new(0.0, 0.0); m];
        for &subcarrier in &state.active {
            for (tap, &basis_value) in basis.row(subcarrier).iter().enumerate() {
                output[subcarrier] += coefficients[tap] * basis_value;
            }
        }
        output
    }

    #[test]
    fn test_dft_eqgain_augmented_qr_matches_normal_equations() {
        let m = 64;
        let mut allocation = vec![SubcarrierType::Null; m];
        ofdmframe_init_default_sctype(&mut allocation).unwrap();
        let mut state = DftEqGainState::new(m, 12, &allocation).unwrap();
        let mut gain: Vec<_> = (0..m)
            .map(|subcarrier| {
                let x = subcarrier as f32;
                Complex32::new((0.17 * x).sin() + 0.2, (0.11 * x).cos() - 0.1)
            })
            .collect();
        let expected = estimate_dft_with_normal_equations(&state, &gain);

        state.estimate(&mut gain).unwrap();

        for (&actual, &expected) in gain.iter().zip(&expected) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 2e-4);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 2e-4);
        }
    }

    #[test]
    fn test_dft_eqgain_reuses_factorization_without_stale_observations() {
        let m = 16;
        let allocation = vec![SubcarrierType::Data; m];
        let mut reused = DftEqGainState::new(m, 4, &allocation).unwrap();
        let mut first: Vec<_> = (0..m).map(|i| Complex32::new(i as f32 * 0.1, -0.25)).collect();
        reused.estimate(&mut first).unwrap();

        let second: Vec<_> = (0..m).map(|i| Complex32::new(-0.2, i as f32 * 0.05)).collect();
        let mut reused_output = second.clone();
        reused.estimate(&mut reused_output).unwrap();
        let mut fresh_output = second;
        DftEqGainState::new(m, 4, &allocation).unwrap().estimate(&mut fresh_output).unwrap();

        for (&actual, &expected) in reused_output.iter().zip(&fresh_output) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-6);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dft_eqgain_reduces_deterministic_noise() {
        let m = 64;
        let taps = 8;
        let mut allocation = vec![SubcarrierType::Null; m];
        ofdmframe_init_default_sctype(&mut allocation).unwrap();
        let mut state = DftEqGainState::new(m, taps, &allocation).unwrap();

        // arbitrarily choose some multipath degradation to apply here
        let multipath_taps = [
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.8, -0.1),
            Complex32::new(0.25, 0.15),
            Complex32::new(-0.1, 0.05),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ];

        let mut clean = vec![Complex32::new(0.0, 0.0); m];
        let basis = MatrixRef::new(&state.basis, m, taps).unwrap();
        for &subcarrier in &state.active {
            clean[subcarrier] = basis.row(subcarrier).iter().zip(&multipath_taps).map(|(&a, &x)| a * x).sum();
        }

        // noisy = clean + noisy channel
        let mut noisy = clean.clone();
        for &subcarrier in &state.active {
            let k = subcarrier as f32;
            noisy[subcarrier] += Complex32::new(0.01 * (0.71 * k).sin(), 0.01 * (0.37 * k).cos());
        }

        let input_error: f32 =
            state.active.iter().map(|&subcarrier| (noisy[subcarrier] - clean[subcarrier]).norm_sqr()).sum();

        let mut fitted = noisy.clone();
        state.estimate(&mut fitted).unwrap();

        let output_error: f32 =
            state.active.iter().map(|&subcarrier| (fitted[subcarrier] - clean[subcarrier]).norm_sqr()).sum();

        assert!(output_error < input_error, "channel fit should reduce observation noise");
        for (&actual, &expected) in state.taps.iter().zip(&multipath_taps) {
            assert!((actual - expected).norm() < 0.01, "recovered tap {actual} differs from {expected}");
        }

        let basis = MatrixRef::new(&state.basis, m, taps).unwrap();
        let ridge = DftEqGainState::EQGAIN_RIDGE * state.active.len() as f32;
        for tap in 0..taps {
            let mut residual = state.taps[tap] * ridge;
            for &subcarrier in &state.active {
                let row = basis.row(subcarrier);
                let fitted: Complex32 = row.iter().zip(&state.taps).map(|(&a, &x)| a * x).sum();
                residual += row[tap].conj() * (fitted - noisy[subcarrier]);
            }
            assert!(residual.norm() < 2e-4, "ridge optimality residual {residual}");
        }
    }

    fn ofdmframesync_test_cfo_noise(epsilon: f32, snr_db: f32, num_data: usize, pad: usize, rng: &mut impl Rng) {
        let m = 64;
        let cp_len = 8;
        let dphi = 2.0 * std::f32::consts::PI * epsilon / m as f32;
        let nstd = 10.0f32.powf(-snr_db / 20.0);

        let mut p = vec![SubcarrierType::Null; m];
        ofdmframe_init_default_sctype(&mut p).unwrap();
        let mut fg = new_framegen(m, cp_len, 0, Some(&p)).unwrap();
        let mut fs = new_framesync(m, cp_len, 0, Some(&p)).unwrap();

        assert_eq!(fs.cfo(), 0.0, "cfo starts at zero");

        let x: Vec<Complex32> =
            (0..m).map(|_| Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * rng.gen::<f32>())).collect();

        let frame = make_frame(&mut fg, m, cp_len, num_data, &x);

        // pad the front with silence to test rejection of false acquisitions
        let mut y = vec![Complex32::new(0.0, 0.0); frame.len() + pad];
        y[pad..].copy_from_slice(&frame);

        // apply noise and cfo
        for (i, v) in y.iter_mut().enumerate() {
            *v *= Complex32::from_polar(1.0, dphi * i as f32);
            let u1 = loop {
                let value = rng.gen::<f32>();
                if value != 0.0 {
                    break value;
                }
            };
            let u2 = rng.gen::<f32>();
            let noise = Complex32::from_polar((-2.0 * u1.ln()).sqrt(), 2.0 * std::f32::consts::PI * u2);
            *v += noise * nstd * std::f32::consts::FRAC_1_SQRT_2;
        }

        let symbols = collect_symbols(&mut fs, &y);

        assert!(!symbols.is_empty(), "no symbol recovered");
        assert!(symbols.len() == num_data, "wrong number of symbols received");

        assert_abs_diff_eq!(fs.cfo(), dphi, epsilon = 1e-2);

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
        assert!(rmse_0 < nstd + 0.5, "payload first rmse {rmse_0} at snr {snr_db} dB ({nstd})");
        let rmse_last = (err_last / c as f32).sqrt();
        assert!(rmse_last < nstd + 0.5, "payload last rmse {rmse_last} at snr {snr_db} dB ({nstd})");
    }

    #[test]
    fn test_ofdmframesync_cfo_noise() {
        let mut rng = StdRng::seed_from_u64(0x1234_5678_09ab_cdef);
        for epsilon in [0.2f32, 0.3f32, 0.4f32] {
            for snr_db in [60.0f32, 30.0, 20.0, 10.0] {
                for num_data in [1, 2, 4, 8, 16, 32] {
                    for pad in [0, 3, 7, 19, 251, 1021] {
                        ofdmframesync_test_cfo_noise(epsilon, snr_db, num_data, pad, &mut rng);
                    }
                }
            }
        }
    }

    fn ofdmframesync_test_delay(delay: usize, gain: f32, method: EqGainMethod, rmse_max: f32) {
        let m = 64;
        let cp_len = 16;

        let mut p = vec![SubcarrierType::Null; m];
        ofdmframe_init_default_sctype(&mut p).unwrap();

        let payload: Vec<Complex32> = (0..m)
            .map(|i| {
                let a = if i % 2 == 0 { 1.0 } else { -1.0 };
                let b = if (i / 5) % 2 == 0 { 1.0 } else { -1.0 };
                Complex32::new(a * std::f32::consts::FRAC_1_SQRT_2, b * std::f32::consts::FRAC_1_SQRT_2)
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
            EqGainMethod::Dft { num_taps } => format!("dft-{num_taps}"),
        };
        let delay_s = if gain < 1.0 { "delay" } else { "lead" };
        assert!(
            rmse < rmse_max,
            "ofdmframesync method ({}) {} {}: rmse actual {} >= expected {}",
            method_s,
            delay_s,
            delay,
            rmse,
            rmse_max
        );
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
        assert_eq!(empty, OfdmFrameSynchronizerBlockOutput { consumed: 0, symbols_written: 0 });

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
