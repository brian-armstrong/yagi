//
// Numerically-controlled synthesizer (direct digital synthesis)
// with internal phase-locked loop (pll) implementation
//

use num_complex::Complex32;
use std::f32::consts::PI;

use crate::error::{Error, Result};

const SYNTH_PLL_BANDWIDTH_DEFAULT: f32 = 0.1;

/// Numerically-controlled synthesizer driven by an arbitrary lookup table
///
/// Unlike [`Osc`](super::Osc), which generates a sinusoid from a fixed-point
/// phase accumulator, `Synth` walks a caller-supplied table of complex samples.
/// That makes it a spreading-sequence generator: with the default frequency it
/// visits each table entry exactly once per cycle, and [`Self::despread`]
/// correlates a received block against the sequence.
///
/// The table is sampled at the *nearest* index rather than interpolated, so
/// [`Self::get_current`] is always one of the table entries. The half-sample
/// outputs are midpoints of adjacent entries, used for early/late timing
/// discrimination in [`Self::despread_triple`].
#[derive(Debug, Clone)]
pub struct Synth {
    theta: f32,   // phase
    d_theta: f32, // frequency
    tab: Vec<Complex32>,
    index: usize, // table index

    prev_half: Complex32,
    current: Complex32,
    next_half: Complex32,

    // phase-locked loop
    alpha: f32,
    beta: f32,
}

impl Synth {
    /// create synth object from a table of complex samples
    ///
    /// The initial frequency visits each table entry once per cycle, i.e.
    /// `2*pi/length`.
    pub fn new(table: &[Complex32]) -> Result<Self> {
        if table.is_empty() {
            return Err(Error::Config("synth table length must be greater than zero".into()));
        }

        let mut q = Self {
            theta: 0.0,
            d_theta: 0.0,
            tab: table.to_vec(),
            index: 0,
            prev_half: Complex32::new(0.0, 0.0),
            current: Complex32::new(0.0, 0.0),
            next_half: Complex32::new(0.0, 0.0),
            alpha: 0.0,
            beta: 0.0,
        };

        // set default pll bandwidth
        q.pll_set_bandwidth(SYNTH_PLL_BANDWIDTH_DEFAULT)?;

        // reset object
        q.reset();

        // default frequency is to visit each sample once
        q.set_frequency(2.0 * PI / table.len() as f32);

        Ok(q)
    }

    /// reset synth object's internals
    pub fn reset(&mut self) {
        self.theta = 0.0;
        self.d_theta = 0.0;

        // reset table index
        self.index = 0;

        // reset pll filter state
        self.pll_reset();

        self.compute_synth();
    }

    /// set frequency
    pub fn set_frequency(&mut self, f: f32) {
        self.d_theta = f;
    }

    /// adjust frequency
    pub fn adjust_frequency(&mut self, df: f32) {
        self.d_theta += df;
    }

    /// set phase
    pub fn set_phase(&mut self, phi: f32) {
        self.theta = phi;
        self.constrain_phase();
        self.compute_synth();
    }

    /// adjust phase
    pub fn adjust_phase(&mut self, dphi: f32) {
        self.theta += dphi;
        self.constrain_phase();
    }

    /// increment internal phase
    pub fn step(&mut self) {
        self.theta += self.d_theta;
        self.constrain_phase();
        self.compute_synth();
    }

    /// get phase
    pub fn get_phase(&self) -> f32 {
        self.theta
    }

    /// get frequency
    pub fn get_frequency(&self) -> f32 {
        self.d_theta
    }

    /// get table length
    pub fn get_length(&self) -> usize {
        self.tab.len()
    }

    /// get table value at the current phase
    pub fn get_current(&self) -> Complex32 {
        self.current
    }

    /// get midpoint between the current value and the previous table entry
    pub fn get_half_previous(&self) -> Complex32 {
        self.prev_half
    }

    /// get midpoint between the current value and the next table entry
    pub fn get_half_next(&self) -> Complex32 {
        self.next_half
    }

    // pll methods

    /// reset pll state, retaining base frequency
    pub fn pll_reset(&mut self) {}

    /// set pll bandwidth
    pub fn pll_set_bandwidth(&mut self, bandwidth: f32) -> Result<()> {
        // validate input
        if bandwidth < 0.0 {
            return Err(Error::Range("synth pll bandwidth must be positive".into()));
        }

        self.alpha = bandwidth; // frequency proportion
        self.beta = self.alpha.sqrt(); // phase proportion
        Ok(())
    }

    /// advance pll phase given a phase error
    pub fn pll_step(&mut self, dphi: f32) {
        // increase frequency proportional to error
        self.adjust_frequency(dphi * self.alpha);

        // increase phase proportional to error
        self.adjust_phase(dphi * self.beta);

        self.compute_synth();
    }

    // mixing functions

    /// rotate input sample up by the current table value (no stepping)
    pub fn mix_up(&self, x: Complex32) -> Complex32 {
        x * self.current
    }

    /// rotate input sample down by the current table value (no stepping)
    pub fn mix_down(&self, x: Complex32) -> Complex32 {
        x * self.current.conj()
    }

    /// rotate input block up by the table value, stepping each sample
    pub fn mix_block_up(&mut self, x: &[Complex32], y: &mut [Complex32]) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::Range("input and output must have the same length".into()));
        }
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            // mix single sample up
            *yi = self.mix_up(*xi);

            // step synth phase
            self.step();
        }
        Ok(())
    }

    /// rotate input block down by the table value, stepping each sample
    pub fn mix_block_down(&mut self, x: &[Complex32], y: &mut [Complex32]) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::Range("input and output must have the same length".into()));
        }
        for (xi, yi) in x.iter().zip(y.iter_mut()) {
            // mix single sample down
            *yi = self.mix_down(*xi);

            // step synth phase
            self.step();
        }
        Ok(())
    }

    /// spread a single symbol across a full table cycle
    ///
    /// `y` must hold at least [`Self::get_length`] samples.
    pub fn spread(&mut self, x: Complex32, y: &mut [Complex32]) -> Result<()> {
        if y.len() < self.tab.len() {
            return Err(Error::Range("synth spread output too small".into()));
        }

        for yi in y[..self.tab.len()].iter_mut() {
            *yi = self.mix_up(x);

            self.step();
        }
        Ok(())
    }

    /// despread a full table cycle back into a single symbol
    ///
    /// `x` must hold at least [`Self::get_length`] samples.
    ///
    /// The correlation is normalized by `sum(|x|*|tab|)` rather than by the code
    /// energy, so the result carries the symbol's *phase* at unit magnitude.
    pub fn despread(&mut self, x: &[Complex32]) -> Result<Complex32> {
        if x.len() < self.tab.len() {
            return Err(Error::Range("synth despread input too small".into()));
        }

        let mut despread = Complex32::new(0.0, 0.0);
        let mut sum = 0.0f32;
        for &xi in &x[..self.tab.len()] {
            let temp = self.mix_down(xi);

            despread += temp;
            sum += xi.norm() * self.current.norm();

            self.step();
        }
        Ok(despread / sum)
    }

    /// despread with early, punctual and late correlators
    ///
    /// The early and late outputs use the half-sample midpoints, giving a timing
    /// discriminant. `x` must hold at least [`Self::get_length`] samples.
    ///
    /// Returns `(early, punctual, late)`.
    pub fn despread_triple(&mut self, x: &[Complex32]) -> Result<(Complex32, Complex32, Complex32)> {
        if x.len() < self.tab.len() {
            return Err(Error::Range("synth despread input too small".into()));
        }

        let mut despread_early = Complex32::new(0.0, 0.0);
        let mut despread_punctual = Complex32::new(0.0, 0.0);
        let mut despread_late = Complex32::new(0.0, 0.0);

        let mut sum_early = 0.0f32;
        let mut sum_punctual = 0.0f32;
        let mut sum_late = 0.0f32;

        for &xi in &x[..self.tab.len()] {
            despread_early += xi * self.prev_half.conj();
            despread_punctual += xi * self.current.conj();
            despread_late += xi * self.next_half.conj();

            sum_early += xi.norm() * self.prev_half.norm();
            sum_punctual += xi.norm() * self.current.norm();
            sum_late += xi.norm() * self.next_half.norm();

            self.step();
        }

        Ok((
            despread_early / sum_early,
            despread_punctual / sum_punctual,
            despread_late / sum_late
        ))
    }

    //
    // internal methods
    //

    /// constrain frequency of synth object to be in (-pi,pi)
    #[allow(dead_code)]
    fn constrain_frequency(&mut self) {
        if self.d_theta > PI {
            self.d_theta -= 2.0 * PI;
        } else if self.d_theta < -PI {
            self.d_theta += 2.0 * PI;
        }
    }

    // constrain phase of synth object to be in (-pi,pi)
    fn constrain_phase(&mut self) {
        // do this constrain in f64. f32::PI is > f64::PI, so this has different
        // wrapping behavior at theta == f32::PI
        let theta = self.theta as f64;
        if theta > std::f64::consts::PI {
            self.theta = (theta - 2.0 * std::f64::consts::PI) as f32;
        } else if theta < -std::f64::consts::PI {
            self.theta = (theta + 2.0 * std::f64::consts::PI) as f32;
        }
    }

    fn compute_synth(&mut self) {
        // assume phase is constrained to be in (-pi,pi)
        let length = self.tab.len();

        // f64 needed here for correct index selection
        let index = self.theta as f64 * length as f64 / (2.0 * std::f64::consts::PI) + 2.0 * length as f64;
        self.index = ((index as f32 + 0.5) as usize) % length;
        debug_assert!(self.index < length);

        let prev_index = (self.index + length - 1) % length;
        let next_index = (self.index + 1) % length;

        self.current = self.tab[self.index];
        let prev = self.tab[prev_index];
        let next = self.tab[next_index];

        self.prev_half = (self.current + prev) / 2.0;
        self.next_half = (self.current + next) / 2.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::sequence::MSequence;

    // unit-magnitude DFT-like sequence: tab[i] = exp(j*2*pi*i/n)
    fn tab_cexp(n: usize) -> Vec<Complex32> {
        (0..n).map(|i| Complex32::from_polar(1.0, 2.0 * PI * i as f32 / n as f32)).collect()
    }

    // a real +/-1 spreading code
    fn tab_bpsk(m: usize) -> Vec<Complex32> {
        let mut ms = MSequence::create_default(m as u32).unwrap();
        let n = ms.get_length() as usize;
        (0..n)
            .map(|_| {
                let chip = if ms.advance() == 0 { -1.0 } else { 1.0 };
                Complex32::new(chip, 0.0)
            })
            .collect()
    }

    #[test]
    fn test_synth_config() {
        assert!(Synth::new(&[]).is_err());

        let tab = tab_cexp(8);
        let q = Synth::new(&tab).unwrap();
        assert_eq!(q.get_length(), 8);

        // default frequency visits each entry once per cycle
        assert_abs_diff_eq!(q.get_frequency(), 2.0 * PI / 8.0, epsilon = 1e-6);
        assert_abs_diff_eq!(q.get_phase(), 0.0, epsilon = 1e-6);

        // negative pll bandwidth is rejected, and leaves the object usable
        let mut q = Synth::new(&tab).unwrap();
        assert!(q.pll_set_bandwidth(-1.0).is_err());
        assert!(q.pll_set_bandwidth(0.0).is_ok());

        // block mixers require matching lengths
        let x = [Complex32::new(1.0, 0.0); 4];
        let mut y = [Complex32::new(0.0, 0.0); 3];
        assert!(q.mix_block_up(&x, &mut y).is_err());
        assert!(q.mix_block_down(&x, &mut y).is_err());

        // spread/despread need a full table cycle
        let mut short = [Complex32::new(0.0, 0.0); 7];
        assert!(q.spread(Complex32::new(1.0, 0.0), &mut short).is_err());
        assert!(q.despread(&short).is_err());
        assert!(q.despread_triple(&short).is_err());
    }

    #[test]
    fn test_synth_step_walks_table() {
        for n in [4usize, 8, 16] {
            let tab = tab_cexp(n);
            let mut q = Synth::new(&tab).unwrap();

            for i in 0..3 * n {
                let expected = tab[i % n];
                let c = q.get_current();
                assert_abs_diff_eq!(c.re, expected.re, epsilon = 1e-5);
                assert_abs_diff_eq!(c.im, expected.im, epsilon = 1e-5);
                q.step();
            }
        }
    }

    #[test]
    fn test_synth_spread_despread_roundtrip() {
        for tab in [tab_cexp(8), tab_cexp(16), tab_bpsk(6)] {
            let n = tab.len();
            for sym in [Complex32::new(1.0, 0.0), Complex32::new(0.5, 0.25), Complex32::new(-0.75, 0.6)] {
                let mut tx = Synth::new(&tab).unwrap();
                let mut chips = vec![Complex32::new(0.0, 0.0); n];
                tx.spread(sym, &mut chips).unwrap();

                // spreading is the symbol times each table entry
                for i in 0..n {
                    let e = sym * tab[i];
                    assert_abs_diff_eq!(chips[i].re, e.re, epsilon = 1e-5);
                    assert_abs_diff_eq!(chips[i].im, e.im, epsilon = 1e-5);
                }

                // despread normalizes by sum(|x|*|tab|), so amplitude drops out
                // and only the phase survives
                let mut rx = Synth::new(&tab).unwrap();
                let y = rx.despread(&chips).unwrap();
                let expected = sym / sym.norm();
                assert_abs_diff_eq!(y.re, expected.re, epsilon = 1e-4);
                assert_abs_diff_eq!(y.im, expected.im, epsilon = 1e-4);
                assert_abs_diff_eq!(y.norm(), 1.0, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_synth_despread_triple() {
        for tab in [tab_bpsk(4), tab_bpsk(6), tab_bpsk(8)] {
            for sym in [Complex32::new(1.0, 0.0), Complex32::new(1.0, 1.0), Complex32::new(0.0, 1.0), Complex32::new(-1.0, 0.0)] {
                let n = tab.len();
                let mut tx = Synth::new(&tab).unwrap();
                let mut chips = vec![Complex32::new(0.0, 0.0); n];
                tx.spread(sym, &mut chips).unwrap();

                let mut rx = Synth::new(&tab).unwrap();
                let expected = sym / sym.norm();

                let (early, punctual, late) = rx.despread_triple(&chips).unwrap();
                assert_abs_diff_eq!(punctual.re, expected.re, epsilon = 1e-4);
                assert_abs_diff_eq!(punctual.im, expected.im, epsilon = 1e-4);
                assert_abs_diff_eq!(early.norm(), late.norm(), epsilon = 1e-4);

                // rotate observed chips right by 1, which makes our receiver see an early code
                chips.rotate_right(1);
                let (early, _punctual, late) = rx.despread_triple(&chips).unwrap();
                assert_abs_diff_eq!(early.re, expected.re, epsilon = 1e-4);
                assert_abs_diff_eq!(early.im, expected.im, epsilon = 1e-4);
                assert!(early.norm() > late.norm());

                // undo the rotation
                chips.rotate_left(1);

                // now rotate left by 1, which gives us a late code
                chips.rotate_left(1);
                let (early, _punctual, late) = rx.despread_triple(&chips).unwrap();
                assert_abs_diff_eq!(late.re, expected.re, epsilon = 1e-4);
                assert_abs_diff_eq!(late.im, expected.im, epsilon = 1e-4);
                assert!(late.norm() > early.norm());
            }
        }
    }

    #[test]
    fn test_synth_despread_rejects_wrong_code() {
        let tab = tab_bpsk(6);
        let n = tab.len();

        // a different code of the same length
        let other: Vec<Complex32> = (0..n).map(|i| Complex32::new(if (i * 3 + 1) % 5 > 2 { 1.0 } else { -1.0 }, 0.0)).collect();

        let sym = Complex32::new(1.0, 0.0);

        let mut matched = Synth::new(&tab).unwrap();
        let mut chips = vec![Complex32::new(0.0, 0.0); n];
        matched.spread(sym, &mut chips).unwrap();

        // despread the matched code with the wrong local sequence
        let mut wrong = Synth::new(&other).unwrap();
        let y = wrong.despread(&chips).unwrap();
        assert!(y.norm() < 0.5, "wrong-code correlation = {}", y.norm());

        // and the right one recovers it
        let mut right = Synth::new(&tab).unwrap();
        let y = right.despread(&chips).unwrap();
        assert_abs_diff_eq!(y.norm(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_synth_pll_tracks_frequency_offset() {
        let n = 16;
        let tab = tab_cexp(n);
        let mut q = Synth::new(&tab).unwrap();
        q.pll_set_bandwidth(0.02).unwrap();

        // run the synth against a reference phase advancing slightly faster
        let f_ref = 2.0 * PI / n as f32 * 1.02;
        let mut theta_ref = 0.0f32;

        for _ in 0..4000 {
            let mut err = theta_ref - q.get_phase();
            while err > PI {
                err -= 2.0 * PI;
            }
            while err < -PI {
                err += 2.0 * PI;
            }

            q.pll_step(err * 0.05);
            q.step();

            theta_ref += f_ref;
            while theta_ref > PI {
                theta_ref -= 2.0 * PI;
            }
        }

        assert_abs_diff_eq!(q.get_frequency(), f_ref, epsilon = 0.02);
    }
}
