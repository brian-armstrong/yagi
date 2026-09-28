use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{self, Sample};

use super::polyphase_bank::FirPolyphaseFilterBank;

/// Polyphase finite impulse response filter that keeps its own sample history.
///
/// This pairs a [`FirPolyphaseFilterBank`] with a history of
/// [`Self::filter_len`] samples. Add samples with [`Self::push`] or
/// [`Self::write`], then evaluate any phase against the current history with
/// [`Self::execute`]. Evaluating a phase does not change the history. A new
/// filter starts with zero-valued history.
#[derive(Clone, Debug)]
#[doc(alias = "FirPfbFilter")]
pub struct FirPolyphaseFilter<T, Coeff = T> {
    w: Window<T>,
    bank: FirPolyphaseFilterBank<T, Coeff>,
}

impl<T, Coeff> FirPolyphaseFilter<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a polyphase filter from existing prototype coefficients.
    ///
    /// The coefficients are split among `num_filters` phases as described in
    /// [`FirPolyphaseFilterBank::new`].
    ///
    /// # Errors
    ///
    /// Returns an error when `num_filters` is zero or `coefficients` is empty.
    pub fn new(num_filters: usize, coefficients: &[Coeff]) -> Result<Self> {
        Self::from_bank(FirPolyphaseFilterBank::new(num_filters, coefficients)?)
    }

    /// Creates a Kaiser-windowed polyphase filter with typical settings.
    ///
    /// See [`FirPolyphaseFilterBank::new_kaiser_simple`] for the design.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_filters` or `filter_delay` is zero.
    pub fn new_kaiser_simple(num_filters: usize, filter_delay: usize) -> Result<Self> {
        Self::from_bank(FirPolyphaseFilterBank::new_kaiser_simple(num_filters, filter_delay)?)
    }

    /// Creates a Kaiser-windowed polyphase filter.
    ///
    /// See [`FirPolyphaseFilterBank::new_kaiser`] for the design and the units
    /// of each parameter.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`FirPolyphaseFilterBank::new_kaiser`].
    pub fn new_kaiser(
        num_filters: usize,
        filter_delay: usize,
        cutoff_frequency: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        Self::from_bank(FirPolyphaseFilterBank::new_kaiser(
            num_filters,
            filter_delay,
            cutoff_frequency,
            stopband_attenuation,
        )?)
    }

    /// Creates a polyphase filter from a pulse-shaping prototype.
    ///
    /// See [`FirPolyphaseFilterBank::new_rnyquist`] for the design and the
    /// units of each parameter.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`FirPolyphaseFilterBank::new_rnyquist`].
    pub fn new_rnyquist(
        filter_shape: filter::FirFilterShape,
        num_filters: usize,
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
    ) -> Result<Self> {
        Self::from_bank(FirPolyphaseFilterBank::new_rnyquist(
            filter_shape,
            num_filters,
            samples_per_symbol,
            filter_delay,
            excess_bandwidth,
        )?)
    }

    /// Creates a polyphase filter from the derivative of a pulse-shaping prototype.
    ///
    /// See [`FirPolyphaseFilterBank::new_drnyquist`] for the design and its
    /// scaling.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`FirPolyphaseFilterBank::new_drnyquist`].
    pub fn new_drnyquist(
        filter_shape: filter::FirFilterShape,
        num_filters: usize,
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
    ) -> Result<Self> {
        Self::from_bank(FirPolyphaseFilterBank::new_drnyquist(
            filter_shape,
            num_filters,
            samples_per_symbol,
            filter_delay,
            excess_bandwidth,
        )?)
    }

    /// Creates a polyphase filter from an existing bank.
    ///
    /// The filter starts with zero-valued history and keeps the bank's scale.
    pub fn from_bank(bank: FirPolyphaseFilterBank<T, Coeff>) -> Result<Self> {
        let w = Window::new(bank.filter_len())?;
        Ok(Self { w, bank })
    }

    /// Clears the sample history to zeros.
    ///
    /// The coefficients and scale are unchanged.
    pub fn reset(&mut self) {
        self.w.reset();
    }

    /// Returns the coefficient bank.
    pub fn bank(&self) -> &FirPolyphaseFilterBank<T, Coeff> {
        &self.bank
    }

    /// Returns the coefficient bank for modification.
    ///
    /// The bank cannot change its phase length, so the history stays valid.
    pub fn bank_mut(&mut self) -> &mut FirPolyphaseFilterBank<T, Coeff> {
        &mut self.bank
    }

    /// Returns the number of phases.
    pub fn num_filters(&self) -> usize {
        self.bank.num_filters()
    }

    /// Returns the number of taps in each phase.
    ///
    /// This is also the number of samples the filter keeps in its history.
    pub fn filter_len(&self) -> usize {
        self.bank.filter_len()
    }

    /// Replaces the prototype coefficients and keeps the sample history.
    ///
    /// The new coefficients must produce the same phase length, as described
    /// in [`FirPolyphaseFilterBank::set_coefficients`].
    ///
    /// # Errors
    ///
    /// Returns an error when the coefficients produce a different phase length.
    /// The filter is unchanged on error.
    pub fn set_coefficients(&mut self, coefficients: &[Coeff]) -> Result<()> {
        self.bank.set_coefficients(coefficients)
    }

    /// Sets the factor applied to every output sample.
    pub fn set_scale(&mut self, scale: Coeff) {
        self.bank.set_scale(scale);
    }

    /// Returns the factor applied to every output sample.
    pub fn scale(&self) -> Coeff {
        self.bank.scale()
    }

    /// Adds one sample to the history and drops the oldest.
    pub fn push(&mut self, input: T) {
        self.w.push(input)
    }

    /// Adds every sample of `input` to the history in order.
    ///
    /// This is the same as calling [`Self::push`] for each sample.
    pub fn write(&mut self, input: &[T]) {
        self.w.write(input)
    }

    /// Evaluates one phase against the current history and returns one output.
    ///
    /// The history is not changed.
    ///
    /// # Errors
    ///
    /// Returns an error when `phase` is not less than [`Self::num_filters`].
    pub fn execute(&mut self, phase: usize) -> Result<T> {
        self.bank.execute(phase, self.w.read())
    }

    /// Pushes each sample of `input` and evaluates one phase after each push.
    ///
    /// `output[j]` is the result after `input[j]` is added. `output` must hold
    /// at least `input.len()` samples, and extra output is left unchanged.
    /// Returns `input.len()`.
    ///
    /// The result is the same as calling [`Self::push`] and [`Self::execute`]
    /// for each sample, up to rounding. Block processing can sum the products
    /// in a different order, so the last bits can differ.
    ///
    /// # Errors
    ///
    /// Returns an error when `phase` is not less than [`Self::num_filters`] or
    /// `output` is shorter than `input`. The filter and `output` are unchanged
    /// on error.
    pub fn execute_block(&mut self, phase: usize, input: &[T], output: &mut [T]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output block length must be at least the input length".into()));
        }
        let (h, block_h) = self.bank.phase_coefficients(phase)?;
        let bank = &self.bank;
        self.w.execute_block_contiguous(input, |indices, history| {
            bank.execute_block_with_coefficients(history, &mut output[indices], h, block_h);
        });
        Ok(input.len())
    }
}
