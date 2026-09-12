use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::Result;
use crate::filter;

use num_complex::ComplexFloat;

use super::polyphase_bank::FirPolyphaseFilterBank;

/// Finite impulse response (FIR) polyphase filter with internal sample history.
#[derive(Clone, Debug)]
#[doc(alias = "FirPfbFilter")]
pub struct FirPolyphaseFilter<T, Coeff = T> {
    w: Window<T>,
    bank: FirPolyphaseFilterBank<T, Coeff>,
}

impl<T, Coeff> FirPolyphaseFilter<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + Default,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Create a polyphase filter with internal sample history.
    pub fn new(num_filters: usize, coefficients: &[Coeff]) -> Result<Self> {
        Self::from_bank(FirPolyphaseFilterBank::new(num_filters, coefficients)?)
    }

    /// Create a Kaiser-designed polyphase filter with default parameters.
    pub fn new_kaiser_simple(num_filters: usize, filter_delay: usize) -> Result<Self> {
        Self::from_bank(FirPolyphaseFilterBank::new_kaiser_simple(num_filters, filter_delay)?)
    }

    /// Create a Kaiser-designed polyphase filter.
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

    /// Create a root-Nyquist polyphase filter.
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

    /// Create a derivative root-Nyquist polyphase filter.
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

    /// Wrap a coefficient bank with newly reset internal history.
    pub fn from_bank(bank: FirPolyphaseFilterBank<T, Coeff>) -> Result<Self> {
        let w = Window::new(bank.filter_len())?;
        Ok(Self { w, bank })
    }

    /// Reset the internal sample history.
    pub fn reset(&mut self) {
        self.w.reset();
    }

    /// Returns the underlying coefficient bank.
    pub fn bank(&self) -> &FirPolyphaseFilterBank<T, Coeff> {
        &self.bank
    }

    /// Returns the underlying coefficient bank mutably.
    pub fn bank_mut(&mut self) -> &mut FirPolyphaseFilterBank<T, Coeff> {
        &mut self.bank
    }

    /// Returns the number of coefficient phases in the bank.
    pub fn num_filters(&self) -> usize {
        self.bank.num_filters()
    }

    /// Returns the number of samples retained in the internal history.
    pub fn filter_len(&self) -> usize {
        self.bank.filter_len()
    }

    /// Replace the coefficients without clearing the internal sample history.
    pub fn set_coefficients(&mut self, coefficients: &[Coeff]) -> Result<()> {
        self.bank.set_coefficients(coefficients)
    }

    /// Set the output scaling for the filter bank.
    pub fn set_scale(&mut self, scale: Coeff) {
        self.bank.set_scale(scale);
    }

    /// Returns the output scaling for the filter bank.
    pub fn scale(&self) -> Coeff {
        self.bank.scale()
    }

    /// Push one input sample into the filter history.
    pub fn push(&mut self, input: T) {
        self.w.push(input)
    }

    /// Write a block of samples into the filter history.
    pub fn write(&mut self, input: &[T]) {
        self.w.write(input)
    }

    /// Execute one phase of the filter bank.
    pub fn execute(&mut self, phase: usize) -> Result<T> {
        self.bank.execute(phase, self.w.read())
    }

    /// Execute one phase for a block of input samples.
    pub fn execute_block(&mut self, phase: usize, input: &[T], output: &mut [T]) -> Result<()> {
        let (h, block_h) = self.bank.phase_coefficients(phase)?;
        let num_samples = input.len().min(output.len());
        let bank = &self.bank;
        self.w.execute_block_contiguous(&input[..num_samples], |indices, history| {
            bank.execute_block_with_coefficients(history, &mut output[indices], h, block_h);
        });
        Ok(())
    }
}
