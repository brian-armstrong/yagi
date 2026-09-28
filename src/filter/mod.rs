//! Stateful filters, filter design, and sample-rate conversion.
//!
//! FIR filters such as [`FirFilter`] convolve the input with a fixed set of
//! coefficients. IIR filters such as [`IirFilter`] also feed back past
//! outputs, which gives a sharper response for the same amount of work but
//! a nonlinear phase. The interpolation and decimation filters and the
//! resamplers change the sample rate as they filter. Most objects design
//! their own coefficients through a `new_*` constructor, such as
//! [`FirFilter::new_kaiser`], or accept coefficients from the functions in
//! [`fir::design`] and [`iir::design`].
//!
//! An object keeps its recent input, and any feedback state, between calls,
//! so a long stream can be processed in pieces of any size. Its `reset`
//! method clears that history so the next input starts a new stream. Check
//! an object's `reset` docs for any settings it also restores.
//!
//! The processing methods follow a few patterns, although not every object
//! offers all of them:
//!
//! - Filters that read a sample history, such as [`FirFilter`] and
//!   [`FractionalDelay`], separate input from output. `push` adds one sample
//!   to the history and `write` adds a slice of them. `execute` then returns
//!   the output for the current history. Some also offer `execute_one`,
//!   which pushes a sample and returns the new output.
//! - Filters with feedback, such as [`IirFilter`], take the input directly.
//!   `execute` consumes one sample and returns one output.
//! - Rate changers take one unit of input per `execute` call. An
//!   interpolator takes one sample and writes a fixed number of outputs, a
//!   decimator takes one group of samples and returns one output, and
//!   [`ArbitraryResampler`] writes as many outputs as the sample produces.
//! - `execute_block` processes a whole slice and writes the outputs to
//!   another slice. It returns the number of outputs written. Size the
//!   output from the object's rate or the count the method documents.
//! - Objects that work in two directions name their methods by direction.
//!   [`HalfBandResampler`] has `interp_execute` and `decim_execute`, and the
//!   Hilbert transforms add `r2c_execute` and `c2r_execute`. Most of these
//!   also have a `_block` form.
//!
//! A few objects differ. [`FftFilter`] processes one fixed-size block per
//! call, and the `execute` of [`SymbolSynchronizer`] takes a whole slice.
//! Each object's docs give its exact calls.
//!
//! Frequencies are normalized to the relevant sample rate and given in cycles
//! per sample, unless an item states another unit. FIR coefficients are in
//! time order, with the first coefficient applied to the newest sample.
//!
//! Generic filters work on `f32` or `Complex32` samples and coefficients. See
//! [`Sample`].

mod autocorrelator;
mod direct_digital_synthesizer;
mod fft_filter;
#[path = "fir/mod.rs"]
mod fir_impl;
mod fractional_delay;
#[path = "iir/mod.rs"]
mod iir_impl;
#[path = "lpc.rs"]
mod lpc_impl;
mod order_statistic_filter;
mod resampler;
mod sample;
mod symbol_synchronizer;

// For the sake of keeping exports clean, create these sort of faux pub mods below
// When adding anything new to filter, remember to update these as well.

/// Finite impulse response filter design.
pub mod fir {
    pub mod design {
        //! FIR coefficient design and response analysis.
        //!
        //! Frequencies are normalized to the sample rate and given in cycles per
        //! sample. Most low-pass designs accept cutoffs from zero to 0.5.
        //! Pulse-shaping designs take `k` samples per symbol and a delay of `m`
        //! symbols, and return `2 * k * m + 1` taps.
        pub use super::super::fir_impl::design::{
            estimate_req_filter_len, estimate_req_filter_len_herrmann, estimate_req_filter_len_kaiser,
            estimate_req_filter_stopband_attenuation, estimate_req_filter_transition_bandwidth, filter_autocorr,
            filter_crosscorr, filter_energy, filter_isi, fir_design_arkaiser, fir_design_doppler, fir_design_farcsech,
            fir_design_farcsech_freqresponse, fir_design_fexp, fir_design_fexp_freqresponse, fir_design_fnyquist,
            fir_design_fsech, fir_design_fsech_freqresponse, fir_design_gmskrx, fir_design_gmsktx, fir_design_hm3,
            fir_design_kaiser, fir_design_notch, fir_design_pm, fir_design_pm_halfband_ft,
            fir_design_pm_halfband_stopband_attenuation, fir_design_pm_lowpass, fir_design_prototype, fir_design_rcos,
            fir_design_rfarcsech, fir_design_rfexp, fir_design_rfsech, fir_design_rkaiser, fir_design_rrcos,
            fir_design_windowf, fir_group_delay, freqrespcf, freqrespf, freqresponse, kaiser_beta_stopband_attenuation,
            FirPmWeightType, ParksMcClellanDesigner, ParksMcClellanResponse,
        };
        pub use super::super::fir_impl::{
            fir_filter_design_dc_blocker, fir_filter_design_firdespm, fir_filter_design_kaiser,
            fir_filter_design_notch, fir_filter_design_rect, fir_filter_design_rnyquist, fir_filter_freqresponse,
            fir_filter_get_psd, fir_filter_groupdelay,
        };
    }
}

/// Infinite impulse response filter design.
pub mod iir {
    pub mod design {
        //! IIR coefficient design and response analysis.
        //!
        //! Frequencies are normalized to the sample rate and given in cycles per
        //! sample. Transfer-function coefficients are in ascending powers of
        //! `z^-1`. Each second-order section has three numerator and three
        //! denominator coefficients.
        //!
        //! The analog prototype designers return low-pass poles and zeros.
        //! [`iir_design`] and [`iir_filter_design_prototype`] turn a prototype
        //! into digital transfer-function or second-order-section coefficients.
        pub use super::super::iir_impl::design::{
            iir_design, iir_design_bessel_analog, iir_design_bilinear_a2d, iir_design_bilinear_z,
            iir_design_butter_analog, iir_design_cheby1_analog, iir_design_cheby2_analog, iir_design_d2sos,
            iir_design_d2tf, iir_design_ellip_analog, iir_design_freqprewarp, iir_design_is_stable, iir_design_lp2bp,
            iir_design_lp2hp, iir_design_pll_active_lag, iir_design_pll_active_pi, iir_group_delay,
        };
        pub use super::super::iir_impl::{
            iir_filter_design_dc_blocker, iir_filter_design_dc_blocker_scale, iir_filter_design_differentiator,
            iir_filter_design_integrator, iir_filter_design_lowpass, iir_filter_design_pll,
            iir_filter_design_prototype, iir_filter_freqresponse, iir_filter_freqresponse_sos, iir_filter_get_psd,
            iir_filter_get_psd_sos, iir_filter_groupdelay, iir_filter_groupdelay_sos,
        };
    }
}

pub mod lpc {
    //! Linear prediction coefficient design.
    pub use super::lpc_impl::design_lpc as design;
}

pub use autocorrelator::Autocorrelator;
pub use direct_digital_synthesizer::DirectDigitalSynthesizer;
pub use fft_filter::FftFilter;
pub use fir_impl::design::FirFilterShape;
pub use fir_impl::{
    FirDecimationFilter, FirFarrowFilter, FirFilter, FirHilbertFilter, FirInterpolationFilter, FirPolyphaseFilter,
    FirPolyphaseFilterBank,
};
pub use fractional_delay::FractionalDelay;
pub use iir_impl::design::{IirBandType, IirFilterShape, IirFormat};
pub use iir_impl::{IirDecimationFilter, IirFilter, IirHilbertFilter, IirInterpolationFilter, IirSecondOrderSection};
pub use order_statistic_filter::OrderStatisticFilter;
pub use resampler::{
    ArbitraryResampler, HalfBandResampler, MultiStageHalfBandResampler, MultiStageResampler, RationalResampler,
    ResampType,
};
pub use sample::Sample;
pub use symbol_synchronizer::SymbolSynchronizer;
