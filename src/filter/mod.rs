// Filter module

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
mod symbol_synchronizer;

// For the sake of keeping exports clean, create these sort of faux pub mods below
// When adding anything new to filter, remember to update these as well.

/// FIR coefficient-design and response-analysis operations.
pub mod fir {
    /// FIR coefficient-design and response-analysis operations.
    pub mod design {
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

/// IIR coefficient-design and response-analysis operations.
pub mod iir {
    /// IIR coefficient-design and response-analysis operations.
    pub mod design {
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

/// Linear-predictive coding design operations.
pub mod lpc {
    pub use super::lpc_impl::design_lpc as design;
}

pub use autocorrelator::Autocorrelator;
pub use direct_digital_synthesizer::DirectDigitalSynthesizer;
pub use fft_filter::{FftFilter, FromComplex32};
pub use fir_impl::design::FirFilterShape;
pub use fir_impl::{
    ComplexNotch, FirDecimationFilter, FirFarrowFilter, FirFilter, FirHilbertFilter, FirInterpolationFilter,
    FirPolyphaseFilter, FirPolyphaseFilterBank,
};
pub use fractional_delay::FractionalDelay;
pub use iir_impl::design::{IirBandType, IirFilterShape, IirFormat};
pub use iir_impl::{IirDecimationFilter, IirFilter, IirHilbertFilter, IirInterpolationFilter, IirSecondOrderSection};
pub use lpc::{design_lpc, levinson};
pub use order_statistic_filter::OrderStatisticFilter;
pub use resampler::{
    ArbitraryResampler, HalfBandResampler, MultiStageHalfBandResampler, MultiStageResampler, RationalResampler,
    Resamp2Coeff, ResampType,
};
pub use symbol_synchronizer::SymbolSynchronizer;
