#![cfg_attr(feature = "simd", feature(portable_simd))]

/// Automatic gain control
pub mod agc;
/// Audio
pub mod audio;
/// Buffer
pub mod buffer;
/// Channel
pub mod channel;
/// Dot product
pub mod dotprod;
/// Equalization
pub mod equalization;
/// FEC (Forward Error Correction)
pub mod fec;
/// FFT (Fast Fourier Transform)
pub mod fft;
/// Filter
pub mod filter;
/// Framing
pub mod framing;
/// Math
pub mod math;
/// Matrix
pub mod matrix;
/// Modem (modulator/demodulator)
pub mod modem;
/// Multichannel
pub mod multichannel;
/// NCO (Numerically Controlled Oscillator)
pub mod nco;
/// Optimization
pub mod optim;
/// Quantization
pub mod quantization;
/// Random number generator
pub mod random;
/// Sequence
pub mod sequence;
/// Utility
pub mod utility;
/// Vector
pub mod vector;
/// Error Handling
pub mod error;
