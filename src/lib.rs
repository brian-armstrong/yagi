//! Digital signal processing building blocks for software-defined radio and communications.
//!
//! Yagi groups functionality into relevant modules. The commonly-used DSP primitives
//! include [`filter`], [`modem`], [`nco`] (oscillators and frequency conversion),
//! and [`fec`] (error correction). The crate also provides synchronization,
//! channel models, and spectral analysis.
//!
//! This crate builds on [liquid-dsp](https://liquidsdr.org/), the C DSP library by
//! Joseph Gaeddert. It brings many of liquid-dsp's algorithms and tests to
//! Rust, with APIs designed for Rust rather than a one-to-one translation of
//! the C interface.
//!
//! Most of yagi's signal-processing APIs support `f32` and
//! [`num_complex::Complex32`] samples. Operations are typically wrapped in
//! stateful containers which enable streaming computation. Yagi exposes a single
//! error type [`error::Error`] with general variants to cover error modes. Refer to
//! module documentation for conventions such as frequency units and coefficient
//! order.
//!
//! # Getting started
//!
//! This example uses [`modem::Modem`] to map integer symbols to constellation points
//! and then recovers them over a clean channel with no synchronization:
//!
//! ```
//! use yagi::modem::{Modem, ModulationScheme};
//!
//! let mut modem = Modem::new(ModulationScheme::Qpsk)?;
//! for symbol in 0..4 {
//!     let point = modem.modulate(symbol)?;
//!     assert_eq!(modem.demodulate(point)?, symbol);
//! }
//! # Ok::<(), yagi::error::Error>(())
//! ```
//!
//! For a larger example with pulse shaping and carrier mixing, see the
//! [README quick start](https://github.com/brian-armstrong/yagi#quick-start).
//!
//! # Features
//!
//! - `simd` enables SIMD kernels and requires a nightly compiler.
//! - `liquid-quirks` selects liquid-dsp-compatible behavior for documented
//!   differences, such as the RS8 packet interleaving described in [`fec`].

#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod agc;
pub mod audio;
pub mod buffer;
pub mod channel;
pub mod dotprod;
pub mod equalization;
pub mod error;
pub mod fec;
pub mod fft;
pub mod filter;
pub mod framing;
pub mod math;
pub mod matrix;
pub mod modem;
pub mod multichannel;
pub mod nco;
pub mod optim;
pub mod quantization;
pub mod random;
pub mod sequence;
pub mod utility;
pub mod vector;
