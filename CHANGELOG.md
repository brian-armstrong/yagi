# Changelog

## 0.5.0 - 2026-10-12

This release further expands on the liquid-dsp port while renaming and reorganizing
previously ported functionality. This is a major breaking change that will hopefully
settle most of the API surface. Further breaking changes are still possible, but
yagi is now recommended for external users.

The minimum supported Rust version is now 1.88. The `simd` feature requires
nightly Rust.

### Breaking changes

- Public types have descriptive names in place of many liquid-derived
  abbreviations. For example, `Osc` is now `Nco`, `Resamp` is
  `ArbitraryResampler`, `Spgram` is `SpectralPeriodogram`, and `Eqlms` is
  `LeastMeanSquaresEqualizer`. The old names are not retained as type aliases.
- Public types now have canonical paths directly under their domain modules,
  such as `yagi::modem::Modem` and `yagi::filter::FirFilter`. FIR and IIR design
  functions live under `yagi::filter::fir::design` and
  `yagi::filter::iir::design`.
- Trivial `get_*` methods use noun getters, such as `rate()` and
  `coefficients()`. Many constructors and setters have been given more explicit
  names or signatures. Redundant slice-length arguments have been removed.
- Block-processing methods have been revised to consume slices directly,
  validate buffer sizes, and report the number of outputs where applicable.
  Review each method's return type and capacity contract when migrating.

### Added and improved

- Added forward error correction, CRC, interleaving, and packet-coding APIs,
  including `Fec`, `Interleaver`, and `Packetizer`.
- Expanded modulation, demodulation, filtering, resampling, equalization,
  framing primitives, and OFDM/channelizer support.
- Added real-to-real FFT transforms, spectral analysis tools, vector
  operations, and numerical optimization objects.
- Added crate and module documentation, runnable API examples, and CI coverage
  for the declared Rust 1.88 MSRV, stable scalar builds, and nightly SIMD.
- Added the `liquid-quirks` feature for specific compatibility behaviors.
  Review it when exchanging encoded data with a liquid-dsp peer. Any
  divergences in wire behavior should be reported so they can be included
  behind this flag.

### Correctness and behavior

- Polyphase FIR banks retain uneven final taps by padding shorter phases,
  rather than discarding the extra taps. This can change designed filter and
  resampler output relative to earlier yagi versions.
- Improved validation of configuration, input values, and output capacities
  across filters, resamplers, modems, and framing primitives.
- Many sources of panics have been removed and others documented. Refer to
  specific object or function documentation for panic behavior.

### Migrating from 0.1.5

The examples below show the naming direction. This is not an exhaustive
mapping of the public API.

| Before | In 0.5 |
| --- | --- |
| `yagi::modem::modem::Modem` | `yagi::modem::Modem` |
| `yagi::nco::Osc` | `yagi::nco::Nco` |
| `yagi::filter::resamp::Resamp` | `yagi::filter::ArbitraryResampler` |
| `yagi::filter::rresamp::Rresamp` | `yagi::filter::RationalResampler` |
| `yagi::fft::spgram::Spgram` | `yagi::fft::SpectralPeriodogram` |
| `yagi::equalization::eqlms::Eqlms` | `yagi::equalization::LeastMeanSquaresEqualizer` |

Search the API documentation for the former liquid-derived name when a type
has moved (some renamed types retain search aliases). For methods, check whether a
`get_*` call became a noun getter, a `new_default` constructor gained a
descriptive name, or a separate length argument can now be derived from a
slice. No compatibility aliases are provided for the old public paths.
