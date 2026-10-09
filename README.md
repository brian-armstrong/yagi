# Yagi DSP

[![GitHub Actions Status](https://img.shields.io/github/actions/workflow/status/brian-armstrong/yagi/rust.yml)](https://github.com/brian-armstrong/yagi/actions/workflows/rust.yml) [![crates.io](https://img.shields.io/crates/v/yagi)](https://crates.io/crates/yagi) [![docs.rs](https://img.shields.io/docsrs/yagi)](https://docs.rs/yagi)

## Batteries-included DSP library for Rust.

yagi aims to provide a wide set of functionality in one crate. This includes FIR and IIR filters, resampling, mixing, equalization, gain correction, OFDM primitives, Reed-Solomon error correction, and PSK, FSK, and QAM modulation. yagi does not contain any protocol-specific features but does offer many of the primitives needed to implement real protocols.

yagi has an MSRV of 1.88. As of the 0.5 release, much of the public API surface is now settled, but breaking
changes can still be expected in 0.6.

### Quick start

Modulate QPSK symbols, pulse-shape and mix them onto a carrier, then recover them with the matching receiver.
```bash
cargo add yagi
cargo add num-complex
```

```rust
use num_complex::Complex32;
use std::f32::consts::PI;
use yagi::filter::{FirDecimationFilter, FirFilterShape, FirInterpolationFilter};
use yagi::modem::{Modem, ModulationScheme};
use yagi::nco::{Nco, NcoBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sps = 8; // samples per symbol
    let m = 4; // pulse-shaping filter delay, in symbols
    let beta = 0.3; // excess bandwidth
    let delay = 0.0; // fractional delay

    let mut modem = Modem::new(ModulationScheme::Qpsk)?;
    let symbols: Vec<u32> = (0..64).map(|i| ((i % 4 + (2 * i) % 7) % 4) as u32).collect();

    // Design matching root-raised-cosine transmit and receive filters
    let mut interp =
        FirInterpolationFilter::<Complex32, f32>::new_prototype(FirFilterShape::Rrcos, sps, m, beta, delay)?;
    let mut decim =
        FirDecimationFilter::<Complex32, f32>::new_prototype(FirFilterShape::Rrcos, sps, m, beta, delay)?;

    // Matching oscillators
    let mut upconvert = Nco::new(NcoBackend::Direct);
    let mut downconvert = Nco::new(NcoBackend::Direct);
    upconvert.set_frequency(0.25 * 2.0 * PI);
    downconvert.set_frequency(0.25 * 2.0 * PI);

    // Modulate each symbol, pulse-shape it, and mix the baseband signal up
    let mut passband = Vec::new();
    let mut shaped = vec![Complex32::new(0.0, 0.0); sps];
    for &symbol in &symbols {
        // Interpolation reads one symbol in and writes `sps` samples out
        interp.execute(modem.modulate(symbol)?, &mut shaped)?;
        for &sample in &shaped {
            passband.push(upconvert.mix_up(sample));
            upconvert.step();
        }
    }

    // Mix back down to baseband, decimate, and demodulate
    let mut downconverted = vec![Complex32::new(0.0, 0.0); sps];
    let mut recovered = Vec::new();
    for samples in passband.chunks_exact(sps) {
        for (dst, &sample) in downconverted.iter_mut().zip(samples) {
            *dst = downconvert.mix_down(sample);
            downconvert.step();
        }
        // Decimation reads `sps` samples in and returns one symbol out
        recovered.push(modem.demodulate(decim.execute(&downconverted)?)?);
    }

    // Each filter delays the stream by `m` symbols, so match starting after that
    let delay = m + m;
    // The last `delay` symbols aren't available to check yet (filters need to be flushed)
    assert_eq!(symbols[..symbols.len() - delay], recovered[delay..]);

    Ok(())
}
```

### A Rust Rewrite of liquid-dsp

At the heart of this library is a Rust implementation of [liquid-dsp](https://liquidsdr.org/), a DSP library originally written in C by Joseph Gaeddert. liquid-dsp is an excellent library, and it has provided the perfect starting place for yagi. yagi has carefully matched the sophisticated math contained in liquid-dsp while
offering Rust-native types and APIs that will serve as a good foundation for new work moving forward. While
this work serves as the basis for yagi's beginning, this library is not attempting to offer an exact 1:1
translation of liquid-dsp and will develop in its own direction moving forward.

liquid-dsp has more than 1300 tests. The testing coverage helps to give confidence in yagi's port to Rust.
The current status of the rewrite and testing efforts can be found at [LIQUID_COMPAT.md](LIQUID_COMPAT.md).
More than 90% of these tests are now ported and passing. yagi currently targets liquid-dsp 1.6.0.

Known gaps:

- Most of `framing` is still unimplemented.
- SPWATERFALL `fft` objects
- `fec_sumproduct`
- liquid's `benchmarks` are not yet implemented

yagi will likely not port liquid's `examples` and `sandbox` binaries.

### `simd` feature
yagi ships with an optional `simd` feature that adds std::simd acceleration. This currently affects the
dot product implementation and enables simd acceleration in the fec dependency. Enabling this feature
requires nightly Rust.

### `liquid-quirks` feature
For certain modules, the `liquid-quirks` feature toggles yagi's behavior to match liquid's output more closely.
This is only recommended when direct interoperability with a liquid peer is required.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
