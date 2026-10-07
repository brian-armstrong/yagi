//! Tapering windows for finite sample blocks.
//!
//! Each sample function takes a zero-based index `i` in `0..wlen` and the
//! window length `wlen`.

use crate::error::{Error, Result};
use crate::math::besseli0f;
use std::f32::consts::PI;

/// Window family selected by [`window`] or [`get_window_type`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// No window family. [`window`] rejects it.
    Unknown,
    /// Hamming window.
    Hamming,
    /// Hann window.
    Hann,
    /// Four-term Blackman-Harris window.
    #[doc(alias = "BlackmanHarris")]
    BlackmanHarris4,
    /// Seven-term Blackman-Harris window.
    BlackmanHarris7,
    /// Kaiser window.
    Kaiser,
    /// Flat-top window.
    FlatTop,
    /// Triangular window.
    Triangular,
    /// Raised-cosine taper window.
    RcosTaper,
    /// Kaiser-Bessel derived window.
    Kbd,
}

/// Metadata describing a named window.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WindowInfo {
    short_name: &'static str,
    long_name: &'static str,
}

// Define window information
// The order of these must match the order of the WindowType variants
const WINDOW_INFO: [WindowInfo; 10] = [
    WindowInfo { short_name: "unknown", long_name: "unknown" },
    WindowInfo { short_name: "hamming", long_name: "Hamming" },
    WindowInfo { short_name: "hann", long_name: "Hann" },
    WindowInfo { short_name: "blackmanharris4", long_name: "Blackman-harris (4-term)" },
    WindowInfo { short_name: "blackmanharris7", long_name: "Blackman-harris (7-term)" },
    WindowInfo { short_name: "kaiser", long_name: "Kaiser-Bessel" },
    WindowInfo { short_name: "flattop", long_name: "flat top" },
    WindowInfo { short_name: "triangular", long_name: "triangular" },
    WindowInfo { short_name: "rcostaper", long_name: "raised-cosine taper" },
    WindowInfo { short_name: "kbd", long_name: "Kaiser-Bessel derived" },
];

/// Prints the short and long name of each window family to standard output.
///
/// The list includes `unknown`. Use [`get_window_type`] to select a window
/// by name in a program.
pub fn print_windows() {
    println!("Available window functions:");
    for info in &WINDOW_INFO {
        println!("  {} - {}", info.short_name, info.long_name);
    }
}

/// Returns the window family with the given short name.
///
/// The names are `"hamming"`, `"hann"`, `"blackmanharris4"`,
/// `"blackmanharris7"`, `"kaiser"`, `"flattop"`, `"triangular"`,
/// `"rcostaper"` and `"kbd"`. Matching is exact and case-sensitive.
/// `"blackmanharris"` also selects the four-term Blackman-Harris window.
/// `"unknown"` returns [`WindowType::Unknown`], which [`window`] rejects.
///
/// # Errors
///
/// Returns an error when `name` is not one of these names.
pub fn get_window_type(name: &str) -> Result<WindowType> {
    if name == "blackmanharris" {
        return Ok(WindowType::BlackmanHarris4);
    }
    for (i, info) in WINDOW_INFO.iter().enumerate() {
        if info.short_name == name {
            return Ok(unsafe { std::mem::transmute::<u8, WindowType>(i as u8) });
        }
    }
    Err(Error::Config("Unknown window type".to_string()))
}

/// Returns sample `i` of a window of family `window_type` and length `wlen`.
///
/// `arg` is used only by the families that take a parameter. It is the beta
/// for [`kaiser`] and [`kbd`], the sub-length for [`triangular`], and the
/// taper length for [`rcos_taper`]. The last two must be whole numbers. See
/// each family's function for its requirements.
///
/// # Errors
///
/// Returns an error for [`WindowType::Unknown`], a zero `wlen`, an `i` of
/// `wlen` or more, or an `arg` the family rejects.
pub fn window(window_type: WindowType, i: usize, wlen: usize, arg: f32) -> Result<f32> {
    match window_type {
        WindowType::Hamming => hamming(i, wlen),
        WindowType::Hann => hann(i, wlen),
        WindowType::BlackmanHarris4 => blackman_harris4(i, wlen),
        WindowType::BlackmanHarris7 => blackman_harris7(i, wlen),
        WindowType::Kaiser => kaiser(i, wlen, arg),
        WindowType::FlatTop => flat_top(i, wlen),
        WindowType::Triangular => triangular(i, wlen, window_arg_as_usize(arg, "Triangular")?),
        WindowType::RcosTaper => rcos_taper(i, wlen, window_arg_as_usize(arg, "Raised-cosine taper")?),
        WindowType::Kbd => kbd(i, wlen, arg),
        WindowType::Unknown => Err(Error::Config("Unknown window type".to_string())),
    }
}

fn validate_window_index(i: usize, wlen: usize, name: &str) -> Result<()> {
    if wlen == 0 {
        return Err(Error::Value(format!("{name} window: window length must be greater than zero")));
    }
    if i >= wlen {
        return Err(Error::Value(format!("{name} window: sample index must be less than window length")));
    }
    Ok(())
}

fn window_arg_as_usize(arg: f32, name: &str) -> Result<usize> {
    if !arg.is_finite() || arg < 0.0 || arg.fract() != 0.0 || arg > usize::MAX as f32 {
        return Err(Error::Value(format!("{name} window: argument must be a non-negative integer")));
    }
    Ok(arg as usize)
}

/// Returns sample `i` of a Kaiser window of length `wlen`.
///
/// `beta` sets the taper. Zero gives a rectangular window, and larger values
/// give lower sidelobes and a wider main lobe. The window is scaled so that
/// the center of an odd-length window is one. A `beta` above about 92
/// overflows the Bessel function, and the samples become NaN or zero.
///
/// # Errors
///
/// Returns an error when `wlen` is zero, `i >= wlen`, or `beta` is negative
/// or not finite.
pub fn kaiser(i: usize, wlen: usize, beta: f32) -> Result<f32> {
    validate_window_index(i, wlen, "Kaiser")?;
    if !beta.is_finite() || beta < 0.0 {
        return Err(Error::Value("Kaiser window: beta must be finite and greater than or equal to zero".to_string()));
    }
    if wlen == 1 {
        return Ok(1.0);
    }

    let t = i as f32 - (wlen - 1) as f32 / 2.0;
    let r = 2.0 * t / (wlen - 1) as f32;
    let a = besseli0f(beta * (1.0 - r * r).sqrt());
    let b = besseli0f(beta);

    Ok(a / b)
}

/// Returns sample `i` of a Hamming window of length `wlen`.
///
/// The coefficients are 0.53836 and 0.46164, so the end samples are 0.07672
/// and the center of an odd-length window is one.
///
/// # Errors
///
/// Returns an error when `wlen` is zero or `i >= wlen`.
pub fn hamming(i: usize, wlen: usize) -> Result<f32> {
    validate_window_index(i, wlen, "Hamming")?;
    if wlen == 1 {
        return Ok(1.0);
    }

    Ok(0.53836 - 0.46164 * ((2.0 * PI * i as f32) / (wlen - 1) as f32).cos())
}

/// Returns sample `i` of a Hann window of length `wlen`.
///
/// The end samples are zero and the center of an odd-length window is one.
///
/// # Errors
///
/// Returns an error when `wlen` is zero or `i >= wlen`.
pub fn hann(i: usize, wlen: usize) -> Result<f32> {
    validate_window_index(i, wlen, "Hann")?;
    if wlen == 1 {
        return Ok(1.0);
    }

    Ok(0.5 - 0.5 * ((2.0 * PI * i as f32) / (wlen - 1) as f32).cos())
}

/// Returns sample `i` of a four-term Blackman-Harris window of length
/// `wlen`.
///
/// The center of an odd-length window is one.
///
/// # Errors
///
/// Returns an error when `wlen` is zero or `i >= wlen`.
#[doc(alias = "blackman_harris")]
pub fn blackman_harris4(i: usize, wlen: usize) -> Result<f32> {
    validate_window_index(i, wlen, "Blackman-Harris")?;
    if wlen == 1 {
        return Ok(1.0);
    }

    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;
    let t = 2.0 * PI * i as f32 / (wlen - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos())
}

/// Returns sample `i` of a seven-term Blackman-Harris window of length
/// `wlen`.
///
/// The coefficients sum to 0.99997, so the center of an odd-length window is
/// slightly below one.
///
/// # Errors
///
/// Returns an error when `wlen` is zero or `i >= wlen`.
pub fn blackman_harris7(i: usize, wlen: usize) -> Result<f32> {
    validate_window_index(i, wlen, "Blackman-Harris 7")?;
    if wlen == 1 {
        return Ok(1.0);
    }

    let a0 = 0.27105;
    let a1 = 0.43329;
    let a2 = 0.21812;
    let a3 = 0.06592;
    let a4 = 0.01081;
    let a5 = 0.00077;
    let a6 = 0.00001;
    let t = 2.0 * PI * i as f32 / (wlen - 1) as f32;

    Ok(
        a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + a4 * (4.0 * t).cos() - a5 * (5.0 * t).cos()
            + a6 * (6.0 * t).cos(),
    )
}

/// Returns sample `i` of a flat-top window of length `wlen`.
///
/// Unlike the other windows, this one is not scaled to a peak of one. For
/// odd `wlen` greater than one, the center sample is 4.636, or 13.3 dB above
/// one. Divide by that value if a unit peak is needed.
///
/// # Errors
///
/// Returns an error when `wlen` is zero or `i >= wlen`.
pub fn flat_top(i: usize, wlen: usize) -> Result<f32> {
    validate_window_index(i, wlen, "Flat-top")?;
    if wlen == 1 {
        return Ok(1.0);
    }

    let a0 = 1.000;
    let a1 = 1.930;
    let a2 = 1.290;
    let a3 = 0.388;
    let a4 = 0.028;
    let t = 2.0 * PI * i as f32 / (wlen - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + a4 * (4.0 * t).cos())
}

/// Returns sample `i` of a triangular window of length `wlen`.
///
/// `n` sets the width of the triangle and must be `wlen - 1`, `wlen`, or
/// `wlen + 1`. With `wlen - 1`, the end samples are zero. The larger values
/// leave them above zero.
///
/// # Errors
///
/// Returns an error when `wlen` is zero, `i >= wlen`, or `n` is zero or not
/// one of the allowed values.
pub fn triangular(i: usize, wlen: usize, n: usize) -> Result<f32> {
    validate_window_index(i, wlen, "Triangular")?;
    if n == 0 {
        return Err(Error::Value("Triangular window: sub-length must be greater than zero".to_string()));
    }
    if n.abs_diff(wlen) > 1 {
        return Err(Error::Value("Triangular window: sub-length must be in wlen+{{-1,0,1}}".to_string()));
    }

    let v0 = i as f32 - (wlen - 1) as f32 / 2.0;
    let v1 = n as f32 / 2.0;
    Ok(1.0 - (v0 / v1).abs())
}

/// Returns sample `i` of a raised-cosine taper window of length `wlen`.
///
/// The window is one except for a half-cosine taper of `t` samples at each
/// end. A `t` of zero gives a rectangular window. `t` may be at most
/// `wlen / 2`, rounded down.
///
/// # Errors
///
/// Returns an error when `wlen` is zero, `i >= wlen`, or `t` is too large.
pub fn rcos_taper(i: usize, wlen: usize, t: usize) -> Result<f32> {
    validate_window_index(i, wlen, "Raised-cosine taper")?;
    if t > wlen / 2 {
        return Err(Error::Value(
            "Raised-cosine taper window: taper length cannot exceed half window length".to_string(),
        ));
    }

    let i = if i > wlen - t - 1 { wlen - i - 1 } else { i };

    if i < t {
        Ok(0.5 - 0.5 * (PI * (i as f32 + 0.5) / t as f32).cos())
    } else {
        Ok(1.0)
    }
}

/// Returns sample `i` of a Kaiser-Bessel derived window of length `wlen`.
///
/// The first half of the window is the square root of the running sum of a
/// Kaiser window of length `wlen / 2 + 1`, divided by its total. The second
/// half mirrors the first. Up to rounding, the samples satisfy
/// `w[i]^2 + w[i + wlen / 2]^2 = 1`, the Princen-Bradley condition for MDCT
/// windows. `beta` is the Kaiser window's parameter, with the same limits as
/// in [`kaiser`].
///
/// Each call recomputes the whole Kaiser window. [`kbd_window`] builds the
/// complete window at once and returns the same values.
///
/// # Errors
///
/// Returns an error when `wlen` is zero or odd, `i >= wlen`, or `beta` is
/// negative or not finite.
pub fn kbd(i: usize, wlen: usize, beta: f32) -> Result<f32> {
    validate_window_index(i, wlen, "KBD")?;
    if !beta.is_finite() || beta < 0.0 {
        return Err(Error::Value("KBD window: beta must be finite and greater than or equal to zero".to_string()));
    }
    if !wlen.is_multiple_of(2) {
        return Err(Error::Value("KBD window: window length must be even".to_string()));
    }

    let m = wlen / 2;
    if i >= m {
        return kbd(wlen - i - 1, wlen, beta);
    }

    let mut w0 = 0.0;
    let mut w1 = 0.0;
    for j in 0..=m {
        let w = kaiser(j, m + 1, beta)?;
        w1 += w;
        if j <= i {
            w0 += w;
        }
    }

    Ok((w0 / w1).sqrt())
}

/// Returns a complete Kaiser-Bessel derived window of length `wlen`.
///
/// The samples are the same as those from [`kbd`].
///
/// # Errors
///
/// Returns an error when `wlen` is zero or odd, or `beta` is negative or not
/// finite.
pub fn kbd_window(wlen: usize, beta: f32) -> Result<Vec<f32>> {
    if wlen == 0 {
        return Err(Error::Value("KBD window: window length must be greater than zero".to_string()));
    }
    if !wlen.is_multiple_of(2) {
        return Err(Error::Value("KBD window: window length must be even".to_string()));
    }
    if !beta.is_finite() || beta < 0.0 {
        return Err(Error::Value("KBD window: beta must be finite and greater than or equal to zero".to_string()));
    }

    let m = wlen / 2;
    let mut w = vec![0.0; wlen];

    let mut w_kaiser = vec![0.0; m + 1];
    for (i, wk) in w_kaiser.iter_mut().enumerate() {
        *wk = kaiser(i, m + 1, beta)?;
    }

    let w_sum: f32 = w_kaiser.iter().sum();

    let mut w_acc = 0.0;
    for i in 0..m {
        w_acc += w_kaiser[i];
        w[i] = (w_acc / w_sum).sqrt();
    }

    for i in 0..m {
        w[wlen - i - 1] = w[i];
    }

    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fft::{Direction, Fft};
    use approx::assert_abs_diff_eq;
    use num_complex::Complex;
    use test_macro::autotest_annotate;

    fn window_testbench(window_type: WindowType, n: usize, arg: f32) {
        let mut w = vec![0.0; n];
        let mut wsum = 0.0;
        for i in 0..n {
            w[i] = window(window_type, i, n, arg).unwrap();
            wsum += w[i];
        }

        // Compute spectral response
        let nfft = 1200;
        let mut buf_time = vec![Complex::new(0.0, 0.0); nfft];
        let mut buf_freq = vec![Complex::new(0.0, 0.0); nfft];
        for i in 0..nfft {
            buf_time[i] = if i < n { Complex::new(w[i] / wsum, 0.0) } else { Complex::new(0.0, 0.0) };
        }
        let fft = Fft::new(nfft, Direction::Forward);
        fft.run(&buf_time, &mut buf_freq);
        fft.shift(&mut buf_freq);

        // Compute bandwidth of window, ensure reasonable range
        let mut bw = 0.0;
        for i in 0..nfft {
            let r = i % 2;
            let l = (i - r) / 2;
            let k = if r == 0 { nfft / 2 - l } else { nfft / 2 + l + r };
            if 20.0 * (buf_freq[k].norm()).log10() < -6.0 {
                bw = i as f32 / nfft as f32;
                break;
            }
        }
        assert!(bw > 0.02);
        assert!(bw < 0.08);

        // Check side lobes at band edges
        for i in 0..nfft {
            let f = i as f32 / nfft as f32 - 0.5;
            if f.abs() > 0.20 {
                assert!(20.0 * (buf_freq[i].norm()).log10() < -40.0);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_window_hamming)]
    fn test_window_hamming() {
        window_testbench(WindowType::Hamming, 71, 0.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_hann)]
    fn test_window_hann() {
        window_testbench(WindowType::Hann, 71, 0.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_blackmanharris)]
    fn test_window_blackmanharris() {
        window_testbench(WindowType::BlackmanHarris4, 71, 0.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_blackmanharris7)]
    fn test_window_blackmanharris7() {
        window_testbench(WindowType::BlackmanHarris7, 71, 0.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_kaiser)]
    fn test_window_kaiser() {
        window_testbench(WindowType::Kaiser, 71, 10.0);
    }

    #[test]
    fn test_window_kaiser_single_sample() {
        assert_eq!(kaiser(0, 1, 10.0).unwrap(), 1.0);
    }

    #[test]
    fn test_window_single_sample() {
        assert_eq!(hamming(0, 1).unwrap(), 1.0);
        assert_eq!(hann(0, 1).unwrap(), 1.0);
        assert_eq!(blackman_harris4(0, 1).unwrap(), 1.0);
        assert_eq!(blackman_harris7(0, 1).unwrap(), 1.0);
        assert_eq!(flat_top(0, 1).unwrap(), 1.0);
        assert_eq!(triangular(0, 1, 1).unwrap(), 1.0);
        assert_eq!(rcos_taper(0, 1, 0).unwrap(), 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_flattop)]
    fn test_window_flattop() {
        window_testbench(WindowType::FlatTop, 71, 0.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_triangular)]
    fn test_window_triangular() {
        window_testbench(WindowType::Triangular, 71, 71.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_rcostaper)]
    fn test_window_rcostaper() {
        window_testbench(WindowType::RcosTaper, 71, 25.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_kbd)]
    fn test_window_kbd() {
        window_testbench(WindowType::Kbd, 72, 0.0);
    }

    fn kbd_window_test(n: usize, beta: f32) {
        let tol = 1e-3;

        // Compute window
        let w = kbd_window(n, beta).unwrap();

        // Square window
        let w2: Vec<f32> = w.iter().map(|&x| x * x).collect();

        // Ensure w[i]^2 + w[i+M]^2 == 1
        let m = n / 2;
        for i in 0..m {
            assert_abs_diff_eq!(w2[i] + w2[(i + m) % n], 1.0, epsilon = tol);
        }

        // Ensure sum(w[i]^2) == n/2
        let sum: f32 = w2.iter().sum();
        assert_abs_diff_eq!(sum, 0.5 * n as f32, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_kbd_n16)]
    fn test_kbd_n16() {
        kbd_window_test(16, 10.0);
    }

    #[test]
    #[autotest_annotate(autotest_kbd_n32)]
    fn test_kbd_n32() {
        kbd_window_test(32, 20.0);
    }

    #[test]
    #[autotest_annotate(autotest_kbd_n48)]
    fn test_kbd_n48() {
        kbd_window_test(48, 12.0);
    }

    #[test]
    #[autotest_annotate(autotest_window_config)]
    fn test_window_config() {
        assert_eq!(print_windows(), ());

        // Check normal cases
        assert_eq!(get_window_type("unknown"), Ok(WindowType::Unknown));
        assert_eq!(get_window_type("hamming"), Ok(WindowType::Hamming));
        assert_eq!(get_window_type("hann"), Ok(WindowType::Hann));
        assert_eq!(get_window_type("blackmanharris4"), Ok(WindowType::BlackmanHarris4));
        assert_eq!(get_window_type("blackmanharris"), Ok(WindowType::BlackmanHarris4));
        assert_eq!(get_window_type("blackmanharris7"), Ok(WindowType::BlackmanHarris7));
        assert_eq!(get_window_type("kaiser"), Ok(WindowType::Kaiser));
        assert_eq!(get_window_type("flattop"), Ok(WindowType::FlatTop));
        assert_eq!(get_window_type("triangular"), Ok(WindowType::Triangular));
        assert_eq!(get_window_type("rcostaper"), Ok(WindowType::RcosTaper));
        assert_eq!(get_window_type("kbd"), Ok(WindowType::Kbd));

        // Check invalid cases
        assert!(get_window_type("invalid window").is_err());

        // Invalid KBD window parameters
        assert!(kbd(12, 10, 10.0).is_err()); // Index exceeds maximum
        assert!(kbd(0, 0, 10.0).is_err()); // Window length is zero
        assert!(kbd(12, 27, 10.0).is_err()); // Window length is odd

        assert!(kbd_window(0, 10.0).is_err()); // Length is zero
        assert!(kbd_window(7, 10.0).is_err()); // Length is odd
        assert!(kbd_window(20, -1.0).is_err()); // Beta value is negative
        assert!(kbd_window(20, f32::NAN).is_err());

        // Invalid Kaiser window parameters
        assert!(kaiser(12, 10, 10.0).is_err()); // Index exceeds maximum
        assert!(kaiser(12, 20, -1.0).is_err()); // Beta value is negative
        assert!(kaiser(0, 10, f32::NAN).is_err());

        for window in [hamming as fn(usize, usize) -> Result<f32>, hann, blackman_harris4, blackman_harris7, flat_top] {
            assert!(window(12, 10).is_err());
            assert!(window(10, 10).is_err());
            assert!(window(0, 0).is_err());
        }

        // Triangular
        assert!(triangular(12, 10, 10).is_err()); // Index exceeds maximum
        assert!(triangular(10, 10, 10).is_err()); // Index exceeds maximum
        assert!(triangular(0, 0, 1).is_err()); // Window length is zero
        assert!(triangular(7, 10, 15).is_err()); // Sub-length is out of range
        assert!(triangular(1, 1, 0).is_err()); // Sub-length is zero

        // Raised-cosine taper
        assert!(rcos_taper(12, 10, 4).is_err()); // Index exceeds maximum
        assert!(rcos_taper(10, 10, 4).is_err()); // Index exceeds maximum
        assert!(rcos_taper(0, 0, 0).is_err()); // Window length is zero
        assert!(rcos_taper(7, 10, 8).is_err()); // Taper length exceeds maximum

        assert!(window(WindowType::Triangular, 0, 10, 2.5).is_err());
        assert!(window(WindowType::RcosTaper, 0, 10, -1.0).is_err());
    }
}
