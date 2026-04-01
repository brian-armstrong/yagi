use std::f32::consts::PI;
use crate::math::bessel::*;
use crate::error::{Error, Result};

/// Enum for window types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    Unknown,
    Hamming,
    Hann,
    BlackmanHarris,
    BlackmanHarris7,
    Kaiser,
    FlatTop,
    Triangular,
    RcosTaper,
    Kbd,
}

/// Struct to hold window information
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowInfo {
    short_name: &'static str,
    long_name: &'static str,
}

// Define window information
const WINDOW_INFO: [WindowInfo; 10] = [
    WindowInfo { short_name: "unknown", long_name: "unknown" },
    WindowInfo { short_name: "hamming", long_name: "Hamming" },
    WindowInfo { short_name: "hann", long_name: "Hann" },
    WindowInfo { short_name: "blackmanharris", long_name: "Blackman-harris (4-term)" },
    WindowInfo { short_name: "blackmanharris7", long_name: "Blackman-harris (7-term)" },
    WindowInfo { short_name: "kaiser", long_name: "Kaiser-Bessel" },
    WindowInfo { short_name: "flattop", long_name: "flat top" },
    WindowInfo { short_name: "triangular", long_name: "triangular" },
    WindowInfo { short_name: "rcostaper", long_name: "raised-cosine taper" },
    WindowInfo { short_name: "kbd", long_name: "Kaiser-Bessel derived" },
];

/// Function to print available window functions
pub fn print_windows() {
    println!("Available window functions:");
    for info in &WINDOW_INFO {
        println!("  {} - {}", info.short_name, info.long_name);
    }
}

/// Function to get WindowType from string
pub fn get_window_type(name: &str) -> Result<WindowType> {
    for (i, info) in WINDOW_INFO.iter().enumerate() {
        if info.short_name == name {
            return Ok(unsafe { std::mem::transmute(i as u8) });
        }
    }
    Err(Error::Config("Unknown window type".to_string()))
}

/// Generic window function
pub fn window(window_type: WindowType, i: usize, wlen: usize, arg: f32) -> Result<f32> {
    match window_type {
        WindowType::Hamming => hamming(i, wlen),
        WindowType::Hann => hann(i, wlen),
        WindowType::BlackmanHarris => blackman_harris(i, wlen),
        WindowType::BlackmanHarris7 => blackman_harris7(i, wlen),
        WindowType::Kaiser => kaiser(i, wlen, arg),
        WindowType::FlatTop => flat_top(i, wlen),
        WindowType::Triangular => triangular(i, wlen, arg as usize),
        WindowType::RcosTaper => rcos_taper(i, wlen, arg as usize),
        WindowType::Kbd => kbd(i, wlen, arg),
        WindowType::Unknown => Err(Error::Config("Unknown window type".to_string())),
    }
}

/// Implement individual window functions
pub fn kaiser(i: usize, wlen: usize, beta: f32) -> Result<f32> {
    if i >= wlen {
        return Err(Error::Value("Kaiser window: sample index must not exceed window length".to_string()));
    }
    if beta < 0.0 {
        return Err(Error::Value("Kaiser window: beta must be greater than or equal to zero".to_string()));
    }

    let t = i as f32 - (wlen - 1) as f32 / 2.0;
    let r = 2.0 * t / (wlen - 1) as f32;
    let a = besseli0f(beta * (1.0 - r * r).sqrt());
    let b = besseli0f(beta);

    Ok(a / b)
}

pub fn hamming(i: usize, wlen: usize) -> Result<f32> {
    if i > wlen {
        return Err(Error::Value("Hamming window: sample index must not exceed window length".to_string()));
    }

    Ok(0.53836 - 0.46164 * ((2.0 * PI * i as f32) / (wlen - 1) as f32).cos())
}

pub fn hann(i: usize, wlen: usize) -> Result<f32> {
    if i > wlen {
        return Err(Error::Value("Hann window: sample index must not exceed window length".to_string()));
    }

    Ok(0.5 - 0.5 * ((2.0 * PI * i as f32) / (wlen - 1) as f32).cos())
}

pub fn blackman_harris(i: usize, wlen: usize) -> Result<f32> {
    if i > wlen {
        return Err(Error::Value("Blackman-Harris window: sample index must not exceed window length".to_string()));
    }

    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;
    let t = 2.0 * PI * i as f32 / (wlen - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos())
}

pub fn blackman_harris7(i: usize, wlen: usize) -> Result<f32> {
    if i > wlen {
        return Err(Error::Value("Blackman-Harris 7 window: sample index must not exceed window length".to_string()));
    }

    let a0 = 0.27105;
    let a1 = 0.43329;
    let a2 = 0.21812;
    let a3 = 0.06592;
    let a4 = 0.01081;
    let a5 = 0.00077;
    let a6 = 0.00001;
    let t = 2.0 * PI * i as f32 / (wlen - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos()
        + a4 * (4.0 * t).cos() - a5 * (5.0 * t).cos() + a6 * (6.0 * t).cos())
}

pub fn flat_top(i: usize, wlen: usize) -> Result<f32> {
    if i > wlen {
        return Err(Error::Value("Flat-top window: sample index must not exceed window length".to_string()));
    }

    let a0 = 1.000;
    let a1 = 1.930;
    let a2 = 1.290;
    let a3 = 0.388;
    let a4 = 0.028;
    let t = 2.0 * PI * i as f32 / (wlen - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + a4 * (4.0 * t).cos())
}

pub fn triangular(i: usize, wlen: usize, n: usize) -> Result<f32> {
    if i > wlen {
        return Err(Error::Value("Triangular window: sample index must not exceed window length".to_string()));
    }
    if n != wlen - 1 && n != wlen && n != wlen + 1 {
        return Err(Error::Value("Triangular window: sub-length must be in wlen+{{-1,0,1}}".to_string()));
    }
    if n == 0 {
        return Err(Error::Value("Triangular window: sub-length must be greater than zero".to_string()));
    }

    let v0 = i as f32 - (wlen - 1) as f32 / 2.0;
    let v1 = n as f32 / 2.0;
    Ok(1.0 - (v0 / v1).abs())
}

pub fn rcos_taper(i: usize, wlen: usize, t: usize) -> Result<f32> {
    if i > wlen {
        return Err(Error::Value("Raised-cosine taper window: sample index must not exceed window length".to_string()));
    }
    if t > wlen / 2 {
        return Err(Error::Value("Raised-cosine taper window: taper length cannot exceed half window length".to_string()));
    }

    let i = if i > wlen - t - 1 { wlen - i - 1 } else { i };

    if i < t {
        Ok(0.5 - 0.5 * (PI * (i as f32 + 0.5) / t as f32).cos())
    } else {
        Ok(1.0)
    }
}

pub fn kbd(i: usize, wlen: usize, beta: f32) -> Result<f32> {
    if i >= wlen {
        return Err(Error::Value("KBD window: index exceeds maximum".to_string()));
    }
    if wlen == 0 {
        return Err(Error::Value("KBD window: window length must be greater than zero".to_string()));
    }
    if wlen % 2 != 0 {
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

pub fn kbd_window(wlen: usize, beta: f32) -> Result<Vec<f32>> {
    if wlen == 0 {
        return Err(Error::Value("KBD window: window length must be greater than zero".to_string()));
    }
    if wlen % 2 != 0 {
        return Err(Error::Value("KBD window: window length must be even".to_string()));
    }
    if beta < 0.0 {
        return Err(Error::Value("KBD window: beta must be positive".to_string()));
    }

    let m = wlen / 2;
    let mut w = vec![0.0; wlen];

    let mut w_kaiser = vec![0.0; m + 1];
    for i in 0..=m {
        w_kaiser[i] = kaiser(i, m + 1, beta)?;
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

    use crate::fft::{Fft, Direction};
    use num_complex::Complex;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;

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
        fft.shift(&mut buf_freq, nfft);

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
        window_testbench(WindowType::BlackmanHarris, 71, 0.0);
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
            assert_relative_eq!(w2[i] + w2[(i + m) % n], 1.0, epsilon = tol);
        }

        // Ensure sum(w[i]^2) == n/2
        let sum: f32 = w2.iter().sum();
        assert_relative_eq!(sum, 0.5 * n as f32, epsilon = tol);
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
        assert_eq!(get_window_type("blackmanharris"), Ok(WindowType::BlackmanHarris));
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

        // Invalid Kaiser window parameters
        assert!(kaiser(12, 10, 10.0).is_err()); // Index exceeds maximum
        assert!(kaiser(12, 20, -1.0).is_err()); // Beta value is negative

        // Hamming
        assert!(hamming(12, 10).is_err()); // Index exceeds maximum

        // Hann
        assert!(hann(12, 10).is_err()); // Index exceeds maximum

        // Blackman-Harris
        assert!(blackman_harris(12, 10).is_err()); // Index exceeds maximum

        // Blackman-Harris 7
        assert!(blackman_harris7(12, 10).is_err()); // Index exceeds maximum

        // Flat-top
        assert!(flat_top(12, 10).is_err()); // Index exceeds maximum

        // Triangular
        assert!(triangular(12, 10, 10).is_err()); // Index exceeds maximum
        assert!(triangular(7, 10, 15).is_err()); // Sub-length is out of range
        assert!(triangular(1, 1, 0).is_err()); // Sub-length is zero

        // Raised-cosine taper
        assert!(rcos_taper(12, 10, 4).is_err()); // Index exceeds maximum
        assert!(rcos_taper(7, 10, 8).is_err()); // Taper length exceeds maximum
    }
}