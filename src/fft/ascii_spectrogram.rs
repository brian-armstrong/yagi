//! Text rendering of a power spectrum.

use crate::error::{Error, Result};
use crate::fft::SpectralPeriodogram;
use crate::math::WindowType;
use num_complex::Complex32;

/// Number of signal levels in the glyph map.
const ASGRAM_NUM_LEVELS: usize = 10;

/// Default display glyphs, from quietest to loudest.
const ASGRAM_DEFAULT_DISPLAY: &str = " .,-+*&NM#";

/// Renders the power spectrum of a sample stream as a row of characters.
///
/// The spectrum comes from a [`SpectralPeriodogram`] with a Hann window of
/// `nfft` samples, a hop of `nfft / 2` and a transform of `4 * nfft` bins.
/// Each of the `nfft` characters averages four adjacent bins in dB, and the
/// row runs from -0.5 cycles per sample on the left to just below 0.5 on the
/// right, with DC near the center. The levels are those of
/// [`SpectralPeriodogram::psd`], so a tone reads at the window's coherent
/// gain rather than its input power.
///
/// [`Self::execute`] returns the row with the level and frequency of the
/// strongest bin, and [`Self::print`] writes them to stdout. Both summarize
/// the transforms completed since the previous call.
#[doc(alias = "asgram")]
#[doc(alias = "asgramcf")]
#[derive(Debug, Clone)]
pub struct AsciiSpectrogram {
    nfft: usize,                                 // transform size (display)
    nfftp: usize,                                // transform size (processing)
    p: usize,                                    // over-sampling rate
    periodogram: SpectralPeriodogram<Complex32>, // spectral periodogram object
    psd: Vec<f32>,                               // power spectral density [nfftp]
    levels: [f32; ASGRAM_NUM_LEVELS],            // threshold for signal levels [dB]
    levelchar: [char; ASGRAM_NUM_LEVELS],        // characters representing levels
    div: f32,                                    // dB per division
    ref_level: f32,                              // dB reference value
}

impl AsciiSpectrogram {
    /// Creates a spectrogram whose rows are `nfft` characters wide.
    ///
    /// A new spectrogram starts with zero-valued history. It draws with the
    /// characters `" .,-+*&NM#"`, a reference level of 0 dB and 10 dB per
    /// division. At that scale, unit-power white noise reads about 0 dB and
    /// draws as spaces. See [`Self::set_display`] and [`Self::set_scale`].
    ///
    /// # Errors
    ///
    /// Returns an error when `nfft` is less than three.
    pub fn new(nfft: usize) -> Result<Self> {
        if nfft < 3 {
            return Err(Error::Config("fft size must be at least 3".into()));
        }

        // derived values
        let p = 4; // over-sampling rate
        let nfftp = nfft * p;

        // create spectral periodogram object (Hann window, length nfft, hop nfft/2)
        let periodogram = SpectralPeriodogram::new(nfftp, WindowType::Hann, nfft, nfft / 2)?;

        let mut q = Self {
            nfft,
            nfftp,
            p,
            periodogram,
            psd: vec![0.0; nfftp],
            levels: [0.0; ASGRAM_NUM_LEVELS],
            levelchar: [' '; ASGRAM_NUM_LEVELS],
            div: 10.0,
            ref_level: 0.0,
        };

        // power spectral density levels and default display
        q.set_display(ASGRAM_DEFAULT_DISPLAY)?;
        q.set_scale(0.0, 10.0)?;

        Ok(q)
    }

    /// Discards the buffered samples and any spectrum accumulated since the
    /// last [`Self::execute`].
    ///
    /// The display characters and scale are unchanged.
    pub fn reset(&mut self) {
        self.periodogram.reset();
    }

    /// Sets the levels at which each display character is drawn.
    ///
    /// A character position gets display character `i` when its level is
    /// above `ref_level + i * div` dB, and the first character otherwise. So
    /// the second character starts at `ref_level + div` dB, and the last at
    /// `ref_level + 9 * div` dB.
    ///
    /// # Errors
    ///
    /// Returns an error when `div` is zero or negative.
    pub fn set_scale(&mut self, ref_level: f32, div: f32) -> Result<()> {
        if div <= 0.0 {
            return Err(Error::Config("div must be greater than zero".into()));
        }

        self.ref_level = ref_level;
        self.div = div;

        for (i, level) in self.levels.iter_mut().enumerate() {
            *level = ref_level + i as f32 * div;
        }
        Ok(())
    }

    /// Sets the ten display characters, from quietest to loudest.
    ///
    /// Any characters can be used, including non-ASCII ones such as block
    /// elements.
    ///
    /// # Errors
    ///
    /// Returns an error when `display` does not hold exactly ten characters.
    pub fn set_display(&mut self, display: &str) -> Result<()> {
        let chars: Vec<char> = display.chars().collect();
        if chars.len() != ASGRAM_NUM_LEVELS {
            return Err(Error::Config("display must be exactly 10 characters".into()));
        }
        self.levelchar.copy_from_slice(&chars);
        Ok(())
    }

    /// Adds one sample to the spectrogram.
    pub fn push(&mut self, x: Complex32) {
        self.periodogram.push(x);
    }

    /// Adds a slice of samples to the spectrogram.
    pub fn write(&mut self, x: &[Complex32]) {
        self.periodogram.write(x);
    }

    /// Returns the rendered row, the peak level in dB and the peak frequency.
    ///
    /// The row holds `nfft` characters. The peak is the strongest of the
    /// `4 * nfft` transform bins, and its frequency is in cycles per sample,
    /// in `[-0.5, 0.5)`.
    ///
    /// The row covers the transforms completed since the previous call. When
    /// at least one has completed, this also resets the spectrogram as
    /// [`Self::reset`] does, so samples not yet in a transform are discarded.
    /// When no transform has completed, this returns `nfft` spaces, whatever
    /// the display characters, with a peak of zero at zero frequency, and
    /// keeps the buffered samples.
    pub fn execute(&mut self) -> (String, f32, f32) {
        // check number of transforms
        if self.periodogram.num_transforms() == 0 {
            return (" ".repeat(self.nfft), 0.0, 0.0);
        }

        // execute spectral periodogram
        self.psd.copy_from_slice(&self.periodogram.psd());
        self.periodogram.reset();

        // find peak
        let mut peakval = self.psd[0];
        let mut peakfreq = -0.5;
        for (i, &v) in self.psd.iter().enumerate() {
            if i == 0 || v > peakval {
                peakval = v;
                peakfreq = i as f32 / self.nfftp as f32 - 0.5;
            }
        }

        // down-sample from nfft*p frequency bins to just nfft by averaging
        // over each run of 'p' bins, then map the result to a glyph
        let mut ascii = String::with_capacity(self.nfft);
        for i in 0..self.nfft {
            let mut psd_val = 0.0;
            for j in 0..self.p {
                psd_val += self.psd[self.p * i + j];
            }
            psd_val /= self.p as f32;

            // determine ascii level (which character to use)
            let mut c = self.levelchar[0];
            for j in 0..ASGRAM_NUM_LEVELS {
                if psd_val > self.levels[j] {
                    c = self.levelchar[j];
                }
            }
            ascii.push(c);
        }

        (ascii, peakval, peakfreq)
    }

    /// Prints the result of [`Self::execute`] to stdout as one line.
    ///
    /// The row appears between `>` and `<`, followed by the peak level in dB
    /// and the peak frequency.
    pub fn print(&mut self) {
        let (ascii, maxval, maxfreq) = self.execute();
        println!(" > {ascii} < pk{maxval:5.1} dB [{maxfreq:5.2}]");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;

    // liquid: autotest_asgramcf_copy — clone mid-stream, push identical samples
    // through both, and confirm the rendered row and peak match bit-for-bit.
    #[test]
    #[test_macro::autotest_annotate(autotest_asgramcf_copy)]
    fn asgramcf_copy() {
        let nfft = 70;
        let num_samples = 844;
        let nstd = 0.1;

        // create object and set parameters
        let mut q0 = AsciiSpectrogram::new(nfft).unwrap();
        q0.set_scale(-20.8, 3.1).unwrap();
        q0.set_display("abcdeFGHIJ").unwrap();

        // generate a bunch of random noise samples
        for _ in 0..num_samples {
            let v = Complex32::new(0.1 + nstd * randnf(), nstd * randnf());
            q0.push(v);
        }

        // copy object and push same samples through both
        let mut q1 = q0.clone();
        for _ in 0..num_samples {
            let v = Complex32::new(0.1 + nstd * randnf(), nstd * randnf());
            q0.push(v);
            q1.push(v);
        }

        // get spectrum and compare outputs
        let (a0, pv0, pf0) = q0.execute();
        let (a1, pv1, pf1) = q1.execute();
        assert_eq!(a0, a1);
        assert_eq!(pv0, pv1);
        assert_eq!(pf0, pf1);
    }

    #[test]
    fn rejects_small_nfft() {
        assert!(AsciiSpectrogram::new(1).is_err());
        assert!(AsciiSpectrogram::new(2).is_err());
        assert!(AsciiSpectrogram::new(3).is_ok());
    }

    #[test]
    fn rejects_bad_scale_and_display() {
        let mut q = AsciiSpectrogram::new(16).unwrap();
        assert!(q.set_scale(0.0, 0.0).is_err());
        assert!(q.set_scale(0.0, -1.0).is_err());
        assert!(q.set_display("short").is_err());
        assert!(q.set_display("waytoolongdisplay").is_err());
        assert!(q.set_display("0123456789").is_ok());
    }

    #[test]
    fn empty_before_any_transform() {
        let mut q = AsciiSpectrogram::new(16).unwrap();
        let (ascii, pv, pf) = q.execute();
        assert_eq!(ascii, " ".repeat(16));
        assert_eq!(pv, 0.0);
        assert_eq!(pf, 0.0);
    }
}
