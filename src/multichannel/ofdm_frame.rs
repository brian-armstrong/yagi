//
// ofdmframe.rs
//
// ofdmframe data, methods common to both generator/synchronizer objects
//  - physical layer convergence procedure (PLCP)
//

use crate::error::{Error, Result};
use crate::fft::{fft_run, Direction};
use crate::math::nextpow2;
use crate::sequence::MaximalLengthSequence;
use num_complex::Complex32;

/// Role of one OFDM subcarrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SubcarrierType {
    /// Disabled subcarrier that carries no energy.
    #[default]
    Null,
    /// Carries a known symbol that the receiver uses to track phase and
    /// timing.
    Pilot,
    /// Carries payload.
    Data,
}

/// Number of subcarriers of each type in an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubcarrierCounts {
    /// Number of disabled carriers.
    pub null: usize,
    /// Number of known-reference carriers.
    pub pilot: usize,
    /// Number of payload carriers.
    pub data: usize,
}

/// Validated OFDM frame layout shared by a generator and a synchronizer.
///
/// The configuration sets the number of subcarriers, the cyclic-prefix and
/// taper lengths, and the role of each subcarrier. A generator and the
/// synchronizer that receives its frames must use equal configurations.
/// The allocation is indexed by natural FFT bin.
/// The default uses 64 subcarriers, a 16-sample prefix, a 4-sample taper,
/// and the standard allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OfdmFrameConfig {
    num_subcarriers: usize,
    cp_len: usize,
    taper_len: usize,
    allocation: Vec<SubcarrierType>,
    counts: SubcarrierCounts,
}

impl OfdmFrameConfig {
    /// Creates a frame configuration.
    ///
    /// `cp_len` is the cyclic-prefix length in samples, and `taper_len` is the
    /// number of samples over which adjacent symbols are cross-faded.
    /// `allocation` gives the role of each subcarrier in natural FFT-bin order
    /// and is copied. `None` selects the allocation that
    /// [`ofdmframe_init_default_sctype`] produces. The allocation must have at
    /// least two pilot subcarriers, one data subcarrier, and one enabled
    /// even-indexed subcarrier for the short preamble.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_subcarriers` is odd or less than eight,
    /// `cp_len` exceeds `num_subcarriers`, `taper_len` exceeds `cp_len`,
    /// `allocation` does not have `num_subcarriers` entries, or the allocation
    /// has too few pilot or data subcarriers, or no enabled even-indexed
    /// subcarrier. The default allocation has too few pilots when
    /// `num_subcarriers` is eight.
    pub fn new(
        num_subcarriers: usize,
        cp_len: usize,
        taper_len: usize,
        allocation: Option<&[SubcarrierType]>,
    ) -> Result<Self> {
        if num_subcarriers < 8 {
            return Err(Error::Config("ofdmframe_config_create(), number of subcarriers must be at least 8".into()));
        }
        if !num_subcarriers.is_multiple_of(2) {
            return Err(Error::Config("ofdmframe_config_create(), number of subcarriers must be even".into()));
        }
        if cp_len > num_subcarriers {
            return Err(Error::Config("ofdmframe_config_create(), cyclic prefix cannot exceed symbol length".into()));
        }
        if taper_len > cp_len {
            return Err(Error::Config("ofdmframe_config_create(), taper length cannot exceed cyclic prefix".into()));
        }

        let allocation = match allocation {
            None => {
                let mut allocation = vec![SubcarrierType::Null; num_subcarriers];
                ofdmframe_init_default_sctype(&mut allocation)?;
                allocation
            }
            Some(allocation) => {
                if allocation.len() != num_subcarriers {
                    return Err(Error::Config(
                        "ofdmframe_config_create(), subcarrier allocation length must match".into(),
                    ));
                }
                allocation.to_vec()
            }
        };

        let counts = ofdmframe_validate_sctype(&allocation)
            .map_err(|_| Error::Config("ofdmframe_config_create(), invalid subcarrier allocation".into()))?;
        if !allocation.iter().step_by(2).any(|&t| t != SubcarrierType::Null) {
            return Err(Error::Config(
                "ofdmframe_config_create(), short preamble needs an enabled even-indexed subcarrier".into(),
            ));
        }

        Ok(Self { num_subcarriers, cp_len, taper_len, allocation, counts })
    }

    /// Returns the number of subcarriers, which is also the FFT size.
    pub fn num_subcarriers(&self) -> usize {
        self.num_subcarriers
    }

    /// Returns the cyclic-prefix length, in samples.
    pub fn cp_len(&self) -> usize {
        self.cp_len
    }

    /// Returns the number of samples over which adjacent symbols are
    /// cross-faded.
    pub fn taper_len(&self) -> usize {
        self.taper_len
    }

    /// Returns the role of each subcarrier, in natural FFT-bin order.
    pub fn allocation(&self) -> &[SubcarrierType] {
        &self.allocation
    }

    /// Returns the number of subcarriers of each type.
    pub fn counts(&self) -> SubcarrierCounts {
        self.counts
    }

    /// Returns the number of samples in each OFDM symbol, including its cyclic
    /// prefix.
    pub fn symbol_len(&self) -> usize {
        self.num_subcarriers + self.cp_len
    }
}

impl Default for OfdmFrameConfig {
    fn default() -> Self {
        Self::new(64, 16, 4, None).expect("valid default OFDM frame configuration")
    }
}

/// clamp the m-sequence register length for a given subcarrier count
fn sequence_bits(m_subcarriers: usize) -> Result<u32> {
    let m = nextpow2(m_subcarriers as u32)?;
    Ok(m.clamp(4, 8))
}

/// generate short sequence symbols
///   p  : subcarrier allocation array
///   s0 : output symbol (freq), [size: p.len()]
///   s0_time : output symbol (time), [size: p.len()]
///
/// returns the number of enabled subcarriers in S0.
pub(crate) fn ofdmframe_init_s0(
    p: &[SubcarrierType],
    s0: &mut [Complex32],
    s0_time: &mut [Complex32],
) -> Result<usize> {
    let num_subcarriers = p.len();
    if s0.len() != num_subcarriers || s0_time.len() != num_subcarriers {
        return Err(Error::Config("ofdmframe_init_s0(), output lengths must match allocation".into()));
    }

    // compute m-sequence length
    let m = sequence_bits(num_subcarriers)?;

    // generate m-sequence generator object
    let mut ms = MaximalLengthSequence::from_degree(m)?;

    let mut m_s0 = 0;

    // short sequence

    // S0 uses only even-numbered subcarriers, giving it a period of half a symbol
    // in the time domain. the synchronizer relies on that repetition to detect
    // the frame and estimate carrier frequency offset.
    for i in 0..num_subcarriers {
        // generate symbol
        let s = ms.generate_symbol(3) & 0x01;

        if p[i] == SubcarrierType::Null {
            // NULL subcarrier
            s0[i] = Complex32::new(0.0, 0.0);
        } else if i % 2 == 0 {
            // even subcarrer
            s0[i] = Complex32::new(if s != 0 { 1.0 } else { -1.0 }, 0.0);
            m_s0 += 1;
        } else {
            // odd subcarrer (ignore)
            s0[i] = Complex32::new(0.0, 0.0);
        }
    }

    // ensure at least one subcarrier was enabled
    if m_s0 == 0 {
        return Err(Error::Config("ofdmframe_init_s0(), no subcarriers enabled; check allocation".into()));
    }

    // run inverse fft to get time-domain sequence
    fft_run(s0, s0_time, Direction::Backward);

    // normalize time-domain sequence level
    let g = 1.0 / (m_s0 as f32).sqrt();
    for x in s0_time.iter_mut() {
        *x *= g;
    }

    Ok(m_s0)
}

/// generate long sequence symbols
///   p  : subcarrier allocation array
///   s1 : output symbol (freq), [size: p.len()]
///   s1_time : output symbol (time), [size: p.len()]
///
/// returns the number of enabled subcarriers in S1.
pub(crate) fn ofdmframe_init_s1(
    p: &[SubcarrierType],
    s1: &mut [Complex32],
    s1_time: &mut [Complex32],
) -> Result<usize> {
    let num_subcarriers = p.len();
    if s1.len() != num_subcarriers || s1_time.len() != num_subcarriers {
        return Err(Error::Config("ofdmframe_init_s1(), output lengths must match allocation".into()));
    }

    // increase m such that the resulting S1 sequence will
    // differ significantly from S0 with the same subcarrier
    // allocation array
    let m = sequence_bits(num_subcarriers)? + 1;

    // generate m-sequence generator object
    let mut ms = MaximalLengthSequence::from_degree(m)?;

    let mut m_s1 = 0;

    // long sequence
    for i in 0..num_subcarriers {
        // generate symbol
        let s = ms.generate_symbol(3) & 0x01;

        if p[i] == SubcarrierType::Null {
            // NULL subcarrier
            s1[i] = Complex32::new(0.0, 0.0);
        } else {
            s1[i] = Complex32::new(if s != 0 { 1.0 } else { -1.0 }, 0.0);
            m_s1 += 1;
        }
    }

    // ensure at least one subcarrier was enabled
    if m_s1 == 0 {
        return Err(Error::Config("ofdmframe_init_s1(), no subcarriers enabled; check allocation".into()));
    }

    // run inverse fft to get time-domain sequence
    fft_run(s1, s1_time, Direction::Backward);

    // normalize time-domain sequence level
    let g = 1.0 / (m_s1 as f32).sqrt();
    for x in s1_time.iter_mut() {
        *x *= g;
    }

    Ok(m_s1)
}

/// Fills `p` with the default subcarrier allocation.
///
/// For an even number `n` of subcarriers, bin zero (DC) is null, and so is a
/// guard band of `2 * g + 1` bins centered on bin `n/2`, where `g` is `n/10`
/// rounded down but at least two. Pilots sit 2, 6, 10 and so on bins from DC
/// on both sides, or 4, 12, 20 and so on when `n` is more than 34. The other
/// bins are data.
///
/// The whole slice is overwritten. The pilot and data counts are not checked,
/// and eight subcarriers give no pilots. [`OfdmFrameConfig::new`] checks them.
///
/// # Errors
///
/// Returns an error when `p` has fewer than six elements. `p` is unchanged on
/// error.
pub fn ofdmframe_init_default_sctype(p: &mut [SubcarrierType]) -> Result<()> {
    // validate input
    let num_subcarriers = p.len();
    if num_subcarriers < 6 {
        return Err(Error::Config("ofdmframe_init_default_sctype(), less than 6 subcarriers".into()));
    }

    let m2 = num_subcarriers / 2;

    // compute guard band
    let g = (num_subcarriers / 10).max(2);

    // designate pilot spacing
    let pilot_spacing = if num_subcarriers > 34 { 8 } else { 4 };
    let p2 = pilot_spacing / 2;

    // initialize as NULL
    p.fill(SubcarrierType::Null);

    // upper band
    for i in 1..m2 - g {
        p[i] = if (i + p2) % pilot_spacing == 0 { SubcarrierType::Pilot } else { SubcarrierType::Data };
    }

    // lower band
    for i in 1..m2 - g {
        let k = num_subcarriers - i;
        p[k] = if (i + p2) % pilot_spacing == 0 { SubcarrierType::Pilot } else { SubcarrierType::Data };
    }

    Ok(())
}

/// Fills `p` with an allocation that occupies the band from `f0` to `f1`.
///
/// `f0` and `f1` are in cycles per sample, in `[-0.5, 0.5]`, with `f0` below
/// `f1`. Each edge is rounded down to a bin, and bins from one edge to the
/// other inclusive are occupied. The edges must be at least six bins apart.
/// Occupied bins are pilots when their natural bin index is a multiple of
/// four, or of eight when the edges are more than 34 bins apart, and data
/// otherwise. DC is not excluded, so bin zero is a pilot when the band covers
/// it. Bins outside the band are null.
///
/// The whole slice is overwritten. The pilot and data counts are not checked.
/// [`OfdmFrameConfig::new`] checks them.
///
/// # Errors
///
/// Returns an error when `p` has fewer than six elements, either edge is
/// outside `[-0.5, 0.5]`, `f0` is not below `f1`, or the edges are fewer than
/// six bins apart. `p` is unchanged on error.
pub fn ofdmframe_init_sctype_range(f0: f32, f1: f32, p: &mut [SubcarrierType]) -> Result<()> {
    // validate input
    let num_subcarriers = p.len();
    if num_subcarriers < 6 {
        return Err(Error::Config("ofdmframe_init_sctype_range(), less than 6 subcarriers".into()));
    }
    if !(-0.5..=0.5).contains(&f0) {
        return Err(Error::Config("ofdmframe_init_sctype_range(), lower frequency edge must be in [-0.5,0.5]".into()));
    }
    if !(-0.5..=0.5).contains(&f1) {
        return Err(Error::Config("ofdmframe_init_sctype_range(), upper frequency edge must be in [-0.5,0.5]".into()));
    }
    if f0 >= f1 {
        return Err(Error::Config(
            "ofdmframe_init_sctype_range(), lower frequency edge must be below upper edge".into(),
        ));
    }

    // get relative edges
    let m0 = ((f0 + 0.5) * num_subcarriers as f32) as i32; // lower subcarrier index
    let m1 = ((f1 + 0.5) * num_subcarriers as f32) as i32; // upper subcarrier index
    let mp = (m1 - m0).min(num_subcarriers as i32);
    if mp < 6 {
        return Err(Error::Config("ofdmframe_init_sctype_range(), less than 6 subcarriers (effectively)".into()));
    };

    // designate pilot spacing
    let pilot_spacing = if mp > 34 { 8 } else { 4 };

    // upper band
    for i in 0..num_subcarriers as i32 {
        // shift
        let k = (i as usize + num_subcarriers / 2) % num_subcarriers;
        p[k] = if i < m0 || i > m1 {
            // guard band
            SubcarrierType::Null
        } else if k.is_multiple_of(pilot_spacing) {
            SubcarrierType::Pilot
        } else {
            SubcarrierType::Data
        };
    }

    Ok(())
}

/// validate subcarrier allocation, counting each type
///   p : subcarrier allocation array
///
/// note liquid also rejects allocations holding an out-of-range type byte;
/// [`SubcarrierType`] makes that unrepresentable.
fn ofdmframe_validate_sctype(p: &[SubcarrierType]) -> Result<SubcarrierCounts> {
    // clear counters
    let mut counts = SubcarrierCounts { null: 0, pilot: 0, data: 0 };

    for t in p {
        // update appropriate counters
        match t {
            SubcarrierType::Null => counts.null += 1,
            SubcarrierType::Pilot => counts.pilot += 1,
            SubcarrierType::Data => counts.data += 1,
        }
    }

    if counts.pilot + counts.data == 0 {
        return Err(Error::Config("ofdmframe_validate_sctype(), must have at least one enabled subcarrier".into()));
    }
    if counts.data == 0 {
        return Err(Error::Config("ofdmframe_validate_sctype(), must have at least one data subcarrier".into()));
    }
    if counts.pilot < 2 {
        return Err(Error::Config("ofdmframe_validate_sctype(), must have at least two pilot subcarriers".into()));
    }

    Ok(counts)
}

/// Formats an allocation as a string in frequency order.
///
/// The string is bracketed and uses `.` for null, `|` for pilot and `+` for
/// data subcarriers. For an even number of subcarriers it starts at the most
/// negative frequency, bin `len / 2`, and ends at the highest positive one.
pub fn ofdmframe_sctype_string(p: &[SubcarrierType]) -> String {
    let num_subcarriers = p.len();
    let mut s = String::with_capacity(num_subcarriers + 2);

    s.push('[');
    for i in 0..num_subcarriers {
        let k = (i + num_subcarriers / 2) % num_subcarriers;
        s.push(match p[k] {
            SubcarrierType::Null => '.',
            SubcarrierType::Pilot => '|',
            SubcarrierType::Data => '+',
        });
    }
    s.push(']');

    s
}

/// Parses an allocation string written in frequency order.
///
/// The string uses the format of [`ofdmframe_sctype_string`], and the result
/// is in natural FFT-bin order. The two functions are inverses for an even
/// number of subcarriers. With an odd number, the parsed allocation is
/// rotated by one bin.
///
/// # Errors
///
/// Returns an error when the string is not bracketed or holds a character
/// other than `.`, `|` or `+`.
pub fn ofdmframe_sctype_from_string(s: &str) -> Result<Vec<SubcarrierType>> {
    if s.len() < 2 || !s.starts_with('[') || !s.ends_with(']') {
        return Err(Error::Config("ofdmframe_sctype_from_string(), string must be bracketed".into()));
    }
    if !s.is_ascii() {
        return Err(Error::Config("ofdmframe_sctype_from_string(), string must be ASCII".into()));
    }
    let num_subcarriers = s.len() - 2;
    let mut p = Vec::with_capacity(num_subcarriers);
    for i in 0..num_subcarriers {
        let k = (i + num_subcarriers / 2) % num_subcarriers;
        let c = s.chars().nth(1 + k).unwrap();
        let t = match c {
            '.' => SubcarrierType::Null,
            '|' => SubcarrierType::Pilot,
            '+' => SubcarrierType::Data,
            _ => return Err(Error::Config(format!("ofdmframe_sctype_from_string(), invalid character '{}'", c))),
        };
        p.push(t);
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    fn ofdm_range_rejects_nan_edges() {
        let mut types = [SubcarrierType::Null; 64];
        assert!(ofdmframe_init_sctype_range(f32::NAN, 0.25, &mut types).is_err());
        assert!(ofdmframe_init_sctype_range(-0.25, f32::NAN, &mut types).is_err());
        assert!(types.iter().all(|t| *t == SubcarrierType::Null));
    }

    #[test]
    fn test_frame_config_validation_and_accessors() {
        assert!(OfdmFrameConfig::new(0, 16, 4, None).is_err());
        assert!(OfdmFrameConfig::new(7, 16, 4, None).is_err());
        assert!(OfdmFrameConfig::new(65, 16, 4, None).is_err());
        assert!(OfdmFrameConfig::new(64, 65, 4, None).is_err());
        assert!(OfdmFrameConfig::new(64, 16, 17, None).is_err());

        let mut short = vec![SubcarrierType::Null; 63];
        assert!(OfdmFrameConfig::new(64, 16, 4, Some(&short)).is_err());

        short.push(SubcarrierType::Null);
        assert!(OfdmFrameConfig::new(64, 16, 4, Some(&short)).is_err());

        let config = OfdmFrameConfig::new(64, 16, 4, None).unwrap();
        assert_eq!(config.num_subcarriers(), 64);
        assert_eq!(config.cp_len(), 16);
        assert_eq!(config.taper_len(), 4);
        assert_eq!(config.symbol_len(), 80);
        assert_eq!(config.allocation().len(), 64);
        let counts = config.counts();
        assert_eq!(counts.null + counts.pilot + counts.data, 64);
    }

    #[test]
    fn frame_config_requires_an_enabled_even_bin_for_s0() {
        let mut allocation = vec![SubcarrierType::Null; 64];
        allocation[1] = SubcarrierType::Pilot;
        allocation[3] = SubcarrierType::Pilot;
        allocation[5] = SubcarrierType::Data;
        assert!(OfdmFrameConfig::new(64, 16, 4, Some(&allocation)).is_err());

        allocation[2] = SubcarrierType::Data;
        let config = OfdmFrameConfig::new(64, 16, 4, Some(&allocation)).unwrap();
        assert_eq!(config.counts().pilot, 2);
        assert_eq!(config.counts().data, 2);
        assert!(crate::multichannel::OfdmFrameGenerator::new(&config).is_ok());
        assert!(crate::multichannel::OfdmFrameSynchronizer::new(&config).is_ok());
    }

    #[test]
    #[autotest_annotate(autotest_ofdmframe_common_config)]
    fn test_ofdmframe_common_config() {
        // check invalid function calls
        let mut empty: [SubcarrierType; 0] = [];
        assert!(ofdmframe_init_default_sctype(&mut empty).is_err()); // too few subcarriers

        assert!(ofdmframe_init_sctype_range(-0.4, 0.4, &mut empty).is_err()); // too few subcarriers

        let mut p = vec![SubcarrierType::Null; 64];
        assert!(ofdmframe_init_sctype_range(-0.7, 0.4, &mut p).is_err()); // frequency out of range
        assert!(ofdmframe_init_sctype_range(-0.4, 0.7, &mut p).is_err()); // frequency out of range
        assert!(ofdmframe_init_sctype_range(-0.2, -0.3, &mut p).is_err()); // frequency out of range
        assert!(ofdmframe_init_sctype_range(0.3, 0.2, &mut p).is_err()); // frequency out of range
        assert!(ofdmframe_init_sctype_range(-0.02, 0.02, &mut p).is_err()); // too few effective

        // generate valid subcarrier allocation
        let num_subcarriers = 120;
        let mut p = vec![SubcarrierType::Null; num_subcarriers];

        // default subcarrier allocation
        assert!(ofdmframe_init_default_sctype(&mut p).is_ok());
        assert!(ofdmframe_validate_sctype(&p).is_ok());

        // subcarrier allocation within an occupied frequency range
        assert!(ofdmframe_init_sctype_range(-0.4, 0.4, &mut p).is_ok());
        assert!(ofdmframe_validate_sctype(&p).is_ok());

        // invalid subcarrier allocations
        p.fill(SubcarrierType::Null);
        assert!(ofdmframe_validate_sctype(&p).is_err());

        p[0] = SubcarrierType::Pilot;
        assert!(ofdmframe_validate_sctype(&p).is_err());

        p[1] = SubcarrierType::Data;
        assert!(ofdmframe_validate_sctype(&p).is_err());
    }

    #[test]
    fn test_validate_rejects_each_way() {
        let mut p = vec![SubcarrierType::Null; 32];

        // nothing enabled at all
        assert!(ofdmframe_validate_sctype(&p).is_err());

        // pilots but no data: passes the "any enabled" check, fails on data
        p[0] = SubcarrierType::Pilot;
        p[1] = SubcarrierType::Pilot;
        p[2] = SubcarrierType::Pilot;
        assert!(ofdmframe_validate_sctype(&p).is_err(), "pilots but no data");

        // data but too few pilots
        p.fill(SubcarrierType::Null);
        p[0] = SubcarrierType::Data;
        assert!(ofdmframe_validate_sctype(&p).is_err(), "no pilots");
        p[1] = SubcarrierType::Pilot;
        assert!(ofdmframe_validate_sctype(&p).is_err(), "one pilot");

        // two pilots and one data is the minimum viable allocation
        p[2] = SubcarrierType::Pilot;
        let counts = ofdmframe_validate_sctype(&p).unwrap();
        assert_eq!(counts.data, 1);
        assert_eq!(counts.pilot, 2);
        assert_eq!(counts.null, 29);
    }

    #[test]
    fn test_default_sctype_layout() {
        let mut p = vec![SubcarrierType::Null; 64];
        ofdmframe_init_default_sctype(&mut p).unwrap();

        // dc subcarrier is always null
        assert_eq!(p[0], SubcarrierType::Null);

        // guard band at the edges: M/10 = 6 subcarriers each side of nyquist
        let g = 64 / 10;
        for i in 0..=g {
            assert_eq!(p[32 - i], SubcarrierType::Null, "upper guard {}", i);
            assert_eq!(p[32 + i], SubcarrierType::Null, "lower guard {}", i);
        }

        let counts = ofdmframe_validate_sctype(&p).unwrap();
        assert_eq!(counts.null + counts.pilot + counts.data, 64);
        // 64 > 34, so pilots are spaced every 8
        assert_eq!(counts.pilot, 6);

        // rendering is centered on dc, so the middle character is the dc null
        let s = ofdmframe_sctype_string(&p);
        assert_eq!(s.len(), 66); // 64 + brackets
        assert_eq!(s.chars().nth(1 + 32).unwrap(), '.');
    }

    #[test]
    fn test_sctype_range_occupies_requested_band() {
        // only subcarriers inside [f0,f1] are enabled
        let num_subcarriers = 128;
        let mut p = vec![SubcarrierType::Null; num_subcarriers];
        ofdmframe_init_sctype_range(-0.25, 0.25, &mut p).unwrap();

        let counts = ofdmframe_validate_sctype(&p).unwrap();
        // half the band is occupied, so about half the subcarriers are null
        assert!(counts.null >= num_subcarriers / 2 - 2);
        assert!(counts.null <= num_subcarriers / 2 + 2);

        // the enabled region is contiguous once centered on dc
        let s = ofdmframe_sctype_string(&p);
        let body: Vec<char> = s.chars().filter(|c| *c != '[' && *c != ']').collect();
        let first = body.iter().position(|c| *c != '.').unwrap();
        let last = body.iter().rposition(|c| *c != '.').unwrap();
        for (i, c) in body.iter().enumerate() {
            if i > first && i < last {
                assert_ne!(*c, '.', "gap inside occupied band at {}", i);
            }
        }
    }

    #[test]
    fn test_sctype_string_roundtrip() {
        let mut p = vec![SubcarrierType::Null; 64];
        ofdmframe_init_default_sctype(&mut p).unwrap();

        let s = ofdmframe_sctype_string(&p);
        let p2 = ofdmframe_sctype_from_string(&s).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn test_sctype_string_rejects_non_ascii() {
        assert!(ofdmframe_sctype_from_string("[€€]").is_err());
    }
}
