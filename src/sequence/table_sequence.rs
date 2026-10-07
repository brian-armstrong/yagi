use super::MaximalLengthSequence;
use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::FRAC_1_SQRT_2;

/// A table of complex chips read at an integer position.
///
/// Each [`Self::step`] moves to the next chip and wraps from the last chip to
/// the first, so the position always matches the number of steps taken.
/// Mixing leaves the position unchanged. Spreading and despreading step once
/// per chip, so a call over a full table cycle ends where it started.
///
/// [`Self::from_msequence`] builds a QPSK table from an m-sequence, and
/// [`Self::new`] takes any table.
#[derive(Clone, Debug)]
pub struct TableSequence {
    table: Vec<Complex32>,
    position: usize,
}

impl TableSequence {
    /// Creates a sequence from a copy of `table`, starting at its first chip.
    ///
    /// # Errors
    ///
    /// Returns an error when `table` is empty.
    pub fn new(table: &[Complex32]) -> Result<Self> {
        if table.is_empty() {
            return Err(Error::Config("table sequence must contain at least one chip".into()));
        }
        Ok(Self { table: table.to_vec(), position: 0 })
    }

    /// Creates a QPSK table of `length` chips from the bits of `sequence`.
    ///
    /// Each chip takes the next two bits, the first for the real part and the
    /// second for the imaginary part. A zero bit gives `-1/sqrt(2)` and a one
    /// gives `1/sqrt(2)`, so every chip has unit magnitude. `sequence` advances
    /// by `2 * length` bits from its current state and is not reset.
    ///
    /// When `length` is a multiple of the period of `sequence`, the table
    /// continues the chip stream exactly as it wraps. Other lengths break the
    /// stream at the wrap.
    ///
    /// # Errors
    ///
    /// Returns an error when `length` is zero. `sequence` is unchanged on
    /// error.
    pub fn from_msequence(sequence: &mut MaximalLengthSequence, length: usize) -> Result<Self> {
        if length == 0 {
            return Err(Error::Config("table sequence must contain at least one chip".into()));
        }
        let mut table = Vec::with_capacity(length);
        for _ in 0..length {
            let re = if sequence.advance() != 0 { FRAC_1_SQRT_2 } else { -FRAC_1_SQRT_2 };
            let im = if sequence.advance() != 0 { FRAC_1_SQRT_2 } else { -FRAC_1_SQRT_2 };
            table.push(Complex32::new(re, im));
        }
        Ok(Self { table, position: 0 })
    }

    /// Returns the number of chips in the table.
    pub fn length(&self) -> usize {
        self.table.len()
    }

    /// Returns the zero-based position of the current chip.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Moves to the chip at `position`.
    ///
    /// # Errors
    ///
    /// Returns an error when `position` is not less than [`Self::length`]. The
    /// position is unchanged on error.
    pub fn set_position(&mut self, position: usize) -> Result<()> {
        if position >= self.table.len() {
            return Err(Error::Range("table sequence position out of range".into()));
        }
        self.position = position;
        Ok(())
    }

    /// Moves to the first chip.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Moves to the next chip, wrapping from the last chip to the first.
    pub fn step(&mut self) {
        self.position = if self.position + 1 == self.table.len() { 0 } else { self.position + 1 };
    }

    /// Returns the chip at the current position.
    pub fn current(&self) -> Complex32 {
        self.table[self.position]
    }

    /// Returns the chip before the current one, which is the last chip when
    /// the position is zero.
    pub fn previous(&self) -> Complex32 {
        let index = if self.position == 0 { self.table.len() - 1 } else { self.position - 1 };
        self.table[index]
    }

    /// Returns the chip after the current one, which is the first chip when
    /// the position is the last.
    pub fn next(&self) -> Complex32 {
        let index = if self.position + 1 == self.table.len() { 0 } else { self.position + 1 };
        self.table[index]
    }

    /// Returns the average of the previous and current chips.
    pub fn half_previous(&self) -> Complex32 {
        (self.previous() + self.current()) / 2.0
    }

    /// Returns the average of the current and next chips.
    pub fn half_next(&self) -> Complex32 {
        (self.current() + self.next()) / 2.0
    }

    /// Returns `input` multiplied by the current chip. The position is
    /// unchanged.
    pub fn mix_up(&self, input: Complex32) -> Complex32 {
        input * self.current()
    }

    /// Returns `input` multiplied by the conjugate of the current chip. The
    /// position is unchanged.
    pub fn mix_down(&self, input: Complex32) -> Complex32 {
        input * self.current().conj()
    }

    /// Spreads `symbol` across one full table cycle, writing one chip per
    /// table entry to `output`.
    ///
    /// The position ends where it started.
    ///
    /// # Errors
    ///
    /// Returns an error when the length of `output` differs from
    /// [`Self::length`]. The sequence and `output` are unchanged on error.
    pub fn spread(&mut self, symbol: Complex32, output: &mut [Complex32]) -> Result<()> {
        if output.len() != self.table.len() {
            return Err(Error::Range("spread output length must match table length".into()));
        }
        self.spread_partial(symbol, output)
    }

    /// Spreads `symbol` across `output`, stepping once per chip.
    ///
    /// `output` can be shorter than the table and can cross its end, so a
    /// symbol can span part of a table cycle.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is empty or longer than the table. The
    /// sequence and `output` are unchanged on error.
    pub fn spread_partial(&mut self, symbol: Complex32, output: &mut [Complex32]) -> Result<()> {
        self.check_partial_length(output.len())?;
        for chip in output {
            *chip = self.mix_up(symbol);
            self.step();
        }
        Ok(())
    }

    /// Correlates one full table cycle of `input` against the table and
    /// returns the normalized result.
    ///
    /// The correlation is divided by the sum of `|input[i]|` times the
    /// magnitude of its chip, so its magnitude is at most one. A symbol spread
    /// by [`Self::spread`] from the same position comes back as
    /// `symbol / |symbol|`. When every term of that sum is zero, such as for
    /// an all-zero `input`, the result is NaN. The position ends where it
    /// started.
    ///
    /// # Errors
    ///
    /// Returns an error when the length of `input` differs from
    /// [`Self::length`]. The sequence is unchanged on error.
    pub fn despread(&mut self, input: &[Complex32]) -> Result<Complex32> {
        if input.len() != self.table.len() {
            return Err(Error::Range("despread input length must match table length".into()));
        }
        self.despread_partial(input)
    }

    /// Correlates `input` as [`Self::despread`] does, stepping once per chip.
    ///
    /// `input` can be shorter than the table and can cross its end.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` is empty or longer than the table. The
    /// sequence is unchanged on error.
    pub fn despread_partial(&mut self, input: &[Complex32]) -> Result<Complex32> {
        self.check_partial_length(input.len())?;
        let mut correlation = Complex32::new(0.0, 0.0);
        let mut norm = 0.0f32;
        for &chip in input {
            correlation += self.mix_down(chip);
            norm += chip.norm() * self.current().norm();
            self.step();
        }
        Ok(correlation / norm)
    }

    /// Correlates one full table cycle of `input` at early, punctual and late
    /// timing, and returns the three results in that order.
    ///
    /// The punctual reference is the current chip, and the early and late
    /// references are [`Self::half_previous`] and [`Self::half_next`]. Each
    /// result is normalized separately, as in [`Self::despread`]. The position
    /// ends where it started.
    ///
    /// # Errors
    ///
    /// Returns an error when the length of `input` differs from
    /// [`Self::length`]. The sequence is unchanged on error.
    pub fn despread_triple(&mut self, input: &[Complex32]) -> Result<(Complex32, Complex32, Complex32)> {
        if input.len() != self.table.len() {
            return Err(Error::Range("despread input length must match table length".into()));
        }
        self.despread_triple_partial(input)
    }

    /// Correlates `input` as [`Self::despread_triple`] does, stepping once per
    /// chip.
    ///
    /// `input` can be shorter than the table and can cross its end.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` is empty or longer than the table. The
    /// sequence is unchanged on error.
    pub fn despread_triple_partial(&mut self, input: &[Complex32]) -> Result<(Complex32, Complex32, Complex32)> {
        self.check_partial_length(input.len())?;
        let mut early = Complex32::new(0.0, 0.0);
        let mut punctual = Complex32::new(0.0, 0.0);
        let mut late = Complex32::new(0.0, 0.0);
        let mut early_norm = 0.0f32;
        let mut punctual_norm = 0.0f32;
        let mut late_norm = 0.0f32;

        for &chip in input {
            let current = self.current();
            let previous = self.half_previous();
            let next = self.half_next();
            early += chip * previous.conj();
            punctual += chip * current.conj();
            late += chip * next.conj();
            let magnitude = chip.norm();
            early_norm += magnitude * previous.norm();
            punctual_norm += magnitude * current.norm();
            late_norm += magnitude * next.norm();
            self.step();
        }

        Ok((early / early_norm, punctual / punctual_norm, late / late_norm))
    }

    fn check_partial_length(&self, length: usize) -> Result<()> {
        if length == 0 || length > self.table.len() {
            return Err(Error::Range("chip count must be in 1..=table length".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_sequence_walks_exactly_and_wraps() {
        assert!(TableSequence::new(&[]).is_err());
        for length in [1usize, 3, 1023, 4096] {
            let table: Vec<_> = (0..length).map(|i| Complex32::new(i as f32, 0.0)).collect();
            let mut sequence = TableSequence::new(&table).unwrap();
            for step in 0..300_000 {
                assert_eq!(sequence.position(), step % length);
                assert_eq!(sequence.current(), table[step % length]);
                sequence.step();
            }
            sequence.reset();
            assert_eq!(sequence.position(), 0);
        }
    }

    #[test]
    fn table_sequence_position_neighbors_and_mixing() {
        let table = [Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0), Complex32::new(-1.0, 0.0)];
        let mut sequence = TableSequence::new(&table).unwrap();
        assert_eq!(sequence.length(), 3);
        assert_eq!(sequence.previous(), table[2]);
        assert_eq!(sequence.next(), table[1]);
        assert_eq!(sequence.half_previous(), (table[2] + table[0]) / 2.0);
        assert_eq!(sequence.half_next(), (table[0] + table[1]) / 2.0);
        sequence.set_position(1).unwrap();
        let symbol = Complex32::new(2.0, -1.0);
        assert_eq!(sequence.mix_up(symbol), symbol * table[1]);
        assert_eq!(sequence.mix_down(symbol), symbol * table[1].conj());
        assert_eq!(sequence.position(), 1);
        assert!(sequence.set_position(3).is_err());
        assert_eq!(sequence.position(), 1);
    }

    #[test]
    fn table_sequence_from_msequence_matches_qpsk_bits() {
        let mut source = MaximalLengthSequence::from_degree(3).unwrap();
        let mut reference = source;
        assert!(TableSequence::from_msequence(&mut source, 0).is_err());
        assert_eq!(source.state(), reference.state());

        let mut sequence = TableSequence::from_msequence(&mut source, 9).unwrap();
        for _ in 0..9 {
            let re = if reference.advance() == 0 { -FRAC_1_SQRT_2 } else { FRAC_1_SQRT_2 };
            let im = if reference.advance() == 0 { -FRAC_1_SQRT_2 } else { FRAC_1_SQRT_2 };
            assert_eq!(sequence.current(), Complex32::new(re, im));
            sequence.step();
        }
        assert_eq!(source.state(), reference.state());
        assert_eq!(sequence.position(), 0);
    }

    #[test]
    fn table_sequence_partial_spread_and_despread_cross_wrap() {
        let table = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, -1.0),
            Complex32::new(1.0, 1.0),
        ];
        let symbol = Complex32::new(0.6, -0.8);
        let mut tx = TableSequence::new(&table).unwrap();
        tx.set_position(4).unwrap();
        let mut chips = [Complex32::new(0.0, 0.0); 3];
        tx.spread_partial(symbol, &mut chips).unwrap();
        assert_eq!(chips, [symbol * table[4], symbol * table[0], symbol * table[1]]);
        assert_eq!(tx.position(), 2);

        let mut rx = TableSequence::new(&table).unwrap();
        rx.set_position(4).unwrap();
        let recovered = rx.despread_partial(&chips).unwrap();
        assert!((recovered - symbol / symbol.norm()).norm() < 1e-6);
        assert_eq!(rx.position(), 2);

        let mut full = [Complex32::new(0.0, 0.0); 5];
        tx.spread(symbol, &mut full).unwrap();
        assert_eq!(tx.position(), 2);
        rx.set_position(2).unwrap();
        assert!((rx.despread(&full).unwrap() - symbol / symbol.norm()).norm() < 1e-6);
        assert_eq!(rx.position(), 2);
    }

    #[test]
    fn table_sequence_triple_correlations_and_rejected_lengths() {
        let table = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, -1.0),
        ];
        let mut sequence = TableSequence::new(&table).unwrap();
        sequence.set_position(3).unwrap();
        let input = [table[3], table[0], table[1]];

        let mut expected = [Complex32::new(0.0, 0.0); 3];
        let mut norms = [0.0f32; 3];
        for (i, &chip) in input.iter().enumerate() {
            let position = (3 + i) % table.len();
            let previous = table[(position + table.len() - 1) % table.len()];
            let current = table[position];
            let next = table[(position + 1) % table.len()];
            for (j, reference) in [(previous + current) / 2.0, current, (current + next) / 2.0].into_iter().enumerate()
            {
                expected[j] += chip * reference.conj();
                norms[j] += chip.norm() * reference.norm();
            }
        }
        let actual = sequence.despread_triple_partial(&input).unwrap();
        for (observed, (&sum, &norm)) in [actual.0, actual.1, actual.2].iter().zip(expected.iter().zip(&norms)) {
            assert!((*observed - sum / norm).norm() < 1e-6);
        }
        assert_eq!(sequence.position(), 2);

        let mut output = [Complex32::new(9.0, 9.0); 2];
        let saved = output;
        assert!(sequence.spread(Complex32::new(1.0, 0.0), &mut output).is_err());
        assert!(sequence.spread_partial(Complex32::new(1.0, 0.0), &mut []).is_err());
        assert!(sequence.despread(&input).is_err());
        assert!(sequence.despread_partial(&[]).is_err());
        assert!(sequence.despread_triple_partial(&[]).is_err());
        assert!(sequence.despread_triple(&input).is_err());
        assert_eq!(output, saved);
        assert_eq!(sequence.position(), 2);
    }
}
