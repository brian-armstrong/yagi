//
// convolutional code polynomials
//
// liquid calls libfec, which has these compiled in. the `fec` crate takes them
// as arguments, so they are duplicated here from the crate's libfec-compatible
// shim (shim/src/lib.rs) to produce identical codewords

/// a convolutional code: inverse rate, constraint length, and generators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvParams {
    /// output bits per input bit
    pub rate: u32,
    /// constraint length
    pub order: u32,
    /// rate polynomials, in octal convention
    pub polys: &'static [u16],
}

// r1/2, K=7, dfree=10
pub const CONV_V27: ConvParams = ConvParams {
    rate: 2,
    order: 7,
    polys: &[0o155, 0o117],
};

// r1/2, K=9, dfree=12
pub const CONV_V29: ConvParams = ConvParams {
    rate: 2,
    order: 9,
    polys: &[0o657, 0o435],
};

// r1/3, K=9, dfree=18
pub const CONV_V39: ConvParams = ConvParams {
    rate: 3,
    order: 9,
    polys: &[0o755, 0o633, 0o447],
};

// r1/6, K=15, dfree<=57 (Heller 1968)
pub const CONV_V615: ConvParams = ConvParams {
    rate: 6,
    order: 15,
    polys: &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
};

// puncturing matrices, from liquid's fec_conv_pmatrix.c; stored as rate rows
// of period entries, row-major, where a true entry means "transmit this bit"

// 2/3-rate K=7 punctured convolutional code
pub const PMATRIX_V27P23: [&[bool]; 2] = [&[true, true], &[true, false]];

// 3/4-rate K=7 punctured convolutional code
pub const PMATRIX_V27P34: [&[bool]; 2] = [&[true, true, false], &[true, false, true]];

// 4/5-rate K=7 punctured convolutional code
pub const PMATRIX_V27P45: [&[bool]; 2] = [
    &[true, true, true, true],
    &[true, false, false, false],
];

// 5/6-rate K=7 punctured convolutional code
pub const PMATRIX_V27P56: [&[bool]; 2] = [
    &[true, true, false, true, false],
    &[true, false, true, false, true],
];

// 6/7-rate K=7 punctured convolutional code
pub const PMATRIX_V27P67: [&[bool]; 2] = [
    &[true, true, true, false, true, false],
    &[true, false, false, true, false, true],
];

// 7/8-rate K=7 punctured convolutional code
pub const PMATRIX_V27P78: [&[bool]; 2] = [
    &[true, true, true, true, false, true, false],
    &[true, false, false, false, true, false, true],
];

// 2/3-rate K=9 punctured convolutional code
pub const PMATRIX_V29P23: [&[bool]; 2] = [&[true, true], &[true, false]];

// 3/4-rate K=9 punctured convolutional code
pub const PMATRIX_V29P34: [&[bool]; 2] = [&[true, true, true], &[true, false, false]];

// 4/5-rate K=9 punctured convolutional code
pub const PMATRIX_V29P45: [&[bool]; 2] = [
    &[true, true, false, true],
    &[true, false, true, false],
];

// 5/6-rate K=9 punctured convolutional code
pub const PMATRIX_V29P56: [&[bool]; 2] = [
    &[true, false, true, true, false],
    &[true, true, false, false, true],
];

// 6/7-rate K=9 punctured convolutional code
pub const PMATRIX_V29P67: [&[bool]; 2] = [
    &[true, true, false, true, true, false],
    &[true, false, true, false, false, true],
];

// 7/8-rate K=9 punctured convolutional code
pub const PMATRIX_V29P78: [&[bool]; 2] = [
    &[true, true, false, true, false, true, true],
    &[true, false, true, false, true, false, false],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_params_well_formed() {
        for params in [CONV_V27, CONV_V29, CONV_V39, CONV_V615] {
            assert_eq!(
                params.polys.len(),
                params.rate as usize,
                "expected {} polynomials for rate {}",
                params.rate,
                params.rate
            );

            for &poly in params.polys {
                assert!(
                    (poly as u32) < (1 << params.order),
                    "polynomial {:#o} does not fit in {} bits",
                    poly,
                    params.order
                );
                // the high bit should be set: a generator of lower degree
                // would mean the stated constraint length is wrong
                assert!(
                    (poly as u32) >= (1 << (params.order - 1)),
                    "polynomial {:#o} has degree below order {}",
                    poly,
                    params.order
                );
            }
        }
    }

    #[test]
    fn test_puncturing_matrices_well_formed() {
        let cases: [(&str, &[&[bool]], usize, usize); 12] = [
            ("v27p23", &PMATRIX_V27P23, 2, 3),
            ("v27p34", &PMATRIX_V27P34, 3, 4),
            ("v27p45", &PMATRIX_V27P45, 4, 5),
            ("v27p56", &PMATRIX_V27P56, 5, 6),
            ("v27p67", &PMATRIX_V27P67, 6, 7),
            ("v27p78", &PMATRIX_V27P78, 7, 8),
            ("v29p23", &PMATRIX_V29P23, 2, 3),
            ("v29p34", &PMATRIX_V29P34, 3, 4),
            ("v29p45", &PMATRIX_V29P45, 4, 5),
            ("v29p56", &PMATRIX_V29P56, 5, 6),
            ("v29p67", &PMATRIX_V29P67, 6, 7),
            ("v29p78", &PMATRIX_V29P78, 7, 8),
        ];

        for (name, matrix, num, den) in cases {
            // one row per output bit of the rate-1/2 base code
            assert_eq!(matrix.len(), 2, "{}: expected 2 rows", name);

            let period = matrix[0].len();
            for row in matrix {
                assert_eq!(row.len(), period, "{}: ragged matrix", name);
            }

            // the period is the numerator of the punctured rate
            assert_eq!(period, num, "{}: unexpected period", name);

            // transmitted bits per period must give the stated rate:
            // `num` input bits produce `den` transmitted bits
            let kept: usize = matrix.iter().flat_map(|r| r.iter()).filter(|&&b| b).count();
            assert_eq!(
                kept, den,
                "{}: keeps {} bits per period, expected {} for rate {}/{}",
                name, kept, den, num, den
            );
        }
    }
}
