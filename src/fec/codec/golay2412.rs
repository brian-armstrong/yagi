//
// Golay(24,12) half-rate forward error-correction code
//
// References:
//  [Lin:2004] Lin, Shu and Costello, Daniel L. Jr., "Error Control
//      Coding," Prentice Hall, New Jersey, 2nd edition, 2004.
//

// P matrix [12 x 12]
const P: [u32; 12] = [
    0x08ed, 0x01db, 0x03b5, 0x0769, 0x0ed1, 0x0da3, 0x0b47, 0x068f, 0x0d1d, 0x0a3b, 0x0477, 0x0ffe,
];

// generator matrix transposed [24 x 12]
const GT: [u32; 24] = [
    0x08ed, 0x01db, 0x03b5, 0x0769, 0x0ed1, 0x0da3, 0x0b47, 0x068f, 0x0d1d, 0x0a3b, 0x0477, 0x0ffe,
    0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001,
];

// parity check matrix [12 x 24]
const H: [u32; 12] = [
    0x008008ed, 0x004001db, 0x002003b5, 0x00100769, 0x00080ed1, 0x00040da3, 0x00020b47, 0x0001068f,
    0x00008d1d, 0x00004a3b, 0x00002477, 0x00001ffe,
];

// multiply input vector with parity check matrix, H
fn matrix_mul(v: u32, a: &[u32]) -> u32 {
    let mut x = 0u32;
    for &row in a {
        x <<= 1;
        // compute dot product mod 2
        x |= (row & v).count_ones() & 1;
    }
    x
}

pub fn golay2412_encode_symbol(sym_dec: u32) -> u32 {
    assert!(sym_dec < (1 << 12), "input symbol too large");

    // compute encoded/transmitted message: v = m*G
    matrix_mul(sym_dec, &GT)
}

// search for p[i] such that w(v+p[i]) <= 2, return None on fail
fn parity_search(v: u32) -> Option<usize> {
    P.iter().position(|&p| (v ^ p).count_ones() <= 2)
}

pub fn golay2412_decode_symbol(sym_enc: u32) -> u32 {
    assert!(sym_enc < (1 << 24), "input symbol too large");

    // compute syndrome vector, s = r*H^T = ( H*r^T )^T
    let s = matrix_mul(sym_enc, &H);

    // compute weight of s (12 bits)
    let ws = s.count_ones();

    // step 2:
    let e_hat = if ws <= 3 {
        // set e_hat = [s 0(12)]
        (s << 12) & 0xfff000
    } else if let Some(i) = parity_search(s) {
        // step 3: search for p[i] s.t. w(s+p[i]) <= 2
        // vector found!
        // NOTE : uj = 1 << (12-j-1)
        ((s ^ P[i]) << 12) | (1 << (11 - i))
    } else {
        // step 4: compute s*P
        let sp = matrix_mul(s, &P);

        // compute weight of sP (12 bits)
        let wsp = sp.count_ones();

        if wsp == 2 || wsp == 3 {
            // step 5: set e = [0, s*P]
            sp
        } else if let Some(i) = parity_search(sp) {
            // step 6: search for p[i] s.t. w(s*P + p[i]) == 2...
            // vector found!
            // NOTE : uj = 1 << (12-j-1)
            //      [      uj << 1 2    ] [    sP + p[j]    ]
            (1 << (23 - i)) | (sp ^ P[i])
        } else {
            // step 7: decoding error
            0
        }
    };

    // step 8: compute estimated transmitted message: v_hat = r + e_hat
    let v_hat = sym_enc ^ e_hat;

    // compute estimated original message: (last 12 bits of encoded message)
    v_hat & 0x0fff
}

/// encode block of data using Golay(24,12) encoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
///  msg_enc        :   encoded message [size: 1 x 2*dec_msg_len]
pub fn golay2412_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    let dec_msg_len = msg_dec.len();

    // determine remainder of input length / 3
    let r = dec_msg_len % 3;

    let mut i = 0usize; // decoded byte counter
    let mut j = 0usize; // encoded byte counter

    while i < dec_msg_len - r {
        // strip three input bytes (two uncoded symbols)
        let s0 = msg_dec[i] as u32;
        let s1 = msg_dec[i + 1] as u32;
        let s2 = msg_dec[i + 2] as u32;

        // pack into two 12-bit symbols
        let m0 = ((s0 << 4) & 0x0ff0) | ((s1 >> 4) & 0x000f);
        let m1 = ((s1 << 8) & 0x0f00) | (s2 & 0x00ff);

        // encode each 12-bit symbol into a 24-bit symbol
        let v0 = golay2412_encode_symbol(m0);
        let v1 = golay2412_encode_symbol(m1);

        // unpack two 24-bit symbols into six 8-bit bytes
        // retaining order of bits in output
        msg_enc[j] = (v0 >> 16) as u8;
        msg_enc[j + 1] = (v0 >> 8) as u8;
        msg_enc[j + 2] = v0 as u8;
        msg_enc[j + 3] = (v1 >> 16) as u8;
        msg_enc[j + 4] = (v1 >> 8) as u8;
        msg_enc[j + 5] = v1 as u8;

        i += 3;
        j += 6;
    }

    // if input length isn't divisible by 3, encode last 1 or two bytes
    while i < dec_msg_len {
        // strip last input symbol
        // extend as 12-bit symbol
        // encode into 24-bit symbol
        let v0 = golay2412_encode_symbol(msg_dec[i] as u32);

        // unpack one 24-bit symbol into three 8-bit bytes, and
        // append to output array
        msg_enc[j] = (v0 >> 16) as u8;
        msg_enc[j + 1] = (v0 >> 8) as u8;
        msg_enc[j + 2] = v0 as u8;

        i += 1;
        j += 3;
    }
}

/// decode block of data using Golay(24,12) decoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 1 x 2*dec_msg_len]
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
pub fn golay2412_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    // determine remainder of input length / 3
    let r = dec_msg_len % 3;

    let mut i = 0usize; // decoded byte counter
    let mut j = 0usize; // encoded byte counter

    while i < dec_msg_len - r {
        // strip six input bytes (two encoded symbols)
        // pack six 8-bit symbols into two 24-bit symbols
        let v0 = ((msg_enc[j] as u32) << 16)
            | ((msg_enc[j + 1] as u32) << 8)
            | (msg_enc[j + 2] as u32);
        let v1 = ((msg_enc[j + 3] as u32) << 16)
            | ((msg_enc[j + 4] as u32) << 8)
            | (msg_enc[j + 5] as u32);

        // decode each symbol into a 12-bit symbol
        let m0_hat = golay2412_decode_symbol(v0);
        let m1_hat = golay2412_decode_symbol(v1);

        // unpack two 12-bit symbols into three 8-bit bytes
        msg_dec[i] = (m0_hat >> 4) as u8;
        msg_dec[i + 1] = (((m0_hat << 4) & 0xf0) | ((m1_hat >> 8) & 0x0f)) as u8;
        msg_dec[i + 2] = m1_hat as u8;

        i += 3;
        j += 6;
    }

    // if input length isn't divisible by 3, decode last 1 or two bytes
    while i < dec_msg_len {
        // strip last input symbol (three bytes)
        // pack three 8-bit symbols into one 24-bit symbol
        let v0 = ((msg_enc[j] as u32) << 16)
            | ((msg_enc[j + 1] as u32) << 8)
            | (msg_enc[j + 2] as u32);

        // decode into a 12-bit symbol
        // retain last 8 bits of 12-bit symbol
        msg_dec[i] = golay2412_decode_symbol(v0) as u8;

        i += 1;
        j += 3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    // generate random error vector with 'n' ones;
    // maybe not efficient but effective
    fn generate_error_vector(n: usize) -> u32 {
        assert!(n <= 24, "cannot generate more than 24 errors");

        let mut rng = rand::thread_rng();
        let mut error_locations = [false; 24];

        for _ in 0..n {
            loop {
                // generate random error location
                let t = rng.gen_range(0..24);

                // check error location
                if !error_locations[t] {
                    error_locations[t] = true;
                    break;
                }
            }
        }

        // generate error vector
        let mut e = 0u32;
        for (i, &loc) in error_locations.iter().enumerate() {
            if loc {
                e |= 1 << i;
            }
        }
        e
    }

    #[test]
    #[autotest_annotate(autotest_golay2412_codec)]
    fn test_golay2412_codec() {
        let num_trials = 50; // number of symbol trials

        let mut rng = rand::thread_rng();

        for num_errors in 0..=3 {
            for _ in 0..num_trials {
                // generate symbol
                let sym_org = rng.gen::<u32>() % (1 << 12);

                // encoded symbol
                let sym_enc = golay2412_encode_symbol(sym_org);

                // generate error vector
                let e = generate_error_vector(num_errors);

                // received symbol
                let sym_rec = sym_enc ^ e;

                // decoded symbol
                let sym_dec = golay2412_decode_symbol(sym_rec);

                // validate data are the same
                assert_eq!(sym_org, sym_dec, "failed with {} errors", num_errors);
            }
        }
    }

    #[test]
    fn test_golay2412_roundtrip_exhaustive() {
        for sym in 0u32..(1 << 12) {
            assert_eq!(golay2412_decode_symbol(golay2412_encode_symbol(sym)), sym);
        }
    }

    #[test]
    fn test_golay2412_single_error_exhaustive() {
        for sym in 0u32..(1 << 12) {
            let enc = golay2412_encode_symbol(sym);
            for bit in 0..24 {
                assert_eq!(
                    golay2412_decode_symbol(enc ^ (1 << bit)),
                    sym,
                    "symbol {:#05x} failed with bit {} flipped",
                    sym,
                    bit
                );
            }
        }
    }

    #[test]
    fn test_golay2412_multi_error_exhaustive_patterns() {
        let mut rng = rand::thread_rng();

        for _ in 0..8 {
            let sym = rng.gen::<u32>() % (1 << 12);
            let enc = golay2412_encode_symbol(sym);

            for a in 0..24 {
                for b in (a + 1)..24 {
                    let e = (1 << a) | (1 << b);
                    assert_eq!(
                        golay2412_decode_symbol(enc ^ e),
                        sym,
                        "symbol {:#05x} failed with bits {},{} flipped",
                        sym,
                        a,
                        b
                    );

                    for c in (b + 1)..24 {
                        let e = e | (1 << c);
                        assert_eq!(
                            golay2412_decode_symbol(enc ^ e),
                            sym,
                            "symbol {:#05x} failed with bits {},{},{} flipped",
                            sym,
                            a,
                            b,
                            c
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_golay2412_block_roundtrip() {
        let msg = [0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34];
        let mut encoded = [0u8; 12]; // 2x input length
        let mut decoded = [0u8; 6];

        golay2412_encode(&msg, &mut encoded);
        golay2412_decode(6, &encoded, &mut decoded);

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_golay2412_block_remainder_lengths() {
        for n in 1..=10usize {
            let msg: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11)).collect();

            let r = n % 3;
            let enc_len = (n - r) / 3 * 6 + r * 3;
            let mut encoded = vec![0u8; enc_len];
            let mut decoded = vec![0u8; n];

            golay2412_encode(&msg, &mut encoded);
            golay2412_decode(n, &encoded, &mut decoded);

            assert_eq!(msg, decoded, "block round trip failed for length {}", n);
        }
    }


}
