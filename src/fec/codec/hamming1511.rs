//
// Hamming(15,11) code
//
// there is no block codec built on this symbol-level primitive, matching liquid

use crate::utility::bits::bdotprod;

// parity bit coverage mask for encoder (collapsed version of figure
// in hamming128, stripping out parity bits P1, P2, P4, P8 and only
// including data bits 1:11)
//
//  parity bit  P1  x   x   .   x   x   .   x   .   x   .   x   = .110 1101 0101
//  coverage    P2  x   .   x   x   .   x   x   .   .   x   x   = .101 1011 0011
//              P4  .   x   x   x   .   .   .   x   x   x   x   = .011 1000 1111
//              P8  .   .   .   .   x   x   x   x   x   x   x   = .000 0111 1111
const M1: u32 = 0x06d5; // .110 1101 0101
const M2: u32 = 0x05b3; // .101 1011 0011
const M4: u32 = 0x038f; // .011 1000 1111
const M8: u32 = 0x007f; // .000 0111 1111

// parity bit coverage mask for decoder; used to compute syndromes
// for decoding a received message (see figure, above).
const S1: u32 = 0x5555; // .101 0101 0101 0101
const S2: u32 = 0x3333; // .011 0011 0011 0011
const S4: u32 = 0x0f0f; // .000 1111 0000 1111
const S8: u32 = 0x00ff; // .000 0000 1111 1111

pub fn hamming1511_encode_symbol(sym_dec: u16) -> u16 {
    assert!(sym_dec < (1 << 11), "input symbol too large");
    let sym = sym_dec as u32;

    // compute parity bits
    let p1 = bdotprod(sym, M1);
    let p2 = bdotprod(sym, M2);
    let p4 = bdotprod(sym, M4);
    let p8 = bdotprod(sym, M8);

    // encode symbol by inserting parity bits with data bits to
    // make a 15-bit symbol
    let sym_enc = ((sym & 0x007f) << 0)
        | ((sym & 0x0380) << 1)
        | ((sym & 0x0400) << 2)
        | (p1 << 14)
        | (p2 << 13)
        | (p4 << 11)
        | (p8 << 7);

    sym_enc as u16
}

pub fn hamming1511_decode_symbol(sym_enc: u16) -> u16 {
    assert!(sym_enc < (1 << 15), "input symbol too large");
    let mut sym = sym_enc as u32;

    // compute syndrome bits
    let s1 = bdotprod(sym, S1);
    let s2 = bdotprod(sym, S2);
    let s4 = bdotprod(sym, S4);
    let s8 = bdotprod(sym, S8);

    // index
    let z = (s8 << 3) | (s4 << 2) | (s2 << 1) | s1;

    // flip bit at this position; z > 15 means there are likely too many
    // errors to correct, so just pass without trying to do anything
    if z != 0 && z <= 15 {
        sym ^= 1 << (15 - z);
    }

    // strip data bits (x) from encoded symbol with parity bits (.)
    //      symbol: [ -..x .xxx .xxx xxxx]
    //                -000 0000 0xxx xxxx   > 0x007f
    //                -000 0111 0000 0000   > 0x0700
    //                -001 0000 0000 0000   > 0x1000
    let sym_dec = (sym & 0x007f) | ((sym & 0x0700) >> 1) | ((sym & 0x1000) >> 2);

    sym_dec as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_hamming1511_codec)]
    fn test_hamming1511_codec() {
        let n = 11; // input symbol size (bits)
        let k = 15; // encoded symbol size (bits)

        let mut rng = rand::thread_rng();

        for i in 0..k {
            // generate symbol
            let sym_org = (rng.gen::<u32>() % (1 << n)) as u16;

            // encoded symbol
            let sym_enc = hamming1511_encode_symbol(sym_org);

            // received symbol, with bit i corrupted
            let sym_rec = sym_enc ^ (1 << (k - i - 1));

            // decoded symbol
            let sym_dec = hamming1511_decode_symbol(sym_rec);

            assert_eq!(sym_org, sym_dec);
        }
    }

    #[test]
    fn test_hamming1511_roundtrip_exhaustive() {
        for sym in 0u16..(1 << 11) {
            assert_eq!(hamming1511_decode_symbol(hamming1511_encode_symbol(sym)), sym);
        }
    }

    #[test]
    fn test_hamming1511_single_error_exhaustive() {
        for sym in 0u16..(1 << 11) {
            let enc = hamming1511_encode_symbol(sym);
            for bit in 0..15 {
                let rec = enc ^ (1 << bit);
                assert_eq!(
                    hamming1511_decode_symbol(rec),
                    sym,
                    "symbol {:#05x} failed with bit {} flipped",
                    sym,
                    bit
                );
            }
        }
    }


}
