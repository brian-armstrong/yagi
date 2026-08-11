//
// (31,26) Hamming code
//
// there is no block codec built on this symbol-level primitive, matching liquid

use crate::utility::bits::bdotprod;

// parity bit coverage mask for encoder (collapsed version of figure
// below, stripping out parity bits P1, P2, P4, P8, P16 and only including
// data bits 1:26)
//
//  bit position    3   5   6   7   8   9   10  11  12  13  14  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30
//                          *               *               *               *               *               *
//  parity bit  P1  x   x   .   x   x   .   x   .   x   .   x   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   = ..11 0110 1010 1101 0101 0101 0101
//  coverage    P2  x   .   x   x   .   x   x   .   .   x   x   .   x   x   .   .   x   x   .   .   x   x   .   .   x   x   = ..10 1101 1001 1011 0011 0011 0011
//              P4  .   x   x   x   .   .   .   x   x   x   x   .   .   .   x   x   x   x   .   .   .   .   x   x   x   x   = ..01 1100 0111 1000 1111 0000 1111
//              P8  .   .   .   .   x   x   x   x   x   x   x   .   .   .   .   .   .   .   x   x   x   x   x   x   x   x   = ..00 0011 1111 1000 0000 1111 1111
//              P16 .   .   .   .   .   .   .   .   .   .   .   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   = ..00 0000 0000 0111 1111 1111 1111
const M1: u32 = 0x036AD555; //  ..11 0110 1010 1101 0101 0101 0101
const M2: u32 = 0x02D9B333; //  ..10 1101 1001 1011 0011 0011 0011
const M4: u32 = 0x01C78F0F; //  ..01 1100 0111 1000 1111 0000 1111
const M8: u32 = 0x003F80FF; //  ..00 0011 1111 1000 0000 1111 1111
const M16: u32 = 0x00007FFF; //  ..00 0000 0000 0111 1111 1111 1111

//  bit position    1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
//                              *               *               *               *               *               *               *
//  parity bit  P1  x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   .   x   = .101 0101 0101 0101 0101 0101 0101 0101
//  coverage    P2  .   x   x   .   .   x   x   .   .   x   x   .   .   x   x   .   .   x   x   .   .   x   x   .   .   x   x   .   .   x   x   = .011 0011 0011 0011 0011 0011 0011 0011
//              P4  .   .   .   x   x   x   x   .   .   .   .   x   x   x   x   .   .   .   .   x   x   x   x   .   .   .   .   x   x   x   x   = .000 1111 0000 1111 0000 1111 0000 1111
//              P8  .   .   .   .   .   .   .   x   x   x   x   x   x   x   x   .   .   .   .   .   .   .   .   x   x   x   x   x   x   x   x   = .000 0000 1111 1111 0000 0000 1111 1111
//              P16 .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   x   = .000 0000 0000 0000 1111 1111 1111 1111
//
// parity bit coverage mask for decoder; used to compute syndromes
// for decoding a received message (see figure, above).
const S1: u32 = 0x55555555; //  .101 0101 0101 0101 0101 0101 0101 0101
const S2: u32 = 0x33333333; //  .011 0011 0011 0011 0011 0011 0011 0011
const S4: u32 = 0x0f0f0f0f; //  .000 1111 0000 1111 0000 1111 0000 1111
const S8: u32 = 0x00ff00ff; //  .000 0000 1111 1111 0000 0000 1111 1111
const S16: u32 = 0x0000ffff; //  .000 0000 0000 0000 1111 1111 1111 1111

pub fn hamming3126_encode_symbol(sym_dec: u32) -> u32 {
    assert!(sym_dec < (1 << 26), "input symbol too large");
    let sym = sym_dec;

    // compute parity bits
    let p1 = bdotprod(sym, M1);
    let p2 = bdotprod(sym, M2);
    let p4 = bdotprod(sym, M4);
    let p8 = bdotprod(sym, M8);
    let p16 = bdotprod(sym, M16);

    // encode symbol by inserting parity bits with data bits to
    // make a 31-bit symbol
    ((sym & 0x00007fff) << 0) //  ..00 0000 0000 0111 1111 1111 1111
        | ((sym & 0x003F8000) << 1) //  ..00 0011 1111 1000 0000 0000 0000
        | ((sym & 0x01C00000) << 2) //  ..01 1100 0000 0000 0000 0000 0000
        | ((sym & 0x02000000) << 3) //  ..10 0000 0000 0000 0000 0000 0000
        | (p1 << 30) // 30 = 31 - 1  (position of P1)
        | (p2 << 29) // 29 = 31 - 2  (position of P2)
        | (p4 << 27) // 27 = 31 - 4  (position of P4)
        | (p8 << 23) // 23 = 31 - 8  (position of P8)
        | (p16 << 15) // 15 = 31 - 16 (position of P16)
}

pub fn hamming3126_decode_symbol(sym_enc: u32) -> u32 {
    assert!(sym_enc < (1u32 << 31), "input symbol too large");
    let mut sym = sym_enc;

    // compute syndrome bits
    let s1 = bdotprod(sym, S1);
    let s2 = bdotprod(sym, S2);
    let s4 = bdotprod(sym, S4);
    let s8 = bdotprod(sym, S8);
    let s16 = bdotprod(sym, S16);

    // index
    let z = (s16 << 4) | (s8 << 3) | (s4 << 2) | (s2 << 1) | s1;

    // flip bit at this position; z > 31 means there are likely too many
    // errors to correct, so just pass without trying to do anything
    if z != 0 && z <= 31 {
        sym ^= 1 << (31 - z);
    }

    // strip data bits from encoded symbol with parity bits
    (sym & 0x00007fff) //  .000 0000 0000 0000 0111 1111 1111 1111
        | ((sym & 0x007f0000) >> 1) //  .000 0000 0111 1111 0000 0000 0000 0000
        | ((sym & 0x07000000) >> 2) //  .000 0111 0000 0000 0000 0000 0000 0000
        | ((sym & 0x10000000) >> 3) //  .001 0000 0000 0000 0000 0000 0000 0000
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_hamming3126_codec)]
    fn test_hamming3126_codec() {
        let n = 26; // input symbol size (bits)
        let k = 31; // encoded symbol size (bits)

        let mut rng = rand::thread_rng();

        for i in 0..k {
            // generate symbol
            let sym_org = rng.gen::<u32>() % (1 << n);

            // encoded symbol
            let sym_enc = hamming3126_encode_symbol(sym_org);

            // received symbol, with bit i corrupted
            let sym_rec = sym_enc ^ (1 << (k - i - 1));

            // decoded symbol
            let sym_dec = hamming3126_decode_symbol(sym_rec);

            assert_eq!(sym_org, sym_dec);
        }
    }

    // 2^26 symbols is too many to sweep exhaustively. sample instead.
    #[test]
    fn test_hamming3126_single_error_sampled() {
        let mut rng = rand::thread_rng();

        for _ in 0..2000 {
            let sym = rng.gen::<u32>() & 0x03ff_ffff;
            let enc = hamming3126_encode_symbol(sym);

            // clean round trip
            assert_eq!(hamming3126_decode_symbol(enc), sym);

            // every single-bit error is correctable
            for bit in 0..31 {
                let rec = enc ^ (1 << bit);
                assert_eq!(
                    hamming3126_decode_symbol(rec),
                    sym,
                    "symbol {:#09x} failed with bit {} flipped",
                    sym,
                    bit
                );
            }
        }
    }


}
