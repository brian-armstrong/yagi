//! Forward error correction, error detection, and packet coding.
//!
//! [`Fec`] encodes and decodes one block with a selected [`FecScheme`].
//! [`Packetizer`] adds a CRC, whitening, two FEC stages, and interleaving for
//! fixed-size packets. Soft-decision input contains one `u8` reliability value
//! per bit: zero favors a zero bit and 255 favors a one bit.

mod codec;
mod crc;
mod fec;
mod interleaver;
mod packetizer;
mod scheme;

pub use codec::{
    golay2412_decode_symbol, golay2412_encode_symbol, hamming1511_decode_symbol, hamming1511_encode_symbol,
    hamming3126_decode_symbol, hamming3126_encode_symbol, secded2216_decode_symbol, secded2216_encode_symbol,
    secded3932_decode_symbol, secded3932_encode_symbol, secded7264_decode_symbol, secded7264_encode_symbol,
    SecdedResult,
};
pub use crc::{
    append_key, check_key, checksum_generate_key, crc16_generate_key, crc24_generate_key, crc32_generate_key,
    crc8_generate_key, generate_key, validate_message, CrcScheme,
};
pub use fec::Fec;
pub use interleaver::Interleaver;
pub use packetizer::{packetizer_compute_dec_msg_len, packetizer_compute_enc_msg_len, Packetizer};
pub use scheme::FecScheme;
