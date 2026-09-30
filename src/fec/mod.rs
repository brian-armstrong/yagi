//! Forward error correction, error detection, and packet coding.
//!
//! [`Fec`] encodes and decodes blocks with one [`FecScheme`]. The CRC
//! functions compute and check error-detection keys for a [`CrcScheme`].
//! [`Packetizer`] combines them into a fixed-length packet with a CRC,
//! whitening, and two FEC stages, each followed by an [`Interleaver`].
//!
//! Lengths are in bytes. A decoded message is the data before coding and an
//! encoded message is the coded result. Soft-decision input holds one `u8`
//! value per encoded bit, eight per byte, from the most significant bit to
//! the least. Zero favors a zero bit and 255 favors a one bit.
//!
//! [`Fec`] decoding corrects what it can and does not report whether it
//! succeeded. Check the result with a CRC, directly or through
//! [`Packetizer`].
//!
//! A [`FecScheme::RsM8`] stage in a [`Packetizer`] interleaves whole bytes by
//! default. Enable the `liquid-quirks` feature to use liquid-dsp's depth-4
//! interleaving and exchange RS8 packets with a liquid-dsp peer.

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
