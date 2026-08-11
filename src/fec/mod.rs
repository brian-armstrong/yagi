// FEC (Forward Error Correction) module

mod codec;
mod crc;
mod fec;
mod interleaver;
mod scheme;

pub use crc::{
    append_key, check_key, checksum_generate_key, crc16_generate_key, crc24_generate_key,
    crc32_generate_key, crc8_generate_key, generate_key, validate_message, CrcScheme,
};
pub use codec::{
    golay2412_decode_symbol, golay2412_encode_symbol, hamming1511_decode_symbol,
    hamming1511_encode_symbol, hamming3126_decode_symbol, hamming3126_encode_symbol,
    secded2216_decode_symbol, secded2216_encode_symbol, secded3932_decode_symbol,
    secded3932_encode_symbol, secded7264_decode_symbol, secded7264_encode_symbol, SecdedResult,
};
pub use fec::Fec;
pub use interleaver::Interleaver;
pub use scheme::FecScheme;
