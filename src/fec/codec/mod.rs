// FEC codec implementations

mod conv_params;
mod convolutional;
mod golay2412;
mod hamming128;
mod hamming1511;
mod hamming3126;
mod hamming74;
mod hamming84;
mod pass;
mod reedsolomon;
mod rep3;
mod rep5;
mod secded;
#[cfg(test)]
mod test_data_conv;
#[cfg(test)]
mod test_data_rs;

pub use golay2412::{
    golay2412_decode, golay2412_decode_symbol, golay2412_encode, golay2412_encode_symbol,
};
pub use hamming128::{hamming128_decode, hamming128_decode_soft, hamming128_encode};
// for the config autotest, which checks its symbol bound
#[cfg(test)]
pub(crate) use hamming128::decode_symbol as hamming128_decode_symbol;
pub use hamming1511::{hamming1511_decode_symbol, hamming1511_encode_symbol};
pub use hamming3126::{hamming3126_decode_symbol, hamming3126_encode_symbol};
pub use hamming74::{hamming74_decode, hamming74_decode_soft, hamming74_encode};
pub use hamming84::{hamming84_decode, hamming84_decode_soft, hamming84_encode};
pub use pass::{pass_decode, pass_encode};
pub use convolutional::{conv_scheme_params, Convolutional};
pub use reedsolomon::ReedSolomon;
pub use rep3::{rep3_decode, rep3_decode_soft, rep3_encode};
pub use rep5::{rep5_decode, rep5_decode_soft, rep5_encode};
pub use secded::{
    secded2216_decode, secded2216_decode_symbol, secded2216_encode, secded2216_encode_symbol,
    secded3932_decode, secded3932_decode_symbol, secded3932_encode, secded3932_encode_symbol,
    secded7264_decode, secded7264_decode_symbol, secded7264_encode, secded7264_encode_symbol,
    SecdedResult,
};
