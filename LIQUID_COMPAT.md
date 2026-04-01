

## null
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| null                                     | ✅ | ❓ |


## libliquid
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| libliquid                                | ✅ | ❓ |


## agc_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| agc_crcf_dc_gain_control                 | ✅ | ✅ |
| agc_crcf_scale                           | ✅ | ✅ |
| agc_crcf_ac_gain_control                 | ✅ | ✅ |
| agc_crcf_rssi_sinusoid                   | ✅ | ✅ |
| agc_crcf_rssi_noise                      | ✅ | ✅ |
| agc_crcf_squelch                         | ✅ | ✅ |
| agc_crcf_lock                            | ✅ | ✅ |
| agc_crcf_invalid_config                  | ✅ | ✅ |
| agc_crcf_copy                            | ✅ | ✅ |


## cvsd
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| cvsd_rmse_sine                           | ✅ | ❓ |
| cvsd_rmse_sine8                          | ✅ | ❓ |
| cvsd_invalid_config                      | ✅ | ❓ |


## cbuffer
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| cbufferf                                 | ✅ | ❓ |
| cbuffercf                                | ✅ | ❓ |
| cbufferf_flow                            | ✅ | ❓ |
| cbufferf_config                          | ✅ | ❓ |
| cbuffer_copy                             | ✅ | ❓ |


## wdelay
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| wdelayf                                  | ✅ | ✅ |
| wdelay_copy                              | ✅ | ✅ |


## buffer_window
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| window_config_errors                     | ✅ | ✅ |
| windowf                                  | ✅ | ✅ |
| window_copy                              | ✅ | ✅ |


## dotprod_rrrf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| dotprod_rrrf_basic                       | ✅ | ✅ |
| dotprod_rrrf_uneven                      | ✅ | ✅ |
| dotprod_rrrf_struct                      | ✅ | ❓ |
| dotprod_rrrf_struct_align                | ✅ | ❓ |
| dotprod_rrrf_rand01                      | ✅ | ✅ |
| dotprod_rrrf_rand02                      | ✅ | ✅ |
| dotprod_rrrf_struct_lengths              | ✅ | ✅ |
| dotprod_rrrf_struct_vs_ordinal           | ✅ | ✅ |


## dotprod_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| dotprod_crcf_rand01                      | ✅ | ✅ |
| dotprod_crcf_rand02                      | ✅ | ✅ |
| dotprod_crcf_struct_vs_ordinal           | ✅ | ✅ |


## dotprod_cccf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| dotprod_cccf_rand16                      | ✅ | ✅ |
| dotprod_cccf_struct_lengths              | ✅ | ✅ |
| dotprod_cccf_struct_vs_ordinal           | ✅ | ✅ |


## sumsqf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| sumsqf_3                                 | ✅ | ❓ |
| sumsqf_4                                 | ✅ | ❓ |
| sumsqf_7                                 | ✅ | ❓ |
| sumsqf_8                                 | ✅ | ❓ |
| sumsqf_15                                | ✅ | ❓ |
| sumsqf_16                                | ✅ | ❓ |


## sumsqcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| sumsqcf_3                                | ✅ | ❓ |
| sumsqcf_4                                | ✅ | ❓ |
| sumsqcf_7                                | ✅ | ❓ |
| sumsqcf_8                                | ✅ | ❓ |
| sumsqcf_15                               | ✅ | ❓ |
| sumsqcf_16                               | ✅ | ❓ |


## eqlms_cccf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| eqlms_00                                 | ✅ | ✅ |
| eqlms_01                                 | ✅ | ✅ |
| eqlms_02                                 | ✅ | ✅ |
| eqlms_03                                 | ✅ | ✅ |
| eqlms_04                                 | ✅ | ✅ |
| eqlms_05                                 | ✅ | ✅ |
| eqlms_06                                 | ✅ | ✅ |
| eqlms_07                                 | ✅ | ✅ |
| eqlms_08                                 | ✅ | ✅ |
| eqlms_09                                 | ✅ | ✅ |
| eqlms_10                                 | ✅ | ✅ |
| eqlms_11                                 | ✅ | ✅ |
| eqlms_config                             | ✅ | ✅ |
| eqlms_cccf_copy                          | ✅ | ✅ |


## eqrls_rrrf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| eqrls_rrrf_01                            | ✅ | ✅ |
| eqrls_rrrf_copy                          | ✅ | ✅ |


## crc
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| reverse_byte                             | ✅ | ❓ |
| reverse_uint16                           | ✅ | ❓ |
| reverse_uint32                           | ✅ | ❓ |
| checksum                                 | ✅ | ❓ |
| crc8                                     | ✅ | ❓ |
| crc16                                    | ✅ | ❓ |
| crc24                                    | ✅ | ❓ |
| crc32                                    | ✅ | ❓ |
| crc_config                               | ✅ | ❓ |


## fec
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fec_r3                                   | ✅ | ❓ |
| fec_r5                                   | ✅ | ❓ |
| fec_h74                                  | ✅ | ❓ |
| fec_h84                                  | ✅ | ❓ |
| fec_h128                                 | ✅ | ❓ |
| fec_g2412                                | ✅ | ❓ |
| fec_secded2216                           | ✅ | ❓ |
| fec_secded3932                           | ✅ | ❓ |
| fec_secded7264                           | ✅ | ❓ |
| fec_v27                                  | ✅ | ❓ |
| fec_v29                                  | ✅ | ❓ |
| fec_v39                                  | ✅ | ❓ |
| fec_v615                                 | ✅ | ❓ |
| fec_v27p23                               | ✅ | ❓ |
| fec_v27p34                               | ✅ | ❓ |
| fec_v27p45                               | ✅ | ❓ |
| fec_v27p56                               | ✅ | ❓ |
| fec_v27p67                               | ✅ | ❓ |
| fec_v27p78                               | ✅ | ❓ |
| fec_v29p23                               | ✅ | ❓ |
| fec_v29p34                               | ✅ | ❓ |
| fec_v29p45                               | ✅ | ❓ |
| fec_v29p56                               | ✅ | ❓ |
| fec_v29p67                               | ✅ | ❓ |
| fec_v29p78                               | ✅ | ❓ |
| fec_rs8                                  | ✅ | ❓ |


## fec_config
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fec_config                               | ✅ | ❓ |
| fec_str2fec                              | ✅ | ❓ |
| fec_is_convolutional                     | ✅ | ❓ |
| fec_is_punctured                         | ✅ | ❓ |
| fec_is_reedsolomon                       | ✅ | ❓ |
| fec_is_hamming                           | ✅ | ❓ |


## fec_copy
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fec_copy_r3                              | ✅ | ❓ |
| fec_copy_r5                              | ✅ | ❓ |
| fec_copy_h74                             | ✅ | ❓ |
| fec_copy_h84                             | ✅ | ❓ |
| fec_copy_h128                            | ✅ | ❓ |
| fec_copy_g2412                           | ✅ | ❓ |
| fec_copy_secded2216                      | ✅ | ❓ |
| fec_copy_secded3932                      | ✅ | ❓ |
| fec_copy_secded7264                      | ✅ | ❓ |
| fec_copy_v27                             | ✅ | ❓ |
| fec_copy_v29                             | ✅ | ❓ |
| fec_copy_v39                             | ✅ | ❓ |
| fec_copy_v615                            | ✅ | ❓ |
| fec_copy_v27p23                          | ✅ | ❓ |
| fec_copy_v27p34                          | ✅ | ❓ |
| fec_copy_v27p45                          | ✅ | ❓ |
| fec_copy_v27p56                          | ✅ | ❓ |
| fec_copy_v27p67                          | ✅ | ❓ |
| fec_copy_v27p78                          | ✅ | ❓ |
| fec_copy_v29p23                          | ✅ | ❓ |
| fec_copy_v29p34                          | ✅ | ❓ |
| fec_copy_v29p45                          | ✅ | ❓ |
| fec_copy_v29p56                          | ✅ | ❓ |
| fec_copy_v29p67                          | ✅ | ❓ |
| fec_copy_v29p78                          | ✅ | ❓ |
| fec_copy_rs8                             | ✅ | ❓ |


## fec_soft
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fecsoft_r3                               | ✅ | ❓ |
| fecsoft_r5                               | ✅ | ❓ |
| fecsoft_h74                              | ✅ | ❓ |
| fecsoft_h84                              | ✅ | ❓ |
| fecsoft_h128                             | ✅ | ❓ |
| fecsoft_v27                              | ✅ | ❓ |
| fecsoft_v29                              | ✅ | ❓ |
| fecsoft_v39                              | ✅ | ❓ |
| fecsoft_v615                             | ✅ | ❓ |
| fecsoft_v27p23                           | ✅ | ❓ |
| fecsoft_v27p34                           | ✅ | ❓ |
| fecsoft_v27p45                           | ✅ | ❓ |
| fecsoft_v27p56                           | ✅ | ❓ |
| fecsoft_v27p67                           | ✅ | ❓ |
| fecsoft_v27p78                           | ✅ | ❓ |
| fecsoft_v29p23                           | ✅ | ❓ |
| fecsoft_v29p34                           | ✅ | ❓ |
| fecsoft_v29p45                           | ✅ | ❓ |
| fecsoft_v29p56                           | ✅ | ❓ |
| fecsoft_v29p67                           | ✅ | ❓ |
| fecsoft_v29p78                           | ✅ | ❓ |
| fecsoft_rs8                              | ✅ | ❓ |


## fec_golay2412
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| golay2412_codec                          | ✅ | ❓ |


## fec_hamming74
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| hamming74_codec                          | ✅ | ❓ |
| hamming74_codec_soft                     | ✅ | ❓ |


## fec_hamming84
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| hamming84_codec                          | ✅ | ❓ |
| hamming84_codec_soft                     | ✅ | ❓ |


## fec_hamming128
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| hamming128_codec                         | ✅ | ❓ |
| hamming128_codec_soft                    | ✅ | ❓ |


## fec_hamming1511
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| hamming1511_codec                        | ✅ | ❓ |


## fec_hamming3126
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| hamming3126_codec                        | ✅ | ❓ |


## fec_reedsolomon
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| reedsolomon_223_255                      | ✅ | ❓ |


## fec_rep3
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| rep3_codec                               | ✅ | ❓ |


## fec_rep5
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| rep5_codec                               | ✅ | ❓ |


## fec_secded2216
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| secded2216_codec_e0                      | ✅ | ❓ |
| secded2216_codec_e1                      | ✅ | ❓ |
| secded2216_codec_e2                      | ✅ | ❓ |


## fec_secded3932
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| secded3932_codec_e0                      | ✅ | ❓ |
| secded3932_codec_e1                      | ✅ | ❓ |
| secded3932_codec_e2                      | ✅ | ❓ |


## fec_secded7264
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| secded7264_codec_e0                      | ✅ | ❓ |
| secded7264_codec_e1                      | ✅ | ❓ |
| secded7264_codec_e2                      | ✅ | ❓ |


## interleaver
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| interleaver_hard_8                       | ✅ | ❓ |
| interleaver_hard_16                      | ✅ | ❓ |
| interleaver_hard_64                      | ✅ | ❓ |
| interleaver_hard_256                     | ✅ | ❓ |
| interleaver_soft_8                       | ✅ | ❓ |
| interleaver_soft_16                      | ✅ | ❓ |
| interleaver_soft_64                      | ✅ | ❓ |
| interleaver_soft_256                     | ✅ | ❓ |


## packetizer_copy
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| packetizer_copy                          | ✅ | ❓ |


## packetizer
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| packetizer_n16_0_0                       | ✅ | ❓ |
| packetizer_n16_0_1                       | ✅ | ❓ |
| packetizer_n16_0_2                       | ✅ | ❓ |


## asgram
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| asgramcf_copy                            | ✅ | ❓ |


## fft_small
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fft_3                                    | ✅ | ✅ |
| fft_5                                    | ✅ | ✅ |
| fft_6                                    | ✅ | ✅ |
| fft_7                                    | ✅ | ✅ |
| fft_9                                    | ✅ | ✅ |


## fft_radix2
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fft_2                                    | ✅ | ✅ |
| fft_4                                    | ✅ | ✅ |
| fft_8                                    | ✅ | ✅ |
| fft_16                                   | ✅ | ✅ |
| fft_32                                   | ✅ | ✅ |
| fft_64                                   | ✅ | ✅ |


## fft_composite
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fft_10                                   | ✅ | ✅ |
| fft_21                                   | ✅ | ✅ |
| fft_22                                   | ✅ | ✅ |
| fft_24                                   | ✅ | ✅ |
| fft_26                                   | ✅ | ✅ |
| fft_30                                   | ✅ | ✅ |
| fft_35                                   | ✅ | ✅ |
| fft_36                                   | ✅ | ✅ |
| fft_48                                   | ✅ | ✅ |
| fft_63                                   | ✅ | ✅ |
| fft_92                                   | ✅ | ✅ |
| fft_96                                   | ✅ | ✅ |
| fft_120                                  | ✅ | ✅ |
| fft_130                                  | ✅ | ✅ |
| fft_192                                  | ✅ | ✅ |


## fft_prime
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fft_17                                   | ✅ | ✅ |
| fft_43                                   | ✅ | ✅ |
| fft_79                                   | ✅ | ✅ |
| fft_157                                  | ✅ | ✅ |
| fft_317                                  | ✅ | ✅ |
| fft_509                                  | ✅ | ✅ |


## fft_r2r
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fft_r2r_REDFT00_n8                       | ✅ | ❓ |
| fft_r2r_REDFT10_n8                       | ✅ | ❓ |
| fft_r2r_REDFT01_n8                       | ✅ | ❓ |
| fft_r2r_REDFT11_n8                       | ✅ | ❓ |
| fft_r2r_RODFT00_n8                       | ✅ | ❓ |
| fft_r2r_RODFT10_n8                       | ✅ | ❓ |
| fft_r2r_RODFT01_n8                       | ✅ | ❓ |
| fft_r2r_RODFT11_n8                       | ✅ | ❓ |
| fft_r2r_REDFT00_n32                      | ✅ | ❓ |
| fft_r2r_REDFT10_n32                      | ✅ | ❓ |
| fft_r2r_REDFT01_n32                      | ✅ | ❓ |
| fft_r2r_REDFT11_n32                      | ✅ | ❓ |
| fft_r2r_RODFT00_n32                      | ✅ | ❓ |
| fft_r2r_RODFT10_n32                      | ✅ | ❓ |
| fft_r2r_RODFT01_n32                      | ✅ | ❓ |
| fft_r2r_RODFT11_n32                      | ✅ | ❓ |
| fft_r2r_REDFT00_n27                      | ✅ | ❓ |
| fft_r2r_REDFT10_n27                      | ✅ | ❓ |
| fft_r2r_REDFT01_n27                      | ✅ | ❓ |
| fft_r2r_REDFT11_n27                      | ✅ | ❓ |
| fft_r2r_RODFT00_n27                      | ✅ | ❓ |
| fft_r2r_RODFT10_n27                      | ✅ | ❓ |
| fft_r2r_RODFT01_n27                      | ✅ | ❓ |
| fft_r2r_RODFT11_n27                      | ✅ | ❓ |


## fft_shift
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fft_shift_4                              | ✅ | ✅ |
| fft_shift_8                              | ✅ | ✅ |


## spgram
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| spgramcf_noise_440                       | ✅ | ✅ |
| spgramcf_noise_1024                      | ✅ | ✅ |
| spgramcf_noise_1200                      | ✅ | ✅ |
| spgramcf_noise_custom_0                  | ✅ | ✅ |
| spgramcf_noise_custom_1                  | ✅ | ✅ |
| spgramcf_noise_custom_2                  | ✅ | ✅ |
| spgramcf_noise_custom_3                  | ✅ | ✅ |
| spgramcf_noise_hamming                   | ✅ | ✅ |
| spgramcf_noise_hann                      | ✅ | ✅ |
| spgramcf_noise_blackmanharris            | ✅ | ✅ |
| spgramcf_noise_blackmanharris7           | ✅ | ✅ |
| spgramcf_noise_kaiser                    | ✅ | ✅ |
| spgramcf_noise_flattop                   | ✅ | ✅ |
| spgramcf_noise_triangular                | ✅ | ✅ |
| spgramcf_noise_rcostaper                 | ✅ | ✅ |
| spgramcf_noise_kbd                       | ✅ | ✅ |
| spgramcf_signal_00                       | ✅ | ✅ |
| spgramcf_signal_01                       | ✅ | ✅ |
| spgramcf_signal_02                       | ✅ | ✅ |
| spgramcf_signal_03                       | ✅ | ✅ |
| spgramcf_signal_04                       | ✅ | ✅ |
| spgramcf_signal_05                       | ✅ | ✅ |
| spgramcf_counters                        | ✅ | ✅ |
| spgramcf_invalid_config                  | ✅ | ✅ |
| spgramcf_standalone                      | ✅ | ✅ |
| spgramcf_short                           | ✅ | ✅ |
| spgramcf_copy                            | ✅ | ❓ |
| spgramcf_null                            | ✅ | ❓ |
| spgram_gnuplot                           | ✅ | ❓ |


## spwaterfall
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| spwaterfall_invalid_config               | ✅ | ❓ |
| spwaterfallcf_noise_440                  | ✅ | ❓ |
| spwaterfallcf_noise_1024                 | ✅ | ❓ |
| spwaterfallcf_noise_1200                 | ✅ | ❓ |
| spwaterfall_operation                    | ✅ | ❓ |
| spwaterfall_copy                         | ✅ | ❓ |
| spwaterfall_gnuplot                      | ✅ | ❓ |


## dds_cccf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| dds_cccf_0                               | ✅ | ❓ |
| dds_cccf_1                               | ✅ | ❓ |
| dds_cccf_2                               | ✅ | ❓ |
| dds_config                               | ✅ | ❓ |
| dds_copy                                 | ✅ | ❓ |


## fdelay_rrrf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fdelay_rrrf_0                            | ✅ | ✅ |
| fdelay_rrrf_1                            | ✅ | ✅ |
| fdelay_rrrf_2                            | ✅ | ✅ |
| fdelay_rrrf_3                            | ✅ | ✅ |
| fdelay_rrrf_4                            | ✅ | ✅ |
| fdelay_rrrf_5                            | ✅ | ✅ |
| fdelay_rrrf_6                            | ✅ | ✅ |
| fdelay_rrrf_7                            | ✅ | ✅ |
| fdelay_rrrf_8                            | ✅ | ✅ |
| fdelay_rrrf_9                            | ✅ | ✅ |
| fdelay_rrrf_config                       | ✅ | ✅ |
| fdelay_rrrf_push_write                   | ✅ | ✅ |


## fftfilt_xxxf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fftfilt_rrrf_data_h4x256                 | ✅ | ✅ |
| fftfilt_rrrf_data_h7x256                 | ✅ | ✅ |
| fftfilt_rrrf_data_h13x256                | ✅ | ✅ |
| fftfilt_rrrf_data_h23x256                | ✅ | ✅ |
| fftfilt_crcf_data_h4x256                 | ✅ | ✅ |
| fftfilt_crcf_data_h7x256                 | ✅ | ✅ |
| fftfilt_crcf_data_h13x256                | ✅ | ✅ |
| fftfilt_crcf_data_h23x256                | ✅ | ✅ |
| fftfilt_cccf_data_h4x256                 | ✅ | ✅ |
| fftfilt_cccf_data_h7x256                 | ✅ | ✅ |
| fftfilt_cccf_data_h13x256                | ✅ | ✅ |
| fftfilt_cccf_data_h23x256                | ✅ | ✅ |
| fftfilt_config                           | ✅ | ✅ |
| fftfilt_copy                             | ✅ | ✅ |


## filter_crosscorr
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| filter_crosscorr_rrrf                    | ✅ | ✅ |


## firdecim
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firdecim_config                          | ✅ | ✅ |
| firdecim_block                           | ✅ | ✅ |


## firdecim_xxxf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firdecim_rrrf_common                     | ✅ | ✅ |
| firdecim_crcf_common                     | ✅ | ✅ |
| firdecim_rrrf_data_M2h4x20               | ✅ | ✅ |
| firdecim_rrrf_data_M3h7x30               | ✅ | ✅ |
| firdecim_rrrf_data_M4h13x40              | ✅ | ✅ |
| firdecim_rrrf_data_M5h23x50              | ✅ | ✅ |
| firdecim_crcf_data_M2h4x20               | ✅ | ✅ |
| firdecim_crcf_data_M3h7x30               | ✅ | ✅ |
| firdecim_crcf_data_M4h13x40              | ✅ | ✅ |
| firdecim_crcf_data_M5h23x50              | ✅ | ✅ |
| firdecim_cccf_data_M2h4x20               | ✅ | ✅ |
| firdecim_cccf_data_M3h7x30               | ✅ | ✅ |
| firdecim_cccf_data_M4h13x40              | ✅ | ✅ |
| firdecim_cccf_data_M5h23x50              | ✅ | ✅ |


## firdes
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| liquid_firdes_rcos                       | ✅ | ✅ |
| liquid_firdes_rrcos                      | ✅ | ✅ |
| firdes_rrcos                             | ✅ | ✅ |
| firdes_rkaiser                           | ✅ | ✅ |
| firdes_arkaiser                          | ✅ | ✅ |
| liquid_firdes_dcblock                    | ✅ | ✅ |
| liquid_firdes_notch                      | ✅ | ✅ |
| liquid_getopt_str2firfilt                | ✅ | ✅ |
| liquid_firdes_config                     | ✅ | ✅ |
| liquid_firdes_estimate                   | ✅ | ✅ |
| firdes_prototype_kaiser                  | ✅ | ✅ |
| firdes_prototype_pm                      | ✅ | ✅ |
| firdes_prototype_rcos                    | ✅ | ✅ |
| firdes_prototype_fexp                    | ✅ | ✅ |
| firdes_prototype_fsech                   | ✅ | ✅ |
| firdes_prototype_farcsech                | ✅ | ✅ |
| firdes_prototype_arkaiser                | ✅ | ✅ |
| firdes_prototype_rkaiser                 | ✅ | ✅ |
| firdes_prototype_rrcos                   | ✅ | ✅ |
| firdes_prototype_hm3                     | ✅ | ✅ |
| firdes_prototype_rfexp                   | ✅ | ✅ |
| firdes_prototype_rfsech                  | ✅ | ✅ |
| firdes_prototype_rfarcsech               | ✅ | ✅ |
| firdes_doppler                           | ✅ | ✅ |
| liquid_freqrespf                         | ✅ | ✅ |
| liquid_freqrespcf                        | ✅ | ✅ |


## firdespm
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firdespm_bandpass_n24                    | ✅ | ✅ |
| firdespm_bandpass_n32                    | ✅ | ✅ |
| firdespm_lowpass                         | ✅ | ✅ |
| firdespm_callback                        | ✅ | ✅ |
| firdespm_halfband_m2_ft400               | ✅ | ✅ |
| firdespm_halfband_m4_ft400               | ✅ | ✅ |
| firdespm_halfband_m4_ft200               | ✅ | ✅ |
| firdespm_halfband_m10_ft200              | ✅ | ✅ |
| firdespm_halfband_m12_ft100              | ✅ | ✅ |
| firdespm_halfband_m20_ft050              | ✅ | ✅ |
| firdespm_halfband_m40_ft050              | ✅ | ✅ |
| firdespm_halfband_m80_ft010              | ✅ | ✅ |
| firdespm_copy                            | ✅ | ✅ |
| firdespm_config                          | ✅ | ✅ |
| firdespm_differentiator                  | ✅ | ✅ |
| firdespm_hilbert                         | ✅ | ✅ |


## firfilt
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firfilt_crcf_kaiser                      | ✅ | ✅ |
| firfilt_crcf_firdespm                    | ✅ | ✅ |
| firfilt_crcf_rect                        | ✅ | ✅ |
| firfilt_crcf_notch                       | ✅ | ✅ |
| firfilt_cccf_notch                       | ✅ | ✅ |
| firfilt_config                           | ✅ | ✅ |
| firfilt_recreate                         | ✅ | ✅ |
| firfilt_push_write                       | ✅ | ✅ |


## firfilt_cccf_notch
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firfilt_cccf_notch_0                     | ✅ | ✅ |
| firfilt_cccf_notch_1                     | ✅ | ✅ |
| firfilt_cccf_notch_2                     | ✅ | ✅ |
| firfilt_cccf_notch_3                     | ✅ | ✅ |
| firfilt_cccf_notch_4                     | ✅ | ✅ |
| firfilt_cccf_notch_5                     | ✅ | ✅ |


## firfilt_coefficients
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firfilt_cccf_coefficients_test           | ✅ | ✅ |


## firfilt_rnyquist
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firfilt_rnyquist_baseline_arkaiser       | ✅ | ✅ |
| firfilt_rnyquist_baseline_rkaiser        | ✅ | ✅ |
| firfilt_rnyquist_baseline_rrc            | ✅ | ✅ |
| firfilt_rnyquist_baseline_hm3            | ✅ | ✅ |
| firfilt_rnyquist_baseline_gmsktxrx       | ✅ | ✅ |
| firfilt_rnyquist_baseline_rfexp          | ✅ | ✅ |
| firfilt_rnyquist_baseline_rfsech         | ✅ | ✅ |
| firfilt_rnyquist_baseline_rfarcsech      | ✅ | ✅ |
| firfilt_rnyquist_0                       | ✅ | ✅ |
| firfilt_rnyquist_1                       | ✅ | ✅ |
| firfilt_rnyquist_2                       | ✅ | ✅ |
| firfilt_rnyquist_3                       | ✅ | ✅ |
| firfilt_rnyquist_4                       | ✅ | ✅ |
| firfilt_rnyquist_5                       | ✅ | ✅ |
| firfilt_rnyquist_6                       | ✅ | ✅ |
| firfilt_rnyquist_7                       | ✅ | ✅ |
| firfilt_rnyquist_8                       | ✅ | ✅ |
| firfilt_rnyquist_9                       | ✅ | ✅ |


## firfilt_xxxf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firfilt_rrrf_data_h4x8                   | ✅ | ✅ |
| firfilt_rrrf_data_h7x16                  | ✅ | ✅ |
| firfilt_rrrf_data_h13x32                 | ✅ | ✅ |
| firfilt_rrrf_data_h23x64                 | ✅ | ✅ |
| firfilt_crcf_data_h4x8                   | ✅ | ✅ |
| firfilt_crcf_data_h7x16                  | ✅ | ✅ |
| firfilt_crcf_data_h13x32                 | ✅ | ✅ |
| firfilt_crcf_data_h23x64                 | ✅ | ✅ |
| firfilt_cccf_data_h4x8                   | ✅ | ✅ |
| firfilt_cccf_data_h7x16                  | ✅ | ✅ |
| firfilt_cccf_data_h13x32                 | ✅ | ✅ |
| firfilt_cccf_data_h23x64                 | ✅ | ✅ |


## firfilt_copy
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firfilt_crcf_copy                        | ✅ | ✅ |


## firhilb
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firhilbf_decim                           | ✅ | ✅ |
| firhilbf_interp                          | ✅ | ✅ |
| firhilbf_psd                             | ✅ | ✅ |
| firhilbf_invalid_config                  | ✅ | ✅ |
| firhilbf_copy_interp                     | ✅ | ✅ |
| firhilbf_copy_decim                      | ✅ | ✅ |


## firinterp
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firinterp_rrrf_common                    | ✅ | ✅ |
| firinterp_crcf_common                    | ✅ | ✅ |
| firinterp_rrrf_generic                   | ✅ | ✅ |
| firinterp_crcf_generic                   | ✅ | ✅ |
| firinterp_crcf_rnyquist_0                | ✅ | ✅ |
| firinterp_crcf_rnyquist_1                | ✅ | ✅ |
| firinterp_crcf_rnyquist_2                | ✅ | ✅ |
| firinterp_crcf_rnyquist_3                | ✅ | ✅ |
| firinterp_copy                           | ✅ | ✅ |
| firinterp_flush                          | ✅ | ✅ |


## firpfb
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firpfb_impulse_response                  | ✅ | ✅ |
| firpfb_crcf_copy                         | ✅ | ✅ |


## groupdelay
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fir_groupdelay_n3                        | ✅ | ✅ |
| iir_groupdelay_n3                        | ✅ | ✅ |
| iir_groupdelay_n8                        | ✅ | ✅ |
| iir_groupdelay_sos_n8                    | ✅ | ✅ |


## iirdecim
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirdecim_copy                            | ✅ | ✅ |


## iirdes
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirdes_butter_2                          | ✅ | ✅ |
| iirdes_ellip_lowpass_0                   | ✅ | ✅ |
| iirdes_ellip_lowpass_1                   | ✅ | ✅ |
| iirdes_ellip_lowpass_2                   | ✅ | ✅ |
| iirdes_ellip_lowpass_3                   | ✅ | ✅ |
| iirdes_ellip_lowpass_4                   | ✅ | ✅ |
| iirdes_cheby1_lowpass_0                  | ✅ | ✅ |
| iirdes_cheby1_lowpass_1                  | ✅ | ✅ |
| iirdes_cheby1_lowpass_2                  | ✅ | ✅ |
| iirdes_cheby1_lowpass_3                  | ✅ | ✅ |
| iirdes_cheby1_lowpass_4                  | ✅ | ✅ |
| iirdes_cheby2_lowpass_0                  | ✅ | ✅ |
| iirdes_cheby2_lowpass_1                  | ✅ | ✅ |
| iirdes_cheby2_lowpass_2                  | ✅ | ✅ |
| iirdes_cheby2_lowpass_3                  | ✅ | ✅ |
| iirdes_cheby2_lowpass_4                  | ✅ | ✅ |
| iirdes_butter_lowpass_0                  | ✅ | ✅ |
| iirdes_butter_lowpass_1                  | ✅ | ✅ |
| iirdes_butter_lowpass_2                  | ✅ | ✅ |
| iirdes_butter_lowpass_3                  | ✅ | ✅ |
| iirdes_butter_lowpass_4                  | ✅ | ✅ |
| iirdes_ellip_highpass                    | ✅ | ✅ |
| iirdes_ellip_bandpass                    | ✅ | ✅ |
| iirdes_ellip_bandstop                    | ✅ | ✅ |
| iirdes_bessel                            | ✅ | ✅ |


## iirdes_support
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirdes_cplxpair_n6                       | ✅ | ✅ |
| iirdes_cplxpair_n20                      | ✅ | ✅ |
| iirdes_dzpk2sosf                         | ✅ | ✅ |
| iirdes_isstable_n2_yes                   | ✅ | ✅ |
| iirdes_isstable_n2_no                    | ✅ | ✅ |


## iirfilt
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirfilt_integrator                       | ✅ | ✅ |
| iirfilt_differentiator                   | ✅ | ✅ |
| iirfilt_dcblock                          | ✅ | ✅ |
| iirfilt_copy_tf                          | ✅ | ✅ |
| iirfilt_copy_sos                         | ✅ | ✅ |
| iirfilt_config                           | ✅ | ✅ |


## iirfilt_xxxf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirfilt_rrrf_h3x64                       | ✅ | ✅ |
| iirfilt_rrrf_h5x64                       | ✅ | ✅ |
| iirfilt_rrrf_h7x64                       | ✅ | ✅ |
| iirfilt_crcf_h3x64                       | ✅ | ✅ |
| iirfilt_crcf_h5x64                       | ✅ | ✅ |
| iirfilt_crcf_h7x64                       | ✅ | ✅ |
| iirfilt_cccf_h3x64                       | ✅ | ✅ |
| iirfilt_cccf_h5x64                       | ✅ | ✅ |
| iirfilt_cccf_h7x64                       | ✅ | ✅ |


## iirfiltsos
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirfiltsos_impulse_n2                    | ✅ | ✅ |
| iirfiltsos_step_n2                       | ✅ | ✅ |
| iirfiltsos_copy                          | ✅ | ✅ |
| iirfiltsos_config                        | ✅ | ✅ |


## iirhilb
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirhilbf_interp_decim                    | ✅ | ✅ |
| iirhilbf_filter                          | ✅ | ✅ |
| iirhilbf_invalid_config                  | ✅ | ✅ |
| iirhilbf_copy_interp                     | ✅ | ✅ |
| iirhilbf_copy_decim                      | ✅ | ✅ |


## iirinterp
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| iirinterp_crcf_M2_O9                     | ✅ | ✅ |
| iirinterp_crcf_M3_O9                     | ✅ | ✅ |
| iirinterp_crcf_M4_O9                     | ✅ | ✅ |
| iirinterp_copy                           | ✅ | ✅ |


## lpc
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| lpc_p4                                   | ✅ | ✅ |
| lpc_p6                                   | ✅ | ✅ |
| lpc_p8                                   | ✅ | ✅ |
| lpc_p10                                  | ✅ | ✅ |
| lpc_p16                                  | ✅ | ✅ |
| lpc_p32                                  | ✅ | ✅ |


## msresamp_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| msresamp_crcf_01                         | ✅ | ✅ |
| msresamp_crcf_02                         | ✅ | ✅ |
| msresamp_crcf_03                         | ✅ | ✅ |
| msresamp_crcf_num_output_0               | ✅ | ✅ |
| msresamp_crcf_num_output_1               | ✅ | ✅ |
| msresamp_crcf_num_output_2               | ✅ | ✅ |
| msresamp_crcf_num_output_3               | ✅ | ✅ |
| msresamp_crcf_num_output_4               | ✅ | ✅ |
| msresamp_crcf_num_output_5               | ✅ | ✅ |
| msresamp_crcf_num_output_6               | ✅ | ✅ |
| msresamp_crcf_num_output_7               | ✅ | ✅ |
| msresamp_crcf_copy                       | ✅ | ✅ |


## msresamp2_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| msresamp2_crcf_interp_01                 | ✅ | ✅ |
| msresamp2_crcf_interp_02                 | ✅ | ✅ |
| msresamp2_crcf_interp_03                 | ✅ | ✅ |
| msresamp2_crcf_interp_04                 | ✅ | ✅ |
| msresamp2_crcf_interp_05                 | ✅ | ✅ |
| msresamp2_crcf_interp_06                 | ✅ | ✅ |
| msresamp2_crcf_interp_07                 | ✅ | ✅ |
| msresamp2_crcf_interp_08                 | ✅ | ✅ |
| msresamp2_crcf_interp_09                 | ✅ | ✅ |
| msresamp2_crcf_interp_10                 | ✅ | ✅ |
| msresamp2_crcf_interp_11                 | ✅ | ✅ |
| msresamp2_copy                           | ✅ | ✅ |


## ordfilt
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| ordfilt_copy                             | ✅ | ✅ |


## rresamp_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| rresamp_crcf_baseline_P1_Q5              | ✅ | ✅ |
| rresamp_crcf_baseline_P2_Q5              | ✅ | ✅ |
| rresamp_crcf_baseline_P3_Q5              | ✅ | ✅ |
| rresamp_crcf_baseline_P6_Q5              | ✅ | ✅ |
| rresamp_crcf_baseline_P8_Q5              | ✅ | ✅ |
| rresamp_crcf_baseline_P9_Q5              | ✅ | ✅ |
| rresamp_crcf_default_P1_Q5               | ✅ | ✅ |
| rresamp_crcf_default_P2_Q5               | ✅ | ✅ |
| rresamp_crcf_default_P3_Q5               | ✅ | ✅ |
| rresamp_crcf_default_P6_Q5               | ✅ | ✅ |
| rresamp_crcf_default_P8_Q5               | ✅ | ✅ |
| rresamp_crcf_default_P9_Q5               | ✅ | ✅ |
| rresamp_crcf_arkaiser_P3_Q5              | ✅ | ✅ |
| rresamp_crcf_arkaiser_P5_Q3              | ✅ | ✅ |
| rresamp_crcf_rrcos_P3_Q5                 | ✅ | ✅ |
| rresamp_crcf_rrcos_P5_Q3                 | ✅ | ✅ |
| rresamp_copy                             | ✅ | ❓ |
| rresamp_config                           | ✅ | ❓ |


## rresamp_crcf_partition
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| rresamp_crcf_part_P1_Q5                  | ✅ | ✅ |
| rresamp_crcf_part_P2_Q5                  | ✅ | ✅ |
| rresamp_crcf_part_P3_Q5                  | ✅ | ✅ |
| rresamp_crcf_part_P6_Q5                  | ✅ | ✅ |
| rresamp_crcf_part_P8_Q5                  | ✅ | ✅ |
| rresamp_crcf_part_P9_Q5                  | ✅ | ✅ |


## resamp_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| resamp_crcf_00                           | ✅ | ✅ |
| resamp_crcf_01                           | ✅ | ✅ |
| resamp_crcf_02                           | ✅ | ✅ |
| resamp_crcf_03                           | ✅ | ✅ |
| resamp_crcf_10                           | ✅ | ✅ |
| resamp_crcf_11                           | ✅ | ✅ |
| resamp_crcf_12                           | ✅ | ✅ |
| resamp_crcf_13                           | ✅ | ✅ |
| resamp_crcf_num_output_0                 | ✅ | ✅ |
| resamp_crcf_num_output_1                 | ✅ | ✅ |
| resamp_crcf_num_output_2                 | ✅ | ✅ |
| resamp_crcf_num_output_3                 | ✅ | ✅ |
| resamp_crcf_num_output_4                 | ✅ | ✅ |
| resamp_crcf_num_output_5                 | ✅ | ✅ |
| resamp_crcf_num_output_6                 | ✅ | ✅ |
| resamp_crcf_num_output_7                 | ✅ | ✅ |
| resamp_crcf_copy                         | ✅ | ✅ |


## resamp2_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| resamp2_analysis                         | ✅ | ✅ |
| resamp2_synthesis                        | ✅ | ✅ |
| resamp2_crcf_filter_0                    | ✅ | ✅ |
| resamp2_crcf_filter_1                    | ✅ | ✅ |
| resamp2_crcf_filter_2                    | ✅ | ✅ |
| resamp2_crcf_filter_3                    | ✅ | ✅ |
| resamp2_crcf_filter_4                    | ✅ | ✅ |
| resamp2_crcf_filter_5                    | ✅ | ✅ |
| resamp2_config                           | ✅ | ✅ |
| resamp2_copy                             | ✅ | ✅ |


## rkaiser
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| liquid_rkaiser_config                    | ✅ | ✅ |


## symsync_copy
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symsync_copy                             | ✅ | ✅ |
| symsync_config                           | ✅ | ✅ |


## symsync_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symsync_crcf_scenario_0                  | ✅ | ✅ |
| symsync_crcf_scenario_1                  | ✅ | ✅ |
| symsync_crcf_scenario_2                  | ✅ | ✅ |
| symsync_crcf_scenario_3                  | ✅ | ✅ |
| symsync_crcf_scenario_4                  | ✅ | ✅ |
| symsync_crcf_scenario_5                  | ✅ | ✅ |
| symsync_crcf_scenario_6                  | ✅ | ✅ |
| symsync_crcf_scenario_7                  | ✅ | ✅ |


## symsync_rrrf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symsync_rrrf_scenario_0                  | ✅ | ✅ |
| symsync_rrrf_scenario_1                  | ✅ | ✅ |
| symsync_rrrf_scenario_2                  | ✅ | ✅ |
| symsync_rrrf_scenario_3                  | ✅ | ✅ |
| symsync_rrrf_scenario_4                  | ✅ | ✅ |
| symsync_rrrf_scenario_5                  | ✅ | ✅ |
| symsync_rrrf_scenario_6                  | ✅ | ✅ |
| symsync_rrrf_scenario_7                  | ✅ | ✅ |


## bpacketsync
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| bpacketsync                              | ✅ | ❓ |


## bsync
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| bsync_rrrf_15                            | ✅ | ❓ |
| bsync_crcf_15                            | ✅ | ❓ |


## detector
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| detector_cccf_n64                        | ✅ | ❓ |
| detector_cccf_n83                        | ✅ | ❓ |
| detector_cccf_n128                       | ✅ | ❓ |
| detector_cccf_n167                       | ✅ | ❓ |
| detector_cccf_n256                       | ✅ | ❓ |
| detector_cccf_n335                       | ✅ | ❓ |
| detector_cccf_n512                       | ✅ | ❓ |
| detector_cccf_n671                       | ✅ | ❓ |
| detector_cccf_n1024                      | ✅ | ❓ |
| detector_cccf_n1341                      | ✅ | ❓ |


## dsssframe64
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| dsssframe64sync                          | ✅ | ❓ |
| dsssframe64_config                       | ✅ | ❓ |
| dsssframe64gen_copy                      | ✅ | ❓ |
| dsssframe64sync_copy                     | ✅ | ❓ |


## dsssframesync
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| dsssframesync                            | ✅ | ❓ |


## flexframesync
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| flexframesync                            | ✅ | ❓ |


## framesync64
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| framesync64                              | ✅ | ❓ |
| framegen64_copy                          | ✅ | ❓ |
| framesync64_copy                         | ✅ | ❓ |
| framesync64_config                       | ✅ | ❓ |
| framesync64_debug_none                   | ✅ | ❓ |
| framesync64_debug_user                   | ✅ | ❓ |
| framesync64_debug_ndet                   | ✅ | ❓ |
| framesync64_debug_head                   | ✅ | ❓ |
| framesync64_debug_rand                   | ✅ | ❓ |
| framesync64_estimation                   | ✅ | ❓ |


## fskframesync
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fskframesync                             | ✅ | ❓ |


## gmskframe
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| gmskframesync_process                    | ✅ | ❓ |
| gmskframesync_multiple                   | ✅ | ❓ |
| gmskframesync_k02_m05_bt20               | ✅ | ❓ |
| gmskframesync_k02_m05_bt30               | ✅ | ❓ |
| gmskframesync_k02_m05_bt40               | ✅ | ❓ |
| gmskframesync_k04_m05_bt20               | ✅ | ❓ |
| gmskframesync_k04_m05_bt30               | ✅ | ❓ |
| gmskframesync_k04_m05_bt40               | ✅ | ❓ |
| gmskframesync_k03_m07_bt20               | ✅ | ❓ |
| gmskframesync_k08_m20_bt15               | ✅ | ❓ |
| gmskframesync_k15_m02_bt40               | ✅ | ❓ |


## msource
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| msourcecf_tone                           | ✅ | ❓ |
| msourcecf_chirp                          | ✅ | ❓ |
| msourcecf_aggregate                      | ✅ | ❓ |
| msourcecf_config                         | ✅ | ❓ |
| msourcecf_accessor                       | ✅ | ❓ |
| msourcecf_copy                           | ✅ | ❓ |


## ofdmflexframe
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| ofdmflexframe_00                         | ✅ | ❓ |
| ofdmflexframe_01                         | ✅ | ❓ |
| ofdmflexframe_02                         | ✅ | ❓ |
| ofdmflexframe_03                         | ✅ | ❓ |
| ofdmflexframe_04                         | ✅ | ❓ |
| ofdmflexframe_05                         | ✅ | ❓ |
| ofdmflexframe_06                         | ✅ | ❓ |
| ofdmflexframe_07                         | ✅ | ❓ |
| ofdmflexframe_08                         | ✅ | ❓ |
| ofdmflexframe_09                         | ✅ | ❓ |
| ofdmflexframegen_config                  | ✅ | ❓ |
| ofdmflexframesync_config                 | ✅ | ❓ |


## qdetector_cccf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qdetector_cccf_linear_n64                | ✅ | ❓ |
| qdetector_cccf_linear_n83                | ✅ | ❓ |
| qdetector_cccf_linear_n128               | ✅ | ❓ |
| qdetector_cccf_linear_n167               | ✅ | ❓ |
| qdetector_cccf_linear_n256               | ✅ | ❓ |
| qdetector_cccf_linear_n335               | ✅ | ❓ |
| qdetector_cccf_linear_n512               | ✅ | ❓ |
| qdetector_cccf_linear_n671               | ✅ | ❓ |
| qdetector_cccf_linear_n1024              | ✅ | ❓ |
| qdetector_cccf_linear_n1341              | ✅ | ❓ |
| qdetector_cccf_gmsk_n64                  | ✅ | ❓ |
| qdetector_cccf_gmsk_n83                  | ✅ | ❓ |
| qdetector_cccf_gmsk_n128                 | ✅ | ❓ |
| qdetector_cccf_gmsk_n167                 | ✅ | ❓ |
| qdetector_cccf_gmsk_n256                 | ✅ | ❓ |
| qdetector_cccf_gmsk_n335                 | ✅ | ❓ |
| qdetector_cccf_gmsk_n512                 | ✅ | ❓ |
| qdetector_cccf_gmsk_n671                 | ✅ | ❓ |
| qdetector_cccf_gmsk_n1024                | ✅ | ❓ |
| qdetector_cccf_gmsk_n1341                | ✅ | ❓ |


## qdetector_cccf_copy
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qdetector_cccf_copy                      | ✅ | ❓ |


## qdsync_cccf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qdsync_cccf_k2                           | ✅ | ❓ |
| qdsync_cccf_k3                           | ✅ | ❓ |
| qdsync_cccf_k4                           | ✅ | ❓ |
| qdsync_set_buf_len                       | ✅ | ❓ |
| qdsync_cccf_copy                         | ✅ | ❓ |
| qdsync_cccf_config                       | ✅ | ❓ |


## qpacketmodem
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qpacketmodem_bpsk                        | ✅ | ❓ |
| qpacketmodem_qpsk                        | ✅ | ❓ |
| qpacketmodem_psk8                        | ✅ | ❓ |
| qpacketmodem_qam16                       | ✅ | ❓ |
| qpacketmodem_sqam32                      | ✅ | ❓ |
| qpacketmodem_qam64                       | ✅ | ❓ |
| qpacketmodem_sqam128                     | ✅ | ❓ |
| qpacketmodem_qam256                      | ✅ | ❓ |
| qpacketmodem_evm                         | ✅ | ❓ |
| qpacketmodem_unmod_bpsk                  | ✅ | ❓ |
| qpacketmodem_unmod_qpsk                  | ✅ | ❓ |
| qpacketmodem_unmod_psk8                  | ✅ | ❓ |
| qpacketmodem_unmod_qam16                 | ✅ | ❓ |
| qpacketmodem_unmod_sqam32                | ✅ | ❓ |
| qpacketmodem_unmod_qam64                 | ✅ | ❓ |
| qpacketmodem_unmod_sqam128               | ✅ | ❓ |
| qpacketmodem_unmod_qam256                | ✅ | ❓ |
| qpacketmodem_copy                        | ✅ | ❓ |


## qpilotsync
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qpilotsync_100_16                        | ✅ | ❓ |
| qpilotsync_200_20                        | ✅ | ❓ |
| qpilotsync_300_24                        | ✅ | ❓ |
| qpilotsync_400_28                        | ✅ | ❓ |
| qpilotsync_500_32                        | ✅ | ❓ |
| qpilotgen_config                         | ✅ | ❓ |
| qpilotsync_config                        | ✅ | ❓ |


## qsource
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qsourcecf_config                         | ✅ | ❓ |


## symstreamcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symstreamcf_psd_k2_m12_b030              | ✅ | ✅ |
| symstreamcf_psd_k4_m12_b030              | ✅ | ✅ |
| symstreamcf_psd_k4_m25_b020              | ✅ | ✅ |
| symstreamcf_psd_k7_m11_b035              | ✅ | ✅ |
| symstreamcf_copy                         | ✅ | ✅ |


## symstreamcf_delay
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symstreamcf_delay_00                     | ✅ | ✅ |
| symstreamcf_delay_01                     | ✅ | ✅ |
| symstreamcf_delay_02                     | ✅ | ✅ |
| symstreamcf_delay_03                     | ✅ | ✅ |
| symstreamcf_delay_04                     | ✅ | ✅ |
| symstreamcf_delay_05                     | ✅ | ✅ |
| symstreamcf_delay_06                     | ✅ | ✅ |
| symstreamcf_delay_07                     | ✅ | ✅ |
| symstreamcf_delay_08                     | ✅ | ✅ |
| symstreamcf_delay_09                     | ✅ | ✅ |
| symstreamcf_delay_10                     | ✅ | ✅ |
| symstreamcf_delay_11                     | ✅ | ✅ |
| symstreamcf_delay_12                     | ✅ | ✅ |
| symstreamcf_delay_13                     | ✅ | ✅ |
| symstreamcf_delay_14                     | ✅ | ✅ |
| symstreamcf_delay_15                     | ✅ | ✅ |
| symstreamcf_delay_16                     | ✅ | ✅ |
| symstreamcf_delay_17                     | ✅ | ✅ |
| symstreamcf_delay_18                     | ✅ | ✅ |
| symstreamcf_delay_19                     | ✅ | ✅ |


## symstreamrcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symstreamrcf_psd_bw200_m12_b030          | ✅ | ✅ |
| symstreamrcf_psd_bw400_m12_b030          | ✅ | ✅ |
| symstreamrcf_psd_bw400_m25_b020          | ✅ | ✅ |
| symstreamrcf_psd_bw700_m11_b035          | ✅ | ✅ |
| symstreamrcf_copy                        | ✅ | ✅ |


## symstreamrcf_delay
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symstreamrcf_delay_00                    | ✅ | ✅ |
| symstreamrcf_delay_01                    | ✅ | ✅ |
| symstreamrcf_delay_02                    | ✅ | ✅ |
| symstreamrcf_delay_03                    | ✅ | ✅ |
| symstreamrcf_delay_04                    | ✅ | ✅ |
| symstreamrcf_delay_05                    | ✅ | ✅ |
| symstreamrcf_delay_06                    | ✅ | ✅ |
| symstreamrcf_delay_07                    | ✅ | ✅ |
| symstreamrcf_delay_08                    | ✅ | ✅ |
| symstreamrcf_delay_09                    | ✅ | ✅ |
| symstreamrcf_delay_10                    | ✅ | ✅ |
| symstreamrcf_delay_11                    | ✅ | ✅ |
| symstreamrcf_delay_12                    | ✅ | ✅ |
| symstreamrcf_delay_13                    | ✅ | ✅ |
| symstreamrcf_delay_14                    | ✅ | ✅ |
| symstreamrcf_delay_15                    | ✅ | ✅ |
| symstreamrcf_delay_16                    | ✅ | ✅ |
| symstreamrcf_delay_17                    | ✅ | ✅ |
| symstreamrcf_delay_18                    | ✅ | ✅ |
| symstreamrcf_delay_19                    | ✅ | ✅ |


## symtrack_cccf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| symtrack_cccf_bpsk                       | ✅ | ❓ |
| symtrack_cccf_qpsk                       | ✅ | ❓ |
| symtrack_cccf_config_invalid             | ✅ | ❓ |
| symtrack_cccf_config_valid               | ✅ | ❓ |


## gcd
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| gcd_one                                  | ✅ | ✅ |
| gcd_edge_cases                           | ✅ | ✅ |
| gcd_base                                 | ✅ | ✅ |


## math_window
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| window_hamming                           | ✅ | ✅ |
| window_hann                              | ✅ | ✅ |
| window_blackmanharris                    | ✅ | ✅ |
| window_blackmanharris7                   | ✅ | ✅ |
| window_kaiser                            | ✅ | ✅ |
| window_flattop                           | ✅ | ✅ |
| window_triangular                        | ✅ | ✅ |
| window_rcostaper                         | ✅ | ✅ |
| window_kbd                               | ✅ | ✅ |
| kbd_n16                                  | ✅ | ✅ |
| kbd_n32                                  | ✅ | ✅ |
| kbd_n48                                  | ✅ | ✅ |
| window_config                            | ✅ | ✅ |


## math
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| Q                                        | ✅ | ✅ |
| sincf                                    | ✅ | ✅ |
| nextpow2                                 | ✅ | ✅ |
| math_config                              | ✅ | ✅ |


## math_bessel
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| lnbesselif                               | ✅ | ✅ |
| besselif                                 | ✅ | ✅ |
| besseli0f                                | ✅ | ✅ |
| besseljf                                 | ✅ | ✅ |
| besselj0f                                | ✅ | ✅ |


## math_gamma
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| gamma                                    | ✅ | ✅ |
| lngamma                                  | ✅ | ✅ |
| uppergamma                               | ✅ | ✅ |
| factorial                                | ✅ | ✅ |
| nchoosek                                 | ✅ | ✅ |


## math_complex
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| cexpf                                    | ✅ | ✅ |
| clogf                                    | ✅ | ✅ |
| csqrtf                                   | ✅ | ✅ |
| casinf                                   | ✅ | ✅ |
| cacosf                                   | ✅ | ✅ |
| catanf                                   | ✅ | ✅ |


## polynomial
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| polyf_fit_q3n3                           | ✅ | ✅ |
| polyf_lagrange_issue165                  | ✅ | ✅ |
| polyf_lagrange                           | ✅ | ✅ |
| polyf_expandroots_4                      | ✅ | ✅ |
| polyf_expandroots_11                     | ✅ | ✅ |
| polycf_expandroots_4                     | ✅ | ✅ |
| polyf_expandroots2_3                     | ✅ | ✅ |
| polyf_mul_2_3                            | ✅ | ✅ |
| poly_expandbinomial_n6                   | ✅ | ✅ |
| poly_binomial_expand_pm_m6_k1            | ✅ | ✅ |
| poly_expandbinomial_pm_m5_k2             | ✅ | ✅ |


## polynomial_findroots
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| polyf_findroots_real                     | ✅ | ✅ |
| polyf_findroots_complex                  | ✅ | ✅ |
| polyf_findroots_mix                      | ✅ | ✅ |
| polyf_findroots_mix2                     | ✅ | ✅ |


## prime
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| prime_small                              | ✅ | ✅ |
| factors                                  | ✅ | ✅ |
| totient                                  | ✅ | ✅ |


## matrixcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| matrixcf_add                             | ✅ | ✅ |
| matrixcf_aug                             | ✅ | ✅ |
| matrixcf_chol                            | ✅ | ✅ |
| matrixcf_inv                             | ✅ | ✅ |
| matrixcf_linsolve                        | ✅ | ✅ |
| matrixcf_ludecomp_crout                  | ✅ | ✅ |
| matrixcf_ludecomp_doolittle              | ✅ | ✅ |
| matrixcf_mul                             | ✅ | ✅ |
| matrixcf_qrdecomp                        | ✅ | ✅ |
| matrixcf_transmul                        | ✅ | ✅ |


## matrixf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| matrixf_add                              | ✅ | ✅ |
| matrixf_aug                              | ✅ | ✅ |
| matrixf_cgsolve                          | ✅ | ✅ |
| matrixf_chol                             | ✅ | ✅ |
| matrixf_gramschmidt                      | ✅ | ✅ |
| matrixf_inv                              | ✅ | ✅ |
| matrixf_linsolve                         | ✅ | ✅ |
| matrixf_ludecomp_crout                   | ✅ | ✅ |
| matrixf_ludecomp_doolittle               | ✅ | ✅ |
| matrixf_mul                              | ✅ | ✅ |
| matrixf_qrdecomp                         | ✅ | ✅ |
| matrixf_transmul                         | ✅ | ✅ |


## smatrixb
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| smatrixb_vmul                            | ✅ | ✅ |
| smatrixb_mul                             | ✅ | ✅ |
| smatrixb_mulf                            | ✅ | ✅ |
| smatrixb_vmulf                           | ✅ | ✅ |


## smatrixf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| smatrixf_vmul                            | ✅ | ✅ |
| smatrixf_mul                             | ✅ | ✅ |


## smatrixi
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| smatrixi_vmul                            | ✅ | ✅ |
| smatrixi_mul                             | ✅ | ✅ |


## ampmodem
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| ampmodem_dsb_carrier_on                  | ✅ | ✅ |
| ampmodem_usb_carrier_on                  | ✅ | ✅ |
| ampmodem_lsb_carrier_on                  | ✅ | ✅ |
| ampmodem_dsb_carrier_off                 | ✅ | ✅ |
| ampmodem_usb_carrier_off                 | ✅ | ✅ |
| ampmodem_lsb_carrier_off                 | ✅ | ✅ |


## cpfskmodem
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| cpfskmodem_bps1_h0p5000_k4_m3_square     | ✅ | ❓ |
| cpfskmodem_bps1_h0p0250_k4_m3_square     | ✅ | ❓ |
| cpfskmodem_bps1_h0p1250_k4_m3_square     | ✅ | ❓ |
| cpfskmodem_bps1_h0p0625_k4_m3_square     | ✅ | ❓ |
| cpfskmodem_bps1_h0p5000_k4_m3_rcosfull   | ✅ | ❓ |
| cpfskmodem_bps1_h0p0250_k4_m3_rcosfull   | ✅ | ❓ |
| cpfskmodem_bps1_h0p1250_k4_m3_rcosfull   | ✅ | ❓ |
| cpfskmodem_bps1_h0p0625_k4_m3_rcosfull   | ✅ | ❓ |
| cpfskmodem_bps1_h0p5000_k4_m3_rcospart   | ✅ | ❓ |
| cpfskmodem_bps1_h0p0250_k4_m3_rcospart   | ✅ | ❓ |
| cpfskmodem_bps1_h0p1250_k4_m3_rcospart   | ✅ | ❓ |
| cpfskmodem_bps1_h0p0625_k4_m3_rcospart   | ✅ | ❓ |
| cpfskmodem_bps1_h0p5000_k4_m3_gmsk       | ✅ | ❓ |
| cpfskmodem_bps1_h0p0250_k4_m3_gmsk       | ✅ | ❓ |
| cpfskmodem_bps1_h0p1250_k4_m3_gmsk       | ✅ | ❓ |
| cpfskmodem_bps1_h0p0625_k4_m3_gmsk       | ✅ | ❓ |
| cpfskmodem_bps2_h0p0250_k4_m3_square     | ✅ | ❓ |
| cpfskmodem_bps3_h0p1250_k4_m3_square     | ✅ | ❓ |
| cpfskmodem_bps4_h0p0625_k4_m3_square     | ✅ | ❓ |
| cpfskmodem_bps1_h0p5_k2_m7_gmsk          | ✅ | ❓ |
| cpfskmodem_bps1_h0p5_k4_m7_gmsk          | ✅ | ❓ |
| cpfskmodem_bps1_h0p5_k6_m7_gmsk          | ✅ | ❓ |
| cpfskmodem_bps1_h0p5_k8_m7_gmsk          | ✅ | ❓ |
| cpfskmodem_spectrum                      | ✅ | ❓ |
| cpfskmodem_config                        | ✅ | ❓ |


## freqmodem
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| freqmodem_kf_0_02                        | ✅ | ✅ |
| freqmodem_kf_0_04                        | ✅ | ✅ |
| freqmodem_kf_0_08                        | ✅ | ✅ |


## fskmodem
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| fskmodem_norm_M2                         | ✅ | ✅ |
| fskmodem_norm_M4                         | ✅ | ✅ |
| fskmodem_norm_M8                         | ✅ | ✅ |
| fskmodem_norm_M16                        | ✅ | ✅ |
| fskmodem_norm_M32                        | ✅ | ✅ |
| fskmodem_norm_M64                        | ✅ | ✅ |
| fskmodem_norm_M128                       | ✅ | ✅ |
| fskmodem_norm_M256                       | ✅ | ✅ |
| fskmodem_norm_M512                       | ✅ | ✅ |
| fskmodem_norm_M1024                      | ✅ | ✅ |
| fskmodem_misc_M2                         | ✅ | ✅ |
| fskmodem_misc_M4                         | ✅ | ✅ |
| fskmodem_misc_M8                         | ✅ | ✅ |
| fskmodem_misc_M16                        | ✅ | ✅ |
| fskmodem_misc_M32                        | ✅ | ✅ |
| fskmodem_misc_M64                        | ✅ | ✅ |
| fskmodem_misc_M128                       | ✅ | ✅ |
| fskmodem_misc_M256                       | ✅ | ✅ |
| fskmodem_misc_M512                       | ✅ | ✅ |
| fskmodem_misc_M1024                      | ✅ | ✅ |
| fskmod_copy                              | ✅ | ✅ |
| fskdem_copy                              | ✅ | ✅ |


## gmskmodem
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| gmskmodem_k4_m3_b025                     | ✅ | ❓ |
| gmskmodem_k2_m3_b025                     | ✅ | ❓ |
| gmskmodem_k3_m3_b025                     | ✅ | ❓ |
| gmskmodem_k5_m3_b025                     | ✅ | ❓ |
| gmskmodem_k8_m3_b033                     | ✅ | ❓ |
| gmskmodem_k4_m1_b025                     | ✅ | ❓ |
| gmskmodem_k4_m2_b025                     | ✅ | ❓ |
| gmskmodem_k4_m8_b025                     | ✅ | ❓ |
| gmskmodem_k4_m3_b020                     | ✅ | ❓ |
| gmskmodem_k4_m3_b033                     | ✅ | ❓ |
| gmskmodem_k4_m3_b050                     | ✅ | ❓ |
| gmskmod_copy                             | ✅ | ❓ |
| gmskdem_copy                             | ✅ | ❓ |


## modem
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| mod_demod_psk2                           | ✅ | ✅ |
| mod_demod_psk4                           | ✅ | ✅ |
| mod_demod_psk8                           | ✅ | ✅ |
| mod_demod_psk16                          | ✅ | ✅ |
| mod_demod_psk32                          | ✅ | ✅ |
| mod_demod_psk64                          | ✅ | ✅ |
| mod_demod_psk128                         | ✅ | ✅ |
| mod_demod_psk256                         | ✅ | ✅ |
| mod_demod_dpsk2                          | ✅ | ✅ |
| mod_demod_dpsk4                          | ✅ | ✅ |
| mod_demod_dpsk8                          | ✅ | ✅ |
| mod_demod_dpsk16                         | ✅ | ✅ |
| mod_demod_dpsk32                         | ✅ | ✅ |
| mod_demod_dpsk64                         | ✅ | ✅ |
| mod_demod_dpsk128                        | ✅ | ✅ |
| mod_demod_dpsk256                        | ✅ | ✅ |
| mod_demod_ask2                           | ✅ | ✅ |
| mod_demod_ask4                           | ✅ | ✅ |
| mod_demod_ask8                           | ✅ | ✅ |
| mod_demod_ask16                          | ✅ | ✅ |
| mod_demod_ask32                          | ✅ | ✅ |
| mod_demod_ask64                          | ✅ | ✅ |
| mod_demod_ask128                         | ✅ | ✅ |
| mod_demod_ask256                         | ✅ | ✅ |
| mod_demod_qam4                           | ✅ | ✅ |
| mod_demod_qam8                           | ✅ | ✅ |
| mod_demod_qam16                          | ✅ | ✅ |
| mod_demod_qam32                          | ✅ | ✅ |
| mod_demod_qam64                          | ✅ | ✅ |
| mod_demod_qam128                         | ✅ | ✅ |
| mod_demod_qam256                         | ✅ | ✅ |
| mod_demod_apsk4                          | ✅ | ✅ |
| mod_demod_apsk8                          | ✅ | ✅ |
| mod_demod_apsk16                         | ✅ | ✅ |
| mod_demod_apsk32                         | ✅ | ✅ |
| mod_demod_apsk64                         | ✅ | ✅ |
| mod_demod_apsk128                        | ✅ | ✅ |
| mod_demod_apsk256                        | ✅ | ✅ |
| mod_demod_bpsk                           | ✅ | ✅ |
| mod_demod_qpsk                           | ✅ | ✅ |
| mod_demod_ook                            | ✅ | ✅ |
| mod_demod_sqam32                         | ✅ | ✅ |
| mod_demod_sqam128                        | ✅ | ✅ |
| mod_demod_V29                            | ✅ | ✅ |
| mod_demod_arb16opt                       | ✅ | ✅ |
| mod_demod_arb32opt                       | ✅ | ✅ |
| mod_demod_arb64opt                       | ✅ | ✅ |
| mod_demod_arb128opt                      | ✅ | ✅ |
| mod_demod_arb256opt                      | ✅ | ✅ |
| mod_demod_arb64vt                        | ✅ | ✅ |
| mod_demod_pi4dqpsk                       | ✅ | ✅ |


## modem_config
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| modem_copy_psk2                          | ✅ | ✅ |
| modem_copy_psk4                          | ✅ | ✅ |
| modem_copy_psk8                          | ✅ | ✅ |
| modem_copy_psk16                         | ✅ | ✅ |
| modem_copy_psk32                         | ✅ | ✅ |
| modem_copy_psk64                         | ✅ | ✅ |
| modem_copy_psk128                        | ✅ | ✅ |
| modem_copy_psk256                        | ✅ | ✅ |
| modem_copy_dpsk2                         | ✅ | ✅ |
| modem_copy_dpsk4                         | ✅ | ✅ |
| modem_copy_dpsk8                         | ✅ | ✅ |
| modem_copy_dpsk16                        | ✅ | ✅ |
| modem_copy_dpsk32                        | ✅ | ✅ |
| modem_copy_dpsk64                        | ✅ | ✅ |
| modem_copy_dpsk128                       | ✅ | ✅ |
| modem_copy_dpsk256                       | ✅ | ✅ |
| modem_copy_ask2                          | ✅ | ✅ |
| modem_copy_ask4                          | ✅ | ✅ |
| modem_copy_ask8                          | ✅ | ✅ |
| modem_copy_ask16                         | ✅ | ✅ |
| modem_copy_ask32                         | ✅ | ✅ |
| modem_copy_ask64                         | ✅ | ✅ |
| modem_copy_ask128                        | ✅ | ✅ |
| modem_copy_ask256                        | ✅ | ✅ |
| modem_copy_qam4                          | ✅ | ✅ |
| modem_copy_qam8                          | ✅ | ✅ |
| modem_copy_qam16                         | ✅ | ✅ |
| modem_copy_qam32                         | ✅ | ✅ |
| modem_copy_qam64                         | ✅ | ✅ |
| modem_copy_qam128                        | ✅ | ✅ |
| modem_copy_qam256                        | ✅ | ✅ |
| modem_copy_apsk4                         | ✅ | ✅ |
| modem_copy_apsk8                         | ✅ | ✅ |
| modem_copy_apsk16                        | ✅ | ✅ |
| modem_copy_apsk32                        | ✅ | ✅ |
| modem_copy_apsk64                        | ✅ | ✅ |
| modem_copy_apsk128                       | ✅ | ✅ |
| modem_copy_apsk256                       | ✅ | ✅ |
| modem_copy_bpsk                          | ✅ | ✅ |
| modem_copy_qpsk                          | ✅ | ✅ |
| modem_copy_ook                           | ✅ | ✅ |
| modem_copy_sqam32                        | ✅ | ✅ |
| modem_copy_sqam128                       | ✅ | ✅ |
| modem_copy_V29                           | ✅ | ✅ |
| modem_copy_arb16opt                      | ✅ | ✅ |
| modem_copy_arb32opt                      | ✅ | ✅ |
| modem_copy_arb64opt                      | ✅ | ✅ |
| modem_copy_arb128opt                     | ✅ | ✅ |
| modem_copy_arb256opt                     | ✅ | ✅ |
| modem_copy_arb64vt                       | ✅ | ✅ |
| modem_copy_pi4dqpsk                      | ✅ | ✅ |
| modem_config                             | ✅ | ❓ |


## modem_demodsoft
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| demodsoft_psk2                           | ✅ | ✅ |
| demodsoft_psk4                           | ✅ | ✅ |
| demodsoft_psk8                           | ✅ | ✅ |
| demodsoft_psk16                          | ✅ | ✅ |
| demodsoft_psk32                          | ✅ | ✅ |
| demodsoft_psk64                          | ✅ | ✅ |
| demodsoft_psk128                         | ✅ | ✅ |
| demodsoft_psk256                         | ✅ | ✅ |
| demodsoft_dpsk2                          | ✅ | ✅ |
| demodsoft_dpsk4                          | ✅ | ✅ |
| demodsoft_dpsk8                          | ✅ | ✅ |
| demodsoft_dpsk16                         | ✅ | ✅ |
| demodsoft_dpsk32                         | ✅ | ✅ |
| demodsoft_dpsk64                         | ✅ | ✅ |
| demodsoft_dpsk128                        | ✅ | ✅ |
| demodsoft_dpsk256                        | ✅ | ✅ |
| demodsoft_ask2                           | ✅ | ✅ |
| demodsoft_ask4                           | ✅ | ✅ |
| demodsoft_ask8                           | ✅ | ✅ |
| demodsoft_ask16                          | ✅ | ✅ |
| demodsoft_ask32                          | ✅ | ✅ |
| demodsoft_ask64                          | ✅ | ✅ |
| demodsoft_ask128                         | ✅ | ✅ |
| demodsoft_ask256                         | ✅ | ✅ |
| demodsoft_qam4                           | ✅ | ✅ |
| demodsoft_qam8                           | ✅ | ✅ |
| demodsoft_qam16                          | ✅ | ✅ |
| demodsoft_qam32                          | ✅ | ✅ |
| demodsoft_qam64                          | ✅ | ✅ |
| demodsoft_qam128                         | ✅ | ✅ |
| demodsoft_qam256                         | ✅ | ✅ |
| demodsoft_apsk4                          | ✅ | ✅ |
| demodsoft_apsk8                          | ✅ | ✅ |
| demodsoft_apsk16                         | ✅ | ✅ |
| demodsoft_apsk32                         | ✅ | ✅ |
| demodsoft_apsk64                         | ✅ | ✅ |
| demodsoft_apsk128                        | ✅ | ✅ |
| demodsoft_apsk256                        | ✅ | ✅ |
| demodsoft_bpsk                           | ✅ | ✅ |
| demodsoft_qpsk                           | ✅ | ✅ |
| demodsoft_ook                            | ✅ | ✅ |
| demodsoft_sqam32                         | ✅ | ✅ |
| demodsoft_sqam128                        | ✅ | ✅ |
| demodsoft_V29                            | ✅ | ✅ |
| demodsoft_arb16opt                       | ✅ | ✅ |
| demodsoft_arb32opt                       | ✅ | ✅ |
| demodsoft_arb64opt                       | ✅ | ✅ |
| demodsoft_arb128opt                      | ✅ | ✅ |
| demodsoft_arb256opt                      | ✅ | ✅ |
| demodsoft_arb64vt                        | ✅ | ✅ |
| demodsoft_pi4dqpsk                       | ✅ | ✅ |


## modem_demodstats
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| demodstats_psk2                          | ✅ | ✅ |
| demodstats_psk4                          | ✅ | ✅ |
| demodstats_psk8                          | ✅ | ✅ |
| demodstats_psk16                         | ✅ | ✅ |
| demodstats_psk32                         | ✅ | ✅ |
| demodstats_psk64                         | ✅ | ✅ |
| demodstats_psk128                        | ✅ | ✅ |
| demodstats_psk256                        | ✅ | ✅ |
| demodstats_dpsk2                         | ✅ | ✅ |
| demodstats_dpsk4                         | ✅ | ✅ |
| demodstats_dpsk8                         | ✅ | ✅ |
| demodstats_dpsk16                        | ✅ | ✅ |
| demodstats_dpsk32                        | ✅ | ✅ |
| demodstats_dpsk64                        | ✅ | ✅ |
| demodstats_dpsk128                       | ✅ | ✅ |
| demodstats_dpsk256                       | ✅ | ✅ |
| demodstats_ask2                          | ✅ | ✅ |
| demodstats_ask4                          | ✅ | ✅ |
| demodstats_ask8                          | ✅ | ✅ |
| demodstats_ask16                         | ✅ | ✅ |
| demodstats_ask32                         | ✅ | ✅ |
| demodstats_ask64                         | ✅ | ✅ |
| demodstats_ask128                        | ✅ | ✅ |
| demodstats_ask256                        | ✅ | ✅ |
| demodstats_qam4                          | ✅ | ✅ |
| demodstats_qam8                          | ✅ | ✅ |
| demodstats_qam16                         | ✅ | ✅ |
| demodstats_qam32                         | ✅ | ✅ |
| demodstats_qam64                         | ✅ | ✅ |
| demodstats_qam128                        | ✅ | ✅ |
| demodstats_qam256                        | ✅ | ✅ |
| demodstats_apsk4                         | ✅ | ✅ |
| demodstats_apsk8                         | ✅ | ✅ |
| demodstats_apsk16                        | ✅ | ✅ |
| demodstats_apsk32                        | ✅ | ✅ |
| demodstats_apsk64                        | ✅ | ✅ |
| demodstats_apsk128                       | ✅ | ✅ |
| demodstats_apsk256                       | ✅ | ✅ |
| demodstats_bpsk                          | ✅ | ✅ |
| demodstats_qpsk                          | ✅ | ✅ |
| demodstats_ook                           | ✅ | ✅ |
| demodstats_sqam32                        | ✅ | ✅ |
| demodstats_sqam128                       | ✅ | ✅ |
| demodstats_V29                           | ✅ | ✅ |
| demodstats_arb16opt                      | ✅ | ✅ |
| demodstats_arb32opt                      | ✅ | ✅ |
| demodstats_arb64opt                      | ✅ | ✅ |
| demodstats_arb128opt                     | ✅ | ✅ |
| demodstats_arb256opt                     | ✅ | ✅ |
| demodstats_arb64vt                       | ✅ | ✅ |


## modem_utilities
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| modemcf_print_schemes                    | ✅ | ❓ |
| modemcf_str2mod                          | ✅ | ❓ |
| modemcf_types                            | ✅ | ❓ |


## firpfbch_crcf_synthesizer
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firpfbch_crcf_synthesis                  | ✅ | ❓ |


## firpfbch_crcf_analyzer
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firpfbch_crcf_analysis                   | ✅ | ❓ |


## firpfbch_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firpfbch_crcf_config                     | ✅ | ❓ |


## firpfbch2_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firpfbch2_crcf_n8                        | ✅ | ❓ |
| firpfbch2_crcf_n16                       | ✅ | ❓ |
| firpfbch2_crcf_n32                       | ✅ | ❓ |
| firpfbch2_crcf_n64                       | ✅ | ❓ |
| firpfbch2_crcf_copy                      | ✅ | ❓ |
| firpfbch2_crcf_config                    | ✅ | ❓ |


## firpfbchr_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| firpfbchr_crcf                           | ✅ | ❓ |
| firpfbchr_crcf_config                    | ✅ | ❓ |


## ofdmframe
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| ofdmframesync_acquire_n64                | ✅ | ❓ |
| ofdmframesync_acquire_n128               | ✅ | ❓ |
| ofdmframesync_acquire_n256               | ✅ | ❓ |
| ofdmframesync_acquire_n512               | ✅ | ❓ |
| ofdmframe_common_config                  | ✅ | ❓ |
| ofdmframegen_config                      | ✅ | ❓ |
| ofdmframesync_config                     | ✅ | ❓ |


## nco_crcf
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| nco_crcf_constrain                       | ✅ | ❓ |
| nco_crcf_copy                            | ✅ | ❓ |
| nco_config                               | ✅ | ❓ |


## nco_crcf_frequency
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| nco_crcf_frequency                       | ✅ | ✅ |


## nco_crcf_mix
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| nco_crcf_mix_nco_0                       | ✅ | ✅ |
| nco_crcf_mix_nco_1                       | ✅ | ✅ |
| nco_crcf_mix_nco_2                       | ✅ | ✅ |
| nco_crcf_mix_nco_3                       | ✅ | ✅ |
| nco_crcf_mix_nco_4                       | ✅ | ✅ |
| nco_crcf_mix_nco_5                       | ✅ | ✅ |
| nco_crcf_mix_nco_6                       | ✅ | ✅ |
| nco_crcf_mix_nco_7                       | ✅ | ✅ |
| nco_crcf_mix_nco_8                       | ✅ | ✅ |
| nco_crcf_mix_nco_9                       | ✅ | ✅ |
| nco_crcf_mix_vco_0                       | ✅ | ✅ |
| nco_crcf_mix_vco_1                       | ✅ | ✅ |
| nco_crcf_mix_vco_2                       | ✅ | ✅ |
| nco_crcf_mix_vco_3                       | ✅ | ✅ |
| nco_crcf_mix_vco_4                       | ✅ | ✅ |
| nco_crcf_mix_vco_5                       | ✅ | ✅ |
| nco_crcf_mix_vco_6                       | ✅ | ✅ |
| nco_crcf_mix_vco_7                       | ✅ | ✅ |
| nco_crcf_mix_vco_8                       | ✅ | ✅ |
| nco_crcf_mix_vco_9                       | ✅ | ✅ |


## nco_crcf_phase
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| nco_crcf_phase                           | ✅ | ✅ |
| nco_basic                                | ✅ | ✅ |
| nco_mixing                               | ✅ | ✅ |
| nco_block_mixing                         | ✅ | ✅ |


## nco_crcf_pll
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| nco_crcf_pll_phase                       | ✅ | ✅ |
| nco_crcf_pll_freq                        | ✅ | ✅ |


## nco_crcf_spectrum
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| nco_crcf_spectrum_nco_f00                | ✅ | ✅ |
| nco_crcf_spectrum_nco_f01                | ✅ | ✅ |
| nco_crcf_spectrum_nco_f02                | ✅ | ✅ |
| nco_crcf_spectrum_nco_f03                | ✅ | ✅ |
| nco_crcf_spectrum_nco_f04                | ✅ | ✅ |
| nco_crcf_spectrum_vco_f00                | ✅ | ✅ |
| nco_crcf_spectrum_vco_f01                | ✅ | ✅ |
| nco_crcf_spectrum_vco_f02                | ✅ | ✅ |
| nco_crcf_spectrum_vco_f03                | ✅ | ✅ |
| nco_crcf_spectrum_vco_f04                | ✅ | ✅ |


## unwrap_phase
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| nco_unwrap_phase                         | ✅ | ❓ |


## gasearch
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| gasearch_peak                            | ✅ | ❓ |
| chromosome_config                        | ✅ | ❓ |
| gasearch_config                          | ✅ | ❓ |


## gradsearch
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| gradsearch_rosenbrock                    | ✅ | ❓ |
| gradsearch_maxutility                    | ✅ | ❓ |


## qnsearch
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qnsearch_rosenbrock                      | ✅ | ❓ |
| qnsearch_config                          | ✅ | ❓ |


## qs1dsearch
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| qs1dsearch_min_01                        | ✅ | ✅ |
| qs1dsearch_min_02                        | ✅ | ✅ |
| qs1dsearch_min_03                        | ✅ | ✅ |
| qs1dsearch_min_05                        | ✅ | ✅ |
| qs1dsearch_min_06                        | ✅ | ✅ |
| qs1dsearch_min_07                        | ✅ | ✅ |
| qs1dsearch_min_08                        | ✅ | ✅ |
| qs1dsearch_min_10                        | ✅ | ✅ |
| qs1dsearch_min_11                        | ✅ | ✅ |
| qs1dsearch_min_12                        | ✅ | ✅ |
| qs1dsearch_min_13                        | ✅ | ✅ |
| qs1dsearch_max_01                        | ✅ | ✅ |
| qs1dsearch_max_02                        | ✅ | ✅ |
| qs1dsearch_max_03                        | ✅ | ✅ |
| qs1dsearch_max_05                        | ✅ | ✅ |
| qs1dsearch_max_06                        | ✅ | ✅ |
| qs1dsearch_max_07                        | ✅ | ✅ |
| qs1dsearch_max_08                        | ✅ | ✅ |
| qs1dsearch_max_10                        | ✅ | ✅ |
| qs1dsearch_max_11                        | ✅ | ✅ |
| qs1dsearch_max_12                        | ✅ | ✅ |
| qs1dsearch_max_13                        | ✅ | ✅ |
| qs1dsearch_config                        | ✅ | ✅ |


## utility
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| optim_rosenbrock                         | ✅ | ❓ |


## compand
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| compand_float                            | ✅ | ❓ |
| compand_cfloat                           | ✅ | ❓ |


## quantize
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| quantize_float_n8                        | ✅ | ❓ |


## scramble
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| scramble_n16                             | ✅ | ✅ |
| scramble_n64                             | ✅ | ✅ |
| scramble_n256                            | ✅ | ✅ |
| scramble_n11                             | ✅ | ✅ |
| scramble_n33                             | ✅ | ✅ |
| scramble_n277                            | ✅ | ✅ |
| scramble_soft_n16                        | ✅ | ✅ |
| scramble_soft_n64                        | ✅ | ✅ |
| scramble_soft_n256                       | ✅ | ✅ |
| scramble_soft_n11                        | ✅ | ✅ |
| scramble_soft_n33                        | ✅ | ✅ |
| scramble_soft_n277                       | ✅ | ✅ |


## random
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| randf                                    | ✅ | ✅ |
| randnf                                   | ✅ | ✅ |
| crandnf                                  | ✅ | ✅ |
| randweibf                                | ✅ | ✅ |
| randricekf                               | ✅ | ✅ |
| randexpf                                 | ✅ | ✅ |
| random_config                            | ✅ | ✅ |


## random_distributions
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| distribution_randnf                      | ✅ | ✅ |


## bsequence
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| bsequence_init                           | ✅ | ✅ |
| bsequence_correlate                      | ✅ | ✅ |
| bsequence_add                            | ✅ | ✅ |
| bsequence_mul                            | ✅ | ✅ |
| bsequence_accumulate                     | ✅ | ✅ |


## complementary_codes
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| complementary_code_n8                    | ✅ | ✅ |
| complementary_code_n16                   | ✅ | ✅ |
| complementary_code_n32                   | ✅ | ✅ |
| complementary_code_n64                   | ✅ | ✅ |
| complementary_code_n128                  | ✅ | ✅ |
| complementary_code_n256                  | ✅ | ✅ |
| complementary_code_n512                  | ✅ | ✅ |


## msequence
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| bsequence_init_msequence                 | ✅ | ✅ |
| msequence_xcorr_m2                       | ✅ | ✅ |
| msequence_xcorr_m3                       | ✅ | ✅ |
| msequence_xcorr_m4                       | ✅ | ✅ |
| msequence_xcorr_m5                       | ✅ | ✅ |
| msequence_xcorr_m6                       | ✅ | ✅ |
| msequence_xcorr_m7                       | ✅ | ✅ |
| msequence_xcorr_m8                       | ✅ | ✅ |
| msequence_xcorr_m9                       | ✅ | ✅ |
| msequence_xcorr_m10                      | ✅ | ✅ |
| msequence_xcorr_m11                      | ✅ | ✅ |
| msequence_xcorr_m12                      | ✅ | ✅ |
| msequence_period_m2                      | ✅ | ✅ |
| msequence_period_m3                      | ✅ | ✅ |
| msequence_period_m4                      | ✅ | ✅ |
| msequence_period_m5                      | ✅ | ✅ |
| msequence_period_m6                      | ✅ | ✅ |
| msequence_period_m7                      | ✅ | ✅ |
| msequence_period_m8                      | ✅ | ✅ |
| msequence_period_m9                      | ✅ | ✅ |
| msequence_period_m10                     | ✅ | ✅ |
| msequence_period_m11                     | ✅ | ✅ |
| msequence_period_m12                     | ✅ | ✅ |
| msequence_period_m13                     | ✅ | ✅ |
| msequence_period_m14                     | ✅ | ✅ |
| msequence_period_m15                     | ✅ | ✅ |
| msequence_period_m16                     | ✅ | ✅ |
| msequence_period_m17                     | ✅ | ✅ |
| msequence_period_m18                     | ✅ | ✅ |
| msequence_period_m19                     | ✅ | ✅ |
| msequence_period_m20                     | ✅ | ✅ |
| msequence_period_m21                     | ✅ | ✅ |
| msequence_period_m22                     | ✅ | ✅ |
| msequence_period_m23                     | ✅ | ✅ |
| msequence_period_m24                     | ✅ | ✅ |
| msequence_period_m25                     | ✅ | ✅ |
| msequence_period_m26                     | ✅ | ✅ |
| msequence_period_m27                     | ✅ | ✅ |
| msequence_period_m28                     | ✅ | ✅ |
| msequence_period_m29                     | ✅ | ✅ |
| msequence_period_m30                     | ✅ | ✅ |
| msequence_period_m31                     | ✅ | ✅ |
| msequence_config                         | ✅ | ✅ |


## bshift_array
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| lbshift                                  | ✅ | ❓ |
| rbshift                                  | ✅ | ❓ |
| lbcircshift                              | ✅ | ❓ |
| rbcircshift                              | ✅ | ❓ |


## count_bits
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| count_ones                               | ✅ | ✅ |
| count_ones_mod2                          | ✅ | ✅ |
| bdotprod                                 | ✅ | ✅ |
| count_leading_zeros                      | ✅ | ✅ |
| msb_index                                | ✅ | ✅ |


## pack_bytes
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| pack_array                               | ✅ | ❓ |
| unpack_array                             | ✅ | ❓ |
| repack_array                             | ✅ | ❓ |
| pack_bytes_01                            | ✅ | ❓ |
| unpack_bytes_01                          | ✅ | ❓ |
| repack_bytes_01                          | ✅ | ❓ |
| repack_bytes_02                          | ✅ | ❓ |
| repack_bytes_03                          | ✅ | ❓ |
| repack_bytes_04_uneven                   | ✅ | ❓ |


## shift_array
| Test | Liquid | Yagi |
| ---- | ------ | ---- |
| lshift                                   | ✅ | ❓ |
| rshift                                   | ✅ | ❓ |
| lcircshift                               | ✅ | ❓ |
| rcircshift                               | ✅ | ❓ |
