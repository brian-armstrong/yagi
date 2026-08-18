use crate::modem::modem::*;


// UIllinois logo
pub(super) const MODEM_ARB_UI64: [Complex32; 64] = [
    Complex32::new( 9.9563e-01,  1.4970e+00), Complex32::new( 7.9767e-01,  1.4970e+00),
    Complex32::new( 5.9971e-01,  1.4970e+00), Complex32::new( 3.9884e-01,  1.4970e+00),
    Complex32::new( 2.0087e-01,  1.4970e+00), Complex32::new( 0.0000e+00,  1.4970e+00),
    Complex32::new(-1.9796e-01,  1.4970e+00), Complex32::new(-3.9592e-01,  1.4970e+00),
    Complex32::new(-5.9680e-01,  1.4970e+00), Complex32::new(-7.9476e-01,  1.4970e+00),
    Complex32::new(-9.9563e-01,  1.4970e+00), Complex32::new(-9.9563e-01,  1.3114e+00),
    Complex32::new(-9.9563e-01,  1.1228e+00), Complex32::new(-9.9563e-01,  9.3413e-01),
    Complex32::new(-9.9563e-01,  7.5150e-01), Complex32::new(-8.1223e-01,  7.5150e-01),
    Complex32::new(-6.2882e-01,  7.5150e-01), Complex32::new(-4.4541e-01,  7.5150e-01),
    Complex32::new(-4.4541e-01,  5.6587e-01), Complex32::new(-4.4541e-01,  3.7725e-01),
    Complex32::new(-4.4541e-01,  1.8862e-01), Complex32::new(-4.4541e-01,  0.0000e+00),
    Complex32::new(-4.4541e-01, -1.8862e-01), Complex32::new(-4.4541e-01, -3.7425e-01),
    Complex32::new(-4.4541e-01, -5.6287e-01), Complex32::new(-8.1223e-01, -7.5150e-01),
    Complex32::new(-6.2882e-01, -7.5150e-01), Complex32::new(-4.4541e-01, -7.5150e-01),
    Complex32::new(-9.9563e-01, -7.5150e-01), Complex32::new(-9.9563e-01, -9.3114e-01),
    Complex32::new(-9.9563e-01, -1.1198e+00), Complex32::new(-9.9563e-01, -1.3084e+00),
    Complex32::new( 9.9563e-01, -1.4940e+00), Complex32::new( 7.9767e-01, -1.4940e+00),
    Complex32::new( 5.9971e-01, -1.4940e+00), Complex32::new( 3.9884e-01, -1.4940e+00),
    Complex32::new( 2.0087e-01, -1.4940e+00), Complex32::new( 0.0000e+00, -1.4940e+00),
    Complex32::new(-1.9796e-01, -1.4940e+00), Complex32::new(-3.9592e-01, -1.4940e+00),
    Complex32::new(-5.9680e-01, -1.4940e+00), Complex32::new(-7.9476e-01, -1.4940e+00),
    Complex32::new(-9.9563e-01, -1.4940e+00), Complex32::new( 9.9563e-01, -7.5150e-01),
    Complex32::new( 9.9563e-01, -9.3114e-01), Complex32::new( 9.9563e-01, -1.1198e+00),
    Complex32::new( 9.9563e-01, -1.3084e+00), Complex32::new( 8.1514e-01, -7.5150e-01),
    Complex32::new( 6.3173e-01, -7.5150e-01), Complex32::new( 4.4833e-01, -7.5150e-01),
    Complex32::new( 4.4833e-01,  5.6587e-01), Complex32::new( 4.4833e-01,  3.7725e-01),
    Complex32::new( 4.4833e-01,  1.8862e-01), Complex32::new( 4.4833e-01,  0.0000e+00),
    Complex32::new( 4.4833e-01, -1.8862e-01), Complex32::new( 4.4833e-01, -3.7425e-01),
    Complex32::new( 4.4833e-01, -5.6287e-01), Complex32::new( 8.1514e-01,  7.5150e-01),
    Complex32::new( 6.3173e-01,  7.5150e-01), Complex32::new( 4.4833e-01,  7.5150e-01),
    Complex32::new( 9.9563e-01,  1.3114e+00), Complex32::new( 9.9563e-01,  1.1228e+00),
    Complex32::new( 9.9563e-01,  9.3413e-01), Complex32::new( 9.9563e-01,  7.5150e-01)
];