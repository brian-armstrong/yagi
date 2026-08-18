use crate::modem::modem::*;


// V.29 star constellation
pub(super) const MODEM_ARB_V29: [Complex32; 16] = [
      Complex32::new( 0.06804100,  0.06804100),   Complex32::new( 0.20412000,  0.00000000), 
      Complex32::new( 0.00000000,  0.20412000),   Complex32::new(-0.06804100,  0.06804100), 
      Complex32::new( 0.00000000, -0.20412000),   Complex32::new( 0.06804100, -0.06804100), 
      Complex32::new(-0.06804100, -0.06804100),   Complex32::new(-0.20412000,  0.00000000), 
      Complex32::new( 0.20412000,  0.20412000),   Complex32::new( 0.34021000,  0.00000000), 
      Complex32::new( 0.00000000,  0.34021000),   Complex32::new(-0.20412000,  0.20412000), 
      Complex32::new( 0.00000000, -0.34021000),   Complex32::new( 0.20412000, -0.20412000), 
      Complex32::new(-0.20412000, -0.20412000),   Complex32::new(-0.34021000,  0.00000000)
];