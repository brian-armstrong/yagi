use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct Direct {}

impl Direct {
    pub fn new() -> Self {
        Direct {}
    }

    pub fn sin(&self, theta: u32) -> f32 {
        (theta as f32 * PI / (i32::MAX as u32 + 1) as f32).sin()
    }

    pub fn cos(&self, theta: u32) -> f32 {
        (theta as f32 * PI / (i32::MAX as u32 + 1) as f32).cos()
    }

    pub fn sin_cos(&self, theta: u32) -> (f32, f32) {
        (self.sin(theta), self.cos(theta))
    }
}

impl Default for Direct {
    fn default() -> Self {
        Self::new()
    }
}
