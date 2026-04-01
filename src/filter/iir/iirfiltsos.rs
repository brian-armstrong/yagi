use crate::error::Result;
use num_complex::ComplexFloat;
use crate::filter::iir::design::iir_group_delay;

/// Struct definition
#[derive(Debug, Clone)]
pub struct IirFilterSos<T, Coeff = T> {
    b: [Coeff; 3],  // feed-forward coefficients
    a: [Coeff; 3],  // feed-back coefficients

    // internal buffering
    x: [T; 3],  // Direct form I  buffer (input)
    y: [T; 3],  // Direct form I  buffer (output)
    v: [T; 3],  // Direct form II buffer
}

impl<T, Coeff> IirFilterSos<T, Coeff>
where
    T: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T>,
    Coeff: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<T, Output = T>,
{
    // create iirfiltsos object
    pub fn new(b: &[Coeff; 3], a: &[Coeff; 3]) -> Result<Self> {
        let mut filter = IirFilterSos {
            b: [Coeff::default(); 3],
            a: [Coeff::default(); 3],
            x: [T::default(); 3],
            y: [T::default(); 3],
            v: [T::default(); 3],
        };

        // set the internal coefficients
        filter.set_coefficients(b, a)?;

        // clear filter state and return object
        filter.reset();
        Ok(filter)
    }

    // set internal filter coefficients
    pub fn set_coefficients(&mut self, b: &[Coeff; 3], a: &[Coeff; 3]) -> Result<()> {
        // retain a0 coefficient for normalization
        let a0 = a[0];

        // copy feed-forward coefficients (numerator)
        self.b[0] = b[0] / a0;
        self.b[1] = b[1] / a0;
        self.b[2] = b[2] / a0;

        // copy feed-back coefficients (denominator)
        self.a[0] = a[0] / a0;  // unity
        self.a[1] = a[1] / a0;
        self.a[2] = a[2] / a0;

        Ok(())
    }

    // clear/reset iirfiltsos object internals
    pub fn reset(&mut self) {
        self.v[0] = T::default();
        self.v[1] = T::default();
        self.v[2] = T::default();

        self.x[0] = T::default();
        self.x[1] = T::default();
        self.x[2] = T::default();

        self.y[0] = T::default();
        self.y[1] = T::default();
        self.y[2] = T::default();
    }

    // compute filter output
    pub fn execute(&mut self, x: T) -> T {
        self.execute_df2(x)
    }

    // compute filter output, direct form I method
    pub fn execute_df1(&mut self, x: T) -> T {
        // advance buffer x
        self.x[2] = self.x[1];
        self.x[1] = self.x[0];
        self.x[0] = x;

        // advance buffer y
        self.y[2] = self.y[1];
        self.y[1] = self.y[0];

        // compute new v
        let v = self.x[0] * self.b[0] +
                self.x[1] * self.b[1] +
                self.x[2] * self.b[2];

        // compute new y[0]
        self.y[0] = v -
                    self.y[1] * self.a[1] -
                    self.y[2] * self.a[2];

        self.y[0]
    }

    // compute filter output, direct form II method
    pub fn execute_df2(&mut self, x: T) -> T {
        // advance buffer
        self.v[2] = self.v[1];
        self.v[1] = self.v[0];

        // compute new v[0]
        self.v[0] = x -
                    self.a[1] * self.v[1] -
                    self.a[2] * self.v[2];

        // compute output y
        self.b[0] * self.v[0] +
        self.b[1] * self.v[1] +
        self.b[2] * self.v[2]
    }

    // compute group delay in samples
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        // copy coefficients
        let b: [f32; 3] = [self.b[0].re(), self.b[1].re(), self.b[2].re()];
        let a: [f32; 3] = [self.a[0].re(), self.a[1].re(), self.a[2].re()];

        Ok(iir_group_delay(&b, &a, fc)? + 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;
    use num_complex::Complex32;
    
    #[test]
    #[autotest_annotate(autotest_iirfiltsos_impulse_n2)]
    fn test_iirfiltsos_impulse_n2() {
        // initialize filter with 2nd-order low-pass butterworth filter
        let a = [
            1.000000000000000,
           -0.942809041582063,
            0.333333333333333
        ];
    
        let b = [
            0.0976310729378175,
            0.1952621458756350,
            0.0976310729378175
        ];
    
        // create identical objects
        let mut q0 = IirFilterSos::<f32, f32>::new(&b, &a).unwrap();
        let mut q1 = IirFilterSos::<f32, f32>::new(&b, &a).unwrap();
    
        // initialize oracle; expected output (generated with octave)
        let test = [
           9.76310729378175e-02,
           2.87309604180767e-01,
           3.35965474513536e-01,
           2.20981418970514e-01,
           9.63547883225231e-02,
           1.71836926400291e-02,
          -1.59173219853878e-02,
          -2.07348926322729e-02,
          -1.42432702548109e-02,
          -6.51705310050832e-03,
          -1.39657983602602e-03,
           8.55642936806248e-04,
           1.27223450919543e-03,
           9.14259886013424e-04,
           4.37894317157432e-04
        ];
    
        let tol = 1e-4f32;
    
        // hit filter with impulse, compare output
        for i in 0..15 {
            // generate input
            let v = if i == 0 { 1.0f32 } else { 0.0f32 };
    
            // run direct-form I
            let y = q0.execute_df1(v);
            assert_relative_eq!(test[i], y, epsilon = tol);
    
            // run direct-form II
            let y = q1.execute_df2(v);
            assert_relative_eq!(test[i], y, epsilon = tol);
        }
    }
    
    #[test]
    #[autotest_annotate(autotest_iirfiltsos_step_n2)]
    fn test_iirfiltsos_step_n2() {
        // initialize filter with 2nd-order low-pass butterworth filter
        let a = [
            1.000000000000000,
           -0.942809041582063,
            0.333333333333333
        ];
    
        let b = [
            0.0976310729378175,
            0.1952621458756350,
            0.0976310729378175
        ];
    
        // create identical objects
        let mut q0 = IirFilterSos::<f32, f32>::new(&b, &a).unwrap();
        let mut q1 = IirFilterSos::<f32, f32>::new(&b, &a).unwrap();
    
        let test = [
           0.0976310729378175,
           0.3849406771185847,
           0.7209061516321208,
           0.9418875706026352,
           1.0382423589251584,
           1.0554260515651877,
           1.0395087295798000,
           1.0187738369475272,
           1.0045305666927162,
           0.9980135135922078,
           0.9966169337561817,
           0.9974725766929878,
           0.9987448112021832,
           0.9996590710881966,
           1.0000969654053542
        ];
    
        let tol = 1e-4f32;
    
        // hit filter with step, compare output
        for i in 0..15 {
            // run direct-form I
            let y = q0.execute_df1(1.0);
            assert_relative_eq!(test[i], y, epsilon = tol);
    
            // run direct-form II
            let y = q1.execute_df2(1.0);
            assert_relative_eq!(test[i], y, epsilon = tol);
        }
    }
    

    #[test]
    #[autotest_annotate(autotest_iirfiltsos_copy)]
    fn test_iirfiltsos_copy() {
        // initialize filter with 2nd-order low-pass butterworth filter
        let a = [1.0000000000000000f32, -0.942809041582063f32, 0.3333333333333333f32];
        let b = [0.0976310729378175f32,  0.195262145875635f32, 0.0976310729378175f32];
    
        // create base object
        let mut q0 = IirFilterSos::<Complex32, f32>::new(&b, &a).unwrap();
    
        // start running input through filter
        let num_samples = 80;
        for _ in 0..num_samples {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.execute(v);
        }
    
        // copy filter
        let mut q1 = q0.clone();
    
        // continue running through both filters
        for _ in 0..num_samples {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            let y0 = q0.execute(v);
            let y1 = q1.execute(v);
    
            // compare result
            assert_eq!(y0, y1);
        }
    }
    
    #[test]
    #[autotest_annotate(autotest_iirfiltsos_config)]
    fn test_iirfiltsos_config() {
        // test copying/creating invalid objects
        // assert!(IirFiltSos::<f32, f32, f32>::clone(None).is_none());
        // nothing to be done - you can't clone an invalid object
    }
}