// Structured dot product object

use std::marker::PhantomData;

use super::{DotProd, DotProdBlockPlan, DotProdKernel};
use crate::error::{Error, Result};

/// Structured dot product object. Holds a fixed coefficient array so a dot
/// product can be executed repeatedly against different inputs.
#[derive(Clone, Debug)]
pub struct DotProduct<T, Coeff> {
    h: Vec<Coeff>, // coefficients array
    executor: DotProdKernel<T, Coeff, T>,
    block: Option<DotProdBlockPlan<[T], Coeff, T>>,
    _input: PhantomData<fn(&[T])>,
}

impl<T, Coeff> DotProduct<T, Coeff>
where
    Coeff: Copy,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Create dot product object from an array of coefficients.
    ///
    /// # Arguments
    ///
    /// * `h` - coefficients array
    pub fn new(h: &[Coeff]) -> Result<Self> {
        let h = Self::checked(h)?;
        Ok(Self {
            h: h.to_vec(),
            executor: <[T] as DotProd<Coeff>>::plan(h.len()),
            block: <[T] as DotProd<Coeff>>::plan_block(h),
            _input: PhantomData,
        })
    }

    /// Create dot product object with time-reversed coefficients.
    ///
    /// # Arguments
    ///
    /// * `h` - time-reversed coefficients array
    pub fn new_rev(h: &[Coeff]) -> Result<Self> {
        let h = Self::checked(h)?;
        let h: Vec<_> = h.iter().rev().copied().collect();
        Ok(Self {
            executor: <[T] as DotProd<Coeff>>::plan(h.len()),
            block: <[T] as DotProd<Coeff>>::plan_block(&h),
            h,
            _input: PhantomData,
        })
    }

    /// Set the coefficients, reusing the existing allocation when the length is
    /// unchanged.
    ///
    /// # Arguments
    ///
    /// * `h` - coefficients array
    pub fn set_coefficients(&mut self, h: &[Coeff]) -> Result<()> {
        let h = Self::checked(h)?;
        if h.len() != self.h.len() {
            self.executor = <[T] as DotProd<Coeff>>::plan(h.len());
        }
        self.h.clear();
        self.h.extend_from_slice(h);
        self.block = <[T] as DotProd<Coeff>>::plan_block(&self.h);
        Ok(())
    }

    /// Set the coefficients in time-reversed order, reusing the existing
    /// allocation when the length is unchanged.
    ///
    /// # Arguments
    ///
    /// * `h` - time-reversed coefficients array
    pub fn set_coefficients_rev(&mut self, h: &[Coeff]) -> Result<()> {
        let h = Self::checked(h)?;
        if h.len() != self.h.len() {
            self.executor = <[T] as DotProd<Coeff>>::plan(h.len());
        }
        self.h.clear();
        self.h.extend(h.iter().rev().copied());
        self.block = <[T] as DotProd<Coeff>>::plan_block(&self.h);
        Ok(())
    }

    /// Returns the coefficients, in the order they are applied.
    pub fn coefficients(&self) -> &[Coeff] {
        &self.h
    }

    /// Returns the length of the dot product.
    pub fn len(&self) -> usize {
        self.h.len()
    }

    /// Returns `true` if the dot product has no coefficients.
    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }

    /// Execute the dot product against an input array.
    ///
    /// # Arguments
    ///
    /// * `x` - input array
    ///
    /// # Panics
    ///
    /// Panics if `x` is not the same length as the coefficients.
    #[inline]
    pub fn execute(&self, x: &[T]) -> T {
        assert_eq!(x.len(), self.h.len(), "Slices must have equal length");
        unsafe { (self.executor)(x, &self.h) }
    }

    /// Execute overlapping dot products over a contiguous input span.
    ///
    /// Produces `y[i] = self.execute(&x[i..i + self.len()])` for every output.
    ///
    /// # Panics
    ///
    /// Panics unless `x.len() == y.len() + self.len() - 1`.
    pub fn execute_block(&self, x: &[T], y: &mut [T]) {
        let expected = y.len().checked_add(self.h.len() - 1)
            .expect("dot product block length overflow");
        assert_eq!(x.len(), expected, "Invalid sliding dot product block length");

        let completed = self.block.as_ref().map_or(0, |block| {
            // respect block executor's input and output widths
            let block_outputs = x.len().saturating_sub(block.input_width - 1).min(y.len());
            if block_outputs < block.output_width {
                0
            } else {
                unsafe { (block.executor)(x, &block.h, y) }
            }
        });
        debug_assert!(completed <= y.len());

        // block execution may leave some samples uncomputed
        // use the fallback executor on whatever's left, one at a time
        for (i, yi) in y[completed..].iter_mut().enumerate() {
            let i = completed + i;
            *yi = unsafe { (self.executor)(&x[i..i + self.h.len()], &self.h) };
        }
    }

    fn checked(h: &[Coeff]) -> Result<&[Coeff]> {
        if h.is_empty() {
            return Err(Error::Config("dotprod length must be greater than zero".into()));
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::random::{crandnf, randnf};
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct)]
    fn test_dotprod_rrrf_struct() {
        const TOL: f32 = 1e-6;

        #[rustfmt::skip]
        let h: Vec<f32> = vec![
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        ];

        let dp = DotProduct::<f32, f32>::new(&h).unwrap();

        let x0 = vec![0.0; 16];
        assert_abs_diff_eq!(dp.execute(&x0), 0.0, epsilon = TOL);

        let x1 = vec![1.0; 16];
        assert_abs_diff_eq!(dp.execute(&x1), 0.0, epsilon = TOL);

        let x2: Vec<f32> = (0..16).map(|i| (i % 2) as f32).collect();
        assert_abs_diff_eq!(dp.execute(&x2), -8.0, epsilon = TOL);

        assert_abs_diff_eq!(dp.execute(&h), 16.0, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct_align)]
    fn test_dotprod_rrrf_struct_align() {
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: Vec<f32> = vec![
            -0.050565,   -0.952580,    0.274320,    1.232400,
             1.268200,    0.565770,    0.800830,    0.923970,
             0.517060,   -0.530340,   -0.378550,   -1.127100,
             1.123100,   -1.006000,   -1.483800,   -0.062007,
        ];

        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            -0.384280,   -0.812030,    0.156930,    1.919500,
             0.564580,   -0.123610,   -0.138640,    0.004984,
            -1.100200,   -0.497620,    0.089977,   -1.745500,
             0.463640,    0.592100,    1.150000,   -1.225400,
        ];

        let test = 3.66411513609863;
        let dp = DotProduct::<f32, f32>::new(&h).unwrap();

        // test data misalignment conditions
        let mut x_buffer = [0.0f32; 20];
        for i in 0..4 {
            x_buffer[i..i + 16].copy_from_slice(&x);
            let y = dp.execute(&x_buffer[i..i + 16]);
            assert_abs_diff_eq!(y, test, epsilon = TOL);
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_rand02)]
    fn test_dotprod_rrrf_rand02() {
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: Vec<f32> = vec![
             2.595300,    1.243600,   -0.818550,   -1.439800,
             0.055795,   -1.476000,    0.445900,    0.325460,
            -3.451200,    0.058528,   -0.246990,    0.476290,
            -0.598780,   -0.885250,    0.464660,   -0.610140,
        ];

        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            -0.917010,   -1.278200,   -0.533190,    2.309200,
             0.592980,    0.964820,    0.183220,   -0.082864,
             0.057171,   -1.186500,   -0.738260,    0.356960,
            -0.144000,   -1.435200,   -0.893420,    1.657800,
        ];

        let test = -8.17832326680587;
        let test_rev = 4.56839328512000;

        assert_abs_diff_eq!(h.dotprod(&x), test, epsilon = TOL);

        let q = DotProduct::<f32, f32>::new(&h).unwrap();
        assert_abs_diff_eq!(q.execute(&x), test, epsilon = TOL);

        let q_rev = DotProduct::<f32, f32>::new_rev(&h).unwrap();
        assert_abs_diff_eq!(q_rev.execute(&x), test_rev, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct_lengths)]
    fn test_dotprod_rrrf_struct_lengths() {
        const TOL: f32 = 2e-6;

        #[rustfmt::skip]
        let x: Vec<f32> = vec![
             0.03117498,  -1.54311769,  -0.58759073,  -0.73882202,
             0.86592259,  -0.26669417,  -0.70153724,  -1.24555787,
            -1.09272288,  -1.41984975,  -1.40299260,   0.95861481,
            -0.67361246,   2.05305710,   1.26576873,  -0.77474848,
            -0.93143252,  -1.05724660,   0.21455006,   1.07554168,
            -0.46703810,   0.68878404,  -1.11900266,  -0.52016966,
             0.61400744,  -0.46506142,  -0.16801031,   0.48237303,
             0.51286055,  -0.57239385,  -0.64462740,  -0.75596668,
             1.95612355,  -0.47917908,   0.52384983,
        ];

        #[rustfmt::skip]
        let h: Vec<f32> = vec![
            -0.12380948,   0.88417134,   2.27373797,  -2.61506417,
             0.35022002,   0.07481393,   0.52984228,  -0.65542307,
            -2.14893606,   0.62466395,   0.07330391,  -1.28014856,
             0.16347776,   0.21238151,   0.05462232,  -0.60290942,
            -1.27658956,   3.05114996,   1.34789601,  -1.22098592,
             1.70899633,  -0.41002037,   3.08009931,  -1.39895771,
            -0.50875066,   0.25817865,   1.08668549,   0.05494174,
            -1.05337166,   1.26772604,   1.00369204,  -0.55129338,
             1.01828299,   0.76014664,  -0.15605569,
        ];

        for (n, expected) in [
            (32, -7.99577847),
            (33, -6.00389114),
            (34, -6.36813751),
            (35, -6.44988725),
        ] {
            let dp = DotProduct::<f32, f32>::new(&h[..n]).unwrap();
            assert_abs_diff_eq!(dp.execute(&x[..n]), expected, epsilon = TOL);
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct_vs_ordinal)]
    fn test_dotprod_rrrf_struct_vs_ordinal() {
        const TOL: f32 = 1e-4;

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| randnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            // expected value (ordinal computation)
            let y_test: f32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();

            // validate result (structured object)
            let dp = DotProduct::<f32, f32>::new(&h).unwrap();
            assert_abs_diff_eq!(dp.execute(&x), y_test, epsilon = TOL);

            // validate result (unstructured)
            let res = h.dotprod(&x);
            assert_abs_diff_eq!(res, y_test, epsilon = TOL);

            // no "run4"
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_rand01)]
    fn test_dotprod_crcf_rand01() {
        use num_complex::Complex32 as Cf32;
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: [f32; 16] = [
             5.5375e-02,  -6.5857e-01,  -1.7657e+00,   7.7444e-01,
             8.0730e-01,  -5.1340e-01,  -9.3437e-02,  -5.6301e-01,
            -6.6480e-01,  -2.1673e+00,   9.0269e-01,   3.5284e+00,
            -9.7835e-01,  -6.9512e-01,  -1.2958e+00,   1.1628e+00,
        ];

        #[rustfmt::skip]
        let x: [Cf32; 16] = [
            Cf32::new( 1.3164e+00,  5.4161e-01),  Cf32::new( 1.8295e-01, -9.0284e-02),
            Cf32::new( 1.3487e+00, -1.8148e+00),  Cf32::new(-7.4696e-01, -4.1792e-01),
            Cf32::new(-9.0551e-01, -4.4294e-01),  Cf32::new( 6.0591e-01, -1.5383e+00),
            Cf32::new(-7.5393e-01, -3.5691e-01),  Cf32::new(-4.5733e-01,  1.1926e-01),
            Cf32::new(-1.4744e-01, -4.7676e-02),  Cf32::new(-1.2422e+00, -2.0213e+00),
            Cf32::new( 3.3208e-02, -1.3756e+00),  Cf32::new(-4.8573e-01,  1.0977e+00),
            Cf32::new( 1.5053e+00,  2.1141e-01),  Cf32::new(-8.4062e-01, -1.0211e+00),
            Cf32::new(-1.3932e+00, -4.8491e-01),  Cf32::new(-1.4234e+00,  2.0333e-01),
        ];

        let test = Cf32::new(-3.35346556487224, 11.78023318618137);
        let test_rev = Cf32::new(3.655541203500000, 4.26531912591000);
        
        let y = h.dotprod(&x);
        // no "run4"
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);

        let dp = DotProduct::<Complex32, f32>::new(&h).unwrap();
        let y = dp.execute(&x);
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);

        let dp_rev = DotProduct::<Complex32, f32>::new_rev(&h).unwrap();
        let y_rev = dp_rev.execute(&x);
        assert_abs_diff_eq!(y_rev.re, test_rev.re, epsilon = TOL);
        assert_abs_diff_eq!(y_rev.im, test_rev.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_rand02)]
    fn test_dotprod_crcf_rand02() {
        use num_complex::Complex32 as Cf32;
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: [f32; 16] = [
             4.7622e-01,   7.1453e-01,  -7.1370e-01,  -1.6457e-01,
            -1.1573e-01,   6.4114e-01,  -1.0688e+00,  -1.6761e+00,
            -1.0376e+00,  -1.0991e+00,  -2.4161e-01,   4.6065e-01,
            -1.0403e+00,  -1.1424e-01,  -1.2371e+00,  -7.9723e-01,
        ];

        #[rustfmt::skip]
        let x: [Cf32; 16] = [
            Cf32::new(-8.3558e-01,  3.0504e-01),  Cf32::new(-6.3004e-01,  2.4680e-01),
            Cf32::new( 9.6908e-01,  1.2978e+00),  Cf32::new(-2.0587e+00,  9.5385e-01),
            Cf32::new( 2.5692e-01, -1.7314e+00),  Cf32::new(-1.2237e+00, -6.2139e-02),
            Cf32::new( 5.0300e-02, -9.2092e-01),  Cf32::new(-1.8816e-01,  7.0746e-02),
            Cf32::new(-2.4177e+00,  8.3177e-01),  Cf32::new( 1.6871e-01, -8.5129e-02),
            Cf32::new( 6.5203e-01,  2.0739e-02),  Cf32::new(-1.2331e-01, -9.7920e-01),
            Cf32::new( 8.2352e-01,  9.1093e-01),  Cf32::new( 1.5161e+00, -9.1865e-01),
            Cf32::new(-2.0892e+00,  2.7759e-02),  Cf32::new(-2.5188e-01,  2.5568e-01),
        ];

        let test = Cf32::new(2.11053363855085, -2.04167493441477);
        let y = h.dotprod(&x);
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_struct_vs_ordinal)]
    fn test_dotprod_crcf_struct_vs_ordinal() {
        const TOL: f32 = 1e-4;

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| randnf()).collect();
            let x: Vec<Complex32> = (0..n).map(|_| crandnf()).collect();

            // expected value (ordinal computation)
            let y_test: Complex32 = h.iter().zip(x.iter()).map(|(&a, &b)| b * a).sum();

            let dp = DotProduct::<Complex32, f32>::new(&h).unwrap();
            let y = dp.execute(&x);
            assert_abs_diff_eq!(y.re, y_test.re, epsilon = TOL);
            assert_abs_diff_eq!(y.im, y_test.im, epsilon = TOL);

            let y = h.dotprod(&x);
            assert_abs_diff_eq!(y.re, y_test.re, epsilon = TOL);
            assert_abs_diff_eq!(y.im, y_test.im, epsilon = TOL);
 
            // no run4
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_cccf_rand16)]
    fn test_dotprod_cccf_rand16() {
        use num_complex::Complex32 as Cf32;
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: [Cf32; 16] = [
            Cf32::new( 0.17702709,  1.38978455),  Cf32::new( 0.91294148,  0.39217381),
            Cf32::new(-0.80607338,  0.76477512),  Cf32::new( 0.05099755, -0.87350051),
            Cf32::new( 0.44513826, -0.49490569),  Cf32::new( 0.14754967,  2.04349962),
            Cf32::new( 1.07246623,  1.08146290),  Cf32::new(-1.14028088,  1.83380899),
            Cf32::new( 0.38105361, -0.45591846),  Cf32::new( 0.32605401,  0.34440081),
            Cf32::new(-0.05477144,  0.60832595),  Cf32::new( 1.81667523, -1.12238075),
            Cf32::new(-0.87190497,  1.10743858),  Cf32::new( 1.30921403,  1.24438643),
            Cf32::new( 0.55524695, -1.94931519),  Cf32::new(-0.87191170,  0.91693119),
        ];

        #[rustfmt::skip]
        let x: [Cf32; 16] = [
            Cf32::new(-2.19591953, -0.93229692),  Cf32::new( 0.17150376,  0.56165114),
            Cf32::new( 1.58354529, -0.50696037),  Cf32::new( 1.40929619,  0.87868803),
            Cf32::new(-0.75505072, -0.30867372),  Cf32::new(-0.09821367, -0.73949106),
            Cf32::new( 0.03785571,  0.72763665),  Cf32::new(-1.20262636, -0.88838102),
            Cf32::new( 0.23323685,  0.12456235),  Cf32::new( 0.34593736,  0.02529594),
            Cf32::new( 0.33669564,  0.39064649),  Cf32::new(-2.45003867, -0.54862205),
            Cf32::new(-2.64870707,  2.33444473),  Cf32::new(-0.92284477, -2.45121397),
            Cf32::new( 0.24852918, -0.62409860),  Cf32::new(-0.87039907,  0.90921212),
        ];

        let test = Cf32::new(-0.604285042605890, -12.390925785344704);
        let test_rev = Cf32::new(3.412365881765360, 6.1885320363931480);

        let y = h.dotprod(&x);
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);

        // no run4

        let dp = DotProduct::<Complex32, Complex32>::new(&h).unwrap();
        let y = dp.execute(&x);
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);

        let dp_rev = DotProduct::<Complex32, Complex32>::new_rev(&h).unwrap();
        let y_rev = dp_rev.execute(&x);
        assert_abs_diff_eq!(y_rev.re, test_rev.re, epsilon = TOL);
        assert_abs_diff_eq!(y_rev.im, test_rev.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_cccf_struct_lengths)]
    fn test_dotprod_cccf_struct_lengths() {
        const TOL: f32 = 4e-6;

        #[rustfmt::skip]
        let h: Vec<Complex32> = [
            ( 1.11555653,  2.30658043), (-0.36133676, -0.10917327),
            ( 0.17714505, -2.14631440), ( 2.20424609,  0.59063608),
            (-0.44699194,  0.23369318), ( 0.60613931,  0.21868288),
            (-1.18746289, -0.52159563), (-0.46277775,  0.75010157),
            ( 0.93796307,  0.28608151), (-2.18699829,  0.38029319),
            ( 0.16145611,  0.18343353), (-0.62653631, -1.79037656),
            (-0.67042462,  0.11044084), ( 0.70333438,  1.78729174),
            (-0.32923580,  0.78514690), ( 0.27534332, -0.56377431),
            ( 0.41492559,  1.37176526), ( 3.25368958,  2.70495218),
            ( 1.63002035, -0.14193750), ( 2.22057186,  0.55056461),
            ( 1.40896777,  0.80722903), (-0.22334033, -0.14227395),
            (-1.48631186,  0.53610531), (-1.91632185,  0.88755083),
            (-0.52054895, -0.35572001), (-1.56515607, -0.41448794),
            (-0.91107117,  0.17059659), (-0.77007659,  2.73381816),
            (-0.46645585,  0.38994666), ( 0.80317663, -0.41756968),
            ( 0.26992512,  0.41828145), (-0.72456446,  1.25002030),
            ( 1.19573306,  0.98449546), ( 1.42491943, -0.55426305),
            ( 1.08243614,  0.35774368),
        ].iter().map(|&(re, im)| Complex32::new(re, im)).collect();

        #[rustfmt::skip]
        let x: Vec<Complex32> = [
            (-0.82466736, -1.39329228), (-1.46176052, -1.96218827),
            (-1.28388174, -0.07152934), (-0.51910014, -0.37915971),
            (-0.65964708, -0.98417534), (-1.40213479, -0.82198463),
            ( 0.86051446,  0.97926463), ( 0.26257342,  0.76586696),
            ( 0.72174183, -1.89884636), (-0.26018863,  1.06920599),
            ( 0.57949117, -0.77431546), ( 0.84635184, -0.81123009),
            (-1.12637629, -0.42027412), (-1.04214881,  0.90519721),
            ( 0.54458433, -1.03487314), (-0.17847893,  2.20358978),
            ( 0.19642532, -0.07449796), (-1.84958229,  0.13218920),
            (-1.49042886,  0.81610408), (-0.27466940, -1.48438409),
            ( 0.29239375,  0.72443343), (-1.20243456, -2.77032750),
            (-0.41784260,  0.77455254), ( 0.37737465, -0.52426993),
            (-1.25500377,  1.76270122), ( 1.55976056, -1.18189171),
            (-0.05111343, -1.18849396), (-1.92966664,  0.66504899),
            (-2.82387897,  1.41128242), (-1.48171326, -0.03347470),
            ( 0.38047273, -1.40969799), ( 1.71995272,  0.00298203),
            ( 0.56040910, -0.12713027), (-0.46653022, -0.65450499),
            ( 0.15515755,  1.58944030),
        ].iter().map(|&(re, im)| Complex32::new(re, im)).collect();

        for (n, expected) in [
            (32, Complex32::new(-11.5100903519506, -15.3575526884014)),
            (33, Complex32::new(-10.7148314918614, -14.9578463360225)),
            (34, Complex32::new(-11.7423673921916, -15.6318827515320)),
            (35, Complex32::new(-12.1430314741466, -13.8559085000689)),
        ] {
            let dp = DotProduct::<Complex32, Complex32>::new(&h[..n]).unwrap();
            let y = dp.execute(&x[..n]);
            assert_abs_diff_eq!(y.re, expected.re, epsilon = TOL);
            assert_abs_diff_eq!(y.im, expected.im, epsilon = TOL);
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_cccf_struct_vs_ordinal)]
    fn test_dotprod_cccf_struct_vs_ordinal() {
        const TOL: f32 = 1e-4;

        for n in 1..=512 {
            let h: Vec<Complex32> = (0..n).map(|_| crandnf()).collect();
            let x: Vec<Complex32> = (0..n).map(|_| crandnf()).collect();

            // expected value (ordinal computation)
            let y_test: Complex32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();

            let dp = DotProduct::<Complex32, Complex32>::new(&h).unwrap();
            let y = dp.execute(&x);
            assert_abs_diff_eq!(y.re, y_test.re, epsilon = TOL);
            assert_abs_diff_eq!(y.im, y_test.im, epsilon = TOL);

            let y = h.dotprod(&x);
            assert_abs_diff_eq!(y.re, y_test.re, epsilon = TOL);
            assert_abs_diff_eq!(y.im, y_test.im, epsilon = TOL);

            // no run4
        }
    }

    #[test]
    fn test_dotprod_rrr_execute_block_matches_execute() {
        for k in (1..=384).chain([512, 1024]) {
            let h: Vec<f32> = (0..k)
                .map(|i| ((i + 1) as f32 * 0.173).sin())
                .collect();
            let dp = DotProduct::<f32, f32>::new(&h).unwrap();

            for n in 0..=20 {
                let offset = k % 4;
                let storage: Vec<f32> = (0..offset + n + k - 1)
                    .map(|i| ((i + 3) as f32 * 0.117).cos())
                    .collect();
                let x = &storage[offset..];
                let expected: Vec<_> = (0..n)
                    .map(|i| dp.execute(&x[i..i + k]))
                    .collect();
                let mut actual = vec![0.0; n];
                dp.execute_block(x, &mut actual);

                for (&actual, &expected) in actual.iter().zip(&expected) {
                    assert_abs_diff_eq!(
                        actual,
                        expected,
                        epsilon = (2.0 * k as f32 * f32::EPSILON).max(1e-5),
                    );
                }
            }
        }
    }

    #[test]
    fn test_dotprod_crc_execute_block_matches_execute() {
        for k in (1..=184).chain([256, 1024]) {
            let h: Vec<f32> = (0..k)
                .map(|i| ((i + 1) as f32 * 0.173).sin())
                .collect();
            let dp = DotProduct::<Complex32, f32>::new(&h).unwrap();

            for n in 0..=10 {
                let offset = k % 3;
                let storage: Vec<Complex32> = (0..offset + n + k - 1)
                    .map(|i| {
                        Complex32::new(
                            ((i + 3) as f32 * 0.117).cos(),
                            ((i + 5) as f32 * 0.091).sin(),
                        )
                    })
                    .collect();
                let x = &storage[offset..];
                let expected: Vec<_> = (0..n)
                    .map(|i| dp.execute(&x[i..i + k]))
                    .collect();
                let mut actual = vec![Complex32::default(); n];
                dp.execute_block(x, &mut actual);

                for (&actual, &expected) in actual.iter().zip(&expected) {
                    assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-4);
                    assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_dotprod_ccc_execute_block_matches_execute() {
        for k in (1..=136).chain([256, 1024]) {
            let h: Vec<Complex32> = (0..k)
                .map(|i| Complex32::new(
                    ((i + 1) as f32 * 0.173).sin(),
                    ((i + 2) as f32 * 0.137).cos(),
                ))
                .collect();
            let dp = DotProduct::<Complex32, Complex32>::new(&h).unwrap();

            for n in 0..=10 {
                let offset = k % 3;
                let storage: Vec<Complex32> = (0..offset + n + k - 1)
                    .map(|i| {
                        Complex32::new(
                            ((i + 3) as f32 * 0.117).cos(),
                            ((i + 5) as f32 * 0.091).sin(),
                        )
                    })
                    .collect();
                let x = &storage[offset..];
                let expected: Vec<_> = (0..n)
                    .map(|i| dp.execute(&x[i..i + k]))
                    .collect();
                let mut actual = vec![Complex32::default(); n];
                dp.execute_block(x, &mut actual);

                for (&actual, &expected) in actual.iter().zip(&expected) {
                    assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-4);
                    assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_dotprod_execute_block_replans_coefficients() {
        let mut dp = DotProduct::<f32, f32>::new(&[1.0, 2.0, 3.0]).unwrap();

        for k in [3, 33, 352, 368] {
            let h: Vec<_> = (0..k)
                .map(|i| ((i + 1) as f32 * 0.173).sin())
                .collect();
            dp.set_coefficients(&h).unwrap();
            let x: Vec<_> = (0..h.len() + 24 - 1)
                .map(|i| ((i + 3) as f32 * 0.117).cos())
                .collect();
            let expected: Vec<_> = x.windows(h.len()).map(|x| dp.execute(x)).collect();
            let mut actual = vec![0.0; 24];
            dp.execute_block(&x, &mut actual);
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
            }
        }

        let h = [7.0, 8.0, 9.0, 10.0, 11.0];
        dp.set_coefficients_rev(&h).unwrap();
        let x: Vec<_> = (0..h.len() + 16 - 1).map(|i| i as f32 * 0.25).collect();
        let expected: Vec<_> = x.windows(h.len()).map(|x| dp.execute(x)).collect();
        let mut actual = vec![0.0; 16];
        dp.execute_block(&x, &mut actual);
        assert_eq!(actual, expected);

        let mut dp = DotProduct::<Complex32, f32>::new(&[1.0, 2.0, 3.0]).unwrap();
        for k in [3, 65, 176] {
            let h: Vec<_> = (0..k)
                .map(|i| ((i + 1) as f32 * 0.173).sin())
                .collect();
            dp.set_coefficients_rev(&h).unwrap();
            let x: Vec<_> = (0..h.len() + 16 - 1)
                .map(|i| Complex32::new(i as f32 * 0.25, i as f32 * -0.125))
                .collect();
            let expected: Vec<_> = x.windows(h.len()).map(|x| dp.execute(x)).collect();
            let mut actual = vec![Complex32::default(); 16];
            dp.execute_block(&x, &mut actual);
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-4);
                assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-4);
            }
        }

        let mut dp = DotProduct::<Complex32, Complex32>::new(&[
            Complex32::new(1.0, 0.5),
            Complex32::new(2.0, -0.25),
            Complex32::new(3.0, 0.125),
        ]).unwrap();
        for k in [3, 17, 80] {
            let h: Vec<_> = (0..k)
                .map(|i| Complex32::new(
                    ((i + 1) as f32 * 0.173).sin(),
                    ((i + 2) as f32 * 0.137).cos(),
                ))
                .collect();
            dp.set_coefficients_rev(&h).unwrap();
            let x: Vec<_> = (0..h.len() + 16 - 1)
                .map(|i| Complex32::new(i as f32 * 0.25, i as f32 * -0.125))
                .collect();
            let expected: Vec<_> = x.windows(h.len()).map(|x| dp.execute(x)).collect();
            let mut actual = vec![Complex32::default(); 16];
            dp.execute_block(&x, &mut actual);
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-4);
                assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-4);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Invalid sliding dot product block length")]
    fn test_dotprod_execute_block_invalid_input_panics() {
        let dp = DotProduct::<f32, f32>::new(&[1.0, 2.0, 3.0]).unwrap();
        dp.execute_block(&[1.0, 2.0, 3.0], &mut [0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "Slices must have equal length")]
    fn test_dotprod_rrr_struct_short_input_panics() {
        let q = DotProduct::<f32, f32>::new(&[1.0, 2.0, 3.0]).unwrap();
        q.execute(&[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "Slices must have equal length")]
    fn test_dotprod_rrr_struct_long_input_panics() {
        let q = DotProduct::<f32, f32>::new(&[1.0, 2.0, 3.0]).unwrap();
        q.execute(&[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dotprod_struct_zero_length_rejected() {
        assert!(DotProduct::<Complex32, Complex32>::new(&[]).is_err());
        assert!(DotProduct::<Complex32, Complex32>::new_rev(&[]).is_err());
        assert!(DotProduct::<Complex32, f32>::new(&[]).is_err());
        assert!(DotProduct::<Complex32, f32>::new_rev(&[]).is_err());
    }
}
