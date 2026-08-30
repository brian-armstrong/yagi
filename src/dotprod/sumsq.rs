// Sum of squares: sum{ |x|^2 }

use num_complex::Complex32;
use super::DotProd;

/// Computes the sum of squares of a real vector: `sum{ x[i]^2 }`
pub fn sumsqf(v: &[f32]) -> f32 {
    // the dotprod is carefully optimized, so just reuse it
    // this does leave out some performance, but this reduces code burden
    v.dotprod(v)
}

/// Computes the sum of squares of a complex vector: `sum{ |x[i]|^2 }`
pub fn sumsqcf(v: &[Complex32]) -> f32 {
    // `|a + bi|^2 = a^2 + b^2`
    // reinterpret the &[Complex32] as a &[f32] that's twice as long
    let v: &[f32] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, 2 * v.len()) };
    sumsqf(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;
    const TOL: f32 = 1e-6;

    #[test]
    #[autotest_annotate(autotest_sumsqf_3)]
    fn test_sumsqf_3() {
        let x = [-0.4546496371984978, 0.4451201395218938, 0.0138788690209525];
        assert_abs_diff_eq!(sumsqf(&x), 0.405030854218017, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqf_4)]
    fn test_sumsqf_4() {
        let x = [0.1322698385026883, -0.0569081631536912, -0.3244384492417431, -0.2872733941910143];
        assert_abs_diff_eq!(sumsqf(&x), 0.208520159567467, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqf_7)]
    fn test_sumsqf_7() {
        #[rustfmt::skip]
        let x = [
            -0.221079351597278, -0.227902662215897,  0.382941891419158,
             0.246800053933030, -0.190152017725480,  0.395758452636014,
             0.211220685416265,
        ];
        assert_abs_diff_eq!(sumsqf(&x), 0.545767182598435, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqf_8)]
    fn test_sumsqf_8() {
        #[rustfmt::skip]
        let x = [
            -0.3405090291337944,  0.5568858414046379, -0.0870643704340343,
             0.1724369367547939, -0.7379946538182081, -0.3514326419380984,
             0.2782541955998314,  0.4354875172406391,
        ];
        assert_abs_diff_eq!(sumsqf(&x), 1.39859872696022, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqf_15)]
    fn test_sumsqf_15() {
        #[rustfmt::skip]
        let x = [
            -0.4630291295549499, -0.2776019612369674, -0.4933486186123937,
            -0.0850997992116534,  0.0117036410972943,  0.0215560948199280,
             0.1203298759952301,  0.5866344749815807,  0.3791165816771581,
            -0.4070288299889871, -0.4971431800502791, -0.2142770391709351,
             0.3330589842198580, -0.0150675851612766, -0.3947266044391958,
        ];
        assert_abs_diff_eq!(sumsqf(&x), 1.77074683901981, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqf_16)]
    fn test_sumsqf_16() {
        #[rustfmt::skip]
        let x = [
            -0.2975264216819841,  0.5642827287388987, -0.7956166087428503,
            -0.1931368701566655, -0.0287212417958668,  0.3697266870899014,
             0.0791822603183984,  0.1668276194302965,  0.2048176237333448,
            -0.0617609549162579,  0.5317006403634014, -0.3964290790294236,
             0.5404940967800361,  0.1755457122664283,  0.1585602895144933,
             0.0791731424937176,
        ];
        assert_abs_diff_eq!(sumsqf(&x), 2.08885480396333, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqcf_3)]
    fn test_sumsqcf_3() {
        let x = [
            Complex32::new(-0.143606511525, -0.137405158308),
            Complex32::new(-0.155077565599, -0.128712786230),
            Complex32::new( 0.259257309730, -0.354313982924),
        ];
        assert_abs_diff_eq!(sumsqcf(&x), 0.272871791516851, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqcf_4)]
    fn test_sumsqcf_4() {
        let x = [
            Complex32::new(-0.027688113439,  0.014257850202),
            Complex32::new( 0.135913101830, -0.193497844930),
            Complex32::new(-0.184688262513, -0.018367564232),
            Complex32::new( 0.033677897260, -0.365996497668),
        ];
        assert_abs_diff_eq!(sumsqcf(&x), 0.226418463954813, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqcf_7)]
    fn test_sumsqcf_7() {
        let x = [
            Complex32::new(-0.052790293375,  0.173778162166),
            Complex32::new( 0.026113336498, -0.228399854303),
            Complex32::new( 0.060259677552, -0.064704230326),
            Complex32::new(-0.085637350173, -0.140391580928),
            Complex32::new( 0.137662823620, -0.049602389650),
            Complex32::new( 0.081078554377,  0.103320097893),
            Complex32::new(-0.140068020211, -0.028552894932),
        ];
        assert_abs_diff_eq!(sumsqcf(&x), 0.179790025178960, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqcf_8)]
    fn test_sumsqcf_8() {
        let x = [
            Complex32::new(-0.114842287937, -0.044108491804),
            Complex32::new(-0.027032488500, -0.098073597323),
            Complex32::new(-0.248865158871, -0.058431293594),
            Complex32::new( 0.152349654138,  0.011146740847),
            Complex32::new( 0.100890388238,  0.037191727983),
            Complex32::new(-0.173317554621, -0.287191794305),
            Complex32::new( 0.159045702603, -0.097006888823),
            Complex32::new(-0.048463564653, -0.123659611524),
        ];
        assert_abs_diff_eq!(sumsqcf(&x), 0.290592731534459, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqcf_15)]
    fn test_sumsqcf_15() {
        let x = [
            Complex32::new(-0.233166865552, -0.325575589001),
            Complex32::new(-0.062157314569, -0.052675113778),
            Complex32::new(-0.184924733094, -0.037448582846),
            Complex32::new(-0.019336799407, -0.146627815330),
            Complex32::new( 0.014671587594, -0.040490423681),
            Complex32::new(-0.070920638099,  0.353056761369),
            Complex32::new( 0.342121380549,  0.016365636789),
            Complex32::new( 0.407809024847, -0.067677610212),
            Complex32::new( 0.166345037028, -0.070618449000),
            Complex32::new(-0.151572833379, -0.241061531174),
            Complex32::new(-0.295395183108,  0.107933512849),
            Complex32::new( 0.214887288420,  0.158211288996),
            Complex32::new( 0.089528110626,  0.534731503540),
            Complex32::new(-0.387245894254,  0.127860010582),
            Complex32::new(-0.123711595377,  0.212526707755),
        ];
        assert_abs_diff_eq!(sumsqcf(&x), 1.44880523546855, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_sumsqcf_16)]
    fn test_sumsqcf_16() {
        let x = [
            Complex32::new(-0.065168142317,  0.069453199546),
            Complex32::new( 0.175268433034, -0.227486860237),
            Complex32::new(-0.190532229460,  0.079975095234),
            Complex32::new( 0.119309235855, -0.238114343006),
            Complex32::new( 0.125737810036,  0.045214179459),
            Complex32::new(-0.197170380197, -0.159688600627),
            Complex32::new( 0.075166226059,  0.148949236785),
            Complex32::new(-0.290229918639,  0.019293769432),
            Complex32::new(-0.145299853755, -0.083512058709),
            Complex32::new(-0.256618190275, -0.450932031739),
            Complex32::new(-0.169487127499,  0.187004249967),
            Complex32::new( 0.203885942759,  0.121347578873),
            Complex32::new(-0.176280563451, -0.304717971490),
            Complex32::new( 0.240587060249, -0.055540407201),
            Complex32::new( 0.022889112723,  0.027170265053),
            Complex32::new( 0.265769617236, -0.023686695049),
        ];
        assert_abs_diff_eq!(sumsqcf(&x), 1.07446555417927, epsilon = TOL);
    }

    #[test]
    fn test_sumsqf_empty() {
        assert_eq!(sumsqf(&[]), 0.0);
        assert_eq!(sumsqcf(&[]), 0.0);
    }

    #[test]
    fn test_sumsqf_exact() {
        for n in 1..=600usize {
            let x = vec![0.5f32; n];
            assert_eq!(sumsqf(&x), n as f32 * 0.25);

            let xc = vec![Complex32::new(0.5, 0.5); n];
            assert_eq!(sumsqcf(&xc), n as f32 * 0.5);
        }
    }

    #[test]
    fn test_sumsqcf_matches_interleaved_real() {
        for n in [1usize, 3, 8, 15, 16, 33, 64, 100] {
            let flat: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.37).sin()).collect();
            let cplx: Vec<Complex32> =
                flat.chunks(2).map(|c| Complex32::new(c[0], c[1])).collect();
            assert_eq!(sumsqcf(&cplx), sumsqf(&flat));
        }
    }
}
