use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimDirection {
    Minimize,
    Maximize,
}

#[derive(Debug)]
pub struct Qs1dSearch<F> {
    // values
    vn: f32,
    va: f32,
    v0: f32,
    vb: f32,
    vp: f32,

    // utilities
    un: f32,
    ua: f32,
    u0: f32,
    ub: f32,
    up: f32,

    // values initialized?
    init: bool,

    // utility closure
    utility: F,
    direction: OptimDirection,
    num_steps: usize,
}

impl<F> Qs1dSearch<F>
where
    F: FnMut(f32) -> f32,
{
    pub fn new(utility: F, direction: OptimDirection) -> Self {
        Self {
            vn: 0.0,
            va: 0.0,
            v0: 0.0,
            vb: 0.0,
            vp: 0.0,
            un: 0.0,
            ua: 0.0,
            u0: 0.0,
            ub: 0.0,
            up: 0.0,
            init: false,
            utility,
            direction,
            num_steps: 0,
        }
    }

    pub fn reset(&mut self) {
        self.init = false;
        self.num_steps = 0;
        self.vn = 0.0;
        self.un = 0.0;
        self.va = 0.0;
        self.ua = 0.0;
        self.v0 = 0.0;
        self.u0 = 0.0;
        self.vb = 0.0;
        self.ub = 0.0;
        self.vp = 0.0;
        self.up = 0.0;
    }

    pub fn init(&mut self, v: f32) -> Result<()> {
        // specify initial step size
        let step = 1e-16;

        // try positive direction
        if self.init_direction(v, step).is_ok() {
            return Ok(());
        }

        // try negative direction
        if self.init_direction(v, -step).is_ok() {
            return Ok(());
        }

        // check edge case where v is exactly the optimum
        self.vn = v - step;
        self.un = (self.utility)(self.vn);
        self.v0 = v;
        self.u0 = (self.utility)(self.v0);
        self.vp = v + step;
        self.up = (self.utility)(self.vp);
        if (self.direction == OptimDirection::Minimize && self.u0 < self.un && self.u0 < self.up) ||
           (self.direction == OptimDirection::Maximize && self.u0 > self.un && self.u0 > self.up)
        {
            self.init = true;
            return Ok(());
        }

        Err(Error::NoConvergence("Failed to initialize search".into()))
    }

    fn init_direction(&mut self, v_init: f32, step: f32) -> Result<()> {
        let mut vn;
        let mut v0 = v_init;
        let mut vp = v_init + step * 0.5;
        let mut un;
        let mut u0 = (self.utility)(v0);
        let mut up = (self.utility)(vp);
        let mut step = step;

        for _ in 0..180 {
            // move values down
            vn = v0;
            v0 = vp;
            un = u0;
            u0 = up;

            vp = v0 + step;
            up = (self.utility)(vp);

            if (self.direction == OptimDirection::Minimize && u0 < un && u0 < up) ||
               (self.direction == OptimDirection::Maximize && u0 > un && u0 > up)
            {
                // skipped over optimum; set internal bounds and return
                let swap = step < 0.0;
                self.vn = if swap { vp } else { vn };
                self.v0 = v0;
                self.vp = if swap { vn } else { vp };

                self.un = if swap { up } else { un };
                self.u0 = u0;
                self.up = if swap { un } else { up };
                self.init = true;
                return Ok(());
            } else if (self.direction == OptimDirection::Minimize && u0 >= un && up > u0) ||
                      (self.direction == OptimDirection::Maximize && u0 <= un && up < u0)
            {
                // clearly moving in wrong direction: exit early
                break;
            }
            step *= 1.5; // increase step size
        }

        Err(Error::NoConvergence("Failed to initialize search direction".into()))
    }

    pub fn init_bounds(&mut self, vn: f32, vp: f32) -> Result<()> {
        // set bounds appropriately
        self.vn = vn.min(vp);
        self.vp = vn.max(vp);
        self.v0 = 0.5 * (vn + vp);

        // evaluate utility
        self.un = (self.utility)(self.vn);
        self.u0 = (self.utility)(self.v0);
        self.up = (self.utility)(self.vp);

        self.init = true;

        Ok(())
    }

    pub fn step(&mut self) -> Result<()> {
        if !self.init {
            return Err(Error::Config("Object has not been properly initialized".into()));
        }

        // compute new candidate points
        self.va = 0.5 * (self.vn + self.v0);
        self.vb = 0.5 * (self.v0 + self.vp);

        // evaluate utility
        self.ua = (self.utility)(self.va);
        self.ub = (self.utility)(self.vb);

        // [ (vn)  va  (v0)  vb  (vp) ]
        // optimum should be va, v0, or vb
        let min_va_is_optimum = self.direction == OptimDirection::Minimize && self.ua < self.u0 && self.ua < self.ub;
        let max_va_is_optimum = self.direction == OptimDirection::Maximize && self.ua > self.u0 && self.ua > self.ub;
        let min_v0_is_optimum = self.direction == OptimDirection::Minimize && self.u0 < self.ua && self.u0 < self.ub;
        let max_v0_is_optimum = self.direction == OptimDirection::Maximize && self.u0 > self.ua && self.u0 > self.ub;

        let va_is_optimum = min_va_is_optimum || max_va_is_optimum;
        let v0_is_optimum = min_v0_is_optimum || max_v0_is_optimum;

        if va_is_optimum {
            // va is optimum
            self.vp = self.v0;
            self.up = self.u0;
            self.v0 = self.va;
            self.u0 = self.ua;
        } else if v0_is_optimum {
            // v0 is optimum
            self.vn = self.va;
            self.un = self.ua;
            self.vp = self.vb;
            self.up = self.ub;
        } else {
            // vb is optimum
            self.vn = self.v0;
            self.un = self.u0;
            self.v0 = self.vb;
            self.u0 = self.ub;
        }

        self.num_steps += 1;
        Ok(())
    }

    pub fn execute(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_num_steps(&self) -> usize {
        self.num_steps
    }

    pub fn get_opt_v(&self) -> f32 {
        self.v0
    }

    pub fn get_opt_u(&self) -> f32 {
        self.u0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;

    fn qs1dsearch_umin(v: f32, v_opt: f32) -> f32 {
        let v = v - v_opt;
        v.tanh().powi(2)
    }

    fn qs1dsearch_umax(v: f32, v_opt: f32) -> f32 {
        -qs1dsearch_umin(v, v_opt)
    }

    // test initialization on single value
    fn test_qs1dsearch(
        utility: fn(f32, f32) -> f32,
        v_opt: f32,
        v_lo: f32,
        v_hi: f32,
        bounds: bool,
        direction: OptimDirection,
    ) {
        // create qs1dsearch object and initialize
        let mut q = Qs1dSearch::new(|v| utility(v, v_opt), direction);
        if bounds {
            q.init_bounds(v_lo, v_hi).unwrap();
        } else {
            q.init(v_lo).unwrap();
        }

        // run search
        for _ in 0..32 {
            q.step().unwrap();
        }

        // check result
        assert_abs_diff_eq!(q.get_opt_v(), v_opt, epsilon = 1e-3);
        assert_abs_diff_eq!(q.get_opt_u(), utility(v_opt, v_opt), epsilon = 1e-3);
    }

    // unbounded:
    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_01)]
    fn qs1dsearch_min_01() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, -40.0, 0.0, false, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_02)]
    fn qs1dsearch_min_02() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, -20.0, 0.0, false, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_03)]
    fn qs1dsearch_min_03() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, -4.0, 0.0, false, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_05)]
    fn qs1dsearch_min_05() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, 0.0, 0.0, false, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_06)]
    fn qs1dsearch_min_06() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, 4.0, 0.0, false, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_07)]
    fn qs1dsearch_min_07() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, 20.0, 0.0, false, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_08)]
    fn qs1dsearch_min_08() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, 40.0, 0.0, false, OptimDirection::Minimize);
    }

    // bounded:
    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_10)]
    fn qs1dsearch_min_10() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, -30.0, 15.0, true, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_11)]
    fn qs1dsearch_min_11() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, -20.0, 15.0, true, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_12)]
    fn qs1dsearch_min_12() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, -10.0, 15.0, true, OptimDirection::Minimize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_min_13)]
    fn qs1dsearch_min_13() {
        test_qs1dsearch(qs1dsearch_umin, 0.0, -0.1, 15.0, true, OptimDirection::Minimize);
    }

    // repeat to maximize

    // unbounded:
    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_01)]
    fn qs1dsearch_max_01() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, -40.0, 0.0, false, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_02)]
    fn qs1dsearch_max_02() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, -20.0, 0.0, false, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_03)]
    fn qs1dsearch_max_03() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, -4.0, 0.0, false, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_05)]
    fn qs1dsearch_max_05() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, 0.0, 0.0, false, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_06)]
    fn qs1dsearch_max_06() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, 4.0, 0.0, false, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_07)]
    fn qs1dsearch_max_07() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, 20.0, 0.0, false, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_08)]
    fn qs1dsearch_max_08() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, 40.0, 0.0, false, OptimDirection::Maximize);
    }

    // bounded:
    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_10)]
    fn qs1dsearch_max_10() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, -30.0, 15.0, true, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_11)]
    fn qs1dsearch_max_11() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, -20.0, 15.0, true, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_12)]
    fn qs1dsearch_max_12() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, -10.0, 15.0, true, OptimDirection::Maximize);
    }

    #[test]
    #[autotest_annotate(autotest_qs1dsearch_max_13)]
    fn qs1dsearch_max_13() {
        test_qs1dsearch(qs1dsearch_umax, 0.0, -0.1, 15.0, true, OptimDirection::Maximize);
    }

    // test configuration
    #[test]
    #[autotest_annotate(autotest_qs1dsearch_config)]
    fn test_qs1dsearch_config() {
        // create proper object and test configurations
        let v_opt = 0.0f32;
        let mut q = Qs1dSearch::new(|v| qs1dsearch_umax(v, v_opt), OptimDirection::Maximize);
        // assert!(q.print().is_ok());

        // test configurations
        assert!(q.step().is_err()); // object not yet initialized
        q.init(20.0).unwrap();

        assert!(q.execute().is_ok());

        // run a few steps
        assert_eq!(0, q.get_num_steps());
        q.step().unwrap();
        q.step().unwrap();
        q.step().unwrap();
        assert_eq!(3, q.get_num_steps());

        // No need to explicitly destroy objects in Rust
    }
}
