use crate::error::{Error, Result};

/// Whether a search looks for the smallest or the largest utility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimDirection {
    /// Prefer smaller utility values.
    Minimize,
    /// Prefer larger utility values.
    Maximize,
}

/// Search for an extremum of a function of one `f32`, by repeatedly
/// narrowing an interval around it.
///
/// The search holds three points: the two ends of an interval and a current
/// estimate inside it. Construct the search from a starting position with
/// [`Self::new`] or from explicit bounds with [`Self::new_with_bounds`], then
/// call [`Self::step`] as often as needed. The interval can later be replaced
/// with [`Self::init`] or [`Self::init_bounds`]. Each step evaluates the
/// midpoints on either side of the estimate and keeps the best of the three
/// inner points, with its two neighbors as the new ends. After
/// [`Self::init_bounds`], every step halves the interval.
///
/// When the interval brackets a single extremum, the estimate converges on
/// it. There is no stopping test, so the caller chooses the number of steps.
#[derive(Debug)]
#[doc(alias = "Qs1dSearch")]
pub struct QuadSectionSearch<F> {
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

impl<F> QuadSectionSearch<F>
where
    F: FnMut(f32) -> f32,
{
    /// Creates a search by looking for an extremum outward from `position`.
    ///
    /// See [`Self::init`] for the bracketing behavior and possible errors.
    pub fn new(utility: F, direction: OptimDirection, position: f32) -> Result<Self> {
        let mut search = Self::uninitialized(utility, direction);
        search.init(position)?;
        Ok(search)
    }

    /// Creates a search from two explicit interval bounds.
    ///
    /// See [`Self::init_bounds`] for the bound requirements and evaluation
    /// behavior. The bounds need not already bracket an extremum.
    pub fn new_with_bounds(utility: F, direction: OptimDirection, first_bound: f32, second_bound: f32) -> Result<Self> {
        let mut search = Self::uninitialized(utility, direction);
        search.init_bounds(first_bound, second_bound)?;
        Ok(search)
    }

    fn uninitialized(utility: F, direction: OptimDirection) -> Self {
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

    /// Clears the interval, the estimate and the step count.
    ///
    /// The utility and direction are kept. Initialize the search again before
    /// the next [`Self::step`].
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

    /// Sets the interval by searching outward from `position` for an extremum.
    ///
    /// The search moves away from `position` in growing steps, first upward
    /// and then downward, until the utility turns. Steps start near `1e-16`
    /// and grow by a factor of 1.5, for up to 180 steps in each direction. A
    /// direction is abandoned as soon as the utility moves the wrong way.
    ///
    /// If `position` is already an extremum, the search probes both sides at
    /// increasing representable distances until it finds a bracket. The step
    /// count is not reset.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when `position` is non-finite, or
    /// [`Error::NoConvergence`] when no bracket is found. On error, the search
    /// interval, estimate, and step count are unchanged. The utility may have
    /// been called during the search.
    pub fn init(&mut self, position: f32) -> Result<()> {
        if !position.is_finite() {
            return Err(Error::Config("initial position must be finite".into()));
        }

        // specify initial step size
        let step = 1e-16;

        // try positive direction
        if self.init_direction(position, step).is_ok() {
            return Ok(());
        }

        // try negative direction
        if self.init_direction(position, -step).is_ok() {
            return Ok(());
        }

        // check whether position is already the optimum. start at a relative
        // f32 spacing and grow until the utility distinguishes both sides.
        let center_utility = (self.utility)(position);
        let mut probe_step = f32::EPSILON * position.abs().max(1.0);
        for _ in 0..32 {
            let left = position - probe_step;
            let right = position + probe_step;
            if !left.is_finite() || !right.is_finite() {
                break;
            }
            if left < position && right > position {
                let left_utility = (self.utility)(left);
                let right_utility = (self.utility)(right);
                if (self.direction == OptimDirection::Minimize
                    && center_utility < left_utility
                    && center_utility < right_utility)
                    || (self.direction == OptimDirection::Maximize
                        && center_utility > left_utility
                        && center_utility > right_utility)
                {
                    self.vn = left;
                    self.un = left_utility;
                    self.v0 = position;
                    self.u0 = center_utility;
                    self.vp = right;
                    self.up = right_utility;
                    self.init = true;
                    return Ok(());
                }
            }
            probe_step *= 2.0;
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

            if (self.direction == OptimDirection::Minimize && u0 < un && u0 < up)
                || (self.direction == OptimDirection::Maximize && u0 > un && u0 > up)
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
            } else if (self.direction == OptimDirection::Minimize && u0 >= un && up > u0)
                || (self.direction == OptimDirection::Maximize && u0 <= un && up < u0)
            {
                // clearly moving in wrong direction: exit early
                break;
            }
            step *= 1.5; // increase step size
        }

        Err(Error::NoConvergence("Failed to initialize search direction".into()))
    }

    /// Sets the interval between `first_bound` and `second_bound`, with the
    /// midpoint as the estimate.
    ///
    /// The ends may be given in either order. Whether they bracket an
    /// extremum is not checked. The utility is called at both ends and the
    /// midpoint. The step count is not reset.
    ///
    /// # Errors
    ///
    /// Returns an error if either bound is non-finite or the bounds are equal.
    /// The search is unchanged and the utility is not called on error.
    pub fn init_bounds(&mut self, first_bound: f32, second_bound: f32) -> Result<()> {
        if !first_bound.is_finite() || !second_bound.is_finite() || first_bound == second_bound {
            return Err(Error::Config("bounds must be finite and distinct".into()));
        }

        // set bounds appropriately
        self.vn = first_bound.min(second_bound);
        self.vp = first_bound.max(second_bound);
        // compute in f64 so finite f32 bounds cannot overflow the sum
        self.v0 = ((first_bound as f64 + second_bound as f64) * 0.5) as f32;

        // evaluate utility
        self.un = (self.utility)(self.vn);
        self.u0 = (self.utility)(self.v0);
        self.up = (self.utility)(self.vp);

        self.init = true;

        Ok(())
    }

    /// Narrows the interval once, calling the utility twice.
    ///
    /// # Errors
    ///
    /// Returns an error when the search has not been initialized. The utility
    /// is not called and the search is unchanged in that case.
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

    /// Returns the number of steps taken since creation or the last
    /// [`Self::reset`].
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Returns the current estimate of the extremum's position.
    ///
    /// Returns `0.0` after [`Self::reset`], until the search is initialized
    /// again.
    pub fn estimate(&self) -> f32 {
        self.v0
    }

    /// Returns the utility at the current estimate.
    ///
    /// Returns `0.0` after [`Self::reset`], until the search is initialized
    /// again.
    pub fn utility(&self) -> f32 {
        self.u0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::cell::Cell;
    use test_macro::autotest_annotate;
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
        // create qs1dsearch object with an initial interval
        let utility_at_opt = |v| utility(v, v_opt);
        let mut q = if bounds {
            QuadSectionSearch::new_with_bounds(utility_at_opt, direction, v_lo, v_hi).unwrap()
        } else {
            QuadSectionSearch::new(utility_at_opt, direction, v_lo).unwrap()
        };

        // run search
        for _ in 0..32 {
            q.step().unwrap();
        }

        // check result
        assert_abs_diff_eq!(q.estimate(), v_opt, epsilon = 1e-3);
        assert_abs_diff_eq!(q.utility(), utility(v_opt, v_opt), epsilon = 1e-3);
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
        let mut q = QuadSectionSearch::new(|v| qs1dsearch_umax(v, v_opt), OptimDirection::Maximize, 20.0).unwrap();
        // assert!(q.print().is_ok());

        q.reset();
        assert!(q.step().is_err());
        q.init(20.0).unwrap();

        // run a few steps
        assert_eq!(0, q.num_steps());
        q.step().unwrap();
        q.step().unwrap();
        q.step().unwrap();
        assert_eq!(3, q.num_steps());

        // No need to explicitly destroy objects in Rust
    }

    #[test]
    fn init_brackets_exact_nonzero_extrema() {
        let minimum =
            QuadSectionSearch::new(|position| (position - 3.0).powi(2) + 1.0, OptimDirection::Minimize, 3.0).unwrap();
        assert_eq!(minimum.estimate(), 3.0);
        assert_eq!(minimum.utility(), 1.0);
        assert_eq!(minimum.num_steps(), 0);

        let maximum =
            QuadSectionSearch::new(|position| 5.0 - (position + 2.0).powi(2), OptimDirection::Maximize, -2.0).unwrap();
        assert_eq!(maximum.estimate(), -2.0);
        assert_eq!(maximum.utility(), 5.0);
    }

    #[test]
    fn failed_init_preserves_previous_search() {
        let mut search = QuadSectionSearch::new_with_bounds(|_| 1.0, OptimDirection::Minimize, -1.0, 1.0).unwrap();
        let mut control = QuadSectionSearch::new_with_bounds(|_| 1.0, OptimDirection::Minimize, -1.0, 1.0).unwrap();

        assert!(matches!(search.init(10.0), Err(Error::NoConvergence(_))));
        assert_eq!(search.estimate(), control.estimate());
        assert_eq!(search.utility(), control.utility());
        assert_eq!(search.num_steps(), control.num_steps());

        for _ in 0..3 {
            search.step().unwrap();
            control.step().unwrap();
            assert_eq!(search.estimate(), control.estimate());
            assert_eq!(search.utility(), control.utility());
            assert_eq!(search.num_steps(), control.num_steps());
        }
    }

    #[test]
    fn new_requires_a_bracket() {
        assert!(matches!(
            QuadSectionSearch::new(|_| 1.0, OptimDirection::Minimize, 10.0),
            Err(Error::NoConvergence(_))
        ));
    }

    #[test]
    fn init_rejects_nonfinite_position_before_calling_utility() {
        let calls = Cell::new(0);
        let mut search = QuadSectionSearch::new_with_bounds(
            |_| {
                calls.set(calls.get() + 1);
                0.0
            },
            OptimDirection::Minimize,
            -1.0,
            1.0,
        )
        .unwrap();
        calls.set(0);
        for position in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(search.init(position), Err(Error::Config(_))));
        }
        assert_eq!(calls.get(), 0);
        search.step().unwrap();

        calls.set(0);
        for position in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                QuadSectionSearch::new(
                    |_| {
                        calls.set(calls.get() + 1);
                        0.0
                    },
                    OptimDirection::Minimize,
                    position,
                ),
                Err(Error::Config(_))
            ));
        }
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn init_bounds_midpoint_is_finite_for_large_finite_bounds() {
        let search =
            QuadSectionSearch::new_with_bounds(|position| position, OptimDirection::Minimize, f32::MAX / 2.0, f32::MAX)
                .unwrap();
        assert!(search.estimate().is_finite());
        assert!(search.estimate() > f32::MAX / 2.0);
        assert!(search.estimate() < f32::MAX);
    }
}
