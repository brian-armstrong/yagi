//! Gradient search (steepest descent) optimizer.

use crate::error::{Error, Result};
use crate::optim::OptimDirection;

/// Steepest-descent search over a vector of parameters.
///
/// Each step estimates the gradient with forward differences, normalizes it,
/// and moves along it, downhill when minimizing and uphill when maximizing.
/// The difference step starts at `1e-6` in parameter units and adapts to the
/// size of the gradient. The move length comes from a line search that starts
/// at the difference step and doubles, for up to 250 tries, until the utility
/// gets worse. It then halves once, back to the last length that helped.
///
/// A step usually improves the utility but is not guaranteed to. When the
/// first line-search probe is already worse, the step still moves by half the
/// difference step. Each step calls the utility `n + k + 3` times, where `n`
/// is the parameter count and `k` the number of line-search probes, plus
/// `n + 1` for each retry of a zero gradient.
#[doc(alias = "gradsearch")]
pub struct GradientSearch<F> {
    // vector to optimize (owned by this object)
    v: Vec<f32>,
    // utility at current position
    u: f32,

    // properties
    delta: f32, // gradient approximation step size
    alpha: f32, // line search step size

    // gradient estimate
    p: Vec<f32>,
    pnorm: f32, // L2-norm of gradient estimate

    utility: F,
    direction: OptimDirection,
}

impl<F> GradientSearch<F>
where
    F: FnMut(&[f32]) -> f32,
{
    /// Creates a search that starts from, and takes ownership of, `parameters`.
    ///
    /// The utility is not called here.
    ///
    /// # Errors
    ///
    /// Returns an error when `parameters` is empty.
    pub fn new(parameters: Vec<f32>, utility: F, direction: OptimDirection) -> Result<Self> {
        let num_parameters = parameters.len();
        if num_parameters == 0 {
            return Err(Error::Config("number of parameters must be greater than zero".into()));
        }

        // set internal properties
        // TODO : make these user-configurable properties
        let delta = 1e-6f32; // gradient approximation step size

        Ok(Self {
            v: parameters,
            u: 0.0,
            delta,
            alpha: delta, // line search step size
            p: vec![0.0; num_parameters],
            pnorm: 0.0,
            utility,
            direction,
        })
    }

    /// Returns the current parameter vector.
    pub fn parameters(&self) -> &[f32] {
        &self.v
    }

    /// Returns the utility at the current vector, as computed by the last
    /// successful [`Self::step`].
    ///
    /// Returns `0.0` before the first successful step, whatever the utility at
    /// the starting vector.
    pub fn utility(&self) -> f32 {
        self.u
    }

    /// Takes one step and returns the utility at the new vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoConvergence`] when the estimated gradient is zero on
    /// 20 tries, with the difference step growing tenfold after each. This
    /// happens for a utility that is flat at the current vector. The vector
    /// and [`Self::utility`] are unchanged on error, but the
    /// difference step keeps its larger value.
    pub fn step(&mut self) -> Result<f32> {
        // ensure norm(p) > 0, otherwise increase delta
        let n = 20;
        let mut i = 0;
        while i < n {
            // compute gradient
            self.compute_gradient();

            // normalize gradient vector
            self.pnorm = Self::norm(&mut self.p);

            if self.pnorm > 0.0 {
                // try to keep delta about 1e-4 * pnorm
                if 1e-4 * self.pnorm < self.delta {
                    self.delta *= 0.90;
                } else if 1e-5 * self.pnorm > self.delta {
                    self.delta *= 1.10;
                }

                break;
            } else {
                // step size is too small to approximate gradient
                self.delta *= 10.0;
            }
            i += 1;
        }

        if i == n {
            return Err(Error::NoConvergence("gradsearch_step(), function ill-conditioned".into()));
        }

        // run line search
        self.alpha = self.linesearch();

        // step in the negative direction of the gradient
        let dir = if self.direction == OptimDirection::Minimize { 1.0 } else { -1.0 };
        for i in 0..self.v.len() {
            self.v[i] -= dir * self.alpha * self.p[i];
        }

        // evaluate utility at current position
        self.u = (self.utility)(&self.v);

        Ok(self.u)
    }

    /// Takes steps until `max_iterations` steps have run or the utility
    /// passes `target_utility`, and returns the last utility.
    ///
    /// Passing the target means going strictly below it when minimizing and
    /// strictly above it when maximizing. When `max_iterations` is zero, no
    /// step runs, but the utility is evaluated once at the unchanged parameters
    /// because construction does not evaluate it.
    ///
    /// # Errors
    ///
    /// Returns the first error from [`Self::step`]. Steps taken before it are
    /// kept.
    pub fn run(&mut self, max_iterations: usize, target_utility: f32) -> Result<f32> {
        if max_iterations == 0 {
            self.u = (self.utility)(&self.v);
            return Ok(self.u);
        }

        let mut num_iterations = 0;
        let mut u;
        loop {
            // increment number of iterations
            num_iterations += 1;

            // step gradient search algorithm
            u = self.step()?;

            // check exit criteria (any one of the following)
            //  * maximum number of iterations met
            //  * maximize utility and target exceeded
            //  * minimize utility and target exceeded
            if num_iterations >= max_iterations
                || (self.direction == OptimDirection::Minimize && u < target_utility)
                || (self.direction == OptimDirection::Maximize && u > target_utility)
            {
                break;
            }
        }

        Ok(u)
    }

    //
    // internal (generic functions)
    //

    /// Compute the gradient of the utility at the current operating point,
    /// storing the estimate into `self.p`.
    fn compute_gradient(&mut self) {
        let n = self.v.len();
        let delta = self.delta;

        // evaluate function at current operating point
        let u0 = (self.utility)(&self.v);

        // operating point for evaluation
        let mut x_prime = self.v.clone();

        for i in 0..n {
            // copy operating point
            x_prime.copy_from_slice(&self.v);

            // increment test vector by delta along dimension 'i'
            x_prime[i] += delta;

            // evaluate new utility
            let u_prime = (self.utility)(&x_prime);

            // compute gradient estimate
            self.p[i] = (u_prime - u0) / delta;
        }
    }

    /// Execute line search; loosely solve
    ///
    /// ```text
    /// min|max phi(alpha) := f(v - alpha*p)
    /// ```
    ///
    /// and return the best guess at `alpha` that achieves this.
    fn linesearch(&mut self) -> f32 {
        let n = self.v.len();

        // evaluate utility at base point
        let mut u0 = (self.utility)(&self.v);

        // initialize step size estimate
        let mut alpha = self.delta;

        // step direction
        let dir = if self.direction == OptimDirection::Minimize { 1.0 } else { -1.0 };

        // test vector
        let mut x_prime = vec![0.0f32; n];

        // run line search
        let mut num_iterations = 0;
        let max_iterations = 250;
        let gamma = 2.00f32;
        loop {
            // increment iteration counter
            num_iterations += 1;

            // update evaluation point
            for i in 0..n {
                x_prime[i] = self.v[i] - dir * alpha * self.p[i];
            }

            // compute utility for line search step
            let uls = (self.utility)(&x_prime);

            // check exit criteria
            if (self.direction == OptimDirection::Minimize && uls > u0)
                || (self.direction == OptimDirection::Maximize && uls < u0)
            {
                // compared this utility to previous; went too far.
                // backtrack step size and stop line search
                alpha *= 1.0 / gamma;
                break;
            } else if num_iterations >= max_iterations {
                // maximum number of iterations met: stop line search
                break;
            } else {
                // save new best estimate, increase step size, and continue
                u0 = uls;
                alpha *= gamma;
            }
        }

        alpha
    }

    /// Normalize vector in place, returning its pre-normalization L2-norm.
    fn norm(v: &mut [f32]) -> f32 {
        // compute l2-norm
        let vnorm = crate::vector::norm(v);

        // scale values (avoiding division by zero)
        let scale = if vnorm == 0.0 { 0.0 } else { 1.0 / vnorm };
        for vi in v.iter_mut() {
            *vi *= scale;
        }

        // return normalization
        vnorm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::rosenbrock;
    use approx::assert_abs_diff_eq;
    use std::cell::Cell;
    use test_macro::autotest_annotate;

    // Find minimum of Rosenbrock function, should be [1 1 1 ...]
    #[test]
    #[autotest_annotate(autotest_gradsearch_rosenbrock)]
    fn gradsearch_rosenbrock() {
        let tol = 1e-2f32; // error tolerance
        let num_parameters = 6; // dimensionality of search (minimum 2)
        let num_iterations = 4000; // number of iterations to run

        // initialize vector for optimization
        let v_opt = vec![0.0f32; num_parameters];

        // create gradsearch object
        let mut gs = GradientSearch::new(v_opt, |v| rosenbrock(v), OptimDirection::Minimize).unwrap();

        // execute search one iteration at a time
        for _ in 0..num_iterations {
            gs.step().unwrap();
        }

        // test results, optimum at [1, 1, 1, ... 1];
        for &vi in gs.parameters() {
            assert_abs_diff_eq!(vi, 1.0, epsilon = tol);
        }

        // test value of utility (should be nearly 0)
        assert_abs_diff_eq!(rosenbrock(gs.parameters()), 0.0, epsilon = tol);
    }

    // Find maximum of: exp{ -sum{ (v[i]-1)^2/sigma_i^2 } }, should be [1 1 1 ...]
    fn utility_max_autotest(v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }

        let mut t = 0.0f32;
        let mut sigma = 1.0f32;
        for &vi in v {
            t += (vi - 1.0) * (vi - 1.0) / (sigma * sigma);

            // increase variance along this dimension
            sigma *= 1.5;
        }

        (-t).exp()
    }

    #[test]
    #[autotest_annotate(autotest_gradsearch_maxutility)]
    fn gradsearch_maxutility() {
        let tol = 1e-2f32; // error tolerance
        let num_parameters = 6; // dimensionality of search (minimum 2)
        let num_iterations = 4000; // number of iterations to run

        // initialize vector for optimization
        let v_opt = vec![0.0f32; num_parameters];

        // create gradsearch object
        let mut gs = GradientSearch::new(v_opt, utility_max_autotest, OptimDirection::Maximize).unwrap();

        // execute search one iteration at a time
        for _ in 0..num_iterations {
            gs.step().unwrap();
        }

        // test results, optimum at [1, 1, 1, ... 1];
        for &vi in gs.parameters() {
            assert_abs_diff_eq!(vi, 1.0, epsilon = tol);
        }

        // test value of utility (should be nearly 1)
        assert_abs_diff_eq!(utility_max_autotest(gs.parameters()), 1.0, epsilon = tol);
    }

    #[test]
    fn gradient_search_rejects_empty_parameters() {
        let search =
            GradientSearch::new(Vec::new(), |_| panic!("utility must not be called"), OptimDirection::Minimize);
        assert!(matches!(search, Err(Error::Config(_))));
    }

    #[test]
    fn run_zero_iterations_evaluates_without_stepping() {
        let calls = Cell::new(0);
        let mut search = GradientSearch::new(
            vec![2.0],
            |parameters| {
                calls.set(calls.get() + 1);
                parameters[0].powi(2)
            },
            OptimDirection::Minimize,
        )
        .unwrap();

        assert_eq!(calls.get(), 0);
        assert_eq!(search.run(0, -1.0).unwrap(), 4.0);
        assert_eq!(search.parameters(), &[2.0]);
        assert_eq!(search.utility(), 4.0);
        assert_eq!(calls.get(), 1);
    }
}
