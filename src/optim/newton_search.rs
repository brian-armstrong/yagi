//! Finite-difference Newton search optimizer.

use crate::error::{Error, Result};
use crate::matrix::{matrix_inv, matrix_mul_vec, MatrixMut};
use crate::optim::common::threshold_switch;
use crate::optim::OptimDirection;

/// Newton-method search over a vector of parameters, with finite-difference
/// derivatives.
///
/// Each step estimates the full Hessian afresh and inverts it rather than
/// updating a quasi-Newton approximation. The gradient uses forward
/// differences with a step of `1e-6` and the Hessian central
/// differences with a step of `0.01`, both in parameter units. Each step calls
/// the utility `2n^2 + 2n + 1` times for `n` parameters, and storage grows as
/// `n^2`.
///
/// Each step scales the Newton step, starting at `1.0`. The scale shrinks
/// after a step that worsens the utility and grows slightly otherwise. There
/// is no line search, so a step can still make the utility worse.
/// For a first-order method, see [`GradientSearch`].
///
/// [`GradientSearch`]: crate::optim::GradientSearch
#[doc(alias = "qnsearch")]
pub struct NewtonSearch<F> {
    // vector to optimize (owned by this object)
    v: Vec<f32>,
    num_parameters: usize,

    gamma: f32,        // nominal stepsize
    delta: f32,        // differential used to estimate the derivative
    gamma_hat: f32,    // adaptive step scale
    v_prime: Vec<f32>, // temporary vector array
    dv: Vec<f32>,      // parameter step vector

    h: Vec<f32>, // Hessian matrix [n x n]

    p: Vec<f32>,         // search direction
    gradient: Vec<f32>,  // gradient approximation
    gradient0: Vec<f32>, // gradient approximation (previous step)

    // External utility function.
    get_utility: F,
    utility: f32,   // current utility
    minimize: bool, // minimize/maximize utility (search direction)
}

impl<F> NewtonSearch<F>
where
    F: FnMut(&[f32]) -> f32,
{
    /// Creates a search that starts from, and takes ownership of, `parameters`.
    ///
    /// The utility is called at `parameters`, twice.
    ///
    /// # Errors
    ///
    /// Returns an error when `parameters` is empty, or when the Hessian size
    /// `parameters.len() * parameters.len()` overflows `usize`.
    pub fn new(parameters: Vec<f32>, mut utility: F, direction: OptimDirection) -> Result<Self> {
        let num_parameters = parameters.len();
        if num_parameters == 0 {
            return Err(Error::Config("number of parameters must be greater than zero".into()));
        }
        let hessian_len = num_parameters
            .checked_mul(num_parameters)
            .ok_or_else(|| Error::Config("Hessian dimensions overflow usize".into()))?;
        let gamma = 1.0f32;
        let u = utility(&parameters);

        let mut q = Self {
            v_prime: vec![0.0; num_parameters],
            dv: vec![0.0; num_parameters],
            h: vec![0.0; hessian_len],
            p: vec![0.0; num_parameters],
            gradient: vec![0.0; num_parameters],
            gradient0: vec![0.0; num_parameters],
            v: parameters,
            num_parameters,
            delta: 1e-6,
            gamma,
            gamma_hat: gamma,
            get_utility: utility,
            utility: u,
            minimize: direction == OptimDirection::Minimize,
        };

        q.reset();
        Ok(q)
    }

    /// Returns the current parameter vector.
    pub fn parameters(&self) -> &[f32] {
        &self.v
    }

    /// Returns the utility at the current vector.
    pub fn utility(&self) -> f32 {
        self.utility
    }

    /// Restores the starting step scale and recomputes the utility at the
    /// current vector.
    ///
    /// The vector and the last search direction are kept.
    pub fn reset(&mut self) {
        self.gamma_hat = self.gamma;

        self.utility = (self.get_utility)(&self.v);
    }

    /// Takes one step and updates the utility.
    ///
    /// When the Hessian estimate cannot be inverted, the step reuses the
    /// previous search direction, which is zero before the first successful
    /// inversion. This is not reported, and a step that worsens the utility is
    /// kept.
    pub fn step(&mut self) {
        let n = self.num_parameters;

        // compute gradient vector
        self.compute_gradient();

        // TODO : perform line search to find optimal gamma

        // compute search direction: p = H^{-1} gradient
        self.compute_hessian();
        // if the Hessian is singular the inverse fails (leave p from the prior step)
        let mut h = MatrixMut::new(&mut self.h, n, n).expect("Hessian storage matches the search dimension");
        if matrix_inv(&mut h).is_ok() {
            let h = h.as_ref();
            let _ = matrix_mul_vec(h, &self.gradient, &mut self.p);
        }

        // compute step vector
        for i in 0..n {
            self.dv[i] = -self.gamma_hat * self.p[i];
        }

        // apply change
        for i in 0..n {
            self.v[i] += self.dv[i];
        }

        // store previous gradient
        self.gradient0.copy_from_slice(&self.gradient);

        // update utility
        let u_prime = (self.get_utility)(&self.v);

        if threshold_switch(u_prime, self.utility, self.minimize) {
            self.gamma_hat *= 0.99;
        } else {
            self.gamma_hat *= 1.001;
        }

        self.utility = u_prime;
    }

    /// Takes steps until `max_iterations` steps have run or the utility
    /// reaches `target_utility`, and returns the last utility.
    ///
    /// Reaching the target means getting to it or below when minimizing, and
    /// to it or above when maximizing. When `max_iterations` is zero, no
    /// step or utility evaluation runs. Each step here calls the utility once
    /// more than [`Self::step`] alone.
    pub fn run(&mut self, max_iterations: usize, target_utility: f32) -> f32 {
        if max_iterations == 0 {
            return self.utility;
        }

        let mut i = 0;
        loop {
            i += 1;
            self.step();
            self.utility = (self.get_utility)(&self.v);

            if !(threshold_switch(self.utility, target_utility, self.minimize) && i < max_iterations) {
                break;
            }
        }

        self.utility
    }

    //
    // internal
    //

    // compute gradient
    fn compute_gradient(&mut self) {
        // reset v_prime
        self.v_prime.copy_from_slice(&self.v);

        for i in 0..self.num_parameters {
            self.v_prime[i] += self.delta;
            let f_prime = (self.get_utility)(&self.v_prime);
            self.v_prime[i] -= self.delta;
            self.gradient[i] = (f_prime - self.utility) / self.delta;
        }
    }

    // compute Hessian
    fn compute_hessian(&mut self) {
        let n = self.num_parameters;
        let delta = 1e-2f32;

        let v = &self.v;
        let v_prime = &mut self.v_prime;
        let get_utility = &mut self.get_utility;
        let mut h = MatrixMut::new(&mut self.h, n, n).expect("Hessian storage matches the search dimension");

        // reset v_prime
        v_prime.copy_from_slice(v);

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    v_prime[i] = v[i] - delta;
                    let f0 = get_utility(v_prime);

                    v_prime[i] = v[i];
                    let f1 = get_utility(v_prime);

                    v_prime[i] = v[i] + delta;
                    let f2 = get_utility(v_prime);

                    let m0 = (f1 - f0) / delta;
                    let m1 = (f2 - f1) / delta;
                    h.row_mut(i)[j] = (m1 - m0) / delta;
                    v_prime[i] = v[i];
                } else {
                    // 0 0
                    v_prime[i] = v[i] - delta;
                    v_prime[j] = v[j] - delta;
                    let f00 = get_utility(v_prime);

                    // 0 1
                    v_prime[i] = v[i] - delta;
                    v_prime[j] = v[j] + delta;
                    let f01 = get_utility(v_prime);

                    // 1 0
                    v_prime[i] = v[i] + delta;
                    v_prime[j] = v[j] - delta;
                    let f10 = get_utility(v_prime);

                    // 1 1
                    v_prime[i] = v[i] + delta;
                    v_prime[j] = v[j] + delta;
                    let f11 = get_utility(v_prime);

                    // compute second partial derivative
                    let m0 = (f01 - f00) / (2.0 * delta);
                    let m1 = (f11 - f10) / (2.0 * delta);
                    let h_ij = (m1 - m0) / (2.0 * delta);
                    h.row_mut(i)[j] = h_ij;
                    h.row_mut(j)[i] = h_ij;
                    v_prime[i] = v[i];
                    v_prime[j] = v[j];
                }
            }
        }
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
    #[autotest_annotate(autotest_qnsearch_rosenbrock)]
    fn qnsearch_rosenbrock() {
        let tol = 1e-2f32; // error tolerance
        let num_parameters = 6; // dimensionality of search (minimum 2)
        let num_iterations = 4000; // number of iterations to run

        // initialize vector for optimization
        let v_opt = vec![0.0f32; num_parameters];

        // create Newton search object
        let mut gs = NewtonSearch::new(v_opt, |v| rosenbrock(v), OptimDirection::Minimize).unwrap();

        // execute search
        let u_opt = gs.run(num_iterations, 0.0);

        // test results, optimum at [1, 1, 1, ... 1];
        for &vi in gs.parameters() {
            assert_abs_diff_eq!(vi, 1.0, epsilon = tol);
        }

        // test value of utility (should be nearly 0)
        assert_abs_diff_eq!(rosenbrock(gs.parameters()), 0.0, epsilon = tol);
        assert!(u_opt < tol);
    }

    // test configuration
    #[test]
    #[autotest_annotate(autotest_qnsearch_config)]
    fn qnsearch_config() {
        // no parameters -> error
        assert!(NewtonSearch::new(Vec::new(), |v| rosenbrock(v), OptimDirection::Minimize).is_err());

        // a null utility is unrepresentable in Rust (the closure is always present)

        // create a proper object
        let v = vec![0.0f32; 8];
        assert!(NewtonSearch::new(v, |v| rosenbrock(v), OptimDirection::Minimize).is_ok());
    }

    #[test]
    fn run_zero_iterations_preserves_search() {
        let calls = Cell::new(0);
        let mut search = NewtonSearch::new(
            vec![2.0],
            |parameters| {
                calls.set(calls.get() + 1);
                parameters[0].powi(2)
            },
            OptimDirection::Minimize,
        )
        .unwrap();

        let calls_before = calls.get();
        let utility = search.utility();
        assert_eq!(search.run(0, -1.0), utility);
        assert_eq!(search.parameters(), &[2.0]);
        assert_eq!(search.utility(), utility);
        assert_eq!(calls.get(), calls_before);
    }

    #[test]
    fn newton_step_reaches_quadratic_extrema_in_both_directions() {
        for direction in [OptimDirection::Minimize, OptimDirection::Maximize] {
            let sign = if direction == OptimDirection::Minimize { 1.0 } else { -1.0 };
            let mut search =
                NewtonSearch::new(vec![0.0], move |parameters| sign * (parameters[0] - 1.0).powi(2), direction)
                    .unwrap();

            search.step();

            assert_abs_diff_eq!(search.parameters()[0], 1.0, epsilon = 0.05);
            assert!(search.utility().abs() < 0.01);
            // A successful maximization must grow, not shrink, the next step.
            assert!(search.gamma_hat > 1.0);
        }
    }

    #[test]
    fn hessian_probes_do_not_inherit_prior_offsets() {
        // At the origin, the analytic Hessian of x*y^2 is all zero.
        // An x offset left by the mixed partial makes d^2 f/dy^2 nonzero.
        let mut search = NewtonSearch::new(
            vec![0.0, 0.0],
            |parameters| parameters[0] * parameters[1] * parameters[1],
            OptimDirection::Minimize,
        )
        .unwrap();

        search.compute_hessian();

        for &entry in &search.h {
            assert_abs_diff_eq!(entry, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn newton_search_converges_from_rosenbrock_valley_edge() {
        let mut search = NewtonSearch::new(vec![-1.2, 1.0], rosenbrock, OptimDirection::Minimize).unwrap();

        let utility = search.run(2000, 1e-4);

        assert!(utility < 1e-4, "utility after search: {utility}");
        assert_abs_diff_eq!(search.parameters()[0], 1.0, epsilon = 0.02);
        assert_abs_diff_eq!(search.parameters()[1], 1.0, epsilon = 0.02);
    }
}
