// fir (finite impulse response) filter design using Parks-McClellan
// algorithm
//
// Much of this program has been borrowed heavily from [McClellan:1973]
// and [Janovetz:1998] with the exception of the Lagrange polynomial
// interpolation formulas and the structured 'firdespm' object. Several
// improvements have been made in the search algorithm to help maintain
// stability and convergence.
//
// References:
//  [Parks:1972] T. W. Parks and J. H. McClellan, "Chebyshev
//      Approximation for Nonrecursive Digital Filters with Linear
//      Phase," IEEE Transactions on Circuit Theory, vol. CT-19,
//      no. 2, March 1972.
//  [McClellan:1973] J. H. McClellan, T. W. Parks, L. R. Rabiner, "A
//      Computer Program for Designing Optimum FIR Linear Phase
//      Digital Filters," IEEE Transactions on Audio and
//      Electroacoustics, vol. AU-21, No. 6, December 1973.
//  [Rabiner:1975] L. R. Rabiner, J. H. McClellan, T. W. Parks, "FIR
//      Digital filter Design Techniques Using Weighted Chebyshev
//      Approximations," Proceedings of the IEEE, March 1975.
//  [Parks:1987] T. W. Parks and C. S. Burrus, "Digital Filter
//      Design," Upper Saddle River, NJ, John Wiley & Sons, Inc., 1987
//      (Section 3.3.3)
//  [Janovetz:1998] J. Janovetz, online: http://www.janovetz.com/jake/

use crate::error::{Error, Result};
use crate::math::{poly_fit_lagrange_barycentric, poly_val_lagrange_barycentric};
use std::cmp::max;

use super::estimate_req_filter_transition_bandwidth;

const IEXT_SEARCH_TOL: f64 = 1e-15;
const MAX_ITERATIONS: usize = 40;

/// Kind of response a Parks-McClellan design approximates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // Reserved for dedicated differentiator and Hilbert constructors.
pub(crate) enum FirPmBandType {
    /// Piecewise-constant band response.
    Bandpass,
    /// Differentiator. Not yet supported.
    Differentiator,
    /// Hilbert transform. Not yet supported.
    Hilbert,
}

/// How a Parks-McClellan error weight varies across a band.
///
/// The variation starts at the band's lower edge and multiplies the band's
/// weight. Larger weights reduce the error at those frequencies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirPmWeightType {
    /// Constant weight.
    Flat,
    /// Weight that grows as `exp(2 * (f - f_low))`.
    Exp,
    /// Weight that grows as `1 + 2.7 * (f - f_low)`.
    Lin,
}

/// Desired response and error weight at one frequency.
///
/// Returned by the callback given to
/// [`ParksMcClellanDesigner::new_with_response`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[doc(alias = "FirPmResponse")]
pub struct ParksMcClellanResponse {
    /// Desired amplitude response. Must be finite.
    pub desired: f64,
    /// Error weight. Must be finite and positive. A larger weight reduces the
    /// error at this frequency relative to others.
    pub weight: f64,
}

// TODO consider not using a struct here. is this reusable? could we combine ctor and execute?

/// Equiripple FIR filter designer based on the Parks-McClellan algorithm.
///
/// The designer finds the even-symmetric, linear-phase filter that minimizes
/// the largest weighted error across a set of bands. Frequencies between
/// bands are left unconstrained.
///
/// Use [`Self::new`] for a constant response in each band, or
/// [`Self::new_with_response`] for a response that varies with frequency.
/// Then call [`Self::execute`] to run the design. An even-length filter always
/// has zero response at 0.5 cycles per sample, so it cannot meet a
/// specification that requires gain there.
#[derive(Debug, Clone)]
#[doc(alias = "FirDesignPm")]
pub struct ParksMcClellanDesigner {
    h_len: usize,                // filter length
    s: usize,                    // odd/even filter length
    r: usize,                    // number of approximating functions
    num_bands: usize,            // number of discrete bands
    grid_size: usize,            // number of points on the grid
    grid_density: usize,         // density of the grid
    btype: FirPmBandType,        // band type
    bands: Vec<f64>,             // bands array [size: 2*num_bands]
    des: Vec<f64>,               // desired response [size: num_bands]
    weights: Vec<f64>,           // weights [size: num_bands]
    wtype: Vec<FirPmWeightType>, // weight type [size: num_bands]
    f: Vec<f64>,                 // frequencies, [0, 0.5]
    d: Vec<f64>,                 // desired response
    w: Vec<f64>,                 // weight
    e: Vec<f64>,                 // error
    x: Vec<f64>,                 // Chebyshev points : cos(2*pi*f)
    alpha: Vec<f64>,             // Lagrange interpolating polynomial
    c: Vec<f64>,                 // interpolants
    rho: f64,                    // extremal weighted error
    iext: Vec<usize>,            // indices of extrema
    num_exchanges: usize,        // number of changes in extrema
                                 // fid: Option<std::fs::File>, // file for debugging
}

impl ParksMcClellanDesigner {
    /// Creates a designer with a constant response in each band.
    ///
    /// `band_edges` holds a lower and an upper edge for each band, in cycles
    /// per sample. The edges must be finite, nondecreasing, and within
    /// `0.0..=0.5`. `desired_response`, `weights`, and `weight_types` each hold
    /// one value per band. Omit the last two for a flat weight of one in every
    /// band.
    ///
    /// # Errors
    ///
    /// Returns an error when `filter_length` is zero, `band_edges` is empty or
    /// has an odd length, an edge is invalid, another slice has the wrong
    /// length, a desired response is not finite, or a weight is not finite and
    /// positive.
    pub fn new(
        filter_length: usize,
        band_edges: &[f32],
        desired_response: &[f32],
        weights: Option<&[f32]>,
        weight_types: Option<&[FirPmWeightType]>,
    ) -> Result<ParksMcClellanDesigner> {
        let mut obj = ParksMcClellanDesigner::new_base(
            filter_length,
            band_edges,
            Some(desired_response),
            weights,
            weight_types,
            FirPmBandType::Bandpass,
        )?;
        obj.create_grid(None)?;
        // TODO : fix grid, weights according to filter type
        Ok(obj)
    }

    /// Creates a designer with a response that varies with frequency.
    ///
    /// `band_edges` holds a lower and an upper edge for each band, in cycles
    /// per sample, with the same limits as in [`Self::new`]. `response` is
    /// called once for each point on the design grid, with its frequency in
    /// cycles per sample. It returns the desired response and weight there.
    ///
    /// Grid points are evenly spaced from the lower edge of each band. The last
    /// point is then moved to the band's upper edge, and it keeps the response
    /// and weight returned for its original frequency. `response` is not called
    /// at the upper edge itself.
    ///
    /// # Errors
    ///
    /// Returns an error when `filter_length` or `band_edges` is invalid, as in
    /// [`Self::new`], or when `response` returns a nonfinite desired response
    /// or a weight that is not finite and positive. An error from `response`
    /// is returned unchanged.
    pub fn new_with_response<F>(
        filter_length: usize,
        band_edges: &[f32],
        mut response: F,
    ) -> Result<ParksMcClellanDesigner>
    where
        F: FnMut(f64) -> Result<ParksMcClellanResponse>,
    {
        let mut obj =
            ParksMcClellanDesigner::new_base(filter_length, band_edges, None, None, None, FirPmBandType::Bandpass)?;
        obj.create_grid(Some(&mut response))?;
        // TODO : fix grid, weights according to filter type
        Ok(obj)
    }

    /// Runs the design and returns `filter_length` taps.
    ///
    /// A successful result is the best design the algorithm found. It does not
    /// certify that the weighted error meets any particular bound.
    ///
    /// # Errors
    ///
    /// Returns an error when the exchange algorithm does not converge within
    /// 40 iterations or its interpolation step fails.
    pub fn execute(&mut self) -> Result<Vec<f32>> {
        self.execute_with_max_iterations(MAX_ITERATIONS)
    }

    fn execute_with_max_iterations(&mut self, max_iterations: usize) -> Result<Vec<f32>> {
        // initial guess of extremal frequencies evenly spaced on F
        // TODO : guarantee at least one extremal frequency lies in each band
        for i in 0..self.r + 1 {
            self.iext[i] = (i * (self.grid_size - 1)) / self.r;
        }

        // iterate over the Remez exchange algorithm
        for _ in 0..max_iterations {
            // compute interpolator
            self.compute_interp()?;

            // compute error
            self.compute_error()?;

            // search for new extremal frequencies
            self.iext_search()?;

            // check stopping criteria
            if self.is_search_complete()? {
                return self.compute_taps();
            }
        }

        Err(Error::NoConvergence(format!(
            "Parks-McClellan exchange failed to converge after {max_iterations} iterations",
        )))
    }

    fn new_base(
        h_len: usize,
        bands: &[f32],
        des: Option<&[f32]>,
        weights: Option<&[f32]>,
        wtype: Option<&[FirPmWeightType]>,
        btype: FirPmBandType,
    ) -> Result<ParksMcClellanDesigner> {
        // basic validation
        if h_len == 0 {
            return Err(Error::Config("Invalid filter length".to_string()));
        }
        if bands.is_empty() {
            return Err(Error::Config("At least one frequency band is required".to_string()));
        }
        if !bands.len().is_multiple_of(2) {
            return Err(Error::Config(format!("Expected pairs of band edges, got {} edges", bands.len())));
        }
        if btype != FirPmBandType::Bandpass {
            return Err(Error::Config(format!("Parks-McClellan filter type {btype:?} is not supported",)));
        }

        let num_band_edges = bands.len();
        let num_bands = num_band_edges / 2;
        if let Some(des) = des {
            if des.len() != num_bands {
                return Err(Error::Config(format!("Expected {num_bands} desired responses, got {}", des.len(),)));
            }
            if des.iter().any(|value| !value.is_finite()) {
                return Err(Error::Config("Desired responses must be finite".to_string()));
            }
        }
        if let Some(weights) = weights {
            if weights.len() != num_bands {
                return Err(Error::Config(format!("Expected {num_bands} weights, got {}", weights.len(),)));
            }
            if weights.iter().any(|weight| !weight.is_finite() || *weight <= 0.0) {
                return Err(Error::Config("Weights must be finite and greater than zero".to_string()));
            }
        }
        if let Some(wtype) = wtype {
            if wtype.len() != num_bands {
                return Err(Error::Config(format!("Expected {num_bands} weight types, got {}", wtype.len(),)));
            }
        }

        // validate filter specification
        if bands.iter().any(|band| !band.is_finite() || *band < 0.0 || *band > 0.5)
            || bands.windows(2).any(|pair| pair[1] < pair[0])
        {
            return Err(Error::Config("Bands must be finite, non-decreasing, and in [0, 0.5]".to_string()));
        }

        // create object
        let s = h_len % 2;
        let n = (h_len - s) / 2;
        let r = n + s;

        // estimate grid size
        let grid_density = 20;
        let mut grid_size = 0;
        let df = 0.5 / (grid_density * r) as f64;
        for i in 0..num_bands {
            let f0 = bands[2 * i] as f64;
            let f1 = bands[2 * i + 1] as f64;
            grid_size += ((f1 - f0) / df + 1.0).floor() as usize;
        }

        let mut obj = ParksMcClellanDesigner {
            h_len,
            s,
            r,
            num_bands,
            grid_size: 0,
            grid_density: 20,
            btype,
            bands: vec![0.0; num_band_edges],
            des: vec![0.0; num_bands],
            weights: vec![1.0; num_bands],
            wtype: vec![FirPmWeightType::Flat; num_bands],
            f: vec![0.0; grid_size],
            d: vec![0.0; grid_size],
            w: vec![0.0; grid_size],
            e: vec![0.0; grid_size],
            x: vec![0.0; r + 1],
            alpha: vec![0.0; r + 1],
            c: vec![0.0; r + 1],
            rho: 0.0,
            iext: vec![0; r + 1],
            num_exchanges: 0,
        };

        // copy input arrays
        for i in 0..num_bands {
            obj.bands[2 * i] = bands[2 * i] as f64;
            obj.bands[2 * i + 1] = bands[2 * i + 1] as f64;
            if let Some(des) = des {
                obj.des[i] = des[i] as f64;
            }
            if let Some(weights) = weights {
                obj.weights[i] = weights[i] as f64;
            }
            if let Some(wtype) = wtype {
                obj.wtype[i] = wtype[i];
            }
        }

        Ok(obj)
    }

    fn create_grid(
        &mut self,
        mut response: Option<&mut dyn FnMut(f64) -> Result<ParksMcClellanResponse>>,
    ) -> Result<()> {
        // frequency step size
        let df = 0.5 / (self.grid_density * self.r) as f64;

        // number of grid points counter
        let mut n = 0;

        for i in 0..self.num_bands {
            // extract band edges
            let f0 = if i == 0 && self.btype != FirPmBandType::Bandpass {
                // ensure first point is not zero for differentiator
                // and Hilbert transforms due to transformation (below)
                self.bands[2 * i].max(df)
            } else {
                self.bands[2 * i]
            };
            let f1 = self.bands[2 * i + 1];

            let num_points = max(1, ((f1 - f0) / df + 0.5).floor() as usize);

            for j in 0..num_points {
                self.f[n] = f0 + j as f64 * df;

                // compute desired response using function if provided
                if let Some(response) = response.as_mut() {
                    let value = response(self.f[n])?;
                    if !value.desired.is_finite() {
                        return Err(Error::Value(format!(
                            "Desired response at frequency {} must be finite",
                            self.f[n],
                        )));
                    }
                    if !value.weight.is_finite() || value.weight <= 0.0 {
                        return Err(Error::Value(format!(
                            "Response weight at frequency {} must be finite and greater than zero",
                            self.f[n],
                        )));
                    }
                    self.d[n] = value.desired;
                    self.w[n] = value.weight;
                } else {
                    self.d[n] = self.des[i];

                    // compute weight, applying weighting function
                    let fw = match self.wtype[i] {
                        FirPmWeightType::Flat => 1.0,
                        FirPmWeightType::Exp => (2.0 * j as f64 * df).exp(),
                        FirPmWeightType::Lin => 1.0 + 2.7 * j as f64 * df,
                    };
                    self.w[n] = self.weights[i] * fw;
                }

                n += 1;
            }

            // force endpoint to be upper edge of frequency band
            self.f[n - 1] = f1;
        }
        self.grid_size = n;

        // take care of special symmetry conditions here
        if self.btype == FirPmBandType::Bandpass {
            if self.s == 0 {
                // even length filter
                for i in 0..self.grid_size {
                    self.d[i] /= (std::f64::consts::PI * self.f[i]).cos();
                    self.w[i] *= (std::f64::consts::PI * self.f[i]).cos();
                }
            }
        } else {
            // differentiator, Hilbert transform
            if self.s == 0 {
                // even length filter
                for i in 0..self.grid_size {
                    self.d[i] /= (std::f64::consts::PI * self.f[i]).sin();
                    self.w[i] *= (std::f64::consts::PI * self.f[i]).sin();
                }
            } else {
                // odd length filter
                for i in 0..self.grid_size {
                    self.d[i] /= (2.0 * std::f64::consts::PI * self.f[i]).sin();
                    self.w[i] *= (2.0 * std::f64::consts::PI * self.f[i]).sin();
                }
            }
        }
        Ok(())
    }

    /// Computes the interpolating polynomial through the current extremal
    /// frequencies.
    fn compute_interp(&mut self) -> Result<()> {
        // compute Chebyshev points on F[iext[]] : cos(2*pi*f)
        for i in 0..self.r + 1 {
            self.x[i] = (2.0 * std::f64::consts::PI * self.f[self.iext[i]]).cos();
        }

        // compute Lagrange interpolating polynomial
        if self.x.windows(2).any(|pair| pair[0] == pair[1]) {
            // this handles the case where create_grid places two entries at the same frequency

            // this degenerate case requires a small weight perturbation to resolve
            // rather than do the perturbation in poly_fit_lagrange_barycentric, do it for this
            // branch specifically
            for (j, weight) in self.alpha.iter_mut().enumerate() {
                *weight = 1.0;
                for (k, &node) in self.x.iter().enumerate() {
                    if j != k {
                        *weight *= self.x[j] - node;
                    }
                }
                if *weight == 0.0 {
                    *weight += 1.0e-9;
                }
                *weight = weight.recip();
            }

            let w0 = self.alpha[0] + 1.0e-9;
            for weight in &mut self.alpha {
                *weight /= w0;
            }
        } else {
            poly_fit_lagrange_barycentric(&self.x, &mut self.alpha)?;
        }

        // compute rho
        let mut t0 = 0.0; // numerator
        let mut t1 = 0.0; // denominator
        for i in 0..self.r + 1 {
            t0 += self.alpha[i] * self.d[self.iext[i]];
            let sgn = if i % 2 == 1 { -1.0 } else { 1.0 };
            t1 += self.alpha[i] / self.w[self.iext[i]] * sgn;
        }
        self.rho = t0 / t1;

        // compute polynomial values (interpolants)
        for i in 0..self.r + 1 {
            let sgn = if i % 2 == 1 { -1.0 } else { 1.0 };
            self.c[i] = self.d[self.iext[i]] - sgn * self.rho / self.w[self.iext[i]];
        }
        Ok(())
    }

    fn compute_error(&mut self) -> Result<()> {
        for i in 0..self.grid_size {
            let xf = (2.0 * std::f64::consts::PI * self.f[i]).cos();
            let h = poly_val_lagrange_barycentric(&self.x, &self.c, &self.alpha, xf)?;
            self.e[i] = self.w[i] * (self.d[i] - h);
        }
        Ok(())
    }

    /// Searches the error curve for `r + 1` extremal indices.
    // TODO : return number of values which have changed (stopping criteria)
    fn iext_search(&mut self) -> Result<()> {
        // found extremal frequency indices
        let nmax = 2 * self.r + 2 * self.num_bands; // max number of extremals
        let mut found_iext = vec![0; nmax];
        let mut num_found = 0;

        // force f=0 into candidate set
        found_iext[num_found] = 0;
        num_found += 1;

        // search inside grid
        for i in 1..self.grid_size - 1 {
            if (((self.e[i] >= 0.0) && (self.e[i - 1] <= self.e[i]) && (self.e[i + 1] <= self.e[i]))
                || ((self.e[i] < 0.0) && (self.e[i - 1] >= self.e[i]) && (self.e[i + 1] >= self.e[i])))
                && num_found < nmax
            {
                found_iext[num_found] = i;
                num_found += 1;
            }
        }

        // force f=0.5 into candidate set
        if num_found < nmax {
            found_iext[num_found] = self.grid_size - 1;
            num_found += 1;
        }

        if num_found < self.r + 1 {
            // too few extremal frequencies found.  Theoretically, this
            // should never happen as the Chebyshev alternation theorem
            // guarantees at least r+1 extrema, however due to finite
            // machine precision, interpolation can be imprecise
            return Err(Error::NoConvergence(format!(
                "Parks-McClellan exchange found {num_found} extrema; need {}",
                self.r + 1,
            )));
        }

        // search extrema and eliminate smallest
        let mut num_extra = num_found - self.r - 1; // number of extra extremal frequencies

        while num_extra > 0 {
            // evaluate sign of first extrema
            let mut last_positive = self.e[found_iext[0]] > 0.0;

            //
            let mut imin = 0; // index of found_iext where _E is a minimum extreme
            let mut alternating_sign = true;
            for i in 1..num_found {
                // update new minimum error extreme
                // TODO: it seems without some small amount of tolerance,
                // the algorithm will not converge correctly
                if self.e[found_iext[i]].abs() < (self.e[found_iext[imin]].abs() - IEXT_SEARCH_TOL) {
                    imin = i;
                }

                if last_positive && self.e[found_iext[i]] < 0.0 {
                    last_positive = false;
                } else if !last_positive && self.e[found_iext[i]] >= 0.0 {
                    last_positive = true;
                } else {
                    // found two extrema with non-alternating sign; delete
                    // the smaller of the two
                    if self.e[found_iext[i]].abs() < self.e[found_iext[i - 1]].abs() {
                        imin = i;
                    } else {
                        imin = i - 1;
                    }
                    alternating_sign = false;
                    break;
                }
            }

            //
            if alternating_sign && num_extra == 1 {
                if self.e[found_iext[0]].abs() < self.e[found_iext[num_found - 1]].abs() {
                    imin = 0;
                } else {
                    imin = num_found - 1;
                }
            }

            num_extra -= 1;
            num_found -= 1;

            // Delete value in 'found_iext' at 'index imin'.  This
            // is equivalent to shifing all values left one position
            // starting at index imin+1. Do this after reducing
            // num_extra (the only valid slots).
            for i in imin..num_found {
                found_iext[i] = found_iext[i + 1];
            }
        }

        // count number of changes
        self.num_exchanges = 0;
        for (&old, &new) in self.iext[..self.r + 1].iter().zip(&found_iext[..self.r + 1]) {
            self.num_exchanges += if old == new { 0 } else { 1 };
        }

        // copy new values
        self.iext[..self.r + 1].copy_from_slice(&found_iext[..self.r + 1]);

        Ok(())
    }

    /// Reports whether the exchange algorithm has converged.
    fn is_search_complete(&mut self) -> Result<bool> {
        // if no extremal frequencies have been exchanged, Remez
        // algorithm has converged
        if self.num_exchanges == 0 {
            return Ok(true);
        }

        let tol = 1e-3f64;
        let mut emin = 0.0;
        let mut emax = 0.0;
        for i in 0..self.r + 1 {
            let e = self.e[self.iext[i]].abs();
            if i == 0 || e < emin {
                emin = e;
            }
            if i == 0 || e > emax {
                emax = e;
            }
        }

        // an exact fit has no error variation and has converged.
        if emax == 0.0 {
            return Ok(true);
        }

        Ok((emax - emin) / emax < tol)
    }

    /// Computes the filter taps from the converged response.
    fn compute_taps(&mut self) -> Result<Vec<f32>> {
        // re-generate interpolator and compute coefficients
        // for best cosine approximation
        self.compute_interp()?;

        let mut h = vec![0.0f32; self.h_len];

        // evaluate Lagrange polynomial on evenly spaced points
        let p = self.r - self.s + 1;
        let mut g = vec![0.0; p];
        for (i, gi) in g.iter_mut().enumerate() {
            let f = (i as f64) / (self.h_len as f64);
            let xf = (2.0 * std::f64::consts::PI * f).cos();
            let cf = poly_val_lagrange_barycentric(&self.x, &self.c, &self.alpha, xf)?;

            let g_val = if self.btype == FirPmBandType::Bandpass && self.s == 1 {
                // odd filter length, even symmetry
                1.0
            } else if self.btype == FirPmBandType::Bandpass && self.s == 0 {
                // even filter length, even symmetry
                (std::f64::consts::PI * i as f64 / self.h_len as f64).cos()
            } else if self.btype != FirPmBandType::Bandpass && self.s == 1 {
                // odd filter length, odd symmetry
                1.0
            } else if self.btype != FirPmBandType::Bandpass && self.s == 0 {
                // even filter length, odd symmetry
                1.0
            } else {
                // all other cases
                1.0
            };

            *gi = cf * g_val;
        }

        // compute inverse DFT (slow method), performing
        // transformation here for different filter types
        // TODO : flesh out computation for other filter types
        if self.btype == FirPmBandType::Bandpass {
            // odd filter length, even symmetry
            for (i, hi) in h.iter_mut().enumerate() {
                let mut v = g[0];
                let f = ((i as f64) - (p - 1) as f64 + 0.5 * (1.0 - self.s as f64)) / (self.h_len as f64);
                for j in 1..self.r {
                    v += 2.0 * g[j] * (2.0 * std::f64::consts::PI * f * j as f64).cos();
                }
                *hi = (v / (self.h_len as f64)) as f32;
            }
        } else if self.btype != FirPmBandType::Bandpass && self.s == 1 {
            // odd filter length, odd symmetry
            return Err(Error::Internal("Filter configuration not yet supported".to_string()));
        } else if self.btype != FirPmBandType::Bandpass && self.s == 0 {
            // even filter length, odd symmetry
            return Err(Error::Internal("Filter configuration not yet supported".to_string()));
        }

        Ok(h)
    }
}

/// Designs an equiripple FIR filter with a constant response in each band.
///
/// This is a shortcut for [`ParksMcClellanDesigner::new`] followed by
/// [`ParksMcClellanDesigner::execute`]. The parameters have the same meaning.
///
/// # Errors
///
/// Returns the same errors as those two calls.
pub fn fir_design_pm(
    filter_length: usize,
    band_edges: &[f32],
    desired_response: &[f32],
    weights: Option<&[f32]>,
    weight_types: Option<&[FirPmWeightType]>,
) -> Result<Vec<f32>> {
    let mut obj = ParksMcClellanDesigner::new(filter_length, band_edges, desired_response, weights, weight_types)?;
    obj.execute()
}

/// Designs an equiripple low-pass FIR filter.
///
/// The filter has `n` taps and a cutoff of `fc` cycles per sample. The
/// transition band is centered on `fc`, and its width is estimated from `n`
/// and the stopband attenuation `as_` in dB. The passband gain is one, unlike
/// the Kaiser and windowed-sinc designers. The stopband weight grows across
/// the band, so the stopband ripple shrinks at higher frequencies.
///
/// # Errors
///
/// Returns an error when `n` is zero, `fc` is nonfinite or outside
/// `0.0..=0.5`, `as_` is nonfinite, zero or negative, or the resulting band
/// specification is invalid. A `fc` near either end of its range can place a
/// band edge outside `0.0..=0.5`. It also returns an error when the design
/// does not converge.
pub fn fir_design_pm_lowpass(n: usize, fc: f32, as_: f32) -> Result<Vec<f32>> {
    if !(0.0..=0.5).contains(&fc) {
        return Err(Error::Config(format!("cutoff frequency ({fc}) must be in [0, 0.5]")));
    }
    if n == 0 {
        return Err(Error::Config("firdespm_lowpass(), filter length must be greater than zero".to_string()));
    }

    let ft = estimate_req_filter_transition_bandwidth(as_, n)?;

    let fp = fc - 0.5 * ft;
    let fs = fc + 0.5 * ft;
    let bands = [0.0f32, fp, fs, 0.5f32];
    let des = [1.0f32, 0.0f32];
    let weights = [1.0f32, 1.0f32];
    let wtype = [FirPmWeightType::Flat, FirPmWeightType::Exp];

    fir_design_pm(n, &bands, &des, Some(&weights), Some(&wtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::test_helpers::{validate_psd_signalf, PsdRegion};
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    fn test_firdespm_lowpass_dc_and_nyquist_response() {
        let h = fir_design_pm_lowpass(51, 0.2, 60.0).unwrap();
        let dc: f32 = h.iter().sum();
        let nyquist: f32 = h.iter().enumerate().map(|(i, &tap)| if i % 2 == 0 { tap } else { -tap }).sum();
        assert_abs_diff_eq!(dc, 1.0, epsilon = 0.003);
        assert_abs_diff_eq!(nyquist, 0.0, epsilon = 0.001);
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_bandpass_n24)]
    fn test_firdespm_bandpass_n24() {
        // [McClellan:1973], Figure 7.

        // Initialize variables
        let n = 24;
        let bands = vec![0.0f32, 0.08, 0.16, 0.5];
        let des = vec![1.0f32, 0.0f32];
        let weights = vec![1.0f32, 1.0f32];
        let tol = 1e-4f32;

        // Initialize pre-determined coefficient array
        #[rustfmt::skip]
        let h0 = vec![
             0.33740917e-2f32,
             0.14938299e-1f32,
             0.10569360e-1f32,
             0.25415067e-2f32,
            -0.15929392e-1f32,
            -0.34085343e-1f32,
            -0.38112177e-1f32,
            -0.14629169e-1f32,
             0.40089541e-1f32,
             0.11540713e-0f32,
             0.18850752e-0f32,
             0.23354606e-0f32,
            // symmetry
             0.23354606e-0f32,
             0.18850752e-0f32,
             0.11540713e-0f32,
             0.40089541e-1f32,
            -0.14629169e-1f32,
            -0.38112177e-1f32,
            -0.34085343e-1f32,
            -0.15929392e-1f32,
             0.25415067e-2f32,
             0.10569360e-1f32,
             0.14938299e-1f32,
             0.33740917e-2f32,
        ];

        // Create filter
        let h = fir_design_pm(n, &bands, &des, Some(&weights), None).unwrap();

        for i in 0..n {
            assert_abs_diff_eq!(h[i], h0[i], epsilon = tol);
        }
    }

    #[test]
    fn test_firdespm_default_weights_are_unit_weights() {
        let n = 24;
        let bands = [0.0, 0.08, 0.16, 0.5];
        let des = [1.0, 0.0];
        let weights = [1.0, 1.0];
        let h_default = fir_design_pm(n, &bands, &des, None, None).unwrap();
        let h_unit = fir_design_pm(n, &bands, &des, Some(&weights), None).unwrap();

        assert_eq!(h_default, h_unit);
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_bandpass_n32)]
    fn test_firdespm_bandpass_n32() {
        // [McClellan:1973], Figure 9.

        // Initialize variables
        let n = 32;
        let bands = vec![0.0f32, 0.1f32, 0.2f32, 0.35f32, 0.425f32, 0.5f32];
        let des = vec![0.0f32, 1.0f32, 0.0f32];
        let weights = vec![10.0f32, 1.0f32, 10.0f32];
        let tol = 1e-4f32;

        // Initialize pre-determined coefficient array
        #[rustfmt::skip]
        let h0 = vec![
            -0.57534121e-2f32,
             0.99027198e-3f32,
             0.75733545e-2f32,
            -0.65141192e-2f32,
             0.13960525e-1f32,
             0.22951469e-2f32,
            -0.19994067e-1f32,
             0.71369560e-2f32,
            -0.39657363e-1f32,
             0.11260114e-1f32,
             0.66233643e-1f32,
            -0.10497223e-1f32,
             0.85136133e-1f32,
            -0.12024993e+0f32,
            -0.29678577e+0f32,
             0.30410917e+0f32,
            // symmetry
             0.30410917e+0f32,
            -0.29678577e+0f32,
            -0.12024993e+0f32,
             0.85136133e-1f32,
            -0.10497223e-1f32,
             0.66233643e-1f32,
             0.11260114e-1f32,
            -0.39657363e-1f32,
             0.71369560e-2f32,
            -0.19994067e-1f32,
             0.22951469e-2f32,
             0.13960525e-1f32,
            -0.65141192e-2f32,
             0.75733545e-2f32,
             0.99027198e-3f32,
            -0.57534121e-2f32,
        ];

        // Create filter
        let h = fir_design_pm(n, &bands, &des, Some(&weights), None).unwrap();

        for i in 0..n {
            assert_abs_diff_eq!(h[i], h0[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_lowpass)]
    fn test_firdespm_lowpass() {
        // design filter
        let n = 51;
        let fc = 0.2f32;
        let as_ = 60.0f32;
        let h = fir_design_pm_lowpass(n, fc, as_).unwrap();

        // verify resulting spectrum
        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: -0.5,  fmax: -0.25, pmin:  0.0,   pmax: -as_, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.15, fmax:  0.15, pmin: -0.02,  pmax: 0.02, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.25,  fmax: 0.5,  pmin:  0.0,   pmax: -as_, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_signalf(&h, &regions).unwrap());
    }

    fn firdespm_response_helper(frequency: f64) -> Result<ParksMcClellanResponse> {
        Ok(ParksMcClellanResponse {
            desired: if frequency < 0.39 { (20.0 * frequency.abs()).exp() } else { 0.0 },
            weight: if frequency < 0.39 { (-10.0 * frequency).exp() } else { 1.0 },
        })
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_callback)]
    fn test_firdespm_response() {
        // design filter
        let n = 81;
        let bands = vec![0.0f32, 0.35f32, 0.4f32, 0.5f32];

        let captured_value = 42;
        let mut num_calls = 0;
        let mut q = ParksMcClellanDesigner::new_with_response(n, &bands, |frequency| {
            assert_eq!(captured_value, 42);
            num_calls += 1;
            firdespm_response_helper(frequency)
        })
        .unwrap();
        assert!(num_calls > 0);
        let h = q.execute().unwrap();

        // verify resulting spectrum
        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: -0.50,  fmax: -0.40, pmin:  0.0, pmax: -20.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.36,  fmax: -0.30, pmin: 52.0, pmax:  62.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin: -0.30,  fmax: -0.20, pmin: 34.0, pmax:  53.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin: -0.20,  fmax: -0.10, pmin: 15.0, pmax:  36.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin: -0.10,  fmax:  0.10, pmin:  0.0, pmax:  19.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.10,  fmax:  0.20, pmin: 15.0, pmax:  36.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.20,  fmax:  0.30, pmin: 34.0, pmax:  53.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.30,  fmax:  0.36, pmin: 52.0, pmax:  62.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.40,  fmax:  0.50, pmin:  0.0, pmax: -20.0, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signalf(&h, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_copy)]
    fn test_firdespm_copy() {
        // create valid object
        let bands = vec![0.0f32, 0.2, 0.3, 0.5]; // regions
        let des = vec![1.0f32, 0.0]; // desired values
        let w = vec![1.0f32, 1.0]; // weights
        let wtype = vec![FirPmWeightType::Flat, FirPmWeightType::Flat];
        let mut q0 = ParksMcClellanDesigner::new(51, &bands, &des, Some(&w), Some(&wtype)).unwrap();

        // copy object
        let mut q1 = q0.clone();

        // execute both
        let h0 = q0.execute().unwrap();
        let h1 = q1.execute().unwrap();

        assert_eq!(h0.as_slice(), h1.as_slice());

        // No need to manually destroy objects in Rust due to RAII
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_config)]
    fn test_firdespm_config() {
        assert!(fir_design_pm_lowpass(51, 0.2, 60.0).is_ok()); // ok
        assert!(fir_design_pm_lowpass(0, 0.2, 60.0).is_err());
        assert!(fir_design_pm_lowpass(51, 0.8, 60.0).is_err());
        assert!(fir_design_pm_lowpass(51, -0.2, 60.0).is_err());

        // try to create object with filter length 0
        assert!(ParksMcClellanDesigner::new(0, &[], &[], None, None).is_err());

        // try to create object with 0 bands
        assert!(ParksMcClellanDesigner::new(71, &[], &[], None, None).is_err());

        // create valid object
        // skipping a print test
        let bands = vec![0.0f32, 0.2, 0.3, 0.5]; // regions
        let des = vec![1.0f32, 0.0]; // desired values
        let w = vec![1.0f32, 1.0]; // weights
        let wtype = vec![FirPmWeightType::Flat, FirPmWeightType::Flat];
        // let q = FirdesPm::new(51, 2, &bands, &des, Some(&w), Some(&wtype), BandType::Bandpass).unwrap();
        // assert!(q.print().is_ok());

        // invalid bands & weights
        let bands_0 = vec![0.0f32, 0.3, 0.2, 0.5]; // overlapping bands
        let bands_1 = vec![-0.1f32, 0.0, 0.3, 0.6]; // bands out of range
        let w_0 = vec![1.0f32, -1.0]; // weights out of range

        // try to create regular object with invalid configuration
        assert!(ParksMcClellanDesigner::new(0, &bands, &des, Some(&w), Some(&wtype)).is_err());
        assert!(ParksMcClellanDesigner::new(51, &bands_0, &des, Some(&w), Some(&wtype)).is_err());
        assert!(ParksMcClellanDesigner::new(51, &bands_1, &des, Some(&w), Some(&wtype)).is_err());
        assert!(ParksMcClellanDesigner::new(51, &bands, &des, Some(&w_0), Some(&wtype)).is_err());

        // try to create response object with invalid configuration
        assert!(ParksMcClellanDesigner::new_with_response(0, &bands, firdespm_response_helper).is_err());
        assert!(ParksMcClellanDesigner::new_with_response(51, &bands_0, firdespm_response_helper).is_err());
        assert!(ParksMcClellanDesigner::new_with_response(51, &bands_1, firdespm_response_helper).is_err());
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_differentiator)]
    fn test_firdespm_differentiator() {
        // not implemented, should return error
        let bands = [0.0, 0.2, 0.3, 0.5];
        let desired_response = [1.0, 0.0];
        let weights = [1.0, 1.0];
        let weight_types = [FirPmWeightType::Flat, FirPmWeightType::Flat];
        let result = ParksMcClellanDesigner::new_base(
            51,
            &bands,
            Some(&desired_response),
            Some(&weights),
            Some(&weight_types),
            FirPmBandType::Differentiator,
        );

        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_hilbert)]
    fn test_firdespm_hilbert() {
        // not implemented, should return error
        let bands = [0.0, 0.2, 0.3, 0.5];
        let desired_response = [1.0, 0.0];
        let weights = [1.0, 1.0];
        let weight_types = [FirPmWeightType::Flat, FirPmWeightType::Flat];
        let result = ParksMcClellanDesigner::new_base(
            51,
            &bands,
            Some(&desired_response),
            Some(&weights),
            Some(&weight_types),
            FirPmBandType::Hilbert,
        );

        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_firdespm_response_error() {
        let bands = [0.0, 0.35, 0.4, 0.5];
        let error = Error::Value("response failed".into());
        let result = ParksMcClellanDesigner::new_with_response(81, &bands, |_| Err(error.clone()));

        assert_eq!(result.unwrap_err(), error);
    }

    #[test]
    fn test_firdespm_rejects_invalid_response_values() {
        let bands = [0.0, 0.35, 0.4, 0.5];
        let invalid_desired = ParksMcClellanDesigner::new_with_response(81, &bands, |_| {
            Ok(ParksMcClellanResponse { desired: f64::NAN, weight: 1.0 })
        });
        let non_finite_weight = ParksMcClellanDesigner::new_with_response(81, &bands, |_| {
            Ok(ParksMcClellanResponse { desired: 1.0, weight: f64::INFINITY })
        });
        let non_positive_weight = ParksMcClellanDesigner::new_with_response(81, &bands, |_| {
            Ok(ParksMcClellanResponse { desired: 1.0, weight: 0.0 })
        });

        assert!(matches!(invalid_desired, Err(Error::Value(_))));
        assert!(matches!(non_finite_weight, Err(Error::Value(_))));
        assert!(matches!(non_positive_weight, Err(Error::Value(_))));
    }

    #[test]
    fn test_firdespm_rejects_mismatched_specification_lengths() {
        let bands = [0.0, 0.2, 0.3, 0.5];
        let extra_bands = [0.0, 0.2, 0.3, 0.5, 0.5];
        let des = [1.0, 0.0];
        let weights = [1.0, 1.0];
        let wtype = [FirPmWeightType::Flat, FirPmWeightType::Flat];
        let short_bands = ParksMcClellanDesigner::new(51, &bands[..3], &des, Some(&weights), Some(&wtype));
        let long_bands = ParksMcClellanDesigner::new(51, &extra_bands, &des, Some(&weights), Some(&wtype));
        let short_des = ParksMcClellanDesigner::new(51, &bands, &des[..1], Some(&weights), Some(&wtype));
        let short_weights = ParksMcClellanDesigner::new(51, &bands, &des, Some(&weights[..1]), Some(&wtype));
        let short_wtype = ParksMcClellanDesigner::new(51, &bands, &des, Some(&weights), Some(&wtype[..1]));
        let response_short_bands = ParksMcClellanDesigner::new_with_response(51, &bands[..3], |_| {
            Ok(ParksMcClellanResponse { desired: 1.0, weight: 1.0 })
        });

        assert!(matches!(short_bands, Err(Error::Config(_))));
        assert!(matches!(long_bands, Err(Error::Config(_))));
        assert!(matches!(short_des, Err(Error::Config(_))));
        assert!(matches!(short_weights, Err(Error::Config(_))));
        assert!(matches!(short_wtype, Err(Error::Config(_))));
        assert!(matches!(response_short_bands, Err(Error::Config(_))));
    }

    #[test]
    fn test_firdespm_rejects_non_finite_specification_values() {
        let bands = [0.0, 0.2, 0.3, 0.5];
        let invalid_bands = [0.0, f32::NAN, 0.3, 0.5];
        let des = [1.0, 0.0];
        let invalid_des = [f32::INFINITY, 0.0];
        let weights = [1.0, 1.0];
        let invalid_weights = [1.0, f32::NAN];
        let bands_result = ParksMcClellanDesigner::new(51, &invalid_bands, &des, Some(&weights), None);
        let des_result = ParksMcClellanDesigner::new(51, &bands, &invalid_des, Some(&weights), None);
        let weights_result = ParksMcClellanDesigner::new(51, &bands, &des, Some(&invalid_weights), None);

        assert!(matches!(bands_result, Err(Error::Config(_))));
        assert!(matches!(des_result, Err(Error::Config(_))));
        assert!(matches!(weights_result, Err(Error::Config(_))));
    }

    #[test]
    fn test_firdespm_iteration_limit_reports_nonconvergence() {
        let bands = [0.0, 0.08, 0.16, 0.5];
        let des = [1.0, 0.0];
        let weights = [1.0, 1.0];
        let mut q = ParksMcClellanDesigner::new(24, &bands, &des, Some(&weights), None).unwrap();

        assert!(matches!(q.execute_with_max_iterations(1), Err(Error::NoConvergence(_)),));
    }

    #[test]
    fn test_firdespm_zero_error_is_converged() {
        let bands = [0.0, 0.2, 0.3, 0.5];
        let des = [0.0, 0.0];
        let weights = [1.0, 1.0];
        let mut q = ParksMcClellanDesigner::new(51, &bands, &des, Some(&weights), None).unwrap();

        q.num_exchanges = 1;
        q.e.fill(0.0);
        assert!(q.is_search_complete().unwrap());
    }

    #[test]
    fn test_firdespm_iext_search_reports_too_few_extrema() {
        let bands = [0.0, 0.16, 0.34, 0.5];
        let des = [1.0, 0.0];
        let weights = [1.0, 1.0];
        let mut q = ParksMcClellanDesigner::new(81, &bands, &des, Some(&weights), None).unwrap();

        // A monotonic error curve has only its two endpoints as extrema.
        for i in 0..q.grid_size {
            q.e[i] = i as f64;
        }

        assert!(matches!(q.iext_search(), Err(Error::NoConvergence(_))));
    }
}
