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
use std::cmp::max;
use crate::math::{poly_fit_lagrange_barycentric, poly_val_lagrange_barycentric};

use super::estimate_req_filter_transition_bandwidth;

const IEXT_SEARCH_TOL: f64 = 1e-15;

/// Parks-McClellan filter design band type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirPmBandType {
    /// regular band-pass filter
    Bandpass,
    /// differentiating filter
    Differentiator,
    /// Hilbert transform
    Hilbert,
}

/// Parks-McClellan filter design weight type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirPmWeightType {
    /// flat weighting
    Flat,
    /// exponential weighting
    Exp,
    /// linear weighting
    Lin,
}

/// A callback function for specifying desired response & weights
pub type FirPmCallback = fn(frequency: f64, userdata: Option<&dyn std::any::Any>, desired: &mut f64, weight: &mut f64) -> Result<()>;

// TODO consider not using a struct here. is this reusable? could we combine ctor and execute?

/// Parks-McClellan filter design object
#[derive(Debug, Clone)]
pub struct FirDesignPm {
    h_len: usize,              // filter length
    s: usize,                  // odd/even filter length
    r: usize,                  // number of approximating functions
    num_bands: usize,          // number of discrete bands
    grid_size: usize,          // number of points on the grid
    grid_density: usize,       // density of the grid
    btype: FirPmBandType,      // band type
    bands: Vec<f64>,           // bands array [size: 2*num_bands]
    des: Vec<f64>,             // desired response [size: num_bands]
    weights: Vec<f64>,         // weights [size: num_bands]
    wtype: Vec<FirPmWeightType>, // weight type [size: num_bands]
    f: Vec<f64>,               // frequencies, [0, 0.5]
    d: Vec<f64>,               // desired response
    w: Vec<f64>,               // weight
    e: Vec<f64>,               // error
    x: Vec<f64>,               // Chebyshev points : cos(2*pi*f)
    alpha: Vec<f64>,           // Lagrange interpolating polynomial
    c: Vec<f64>,               // interpolants
    rho: f64,                  // extremal weighted error
    iext: Vec<usize>,          // indices of extrema
    num_exchanges: usize,      // number of changes in extrema
    // fid: Option<std::fs::File>, // file for debugging
}

impl FirDesignPm {
    /// create new Parks-McClellan filter design object
    /// 
    /// # Arguments
    /// 
    /// * `h_len` - filter length
    /// * `num_bands` - number of bands
    /// * `bands` - band edges, f in [0,0.5], [size: 2*num_bands]
    /// * `des` - desired response, [size: num_bands]
    /// * `weights` - response weighting, [size: num_bands]
    /// * `wtype` - weight types (e.g. `FirPmWeightType::Flat`) [size: num_bands]
    /// * `btype` - band type (e.g. `FirPmBandType::Bandpass`)
    /// 
    /// # Returns
    /// 
    /// A new Parks-McClellan filter design object
    pub fn new(
        h_len: usize,
        num_bands: usize,
        bands: &[f32],
        des: &[f32],
        weights: Option<&[f32]>,
        wtype: Option<&[FirPmWeightType]>,
        btype: FirPmBandType,
    ) -> Result<FirDesignPm> {
        let mut obj = FirDesignPm::_new(h_len, num_bands, bands, Some(des), weights, wtype, btype)?;
        obj.create_grid(None, None)?;
        // TODO : fix grid, weights according to filter type
        Ok(obj)
    }

    // TODO consider using a FnMut instead of callback+userdata

    /// create Parks-McClellan filter design object with user-defined callback
    /// 
    /// # Arguments
    /// 
    /// * `h_len` - filter length
    /// * `num_bands` - number of bands
    /// * `bands` - band edges, f in [0,0.5], [size: num_bands]
    /// * `btype` - band type (e.g. `FirPmBandType::Bandpass`)
    /// * `callback` - user-defined callback for specifying desired response & weights
    /// * `userdata` - user-defined data structure for callback function
    /// 
    /// # Returns
    /// 
    /// A new Parks-McClellan filter design object
    pub fn new_with_callback(
        h_len: usize,
        num_bands: usize,
        bands: &[f32],
        btype: FirPmBandType,
        callback: FirPmCallback,
        userdata: Option<&dyn std::any::Any>,
    ) -> Result<FirDesignPm> {
        let mut obj = FirDesignPm::_new(h_len, num_bands, bands, None, None, None, btype)?;
        obj.create_grid(Some(callback), userdata)?;
        // TODO : fix grid, weights according to filter type
        Ok(obj)
    }

    /// execute filter design and return the filter coefficients
    /// 
    /// # Returns
    /// 
    /// A vec of filter coefficients
    pub fn execute(&mut self) -> Result<Vec<f32>> {
        // initial guess of extremal frequencies evenly spaced on F 
        // TODO : guarantee at least one extremal frequency lies in each band
        for i in 0..self.r+1 {
            self.iext[i] = (i * (self.grid_size-1)) / self.r;
        }

        // iterate over the Remez exchange algorithm
        let max_iterations = 40;
        for _ in 0..max_iterations {
            // compute interpolator
            self.compute_interp()?;

            // compute error
            self.compute_error()?;

            // search for new extremal frequencies
            self.iext_search()?;

            // check stopping criteria
            if self.is_search_complete()? {
                break;
            }
        }
        let h = self.compute_taps()?;
        Ok(h)
    }

    fn _new(
        h_len: usize,
        num_bands: usize,
        bands: &[f32],
        des: Option<&[f32]>,
        weights: Option<&[f32]>,
        wtype: Option<&[FirPmWeightType]>,
        btype: FirPmBandType,
    ) -> Result<FirDesignPm> {
        // basic validation
        if h_len == 0 {
            return Err(Error::Config("Invalid filter length".to_string()));
        }
        if num_bands == 0 {
            return Err(Error::Config("Invalid number of bands".to_string()));
        }
        
        // validate filter specification
        let mut bands_valid = true;
        let mut weights_valid = true;
        for i in 0..2*num_bands {
            if bands[i] < 0.0 || bands[i] > 0.5 {
                bands_valid = false;
            }
            if i > 0 && bands[i] < bands[i-1] {
                bands_valid = false;
            }
        }
        if weights.is_some() {
            for i in 0..num_bands {
                if weights.unwrap()[i] <= 0.0 {
                    weights_valid = false;
                }
            }
        }

        if !bands_valid {
            return Err(Error::Config("Invalid bands".to_string()));
        }
        if !weights_valid {
            return Err(Error::Config("Invalid weights".to_string()));
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
            let f0 = bands[2*i] as f64;
            let f1 = bands[2*i+1] as f64;
            grid_size += ((f1 - f0) / df + 1.0).floor() as usize;
        }

        let mut obj = FirDesignPm {
            h_len,
            s,
            r,
            num_bands,
            grid_size: 0,
            grid_density: 20,
            btype,
            bands: vec![0.0; 2*num_bands],
            des: vec![0.0; num_bands],
            weights: vec![0.0; num_bands],
            wtype: vec![FirPmWeightType::Flat; num_bands],
            f: vec![0.0; grid_size],
            d: vec![0.0; grid_size],
            w: vec![0.0; grid_size],
            e: vec![0.0; grid_size],
            x: vec![0.0; r+1],
            alpha: vec![0.0; r+1],
            c: vec![0.0; r+1],
            rho: 0.0,
            iext: vec![0; r+1],
            num_exchanges: 0,
        };

        // copy input arrays
        for i in 0..num_bands {
            obj.bands[2*i] = bands[2*i] as f64;
            obj.bands[2*i+1] = bands[2*i+1] as f64;
            if des.is_some() {
                obj.des[i] = des.unwrap()[i] as f64;
            }
            if weights.is_some() {
                obj.weights[i] = weights.unwrap()[i] as f64;
            }
            if wtype.is_some() {
                obj.wtype[i] = wtype.unwrap()[i];
            }
        }

        Ok(obj)
    }

    fn create_grid(
        &mut self,
        callback: Option<FirPmCallback>,
        userdata: Option<&dyn std::any::Any>,
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
                self.bands[2*i].max(df)
            } else {
                self.bands[2*i]
            };
            let f1 = self.bands[2*i+1];

            let num_points = max(1, ((f1 - f0) / df + 0.5).floor() as usize);

            for j in 0..num_points {
                self.f[n] = f0 + j as f64 * df;
                
                // compute desired response using callback if provided
                if callback.is_some() {
                    callback.unwrap()(self.f[n], userdata, &mut self.d[n], &mut self.w[n])?;
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
            self.f[n-1] = f1;
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

    /// compute interpolating polynomial
    fn compute_interp(&mut self) -> Result<()> {
        // compute Chebyshev points on F[iext[]] : cos(2*pi*f)
        for i in 0..self.r+1 {
            self.x[i] = (2.0 * std::f64::consts::PI * self.f[self.iext[i]]).cos();
        }

        // compute Lagrange interpolating polynomial
        poly_fit_lagrange_barycentric(&self.x, self.r+1, &mut self.alpha);
        
        // compute rho
        let mut t0 = 0.0;  // numerator
        let mut t1 = 0.0;  // denominator
        for i in 0..self.r+1 {
            t0 += self.alpha[i] * self.d[self.iext[i]];
            let sgn = if i % 2 == 1 { -1.0 } else { 1.0 };
            t1 += self.alpha[i] / self.w[self.iext[i]] * sgn;
        }
        self.rho = t0 / t1;

        // compute polynomial values (interpolants)
        for i in 0..self.r+1 {
            let sgn = if i % 2 == 1 { -1.0 } else { 1.0 };
            self.c[i] = self.d[self.iext[i]] - sgn * self.rho / self.w[self.iext[i]];
        }
        Ok(())
    }

    fn compute_error(&mut self) -> Result<()> {
        for i in 0..self.grid_size {
            let xf = (2.0 * std::f64::consts::PI * self.f[i]).cos();
            let h = poly_val_lagrange_barycentric(&self.x, &self.c, &self.alpha, xf, self.r+1);
            self.e[i] = self.w[i] * (self.d[i] - h);
        }
        Ok(())
    }

    /// search error curve for r+1 extremal indices
    /// TODO : return number of values which have changed (stopping criteria)
    fn iext_search(&mut self) -> Result<()> {
        // found extremal frequency indices
        let nmax = 2*self.r + 2*self.num_bands; // max number of extremals
        let mut found_iext = vec![0; nmax];
        let mut num_found = 0;

        // force f=0 into candidate set
        found_iext[num_found] = 0;
        num_found += 1;

        // search inside grid
        for i in 1..self.grid_size-1 {
            if ((self.e[i] >= 0.0) && (self.e[i-1] <= self.e[i]) && (self.e[i+1] <= self.e[i])) ||
               ((self.e[i] < 0.0) && (self.e[i-1] >= self.e[i]) && (self.e[i+1] >= self.e[i])) {
                if num_found < nmax {
                    found_iext[num_found] = i;
                    num_found += 1;
                }
            }
        }

        // force f=0.5 into candidate set
        if num_found < nmax {
            found_iext[num_found] = self.grid_size-1;
            num_found += 1;
        }

        if num_found < self.r+1 {
            // too few extremal frequencies found.  Theoretically, this
            // should never happen as the Chebyshev alternation theorem
            // guarantees at least r+1 extrema, however due to finite
            // machine precision, interpolation can be imprecise
            self.num_exchanges = 0;
            // TODO investigate if we should return error
            // return Err(Error::Internal("Too few extremal frequencies found".to_string()));
            return Ok(());
        }

        // search extrema and eliminate smallest
        let mut num_extra = num_found - self.r - 1; // number of extra extremal frequencies

        while num_extra > 0 {
            // evaluate sign of first extrema
            let mut last_positive = self.e[found_iext[0]] > 0.0;

            //
            let mut imin = 0;    // index of found_iext where _E is a minimum extreme
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
                    if self.e[found_iext[i]].abs() < self.e[found_iext[i-1]].abs() {
                        imin = i;
                    } else {
                        imin = i-1;
                    }
                    alternating_sign = false;
                    break;
                }   
            }

            //
            if alternating_sign && num_extra == 1 {
                if self.e[found_iext[0]].abs() < self.e[found_iext[num_found-1]].abs() {
                    imin = 0;
                } else {
                    imin = num_found-1;
                }
            }

            // Delete value in 'found_iext' at 'index imin'.  This
            // is equivalent to shifing all values left one position
            // starting at index imin+1
            for i in imin..num_found {
                found_iext[i] = found_iext[i+1];
            }

            num_extra -= 1;
            num_found -= 1;
        }

        // count number of changes
        self.num_exchanges = 0;
        for i in 0..self.r+1 {
            self.num_exchanges += if self.iext[i] == found_iext[i] { 0 } else { 1 };
        }

        // copy new values
        for i in 0..self.r+1 {
            self.iext[i] = found_iext[i];
        }

        Ok(())
    }

    /// evaluate result to determine if Remez exchange algorithm
    /// has converged
    fn is_search_complete(&mut self) -> Result<bool> {
        // if no extremal frequencies have been exchanged, Remez
        // algorithm has converged
        if self.num_exchanges == 0 {
            return Ok(true);
        }

        let tol = 1e-3f64;
        let mut emin = 0.0;
        let mut emax = 0.0;
        for i in 0..self.r+1 {
            let e = self.e[self.iext[i]].abs();
            if i == 0 || e < emin {
                emin = e;
            }
            if i == 0 || e > emax {
                emax = e;
            }
        }
        Ok((emax - emin) / emax < tol)
    }

    /// compute filter taps (coefficients) from result
    fn compute_taps(&mut self) -> Result<Vec<f32>> {
        // re-generate interpolator and compute coefficients
        // for best cosine approximation
        self.compute_interp()?;

        let mut h = vec![0.0f32; self.h_len];

        // evaluate Lagrange polynomial on evenly spaced points 
        let p = self.r - self.s + 1;
        let mut g = vec![0.0; p];
        for i in 0..p {
            let f = (i as f64) / (self.h_len as f64);
            let xf = (2.0 * std::f64::consts::PI * f).cos();
            let cf = poly_val_lagrange_barycentric(&self.x, &self.c, &self.alpha, xf, self.r+1);

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

            g[i] = cf * g_val;
        }

        // compute inverse DFT (slow method), performing
        // transformation here for different filter types
        // TODO : flesh out computation for other filter types
        if self.btype == FirPmBandType::Bandpass {
            // odd filter length, even symmetry
            for i in 0..self.h_len {
                let mut v = g[0];
                let f = ((i as f64) - (p-1) as f64 + 0.5 * (1.0 - self.s as f64)) / (self.h_len as f64);
                for j in 1..self.r {
                    v += 2.0 * g[j] * (2.0 * std::f64::consts::PI * f * j as f64).cos();
                }
                h[i] = (v / (self.h_len as f64)) as f32;
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

/// Run filter design (full life cycle of object)
/// 
/// # Arguments
/// 
/// * `h_len` : length of filter (number of taps)
/// * `num_bands` : number of frequency bands
/// * `bands` : band edges, f in [0,0.5], [size: num_bands x 2]
/// * `des` : desired response [size: num_bands x 1]
/// * `weights` : response weighting [size: num_bands x 1]
/// * `wtype` : weight types (e.g. `FirPmWeightType::Flat`) [size: num_bands x 1]
/// * `btype` : band type (e.g. `FirPmBandType::Bandpass`)
/// 
/// # Returns
/// 
/// A vec of filter coefficients
pub fn fir_design_pm(
    h_len: usize, 
    num_bands: usize, 
    bands: &[f32], 
    des: &[f32], 
    weights: Option<&[f32]>, 
    wtype: Option<&[FirPmWeightType]>, 
    btype: FirPmBandType) -> Result<Vec<f32>> {

    let mut obj = FirDesignPm::new(h_len, num_bands, bands, des, weights, wtype, btype)?;
    obj.execute()
}

/// Run filter design for basic low-pass filter
/// 
/// # Arguments
/// 
/// * `n` : filter length, n > 0
/// * `fc` : cutoff frequency, 0 < fc < 0.5
/// * `as_` : stop-band attenuation \[dB\], as_ > 0
/// * `mu` : fractional sample offset, -0.5 < mu < 0.5 \[ignored\]
/// 
/// # Returns
/// 
/// A vec of filter coefficients
pub fn fir_design_pm_lowpass(
    n: usize, 
    fc: f32, 
    as_: f32, 
    mu: f32) -> Result<Vec<f32>> {
    if mu < -0.5 || mu > 0.5 {
        return Err(Error::Config("firdespm_lowpass(), mu (%12.4e) out of range [-0.5,0.5]".to_string()));
    }
    if fc < 0.0 || fc > 0.5 {
        return Err(Error::Config("firdespm_lowpass(), cutoff frequency (%12.4e) out of range (0, 0.5)".to_string()));
    }
    if n == 0 {
        return Err(Error::Config("firdespm_lowpass(), filter length must be greater than zero".to_string()));
    }

    let ft = estimate_req_filter_transition_bandwidth(as_, n)?;

    let fp = fc - 0.5 * ft;
    let fs = fc + 0.5 * ft;
    let num_bands = 2;
    let bands = [0.0f32, fp, fs, 0.5f32];
    let des = [1.0f32, 0.0f32];
    let weights = [1.0f32, 1.0f32];
    let wtype = [FirPmWeightType::Flat, FirPmWeightType::Exp];
    let btype = FirPmBandType::Bandpass;

    fir_design_pm(n, num_bands, &bands, &des, Some(&weights), Some(&wtype), btype)
}


#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_signalf};

    #[test]
    #[autotest_annotate(autotest_firdespm_bandpass_n24)]
    fn test_firdespm_bandpass_n24() {
        // [McClellan:1973], Figure 7.

        // Initialize variables
        let n = 24;
        let num_bands = 2;
        let bands = vec![0.0f32, 0.08, 0.16, 0.5];
        let des = vec![1.0f32, 0.0f32];
        let weights = vec![1.0f32, 1.0f32];
        let btype = FirPmBandType::Bandpass;
        let tol = 1e-4f32;

        // Initialize pre-determined coefficient array
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
        let h = fir_design_pm(n, num_bands, &bands, &des, Some(&weights), None, btype).unwrap();

        for i in 0..n {
            assert_relative_eq!(h[i], h0[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_bandpass_n32)]
    fn test_firdespm_bandpass_n32() {
        // [McClellan:1973], Figure 9.

        // Initialize variables
        let n = 32;
        let num_bands = 3;  
        let bands = vec![0.0f32, 0.1f32, 0.2f32, 0.35f32, 0.425f32, 0.5f32];
        let des = vec![0.0f32, 1.0f32, 0.0f32];
        let weights = vec![10.0f32, 1.0f32, 10.0f32];
        let btype = FirPmBandType::Bandpass;
        let tol = 1e-4f32;

        // Initialize pre-determined coefficient array
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
        let h = fir_design_pm(n, num_bands, &bands, &des, Some(&weights), None, btype).unwrap();

        for i in 0..n {
            assert_relative_eq!(h[i], h0[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_lowpass)]
    fn test_firdespm_lowpass() {
        // design filter
        let n  = 51;
        let fc = 0.2f32;
        let as_ = 60.0f32;
        let mu = 0.0f32;
        let h = fir_design_pm_lowpass(n, fc, as_, mu).unwrap();

        // verify resulting spectrum
        let regions = [
            PsdRegion { fmin: -0.5,  fmax: -0.25, pmin:  0.0,   pmax: -as_, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.15, fmax:  0.15, pmin: -0.02,  pmax: 0.02, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.25,  fmax: 0.5,  pmin:  0.0,   pmax: -as_, test_lo: false, test_hi: true },
        ];
        
        assert!(validate_psd_signalf(&h, &regions).unwrap());
    }

    fn callback_firdespm_helper(
        frequency: f64,
        userdata: Option<&dyn std::any::Any>,
        desired: &mut f64,
        weight: &mut f64
    ) -> Result<()> {
        assert!(userdata.is_some());
        let userdata = userdata.unwrap();
        assert!(userdata.downcast_ref::<i32>().is_some());
        let userdata = userdata.downcast_ref::<i32>().unwrap();
        assert_eq!(*userdata, 42);

        *desired = if frequency < 0.39 {
            (20.0 * frequency.abs()).exp()
        } else {
            0.0
        };
        *weight = if frequency < 0.39 {
            (-10.0 * frequency).exp()
        } else {
            1.0
        };
        Ok(())
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_callback)]
    fn test_firdespm_callback() {
        // design filter
        let n = 81;
        let num_bands = 2;
        let bands = vec![0.0f32, 0.35f32, 0.4f32, 0.5f32];
        let btype = FirPmBandType::Bandpass;

        let userdata = 42;
        let mut q = FirDesignPm::new_with_callback(n, num_bands, &bands, btype, callback_firdespm_helper, Some(&userdata)).unwrap();
        let h = q.execute().unwrap();

        // verify resulting spectrum
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
        let bands = vec![0.0f32, 0.2, 0.3, 0.5];  // regions
        let des = vec![1.0f32, 0.0];              // desired values
        let w = vec![1.0f32, 1.0];                // weights
        let wtype = vec![FirPmWeightType::Flat, FirPmWeightType::Flat];
        let mut q0 = FirDesignPm::new(51, 2, &bands, &des, Some(&w), Some(&wtype), FirPmBandType::Bandpass).unwrap();

        // copy object
        let mut q1 = q0.clone();

        // execute both
        let h0 = q0.execute().unwrap();
        let h1 = q1.execute().unwrap();

        assert_relative_eq!(h0.as_slice(), h1.as_slice(), epsilon = f32::EPSILON);

        // No need to manually destroy objects in Rust due to RAII
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_config)]
    fn test_firdespm_config() {
        assert!(fir_design_pm_lowpass(51, 0.2, 60.0, 0.0).is_ok()); // ok
        assert!(fir_design_pm_lowpass(0, 0.2, 60.0, 0.0).is_err());
        assert!(fir_design_pm_lowpass(51, 0.2, 60.0, -1.0).is_err());
        assert!(fir_design_pm_lowpass(51, 0.2, 60.0, 1.0).is_err());
        assert!(fir_design_pm_lowpass(51, 0.8, 60.0, 0.0).is_err());
        assert!(fir_design_pm_lowpass(51, -0.2, 60.0, 0.0).is_err());

        // try to create object with filter length 0
        assert!(FirDesignPm::new(0, 3, &[], &[], None, None, FirPmBandType::Bandpass).is_err());

        // try to create object with 0 bands
        assert!(FirDesignPm::new(71, 0, &[], &[], None, None, FirPmBandType::Bandpass).is_err());

        // create valid object
        // skipping a print test
        let bands = vec![0.0f32, 0.2, 0.3, 0.5];  // regions
        let des = vec![1.0f32, 0.0];              // desired values
        let w = vec![1.0f32, 1.0];                // weights
        let wtype = vec![FirPmWeightType::Flat, FirPmWeightType::Flat];
        // let q = FirdesPm::new(51, 2, &bands, &des, Some(&w), Some(&wtype), BandType::Bandpass).unwrap();
        // assert!(q.print().is_ok());

        // invalid bands & weights
        let bands_0 = vec![0.0f32, 0.3, 0.2, 0.5]; // overlapping bands
        let bands_1 = vec![-0.1f32, 0.0, 0.3, 0.6]; // bands out of range
        let w_0 = vec![1.0f32, -1.0];           // weights out of range

        // try to create regular object with invalid configuration
        assert!(FirDesignPm::new(0, 2, &bands, &des, Some(&w), Some(&wtype), FirPmBandType::Bandpass).is_err());
        assert!(FirDesignPm::new(51, 0, &bands, &des, Some(&w), Some(&wtype), FirPmBandType::Bandpass).is_err());
        assert!(FirDesignPm::new(51, 2, &bands_0, &des, Some(&w), Some(&wtype), FirPmBandType::Bandpass).is_err());
        assert!(FirDesignPm::new(51, 2, &bands_1, &des, Some(&w), Some(&wtype), FirPmBandType::Bandpass).is_err());
        assert!(FirDesignPm::new(51, 2, &bands, &des, Some(&w_0), Some(&wtype), FirPmBandType::Bandpass).is_err());

        // try to create callback object with invalid configuration
        assert!(FirDesignPm::new_with_callback(0, 2, &bands, FirPmBandType::Bandpass, callback_firdespm_helper, None).is_err());
        assert!(FirDesignPm::new_with_callback(51, 0, &bands, FirPmBandType::Bandpass, callback_firdespm_helper, None).is_err());
        assert!(FirDesignPm::new_with_callback(51, 2, &bands_0, FirPmBandType::Bandpass, callback_firdespm_helper, None).is_err());
        assert!(FirDesignPm::new_with_callback(51, 2, &bands_1, FirPmBandType::Bandpass, callback_firdespm_helper, None).is_err());
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_differentiator)]
    fn test_firdespm_differentiator() {
        // create valid object
        let n = 51;
        let bands = vec![0.0f32, 0.2, 0.3, 0.5];  // regions
        let des = vec![1.0f32, 0.0];              // desired values
        let w = vec![1.0f32, 1.0];                // weights
        let wtype = vec![FirPmWeightType::Flat, FirPmWeightType::Flat];
        let btype = FirPmBandType::Differentiator;
        let mut q = FirDesignPm::new(n, 2, &bands, &des, Some(&w), Some(&wtype), btype).unwrap();
        // unsupported configuration
        assert!(q.execute().is_err());
    }

    #[test]
    #[autotest_annotate(autotest_firdespm_hilbert)]
    fn test_firdespm_hilbert() {
        // create valid object
        let n = 51;
        let bands = vec![0.0f32, 0.2, 0.3, 0.5];  // regions
        let des = vec![1.0f32, 0.0];              // desired values
        let w = vec![1.0f32, 1.0];                // weights
        let wtype = vec![FirPmWeightType::Flat, FirPmWeightType::Flat];
        let btype = FirPmBandType::Hilbert;
        let mut q = FirDesignPm::new(n, 2, &bands, &des, Some(&w), Some(&wtype), btype).unwrap();
        // unsupported configuration
        assert!(q.execute().is_err());
    }
}
