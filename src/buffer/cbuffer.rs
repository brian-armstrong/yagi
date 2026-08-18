//
// circular buffer
//

use crate::error::{Error, Result};

/// circular (ring) buffer with zero-copy reads
///
/// A FIFO queue of fixed capacity.
///
/// # Zero-copy reads and writes
///
/// [`Self::read`] borrows a contiguous run of samples out of the ring without
/// copying and without consuming them, [`Self::release`] consumes. This lets a
/// consumer feed a filter or FFT straight from the ring's storage. Similarly,
/// [`Self::reserve`] borrows a contiguous run of samples for writing, and
/// [`Self::commit`] advances the internal write index after write.
///
/// ```
/// # use yagi::buffer::cbuffer::CBuffer;
/// let mut q = CBuffer::<f32>::new(16).unwrap();
/// q.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
///
/// let n = {
///     let samples = q.read(3);
///     assert_eq!(samples, &[1.0, 2.0, 3.0]);
///     samples.len()
/// };
/// q.release(n).unwrap();
/// assert_eq!(q.size(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CBuffer<T> {
    // allocated memory array: max_size + max_read - 1 elements, the tail being
    // the mirror region used to linearize wrapping reads
    v: Vec<T>,

    // length of buffer
    max_size: usize,

    // maximum number of elements that can be read at any given time
    max_read: usize,

    // number of elements currently in buffer
    num_elements: usize,

    // index to read
    read_index: usize,

    // index to write
    write_index: usize,
}

impl<T: Default + Clone + Copy> CBuffer<T> {
    /// create circular buffer object
    ///
    ///  max_size   :   maximum buffer size
    pub fn new(max_size: usize) -> Result<Self> {
        Self::new_max(max_size, max_size)
    }

    /// create circular buffer object of a particular size, specifying the
    /// maximum number of elements that can be read at any one time
    ///
    ///  max_size   :   maximum buffer size
    ///  max_read   :   maximum size of a single borrowed read
    pub fn new_max(max_size: usize, max_read: usize) -> Result<Self> {
        if max_size == 0 {
            return Err(Error::Config("cbuffer size must be greater than zero".into()));
        }
        if max_read == 0 {
            return Err(Error::Config("cbuffer max_read must be greater than zero".into()));
        }
        if max_read > max_size {
            return Err(Error::Config("cbuffer max_read cannot exceed max_size".into()));
        }

        let num_allocated = max_size + max_read - 1;

        Ok(Self {
            v: vec![T::default(); num_allocated],
            max_size,
            max_read,
            num_elements: 0,
            read_index: 0,
            write_index: 0,
        })
    }

    /// clear internal buffer
    pub fn reset(&mut self) {
        self.read_index = 0;
        self.write_index = 0;
        self.num_elements = 0;
    }

    /// get number of elements currently in buffer
    pub fn size(&self) -> usize {
        self.num_elements
    }

    /// get maximum number of elements the buffer can hold
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// get maximum number of elements that can be read at any one time
    pub fn max_read(&self) -> usize {
        self.max_read
    }

    /// get number of elements available for writing
    pub fn space_available(&self) -> usize {
        self.max_size - self.num_elements
    }

    /// is buffer empty?
    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// is buffer full?
    pub fn is_full(&self) -> bool {
        self.num_elements == self.max_size
    }

    /// write a single sample into the buffer
    ///
    ///  v          :   input sample
    pub fn push(&mut self, v: T) -> Result<()> {
        // ensure buffer isn't already full
        if self.num_elements == self.max_size {
            return Err(Error::Range("cbuffer push(), no space available".into()));
        }

        // add sample at write index
        self.v[self.write_index] = v;

        // update write index
        self.write_index = (self.write_index + 1) % self.max_size;

        // increment number of elements
        self.num_elements += 1;
        Ok(())
    }

    /// write samples to the buffer
    ///
    ///  v          :   input samples
    pub fn write(&mut self, v: &[T]) -> Result<()> {
        let n = v.len();

        // ensure number of samples to write doesn't exceed space available
        if n > self.max_size - self.num_elements {
            return Err(Error::Range(
                "cbuffer write(), cannot write more elements than are available".into(),
            ));
        }

        self.num_elements += n;

        // space available at end of buffer
        let k = self.max_size - self.write_index;

        // check for condition where we need to wrap around
        if n > k {
            self.v[self.write_index..self.write_index + k].copy_from_slice(&v[..k]);
            self.v[..n - k].copy_from_slice(&v[k..]);
            self.write_index = n - k;
        } else {
            self.v[self.write_index..self.write_index + n].copy_from_slice(v);
            // wrap the write_index so that it's valid for reserve
            self.write_index = (self.write_index + n) % self.max_size;
        }
        Ok(())
    }

    /// remove and return a single element from the buffer
    pub fn pop(&mut self) -> Result<T> {
        // ensure there is at least one element
        if self.num_elements == 0 {
            return Err(Error::Range("cbuffer pop(), no elements available".into()));
        }

        let v = self.v[self.read_index];

        // increment read index
        self.read_index = (self.read_index + 1) % self.max_size;

        // decrement number of elements in the buffer
        self.num_elements -= 1;
        Ok(v)
    }

    /// borrow a contiguous run of samples without consuming them
    ///
    /// The returned slice may be shorter than `num_requested`: it is clamped to
    /// the number of elements held and to [`Self::max_read`]. Call
    /// [`Self::release`] to consume what was used.
    pub fn read(&mut self, num_requested: usize) -> &[T] {
        // adjust number requested depending upon availability
        let mut n = num_requested.min(self.num_elements);

        // restrict maximum number of elements to originally specified value
        n = n.min(self.max_read);

        // linearize the tail end of the buffer if necessary
        let contiguous = self.max_size - self.read_index;
        if n > contiguous {
            let spill = n - contiguous;
            self.v.copy_within(..spill, self.max_size);
        }

        &self.v[self.read_index..self.read_index + n]
    }

    /// release `n` samples from the buffer
    ///
    /// Advances the read index past samples previously handed out by
    /// [`Self::read`].
    pub fn release(&mut self, n: usize) -> Result<()> {
        // advance read_index by n making sure not to step on write_index
        if n > self.num_elements {
            return Err(Error::Range(
                "cbuffer release(), cannot release more elements in buffer than exist".into(),
            ));
        }

        self.read_index = (self.read_index + n) % self.max_size;
        self.num_elements -= n;
        Ok(())
    }

    /// reserve space for up to `n` samples, returning a mutable slice to write
    /// into
    ///
    /// Write into the returned slice, then call [`Self::commit`] with the number
    /// of samples actually produced, which may be fewer than the slice length.
    /// The buffer is not advanced until committed.
    ///
    /// The slice is **shorter than `n`** when the request does not fit
    /// contiguously. Reserve again after committing to get the rest,
    /// mirroring how [`Self::read`] clamps and reports through the slice length.
    ///
    /// ```
    /// # use yagi::buffer::cbuffer::CBuffer;
    /// let mut q = CBuffer::<f32>::new(8).unwrap();
    /// // leave the write index near the end of the ring
    /// q.write(&[0.0; 6]).unwrap();
    /// q.release(6).unwrap();
    ///
    /// let mut remaining = 8;
    /// let mut next = 0.0;
    /// while remaining > 0 {
    ///     let slice = q.reserve(remaining).unwrap();
    ///     let n = slice.len();
    ///     for s in slice.iter_mut() {
    ///         *s = next;
    ///         next += 1.0;
    ///     }
    ///     q.commit(n).unwrap();
    ///     remaining -= n;
    /// }
    /// assert!(q.is_full());
    /// ```
    ///
    /// Returns an error only when the buffer has no space at all, so that a
    /// caller looping on the slice length cannot spin on an empty reservation.
    pub fn reserve(&mut self, n: usize) -> Result<&mut [T]> {
        let available = self.max_size - self.num_elements;
        if n > 0 && available == 0 {
            return Err(Error::Range("cbuffer reserve(), no space available".into()));
        }

        // a reservation is a plain slice, so it cannot straddle the wrap
        let contiguous = self.max_size - self.write_index;
        let n = n.min(available).min(contiguous);

        Ok(&mut self.v[self.write_index..self.write_index + n])
    }

    /// commit `n` samples written into the slice from [`Self::reserve`]
    ///
    /// `n` must be no greater than the length of the slice `reserve` returned.
    pub fn commit(&mut self, n: usize) -> Result<()> {
        let contiguous = self.max_size - self.write_index;
        if n > self.max_size - self.num_elements || n > contiguous {
            return Err(Error::Range(
                "cbuffer commit(), more samples than reserve() could have returned".into(),
            ));
        }

        self.write_index = (self.write_index + n) % self.max_size;
        self.num_elements += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;
    use crate::random::{randf, randnf};

    #[test]
    #[autotest_annotate(autotest_cbufferf)]
    fn test_cbufferf() {
        // input array of values
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // output test arrays
        let test1 = [1.0f32, 2.0, 3.0, 4.0];
        let test2 = [3.0f32, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let test3 = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0];
        let test4 = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0];

        // create new circular buffer with 10 elements
        let mut q = CBuffer::<f32>::new(10).unwrap();
        // cbuffer: { <empty> }
        assert!(q.is_empty());

        // part 1: write 4 elements to the buffer
        q.write(&v[..4]).unwrap();
        // cbuffer: {1 2 3 4}
        assert!(!q.is_empty());

        // part 2: try to read 4 elements
        assert_eq!(q.read(4), &test1);

        // part 3: release two elements, write 8 more, read 10
        q.release(2).unwrap();
        // cbuffer: {3 4}
        q.write(&v[..8]).unwrap();
        // cbuffer: {3 4 1 2 3 4 5 6 7 8}
        assert_eq!(q.read(10), &test2);

        // part 4: pop single element from buffer
        assert_eq!(q.size(), 10);
        assert_eq!(q.pop().unwrap(), 3.0);
        // cbuffer: {4 1 2 3 4 5 6 7 8}
        assert_eq!(q.size(), 9);

        // part 5: release three elements, and try reading 10
        q.release(3).unwrap();
        // cbuffer: {3 4 5 6 7 8}
        assert_eq!(q.read(10), &test3);

        // part 6: test pushing multiple elements
        q.push(1.0).unwrap();
        q.push(2.0).unwrap();
        q.push(3.0).unwrap();
        // cbuffer: {3 4 5 6 7 8 1 2 3}
        assert_eq!(q.read(10), &test4);

        // part 7: add one more element; buffer should be full
        assert!(!q.is_full());
        q.push(1.0).unwrap();
        // cbuffer: {3 4 5 6 7 8 1 2 3 1}
        assert!(q.is_full());
    }

    #[test]
    #[autotest_annotate(autotest_cbuffercf)]
    fn test_cbuffercf() {
        let c = |re: f32, im: f32| Complex32::new(re, im);

        // input array of values
        let v = [
            c(1.0, -1.0), c(2.0, 2.0), c(3.0, -3.0), c(4.0, 4.0),
            c(5.0, -5.0), c(6.0, 6.0), c(7.0, -7.0), c(8.0, 8.0),
        ];

        // output test arrays
        let test1 = [c(1.0, -1.0), c(2.0, 2.0), c(3.0, -3.0), c(4.0, 4.0)];
        let test2 = [
            c(3.0, -3.0), c(4.0, 4.0), c(1.0, -1.0), c(2.0, 2.0), c(3.0, -3.0),
            c(4.0, 4.0), c(5.0, -5.0), c(6.0, 6.0), c(7.0, -7.0), c(8.0, 8.0),
        ];
        let test3 = [
            c(3.0, -3.0), c(4.0, 4.0), c(5.0, -5.0), c(6.0, 6.0), c(7.0, -7.0), c(8.0, 8.0),
        ];
        let test4 = [
            c(3.0, -3.0), c(4.0, 4.0), c(5.0, -5.0), c(6.0, 6.0), c(7.0, -7.0),
            c(8.0, 8.0), c(1.0, -1.0), c(2.0, 2.0), c(3.0, -3.0),
        ];

        // create new circular buffer with 10 elements
        let mut q = CBuffer::<Complex32>::new(10).unwrap();
        assert!(q.is_empty());

        // part 1: write 4 elements to the buffer
        q.write(&v[..4]).unwrap();
        assert!(!q.is_empty());

        // part 2: try to read 4 elements
        assert_eq!(q.read(4), &test1);

        // part 3: release two elements, write 8 more, read 10
        q.release(2).unwrap();
        q.write(&v[..8]).unwrap();
        assert_eq!(q.read(10), &test2);

        // part 4: pop single element from buffer
        assert_eq!(q.size(), 10);
        assert_eq!(q.pop().unwrap(), c(3.0, -3.0));
        assert_eq!(q.size(), 9);

        // part 5: release three elements, and try reading 10
        q.release(3).unwrap();
        assert_eq!(q.read(10), &test3);

        // part 6: test pushing multiple elements
        q.push(c(1.0, -1.0)).unwrap();
        q.push(c(2.0, 2.0)).unwrap();
        q.push(c(3.0, -3.0)).unwrap();
        assert_eq!(q.read(10), &test4);

        // part 7: add one more element; buffer should be full
        assert!(!q.is_full());
        q.push(c(1.0, -1.0)).unwrap();
        assert!(q.is_full());
    }

    // test general flow
    #[test]
    #[autotest_annotate(autotest_cbufferf_flow)]
    fn test_cbufferf_flow() {
        // options
        let max_size = 48; // maximum number of elements in buffer
        let max_read = 17; // maximum number of elements to read
        let num_elements = 1200; // total number of elements for run

        // temporary buffer to write samples before sending to cbuffer
        let mut write_buffer = vec![0.0f32; max_size];

        // create new circular buffer
        let mut q = CBuffer::<f32>::new_max(max_size, max_read).unwrap();

        let mut write_id = 0usize; // running total number of values written
        let mut read_id = 0usize; // running total number of values read

        loop {
            // write some values
            let num_available_to_write = q.space_available();

            // write samples if space is available
            if num_available_to_write > 0 {
                // number of elements to write
                let num_to_write =
                    (randf() * num_available_to_write as f32) as usize % num_available_to_write + 1;

                // generate samples to write
                for i in 0..num_to_write {
                    write_buffer[i] = write_id as f32;
                    write_id += 1;
                }

                // write samples
                q.write(&write_buffer[..num_to_write]).unwrap();
            }

            // read some values
            let num_available_to_read = q.size();

            // read samples if available
            if num_available_to_read > 0 {
                // number of elements to read
                let num_to_read = (randf() * num_available_to_read as f32) as usize
                    % num_available_to_read;

                // read samples and compare
                let num_read = {
                    let r = q.read(num_to_read);
                    for (i, &value) in r.iter().enumerate() {
                        assert_eq!(
                            value,
                            (read_id + i) as f32,
                            "read {} at offset {}, expected {}",
                            value,
                            i,
                            read_id + i
                        );
                    }
                    r.len()
                };
                read_id += num_read;

                // release all the samples that were read
                q.release(num_read).unwrap();
            }

            if read_id >= num_elements {
                break;
            }
        }
    }

    // test invalid configurations, etc.
    #[test]
    #[autotest_annotate(autotest_cbufferf_config)]
    fn test_cbufferf_config() {
        // options
        let max_size = 48; // maximum number of elements in buffer
        let max_read = 17; // maximum number of elements to read

        // create new circular buffer
        let mut q = CBuffer::<f32>::new_max(max_size, max_read).unwrap();

        assert_eq!(q.max_size(), max_size);
        assert_eq!(q.max_read(), max_read);

        // fill buffer with zeros
        while q.space_available() > 0 {
            assert!(q.push(0.0).is_ok());
        }

        // buffer full; cannot write more
        assert!(q.push(0.0).is_err());

        // reset
        q.reset();

        // buffer empty; cannot pop element or release any values
        assert!(q.pop().is_err());
        assert!(q.release(1).is_err());

        // liquid does not reject these, but neither can be satisfied
        assert!(CBuffer::<f32>::new(0).is_err());
        assert!(CBuffer::<f32>::new_max(8, 0).is_err());
        assert!(CBuffer::<f32>::new_max(8, 9).is_err());
    }

    // test copy
    #[test]
    #[autotest_annotate(autotest_cbuffer_copy)]
    fn test_cbuffer_copy() {
        // create base object
        let wlen = 20;
        let mut q0 = CBuffer::<Complex32>::new(wlen).unwrap();

        // write some values
        for _ in 0..wlen {
            q0.push(Complex32::new(randnf(), randnf())).unwrap();
        }
        q0.release(13).unwrap();

        // copy object
        let mut q1 = q0.clone();

        // write a few more values
        for _ in 0..12 {
            let v = Complex32::new(randnf(), randnf());
            q0.push(v).unwrap();
            q1.push(v).unwrap();
        }
        q0.release(4).unwrap();
        q1.release(4).unwrap();

        // check object values
        assert_eq!(q0.space_available(), q1.space_available());

        // read buffers and compare
        let n = q0.space_available();
        let r0 = q0.read(n).to_vec();
        let r1 = q1.read(n).to_vec();
        assert_eq!(r0, r1);
    }

    // the mirror must make a wrapping read contiguous and correct, for every
    // possible alignment of the read index against the wrap
    #[test]
    fn test_cbuffer_read_spans_wrap() {
        let max_size = 8;
        for offset in 0..max_size {
            let mut q = CBuffer::<f32>::new(max_size).unwrap();

            // advance the read/write indices to `offset` without leaving data
            for i in 0..offset {
                q.push(i as f32).unwrap();
            }
            q.release(offset).unwrap();
            assert!(q.is_empty());

            // fill completely: the contents now straddle the wrap
            let expected: Vec<f32> = (0..max_size).map(|i| 100.0 + i as f32).collect();
            q.write(&expected).unwrap();
            assert!(q.is_full());

            // a full-length read must be contiguous and in order
            assert_eq!(q.read(max_size), &expected[..], "offset {}", offset);

            // and reading it in two halves must agree
            q.release(3).unwrap();
            assert_eq!(q.read(max_size - 3), &expected[3..], "offset {}", offset);
        }
    }

    // read is clamped by both occupancy and max_read, and never consumes
    #[test]
    fn test_cbuffer_read_clamps_and_does_not_consume() {
        let mut q = CBuffer::<f32>::new_max(16, 4).unwrap();
        q.write(&[1.0, 2.0, 3.0]).unwrap();

        // clamped by occupancy
        assert_eq!(q.read(10).len(), 3);
        // repeated reads return the same data: read does not consume
        assert_eq!(q.read(10), &[1.0, 2.0, 3.0]);
        assert_eq!(q.size(), 3);

        // clamped by max_read
        q.write(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(q.read(6).len(), 4);
        assert_eq!(q.read(6), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(q.size(), 6);

        // an empty buffer yields an empty slice rather than an error
        q.reset();
        assert!(q.read(4).is_empty());
    }

    // reserve/commit must be equivalent to write, and commit may be short
    #[test]
    fn test_cbuffer_reserve_commit_matches_write() {
        let mut a = CBuffer::<f32>::new(16).unwrap();
        let mut b = CBuffer::<f32>::new(16).unwrap();

        a.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let slice = b.reserve(4).unwrap();
        slice.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        b.commit(4).unwrap();

        assert_eq!(a.size(), b.size());
        assert_eq!(a.read(4), b.read(4));

        // committing fewer than reserved keeps only what was committed
        let slice = b.reserve(4).unwrap();
        slice.copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        b.commit(2).unwrap();
        assert_eq!(b.size(), 6);
        assert_eq!(b.read(6), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // reserving nothing is fine
        assert!(b.reserve(0).unwrap().is_empty());

        // asking for more than fits is clamped
        assert_eq!(b.reserve(11).unwrap().len(), 10);
    }

    // a reservation is a plain slice, so it stops at the end of the ring and the
    // caller picks up the remainder on the next reserve
    #[test]
    fn test_cbuffer_reserve_truncates_at_wrap() {
        let mut q = CBuffer::<f32>::new(8).unwrap();

        // push the write index to 6, leaving 2 contiguous slots of 8 available
        q.write(&[0.0; 6]).unwrap();
        q.release(6).unwrap();
        assert_eq!(q.space_available(), 8);

        assert_eq!(q.reserve(2).unwrap().len(), 2);
        // truncated at the wrap rather than rejected
        assert_eq!(q.reserve(3).unwrap().len(), 2);
        assert_eq!(q.reserve(8).unwrap().len(), 2);

        // filling in two reservations covers the whole ring
        let first = {
            let slice = q.reserve(8).unwrap();
            for (i, s) in slice.iter_mut().enumerate() {
                *s = i as f32;
            }
            slice.len()
        };
        q.commit(first).unwrap();

        let second = {
            let slice = q.reserve(8 - first).unwrap();
            for (i, s) in slice.iter_mut().enumerate() {
                *s = (first + i) as f32;
            }
            slice.len()
        };
        q.commit(second).unwrap();

        assert_eq!(first + second, 8);
        assert!(q.is_full());
        let expected: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert_eq!(q.read(8), &expected[..]);
    }

    // a full buffer errors rather than handing back an empty reservation
    #[test]
    fn test_cbuffer_reserve_full_errors() {
        let mut q = CBuffer::<f32>::new(4).unwrap();
        q.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(q.is_full());

        assert!(q.reserve(1).is_err());
        // asking for nothing is still fine
        assert!(q.reserve(0).unwrap().is_empty());

        // once space frees up, reserving works again
        q.release(2).unwrap();
        assert!(!q.reserve(1).unwrap().is_empty());
    }

    // occupancy accounting must hold under an arbitrary mix of operations
    #[test]
    fn test_cbuffer_occupancy_invariants() {
        let mut q = CBuffer::<f32>::new(10).unwrap();
        let mut expected: std::collections::VecDeque<f32> = std::collections::VecDeque::new();

        // a fixed, deliberately awkward schedule of writes and reads
        let ops: &[(usize, usize)] = &[
            (7, 3), (5, 8), (4, 0), (6, 9), (10, 10), (1, 1), (9, 5), (3, 7),
        ];

        let mut next = 0.0f32;
        for &(to_write, to_read) in ops {
            let n = to_write.min(q.space_available());
            let batch: Vec<f32> = (0..n).map(|i| next + i as f32).collect();
            q.write(&batch).unwrap();
            next += n as f32;
            expected.extend(batch);

            assert_eq!(q.size(), expected.len());
            assert_eq!(q.space_available(), 10 - expected.len());
            assert_eq!(q.is_empty(), expected.is_empty());
            assert_eq!(q.is_full(), expected.len() == 10);

            let got = q.read(to_read).to_vec();
            for (i, &value) in got.iter().enumerate() {
                assert_eq!(value, expected[i]);
            }
            q.release(got.len()).unwrap();
            for _ in 0..got.len() {
                expected.pop_front();
            }

            assert_eq!(q.size(), expected.len());
        }
    }

}
