use crate::error::Result;

#[derive(Debug, Clone)]
pub struct WDelay<T> {
    v: Vec<T>,
    delay: usize,
    read_index: usize,
}

impl<T: Default + Clone + Copy> WDelay<T> {
    pub fn create(delay: usize) -> Result<Self> {
        let mut wdelay = WDelay {
            v: vec![T::default(); delay + 1],
            delay,
            read_index: 0,
        };

        wdelay.reset();
        Ok(wdelay)
    }

    pub fn recreate(&mut self, delay: usize) -> Result<()> {
        if delay == self.delay {
            return Ok(());
        }

        let mut vtmp = Vec::with_capacity(self.delay + 1);
        for i in 0..=self.delay {
            vtmp.push(self.v[(i + self.read_index) % (self.delay + 1)]);
        }

        *self = WDelay::create(delay)?;

        for v in vtmp.iter() {
            self.push(*v);
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.read_index = 0;
        self.v.fill(T::default());
    }

    pub fn read(&self) -> T {
        self.v[self.read_index]
    }

    pub fn push(&mut self, value: T) {
        self.v[self.read_index] = value;
        self.read_index = (self.read_index + 1) % (self.delay + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;

    #[test]
    #[autotest_annotate(autotest_wdelayf)]
    fn test_wdelayf() {
        // create wdelay
        // wdelay: 0 0 0 0 0
        let mut w = WDelay::<f32>::create(4).unwrap();

        assert_relative_eq!(w.read(), 0.0);

        let x0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y0_test = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y0 = [0.0; 10];

        for i in 0..10 {
            w.push(x0[i]);
            y0[i] = w.read();
            //println!("{:3} : {:6.2} ({:6.2})", i, y0[i], y0_test[i]);
        }
        // 6 7 8 9 10
        assert_eq!(y0, y0_test);

        // re-create wdelay object
        // wdelay: 0 0 6 7 8 9 10
        w.recreate(6).unwrap();

        let x1 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 2.0, 2.0];
        let y1_test = [0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 3.0, 4.0, 5.0, 6.0];
        let mut y1 = [0.0; 10];
        for i in 0..10 {
            w.push(x1[i]);
            y1[i] = w.read();
            //println!("{:3} : {:6.2} ({:6.2})", i, y1[i], y1_test[i]);
        }
        // wdelay: 6 7 8 9 2 2 2
        assert_eq!(y1, y1_test);

        // re-create wdelay object
        // wdelay: 7 8 9 2 2 2
        w.recreate(5).unwrap();

        let x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        let y2_test = [8.0, 9.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut y2 = [0.0; 10];

        for i in 0..10 {
            w.push(x2[i]);
            y2[i] = w.read();
            //println!("{:3} : {:6.2} ({:6.2})", i, y2[i], y2_test[i]);
        }
        // wdelay: 1 1 1 2 3 4
        assert_eq!(y2, y2_test);

        // TODO: Implement print method if needed
        // assert_eq!(wdelayf_print(w), LIQUID_OK);
    }

    #[test]
    #[autotest_annotate(autotest_wdelay_copy)]
    fn test_wdelay_copy() {
        // create base object
        let delay = 20;
        let mut q0 = WDelay::<Complex<f32>>::create(delay).unwrap();

        // write some values
        // TODO randnf()
        for _ in 0..delay {
            let v = Complex::new(rand::random::<f32>(), rand::random::<f32>());
            q0.push(v);
        }

        // copy object
        let mut q1 = q0.clone();

        // write a few more values
        // TODO randnf()
        for _ in 0..64 {
            let v = Complex::new(rand::random::<f32>(), rand::random::<f32>());
            q0.push(v);
            q1.push(v);

            let y0 = q0.read().norm();
            let y1 = q1.read().norm();
            assert_relative_eq!(y0, y1);
        }
    }
}