use crate::data::RNG_SEED;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug)]
pub(crate) struct Mat {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) data: Vec<f32>,
}

#[macro_export]
macro_rules! MAT_AT {
    ($mat:expr, $row:expr, $col:expr) => {
        $mat.data[$row * $mat.columns + $col]
    };
}

#[macro_export]
macro_rules! MAT_PRINT {
    ($mat:expr, $decimals:literal) => {
        $mat.print($decimals, stringify!($mat));
    };
}

impl Mat {
    pub(crate) fn new(rows: usize, columns: usize) -> Mat {
        Mat {
            rows,
            columns,
            data: vec![0.0; rows * columns],
        }
    }

    pub(crate) fn print(&self, decimals: usize, name: &str) {
        println!("{} = [", name);
        for row in 0..self.rows {
            for col in 0..self.columns {
                print!("  {:.*} ", decimals, MAT_AT!(self, row, col));
            }
            println!();
        }
        println!("]");
    }

    pub(crate) fn fill(&mut self, value: f32) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                MAT_AT!(self, row, col) = value;
            }
        }
    }

    pub(crate) fn fill_rand(&mut self, low: f32, high: f32) {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        for index in 0..self.data.len() {
            self.data[index] = rng.gen_range(low..high);
        }
    }

    pub(crate) fn add(&mut self, other: &Mat) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);

        for row in 0..self.rows {
            for col in 0..self.columns {
                MAT_AT!(self, row, col) += MAT_AT!(other, row, col);
            }
        }
    }

    pub(crate) fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows);
        assert!(col < self.columns);

        MAT_AT!(self, row, col) = value;
    }

    pub(crate) fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows);
        assert!(col < self.columns);

        MAT_AT!(self, row, col)
    }

    pub(crate) fn sigmoid(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                MAT_AT!(self, row, col) = sigmoidf(MAT_AT!(self, row, col));
            }
        }
    }
}

pub(crate) fn mat_dot(dst: &mut Mat, a: &Mat, b: &Mat) {
    assert_eq!(a.columns, b.rows);
    assert_eq!(dst.rows, a.rows);
    assert_eq!(dst.columns, b.columns);

    for row in 0..dst.rows {
        for col in 0..dst.columns {
            let mut sum = 0.0;
            for k in 0..a.columns {
                sum += MAT_AT!(a, row, k) * MAT_AT!(b, k, col);
            }
            MAT_AT!(dst, row, col) = sum;
        }
    }
}

pub(crate) fn mat_sum(dst: &mut Mat, a: &Mat, b: &Mat) {
    assert_eq!(dst.rows, a.rows);
    assert_eq!(dst.columns, a.columns);
    assert_eq!(dst.rows, b.rows);
    assert_eq!(dst.columns, b.columns);

    for row in 0..dst.rows {
        for col in 0..dst.columns {
            MAT_AT!(dst, row, col) = MAT_AT!(a, row, col) + MAT_AT!(b, row, col);
        }
    }
}

pub(crate) fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
