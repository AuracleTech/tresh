use crate::data::RNG_SEED;
use crate::data::{LEARN_RATE, STEP};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub(crate) struct Mat {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub(crate) fn new(rows: usize, columns: usize) -> Mat {
        Mat {
            rows,
            columns,
            data: vec![0.0; rows * columns],
        }
    }

    pub(crate) fn fill(&mut self, value: f32) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, value);
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
                self.set(row, col, self.get(row, col) + other.get(row, col));
            }
        }
    }

    pub(crate) fn sub(&mut self, other: &Mat) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);

        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, self.get(row, col) - other.get(row, col));
            }
        }
    }

    pub(crate) fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows);
        assert!(col < self.columns);

        self.data[row * self.columns + col] = value;
    }

    pub(crate) fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows);
        assert!(col < self.columns);

        self.data[row * self.columns + col]
    }

    pub(crate) fn sigmoid(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, sigmoid(self.get(row, col)));
            }
        }
    }

    pub(crate) fn row(&self, row: usize) -> Mat {
        let mut result = Mat::new(1, self.columns);

        for col in 0..self.columns {
            result.set(0, col, self.get(row, col));
        }

        result
    }

    pub(crate) fn dot(&mut self, other: &Mat) {
        assert_eq!(self.columns, other.rows);

        let mut result = Mat::new(self.rows, other.columns);

        for row in 0..self.rows {
            for col in 0..other.columns {
                let mut sum = 0.0;
                for k in 0..self.columns {
                    sum += self.get(row, k) * other.get(k, col);
                }
                result.set(row, col, sum);
            }
        }

        *self = result;
    }

    pub(crate) fn dotf(&mut self, value: f32) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, self.get(row, col) * value);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct Brain {
    w: Vec<Mat>,
    b: Vec<Mat>,
    a: Vec<Mat>,
}

impl Brain {
    pub(crate) fn new(arch: &[usize]) -> Self {
        assert!(arch.len() > 1);
        assert!(arch[..].iter().all(|&x| x > 0));

        let mut brain = Self {
            w: Vec::new(),
            b: Vec::new(),
            a: Vec::new(),
        };

        brain.a.push(Mat::new(1, arch[0]));

        for i in 1..arch.len() {
            brain.w.push(Mat::new(arch[i - 1], arch[i]));
            brain.b.push(Mat::new(1, arch[i]));
            brain.a.push(Mat::new(1, arch[i]));
        }

        brain
    }

    pub(crate) fn rand(&mut self, low: f32, high: f32) {
        for i in 0..self.w.len() {
            self.w[i].fill_rand(low, high);
            self.b[i].fill_rand(low, high);
        }
    }

    pub(crate) fn print(&self) {
        println!("Brain = [");
        for i in 0..self.w.len() {
            println!("    Layer {} = [", i);
            println!("        w = {:?}", self.w[i]);
            println!("        b = {:?}", self.b[i]);
            println!("    ]");
        }
        println!("]");
    }

    pub(crate) fn forward(&mut self) {
        for i in 0..self.w.len() {
            self.a[i + 1] = self.a[i].clone();
            self.a[i + 1].dot(&self.w[i]);

            self.a[i + 1].add(&self.b[i]);

            self.a[i + 1].sigmoid();
        }
    }

    pub(crate) fn input(&mut self, input: &Mat) {
        assert_eq!(input.rows, 1);
        assert_eq!(input.columns, self.w[0].rows);

        self.a[0] = input.clone();
    }

    pub(crate) fn output(&self) -> &Mat {
        &self.a[self.a.len() - 1]
    }

    pub(crate) fn cost(&mut self, ti: &Mat, to: &Mat) -> f32 {
        assert_eq!(ti.rows, to.rows);
        assert_eq!(to.columns, self.output().columns);

        let mut cost = 0.0;

        for row in 0..ti.rows {
            let truth_in = ti.row(row);
            let truth_out = to.row(row);

            self.input(&truth_in);
            self.forward();

            for col in 0..to.columns {
                let d = self.output().get(0, col) - truth_out.get(0, col);
                cost += d * d;
            }
        }

        cost / ti.rows as f32
    }

    pub(crate) fn finite_diff(&mut self, grad: &mut Brain, truth_in: &Mat, truth_out: &Mat) {
        let mut saved;

        let cost_start = self.cost(truth_in, truth_out);

        for layer_index in 0..self.w.len() {
            for row in 0..self.w[layer_index].rows {
                for col in 0..self.w[layer_index].columns {
                    saved = self.w[layer_index].get(row, col);
                    self.w[layer_index].set(row, col, saved + STEP);
                    let cost_new = self.cost(truth_in, truth_out);
                    grad.w[layer_index].set(row, col, (cost_new - cost_start) / STEP);
                    self.w[layer_index].set(row, col, saved);
                }
            }

            for row in 0..self.b[layer_index].rows {
                for col in 0..self.b[layer_index].columns {
                    saved = self.b[layer_index].get(row, col);
                    self.b[layer_index].set(row, col, saved + STEP);
                    let cost_new = self.cost(truth_in, truth_out);
                    grad.b[layer_index].set(row, col, (cost_new - cost_start) / STEP);
                    self.b[layer_index].set(row, col, saved);
                }
            }
        }
    }

    pub(crate) fn learn(&mut self, grad: &mut Brain) {
        for layer_index in 0..self.w.len() {
            grad.w[layer_index].dotf(LEARN_RATE);
            self.w[layer_index].sub(&grad.w[layer_index]);
            grad.b[layer_index].dotf(LEARN_RATE);
            self.b[layer_index].sub(&grad.b[layer_index]);
        }
    }
}

// TODO optimize step depending on the new cost
