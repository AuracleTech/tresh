use crate::data::RNG_SEED;
use crate::data::{LEARN_RATE, STEP};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        Matrix {
            rows,
            columns,
            data: vec![0.; rows * columns],
        }
    }

    pub fn fill(&mut self, value: f32) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, value);
            }
        }
    }

    pub fn fill_rand(&mut self, low: f32, high: f32) {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        for index in 0..self.data.len() {
            self.data[index] = rng.gen_range(low..high);
        }
    }

    pub fn add(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);

        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, self.get(row, col) + other.get(row, col));
            }
        }
    }

    pub fn sub(&mut self, other: &Matrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.columns, other.columns);

        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, self.get(row, col) - other.get(row, col));
            }
        }
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows);
        assert!(col < self.columns);

        self.data[row * self.columns + col] = value;
    }

    // FIX REPLACE SET AND GET BY ONLY ONE FUNCTION CALLED AT
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows);
        assert!(col < self.columns);

        self.data[row * self.columns + col]
    }

    pub fn sigmoid(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, sigmoid(self.get(row, col)));
            }
        }
    }

    pub fn row(&self, row: usize) -> Matrix {
        let mut result = Matrix::new(1, self.columns);

        for col in 0..self.columns {
            result.set(0, col, self.get(row, col));
        }

        result
    }

    pub fn dot(&mut self, other: &Matrix) {
        assert_eq!(self.columns, other.rows);

        let mut result = Matrix::new(self.rows, other.columns);

        for row in 0..self.rows {
            for col in 0..other.columns {
                let mut sum = 0.;
                for k in 0..self.columns {
                    sum += self.get(row, k) * other.get(k, col);
                }
                result.set(row, col, sum);
            }
        }

        *self = result;
    }

    pub fn dotf(&mut self, value: f32) {
        for row in 0..self.rows {
            for col in 0..self.columns {
                self.set(row, col, self.get(row, col) * value);
            }
        }
    }

    pub fn from_2d_vec(data: &Vec<Vec<f32>>) -> Matrix {
        let rows = data.len();
        let columns = data[0].len();

        let mut result = Matrix::new(rows, columns);

        for row in 0..rows {
            for col in 0..columns {
                result.set(row, col, data[row][col]);
            }
        }

        result
    }
}

// TODO improve display for matrices
impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();

        for row in 0..self.rows {
            if self.rows > 1 {
                result += "\n";
            }
            for col in 0..self.columns {
                result += &format!("{:.2} ", self.get(row, col));
            }
        }

        write!(f, "{}", result)
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    w: Vec<Matrix>,
    b: Vec<Matrix>,
    a: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(arch: &[usize]) -> Self {
        assert!(arch.len() > 1);
        assert!(arch[..].iter().all(|&x| x > 0));

        let mut brain = Self {
            w: Vec::new(),
            b: Vec::new(),
            a: Vec::new(),
        };

        brain.a.push(Matrix::new(1, arch[0]));

        for i in 1..arch.len() {
            brain.w.push(Matrix::new(arch[i - 1], arch[i]));
            brain.b.push(Matrix::new(1, arch[i]));
            brain.a.push(Matrix::new(1, arch[i]));
        }

        brain
    }

    pub fn rand(&mut self, low: f32, high: f32) {
        for i in 0..self.w.len() {
            self.w[i].fill_rand(low, high);
            self.b[i].fill_rand(low, high);
        }
    }

    // TODO change for impl Display
    pub fn to_string(&mut self) -> String {
        let mut result = "[\n".to_string();
        for i in 0..self.w.len() {
            result += &format!("_ _ a[{}] {:?}\n", i, self.a[i]);
            result += "\n";
            result += &format!("_ _ w[{}] {:?}\n", i, self.w[i]);
            result += &format!("_ _ b[{}] {:?}\n", i, self.b[i]);
        }
        result += &format!("_ _ a[{}] {:?}\n", self.a.len() - 1, self.output());
        result += "]";
        result
    }

    pub fn forward(&mut self) {
        for i in 0..self.w.len() {
            self.a[i + 1] = self.a[i].clone();
            self.a[i + 1].dot(&self.w[i]);

            self.a[i + 1].add(&self.b[i]);

            self.a[i + 1].sigmoid();
        }
    }

    pub fn input(&mut self, input: &Matrix) {
        assert_eq!(input.rows, 1);
        assert_eq!(input.columns, self.w[0].rows);

        self.a[0] = input.clone();
    }

    pub fn output(&mut self) -> &mut Matrix {
        let layers = self.a.len();
        &mut self.a[layers - 1]
    }

    pub fn cost(&mut self, truth_in: &Matrix, truth_out: &Matrix) -> f32 {
        assert_eq!(truth_in.rows, truth_out.rows);
        assert_eq!(truth_out.columns, self.output().columns);

        let mut cost = 0.;

        for row in 0..truth_in.rows {
            let truth_in = truth_in.row(row);
            let truth_out = truth_out.row(row);

            self.input(&truth_in);
            self.forward();

            for col in 0..truth_out.columns {
                let d = self.output().get(0, col) - truth_out.get(0, col);
                cost += d * d;
            }
        }

        cost / truth_in.rows as f32
    }

    pub fn finite_diff(&mut self, grad: &mut NeuralNetwork, truth_in: &Matrix, truth_out: &Matrix) {
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

    pub fn backprop(&mut self, g: &mut NeuralNetwork, ti: &Matrix, to: &Matrix) {
        assert_eq!(ti.rows, to.rows);
        assert_eq!(self.output().columns, to.columns);
        let n: usize = ti.rows;

        g.fill(0.);

        // i - current sample
        // l - current layer
        // j - current activation
        // k - previous activation

        for i in 0..n {
            self.input(&ti.row(i));
            self.forward();

            for j in 0..=self.w.len() {
                g.a[j].fill(0.);
            }

            for j in 0..to.columns {
                let d = self.output().get(0, j) - to.get(i, j);
                g.output().set(0, j, d);
            }

            // TEST for (size_t l = nn.count; l > 0; --l) {
            for l in (1..=self.w.len()).rev() {
                for j in 0..self.a[l].columns {
                    let a = self.a[l].get(0, j);
                    let da = g.a[l].get(0, j);

                    let mut nb = g.b[l - 1].get(0, j);
                    nb += 2. * da * a * (1. - a);
                    g.b[l - 1].set(0, j, nb);

                    for k in 0..self.a[l - 1].columns {
                        // j - weight matrix col
                        // k - weight matrix row
                        let pa = self.a[l - 1].get(0, k);
                        let w = self.w[l - 1].get(k, j);

                        let mut naw = g.w[l - 1].get(k, j);
                        naw += 2. * da * a * (1. - a) * pa;
                        g.w[l - 1].set(k, j, naw);

                        let mut na = g.a[l - 1].get(0, k);
                        na += 2. * da * a * (1. - a) * w;
                        g.a[l - 1].set(0, k, na);
                    }
                }
            }
        }

        for i in 0..g.w.len() {
            for j in 0..g.w[i].rows {
                for k in 0..g.w[i].columns {
                    let mut nw = g.w[i].get(j, k);
                    nw /= n as f32;
                    g.w[i].set(j, k, nw);
                }
            }

            for j in 0..g.b[i].rows {
                for k in 0..g.b[i].columns {
                    let mut nb = g.b[i].get(j, k);
                    nb /= n as f32;
                    g.b[i].set(j, k, nb);
                }
            }
        }
    }

    pub fn learn(&mut self, grad: &mut NeuralNetwork) {
        for layer_index in 0..self.w.len() {
            grad.w[layer_index].dotf(LEARN_RATE);
            self.w[layer_index].sub(&grad.w[layer_index]);
            grad.b[layer_index].dotf(LEARN_RATE);
            self.b[layer_index].sub(&grad.b[layer_index]);
        }
    }

    pub fn fill(&mut self, value: f32) {
        for i in 0..self.w.len() {
            self.w[i].fill(value);
            self.b[i].fill(value);
            self.a[i].fill(value);
        }
        let i = self.a.len() - 1;
        self.a[i].fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nn_finite_diff_xor_gate() {
        const EPOCHS: usize = 20_000;

        let truth_in = Matrix::from_2d_vec(&vec![
            vec![0., 0.],
            vec![1., 0.],
            vec![0., 1.],
            vec![1., 1.],
        ]);
        let truth_out = Matrix::from_2d_vec(&vec![
            vec![0.], //
            vec![1.], //
            vec![1.], //
            vec![0.], //
        ]);

        let arch = [2, 2, 1];
        let mut nn = NeuralNetwork::new(&arch);
        let mut grad = NeuralNetwork::new(&arch);

        nn.rand(0., 1.);
        let cost_init = nn.cost(&truth_in, &truth_out);
        assert!(cost_init >= 0.0);

        for _epoch in 1..=EPOCHS {
            nn.finite_diff(&mut grad, &truth_in, &truth_out);
            nn.learn(&mut grad);
        }

        for row in 0..truth_in.rows {
            let truth_in = truth_in.row(row);
            let truth_out = truth_out.row(row);

            nn.input(&truth_in);
            nn.forward();

            let cost = nn.cost(&truth_in, &truth_out);
            assert!(cost < cost_init);
            assert!(cost < 0.01);
        }
    }
}
