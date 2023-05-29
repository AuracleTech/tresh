#[allow(dead_code)]
mod perco;
use crate::{
    data::{EPOCHS, EPOCHS_PER_PRINT, LEARN_RATE, STEP},
    emit,
    nn::perco::mat_dot,
    MAT_AT,
};
use perco::Mat;

#[derive(Debug)]
struct Xor {
    a0: Mat,

    w1: Mat,
    b1: Mat,
    a1: Mat,

    w2: Mat,
    b2: Mat,
    a2: Mat,
}

fn mat_row(mat: &Mat, row: usize) -> Mat {
    let mut result = Mat::new(1, mat.columns);

    for col in 0..mat.columns {
        MAT_AT!(result, 0, col) = MAT_AT!(mat, row, col);
    }

    result
}

impl Xor {
    pub(crate) fn forward(&mut self) {
        // First layer forward pass
        mat_dot(&mut self.a1, &self.a0, &self.w1);
        self.a1.add(&self.b1);
        self.a1.sigmoid();

        // Second layer forward pass
        mat_dot(&mut self.a2, &self.a1, &self.w2);
        self.a2.add(&self.b2);
        self.a2.sigmoid();
    }

    pub(crate) fn cost(&mut self, ti: &Mat, to: &Mat) -> f32 {
        assert_eq!(ti.rows, to.rows);
        assert_eq!(to.columns, self.a2.columns);

        let mut cost = 0.0;

        for i in 0..ti.rows {
            let x = mat_row(&ti, i);
            let y = mat_row(&to, i);

            self.a0 = x;
            self.forward();

            for j in 0..to.columns {
                let d = MAT_AT!(self.a2, 0, j) - MAT_AT!(y, 0, j);
                cost += d * d;
            }
        }

        cost / ti.rows as f32
    }
}

fn finite_difference(m: &mut Xor, g: &mut Xor, ti: &Mat, to: &Mat) {
    let mut saved;

    let c = m.cost(ti, to);

    for i in 0..m.w1.rows {
        for j in 0..m.w1.columns {
            saved = MAT_AT!(m.w1, i, j);
            MAT_AT!(m.w1, i, j) = saved + STEP;
            let c2 = m.cost(ti, to);
            MAT_AT!(g.w1, i, j) = (c2 - c) / STEP;
            MAT_AT!(m.w1, i, j) = saved;
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.columns {
            saved = MAT_AT!(m.b1, i, j);
            MAT_AT!(m.b1, i, j) = saved + STEP;
            let c2 = m.cost(ti, to);
            MAT_AT!(g.b1, i, j) = (c2 - c) / STEP;
            MAT_AT!(m.b1, i, j) = saved;
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.columns {
            saved = MAT_AT!(m.w2, i, j);
            MAT_AT!(m.w2, i, j) = saved + STEP;
            let c2 = m.cost(ti, to);
            MAT_AT!(g.w2, i, j) = (c2 - c) / STEP;
            MAT_AT!(m.w2, i, j) = saved;
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.columns {
            saved = MAT_AT!(m.b2, i, j);
            MAT_AT!(m.b2, i, j) = saved + STEP;
            let c2 = m.cost(ti, to);
            MAT_AT!(g.b2, i, j) = (c2 - c) / STEP;
            MAT_AT!(m.b2, i, j) = saved;
        }
    }
}

fn xor_learn(m: &mut Xor, g: &mut Xor) {
    for i in 0..m.w1.rows {
        for j in 0..m.w1.columns {
            MAT_AT!(m.w1, i, j) -= MAT_AT!(g.w1, i, j) * LEARN_RATE;
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.columns {
            MAT_AT!(m.b1, i, j) -= MAT_AT!(g.b1, i, j) * LEARN_RATE;
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.columns {
            MAT_AT!(m.w2, i, j) -= MAT_AT!(g.w2, i, j) * LEARN_RATE;
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.columns {
            MAT_AT!(m.b2, i, j) -= MAT_AT!(g.b2, i, j) * LEARN_RATE;
        }
    }
}

pub(crate) fn run(window: &tauri::Window) {
    // Truth table
    let td = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ];

    // Input (from truth table)
    let mut ti = Mat::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            ti.set(i, j, td[i][j]);
        }
    }

    // Target (from truth table)
    let mut to = Mat::new(4, 1);
    for i in 0..4 {
        to.set(i, 0, td[i][2]);
    }

    // Network
    let mut m = Xor {
        a0: Mat::new(1, 2),
        w1: Mat::new(2, 2),
        b1: Mat::new(1, 2),
        a1: Mat::new(1, 2),
        w2: Mat::new(2, 1),
        b2: Mat::new(1, 1),
        a2: Mat::new(1, 1),
    };
    let mut g: Xor = Xor {
        a0: Mat::new(1, 2),
        w1: Mat::new(2, 2),
        b1: Mat::new(1, 2),
        a1: Mat::new(1, 2),
        w2: Mat::new(2, 1),
        b2: Mat::new(1, 1),
        a2: Mat::new(1, 1),
    };

    m.w1.fill_rand(0.0, 1.0);
    m.b1.fill_rand(0.0, 1.0);

    m.w2.fill_rand(0.0, 1.0);
    m.b2.fill_rand(0.0, 1.0);

    // Print cost
    let c = m.cost(&ti, &to);
    emit(window, format!("Cost = {}", c));

    for epoch in 1..=EPOCHS {
        finite_difference(&mut m, &mut g, &ti, &to);
        xor_learn(&mut m, &mut g);
        if epoch % EPOCHS_PER_PRINT == 0 {
            let cost = m.cost(&ti, &to);
            emit(window, format!("Cost after epoch {}: {}", epoch, cost));
        }
    }

    let c = m.cost(&ti, &to);
    emit(window, format!("Cost = {}", c));

    // Validate
    for i in 0..2 {
        for j in 0..2 {
            m.a0.set(0, 0, i as f32);
            m.a0.set(0, 1, j as f32);

            m.forward();

            let y = m.a2.get(0, 0);

            emit(window, format!("{} XOR {} = {}", i, j, y));
        }
    }
}
