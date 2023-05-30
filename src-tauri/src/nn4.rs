use crate::{
    brain::Mat,
    data::{EPOCHS, EPOCHS_PER_PRINT, LEARN_RATE, STEP},
    emit,
};

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

pub(crate) fn mat_dot(dst: &mut Mat, a: &Mat, b: &Mat) {
    assert_eq!(a.columns, b.rows);
    assert_eq!(dst.rows, a.rows);
    assert_eq!(dst.columns, b.columns);

    for row in 0..dst.rows {
        for col in 0..dst.columns {
            let mut sum = 0.0;
            for k in 0..a.columns {
                sum += a.get(row, k) * b.get(k, col);
            }
            dst.set(row, col, sum);
        }
    }
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

        for row in 0..ti.rows {
            let truth_in = ti.row(row);
            let truth_out = to.row(row);

            self.a0 = truth_in;
            self.forward();

            for col in 0..to.columns {
                let d = self.a2.get(0, col) - truth_out.get(0, col);
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
            saved = m.w1.get(i, j);
            m.w1.set(i, j, saved + STEP);
            let c2 = m.cost(ti, to);
            g.w1.set(i, j, (c2 - c) / STEP);
            m.w1.set(i, j, saved);
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.columns {
            saved = m.b1.get(i, j);
            m.b1.set(i, j, saved + STEP);
            let c2 = m.cost(ti, to);
            g.b1.set(i, j, (c2 - c) / STEP);
            m.b1.set(i, j, saved);
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.columns {
            saved = m.w2.get(i, j);
            m.w2.set(i, j, saved + STEP);
            let c2 = m.cost(ti, to);
            g.w2.set(i, j, (c2 - c) / STEP);
            m.w2.set(i, j, saved);
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.columns {
            saved = m.b2.get(i, j);
            m.b2.set(i, j, saved + STEP);
            let c2 = m.cost(ti, to);
            g.b2.set(i, j, (c2 - c) / STEP);
            m.b2.set(i, j, saved);
        }
    }
}

fn xor_learn(m: &mut Xor, g: &mut Xor) {
    for i in 0..m.w1.rows {
        for j in 0..m.w1.columns {
            m.w1.set(i, j, m.w1.get(i, j) - g.w1.get(i, j) * LEARN_RATE);
        }
    }

    for i in 0..m.b1.rows {
        for j in 0..m.b1.columns {
            m.b1.set(i, j, m.b1.get(i, j) - g.b1.get(i, j) * LEARN_RATE);
        }
    }

    for i in 0..m.w2.rows {
        for j in 0..m.w2.columns {
            m.w2.set(i, j, m.w2.get(i, j) - g.w2.get(i, j) * LEARN_RATE);
        }
    }

    for i in 0..m.b2.rows {
        for j in 0..m.b2.columns {
            m.b2.set(i, j, m.b2.get(i, j) - g.b2.get(i, j) * LEARN_RATE);
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

    emit(window, "<hr>");
    emit(window, "Validation");
    emit(window, "<hr>");

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
