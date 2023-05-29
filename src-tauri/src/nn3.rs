use super::emit;
use crate::data::{EPOCHS, EPOCHS_PER_PRINT, LEARN_RATE, STEP, TRAINING_DATA};
use crate::math::sigmoid;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug)]
struct XorGate {
    or_w1: f32,
    or_w2: f32,
    or_bias: f32,

    nand_w1: f32,
    nand_w2: f32,
    nand_bias: f32,

    and_w1: f32,
    and_w2: f32,
    and_bias: f32,
}

// Returns the values that we need to subtract from the original model to drive it towards the local minimum of the cost function
fn finite_diff(m: &mut XorGate) -> XorGate {
    let mut g = XorGate {
        or_w1: 0.0,
        or_w2: 0.0,
        or_bias: 0.0,

        nand_w1: 0.0,
        nand_w2: 0.0,
        nand_bias: 0.0,

        and_w1: 0.0,
        and_w2: 0.0,
        and_bias: 0.0,
    };
    let original_cost = cost(m);

    let saved = m.or_w1;
    m.or_w1 += STEP;
    g.or_w1 = (cost(m) - original_cost) / STEP;
    m.or_w1 = saved;

    let saved = m.or_w2;
    m.or_w2 += STEP;
    g.or_w2 = (cost(m) - original_cost) / STEP;
    m.or_w2 = saved;

    let saved = m.or_bias;
    m.or_bias += STEP;
    g.or_bias = (cost(m) - original_cost) / STEP;
    m.or_bias = saved;

    let saved = m.nand_w1;
    m.nand_w1 += STEP;
    g.nand_w1 = (cost(m) - original_cost) / STEP;
    m.nand_w1 = saved;

    let saved = m.nand_w2;
    m.nand_w2 += STEP;
    g.nand_w2 = (cost(m) - original_cost) / STEP;
    m.nand_w2 = saved;

    let saved = m.nand_bias;
    m.nand_bias += STEP;
    g.nand_bias = (cost(m) - original_cost) / STEP;
    m.nand_bias = saved;

    let saved = m.and_w1;
    m.and_w1 += STEP;
    g.and_w1 = (cost(m) - original_cost) / STEP;
    m.and_w1 = saved;

    let saved = m.and_w2;
    m.and_w2 += STEP;
    g.and_w2 = (cost(m) - original_cost) / STEP;
    m.and_w2 = saved;

    let saved = m.and_bias;
    m.and_bias += STEP;
    g.and_bias = (cost(m) - original_cost) / STEP;
    m.and_bias = saved;

    return g;
}

fn train(m: &mut XorGate, g: &XorGate) {
    m.or_w1 -= g.or_w1 * LEARN_RATE;
    m.or_w2 -= g.or_w2 * LEARN_RATE;
    m.or_bias -= g.or_bias * LEARN_RATE;

    m.nand_w1 -= g.nand_w1 * LEARN_RATE;
    m.nand_w2 -= g.nand_w2 * LEARN_RATE;
    m.nand_bias -= g.nand_bias * LEARN_RATE;

    m.and_w1 -= g.and_w1 * LEARN_RATE;
    m.and_w2 -= g.and_w2 * LEARN_RATE;
    m.and_bias -= g.and_bias * LEARN_RATE;
}

fn cost(m: &XorGate) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAINING_DATA.len() {
        let x1 = TRAINING_DATA[i][0];
        let x2 = TRAINING_DATA[i][1];
        let y = forward(m, x1, x2);
        let expected = TRAINING_DATA[i][TRAINING_DATA.len() - 2];
        let d = y - expected;
        result += d * d;
    }
    result /= TRAINING_DATA.len() as f32;
    result
}

fn forward(m: &XorGate, x1: f32, x2: f32) -> f32 {
    // First layer
    let a = sigmoid((m.or_w1 * x1) + (m.or_w2 * x2) + m.or_bias);
    let b = sigmoid((m.nand_w1 * x1) + (m.nand_w2 * x2) + m.nand_bias);
    // Second layer
    sigmoid((a * m.and_w1) + (b * m.and_w2) + m.and_bias)
}

pub(crate) fn run(window: &tauri::Window) {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);

    emit(window, format!("Seed {}", seed));

    let mut m = XorGate {
        or_w1: rng.gen(),
        or_w2: rng.gen(),
        or_bias: rng.gen(),
        nand_w1: rng.gen(),
        nand_w2: rng.gen(),
        nand_bias: rng.gen(),
        and_w1: rng.gen(),
        and_w2: rng.gen(),
        and_bias: rng.gen(),
    };

    for epoch in 1..=EPOCHS {
        let g = finite_diff(&mut m);
        train(&mut m, &g);
        let new_cost = cost(&m);
        if epoch % EPOCHS_PER_PRINT == 0 {
            emit(window, format!("Cost after epoch {}: {}", epoch, new_cost));
        }
    }
    format!("Final cost: {}", cost(&m));

    emit(window, "<hr/>");

    // XOR?
    for i in 0..2 {
        for j in 0..2 {
            let x1 = i as f32;
            let x2 = j as f32;
            let y = forward(&m, x1, x2);
            emit(window, format!("?XOR({},{}) = {}", x1, x2, y));
        }
    }

    emit(window, "<hr/>");

    // OR?
    for i in 0..2 {
        for j in 0..2 {
            let val = sigmoid((m.or_w1 * i as f32) + (m.or_w2 * j as f32) + m.or_bias);
            emit(window, format!("?OR({}, {}) = {}", i, j, val));
        }
    }

    emit(window, "<hr/>");

    // NAND?
    for i in 0..2 {
        for j in 0..2 {
            let val = sigmoid((m.nand_w1 * i as f32) + (m.nand_w2 * j as f32) + m.nand_bias);
            emit(window, format!("?NAND({}, {}) = {}", i, j, val));
        }
    }

    emit(window, "<hr/>");

    // AND?
    for i in 0..2 {
        for j in 0..2 {
            let val = sigmoid((m.and_w1 * i as f32) + (m.and_w2 * j as f32) + m.and_bias);
            emit(window, format!("?AND({}, {}) = {}", i, j, val));
        }
    }
}
