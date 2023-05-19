use super::{emit, EPOCHS_PER_UPDATE};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

const TRAIN_LEN: usize = 4;
const OR_GATE: [[f32; 3]; TRAIN_LEN] = [
    [0.0, 0.0, 0.0], //
    [1.0, 0.0, 1.0], //
    [0.0, 1.0, 1.0], //
    [1.0, 1.0, 1.0], //
];
const AND_GATE: [[f32; 3]; TRAIN_LEN] = [
    [0.0, 0.0, 0.0], //
    [1.0, 0.0, 0.0], //
    [0.0, 1.0, 0.0], //
    [1.0, 1.0, 1.0], //
];
const NAND_GATE: [[f32; 3]; TRAIN_LEN] = [
    [0.0, 0.0, 1.0], //
    [1.0, 0.0, 1.0], //
    [0.0, 1.0, 1.0], //
    [1.0, 1.0, 0.0], //
];

const TRAIN: &[[f32; 3]; 4] = &NAND_GATE;
const LEARNING_RATE: f32 = 1e-0;
const EPOCHS: usize = 10000;

fn cost(w1: f32, w2: f32, bias: f32) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAIN_LEN {
        let x1 = TRAIN[i][0];
        let x2 = TRAIN[i][1];
        let y = sigmoid((x1 * w1) + (x2 * w2) + bias);
        let expected = TRAIN[i][TRAIN_LEN - 2];
        let d = y - expected;
        result += d * d;
    }
    result /= TRAIN_LEN as f32;
    result
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn run(window: &tauri::Window) {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w1: f32 = rng.gen();
    let mut w2: f32 = rng.gen();
    let mut bias: f32 = rng.gen_range(0..=2) as f32;
    let mut step: f32 = if rng.gen::<bool>() { 1.0 } else { -1.0 };
    step *= 1e-1;
    emit(&window, "result", format!("step: {}", step));

    for epoch in 1..=EPOCHS {
        let c = cost(w1, w2, bias);
        let stepped_cost_w1 = cost(w1 + step, w2, bias);
        let stepped_cost_w2 = cost(w1, w2 + step, bias);
        let stepped_cost_bias = cost(w1, w2, bias + step);
        let derivative_w1 = (stepped_cost_w1 - c) / step;
        let derivative_w2 = (stepped_cost_w2 - c) / step;
        let derivative_bias = (stepped_cost_bias - c) / step;
        w1 -= LEARNING_RATE * derivative_w1;
        w2 -= LEARNING_RATE * derivative_w2;
        bias -= LEARNING_RATE * derivative_bias;

        if epoch % EPOCHS_PER_UPDATE == 0 || epoch == 0 {
            emit(
                &window,
                "epoch",
                format!(
                    "epoch: {}, cost(w1, w2, bias): {}",
                    epoch,
                    cost(w1, w2, bias)
                ),
            );
        }
    }
    emit(&window, "result", "RESULTS");
    emit(&window, "result", format!("w1: {}", w1));
    emit(&window, "result", format!("w2: {}", w2));
    emit(&window, "result", format!("bias: {}", bias));
    emit(
        &window,
        "result",
        format!("cost(w1, w2, bias): {}", cost(w1, w2, bias)),
    );

    for i in 0..=1 {
        for j in 0..=1 {
            emit(
                &window,
                "result",
                format!(
                    "sigmoid(i as f32 * w1 + j as f32 * w2 + bias) = sigmoid({} * {} + {} * {} + {}) = {}",
                    i,
                    w1,
                    j,
                    w2,
                    bias,
                    sigmoid(i as f32 * w1 + j as f32 * w2 + bias)
                ),
            );
        }
    }
}
