use super::emit;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

const TRAIN: [[f32; 2]; 5] = [
    [0.0, 0.0], //
    [1.0, 2.0], //
    [2.0, 4.0], //
    [3.0, 6.0], //
    [4.0, 8.0], //
];
const TRAIN_LEN: usize = TRAIN.len();
const LEARNING_RATE: f32 = 1e-3;
const STEP: f32 = 1e-3;
const EPOCHS: usize = 10000;

fn cost(w: f32, bias: f32) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAIN_LEN {
        let x = TRAIN[i][0];
        let y = x * w + bias;
        let expected = TRAIN[i][1];
        let d = y - expected;
        result += d * d;
    }
    result /= TRAIN_LEN as f32;
    result
}

const EPOCHS_PER_UPDATE: usize = 250;

pub(crate) fn run(window: &tauri::Window) {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w: f32 = rng.gen_range(0..=10) as f32;
    let mut bias: f32 = rng.gen_range(0..=5) as f32;

    for epoch in 0..EPOCHS {
        let c = cost(w, bias);
        let stepped_cost_weight = cost(w + STEP, bias);
        let stepped_cost_bias = cost(w, bias + STEP);
        let derivative_weight = (stepped_cost_weight - c) / STEP;
        let derivative_bias = (stepped_cost_bias - c) / STEP;
        w -= LEARNING_RATE * derivative_weight;
        bias -= LEARNING_RATE * derivative_bias;
        if epoch % EPOCHS_PER_UPDATE == 0 {
            emit(
                &window,
                "epoch",
                format!("epoch: {}, cost(w, bias): {}", epoch, cost(w, bias)),
            );
        }
    }
    emit(&window, "result", "RESULTS".to_string());
    emit(&window, "result", format!("w: {}", w));
    emit(&window, "result", format!("bias: {}", bias));
}
