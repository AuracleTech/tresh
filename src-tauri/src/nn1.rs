use super::emit;
use crate::data::{EPOCHS, LEARNING_RATE, STEP};
use crate::data::{EPOCHS_PER_PRINT, TRAINING_DATA};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn cost(w: f32, bias: f32) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAINING_DATA.len() {
        let x = TRAINING_DATA[i][0];
        let y = x * w + bias;
        let expected = TRAINING_DATA[i][1];
        let d = y - expected;
        result += d * d;
    }
    result /= TRAINING_DATA.len() as f32;
    result
}

pub(crate) fn run(window: &tauri::Window) {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w: f32 = rng.gen_range(0..=10) as f32;
    let mut bias: f32 = rng.gen_range(0..=5) as f32;

    for epoch in 1..=EPOCHS {
        let c = cost(w, bias);
        let stepped_cost_weight = cost(w + STEP, bias);
        let stepped_cost_bias = cost(w, bias + STEP);
        let derivative_weight = (stepped_cost_weight - c) / STEP;
        let derivative_bias = (stepped_cost_bias - c) / STEP;
        w -= LEARNING_RATE * derivative_weight;
        bias -= LEARNING_RATE * derivative_bias;
        if epoch % EPOCHS_PER_PRINT == 0 {
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
