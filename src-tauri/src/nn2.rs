use super::emit;
use crate::brain::sigmoid;
use crate::data::{EPOCHS, EPOCHS_PER_PRINT, LEARN_RATE, TRAINING_DATA};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn cost(w1: f32, w2: f32, bias: f32) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAINING_DATA.len() {
        let x1 = TRAINING_DATA[i][0];
        let x2 = TRAINING_DATA[i][1];
        let y = sigmoid((x1 * w1) + (x2 * w2) + bias);
        let expected = TRAINING_DATA[i][TRAINING_DATA.len() - 2];
        let d = y - expected;
        result += d * d;
    }
    result /= TRAINING_DATA.len() as f32;
    result
}

pub(crate) fn run(window: &tauri::Window) {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w1: f32 = rng.gen();
    let mut w2: f32 = rng.gen();
    let mut bias: f32 = rng.gen_range(0..=2) as f32;
    let mut step: f32 = if rng.gen::<bool>() { 1.0 } else { -1.0 };
    step *= 1e-1;
    emit(&window, format!("step: {}", step));

    for epoch in 1..=EPOCHS {
        let c = cost(w1, w2, bias);
        let stepped_cost_w1 = cost(w1 + step, w2, bias);
        let stepped_cost_w2 = cost(w1, w2 + step, bias);
        let stepped_cost_bias = cost(w1, w2, bias + step);
        let derivative_w1 = (stepped_cost_w1 - c) / step;
        let derivative_w2 = (stepped_cost_w2 - c) / step;
        let derivative_bias = (stepped_cost_bias - c) / step;
        w1 -= LEARN_RATE * derivative_w1;
        w2 -= LEARN_RATE * derivative_w2;
        bias -= LEARN_RATE * derivative_bias;

        if epoch % EPOCHS_PER_PRINT == 0 {
            emit(
                &window,
                format!(
                    "epoch: {}, cost(w1, w2, bias): {}",
                    epoch,
                    cost(w1, w2, bias)
                ),
            );
        }
    }
    emit(&window, "RESULTS");
    emit(&window, format!("w1: {}", w1));
    emit(&window, format!("w2: {}", w2));
    emit(&window, format!("bias: {}", bias));
    emit(
        &window,
        format!("cost(w1, w2, bias): {}", cost(w1, w2, bias)),
    );

    for i in 0..=1 {
        for j in 0..=1 {
            emit(
                &window,
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
