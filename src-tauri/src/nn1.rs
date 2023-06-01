use super::emit;
use crate::data::EPOCHS_PER_PRINT;
use crate::data::{EPOCHS, LEARN_RATE, RNG_SEED, STEP};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

const TRAINING_DATA: [[f32; 2]; 5] = [
    [0.0, 0.0], //
    [1.0, 2.0], //
    [2.0, 4.0], //
    [3.0, 6.0], //
    [4.0, 8.0], //
];

fn cost(w: f32) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAINING_DATA.len() {
        let x = TRAINING_DATA[i][0];
        let y = x * w;
        let expected = TRAINING_DATA[i][1];
        let d = y - expected;
        result += d * d;
    }
    result /= TRAINING_DATA.len() as f32;
    result
}

fn dcost(w: f32) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAINING_DATA.len() {
        let x = TRAINING_DATA[i][0];
        let y = TRAINING_DATA[i][1];
        result += 2.0 * (x * w - y) * x;
    }
    result /= TRAINING_DATA.len() as f32;
    result
}

const USE_FINITE_DIFF: bool = false;

pub fn run(window: &tauri::Window) {
    let seed: u64 = RNG_SEED;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w: f32 = rng.gen_range(0..=10) as f32;

    emit(&window, format!("cost(w): {}", cost(w)));

    for epoch in 1..=EPOCHS {
        let dw = if USE_FINITE_DIFF {
            (cost(w + STEP) - cost(w)) / STEP
        } else {
            dcost(w)
        };
        w -= dw * LEARN_RATE;

        if epoch % EPOCHS_PER_PRINT == 0 {
            emit(&window, format!("cost {} epoch {}", cost(w), epoch));
        }
    }

    emit(&window, "<hr>");
    emit(&window, format!("w {}", w));
}
