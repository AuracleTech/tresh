use super::emit;
use crate::brain::sigmoid;
use crate::data::{EPOCHS, EPOCHS_PER_PRINT, LEARN_RATE, STEP, TRAINING_DATA};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

fn cost(w1: f32, w2: f32, b: f32) -> f32 {
    let mut result = 0.0;
    for i in 0..TRAINING_DATA.len() {
        let x1 = TRAINING_DATA[i][0];
        let x2 = TRAINING_DATA[i][1];
        let expected = TRAINING_DATA[i][TRAINING_DATA.len() - 2];
        let y = sigmoid((x1 * w1) + (x2 * w2) + b);
        let d = y - expected;
        result += d * d;
    }
    result /= TRAINING_DATA.len() as f32;
    result
}

fn dcost(w1: f32, w2: f32, b: f32) -> (f32, f32, f32) {
    let c = cost(w1, w2, b);
    let dw1 = (cost(w1 + STEP, w2, b) - c) / STEP;
    let dw2 = (cost(w1, w2 + STEP, b) - c) / STEP;
    let db = (cost(w1, w2, b + STEP) - c) / STEP;
    (dw1, dw2, db)
}

fn gcost(w1: f32, w2: f32, b: f32) -> (f32, f32, f32) {
    let mut dw1 = 0.0;
    let mut dw2 = 0.0;
    let mut db = 0.0;
    for i in 0..TRAINING_DATA.len() {
        let xi = TRAINING_DATA[i][0];
        let yi = TRAINING_DATA[i][1];
        let zi = TRAINING_DATA[i][2];
        let ai = sigmoid((xi * w1) + (yi * w2) + b);
        let di = 2.0 * (ai - zi) * ai * (1.0 - ai);
        dw1 += di * xi;
        dw2 += di * yi;
        db += di;
    }
    dw1 /= TRAINING_DATA.len() as f32;
    dw2 /= TRAINING_DATA.len() as f32;
    db /= TRAINING_DATA.len() as f32;
    (dw1, dw2, db)
}

const USE_GCOST: bool = true;

pub fn run(window: &tauri::Window) {
    let seed: u64 = rand::random();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w1: f32 = rng.gen();
    let mut w2: f32 = rng.gen();
    let mut b: f32 = rng.gen_range(0..=2) as f32;
    emit(&window, format!("step: {}", STEP));

    for epoch in 1..=EPOCHS {
        let (dw1, dw2, db) = if USE_GCOST {
            gcost(w1, w2, b)
        } else {
            dcost(w1, w2, b)
        };
        w1 -= dw1 * LEARN_RATE;
        w2 -= dw2 * LEARN_RATE;
        b -= db * LEARN_RATE;

        if epoch % EPOCHS_PER_PRINT == 0 {
            emit(&window, format!("cost {} epoch {}", cost(w1, w2, b), epoch));
        }
    }
    emit(&window, format!("w1 {}", w1));
    emit(&window, format!("w2 {}", w2));
    emit(&window, format!("b {}", b));

    for i in 0..TRAINING_DATA.len() {
        let x1 = TRAINING_DATA[i][0];
        let x2 = TRAINING_DATA[i][1];
        let expected = TRAINING_DATA[i][TRAINING_DATA.len() - 2];
        let y = sigmoid((x1 * w1) + (x2 * w2) + b);
        emit(
            &window,
            format!("{} | {} = {} = E[{}]", x1, x2, y, expected),
        );
    }
}
