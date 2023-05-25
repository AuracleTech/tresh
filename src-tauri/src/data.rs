pub(crate) const LEARNING_RATE: f32 = 1e-1;
pub(crate) const STEP: f32 = 1e-1;
pub(crate) const EPOCHS: usize = 1000000;
const TOTAL_PRINTS_DURING_EPOCHS: usize = 20;
pub(crate) const EPOCHS_PER_PRINT: usize = EPOCHS / TOTAL_PRINTS_DURING_EPOCHS;

const OR_GATE: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0], //
    [1.0, 0.0, 1.0], //
    [0.0, 1.0, 1.0], //
    [1.0, 1.0, 1.0], //
];
const AND_GATE: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0], //
    [1.0, 0.0, 0.0], //
    [0.0, 1.0, 0.0], //
    [1.0, 1.0, 1.0], //
];
const NAND_GATE: [[f32; 3]; 4] = [
    [0.0, 0.0, 1.0], //
    [1.0, 0.0, 1.0], //
    [0.0, 1.0, 1.0], //
    [1.0, 1.0, 0.0], //
];
const XOR_GATE: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0], //
    [1.0, 0.0, 1.0], //
    [0.0, 1.0, 1.0], //
    [1.0, 1.0, 0.0], //
];
const NOR_GATE: [[f32; 3]; 4] = [
    [0.0, 0.0, 1.0], //
    [1.0, 0.0, 0.0], //
    [0.0, 1.0, 0.0], //
    [1.0, 1.0, 0.0], //
];
// y = mx + b = m2 + 0
const LINEAR_DATA_EQUATION: [[f32; 2]; 5] = [
    [0.0, 0.0], //
    [1.0, 2.0], //
    [2.0, 4.0], //
    [3.0, 6.0], //
    [4.0, 8.0], //
];

pub(crate) const TRAINING_DATA: &[[f32; XOR_GATE[0].len()]; XOR_GATE.len()] = &XOR_GATE;
