pub const RNG_SEED: u64 = 0xdeadbeef;
pub const LEARN_RATE: f32 = 1.;
pub const STEP: f32 = 1e-1;
pub const EPOCHS: usize = 1000;
// pub const EPOCHS: usize = 1;
const TOTAL_PRINTS_DURING_EPOCHS: usize = 10;
pub const EPOCHS_PER_PRINT: usize = if TOTAL_PRINTS_DURING_EPOCHS > EPOCHS {
    EPOCHS
} else {
    EPOCHS / TOTAL_PRINTS_DURING_EPOCHS
};

const OR_GATE: [[f32; 3]; 4] = [
    [0., 0., 0.], //
    [1., 0., 1.], //
    [0., 1., 1.], //
    [1., 1., 1.], //
];
const AND_GATE: [[f32; 3]; 4] = [
    [0., 0., 0.], //
    [1., 0., 0.], //
    [0., 1., 0.], //
    [1., 1., 1.], //
];
const NAND_GATE: [[f32; 3]; 4] = [
    [0., 0., 1.], //
    [1., 0., 1.], //
    [0., 1., 1.], //
    [1., 1., 0.], //
];
const XOR_GATE: [[f32; 3]; 4] = [
    [0., 0., 0.], //
    [1., 0., 1.], //
    [0., 1., 1.], //
    [1., 1., 0.], //
];
const NOR_GATE: [[f32; 3]; 4] = [
    [0., 0., 1.], //
    [1., 0., 0.], //
    [0., 1., 0.], //
    [1., 1., 0.], //
];

pub const TRAINING_DATA: &[[f32; AND_GATE[0].len()]; AND_GATE.len()] = &AND_GATE;
