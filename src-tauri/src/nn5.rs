use crate::{
    brain::{Brain, Matrix},
    data::{EPOCHS, EPOCHS_PER_PRINT},
    emit,
};

pub(crate) fn run(window: &tauri::Window) {
    let truth = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ];

    let mut truth_in = Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            truth_in.set(i, j, truth[i][j]);
        }
    }

    let mut truth_out = Matrix::new(4, 1);
    for i in 0..4 {
        truth_out.set(i, 0, truth[i][2]);
    }

    let arch = [2, 2, 1];
    let mut nn = Brain::new(&arch);
    let mut grad = Brain::new(&arch);

    nn.rand(0.0, 1.0);
    let cost_init = nn.cost(&truth_in, &truth_out);
    emit(window, format!("Cost pre-training: {}", cost_init));

    for epoch in 1..=EPOCHS {
        nn.finite_diff(&mut grad, &truth_in, &truth_out);
        nn.learn(&mut grad);
        if epoch % EPOCHS_PER_PRINT == 0 {
            let new_cost = nn.cost(&truth_in, &truth_out);
            emit(window, format!("Cost epoch {}: {}", epoch, new_cost));
        }
    }

    emit(window, "<hr>");

    for i in 0..4 {
        nn.input(&truth_in.row(i));
        nn.forward();
        let output = nn.output();
        emit(
            window,
            format!("{} xor {} = {}", truth[i][0], truth[i][1], output.get(0, 0)),
        );
    }

    emit(window, "<hr>");

    emit(window, nn.to_string());
}
