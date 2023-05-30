use crate::{
    brain::{Brain, Mat},
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

    let mut truth_in = Mat::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            truth_in.set(i, j, truth[i][j]);
        }
    }

    let mut truth_out = Mat::new(4, 1);
    for i in 0..4 {
        truth_out.set(i, 0, truth[i][2]);
    }

    let arch = [2, 2, 1];
    let mut nn = Brain::new(&arch);
    let mut grad = Brain::new(&arch);

    nn.rand(0.0, 1.0);
    let cost_init = nn.cost(&truth_in, &truth_out);
    emit(window, format!("Cost before training: {}", cost_init));

    for epoch in 1..=EPOCHS {
        nn.finite_diff(&mut grad, &truth_in, &truth_out);
        nn.learn(&mut grad);

        // nn.print(); // FIX TEMP
        // let new_cost = nn.cost(&truth_in, &truth_out); // FIX TEMP
        // emit(window, format!("Cost after epoch {}: {}", epoch, new_cost)); // FIX TEMP
        // std::process::exit(0); // FIX TEMP
        if epoch % EPOCHS_PER_PRINT == 0 {
            let new_cost = nn.cost(&truth_in, &truth_out);
            emit(window, format!("Cost after epoch {}: {}", epoch, new_cost));
        }
    }

    for i in 0..4 {
        nn.input(&truth_in.row(i));
        nn.forward();
        let output = nn.output();
        emit(
            window,
            format!("{} xor {} = {}", truth[i][0], truth[i][1], output.get(0, 0)),
        );
    }
}
