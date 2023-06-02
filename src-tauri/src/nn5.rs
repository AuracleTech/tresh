use crate::{
    data::{EPOCHS, EPOCHS_PER_PRINT},
    emit,
    neural_network::{Matrix, NeuralNetwork},
};

pub fn run(window: &tauri::Window) {
    let truth_in = Matrix::from_2d_vec(&vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.1],
        vec![0.0, 0.1, 0.0, 0.1],
        vec![0.0, 0.1, 1.0, 0.0],
    ]);
    let truth_out = Matrix::from_2d_vec(&vec![
        vec![0.0, 0.0],
        vec![0.0, 0.1],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);

    // let truth_in = Matrix::from_2d_vec(&vec![
    //     vec![0.0, 0.0],
    //     vec![1.0, 0.0],
    //     vec![0.0, 1.0],
    //     vec![1.0, 1.0],
    // ]);
    // let truth_out = Matrix::from_2d_vec(&vec![
    //     vec![0.0], //
    //     vec![1.0], //
    //     vec![1.0], //
    //     vec![0.0], //
    // ]);

    let arch = [2, 2, 1];
    let mut nn = NeuralNetwork::new(&arch);
    let mut grad = NeuralNetwork::new(&arch);

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

    for i in 0..truth_in.rows {
        nn.input(&truth_in.row(i));
        nn.forward();
        let output = nn.output();
        emit(
            window,
            format!(
                "{} = {}",
                truth_in.row(i).to_string(),
                output.row(0).to_string()
            ),
        );
    }

    // for i in 0..4 {
    //     nn.input(&truth_in.row(i));
    //     nn.forward();
    //     let output = nn.output();
    //     emit(
    //         window,
    //         format!("{} xor {} = {}", truth[i][0], truth[i][1], output.get(0, 0)),
    //     );
    // }

    emit(window, "<hr>");

    emit(window, nn.to_string());
}
