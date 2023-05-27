use self::perco::Mat;
mod perco;

pub(crate) fn run(_window: &tauri::Window) {
    let mut a_data = vec![0.0; 2];
    let mut a = Mat::new(1, 2, &mut a_data);
    a.fill_rand(5.0, 10.0);

    let mut id_data = vec![1.0, 0.0, 0.0, 1.0];
    let mut id = Mat::new(2, 2, &mut id_data);

    let mut b_data = vec![1.0; 4];
    let mut b = Mat::new(2, 2, &mut b_data);

    let mut dst_data = vec![0.0; 2];
    let mut dst = Mat::new(1, 2, &mut dst_data);

    println!("a:");
    a.print(4);
    println!("id:");
    id.print(4);
    println!("b:");
    b.print(4);

    println!("dst: a * id");
    perco::mat_dot(&mut dst, &a, &id);
    dst.print(4);

    println!("dst: a * b");
    perco::mat_dot(&mut dst, &a, &b);
    dst.print(4);

    println!("dst: a + a");
    perco::mat_sum(&mut dst, &a, &a);
    dst.print(4);
}
