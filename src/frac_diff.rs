pub fn get_weights_ffd(d: f64, threshold: f64) -> Vec<f64> {
    let mut w = vec![1.];
    let mut k = 1.0;
    loop {
        let w_: f64 = -w.last().unwrap() / k * (d - k + 1.0);
        if w_.abs() < threshold {
            break;
        }
        w.push(w_);
        k += 1.0;
    }
    w.reverse();
    w
}
