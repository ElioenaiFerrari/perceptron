pub struct Input {
    pub value: f64,
    pub weight: f64,
}

pub fn gereral_fn(n_inputs: i32, bias: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..n_inputs {
        sum += i as f64;
    }
    sum + bias
}
