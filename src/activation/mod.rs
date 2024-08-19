pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

const E: f64 = 2.718281828459045;

pub fn tahn(x: f64) -> f64 {
    // fazer calculo sem usar lib
    let a = E.powf(x);
    let b = E.powf(-x);
    (a - b) / (a + b)
}

pub fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

pub fn elu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        1.0 * (E.powf(x) - 1.0)
    }
}
