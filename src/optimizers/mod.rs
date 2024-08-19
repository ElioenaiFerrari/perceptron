pub fn sgd(x: f64, y: f64, alpha: f64) -> f64 {
    x - alpha * y
}

pub fn adam(x: f64, y: f64, alpha: f64, beta1: f64, beta2: f64, epsilon: f64) -> f64 {
    let mut m = 0.0;
    let mut v = 0.0;
    let mut t = 0.0;
    let beta1 = beta1;
    let beta2 = beta2;
    let epsilon = epsilon;
    let alpha = alpha;
    t += 1.0;
    m = beta1 * m + (1.0 - beta1) * y;
    v = beta2 * v + (1.0 - beta2) * y.powi(2);
    let m_hat = m / (1.0 - beta1.powf(t));
    let v_hat = v / (1.0 - beta2.powf(t));
    x - alpha * m_hat / (v_hat.sqrt() + epsilon)
}
