pub fn linear_regression(x_values : &Vec<f32>, y_values : &Vec<f32>, epochs: i32, learning_rate : f32) -> (f32, f32) {
    let mut b_current : f32 = 0f32;
    let mut m_current : f32 = 0f32;
    
    for _ in 0..epochs {
        let tuple : (f32, f32) = step(b_current, m_current, x_values, y_values, learning_rate);
        b_current = tuple.0;
        m_current = tuple.1;
    }

    return (b_current, m_current);
}

fn step(b_current : f32, m_current: f32, x_values : &Vec<f32>, y_values : &Vec<f32>, learning_rate: f32) -> (f32, f32) {
    let mut b_gradient : f32 = 0f32;
    let mut m_gradient : f32 = 0f32;
    let length = y_values.len();

    for i in 0..length {
        b_gradient += -(2f32 / length as f32) * (y_values[i] - ((m_current * x_values[i]) + b_current));
        m_gradient += -(2f32 / length as f32) * x_values[i] * (y_values[i] - ((m_current * x_values[i]) + b_current));
    }

    let new_b = b_current - (learning_rate * b_gradient);
    let new_m = m_current - (learning_rate * m_gradient);

    return (new_b, new_m);
}