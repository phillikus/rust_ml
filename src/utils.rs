pub fn mean(values : &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    return values.iter().sum::<f32>() / (values.len() as f32);
}

pub fn variance(values : &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    let mean = mean(values);
    return values.iter().map(|x| f32::powf(x - mean, 2 as f32)).sum::<f32>() / values.len() as f32;
}

pub fn covariance(x_values : &Vec<f32>, y_values : &Vec<f32>) -> f32 {
    if x_values.len() != y_values.len() {
        panic!("x_values and y_values must be of equal length.");
    }

    let length : usize = x_values.len();
    
    if length == 0usize {
        return 0f32;
    }

    let mut covariance : f32 = 0f32;
    let mean_x = mean(x_values);
    let mean_y = mean(y_values);

    for i in 0..length {
        covariance += (x_values[i] - mean_x) * (y_values[i] - mean_y)
    }

    return covariance / length as f32;        
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_should_return_0_for_empty_vec() {
        let values = vec![];
        assert_eq!(0f32, mean(&values));
    }

    #[test]
    fn mean_should_return_expected_value_for_vec() {
        let values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        assert_eq!(3f32, mean(&values));
    }

    #[test]
    fn variance_should_return_0_for_empty_vec() {
        let values = vec![];
        assert_eq!(0f32, variance(&values));
    }

    #[test]
    fn variance_should_return_expected_value_for_vec() {
        let values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        assert_eq!(2f32, variance(&values));
    }

    #[test]
    fn covariance_should_return_0_for_empty_vecs() {
        let x_values = vec![];
        let y_values = vec![];
        assert_eq!(0f32, covariance(&x_values, &y_values));
    }

    #[test]
    fn covariance_should_be_positive_for_linear_vecs() {
        let x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let y_values = vec![1f32, 3f32, 2f32, 3f32, 5f32];
        assert_eq!(1.6f32, covariance(&x_values, &y_values));
    }

    #[test]
    fn covariance_should_be_negative_for_inverse_vecs() {
        let x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let y_values = vec![0.5f32, 4f32, 1f32, -5f32, 4f32];
        assert_eq!(-0.4f32, covariance(&x_values, &y_values));
    }
}