fn mean(values : &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    return values.iter().sum::<f32>() / (values.len() as f32);
}

fn variance(values : &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    let mean = mean(values);
    return values.iter().map(|x| f32::powf(x - mean, 2 as f32)).sum::<f32>() / values.len() as f32;
}

fn covariance(x_values : &Vec<f32>, y_values : &Vec<f32>) -> f32 {
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

    for i in 0..length-1 {
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
    fn covariance_should_be_greater_than_0_value_for_linear_vecs() {
        let mut x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let mut y_values = vec![5f32, 10f32, 15f32, 20f32, 25f32];
        assert_eq!(6f32, covariance(&x_values, &y_values));
    }

    #[test]
    fn covariance_should_be_smaller_than_1_for_unrelated_vecs() {
        let mut x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let mut y_values = vec![0.5f32, 4f32, 1f32, -5f32, 32f32];
        assert_eq!(0.6f32, covariance(&x_values, &y_values));
    }
}