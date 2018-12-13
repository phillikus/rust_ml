use utils::stat;

pub struct LinearRegression {
    pub coefficient: Option<f32>,
    pub intercept: Option<f32>
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression { coefficient: None, intercept: None }
    }

    pub fn fit(&mut self, x_values : &Vec<f32>, y_values : &Vec<f32>) {
        let b1 = stat::covariance(x_values, y_values) / stat::variance(x_values);
        let b0 = stat::mean(y_values) - b1 * stat::mean(x_values);

        self.intercept = Some(b0);
        self.coefficient = Some(b1);       
    }   

    pub fn predict(&self, x : f32) -> f32 {
        if self.coefficient.is_none() || self.intercept.is_none() {
            panic!("fit(..) must be called first");
        }

        let b0 = self.intercept.unwrap();
        let b1 = self.coefficient.unwrap();

        return b0 + b1 * x;
    }

    pub fn predict_list(&self, x_values : &Vec<f32>) -> Vec<f32> {
        let mut predictions = Vec::new();

        for i in 0..x_values.len() {
            predictions.push(self.predict(x_values[i]));
        }

        return predictions;
    }

    pub fn evaluate(&self, x_test : &Vec<f32>, y_test: &Vec<f32>) -> f32 {
        if self.coefficient.is_none() || self.intercept.is_none() {
            panic!("fit(..) must be called first");
        }

        let y_predicted = self.predict_list(x_test);
        return self.root_mean_squared_error(y_test, &y_predicted);
    }

    fn root_mean_squared_error(&self, actual : &Vec<f32>, predicted : &Vec<f32>) -> f32 {
        let mut sum_error = 0f32;
        let length = actual.len();

        for i in 0..length {
            sum_error += f32::powf(predicted[i] - actual[i], 2f32);
        }

        let mean_error = sum_error / length as f32;
        return mean_error.sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn should_panic_for_empty_vecs() {
        let mut model = LinearRegression::new();
        let x_values = vec![];
        let y_values = vec![];

        model.fit(&x_values, &y_values);

        assert_delta!(0.4, model.intercept.unwrap(), 0.00001);
        assert_delta!(0.8f32, model.coefficient.unwrap(), 0.00001);
    }

    #[test]
    fn should_fit_coefficients_correctly() {
        let mut model = LinearRegression::new();
        let x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let y_values = vec![1f32, 3f32, 2f32, 3f32, 5f32];

        model.fit(&x_values, &y_values);

        assert_delta!(0.4f32, model.intercept.unwrap(), 0.00001);
        assert_delta!(0.8f32, model.coefficient.unwrap(), 0.00001);
    }

    #[test]
    fn should_predict_correctly() {
        let mut model = LinearRegression::new();
        let x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let y_values = vec![1f32, 3f32, 2f32, 3f32, 5f32];

        model.fit(&x_values, &y_values);

        assert_delta!(1.2f32, model.predict(1f32), 0.00001);
        assert_delta!(2f32, model.predict(2f32), 0.00001);
        assert_delta!(2.8f32, model.predict(3f32), 0.00001);
        assert_delta!(3.6f32, model.predict(4f32), 0.00001);
        assert_delta!(4.4f32, model.predict(5f32), 0.00001);
    }

    #[test]
    fn should_predict_list_correctly() {
        let mut model = LinearRegression::new();
        let x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let y_values = vec![1f32, 3f32, 2f32, 3f32, 5f32];

        model.fit(&x_values, &y_values);
        let predictions = model.predict_list(&x_values);

        assert_delta!(1.2f32, predictions[0], 0.00001);
        assert_delta!(2f32, predictions[1],0.00001);
        assert_delta!(2.8f32, predictions[2], 0.00001);
        assert_delta!(3.6f32, predictions[3], 0.00001);
        assert_delta!(4.4f32, predictions[4], 0.00001);
    }

    #[test]
    fn should_evaluate_correctly() {
        let mut model = LinearRegression::new();
        let x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
        let y_values = vec![1f32, 3f32, 2f32, 3f32, 5f32];

        model.fit(&x_values, &y_values);
        assert_delta!(0.693, model.evaluate(&x_values, &y_values), 0.00001);
    }
}