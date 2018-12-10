mod utils;

pub struct LinearRegression {
    coefficients: Option<Vec<f32>>,
    intercept: Option<f32>
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression { coefficients: None, intercept: None }
    }

    pub fn fit(x_train : &Vec<f32>, y_train : &Vec<f32>) {

    }

    pub fn predict(x : f32) -> f32 {
        return 23.3;
    }        
}