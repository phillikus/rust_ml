extern crate dataplotlib;
extern crate linear_regression;

use dataplotlib::util::{linspace, zip2};
use dataplotlib::plotbuilder::PlotBuilder2D;
use dataplotlib::plotter::Plotter;

pub fn main() {
    let mut model = linear_regression::LinearRegression::new();
    let x_values = vec![1f32, 2f32, 3f32, 4f32, 5f32];
    let y_values = vec![1f32, 3f32, 2f32, 3f32, 5f32];

    model.fit(&x_values, &y_values);

    let accuracy = model.evaluate(&x_values, &y_values);
    println!("Accuracy: {0}", accuracy);

    let mut pb = PlotBuilder2D::new();
    let x_values_f64 = x_values.into_iter().map(|x| x as f64).collect();
    let y_values_f64 = y_values.into_iter().map(|x| x as f64).collect();

    pb.add_color_xy(zip2(&x_values_f64, &y_values_f64), [1.0, 0.0, 0.0, 1.0]);
    
    let mut plt = Plotter::new();
    plt.plot2d(pb);
    plt.join();
}