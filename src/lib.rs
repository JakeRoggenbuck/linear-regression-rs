use std::iter::zip;

trait Regression {
    fn squared_error(&mut self, f: &dyn Fn(f64) -> f64) -> f64;
    fn mean_squared_error(&mut self, f: &dyn Fn(f64) -> f64) -> f64;
    fn gradient_descent(&mut self, slope: f64, b: f64, learning_rate: f64) -> (f64, f64);
    fn regression(&mut self, epoch: i32, learning_rate: f64) -> (f64, f64);
}

struct Frame {
    y: Vec<f64>,
    x: Vec<f64>,
    verbose: bool,
}

impl Regression for Frame {
    fn squared_error(&mut self, f: &dyn Fn(f64) -> f64) -> f64 {
        let mut error = 0.0;

        for (x, y) in zip(&self.x, &self.y) {
            let delta = y - f(*x);
            error += delta * delta;
        }

        error
    }

    fn mean_squared_error(&mut self, f: &dyn Fn(f64) -> f64) -> f64 {
        self.squared_error(f) / self.x.len() as f64
    }

    fn gradient_descent(&mut self, slope: f64, b: f64, learning_rate: f64) -> (f64, f64) {
        let length = self.x.len() as f64;

        let mut slope_gradient = 0.0;
        let mut b_gradient = 0.0;

        for (x, y) in zip(&self.x, &self.y) {
            // Partial derivative with respect to slope
            slope_gradient += -(2.0 / length) * x * (y - (slope * x + b));
            // Partial derivative with respect to b
            b_gradient += -(2.0 / length) * (y - (slope * x + b));
        }

        (
            slope - slope_gradient * learning_rate,
            b - b_gradient * learning_rate,
        )
    }

    fn regression(&mut self, epoch: i32, learning_rate: f64) -> (f64, f64) {
        let mut slope = 0.0;
        let mut b = 0.0;

        if self.verbose {
            for x in 0..epoch {
                (slope, b) = self.gradient_descent(slope, b, learning_rate);

                println!("Epoch: {}", x);
                println!("y = {}x + {}", slope, b);
            }
        } else {
            for _ in 0..epoch {
                (slope, b) = self.gradient_descent(slope, b, learning_rate);
            }
        }

        (slope, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_error_test() {
        let mut frame = Frame {
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            y: vec![1.0, 2.0, 4.0, 4.0, 5.0],
            verbose: false,
        };

        assert_eq!(frame.squared_error(&|x| x), 1.0);

        frame.x[0] = 8.0;
        frame.x[2] = 2.0;

        assert_eq!(frame.squared_error(&|x| x), 53.0);
    }

    #[test]
    fn mean_squared_error_test() {
        let mut frame = Frame {
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            y: vec![1.0, 2.0, 4.0, 4.0, 5.0],
            verbose: false,
        };

        assert_eq!(frame.mean_squared_error(&|x| x), 0.2);

        frame.x[0] = 8.0;
        frame.x[2] = 2.0;

        assert_eq!(frame.mean_squared_error(&|x| x), 10.6);
    }

    #[test]
    fn regression_test() {
        // f(x) = x
        let mut frame = Frame {
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            y: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            verbose: false,
        };

        let (slope, b) = frame.regression(1_000_000, 0.0001);

        assert!(f64::abs(slope - 1.0) < 0.00001);
        assert!(f64::abs(b) < 0.00001);

        // f(x) = 3x + 4
        let mut frame = Frame {
            x: vec![3.0, 2.0, 1.0, 4.3, 3.4, 8.2, 1.1, 4.5, 6.7],
            y: vec![13.0, 10.0, 7.0, 16.9, 14.2, 28.6, 7.3, 17.5, 24.1],
            verbose: false,
        };

        let (slope, b) = frame.regression(1_000_000, 0.0001);

        assert!(f64::abs(slope - 3.0) < 0.00001);
        assert!(f64::abs(b - 4.0) < 0.00001);
    }
}
