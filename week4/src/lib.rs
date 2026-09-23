//! One interface, three explicit steppers.
//!
//! A step does not know what the state means. It only asks a rate function
//! `F` for `dy/dt` at a few trial states and blends those slopes.

pub mod line;

pub trait Integrator {
    /// Advance `y` by one step of size `h`. `rate` is `F` in `dy/dt = F(y)`.
    fn step<F>(&self, y: &mut [f64], h: f64, rate: &F)
    where
        F: Fn(&[f64]) -> Vec<f64>;
}

pub struct Euler;
pub struct Rk2;
pub struct Rk4;

impl Integrator for Euler {
    fn step<F>(&self, y: &mut [f64], h: f64, rate: &F)
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        // y <- y + h F(y)
        let k1 = rate(y);
        axpy(y, h, &k1);
    }
}

impl Integrator for Rk2 {
    fn step<F>(&self, y: &mut [f64], h: f64, rate: &F)
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        // Trial halfway, then trust the slope found there for the full step.
        let k1 = rate(y);
        let k2 = rate(&shift(y, 0.5 * h, &k1));
        axpy(y, h, &k2);
    }
}

impl Integrator for Rk4 {
    fn step<F>(&self, y: &mut [f64], h: f64, rate: &F)
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(y);
        let k2 = rate(&shift(y, 0.5 * h, &k1));
        let k3 = rate(&shift(y, 0.5 * h, &k2));
        let k4 = rate(&shift(y, h, &k3));
        // Weights 1, 2, 2, 1. Equal weights would still run, but only at order 2.
        for (yi, (((a, b), c), d)) in y
            .iter_mut()
            .zip(k1.iter().zip(k2.iter()).zip(k3.iter()).zip(k4.iter()))
        {
            *yi += (h / 6.0) * (a + 2.0 * b + 2.0 * c + d);
        }
    }
}

/// Integrate `dy/dt = rate(y)` from `y` at t = 0 to t = `t_end` with fixed step `h`.
pub fn integrate<I, F>(method: &I, y: &mut [f64], h: f64, t_end: f64, rate: &F)
where
    I: Integrator + ?Sized,
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = (t_end / h).round() as usize;
    for _ in 0..n {
        method.step(y, h, rate);
    }
}

fn shift(y: &[f64], h: f64, k: &[f64]) -> Vec<f64> {
    y.iter().zip(k).map(|(yi, ki)| yi + h * ki).collect()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decay(y: &[f64]) -> Vec<f64> {
        y.iter().map(|v| -v).collect()
    }

    fn final_error(method: &impl Integrator, h: f64) -> f64 {
        let mut y = vec![1.0];
        integrate(method, &mut y, h, 1.0, &decay);
        (y[0] - (-1.0_f64).exp()).abs()
    }

    #[test]
    fn one_euler_step_follows_the_opening_slope() {
        let mut y = vec![1.0];
        Euler.step(&mut y, 0.1, &decay);
        assert!((y[0] - 0.9).abs() < 1e-12);
    }

    fn assert_order(method: &impl Integrator, order: f64) {
        let coarse = final_error(method, 0.05);
        let fine = final_error(method, 0.025);
        let measured = (coarse / fine).log2();
        assert!(
            (measured - order).abs() < 0.15,
            "order {order}: measured {measured}, errors {coarse} -> {fine}"
        );
    }

    #[test]
    fn errors_fall_as_h_h2_and_h4() {
        assert_order(&Euler, 1.0);
        assert_order(&Rk2, 2.0);
        assert_order(&Rk4, 4.0);
    }
}
