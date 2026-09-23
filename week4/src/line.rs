//! Advection–diffusion on a periodic line, `u_t + c u_x = ν u_xx`.
//!
//! Space is discretized first. What remains is `du/dt = F(u)`, which the
//! integrators in the crate root already know how to advance.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    /// Multiply each Fourier coefficient by its wavenumber.
    Fourier,
    /// Second-order centred differences with periodic wraparound.
    Centered,
}

pub struct Line {
    n: usize,
    c: f64,
    nu: f64,
    scheme: Scheme,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Line {
    pub fn new(n: usize, c: f64, nu: f64, scheme: Scheme) -> Self {
        assert!(n >= 2 && n % 2 == 0, "even n, so the Nyquist mode is a grid mode");
        let mut planner = FftPlanner::<f64>::new();
        Self {
            n,
            c,
            nu,
            scheme,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn dx(&self) -> f64 {
        2.0 * PI / self.n as f64
    }

    /// `F(u) = -c u_x + ν u_xx` on the grid.
    pub fn rate(&self, u: &[f64]) -> Vec<f64> {
        match self.scheme {
            Scheme::Fourier => {
                let spec = self.forward(u);
                let ux = self.inverse(&apply(&spec, |k| ik(k, self.n)));
                let uxx = self.inverse(&apply(&spec, |k| Complex::new(-k * k, 0.0)));
                ux.iter()
                    .zip(uxx)
                    .map(|(x, xx)| -self.c * x + self.nu * xx)
                    .collect()
            }
            Scheme::Centered => {
                let dx = self.dx();
                (0..self.n)
                    .map(|j| {
                        let jm = (j + self.n - 1) % self.n;
                        let jp = (j + 1) % self.n;
                        let ux = (u[jp] - u[jm]) / (2.0 * dx);
                        let uxx = (u[jp] - 2.0 * u[j] + u[jm]) / (dx * dx);
                        -self.c * ux + self.nu * uxx
                    })
                    .collect()
            }
        }
    }

    fn forward(&self, u: &[f64]) -> Vec<Complex<f64>> {
        let mut spec: Vec<Complex<f64>> = u.iter().map(|&v| Complex::new(v, 0.0)).collect();
        self.forward.process(&mut spec);
        spec
    }

    fn inverse(&self, spec: &[Complex<f64>]) -> Vec<f64> {
        let mut spec = spec.to_vec();
        self.inverse.process(&mut spec);
        let scale = self.n as f64;
        spec.iter().map(|z| z.re / scale).collect()
    }
}

/// FFT bin `index` holds wavenumber `k`, in the order `0, 1, …, n/2-1, -n/2, …, -1`.
pub fn wavenumber(index: usize, n: usize) -> f64 {
    let n = n as i32;
    let index = index as i32;
    let k = if index < n / 2 { index } else { index - n };
    k as f64
}

/// Multiplier `i k` for a first derivative. The Nyquist mode `k = -n/2` has no
/// partner `+n/2` on the grid, so its odd derivative is set to zero.
fn ik(k: f64, n: usize) -> Complex<f64> {
    if k == -(n as f64) / 2.0 {
        Complex::new(0.0, 0.0)
    } else {
        Complex::new(0.0, k)
    }
}

fn apply(spec: &[Complex<f64>], multiplier: impl Fn(f64) -> Complex<f64>) -> Vec<Complex<f64>> {
    let n = spec.len();
    spec.iter()
        .enumerate()
        .map(|(index, coeff)| coeff * multiplier(wavenumber(index, n)))
        .collect()
}

/// Exact PDE solution of one real wave: `exp(-ν k² t) cos(k (x - c t))`.
pub fn exact_wave(n: usize, k: i32, c: f64, nu: f64, t: f64) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;
    let decay = (-nu * f64::from(k * k) * t).exp();
    (0..n)
        .map(|j| decay * (f64::from(k) * (j as f64 * dx - c * t)).cos())
        .collect()
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{integrate, Euler, Integrator, Rk2, Rk4};

    fn max_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    #[test]
    fn fourier_derivative_of_a_represented_wave_is_exact() {
        let n = 64;
        let k = 3.0;
        let u: Vec<f64> = (0..n)
            .map(|j| (k * 2.0 * PI * j as f64 / n as f64).cos())
            .collect();
        let line = Line::new(n, 0.0, 0.0, Scheme::Fourier);
        let spec = line.forward(&u);
        let ux = line.inverse(&apply(&spec, |m| ik(m, n)));
        let uxx = line.inverse(&apply(&spec, |m| Complex::new(-m * m, 0.0)));
        let ux_exact: Vec<f64> = (0..n)
            .map(|j| -k * (k * 2.0 * PI * j as f64 / n as f64).sin())
            .collect();
        let uxx_exact: Vec<f64> = u.iter().map(|v| -k * k * v).collect();
        assert!(max_abs_diff(&ux, &ux_exact) < 1e-12);
        assert!(max_abs_diff(&uxx, &uxx_exact) < 1e-12);
    }

    #[test]
    fn nyquist_mode_decays_but_does_not_travel() {
        let n = 32;
        let u: Vec<f64> = (0..n).map(|j| if j % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let line = Line::new(n, 1.0, 0.0, Scheme::Fourier);
        let spec = line.forward(&u);
        let ux = line.inverse(&apply(&spec, |m| ik(m, n)));
        let uxx = line.inverse(&apply(&spec, |m| Complex::new(-m * m, 0.0)));
        assert!(max_norm(&ux) < 1e-12);
        let k2 = ((n / 2) as f64).powi(2);
        assert!(max_abs_diff(&uxx, &u.iter().map(|v| -k2 * v).collect::<Vec<_>>()) < 1e-10);
    }

    #[test]
    fn centred_differences_are_second_order() {
        let wave = |n: usize| {
            (0..n)
                .map(|j| (3.0 * 2.0 * PI * j as f64 / n as f64).cos())
                .collect::<Vec<_>>()
        };
        let err = |n: usize| {
            let u = wave(n);
            let dx = 2.0 * PI / n as f64;
            let numerical: Vec<f64> = (0..n)
                .map(|j| {
                    let jm = (j + n - 1) % n;
                    let jp = (j + 1) % n;
                    (u[jp] - u[jm]) / (2.0 * dx)
                })
                .collect();
            let exact: Vec<f64> = (0..n)
                .map(|j| -3.0 * (3.0 * 2.0 * PI * j as f64 / n as f64).sin())
                .collect();
            max_abs_diff(&numerical, &exact)
        };
        let ratio = err(32) / err(64);
        assert!((ratio - 4.0).abs() < 0.15, "ratio {ratio}");
    }

    fn wave_error(method: &impl Integrator, scheme: Scheme, h: f64) -> f64 {
        let n = 64;
        let k = 1;
        let c = 1.0;
        let nu = 0.05;
        let t_end = 1.0;
        let line = Line::new(n, c, nu, scheme);
        let mut u = exact_wave(n, k, c, nu, 0.0);
        integrate(method, &mut u, h, t_end, &|y| line.rate(y));
        max_abs_diff(&u, &exact_wave(n, k, c, nu, t_end))
    }

    #[test]
    fn fourier_rk4_time_error_beats_the_spatial_error_of_differences() {
        let spectral = wave_error(&Rk4, Scheme::Fourier, 0.01);
        let finite = wave_error(&Rk4, Scheme::Centered, 0.01);
        assert!(spectral < 1e-9, "spectral {spectral}");
        assert!(finite > 1e-4, "finite difference {finite}");
    }

    #[test]
    fn three_integrators_keep_their_orders_on_the_wave() {
        assert_order(&Euler, 1.0);
        assert_order(&Rk2, 2.0);
        assert_order(&Rk4, 4.0);
    }

    fn assert_order(method: &impl Integrator, order: f64) {
        let coarse = wave_error(method, Scheme::Fourier, 0.05);
        let fine = wave_error(method, Scheme::Fourier, 0.025);
        let measured = (coarse / fine).log2();
        assert!(
            (measured - order).abs() < 0.2,
            "order {order}: measured {measured}, errors {coarse} -> {fine}"
        );
    }
}
