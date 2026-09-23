//! Stability map of the three steppers, and a pulse on either side of RK4's limit.
//!
//! Run from week4/scripts/line_stability.sh. Writes evidence/line-stability.png.

use plotters::prelude::*;
use std::f64::consts::PI;
use std::path::PathBuf;
use week4::line::{Line, Scheme};
use week4::{growth_factor, Integrator, Rk4};

const RE_MIN: f64 = -3.5;
const RE_MAX: f64 = 0.8;
const IM_MIN: f64 = -3.2;
const IM_MAX: f64 = 3.2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = root_dir.join("evidence");
    std::fs::create_dir_all(&out_dir)?;
    let out = out_dir.join("line-stability.png");

    let (real_root, imag_root) = axis_crossings();
    let h_touch = step_where_mode_touches(31);
    let growth_056 = growth_factor(&Rk4, -0.05 * 31.0 * 31.0, -31.0, 0.056);
    println!("RK4 real crossing {real_root:.4}, imag crossing {imag_root:.4}");
    println!("mode |k|=31 touches |R|=1 at h={h_touch:.4}");
    println!("measured growth per step at h=0.056 for |k|=31: {growth_056:.3}");

    let stable = pulse(0.045);
    let unstable = pulse(0.056);
    let stable_max = field_max(&stable.1);
    let unstable_max = field_max(&unstable.1);
    println!(
        "pulse max |u|: h=0.045 -> {stable_max:.3} at t={:.3}; h=0.056 -> {unstable_max:.3} at t={:.3}",
        stable.0.last().copied().unwrap_or(0.0),
        unstable.0.last().copied().unwrap_or(0.0),
    );

    draw(&out, &stable, &unstable, real_root, imag_root)?;
    println!("wrote {}", out.display());
    Ok(())
}

fn pulse(h: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 64;
    let line = Line::new(n, 1.0, 0.05, Scheme::Fourier);
    let mut u = periodic_gaussian(n, PI / 2.0, 0.35);
    let mut times = vec![0.0];
    let mut frames = vec![u.clone()];
    let mut t = 0.0;
    while t + h <= 6.0 + 1e-12 {
        Rk4.step(&mut u, h, &|y| line.rate(y));
        t += h;
        for value in &mut u {
            if !value.is_finite() {
                *value = value.signum() * 1.0e6;
            }
        }
        times.push(t);
        frames.push(u.clone());
    }
    (times, frames)
}

fn periodic_gaussian(n: usize, center: f64, sigma: f64) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;
    (0..n)
        .map(|j| {
            let x = j as f64 * dx;
            (-5..=5)
                .map(|image| {
                    let d = x - center - 2.0 * PI * f64::from(image);
                    (-d * d / (2.0 * sigma * sigma)).exp()
                })
                .sum()
        })
        .collect()
}

fn field_max(frames: &[Vec<f64>]) -> f64 {
    frames
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0, f64::max)
}

/// `|1 + z + ... + z^p / p!|`, the stability function of a p-stage explicit method.
fn stability_modulus(re: f64, im: f64, order: usize) -> f64 {
    let mut term = (1.0, 0.0);
    let mut sum = term;
    for k in 1..=order {
        term = (
            (term.0 * re - term.1 * im) / k as f64,
            (term.0 * im + term.1 * re) / k as f64,
        );
        sum = (sum.0 + term.0, sum.1 + term.1);
    }
    sum.0.hypot(sum.1)
}

fn axis_crossings() -> (f64, f64) {
    let real_root = bisect(-3.2, -2.2, |x| stability_modulus(x, 0.0, 4) - 1.0);
    let imag_root = bisect(2.2, 3.2, |y| stability_modulus(0.0, y, 4) - 1.0);
    (real_root, imag_root)
}

fn step_where_mode_touches(k: i32) -> f64 {
    let k = f64::from(k);
    let lambda_re = -0.05 * k * k;
    let lambda_im = -k;
    bisect(0.02, 0.08, |h| {
        stability_modulus(lambda_re * h, lambda_im * h, 4) - 1.0
    })
}

fn bisect(mut a: f64, mut b: f64, f: impl Fn(f64) -> f64) -> f64 {
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        if f(a) * f(m) <= 0.0 {
            b = m;
        } else {
            a = m;
        }
    }
    0.5 * (a + b)
}

fn draw(
    out: &std::path::Path,
    stable: &(Vec<f64>, Vec<Vec<f64>>),
    unstable: &(Vec<f64>, Vec<Vec<f64>>),
    real_root: f64,
    imag_root: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let canvas = BitMapBackend::new(out, (1500, 540)).into_drawing_area();
    canvas.fill(&WHITE)?;
    let panels = canvas.split_evenly((1, 3));

    draw_stability(&panels[0], real_root, imag_root)?;
    draw_pulse(&panels[1], "h = 0.045", stable)?;
    draw_pulse(&panels[2], "h = 0.056", unstable)?;
    canvas.present()?;
    Ok(())
}

fn draw_stability(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    real_root: f64,
    imag_root: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption("growth factor per step, measured with RK4", ("sans-serif", 18))
        .margin(8)
        .x_label_area_size(38)
        .y_label_area_size(48)
        .build_cartesian_2d(RE_MIN..RE_MAX, IM_MIN..IM_MAX)?;
    chart
        .configure_mesh()
        .x_desc("Re(z),  z = lambda h")
        .y_desc("Im(z)")
        .draw()?;

    let n = 200;
    let dre = (RE_MAX - RE_MIN) / n as f64;
    let dim = (IM_MAX - IM_MIN) / n as f64;
    for i in 0..n {
        for j in 0..n {
            let re = RE_MIN + (i as f64 + 0.5) * dre;
            let im = IM_MIN + (j as f64 + 0.5) * dim;
            let growth = growth_factor(&Rk4, re, im, 1.0);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(re - dre / 2.0, im - dim / 2.0), (re + dre / 2.0, im + dim / 2.0)],
                growth_color(growth).filled(),
            )))?;
        }
    }

    for (order, label) in [(1, "Euler"), (2, "midpoint"), (4, "RK4")] {
        let pts = boundary(order);
        let style = if order == 4 { BLACK.stroke_width(2) } else { BLACK.stroke_width(1) };
        chart
            .draw_series(DashedLineSeries::new(pts, 7, if order == 4 { 0 } else { 5 }, style))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 16, y)], style)
            });
    }

    let inside = mode_dots(0.045);
    let outside = mode_dots(0.056);
    chart
        .draw_series(inside.iter().map(|&p| Circle::new(p, 3, BLUE.filled())))?
        .label("modes, h=0.045")
        .legend(|(x, y)| Circle::new((x + 8, y), 3, BLUE.filled()));
    chart
        .draw_series(outside.iter().map(|&p| Circle::new(p, 3, RED.filled())))?
        .label("modes, h=0.056")
        .legend(|(x, y)| Circle::new((x + 8, y), 3, RED.filled()));
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .draw()?;

    chart.draw_series(std::iter::once(Text::new(
        format!("crossings {real_root:.3}, ±{imag_root:.2} i"),
        (RE_MIN + 0.08, IM_MAX - 0.28),
        ("sans-serif", 14).into_font().color(&BLACK),
    )))?;
    Ok(())
}

fn draw_pulse(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    title: &str,
    run: &(Vec<f64>, Vec<Vec<f64>>),
) -> Result<(), Box<dyn std::error::Error>> {
    let (times, frames) = run;
    let t_last = *times.last().unwrap();
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 18))
        .margin(8)
        .x_label_area_size(38)
        .y_label_area_size(48)
        .build_cartesian_2d(0.0..2.0 * PI, t_last..0.0)?;
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("time t, downward")
        .draw()?;

    let n = frames[0].len();
    let dx = 2.0 * PI / n as f64;
    let limit = 1.0;
    for (row, frame) in frames.iter().enumerate() {
        let t0 = times[row];
        let t1 = if row + 1 < times.len() { times[row + 1] } else { t0 + (t0 - times[row.saturating_sub(1)]).max(1e-3) };
        for (j, value) in frame.iter().enumerate() {
            let x0 = j as f64 * dx;
            let painted = (value / limit).clamp(-1.0, 1.0);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, t0), (x0 + dx, t1)],
                pulse_color(painted).filled(),
            )))?;
        }
    }
    Ok(())
}

fn mode_dots(h: f64) -> Vec<(f64, f64)> {
    let n = 64;
    (-n / 2..n / 2)
        .map(|k| {
            let k = k as f64;
            let lambda_re = -0.05 * k * k;
            let lambda_im = if k == -(n as f64) / 2.0 { 0.0 } else { -k };
            (lambda_re * h, lambda_im * h)
        })
        .collect()
}

fn boundary(order: usize) -> Vec<(f64, f64)> {
    let mut pts = Vec::new();
    let n = 360;
    let dre = (RE_MAX - RE_MIN) / n as f64;
    let dim = (IM_MAX - IM_MIN) / n as f64;
    for i in 0..n {
        for j in 0..n {
            let re = RE_MIN + i as f64 * dre;
            let im = IM_MIN + j as f64 * dim;
            let a = stability_modulus(re, im, order);
            let b = stability_modulus(re + dre, im, order);
            if (a - 1.0) * (b - 1.0) < 0.0 {
                let t = (1.0 - a) / (b - a);
                pts.push((re + t * dre, im));
            }
        }
    }
    pts.sort_by(|p, q| {
        let ap = (p.1).atan2(p.0 + 1.0);
        let aq = (q.1).atan2(q.0 + 1.0);
        ap.partial_cmp(&aq).unwrap()
    });
    pts
}

fn growth_color(growth: f64) -> RGBColor {
    let t = if growth.is_finite() && growth > 0.0 {
        ((growth.log10().clamp(-1.0, 1.0)) + 1.0) / 2.0
    } else {
        1.0
    };
    ramp(t)
}

fn pulse_color(clipped: f64) -> RGBColor {
    ramp((clipped + 1.0) / 2.0)
}

fn ramp(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let u = t / 0.5;
        (40.0 + 215.0 * u, 70.0 + 185.0 * u, 170.0 + 85.0 * u)
    } else {
        let u = (t - 0.5) / 0.5;
        (255.0, 255.0 - 175.0 * u, 255.0 - 215.0 * u)
    };
    RGBColor(r as u8, g as u8, b as u8)
}
