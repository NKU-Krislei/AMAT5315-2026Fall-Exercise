//! Pair energy as colour and force as arrows around one atom at the origin.
//!
//! From week2/:
//! cargo run --manifest-path md/Cargo.toml --example field

use md::{energy, force};
use plotters::prelude::*;
use std::f64::consts::PI;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("field.png");
    let r0 = 2.0_f64.powf(1.0 / 6.0);
    let lim = 2.4_f64;

    let root = BitMapBackend::new(&out, (920, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let (plot_area, bar_area) = root.split_horizontally(780);

    let mut chart = ChartBuilder::on(&plot_area)
        .caption("Pair energy (colour) and force (arrows)", ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(36)
        .y_label_area_size(44)
        .build_cartesian_2d(-lim..lim, -lim..lim)?;
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    let n = 180;
    let ds = 2.0 * lim / n as f64;
    for i in 0..n {
        for j in 0..n {
            let x = -lim + (i as f64 + 0.5) * ds;
            let y = -lim + (j as f64 + 0.5) * ds;
            let r = (x * x + y * y).sqrt();
            let u = if r < 1e-8 { 1.0 } else { energy(r).min(1.0) };
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x - ds / 2.0, y - ds / 2.0), (x + ds / 2.0, y + ds / 2.0)],
                energy_color(u).filled(),
            )))?;
        }
    }

    // Compressed radial arrows: F > 0 points outward (repulsion).
    let n_ring = 7;
    let n_ang = 16;
    for ir in 1..=n_ring {
        let r = lim * ir as f64 / (n_ring as f64 + 0.4);
        if r < 0.35 {
            continue;
        }
        let f = force(r);
        let len = 0.22 * (0.12 * f.abs()).tanh().copysign(f);
        for ia in 0..n_ang {
            let th = 2.0 * PI * ia as f64 / n_ang as f64;
            let (ct, st) = (th.cos(), th.sin());
            let (x, y) = (r * ct, r * st);
            let (dx, dy) = (len * ct, len * st);
            let to = (x + dx, y + dy);
            let arrow_len = (dx * dx + dy * dy).sqrt();
            if arrow_len < 1e-12 {
                continue;
            }
            let (ux, uy) = (dx / arrow_len, dy / arrow_len);
            let head = 0.07;
            let wing = 0.035;
            let hx = to.0 - head * ux;
            let hy = to.1 - head * uy;
            chart.draw_series(LineSeries::new([(x, y), (hx, hy)], BLACK.stroke_width(2)))?;
            chart.draw_series(std::iter::once(Polygon::new(
                [
                    to,
                    (hx + wing * uy, hy - wing * ux),
                    (hx - wing * uy, hy + wing * ux),
                ],
                BLACK.filled(),
            )))?;
        }
    }

    let dashes = 72;
    for k in 0..dashes {
        if k % 2 == 0 {
            let t0 = 2.0 * PI * k as f64 / dashes as f64;
            let t1 = 2.0 * PI * (k as f64 + 1.0) / dashes as f64;
            chart.draw_series(LineSeries::new(
                [
                    (r0 * t0.cos(), r0 * t0.sin()),
                    (r0 * t1.cos(), r0 * t1.sin()),
                ],
                BLACK.stroke_width(2),
            ))?;
        }
    }

    // Colour bar for U, capped at 1.
    let mut bar = ChartBuilder::on(&bar_area)
        .margin_top(50)
        .margin_bottom(50)
        .margin_left(8)
        .margin_right(48)
        .build_cartesian_2d(0.0..1.0, -1.0..1.0)?;
    bar.configure_mesh()
        .disable_x_mesh()
        .disable_x_axis()
        .y_desc("U(r)")
        .draw()?;
    let nb = 200;
    for i in 0..nb {
        let u0 = -1.0 + 2.0 * i as f64 / nb as f64;
        let u1 = -1.0 + 2.0 * (i + 1) as f64 / nb as f64;
        bar.draw_series(std::iter::once(Rectangle::new(
            [(0.0, u0), (1.0, u1)],
            energy_color(0.5 * (u0 + u1)).filled(),
        )))?;
    }

    root.present()?;
    println!("wrote {}", out.display());
    Ok(())
}

fn energy_color(u: f64) -> RGBColor {
    let u = u.clamp(-1.0, 1.0);
    if u < 0.0 {
        let t = u + 1.0;
        RGBColor(lerp(12, 250, t), lerp(50, 248, t), lerp(140, 220, t))
    } else {
        RGBColor(lerp(250, 196, u), lerp(248, 28, u), lerp(220, 28, u))
    }
}

fn lerp(a: i32, b: i32, t: f64) -> u8 {
    (a as f64 + (b - a) as f64 * t).round() as u8
}
