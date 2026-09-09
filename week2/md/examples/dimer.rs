//! Relative energy error of the two-atom dimer under Euler and velocity-Verlet.
//!
//! From week2/:
//! cargo run --manifest-path md/Cargo.toml --example dimer

use md::{relative_energy_errors, Euler, System, VelocityVerlet};
use plotters::prelude::*;
use std::path::PathBuf;

fn dimer() -> System {
    System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0, 0.0], [0.0, 0.0]])
}

fn with_t0(errors: &[f64], dt: f64) -> Vec<(f64, f64)> {
    let mut pts = vec![(0.0, 0.0)];
    pts.extend(
        errors
            .iter()
            .enumerate()
            .map(|(k, e)| ((k as f64 + 1.0) * dt, *e)),
    );
    pts
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const DT: f64 = 0.01;

    let err_euler = relative_energy_errors(&Euler, &mut dimer(), DT, 500);
    let err_verlet_short = relative_energy_errors(&VelocityVerlet, &mut dimer(), DT, 500);
    let err_verlet_long = relative_energy_errors(&VelocityVerlet, &mut dimer(), DT, 5000);

    let out: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dimer.png");

    let root = BitMapBackend::new(&out, (1100, 420)).into_drawing_area();
    root.fill(&WHITE)?;
    let (left, right) = root.split_horizontally(550);

    let mut left_chart = ChartBuilder::on(&left)
        .caption("Relative energy error, 500 steps", ("sans-serif", 16))
        .margin(20)
        .x_label_area_size(36)
        .y_label_area_size(48)
        .build_cartesian_2d(0.0..5.0, -0.2..2.2)?;
    left_chart
        .configure_mesh()
        .x_desc("time t")
        .y_desc("(E(t) - E0) / |E0|")
        .draw()?;
    left_chart
        .draw_series(LineSeries::new(
            with_t0(&err_euler, DT),
            RED.stroke_width(2),
        ))?
        .label("forward Euler")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
    left_chart
        .draw_series(LineSeries::new(
            with_t0(&err_verlet_short, DT),
            BLUE.stroke_width(2),
        ))?
        .label("velocity-Verlet")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));
    left_chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()?;

    let mut right_chart = ChartBuilder::on(&right)
        .caption("Verlet alone, 5000 steps, x1000", ("sans-serif", 16))
        .margin(20)
        .x_label_area_size(36)
        .y_label_area_size(48)
        .build_cartesian_2d(0.0..50.0, -0.5..0.5)?;
    right_chart
        .configure_mesh()
        .x_desc("time t")
        .y_desc("error x 10^3")
        .draw()?;
    right_chart.draw_series(LineSeries::new(
        with_t0(&err_verlet_long, DT)
            .into_iter()
            .map(|(t, e)| (t, e * 1000.0)),
        BLUE.stroke_width(2),
    ))?;

    root.present()?;
    println!("wrote {}", out.display());
    Ok(())
}
