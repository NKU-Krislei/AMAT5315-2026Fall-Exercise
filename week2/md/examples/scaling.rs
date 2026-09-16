//! Log-log seconds per step vs N for naive and cell-list forces.
//!
//! From week2/:
//! cargo run --manifest-path md/Cargo.toml --example scaling

use plotters::prelude::*;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Wall-clock medians from three runs: --eq-steps 100 --steps 500 (600 steps).
    let naive = [(100.0, 0.0126 / 600.0), (400.0, 0.0899 / 600.0), (1600.0, 1.2009 / 600.0)];
    let cells = [(100.0, 0.0161 / 600.0), (400.0, 0.0397 / 600.0), (1600.0, 0.1725 / 600.0)];

    let out: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("scaling.png");
    let root = BitMapBackend::new(&out, (720, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Seconds per step vs N", ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(56)
        .build_cartesian_2d(
            (80.0..2000.0).log_scale(),
            (1e-5..5e-3).log_scale(),
        )?;
    chart
        .configure_mesh()
        .x_desc("atoms N")
        .y_desc("seconds / step")
        .draw()?;
    chart
        .draw_series(LineSeries::new(naive, BLUE.stroke_width(2)))?
        .label("naive")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));
    chart.draw_series(PointSeries::of_element(
        naive,
        5,
        BLUE,
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;
    chart
        .draw_series(LineSeries::new(cells, RED.stroke_width(2)))?
        .label("cells")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart.draw_series(PointSeries::of_element(
        cells,
        5,
        RED,
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    println!("wrote {}", out.display());
    Ok(())
}
