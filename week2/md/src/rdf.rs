use std::f64::consts::PI;

fn min_image(d: f64, length: f64) -> f64 {
    d - length * (d / length).round()
}

/// Radial distribution g(r) averaged over atoms and the given frames.
pub fn radial_distribution(
    frames: &[Vec<[f64; 2]>],
    box_xy: [f64; 2],
    rho: f64,
    dr: f64,
) -> Vec<(f64, f64)> {
    let rmax = 0.5 * box_xy[0].min(box_xy[1]);
    let n_bins = (rmax / dr).floor() as usize;
    if n_bins == 0 || frames.is_empty() {
        return Vec::new();
    }
    let n = frames[0].len();
    let mut hist = vec![0.0; n_bins];
    for pos in frames {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = min_image(pos[i][0] - pos[j][0], box_xy[0]);
                let dy = min_image(pos[i][1] - pos[j][1], box_xy[1]);
                let r = (dx * dx + dy * dy).sqrt();
                if r >= rmax {
                    continue;
                }
                let bin = (r / dr).floor() as usize;
                if bin < n_bins {
                    hist[bin] += 1.0;
                }
            }
        }
    }
    let n_samples = (frames.len() * n) as f64;
    hist.iter()
        .enumerate()
        .map(|(k, &count)| {
            let r0 = k as f64 * dr;
            let r1 = r0 + dr;
            let expected = rho * PI * (r1 * r1 - r0 * r0);
            let g = if expected > 0.0 {
                (count / n_samples) / expected
            } else {
                0.0
            };
            (0.5 * (r0 + r1), g)
        })
        .collect()
}
