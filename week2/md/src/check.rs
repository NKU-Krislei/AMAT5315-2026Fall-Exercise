use crate::simulate::RC;
use crate::system::{kinetic_energy, potential_energy, System};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct RunJson {
    n: usize,
    #[serde(rename = "box")]
    box_xy: [f64; 2],
}

#[derive(Debug, Deserialize)]
struct Frame {
    pos: Vec<[f64; 2]>,
    vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    e_pot: f64,
    #[serde(rename = "E_kin")]
    e_kin: f64,
}

#[derive(Debug)]
pub struct CheckReport {
    pub drift: f64,
    pub t_speed: f64,
    pub chi2: f64,
    pub pass: bool,
}

pub fn check_artifacts(dir: &Path) -> Result<CheckReport, String> {
    let run: RunJson = serde_json::from_str(
        &fs::read_to_string(dir.join("run.json")).map_err(|e| format!("run.json: {e}"))?,
    )
    .map_err(|e| format!("run.json parse: {e}"))?;
    let traj = fs::read_to_string(dir.join("traj.jsonl")).map_err(|e| format!("traj.jsonl: {e}"))?;
    let frames: Vec<Frame> = traj
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("traj frame: {e}")))
        .collect::<Result<_, _>>()?;
    if frames.is_empty() {
        return Err("traj.jsonl has no frames".into());
    }

    let mut energies = Vec::with_capacity(frames.len());
    let mut speeds = Vec::new();
    for frame in &frames {
        if frame.pos.len() != run.n || frame.vel.len() != run.n {
            return Err("frame atom count does not match run.json n".into());
        }
        let sys = System::periodic(frame.pos.clone(), frame.vel.clone(), run.box_xy, RC);
        let e_pot = potential_energy(&sys);
        let e_kin = kinetic_energy(&sys);
        let scale_pot = e_pot.abs().max(1.0);
        let scale_kin = e_kin.abs().max(1.0);
        if (e_pot - frame.e_pot).abs() > 1e-9 * scale_pot {
            return Err(format!(
                "stored E_pot {} disagrees with recomputed {e_pot}",
                frame.e_pot
            ));
        }
        if (e_kin - frame.e_kin).abs() > 1e-9 * scale_kin {
            return Err(format!(
                "stored E_kin {} disagrees with recomputed {e_kin}",
                frame.e_kin
            ));
        }
        energies.push(e_kin + e_pot);
        for v in &frame.vel {
            speeds.push((v[0] * v[0] + v[1] * v[1]).sqrt());
        }
    }

    let e0 = energies[0];
    let k = 1.max(frames.len() / 10);
    let mean_first = energies[..k].iter().sum::<f64>() / k as f64;
    let mean_last = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    let drift = (mean_last - mean_first).abs() / e0.abs();

    let m = speeds.len() as f64;
    let mean_v2 = speeds.iter().map(|v| v * v).sum::<f64>() / m;
    let t_speed = mean_v2 / 2.0;
    let chi2 = speed_chi2(&speeds, t_speed);

    let pass = drift < 2e-3 && (t_speed - 0.5).abs() < 0.05 && chi2 < 2.0;
    Ok(CheckReport {
        drift,
        t_speed,
        chi2,
        pass,
    })
}

fn speed_chi2(speeds: &[f64], t_speed: f64) -> f64 {
    const BINS: usize = 24;
    let mut edges = [0.0; 25];
    for k in 0..BINS {
        let frac = k as f64 / BINS as f64;
        edges[k] = (-2.0 * t_speed * (1.0 - frac).ln()).sqrt();
    }
    edges[BINS] = f64::INFINITY;
    let mut obs = [0.0; BINS];
    for &v in speeds {
        let mut bin = BINS - 1;
        for k in 0..BINS {
            if v < edges[k + 1] {
                bin = k;
                break;
            }
        }
        obs[bin] += 1.0;
    }
    let expected = speeds.len() as f64 / BINS as f64;
    let mut sum = 0.0;
    for o in obs {
        let d = o - expected;
        sum += d * d / expected;
    }
    sum / 22.0
}

pub fn print_report(report: &CheckReport) {
    println!(
        "secular drift   {:.6e}   < 2e-3     {}",
        report.drift,
        if report.drift < 2e-3 { "ok" } else { "FAIL" }
    );
    println!(
        "temperature     {:.6}     |T-0.5|<0.05 {}",
        report.t_speed,
        if (report.t_speed - 0.5).abs() < 0.05 {
            "ok"
        } else {
            "FAIL"
        }
    );
    println!(
        "chi2/dof        {:.6}     < 2         {}",
        report.chi2,
        if report.chi2 < 2.0 { "ok" } else { "FAIL" }
    );
    if report.pass {
        println!("PASS");
    } else {
        println!("FAIL");
    }
}
