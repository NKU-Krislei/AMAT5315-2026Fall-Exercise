use crate::rdf::radial_distribution;
use plotters::prelude::*;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::process::Command;

const DR: f64 = 0.05;

#[derive(Deserialize)]
struct RunJson {
    rho: f64,
    #[serde(rename = "box")]
    box_xy: [f64; 2],
}

#[derive(Deserialize)]
struct Frame {
    pos: Vec<[f64; 2]>,
}

pub fn write_video(dir: &Path, out: &Path) -> Result<(), String> {
    let ffmpeg = "ffmpeg";
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

    let tmp = std::env::temp_dir().join(format!("md-video-{}", std::process::id()));
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let result = write_frames_and_encode(&tmp, &run, &frames, ffmpeg, out);
    let _ = fs::remove_dir_all(&tmp);
    result
}

fn write_frames_and_encode(
    tmp: &Path,
    run: &RunJson,
    frames: &[Frame],
    ffmpeg: &str,
    out: &Path,
) -> Result<(), String> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut pos_so_far: Vec<Vec<[f64; 2]>> = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        pos_so_far.push(frame.pos.clone());
        let g = radial_distribution(&pos_so_far, run.box_xy, run.rho, DR);
        let png = tmp.join(format!("frame_{:04}.png", i + 1));
        draw_frame(&png, run.box_xy, &frame.pos, &g)?;
    }
    let pattern = tmp.join("frame_%04d.png");
    let status = Command::new(ffmpeg)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "20",
            "-i",
            pattern.to_str().unwrap(),
            "-vf",
            "scale=800:400",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "28",
        ])
        .arg(out)
        .status()
        .map_err(|e| format!("failed to spawn ffmpeg: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg failed with {status}"));
    }
    Ok(())
}

fn draw_frame(
    path: &Path,
    box_xy: [f64; 2],
    pos: &[[f64; 2]],
    g: &[(f64, f64)],
) -> Result<(), String> {
    let root = BitMapBackend::new(path, (800, 400)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| e.to_string())?;
    let (left, right) = root.split_horizontally(400);

    let mut atoms = ChartBuilder::on(&left)
        .margin(10)
        .x_label_area_size(24)
        .y_label_area_size(28)
        .build_cartesian_2d(0.0..box_xy[0], 0.0..box_xy[1])
        .map_err(|e| e.to_string())?;
    atoms.configure_mesh().draw().map_err(|e| e.to_string())?;
    for p in pos {
        atoms
            .draw_series(std::iter::once(Circle::new(
                (p[0], p[1]),
                4,
                BLUE.filled(),
            )))
            .map_err(|e| e.to_string())?;
    }

    let rmax = g.last().map(|(r, _)| *r).unwrap_or(1.0).max(1.0);
    let gmax = g
        .iter()
        .map(|(_, y)| *y)
        .fold(1.0_f64, f64::max)
        .max(1.0);
    let mut rdf = ChartBuilder::on(&right)
        .margin(10)
        .x_label_area_size(24)
        .y_label_area_size(28)
        .build_cartesian_2d(0.0..rmax, 0.0..gmax)
        .map_err(|e| e.to_string())?;
    rdf.configure_mesh()
        .x_desc("r")
        .y_desc("g(r)")
        .draw()
        .map_err(|e| e.to_string())?;
    rdf.draw_series(LineSeries::new(g.iter().copied(), RED.stroke_width(2)))
        .map_err(|e| e.to_string())?;
    root.present().map_err(|e| e.to_string())?;
    Ok(())
}
