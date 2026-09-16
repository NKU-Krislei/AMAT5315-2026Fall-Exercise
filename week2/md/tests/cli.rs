use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn tmp_out() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cli_io");
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn run_writes_run_json_and_traj_jsonl() {
    let out = tmp_out();
    let exe = env!("CARGO_BIN_EXE_md");
    let status = Command::new(exe)
        .args([
            "run",
            "--n",
            "36",
            "--rho",
            "0.8",
            "--temperature",
            "0.5",
            "--dt",
            "0.01",
            "--eq-steps",
            "50",
            "--steps",
            "50",
            "--sample-every",
            "50",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(&out)
        .status()
        .expect("spawn md");
    assert!(status.success(), "md run failed: {status}");

    let run: Value =
        serde_json::from_str(&fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    for key in [
        "n",
        "rho",
        "box",
        "dt",
        "temperature",
        "eq_steps",
        "steps",
        "sample_every",
        "seed",
        "integrator",
    ] {
        assert!(run.get(key).is_some(), "missing {key}");
    }
    assert_eq!(run["n"], 36);
    assert_eq!(run["integrator"], "velocity-verlet");

    let traj = fs::read_to_string(out.join("traj.jsonl")).unwrap();
    let frames: Vec<Value> = traj
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0]["step"], 50);
    assert!(frames[0]["t"].as_f64().unwrap() > 0.0);
    assert!(frames[0].get("pos").is_some());
    assert!(frames[0].get("vel").is_some());
    assert!(frames[0].get("E_pot").is_some());
    assert!(frames[0].get("E_kin").is_some());
}

#[test]
fn contract_run_passes_physics() {
    if cfg!(debug_assertions) {
        eprintln!("skipping contract run in debug; cargo test --release runs it");
        return;
    }
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("contract");
    let _ = fs::remove_dir_all(&out);
    let exe = env!("CARGO_BIN_EXE_md");
    let status = Command::new(exe)
        .args(["run", "--out"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let status = Command::new(exe).arg("check").arg(&out).status().unwrap();
    assert!(status.success(), "md check should PASS on the contract run");
}

#[test]
fn ramp_to_is_recorded_and_last_temperature_matches_schedule() {
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ramp");
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).unwrap();
    let exe = env!("CARGO_BIN_EXE_md");
    let status = Command::new(exe)
        .args([
            "run",
            "--n",
            "36",
            "--temperature",
            "0.2",
            "--ramp-to",
            "1.2",
            "--eq-steps",
            "50",
            "--steps",
            "50",
            "--sample-every",
            "50",
            "--seed",
            "2026",
            "--out",
        ])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "md run --ramp-to failed: {status}");

    let run: Value =
        serde_json::from_str(&fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["ramp_to"], 1.2);

    let traj = fs::read_to_string(out.join("traj.jsonl")).unwrap();
    let frame: Value = serde_json::from_str(traj.lines().next().unwrap()).unwrap();
    let e_kin = frame["E_kin"].as_f64().unwrap();
    let n = 36.0;
    let t_thermo = 2.0 * e_kin / (2.0 * n - 2.0);
    assert!(
        (t_thermo - 1.2).abs() < 1e-9,
        "last thermo T={t_thermo}, expected 1.2 after the final production rescale"
    );
}
