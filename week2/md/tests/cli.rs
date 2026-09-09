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
