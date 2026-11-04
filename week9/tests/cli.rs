use std::path::PathBuf;
use std::process::Command;

fn infer() -> Command {
    Command::new(env!("CARGO_BIN_EXE_infer"))
}

#[test]
fn ad_run_records_the_evidence_mass_and_disease_probabilities() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("target").join("cli-ad");
    let _ = std::fs::remove_dir_all(&out);
    let output = infer()
        .arg("--model")
        .arg(root.join("asia.uai"))
        .arg("--evidence")
        .arg("AD")
        .arg("--out")
        .arg(&out)
        .output()
        .expect("spawn infer");
    assert!(
        output.status.success(),
        "stderr {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("evidence probability"));
    assert!(stdout.contains("disease T=yes"));
    assert!(stdout.contains("disease L=yes"));
    assert!(stdout.contains("disease B=yes"));
    let json = std::fs::read_to_string(out.join("run.json")).expect("run.json");
    assert!(json.contains("0.004501375"), "{json}");
    assert!(json.contains("\"name\": \"A\""));
    assert!(json.contains("\"name\": \"D\""));
}

#[test]
fn missing_evidence_is_rejected() {
    let output = infer()
        .arg("--model")
        .arg("asia.uai")
        .arg("--out")
        .arg("artifacts/missing")
        .output()
        .expect("spawn infer");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("missing --evidence"), "{stderr}");
}

#[test]
fn checks_print_the_answer_key() {
    let output = infer().arg("checks").output().expect("spawn infer");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("4"));
    assert!(stdout.contains("10"));
    assert!(stdout.contains("12"));
    assert!(stdout.contains("12.60256209895193"));
}
