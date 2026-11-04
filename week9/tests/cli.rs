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

#[test]
fn order_commands_replay_the_same_evidence_mass() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let base = root.join("target").join("order-flow");
    let _ = std::fs::remove_dir_all(&base);
    let model = root.join("asia.uai");
    let random = base.join("random");
    let greedy = base.join("greedy");
    let analysis = base.join("analysis");
    let replay = base.join("replay");
    let random_run = infer()
        .arg("--model")
        .arg(&model)
        .arg("--evidence")
        .arg("AD")
        .arg("--order")
        .arg("random")
        .arg("--seed")
        .arg("42")
        .arg("--out")
        .arg(&random)
        .output()
        .expect("random");
    assert!(
        random_run.status.success(),
        "{}",
        String::from_utf8_lossy(&random_run.stderr)
    );
    let greedy_run = infer()
        .arg("--model")
        .arg(&model)
        .arg("--evidence")
        .arg("AD")
        .arg("--order")
        .arg("greedy")
        .arg("--out")
        .arg(&greedy)
        .output()
        .expect("greedy");
    assert!(
        greedy_run.status.success(),
        "{}",
        String::from_utf8_lossy(&greedy_run.stderr)
    );
    let analyze = infer()
        .arg("--model")
        .arg(&model)
        .arg("--evidence")
        .arg("AD")
        .arg("--analyze")
        .arg(random.join("order.json"))
        .arg("--out")
        .arg(&analysis)
        .output()
        .expect("analyze");
    assert!(
        analyze.status.success(),
        "{}",
        String::from_utf8_lossy(&analyze.stderr)
    );
    assert!(!analysis.join("run.json").exists());
    let replay_run = infer()
        .arg("--model")
        .arg(&model)
        .arg("--evidence")
        .arg("AD")
        .arg("--plan")
        .arg(greedy.join("order.json"))
        .arg("--out")
        .arg(&replay)
        .output()
        .expect("replay");
    assert!(
        replay_run.status.success(),
        "{}",
        String::from_utf8_lossy(&replay_run.stderr)
    );
    let replay_text = String::from_utf8(replay_run.stdout).expect("utf8");
    assert!(replay_text.contains("0.004501375"), "{replay_text}");
    let both = infer()
        .arg("--model")
        .arg(&model)
        .arg("--evidence")
        .arg("AD")
        .arg("--order")
        .arg("greedy")
        .arg("--analyze")
        .arg(random.join("order.json"))
        .arg("--out")
        .arg(base.join("both"))
        .output()
        .expect("exclusive");
    assert!(!both.status.success());
}
