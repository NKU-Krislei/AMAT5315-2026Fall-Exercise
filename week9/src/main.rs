use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use infer::{
    ad_tuberculosis_diagram, compare, comparison_json, comparison_text, format_checks,
    matrix_and_ising, Model,
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("infer: {err}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<String, String> {
    match args.first().map(String::as_str) {
        Some("checks") => {
            let flags = flags(&args[1..])?;
            let checks = matrix_and_ising();
            let text = format_checks(&checks);
            if let Some(path) = flags.get("out") {
                fs::write(path, &text).map_err(|err| err.to_string())?;
            }
            if checks.ab_error >= 1e-12
                || checks.trace_error >= 1e-12
                || checks.ising_error >= 1e-12
            {
                return Err(format!("numeric check exceeded 1e-12\n{text}"));
            }
            Ok(text)
        }
        Some("diagram") => {
            let flags = flags(&args[1..])?;
            let model_path = required(&flags, "model")?;
            let out = required(&flags, "out")?;
            let model = Model::load(Path::new(model_path))?;
            let svg = ad_tuberculosis_diagram(&model);
            if let Some(parent) = Path::new(out).parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).map_err(|err| err.to_string())?;
                }
            }
            fs::write(out, &svg).map_err(|err| err.to_string())?;
            Ok(format!("wrote {out}\n"))
        }
        _ => infer_case(args),
    }
}

fn infer_case(args: &[String]) -> Result<String, String> {
    let flags = flags(args)?;
    reject_unknown(&flags, &["model", "evidence", "out"])?;
    let model = Model::load(Path::new(required(&flags, "model")?))?;
    let label = required(&flags, "evidence")?;
    let out = Path::new(required(&flags, "out")?);
    let comparison = compare(&model, label)?;
    fs::create_dir_all(out).map_err(|err| err.to_string())?;
    let json = serde_json::to_string_pretty(&comparison_json(&comparison))
        .map_err(|err| err.to_string())?;
    fs::write(out.join("run.json"), json + "\n").map_err(|err| err.to_string())?;
    Ok(comparison_text(&comparison))
}

fn flags(args: &[String]) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut parsed = std::collections::BTreeMap::new();
    let mut cursor = args.iter();
    while let Some(arg) = cursor.next() {
        let key = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument {arg}"))?;
        let value = cursor
            .next()
            .ok_or_else(|| format!("missing value for --{key}"))?;
        if parsed.insert(key.to_string(), value.clone()).is_some() {
            return Err(format!("duplicate --{key}"));
        }
    }
    Ok(parsed)
}

fn required<'a>(
    flags: &'a std::collections::BTreeMap<String, String>,
    name: &str,
) -> Result<&'a str, String> {
    flags
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing --{name}"))
}

fn reject_unknown(
    flags: &std::collections::BTreeMap<String, String>,
    allowed: &[&str],
) -> Result<(), String> {
    for key in flags.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown argument --{key}"));
        }
    }
    Ok(())
}
