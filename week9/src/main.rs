use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use infer::{
    ad_tuberculosis_diagram, compare, comparison_json, comparison_text, enumerate, execute_scalar,
    format_checks, matrix_and_ising, parse_evidence, query_inputs, recount_plan, scalar_plan,
    Model, OrderChoice, ScalarPlan,
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
    let ordering = ["order", "seed", "plan", "analyze"]
        .iter()
        .any(|name| flags.contains_key(*name));
    if !ordering {
        reject_unknown(&flags, &["model", "evidence", "out"])?;
        let model = Model::load(Path::new(required(&flags, "model")?))?;
        let label = required(&flags, "evidence")?;
        let out = Path::new(required(&flags, "out")?);
        let comparison = compare(&model, label)?;
        fs::create_dir_all(out).map_err(|err| err.to_string())?;
        let json = serde_json::to_string_pretty(&comparison_json(&comparison))
            .map_err(|err| err.to_string())?;
        fs::write(out.join("run.json"), json + "\n").map_err(|err| err.to_string())?;
        return Ok(comparison_text(&comparison));
    }
    reject_unknown(
        &flags,
        &[
            "model", "evidence", "out", "order", "seed", "plan", "analyze",
        ],
    )?;
    let modes = ["order", "plan", "analyze"]
        .iter()
        .filter(|name| flags.contains_key(**name))
        .count();
    if modes > 1 {
        return Err("order, plan and analyze are mutually exclusive".to_string());
    }
    if modes == 0 {
        return Err("seed is only valid with --order random".to_string());
    }
    let model = Model::load(Path::new(required(&flags, "model")?))?;
    let label = required(&flags, "evidence")?;
    let evidence = parse_evidence(label)?;
    let out = Path::new(required(&flags, "out")?);
    fs::create_dir_all(out).map_err(|err| err.to_string())?;

    if let Some(path) = flags.get("analyze") {
        if flags.contains_key("seed") || flags.contains_key("order") {
            return Err("seed is only valid with --order random".to_string());
        }
        let plan = read_plan(path)?;
        ensure_same_query(&model, &evidence, &plan)?;
        let cost = recount_plan(&model, &plan)?;
        write_json(&out.join("complexity.json"), &cost)?;
        return Ok(format!(
            "analyzed {path}\nwork={}\nlargest_tensor={}\nlargest intermediate entries={} labels={}\n",
            cost.work,
            cost.largest_tensor,
            cost.largest_intermediate.entries,
            label_list(&cost.largest_intermediate.retained)
        ));
    }

    let plan = if let Some(path) = flags.get("plan") {
        if flags.contains_key("seed") {
            return Err("seed is only valid with --order random".to_string());
        }
        let plan = read_plan(path)?;
        ensure_same_query(&model, &evidence, &plan)?;
        plan
    } else {
        let choice = match flags.get("order").map(String::as_str) {
            Some("greedy") => {
                if flags.contains_key("seed") {
                    return Err("seed is only valid with --order random".to_string());
                }
                OrderChoice::Greedy
            }
            Some("random") => {
                let seed = required(&flags, "seed")?
                    .parse::<u64>()
                    .map_err(|err| format!("seed: {err}"))?;
                OrderChoice::Random { seed }
            }
            Some(other) => return Err(format!("unknown order {other}; expected random or greedy")),
            None => return Err("missing --order".to_string()),
        };
        scalar_plan(&model, &evidence, choice)?
    };

    let planned = execute_scalar(&model, &evidence, &plan)?;
    let exact = enumerate(&model, &evidence).evidence_mass;
    let cost = recount_plan(&model, &plan)?;
    let relative = (planned - exact).abs() / exact.abs();
    write_json(&out.join("order.json"), &plan)?;
    write_json(&out.join("complexity.json"), &cost)?;
    let report = serde_json::json!({
        "evidence": label,
        "observations": evidence.iter().map(|(id, state)| serde_json::json!({
            "id": id,
            "name": label_name(*id),
            "state": state,
        })).collect::<Vec<_>>(),
        "evidence_mass": {"enumeration": exact, "plan": planned},
        "relative_evidence_mass_error": relative,
        "method": plan.method,
        "seed": plan.seed,
        "work": cost.work,
        "largest_tensor": cost.largest_tensor,
        "largest_intermediate": {
            "entries": cost.largest_intermediate.entries,
            "retained": cost.largest_intermediate.retained,
            "labels": cost.largest_intermediate.retained.iter().map(|id| label_name(*id)).collect::<Vec<_>>(),
        },
    });
    write_json(&out.join("run.json"), &report)?;
    Ok(format!(
        "evidence {label}\nmethod {} seed {}\nevidence probability enumeration={exact:.17} plan={planned:.17} relative_error={relative:.3e}\nwork={} largest_tensor={}\nlargest intermediate entries={} labels={}\n",
        plan.method,
        plan.seed
            .map(|seed| seed.to_string())
            .unwrap_or_else(|| "none".to_string()),
        cost.work,
        cost.largest_tensor,
        cost.largest_intermediate.entries,
        label_list(&cost.largest_intermediate.retained)
    ))
}

fn read_plan(path: &str) -> Result<ScalarPlan, String> {
    let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
    serde_json::from_str(&text).map_err(|err| format!("order plan: {err}"))
}

fn ensure_same_query(
    model: &Model,
    evidence: &[(usize, usize)],
    plan: &ScalarPlan,
) -> Result<(), String> {
    if query_inputs(model, evidence) != plan.inputs || !plan.output.is_empty() {
        return Err("saved plan does not match this model and evidence".to_string());
    }
    Ok(())
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value).map_err(|err| err.to_string())?;
    fs::write(path, json + "\n").map_err(|err| err.to_string())
}

fn label_name(id: usize) -> &'static str {
    ["A", "T", "S", "L", "B", "E", "X", "D"]
        .get(id)
        .copied()
        .unwrap_or("?")
}

fn label_list(labels: &[usize]) -> String {
    labels
        .iter()
        .map(|id| label_name(*id))
        .collect::<Vec<_>>()
        .join(",")
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
