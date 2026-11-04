use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use infer::{
    ad_tuberculosis_diagram, belief_propagation, compare, comparison_json, comparison_text,
    conditioned_plan, enumerate, execute_conditioned, execute_scalar, format_checks,
    log10_partition, matrix_and_ising, normalize_open, parse_evidence, query_inputs,
    read_evidence_file, read_log10_partition, read_marginals, recount, recount_plan, scalar_plan,
    treesa_settings, BpSettings, Model, OrderChoice, ScalarPlan,
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
    if flags.contains_key("evidence-file") {
        return evidence_file_case(&flags);
    }
    match flags.get("method").map(String::as_str) {
        Some("bp") => return bp_case(&flags),
        Some("exact") | None => {}
        Some(other) => return Err(format!("unknown method {other}; expected exact or bp")),
    }
    if ["damping", "tolerance", "max-sweeps"]
        .iter()
        .any(|name| flags.contains_key(*name))
    {
        return Err("damping, tolerance and max-sweeps are required with bp only".to_string());
    }
    let ordering = ["order", "seed", "plan", "analyze"]
        .iter()
        .any(|name| flags.contains_key(*name));
    if !ordering {
        reject_unknown(&flags, &["model", "evidence", "out", "method"])?;
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
            "model", "evidence", "out", "method", "order", "seed", "plan", "analyze",
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

fn evidence_file_case(
    flags: &std::collections::BTreeMap<String, String>,
) -> Result<String, String> {
    reject_unknown(
        flags,
        &[
            "model",
            "evidence-file",
            "queries",
            "out",
            "method",
            "order",
            "damping",
            "tolerance",
            "max-sweeps",
            "pr",
            "mar",
        ],
    )?;
    let model = Model::load(Path::new(required(flags, "model")?))?;
    let evidence = read_evidence_file(Path::new(required(flags, "evidence-file")?))?;
    let queries = parse_queries(required(flags, "queries")?)?;
    let published_log10 = read_log10_partition(Path::new(required(flags, "pr")?))?;
    let published = read_marginals(Path::new(required(flags, "mar")?))?;
    let out = Path::new(required(flags, "out")?);
    fs::create_dir_all(out).map_err(|err| err.to_string())?;
    for &query in &queries {
        let marginal = published
            .get(query)
            .ok_or_else(|| format!("query {query} is outside the MAR file"))?;
        let dim = *model
            .dims
            .get(query)
            .ok_or_else(|| format!("query {query} is outside the model"))?;
        if marginal.len() != dim {
            return Err(format!(
                "query {query} has no {dim}-state published marginal"
            ));
        }
    }
    match flags.get("method").map(String::as_str) {
        Some("bp") => bp_evidence_file(
            flags,
            &model,
            &evidence,
            &queries,
            &published,
            published_log10,
            out,
        ),
        Some("exact") | None => exact_evidence_file(
            flags,
            &model,
            &evidence,
            &queries,
            &published,
            published_log10,
            out,
        ),
        Some(other) => Err(format!("unknown method {other}; expected exact or bp")),
    }
}

fn exact_evidence_file(
    flags: &std::collections::BTreeMap<String, String>,
    model: &Model,
    evidence: &[(usize, usize)],
    queries: &[usize],
    published: &[Vec<f64>],
    published_log10: f64,
    out: &Path,
) -> Result<String, String> {
    if ["damping", "tolerance", "max-sweeps"]
        .iter()
        .any(|name| flags.contains_key(*name))
    {
        return Err("damping, tolerance and max-sweeps are required with bp only".to_string());
    }
    let choice = match flags.get("order").map(String::as_str) {
        Some("greedy") => OrderChoice::Greedy,
        Some("treesa") => OrderChoice::TreeSA,
        Some(other) => return Err(format!("unknown order {other}; expected greedy or treesa")),
        None => return Err("missing --order".to_string()),
    };
    let started = Instant::now();
    let plan = conditioned_plan(model, evidence, choice)?;
    let sizes = (0..model.dims.len())
        .map(|id| (id, model.dims[id]))
        .collect();
    let cost = recount(&plan.inputs, &plan.output, &plan.tree, &sizes)?;
    let partition = execute_conditioned(model, evidence, &plan, &[])?;
    let log10_z = log10_partition(&partition)?;
    let mut query_rows = Vec::new();
    let mut text = format!(
        "log10 partition {log10_z:.16} published {published_log10:.16} abs_error {:.3e}\n",
        (log10_z - published_log10).abs()
    );
    for &query in queries {
        let open = execute_conditioned(model, evidence, &plan, &[query])?;
        let marginal = normalize_open(&open)?;
        let error = max_abs_slice(&marginal, &published[query]);
        text.push_str(&format!(
            "query {query} marginal {} abs_error {error:.3e}\n",
            format_row(&marginal)
        ));
        query_rows.push(serde_json::json!({
            "id": query,
            "marginal": marginal,
            "published": published[query],
            "abs_error": error,
        }));
    }
    let elapsed = started.elapsed().as_secs_f64();
    let settings = if plan.method == "treesa" {
        serde_json::to_value(treesa_settings()).map_err(|err| err.to_string())?
    } else {
        serde_json::Value::Null
    };
    write_json(&out.join("order.json"), &plan)?;
    write_json(&out.join("complexity.json"), &cost)?;
    write_json(
        &out.join("run.json"),
        &serde_json::json!({
            "method": plan.method,
            "observations": evidence.iter().map(|(id, state)| serde_json::json!({"id": id, "state": state})).collect::<Vec<_>>(),
            "log10_partition": log10_z,
            "published_log10_partition": published_log10,
            "log10_partition_abs_error": (log10_z - published_log10).abs(),
            "queries": query_rows,
            "work": cost.work,
            "largest_tensor": cost.largest_tensor,
            "largest_intermediate_entries": cost.largest_intermediate.entries,
            "largest_intermediate_labels": cost.largest_intermediate.retained,
            "elapsed_seconds": elapsed,
            "treesa": settings,
        }),
    )?;
    text.push_str(&format!(
        "work {} largest_tensor {} largest_intermediate {}\nelapsed_seconds {elapsed:.3}\n",
        cost.work, cost.largest_tensor, cost.largest_intermediate.entries
    ));
    Ok(text)
}

fn bp_evidence_file(
    flags: &std::collections::BTreeMap<String, String>,
    model: &Model,
    evidence: &[(usize, usize)],
    queries: &[usize],
    published: &[Vec<f64>],
    _published_log10: f64,
    out: &Path,
) -> Result<String, String> {
    if flags.contains_key("order") {
        return Err("order is not used with belief propagation".to_string());
    }
    let settings = BpSettings {
        damping: required(flags, "damping")?
            .parse()
            .map_err(|err| format!("damping: {err}"))?,
        tolerance: required(flags, "tolerance")?
            .parse()
            .map_err(|err| format!("tolerance: {err}"))?,
        max_sweeps: required(flags, "max-sweeps")?
            .parse()
            .map_err(|err| format!("max-sweeps: {err}"))?,
    };
    let started = Instant::now();
    let result = belief_propagation(model, evidence, settings)?;
    let elapsed = started.elapsed().as_secs_f64();
    let query_error = |beliefs: &[Vec<f64>]| {
        queries
            .iter()
            .map(|&query| max_abs_slice(&beliefs[query], &published[query]))
            .fold(0.0, f64::max)
    };
    let history: Vec<serde_json::Value> = result
        .sweeps
        .iter()
        .enumerate()
        .map(|(index, sweep)| {
            let mut beliefs = serde_json::Map::new();
            for &query in queries {
                beliefs.insert(query.to_string(), serde_json::json!(sweep.beliefs[query]));
            }
            serde_json::json!({
                "sweep": index + 1,
                "residual": sweep.residual,
                "cumulative_work": sweep.cumulative_work,
                "max_query_error": query_error(&sweep.beliefs),
                "beliefs": beliefs,
            })
        })
        .collect();
    let final_error = query_error(&result.beliefs);
    let mut query_rows = Vec::new();
    for &query in queries {
        let error = max_abs_slice(&result.beliefs[query], &published[query]);
        query_rows.push(serde_json::json!({
            "id": query,
            "belief": result.beliefs[query],
            "published": published[query],
            "abs_error": error,
        }));
    }
    write_json(
        &out.join("run.json"),
        &serde_json::json!({
            "method": "bp",
            "observations": evidence.iter().map(|(id, state)| serde_json::json!({"id": id, "state": state})).collect::<Vec<_>>(),
            "damping": settings.damping,
            "tolerance": settings.tolerance,
            "max_sweeps": settings.max_sweeps,
            "converged": result.converged,
            "residual": result.residual,
            "cumulative_work": result.cumulative_work,
            "queries": query_rows,
            "max_query_error": final_error,
            "elapsed_seconds": elapsed,
            "history": history,
        }),
    )?;
    let mut text = String::new();
    for (index, sweep) in result.sweeps.iter().enumerate() {
        text.push_str(&format!(
            "sweep {} residual {:.3e} work {}\n",
            index + 1,
            sweep.residual,
            sweep.cumulative_work
        ));
    }
    text.push_str(&format!(
        "converged {} residual {:.3e} max_query_error {:.6} elapsed_seconds {elapsed:.3}\n",
        result.converged, result.residual, final_error
    ));
    Ok(text)
}

fn parse_queries(text: &str) -> Result<Vec<usize>, String> {
    text.split(',')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|err| format!("query {part}: {err}"))
        })
        .collect()
}

fn format_row(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.10}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn max_abs_slice(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn bp_case(flags: &std::collections::BTreeMap<String, String>) -> Result<String, String> {
    reject_unknown(
        flags,
        &[
            "model",
            "evidence",
            "out",
            "method",
            "damping",
            "tolerance",
            "max-sweeps",
        ],
    )?;
    let model = Model::load(Path::new(required(flags, "model")?))?;
    let label = required(flags, "evidence")?;
    let evidence = parse_evidence(label)?;
    let settings = BpSettings {
        damping: required(flags, "damping")?
            .parse()
            .map_err(|err| format!("damping: {err}"))?,
        tolerance: required(flags, "tolerance")?
            .parse()
            .map_err(|err| format!("tolerance: {err}"))?,
        max_sweeps: required(flags, "max-sweeps")?
            .parse()
            .map_err(|err| format!("max-sweeps: {err}"))?,
    };
    let out = Path::new(required(flags, "out")?);
    fs::create_dir_all(out).map_err(|err| err.to_string())?;
    let exact = enumerate(&model, &evidence);
    let result = belief_propagation(&model, &evidence, settings)?;
    let history: Vec<serde_json::Value> = result
        .sweeps
        .iter()
        .enumerate()
        .map(|(index, sweep)| {
            serde_json::json!({
                "sweep": index + 1,
                "residual": sweep.residual,
                "cumulative_work": sweep.cumulative_work,
                "marginal_error": max_marginal_error(&exact.marginals, &sweep.beliefs),
                "beliefs": named_beliefs(&sweep.beliefs),
            })
        })
        .collect();
    let report = serde_json::json!({
        "evidence": label,
        "observations": evidence.iter().map(|(id, state)| serde_json::json!({
            "id": id,
            "name": label_name(*id),
            "state": state,
        })).collect::<Vec<_>>(),
        "method": "bp",
        "damping": settings.damping,
        "tolerance": settings.tolerance,
        "max_sweeps": settings.max_sweeps,
        "converged": result.converged,
        "residual": result.residual,
        "cumulative_work": result.cumulative_work,
        "marginal_error": max_marginal_error(&exact.marginals, &result.beliefs),
        "node_marginals": {
            "bp": named_beliefs(&result.beliefs),
            "enumeration": named_beliefs(&exact.marginals),
        },
        "history": history,
    });
    write_json(&out.join("run.json"), &report)?;
    let mut text = String::new();
    for (index, sweep) in result.sweeps.iter().enumerate() {
        text.push_str(&format!(
            "sweep {} residual {:.3e} work {}\n",
            index + 1,
            sweep.residual,
            sweep.cumulative_work
        ));
    }
    text.push_str(&format!(
        "converged {} residual {:.3e} marginal_error {:.6}\n",
        result.converged,
        result.residual,
        max_marginal_error(&exact.marginals, &result.beliefs)
    ));
    Ok(text)
}

fn named_beliefs(beliefs: &[Vec<f64>]) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for (id, belief) in beliefs.iter().enumerate() {
        object.insert(label_name(id).to_string(), serde_json::json!(belief));
    }
    serde_json::Value::Object(object)
}

fn max_marginal_error(exact: &[Vec<f64>], beliefs: &[Vec<f64>]) -> f64 {
    exact
        .iter()
        .zip(beliefs)
        .flat_map(|(left, right)| left.iter().zip(right))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
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
