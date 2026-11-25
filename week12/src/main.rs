//! Command line for the week 12 spin-glass study.
//!
//! stdout is a tab-separated table. The raw evidence is `<out>/run.json`.

mod glass;
mod lp;
mod nn;
mod search;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use serde_json::{json, Value};

use glass::{random_instance, square, tetrahedron, SpinGlass};
use lp::{fractional_count, solve_diet, LpModel, MilpModel};
use nn::Mlp;
use search::{search, Policy, SearchOutput};

struct Args {
    mode: String,
    spins: usize,
    edges: usize,
    count: usize,
    seed: u64,
    out: PathBuf,
    policy: Option<Policy>,
    export_labels: bool,
    model: Option<PathBuf>,
}

fn main() {
    let args = parse_args();
    fs::create_dir_all(&args.out).unwrap_or_else(|err| {
        panic!("cannot create {}: {err}", args.out.display());
    });
    match args.mode.as_str() {
        "gap" => run_gap(&args),
        "bnb" => run_bnb(&args),
        "eval" => run_eval(&args),
        other => {
            eprintln!("unknown mode {other}; expected gap, bnb, or eval");
            process::exit(2);
        }
    }
}

fn run_gap(args: &Args) {
    let started = Instant::now();
    let square_glass = square();
    let square_lp = solve_root_lp(&square_glass);
    let (square_exact, square_x) = MilpModel::new(&square_glass).solve();
    let square_rounded = square_glass.rounded_energy(&square_lp.x);
    let (diet_x, diet_cost) = solve_diet();

    let mut instances = Vec::with_capacity(args.count);
    for id in 0..args.count {
        let glass = random_instance(args.spins, args.edges, args.seed, id as u64);
        let relaxed = solve_root_lp(&glass);
        let (exact, _) = MilpModel::new(&glass).solve();
        let rounded = glass.rounded_energy(&relaxed.x);
        let fractional = fractional_count(&relaxed.x);
        instances.push(json!({
            "id": id,
            "lp": relaxed.energy,
            "exact": exact,
            "gap": exact - relaxed.energy,
            "fractional_spins": fractional,
            "rounded": rounded,
            "excess": rounded - exact,
            "lp_x": relaxed.x,
        }));
        eprintln!(
            "gap instance {id}: lp {:.6} exact {exact:.6} excess {:.6}",
            relaxed.energy,
            rounded - exact
        );
    }

    let mean = |key: &str| mean_of(&instances, key);
    let suboptimal = instances
        .iter()
        .filter(|item| item["excess"].as_f64().unwrap_or(0.0) > 1e-6)
        .count();
    let runtime_ms = elapsed_ms(started);
    let summary = json!({
        "square_lp": square_lp.energy,
        "square_lp_x": square_lp.x,
        "square_exact": square_exact,
        "square_exact_x": square_x,
        "square_rounded": square_rounded,
        "diet_x": diet_x,
        "diet_cost": diet_cost,
        "mean_lp": mean("lp"),
        "mean_exact": mean("exact"),
        "mean_gap": mean("gap"),
        "mean_fractional_spins": mean("fractional_spins"),
        "suboptimal_roundings": suboptimal,
        "mean_rounding_excess": mean("excess"),
        "runtime_ms": runtime_ms,
    });
    let document = json!({
        "schema": "branchopt.v1",
        "mode": "gap",
        "seed": args.seed,
        "parameters": parameters(args),
        "summary": summary,
        "evidence": {"instances": instances},
    });
    write_run(&args.out, &document);
    print_table(&[
        row("gap", "square_lp", square_lp.energy, runtime_ms),
        row("gap", "square_exact", square_exact, runtime_ms),
        row("gap", "square_rounded", square_rounded, runtime_ms),
        row("gap", "diet_cost", diet_cost, runtime_ms),
        row("gap", "mean_lp", summary["mean_lp"].as_f64().unwrap(), runtime_ms),
        row("gap", "mean_exact", summary["mean_exact"].as_f64().unwrap(), runtime_ms),
        row("gap", "mean_gap", summary["mean_gap"].as_f64().unwrap(), runtime_ms),
        row(
            "gap",
            "mean_fractional_spins",
            summary["mean_fractional_spins"].as_f64().unwrap(),
            runtime_ms,
        ),
        row("gap", "suboptimal_roundings", suboptimal as f64, runtime_ms),
        row(
            "gap",
            "mean_rounding_excess",
            summary["mean_rounding_excess"].as_f64().unwrap(),
            runtime_ms,
        ),
    ]);
}

fn run_bnb(args: &Args) {
    let started = Instant::now();
    let policy = args.policy.expect("--policy is required for bnb");
    if args.export_labels && policy != Policy::Strong {
        eprintln!("--export-labels records Strong Branching choices; pass --policy strong");
        process::exit(2);
    }
    let tetra = search(&tetrahedron(), Policy::MostFrac, None, true, None, 0);
    let mut instances = Vec::with_capacity(args.count);
    let mut train_labels = Vec::new();
    let mut validation_labels = Vec::new();
    for id in 0..args.count {
        let glass = random_instance(args.spins, args.edges, args.seed, id as u64);
        let label_limit = args.export_labels.then_some(80);
        let result = search(&glass, policy, None, false, label_limit, id as u64);
        let (highs, _) = MilpModel::new(&glass).solve();
        let error = (result.energy - highs).abs();
        instances.push(json!({
            "id": id,
            "energy": result.energy,
            "highs": highs,
            "error": error,
            "nodes": result.nodes,
            "lp_solves": result.lp_solves,
            "time_ms": result.wall_ms,
        }));
        let bucket = if id < 36 {
            &mut train_labels
        } else {
            &mut validation_labels
        };
        for label in result.labels {
            bucket.push(label_json(&label));
        }
        eprintln!(
            "bnb {} instance {id}: energy {:.6} nodes {} lps {} error {:.3e}",
            policy.name(),
            result.energy,
            result.nodes,
            result.lp_solves,
            error
        );
    }
    let runtime_ms = elapsed_ms(started);
    let summary = json!({
        "policy": policy.name(),
        "mean_nodes": mean_of(&instances, "nodes"),
        "mean_lp_solves": mean_of(&instances, "lp_solves"),
        "mean_time_ms": mean_of(&instances, "time_ms"),
        "max_abs_error": max_of(&instances, "error"),
        "runtime_ms": runtime_ms,
        "train_nodes": train_labels.len(),
        "validation_nodes": validation_labels.len(),
    });
    let document = json!({
        "schema": "branchopt.v1",
        "mode": "bnb",
        "seed": args.seed,
        "parameters": parameters(args),
        "summary": summary,
        "evidence": {
            "tetrahedron": tetra.trace.iter().map(trace_json).collect::<Vec<_>>(),
            "instances": instances,
        },
    });
    write_run(&args.out, &document);
    if args.export_labels {
        let dataset = json!({
            "schema": "branchopt.labels.v1",
            "seed": args.seed,
            "spins": args.spins,
            "edges": args.edges,
            "train": train_labels,
            "validation": validation_labels,
        });
        let path = args.out.join("dataset.json");
        fs::write(&path, serde_json::to_string(&dataset).expect("dataset json"))
            .unwrap_or_else(|err| panic!("failed to write {}: {err}", path.display()));
        eprintln!(
            "wrote {} train nodes and {} validation nodes",
            dataset["train"].as_array().unwrap().len(),
            dataset["validation"].as_array().unwrap().len()
        );
    }
    print_table(&[
        row("bnb", "mean_nodes", summary["mean_nodes"].as_f64().unwrap(), runtime_ms),
        row(
            "bnb",
            "mean_lp_solves",
            summary["mean_lp_solves"].as_f64().unwrap(),
            runtime_ms,
        ),
        row(
            "bnb",
            "mean_time_ms",
            summary["mean_time_ms"].as_f64().unwrap(),
            runtime_ms,
        ),
        row(
            "bnb",
            "max_abs_error",
            summary["max_abs_error"].as_f64().unwrap(),
            runtime_ms,
        ),
        row("bnb", "tetrahedron_nodes", tetra.nodes as f64, runtime_ms),
    ]);
}

fn run_eval(args: &Args) {
    let started = Instant::now();
    let model_path = args
        .model
        .clone()
        .expect("--model is required for eval");
    let mlp = Mlp::load(&model_path);
    let probe_diffs = mlp.probe_differences();
    let max_probe_diff = probe_diffs.iter().copied().fold(0.0, f64::max);
    let policies = [
        Policy::MostFrac,
        Policy::Strong,
        Policy::Neural,
        Policy::RootNeural,
    ];
    let mut policy_evidence = serde_json::Map::new();
    let mut policy_summary = serde_json::Map::new();
    let mut stored: Vec<Vec<Value>> = Vec::new();
    for policy in policies {
        let mut instances = Vec::with_capacity(args.count);
        for id in 0..args.count {
            let glass = random_instance(args.spins, args.edges, args.seed, id as u64);
            let result = search(&glass, policy, Some(&mlp), false, None, id as u64);
            let (highs, _) = MilpModel::new(&glass).solve();
            instances.push(instance_eval_json(&result, id, highs));
            eprintln!(
                "eval {} {} instance {id}: nodes {} time {:.2} ms infer {:.3} ms",
                args.spins,
                policy.name(),
                result.nodes,
                result.wall_ms,
                result.infer_ms
            );
        }
        policy_summary.insert(
            policy.name().to_string(),
            json!({
                "mean_nodes": mean_of(&instances, "nodes"),
                "mean_lp_solves": mean_of(&instances, "lp_solves"),
                "mean_time_ms": mean_of(&instances, "time_ms"),
                "mean_lp_ms": mean_of(&instances, "lp_ms"),
                "mean_infer_ms": mean_of(&instances, "infer_ms"),
                "max_abs_error": max_of(&instances, "error"),
                "lp_equals_nodes": lp_equals_nodes(&instances),
            }),
        );
        stored.push(instances);
    }
    for (policy, instances) in policies.iter().zip(stored.iter()) {
        policy_evidence.insert(policy.name().to_string(), json!({ "instances": instances }));
    }
    let most = &stored[0];
    let neural = &stored[2];
    let mut best_index = 0usize;
    let mut best_saved = f64::NEG_INFINITY;
    for (index, (left, right)) in most.iter().zip(neural).enumerate() {
        let saved = left["nodes"].as_f64().unwrap() - right["nodes"].as_f64().unwrap();
        if saved > best_saved {
            best_saved = saved;
            best_index = index;
        }
    }
    let max_error = policies
        .iter()
        .map(|policy| {
            policy_summary[policy.name()]["max_abs_error"]
                .as_f64()
                .unwrap()
        })
        .fold(0.0, f64::max);
    let runtime_ms = elapsed_ms(started);
    let best_saving = json!({
        "instance": best_index,
        "energy": neural[best_index]["energy"],
        "nodes_saved": best_saved,
        "most_frac": {
            "splits": most[best_index]["splits"],
            "nodes": most[best_index]["nodes"],
            "time_ms": most[best_index]["time_ms"],
        },
        "neural": {
            "splits": neural[best_index]["splits"],
            "nodes": neural[best_index]["nodes"],
            "time_ms": neural[best_index]["time_ms"],
        },
    });
    let document = json!({
        "schema": "branchopt.v1",
        "mode": "eval",
        "seed": args.seed,
        "parameters": parameters(args),
        "summary": {
            "max_probe_diff": max_probe_diff,
            "max_abs_error": max_error,
            "runtime_ms": runtime_ms,
            "policies": Value::Object(policy_summary.clone()),
            "best_saving": best_saving,
        },
        "evidence": {
            "probes": probe_diffs.iter().enumerate().map(|(index, diff)| json!({
                "index": index,
                "max_abs_diff": diff,
            })).collect::<Vec<_>>(),
            "policies": Value::Object(policy_evidence),
        },
    });
    write_run(&args.out, &document);
    let mut rows = vec![
        row("eval", "max_probe_diff", max_probe_diff, runtime_ms),
        row("eval", "max_abs_error", max_error, runtime_ms),
    ];
    for policy in policies {
        let item = &policy_summary[policy.name()];
        rows.push(row(
            "eval",
            &format!("{}_mean_nodes", policy.name()),
            item["mean_nodes"].as_f64().unwrap(),
            runtime_ms,
        ));
        rows.push(row(
            "eval",
            &format!("{}_mean_time_ms", policy.name()),
            item["mean_time_ms"].as_f64().unwrap(),
            runtime_ms,
        ));
    }
    print_table(&rows);
}

fn solve_root_lp(glass: &SpinGlass) -> lp::Relaxation {
    let lower = vec![0.0; glass.n];
    let upper = vec![1.0; glass.n];
    LpModel::new(glass).solve(&lower, &upper)
}

fn instance_eval_json(result: &SearchOutput, id: usize, highs: f64) -> Value {
    json!({
        "id": id,
        "energy": result.energy,
        "highs": highs,
        "error": (result.energy - highs).abs(),
        "nodes": result.nodes,
        "lp_solves": result.lp_solves,
        "time_ms": result.wall_ms,
        "lp_ms": result.lp_ms,
        "infer_ms": result.infer_ms,
        "splits": result.splits,
    })
}

fn label_json(label: &search::LabelNode) -> Value {
    json!({
        "instance": label.instance,
        "candidates": label.candidates,
        "features": label.features,
        "gains0": label.gains0,
        "gains1": label.gains1,
        "scores": label.scores,
        "pruned_children": label.pruned_children,
    })
}

fn trace_json(node: &search::TraceNode) -> Value {
    json!({
        "node": node.node,
        "branch": node.branch,
        "status": node.status,
        "lp": node.lp,
        "x": node.x,
    })
}

fn parameters(args: &Args) -> Value {
    json!({
        "mode": args.mode,
        "spins": args.spins,
        "edges": args.edges,
        "count": args.count,
        "seed": args.seed,
        "policy": args.policy.map(|policy| policy.name()),
        "export_labels": args.export_labels,
        "model": args.model.as_ref().map(|path| path.display().to_string()),
        "presolve": false,
        "threads": 1,
    })
}

fn write_run(out: &PathBuf, document: &Value) {
    let path = out.join("run.json");
    fs::write(&path, serde_json::to_string_pretty(document).expect("run json"))
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", path.display()));
}

fn mean_of(items: &[Value], key: &str) -> f64 {
    if items.is_empty() {
        return 0.0;
    }
    items.iter().map(|item| item[key].as_f64().unwrap()).sum::<f64>() / items.len() as f64
}

fn max_of(items: &[Value], key: &str) -> f64 {
    items
        .iter()
        .map(|item| item[key].as_f64().unwrap())
        .fold(0.0, f64::max)
}

fn lp_equals_nodes(items: &[Value]) -> bool {
    items.iter().all(|item| item["nodes"] == item["lp_solves"])
}

struct TableRow {
    mode: String,
    metric: String,
    value: f64,
    runtime_ms: f64,
}

fn row(mode: &str, metric: &str, value: f64, runtime_ms: f64) -> TableRow {
    TableRow {
        mode: mode.to_string(),
        metric: metric.to_string(),
        value,
        runtime_ms,
    }
}

fn print_table(rows: &[TableRow]) {
    println!("mode\tmetric\tvalue\truntime_ms");
    for row in rows {
        println!(
            "{}\t{}\t{}\t{:.3}",
            row.mode,
            row.metric,
            format_value(row.value),
            row.runtime_ms
        );
    }
}

fn format_value(value: f64) -> String {
    if value != 0.0 && value.abs() < 1e-4 {
        format!("{value:.3e}")
    } else {
        format!("{value:.6}")
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn parse_args() -> Args {
    let mut mode = None;
    let mut spins = None;
    let mut edges = None;
    let mut count = None;
    let mut seed = None;
    let mut out = None;
    let mut policy = None;
    let mut export_labels = false;
    let mut model = None;
    let mut argv = env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--mode" => mode = Some(need(&mut argv, "--mode")),
            "--spins" => spins = Some(parse_usize(&need(&mut argv, "--spins"), "--spins")),
            "--edges" => edges = Some(parse_usize(&need(&mut argv, "--edges"), "--edges")),
            "--count" => count = Some(parse_usize(&need(&mut argv, "--count"), "--count")),
            "--seed" => seed = Some(parse_u64(&need(&mut argv, "--seed"), "--seed")),
            "--out" => out = Some(PathBuf::from(need(&mut argv, "--out"))),
            "--policy" => {
                let text = need(&mut argv, "--policy");
                policy = Some(Policy::parse(&text).unwrap_or_else(|| {
                    eprintln!("unknown policy {text}");
                    process::exit(2);
                }));
            }
            "--export-labels" => export_labels = true,
            "--model" => model = Some(PathBuf::from(need(&mut argv, "--model"))),
            "--help" | "-h" => {
                eprintln!(
                    "branchopt --mode gap|bnb|eval --spins N --edges M --count K --seed S --out DIR [--policy RULE] [--export-labels] [--model FILE]"
                );
                process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}");
                process::exit(2);
            }
        }
    }
    Args {
        mode: mode.expect("--mode is required"),
        spins: spins.expect("--spins is required"),
        edges: edges.expect("--edges is required"),
        count: count.expect("--count is required"),
        seed: seed.expect("--seed is required"),
        out: out.expect("--out is required"),
        policy,
        export_labels,
        model,
    }
}

fn need(argv: &mut impl Iterator<Item = String>, flag: &str) -> String {
    argv.next().unwrap_or_else(|| {
        eprintln!("{flag} needs a value");
        process::exit(2);
    })
}

fn parse_usize(text: &str, flag: &str) -> usize {
    text.parse().unwrap_or_else(|_| {
        eprintln!("{flag} must be an integer, got {text}");
        process::exit(2);
    })
}

fn parse_u64(text: &str, flag: &str) -> u64 {
    text.parse().unwrap_or_else(|_| {
        eprintln!("{flag} must be an integer, got {text}");
        process::exit(2);
    })
}
