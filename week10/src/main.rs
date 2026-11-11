//! `thermo` command line tool: square-lattice Ising partition functions.
//!
//! Usage (see thermo.design.toml):
//!   thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
//!
//! No argument has a default; every run states its geometry.

use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::json;

use thermo::boundary::contract_open_square;
use thermo::evidence as ev;
use thermo::trg::contract_periodic_square;

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct Args {
    method: String,
    l: usize,
    k: f64,
    chi: usize,
    out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str| -> Result<String, String> {
        let pos = raw
            .iter()
            .position(|a| a == name)
            .ok_or_else(|| format!("missing required argument {name}"))?;
        raw.get(pos + 1)
            .cloned()
            .ok_or_else(|| format!("argument {name} needs a value"))
    };
    let method = get("--method")?;
    let l: usize = get("--l")?
        .parse()
        .map_err(|_| format!("--l must be an integer >= 2"))?;
    let k: f64 = get("--k")?
        .parse()
        .map_err(|_| format!("--k must be a number >= 0"))?;
    let chi: usize = get("--chi")?
        .parse()
        .map_err(|_| format!("--chi must be an integer >= 1"))?;
    let out = PathBuf::from(get("--out")?);
    if l < 2 {
        return Err("--l must be at least 2".to_string());
    }
    if !k.is_finite() || k < 0.0 {
        return Err("--k must be finite and >= 0".to_string());
    }
    if chi < 1 {
        return Err("--chi must be at least 1".to_string());
    }
    if method != "boundary" && method != "trg" {
        return Err(format!("unknown method {method}"));
    }
    if method == "trg" && !l.is_power_of_two() {
        return Err("--l must be a power of two for TRG".to_string());
    }
    Ok(Args { method, l, k, chi, out })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("thermo: {e}");
            return ExitCode::FAILURE;
        }
    };
    match args.method.as_str() {
        "boundary" => run_boundary(&args),
        "trg" => run_trg(&args),
        _ => ExitCode::FAILURE,
    }
}

fn run_trg(args: &Args) -> ExitCode {
    let res = contract_periodic_square(args.l, args.k, args.chi);
    if let Err(e) = write_trg_evidence(args, &res) {
        eprintln!("thermo: writing evidence failed: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "trg (periodic square) L={} K={} chi={}: log Z = {:.12}, g = {:.12}, solver {:.3}s",
        args.l, args.k, args.chi, res.log_z, res.g, res.elapsed_sec
    );
    ExitCode::SUCCESS
}

fn run_boundary(args: &Args) -> ExitCode {
    let res = contract_open_square(args.l, args.k, args.chi);
    if let Err(e) = write_boundary_evidence(args, &res) {
        eprintln!("thermo: writing evidence failed: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "boundary (open square) L={} K={} chi={}: log Z = {:.12}, g = {:.12}, solver {:.3}s",
        args.l, args.k, args.chi, res.log_z, res.g, res.elapsed_sec
    );
    ExitCode::SUCCESS
}

fn write_boundary_evidence(args: &Args, res: &thermo::boundary::BoundaryResult) -> std::io::Result<()> {
    ev::ensure_dir(&args.out)?;
    let mut dims = Vec::new();
    let mut flat: Vec<f64> = Vec::new();
    for core in &res.cores {
        dims.push([core.dl, core.dp, core.dr]);
        flat.extend_from_slice(&core.data);
    }
    ev::write_f64_bin(&args.out.join("final_cores.bin"), &flat)?;
    let max_disc = res
        .row_diag
        .iter()
        .map(|d| d.max_discarded)
        .fold(0.0f64, f64::max);
    let total_time = res.elapsed_sec;
    let run = json!({
        "tool": "thermo",
        "version": VERSION,
        "method": "boundary",
        "model": "square-lattice Ising",
        "hamiltonian": "H = -sum_edges s_i*s_j, s_i in {+1,-1}, J = k_B = 1, no field",
        "geometry": "open",
        "settings": {
            "l": args.l,
            "k": args.k,
            "chi": args.chi,
        },
        "log_z": res.log_z,
        "g": res.g,
        "ell": res.ell,
        "closure_value": res.closure_value,
        "row_scales": res.row_diag.iter().map(|d| d.scale).collect::<Vec<_>>(),
        "truncation": {
            "max_discarded_relative_weight": max_disc,
            "per_row": res.row_diag,
        },
        "final_cores_file": "final_cores.bin",
        "final_cores_dims": dims,
        "recompute_hint": "contract the cores in order, closing every physical leg with b(K) = [1/sqrt(cosh K), 0]; log Z = ell + log(scalar)",
        "solver_time_sec": total_time,
        "timestamp_utc": ev::utc_timestamp(),
    });
    ev::write_json(&args.out.join("run.json"), &run)
}

fn write_trg_evidence(args: &Args, res: &thermo::trg::TrgResult) -> std::io::Result<()> {
    ev::ensure_dir(&args.out)?;
    // Per-step split factors: one binary per step, dims in the JSON.
    let mut steps_json = Vec::new();
    for st in &res.steps {
        let fname = format!("step{:02}_factors.bin", st.step);
        let mut flat: Vec<f64> = Vec::new();
        for f in &st.factors {
            flat.extend_from_slice(f);
        }
        ev::write_f64_bin(&args.out.join(&fname), &flat)?;
        steps_json.push(json!({
            "step": st.step,
            "num_tensors": st.num_tensors,
            "scale": st.scale,
            "kept": st.kept,
            "discarded": st.discarded,
            "singular_values_1": st.spectrum_1,
            "singular_values_2": st.spectrum_2,
            "factors_dims": st.factors_dims,
            "factors_file": fname,
        }));
    }
    ev::write_f64_bin(&args.out.join("final_tensor.bin"), &res.final_tensor)?;
    let run = json!({
        "tool": "thermo",
        "version": VERSION,
        "method": "trg",
        "model": "square-lattice Ising",
        "hamiltonian": "H = -sum_edges s_i*s_j, s_i in {+1,-1}, J = k_B = 1, no field",
        "geometry": "periodic",
        "settings": {
            "l": args.l,
            "k": args.k,
            "chi": args.chi,
        },
        "log_z": res.log_z,
        "g": res.g,
        "ell": res.ell,
        "closure_z": res.closure_z,
        "steps": steps_json,
        "final_tensor_file": "final_tensor.bin",
        "final_tensor_dims": res.final_dims,
        "recompute_hint": "log Z = ell + log(sum_{i,j} A[i,j,i,j]) from the saved final tensor",
        "solver_time_sec": res.elapsed_sec,
        "timestamp_utc": ev::utc_timestamp(),
    });
    ev::write_json(&args.out.join("run.json"), &run)
}
