//! `svd` command line tool: Part 1 hands-on (SVD compression evidence).
//!
//! Usage:
//!   svd gen   --out runs/svd     # generate matrix + spectrum + reconstructions
//!   svd check --out runs/svd     # independently recompute errors from the files

use std::path::PathBuf;
use std::process::ExitCode;

use thermo::svd_evidence::{self, GenSettings};

const SIZE: usize = 512;
const SEED: u64 = 2026;
const RANKS: [usize; 5] = [16, 64, 128, 256, 512];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: svd <gen|check> --out <dir>");
        return ExitCode::FAILURE;
    }
    let (cmd, out) = match (args[0].as_str(), args[1].as_str()) {
        ("gen" | "check", "--out") => (args[0].as_str(), PathBuf::from(&args[2])),
        _ => {
            eprintln!("usage: svd <gen|check> --out <dir>");
            return ExitCode::FAILURE;
        }
    };
    match cmd {
        "gen" => match svd_evidence::generate(
            &out,
            &GenSettings {
                size: SIZE,
                seed: SEED,
                ranks: RANKS.to_vec(),
            },
        ) {
            Ok(meta) => {
                println!(
                    "generated {}x{} matrix, seed {}, ranks {:?}, ||M||_F = {:.6}",
                    SIZE, SIZE, SEED, RANKS, meta["norm_fro"].as_f64().unwrap()
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("svd gen failed: {e}");
                ExitCode::FAILURE
            }
        },
        _ => match svd_evidence::check(&out) {
            Ok(rep) => {
                println!("independent recomputation from {}:", out.display());
                println!(
                    "  {:>5} {:>14} {:>14} {:>10} {:>12}",
                    "chi", "e_measured", "e_predicted", "|diff|", "factor/orig"
                );
                for r in &rep.rows {
                    println!(
                        "  {:>5} {:>14.9} {:>14.9} {:>10.2e} {:>12.3}",
                        r.chi,
                        r.measured,
                        r.predicted,
                        r.diff,
                        r.factor_entries as f64 / rep.original_entries as f64
                    );
                }
                println!("  ||M||_F = {:.6}", rep.norm_fro);
                println!(
                    "  max |measured - predicted| = {:.3e} (require < 1e-8): {}",
                    rep.max_diff,
                    if rep.max_diff < 1e-8 { "PASS" } else { "FAIL" }
                );
                println!(
                    "  full-rank error e_512 = {:.3e} (require < 1e-8): {}",
                    rep.full_rank_error,
                    if rep.full_rank_error < 1e-8 { "PASS" } else { "FAIL" }
                );
                println!(
                    "  two-factor storage equals the original at chi = {}",
                    rep.storage_parity_chi
                );
                println!(
                    "  error first reaches 10% at chi = {} (factor storage = {:.3}x original)",
                    rep.chi_10pct, rep.factor_over_original_at_10pct
                );
                if rep.passed(1e-8, 1e-8) {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(2)
                }
            }
            Err(e) => {
                eprintln!("svd check failed: {e}");
                ExitCode::FAILURE
            }
        },
    }
}
