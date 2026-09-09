use clap::{Parser, Subcommand};
use md::{run_simulation, RunParams};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "md")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run {
        #[arg(long, default_value_t = 100)]
        n: usize,
        #[arg(long, default_value_t = 0.8)]
        rho: f64,
        #[arg(long, default_value_t = 0.5)]
        temperature: f64,
        #[arg(long, default_value_t = 0.01)]
        dt: f64,
        #[arg(long, default_value_t = 2000)]
        eq_steps: usize,
        #[arg(long, default_value_t = 10000)]
        steps: usize,
        #[arg(long, default_value_t = 50)]
        sample_every: usize,
        #[arg(long, default_value_t = 2026)]
        seed: u64,
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,
    },
    Check {
        dir: PathBuf,
    },
    Video {
        dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Commands::Run {
            n,
            rho,
            temperature,
            dt,
            eq_steps,
            steps,
            sample_every,
            seed,
            out,
        } => {
            let params = RunParams {
                n,
                rho,
                temperature,
                dt,
                eq_steps,
                steps,
                sample_every,
                seed,
            };
            if let Err(e) = run_simulation(&params, &out) {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Commands::Check { .. } => {
            eprintln!("md check is not implemented yet");
            ExitCode::FAILURE
        }
        Commands::Video { .. } => {
            eprintln!("md video is not implemented yet");
            ExitCode::FAILURE
        }
    }
}
