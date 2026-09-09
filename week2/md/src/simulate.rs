use crate::integrators::{advance, VelocityVerlet};
use crate::lattice::{rescale_temperature, seed_maxwell_boltzmann, triangular_lattice};
use crate::system::{
    compute_accelerations, kinetic_energy, potential_energy, wrap_positions, System,
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const RC: f64 = 2.5;
pub const EQ_RESCALE_EVERY: usize = 50;

#[derive(Clone, Debug)]
pub struct RunParams {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
}

impl Default for RunParams {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
        }
    }
}

#[derive(Serialize)]
struct RunJson {
    n: usize,
    rho: f64,
    #[serde(rename = "box")]
    box_xy: [f64; 2],
    dt: f64,
    temperature: f64,
    eq_steps: usize,
    steps: usize,
    sample_every: usize,
    seed: u64,
    integrator: &'static str,
}

#[derive(Serialize)]
struct Frame {
    step: usize,
    t: f64,
    pos: Vec<[f64; 2]>,
    vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    e_pot: f64,
    #[serde(rename = "E_kin")]
    e_kin: f64,
}

pub fn run_simulation(params: &RunParams, out_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let (positions, box_xy) = triangular_lattice(params.n, params.rho);
    let velocities = vec![[0.0, 0.0]; params.n];
    let mut system = System::periodic(positions, velocities, box_xy, RC);
    seed_maxwell_boltzmann(&mut system, params.temperature, params.seed);
    compute_accelerations(&mut system);

    let integrator = VelocityVerlet;
    for step in 1..=params.eq_steps {
        advance(&integrator, &mut system, params.dt);
        wrap_positions(&mut system);
        if step % EQ_RESCALE_EVERY == 0 {
            rescale_temperature(&mut system, params.temperature);
            compute_accelerations(&mut system);
        }
    }

    let run = RunJson {
        n: params.n,
        rho: params.rho,
        box_xy,
        dt: params.dt,
        temperature: params.temperature,
        eq_steps: params.eq_steps,
        steps: params.steps,
        sample_every: params.sample_every,
        seed: params.seed,
        integrator: "velocity-verlet",
    };
    fs::write(
        out_dir.join("run.json"),
        serde_json::to_string_pretty(&run).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let mut traj =
        BufWriter::new(File::create(out_dir.join("traj.jsonl")).map_err(|e| e.to_string())?);
    for step in 1..=params.steps {
        advance(&integrator, &mut system, params.dt);
        wrap_positions(&mut system);
        if step % params.sample_every == 0 {
            let frame = Frame {
                step,
                t: step as f64 * params.dt,
                pos: system.positions.clone(),
                vel: system.velocities.clone(),
                e_pot: potential_energy(&system),
                e_kin: kinetic_energy(&system),
            };
            writeln!(
                traj,
                "{}",
                serde_json::to_string(&frame).map_err(|e| e.to_string())?
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
