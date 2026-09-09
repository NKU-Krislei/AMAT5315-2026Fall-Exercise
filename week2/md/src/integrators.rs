use crate::system::{compute_accelerations, total_energy, System};

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}

pub struct Euler;

impl Integrator for Euler {
    fn step(&self, system: &mut System, dt: f64) {
        compute_accelerations(system);
        let n = system.n_atoms();
        for i in 0..n {
            system.positions[i][0] += dt * system.velocities[i][0];
            system.positions[i][1] += dt * system.velocities[i][1];
            system.velocities[i][0] += dt * system.accelerations[i][0];
            system.velocities[i][1] += dt * system.accelerations[i][1];
        }
    }
}

pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        let n = system.n_atoms();
        let half = 0.5 * dt;
        for i in 0..n {
            system.velocities[i][0] += half * system.accelerations[i][0];
            system.velocities[i][1] += half * system.accelerations[i][1];
            system.positions[i][0] += dt * system.velocities[i][0];
            system.positions[i][1] += dt * system.velocities[i][1];
        }
        compute_accelerations(system);
        for i in 0..n {
            system.velocities[i][0] += half * system.accelerations[i][0];
            system.velocities[i][1] += half * system.accelerations[i][1];
        }
    }
}

pub fn relative_energy_errors(
    method: &impl Integrator,
    system: &mut System,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let e0 = total_energy(system);
    let denom = e0.abs();
    compute_accelerations(system);
    let mut errors = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        advance(method, system, dt);
        let e = total_energy(system);
        errors.push((e - e0) / denom);
    }
    errors
}
