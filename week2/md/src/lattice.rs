use crate::system::{kinetic_energy, System};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

pub fn triangular_lattice(n: usize, rho: f64) -> (Vec<[f64; 2]>, [f64; 2]) {
    let n_side = (n as f64).sqrt().round() as usize;
    assert_eq!(n_side * n_side, n, "n must be a square");
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 3.0_f64.sqrt() / 2.0 * a;
    let box_xy = [n_side as f64 * a, n_side as f64 * h];
    let mut positions = Vec::with_capacity(n);
    for j in 0..n_side {
        for i in 0..n_side {
            let x = (i as f64 + 0.5 * (j % 2) as f64) * a;
            let y = j as f64 * h;
            positions.push([x, y]);
        }
    }
    (positions, box_xy)
}

fn unit_gaussian(rng: &mut StdRng) -> f64 {
    let u1 = rng.r#gen::<f64>().max(1e-12);
    let u2 = rng.r#gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

pub fn seed_maxwell_boltzmann(system: &mut System, temperature: f64, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let s = temperature.sqrt();
    for v in &mut system.velocities {
        v[0] = s * unit_gaussian(&mut rng);
        v[1] = s * unit_gaussian(&mut rng);
    }
    let n = system.n_atoms() as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for v in &system.velocities {
        mx += v[0];
        my += v[1];
    }
    mx /= n;
    my /= n;
    for v in &mut system.velocities {
        v[0] -= mx;
        v[1] -= my;
    }
    rescale_temperature(system, temperature);
}

pub fn rescale_temperature(system: &mut System, temperature: f64) {
    let n = system.n_atoms();
    assert!(n > 1);
    let t_thermo = 2.0 * kinetic_energy(system) / (2 * n - 2) as f64;
    let factor = (temperature / t_thermo).sqrt();
    for v in &mut system.velocities {
        v[0] *= factor;
        v[1] *= factor;
    }
}
