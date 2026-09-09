use crate::{energy, force};

#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub accelerations: Vec<[f64; 2]>,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        let accelerations = vec![[0.0, 0.0]; positions.len()];
        Self {
            positions,
            velocities,
            accelerations,
        }
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

pub fn compute_accelerations(system: &mut System) {
    let n = system.n_atoms();
    for a in &mut system.accelerations {
        *a = [0.0, 0.0];
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = system.positions[i][0] - system.positions[j][0];
            let dy = system.positions[i][1] - system.positions[j][1];
            let r = (dx * dx + dy * dy).sqrt();
            let f = force(r);
            let fx = f * dx / r;
            let fy = f * dy / r;
            system.accelerations[i][0] += fx;
            system.accelerations[i][1] += fy;
            system.accelerations[j][0] -= fx;
            system.accelerations[j][1] -= fy;
        }
    }
}

pub fn kinetic_energy(system: &System) -> f64 {
    system
        .velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum()
}

pub fn potential_energy(system: &System) -> f64 {
    let n = system.n_atoms();
    let mut u = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = system.positions[i][0] - system.positions[j][0];
            let dy = system.positions[i][1] - system.positions[j][1];
            let r = (dx * dx + dy * dy).sqrt();
            u += energy(r);
        }
    }
    u
}

pub fn total_energy(system: &System) -> f64 {
    kinetic_energy(system) + potential_energy(system)
}
