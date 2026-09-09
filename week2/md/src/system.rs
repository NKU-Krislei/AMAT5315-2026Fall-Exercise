use crate::{energy, force};

#[derive(Clone, Copy, Debug)]
pub enum Interaction {
    Open,
    Periodic { box_xy: [f64; 2], rc: f64 },
}

#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub accelerations: Vec<[f64; 2]>,
    pub interaction: Interaction,
}

impl System {
    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> Self {
        assert_eq!(positions.len(), velocities.len());
        let accelerations = vec![[0.0, 0.0]; positions.len()];
        Self {
            positions,
            velocities,
            accelerations,
            interaction: Interaction::Open,
        }
    }

    pub fn periodic(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_xy: [f64; 2],
        rc: f64,
    ) -> Self {
        let mut sys = Self::new(positions, velocities);
        sys.interaction = Interaction::Periodic { box_xy, rc };
        sys
    }

    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}

fn min_image(d: f64, length: f64) -> f64 {
    d - length * (d / length).round()
}

fn pair_vector(system: &System, i: usize, j: usize) -> (f64, f64, f64) {
    let mut dx = system.positions[i][0] - system.positions[j][0];
    let mut dy = system.positions[i][1] - system.positions[j][1];
    if let Interaction::Periodic { box_xy, .. } = system.interaction {
        dx = min_image(dx, box_xy[0]);
        dy = min_image(dy, box_xy[1]);
    }
    let r = (dx * dx + dy * dy).sqrt();
    (dx, dy, r)
}

pub fn wrap_positions(system: &mut System) {
    let Interaction::Periodic { box_xy, .. } = system.interaction else {
        return;
    };
    for p in &mut system.positions {
        p[0] = p[0].rem_euclid(box_xy[0]);
        p[1] = p[1].rem_euclid(box_xy[1]);
    }
}

pub fn compute_accelerations(system: &mut System) {
    let n = system.n_atoms();
    for a in &mut system.accelerations {
        *a = [0.0, 0.0];
    }
    let rc = match system.interaction {
        Interaction::Open => f64::INFINITY,
        Interaction::Periodic { rc, .. } => rc,
    };
    for i in 0..n {
        for j in (i + 1)..n {
            let (dx, dy, r) = pair_vector(system, i, j);
            if r >= rc || r == 0.0 {
                continue;
            }
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
    let (rc, shift) = match system.interaction {
        Interaction::Open => (f64::INFINITY, 0.0),
        Interaction::Periodic { rc, .. } => (rc, energy(rc)),
    };
    let mut u = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let (_, _, r) = pair_vector(system, i, j);
            if r >= rc {
                continue;
            }
            u += energy(r) - shift;
        }
    }
    u
}

pub fn total_energy(system: &System) -> f64 {
    kinetic_energy(system) + potential_energy(system)
}
