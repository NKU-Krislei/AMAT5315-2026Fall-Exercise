use crate::{energy, force};
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug)]
pub enum Interaction {
    Open,
    Periodic { box_xy: [f64; 2], rc: f64 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForceMode {
    Naive,
    #[default]
    Cells,
}

impl fmt::Display for ForceMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForceMode::Naive => write!(f, "naive"),
            ForceMode::Cells => write!(f, "cells"),
        }
    }
}

impl FromStr for ForceMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "naive" => Ok(ForceMode::Naive),
            "cells" => Ok(ForceMode::Cells),
            other => Err(format!("unknown --force {other}; use naive or cells")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub accelerations: Vec<[f64; 2]>,
    pub interaction: Interaction,
    pub force_mode: ForceMode,
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
            force_mode: ForceMode::Naive,
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
        sys.force_mode = ForceMode::Cells;
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

fn use_cells(system: &System) -> bool {
    matches!(
        (system.interaction, system.force_mode),
        (Interaction::Periodic { .. }, ForceMode::Cells)
    )
}

fn cell_grid(box_xy: [f64; 2], rc: f64) -> Option<(usize, usize, f64, f64)> {
    let nx = (box_xy[0] / rc).floor() as usize;
    let ny = (box_xy[1] / rc).floor() as usize;
    if nx < 1 || ny < 1 {
        return None;
    }
    Some((nx, ny, box_xy[0] / nx as f64, box_xy[1] / ny as f64))
}

fn bin_coord(x: f64, width: f64, n: usize) -> usize {
    let c = (x / width).floor() as isize;
    let n = n as isize;
    ((c.rem_euclid(n)) as usize).min(n as usize - 1)
}

fn neighbour_cells(cx: usize, cy: usize, nx: usize, ny: usize) -> Vec<(usize, usize)> {
    let mut cells = Vec::with_capacity(9);
    for dy in -1..=1_isize {
        for dx in -1..=1_isize {
            let ncx = (cx as isize + dx).rem_euclid(nx as isize) as usize;
            let ncy = (cy as isize + dy).rem_euclid(ny as isize) as usize;
            if !cells.contains(&(ncx, ncy)) {
                cells.push((ncx, ncy));
            }
        }
    }
    cells
}

fn build_cells(system: &System, nx: usize, ny: usize, wx: f64, wy: f64) -> Vec<Vec<usize>> {
    let mut cells = vec![Vec::new(); nx * ny];
    for (i, p) in system.positions.iter().enumerate() {
        let cx = bin_coord(p[0], wx, nx);
        let cy = bin_coord(p[1], wy, ny);
        cells[cy * nx + cx].push(i);
    }
    cells
}

fn pair_force(system: &System, i: usize, j: usize, rc: f64) -> Option<(f64, f64)> {
    let (dx, dy, r) = pair_vector(system, i, j);
    if r >= rc || r == 0.0 {
        return None;
    }
    let f = force(r);
    Some((f * dx / r, f * dy / r))
}

fn for_each_unique_pair<F>(system: &System, mut visit: F)
where
    F: FnMut(usize, usize),
{
    let n = system.n_atoms();
    if !use_cells(system) {
        for i in 0..n {
            for j in (i + 1)..n {
                visit(i, j);
            }
        }
        return;
    }
    let Interaction::Periodic { box_xy, rc } = system.interaction else {
        return;
    };
    let Some((nx, ny, wx, wy)) = cell_grid(box_xy, rc) else {
        for i in 0..n {
            for j in (i + 1)..n {
                visit(i, j);
            }
        }
        return;
    };
    let occupants = build_cells(system, nx, ny, wx, wy);
    for cy in 0..ny {
        for cx in 0..nx {
            let mine = occupants[cy * nx + cx].clone();
            let neigh = neighbour_cells(cx, cy, nx, ny);
            for i in mine {
                for &(ncx, ncy) in &neigh {
                    for &j in &occupants[ncy * nx + ncx] {
                        if j > i {
                            visit(i, j);
                        }
                    }
                }
            }
        }
    }
}

pub fn compute_accelerations(system: &mut System) {
    for a in &mut system.accelerations {
        *a = [0.0, 0.0];
    }
    let rc = match system.interaction {
        Interaction::Open => f64::INFINITY,
        Interaction::Periodic { rc, .. } => rc,
    };
    let mut deltas = vec![[0.0, 0.0]; system.n_atoms()];
    for_each_unique_pair(system, |i, j| {
        if let Some((fx, fy)) = pair_force(system, i, j, rc) {
            deltas[i][0] += fx;
            deltas[i][1] += fy;
            deltas[j][0] -= fx;
            deltas[j][1] -= fy;
        }
    });
    system.accelerations = deltas;
}

pub fn kinetic_energy(system: &System) -> f64 {
    system
        .velocities
        .iter()
        .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
        .sum()
}

pub fn potential_energy(system: &System) -> f64 {
    let (rc, shift) = match system.interaction {
        Interaction::Open => (f64::INFINITY, 0.0),
        Interaction::Periodic { rc, .. } => (rc, energy(rc)),
    };
    let mut u = 0.0;
    for_each_unique_pair(system, |i, j| {
        let (_, _, r) = pair_vector(system, i, j);
        if r < rc {
            u += energy(r) - shift;
        }
    });
    u
}

pub fn total_energy(system: &System) -> f64 {
    kinetic_energy(system) + potential_energy(system)
}
