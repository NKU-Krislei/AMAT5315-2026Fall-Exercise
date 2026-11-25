//! HiGHS LP relaxation and exact MILP for one spin glass.
//!
//! Presolve is off and each LP uses the simplex solver, so the relaxation is
//! solved as written and returns a vertex. A fresh model is built for every
//! call so a previous basis cannot change which optimal vertex comes back.

use highs::{Col, HighsModelStatus, Model, RowProblem, Sense};

use crate::glass::SpinGlass;

const FRAC_TOL: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct Relaxation {
    pub optimal: bool,
    pub energy: f64,
    pub x: Vec<f64>,
    pub z: Vec<f64>,
    pub row_duals: Vec<f64>,
}

pub struct LpModel {
    template: RowProblem,
    x_cols: Vec<Col>,
    n: usize,
    m: usize,
    constant: f64,
}

impl LpModel {
    pub fn new(glass: &SpinGlass) -> Self {
        let (template, x_cols) = build_problem(glass, false);
        Self {
            template,
            x_cols,
            n: glass.n,
            m: glass.m(),
            constant: glass.constant(),
        }
    }

    pub fn solve(&self, lower: &[f64], upper: &[f64]) -> Relaxation {
        let mut model = self.template.clone().optimise(Sense::Minimise);
        configure(&mut model, true);
        for i in 0..self.n {
            model.change_column_bounds(self.x_cols[i], lower[i]..=upper[i]);
        }
        read_relaxation(model.solve(), self.n, self.m, self.constant)
    }
}

pub struct MilpModel {
    template: RowProblem,
    n: usize,
    constant: f64,
    glass: SpinGlass,
}

impl MilpModel {
    pub fn new(glass: &SpinGlass) -> Self {
        let (template, _) = build_problem(glass, true);
        Self {
            template,
            n: glass.n,
            constant: glass.constant(),
            glass: glass.clone(),
        }
    }

    pub fn solve(&self) -> (f64, Vec<f64>) {
        let mut model = self.template.clone().optimise(Sense::Minimise);
        configure(&mut model, false);
        let solved = model.solve();
        assert_eq!(
            solved.status(),
            HighsModelStatus::Optimal,
            "exact spin-glass MILP was not optimal"
        );
        let x = solved.get_solution().columns()[..self.n].to_vec();
        let energy = solved.objective_value() + self.constant;
        let spins = SpinGlass::bits_to_spins(&x);
        let recomputed = self.glass.energy_spins(&spins);
        assert!(
            (recomputed - energy).abs() < 1e-5,
            "MILP energy {energy} disagrees with spin energy {recomputed}"
        );
        (energy, x)
    }
}

pub fn solve_diet() -> (Vec<f64>, f64) {
    let mut problem = RowProblem::default();
    let x1 = problem.add_column(6.0, 0.0..);
    let x2 = problem.add_column(3.5, 0.0..);
    problem.add_row(8.0.., &[(x2, 7.0)]);
    problem.add_row(15.0.., &[(x1, 4.0), (x2, 2.0)]);
    problem.add_row(3.0.., &[(x1, 2.0), (x2, 1.0)]);
    let mut model = problem.optimise(Sense::Minimise);
    configure(&mut model, true);
    let solved = model.solve();
    assert_eq!(solved.status(), HighsModelStatus::Optimal, "diet LP failed");
    (
        solved.get_solution().columns().to_vec(),
        solved.objective_value(),
    )
}

pub fn fractional_count(x: &[f64]) -> usize {
    free_indices(x).len()
}

pub fn free_indices(x: &[f64]) -> Vec<usize> {
    x.iter()
        .enumerate()
        .filter(|(_, &xi)| xi > FRAC_TOL && xi < 1.0 - FRAC_TOL)
        .map(|(i, _)| i)
        .collect()
}

fn build_problem(glass: &SpinGlass, integer_spins: bool) -> (RowProblem, Vec<Col>) {
    let mut problem = RowProblem::default();
    let mut x_cols = Vec::with_capacity(glass.n);
    for i in 0..glass.n {
        let cost = -2.0 * glass.fields[i];
        let column = if integer_spins {
            problem.add_integer_column(cost, 0.0..=1.0)
        } else {
            problem.add_column(cost, 0.0..=1.0)
        };
        x_cols.push(column);
        let _ = i;
    }
    let mut z_cols = Vec::with_capacity(glass.m());
    for e in 0..glass.m() {
        z_cols.push(problem.add_column(-2.0 * glass.couplings[e], 0.0..=1.0));
    }
    for (e, &(u, v)) in glass.edges.iter().enumerate() {
        let z = z_cols[e];
        let xu = x_cols[u];
        let xv = x_cols[v];
        // z <= xu + xv
        problem.add_row(..=0.0, &[(z, 1.0), (xu, -1.0), (xv, -1.0)]);
        // z <= 2 - xu - xv
        problem.add_row(..=2.0, &[(z, 1.0), (xu, 1.0), (xv, 1.0)]);
        // z >= xu - xv
        problem.add_row(..=0.0, &[(xu, 1.0), (xv, -1.0), (z, -1.0)]);
        // z >= xv - xu
        problem.add_row(..=0.0, &[(xu, -1.0), (xv, 1.0), (z, -1.0)]);
    }
    (problem, x_cols)
}

fn read_relaxation(
    solved: highs::SolvedModel,
    n: usize,
    m: usize,
    constant: f64,
) -> Relaxation {
    if solved.status() != HighsModelStatus::Optimal {
        return Relaxation {
            optimal: false,
            energy: f64::INFINITY,
            x: Vec::new(),
            z: Vec::new(),
            row_duals: Vec::new(),
        };
    }
    let solution = solved.get_solution();
    let cols = solution.columns();
    Relaxation {
        optimal: true,
        energy: solved.objective_value() + constant,
        x: cols[..n].to_vec(),
        z: cols[n..n + m].to_vec(),
        row_duals: solution.dual_rows().to_vec(),
    }
}

fn configure(model: &mut Model, simplex: bool) {
    model.make_quiet();
    model.set_option("presolve", "off");
    model.set_option("threads", 1_i32);
    model.set_option("parallel", "off");
    if simplex {
        model.set_option("solver", "simplex");
    }
}
