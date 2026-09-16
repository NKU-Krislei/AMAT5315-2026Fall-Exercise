pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Lennard-Jones pair energy \(U(r) = 4[r^{-12} - r^{-6}]\) in reduced units.
pub fn energy(r: f64) -> f64 {
    let r6 = r.powi(6);
    let r12 = r6 * r6;
    4.0 * (1.0 / r12 - 1.0 / r6)
}

/// Scalar pair force \(F(r) = 24/r [2r^{-12} - r^{-6}]\); positive means repulsion.
pub fn force(r: f64) -> f64 {
    let r6 = r.powi(6);
    let r12 = r6 * r6;
    (24.0 / r) * (2.0 / r12 - 1.0 / r6)
}

mod system;
mod integrators;
mod lattice;
mod simulate;
mod check;
mod rdf;
mod video;

pub use check::{check_artifacts, print_report};
pub use integrators::{advance, relative_energy_errors, Euler, Integrator, VelocityVerlet};
pub use simulate::{run_simulation, RunParams};
pub use video::write_video;
pub use system::{
    compute_accelerations, kinetic_energy, potential_energy, total_energy, wrap_positions,
    ForceMode, Interaction, System,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }

    #[test]
    fn well_depth() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn force_matches_energy_derivative() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        let h = 1e-5;
        // Separations straddle r0: inside the core, at the minimum, and in the well.
        let rs = [0.95, r0, 1.2, 1.5, 2.0];
        for r in rs {
            let numerical = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let f = force(r);
            let tol = 1e-6 * f.abs().max(1.0);
            assert!(
                (f - numerical).abs() < tol,
                "r={r}: force={f} numerical={numerical} tol={tol}"
            );
        }
    }
}
