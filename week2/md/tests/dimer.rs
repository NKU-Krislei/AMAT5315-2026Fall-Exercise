use md::{relative_energy_errors, Euler, System, VelocityVerlet};

#[test]
fn dimer_euler_drifts_verlet_conserves() {
    const DT: f64 = 0.01;
    const N: usize = 500;

    let positions = vec![[0.0, 0.0], [1.2, 0.0]];
    let velocities = vec![[0.0, 0.0], [0.0, 0.0]];

    let mut euler_system = System::new(positions.clone(), velocities.clone());
    let mut verlet_system = System::new(positions, velocities);

    let err_euler = relative_energy_errors(&Euler, &mut euler_system, DT, N);
    let err_verlet = relative_energy_errors(&VelocityVerlet, &mut verlet_system, DT, N);

    assert_eq!(err_euler.len(), N);
    assert_eq!(err_verlet.len(), N);

    let max_verlet = err_verlet.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    assert!(
        max_verlet < 1e-3,
        "Verlet max |err| = {max_verlet}, want < 1e-3"
    );

    let euler_final = err_euler[N - 1].abs();
    assert!(
        euler_final > 0.5,
        "Euler |err[499]| = {euler_final}, want > 0.5"
    );
}
