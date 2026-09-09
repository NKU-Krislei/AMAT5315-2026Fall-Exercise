use md::{compute_accelerations, energy, potential_energy, System};

const RC: f64 = 2.5;

#[test]
fn periodic_forces_sum_to_zero() {
    let positions = vec![
        [1.0, 1.0],
        [3.5, 2.0],
        [6.0, 5.0],
        [8.0, 8.0],
        [2.0, 9.0],
    ];
    let velocities = vec![[0.0, 0.0]; 5];
    let mut sys = System::periodic(positions, velocities, [12.0, 12.0], RC);
    compute_accelerations(&mut sys);
    let mut fx = 0.0_f64;
    let mut fy = 0.0_f64;
    for a in &sys.accelerations {
        fx += a[0];
        fy += a[1];
    }
    assert!(fx.abs() < 1e-12, "sum Fx = {fx}");
    assert!(fy.abs() < 1e-12, "sum Fy = {fy}");
}

#[test]
fn cutoff_potential_continuous_from_inside() {
    let r_in = RC - 1e-4;
    let mut sys = System::periodic(
        vec![[0.0, 0.0], [r_in, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]],
        [20.0, 20.0],
        RC,
    );
    let u_in = potential_energy(&sys);
    let expected = energy(r_in) - energy(RC);
    assert!(
        (u_in - expected).abs() < 1e-12,
        "inside: U={u_in} expected={expected}"
    );
    assert!(
        energy(r_in).abs() > 1e-5,
        "unshifted U(r) must not already be ~0, else the shift is untested"
    );

    sys.positions[1][0] = RC + 1e-4;
    let u_out = potential_energy(&sys);
    assert!(u_out.abs() < 1e-15, "outside cutoff U={u_out}");
}
