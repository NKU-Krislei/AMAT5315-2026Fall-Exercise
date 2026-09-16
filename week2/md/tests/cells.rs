use md::{compute_accelerations, potential_energy, ForceMode, System};

const RC: f64 = 2.5;

fn assert_force_paths_agree(positions: Vec<[f64; 2]>, box_xy: [f64; 2]) {
    let n = positions.len();
    let vel = vec![[0.0, 0.0]; n];
    let mut naive = System::periodic(positions.clone(), vel.clone(), box_xy, RC);
    naive.force_mode = ForceMode::Naive;
    let mut cells = System::periodic(positions, vel, box_xy, RC);
    cells.force_mode = ForceMode::Cells;
    compute_accelerations(&mut naive);
    compute_accelerations(&mut cells);
    for i in 0..n {
        for k in 0..2 {
            let d = (naive.accelerations[i][k] - cells.accelerations[i][k]).abs();
            assert!(
                d < 1e-12,
                "a[{i}][{k}]: naive={} cells={} diff={d}",
                naive.accelerations[i][k],
                cells.accelerations[i][k]
            );
        }
    }
    let u_naive = potential_energy(&naive);
    let u_cells = potential_energy(&cells);
    let tol = 1e-12 * u_naive.abs().max(1.0);
    assert!(
        (u_naive - u_cells).abs() < tol,
        "U naive={u_naive} cells={u_cells} tol={tol}"
    );
}

#[test]
fn perturbed_lattice_matches_naive() {
    let mut positions = Vec::new();
    let a = 1.2;
    let n_side = 4;
    let box_xy = [n_side as f64 * a, n_side as f64 * a];
    let shifts = [0.07, -0.11, 0.19, -0.03];
    for j in 0..n_side {
        for i in 0..n_side {
            let x = (i as f64 * a + shifts[i] + box_xy[0]) % box_xy[0];
            let y = (j as f64 * a + shifts[j] + box_xy[1]) % box_xy[1];
            positions.push([x, y]);
        }
    }
    assert_force_paths_agree(positions, box_xy);
}

#[test]
fn pair_across_periodic_boundary_matches_naive() {
    let box_xy = [12.0, 12.0];
    let positions = vec![[0.2, 6.0], [11.7, 6.1], [5.0, 5.0]];
    assert_force_paths_agree(positions, box_xy);
}

#[test]
fn pairs_at_cutoff_match_naive() {
    let box_xy = [20.0, 20.0];
    let r_in = RC - 1e-4;
    let r_out = RC + 1e-4;
    assert_force_paths_agree(vec![[0.0, 0.0], [r_in, 0.0]], box_xy);
    assert_force_paths_agree(vec![[0.0, 0.0], [r_out, 0.0]], box_xy);
}

#[test]
fn two_cell_wide_box_matches_naive() {
    // nx = floor(5.0 / 2.5) = 2, so wrap maps both cells onto each other twice.
    let box_xy = [5.0, 5.0];
    let positions = vec![[0.3, 0.4], [2.6, 0.5], [0.4, 2.7], [2.8, 2.9]];
    assert_force_paths_agree(positions, box_xy);
}
