//! Part 2 verification checks from the learning sheet.
//!
//! - open 4x4 square at chi=8 against spin enumeration at K=0, 0.44, 0.7
//!   within 1e-8;
//! - at K=0.44, c_V from finite differences (h=0.0025) against
//!   K^2 Var(H)/16 within 1e-3.

use thermo::boundary::contract_open_square;
use thermo::ising::enumerate;

fn g_of(l: usize, k: f64, chi: usize) -> f64 {
    contract_open_square(l, k, chi).g
}

#[test]
fn open_4x4_matches_enumeration() {
    for k in [0.0f64, 0.44, 0.7] {
        let res = contract_open_square(4, k, 8);
        let en = enumerate(4, k, false);
        let g_enum = en.z.ln() / 16.0;
        let diff = (res.g - g_enum).abs();
        assert!(
            diff < 1e-8,
            "K={k}: boundary g={:.12} vs enumeration g={:.12} (|diff|={diff:.2e})",
            res.g,
            g_enum,
        );
        let logz_diff = (res.log_z - en.z.ln()).abs();
        assert!(logz_diff < 1e-6, "K={k}: log Z diff {logz_diff:.2e}");
    }
}

#[test]
fn open_4x4_specific_heat_matches_variance() {
    let k = 0.44f64;
    let h = 0.0025f64;
    let cv_fd = k * k * (g_of(4, k + h, 8) - 2.0 * g_of(4, k, 8) + g_of(4, k - h, 8)) / (h * h);
    let en = enumerate(4, k, false);
    let cv_var = k * k * en.var_h / 16.0;
    let diff = (cv_fd - cv_var).abs();
    assert!(
        diff < 1e-3,
        "c_V finite difference {cv_fd:.6} vs K^2 Var(H)/16 {cv_var:.6} (|diff|={diff:.2e})"
    );
    println!("open 4x4 c_V(K=0.44) = {cv_fd:.6} (variance reference {cv_var:.6})");
}

#[test]
fn open_16_g_at_zero_temperature_limit() {
    // At K=0 every configuration has weight one: g = log 2 exactly.
    let res = contract_open_square(16, 0.0, 16);
    assert!(
        (res.g - 2.0f64.ln()).abs() < 1e-10,
        "g(0) = {} vs log 2",
        res.g
    );
}

#[test]
fn open_2x2_matches_enumeration() {
    for k in [0.3f64, 0.44] {
        let res = contract_open_square(2, k, 8);
        let en = enumerate(2, k, false);
        assert!(
            (res.log_z - en.z.ln()).abs() < 1e-10,
            "K={k}: log Z {} vs {}",
            res.log_z,
            en.z.ln()
        );
    }
}

#[test]
fn boundary_64_runs_and_records_scales() {
    // The stated case from the design file, checked for consistency between
    // g, log Z and the accumulated scale.
    let res = contract_open_square(8, 0.45, 16);
    assert_eq!(res.row_diag.len(), 8);
    let recomputed_logz = res.ell + res.closure_value.ln();
    assert!((recomputed_logz - res.log_z).abs() < 1e-12);
    assert!((res.g - res.log_z / 64.0).abs() < 1e-14);
}
