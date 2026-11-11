//! Part 3 verification checks from the learning sheet.
//!
//! - periodic spin enumeration at K=0.44: L=2, chi=16 within 1e-8;
//!   L=4, chi=8 within 2e-3 (one bond per site in each positive direction,
//!   including the distinct parallel bonds at L=2);
//! - at L=2, c_V with h=0.0025 against K^2 Var(H)/4 within 1e-3.

use thermo::ising::enumerate;
use thermo::trg::contract_periodic_square;

#[test]
fn trg_l2_matches_enumeration() {
    for k in [0.0f64, 0.44, 0.7] {
        let res = contract_periodic_square(2, k, 16);
        let en = enumerate(2, k, true);
        let g_enum = en.z.ln() / 4.0;
        let diff = (res.g - g_enum).abs();
        assert!(
            diff < 1e-8,
            "K={k}: TRG g={:.12} vs enumeration g={:.12} (|diff|={diff:.2e})",
            res.g,
            g_enum
        );
    }
}

#[test]
fn trg_l4_chi8_close_to_enumeration() {
    let k = 0.44f64;
    let res = contract_periodic_square(4, k, 8);
    let en = enumerate(4, k, true);
    let g_enum = en.z.ln() / 16.0;
    let diff = (res.g - g_enum).abs();
    assert!(
        diff < 2e-3,
        "L=4 chi=8: TRG g={:.6} vs enumeration g={:.6} (|diff|={diff:.2e})",
        res.g,
        g_enum
    );
    println!("trg L=4 chi=8 g(K=0.44) = {:.6} (enumeration {g_enum:.6})", res.g);
}

#[test]
fn trg_l2_specific_heat_matches_variance() {
    let k = 0.44f64;
    let h = 0.0025f64;
    let g = |kk: f64| contract_periodic_square(2, kk, 16).g;
    let cv_fd = k * k * (g(k + h) - 2.0 * g(k) + g(k - h)) / (h * h);
    let en = enumerate(2, k, true);
    let cv_var = k * k * en.var_h / 4.0;
    let diff = (cv_fd - cv_var).abs();
    assert!(
        diff < 1e-3,
        "c_V finite difference {cv_fd:.6} vs K^2 Var(H)/4 {cv_var:.6} (|diff|={diff:.2e})"
    );
    println!("trg L=2 c_V(K=0.44) = {cv_fd:.6} (variance reference {cv_var:.6})");
}

#[test]
fn trg_states_consistency() {
    // The stated million-spin case parameters, at a smaller size: the saved
    // pieces must reconstruct log Z.
    let res = contract_periodic_square(16, 0.44, 16);
    let recomputed = res.ell + res.closure_z.ln();
    assert!((recomputed - res.log_z).abs() < 1e-10);
    assert_eq!(res.steps.len(), 8);
    assert_eq!(res.steps[0].num_tensors, 256);
    assert_eq!(res.steps.last().unwrap().num_tensors, 2);
}

#[test]
fn trg_g_at_zero_temperature_limit() {
    let res = contract_periodic_square(8, 0.0, 16);
    assert!((res.g - 2.0f64.ln()).abs() < 1e-12);
}
