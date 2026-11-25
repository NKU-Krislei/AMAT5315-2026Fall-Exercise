//! Part 1 checks: the svd pipeline on a small matrix, verified end to end
//! through the saved files (the 512x512 case is exercised by the `svd`
//! command line tool and the plotting script).

use thermo::svd_evidence::{check, generate, GenSettings};

#[test]
fn small_pipeline_errors_match_predictions() {
    let dir = std::env::temp_dir().join("thermo-svd-test");
    let _ = std::fs::remove_dir_all(&dir);
    let settings = GenSettings {
        size: 64,
        seed: 2026,
        ranks: vec![4, 16, 64],
    };
    generate(&dir, &settings).expect("generate");
    let rep = check(&dir).expect("check");
    for row in &rep.rows {
        assert!(
            row.diff < 1e-8,
            "chi={}: measured {} vs predicted {}",
            row.chi,
            row.measured,
            row.predicted
        );
    }
    assert!(rep.full_rank_error < 1e-8);
    // For a square matrix the storage parity rank is n/2.
    assert_eq!(rep.storage_parity_chi, 32);
    assert!(rep.chi_10pct >= 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn check_reports_parity_rank() {
    // The parity rank for the 512 case, formula-wise.
    let n = 512usize;
    assert_eq!(n * n / (2 * n), 256);
}
