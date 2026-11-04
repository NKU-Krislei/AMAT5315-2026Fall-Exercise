use std::path::PathBuf;

use infer::{ad_tuberculosis_diagram, contract, enumerate, matrix_and_ising, Model};

fn asia_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("asia.uai")
}

#[test]
fn asia_tuberculosis_table_is_last_axis_fastest() {
    let model = Model::load(&asia_path()).expect("asia.uai");
    assert_eq!(model.dims, vec![2, 2, 2, 2, 2, 2, 2, 2]);
    assert_eq!(model.factors.len(), 8);

    let prior_a = model.factor_scope(&[0]);
    assert_eq!(prior_a.probability(&[0]), 0.01);
    assert_eq!(prior_a.probability(&[1]), 0.99);

    let tuberculosis = model.factor_scope(&[1, 0]);
    assert_eq!(tuberculosis.scope, vec![1, 0]);
    assert_eq!(tuberculosis.probability(&[0, 0]), 0.05);
    assert_eq!(tuberculosis.probability(&[0, 1]), 0.01);
    assert_eq!(tuberculosis.probability(&[1, 0]), 0.95);
    assert_eq!(tuberculosis.probability(&[1, 1]), 0.99);
}

fn close(actual: f64, expected: f64) {
    let error = (actual - expected).abs();
    assert!(
        error < 1e-12,
        "actual={actual:.17} expected={expected:.17} error={error:.3e}"
    );
}

#[test]
fn enumeration_matches_published_asia_masses_and_joints() {
    let model = Model::load(&asia_path()).expect("asia.uai");

    let prior = enumerate(&model, &[]);
    close(prior.evidence_mass, 1.0);
    close(prior.marginals[1][0], 0.0104);
    close(prior.joint_tl[0][0], 0.000572);

    let ad = enumerate(&model, &[(0, 0), (7, 0)]);
    close(ad.evidence_mass, 0.004501375);
    close(ad.marginals[1][0], 0.000395 / 0.004501375);
    close(ad.marginals[0][0], 1.0);
    close(ad.marginals[7][1], 0.0);

    let adx = enumerate(&model, &[(0, 0), (7, 0), (6, 0)]);
    close(adx.evidence_mass, 0.00098822675);
    close(adx.joint_tl[0][0], 0.022213525387771584);
    close(adx.marginals[1][0], 0.39171172000757926);
    close(adx.marginals[3][0], 0.44427050775543164);
    close(adx.marginals[4][0], 0.62882177597398581);
    for marginal in &adx.marginals {
        close(marginal.iter().sum(), 1.0);
    }
    close(adx.joint_tl.iter().flatten().copied().sum(), 1.0);
}

#[test]
fn contraction_matches_enumeration_on_every_patient() {
    let model = Model::load(&asia_path()).expect("asia.uai");
    let cases = [
        Vec::new(),
        vec![(0, 0), (7, 0)],
        vec![(0, 0), (7, 0), (6, 0)],
    ];
    for evidence in cases {
        let exact = enumerate(&model, &evidence);
        let approx = contract(&model, &evidence);
        let mass_error = (approx.evidence_mass - exact.evidence_mass).abs() / exact.evidence_mass;
        assert!(
            mass_error < 1e-10,
            "evidence {evidence:?} relative mass error {mass_error:.3e}"
        );
        for (left, right) in exact.marginals.iter().zip(&approx.marginals) {
            for (a, b) in left.iter().zip(right) {
                let error = (a - b).abs();
                assert!(error < 1e-10, "marginal error {error:.3e}");
            }
        }
        for (left, right) in exact.joint_tl.iter().zip(&approx.joint_tl) {
            for (a, b) in left.iter().zip(right) {
                let error = (a - b).abs();
                assert!(error < 1e-10, "joint error {error:.3e}");
            }
        }
    }
}

#[test]
fn matrix_product_trace_and_ising_ring_match_the_answer_key() {
    let checks = matrix_and_ising();
    let expected_ab = [4.0, 10.0, 4.0, 8.0];
    for (actual, expected) in checks.ab_col_major.iter().zip(expected_ab) {
        close(*actual, expected);
    }
    close(checks.trace, 12.0);
    close(
        checks.ising_contraction,
        2.0 * 1.5_f64.exp() + 6.0 * (-0.5_f64).exp(),
    );
    close(checks.ising_enumeration, checks.ising_contraction);
    assert!(checks.ab_error < 1e-12);
    assert!(checks.trace_error < 1e-12);
    assert!(checks.ising_error < 1e-12);
}

#[test]
fn ad_diagram_fixes_observations_and_leaves_only_tuberculosis_open() {
    let model = Model::load(&asia_path()).expect("asia.uai");
    let svg = ad_tuberculosis_diagram(&model);
    for factor in &model.factors {
        let axes = factor
            .scope
            .iter()
            .map(|var| var.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert!(
            svg.contains(&format!("data-axes=\"{axes}\"")),
            "missing axes {axes}"
        );
    }
    assert_eq!(svg.matches("data-wire=\"S\"").count(), 4);
    assert!(svg.contains("data-open=\"T\""));
    assert!(svg.contains("data-fixed=\"A=0\""));
    assert!(svg.contains("data-fixed=\"D=0\""));
    assert!(svg.contains("data-sum=\"X\""));
    assert!(!svg.contains("data-open=\"L\""));
    assert!(!svg.contains("data-open=\"A\""));
}
