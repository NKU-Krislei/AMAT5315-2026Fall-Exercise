use std::path::PathBuf;

use infer::{belief_propagation, enumerate, BpSettings, Model};

fn chain() -> Model {
    Model::parse(
        "MARKOV
3
2 2 2
3
1 0
2 1 0
2 2 1
2 0.2 0.8
4 0.9 0.1 0.3 0.7
4 0.6 0.4 0.2 0.8
",
    )
    .expect("chain")
}

fn settings() -> BpSettings {
    BpSettings {
        damping: 0.5,
        tolerance: 1e-10,
        max_sweeps: 1000,
    }
}

fn max_error(exact: &[Vec<f64>], beliefs: &[Vec<f64>]) -> f64 {
    exact
        .iter()
        .zip(beliefs)
        .flat_map(|(left, right)| left.iter().zip(right))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

#[test]
fn belief_propagation_is_exact_on_a_chain() {
    let model = chain();
    let evidence = [(0, 0)];
    let exact = enumerate(&model, &evidence);
    let result = belief_propagation(&model, &evidence, settings()).expect("bp");
    assert!(result.converged);
    let error = max_error(&exact.marginals, &result.beliefs);
    assert!(error < 1e-8, "chain marginal error {error:.3e}");
    assert!((result.beliefs[0][0] - 1.0).abs() < 1e-12);
}

#[test]
fn loopy_asia_can_converge_to_the_wrong_marginal() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("asia.uai");
    let model = Model::load(&path).expect("asia");
    let evidence = [(0, 0), (7, 0), (6, 0)];
    let exact = enumerate(&model, &evidence);
    let result = belief_propagation(&model, &evidence, settings()).expect("bp");
    assert!(result.converged, "residual {}", result.residual);
    assert!(result.residual < 1e-10);
    let error = max_error(&exact.marginals, &result.beliefs);
    assert!(
        (error - 0.01061).abs() < 5e-4,
        "marginal error {error:.6}, sweeps {}",
        result.sweeps.len()
    );
    assert!(result.cumulative_work > 0);
    for belief in &result.beliefs {
        let sum: f64 = belief.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}
