use std::path::PathBuf;

use infer::{
    conditioned_plan, execute_conditioned, log10_partition, normalize_open, parse_evidence, Model,
    OrderChoice,
};

fn asia() -> Model {
    Model::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("asia.uai")).unwrap()
}

#[test]
fn conditioning_keeps_the_scalar_asia_factor() {
    let conditioned = asia().condition(&parse_evidence("AD").unwrap()).unwrap();
    let scalars: Vec<_> = conditioned
        .factors
        .iter()
        .filter(|factor| factor.scope.is_empty())
        .collect();
    assert_eq!(scalars.len(), 1);
    assert!((scalars[0].values[0] - 0.01).abs() < 1e-15);
}

#[test]
fn log_contraction_matches_the_asia_evidence_mass() {
    let model = asia();
    let evidence = parse_evidence("AD").unwrap();
    let plan = conditioned_plan(&model, &evidence, OrderChoice::Greedy).unwrap();
    let partition = execute_conditioned(&model, &evidence, &plan, &[]).unwrap();
    let mass = 10f64.powf(log10_partition(&partition).unwrap());
    assert!((mass - 0.004501375).abs() < 1e-12);

    let open = execute_conditioned(&model, &evidence, &plan, &[1]).unwrap();
    let marginal = normalize_open(&open).unwrap();
    assert!((marginal[0] - 0.0877509649829219).abs() < 1e-12);
}

#[test]
fn multiplying_one_table_by_ten_adds_one_to_log10_partition() {
    let model = asia();
    let evidence = parse_evidence("AD").unwrap();
    let plan = conditioned_plan(&model, &evidence, OrderChoice::Greedy).unwrap();
    let original =
        log10_partition(&execute_conditioned(&model, &evidence, &plan, &[]).unwrap()).unwrap();
    let mut scaled = model.clone();
    for value in &mut scaled.factors[0].values {
        *value *= 10.0;
    }
    let shifted =
        log10_partition(&execute_conditioned(&scaled, &evidence, &plan, &[]).unwrap()).unwrap();
    assert!((shifted - original - 1.0).abs() < 1e-9);
    let marginal =
        normalize_open(&execute_conditioned(&scaled, &evidence, &plan, &[1]).unwrap()).unwrap();
    assert!((marginal[0] - 0.0877509649829219).abs() < 1e-12);
}

#[test]
fn rescaling_recovers_a_partition_that_underflows_f64() {
    let mut factors = String::new();
    for var in 0..2 {
        factors.push_str(&format!("1 {var}\n"));
    }
    let model = Model::parse(&format!(
        "MARKOV\n2\n2 2\n2\n{factors}2 1e-200 1e-200\n2 1e-200 1e-200\n"
    ))
    .unwrap();
    let plan = conditioned_plan(&model, &[], OrderChoice::Greedy).unwrap();
    let naive = model.factors[0].values[0] * model.factors[1].values[0] * 4.0;
    assert_eq!(naive, 0.0);
    let log10 = log10_partition(&execute_conditioned(&model, &[], &plan, &[]).unwrap()).unwrap();
    let expected = 4.0_f64.log10() - 400.0;
    assert!((log10 - expected).abs() < 1e-9);
}
