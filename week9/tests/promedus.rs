use std::path::PathBuf;

use serde_json::Value;

fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(name)
        .join("run.json");
    let text = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn promedus_orders_agree_with_each_other_and_the_published_answers() {
    let greedy = load("promedus-greedy");
    let treesa = load("promedus-treesa");
    let reference = -7.806746304582435;
    for run in [&greedy, &treesa] {
        let log10 = run["log10_partition"].as_f64().unwrap();
        assert!((log10 - reference).abs() < 1e-12);
        assert!(run["log10_partition_abs_error"].as_f64().unwrap() < 1e-4);
        for query in run["queries"].as_array().unwrap() {
            assert!(query["abs_error"].as_f64().unwrap() < 1e-5);
        }
    }
    let gap = (greedy["log10_partition"].as_f64().unwrap()
        - treesa["log10_partition"].as_f64().unwrap())
    .abs();
    assert!(gap < 1e-10);
    assert!(treesa["work"].as_u64().unwrap() < greedy["work"].as_u64().unwrap());
    assert_eq!(
        greedy["largest_intermediate_entries"].as_u64().unwrap(),
        4_194_304
    );
    assert_eq!(
        treesa["largest_intermediate_entries"].as_u64().unwrap(),
        131_072
    );
}

#[test]
fn promedus_belief_propagation_remains_unconverged_and_inaccurate() {
    let run = load("promedus-bp");
    assert_eq!(run["converged"], Value::Bool(false));
    assert!(run["residual"].as_f64().unwrap() > 1e-10);
    let error = run["max_query_error"].as_f64().unwrap();
    assert!((error - 0.382353).abs() < 1e-4);
    assert_eq!(run["history"].as_array().unwrap().len(), 1000);
}
