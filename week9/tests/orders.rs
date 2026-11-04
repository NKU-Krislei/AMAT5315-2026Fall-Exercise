use std::collections::HashMap;
use std::path::PathBuf;

use infer::{recount, PlanTree};

fn sizes_ten() -> HashMap<usize, usize> {
    (0..4).map(|label| (label, 10)).collect()
}

#[test]
fn left_associated_matrix_chain_costs_two_thousand() {
    let inputs = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
    let output = vec![0, 3];
    let tree = PlanTree::join(
        PlanTree::join(PlanTree::leaf(0), PlanTree::leaf(1)),
        PlanTree::leaf(2),
    );
    let cost = recount(&inputs, &output, &tree, &sizes_ten()).expect("cost");
    assert_eq!(cost.work, 2000);
    assert_eq!(cost.largest_tensor, 100);
    assert_eq!(cost.steps.len(), 2);
    assert_eq!(cost.steps.iter().map(|step| step.cost).sum::<u64>(), 2000);
}

#[test]
fn contracting_the_disconnected_pair_first_costs_twenty_thousand() {
    let inputs = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
    let output = vec![0, 3];
    let tree = PlanTree::join(
        PlanTree::join(PlanTree::leaf(0), PlanTree::leaf(2)),
        PlanTree::leaf(1),
    );
    let cost = recount(&inputs, &output, &tree, &sizes_ten()).expect("cost");
    assert_eq!(cost.work, 20000);
    assert_eq!(cost.largest_tensor, 10000);
}

#[test]
fn asia_plans_agree_with_enumeration_and_replay() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("asia.uai");
    let model = infer::Model::load(&path).expect("asia");
    let evidence = [(0, 0), (7, 0)];
    let exact = infer::enumerate(&model, &evidence).evidence_mass;
    let greedy = infer::scalar_plan(&model, &evidence, infer::OrderChoice::Greedy).expect("greedy");
    let random = infer::scalar_plan(&model, &evidence, infer::OrderChoice::Random { seed: 42 })
        .expect("random");
    let again = infer::scalar_plan(&model, &evidence, infer::OrderChoice::Random { seed: 42 })
        .expect("repeat");
    assert_eq!(random.tree, again.tree);
    let greedy_mass = infer::execute_scalar(&model, &evidence, &greedy).expect("execute greedy");
    let random_mass = infer::execute_scalar(&model, &evidence, &random).expect("execute random");
    assert!((greedy_mass - exact).abs() / exact < 1e-10);
    assert!((random_mass - exact).abs() / exact < 1e-10);
    let greedy_cost = infer::recount_plan(&model, &greedy).expect("recount");
    let random_cost = infer::recount_plan(&model, &random).expect("recount");
    assert_eq!(
        greedy_cost.work,
        greedy_cost.steps.iter().map(|step| step.cost).sum::<u64>()
    );
    assert!(greedy_cost.largest_tensor >= 8);
    assert!(random_cost.work > greedy_cost.work);
}
