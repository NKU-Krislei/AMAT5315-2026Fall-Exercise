use std::collections::{HashMap, HashSet};

use omeco::{optimize_code, EinCode, GreedyMethod, NestedEinsum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
use tenferro_tensor::Tensor;

use crate::Model;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderChoice {
    Greedy,
    Random { seed: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlanTree {
    Leaf {
        tensor: usize,
    },
    Join {
        left: Box<PlanTree>,
        right: Box<PlanTree>,
    },
}

impl PlanTree {
    pub fn leaf(tensor: usize) -> Self {
        Self::Leaf { tensor }
    }

    pub fn join(left: Self, right: Self) -> Self {
        Self::Join {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarPlan {
    pub method: String,
    pub seed: Option<u64>,
    pub inputs: Vec<Vec<usize>>,
    pub output: Vec<usize>,
    pub tree: PlanTree,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractionStep {
    pub union: Vec<usize>,
    pub retained: Vec<usize>,
    pub cost: u64,
    pub entries: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Complexity {
    pub work: u64,
    pub largest_tensor: u64,
    pub largest_intermediate: ContractionStep,
    pub steps: Vec<ContractionStep>,
}

pub fn scalar_plan(
    model: &Model,
    evidence: &[(usize, usize)],
    choice: OrderChoice,
) -> Result<ScalarPlan, String> {
    let inputs = query_inputs(model, evidence);
    let output = Vec::new();
    let sizes = label_sizes(model, &inputs)?;
    let (method, seed, tree) = match choice {
        OrderChoice::Greedy => (
            "greedy".to_string(),
            None,
            greedy_tree(&inputs, &output, &sizes)?,
        ),
        OrderChoice::Random { seed } => (
            "random".to_string(),
            Some(seed),
            random_tree(&inputs, &output, seed),
        ),
    };
    Ok(ScalarPlan {
        method,
        seed,
        inputs,
        output,
        tree,
    })
}

pub fn recount(
    inputs: &[Vec<usize>],
    output: &[usize],
    tree: &PlanTree,
    sizes: &HashMap<usize, usize>,
) -> Result<Complexity, String> {
    let mut steps = Vec::new();
    let (_, largest_tensor) = walk(tree, inputs, output, sizes, &mut steps)?;
    let work = steps.iter().try_fold(0u64, |acc, step| {
        acc.checked_add(step.cost).ok_or("work overflow")
    })?;
    let largest_intermediate = steps
        .iter()
        .max_by_key(|step| step.entries)
        .cloned()
        .ok_or("contraction plan has no pair step")?;
    Ok(Complexity {
        work,
        largest_tensor,
        largest_intermediate,
        steps,
    })
}

pub fn recount_plan(model: &Model, plan: &ScalarPlan) -> Result<Complexity, String> {
    let sizes = label_sizes(model, &plan.inputs)?;
    recount(&plan.inputs, &plan.output, &plan.tree, &sizes)
}

pub fn execute_scalar(
    model: &Model,
    evidence: &[(usize, usize)],
    plan: &ScalarPlan,
) -> Result<f64, String> {
    let inputs = query_inputs(model, evidence);
    if inputs != plan.inputs || plan.output != Vec::<usize>::new() {
        return Err("saved plan does not match this model and evidence".to_string());
    }
    let tensors = input_tensors(model, evidence)?;
    let mut engine = CpuEngine::serial();
    let result = execute_node(
        &plan.tree,
        &tensors,
        &plan.inputs,
        &plan.output,
        &mut engine,
    )?;
    let values = result.as_slice::<f64>().map_err(|err| err.to_string())?;
    if values.len() != 1 {
        return Err(format!(
            "scalar contraction produced {} entries",
            values.len()
        ));
    }
    Ok(values[0])
}

pub fn query_inputs(model: &Model, evidence: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut inputs: Vec<Vec<usize>> = model
        .factors
        .iter()
        .map(|factor| factor.scope.clone())
        .collect();
    for &(var, _) in evidence {
        inputs.push(vec![var]);
    }
    inputs
}

fn label_sizes(model: &Model, inputs: &[Vec<usize>]) -> Result<HashMap<usize, usize>, String> {
    let mut sizes = HashMap::new();
    for labels in inputs {
        for &label in labels {
            let dim = *model
                .dims
                .get(label)
                .ok_or_else(|| format!("label {label} is not a model variable"))?;
            sizes.insert(label, dim);
        }
    }
    Ok(sizes)
}

fn greedy_tree(
    inputs: &[Vec<usize>],
    output: &[usize],
    sizes: &HashMap<usize, usize>,
) -> Result<PlanTree, String> {
    let code = EinCode::new(inputs.to_vec(), output.to_vec());
    let nested = optimize_code(&code, sizes, &GreedyMethod::default())
        .ok_or("greedy search did not return a contraction tree")?;
    Ok(from_nested(&nested))
}

fn from_nested(tree: &NestedEinsum<usize>) -> PlanTree {
    match tree {
        NestedEinsum::Leaf { tensor_index } => PlanTree::leaf(*tensor_index),
        NestedEinsum::Node { args, .. } => {
            let mut args = args.iter();
            let mut acc = from_nested(args.next().expect("contraction node has a child"));
            for arg in args {
                acc = PlanTree::join(acc, from_nested(arg));
            }
            acc
        }
    }
}

fn random_tree(inputs: &[Vec<usize>], _output: &[usize], seed: u64) -> PlanTree {
    let mut live: Vec<PlanTree> = (0..inputs.len()).map(PlanTree::leaf).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    while live.len() > 1 {
        let left_index = rng.random_range(0..live.len());
        let mut right_index = rng.random_range(0..live.len() - 1);
        if right_index >= left_index {
            right_index += 1;
        }
        let (first, second) = if left_index > right_index {
            (left_index, right_index)
        } else {
            (right_index, left_index)
        };
        let first_tree = live.remove(first);
        let second_tree = live.remove(second);
        live.push(PlanTree::join(first_tree, second_tree));
    }
    live.pop().expect("random contraction keeps one tree")
}

fn walk(
    tree: &PlanTree,
    inputs: &[Vec<usize>],
    output: &[usize],
    sizes: &HashMap<usize, usize>,
    steps: &mut Vec<ContractionStep>,
) -> Result<(Vec<usize>, u64), String> {
    match tree {
        PlanTree::Leaf { tensor } => {
            let labels = inputs[*tensor].clone();
            let entries = product(&labels, sizes)?;
            Ok((labels, entries))
        }
        PlanTree::Join { left, right } => {
            let (left_labels, left_largest) = walk(left, inputs, output, sizes, steps)?;
            let (right_labels, right_largest) = walk(right, inputs, output, sizes, steps)?;
            let union = union_labels(&left_labels, &right_labels);
            let retained = result_labels(tree, inputs, output);
            let cost = product(&union, sizes)?;
            let entries = product(&retained, sizes)?;
            steps.push(ContractionStep {
                union,
                retained: retained.clone(),
                cost,
                entries,
            });
            Ok((retained, left_largest.max(right_largest).max(entries)))
        }
    }
}

fn result_labels(tree: &PlanTree, inputs: &[Vec<usize>], output: &[usize]) -> Vec<usize> {
    match tree {
        PlanTree::Leaf { tensor } => inputs[*tensor].clone(),
        PlanTree::Join { left, right } => {
            let inside = leaf_indices(tree);
            let mut outside = HashSet::new();
            for (index, labels) in inputs.iter().enumerate() {
                if !inside.contains(&index) {
                    outside.extend(labels.iter().copied());
                }
            }
            let mut seen = HashSet::new();
            let mut retained = Vec::new();
            for label in result_labels(left, inputs, output)
                .into_iter()
                .chain(result_labels(right, inputs, output))
            {
                if seen.insert(label) && (output.contains(&label) || outside.contains(&label)) {
                    retained.push(label);
                }
            }
            retained
        }
    }
}

fn leaf_indices(tree: &PlanTree) -> HashSet<usize> {
    let mut indices = HashSet::new();
    collect_leaves(tree, &mut indices);
    indices
}

fn collect_leaves(tree: &PlanTree, indices: &mut HashSet<usize>) {
    match tree {
        PlanTree::Leaf { tensor } => {
            indices.insert(*tensor);
        }
        PlanTree::Join { left, right } => {
            collect_leaves(left, indices);
            collect_leaves(right, indices);
        }
    }
}

fn union_labels(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut union = Vec::new();
    for label in left.iter().chain(right) {
        if seen.insert(*label) {
            union.push(*label);
        }
    }
    union
}

fn product(labels: &[usize], sizes: &HashMap<usize, usize>) -> Result<u64, String> {
    let mut acc = 1u64;
    for label in labels {
        let dim = *sizes
            .get(label)
            .ok_or_else(|| format!("missing size for label {label}"))? as u64;
        acc = acc
            .checked_mul(dim)
            .ok_or_else(|| format!("dimension product overflow at label {label}"))?;
    }
    Ok(acc)
}

fn input_tensors(model: &Model, evidence: &[(usize, usize)]) -> Result<Vec<Tensor>, String> {
    let mut tensors = Vec::new();
    for factor in &model.factors {
        tensors.push(
            Tensor::from_vec_row_major(factor.dims.clone(), factor.values.clone())
                .map_err(|err| err.to_string())?,
        );
    }
    for &(var, state) in evidence {
        let dim = *model
            .dims
            .get(var)
            .ok_or_else(|| format!("evidence variable {var} is outside the model"))?;
        if state >= dim {
            return Err(format!("evidence state {state} is outside variable {var}"));
        }
        let mut one_hot = vec![0.0; dim];
        one_hot[state] = 1.0;
        tensors
            .push(Tensor::from_vec_row_major(vec![dim], one_hot).map_err(|err| err.to_string())?);
    }
    Ok(tensors)
}

fn execute_node(
    tree: &PlanTree,
    tensors: &[Tensor],
    inputs: &[Vec<usize>],
    output: &[usize],
    engine: &mut CpuEngine,
) -> Result<Tensor, String> {
    match tree {
        PlanTree::Leaf { tensor } => tensors[*tensor].duplicate().map_err(|err| err.to_string()),
        PlanTree::Join { left, right } => {
            let left_tensor = execute_node(left, tensors, inputs, output, engine)?;
            let right_tensor = execute_node(right, tensors, inputs, output, engine)?;
            let left_labels = labels_u32(&result_labels(left, inputs, output));
            let right_labels = labels_u32(&result_labels(right, inputs, output));
            let out_labels = labels_u32(&result_labels(tree, inputs, output));
            let subscripts = EinsumSubscripts::new(&[&left_labels, &right_labels], &out_labels);
            [&left_tensor, &right_tensor]
                .einsum_subscripts(&subscripts, engine)
                .map_err(|err| err.to_string())
        }
    }
}

fn labels_u32(labels: &[usize]) -> Vec<u32> {
    labels.iter().map(|&label| label as u32).collect()
}
