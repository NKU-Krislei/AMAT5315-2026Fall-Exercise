use crate::model::Model;

/// Normalized node marginals and the ASIA \(T,L\) joint, plus the evidence mass.
///
/// `joint_tl[t][l]` is the joint of variables 1 and 3. Those are tuberculosis
/// and lung cancer in ASIA.
#[derive(Debug, Clone, PartialEq)]
pub struct Posterior {
    pub evidence_mass: f64,
    pub marginals: Vec<Vec<f64>>,
    pub joint_tl: Vec<Vec<f64>>,
}

pub fn enumerate(model: &Model, evidence: &[(usize, usize)]) -> Posterior {
    let n = model.dims.len();
    let mut fixed = vec![None; n];
    for &(var, state) in evidence {
        fixed[var] = Some(state);
    }

    let mut assignment = vec![0usize; n];
    let mut mass = 0.0;
    let mut marginals: Vec<Vec<f64>> = model.dims.iter().map(|&dim| vec![0.0; dim]).collect();
    let track_joint = n > 3;
    let mut joint_tl = if track_joint {
        vec![vec![0.0; model.dims[3]]; model.dims[1]]
    } else {
        Vec::new()
    };

    loop {
        let agrees = assignment
            .iter()
            .enumerate()
            .all(|(var, state)| fixed[var].is_none_or(|observed| observed == *state));
        if agrees {
            let weight = joint_weight(model, &assignment);
            mass += weight;
            for (var, &state) in assignment.iter().enumerate() {
                marginals[var][state] += weight;
            }
            if track_joint {
                joint_tl[assignment[1]][assignment[3]] += weight;
            }
        }
        if !advance(&mut assignment, &model.dims) {
            break;
        }
    }

    for marginal in &mut marginals {
        for entry in marginal.iter_mut() {
            *entry /= mass;
        }
    }
    for row in &mut joint_tl {
        for entry in row.iter_mut() {
            *entry /= mass;
        }
    }

    Posterior {
        evidence_mass: mass,
        marginals,
        joint_tl,
    }
}

fn joint_weight(model: &Model, assignment: &[usize]) -> f64 {
    let mut weight = 1.0;
    for factor in &model.factors {
        let mut offset = 0;
        for (axis, &var) in factor.scope.iter().enumerate() {
            offset = offset * factor.dims[axis] + assignment[var];
        }
        weight *= factor.values[offset];
    }
    weight
}

pub fn contract(model: &Model, evidence: &[(usize, usize)]) -> Posterior {
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let mass = contract_values(model, evidence, &[], &mut engine)[0];
    let mut marginals = Vec::with_capacity(model.dims.len());
    for var in 0..model.dims.len() {
        if let Some((_, state)) = evidence.iter().find(|(observed, _)| *observed == var) {
            let mut marginal = vec![0.0; model.dims[var]];
            marginal[*state] = 1.0;
            marginals.push(marginal);
        } else {
            let raw = contract_values(model, evidence, &[var], &mut engine);
            marginals.push(raw.into_iter().map(|value| value / mass).collect());
        }
    }
    let raw_joint = contract_values(model, evidence, &[1, 3], &mut engine);
    let rows = model.dims[1];
    let cols = model.dims[3];
    let mut joint_tl = vec![vec![0.0; cols]; rows];
    for col in 0..cols {
        for row in 0..rows {
            joint_tl[row][col] = raw_joint[row + rows * col] / mass;
        }
    }
    Posterior {
        evidence_mass: mass,
        marginals,
        joint_tl,
    }
}

fn contract_values(
    model: &Model,
    evidence: &[(usize, usize)],
    open: &[usize],
    engine: &mut tenferro_cpu::CpuEngine,
) -> Vec<f64> {
    use tenferro_einsum::{EinsumSubscripts, TensorEinsumExt};
    use tenferro_tensor::Tensor;

    let mut tensors = Vec::new();
    let mut labels = Vec::new();
    for factor in &model.factors {
        let tensor = Tensor::from_vec_row_major(factor.dims.clone(), factor.values.clone())
            .expect("factor table fits its shape");
        tensors.push(tensor);
        labels.push(
            factor
                .scope
                .iter()
                .map(|&var| var as u32)
                .collect::<Vec<_>>(),
        );
    }
    let mut observed = vec![None; model.dims.len()];
    for &(var, state) in evidence {
        observed[var] = Some(state);
    }
    for (var, state) in observed.into_iter().enumerate() {
        let Some(state) = state else {
            continue;
        };
        let mut one_hot = vec![0.0; model.dims[var]];
        one_hot[state] = 1.0;
        let tensor = Tensor::from_vec_row_major(vec![model.dims[var]], one_hot)
            .expect("observation cap fits its variable");
        tensors.push(tensor);
        labels.push(vec![var as u32]);
    }

    let label_refs: Vec<&[u32]> = labels.iter().map(Vec::as_slice).collect();
    let output: Vec<u32> = open.iter().map(|&var| var as u32).collect();
    let subscripts = EinsumSubscripts::new(&label_refs, &output);
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let result = refs
        .as_slice()
        .einsum_subscripts(&subscripts, engine)
        .expect("tensor contraction");
    result
        .as_slice::<f64>()
        .expect("contraction result is f64")
        .to_vec()
}

fn advance(assignment: &mut [usize], dims: &[usize]) -> bool {
    for var in (0..assignment.len()).rev() {
        assignment[var] += 1;
        if assignment[var] < dims[var] {
            return true;
        }
        assignment[var] = 0;
    }
    false
}
