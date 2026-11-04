use crate::Model;

#[derive(Debug, Clone, Copy)]
pub struct BpSettings {
    pub damping: f64,
    pub tolerance: f64,
    pub max_sweeps: usize,
}

#[derive(Debug, Clone)]
pub struct BpSweep {
    pub residual: f64,
    pub cumulative_work: u64,
    pub beliefs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct BpResult {
    pub converged: bool,
    pub residual: f64,
    pub cumulative_work: u64,
    pub beliefs: Vec<Vec<f64>>,
    pub sweeps: Vec<BpSweep>,
}

struct Reduced {
    scope: Vec<usize>,
    dims: Vec<usize>,
    values: Vec<f64>,
}

struct Edge {
    var: usize,
    factor: usize,
}

pub fn belief_propagation(
    model: &Model,
    evidence: &[(usize, usize)],
    settings: BpSettings,
) -> Result<BpResult, String> {
    if !(0.0..=1.0).contains(&settings.damping) {
        return Err("damping must be between 0 and 1".to_string());
    }
    if settings.tolerance <= 0.0 {
        return Err("tolerance must be positive".to_string());
    }
    if settings.max_sweeps == 0 {
        return Err("max sweeps must be positive".to_string());
    }
    let observed = observation_map(model, evidence)?;
    let factors = reduced_factors(model, &observed);
    let (edges, var_edges) = edges_of(model.dims.len(), &factors);
    let mut var_to_factor = initial_messages(model, &edges);
    let mut factor_to_var = var_to_factor.clone();
    let mut sweeps = Vec::new();
    let mut cumulative_work = 0u64;

    for _ in 1..=settings.max_sweeps {
        let mut hat_var_to_factor = Vec::with_capacity(edges.len());
        let mut hat_factor_to_var = Vec::with_capacity(edges.len());
        for edge_index in 0..edges.len() {
            let (message, work) =
                variable_to_factor(edge_index, model, &edges, &var_edges, &factor_to_var)?;
            cumulative_work += work;
            hat_var_to_factor.push(message);
        }
        for edge_index in 0..edges.len() {
            let (message, work) =
                factor_to_variable(edge_index, &factors, &edges, &var_edges, &var_to_factor)?;
            cumulative_work += work;
            hat_factor_to_var.push(message);
        }
        let residual = max_difference(&hat_var_to_factor, &var_to_factor)
            .max(max_difference(&hat_factor_to_var, &factor_to_var));
        let beliefs = beliefs_of(
            model,
            &observed,
            &factors,
            &edges,
            &var_edges,
            &hat_factor_to_var,
        )?;
        sweeps.push(BpSweep {
            residual,
            cumulative_work,
            beliefs,
        });
        if residual < settings.tolerance || sweeps.len() == settings.max_sweeps {
            break;
        }
        damp(&mut var_to_factor, &hat_var_to_factor, settings.damping);
        damp(&mut factor_to_var, &hat_factor_to_var, settings.damping);
    }

    let last = sweeps.last().expect("max sweeps is positive");
    Ok(BpResult {
        converged: last.residual < settings.tolerance,
        residual: last.residual,
        cumulative_work: last.cumulative_work,
        beliefs: last.beliefs.clone(),
        sweeps,
    })
}

fn observation_map(
    model: &Model,
    evidence: &[(usize, usize)],
) -> Result<Vec<Option<usize>>, String> {
    let mut observed = vec![None; model.dims.len()];
    for &(var, state) in evidence {
        let dim = *model
            .dims
            .get(var)
            .ok_or_else(|| format!("evidence variable {var} is outside the model"))?;
        if state >= dim {
            return Err(format!("evidence state {state} is outside variable {var}"));
        }
        observed[var] = Some(state);
    }
    Ok(observed)
}

fn reduced_factors(model: &Model, observed: &[Option<usize>]) -> Vec<Reduced> {
    let mut reduced = Vec::new();
    for factor in &model.factors {
        let mut scope = Vec::new();
        let mut dims = Vec::new();
        let mut fixed = Vec::new();
        for (axis, &var) in factor.scope.iter().enumerate() {
            if let Some(state) = observed[var] {
                fixed.push((axis, state));
            } else {
                scope.push(var);
                dims.push(factor.dims[axis]);
            }
        }
        if scope.is_empty() {
            continue;
        }
        let count: usize = dims.iter().copied().product();
        let mut values = vec![0.0; count];
        for index in 0..count {
            let mut states = vec![0; factor.scope.len()];
            let mut rest = index;
            for axis in (0..scope.len()).rev() {
                let original = factor
                    .scope
                    .iter()
                    .position(|&var| var == scope[axis])
                    .unwrap();
                states[original] = rest % dims[axis];
                rest /= dims[axis];
            }
            for &(axis, state) in &fixed {
                states[axis] = state;
            }
            let mut offset = 0;
            for (axis, &state) in states.iter().enumerate() {
                offset = offset * factor.dims[axis] + state;
            }
            values[index] = factor.values[offset];
        }
        reduced.push(Reduced {
            scope,
            dims,
            values,
        });
    }
    reduced
}

fn edges_of(nvars: usize, factors: &[Reduced]) -> (Vec<Edge>, Vec<Vec<usize>>) {
    let mut edges = Vec::new();
    let mut var_edges = vec![Vec::new(); nvars];
    for (factor, reduced) in factors.iter().enumerate() {
        for &var in &reduced.scope {
            var_edges[var].push(edges.len());
            edges.push(Edge { var, factor });
        }
    }
    (edges, var_edges)
}

fn initial_messages(model: &Model, edges: &[Edge]) -> Vec<Vec<f64>> {
    edges
        .iter()
        .map(|edge| {
            let dim = model.dims[edge.var];
            vec![1.0 / dim as f64; dim]
        })
        .collect()
}

fn variable_to_factor(
    edge_index: usize,
    model: &Model,
    edges: &[Edge],
    var_edges: &[Vec<usize>],
    factor_to_var: &[Vec<f64>],
) -> Result<(Vec<f64>, u64), String> {
    let edge = &edges[edge_index];
    let dim = model.dims[edge.var];
    let mut message = vec![1.0; dim];
    let mut others = 0u64;
    for &incoming in &var_edges[edge.var] {
        if incoming == edge_index {
            continue;
        }
        others += 1;
        for state in 0..dim {
            message[state] *= factor_to_var[incoming][state];
        }
    }
    normalize(&mut message)?;
    Ok((message, dim as u64 * others))
}

fn factor_to_variable(
    edge_index: usize,
    factors: &[Reduced],
    edges: &[Edge],
    var_edges: &[Vec<usize>],
    var_to_factor: &[Vec<f64>],
) -> Result<(Vec<f64>, u64), String> {
    let edge = &edges[edge_index];
    let factor = &factors[edge.factor];
    let var_axis = factor
        .scope
        .iter()
        .position(|&var| var == edge.var)
        .expect("edge variable is in its factor");
    let mut message = vec![0.0; factor.dims[var_axis]];
    for index in 0..factor.values.len() {
        let states = decode(index, factor);
        let mut weight = factor.values[index];
        for (axis, &var) in factor.scope.iter().enumerate() {
            if axis == var_axis {
                continue;
            }
            let incoming = edge_between(var, edge.factor, edges, var_edges);
            weight *= var_to_factor[incoming][states[axis]];
        }
        message[states[var_axis]] += weight;
    }
    normalize(&mut message)?;
    Ok((message, factor.values.len() as u64))
}

fn beliefs_of(
    model: &Model,
    observed: &[Option<usize>],
    factors: &[Reduced],
    edges: &[Edge],
    var_edges: &[Vec<usize>],
    factor_to_var: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let mut beliefs = Vec::with_capacity(model.dims.len());
    for (var, &dim) in model.dims.iter().enumerate() {
        if let Some(state) = observed[var] {
            let mut belief = vec![0.0; dim];
            belief[state] = 1.0;
            beliefs.push(belief);
            continue;
        }
        if var_edges[var].is_empty() && factors.iter().all(|factor| !factor.scope.contains(&var)) {
            beliefs.push(vec![1.0 / dim as f64; dim]);
            continue;
        }
        let mut belief = vec![1.0; dim];
        for &edge_index in &var_edges[var] {
            let _ = edges;
            for state in 0..dim {
                belief[state] *= factor_to_var[edge_index][state];
            }
        }
        normalize(&mut belief)?;
        beliefs.push(belief);
    }
    Ok(beliefs)
}

fn decode(index: usize, factor: &Reduced) -> Vec<usize> {
    let mut rest = index;
    let mut states = vec![0; factor.scope.len()];
    for axis in (0..factor.scope.len()).rev() {
        states[axis] = rest % factor.dims[axis];
        rest /= factor.dims[axis];
    }
    states
}

fn edge_between(var: usize, factor: usize, edges: &[Edge], var_edges: &[Vec<usize>]) -> usize {
    var_edges[var]
        .iter()
        .copied()
        .find(|&edge| edges[edge].factor == factor)
        .expect("factor and variable share an edge")
}

fn normalize(message: &mut [f64]) -> Result<(), String> {
    let sum: f64 = message.iter().sum();
    if sum == 0.0 {
        return Err("message sum is zero".to_string());
    }
    for entry in message {
        *entry /= sum;
    }
    Ok(())
}

fn max_difference(updated: &[Vec<f64>], previous: &[Vec<f64>]) -> f64 {
    updated
        .iter()
        .zip(previous)
        .flat_map(|(left, right)| left.iter().zip(right))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn damp(messages: &mut [Vec<f64>], updated: &[Vec<f64>], damping: f64) {
    for (message, proposal) in messages.iter_mut().zip(updated) {
        for (entry, proposed) in message.iter_mut().zip(proposal) {
            *entry = (1.0 - damping) * *entry + damping * *proposed;
        }
    }
}
