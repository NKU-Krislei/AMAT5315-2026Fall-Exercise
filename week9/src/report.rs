use crate::{contract, enumerate, Model, Posterior};

const NAMES: [&str; 8] = ["A", "T", "S", "L", "B", "E", "X", "D"];
const DISEASES: [usize; 3] = [1, 3, 4];

pub fn parse_evidence(label: &str) -> Result<Vec<(usize, usize)>, String> {
    match label {
        "prior" => Ok(Vec::new()),
        "AD" => Ok(vec![(0, 0), (7, 0)]),
        "ADX" => Ok(vec![(0, 0), (7, 0), (6, 0)]),
        other => Err(format!(
            "unknown evidence {other}; expected prior, AD, or ADX"
        )),
    }
}

pub struct Comparison {
    pub label: String,
    pub observations: Vec<(usize, usize)>,
    pub enumeration: Posterior,
    pub contraction: Posterior,
}

pub fn compare(model: &Model, label: &str) -> Result<Comparison, String> {
    let observations = parse_evidence(label)?;
    Ok(Comparison {
        label: label.to_string(),
        enumeration: enumerate(model, &observations),
        contraction: contract(model, &observations),
        observations,
    })
}

pub fn max_marginal_error(comparison: &Comparison) -> f64 {
    comparison
        .enumeration
        .marginals
        .iter()
        .zip(&comparison.contraction.marginals)
        .flat_map(|(left, right)| left.iter().zip(right))
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

pub fn relative_mass_error(comparison: &Comparison) -> f64 {
    let exact = comparison.enumeration.evidence_mass;
    (comparison.contraction.evidence_mass - exact).abs() / exact.abs()
}

pub fn comparison_json(comparison: &Comparison) -> serde_json::Value {
    let names: Vec<&str> = (0..comparison.enumeration.marginals.len())
        .map(var_name)
        .collect();
    serde_json::json!({
        "evidence": comparison.label,
        "observations": comparison.observations.iter().map(|(id, state)| {
            serde_json::json!({
                "id": id,
                "name": var_name(*id),
                "state": state,
            })
        }).collect::<Vec<_>>(),
        "evidence_mass": {
            "enumeration": comparison.enumeration.evidence_mass,
            "contraction": comparison.contraction.evidence_mass,
        },
        "relative_evidence_mass_error": relative_mass_error(comparison),
        "max_absolute_marginal_error": max_marginal_error(comparison),
        "node_marginals": {
            "enumeration": named_marginals(&comparison.enumeration, &names),
            "contraction": named_marginals(&comparison.contraction, &names),
        },
        "joint_TL": {
            "variables": ["T", "L"],
            "enumeration": comparison.enumeration.joint_tl,
            "contraction": comparison.contraction.joint_tl,
            "product_of_marginals_enumeration": product_tl(&comparison.enumeration),
            "dependence_enumeration": dependence(&comparison.enumeration),
        },
    })
}

pub fn comparison_text(comparison: &Comparison) -> String {
    let mut lines = vec![
        format!("evidence {}", comparison.label),
        format!(
            "evidence probability enumeration={:.17} contraction={:.17} relative_error={:.3e}",
            comparison.enumeration.evidence_mass,
            comparison.contraction.evidence_mass,
            relative_mass_error(comparison)
        ),
    ];
    for var in DISEASES {
        let name = var_name(var);
        let exact = comparison.enumeration.marginals[var][0];
        let approx = comparison.contraction.marginals[var][0];
        lines.push(format!(
            "disease {name}=yes enumeration={:.17} contraction={:.17} absolute_error={:.3e}",
            exact,
            approx,
            (exact - approx).abs()
        ));
    }
    lines.push(format!(
        "max absolute marginal error={:.3e}",
        max_marginal_error(comparison)
    ));
    let dependence = dependence(&comparison.enumeration);
    lines.push(format!(
        "joint T=0,L=0 minus product = {:.17}",
        dependence[0][0]
    ));
    lines.join("\n") + "\n"
}

fn named_marginals(posterior: &Posterior, names: &[&str]) -> serde_json::Value {
    let mut object = serde_json::Map::new();
    for (name, marginal) in names.iter().zip(&posterior.marginals) {
        object.insert((*name).to_string(), serde_json::json!(marginal));
    }
    serde_json::Value::Object(object)
}

fn product_tl(posterior: &Posterior) -> Vec<Vec<f64>> {
    let mut product = vec![vec![0.0; posterior.marginals[3].len()]; posterior.marginals[1].len()];
    for (t, row) in product.iter_mut().enumerate() {
        for (l, entry) in row.iter_mut().enumerate() {
            *entry = posterior.marginals[1][t] * posterior.marginals[3][l];
        }
    }
    product
}

fn dependence(posterior: &Posterior) -> Vec<Vec<f64>> {
    product_tl(posterior)
        .into_iter()
        .zip(&posterior.joint_tl)
        .map(|(row, joint_row)| {
            row.into_iter()
                .zip(joint_row)
                .map(|(product, joint)| joint - product)
                .collect()
        })
        .collect()
}

fn var_name(id: usize) -> &'static str {
    NAMES.get(id).copied().unwrap_or("?")
}
