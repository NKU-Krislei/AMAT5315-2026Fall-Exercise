use std::collections::HashMap;
use std::path::Path;

/// One local factor. `scope` is the UAI axis order; the last variable changes fastest.
#[derive(Debug, Clone, PartialEq)]
pub struct Factor {
    pub scope: Vec<usize>,
    pub dims: Vec<usize>,
    pub values: Vec<f64>,
}

impl Factor {
    pub fn probability(&self, states: &[usize]) -> f64 {
        assert_eq!(states.len(), self.scope.len());
        let mut offset = 0;
        for (axis, &state) in states.iter().enumerate() {
            offset = offset * self.dims[axis] + state;
        }
        self.values[offset]
    }
}

/// A MARKOV UAI model: a product of factor tables.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub dims: Vec<usize>,
    pub factors: Vec<Factor>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut cursor = 0;
        let mut next = |what: &str| -> Result<&str, String> {
            let token = tokens
                .get(cursor)
                .copied()
                .ok_or_else(|| format!("unexpected end of UAI file while reading {what}"))?;
            cursor += 1;
            Ok(token)
        };

        let header = next("header")?;
        if header != "MARKOV" {
            return Err(format!("expected MARKOV header, found {header}"));
        }
        let nvars: usize = next("variable count")?
            .parse()
            .map_err(|err| format!("variable count: {err}"))?;
        let mut dims = Vec::with_capacity(nvars);
        for _ in 0..nvars {
            dims.push(
                next("dimension")?
                    .parse()
                    .map_err(|err| format!("dimension: {err}"))?,
            );
        }
        let nfactors: usize = next("factor count")?
            .parse()
            .map_err(|err| format!("factor count: {err}"))?;
        let mut scopes = Vec::with_capacity(nfactors);
        for _ in 0..nfactors {
            let arity: usize = next("scope length")?
                .parse()
                .map_err(|err| format!("scope length: {err}"))?;
            let mut scope = Vec::with_capacity(arity);
            for _ in 0..arity {
                let var: usize = next("scope variable")?
                    .parse()
                    .map_err(|err| format!("scope variable: {err}"))?;
                if var >= nvars {
                    return Err(format!("variable {var} is outside 0..{nvars}"));
                }
                scope.push(var);
            }
            scopes.push(scope);
        }

        let mut factors = Vec::with_capacity(nfactors);
        for scope in scopes {
            let count: usize = next("table length")?
                .parse()
                .map_err(|err| format!("table length: {err}"))?;
            let factor_dims: Vec<usize> = scope.iter().map(|&var| dims[var]).collect();
            let expected: usize = factor_dims.iter().product();
            if count != expected {
                return Err(format!(
                    "table length {count} does not match scope {scope:?} size {expected}"
                ));
            }
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(
                    next("table entry")?
                        .parse()
                        .map_err(|err| format!("table entry: {err}"))?,
                );
            }
            factors.push(Factor {
                scope,
                dims: factor_dims,
                values,
            });
        }
        if cursor != tokens.len() {
            return Err(format!(
                "UAI file has {} unused tokens",
                tokens.len() - cursor
            ));
        }
        Ok(Self { dims, factors })
    }

    pub fn factor_scope(&self, scope: &[usize]) -> &Factor {
        self.factors
            .iter()
            .find(|factor| factor.scope == scope)
            .unwrap_or_else(|| panic!("missing factor with scope {scope:?}"))
    }

    /// Fix observed variables and keep every resulting table, including scalars.
    pub fn condition(&self, evidence: &[(usize, usize)]) -> Result<Self, String> {
        let mut observed = HashMap::new();
        for &(var, state) in evidence {
            let dim = *self
                .dims
                .get(var)
                .ok_or_else(|| format!("evidence variable {var} is outside the model"))?;
            if state >= dim {
                return Err(format!("evidence state {state} is outside variable {var}"));
            }
            if let Some(previous) = observed.insert(var, state) {
                if previous != state {
                    return Err(format!("variable {var} is fixed to two states"));
                }
            }
        }
        let factors = self
            .factors
            .iter()
            .map(|factor| factor.condition(&observed))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            dims: self.dims.clone(),
            factors,
        })
    }
}

impl Factor {
    fn condition(&self, observed: &HashMap<usize, usize>) -> Result<Self, String> {
        let mut scope = Vec::new();
        let mut dims = Vec::new();
        let mut fixed = Vec::new();
        for (axis, &var) in self.scope.iter().enumerate() {
            if let Some(state) = observed.get(&var).copied() {
                fixed.push((axis, state));
            } else {
                scope.push(var);
                dims.push(self.dims[axis]);
            }
        }
        let count = dims.iter().copied().product();
        let mut values = vec![0.0; count];
        for index in 0..count {
            let mut states = vec![0; self.scope.len()];
            let mut rest = index;
            for axis in (0..scope.len()).rev() {
                let original = self
                    .scope
                    .iter()
                    .position(|&var| var == scope[axis])
                    .ok_or_else(|| format!("variable {} left its factor scope", scope[axis]))?;
                states[original] = rest % dims[axis];
                rest /= dims[axis];
            }
            for &(axis, state) in &fixed {
                states[axis] = state;
            }
            values[index] = self.probability(&states);
        }
        Ok(Self {
            scope,
            dims,
            values,
        })
    }
}

/// Read one UAI evidence instance: a count followed by variable/state pairs.
pub fn read_evidence_file(path: &Path) -> Result<Vec<(usize, usize)>, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let count: usize = tokens
        .first()
        .ok_or("evidence file is empty")?
        .parse()
        .map_err(|err| format!("evidence count: {err}"))?;
    if tokens.len() != 1 + 2 * count {
        return Err(format!(
            "evidence file has {} tokens after the count; expected {}",
            tokens.len().saturating_sub(1),
            2 * count
        ));
    }
    let mut evidence = Vec::with_capacity(count);
    for pair in tokens[1..].chunks(2) {
        let var = pair[0]
            .parse()
            .map_err(|err| format!("evidence variable: {err}"))?;
        let state = pair[1]
            .parse()
            .map_err(|err| format!("evidence state: {err}"))?;
        evidence.push((var, state));
    }
    Ok(evidence)
}

pub fn read_log10_partition(path: &Path) -> Result<f64, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() != 2 || tokens[0] != "PR" {
        return Err("expected a PR file with one log10 partition value".to_string());
    }
    tokens[1]
        .parse()
        .map_err(|err| format!("published log10 partition: {err}"))
}

pub fn read_marginals(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.first().copied() != Some("MAR") {
        return Err("expected a MAR file".to_string());
    }
    let mut cursor = 1;
    let nvars: usize = tokens
        .get(cursor)
        .ok_or("MAR file is missing the variable count")?
        .parse()
        .map_err(|err| format!("MAR variable count: {err}"))?;
    cursor += 1;
    let mut marginals = Vec::with_capacity(nvars);
    for _ in 0..nvars {
        let dim: usize = tokens
            .get(cursor)
            .ok_or("MAR file ended inside a marginal")?
            .parse()
            .map_err(|err| format!("MAR dimension: {err}"))?;
        cursor += 1;
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            let value = tokens
                .get(cursor)
                .ok_or("MAR file ended inside a marginal")?
                .parse()
                .map_err(|err| format!("MAR probability: {err}"))?;
            cursor += 1;
            row.push(value);
        }
        marginals.push(row);
    }
    if cursor != tokens.len() {
        return Err(format!(
            "MAR file has {} unused tokens",
            tokens.len() - cursor
        ));
    }
    Ok(marginals)
}
