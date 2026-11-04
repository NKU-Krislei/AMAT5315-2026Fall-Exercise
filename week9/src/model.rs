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
}
