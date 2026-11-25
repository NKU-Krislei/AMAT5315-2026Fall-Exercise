//! Shared MLP used to imitate Strong Branching.
//!
//! The score is `w2^T tanh(W1^T φ + b1) + b2` with `W1` stored as 12 by 48.
//! Features are standardized with the training mean and standard deviation
//! before the linear layer.

use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Mlp {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: f64,
    pub mu: Vec<f64>,
    pub std: Vec<f64>,
    pub probes: Vec<Probe>,
}

#[derive(Clone, Debug)]
pub struct Probe {
    #[allow(dead_code)]
    pub instance: i64,
    pub features: Vec<Vec<f64>>,
    pub probabilities: Vec<f64>,
}

impl Mlp {
    pub fn load(path: &Path) -> Self {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let value: Value = serde_json::from_str(&text)
            .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()));
        let w1 = matrix(value.get("w1").expect("model is missing w1"));
        let b1 = vector(value.get("b1").expect("model is missing b1"));
        let w2 = vector(value.get("w2").expect("model is missing w2"));
        let b2 = value
            .get("b2")
            .and_then(|v| v.as_f64())
            .expect("model is missing b2");
        let mu = vector(value.get("mu").expect("model is missing mu"));
        let std = vector(value.get("std").expect("model is missing std"));
        assert_eq!(w1.len(), 12, "W1 must have 12 rows");
        assert!(w1.iter().all(|row| row.len() == 48), "W1 must have 48 columns");
        assert_eq!(b1.len(), 48);
        assert_eq!(w2.len(), 48);
        assert_eq!(mu.len(), 12);
        assert_eq!(std.len(), 12);
        let probes = value
            .get("probes")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .map(|item| Probe {
                        instance: item.get("instance").and_then(|v| v.as_i64()).unwrap_or(-1),
                        features: matrix(item.get("features").expect("probe features")),
                        probabilities: vector(
                            item.get("probabilities").expect("probe probabilities"),
                        ),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            w1,
            b1,
            w2,
            b2,
            mu,
            std,
            probes,
        }
    }

    pub fn score(&self, raw: &[f64]) -> f64 {
        assert_eq!(raw.len(), 12);
        let mut hidden = [0.0; 48];
        for j in 0..48 {
            let mut activation = self.b1[j];
            for i in 0..12 {
                let scale = self.std[i].max(1e-6);
                let standardized = (raw[i] - self.mu[i]) / scale;
                activation += self.w1[i][j] * standardized;
            }
            hidden[j] = activation.tanh();
        }
        let mut score = self.b2;
        for j in 0..48 {
            score += self.w2[j] * hidden[j];
        }
        score
    }

    pub fn probe_differences(&self) -> Vec<f64> {
        self.probes
            .iter()
            .map(|probe| {
                let logits: Vec<f64> = probe.features.iter().map(|phi| self.score(phi)).collect();
                let probs = softmax(&logits);
                assert_eq!(
                    probs.len(),
                    probe.probabilities.len(),
                    "probe probability length mismatch"
                );
                probs
                    .iter()
                    .zip(&probe.probabilities)
                    .map(|(got, expected)| (got - expected).abs())
                    .fold(0.0, f64::max)
            })
            .collect()
    }
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum = exps.iter().sum::<f64>();
    exps.into_iter().map(|value| value / sum).collect()
}

fn vector(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("expected a numeric array")
        .iter()
        .map(|item| item.as_f64().expect("expected a float"))
        .collect()
}

fn matrix(value: &Value) -> Vec<Vec<f64>> {
    value
        .as_array()
        .expect("expected a matrix")
        .iter()
        .map(vector)
        .collect()
}
