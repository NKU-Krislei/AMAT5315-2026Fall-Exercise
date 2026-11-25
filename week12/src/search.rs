//! Depth-first branch-and-bound over spin-glass LP relaxations.
//!
//! The branching rule only chooses a fractional spin. Pruning and incumbent
//! updates stay with the LP, so a wrong network cannot change the ground state.

use std::time::Instant;

use crate::glass::{GraphInfo, SpinGlass};
use crate::lp::{free_indices, LpModel, Relaxation};
use crate::nn::Mlp;

const PRUNE_TOL: f64 = 1e-6;
const GAIN_CUT: f64 = 1.0e6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    MostFrac,
    Strong,
    Neural,
    RootNeural,
}

impl Policy {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "most-frac" => Some(Self::MostFrac),
            "strong" => Some(Self::Strong),
            "neural" => Some(Self::Neural),
            "root-neural" => Some(Self::RootNeural),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::MostFrac => "most-frac",
            Self::Strong => "strong",
            Self::Neural => "neural",
            Self::RootNeural => "root-neural",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TraceNode {
    pub node: usize,
    pub branch: String,
    pub status: String,
    pub lp: Option<f64>,
    pub x: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct LabelNode {
    pub instance: u64,
    pub candidates: Vec<usize>,
    pub features: Vec<[f64; 12]>,
    pub gains0: Vec<f64>,
    pub gains1: Vec<f64>,
    pub scores: Vec<f64>,
    pub pruned_children: Vec<u32>,
}

pub struct SearchOutput {
    pub energy: f64,
    pub nodes: u64,
    pub lp_solves: u64,
    pub wall_ms: f64,
    pub lp_ms: f64,
    pub infer_ms: f64,
    pub splits: Vec<usize>,
    pub trace: Vec<TraceNode>,
    pub labels: Vec<LabelNode>,
}

struct Pending {
    lower: Vec<f64>,
    upper: Vec<f64>,
    branch: String,
}

struct Clock {
    lp_solves: u64,
    lp_ms: f64,
    infer_ms: f64,
}

pub fn search(
    glass: &SpinGlass,
    policy: Policy,
    mlp: Option<&Mlp>,
    trace: bool,
    label_limit: Option<usize>,
    instance_id: u64,
) -> SearchOutput {
    let started = Instant::now();
    let lp = LpModel::new(glass);
    let info = GraphInfo::new(glass);
    let mut stack = vec![Pending {
        lower: vec![0.0; glass.n],
        upper: vec![1.0; glass.n],
        branch: "root".to_string(),
    }];
    let mut z_best = f64::INFINITY;
    let mut nodes = 0_u64;
    let mut clock = Clock {
        lp_solves: 0,
        lp_ms: 0.0,
        infer_ms: 0.0,
    };
    let mut splits = Vec::new();
    let mut trace_nodes = Vec::new();
    let mut labels = Vec::new();
    let mut root_rank: Option<Vec<f64>> = None;

    while let Some(pending) = stack.pop() {
        nodes += 1;
        let solution = timed_lp(
            &mut clock,
            &lp,
            &pending.lower,
            &pending.upper,
        );
        if !solution.optimal || solution.energy >= z_best - PRUNE_TOL {
            if trace {
                trace_nodes.push(TraceNode {
                    node: nodes as usize,
                    branch: pending.branch,
                    status: if solution.optimal {
                        "pruned".to_string()
                    } else {
                        "infeasible".to_string()
                    },
                    lp: solution.optimal.then_some(solution.energy),
                    x: solution.x.clone(),
                });
            }
            continue;
        }

        let free = free_indices(&solution.x);
        if free.is_empty() {
            z_best = solution.energy;
            if trace {
                trace_nodes.push(TraceNode {
                    node: nodes as usize,
                    branch: pending.branch,
                    status: "incumbent".to_string(),
                    lp: Some(solution.energy),
                    x: solution.x.clone(),
                });
            }
            continue;
        }

        if trace {
            trace_nodes.push(TraceNode {
                node: nodes as usize,
                branch: pending.branch.clone(),
                status: "fractional".to_string(),
                lp: Some(solution.energy),
                x: solution.x.clone(),
            });
        }

        let fixed = fixed_flags(&pending.lower, &pending.upper);
        let chosen = match policy {
            Policy::MostFrac => pick_most_frac(&free, &solution.x),
            Policy::Strong => {
                let (chosen, gains0, gains1, scores, pruned) = pick_strong(
                    &lp,
                    &pending.lower,
                    &pending.upper,
                    &free,
                    solution.energy,
                    z_best,
                    &mut clock,
                );
                if let Some(limit) = label_limit {
                    if free.len() >= 2 && labels.len() < limit {
                        let features = free
                            .iter()
                            .map(|&i| {
                                spin_features(
                                    glass,
                                    &info,
                                    &solution,
                                    &fixed,
                                    z_best,
                                    solution.energy,
                                    i,
                                )
                            })
                            .collect();
                        labels.push(LabelNode {
                            instance: instance_id,
                            candidates: free.clone(),
                            features,
                            gains0,
                            gains1,
                            scores,
                            pruned_children: pruned,
                        });
                    }
                }
                chosen
            }
            Policy::Neural => {
                let network = mlp.expect("neural branching needs a trained model");
                let infer = Instant::now();
                let chosen = pick_scored(&free, |i| {
                    network.score(&spin_features(
                        glass,
                        &info,
                        &solution,
                        &fixed,
                        z_best,
                        solution.energy,
                        i,
                    ))
                });
                clock.infer_ms += elapsed_ms(infer);
                chosen
            }
            Policy::RootNeural => {
                let network = mlp.expect("root-neural branching needs a trained model");
                if root_rank.is_none() {
                    let infer = Instant::now();
                    let scores: Vec<f64> = (0..glass.n)
                        .map(|i| {
                            network.score(&spin_features(
                                glass,
                                &info,
                                &solution,
                                &fixed,
                                z_best,
                                solution.energy,
                                i,
                            ))
                        })
                        .collect();
                    root_rank = Some(standardize_scores(&scores));
                    clock.infer_ms += elapsed_ms(infer);
                }
                let rank = root_rank.as_ref().expect("root scores");
                pick_root(rank, &free, &solution.x)
            }
        };

        if splits.len() < 2 {
            splits.push(chosen);
        }
        push_children(&mut stack, &pending, chosen, solution.x[chosen]);
    }

    SearchOutput {
        energy: z_best,
        nodes,
        lp_solves: clock.lp_solves,
        wall_ms: elapsed_ms(started),
        lp_ms: clock.lp_ms,
        infer_ms: clock.infer_ms,
        splits,
        trace: trace_nodes,
        labels,
    }
}

fn timed_lp(clock: &mut Clock, lp: &LpModel, lower: &[f64], upper: &[f64]) -> Relaxation {
    let started = Instant::now();
    let solution = lp.solve(lower, upper);
    clock.lp_ms += elapsed_ms(started);
    clock.lp_solves += 1;
    solution
}

fn pick_most_frac(free: &[usize], x: &[f64]) -> usize {
    let mut best = free[0];
    let mut best_dist = (x[best] - 0.5).abs();
    for &i in &free[1..] {
        let dist = (x[i] - 0.5).abs();
        if dist < best_dist {
            best = i;
            best_dist = dist;
        }
    }
    best
}

fn pick_strong(
    lp: &LpModel,
    lower: &[f64],
    upper: &[f64],
    free: &[usize],
    parent_energy: f64,
    z_best: f64,
    clock: &mut Clock,
) -> (usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>) {
    let mut best_i = free[0];
    let mut best_score = f64::NEG_INFINITY;
    let mut gains0 = Vec::with_capacity(free.len());
    let mut gains1 = Vec::with_capacity(free.len());
    let mut scores = Vec::with_capacity(free.len());
    let mut pruned = Vec::with_capacity(free.len());
    for &i in free {
        let (gain0, cut0) = child_gain(lp, lower, upper, i, 0.0, parent_energy, z_best, clock);
        let (gain1, cut1) = child_gain(lp, lower, upper, i, 1.0, parent_energy, z_best, clock);
        let score = gain0.max(PRUNE_TOL) * gain1.max(PRUNE_TOL);
        if score > best_score {
            best_score = score;
            best_i = i;
        }
        gains0.push(gain0);
        gains1.push(gain1);
        scores.push(score);
        pruned.push(u32::from(cut0) + u32::from(cut1));
    }
    (best_i, gains0, gains1, scores, pruned)
}

fn child_gain(
    lp: &LpModel,
    lower: &[f64],
    upper: &[f64],
    spin: usize,
    value: f64,
    parent_energy: f64,
    z_best: f64,
    clock: &mut Clock,
) -> (f64, bool) {
    let mut child_lo = lower.to_vec();
    let mut child_hi = upper.to_vec();
    child_lo[spin] = value;
    child_hi[spin] = value;
    let child = timed_lp(clock, lp, &child_lo, &child_hi);
    if !child.optimal || child.energy >= z_best - PRUNE_TOL {
        (GAIN_CUT, true)
    } else {
        (child.energy - parent_energy, false)
    }
}

fn pick_scored(free: &[usize], mut score_of: impl FnMut(usize) -> f64) -> usize {
    let mut best = free[0];
    let mut best_score = f64::NEG_INFINITY;
    for &i in free {
        let score = score_of(i);
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best
}

fn pick_root(rank: &[f64], free: &[usize], x: &[f64]) -> usize {
    pick_scored(free, |i| {
        let clamped = x[i].clamp(0.0, 1.0);
        let frac = 2.0 * clamped.min(1.0 - clamped);
        rank[i] + 3.0 * frac
    })
}

fn standardize_scores(scores: &[f64]) -> Vec<f64> {
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores
        .iter()
        .map(|score| {
            let delta = score - mean;
            delta * delta
        })
        .sum::<f64>()
        / n;
    let scale = variance.sqrt().max(1e-6);
    scores
        .iter()
        .map(|score| (score - mean) / scale)
        .collect()
}

fn push_children(stack: &mut Vec<Pending>, parent: &Pending, spin: usize, value: f64) {
    let make = |bit: f64| {
        let mut lower = parent.lower.clone();
        let mut upper = parent.upper.clone();
        lower[spin] = bit;
        upper[spin] = bit;
        Pending {
            lower,
            upper,
            branch: format!("x{spin}={bit:.0}"),
        }
    };
    let nearer_one = (value - 1.0).abs() <= value.abs();
    if nearer_one {
        stack.push(make(0.0));
        stack.push(make(1.0));
    } else {
        stack.push(make(1.0));
        stack.push(make(0.0));
    }
}

fn fixed_flags(lower: &[f64], upper: &[f64]) -> Vec<bool> {
    lower
        .iter()
        .zip(upper)
        .map(|(lo, hi)| hi - lo <= 1e-9)
        .collect()
}

fn spin_features(
    glass: &SpinGlass,
    info: &GraphInfo,
    solution: &Relaxation,
    fixed: &[bool],
    z_best: f64,
    z_node: f64,
    i: usize,
) -> [f64; 12] {
    let x = &solution.x;
    let degree = info.degree[i];
    let degree_denom = degree.max(1) as f64;
    let mut positive = 0.0;
    let mut negative = 0.0;
    let mut field = glass.fields[i];
    let mut dual_sum = 0.0;
    let mut gap_sum = 0.0;
    let mut neighbor_sum = 0.0;
    let spin_i = spin_value(x[i]);
    let dual_scale = solution
        .row_duals
        .iter()
        .map(|dual| dual.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    for &(neighbor, edge) in &info.adj[i] {
        let coupling = glass.couplings[edge];
        if coupling > 0.0 {
            positive += 1.0;
        } else if coupling < 0.0 {
            negative += 1.0;
        }
        let spin_j = spin_value(x[neighbor]);
        field += coupling * spin_j;
        let mut abs_dual = 0.0;
        for row in 0..4 {
            abs_dual += solution.row_duals[edge * 4 + row].abs();
        }
        dual_sum += abs_dual / dual_scale;
        let implied = 0.5 * (1.0 - spin_i * spin_j);
        gap_sum += (solution.z[edge] - implied).abs();
        neighbor_sum += if fixed[neighbor] {
            0.0
        } else {
            frac_dist(x[neighbor])
        };
    }
    let mean_denom = if degree == 0 { 1.0 } else { degree as f64 };
    let effective = field / (degree as f64 + 1.0);
    let gap_ratio = if z_best.is_finite() {
        ((z_best - z_node) / z_best.abs().max(1.0)).clamp(0.0, 2.0)
    } else {
        1.0
    };
    [
        x[i].clamp(0.0, 1.0),
        frac_dist(x[i]),
        glass.fields[i],
        glass.fields[i].abs(),
        degree as f64 / info.d_max.max(1) as f64,
        (positive - negative) / degree_denom,
        effective,
        effective.abs(),
        dual_sum / mean_denom,
        gap_sum / mean_denom,
        neighbor_sum / mean_denom,
        gap_ratio,
    ]
}

fn spin_value(x: f64) -> f64 {
    1.0 - 2.0 * x.clamp(0.0, 1.0)
}

fn frac_dist(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    2.0 * x.min(1.0 - x)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}
