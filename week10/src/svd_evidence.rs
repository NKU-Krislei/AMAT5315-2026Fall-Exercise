//! Part 1 hands-on: SVD compression evidence for a Gaussian random matrix.
//!
//! `generate` builds the matrix (independent standard-normal entries, recorded
//! generator and seed), computes its SVD, and saves the matrix, the singular
//! values, and the truncated reconstructions at the requested ranks.
//!
//! `check` reads back only the saved files, independently recomputes the
//! relative Frobenius errors, and compares them with the discarded-spectrum
//! prediction.

use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde_json::json;

use crate::evidence as ev;
use crate::svd::{svd_row_major, Svd};

pub const GENERATOR: &str = "rand StdRng (ChaCha12 core) seeded with SeedableRng::seed_from_u64; entries sampled with rand_distr::StandardNormal (ziggurat); crate versions are pinned in Cargo.lock";

pub struct GenSettings {
    pub size: usize,
    pub seed: u64,
    pub ranks: Vec<usize>,
}

/// Squared Frobenius norm of a row-major matrix.
fn fro2(m: usize, n: usize, a: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..m {
        let mut row_acc = 0.0;
        for j in 0..n {
            let v = a[i * n + j];
            row_acc += v * v;
        }
        acc += row_acc;
    }
    acc
}

/// Reconstruct the rank-`chi` approximation M_chi from truncated factors.
pub fn reconstruct(f: &Svd) -> Vec<f64> {
    let (m, n, k) = (f.m, f.n, f.k);
    let mut out = vec![0.0; m * n];
    for a in 0..k {
        let s = f.s[a];
        for i in 0..m {
            let u = f.u_at(i, a) * s;
            if u != 0.0 {
                let vt_ofs = a * n;
                for j in 0..n {
                    out[i * n + j] += u * f.vt[vt_ofs + j];
                }
            }
        }
    }
    out
}

pub struct CheckRow {
    pub chi: usize,
    pub measured: f64,
    pub predicted: f64,
    pub diff: f64,
    pub factor_entries: usize,
}

pub struct CheckReport {
    pub rows: Vec<CheckRow>,
    pub norm_fro: f64,
    pub original_entries: usize,
    pub storage_parity_chi: usize,
    pub chi_10pct: usize,
    pub factor_over_original_at_10pct: f64,
    pub max_diff: f64,
    pub full_rank_error: f64,
}

impl CheckReport {
    pub fn passed(&self, tol: f64, full_tol: f64) -> bool {
        self.max_diff < tol && self.full_rank_error < full_tol
    }
}

pub fn generate(out: &Path, settings: &GenSettings) -> std::io::Result<serde_json::Value> {
    ev::ensure_dir(out)?;
    let n = settings.size;
    let t0 = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let m: Vec<f64> = (0..n * n)
        .map(|_| rng.sample(StandardNormal))
        .collect();
    let gen_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let f = svd_row_major(n, n, &m);
    let svd_time = t1.elapsed();

    ev::write_f64_bin(&out.join("matrix.bin"), &m)?;
    ev::write_f64_bin(&out.join("singular_values.bin"), &f.s)?;
    let mut files = vec!["matrix.bin".to_string(), "singular_values.bin".to_string()];
    let mut rank_files = Vec::new();
    for &chi in &settings.ranks {
        let t = f.truncated(chi);
        let rec = reconstruct(&t);
        let name = format!("reconstruction_r{}.bin", chi);
        ev::write_f64_bin(&out.join(&name), &rec)?;
        rank_files.push(json!({"chi": chi, "file": name}));
        files.push(name);
    }

    let norm_fro = fro2(n, n, &m).sqrt();
    let meta = json!({
        "tool": "svd",
        "purpose": "SVD compression evidence for a Gaussian matrix",
        "size": n,
        "seed": settings.seed,
        "generator": GENERATOR,
        "ranks": settings.ranks,
        "norm_fro": norm_fro,
        "singular_values_sorted": "descending",
        "files": files,
        "reconstructions": rank_files,
        "generation_time_sec": gen_time.as_secs_f64(),
        "svd_time_sec": svd_time.as_secs_f64(),
        "timestamp_utc": ev::utc_timestamp(),
    });
    ev::write_json(&out.join("meta.json"), &meta)?;
    Ok(meta)
}

fn discarded_spectrum_prediction(s: &[f64], chi: usize, norm_fro: f64) -> f64 {
    let tail: f64 = s.iter().skip(chi).map(|x| x * x).sum();
    tail.sqrt() / norm_fro
}

/// Recompute errors from the files saved by `generate` only.
pub fn check(out: &Path) -> std::io::Result<CheckReport> {
    let meta = ev::read_json(&out.join("meta.json"))?;
    let n = meta["size"].as_u64().unwrap() as usize;
    let m = ev::read_f64_bin(&out.join("matrix.bin"))?;
    let s = ev::read_f64_bin(&out.join("singular_values.bin"))?;
    assert_eq!(m.len(), n * n);
    assert_eq!(s.len(), n.min(n));
    let norm_fro = fro2(n, n, &m).sqrt();

    let ranks: Vec<usize> = meta["ranks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let mut rows = Vec::new();
    for &chi in &ranks {
        let rec = ev::read_f64_bin(&out.join(format!("reconstruction_r{}.bin", chi)))?;
        assert_eq!(rec.len(), n * n);
        let mut d2 = 0.0;
        for i in 0..n {
            let mut row_acc = 0.0;
            for j in 0..n {
                let d = m[i * n + j] - rec[i * n + j];
                row_acc += d * d;
            }
            d2 += row_acc;
        }
        let measured = d2.sqrt() / norm_fro;
        let predicted = discarded_spectrum_prediction(&s, chi.min(s.len()), norm_fro);
        let factor_entries = 2 * n * chi;
        rows.push(CheckRow {
            chi,
            measured,
            predicted,
            diff: (measured - predicted).abs(),
            factor_entries,
        });
    }
    let max_diff = rows.iter().map(|r| r.diff).fold(0.0f64, f64::max);
    let full_rank_error = rows
        .iter()
        .find(|r| r.chi >= s.len())
        .map(|r| r.measured)
        .unwrap_or(f64::INFINITY);
    let original_entries = n * n;
    let storage_parity_chi = original_entries / (2 * n);
    let mut chi_10pct = 0;
    for chi in 1..=s.len() {
        if discarded_spectrum_prediction(&s, chi, norm_fro) <= 0.10 {
            chi_10pct = chi;
            break;
        }
    }
    let factor_over_original_at_10pct = (2.0 * n as f64 * chi_10pct as f64) / original_entries as f64;
    Ok(CheckReport {
        rows,
        norm_fro,
        original_entries,
        storage_parity_chi,
        chi_10pct,
        factor_over_original_at_10pct,
        max_diff,
        full_rank_error,
    })
}

#[allow(dead_code)]
pub fn matrix_path(out: &Path) -> PathBuf {
    out.join("matrix.bin")
}
