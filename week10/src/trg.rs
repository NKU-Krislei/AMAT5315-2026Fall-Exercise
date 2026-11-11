//! Tensor renormalization group (TRG) contraction of a periodic square
//! (Algorithm 5).
//!
//! Each round normalizes the uniform site tensor by its largest entry,
//! splits it twice along the two diagonals (SVD with cap chi, singular
//! values shared as square roots), and regroups four half-tensors from a
//! plaquette into one new site tensor. Two steps halve the side length, so
//! n = 2 log2 L rounds leave a single tensor whose opposite legs are joined
//! pairwise: z = sum_{i,j} A[i,j,i,j].

use std::time::Instant;

use crate::ising::site_tensor_a;
use crate::svd::split4_full;
use crate::svd::Split4;

/// Diagnostics and factors of one TRG round.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrgStepDiag {
    pub step: usize,
    pub num_tensors: usize,
    pub scale: f64,
    /// Retained directions of the two diagonal splits.
    pub kept: [usize; 2],
    /// Largest discarded relative weight of the two splits.
    pub discarded: [f64; 2],
    /// Full singular spectra of the two splits, descending.
    pub spectrum_1: Vec<f64>,
    pub spectrum_2: Vec<f64>,
    /// Factor shapes, in the order S1, S2, S3, S4.
    pub factors_dims: Vec<Vec<usize>>,
    /// Raw factors, in the order S1, S2, S3, S4 (row-major).
    pub factors: Vec<Vec<f64>>,
}

/// Result of the TRG contraction.
pub struct TrgResult {
    pub log_z: f64,
    pub g: f64,
    pub ell: f64,
    pub steps: Vec<TrgStepDiag>,
    pub final_tensor: Vec<f64>,
    pub final_dims: [usize; 4],
    pub closure_z: f64,
    pub elapsed_sec: f64,
}

#[inline]
fn idx3(dims: &[usize], i: usize, j: usize, k: usize) -> usize {
    (i * dims[1] + j) * dims[2] + k
}

/// A'_{r u l d} = sum_{w,a,b,gamma} S1[w,a,r] S2[a,b,u] S3[l,b,gamma] S4[d,gamma,w].
///
/// Computed as A' = P1 * P2 with
///   P1[(r u), (w b)] = sum_a S1[w,a,r] S2[a,b,u]
///   P2[(w b), (l d)] = sum_gamma S3[l,b,gamma] S4[d,gamma,w]
fn regroup(
    s1: (&[usize], &[f64]),
    s2: (&[usize], &[f64]),
    s3: (&[usize], &[f64]),
    s4: (&[usize], &[f64]),
) -> ([usize; 4], Vec<f64>) {
    let (d1, s1v) = s1; // (l, u, q1)
    let (d2, s2v) = s2; // (d, l, q2)
    let (d3, s3v) = s3; // (q1, r, d)
    let (d4, s4v) = s4; // (q2, u, r)
    let q1 = d1[2];
    let q2 = d2[2];
    let old_l = d1[0]; // == d2[1] == old leg dims
    let old_u = d1[1]; // == d2[0]
    let old_r = d3[1]; // == d4[2]
    let old_d = d3[2]; // == d4[1]
    debug_assert_eq!(old_l, old_r);
    debug_assert_eq!(old_u, old_d);

    let nru = q1 * q2;
    let nwb = old_l * old_r;
    let nld = q1 * q2;

    // P1[(r u), (w b)]
    let mut p1 = vec![0.0f64; nru * nwb];
    for w in 0..old_l {
        for r in 0..q1 {
            for b in 0..old_r {
                for u in 0..q2 {
                    let mut acc = 0.0;
                    for a in 0..old_u {
                        acc += s1v[idx3(d1, w, a, r)] * s2v[idx3(d2, a, b, u)];
                    }
                    p1[(r * q2 + u) * nwb + (w * old_r + b)] = acc;
                }
            }
        }
    }
    // P2[(w b), (l d)]
    let mut p2 = vec![0.0f64; nwb * nld];
    for w in 0..old_r {
        for b in 0..old_r {
            for l in 0..q1 {
                for d in 0..q2 {
                    let mut acc = 0.0;
                    for g in 0..old_d {
                        acc += s3v[idx3(d3, l, b, g)] * s4v[idx3(d4, d, g, w)];
                    }
                    p2[(w * old_r + b) * nld + (l * q2 + d)] = acc;
                }
            }
        }
    }
    // A' = P1 * P2 (i-k-j order; inner loop vectorizes)
    let mut out = vec![0.0f64; nru * nld];
    for i in 0..nru {
        let crow = &mut out[i * nld..(i + 1) * nld];
        for k in 0..nwb {
            let p = p1[i * nwb + k];
            if p == 0.0 {
                continue;
            }
            let row = &p2[k * nld..(k + 1) * nld];
            for (cv, &rv) in crow.iter_mut().zip(row.iter()) {
                *cv += p * rv;
            }
        }
    }
    ([q1, q2, q1, q2], out)
}

fn discarded_weight(spectrum: &[f64], kept: usize) -> f64 {
    let total2: f64 = spectrum.iter().map(|x| x * x).sum();
    if total2 == 0.0 {
        return 0.0;
    }
    let tail2: f64 = spectrum.iter().skip(kept).map(|x| x * x).sum();
    (tail2 / total2).sqrt()
}

/// Algorithm 5: square-lattice TRG for a periodic L x L torus.
///
/// `l` must be a power of two and at least 2.
pub fn contract_periodic_square(l: usize, k: f64, chi: usize) -> TrgResult {
    assert!(l >= 2, "side length must be at least 2");
    assert!(l.is_power_of_two(), "TRG needs a power-of-two side length");
    assert!(chi >= 1, "chi must be at least 1");
    let t_start = Instant::now();

    let n_steps = 2 * l.ilog2() as usize;
    let mut num_tensors = l * l;
    let mut ell = 0.0f64;

    let mut dims: [usize; 4] = [2, 2, 2, 2];
    let mut data: Vec<f64> = site_tensor_a(k).data;

    let mut steps: Vec<TrgStepDiag> = Vec::with_capacity(n_steps);
    for j in 1..=n_steps {
        // Normalize by the largest absolute entry.
        let c = data.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        for x in data.iter_mut() {
            *x /= c;
        }
        ell += (num_tensors as f64) * c.ln();

        // Two diagonal splits of the same tensor.
        // (S1(l,u,q), S3(q,r,d)) <- Split(A, (l,u), (r,d), chi)
        // (S2(d,l,q), S4(q,u,r)) <- Split(A, (d,l), (u,r), chi)
        let f1: Split4 = split4_full(dims, &data, [2, 1], [0, 3], chi);
        let f2: Split4 = split4_full(dims, &data, [3, 2], [1, 0], chi);

        let s1: (&[usize], &[f64]) = (&f1.xdims, f1.x.as_slice());
        let s2: (&[usize], &[f64]) = (&f2.xdims, f2.x.as_slice());
        let s3: (&[usize], &[f64]) = (&f1.ydims, f1.y.as_slice());
        let s4: (&[usize], &[f64]) = (&f2.ydims, f2.y.as_slice());

        let (newdims, newdata) = regroup(s1, s2, s3, s4);

        steps.push(TrgStepDiag {
            step: j,
            num_tensors,
            scale: c,
            kept: [f1.kept, f2.kept],
            discarded: [
                discarded_weight(&f1.spectrum, f1.kept),
                discarded_weight(&f2.spectrum, f2.kept),
            ],
            spectrum_1: f1.spectrum.clone(),
            spectrum_2: f2.spectrum.clone(),
            factors_dims: vec![
                f1.xdims.clone(), f2.xdims.clone(), f1.ydims.clone(), f2.ydims.clone(),
            ],
            factors: vec![
                f1.x.clone(),
                f2.x.clone(),
                f1.y.clone(),
                f2.y.clone(),
            ],
        });

        dims = newdims;
        data = newdata;
        num_tensors /= 2;
    }

    // Close the final tensor by joining opposite legs in pairs.
    let [dr, du, dl, dd] = dims;
    debug_assert_eq!(dr, dl);
    debug_assert_eq!(du, dd);
    let mut z = 0.0f64;
    for i in 0..dr {
        for j in 0..du {
            let o = ((i * du + j) * dl + i) * dd + j;
            z += data[o];
        }
    }
    let log_z = ell + z.ln();
    let g = log_z / (l * l) as f64;
    let elapsed_sec = t_start.elapsed().as_secs_f64();
    TrgResult {
        log_z,
        g,
        ell,
        steps,
        final_tensor: data,
        final_dims: dims,
        closure_z: z,
        elapsed_sec,
    }
}
