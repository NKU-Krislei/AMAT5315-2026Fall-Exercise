//! SVD utilities.
//!
//! Wraps faer's SVD in a row-major interface and implements the two routines from
//! the learning sheet: Algorithm 1 (TruncatedSVD) and Algorithm 2 (Split).

use dyn_stack::MemBuffer;
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors, SvdParams};
use faer::{Auto, Mat, Par, Spec};

/// Result of an SVD of an `m x n` matrix: `a = U diag(s) Vt`, row-major storage.
///
/// `k = min(m, n)`. The singular values are sorted in descending order and the
/// columns of `U` / rows of `Vt` are permuted to match.
#[derive(Debug, Clone)]
pub struct Svd {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    pub vt: Vec<f64>,
}

impl Svd {
    #[inline]
    pub fn u_at(&self, i: usize, a: usize) -> f64 {
        self.u[i * self.k + a]
    }
    #[inline]
    pub fn vt_at(&self, a: usize, j: usize) -> f64 {
        self.vt[a * self.n + j]
    }

    /// Rank-`chi` truncation (Algorithm 1): keep at most `chi` singular directions.
    pub fn truncated(&self, chi: usize) -> Svd {
        let kk = chi.min(self.k);
        let mut u = vec![0.0; self.m * kk];
        let mut vt = vec![0.0; kk * self.n];
        for i in 0..self.m {
            for a in 0..kk {
                u[i * kk + a] = self.u_at(i, a);
            }
        }
        for a in 0..kk {
            for j in 0..self.n {
                vt[a * self.n + j] = self.vt_at(a, j);
            }
        }
        Svd {
            m: self.m,
            n: self.n,
            k: kk,
            u,
            s: self.s[..kk].to_vec(),
            vt,
        }
    }
}

/// Compute the full (thin) SVD of the row-major `m x n` matrix `a`.
pub fn svd_row_major(m: usize, n: usize, a: &[f64]) -> Svd {
    assert_eq!(a.len(), m * n);
    let mut mat = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            *mat.get_mut(i, j) = a[i * n + j];
        }
    }
    let k = m.min(n);
    let params = Spec::new(<SvdParams as Auto<f64>>::auto());
    let par = Par::rayon(0);
    let req = svd_scratch::<f64>(m, n, ComputeSvdVectors::Thin, ComputeSvdVectors::Thin, par, params);
    let mut mem = MemBuffer::new(req);
    let stack = dyn_stack::MemStack::new(&mut mem);
    let mut smat = Mat::<f64>::zeros(m, n);
    let mut u = Mat::<f64>::zeros(m, k);
    let mut v = Mat::<f64>::zeros(n, k);
    svd(
        mat.as_ref(),
        smat.as_mut().diagonal_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        stack,
        params,
    )
    .expect("SVD failed to converge");
    // faer returns M = U diag(s) V^H; for real matrices Vt = V^T.
    let mut out = Svd {
        m,
        n,
        k,
        u: vec![0.0; m * k],
        s: vec![0.0; k],
        vt: vec![0.0; k * n],
    };
    let mut perm: Vec<usize> = (0..k).collect();
    perm.sort_by(|&x, &y| {
        smat.get(y, y)
            .partial_cmp(smat.get(x, x))
            .unwrap()
    });
    for (col, &src) in perm.iter().enumerate() {
        out.s[col] = *smat.get(src, src);
        for i in 0..m {
            out.u[i * k + col] = *u.get(i, src);
        }
        for j in 0..n {
            out.vt[col * n + j] = *v.get(j, src);
        }
    }
    out
}

/// Algorithm 1: TruncatedSVD(M, chi). Returns factors keeping at most `chi`
/// singular directions.
pub fn truncated_svd(m: usize, n: usize, a: &[f64], chi: usize) -> Svd {
    assert!(chi >= 1, "rank cap chi must be at least 1");
    svd_row_major(m, n, a).truncated(chi)
}

/// Algorithm 2 wrapper: Split(T, I, J, chi) returning only the two factors.
pub fn split4(
    dims: [usize; 4],
    data: &[f64],
    group_left: [usize; 2],
    group_right: [usize; 2],
    chi: usize,
) -> (Vec<usize>, Vec<f64>, Vec<usize>, Vec<f64>) {
    let f = split4_full(dims, data, group_left, group_right, chi);
    (f.xdims, f.x, f.ydims, f.y)
}

/// Full output of a four-leg tensor split, including the untruncated
/// singular spectrum for diagnostics.
#[derive(Debug, Clone)]
pub struct Split4 {
    /// Factor X with legs `group_left + [q]`, row-major.
    pub xdims: Vec<usize>,
    pub x: Vec<f64>,
    /// Factor Y with legs `[q] + group_right`, row-major.
    pub ydims: Vec<usize>,
    pub y: Vec<f64>,
    /// Full singular spectrum of the grouped matrix, descending.
    pub spectrum: Vec<f64>,
    /// Number of retained directions (min(chi, spectrum.len())).
    pub kept: usize,
}

/// Algorithm 2: Split(T, I, J, chi) for a four-leg tensor stored row-major in
/// the leg order `(a0, a1, a2, a3)`.
///
/// `group_left` / `group_right` list leg positions (each list must be a
/// permutation-free partition of {0,1,2,3}). The tensor is permuted so that the
/// left group comes first, flattened into a matrix, truncated-SVD'd with cap
/// `chi`, and the singular values are shared as sqrt(s) between the two
/// factors.
pub fn split4_full(
    dims: [usize; 4],
    data: &[f64],
    group_left: [usize; 2],
    group_right: [usize; 2],
    chi: usize,
) -> Split4 {
    let mut perm = [group_left[0], group_left[1], group_right[0], group_right[1]];
    perm.sort(); // sanity: the two groups must partition the legs
    debug_assert_eq!(perm, [0, 1, 2, 3], "groups must partition the four legs");

    let dl: usize = dims[group_left[0]] * dims[group_left[1]];
    let dr: usize = dims[group_right[0]] * dims[group_right[1]];
    // Flatten each group with the last index varying fastest (Algorithm 2).
    // data is row-major over (a0, a1, a2, a3); we gather M[(I), (J)].
    let mut mbuf = vec![0.0; dl * dr];
    {
    let strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
    let gl = group_left;
    let gr = group_right;
    let sl = [strides[gl[0]], strides[gl[1]]];
    let sr = [strides[gr[0]], strides[gr[1]]];
    let ldim = [dims[gl[0]], dims[gl[1]]];
    let rdim = [dims[gr[0]], dims[gr[1]]];
    for i0 in 0..ldim[0] {
        for i1 in 0..ldim[1] {
            let row = i0 * ldim[1] + i1;
            for j0 in 0..rdim[0] {
                for j1 in 0..rdim[1] {
                    let col = j0 * rdim[1] + j1;
                    let src = i0 * sl[0] + i1 * sl[1] + j0 * sr[0] + j1 * sr[1];
                    mbuf[row * dr + col] = data[src];
                }
            }
        }
    }
    } // scope borrow of strides

    // Full spectrum first (for diagnostics), then truncate.
    let full = svd_row_major(dl, dr, &mbuf);
    let f = full.truncated(chi);
    let q = f.k;
    let mut sqrt_s = vec![0.0; q];
    for a in 0..q {
        sqrt_s[a] = f.s[a].sqrt();
    }
    // x has legs group_left + [q]: x[I, a] = U[I, a] * sqrt(s_a)
    let mut x = vec![0.0; dl * q];
    for i in 0..dl {
        for a in 0..q {
            x[i * q + a] = f.u_at(i, a) * sqrt_s[a];
        }
    }
    // y has legs [q] + group_right: y[a, J] = sqrt(s_a) * Vt[a, J]
    let mut y = vec![0.0; q * dr];
    for a in 0..q {
        for j in 0..dr {
            y[a * dr + j] = sqrt_s[a] * f.vt_at(a, j);
        }
    }
    let xdims: Vec<usize> = vec![dims[group_left[0]], dims[group_left[1]], q];
    let ydims: Vec<usize> = vec![q, dims[group_right[0]], dims[group_right[1]]];
    Split4 {
        xdims,
        x,
        ydims,
        y,
        spectrum: full.s,
        kept: q,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn svd_reconstructs_and_sorts() {
        // 4x3 matrix with known spectrum 4, 2, 1.
        let m = 4;
        let n = 3;
        let a: Vec<f64> = vec![
            3.0 / 2.0 * 2.0, 1.0 / 2.0 * 2.0, 0.0, //
            1.0 / 2.0 * 2.0, 3.0 / 2.0 * 2.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0,
        ];
        let f = svd_row_major(m, n, &a);
        assert_eq!(f.k, 3);
        assert!(approx_eq(f.s[0], 4.0, 1e-12), "s0 = {}", f.s[0]);
        assert!(approx_eq(f.s[1], 2.0, 1e-12), "s1 = {}", f.s[1]);
        assert!(approx_eq(f.s[2], 1.0, 1e-12), "s2 = {}", f.s[2]);
        for i in 0..m {
            for j in 0..n {
                let mut v = 0.0;
                for a in 0..f.k {
                    v += f.u_at(i, a) * f.s[a] * f.vt_at(a, j);
                }
                assert!(
                    approx_eq(v, a[i * n + j], 1e-12),
                    "reconstruction at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn truncated_error_matches_discarded_spectrum() {
        // Random-ish matrix: truncated SVD error must equal the discarded tail.
        let n = 40;
        let mut a = vec![0.0; n * n];
        let mut state = 12345u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for v in a.iter_mut() {
            *v = next() * 10.0;
        }
        let f = svd_row_major(n, n, &a);
        let fro = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        for chi in [1usize, 5, 17, 40] {
            let t = f.truncated(chi);
            let mut diff2 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let mut rec = 0.0;
                    for a2 in 0..t.k {
                        rec += t.u_at(i, a2) * t.s[a2] * t.vt_at(a2, j);
                    }
                    let d = a[i * n + j] - rec;
                    diff2 += d * d;
                }
            }
            let tail: f64 = f.s.iter().skip(t.k).map(|x| x * x).sum();
            let e_meas = diff2.sqrt() / fro;
            let e_pred = tail.sqrt() / fro;
            assert!(
                (e_meas - e_pred).abs() < 1e-10,
                "chi={chi}: measured {e_meas} vs predicted {e_pred}"
            );
        }
    }

    #[test]
    fn split4_is_exact_when_chi_is_large() {
        // Split a 2x2x2x2 tensor with chi=4 (full rank): the two factors must
        // contract back to the original tensor.
        let dims = [2, 2, 2, 2];
        let data: Vec<f64> = (0..16u32).map(|i| (i as f64) - 7.5).collect();
        let (xd, x, yd, y) = split4(dims, &data, [0, 1], [2, 3], 4);
        assert_eq!(xd, vec![2, 2, 4]);
        assert_eq!(yd, vec![4, 2, 2]);
        for a0 in 0..2 {
            for a1 in 0..2 {
                for a2 in 0..2 {
                    for a3 in 0..2 {
                        let mut v = 0.0;
                        for q in 0..4 {
                            let xi = (a0 * 2 + a1) * 4 + q;
                            let yi = q * 4 + a2 * 2 + a3;
                            v += x[xi] * y[yi];
                        }
                        let orig = data[((a0 * 2 + a1) * 2 + a2) * 2 + a3];
                        assert!(
                            approx_eq(v, orig, 1e-12),
                            "({a0},{a1},{a2},{a3}): {v} vs {orig}"
                        );
                    }
                }
            }
        }
    }
}
