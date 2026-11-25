//! Boundary-MPS contraction of an open square (Algorithm 3 and 4).
//!
//! A chain of three-index cores B[i]_{alpha, p, beta} summarises the rows
//! already absorbed. Each row absorption grows the horizontal bonds from
//! chi to 2*chi; Algorithm 3 (canonicalize right-to-left, then truncate
//! left-to-right) brings them back to at most chi. Only the last core carries
//! the boundary scale, so normalising it rescales the whole boundary; the
//! logarithms are accumulated in `ell` and restored at the end.

use std::time::Instant;

use crate::ising::{edge_vector_b, row_tensor_t, Tensor4};
use crate::svd::{svd_row_major, Svd};

/// One MPS core with legs (left bond, physical, right bond), row-major.
#[derive(Debug, Clone)]
pub struct Core {
    pub dl: usize,
    pub dp: usize,
    pub dr: usize,
    pub data: Vec<f64>,
}

impl Core {
    pub fn zeros(dl: usize, dp: usize, dr: usize) -> Self {
        Core {
            dl,
            dp,
            dr,
            data: vec![0.0; dl * dp * dr],
        }
    }

    #[inline]
    pub fn at(&self, a: usize, p: usize, b: usize) -> f64 {
        self.data[(a * self.dp + p) * self.dr + b]
    }

    #[inline]
    pub fn set(&mut self, a: usize, p: usize, b: usize, v: f64) {
        let o = (a * self.dp + p) * self.dr + b;
        self.data[o] = v;
    }

    pub fn fro_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

/// Per-row diagnostics of one compression.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RowDiag {
    pub row: usize,
    pub scale: f64,
    /// Largest discarded-tail weight of the row's compression, relative to
    /// the full spectrum: max over bonds of sqrt(sum_{a>chi} s_a^2 / sum_a s_a^2).
    pub max_discarded: f64,
    /// Bond dimensions after compression (L-1 entries).
    pub bond_dims: Vec<usize>,
}

/// Result of the boundary-MPS contraction.
pub struct BoundaryResult {
    pub log_z: f64,
    pub g: f64,
    pub ell: f64,
    pub row_diag: Vec<RowDiag>,
    pub cores: Vec<Core>,
    pub closure_value: f64,
    pub elapsed_sec: f64,
}

/// View a core as a matrix.
///
/// `group_left == true` flattens M[(alpha p), beta] (pass 2 of Algorithm 3);
/// `group_left == false` flattens M[alpha, (p beta)] (pass 1), with p major
/// and beta fastest inside each group.
fn core_as_matrix(c: &Core, group_left: bool) -> (usize, usize, Vec<f64>) {
    if group_left {
        let m = c.dl * c.dp;
        let n = c.dr;
        let mut buf = vec![0.0; m * n];
        for a in 0..c.dl {
            for p in 0..c.dp {
                let row = a * c.dp + p;
                for b in 0..c.dr {
                    buf[row * n + b] = c.at(a, p, b);
                }
            }
        }
        (m, n, buf)
    } else {
        let m = c.dl;
        let n = c.dp * c.dr;
        let mut buf = vec![0.0; m * n];
        for a in 0..c.dl {
            for p in 0..c.dp {
                for b in 0..c.dr {
                    buf[a * n + p * c.dr + b] = c.at(a, p, b);
                }
            }
        }
        (m, n, buf)
    }
}

/// Algorithm 3: canonicalize and truncate an MPS to bond dimension chi.
///
/// Pass 1 (right-to-left, exact SVDs, no truncation) makes cores 2..L
/// right-isometric. Pass 2 (left-to-right, truncated SVDs) caps each bond
/// at chi and leaves cores 1..L-1 left-isometric. Returns the largest
/// discarded relative weight over the truncated bonds.
pub fn compress(cores: &[Core], chi: usize) -> (Vec<Core>, f64) {
    let mut cs: Vec<Core> = cores.to_vec();
    let lc = cs.len();
    // Pass 1: for i = L down to 2, SVD B[i] as M[alpha, (p beta)], exact.
    for i in (1..lc).rev() {
        let (m, n, buf) = core_as_matrix(&cs[i], false);
        let f: Svd = svd_row_major(m, n, &buf);
        let q = f.k;
        // B[i] <- V^T reshaped (q, p, beta)
        let mut newc = Core::zeros(q, cs[i].dp, cs[i].dr);
        for a in 0..q {
            for p in 0..cs[i].dp {
                for bcol in 0..cs[i].dr {
                    newc.set(a, p, bcol, f.vt_at(a, p * cs[i].dr + bcol));
                }
            }
        }
        cs[i] = newc;
        // B[i-1]_{alpha p q} <- sum_gamma B[i-1]_{alpha p gamma} U[gamma q] s_q
        let prev = &mut cs[i - 1];
        let rows = prev.dl * prev.dp;
        let cols = prev.dr;
        let mut newprev = Core::zeros(prev.dl, prev.dp, q);
        for r in 0..rows {
            for a in 0..q {
                let mut acc = 0.0;
                for g in 0..cols {
                    acc += prev.data[r * cols + g] * f.u_at(g, a) * f.s[a];
                }
                newprev.data[r * q + a] = acc;
            }
        }
        *prev = newprev;
    }
    // Pass 2: for i = 1 to L-1, TruncatedSVD B[i] as M[(alpha p), beta].
    let mut max_disc = 0.0f64;
    for i in 0..lc - 1 {
        let (m, n, buf) = core_as_matrix(&cs[i], true);
        // Full spectrum first so the discarded weight is measurable.
        let full = svd_row_major(m, n, &buf);
        let total2: f64 = full.s.iter().map(|x| x * x).sum();
        let f = full.truncated(chi);
        let disc2: f64 = full.s.iter().skip(f.k).map(|x| x * x).sum();
        if total2 > 0.0 {
            max_disc = max_disc.max((disc2 / total2).sqrt());
        }
        let q = f.k;
        // B[i] <- U reshaped (alpha, p, q)
        let mut newc = Core::zeros(cs[i].dl, cs[i].dp, q);
        for a in 0..cs[i].dl {
            for p in 0..cs[i].dp {
                for t in 0..q {
                    newc.set(a, p, t, f.u_at(a * cs[i].dp + p, t));
                }
            }
        }
        // B[i+1]_{q p beta} <- sum_gamma s_q V^T[q gamma] B[i+1]_{gamma p beta}
        let next = &mut cs[i + 1];
        let rows = next.dl;
        let cols = next.dp * next.dr;
        let mut newnext = Core::zeros(q, next.dp, next.dr);
        for t in 0..q {
            let st = f.s[t];
            for r in 0..cols {
                let mut acc = 0.0;
                for g in 0..rows {
                    acc += f.vt_at(t, g) * next.data[g * cols + r];
                }
                newnext.data[t * cols + r] = st * acc;
            }
        }
        *next = newnext;
        cs[i] = newc;
    }
    (cs, max_disc)
}

/// Algorithm 4: boundary-MPS contraction of an open L x L square.
pub fn contract_open_square(l: usize, k: f64, chi: usize) -> BoundaryResult {
    assert!(l >= 2, "side length must be at least 2");
    assert!(chi >= 1, "chi must be at least 1");
    let t_start = Instant::now();

    let b = edge_vector_b(k);
    // Row tensors; the outward legs at the ends of each row are closed with b.
    let rows: Vec<Tensor4> = (0..l)
        .map(|i| row_tensor_t(k, i == 0, i == l - 1))
        .collect();

    // Line 1: B[i] <- reshape(b, (1, 2, 1)) for every i; top edge.
    let mut cores: Vec<Core> = (0..l)
        .map(|_| Core {
            dl: 1,
            dp: 2,
            dr: 1,
            data: vec![b[0], b[1]],
        })
        .collect();

    let mut ell = 0.0f64;
    let mut row_diag: Vec<RowDiag> = Vec::with_capacity(l);
    for y in 0..l {
        // Line 5: C[i]_{(alpha l) d (beta r)} <- sum_u B[i]_{alpha u beta} T[i]_{l u d r}
        let mut c_cores: Vec<Core> = Vec::with_capacity(l);
        for i in 0..l {
            let bc = &cores[i];
            let t = &rows[i];
            let dl = bc.dl * t.dims[0];
            let dr = bc.dr * t.dims[3];
            let dp = t.dims[2];
            let mut c = Core::zeros(dl, dp, dr);
            for a in 0..bc.dl {
                for u in 0..bc.dp {
                    for bb in 0..bc.dr {
                        let bval = bc.at(a, u, bb);
                        if bval == 0.0 {
                            continue;
                        }
                        for tl in 0..t.dims[0] {
                            for d in 0..t.dims[2] {
                                for tr in 0..t.dims[3] {
                                    let v = bval * t.at([tl, u, d, tr]);
                                    if v != 0.0 {
                                        let la = a * t.dims[0] + tl;
                                        let rb = bb * t.dims[3] + tr;
                                        let o = (la * dp + d) * dr + rb;
                                        c.data[o] += v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            c_cores.push(c);
        }
        // Line 6: B <- Compress(C, chi).
        let (mut newcores, disc) = compress(&c_cores, chi);
        // Lines 8-10: rescale the whole boundary through the last core only.
        let c = newcores[l - 1].fro_norm();
        for v in newcores[l - 1].data.iter_mut() {
            *v /= c;
        }
        ell += c.ln();
        row_diag.push(RowDiag {
            row: y + 1,
            scale: c,
            max_discarded: disc,
            bond_dims: (0..l - 1).map(|i| newcores[i + 1].dl).collect(),
        });
        cores = newcores;
    }

    // Lines 13-15: close the bottom edge with b.
    let mut v: Vec<f64> = vec![1.0];
    for core in &cores {
        let mut nv = vec![0.0; core.dr];
        for (bidx, &bv) in b.iter().enumerate() {
            if bv == 0.0 {
                continue;
            }
            for (ai, &av) in v.iter().enumerate() {
                if av == 0.0 {
                    continue;
                }
                for be in 0..core.dr {
                    nv[be] += av * bv * core.at(ai, bidx, be);
                }
            }
        }
        v = nv;
    }
    debug_assert_eq!(v.len(), 1, "the chain must close to a scalar");
    let z = v[0];
    let log_z = ell + z.ln();
    let g = log_z / (l * l) as f64;
    let elapsed_sec = t_start.elapsed().as_secs_f64();
    BoundaryResult {
        log_z,
        g,
        ell,
        row_diag,
        cores,
        closure_value: z,
        elapsed_sec,
    }
}
