//! Square-lattice Ising model tensors and reference enumerations.
//!
//! Zero field, coupling J = 1, Boltzmann constant 1. The Hamiltonian is
//! H(s) = -sum over edges of s_i*s_j with s_i in {+1, -1}. The Boltzmann
//! weight of one bond is F(s, t) = exp(K*s*t) with K = 1/T.

/// Bond split factors W from the sheet: F = W W^T with
/// W = [[sqrt(cosh K), sqrt(sinh K)], [sqrt(cosh K), -sqrt(sinh K)]],
/// rows indexed by the spin s = +1, -1 and columns by the factor index.
pub fn bond_split_w(k: f64) -> [[f64; 2]; 2] {
    let c = k.cosh().sqrt();
    let s = k.sinh().sqrt();
    [[c, s], [c, -s]]
}

/// Edge vector closing an absent bond at an open edge:
/// b = [1/sqrt(cosh K), 0]^T, so that W b = (1, 1)^T.
pub fn edge_vector_b(k: f64) -> [f64; 2] {
    [1.0 / k.cosh().sqrt(), 0.0]
}

/// A dense tensor with up to four legs, stored row-major in `dims` order.
#[derive(Debug, Clone)]
pub struct Tensor4 {
    pub dims: [usize; 4],
    pub data: Vec<f64>,
}

impl Tensor4 {
    pub fn zeros(dims: [usize; 4]) -> Self {
        let len: usize = dims.iter().product();
        Tensor4 {
            dims,
            data: vec![0.0; len],
        }
    }

    #[inline]
    pub fn idx(dims: &[usize; 4], i: [usize; 4]) -> usize {
        ((i[0] * dims[1] + i[1]) * dims[2] + i[2]) * dims[3] + i[3]
    }

    #[inline]
    pub fn at(&self, i: [usize; 4]) -> f64 {
        self.data[Self::idx(&self.dims, i)]
    }

    #[inline]
    pub fn set(&mut self, i: [usize; 4], v: f64) {
        let o = Self::idx(&self.dims, i);
        self.data[o] = v;
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0f64, |a, &x| a.max(x.abs()))
    }
}

/// Site tensor A[r, u, l, d] = sum_s W[s,r] W[s,u] W[s,l] W[s,d].
///
/// The legs are bond-factor indices; contracting the facing legs of two
/// neighbouring sites restores the bond weight F(s, t).
pub fn site_tensor_a(k: f64) -> Tensor4 {
    let w = bond_split_w(k);
    let mut a = Tensor4::zeros([2, 2, 2, 2]);
    for r in 0..2 {
        for u in 0..2 {
            for l in 0..2 {
                for d in 0..2 {
                    let v = (0..2)
                        .map(|s| w[s][r] * w[s][u] * w[s][l] * w[s][d])
                        .sum::<f64>();
                    a.set([r, u, l, d], v);
                }
            }
        }
    }
    a
}

/// Row tensor T[i] with legs (l, u, d, r), each of dimension two.
///
/// `left_closed`/`right_closed` contract the outward horizontal legs with the
/// edge vector b (open boundary), leaving axes of size one.
pub fn row_tensor_t(k: f64, left_closed: bool, right_closed: bool) -> Tensor4 {
    let a = site_tensor_a(k);
    let b = edge_vector_b(k);
    let mut dims = [2usize; 4];
    if left_closed {
        dims[0] = 1;
    }
    if right_closed {
        dims[3] = 1;
    }
    let mut t = Tensor4::zeros(dims);
    // T[l,u,d,r] = sum_{lp, rp} cl(l,lp) * A[rp,u,lp,d] * cr(r,rp) with
    // cl = b[lp] when the left leg is closed, else delta_{l,lp};
    // cr = b[rp] when the right leg is closed, else delta_{r,rp}.
    for u in 0..2usize {
        for d in 0..2usize {
            for r in 0..dims[3] {
                for l in 0..dims[0] {
                    let mut v = 0.0;
                    for lp in 0..2usize {
                        let cl = if left_closed {
                            b[lp]
                        } else if lp == l {
                            1.0
                        } else {
                            0.0
                        };
                        for rp in 0..2usize {
                            let cr = if right_closed {
                                b[rp]
                            } else if rp == r {
                                1.0
                            } else {
                                0.0
                            };
                            v += cl * cr * a.at([rp, u, lp, d]);
                        }
                    }
                    t.set([l, u, d, r], v);
                }
            }
        }
    }
    t
}

/// Enumeration of the full partition function and energy statistics.
///
/// One bond is counted per site in each positive direction (right and down),
/// with wrapping for periodic geometry. At L = 2 the two parallel bonds
/// between a pair of sites are counted separately, as required.
pub struct EnumResult {
    pub z: f64,
    pub mean_h: f64,
    pub var_h: f64,
}

pub fn enumerate(l: usize, k: f64, periodic: bool) -> EnumResult {
    assert!(l >= 1);
    let n = l * l;
    // Bond list: (site a, site b) by flat index.
    let site = |x: usize, y: usize| y * l + x;
    let mut bonds: Vec<(usize, usize)> = Vec::new();
    for y in 0..l {
        for x in 0..l {
            if periodic || x + 1 < l {
                bonds.push((site(x, y), site((x + 1) % l, y)));
            }
            if periodic || y + 1 < l {
                bonds.push((site(x, y), site(x, (y + 1) % l)));
            }
        }
    }
    let mut z = 0.0;
    let mut h_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut spins = vec![0i64; n];
    for cfg in 0..(1u64 << n) {
        for (i, s) in spins.iter_mut().enumerate() {
            *s = if (cfg >> i) & 1 == 0 { 1 } else { -1 };
        }
        let mut sum_ss = 0i64;
        for &(a, b) in &bonds {
            sum_ss += spins[a] * spins[b];
        }
        let h = -(sum_ss as f64);
        let w = (k * sum_ss as f64).exp();
        z += w;
        h_sum += h * w;
        h2_sum += h * h * w;
    }
    let mean_h = h_sum / z;
    EnumResult {
        z,
        mean_h,
        var_h: h2_sum / z - mean_h * mean_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_split_reproduces_bond_weight() {
        for k in [0.0f64, 0.44, 0.7, 1.3] {
            let w = bond_split_w(k);
            for s in 0..2 {
                for t in 0..2 {
                    let sv = if s == 0 { 1.0 } else { -1.0 };
                    let tv = if t == 0 { 1.0 } else { -1.0 };
                    let f = (k * sv * tv).exp();
                    let wwt: f64 = (0..2).map(|a| w[s][a] * w[t][a]).sum();
                    assert!(
                        (f - wwt).abs() < 1e-14,
                        "K={k}: F={f} vs WW^T={wwt}"
                    );
                }
            }
        }
    }

    #[test]
    fn open_edge_vector_removes_bond() {
        for k in [0.44f64, 0.7] {
            let w = bond_split_w(k);
            let b = edge_vector_b(k);
            for s in 0..2 {
                let v: f64 = (0..2).map(|a| w[s][a] * b[a]).sum();
                assert!((v - 1.0).abs() < 1e-14, "W b = {v}, expected 1");
            }
        }
    }

    #[test]
    fn open_l1_partition_function() {
        // A single site: Z = 2 at any K (no bonds).
        let r = enumerate(1, 0.44, false);
        assert!((r.z - 2.0).abs() < 1e-14);
        assert!(r.var_h.abs() < 1e-14);
    }

    #[test]
    fn periodic_l2_counts_double_bonds() {
        // 2x2 periodic torus: 8 bonds. At K=0, Z = 16.
        let r0 = enumerate(2, 0.0, true);
        assert!((r0.z - 16.0).abs() < 1e-12);
        // Fully aligned configuration: H = -8, weight e^{8K}.
        let r = enumerate(2, 0.44, true);
        let e = (8.0 * 0.44f64).exp();
        assert!(r.z > e, "Z must exceed the aligned weight");
    }

    #[test]
    fn open_l2_two_bonds() {
        // 2x2 open square: 4 bonds. Z(K) = sum over 16 configs; check K=0.
        let r = enumerate(2, 0.0, false);
        assert!((r.z - 16.0).abs() < 1e-12);
        // and the two aligned configs contribute 2 e^{4K} at K=0.44
        let r = enumerate(2, 0.44, false);
        let e = (4.0 * 0.44f64).exp();
        assert!(r.z > 2.0 * e);
    }
}
