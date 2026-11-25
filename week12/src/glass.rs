//! Spin glasses and the SplitMix64 instance generator from the design file.

#[derive(Clone, Debug)]
pub struct SpinGlass {
    pub n: usize,
    pub edges: Vec<(usize, usize)>,
    pub couplings: Vec<f64>,
    pub fields: Vec<f64>,
}

impl SpinGlass {
    pub fn m(&self) -> usize {
        self.edges.len()
    }

    /// Constant offset C = sum J + sum h. HiGHS minimizes the linear part only.
    pub fn constant(&self) -> f64 {
        self.couplings.iter().sum::<f64>() + self.fields.iter().sum::<f64>()
    }

    pub fn energy_spins(&self, spins: &[f64]) -> f64 {
        let mut energy = 0.0;
        for (e, &(u, v)) in self.edges.iter().enumerate() {
            energy += self.couplings[e] * spins[u] * spins[v];
        }
        for i in 0..self.n {
            energy += self.fields[i] * spins[i];
        }
        energy
    }

    /// Nearest rounding: x_i >= 0.5 becomes bit 1, and s_i = 1 - 2 x_i.
    pub fn rounded_energy(&self, x: &[f64]) -> f64 {
        let spins: Vec<f64> = x
            .iter()
            .map(|&xi| if xi >= 0.5 { -1.0 } else { 1.0 })
            .collect();
        self.energy_spins(&spins)
    }

    pub fn bits_to_spins(x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&xi| if xi >= 0.5 { -1.0 } else { 1.0 })
            .collect()
    }
}

pub struct GraphInfo {
    /// Neighbor and edge index, neighbors in increasing index order.
    pub adj: Vec<Vec<(usize, usize)>>,
    pub degree: Vec<usize>,
    pub d_max: usize,
}

impl GraphInfo {
    pub fn new(glass: &SpinGlass) -> Self {
        let mut adj = vec![Vec::new(); glass.n];
        for (e, &(u, v)) in glass.edges.iter().enumerate() {
            adj[u].push((v, e));
            adj[v].push((u, e));
        }
        for neighbors in &mut adj {
            neighbors.sort_by_key(|(j, _)| *j);
        }
        let degree: Vec<usize> = adj.iter().map(|n| n.len()).collect();
        let d_max = degree.iter().copied().max().unwrap_or(0);
        Self { adj, degree, d_max }
    }
}

#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn index(&mut self, k: usize) -> usize {
        (self.next() % k as u64) as usize
    }
}

pub fn instance_rng(seed: u64, id: u64) -> SplitMix64 {
    // The design file sets the instance state to the first `next` return value
    // of a generator whose state is `seed XOR (id * constant)`. That choice
    // reproduces the published 16-instance energies.
    let mut generator = SplitMix64 {
        state: seed ^ id.wrapping_mul(0xD1B5_4A32_D192_ED03),
    };
    SplitMix64 {
        state: generator.next(),
    }
}

pub fn random_instance(n: usize, m: usize, seed: u64, id: u64) -> SpinGlass {
    assert!(n >= 2, "need at least two spins");
    assert!(m <= n * (n - 1) / 2, "too many bonds for the complete graph");
    let mut rng = instance_rng(seed, id);
    let mut present = vec![vec![false; n]; n];
    let mut degree = vec![0usize; n];
    let mut edges = Vec::with_capacity(m);
    let min_degree = 2.min(n - 1);

    for i in 0..n {
        while degree[i] < min_degree {
            let j = rng.index(n);
            if j != i {
                let (u, v) = if i < j { (i, j) } else { (j, i) };
                if !present[u][v] {
                    present[u][v] = true;
                    edges.push((u, v));
                    degree[u] += 1;
                    degree[v] += 1;
                }
            }
        }
    }
    while edges.len() < m {
        let i = rng.index(n);
        let j = rng.index(n);
        if i != j {
            let (u, v) = if i < j { (i, j) } else { (j, i) };
            if !present[u][v] {
                present[u][v] = true;
                edges.push((u, v));
                degree[u] += 1;
                degree[v] += 1;
            }
        }
    }
    edges.sort_unstable();
    let couplings = (0..edges.len())
        .map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    let fields = (0..n)
        .map(|_| (rng.next() % 3) as f64 - 1.0)
        .collect();
    SpinGlass {
        n,
        edges,
        couplings,
        fields,
    }
}

pub fn square() -> SpinGlass {
    SpinGlass {
        n: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
        couplings: vec![1.0, 1.0, 1.0, -1.0],
        fields: vec![0.0; 4],
    }
}

pub fn tetrahedron() -> SpinGlass {
    let mut edges = Vec::new();
    for u in 0..4 {
        for v in (u + 1)..4 {
            edges.push((u, v));
        }
    }
    SpinGlass {
        n: 4,
        edges,
        couplings: vec![1.0; 6],
        fields: vec![1.0, 0.0, 0.0, 0.0],
    }
}
