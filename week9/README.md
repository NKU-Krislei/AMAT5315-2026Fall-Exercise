# Week 9: Probabilistic inference by tensor contraction

Exact inference for the ASIA network is in this directory. Contraction orders, belief propagation, and the Promedus benchmark are added in later builds.

Rust 1.96.0, tenferro 0.9.0 (`tenferro-tensor`, `tenferro-cpu`, `tenferro-einsum`, native CPU, no autodiff or GPU), and 64-bit arithmetic. Plots use the conda environment `modernSC`.

```bash
cargo +1.96.0 test
cargo +1.96.0 install --locked --path .
```

The ASIA tables are the pinned TensorInference example `935ed86f8e473f523cac9461c45a1f470e74f0f9`, saved as `asia.uai`. Variable order is A, T, S, L, B, E, X, D. State 0 means yes. The last variable in a scope changes fastest, and the tables are not altered.

## Exact inference

`prior` has no observations. `AD` fixes a visit to Asia and dyspnoea (`A=0`, `D=0`). `ADX` also fixes a positive X-ray (`X=0`).

```bash
cargo +1.96.0 run --quiet -- checks --out evidence/checks.txt
cargo +1.96.0 run --quiet -- diagram --model asia.uai --out evidence/graphs.svg
cargo +1.96.0 run --quiet -- --model asia.uai --evidence prior --out artifacts/prior
cargo +1.96.0 run --quiet -- --model asia.uai --evidence AD --out artifacts/AD
cargo +1.96.0 run --quiet -- --model asia.uai --evidence ADX --out artifacts/ADX
conda run -n modernSC python plot_asia.py
```

Enumeration and contraction agree within absolute marginal error `1e-10` and relative evidence-mass error `1e-10`. Evidence masses are 1, 0.004501375, and 0.00098822675. The positive X-ray raises tuberculosis and lung cancer and lowers bronchitis. In ADX, `p(T=0, L=0 | evidence)` is about 0.0222 while the product of those marginals is about 0.1740; `evidence/joint.png` keeps that gap.

`evidence/checks.txt` records `AB` in column-major order `[4, 10, 4, 8]`, `tr(AB) = 12`, and the three-spin Ising ring `Z = 12.60256209895193` at `beta = 0.5`, `J = 1`, spins `±1`, no external field. The contraction and the eight-configuration sum differ by less than `1e-12`.

`evidence/graphs.svg` is the AD tuberculosis query. Each factor is labeled with its UAI axes. Smoking is one shared index of `p(S)`, `p(L|S)`, and `p(B|S)`. `A` and `D` are fixed at 0, `X` is summed, and only `T` stays open.
