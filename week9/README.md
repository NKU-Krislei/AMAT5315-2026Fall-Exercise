# Week 9: Probabilistic inference by tensor contraction

Exact inference, contraction orders, belief propagation, and the Promedus_14 benchmark are in this directory.

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

## Contraction order

The scalar AD query keeps the factor tables intact and attaches one-hot caps for `A=0` and `D=0`. Random pairings use a seeded generator. Greedy pairings use omeco `GreedyMethod`. Costs are recounted from the saved tree: a step's work is the product of the dimensions in the union of its two inputs, and the largest tensor is the biggest input or intermediate.

```bash
cargo +1.96.0 run --quiet -- --model asia.uai --evidence AD --order random --seed 42 --out runs/random
cargo +1.96.0 run --quiet -- --model asia.uai --evidence AD --order greedy --out runs/greedy
cargo +1.96.0 run --quiet -- --model asia.uai --evidence AD --analyze runs/random/order.json --out runs/analysis
cargo +1.96.0 run --quiet -- --model asia.uai --evidence AD --plan runs/greedy/order.json --out runs/replay
conda run -n modernSC python plot_orders.py
```

Both orders give `p(A=0, D=0) = 0.004501375`, within relative error `1e-10` of enumeration. Replaying the greedy plan reproduces that mass. Analysis recounts the random plan without contracting it. This seed-42 random tree has work 248 and a largest tensor of 32 entries, with labels E, L, A, B, S. The greedy tree has work 54 and a largest tensor of 8 entries, which is the input `p(E|T,L)`; its largest intermediate has 4 entries, labels E, L. `evidence/orders.png` draws both trees and marks those intermediates. Another random generator can produce a different tree; the saved plans are the ones being counted.

## Loopy belief propagation

Observations are written into the factor tables before the updates. Messages start uniform. Each sweep reads the current messages and writes a normalized proposal for every directed edge; the residual is the largest absolute change in that undamped proposal, including the returned state. The stored messages are then damped by 0.5. Beliefs multiply every incoming factor-to-variable message and omit none of them. Update work counts variable-to-factor products and one visit per factor entry in each factor-to-variable message.

```bash
cargo +1.96.0 run --quiet -- --model asia.uai --evidence prior --method bp \
  --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-prior
cargo +1.96.0 run --quiet -- --model asia.uai --evidence AD --method bp \
  --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-AD
cargo +1.96.0 run --quiet -- --model asia.uai --evidence ADX --method bp \
  --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/bp-ADX
conda run -n modernSC python plot_bp.py
```

These directories do not replace `artifacts/prior`, `artifacts/AD`, or `artifacts/ADX`. ADX stops at sweep 94 with undamped residual `8.225e-11` and maximum marginal error `0.010612`. The messages have stopped; the disease probabilities have not matched enumeration. `evidence/bp.png` shows that gap for prior, AD, and ADX. A chain factor graph, checked in `tests/bp.rs`, matches enumeration, so the ADX gap is the loop rather than a message that includes its recipient.

## Promedus_14

The UAI 2014 files are mirrored from `mroavi/uai-2014-inference-competition` revision `22005844423f5bca62798548df808cc418fc0d7d` into `promedus/`. The network has 414 binary variables and 414 factors. The evidence file fixes nine variables, all in state 1: 41, 47, 26, 29, 28, 14, 123, 4, and 147. Tables are not edited. Conditioning removes an observed axis and keeps the resulting scalar factors. Each intermediate is divided by its maximum entry, and the removed factor is added back in the natural log. The queries are `log10` of the evidence partition and the marginals of variables 59, 102, and 139.

Greedy search is omeco `GreedyMethod` with alpha 0 and temperature 0. TreeSA is omeco `TreeSA::fast` scored by time only: 1 trial, 20 iterations, 100 inverse temperatures from 0.16 to 15.01. The trial seed is `trial_idx + 42` inside omeco and is not a command-line argument. Costs below are recounted from the saved trees.

```bash
cargo +1.96.0 run --release --quiet -- --model promedus/Promedus_14.uai \
  --evidence-file promedus/Promedus_14.uai.evid --queries 59,102,139 \
  --order greedy --pr promedus/Promedus_14.uai.PR --mar promedus/Promedus_14.uai.MAR \
  --out runs/promedus-greedy
cargo +1.96.0 run --release --quiet -- --model promedus/Promedus_14.uai \
  --evidence-file promedus/Promedus_14.uai.evid --queries 59,102,139 \
  --order treesa --pr promedus/Promedus_14.uai.PR --mar promedus/Promedus_14.uai.MAR \
  --out runs/promedus-treesa
cargo +1.96.0 run --release --quiet -- --model promedus/Promedus_14.uai \
  --evidence-file promedus/Promedus_14.uai.evid --queries 59,102,139 \
  --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 \
  --pr promedus/Promedus_14.uai.PR --mar promedus/Promedus_14.uai.MAR \
  --out runs/promedus-bp
conda run -n modernSC python plot_promedus.py
```

Greedy returns `log10 Z = -7.806746304582435` and TreeSA returns `-7.806746304582434`. The two differ by about `2e-15`. The published PR file rounds this to `-7.80675`, an absolute gap of `3.7e-6`. The three marginals match the MAR file within `5e-7`. State 1 is 0.1975882582, 0.1497534323, and 0.6175220184. Greedy work is 1,098,677,865 with a largest intermediate of 4,194,304 entries. TreeSA work is 10,523,488 with a largest intermediate of 131,072 entries: about 104 times less work and 32 times less memory. The reference greedy tree has less work, 667,111,481, at the same peak size. This TreeSA peak is the reference 131,072. Multiplying one ASIA table by 10 increases `log10 Z` by 1 and leaves the normalized marginal fixed; the same rescaling recovers a two-factor partition whose raw product underflows to zero.

Belief propagation uses the ASIA settings. After 1000 sweeps it is still unconverged: residual `2.171e-10`, cumulative update work 9,704,000, elapsed 0.235 seconds. The state-1 beliefs are about `8.56e-5`, `2.51e-4`, and `0.999875`. The largest absolute error against the MAR file is `0.382353`, on variable 139. `evidence/promedus.png` puts the order costs beside that trace. The exact runs contract one scalar and three open variables; BP updates every message, so the two work columns are different jobs.


